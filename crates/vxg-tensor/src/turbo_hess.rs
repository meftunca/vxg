//! Full-width (`SB_COLS = 4096`) blocked `32`-column reverse `LDL^T` Hessian with descending
//! `32`-block activation-energy `act_order` and sample-adaptive PSD precision shrinkage.

use crate::turbo_linalg::{rotate_rows_fwht32, sgemm_nt};
use crate::wht2::fwht32_inplace;
use crate::wht_turbo_gptq::HadamardBlockChol;

const SB_COLS: usize = 4096;
const SQRT32: f64 = 5.656_854_249_492_381;
const OFF_SHRINK_C: f64 = 0.85;
const DIAG_DAMP_BASE: f64 = 0.055;

pub(crate) struct SuperBlockBlock {
    pub(crate) c0: usize,
    pub(crate) bw: usize,
    /// Permutation of `0..bw/32`: `blk_ord[sbi]` is the original `32`-block quantized at step `sbi`.
    pub(crate) blk_ord: Vec<usize>,
    /// Diagonal of regularized permuted Hessian (`[bw]`).
    pub(crate) h_diag: Vec<f32>,
    /// Schur-complement pivots of permuted reverse `LDL^T` (`[bw]`).
    pub(crate) alpha: Vec<f32>,
    /// Row-major `[bw, bw]` unit lower-triangular `M` (`H_perm = M^T diag(alpha) M`).
    pub(crate) m: Vec<f32>,
    /// Optional `[bw, bw]` drift operator `B = alpha_eff * H_drift^{-1} C_b^T` (unpermuted).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) b_drift: Option<Vec<f32>>,
}

pub(crate) struct SuperBlockHessian {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) x_stu_rot: Vec<f32>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) ch_amp: Vec<f32>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) tok_w: Vec<f32>,
    pub(crate) blocks: Vec<SuperBlockBlock>,
}

fn solve_cholesky_32(h32: &[f64; 1024], rhs_mat: &[f32; 1024], scale: f64, vlen: usize) -> [f32; 1024] {
    let mut l = [0.0f64; 1024];
    for i in 0..32 {
        let mut diag = h32[i * 32 + i];
        for p in 0..i {
            diag -= l[i * 32 + p] * l[i * 32 + p];
        }
        let lii = diag.max(1e-12).sqrt();
        l[i * 32 + i] = lii;
        let inv = 1.0 / lii;
        for r in (i + 1)..32 {
            let mut sum = h32[r * 32 + i];
            for p in 0..i {
                sum -= l[r * 32 + p] * l[i * 32 + p];
            }
            l[r * 32 + i] = sum * inv;
        }
    }
    let (mut b32, mut tmp) = ([0.0f32; 1024], [0.0f64; 32]);
    for col in 0..vlen {
        for i in 0..32 {
            let mut sum = scale * (rhs_mat[i * 32 + col] as f64);
            for p in 0..i {
                sum -= l[i * 32 + p] * tmp[p];
            }
            tmp[i] = sum / l[i * 32 + i];
        }
        for i in (0..32).rev() {
            let mut sum = tmp[i];
            for r in (i + 1)..32 {
                sum -= l[r * 32 + i] * tmp[r];
            }
            tmp[i] = sum / l[i * 32 + i];
            b32[i * 32 + col] = tmp[i] as f32;
        }
    }
    for j in 0..vlen {
        let row = &mut b32[j * 32..j * 32 + vlen];
        let norm = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        if norm > 0.18 {
            let sc = (0.18 / norm) as f32;
            for v in row {
                *v *= sc;
            }
        }
    }
    b32
}

#[allow(clippy::too_many_arguments)]
fn solve_drift_block32(
    c0: usize, bw: usize, cols: usize, n_tok: usize, eff_n: f64, inv_md: f64,
    padded_cols: usize, xb_t: &[f32], dxr: &[f32], tok_w: &[f32],
    h_emp: &[f32], prior32: &[f64], blk_rel: &[f64],
) -> Option<Vec<f32>> {
    let mut b_out = vec![0.0f32; bw * bw];
    let mut any_nonzero = false;
    let scale_n = (1.0 / eff_n) as f32;
    let mut dxb_32 = vec![0.0f32; 32 * n_tok];
    let mut c_bt_32 = [0.0f32; 1024];

    for sb32 in 0..(bw / 32) {
        let (b0, gc0) = (sb32 * 32, c0 + sb32 * 32);
        if gc0 >= cols {
            continue;
        }
        let vlen = (cols - gc0).min(32);
        let xb_32 = &xb_t[b0 * n_tok..(b0 + 32) * n_tok];
        dxb_32.fill(0.0);
        for t in 0..n_tok {
            let r_dx = &dxr[t * padded_cols + gc0..t * padded_cols + gc0 + vlen];
            let tw = tok_w[t];
            for j in 0..vlen {
                dxb_32[j * n_tok + t] = r_dx[j] * tw;
            }
        }
        let (mut xs_n2, mut dx_n2, mut tok_prod) = (0.0f64, 0.0f64, 0.0f64);
        for t in 0..n_tok {
            let (mut xs_t, mut dx_t) = (0.0f64, 0.0f64);
            for j in 0..32 {
                let (sv, dv) = (xb_32[j * n_tok + t] as f64, dxb_32[j * n_tok + t] as f64);
                xs_t += sv * sv;
                dx_t += dv * dv;
            }
            xs_n2 += xs_t;
            dx_n2 += dx_t;
            tok_prod += xs_t * dx_t;
        }
        if dx_n2 <= 1e-16 * xs_n2 || xs_n2 <= 1e-30 {
            continue;
        }
        c_bt_32.fill(0.0);
        sgemm_nt(32, 32, n_tok, scale_n, xb_32, n_tok, &dxb_32, n_tok, 0.0, &mut c_bt_32, 32, 1);
        let c32_f2_raw: f64 = c_bt_32.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let js = (1.0 - 2.5 * (tok_prod / (eff_n * eff_n)) / c32_f2_raw.max(1e-30)).max(0.0);
        if js <= 1e-4 {
            continue;
        }
        let mut h32 = [0.0f64; 1024];
        let p_sl = &prior32[sb32 * 1024..(sb32 + 1) * 1024];
        for u in 0..32 {
            for v in 0..32 {
                h32[u * 32 + v] = if u >= vlen || v >= vlen {
                    if u == v { 1.0 } else { 0.0 }
                } else {
                    0.80 * (h_emp[(b0 + u) * bw + (b0 + v)] as f64) * inv_md
                        + 0.20 * p_sl[u * 32 + v]
                        + if u == v { 0.20 * blk_rel[sb32] } else { 0.0 }
                };
            }
        }
        let alpha_eff = 0.55 * js * (1.0 / (1.0 + (dx_n2 / xs_n2).sqrt())).clamp(0.25, 1.0);
        let b32 = solve_cholesky_32(&h32, &c_bt_32, alpha_eff * inv_md, vlen);
        for j in 0..vlen {
            for col in 0..vlen {
                b_out[(b0 + j) * bw + (b0 + col)] = b32[j * 32 + col];
            }
        }
        any_nonzero = true;
    }
    any_nonzero.then_some(b_out)
}

/// Blocked `32`-column reverse `LDL^T`: `S = M^T diag(alpha) M`, using `sgemm_nt` for rank-32 Schur updates.
fn blocked_reverse_ldl(s: &mut [f32], bw: usize, threads: usize) -> (Vec<f32>, Vec<f32>) {
    let mut alpha = vec![1.0f32; bw];
    let mut m = vec![0.0f32; bw * bw];
    let mut a_pan = vec![0.0f32; bw * 32];
    let mut b_pan = vec![0.0f32; bw * 32];

    let mut k0 = bw;
    while k0 >= 32 {
        k0 -= 32;
        for k in (k0..k0 + 32).rev() {
            let ak = (s[k * bw + k] as f64).max(1e-6);
            alpha[k] = ak as f32;
            m[k * bw + k] = 1.0;
            let inv_ak = 1.0 / ak;
            for j in 0..k {
                m[k * bw + j] = ((s[k * bw + j] as f64) * inv_ak) as f32;
            }
            for u in k0..k {
                let a_mu = ak * (m[k * bw + u] as f64);
                let row_u = &mut s[u * bw..u * bw + u + 1];
                let row_m = &m[k * bw..k * bw + u + 1];
                for v in 0..=u {
                    row_u[v] = ((row_u[v] as f64) - a_mu * (row_m[v] as f64)) as f32;
                }
            }
        }
        if k0 > 0 {
            for u in 0..k0 {
                let a_row = &mut a_pan[u * 32..(u + 1) * 32];
                let b_row = &mut b_pan[u * 32..(u + 1) * 32];
                for p in 0..32 {
                    let mv = m[(k0 + p) * bw + u];
                    a_row[p] = mv;
                    b_row[p] = alpha[k0 + p] * mv;
                }
            }
            sgemm_nt(
                k0, k0, 32, -1.0, &a_pan[..k0 * 32], 32, &b_pan[..k0 * 32], 32, 1.0, s, bw, threads,
            );
        }
    }
    (alpha, m)
}

#[allow(clippy::too_many_arguments)]
fn build_superblock(
    c0: usize, bw: usize, n_tok: usize, eff_n: f64, cols: usize, padded_cols: usize,
    x_stu_rot: &[f32], dx_rot: Option<&[f32]>, ch_amp: &[f32], tok_w: &[f32], threads: usize,
) -> SuperBlockBlock {
    let mut xb_t = vec![0.0f32; bw * n_tok];
    let w_tr = threads.clamp(1, 32).min(bw.div_ceil(64).max(1));
    if w_tr > 1 && bw * n_tok >= 262_144 {
        let jpw = bw.div_ceil(w_tr);
        std::thread::scope(|sc| {
            for (ci, dst_blk) in xb_t.chunks_mut(jpw * n_tok).enumerate() {
                let j0 = ci * jpw;
                let nj = dst_blk.len() / n_tok;
                sc.spawn(move || {
                    for t in 0..n_tok {
                        let row = &x_stu_rot[t * padded_cols + c0 + j0..t * padded_cols + c0 + j0 + nj];
                        let tw = tok_w[t];
                        for dj in 0..nj { dst_blk[dj * n_tok + t] = row[dj] * tw; }
                    }
                });
            }
        });
    } else {
        for t in 0..n_tok {
            let row = &x_stu_rot[t * padded_cols + c0..t * padded_cols + c0 + bw];
            let tw = tok_w[t];
            for j in 0..bw { xb_t[j * n_tok + t] = row[j] * tw; }
        }
    }
    let mut h_emp = vec![0.0f32; bw * bw];
    let scale_n = (1.0 / eff_n) as f32;
    sgemm_nt(bw, bw, n_tok, scale_n, &xb_t, n_tok, &xb_t, n_tok, 0.0, &mut h_emp, bw, threads);

    let (mut d_sum, mut d_cnt) = (0.0f64, 0usize);
    for k in 0..bw {
        if c0 + k < cols {
            d_sum += (h_emp[k * bw + k] as f64).max(0.0);
            d_cnt += 1;
        }
    }
    let mean_d = (if d_cnt > 0 { d_sum / d_cnt as f64 } else { 1.0 }).max(1e-30);
    let inv_md = 1.0 / mean_d;

    let nb = bw / 32;
    let mut prior32 = vec![0.0f64; nb * 1024];
    let mut blk_rel = vec![1.0f64; nb];
    let mut blk_trace = vec![-1.0f64; nb];
    for sb32 in 0..nb {
        let gc0 = c0 + sb32 * 32;
        let amps = if gc0 < cols { &ch_amp[gc0..cols.min(gc0 + 32)] } else { &[] };
        let chol = HadamardBlockChol::from_amps(amps);
        let rel_blk = if gc0 < cols { ((chol.mean_d as f64) * inv_md).clamp(0.15, 6.0) } else { 1.0 };
        blk_rel[sb32] = rel_blk;
        let p_scale = (rel_blk / (chol.mean_d as f64).max(1e-30)) as f32;
        let mut spec = chol.d_unrot;
        for v in &mut spec {
            *v *= p_scale;
        }
        fwht32_inplace(&mut spec);
        let pslice = &mut prior32[sb32 * 1024..(sb32 + 1) * 1024];
        for u in 0..32 {
            for v in 0..32 {
                pslice[u * 32 + v] = (spec[u ^ v] as f64) / SQRT32;
            }
        }
        if gc0 < cols {
            let vlen = (cols - gc0).min(32);
            let mut tr = 0.0f64;
            for j in 0..vlen {
                let k = sb32 * 32 + j;
                tr += (h_emp[k * bw + k] as f64).max(0.0);
            }
            blk_trace[sb32] = tr * inv_md;
        }
    }

    let mut blk_ord: Vec<usize> = (0..nb).collect();
    blk_ord.sort_by(|&a, &b| blk_trace[b].total_cmp(&blk_trace[a]).then_with(|| a.cmp(&b)));

    // Convex PSD shrinkage: guaranteed lambda_min(S) >= diag_damp for any N/bw ratio.
    let lam_cross = (eff_n / (eff_n + OFF_SHRINK_C * (bw as f64))).clamp(0.35, 0.88);
    let rem = 1.0 - lam_cross;
    let lam_in = 0.45 * rem;
    let lam_prior = 0.35 * rem;
    let lam_diag = 0.20 * rem;
    let diag_damp = (DIAG_DAMP_BASE * (1.0 + 0.55 * (bw as f64) / eff_n)).clamp(0.03, 0.16);

    let mut s = vec![0.0f32; bw * bw];
    let w_s = threads.clamp(1, 32).min(nb.max(1));
    let bpw_s = nb.div_ceil(w_s);
    std::thread::scope(|sc| {
        let (bo_ref, he_ref, pr_ref) = (&blk_ord, &h_emp, &prior32);
        for (ci, s_blk) in s.chunks_mut(bpw_s * 32 * bw).enumerate() {
            let bu0 = ci * bpw_s;
            let nbu = s_blk.len() / (32 * bw);
            sc.spawn(move || {
                for dbu in 0..nbu {
                    let bu = bu0 + dbu;
                    let ou = bo_ref[bu];
                    let ou0 = ou * 32;
                    let p_sl = &pr_ref[ou * 1024..(ou + 1) * 1024];
                    for bv in 0..nb {
                        let ov = bo_ref[bv];
                        let v0 = bv * 32;
                        let ov0 = ov * 32;
                        let same = bu == bv;
                        for du in 0..32 {
                            let gu = c0 + ou0 + du;
                            let valid_u = gu < cols;
                            let dst_row = &mut s_blk[(dbu * 32 + du) * bw + v0..(dbu * 32 + du) * bw + v0 + 32];
                            let h_row = &he_ref[(ou0 + du) * bw + ov0..(ou0 + du) * bw + ov0 + 32];
                            for dv in 0..32 {
                                let gv = c0 + ov0 + dv;
                                let val = if !valid_u || gv >= cols {
                                    if same && du == dv { 1.0 } else { 0.0 }
                                } else {
                                    let hn = (h_row[dv] as f64) * inv_md;
                                    if same {
                                        let mut v = (lam_cross + lam_in) * hn + lam_prior * p_sl[du * 32 + dv];
                                        if du == dv { v += lam_diag * hn + diag_damp; }
                                        v
                                    } else {
                                        lam_cross * hn
                                    }
                                };
                                dst_row[dv] = val as f32;
                            }
                        }
                    }
                }
            });
        }
    });
    for k in 0..bw {
        if s[k * bw + k] < 1e-6 {
            s[k * bw + k] = 1e-6;
        }
    }
    let h_diag: Vec<f32> = (0..bw).map(|k| s[k * bw + k]).collect();
    let b_drift = dx_rot.and_then(|dxr| {
        solve_drift_block32(
            c0, bw, cols, n_tok, eff_n, inv_md, padded_cols, &xb_t, dxr, tok_w, &h_emp, &prior32, &blk_rel,
        )
    });
    drop(h_emp);
    drop(xb_t);

    let (alpha, m) = blocked_reverse_ldl(&mut s, bw, threads);
    SuperBlockBlock { c0, bw, blk_ord, h_diag, alpha, m, b_drift }
}

impl SuperBlockHessian {
    pub(crate) fn build(
        x_stu: &[f32], x_ref: Option<&[f32]>, n_tok: usize, cols: usize, padded_cols: usize, threads: usize,
    ) -> Self {
        let inv_c = 1.0 / (cols.max(1) as f64);
        let rms: Vec<f64> = (0..n_tok)
            .map(|t| {
                let s2: f64 = x_stu[t * cols..(t + 1) * cols].iter().map(|&v| (v as f64) * (v as f64)).sum();
                (s2 * inv_c).sqrt()
            })
            .collect();
        let mut sorted = rms.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let med = sorted.get(n_tok / 2).copied().unwrap_or(1.0).max(1e-12);
        let cap = 4.0 * med;
        let tok_w: Vec<f32> = rms.iter().map(|&r| if r > cap { (cap / r) as f32 } else { 1.0 }).collect();
        let eff_n: f64 = tok_w.iter().map(|&w| (w as f64) * (w as f64)).sum::<f64>().max(1.0);

        let mut sums = vec![0.0f64; cols];
        for (t, row) in x_stu.chunks_exact(cols).enumerate() {
            let tw = tok_w[t] as f64;
            let tw2 = tw * tw;
            for c in 0..cols {
                let v = row[c] as f64;
                sums[c] += tw2 * v * v;
            }
        }
        let ch_amp: Vec<f32> = (0..cols).map(|c| ((sums[c] / eff_n).sqrt() as f32).max(1e-4)).collect();

        let x_stu_rot = rotate_rows_fwht32(x_stu, n_tok, cols, padded_cols, threads);
        let dx_rot = x_ref.map(|xr| {
            let dx: Vec<f32> = xr.iter().zip(x_stu).map(|(&r, &s)| r - s).collect();
            rotate_rows_fwht32(&dx, n_tok, cols, padded_cols, threads)
        });

        let sb_cols = SB_COLS;
        let n_sb = padded_cols.div_ceil(sb_cols);
        let blocks: Vec<SuperBlockBlock> = if cfg!(vxt_has_cblas) || n_sb <= 1 {
            (0..n_sb)
                .map(|bi| {
                    let c0 = bi * sb_cols;
                    let bw = (padded_cols - c0).min(sb_cols);
                    build_superblock(
                        c0, bw, n_tok, eff_n, cols, padded_cols, &x_stu_rot, dx_rot.as_deref(), &ch_amp, &tok_w, threads,
                    )
                })
                .collect()
        } else {
            let sb_w = threads.clamp(1, 32).min(n_sb.max(1));
            let sub_t = (threads / sb_w.max(1)).max(1);
            let next_sb = std::sync::atomic::AtomicUsize::new(0);
            let slots: Vec<std::sync::Mutex<Option<SuperBlockBlock>>> =
                (0..n_sb).map(|_| std::sync::Mutex::new(None)).collect();
            let (xsr, dxr, camp, tw_ref) = (&x_stu_rot, dx_rot.as_deref(), &ch_amp, &tok_w);
            std::thread::scope(|sc| {
                for _ in 0..sb_w {
                    let (next_ref, slots_ref) = (&next_sb, &slots);
                    sc.spawn(move || loop {
                        let bi = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if bi >= n_sb {
                            break;
                        }
                        let c0 = bi * sb_cols;
                        let bw = (padded_cols - c0).min(sb_cols);
                        let blk = build_superblock(
                            c0, bw, n_tok, eff_n, cols, padded_cols, xsr, dxr, camp, tw_ref, sub_t,
                        );
                        *slots_ref[bi].lock().unwrap() = Some(blk);
                    });
                }
            });
            slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
        };

        #[cfg(not(test))]
        let x_stu_rot = Vec::new();
        Self { x_stu_rot, ch_amp, tok_w, blocks }
    }
}

#[cfg(test)]
mod tests {
    use super::SuperBlockHessian;
    use crate::turbo_linalg::{rotate_rows_fwht32, sgemm_nt};

    #[test]
    fn superblock_ldl_and_drift_identities() {
        let (n_tok, cols) = (256usize, 256usize);
        let mut x_stu = vec![0.0f32; n_tok * cols];
        let mut rng = 0x1234_5678_9abc_def0u64;
        for t in 0..n_tok {
            let factor = if t % 7 == 0 { 1.8 } else { 0.8 };
            for c in 0..cols {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let z = (((rng >> 40) as f32) / 16777216.0 - 0.5) * factor;
                x_stu[t * cols + c] = if c > 0 { 0.35 * x_stu[t * cols + c - 1] + z } else { z };
            }
        }
        let x_ref: Vec<f32> = x_stu.iter().map(|&v| 1.15 * v).collect();
        let hess = SuperBlockHessian::build(&x_stu, Some(&x_ref), n_tok, cols, cols, 2);
        let b = &hess.blocks[0];
        assert!(
            b.alpha.iter().all(|&a| a >= 0.05),
            "alpha floor violated: min={:?}",
            b.alpha.iter().copied().fold(f32::INFINITY, f32::min)
        );
        let mut seen = vec![false; b.bw / 32];
        for &bi in &b.blk_ord {
            seen[bi] = true;
        }
        assert!(seen.iter().all(|&v| v), "blk_ord must be a valid permutation");

        for j in 0..b.bw {
            let mut diag_rec = 0.0f64;
            for k in j..b.bw {
                let mkj = b.m[k * b.bw + j] as f64;
                diag_rec += (b.alpha[k] as f64) * mkj * mkj;
            }
            let rel = (diag_rec - b.h_diag[j] as f64).abs() / (b.h_diag[j] as f64).max(1e-8);
            assert!(rel < 1e-4, "diag mismatch at {j}: rel={rel}");
        }

        let bd = b.b_drift.as_ref().expect("expected drift operator");
        let rows = 16usize;
        let w: Vec<f32> = (0..rows * cols).map(|i| (((i * 17) % 23) as f32 - 11.0) * 0.04).collect();
        let mut w_star = w.clone();
        sgemm_nt(rows, cols, cols, 1.0, &w, cols, bd, cols, 1.0, &mut w_star, cols, 1);

        let (mut y_ref, mut y_base, mut y_star) =
            (vec![0.0; n_tok * rows], vec![0.0; n_tok * rows], vec![0.0; n_tok * rows]);
        let xr_rot = rotate_rows_fwht32(&x_ref, n_tok, cols, cols, 1);
        sgemm_nt(n_tok, rows, cols, 1.0, &xr_rot, cols, &w, cols, 0.0, &mut y_ref, rows, 1);
        sgemm_nt(n_tok, rows, cols, 1.0, &hess.x_stu_rot, cols, &w, cols, 0.0, &mut y_base, rows, 1);
        sgemm_nt(n_tok, rows, cols, 1.0, &hess.x_stu_rot, cols, &w_star, cols, 0.0, &mut y_star, rows, 1);
        let e_base: f64 = y_ref.iter().zip(&y_base).map(|(a, c)| ((a - c) as f64).powi(2)).sum();
        let e_star: f64 = y_ref.iter().zip(&y_star).map(|(a, c)| ((a - c) as f64).powi(2)).sum();
        assert!(e_star < e_base * 0.85, "drift correction did not reduce loss: {e_star} vs {e_base}");

        let x_noisy: Vec<f32> = x_stu
            .iter()
            .map(|&v| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                v + (((rng >> 40) as f32) / 16777216.0 - 0.5) * 0.05
            })
            .collect();
        let hess_n = SuperBlockHessian::build(&x_stu, Some(&x_noisy), n_tok, cols, cols, 2);
        let max_rn = hess_n.blocks[0].b_drift.as_ref().map_or(0.0f64, |bmat| {
            bmat.chunks_exact(cols)
                .map(|r| r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt())
                .fold(0.0, f64::max)
        });
        assert!(max_rn < 0.03, "James-Stein failed to suppress spurious drift: max_rn={max_rn}");
    }
}
