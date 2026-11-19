//! 8D Gosset Lattice (`E8`) Vector Quantizer with `32`-column `10-bit f16` R-D optimal discrete
//! scale search, full-width `act_order` `LDL-GPTQ`, 2D Lagrangian water-filling, and zero-LUT
//! SIMD bitfield packing (`VXT-Turbo-E8`, `bits = 7`).

use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::turbo_linalg::{rotate_rows_fwht32, sgemm_nt, SuperBlockHessian};
use crate::wht2::fwht32_inplace;
use crate::wht_turbo::{apply_col_sign, TurboWhtMatrix};
use crate::wht_turbo_gptq::HadamardBlockChol;
use crate::wht_turbo_gptq_4d::round_f16_12bit;
use crate::{Error, Result};

const E8_NORM_MSE: f32 = 0.071_682; // E8 lattice normalized second moment G(E8)
const TWO_LN2: f32 = 1.386_294_4;
const LAGR_P: f32 = 0.72;
const RD_RADIUS: i32 = 2;
const BETA_RD_COMP: f32 = 1.005;

/// Packs an `8D` Gosset lattice vector (`u_j in 0..=15`, `sum(u_j) & 1 == 0`, `c in 0..=1`) into `u32`.
#[inline(always)]
pub(crate) fn pack_e8_word(u: &[u8; 8], c: u8) -> u32 {
    debug_assert_eq!((u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7]) & 1, 0);
    (u[0] as u32)
        | ((u[1] as u32) << 4) | ((u[2] as u32) << 8) | ((u[3] as u32) << 12)
        | ((u[4] as u32) << 16) | ((u[5] as u32) << 20) | ((u[6] as u32) << 24)
        | (((u[7] as u32) >> 1) << 28) | (((c & 1) as u32) << 31)
}

/// Decodes a packed `E8` `u32` word into `8` `f32` coordinates in `[-8.0, 7.5]`.
#[inline(always)]
pub(crate) fn decode_e8_word(w: u32) -> [f32; 8] {
    let m7 = w & 0x0fff_ffff;
    let mut p = m7 ^ (m7 >> 4);
    p ^= p >> 8;
    let u7 = ((w >> 27) & 14) | ((p ^ (p >> 16)) & 1);
    let bias = if (w >> 31) != 0 { -7.5f32 } else { -8.0f32 };
    [
        ((w & 15) as f32) + bias, (((w >> 4) & 15) as f32) + bias,
        (((w >> 8) & 15) as f32) + bias, (((w >> 12) & 15) as f32) + bias,
        (((w >> 16) & 15) as f32) + bias, (((w >> 20) & 15) as f32) + bias,
        (((w >> 24) & 15) as f32) + bias, (u7 as f32) + bias,
    ]
}

/// Computes `<decode_e8_word(w), xv[0..8]>` without lookup tables.
#[inline(always)]
pub(crate) fn dot_e8_word(w: u32, xv: &[f32]) -> f32 {
    let q = decode_e8_word(w);
    let s0 = q[0].mul_add(xv[0], q[1].mul_add(xv[1], q[2].mul_add(xv[2], q[3] * xv[3])));
    let s1 = q[4].mul_add(xv[4], q[5].mul_add(xv[5], q[6].mul_add(xv[6], q[7] * xv[7])));
    s0 + s1
}

#[inline(always)]
fn opp_parity_u4(ub: u8, yf: f32) -> u8 {
    if ub == 0 { 1 } else if ub == 15 { 14 } else if yf >= (ub as f32) { ub + 1 } else { ub - 1 }
}

/// Solves one `8D` Gosset lattice vector under an `8x8` unit lower-triangular `LDL^T` metric.
#[inline]
pub(crate) fn solve_e8_8d_gptq(
    wk8: &[f32; 8], wb8: &[f32; 8], al8: &[f32; 8], m8: &[[f32; 8]; 8],
) -> ([f32; 8], u32) {
    let (mut best_loss, mut best_q, mut best_w) = (f32::INFINITY, [0.0f32; 8], 0u32);
    for c in 0u8..=1u8 {
        let bias = if c != 0 { -7.5f32 } else { -8.0f32 };
        let (mut u_sel, mut q_sel, mut alt_u) = ([8u8; 8], [0.0f32; 8], [8u8; 8]);
        let (mut y_arr, mut par_sum) = ([0.0f32; 8], 0u8);
        for i in 0..8 {
            let mut yi = wk8[i];
            for j in 0..i { yi += m8[i][j] * (wb8[j] - q_sel[j]); }
            y_arr[i] = yi;
            let yf = yi - bias;
            let ub = (yf.round() as i32).clamp(0, 15) as u8;
            u_sel[i] = ub;
            q_sel[i] = (ub as f32) + bias;
            alt_u[i] = opp_parity_u4(ub, yf);
            par_sum ^= ub & 1;
        }
        let tot_loss = if par_sum != 0 {
            let (mut best_fl, mut best_fu, mut best_fq) = (f32::INFINITY, u_sel, q_sel);
            for idx_f in 0..8 {
                let (mut tu, mut tq) = (u_sel, q_sel);
                tu[idx_f] = alt_u[idx_f];
                tq[idx_f] = (tu[idx_f] as f32) + bias;
                for i in (idx_f + 1)..8 {
                    let mut yi = wk8[i];
                    for j in 0..i { yi += m8[i][j] * (wb8[j] - tq[j]); }
                    let yf = yi - bias;
                    let mut ub = (yf.round() as i32).clamp(0, 15) as u8;
                    if i == 7 {
                        let req = (tu[0] ^ tu[1] ^ tu[2] ^ tu[3] ^ tu[4] ^ tu[5] ^ tu[6]) & 1;
                        if (ub & 1) != req { ub = opp_parity_u4(ub, yf); }
                    }
                    tu[i] = ub;
                    tq[i] = (ub as f32) + bias;
                }
                let mut fl = 0.0f32;
                for i in 0..8 {
                    let mut vi = wk8[i] - tq[i];
                    for j in 0..i { vi += m8[i][j] * (wb8[j] - tq[j]); }
                    fl = al8[i].mul_add(vi * vi, fl);
                }
                if fl < best_fl { best_fl = fl; best_fu = tu; best_fq = tq; }
            }
            u_sel = best_fu;
            q_sel = best_fq;
            best_fl
        } else {
            let mut fl = 0.0f32;
            for i in 0..8 {
                let d = y_arr[i] - q_sel[i];
                fl = al8[i].mul_add(d * d, fl);
            }
            fl
        };
        if tot_loss < best_loss {
            best_loss = tot_loss;
            best_q = q_sel;
            best_w = pack_e8_word(&u_sel, c);
        }
    }
    (best_q, best_w)
}

/// Solves one `32`-column `FWHT_32` block (`4` `E8` vectors + R-D optimal `10-bit f16` scale).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn solve_block32_e8_gptq(
    lc0: usize, bw: usize, alpha: &[f32], m: &[f32], h_diag: &[f32],
    work_blk: &[f32], wt_blk: &[f32], beta: f32, rd_radius: i32, rd_slope: f32,
    w_out: &mut [u32; 4], wq_out: &mut [f32; 32],
) -> u16 {
    let mut v_tgt = [0.0f32; 32];
    let (mut e0, mut h_sum, mut al_sum, mut max_abs) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for k in 0..32 {
        let mut s = work_blk[k];
        let mk = &m[(lc0 + k) * bw + lc0..];
        for j in 0..k { s += mk[j] * wt_blk[j]; }
        v_tgt[k] = s;
        let wk = work_blk[k];
        let hk = h_diag[lc0 + k];
        e0 += hk * wk * wk;
        h_sum += hk;
        al_sum += alpha[lc0 + k];
        let av = wk.abs();
        if av > max_abs { max_abs = av; }
    }
    let rms = (e0 / h_sum.max(1e-12)).sqrt();
    let sc_init = (rms / beta.max(0.5)).max(max_abs / 7.25);
    let sf0 = round_f16_12bit(f32_to_f16(sc_init));
    let sc0 = f16_to_f32(sf0);
    wq_out.fill(0.0);
    w_out.fill(pack_e8_word(&[8; 8], 0));
    if sc0 <= 0.0 { return 0; }

    let (mut m8_arr, mut al8_arr) = ([[[0.0f32; 8]; 8]; 4], [[1.0f32; 8]; 4]);
    for q8 in 0..4 {
        let b0 = lc0 + q8 * 8;
        for i in 0..8 {
            al8_arr[q8][i] = alpha[b0 + i];
            for j in 0..i { m8_arr[q8][i][j] = m[(b0 + i) * bw + b0 + j]; }
        }
    }

    let eval_given_sc = |sc: f32| -> (f32, [f32; 32], [u32; 4]) {
        let inv_sc = 1.0 / sc;
        let (mut wk, mut wb) = ([0.0f32; 32], [0.0f32; 32]);
        for k in 0..32 { wk[k] = work_blk[k] * inv_sc; wb[k] = wt_blk[k] * inv_sc; }
        let (mut q32, mut words) = ([0.0f32; 32], [0u32; 4]);
        for q8 in 0..4 {
            let i0 = q8 * 8;
            let (oq, ow) = solve_e8_8d_gptq(
                wk[i0..i0 + 8].try_into().unwrap(),
                wb[i0..i0 + 8].try_into().unwrap(),
                &al8_arr[q8], &m8_arr[q8],
            );
            q32[i0..i0 + 8].copy_from_slice(&oq);
            words[q8] = ow;
            for r in (i0 + 8)..32 {
                let mr = &m[(lc0 + r) * bw + lc0 + i0..];
                let mut acc = 0.0f32;
                for j in 0..8 { acc = mr[j].mul_add(wb[i0 + j] - oq[j], acc); }
                wk[r] += acc;
            }
        }
        let mut loss = 0.0f32;
        for k in 0..32 {
            let mut s = q32[k];
            let mk = &m[(lc0 + k) * bw + lc0..];
            for j in 0..k { s += mk[j] * q32[j]; }
            let d = v_tgt[k] - sc * s;
            loss = alpha[lc0 + k].mul_add(d * d, loss);
        }
        (loss, q32, words)
    };

    // Exact Lagrangian R-D scale search over neighboring 10-bit f16 discrete grid points:
    // min_sf  D(sf) + rd_slope * (al_sum * G(E8) * sc0^2) * (-log2(sc / sc0)).
    let rd_lam = rd_slope * al_sum * E8_NORM_MSE * sc0 * sc0;
    let (mut best_obj, mut best_sf, mut best_sc) = (f32::INFINITY, sf0, sc0);
    let (mut best_q, mut best_w) = ([0.0f32; 32], [0u32; 4]);
    let sf0_i = sf0 as i32;
    for d in -rd_radius..=rd_radius {
        let cand_i = sf0_i + d * 0x40;
        if !(0x0400..=0x7bc0).contains(&cand_i) { continue; }
        let sf = cand_i as u16;
        let sc = f16_to_f32(sf);
        if sc <= 0.0 { continue; }
        let (loss, q32, words) = eval_given_sc(sc);
        let obj = loss - rd_lam * (sc / sc0).log2();
        if obj < best_obj {
            best_obj = obj;
            best_sf = sf;
            best_sc = sc;
            best_q = q32;
            best_w = words;
        }
    }
    for k in 0..32 { wq_out[k] = best_sc * best_q[k]; }
    *w_out = best_w;
    best_sf
}

/// Encodes `[rows, cols]` into `TurboWhtMatrix` (`bits = 7`) using full-width `act_order`
/// `SuperBlockHessian` `LDL-GPTQ`, 2D Lagrangian water-filling, and discrete R-D scale search.
pub(crate) fn encode_e8_superblock_gptq<F>(
    rows: usize, cols: usize, mut fetch_rows: F, hess: &SuperBlockHessian,
    base_beta: f32, mut on_chunk: Option<&mut dyn FnMut(usize, usize, &[f32])>, threads: usize,
) -> Result<TurboWhtMatrix>
where
    F: FnMut(usize, usize) -> Result<Vec<f32>>,
{
    let (padded_rows, padded_cols) = (rows.div_ceil(4) * 4, cols.div_ceil(256) * 256);
    let (row_tiles, steps, n_blks32) = (padded_rows / 4, padded_cols / 256, padded_cols / 32);
    let mut scales_f16 = vec![0u16; row_tiles * steps * 32];
    let mut packed_words = vec![pack_e8_word(&[8; 8], 0); row_tiles * steps * 128];
    let chunk_rows = if cols > 4096 { 256usize } else { 512usize };
    let workers = threads.clamp(1, 32);
    let eff_base_beta = (base_beta * BETA_RD_COMP).clamp(0.75, 4.00);

    let mut blk_sqrt_m = vec![1.0f32; n_blks32];
    for sb in &hess.blocks {
        for (sbi, &orig_b) in sb.blk_ord.iter().enumerate() {
            let g32 = sb.c0 / 32 + orig_b;
            let lc0 = sbi * 32;
            let mean_m: f32 = sb.alpha[lc0..lc0 + 32].iter().sum::<f32>() * (1.0 / 32.0);
            blk_sqrt_m[g32] = mean_m.max(1e-6).sqrt();
        }
    }
    let valid_blks = cols.div_ceil(32).min(n_blks32).max(1);

    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + chunk_rows).min(rows);
        let rc = r1 - r0;
        let w_chunk = fetch_rows(r0, r1)?;
        if w_chunk.len() != rc * cols {
            return Err(Error::InvalidShape("fetch_rows returned wrong element count"));
        }
        if let Some(cb) = on_chunk.as_deref_mut() { cb(r0, r1, &w_chunk); }

        let w_rot = rotate_rows_fwht32(&w_chunk, rc, cols, padded_cols, threads);
        let mut blk_betas = vec![eff_base_beta; rc * n_blks32];
        for r in 0..rc {
            let dst = &mut blk_betas[r * n_blks32..(r + 1) * n_blks32];
            let row_w = &w_rot[r * padded_cols..(r + 1) * padded_cols];
            let mut log_sum = 0.0f64;
            for g32 in 0..valid_blks {
                let sl = &row_w[g32 * 32..(g32 + 1) * 32];
                let rms = (sl.iter().map(|&v| v * v).sum::<f32>() * (1.0 / 32.0)).max(1e-12).sqrt();
                let sc = (blk_sqrt_m[g32] * rms).max(1e-12).powf(LAGR_P);
                dst[g32] = sc;
                log_sum += (sc as f64).ln();
            }
            let inv_gm = (-(log_sum / valid_blks as f64)).exp() as f32;
            for g32 in 0..valid_blks {
                dst[g32] = (eff_base_beta * (dst[g32] * inv_gm).clamp(0.48, 2.10)).clamp(0.72, 4.00);
            }
            for g32 in valid_blks..n_blks32 { dst[g32] = eff_base_beta; }
        }

        let mut sf_mat = vec![0u16; rc * n_blks32];
        let mut w4_mat = vec![0u32; rc * n_blks32 * 4];
        let mut err_panel = vec![0.0f32; rc * 32];

        for sb in &hess.blocks {
            let nb = sb.bw / 32;
            let mut w_sb = vec![0.0f32; rc * sb.bw];
            for r in 0..rc {
                let src_row = &w_rot[r * padded_cols..(r + 1) * padded_cols];
                let dst_row = &mut w_sb[r * sb.bw..(r + 1) * sb.bw];
                for (sbi, &orig_b) in sb.blk_ord.iter().enumerate() {
                    let src_c0 = sb.c0 + orig_b * 32;
                    dst_row[sbi * 32..(sbi + 1) * 32].copy_from_slice(&src_row[src_c0..src_c0 + 32]);
                }
            }
            let mut work_sb = w_sb.clone();

            for sbi in 0..nb {
                let lc0 = sbi * 32;
                let orig_g32 = sb.c0 / 32 + sb.blk_ord[sbi];
                let w_p = workers.min(rc.max(1));
                let rpw = rc.div_ceil(w_p);
                std::thread::scope(|s| {
                    let (wt_ref, wk_ref, bb_ref) = (&w_sb, &work_sb, &blk_betas);
                    for (ci, ((sf_b, w4_b), ep_b)) in sf_mat
                        .chunks_mut(rpw * n_blks32)
                        .zip(w4_mat.chunks_mut(rpw * n_blks32 * 4))
                        .zip(err_panel.chunks_mut(rpw * 32))
                        .enumerate()
                    {
                        let (rb0, nrb) = (ci * rpw, ep_b.len() / 32);
                        s.spawn(move || {
                            let (mut w_out, mut wq_out) = ([0u32; 4], [0.0f32; 32]);
                            for dr in 0..nrb {
                                let r = rb0 + dr;
                                let sf = solve_block32_e8_gptq(
                                    lc0, sb.bw, &sb.alpha, &sb.m, &sb.h_diag,
                                    &wk_ref[r * sb.bw + lc0..r * sb.bw + lc0 + 32],
                                    &wt_ref[r * sb.bw + lc0..r * sb.bw + lc0 + 32],
                                    bb_ref[r * n_blks32 + orig_g32], RD_RADIUS, TWO_LN2,
                                    &mut w_out, &mut wq_out,
                                );
                                sf_b[dr * n_blks32 + orig_g32] = sf;
                                let off4 = (dr * n_blks32 + orig_g32) * 4;
                                w4_b[off4..off4 + 4].copy_from_slice(&w_out);
                                let wt = &wt_ref[r * sb.bw + lc0..r * sb.bw + lc0 + 32];
                                for k in 0..32 { ep_b[dr * 32 + k] = wt[k] - wq_out[k]; }
                            }
                        });
                    }
                });
                if lc0 + 32 < sb.bw {
                    sgemm_nt(
                        rc, sb.bw - (lc0 + 32), 32, 1.0, &err_panel, 32,
                        &sb.m[(lc0 + 32) * sb.bw + lc0..], sb.bw, 1.0, &mut work_sb[lc0 + 32..], sb.bw, threads,
                    );
                }
            }
        }

        let tile0 = r0 / 4;
        for tl in 0..rc.div_ceil(4) {
            let g_tile = tile0 + tl;
            for step in 0..steps {
                let ts = g_tile * steps + step;
                let sc32 = &mut scales_f16[ts * 32..(ts + 1) * 32];
                let pw128 = &mut packed_words[ts * 128..(ts + 1) * 128];
                for sr in 0..4 {
                    let r = tl * 4 + sr;
                    if r >= rc { continue; }
                    for b8 in 0..8 {
                        let g32 = step * 8 + b8;
                        sc32[b8 * 4 + sr] = sf_mat[r * n_blks32 + g32];
                        let off4 = (r * n_blks32 + g32) * 4;
                        for sub in 0..4 { pw128[(b8 * 4 + sub) * 4 + sr] = w4_mat[off4 + sub]; }
                    }
                }
            }
        }
        r0 = r1;
    }
    Ok(TurboWhtMatrix { bits: 7, rows, cols, padded_rows, padded_cols, disk_bytes: 0, scales_f16, packed_words })
}

/// Encodes `[rows, cols]` into `bits = 7` using analytical `32x32` `HadamardBlockChol` (`encode` / `token_embd`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_e8_planes<F>(
    rows: usize, cols: usize, padded_cols: usize, weights: &[f32],
    h_rot: Option<&[f32]>, ch_amp: Option<&[f32]>, row_beta: F, threads: usize,
) -> (Vec<u16>, Vec<u32>)
where
    F: Fn(usize) -> f32 + Sync,
{
    let padded_rows = rows.div_ceil(4) * 4;
    let (row_tiles, steps, n_blks32) = (padded_rows / 4, padded_cols / 256, padded_cols / 32);
    let mut blk_m = vec![[0.0f32; 1024]; n_blks32];
    let mut blk_al = vec![[1.0f32; 32]; n_blks32];
    let mut blk_hd = vec![[1.0f32; 32]; n_blks32];
    for b in 0..n_blks32 {
        let c0 = b * 32;
        let amps = ch_amp.map_or(&[][..], |a| if c0 < a.len() { &a[c0..a.len().min(c0 + 32)] } else { &[] });
        let chol = HadamardBlockChol::from_amps(amps);
        let blk_sc = h_rot.and_then(|hr| hr.get(c0).copied()).unwrap_or(chol.mean_d).max(1e-6);
        let ratio = blk_sc / chol.mean_d.max(1e-12);
        let mut spec = chol.d_unrot;
        for v in &mut spec { *v *= ratio; }
        fwht32_inplace(&mut spec);
        let inv_sq32 = 1.0 / 32.0f64.sqrt();
        let mut s = [0.0f64; 1024];
        for u in 0..32 {
            for v in 0..32 { s[u * 32 + v] = (spec[u ^ v] as f64) * inv_sq32; }
            s[u * 32 + u] = s[u * 32 + u].max(1e-8);
            blk_hd[b][u] = s[u * 32 + u] as f32;
        }
        for k in (0..32).rev() {
            let ak = s[k * 32 + k].max(1e-8);
            blk_al[b][k] = ak as f32;
            blk_m[b][k * 32 + k] = 1.0;
            for j in 0..k { blk_m[b][k * 32 + j] = (s[k * 32 + j] / ak) as f32; }
            for u in 0..k {
                let mu = blk_m[b][k * 32 + u] as f64;
                for v in 0..=u { s[u * 32 + v] -= ak * mu * (blk_m[b][k * 32 + v] as f64); }
            }
        }
    }

    let mut scales_f16 = vec![0u16; row_tiles * steps * 32];
    let mut packed_words = vec![pack_e8_word(&[8; 8], 0); row_tiles * steps * 128];
    let workers = threads.max(1).min(row_tiles.max(1));

    let encode_slice = |t0: usize, sc_sl: &mut [u16], pw_sl: &mut [u32]| {
        let mut r_rot = [0.0f32; 32];
        let (mut w_out, mut wq_out) = ([0u32; 4], [0.0f32; 32]);
        for (dt, (sc_tile, pw_tile)) in sc_sl
            .chunks_exact_mut(steps * 32)
            .zip(pw_sl.chunks_exact_mut(steps * 128))
            .enumerate()
        {
            let tile = t0 + dt;
            for sr in 0..4 {
                let r = tile * 4 + sr;
                if r >= rows { continue; }
                let beta = row_beta(r).clamp(0.85, 3.40);
                let src = &weights[r * cols..(r + 1) * cols];
                for step in 0..steps {
                    let sc32 = &mut sc_tile[step * 32..(step + 1) * 32];
                    let pw128 = &mut pw_tile[step * 128..(step + 1) * 128];
                    for b8 in 0..8 {
                        let g32 = step * 8 + b8;
                        let c0 = g32 * 32;
                        r_rot.fill(0.0);
                        for j in 0..32 {
                            if c0 + j < cols { r_rot[j] = apply_col_sign(src[c0 + j], c0 + j); }
                        }
                        fwht32_inplace(&mut r_rot);
                        sc32[b8 * 4 + sr] = solve_block32_e8_gptq(
                            0, 32, &blk_al[g32], &blk_m[g32], &blk_hd[g32],
                            &r_rot, &r_rot, beta, RD_RADIUS, TWO_LN2, &mut w_out, &mut wq_out,
                        );
                        for sub in 0..4 { pw128[(b8 * 4 + sub) * 4 + sr] = w_out[sub]; }
                    }
                }
            }
        }
    };

    if workers <= 1 || row_tiles < 8 {
        encode_slice(0, &mut scales_f16, &mut packed_words);
    } else {
        let tpw = row_tiles.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, (sc_c, pw_c)) in scales_f16
                .chunks_mut(tpw * steps * 32)
                .zip(packed_words.chunks_mut(tpw * steps * 128))
                .enumerate()
            {
                s.spawn(move || encode_slice(ci * tpw, sc_c, pw_c));
            }
        });
    }
    (scales_f16, packed_words)
}

#[cfg(test)]
mod tests {
    use super::{decode_e8_word, encode_e8_superblock_gptq, pack_e8_word};
    use crate::format::{read_vxt_model, write_vxt_model, PackedTensor, VxtEntry};
    use crate::turbo_linalg::{sgemm_nt, SuperBlockHessian};

    #[test]
    fn e8_pack_decode_parity_and_ec_roundtrip() {
        for c in 0u8..=1 {
            let shift = if c != 0 { 0.5f32 } else { 0.0f32 };
            let u = [0u8, 15, 3, 12, 8, 7, 11, 4];
            let q = decode_e8_word(pack_e8_word(&u, c));
            for j in 0..8 { assert!((q[j] - ((u[j] as f32 - 8.0) + shift)).abs() < 1e-6); }
        }
        let (rows, cols, n_tok) = (256usize, 512usize, 256usize);
        let mut s = 0x517c_c1b7_2722_0a95u64;
        let mut rnd = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f32) / 8388608.0 - 1.0
        };
        let x_tr: Vec<f32> = (0..n_tok * cols).map(|i| rnd() * if (i % cols) % 31 == 0 { 5.0 } else { 1.0 }).collect();
        let x_te: Vec<f32> = (0..n_tok * cols).map(|i| rnd() * if (i % cols) % 31 == 0 { 5.0 } else { 1.0 }).collect();
        let w: Vec<f32> = (0..rows * cols).map(|i| rnd() * if ((i % cols) / 32) % 2 == 0 { 0.08 } else { 0.03 }).collect();

        let hess = SuperBlockHessian::build(&x_tr, None, n_tok, cols, cols, 2);
        let m_e8 = encode_e8_superblock_gptq(rows, cols, |r0, r1| Ok(w[r0 * cols..r1 * cols].to_vec()), &hess, 1.32, None, 2).unwrap();
        let m_t3 = crate::wht_turbo::TurboWhtMatrix::encode_with_ch_amp_threads(3, rows, cols, &w, None, Some(&hess.ch_amp), |_| 0.0, 2).unwrap();
        assert!(m_e8.scales_f16.iter().all(|&v| (v & 0x003f) == 0));

        let bytes = write_vxt_model(&[], &[
            VxtEntry { name: "e8".into(), tensor: PackedTensor::TurboWht(m_e8.clone()) },
            VxtEntry { name: "t3".into(), tensor: PackedTensor::TurboWht(m_t3) },
        ]).unwrap();
        let (_, entries) = read_vxt_model(&bytes).unwrap();
        let PackedTensor::TurboWht(dec) = &entries[0].tensor else { panic!("expected TurboWht") };
        assert_eq!(dec.scales_f16, m_e8.scales_f16);
        assert_eq!(dec.packed_words, m_e8.packed_words);
        assert!(dec.bits_per_weight() < entries[1].tensor.bits_per_weight(), "e8={:.4} vs t3={:.4}", dec.bits_per_weight(), entries[1].tensor.bits_per_weight());

        let (w_dec, w_t3) = (entries[0].tensor.decode_f32(), entries[1].tensor.decode_f32());
        let (mut y_ref, mut y_q, mut y_t3) = (vec![0.0f32; n_tok * rows], vec![0.0f32; n_tok * rows], vec![0.0f32; n_tok * rows]);
        sgemm_nt(n_tok, rows, cols, 1.0, &x_te, cols, &w, cols, 0.0, &mut y_ref, rows, 2);
        sgemm_nt(n_tok, rows, cols, 1.0, &x_te, cols, &w_dec, cols, 0.0, &mut y_q, rows, 2);
        sgemm_nt(n_tok, rows, cols, 1.0, &x_te, cols, &w_t3, cols, 0.0, &mut y_t3, rows, 2);
        let p_sig: f64 = y_ref.iter().map(|&v| (v as f64).powi(2)).sum();
        let p_err: f64 = y_ref.iter().zip(&y_q).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let p_et3: f64 = y_ref.iter().zip(&y_t3).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let (sqnr_e8, sqnr_t3) = (10.0 * (p_sig / p_err).log10(), 10.0 * (p_sig / p_et3).log10());
        assert!(sqnr_e8 > sqnr_t3 + 1.10, "e8={sqnr_e8:.2} dB ({:.3} bpw) vs t3={sqnr_t3:.2} dB ({:.3} bpw)", dec.bits_per_weight(), entries[1].tensor.bits_per_weight());
    }
}
