//! Multi-threaded CPU linear algebra (`sgemm_nt`), FWHT-32 batch rotation, cgroup-aware thread
//! budgeting, and Super-Block (`SB = 256`) `LDL^T` Hessian + `32x32` block James-Stein drift operator.

use crate::wht2::fwht32_inplace;
use crate::wht_turbo::apply_col_sign;
use crate::wht_turbo_gptq::HadamardBlockChol;

pub(crate) const SB_COLS: usize = 256;
const SQRT32: f64 = 5.656_854_249_492_381;

/// Determines safe worker count honoring Linux container CPU quotas (`cgroup` v2 & v1).
pub(crate) fn effective_cpu_threads(requested: usize) -> usize {
    let host = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut cap = host;
    #[cfg(target_os = "linux")]
    {
        if let Ok(s) = std::fs::read_to_string("/sys/fs/cgroup/cpu.max") {
            let mut it = s.split_whitespace();
            if let (Some(Ok(qv)), Some(Ok(pv))) = (it.next().map(str::parse::<u64>), it.next().map(str::parse::<u64>)) {
                if pv > 0 { cap = cap.min((qv / pv).max(1) as usize); }
            }
        } else if let (Ok(qs), Ok(ps)) = (
            std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        ) {
            if let (Ok(qv), Ok(pv)) = (qs.trim().parse::<i64>(), ps.trim().parse::<i64>()) {
                if qv > 0 && pv > 0 { cap = cap.min((qv / pv).max(1) as usize); }
            }
        }
    }
    let cap = cap.clamp(1, 32);
    if requested > 0 { requested.min(cap).max(1) } else { cap }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn sgemm_nt_band_inner<const USE_FMA: bool>(
    m: usize, n: usize, k: usize, alpha: f32,
    a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize,
) {
    macro_rules! fm { ($x:expr, $y:expr, $z:expr) => { if USE_FMA { ($x).mul_add($y, $z) } else { $z + ($x) * ($y) } }; }
    #[inline(always)]
    fn hsum8(v: [f32; 8]) -> f32 { (v[0] + v[4]) + (v[1] + v[5]) + (v[2] + v[6]) + (v[3] + v[7]) }
    const KC: usize = 256;
    for k0 in (0..k).step_by(KC) {
        let k1 = (k0 + KC).min(k);
        let mut r = 0usize;
        while r + 2 <= m {
            let (a0, a1) = (&a[r * lda..], &a[(r + 1) * lda..]);
            let mut j = 0usize;
            while j + 4 <= n {
                let (b0, b1, b2, b3) = (&b[j * ldb..], &b[(j + 1) * ldb..], &b[(j + 2) * ldb..], &b[(j + 3) * ldb..]);
                let (mut s00, mut s01, mut s02, mut s03) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
                let (mut s10, mut s11, mut s12, mut s13) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
                let mut p = k0;
                while p + 8 <= k1 {
                    let (av0, av1) = (&a0[p..p + 8], &a1[p..p + 8]);
                    let (bv0, bv1, bv2, bv3) = (&b0[p..p + 8], &b1[p..p + 8], &b2[p..p + 8], &b3[p..p + 8]);
                    for l in 0..8 {
                        s00[l] = fm!(av0[l], bv0[l], s00[l]); s01[l] = fm!(av0[l], bv1[l], s01[l]);
                        s02[l] = fm!(av0[l], bv2[l], s02[l]); s03[l] = fm!(av0[l], bv3[l], s03[l]);
                        s10[l] = fm!(av1[l], bv0[l], s10[l]); s11[l] = fm!(av1[l], bv1[l], s11[l]);
                        s12[l] = fm!(av1[l], bv2[l], s12[l]); s13[l] = fm!(av1[l], bv3[l], s13[l]);
                    }
                    p += 8;
                }
                while p < k1 {
                    let (av0, av1) = (a0[p], a1[p]);
                    s00[0] = fm!(av0, b0[p], s00[0]); s01[0] = fm!(av0, b1[p], s01[0]);
                    s02[0] = fm!(av0, b2[p], s02[0]); s03[0] = fm!(av0, b3[p], s03[0]);
                    s10[0] = fm!(av1, b0[p], s10[0]); s11[0] = fm!(av1, b1[p], s11[0]);
                    s12[0] = fm!(av1, b2[p], s12[0]); s13[0] = fm!(av1, b3[p], s13[0]);
                    p += 1;
                }
                let vs = [[hsum8(s00), hsum8(s01), hsum8(s02), hsum8(s03)], [hsum8(s10), hsum8(s11), hsum8(s12), hsum8(s13)]];
                for rr in 0..2 {
                    let crow = &mut c[(r + rr) * ldc + j..(r + rr) * ldc + j + 4];
                    for cc in 0..4 { crow[cc] = fm!(alpha, vs[rr][cc], crow[cc]); }
                }
                j += 4;
            }
            for jj in j..n {
                let bj = &b[jj * ldb..];
                let (mut t0, mut t1) = ([0.0f32; 8], [0.0f32; 8]);
                let mut p = k0;
                while p + 8 <= k1 {
                    for l in 0..8 {
                        let bv = bj[p + l];
                        t0[l] = fm!(a0[p + l], bv, t0[l]); t1[l] = fm!(a1[p + l], bv, t1[l]);
                    }
                    p += 8;
                }
                while p < k1 {
                    let bv = bj[p];
                    t0[0] = fm!(a0[p], bv, t0[0]); t1[0] = fm!(a1[p], bv, t1[0]);
                    p += 1;
                }
                c[r * ldc + jj] = fm!(alpha, hsum8(t0), c[r * ldc + jj]);
                c[(r + 1) * ldc + jj] = fm!(alpha, hsum8(t1), c[(r + 1) * ldc + jj]);
            }
            r += 2;
        }
        for rr in r..m {
            let ar = &a[rr * lda..];
            let crow = &mut c[rr * ldc..rr * ldc + n];
            for (jj, cv) in crow.iter_mut().enumerate() {
                let bj = &b[jj * ldb..];
                let mut acc = [0.0f32; 8];
                let mut p = k0;
                while p + 8 <= k1 {
                    for l in 0..8 { acc[l] = fm!(ar[p + l], bj[p + l], acc[l]); }
                    p += 8;
                }
                while p < k1 { acc[0] = fm!(ar[p], bj[p], acc[0]); p += 1; }
                *cv = fm!(alpha, hsum8(acc), *cv);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_nt_band_avx2(
    m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize, c: &mut [f32], ldc: usize,
) {
    sgemm_nt_band_inner::<true>(m, n, k, alpha, a, lda, b, ldb, c, ldc);
}

#[allow(clippy::too_many_arguments)]
fn sgemm_nt_band(
    m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize, c: &mut [f32], ldc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe { sgemm_nt_band_avx2(m, n, k, alpha, a, lda, b, ldb, c, ldc); }
            return;
        }
        sgemm_nt_band_inner::<false>(m, n, k, alpha, a, lda, b, ldb, c, ldc);
    }
    #[cfg(not(target_arch = "x86_64"))]
    sgemm_nt_band_inner::<true>(m, n, k, alpha, a, lda, b, ldb, c, ldc);
}

/// Computes row-major `C[0..m, 0..n] = alpha * A[0..m, 0..k] * B[0..n, 0..k]^T + beta * C[0..m, 0..n]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_nt(
    m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize, threads: usize,
) {
    if m == 0 || n == 0 { return; }
    for r in 0..m {
        let crow = &mut c[r * ldc..r * ldc + n];
        if beta == 0.0 { crow.fill(0.0); } else if beta != 1.0 { for v in crow { *v *= beta; } }
    }
    if k == 0 || alpha == 0.0 { return; }
    let work = m.saturating_mul(n).saturating_mul(k);
    let workers = threads.clamp(1, 32);
    if workers > 1 && work >= 262_144 && m >= 32 {
        let rows_per_w = m.div_ceil(workers).div_ceil(4) * 4;
        std::thread::scope(|s| {
            let (mut rem_c, mut r0) = (&mut c[..], 0usize);
            while r0 < m {
                let rc = (m - r0).min(rows_per_w);
                let (c_band, rest) = if r0 + rc < m { rem_c.split_at_mut(rc * ldc) } else { (rem_c, &mut [][..]) };
                rem_c = rest;
                let a_band = &a[r0 * lda..];
                s.spawn(move || sgemm_nt_band(rc, n, k, alpha, a_band, lda, b, ldb, c_band, ldc));
                r0 += rc;
            }
        });
    } else if workers > 1 && work >= 262_144 && n >= 64 {
        let cols_per_w = n.div_ceil(workers).div_ceil(4) * 4;
        let mut parts: Vec<(usize, usize, Vec<f32>)> = (0..n.div_ceil(cols_per_w))
            .map(|ci| { let c0 = ci * cols_per_w; let nc = (n - c0).min(cols_per_w); (c0, nc, vec![0.0f32; m * nc]) })
            .collect();
        std::thread::scope(|s| {
            for (c0, nc, buf) in &mut parts {
                let (c0v, ncv) = (*c0, *nc);
                s.spawn(move || sgemm_nt_band(m, ncv, k, alpha, a, lda, &b[c0v * ldb..], ldb, buf, ncv));
            }
        });
        for (c0, nc, buf) in parts {
            for r in 0..m { for j in 0..nc { c[r * ldc + c0 + j] += buf[r * nc + j]; } }
        }
    } else {
        sgemm_nt_band(m, n, k, alpha, a, lda, b, ldb, c, ldc);
    }
}

/// Applies Turbo column signs and `FWHT_32` to `x[n_tok, cols]` producing `[n_tok, padded_cols]`.
pub(crate) fn rotate_rows_fwht32(x: &[f32], n_tok: usize, cols: usize, padded_cols: usize, threads: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * padded_cols];
    let rot_chunk = |t0: usize, dst: &mut [f32]| {
        for (dt, row_out) in dst.chunks_exact_mut(padded_cols).enumerate() {
            let src = &x[(t0 + dt) * cols..(t0 + dt + 1) * cols];
            for c in 0..cols { row_out[c] = apply_col_sign(src[c], c); }
            for blk in row_out.chunks_exact_mut(32) { fwht32_inplace(blk.try_into().unwrap()); }
        }
    };
    let workers = threads.clamp(1, 32).min(n_tok.max(1));
    if workers > 1 && n_tok >= 64 {
        let tpw = n_tok.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, ch) in out.chunks_mut(tpw * padded_cols).enumerate() {
                s.spawn(move || rot_chunk(ci * tpw, ch));
            }
        });
    } else {
        rot_chunk(0, &mut out);
    }
    out
}

pub(crate) struct SuperBlockBlock {
    pub(crate) c0: usize,
    pub(crate) bw: usize,
    pub(crate) h_diag: Vec<f32>,
    pub(crate) alpha: Vec<f32>,
    /// Row-major `[bw, bw]` unit lower-triangular `M` (`A = M^T diag(alpha) M`).
    pub(crate) m: Vec<f32>,
    /// Optional `[bw, bw]` drift operator `B = alpha_eff * H_drift^{-1} C_b^T` (`W^* += W_b B^T`).
    pub(crate) b_drift: Option<Vec<f32>>,
}

pub(crate) struct SuperBlockHessian {
    pub(crate) x_stu_rot: Vec<f32>,
    pub(crate) ch_amp: Vec<f32>,
    pub(crate) tok_w: Vec<f32>,
    pub(crate) blocks: Vec<SuperBlockBlock>,
}

fn solve_cholesky_32(h32: &[f64; 1024], rhs_mat: &[f32; 1024], scale: f64, vlen: usize) -> [f32; 1024] {
    let mut l = [0.0f64; 1024];
    for i in 0..32 {
        let mut diag = h32[i * 32 + i];
        for p in 0..i { diag -= l[i * 32 + p] * l[i * 32 + p]; }
        let lii = diag.max(1e-12).sqrt();
        l[i * 32 + i] = lii;
        let inv = 1.0 / lii;
        for r in (i + 1)..32 {
            let mut sum = h32[r * 32 + i];
            for p in 0..i { sum -= l[r * 32 + p] * l[i * 32 + p]; }
            l[r * 32 + i] = sum * inv;
        }
    }
    let (mut b32, mut tmp) = ([0.0f32; 1024], [0.0f64; 32]);
    for col in 0..vlen {
        for i in 0..32 {
            let mut sum = scale * (rhs_mat[i * 32 + col] as f64);
            for p in 0..i { sum -= l[i * 32 + p] * tmp[p]; }
            tmp[i] = sum / l[i * 32 + i];
        }
        for i in (0..32).rev() {
            let mut sum = tmp[i];
            for r in (i + 1)..32 { sum -= l[r * 32 + i] * tmp[r]; }
            tmp[i] = sum / l[i * 32 + i];
            b32[i * 32 + col] = tmp[i] as f32;
        }
    }
    for j in 0..vlen {
        let row = &mut b32[j * 32..j * 32 + vlen];
        let norm = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        if norm > 0.18 {
            let sc = (0.18 / norm) as f32;
            for v in row { *v *= sc; }
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
        if gc0 >= cols { continue; }
        let vlen = (cols - gc0).min(32);
        let xb_32 = &xb_t[b0 * n_tok..(b0 + 32) * n_tok];
        dxb_32.fill(0.0);
        for t in 0..n_tok {
            let r_dx = &dxr[t * padded_cols + gc0..t * padded_cols + gc0 + vlen];
            let tw = tok_w[t];
            for j in 0..vlen { dxb_32[j * n_tok + t] = r_dx[j] * tw; }
        }
        let (mut xs_n2, mut dx_n2, mut tok_prod) = (0.0f64, 0.0f64, 0.0f64);
        for t in 0..n_tok {
            let (mut xs_t, mut dx_t) = (0.0f64, 0.0f64);
            for j in 0..32 {
                let (sv, dv) = (xb_32[j * n_tok + t] as f64, dxb_32[j * n_tok + t] as f64);
                xs_t += sv * sv;
                dx_t += dv * dv;
            }
            xs_n2 += xs_t; dx_n2 += dx_t; tok_prod += xs_t * dx_t;
        }
        if dx_n2 <= 1e-16 * xs_n2 || xs_n2 <= 1e-30 { continue; }
        c_bt_32.fill(0.0);
        sgemm_nt(32, 32, n_tok, scale_n, xb_32, n_tok, &dxb_32, n_tok, 0.0, &mut c_bt_32, 32, 1);
        let c32_f2_raw: f64 = c_bt_32.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let js = (1.0 - 2.5 * (tok_prod / (eff_n * eff_n)) / c32_f2_raw.max(1e-30)).max(0.0);
        if js <= 1e-4 { continue; }
        let mut h32 = [0.0f64; 1024];
        let p_sl = &prior32[sb32 * 1024..(sb32 + 1) * 1024];
        for u in 0..32 {
            for v in 0..32 {
                h32[u * 32 + v] = if u >= vlen || v >= vlen { if u == v { 1.0 } else { 0.0 } } else {
                    0.80 * (h_emp[(b0 + u) * bw + (b0 + v)] as f64) * inv_md
                        + 0.20 * p_sl[u * 32 + v] + if u == v { 0.20 * blk_rel[sb32] } else { 0.0 }
                };
            }
        }
        let alpha_eff = 0.55 * js * (1.0 / (1.0 + (dx_n2 / xs_n2).sqrt())).clamp(0.25, 1.0);
        let b32 = solve_cholesky_32(&h32, &c_bt_32, alpha_eff * inv_md, vlen);
        for j in 0..vlen { for col in 0..vlen { b_out[(b0 + j) * bw + (b0 + col)] = b32[j * 32 + col]; } }
        any_nonzero = true;
    }
    any_nonzero.then_some(b_out)
}

#[allow(clippy::too_many_arguments)]
fn build_superblock(
    c0: usize, n_tok: usize, eff_n: f64, cols: usize, padded_cols: usize,
    x_stu_rot: &[f32], dx_rot: Option<&[f32]>, ch_amp: &[f32], tok_w: &[f32], threads: usize,
) -> SuperBlockBlock {
    let bw = (padded_cols - c0).min(SB_COLS);
    let mut xb_t = vec![0.0f32; bw * n_tok];
    for t in 0..n_tok {
        let row = &x_stu_rot[t * padded_cols + c0..t * padded_cols + c0 + bw];
        let tw = tok_w[t];
        for j in 0..bw {
            xb_t[j * n_tok + t] = row[j] * tw;
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

    let mut prior32 = vec![0.0f64; (bw / 32) * 1024];
    let mut blk_rel = vec![1.0f64; bw / 32];
    for sb32 in 0..(bw / 32) {
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
    }

    let mut s = vec![0.0f64; bw * bw];
    for u in 0..bw {
        let valid_u = c0 + u < cols;
        let sb32 = u / 32;
        let u32 = u % 32;
        let g128 = u / 128;
        for v in 0..bw {
            let valid_v = c0 + v < cols;
            s[u * bw + v] = if !valid_u || !valid_v {
                if u == v { 1.0 } else { 0.0 }
            } else {
                let hn = (h_emp[u * bw + v] as f64) * inv_md;
                if sb32 == v / 32 {
                    0.70 * hn + 0.20 * prior32[sb32 * 1024 + u32 * 32 + (v % 32)] + if u == v { 0.10 } else { 0.0 }
                } else if g128 == v / 128 {
                    0.35 * hn
                } else {
                    0.20 * hn
                }
            };
        }
        s[u * bw + u] = s[u * bw + u].max(1e-6);
    }

    let h_diag: Vec<f32> = (0..bw).map(|k| s[k * bw + k] as f32).collect();
    let b_drift = dx_rot.and_then(|dxr| {
        solve_drift_block32(c0, bw, cols, n_tok, eff_n, inv_md, padded_cols, &xb_t, dxr, tok_w, &h_emp, &prior32, &blk_rel)
    });

    // Reverse-order f64 LDL^T: H_reg = M^T diag(alpha) M, M unit lower-triangular.
    let mut alpha = vec![1.0f32; bw];
    let mut m = vec![0.0f32; bw * bw];
    let mut mk = vec![0.0f64; bw];
    for k in (0..bw).rev() {
        let ak = s[k * bw + k].max(1e-6);
        alpha[k] = ak as f32;
        m[k * bw + k] = 1.0;
        let inv_ak = 1.0 / ak;
        for j in 0..k {
            mk[j] = s[k * bw + j] * inv_ak;
            m[k * bw + j] = mk[j] as f32;
        }
        for u in 0..k {
            let a_mu = ak * mk[u];
            let row_u = &mut s[u * bw..u * bw + u + 1];
            for v in 0..=u {
                row_u[v] -= a_mu * mk[v];
            }
        }
    }
    SuperBlockBlock { c0, bw, h_diag, alpha, m, b_drift }
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

        let n_sb = padded_cols.div_ceil(SB_COLS);
        let sb_w = threads.clamp(1, 32).min(n_sb.max(1));
        let sub_t = (threads / sb_w.max(1)).max(1);
        let next_sb = std::sync::atomic::AtomicUsize::new(0);
        let slots: Vec<std::sync::Mutex<Option<SuperBlockBlock>>> = (0..n_sb).map(|_| std::sync::Mutex::new(None)).collect();
        let (xsr, dxr, camp, tw_ref) = (&x_stu_rot, dx_rot.as_deref(), &ch_amp, &tok_w);
        std::thread::scope(|sc| {
            for _ in 0..sb_w {
                let (next_ref, slots_ref) = (&next_sb, &slots);
                sc.spawn(move || loop {
                    let bi = next_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if bi >= n_sb {
                        break;
                    }
                    let blk = build_superblock(bi * SB_COLS, n_tok, eff_n, cols, padded_cols, xsr, dxr, camp, tw_ref, sub_t);
                    *slots_ref[bi].lock().unwrap() = Some(blk);
                });
            }
        });
        let blocks: Vec<SuperBlockBlock> = slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();

        Self { x_stu_rot, ch_amp, tok_w, blocks }
    }
}

#[cfg(test)]
mod tests {
    use super::{sgemm_nt, SuperBlockHessian};

    #[test]
    fn sgemm_nt_matches_naive_across_shapes() {
        for &(m, n, k, beta) in &[(37usize, 53usize, 69usize, 0.0f32), (64, 128, 256, 0.5)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 29) as f32 - 14.0) * 0.03).collect();
            let b: Vec<f32> = (0..n * k).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
            let mut c_ref = vec![1.25f32; m * n];
            let mut c_got = c_ref.clone();
            for r in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for p in 0..k {
                        sum += (a[r * k + p] as f64) * (b[j * k + p] as f64);
                    }
                    c_ref[r * n + j] = (0.75 * sum + (beta as f64) * (c_ref[r * n + j] as f64)) as f32;
                }
            }
            sgemm_nt(m, n, k, 0.75, &a, k, &b, k, beta, &mut c_got, n, 4);
            let max_err = c_ref.iter().zip(&c_got).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 1e-4, "shape ({m},{n},{k}) beta={beta} max_err={max_err}");
        }
    }

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
        assert!(b.alpha.iter().all(|&a| a >= 0.099), "alpha floor violated: min={:?}", b.alpha.iter().copied().fold(f32::INFINITY, f32::min));

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

        let (mut y_ref, mut y_base, mut y_star) = (vec![0.0; n_tok * rows], vec![0.0; n_tok * rows], vec![0.0; n_tok * rows]);
        let xr_rot = super::rotate_rows_fwht32(&x_ref, n_tok, cols, cols, 1);
        sgemm_nt(n_tok, rows, cols, 1.0, &xr_rot, cols, &w, cols, 0.0, &mut y_ref, rows, 1);
        sgemm_nt(n_tok, rows, cols, 1.0, &hess.x_stu_rot, cols, &w, cols, 0.0, &mut y_base, rows, 1);
        sgemm_nt(n_tok, rows, cols, 1.0, &hess.x_stu_rot, cols, &w_star, cols, 0.0, &mut y_star, rows, 1);
        let e_base: f64 = y_ref.iter().zip(&y_base).map(|(a, c)| ((a - c) as f64).powi(2)).sum();
        let e_star: f64 = y_ref.iter().zip(&y_star).map(|(a, c)| ((a - c) as f64).powi(2)).sum();
        assert!(e_star < e_base * 0.85, "drift correction did not reduce loss: {e_star} vs {e_base}");

        let x_noisy: Vec<f32> = x_stu.iter().map(|&v| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            v + (((rng >> 40) as f32) / 16777216.0 - 0.5) * 0.05
        }).collect();
        let hess_n = SuperBlockHessian::build(&x_stu, Some(&x_noisy), n_tok, cols, cols, 2);
        let max_rn = hess_n.blocks[0].b_drift.as_ref().map_or(0.0f64, |bmat| {
            bmat.chunks_exact(cols).map(|r| r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()).fold(0.0, f64::max)
        });
        assert!(max_rn < 0.03, "James-Stein failed to suppress spurious drift: max_rn={max_rn}");
    }
}
