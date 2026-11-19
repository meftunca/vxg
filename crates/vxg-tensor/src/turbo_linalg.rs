//! Multi-threaded CPU / BLAS linear algebra (`sgemm_nt`), FWHT-32 batch rotation, and
//! cgroup-aware thread budgeting.

pub(crate) use crate::turbo_hess::SuperBlockHessian;
use crate::wht2::fwht32_inplace;
use crate::wht_turbo::apply_col_sign;

#[cfg(vxt_has_cblas)]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

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
    let work = m.saturating_mul(n).saturating_mul(k);
    #[cfg(vxt_has_cblas)]
    if k > 0 && alpha != 0.0 && work >= 65_536
        && m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize
        && lda <= i32::MAX as usize && ldb <= i32::MAX as usize && ldc <= i32::MAX as usize
    {
        unsafe {
            cblas_sgemm(
                101, 111, 112, m as i32, n as i32, k as i32, alpha,
                a.as_ptr(), lda as i32, b.as_ptr(), ldb as i32, beta, c.as_mut_ptr(), ldc as i32,
            );
        }
        return;
    }
    for r in 0..m {
        let crow = &mut c[r * ldc..r * ldc + n];
        if beta == 0.0 { crow.fill(0.0); } else if beta != 1.0 { for v in crow { *v *= beta; } }
    }
    if k == 0 || alpha == 0.0 { return; }
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

#[cfg(test)]
mod tests {
    use super::sgemm_nt;

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
}
