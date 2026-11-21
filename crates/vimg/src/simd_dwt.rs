//! Hardware SIMD acceleration (`ARMv8 NEON` on `aarch64` + runtime-detected `AVX2` on `x86_64`)
//! with 100% deterministic scalar fallbacks (`wasm32`, generic targets) for:
//! - Horizontal & transpose-free row-major vertical `CDF 9/7` (`f32`) 2D wavelet lifting
//! - Horizontal & transpose-free row-major vertical `CDF 5/3` (`i32`) 2D wavelet lifting
//! - Vectorized `f32 -> i16` / `f32 -> i32` round-half-away-from-zero (`FCVTAS`) conversion.
//!
//! Uses non-fused `mul + add` in `f32` lifting so `aarch64`, `x86_64` AVX2, and `wasm32` produce
//! IEEE-754 bit-identical outputs for cross-architecture deterministic `.vimg` archives.

const A97: f32 = -1.586_134_3;
const B97: f32 = -0.052_980_12;
const G97: f32 = 0.882_911_1;
const D97: f32 = 0.443_506_87;
const K97: f32 = 1.230_174_1;
const INV_K97: f32 = 1.0 / K97;

// ============================================================================
// 1. f32 <-> i16/i32 Rounding SIMD (Exact Rust `f32::round()` Semantics)
// ============================================================================

/// Converts `src` (`f32`) to `i16` using exact round-half-away-from-zero + saturation.
#[inline]
pub fn f32_to_i16_round_slice(src: &[f32]) -> Vec<i16> {
    let n = src.len();
    let mut out = vec![0i16; n];
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 8 <= n {
            let v0 = vcvtaq_s32_f32(vld1q_f32(src.as_ptr().add(i)));
            let v1 = vcvtaq_s32_f32(vld1q_f32(src.as_ptr().add(i + 4)));
            let packed = vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
            vst1q_s16(out.as_mut_ptr().add(i), packed);
            i += 8;
        }
        while i + 4 <= n {
            let v0 = vcvtaq_s32_f32(vld1q_f32(src.as_ptr().add(i)));
            vst1_s16(out.as_mut_ptr().add(i), vqmovn_s32(v0));
            i += 4;
        }
    }

    while i < n {
        out[i] = src[i].round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        i += 1;
    }
    out
}

/// Converts `src` (`f32`) to `i32` using exact round-half-away-from-zero (`FCVTAS`).
#[inline]
pub fn f32_to_i32_round_slice(src: &[f32]) -> Vec<i32> {
    let n = src.len();
    let mut out = vec![0i32; n];
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 8 <= n {
            let v0 = vcvtaq_s32_f32(vld1q_f32(src.as_ptr().add(i)));
            let v1 = vcvtaq_s32_f32(vld1q_f32(src.as_ptr().add(i + 4)));
            vst1q_s32(out.as_mut_ptr().add(i), v0);
            vst1q_s32(out.as_mut_ptr().add(i + 4), v1);
            i += 8;
        }
        while i + 4 <= n {
            let v0 = vcvtaq_s32_f32(vld1q_f32(src.as_ptr().add(i)));
            vst1q_s32(out.as_mut_ptr().add(i), v0);
            i += 4;
        }
    }

    while i < n {
        out[i] = src[i].round() as i32;
        i += 1;
    }
    out
}

// ============================================================================
// 2. CDF 9/7 SIMD Primitives (`f32`, IEEE-754 Bit-Exact Across Targets)
// ============================================================================

/// Vectorized `dst[x] += coeff * (l[x] + r[x])` for `f32` slices of equal length.
#[inline]
fn fma_pair_f32(dst: &mut [f32], l: &[f32], r: &[f32], coeff: f32) {
    let w = dst.len();
    let mut x = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vc = vdupq_n_f32(coeff);
        while x + 8 <= w {
            let d0 = vld1q_f32(dst.as_ptr().add(x));
            let s0 = vaddq_f32(vld1q_f32(l.as_ptr().add(x)), vld1q_f32(r.as_ptr().add(x)));
            vst1q_f32(dst.as_mut_ptr().add(x), vaddq_f32(d0, vmulq_f32(vc, s0)));

            let d1 = vld1q_f32(dst.as_ptr().add(x + 4));
            let s1 = vaddq_f32(vld1q_f32(l.as_ptr().add(x + 4)), vld1q_f32(r.as_ptr().add(x + 4)));
            vst1q_f32(dst.as_mut_ptr().add(x + 4), vaddq_f32(d1, vmulq_f32(vc, s1)));
            x += 8;
        }
        while x + 4 <= w {
            let d0 = vld1q_f32(dst.as_ptr().add(x));
            let s0 = vaddq_f32(vld1q_f32(l.as_ptr().add(x)), vld1q_f32(r.as_ptr().add(x)));
            vst1q_f32(dst.as_mut_ptr().add(x), vaddq_f32(d0, vmulq_f32(vc, s0)));
            x += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            x = fma_pair_f32_avx2(dst, l, r, coeff);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let vc = f32x4_splat(coeff);
        while x + 4 <= w {
            let d0 = v128_load(dst.as_ptr().add(x).cast());
            let s0 = f32x4_add(v128_load(l.as_ptr().add(x).cast()), v128_load(r.as_ptr().add(x).cast()));
            v128_store(dst.as_mut_ptr().add(x).cast(), f32x4_add(d0, f32x4_mul(vc, s0)));
            x += 4;
        }
    }

    while x < w {
        dst[x] += coeff * (l[x] + r[x]);
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fma_pair_f32_avx2(dst: &mut [f32], l: &[f32], r: &[f32], coeff: f32) -> usize {
    use core::arch::x86_64::*;
    let w = dst.len();
    let vc = _mm256_set1_ps(coeff);
    let mut x = 0usize;
    while x + 8 <= w {
        let d0 = _mm256_loadu_ps(dst.as_ptr().add(x));
        let sum = _mm256_add_ps(_mm256_loadu_ps(l.as_ptr().add(x)), _mm256_loadu_ps(r.as_ptr().add(x)));
        _mm256_storeu_ps(dst.as_mut_ptr().add(x), _mm256_add_ps(d0, _mm256_mul_ps(vc, sum)));
        x += 8;
    }
    x
}

#[inline]
fn scale_slice_f32(dst: &mut [f32], factor: f32) {
    let w = dst.len();
    let mut x = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let vf = vdupq_n_f32(factor);
        while x + 8 <= w {
            let v0 = vld1q_f32(dst.as_ptr().add(x));
            let v1 = vld1q_f32(dst.as_ptr().add(x + 4));
            vst1q_f32(dst.as_mut_ptr().add(x), vmulq_f32(v0, vf));
            vst1q_f32(dst.as_mut_ptr().add(x + 4), vmulq_f32(v1, vf));
            x += 8;
        }
        while x + 4 <= w {
            let v0 = vld1q_f32(dst.as_ptr().add(x));
            vst1q_f32(dst.as_mut_ptr().add(x), vmulq_f32(v0, vf));
            x += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            x = scale_slice_f32_avx2(dst, factor);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let vf = f32x4_splat(factor);
        while x + 4 <= w {
            let v0 = v128_load(dst.as_ptr().add(x).cast());
            v128_store(dst.as_mut_ptr().add(x).cast(), f32x4_mul(v0, vf));
            x += 4;
        }
    }

    while x < w {
        dst[x] *= factor;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_slice_f32_avx2(dst: &mut [f32], factor: f32) -> usize {
    use core::arch::x86_64::*;
    let w = dst.len();
    let vf = _mm256_set1_ps(factor);
    let mut x = 0usize;
    while x + 8 <= w {
        let v0 = _mm256_loadu_ps(dst.as_ptr().add(x));
        _mm256_storeu_ps(dst.as_mut_ptr().add(x), _mm256_mul_ps(v0, vf));
        x += 8;
    }
    x
}

/// SIMD 1D forward `CDF 9/7` lifting on a contiguous row `data[0..n]`.
#[inline]
pub fn dwt97_1d_forward_simd(data: &mut [f32], scratch: &mut [f32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);

    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 4 <= nh {
            let pair = vld2q_f32(data.as_ptr().add(i * 2));
            vst1q_f32(low.as_mut_ptr().add(i), pair.0);
            vst1q_f32(high.as_mut_ptr().add(i), pair.1);
            i += 4;
        }
    }
    while i < nh {
        low[i] = data[i * 2];
        high[i] = data[i * 2 + 1];
        i += 1;
    }
    if nl > nh {
        low[nh] = data[nh * 2];
    }

    if nh > 1 {
        fma_pair_f32(&mut high[..nh - 1], &low[..nh - 1], &low[1..nh], A97);
    }
    high[nh - 1] += A97 * (low[nh - 1] + low[nh.min(nl - 1)]);

    low[0] += B97 * (high[0] + high[0]);
    if nh > 1 {
        fma_pair_f32(&mut low[1..nh], &high[..nh - 1], &high[1..nh], B97);
    }
    if nl > nh {
        low[nh] += B97 * (high[nh - 1] + high[nh - 1]);
    }

    if nh > 1 {
        fma_pair_f32(&mut high[..nh - 1], &low[..nh - 1], &low[1..nh], G97);
    }
    high[nh - 1] += G97 * (low[nh - 1] + low[nh.min(nl - 1)]);

    low[0] += D97 * (high[0] + high[0]);
    if nh > 1 {
        fma_pair_f32(&mut low[1..nh], &high[..nh - 1], &high[1..nh], D97);
    }
    if nl > nh {
        low[nh] += D97 * (high[nh - 1] + high[nh - 1]);
    }

    scale_slice_f32(low, INV_K97);
    scale_slice_f32(high, K97);
    data[..nl].copy_from_slice(low);
    data[nl..n].copy_from_slice(high);
}

/// SIMD 1D inverse `CDF 9/7` lifting on a contiguous row `data[0..n]`.
#[inline]
pub fn dwt97_1d_inverse_simd(data: &mut [f32], scratch: &mut [f32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);

    low.copy_from_slice(&data[..nl]);
    high.copy_from_slice(&data[nl..n]);
    scale_slice_f32(low, K97);
    scale_slice_f32(high, INV_K97);

    low[0] += -D97 * (high[0] + high[0]);
    if nh > 1 {
        fma_pair_f32(&mut low[1..nh], &high[..nh - 1], &high[1..nh], -D97);
    }
    if nl > nh {
        low[nh] += -D97 * (high[nh - 1] + high[nh - 1]);
    }

    if nh > 1 {
        fma_pair_f32(&mut high[..nh - 1], &low[..nh - 1], &low[1..nh], -G97);
    }
    high[nh - 1] += -G97 * (low[nh - 1] + low[nh.min(nl - 1)]);

    low[0] += -B97 * (high[0] + high[0]);
    if nh > 1 {
        fma_pair_f32(&mut low[1..nh], &high[..nh - 1], &high[1..nh], -B97);
    }
    if nl > nh {
        low[nh] += -B97 * (high[nh - 1] + high[nh - 1]);
    }

    if nh > 1 {
        fma_pair_f32(&mut high[..nh - 1], &low[..nh - 1], &low[1..nh], -A97);
    }
    high[nh - 1] += -A97 * (low[nh - 1] + low[nh.min(nl - 1)]);

    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 4 <= nh {
            let lv = vld1q_f32(low.as_ptr().add(i));
            let hv = vld1q_f32(high.as_ptr().add(i));
            vst2q_f32(data.as_mut_ptr().add(i * 2), float32x4x2_t(lv, hv));
            i += 4;
        }
    }
    while i < nh {
        data[i * 2] = low[i];
        data[i * 2 + 1] = high[i];
        i += 1;
    }
    if nl > nh {
        data[nh * 2] = low[nh];
    }
}

/// Row-major SIMD vertical forward `CDF 9/7` on top-left `(w, h)` of `buf` (stride `stride`).
pub fn dwt97_2d_vertical_forward(buf: &mut [f32], stride: usize, w: usize, h: usize, tmp: &mut [f32]) {
    if h <= 1 {
        return;
    }
    let nl = h.div_ceil(2);
    let nh = h / 2;
    let (low, high) = tmp[..h * w].split_at_mut(nl * w);

    for i in 0..nl {
        low[i * w..(i + 1) * w].copy_from_slice(&buf[(i * 2) * stride..(i * 2) * stride + w]);
    }
    for i in 0..nh {
        high[i * w..(i + 1) * w].copy_from_slice(&buf[(i * 2 + 1) * stride..(i * 2 + 1) * stride + w]);
    }

    for i in 0..nh {
        let r_idx = (i + 1).min(nl - 1);
        fma_pair_f32(&mut high[i * w..(i + 1) * w], &low[i * w..(i + 1) * w], &low[r_idx * w..(r_idx + 1) * w], A97);
    }
    for i in 0..nl {
        let l_idx = i.saturating_sub(1);
        let r_idx = i.min(nh - 1);
        fma_pair_f32(&mut low[i * w..(i + 1) * w], &high[l_idx * w..(l_idx + 1) * w], &high[r_idx * w..(r_idx + 1) * w], B97);
    }
    for i in 0..nh {
        let r_idx = (i + 1).min(nl - 1);
        fma_pair_f32(&mut high[i * w..(i + 1) * w], &low[i * w..(i + 1) * w], &low[r_idx * w..(r_idx + 1) * w], G97);
    }
    for i in 0..nl {
        let l_idx = i.saturating_sub(1);
        let r_idx = i.min(nh - 1);
        fma_pair_f32(&mut low[i * w..(i + 1) * w], &high[l_idx * w..(l_idx + 1) * w], &high[r_idx * w..(r_idx + 1) * w], D97);
    }

    scale_slice_f32(low, INV_K97);
    scale_slice_f32(high, K97);

    for i in 0..nl {
        buf[i * stride..i * stride + w].copy_from_slice(&low[i * w..(i + 1) * w]);
    }
    for i in 0..nh {
        buf[(nl + i) * stride..(nl + i) * stride + w].copy_from_slice(&high[i * w..(i + 1) * w]);
    }
}

/// Row-major SIMD vertical inverse `CDF 9/7` on top-left `(w, h)` of `buf` (stride `stride`).
pub fn dwt97_2d_vertical_inverse(buf: &mut [f32], stride: usize, w: usize, h: usize, tmp: &mut [f32]) {
    if h <= 1 {
        return;
    }
    let nl = h.div_ceil(2);
    let nh = h / 2;
    let (low, high) = tmp[..h * w].split_at_mut(nl * w);

    for i in 0..nl {
        low[i * w..(i + 1) * w].copy_from_slice(&buf[i * stride..i * stride + w]);
    }
    for i in 0..nh {
        high[i * w..(i + 1) * w].copy_from_slice(&buf[(nl + i) * stride..(nl + i) * stride + w]);
    }

    scale_slice_f32(low, K97);
    scale_slice_f32(high, INV_K97);

    for i in 0..nl {
        let l_idx = i.saturating_sub(1);
        let r_idx = i.min(nh - 1);
        fma_pair_f32(&mut low[i * w..(i + 1) * w], &high[l_idx * w..(l_idx + 1) * w], &high[r_idx * w..(r_idx + 1) * w], -D97);
    }
    for i in 0..nh {
        let r_idx = (i + 1).min(nl - 1);
        fma_pair_f32(&mut high[i * w..(i + 1) * w], &low[i * w..(i + 1) * w], &low[r_idx * w..(r_idx + 1) * w], -G97);
    }
    for i in 0..nl {
        let l_idx = i.saturating_sub(1);
        let r_idx = i.min(nh - 1);
        fma_pair_f32(&mut low[i * w..(i + 1) * w], &high[l_idx * w..(l_idx + 1) * w], &high[r_idx * w..(r_idx + 1) * w], -B97);
    }
    for i in 0..nh {
        let r_idx = (i + 1).min(nl - 1);
        fma_pair_f32(&mut high[i * w..(i + 1) * w], &low[i * w..(i + 1) * w], &low[r_idx * w..(r_idx + 1) * w], -A97);
    }

    for i in 0..nl {
        buf[(i * 2) * stride..(i * 2) * stride + w].copy_from_slice(&low[i * w..(i + 1) * w]);
    }
    for i in 0..nh {
        buf[(i * 2 + 1) * stride..(i * 2 + 1) * stride + w].copy_from_slice(&high[i * w..(i + 1) * w]);
    }
}

// ============================================================================
// 3. CDF 5/3 Bit-Exact Integer SIMD (`i32`)
// ============================================================================

/// Bit-exact SIMD `dst[x] -= (l[x] + r[x]) >> 1` (or `+=` when `inv = true`).
#[inline]
fn lift53_predict_row(dst: &mut [i32], l: &[i32], r: &[i32], inv: bool) {
    let w = dst.len();
    let mut x = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while x + 4 <= w {
            let d = vld1q_s32(dst.as_ptr().add(x));
            let s = vshrq_n_s32::<1>(vaddq_s32(vld1q_s32(l.as_ptr().add(x)), vld1q_s32(r.as_ptr().add(x))));
            vst1q_s32(dst.as_mut_ptr().add(x), if inv { vaddq_s32(d, s) } else { vsubq_s32(d, s) });
            x += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            x = lift53_predict_row_avx2(dst, l, r, inv);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        while x + 4 <= w {
            let d = v128_load(dst.as_ptr().add(x).cast());
            let s = i32x4_shr(i32x4_add(v128_load(l.as_ptr().add(x).cast()), v128_load(r.as_ptr().add(x).cast())), 1);
            let res = if inv { i32x4_add(d, s) } else { i32x4_sub(d, s) };
            v128_store(dst.as_mut_ptr().add(x).cast(), res);
            x += 4;
        }
    }

    if inv {
        while x < w {
            dst[x] += (l[x] + r[x]) >> 1;
            x += 1;
        }
    } else {
        while x < w {
            dst[x] -= (l[x] + r[x]) >> 1;
            x += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lift53_predict_row_avx2(dst: &mut [i32], l: &[i32], r: &[i32], inv: bool) -> usize {
    use core::arch::x86_64::*;
    let w = dst.len();
    let mut x = 0usize;
    while x + 8 <= w {
        let d = _mm256_loadu_si256(dst.as_ptr().add(x).cast());
        let s = _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_loadu_si256(l.as_ptr().add(x).cast()), _mm256_loadu_si256(r.as_ptr().add(x).cast())),
            1,
        );
        let res = if inv { _mm256_add_epi32(d, s) } else { _mm256_sub_epi32(d, s) };
        _mm256_storeu_si256(dst.as_mut_ptr().add(x).cast(), res);
        x += 8;
    }
    x
}

/// Bit-exact SIMD `dst[x] += (l[x] + r[x] + 2) >> 2` (or `-=` when `inv = true`).
#[inline]
fn lift53_update_row(dst: &mut [i32], l: &[i32], r: &[i32], inv: bool) {
    let w = dst.len();
    let mut x = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v2 = vdupq_n_s32(2);
        while x + 4 <= w {
            let d = vld1q_s32(dst.as_ptr().add(x));
            let sum = vaddq_s32(vaddq_s32(vld1q_s32(l.as_ptr().add(x)), vld1q_s32(r.as_ptr().add(x))), v2);
            let s = vshrq_n_s32::<2>(sum);
            vst1q_s32(dst.as_mut_ptr().add(x), if inv { vsubq_s32(d, s) } else { vaddq_s32(d, s) });
            x += 4;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            x = lift53_update_row_avx2(dst, l, r, inv);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let v2 = i32x4_splat(2);
        while x + 4 <= w {
            let d = v128_load(dst.as_ptr().add(x).cast());
            let sum = i32x4_add(i32x4_add(v128_load(l.as_ptr().add(x).cast()), v128_load(r.as_ptr().add(x).cast())), v2);
            let s = i32x4_shr(sum, 2);
            let res = if inv { i32x4_sub(d, s) } else { i32x4_add(d, s) };
            v128_store(dst.as_mut_ptr().add(x).cast(), res);
            x += 4;
        }
    }

    if inv {
        while x < w {
            dst[x] -= (l[x] + r[x] + 2) >> 2;
            x += 1;
        }
    } else {
        while x < w {
            dst[x] += (l[x] + r[x] + 2) >> 2;
            x += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lift53_update_row_avx2(dst: &mut [i32], l: &[i32], r: &[i32], inv: bool) -> usize {
    use core::arch::x86_64::*;
    let w = dst.len();
    let v2 = _mm256_set1_epi32(2);
    let mut x = 0usize;
    while x + 8 <= w {
        let d = _mm256_loadu_si256(dst.as_ptr().add(x).cast());
        let sum = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_loadu_si256(l.as_ptr().add(x).cast()), _mm256_loadu_si256(r.as_ptr().add(x).cast())),
            v2,
        );
        let s = _mm256_srai_epi32(sum, 2);
        let res = if inv { _mm256_sub_epi32(d, s) } else { _mm256_add_epi32(d, s) };
        _mm256_storeu_si256(dst.as_mut_ptr().add(x).cast(), res);
        x += 8;
    }
    x
}

/// Bit-exact SIMD 1D forward `CDF 5/3` lifting on a contiguous row `data[0..n]`.
#[inline]
pub fn dwt53_1d_forward_simd(data: &mut [i32], scratch: &mut [i32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);

    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 4 <= nh {
            let pair = vld2q_s32(data.as_ptr().add(i * 2));
            vst1q_s32(low.as_mut_ptr().add(i), pair.0);
            vst1q_s32(high.as_mut_ptr().add(i), pair.1);
            i += 4;
        }
    }
    while i < nh {
        low[i] = data[i * 2];
        high[i] = data[i * 2 + 1];
        i += 1;
    }
    if nl > nh {
        low[nh] = data[nh * 2];
    }

    if nh > 1 {
        lift53_predict_row(&mut high[..nh - 1], &low[..nh - 1], &low[1..nh], false);
    }
    high[nh - 1] -= (low[nh - 1] + low[nh.min(nl - 1)]) >> 1;

    low[0] += (high[0] + high[0] + 2) >> 2;
    if nh > 1 {
        lift53_update_row(&mut low[1..nh], &high[..nh - 1], &high[1..nh], false);
    }
    if nl > nh {
        low[nh] += (high[nh - 1] + high[nh - 1] + 2) >> 2;
    }

    data[..nl].copy_from_slice(low);
    data[nl..n].copy_from_slice(high);
}

/// Bit-exact SIMD 1D inverse `CDF 5/3` lifting on a contiguous row `data[0..n]`.
#[inline]
pub fn dwt53_1d_inverse_simd(data: &mut [i32], scratch: &mut [i32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);

    low.copy_from_slice(&data[..nl]);
    high.copy_from_slice(&data[nl..n]);

    low[0] -= (high[0] + high[0] + 2) >> 2;
    if nh > 1 {
        lift53_update_row(&mut low[1..nh], &high[..nh - 1], &high[1..nh], true);
    }
    if nl > nh {
        low[nh] -= (high[nh - 1] + high[nh - 1] + 2) >> 2;
    }

    if nh > 1 {
        lift53_predict_row(&mut high[..nh - 1], &low[..nh - 1], &low[1..nh], true);
    }
    high[nh - 1] += (low[nh - 1] + low[nh.min(nl - 1)]) >> 1;

    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 4 <= nh {
            let lv = vld1q_s32(low.as_ptr().add(i));
            let hv = vld1q_s32(high.as_ptr().add(i));
            vst2q_s32(data.as_mut_ptr().add(i * 2), int32x4x2_t(lv, hv));
            i += 4;
        }
    }
    while i < nh {
        data[i * 2] = low[i];
        data[i * 2 + 1] = high[i];
        i += 1;
    }
    if nl > nh {
        data[nh * 2] = low[nh];
    }
}

/// Bit-exact row-major SIMD vertical forward `CDF 5/3` lifting.
pub fn dwt53_2d_vertical_forward(buf: &mut [i32], stride: usize, w: usize, h: usize, tmp: &mut [i32]) {
    if h <= 1 {
        return;
    }
    let nl = h.div_ceil(2);
    let nh = h / 2;
    let (low, high) = tmp[..h * w].split_at_mut(nl * w);

    for i in 0..nl {
        low[i * w..(i + 1) * w].copy_from_slice(&buf[(i * 2) * stride..(i * 2) * stride + w]);
    }
    for i in 0..nh {
        high[i * w..(i + 1) * w].copy_from_slice(&buf[(i * 2 + 1) * stride..(i * 2 + 1) * stride + w]);
    }
    for i in 0..nh {
        let r_idx = (i + 1).min(nl - 1);
        lift53_predict_row(&mut high[i * w..(i + 1) * w], &low[i * w..(i + 1) * w], &low[r_idx * w..(r_idx + 1) * w], false);
    }
    for i in 0..nl {
        let l_idx = i.saturating_sub(1);
        let r_idx = i.min(nh - 1);
        lift53_update_row(&mut low[i * w..(i + 1) * w], &high[l_idx * w..(l_idx + 1) * w], &high[r_idx * w..(r_idx + 1) * w], false);
    }
    for i in 0..nl {
        buf[i * stride..i * stride + w].copy_from_slice(&low[i * w..(i + 1) * w]);
    }
    for i in 0..nh {
        buf[(nl + i) * stride..(nl + i) * stride + w].copy_from_slice(&high[i * w..(i + 1) * w]);
    }
}

/// Bit-exact row-major SIMD vertical inverse `CDF 5/3` lifting.
pub fn dwt53_2d_vertical_inverse(buf: &mut [i32], stride: usize, w: usize, h: usize, tmp: &mut [i32]) {
    if h <= 1 {
        return;
    }
    let nl = h.div_ceil(2);
    let nh = h / 2;
    let (low, high) = tmp[..h * w].split_at_mut(nl * w);

    for i in 0..nl {
        low[i * w..(i + 1) * w].copy_from_slice(&buf[i * stride..i * stride + w]);
    }
    for i in 0..nh {
        high[i * w..(i + 1) * w].copy_from_slice(&buf[(nl + i) * stride..(nl + i) * stride + w]);
    }
    for i in 0..nl {
        let l_idx = i.saturating_sub(1);
        let r_idx = i.min(nh - 1);
        lift53_update_row(&mut low[i * w..(i + 1) * w], &high[l_idx * w..(l_idx + 1) * w], &high[r_idx * w..(r_idx + 1) * w], true);
    }
    for i in 0..nh {
        let r_idx = (i + 1).min(nl - 1);
        lift53_predict_row(&mut high[i * w..(i + 1) * w], &low[i * w..(i + 1) * w], &low[r_idx * w..(r_idx + 1) * w], true);
    }
    for i in 0..nl {
        buf[(i * 2) * stride..(i * 2) * stride + w].copy_from_slice(&low[i * w..(i + 1) * w]);
    }
    for i in 0..nh {
        buf[(i * 2 + 1) * stride..(i * 2 + 1) * stride + w].copy_from_slice(&high[i * w..(i + 1) * w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavelet::{dwt53_1d_forward, dwt53_1d_inverse, dwt97_1d_forward, dwt97_1d_inverse};

    #[test]
    fn simd_cdf53_1d_matches_scalar_bit_for_bit() {
        for n in [1, 2, 3, 7, 16, 17, 64, 65, 255, 256] {
            let src: Vec<i32> = (0..n).map(|i| ((i as i32 * 53 - 700) % 511) - 255).collect();
            let mut a = src.clone();
            let mut b = src.clone();
            let mut sa = vec![0i32; n];
            let mut sb = vec![0i32; n];
            dwt53_1d_forward(&mut a, &mut sa);
            dwt53_1d_forward_simd(&mut b, &mut sb);
            assert_eq!(a, b, "dwt53 forward mismatch at n={n}");

            dwt53_1d_inverse(&mut a, &mut sa);
            dwt53_1d_inverse_simd(&mut b, &mut sb);
            assert_eq!(a, src, "scalar roundtrip failed at n={n}");
            assert_eq!(b, src, "simd roundtrip failed at n={n}");
        }
    }

    #[test]
    fn simd_cdf97_1d_matches_scalar_ieee_exact() {
        for n in [1, 2, 3, 7, 16, 17, 64, 65, 255, 256] {
            let src: Vec<f32> = (0..n).map(|i| (((i as i32 * 41 - 300) % 256) - 128) as f32).collect();
            let mut a = src.clone();
            let mut b = src.clone();
            let mut sa = vec![0.0f32; n];
            let mut sb = vec![0.0f32; n];
            dwt97_1d_forward(&mut a, &mut sa);
            dwt97_1d_forward_simd(&mut b, &mut sb);
            for i in 0..n {
                assert_eq!(a[i].to_bits(), b[i].to_bits(), "dwt97 fwd bit diff n={n} i={i}: {} vs {}", a[i], b[i]);
            }
            dwt97_1d_inverse(&mut a, &mut sa);
            dwt97_1d_inverse_simd(&mut b, &mut sb);
            for i in 0..n {
                assert!((a[i] - b[i]).abs() <= 1e-4, "dwt97 inv diff n={n} i={i}: {} vs {}", a[i], b[i]);
            }
        }
    }

    #[test]
    fn simd_f32_rounding_matches_rust_round_exact() {
        let vals = [
            -35000.0f32, -32768.0, -2.5, -1.5, -0.5, -0.49, 0.0, 0.49, 0.5, 1.5, 2.5, 127.6, 32767.0, 40000.0,
        ];
        let got16 = f32_to_i16_round_slice(&vals);
        let got32 = f32_to_i32_round_slice(&vals);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(got16[i], v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
            assert_eq!(got32[i], v.round() as i32);
        }
    }
}
