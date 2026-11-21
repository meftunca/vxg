//! Hardware SIMD acceleration (`ARMv8 NEON` on `aarch64` + runtime-detected `AVX2` on `x86_64`)
//! with 100% bit-exact scalar fallbacks (`wasm32`, generic targets) for:
//! - `RGB <-> YCoCg-R` planar color transforms
//! - `4:2:0` box downsample & `2×2` bilinear chroma upsample.

// ============================================================================
// 1. RGB <-> YCoCg-R Color Transform SIMD
// ============================================================================

#[inline]
pub fn split_rgb_ycocg_r(pixels: &[u8], y_p: &mut [i16], co_p: &mut [i16], cg_p: &mut [i16]) {
    let n = y_p.len();
    debug_assert_eq!(pixels.len(), n * 3);
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 16 <= n {
            let rgb = vld3q_u8(pixels.as_ptr().add(i * 3));
            let r_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.0)));
            let g_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.1)));
            let b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(rgb.2)));
            let co_lo = vsubq_s16(r_lo, b_lo);
            let t_lo = vaddq_s16(b_lo, vshrq_n_s16::<1>(co_lo));
            let cg_lo = vsubq_s16(g_lo, t_lo);
            let y_lo = vaddq_s16(t_lo, vshrq_n_s16::<1>(cg_lo));

            let r_hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb.0));
            let g_hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb.1));
            let b_hi = vreinterpretq_s16_u16(vmovl_high_u8(rgb.2));
            let co_hi = vsubq_s16(r_hi, b_hi);
            let t_hi = vaddq_s16(b_hi, vshrq_n_s16::<1>(co_hi));
            let cg_hi = vsubq_s16(g_hi, t_hi);
            let y_hi = vaddq_s16(t_hi, vshrq_n_s16::<1>(cg_hi));

            vst1q_s16(y_p.as_mut_ptr().add(i), y_lo);
            vst1q_s16(y_p.as_mut_ptr().add(i + 8), y_hi);
            vst1q_s16(co_p.as_mut_ptr().add(i), co_lo);
            vst1q_s16(co_p.as_mut_ptr().add(i + 8), co_hi);
            vst1q_s16(cg_p.as_mut_ptr().add(i), cg_lo);
            vst1q_s16(cg_p.as_mut_ptr().add(i + 8), cg_hi);
            i += 16;
        }
    }

    while i < n {
        let o = i * 3;
        let (r, g, b) = (pixels[o] as i16, pixels[o + 1] as i16, pixels[o + 2] as i16);
        let co = r - b;
        let t = b + (co >> 1);
        let cg = g - t;
        y_p[i] = t + (cg >> 1);
        co_p[i] = co;
        cg_p[i] = cg;
        i += 1;
    }
}

#[inline]
pub fn merge_ycocg_r_rgb(y_p: &[i16], co_p: &[i16], cg_p: &[i16], out: &mut [u8]) {
    let n = y_p.len();
    debug_assert_eq!(out.len(), n * 3);
    let mut i = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while i + 16 <= n {
            let y_lo = vld1q_s16(y_p.as_ptr().add(i));
            let co_lo = vld1q_s16(co_p.as_ptr().add(i));
            let cg_lo = vld1q_s16(cg_p.as_ptr().add(i));
            let t_lo = vsubq_s16(y_lo, vshrq_n_s16::<1>(cg_lo));
            let g_lo = vaddq_s16(cg_lo, t_lo);
            let b_lo = vsubq_s16(t_lo, vshrq_n_s16::<1>(co_lo));
            let r_lo = vaddq_s16(b_lo, co_lo);

            let y_hi = vld1q_s16(y_p.as_ptr().add(i + 8));
            let co_hi = vld1q_s16(co_p.as_ptr().add(i + 8));
            let cg_hi = vld1q_s16(cg_p.as_ptr().add(i + 8));
            let t_hi = vsubq_s16(y_hi, vshrq_n_s16::<1>(cg_hi));
            let g_hi = vaddq_s16(cg_hi, t_hi);
            let b_hi = vsubq_s16(t_hi, vshrq_n_s16::<1>(co_hi));
            let r_hi = vaddq_s16(b_hi, co_hi);

            let r8 = vcombine_u8(vqmovun_s16(r_lo), vqmovun_s16(r_hi));
            let g8 = vcombine_u8(vqmovun_s16(g_lo), vqmovun_s16(g_hi));
            let b8 = vcombine_u8(vqmovun_s16(b_lo), vqmovun_s16(b_hi));
            vst3q_u8(out.as_mut_ptr().add(i * 3), uint8x16x3_t(r8, g8, b8));
            i += 16;
        }
    }

    while i < n {
        let y = y_p[i];
        let co = co_p[i];
        let cg = cg_p[i];
        let t = y - (cg >> 1);
        let g = cg + t;
        let b = t - (co >> 1);
        let r = b + co;
        let o = i * 3;
        out[o] = r.clamp(0, 255) as u8;
        out[o + 1] = g.clamp(0, 255) as u8;
        out[o + 2] = b.clamp(0, 255) as u8;
        i += 1;
    }
}

// ============================================================================
// 2. 4:2:0 Chroma Downsample & Bilinear Upsample SIMD
// ============================================================================

#[inline]
pub fn downsample_2x2_row(row0: &[i16], row1: &[i16], out_row: &mut [i16]) {
    let w = row0.len();
    let cw = out_row.len();
    let mut cx = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let pairs_w = w / 2;
        while cx + 8 <= pairs_w {
            let p0 = vld2q_s16(row0.as_ptr().add(cx * 2));
            let p1 = vld2q_s16(row1.as_ptr().add(cx * 2));
            let s0 = vaddq_s16(p0.0, p0.1);
            let s1 = vaddq_s16(p1.0, p1.1);
            let avg = vrshrq_n_s16::<2>(vaddq_s16(s0, s1));
            vst1q_s16(out_row.as_mut_ptr().add(cx), avg);
            cx += 8;
        }
    }

    while cx < cw {
        let x0 = cx * 2;
        let x1 = (x0 + 1).min(w - 1);
        let sum = row0[x0] as i32 + row0[x1] as i32 + row1[x0] as i32 + row1[x1] as i32;
        out_row[cx] = ((sum + 2) >> 2) as i16;
        cx += 1;
    }
}

/// Bit-exact SIMD `2×2` bilinear upsample row: expands `r0/r1 (cw)` into `out_row (width)`.
#[inline]
pub fn upsample_2x2_bilinear_row(r0: &[i16], r1: &[i16], out_row: &mut [i16]) {
    let cw = r0.len();
    let width = out_row.len();
    if cw == 0 || width == 0 {
        return;
    }
    out_row[0] = ((r0[0] as i32 * 12 + r1[0] as i32 * 4 + 8) >> 4) as i16;
    if width == 1 {
        return;
    }
    let c1 = 1.min(cw - 1);
    out_row[1] = ((r0[0] as i32 * 9 + r0[c1] as i32 * 3 + r1[0] as i32 * 3 + r1[c1] as i32 + 8) >> 4) as i16;

    #[allow(unused_mut)]
    let mut cx = 1usize;
    #[allow(unused_variables)]
    let safe_end_cx = (cw.saturating_sub(1)).min(width / 2);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        while cx + 8 <= safe_end_cx {
            let r0_c = vld1q_s16(r0.as_ptr().add(cx));
            let r0_l = vld1q_s16(r0.as_ptr().add(cx - 1));
            let r0_r = vld1q_s16(r0.as_ptr().add(cx + 1));
            let r1_c = vld1q_s16(r1.as_ptr().add(cx));
            let r1_l = vld1q_s16(r1.as_ptr().add(cx - 1));
            let r1_r = vld1q_s16(r1.as_ptr().add(cx + 1));

            // Exact 32-bit widening accumulation — overflow-free for 8-bit, 10-bit & 12-bit HDR!
            let base_lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(r0_c), 9), vget_low_s16(r1_c), 3);
            let base_hi = vmlal_high_n_s16(vmull_high_n_s16(r0_c, 9), r1_c, 3);

            let ev_lo = vaddq_s32(vmlal_n_s16(base_lo, vget_low_s16(r0_l), 3), vmovl_s16(vget_low_s16(r1_l)));
            let ev_hi = vaddq_s32(vmlal_high_n_s16(base_hi, r0_l, 3), vmovl_high_s16(r1_l));
            let even = vcombine_s16(vrshrn_n_s32::<4>(ev_lo), vrshrn_n_s32::<4>(ev_hi));

            let od_lo = vaddq_s32(vmlal_n_s16(base_lo, vget_low_s16(r0_r), 3), vmovl_s16(vget_low_s16(r1_r)));
            let od_hi = vaddq_s32(vmlal_high_n_s16(base_hi, r0_r, 3), vmovl_high_s16(r1_r));
            let odd = vcombine_s16(vrshrn_n_s32::<4>(od_lo), vrshrn_n_s32::<4>(od_hi));

            vst2q_s16(out_row.as_mut_ptr().add(cx * 2), int16x8x2_t(even, odd));
            cx += 8;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            cx = upsample_2x2_bilinear_row_avx2(r0, r1, out_row, cx, safe_end_cx);
        }
    }

    let mut x = cx * 2;
    while x < width {
        let cx0 = (x / 2).min(cw - 1);
        let cx1 = if (x & 1) == 0 {
            cx0.saturating_sub(1)
        } else {
            (cx0 + 1).min(cw - 1)
        };
        let v = (r0[cx0] as i32) * 9 + (r0[cx1] as i32) * 3 + (r1[cx0] as i32) * 3 + (r1[cx1] as i32);
        out_row[x] = ((v + 8) >> 4) as i16;
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn upsample_2x2_bilinear_row_avx2(
    r0: &[i16],
    r1: &[i16],
    out_row: &mut [i16],
    mut cx: usize,
    safe_end_cx: usize,
) -> usize {
    use core::arch::x86_64::*;
    let v9 = _mm256_set1_epi32(9);
    let v3 = _mm256_set1_epi32(3);
    let v8 = _mm256_set1_epi32(8);
    // Exact 32-bit widening accumulation — overflow-free for 8-bit, 10-bit & 12-bit HDR!
    while cx + 8 <= safe_end_cx {
        let r0_c = _mm256_cvtepi16_epi32(_mm_loadu_si128(r0.as_ptr().add(cx).cast()));
        let r0_l = _mm256_cvtepi16_epi32(_mm_loadu_si128(r0.as_ptr().add(cx - 1).cast()));
        let r0_r = _mm256_cvtepi16_epi32(_mm_loadu_si128(r0.as_ptr().add(cx + 1).cast()));
        let r1_c = _mm256_cvtepi16_epi32(_mm_loadu_si128(r1.as_ptr().add(cx).cast()));
        let r1_l = _mm256_cvtepi16_epi32(_mm_loadu_si128(r1.as_ptr().add(cx - 1).cast()));
        let r1_r = _mm256_cvtepi16_epi32(_mm_loadu_si128(r1.as_ptr().add(cx + 1).cast()));

        let base = _mm256_add_epi32(_mm256_mullo_epi32(r0_c, v9), _mm256_mullo_epi32(r1_c, v3));
        let even32 = _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_add_epi32(base, _mm256_mullo_epi32(r0_l, v3)), _mm256_add_epi32(r1_l, v8)),
            4,
        );
        let odd32 = _mm256_srai_epi32(
            _mm256_add_epi32(_mm256_add_epi32(base, _mm256_mullo_epi32(r0_r, v3)), _mm256_add_epi32(r1_r, v8)),
            4,
        );
        // packs_epi32 operates per 128-bit lane:
        // lane0 = [e0,o0,e1,o1, e2,o2,e3,o3], lane1 = [e4,o4,e5,o5, e6,o6,e7,o7] — already exact!
        let lo32 = _mm256_unpacklo_epi32(even32, odd32);
        let hi32 = _mm256_unpackhi_epi32(even32, odd32);
        let packed = _mm256_packs_epi32(lo32, hi32);
        _mm256_storeu_si256(out_row.as_mut_ptr().add(cx * 2).cast(), packed);
        cx += 8;
    }
    cx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{rgb_to_ycocg_r, ycocg_r_to_rgb};

    #[test]
    fn simd_ycocg_r_and_420_match_scalar_bit_for_bit() {
        let n = 259; // tests both NEON 16-wide body and odd tail!
        let px: Vec<u8> = (0..n * 3).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let (mut y, mut co, mut cg) = (vec![0i16; n], vec![0i16; n], vec![0i16; n]);
        split_rgb_ycocg_r(&px, &mut y, &mut co, &mut cg);
        for i in 0..n {
            let (sy, sco, scg) = rgb_to_ycocg_r(px[i * 3], px[i * 3 + 1], px[i * 3 + 2]);
            assert_eq!((y[i], co[i], cg[i]), (sy, sco, scg));
        }
        let mut rec = vec![0u8; n * 3];
        merge_ycocg_r_rgb(&y, &co, &cg, &mut rec);
        assert_eq!(rec, px);

        let r0: Vec<i16> = (0..65).map(|i| ((i * 19 - 500) % 255) as i16).collect();
        let r1: Vec<i16> = (0..65).map(|i| ((i * 23 - 400) % 255) as i16).collect();
        let mut ds = vec![0i16; 33];
        downsample_2x2_row(&r0, &r1, &mut ds);
        for (cx, &got) in ds.iter().enumerate() {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(64);
            let exp = ((r0[x0] as i32 + r0[x1] as i32 + r1[x0] as i32 + r1[x1] as i32 + 2) >> 2) as i16;
            assert_eq!(got, exp);
        }
        let ds2: Vec<i16> = ds.iter().map(|&v| -v / 2 + 17).collect();
        let mut up = vec![0i16; 65];
        upsample_2x2_bilinear_row(&ds, &ds2, &mut up);
        for (x, &got) in up.iter().enumerate() {
            let cx0 = (x / 2).min(32);
            let cx1 = if (x & 1) == 0 { cx0.saturating_sub(1) } else { (cx0 + 1).min(32) };
            let exp = ((ds[cx0] as i32 * 9 + ds[cx1] as i32 * 3 + ds2[cx0] as i32 * 3 + ds2[cx1] as i32 + 8) >> 4) as i16;
            assert_eq!(got, exp, "upsample mismatch at x={x}");
        }
        // 12-bit HDR extreme range (±4095) must never overflow 16-bit intermediates
        let hdr_r0: Vec<i16> = (0..33).map(|i| if i % 2 == 0 { 4095 } else { -4095 }).collect();
        let hdr_r1: Vec<i16> = (0..33).map(|i| 4095 - (i as i16) * 200).collect();
        let mut hdr_up = vec![0i16; 65];
        upsample_2x2_bilinear_row(&hdr_r0, &hdr_r1, &mut hdr_up);
        for (x, &got) in hdr_up.iter().enumerate() {
            let cx0 = (x / 2).min(32);
            let cx1 = if (x & 1) == 0 { cx0.saturating_sub(1) } else { (cx0 + 1).min(32) };
            let exp = ((hdr_r0[cx0] as i32 * 9 + hdr_r0[cx1] as i32 * 3 + hdr_r1[cx0] as i32 * 3 + hdr_r1[cx1] as i32 + 8) >> 4) as i16;
            assert_eq!(got, exp, "12-bit HDR upsample mismatch at x={x}");
        }
        let (ey, eco, ecg) = (vec![300i16; 19], vec![250i16; 19], vec![-250i16; 19]);
        let mut out_e = vec![0u8; 19 * 3];
        merge_ycocg_r_rgb(&ey, &eco, &ecg, &mut out_e);
        for i in 0..19 {
            let (r, g, b) = ycocg_r_to_rgb(ey[i], eco[i], ecg[i]);
            assert_eq!((out_e[i * 3], out_e[i * 3 + 1], out_e[i * 3 + 2]), (r, g, b));
        }
    }
}
