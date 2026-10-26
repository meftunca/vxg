/// Fused inclusive prefix sum + optional Subtract-Green inverse + interleave:
/// `c_p[i] = first[p] + sum(planes[p][..=i])`, and if `sub_green`, `c_0 += c_1`, `c_2 += c_1`.
/// Each channel keeps its own SIMD-resident carry, so 3–4 independent dependency chains run
/// in parallel instead of one serial GPR<->SIMD round trip per 16 bytes.
fn prefix_interleave_planes(
    p_ptrs: [*const u8; 8],
    const_plane: [bool; 8],
    first: &[u8; 8],
    stride: usize,
    count: usize,
    sub_green: bool,
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= count * stride);
    if (0..stride).all(|p| const_plane[p]) {
        let g = first[1];
        let c0 = if sub_green { first[0].wrapping_add(g) } else { first[0] };
        let c1 = g;
        let c2 = if sub_green && stride >= 3 {
            first[2].wrapping_add(g)
        } else {
            first[2]
        };
        if stride == 4 {
            let px = u32::from_le_bytes([c0, c1, c2, first[3]]);
            let mut idx = 0usize;
            let dst_ptr = dst.as_mut_ptr().cast::<u32>();
            unsafe {
                while idx < count {
                    dst_ptr.add(idx).write_unaligned(px);
                    idx += 1;
                }
            }
        } else if stride == 2 {
            let px = u16::from_le_bytes([first[0], first[1]]);
            let mut idx = 0usize;
            let dst_ptr = dst.as_mut_ptr().cast::<u16>();
            unsafe {
                while idx < count {
                    dst_ptr.add(idx).write_unaligned(px);
                    idx += 1;
                }
            }
        } else if stride == 8 {
            let px = u64::from_le_bytes(*first);
            let mut idx = 0usize;
            let dst_ptr = dst.as_mut_ptr().cast::<u64>();
            unsafe {
                while idx < count {
                    dst_ptr.add(idx).write_unaligned(px);
                    idx += 1;
                }
            }
        } else {
            for idx in 0..count {
                dst[idx * 3] = c0;
                dst[idx * 3 + 1] = c1;
                dst[idx * 3 + 2] = c2;
            }
        }
        return;
    }

    let mut acc = *first;
    let mut idx = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let zero = vdupq_n_u8(0);
        macro_rules! scan {
            ($v:expr, $a:ident) => {{
                let mut s = $v;
                s = vaddq_u8(s, vextq_u8::<15>(zero, s));
                s = vaddq_u8(s, vextq_u8::<14>(zero, s));
                s = vaddq_u8(s, vextq_u8::<12>(zero, s));
                s = vaddq_u8(s, vextq_u8::<8>(zero, s));
                s = vaddq_u8(s, $a);
                $a = vdupq_laneq_u8::<15>(s);
                s
            }};
        }
        let dst_ptr = dst.as_mut_ptr();
        let mut a0 = vdupq_n_u8(acc[0]);
        let mut a1 = vdupq_n_u8(acc[1]);
        if stride == 4 {
            let mut a2 = vdupq_n_u8(acc[2]);
            let mut a3 = vdupq_n_u8(acc[3]);
            let (p0, p1, p2, p3) = (p_ptrs[0], p_ptrs[1], p_ptrs[2], p_ptrs[3]);
            if const_plane[3] && !const_plane[0] && !const_plane[1] && !const_plane[2] {
                if sub_green {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        vst4q_u8(
                            dst_ptr.add(idx * 4),
                            uint8x16x4_t(vaddq_u8(s0, s1), s1, vaddq_u8(s2, s1), a3),
                        );
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        vst4q_u8(dst_ptr.add(idx * 4), uint8x16x4_t(s0, s1, s2, a3));
                        idx += 16;
                    }
                }
            } else if !const_plane[0] && !const_plane[1] && !const_plane[2] && !const_plane[3] {
                if sub_green {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        let s3 = scan!(vld1q_u8(p3.add(idx)), a3);
                        vst4q_u8(
                            dst_ptr.add(idx * 4),
                            uint8x16x4_t(vaddq_u8(s0, s1), s1, vaddq_u8(s2, s1), s3),
                        );
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        let s3 = scan!(vld1q_u8(p3.add(idx)), a3);
                        vst4q_u8(dst_ptr.add(idx * 4), uint8x16x4_t(s0, s1, s2, s3));
                        idx += 16;
                    }
                }
            }
            acc[0] = vgetq_lane_u8::<0>(a0);
            acc[1] = vgetq_lane_u8::<0>(a1);
            acc[2] = vgetq_lane_u8::<0>(a2);
            acc[3] = vgetq_lane_u8::<0>(a3);
        } else if stride == 3 && !const_plane[0] && !const_plane[1] && !const_plane[2] {
            let mut a2 = vdupq_n_u8(acc[2]);
            let (p0, p1, p2) = (p_ptrs[0], p_ptrs[1], p_ptrs[2]);
            if sub_green {
                while idx + 16 <= count {
                    let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                    let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                    let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                    vst3q_u8(
                        dst_ptr.add(idx * 3),
                        uint8x16x3_t(vaddq_u8(s0, s1), s1, vaddq_u8(s2, s1)),
                    );
                    idx += 16;
                }
            } else {
                while idx + 16 <= count {
                    let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                    let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                    let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                    vst3q_u8(dst_ptr.add(idx * 3), uint8x16x3_t(s0, s1, s2));
                    idx += 16;
                }
            }
            acc[0] = vgetq_lane_u8::<0>(a0);
            acc[1] = vgetq_lane_u8::<0>(a1);
            acc[2] = vgetq_lane_u8::<0>(a2);
        } else if stride == 2 && !const_plane[0] && !const_plane[1] {
            let (p0, p1) = (p_ptrs[0], p_ptrs[1]);
            while idx + 16 <= count {
                let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                vst2q_u8(dst_ptr.add(idx * 2), uint8x16x2_t(s0, s1));
                idx += 16;
            }
            acc[0] = vgetq_lane_u8::<0>(a0);
            acc[1] = vgetq_lane_u8::<0>(a1);
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        macro_rules! scan {
            ($v:expr, $a:ident) => {{
                let mut s = $v;
                s = _mm_add_epi8(s, _mm_slli_si128::<1>(s));
                s = _mm_add_epi8(s, _mm_slli_si128::<2>(s));
                s = _mm_add_epi8(s, _mm_slli_si128::<4>(s));
                s = _mm_add_epi8(s, _mm_slli_si128::<8>(s));
                s = _mm_add_epi8(s, $a);
                let hi8 = _mm_unpackhi_epi8(s, s);
                let hi16 = _mm_unpackhi_epi16(hi8, hi8);
                $a = _mm_shuffle_epi32::<0xff>(hi16);
                s
            }};
        }
        macro_rules! store_interleave4 {
            ($dst:expr, $r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
                let t0 = _mm_unpacklo_epi8($r0, $r1);
                let t1 = _mm_unpackhi_epi8($r0, $r1);
                let t2 = _mm_unpacklo_epi8($r2, $r3);
                let t3 = _mm_unpackhi_epi8($r2, $r3);
                let d = $dst;
                _mm_storeu_si128(d.cast::<__m128i>(), _mm_unpacklo_epi16(t0, t2));
                _mm_storeu_si128(d.add(16).cast::<__m128i>(), _mm_unpackhi_epi16(t0, t2));
                _mm_storeu_si128(d.add(32).cast::<__m128i>(), _mm_unpacklo_epi16(t1, t3));
                _mm_storeu_si128(d.add(48).cast::<__m128i>(), _mm_unpackhi_epi16(t1, t3));
            }};
        }
        let dst_ptr = dst.as_mut_ptr();
        let mut a0 = _mm_set1_epi8(acc[0] as i8);
        let mut a1 = _mm_set1_epi8(acc[1] as i8);
        if stride == 4 {
            let mut a2 = _mm_set1_epi8(acc[2] as i8);
            let mut a3 = _mm_set1_epi8(acc[3] as i8);
            let (p0, p1, p2, p3) = (p_ptrs[0], p_ptrs[1], p_ptrs[2], p_ptrs[3]);
            let (c0, c1, c2, c3) = (const_plane[0], const_plane[1], const_plane[2], const_plane[3]);
            while idx + 16 <= count {
                let s0 = if c0 {
                    a0
                } else {
                    scan!(_mm_loadu_si128(p0.add(idx).cast::<__m128i>()), a0)
                };
                let s1 = if c1 {
                    a1
                } else {
                    scan!(_mm_loadu_si128(p1.add(idx).cast::<__m128i>()), a1)
                };
                let s2 = if c2 {
                    a2
                } else {
                    scan!(_mm_loadu_si128(p2.add(idx).cast::<__m128i>()), a2)
                };
                let s3 = if c3 {
                    a3
                } else {
                    scan!(_mm_loadu_si128(p3.add(idx).cast::<__m128i>()), a3)
                };
                if sub_green {
                    store_interleave4!(
                        dst_ptr.add(idx * 4),
                        _mm_add_epi8(s0, s1),
                        s1,
                        _mm_add_epi8(s2, s1),
                        s3
                    );
                } else {
                    store_interleave4!(dst_ptr.add(idx * 4), s0, s1, s2, s3);
                }
                idx += 16;
            }
            acc[0] = _mm_cvtsi128_si32(a0) as u8;
            acc[1] = _mm_cvtsi128_si32(a1) as u8;
            acc[2] = _mm_cvtsi128_si32(a2) as u8;
            acc[3] = _mm_cvtsi128_si32(a3) as u8;
        } else if stride == 3 {
            let mut a2 = _mm_set1_epi8(acc[2] as i8);
            let (p0, p1, p2) = (p_ptrs[0], p_ptrs[1], p_ptrs[2]);
            let (c0, c1, c2) = (const_plane[0], const_plane[1], const_plane[2]);
            let mut b0 = [0u8; 16];
            let mut b1 = [0u8; 16];
            let mut b2 = [0u8; 16];
            while idx + 16 <= count {
                let s0 = if c0 {
                    a0
                } else {
                    scan!(_mm_loadu_si128(p0.add(idx).cast::<__m128i>()), a0)
                };
                let s1 = if c1 {
                    a1
                } else {
                    scan!(_mm_loadu_si128(p1.add(idx).cast::<__m128i>()), a1)
                };
                let s2 = if c2 {
                    a2
                } else {
                    scan!(_mm_loadu_si128(p2.add(idx).cast::<__m128i>()), a2)
                };
                let (r0, r2) = if sub_green {
                    (_mm_add_epi8(s0, s1), _mm_add_epi8(s2, s1))
                } else {
                    (s0, s2)
                };
                _mm_storeu_si128(b0.as_mut_ptr().cast::<__m128i>(), r0);
                _mm_storeu_si128(b1.as_mut_ptr().cast::<__m128i>(), s1);
                _mm_storeu_si128(b2.as_mut_ptr().cast::<__m128i>(), r2);
                let out_p = dst_ptr.add(idx * 3);
                for k in 0..16 {
                    *out_p.add(k * 3) = b0[k];
                    *out_p.add(k * 3 + 1) = b1[k];
                    *out_p.add(k * 3 + 2) = b2[k];
                }
                idx += 16;
            }
            acc[0] = _mm_cvtsi128_si32(a0) as u8;
            acc[1] = _mm_cvtsi128_si32(a1) as u8;
            acc[2] = _mm_cvtsi128_si32(a2) as u8;
        } else if stride == 2 {
            let (p0, p1) = (p_ptrs[0], p_ptrs[1]);
            let (c0, c1) = (const_plane[0], const_plane[1]);
            while idx + 16 <= count {
                let s0 = if c0 {
                    a0
                } else {
                    scan!(_mm_loadu_si128(p0.add(idx).cast::<__m128i>()), a0)
                };
                let s1 = if c1 {
                    a1
                } else {
                    scan!(_mm_loadu_si128(p1.add(idx).cast::<__m128i>()), a1)
                };
                let d = dst_ptr.add(idx * 2);
                _mm_storeu_si128(d.cast::<__m128i>(), _mm_unpacklo_epi8(s0, s1));
                _mm_storeu_si128(d.add(16).cast::<__m128i>(), _mm_unpackhi_epi8(s0, s1));
                idx += 16;
            }
            acc[0] = _mm_cvtsi128_si32(a0) as u8;
            acc[1] = _mm_cvtsi128_si32(a1) as u8;
        }
    }
    while idx < count {
        for p in 0..stride {
            if !const_plane[p] {
                acc[p] = acc[p].wrapping_add(unsafe { *p_ptrs[p].add(idx) });
            }
        }
        if sub_green {
            let g = acc[1];
            dst[idx * stride] = acc[0].wrapping_add(g);
            dst[idx * stride + 1] = g;
            dst[idx * stride + 2] = acc[2].wrapping_add(g);
            if stride == 4 {
                dst[idx * stride + 3] = acc[3];
            }
        } else if stride == 8 {
            unsafe {
                dst.as_mut_ptr()
                    .cast::<u64>()
                    .add(idx)
                    .write_unaligned(u64::from_le_bytes(acc));
            }
        } else {
            for p in 0..stride {
                dst[idx * stride + p] = acc[p];
            }
        }
        idx += 1;
    }
}

