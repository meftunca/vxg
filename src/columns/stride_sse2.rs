#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn extract_stride_planes_sse2(
    body: &[u8],
    comb: &mut [u8],
    stride: usize,
    count: usize,
    words: usize,
    use_sub_green: bool,
    first_bytes: &mut [u8; 8],
    all_zero: &mut [bool; 8],
    any_nz_words_arr: &mut [u64; 8],
    has_any_nz_words: &mut bool,
) {
    let body_ptr = body.as_ptr();
    let comb_ptr = comb.as_mut_ptr();
    unsafe {
        use core::arch::x86_64::*;
        let zero = _mm_setzero_si128();
        if stride == 4 && count >= 16 {
            let track_nz = words <= 512;
            *has_any_nz_words = track_nz;
            let p0 = comb_ptr;
            let p1 = comb_ptr.add(count);
            let p2 = comb_ptr.add(2 * count);
            let p3 = comb_ptr.add(3 * count);
            let f0 = *body_ptr;
            let f1 = *body_ptr.add(1);
            let f2 = *body_ptr.add(2);
            let f3 = *body_ptr.add(3);
            *first_bytes = if use_sub_green {
                [f0.wrapping_sub(f1), f1, f2.wrapping_sub(f1), f3, 0, 0, 0, 0]
            } else {
                [f0, f1, f2, f3, 0, 0, 0, 0]
            };
                let mut prev0 = _mm_set1_epi8(first_bytes[0] as i8);
                let mut prev1 = _mm_set1_epi8(first_bytes[1] as i8);
                let mut prev2 = _mm_set1_epi8(first_bytes[2] as i8);
                let mut prev3 = _mm_set1_epi8(first_bytes[3] as i8);
                let mut or0 = zero;
                let mut or1 = zero;
                let mut or2 = zero;
                let mut or3 = zero;
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let src = body_ptr.add(idx * 4);
                    let prev_u32 = body_ptr
                        .add(if idx > 0 { (idx - 1) * 4 } else { 0 })
                        .cast::<u32>()
                        .read_unaligned();
                    let pv = _mm_set1_epi32(prev_u32 as i32);
                    let v0 = _mm_loadu_si128(src.cast::<__m128i>());
                    let v1 = _mm_loadu_si128(src.add(16).cast::<__m128i>());
                    let v2 = _mm_loadu_si128(src.add(32).cast::<__m128i>());
                    let v3 = _mm_loadu_si128(src.add(48).cast::<__m128i>());
                    let diff = _mm_or_si128(
                        _mm_or_si128(_mm_xor_si128(v0, pv), _mm_xor_si128(v1, pv)),
                        _mm_or_si128(_mm_xor_si128(v2, pv), _mm_xor_si128(v3, pv)),
                    );
                    if _mm_movemask_epi8(_mm_cmpeq_epi8(diff, zero)) == 0xffff {
                        _mm_storeu_si128(p0.add(idx).cast::<__m128i>(), zero);
                        _mm_storeu_si128(p1.add(idx).cast::<__m128i>(), zero);
                        _mm_storeu_si128(p2.add(idx).cast::<__m128i>(), zero);
                        _mm_storeu_si128(p3.add(idx).cast::<__m128i>(), zero);
                        idx += 16;
                        continue;
                    }
                    if track_nz {
                        let w = idx >> 6;
                        any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                    }
                    let a0 = _mm_unpacklo_epi8(v0, v2);
                    let a1 = _mm_unpackhi_epi8(v0, v2);
                    let a2 = _mm_unpacklo_epi8(v1, v3);
                    let a3 = _mm_unpackhi_epi8(v1, v3);
                    let b0 = _mm_unpacklo_epi8(a0, a1);
                    let b1 = _mm_unpackhi_epi8(a0, a1);
                    let b2 = _mm_unpacklo_epi8(a2, a3);
                    let b3 = _mm_unpackhi_epi8(a2, a3);
                    let m0 = _mm_unpacklo_epi8(b0, b1);
                    let m1 = _mm_unpackhi_epi8(b0, b1);
                    let m2 = _mm_unpacklo_epi8(b2, b3);
                    let m3 = _mm_unpackhi_epi8(b2, b3);
                    let raw0 = _mm_unpacklo_epi32(m0, m2);
                    let c1 = _mm_unpackhi_epi32(m0, m2);
                    let raw2 = _mm_unpacklo_epi32(m1, m3);
                    let c3 = _mm_unpackhi_epi32(m1, m3);
                    let (c0, c2) = if use_sub_green {
                        (_mm_sub_epi8(raw0, c1), _mm_sub_epi8(raw2, c1))
                    } else {
                        (raw0, raw2)
                    };
                    let lag0 = _mm_or_si128(_mm_slli_si128::<1>(c0), _mm_srli_si128::<15>(prev0));
                    let lag1 = _mm_or_si128(_mm_slli_si128::<1>(c1), _mm_srli_si128::<15>(prev1));
                    let lag2 = _mm_or_si128(_mm_slli_si128::<1>(c2), _mm_srli_si128::<15>(prev2));
                    let lag3 = _mm_or_si128(_mm_slli_si128::<1>(c3), _mm_srli_si128::<15>(prev3));
                    prev0 = c0;
                    prev1 = c1;
                    prev2 = c2;
                    prev3 = c3;
                    let d0 = _mm_sub_epi8(c0, lag0);
                    let d1 = _mm_sub_epi8(c1, lag1);
                    let d2 = _mm_sub_epi8(c2, lag2);
                    let d3 = _mm_sub_epi8(c3, lag3);
                    or0 = _mm_or_si128(or0, d0);
                    or1 = _mm_or_si128(or1, d1);
                    or2 = _mm_or_si128(or2, d2);
                    or3 = _mm_or_si128(or3, d3);
                    _mm_storeu_si128(p0.add(idx).cast::<__m128i>(), d0);
                    _mm_storeu_si128(p1.add(idx).cast::<__m128i>(), d1);
                    _mm_storeu_si128(p2.add(idx).cast::<__m128i>(), d2);
                    _mm_storeu_si128(p3.add(idx).cast::<__m128i>(), d3);
                    idx += 16;
                }
                let mut s_or = [
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or0, zero)) != 0xffff) as u8,
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or1, zero)) != 0xffff) as u8,
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or2, zero)) != 0xffff) as u8,
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or3, zero)) != 0xffff) as u8,
                ];
                let ptrs = [p0, p1, p2, p3];
                while idx < count {
                    let g = *body_ptr.add(idx * 4 + 1);
                    let pg = *body_ptr.add((idx - 1) * 4 + 1);
                    let dg = g.wrapping_sub(pg);
                    for p in 0..4 {
                        let cur = *body_ptr.add(idx * 4 + p);
                        let prev = *body_ptr.add((idx - 1) * 4 + p);
                        let d = cur.wrapping_sub(prev);
                        let v = if use_sub_green && (p == 0 || p == 2) {
                            d.wrapping_sub(dg)
                        } else {
                            d
                        };
                        *ptrs[p].add(idx) = v;
                        if v != 0 && track_nz {
                            let w = idx >> 6;
                            any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                        }
                        s_or[p] |= v;
                    }
                    idx += 1;
                }
                *all_zero = [
                    s_or[0] == 0,
                    s_or[1] == 0,
                    s_or[2] == 0,
                    s_or[3] == 0,
                    false,
                    false,
                    false,
                    false,
                ];
            } else if stride == 3 && count >= 16 {
                let track_nz = words <= 512;
                *has_any_nz_words = track_nz;
                let p0 = comb_ptr;
                let p1 = comb_ptr.add(count);
                let p2 = comb_ptr.add(2 * count);
                let f0 = *body_ptr;
                let f1 = *body_ptr.add(1);
                let f2 = *body_ptr.add(2);
                *first_bytes = if use_sub_green {
                    [f0.wrapping_sub(f1), f1, f2.wrapping_sub(f1), 0, 0, 0, 0, 0]
                } else {
                    [f0, f1, f2, 0, 0, 0, 0, 0]
                };
                let mut prev0 = _mm_set1_epi8(first_bytes[0] as i8);
                let mut prev1 = _mm_set1_epi8(first_bytes[1] as i8);
                let mut prev2 = _mm_set1_epi8(first_bytes[2] as i8);
                let mut or0 = zero;
                let mut or1 = zero;
                let mut or2 = zero;
                let mut idx = 0usize;
                let mut t0 = [0u8; 16];
                let mut t1 = [0u8; 16];
                let mut t2 = [0u8; 16];
                while idx + 16 <= count {
                    let src = body_ptr.add(idx * 3);
                    for k in 0..16 {
                        t0[k] = *src.add(k * 3);
                        t1[k] = *src.add(k * 3 + 1);
                        t2[k] = *src.add(k * 3 + 2);
                    }
                    let raw0 = _mm_loadu_si128(t0.as_ptr().cast::<__m128i>());
                    let c1 = _mm_loadu_si128(t1.as_ptr().cast::<__m128i>());
                    let raw2 = _mm_loadu_si128(t2.as_ptr().cast::<__m128i>());
                    let (c0, c2) = if use_sub_green {
                        (_mm_sub_epi8(raw0, c1), _mm_sub_epi8(raw2, c1))
                    } else {
                        (raw0, raw2)
                    };
                    let lag0 = _mm_or_si128(_mm_slli_si128::<1>(c0), _mm_srli_si128::<15>(prev0));
                    let lag1 = _mm_or_si128(_mm_slli_si128::<1>(c1), _mm_srli_si128::<15>(prev1));
                    let lag2 = _mm_or_si128(_mm_slli_si128::<1>(c2), _mm_srli_si128::<15>(prev2));
                    prev0 = c0;
                    prev1 = c1;
                    prev2 = c2;
                    let d0 = _mm_sub_epi8(c0, lag0);
                    let d1 = _mm_sub_epi8(c1, lag1);
                    let d2 = _mm_sub_epi8(c2, lag2);
                    if track_nz
                        && _mm_movemask_epi8(_mm_cmpeq_epi8(
                            _mm_or_si128(_mm_or_si128(d0, d1), d2),
                            zero,
                        )) != 0xffff
                    {
                        let w = idx >> 6;
                        any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                    }
                    or0 = _mm_or_si128(or0, d0);
                    or1 = _mm_or_si128(or1, d1);
                    or2 = _mm_or_si128(or2, d2);
                    _mm_storeu_si128(p0.add(idx).cast::<__m128i>(), d0);
                    _mm_storeu_si128(p1.add(idx).cast::<__m128i>(), d1);
                    _mm_storeu_si128(p2.add(idx).cast::<__m128i>(), d2);
                    idx += 16;
                }
                let mut s_or = [
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or0, zero)) != 0xffff) as u8,
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or1, zero)) != 0xffff) as u8,
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or2, zero)) != 0xffff) as u8,
                ];
                let ptrs = [p0, p1, p2];
                while idx < count {
                    let g = *body_ptr.add(idx * 3 + 1);
                    let pg = *body_ptr.add((idx - 1) * 3 + 1);
                    let dg = g.wrapping_sub(pg);
                    for p in 0..3 {
                        let cur = *body_ptr.add(idx * 3 + p);
                        let prev = *body_ptr.add((idx - 1) * 3 + p);
                        let d = cur.wrapping_sub(prev);
                        let v = if use_sub_green && (p == 0 || p == 2) {
                            d.wrapping_sub(dg)
                        } else {
                            d
                        };
                        *ptrs[p].add(idx) = v;
                        if v != 0 && track_nz {
                            let w = idx >> 6;
                            any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                        }
                        s_or[p] |= v;
                    }
                    idx += 1;
                }
                *all_zero = [
                    s_or[0] == 0,
                    s_or[1] == 0,
                    s_or[2] == 0,
                    false,
                    false,
                    false,
                    false,
                    false,
                ];
            } else if stride == 2 && count >= 16 {
                let p0 = comb_ptr;
                let p1 = comb_ptr.add(count);
                first_bytes[0] = *body_ptr;
                first_bytes[1] = *body_ptr.add(1);
                let mut prev0 = _mm_set1_epi8(first_bytes[0] as i8);
                let mut prev1 = _mm_set1_epi8(first_bytes[1] as i8);
                let mut or0 = zero;
                let mut or1 = zero;
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let src = body_ptr.add(idx * 2);
                    let v0 = _mm_loadu_si128(src.cast::<__m128i>());
                    let v1 = _mm_loadu_si128(src.add(16).cast::<__m128i>());
                    let a0 = _mm_unpacklo_epi8(v0, v1);
                    let a1 = _mm_unpackhi_epi8(v0, v1);
                    let b0 = _mm_unpacklo_epi8(a0, a1);
                    let b1 = _mm_unpackhi_epi8(a0, a1);
                    let m0 = _mm_unpacklo_epi8(b0, b1);
                    let m1 = _mm_unpackhi_epi8(b0, b1);
                    let c0 = _mm_unpacklo_epi8(m0, m1);
                    let c1 = _mm_unpackhi_epi8(m0, m1);
                    let lag0 = _mm_or_si128(_mm_slli_si128::<1>(c0), _mm_srli_si128::<15>(prev0));
                    let lag1 = _mm_or_si128(_mm_slli_si128::<1>(c1), _mm_srli_si128::<15>(prev1));
                    prev0 = c0;
                    prev1 = c1;
                    let d0 = _mm_sub_epi8(c0, lag0);
                    let d1 = _mm_sub_epi8(c1, lag1);
                    or0 = _mm_or_si128(or0, d0);
                    or1 = _mm_or_si128(or1, d1);
                    _mm_storeu_si128(p0.add(idx).cast::<__m128i>(), d0);
                    _mm_storeu_si128(p1.add(idx).cast::<__m128i>(), d1);
                    idx += 16;
                }
                let mut s_or = [
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or0, zero)) != 0xffff) as u8,
                    (_mm_movemask_epi8(_mm_cmpeq_epi8(or1, zero)) != 0xffff) as u8,
                ];
                let ptrs = [p0, p1];
                while idx < count {
                    for p in 0..2 {
                        let d = (*body_ptr.add(idx * 2 + p))
                            .wrapping_sub(*body_ptr.add((idx - 1) * 2 + p));
                        *ptrs[p].add(idx) = d;
                        s_or[p] |= d;
                    }
                    idx += 1;
                }
                all_zero[0] = s_or[0] == 0;
                all_zero[1] = s_or[1] == 0;
            } else {
                for p in 0..stride {
                    first_bytes[p] = body[p];
                    let mut prev = first_bytes[p];
                    let mut s_or = 0u8;
                    let p_slice = &mut comb[p * count..(p + 1) * count];
                    p_slice[0] = 0;
                    for idx in 1..count {
                        let cur = body[idx * stride + p];
                        let d = cur.wrapping_sub(prev);
                        p_slice[idx] = d;
                        s_or |= d;
                        prev = cur;
                    }
                    all_zero[p] = s_or == 0;
                }
            }
        }
}

