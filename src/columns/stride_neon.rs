#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn extract_stride_planes_neon(
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
        use core::arch::aarch64::*;
        let zero = vdupq_n_u8(0);
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
                let mut prev0 = vdupq_n_u8(first_bytes[0]);
                let mut prev1 = vdupq_n_u8(first_bytes[1]);
                let mut prev2 = vdupq_n_u8(first_bytes[2]);
                let mut prev3 = vdupq_n_u8(first_bytes[3]);
                let mut or0 = zero;
                let mut or1 = zero;
                let mut or2 = zero;
                let mut or3 = zero;
                let mut idx = 0usize;
                if use_sub_green {
                    while idx + 16 <= count {
                        let src_p = body_ptr.add(idx * 4);
                        let prev_u32 = body_ptr
                            .add(if idx > 0 { (idx - 1) * 4 } else { 0 })
                            .cast::<u32>()
                            .read_unaligned();
                        let pv = vdupq_n_u32(prev_u32);
                        let q0 = vld1q_u32(src_p.cast::<u32>());
                        let q1 = vld1q_u32(src_p.add(16).cast::<u32>());
                        let q2 = vld1q_u32(src_p.add(32).cast::<u32>());
                        let q3 = vld1q_u32(src_p.add(48).cast::<u32>());
                        let diff = vorrq_u32(
                            vorrq_u32(veorq_u32(q0, pv), veorq_u32(q1, pv)),
                            vorrq_u32(veorq_u32(q2, pv), veorq_u32(q3, pv)),
                        );
                        if vmaxvq_u32(diff) == 0 {
                            vst1q_u8(p0.add(idx), zero);
                            vst1q_u8(p1.add(idx), zero);
                            vst1q_u8(p2.add(idx), zero);
                            vst1q_u8(p3.add(idx), zero);
                            idx += 16;
                            continue;
                        }
                        if track_nz {
                            let w = idx >> 6;
                            any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                        }
                        let cur4 = vld4q_u8(src_p);
                        let c0 = vsubq_u8(cur4.0, cur4.1);
                        let c1 = cur4.1;
                        let c2 = vsubq_u8(cur4.2, cur4.1);
                        let c3 = cur4.3;
                        let lag0 = vextq_u8::<15>(prev0, c0);
                        let lag1 = vextq_u8::<15>(prev1, c1);
                        let lag2 = vextq_u8::<15>(prev2, c2);
                        let lag3 = vextq_u8::<15>(prev3, c3);
                        prev0 = c0;
                        prev1 = c1;
                        prev2 = c2;
                        prev3 = c3;
                        let d0 = vsubq_u8(c0, lag0);
                        let d1 = vsubq_u8(c1, lag1);
                        let d2 = vsubq_u8(c2, lag2);
                        let d3 = vsubq_u8(c3, lag3);
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        or3 = vorrq_u8(or3, d3);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        vst1q_u8(p3.add(idx), d3);
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let src_p = body_ptr.add(idx * 4);
                        let prev_u32 = body_ptr
                            .add(if idx > 0 { (idx - 1) * 4 } else { 0 })
                            .cast::<u32>()
                            .read_unaligned();
                        let pv = vdupq_n_u32(prev_u32);
                        let q0 = vld1q_u32(src_p.cast::<u32>());
                        let q1 = vld1q_u32(src_p.add(16).cast::<u32>());
                        let q2 = vld1q_u32(src_p.add(32).cast::<u32>());
                        let q3 = vld1q_u32(src_p.add(48).cast::<u32>());
                        let diff = vorrq_u32(
                            vorrq_u32(veorq_u32(q0, pv), veorq_u32(q1, pv)),
                            vorrq_u32(veorq_u32(q2, pv), veorq_u32(q3, pv)),
                        );
                        if vmaxvq_u32(diff) == 0 {
                            vst1q_u8(p0.add(idx), zero);
                            vst1q_u8(p1.add(idx), zero);
                            vst1q_u8(p2.add(idx), zero);
                            vst1q_u8(p3.add(idx), zero);
                            idx += 16;
                            continue;
                        }
                        if track_nz {
                            let w = idx >> 6;
                            any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                        }
                        let cur4 = vld4q_u8(src_p);
                        let lag0 = vextq_u8::<15>(prev0, cur4.0);
                        let lag1 = vextq_u8::<15>(prev1, cur4.1);
                        let lag2 = vextq_u8::<15>(prev2, cur4.2);
                        let lag3 = vextq_u8::<15>(prev3, cur4.3);
                        prev0 = cur4.0;
                        prev1 = cur4.1;
                        prev2 = cur4.2;
                        prev3 = cur4.3;
                        let d0 = vsubq_u8(cur4.0, lag0);
                        let d1 = vsubq_u8(cur4.1, lag1);
                        let d2 = vsubq_u8(cur4.2, lag2);
                        let d3 = vsubq_u8(cur4.3, lag3);
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        or3 = vorrq_u8(or3, d3);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        vst1q_u8(p3.add(idx), d3);
                        idx += 16;
                    }
                }
                let mut s_or = [
                    vmaxvq_u8(or0),
                    vmaxvq_u8(or1),
                    vmaxvq_u8(or2),
                    vmaxvq_u8(or3),
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
                let mut prev0 = vdupq_n_u8(first_bytes[0]);
                let mut prev1 = vdupq_n_u8(first_bytes[1]);
                let mut prev2 = vdupq_n_u8(first_bytes[2]);
                let mut or0 = zero;
                let mut or1 = zero;
                let mut or2 = zero;
                let mut idx = 0usize;
                if use_sub_green {
                    while idx + 16 <= count {
                        let src = body_ptr.add(idx * 3);
                        if idx > 0 {
                            let prev_src = src.sub(3);
                            let e0 = vceqq_u8(vld1q_u8(src), vld1q_u8(prev_src));
                            let e1 = vceqq_u8(vld1q_u8(src.add(16)), vld1q_u8(prev_src.add(16)));
                            let e2 = vceqq_u8(vld1q_u8(src.add(32)), vld1q_u8(prev_src.add(32)));
                            if vminvq_u8(vandq_u8(vandq_u8(e0, e1), e2)) == 0xff {
                                vst1q_u8(p0.add(idx), zero);
                                vst1q_u8(p1.add(idx), zero);
                                vst1q_u8(p2.add(idx), zero);
                                idx += 16;
                                continue;
                            }
                        }
                        let cur3 = vld3q_u8(src);
                        let c0 = vsubq_u8(cur3.0, cur3.1);
                        let c1 = cur3.1;
                        let c2 = vsubq_u8(cur3.2, cur3.1);
                        let lag0 = vextq_u8::<15>(prev0, c0);
                        let lag1 = vextq_u8::<15>(prev1, c1);
                        let lag2 = vextq_u8::<15>(prev2, c2);
                        prev0 = c0;
                        prev1 = c1;
                        prev2 = c2;
                        let d0 = vsubq_u8(c0, lag0);
                        let d1 = vsubq_u8(c1, lag1);
                        let d2 = vsubq_u8(c2, lag2);
                        if track_nz && vmaxvq_u8(vorrq_u8(vorrq_u8(d0, d1), d2)) != 0 {
                            let w = idx >> 6;
                            any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                        }
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let src = body_ptr.add(idx * 3);
                        if idx > 0 {
                            let prev_src = src.sub(3);
                            let e0 = vceqq_u8(vld1q_u8(src), vld1q_u8(prev_src));
                            let e1 = vceqq_u8(vld1q_u8(src.add(16)), vld1q_u8(prev_src.add(16)));
                            let e2 = vceqq_u8(vld1q_u8(src.add(32)), vld1q_u8(prev_src.add(32)));
                            if vminvq_u8(vandq_u8(vandq_u8(e0, e1), e2)) == 0xff {
                                vst1q_u8(p0.add(idx), zero);
                                vst1q_u8(p1.add(idx), zero);
                                vst1q_u8(p2.add(idx), zero);
                                idx += 16;
                                continue;
                            }
                        }
                        let cur3 = vld3q_u8(src);
                        let lag0 = vextq_u8::<15>(prev0, cur3.0);
                        let lag1 = vextq_u8::<15>(prev1, cur3.1);
                        let lag2 = vextq_u8::<15>(prev2, cur3.2);
                        prev0 = cur3.0;
                        prev1 = cur3.1;
                        prev2 = cur3.2;
                        let d0 = vsubq_u8(cur3.0, lag0);
                        let d1 = vsubq_u8(cur3.1, lag1);
                        let d2 = vsubq_u8(cur3.2, lag2);
                        if track_nz && vmaxvq_u8(vorrq_u8(vorrq_u8(d0, d1), d2)) != 0 {
                            let w = idx >> 6;
                            any_nz_words_arr[w >> 6] |= 1u64 << (w & 63);
                        }
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        idx += 16;
                    }
                }
                let mut s_or = [vmaxvq_u8(or0), vmaxvq_u8(or1), vmaxvq_u8(or2)];
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
                let mut prev0 = vdupq_n_u8(first_bytes[0]);
                let mut prev1 = vdupq_n_u8(first_bytes[1]);
                let mut or0 = vdupq_n_u8(0);
                let mut or1 = vdupq_n_u8(0);
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let cur2 = vld2q_u8(body_ptr.add(idx * 2));
                    let lag0 = vextq_u8::<15>(prev0, cur2.0);
                    let lag1 = vextq_u8::<15>(prev1, cur2.1);
                    prev0 = cur2.0;
                    prev1 = cur2.1;
                    let d0 = vsubq_u8(cur2.0, lag0);
                    let d1 = vsubq_u8(cur2.1, lag1);
                    or0 = vorrq_u8(or0, d0);
                    or1 = vorrq_u8(or1, d1);
                    vst1q_u8(p0.add(idx), d0);
                    vst1q_u8(p1.add(idx), d1);
                    idx += 16;
                }
                let mut s_or = [vmaxvq_u8(or0), vmaxvq_u8(or1)];
                let ptrs = [p0, p1];
                while idx < count {
                    for p in 0..2 {
                        let d = (*body_ptr.add(idx * 2 + p)).wrapping_sub(*body_ptr.add((idx - 1) * 2 + p));
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

