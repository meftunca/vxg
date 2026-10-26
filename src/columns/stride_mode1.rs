fn unpack_stride_mode1_into(
    cursor: &mut Cursor<'_>,
    out: &mut [u8],
    prefix_len: usize,
    stride: usize,
    count: usize,
    tail_len: usize,
) -> Result<()> {
    let words = count.div_ceil(64);
    let super_words = words.div_ceil(64);
    let super_pad_bits = super_words * 64 - words;
    let pad_bits = words * 64 - count;
    let plane_cap = words * 64;

    STRIDE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        for p in 0..stride {
            if scratch.planes[p].len() < plane_cap {
                scratch.planes[p].resize(plane_cap, 0);
            }
        }

        for p in 0..stride {
            let first_byte = cursor.byte()?;
            let nz_masks_len = cursor.uleb()?;
            if nz_masks_len > words {
                return Err(invalid("stride nonzero mask count"));
            }
            let nz_len = cursor.uleb()?;
            if nz_len >= count || nz_len < nz_masks_len {
                return Err(invalid("stride nonzero count"));
            }
            let super_slice = cursor.take(super_words * 8)?;
            let mask_slice = cursor.take(nz_masks_len * 8)?;
            let nz_slice = cursor.take(nz_len)?;
            let plane_ptr = scratch.planes[p].as_mut_ptr();
            let super_ptr = super_slice.as_ptr().cast::<u64>();
            let mask_ptr = mask_slice.as_ptr().cast::<u64>();
            let nz_ptr = nz_slice.as_ptr();
            let mut nz_mask_idx = 0usize;
            let mut nz_idx = 0usize;
            let mut acc = first_byte;

            for sw in 0..super_words {
                let sm = u64::from_le(unsafe { super_ptr.add(sw).read_unaligned() });
                if sw + 1 == super_words
                    && super_pad_bits > 0
                    && (sm >> (64 - super_pad_bits)) != 0
                {
                    return Err(invalid("stride supermask padding bits"));
                }
                let w_start = sw * 64;
                let w_end = (w_start + 64).min(words);
                let w_count = w_end - w_start;
                if sm == 0 {
                    unsafe {
                        std::ptr::write_bytes(plane_ptr.add(w_start * 64), acc, w_count * 64);
                    }
                    continue;
                }
                let sm_pop = sm.count_ones() as usize;
                if nz_mask_idx + sm_pop > nz_masks_len {
                    return Err(invalid("stride nonzero mask underflow"));
                }
                let mut bw = 0usize;
                while bw < w_count {
                    let rem_sm = sm >> bw;
                    if rem_sm == 0 {
                        unsafe {
                            std::ptr::write_bytes(
                                plane_ptr.add((w_start + bw) * 64),
                                acc,
                                (w_count - bw) * 64,
                            );
                        }
                        break;
                    }
                    let z_words = (rem_sm.trailing_zeros() as usize).min(w_count - bw);
                    if z_words > 0 {
                        unsafe {
                            std::ptr::write_bytes(
                                plane_ptr.add((w_start + bw) * 64),
                                acc,
                                z_words * 64,
                            );
                        }
                        bw += z_words;
                        if bw >= w_count {
                            break;
                        }
                    }
                    let w = w_start + bw;
                    let base = w * 64;
                    let mut m = u64::from_le(unsafe { mask_ptr.add(nz_mask_idx).read_unaligned() });
                    nz_mask_idx += 1;
                    if m == 0 {
                        return Err(invalid("stride canonical zero mask"));
                    }
                    if w == 0 && (m & 1) != 0 {
                        return Err(invalid("stride first byte bit"));
                    }
                    if w + 1 == words && pad_bits > 0 && (m >> (64 - pad_bits)) != 0 {
                        return Err(invalid("stride padding bits"));
                    }
                    let pop = m.count_ones() as usize;
                    if nz_idx + pop > nz_len {
                        return Err(invalid("stride nonzero underflow"));
                    }
                    if pop == 1 {
                        let bit = m.trailing_zeros() as usize;
                        let v = unsafe { *nz_ptr.add(nz_idx) };
                        if v == 0 {
                            return Err(invalid("stride canonical zero residual"));
                        }
                        nz_idx += 1;
                        unsafe {
                            if bit > 0 {
                                std::ptr::write_bytes(plane_ptr.add(base), acc, bit);
                            }
                            acc = acc.wrapping_add(v);
                            std::ptr::write_bytes(plane_ptr.add(base + bit), acc, 64 - bit);
                        }
                        bw += 1;
                        continue;
                    }
                    unsafe {
                        std::ptr::write_bytes(plane_ptr.add(base), 0, 64);
                    }
                    while m != 0 {
                        let bit = m.trailing_zeros() as usize;
                        m &= m - 1;
                        unsafe {
                            let v = *nz_ptr.add(nz_idx);
                            if v == 0 {
                                return Err(invalid("stride canonical zero residual"));
                            }
                            *plane_ptr.add(base + bit) = v;
                        }
                        nz_idx += 1;
                    }
                    // Fused 64-byte inclusive prefix sum:
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use core::arch::aarch64::*;
                        let zero = vdupq_n_u8(0);
                        for v_idx in 0..4 {
                            let vp = plane_ptr.add(base + v_idx * 16);
                            let mut s = vld1q_u8(vp);
                            s = vaddq_u8(s, vextq_u8::<15>(zero, s));
                            s = vaddq_u8(s, vextq_u8::<14>(zero, s));
                            s = vaddq_u8(s, vextq_u8::<12>(zero, s));
                            s = vaddq_u8(s, vextq_u8::<8>(zero, s));
                            s = vaddq_u8(s, vdupq_n_u8(acc));
                            acc = vgetq_lane_u8::<15>(s);
                            vst1q_u8(vp, s);
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    unsafe {
                        for b in 0..64 {
                            let bp = plane_ptr.add(base + b);
                            acc = acc.wrapping_add(*bp);
                            *bp = acc;
                        }
                    }
                    bw += 1;
                }
            }
            if nz_mask_idx != nz_masks_len || nz_idx != nz_len {
                return Err(invalid("stride nonzero consumption"));
            }
        }

        let tail = cursor.take(tail_len)?;
        if !cursor.done() {
            return Err(invalid("stride trailing bytes"));
        }

        let dst_body = &mut out[prefix_len..prefix_len + count * stride];
        #[allow(unused_variables)]
        let dst_ptr = dst_body.as_mut_ptr();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            if stride == 4 {
                let p0 = scratch.planes[0].as_ptr();
                let p1 = scratch.planes[1].as_ptr();
                let p2 = scratch.planes[2].as_ptr();
                let p3 = scratch.planes[3].as_ptr();
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let v = uint8x16x4_t(
                        vld1q_u8(p0.add(idx)),
                        vld1q_u8(p1.add(idx)),
                        vld1q_u8(p2.add(idx)),
                        vld1q_u8(p3.add(idx)),
                    );
                    vst4q_u8(dst_ptr.add(idx * 4), v);
                    idx += 16;
                }
                while idx < count {
                    *dst_ptr.add(idx * 4) = *p0.add(idx);
                    *dst_ptr.add(idx * 4 + 1) = *p1.add(idx);
                    *dst_ptr.add(idx * 4 + 2) = *p2.add(idx);
                    *dst_ptr.add(idx * 4 + 3) = *p3.add(idx);
                    idx += 1;
                }
            } else if stride == 2 {
                let p0 = scratch.planes[0].as_ptr();
                let p1 = scratch.planes[1].as_ptr();
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let v = uint8x16x2_t(vld1q_u8(p0.add(idx)), vld1q_u8(p1.add(idx)));
                    vst2q_u8(dst_ptr.add(idx * 2), v);
                    idx += 16;
                }
                while idx < count {
                    *dst_ptr.add(idx * 2) = *p0.add(idx);
                    *dst_ptr.add(idx * 2 + 1) = *p1.add(idx);
                    idx += 1;
                }
            } else if stride == 8 {
                for idx in 0..count {
                    for p in 0..8 {
                        *dst_ptr.add(idx * 8 + p) = *scratch.planes[p].as_ptr().add(idx);
                    }
                }
            } else {
                let p0 = scratch.planes[0].as_ptr();
                let p1 = scratch.planes[1].as_ptr();
                let p2 = scratch.planes[2].as_ptr();
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let v = uint8x16x3_t(
                        vld1q_u8(p0.add(idx)),
                        vld1q_u8(p1.add(idx)),
                        vld1q_u8(p2.add(idx)),
                    );
                    vst3q_u8(dst_ptr.add(idx * 3), v);
                    idx += 16;
                }
                while idx < count {
                    *dst_ptr.add(idx * 3) = *p0.add(idx);
                    *dst_ptr.add(idx * 3 + 1) = *p1.add(idx);
                    *dst_ptr.add(idx * 3 + 2) = *p2.add(idx);
                    idx += 1;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for idx in 0..count {
                for p in 0..stride {
                    dst_body[idx * stride + p] = scratch.planes[p][idx];
                }
            }
        }
        out[prefix_len + count * stride..].copy_from_slice(tail);
        Ok(())
    })
}

