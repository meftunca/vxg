fn unpack_stride_into(cursor: &mut Cursor<'_>, out: &mut [u8]) -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        unsafe fn unpack_stride_into_x86_v3(cursor: &mut Cursor<'_>, out: &mut [u8]) -> Result<()> {
            unpack_stride_into_impl(cursor, out)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
        {
            return unsafe { unpack_stride_into_x86_v3(cursor, out) };
        }
    }
    unpack_stride_into_impl(cursor, out)
}

#[inline(always)]
fn unpack_stride_into_impl(cursor: &mut Cursor<'_>, out: &mut [u8]) -> Result<()> {
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE || raw_len != out.len() {
        return Err(invalid("stride raw length"));
    }
    let prefix_len = cursor.uleb()?;
    if prefix_len > raw_len {
        return Err(invalid("stride prefix length"));
    }
    let prefix = cursor.take(prefix_len)?;
    out[..prefix_len].copy_from_slice(prefix);

    let stride = cursor.byte()? as usize;
    let mode = cursor.byte()?;
    if !matches!(stride, 2 | 3 | 4 | 8) || !matches!(mode, 1..=3) || (mode == 3 && !matches!(stride, 3 | 4)) {
        return Err(invalid("stride header"));
    }
    let body_len = raw_len - prefix_len;
    let count = body_len / stride;
    if count < 64 {
        return Err(invalid("stride count"));
    }
    let tail_len = body_len - count * stride;
    let words = count.div_ceil(64);
    let super_words = words.div_ceil(64);
    let super_pad_bits = super_words * 64 - words;
    let pad_bits = words * 64 - count;
    let plane_cap = words * 64;

    if mode == 2 || mode == 3 {
        let sub_green = mode == 3;
        if stride == 3 || stride == 4 {
            let saved_pos = cursor.pos;
            let mut firsts = [0u8; 8];
            let mut const_plane = [false; 4];
            let mut super_ptrs = [std::ptr::null::<u64>(); 4];
            let mut mask_ptrs = [std::ptr::null::<u64>(); 4];
            let mut nz_ptrs = [std::ptr::null::<u8>(); 4];
            let mut all_k24 = true;
            for p in 0..stride {
                let Ok(fb) = cursor.byte() else {
                    all_k24 = false;
                    break;
                };
                let Ok(kind) = cursor.byte() else {
                    all_k24 = false;
                    break;
                };
                firsts[p] = fb;
                if kind == 4 {
                    const_plane[p] = true;
                } else if kind == 2 {
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
                    let s_ptr = super_slice.as_ptr().cast::<u64>();
                    let m_ptr = mask_slice.as_ptr().cast::<u64>();
                    let mut nz_mask_idx = 0usize;
                    let mut nz_idx = 0usize;
                    for sw in 0..super_words {
                        let sm = u64::from_le(unsafe { s_ptr.add(sw).read_unaligned() });
                        if sw + 1 == super_words
                            && super_pad_bits > 0
                            && (sm >> (64 - super_pad_bits)) != 0
                        {
                            return Err(invalid("stride supermask padding bits"));
                        }
                        let mut rem_sm = sm;
                        while rem_sm != 0 {
                            let bw = rem_sm.trailing_zeros() as usize;
                            rem_sm &= rem_sm - 1;
                            if nz_mask_idx >= nz_masks_len {
                                return Err(invalid("stride nonzero mask underflow"));
                            }
                            let w = sw * 64 + bw;
                            let m = u64::from_le(unsafe { m_ptr.add(nz_mask_idx).read_unaligned() });
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
                            nz_idx += pop;
                        }
                    }
                    if nz_mask_idx != nz_masks_len || nz_idx != nz_len {
                        return Err(invalid("stride nonzero consumption"));
                    }
                    if find_byte(0, nz_slice).is_some() {
                        return Err(invalid("stride canonical zero residual"));
                    }
                    super_ptrs[p] = s_ptr;
                    mask_ptrs[p] = m_ptr;
                    nz_ptrs[p] = nz_slice.as_ptr();
                } else {
                    all_k24 = false;
                    break;
                }
            }
            if all_k24 {
                let tail = cursor.take(tail_len)?;
                if !cursor.done() {
                    return Err(invalid("stride trailing bytes"));
                }
                let px = &mut out[prefix_len..prefix_len + count * stride];
                let mut acc = firsts;
                let mut nz_mask_idx = [0usize; 4];
                let mut nz_idx = [0usize; 4];
                let mut run_start = 0usize;
                if stride == 4 {
                    let dst_u32 = px.as_mut_ptr().cast::<u32>();
                    let make_px4 = |a: &[u8; 8]| -> u32 {
                        let g = a[1];
                        u32::from_le_bytes([
                            if sub_green { a[0].wrapping_add(g) } else { a[0] },
                            g,
                            if sub_green { a[2].wrapping_add(g) } else { a[2] },
                            a[3],
                        ])
                    };
                    let mut cur_px32 = make_px4(&acc);
                    for sw in 0..super_words {
                        let sm0 = if const_plane[0] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[0].add(sw).read_unaligned() })
                        };
                        let sm1 = if const_plane[1] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[1].add(sw).read_unaligned() })
                        };
                        let sm2 = if const_plane[2] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[2].add(sw).read_unaligned() })
                        };
                        let sm3 = if const_plane[3] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[3].add(sw).read_unaligned() })
                        };
                        let mut sm_any = sm0 | sm1 | sm2 | sm3;
                        while sm_any != 0 {
                            let bw = sm_any.trailing_zeros() as usize;
                            sm_any &= sm_any - 1;
                            let w_bit = 1u64 << bw;
                            let m0 = if (sm0 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[0].add(nz_mask_idx[0]).read_unaligned()
                                });
                                nz_mask_idx[0] += 1;
                                v
                            } else {
                                0
                            };
                            let m1 = if (sm1 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[1].add(nz_mask_idx[1]).read_unaligned()
                                });
                                nz_mask_idx[1] += 1;
                                v
                            } else {
                                0
                            };
                            let m2 = if (sm2 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[2].add(nz_mask_idx[2]).read_unaligned()
                                });
                                nz_mask_idx[2] += 1;
                                v
                            } else {
                                0
                            };
                            let m3 = if (sm3 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[3].add(nz_mask_idx[3]).read_unaligned()
                                });
                                nz_mask_idx[3] += 1;
                                v
                            } else {
                                0
                            };
                            let px_base = (sw * 64 + bw) * 64;
                            let mut m_any = m0 | m1 | m2 | m3;
                            while m_any != 0 {
                                let bit = m_any.trailing_zeros() as usize;
                                m_any &= m_any - 1;
                                let px_idx = px_base + bit;
                                unsafe {
                                    fill_px4_run(dst_u32, run_start, px_idx, cur_px32);
                                }
                                let p_bit = 1u64 << bit;
                                if (m0 & p_bit) != 0 {
                                    acc[0] = acc[0].wrapping_add(unsafe { *nz_ptrs[0].add(nz_idx[0]) });
                                    nz_idx[0] += 1;
                                }
                                if (m1 & p_bit) != 0 {
                                    acc[1] = acc[1].wrapping_add(unsafe { *nz_ptrs[1].add(nz_idx[1]) });
                                    nz_idx[1] += 1;
                                }
                                if (m2 & p_bit) != 0 {
                                    acc[2] = acc[2].wrapping_add(unsafe { *nz_ptrs[2].add(nz_idx[2]) });
                                    nz_idx[2] += 1;
                                }
                                if (m3 & p_bit) != 0 {
                                    acc[3] = acc[3].wrapping_add(unsafe { *nz_ptrs[3].add(nz_idx[3]) });
                                    nz_idx[3] += 1;
                                }
                                cur_px32 = make_px4(&acc);
                                run_start = px_idx;
                            }
                        }
                    }
                    unsafe {
                        fill_px4_run(dst_u32, run_start, count, cur_px32);
                    }
                } else {
                    let dst_ptr = px.as_mut_ptr();
                    let make_px3 = |a: &[u8; 8]| -> (u8, u8, u8) {
                        let g = a[1];
                        (
                            if sub_green { a[0].wrapping_add(g) } else { a[0] },
                            g,
                            if sub_green { a[2].wrapping_add(g) } else { a[2] },
                        )
                    };
                    let (mut c0, mut c1, mut c2) = make_px3(&acc);
                    for sw in 0..super_words {
                        let sm0 = if const_plane[0] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[0].add(sw).read_unaligned() })
                        };
                        let sm1 = if const_plane[1] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[1].add(sw).read_unaligned() })
                        };
                        let sm2 = if const_plane[2] {
                            0
                        } else {
                            u64::from_le(unsafe { super_ptrs[2].add(sw).read_unaligned() })
                        };
                        let mut sm_any = sm0 | sm1 | sm2;
                        while sm_any != 0 {
                            let bw = sm_any.trailing_zeros() as usize;
                            sm_any &= sm_any - 1;
                            let w_bit = 1u64 << bw;
                            let m0 = if (sm0 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[0].add(nz_mask_idx[0]).read_unaligned()
                                });
                                nz_mask_idx[0] += 1;
                                v
                            } else {
                                0
                            };
                            let m1 = if (sm1 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[1].add(nz_mask_idx[1]).read_unaligned()
                                });
                                nz_mask_idx[1] += 1;
                                v
                            } else {
                                0
                            };
                            let m2 = if (sm2 & w_bit) != 0 {
                                let v = u64::from_le(unsafe {
                                    mask_ptrs[2].add(nz_mask_idx[2]).read_unaligned()
                                });
                                nz_mask_idx[2] += 1;
                                v
                            } else {
                                0
                            };
                            let px_base = (sw * 64 + bw) * 64;
                            let mut m_any = m0 | m1 | m2;
                            while m_any != 0 {
                                let bit = m_any.trailing_zeros() as usize;
                                m_any &= m_any - 1;
                                let px_idx = px_base + bit;
                                unsafe {
                                    fill_px3_run(dst_ptr, run_start, px_idx, c0, c1, c2);
                                }
                                let p_bit = 1u64 << bit;
                                if (m0 & p_bit) != 0 {
                                    acc[0] = acc[0].wrapping_add(unsafe { *nz_ptrs[0].add(nz_idx[0]) });
                                    nz_idx[0] += 1;
                                }
                                if (m1 & p_bit) != 0 {
                                    acc[1] = acc[1].wrapping_add(unsafe { *nz_ptrs[1].add(nz_idx[1]) });
                                    nz_idx[1] += 1;
                                }
                                if (m2 & p_bit) != 0 {
                                    acc[2] = acc[2].wrapping_add(unsafe { *nz_ptrs[2].add(nz_idx[2]) });
                                    nz_idx[2] += 1;
                                }
                                (c0, c1, c2) = make_px3(&acc);
                                run_start = px_idx;
                            }
                        }
                    }
                    unsafe {
                        fill_px3_run(dst_ptr, run_start, count, c0, c1, c2);
                    }
                }
                out[prefix_len + count * stride..].copy_from_slice(tail);
                return Ok(());
            }
            cursor.pos = saved_pos;
        }
        return STRIDE_PLANES2.with(|cell| {
            let mut planes = cell.borrow_mut();
            STRIDE_COMBINED.with(|comb_cell| {
                let mut combined = comb_cell.borrow_mut();
                let mut firsts = [0u8; 8];
                let mut const_plane = [false; 8];
                let mut p_ptrs = [std::ptr::null::<u8>(); 8];
                // `p_ptrs` keeps raw pointers into `combined`, so it must never reallocate while
                // they are live: size it for the whole stride up front (a later run used to grow
                // it and leave the earlier runs' pointers dangling on a cold thread).
                if combined.len() < stride * count {
                    combined.resize(stride * count, 0);
                }
                let mut p = 0usize;
                while p < stride {
                    firsts[p] = cursor.byte()?;
                    let kind = cursor.byte()?;
                    match kind {
                        4 => {
                            const_plane[p] = true;
                            p += 1;
                        }
                        3 => {
                            let run_len = cursor.byte()? as usize;
                            if run_len < 2 || p + run_len > stride {
                                return Err(invalid("stride combined run"));
                            }
                            for k in 1..run_len {
                                firsts[p + k] = cursor.byte()?;
                            }
                            let comb_len = run_len * count;
                            let len = cursor.uleb()?;
                            if len >= comb_len {
                                return Err(invalid("stride plane length"));
                            }
                            let mut sub = Cursor::new(cursor.take(len)?);
                            if sub.byte()? != FAST_LZ_PACKET {
                                return Err(invalid("stride plane packet"));
                            }
                            let req_end = (p + run_len) * count;
                            if combined.len() < req_end {
                                combined.resize(req_end, 0);
                            }
                            let dst = &mut combined[p * count..req_end];
                            unpack_fast_lz_into(&mut sub, dst)?;
                            let dst_ptr = dst.as_ptr();
                            for k in 0..run_len {
                                if dst[k * count] != 0 {
                                    return Err(invalid("stride plane first delta"));
                                }
                                p_ptrs[p + k] = unsafe { dst_ptr.add(k * count) };
                            }
                            p += run_len;
                        }
                        0 => {
                            if planes[p].len() < plane_cap {
                                planes[p].resize(plane_cap, 0);
                            }
                            let dst = &mut planes[p][..count];
                            dst.copy_from_slice(cursor.take(count)?);
                            if dst[0] != 0 {
                                return Err(invalid("stride plane first delta"));
                            }
                            p_ptrs[p] = planes[p].as_ptr();
                            p += 1;
                        }
                        1 => {
                            if planes[p].len() < plane_cap {
                                planes[p].resize(plane_cap, 0);
                            }
                            let dst = &mut planes[p][..count];
                            let len = cursor.uleb()?;
                            if len >= count {
                                return Err(invalid("stride plane length"));
                            }
                            let mut sub = Cursor::new(cursor.take(len)?);
                            if sub.byte()? != FAST_LZ_PACKET {
                                return Err(invalid("stride plane packet"));
                            }
                            unpack_fast_lz_into(&mut sub, dst)?;
                            if dst[0] != 0 {
                                return Err(invalid("stride plane first delta"));
                            }
                            p_ptrs[p] = planes[p].as_ptr();
                            p += 1;
                        }
                        2 => {
                            if planes[p].len() < plane_cap {
                                planes[p].resize(plane_cap, 0);
                            }
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
                            let plane_ptr = planes[p].as_mut_ptr();
                            let super_ptr = super_slice.as_ptr().cast::<u64>();
                            let mask_ptr = mask_slice.as_ptr().cast::<u64>();
                            let nz_ptr = nz_slice.as_ptr();
                            let mut nz_mask_idx = 0usize;
                            let mut nz_idx = 0usize;
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
                                        std::ptr::write_bytes(
                                            plane_ptr.add(w_start * 64),
                                            0,
                                            w_count * 64,
                                        );
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
                                                0,
                                                (w_count - bw) * 64,
                                            );
                                        }
                                        break;
                                    }
                                    let z_words =
                                        (rem_sm.trailing_zeros() as usize).min(w_count - bw);
                                    if z_words > 0 {
                                        unsafe {
                                            std::ptr::write_bytes(
                                                plane_ptr.add((w_start + bw) * 64),
                                                0,
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
                                    let mut m = u64::from_le(unsafe {
                                        mask_ptr.add(nz_mask_idx).read_unaligned()
                                    });
                                    nz_mask_idx += 1;
                                    if m == 0 {
                                        return Err(invalid("stride canonical zero mask"));
                                    }
                                    if w == 0 && (m & 1) != 0 {
                                        return Err(invalid("stride first byte bit"));
                                    }
                                    if w + 1 == words
                                        && pad_bits > 0
                                        && (m >> (64 - pad_bits)) != 0
                                    {
                                        return Err(invalid("stride padding bits"));
                                    }
                                    let pop = m.count_ones() as usize;
                                    if nz_idx + pop > nz_len {
                                        return Err(invalid("stride nonzero underflow"));
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
                                                return Err(invalid(
                                                    "stride canonical zero residual",
                                                ));
                                            }
                                            *plane_ptr.add(base + bit) = v;
                                        }
                                        nz_idx += 1;
                                    }
                                    bw += 1;
                                }
                            }
                            if nz_mask_idx != nz_masks_len || nz_idx != nz_len {
                                return Err(invalid("stride nonzero consumption"));
                            }
                            p_ptrs[p] = planes[p].as_ptr();
                            p += 1;
                        }
                        _ => return Err(invalid("stride plane kind")),
                    }
                }
                let tail = cursor.take(tail_len)?;
                if !cursor.done() {
                    return Err(invalid("stride trailing bytes"));
                }
                let px = &mut out[prefix_len..prefix_len + count * stride];
                prefix_interleave_planes(p_ptrs, const_plane, &firsts, stride, count, sub_green, px);
                out[prefix_len + count * stride..].copy_from_slice(tail);
                Ok(())
            })
        });
    }
    unpack_stride_mode1_into(cursor, out, prefix_len, stride, count, tail_len)
}

