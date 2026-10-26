/// Pack a low-cardinality byte stream (e.g. Base64, hex/UUID, ASCII identifiers, XML/JSON literals)
/// using a 4-bit (<=16), 5-bit (<=32), or 6-bit (<=64) planar 16-lane bit-packed stream + raw escape stream.
fn pack_palette_stream(data: &[u8], out: &mut Vec<u8>) -> bool {
    let n = data.len();
    if n < 128 {
        return false;
    }
    // Fast 256-byte cardinality probe at start (and middle for large streams) to reject high-entropy binary data
    let s1 = n.min(256);
    let mut bmap = [0u64; 4];
    for &b in &data[..s1] {
        bmap[(b >> 6) as usize] |= 1u64 << (b & 63);
    }
    let sample_uniq =
        bmap[0].count_ones() + bmap[1].count_ones() + bmap[2].count_ones() + bmap[3].count_ones();
    if !(2..=76).contains(&sample_uniq) {
        return false;
    }
    if n >= 1024 {
        let mid = n / 2;
        let mut bmap2 = [0u64; 4];
        for &b in &data[mid..mid + 256] {
            bmap2[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        let mid_uniq = bmap2[0].count_ones()
            + bmap2[1].count_ones()
            + bmap2[2].count_ones()
            + bmap2[3].count_ones();
        if mid_uniq > 78 {
            return false;
        }
    }

    let mut counts = [0u32; 256];
    let chunks4 = n / 4;
    let ptr = data.as_ptr();
    for i in 0..chunks4 {
        unsafe {
            let p = ptr.add(i * 4);
            counts[*p as usize] += 1;
            counts[*p.add(1) as usize] += 1;
            counts[*p.add(2) as usize] += 1;
            counts[*p.add(3) as usize] += 1;
        }
    }
    for &b in &data[chunks4 * 4..] {
        counts[b as usize] += 1;
    }

    let mut syms = [(0u32, 0u8); 256];
    let mut uniq = 0usize;
    for b in 0..256 {
        let c = counts[b];
        if c > 0 {
            syms[uniq] = (c, b as u8);
            uniq += 1;
        }
    }
    if uniq < 2 {
        return false;
    }
    syms[..uniq].sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut best_bits = 0u8;
    let mut best_pal_len = 0usize;
    let mut best_escaped = 0usize;
    let mut best_total = usize::MAX;

    for bits in [4u8, 5, 6] {
        let cap = 1usize << bits;
        let (pal_len, escaped) = if uniq <= cap {
            (uniq, 0usize)
        } else {
            let keep = cap - 1;
            let mut top_sum = 0usize;
            for s in &syms[..keep] {
                top_sum += s.0 as usize;
            }
            (keep, n - top_sum)
        };
        if escaped * 8 > n {
            continue;
        }
        let packed_len = (n * (bits as usize)).div_ceil(8);
        let esc_hdr = if escaped > 0 { uleb_len(escaped) } else { 0 };
        let total = 2 + pal_len + esc_hdr + packed_len + escaped;
        if total < best_total {
            best_total = total;
            best_bits = bits;
            best_pal_len = pal_len;
            best_escaped = escaped;
        }
    }

    if best_bits == 0 || best_total + 16 + (n >> 5) >= n {
        return false;
    }

    let esc_code = if best_escaped > 0 {
        best_pal_len as u8
    } else {
        0
    };
    let mut map = [esc_code; 256];
    let mut pal = [0u8; 64];
    for (idx, &(_, sym)) in syms[..best_pal_len].iter().enumerate() {
        pal[idx] = sym;
        map[sym as usize] = idx as u8;
    }

    let packed_len = (n * (best_bits as usize)).div_ceil(8);
    out.clear();
    out.reserve(best_total + 16);
    out.push(best_bits | if best_escaped > 0 { 0x80 } else { 0 });
    out.push(best_pal_len as u8);
    out.extend_from_slice(&pal[..best_pal_len]);
    if best_escaped > 0 {
        put_uleb(out, best_escaped);
    }

    let packed_start = out.len();
    out.resize(packed_start + packed_len, 0);
    let chunks16 = n / 16;
    let rem = n % 16;

    const M4_64: u64 = 0x0f0f0f0f_0f0f0f0f;
    const M1_64: u64 = 0x01010101_01010101;
    const M2_64: u64 = 0x03030303_03030303;

    if best_escaped == 0 {
        let mut p_dst = unsafe { out.as_mut_ptr().add(packed_start) };
        for blk in 0..chunks16 {
            let base = blk * 16;
            let lo_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + i) } as usize]);
            let hi_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + 8 + i) } as usize]);
            let c_lo = u64::from_le_bytes(lo_arr);
            let c_hi = u64::from_le_bytes(hi_arr);
            let p4 = (c_lo & M4_64) | ((c_hi & M4_64) << 4);
            unsafe {
                p_dst.cast::<u64>().write_unaligned(p4.to_le());
                if best_bits == 4 {
                    p_dst = p_dst.add(8);
                } else if best_bits == 5 {
                    let q0 = ((c_lo >> 4) & M1_64) | (((c_hi >> 4) & M1_64) << 1);
                    let q1 = ((q0 as u32) & 0x03030303) | (((q0 >> 30) as u32) & 0x0c0c0c0c);
                    let q3 = ((q1 & 0x0f0f) | ((q1 >> 12) & 0xf0f0)) as u16;
                    p_dst.add(8).cast::<u16>().write_unaligned(q3.to_le());
                    p_dst = p_dst.add(10);
                } else {
                    let r0 = ((c_lo >> 4) & M2_64) | (((c_hi >> 4) & M2_64) << 2);
                    let p2 = ((r0 as u32) & 0x0f0f0f0f) | (((r0 >> 28) as u32) & 0xf0f0f0f0);
                    p_dst.add(8).cast::<u32>().write_unaligned(p2.to_le());
                    p_dst = p_dst.add(12);
                }
            }
        }
        if rem > 0 {
            let base = chunks16 * 16;
            let mut acc = 0u128;
            for i in 0..rem {
                let b = unsafe { *ptr.add(base + i) };
                acc |= (map[b as usize] as u128) << (i * (best_bits as usize));
            }
            let tail_bytes = (rem * (best_bits as usize)).div_ceil(8);
            let le = acc.to_le_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(le.as_ptr(), p_dst, tail_bytes);
            }
        }
    } else {
        let stride = (best_bits as usize) * 2;
        let esc_mask64 = M1_64 << best_bits;
        for blk in 0..chunks16 {
            let base = blk * 16;
            let lo_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + i) } as usize]);
            let hi_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + 8 + i) } as usize]);
            let c_lo = u64::from_le_bytes(lo_arr);
            let c_hi = u64::from_le_bytes(hi_arr);
            let m_lo = (c_lo + M1_64) & esc_mask64;
            let m_hi = (c_hi + M1_64) & esc_mask64;
            if (m_lo | m_hi) != 0 {
                let mut ml = m_lo;
                while ml != 0 {
                    let lane = (ml.trailing_zeros() >> 3) as usize;
                    out.push(unsafe { *ptr.add(base + lane) });
                    ml &= ml - 1;
                }
                let mut mh = m_hi;
                while mh != 0 {
                    let lane = 8 + ((mh.trailing_zeros() >> 3) as usize);
                    out.push(unsafe { *ptr.add(base + lane) });
                    mh &= mh - 1;
                }
            }
            let p_dst = unsafe { out.as_mut_ptr().add(packed_start + blk * stride) };
            let p4 = (c_lo & M4_64) | ((c_hi & M4_64) << 4);
            unsafe {
                p_dst.cast::<u64>().write_unaligned(p4.to_le());
                if best_bits == 5 {
                    let q0 = ((c_lo >> 4) & M1_64) | (((c_hi >> 4) & M1_64) << 1);
                    let q1 = ((q0 as u32) & 0x03030303) | (((q0 >> 30) as u32) & 0x0c0c0c0c);
                    let q3 = ((q1 & 0x0f0f) | ((q1 >> 12) & 0xf0f0)) as u16;
                    p_dst.add(8).cast::<u16>().write_unaligned(q3.to_le());
                } else if best_bits == 6 {
                    let r0 = ((c_lo >> 4) & M2_64) | (((c_hi >> 4) & M2_64) << 2);
                    let p2 = ((r0 as u32) & 0x0f0f0f0f) | (((r0 >> 28) as u32) & 0xf0f0f0f0);
                    p_dst.add(8).cast::<u32>().write_unaligned(p2.to_le());
                }
            }
        }
        if rem > 0 {
            let base = chunks16 * 16;
            let mut acc = 0u128;
            for i in 0..rem {
                let b = unsafe { *ptr.add(base + i) };
                let code = map[b as usize];
                if code == esc_code {
                    out.push(b);
                }
                acc |= (code as u128) << (i * (best_bits as usize));
            }
            let tail_bytes = (rem * (best_bits as usize)).div_ceil(8);
            let le = acc.to_le_bytes();
            unsafe {
                let p_dst = out.as_mut_ptr().add(packed_start + chunks16 * stride);
                std::ptr::copy_nonoverlapping(le.as_ptr(), p_dst, tail_bytes);
            }
        }
    }
    true
}

fn unpack_palette_stream(src: &[u8], out: &mut [u8]) -> Result<()> {
    let n = out.len();
    let mut cursor = Cursor::new(src);
    let hdr = cursor.byte()?;
    let bits = hdr & 0x7f;
    let has_esc = (hdr & 0x80) != 0;
    if !matches!(bits, 4..=6) {
        return Err(invalid("palette bits"));
    }
    let cap = 1usize << bits;
    let pal_len = cursor.byte()? as usize;
    if pal_len == 0 || (has_esc && pal_len != cap - 1) || (!has_esc && pal_len > cap) {
        return Err(invalid("palette length"));
    }
    let pal_slice = cursor.take(pal_len)?;
    let esc_len = if has_esc { cursor.uleb()? } else { 0 };
    let packed_len = (n * (bits as usize)).div_ceil(8);
    let packed = cursor.take(packed_len)?;
    let esc_bytes = cursor.take(esc_len)?;
    if !cursor.done() {
        return Err(invalid("palette trailing bytes"));
    }

    let mut pal = [0u8; 64];
    pal[..pal_len].copy_from_slice(pal_slice);

    let chunks16 = n / 16;
    let rem = n % 16;
    let mut p_src = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut esc_pos = 0usize;

    const M4_64: u64 = 0x0f0f0f0f_0f0f0f0f;
    const M1_64: u64 = 0x01010101_01010101;
    let esc_mask64: u64 = M1_64 << bits;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "aarch64")]
    let (tbl1, tbl2, tbl4) = unsafe {
        let v0 = vld1q_u8(pal.as_ptr());
        let v1 = vld1q_u8(pal.as_ptr().add(16));
        let v2 = vld1q_u8(pal.as_ptr().add(32));
        let v3 = vld1q_u8(pal.as_ptr().add(48));
        (v0, uint8x16x2_t(v0, v1), uint8x16x4_t(v0, v1, v2, v3))
    };

    for blk in 0..chunks16 {
        let p4 = u64::from_le(unsafe { p_src.cast::<u64>().read_unaligned() });
        let mut idx_lo = p4 & M4_64;
        let mut idx_hi = (p4 >> 4) & M4_64;
        if bits == 4 {
            p_src = unsafe { p_src.add(8) };
        } else if bits == 5 {
            let q3 = u16::from_le(unsafe { p_src.add(8).cast::<u16>().read_unaligned() }) as u32;
            p_src = unsafe { p_src.add(10) };
            let q1 = (q3 & 0x0f0f) | ((q3 & 0xf0f0) << 12);
            let q0 = ((q1 & 0x03030303) as u64) | (((q1 & 0x0c0c0c0c) as u64) << 30);
            idx_lo |= (q0 & M1_64) << 4;
            idx_hi |= (q0 & 0x02020202_02020202) << 3;
        } else {
            let p2 = u32::from_le(unsafe { p_src.add(8).cast::<u32>().read_unaligned() });
            p_src = unsafe { p_src.add(12) };
            let r0 = ((p2 & 0x0f0f0f0f) as u64) | (((p2 & 0xf0f0f0f0) as u64) << 28);
            idx_lo |= (r0 & 0x03030303_03030303) << 4;
            idx_hi |= (r0 & 0x0c0c0c0c_0c0c0c0c) << 2;
        }

        let dst = unsafe { out_ptr.add(blk * 16) };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let idx_v =
                vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(idx_lo), vcreate_u64(idx_hi)));
            let syms_v = if bits == 4 {
                vqtbl1q_u8(tbl1, idx_v)
            } else if bits == 5 {
                vqtbl2q_u8(tbl2, idx_v)
            } else {
                vqtbl4q_u8(tbl4, idx_v)
            };
            vst1q_u8(dst, syms_v);
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "ssse3"))]
        unsafe {
            use core::arch::x86_64::*;
            let v0 = _mm_loadu_si128(pal.as_ptr().cast::<__m128i>());
            let v1 = _mm_loadu_si128(pal.as_ptr().add(16).cast::<__m128i>());
            let v2 = _mm_loadu_si128(pal.as_ptr().add(32).cast::<__m128i>());
            let v3 = _mm_loadu_si128(pal.as_ptr().add(48).cast::<__m128i>());
            let idx_v = _mm_set_epi64x(idx_hi as i64, idx_lo as i64);
            let syms_v = if bits == 4 {
                _mm_shuffle_epi8(v0, idx_v)
            } else if bits == 5 {
                let s0 = _mm_shuffle_epi8(v0, idx_v);
                let s1 = _mm_shuffle_epi8(v1, idx_v);
                let m1 = _mm_slli_epi16(idx_v, 3);
                _mm_blendv_epi8(s0, s1, m1)
            } else {
                let m_lo = _mm_slli_epi16(idx_v, 3);
                let m_hi = _mm_slli_epi16(idx_v, 2);
                let s01 = _mm_blendv_epi8(
                    _mm_shuffle_epi8(v0, idx_v),
                    _mm_shuffle_epi8(v1, idx_v),
                    m_lo,
                );
                let s23 = _mm_blendv_epi8(
                    _mm_shuffle_epi8(v2, idx_v),
                    _mm_shuffle_epi8(v3, idx_v),
                    m_lo,
                );
                _mm_blendv_epi8(s01, s23, m_hi)
            };
            _mm_storeu_si128(dst.cast::<__m128i>(), syms_v);
        }
        #[cfg(not(any(
            target_arch = "aarch64",
            all(target_arch = "x86_64", target_feature = "ssse3")
        )))]
        unsafe {
            let lo_b = idx_lo.to_le_bytes();
            let hi_b = idx_hi.to_le_bytes();
            let pal_p = pal.as_ptr();
            let w_lo = (*pal_p.add((lo_b[0] as usize) & 63) as u64)
                | ((*pal_p.add((lo_b[1] as usize) & 63) as u64) << 8)
                | ((*pal_p.add((lo_b[2] as usize) & 63) as u64) << 16)
                | ((*pal_p.add((lo_b[3] as usize) & 63) as u64) << 24)
                | ((*pal_p.add((lo_b[4] as usize) & 63) as u64) << 32)
                | ((*pal_p.add((lo_b[5] as usize) & 63) as u64) << 40)
                | ((*pal_p.add((lo_b[6] as usize) & 63) as u64) << 48)
                | ((*pal_p.add((lo_b[7] as usize) & 63) as u64) << 56);
            let w_hi = (*pal_p.add((hi_b[0] as usize) & 63) as u64)
                | ((*pal_p.add((hi_b[1] as usize) & 63) as u64) << 8)
                | ((*pal_p.add((hi_b[2] as usize) & 63) as u64) << 16)
                | ((*pal_p.add((hi_b[3] as usize) & 63) as u64) << 24)
                | ((*pal_p.add((hi_b[4] as usize) & 63) as u64) << 32)
                | ((*pal_p.add((hi_b[5] as usize) & 63) as u64) << 40)
                | ((*pal_p.add((hi_b[6] as usize) & 63) as u64) << 48)
                | ((*pal_p.add((hi_b[7] as usize) & 63) as u64) << 56);
            dst.cast::<u64>().write_unaligned(w_lo.to_le());
            dst.add(8).cast::<u64>().write_unaligned(w_hi.to_le());
        }

        if has_esc {
            let mut m_lo = (idx_lo + M1_64) & esc_mask64;
            let mut m_hi = (idx_hi + M1_64) & esc_mask64;
            if (m_lo | m_hi) != 0 {
                while m_lo != 0 {
                    let lane = (m_lo.trailing_zeros() >> 3) as usize;
                    if esc_pos >= esc_len {
                        return Err(invalid("palette escape underflow"));
                    }
                    unsafe {
                        *dst.add(lane) = *esc_bytes.get_unchecked(esc_pos);
                    }
                    esc_pos += 1;
                    m_lo &= m_lo - 1;
                }
                while m_hi != 0 {
                    let lane = 8 + ((m_hi.trailing_zeros() >> 3) as usize);
                    if esc_pos >= esc_len {
                        return Err(invalid("palette escape underflow"));
                    }
                    unsafe {
                        *dst.add(lane) = *esc_bytes.get_unchecked(esc_pos);
                    }
                    esc_pos += 1;
                    m_hi &= m_hi - 1;
                }
            }
        }
    }

    if rem > 0 {
        let tail_bytes = (rem * (bits as usize)).div_ceil(8);
        let mut le = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(p_src, le.as_mut_ptr(), tail_bytes);
        }
        let acc = u128::from_le_bytes(le);
        let mask = (1u128 << bits) - 1;
        let esc_code = pal_len as u128;
        let base = chunks16 * 16;
        for i in 0..rem {
            let code = (acc >> (i * (bits as usize))) & mask;
            if has_esc && code == esc_code {
                if esc_pos >= esc_len {
                    return Err(invalid("palette escape underflow"));
                }
                out[base + i] = esc_bytes[esc_pos];
                esc_pos += 1;
            } else if (code as usize) < pal_len {
                out[base + i] = pal[code as usize];
            } else {
                return Err(invalid("palette symbol out of range"));
            }
        }
    }
    if esc_pos != esc_len {
        return Err(invalid("palette unused escapes"));
    }
    Ok(())
}

