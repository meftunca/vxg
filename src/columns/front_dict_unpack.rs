#[inline(always)]
fn read_front_dict_palette(cursor: &mut Cursor<'_>) -> Result<([u8; 32], u8, bool)> {
    let hdr = cursor.byte()?;
    let pal_mixed = (hdr & 0x80) != 0;
    let pal_len = hdr & 0x7f;
    if pal_len == 0 || pal_len > if pal_mixed { 16 } else { 32 } {
        return Err(invalid("column front dict palette length"));
    }
    let pal_slice = cursor.take(pal_len as usize)?;
    for w in pal_slice.windows(2) {
        if w[0] >= w[1] {
            return Err(invalid("column front dict palette ordering"));
        }
    }
    let mut pal = [0u8; 32];
    pal[..pal_len as usize].copy_from_slice(pal_slice);
    Ok((pal, pal_len, pal_mixed))
}

#[inline(always)]
fn unpack_nibble_bytes_into(
    src: &[u8],
    char_len: usize,
    pal: &[u8; 32],
    pal_len: u8,
    dst: *mut u8,
) -> Result<()> {
    if pal_len > 16 {
        let mut j = 0usize;
        let mut s_idx = 0usize;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let v_pal = uint8x16x2_t(vld1q_u8(pal.as_ptr()), vld1q_u8(pal.as_ptr().add(16)));
            while j + 8 <= char_len && s_idx + 8 <= src.len() {
                let w = u64::from_le(src.as_ptr().add(s_idx).cast::<u64>().read_unaligned());
                let idx_u64 = (w & 0x1f)
                    | (((w >> 5) & 0x1f) << 8)
                    | (((w >> 10) & 0x1f) << 16)
                    | (((w >> 15) & 0x1f) << 24)
                    | (((w >> 20) & 0x1f) << 32)
                    | (((w >> 25) & 0x1f) << 40)
                    | (((w >> 30) & 0x1f) << 48)
                    | (((w >> 35) & 0x1f) << 56);
                let v_idx = vreinterpret_u8_u64(vdup_n_u64(idx_u64));
                if pal_len < 32 && vmaxv_u8(v_idx) >= pal_len {
                    return Err(invalid("column front dict 5bit out of bounds"));
                }
                let ascii = vqtbl2_u8(v_pal, v_idx);
                vst1_u8(dst.add(j), ascii);
                j += 8;
                s_idx += 5;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            let pal_p = pal.as_ptr();
            while j + 8 <= char_len && s_idx + 8 <= src.len() {
                let w = u64::from_le(src.as_ptr().add(s_idx).cast::<u64>().read_unaligned());
                let i0 = (w & 0x1f) as usize;
                let i1 = ((w >> 5) & 0x1f) as usize;
                let i2 = ((w >> 10) & 0x1f) as usize;
                let i3 = ((w >> 15) & 0x1f) as usize;
                let i4 = ((w >> 20) & 0x1f) as usize;
                let i5 = ((w >> 25) & 0x1f) as usize;
                let i6 = ((w >> 30) & 0x1f) as usize;
                let i7 = ((w >> 35) & 0x1f) as usize;
                if pal_len < 32
                    && (i0.max(i1).max(i2).max(i3).max(i4).max(i5).max(i6).max(i7))
                        >= pal_len as usize
                {
                    return Err(invalid("column front dict 5bit out of bounds"));
                }
                let out_w = (*pal_p.add(i0) as u64)
                    | ((*pal_p.add(i1) as u64) << 8)
                    | ((*pal_p.add(i2) as u64) << 16)
                    | ((*pal_p.add(i3) as u64) << 24)
                    | ((*pal_p.add(i4) as u64) << 32)
                    | ((*pal_p.add(i5) as u64) << 40)
                    | ((*pal_p.add(i6) as u64) << 48)
                    | ((*pal_p.add(i7) as u64) << 56);
                dst.add(j).cast::<u64>().write_unaligned(out_w.to_le());
                j += 8;
                s_idx += 5;
            }
        }
        let mut acc = 0u64;
        let mut bits = 0u32;
        while j < char_len {
            if bits < 5 {
                if s_idx + 4 <= src.len() {
                    let w = u32::from_le(unsafe {
                        src.as_ptr().add(s_idx).cast::<u32>().read_unaligned()
                    }) as u64;
                    acc |= w << bits;
                    bits += 32;
                    s_idx += 4;
                } else {
                    while s_idx < src.len() {
                        acc |= (src[s_idx] as u64) << bits;
                        bits += 8;
                        s_idx += 1;
                    }
                }
            }
            let idx = (acc & 0x1f) as u8;
            if idx >= pal_len {
                return Err(invalid("column front dict 5bit out of bounds"));
            }
            unsafe {
                *dst.add(j) = *pal.get_unchecked(idx as usize);
            }
            acc >>= 5;
            bits -= 5;
            j += 1;
        }
        if acc != 0 {
            return Err(invalid("column front dict 5bit padding"));
        }
        return Ok(());
    }
    let mut j = 0usize;
    let mut b_idx = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v_pal = vld1q_u8(pal.as_ptr());
        while j + 16 <= char_len {
            let v8 = vld1_u8(src.as_ptr().add(b_idx));
            let hi = vshr_n_u8(v8, 4);
            let lo = vand_u8(v8, vdup_n_u8(0x0f));
            let zipped = vcombine_u8(vzip1_u8(hi, lo), vzip2_u8(hi, lo));
            if pal_len < 16 && vmaxvq_u8(zipped) >= pal_len {
                return Err(invalid("column front dict nibble out of bounds"));
            }
            let ascii = vqtbl1q_u8(v_pal, zipped);
            vst1q_u8(dst.add(j), ascii);
            j += 16;
            b_idx += 8;
        }
        if j + 8 <= char_len {
            let w4 = src.as_ptr().add(b_idx).cast::<u32>().read_unaligned();
            let v8 = vreinterpret_u8_u32(vdup_n_u32(w4));
            let hi = vshr_n_u8(v8, 4);
            let lo = vand_u8(v8, vdup_n_u8(0x0f));
            let zipped = vzip1_u8(hi, lo);
            if pal_len < 16 && vmaxv_u8(zipped) >= pal_len {
                return Err(invalid("column front dict nibble out of bounds"));
            }
            let ascii = vqtbl1_u8(v_pal, zipped);
            vst1_u8(dst.add(j), ascii);
            j += 8;
            b_idx += 4;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        let pal_p = pal.as_ptr();
        while j + 8 <= char_len {
            let b0 = *src.get_unchecked(b_idx);
            let b1 = *src.get_unchecked(b_idx + 1);
            let b2 = *src.get_unchecked(b_idx + 2);
            let b3 = *src.get_unchecked(b_idx + 3);
            let i0 = (b0 >> 4) as usize;
            let i1 = (b0 & 0x0f) as usize;
            let i2 = (b1 >> 4) as usize;
            let i3 = (b1 & 0x0f) as usize;
            let i4 = (b2 >> 4) as usize;
            let i5 = (b2 & 0x0f) as usize;
            let i6 = (b3 >> 4) as usize;
            let i7 = (b3 & 0x0f) as usize;
            if pal_len < 16
                && (i0.max(i1).max(i2).max(i3).max(i4).max(i5).max(i6).max(i7))
                    >= pal_len as usize
            {
                return Err(invalid("column front dict nibble out of bounds"));
            }
            let out_w = (*pal_p.add(i0) as u64)
                | ((*pal_p.add(i1) as u64) << 8)
                | ((*pal_p.add(i2) as u64) << 16)
                | ((*pal_p.add(i3) as u64) << 24)
                | ((*pal_p.add(i4) as u64) << 32)
                | ((*pal_p.add(i5) as u64) << 40)
                | ((*pal_p.add(i6) as u64) << 48)
                | ((*pal_p.add(i7) as u64) << 56);
            dst.add(j).cast::<u64>().write_unaligned(out_w.to_le());
            j += 8;
            b_idx += 4;
        }
    }
    while j + 1 < char_len {
        let b = unsafe { *src.get_unchecked(b_idx) };
        let hi = b >> 4;
        let lo = b & 0x0f;
        if hi >= pal_len || lo >= pal_len {
            return Err(invalid("column front dict nibble out of bounds"));
        }
        unsafe {
            *dst.add(j) = *pal.get_unchecked(hi as usize);
            *dst.add(j + 1) = *pal.get_unchecked(lo as usize);
        }
        j += 2;
        b_idx += 1;
    }
    if j < char_len {
        let b = unsafe { *src.get_unchecked(b_idx) };
        let hi = b >> 4;
        let lo = b & 0x0f;
        if hi >= pal_len || lo != 0 {
            return Err(invalid("column front dict nibble padding"));
        }
        unsafe {
            *dst.add(j) = *pal.get_unchecked(hi as usize);
        }
    }
    Ok(())
}

/// Unpack `count` `width`-bit integers (`width in 0..=7`) into `out`, adding `base`
/// to each value, and simultaneously verify that every raw value is `< bound`
/// (`1 <= bound <= 128`) and tail padding bits are zero.
///
/// Processes 8 rows (`width` bytes) per iteration using 64-bit SWAR with zero branches.
#[inline(always)]
fn unpack_and_validate_u8(
    bytes: &[u8],
    count: usize,
    width: u8,
    base: u8,
    bound: usize,
    out: &mut Vec<u8>,
) -> Result<usize> {
    let expected_len = bit_bytes(count, width)?;
    if bytes.len() != expected_len {
        return Err(Error::Truncated);
    }
    let used_bits = count * (width as usize);
    let tail_bits = used_bits & 7;
    if tail_bits != 0 {
        let last = *bytes.last().ok_or(Error::Truncated)?;
        if (last >> tail_bits) != 0 {
            return Err(invalid("nonzero column bit padding"));
        }
    }
    let start = out.len();
    out.reserve(count + 8);
    if width == 0 {
        unsafe {
            std::ptr::write_bytes(out.as_mut_ptr().add(start), base, count);
            out.set_len(start + count);
        }
        return Ok(start);
    }
    if width == 8 {
        if bound < 256 {
            let b_max = bound as u8;
            let mut i = 0usize;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                while i + 16 <= count {
                    let v = vld1q_u8(bytes.as_ptr().add(i));
                    if vmaxvq_u8(v) >= b_max {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    i += 16;
                }
            }
            while i < count {
                if bytes[i] >= b_max {
                    return Err(invalid("column bit index out of bounds"));
                }
                i += 1;
            }
        }
        unsafe {
            let dst = out.as_mut_ptr().add(start);
            if base == 0 {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, count);
            } else {
                for (i, &b) in bytes.iter().enumerate() {
                    *dst.add(i) = b.wrapping_add(base);
                }
            }
            out.set_len(start + count);
        }
        return Ok(start);
    }
    let w = width as u32;
    let m = (1u64 << w) - 1;
    let check_bound = bound < (1usize << w);
    let bias = ((128usize.saturating_sub(bound)) as u64).wrapping_mul(0x0101_0101_0101_0101);
    let base_splat = (base as u64).wrapping_mul(0x0101_0101_0101_0101);
    let mut bad_bits = 0u64;
    let mut row = 0usize;
    let mut byte_pos = 0usize;
    let step = width as usize;
    let src_ptr = bytes.as_ptr();
    let dst_ptr = unsafe { out.as_mut_ptr().add(start) };

    while row < count {
        let v = if byte_pos + 8 <= bytes.len() {
            u64::from_le(unsafe { src_ptr.add(byte_pos).cast::<u64>().read_unaligned() })
        } else {
            u64::from_le(unsafe {
                read_u128_partial(src_ptr.add(byte_pos), bytes.len() - byte_pos) as u64
            })
        };
        let u8x8 = (v & m)
            | (((v >> w) & m) << 8)
            | (((v >> (2 * w)) & m) << 16)
            | (((v >> (3 * w)) & m) << 24)
            | (((v >> (4 * w)) & m) << 32)
            | (((v >> (5 * w)) & m) << 40)
            | (((v >> (6 * w)) & m) << 48)
            | (((v >> (7 * w)) & m) << 56);
        if check_bound {
            bad_bits |= u8x8.wrapping_add(bias);
        }
        let out_word = if base == 0 {
            u8x8
        } else {
            u8x8.wrapping_add(base_splat)
        };
        unsafe {
            dst_ptr.add(row).cast::<u64>().write_unaligned(out_word.to_le());
        }
        row += 8;
        byte_pos += step;
    }
    if check_bound && (bad_bits & 0x8080_8080_8080_8080) != 0 {
        return Err(invalid("column bit index out of bounds"));
    }
    unsafe {
        out.set_len(start + count);
    }
    Ok(start)
}

