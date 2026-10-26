fn unpack_front_dict_indices(
    cursor: &mut Cursor<'_>,
    count: usize,
    entries_len: usize,
    width: u8,
    idx_mode: u8,
    u8_arena: &mut Vec<u8>,
) -> Result<usize> {
    match idx_mode {
        0 => {
            let packed_len = bit_bytes(count, width)?;
            let packed_bytes = cursor.take(packed_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            unpack_and_validate_u8(packed_bytes, count, width, 0, entries_len, u8_arena)
        }
        1 => {
            let bm_len = bit_bytes(count, 1)?;
            let bm_bytes = cursor.take(bm_len)?;
            let _ = FastBitReader::new_validated(bm_bytes, count, 1)?;
            if count > 0 && (bm_bytes[0] & 1) != 0 {
                return Err(invalid("column front dict repeat bit0"));
            }
            let mut ones = 0usize;
            for &b in bm_bytes {
                ones += b.count_ones() as usize;
            }
            let exc_count = count - ones;
            let exc_len = bit_bytes(exc_count, width)?;
            let exc_bytes = cursor.take(exc_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let dst_offset = u8_arena.len();
            u8_arena.reserve(count + exc_count + 16);
            unsafe {
                u8_arena.set_len(dst_offset + count);
            }
            let exc_offset =
                unpack_and_validate_u8(exc_bytes, exc_count, width, 0, entries_len, u8_arena)?;
            unsafe {
                let exc_ptr = u8_arena.as_ptr().add(exc_offset);
                let dst_ptr = u8_arena.as_mut_ptr().add(dst_offset);
                let mut exc_idx = 0usize;
                let full_bytes = count >> 3;
                for byte_idx in 0..full_bytes {
                    let b = *bm_bytes.get_unchecked(byte_idx);
                    let d = dst_ptr.add(byte_idx << 3);
                    if b == 0xff {
                        let v =
                            (*exc_ptr.add(exc_idx - 1) as u64).wrapping_mul(0x0101_0101_0101_0101);
                        d.cast::<u64>().write_unaligned(v);
                    } else if b == 0x00 {
                        d.cast::<u64>()
                            .write_unaligned(exc_ptr.add(exc_idx).cast::<u64>().read_unaligned());
                        exc_idx += 8;
                    } else {
                        let nb = (!b) as usize;
                        exc_idx += nb & 1;
                        *d = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 1) & 1;
                        *d.add(1) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 2) & 1;
                        *d.add(2) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 3) & 1;
                        *d.add(3) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 4) & 1;
                        *d.add(4) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 5) & 1;
                        *d.add(5) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 6) & 1;
                        *d.add(6) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 7) & 1;
                        *d.add(7) = *exc_ptr.add(exc_idx - 1);
                    }
                }
                let rem = count & 7;
                if rem != 0 {
                    let nb = (!*bm_bytes.get_unchecked(full_bytes)) as usize;
                    let d = dst_ptr.add(full_bytes << 3);
                    for bit in 0..rem {
                        exc_idx += (nb >> bit) & 1;
                        *d.add(bit) = *exc_ptr.add(exc_idx - 1);
                    }
                }
                u8_arena.set_len(dst_offset + count);
            }
            Ok(dst_offset)
        }
        2 => {
            let mode_rank = cursor.uleb()?;
            if mode_rank >= entries_len {
                return Err(invalid("column front dict mode rank"));
            }
            let bm_len = bit_bytes(count, 1)?;
            let bm_bytes = cursor.take(bm_len)?;
            let _ = FastBitReader::new_validated(bm_bytes, count, 1)?;
            let mut ones = 0usize;
            for &b in bm_bytes {
                ones += b.count_ones() as usize;
            }
            let exc_count = count - ones;
            let exc_bound = entries_len - 1;
            let exc_width = bit_width(exc_bound.saturating_sub(1) as u64);
            let exc_len = bit_bytes(exc_count, exc_width)?;
            let exc_bytes = cursor.take(exc_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let dst_offset = u8_arena.len();
            u8_arena.reserve(count + exc_count + 16);
            let mr = mode_rank as u8;
            unsafe {
                std::ptr::write_bytes(u8_arena.as_mut_ptr().add(dst_offset), mr, count);
                u8_arena.set_len(dst_offset + count);
            }
            let exc_offset = unpack_and_validate_u8(
                exc_bytes,
                exc_count,
                exc_width,
                0,
                exc_bound.max(1),
                u8_arena,
            )?;
            unsafe {
                let exc_ptr = u8_arena.as_ptr().add(exc_offset);
                let dst_ptr = u8_arena.as_mut_ptr().add(dst_offset);
                let mut exc_idx = 0usize;
                let full_words = count >> 6;
                for word_idx in 0..full_words {
                    let base = word_idx << 6;
                    let w = u64::from_le(
                        bm_bytes
                            .as_ptr()
                            .add(word_idx << 3)
                            .cast::<u64>()
                            .read_unaligned(),
                    );
                    let mut zeros = !w;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let v = *exc_ptr.add(exc_idx);
                        exc_idx += 1;
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                for byte_idx in (full_words << 3)..bm_bytes.len() {
                    let b = *bm_bytes.get_unchecked(byte_idx);
                    let base = byte_idx << 3;
                    let rem = (count - base).min(8);
                    let mask = ((1u16 << rem) - 1) as u8;
                    let mut zeros = (!b) & mask;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let v = *exc_ptr.add(exc_idx);
                        exc_idx += 1;
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                u8_arena.set_len(dst_offset + count);
            }
            Ok(dst_offset)
        }
        3 => {
            let mode_rank = cursor.uleb()?;
            if mode_rank >= entries_len {
                return Err(invalid("column front dict mode rank"));
            }
            let bm_len = bit_bytes(count, 1)?;
            let bm_bytes = cursor.take(bm_len)?;
            let _ = FastBitReader::new_validated(bm_bytes, count, 1)?;
            let mut mode_ones = 0usize;
            for &b in bm_bytes {
                mode_ones += b.count_ones() as usize;
            }
            let rem_count = count - mode_ones;
            let rep_len = bit_bytes(rem_count, 1)?;
            let rep_bytes = cursor.take(rep_len)?;
            let _ = FastBitReader::new_validated(rep_bytes, rem_count, 1)?;
            if rem_count > 0 && (rep_bytes[0] & 1) != 0 {
                return Err(invalid("column front dict repeat bit0"));
            }
            let mut rep_ones = 0usize;
            for &b in rep_bytes {
                rep_ones += b.count_ones() as usize;
            }
            let exc_count = rem_count - rep_ones;
            let exc_bound = entries_len - 1;
            let exc_width = bit_width(exc_bound.saturating_sub(1) as u64);
            let exc_len = bit_bytes(exc_count, exc_width)?;
            let exc_bytes = cursor.take(exc_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let dst_offset = u8_arena.len();
            u8_arena.reserve(count + exc_count + 16);
            let mr = mode_rank as u8;
            unsafe {
                std::ptr::write_bytes(u8_arena.as_mut_ptr().add(dst_offset), mr, count);
                u8_arena.set_len(dst_offset + count);
            }
            let exc_offset = unpack_and_validate_u8(
                exc_bytes,
                exc_count,
                exc_width,
                0,
                exc_bound.max(1),
                u8_arena,
            )?;
            unsafe {
                let exc_ptr = u8_arena.as_ptr().add(exc_offset);
                let dst_ptr = u8_arena.as_mut_ptr().add(dst_offset);
                let mut nm_idx = 0usize;
                let mut exc_idx = 0usize;
                let full_words = count >> 6;
                for word_idx in 0..full_words {
                    let base = word_idx << 6;
                    let w = u64::from_le(
                        bm_bytes
                            .as_ptr()
                            .add(word_idx << 3)
                            .cast::<u64>()
                            .read_unaligned(),
                    );
                    let mut zeros = !w;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let is_rep = (*rep_bytes.get_unchecked(nm_idx >> 3) >> (nm_idx & 7)) & 1;
                        nm_idx += 1;
                        exc_idx += (is_rep ^ 1) as usize;
                        let v = *exc_ptr.add(exc_idx - 1);
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                for byte_idx in (full_words << 3)..bm_bytes.len() {
                    let b = *bm_bytes.get_unchecked(byte_idx);
                    let base = byte_idx << 3;
                    let rem = (count - base).min(8);
                    let mask = ((1u16 << rem) - 1) as u8;
                    let mut zeros = (!b) & mask;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let is_rep = (*rep_bytes.get_unchecked(nm_idx >> 3) >> (nm_idx & 7)) & 1;
                        nm_idx += 1;
                        exc_idx += (is_rep ^ 1) as usize;
                        let v = *exc_ptr.add(exc_idx - 1);
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                u8_arena.set_len(dst_offset + count);
            }
            Ok(dst_offset)
        }
        _ => Err(invalid("column front dict index mode")),
    }
}

/// Unpack `count` `width`-bit integers (`width in 1..=14`) into `out`, adding `min`
/// to each value (where `min + (1 << width) - 1 < 65536`), and validate tail padding bits.
///
/// Processes 4 `u16` values per iteration using a single 64-bit load and SWAR add.
#[inline(always)]
fn unpack_u16_swar(
    bytes: &[u8],
    count: usize,
    width: u8,
    min: u16,
    out: &mut Vec<u16>,
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
    let w = width as u32;
    let m = (1u64 << w) - 1;
    let src_ptr = bytes.as_ptr();
    let dst_ptr = unsafe { out.as_mut_ptr().add(start) };

    if width <= 7 {
        let step = width as usize;
        let mut row = 0usize;
        let mut byte_pos = 0usize;
        #[cfg(target_arch = "aarch64")]
        let v_min = unsafe { core::arch::aarch64::vdupq_n_u16(min) };
        #[cfg(target_arch = "x86_64")]
        let v_min = unsafe { core::arch::x86_64::_mm_set1_epi16(min as i16) };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let min_splat = (min as u64).wrapping_mul(0x0001_0001_0001_0001);

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
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let v8 = vreinterpret_u8_u64(vdup_n_u64(u8x8));
                let v16 = vaddq_u16(vmovl_u8(v8), v_min);
                vst1q_u16(dst_ptr.add(row), v16);
            }
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use core::arch::x86_64::*;
                let v8 = _mm_cvtsi64_si128(u8x8 as i64);
                let v16 = _mm_add_epi16(_mm_unpacklo_epi8(v8, _mm_setzero_si128()), v_min);
                _mm_storeu_si128(dst_ptr.add(row).cast::<__m128i>(), v16);
            }
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            unsafe {
                let lo = (u8x8 & 0xff)
                    | (((u8x8 >> 8) & 0xff) << 16)
                    | (((u8x8 >> 16) & 0xff) << 32)
                    | (((u8x8 >> 24) & 0xff) << 48);
                let hi = ((u8x8 >> 32) & 0xff)
                    | (((u8x8 >> 40) & 0xff) << 16)
                    | (((u8x8 >> 48) & 0xff) << 32)
                    | (((u8x8 >> 56) & 0xff) << 48);
                dst_ptr
                    .add(row)
                    .cast::<u64>()
                    .write_unaligned(lo.wrapping_add(min_splat).to_le());
                dst_ptr
                    .add(row + 4)
                    .cast::<u64>()
                    .write_unaligned(hi.wrapping_add(min_splat).to_le());
            }
            row += 8;
            byte_pos += step;
        }
        unsafe {
            out.set_len(start + count);
        }
        return Ok(start);
    }

    let min_splat = (min as u64).wrapping_mul(0x0001_0001_0001_0001);
    let step_bits = (width as usize) * 4;
    let mut row = 0usize;
    let mut bit_pos = 0usize;

    while row < count {
        let byte_pos = bit_pos >> 3;
        let bit_off = (bit_pos & 7) as u32;
        let raw = if byte_pos + 8 <= bytes.len() {
            u64::from_le(unsafe { src_ptr.add(byte_pos).cast::<u64>().read_unaligned() })
        } else {
            u64::from_le(unsafe {
                read_u128_partial(src_ptr.add(byte_pos), bytes.len() - byte_pos) as u64
            })
        };
        let v = raw >> bit_off;
        let u16x4 = (v & m)
            | (((v >> w) & m) << 16)
            | (((v >> (2 * w)) & m) << 32)
            | (((v >> (3 * w)) & m) << 48);
        let out_word = u16x4.wrapping_add(min_splat);
        unsafe {
            dst_ptr.add(row).cast::<u64>().write_unaligned(out_word.to_le());
        }
        row += 4;
        bit_pos += step_bits;
    }
    unsafe {
        out.set_len(start + count);
    }
    Ok(start)
}

#[inline(always)]
unsafe fn write_hex8_from_4bytes(dst: *mut u8, src: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vreinterpret_u8_u32(vdup_n_u32(src.cast::<u32>().read_unaligned()));
        let hi = vshr_n_u8(v, 4);
        let lo = vand_u8(v, vdup_n_u8(0x0f));
        let zipped = vzip1_u8(hi, lo);
        let hex_lut = vld1q_u8(HEX.as_ptr());
        let ascii = vqtbl1_u8(hex_lut, zipped);
        vst1_u8(dst, ascii);
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        let b0 = *HEX_BYTE_LUT.get_unchecked(*src as usize) as u64;
        let b1 = *HEX_BYTE_LUT.get_unchecked(*src.add(1) as usize) as u64;
        let b2 = *HEX_BYTE_LUT.get_unchecked(*src.add(2) as usize) as u64;
        let b3 = *HEX_BYTE_LUT.get_unchecked(*src.add(3) as usize) as u64;
        let lo = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
        dst.cast::<u64>().write_unaligned(lo);
    }
}

#[inline(always)]
unsafe fn write_hex16_from_8bytes(dst: *mut u8, src: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1_u8(src);
        let hi = vshr_n_u8(v, 4);
        let lo = vand_u8(v, vdup_n_u8(0x0f));
        let zipped = vcombine_u8(vzip1_u8(hi, lo), vzip2_u8(hi, lo));
        let hex_lut = vld1q_u8(HEX.as_ptr());
        let ascii = vqtbl1q_u8(hex_lut, zipped);
        vst1q_u8(dst, ascii);
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        write_hex8_from_4bytes(dst, src);
        write_hex8_from_4bytes(dst.add(8), src.add(4));
    }
}

