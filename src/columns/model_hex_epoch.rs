#[inline(always)]
fn hex_nibble(byte: u8) -> Option<u8> {
    let n = unsafe { *HEX_NIBBLE_LUT.get_unchecked(byte as usize) };
    if n != 0xff { Some(n) } else { None }
}

#[inline(always)]
fn parse_hex_u32_le(digits: &[u8]) -> Option<u32> {
    if digits.len() != 8 {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1_u8(digits.as_ptr());
        let d_dec = vsub_u8(v, vdup_n_u8(b'0'));
        let is_dec = vclt_u8(d_dec, vdup_n_u8(10));
        let is_hex = vclt_u8(vsub_u8(v, vdup_n_u8(b'a')), vdup_n_u8(6));
        let valid = vorr_u8(is_dec, is_hex);
        if vget_lane_u64(vreinterpret_u64_u8(valid), 0) != u64::MAX {
            return None;
        }
        let nib = vsub_u8(d_dec, vand_u8(is_hex, vdup_n_u8(39)));
        let nib16 = vreinterpret_u16_u8(nib);
        let paired = vorr_u16(vshl_n_u16(nib16, 4), vshr_n_u16(nib16, 8));
        let wide = vcombine_u16(paired, paired);
        return Some(vget_lane_u32(vreinterpret_u32_u8(vmovn_u16(wide)), 0));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let raw = digits.as_ptr().cast::<u64>().read_unaligned();
        let v = _mm_cvtsi64_si128(raw as i64);
        let d_dec = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
        let is_dec = _mm_cmpeq_epi8(_mm_min_epu8(d_dec, _mm_set1_epi8(9)), d_dec);
        let d_hex = _mm_sub_epi8(v, _mm_set1_epi8(b'a' as i8));
        let is_hex = _mm_cmpeq_epi8(_mm_min_epu8(d_hex, _mm_set1_epi8(5)), d_hex);
        let valid = _mm_or_si128(is_dec, is_hex);
        if (_mm_movemask_epi8(valid) & 0xff) != 0xff {
            return None;
        }
        let nib = _mm_sub_epi8(d_dec, _mm_and_si128(is_hex, _mm_set1_epi8(39)));
        let paired = _mm_and_si128(
            _mm_or_si128(_mm_slli_epi16(nib, 4), _mm_srli_epi16(nib, 8)),
            _mm_set1_epi16(0x00ff),
        );
        let packed = _mm_packus_epi16(paired, paired);
        return Some(_mm_cvtsi128_si32(packed) as u32);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let mut raw = [0u8; 4];
        let mut acc_err = 0u8;
        for i in 0..4 {
            let hi = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2] as usize) };
            let lo = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2 + 1] as usize) };
            acc_err |= hi | lo;
            raw[i] = (hi << 4) | lo;
        }
        if acc_err & 0xf0 != 0 {
            return None;
        }
        Some(u32::from_le_bytes(raw))
    }
}

#[inline(always)]
fn parse_hex_u64_le(digits: &[u8]) -> Option<u64> {
    if digits.len() != 16 {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1q_u8(digits.as_ptr());
        let d_dec = vsubq_u8(v, vdupq_n_u8(b'0'));
        let is_dec = vcltq_u8(d_dec, vdupq_n_u8(10));
        let is_hex = vcltq_u8(vsubq_u8(v, vdupq_n_u8(b'a')), vdupq_n_u8(6));
        let valid = vorrq_u8(is_dec, is_hex);
        let mask64 = vget_lane_u64(
            vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(valid), 4)),
            0,
        );
        if mask64 != u64::MAX {
            return None;
        }
        let nib = vsubq_u8(d_dec, vandq_u8(is_hex, vdupq_n_u8(39)));
        let nib16 = vreinterpretq_u16_u8(nib);
        let paired = vorrq_u16(vshlq_n_u16(nib16, 4), vshrq_n_u16(nib16, 8));
        return Some(vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(paired)), 0));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_loadu_si128(digits.as_ptr().cast::<__m128i>());
        let d_dec = _mm_sub_epi8(v, _mm_set1_epi8(b'0' as i8));
        let is_dec = _mm_cmpeq_epi8(_mm_min_epu8(d_dec, _mm_set1_epi8(9)), d_dec);
        let d_hex = _mm_sub_epi8(v, _mm_set1_epi8(b'a' as i8));
        let is_hex = _mm_cmpeq_epi8(_mm_min_epu8(d_hex, _mm_set1_epi8(5)), d_hex);
        let valid = _mm_or_si128(is_dec, is_hex);
        if _mm_movemask_epi8(valid) != 0xffff {
            return None;
        }
        let nib = _mm_sub_epi8(d_dec, _mm_and_si128(is_hex, _mm_set1_epi8(39)));
        let paired = _mm_and_si128(
            _mm_or_si128(_mm_slli_epi16(nib, 4), _mm_srli_epi16(nib, 8)),
            _mm_set1_epi16(0x00ff),
        );
        let packed = _mm_packus_epi16(paired, paired);
        return Some(_mm_cvtsi128_si64(packed) as u64);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let mut raw = [0u8; 8];
        let mut acc_err = 0u8;
        for i in 0..8 {
            let hi = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2] as usize) };
            let lo = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2 + 1] as usize) };
            acc_err |= hi | lo;
            raw[i] = (hi << 4) | lo;
        }
        if acc_err & 0xf0 != 0 {
            return None;
        }
        Some(u64::from_le_bytes(raw))
    }
}

#[inline(always)]
fn split_hex_suffix(value: &[u8]) -> Option<(&[u8], &[u8])> {
    if value.len() < 4 || value.first() != Some(&b'"') || value.last() != Some(&b'"') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    let mut start = inner.len();
    while start != 0 && hex_nibble(inner[start - 1]).is_some() {
        start -= 1;
    }
    let digits = &inner[start..];
    if digits.is_empty() || digits.len() % 2 != 0 {
        return None;
    }
    Some((&inner[..start], digits))
}

#[inline(always)]
fn pack_hex_suffix_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let (prefix, first_digits) = split_hex_suffix(values[0])?;
    let raw_len = first_digits.len() / 2;
    if raw_len == 0 || raw_len > 64 {
        return None;
    }
    let size = 1 + uleb_len(prefix.len()) + prefix.len() + uleb_len(raw_len) + n * raw_len;
    if size >= limit {
        return None;
    }
    let expected_val_len = 2 + prefix.len() + first_digits.len();
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_HEX_SUFFIX);
    put_bytes(&mut out, prefix);
    put_uleb(&mut out, raw_len);
    let prefix_end = 1 + prefix.len();
    let qprefix = &values[0][..prefix_end];
    if raw_len == 4 {
        for &value in values {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            if prefix_end > 1 && hex_nibble(value[prefix_end - 1]).is_some() {
                return None;
            }
            let word = parse_hex_u32_le(&value[prefix_end..expected_val_len - 1])?;
            out.extend_from_slice(&word.to_le_bytes());
        }
    } else if raw_len == 8 {
        for &value in values {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            if prefix_end > 1 && hex_nibble(value[prefix_end - 1]).is_some() {
                return None;
            }
            let word = parse_hex_u64_le(&value[prefix_end..expected_val_len - 1])?;
            out.extend_from_slice(&word.to_le_bytes());
        }
    } else {
        for &value in values {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            if prefix_end > 1 && hex_nibble(value[prefix_end - 1]).is_some() {
                return None;
            }
            let digits = &value[prefix_end..expected_val_len - 1];
            for pair in digits.chunks_exact(2) {
                let hi = hex_nibble(pair[0])?;
                let lo = hex_nibble(pair[1])?;
                out.push((hi << 4) | lo);
            }
        }
    }
    Some(out)
}

#[inline(always)]
fn pack_epoch_counter_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let (literal, first_digits) = split_hex_suffix(values[0])?;
    if first_digits.len() != 16 {
        return None;
    }
    let base_overhead = 1 + uleb_len(literal.len()) + literal.len() + 3 + ((n + 7) >> 3);
    if base_overhead + 5 >= limit {
        return None;
    }
    let expected_val_len = 2 + literal.len() + 16;
    let prefix_end = 1 + literal.len();
    let qprefix = &values[0][..prefix_end];
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.epoch_prefixes.clear();
        sc.epoch_prefixes.reserve(n);
        sc.epoch_flags.clear();
        sc.epoch_flags.reserve(n);
        sc.epoch_explicit.clear();
        let pref_ptr = sc.epoch_prefixes.as_mut_ptr();
        let flag_ptr = sc.epoch_flags.as_mut_ptr();
        let mut unique_prefixes = [0u32; MAX_EPOCH_PREFIXES];
        let mut unique_len = 0usize;
        let mut last_prefix = 0u32;
        let mut last_pid = u8::MAX;
        let mut previous = u64::MAX - 1;
        let mut explicit_bytes = 0usize;
        let qprefix_u32 = if prefix_end == 3 {
            Some((qprefix[0] as u32) | ((qprefix[1] as u32) << 8) | ((qprefix[2] as u32) << 16))
        } else {
            None
        };

        for (idx, &value) in values.iter().enumerate() {
            if value.len() != expected_val_len
                || unsafe { *value.get_unchecked(expected_val_len - 1) } != b'"'
            {
                return None;
            }
            if let Some(qp3) = qprefix_u32 {
                let w3 = unsafe { (value.as_ptr() as *const u32).read_unaligned().to_le() & 0x00ff_ffff };
                if w3 != qp3 {
                    return None;
                }
            } else if !slice_eq_fast(unsafe { value.get_unchecked(..prefix_end) }, qprefix) {
                return None;
            }
            let digits = unsafe { value.get_unchecked(prefix_end..expected_val_len - 1) };
            let word64 = parse_hex_u64_le(digits)?;
            let prefix = (word64 as u32).swap_bytes();
            let suffix = ((word64 >> 32) as u32).swap_bytes();
            let pid = if prefix == last_prefix && last_pid != u8::MAX {
                last_pid
            } else if let Some(pos) = unique_prefixes[..unique_len].iter().position(|&p| p == prefix) {
                last_prefix = prefix;
                last_pid = pos as u8;
                last_pid
            } else {
                if unique_len >= MAX_EPOCH_PREFIXES {
                    return None;
                }
                let id = unique_len as u8;
                unique_prefixes[unique_len] = prefix;
                unique_len += 1;
                last_prefix = prefix;
                last_pid = id;
                id
            };
            unsafe {
                *pref_ptr.add(idx) = pid;
            }
            let current = u64::from(suffix);
            if idx > 0 && previous + 1 == current {
                unsafe {
                    *flag_ptr.add(idx) = 0u8;
                }
            } else {
                unsafe {
                    *flag_ptr.add(idx) = 1u8;
                }
                sc.epoch_explicit.push(current);
                explicit_bytes += uleb_len(current as usize);
                if base_overhead + 4 * unique_len + explicit_bytes >= limit {
                    return None;
                }
                if idx == 7 && unique_len >= 7 && sc.epoch_explicit.len() == 8 {
                    return None;
                }
            }
            previous = current;
        }

        if unique_len == 0 {
            return None;
        }
        unsafe {
            sc.epoch_prefixes.set_len(n);
            sc.epoch_flags.set_len(n);
        }
        let mut order: [u8; MAX_EPOCH_PREFIXES] = core::array::from_fn(|i| i as u8);
        let order_slice = &mut order[..unique_len];
        order_slice.sort_unstable_by_key(|&id| unique_prefixes[id as usize]);
        let mut rank_of = [0u64; MAX_EPOCH_PREFIXES];
        for (rank, &orig_id) in order_slice.iter().enumerate() {
            rank_of[orig_id as usize] = rank as u64;
        }

        let width = bit_width((unique_len - 1) as u64);
        let size = 1
            + uleb_len(literal.len())
            + literal.len()
            + uleb_len(unique_len)
            + 4 * unique_len
            + 1
            + bit_bytes(n, width).ok()?
            + bit_bytes(n, 1).ok()?
            + uleb_len(sc.epoch_explicit.len())
            + sc.epoch_explicit
                .iter()
                .map(|value| uleb_len(*value as usize))
                .sum::<usize>();
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_EPOCH_COUNTER);
        put_bytes(&mut out, literal);
        put_uleb(&mut out, unique_len);
        for &orig_id in order_slice.iter() {
            out.extend_from_slice(&unique_prefixes[orig_id as usize].to_le_bytes());
        }
        out.push(width);
        push_bit_values(
            &mut out,
            n,
            width,
            sc.epoch_prefixes
                .iter()
                .map(|&pid| unsafe { *rank_of.get_unchecked(pid as usize) }),
        );
        push_bit_values(
            &mut out,
            n,
            1,
            sc.epoch_flags.iter().map(|&f| u64::from(f)),
        );
        put_uleb(&mut out, sc.epoch_explicit.len());
        for &value in &sc.epoch_explicit {
            put_uleb(&mut out, value as usize);
        }
        Some(out)
    })
}

#[inline(always)]
fn segment_byte_ok(byte: u8) -> bool {
    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b' '
}

#[inline(always)]
fn canonical_segment_decimal(segment: &[u8]) -> Option<u64> {
    if segment.len() > MAX_DECIMAL_SEGMENT_DIGITS {
        return None;
    }
    parse_decimal(segment)
}

#[inline(always)]
fn validate_segment_slice(segment: &[u8]) -> bool {
    !segment.is_empty()
        && segment.len() <= MAX_SEGMENT_LEN
        && segment.iter().all(|&byte| segment_byte_ok(byte))
}

#[inline(always)]
fn next_semi(data: &[u8], mut pos: usize) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    unsafe {
        while pos + 8 <= len {
            let x = ptr.add(pos).cast::<u64>().read_unaligned().to_le() ^ 0x3b3b_3b3b_3b3b_3b3b;
            let z = x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
            if z != 0 {
                return Some(pos + (z.trailing_zeros() >> 3) as usize);
            }
            pos += 8;
        }
        if pos < len {
            if len >= 8 {
                let shift = (8 - (len - pos)) << 3;
                let x = (ptr.add(len - 8).cast::<u64>().read_unaligned().to_le() >> shift)
                    ^ 0x3b3b_3b3b_3b3b_3b3b;
                let z = x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080;
                return if z != 0 {
                    Some(pos + (z.trailing_zeros() >> 3) as usize)
                } else {
                    None
                };
            }
            if pos + 4 <= len {
                let x = ptr.add(pos).cast::<u32>().read_unaligned().to_le() ^ 0x3b3b_3b3b;
                let z = x.wrapping_sub(0x0101_0101) & !x & 0x8080_8080;
                if z != 0 {
                    return Some(pos + (z.trailing_zeros() >> 3) as usize);
                }
                pos += 4;
            }
            while pos < len {
                if *ptr.add(pos) == b';' {
                    return Some(pos);
                }
                pos += 1;
            }
        }
        None
    }
}

#[inline(always)]
fn seg_words24(seg: &[u8]) -> (u64, u64, u64) {
    let len = seg.len();
    let ptr = seg.as_ptr();
    unsafe {
        if len >= 8 {
            let w0 = ptr.cast::<u64>().read_unaligned();
            let w1 = ptr.add(len - 8).cast::<u64>().read_unaligned();
            let w2 = if len > 16 {
                ptr.add(8).cast::<u64>().read_unaligned()
            } else {
                0
            };
            (w0, w1, w2)
        } else if len >= 4 {
            (
                (ptr.cast::<u32>().read_unaligned() as u64)
                    | ((ptr.add(len - 4).cast::<u32>().read_unaligned() as u64) << 32),
                0,
                0,
            )
        } else {
            (
                (*ptr as u64)
                    | ((*ptr.add(len >> 1) as u64) << 8)
                    | ((*ptr.add(len - 1) as u64) << 16),
                0,
                0,
            )
        }
    }
}

