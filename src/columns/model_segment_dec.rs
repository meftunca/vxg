fn pack_segment_dictionary_model<'a>(values: &[&'a [u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 6 + (n + 7) / 8 >= limit {
        return None;
    }
    const HT_CAP: usize = 256;
    const HT_MASK: usize = HT_CAP - 1;
    let mut ht_slots = [u8::MAX; HT_CAP];
    let mut dict_slices = [core::mem::MaybeUninit::<&'a [u8]>::uninit(); MAX_SEGMENT_DICT];
    let mut dict_w0 = [0u64; MAX_SEGMENT_DICT];
    let mut dict_w1 = [0u64; MAX_SEGMENT_DICT];
    let mut dict_w2 = [0u64; MAX_SEGMENT_DICT];
    let mut dict_lens = [0u8; MAX_SEGMENT_DICT];
    let mut dict_len = 0usize;
    let mut dict_bytes_sum = 0usize;
    let mut max_decimal = 0u64;
    let mut max_pad = 0u64;

    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.seg_tokens.clear();
        sc.seg_tokens.reserve(n * (MAX_SEGMENTS_PER_VALUE + 1));
        sc.seg_decimals.clear();
        sc.seg_decimals.reserve(n * MAX_SEGMENTS_PER_VALUE);
        sc.seg_pads.clear();
        sc.seg_pads.reserve(n);

        let tok_ptr = sc.seg_tokens.as_mut_ptr();
        let dec_ptr = sc.seg_decimals.as_mut_ptr();
        let pad_ptr = sc.seg_pads.as_mut_ptr();
        let mut tok_len = 0usize;
        let mut dec_len = 0usize;
        let mut last_w0 = 0u64;
        let mut last_w1 = 0u64;
        let mut last_w2 = 0u64;
        let mut last_len = 0usize;
        let mut last_id: u8 = u8::MAX;

        for (row_idx, &value) in values.iter().enumerate() {
            let val_len = value.len();
            if val_len < 2 || value[0] != b'"' || value[val_len - 1] != b'"' {
                return None;
            }
            let inner = &value[1..val_len - 1];
            let mut end = inner.len();
            let inner_ptr = inner.as_ptr();
            while end >= 8 {
                let w = unsafe {
                    inner_ptr.add(end - 8).cast::<u64>().read_unaligned().to_le()
                        ^ 0x2020_2020_2020_2020
                };
                if w != 0 {
                    end -= (w.leading_zeros() >> 3) as usize;
                    break;
                }
                end -= 8;
            }
            if end < 8 {
                while end != 0 && unsafe { *inner_ptr.add(end - 1) } == b' ' {
                    end -= 1;
                }
            }
            let pad = inner.len() - end;
            if pad > MAX_PAD {
                return None;
            }
            unsafe {
                *pad_ptr.add(row_idx) = pad as u16;
            }
            if (pad as u64) > max_pad {
                max_pad = pad as u64;
            }
            let content = unsafe { inner.get_unchecked(..end) };
            if !content.is_empty() {
                let mut seg_count = 0usize;
                let mut start = 0usize;
                let c_len = content.len();
                let scan_slice = unsafe { content.get_unchecked(..c_len - 1) };
                loop {
                    let is_last;
                    let seg = if start < scan_slice.len() {
                        if let Some(pos) = next_semi(scan_slice, start) {
                            if unsafe { *content.get_unchecked(pos + 1) } != b' ' {
                                return None;
                            }
                            let s = unsafe { content.get_unchecked(start..pos) };
                            start = pos + 2;
                            is_last = false;
                            s
                        } else {
                            is_last = true;
                            unsafe { content.get_unchecked(start..) }
                        }
                    } else {
                        is_last = true;
                        unsafe { content.get_unchecked(start..) }
                    };
                    let s_len = seg.len();
                    if s_len == 0 || s_len > MAX_SEGMENT_LEN {
                        return None;
                    }
                    seg_count += 1;
                    if seg_count > MAX_SEGMENTS_PER_VALUE {
                        return None;
                    }
                    if seg[0].is_ascii_digit() {
                        if let Some(dec) = canonical_segment_decimal(seg) {
                            unsafe {
                                *tok_ptr.add(tok_len) = 0xff;
                                tok_len += 1;
                                *dec_ptr.add(dec_len) = dec;
                                dec_len += 1;
                            }
                            if dec > max_decimal {
                                max_decimal = dec;
                            }
                            if is_last {
                                break;
                            }
                            continue;
                        }
                    }
                    let (w0, w1, w2) = seg_words24(seg);
                    let id = if s_len <= 24
                        && s_len == last_len
                        && w0 == last_w0
                        && w1 == last_w1
                        && w2 == last_w2
                        && last_id != u8::MAX
                    {
                        last_id
                    } else {
                        let m = (w0 ^ w1.rotate_left(31) ^ w2.rotate_left(17) ^ (s_len as u64))
                            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                        let mut pos = ((m ^ (m >> 29)) as usize) & HT_MASK;
                        let found_id = loop {
                            let slot = ht_slots[pos];
                            if slot == u8::MAX {
                                if dict_len >= MAX_SEGMENT_DICT || !validate_segment_slice(seg) {
                                    return None;
                                }
                                let id = dict_len as u8;
                                dict_slices[dict_len].write(seg);
                                dict_w0[dict_len] = w0;
                                dict_w1[dict_len] = w1;
                                dict_w2[dict_len] = w2;
                                dict_lens[dict_len] = s_len as u8;
                                dict_len += 1;
                                dict_bytes_sum += 1 + s_len;
                                ht_slots[pos] = id;
                                break id;
                            }
                            let idx = slot as usize;
                            if dict_lens[idx] == s_len as u8
                                && dict_w0[idx] == w0
                                && dict_w1[idx] == w1
                                && dict_w2[idx] == w2
                                && (s_len <= 24
                                    || slice_eq_fast(
                                        unsafe {
                                            dict_slices[idx]
                                                .assume_init()
                                                .get_unchecked(16..s_len - 8)
                                        },
                                        unsafe { seg.get_unchecked(16..s_len - 8) },
                                    ))
                            {
                                break slot;
                            }
                            pos = (pos + 1) & HT_MASK;
                        };
                        if s_len <= 24 {
                            last_w0 = w0;
                            last_w1 = w1;
                            last_w2 = w2;
                            last_len = s_len;
                            last_id = found_id;
                        }
                        found_id
                    };
                    unsafe {
                        *tok_ptr.add(tok_len) = id;
                        tok_len += 1;
                    }
                    if is_last {
                        break;
                    }
                }
            }
            unsafe {
                *tok_ptr.add(tok_len) = 0xfe;
                tok_len += 1;
            }

            // Periodic lower bound check on final SegmentDict size:
            if (row_idx & 15) == 15 {
                let min_tokens = tok_len + (n - 1 - row_idx);
                let min_w = bit_width((dict_len + 1) as u64) as usize;
                let dec_w = bit_width(max_decimal) as usize;
                if 6 + dict_bytes_sum
                    + (min_tokens * min_w + 7) / 8
                    + (dec_len * dec_w + 7) / 8
                    >= limit
                {
                    return None;
                }
            }
        }

        unsafe {
            sc.seg_tokens.set_len(tok_len);
            sc.seg_decimals.set_len(dec_len);
            sc.seg_pads.set_len(n);
        }

        let width = bit_width((dict_len + 1) as u64);
        let decimal_width = bit_width(max_decimal);
        let pad_width = bit_width(max_pad);
        let size = 1
            + uleb_len(dict_len)
            + dict_bytes_sum
            + 1
            + uleb_len(tok_len)
            + bit_bytes(tok_len, width).ok()?
            + uleb_len(dec_len)
            + 1
            + bit_bytes(dec_len, decimal_width).ok()?
            + 1
            + bit_bytes(n, pad_width).ok()?;
        if size >= limit {
            return None;
        }

        let dict_init: &[&'a [u8]] = unsafe {
            std::slice::from_raw_parts(dict_slices.as_ptr().cast::<&'a [u8]>(), dict_len)
        };
        let mut order: [u8; MAX_SEGMENT_DICT] = core::array::from_fn(|i| i as u8);
        let order_slice = &mut order[..dict_len];
        order_slice.sort_unstable_by(|&a, &b| dict_init[a as usize].cmp(dict_init[b as usize]));
        let mut token_map = [0u64; 256];
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_SEGMENT_DICT);
        put_uleb(&mut out, dict_len);
        for (rank, &orig_id) in order_slice.iter().enumerate() {
            token_map[orig_id as usize] = rank as u64;
            put_bytes(&mut out, dict_init[orig_id as usize]);
        }
        token_map[0xfe] = dict_len as u64;
        token_map[0xff] = dict_len as u64 + 1;

        out.push(width);
        put_uleb(&mut out, tok_len);
        push_bit_values(
            &mut out,
            tok_len,
            width,
            sc.seg_tokens
                .iter()
                .map(|&tok| unsafe { *token_map.get_unchecked(tok as usize) }),
        );
        put_uleb(&mut out, dec_len);
        out.push(decimal_width);
        push_bit_values(
            &mut out,
            dec_len,
            decimal_width,
            sc.seg_decimals.iter().copied(),
        );
        out.push(pad_width);
        push_bit_values(
            &mut out,
            n,
            pad_width,
            sc.seg_pads.iter().map(|&p| u64::from(p)),
        );
        Some(out)
    })
}

#[inline(always)]
fn split_decimal_suffix(value: &[u8]) -> Option<(&[u8], u64)> {
    if value.len() < 4 || value.first() != Some(&b'"') || value.last() != Some(&b'"') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    let mut start = inner.len();
    while start != 0 && inner[start - 1].is_ascii_digit() {
        start -= 1;
    }
    let decimal = parse_decimal(&inner[start..])?;
    Some((&inner[..start], decimal))
}

#[inline(never)]
fn pack_decimal_suffix_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let v0 = values[0];
    let (prefix, first_val) = split_decimal_suffix(v0)?;
    if 1 + uleb_len(prefix.len()) + prefix.len() + 8 + 1 >= limit {
        return None;
    }
    let prefix_end = 1 + prefix.len();
    if prefix_end > 1 && v0[prefix_end - 1].is_ascii_digit() {
        return None;
    }
    let quoted_prefix = &v0[..prefix_end];
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.parsed_u64.clear();
        sc.parsed_u64.reserve(n);
        let parsed_ptr = sc.parsed_u64.as_mut_ptr();
        unsafe {
            *parsed_ptr = first_val;
        }
        let mut min = first_val;
        let mut max = first_val;
        if (8..=16).contains(&prefix_end) {
            let qp = quoted_prefix.as_ptr();
            let q0 = unsafe { (qp as *const u64).read_unaligned() };
            let q1 = unsafe { (qp.add(prefix_end - 8) as *const u64).read_unaligned() };
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2 || unsafe { *value.get_unchecked(val_len - 1) } != b'"' {
                    return None;
                }
                let vp = value.as_ptr();
                let w0 = unsafe { (vp as *const u64).read_unaligned() };
                let w1 = unsafe { (vp.add(prefix_end - 8) as *const u64).read_unaligned() };
                if (w0 ^ q0) | (w1 ^ q1) != 0 {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        } else if (4..8).contains(&prefix_end) {
            let qp = quoted_prefix.as_ptr();
            let q0 = unsafe { (qp as *const u32).read_unaligned() };
            let q1 = unsafe { (qp.add(prefix_end - 4) as *const u32).read_unaligned() };
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2 || unsafe { *value.get_unchecked(val_len - 1) } != b'"' {
                    return None;
                }
                let vp = value.as_ptr();
                let w0 = unsafe { (vp as *const u32).read_unaligned() };
                let w1 = unsafe { (vp.add(prefix_end - 4) as *const u32).read_unaligned() };
                if (w0 ^ q0) | (w1 ^ q1) != 0 {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        } else if (32..=48).contains(&prefix_end) {
            let qp = quoted_prefix.as_ptr();
            let q0 = unsafe { (qp as *const u128).read_unaligned() };
            let q1 = unsafe { (qp.add(16) as *const u128).read_unaligned() };
            let q2 = unsafe { (qp.add(prefix_end - 16) as *const u128).read_unaligned() };
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2 || unsafe { *value.get_unchecked(val_len - 1) } != b'"' {
                    return None;
                }
                let vp = value.as_ptr();
                let w0 = unsafe { (vp as *const u128).read_unaligned() };
                let w1 = unsafe { (vp.add(16) as *const u128).read_unaligned() };
                let w2 = unsafe { (vp.add(prefix_end - 16) as *const u128).read_unaligned() };
                if (w0 ^ q0) | (w1 ^ q1) | (w2 ^ q2) != 0 {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        } else {
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2
                    || unsafe { *value.get_unchecked(val_len - 1) } != b'"'
                    || !slice_eq_fast(unsafe { value.get_unchecked(..prefix_end) }, quoted_prefix)
                {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        }
        unsafe {
            sc.parsed_u64.set_len(n);
        }
        let width = bit_width(max.checked_sub(min)?);
        let size = 1
            + uleb_len(prefix.len())
            + prefix.len()
            + 8
            + 1
            + bit_bytes(n, width).ok()?;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DECIMAL_SUFFIX);
        put_bytes(&mut out, prefix);
        out.extend_from_slice(&min.to_le_bytes());
        out.push(width);
        push_bit_values(
            &mut out,
            n,
            width,
            sc.parsed_u64.iter().map(|&v| v - min),
        );
        Some(out)
    })
}

