#[inline(always)]
fn push_nibble_bytes(out: &mut Vec<u8>, s: &[u8], lut_ptr: *const u8) {
    let len = s.len();
    if len == 0 {
        return;
    }
    let needed = (len + 1) >> 1;
    let old_len = out.len();
    out.reserve(needed);
    unsafe {
        let src = s.as_ptr();
        let mut dst = out.as_mut_ptr().add(old_len);
        let mut j = 0usize;
        while j + 1 < len {
            let hi = *lut_ptr.add(*src.add(j) as usize);
            let lo = *lut_ptr.add(*src.add(j + 1) as usize);
            *dst = (hi << 4) | lo;
            dst = dst.add(1);
            j += 2;
        }
        if j < len {
            let hi = *lut_ptr.add(*src.add(j) as usize);
            *dst = hi << 4;
        }
        out.set_len(old_len + needed);
    }
}

#[inline(always)]
fn push_5bit_bytes(out: &mut Vec<u8>, s: &[u8], lut_ptr: *const u8) {
    let mut acc = 0u64;
    let mut bits = 0u32;
    for &b in s {
        let v = unsafe { *lut_ptr.add(b as usize) as u64 };
        acc |= v << bits;
        bits += 5;
        if bits >= 32 {
            out.extend_from_slice(&(acc as u32).to_le_bytes());
            acc >>= 32;
            bits -= 32;
        }
    }
    while bits > 0 {
        out.push(acc as u8);
        acc >>= 8;
        bits = bits.saturating_sub(8);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_front_dict_packet(
    u: usize,
    n: usize,
    width: u8,
    packed_len: usize,
    plain_dict_bytes: usize,
    entries: &[&[u8]],
    ranks: &[u8],
    limit: usize,
    allow_front: bool,
) -> Option<Vec<u8>> {
    if !allow_front || u < 2 || u > 256 {
        if u >= n {
            return None;
        }
        let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DICTIONARY);
        put_uleb(&mut out, u);
        for &entry in entries {
            put_bytes(&mut out, entry);
        }
        out.push(width);
        push_bit_values(&mut out, n, width, ranks.iter().map(|&r| r as u64));
        return Some(out);
    }

    // 1. Precalculate front+back delta dictionary stats and 256-bit char masks over `mid` in-place:
    const HEX_M0: u64 = 0x03ff_0000_0000_0000; // '0'..='9' (48..=57)
    const HEX_M1: u64 = 0x0000_007e_0000_0000; // 'a'..='f' (97..=102)
    let mut lcp_lcs_uninit = [core::mem::MaybeUninit::<(u32, u32)>::uninit(); 256];
    let mut is_hex_mid_uninit = [core::mem::MaybeUninit::<bool>::uninit(); 256];
    let lcp_lcs_ptr = lcp_lcs_uninit.as_mut_ptr().cast::<(u32, u32)>();
    let is_hex_mid_ptr = is_hex_mid_uninit.as_mut_ptr().cast::<bool>();
    let first = unsafe { *entries.get_unchecked(0) };
    let mut front_dict_bytes = uleb_len(first.len()) + first.len();
    let mut all_4bit_saved = 0usize;
    let mut all_5bit_saved = 0usize;
    let mut hex_nibble_bytes = uleb_len(first.len()) + first.len();
    let mut char_mask = [0u64; 4];
    let mut hex_mask = [0u64; 2];
    let mut prev = first;
    for i in 1..u {
        let cur = unsafe { *entries.get_unchecked(i) };
        let (lcp, lcs) = common_prefix_suffix(prev, cur);
        unsafe {
            *lcp_lcs_ptr.add(i) = (lcp as u32, lcs as u32);
        }
        let mid = &cur[lcp..cur.len() - lcs];
        let base_hdr = uleb_len(lcp) + uleb_len(lcs);
        front_dict_bytes += base_hdr + uleb_len(mid.len()) + mid.len();
        all_4bit_saved += mid.len() >> 1;
        all_5bit_saved += mid.len() - ((mid.len() * 5 + 7) >> 3);
        let mut m0 = 0u64;
        let mut m1 = 0u64;
        let mut m_hi = 0u8;
        for &b in mid {
            let bit = 1u64 << (b & 63);
            if b < 64 {
                m0 |= bit;
            } else {
                m1 |= bit;
            }
            m_hi |= b;
        }
        char_mask[0] |= m0;
        if (m_hi & 0x80) == 0 {
            char_mask[1] |= m1;
        } else {
            m1 = 0;
            for &b in mid {
                if b >= 64 {
                    char_mask[(b >> 6) as usize] |= 1u64 << (b & 63);
                }
            }
        }
        if mid.len() >= 2 && ((m0 & !HEX_M0) | (m1 & !HEX_M1) | ((m_hi & 0x80) as u64)) == 0 {
            unsafe {
                *is_hex_mid_ptr.add(i) = true;
            }
            hex_mask[0] |= m0;
            hex_mask[1] |= m1;
            hex_nibble_bytes += base_hdr + uleb_len(mid.len() << 1) + ((mid.len() + 1) >> 1);
        } else {
            unsafe {
                *is_hex_mid_ptr.add(i) = false;
            }
            hex_nibble_bytes += base_hdr + uleb_len((mid.len() << 1) | 1) + mid.len();
        }
        prev = cur;
    }

    // Choose best dictionary encoding (`dict_mode`: 0 = plain, 1 = front+back delta,
    // 2 = FAST_LZ front+back delta, 3 = 4-bit/5-bit palette-packed front+back delta):
    let mut dict_mode = if front_dict_bytes < plain_dict_bytes || u == n {
        1u8
    } else {
        0u8
    };
    let mut chosen_dict_bytes = if u == n {
        front_dict_bytes
    } else {
        front_dict_bytes.min(plain_dict_bytes)
    };
    let mut pal = [0u8; 32];
    let mut pal_len = 0usize;
    let mut pal_mixed = false;
    let all_pop = (char_mask[0].count_ones()
        + char_mask[1].count_ones()
        + char_mask[2].count_ones()
        + char_mask[3].count_ones()) as usize;
    if all_pop >= 1 && all_pop <= 32 {
        let saved = if all_pop <= 16 {
            all_4bit_saved
        } else {
            all_5bit_saved
        };
        if saved > all_pop + 4 {
            let packed_dict_bytes = 1 + all_pop + front_dict_bytes - saved;
            if packed_dict_bytes + 4 < chosen_dict_bytes {
                for word_idx in 0..4 {
                    let mut w = char_mask[word_idx];
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        pal[pal_len] = ((word_idx << 6) | bit) as u8;
                        pal_len += 1;
                    }
                }
                dict_mode = 3;
                chosen_dict_bytes = packed_dict_bytes;
            }
        }
    } else {
        let hex_pop = (hex_mask[0].count_ones() + hex_mask[1].count_ones()) as usize;
        if hex_pop >= 1 && hex_pop <= 16 {
            let nibble_dict_bytes = 1 + hex_pop + hex_nibble_bytes;
            if nibble_dict_bytes + 8 < chosen_dict_bytes {
                for word_idx in 0..2 {
                    let mut w = hex_mask[word_idx];
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        pal[pal_len] = ((word_idx << 6) | bit) as u8;
                        pal_len += 1;
                    }
                }
                pal_mixed = true;
                dict_mode = 3;
                chosen_dict_bytes = nibble_dict_bytes;
            }
        }
    }

    // 2. Choose best index encoding (`idx_mode`: 0 = plain, 1 = repeat-prev, 2 = mode-exc, 3 = mode + non-mode repeat):
    let mut idx_mode = 0u8;
    let mut chosen_idx_bytes = packed_len;
    let mut mode_rank = 0u8;
    let mut exc1 = 0usize;
    let mut rem2 = 0usize;
    let mut exc3 = 0usize;
    let exc_bound = u - 1;
    let exc_width = bit_width(exc_bound.saturating_sub(1) as u64);

    if packed_len >= 24 && u < n {
        let mut freq4_uninit = [core::mem::MaybeUninit::<[u32; 4]>::uninit(); 256];
        let freq4: &mut [[u32; 4]] = unsafe {
            let ptr = freq4_uninit.as_mut_ptr().cast::<[u32; 4]>();
            let s = std::slice::from_raw_parts_mut(ptr, u);
            s.fill([0u32; 4]);
            s
        };
        let mut same_prev = 0usize;
        let mut prev_r = u8::MAX;
        let r_ptr = ranks.as_ptr();
        let mut idx = 0usize;
        while idx + 4 <= n {
            unsafe {
                let r0 = *r_ptr.add(idx);
                let r1 = *r_ptr.add(idx + 1);
                let r2 = *r_ptr.add(idx + 2);
                let r3 = *r_ptr.add(idx + 3);
                freq4.get_unchecked_mut(r0 as usize)[0] += 1;
                freq4.get_unchecked_mut(r1 as usize)[1] += 1;
                freq4.get_unchecked_mut(r2 as usize)[2] += 1;
                freq4.get_unchecked_mut(r3 as usize)[3] += 1;
                same_prev += (r0 == prev_r) as usize
                    + (r1 == r0) as usize
                    + (r2 == r1) as usize
                    + (r3 == r2) as usize;
                prev_r = r3;
            }
            idx += 4;
        }
        while idx < n {
            unsafe {
                let r = *r_ptr.add(idx);
                freq4.get_unchecked_mut(r as usize)[0] += 1;
                same_prev += (r == prev_r) as usize;
                prev_r = r;
            }
            idx += 1;
        }
        let mut mode_count = 0usize;
        for (r, c4) in freq4.iter().enumerate() {
            let c = (c4[0] + c4[1] + c4[2] + c4[3]) as usize;
            if c > mode_count {
                mode_count = c;
                mode_rank = r as u8;
            }
        }
        let bm_n_bytes = (n + 7) / 8;
        exc1 = n - same_prev;
        let idx1_bytes = bm_n_bytes + (exc1 * (width as usize) + 7) / 8;
        rem2 = n - mode_count;
        let idx2_bytes =
            uleb_len(mode_rank as usize) + bm_n_bytes + (rem2 * (exc_width as usize) + 7) / 8;

        if idx1_bytes + 8 < chosen_idx_bytes {
            idx_mode = 1;
            chosen_idx_bytes = idx1_bytes;
        }
        if idx2_bytes + 8 < chosen_idx_bytes {
            idx_mode = 2;
            chosen_idx_bytes = idx2_bytes;
        }
        if exc_width >= 2 && rem2 * (exc_width as usize) > rem2 + 64 {
            let mut non_mode_rep = 0usize;
            let mut last_nm = u8::MAX;
            for &r in ranks {
                if r != mode_rank {
                    if r == last_nm {
                        non_mode_rep += 1;
                    } else {
                        last_nm = r;
                    }
                }
            }
            exc3 = rem2 - non_mode_rep;
            let idx3_bytes = uleb_len(mode_rank as usize)
                + bm_n_bytes
                + (rem2 + 7) / 8
                + (exc3 * (exc_width as usize) + 7) / 8;
            if idx3_bytes + 8 < chosen_idx_bytes {
                idx_mode = 3;
                chosen_idx_bytes = idx3_bytes;
            }
        }
    }

    let mut built_front_in_scratch = false;
    if dict_mode != 3
        && u >= 16
        && front_dict_bytes >= 512
        && front_dict_bytes <= 65536
        && 1 + uleb_len(u) + 1 + chosen_idx_bytes + 32 < limit
    {
        MODEL_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            let ModelScratch {
                front_buf,
                front_lz,
                ..
            } = &mut *sc;
            front_buf.clear();
            front_buf.reserve(front_dict_bytes);
            put_bytes(front_buf, first);
            for i in 1..u {
                let (lcp, lcs) = unsafe {
                    let p = *lcp_lcs_ptr.add(i);
                    (p.0 as usize, p.1 as usize)
                };
                let cur = unsafe { *entries.get_unchecked(i) };
                let mid = &cur[lcp..cur.len() - lcs];
                put_uleb(front_buf, lcp);
                put_uleb(front_buf, lcs);
                put_bytes(front_buf, mid);
            }
            built_front_in_scratch = true;
            front_lz.clear();
            front_lz.reserve(front_dict_bytes / 2);
            if pack_fast_lz_into(front_buf, front_lz).is_some() {
                let lz_total = uleb_len(front_lz.len()) + front_lz.len();
                if lz_total + 32 + (front_dict_bytes >> 3) < chosen_dict_bytes {
                    dict_mode = 2;
                    chosen_dict_bytes = lz_total;
                }
            }
        });
    }

    if dict_mode == 0 && idx_mode == 0 {
        if u >= n {
            return None;
        }
        let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DICTIONARY);
        put_uleb(&mut out, u);
        for &entry in entries {
            put_bytes(&mut out, entry);
        }
        out.push(width);
        push_bit_values(&mut out, n, width, ranks.iter().map(|&r| r as u64));
        return Some(out);
    }

    let size = 1 + uleb_len(u) + 1 + chosen_dict_bytes + chosen_idx_bytes;
    if size >= limit {
        return None;
    }
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_FRONT_DICT);
    put_uleb(&mut out, u);
    out.push(dict_mode | (idx_mode << 2));
    match dict_mode {
        0 => {
            for &entry in entries {
                put_bytes(&mut out, entry);
            }
        }
        1 => {
            if built_front_in_scratch {
                MODEL_SCRATCH.with(|cell| {
                    let sc = cell.borrow();
                    out.extend_from_slice(&sc.front_buf);
                });
            } else {
                put_bytes(&mut out, first);
                for i in 1..u {
                    let (lcp, lcs) = unsafe {
                        let p = *lcp_lcs_ptr.add(i);
                        (p.0 as usize, p.1 as usize)
                    };
                    let cur = unsafe { *entries.get_unchecked(i) };
                    let mid = &cur[lcp..cur.len() - lcs];
                    put_uleb(&mut out, lcp);
                    put_uleb(&mut out, lcs);
                    put_bytes(&mut out, mid);
                }
            }
        }
        2 => {
            MODEL_SCRATCH.with(|cell| {
                let sc = cell.borrow();
                put_uleb(&mut out, sc.front_lz.len());
                out.extend_from_slice(&sc.front_lz);
            });
        }
        3 => {
            let mut lut_uninit = [core::mem::MaybeUninit::<u8>::uninit(); 256];
            let lut_ptr = lut_uninit.as_mut_ptr().cast::<u8>();
            for (idx, &b) in pal[..pal_len].iter().enumerate() {
                unsafe {
                    *lut_ptr.add(b as usize) = idx as u8;
                }
            }
            out.push((pal_len as u8) | if pal_mixed { 0x80 } else { 0 });
            out.extend_from_slice(&pal[..pal_len]);
            put_bytes(&mut out, first);
            if !pal_mixed {
                for i in 1..u {
                    let (lcp, lcs) = unsafe {
                        let p = *lcp_lcs_ptr.add(i);
                        (p.0 as usize, p.1 as usize)
                    };
                    let cur = unsafe { *entries.get_unchecked(i) };
                    let mid = &cur[lcp..cur.len() - lcs];
                    put_uleb(&mut out, lcp);
                    put_uleb(&mut out, lcs);
                    put_uleb(&mut out, mid.len());
                    if pal_len <= 16 {
                        push_nibble_bytes(&mut out, mid, lut_ptr);
                    } else {
                        push_5bit_bytes(&mut out, mid, lut_ptr);
                    }
                }
            } else {
                for i in 1..u {
                    let (lcp, lcs) = unsafe {
                        let p = *lcp_lcs_ptr.add(i);
                        (p.0 as usize, p.1 as usize)
                    };
                    let cur = unsafe { *entries.get_unchecked(i) };
                    let mid = &cur[lcp..cur.len() - lcs];
                    put_uleb(&mut out, lcp);
                    put_uleb(&mut out, lcs);
                    if unsafe { *is_hex_mid_ptr.add(i) } {
                        put_uleb(&mut out, mid.len() << 1);
                        push_nibble_bytes(&mut out, mid, lut_ptr);
                    } else {
                        put_uleb(&mut out, (mid.len() << 1) | 1);
                        out.extend_from_slice(mid);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    match idx_mode {
        0 => {
            push_bit_values(&mut out, n, width, ranks.iter().map(|&r| r as u64));
        }
        1 => {
            push_bit_values(
                &mut out,
                n,
                1,
                (0..n).map(|i| u64::from(i > 0 && ranks[i] == ranks[i - 1])),
            );
            push_bit_values(
                &mut out,
                exc1,
                width,
                (0..n).filter_map(|i| {
                    if i == 0 || ranks[i] != ranks[i - 1] {
                        Some(ranks[i] as u64)
                    } else {
                        None
                    }
                }),
            );
        }
        2 => {
            put_uleb(&mut out, mode_rank as usize);
            push_bit_values(
                &mut out,
                n,
                1,
                ranks.iter().map(|&r| u64::from(r == mode_rank)),
            );
            push_bit_values(
                &mut out,
                rem2,
                exc_width,
                ranks.iter().filter_map(|&r| {
                    if r == mode_rank {
                        None
                    } else if r < mode_rank {
                        Some(r as u64)
                    } else {
                        Some((r - 1) as u64)
                    }
                }),
            );
        }
        3 => {
            put_uleb(&mut out, mode_rank as usize);
            push_bit_values(
                &mut out,
                n,
                1,
                ranks.iter().map(|&r| u64::from(r == mode_rank)),
            );
            let mut prev_nm = u8::MAX;
            push_bit_values(
                &mut out,
                rem2,
                1,
                ranks.iter().filter_map(|&r| {
                    if r == mode_rank {
                        None
                    } else if r == prev_nm {
                        Some(1u64)
                    } else {
                        prev_nm = r;
                        Some(0u64)
                    }
                }),
            );
            let mut prev_nm2 = u8::MAX;
            push_bit_values(
                &mut out,
                exc3,
                exc_width,
                ranks.iter().filter_map(|&r| {
                    if r == mode_rank || r == prev_nm2 {
                        None
                    } else {
                        prev_nm2 = r;
                        Some(if r < mode_rank {
                            r as u64
                        } else {
                            (r - 1) as u64
                        })
                    }
                }),
            );
        }
        _ => unreachable!(),
    }
    Some(out)
}

