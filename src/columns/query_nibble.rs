#[inline(always)]
fn lc_hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

/// Direct binary nibble scanner on `MODEL_HEX_SUFFIX` (`raw_slice` has `ones * width` raw bytes).
/// - When `collect_limit == None`, returns `Ok(Some(vec![0]))` if at least 1 row matches,
///   `Ok(Some(vec![]))` if 0 rows match, or `Ok(None)` if `needle` overlaps prefix/quote.
/// - When `collect_limit == Some(lim)`, returns up to `lim` sorted matching row indices `0..ones`!
fn hex_suffix_scan_nibble_hits(
    col_bytes: &[u8],
    ones: usize,
    needle: &[u8],
    collect_limit: Option<usize>,
) -> Result<Option<Vec<usize>>> {
    if col_bytes.first() != Some(&MODEL_HEX_SUFFIX) || needle.is_empty() {
        return Ok(None);
    }
    let mut c = Cursor::new(&col_bytes[1..]);
    let plen = c.uleb()?;
    let prefix = c.take(plen)?;
    let width_hdr = c.uleb()?;
    let (quoted, width) = if (1..=64).contains(&width_hdr) {
        (true, width_hdr)
    } else if (68..=128).contains(&width_hdr) {
        (false, width_hdr - 64)
    } else {
        return Err(invalid("column hex width"));
    };
    if plen + 2 > 128 {
        return Ok(None);
    }
    let mut lead_buf = [0u8; 130];
    let lead_len = if quoted {
        lead_buf[0] = b'"';
        lead_buf[1..1 + plen].copy_from_slice(prefix);
        1 + plen
    } else {
        lead_buf[..plen].copy_from_slice(prefix);
        plen
    };
    let lead = &lead_buf[..lead_len];
    if contains_subslice(lead, needle) {
        let lim = collect_limit.unwrap_or(1).min(ones);
        return Ok(Some((0..lim).collect()));
    }
    // Check if `needle` can overlap `lead` at start or `"` at end:
    let max_p = (needle.len() - 1).min(lead.len());
    for p in 1..=max_p {
        if lead.ends_with(&needle[..p]) {
            return Ok(None);
        }
    }
    let nlen = needle.len();
    if nlen > width * 2 + usize::from(quoted) {
        return Ok(Some(Vec::new()));
    }
    if quoted && needle.last() == Some(&b'"') {
        return Ok(None);
    }
    if nlen > width * 2 || nlen > 128 {
        return Ok(Some(Vec::new()));
    }
    let mut nibs = [0u8; 128];
    for (i, &b) in needle.iter().enumerate() {
        let Some(nb) = lc_hex_nibble(b) else {
            return Ok(Some(Vec::new()));
        };
        nibs[i] = nb;
    }
    if ones == 0 {
        return Ok(Some(Vec::new()));
    }
    let raw_total = ones.checked_mul(width).ok_or(Error::SizeOverflow)?;
    let raw_slice = c.take(raw_total)?;

    let want_only_bool = collect_limit.is_none();
    let lim = collect_limit.unwrap_or(1);
    let mut hits = Vec::new();

    if nlen >= 4 {
        let mut even = [0u8; 64];
        let even_len = nlen / 2;
        for k in 0..even_len {
            even[k] = (nibs[2 * k] << 4) | nibs[2 * k + 1];
        }
        let even_sub = &even[..even_len];
        let even_row_bytes = (nlen + 1) / 2;

        let mut odd = [0u8; 64];
        let odd_len = (nlen - 1) / 2;
        for k in 0..odd_len {
            odd[k] = (nibs[2 * k + 1] << 4) | nibs[2 * k + 2];
        }
        let odd_sub = &odd[..odd_len];
        let odd_row_bytes = 1 + nlen / 2;

        let mut pos = 0usize;
        while pos + even_len <= raw_slice.len() {
            let Some(rel) = find_byte(even_sub[0], &raw_slice[pos..]) else {
                break;
            };
            let at = pos + rel;
            if at + even_len > raw_slice.len() {
                break;
            }
            if raw_slice[at..at + even_len] == *even_sub {
                let col_off = at % width;
                if col_off + even_row_bytes <= width
                    && (nlen % 2 == 0 || (raw_slice[at + even_len] >> 4) == nibs[nlen - 1])
                {
                    if want_only_bool {
                        return Ok(Some(vec![at / width]));
                    }
                    hits.push(at / width);
                }
            }
            pos = at + 1;
        }

        let mut opos = 1usize;
        while opos + odd_len <= raw_slice.len() {
            let Some(rel) = find_byte(odd_sub[0], &raw_slice[opos..]) else {
                break;
            };
            let at = opos + rel;
            if at + odd_len > raw_slice.len() {
                break;
            }
            if raw_slice[at..at + odd_len] == *odd_sub {
                let start_b = at - 1;
                let col_off = start_b % width;
                if col_off + odd_row_bytes <= width
                    && (raw_slice[start_b] & 0x0f) == nibs[0]
                    && (nlen % 2 == 1 || (raw_slice[at + odd_len] >> 4) == nibs[nlen - 1])
                {
                    if want_only_bool {
                        return Ok(Some(vec![start_b / width]));
                    }
                    hits.push(start_b / width);
                }
            }
            opos = at + 1;
        }

        if hits.len() > 1 {
            hits.sort_unstable();
            hits.dedup();
        }
        if hits.len() > lim {
            hits.truncate(lim);
        }
        return Ok(Some(hits));
    }

    // Short hex needle (`nlen` in `1..=3`): scan per row nibbles directly.
    for r in 0..ones {
        let rb = &raw_slice[r * width..(r + 1) * width];
        let hex_len = width * 2;
        let mut matched = false;
        'nstart: for s in 0..=(hex_len - nlen) {
            for k in 0..nlen {
                let idx = s + k;
                let nib = if (idx & 1) == 0 {
                    rb[idx >> 1] >> 4
                } else {
                    rb[idx >> 1] & 0x0f
                };
                if nib != nibs[k] {
                    continue 'nstart;
                }
            }
            matched = true;
            break;
        }
        if matched {
            hits.push(r);
            if want_only_bool || hits.len() >= lim {
                return Ok(Some(hits));
            }
        }
    }
    Ok(Some(hits))
}

/// Positional fixed-width / alphabet check on `MODEL_PATTERN` when `needle` is isolated to one slot.
fn pattern_can_contain_isolated(mut c: Cursor<'_>, needle: &[u8]) -> Result<bool> {
    if needle.is_empty() {
        return Ok(true);
    }
    let ntok = c.uleb()?;
    if !(1..=PATTERN_MAX_TOKENS).contains(&ntok) {
        return Err(invalid("column pattern tokens"));
    }
    let mut pos_masks = [[0u64; 4]; 128];
    let mut pos_len = 0usize;
    let mut all_fixed = true;
    let mut max_total_len = 0usize;
    let mut char_mask = [0u64; 4];

    for _ in 0..ntok {
        match c.byte()? {
            0 => {
                let width = c.byte()? as usize;
                let min = c.uleb()? as u64;
                let radix = c.uleb()? as u64;
                if !(1..=PATTERN_MAX_DIGITS).contains(&width) || radix == 0 {
                    return Err(invalid("column pattern digits"));
                }
                let hi = min.saturating_add(radix - 1);
                max_total_len += width;
                add_bytes_to_mask(&mut char_mask, b"0123456789");
                if all_fixed && pos_len + width <= 128 && decimal_digits_u64(hi) <= width {
                    let mut lo_b = [0u8; 24];
                    let mut hi_b = [0u8; 24];
                    format_padded_u64_stack(min, width, &mut lo_b);
                    format_padded_u64_stack(hi, width, &mut hi_b);
                    let mut lcp = 0usize;
                    while lcp < width && lo_b[lcp] == hi_b[lcp] {
                        lcp += 1;
                    }
                    for p in 0..width {
                        if p < lcp {
                            add_bytes_to_mask(&mut pos_masks[pos_len + p], &lo_b[p..p + 1]);
                        } else if p == lcp {
                            for d in lo_b[lcp]..=hi_b[lcp] {
                                add_bytes_to_mask(&mut pos_masks[pos_len + p], &[d]);
                            }
                        } else {
                            add_bytes_to_mask(&mut pos_masks[pos_len + p], b"0123456789");
                        }
                    }
                    pos_len += width;
                } else {
                    all_fixed = false;
                }
            }
            1 => {
                let len = c.uleb()?;
                let lit = c.take(len)?;
                max_total_len += len;
                add_bytes_to_mask(&mut char_mask, lit);
                if all_fixed && pos_len + len <= 128 {
                    for (p, &b) in lit.iter().enumerate() {
                        add_bytes_to_mask(&mut pos_masks[pos_len + p], &[b]);
                    }
                    pos_len += len;
                } else {
                    all_fixed = false;
                }
            }
            2 => {
                let entries = c.uleb()?;
                if !(2..=PATTERN_MAX_DICT).contains(&entries) {
                    return Err(invalid("column pattern dictionary"));
                }
                let mut fixed_w: Option<usize> = None;
                let mut same_w = true;
                let mut max_e = 0usize;
                let base_pos = pos_len;
                for _ in 0..entries {
                    let len = c.uleb()?;
                    let e = c.take(len)?;
                    add_bytes_to_mask(&mut char_mask, e);
                    max_e = max_e.max(len);
                    match fixed_w {
                        None => fixed_w = Some(len),
                        Some(w) if w != len => same_w = false,
                        _ => {}
                    }
                    if all_fixed && same_w && base_pos + len <= 128 {
                        for (p, &b) in e.iter().enumerate() {
                            add_bytes_to_mask(&mut pos_masks[base_pos + p], &[b]);
                        }
                    }
                }
                max_total_len += max_e;
                if all_fixed && same_w && base_pos + fixed_w.unwrap_or(0) <= 128 {
                    pos_len += fixed_w.unwrap_or(0);
                } else {
                    all_fixed = false;
                }
            }
            _ => return Err(invalid("column pattern token kind")),
        }
    }
    if needle.len() > max_total_len || !needle.iter().all(|&b| mask_has_byte(&char_mask, b)) {
        return Ok(false);
    }
    if all_fixed {
        if needle.len() > pos_len {
            return Ok(false);
        }
        let max_start = pos_len - needle.len();
        's: for start in 0..=max_start {
            for (k, &b) in needle.iter().enumerate() {
                if !mask_has_byte(&pos_masks[start + k], b) {
                    continue 's;
                }
            }
            return Ok(true);
        }
        return Ok(false);
    }
    Ok(true)
}

/// Evaluate isolated substring `needle` on ONLY `col_bytes` (`ones` rows), collecting up to `rem`
/// matching template-row indices without unpacking any of the other `slots - 1` columns.
fn eval_substr_on_single_slot<'a>(
    col_bytes: &'a [u8],
    ones: usize,
    rem: usize,
    needle: &[u8],
    scratch: &mut DecodeScratch,
) -> Result<(Option<PreparedColumn<'a>>, Vec<usize>)> {
    if rem == 0 || ones == 0 {
        return Ok((None, Vec::new()));
    }
    if let Some(hits) = hex_suffix_scan_nibble_hits(col_bytes, ones, needle, Some(rem))? {
        return Ok((None, hits));
    }
    let col = PreparedColumn::parse_and_validate(col_bytes, ones, scratch)?;
    let mut hits = Vec::new();
    match col.const_full {
        ConstValue::Slice(s) => {
            if contains_subslice(s, needle) {
                hits.extend(0..ones.min(rem));
            }
            return Ok((Some(col), hits));
        }
        ConstValue::Inline { ref buf, len } => {
            if contains_subslice(&buf[..len as usize], needle) {
                hits.extend(0..ones.min(rem));
            }
            return Ok((Some(col), hits));
        }
        ConstValue::QuotedDecimal {
            prefix,
            ref buf,
            len,
        } => {
            let mut tmp = Vec::with_capacity(prefix.len() + len as usize + 2);
            tmp.push(b'"');
            tmp.extend_from_slice(prefix);
            tmp.extend_from_slice(&buf[..len as usize]);
            tmp.push(b'"');
            if contains_subslice(&tmp, needle) {
                hits.extend(0..ones.min(rem));
            }
            return Ok((Some(col), hits));
        }
        ConstValue::None => {}
    }

    if let ColumnStream::Dictionary { u8_offset, .. } = col.stream {
        if let Some((slice_off, entries)) = col.dict_slices {
            let mut valid = vec![false; entries];
            let mut match_count = 0usize;
            let mut single_id = 0u8;
            for (e, slot_ok) in valid.iter_mut().enumerate() {
                let (ptr, len) = scratch.slice_arena[slice_off + e];
                let eb = unsafe { std::slice::from_raw_parts(ptr, len) };
                if contains_subslice(eb, needle) {
                    *slot_ok = true;
                    match_count += 1;
                    single_id = e as u8;
                }
            }
            if match_count == 0 {
                return Ok((Some(col), hits));
            }
            if match_count == entries {
                hits.extend(0..ones.min(rem));
                return Ok((Some(col), hits));
            }
            let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];
            if match_count == 1 && entries <= 256 {
                let mut pos = 0usize;
                while pos < ones {
                    let Some(rel) = find_byte(single_id, &ids[pos..]) else {
                        break;
                    };
                    let t = pos + rel;
                    hits.push(t);
                    if hits.len() >= rem {
                        break;
                    }
                    pos = t + 1;
                }
            } else {
                for (t, &id) in ids.iter().enumerate() {
                    if *valid.get(id as usize).unwrap_or(&false) {
                        hits.push(t);
                        if hits.len() >= rem {
                            break;
                        }
                    }
                }
            }
            return Ok((Some(col), hits));
        }
    }

    let (str_base, unbound_base, exec_base, piece_base) = (
        scratch.str_arena.len(),
        scratch.unbound_col_arena.len(),
        scratch.exec_col_arena.len(),
        scratch.piece_arena.len(),
    );
    let unbound = UnboundTemplate::build(&[b"", b""], &[col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let mut val_buf = vec![0u8; tpl.row_budget.max(512) + 64];
    for t in 0..ones {
        let written = unsafe {
            let sp = val_buf.as_mut_ptr();
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        if contains_subslice(&val_buf[..written], needle) {
            hits.push(t);
            if hits.len() >= rem {
                break;
            }
        }
    }
    scratch.str_arena.truncate(str_base);
    scratch.unbound_col_arena.truncate(unbound_base);
    scratch.exec_col_arena.truncate(exec_base);
    scratch.piece_arena.truncate(piece_base);
    Ok((Some(col), hits))
}
