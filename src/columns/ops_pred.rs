#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotLookup {
    Slot(usize),
    Absent,
    Fallback,
}

fn json_single_line_template_slot(segs: &[&[u8]], target_key: &[u8]) -> Option<Option<usize>> {
    let slots = segs.len().saturating_sub(1);
    if slots == 0 {
        return None;
    }
    let mut matched_slot = None;
    for s in 0..slots {
        let seg = segs[s];
        if seg.contains(&b'\n') {
            return None;
        }
        let mut p = 0usize;
        skip_space(seg, &mut p);
        if seg.get(p) != Some(if s == 0 { &b'{' } else { &b',' }) {
            return None;
        }
        p += 1;
        skip_space(seg, &mut p);
        let (k_start, k_end) = (p, scan_string(seg, p)?);
        p = k_end;
        skip_space(seg, &mut p);
        if seg.get(p) != Some(&b':') {
            return None;
        }
        p += 1;
        skip_space(seg, &mut p);
        if p != seg.len() {
            return None;
        }
        if matched_slot.is_none()
            && crate::ops::unescape_json_str(&seg[k_start..k_end])?.as_ref() == target_key
        {
            matched_slot = Some(s);
        }
    }
    let last_body = segs[slots]
        .strip_suffix(b"\r\n")
        .or_else(|| segs[slots].strip_suffix(b"\n"))
        .unwrap_or(segs[slots]);
    if last_body.contains(&b'\n') {
        return None;
    }
    let mut lp = 0usize;
    skip_space(last_body, &mut lp);
    if last_body.get(lp) != Some(&b'}') {
        return None;
    }
    lp += 1;
    skip_space(last_body, &mut lp);
    if last_body.get(lp) == Some(&b',') {
        lp += 1;
        skip_space(last_body, &mut lp);
    }
    (lp == last_body.len()).then_some(matched_slot)
}

fn resolve_template_slot(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    col: crate::ops::ColTarget<'_>,
) -> Result<SlotLookup> {
    let slots = col_slices.len();
    match col {
        crate::ops::ColTarget::JsonKey(key) => {
            if let Some(found) = json_single_line_template_slot(segs, key.as_bytes()) {
                return Ok(match found {
                    Some(s) => SlotLookup::Slot(s),
                    None => SlotLookup::Absent,
                });
            }
            if !can_template_match_fast(segs, col_slices, b"{")? {
                return Ok(SlotLookup::Absent);
            }
            if !key.is_empty()
                && !key.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20)
                && !can_template_match_fast(segs, col_slices, key.as_bytes())?
            {
                return Ok(SlotLookup::Absent);
            }
            Ok(SlotLookup::Fallback)
        }
        crate::ops::ColTarget::Delimited { delim, index } => {
            if delimited_template_delim(segs) == Some(delim) {
                return Ok(if index < slots {
                    SlotLookup::Slot(index)
                } else {
                    SlotLookup::Absent
                });
            }
            if !can_template_match_fast(segs, col_slices, &[delim])? {
                return Ok(SlotLookup::Absent);
            }
            Ok(SlotLookup::Fallback)
        }
    }
}

/// Extract a conservative 256-bit byte-alphabet mask from a single compressed column slice.
fn col_bytes_alphabet_mask(col_bytes: &[u8]) -> Result<Option<[u64; 4]>> {
    let mut c = Cursor::new(col_bytes);
    let tag = c.byte()?;
    let mut m = [0u64; 4];
    match tag {
        MODEL_RAW => {
            while !c.done() {
                let len = c.uleb()?;
                add_bytes_to_mask(&mut m, c.take(len)?);
            }
        }
        MODEL_DICTIONARY => {
            let entries_len = c.uleb()?;
            for _ in 0..entries_len {
                let len = c.uleb()?;
                add_bytes_to_mask(&mut m, c.take(len)?);
            }
        }
        MODEL_FRONT_DICT => {
            let entries_len = c.uleb()?;
            let dict_mode = c.byte()? & 0x03;
            if dict_mode == 0 {
                for _ in 0..entries_len {
                    let len = c.uleb()?;
                    add_bytes_to_mask(&mut m, c.take(len)?);
                }
            } else if dict_mode == 1 {
                let len0 = c.uleb()?;
                add_bytes_to_mask(&mut m, c.take(len0)?);
                for _ in 1..entries_len {
                    let _ = c.uleb()?;
                    let _ = c.uleb()?;
                    let mid_len = c.uleb()?;
                    add_bytes_to_mask(&mut m, c.take(mid_len)?);
                }
            } else if dict_mode == 2 {
                let lz_len = c.uleb()?;
                let lz_bytes = c.take(lz_len)?;
                if lz_bytes.first() != Some(&FAST_LZ_PACKET)
                    || !fast_lz_alphabet_mask(Cursor::new(&lz_bytes[1..]), &mut m)
                {
                    return Ok(None);
                }
            } else if dict_mode == 3 {
                let (pal, pal_len, _) = read_front_dict_palette(&mut c)?;
                let len0 = c.uleb()?;
                add_bytes_to_mask(&mut m, &pal[..pal_len as usize]);
                add_bytes_to_mask(&mut m, c.take(len0)?);
            } else {
                return Ok(None);
            }
        }
        MODEL_LZ_RAW => {
            if !fast_lz_alphabet_mask(c, &mut m) {
                return Ok(None);
            }
        }
        MODEL_DECIMAL_AFFINE | MODEL_DECIMAL_PACKED | MODEL_DECIMAL_DELTA => {
            add_bytes_to_mask(&mut m, b"0123456789");
        }
        MODEL_DECIMAL_SUFFIX => {
            let plen = c.uleb()?;
            add_bytes_to_mask(&mut m, c.take(plen)?);
            add_bytes_to_mask(&mut m, b"\"0123456789");
        }
        MODEL_HEX_SUFFIX | MODEL_EPOCH_COUNTER | MODEL_GF2_AFFINE => {
            let plen = c.uleb()?;
            add_bytes_to_mask(&mut m, c.take(plen)?);
            let is_quoted = if tag == MODEL_HEX_SUFFIX { c.uleb()? <= 64 } else { true };
            add_bytes_to_mask(&mut m, b"0123456789abcdef");
            if is_quoted {
                add_bytes_to_mask(&mut m, b"\"");
            }
        }
        MODEL_TIMESTAMP => add_bytes_to_mask(&mut m, b"\"0123456789-:TZ"),
        MODEL_PATTERN => pattern_col_alphabet(&mut c, &mut m)?,
        _ => return Ok(None),
    }
    Ok(Some(m))
}

fn can_col_match_pred_fast(
    col_bytes: &[u8],
    ones: usize,
    pred: &crate::ops::ColPred<'_>,
) -> Result<bool> {
    if ones == 0 {
        return Ok(false);
    }
    match pred.pred {
        crate::ops::ValuePred::Num(cmp) => {
            if !can_numeric_col_match_fast(col_bytes, ones, cmp) {
                return Ok(false);
            }
            if let Some(m) = col_bytes_alphabet_mask(col_bytes)? {
                if !(b'0'..=b'9').any(|d| mask_has_byte(&m, d)) {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        crate::ops::ValuePred::Str(sc) => {
            let Some(m) = col_bytes_alphabet_mask(col_bytes)? else {
                return Ok(true);
            };
            if matches!(pred.col, crate::ops::ColTarget::JsonKey(_)) && !mask_has_byte(&m, b'"') {
                return Ok(false);
            }
            let req = sc.required_slice();
            if !req.is_empty() && !mask_has_byte(&m, b'\\') {
                for &b in req {
                    if !mask_has_byte(&m, b) {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
    }
}

/// Evaluate `pred` on single compressed column `col_bytes` (`ones` template rows), decoding ONLY
/// that column and returning up to `rem` matching template-row indices (`0..ones`).
fn eval_col_pred_on_slot<'a>(
    col_bytes: &'a [u8],
    ones: usize,
    rem: usize,
    pred: &crate::ops::ColPred<'_>,
    scratch: &mut DecodeScratch,
) -> Result<(Option<PreparedColumn<'a>>, Vec<usize>)> {
    if rem == 0 || !can_col_match_pred_fast(col_bytes, ones, pred)? {
        return Ok((None, Vec::new()));
    }
    let col = PreparedColumn::parse_and_validate(col_bytes, ones, scratch)?;
    let mut hits = Vec::new();

    match col.const_full {
        ConstValue::Slice(s) => {
            if crate::ops::eval_token_pred(pred.col, s, pred.pred) {
                hits.extend(0..ones.min(rem));
            }
            return Ok((Some(col), hits));
        }
        ConstValue::Inline { ref buf, len } => {
            if crate::ops::eval_token_pred(pred.col, &buf[..len as usize], pred.pred) {
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
            if crate::ops::eval_token_pred(pred.col, &tmp, pred.pred) {
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
                let entry_bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
                if crate::ops::eval_token_pred(pred.col, entry_bytes, pred.pred) {
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

    if let crate::ops::ValuePred::Num(cmp) = pred.pred {
        if matches!(col.const_lead, LeadValue::Empty) && col.const_trail.is_empty() {
            match col.stream {
                ColumnStream::DecimalSmall { u16_offset } => {
                    for t in 0..ones {
                        if cmp.test(u64::from(scratch.u16_arena[u16_offset + t])) {
                            hits.push(t);
                            if hits.len() >= rem {
                                break;
                            }
                        }
                    }
                    return Ok((Some(col), hits));
                }
                ColumnStream::DecimalValues { u64_offset } => {
                    for t in 0..ones {
                        if cmp.test(scratch.u64_arena[u64_offset + t]) {
                            hits.push(t);
                            if hits.len() >= rem {
                                break;
                            }
                        }
                    }
                    return Ok((Some(col), hits));
                }
                ColumnStream::DecimalAffineSmall { current, step }
                | ColumnStream::DecimalAffine { current, step } => {
                    for t in 0..ones {
                        if cmp.test(current.wrapping_add(step.wrapping_mul(t as u64))) {
                            hits.push(t);
                            if hits.len() >= rem {
                                break;
                            }
                        }
                    }
                    return Ok((Some(col), hits));
                }
                ColumnStream::DecimalPacked { min, mut reader } => {
                    for t in 0..ones {
                        if cmp.test(min.wrapping_add(reader.next_u64())) {
                            hits.push(t);
                            if hits.len() >= rem {
                                break;
                            }
                        }
                    }
                    return Ok((Some(col), hits));
                }
                _ => {}
            }
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
        if crate::ops::eval_token_pred(pred.col, &val_buf[..written], pred.pred) {
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

/// Search a `GRAMMAR_PACKET` block for `pred`, materializing non-target columns ONLY if at least
/// one template row matches. Returns `Ok(None)` to request per-line fallback.
pub(crate) fn search_grammar_block_col_pred(
    payload: &[u8],
    block_index: usize,
    global_row_base: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    pred: &crate::ops::ColPred<'_>,
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
) -> Result<Option<usize>> {
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(None);
    };
    if out.len() >= limit || hdr.row_count == 0 {
        return Ok(Some(hdr.row_count));
    }
    if carry.is_some() && !hdr.first_is_raw {
        return Ok(None);
    }

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.raw_arena.reserve(hdr.raw_len.saturating_mul(2) + 16384);

        let raw_slice_offset = {
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            read_grammar_raw_rows(&mut hdr.cursor, hdr.raw_count, raw_arena, slice_arena)?
        };

        if hdr.template_count > 1 {
            let mut multi_cols: Vec<Vec<&[u8]>> = Vec::with_capacity(hdr.template_count);
            for t in 0..hdr.template_count {
                let slots = hdr.slot_counts[t];
                let mut tc = Vec::with_capacity(slots);
                for _ in 0..slots {
                    let len = hdr.cursor.uleb()?;
                    tc.push(hdr.cursor.take(len)?);
                }
                multi_cols.push(tc);
            }
            if !hdr.cursor.done() {
                return Err(invalid("column packet trailing bytes"));
            }
            for t in 0..hdr.template_count {
                let cnt = hdr.tpl_row_counts[t];
                if cnt == 0 {
                    continue;
                }
                match resolve_template_slot(hdr.segs(t), &multi_cols[t], pred.col)? {
                    SlotLookup::Fallback => return Ok(None),
                    SlotLookup::Absent => {}
                    SlotLookup::Slot(s) => {
                        let (_, h) =
                            eval_col_pred_on_slot(multi_cols[t][s], cnt, 1, pred, &mut scratch)?;
                        if !h.is_empty() {
                            return Ok(None);
                        }
                    }
                }
            }
            // All templates proved 0 hits! Handle only carry + raw rows without column unpack.
            return finish_raw_only_search(
                &hdr,
                raw_slice_offset,
                block_index,
                global_row_base,
                carry,
                pred,
                limit,
                out,
                &scratch,
            );
        }

        let slots = hdr.slot_counts[0];
        let ones = hdr.tpl_row_counts[0];
        let mut col_slices = [&[][..]; MAX_SLOTS];
        for col_slot in col_slices.iter_mut().take(slots) {
            let len = hdr.cursor.uleb()?;
            *col_slot = hdr.cursor.take(len)?;
        }
        if !hdr.cursor.done() {
            return Err(invalid("column packet trailing bytes"));
        }

        let t0_segs = hdr.segs(0);
        let lookup = if ones > 0 {
            resolve_template_slot(t0_segs, &col_slices[..slots], pred.col)?
        } else {
            SlotLookup::Absent
        };
        if lookup == SlotLookup::Fallback {
            return Ok(None);
        }

        let had_incoming_carry = carry.is_some();
        let base_row = global_row_base - usize::from(had_incoming_carry);
        let (raw_start, raw_end, next_carry, early_ret) = advance_search_carry(
            &hdr,
            raw_slice_offset,
            block_index,
            base_row,
            carry,
            pred,
            limit,
            out,
            &scratch,
        );
        if let Some(ret) = early_ret {
            return Ok(Some(ret));
        }
        let new_rows = hdr.row_count - usize::from(had_incoming_carry);
        if out.len() >= limit {
            *carry = next_carry;
            return Ok(Some(new_rows));
        }

        let rem_limit = limit - out.len();
        let (target_col_opt, hits) = match lookup {
            SlotLookup::Slot(s) => {
                let (c_opt, h) =
                    eval_col_pred_on_slot(col_slices[s], ones, rem_limit, pred, &mut scratch)?;
                (c_opt.map(|c| (s, c)), h)
            }
            SlotLookup::Absent => (None, Vec::new()),
            SlotLookup::Fallback => unreachable!(),
        };

        let mut local_matches = Vec::<(usize, Vec<u8>)>::new();
        for r_idx in raw_start..raw_end {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            if crate::ops::line_matches_col_pred(slice, pred) {
                let row_idx = nth_selector_bit(hdr.selector_slice, hdr.row_count, false, r_idx);
                local_matches.push((row_idx, slice.to_vec()));
                if local_matches.len() >= rem_limit && hits.is_empty() {
                    break;
                }
            }
        }

        if !hits.is_empty() {
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            for s in 0..slots {
                cols[s] = match target_col_opt {
                    Some((ts, tc)) if ts == s => tc,
                    _ => PreparedColumn::parse_and_validate(col_slices[s], ones, &mut scratch)?,
                };
            }
            let unbound = UnboundTemplate::build(t0_segs, &cols[..slots], &mut scratch);
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut tpl = unbound.bind(&mut scratch);
            let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
            let mut cur_t = 0usize;
            for &t_idx in &hits {
                unsafe {
                    tpl.skip_rows(t_idx - cur_t);
                    let sp = row_buf.as_mut_ptr();
                    let ep = tpl.emit_row(sp);
                    let written = ep.offset_from(sp) as usize;
                    cur_t = t_idx + 1;
                    let row_idx = if hdr.raw_count == 0 {
                        t_idx
                    } else {
                        nth_selector_bit(hdr.selector_slice, hdr.row_count, true, t_idx)
                    };
                    local_matches.push((row_idx, row_buf[..written].to_vec()));
                }
            }
        }

        if hdr.raw_count > 0 && local_matches.len() > 1 {
            local_matches.sort_unstable_by_key(|&(r, _)| r);
        }
        for (row_idx, bytes) in local_matches.into_iter().take(rem_limit) {
            out.push(crate::query::RowMatch {
                block_index,
                row_index: row_idx,
                global_row_index: base_row + row_idx,
                bytes,
            });
        }
        *carry = next_carry;
        Ok(Some(new_rows))
    })
}
