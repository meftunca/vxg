#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotLookup {
    Slot(usize),
    Absent,
    Fallback,
}

#[inline]
fn json_single_line_template_slot(segs: &[&[u8]], target_key: &[u8]) -> Option<Option<usize>> {
    json_single_line_template_slot_cached(segs, target_key)
}

fn json_single_line_template_slot_uncached(
    segs: &[&[u8]],
    target_key: &[u8],
) -> Option<Option<usize>> {
    let slots = segs.len().saturating_sub(1);
    if slots == 0 {
        return None;
    }
    let mut path = [0u8; 512];
    let mut path_len = 0usize;
    let mut depth_lens = [0usize; 16];
    let mut depth = 0usize;
    let mut top_exact_slot = None;
    let mut nested_slot = None;
    let mut has_prefix_slot = false;
    for s in 0..slots {
        let seg = segs[s];
        if seg.contains(&b'\n') {
            return None;
        }
        let mut p = 0usize;
        skip_space(seg, &mut p);
        if s == 0 {
            if seg.get(p) != Some(&b'{') {
                return None;
            }
            p += 1;
            depth = 1;
            depth_lens[0] = 0;
        } else {
            while seg.get(p) == Some(&b'}') {
                if depth <= 1 {
                    return None;
                }
                depth -= 1;
                path_len = depth_lens[depth - 1];
                p += 1;
                skip_space(seg, &mut p);
            }
            if seg.get(p) != Some(&b',') {
                return None;
            }
            p += 1;
        }
        loop {
            skip_space(seg, &mut p);
            let (k_start, k_end) = (p, scan_string(seg, p)?);
            p = k_end;
            let k_dec = crate::ops::unescape_json_str(&seg[k_start..k_end])?;
            let kb = k_dec.as_ref();
            skip_space(seg, &mut p);
            if seg.get(p) != Some(&b':') {
                return None;
            }
            p += 1;
            skip_space(seg, &mut p);
            if seg.get(p) == Some(&b'{') {
                if (depth == 1 && kb == target_key)
                    || (path_len > 0
                        && path_len + kb.len() == target_key.len()
                        && target_key.starts_with(&path[..path_len])
                        && &target_key[path_len..] == kb)
                {
                    return None;
                }
                if depth >= 16 || path_len + kb.len() + 1 > path.len() {
                    return None;
                }
                path[path_len..path_len + kb.len()].copy_from_slice(kb);
                path[path_len + kb.len()] = b'.';
                path_len += kb.len() + 1;
                depth_lens[depth] = path_len;
                depth += 1;
                p += 1;
                continue;
            }
            if p != seg.len() {
                return None;
            }
            if depth == 1 && kb == target_key {
                if top_exact_slot.is_none() {
                    top_exact_slot = Some(s);
                }
            } else if path_len > 0
                && path_len + kb.len() == target_key.len()
                && target_key.starts_with(&path[..path_len])
                && &target_key[path_len..] == kb
            {
                if nested_slot.is_none() {
                    nested_slot = Some(s);
                }
            } else if path_len + kb.len() < target_key.len()
                && target_key.starts_with(&path[..path_len])
                && target_key[path_len..].starts_with(kb)
                && target_key[path_len + kb.len()] == b'.'
            {
                has_prefix_slot = true;
            }
            break;
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
    for _ in 0..depth {
        skip_space(last_body, &mut lp);
        if last_body.get(lp) != Some(&b'}') {
            return None;
        }
        lp += 1;
    }
    skip_space(last_body, &mut lp);
    if last_body.get(lp) == Some(&b',') {
        lp += 1;
        skip_space(last_body, &mut lp);
    }
    if lp != last_body.len() {
        return None;
    }
    if let Some(s) = top_exact_slot {
        Some(Some(s))
    } else if has_prefix_slot {
        None
    } else {
        Some(nested_slot)
    }
}

#[inline]
fn resolve_template_slot(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    col: crate::ops::ColTarget<'_>,
) -> Result<SlotLookup> {
    resolve_template_slot_with_delim(segs, col_slices, col, None)
}

fn resolve_template_slot_with_delim(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    col: crate::ops::ColTarget<'_>,
    known_delim: Option<Option<u8>>,
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
            if !key.is_empty() && !key.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20) {
                for part in key.split('.') {
                    if !part.is_empty()
                        && !can_template_match_fast(segs, col_slices, part.as_bytes())?
                    {
                        return Ok(SlotLookup::Absent);
                    }
                }
            }
            Ok(SlotLookup::Fallback)
        }
        crate::ops::ColTarget::Delimited { delim, index } => {
            let tpl_delim = known_delim.unwrap_or_else(|| delimited_template_delim(segs));
            if tpl_delim == Some(delim) {
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
        MODEL_TIMESTAMP => {
            let unquoted = (c.u16()? & 0x8000) != 0;
            add_bytes_to_mask(&mut m, if unquoted { b"0123456789-:TZ" } else { b"\"0123456789-:TZ" });
        }
        MODEL_PATTERN => pattern_col_alphabet(&mut c, &mut m)?,
        MODEL_SCALED_DECIMAL => scaled_dec_col_alphabet(&mut c, &mut m)?,
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
    let mut hc = Cursor::new(col_bytes);
    if let Ok(tag) = hc.byte() {
        match tag {
            MODEL_RAW => {
                while !hc.done() {
                    let len = hc.uleb()?;
                    if crate::ops::eval_token_pred(pred.col, hc.take(len)?, pred.pred) {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
            MODEL_DICTIONARY => {
                let entries_len = hc.uleb()?;
                for _ in 0..entries_len {
                    let len = hc.uleb()?;
                    if crate::ops::eval_token_pred(pred.col, hc.take(len)?, pred.pred) {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
            MODEL_FRONT_DICT => {
                let entries_len = hc.uleb()?;
                let dict_mode = hc.byte()? & 0x03;
                if let Some(found) = front_dict_visit_entries(&mut hc, entries_len, dict_mode, |e| {
                    crate::ops::eval_token_pred(pred.col, e, pred.pred)
                })? {
                    return Ok(found);
                }
            }
            _ => {}
        }
    }
    match pred.pred {
        crate::ops::ValuePred::Num(cmp) => {
            if col_bytes.first() == Some(&MODEL_SCALED_DECIMAL)
                && col_bytes.get(1).copied().unwrap_or(0) > 0
            {
                return Ok(false);
            }
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
        crate::ops::ValuePred::Dec(dcmp) => {
            if dec_col_zone_truth(col_bytes, ones, dcmp) == Some(TemplateTruth::AlwaysFalse) {
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
            let is_bool_cmp = matches!(
                sc,
                crate::ops::StrCmp::Eq(b"true" | b"false")
                    | crate::ops::StrCmp::Ne(b"true" | b"false")
            );
            if matches!(pred.col, crate::ops::ColTarget::JsonKey(_))
                && !is_bool_cmp
                && !mask_has_byte(&m, b'"')
            {
                return Ok(false);
            }
            let req = sc.required_slice();
            if !req.is_empty() && !mask_has_byte(&m, b'\\') {
                for &b in req {
                    if !mask_has_byte(&m, b) {
                        return Ok(false);
                    }
                }
                if !col_can_contain_isolated(col_bytes, ones, req)? {
                    return Ok(false);
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
            let mut valid_inline = [false; 256];
            let mut valid_ov = if entries > 256 {
                vec![false; entries]
            } else {
                Vec::new()
            };
            let valid: &mut [bool] = if entries <= 256 {
                &mut valid_inline[..entries]
            } else {
                &mut valid_ov[..entries]
            };
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

    let u64_dom = match pred.pred {
        crate::ops::ValuePred::Num(c) => Some(crate::ops::dec_num::U64DomainCmp::Cmp(c)),
        crate::ops::ValuePred::Dec(dc) => Some(dc.to_u64_domain()),
        crate::ops::ValuePred::Str(_) => None,
    };
    if let Some(dom) = u64_dom {
        if matches!(col.const_lead, LeadValue::Empty)
            && col.const_trail.is_empty()
            && matches!(
                col.stream,
                ColumnStream::DecimalSmall { .. }
                    | ColumnStream::DecimalValues { .. }
                    | ColumnStream::DecimalAffineSmall { .. }
                    | ColumnStream::DecimalAffine { .. }
                    | ColumnStream::DecimalPacked { .. }
            )
        {
            let cmp = match dom {
                crate::ops::dec_num::U64DomainCmp::AlwaysFalse => return Ok((Some(col), hits)),
                crate::ops::dec_num::U64DomainCmp::AlwaysTrue => {
                    hits.extend(0..ones.min(rem));
                    return Ok((Some(col), hits));
                }
                crate::ops::dec_num::U64DomainCmp::Cmp(c) => c,
            };
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
