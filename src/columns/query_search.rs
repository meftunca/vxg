/// Search a `GRAMMAR_PACKET` payload in the compressed domain for rows containing `needle`.
/// Appends up to `limit` total matches into `out` and returns `Ok(Some(new_logical_rows))`.
pub(crate) fn search_grammar_block(
    payload: &[u8],
    block_index: usize,
    global_row_base: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    needle: &[u8],
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
    num: Option<&crate::query::NumFilter>,
) -> Result<Option<usize>> {
    if payload.first() != Some(&GRAMMAR_PACKET) {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&payload[1..]);
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE {
        return Err(invalid("column raw length"));
    }
    let template_count = cursor.uleb()?;
    if template_count > MAX_TEMPLATES {
        return Err(invalid("column template count"));
    }
    let dictionary_len = cursor.uleb()?;
    if dictionary_len > template_count.saturating_mul(MAX_SLOTS + 1) {
        return Err(invalid("column segment dictionary count"));
    }
    let mut seg_dict_inline = [&[][..]; 128];
    let mut seg_dict_overflow = if dictionary_len > 128 {
        vec![&[][..]; dictionary_len]
    } else {
        Vec::new()
    };
    let seg_dict: &mut [&[u8]] = if dictionary_len <= 128 {
        &mut seg_dict_inline[..dictionary_len]
    } else {
        &mut seg_dict_overflow[..dictionary_len]
    };
    for i in 0..dictionary_len {
        let len = cursor.uleb()?;
        let seg = cursor.take(len)?;
        if i > 0 && seg_dict[i - 1] >= seg {
            return Err(invalid("column segment dictionary ordering"));
        }
        seg_dict[i] = seg;
    }
    let mut slot_counts_inline = [0usize; 16];
    let mut slot_counts_overflow = if template_count > 16 {
        vec![0usize; template_count]
    } else {
        Vec::new()
    };
    let slot_counts: &mut [usize] = if template_count <= 16 {
        &mut slot_counts_inline[..template_count]
    } else {
        &mut slot_counts_overflow[..template_count]
    };
    for s in slot_counts.iter_mut() {
        let slots = cursor.uleb()?;
        if slots > MAX_SLOTS {
            return Err(invalid("column slot count"));
        }
        *s = slots;
    }
    let segment_width = cursor.byte()?;
    if segment_width != bit_width(dictionary_len.saturating_sub(1) as u64) {
        return Err(invalid("column segment selector width"));
    }
    let mut tpl0_segs_inline = [&[][..]; MAX_SLOTS + 1];
    let mut tpl_segs = if template_count > 1 {
        Vec::with_capacity(template_count)
    } else {
        Vec::new()
    };
    for (t_idx, &slots) in slot_counts.iter().enumerate() {
        if dictionary_len == 0 {
            return Err(invalid("empty column segment dictionary"));
        }
        let packed_len = bit_bytes(slots + 1, segment_width)?;
        let packed_slice = cursor.take(packed_len)?;
        let mut reader = FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
        if template_count == 1 && t_idx == 0 {
            for s in 0..=slots {
                let idx = reader.next_u64() as usize;
                if idx >= dictionary_len {
                    return Err(invalid("column segment dictionary index"));
                }
                tpl0_segs_inline[s] = seg_dict[idx];
            }
        } else {
            let mut segs = Vec::with_capacity(slots + 1);
            for _ in 0..=slots {
                let idx = reader.next_u64() as usize;
                if idx >= dictionary_len {
                    return Err(invalid("column segment dictionary index"));
                }
                segs.push(seg_dict[idx]);
            }
            tpl_segs.push(segs);
        }
    }
    let row_count = cursor.uleb()?;
    if row_count > raw_len.saturating_add(1) {
        return Err(invalid("column row count"));
    }
    if out.len() >= limit || row_count == 0 {
        return Ok(Some(row_count));
    }
    let selector_width = cursor.byte()?;
    if selector_width != bit_width(template_count as u64) {
        return Err(invalid("column row selector width"));
    }
    let selector_len = bit_bytes(row_count, selector_width)?;
    let selector_slice = cursor.take(selector_len)?;

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.raw_arena.reserve(raw_len.saturating_mul(2) + 16384);

        if num.is_some() && !(template_count == 1 && selector_width == 1) {
            // Numeric predicates have a column fast path only for single-template blocks.
            return Ok(None);
        }
        if template_count == 1 && selector_width == 1 {
            let _ = FastBitReader::new_validated(selector_slice, row_count, 1)?;
            let mut ones = 0usize;
            for &b in selector_slice {
                ones += b.count_ones() as usize;
            }
            let raw_count = row_count - ones;
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            let raw_slice_offset =
                read_grammar_raw_rows(&mut cursor, raw_count, raw_arena, slice_arena)?;

            let slots = slot_counts[0];
            let mut col_slices = [&[][..]; MAX_SLOTS];
            for s in 0..slots {
                let len = cursor.uleb()?;
                col_slices[s] = cursor.take(len)?;
            }
            if !cursor.done() {
                return Err(invalid("column packet trailing bytes"));
            }

            let had_incoming_carry = carry.is_some();
            let base_row = if had_incoming_carry {
                global_row_base - 1
            } else {
                global_row_base
            };
            let first_is_raw = (selector_slice[0] & 1) == 0;
            let last_is_raw =
                ((selector_slice[(row_count - 1) >> 3] >> ((row_count - 1) & 7)) & 1) == 0;

            let row_hit = |b: &[u8]| match num {
                Some(nf) => crate::query::line_numeric_match(b, nf),
                None => contains_subslice(b, needle),
            };
            let mut raw_start = 0usize;
            if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                if first_is_raw && raw_count > 0 {
                    let (ptr, len) = scratch.slice_arena[raw_slice_offset];
                    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                    c_bytes.extend_from_slice(slice);
                    raw_start = 1;
                    if !c_bytes.ends_with(b"\n") && row_count == 1 {
                        *carry = Some((c_blk, c_row, c_glob, c_bytes));
                        return Ok(Some(0));
                    }
                }
                if row_hit(&c_bytes) && out.len() < limit {
                    out.push(crate::query::RowMatch {
                        block_index: c_blk,
                        row_index: c_row,
                        global_row_index: c_glob,
                        bytes: c_bytes,
                    });
                }
            }

            let mut raw_end = raw_count;
            let mut next_carry: Option<(usize, usize, usize, Vec<u8>)> = None;
            if last_is_raw && raw_end > raw_start {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + raw_count - 1];
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                if !slice.ends_with(b"\n") {
                    raw_end = raw_count - 1;
                    next_carry = Some((
                        block_index,
                        row_count - 1,
                        base_row + row_count - 1,
                        slice.to_vec(),
                    ));
                }
            }
            let new_rows = row_count - usize::from(had_incoming_carry);
            if out.len() >= limit {
                *carry = next_carry;
                return Ok(Some(new_rows));
            }

            let t0_segs = &tpl0_segs_inline[..=slots];
            let mut num_plan = None;
            let can_tpl_match = if let Some(nf) = num {
                if ones > 0 {
                    num_plan = numeric_hits(
                        nf,
                        t0_segs,
                        &col_slices[..slots],
                        ones,
                        limit - out.len(),
                        &mut scratch,
                    )?;
                }
                match &num_plan {
                    Some((_, _, hits)) => !hits.is_empty(),
                    None => ones > 0,
                }
            } else {
                ones > 0 && can_template_match_fast(t0_segs, &col_slices[..slots], needle)?
            };
            let rem_limit = limit - out.len();
            let mut local_matches = Vec::<(usize, Vec<u8>)>::new();
            for r_idx in raw_start..raw_end {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                if row_hit(slice) {
                    let row_idx = nth_selector_bit(selector_slice, row_count, false, r_idx);
                    local_matches.push((row_idx, slice.to_vec()));
                    if local_matches.len() >= rem_limit && !can_tpl_match {
                        break;
                    }
                }
            }

            if can_tpl_match {
                let mut cols_uninit =
                    core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
                let cols: &mut [PreparedColumn; MAX_SLOTS] =
                    unsafe { cols_uninit.assume_init_mut() };
                let mut active_dict_lens = [0usize; MAX_SLOTS];
                let mut act_idx = 0usize;
                for s in 0..slots {
                    cols[s] = match &num_plan {
                        Some((t, col, _)) if *t == s => *col,
                        _ => PreparedColumn::parse_and_validate(col_slices[s], ones, &mut scratch)?,
                    };
                    if matches!(cols[s].const_full, ConstValue::None) {
                        active_dict_lens[act_idx] = cols[s].dict_slices.map_or(0, |(_, dl)| dl);
                        act_idx += 1;
                    }
                }
                let unbound = UnboundTemplate::build(t0_segs, &cols[..slots], &mut scratch);
                let filter = if let Some((_, _, hits)) = num_plan.take() {
                    TplFilter::Hits(hits)
                } else if num.is_some() {
                    TplFilter::CheckAll
                } else {
                    classify_template_filter(
                        &unbound,
                        &active_dict_lens[..act_idx],
                        &scratch,
                        needle,
                    )
                };
                if !matches!(filter, TplFilter::Never) {
                    let unbound_len = scratch.unbound_col_arena.len();
                    scratch.exec_col_arena.reserve(unbound_len);
                    let mut tpl = unbound.bind(&mut scratch);
                    let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
                    let mut cur_t = 0usize;

                    let mut emit_and_check = |t_idx: usize, verify: bool| -> bool {
                        unsafe {
                            tpl.skip_rows(t_idx - cur_t);
                            let start_p = row_buf.as_mut_ptr();
                            let end_p = tpl.emit_row(start_p);
                            let written = end_p.offset_from(start_p) as usize;
                            cur_t = t_idx + 1;
                            let row_slice = &row_buf[..written];
                            if !verify || row_hit(row_slice) {
                                let row_idx = if raw_count == 0 {
                                    t_idx
                                } else {
                                    nth_selector_bit(selector_slice, row_count, true, t_idx)
                                };
                                local_matches.push((row_idx, row_slice.to_vec()));
                                return true;
                            }
                            false
                        }
                    };

                    match filter {
                        TplFilter::Never => unreachable!(),
                        TplFilter::Hits(hits) => {
                            for &t in &hits {
                                emit_and_check(t, false);
                            }
                        }
                        TplFilter::Always => {
                            for t_idx in 0..ones.min(rem_limit) {
                                emit_and_check(t_idx, false);
                            }
                        }
                        TplFilter::DictSingle {
                            u8_offset,
                            valid_ids,
                            single_id: Some(target_id),
                        } => {
                            let _ = valid_ids;
                            let idx_slice = &scratch.u8_arena[u8_offset..u8_offset + ones];
                            let mut pos = 0usize;
                            let mut tpl_found = 0usize;
                            while pos < ones {
                                let Some(rel) = find_byte(target_id, &idx_slice[pos..]) else {
                                    break;
                                };
                                let t_idx = pos + rel;
                                if emit_and_check(t_idx, true) {
                                    tpl_found += 1;
                                    if tpl_found >= rem_limit {
                                        break;
                                    }
                                }
                                pos = t_idx + 1;
                            }
                        }
                        TplFilter::DictSingle {
                            u8_offset,
                            valid_ids,
                            single_id: None,
                        } => {
                            let mut tpl_found = 0usize;
                            for t_idx in 0..ones {
                                let id = scratch.u8_arena[u8_offset + t_idx] as usize;
                                if valid_ids[id] && emit_and_check(t_idx, true) {
                                    tpl_found += 1;
                                    if tpl_found >= rem_limit {
                                        break;
                                    }
                                }
                            }
                        }
                        TplFilter::CheckAll => {
                            let mut tpl_found = 0usize;
                            for t_idx in 0..ones {
                                if emit_and_check(t_idx, true) {
                                    tpl_found += 1;
                                    if tpl_found >= rem_limit {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if raw_count > 0 && local_matches.len() > 1 {
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
            return Ok(Some(new_rows));
        }

        // Multi-template or 0-template blocks: iterate row-by-row and stop at `limit`.
        let sel_u8_offset = unpack_and_validate_u8(
            selector_slice,
            row_count,
            selector_width,
            0,
            template_count + 1,
            &mut scratch.u8_arena,
        )?;
        let mut sel_counts = [0usize; 129];
        for &sel in &scratch.u8_arena[sel_u8_offset..sel_u8_offset + row_count] {
            sel_counts[sel as usize] += 1;
        }
        let raw_total = sel_counts[0];
        let DecodeScratch {
            ref mut raw_arena,
            ref mut slice_arena,
            ..
        } = *scratch;
        let raw_slice_offset =
            read_grammar_raw_rows(&mut cursor, raw_total, raw_arena, slice_arena)?;
        let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
        let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
        let mut unbound_templates = vec![UnboundTemplate::default(); template_count];
        for tpl_idx in 0..template_count {
            let slots = slot_counts[tpl_idx];
            let count = sel_counts[tpl_idx + 1];
            for s in 0..slots {
                let len = cursor.uleb()?;
                let bytes = cursor.take(len)?;
                cols[s] = PreparedColumn::parse_and_validate(bytes, count, &mut scratch)?;
            }
            unbound_templates[tpl_idx] =
                UnboundTemplate::build(&tpl_segs[tpl_idx], &cols[..slots], &mut scratch);
        }
        if !cursor.done() {
            return Err(invalid("column packet trailing bytes"));
        }
        let total_unbound_cols = scratch.unbound_col_arena.len();
        scratch.exec_col_arena.reserve(total_unbound_cols);
        let mut prepared_templates = Vec::with_capacity(template_count);
        for unbound in unbound_templates {
            prepared_templates.push(unbound.bind(&mut scratch));
        }
        let had_incoming_carry = carry.is_some();
        let base_row = if had_incoming_carry {
            global_row_base - 1
        } else {
            global_row_base
        };
        let new_rows = row_count - usize::from(had_incoming_carry);
        let mut raw_idx = 0usize;
        let mut row_buf = vec![0u8; 4096];
        for r in 0..row_count {
            let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
            let row_slice: &[u8] = if sel != 0 {
                let tpl = &mut prepared_templates[sel - 1];
                if tpl.row_budget > row_buf.len() {
                    row_buf.resize(tpl.row_budget + 64, 0);
                }
                let written = unsafe {
                    let start_p = row_buf.as_mut_ptr();
                    let end_p = tpl.emit_row(start_p);
                    end_p.offset_from(start_p) as usize
                };
                &row_buf[..written]
            } else {
                let (raw_ptr, len) = scratch.slice_arena[raw_slice_offset + raw_idx];
                raw_idx += 1;
                unsafe { std::slice::from_raw_parts(raw_ptr, len) }
            };
            if r == 0 {
                if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                    c_bytes.extend_from_slice(row_slice);
                    if !c_bytes.ends_with(b"\n") && row_count == 1 {
                        *carry = Some((c_blk, c_row, c_glob, c_bytes));
                        return Ok(Some(new_rows));
                    }
                    if contains_subslice(&c_bytes, needle) && out.len() < limit {
                        out.push(crate::query::RowMatch {
                            block_index: c_blk,
                            row_index: c_row,
                            global_row_index: c_glob,
                            bytes: c_bytes,
                        });
                    }
                    continue;
                }
            }
            if r == row_count - 1 && !row_slice.ends_with(b"\n") {
                *carry = Some((block_index, r, base_row + r, row_slice.to_vec()));
                break;
            }
            if contains_subslice(row_slice, needle) && out.len() < limit {
                out.push(crate::query::RowMatch {
                    block_index,
                    row_index: r,
                    global_row_index: base_row + r,
                    bytes: row_slice.to_vec(),
                });
            }
        }
        Ok(Some(new_rows))
    })
}
