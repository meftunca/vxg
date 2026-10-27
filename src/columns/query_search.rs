enum TplFilter {
    Never,
    Always,
    DictSingle {
        u8_offset: usize,
        valid_ids: Box<[bool; 256]>,
        single_id: Option<u8>,
    },
    CheckAll,
}

fn classify_template_filter(
    unbound: &UnboundTemplate,
    active_dict_lens: &[usize],
    scratch: &DecodeScratch,
    needle: &[u8],
) -> TplFilter {
    if needle.is_empty() {
        return TplFilter::Always;
    }
    let head_slice = &scratch.str_arena[unbound.head_range.0..unbound.head_range.1];
    if contains_subslice(head_slice, needle) {
        return TplFilter::Always;
    }
    let active_cols = &scratch.unbound_col_arena[unbound.col_start..unbound.col_end];
    let mut non_dict_col_mask = [0u64; 4];
    let mut static_mask = [0u64; 4];
    add_bytes_to_mask(&mut static_mask, head_slice);

    for ucol in active_cols {
        let after_slice = &scratch.str_arena[ucol.after_range.0..ucol.after_range.1];
        if contains_subslice(after_slice, needle) {
            return TplFilter::Always;
        }
        add_bytes_to_mask(&mut static_mask, after_slice);
        match ucol.stream {
            ColumnStream::Dictionary { .. } => {}
            ColumnStream::DecimalAffineSmall { .. }
            | ColumnStream::DecimalAffine { .. }
            | ColumnStream::DecimalSmall { .. }
            | ColumnStream::DecimalPacked { .. }
            | ColumnStream::DecimalValues { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"-0123456789\"");
            }
            ColumnStream::TimestampMinSec { .. } | ColumnStream::TimestampFull { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"0123456789-:T\"");
            }
            ColumnStream::HexSuffix4 { .. }
            | ColumnStream::HexSuffix8 { .. }
            | ColumnStream::HexSuffix { .. }
            | ColumnStream::Copy16 { .. }
            | ColumnStream::PeriodicHex { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"0123456789abcdef\"");
            }
            ColumnStream::SegmentDict { .. }
            | ColumnStream::Raw { .. }
            | ColumnStream::PeriodicString { .. } => {
                non_dict_col_mask = [u64::MAX; 4];
            }
        }
    }

    let mut total_mask = [
        static_mask[0] | non_dict_col_mask[0],
        static_mask[1] | non_dict_col_mask[1],
        static_mask[2] | non_dict_col_mask[2],
        static_mask[3] | non_dict_col_mask[3],
    ];
    for (idx, ucol) in active_cols.iter().enumerate() {
        if let ColumnStream::Dictionary { piece_offset, .. } = ucol.stream {
            let dlen = active_dict_lens[idx];
            for e in 0..dlen {
                let piece = unsafe { scratch.piece_arena[piece_offset + e].as_slice() };
                add_bytes_to_mask(&mut total_mask, piece);
            }
        }
    }

    for &b in needle {
        if !mask_has_byte(&total_mask, b) {
            return TplFilter::Never;
        }
    }

    let mut best_filter: Option<(usize, TplFilter)> = None;
    for k in 0..needle.len() {
        let b = needle[k];
        if mask_has_byte(&non_dict_col_mask, b) {
            continue;
        }
        let mut aligns_static = piece_can_align_at(head_slice, needle, k);
        if !aligns_static {
            for ucol in active_cols {
                let after_slice = &scratch.str_arena[ucol.after_range.0..ucol.after_range.1];
                if piece_can_align_at(after_slice, needle, k) {
                    aligns_static = true;
                    break;
                }
            }
        }
        if aligns_static {
            continue;
        }

        // `needle[k]` cannot come from any static segment or non-dictionary column.
        // It MUST come from a `Dictionary` column whose entry aligns with `needle` at `k`.
        let mut matching_cols = 0usize;
        let mut candidate_u8_off = 0usize;
        let mut candidate_dlen = 0usize;
        let mut candidate_ids = [false; 256];
        let mut candidate_count = 0usize;
        let mut last_id = 0u8;

        for (idx, ucol) in active_cols.iter().enumerate() {
            if let ColumnStream::Dictionary {
                piece_offset,
                u8_offset,
                ..
            } = ucol.stream
            {
                let dlen = active_dict_lens[idx].min(256);
                let mut col_matches = 0usize;
                let mut local_ids = [false; 256];
                let mut local_last = 0u8;
                for e in 0..dlen {
                    let piece = unsafe { scratch.piece_arena[piece_offset + e].as_slice() };
                    if piece_can_align_at(piece, needle, k) {
                        local_ids[e] = true;
                        local_last = e as u8;
                        col_matches += 1;
                    }
                }
                if col_matches > 0 {
                    matching_cols += 1;
                    if matching_cols == 1 {
                        candidate_u8_off = u8_offset;
                        candidate_dlen = dlen;
                        candidate_ids = local_ids;
                        candidate_count = col_matches;
                        last_id = local_last;
                    }
                }
            }
        }

        if matching_cols == 0 {
            return TplFilter::Never;
        }
        if matching_cols == 1 && candidate_count < candidate_dlen {
            let is_better = best_filter
                .as_ref()
                .is_none_or(|(prev_cnt, _)| candidate_count < *prev_cnt);
            if is_better {
                let single_id = (candidate_count == 1).then_some(last_id);
                best_filter = Some((
                    candidate_count,
                    TplFilter::DictSingle {
                        u8_offset: candidate_u8_off,
                        valid_ids: Box::new(candidate_ids),
                        single_id,
                    },
                ));
                if candidate_count == 1 {
                    break;
                }
            }
        }
    }

    best_filter.map_or(TplFilter::CheckAll, |(_, f)| f)
}

/// Find the row index (`0..row_count`) of the `nth` occurrence of `target_bit` (`true` = 1, `false` = 0)
/// inside a 1-bit packed `selector_slice`.
fn nth_selector_bit(selector_slice: &[u8], row_count: usize, target_one: bool, mut nth: usize) -> usize {
    for (byte_idx, &b) in selector_slice.iter().enumerate() {
        let base_row = byte_idx * 8;
        let valid_bits = row_count.saturating_sub(base_row).min(8);
        let mask = if valid_bits == 8 { 0xffu8 } else { (1u8 << valid_bits) - 1 };
        let word = (if target_one { b } else { !b }) & mask;
        let ones = word.count_ones() as usize;
        if nth < ones {
            for bit in 0..valid_bits {
                if ((word >> bit) & 1) != 0 {
                    if nth == 0 {
                        return base_row + bit;
                    }
                    nth -= 1;
                }
            }
        }
        nth -= ones;
    }
    row_count
}

/// Search a `GRAMMAR_PACKET` payload in the compressed domain for rows containing `needle`.
/// Appends up to `limit` total matches into `out` and returns `Ok(Some(row_count))`.
pub(crate) fn search_grammar_block(
    payload: &[u8],
    block_index: usize,
    global_row_base: usize,
    needle: &[u8],
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
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
    let mut seg_dict = vec![&[][..]; dictionary_len];
    for i in 0..dictionary_len {
        let len = cursor.uleb()?;
        let seg = cursor.take(len)?;
        if i > 0 && seg_dict[i - 1] >= seg {
            return Err(invalid("column segment dictionary ordering"));
        }
        seg_dict[i] = seg;
    }
    let mut slot_counts = vec![0usize; template_count];
    for s in &mut slot_counts {
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
    let mut tpl_segs = Vec::with_capacity(template_count);
    for &slots in &slot_counts {
        if dictionary_len == 0 {
            return Err(invalid("empty column segment dictionary"));
        }
        let packed_len = bit_bytes(slots + 1, segment_width)?;
        let packed_slice = cursor.take(packed_len)?;
        let mut reader = FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
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
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            let mut active_dict_lens = [0usize; MAX_SLOTS];
            let mut act_idx = 0usize;
            for s in 0..slots {
                let len = cursor.uleb()?;
                let bytes = cursor.take(len)?;
                cols[s] = PreparedColumn::parse_and_validate(bytes, ones, &mut scratch)?;
                if matches!(cols[s].const_full, ConstValue::None) {
                    active_dict_lens[act_idx] = cols[s].dict_slices.map_or(0, |(_, dl)| dl);
                    act_idx += 1;
                }
            }
            if !cursor.done() {
                return Err(invalid("column packet trailing bytes"));
            }
            let unbound = UnboundTemplate::build(&tpl_segs[0], &cols[..slots], &mut scratch);
            let filter =
                classify_template_filter(&unbound, &active_dict_lens[..act_idx], &scratch, needle);

            let rem_limit = limit - out.len();
            let mut local_matches = Vec::<(usize, Vec<u8>)>::new();
            for r_idx in 0..raw_count {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                if contains_subslice(slice, needle) {
                    let row_idx = nth_selector_bit(selector_slice, row_count, false, r_idx);
                    local_matches.push((row_idx, slice.to_vec()));
                    if local_matches.len() >= rem_limit && matches!(filter, TplFilter::Never) {
                        break;
                    }
                }
            }

            if !matches!(filter, TplFilter::Never) && ones > 0 {
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
                        if !verify || contains_subslice(row_slice, needle) {
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

            if raw_count > 0 && local_matches.len() > 1 {
                local_matches.sort_unstable_by_key(|&(r, _)| r);
            }
            for (row_idx, bytes) in local_matches.into_iter().take(rem_limit) {
                out.push(crate::query::RowMatch {
                    block_index,
                    row_index: row_idx,
                    global_row_index: global_row_base + row_idx,
                    bytes,
                });
            }
            return Ok(Some(row_count));
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
        let mut raw_idx = 0usize;
        let mut row_buf = vec![0u8; 4096];
        for r in 0..row_count {
            let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
            if sel != 0 {
                let tpl = &mut prepared_templates[sel - 1];
                if tpl.row_budget > row_buf.len() {
                    row_buf.resize(tpl.row_budget + 64, 0);
                }
                let written = unsafe {
                    let start_p = row_buf.as_mut_ptr();
                    let end_p = tpl.emit_row(start_p);
                    end_p.offset_from(start_p) as usize
                };
                if contains_subslice(&row_buf[..written], needle) {
                    out.push(crate::query::RowMatch {
                        block_index,
                        row_index: r,
                        global_row_index: global_row_base + r,
                        bytes: row_buf[..written].to_vec(),
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            } else {
                let (raw_ptr, len) = scratch.slice_arena[raw_slice_offset + raw_idx];
                raw_idx += 1;
                let s = unsafe { std::slice::from_raw_parts(raw_ptr, len) };
                if contains_subslice(s, needle) {
                    out.push(crate::query::RowMatch {
                        block_index,
                        row_index: r,
                        global_row_index: global_row_base + r,
                        bytes: s.to_vec(),
                    });
                    if out.len() >= limit {
                        break;
                    }
                }
            }
        }
        Ok(Some(row_count))
    })
}
