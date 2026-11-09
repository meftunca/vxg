// Block-level compressed-domain Apache Arrow scanner (`GRAMMAR_PACKET`), coordinating
// `PredExpr` template masks, carry boundary fragments, CSV header row-0 skipping, and
// per-column `push_slot_column_masked` dispatch.

pub(crate) fn scan_arrow_grammar_block_par(
    payload: &[u8],
    had_in_carry: bool,
    mut skip_first_row: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    rem_limit: usize,
    sinks: &mut [crate::arrow_builder::ColBuilder],
) -> Result<Option<usize>> {
    let mut dummy: Option<Vec<u8>> = had_in_carry.then(Vec::new);
    scan_arrow_grammar_block_inner(
        payload,
        &mut dummy,
        false,
        &mut skip_first_row,
        where_expr,
        cols,
        rem_limit,
        sinks,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_arrow_grammar_block_seq(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    skip_first_row: &mut bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    rem_limit: usize,
    sinks: &mut [crate::arrow_builder::ColBuilder],
) -> Result<Option<usize>> {
    scan_arrow_grammar_block_inner(
        payload,
        carry,
        true,
        skip_first_row,
        where_expr,
        cols,
        rem_limit,
        sinks,
    )
}

#[allow(clippy::too_many_arguments)]
fn scan_arrow_grammar_block_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    skip_first_row: &mut bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    rem_limit: usize,
    sinks: &mut [crate::arrow_builder::ColBuilder],
) -> Result<Option<usize>> {
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(None);
    };
    if rem_limit == 0 || hdr.row_count == 0 {
        return Ok(Some(0));
    }
    if carry.is_some() && !hdr.first_is_raw {
        return Ok(None);
    }

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        ensure_proj_raw_arena(&hdr, cols.len(), &mut scratch.raw_arena);
        let raw_slice_offset = {
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            read_grammar_raw_rows(&mut hdr.cursor, hdr.raw_count, raw_arena, slice_arena)?
        };
        let raw_mark = scratch.raw_arena.len();
        let slice_mark = scratch.slice_arena.len();
        let multi_cols = read_all_template_col_slices(&mut hdr)?;

        let mut masks: [TplMask; 16] = std::array::from_fn(|_| TplMask::Empty);
        let mut tpl_slots: Vec<Vec<Option<usize>>> = Vec::with_capacity(hdr.template_count);
        let mut any_active_tpl = false;

        for (t, mask_slot) in masks.iter_mut().enumerate().take(hdr.template_count) {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
                tpl_slots.push(Vec::new());
                continue;
            }
            let segs = hdr.segs(t);
            let tcols = multi_cols.get(t);
            let mask = if let Some(expr) = where_expr {
                let Some(m) = eval_expr_on_template(
                    segs, tcols, cnt, expr, hdr.raw_len, raw_mark, slice_mark, &mut scratch,
                )?
                else {
                    return Ok(None);
                };
                m
            } else {
                TplMask::All
            };
            if mask.popcount(cnt) == 0 {
                tpl_slots.push(Vec::new());
                continue;
            }
            let known_delim = (t == 0).then_some(hdr.tpl0_delim);
            let mut slot_map = Vec::with_capacity(cols.len());
            for &col in cols {
                if matches!(col, crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()) {
                    return Ok(None);
                }
                match resolve_template_slot_with_delim(segs, tcols, col, known_delim)? {
                    SlotLookup::Slot(s) => slot_map.push(Some(s)),
                    SlotLookup::Absent => slot_map.push(None),
                    SlotLookup::Fallback => return Ok(None),
                }
            }
            any_active_tpl = true;
            *mask_slot = mask;
            tpl_slots.push(slot_map);
        }

        let peek_skip_raw0 =
            carry.is_none() && *skip_first_row && hdr.row_count > 0 && hdr.first_is_raw && hdr.raw_count > 0;
        if peek_skip_raw0 && !eval_carry && hdr.row_count == 1 {
            let (p, l) = scratch.slice_arena[raw_slice_offset];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                return Ok(None);
            }
        }
        let peek_raw_start = usize::from(
            (carry.is_some() && hdr.first_is_raw && hdr.raw_count > 0) || peek_skip_raw0,
        );
        let mut peek_raw_end = hdr.raw_count;
        if hdr.last_is_raw && peek_raw_end > peek_raw_start {
            let (p, l) = scratch.slice_arena[raw_slice_offset + peek_raw_end - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                peek_raw_end -= 1;
            }
        }
        if any_active_tpl && (hdr.template_count > 1 || peek_raw_start != peek_raw_end) {
            return Ok(None);
        }

        let mut emitted = 0usize;
        let mut raw_start = 0usize;
        if let Some(mut c_bytes) = carry.take() {
            if hdr.first_is_raw && hdr.raw_count > 0 {
                let (p, l) = scratch.slice_arena[raw_slice_offset];
                let s = unsafe { std::slice::from_raw_parts(p, l) };
                if eval_carry {
                    c_bytes.extend_from_slice(s);
                }
                raw_start = 1;
                if eval_carry && !c_bytes.ends_with(b"\n") && hdr.row_count == 1 {
                    *carry = Some(c_bytes);
                    return Ok(Some(0));
                }
            }
            if eval_carry {
                if *skip_first_row {
                    *skip_first_row = false;
                } else if emitted < rem_limit
                    && where_expr.is_none_or(|e| crate::ops::line_matches_expr(&c_bytes, e))
                {
                    for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                        sink.push_line_col(&c_bytes, col);
                    }
                    emitted += 1;
                }
            }
        } else if *skip_first_row && hdr.row_count > 0 && hdr.first_is_raw && hdr.raw_count > 0 {
            let (p, l) = scratch.slice_arena[raw_slice_offset];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if hdr.row_count == 1 && !s.ends_with(b"\n") {
                *carry = Some(s.to_vec());
                return Ok(Some(0));
            }
            raw_start = 1;
            *skip_first_row = false;
        }

        let mut raw_end = hdr.raw_count;
        let mut next_carry: Option<Vec<u8>> = None;
        if hdr.last_is_raw && raw_end > raw_start {
            let (p, l) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                if eval_carry {
                    next_carry = Some(s.to_vec());
                }
                raw_end -= 1;
            }
        }
        if emitted >= rem_limit {
            *carry = next_carry;
            return Ok(Some(emitted));
        }

        if !any_active_tpl {
            if *skip_first_row {
                *skip_first_row = false;
            }
            for ri in raw_start..raw_end {
                if emitted >= rem_limit {
                    break;
                }
                let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
                let s = unsafe { std::slice::from_raw_parts(p, l) };
                if where_expr.is_none_or(|e| crate::ops::line_matches_expr(s, e)) {
                    for (sink, &col) in sinks.iter_mut().zip(cols.iter()) {
                        sink.push_line_col(s, col);
                    }
                    emitted += 1;
                }
            }
            *carry = next_carry;
            return Ok(Some(emitted));
        }

        debug_assert!(hdr.template_count == 1 && raw_start == raw_end);
        let ones = hdr.tpl_row_counts[0];
        if *skip_first_row {
            *skip_first_row = false;
            match masks[0] {
                TplMask::Empty => {}
                TplMask::All => {
                    if ones <= 1 {
                        masks[0] = TplMask::Empty;
                    } else {
                        let mut bs = RowBitset::all_zeros(ones);
                        bs.invert();
                        bs.words_mut()[0] &= !1u64;
                        masks[0] = TplMask::Bits(bs, ones - 1);
                    }
                }
                TplMask::Bits(ref mut bs, ref mut pop) => {
                    if (bs.words()[0] & 1) != 0 {
                        bs.words_mut()[0] &= !1u64;
                        *pop -= 1;
                        if *pop == 0 {
                            masks[0] = TplMask::Empty;
                        }
                    }
                }
            }
        }
        let pop = masks[0].popcount(ones);
        if pop > 0 {
            let start_len = sinks.first().map_or(0, |s| s.len);
            let tcols = multi_cols.get(0);
            let slot_map = &tpl_slots[0];
            for ((sink, &col), &s_opt) in sinks.iter_mut().zip(cols.iter()).zip(slot_map.iter()) {
                if let Some(s) = s_opt {
                    push_slot_column_masked(
                        tcols[s],
                        ones,
                        &masks[0],
                        col,
                        hdr.raw_len,
                        raw_mark,
                        slice_mark,
                        &mut scratch,
                        sink,
                    )?;
                } else {
                    sink.push_nulls(pop);
                }
            }
            let allowed = (rem_limit - emitted).min(pop);
            if allowed < pop {
                for sink in sinks.iter_mut() {
                    sink.truncate(start_len + allowed);
                }
            }
            emitted += allowed;
        }
        *carry = next_carry;
        Ok(Some(emitted))
    })
}
