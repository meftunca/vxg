// Compressed-domain `GRAMMAR_PACKET` block execution & candidate materialization for `Top-K`.

#[allow(clippy::too_many_arguments)]
fn materialize_topk_winners(
    winners: Vec<CandItem>,
    segs: &[&[u8]],
    tcols: &[&[u8]],
    slots: usize,
    cnt: usize,
    hdr_raw_len: usize,
    block_index: usize,
    project_cols: Option<&[crate::ops::ColTarget<'_>]>,
    shared_bound: Option<&std::sync::atomic::AtomicU64>,
    heap: &mut crate::ops::top_k::TopKHeap,
    scratch: &mut DecodeScratch,
) -> Result<bool> {
    if let Some(pcols) = project_cols {
        let Some(plan) = plan_template_projection(segs, tcols, pcols)? else {
            return Ok(false);
        };
        match plan {
            TemplateProjPlan::SkipAll => return Ok(true),
            TemplateProjPlan::ConstJsonEmpty { crlf } => {
                let lit: &[u8] = if crlf { b"{}\r\n" } else { b"{}\n" };
                for w in winners {
                    heap.push_entry(crate::ops::top_k::HeapEntry {
                        dir: heap.dir(),
                        key: w.key,
                        block_index,
                        row_index: w.block_row,
                        global_row_index: w.global_row,
                        bytes: lit.to_vec(),
                    });
                }
            }
            TemplateProjPlan::Slots {
                ref seg_bytes,
                ref seg_ranges,
                ref slot_map,
                ..
            } => {
                let ub = prepare_projected_unbound(
                    seg_bytes, seg_ranges, slot_map, tcols, cnt, hdr_raw_len, scratch,
                )?;
                let n_exec = scratch.unbound_col_arena.len();
                scratch.exec_col_arena.reserve(n_exec);
                let mut tpl = ub.bind(scratch);
                let mut row_buf = vec![0u8; tpl.row_budget.max(1024) + 64];
                let mut cur_t = 0usize;
                for w in winners {
                    let written = unsafe {
                        tpl.skip_rows(w.tpl_idx - cur_t);
                        let sp = row_buf.as_mut_ptr();
                        let ep = tpl.emit_row(sp);
                        ep.offset_from(sp) as usize
                    };
                    cur_t = w.tpl_idx + 1;
                    heap.push_entry(crate::ops::top_k::HeapEntry {
                        dir: heap.dir(),
                        key: w.key,
                        block_index,
                        row_index: w.block_row,
                        global_row_index: w.global_row,
                        bytes: row_buf[..written].to_vec(),
                    });
                }
            }
        }
    } else {
        ensure_full_raw_arena(hdr_raw_len, &mut scratch.raw_arena);
        let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
        let pcols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
        for s in 0..slots {
            pcols[s] = PreparedColumn::parse_and_validate(tcols[s], cnt, scratch)?;
        }
        let unbound = UnboundTemplate::build(segs, &pcols[..slots], scratch);
        let n_exec = scratch.unbound_col_arena.len();
        scratch.exec_col_arena.reserve(n_exec);
        let mut tpl = unbound.bind(scratch);
        let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
        let mut cur_t = 0usize;
        for w in winners {
            let written = unsafe {
                tpl.skip_rows(w.tpl_idx - cur_t);
                let sp = row_buf.as_mut_ptr();
                let ep = tpl.emit_row(sp);
                ep.offset_from(sp) as usize
            };
            cur_t = w.tpl_idx + 1;
            heap.push_entry(crate::ops::top_k::HeapEntry {
                dir: heap.dir(),
                key: w.key,
                block_index,
                row_index: w.block_row,
                global_row_index: w.global_row,
                bytes: row_buf[..written].to_vec(),
            });
        }
    }
    heap.update_shared_bound(shared_bound);
    Ok(true)
}

#[inline]
fn selector_all_ones(sel: &[u8], row_count: usize) -> bool {
    let full = row_count >> 3;
    let rem = row_count & 7;
    let mut chunks = sel[..full].chunks_exact(8);
    for ch in &mut chunks {
        if u64::from_ne_bytes(ch.try_into().unwrap()) != u64::MAX {
            return false;
        }
    }
    for &b in chunks.remainder() {
        if b != 0xff {
            return false;
        }
    }
    rem == 0 || sel[full] == ((1u8 << rem) - 1)
}

/// Register-only single-template delimited block pruner: avoids constructing `OpsGrammarHeader`
/// (`880 B`) and `TplColSlices` (`1 KiB`) when a cached single-template delimited block has no
/// raw rows and produces zero `Top-K` candidates.
#[allow(clippy::too_many_arguments)]
fn try_prune_cached_delim_topk(
    payload: &[u8],
    global_row_base: usize,
    project_cols: Option<&[crate::ops::ColTarget<'_>]>,
    order_col: crate::ops::ColTarget<'_>,
    shared_bound: Option<&std::sync::atomic::AtomicU64>,
    heap: &crate::ops::top_k::TopKHeap,
) -> Result<Option<usize>> {
    let crate::ops::ColTarget::Delimited {
        delim: od,
        index: os,
    } = order_col
    else {
        return Ok(None);
    };
    if od.is_ascii_digit() || payload.first() != Some(&GRAMMAR_PACKET) {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&payload[1..]);
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE {
        return Err(invalid("column raw length"));
    }
    let rest = &payload[1 + cursor.pos..];
    let Some((slots, Some(d), wlen)) = try_match_cached_single_tpl(rest) else {
        return Ok(None);
    };
    if d != od {
        return Ok(None);
    }
    if let Some(pcols) = project_cols {
        let crate::ops::ColTarget::Delimited { delim: pd, .. } = pcols[0] else {
            return Ok(None);
        };
        if pd != d || pd.is_ascii_digit() {
            return Ok(None);
        }
    }
    cursor.pos += wlen;
    let row_count = cursor.uleb()?;
    if row_count > raw_len.saturating_add(1) {
        return Err(invalid("column row count"));
    }
    if row_count == 0 {
        return Ok(None);
    }
    let sel_w = cursor.byte()?;
    if sel_w != 1 {
        return Err(invalid("column row selector width"));
    }
    let sel_len = bit_bytes(row_count, 1)?;
    let sel_slice = cursor.take(sel_len)?;
    if !selector_all_ones(sel_slice, row_count) {
        return Ok(None);
    }
    let mut col_slices: [&[u8]; MAX_SLOTS] = [&[]; MAX_SLOTS];
    read_single_tpl_col_slices(&mut cursor, slots, &mut col_slices)?;
    let order_col_bytes: &[u8] = if os < slots { col_slices[os] } else { &[] };
    let mut row_map = TplRowMap::Offset(0);
    let mut cands = CandTracker::new(heap.capacity_k(), heap.dir());
    let scanned = if os >= slots {
        push_const_key_cands(
            crate::ops::top_k::SortKeyRef::Missing,
            row_count,
            &TplMask::All,
            &mut row_map,
            global_row_base,
            heap,
            &mut cands,
        );
        true
    } else {
        try_scan_scaled_dec_topk(
            order_col_bytes,
            row_count,
            &TplMask::All,
            &mut row_map,
            global_row_base,
            false,
            heap,
            &mut cands,
        )? || try_scan_decimal_topk(
            order_col_bytes,
            row_count,
            &TplMask::All,
            &mut row_map,
            global_row_base,
            false,
            shared_bound,
            heap,
            &mut cands,
        )?
    };
    if scanned && cands.is_empty() {
        Ok(Some(row_count))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn topk_grammar_block(
    payload: &[u8],
    block_index: usize,
    global_row_base: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    eval_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    project_cols: Option<&[crate::ops::ColTarget<'_>]>,
    order_col: crate::ops::ColTarget<'_>,
    shared_bound: Option<&std::sync::atomic::AtomicU64>,
    heap: &mut crate::ops::top_k::TopKHeap,
) -> Result<Option<usize>> {
    if carry.is_none() && where_expr.is_none() {
        if let Some(rc) = try_prune_cached_delim_topk(
            payload,
            global_row_base,
            project_cols,
            order_col,
            shared_bound,
            heap,
        )? {
            return Ok(Some(rc));
        }
    }
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(None);
    };
    if hdr.row_count == 0 {
        return Ok(Some(0));
    }
    if carry.is_some() && !hdr.first_is_raw {
        return Ok(None);
    }

    // Zero-`DecodeScratch` & zero-alloc fast path for single-template blocks without raw rows or `WHERE`.
    if hdr.template_count == 1 && hdr.raw_count == 0 && carry.is_none() && where_expr.is_none() {
        let multi_cols = read_all_template_col_slices(&mut hdr)?;
        let segs = hdr.segs(0);
        let tcols = multi_cols.get(0);
        let kd = Some(hdr.tpl0_delim);
        if let Some(pcols) = project_cols {
            match check_template_projection_fast(segs, tcols, pcols, kd)? {
                ProjPrecheck::Fallback => return Ok(None),
                ProjPrecheck::SkipAll => return Ok(Some(hdr.row_count)),
                ProjPrecheck::Ok => {}
            }
        }
        let order_lookup = resolve_template_slot_with_delim(segs, tcols, order_col, kd)?;
        if matches!(order_lookup, SlotLookup::Fallback) {
            return Ok(None);
        }
        let cnt = hdr.tpl_row_counts[0];
        let mut row_map = TplRowMap::Offset(0);
        let mut cands = CandTracker::new(heap.capacity_k(), heap.dir());
        let digit_delim = matches!(
            order_col,
            crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
        );
        let scanned_no_scratch = match order_lookup {
            SlotLookup::Absent => {
                push_const_key_cands(
                    crate::ops::top_k::SortKeyRef::Missing,
                    cnt,
                    &TplMask::All,
                    &mut row_map,
                    global_row_base,
                    heap,
                    &mut cands,
                );
                true
            }
            SlotLookup::Slot(os) => {
                try_scan_scaled_dec_topk(
                    tcols[os],
                    cnt,
                    &TplMask::All,
                    &mut row_map,
                    global_row_base,
                    digit_delim,
                    heap,
                    &mut cands,
                )? || try_scan_decimal_topk(
                    tcols[os],
                    cnt,
                    &TplMask::All,
                    &mut row_map,
                    global_row_base,
                    digit_delim,
                    shared_bound,
                    heap,
                    &mut cands,
                )?
            }
            SlotLookup::Fallback => unreachable!(),
        };
        if scanned_no_scratch && cands.is_empty() {
            return Ok(Some(hdr.row_count));
        }
        return DECODE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            ensure_proj_raw_arena(&hdr, project_cols.map_or(0, |c| c.len()), &mut scratch.raw_arena);
            if !scanned_no_scratch {
                if let SlotLookup::Slot(os) = order_lookup {
                    scan_order_slot_cands_scratch(
                        tcols[os],
                        cnt,
                        &TplMask::All,
                        &mut row_map,
                        global_row_base,
                        order_col,
                        digit_delim,
                        hdr.raw_len,
                        shared_bound,
                        heap,
                        &mut cands,
                        &mut scratch,
                    )?;
                    reset_scratch_to(&mut scratch, 0, 0);
                }
            }
            if !cands.is_empty()
                && !materialize_topk_winners(
                    cands.into_tpl_sorted(),
                    segs,
                    tcols,
                    hdr.slot_counts[0],
                    cnt,
                    hdr.raw_len,
                    block_index,
                    project_cols,
                    shared_bound,
                    heap,
                    &mut scratch,
                )?
            {
                return Ok(None);
            }
            Ok(Some(hdr.row_count))
        });
    }

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        ensure_proj_raw_arena(&hdr, project_cols.map_or(0, |c| c.len()), &mut scratch.raw_arena);
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

        // Precheck all templates BEFORE `carry.take()` so fallback (`Ok(None)`) never drops `carry`.
        let mut tpl_masks: [Option<(TplMask, SlotLookup)>; 16] = [const { None }; 16];
        for (t, out_slot) in tpl_masks.iter_mut().enumerate().take(hdr.template_count) {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
                continue;
            }
            let segs = hdr.segs(t);
            let tcols = multi_cols.get(t);
            let kd = (t == 0).then_some(hdr.tpl0_delim);
            if let Some(pcols) = project_cols {
                match check_template_projection_fast(segs, tcols, pcols, kd)? {
                    ProjPrecheck::Fallback => return Ok(None),
                    ProjPrecheck::SkipAll => continue,
                    ProjPrecheck::Ok => {}
                }
            }
            let order_lookup = resolve_template_slot_with_delim(segs, tcols, order_col, kd)?;
            if matches!(order_lookup, SlotLookup::Fallback) {
                return Ok(None);
            }
            let mask = if let Some(expr) = where_expr {
                let Some(m) = eval_expr_on_template(
                    segs,
                    tcols,
                    cnt,
                    expr,
                    hdr.raw_len,
                    raw_mark,
                    slice_mark,
                    &mut scratch,
                )?
                else {
                    return Ok(None);
                };
                m
            } else {
                TplMask::All
            };
            if mask.popcount(cnt) > 0 {
                *out_slot = Some((mask, order_lookup));
            }
        }

        let had_incoming_carry = carry.is_some();
        let base_row = global_row_base - usize::from(had_incoming_carry);
        let new_rows = hdr.row_count - usize::from(had_incoming_carry);

        let mut raw_start = 0usize;
        if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
            if hdr.first_is_raw && hdr.raw_count > 0 {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset];
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                if eval_carry {
                    c_bytes.extend_from_slice(slice);
                }
                raw_start = 1;
                if eval_carry && !c_bytes.ends_with(b"\n") && hdr.row_count == 1 {
                    *carry = Some((c_blk, c_row, c_glob, c_bytes));
                    return Ok(Some(0));
                }
            }
            if eval_carry {
                heap.push_line_if_match(
                    &c_bytes, c_blk, c_row, c_glob, where_expr, project_cols, order_col,
                    shared_bound,
                );
            }
        }
        let mut raw_end = hdr.raw_count;
        let mut next_carry = None;
        if hdr.last_is_raw && raw_end > raw_start {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            if !slice.ends_with(b"\n") {
                if eval_carry {
                    let last_r = hdr.row_count - 1;
                    next_carry = Some((block_index, last_r, base_row + last_r, slice.to_vec()));
                }
                raw_end -= 1;
            }
        }

        let mut multi_tpl_rows: Vec<Vec<usize>> = Vec::new();
        let mut multi_raw_rows: Vec<usize> = Vec::new();
        if hdr.template_count > 1 {
            let sel_off = unpack_and_validate_u8(
                hdr.selector_slice,
                hdr.row_count,
                hdr.selector_width,
                0,
                hdr.template_count + 1,
                &mut scratch.u8_arena,
            )?;
            multi_tpl_rows = (0..hdr.template_count)
                .map(|t| Vec::with_capacity(hdr.tpl_row_counts[t]))
                .collect();
            multi_raw_rows.reserve(hdr.raw_count);
            for r in 0..hdr.row_count {
                let sel = scratch.u8_arena[sel_off + r] as usize;
                if sel == 0 {
                    multi_raw_rows.push(r);
                } else {
                    multi_tpl_rows[sel - 1].push(r);
                }
            }
            reset_scratch_to(&mut scratch, raw_mark, slice_mark);
        }

        for (t, slot_plan) in tpl_masks.iter().enumerate().take(hdr.template_count) {
            let Some((ref mask, order_lookup)) = *slot_plan else {
                continue;
            };
            let cnt = hdr.tpl_row_counts[t];
            let segs = hdr.segs(t);
            let tcols = multi_cols.get(t);
            let mut row_map = if hdr.template_count > 1 {
                TplRowMap::Explicit(&multi_tpl_rows[t])
            } else if hdr.raw_count == usize::from(hdr.first_is_raw) + usize::from(hdr.last_is_raw)
            {
                TplRowMap::Offset(usize::from(hdr.first_is_raw))
            } else {
                TplRowMap::Cursor(SelectorBitCursor::new(
                    hdr.selector_slice,
                    hdr.row_count,
                    true,
                ))
            };

            let mut cands = CandTracker::new(heap.capacity_k(), heap.dir());
            match order_lookup {
                SlotLookup::Absent => {
                    push_const_key_cands(
                        crate::ops::top_k::SortKeyRef::Missing,
                        cnt,
                        mask,
                        &mut row_map,
                        base_row,
                        heap,
                        &mut cands,
                    );
                }
                SlotLookup::Slot(os) => {
                    scan_order_slot_cands(
                        tcols[os],
                        cnt,
                        mask,
                        &mut row_map,
                        base_row,
                        order_col,
                        hdr.raw_len,
                        shared_bound,
                        heap,
                        &mut cands,
                        &mut scratch,
                    )?;
                    reset_scratch_to(&mut scratch, raw_mark, slice_mark);
                }
                SlotLookup::Fallback => unreachable!(),
            }

            if !cands.is_empty() {
                materialize_topk_winners(
                    cands.into_tpl_sorted(),
                    segs,
                    tcols,
                    hdr.slot_counts[t],
                    cnt,
                    hdr.raw_len,
                    block_index,
                    project_cols,
                    shared_bound,
                    heap,
                    &mut scratch,
                )?;
                reset_scratch_to(&mut scratch, raw_mark, slice_mark);
            }
        }

        if raw_start < raw_end {
            let mut raw_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, false);
            for ri in raw_start..raw_end {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + ri];
                let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                let b_row = if hdr.template_count > 1 {
                    multi_raw_rows[ri]
                } else {
                    raw_cur.advance_to(ri)
                };
                heap.push_line_if_match(
                    s, block_index, b_row, base_row + b_row, where_expr, project_cols, order_col,
                    shared_bound,
                );
            }
        }

        *carry = next_carry;
        Ok(Some(new_rows))
    })
}
