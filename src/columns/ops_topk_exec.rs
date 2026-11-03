// Compressed-domain `GRAMMAR_PACKET` block execution & candidate materialization for `Top-K`.

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
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(None);
    };
    if hdr.row_count == 0 {
        return Ok(Some(0));
    }
    if carry.is_some() && !hdr.first_is_raw {
        return Ok(None);
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
                    &c_bytes,
                    c_blk,
                    c_row,
                    c_glob,
                    where_expr,
                    project_cols,
                    order_col,
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

        for t in 0..hdr.template_count {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
                continue;
            }
            let segs = hdr.segs(t);
            let tcols = multi_cols.get(t);
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
            if mask.popcount(cnt) == 0 {
                continue;
            }
            let proj_plan = if let Some(pcols) = project_cols {
                let Some(pl) = plan_template_projection(segs, tcols, pcols)? else {
                    return Ok(None);
                };
                if matches!(pl, TemplateProjPlan::SkipAll) {
                    continue;
                }
                Some(pl)
            } else {
                None
            };
            let order_lookup = resolve_template_slot(segs, tcols, order_col)?;
            if matches!(order_lookup, SlotLookup::Fallback) {
                return Ok(None);
            }

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
                        &mask,
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
                        &mask,
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

            if cands.is_empty() {
                continue;
            }
            let winners = cands.into_tpl_sorted();
            if let Some(plan) = proj_plan {
                match plan {
                    TemplateProjPlan::SkipAll => unreachable!(),
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
                            seg_bytes,
                            seg_ranges,
                            slot_map,
                            tcols,
                            cnt,
                            hdr.raw_len,
                            &mut scratch,
                        )?;
                        let n_exec = scratch.unbound_col_arena.len();
                        scratch.exec_col_arena.reserve(n_exec);
                        let mut tpl = ub.bind(&mut scratch);
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
                ensure_full_raw_arena(hdr.raw_len, &mut scratch.raw_arena);
                let slots = hdr.slot_counts[t];
                let mut cols_uninit =
                    core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
                let pcols: &mut [PreparedColumn; MAX_SLOTS] =
                    unsafe { cols_uninit.assume_init_mut() };
                for s in 0..slots {
                    pcols[s] = PreparedColumn::parse_and_validate(tcols[s], cnt, &mut scratch)?;
                }
                let unbound = UnboundTemplate::build(segs, &pcols[..slots], &mut scratch);
                let n_exec = scratch.unbound_col_arena.len();
                scratch.exec_col_arena.reserve(n_exec);
                let mut tpl = unbound.bind(&mut scratch);
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
            reset_scratch_to(&mut scratch, raw_mark, slice_mark);
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
                    s,
                    block_index,
                    b_row,
                    base_row + b_row,
                    where_expr,
                    project_cols,
                    order_col,
                    shared_bound,
                );
            }
        }

        *carry = next_carry;
        Ok(Some(new_rows))
    })
}
