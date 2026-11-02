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
        scratch.raw_arena.reserve(hdr.raw_len.saturating_mul(4) + 163_840);

        let raw_slice_offset = {
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            read_grammar_raw_rows(&mut hdr.cursor, hdr.raw_count, raw_arena, slice_arena)?
        };

        if hdr.template_count > 1 {
            let raw_mark = scratch.raw_arena.len();
            let slice_mark = scratch.slice_arena.len();
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
            let rem_cap = limit - out.len();
            let mut tpl_hits: Vec<Vec<usize>> = Vec::with_capacity(hdr.template_count);
            let mut any_tpl_hit = false;
            for t in 0..hdr.template_count {
                let cnt = hdr.tpl_row_counts[t];
                if cnt == 0 {
                    tpl_hits.push(Vec::new());
                    continue;
                }
                match resolve_template_slot(hdr.segs(t), &multi_cols[t], pred.col)? {
                    SlotLookup::Fallback => return Ok(None),
                    SlotLookup::Absent => tpl_hits.push(Vec::new()),
                    SlotLookup::Slot(s) => {
                        let (_, h) = eval_col_pred_on_slot(
                            multi_cols[t][s],
                            cnt,
                            rem_cap,
                            pred,
                            &mut scratch,
                        )?;
                        reset_scratch_to(&mut scratch, raw_mark, slice_mark);
                        if !h.is_empty() {
                            any_tpl_hit = true;
                        }
                        tpl_hits.push(h);
                    }
                }
            }
            if !any_tpl_hit {
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
            let sel_u8_offset = unpack_and_validate_u8(
                hdr.selector_slice,
                hdr.row_count,
                hdr.selector_width,
                0,
                hdr.template_count + 1,
                &mut scratch.u8_arena,
            )?;
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            let mut unbounds: Vec<Option<UnboundTemplate>> = vec![None; hdr.template_count];
            for t in 0..hdr.template_count {
                if tpl_hits[t].is_empty() {
                    continue;
                }
                let slots = hdr.slot_counts[t];
                let cnt = hdr.tpl_row_counts[t];
                for s in 0..slots {
                    cols[s] = PreparedColumn::parse_and_validate(multi_cols[t][s], cnt, &mut scratch)?;
                }
                unbounds[t] = Some(UnboundTemplate::build(hdr.segs(t), &cols[..slots], &mut scratch));
            }
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut bound_tpls: Vec<Option<PreparedTemplate>> =
                unbounds.into_iter().map(|u| u.map(|ub| ub.bind(&mut scratch))).collect();
            let mut tpl_cur = [0usize; 16];
            let mut emitted_t = [0usize; 16];
            let mut hit_pos = [0usize; 16];
            let mut raw_cur = 0usize;
            let mut row_buf = vec![0u8; 4096];
            for r in 0..hdr.row_count {
                if out.len() >= limit {
                    break;
                }
                let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
                if sel == 0 {
                    let ri = raw_cur;
                    raw_cur += 1;
                    if ri >= raw_start && ri < raw_end {
                        let (ptr, len) = scratch.slice_arena[raw_slice_offset + ri];
                        let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                        if crate::ops::line_matches_col_pred(s, pred) {
                            out.push(crate::query::RowMatch {
                                block_index,
                                row_index: r,
                                global_row_index: base_row + r,
                                bytes: s.to_vec(),
                            });
                        }
                    }
                } else {
                    let t = sel - 1;
                    let ti = tpl_cur[t];
                    tpl_cur[t] += 1;
                    let hp = hit_pos[t];
                    if hp < tpl_hits[t].len() && tpl_hits[t][hp] == ti {
                        hit_pos[t] = hp + 1;
                        let tpl = bound_tpls[t].as_mut().unwrap();
                        if tpl.row_budget > row_buf.len() {
                            row_buf.resize(tpl.row_budget + 64, 0);
                        }
                        let written = unsafe {
                            tpl.skip_rows(ti - emitted_t[t]);
                            let sp = row_buf.as_mut_ptr();
                            let ep = tpl.emit_row(sp);
                            ep.offset_from(sp) as usize
                        };
                        emitted_t[t] = ti + 1;
                        out.push(crate::query::RowMatch {
                            block_index,
                            row_index: r,
                            global_row_index: base_row + r,
                            bytes: row_buf[..written].to_vec(),
                        });
                    }
                }
            }
            *carry = next_carry;
            return Ok(Some(new_rows));
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
        let mut raw_sel_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, false);
        for r_idx in raw_start..raw_end {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            if crate::ops::line_matches_col_pred(slice, pred) {
                let row_idx = raw_sel_cur.advance_to(r_idx);
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
            let mut tpl_sel_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, true);
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
                        tpl_sel_cur.advance_to(t_idx)
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

pub(crate) enum GrammarFilterBytesResult {
    Fallback,
    Empty,
    AllCompleteMatch,
    Bytes,
}

/// Evaluate `pred` on the owned complete rows of a `GRAMMAR_PACKET` block for `filter_frame`,
/// writing matching rows straight into `dst` without per-row `RowMatch` `Vec<u8>` allocations,
/// and detecting 100%-matching intact blocks (`AllCompleteMatch`) for zero-copy pass-through.
pub(crate) fn filter_grammar_block_into_bytes(
    payload: &[u8],
    had_in_carry: bool,
    pred: &crate::ops::ColPred<'_>,
    dst: &mut Vec<u8>,
) -> Result<GrammarFilterBytesResult> {
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(GrammarFilterBytesResult::Fallback);
    };
    if hdr.row_count == 0 {
        return Ok(GrammarFilterBytesResult::Empty);
    }
    if had_in_carry && !hdr.first_is_raw {
        return Ok(GrammarFilterBytesResult::Fallback);
    }
    if hdr.template_count != 1 {
        return Ok(GrammarFilterBytesResult::Fallback);
    }

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.raw_arena.reserve(hdr.raw_len.saturating_mul(4) + 163_840);

        let raw_slice_offset = {
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            read_grammar_raw_rows(&mut hdr.cursor, hdr.raw_count, raw_arena, slice_arena)?
        };

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
            return Ok(GrammarFilterBytesResult::Fallback);
        }

        let raw_start = usize::from(had_in_carry && hdr.first_is_raw && hdr.raw_count > 0);
        let mut raw_end = hdr.raw_count;
        let mut has_unterm_tail = false;
        if hdr.last_is_raw && raw_end > raw_start {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + hdr.raw_count - 1];
            let s = unsafe { std::slice::from_raw_parts(ptr, len) };
            if !s.ends_with(b"\n") {
                raw_end = hdr.raw_count - 1;
                has_unterm_tail = true;
            }
        }

        let (target_col_opt, hits) = match lookup {
            SlotLookup::Slot(s) => {
                let (c_opt, h) =
                    eval_col_pred_on_slot(col_slices[s], ones, ones, pred, &mut scratch)?;
                (c_opt.map(|c| (s, c)), h)
            }
            SlotLookup::Absent => (None, Vec::new()),
            SlotLookup::Fallback => unreachable!(),
        };

        let mut raw_hits: Vec<(usize, usize)> = Vec::new();
        let mut raw_sel_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, false);
        for r_idx in raw_start..raw_end {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            if crate::ops::line_matches_col_pred(slice, pred) {
                raw_hits.push((raw_sel_cur.advance_to(r_idx), r_idx));
            }
        }

        if hits.is_empty() && raw_hits.is_empty() {
            return Ok(GrammarFilterBytesResult::Empty);
        }
        if !had_in_carry
            && !has_unterm_tail
            && hits.len() == ones
            && raw_hits.len() == hdr.raw_count
        {
            return Ok(GrammarFilterBytesResult::AllCompleteMatch);
        }

        let mut rm_pos = 0usize;
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
            let budget = tpl.row_budget.max(512) + 64;
            dst.reserve(hits.len().saturating_mul(96) + budget);
            let mut cur_t = 0usize;
            let mut tpl_sel_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, true);
            for &t_idx in &hits {
                let row_idx = if hdr.raw_count == 0 {
                    t_idx
                } else {
                    tpl_sel_cur.advance_to(t_idx)
                };
                while rm_pos < raw_hits.len() && raw_hits[rm_pos].0 < row_idx {
                    let (ptr, len) = scratch.slice_arena[raw_slice_offset + raw_hits[rm_pos].1];
                    dst.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
                    rm_pos += 1;
                }
                dst.reserve(budget);
                unsafe {
                    tpl.skip_rows(t_idx - cur_t);
                    let cur_len = dst.len();
                    let sp = dst.as_mut_ptr().add(cur_len);
                    let ep = tpl.emit_row(sp);
                    let written = ep.offset_from(sp) as usize;
                    dst.set_len(cur_len + written);
                }
                cur_t = t_idx + 1;
            }
        }
        while rm_pos < raw_hits.len() {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + raw_hits[rm_pos].1];
            dst.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
            rm_pos += 1;
        }
        Ok(GrammarFilterBytesResult::Bytes)
    })
}
