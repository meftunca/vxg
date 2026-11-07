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

        if hdr.template_count > 1 {
            let raw_mark = scratch.raw_arena.len();
            let slice_mark = scratch.slice_arena.len();
            let multi_cols = read_all_template_col_slices(&mut hdr)?;
            let mut tpl_hits: [Vec<usize>; MAX_TEMPLATES] =
                std::array::from_fn(|_| Vec::new());
            let mut any_tpl_hit = false;
            let mut all_tpl_match = true;
            for t in 0..hdr.template_count {
                let cnt = hdr.tpl_row_counts[t];
                if cnt == 0 {
                    continue;
                }
                let tc = multi_cols.get(t);
                match resolve_template_slot(hdr.segs(t), tc, pred.col)? {
                    SlotLookup::Fallback => return Ok(GrammarFilterBytesResult::Fallback),
                    SlotLookup::Absent => {
                        all_tpl_match = false;
                    }
                    SlotLookup::Slot(s) => {
                        let (_, h) =
                            eval_col_pred_on_slot(tc[s], cnt, cnt, pred, &mut scratch)?;
                        reset_scratch_to(&mut scratch, raw_mark, slice_mark);
                        if h.len() != cnt {
                            all_tpl_match = false;
                        }
                        if !h.is_empty() {
                            any_tpl_hit = true;
                            tpl_hits[t] = h;
                        }
                    }
                }
            }

            let mut raw_match_mask = 0u64;
            let mut raw_hits_vec: Vec<usize> = Vec::new();
            let mut raw_match_cnt = 0usize;
            for r_idx in raw_start..raw_end {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                if crate::ops::line_matches_col_pred(slice, pred) {
                    raw_match_cnt += 1;
                    if r_idx < 64 {
                        raw_match_mask |= 1u64 << r_idx;
                    } else {
                        raw_hits_vec.push(r_idx);
                    }
                }
            }

            if !any_tpl_hit && raw_match_cnt == 0 {
                return Ok(GrammarFilterBytesResult::Empty);
            }
            if !had_in_carry
                && !has_unterm_tail
                && all_tpl_match
                && raw_match_cnt == hdr.raw_count
            {
                return Ok(GrammarFilterBytesResult::AllCompleteMatch);
            }

            if !any_tpl_hit {
                for r_idx in raw_start..raw_end {
                    let hit = if r_idx < 64 {
                        ((raw_match_mask >> r_idx) & 1) != 0
                    } else {
                        raw_hits_vec.binary_search(&r_idx).is_ok()
                    };
                    if hit {
                        let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
                        dst.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
                    }
                }
                return Ok(GrammarFilterBytesResult::Bytes);
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
            let mut unbounds: [Option<UnboundTemplate>; MAX_TEMPLATES] =
                std::array::from_fn(|_| None);
            for t in 0..hdr.template_count {
                if tpl_hits[t].is_empty() {
                    continue;
                }
                let slots = hdr.slot_counts[t];
                let cnt = hdr.tpl_row_counts[t];
                let tc = multi_cols.get(t);
                for s in 0..slots {
                    cols[s] = PreparedColumn::parse_and_validate(tc[s], cnt, &mut scratch)?;
                }
                unbounds[t] = Some(UnboundTemplate::build(hdr.segs(t), &cols[..slots], &mut scratch));
            }
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut bound_tpls: [Option<PreparedTemplate>; MAX_TEMPLATES] =
                unbounds.map(|u| u.map(|ub| ub.bind(&mut scratch)));
            let mut tpl_cur = [0usize; MAX_TEMPLATES];
            let mut emitted_t = [0usize; MAX_TEMPLATES];
            let mut hit_pos = [0usize; MAX_TEMPLATES];
            let mut raw_cur = 0usize;
            for r in 0..hdr.row_count {
                let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
                if sel == 0 {
                    let ri = raw_cur;
                    raw_cur += 1;
                    if ri >= raw_start && ri < raw_end {
                        let hit = if ri < 64 {
                            ((raw_match_mask >> ri) & 1) != 0
                        } else {
                            raw_hits_vec.binary_search(&ri).is_ok()
                        };
                        if hit {
                            let (ptr, len) = scratch.slice_arena[raw_slice_offset + ri];
                            dst.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, len) });
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
                        dst.reserve(tpl.row_budget.max(512) + 64);
                        unsafe {
                            tpl.skip_rows(ti - emitted_t[t]);
                            let cur_len = dst.len();
                            let sp = dst.as_mut_ptr().add(cur_len);
                            let ep = tpl.emit_row(sp);
                            let written = ep.offset_from(sp) as usize;
                            dst.set_len(cur_len + written);
                        }
                        emitted_t[t] = ti + 1;
                    }
                }
            }
            return Ok(GrammarFilterBytesResult::Bytes);
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
            return Ok(GrammarFilterBytesResult::Fallback);
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
