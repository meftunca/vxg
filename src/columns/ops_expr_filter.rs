#[allow(clippy::too_many_arguments)]
fn advance_expr_carry(
    hdr: &OpsGrammarHeader<'_>,
    raw_slice_offset: usize,
    block_index: usize,
    base_row: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    expr: &crate::ops::PredExpr<'_>,
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
    scratch: &DecodeScratch,
) -> (
    usize,
    usize,
    Option<(usize, usize, usize, Vec<u8>)>,
    Option<usize>,
) {
    let mut raw_start = 0usize;
    if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
        if hdr.first_is_raw && hdr.raw_count > 0 {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset];
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            c_bytes.extend_from_slice(slice);
            raw_start = 1;
            if !c_bytes.ends_with(b"\n") && hdr.row_count == 1 {
                *carry = Some((c_blk, c_row, c_glob, c_bytes));
                return (1, 1, None, Some(0));
            }
        }
        if crate::ops::line_matches_expr(&c_bytes, expr) && out.len() < limit {
            out.push(crate::query::RowMatch {
                block_index: c_blk,
                row_index: c_row,
                global_row_index: c_glob,
                bytes: c_bytes,
            });
        }
    }
    let mut raw_end = hdr.raw_count;
    let mut next_carry = None;
    if hdr.last_is_raw && raw_end > raw_start {
        let (ptr, len) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        if !slice.ends_with(b"\n") {
            let last_r = hdr.row_count - 1;
            next_carry = Some((block_index, last_r, base_row + last_r, slice.to_vec()));
            raw_end -= 1;
        }
    }
    (raw_start, raw_end, next_carry, None)
}

pub(crate) fn search_grammar_block_expr(
    payload: &[u8],
    block_index: usize,
    global_row_base: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    expr: &crate::ops::PredExpr<'_>,
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
        ensure_raw_rows_arena(&hdr, &mut scratch.raw_arena);
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

        let rem_cap = limit - out.len();
        let mut tpl_hits: Vec<Vec<usize>> = Vec::with_capacity(hdr.template_count);
        let mut any_tpl_hit = false;
        for t in 0..hdr.template_count {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
                tpl_hits.push(Vec::new());
                continue;
            }
            let Some(mask) = eval_expr_on_template(
                hdr.segs(t),
                multi_cols.get(t),
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
            let h = mask.into_hits(cnt, rem_cap);
            if !h.is_empty() {
                any_tpl_hit = true;
            }
            tpl_hits.push(h);
        }

        let had_incoming_carry = carry.is_some();
        let base_row = global_row_base - usize::from(had_incoming_carry);
        let (raw_start, raw_end, next_carry, early_ret) = advance_expr_carry(
            &hdr,
            raw_slice_offset,
            block_index,
            base_row,
            carry,
            expr,
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

        if !any_tpl_hit {
            for ri in raw_start..raw_end {
                if out.len() >= limit {
                    break;
                }
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + ri];
                let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                if crate::ops::line_matches_expr(s, expr) {
                    let row_idx =
                        nth_zero_selector(hdr.selector_slice, hdr.row_count, hdr.selector_width, ri);
                    out.push(crate::query::RowMatch {
                        block_index,
                        row_index: row_idx,
                        global_row_index: base_row + row_idx,
                        bytes: s.to_vec(),
                    });
                }
            }
            *carry = next_carry;
            return Ok(Some(new_rows));
        }

        ensure_full_raw_arena(hdr.raw_len, &mut scratch.raw_arena);
        if hdr.template_count == 1 {
            let rem_limit = limit - out.len();
            let mut local_matches: Vec<(usize, Vec<u8>)> = Vec::new();
            let mut raw_sel_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, false);
            for ri in raw_start..raw_end {
                let (ptr, len) = scratch.slice_arena[raw_slice_offset + ri];
                let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                if crate::ops::line_matches_expr(s, expr) {
                    local_matches.push((raw_sel_cur.advance_to(ri), s.to_vec()));
                }
            }
            let slots = hdr.slot_counts[0];
            let ones = hdr.tpl_row_counts[0];
            let tcols0 = multi_cols.get(0);
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            for s in 0..slots {
                cols[s] = PreparedColumn::parse_and_validate(tcols0[s], ones, &mut scratch)?;
            }
            let unbound = UnboundTemplate::build(hdr.segs(0), &cols[..slots], &mut scratch);
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut tpl = unbound.bind(&mut scratch);
            let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
            let mut cur_t = 0usize;
            let mut tpl_sel_cur = SelectorBitCursor::new(hdr.selector_slice, hdr.row_count, true);
            for &t_idx in &tpl_hits[0] {
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
            let tcols = multi_cols.get(t);
            for s in 0..slots {
                cols[s] = PreparedColumn::parse_and_validate(tcols[s], cnt, &mut scratch)?;
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
                    if crate::ops::line_matches_expr(s, expr) {
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
                    hit_pos[t] += 1;
                    let tpl = bound_tpls[t].as_mut().unwrap();
                    let need = tpl.row_budget.max(4096) + 64;
                    if row_buf.len() < need {
                        row_buf.resize(need, 0);
                    }
                    unsafe {
                        tpl.skip_rows(ti - emitted_t[t]);
                        let sp = row_buf.as_mut_ptr();
                        let ep = tpl.emit_row(sp);
                        let written = ep.offset_from(sp) as usize;
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
        }
        *carry = next_carry;
        Ok(Some(new_rows))
    })
}

pub(crate) fn filter_grammar_block_expr_into_bytes(
    payload: &[u8],
    had_in_carry: bool,
    expr: &crate::ops::PredExpr<'_>,
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
        ensure_raw_rows_arena(&hdr, &mut scratch.raw_arena);
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

        let mut tpl_hits: Vec<Vec<usize>> = Vec::with_capacity(hdr.template_count);
        let mut total_tpl_matched = 0usize;
        let mut total_tpl_rows = 0usize;
        for t in 0..hdr.template_count {
            let cnt = hdr.tpl_row_counts[t];
            total_tpl_rows += cnt;
            if cnt == 0 {
                tpl_hits.push(Vec::new());
                continue;
            }
            let Some(mask) = eval_expr_on_template(
                hdr.segs(t),
                multi_cols.get(t),
                cnt,
                expr,
                hdr.raw_len,
                raw_mark,
                slice_mark,
                &mut scratch,
            )?
            else {
                return Ok(GrammarFilterBytesResult::Fallback);
            };
            let h = mask.into_hits(cnt, usize::MAX);
            total_tpl_matched += h.len();
            tpl_hits.push(h);
        }

        let raw_start = usize::from(had_in_carry && hdr.first_is_raw && hdr.raw_count > 0);
        let has_unterm_last = if hdr.last_is_raw && hdr.raw_count > 0 {
            let (p, l) = scratch.slice_arena[raw_slice_offset + hdr.raw_count - 1];
            !unsafe { std::slice::from_raw_parts(p, l) }.ends_with(b"\n")
        } else {
            false
        };
        let raw_end = hdr.raw_count.saturating_sub(usize::from(has_unterm_last));

        let mut matched_raw = 0usize;
        for ri in raw_start..raw_end {
            let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if crate::ops::line_matches_expr(s, expr) {
                matched_raw += 1;
            }
        }

        if total_tpl_matched == 0 && matched_raw == 0 {
            return Ok(GrammarFilterBytesResult::Empty);
        }
        if !had_in_carry
            && !has_unterm_last
            && total_tpl_matched == total_tpl_rows
            && matched_raw == hdr.raw_count
        {
            return Ok(GrammarFilterBytesResult::AllCompleteMatch);
        }

        if total_tpl_matched == 0 {
            for ri in raw_start..raw_end {
                let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
                let s = unsafe { std::slice::from_raw_parts(p, l) };
                if crate::ops::line_matches_expr(s, expr) {
                    dst.extend_from_slice(s);
                }
            }
            return Ok(GrammarFilterBytesResult::Bytes);
        }

        ensure_full_raw_arena(hdr.raw_len, &mut scratch.raw_arena);
        if hdr.template_count == 1 && hdr.raw_count == 0 {
            let slots = hdr.slot_counts[0];
            let ones = hdr.tpl_row_counts[0];
            let tcols0 = multi_cols.get(0);
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            for s in 0..slots {
                cols[s] = PreparedColumn::parse_and_validate(tcols0[s], ones, &mut scratch)?;
            }
            let unbound = UnboundTemplate::build(hdr.segs(0), &cols[..slots], &mut scratch);
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut tpl = unbound.bind(&mut scratch);
            let budget = tpl.row_budget.max(4096) + 64;
            dst.reserve(tpl_hits[0].len() * 64 + budget);
            let mut cur_t = 0usize;
            for &t_idx in &tpl_hits[0] {
                let pos = dst.len();
                dst.reserve(budget);
                unsafe {
                    tpl.skip_rows(t_idx - cur_t);
                    let sp = dst.as_mut_ptr().add(pos);
                    let ep = tpl.emit_row(sp);
                    let written = ep.offset_from(sp) as usize;
                    dst.set_len(pos + written);
                    cur_t = t_idx + 1;
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
        let mut unbounds: Vec<Option<UnboundTemplate>> = vec![None; hdr.template_count];
        for t in 0..hdr.template_count {
            if tpl_hits[t].is_empty() {
                continue;
            }
            let slots = hdr.slot_counts[t];
            let cnt = hdr.tpl_row_counts[t];
            let tcols = multi_cols.get(t);
            for s in 0..slots {
                cols[s] = PreparedColumn::parse_and_validate(tcols[s], cnt, &mut scratch)?;
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
        for r in 0..hdr.row_count {
            let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
            if sel == 0 {
                let ri = raw_cur;
                raw_cur += 1;
                if ri >= raw_start && ri < raw_end {
                    let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
                    let s = unsafe { std::slice::from_raw_parts(p, l) };
                    if crate::ops::line_matches_expr(s, expr) {
                        dst.extend_from_slice(s);
                    }
                }
            } else {
                let t = sel - 1;
                let ti = tpl_cur[t];
                tpl_cur[t] += 1;
                let hp = hit_pos[t];
                if hp < tpl_hits[t].len() && tpl_hits[t][hp] == ti {
                    hit_pos[t] += 1;
                    let tpl = bound_tpls[t].as_mut().unwrap();
                    let budget = tpl.row_budget.max(4096) + 64;
                    let pos = dst.len();
                    dst.reserve(budget);
                    unsafe {
                        tpl.skip_rows(ti - emitted_t[t]);
                        let sp = dst.as_mut_ptr().add(pos);
                        let ep = tpl.emit_row(sp);
                        let written = ep.offset_from(sp) as usize;
                        dst.set_len(pos + written);
                        emitted_t[t] = ti + 1;
                    }
                }
            }
        }
        Ok(GrammarFilterBytesResult::Bytes)
    })
}
