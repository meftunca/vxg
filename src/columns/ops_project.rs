pub(crate) fn select_grammar_block_par_into(
    payload: &[u8],
    had_in_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    rem_limit: usize,
    dst: &mut Vec<u8>,
) -> Result<Option<usize>> {
    let mut dummy: Option<Vec<u8>> = had_in_carry.then(Vec::new);
    select_grammar_block_inner(payload, &mut dummy, false, where_expr, cols, rem_limit, dst)
}

pub(crate) fn select_grammar_block_seq_into(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    rem_limit: usize,
    dst: &mut Vec<u8>,
) -> Result<Option<usize>> {
    select_grammar_block_inner(payload, carry, true, where_expr, cols, rem_limit, dst)
}

fn select_grammar_block_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    rem_limit: usize,
    dst: &mut Vec<u8>,
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
        let mut plans: Vec<TemplateProjPlan> = Vec::with_capacity(hdr.template_count);
        let mut any_active_tpl = false;
        for (t, mask_slot) in masks.iter_mut().enumerate().take(hdr.template_count) {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
                plans.push(TemplateProjPlan::SkipAll);
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
                plans.push(TemplateProjPlan::SkipAll);
                continue;
            }
            let Some(plan) = plan_template_projection(segs, tcols, cols)? else {
                return Ok(None);
            };
            if !matches!(plan, TemplateProjPlan::SkipAll) {
                any_active_tpl = true;
            }
            *mask_slot = mask;
            plans.push(plan);
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
            if eval_carry
                && emitted < rem_limit
                && where_expr.is_none_or(|e| crate::ops::line_matches_expr(&c_bytes, e))
                && crate::ops::project_line(&c_bytes, cols, dst)
            {
                emitted += 1;
            }
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
            for ri in raw_start..raw_end {
                if emitted >= rem_limit {
                    break;
                }
                let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
                let s = unsafe { std::slice::from_raw_parts(p, l) };
                if where_expr.is_none_or(|e| crate::ops::line_matches_expr(s, e))
                    && crate::ops::project_line(s, cols, dst)
                {
                    emitted += 1;
                }
            }
            *carry = next_carry;
            return Ok(Some(emitted));
        }

        if hdr.template_count == 1 && raw_start == raw_end {
            let ones = hdr.tpl_row_counts[0];
            match &plans[0] {
                TemplateProjPlan::SkipAll => {}
                TemplateProjPlan::ConstJsonEmpty { crlf } => {
                    let lit: &[u8] = if *crlf { b"{}\r\n" } else { b"{}\n" };
                    let n = masks[0].popcount(ones).min(rem_limit - emitted);
                    dst.reserve(n * lit.len());
                    for _ in 0..n {
                        dst.extend_from_slice(lit);
                    }
                    emitted += n;
                }
                TemplateProjPlan::Slots {
                    seg_bytes,
                    seg_ranges,
                    slot_map,
                    ..
                } => {
                    let ub = prepare_projected_unbound(
                        seg_bytes,
                        seg_ranges,
                        slot_map,
                        multi_cols.get(0),
                        ones,
                        hdr.raw_len,
                        &mut scratch,
                    )?;
                    let n_exec = scratch.unbound_col_arena.len();
                    scratch.exec_col_arena.reserve(n_exec);
                    let mut tpl = ub.bind(&mut scratch);
                    let budget = tpl.row_budget.max(1024) + 64;
                    let cap_rows = masks[0].popcount(ones).min(rem_limit - emitted);
                    match &masks[0] {
                        TplMask::All => {
                            emit_tpl_rows_all(&mut tpl, cap_rows, dst);
                            emitted += cap_rows;
                        }
                        TplMask::Bits(bs, _) => {
                            dst.reserve(cap_rows * 32 + budget);
                            let mut cur_t = 0usize;
                            'words: for (wi, &word0) in bs.words().iter().enumerate() {
                                let mut w = word0;
                                let base = wi << 6;
                                while w != 0 {
                                    if emitted >= rem_limit {
                                        break 'words;
                                    }
                                    let t_idx = base + w.trailing_zeros() as usize;
                                    let pos = dst.len();
                                    dst.reserve(budget);
                                    unsafe {
                                        tpl.skip_rows(t_idx - cur_t);
                                        let sp = dst.as_mut_ptr().add(pos);
                                        let ep = tpl.emit_row(sp);
                                        dst.set_len(pos + ep.offset_from(sp) as usize);
                                    }
                                    cur_t = t_idx + 1;
                                    emitted += 1;
                                    w &= w - 1;
                                }
                            }
                        }
                        TplMask::Empty => {}
                    }
                }
            }
            *carry = next_carry;
            return Ok(Some(emitted));
        }

        let sel_u8_offset = unpack_and_validate_u8(
            hdr.selector_slice,
            hdr.row_count,
            hdr.selector_width,
            0,
            hdr.template_count + 1,
            &mut scratch.u8_arena,
        )?;
        let mut unbounds: Vec<Option<UnboundTemplate>> = Vec::with_capacity(hdr.template_count);
        for t in 0..hdr.template_count {
            if let TemplateProjPlan::Slots {
                ref seg_bytes,
                ref seg_ranges,
                ref slot_map,
                ..
            } = plans[t]
            {
                unbounds.push(Some(prepare_projected_unbound(
                    seg_bytes,
                    seg_ranges,
                    slot_map,
                    multi_cols.get(t),
                    hdr.tpl_row_counts[t],
                    hdr.raw_len,
                    &mut scratch,
                )?));
            } else {
                unbounds.push(None);
            }
        }
        let n_exec = scratch.unbound_col_arena.len();
        scratch.exec_col_arena.reserve(n_exec);
        let mut bound_tpls: Vec<Option<PreparedTemplate>> = unbounds
            .into_iter()
            .map(|u| u.map(|ub| ub.bind(&mut scratch)))
            .collect();
        let mut tpl_cur = [0usize; 16];
        let mut emitted_t = [0usize; 16];
        let mut raw_cur = 0usize;
        for r in 0..hdr.row_count {
            if emitted >= rem_limit {
                break;
            }
            let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
            if sel == 0 {
                let ri = raw_cur;
                raw_cur += 1;
                if ri >= raw_start && ri < raw_end {
                    let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
                    let s = unsafe { std::slice::from_raw_parts(p, l) };
                    if where_expr.is_none_or(|e| crate::ops::line_matches_expr(s, e))
                        && crate::ops::project_line(s, cols, dst)
                    {
                        emitted += 1;
                    }
                }
            } else {
                let t = sel - 1;
                let ti = tpl_cur[t];
                tpl_cur[t] += 1;
                let matched = match &masks[t] {
                    TplMask::Empty => false,
                    TplMask::All => true,
                    TplMask::Bits(bs, _) => (bs.words()[ti >> 6] & (1u64 << (ti & 63))) != 0,
                };
                if !matched {
                    continue;
                }
                match plans[t] {
                    TemplateProjPlan::SkipAll => {}
                    TemplateProjPlan::ConstJsonEmpty { crlf } => {
                        dst.extend_from_slice(if crlf { b"{}\r\n" } else { b"{}\n" });
                        emitted += 1;
                    }
                    TemplateProjPlan::Slots { .. } => {
                        let tpl = bound_tpls[t].as_mut().unwrap();
                        let budget = tpl.row_budget.max(1024) + 64;
                        let pos = dst.len();
                        dst.reserve(budget);
                        unsafe {
                            tpl.skip_rows(ti - emitted_t[t]);
                            let sp = dst.as_mut_ptr().add(pos);
                            let ep = tpl.emit_row(sp);
                            dst.set_len(pos + ep.offset_from(sp) as usize);
                        }
                        emitted_t[t] = ti + 1;
                        emitted += 1;
                    }
                }
            }
        }
        *carry = next_carry;
        Ok(Some(emitted))
    })
}

pub(crate) fn project_grammar_block(
    payload: &[u8],
    had_in_carry: bool,
    opt_tail_row: Option<&[u8]>,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    cols: &[crate::ops::ColTarget<'_>],
    need_checksum: bool,
    crc_buf: &mut Vec<u8>,
) -> Result<GrammarProjectOut> {
    if let Some(mut hdr) = parse_grammar_for_ops(payload)? {
        if hdr.row_count == 0 {
            return Ok(GrammarProjectOut::Empty);
        }
        if had_in_carry && !hdr.first_is_raw {
            return Ok(GrammarProjectOut::Fallback);
        }
        let ones = hdr.tpl_row_counts[0];
        if hdr.template_count == 1 && ones > 0 && opt_tail_row.is_none_or(|t| !t.is_empty() && t.len() <= 65536) {
            let splice_attempt: Result<Option<GrammarProjectOut>> = DECODE_SCRATCH.with(|cell| {
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
                let raw_start = usize::from(had_in_carry && hdr.first_is_raw && hdr.raw_count > 0);
                let mut raw_end = hdr.raw_count;
                if hdr.last_is_raw && raw_end > raw_start {
                    let (p, l) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
                    let s = unsafe { std::slice::from_raw_parts(p, l) };
                    if !s.ends_with(b"\n") {
                        raw_end -= 1;
                    }
                }
                if raw_start != raw_end {
                    return Ok(None);
                }
                let raw_mark = scratch.raw_arena.len();
                let slice_mark = scratch.slice_arena.len();
                let multi_cols = read_all_template_col_slices(&mut hdr)?;
                let segs = hdr.segs(0);
                let tcols = multi_cols.get(0);
                let mask = if let Some(expr) = where_expr {
                    let Some(m) = eval_expr_on_template(
                        segs, tcols, ones, expr, hdr.raw_len, raw_mark, slice_mark, &mut scratch,
                    )?
                    else {
                        return Ok(Some(GrammarProjectOut::Fallback));
                    };
                    m
                } else {
                    TplMask::All
                };
                if matches!(mask, TplMask::Empty) {
                    return Ok(Some(GrammarProjectOut::Empty));
                }
                let Some(plan) = plan_template_projection(segs, tcols, cols)? else {
                    return Ok(Some(GrammarProjectOut::Fallback));
                };
                if matches!(plan, TemplateProjPlan::SkipAll) {
                    return Ok(Some(GrammarProjectOut::Empty));
                }
                if matches!(mask, TplMask::All) {
                    if let TemplateProjPlan::Slots {
                        ref seg_bytes,
                        ref seg_ranges,
                        ref slot_map,
                        can_splice: true,
                    } = plan
                    {
                        crc_buf.clear();
                        let ub = prepare_projected_unbound(
                            seg_bytes,
                            seg_ranges,
                            slot_map,
                            tcols,
                            ones,
                            hdr.raw_len,
                            &mut scratch,
                        )?;
                        let n_exec = scratch.unbound_col_arena.len();
                        scratch.exec_col_arena.reserve(n_exec);
                        let mut tpl = ub.bind(&mut scratch);
                        emit_tpl_rows_all(&mut tpl, ones, crc_buf);
                        let tpl_raw_len = crc_buf.len();
                        if let Some(tail) = opt_tail_row {
                            crc_buf.extend_from_slice(tail);
                        }
                        let new_raw_len = crc_buf.len();
                        if !(512..=MAX_BLOCK_SIZE).contains(&new_raw_len) {
                            return Ok(Some(if tpl_raw_len == 0 {
                                GrammarProjectOut::Empty
                            } else {
                                GrammarProjectOut::Bytes(crc_buf[..tpl_raw_len].to_vec())
                            }));
                        }
                        let out = build_spliced_grammar_packet(
                            new_raw_len,
                            ones,
                            opt_tail_row,
                            seg_bytes,
                            seg_ranges,
                            slot_map,
                            tcols,
                        );
                        if out.len() >= new_raw_len {
                            return Ok(Some(if tpl_raw_len == 0 {
                                GrammarProjectOut::Empty
                            } else {
                                GrammarProjectOut::Bytes(crc_buf[..tpl_raw_len].to_vec())
                            }));
                        }
                        let crc = if need_checksum {
                            crate::checksum::crc32c(crc_buf)
                        } else {
                            0
                        };
                        return Ok(Some(GrammarProjectOut::Spliced {
                            payload: out,
                            new_raw_len,
                            crc,
                        }));
                    }
                }
                Ok(None)
            });
            if let Some(res) = splice_attempt? {
                return Ok(res);
            }
        }
    }
    let mut out_bytes = Vec::new();
    match select_grammar_block_par_into(
        payload,
        had_in_carry,
        where_expr,
        cols,
        usize::MAX,
        &mut out_bytes,
    )? {
        None => Ok(GrammarProjectOut::Fallback),
        Some(_) if out_bytes.is_empty() => Ok(GrammarProjectOut::Empty),
        Some(_) => Ok(GrammarProjectOut::Bytes(out_bytes)),
    }
}
