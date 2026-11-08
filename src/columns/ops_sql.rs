pub(crate) fn aggregate_multi_grammar_block_seq(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    agg_cols: &[crate::ops::ColTarget<'_>],
) -> Result<Option<crate::ops::multi_agg::MultiAggAcc>> {
    aggregate_multi_grammar_block_inner(payload, carry, true, true, where_expr, agg_cols)
}

pub(crate) fn aggregate_multi_grammar_block_chunk(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_in_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    agg_cols: &[crate::ops::ColTarget<'_>],
) -> Result<Option<crate::ops::multi_agg::MultiAggAcc>> {
    aggregate_multi_grammar_block_inner(payload, carry, eval_in_carry, true, where_expr, agg_cols)
}

fn aggregate_multi_grammar_block_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    save_out_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    agg_cols: &[crate::ops::ColTarget<'_>],
) -> Result<Option<crate::ops::multi_agg::MultiAggAcc>> {
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(None);
    };
    let ncols = agg_cols.len();
    if hdr.row_count == 0 {
        return Ok(Some(crate::ops::multi_agg::MultiAggAcc::with_cols(ncols)));
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

        let mut acc = crate::ops::multi_agg::MultiAggAcc::with_cols(ncols);
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
            let pop = mask.popcount(cnt);
            if pop == 0 {
                continue;
            }
            let mut slot_plan = Vec::with_capacity(ncols);
            for &acol in agg_cols {
                match resolve_template_slot(segs, tcols, acol)? {
                    SlotLookup::Fallback => return Ok(None),
                    SlotLookup::Absent => slot_plan.push(None),
                    SlotLookup::Slot(s) => slot_plan.push(Some(s)),
                }
            }
            acc.row_count += pop as u64;
            for (i, (&acol, &s_opt)) in agg_cols.iter().zip(slot_plan.iter()).enumerate() {
                match s_opt {
                    None => acc.cols[i].add_rows_only(pop as u64),
                    Some(s) => aggregate_slot_with_mask(
                        tcols[s],
                        cnt,
                        &mask,
                        acol,
                        &mut acc.cols[i],
                        hdr.raw_len,
                        raw_mark,
                        slice_mark,
                        &mut scratch,
                    )?,
                }
            }
        }

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
                    return Ok(Some(acc));
                }
            }
            if eval_carry {
                crate::ops::multi_agg::agg_multi_line(&c_bytes, where_expr, agg_cols, &mut acc);
            }
        }
        let mut raw_end = hdr.raw_count;
        if hdr.last_is_raw && raw_end > raw_start {
            let (p, l) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                if save_out_carry {
                    *carry = Some(s.to_vec());
                }
                raw_end -= 1;
            }
        }
        for ri in raw_start..raw_end {
            let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            crate::ops::multi_agg::agg_multi_line(s, where_expr, agg_cols, &mut acc);
        }
        Ok(Some(acc))
    })
}

enum SqlGroupSlotView<'a> {
    Const(Option<crate::ops::GroupKey>),
    Dict {
        u8_offset: usize,
        keys: Vec<Option<crate::ops::GroupKey>>,
    },
    U64 {
        view: SlotNumView,
        is_str: bool,
    },
    Generic {
        flat: Vec<u8>,
        spans: Vec<(usize, usize)>,
        gcol: crate::ops::ColTarget<'a>,
    },
}

#[inline]
fn fmt_u64_group_key(mut v: u64, is_str: bool) -> crate::ops::GroupKey {
    let mut buf = [0u8; 24];
    let mut i = buf.len();
    loop {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    let bytes = buf[i..].to_vec();
    if is_str {
        crate::ops::GroupKey::Str(bytes)
    } else {
        crate::ops::GroupKey::RawJson(bytes)
    }
}

impl<'a> SqlGroupSlotView<'a> {
    #[inline]
    fn get(&self, idx: usize, scratch: &DecodeScratch) -> Option<crate::ops::GroupKey> {
        match *self {
            Self::Const(ref k) => k.clone(),
            Self::Dict {
                u8_offset,
                ref keys,
            } => keys[scratch.u8_arena[u8_offset + idx] as usize].clone(),
            Self::U64 { ref view, is_str } => {
                Some(fmt_u64_group_key(view.get_u64(idx, scratch), is_str))
            }
            Self::Generic {
                ref flat,
                ref spans,
                gcol,
            } => {
                let (s, e) = spans[idx];
                crate::ops::decode_group_token_ref(gcol, &flat[s..e]).map(|(is_s, k)| {
                    if is_s {
                        crate::ops::GroupKey::Str(k.into_owned())
                    } else {
                        crate::ops::GroupKey::RawJson(k.into_owned())
                    }
                })
            }
        }
    }
}

fn build_sql_group_slot_view<'a>(
    g_col_bytes: &[u8],
    ones: usize,
    gcol: crate::ops::ColTarget<'a>,
    hdr_raw_len: usize,
    scratch: &mut DecodeScratch,
) -> Result<SqlGroupSlotView<'a>> {
    ensure_col_raw_arena(g_col_bytes, hdr_raw_len, &mut scratch.raw_arena);
    let col = PreparedColumn::parse_and_validate(g_col_bytes, ones, scratch)?;
    let const_opt: Option<Vec<u8>> = match col.const_full {
        ConstValue::Slice(s) => Some(s.to_vec()),
        ConstValue::Inline { ref buf, len } => Some(buf[..len as usize].to_vec()),
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
            Some(tmp)
        }
        ConstValue::None => None,
    };
    if let Some(raw_tok) = const_opt {
        let gk = crate::ops::decode_group_token_ref(gcol, &raw_tok).map(|(is_s, k)| {
            if is_s {
                crate::ops::GroupKey::Str(k.into_owned())
            } else {
                crate::ops::GroupKey::RawJson(k.into_owned())
            }
        });
        return Ok(SqlGroupSlotView::Const(gk));
    }

    if let ColumnStream::Dictionary { u8_offset, .. } = col.stream {
        if let Some((slice_off, entries)) = col.dict_slices {
            if entries <= 256 {
                let mut keys = Vec::with_capacity(entries);
                for e in 0..entries {
                    let (ptr, len) = scratch.slice_arena[slice_off + e];
                    let raw_tok = unsafe { std::slice::from_raw_parts(ptr, len) };
                    let gk = crate::ops::decode_group_token_ref(gcol, raw_tok).map(|(is_s, k)| {
                        if is_s {
                            crate::ops::GroupKey::Str(k.into_owned())
                        } else {
                            crate::ops::GroupKey::RawJson(k.into_owned())
                        }
                    });
                    keys.push(gk);
                }
                return Ok(SqlGroupSlotView::Dict { u8_offset, keys });
            }
        }
    }

    if matches!(col.const_lead, LeadValue::Empty) && col.const_trail.is_empty() {
        let num_v = match col.stream {
            ColumnStream::DecimalSmall { u16_offset } => Some(SlotNumView::U16 { u16_offset }),
            ColumnStream::DecimalValues { u64_offset } => Some(SlotNumView::U64 { u64_offset }),
            ColumnStream::DecimalAffineSmall { current, step }
            | ColumnStream::DecimalAffine { current, step } => Some(SlotNumView::Affine {
                first: current,
                step,
            }),
            ColumnStream::DecimalPacked { min, mut reader } => {
                let u64_offset = scratch.u64_arena.len();
                scratch.u64_arena.reserve(ones);
                for _ in 0..ones {
                    scratch.u64_arena.push(min.wrapping_add(reader.next_u64()));
                }
                Some(SlotNumView::U64 { u64_offset })
            }
            _ => None,
        };
        if let Some(view) = num_v {
            let is_str = matches!(gcol, crate::ops::ColTarget::Delimited { .. });
            return Ok(SqlGroupSlotView::U64 { view, is_str });
        }
    }

    let unbound_mark = scratch.unbound_col_arena.len();
    let exec_mark = scratch.exec_col_arena.len();
    let unbound = UnboundTemplate::build(&[b"", b""], &[col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let budget = tpl.row_budget.max(512) + 64;
    let mut flat: Vec<u8> = Vec::with_capacity(ones * 16 + budget);
    let mut spans = Vec::with_capacity(ones);
    for _ in 0..ones {
        let s = flat.len();
        flat.reserve(budget);
        let written = unsafe {
            let sp = flat.as_mut_ptr().add(s);
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        unsafe {
            flat.set_len(s + written);
        }
        spans.push((s, s + written));
    }
    scratch.exec_col_arena.truncate(exec_mark);
    scratch.unbound_col_arena.truncate(unbound_mark);
    Ok(SqlGroupSlotView::Generic { flat, spans, gcol })
}

pub(crate) fn multi_group_by_grammar_block_seq(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    group_cols: &[crate::ops::ColTarget<'_>],
    agg_cols: &[crate::ops::ColTarget<'_>],
    out_map: &mut crate::ops::multi_agg::MultiGroupMap,
) -> Result<bool> {
    multi_group_by_grammar_block_inner(
        payload, carry, true, true, where_expr, group_cols, agg_cols, out_map,
    )
}

pub(crate) fn multi_group_by_grammar_block_chunk(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_in_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    group_cols: &[crate::ops::ColTarget<'_>],
    agg_cols: &[crate::ops::ColTarget<'_>],
    out_map: &mut crate::ops::multi_agg::MultiGroupMap,
) -> Result<bool> {
    multi_group_by_grammar_block_inner(
        payload,
        carry,
        eval_in_carry,
        true,
        where_expr,
        group_cols,
        agg_cols,
        out_map,
    )
}

#[allow(clippy::too_many_arguments)]
fn multi_group_by_grammar_block_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    save_out_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    group_cols: &[crate::ops::ColTarget<'_>],
    agg_cols: &[crate::ops::ColTarget<'_>],
    out_map: &mut crate::ops::multi_agg::MultiGroupMap,
) -> Result<bool> {
    if group_cols.iter().any(|&gc| {
        matches!(
            gc,
            crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
        )
    }) {
        return Ok(false);
    }
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(false);
    };
    if hdr.row_count == 0 {
        return Ok(true);
    }
    if carry.is_some() && !hdr.first_is_raw {
        return Ok(false);
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

        type TplPlan = (TplMask, Vec<usize>, Vec<Option<usize>>);
        let mut plans: Vec<Option<TplPlan>> = Vec::with_capacity(hdr.template_count);
        for t in 0..hdr.template_count {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
                plans.push(None);
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
                    return Ok(false);
                };
                m
            } else {
                TplMask::All
            };
            if mask.popcount(cnt) == 0 {
                plans.push(None);
                continue;
            }
            let mut g_slots = Vec::with_capacity(group_cols.len());
            let mut g_absent = false;
            for &gc in group_cols {
                match resolve_template_slot(segs, tcols, gc)? {
                    SlotLookup::Fallback => return Ok(false),
                    SlotLookup::Absent => {
                        g_absent = true;
                        break;
                    }
                    SlotLookup::Slot(s) => g_slots.push(s),
                }
            }
            if g_absent {
                plans.push(None);
                continue;
            }
            let mut a_slots = Vec::with_capacity(agg_cols.len());
            for &ac in agg_cols {
                match resolve_template_slot(segs, tcols, ac)? {
                    SlotLookup::Fallback => return Ok(false),
                    SlotLookup::Absent => a_slots.push(None),
                    SlotLookup::Slot(s) => a_slots.push(Some(s)),
                }
            }
            plans.push(Some((mask, g_slots, a_slots)));
        }

        for (t, plan_opt) in plans.into_iter().enumerate() {
            let Some((mask, g_slots, a_slots)) = plan_opt else {
                continue;
            };
            let ones = hdr.tpl_row_counts[t];
            let tcols = multi_cols.get(t);
            reset_scratch_to(&mut scratch, raw_mark, slice_mark);

            let mut g_views = Vec::with_capacity(g_slots.len());
            for (&gs, &gc) in g_slots.iter().zip(group_cols.iter()) {
                g_views.push(build_sql_group_slot_view(
                    tcols[gs],
                    ones,
                    gc,
                    hdr.raw_len,
                    &mut scratch,
                )?);
            }
            let mut a_views = Vec::with_capacity(a_slots.len());
            for (&as_opt, &ac) in a_slots.iter().zip(agg_cols.iter()) {
                match as_opt {
                    None => a_views.push(SlotNumView::Const(ParsedSlotNum::None)),
                    Some(s) => a_views.push(build_slot_num_view(
                        tcols[s],
                        ones,
                        ac,
                        hdr.raw_len,
                        &mut scratch,
                    )?),
                }
            }

            let mut visit_idx = |idx: usize| {
                let mut keys = Vec::with_capacity(g_views.len());
                for gv in &g_views {
                    let Some(k) = gv.get(idx, &scratch) else {
                        return;
                    };
                    keys.push(k);
                }
                let mut aggs = vec![crate::ops::AggStats::default(); a_views.len()];
                for (st, av) in aggs.iter_mut().zip(a_views.iter()) {
                    add_parsed_slot_num(st, av.get(idx, &scratch), 1);
                }
                out_map.add_entry(keys, 1, &aggs);
            };

            match mask {
                TplMask::All => {
                    for idx in 0..ones {
                        visit_idx(idx);
                    }
                }
                TplMask::Bits(bs, _) => {
                    for (wi, &word0) in bs.words().iter().enumerate() {
                        let mut w = word0;
                        let base = wi << 6;
                        while w != 0 {
                            visit_idx(base + w.trailing_zeros() as usize);
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => unreachable!(),
            }
            reset_scratch_to(&mut scratch, raw_mark, slice_mark);
        }

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
                    return Ok(true);
                }
            }
            if eval_carry {
                crate::ops::multi_agg::multi_group_by_line(
                    &c_bytes, where_expr, group_cols, agg_cols, out_map,
                );
            }
        }
        let mut raw_end = hdr.raw_count;
        if hdr.last_is_raw && raw_end > raw_start {
            let (p, l) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                if save_out_carry {
                    *carry = Some(s.to_vec());
                }
                raw_end -= 1;
            }
        }
        for ri in raw_start..raw_end {
            let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            crate::ops::multi_agg::multi_group_by_line(
                s, where_expr, group_cols, agg_cols, out_map,
            );
        }
        Ok(true)
    })
}
