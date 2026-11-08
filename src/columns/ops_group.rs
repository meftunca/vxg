#[allow(clippy::too_many_arguments)]
fn group_by_template_resolved<'a>(
    g_col_bytes: &'a [u8],
    a_slot_bytes: Option<(&'a [u8], crate::ops::ColTarget<'_>)>,
    ones: usize,
    mask: &TplMask,
    group_col: crate::ops::ColTarget<'_>,
    out_map: &mut crate::ops::agg::GroupMap,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<()> {
    use std::borrow::Cow;

    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(());
    }
    if try_group_by_dict_packed_fast(g_col_bytes, a_slot_bytes, ones, mask, group_col, out_map)? {
        return Ok(());
    }
    if try_group_by_decimal_packed_fast(
        g_col_bytes,
        a_slot_bytes,
        ones,
        mask,
        group_col,
        out_map,
        hdr_raw_len,
        raw_mark,
        slice_mark,
        scratch,
    )? {
        return Ok(());
    }

    ensure_col_raw_arena(g_col_bytes, hdr_raw_len, &mut scratch.raw_arena);
    let g_col = PreparedColumn::parse_and_validate(g_col_bytes, ones, scratch)?;

    // 1. Constant group column: entire template belongs to at most 1 group!
    let const_g: Option<Cow<'a, [u8]>> = match g_col.const_full {
        ConstValue::Slice(s) => Some(Cow::Borrowed(s)),
        ConstValue::Inline { ref buf, len } => Some(Cow::Owned(buf[..len as usize].to_vec())),
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
            Some(Cow::Owned(tmp))
        }
        ConstValue::None => None,
    };
    if let Some(raw_gtok) = const_g {
        reset_scratch_to(scratch, raw_mark, slice_mark);
        let Some((is_str, gkey)) = crate::ops::decode_group_token_ref(group_col, &raw_gtok) else {
            return Ok(());
        };
        let mut st = crate::ops::AggStats::default();
        if let Some((a_bytes, acol)) = a_slot_bytes {
            aggregate_slot_with_mask(
                a_bytes,
                ones,
                mask,
                acol,
                &mut st,
                hdr_raw_len,
                raw_mark,
                slice_mark,
                scratch,
            )?;
        } else {
            st.add_rows_only(pop as u64);
        }
        out_map.merge_decoded(is_str, &gkey, &st);
        return Ok(());
    }

    // 2. Dictionary group column (`|D| <= 256`): zero-alloc borrow from payload + compact L1 buckets!
    if let ColumnStream::Dictionary {
        u8_offset: g_u8_off,
        ..
    } = g_col.stream
    {
        if let Some((slice_off, entries)) = g_col.dict_slices {
            if entries <= 256 {
                const NONE_KEY: Option<(bool, Cow<'static, [u8]>)> = None;
                let mut keys_inline: [Option<(bool, Cow<'a, [u8]>)>; 16] = [NONE_KEY; 16];
                let mut keys_overflow: Vec<Option<(bool, Cow<'a, [u8]>)>> = if entries > 16 {
                    Vec::with_capacity(entries)
                } else {
                    Vec::new()
                };
                let g_start = g_col_bytes.as_ptr() as usize;
                let g_end = g_start + g_col_bytes.len();
                for e in 0..entries {
                    let (ptr, len) = scratch.slice_arena[slice_off + e];
                    let p_addr = ptr as usize;
                    let decoded = if p_addr >= g_start && p_addr + len <= g_end {
                        let raw_tok: &'a [u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
                        crate::ops::decode_group_token_ref(group_col, raw_tok)
                    } else {
                        let raw_tok = unsafe { std::slice::from_raw_parts(ptr, len) };
                        crate::ops::decode_group_token_ref(group_col, raw_tok)
                            .map(|(is_s, k)| (is_s, Cow::Owned(k.into_owned())))
                    };
                    if entries <= 16 {
                        keys_inline[e] = decoded;
                    } else {
                        keys_overflow.push(decoded);
                    }
                }
                let dict_keys: &[Option<(bool, Cow<'a, [u8]>)>] = if entries <= 16 {
                    &keys_inline[..entries]
                } else {
                    &keys_overflow[..entries]
                };

                if let Some((a_bytes, acol)) = a_slot_bytes {
                    let a_view = build_slot_num_view(a_bytes, ones, acol, hdr_raw_len, scratch)?;
                    let mut inline_buckets = [crate::ops::AggStats::default(); 16];
                    let mut big_buckets: Vec<crate::ops::AggStats> = if entries > 16 {
                        vec![crate::ops::AggStats::default(); entries]
                    } else {
                        Vec::new()
                    };
                    let buckets: &mut [crate::ops::AggStats] = if entries <= 16 {
                        &mut inline_buckets[..entries]
                    } else {
                        &mut big_buckets[..entries]
                    };
                    match (&a_view, mask) {
                        (SlotNumView::U16 { u16_offset }, TplMask::All) => {
                            let ids = &scratch.u8_arena[g_u8_off..g_u8_off + ones];
                            let vals = &scratch.u16_arena[*u16_offset..*u16_offset + ones];
                            for (&id, &v) in ids.iter().zip(vals.iter()) {
                                buckets[id as usize].add_num(u64::from(v));
                            }
                        }
                        (SlotNumView::U64 { u64_offset }, TplMask::All) => {
                            let ids = &scratch.u8_arena[g_u8_off..g_u8_off + ones];
                            let vals = &scratch.u64_arena[*u64_offset..*u64_offset + ones];
                            for (&id, &v) in ids.iter().zip(vals.iter()) {
                                buckets[id as usize].add_num(v);
                            }
                        }
                        (_, TplMask::All) => {
                            for idx in 0..ones {
                                let id = scratch.u8_arena[g_u8_off + idx] as usize;
                                add_parsed_slot_num(&mut buckets[id], a_view.get(idx, scratch), 1);
                            }
                        }
                        (_, TplMask::Bits(bs, _)) => {
                            for (wi, &word0) in bs.words().iter().enumerate() {
                                let mut w = word0;
                                let base = wi << 6;
                                while w != 0 {
                                    let idx = base + w.trailing_zeros() as usize;
                                    let id = scratch.u8_arena[g_u8_off + idx] as usize;
                                    add_parsed_slot_num(&mut buckets[id], a_view.get(idx, scratch), 1);
                                    w &= w - 1;
                                }
                            }
                        }
                        (_, TplMask::Empty) => unreachable!(),
                    }
                    for (e, dk) in dict_keys.iter().enumerate() {
                        if buckets[e].row_count > 0 {
                            if let Some((is_str, ref k)) = *dk {
                                out_map.merge_decoded(is_str, k, &buckets[e]);
                            }
                        }
                    }
                } else {
                    let ids = &scratch.u8_arena[g_u8_off..g_u8_off + ones];
                    let mut freq_small = [0u32; 16];
                    let mut freq_big = [0u32; 256];
                    let freq: &mut [u32] = if entries <= 16 {
                        &mut freq_small[..entries]
                    } else {
                        &mut freq_big[..entries]
                    };
                    match mask {
                        TplMask::All => {
                            for &id in ids {
                                freq[id as usize] += 1;
                            }
                        }
                        TplMask::Bits(bs, _) => {
                            for (wi, &word0) in bs.words().iter().enumerate() {
                                let mut w = word0;
                                let base = wi << 6;
                                while w != 0 {
                                    let idx = base + w.trailing_zeros() as usize;
                                    freq[ids[idx] as usize] += 1;
                                    w &= w - 1;
                                }
                            }
                        }
                        TplMask::Empty => unreachable!(),
                    }
                    for (e, dk) in dict_keys.iter().enumerate() {
                        let c = u64::from(freq[e]);
                        if c > 0 {
                            if let Some((is_str, ref k)) = *dk {
                                let mut st = crate::ops::AggStats::default();
                                st.add_rows_only(c);
                                out_map.merge_decoded(is_str, k, &st);
                            }
                        }
                    }
                }
                reset_scratch_to(scratch, raw_mark, slice_mark);
                return Ok(());
            }
        }
    }

    // 3. Pure canonical `u64` group column: aggregate in binary `u64` buckets without per-row ASCII formatting!
    if matches!(g_col.const_lead, LeadValue::Empty) && g_col.const_trail.is_empty() {
        let g_num_kind = match g_col.stream {
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
        if let Some(g_view) = g_num_kind {
            let a_view = if let Some((a_bytes, acol)) = a_slot_bytes {
                Some(build_slot_num_view(a_bytes, ones, acol, hdr_raw_len, scratch)?)
            } else {
                None
            };
            match mask {
                TplMask::All => {
                    for idx in 0..ones {
                        let gk = g_view.get_u64(idx, scratch);
                        if let Some(ref av) = a_view {
                            match av.get(idx, scratch) {
                                ParsedSlotNum::U64(v) => out_map.add_u64_num(group_col, gk, v),
                                ParsedSlotNum::Dec(d) => out_map.add_u64_dec(group_col, gk, d),
                                ParsedSlotNum::None => out_map.add_u64_row_only(group_col, gk),
                            }
                        } else {
                            out_map.add_u64_row_only(group_col, gk);
                        }
                    }
                }
                TplMask::Bits(bs, _) => {
                    for (wi, &word0) in bs.words().iter().enumerate() {
                        let mut w = word0;
                        let base = wi << 6;
                        while w != 0 {
                            let idx = base + w.trailing_zeros() as usize;
                            let gk = g_view.get_u64(idx, scratch);
                            if let Some(ref av) = a_view {
                                match av.get(idx, scratch) {
                                    ParsedSlotNum::U64(v) => out_map.add_u64_num(group_col, gk, v),
                                    ParsedSlotNum::Dec(d) => out_map.add_u64_dec(group_col, gk, d),
                                    ParsedSlotNum::None => out_map.add_u64_row_only(group_col, gk),
                                }
                            } else {
                                out_map.add_u64_row_only(group_col, gk);
                            }
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => unreachable!(),
            }
            reset_scratch_to(scratch, raw_mark, slice_mark);
            return Ok(());
        }
    }

    // 4. Generic single-slot group column: emit only `g_slot` (and `a_slot` if present), skipping all other columns!
    let unbound = UnboundTemplate::build(&[b"", b""], &[g_col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let budget = tpl.row_budget.max(512) + 64;
    let mut g_flat: Vec<u8> = Vec::with_capacity(ones * 16 + budget);
    let mut g_spans: Vec<(usize, usize)> = Vec::with_capacity(ones);
    for _ in 0..ones {
        let s = g_flat.len();
        g_flat.reserve(budget);
        let written = unsafe {
            let sp = g_flat.as_mut_ptr().add(s);
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        unsafe {
            g_flat.set_len(s + written);
        }
        g_spans.push((s, s + written));
    }
    reset_scratch_to(scratch, raw_mark, slice_mark);

    let a_view = if let Some((a_bytes, acol)) = a_slot_bytes {
        Some(build_slot_num_view(a_bytes, ones, acol, hdr_raw_len, scratch)?)
    } else {
        None
    };

    let mut visit_idx = |idx: usize| {
        let (gs, ge) = g_spans[idx];
        let Some((is_str, gkey)) = crate::ops::decode_group_token_ref(group_col, &g_flat[gs..ge])
        else {
            return;
        };
        let mut st = crate::ops::AggStats::default();
        if let Some(ref av) = a_view {
            add_parsed_slot_num(&mut st, av.get(idx, scratch), 1);
        } else {
            st.add_rows_only(1);
        }
        out_map.merge_decoded(is_str, &gkey, &st);
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
    reset_scratch_to(scratch, raw_mark, slice_mark);
    Ok(())
}

pub(crate) fn group_by_grammar_block_seq(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    group_col: crate::ops::ColTarget<'_>,
    agg_col: Option<crate::ops::ColTarget<'_>>,
    out_map: &mut crate::ops::agg::GroupMap,
) -> Result<bool> {
    group_by_grammar_block_inner(payload, carry, true, true, where_expr, group_col, agg_col, out_map)
}

pub(crate) fn group_by_grammar_block_chunk(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_in_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    group_col: crate::ops::ColTarget<'_>,
    agg_col: Option<crate::ops::ColTarget<'_>>,
    out_map: &mut crate::ops::agg::GroupMap,
) -> Result<bool> {
    group_by_grammar_block_inner(
        payload,
        carry,
        eval_in_carry,
        true,
        where_expr,
        group_col,
        agg_col,
        out_map,
    )
}

#[allow(clippy::too_many_arguments)]
fn group_by_grammar_block_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    save_out_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    group_col: crate::ops::ColTarget<'_>,
    agg_col: Option<crate::ops::ColTarget<'_>>,
    out_map: &mut crate::ops::agg::GroupMap,
) -> Result<bool> {
    if matches!(
        group_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
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

        // Precheck all templates BEFORE mutating `out_map` so fallback leaves `out_map` untouched
        // while avoiding per-block `GroupMap` heap allocation!
        const EMPTY_PLAN: Option<(TplMask, usize, Option<usize>)> = None;
        let mut plans: [Option<(TplMask, usize, Option<usize>)>; 16] = [EMPTY_PLAN; 16];
        for (t, plan_slot) in plans.iter_mut().enumerate().take(hdr.template_count) {
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
                    return Ok(false);
                };
                m
            } else {
                TplMask::All
            };
            if mask.popcount(cnt) == 0 {
                continue;
            }
            let g_slot = match resolve_template_slot(segs, tcols, group_col)? {
                SlotLookup::Fallback => return Ok(false),
                SlotLookup::Absent => continue,
                SlotLookup::Slot(s) => s,
            };
            let a_slot = match agg_col {
                None => None,
                Some(acol) => match resolve_template_slot(segs, tcols, acol)? {
                    SlotLookup::Fallback => return Ok(false),
                    SlotLookup::Absent => None,
                    SlotLookup::Slot(s) => Some(s),
                },
            };
            *plan_slot = Some((mask, g_slot, a_slot));
        }

        for (t, plan_opt) in plans.iter().enumerate().take(hdr.template_count) {
            if let Some((ref mask, g_slot, a_slot_opt)) = *plan_opt {
                let tcols = multi_cols.get(t);
                let a_slot_bytes = a_slot_opt.map(|s| (tcols[s], agg_col.unwrap()));
                group_by_template_resolved(
                    tcols[g_slot],
                    a_slot_bytes,
                    hdr.tpl_row_counts[t],
                    mask,
                    group_col,
                    out_map,
                    hdr.raw_len,
                    raw_mark,
                    slice_mark,
                    &mut scratch,
                )?;
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
                    return Ok(true);
                }
            }
            if eval_carry {
                crate::ops::agg::group_by_line(&c_bytes, where_expr, group_col, agg_col, out_map);
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
            crate::ops::agg::group_by_line(s, where_expr, group_col, agg_col, out_map);
        }
        Ok(true)
    })
}
