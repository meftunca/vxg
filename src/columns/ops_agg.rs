pub(crate) fn count_grammar_block_expr(
    payload: &[u8],
    had_in_carry: bool,
    expr: &crate::ops::PredExpr<'_>,
) -> Result<Option<u64>> {
    let mut dummy: Option<Vec<u8>> = had_in_carry.then(Vec::new);
    count_grammar_block_expr_inner(payload, &mut dummy, false, expr)
}

pub(crate) fn count_grammar_block_expr_seq(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    expr: &crate::ops::PredExpr<'_>,
) -> Result<Option<u64>> {
    count_grammar_block_expr_inner(payload, carry, true, expr)
}

fn count_grammar_block_expr_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    expr: &crate::ops::PredExpr<'_>,
) -> Result<Option<u64>> {
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

        let mut total = 0u64;
        for t in 0..hdr.template_count {
            let cnt = hdr.tpl_row_counts[t];
            if cnt == 0 {
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
            total += mask.popcount(cnt) as u64;
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
                    return Ok(Some(0));
                }
            }
            if eval_carry && crate::ops::line_matches_expr(&c_bytes, expr) {
                total += 1;
            }
        }
        let mut raw_end = hdr.raw_count;
        if hdr.last_is_raw && raw_end > raw_start {
            let (p, l) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                if eval_carry {
                    *carry = Some(s.to_vec());
                }
                raw_end -= 1;
            }
        }
        for ri in raw_start..raw_end {
            let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if crate::ops::line_matches_expr(s, expr) {
                total += 1;
            }
        }
        Ok(Some(total))
    })
}

enum SlotNumView {
    Const(Option<u64>),
    U16 { u16_offset: usize },
    U64 { u64_offset: usize },
    Affine { first: u64, step: u64 },
    Dict {
        u8_offset: usize,
        lut: Box<[Option<u64>; 256]>,
        entries: usize,
    },
    PerRow(Vec<Option<u64>>),
}

impl SlotNumView {
    #[inline(always)]
    fn get(&self, idx: usize, scratch: &DecodeScratch) -> Option<u64> {
        match *self {
            Self::Const(v) => v,
            Self::U16 { u16_offset } => Some(u64::from(scratch.u16_arena[u16_offset + idx])),
            Self::U64 { u64_offset } => Some(scratch.u64_arena[u64_offset + idx]),
            Self::Affine { first, step } => {
                Some(first.wrapping_add(step.wrapping_mul(idx as u64)))
            }
            Self::Dict {
                u8_offset, ref lut, ..
            } => lut[scratch.u8_arena[u8_offset + idx] as usize],
            Self::PerRow(ref v) => v[idx],
        }
    }
}

fn build_slot_num_view(
    col_bytes: &[u8],
    ones: usize,
    target_col: crate::ops::ColTarget<'_>,
    hdr_raw_len: usize,
    scratch: &mut DecodeScratch,
) -> Result<SlotNumView> {
    let digit_delim = matches!(
        target_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    );
    ensure_col_raw_arena(col_bytes, hdr_raw_len, &mut scratch.raw_arena);
    let col = PreparedColumn::parse_and_validate(col_bytes, ones, scratch)?;
    match col.const_full {
        ConstValue::Slice(s) => {
            let v = if digit_delim {
                None
            } else {
                crate::ops::parse_canonical_u64(s)
            };
            return Ok(SlotNumView::Const(v));
        }
        ConstValue::Inline { ref buf, len } => {
            let v = if digit_delim {
                None
            } else {
                crate::ops::parse_canonical_u64(&buf[..len as usize])
            };
            return Ok(SlotNumView::Const(v));
        }
        ConstValue::QuotedDecimal { .. } => {
            return Ok(SlotNumView::Const(None));
        }
        ConstValue::None => {}
    }

    if !digit_delim && matches!(col.const_lead, LeadValue::Empty) && col.const_trail.is_empty() {
        match col.stream {
            ColumnStream::DecimalSmall { u16_offset } => {
                return Ok(SlotNumView::U16 { u16_offset });
            }
            ColumnStream::DecimalValues { u64_offset } => {
                return Ok(SlotNumView::U64 { u64_offset });
            }
            ColumnStream::DecimalAffineSmall { current, step }
            | ColumnStream::DecimalAffine { current, step } => {
                return Ok(SlotNumView::Affine {
                    first: current,
                    step,
                });
            }
            ColumnStream::DecimalPacked { min, mut reader } => {
                let u64_offset = scratch.u64_arena.len();
                scratch.u64_arena.reserve(ones);
                for _ in 0..ones {
                    scratch.u64_arena.push(min.wrapping_add(reader.next_u64()));
                }
                return Ok(SlotNumView::U64 { u64_offset });
            }
            _ => {}
        }
    }

    if !digit_delim {
        if let ColumnStream::Dictionary { u8_offset, .. } = col.stream {
            if let Some((slice_off, entries)) = col.dict_slices {
                if entries <= 256 {
                    let mut lut = Box::new([None; 256]);
                    for e in 0..entries {
                        let (ptr, len) = scratch.slice_arena[slice_off + e];
                        let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                        lut[e] = crate::ops::parse_canonical_u64(s);
                    }
                    return Ok(SlotNumView::Dict {
                        u8_offset,
                        lut,
                        entries,
                    });
                }
            }
        }
        // Quoted strings / timestamps / hex with non-digit lead or trail can never be canonical u64.
        let lead_non_digit = match col.const_lead {
            LeadValue::Empty => false,
            LeadValue::QuoteOnly | LeadValue::QuotePrefix(_) => true,
            LeadValue::Slice(s) => !s.iter().all(u8::is_ascii_digit),
            LeadValue::Inline { ref buf, len } => !buf[..len as usize].iter().all(u8::is_ascii_digit),
        };
        if lead_non_digit || !col.const_trail.iter().all(u8::is_ascii_digit) {
            return Ok(SlotNumView::Const(None));
        }
    }

    let unbound = UnboundTemplate::build(&[b"", b""], &[col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let mut val_buf = vec![0u8; tpl.row_budget.max(512) + 64];
    let mut vals = Vec::with_capacity(ones);
    for _ in 0..ones {
        let written = unsafe {
            let sp = val_buf.as_mut_ptr();
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        let v = if !digit_delim {
            crate::ops::parse_canonical_u64(&val_buf[..written])
        } else {
            None
        };
        vals.push(v);
    }
    Ok(SlotNumView::PerRow(vals))
}

#[allow(clippy::too_many_arguments)]
fn aggregate_slot_with_mask(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    target_col: crate::ops::ColTarget<'_>,
    stats: &mut crate::ops::AggStats,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<()> {
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(());
    }
    let view = build_slot_num_view(col_bytes, ones, target_col, hdr_raw_len, scratch)?;
    match (&view, mask) {
        (SlotNumView::Const(Some(v)), _) => {
            stats.add_const_num(*v, pop as u64);
        }
        (SlotNumView::Const(None), _) => {
            stats.add_rows_only(pop as u64);
        }
        (SlotNumView::Affine { first, step }, TplMask::All) => {
            stats.add_affine_all(*first, *step, ones);
        }
        (SlotNumView::U16 { u16_offset }, TplMask::All) => {
            let slice = &scratch.u16_arena[*u16_offset..*u16_offset + ones];
            let mut sum = 0u64;
            let mut lo = u16::MAX;
            let mut hi = 0u16;
            for &v in slice {
                sum += u64::from(v);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            stats.row_count += ones as u64;
            stats.num_count += ones as u64;
            stats.sum += u128::from(sum);
            let lo64 = u64::from(lo);
            let hi64 = u64::from(hi);
            stats.min = Some(stats.min.map_or(lo64, |m| m.min(lo64)));
            stats.max = Some(stats.max.map_or(hi64, |m| m.max(hi64)));
        }
        (SlotNumView::U64 { u64_offset }, TplMask::All) => {
            let slice = &scratch.u64_arena[*u64_offset..*u64_offset + ones];
            let mut sum = 0u128;
            let mut lo = u64::MAX;
            let mut hi = 0u64;
            for &v in slice {
                sum += u128::from(v);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            stats.row_count += ones as u64;
            stats.num_count += ones as u64;
            stats.sum += sum;
            stats.min = Some(stats.min.map_or(lo, |m| m.min(lo)));
            stats.max = Some(stats.max.map_or(hi, |m| m.max(hi)));
        }
        (
            SlotNumView::Dict {
                u8_offset,
                lut,
                entries,
            },
            _,
        ) => {
            let ids = &scratch.u8_arena[*u8_offset..*u8_offset + ones];
            let mut freq = [0u32; 256];
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
            for e in 0..*entries {
                let c = u64::from(freq[e]);
                if c == 0 {
                    continue;
                }
                if let Some(v) = lut[e] {
                    stats.add_const_num(v, c);
                } else {
                    stats.add_rows_only(c);
                }
            }
        }
        (_, TplMask::All) => {
            for idx in 0..ones {
                if let Some(v) = view.get(idx, scratch) {
                    stats.add_num(v);
                } else {
                    stats.add_rows_only(1);
                }
            }
        }
        (_, TplMask::Bits(bs, _)) => {
            for (wi, &word0) in bs.words().iter().enumerate() {
                let mut w = word0;
                let base = wi << 6;
                while w != 0 {
                    let idx = base + w.trailing_zeros() as usize;
                    if let Some(v) = view.get(idx, scratch) {
                        stats.add_num(v);
                    } else {
                        stats.add_rows_only(1);
                    }
                    w &= w - 1;
                }
            }
        }
        (_, TplMask::Empty) => unreachable!(),
    }
    reset_scratch_to(scratch, raw_mark, slice_mark);
    Ok(())
}

pub(crate) fn aggregate_grammar_block(
    payload: &[u8],
    had_in_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    agg_col: crate::ops::ColTarget<'_>,
) -> Result<Option<crate::ops::AggStats>> {
    let mut dummy: Option<Vec<u8>> = had_in_carry.then(Vec::new);
    aggregate_grammar_block_inner(payload, &mut dummy, false, where_expr, agg_col)
}

pub(crate) fn aggregate_grammar_block_seq(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    agg_col: crate::ops::ColTarget<'_>,
) -> Result<Option<crate::ops::AggStats>> {
    aggregate_grammar_block_inner(payload, carry, true, where_expr, agg_col)
}

fn aggregate_grammar_block_inner(
    payload: &[u8],
    carry: &mut Option<Vec<u8>>,
    eval_carry: bool,
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    agg_col: crate::ops::ColTarget<'_>,
) -> Result<Option<crate::ops::AggStats>> {
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(None);
    };
    if hdr.row_count == 0 {
        return Ok(Some(crate::ops::AggStats::default()));
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

        let mut stats = crate::ops::AggStats::default();
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
            match resolve_template_slot(segs, tcols, agg_col)? {
                SlotLookup::Fallback => return Ok(None),
                SlotLookup::Absent => stats.add_rows_only(pop as u64),
                SlotLookup::Slot(s) => aggregate_slot_with_mask(
                    tcols[s],
                    cnt,
                    &mask,
                    agg_col,
                    &mut stats,
                    hdr.raw_len,
                    raw_mark,
                    slice_mark,
                    &mut scratch,
                )?,
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
                    return Ok(Some(stats));
                }
            }
            if eval_carry {
                crate::ops::agg_line(&c_bytes, where_expr, agg_col, &mut stats);
            }
        }
        let mut raw_end = hdr.raw_count;
        if hdr.last_is_raw && raw_end > raw_start {
            let (p, l) = scratch.slice_arena[raw_slice_offset + raw_end - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            if !s.ends_with(b"\n") {
                if eval_carry {
                    *carry = Some(s.to_vec());
                }
                raw_end -= 1;
            }
        }
        for ri in raw_start..raw_end {
            let (p, l) = scratch.slice_arena[raw_slice_offset + ri];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            crate::ops::agg_line(s, where_expr, agg_col, &mut stats);
        }
        Ok(Some(stats))
    })
}
