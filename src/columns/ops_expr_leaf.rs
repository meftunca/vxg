#[inline(always)]
fn read_packed_bits_at(packed: &[u8], bit_pos: usize, width: u8) -> u64 {
    if width == 0 {
        return 0;
    }
    let byte_idx = bit_pos >> 3;
    let bit_off = (bit_pos & 7) as u32;
    let w = width as u32;
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    if byte_idx + 8 <= packed.len() && bit_off + w <= 64 {
        let raw = unsafe {
            packed
                .as_ptr()
                .add(byte_idx)
                .cast::<u64>()
                .read_unaligned()
                .to_le()
        };
        (raw >> bit_off) & mask
    } else {
        let mut raw = 0u128;
        let rem = packed.len().saturating_sub(byte_idx).min(16);
        for i in 0..rem {
            raw |= u128::from(unsafe { *packed.get_unchecked(byte_idx + i) }) << (8 * i);
        }
        ((raw >> bit_off) as u64) & mask
    }
}

#[inline(always)]
fn num_cmp_always_true(cmp: crate::query::NumCmp, lo: u64, hi: u64) -> bool {
    match cmp {
        crate::query::NumCmp::Eq(x) => lo == x && hi == x,
        crate::query::NumCmp::Ne(x) => x < lo || x > hi,
        crate::query::NumCmp::Gt(x) => lo > x,
        crate::query::NumCmp::Ge(x) => lo >= x,
        crate::query::NumCmp::Lt(x) => hi < x,
        crate::query::NumCmp::Le(x) => hi <= x,
        crate::query::NumCmp::Range(a, b) => a <= b && lo >= a && hi <= b,
    }
}

#[inline(always)]
fn numeric_col_zone_truth(
    col_bytes: &[u8],
    ones: usize,
    cmp: crate::query::NumCmp,
) -> Option<TemplateTruth> {
    if ones == 0 {
        return Some(TemplateTruth::AlwaysFalse);
    }
    let (lo, hi) = decimal_header_bounds(col_bytes, ones)?;
    if !num_cmp_overlaps_bounds(cmp, lo, hi) {
        Some(TemplateTruth::AlwaysFalse)
    } else if num_cmp_always_true(cmp, lo, hi) {
        Some(TemplateTruth::AlwaysTrue)
    } else {
        Some(TemplateTruth::Maybe)
    }
}

/// Map `cmp` on `v = min + r` (`0 <= r <= max_r`) to `(r_lo, r_hi, is_neq)`.
/// Returns `None` when the predicate can never match `[min..=min+max_r]`.
#[inline(always)]
fn residual_interval(
    cmp: crate::query::NumCmp,
    min: u64,
    max_r: u64,
) -> Option<(u64, u64, bool)> {
    let hi_val = min.saturating_add(max_r);
    match cmp {
        crate::query::NumCmp::Eq(x) => {
            if x < min || x > hi_val {
                None
            } else {
                let d = x - min;
                Some((d, d, false))
            }
        }
        crate::query::NumCmp::Ne(x) => {
            if x < min || x > hi_val {
                Some((0, max_r, false))
            } else if max_r == 0 {
                None
            } else {
                let d = x - min;
                Some((d, d, true))
            }
        }
        crate::query::NumCmp::Ge(x) => {
            if x > hi_val {
                None
            } else {
                Some((x.saturating_sub(min), max_r, false))
            }
        }
        crate::query::NumCmp::Gt(x) => {
            let Some(xp1) = x.checked_add(1) else {
                return None;
            };
            if xp1 > hi_val {
                None
            } else {
                Some((xp1.saturating_sub(min), max_r, false))
            }
        }
        crate::query::NumCmp::Le(x) => {
            if x < min {
                None
            } else {
                Some((0, (x - min).min(max_r), false))
            }
        }
        crate::query::NumCmp::Lt(x) => {
            if x <= min {
                None
            } else {
                Some((0, (x - 1 - min).min(max_r), false))
            }
        }
        crate::query::NumCmp::Range(a, b) => {
            if a > b || a > hi_val || b < min {
                None
            } else {
                Some((a.saturating_sub(min), (b - min).min(max_r), false))
            }
        }
    }
}

#[inline(always)]
fn unpack_u16_chunk64(packed: &[u8], start_row: usize, count: usize, width: u8, out: &mut [u16; 64]) {
    let w = width as u32;
    let m = (1u64 << w) - 1;
    let mut bit_pos = start_row * (width as usize);
    let mut i = 0usize;
    if w <= 14 {
        let step_bits = (width as usize) * 4;
        while i + 4 <= count {
            let byte_idx = bit_pos >> 3;
            let bit_off = (bit_pos & 7) as u32;
            let v = if byte_idx + 8 <= packed.len() && bit_off + 4 * w <= 64 {
                let raw = unsafe {
                    packed.as_ptr().add(byte_idx).cast::<u64>().read_unaligned().to_le()
                };
                raw >> bit_off
            } else if byte_idx + 16 <= packed.len() {
                let raw = unsafe {
                    packed.as_ptr().add(byte_idx).cast::<u128>().read_unaligned().to_le()
                };
                (raw >> bit_off) as u64
            } else {
                let mut r = 0u128;
                let rem = packed.len().saturating_sub(byte_idx).min(16);
                for k in 0..rem {
                    r |= u128::from(unsafe { *packed.get_unchecked(byte_idx + k) }) << (8 * k);
                }
                (r >> bit_off) as u64
            };
            let u16x4 = (v & m)
                | (((v >> w) & m) << 16)
                | (((v >> (2 * w)) & m) << 32)
                | (((v >> (3 * w)) & m) << 48);
            unsafe {
                out.as_mut_ptr().add(i).cast::<u64>().write_unaligned(u16x4.to_le());
            }
            i += 4;
            bit_pos += step_bits;
        }
    }
    while i < count {
        out[i] = read_packed_bits_at(packed, bit_pos, width) as u16;
        i += 1;
        bit_pos += width as usize;
    }
}

#[inline(always)]
fn cmp_u16_chunk64_mask(buf: &[u16; 64], count: usize, lo: u16, hi: u16, is_neq: bool) -> u64 {
    let count_mask = if count >= 64 { u64::MAX } else { (1u64 << count) - 1 };
    #[cfg(target_arch = "aarch64")]
    if count == 64 {
        unsafe {
            use core::arch::aarch64::*;
            const BIT_WEIGHTS: [u8; 16] =
                [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
            let vw = vld1q_u8(BIT_WEIGHTS.as_ptr());
            let vlo = vdupq_n_u16(lo);
            let vhi = vdupq_n_u16(hi);
            let p = buf.as_ptr();
            let mut mask64 = 0u64;
            for g in 0..4usize {
                let a = vld1q_u16(p.add(g * 16));
                let b = vld1q_u16(p.add(g * 16 + 8));
                let (ca, cb) = if is_neq {
                    (vmvnq_u16(vceqq_u16(a, vlo)), vmvnq_u16(vceqq_u16(b, vlo)))
                } else if lo == hi {
                    (vceqq_u16(a, vlo), vceqq_u16(b, vlo))
                } else if lo == 0 {
                    (vcleq_u16(a, vhi), vcleq_u16(b, vhi))
                } else {
                    (
                        vandq_u16(vcgeq_u16(a, vlo), vcleq_u16(a, vhi)),
                        vandq_u16(vcgeq_u16(b, vlo), vcleq_u16(b, vhi)),
                    )
                };
                let n = vuzp1q_u8(vreinterpretq_u8_u16(ca), vreinterpretq_u8_u16(cb));
                let m = vandq_u8(n, vw);
                let m16 =
                    (vaddv_u8(vget_low_u8(m)) as u64) | ((vaddv_u8(vget_high_u8(m)) as u64) << 8);
                mask64 |= m16 << (g * 16);
            }
            return mask64;
        }
    }
    let mut w = 0u64;
    if is_neq {
        for (i, &v) in buf[..count].iter().enumerate() {
            w |= u64::from(v != lo) << i;
        }
    } else if lo == hi {
        for (i, &v) in buf[..count].iter().enumerate() {
            w |= u64::from(v == lo) << i;
        }
    } else {
        for (i, &v) in buf[..count].iter().enumerate() {
            w |= u64::from(v >= lo && v <= hi) << i;
        }
    }
    w & count_mask
}

/// Fast zero-arena masked evaluator for `MODEL_DECIMAL_PACKED` columns.
/// Mutates `active` in place and returns `Ok(true)` when handled without touching `DecodeScratch`.
fn try_eval_decimal_packed_masked(
    col_bytes: &[u8],
    ones: usize,
    pred: &crate::ops::ColPred<'_>,
    active: &mut TplMask,
) -> Result<bool> {
    if matches!(active, TplMask::Empty) {
        return Ok(true);
    }
    let crate::ops::ValuePred::Num(cmp) = pred.pred else {
        return Ok(false);
    };
    if matches!(
        pred.col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
        return Ok(false);
    }
    if ones < 2 || col_bytes.first() != Some(&MODEL_DECIMAL_PACKED) {
        return Ok(false);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let min = cursor.u64()?;
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column packed decimal width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed_bytes = cursor.take(packed_len)?;
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let _ = FastBitReader::new_validated(packed_bytes, ones, width)?;
    let max_r = if width == 0 {
        0
    } else if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if min.checked_add(max_r).is_none() {
        return Ok(false);
    }
    let Some((r_lo, r_hi, is_neq)) = residual_interval(cmp, min, max_r) else {
        *active = TplMask::Empty;
        return Ok(true);
    };
    if !is_neq && r_lo == 0 && r_hi >= max_r {
        return Ok(true);
    }
    if width == 0 {
        let ok = if is_neq { 0 != r_lo } else { r_lo == 0 };
        if !ok {
            *active = TplMask::Empty;
        }
        return Ok(true);
    }

    let wlen = ones.div_ceil(64);
    let mut buf = [0u16; 64];
    match active {
        TplMask::Empty => {}
        TplMask::All => {
            let mut bs = RowBitset::all_zeros(ones);
            let ws = bs.words_mut();
            let mut pop = 0usize;
            if width <= 16 {
                let (lo16, hi16) = (r_lo as u16, r_hi as u16);
                for (wi, out_slot) in ws.iter_mut().enumerate().take(wlen) {
                    let start_row = wi << 6;
                    let cnt = (ones - start_row).min(64);
                    unpack_u16_chunk64(packed_bytes, start_row, cnt, width, &mut buf);
                    let m = cmp_u16_chunk64_mask(&buf, cnt, lo16, hi16, is_neq);
                    *out_slot = m;
                    pop += m.count_ones() as usize;
                }
            } else {
                let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
                for (wi, out_slot) in ws.iter_mut().enumerate().take(wlen) {
                    let start_row = wi << 6;
                    let cnt = (ones - start_row).min(64);
                    let mut m = 0u64;
                    for bit in 0..cnt {
                        let r = rdr.next_u64();
                        let ok = if is_neq { r != r_lo } else { r >= r_lo && r <= r_hi };
                        m |= u64::from(ok) << bit;
                    }
                    *out_slot = m;
                    pop += m.count_ones() as usize;
                }
            }
            *active = TplMask::from_pop(bs, pop, ones);
        }
        TplMask::Bits(ref mut bs, ref mut cur_pop) => {
            let ws = bs.words_mut();
            let mut new_pop = 0usize;
            for (wi, slot_w) in ws.iter_mut().enumerate().take(wlen) {
                let w0 = *slot_w;
                if w0 == 0 {
                    continue;
                }
                let start_row = wi << 6;
                let cnt = (ones - start_row).min(64);
                let out_w = if w0.count_ones() <= 16 || width > 16 {
                    let mut w = w0;
                    let mut kept = 0u64;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let r = read_packed_bits_at(
                            packed_bytes,
                            (start_row + bit) * (width as usize),
                            width,
                        );
                        let ok = if is_neq { r != r_lo } else { r >= r_lo && r <= r_hi };
                        if ok {
                            kept |= 1u64 << bit;
                        }
                        w &= w - 1;
                    }
                    kept
                } else {
                    unpack_u16_chunk64(packed_bytes, start_row, cnt, width, &mut buf);
                    w0 & cmp_u16_chunk64_mask(&buf, cnt, r_lo as u16, r_hi as u16, is_neq)
                };
                *slot_w = out_w;
                new_pop += out_w.count_ones() as usize;
            }
            if new_pop == 0 {
                *active = TplMask::Empty;
            } else {
                *cur_pop = new_pop;
            }
        }
    }
    Ok(true)
}

#[inline]
fn expr_child_cost(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    expr: &crate::ops::PredExpr<'_>,
) -> u16 {
    match *expr {
        crate::ops::PredExpr::True => 0,
        crate::ops::PredExpr::Col(ref p) => {
            let Ok(SlotLookup::Slot(s)) = resolve_template_slot(segs, col_slices, p.col) else {
                return 100;
            };
            let cb = col_slices[s];
            match cb.first().copied() {
                Some(MODEL_DICTIONARY) => {
                    let entries = cb.get(1).copied().unwrap_or(64);
                    if entries <= 16 { 1 + u16::from(entries >> 2) } else { 25 }
                }
                Some(MODEL_DECIMAL_AFFINE) => 2,
                Some(MODEL_DECIMAL_PACKED) if cb.len() >= 10 => 6 + u16::from(cb[9]),
                Some(MODEL_SCALED_DECIMAL) if cb.len() >= 11 => 6 + u16::from(cb[10]),
                Some(MODEL_FRONT_DICT) => 80,
                _ => 120,
            }
        }
        crate::ops::PredExpr::Not(inner) => expr_child_cost(segs, col_slices, inner).saturating_add(5),
        crate::ops::PredExpr::All(_) | crate::ops::PredExpr::Any(_) => 150,
        crate::ops::PredExpr::RowContains(_) => 300,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_expr_into_and(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    expr: &crate::ops::PredExpr<'_>,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
    acc: &mut TplMask,
) -> Result<bool> {
    if matches!(*acc, TplMask::Empty) {
        return Ok(true);
    }
    match *expr {
        crate::ops::PredExpr::True => Ok(true),
        crate::ops::PredExpr::Col(ref p) => match resolve_template_slot(segs, col_slices, p.col)? {
            SlotLookup::Absent => {
                *acc = TplMask::Empty;
                Ok(true)
            }
            SlotLookup::Fallback => Ok(false),
            SlotLookup::Slot(s) => {
                if try_eval_decimal_packed_masked(col_slices[s], ones, p, acc)?
                    || try_eval_scaled_dec_masked(col_slices[s], ones, p, acc)?
                    || try_eval_dict_masked(col_slices[s], ones, p, acc)?
                {
                    return Ok(true);
                }
                let cm = eval_col_pred_mask(
                    col_slices[s],
                    ones,
                    p,
                    hdr_raw_len,
                    raw_mark,
                    slice_mark,
                    scratch,
                )?;
                let prev = std::mem::replace(acc, TplMask::Empty);
                *acc = prev.bitand(cm, ones);
                Ok(true)
            }
        },
        crate::ops::PredExpr::All(children) => {
            if children.len() <= 16 {
                let mut order: [u8; 16] = core::array::from_fn(|i| i as u8);
                let mut costs = [0u16; 16];
                for (i, c) in children.iter().enumerate() {
                    costs[i] = expr_child_cost(segs, col_slices, c);
                }
                order[..children.len()]
                    .sort_unstable_by_key(|&idx| costs[idx as usize]);
                for &idx in &order[..children.len()] {
                    if matches!(*acc, TplMask::Empty) {
                        break;
                    }
                    if !eval_expr_into_and(
                        segs,
                        col_slices,
                        ones,
                        &children[idx as usize],
                        hdr_raw_len,
                        raw_mark,
                        slice_mark,
                        scratch,
                        acc,
                    )? {
                        return Ok(false);
                    }
                }
            } else {
                for c in children {
                    if matches!(*acc, TplMask::Empty) {
                        break;
                    }
                    if !eval_expr_into_and(
                        segs,
                        col_slices,
                        ones,
                        c,
                        hdr_raw_len,
                        raw_mark,
                        slice_mark,
                        scratch,
                        acc,
                    )? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
        _ => {
            let Some(cm) = eval_expr_inner(
                segs,
                col_slices,
                ones,
                expr,
                hdr_raw_len,
                raw_mark,
                slice_mark,
                scratch,
            )?
            else {
                return Ok(false);
            };
            let prev = std::mem::replace(acc, TplMask::Empty);
            *acc = prev.bitand(cm, ones);
            Ok(true)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_row_contains_mask(
    segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    needle: &[u8],
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<TplMask> {
    match precheck_template_substr(segs, col_slices, ones, needle)? {
        TemplatePreCheck::Never => Ok(TplMask::Empty),
        TemplatePreCheck::Always => Ok(TplMask::All),
        TemplatePreCheck::SingleSlot(s) => {
            ensure_col_raw_arena(col_slices[s], hdr_raw_len, &mut scratch.raw_arena);
            let (_, hits) = eval_substr_on_single_slot(col_slices[s], ones, ones, needle, scratch)?;
            reset_scratch_to(scratch, raw_mark, slice_mark);
            Ok(TplMask::from_hits(&hits, ones))
        }
        TemplatePreCheck::CheckTemplate => {
            ensure_full_raw_arena(hdr_raw_len, &mut scratch.raw_arena);
            let slots = col_slices.len();
            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
            for s in 0..slots {
                cols[s] = PreparedColumn::parse_and_validate(col_slices[s], ones, scratch)?;
            }
            let unbound = UnboundTemplate::build(segs, &cols[..slots], scratch);
            let n_exec = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(n_exec);
            let mut tpl = unbound.bind(scratch);
            let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
            let mut bs = RowBitset::all_zeros(ones);
            for t in 0..ones {
                let written = unsafe {
                    let sp = row_buf.as_mut_ptr();
                    let ep = tpl.emit_row(sp);
                    ep.offset_from(sp) as usize
                };
                if contains_subslice(&row_buf[..written], needle) {
                    bs.set(t);
                }
            }
            reset_scratch_to(scratch, raw_mark, slice_mark);
            Ok(TplMask::normalize(bs, ones))
        }
    }
}
