#[derive(Clone, Copy)]
enum ParsedSlotNum {
    None,
    U64(u64),
    Dec(crate::ops::DecNum),
}

#[inline(always)]
fn parse_slot_num_bytes(s: &[u8], digit_delim: bool) -> ParsedSlotNum {
    if digit_delim {
        return ParsedSlotNum::None;
    }
    if let Some(v) = crate::ops::parse_canonical_u64(s) {
        return ParsedSlotNum::U64(v);
    }
    if let Some(d) = crate::ops::DecNum::parse(s) {
        return ParsedSlotNum::Dec(d);
    }
    ParsedSlotNum::None
}

#[inline(always)]
fn add_parsed_slot_num(stats: &mut crate::ops::AggStats, p: ParsedSlotNum, count: u64) {
    if count == 0 {
        return;
    }
    match p {
        ParsedSlotNum::None => stats.add_rows_only(count),
        ParsedSlotNum::U64(v) => stats.add_const_num(v, count),
        ParsedSlotNum::Dec(d) => stats.add_scaled_batch(
            d.mantissa * (count as i128),
            d.mantissa,
            d.mantissa,
            d.scale,
            count,
        ),
    }
}

/// Unpack `MODEL_SCALED_DECIMAL` residuals directly into `scratch.u64_arena` without ASCII formatting.
fn try_unpack_scaled_dec_view(
    col_bytes: &[u8],
    ones: usize,
    digit_delim: bool,
    u64_arena: &mut Vec<u64>,
) -> Result<Option<(u8, i64, Option<usize>)>> {
    if digit_delim || ones < 2 || col_bytes.first() != Some(&MODEL_SCALED_DECIMAL) {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let scale = cursor.byte()?;
    if scale > 9 {
        return Err(invalid("column scaled decimal scale"));
    }
    let min_v = cursor.i64()?;
    if scale == 0 && min_v >= 0 {
        return Err(invalid("column scaled decimal noncanonical unsigned"));
    }
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column scaled decimal width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed_bytes = cursor.take(packed_len)?;
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let max_r = if width == 0 {
        0
    } else if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
        return Err(invalid("column scaled decimal overflow"));
    }
    if width == 0 {
        BitReader::new(packed_bytes).finish_zero_padded()?;
        return Ok(Some((scale, min_v, None)));
    }
    let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
    let off = u64_arena.len();
    u64_arena.reserve(ones);
    for _ in 0..ones {
        u64_arena.push(rdr.next_u64());
    }
    Ok(Some((scale, min_v, Some(off))))
}

fn scaled_dec_col_alphabet(c: &mut Cursor<'_>, mask: &mut [u64; 4]) -> Result<()> {
    let scale = c.byte()?;
    let min_v = c.i64()?;
    add_bytes_to_mask(mask, b"0123456789");
    if min_v < 0 {
        add_bytes_to_mask(mask, b"-");
    }
    if scale > 0 {
        add_bytes_to_mask(mask, b".");
    }
    Ok(())
}

fn dec_col_zone_truth(
    col_bytes: &[u8],
    ones: usize,
    dcmp: crate::ops::DecCmp,
) -> Option<TemplateTruth> {
    use crate::ops::dec_num::ScaledResidualCmp;
    if ones == 0 {
        return Some(TemplateTruth::AlwaysFalse);
    }
    let mut c = Cursor::new(col_bytes);
    let tag = c.byte().ok()?;
    match tag {
        MODEL_SCALED_DECIMAL => {
            let scale = c.byte().ok()?;
            let min_v = c.i64().ok()?;
            let width = c.byte().ok()?;
            if scale > 9 || width > 64 {
                return None;
            }
            let max_r = if width == 0 {
                0
            } else if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
                return None;
            }
            Some(match dcmp.to_scaled_i64_residual_i128(scale, min_v as i128, max_r) {
                ScaledResidualCmp::AlwaysFalse => TemplateTruth::AlwaysFalse,
                ScaledResidualCmp::AlwaysTrue => TemplateTruth::AlwaysTrue,
                ScaledResidualCmp::Interval { .. } => TemplateTruth::Maybe,
            })
        }
        MODEL_DECIMAL_PACKED | MODEL_DECIMAL_AFFINE | MODEL_DECIMAL_DELTA => match dcmp.to_u64_domain() {
            crate::ops::dec_num::U64DomainCmp::AlwaysFalse => Some(TemplateTruth::AlwaysFalse),
            crate::ops::dec_num::U64DomainCmp::AlwaysTrue => Some(TemplateTruth::AlwaysTrue),
            crate::ops::dec_num::U64DomainCmp::Cmp(ncmp) => {
                numeric_col_zone_truth(col_bytes, ones, ncmp)
            }
        },
        _ => None,
    }
}

/// Zero-arena NEON/SWAR masked evaluator for `MODEL_SCALED_DECIMAL` (and `DecCmp` on `MODEL_DECIMAL_PACKED`).
fn try_eval_scaled_dec_masked(
    col_bytes: &[u8],
    ones: usize,
    pred: &crate::ops::ColPred<'_>,
    active: &mut TplMask,
) -> Result<bool> {
    use crate::ops::dec_num::{ScaledResidualCmp, U64DomainCmp};
    if matches!(active, TplMask::Empty) {
        return Ok(true);
    }
    if matches!(
        pred.col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
        return Ok(false);
    }
    if ones < 2 {
        return Ok(false);
    }
    let Some(&tag) = col_bytes.first() else {
        return Ok(false);
    };
    if tag == MODEL_DECIMAL_PACKED {
        let crate::ops::ValuePred::Dec(dcmp) = pred.pred else {
            return Ok(false);
        };
        return match dcmp.to_u64_domain() {
            U64DomainCmp::AlwaysFalse => {
                *active = TplMask::Empty;
                Ok(true)
            }
            U64DomainCmp::AlwaysTrue => Ok(true),
            U64DomainCmp::Cmp(ncmp) => {
                let p2 = crate::ops::ColPred {
                    col: pred.col,
                    pred: crate::ops::ValuePred::Num(ncmp),
                };
                try_eval_decimal_packed_masked(col_bytes, ones, &p2, active)
            }
        };
    }
    if tag != MODEL_SCALED_DECIMAL {
        return Ok(false);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let scale = cursor.byte()?;
    if scale > 9 {
        return Err(invalid("column scaled decimal scale"));
    }
    let min_v = cursor.i64()?;
    if scale == 0 && min_v >= 0 {
        return Err(invalid("column scaled decimal noncanonical unsigned"));
    }
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column scaled decimal width"));
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
    if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
        return Err(invalid("column scaled decimal overflow"));
    }

    let dcmp = match pred.pred {
        crate::ops::ValuePred::Dec(d) => d,
        crate::ops::ValuePred::Num(_) => {
            if scale > 0 {
                *active = TplMask::Empty;
                return Ok(true);
            }
            return Ok(false);
        }
        crate::ops::ValuePred::Str(_) => return Ok(false),
    };

    let (r_lo, r_hi, is_neq) =
        match dcmp.to_scaled_i64_residual_i128(scale, min_v as i128, max_r) {
            ScaledResidualCmp::AlwaysFalse => {
                *active = TplMask::Empty;
                return Ok(true);
            }
            ScaledResidualCmp::AlwaysTrue => return Ok(true),
            ScaledResidualCmp::Interval { lo, hi, is_neq } => (lo, hi, is_neq),
        };
    if width == 0 {
        BitReader::new(packed_bytes).finish_zero_padded()?;
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
                    let cnt = (ones - (wi << 6)).min(64);
                    let mut m = 0u64;
                    for bit in 0..cnt {
                        let r = rdr.next_u64();
                        let ok = if is_neq {
                            r != r_lo
                        } else {
                            r >= r_lo && r <= r_hi
                        };
                        m |= u64::from(ok) << bit;
                    }
                    *out_slot = m;
                    pop += m.count_ones() as usize;
                }
            }
            *active = TplMask::from_pop(bs, pop, ones);
        }
        TplMask::Bits(ref mut bs, ref mut pop_ref) => {
            let mut new_pop = 0usize;
            for (wi, slot) in bs.words_mut().iter_mut().enumerate().take(wlen) {
                let mut w = *slot;
                if w == 0 {
                    continue;
                }
                let base_row = wi << 6;
                let cnt = (ones - base_row).min(64);
                if width <= 16 && w.count_ones() >= 12 {
                    unpack_u16_chunk64(packed_bytes, base_row, cnt, width, &mut buf);
                    let m = cmp_u16_chunk64_mask(&buf, cnt, r_lo as u16, r_hi as u16, is_neq);
                    w &= m;
                } else {
                    let mut cur = w;
                    while cur != 0 {
                        let bit = cur.trailing_zeros() as usize;
                        let r =
                            read_packed_bits_at(packed_bytes, (base_row + bit) * width as usize, width);
                        let ok = if is_neq {
                            r != r_lo
                        } else {
                            r >= r_lo && r <= r_hi
                        };
                        if !ok {
                            w &= !(1u64 << bit);
                        }
                        cur &= cur - 1;
                    }
                }
                *slot = w;
                new_pop += w.count_ones() as usize;
            }
            if new_pop == 0 {
                *active = TplMask::Empty;
            } else {
                *pop_ref = new_pop;
            }
        }
    }
    Ok(true)
}

/// Zero-arena exact compressed-domain aggregation on `MODEL_SCALED_DECIMAL`.
fn try_agg_scaled_dec_fast(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    target_col: crate::ops::ColTarget<'_>,
    stats: &mut crate::ops::AggStats,
) -> Result<bool> {
    if ones < 2 || col_bytes.first() != Some(&MODEL_SCALED_DECIMAL) {
        return Ok(false);
    }
    if matches!(
        target_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
        return Ok(false);
    }
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(true);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let scale = cursor.byte()?;
    if scale > 9 {
        return Err(invalid("column scaled decimal scale"));
    }
    let min_v = cursor.i64()?;
    if scale == 0 && min_v >= 0 {
        return Err(invalid("column scaled decimal noncanonical unsigned"));
    }
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column scaled decimal width"));
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
    if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
        return Err(invalid("column scaled decimal overflow"));
    }
    if width == 0 {
        BitReader::new(packed_bytes).finish_zero_padded()?;
        let mv = min_v as i128;
        stats.add_scaled_batch(mv * (pop as i128), mv, mv, scale, pop as u64);
        return Ok(true);
    }

    let mut res_sum = 0u128;
    let mut res_lo = u64::MAX;
    let mut res_hi = 0u64;
    match mask {
        TplMask::All => {
            let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
            for _ in 0..ones {
                let r = rdr.next_u64();
                res_sum += u128::from(r);
                res_lo = res_lo.min(r);
                res_hi = res_hi.max(r);
            }
        }
        TplMask::Bits(bs, _) => {
            for (wi, &w0) in bs.words().iter().enumerate() {
                let mut w = w0;
                let base = wi << 6;
                while w != 0 {
                    let row = base + w.trailing_zeros() as usize;
                    let r = read_packed_bits_at(packed_bytes, row * width as usize, width);
                    res_sum += u128::from(r);
                    res_lo = res_lo.min(r);
                    res_hi = res_hi.max(r);
                    w &= w - 1;
                }
            }
        }
        TplMask::Empty => return Ok(true),
    }
    let min_i128 = min_v as i128;
    let raw_sum = min_i128 * (pop as i128) + (res_sum as i128);
    let raw_lo = min_i128 + (res_lo as i128);
    let raw_hi = min_i128 + (res_hi as i128);
    stats.add_scaled_batch(raw_sum, raw_lo, raw_hi, scale, pop as u64);
    Ok(true)
}

/// Compressed-domain Top-K scanner for `MODEL_SCALED_DECIMAL` (zero ASCII formatting).
#[allow(clippy::too_many_arguments)]
fn try_scan_scaled_dec_topk(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    row_map: &mut TplRowMap<'_>,
    base_row: usize,
    digit_delim: bool,
    outer: &crate::ops::top_k::TopKHeap,
    cands: &mut CandTracker,
) -> Result<bool> {
    use crate::ops::top_k::SortKeyRef;
    use crate::ops::{DecNum, SortDir};

    if digit_delim || ones < 2 || col_bytes.first() != Some(&MODEL_SCALED_DECIMAL) {
        return Ok(false);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let scale = cursor.byte()?;
    if scale > 9 {
        return Err(invalid("column scaled decimal scale"));
    }
    let min_v = cursor.i64()?;
    if scale == 0 && min_v >= 0 {
        return Err(invalid("column scaled decimal noncanonical unsigned"));
    }
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column scaled decimal width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed_bytes = cursor.take(packed_len)?;
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let max_r = if width == 0 {
        0
    } else if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
        return Err(invalid("column scaled decimal overflow"));
    }
    if width == 0 {
        BitReader::new(packed_bytes).finish_zero_padded()?;
        let ck = SortKeyRef::Dec(DecNum::new(min_v as i128, scale));
        push_const_key_cands(ck, ones, mask, row_map, base_row, outer, cands);
        return Ok(true);
    }
    let _ = FastBitReader::new_validated(packed_bytes, ones, width)?;
    let best_col_key = if outer.dir() == SortDir::Desc {
        DecNum::new((min_v as i128) + (max_r as i128), scale)
    } else {
        DecNum::new(min_v as i128, scale)
    };
    if !outer.can_accept_key_from(SortKeyRef::Dec(best_col_key), base_row) {
        return Ok(true);
    }

    let min_i128 = min_v as i128;
    match mask {
        TplMask::All => {
            let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
            for ti in 0..ones {
                let k = SortKeyRef::Dec(DecNum::new(min_i128 + (rdr.next_u64() as i128), scale));
                if outer.can_accept_key_from(k, base_row + ti) && cands.can_accept(k, base_row + ti)
                {
                    let b_row = row_map.row_of(ti);
                    cands.try_push(k, ti, b_row, base_row + b_row, outer);
                }
            }
        }
        TplMask::Bits(bs, _) => {
            for (wi, &w0) in bs.words().iter().enumerate() {
                let mut w = w0;
                let base = wi << 6;
                while w != 0 {
                    let ti = base + w.trailing_zeros() as usize;
                    let r = read_packed_bits_at(packed_bytes, ti * width as usize, width);
                    let k = SortKeyRef::Dec(DecNum::new(min_i128 + (r as i128), scale));
                    if outer.can_accept_key_from(k, base_row + ti)
                        && cands.can_accept(k, base_row + ti)
                    {
                        let b_row = row_map.row_of(ti);
                        cands.try_push(k, ti, b_row, base_row + b_row, outer);
                    }
                    w &= w - 1;
                }
            }
        }
        TplMask::Empty => {}
    }
    Ok(true)
}
