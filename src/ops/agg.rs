//! Compressed-domain scalar aggregations (`aggregate_column`) and `GROUP BY` (`group_by`)
//! without full-frame decompression or per-row string materialization.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::{query, DecodeOptions, Frame, Result, BLOCK_COLUMN};

use super::dec_num::{DecNum, POW10_I128};
use super::expr::{line_matches_expr, PredExpr};
use super::line::{
    decode_delim_str, find_line_col_span, parse_canonical_u64, scan_json_val_end,
    unescape_json_str, ColTarget,
};
use super::collect_par_blocks;

/// Summary statistics accumulated over matching records (`COUNT(*)`, `COUNT(col)`, `SUM`, `MIN`, `MAX`),
/// supporting both canonical `u64` integers and exact signed/fixed-point decimals ([`DecNum`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AggStats {
    /// Total number of records satisfying the `WHERE` predicate (`COUNT(*)`).
    pub row_count: u64,
    /// Number of matching records where `agg_col` is a canonical number (`COUNT(agg_col)`).
    pub num_count: u64,
    /// Exact 128-bit sum when all observed values are unsigned `u64` integers (`0` when `has_signed_or_dec`).
    pub sum: u128,
    /// Minimum `u64` value when `!has_signed_or_dec`, or `None` when `num_count == 0`.
    pub min: Option<u64>,
    /// Maximum `u64` value when `!has_signed_or_dec`, or `None` when `num_count == 0`.
    pub max: Option<u64>,
    /// True once any negative integer or fractional decimal (`19.99`, `-5`) has been accumulated.
    pub has_signed_or_dec: bool,
    /// Canonical minimal base-10 scale (`0..=18`) for `signed_sum`, `signed_min`, `signed_max`.
    pub dec_scale: u8,
    /// Exact scaled signed sum (`actual_sum = signed_sum * 10^(-dec_scale)` when `has_signed_or_dec`).
    pub signed_sum: i128,
    /// Exact scaled signed minimum (`actual_min = signed_min * 10^(-dec_scale)` when `has_signed_or_dec`).
    pub signed_min: Option<i128>,
    /// Exact scaled signed maximum (`actual_max = signed_max * 10^(-dec_scale)` when `has_signed_or_dec`).
    pub signed_max: Option<i128>,
}

impl AggStats {
    /// Exact [`DecNum`] sum across all accumulated numeric values (`0` when `num_count == 0`).
    #[inline]
    pub fn sum_dec(&self) -> DecNum {
        if self.num_count == 0 {
            DecNum::ZERO
        } else if self.has_signed_or_dec {
            DecNum::new(self.signed_sum, self.dec_scale)
        } else {
            DecNum::new(self.sum as i128, 0)
        }
    }

    /// Exact [`DecNum`] minimum, or `None` when `num_count == 0`.
    #[inline]
    pub fn min_dec(&self) -> Option<DecNum> {
        if self.has_signed_or_dec {
            self.signed_min.map(|m| DecNum::new(m, self.dec_scale))
        } else {
            self.min.map(DecNum::from_u64)
        }
    }

    /// Exact [`DecNum`] maximum, or `None` when `num_count == 0`.
    #[inline]
    pub fn max_dec(&self) -> Option<DecNum> {
        if self.has_signed_or_dec {
            self.signed_max.map(|m| DecNum::new(m, self.dec_scale))
        } else {
            self.max.map(DecNum::from_u64)
        }
    }

    #[inline]
    pub fn avg(&self) -> Option<f64> {
        (self.num_count > 0).then(|| {
            if self.has_signed_or_dec {
                let s = (self.signed_sum as f64) / (POW10_I128[self.dec_scale as usize] as f64);
                s / (self.num_count as f64)
            } else {
                (self.sum as f64) / (self.num_count as f64)
            }
        })
    }

    #[cold]
    fn promote_to_signed(&mut self) {
        if !self.has_signed_or_dec {
            self.has_signed_or_dec = true;
            self.dec_scale = 0;
            self.signed_sum = self.sum as i128;
            self.signed_min = self.min.map(|m| m as i128);
            self.signed_max = self.max.map(|m| m as i128);
            self.sum = 0;
            self.min = None;
            self.max = None;
        }
    }

    #[inline]
    fn normalize_dec(&mut self) {
        while self.dec_scale > 0 && self.signed_sum % 10 == 0 {
            let Some(lo) = self.signed_min else { break };
            let Some(hi) = self.signed_max else { break };
            if lo % 10 != 0 || hi % 10 != 0 {
                break;
            }
            self.signed_sum /= 10;
            self.signed_min = Some(lo / 10);
            self.signed_max = Some(hi / 10);
            self.dec_scale -= 1;
        }
    }

    #[inline]
    fn raise_scale(&mut self, target_scale: u8) {
        if target_scale > self.dec_scale {
            let mul = POW10_I128[(target_scale - self.dec_scale) as usize];
            self.signed_sum = self.signed_sum.saturating_mul(mul);
            self.signed_min = self.signed_min.map(|m| m.saturating_mul(mul));
            self.signed_max = self.signed_max.map(|m| m.saturating_mul(mul));
            self.dec_scale = target_scale;
        }
    }

    #[inline]
    pub fn merge(&mut self, other: &Self) {
        if other.row_count == 0 {
            return;
        }
        if !self.has_signed_or_dec && !other.has_signed_or_dec {
            self.row_count += other.row_count;
            if other.num_count > 0 {
                self.num_count += other.num_count;
                self.sum += other.sum;
                if let Some(omin) = other.min {
                    self.min = Some(self.min.map_or(omin, |m| m.min(omin)));
                }
                if let Some(omax) = other.max {
                    self.max = Some(self.max.map_or(omax, |m| m.max(omax)));
                }
            }
            return;
        }
        self.row_count += other.row_count;
        if other.num_count == 0 {
            return;
        }
        self.promote_to_signed();
        let mut rhs = *other;
        rhs.promote_to_signed();
        let sc = self.dec_scale.max(rhs.dec_scale);
        self.raise_scale(sc);
        rhs.raise_scale(sc);
        self.num_count += rhs.num_count;
        self.signed_sum = self.signed_sum.saturating_add(rhs.signed_sum);
        if let Some(omin) = rhs.signed_min {
            self.signed_min = Some(self.signed_min.map_or(omin, |m| m.min(omin)));
        }
        if let Some(omax) = rhs.signed_max {
            self.signed_max = Some(self.signed_max.map_or(omax, |m| m.max(omax)));
        }
        self.normalize_dec();
    }

    #[inline]
    pub(crate) fn add_rows_only(&mut self, n: u64) {
        self.row_count += n;
    }

    #[inline]
    pub(crate) fn add_dec(&mut self, d: DecNum) {
        self.promote_to_signed();
        self.row_count += 1;
        self.num_count += 1;
        let sc = self.dec_scale.max(d.scale);
        self.raise_scale(sc);
        let dm = d.mantissa.saturating_mul(POW10_I128[(sc - d.scale) as usize]);
        self.signed_sum = self.signed_sum.saturating_add(dm);
        self.signed_min = Some(self.signed_min.map_or(dm, |m| m.min(dm)));
        self.signed_max = Some(self.signed_max.map_or(dm, |m| m.max(dm)));
        self.normalize_dec();
    }

    #[inline]
    pub(crate) fn add_scaled_batch(
        &mut self,
        raw_sum: i128,
        raw_min: i128,
        raw_max: i128,
        scale: u8,
        count: u64,
    ) {
        if count == 0 {
            return;
        }
        if scale == 0 && raw_min >= 0 && !self.has_signed_or_dec {
            let lo = raw_min as u64;
            let hi = raw_max as u64;
            self.row_count += count;
            self.num_count += count;
            self.sum += raw_sum as u128;
            self.min = Some(self.min.map_or(lo, |m| m.min(lo)));
            self.max = Some(self.max.map_or(hi, |m| m.max(hi)));
            return;
        }
        self.promote_to_signed();
        self.row_count += count;
        self.num_count += count;
        let sc = self.dec_scale.max(scale);
        self.raise_scale(sc);
        let mul = POW10_I128[(sc - scale) as usize];
        let s = raw_sum.saturating_mul(mul);
        let lo = raw_min.saturating_mul(mul);
        let hi = raw_max.saturating_mul(mul);
        self.signed_sum = self.signed_sum.saturating_add(s);
        self.signed_min = Some(self.signed_min.map_or(lo, |m| m.min(lo)));
        self.signed_max = Some(self.signed_max.map_or(hi, |m| m.max(hi)));
        self.normalize_dec();
    }

    #[inline]
    pub(crate) fn add_num(&mut self, v: u64) {
        if self.has_signed_or_dec {
            self.add_dec(DecNum::from_u64(v));
            return;
        }
        self.row_count += 1;
        self.num_count += 1;
        self.sum += u128::from(v);
        self.min = Some(self.min.map_or(v, |m| m.min(v)));
        self.max = Some(self.max.map_or(v, |m| m.max(v)));
    }

    #[inline]
    pub(crate) fn add_const_num(&mut self, v: u64, n: u64) {
        if n == 0 {
            return;
        }
        if self.has_signed_or_dec {
            let mv = v as i128;
            self.add_scaled_batch(mv * (n as i128), mv, mv, 0, n);
            return;
        }
        self.row_count += n;
        self.num_count += n;
        self.sum += u128::from(v) * u128::from(n);
        self.min = Some(self.min.map_or(v, |m| m.min(v)));
        self.max = Some(self.max.map_or(v, |m| m.max(v)));
    }

    #[inline]
    pub(crate) fn add_affine_all(&mut self, first: u64, step_u64: u64, count: usize) {
        if count == 0 {
            return;
        }
        let last = first.wrapping_add(step_u64.wrapping_mul((count - 1) as u64));
        let n = count as u128;
        let ends = u128::from(first) + u128::from(last);
        let total = if (count & 1) == 0 {
            (n >> 1) * ends
        } else {
            n * (ends >> 1)
        };
        let (lo, hi) = if first <= last {
            (first, last)
        } else {
            (last, first)
        };
        if self.has_signed_or_dec {
            self.add_scaled_batch(total as i128, lo as i128, hi as i128, 0, count as u64);
            return;
        }
        self.row_count += count as u64;
        self.num_count += count as u64;
        self.sum += total;
        self.min = Some(self.min.map_or(lo, |m| m.min(lo)));
        self.max = Some(self.max.map_or(hi, |m| m.max(hi)));
    }
}

include!("group_map.rs");

/// Decode a raw column wire token into `(is_str, key_bytes)` for `GROUP BY`.
#[inline]
pub fn decode_group_token_ref<'a>(
    col: ColTarget<'_>,
    raw_tok: &'a [u8],
) -> Option<(bool, Cow<'a, [u8]>)> {
    if raw_tok.is_empty() || raw_tok.contains(&b'\n') || raw_tok.contains(&b'\r') {
        return None;
    }
    match col {
        ColTarget::Delimited { .. } => decode_delim_str(raw_tok).map(|c| (true, c)),
        ColTarget::JsonKey(_) => {
            if raw_tok.first() == Some(&b'"') {
                unescape_json_str(raw_tok).map(|c| (true, c))
            } else if scan_json_val_end(raw_tok, 0) == Some(raw_tok.len()) {
                Some((false, Cow::Borrowed(raw_tok)))
            } else {
                None
            }
        }
    }
}

/// Reference single-record updater for [`aggregate_column`](super::aggregate_column).
#[inline]
pub fn agg_line(
    line: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_col: ColTarget<'_>,
    stats: &mut AggStats,
) {
    if let Some(expr) = where_expr {
        if !line_matches_expr(line, expr) {
            return;
        }
    }
    if let Some((s, e)) = find_line_col_span(line, agg_col) {
        let tok = &line[s..e];
        if let Some(v) = parse_canonical_u64(tok) {
            stats.add_num(v);
            return;
        }
        if let Some(d) = super::dec_num::DecNum::parse(tok) {
            stats.add_dec(d);
            return;
        }
    }
    stats.add_rows_only(1);
}

/// Reference single-record updater for [`group_by`](super::group_by).
#[inline]
pub(crate) fn group_by_line(
    line: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
    out: &mut GroupMap,
) {
    if let Some(expr) = where_expr {
        if !line_matches_expr(line, expr) {
            return;
        }
    }
    let Some((gs, ge)) = find_line_col_span(line, group_col) else {
        return;
    };
    let Some((is_str, gkey)) = decode_group_token_ref(group_col, &line[gs..ge]) else {
        return;
    };
    let mut delta = AggStats::default();
    if let Some(acol) = agg_col {
        if let Some((as_, ae)) = find_line_col_span(line, acol) {
            let tok = &line[as_..ae];
            if let Some(v) = parse_canonical_u64(tok) {
                delta.add_num(v);
            } else if let Some(d) = super::dec_num::DecNum::parse(tok) {
                delta.add_dec(d);
            } else {
                delta.add_rows_only(1);
            }
        } else {
            delta.add_rows_only(1);
        }
    } else {
        delta.add_rows_only(1);
    }
    out.merge_decoded(is_str, &gkey, &delta);
}

pub(crate) fn aggregate_column_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_col: ColTarget<'_>,
    options: &DecodeOptions,
) -> Result<AggStats> {
    options.validate()?;
    let extra = [agg_col];
    let prepared = crate::mmap::prepare_query_bytes(
        frame_bytes,
        where_expr,
        &extra,
        true,
        options.threads,
    );
    let frame_bytes = prepared.as_ref();
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        if in_blocks.len() > 1 {
            return super::par_chunk::parallel_reduce_blocks(
                &in_blocks,
                options,
                |payload, carry, eval_in_carry, acc: &mut AggStats| {
                    if let Some(st) = crate::columns::aggregate_grammar_block_chunk(
                        payload,
                        carry,
                        eval_in_carry,
                        where_expr,
                        agg_col,
                    )? {
                        acc.merge(&st);
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                },
                |line, acc: &mut AggStats| agg_line(line, where_expr, agg_col, acc),
                |dst, src| dst.merge(&src),
            );
        }
    }

    let mut total = AggStats::default();
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let payload = parsed.blocks[local_blk].payload;
                if let Some(st) = crate::columns::aggregate_grammar_block_seq(
                    payload,
                    &mut carry,
                    where_expr,
                    agg_col,
                )? {
                    total.merge(&st);
                    continue;
                }
            }
            query::decode_block_into(f_bytes, local_blk, info.raw_size, options, &mut buf)?;
            if buf.is_empty() {
                continue;
            }
            let lines: Vec<&[u8]> = buf.split_inclusive(|&b| b == b'\n').collect();
            let blk_rows = lines.len();
            for (row_idx, &line_bytes) in lines.iter().enumerate() {
                if row_idx == 0 {
                    if let Some(mut c_bytes) = carry.take() {
                        c_bytes.extend_from_slice(line_bytes);
                        if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                            carry = Some(c_bytes);
                            break;
                        }
                        agg_line(&c_bytes, where_expr, agg_col, &mut total);
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some(line_bytes.to_vec());
                    break;
                }
                agg_line(line_bytes, where_expr, agg_col, &mut total);
            }
        }
    }
    if let Some(c_bytes) = carry.take() {
        agg_line(&c_bytes, where_expr, agg_col, &mut total);
    }
    Ok(total)
}

pub(crate) fn group_by_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
    options: &DecodeOptions,
) -> Result<Vec<GroupRow>> {
    options.validate()?;
    let mut extra = vec![group_col];
    if let Some(ac) = agg_col {
        extra.push(ac);
    }
    let prepared = crate::mmap::prepare_query_bytes(
        frame_bytes,
        where_expr,
        &extra,
        true,
        options.threads,
    );
    let frame_bytes = prepared.as_ref();
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        if in_blocks.len() > 1 {
            let total_map = super::par_chunk::parallel_reduce_blocks(
                &in_blocks,
                options,
                |payload, carry, eval_in_carry, acc: &mut GroupMap| {
                    crate::columns::group_by_grammar_block_chunk(
                        payload,
                        carry,
                        eval_in_carry,
                        where_expr,
                        group_col,
                        agg_col,
                        acc,
                    )
                },
                |line, acc: &mut GroupMap| group_by_line(line, where_expr, group_col, agg_col, acc),
                |dst, src| dst.merge_map(src),
            )?;
            return Ok(total_map.into_sorted_rows());
        }
    }

    let mut total_map = GroupMap::default();
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let payload = parsed.blocks[local_blk].payload;
                if crate::columns::group_by_grammar_block_seq(
                    payload,
                    &mut carry,
                    where_expr,
                    group_col,
                    agg_col,
                    &mut total_map,
                )? {
                    continue;
                }
            }
            query::decode_block_into(f_bytes, local_blk, info.raw_size, options, &mut buf)?;
            if buf.is_empty() {
                continue;
            }
            let lines: Vec<&[u8]> = buf.split_inclusive(|&b| b == b'\n').collect();
            let blk_rows = lines.len();
            for (row_idx, &line_bytes) in lines.iter().enumerate() {
                if row_idx == 0 {
                    if let Some(mut c_bytes) = carry.take() {
                        c_bytes.extend_from_slice(line_bytes);
                        if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                            carry = Some(c_bytes);
                            break;
                        }
                        group_by_line(&c_bytes, where_expr, group_col, agg_col, &mut total_map);
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some(line_bytes.to_vec());
                    break;
                }
                group_by_line(line_bytes, where_expr, group_col, agg_col, &mut total_map);
            }
        }
    }
    if let Some(c_bytes) = carry.take() {
        group_by_line(&c_bytes, where_expr, group_col, agg_col, &mut total_map);
    }
    Ok(total_map.into_sorted_rows())
}
