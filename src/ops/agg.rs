//! Compressed-domain scalar aggregations (`aggregate_column`) and `GROUP BY` (`group_by`)
//! without full-frame decompression or per-row string materialization.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{query, DecodeOptions, Frame, Result, BLOCK_COLUMN};

use super::expr::{line_matches_expr, PredExpr};
use super::line::{
    decode_delim_str, find_line_col_span, parse_canonical_u64, scan_json_val_end,
    unescape_json_str, ColTarget,
};
use super::{build_par_meta_and_boundary, collect_par_blocks, SyncCell};

/// Summary statistics accumulated over matching records (`COUNT(*)`, `COUNT(col)`, `SUM`, `MIN`, `MAX`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AggStats {
    /// Total number of records satisfying the `WHERE` predicate (`COUNT(*)`).
    pub row_count: u64,
    /// Number of matching records where `agg_col` is a canonical unsigned integer (`COUNT(agg_col)`).
    pub num_count: u64,
    /// Exact 128-bit sum across numeric matches (`SUM(agg_col)`).
    pub sum: u128,
    /// Minimum numeric value (`MIN(agg_col)`), or `None` when `num_count == 0`.
    pub min: Option<u64>,
    /// Maximum numeric value (`MAX(agg_col)`), or `None` when `num_count == 0`.
    pub max: Option<u64>,
}

impl AggStats {
    #[inline]
    pub fn avg(&self) -> Option<f64> {
        (self.num_count > 0).then(|| (self.sum as f64) / (self.num_count as f64))
    }

    #[inline]
    pub fn merge(&mut self, other: &Self) {
        if other.row_count == 0 {
            return;
        }
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
    }

    #[inline]
    pub(crate) fn add_rows_only(&mut self, n: u64) {
        self.row_count += n;
    }

    #[inline]
    pub(crate) fn add_num(&mut self, v: u64) {
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
        if let Some(v) = parse_canonical_u64(&line[s..e]) {
            stats.add_num(v);
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
            if let Some(v) = parse_canonical_u64(&line[as_..ae]) {
                delta.add_num(v);
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

pub(super) fn aggregate_column_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_col: ColTarget<'_>,
    options: &DecodeOptions,
) -> Result<AggStats> {
    options.validate()?;
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        let nb = in_blocks.len();
        if nb > 1 {
            let (meta, boundary_lines) = build_par_meta_and_boundary(&in_blocks, options)?;
            let mut total = AggStats::default();
            for bl in boundary_lines.iter() {
                agg_line(bl.bytes, where_expr, agg_col, &mut total);
            }
            let workers = options.threads.min(nb);
            let next = AtomicUsize::new(0);
            let failed = AtomicBool::new(false);
            let worker_stats: Vec<SyncCell<Result<AggStats>>> =
                (0..workers).map(|_| SyncCell::new(Ok(AggStats::default()))).collect();
            let step = if nb >= workers * 8 { 4 } else { 1 };

            let run_worker = |wid: usize| {
                let mut dec_buf = Vec::new();
                let mut local = AggStats::default();
                loop {
                    if failed.load(Ordering::Relaxed) {
                        break;
                    }
                    let base = next.fetch_add(step, Ordering::Relaxed);
                    if base >= nb {
                        break;
                    }
                    let end = (base + step).min(nb);
                    for b in base..end {
                        let mb = &meta[b];
                        if mb.row_count == 0 {
                            continue;
                        }
                        let ib = &in_blocks[b];
                        let res: Result<()> = (|| {
                            if mb.decoded_raw.is_none() {
                                if let Some(st) = crate::columns::aggregate_grammar_block(
                                    ib.payload,
                                    mb.had_in_carry,
                                    where_expr,
                                    agg_col,
                                )? {
                                    local.merge(&st);
                                    return Ok(());
                                }
                            }
                            let raw: &[u8] = if let Some(ref pre) = mb.decoded_raw {
                                pre
                            } else {
                                query::decode_block_into(
                                    ib.f_bytes,
                                    ib.local_blk,
                                    ib.info.raw_size,
                                    options,
                                    &mut dec_buf,
                                )?;
                                &dec_buf
                            };
                            if raw.is_empty() {
                                return Ok(());
                            }
                            let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
                            let start_i = usize::from(mb.had_in_carry);
                            let end_i = lines.len()
                                - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
                            for &lb in lines.iter().take(end_i).skip(start_i) {
                                agg_line(lb, where_expr, agg_col, &mut local);
                            }
                            Ok(())
                        })();
                        if let Err(e) = res {
                            unsafe {
                                *worker_stats[wid].get() = Err(e);
                            }
                            failed.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
                unsafe {
                    *worker_stats[wid].get() = Ok(local);
                }
            };

            crate::pool::parallel_run(workers, run_worker);
            for slot in worker_stats {
                total.merge(&slot.into_inner()?);
            }
            return Ok(total);
        }
    }

    let mut total = AggStats::default();
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
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

pub(super) fn group_by_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
    options: &DecodeOptions,
) -> Result<Vec<GroupRow>> {
    options.validate()?;
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        let nb = in_blocks.len();
        if nb > 1 {
            let (meta, boundary_lines) = build_par_meta_and_boundary(&in_blocks, options)?;
            let mut total_map = GroupMap::default();
            for bl in boundary_lines.iter() {
                group_by_line(bl.bytes, where_expr, group_col, agg_col, &mut total_map);
            }
            let workers = options.threads.min(nb);
            let next = AtomicUsize::new(0);
            let failed = AtomicBool::new(false);
            let worker_maps: Vec<SyncCell<Result<GroupMap>>> =
                (0..workers).map(|_| SyncCell::new(Ok(GroupMap::default()))).collect();
            let step = if nb >= workers * 8 { 4 } else { 1 };

            let run_worker = |wid: usize| {
                let mut dec_buf = Vec::new();
                let mut local = GroupMap::default();
                loop {
                    if failed.load(Ordering::Relaxed) {
                        break;
                    }
                    let base = next.fetch_add(step, Ordering::Relaxed);
                    if base >= nb {
                        break;
                    }
                    let end = (base + step).min(nb);
                    for b in base..end {
                        let mb = &meta[b];
                        if mb.row_count == 0 {
                            continue;
                        }
                        let ib = &in_blocks[b];
                        let res: Result<()> = (|| {
                            if mb.decoded_raw.is_none()
                                && crate::columns::group_by_grammar_block(
                                    ib.payload,
                                    mb.had_in_carry,
                                    where_expr,
                                    group_col,
                                    agg_col,
                                    &mut local,
                                )?
                            {
                                return Ok(());
                            }
                            let raw: &[u8] = if let Some(ref pre) = mb.decoded_raw {
                                pre
                            } else {
                                query::decode_block_into(
                                    ib.f_bytes,
                                    ib.local_blk,
                                    ib.info.raw_size,
                                    options,
                                    &mut dec_buf,
                                )?;
                                &dec_buf
                            };
                            if raw.is_empty() {
                                return Ok(());
                            }
                            let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
                            let start_i = usize::from(mb.had_in_carry);
                            let end_i = lines.len()
                                - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
                            for &lb in lines.iter().take(end_i).skip(start_i) {
                                group_by_line(lb, where_expr, group_col, agg_col, &mut local);
                            }
                            Ok(())
                        })();
                        if let Err(e) = res {
                            unsafe {
                                *worker_maps[wid].get() = Err(e);
                            }
                            failed.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
                unsafe {
                    *worker_maps[wid].get() = Ok(local);
                }
            };

            crate::pool::parallel_run(workers, run_worker);
            for slot in worker_maps {
                total_map.merge_map(slot.into_inner()?);
            }
            return Ok(total_map.into_sorted_rows());
        }
    }

    let mut total_map = GroupMap::default();
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
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
