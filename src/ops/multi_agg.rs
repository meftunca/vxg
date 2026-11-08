//! Multi-column scalar aggregation and multi-key / multi-column `GROUP BY` reducer
//! backing `vxg::sql` compressed-domain execution.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::{query, DecodeOptions, Frame, Result, BLOCK_COLUMN};

use super::agg::{decode_group_token_ref, AggStats, GroupKey};
use super::collect_par_blocks;
use super::csv_header::{
    resolve_header_for_targets, with_opt_mapped_expr, DelimHeader, MappedExpr,
};
use super::dec_num::DecNum;
use super::expr::{line_matches_expr, PredExpr};
use super::line::{find_line_col_span, parse_canonical_u64, ColTarget};

#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h = (h.rotate_left(5) ^ u64::from(b)).wrapping_mul(0x517cc1b727220a95);
        }
        self.0 = h;
    }
}

type FxBuildHasher = BuildHasherDefault<FxHasher>;

#[derive(Clone, Debug, Default)]
pub(crate) struct MultiAggAcc {
    pub row_count: u64,
    pub cols: Vec<AggStats>,
}

impl MultiAggAcc {
    #[inline]
    pub fn with_cols(n: usize) -> Self {
        Self {
            row_count: 0,
            cols: vec![AggStats::default(); n],
        }
    }

    #[inline]
    pub fn ensure_len(&mut self, n: usize) {
        if self.cols.len() < n {
            self.cols.resize(n, AggStats::default());
        }
    }

    #[inline]
    pub fn merge(&mut self, other: &Self) {
        if other.row_count == 0 && other.cols.is_empty() {
            return;
        }
        self.row_count += other.row_count;
        if self.cols.len() < other.cols.len() {
            self.cols.resize(other.cols.len(), AggStats::default());
        }
        for (dst, src) in self.cols.iter_mut().zip(other.cols.iter()) {
            dst.merge(src);
        }
    }
}

#[inline]
pub(crate) fn agg_multi_line(
    line: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_cols: &[ColTarget<'_>],
    acc: &mut MultiAggAcc,
) {
    if let Some(expr) = where_expr {
        if !line_matches_expr(line, expr) {
            return;
        }
    }
    acc.ensure_len(agg_cols.len());
    acc.row_count += 1;
    for (st, &acol) in acc.cols.iter_mut().zip(agg_cols.iter()) {
        if let Some((s, e)) = find_line_col_span(line, acol) {
            let tok = &line[s..e];
            if let Some(v) = parse_canonical_u64(tok) {
                st.add_num(v);
                continue;
            }
            if let Some(d) = DecNum::parse(tok) {
                st.add_dec(d);
                continue;
            }
        }
        st.add_rows_only(1);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MultiGroupRow {
    pub keys: Vec<GroupKey>,
    pub row_count: u64,
    pub aggs: Vec<AggStats>,
}

#[derive(Default)]
pub(crate) struct MultiGroupMap {
    map: HashMap<Vec<GroupKey>, (u64, Vec<AggStats>), FxBuildHasher>,
}

impl MultiGroupMap {
    #[inline]
    pub fn add_entry(&mut self, keys: Vec<GroupKey>, row_delta: u64, agg_deltas: &[AggStats]) {
        if row_delta == 0 {
            return;
        }
        if let Some((rc, aggs)) = self.map.get_mut(&keys) {
            *rc += row_delta;
            for (dst, src) in aggs.iter_mut().zip(agg_deltas.iter()) {
                dst.merge(src);
            }
        } else {
            self.map.insert(keys, (row_delta, agg_deltas.to_vec()));
        }
    }

    pub fn merge_from(&mut self, other: Self) {
        if self.map.is_empty() {
            self.map = other.map;
            return;
        }
        for (keys, (rc, aggs)) in other.map {
            self.add_entry(keys, rc, &aggs);
        }
    }

    pub fn into_rows(self) -> Vec<MultiGroupRow> {
        let mut out: Vec<MultiGroupRow> = self
            .map
            .into_iter()
            .map(|(keys, (row_count, aggs))| MultiGroupRow {
                keys,
                row_count,
                aggs,
            })
            .collect();
        out.sort_unstable_by(|a, b| a.keys.cmp(&b.keys));
        out
    }
}

#[inline]
pub(crate) fn multi_group_by_line(
    line: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_cols: &[ColTarget<'_>],
    agg_cols: &[ColTarget<'_>],
    out: &mut MultiGroupMap,
) {
    if let Some(expr) = where_expr {
        if !line_matches_expr(line, expr) {
            return;
        }
    }
    let mut keys = Vec::with_capacity(group_cols.len());
    for &gcol in group_cols {
        let Some((gs, ge)) = find_line_col_span(line, gcol) else {
            return;
        };
        let Some((is_str, gkey)) = decode_group_token_ref(gcol, &line[gs..ge]) else {
            return;
        };
        keys.push(if is_str {
            GroupKey::Str(gkey.into_owned())
        } else {
            GroupKey::RawJson(gkey.into_owned())
        });
    }
    let mut aggs = vec![AggStats::default(); agg_cols.len()];
    for (st, &acol) in aggs.iter_mut().zip(agg_cols.iter()) {
        if let Some((as_, ae)) = find_line_col_span(line, acol) {
            let tok = &line[as_..ae];
            if let Some(v) = parse_canonical_u64(tok) {
                st.add_num(v);
                continue;
            }
            if let Some(d) = DecNum::parse(tok) {
                st.add_dec(d);
                continue;
            }
        }
        st.add_rows_only(1);
    }
    out.add_entry(keys, 1, &aggs);
}

fn aggregate_multi_raw(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_cols: &[ColTarget<'_>],
    options: &DecodeOptions,
) -> Result<MultiAggAcc> {
    options.validate()?;
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    let ncols = agg_cols.len();
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        if in_blocks.len() > 1 {
            let mut res = super::par_chunk::parallel_reduce_blocks(
                &in_blocks,
                options,
                |payload, carry, eval_in_carry, acc: &mut MultiAggAcc| {
                    acc.ensure_len(ncols);
                    if let Some(part) = crate::columns::aggregate_multi_grammar_block_chunk(
                        payload,
                        carry,
                        eval_in_carry,
                        where_expr,
                        agg_cols,
                    )? {
                        acc.merge(&part);
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                },
                |line, acc: &mut MultiAggAcc| agg_multi_line(line, where_expr, agg_cols, acc),
                |dst, src| dst.merge(&src),
            )?;
            res.ensure_len(ncols);
            return Ok(res);
        }
    }

    let mut total = MultiAggAcc::with_cols(ncols);
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
                if let Some(part) = crate::columns::aggregate_multi_grammar_block_seq(
                    payload,
                    &mut carry,
                    where_expr,
                    agg_cols,
                )? {
                    total.merge(&part);
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
                        agg_multi_line(&c_bytes, where_expr, agg_cols, &mut total);
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some(line_bytes.to_vec());
                    break;
                }
                agg_multi_line(line_bytes, where_expr, agg_cols, &mut total);
            }
        }
    }
    if let Some(c_bytes) = carry.take() {
        agg_multi_line(&c_bytes, where_expr, agg_cols, &mut total);
    }
    Ok(total)
}

pub(crate) fn aggregate_multi_columns(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_cols: &[ColTarget<'_>],
    options: &DecodeOptions,
) -> Result<MultiAggAcc> {
    if let Some(hdr) = resolve_header_for_targets(frame_bytes, where_expr, agg_cols, options)? {
        let m_cols: Vec<ColTarget<'_>> = agg_cols
            .iter()
            .map(|&c| hdr.resolve_col(c).unwrap_or(c))
            .collect();
        let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
        return with_opt_mapped_expr(m_where.as_ref(), |mw| {
            let mut acc = aggregate_multi_raw(frame_bytes, mw, &m_cols, options)?;
            let mut hdr_acc = MultiAggAcc::with_cols(m_cols.len());
            agg_multi_line(&hdr.header_line, mw, &m_cols, &mut hdr_acc);
            acc.row_count = acc.row_count.saturating_sub(hdr_acc.row_count);
            for (st, hst) in acc.cols.iter_mut().zip(hdr_acc.cols.iter()) {
                st.row_count = st.row_count.saturating_sub(hst.row_count);
            }
            Ok(acc)
        });
    }
    aggregate_multi_raw(frame_bytes, where_expr, agg_cols, options)
}

fn multi_group_by_raw(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_cols: &[ColTarget<'_>],
    agg_cols: &[ColTarget<'_>],
    options: &DecodeOptions,
) -> Result<Vec<MultiGroupRow>> {
    options.validate()?;
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        if in_blocks.len() > 1 {
            let total_map = super::par_chunk::parallel_reduce_blocks(
                &in_blocks,
                options,
                |payload, carry, eval_in_carry, acc: &mut MultiGroupMap| {
                    crate::columns::multi_group_by_grammar_block_chunk(
                        payload,
                        carry,
                        eval_in_carry,
                        where_expr,
                        group_cols,
                        agg_cols,
                        acc,
                    )
                },
                |line, acc: &mut MultiGroupMap| {
                    multi_group_by_line(line, where_expr, group_cols, agg_cols, acc)
                },
                |dst, src| dst.merge_from(src),
            )?;
            return Ok(total_map.into_rows());
        }
    }

    let mut total_map = MultiGroupMap::default();
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
                if crate::columns::multi_group_by_grammar_block_seq(
                    payload,
                    &mut carry,
                    where_expr,
                    group_cols,
                    agg_cols,
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
                        multi_group_by_line(
                            &c_bytes,
                            where_expr,
                            group_cols,
                            agg_cols,
                            &mut total_map,
                        );
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some(line_bytes.to_vec());
                    break;
                }
                multi_group_by_line(
                    line_bytes,
                    where_expr,
                    group_cols,
                    agg_cols,
                    &mut total_map,
                );
            }
        }
    }
    if let Some(c_bytes) = carry.take() {
        multi_group_by_line(&c_bytes, where_expr, group_cols, agg_cols, &mut total_map);
    }
    Ok(total_map.into_rows())
}

pub(crate) fn subtract_header_multi_group(
    hdr: &DelimHeader,
    mapped_where: Option<&PredExpr<'_>>,
    mapped_gcols: &[ColTarget<'_>],
    rows: &mut Vec<MultiGroupRow>,
) {
    if !mapped_where.is_none_or(|w| line_matches_expr(&hdr.header_line, w)) {
        return;
    }
    let mut target_keys = Vec::with_capacity(mapped_gcols.len());
    for &gcol in mapped_gcols {
        let Some((gs, ge)) = find_line_col_span(&hdr.header_line, gcol) else {
            return;
        };
        let Some((is_str, gkey)) = decode_group_token_ref(gcol, &hdr.header_line[gs..ge]) else {
            return;
        };
        target_keys.push(if is_str {
            GroupKey::Str(gkey.into_owned())
        } else {
            GroupKey::RawJson(gkey.into_owned())
        });
    }
    if let Ok(pos) = rows.binary_search_by(|r| r.keys.cmp(&target_keys)) {
        rows[pos].row_count = rows[pos].row_count.saturating_sub(1);
        for st in &mut rows[pos].aggs {
            st.row_count = st.row_count.saturating_sub(1);
        }
        if rows[pos].row_count == 0 {
            rows.remove(pos);
        }
    }
}

pub(crate) fn multi_group_by(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_cols: &[ColTarget<'_>],
    agg_cols: &[ColTarget<'_>],
    options: &DecodeOptions,
) -> Result<Vec<MultiGroupRow>> {
    let mut extras = Vec::with_capacity(group_cols.len() + agg_cols.len());
    extras.extend_from_slice(group_cols);
    extras.extend_from_slice(agg_cols);
    if let Some(hdr) = resolve_header_for_targets(frame_bytes, where_expr, &extras, options)? {
        let m_gcols: Vec<ColTarget<'_>> = group_cols
            .iter()
            .map(|&c| hdr.resolve_col(c).unwrap_or(c))
            .collect();
        let m_acols: Vec<ColTarget<'_>> = agg_cols
            .iter()
            .map(|&c| hdr.resolve_col(c).unwrap_or(c))
            .collect();
        let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
        return with_opt_mapped_expr(m_where.as_ref(), |mw| {
            let mut rows = multi_group_by_raw(frame_bytes, mw, &m_gcols, &m_acols, options)?;
            subtract_header_multi_group(&hdr, mw, &m_gcols, &mut rows);
            Ok(rows)
        });
    }
    multi_group_by_raw(frame_bytes, where_expr, group_cols, agg_cols, options)
}
