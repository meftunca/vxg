//! Bounded-memory multi-frame streaming compressed-domain execution (`FrameReader`,
//! `verify_frame`, `verify_stream`, and `*_stream` query/analytics/projection/mutation APIs)
//! with exact cross-frame split-line stitching.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::{
    ops::{
        csv_header::{
            resolve_header_for_targets, subtract_header_agg, subtract_header_group_by,
            with_opt_mapped_expr, DelimHeader, MappedExpr,
        },
        line_matches_expr, validate_projection_cols, AggStats, ColTarget, GroupKey, GroupRow,
        PredExpr,
    },
    DecodeOptions, Error, Result,
};

mod peeler;
mod top_k;
mod transform;
mod verify;

pub(crate) use peeler::{write_all_vxg, FramePeeler};
pub use top_k::{select_top_k_by_stream, top_k_by_stream};
pub use transform::{
    filter_frame_where_stream, map_column_stream, project_frame_stream, update_column_stream,
};
pub use verify::{verify_frame, verify_stream, FrameReader};

/// Counts total structured records across a `VXG1` stream (`R: Read`) with bounded memory
/// and exact cross-frame split-line stitching.
pub fn frame_row_count_stream<R: Read>(reader: R, dec_options: &DecodeOptions) -> Result<u64> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = 0u64;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            total = total
                .checked_add(crate::frame_row_count(eff)? as u64)
                .ok_or(Error::SizeOverflow)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        total = total
            .checked_add(crate::frame_row_count(tail_eff)? as u64)
            .ok_or(Error::SizeOverflow)?;
    }
    Ok(total)
}

fn count_step<'e>(
    eff: &[u8],
    expr: &PredExpr<'e>,
    sub_dec: &DecodeOptions,
    first: &mut bool,
    m_opt: &mut Option<MappedExpr<'e>>,
) -> Result<u64> {
    if !*first {
        *first = true;
        if let Some(hdr) = resolve_header_for_targets(eff, Some(expr), &[], sub_dec)? {
            let m = MappedExpr::from_expr(expr, &hdr);
            let c = m.with_pred_expr(|me| {
                let cnt = crate::ops::expr::count_where_impl(eff, me, sub_dec)?;
                Ok::<u64, Error>(cnt.saturating_sub(u64::from(line_matches_expr(&hdr.header_line, me))))
            })?;
            *m_opt = Some(m);
            return Ok(c);
        }
    }
    if let Some(ref m) = *m_opt {
        m.with_pred_expr(|me| crate::ops::expr::count_where_impl(eff, me, sub_dec))
    } else {
        crate::ops::expr::count_where_impl(eff, expr, sub_dec)
    }
}

/// Counts records matching compound `expr` across a `VXG1` stream (`R: Read`) with bounded memory.
pub fn count_where_stream<R: Read>(
    reader: R,
    expr: &PredExpr<'_>,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = 0u64;
    let mut first_seen = false;
    let mut mapped_opt = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            total = total
                .checked_add(count_step(eff, expr, &sub_dec, &mut first_seen, &mut mapped_opt)?)
                .ok_or(Error::SizeOverflow)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        total = total
            .checked_add(count_step(
                tail_eff,
                expr,
                &sub_dec,
                &mut first_seen,
                &mut mapped_opt,
            )?)
            .ok_or(Error::SizeOverflow)?;
    }
    Ok(total)
}

fn filter_step<'e>(
    eff: &[u8],
    expr: &PredExpr<'e>,
    rem_n: usize,
    sub_dec: &DecodeOptions,
    first: &mut bool,
    m_opt: &mut Option<MappedExpr<'e>>,
) -> Result<Vec<crate::RowMatch>> {
    if !*first {
        *first = true;
        if let Some(hdr) = resolve_header_for_targets(eff, Some(expr), &[], sub_dec)? {
            let m = MappedExpr::from_expr(expr, &hdr);
            let rows = m.with_pred_expr(|me| {
                if !line_matches_expr(&hdr.header_line, me) {
                    return crate::ops::expr::filter_where_impl(eff, me, rem_n, sub_dec);
                }
                let mut rs = crate::ops::expr::filter_where_impl(
                    eff,
                    me,
                    rem_n.saturating_add(1),
                    sub_dec,
                )?;
                rs.retain(|r| r.global_row_index != 0);
                rs.truncate(rem_n);
                Ok(rs)
            })?;
            *m_opt = Some(m);
            return Ok(rows);
        }
    }
    if let Some(ref m) = *m_opt {
        m.with_pred_expr(|me| crate::ops::expr::filter_where_impl(eff, me, rem_n, sub_dec))
    } else {
        crate::ops::expr::filter_where_impl(eff, expr, rem_n, sub_dec)
    }
}

/// Streams records matching compound `expr` from `reader` directly into `writer` with bounded
/// memory, stopping early as soon as `limit` matching records have been emitted.
/// Returns the number of matching records written.
pub fn filter_where_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    expr: &PredExpr<'_>,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    dec_options.validate()?;
    if limit == 0 {
        writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
        return Ok(0);
    }
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut rem = limit;
    let mut written_rows = 0u64;
    let mut first_seen = false;
    let mut mapped_opt = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let matches = filter_step(eff, expr, rem, &sub_dec, &mut first_seen, &mut mapped_opt)?;
            for m in &matches {
                write_all_vxg(&mut writer, &m.bytes)?;
            }
            written_rows += matches.len() as u64;
            rem = rem.saturating_sub(matches.len());
            if rem == 0 {
                writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
                return Ok(written_rows);
            }
        }
    }
    if rem > 0 {
        if let Some(tail_eff) = peeler.finish_tail()? {
            let matches =
                filter_step(tail_eff, expr, rem, &sub_dec, &mut first_seen, &mut mapped_opt)?;
            for m in &matches {
                write_all_vxg(&mut writer, &m.bytes)?;
            }
            written_rows += matches.len() as u64;
        }
    }
    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
    Ok(written_rows)
}

fn select_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    cols: &[ColTarget<'e>],
    rem_n: usize,
    sub_dec: &DecodeOptions,
    first: &mut bool,
    st: &mut Option<(Option<MappedExpr<'e>>, Vec<ColTarget<'e>>)>,
) -> Result<Vec<u8>> {
    if !*first {
        *first = true;
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, cols, sub_dec)? {
            let m_cols: Vec<ColTarget<'e>> = cols
                .iter()
                .map(|&c| hdr.resolve_col(c).unwrap_or(c))
                .collect();
            let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            let out = with_opt_mapped_expr(m_where.as_ref(), |mw| {
                if !mw.is_none_or(|w| line_matches_expr(&hdr.header_line, w)) {
                    return crate::ops::project::select_columns_impl(eff, mw, &m_cols, rem_n, sub_dec);
                }
                let mut b = crate::ops::project::select_columns_impl(
                    eff,
                    mw,
                    &m_cols,
                    rem_n.saturating_add(1),
                    sub_dec,
                )?;
                if let Some(nl) = b.iter().position(|&ch| ch == b'\n') {
                    b.drain(..=nl);
                } else {
                    b.clear();
                }
                Ok(b)
            })?;
            *st = Some((m_where, m_cols));
            return Ok(out);
        }
    }
    if let Some((ref m_where, ref m_cols)) = *st {
        with_opt_mapped_expr(m_where.as_ref(), |mw| {
            crate::ops::project::select_columns_impl(eff, mw, m_cols, rem_n, sub_dec)
        })
    } else {
        crate::ops::project::select_columns_impl(eff, where_expr, cols, rem_n, sub_dec)
    }
}

/// Streams projected columns (`SELECT cols WHERE ...`) from `reader` directly into `writer`
/// with bounded memory, stopping early as soon as `limit` records have been emitted.
/// Returns the number of projected records written.
pub fn select_columns_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    dec_options.validate()?;
    validate_projection_cols(cols)?;
    if limit == 0 {
        writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
        return Ok(0);
    }
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut rem = limit;
    let mut written_rows = 0u64;
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let proj = select_step(
                eff,
                where_expr,
                cols,
                rem,
                &sub_dec,
                &mut first_seen,
                &mut mapped_state,
            )?;
            if !proj.is_empty() {
                let rows = proj.split_inclusive(|&b| b == b'\n').count();
                write_all_vxg(&mut writer, &proj)?;
                written_rows += rows as u64;
                rem = rem.saturating_sub(rows);
                if rem == 0 {
                    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
                    return Ok(written_rows);
                }
            }
        }
    }
    if rem > 0 {
        if let Some(tail_eff) = peeler.finish_tail()? {
            let proj = select_step(
                tail_eff,
                where_expr,
                cols,
                rem,
                &sub_dec,
                &mut first_seen,
                &mut mapped_state,
            )?;
            if !proj.is_empty() {
                let rows = proj.split_inclusive(|&b| b == b'\n').count();
                write_all_vxg(&mut writer, &proj)?;
                written_rows += rows as u64;
            }
        }
    }
    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
    Ok(written_rows)
}

fn agg_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    col: ColTarget<'e>,
    sub_dec: &DecodeOptions,
    first: &mut bool,
    st: &mut Option<(Option<MappedExpr<'e>>, ColTarget<'e>)>,
) -> Result<AggStats> {
    if !*first {
        *first = true;
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, &[col], sub_dec)? {
            let m_col = hdr.resolve_col(col).unwrap_or(col);
            let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            let part = with_opt_mapped_expr(m_where.as_ref(), |mw| {
                let mut s = crate::ops::agg::aggregate_column_impl(eff, mw, m_col, sub_dec)?;
                subtract_header_agg(&hdr, mw, m_col, &mut s);
                Ok::<AggStats, Error>(s)
            })?;
            *st = Some((m_where, m_col));
            return Ok(part);
        }
    }
    if let Some((ref m_where, m_col)) = *st {
        with_opt_mapped_expr(m_where.as_ref(), |mw| {
            crate::ops::agg::aggregate_column_impl(eff, mw, m_col, sub_dec)
        })
    } else {
        crate::ops::agg::aggregate_column_impl(eff, where_expr, col, sub_dec)
    }
}

/// Computes exact `AggStats` (`row_count`, `count`, `sum`, `min`, `max`, `avg`) across an
/// arbitrarily large `VXG1` stream (`R: Read`) in `O(1)` memory.
pub fn aggregate_column_stream<R: Read>(
    reader: R,
    where_expr: Option<&PredExpr<'_>>,
    col: ColTarget<'_>,
    dec_options: &DecodeOptions,
) -> Result<AggStats> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut total = AggStats::default();
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let part = agg_step(
                eff,
                where_expr,
                col,
                &sub_dec,
                &mut first_seen,
                &mut mapped_state,
            )?;
            total.merge(&part);
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let part = agg_step(
            tail_eff,
            where_expr,
            col,
            &sub_dec,
            &mut first_seen,
            &mut mapped_state,
        )?;
        total.merge(&part);
    }
    Ok(total)
}

type GrpStreamState<'e> = (
    Option<MappedExpr<'e>>,
    ColTarget<'e>,
    Option<ColTarget<'e>>,
    DelimHeader,
);

fn grp_step<'e>(
    eff: &[u8],
    where_expr: Option<&PredExpr<'e>>,
    group_col: ColTarget<'e>,
    agg_col: Option<ColTarget<'e>>,
    sub_dec: &DecodeOptions,
    first: &mut bool,
    st: &mut Option<GrpStreamState<'e>>,
) -> Result<Vec<GroupRow>> {
    if !*first {
        *first = true;
        let mut extras = [group_col, group_col];
        let n_ext = if let Some(ac) = agg_col {
            extras[1] = ac;
            2
        } else {
            1
        };
        if let Some(hdr) = resolve_header_for_targets(eff, where_expr, &extras[..n_ext], sub_dec)? {
            let m_gcol = hdr.resolve_col(group_col).unwrap_or(group_col);
            let m_acol = agg_col.map(|ac| hdr.resolve_col(ac).unwrap_or(ac));
            let m_where = where_expr.map(|w| MappedExpr::from_expr(w, &hdr));
            let rows = with_opt_mapped_expr(m_where.as_ref(), |mw| {
                let mut rs = crate::ops::agg::group_by_impl(eff, mw, m_gcol, m_acol, sub_dec)?;
                subtract_header_group_by(&hdr, mw, m_gcol, &mut rs);
                Ok::<Vec<GroupRow>, Error>(rs)
            })?;
            *st = Some((m_where, m_gcol, m_acol, hdr));
            return Ok(rows);
        }
    }
    if let Some((ref m_where, m_gcol, m_acol, _)) = *st {
        with_opt_mapped_expr(m_where.as_ref(), |mw| {
            crate::ops::agg::group_by_impl(eff, mw, m_gcol, m_acol, sub_dec)
        })
    } else {
        crate::ops::agg::group_by_impl(eff, where_expr, group_col, agg_col, sub_dec)
    }
}

/// Computes exact `GROUP BY` aggregations across an arbitrarily large `VXG1` stream (`R: Read`)
/// in `O(|distinct_groups|)` memory.
pub fn group_by_stream<R: Read>(
    reader: R,
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
    dec_options: &DecodeOptions,
) -> Result<Vec<GroupRow>> {
    dec_options.validate()?;
    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;
    let mut merged: BTreeMap<GroupKey, AggStats> = BTreeMap::new();
    let mut first_seen = false;
    let mut mapped_state = None;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let rows = grp_step(
                eff,
                where_expr,
                group_col,
                agg_col,
                &sub_dec,
                &mut first_seen,
                &mut mapped_state,
            )?;
            for r in rows {
                merged.entry(r.key).or_default().merge(&r.stats);
            }
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let rows = grp_step(
            tail_eff,
            where_expr,
            group_col,
            agg_col,
            &sub_dec,
            &mut first_seen,
            &mut mapped_state,
        )?;
        for r in rows {
            merged.entry(r.key).or_default().merge(&r.stats);
        }
    }
    Ok(merged
        .into_iter()
        .map(|(key, stats)| GroupRow { key, stats })
        .collect())
}
