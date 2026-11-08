//! Compressed-domain structured operations, boolean predicates, aggregations, projection, and Top-K.

use crate::{DecodeOptions, Options, Result, RowMatch};

pub(crate) mod agg;
pub(crate) mod csv_header;
pub(crate) mod dec_num;
pub(crate) mod emit;
pub(crate) mod expr;
mod filter_col;
mod filter_frame;
pub(crate) mod line;
pub(crate) mod multi_agg;
pub(crate) mod par_chunk;
pub(crate) mod par_meta;
mod parse_expr;
pub(crate) mod project;
pub(crate) mod project_line;
pub(crate) mod top_k;
mod transform;

pub(super) use par_meta::{
    build_boundary_set, build_par_meta_and_boundary, collect_par_blocks, inspect_par_meta,
    ParInBlock, ParMeta, SyncCell,
};

pub use agg::{agg_line, decode_group_token_ref, AggStats, GroupKey, GroupRow};
pub use dec_num::{format_scaled_i128, DecCmp, DecNum};
pub use expr::{line_matches_expr, PredExpr};
pub(crate) use line::parse_canonical_u64;
pub use line::{
    apply_line_col_op, apply_token_op, decode_delim_str, encode_delim_str, encode_json_str,
    eval_token_pred, find_delimited_col_span, find_json_key_span, find_line_col_span,
    format_col_str, line_matches_col_pred, unescape_json_str, ColOp, ColPred, ColTarget, StrCmp,
    ValuePred,
};
pub use parse_expr::{
    parse_col_list, parse_col_target, with_opt_where, ParsedColPred, ParsedWhere,
};
pub use project_line::project_line;
pub(crate) use project_line::validate_projection_cols;
pub use top_k::{extract_line_sort_key, SortDir, SortKey};

/// Find up to `limit` records in `frame_bytes` satisfying column predicate `pred`
/// directly in the compressed domain.
pub fn filter_column(
    frame_bytes: &[u8],
    pred: &ColPred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, None, &[pred.col], options)?
    {
        let mp = ColPred {
            col: hdr.resolve_col(pred.col).unwrap_or(pred.col),
            pred: pred.pred,
        };
        if !line_matches_col_pred(&hdr.header_line, &mp) {
            return filter_col::filter_column_impl(frame_bytes, &mp, limit, options);
        }
        if limit == 0 {
            return filter_col::filter_column_impl(frame_bytes, &mp, 0, options);
        }
        let mut rows =
            filter_col::filter_column_impl(frame_bytes, &mp, limit.saturating_add(1), options)?;
        rows.retain(|r| r.global_row_index != 0);
        rows.truncate(limit);
        return Ok(rows);
    }
    filter_col::filter_column_impl(frame_bytes, pred, limit, options)
}

/// Produce a new canonical VXG frame containing only records satisfying `pred`,
/// skipping non-matching blocks and columns in the compressed domain.
pub fn filter_frame(
    frame_bytes: &[u8],
    pred: &ColPred<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, None, &[pred.col], dec_options)?
    {
        let mp = ColPred {
            col: hdr.resolve_col(pred.col).unwrap_or(pred.col),
            pred: pred.pred,
        };
        if line_matches_col_pred(&hdr.header_line, &mp) {
            let stripped = csv_header::strip_delim_header_bytes(frame_bytes, &hdr, dec_options)?;
            return filter_frame::filter_frame_impl(&stripped, &mp, enc_options, dec_options);
        }
        return filter_frame::filter_frame_impl(frame_bytes, &mp, enc_options, dec_options);
    }
    filter_frame::filter_frame_impl(frame_bytes, pred, enc_options, dec_options)
}

/// Conditionally transform `target_col` with `op` on every record matching `where_pred`,
/// copying untouched blocks byte-for-byte and splicing only `target_col` on modified
/// `GRAMMAR_PACKET` blocks.
pub fn update_column(
    frame_bytes: &[u8],
    where_pred: &ColPred<'_>,
    target_col: ColTarget<'_>,
    op: ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    transform::transform_frame(
        frame_bytes,
        Some(where_pred),
        target_col,
        &op,
        enc_options,
        dec_options,
    )
}

/// Transform `target_col` with `op` across every record in `frame_bytes`, splicing only
/// `target_col` on `GRAMMAR_PACKET` blocks without decoding or re-encoding untouched columns.
pub fn map_column(
    frame_bytes: &[u8],
    target_col: ColTarget<'_>,
    op: ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    transform::transform_frame(frame_bytes, None, target_col, &op, enc_options, dec_options)
}

/// Count records in `frame_bytes` matching compound boolean `expr` directly in the
/// compressed domain without materializing record bytes.
pub fn count_where(
    frame_bytes: &[u8],
    expr: &PredExpr<'_>,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, Some(expr), &[], dec_options)?
    {
        let mapped = csv_header::MappedExpr::from_expr(expr, &hdr);
        return mapped.with_pred_expr(|me| {
            let cnt = expr::count_where_impl(frame_bytes, me, dec_options)?;
            Ok(cnt.saturating_sub(u64::from(line_matches_expr(&hdr.header_line, me))))
        });
    }
    expr::count_where_impl(frame_bytes, expr, dec_options)
}

/// Find up to `limit` records in `frame_bytes` matching compound boolean `expr`,
/// pruning blocks and columns in the compressed domain via short-circuit bitset evaluation.
pub fn filter_where(
    frame_bytes: &[u8],
    expr: &PredExpr<'_>,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, Some(expr), &[], dec_options)?
    {
        let mapped = csv_header::MappedExpr::from_expr(expr, &hdr);
        return mapped.with_pred_expr(|me| {
            if !line_matches_expr(&hdr.header_line, me) || limit == 0 {
                return expr::filter_where_impl(frame_bytes, me, limit, dec_options);
            }
            let mut rows =
                expr::filter_where_impl(frame_bytes, me, limit.saturating_add(1), dec_options)?;
            rows.retain(|r| r.global_row_index != 0);
            rows.truncate(limit);
            Ok(rows)
        });
    }
    expr::filter_where_impl(frame_bytes, expr, limit, dec_options)
}

/// Produce a new canonical VXG frame containing only records satisfying compound `expr`,
/// passing 100%-matching compressed blocks through zero-copy and skipping 0%-matching blocks.
pub fn filter_frame_where(
    frame_bytes: &[u8],
    expr: &PredExpr<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, Some(expr), &[], dec_options)?
    {
        let mapped = csv_header::MappedExpr::from_expr(expr, &hdr);
        return mapped.with_pred_expr(|me| {
            if line_matches_expr(&hdr.header_line, me) {
                let stripped =
                    csv_header::strip_delim_header_bytes(frame_bytes, &hdr, dec_options)?;
                return filter_frame::filter_frame_expr_impl(&stripped, me, enc_options, dec_options);
            }
            filter_frame::filter_frame_expr_impl(frame_bytes, me, enc_options, dec_options)
        });
    }
    filter_frame::filter_frame_expr_impl(frame_bytes, expr, enc_options, dec_options)
}

/// Compute `COUNT(*)`, `COUNT(agg_col)`, `SUM(agg_col)`, `MIN(agg_col)`, and `MAX(agg_col)`
/// over records satisfying `where_expr` directly on compressed binary/dictionary columns.
pub fn aggregate_column(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    agg_col: ColTarget<'_>,
    dec_options: &DecodeOptions,
) -> Result<AggStats> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, where_expr, &[agg_col], dec_options)?
    {
        let m_col = hdr.resolve_col(agg_col).unwrap_or(agg_col);
        let m_where = where_expr.map(|w| csv_header::MappedExpr::from_expr(w, &hdr));
        return csv_header::with_opt_mapped_expr(m_where.as_ref(), |mw| {
            let mut st = agg::aggregate_column_impl(frame_bytes, mw, m_col, dec_options)?;
            csv_header::subtract_header_agg(&hdr, mw, m_col, &mut st);
            Ok(st)
        });
    }
    agg::aggregate_column_impl(frame_bytes, where_expr, agg_col, dec_options)
}

/// Group records satisfying `where_expr` by `group_col` and optionally aggregate `agg_col`
/// directly in the compressed domain, returning deterministically sorted [`GroupRow`]s.
pub fn group_by(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
    dec_options: &DecodeOptions,
) -> Result<Vec<GroupRow>> {
    let mut extras = [group_col, group_col];
    let n_ext = if let Some(ac) = agg_col {
        extras[1] = ac;
        2
    } else {
        1
    };
    if let Some(hdr) = csv_header::resolve_header_for_targets(
        frame_bytes,
        where_expr,
        &extras[..n_ext],
        dec_options,
    )? {
        let m_gcol = hdr.resolve_col(group_col).unwrap_or(group_col);
        let m_acol = agg_col.map(|ac| hdr.resolve_col(ac).unwrap_or(ac));
        let m_where = where_expr.map(|w| csv_header::MappedExpr::from_expr(w, &hdr));
        return csv_header::with_opt_mapped_expr(m_where.as_ref(), |mw| {
            let mut rows = agg::group_by_impl(frame_bytes, mw, m_gcol, m_acol, dec_options)?;
            csv_header::subtract_header_group_by(&hdr, mw, m_gcol, &mut rows);
            Ok(rows)
        });
    }
    agg::group_by_impl(frame_bytes, where_expr, group_col, agg_col, dec_options)
}

/// Project `cols` across up to `limit` records matching `where_expr` in the compressed domain.
pub fn select_columns(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, where_expr, cols, dec_options)?
    {
        let m_cols: Vec<ColTarget<'_>> = cols
            .iter()
            .map(|&c| hdr.resolve_col(c).unwrap_or(c))
            .collect();
        let m_where = where_expr.map(|w| csv_header::MappedExpr::from_expr(w, &hdr));
        return csv_header::with_opt_mapped_expr(m_where.as_ref(), |mw| {
            let hdr_matches = mw.is_none_or(|w| line_matches_expr(&hdr.header_line, w));
            if !hdr_matches || limit == 0 {
                return project::select_columns_impl(frame_bytes, mw, &m_cols, limit, dec_options);
            }
            let mut out = project::select_columns_impl(
                frame_bytes,
                mw,
                &m_cols,
                limit.saturating_add(1),
                dec_options,
            )?;
            if let Some(nl) = out.iter().position(|&b| b == b'\n') {
                out.drain(..=nl);
            } else {
                out.clear();
            }
            Ok(out)
        });
    }
    project::select_columns_impl(frame_bytes, where_expr, cols, limit, dec_options)
}

/// Produce a canonical VXG frame containing only projected `cols` matching `where_expr`.
pub fn project_frame(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, where_expr, cols, dec_options)?
    {
        let m_cols: Vec<ColTarget<'_>> = cols
            .iter()
            .map(|&c| hdr.resolve_col(c).unwrap_or(c))
            .collect();
        let m_where = where_expr.map(|w| csv_header::MappedExpr::from_expr(w, &hdr));
        return csv_header::with_opt_mapped_expr(m_where.as_ref(), |mw| {
            if mw.is_some_and(|w| line_matches_expr(&hdr.header_line, w)) {
                let stripped =
                    csv_header::strip_delim_header_bytes(frame_bytes, &hdr, dec_options)?;
                return project::project_frame_impl(
                    &stripped,
                    mw,
                    &m_cols,
                    enc_options,
                    dec_options,
                );
            }
            project::project_frame_impl(frame_bytes, mw, &m_cols, enc_options, dec_options)
        });
    }
    project::project_frame_impl(frame_bytes, where_expr, cols, enc_options, dec_options)
}

/// Find the global `Top-K` (`ORDER BY order_col [ASC|DESC] LIMIT limit`) records satisfying
/// `where_expr` directly in the compressed domain, skipping dominated numeric/dictionary blocks
/// via header zone-maps and unpacking unselected columns only on the `<= limit` winning rows.
pub fn top_k_by(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    order_col: ColTarget<'_>,
    dir: SortDir,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, where_expr, &[order_col], dec_options)?
    {
        let m_ocol = hdr.resolve_col(order_col).unwrap_or(order_col);
        let m_where = where_expr.map(|w| csv_header::MappedExpr::from_expr(w, &hdr));
        return csv_header::with_opt_mapped_expr(m_where.as_ref(), |mw| {
            top_k::top_k_by_impl(frame_bytes, mw, m_ocol, dir, limit, true, dec_options)
        });
    }
    top_k::top_k_by_impl(frame_bytes, where_expr, order_col, dir, limit, false, dec_options)
}

/// Project `cols` across the global `Top-K` (`SELECT cols WHERE ... ORDER BY order_col [ASC|DESC] LIMIT limit`)
/// records in the compressed domain (`order_col` need not appear in `cols`).
pub fn select_top_k_by(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    order_col: ColTarget<'_>,
    dir: SortDir,
    limit: usize,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let mut extras = Vec::with_capacity(cols.len() + 1);
    extras.extend_from_slice(cols);
    extras.push(order_col);
    if let Some(hdr) =
        csv_header::resolve_header_for_targets(frame_bytes, where_expr, &extras, dec_options)?
    {
        let m_cols: Vec<ColTarget<'_>> = cols
            .iter()
            .map(|&c| hdr.resolve_col(c).unwrap_or(c))
            .collect();
        let m_ocol = hdr.resolve_col(order_col).unwrap_or(order_col);
        let m_where = where_expr.map(|w| csv_header::MappedExpr::from_expr(w, &hdr));
        return csv_header::with_opt_mapped_expr(m_where.as_ref(), |mw| {
            top_k::select_top_k_by_impl(
                frame_bytes,
                mw,
                &m_cols,
                m_ocol,
                dir,
                limit,
                true,
                dec_options,
            )
        });
    }
    top_k::select_top_k_by_impl(
        frame_bytes,
        where_expr,
        cols,
        order_col,
        dir,
        limit,
        false,
        dec_options,
    )
}
