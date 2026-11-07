//! Compressed-domain structured operations, boolean predicates, aggregations, projection, and Top-K.

use crate::{DecodeOptions, Options, Result, RowMatch};

pub(crate) mod agg;
pub(crate) mod emit;
pub(crate) mod expr;
mod filter_col;
mod filter_frame;
pub(crate) mod line;
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
    top_k::top_k_by_impl(frame_bytes, where_expr, order_col, dir, limit, dec_options)
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
    top_k::select_top_k_by_impl(
        frame_bytes,
        where_expr,
        cols,
        order_col,
        dir,
        limit,
        dec_options,
    )
}
