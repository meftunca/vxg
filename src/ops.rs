//! Compressed-domain structured operations, boolean predicates, aggregations, projection, and Top-K.

use std::borrow::Cow;
use std::cell::UnsafeCell;

use crate::{frame, query, DecodeOptions, Frame, Options, Result, RowMatch, BLOCK_COLUMN};

pub(crate) mod agg;
pub(crate) mod emit;
pub(crate) mod expr;
mod filter_col;
mod filter_frame;
pub(crate) mod line;
mod parse_expr;
pub(crate) mod project;
pub(crate) mod project_line;
pub(crate) mod top_k;
mod transform;

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

pub(super) struct SyncCell<T>(UnsafeCell<T>);

// SAFETY: Each index in a `Vec<SyncCell<T>>` is written at most once by the unique worker
// thread that claims block `b` via atomic `fetch_add`, and read only after `std::thread::scope` joins.
unsafe impl<T: Send> Sync for SyncCell<T> {}

impl<T> SyncCell<T> {
    #[inline]
    pub(super) const fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }

    #[inline]
    pub(super) fn get(&self) -> *mut T {
        self.0.get()
    }

    #[inline]
    pub(super) fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

pub(super) struct ParInBlock<'a> {
    pub(super) f_bytes: &'a [u8],
    pub(super) local_blk: usize,
    pub(super) info: frame::BlockInfo,
    pub(super) payload: &'a [u8],
}

pub(super) struct ParMeta<'a> {
    pub(super) decoded_raw: Option<Vec<u8>>,
    pub(super) row_count: usize,
    pub(super) had_in_carry: bool,
    pub(super) cur_global: usize,
    pub(super) first_frag: Option<Cow<'a, [u8]>>,
    pub(super) last_unterm_frag: Option<Cow<'a, [u8]>>,
}

pub(super) struct BoundaryLine {
    pub(super) block_index: usize,
    pub(super) row_index: usize,
    pub(super) global_row_index: usize,
    pub(super) bytes: Vec<u8>,
}

pub(super) fn collect_par_blocks<'a>(
    frame_bytes: &'a [u8],
    frames: &[Frame<'a>],
) -> Vec<ParInBlock<'a>> {
    let mut in_blocks = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            in_blocks.push(ParInBlock {
                f_bytes,
                local_blk,
                info,
                payload: &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
            });
        }
    }
    in_blocks
}

pub(super) fn build_par_meta_and_boundary<'a>(
    in_blocks: &[ParInBlock<'a>],
    options: &DecodeOptions,
) -> Result<(Vec<ParMeta<'a>>, Vec<BoundaryLine>)> {
    let nb = in_blocks.len();
    let mut meta: Vec<ParMeta<'a>> = Vec::with_capacity(nb);
    let mut non_empty = Vec::with_capacity(nb);
    let mut incoming = false;
    let mut any_unterm = false;
    let mut cur_global = 0usize;

    for (b, ib) in in_blocks.iter().enumerate() {
        let had_in = incoming;
        let cg = cur_global;
        if ib.info.kind == BLOCK_COLUMN {
            if let Some(gb) = crate::columns::peek_grammar_boundary_fast(ib.payload, had_in)? {
                if gb.row_count > 0 {
                    cur_global += gb.row_count - usize::from(had_in);
                    incoming = gb.last_unterminated_raw.is_some();
                    any_unterm |= incoming;
                    non_empty.push(b);
                }
                meta.push(ParMeta {
                    decoded_raw: None,
                    row_count: gb.row_count,
                    had_in_carry: had_in,
                    cur_global: cg,
                    first_frag: gb.first_raw,
                    last_unterm_frag: gb.last_unterminated_raw,
                });
                continue;
            }
        }
        let mut raw_vec = Vec::new();
        query::decode_block_into(
            ib.f_bytes,
            ib.local_blk,
            ib.info.raw_size,
            options,
            &mut raw_vec,
        )?;
        let rc = raw_vec.split_inclusive(|&c| c == b'\n').count();
        let ends_nl = raw_vec.last() == Some(&b'\n');
        let first_frag = if had_in && !raw_vec.is_empty() {
            let end = raw_vec
                .iter()
                .position(|&c| c == b'\n')
                .map_or(raw_vec.len(), |i| i + 1);
            Some(Cow::Owned(raw_vec[..end].to_vec()))
        } else {
            None
        };
        let last_unterm_frag = if !raw_vec.is_empty() && !ends_nl {
            let st = raw_vec
                .iter()
                .rposition(|&c| c == b'\n')
                .map_or(0, |i| i + 1);
            Some(Cow::Owned(raw_vec[st..].to_vec()))
        } else {
            None
        };
        if rc > 0 {
            cur_global += rc - usize::from(had_in);
            incoming = !ends_nl;
            any_unterm |= incoming;
            non_empty.push(b);
        }
        meta.push(ParMeta {
            decoded_raw: Some(raw_vec),
            row_count: rc,
            had_in_carry: had_in,
            cur_global: cg,
            first_frag,
            last_unterm_frag,
        });
    }

    let mut boundary_lines = Vec::new();
    if any_unterm {
        let mut k_idx = 0usize;
        while k_idx < non_empty.len() {
            let b0 = non_empty[k_idx];
            let Some(ref f0) = meta[b0].last_unterm_frag else {
                k_idx += 1;
                continue;
            };
            let base_b0 = meta[b0].cur_global - usize::from(meta[b0].had_in_carry);
            let start_row = meta[b0].row_count - 1;
            let start_glob = base_b0 + start_row;
            let mut full = f0.as_ref().to_vec();
            let mut m = k_idx + 1;
            while m < non_empty.len() {
                let bm = non_empty[m];
                let Some(ref fm) = meta[bm].first_frag else {
                    break;
                };
                full.extend_from_slice(fm);
                if meta[bm].row_count == 1 && meta[bm].last_unterm_frag.is_some() {
                    m += 1;
                } else {
                    break;
                }
            }
            boundary_lines.push(BoundaryLine {
                block_index: b0,
                row_index: start_row,
                global_row_index: start_glob,
                bytes: full,
            });
            k_idx = m.max(k_idx + 1);
        }
    }
    Ok((meta, boundary_lines))
}

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
