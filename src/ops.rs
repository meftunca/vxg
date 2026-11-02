//! Compressed-domain structured operations, boolean predicates, aggregations, and projection.

use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{frame, query, DecodeOptions, Frame, Options, Result, RowMatch, BLOCK_COLUMN};

pub(crate) mod agg;
mod emit;
pub(crate) mod expr;
mod filter_frame;
pub(crate) mod line;
pub(crate) mod project;
pub(crate) mod project_line;
mod transform;

pub(crate) use line::parse_canonical_u64;
pub use agg::{agg_line, decode_group_token_ref, AggStats, GroupKey, GroupRow};
pub use expr::{line_matches_expr, PredExpr};
pub use project_line::project_line;
pub use line::{
    apply_line_col_op, apply_token_op, decode_delim_str, encode_delim_str, encode_json_str,
    eval_token_pred, find_delimited_col_span, find_json_key_span, find_line_col_span,
    format_col_str, line_matches_col_pred, unescape_json_str, ColOp, ColPred, ColTarget, StrCmp,
    ValuePred,
};

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

fn filter_column_parallel(
    in_blocks: &[ParInBlock<'_>],
    pred: &ColPred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    let nb = in_blocks.len();
    let (meta, boundary_lines) = build_par_meta_and_boundary(in_blocks, options)?;
    let mut boundary_matches = Vec::new();
    for bl in boundary_lines {
        if line_matches_col_pred(&bl.bytes, pred) {
            boundary_matches.push(RowMatch {
                block_index: bl.block_index,
                row_index: bl.row_index,
                global_row_index: bl.global_row_index,
                bytes: bl.bytes,
            });
        }
    }

    let workers = options.threads.min(nb.max(1));
    let next = AtomicUsize::new(0);
    let cutoff = AtomicUsize::new(nb);
    let failed = AtomicBool::new(false);
    let slots: Vec<SyncCell<Option<Result<Vec<RowMatch>>>>> =
        (0..nb).map(|_| SyncCell::new(None)).collect();
    let step = if limit == usize::MAX && nb >= workers * 8 {
        4
    } else {
        1
    };

    let run_worker = || {
        let mut dec_buf = Vec::new();
        loop {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            let base = next.fetch_add(step, Ordering::Relaxed);
            if base >= nb || base > cutoff.load(Ordering::Relaxed) {
                break;
            }
            let end = (base + step).min(nb);
            for b in base..end {
                if b > cutoff.load(Ordering::Relaxed) {
                    break;
                }
                let mb = &meta[b];
                if mb.row_count == 0 {
                    continue;
                }
                let ib = &in_blocks[b];
                let blk_lim = limit.saturating_add(usize::from(mb.had_in_carry));
                let res: Result<Vec<RowMatch>> = (|| {
                    let mut local_out = Vec::new();
                    if mb.decoded_raw.is_none() {
                        let mut dummy = if mb.had_in_carry {
                            Some((usize::MAX, 0, 0, Vec::new()))
                        } else {
                            None
                        };
                        if crate::columns::search_grammar_block_col_pred(
                            ib.payload,
                            b,
                            mb.cur_global,
                            &mut dummy,
                            pred,
                            blk_lim,
                            &mut local_out,
                        )?
                        .is_some()
                        {
                            local_out.retain(|rm| rm.block_index != usize::MAX);
                            return Ok(local_out);
                        }
                        local_out.clear();
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
                        return Ok(local_out);
                    }
                    let base_row = mb.cur_global - usize::from(mb.had_in_carry);
                    let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
                    let start_i = usize::from(mb.had_in_carry);
                    let end_i = lines.len()
                        - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
                    for (row_idx, &lb) in lines.iter().enumerate().take(end_i).skip(start_i) {
                        if line_matches_col_pred(lb, pred) {
                            local_out.push(RowMatch {
                                block_index: b,
                                row_index: row_idx,
                                global_row_index: base_row + row_idx,
                                bytes: lb.to_vec(),
                            });
                            if local_out.len() >= limit {
                                break;
                            }
                        }
                    }
                    Ok(local_out)
                })();
                match res {
                    Ok(v) => {
                        if limit < usize::MAX && v.len() >= limit {
                            cutoff.fetch_min(b, Ordering::Relaxed);
                        }
                        if !v.is_empty() {
                            unsafe {
                                *slots[b].get() = Some(Ok(v));
                            }
                        }
                    }
                    Err(e) => {
                        unsafe {
                            *slots[b].get() = Some(Err(e));
                        }
                        failed.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    };

    std::thread::scope(|s| {
        for _ in 1..workers {
            s.spawn(run_worker);
        }
        run_worker();
    });

    let cut = cutoff.load(Ordering::Relaxed);
    let mut out = Vec::new();
    for (b, slot) in slots.into_iter().enumerate() {
        if let Some(res) = slot.into_inner() {
            let v = res?;
            if b <= cut {
                out.extend(v);
            }
        }
    }
    if !boundary_matches.is_empty() {
        out.extend(boundary_matches);
        out.sort_unstable_by_key(|rm| rm.global_row_index);
    }
    if out.len() > limit {
        out.truncate(limit);
    }
    Ok(out)
}

/// Find up to `limit` records in `frame_bytes` satisfying column predicate `pred`
/// directly in the compressed domain.
pub fn filter_column(
    frame_bytes: &[u8],
    pred: &ColPred<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    options.validate()?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        if in_blocks.len() > 1 {
            return filter_column_parallel(&in_blocks, pred, limit, options);
        }
    }

    let mut out = Vec::new();
    let mut cur_global = 0usize;
    let mut carry: Option<(usize, usize, usize, Vec<u8>)> = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut global_blk = 0usize;

    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            let blk_idx = global_blk;
            global_blk += 1;
            if out.len() >= limit && carry.is_none() {
                return Ok(out);
            }
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
                if let Some(new_rows) = crate::columns::search_grammar_block_col_pred(
                    payload,
                    blk_idx,
                    cur_global,
                    &mut carry,
                    pred,
                    limit,
                    &mut out,
                )? {
                    cur_global += new_rows;
                    if out.len() >= limit && carry.is_none() {
                        return Ok(out);
                    }
                    continue;
                }
            }
            query::decode_block_into(f_bytes, local_blk, info.raw_size, options, &mut buf)?;
            let raw: &[u8] = &buf;
            if raw.is_empty() {
                continue;
            }
            let had_carry = carry.is_some();
            let base_row = cur_global - usize::from(had_carry);
            let lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
            let blk_rows = lines.len();
            for (row_idx, &line_bytes) in lines.iter().enumerate() {
                if row_idx == 0 {
                    if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                        c_bytes.extend_from_slice(line_bytes);
                        if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                            carry = Some((c_blk, c_row, c_glob, c_bytes));
                            break;
                        }
                        if line_matches_col_pred(&c_bytes, pred) && out.len() < limit {
                            out.push(RowMatch {
                                block_index: c_blk,
                                row_index: c_row,
                                global_row_index: c_glob,
                                bytes: c_bytes,
                            });
                        }
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some((blk_idx, row_idx, base_row + row_idx, line_bytes.to_vec()));
                    break;
                }
                if line_matches_col_pred(line_bytes, pred) && out.len() < limit {
                    out.push(RowMatch {
                        block_index: blk_idx,
                        row_index: row_idx,
                        global_row_index: base_row + row_idx,
                        bytes: line_bytes.to_vec(),
                    });
                }
            }
            cur_global = base_row + blk_rows;
        }
    }
    if let Some((c_blk, c_row, c_glob, c_bytes)) = carry.take() {
        if line_matches_col_pred(&c_bytes, pred) && out.len() < limit {
            out.push(RowMatch {
                block_index: c_blk,
                row_index: c_row,
                global_row_index: c_glob,
                bytes: c_bytes,
            });
        }
    }
    Ok(out)
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
