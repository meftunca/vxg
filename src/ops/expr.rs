//! Compound boolean predicate expressions (`AND` / `OR` / `NOT` / column / substring)
//! evaluated directly in the compressed domain (`count_where`, `filter_where`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{query, DecodeOptions, Error, Frame, Result, RowMatch, BLOCK_COLUMN};

use super::line::{line_matches_col_pred, ColPred};
use super::{build_par_meta_and_boundary, collect_par_blocks, SyncCell};

/// Composable boolean predicate tree evaluated over structured or raw records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredExpr<'a> {
    /// Matches every record (`TRUE`).
    True,
    /// Structured column predicate (`JsonKey` or `Delimited`, string or numeric).
    Col(ColPred<'a>),
    /// Raw byte-substring predicate (`record` contains `needle`).
    RowContains(&'a [u8]),
    /// Logical conjunction (`AND`): matches iff every child matches (`&[] => true`).
    All(&'a [PredExpr<'a>]),
    /// Logical disjunction (`OR`): matches iff at least one child matches (`&[] => false`).
    Any(&'a [PredExpr<'a>]),
    /// Logical negation (`NOT`): matches iff the inner expression does not match.
    Not(&'a PredExpr<'a>),
}

impl<'a> PredExpr<'a> {
    #[inline]
    pub const fn col(p: ColPred<'a>) -> Self {
        Self::Col(p)
    }

    #[inline]
    pub const fn contains(needle: &'a [u8]) -> Self {
        Self::RowContains(needle)
    }

    #[inline]
    pub const fn all(children: &'a [PredExpr<'a>]) -> Self {
        Self::All(children)
    }

    #[inline]
    pub const fn any(children: &'a [PredExpr<'a>]) -> Self {
        Self::Any(children)
    }

    #[inline]
    pub const fn not(inner: &'a PredExpr<'a>) -> Self {
        Self::Not(inner)
    }
}

/// Exact single-record reference evaluator for [`PredExpr`].
#[inline]
pub fn line_matches_expr(line: &[u8], expr: &PredExpr<'_>) -> bool {
    match *expr {
        PredExpr::True => true,
        PredExpr::Col(ref p) => line_matches_col_pred(line, p),
        PredExpr::RowContains(needle) => crate::columns::contains_subslice(line, needle),
        PredExpr::All(children) => children.iter().all(|c| line_matches_expr(line, c)),
        PredExpr::Any(children) => children.iter().any(|c| line_matches_expr(line, c)),
        PredExpr::Not(inner) => !line_matches_expr(line, inner),
    }
}

fn count_where_parallel(
    frame_bytes: &[u8],
    frames: &[Frame<'_>],
    expr: &PredExpr<'_>,
    options: &DecodeOptions,
) -> Result<Option<u64>> {
    let in_blocks = collect_par_blocks(frame_bytes, frames);
    if in_blocks.len() <= 1 {
        return Ok(None);
    }
    let total = super::par_chunk::parallel_reduce_blocks(
        &in_blocks,
        options,
        |payload, carry, eval_in_carry, acc: &mut u64| {
            if let Some(cnt) =
                crate::columns::count_grammar_block_expr_chunk(payload, carry, eval_in_carry, expr)?
            {
                *acc += cnt;
                Ok(true)
            } else {
                Ok(false)
            }
        },
        |line, acc: &mut u64| {
            if line_matches_expr(line, expr) {
                *acc += 1;
            }
        },
        |dst, src| *dst += src,
    )?;
    Ok(Some(total))
}

pub(crate) fn count_where_impl(
    frame_bytes: &[u8],
    expr: &PredExpr<'_>,
    options: &DecodeOptions,
) -> Result<u64> {
    options.validate()?;
    if matches!(expr, PredExpr::True) {
        return Ok(query::frame_row_count(frame_bytes)? as u64);
    }
    let frames = Frame::parse_all_for_query(
        frame_bytes,
        options.max_output,
        Some(expr),
        &[],
        true,
        options.threads,
    )?;
    if options.threads > 1 {
        if let Some(total) = count_where_parallel(frame_bytes, &frames, expr, options)? {
            return Ok(total);
        }
    }

    let mut total = 0u64;
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();

    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if info.kind == BLOCK_COLUMN {
                let b = &parsed.blocks[local_blk];
                if let Some(cnt) = crate::columns::with_v6_cols(b.col_slices, || {
                    crate::columns::count_grammar_block_expr_seq(b.payload, &mut carry, expr)
                })? {
                    total = total.checked_add(cnt).ok_or(Error::SizeOverflow)?;
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
                        if line_matches_expr(&c_bytes, expr) {
                            total += 1;
                        }
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some(line_bytes.to_vec());
                    break;
                }
                if line_matches_expr(line_bytes, expr) {
                    total += 1;
                }
            }
        }
    }
    if let Some(c_bytes) = carry.take() {
        if line_matches_expr(&c_bytes, expr) {
            total += 1;
        }
    }
    Ok(total)
}

fn filter_where_parallel(
    frame_bytes: &[u8],
    frames: &[Frame<'_>],
    expr: &PredExpr<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Option<Vec<RowMatch>>> {
    let in_blocks = collect_par_blocks(frame_bytes, frames);
    let nb = in_blocks.len();
    if nb <= 1 {
        return Ok(None);
    }
    let (meta, boundary_lines) = build_par_meta_and_boundary(&in_blocks, options)?;
    let mut boundary_matches = Vec::new();
    for bl in boundary_lines.iter() {
        if line_matches_expr(bl.bytes, expr) {
            boundary_matches.push(RowMatch {
                block_index: bl.block_index,
                row_index: bl.row_index,
                global_row_index: bl.global_row_index,
                bytes: bl.bytes.to_vec(),
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

    let run_worker = |_wid: usize| {
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
                let res: Result<Vec<RowMatch>> = crate::columns::with_v6_cols(ib.col_slices, || {
                    let mut local_out = Vec::new();
                    if mb.decoded_raw.is_none() {
                        let mut dummy = if mb.had_in_carry {
                            Some((usize::MAX, 0, 0, Vec::new()))
                        } else {
                            None
                        };
                        if crate::columns::search_grammar_block_expr(
                            ib.payload,
                            b,
                            mb.cur_global,
                            &mut dummy,
                            expr,
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
                        ib.decode_into(options, &mut dec_buf)?;
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
                        if line_matches_expr(lb, expr) {
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
                });
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

    crate::pool::parallel_run(workers, run_worker);

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
    Ok(Some(out))
}

pub(crate) fn filter_where_impl(
    frame_bytes: &[u8],
    expr: &PredExpr<'_>,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    match *expr {
        PredExpr::Col(ref p) => return super::filter_column(frame_bytes, p, limit, options),
        PredExpr::RowContains(needle) => {
            return query::filter_rows(frame_bytes, needle, limit, options)
        }
        _ => {}
    }
    options.validate()?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let frames = Frame::parse_all_for_query(
        frame_bytes,
        options.max_output,
        Some(expr),
        &[],
        false,
        options.threads,
    )?;
    if options.threads > 1 {
        if let Some(rows) = filter_where_parallel(frame_bytes, &frames, expr, limit, options)? {
            return Ok(rows);
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
                let b = &parsed.blocks[local_blk];
                if let Some(new_rows) = crate::columns::with_v6_cols(b.col_slices, || {
                    crate::columns::search_grammar_block_expr(
                        b.payload,
                        blk_idx,
                        cur_global,
                        &mut carry,
                        expr,
                        limit,
                        &mut out,
                    )
                })? {
                    cur_global += new_rows;
                    if out.len() >= limit && carry.is_none() {
                        return Ok(out);
                    }
                    continue;
                }
            }
            query::decode_block_into(f_bytes, local_blk, info.raw_size, options, &mut buf)?;
            if buf.is_empty() {
                continue;
            }
            let had_carry = carry.is_some();
            let base_row = cur_global - usize::from(had_carry);
            let lines: Vec<&[u8]> = buf.split_inclusive(|&b| b == b'\n').collect();
            let blk_rows = lines.len();
            for (row_idx, &line_bytes) in lines.iter().enumerate() {
                if row_idx == 0 {
                    if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
                        c_bytes.extend_from_slice(line_bytes);
                        if !c_bytes.ends_with(b"\n") && blk_rows == 1 {
                            carry = Some((c_blk, c_row, c_glob, c_bytes));
                            break;
                        }
                        if line_matches_expr(&c_bytes, expr) && out.len() < limit {
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
                if line_matches_expr(line_bytes, expr) && out.len() < limit {
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
        if line_matches_expr(&c_bytes, expr) && out.len() < limit {
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
