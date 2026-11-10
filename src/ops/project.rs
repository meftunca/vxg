//! Compressed-domain column projection (`select_columns`, `project_frame`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{query, simd, DecodeOptions, Frame, Options, Result, BLOCK_COLUMN};

use super::emit::{emit_ready_blocks, encode_raw_chunks, BytesRef, ReadyBlock};
use super::expr::{line_matches_expr, PredExpr};
use super::line::ColTarget;
use super::project_line::{project_line, validate_projection_cols};
use super::{build_par_meta_and_boundary, collect_par_blocks, ParInBlock, ParMeta, SyncCell};

fn eval_select_block(
    ib: &ParInBlock<'_>,
    mb: &ParMeta<'_>,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    blk_limit: usize,
    options: &DecodeOptions,
    dec_buf: &mut Vec<u8>,
) -> Result<(usize, Vec<u8>)> {
    if mb.row_count == 0 || blk_limit == 0 {
        return Ok((0, Vec::new()));
    }
    let mut out = Vec::new();
    if mb.decoded_raw.is_none() {
        if let Some(emitted) = crate::columns::with_v6_cols(ib.col_slices, || {
            crate::columns::select_grammar_block_par_into(
                ib.payload,
                mb.had_in_carry,
                where_expr,
                cols,
                blk_limit,
                &mut out,
            )
        })? {
            return Ok((emitted, out));
        }
        out.clear();
    }
    let raw: &[u8] = if let Some(ref pre) = mb.decoded_raw {
        pre
    } else {
        ib.decode_into(options, dec_buf)?;
        dec_buf
    };
    if raw.is_empty() {
        return Ok((0, out));
    }
    let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
    let start_i = usize::from(mb.had_in_carry);
    let end_i = lines.len() - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
    let mut emitted = 0usize;
    for &lb in lines.iter().take(end_i).skip(start_i) {
        if emitted >= blk_limit {
            break;
        }
        if where_expr.is_none_or(|e| line_matches_expr(lb, e)) && project_line(lb, cols, &mut out) {
            emitted += 1;
        }
    }
    Ok((emitted, out))
}

fn select_columns_parallel(
    in_blocks: &[ParInBlock<'_>],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let nb = in_blocks.len();
    let (meta, boundary_lines) = build_par_meta_and_boundary(in_blocks, options)?;
    let mut boundary_after: Vec<Option<Vec<u8>>> = if boundary_lines.is_empty() {
        Vec::new()
    } else {
        vec![None; nb]
    };
    for bl in boundary_lines.iter() {
        if where_expr.is_none_or(|e| line_matches_expr(bl.bytes, e)) {
            let mut proj = Vec::new();
            if project_line(bl.bytes, cols, &mut proj) {
                boundary_after[bl.block_index] = Some(proj);
            }
        }
    }

    let workers = options.threads.min(nb.max(1));
    let next = AtomicUsize::new(0);
    let cutoff = AtomicUsize::new(nb);
    let failed = AtomicBool::new(false);
    let done_emitted: Vec<AtomicUsize> = if limit < usize::MAX {
        (0..nb).map(|_| AtomicUsize::new(usize::MAX)).collect()
    } else {
        Vec::new()
    };
    let (prefix_lock, prefix_idx, prefix_sum) =
        (AtomicBool::new(false), AtomicUsize::new(0), AtomicUsize::new(0));
    let slots: Vec<SyncCell<Option<Result<(usize, Vec<u8>)>>>> = (0..nb).map(|_| SyncCell::new(None)).collect();
    let step = if nb >= workers * 8 { 4 } else { 1 };

    let advance_prefix = |b: usize, emitted: usize| {
        if done_emitted.is_empty() {
            return;
        }
        done_emitted[b].store(emitted, Ordering::Release);
        if prefix_idx.load(Ordering::Acquire) <= b
            && prefix_lock
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        {
            let (mut idx, mut sum) = (
                prefix_idx.load(Ordering::Relaxed),
                prefix_sum.load(Ordering::Relaxed),
            );
            while idx < nb {
                let e = done_emitted[idx].load(Ordering::Acquire);
                if e == usize::MAX {
                    break;
                }
                sum = sum.saturating_add(e);
                if let Some(Some(_)) = boundary_after.get(idx) {
                    sum = sum.saturating_add(1);
                }
                if sum >= limit {
                    cutoff.fetch_min(idx, Ordering::Relaxed);
                    idx += 1;
                    break;
                }
                idx += 1;
            }
            prefix_sum.store(sum, Ordering::Relaxed);
            prefix_idx.store(idx, Ordering::Release);
            prefix_lock.store(false, Ordering::Release);
        }
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
                if meta[b].row_count == 0 {
                    advance_prefix(b, 0);
                    continue;
                }
                let res = eval_select_block(&in_blocks[b], &meta[b], where_expr, cols, limit, options, &mut dec_buf);
                match res {
                    Ok((emitted, v)) => {
                        if emitted > 0 {
                            unsafe { *slots[b].get() = Some(Ok((emitted, v))); }
                        }
                        if limit < usize::MAX && emitted >= limit {
                            cutoff.fetch_min(b, Ordering::Relaxed);
                        }
                        advance_prefix(b, emitted);
                    }
                    Err(e) => {
                        unsafe { *slots[b].get() = Some(Err(e)); }
                        failed.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    };

    crate::pool::parallel_run(workers, run_worker);

    let cut = cutoff.load(Ordering::Relaxed);
    let mut unpacked: Vec<Option<(usize, Vec<u8>)>> = Vec::with_capacity(nb);
    let mut total_bytes = 0usize;
    for (b, slot) in slots.into_iter().enumerate() {
        if let Some(res) = slot.into_inner() {
            let (em, v) = res?;
            if b <= cut {
                total_bytes += v.len();
                unpacked.push(Some((em, v)));
            } else {
                unpacked.push(None);
            }
        } else {
            unpacked.push(None);
        }
        if b <= cut {
            if let Some(Some(ref bm)) = boundary_after.get(b) {
                total_bytes += bm.len();
            }
        }
    }

    let mut out = Vec::with_capacity(total_bytes);
    let mut total_emitted = 0usize;
    for (b, item) in unpacked.into_iter().enumerate() {
        if b > cut || total_emitted >= limit {
            break;
        }
        if let Some((em, v)) = item {
            let need = limit - total_emitted;
            if em <= need {
                out.extend_from_slice(&v);
                total_emitted += em;
            } else {
                let mut seen = 0usize;
                for (i, &byte) in v.iter().enumerate() {
                    if byte == b'\n' {
                        seen += 1;
                        if seen == need {
                            out.extend_from_slice(&v[..=i]);
                            break;
                        }
                    }
                }
                break;
            }
        }
        if total_emitted < limit {
            if let Some(Some(ref bm)) = boundary_after.get(b) {
                out.extend_from_slice(bm);
                total_emitted += 1;
            }
        }
    }
    Ok(out)
}

pub(crate) fn select_columns_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    options.validate()?;
    validate_projection_cols(cols)?;
    let frames = Frame::parse_all_for_query(
        frame_bytes,
        options.max_output,
        where_expr,
        cols,
        true,
        options.threads,
    )?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    if options.threads > 1 {
        let in_blocks = collect_par_blocks(frame_bytes, &frames);
        if in_blocks.len() > 1 {
            return select_columns_parallel(&in_blocks, where_expr, cols, limit, options);
        }
    }

    let mut out = Vec::new();
    let mut emitted = 0usize;
    let mut carry: Option<Vec<u8>> = None;
    let mut buf: Vec<u8> = Vec::new();

    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        for (local_blk, info) in parsed.blocks().enumerate() {
            if emitted >= limit && carry.is_none() {
                return Ok(out);
            }
            if info.kind == BLOCK_COLUMN {
                let b = &parsed.blocks[local_blk];
                if let Some(n) = crate::columns::with_v6_cols(b.col_slices, || {
                    crate::columns::select_grammar_block_seq_into(
                        b.payload,
                        &mut carry,
                        where_expr,
                        cols,
                        limit - emitted,
                        &mut out,
                    )
                })? {
                    emitted += n;
                    if emitted >= limit && carry.is_none() {
                        return Ok(out);
                    }
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
                        if emitted < limit
                            && where_expr.is_none_or(|e| line_matches_expr(&c_bytes, e))
                            && project_line(&c_bytes, cols, &mut out)
                        {
                            emitted += 1;
                        }
                        continue;
                    }
                }
                if row_idx + 1 == blk_rows && !line_bytes.ends_with(b"\n") {
                    carry = Some(line_bytes.to_vec());
                    break;
                }
                if emitted < limit
                    && where_expr.is_none_or(|e| line_matches_expr(line_bytes, e))
                    && project_line(line_bytes, cols, &mut out)
                {
                    emitted += 1;
                }
            }
        }
    }
    if let Some(c_bytes) = carry.take() {
        if emitted < limit
            && where_expr.is_none_or(|e| line_matches_expr(&c_bytes, e))
            && project_line(&c_bytes, cols, &mut out)
        {
            let _ = emitted + 1;
        }
    }
    Ok(out)
}

enum ProjBlockOut {
    Empty,
    Spliced {
        payload: Vec<u8>,
        new_raw_len: usize,
        crc: u32,
    },
    Bytes(Vec<u8>),
}

#[allow(clippy::too_many_arguments)]
fn eval_project_block(
    ib: &ParInBlock<'_>,
    mb: &ParMeta<'_>,
    opt_tail_row: Option<&[u8]>,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    need_checksum: bool,
    dec_options: &DecodeOptions,
    dec_buf: &mut Vec<u8>,
    crc_buf: &mut Vec<u8>,
) -> Result<ProjBlockOut> {
    if mb.row_count == 0 {
        return Ok(ProjBlockOut::Empty);
    }
    if mb.decoded_raw.is_none() {
        match crate::columns::with_v6_cols(ib.col_slices, || {
            crate::columns::project_grammar_block(
                ib.payload,
                mb.had_in_carry,
                opt_tail_row,
                where_expr,
                cols,
                need_checksum,
                crc_buf,
            )
        })? {
            crate::columns::GrammarProjectOut::Empty => return Ok(ProjBlockOut::Empty),
            crate::columns::GrammarProjectOut::Spliced {
                payload,
                new_raw_len,
                crc,
            } => {
                return Ok(ProjBlockOut::Spliced {
                    payload,
                    new_raw_len,
                    crc,
                });
            }
            crate::columns::GrammarProjectOut::Bytes(v) => return Ok(ProjBlockOut::Bytes(v)),
            crate::columns::GrammarProjectOut::Fallback => {}
        }
    }
    let raw: &[u8] = if let Some(ref pre) = mb.decoded_raw {
        pre
    } else {
        ib.decode_into(dec_options, dec_buf)?;
        dec_buf
    };
    if raw.is_empty() {
        return Ok(ProjBlockOut::Empty);
    }
    let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
    let start_i = usize::from(mb.had_in_carry);
    let end_i = lines.len() - usize::from(lines.last().is_some_and(|l| !l.ends_with(b"\n")));
    let mut out = Vec::new();
    for &lb in lines.iter().take(end_i).skip(start_i) {
        if where_expr.is_none_or(|e| line_matches_expr(lb, e)) {
            project_line(lb, cols, &mut out);
        }
    }
    Ok(if out.is_empty() {
        ProjBlockOut::Empty
    } else {
        ProjBlockOut::Bytes(out)
    })
}

pub(super) fn project_frame_impl(
    frame_bytes: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    enc_options.validate()?;
    let mut dec_eff = dec_options.clone();
    dec_eff.threads = dec_options.threads.max(enc_options.threads);
    dec_eff.validate()?;
    validate_projection_cols(cols)?;

    if enc_options.legacy_v2 {
        let projected = select_columns_impl(frame_bytes, where_expr, cols, usize::MAX, &dec_eff)?;
        return crate::compress(&projected, enc_options);
    }

    let frames = Frame::parse_all_for_query(
        frame_bytes,
        dec_eff.max_output,
        where_expr,
        cols,
        true,
        dec_eff.threads,
    )?;
    let in_blocks = collect_par_blocks(frame_bytes, &frames);
    let nb = in_blocks.len();
    if nb == 0 {
        return crate::compress(&[], enc_options);
    }

    let (meta, boundary_lines) = build_par_meta_and_boundary(&in_blocks, &dec_eff)?;
    let mut boundary_after: Vec<Option<Vec<u8>>> = if boundary_lines.is_empty() {
        Vec::new()
    } else {
        vec![None; nb]
    };
    for bl in boundary_lines.iter() {
        if where_expr.is_none_or(|e| line_matches_expr(bl.bytes, e)) {
            let mut proj = Vec::new();
            if project_line(bl.bytes, cols, &mut proj) && !proj.is_empty() {
                boundary_after[bl.block_index] = Some(proj);
            }
        }
    }

    let workers = dec_eff.threads.min(nb.max(1));
    let mut results: Vec<ProjBlockOut> = Vec::with_capacity(nb);
    if workers <= 1 {
        let mut dec_buf = Vec::new();
        let mut crc_buf = Vec::new();
        for b in 0..nb {
            let opt_tail = boundary_after.get(b).and_then(|o| o.as_deref());
            results.push(eval_project_block(
                &in_blocks[b],
                &meta[b],
                opt_tail,
                where_expr,
                cols,
                enc_options.checksum,
                &dec_eff,
                &mut dec_buf,
                &mut crc_buf,
            )?);
        }
    } else {
        let slots: Vec<SyncCell<Option<Result<ProjBlockOut>>>> =
            (0..nb).map(|_| SyncCell::new(None)).collect();
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let step = if nb >= workers * 8 { 4 } else { 1 };
        let run_worker = |_wid: usize| {
            let mut dec_buf = Vec::new();
            let mut crc_buf = Vec::new();
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
                    let opt_tail = boundary_after.get(b).and_then(|o| o.as_deref());
                    let r = eval_project_block(
                        &in_blocks[b],
                        &meta[b],
                        opt_tail,
                        where_expr,
                        cols,
                        enc_options.checksum,
                        &dec_eff,
                        &mut dec_buf,
                        &mut crc_buf,
                    );
                    let is_err = r.is_err();
                    unsafe {
                        *slots[b].get() = Some(r);
                    }
                    if is_err {
                        failed.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        };
        crate::pool::parallel_run(workers, run_worker);
        for slot in slots {
            results.push(slot.into_inner().unwrap()?);
        }
    }

    let engine = simd::Engine::new(enc_options.simd)?;
    let mut ready: Vec<ReadyBlock<'_>> = Vec::with_capacity(nb);
    let mut pending_bytes = Vec::new();

    for (b, res) in results.into_iter().enumerate() {
        let mut spliced_absorbed_tail = false;
        match res {
            ProjBlockOut::Empty => {}
            ProjBlockOut::Bytes(v) => pending_bytes.extend_from_slice(&v),
            ProjBlockOut::Spliced {
                payload,
                new_raw_len,
                crc,
            } => {
                spliced_absorbed_tail = true;
                if !pending_bytes.is_empty() {
                    crate::encode::ENCODE_SCRATCH_POOL.with(|cell| {
                        encode_raw_chunks(
                            &pending_bytes,
                            enc_options.block_size,
                            enc_options,
                            engine,
                            &mut cell.borrow_mut(),
                            &mut ready,
                        )
                    })?;
                    pending_bytes.clear();
                }
                ready.push(ReadyBlock {
                    payload: BytesRef::Owned(payload),
                    dict: &[],
                    kind: BLOCK_COLUMN,
                    raw_size: new_raw_len,
                    checksum: crc,
                });
            }
        }
        if !spliced_absorbed_tail {
            if let Some(Some(ref bm)) = boundary_after.get(b) {
                pending_bytes.extend_from_slice(bm);
            }
        }
    }

    if !pending_bytes.is_empty() {
        if ready.is_empty() {
            return crate::compress(&pending_bytes, enc_options);
        }
        crate::encode::ENCODE_SCRATCH_POOL.with(|cell| {
            encode_raw_chunks(
                &pending_bytes,
                enc_options.block_size,
                enc_options,
                engine,
                &mut cell.borrow_mut(),
                &mut ready,
            )
        })?;
    }

    if ready.is_empty() {
        return crate::compress(&[], enc_options);
    }
    emit_ready_blocks(&ready, enc_options)
}
