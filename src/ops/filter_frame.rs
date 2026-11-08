use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{checksum::crc32c, query, DecodeOptions, Frame, Options, Result};

use super::emit::{emit_ready_blocks, BytesRef, ReadyBlock};
use super::expr::{line_matches_expr, PredExpr};
use super::line::ColPred;
use super::{
    build_boundary_set, collect_par_blocks, inspect_par_meta, ParInBlock, ParMeta, SyncCell,
};

enum BlockFilterOut {
    Empty,
    AllComplete,
    Bytes(Vec<u8>),
}

fn eval_one_filter_block(
    ib: &ParInBlock<'_>,
    meta: &ParMeta<'_>,
    expr: &PredExpr<'_>,
    dec_options: &DecodeOptions,
    dec_buf: &mut Vec<u8>,
) -> Result<BlockFilterOut> {
    if meta.row_count == 0 {
        return Ok(BlockFilterOut::Empty);
    }
    if meta.decoded_raw.is_none() {
        let mut out_bytes = Vec::new();
        let g_res = match *expr {
            PredExpr::Col(ref pred) => crate::columns::filter_grammar_block_into_bytes(
                ib.payload,
                meta.had_in_carry,
                pred,
                &mut out_bytes,
            )?,
            _ => crate::columns::filter_grammar_block_expr_into_bytes(
                ib.payload,
                meta.had_in_carry,
                expr,
                &mut out_bytes,
            )?,
        };
        match g_res {
            crate::columns::GrammarFilterBytesResult::Empty => return Ok(BlockFilterOut::Empty),
            crate::columns::GrammarFilterBytesResult::AllCompleteMatch => {
                return Ok(BlockFilterOut::AllComplete);
            }
            crate::columns::GrammarFilterBytesResult::Bytes => {
                return Ok(if out_bytes.is_empty() {
                    BlockFilterOut::Empty
                } else {
                    BlockFilterOut::Bytes(out_bytes)
                });
            }
            crate::columns::GrammarFilterBytesResult::Fallback => {}
        }
    }

    let raw: &[u8] = if let Some(ref pre) = meta.decoded_raw {
        pre
    } else {
        query::decode_block_into(
            ib.f_bytes,
            ib.local_blk,
            ib.info.raw_size,
            dec_options,
            dec_buf,
        )?;
        dec_buf
    };
    if raw.is_empty() {
        return Ok(BlockFilterOut::Empty);
    }
    let lines: Vec<&[u8]> = raw.split_inclusive(|&c| c == b'\n').collect();
    let start_i = usize::from(meta.had_in_carry);
    let has_unterm = lines.last().is_some_and(|l| !l.ends_with(b"\n"));
    let end_i = lines.len() - usize::from(has_unterm);
    if start_i >= end_i {
        return Ok(BlockFilterOut::Empty);
    }
    let mut out_bytes = Vec::new();
    let mut matched = 0usize;
    for &lb in &lines[start_i..end_i] {
        if line_matches_expr(lb, expr) {
            out_bytes.extend_from_slice(lb);
            matched += 1;
        }
    }
    if matched == 0 {
        Ok(BlockFilterOut::Empty)
    } else if !meta.had_in_carry && !has_unterm && matched == lines.len() {
        Ok(BlockFilterOut::AllComplete)
    } else {
        Ok(BlockFilterOut::Bytes(out_bytes))
    }
}

pub(super) fn filter_frame_impl(
    frame_bytes: &[u8],
    pred: &ColPred<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    filter_frame_expr_impl(frame_bytes, &PredExpr::Col(*pred), enc_options, dec_options)
}

pub(super) fn filter_frame_expr_impl(
    frame_bytes: &[u8],
    expr: &PredExpr<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    enc_options.validate()?;
    let mut dec_eff = dec_options.clone();
    dec_eff.threads = dec_options.threads.max(enc_options.threads);
    dec_eff.validate()?;

    let frames = Frame::parse_all(frame_bytes, dec_eff.max_output)?;
    let in_blocks = collect_par_blocks(frame_bytes, &frames);
    let nb = in_blocks.len();
    if nb == 0 {
        return crate::compress(&[], enc_options);
    }

    let (meta, non_empty, any_unterm) = inspect_par_meta(&in_blocks, &dec_eff)?;
    let boundary_set = build_boundary_set(&meta, &non_empty, any_unterm);
    let mut boundary_after: Vec<Option<&[u8]>> = if boundary_set.is_empty() {
        Vec::new()
    } else {
        vec![None; nb]
    };
    let mut any_boundary_match = false;
    for bl in boundary_set.iter() {
        if line_matches_expr(bl.bytes, expr) {
            boundary_after[bl.block_index] = Some(bl.bytes);
            any_boundary_match = true;
        }
    }

    let workers = dec_eff.threads.min(nb.max(1));
    let mut results: Vec<BlockFilterOut> = Vec::with_capacity(nb);
    if workers <= 1 {
        let mut dec_buf = Vec::new();
        for b in 0..nb {
            results.push(eval_one_filter_block(
                &in_blocks[b],
                &meta[b],
                expr,
                &dec_eff,
                &mut dec_buf,
            )?);
        }
    } else {
        let slots: Vec<SyncCell<Option<Result<BlockFilterOut>>>> =
            (0..nb).map(|_| SyncCell::new(None)).collect();
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let step = if nb >= workers * 8 { 4 } else { 1 };
        let run_worker = |_wid: usize| {
            let mut dec_buf = Vec::new();
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
                    let r =
                        eval_one_filter_block(&in_blocks[b], &meta[b], expr, &dec_eff, &mut dec_buf);
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

    // Zero-copy fast path: every non-empty block matched 100% of its complete rows with no
    // cross-block carry fragments.
    if !enc_options.legacy_v2
        && !any_unterm
        && !non_empty.is_empty()
        && non_empty
            .iter()
            .all(|&b| matches!(results[b], BlockFilterOut::AllComplete))
    {
        let mut ready = Vec::with_capacity(non_empty.len());
        let mut crc_buf = Vec::new();
        for &b in &non_empty {
            let ib = &in_blocks[b];
            let csum = if !enc_options.checksum {
                0
            } else if ib.frame_checksummed {
                ib.info.checksum
            } else if let Some(ref pre) = meta[b].decoded_raw {
                crc32c(pre)
            } else {
                ib.decode_into(&dec_eff, &mut crc_buf)?;
                crc32c(&crc_buf)
            };
            ready.push(ReadyBlock {
                payload: BytesRef::Borrowed(ib.payload),
                dict: ib.dict,
                kind: ib.info.kind,
                raw_size: ib.info.raw_size,
                checksum: csum,
            });
        }
        return emit_ready_blocks(&ready, enc_options);
    }

    let mut total_len = 0usize;
    for (b, res) in results.iter().enumerate() {
        match res {
            BlockFilterOut::Empty => {}
            BlockFilterOut::AllComplete => total_len += in_blocks[b].info.raw_size,
            BlockFilterOut::Bytes(v) => total_len += v.len(),
        }
        if any_boundary_match {
            if let Some(bm) = boundary_after[b] {
                total_len += bm.len();
            }
        }
    }
    if total_len == 0 {
        return crate::compress(&[], enc_options);
    }

    let mut raw = Vec::with_capacity(total_len);
    let mut dec_opts_full = dec_eff.clone();
    for (b, res) in results.into_iter().enumerate() {
        match res {
            BlockFilterOut::Empty => {}
            BlockFilterOut::Bytes(v) => raw.extend_from_slice(&v),
            BlockFilterOut::AllComplete => {
                if let Some(ref pre) = meta[b].decoded_raw {
                    raw.extend_from_slice(pre);
                } else {
                    let ib = &in_blocks[b];
                    let pos = raw.len();
                    raw.resize(pos + ib.info.raw_size, 0);
                    dec_opts_full.max_output = dec_opts_full.max_output.max(ib.info.raw_size);
                    crate::extract_block_into(
                        ib.f_bytes,
                        ib.local_blk,
                        &mut raw[pos..pos + ib.info.raw_size],
                        &dec_opts_full,
                    )?;
                }
            }
        }
        if any_boundary_match {
            if let Some(bm) = boundary_after[b] {
                raw.extend_from_slice(bm);
            }
        }
    }
    crate::compress(&raw, enc_options)
}
