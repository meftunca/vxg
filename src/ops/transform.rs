use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{checksum::crc32c, query, simd, DecodeOptions, Frame, Options, Result, BLOCK_COLUMN};

use super::emit::{emit_ready_blocks, encode_raw_chunks, BytesRef, ReadyBlock};
use super::line::{apply_line_col_op, ColOp, ColPred, ColTarget};
use super::{collect_par_blocks, inspect_par_meta, ParInBlock, SyncCell};

fn naive_full_fallback(
    frame_bytes: &[u8],
    where_pred: Option<&ColPred<'_>>,
    target_col: ColTarget<'_>,
    op: &ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let raw = crate::decompress(frame_bytes, dec_options)?;
    let mut out = Vec::with_capacity(raw.len());
    for line_bytes in raw.split_inclusive(|&b| b == b'\n') {
        if let Some(nl) = apply_line_col_op(line_bytes, where_pred, target_col, op) {
            out.extend_from_slice(&nl);
        } else {
            out.extend_from_slice(line_bytes);
        }
    }
    crate::compress(&out, enc_options)
}

#[allow(clippy::too_many_arguments)]
fn transform_raw_block_lines(
    raw: &[u8],
    had_incoming_carry: bool,
    has_following_block: bool,
    override_first: Option<&[u8]>,
    override_last: Option<&[u8]>,
    where_pred: Option<&ColPred<'_>>,
    target_col: ColTarget<'_>,
    op: &ColOp<'_>,
) -> Option<Vec<u8>> {
    let lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
    let n = lines.len();
    if n == 0 {
        return None;
    }
    let mut any_changed = false;
    let mut out = Vec::with_capacity(raw.len() + 64);
    for (i, &ln) in lines.iter().enumerate() {
        let next_opt = if i == 0 && had_incoming_carry {
            override_first.map(|b| b.to_vec())
        } else if i + 1 == n && !ln.ends_with(b"\n") && has_following_block {
            override_last.map(|b| b.to_vec())
        } else {
            apply_line_col_op(ln, where_pred, target_col, op)
        };
        if let Some(nb) = next_opt {
            if nb.as_slice() != ln {
                any_changed = true;
            }
            out.extend_from_slice(&nb);
        } else {
            out.extend_from_slice(ln);
        }
    }
    any_changed.then_some(out)
}

#[allow(clippy::too_many_arguments)]
fn process_one_block<'a>(
    b: usize,
    nb: usize,
    ib: &ParInBlock<'a>,
    predecoded_raw: Option<&[u8]>,
    had_in_carry: bool,
    ov_first: Option<&[u8]>,
    ov_last: Option<&[u8]>,
    where_pred: Option<&ColPred<'_>>,
    target_col: ColTarget<'_>,
    op: &ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
    engine: simd::Engine,
    crc_buf: &mut Vec<u8>,
    out_ready: &mut Vec<ReadyBlock<'a>>,
) -> Result<bool> {
    let has_next = b + 1 < nb;
    let mut fallback_raw: Option<Vec<u8>> = None;
    if predecoded_raw.is_none() {
        match crate::columns::rewrite_grammar_block(
            ib.payload,
            had_in_carry,
            has_next,
            ov_first,
            ov_last,
            where_pred,
            target_col,
            op,
        )? {
            crate::columns::GrammarRewrite::Untouched => {
                let csum = if !enc_options.checksum {
                    0
                } else if ib.frame_checksummed {
                    ib.info.checksum
                } else {
                    query::decode_block_into(
                        ib.f_bytes,
                        ib.local_blk,
                        ib.info.raw_size,
                        dec_options,
                        crc_buf,
                    )?;
                    crc32c(crc_buf)
                };
                out_ready.push(ReadyBlock {
                    payload: BytesRef::Borrowed(ib.payload),
                    kind: ib.info.kind,
                    raw_size: ib.info.raw_size,
                    checksum: csum,
                });
                return Ok(true);
            }
            crate::columns::GrammarRewrite::Rewritten {
                payload,
                new_raw_len,
            } => {
                let csum = if enc_options.checksum {
                    crc_buf.resize(new_raw_len, 0);
                    crate::columns::decode_block_into(&payload, crc_buf)?;
                    crc32c(crc_buf)
                } else {
                    0
                };
                out_ready.push(ReadyBlock {
                    payload: BytesRef::Owned(payload),
                    kind: BLOCK_COLUMN,
                    raw_size: new_raw_len,
                    checksum: csum,
                });
                return Ok(true);
            }
            crate::columns::GrammarRewrite::Fallback => {
                let mut raw_vec = Vec::new();
                query::decode_block_into(
                    ib.f_bytes,
                    ib.local_blk,
                    ib.info.raw_size,
                    dec_options,
                    &mut raw_vec,
                )?;
                fallback_raw = Some(raw_vec);
            }
        }
    }
    let raw_ref = predecoded_raw.unwrap_or_else(|| fallback_raw.as_deref().unwrap());
    if let Some(new_raw) = transform_raw_block_lines(
        raw_ref,
        had_in_carry,
        has_next,
        ov_first,
        ov_last,
        where_pred,
        target_col,
        op,
    ) {
        if new_raw.is_empty() {
            return Ok(false);
        }
        let cs = if new_raw.len() <= 1_048_576 {
            new_raw.len()
        } else {
            enc_options.block_size
        };
        crate::encode::ENCODE_SCRATCH_POOL.with(|cell| {
            encode_raw_chunks(
                &new_raw,
                cs,
                enc_options,
                engine,
                &mut cell.borrow_mut(),
                out_ready,
            )
        })?;
    } else {
        let csum = if !enc_options.checksum {
            0
        } else if ib.frame_checksummed {
            ib.info.checksum
        } else {
            crc32c(raw_ref)
        };
        out_ready.push(ReadyBlock {
            payload: BytesRef::Borrowed(ib.payload),
            kind: ib.info.kind,
            raw_size: ib.info.raw_size,
            checksum: csum,
        });
    }
    Ok(true)
}

pub(super) fn transform_frame(
    frame_bytes: &[u8],
    where_pred: Option<&ColPred<'_>>,
    target_col: ColTarget<'_>,
    op: &ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<Vec<u8>> {
    enc_options.validate()?;
    dec_options.validate()?;
    if enc_options.legacy_v2 {
        return naive_full_fallback(
            frame_bytes,
            where_pred,
            target_col,
            op,
            enc_options,
            dec_options,
        );
    }
    let mut dec_eff = dec_options.clone();
    dec_eff.threads = dec_options.threads.max(enc_options.threads);
    dec_eff.validate()?;

    let frames = Frame::parse_all(frame_bytes, dec_eff.max_output)?;
    let in_blocks = collect_par_blocks(frame_bytes, &frames);
    let nb = in_blocks.len();
    if nb == 0 {
        return crate::compress(&[], enc_options);
    }

    let (meta, _non_empty, any_unterm) = inspect_par_meta(&in_blocks, &dec_eff)?;

    let mut override_first: Vec<Option<Vec<u8>>> = Vec::new();
    let mut override_last: Vec<Option<Vec<u8>>> = Vec::new();
    if any_unterm && nb > 1 {
        override_first.resize_with(nb, || None);
        override_last.resize_with(nb, || None);
        let mut chain = Vec::with_capacity(4);
        let mut orig_lens = Vec::with_capacity(4);
        let mut full = Vec::with_capacity(256);
        let mut b_idx = 0usize;
        while b_idx + 1 < nb {
            let Some(ref f0) = meta[b_idx].last_unterm_frag else {
                b_idx += 1;
                continue;
            };
            chain.clear();
            orig_lens.clear();
            full.clear();
            chain.push(b_idx);
            orig_lens.push(f0.len());
            full.extend_from_slice(f0);
            let mut m = b_idx + 1;
            while m < nb {
                let Some(ref fm) = meta[m].first_frag else {
                    break;
                };
                full.extend_from_slice(fm);
                chain.push(m);
                orig_lens.push(fm.len());
                if meta[m].row_count == 1 && meta[m].last_unterm_frag.is_some() && m + 1 < nb {
                    m += 1;
                } else {
                    break;
                }
            }
            if chain.len() >= 2 {
                if let Some(new_full) = apply_line_col_op(&full, where_pred, target_col, op) {
                    let k = chain.len();
                    let body = &new_full[..new_full.len().saturating_sub(1)];
                    if new_full.len() < k || body.contains(&b'\n') {
                        return naive_full_fallback(
                            frame_bytes,
                            where_pred,
                            target_col,
                            op,
                            enc_options,
                            dec_options,
                        );
                    }
                    let mut pos = 0usize;
                    for j in 0..k {
                        let plen = if j + 1 == k {
                            new_full.len() - pos
                        } else {
                            let rem = new_full.len() - pos;
                            orig_lens[j].clamp(1, rem - (k - 1 - j))
                        };
                        let piece = new_full[pos..pos + plen].to_vec();
                        pos += plen;
                        if j == 0 {
                            override_last[chain[0]] = Some(piece);
                        } else {
                            override_first[chain[j]] = Some(piece);
                        }
                    }
                }
                b_idx = m;
            } else {
                b_idx += 1;
            }
        }
    }

    let engine = simd::Engine::new(enc_options.simd)?;
    let workers = enc_options.threads.min(nb.max(1));
    let mut ready: Vec<ReadyBlock<'_>> = Vec::with_capacity(nb);

    if workers <= 1 {
        let mut crc_buf = Vec::new();
        for (b, ib) in in_blocks.iter().enumerate() {
            let ov_f = override_first.get(b).and_then(|o| o.as_deref());
            let ov_l = override_last.get(b).and_then(|o| o.as_deref());
            if !process_one_block(
                b,
                nb,
                ib,
                meta[b].decoded_raw.as_deref(),
                meta[b].had_in_carry,
                ov_f,
                ov_l,
                where_pred,
                target_col,
                op,
                enc_options,
                &dec_eff,
                engine,
                &mut crc_buf,
                &mut ready,
            )? {
                return naive_full_fallback(
                    frame_bytes,
                    where_pred,
                    target_col,
                    op,
                    enc_options,
                    dec_options,
                );
            }
        }
    } else {
        let next = AtomicUsize::new(0);
        let need_fallback = AtomicBool::new(false);
        let slots: Vec<SyncCell<Option<Result<Vec<ReadyBlock<'_>>>>>> =
            (0..nb).map(|_| SyncCell::new(None)).collect();
        let step = if nb >= workers * 8 { 2 } else { 1 };
        let run_worker = |_wid: usize| {
            let mut crc_buf = Vec::new();
            loop {
                if need_fallback.load(Ordering::Relaxed) {
                    break;
                }
                let base = next.fetch_add(step, Ordering::Relaxed);
                if base >= nb {
                    break;
                }
                let end = (base + step).min(nb);
                for b in base..end {
                    if need_fallback.load(Ordering::Relaxed) {
                        return;
                    }
                    let mut local = Vec::with_capacity(1);
                    let ov_f = override_first.get(b).and_then(|o| o.as_deref());
                    let ov_l = override_last.get(b).and_then(|o| o.as_deref());
                    let res = process_one_block(
                        b,
                        nb,
                        &in_blocks[b],
                        meta[b].decoded_raw.as_deref(),
                        meta[b].had_in_carry,
                        ov_f,
                        ov_l,
                        where_pred,
                        target_col,
                        op,
                        enc_options,
                        &dec_eff,
                        engine,
                        &mut crc_buf,
                        &mut local,
                    );
                    match res {
                        Ok(true) => unsafe {
                            *slots[b].get() = Some(Ok(local));
                        },
                        Ok(false) => {
                            need_fallback.store(true, Ordering::Relaxed);
                            return;
                        }
                        Err(e) => {
                            unsafe {
                                *slots[b].get() = Some(Err(e));
                            }
                            need_fallback.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            }
        };
        crate::pool::parallel_run(workers, run_worker);
        let mut unpacked = Vec::with_capacity(nb);
        for slot in slots {
            if let Some(res) = slot.into_inner() {
                unpacked.push(res?);
            }
        }
        if need_fallback.load(Ordering::Relaxed) {
            return naive_full_fallback(
                frame_bytes,
                where_pred,
                target_col,
                op,
                enc_options,
                dec_options,
            );
        }
        for v in unpacked {
            ready.extend(v);
        }
    }

    emit_ready_blocks(&ready, enc_options)
}
