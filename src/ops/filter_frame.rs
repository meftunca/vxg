use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{
    checksum::crc32c, frame, query, DecodeOptions, Frame, Options, Result, BLOCK_COLUMN,
};

use super::emit::{emit_ready_blocks, BytesRef, ReadyBlock};
use super::line::{line_matches_col_pred, ColPred};
use super::SyncCell;

struct FilterBlock<'a> {
    f_bytes: &'a [u8],
    local_idx: usize,
    frame_checksummed: bool,
    info: frame::BlockInfo,
    payload: &'a [u8],
}

struct FilterMeta<'a> {
    decoded_raw: Option<Vec<u8>>,
    row_count: usize,
    had_in_carry: bool,
    first_frag: Option<Cow<'a, [u8]>>,
    last_unterm_frag: Option<Cow<'a, [u8]>>,
}

enum BlockFilterOut {
    Empty,
    AllComplete,
    Bytes(Vec<u8>),
}

fn eval_one_filter_block(
    ib: &FilterBlock<'_>,
    meta: &FilterMeta<'_>,
    pred: &ColPred<'_>,
    dec_options: &DecodeOptions,
    dec_buf: &mut Vec<u8>,
) -> Result<BlockFilterOut> {
    if meta.row_count == 0 {
        return Ok(BlockFilterOut::Empty);
    }
    if meta.decoded_raw.is_none() {
        let mut out_bytes = Vec::new();
        match crate::columns::filter_grammar_block_into_bytes(
            ib.payload,
            meta.had_in_carry,
            pred,
            &mut out_bytes,
        )? {
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
            ib.local_idx,
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
        if line_matches_col_pred(lb, pred) {
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
    enc_options.validate()?;
    let mut dec_eff = dec_options.clone();
    dec_eff.threads = dec_options.threads.max(enc_options.threads);
    dec_eff.validate()?;

    let frames = Frame::parse_all(frame_bytes, dec_eff.max_output)?;
    let mut in_blocks = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        let csum = parsed.checksummed();
        for (local_idx, info) in parsed.blocks().enumerate() {
            in_blocks.push(FilterBlock {
                f_bytes,
                local_idx,
                frame_checksummed: csum,
                info,
                payload: &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
            });
        }
    }
    let nb = in_blocks.len();
    if nb == 0 {
        return crate::compress(&[], enc_options);
    }

    let mut meta: Vec<FilterMeta<'_>> = Vec::with_capacity(nb);
    let mut non_empty = Vec::with_capacity(nb);
    let mut incoming = false;
    let mut any_unterm = false;

    for (b, ib) in in_blocks.iter().enumerate() {
        let had_in = incoming;
        if ib.info.kind == BLOCK_COLUMN {
            if let Some(gb) = crate::columns::peek_grammar_boundary_fast(ib.payload, had_in)? {
                if gb.row_count > 0 {
                    incoming = gb.last_unterminated_raw.is_some();
                    any_unterm |= incoming;
                    non_empty.push(b);
                }
                meta.push(FilterMeta {
                    decoded_raw: None,
                    row_count: gb.row_count,
                    had_in_carry: had_in,
                    first_frag: gb.first_raw,
                    last_unterm_frag: gb.last_unterminated_raw,
                });
                continue;
            }
        }
        let mut raw_vec = Vec::new();
        query::decode_block_into(
            ib.f_bytes,
            ib.local_idx,
            ib.info.raw_size,
            &dec_eff,
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
            incoming = !ends_nl;
            any_unterm |= incoming;
            non_empty.push(b);
        }
        meta.push(FilterMeta {
            decoded_raw: Some(raw_vec),
            row_count: rc,
            had_in_carry: had_in,
            first_frag,
            last_unterm_frag,
        });
    }

    let mut boundary_after: Vec<Option<Vec<u8>>> = if any_unterm {
        vec![None; nb]
    } else {
        Vec::new()
    };
    let mut any_boundary_match = false;
    if any_unterm {
        let mut k_idx = 0usize;
        while k_idx < non_empty.len() {
            let b0 = non_empty[k_idx];
            let Some(ref f0) = meta[b0].last_unterm_frag else {
                k_idx += 1;
                continue;
            };
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
            if line_matches_col_pred(&full, pred) {
                boundary_after[b0] = Some(full);
                any_boundary_match = true;
            }
            k_idx = m.max(k_idx + 1);
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
                pred,
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
        let run_worker = || {
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
                        eval_one_filter_block(&in_blocks[b], &meta[b], pred, &dec_eff, &mut dec_buf);
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
        std::thread::scope(|s| {
            for _ in 1..workers {
                s.spawn(run_worker);
            }
            run_worker();
        });
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
                query::decode_block_into(
                    ib.f_bytes,
                    ib.local_idx,
                    ib.info.raw_size,
                    &dec_eff,
                    &mut crc_buf,
                )?;
                crc32c(&crc_buf)
            };
            ready.push(ReadyBlock {
                payload: BytesRef::Borrowed(ib.payload),
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
            if let Some(ref bm) = boundary_after[b] {
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
                        ib.local_idx,
                        &mut raw[pos..pos + ib.info.raw_size],
                        &dec_opts_full,
                    )?;
                }
            }
        }
        if any_boundary_match {
            if let Some(ref bm) = boundary_after[b] {
                raw.extend_from_slice(bm);
            }
        }
    }
    crate::compress(&raw, enc_options)
}
