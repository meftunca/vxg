use crate::{
    block, checksum::crc32c, frame, query, simd, DecodeOptions, Error, Frame, Options, Result,
    BLOCK_COLUMN,
};

use super::line::{apply_line_col_op, ColOp, ColPred, ColTarget};

struct InBlock<'a> {
    f_bytes: &'a [u8],
    local_idx: usize,
    frame_checksummed: bool,
    info: frame::BlockInfo,
    payload: &'a [u8],
}

struct BlockMeta {
    decoded_raw: Option<Vec<u8>>,
    row_count: usize,
    first_frag: Option<Vec<u8>>,
    last_unterm_frag: Option<Vec<u8>>,
}

enum BytesRef<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl BytesRef<'_> {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            BytesRef::Borrowed(s) => s,
            BytesRef::Owned(v) => v,
        }
    }
}

struct ReadyBlock<'a> {
    payload: BytesRef<'a>,
    kind: u32,
    raw_size: usize,
    checksum: u32,
}

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

fn encode_raw_chunks<'a>(
    raw: &[u8],
    chunk_size: usize,
    enc_options: &Options,
    engine: simd::Engine,
    scratch: &mut block::Scratch,
    out: &mut Vec<ReadyBlock<'a>>,
) -> Result<()> {
    for chunk in raw.chunks(chunk_size.max(1)) {
        scratch.payloads_mut().clear();
        let (kind, off, len) = match crate::columns::encode_grammar_block_into(
            chunk,
            scratch.payloads_mut(),
        )? {
            Some((po, pl)) => (block::COLUMN, po, pl),
            None if (512..=65536).contains(&chunk.len()) => {
                let p = scratch.payloads_mut();
                let o = p.len();
                p.extend_from_slice(chunk);
                (block::RAW, o, chunk.len())
            }
            None => {
                let enc = block::encode::<false>(chunk, scratch, engine, enc_options.acceleration);
                (enc.kind, enc.offset, enc.len)
            }
        };
        let bytes = scratch.payloads_mut()[off..off + len].to_vec();
        out.push(ReadyBlock {
            payload: BytesRef::Owned(bytes),
            kind,
            raw_size: chunk.len(),
            checksum: if enc_options.checksum { crc32c(chunk) } else { 0 },
        });
    }
    Ok(())
}

fn append_v4_frame(
    out: &mut Vec<u8>,
    blocks: &[ReadyBlock<'_>],
    block_size: usize,
    checksummed: bool,
) -> Result<()> {
    let version = frame::VERSION_RELEASE;
    let esize = frame::ENTRY_SIZE_V4;
    let count = blocks.len();
    let mut raw_total = 0usize;
    let mut payload_total = 0usize;
    for b in blocks {
        raw_total = raw_total.checked_add(b.raw_size).ok_or(Error::SizeOverflow)?;
        payload_total = payload_total
            .checked_add(b.payload.as_slice().len())
            .ok_or(Error::SizeOverflow)?;
    }
    let dir_bytes = esize.checked_mul(count).ok_or(Error::SizeOverflow)?;
    let frame_start = out.len();
    let table_end = frame_start
        .checked_add(frame::HEADER_SIZE)
        .and_then(|x| x.checked_add(dir_bytes))
        .ok_or(Error::SizeOverflow)?;
    out.try_reserve(frame::HEADER_SIZE + dir_bytes + payload_total)
        .map_err(|_| Error::SizeOverflow)?;
    out.extend_from_slice(frame::MAGIC);
    out.extend_from_slice(&version.to_le_bytes());
    out.extend_from_slice(&(u16::from(checksummed)).to_le_bytes());
    out.extend_from_slice(&(raw_total as u64).to_le_bytes());
    out.extend_from_slice(&(block_size as u32).to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.resize(table_end, 0);
    for (i, b) in blocks.iter().enumerate() {
        frame::write_entry(
            out,
            frame_start + frame::HEADER_SIZE + i * esize,
            version,
            0,
            0,
            b.raw_size,
            b.payload.as_slice().len(),
            if checksummed { b.checksum } else { 0 },
            b.kind,
        );
    }
    for b in blocks {
        out.extend_from_slice(b.payload.as_slice());
    }
    Ok(())
}

#[inline]
fn is_valid_block_size(bs: usize) -> bool {
    (4096..=1_048_576).contains(&bs) && bs % 64 == 0
}

#[inline]
fn single_block_bs(raw_size: usize, pref_bs: usize) -> usize {
    if raw_size <= pref_bs {
        pref_bs
    } else {
        raw_size.next_multiple_of(64).clamp(4096, 1_048_576)
    }
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
    let frames = Frame::parse_all(frame_bytes, dec_options.max_output)?;
    let mut in_blocks = Vec::new();
    for (f_bytes, parsed) in query::frame_slices(frame_bytes, &frames) {
        let csum = parsed.checksummed();
        for (local_idx, info) in parsed.blocks().enumerate() {
            in_blocks.push(InBlock {
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

    let mut meta = Vec::with_capacity(nb);
    let mut had_in_carry = vec![false; nb];
    let mut incoming = false;
    for (b, ib) in in_blocks.iter().enumerate() {
        had_in_carry[b] = incoming;
        if ib.info.kind == BLOCK_COLUMN {
            if let Some(gb) = crate::columns::peek_grammar_boundary(ib.payload, incoming)? {
                if gb.row_count > 0 {
                    incoming = gb.last_unterminated_raw.is_some();
                }
                meta.push(BlockMeta {
                    decoded_raw: None,
                    row_count: gb.row_count,
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
            dec_options,
            &mut raw_vec,
        )?;
        let rc = raw_vec.split_inclusive(|&c| c == b'\n').count();
        let ends_nl = raw_vec.last() == Some(&b'\n');
        let first_frag = if incoming && !raw_vec.is_empty() {
            let end = raw_vec
                .iter()
                .position(|&c| c == b'\n')
                .map_or(raw_vec.len(), |i| i + 1);
            Some(raw_vec[..end].to_vec())
        } else {
            None
        };
        let last_unterm_frag = if !raw_vec.is_empty() && !ends_nl {
            let st = raw_vec
                .iter()
                .rposition(|&c| c == b'\n')
                .map_or(0, |i| i + 1);
            Some(raw_vec[st..].to_vec())
        } else {
            None
        };
        if rc > 0 {
            incoming = !ends_nl;
        }
        meta.push(BlockMeta {
            decoded_raw: Some(raw_vec),
            row_count: rc,
            first_frag,
            last_unterm_frag,
        });
    }

    let mut override_first: Vec<Option<Vec<u8>>> = vec![None; nb];
    let mut override_last: Vec<Option<Vec<u8>>> = vec![None; nb];
    let mut b_idx = 0usize;
    while b_idx + 1 < nb {
        let Some(ref f0) = meta[b_idx].last_unterm_frag else {
            b_idx += 1;
            continue;
        };
        let mut chain = vec![b_idx];
        let mut orig_lens = vec![f0.len()];
        let mut full = f0.clone();
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

    let engine = simd::Engine::new(enc_options.simd)?;
    let mut enc_scratch = block::Scratch::new();
    let mut crc_buf = Vec::new();
    let mut ready: Vec<ReadyBlock<'_>> = Vec::with_capacity(nb);

    for (b, ib) in in_blocks.iter().enumerate() {
        let has_next = b + 1 < nb;
        if meta[b].decoded_raw.is_none() {
            match crate::columns::rewrite_grammar_block(
                ib.payload,
                had_in_carry[b],
                has_next,
                override_first[b].as_deref(),
                override_last[b].as_deref(),
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
                            ib.local_idx,
                            ib.info.raw_size,
                            dec_options,
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
                    continue;
                }
                crate::columns::GrammarRewrite::Rewritten {
                    payload,
                    new_raw_len,
                } => {
                    let csum = if enc_options.checksum {
                        crc_buf.resize(new_raw_len, 0);
                        crate::columns::decode_block_into(&payload, &mut crc_buf)?;
                        crc32c(&crc_buf)
                    } else {
                        0
                    };
                    ready.push(ReadyBlock {
                        payload: BytesRef::Owned(payload),
                        kind: BLOCK_COLUMN,
                        raw_size: new_raw_len,
                        checksum: csum,
                    });
                    continue;
                }
                crate::columns::GrammarRewrite::Fallback => {
                    let mut raw_vec = Vec::new();
                    query::decode_block_into(
                        ib.f_bytes,
                        ib.local_idx,
                        ib.info.raw_size,
                        dec_options,
                        &mut raw_vec,
                    )?;
                    meta[b].decoded_raw = Some(raw_vec);
                }
            }
        }
        let raw_ref = meta[b].decoded_raw.as_ref().unwrap();
        if let Some(new_raw) = transform_raw_block_lines(
            raw_ref,
            had_in_carry[b],
            has_next,
            override_first[b].as_deref(),
            override_last[b].as_deref(),
            where_pred,
            target_col,
            op,
        ) {
            if new_raw.is_empty() {
                return naive_full_fallback(
                    frame_bytes,
                    where_pred,
                    target_col,
                    op,
                    enc_options,
                    dec_options,
                );
            }
            let cs = if new_raw.len() <= 1_048_576 {
                new_raw.len()
            } else {
                enc_options.block_size
            };
            encode_raw_chunks(
                &new_raw,
                cs,
                enc_options,
                engine,
                &mut enc_scratch,
                &mut ready,
            )?;
        } else {
            let csum = if !enc_options.checksum {
                0
            } else if ib.frame_checksummed {
                ib.info.checksum
            } else {
                crc32c(raw_ref)
            };
            ready.push(ReadyBlock {
                payload: BytesRef::Borrowed(ib.payload),
                kind: ib.info.kind,
                raw_size: ib.info.raw_size,
                checksum: csum,
            });
        }
    }

    let mut out = Vec::new();
    if ready.len() == 1 {
        let bs = single_block_bs(ready[0].raw_size, enc_options.block_size);
        append_v4_frame(&mut out, &ready, bs, enc_options.checksum)?;
        return Ok(out);
    }

    let cand_bs = if is_valid_block_size(ready[0].raw_size) {
        ready[0].raw_size
    } else {
        enc_options.block_size
    };
    if ready[..ready.len() - 1]
        .iter()
        .all(|rb| rb.raw_size == cand_bs)
        && ready.last().unwrap().raw_size <= cand_bs
    {
        append_v4_frame(&mut out, &ready, cand_bs, enc_options.checksum)?;
        return Ok(out);
    }

    let mut idx = 0usize;
    while idx < ready.len() {
        let r0 = ready[idx].raw_size;
        if idx + 1 < ready.len() && is_valid_block_size(r0) {
            let mut end = idx + 1;
            while end < ready.len() {
                let rn = ready[end].raw_size;
                if rn == r0 {
                    end += 1;
                } else if rn < r0 {
                    end += 1;
                    break;
                } else {
                    break;
                }
            }
            append_v4_frame(&mut out, &ready[idx..end], r0, enc_options.checksum)?;
            idx = end;
        } else {
            let bs = single_block_bs(r0, enc_options.block_size);
            append_v4_frame(&mut out, &ready[idx..idx + 1], bs, enc_options.checksum)?;
            idx += 1;
        }
    }
    Ok(out)
}
