use crate::{block, checksum::crc32c, frame, simd, Error, Options, Result};

pub(crate) enum BytesRef<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl BytesRef<'_> {
    #[inline]
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            BytesRef::Borrowed(s) => s,
            BytesRef::Owned(v) => v,
        }
    }
}

pub(crate) struct ReadyBlock<'a> {
    pub(crate) payload: BytesRef<'a>,
    pub(crate) dict: &'a [u8],
    pub(crate) kind: u32,
    pub(crate) raw_size: usize,
    pub(crate) checksum: u32,
}

#[inline]
fn is_valid_block_size(bs: usize) -> bool {
    (4096..=1_048_576).contains(&bs) && bs % 64 == 0
}

#[inline]
fn single_block_bs(raw_size: usize, pref_bs: usize) -> usize {
    if raw_size <= pref_bs && is_valid_block_size(pref_bs) {
        pref_bs
    } else {
        raw_size.next_multiple_of(64).clamp(4096, 1_048_576)
    }
}

#[inline]
fn same_dict(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && (a.as_ptr() == b.as_ptr() || a == b)
}

#[inline]
fn v5_count_ok(raw_total: usize, bs: usize, count: usize) -> bool {
    if raw_total == 0 {
        return count == 0;
    }
    let min_c = raw_total.div_ceil(bs);
    let max_c = raw_total.min((min_c.saturating_mul(8) / 3).max(min_c + 4));
    (min_c..=max_c).contains(&count)
}

fn append_frame(
    out: &mut Vec<u8>,
    blocks: &[ReadyBlock<'_>],
    block_size: usize,
    checksummed: bool,
) -> Result<()> {
    let esize = frame::ENTRY_SIZE_V4;
    let count = blocks.len();
    let mut raw_total = 0usize;
    let mut payload_total = 0usize;
    let mut active_dict: Option<&[u8]> = None;
    for b in blocks {
        raw_total = raw_total
            .checked_add(b.raw_size)
            .ok_or(Error::SizeOverflow)?;
        payload_total = payload_total
            .checked_add(b.payload.as_slice().len())
            .ok_or(Error::SizeOverflow)?;
        if b.kind == block::DICT_LZ && !b.dict.is_empty() && active_dict.is_none() {
            active_dict = Some(b.dict);
        }
    }
    let expected_fixed = raw_total / block_size + usize::from(raw_total % block_size != 0);
    let is_fixed = count == expected_fixed
        && (count == 0
            || (blocks[..count - 1].iter().all(|b| b.raw_size == block_size)
                && blocks[count - 1].raw_size <= block_size
                && blocks.iter().all(|b| b.raw_size > 0)));
    let has_var_raw = !is_fixed;
    let version = if active_dict.is_some() || has_var_raw {
        frame::VERSION_V5
    } else {
        frame::VERSION_RELEASE
    };
    let wire_opt = active_dict.map(|d| (crate::dict_lz::encode_wire_dict(d, 3), d.len()));
    let dir_bytes = esize.checked_mul(count).ok_or(Error::SizeOverflow)?;
    let var_raw_bytes = if has_var_raw {
        count.checked_mul(4).ok_or(Error::SizeOverflow)?
    } else {
        0
    };
    let dict_bytes = wire_opt.as_ref().map_or(0, |(w, _)| w.len());
    let frame_start = out.len();
    let base_dir_end = frame_start
        .checked_add(frame::HEADER_SIZE)
        .and_then(|x| x.checked_add(dir_bytes))
        .ok_or(Error::SizeOverflow)?;
    let table_end = base_dir_end
        .checked_add(var_raw_bytes)
        .and_then(|x| x.checked_add(dict_bytes))
        .ok_or(Error::SizeOverflow)?;
    out.try_reserve(table_end - frame_start + payload_total)
        .map_err(|_| Error::SizeOverflow)?;
    out.resize(table_end, 0);
    frame::v5::write_header_bytes(
        &mut out[frame_start..table_end],
        version,
        checksummed,
        has_var_raw,
        wire_opt.as_ref().map(|(w, l)| (w.as_slice(), *l)),
        raw_total,
        block_size,
        count,
    );
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
        if has_var_raw {
            let vr_pos = base_dir_end + i * 4;
            out[vr_pos..vr_pos + 4].copy_from_slice(&(b.raw_size as u32).to_le_bytes());
        }
    }
    if let Some((wire, _)) = wire_opt {
        let d_pos = base_dir_end + var_raw_bytes;
        out[d_pos..d_pos + wire.len()].copy_from_slice(&wire);
    }
    for b in blocks {
        out.extend_from_slice(b.payload.as_slice());
    }
    Ok(())
}

pub(crate) fn emit_ready_blocks(
    ready: &[ReadyBlock<'_>],
    enc_options: &Options,
) -> Result<Vec<u8>> {
    if ready.is_empty() || ready.iter().all(|b| b.raw_size == 0) {
        return crate::compress(&[], enc_options);
    }
    let est_cap: usize = ready
        .iter()
        .map(|rb| 48 + rb.payload.as_slice().len())
        .sum();
    let mut out = Vec::with_capacity(est_cap);
    let mut idx = 0usize;
    while idx < ready.len() {
        if ready[idx].raw_size == 0 {
            idx += 1;
            continue;
        }
        let r0 = ready[idx].raw_size;
        let bs = if is_valid_block_size(r0) && (idx + 1 < ready.len() && ready[idx + 1].raw_size == r0) {
            r0
        } else {
            single_block_bs(r0, enc_options.block_size)
        };
        let mut group_dict: Option<&[u8]> = if ready[idx].kind == block::DICT_LZ && !ready[idx].dict.is_empty() {
            Some(ready[idx].dict)
        } else {
            None
        };
        let mut group_raw = r0;
        let mut end = idx + 1;
        while end < ready.len() {
            let rb = &ready[end];
            if rb.raw_size == 0 || rb.raw_size > bs {
                break;
            }
            if rb.kind == block::DICT_LZ && !rb.dict.is_empty() {
                match group_dict {
                    Some(d) if !same_dict(d, rb.dict) => break,
                    None => group_dict = Some(rb.dict),
                    Some(_) => {}
                }
            }
            let next_raw = match group_raw.checked_add(rb.raw_size) {
                Some(v) => v,
                None => break,
            };
            if !v5_count_ok(next_raw, bs, end - idx + 1) {
                break;
            }
            group_raw = next_raw;
            end += 1;
        }
        append_frame(&mut out, &ready[idx..end], bs, enc_options.checksum)?;
        idx = end;
    }
    Ok(out)
}

pub(super) fn encode_raw_chunks<'a>(
    raw: &[u8],
    chunk_size: usize,
    enc_options: &Options,
    engine: simd::Engine,
    scratch: &mut block::Scratch,
    out: &mut Vec<ReadyBlock<'a>>,
) -> Result<()> {
    let mut raw_off = 0usize;
    for chunk in raw.chunks(chunk_size.max(1)) {
        let leading_carry = raw_off > 0 && raw[raw_off - 1] != b'\n';
        raw_off += chunk.len();
        scratch.payloads_mut().clear();
        let (kind, off, len) = match crate::columns::encode_grammar_block_into_with_carry(
            chunk,
            leading_carry,
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
                    let enc =
                        block::encode::<false>(chunk, scratch, engine, enc_options.acceleration);
                    (enc.kind, enc.offset, enc.len)
                }
            };
        let bytes = scratch.payloads_mut()[off..off + len].to_vec();
        out.push(ReadyBlock {
            payload: BytesRef::Owned(bytes),
            dict: &[],
            kind,
            raw_size: chunk.len(),
            checksum: if enc_options.checksum {
                crc32c(chunk)
            } else {
                0
            },
        });
    }
    Ok(())
}
