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
    if raw_size <= pref_bs {
        pref_bs
    } else {
        raw_size.next_multiple_of(64).clamp(4096, 1_048_576)
    }
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
        raw_total = raw_total
            .checked_add(b.raw_size)
            .ok_or(Error::SizeOverflow)?;
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

pub(crate) fn emit_ready_blocks(
    ready: &[ReadyBlock<'_>],
    enc_options: &Options,
) -> Result<Vec<u8>> {
    let est_cap: usize = ready
        .iter()
        .map(|rb| 40 + rb.payload.as_slice().len())
        .sum();
    let mut out = Vec::with_capacity(est_cap);
    if ready.len() == 1 {
        let bs = single_block_bs(ready[0].raw_size, enc_options.block_size);
        append_v4_frame(&mut out, ready, bs, enc_options.checksum)?;
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
        append_v4_frame(&mut out, ready, cand_bs, enc_options.checksum)?;
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

pub(super) fn encode_raw_chunks<'a>(
    raw: &[u8],
    chunk_size: usize,
    enc_options: &Options,
    engine: simd::Engine,
    scratch: &mut block::Scratch,
    out: &mut Vec<ReadyBlock<'a>>,
) -> Result<()> {
    for chunk in raw.chunks(chunk_size.max(1)) {
        scratch.payloads_mut().clear();
        let (kind, off, len) =
            match crate::columns::encode_grammar_block_into(chunk, scratch.payloads_mut())? {
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
