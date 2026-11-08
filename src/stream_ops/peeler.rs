//! Cross-frame split-line peeler (`FramePeeler`) for bounded-memory streaming execution.

use std::io::Write;

use crate::{
    block,
    checksum::crc32c,
    ops::emit::{emit_ready_blocks, BytesRef, ReadyBlock},
    DecodeOptions, Error, Frame, Options, Result, BLOCK_COLUMN,
};

pub(crate) fn write_all_vxg<W: Write>(writer: &mut W, bytes: &[u8]) -> Result<()> {
    writer
        .write_all(bytes)
        .map_err(|e| Error::Backend(e.to_string()))
}

pub(crate) struct FramePeeler {
    carry: Vec<u8>,
    work_buf: Vec<u8>,
    dec_blk: Vec<u8>,
}

impl FramePeeler {
    pub(crate) fn new() -> Self {
        Self {
            carry: Vec::new(),
            work_buf: Vec::new(),
            dec_blk: Vec::new(),
        }
    }

    fn push_raw_chunks<'a>(
        raw: &[u8],
        chunk_size: usize,
        checksummed: bool,
        ready: &mut Vec<ReadyBlock<'a>>,
    ) {
        for chunk in raw.chunks(chunk_size.max(1)) {
            ready.push(ReadyBlock {
                payload: BytesRef::Owned(chunk.to_vec()),
                dict: &[],
                kind: block::RAW,
                raw_size: chunk.len(),
                checksum: if checksummed { crc32c(chunk) } else { 0 },
            });
        }
    }

    fn block_ends_with_nl_fast(
        f_bytes: &[u8],
        info: &crate::frame::BlockInfo,
    ) -> Result<Option<bool>> {
        let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
        if info.kind == BLOCK_COLUMN {
            if let Some((rows, ends_nl)) = crate::columns::grammar_block_row_meta(payload)? {
                if rows > 0 {
                    return Ok(Some(ends_nl));
                }
            }
            return Ok(None);
        }
        if info.kind == block::RAW && !payload.is_empty() {
            return Ok(Some(payload.last() == Some(&b'\n')));
        }
        Ok(None)
    }

    /// Peels any unterminated trailing line off `f_bytes` and prepends any incoming `self.carry`
    /// from earlier frames, keeping all interior compressed `GRAMMAR_PACKET` blocks untouched.
    /// Returns `Ok(None)` if the frame contributed `0` complete `\n`-terminated lines.
    pub(crate) fn peel_frame<'a>(
        &'a mut self,
        f_bytes: &'a [u8],
        dec: &DecodeOptions,
    ) -> Result<Option<&'a [u8]>> {
        let parsed = Frame::parse(f_bytes, usize::MAX)?;
        let nb = parsed.block_count();
        if nb == 0 || parsed.raw_size() == 0 {
            return Ok(None);
        }
        let blocks: Vec<crate::frame::BlockInfo> = parsed.blocks().collect();
        let last_ends_nl_fast = Self::block_ends_with_nl_fast(f_bytes, &blocks[nb - 1])?;

        if self.carry.is_empty() && last_ends_nl_fast == Some(true) {
            return Ok(Some(f_bytes));
        }

        let bs = parsed.block_size();
        let csum = parsed.checksummed();
        let emit_opts = Options {
            block_size: bs,
            checksum: csum,
            ..Options::default()
        };

        if parsed.version() < 2 {
            let full = crate::decompress(f_bytes, dec)?;
            let Some(last_nl) = full.iter().rposition(|&b| b == b'\n') else {
                self.carry.extend_from_slice(&full);
                return Ok(None);
            };
            let mut ready = Vec::new();
            if !self.carry.is_empty() {
                Self::push_raw_chunks(&self.carry, bs, csum, &mut ready);
            }
            Self::push_raw_chunks(&full[..=last_nl], bs, csum, &mut ready);
            self.carry.clear();
            self.carry.extend_from_slice(&full[last_nl + 1..]);
            self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
            return Ok(Some(&self.work_buf));
        }

        if last_ends_nl_fast == Some(true) {
            let mut ready = Vec::with_capacity(nb + 2);
            Self::push_raw_chunks(&self.carry, bs, csum, &mut ready);
            self.carry.clear();
            for info in &blocks {
                ready.push(ReadyBlock {
                    payload: BytesRef::Borrowed(
                        &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
                    ),
                    dict: parsed.dict(),
                    kind: info.kind,
                    raw_size: info.raw_size,
                    checksum: info.checksum,
                });
            }
            self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
            return Ok(Some(&self.work_buf));
        }

        let mut tail_rev_parts: Vec<Vec<u8>> = Vec::new();
        let mut split_blk: Option<(usize, Vec<u8>)> = None;

        for idx in (0..nb).rev() {
            let info = &blocks[idx];
            self.dec_blk.clear();
            self.dec_blk.resize(info.raw_size, 0);
            let mut d_opts = dec.clone();
            d_opts.max_output = d_opts.max_output.max(info.raw_size);
            parsed.decode_block_into(idx, &mut self.dec_blk, &d_opts)?;
            if let Some(pos) = self.dec_blk.iter().rposition(|&b| b == b'\n') {
                if pos + 1 < self.dec_blk.len() {
                    tail_rev_parts.push(self.dec_blk[pos + 1..].to_vec());
                }
                split_blk = Some((idx, self.dec_blk[..=pos].to_vec()));
                break;
            } else if !self.dec_blk.is_empty() {
                tail_rev_parts.push(self.dec_blk.clone());
            }
        }

        let Some((split_idx, prefix_with_nl)) = split_blk else {
            for part in tail_rev_parts.iter().rev() {
                self.carry.extend_from_slice(part);
            }
            return Ok(None);
        };

        if self.carry.is_empty() && split_idx + 1 == nb && tail_rev_parts.is_empty() {
            return Ok(Some(f_bytes));
        }

        let mut ready = Vec::with_capacity(split_idx + 3);
        if !self.carry.is_empty() {
            Self::push_raw_chunks(&self.carry, bs, csum, &mut ready);
            self.carry.clear();
        }
        for info in &blocks[..split_idx] {
            ready.push(ReadyBlock {
                payload: BytesRef::Borrowed(
                    &f_bytes[info.payload_offset..info.payload_offset + info.payload_size],
                ),
                dict: parsed.dict(),
                kind: info.kind,
                raw_size: info.raw_size,
                checksum: info.checksum,
            });
        }
        Self::push_raw_chunks(&prefix_with_nl, bs, csum, &mut ready);

        for part in tail_rev_parts.iter().rev() {
            self.carry.extend_from_slice(part);
        }

        self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
        Ok(Some(&self.work_buf))
    }

    /// Emits any final unterminated trailing line left at `EOF` as a single `VXG1` frame.
    pub(crate) fn finish_tail(&mut self) -> Result<Option<&[u8]>> {
        if self.carry.is_empty() {
            return Ok(None);
        }
        let mut ready = Vec::new();
        Self::push_raw_chunks(&self.carry, 65536, false, &mut ready);
        self.carry.clear();
        let emit_opts = Options {
            checksum: false,
            ..Options::default()
        };
        self.work_buf = emit_ready_blocks(&ready, &emit_opts)?;
        Ok(Some(&self.work_buf))
    }
}
