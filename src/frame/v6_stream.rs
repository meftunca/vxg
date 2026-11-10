//! Incremental per-column compressed stream accumulator (`V6StreamBuilder`) for [`crate::StreamEncoder`].
//!
//! Instead of buffering $N$ raw blocks in memory, each micro-batch of blocks is encoded immediately,
//! its compressed column slices are appended to per-column buffers (`col_bufs[0..slots]`), and when
//! the group reaches `~1 MiB`/column (`slots * TARGET_COL_STREAM_BYTES`), the streams are written
//! sequentially to the underlying writer without allocating a second copy of the group payload.

use std::io::{self, Write};

use crate::checksum::crc32c;
use crate::frame::v5::{
    self, block_raw_size, block_row_meta_raw, parse_v1_to_v5_header, write_header_bytes,
    FLAG_COL_GROUP,
};
use crate::frame::{entry_size, read_u32, HEADER_SIZE, V4_SIZE_MASK, VERSION_V6};
use crate::{block, columns, Error, Options, Result};

use super::{GROUP_CUM_FLAG, GROUP_MAGIC, GROUP_TRAILER_MAGIC, TARGET_COL_STREAM_BYTES};

#[derive(Default)]
pub(crate) struct V6StreamBuilder {
    pub eff_opts: Option<Options>,
    slots: Option<usize>,
    has_var_raw: bool,
    has_row_index: bool,
    /// `(word, csum, raw_size, row_meta)` per block in the active group.
    b_meta: Vec<(u32, u32, u32, u32)>,
    s0_buf: Vec<u8>,
    s0_cum: Vec<u32>,
    col_bufs: Vec<Vec<u8>>,
    col_cums: Vec<Vec<u32>>,
}

impl V6StreamBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total compressed bytes currently buffered across `s0_buf` and `col_bufs`.
    pub fn buffered_compressed_bytes(&self) -> usize {
        self.s0_buf.len() + self.col_bufs.iter().map(Vec::len).sum::<usize>()
    }

    pub fn resolve_eff_bs(&mut self, raw_buf: &[u8], base_opts: &Options, is_final: bool) -> usize {
        let len = raw_buf.len();
        if self.eff_opts.is_none() && (len >= 65_536 || is_final) {
            let mut probe = base_opts.clone();
            probe.col_group = false;
            if probe.record_aligned && probe.block_size == 65_536 && len >= 4096 {
                let sample = &raw_buf[..len.min(65_536)];
                let nl = sample.iter().filter(|&&c| c == b'\n').count().max(1);
                let avg = sample.len() / nl;
                if avg >= 110 && sample.first() == Some(&b'{') {
                    probe.block_size = (2048 * avg)
                        .div_ceil(65_536)
                        .saturating_mul(65_536)
                        .clamp(65_536, 524_288);
                }
            } else if len >= 1_048_576 {
                probe = v5::resolve_effective_options(raw_buf, &probe);
                probe.col_group = false;
            }
            self.eff_opts = Some(probe);
        }
        self.eff_opts
            .as_ref()
            .map_or(base_opts.block_size, |o| o.block_size)
    }

    /// Ingest a freshly encoded `v4`/`v5` micro-frame (`1..8` blocks) into the per-column
    /// compressed buffers, flushing completed groups sequentially to `writer`.
    pub fn ingest_v5_micro_frame<W: Write>(
        &mut self,
        v5_bytes: &[u8],
        writer: &mut W,
        base_opts: &Options,
    ) -> Result<u64> {
        let hdr = parse_v1_to_v5_header(v5_bytes, usize::MAX)?;
        if hdr.count == 0 {
            return Ok(0);
        }
        // If the micro-frame trained a frame-shared LZ dictionary (`BLOCK_DICT_LZ`), flush any open
        // column group first and emit this micro-frame repacked as a self-contained v6 frame.
        if hdr.has_dict {
            let mut written = self.flush_group(writer, base_opts)?;
            let repacked = super::repack_v4_v5_to_v6(v5_bytes, usize::MAX)?;
            writer
                .write_all(&repacked)
                .map_err(|e| Error::Backend(e.to_string()))?;
            written += repacked.len() as u64;
            return Ok(written);
        }

        let mut total_written = 0u64;
        let esize = entry_size(hdr.version);
        let mut next_payload = hdr.payload_start;
        let mut next_raw = 0usize;

        for i in 0..hdr.count {
            let p = HEADER_SIZE + i * esize;
            let rn = block_raw_size(v5_bytes, &hdr, i, next_raw)?;
            let row_meta = block_row_meta_raw(v5_bytes, &hdr, i, rn)?.unwrap_or(0);
            next_raw += rn;
            let word = read_u32(v5_bytes, p)?;
            let cn = (word & V4_SIZE_MASK) as usize;
            let csum = read_u32(v5_bytes, p + 4)?;
            let kind = word >> 30;
            let payload = &v5_bytes[next_payload..next_payload + cn];
            next_payload += cn;

            let split = if kind == block::COLUMN {
                columns::split_grammar_packet_streams(payload)
            } else {
                None
            };
            let (s0_len, spans_ref): (usize, &[(usize, usize)]) = match split {
                Some((s0, ref sp)) => (s0, sp.as_slice()),
                None => (payload.len(), &[]),
            };
            let blk_slots = spans_ref.len();

            if let Some(cur_slots) = self.slots {
                if cur_slots != blk_slots || self.b_meta.len() >= 4095 {
                    total_written += self.flush_group(writer, base_opts)?;
                }
            }
            if self.slots.is_none() {
                self.slots = Some(blk_slots);
                self.has_var_raw = hdr.has_var_raw || base_opts.record_aligned;
                self.has_row_index = hdr.has_row_index || base_opts.row_index;
                if self.col_bufs.len() < blk_slots {
                    self.col_bufs.resize_with(blk_slots, || {
                        Vec::with_capacity(TARGET_COL_STREAM_BYTES + 65_536)
                    });
                    self.col_cums
                        .resize_with(blk_slots, || Vec::with_capacity(256));
                }
                for c in 0..blk_slots {
                    self.col_bufs[c].clear();
                    self.col_cums[c].clear();
                }
            } else {
                self.has_var_raw |= hdr.has_var_raw;
                self.has_row_index |= hdr.has_row_index && row_meta != 0;
            }

            self.b_meta.push((word, csum, rn as u32, row_meta));
            self.s0_buf.extend_from_slice(&payload[..s0_len]);
            self.s0_cum.push(self.s0_buf.len() as u32);
            for (c, &(st, en)) in spans_ref.iter().enumerate() {
                self.col_bufs[c].extend_from_slice(&payload[st..en]);
                self.col_cums[c].push(self.col_bufs[c].len() as u32);
            }

            let target = blk_slots.max(1).saturating_mul(TARGET_COL_STREAM_BYTES);
            let col_bytes: usize = if blk_slots > 0 {
                self.col_bufs[..blk_slots].iter().map(Vec::len).sum()
            } else {
                self.s0_buf.len()
            };
            if col_bytes >= target || self.b_meta.len() >= 4095 {
                total_written += self.flush_group(writer, base_opts)?;
            }
        }
        Ok(total_written)
    }

    /// Write the currently open column group sequentially to `writer` as a `VERSION_V6` frame
    /// without allocating a second copy of the compressed streams.
    pub fn flush_group<W: Write>(&mut self, writer: &mut W, base_opts: &Options) -> Result<u64> {
        let count = self.b_meta.len();
        if count == 0 {
            return Ok(0);
        }
        let slots = self.slots.unwrap_or(0);
        let eff_bs = self
            .eff_opts
            .as_ref()
            .map_or(base_opts.block_size, |o| o.block_size);
        let raw_size: usize = self.b_meta.iter().map(|m| m.2 as usize).sum();
        let expected_fixed_count = raw_size / eff_bs + usize::from(raw_size % eff_bs != 0);
        let any_non_fixed = self
            .b_meta
            .iter()
            .enumerate()
            .any(|(i, m)| (i + 1 < count && m.2 as usize != eff_bs) || m.2 as usize > eff_bs);
        let has_var_raw = self.has_var_raw || count != expected_fixed_count || any_non_fixed;
        let has_row_index = self.has_row_index && self.b_meta.iter().all(|m| m.3 != 0);

        let slot_flags = (slots as u16) | GROUP_CUM_FLAG;
        let bdir_len = count * 12;
        let cum_len = (slots + 1) * count * 4;
        let gh_len = 16 + bdir_len + cum_len;
        let mut gh = Vec::with_capacity(gh_len);
        gh.extend_from_slice(GROUP_MAGIC);
        gh.extend_from_slice(&0u32.to_le_bytes());
        gh.extend_from_slice(&0u32.to_le_bytes());
        gh.extend_from_slice(&(count as u16).to_le_bytes());
        gh.extend_from_slice(&slot_flags.to_le_bytes());
        for &(word, csum, rn, _) in &self.b_meta {
            gh.extend_from_slice(&word.to_le_bytes());
            gh.extend_from_slice(&csum.to_le_bytes());
            gh.extend_from_slice(&rn.to_le_bytes());
        }
        for &off in &self.s0_cum {
            gh.extend_from_slice(&off.to_le_bytes());
        }
        let mut total_col_bytes = 0usize;
        for c in 0..slots {
            for &off in &self.col_cums[c] {
                gh.extend_from_slice(&off.to_le_bytes());
            }
            total_col_bytes += self.col_bufs[c].len();
        }
        let hdr_crc = crc32c(&gh[8..]);
        gh[4..8].copy_from_slice(&hdr_crc.to_le_bytes());

        let streams_len = self.s0_buf.len() + total_col_bytes;
        let v6_payload_len = (gh_len + streams_len + gh_len + 8) as u64;

        let prefix_len = HEADER_SIZE
            + count * 8
            + if has_var_raw { count * 4 } else { 0 }
            + if has_row_index { count * 4 } else { 0 }
            + 32;
        let mut hdr_buf = vec![0u8; prefix_len];
        write_header_bytes(
            &mut hdr_buf[..HEADER_SIZE],
            VERSION_V6,
            base_opts.checksum,
            has_var_raw,
            has_row_index,
            None,
            raw_size,
            eff_bs,
            count,
        );
        let flags = u16::from_le_bytes([hdr_buf[6], hdr_buf[7]]) | FLAG_COL_GROUP;
        hdr_buf[6..8].copy_from_slice(&flags.to_le_bytes());
        let dr_packed = 1u32 << 20;
        hdr_buf[28..32].copy_from_slice(&dr_packed.to_le_bytes());

        let mut p = HEADER_SIZE;
        for &(word, csum, _, _) in &self.b_meta {
            hdr_buf[p..p + 4].copy_from_slice(&word.to_le_bytes());
            hdr_buf[p + 4..p + 8].copy_from_slice(&csum.to_le_bytes());
            p += 8;
        }
        if has_var_raw {
            for &(_, _, rn, _) in &self.b_meta {
                hdr_buf[p..p + 4].copy_from_slice(&rn.to_le_bytes());
                p += 4;
            }
        }
        if has_row_index {
            for &(_, _, _, rm) in &self.b_meta {
                hdr_buf[p..p + 4].copy_from_slice(&rm.to_le_bytes());
                p += 4;
            }
        }
        let mut entry_raw = [0u8; 16];
        entry_raw[12..14].copy_from_slice(&(count as u16).to_le_bytes());
        entry_raw[14..16].copy_from_slice(&(slots as u16).to_le_bytes());
        let dir_crc = crc32c(&entry_raw);
        hdr_buf[p..p + 4].copy_from_slice(&1u32.to_le_bytes());
        hdr_buf[p + 4..p + 12].copy_from_slice(&v6_payload_len.to_le_bytes());
        hdr_buf[p + 12..p + 16].copy_from_slice(&dir_crc.to_le_bytes());
        hdr_buf[p + 16..p + 32].copy_from_slice(&entry_raw);

        let write_res = (|| -> io::Result<()> {
            writer.write_all(&hdr_buf)?;
            writer.write_all(&gh)?;
            writer.write_all(&self.s0_buf)?;
            for c in 0..slots {
                writer.write_all(&self.col_bufs[c])?;
            }
            writer.write_all(&gh)?;
            writer.write_all(&(gh_len as u32).to_le_bytes())?;
            writer.write_all(GROUP_TRAILER_MAGIC)?;
            Ok(())
        })();
        write_res.map_err(|e| Error::Backend(e.to_string()))?;

        self.b_meta.clear();
        self.s0_buf.clear();
        self.s0_cum.clear();
        for c in 0..slots {
            self.col_bufs[c].clear();
            self.col_cums[c].clear();
        }
        self.slots = None;
        Ok(hdr_buf.len() as u64 + v6_payload_len)
    }
}
