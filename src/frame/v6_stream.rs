//! `VERSION_V6` group serializer (`append_serialized_v6_group`, `repack_v4_v5_to_v6`) and
//! incremental per-column compressed stream accumulator (`V6StreamBuilder`) for [`crate::StreamEncoder`].

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::checksum::crc32c;
use crate::frame::v5::{
    self, block_raw_size, block_row_meta_raw, parse_v1_to_v5_header, write_header_bytes,
    FLAG_COL_GROUP,
};
use crate::frame::{
    entry_size, read_u32, HEADER_SIZE, V4_SIZE_MASK, VERSION_V4, VERSION_V5, VERSION_V6,
};
use crate::{block, columns, Error, Options, Result};

use super::{
    compute_v6_dir_crc, put_uleb_usize, uleb_len_usize, GroupDirEntry, GROUP_CUM_FLAG, GROUP_MAGIC,
    GROUP_TRAILER_MAGIC, GROUP_V2_FLAG, TARGET_COL_STREAM_BYTES,
};

#[derive(Clone)]
pub(crate) struct BlockSplit<'a> {
    pub payload: &'a [u8],
    pub p_after_raw_len: usize, pub s0_tpl_end: usize, pub s0_hdr_end: usize, pub s0_raw_end: usize,
    pub all_ones_rc: Option<u16>,
    pub spans: Vec<(usize, usize)>,
    pub word: u32, pub csum: u32, pub raw_size: u32,
}

#[inline]
fn extend_with_crc(dst: &mut Vec<u8>, slice: &[u8]) {
    dst.extend_from_slice(slice);
    dst.extend_from_slice(&crc32c(slice).to_le_bytes());
}

#[inline]
fn extend_mode3_s0(dst: &mut Vec<u8>, sel_tail: &[u8]) {
    let st = dst.len();
    dst.push(0u8);
    dst.extend_from_slice(sel_tail);
    let c = crc32c(&dst[st..]);
    dst.extend_from_slice(&c.to_le_bytes());
}

pub(crate) fn append_serialized_v6_group(
    v6_payload: &mut Vec<u8>,
    start_block: u32,
    slots: usize,
    splits: &[BlockSplit<'_>],
) {
    let count = splits.len();
    let mut canon_s0 = Vec::new();
    let mut canon_tail: &[u8] = &[];
    let mut canon_tpl: &[u8] = &[];
    if slots > 0 && count > 0 {
        let b0 = if count > 1 && splits[0].all_ones_rc.is_none() && splits[1].all_ones_rc.is_some() {
            &splits[1]
        } else {
            &splits[0]
        };
        let tail0 = &b0.payload[b0.p_after_raw_len..b0.s0_hdr_end];
        let max_rn = splits.iter().map(|b| b.raw_size as usize).max().unwrap_or(0);
        if 1 + uleb_len_usize(max_rn) + tail0.len() <= 16_384 {
            canon_s0.reserve(1 + uleb_len_usize(max_rn) + tail0.len());
            canon_s0.push(1u8);
            put_uleb_usize(&mut canon_s0, max_rn);
            canon_s0.extend_from_slice(tail0);
            canon_tail = tail0;
            canon_tpl = &b0.payload[b0.p_after_raw_len..b0.s0_tpl_end];
        }
    }
    let num_streams = if slots > 0 { slots + 2 } else { 1 };
    let slot_flags = (slots as u16) | GROUP_CUM_FLAG | GROUP_V2_FLAG;
    let gh_len = 18 + count * 12 + num_streams * count * 4 + canon_s0.len();

    let mut gh = Vec::with_capacity(gh_len);
    gh.extend_from_slice(GROUP_MAGIC);
    gh.extend_from_slice(&0u32.to_le_bytes());
    gh.extend_from_slice(&start_block.to_le_bytes());
    gh.extend_from_slice(&(count as u16).to_le_bytes());
    gh.extend_from_slice(&slot_flags.to_le_bytes());
    gh.extend_from_slice(&(canon_s0.len() as u16).to_le_bytes());

    for b in splits {
        gh.extend_from_slice(&b.word.to_le_bytes());
        gh.extend_from_slice(&b.csum.to_le_bytes());
        gh.extend_from_slice(&b.raw_size.to_le_bytes());
    }
    let classify_s0 = |b: &BlockSplit<'_>| -> u8 {
        if slots > 0 && !canon_s0.is_empty() {
            if &b.payload[b.p_after_raw_len..b.s0_hdr_end] == canon_tail {
                return 0;
            }
            if &b.payload[b.p_after_raw_len..b.s0_tpl_end] == canon_tpl {
                if b.all_ones_rc.is_some() {
                    return 1;
                }
                if b.s0_hdr_end >= b.s0_tpl_end + 3 {
                    return 3;
                }
            }
        }
        2
    };
    let mut acc0 = 0u32;
    for b in splits {
        match classify_s0(b) {
            0 => {}
            1 => acc0 += 6,
            3 => acc0 += (1 + (b.s0_hdr_end - b.s0_tpl_end) + 4) as u32,
            _ => acc0 += (b.s0_hdr_end + 4) as u32,
        }
        gh.extend_from_slice(&acc0.to_le_bytes());
    }
    let mut total_col_bytes = 0usize;
    for c in 0..slots {
        let mut acc = 0u32;
        for b in splits {
            let (st, en) = b.spans[c];
            acc += ((en - st) + 4) as u32;
            gh.extend_from_slice(&acc.to_le_bytes());
        }
        total_col_bytes += acc as usize;
    }
    let mut acc_rr = 0u32;
    if slots > 0 {
        for b in splits {
            if b.s0_raw_end > b.s0_hdr_end {
                acc_rr += ((b.s0_raw_end - b.s0_hdr_end) + 4) as u32;
            }
            gh.extend_from_slice(&acc_rr.to_le_bytes());
        }
    }
    gh.extend_from_slice(&canon_s0);
    let hdr_crc = crc32c(&gh[8..]);
    gh[4..8].copy_from_slice(&hdr_crc.to_le_bytes());

    let streams_bytes = (acc0 as usize) + total_col_bytes + (acc_rr as usize);
    v6_payload.reserve(gh_len * 2 + streams_bytes + 8);
    v6_payload.extend_from_slice(&gh);
    for b in splits {
        match classify_s0(b) {
            0 => {}
            1 => extend_with_crc(v6_payload, &b.all_ones_rc.unwrap().to_le_bytes()),
            3 => extend_mode3_s0(v6_payload, &b.payload[b.s0_tpl_end..b.s0_hdr_end]),
            _ => extend_with_crc(v6_payload, &b.payload[..b.s0_hdr_end]),
        }
    }
    for c in 0..slots {
        for b in splits {
            let (st, en) = b.spans[c];
            extend_with_crc(v6_payload, &b.payload[st..en]);
        }
    }
    if slots > 0 {
        for b in splits {
            if b.s0_raw_end > b.s0_hdr_end {
                extend_with_crc(v6_payload, &b.payload[b.s0_hdr_end..b.s0_raw_end]);
            }
        }
    }
    v6_payload.extend_from_slice(&gh);
    v6_payload.extend_from_slice(&(gh_len as u32).to_le_bytes());
    v6_payload.extend_from_slice(GROUP_TRAILER_MAGIC);
}

#[inline]
fn make_block_split(payload: &[u8], word: u32, csum: u32, raw_size: u32) -> BlockSplit<'_> {
    if (word >> 30) == block::COLUMN {
        if let Some((p_after_raw_len, s0_tpl_end, s0_hdr_end, s0_raw_end, all_ones_rc, spans)) =
            columns::split_grammar_packet_streams(payload)
        {
            return BlockSplit { payload, p_after_raw_len, s0_tpl_end, s0_hdr_end, s0_raw_end, all_ones_rc, spans, word, csum, raw_size };
        }
    }
    BlockSplit { payload, p_after_raw_len: 0, s0_tpl_end: 0, s0_hdr_end: payload.len(), s0_raw_end: payload.len(), all_ones_rc: None, spans: Vec::new(), word, csum, raw_size }
}

pub(crate) fn repack_v4_v5_to_v6(v5_bytes: &[u8], bound: usize) -> Result<Vec<u8>> {
    let hdr = parse_v1_to_v5_header(v5_bytes, usize::MAX)?;
    if hdr.version < VERSION_V4 || hdr.version > VERSION_V5 {
        return Ok(v5_bytes.to_vec());
    }
    let esize = entry_size(hdr.version);
    let mut block_ranges: Vec<(usize, usize, u32, u32, u32)> = Vec::with_capacity(hdr.count);
    let (mut next_payload, mut next_raw, mut raw_col_blocks) = (hdr.payload_start, 0usize, 0usize);
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * esize;
        let rn = block_raw_size(v5_bytes, &hdr, i, next_raw)?;
        next_raw += rn;
        let word = read_u32(v5_bytes, p)?;
        raw_col_blocks += usize::from((word >> 30) == block::COLUMN);
        let cn = (word & V4_SIZE_MASK) as usize;
        let csum = read_u32(v5_bytes, p + 4)?;
        let end = next_payload.checked_add(cn).ok_or(Error::SizeOverflow)?;
        if end > v5_bytes.len() {
            return Err(Error::Truncated);
        }
        block_ranges.push((next_payload, end, word, csum, rn as u32));
        next_payload = end;
    }
    if hdr.count < 2 || raw_col_blocks * 2 < hdr.count || (hdr.block_size >= 65_536 && v5_bytes.len() < 65_536) {
        return Ok(v5_bytes.to_vec());
    }

    let mut splits: Vec<BlockSplit<'_>> = vec![make_block_split(&[], 0, 0, 0); hdr.count];
    if hdr.count > 0 {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
            .clamp(1, 16)
            .min(hdr.count);
        if workers <= 1 || hdr.count < 8 {
            for (i, &(st, en, word, csum, raw_size)) in block_ranges.iter().enumerate() {
                splits[i] = make_block_split(&v5_bytes[st..en], word, csum, raw_size);
            }
        } else {
            let next = AtomicUsize::new(0);
            let splits_ptr = splits.as_mut_ptr() as usize;
            let br_slice = block_ranges.as_slice();
            crate::pool::parallel_try_run(workers, |_| {
                loop {
                    let base = next.fetch_add(16, Ordering::Relaxed);
                    if base >= br_slice.len() {
                        return Ok(());
                    }
                    for i in base..(base + 16).min(br_slice.len()) {
                        let (st, en, word, csum, raw_size) = br_slice[i];
                        let item = make_block_split(&v5_bytes[st..en], word, csum, raw_size);
                        unsafe { *(splits_ptr as *mut BlockSplit<'_>).add(i) = item; }
                    }
                }
            })?;
        }
    }

    let col_blocks = splits.iter().filter(|b| !b.spans.is_empty()).count();
    if hdr.count < 2 || col_blocks * 2 < hdr.count || (hdr.block_size >= 65_536 && v5_bytes.len() < 65_536) {
        return Ok(v5_bytes.to_vec());
    }
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < splits.len() {
        let slots = splits[idx].spans.len();
        let target_bytes = slots.max(1).saturating_mul(TARGET_COL_STREAM_BYTES);
        let (start, mut acc_bytes) = (idx, 0usize);
        while idx < splits.len() && splits[idx].spans.len() == slots && (idx - start) < 4095 {
            let b_bytes = if slots > 0 {
                splits[idx].payload.len() - splits[idx].s0_raw_end
            } else {
                splits[idx].payload.len()
            };
            if idx > start && acc_bytes + b_bytes > target_bytes {
                break;
            }
            acc_bytes += b_bytes;
            idx += 1;
        }
        groups.push((start, idx - start, slots));
    }
    if groups.len() > 4095 {
        return Ok(v5_bytes.to_vec());
    }

    let group_dir_bytes = 16 + 16 * groups.len();
    let prefix_len = hdr.group_dir_offset + group_dir_bytes;
    let mut out = Vec::with_capacity(v5_bytes.len() + group_dir_bytes + groups.len() * 16_384);
    out.resize(prefix_len, 0);
    let mut dir_entries = Vec::with_capacity(groups.len());
    for &(start, count, slots) in &groups {
        dir_entries.push(GroupDirEntry { payload_rel_offset: (out.len() - prefix_len) as u64, start_block: start as u32, block_count: count as u16, slot_count: slots as u16 });
        append_serialized_v6_group(&mut out, start as u32, slots, &splits[start..start + count]);
    }
    if out.len() > bound {
        return Ok(v5_bytes.to_vec());
    }
    let v6_payload_len = (out.len() - prefix_len) as u64;
    out[..hdr.group_dir_offset].copy_from_slice(&v5_bytes[..hdr.group_dir_offset]);
    let dict_wire = hdr.has_dict.then(|| (&v5_bytes[hdr.dict_offset..hdr.dict_offset + hdr.dict_wire_len], hdr.dict_raw_len));
    write_header_bytes(&mut out[..HEADER_SIZE], VERSION_V6, hdr.checksummed, hdr.has_var_raw, hdr.has_row_index, dict_wire, hdr.raw_size, hdr.block_size, hdr.count);
    let flags = u16::from_le_bytes([out[6], out[7]]) | FLAG_COL_GROUP;
    out[6..8].copy_from_slice(&flags.to_le_bytes());
    let dr_packed = (hdr.dict_raw_len as u32) | ((dir_entries.len() as u32) << 20);
    out[28..32].copy_from_slice(&dr_packed.to_le_bytes());

    let mut entries_raw = Vec::with_capacity(16 * dir_entries.len());
    for e in &dir_entries {
        entries_raw.extend_from_slice(&e.payload_rel_offset.to_le_bytes());
        entries_raw.extend_from_slice(&e.start_block.to_le_bytes());
        entries_raw.extend_from_slice(&e.block_count.to_le_bytes());
        entries_raw.extend_from_slice(&e.slot_count.to_le_bytes());
    }
    let p = hdr.group_dir_offset;
    out[p..p + 4].copy_from_slice(&(dir_entries.len() as u32).to_le_bytes());
    out[p + 4..p + 12].copy_from_slice(&v6_payload_len.to_le_bytes());
    let dir_crc = compute_v6_dir_crc(&out[..p + 12], &entries_raw);
    out[p + 12..p + 16].copy_from_slice(&dir_crc.to_le_bytes());
    out[p + 16..prefix_len].copy_from_slice(&entries_raw);
    Ok(out)
}

#[derive(Default)]
pub(crate) struct V6StreamBuilder {
    pub eff_opts: Option<Options>,
    pub saw_columnar_no_dict: bool,
    slots: Option<usize>,
    has_var_raw: bool,
    has_row_index: bool,
    b_meta: Vec<(u32, u32, u32, u32)>,
    canon_s0: Vec<u8>,
    canon_tail_start: usize,
    canon_tpl_end: usize,
    s0_buf: Vec<u8>,
    s0_cum: Vec<u32>,
    col_bufs: Vec<Vec<u8>>,
    col_cums: Vec<Vec<u32>>,
    raw_rows_buf: Vec<u8>,
    raw_rows_cum: Vec<u32>,
}

impl V6StreamBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn buffered_compressed_bytes(&self) -> usize {
        self.s0_buf.len()
            + self.raw_rows_buf.len()
            + self.col_bufs.iter().map(Vec::len).sum::<usize>()
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
        self.eff_opts.as_ref().map_or(base_opts.block_size, |o| o.block_size)
    }

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
        if hdr.has_dict {
            self.saw_columnar_no_dict = false;
            let mut written = self.flush_group(writer, base_opts)?;
            let repacked = repack_v4_v5_to_v6(v5_bytes, usize::MAX)?;
            writer.write_all(&repacked).map_err(|e| Error::Backend(e.to_string()))?;
            written += repacked.len() as u64;
            return Ok(written);
        }

        let mut total_written = 0u64;
        let esize = entry_size(hdr.version);
        let (mut next_payload, mut next_raw) = (hdr.payload_start, 0usize);
        let eff_bs = self.eff_opts.as_ref().map_or(base_opts.block_size, |o| o.block_size);

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

            let split = (kind == block::COLUMN)
                .then(|| columns::split_grammar_packet_streams(payload))
                .flatten();
            let (p_after_rn, s0_tpl_end, s0_hdr_end, s0_raw_end, all_ones_rc, spans_ref): (
                usize, usize, usize, usize, Option<u16>, &[(usize, usize)],
            ) = match split {
                Some((pa, stpl, sh, sr, rc, ref sp)) => (pa, stpl, sh, sr, rc, sp.as_slice()),
                None => (0, 0, payload.len(), payload.len(), None, &[]),
            };
            let blk_slots = spans_ref.len();
            self.saw_columnar_no_dict = blk_slots > 0;

            if let Some(cur_slots) = self.slots {
                if cur_slots != blk_slots || self.b_meta.len() >= 4095 {
                    total_written += self.flush_group(writer, base_opts)?;
                }
            }
            if self.slots.is_none() {
                self.slots = Some(blk_slots);
                self.has_var_raw = hdr.has_var_raw || base_opts.record_aligned;
                self.has_row_index = hdr.has_row_index || base_opts.row_index;
                self.canon_s0.clear();
                self.canon_tail_start = 0;
                self.canon_tpl_end = 0;
                if blk_slots > 0 {
                    let tail0 = &payload[p_after_rn..s0_hdr_end];
                    let max_rn = eff_bs.max(rn);
                    if 1 + uleb_len_usize(max_rn) + tail0.len() <= 16_384 {
                        self.canon_s0.push(1u8);
                        put_uleb_usize(&mut self.canon_s0, max_rn);
                        self.canon_tail_start = self.canon_s0.len();
                        self.canon_tpl_end = self.canon_tail_start + (s0_tpl_end - p_after_rn);
                        self.canon_s0.extend_from_slice(tail0);
                    }
                }
                if self.col_bufs.len() < blk_slots {
                    self.col_bufs.resize_with(blk_slots, || {
                        Vec::with_capacity(TARGET_COL_STREAM_BYTES + 65_536)
                    });
                    self.col_cums.resize_with(blk_slots, || Vec::with_capacity(256));
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
            if blk_slots > 0 && !self.canon_s0.is_empty() {
                if &payload[p_after_rn..s0_hdr_end] == &self.canon_s0[self.canon_tail_start..] {
                    // 0 bytes in Stream 0
                } else if &payload[p_after_rn..s0_tpl_end] == &self.canon_s0[self.canon_tail_start..self.canon_tpl_end] {
                    if let Some(rc) = all_ones_rc {
                        extend_with_crc(&mut self.s0_buf, &rc.to_le_bytes());
                    } else if s0_hdr_end >= s0_tpl_end + 3 {
                        extend_mode3_s0(&mut self.s0_buf, &payload[s0_tpl_end..s0_hdr_end]);
                    } else {
                        extend_with_crc(&mut self.s0_buf, &payload[..s0_hdr_end]);
                    }
                } else {
                    extend_with_crc(&mut self.s0_buf, &payload[..s0_hdr_end]);
                }
            } else {
                extend_with_crc(&mut self.s0_buf, &payload[..s0_hdr_end]);
            }
            self.s0_cum.push(self.s0_buf.len() as u32);
            for (c, &(st, en)) in spans_ref.iter().enumerate() {
                extend_with_crc(&mut self.col_bufs[c], &payload[st..en]);
                self.col_cums[c].push(self.col_bufs[c].len() as u32);
            }
            if blk_slots > 0 {
                if s0_raw_end > s0_hdr_end {
                    extend_with_crc(&mut self.raw_rows_buf, &payload[s0_hdr_end..s0_raw_end]);
                }
                self.raw_rows_cum.push(self.raw_rows_buf.len() as u32);
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

    pub fn flush_group<W: Write>(&mut self, writer: &mut W, base_opts: &Options) -> Result<u64> {
        let count = self.b_meta.len();
        if count == 0 {
            return Ok(0);
        }
        let slots = self.slots.unwrap_or(0);
        let eff_bs = self.eff_opts.as_ref().map_or(base_opts.block_size, |o| o.block_size);
        let raw_size: usize = self.b_meta.iter().map(|m| m.2 as usize).sum();
        let expected_fixed_count = raw_size / eff_bs + usize::from(raw_size % eff_bs != 0);
        let any_non_fixed = self
            .b_meta
            .iter()
            .enumerate()
            .any(|(i, m)| (i + 1 < count && m.2 as usize != eff_bs) || m.2 as usize > eff_bs);
        let has_var_raw = self.has_var_raw || count != expected_fixed_count || any_non_fixed;
        let has_row_index = self.has_row_index && self.b_meta.iter().all(|m| m.3 != 0);

        let num_streams = if slots > 0 { slots + 2 } else { 1 };
        let slot_flags = (slots as u16) | GROUP_CUM_FLAG | GROUP_V2_FLAG;
        let gh_len = 18 + count * 12 + num_streams * count * 4 + self.canon_s0.len();
        let mut gh = Vec::with_capacity(gh_len);
        gh.extend_from_slice(GROUP_MAGIC);
        gh.extend_from_slice(&[0u8; 8]);
        gh.extend_from_slice(&(count as u16).to_le_bytes());
        gh.extend_from_slice(&slot_flags.to_le_bytes());
        gh.extend_from_slice(&(self.canon_s0.len() as u16).to_le_bytes());
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
        if slots > 0 {
            for &off in &self.raw_rows_cum {
                gh.extend_from_slice(&off.to_le_bytes());
            }
        }
        gh.extend_from_slice(&self.canon_s0);
        let hdr_crc = crc32c(&gh[8..]);
        gh[4..8].copy_from_slice(&hdr_crc.to_le_bytes());

        let streams_len = self.s0_buf.len() + total_col_bytes + self.raw_rows_buf.len();
        let v6_payload_len = (gh_len * 2 + streams_len + 8) as u64;
        let prefix_len = HEADER_SIZE + count * 8 + usize::from(has_var_raw) * count * 4 + usize::from(has_row_index) * count * 4 + 32;
        let mut hdr_buf = vec![0u8; prefix_len];
        write_header_bytes(&mut hdr_buf[..HEADER_SIZE], VERSION_V6, base_opts.checksum, has_var_raw, has_row_index, None, raw_size, eff_bs, count);
        let flags = u16::from_le_bytes([hdr_buf[6], hdr_buf[7]]) | FLAG_COL_GROUP;
        hdr_buf[6..8].copy_from_slice(&flags.to_le_bytes());
        hdr_buf[28..32].copy_from_slice(&(1u32 << 20).to_le_bytes());

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
        hdr_buf[p..p + 4].copy_from_slice(&1u32.to_le_bytes());
        hdr_buf[p + 4..p + 12].copy_from_slice(&v6_payload_len.to_le_bytes());
        let dir_crc = compute_v6_dir_crc(&hdr_buf[..p + 12], &entry_raw);
        hdr_buf[p + 12..p + 16].copy_from_slice(&dir_crc.to_le_bytes());
        hdr_buf[p + 16..p + 32].copy_from_slice(&entry_raw);

        (|| -> io::Result<()> {
            writer.write_all(&hdr_buf)?;
            writer.write_all(&gh)?;
            writer.write_all(&self.s0_buf)?;
            for c in 0..slots { writer.write_all(&self.col_bufs[c])?; }
            if slots > 0 { writer.write_all(&self.raw_rows_buf)?; }
            writer.write_all(&gh)?;
            writer.write_all(&(gh_len as u32).to_le_bytes())?;
            writer.write_all(GROUP_TRAILER_MAGIC)
        })().map_err(|e| Error::Backend(e.to_string()))?;

        self.b_meta.clear();
        self.canon_s0.clear();
        self.s0_buf.clear();
        self.s0_cum.clear();
        self.raw_rows_buf.clear();
        self.raw_rows_cum.clear();
        for c in 0..slots {
            self.col_bufs[c].clear();
            self.col_cums[c].clear();
        }
        self.slots = None;
        Ok(hdr_buf.len() as u64 + v6_payload_len)
    }
}
