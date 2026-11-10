//! `VERSION_V6` (`6`): Column-grouped macro-block layout with dynamic `~1 MiB`/column sizing,
//! independent per-group & per-block `CRC32C` resilience, and $O(1)$ group offset directory.

use crate::checksum::crc32c;
use crate::simd::Engine;
use crate::{block, DecodeOptions, Error, Result};

use super::recover::{FailedBlock, RecoveredBlock, RecoveryReport};
use super::v5::{
    block_raw_size, block_row_meta_raw, decode_header_dict, parse_v1_to_v5_header,
    write_header_bytes, ParsedHeader, FLAG_COL_GROUP,
};
use super::{
    entry_size, read_u32, read_u64, valid_kind_and_size, Block, BlockInfo, Frame, HEADER_SIZE,
    V4_SIZE_MASK,
};

pub const VERSION_V6: u16 = 6;
pub const GROUP_MAGIC: &[u8; 4] = b"VXG6";
pub const TARGET_COL_STREAM_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy)]
pub struct GroupDirEntry {
    pub payload_rel_offset: u64,
    pub start_block: u32,
    pub block_count: u16,
    pub slot_count: u16,
}

#[inline]
pub(crate) fn is_v6_frame(src: &[u8]) -> bool {
    src.len() >= HEADER_SIZE
        && &src[0..4] == super::MAGIC
        && u16::from_le_bytes([src[4], src[5]]) == VERSION_V6
}

struct BlockSplit<'a> {
    payload: &'a [u8],
    stream0_len: usize,
    spans: Vec<(usize, usize)>,
}

pub(crate) fn repack_v4_v5_to_v6(v5_bytes: &[u8], bound: usize) -> Result<Vec<u8>> {
    let hdr = parse_v1_to_v5_header(v5_bytes, usize::MAX)?;
    if hdr.version < super::VERSION_V4 || hdr.version > super::VERSION_V5 {
        return Ok(v5_bytes.to_vec());
    }
    let esize = entry_size(hdr.version);
    let mut splits: Vec<BlockSplit<'_>> = Vec::with_capacity(hdr.count);
    let mut next_payload = hdr.payload_start;
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * esize;
        let word = read_u32(v5_bytes, p)?;
        let cn = (word & V4_SIZE_MASK) as usize;
        let kind = word >> 30;
        let end = next_payload.checked_add(cn).ok_or(Error::SizeOverflow)?;
        let payload = v5_bytes.get(next_payload..end).ok_or(Error::Truncated)?;
        next_payload = end;
        if kind == block::COLUMN {
            if let Some((s0, spans)) = crate::columns::split_grammar_packet_streams(payload) {
                splits.push(BlockSplit {
                    payload,
                    stream0_len: s0,
                    spans,
                });
                continue;
            }
        }
        splits.push(BlockSplit {
            payload,
            stream0_len: payload.len(),
            spans: Vec::new(),
        });
    }

    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut idx = 0usize;
    while idx < splits.len() {
        let slots = splits[idx].spans.len();
        let target_bytes = if slots > 0 {
            slots.saturating_mul(TARGET_COL_STREAM_BYTES)
        } else {
            TARGET_COL_STREAM_BYTES
        };
        let start = idx;
        let mut acc_bytes = 0usize;
        while idx < splits.len() && splits[idx].spans.len() == slots && (idx - start) < 4095 {
            let b_bytes = if slots > 0 {
                splits[idx].payload.len() - splits[idx].stream0_len
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

    let mut v6_payload: Vec<u8> = Vec::with_capacity(v5_bytes.len());
    let mut dir_entries: Vec<GroupDirEntry> = Vec::with_capacity(groups.len());
    for &(start, count, slots) in &groups {
        let rel_off = v6_payload.len() as u64;
        dir_entries.push(GroupDirEntry {
            payload_rel_offset: rel_off,
            start_block: start as u32,
            block_count: count as u16,
            slot_count: slots as u16,
        });
        let mut max_len = 0usize;
        for b in &splits[start..start + count] {
            max_len = max_len.max(b.stream0_len);
            for &(s, e) in &b.spans {
                max_len = max_len.max(e - s);
            }
        }
        let use_u32 = max_len > u16::MAX as usize;
        let slot_flags = (slots as u16) | if use_u32 { 0x8000 } else { 0 };
        let g_start = v6_payload.len();
        v6_payload.extend_from_slice(GROUP_MAGIC);
        v6_payload.extend_from_slice(&0u32.to_le_bytes());
        v6_payload.extend_from_slice(&(start as u32).to_le_bytes());
        v6_payload.extend_from_slice(&(count as u16).to_le_bytes());
        v6_payload.extend_from_slice(&slot_flags.to_le_bytes());
        let push_len = |buf: &mut Vec<u8>, len: usize| {
            if use_u32 {
                buf.extend_from_slice(&(len as u32).to_le_bytes());
            } else {
                buf.extend_from_slice(&(len as u16).to_le_bytes());
            }
        };
        for b in &splits[start..start + count] {
            push_len(&mut v6_payload, b.stream0_len);
        }
        for c in 0..slots {
            for b in &splits[start..start + count] {
                let (s, e) = b.spans[c];
                push_len(&mut v6_payload, e - s);
            }
        }
        let hdr_crc = crc32c(&v6_payload[g_start + 8..]);
        v6_payload[g_start + 4..g_start + 8].copy_from_slice(&hdr_crc.to_le_bytes());
        for b in &splits[start..start + count] {
            v6_payload.extend_from_slice(&b.payload[..b.stream0_len]);
        }
        for c in 0..slots {
            for b in &splits[start..start + count] {
                let (s, e) = b.spans[c];
                v6_payload.extend_from_slice(&b.payload[s..e]);
            }
        }
    }

    let group_dir_bytes = 16 + 16 * dir_entries.len();
    let total_len = hdr.group_dir_offset + group_dir_bytes + v6_payload.len();
    if total_len > bound {
        return Ok(v5_bytes.to_vec());
    }
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&v5_bytes[..hdr.group_dir_offset]);
    let dict_wire = if hdr.has_dict {
        Some((
            &v5_bytes[hdr.dict_offset..hdr.dict_offset + hdr.dict_wire_len],
            hdr.dict_raw_len,
        ))
    } else {
        None
    };
    write_header_bytes(
        &mut out[..HEADER_SIZE],
        VERSION_V6,
        hdr.checksummed,
        hdr.has_var_raw,
        hdr.has_row_index,
        dict_wire,
        hdr.raw_size,
        hdr.block_size,
        hdr.count,
    );
    let mut flags = u16::from_le_bytes([out[6], out[7]]);
    flags |= FLAG_COL_GROUP;
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
    let dir_crc = crc32c(&entries_raw);
    out.extend_from_slice(&(dir_entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&(v6_payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&dir_crc.to_le_bytes());
    out.extend_from_slice(&entries_raw);
    out.extend_from_slice(&v6_payload);
    Ok(out)
}

pub(crate) fn parse_v6_group_dir(
    src: &[u8],
    hdr: &ParsedHeader,
) -> Result<(u64, Vec<GroupDirEntry>)> {
    if src.len() < hdr.payload_start {
        return Err(Error::Truncated);
    }
    let p = hdr.group_dir_offset;
    let gc = read_u32(src, p)? as usize;
    if gc != hdr.group_count {
        return Err(Error::InvalidFrame("group_count mismatch"));
    }
    let total_v6_payload = read_u64(src, p + 4)?;
    let expected_crc = read_u32(src, p + 12)?;
    let entries_start = p + 16;
    let entries_end = entries_start
        .checked_add(gc.checked_mul(16).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    if entries_end != hdr.payload_start {
        return Err(Error::InvalidFrame("group directory offset mismatch"));
    }
    let entries_slice = src.get(entries_start..entries_end).ok_or(Error::Truncated)?;
    if crc32c(entries_slice) != expected_crc {
        return Err(Error::InvalidFrame("group directory crc mismatch"));
    }
    let mut entries = Vec::with_capacity(gc);
    let mut expected_start = 0u32;
    for i in 0..gc {
        let ep = entries_start + i * 16;
        let rel = read_u64(src, ep)?;
        let sb = read_u32(src, ep + 8)?;
        let bc = u16::from_le_bytes([src[ep + 12], src[ep + 13]]);
        let sc = u16::from_le_bytes([src[ep + 14], src[ep + 15]]);
        if sb != expected_start || bc == 0 || sc > 64 || rel > total_v6_payload {
            return Err(Error::InvalidFrame("invalid group directory entry"));
        }
        expected_start = expected_start
            .checked_add(bc as u32)
            .ok_or(Error::SizeOverflow)?;
        entries.push(GroupDirEntry {
            payload_rel_offset: rel,
            start_block: sb,
            block_count: bc,
            slot_count: sc,
        });
    }
    if expected_start as usize != hdr.count {
        return Err(Error::InvalidFrame("group blocks != frame block count"));
    }
    Ok((total_v6_payload, entries))
}

pub(crate) struct ParsedGroupHeader<'a> {
    pub count: usize,
    pub slots: usize,
    pub use_u32: bool,
    pub lens_bytes: &'a [u8],
    pub data_offset: usize,
}

impl<'a> ParsedGroupHeader<'a> {
    pub fn parse(group_slice: &'a [u8], expected_start: u32) -> Result<Self> {
        if group_slice.len() < 16 {
            return Err(Error::Truncated);
        }
        if &group_slice[0..4] != GROUP_MAGIC {
            return Err(Error::InvalidFrame("bad v6 group magic"));
        }
        let expected_crc = read_u32(group_slice, 4)?;
        let sb = read_u32(group_slice, 8)?;
        let count = u16::from_le_bytes([group_slice[12], group_slice[13]]) as usize;
        let slot_flags = u16::from_le_bytes([group_slice[14], group_slice[15]]);
        let slots = (slot_flags & 0x7fff) as usize;
        let use_u32 = (slot_flags & 0x8000) != 0;
        if sb != expected_start || count == 0 || slots > 64 {
            return Err(Error::InvalidFrame("bad v6 group header fields"));
        }
        let num_lens = (slots + 1).checked_mul(count).ok_or(Error::SizeOverflow)?;
        let lens_byte_len = num_lens * if use_u32 { 4 } else { 2 };
        let data_offset = 16usize.checked_add(lens_byte_len).ok_or(Error::SizeOverflow)?;
        let hdr_body = group_slice
            .get(8..data_offset)
            .ok_or(Error::Truncated)?;
        if crc32c(hdr_body) != expected_crc {
            return Err(Error::InvalidFrame("v6 group header crc mismatch"));
        }
        Ok(Self {
            count,
            slots,
            use_u32,
            lens_bytes: &group_slice[16..data_offset],
            data_offset,
        })
    }

    #[inline]
    pub fn stream_len(&self, stream_idx: usize, block_in_group: usize) -> usize {
        let idx = stream_idx * self.count + block_in_group;
        if self.use_u32 {
            let p = idx * 4;
            u32::from_le_bytes(self.lens_bytes[p..p + 4].try_into().unwrap()) as usize
        } else {
            let p = idx * 2;
            u16::from_le_bytes(self.lens_bytes[p..p + 2].try_into().unwrap()) as usize
        }
    }
}

pub(crate) fn validate_v6_header_and_dir(src: &[u8], hdr: &ParsedHeader) -> Result<(usize, usize)> {
    let (total_v6_payload, _) = parse_v6_group_dir(src, hdr)?;
    let esize = entry_size(hdr.version);
    let mut next_raw = 0usize;
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * esize;
        let rn = block_raw_size(src, hdr, i, next_raw)?;
        let _ = block_row_meta_raw(src, hdr, i, rn)?;
        let word = read_u32(src, p)?;
        let cn = (word & V4_SIZE_MASK) as usize;
        let csum = read_u32(src, p + 4)?;
        let kind = word >> 30;
        if !valid_kind_and_size(hdr.version, kind, cn, rn) || (!hdr.checksummed && csum != 0) {
            return Err(Error::InvalidFrame("block kind/size"));
        }
        next_raw += rn;
    }
    if next_raw != hdr.raw_size {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    let total_frame = hdr
        .payload_start
        .checked_add(usize::try_from(total_v6_payload).map_err(|_| Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    Ok((total_frame, hdr.raw_size))
}

pub(crate) fn parse_v6_frame<'a>(src: &'a [u8], hdr: &ParsedHeader) -> Result<Frame<'a>> {
    let (total_frame, _) = validate_v6_header_and_dir(src, hdr)?;
    if src.len() != total_frame {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    let (_, entries) = parse_v6_group_dir(src, hdr)?;
    let dict = decode_header_dict(src, hdr)?;
    let esize = entry_size(hdr.version);
    let mut row_index = if hdr.has_row_index {
        Some(Vec::with_capacity(hdr.count))
    } else {
        None
    };
    let mut infos: Vec<BlockInfo> = Vec::with_capacity(hdr.count);
    let mut next_raw = 0usize;
    let mut total_cn = 0usize;
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * esize;
        let rn = block_raw_size(src, hdr, i, next_raw)?;
        if let Some(word) = block_row_meta_raw(src, hdr, i, rn)? {
            if let Some(ref mut ri) = row_index {
                ri.push(word);
            }
        }
        let word = read_u32(src, p)?;
        let cn = (word & V4_SIZE_MASK) as usize;
        let checksum = read_u32(src, p + 4)?;
        let kind = word >> 30;
        infos.push(BlockInfo {
            raw_offset: next_raw,
            payload_offset: hdr.payload_start + total_cn,
            raw_size: rn,
            payload_size: cn,
            checksum,
            kind,
        });
        next_raw += rn;
        total_cn = total_cn.checked_add(cn).ok_or(Error::SizeOverflow)?;
    }

    let mut arena = vec![0u8; total_cn];
    let v6_payload = &src[hdr.payload_start..total_frame];
    for (gi, entry) in entries.iter().enumerate() {
        let g_start = entry.payload_rel_offset as usize;
        let g_end = entries
            .get(gi + 1)
            .map(|e| e.payload_rel_offset as usize)
            .unwrap_or(v6_payload.len());
        let g_slice = v6_payload.get(g_start..g_end).ok_or(Error::Truncated)?;
        let gh = ParsedGroupHeader::parse(g_slice, entry.start_block)?;
        if gh.count != entry.block_count as usize || gh.slots != entry.slot_count as usize {
            return Err(Error::InvalidFrame("v6 group header mismatch with dir"));
        }
        let sb = entry.start_block as usize;
        let mut write_cursors: Vec<usize> = (0..gh.count)
            .map(|bi| infos[sb + bi].payload_offset - hdr.payload_start)
            .collect();
        let mut cursor = gh.data_offset;
        for s in 0..=gh.slots {
            for bi in 0..gh.count {
                let slen = gh.stream_len(s, bi);
                let chunk = g_slice
                    .get(cursor..cursor.checked_add(slen).ok_or(Error::SizeOverflow)?)
                    .ok_or(Error::Truncated)?;
                cursor += slen;
                let dst_off = write_cursors[bi];
                let blk_end =
                    infos[sb + bi].payload_offset - hdr.payload_start + infos[sb + bi].payload_size;
                if dst_off.checked_add(slen).is_none_or(|e| e > blk_end) {
                    return Err(Error::InvalidFrame("v6 stream overflows block cn"));
                }
                arena[dst_off..dst_off + slen].copy_from_slice(chunk);
                write_cursors[bi] += slen;
            }
        }
        if cursor != g_slice.len() {
            return Err(Error::InvalidFrame("v6 group trailing bytes"));
        }
        for bi in 0..gh.count {
            let expected_end =
                infos[sb + bi].payload_offset - hdr.payload_start + infos[sb + bi].payload_size;
            if write_cursors[bi] != expected_end {
                return Err(Error::InvalidFrame("v6 block cn mismatch"));
            }
        }
    }

    let arena_ptr = arena.as_ptr();
    let mut blocks = Vec::with_capacity(hdr.count);
    for info in infos {
        let rel = info.payload_offset - hdr.payload_start;
        let payload = unsafe { std::slice::from_raw_parts(arena_ptr.add(rel), info.payload_size) };
        blocks.push(Block { info, payload });
    }
    Ok(Frame {
        blocks,
        dict,
        row_index,
        raw_size: hdr.raw_size,
        block_size: hdr.block_size,
        checksummed: hdr.checksummed,
        version: hdr.version,
        _owned_arena: arena,
    })
}

pub(crate) fn peek_v6_block0_stream0(src: &[u8], max_output: usize) -> Option<(u32, usize, &[u8])> {
    let hdr = parse_v1_to_v5_header(src, max_output).ok()?;
    if hdr.version != VERSION_V6 || hdr.count == 0 || src.len() < hdr.payload_start {
        return None;
    }
    let rn = block_raw_size(src, &hdr, 0, 0).ok()?;
    let word = read_u32(src, HEADER_SIZE).ok()?;
    let kind = word >> 30;
    let g_slice = src.get(hdr.payload_start..)?;
    let gh = ParsedGroupHeader::parse(g_slice, 0).ok()?;
    let s0_len = gh.stream_len(0, 0);
    let stream0 = g_slice.get(gh.data_offset..gh.data_offset.checked_add(s0_len)?)?;
    Some((kind, rn, stream0))
}

pub(crate) fn recover_v6_frame(src: &[u8], options: &DecodeOptions) -> Result<RecoveryReport> {
    options.validate()?;
    let hdr = parse_v1_to_v5_header(src, options.max_output)?;
    let (total_v6_payload, entries) = parse_v6_group_dir(src, &hdr)?;
    let dict = decode_header_dict(src, &hdr)?;
    let engine = Engine::new(options.simd)?;
    let mut infos: Vec<(usize, usize, usize, u32, u32)> = Vec::with_capacity(hdr.count);
    let mut next_raw = 0usize;
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * 8;
        let rn = block_raw_size(src, &hdr, i, next_raw)?;
        let word = read_u32(src, p)?;
        let cn = (word & V4_SIZE_MASK) as usize;
        let csum = read_u32(src, p + 4)?;
        let kind = word >> 30;
        infos.push((next_raw, rn, cn, csum, kind));
        next_raw += rn;
    }
    let mut data = Vec::with_capacity(hdr.raw_size);
    let mut recovered_blocks = Vec::new();
    let mut failed_blocks = Vec::new();
    let v6_end = hdr.payload_start.saturating_add(total_v6_payload as usize).min(src.len());
    let v6_payload = src.get(hdr.payload_start..v6_end).ok_or(Error::Truncated)?;
    let verify_csum = hdr.checksummed && options.verify_checksum;
    let mut scratch = Vec::new();

    for (gi, entry) in entries.iter().enumerate() {
        let g_start = entry.payload_rel_offset as usize;
        let g_end = entries.get(gi + 1).map(|e| e.payload_rel_offset as usize).unwrap_or(total_v6_payload as usize);
        let (sb, bc) = (entry.start_block as usize, entry.block_count as usize);
        let group_res = (|| -> Result<(ParsedGroupHeader<'_>, &[u8])> {
            let gs = v6_payload.get(g_start..g_end).ok_or(Error::Truncated)?;
            let gh = ParsedGroupHeader::parse(gs, entry.start_block)?;
            if gh.count != bc || gh.slots != entry.slot_count as usize {
                return Err(Error::InvalidFrame("v6 group header count/slot mismatch"));
            }
            let mut total_s = gh.data_offset;
            for s in 0..=gh.slots {
                for bi in 0..gh.count {
                    total_s = total_s.checked_add(gh.stream_len(s, bi)).ok_or(Error::SizeOverflow)?;
                }
            }
            if total_s > gs.len() {
                return Err(Error::Truncated);
            }
            Ok((gh, gs))
        })();

        let (gh, gs) = match group_res {
            Ok(v) => v,
            Err(err) => {
                for bi in 0..bc {
                    let idx = sb + bi;
                    let (raw_offset, raw_size, _, _, _) = infos[idx];
                    failed_blocks.push(FailedBlock { index: idx, raw_offset, raw_size, reason: err.to_string() });
                }
                continue;
            }
        };

        let mut stream_bases = Vec::with_capacity(gh.slots + 1);
        let mut cur = gh.data_offset;
        for s in 0..=gh.slots {
            stream_bases.push(cur);
            for bi in 0..gh.count {
                cur += gh.stream_len(s, bi);
            }
        }
        for bi in 0..bc {
            let idx = sb + bi;
            let (raw_offset, raw_size, cn, csum, kind) = infos[idx];
            let mut blk_buf = Vec::with_capacity(cn);
            for (s, &s_base) in stream_bases.iter().enumerate() {
                let off: usize = s_base + (0..bi).map(|prev| gh.stream_len(s, prev)).sum::<usize>();
                let slen = gh.stream_len(s, bi);
                blk_buf.extend_from_slice(&gs[off..off + slen]);
            }
            scratch.resize(raw_size, 0);
            let dec_res = if blk_buf.len() != cn {
                Err(Error::InvalidFrame("v6 block cn mismatch"))
            } else {
                block::decode_with_dict(&blk_buf, kind, &mut scratch, engine, true, &dict)
            };
            match dec_res {
                Ok(()) if !verify_csum || crc32c(&scratch) == csum => {
                    let output_offset = data.len();
                    data.extend_from_slice(&scratch);
                    recovered_blocks.push(RecoveredBlock { index: idx, raw_offset, raw_size, output_offset, kind });
                }
                Ok(()) => failed_blocks.push(FailedBlock { index: idx, raw_offset, raw_size, reason: format!("checksum mismatch in block {idx}") }),
                Err(error) => failed_blocks.push(FailedBlock { index: idx, raw_offset, raw_size, reason: error.to_string() }),
            }
        }
    }
    Ok(RecoveryReport { data, total_blocks: hdr.count, expected_raw_size: hdr.raw_size, recovered_blocks, failed_blocks })
}

#[path = "v6_query.rs"]
mod v6_query;
pub(crate) use v6_query::{
    extract_block_v6_into, extract_row_range_v6_indexed, fast_frame_row_count, pread_v6_for_query,
    reassemble_v6_for_query_warm,
};

