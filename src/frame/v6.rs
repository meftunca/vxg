//! `VERSION_V6` (`6`): Column-grouped macro-block layout with dynamic `~1 MiB`/column sizing,
//! cumulative $O(1)$ per-block stream offset tables, trailing backup group headers (`b"6GXV"`),
//! per-block directory redundancy, and zero-copy in-place query execution.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::checksum::crc32c;
use crate::{block, columns, Error, Result};

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
pub const GROUP_TRAILER_MAGIC: &[u8; 4] = b"6GXV";
pub const GROUP_CUM_FLAG: u16 = 0x4000;
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

#[derive(Clone)]
pub(crate) struct BlockSplit<'a> {
    pub payload: &'a [u8],
    pub stream0_len: usize,
    pub spans: Vec<(usize, usize)>,
    pub word: u32,
    pub csum: u32,
    pub raw_size: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn append_serialized_v6_group(
    v6_payload: &mut Vec<u8>,
    start_block: u32,
    slots: usize,
    b_meta: &[(u32, u32, u32)],
    s0_slices: &[&[u8]],
    col_slices_per_stream: &[Vec<&[u8]>],
) {
    let count = s0_slices.len();
    debug_assert_eq!(count, b_meta.len());
    debug_assert_eq!(slots, col_slices_per_stream.len());
    let slot_flags = (slots as u16) | GROUP_CUM_FLAG;
    let bdir_len = count * 12;
    let cum_len = (slots + 1) * count * 4;
    let gh_len = 16 + bdir_len + cum_len;

    let mut gh = Vec::with_capacity(gh_len);
    gh.extend_from_slice(GROUP_MAGIC);
    gh.extend_from_slice(&0u32.to_le_bytes());
    gh.extend_from_slice(&start_block.to_le_bytes());
    gh.extend_from_slice(&(count as u16).to_le_bytes());
    gh.extend_from_slice(&slot_flags.to_le_bytes());

    for &(word, csum, rn) in b_meta {
        gh.extend_from_slice(&word.to_le_bytes());
        gh.extend_from_slice(&csum.to_le_bytes());
        gh.extend_from_slice(&rn.to_le_bytes());
    }

    let mut acc0 = 0u32;
    for s0 in s0_slices {
        acc0 += s0.len() as u32;
        gh.extend_from_slice(&acc0.to_le_bytes());
    }
    let mut total_col_bytes = 0usize;
    for s_cols in col_slices_per_stream {
        let mut acc = 0u32;
        for sl in s_cols {
            acc += sl.len() as u32;
            gh.extend_from_slice(&acc.to_le_bytes());
        }
        total_col_bytes += acc as usize;
    }
    let hdr_crc = crc32c(&gh[8..]);
    gh[4..8].copy_from_slice(&hdr_crc.to_le_bytes());

    v6_payload.reserve(gh_len + (acc0 as usize) + total_col_bytes + gh_len + 8);
    v6_payload.extend_from_slice(&gh);
    for s0 in s0_slices {
        v6_payload.extend_from_slice(s0);
    }
    for s_cols in col_slices_per_stream {
        for sl in s_cols {
            v6_payload.extend_from_slice(sl);
        }
    }
    // Trailing backup copy of the group header + `(gh_len as u32)` + `b"6GXV"` trailer.
    v6_payload.extend_from_slice(&gh);
    v6_payload.extend_from_slice(&(gh_len as u32).to_le_bytes());
    v6_payload.extend_from_slice(GROUP_TRAILER_MAGIC);
}

#[inline]
fn make_block_split(payload: &[u8], word: u32, csum: u32, raw_size: u32) -> BlockSplit<'_> {
    if (word >> 30) == block::COLUMN {
        if let Some((s0, spans)) = columns::split_grammar_packet_streams(payload) {
            return BlockSplit {
                payload,
                stream0_len: s0,
                spans,
                word,
                csum,
                raw_size,
            };
        }
    }
    BlockSplit {
        payload,
        stream0_len: payload.len(),
        spans: Vec::new(),
        word,
        csum,
        raw_size,
    }
}

pub(crate) fn repack_v4_v5_to_v6(v5_bytes: &[u8], bound: usize) -> Result<Vec<u8>> {
    let hdr = parse_v1_to_v5_header(v5_bytes, usize::MAX)?;
    if hdr.version < super::VERSION_V4 || hdr.version > super::VERSION_V5 {
        return Ok(v5_bytes.to_vec());
    }
    let esize = entry_size(hdr.version);
    let mut block_ranges: Vec<(usize, usize, u32, u32, u32)> = Vec::with_capacity(hdr.count);
    let mut next_payload = hdr.payload_start;
    let mut next_raw = 0usize;
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * esize;
        let rn = block_raw_size(v5_bytes, &hdr, i, next_raw)?;
        next_raw += rn;
        let word = read_u32(v5_bytes, p)?;
        let cn = (word & V4_SIZE_MASK) as usize;
        let csum = read_u32(v5_bytes, p + 4)?;
        let end = next_payload.checked_add(cn).ok_or(Error::SizeOverflow)?;
        if end > v5_bytes.len() {
            return Err(Error::Truncated);
        }
        block_ranges.push((next_payload, end, word, csum, rn as u32));
        next_payload = end;
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
                    let end = (base + 16).min(br_slice.len());
                    for i in base..end {
                        let (st, en, word, csum, raw_size) = br_slice[i];
                        let item = make_block_split(&v5_bytes[st..en], word, csum, raw_size);
                        unsafe {
                            *(splits_ptr as *mut BlockSplit<'_>).add(i) = item;
                        }
                    }
                }
            })?;
        }
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

    let group_dir_bytes = 16 + 16 * groups.len();
    let prefix_len = hdr.group_dir_offset + group_dir_bytes;
    let mut out: Vec<u8> = Vec::with_capacity(v5_bytes.len() + group_dir_bytes + groups.len() * 16_384);
    out.resize(prefix_len, 0);
    let mut dir_entries: Vec<GroupDirEntry> = Vec::with_capacity(groups.len());
    for &(start, count, slots) in &groups {
        let rel_off = (out.len() - prefix_len) as u64;
        dir_entries.push(GroupDirEntry {
            payload_rel_offset: rel_off,
            start_block: start as u32,
            block_count: count as u16,
            slot_count: slots as u16,
        });
        let mut b_meta = Vec::with_capacity(count);
        let mut s0_slices = Vec::with_capacity(count);
        let mut col_slices_per_stream: Vec<Vec<&[u8]>> =
            (0..slots).map(|_| Vec::with_capacity(count)).collect();
        for b in &splits[start..start + count] {
            b_meta.push((b.word, b.csum, b.raw_size));
            s0_slices.push(&b.payload[..b.stream0_len]);
            for (c, &(s, e)) in b.spans.iter().enumerate() {
                col_slices_per_stream[c].push(&b.payload[s..e]);
            }
        }
        append_serialized_v6_group(
            &mut out,
            start as u32,
            slots,
            &b_meta,
            &s0_slices,
            &col_slices_per_stream,
        );
    }

    let total_len = out.len();
    if total_len > bound {
        return Ok(v5_bytes.to_vec());
    }
    let v6_payload_len = (total_len - prefix_len) as u64;
    out[..hdr.group_dir_offset].copy_from_slice(&v5_bytes[..hdr.group_dir_offset]);
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
    let dir_crc = crc32c(&entries_raw);
    let gd = &mut out[hdr.group_dir_offset..prefix_len];
    gd[0..4].copy_from_slice(&(dir_entries.len() as u32).to_le_bytes());
    gd[4..12].copy_from_slice(&v6_payload_len.to_le_bytes());
    gd[12..16].copy_from_slice(&dir_crc.to_le_bytes());
    gd[16..].copy_from_slice(&entries_raw);
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

pub(crate) fn parse_v6_block_infos_and_ri(
    src: &[u8],
    hdr: &ParsedHeader,
) -> Result<(Vec<BlockInfo>, Option<Vec<u32>>)> {
    let esize = entry_size(hdr.version);
    let mut row_index = if hdr.has_row_index {
        Some(Vec::with_capacity(hdr.count))
    } else {
        None
    };
    let mut infos = Vec::with_capacity(hdr.count);
    let mut next_raw = 0usize;
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
        if !valid_kind_and_size(hdr.version, kind, cn, rn) || (!hdr.checksummed && checksum != 0) {
            return Err(Error::InvalidFrame("block kind/size"));
        }
        infos.push(BlockInfo {
            raw_offset: next_raw,
            payload_offset: 0,
            raw_size: rn,
            payload_size: cn,
            checksum,
            kind,
        });
        next_raw += rn;
    }
    if next_raw != hdr.raw_size {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    Ok((infos, row_index))
}

pub(crate) fn validate_v6_header_and_dir(src: &[u8], hdr: &ParsedHeader) -> Result<(usize, usize)> {
    let (total_v6_payload, _) = parse_v6_group_dir(src, hdr)?;
    let _ = parse_v6_block_infos_and_ri(src, hdr)?;
    let total_frame = hdr
        .payload_start
        .checked_add(usize::try_from(total_v6_payload).map_err(|_| Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    Ok((total_frame, hdr.raw_size))
}

/// Zero-copy `VERSION_V6` parser: binds `Block::payload` to `Stream 0` and `Block::col_slices`
/// directly to `Stream 1..=S` without copying or reassembling blocks into `v5` layout.
pub(crate) fn parse_v6_frame<'a>(src: &'a [u8], hdr: &ParsedHeader) -> Result<Frame<'a>> {
    parse_v6_frame_selective(src, hdr, None)
}

pub(crate) fn parse_v6_frame_selective<'a>(
    src: &'a [u8],
    hdr: &ParsedHeader,
    targets: Option<&[crate::ops::ColTarget<'_>]>,
) -> Result<Frame<'a>> {
    let (total_v6_payload, entries) = parse_v6_group_dir(src, hdr)?;
    let total_frame = hdr
        .payload_start
        .checked_add(usize::try_from(total_v6_payload).map_err(|_| Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    if src.len() != total_frame {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    let dict = decode_header_dict(src, hdr)?;
    let (infos, row_index) = parse_v6_block_infos_and_ri(src, hdr)?;

    let total_col_refs: usize = entries
        .iter()
        .map(|e| (e.block_count as usize) * (e.slot_count as usize))
        .sum();
    let mut owned_col_slices: Vec<&'static [u8]> = Vec::with_capacity(total_col_refs);
    let mut blocks: Vec<Block<'a>> = Vec::with_capacity(hdr.count);
    let v6_payload = &src[hdr.payload_start..total_frame];

    for (gi, entry) in entries.iter().enumerate() {
        let g_start = entry.payload_rel_offset as usize;
        let g_end = entries
            .get(gi + 1)
            .map(|e| e.payload_rel_offset as usize)
            .unwrap_or(v6_payload.len());
        let g_slice = v6_payload.get(g_start..g_end).ok_or(Error::Truncated)?;
        let gh = if targets.is_some() {
            ParsedGroupHeader::parse_for_block(g_slice, entry.start_block, 0)?
        } else {
            ParsedGroupHeader::parse(g_slice, entry.start_block)?
        };
        if gh.count != entry.block_count as usize || gh.slots != entry.slot_count as usize {
            return Err(Error::InvalidFrame("v6 group header mismatch with dir"));
        }
        let sb = entry.start_block as usize;
        let mut stream_bases = [0usize; 65];
        let mut cur = gh.data_offset;
        for s in 0..=gh.slots {
            stream_bases[s] = cur;
            let st_len = gh.stream_total_len(s);
            cur = cur.checked_add(st_len).ok_or(Error::SizeOverflow)?;
        }
        let expected_trailer = if gh.is_cum { gh.data_offset + 8 } else { 0 };
        if cur > g_slice.len()
            || (cur != g_slice.len() && cur + expected_trailer != g_slice.len())
        {
            return Err(Error::InvalidFrame("v6 group trailing bytes"));
        }
        let req_mask: u64 = match targets {
            Some(t_list) if gh.slots > 0 => {
                if t_list.is_empty() {
                    0
                } else {
                    let mut m = 0u64;
                    let mut ok = true;
                    for bi in [0usize, gh.count - 1] {
                        let (st, en) = gh.stream_slice_range(0, bi);
                        let s0 = g_slice
                            .get(stream_bases[0] + st..stream_bases[0] + en)
                            .ok_or(Error::Truncated)?;
                        match columns::resolve_stream0_required_slots(s0, t_list) {
                            Some(bits) => m |= bits,
                            None => {
                                ok = false;
                                break;
                            }
                        }
                        if gh.count == 1 {
                            break;
                        }
                    }
                    if ok { m } else { u64::MAX }
                }
            }
            _ => u64::MAX,
        };

        for bi in 0..gh.count {
            let mut info = infos[sb + bi];
            let (s0_st, s0_en) = gh.stream_slice_range(0, bi);
            if s0_en < s0_st {
                return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
            }
            let s0_off_in_group = stream_bases[0] + s0_st;
            let s0_len = s0_en - s0_st;
            let s0_slice = g_slice
                .get(s0_off_in_group..s0_off_in_group + s0_len)
                .ok_or(Error::Truncated)?;
            info.payload_offset = hdr.payload_start + g_start + s0_off_in_group;
            let mut sum_cn = s0_len;
            let col_slices: &'a [&'a [u8]] = if gh.slots > 0 {
                let base_idx = owned_col_slices.len();
                for s in 0..gh.slots {
                    if ((req_mask >> s) & 1) == 0 {
                        owned_col_slices.push(&[]);
                        continue;
                    }
                    let (c_st, c_en) = gh.stream_slice_range(s + 1, bi);
                    if c_en < c_st {
                        return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
                    }
                    let c_off = stream_bases[s + 1] + c_st;
                    let c_len = c_en - c_st;
                    sum_cn += c_len;
                    let c_slice = g_slice.get(c_off..c_off + c_len).ok_or(Error::Truncated)?;
                    let static_c: &'static [u8] = unsafe { std::mem::transmute(c_slice) };
                    owned_col_slices.push(static_c);
                }
                unsafe {
                    std::slice::from_raw_parts(
                        owned_col_slices.as_ptr().add(base_idx) as *const &'a [u8],
                        gh.slots,
                    )
                }
            } else {
                &[]
            };
            if req_mask == u64::MAX && sum_cn != info.payload_size {
                return Err(Error::InvalidFrame("v6 block cn mismatch"));
            }
            blocks.push(Block {
                info,
                payload: s0_slice,
                col_slices,
            });
        }
    }

    Ok(Frame {
        blocks,
        dict,
        row_index,
        raw_size: hdr.raw_size,
        block_size: hdr.block_size,
        checksummed: hdr.checksummed,
        version: hdr.version,
        _owned_arena: Vec::new(),
        _owned_col_slices: owned_col_slices,
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

#[path = "v6_point.rs"]
pub(crate) mod v6_point;
pub(crate) use v6_point::{
    extract_row_range_v6_indexed, try_extract_block_v6_into, ParsedGroupHeader,
};

#[path = "v6_query.rs"]
mod v6_query;
pub(crate) use v6_query::{
    decode_v6_direct_into, fast_frame_row_count, parse_v6_for_query,
};

#[path = "v6_stream.rs"]
mod v6_stream;
pub(crate) use v6_stream::V6StreamBuilder;
