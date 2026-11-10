//! `VERSION_V6` (`6`): Column-grouped macro-block layout with dynamic `~1 MiB`/column sizing,
//! canonical `Stream 0` header deduplication + separate `raw_rows` stream (`GROUP_V2_FLAG`),
//! per-stream-slice hardware `CRC32C` verification, cumulative $O(1)$ per-block stream offset
//! tables, trailing backup group headers (`b"6GXV"`), and zero-copy in-place query execution.

use crate::checksum::crc32c;
use crate::{columns, Error, Result};

use super::v5::{
    block_raw_size, block_row_meta_raw, parse_v1_to_v5_header, unpack_block_row_meta, ParsedHeader,
};
use super::{
    entry_size, read_u32, read_u64, valid_kind_and_size, BlockInfo, HEADER_SIZE, V4_SIZE_MASK,
};

pub const VERSION_V6: u16 = 6;
pub const GROUP_MAGIC: &[u8; 4] = b"VXG6";
pub const GROUP_TRAILER_MAGIC: &[u8; 4] = b"6GXV";
pub const GROUP_CUM_FLAG: u16 = 0x4000;
pub const GROUP_V2_FLAG: u16 = 0x2000;
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

#[inline]
pub(crate) fn uleb_len_usize(mut v: usize) -> usize {
    let mut n = 1usize;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

#[inline]
pub(crate) fn put_uleb_usize(out: &mut Vec<u8>, mut v: usize) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
pub(crate) fn compute_v6_dir_crc(prefix_before_crc: &[u8], entries_raw: &[u8]) -> u32 {
    crc32c(prefix_before_crc) ^ crc32c(entries_raw).rotate_left(1)
}

#[derive(Clone, Copy)]
pub(crate) struct ParsedGroupHeader<'a> {
    pub start_block: u32,
    pub count: usize,
    pub slots: usize,
    pub num_streams: usize,
    pub is_cum: bool,
    pub is_v2: bool,
    pub use_u32: bool,
    pub block_dir_bytes: &'a [u8],
    pub lens_bytes: &'a [u8],
    pub canon_s0: &'a [u8],
    pub data_offset: usize,
}

impl<'a> ParsedGroupHeader<'a> {
    fn parse_raw(group_slice: &'a [u8]) -> Result<(Self, u32, &'a [u8])> {
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
        let (is_cum, is_v2, slots) = (
            (slot_flags & GROUP_CUM_FLAG) != 0,
            (slot_flags & GROUP_V2_FLAG) != 0,
            (slot_flags & 0x1fff) as usize,
        );
        let use_u32 = is_cum || (slot_flags & 0x8000) != 0;
        if count == 0 || slots > 64 {
            return Err(Error::InvalidFrame("bad v6 group header fields"));
        }
        let (fixed_hdr, canon_s0_len) = if is_v2 {
            if group_slice.len() < 18 {
                return Err(Error::Truncated);
            }
            (18usize, u16::from_le_bytes([group_slice[16], group_slice[17]]) as usize)
        } else {
            (16usize, 0usize)
        };
        let num_streams = if is_v2 && slots > 0 { slots + 2 } else { slots + 1 };
        let bdir_len = if is_cum { count * 12 } else { 0 };
        let num_lens = num_streams.checked_mul(count).ok_or(Error::SizeOverflow)?;
        let lens_byte_len = num_lens * if use_u32 { 4 } else { 2 };
        let canon_start = fixed_hdr
            .checked_add(bdir_len)
            .and_then(|v| v.checked_add(lens_byte_len))
            .ok_or(Error::SizeOverflow)?;
        let data_offset = canon_start.checked_add(canon_s0_len).ok_or(Error::SizeOverflow)?;
        let hdr_body = group_slice.get(8..data_offset).ok_or(Error::Truncated)?;
        Ok((
            Self {
                start_block: sb,
                count,
                slots,
                num_streams,
                is_cum,
                is_v2,
                use_u32,
                block_dir_bytes: &group_slice[fixed_hdr..fixed_hdr + bdir_len],
                lens_bytes: &group_slice[fixed_hdr + bdir_len..canon_start],
                canon_s0: &group_slice[canon_start..data_offset],
                data_offset,
            },
            expected_crc,
            hdr_body,
        ))
    }

    pub fn parse_any_start(group_slice: &'a [u8]) -> Result<Self> {
        let (gh, expected_crc, hdr_body) = Self::parse_raw(group_slice)?;
        if crc32c(hdr_body) != expected_crc {
            return Err(Error::InvalidFrame("v6 group header crc mismatch"));
        }
        if gh.is_cum {
            for s in 0..gh.num_streams {
                let mut prev = 0usize;
                for bi in 0..gh.count {
                    let (st, en) = gh.stream_slice_range(s, bi);
                    if st != prev || en < st {
                        return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
                    }
                    prev = en;
                }
            }
        }
        Ok(gh)
    }

    pub fn parse(group_slice: &'a [u8], expected_start: u32) -> Result<Self> {
        let gh = Self::parse_any_start(group_slice)?;
        if gh.start_block != expected_start {
            return Err(Error::InvalidFrame("bad v6 group start_block"));
        }
        Ok(gh)
    }

    pub fn parse_for_block(group_slice: &'a [u8], expected_start: u32, bi: usize) -> Result<Self> {
        let (gh, expected_crc, hdr_body) = Self::parse_raw(group_slice)?;
        if gh.start_block != expected_start || bi >= gh.count {
            return Err(Error::InvalidFrame("bad v6 group header fields"));
        }
        if crc32c(hdr_body) != expected_crc {
            return Err(Error::InvalidFrame("v6 group header crc mismatch"));
        }
        Ok(gh)
    }

    pub fn parse_resilient(group_slice: &'a [u8], expected_start: u32) -> Result<Self> {
        if let Ok(gh) = Self::parse(group_slice, expected_start) {
            if gh.data_offset.saturating_add(gh.total_streams_bytes()) <= group_slice.len() {
                return Ok(gh);
            }
        }
        if group_slice.len() >= 24 && &group_slice[group_slice.len() - 4..] == GROUP_TRAILER_MAGIC {
            let gh_len = read_u32(group_slice, group_slice.len() - 8)? as usize;
            if gh_len >= 16 && gh_len + 8 <= group_slice.len() {
                let bk_start = group_slice.len() - 8 - gh_len;
                if let Ok(gh) = Self::parse(&group_slice[bk_start..bk_start + gh_len], expected_start) {
                    if gh.data_offset.saturating_add(gh.total_streams_bytes()) <= bk_start {
                        return Ok(gh);
                    }
                }
            }
        }
        Err(Error::InvalidFrame("v6 group header and backup both corrupt"))
    }

    #[inline]
    pub fn unwrap_verified_slice<'b>(raw: &'b [u8], is_v2: bool, block: usize) -> Result<&'b [u8]> {
        if !is_v2 || raw.is_empty() {
            return Ok(raw);
        }
        if raw.len() < 4 {
            return Err(Error::InvalidFrame("v6 stream slice shorter than crc"));
        }
        let (body, crc_bytes) = raw.split_at(raw.len() - 4);
        if columns::v6_verify_slice_crc() {
            let exp = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
            if crc32c(body) != exp {
                return Err(Error::Checksum { block });
            }
        }
        Ok(body)
    }

    #[inline]
    pub fn verify_block_cn(&self, bi: usize, rn: usize, exp_s0_len: usize, expected_cn: usize) -> bool {
        if !self.is_v2 {
            return (0..self.num_streams).map(|s| self.stream_len(s, bi)).sum::<usize>() == expected_cn;
        }
        if self.slots == 0 {
            return self.stream_len(0, bi).checked_sub(4) == Some(expected_cn);
        }
        let s0_wire = self.stream_len(0, bi);
        if s0_wire != 0 && s0_wire < 4 {
            return false;
        }
        let mut total = if s0_wire == exp_s0_len + 4 {
            exp_s0_len
        } else {
            let Some(rel) = self.canon_s0.get(1..).and_then(|s| s.iter().position(|&b| (b & 0x80) == 0)) else {
                return false;
            };
            let Some(tail_len) = exp_s0_len.checked_sub(2 + rel) else { return false };
            1 + uleb_len_usize(rn) + tail_len
        };
        for c in 0..self.slots {
            let cw = self.stream_len(c + 1, bi);
            if cw < 4 { return false; }
            total += cw - 4;
        }
        let rrw = self.stream_len(self.slots + 1, bi);
        if rrw > 0 {
            if rrw < 4 { return false; }
            total += rrw - 4;
        }
        total == expected_cn
    }

    pub fn resolve_req_mask(
        &self,
        g_slice: &[u8],
        stream_bases: &[usize; 66],
        targets: Option<&[crate::ops::ColTarget<'_>]>,
    ) -> u64 {
        let Some(t_list) = targets else { return u64::MAX };
        if self.slots == 0 || self.count == 0 { return u64::MAX; }
        if t_list.is_empty() { return 0; }
        if self.is_v2 && !self.canon_s0.is_empty() && self.stream_total_len(0) == 0 {
            return columns::resolve_stream0_required_slots(self.canon_s0, t_list).unwrap_or(u64::MAX);
        }
        let mut mask = 0u64;
        for bi in [0usize, self.count - 1] {
            let (st, en) = self.stream_slice_range(0, bi);
            let Some(raw) = g_slice.get(stream_bases[0] + st..stream_bases[0] + en) else { return u64::MAX };
            let s0_hdr = if self.is_v2 && !self.canon_s0.is_empty() && (raw.is_empty() || raw.len() == 6 || raw.first() == Some(&0)) {
                self.canon_s0
            } else if self.is_v2 && raw.len() >= 4 {
                &raw[..raw.len() - 4]
            } else {
                raw
            };
            match columns::resolve_stream0_required_slots(s0_hdr, t_list) {
                Some(m) => mask |= m,
                None => return u64::MAX,
            }
            if self.count == 1 { break; }
        }
        mask
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resolve_block_s0(
        &self,
        g_slice: &'a [u8],
        stream_bases: &[usize; 66],
        bi: usize,
        blk_idx: usize,
        stitch_arena: &mut Vec<u8>,
        stitched_ranges: &mut Vec<(usize, usize, usize)>,
        rc_cache: &mut Vec<(u16, usize, usize)>,
        block_vec_idx: usize,
    ) -> Result<(&'a [u8], usize, usize)> {
        let (s0_st, s0_en) = self.stream_slice_range(0, bi);
        if s0_en < s0_st {
            return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
        }
        if !self.is_v2 {
            let s0_off = stream_bases[0] + s0_st;
            let s0_slice = g_slice.get(s0_off..stream_bases[0] + s0_en).ok_or(Error::Truncated)?;
            return Ok((s0_slice, s0_off, s0_slice.len()));
        }
        let s0_raw = g_slice.get(stream_bases[0] + s0_st..stream_bases[0] + s0_en).ok_or(Error::Truncated)?;
        let mut s0_in_arena = None;
        let (s0_hdr, s0_off, exp_len) = if s0_raw.is_empty() {
            if self.canon_s0.is_empty() {
                return Err(Error::InvalidFrame("missing v6 canon_s0"));
            }
            (self.canon_s0, self.data_offset - self.canon_s0.len(), self.canon_s0.len())
        } else {
            let body = Self::unwrap_verified_slice(s0_raw, true, blk_idx)?;
            if body.len() == 2 {
                let rc = u16::from_le_bytes([body[0], body[1]]);
                let (st, en) = if let Some(&(_, st, en)) = rc_cache.iter().find(|&&(r, _, _)| r == rc) {
                    (st, en)
                } else {
                    let (st, en) = columns::expand_v6_compact_s0(self.canon_s0, rc as usize, stitch_arena)
                        .ok_or(Error::InvalidFrame("bad v6 compact s0"))?;
                    rc_cache.push((rc, st, en));
                    (st, en)
                };
                s0_in_arena = Some((st, en));
                (&[][..], stream_bases[0] + s0_st, en - st)
            } else if body.first() == Some(&0) {
                let (st, en) = columns::expand_v6_tpl_s0(self.canon_s0, &body[1..], stitch_arena)
                    .ok_or(Error::InvalidFrame("bad v6 tpl s0"))?;
                s0_in_arena = Some((st, en));
                (&[][..], stream_bases[0] + s0_st, en - st)
            } else {
                (body, stream_bases[0] + s0_st, body.len())
            }
        };
        if self.slots > 0 {
            let (rr_st, rr_en) = self.stream_slice_range(self.slots + 1, bi);
            if rr_en < rr_st {
                return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
            }
            if rr_en > rr_st {
                let rr_raw = g_slice.get(stream_bases[self.slots + 1] + rr_st..stream_bases[self.slots + 1] + rr_en).ok_or(Error::Truncated)?;
                let rr_body = Self::unwrap_verified_slice(rr_raw, true, blk_idx)?;
                let st = stitch_arena.len();
                if let Some((a_st, a_en)) = s0_in_arena {
                    stitch_arena.extend_from_within(a_st..a_en);
                } else {
                    stitch_arena.extend_from_slice(s0_hdr);
                }
                stitch_arena.extend_from_slice(rr_body);
                stitched_ranges.push((block_vec_idx, st, stitch_arena.len()));
                return Ok((&[], s0_off, exp_len));
            }
        }
        if let Some((a_st, a_en)) = s0_in_arena {
            stitched_ranges.push((block_vec_idx, a_st, a_en));
        }
        Ok((s0_hdr, s0_off, exp_len))
    }

    #[inline]
    pub fn stream_slice_range(&self, stream_idx: usize, block_in_group: usize) -> (usize, usize) {
        let base = stream_idx * self.count;
        if self.is_cum {
            let start = if block_in_group == 0 {
                0
            } else {
                let p = (base + block_in_group - 1) * 4;
                u32::from_le_bytes(self.lens_bytes[p..p + 4].try_into().unwrap()) as usize
            };
            let ep = (base + block_in_group) * 4;
            let end = u32::from_le_bytes(self.lens_bytes[ep..ep + 4].try_into().unwrap()) as usize;
            (start, end)
        } else {
            let mut off = 0usize;
            for b in 0..block_in_group {
                off += self.stream_len(stream_idx, b);
            }
            (off, off + self.stream_len(stream_idx, block_in_group))
        }
    }

    #[inline]
    pub fn stream_total_len(&self, stream_idx: usize) -> usize {
        if self.count == 0 { 0 } else { self.stream_slice_range(stream_idx, self.count - 1).1 }
    }

    #[inline]
    pub fn total_streams_bytes(&self) -> usize {
        (0..self.num_streams).map(|s| self.stream_total_len(s)).sum()
    }

    #[inline]
    pub fn stream_len(&self, stream_idx: usize, block_in_group: usize) -> usize {
        if self.is_cum {
            let (st, en) = self.stream_slice_range(stream_idx, block_in_group);
            en - st
        } else {
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

    #[inline]
    pub fn backup_block_meta(&self, bi: usize) -> Option<(u32, u32, usize)> {
        if !self.is_cum {
            return None;
        }
        let p = bi * 12;
        let word = u32::from_le_bytes(self.block_dir_bytes[p..p + 4].try_into().ok()?);
        let csum = u32::from_le_bytes(self.block_dir_bytes[p + 4..p + 8].try_into().ok()?);
        let rn = u32::from_le_bytes(self.block_dir_bytes[p + 8..p + 12].try_into().ok()?) as usize;
        Some((word, csum, rn))
    }
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
    let v2_crc = compute_v6_dir_crc(&src[..p + 12], entries_slice);
    if v2_crc != expected_crc && crc32c(entries_slice) != expected_crc {
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
        expected_start = expected_start.checked_add(bc as u32).ok_or(Error::SizeOverflow)?;
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
    let mut row_index = hdr.has_row_index.then(|| Vec::with_capacity(hdr.count));
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

pub(crate) fn peek_v6_block0_stream0(src: &[u8], max_output: usize) -> Option<(u32, usize, &[u8])> {
    let hdr = parse_v1_to_v5_header(src, max_output).ok()?;
    if hdr.version != VERSION_V6 || hdr.count == 0 || src.len() < hdr.payload_start {
        return None;
    }
    let rn = block_raw_size(src, &hdr, 0, 0).ok()?;
    let kind = read_u32(src, HEADER_SIZE).ok()? >> 30;
    let g_slice = src.get(hdr.payload_start..)?;
    let gh = ParsedGroupHeader::parse_for_block(g_slice, 0, 0).ok()?;
    let s0_len = gh.stream_len(0, 0);
    let raw = g_slice.get(gh.data_offset..gh.data_offset.checked_add(s0_len)?)?;
    if gh.is_v2 && !gh.canon_s0.is_empty() && (s0_len == 0 || s0_len == 6 || raw.first() == Some(&0)) {
        return Some((kind, rn, gh.canon_s0));
    }
    let stream0 = ParsedGroupHeader::unwrap_verified_slice(raw, gh.is_v2, 0).ok()?;
    Some((kind, rn, stream0))
}

pub(crate) fn fast_frame_row_count(frame_bytes: &[u8]) -> Option<Result<usize>> {
    let first_res = crate::mmap::pread::is_first_page_resident(frame_bytes);
    if !is_v6_frame(frame_bytes) && first_res {
        return None;
    }
    let cold = (!first_res)
        .then(|| crate::mmap::peek_cold_prefix(frame_bytes, 262_144))
        .flatten();
    let probe = cold.as_deref().unwrap_or(frame_bytes);
    let hdr = parse_v1_to_v5_header(probe, usize::MAX).ok()?;
    if !hdr.has_row_index || probe.len() < hdr.payload_start {
        return None;
    }
    if hdr.version == VERSION_V6 {
        let (total_frame, _) = validate_v6_header_and_dir(probe, &hdr).ok()?;
        if total_frame != frame_bytes.len() {
            return None;
        }
    } else if hdr.version == crate::frame::VERSION_V5 {
        let esize = entry_size(hdr.version);
        let (mut next_payload, mut next_raw) = (hdr.payload_start, 0usize);
        for i in 0..hdr.count {
            let p = HEADER_SIZE + i * esize;
            let rn = block_raw_size(probe, &hdr, i, next_raw).ok()?;
            let word = read_u32(probe, p).ok()?;
            let (cn, csum, kind) = (
                (word & V4_SIZE_MASK) as usize,
                read_u32(probe, p + 4).ok()?,
                word >> 30,
            );
            if !valid_kind_and_size(hdr.version, kind, cn, rn) || (!hdr.checksummed && csum != 0) {
                return None;
            }
            next_payload = next_payload.checked_add(cn)?;
            next_raw += rn;
        }
        if next_raw != hdr.raw_size || next_payload != frame_bytes.len() {
            return None;
        }
    } else {
        return None;
    }
    let (mut total, mut prev_open, mut next_raw) = (0usize, false, 0usize);
    for i in 0..hdr.count {
        let rn = block_raw_size(probe, &hdr, i, next_raw).ok()?;
        next_raw += rn;
        let word = block_row_meta_raw(probe, &hdr, i, rn).ok()??;
        let (rows, ends_nl) = unpack_block_row_meta(word);
        if rows > 0 {
            total = total.checked_add(if prev_open { rows - 1 } else { rows })?;
            prev_open = !ends_nl;
        }
    }
    Some(Ok(total))
}

#[path = "v6_point.rs"]
pub(crate) mod v6_point;
pub(crate) use v6_point::{extract_row_range_v6_indexed, try_extract_block_v6_into};

#[path = "v6_query.rs"]
mod v6_query;
pub(crate) use v6_query::{decode_v6_direct_into, parse_v6_for_query, parse_v6_frame};

#[path = "v6_stream.rs"]
mod v6_stream;
pub(crate) use v6_stream::{repack_v4_v5_to_v6, V6StreamBuilder};
