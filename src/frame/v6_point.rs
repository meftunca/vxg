//! `VERSION_V6` group header parser (`ParsedGroupHeader`), $O(1)$-offset point access
//! (`extract_block_v6_into`, `extract_row_range_v6_indexed`), concurrent cold `pread`,
//! and `fast_frame_row_count`.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::checksum::crc32c;
use crate::mmap::pread::{pread_at_exact, resolve_backing_fd};
use crate::simd::Engine;
use crate::{block, columns, DecodeOptions, Error, Result, BLOCK_COLUMN};

use crate::frame::v5::{
    decode_header_dict, pack_block_row_meta, parse_v1_to_v5_header, unpack_block_row_meta,
    ParsedHeader,
};
use crate::frame::{read_u32, valid_kind_and_size, HEADER_SIZE, V4_SIZE_MASK};
use super::{
    is_v6_frame, parse_v6_group_dir, GROUP_CUM_FLAG, GROUP_MAGIC, GROUP_TRAILER_MAGIC,
};

#[derive(Clone, Copy)]
pub(crate) struct ParsedGroupHeader<'a> {
    pub start_block: u32,
    pub count: usize,
    pub slots: usize,
    pub is_cum: bool,
    pub use_u32: bool,
    pub block_dir_bytes: &'a [u8],
    pub lens_bytes: &'a [u8],
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
        let (is_cum, slots, use_u32) = (
            (slot_flags & GROUP_CUM_FLAG) != 0,
            (slot_flags & 0x3fff) as usize,
            (slot_flags & GROUP_CUM_FLAG) != 0 || (slot_flags & 0x8000) != 0,
        );
        if count == 0 || slots > 64 {
            return Err(Error::InvalidFrame("bad v6 group header fields"));
        }
        let bdir_len = if is_cum { count * 12 } else { 0 };
        let num_lens = (slots + 1).checked_mul(count).ok_or(Error::SizeOverflow)?;
        let lens_byte_len = num_lens * if use_u32 { 4 } else { 2 };
        let data_offset = 16usize.checked_add(bdir_len).and_then(|v| v.checked_add(lens_byte_len)).ok_or(Error::SizeOverflow)?;
        let hdr_body = group_slice.get(8..data_offset).ok_or(Error::Truncated)?;
        Ok((
            Self {
                start_block: sb,
                count,
                slots,
                is_cum,
                use_u32,
                block_dir_bytes: &group_slice[16..16 + bdir_len],
                lens_bytes: &group_slice[16 + bdir_len..data_offset],
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
            for s in 0..=gh.slots {
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
        if gh.data_offset <= 32_768 || group_slice.len() == gh.data_offset {
            if crc32c(hdr_body) != expected_crc {
                return Err(Error::InvalidFrame("v6 group header crc mismatch"));
            }
        } else if gh.is_cum && group_slice.len() >= gh.data_offset * 2 + 8 {
            let bk_off = group_slice.len() - 8 - gh.data_offset;
            if &group_slice[group_slice.len() - 4..] != GROUP_TRAILER_MAGIC
                || read_u32(group_slice, bk_off + 4)? != expected_crc
            {
                return Err(Error::InvalidFrame("v6 group header crc mismatch"));
            }
        }
        Ok(gh)
    }

    pub fn parse_resilient(group_slice: &'a [u8], expected_start: u32) -> Result<Self> {
        if let Ok(gh) = Self::parse(group_slice, expected_start) {
            if gh.data_offset.saturating_add(gh.total_streams_bytes()) <= group_slice.len() {
                return Ok(gh);
            }
        }
        if group_slice.len() >= 24
            && &group_slice[group_slice.len() - 4..] == GROUP_TRAILER_MAGIC
        {
            let gh_len = read_u32(group_slice, group_slice.len() - 8)? as usize;
            if gh_len >= 16 && gh_len + 8 <= group_slice.len() {
                let bk_start = group_slice.len() - 8 - gh_len;
                let bk_slice = &group_slice[bk_start..bk_start + gh_len];
                if let Ok(gh) = Self::parse(bk_slice, expected_start) {
                    if gh.data_offset.saturating_add(gh.total_streams_bytes()) <= bk_start {
                        return Ok(gh);
                    }
                }
            }
        }
        Err(Error::InvalidFrame("v6 group header and backup both corrupt"))
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
        if self.count == 0 {
            return 0;
        }
        self.stream_slice_range(stream_idx, self.count - 1).1
    }

    #[inline]
    pub fn total_streams_bytes(&self) -> usize {
        (0..=self.slots).map(|s| self.stream_total_len(s)).sum()
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

pub(crate) fn load_cold_v6_prefix(
    fd: i32,
    file_off: u64,
    frame_len: usize,
) -> Result<(Vec<u8>, ParsedHeader)> {
    let init_len = frame_len.min(131_072);
    if init_len < HEADER_SIZE {
        return Err(Error::Truncated);
    }
    let mut prefix = vec![0u8; init_len];
    pread_at_exact(fd, file_off, &mut prefix).map_err(|e| Error::Backend(e.to_string()))?;
    let hdr = parse_v1_to_v5_header(&prefix, usize::MAX)?;
    if hdr.payload_start > frame_len {
        return Err(Error::Truncated);
    }
    if hdr.payload_start > prefix.len() {
        let old = prefix.len();
        prefix.resize(hdr.payload_start, 0);
        pread_at_exact(fd, file_off + old as u64, &mut prefix[old..])
            .map_err(|e| Error::Backend(e.to_string()))?;
    } else {
        prefix.truncate(hdr.payload_start);
    }
    Ok((prefix, hdr))
}

fn pread_block_streams_concurrent(
    fd: i32,
    g_disk_base: u64,
    gh: &ParsedGroupHeader<'_>,
    bi: usize,
    cn: usize,
    threads: usize,
) -> Result<(Vec<u8>, Vec<(usize, usize)>)> {
    let mut blk_buf = vec![0u8; cn];
    let mut tasks: Vec<(u64, usize, usize)> = Vec::with_capacity(gh.slots + 1);
    let mut slice_ranges: Vec<(usize, usize)> = Vec::with_capacity(gh.slots + 1);
    let mut write_pos = 0usize;
    let mut stream_base = gh.data_offset as u64;
    for s in 0..=gh.slots {
        let (st, en) = gh.stream_slice_range(s, bi);
        if en < st {
            return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
        }
        let slen = en - st;
        if write_pos + slen > cn {
            return Err(Error::InvalidFrame("v6 stream overflows block cn"));
        }
        let disk_off = g_disk_base + stream_base + st as u64;
        if slen > 0 {
            tasks.push((disk_off, write_pos, slen));
        }
        slice_ranges.push((write_pos, write_pos + slen));
        write_pos += slen;
        stream_base += gh.stream_total_len(s) as u64;
    }
    if write_pos != cn {
        return Err(Error::InvalidFrame("v6 block cn mismatch"));
    }
    let mut coalesced: Vec<(u64, usize, usize)> = Vec::with_capacity(tasks.len());
    for (d_off, b_off, slen) in tasks {
        if let Some(last) = coalesced.last_mut() {
            if last.0 + (last.2 as u64) == d_off && last.1 + last.2 == b_off {
                last.2 += slen;
                continue;
            }
        }
        coalesced.push((d_off, b_off, slen));
    }
    let workers = threads.clamp(1, 16).min(coalesced.len().max(1));
    if workers <= 1 || coalesced.len() <= 1 {
        for &(d_off, b_off, slen) in &coalesced {
            pread_at_exact(fd, d_off, &mut blk_buf[b_off..b_off + slen])
                .map_err(|e| Error::Backend(e.to_string()))?;
        }
    } else {
        let next = AtomicUsize::new(0);
        let buf_ptr = blk_buf.as_mut_ptr() as usize;
        crate::pool::parallel_try_run(workers, |_| {
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= coalesced.len() {
                    return Ok(());
                }
                let (d_off, b_off, slen) = coalesced[i];
                let dst =
                    unsafe { std::slice::from_raw_parts_mut((buf_ptr + b_off) as *mut u8, slen) };
                pread_at_exact(fd, d_off, dst).map_err(|e| Error::Backend(e.to_string()))?;
            }
        })?;
    }
    Ok((blk_buf, slice_ranges))
}

pub(crate) fn try_extract_block_v6_into(
    src: &[u8],
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Option<Result<usize>> {
    let first_resident = crate::mmap::pread::is_first_page_resident(src);
    if !first_resident {
        if let Some((fd, file_off)) = resolve_backing_fd(src) {
            if let Ok((prefix, hdr)) = load_cold_v6_prefix(fd, file_off, src.len()) {
                if hdr.version != crate::VERSION_V6 {
                    return None;
                }
                if let Err(e) = options.validate() {
                    return Some(Err(e));
                }
                if let Ok(n) = extract_block_v6_cold_pread_with_prefix(
                    fd, file_off, prefix, hdr, index, out, options,
                ) {
                    return Some(Ok(n));
                }
            }
        }
    }
    if !is_v6_frame(src) {
        return None;
    }
    Some(extract_block_v6_into(src, index, out, options))
}

pub(crate) fn extract_block_v6_into(
    src: &[u8],
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    options.validate()?;
    let hdr = parse_v1_to_v5_header(src, usize::MAX)?;
    if index >= hdr.count {
        return Err(Error::InvalidOption("block index"));
    }
    let (total_v6_payload, entries) = parse_v6_group_dir(src, &hdr)?;
    let total_frame = hdr.payload_start.checked_add(total_v6_payload as usize).ok_or(Error::SizeOverflow)?;
    if src.len() < total_frame {
        return Err(Error::Truncated);
    }
    let target_rn = if hdr.has_var_raw {
        read_u32(src, hdr.var_raw_offset + index * 4)? as usize
    } else if index + 1 < hdr.count {
        hdr.block_size
    } else {
        hdr.raw_size - index * hdr.block_size
    };
    if target_rn > options.max_output {
        return Err(Error::OutputLimit { requested: target_rn as u64, limit: options.max_output });
    }
    if out.len() < target_rn {
        return Err(Error::OutputSize { expected: target_rn, actual: out.len() });
    }
    let p = HEADER_SIZE + index * 8;
    let word = read_u32(src, p)?;
    let (cn, csum, kind) = ((word & V4_SIZE_MASK) as usize, read_u32(src, p + 4)?, word >> 30);
    let gi = entries.partition_point(|e| (e.start_block as usize) + (e.block_count as usize) <= index);
    let entry = entries.get(gi).ok_or(Error::InvalidFrame("block not in v6 groups"))?;
    let g_start = hdr.payload_start + entry.payload_rel_offset as usize;
    let g_end = entries.get(gi + 1).map(|e| hdr.payload_start + e.payload_rel_offset as usize).unwrap_or(total_frame);
    let gs = src.get(g_start..g_end).ok_or(Error::Truncated)?;
    let bi = index - entry.start_block as usize;
    let gh = ParsedGroupHeader::parse_for_block(gs, entry.start_block, bi)?;

    let mut cur = gh.data_offset;
    let (s0_st, s0_en) = gh.stream_slice_range(0, bi);
    if s0_en < s0_st {
        return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
    }
    let s0_slice = gs.get(cur + s0_st..cur + s0_en).ok_or(Error::Truncated)?;
    cur += gh.stream_total_len(0);
    let dict = if kind == block::DICT_LZ { decode_header_dict(src, &hdr)? } else { Vec::new() };
    let dst = &mut out[..target_rn];
    let eng = Engine::new(options.simd)?;
    thread_local! {
        static TLS_BLK_BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    TLS_BLK_BUF.with(|cell| -> Result<()> {
        let mut blk_buf = cell.borrow_mut();
        blk_buf.clear();
        blk_buf.reserve(cn);
        blk_buf.extend_from_slice(s0_slice);
        for s in 0..gh.slots {
            let (c_st, c_en) = gh.stream_slice_range(s + 1, bi);
            if c_en < c_st {
                return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
            }
            let cs = gs.get(cur + c_st..cur + c_en).ok_or(Error::Truncated)?;
            blk_buf.extend_from_slice(cs);
            cur += gh.stream_total_len(s + 1);
        }
        if blk_buf.len() != cn {
            return Err(Error::InvalidFrame("v6 block cn mismatch"));
        }
        block::decode_with_dict(&blk_buf, kind, dst, eng, true, &dict)
    })?;
    if hdr.checksummed && options.verify_checksum && crc32c(dst) != csum {
        return Err(Error::Checksum { block: index });
    }
    Ok(target_rn)
}

fn extract_block_v6_cold_pread_with_prefix(
    fd: i32,
    file_off: u64,
    prefix: Vec<u8>,
    hdr: ParsedHeader,
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    if index >= hdr.count {
        return Err(Error::InvalidOption("block index"));
    }
    let (total_v6_payload, entries) = parse_v6_group_dir(&prefix, &hdr)?;
    let target_rn = if hdr.has_var_raw {
        read_u32(&prefix, hdr.var_raw_offset + index * 4)? as usize
    } else if index + 1 < hdr.count {
        hdr.block_size
    } else {
        hdr.raw_size - index * hdr.block_size
    };
    if out.len() < target_rn {
        return Err(Error::OutputSize { expected: target_rn, actual: out.len() });
    }
    let p = HEADER_SIZE + index * 8;
    let word = read_u32(&prefix, p)?;
    let (cn, csum, kind) = ((word & V4_SIZE_MASK) as usize, read_u32(&prefix, p + 4)?, word >> 30);
    if !valid_kind_and_size(hdr.version, kind, cn, target_rn) {
        return Err(Error::InvalidFrame("block kind/size"));
    }
    let gi = entries.partition_point(|e| (e.start_block as usize) + (e.block_count as usize) <= index);
    let entry = entries.get(gi).ok_or(Error::InvalidFrame("block not in v6 groups"))?;
    let g_rel = entry.payload_rel_offset;
    let g_end_rel = entries.get(gi + 1).map(|e| e.payload_rel_offset).unwrap_or(total_v6_payload);
    let g_len = (g_end_rel - g_rel) as usize;
    let g_disk_base = file_off + hdr.payload_start as u64 + g_rel;
    let hdr_cap = (16 + (entry.block_count as usize) * 12 + (entry.slot_count as usize + 1) * (entry.block_count as usize) * 4).min(g_len);
    let mut gh_buf = vec![0u8; hdr_cap];
    pread_at_exact(fd, g_disk_base, &mut gh_buf).map_err(|e| Error::Backend(e.to_string()))?;
    let bi = index - entry.start_block as usize;
    let gh = ParsedGroupHeader::parse_for_block(&gh_buf, entry.start_block, bi)?;
    let (blk_buf, _) =
        pread_block_streams_concurrent(fd, g_disk_base, &gh, bi, cn, options.threads.max(8))?;
    let dict = if kind == block::DICT_LZ { decode_header_dict(&prefix, &hdr)? } else { Vec::new() };
    let dst = &mut out[..target_rn];
    block::decode_with_dict(&blk_buf, kind, dst, Engine::new(options.simd)?, true, &dict)?;
    if hdr.checksummed && options.verify_checksum && crc32c(dst) != csum {
        return Err(Error::Checksum { block: index });
    }
    Ok(target_rn)
}

pub(crate) fn extract_row_range_v6_indexed(
    frame_bytes: &[u8],
    start_row: usize,
    end_row: usize,
    options: &DecodeOptions,
) -> Option<Result<Vec<u8>>> {
    let first_resident = crate::mmap::pread::is_first_page_resident(frame_bytes);
    if first_resident && !is_v6_frame(frame_bytes) {
        return None;
    }
    let cold_fd = if !first_resident { resolve_backing_fd(frame_bytes) } else { None };
    let cold_prefix = cold_fd.and_then(|(fd, f_off)| load_cold_v6_prefix(fd, f_off, frame_bytes.len()).ok().map(|(p, _)| p));
    let probe = cold_prefix.as_deref().unwrap_or(frame_bytes);
    if !is_v6_frame(probe) {
        return None;
    }
    let hdr = parse_v1_to_v5_header(probe, options.max_output).ok()?;
    if !hdr.has_row_index || probe.len() < hdr.payload_start {
        return None;
    }
    let (total_v6_payload, entries) = parse_v6_group_dir(probe, &hdr).ok()?;
    let total_frame = hdr.payload_start.checked_add(total_v6_payload as usize)?;
    if total_frame != frame_bytes.len() {
        return None;
    }

    let mut out = Vec::new();
    let (mut cur_global, mut prev_open) = (0usize, false);
    for blk_idx in 0..hdr.count {
        let ri_word = read_u32(probe, hdr.row_index_offset + blk_idx * 4).ok()?;
        let (blk_rows, ends_nl) = unpack_block_row_meta(ri_word);
        let base_row = if prev_open { cur_global - 1 } else { cur_global };
        let end_excl = base_row + blk_rows;
        if blk_rows > 0 {
            cur_global = end_excl;
            prev_open = !ends_nl;
        }
        if base_row >= end_row {
            break;
        }
        if end_excl <= start_row || blk_rows == 0 {
            continue;
        }

        let word = read_u32(probe, HEADER_SIZE + blk_idx * 8).ok()?;
        let (cn, kind) = ((word & V4_SIZE_MASK) as usize, word >> 30);
        let gi = entries.partition_point(|e| (e.start_block as usize) + (e.block_count as usize) <= blk_idx);
        let entry = entries.get(gi)?;
        let bi = blk_idx - entry.start_block as usize;

        if kind == BLOCK_COLUMN {
            let local_start = start_row.saturating_sub(base_row).min(blk_rows);
            let local_end = end_row.saturating_sub(base_row).min(blk_rows);
            let cold_buf: Vec<u8>;
            let mut col_slices: [&[u8]; 64] = [&[]; 64];
            let (s0_slice, slots): (&[u8], usize) = if let Some((fd, file_off)) = cold_fd {
                let g_rel = entry.payload_rel_offset;
                let g_end_rel = entries.get(gi + 1).map(|e| e.payload_rel_offset).unwrap_or(total_v6_payload);
                let g_disk = file_off + hdr.payload_start as u64 + g_rel;
                let hdr_cap = (16 + (entry.block_count as usize) * 12 + (entry.slot_count as usize + 1) * (entry.block_count as usize) * 4).min((g_end_rel - g_rel) as usize);
                let mut gh_buf = vec![0u8; hdr_cap];
                if let Err(e) = pread_at_exact(fd, g_disk, &mut gh_buf) {
                    return Some(Err(Error::Backend(e.to_string())));
                }
                let gh = match ParsedGroupHeader::parse_for_block(&gh_buf, entry.start_block, bi) {
                    Ok(g) => g,
                    Err(e) => return Some(Err(e)),
                };
                let ranges = match pread_block_streams_concurrent(fd, g_disk, &gh, bi, cn, options.threads.max(8)) {
                    Ok((b, r)) => { cold_buf = b; r }
                    Err(e) => return Some(Err(e)),
                };
                for s in 0..gh.slots {
                    col_slices[s] = &cold_buf[ranges[s + 1].0..ranges[s + 1].1];
                }
                (&cold_buf[ranges[0].0..ranges[0].1], gh.slots)
            } else {
                let g_start = hdr.payload_start + entry.payload_rel_offset as usize;
                let g_end = entries.get(gi + 1).map(|e| hdr.payload_start + e.payload_rel_offset as usize).unwrap_or(total_frame);
                let gs = frame_bytes.get(g_start..g_end)?;
                let gh = match ParsedGroupHeader::parse_for_block(gs, entry.start_block, bi) {
                    Ok(g) => g,
                    Err(e) => return Some(Err(e)),
                };
                let mut cur = gh.data_offset;
                let (s0_st, s0_en) = gh.stream_slice_range(0, bi);
                if s0_en < s0_st {
                    return Some(Err(Error::InvalidFrame("v6 non-monotonic cumulative offset")));
                }
                let s0 = gs.get(cur + s0_st..cur + s0_en)?;
                cur += gh.stream_total_len(0);
                for s in 0..gh.slots {
                    let (c_st, c_en) = gh.stream_slice_range(s + 1, bi);
                    if c_en < c_st {
                        return Some(Err(Error::InvalidFrame("v6 non-monotonic cumulative offset")));
                    }
                    col_slices[s] = gs.get(cur + c_st..cur + c_en)?;
                    cur += gh.stream_total_len(s + 1);
                }
                (s0, gh.slots)
            };
            if let Ok(Some((r_cnt, r_nl))) = columns::grammar_block_row_meta(s0_slice) {
                if r_cnt != blk_rows || r_nl != ends_nl {
                    return Some(Err(Error::InvalidFrame("row_index mismatch")));
                }
                if let Ok(Some(_)) = columns::with_v6_cols(&col_slices[..slots], || {
                    columns::unpack_grammar_row_range(s0_slice, local_start, local_end, &mut out)
                }) {
                    continue;
                }
            }
        }

        let rn = if hdr.has_var_raw {
            read_u32(probe, hdr.var_raw_offset + blk_idx * 4).ok()? as usize
        } else if blk_idx + 1 < hdr.count {
            hdr.block_size
        } else {
            hdr.raw_size - blk_idx * hdr.block_size
        };
        let mut raw = vec![0u8; rn];
        if let Err(e) = extract_block_v6_into(frame_bytes, blk_idx, &mut raw, options) {
            return Some(Err(e));
        }
        if pack_block_row_meta(&raw) != ri_word {
            return Some(Err(Error::InvalidFrame("row_index mismatch")));
        }
        for (r, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
            let g_row = base_row + r;
            if g_row >= start_row && g_row < end_row {
                out.extend_from_slice(line);
            }
        }
    }
    Some(Ok(out))
}
