//! `VERSION_V6` $O(1)$-offset point access (`extract_block_v6_into`,
//! `extract_row_range_v6_indexed`) and concurrent cold `pread`.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::checksum::crc32c;
use crate::mmap::pread::{pread_at_exact, resolve_backing_fd};
use crate::simd::Engine;
use crate::{block, columns, DecodeOptions, Error, Result, BLOCK_COLUMN};

use crate::frame::v5::{
    block_raw_size, decode_header_dict, pack_block_row_meta, parse_v1_to_v5_header,
    unpack_block_row_meta, ParsedHeader,
};
use crate::frame::{
    entry_size, read_u32, valid_kind_and_size, HEADER_SIZE, V4_SIZE_MASK, VERSION_V4, VERSION_V5,
};
use super::{
    is_v6_frame, parse_v6_group_dir, GroupDirEntry, ParsedGroupHeader, GROUP_V2_FLAG, VERSION_V6,
};

pub(crate) fn load_cold_v6_prefix(
    fd: i32,
    file_off: u64,
    frame_len: usize,
) -> Result<(Vec<u8>, ParsedHeader)> {
    let init_len = frame_len.min(98_304);
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
    gh_buf: &[u8],
    gh: &ParsedGroupHeader<'_>,
    bi: usize,
    threads: usize,
    blk_buf: &mut Vec<u8>,
) -> Result<Vec<(usize, usize)>> {
    let mut tasks: Vec<(u64, usize, usize)> = Vec::with_capacity(gh.num_streams);
    let mut cached_copies: Vec<(usize, usize, usize)> = Vec::new();
    let mut slice_ranges: Vec<(usize, usize)> = Vec::with_capacity(gh.num_streams);
    let (mut write_pos, mut stream_base) = (0usize, gh.data_offset as u64);
    for s in 0..gh.num_streams {
        let (st, en) = gh.stream_slice_range(s, bi);
        if en < st {
            return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
        }
        let slen = en - st;
        if slen > 0 {
            let rel_st = (stream_base as usize).saturating_add(st);
            let rel_en = rel_st.saturating_add(slen);
            if rel_en <= gh_buf.len() {
                cached_copies.push((rel_st, write_pos, slen));
            } else {
                tasks.push((g_disk_base + stream_base + st as u64, write_pos, slen));
            }
        }
        slice_ranges.push((write_pos, write_pos + slen));
        write_pos += slen;
        stream_base += gh.stream_total_len(s) as u64;
    }
    blk_buf.clear();
    blk_buf.resize(write_pos, 0);
    for &(rel_st, b_off, slen) in &cached_copies {
        blk_buf[b_off..b_off + slen].copy_from_slice(&gh_buf[rel_st..rel_st + slen]);
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
    let workers = threads.clamp(1, 24).min(coalesced.len().max(1));
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
    Ok(slice_ranges)
}

fn read_cold_group_header(
    fd: i32,
    g_disk_base: u64,
    g_len: usize,
    entry: &GroupDirEntry,
    gh_buf: &mut Vec<u8>,
) -> Result<()> {
    let hdr_cap = (18
        + (entry.block_count as usize) * 12
        + (entry.slot_count as usize + 2) * (entry.block_count as usize) * 4
        + 4096)
        .min(g_len);
    gh_buf.clear();
    gh_buf.resize(hdr_cap, 0);
    pread_at_exact(fd, g_disk_base, gh_buf).map_err(|e| Error::Backend(e.to_string()))?;
    if gh_buf.len() >= 18 && (u16::from_le_bytes([gh_buf[14], gh_buf[15]]) & GROUP_V2_FLAG) != 0 {
        let canon_len = u16::from_le_bytes([gh_buf[16], gh_buf[17]]) as usize;
        let slots = (u16::from_le_bytes([gh_buf[14], gh_buf[15]]) & 0x1fff) as usize;
        let count = u16::from_le_bytes([gh_buf[12], gh_buf[13]]) as usize;
        let num_streams = if slots > 0 { slots + 2 } else { 1 };
        let needed = (18 + count * 12 + num_streams * count * 4 + canon_len).min(g_len);
        if needed > gh_buf.len() {
            let old = gh_buf.len();
            gh_buf.resize(needed, 0);
            pread_at_exact(fd, g_disk_base + old as u64, &mut gh_buf[old..])
                .map_err(|e| Error::Backend(e.to_string()))?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_v6_block_slices<'a>(
    frame_bytes: &'a [u8],
    hdr: &ParsedHeader,
    entries: &[GroupDirEntry],
    total_v6_payload: u64,
    cold_fd: Option<(i32, u64)>,
    blk_idx: usize,
    rn: usize,
    cn: usize,
    threads: usize,
    gh_buf: &'a mut Vec<u8>,
    cold_buf: &'a mut Vec<u8>,
    stitch_buf: &'a mut Vec<u8>,
) -> Result<(usize, &'a [u8], [&'a [u8]; 64])> {
    let gi = entries.partition_point(|e| (e.start_block as usize) + (e.block_count as usize) <= blk_idx);
    let entry = entries.get(gi).ok_or(Error::InvalidFrame("block not in v6 groups"))?;
    let bi = blk_idx - entry.start_block as usize;
    let mut col_slices: [&'a [u8]; 64] = [&[]; 64];

    let (slots, is_v2, canon_s0, s0_raw, rr_raw, gh_bi_ok) = if let Some((fd, file_off)) = cold_fd {
        let g_rel = entry.payload_rel_offset;
        let g_end_rel = entries.get(gi + 1).map(|e| e.payload_rel_offset).unwrap_or(total_v6_payload);
        let g_disk = file_off + hdr.payload_start as u64 + g_rel;
        read_cold_group_header(fd, g_disk, (g_end_rel - g_rel) as usize, entry, gh_buf)?;
        let gh = ParsedGroupHeader::parse_for_block(gh_buf, entry.start_block, bi)?;
        let ranges = pread_block_streams_concurrent(fd, g_disk, gh_buf, &gh, bi, threads.max(24), cold_buf)?;
        let cb: &'a [u8] = cold_buf.as_slice();
        for s in 0..gh.slots {
            col_slices[s] = &cb[ranges[s + 1].0..ranges[s + 1].1];
        }
        let s0_r = &cb[ranges[0].0..ranges[0].1];
        let rr_r = if gh.is_v2 && gh.slots > 0 {
            &cb[ranges[gh.slots + 1].0..ranges[gh.slots + 1].1]
        } else {
            &[]
        };
        (gh.slots, gh.is_v2, gh.canon_s0, s0_r, rr_r, gh)
    } else {
        let total_frame = hdr.payload_start + total_v6_payload as usize;
        let g_start = hdr.payload_start + entry.payload_rel_offset as usize;
        let g_end = entries
            .get(gi + 1)
            .map(|e| hdr.payload_start + e.payload_rel_offset as usize)
            .unwrap_or(total_frame);
        let gs = frame_bytes.get(g_start..g_end).ok_or(Error::Truncated)?;
        let gh = ParsedGroupHeader::parse_for_block(gs, entry.start_block, bi)?;
        let mut cur = gh.data_offset;
        let (s0_st, s0_en) = gh.stream_slice_range(0, bi);
        if s0_en < s0_st {
            return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
        }
        let s0_r = gs.get(cur + s0_st..cur + s0_en).ok_or(Error::Truncated)?;
        cur += gh.stream_total_len(0);
        for s in 0..gh.slots {
            let (c_st, c_en) = gh.stream_slice_range(s + 1, bi);
            if c_en < c_st {
                return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
            }
            col_slices[s] = gs.get(cur + c_st..cur + c_en).ok_or(Error::Truncated)?;
            cur += gh.stream_total_len(s + 1);
        }
        let rr_r = if gh.is_v2 && gh.slots > 0 {
            let (rr_st, rr_en) = gh.stream_slice_range(gh.slots + 1, bi);
            if rr_en < rr_st {
                return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
            }
            gs.get(cur + rr_st..cur + rr_en).ok_or(Error::Truncated)?
        } else {
            &[]
        };
        (gh.slots, gh.is_v2, gh.canon_s0, s0_r, rr_r, gh)
    };

    stitch_buf.clear();
    let mut s0_in_stitch = false;
    let s0_hdr: &[u8] = if s0_raw.is_empty() && is_v2 && !canon_s0.is_empty() {
        canon_s0
    } else {
        let body = ParsedGroupHeader::unwrap_verified_slice(s0_raw, is_v2, blk_idx)?;
        if is_v2 && body.len() == 2 {
            let rc = u16::from_le_bytes([body[0], body[1]]) as usize;
            let _ = columns::expand_v6_compact_s0(canon_s0, rc, stitch_buf)
                .ok_or(Error::InvalidFrame("bad v6 compact s0"))?;
            s0_in_stitch = true;
            &[]
        } else if is_v2 && body.first() == Some(&0) {
            let _ = columns::expand_v6_tpl_s0(canon_s0, &body[1..], stitch_buf)
                .ok_or(Error::InvalidFrame("bad v6 tpl s0"))?;
            s0_in_stitch = true;
            &[]
        } else {
            body
        }
    };
    let exp_s0_len = if s0_in_stitch { stitch_buf.len() } else { s0_hdr.len() };
    if !gh_bi_ok.verify_block_cn(bi, rn, exp_s0_len, cn) {
        return Err(Error::InvalidFrame("v6 block cn mismatch"));
    }
    let rr_body = ParsedGroupHeader::unwrap_verified_slice(rr_raw, is_v2, blk_idx)?;
    let s0_payload = if rr_body.is_empty() {
        if s0_in_stitch { stitch_buf.as_slice() } else { s0_hdr }
    } else {
        if !s0_in_stitch {
            stitch_buf.reserve(s0_hdr.len() + rr_body.len());
            stitch_buf.extend_from_slice(s0_hdr);
        }
        stitch_buf.extend_from_slice(rr_body);
        stitch_buf.as_slice()
    };
    Ok((slots, s0_payload, col_slices))
}

fn extract_block_v5_cold_pread_with_prefix(
    fd: i32,
    file_off: u64,
    prefix: &[u8],
    hdr: &ParsedHeader,
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    if index >= hdr.count {
        return Err(Error::InvalidOption("block index"));
    }
    let esize = entry_size(hdr.version);
    let (mut next_payload, mut next_raw) = (hdr.payload_start, 0usize);
    for i in 0..index {
        let rn = block_raw_size(prefix, hdr, i, next_raw)?;
        let cn = (read_u32(prefix, HEADER_SIZE + i * esize)? & V4_SIZE_MASK) as usize;
        next_raw += rn;
        next_payload = next_payload.checked_add(cn).ok_or(Error::SizeOverflow)?;
    }
    let target_rn = block_raw_size(prefix, hdr, index, next_raw)?;
    if target_rn > options.max_output {
        return Err(Error::OutputLimit { requested: target_rn as u64, limit: options.max_output });
    }
    if out.len() < target_rn {
        return Err(Error::OutputSize { expected: target_rn, actual: out.len() });
    }
    let p = HEADER_SIZE + index * esize;
    let word = read_u32(prefix, p)?;
    let (cn, csum, kind) = ((word & V4_SIZE_MASK) as usize, read_u32(prefix, p + 4)?, word >> 30);
    if !valid_kind_and_size(hdr.version, kind, cn, target_rn) {
        return Err(Error::InvalidFrame("block kind/size"));
    }
    let mut payload = vec![0u8; cn];
    pread_at_exact(fd, file_off + next_payload as u64, &mut payload)
        .map_err(|e| Error::Backend(e.to_string()))?;
    let dict = if kind == block::DICT_LZ { decode_header_dict(prefix, hdr)? } else { Vec::new() };
    let dst = &mut out[..target_rn];
    block::decode_with_dict(&payload, kind, dst, Engine::new(options.simd)?, true, &dict)?;
    if hdr.checksummed && options.verify_checksum && crc32c(dst) != csum {
        return Err(Error::Checksum { block: index });
    }
    Ok(target_rn)
}

pub(crate) fn try_extract_block_v6_into(
    src: &[u8],
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Option<Result<usize>> {
    if !crate::mmap::pread::is_first_page_resident(src) {
        if let Some((fd, file_off)) = resolve_backing_fd(src) {
            if let Ok((prefix, hdr)) = load_cold_v6_prefix(fd, file_off, src.len()) {
                if let Err(e) = options.validate() {
                    return Some(Err(e));
                }
                if hdr.version == VERSION_V6 {
                    if let Ok(n) = extract_block_v6_impl(
                        src, &prefix, &hdr, Some((fd, file_off)), index, out, options,
                    ) {
                        return Some(Ok(n));
                    }
                } else if (VERSION_V4..=VERSION_V5).contains(&hdr.version) {
                    if let Ok(n) = extract_block_v5_cold_pread_with_prefix(
                        fd, file_off, &prefix, &hdr, index, out, options,
                    ) {
                        return Some(Ok(n));
                    }
                }
                return None;
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
    extract_block_v6_impl(src, src, &hdr, None, index, out, options)
}

fn extract_block_v6_impl(
    src: &[u8],
    probe: &[u8],
    hdr: &ParsedHeader,
    cold_fd: Option<(i32, u64)>,
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    if index >= hdr.count {
        return Err(Error::InvalidOption("block index"));
    }
    let (total_v6_payload, entries) = parse_v6_group_dir(probe, hdr)?;
    let total_frame = hdr.payload_start.checked_add(total_v6_payload as usize).ok_or(Error::SizeOverflow)?;
    if src.len() < total_frame {
        return Err(Error::Truncated);
    }
    let target_rn = if hdr.has_var_raw {
        read_u32(probe, hdr.var_raw_offset + index * 4)? as usize
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
    let word = read_u32(probe, p)?;
    let (cn, csum, kind) = ((word & V4_SIZE_MASK) as usize, read_u32(probe, p + 4)?, word >> 30);
    if !valid_kind_and_size(hdr.version, kind, cn, target_rn) {
        return Err(Error::InvalidFrame("block kind/size"));
    }
    let (mut gh_buf, mut cold_buf, mut stitch_buf) = (Vec::new(), Vec::new(), Vec::new());
    let (slots, s0_payload, col_slices) = load_v6_block_slices(
        src, hdr, &entries, total_v6_payload, cold_fd, index, target_rn, cn, options.threads,
        &mut gh_buf, &mut cold_buf, &mut stitch_buf,
    )?;
    let dict = if kind == block::DICT_LZ { decode_header_dict(probe, hdr)? } else { Vec::new() };
    let dst = &mut out[..target_rn];
    let eng = Engine::new(options.simd)?;
    if slots == 0 {
        block::decode_with_dict(s0_payload, kind, dst, eng, true, &dict)?;
    } else {
        columns::with_v6_cols(&col_slices[..slots], || {
            block::decode_with_dict(s0_payload, kind, dst, eng, true, &dict)
        })
        .map_err(|e| match e {
            Error::Checksum { .. } => Error::Checksum { block: index },
            other => other,
        })?;
    }
    if hdr.checksummed && options.verify_checksum && crc32c(dst) != csum {
        return Err(Error::Checksum { block: index });
    }
    Ok(target_rn)
}

fn extract_row_range_v5_cold_pread(
    fd: i32,
    file_off: u64,
    prefix: &[u8],
    hdr: &ParsedHeader,
    start_row: usize,
    end_row: usize,
) -> Option<Result<Vec<u8>>> {
    let esize = entry_size(hdr.version);
    let mut out = Vec::new();
    let (mut cur_global, mut prev_open) = (0usize, false);
    let (mut next_payload, mut next_raw) = (hdr.payload_start, 0usize);
    for blk_idx in 0..hdr.count {
        let rn = block_raw_size(prefix, hdr, blk_idx, next_raw).ok()?;
        let word = read_u32(prefix, HEADER_SIZE + blk_idx * esize).ok()?;
        let (cn, kind) = ((word & V4_SIZE_MASK) as usize, word >> 30);
        let pay_off = next_payload;
        next_raw += rn;
        next_payload = next_payload.checked_add(cn)?;
        let ri_word = read_u32(prefix, hdr.row_index_offset + blk_idx * 4).ok()?;
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
        if end_excl <= start_row || blk_rows == 0 || kind != BLOCK_COLUMN {
            if end_excl > start_row && blk_rows > 0 {
                return None;
            }
            continue;
        }
        let mut payload = vec![0u8; cn];
        if pread_at_exact(fd, file_off + pay_off as u64, &mut payload).is_err() {
            return None;
        }
        let local_start = start_row.saturating_sub(base_row).min(blk_rows);
        let local_end = end_row.saturating_sub(base_row).min(blk_rows);
        let (r_cnt, r_nl) = columns::grammar_block_row_meta(&payload).ok()??;
        if r_cnt != blk_rows || r_nl != ends_nl {
            return Some(Err(Error::InvalidFrame("row_index mismatch")));
        }
        columns::unpack_grammar_row_range(&payload, local_start, local_end, &mut out).ok()??;
    }
    Some(Ok(out))
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
    let cold_prefix = cold_fd.and_then(|(fd, f_off)| {
        load_cold_v6_prefix(fd, f_off, frame_bytes.len()).ok().map(|(p, _)| p)
    });
    let probe = cold_prefix.as_deref().unwrap_or(frame_bytes);
    let hdr = parse_v1_to_v5_header(probe, options.max_output).ok()?;
    if !hdr.has_row_index || probe.len() < hdr.payload_start {
        return None;
    }
    if !is_v6_frame(probe) {
        if let Some((fd, file_off)) = cold_fd {
            if (VERSION_V4..=VERSION_V5).contains(&hdr.version) {
                return extract_row_range_v5_cold_pread(fd, file_off, probe, &hdr, start_row, end_row);
            }
        }
        return None;
    }
    let (total_v6_payload, entries) = parse_v6_group_dir(probe, &hdr).ok()?;
    let total_frame = hdr.payload_start.checked_add(total_v6_payload as usize)?;
    if total_frame != frame_bytes.len() {
        return None;
    }

    let mut out = Vec::new();
    let (mut cur_global, mut prev_open) = (0usize, false);
    let (mut gh_buf, mut cold_buf, mut stitch_buf) = (Vec::new(), Vec::new(), Vec::new());
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
        let rn = if hdr.has_var_raw {
            read_u32(probe, hdr.var_raw_offset + blk_idx * 4).ok()? as usize
        } else if blk_idx + 1 < hdr.count {
            hdr.block_size
        } else {
            hdr.raw_size - blk_idx * hdr.block_size
        };

        if kind == BLOCK_COLUMN {
            let local_start = start_row.saturating_sub(base_row).min(blk_rows);
            let local_end = end_row.saturating_sub(base_row).min(blk_rows);
            let (slots, s0_slice, col_slices) = match load_v6_block_slices(
                frame_bytes, &hdr, &entries, total_v6_payload, cold_fd, blk_idx, rn, cn,
                options.threads, &mut gh_buf, &mut cold_buf, &mut stitch_buf,
            ) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            match columns::grammar_block_row_meta(s0_slice) {
                Ok(Some((r_cnt, r_nl))) => {
                    if r_cnt != blk_rows || r_nl != ends_nl {
                        return Some(Err(Error::InvalidFrame("row_index mismatch")));
                    }
                    match columns::with_v6_cols(&col_slices[..slots], || {
                        columns::unpack_grammar_row_range(s0_slice, local_start, local_end, &mut out)
                    }) {
                        Ok(Some(_)) => continue,
                        Ok(None) => {}
                        Err(e) => return Some(Err(e)),
                    }
                }
                Ok(None) => {}
                Err(e) => return Some(Err(e)),
            }
        }

        let mut raw = vec![0u8; rn];
        if let Err(e) = extract_block_v6_impl(frame_bytes, probe, &hdr, cold_fd, blk_idx, &mut raw, options) {
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
