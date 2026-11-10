//! Cold selective `pread` and warm selective reassembly for `VERSION_V6` column-grouped frames,
//! plus single-block random-access extraction (`extract_block_v6_into`).

use crate::checksum::crc32c;
use crate::mmap::pread::{pread_at_exact, resolve_backing_fd};
use crate::mmap::sys::is_slice_resident;
use crate::ops::{ColTarget, PredExpr};
use crate::simd::Engine;
use crate::{block, DecodeOptions, Error, Result};

use crate::frame::v5::{
    block_raw_size, block_row_meta_raw, decode_header_dict, pack_block_row_meta,
    parse_v1_to_v5_header, unpack_block_row_meta, ParsedHeader, FLAG_COL_GROUP,
};
use crate::frame::{
    entry_size, read_u32, valid_kind_and_size, HEADER_SIZE, V4_SIZE_MASK, VERSION_V5,
};
use super::{
    is_v6_frame, parse_v6_group_dir, validate_v6_header_and_dir, ParsedGroupHeader, VERSION_V6,
};

fn collect_pred_cols<'a>(expr: &PredExpr<'a>, out: &mut Vec<ColTarget<'a>>) -> bool {
    match expr {
        PredExpr::True => true,
        PredExpr::Col(p) => {
            if !out.contains(&p.col) {
                out.push(p.col);
            }
            true
        }
        PredExpr::All(children) | PredExpr::Any(children) => {
            children.iter().all(|c| collect_pred_cols(c, out))
        }
        PredExpr::Not(a) => collect_pred_cols(a, out),
        PredExpr::RowContains(_) => false,
    }
}

fn collect_expr_targets<'a>(
    where_expr: Option<&PredExpr<'a>>,
    extra_cols: &[ColTarget<'a>],
    allow_partial_cols: bool,
) -> Option<Vec<ColTarget<'a>>> {
    if !allow_partial_cols {
        return None;
    }
    let mut cols = Vec::with_capacity(extra_cols.len() + 4);
    if let Some(expr) = where_expr {
        if !collect_pred_cols(expr, &mut cols) {
            return None;
        }
    }
    for &c in extra_cols {
        if !cols.contains(&c) {
            cols.push(c);
        }
    }
    Some(cols)
}

fn load_cold_v6_prefix(fd: i32, file_off: u64, frame_len: usize) -> Result<(Vec<u8>, ParsedHeader)> {
    let init_len = frame_len.min(65_536);
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

pub(crate) fn extract_block_v6_into(
    src: &[u8],
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    options.validate()?;
    if !is_slice_resident(src) {
        if let Some((fd, file_off)) = resolve_backing_fd(src) {
            if let Ok(n) = extract_block_v6_cold_pread(fd, file_off, src.len(), index, out, options) {
                return Ok(n);
            }
        }
    }
    let hdr = parse_v1_to_v5_header(src, usize::MAX)?;
    let (total_frame, _) = validate_v6_header_and_dir(src, &hdr)?;
    if src.len() < total_frame {
        return Err(Error::Truncated);
    }
    if index >= hdr.count {
        return Err(Error::InvalidOption("block index"));
    }
    let (_, entries) = parse_v6_group_dir(src, &hdr)?;
    let dict = decode_header_dict(src, &hdr)?;
    let (mut next_raw, mut target_rn) = (0usize, 0usize);
    for i in 0..=index {
        target_rn = block_raw_size(src, &hdr, i, next_raw)?;
        next_raw += target_rn;
    }
    if target_rn > options.max_output {
        return Err(Error::OutputLimit { requested: target_rn as u64, limit: options.max_output });
    }
    if out.len() < target_rn {
        return Err(Error::OutputSize { expected: target_rn, actual: out.len() });
    }
    let p = HEADER_SIZE + index * 8;
    let word = read_u32(src, p)?;
    let (cn, csum, kind) = ((word & V4_SIZE_MASK) as usize, read_u32(src, p + 4)?, word >> 30);
    let gi = entries
        .iter()
        .position(|e| index >= e.start_block as usize && index < e.start_block as usize + e.block_count as usize)
        .ok_or(Error::InvalidFrame("block not in v6 groups"))?;
    let entry = &entries[gi];
    let g_start = hdr.payload_start + entry.payload_rel_offset as usize;
    let g_end = entries.get(gi + 1).map(|e| hdr.payload_start + e.payload_rel_offset as usize).unwrap_or(total_frame);
    let gs = src.get(g_start..g_end).ok_or(Error::Truncated)?;
    let gh = ParsedGroupHeader::parse(gs, entry.start_block)?;
    let bi = index - entry.start_block as usize;
    let mut blk_buf = Vec::with_capacity(cn);
    let mut cur = gh.data_offset;
    for s in 0..=gh.slots {
        let mut s_off = cur;
        for prev in 0..gh.count {
            let l = gh.stream_len(s, prev);
            if prev < bi {
                s_off += l;
            }
            cur += l;
        }
        let slen = gh.stream_len(s, bi);
        let chunk = gs.get(s_off..s_off.checked_add(slen).ok_or(Error::SizeOverflow)?).ok_or(Error::Truncated)?;
        blk_buf.extend_from_slice(chunk);
    }
    if blk_buf.len() != cn {
        return Err(Error::InvalidFrame("v6 block cn mismatch"));
    }
    let dst = &mut out[..target_rn];
    block::decode_with_dict(&blk_buf, kind, dst, Engine::new(options.simd)?, true, &dict)?;
    if hdr.checksummed && options.verify_checksum && crc32c(dst) != csum {
        return Err(Error::Checksum { block: index });
    }
    Ok(target_rn)
}

fn extract_block_v6_cold_pread(
    fd: i32,
    file_off: u64,
    frame_len: usize,
    index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    let (prefix, hdr) = load_cold_v6_prefix(fd, file_off, frame_len)?;
    if index >= hdr.count {
        return Err(Error::InvalidOption("block index"));
    }
    let (total_v6_payload, entries) = parse_v6_group_dir(&prefix, &hdr)?;
    let dict = decode_header_dict(&prefix, &hdr)?;
    let (mut next_raw, mut target_rn) = (0usize, 0usize);
    for i in 0..=index {
        target_rn = block_raw_size(&prefix, &hdr, i, next_raw)?;
        next_raw += target_rn;
    }
    if out.len() < target_rn {
        return Err(Error::OutputSize { expected: target_rn, actual: out.len() });
    }
    let p = HEADER_SIZE + index * 8;
    let word = read_u32(&prefix, p)?;
    let (cn, csum, kind) = ((word & V4_SIZE_MASK) as usize, read_u32(&prefix, p + 4)?, word >> 30);
    if !valid_kind_and_size(hdr.version, kind, cn, target_rn) {
        return Err(Error::InvalidFrame("block kind/size"));
    }
    let gi = entries
        .iter()
        .position(|e| index >= e.start_block as usize && index < e.start_block as usize + e.block_count as usize)
        .ok_or(Error::InvalidFrame("block not in v6 groups"))?;
    let entry = &entries[gi];
    let g_rel = entry.payload_rel_offset;
    let g_end_rel = entries.get(gi + 1).map(|e| e.payload_rel_offset).unwrap_or(total_v6_payload);
    let g_len = (g_end_rel - g_rel) as usize;
    let g_disk_base = file_off + hdr.payload_start as u64 + g_rel;
    let hdr_cap = (16 + (entry.slot_count as usize + 1) * (entry.block_count as usize) * 4).min(g_len);
    let mut gh_buf = vec![0u8; hdr_cap];
    pread_at_exact(fd, g_disk_base, &mut gh_buf).map_err(|e| Error::Backend(e.to_string()))?;
    let gh = ParsedGroupHeader::parse(&gh_buf, entry.start_block)?;
    let bi = index - entry.start_block as usize;
    let mut blk_buf = vec![0u8; cn];
    let (mut write_pos, mut cur) = (0usize, gh.data_offset as u64);
    for s in 0..=gh.slots {
        let mut s_off = cur;
        for prev in 0..gh.count {
            let l = gh.stream_len(s, prev) as u64;
            if prev < bi {
                s_off += l;
            }
            cur += l;
        }
        let slen = gh.stream_len(s, bi);
        if write_pos + slen > cn {
            return Err(Error::InvalidFrame("v6 stream overflows block cn"));
        }
        pread_at_exact(fd, g_disk_base + s_off, &mut blk_buf[write_pos..write_pos + slen])
            .map_err(|e| Error::Backend(e.to_string()))?;
        write_pos += slen;
    }
    if write_pos != cn {
        return Err(Error::InvalidFrame("v6 block cn mismatch"));
    }
    let dst = &mut out[..target_rn];
    block::decode_with_dict(&blk_buf, kind, dst, Engine::new(options.simd)?, true, &dict)?;
    if hdr.checksummed && options.verify_checksum && crc32c(dst) != csum {
        return Err(Error::Checksum { block: index });
    }
    Ok(target_rn)
}

fn compute_group_req_mask(
    stream0_buf: &[u8],
    gh: &ParsedGroupHeader<'_>,
    targets: Option<&[ColTarget<'_>]>,
) -> u64 {
    let Some(t_list) = targets else {
        return u64::MAX;
    };
    if gh.slots == 0 {
        return u64::MAX;
    }
    if t_list.is_empty() {
        return 0;
    }
    let mut mask = 0u64;
    let mut pos = 0usize;
    for bi in 0..gh.count {
        let slen = gh.stream_len(0, bi);
        let s0 = &stream0_buf[pos..pos + slen];
        pos += slen;
        if bi == 0 || bi + 1 == gh.count {
            match crate::columns::resolve_stream0_required_slots(s0, t_list) {
                Some(m) => mask |= m,
                None => return u64::MAX,
            }
        }
    }
    mask
}

fn reconstruct_group_blob<'a>(
    gh: &ParsedGroupHeader<'_>,
    s0_slice: &[u8],
    req_mask: u64,
    mut col_slice_fn: impl FnMut(usize, usize, usize) -> &'a [u8],
) -> (Vec<usize>, Vec<u8>) {
    let mut s0_cur = 0usize;
    let mut col_curs = vec![0usize; gh.slots];
    let mut block_cns = Vec::with_capacity(gh.count);
    let mut blob = Vec::new();
    for bi in 0..gh.count {
        let b_start = blob.len();
        let l0 = gh.stream_len(0, bi);
        blob.extend_from_slice(&s0_slice[s0_cur..s0_cur + l0]);
        s0_cur += l0;
        for slot in 0..gh.slots {
            let lc = gh.stream_len(slot + 1, bi);
            if ((req_mask >> slot) & 1) != 0 {
                let cur = col_curs[slot];
                blob.extend_from_slice(col_slice_fn(slot, cur, lc));
                col_curs[slot] = cur + lc;
            } else {
                blob.push(0u8);
            }
        }
        block_cns.push(blob.len() - b_start);
    }
    (block_cns, blob)
}

fn assemble_synthetic_v5(
    prefix: &[u8],
    hdr: &ParsedHeader,
    group_blobs: Vec<(usize, Vec<usize>, Vec<u8>)>,
) -> Vec<u8> {
    let total_payload: usize = group_blobs.iter().map(|(_, _, b)| b.len()).sum();
    let mut out = Vec::with_capacity(hdr.group_dir_offset + total_payload);
    out.extend_from_slice(&prefix[..hdr.group_dir_offset]);
    out[4..6].copy_from_slice(&VERSION_V5.to_le_bytes());
    let flags = u16::from_le_bytes([out[6], out[7]]) & !FLAG_COL_GROUP;
    out[6..8].copy_from_slice(&flags.to_le_bytes());
    out[28..32].copy_from_slice(&(hdr.dict_raw_len as u32).to_le_bytes());
    for (sb, block_cns, blob) in group_blobs {
        for (bi, &new_cn) in block_cns.iter().enumerate() {
            let p = HEADER_SIZE + (sb + bi) * 8;
            let old_word = u32::from_le_bytes(out[p..p + 4].try_into().unwrap());
            let kind = old_word >> 30;
            let new_word = ((new_cn as u32) & V4_SIZE_MASK) | (kind << 30);
            out[p..p + 4].copy_from_slice(&new_word.to_le_bytes());
        }
        out.extend_from_slice(&blob);
    }
    out
}

pub(crate) fn pread_v6_for_query(
    fd: i32,
    file_off: u64,
    frame_len: usize,
    where_expr: Option<&PredExpr<'_>>,
    extra_cols: &[ColTarget<'_>],
    allow_partial_cols: bool,
    threads: usize,
) -> Result<Vec<u8>> {
    let (prefix, hdr) = load_cold_v6_prefix(fd, file_off, frame_len)?;
    let (total_v6_payload, entries) = parse_v6_group_dir(&prefix, &hdr)?;
    let opt_targets = collect_expr_targets(where_expr, extra_cols, allow_partial_cols);
    let num_groups = entries.len();
    let mut results: Vec<Option<(usize, Vec<usize>, Vec<u8>)>> =
        (0..num_groups).map(|_| None).collect();
    let results_ptr = results.as_mut_ptr() as usize;
    let next_g = std::sync::atomic::AtomicUsize::new(0);
    let workers = threads.clamp(1, 16).min(num_groups.max(1));

    crate::pool::parallel_try_run(workers, |_| {
        loop {
            let gi = next_g.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if gi >= num_groups {
                return Ok(());
            }
            let entry = &entries[gi];
            let g_rel = entry.payload_rel_offset;
            let g_end_rel = entries
                .get(gi + 1)
                .map(|e| e.payload_rel_offset)
                .unwrap_or(total_v6_payload);
            let g_len = (g_end_rel - g_rel) as usize;
            let g_disk = file_off + hdr.payload_start as u64 + g_rel;

            let init_read = g_len.min(65_536);
            let mut init_buf = vec![0u8; init_read];
            pread_at_exact(fd, g_disk, &mut init_buf)
                .map_err(|e| Error::Backend(e.to_string()))?;
            let (s0_total, data_off) = {
                let gh0 = ParsedGroupHeader::parse(&init_buf, entry.start_block)?;
                let st: usize = (0..gh0.count).map(|bi| gh0.stream_len(0, bi)).sum();
                (st, gh0.data_offset)
            };
            let s0_end = data_off + s0_total;
            if s0_end > init_buf.len() {
                let old = init_buf.len();
                init_buf.resize(s0_end, 0);
                pread_at_exact(fd, g_disk + old as u64, &mut init_buf[old..])
                    .map_err(|e| Error::Backend(e.to_string()))?;
            }
            let gh = ParsedGroupHeader::parse(&init_buf, entry.start_block)?;
            let s0_slice = &init_buf[gh.data_offset..s0_end];
            let req_mask = compute_group_req_mask(s0_slice, &gh, opt_targets.as_deref());

            let mut col_totals = vec![0usize; gh.slots];
            let mut col_rel_offs = vec![0usize; gh.slots];
            let mut cur_off = s0_end;
            for c in 0..gh.slots {
                col_rel_offs[c] = cur_off;
                let ct: usize = (0..gh.count).map(|bi| gh.stream_len(c + 1, bi)).sum();
                col_totals[c] = ct;
                cur_off += ct;
            }
            let mut col_buffers: Vec<Vec<u8>> = (0..gh.slots).map(|_| Vec::new()).collect();
            let mut c = 0usize;
            while c < gh.slots {
                if ((req_mask >> c) & 1) == 0 {
                    c += 1;
                    continue;
                }
                let run_start = c;
                let mut run_bytes = col_totals[c];
                c += 1;
                while c < gh.slots && ((req_mask >> c) & 1) != 0 {
                    run_bytes += col_totals[c];
                    c += 1;
                }
                let mut coalesced = vec![0u8; run_bytes];
                let disk_start = g_disk + col_rel_offs[run_start] as u64;
                const CHUNK: usize = 1 << 20;
                let mut off = 0usize;
                while off < run_bytes {
                    let take = (run_bytes - off).min(CHUNK);
                    pread_at_exact(fd, disk_start + off as u64, &mut coalesced[off..off + take])
                        .map_err(|e| Error::Backend(e.to_string()))?;
                    off += take;
                }
                let mut rel = 0usize;
                for slot in run_start..c {
                    let len = col_totals[slot];
                    col_buffers[slot] = coalesced[rel..rel + len].to_vec();
                    rel += len;
                }
            }
            let (block_cns, blob) = reconstruct_group_blob(&gh, s0_slice, req_mask, |slot, cur, lc| {
                &col_buffers[slot][cur..cur + lc]
            });
            let slot_ptr = unsafe { &mut *(results_ptr as *mut Option<(usize, Vec<usize>, Vec<u8>)>).add(gi) };
            *slot_ptr = Some((entry.start_block as usize, block_cns, blob));
        }
    })?;

    let group_blobs: Vec<(usize, Vec<usize>, Vec<u8>)> =
        results.into_iter().map(|r| r.unwrap()).collect();
    Ok(assemble_synthetic_v5(&prefix, &hdr, group_blobs))
}

pub(crate) fn reassemble_v6_for_query_warm(
    src: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    extra_cols: &[ColTarget<'_>],
    allow_partial_cols: bool,
    threads: usize,
) -> Result<Vec<u8>> {
    let hdr = parse_v1_to_v5_header(src, usize::MAX)?;
    let (total_v6_payload, entries) = parse_v6_group_dir(src, &hdr)?;
    let v6_end = hdr
        .payload_start
        .checked_add(total_v6_payload as usize)
        .ok_or(Error::SizeOverflow)?;
    let v6_payload = src.get(hdr.payload_start..v6_end).ok_or(Error::Truncated)?;
    let opt_targets = collect_expr_targets(where_expr, extra_cols, allow_partial_cols);
    let num_groups = entries.len();
    let mut results: Vec<Option<(usize, Vec<usize>, Vec<u8>)>> =
        (0..num_groups).map(|_| None).collect();
    let results_ptr = results.as_mut_ptr() as usize;
    let next_g = std::sync::atomic::AtomicUsize::new(0);
    let workers = threads.clamp(1, 16).min(num_groups.max(1));

    crate::pool::parallel_try_run(workers, |_| {
        loop {
            let gi = next_g.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if gi >= num_groups {
                return Ok(());
            }
            let entry = &entries[gi];
            let g_start = entry.payload_rel_offset as usize;
            let g_end = entries
                .get(gi + 1)
                .map(|e| e.payload_rel_offset as usize)
                .unwrap_or(v6_payload.len());
            let gs = v6_payload.get(g_start..g_end).ok_or(Error::Truncated)?;
            let gh = ParsedGroupHeader::parse(gs, entry.start_block)?;
            let s0_total: usize = (0..gh.count).map(|bi| gh.stream_len(0, bi)).sum();
            let s0_end = gh.data_offset + s0_total;
            let s0_slice = gs.get(gh.data_offset..s0_end).ok_or(Error::Truncated)?;
            let req_mask = compute_group_req_mask(s0_slice, &gh, opt_targets.as_deref());

            let mut col_offs = vec![0usize; gh.slots];
            let mut cur_off = s0_end;
            for c in 0..gh.slots {
                col_offs[c] = cur_off;
                let ct: usize = (0..gh.count).map(|bi| gh.stream_len(c + 1, bi)).sum();
                cur_off += ct;
            }
            let (block_cns, blob) = reconstruct_group_blob(&gh, s0_slice, req_mask, |slot, cur, lc| {
                let off = col_offs[slot] + cur;
                &gs[off..off + lc]
            });
            let slot_ptr = unsafe { &mut *(results_ptr as *mut Option<(usize, Vec<usize>, Vec<u8>)>).add(gi) };
            *slot_ptr = Some((entry.start_block as usize, block_cns, blob));
        }
    })?;

    let group_blobs: Vec<(usize, Vec<usize>, Vec<u8>)> =
        results.into_iter().map(|r| r.unwrap()).collect();
    Ok(assemble_synthetic_v5(&src[..hdr.payload_start], &hdr, group_blobs))
}

pub(crate) fn fast_frame_row_count(frame_bytes: &[u8]) -> Option<Result<usize>> {
    if !is_v6_frame(frame_bytes) && is_slice_resident(frame_bytes) {
        return None;
    }
    let cold = crate::mmap::peek_cold_prefix(frame_bytes, 262_144);
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
    } else if hdr.version == VERSION_V5 {
        let esize = entry_size(hdr.version);
        let mut next_payload = hdr.payload_start;
        let mut next_raw = 0usize;
        for i in 0..hdr.count {
            let p = HEADER_SIZE + i * esize;
            let rn = block_raw_size(probe, &hdr, i, next_raw).ok()?;
            let word = read_u32(probe, p).ok()?;
            let cn = (word & V4_SIZE_MASK) as usize;
            let csum = read_u32(probe, p + 4).ok()?;
            let kind = word >> 30;
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
    let mut total = 0usize;
    let mut prev_open = false;
    let mut next_raw = 0usize;
    for i in 0..hdr.count {
        let rn = block_raw_size(probe, &hdr, i, next_raw).ok()?;
        next_raw += rn;
        let word = block_row_meta_raw(probe, &hdr, i, rn).ok()??;
        let (rows, ends_nl) = unpack_block_row_meta(word);
        if rows > 0 {
            let new_rows = if prev_open { rows - 1 } else { rows };
            total = total.checked_add(new_rows)?;
            prev_open = !ends_nl;
        }
    }
    Some(Ok(total))
}

pub(crate) fn extract_row_range_v6_indexed(
    frame_bytes: &[u8],
    start_row: usize,
    end_row: usize,
    options: &DecodeOptions,
) -> Option<Result<Vec<u8>>> {
    if !is_v6_frame(frame_bytes) {
        return None;
    }
    let cold = crate::mmap::peek_cold_prefix(frame_bytes, 262_144);
    let probe = cold.as_deref().unwrap_or(frame_bytes);
    let hdr = parse_v1_to_v5_header(probe, options.max_output).ok()?;
    if !hdr.has_row_index || probe.len() < hdr.payload_start {
        return None;
    }
    let (total_frame, _) = validate_v6_header_and_dir(probe, &hdr).ok()?;
    if total_frame != frame_bytes.len() {
        return None;
    }
    let mut spans: Vec<(usize, usize, usize, u32)> = Vec::with_capacity(hdr.count);
    let mut cur_global = 0usize;
    let mut prev_open = false;
    let mut next_raw = 0usize;
    for i in 0..hdr.count {
        let rn = block_raw_size(probe, &hdr, i, next_raw).ok()?;
        next_raw += rn;
        let word = block_row_meta_raw(probe, &hdr, i, rn).ok()??;
        let (blk_rows, ends_nl) = unpack_block_row_meta(word);
        let base_row = if prev_open { cur_global - 1 } else { cur_global };
        if blk_rows > 0 {
            cur_global = base_row + blk_rows;
            prev_open = !ends_nl;
        }
        spans.push((base_row, base_row + blk_rows, rn, word));
    }
    let first_blk = spans.partition_point(|&(_, end_excl, _, _)| end_excl <= start_row);
    let mut out = Vec::new();
    for (blk_idx, &(base_row, end_excl, rn, ri_word)) in spans.iter().enumerate().skip(first_blk) {
        if base_row >= end_row {
            break;
        }
        if end_excl <= start_row || rn == 0 {
            continue;
        }
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

