//! In-place warm/hybrid/cold query parser (`parse_v6_for_query`) for `VERSION_V6`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::checksum::crc32c;
use crate::mmap::pread::{pread_at_exact, resolve_backing_fd};
use crate::mmap::sys::{ChunkResidency, CHUNK_BYTES};
use crate::ops::{ColTarget, PredExpr};
use crate::simd::Engine;
use crate::{block, columns, DecodeOptions, Error, Result};

use crate::frame::v5::{
    block_raw_size, block_row_meta_raw, decode_header_dict, parse_v1_to_v5_header,
    unpack_block_row_meta,
};
use crate::frame::{
    entry_size, read_u32, valid_kind_and_size, Block, Frame, HEADER_SIZE, V4_SIZE_MASK,
};
use super::v6_point::load_cold_v6_prefix;
use super::{
    is_v6_frame, parse_v6_block_infos_and_ri, parse_v6_frame_selective, parse_v6_group_dir,
    validate_v6_header_and_dir, ParsedGroupHeader, VERSION_V6,
};

fn collect_pred_cols<'a>(expr: &PredExpr<'a>, out: &mut Vec<ColTarget<'a>>) -> bool {
    match expr {
        PredExpr::True => true,
        PredExpr::Col(p) => {
            if !out.contains(&p.col) { out.push(p.col); }
            true
        }
        PredExpr::All(ch) | PredExpr::Any(ch) => ch.iter().all(|c| collect_pred_cols(c, out)),
        PredExpr::Not(a) => collect_pred_cols(a, out),
        PredExpr::RowContains(_) => false,
    }
}

fn collect_expr_targets<'a>(
    where_expr: Option<&PredExpr<'a>>,
    extra_cols: &[ColTarget<'a>],
    allow_partial_cols: bool,
) -> Option<Vec<ColTarget<'a>>> {
    if !allow_partial_cols { return None; }
    let mut cols = Vec::with_capacity(extra_cols.len() + 4);
    if let Some(expr) = where_expr {
        if !collect_pred_cols(expr, &mut cols) { return None; }
    }
    for &c in extra_cols {
        if !cols.contains(&c) { cols.push(c); }
    }
    Some(cols)
}

fn compute_group_req_mask(
    stream0_buf: &[u8],
    s0_ranges: &[(usize, usize)],
    slots: usize,
    targets: Option<&[ColTarget<'_>]>,
) -> u64 {
    let Some(t_list) = targets else { return u64::MAX; };
    if slots == 0 || s0_ranges.is_empty() { return u64::MAX; }
    if t_list.is_empty() { return 0; }
    let mut mask = 0u64;
    for bi in [0usize, s0_ranges.len() - 1] {
        let (st, en) = s0_ranges[bi];
        match columns::resolve_stream0_required_slots(&stream0_buf[st..en], t_list) {
            Some(m) => mask |= m,
            None => return u64::MAX,
        }
        if s0_ranges.len() == 1 { break; }
    }
    mask
}

/// Parse a `VERSION_V6` frame for a query using `res: &ChunkResidency`:
/// - When all `1 MiB` chunks are resident in the OS page cache (`res.all_resident()`), executes
///   100% zero-copy in-place via [`parse_v6_frame`].
/// - When partially or fully cold, borrows any resident `1 MiB` stream slices zero-copy from `src`
///   and `pread`s only the non-resident required streams (`Stream 0` + queried columns) into
///   `_owned_arena`, binding `Block::payload` and `Block::col_slices` directly without reassembling `v5` blocks.
pub(crate) fn parse_v6_for_query<'a>(
    src: &'a [u8],
    max_output: usize,
    where_expr: Option<&PredExpr<'_>>,
    extra_cols: &[ColTarget<'_>],
    allow_partial_cols: bool,
    threads: usize,
    res: &ChunkResidency,
) -> Result<Frame<'a>> {
    let opt_targets = collect_expr_targets(where_expr, extra_cols, allow_partial_cols);
    if res.all_resident() {
        let hdr = parse_v1_to_v5_header(src, max_output)?;
        return parse_v6_frame_selective(src, &hdr, opt_targets.as_deref());
    }
    let Some((fd, file_off)) = resolve_backing_fd(src) else {
        let hdr = parse_v1_to_v5_header(src, max_output)?;
        return parse_v6_frame_selective(src, &hdr, opt_targets.as_deref());
    };

    let (prefix_owned, hdr) = if res.is_chunk_resident(0) {
        let h = parse_v1_to_v5_header(src, max_output)?;
        if res.is_range_resident(0, h.payload_start) {
            (None, h)
        } else {
            load_cold_v6_prefix(fd, file_off, src.len())?
                .pipe(|(p, h)| Ok::<_, Error>((Some(p), h)))?
        }
    } else {
        let (p, h) = load_cold_v6_prefix(fd, file_off, src.len())?;
        if h.raw_size > max_output {
            return Err(Error::OutputLimit {
                requested: h.raw_size as u64,
                limit: max_output,
            });
        }
        (Some(p), h)
    };
    let prefix: &[u8] = prefix_owned.as_deref().unwrap_or(&src[..hdr.payload_start]);
    let (total_v6_payload, entries) = parse_v6_group_dir(prefix, &hdr)?;
    let dict = decode_header_dict(prefix, &hdr)?;
    let (infos, row_index) = parse_v6_block_infos_and_ri(prefix, &hdr)?;
    let num_groups = entries.len();
    type GroupLoaded = (
        Vec<u8>,
        bool,
        usize,
        Vec<(usize, usize)>,
        usize,
        Vec<Option<(bool, usize, usize, Vec<(usize, usize)>)>>,
    );
    let mut group_results: Vec<Option<GroupLoaded>> = (0..num_groups).map(|_| None).collect();
    let results_ptr = group_results.as_mut_ptr() as usize;
    let next_g = AtomicUsize::new(0);
    let workers = threads.clamp(1, 16).min(num_groups.max(1));

    crate::pool::parallel_try_run(workers, |_| {
        loop {
            let gi = next_g.fetch_add(1, Ordering::Relaxed);
            if gi >= num_groups {
                return Ok(());
            }
            let entry = &entries[gi];
            let g_rel = entry.payload_rel_offset as usize;
            let g_end_rel = entries
                .get(gi + 1)
                .map(|e| e.payload_rel_offset as usize)
                .unwrap_or(total_v6_payload as usize);
            let g_len = g_end_rel - g_rel;
            let g_abs = hdr.payload_start + g_rel;
            let g_disk = file_off + g_abs as u64;

            let bc = entry.block_count as usize;
            let sc = entry.slot_count as usize;
            let est_hdr_s0 = (16 + bc * 12 + (sc + 1) * bc * 4 + bc * 256).min(g_len);
            let mut g_arena: Vec<u8> = Vec::new();

            let (
                s0_in_src,
                s0_base_off,
                s0_total,
                data_offset,
                gh_slots,
                s0_ranges,
                mut col_ranges_by_slot,
                col_totals,
            ) = {
                let (gh, in_src, base_off) = if res.is_range_resident(g_abs, g_abs + est_hdr_s0) {
                    let gh0 = ParsedGroupHeader::parse(&src[g_abs..g_abs + g_len], entry.start_block)?;
                    let s0_end = gh0.data_offset + gh0.stream_total_len(0);
                    if res.is_range_resident(g_abs, g_abs + s0_end) {
                        (gh0, true, g_abs + gh0.data_offset)
                    } else {
                        g_arena.resize(s0_end, 0);
                        pread_at_exact(fd, g_disk, &mut g_arena)
                            .map_err(|e| Error::Backend(e.to_string()))?;
                        let gh_parsed = ParsedGroupHeader::parse(&g_arena, entry.start_block)?;
                        let doff = gh_parsed.data_offset;
                        (gh_parsed, false, doff)
                    }
                } else {
                    let init_read = g_len.min(65_536);
                    g_arena.resize(init_read, 0);
                    pread_at_exact(fd, g_disk, &mut g_arena)
                        .map_err(|e| Error::Backend(e.to_string()))?;
                    let (s0_tot0, data_off) = {
                        let gh0 = ParsedGroupHeader::parse(&g_arena, entry.start_block)?;
                        (gh0.stream_total_len(0), gh0.data_offset)
                    };
                    let s0_end = data_off + s0_tot0;
                    if s0_end > g_arena.len() {
                        let old = g_arena.len();
                        g_arena.resize(s0_end, 0);
                        pread_at_exact(fd, g_disk + old as u64, &mut g_arena[old..])
                            .map_err(|e| Error::Backend(e.to_string()))?;
                    }
                    let gh_parsed = ParsedGroupHeader::parse(&g_arena, entry.start_block)?;
                    (gh_parsed, false, data_off)
                };
                let s0_tot = gh.stream_total_len(0);
                let doff = gh.data_offset;
                let cnt = gh.count;
                let sls = gh.slots;
                let s0_r: Vec<(usize, usize)> =
                    (0..cnt).map(|bi| gh.stream_slice_range(0, bi)).collect();
                let mut col_r: Vec<Vec<(usize, usize)>> = Vec::with_capacity(sls);
                let mut col_t: Vec<usize> = Vec::with_capacity(sls);
                for c in 0..sls {
                    col_t.push(gh.stream_total_len(c + 1));
                    col_r.push((0..cnt).map(|bi| gh.stream_slice_range(c + 1, bi)).collect());
                }
                (in_src, base_off, s0_tot, doff, sls, s0_r, col_r, col_t)
            };

            let s0_slice = if s0_in_src {
                &src[s0_base_off..s0_base_off + s0_total]
            } else {
                &g_arena[s0_base_off..s0_base_off + s0_total]
            };
            let req_mask =
                compute_group_req_mask(s0_slice, &s0_ranges, gh_slots, opt_targets.as_deref());

            let mut col_rel_offs = vec![0usize; gh_slots];
            let mut cur_off = data_offset + s0_total;
            for c in 0..gh_slots {
                col_rel_offs[c] = cur_off;
                cur_off += col_totals[c];
            }

            let mut col_info: Vec<Option<(bool, usize, usize, Vec<(usize, usize)>)>> =
                vec![None; gh_slots];
            let mut c = 0usize;
            while c < gh_slots {
                if ((req_mask >> c) & 1) == 0 {
                    c += 1;
                    continue;
                }
                let c_abs_st = g_abs + col_rel_offs[c];
                let c_abs_en = c_abs_st + col_totals[c];
                if res.is_range_resident(c_abs_st, c_abs_en) {
                    let ranges = std::mem::take(&mut col_ranges_by_slot[c]);
                    col_info[c] = Some((true, c_abs_st, col_totals[c], ranges));
                    c += 1;
                    continue;
                }
                let run_start = c;
                let mut run_bytes = col_totals[c];
                c += 1;
                while c < gh_slots && ((req_mask >> c) & 1) != 0 {
                    let nxt_st = g_abs + col_rel_offs[c];
                    let nxt_en = nxt_st + col_totals[c];
                    if res.is_range_resident(nxt_st, nxt_en) {
                        break;
                    }
                    run_bytes += col_totals[c];
                    c += 1;
                }
                let arena_run_base = g_arena.len();
                g_arena.resize(arena_run_base + run_bytes, 0);
                let disk_start = g_disk + col_rel_offs[run_start] as u64;
                let mut off = 0usize;
                while off < run_bytes {
                    let take = (run_bytes - off).min(CHUNK_BYTES);
                    pread_at_exact(
                        fd,
                        disk_start + off as u64,
                        &mut g_arena[arena_run_base + off..arena_run_base + off + take],
                    )
                    .map_err(|e| Error::Backend(e.to_string()))?;
                    off += take;
                }
                let mut rel = 0usize;
                for slot in run_start..c {
                    let len = col_totals[slot];
                    let ranges = std::mem::take(&mut col_ranges_by_slot[slot]);
                    col_info[slot] = Some((false, arena_run_base + rel, len, ranges));
                    rel += len;
                }
            }

            let slot_ptr = unsafe { &mut *(results_ptr as *mut Option<GroupLoaded>).add(gi) };
            *slot_ptr = Some((
                g_arena,
                s0_in_src,
                s0_base_off,
                s0_ranges,
                gh_slots,
                col_info,
            ));
        }
    })?;

    let total_arena_bytes: usize = group_results
        .iter()
        .map(|r| r.as_ref().unwrap().0.len())
        .sum();
    let mut flat_arena: Vec<u8> = Vec::with_capacity(total_arena_bytes);
    let mut group_arena_bases: Vec<usize> = Vec::with_capacity(num_groups);
    for r in &group_results {
        let ga = &r.as_ref().unwrap().0;
        group_arena_bases.push(flat_arena.len());
        flat_arena.extend_from_slice(ga);
    }
    let arena_ptr = flat_arena.as_ptr();

    let total_col_refs: usize = entries
        .iter()
        .map(|e| (e.block_count as usize) * (e.slot_count as usize))
        .sum();
    let mut owned_col_slices: Vec<&'static [u8]> = Vec::with_capacity(total_col_bytes_or_refs(total_col_refs));
    let mut blocks: Vec<Block<'a>> = Vec::with_capacity(hdr.count);

    for (gi, entry) in entries.iter().enumerate() {
        let sb = entry.start_block as usize;
        let ga_base = group_arena_bases[gi];
        let (_, s0_in_src, s0_base_off, ref s0_ranges, slots, ref col_info) =
            *group_results[gi].as_ref().unwrap();
        let count = s0_ranges.len();
        for bi in 0..count {
            let mut info = infos[sb + bi];
            let (s0_st, s0_en) = s0_ranges[bi];
            let s0_len = s0_en - s0_st;
            let s0_slice: &'a [u8] = if s0_in_src {
                let abs = s0_base_off + s0_st;
                info.payload_offset = abs;
                &src[abs..abs + s0_len]
            } else {
                let off = ga_base + s0_base_off + s0_st;
                info.payload_offset = off;
                unsafe { std::slice::from_raw_parts(arena_ptr.add(off), s0_len) }
            };
            let col_slices: &'a [&'a [u8]] = if slots > 0 {
                let base_idx = owned_col_slices.len();
                for s in 0..slots {
                    let c_slice: &'static [u8] = match col_info[s] {
                        Some((in_src, base_off, _, ref c_ranges)) => {
                            let (c_st, c_en) = c_ranges[bi];
                            let c_len = c_en - c_st;
                            if in_src {
                                let sl = &src[base_off + c_st..base_off + c_en];
                                unsafe { std::mem::transmute(sl) }
                            } else {
                                let off = ga_base + base_off + c_st;
                                unsafe { std::slice::from_raw_parts(arena_ptr.add(off), c_len) }
                            }
                        }
                        None => &[],
                    };
                    owned_col_slices.push(c_slice);
                }
                unsafe {
                    std::slice::from_raw_parts(
                        owned_col_slices.as_ptr().add(base_idx) as *const &'a [u8],
                        slots,
                    )
                }
            } else {
                &[]
            };
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
        _owned_arena: flat_arena,
        _owned_col_slices: owned_col_slices,
    })
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

/// Zero-allocation direct parallel decoder for `VERSION_V6` frames (`decompress` / `decompress_into`),
/// avoiding `Frame::parse`'s per-block `col_slices` heap allocation by computing stream slices on each
/// worker thread's stack.
pub(crate) fn decode_v6_direct_into(
    src: &[u8],
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<()> {
    let hdr = parse_v1_to_v5_header(src, options.max_output)?;
    let (total_v6_payload, entries) = parse_v6_group_dir(src, &hdr)?;
    let total_frame = hdr
        .payload_start
        .checked_add(usize::try_from(total_v6_payload).map_err(|_| Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    if src.len() < total_frame {
        return Err(Error::Truncated);
    }
    if src.len() != total_frame {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    if out.len() != hdr.raw_size {
        return Err(Error::OutputSize {
            expected: hdr.raw_size,
            actual: out.len(),
        });
    }
    let dict = decode_header_dict(src, &hdr)?;
    let (infos, _) = parse_v6_block_infos_and_ri(src, &hdr)?;
    if hdr.count == 0 {
        return Ok(());
    }
    let v6_payload = &src[hdr.payload_start..total_frame];
    let mut groups: Vec<(&[u8], ParsedGroupHeader<'_>, [usize; 65])> =
        Vec::with_capacity(entries.len());
    for (gi, entry) in entries.iter().enumerate() {
        let g_start = entry.payload_rel_offset as usize;
        let g_end = entries
            .get(gi + 1)
            .map(|e| e.payload_rel_offset as usize)
            .unwrap_or(v6_payload.len());
        let g_slice = v6_payload.get(g_start..g_end).ok_or(Error::Truncated)?;
        let gh = ParsedGroupHeader::parse_for_block(g_slice, entry.start_block, 0)?;
        if gh.count != entry.block_count as usize || gh.slots != entry.slot_count as usize {
            return Err(Error::InvalidFrame("v6 group header mismatch with dir"));
        }
        let mut stream_bases = [0usize; 65];
        let mut cur = gh.data_offset;
        for s in 0..=gh.slots {
            stream_bases[s] = cur;
            cur = cur
                .checked_add(gh.stream_total_len(s))
                .ok_or(Error::SizeOverflow)?;
        }
        let expected_trailer = if gh.is_cum { gh.data_offset + 8 } else { 0 };
        if cur > g_slice.len()
            || (cur != g_slice.len() && cur + expected_trailer != g_slice.len())
        {
            return Err(Error::InvalidFrame("v6 group trailing bytes"));
        }
        groups.push((g_slice, gh, stream_bases));
    }

    let engine = Engine::new(options.simd)?;
    let workers = options.threads.min(hdr.count);
    let verify = hdr.checksummed && options.verify_checksum;
    let dict_ref = dict.as_slice();
    let out_addr = out.as_mut_ptr() as usize;
    let next_idx = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let max_batch = (hdr.count / (workers * 8)).clamp(1, 4);
    let tail_cutoff = hdr.count.saturating_sub(workers * 4);

    let decode_one = |idx: usize, blk_buf: &mut Vec<u8>| -> Result<()> {
        let info = &infos[idx];
        let gi = groups.partition_point(|(_, gh, _)| (gh.start_block as usize) + gh.count <= idx);
        let &(g_slice, ref gh, ref stream_bases) = &groups[gi];
        let bi = idx - gh.start_block as usize;
        let (s0_st, s0_en) = gh.stream_slice_range(0, bi);
        if s0_en < s0_st {
            return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
        }
        let s0_off = stream_bases[0] + s0_st;
        let s0_len = s0_en - s0_st;
        let s0_slice = g_slice
            .get(s0_off..s0_off + s0_len)
            .ok_or(Error::Truncated)?;
        let target = unsafe {
            std::slice::from_raw_parts_mut((out_addr + info.raw_offset) as *mut u8, info.raw_size)
        };
        if gh.slots == 0 {
            if s0_len != info.payload_size {
                return Err(Error::InvalidFrame("v6 block cn mismatch"));
            }
            block::decode_with_dict(s0_slice, info.kind, target, engine, true, dict_ref)?;
        } else {
            blk_buf.clear();
            blk_buf.reserve(info.payload_size);
            blk_buf.extend_from_slice(s0_slice);
            for s in 0..gh.slots {
                let (c_st, c_en) = gh.stream_slice_range(s + 1, bi);
                if c_en < c_st {
                    return Err(Error::InvalidFrame("v6 non-monotonic cumulative offset"));
                }
                let c_off = stream_bases[s + 1] + c_st;
                let c_len = c_en - c_st;
                let cs = g_slice.get(c_off..c_off + c_len).ok_or(Error::Truncated)?;
                blk_buf.extend_from_slice(cs);
            }
            if blk_buf.len() != info.payload_size {
                return Err(Error::InvalidFrame("v6 block cn mismatch"));
            }
            block::decode_with_dict(blk_buf, info.kind, target, engine, true, dict_ref)?;
        }
        if verify && crc32c(target) != info.checksum {
            return Err(Error::Checksum { block: idx });
        }
        Ok(())
    };

    if workers <= 1 {
        let mut blk_buf = Vec::with_capacity(65_536);
        for idx in 0..hdr.count {
            decode_one(idx, &mut blk_buf)?;
        }
        return Ok(());
    }
    crate::pool::parallel_try_run(workers, |_| {
        let mut blk_buf = Vec::with_capacity(65_536);
        loop {
            if aborted.load(Ordering::Relaxed) {
                return Ok(());
            }
            let step = if next_idx.load(Ordering::Relaxed) < tail_cutoff {
                max_batch
            } else {
                1
            };
            let start = next_idx.fetch_add(step, Ordering::Relaxed);
            if start >= hdr.count {
                return Ok(());
            }
            let end = (start + step).min(hdr.count);
            for idx in start..end {
                if let Err(e) = decode_one(idx, &mut blk_buf) {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(e);
                }
            }
        }
    })
}

#[inline]
const fn total_col_bytes_or_refs(n: usize) -> usize {
    n
}

trait PipeExt: Sized {
    fn pipe<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}
impl<T> PipeExt for T {}
