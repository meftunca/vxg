//! In-place warm/hybrid/cold query parser (`parse_v6_for_query`, `parse_v6_frame`) for `VERSION_V6`.

use std::cell::RefCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use crate::mmap::pread::{pread_at_exact, resolve_backing_fd};
use crate::mmap::sys::{ChunkResidency, CHUNK_BYTES};
use crate::ops::{ColTarget, PredExpr};
use crate::{columns, DecodeOptions, Error, Result};

use crate::frame::v5::{decode_header_dict, parse_v1_to_v5_header, ParsedHeader};
use crate::frame::{read_u32, Block, Frame};
use super::v6_point::load_cold_v6_prefix;
use super::{
    parse_v6_block_infos_and_ri, parse_v6_group_dir, peek_v6_block0_stream0, ParsedGroupHeader,
    GROUP_V2_FLAG, VERSION_V6,
};

struct CachedV6Parse {
    key: (usize, usize, u32, u32, u64),
    blocks: Vec<Block<'static>>,
    _arena: Arc<Vec<u8>>,
    _col_slices: Arc<Vec<&'static [u8]>>,
    dict: Vec<u8>,
    row_index: Option<Vec<u32>>,
    raw_size: usize,
    block_size: usize,
    checksummed: bool,
}

thread_local! {
    static V6_PARSE_CACHE: RefCell<([Option<CachedV6Parse>; 8], usize)> =
        const { RefCell::new(([None, None, None, None, None, None, None, None], 0)) };
}

pub(crate) fn parse_v6_frame<'a>(src: &'a [u8], hdr: &ParsedHeader) -> Result<Frame<'a>> {
    parse_v6_frame_selective(src, hdr, None)
}

pub(crate) fn parse_v6_frame_selective<'a>(
    src: &'a [u8],
    hdr: &ParsedHeader,
    targets: Option<&[ColTarget<'_>]>,
) -> Result<Frame<'a>> {
    let cache_key_opt = if src.len() >= 131_072 && src.len() >= hdr.payload_start + 8 {
        let first_mask = match targets {
            None => Some(u64::MAX),
            Some([]) => Some(0u64),
            Some(t_list) => peek_v6_block0_stream0(src, usize::MAX)
                .and_then(|(_, _, s0)| columns::resolve_stream0_required_slots(s0, t_list)),
        };
        first_mask.map(|m| (src.as_ptr() as usize, src.len(), read_u32(src, hdr.group_dir_offset + 12).unwrap_or(0), read_u32(src, hdr.payload_start + 4).unwrap_or(0), m))
    } else {
        None
    };
    if let Some(key) = cache_key_opt {
        if let Some(cached) = V6_PARSE_CACHE.with(|cell| {
            let borrow = cell.borrow();
            borrow.0.iter().flatten().find(|s| s.key == key).map(|s| {
                let b_st: Vec<Block<'static>> = s.blocks.iter().map(|b| Block { info: b.info, payload: b.payload, col_slices: b.col_slices }).collect();
                Frame {
                    blocks: unsafe { std::mem::transmute::<Vec<Block<'static>>, Vec<Block<'a>>>(b_st) },
                    dict: s.dict.clone(), row_index: s.row_index.clone(), raw_size: s.raw_size,
                    block_size: s.block_size, checksummed: s.checksummed, version: VERSION_V6,
                    _owned_arena: Vec::new(), _owned_col_slices: Vec::new(),
                }
            })
        }) {
            return Ok(cached);
        }
    }

    let (total_v6_payload, entries) = parse_v6_group_dir(src, hdr)?;
    let total_frame = hdr.payload_start.checked_add(usize::try_from(total_v6_payload).map_err(|_| Error::SizeOverflow)?).ok_or(Error::SizeOverflow)?;
    if src.len() != total_frame {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    let dict = decode_header_dict(src, hdr)?;
    let (infos, row_index) = parse_v6_block_infos_and_ri(src, hdr)?;
    let total_col_refs: usize = entries.iter().map(|e| (e.block_count as usize) * (e.slot_count as usize)).sum();
    let mut owned_col_slices: Vec<&'static [u8]> = Vec::with_capacity(total_col_refs);
    let mut stitch_arena: Vec<u8> = Vec::new();
    let mut stitched_ranges: Vec<(usize, usize, usize)> = Vec::new();
    let mut blocks: Vec<Block<'a>> = Vec::with_capacity(hdr.count);
    let v6_payload = &src[hdr.payload_start..total_frame];
    let (mut all_groups_same_mask, mut seen_mask) = (true, None::<u64>);

    for (gi, entry) in entries.iter().enumerate() {
        let g_start = entry.payload_rel_offset as usize;
        let g_end = entries.get(gi + 1).map(|e| e.payload_rel_offset as usize).unwrap_or(v6_payload.len());
        let g_slice = v6_payload.get(g_start..g_end).ok_or(Error::Truncated)?;
        let gh = ParsedGroupHeader::parse(g_slice, entry.start_block)?;
        if gh.count != entry.block_count as usize || gh.slots != entry.slot_count as usize {
            return Err(Error::InvalidFrame("v6 group header mismatch with dir"));
        }
        let sb = entry.start_block as usize;
        let mut stream_bases = [0usize; 66];
        let mut cur = gh.data_offset;
        for s in 0..gh.num_streams {
            stream_bases[s] = cur;
            cur = cur.checked_add(gh.stream_total_len(s)).ok_or(Error::SizeOverflow)?;
        }
        let expected_trailer = if gh.is_cum { gh.data_offset + 8 } else { 0 };
        if cur > g_slice.len() || (cur != g_slice.len() && cur + expected_trailer != g_slice.len()) {
            return Err(Error::InvalidFrame("v6 group trailing bytes"));
        }
        let req_mask = gh.resolve_req_mask(g_slice, &stream_bases, targets);
        if seen_mask.replace(req_mask).is_some_and(|p| p != req_mask) {
            all_groups_same_mask = false;
        }
        let mut rc_cache: Vec<(u16, usize, usize)> = Vec::new();
        for bi in 0..gh.count {
            let blk_idx = sb + bi;
            let mut info = infos[blk_idx];
            let (s0_slice, s0_off_in_group, exp_s0_len) = gh.resolve_block_s0(
                g_slice, &stream_bases, bi, blk_idx, &mut stitch_arena, &mut stitched_ranges, &mut rc_cache, blocks.len(),
            )?;
            if req_mask == u64::MAX && !gh.verify_block_cn(bi, info.raw_size, exp_s0_len, info.payload_size) {
                return Err(Error::InvalidFrame("v6 block cn mismatch"));
            }
            info.payload_offset = hdr.payload_start + g_start + s0_off_in_group;
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
                    let c_slice = g_slice.get(c_off..stream_bases[s + 1] + c_en).ok_or(Error::Truncated)?;
                    owned_col_slices.push(unsafe { std::mem::transmute::<&'a [u8], &'static [u8]>(c_slice) });
                }
                unsafe { std::slice::from_raw_parts(owned_col_slices.as_ptr().add(base_idx) as *const &'a [u8], gh.slots) }
            } else {
                &[]
            };
            blocks.push(Block { info, payload: s0_slice, col_slices });
        }
    }
    for (b_idx, st, en) in stitched_ranges {
        blocks[b_idx].payload = unsafe { std::slice::from_raw_parts(stitch_arena.as_ptr().add(st), en - st) };
    }
    if let Some(key) = cache_key_opt {
        if all_groups_same_mask && seen_mask == Some(key.4) {
            let arc_arena = Arc::new(stitch_arena);
            let arc_cols = Arc::new(owned_col_slices);
            let cols_ptr = arc_cols.as_ptr();
            let mut cursor = 0usize;
            for b in &mut blocks {
                let n = b.col_slices.len();
                if n > 0 {
                    b.col_slices = unsafe { std::slice::from_raw_parts(cols_ptr.add(cursor) as *const &'a [u8], n) };
                    cursor += n;
                }
            }
            let blocks_static: Vec<Block<'static>> = blocks.iter().map(|b| unsafe {
                Block {
                    info: b.info,
                    payload: std::mem::transmute::<&'a [u8], &'static [u8]>(b.payload),
                    col_slices: std::mem::transmute::<&'a [&'a [u8]], &'static [&'static [u8]]>(b.col_slices),
                }
            }).collect();
            V6_PARSE_CACHE.with(|cell| {
                let mut borrow = cell.borrow_mut();
                let slot_idx = borrow.1 & 7;
                borrow.1 = borrow.1.wrapping_add(1);
                borrow.0[slot_idx] = Some(CachedV6Parse {
                    key, blocks: blocks_static, _arena: arc_arena, _col_slices: arc_cols, dict: dict.clone(),
                    row_index: row_index.clone(), raw_size: hdr.raw_size, block_size: hdr.block_size, checksummed: hdr.checksummed,
                });
            });
            return Ok(Frame {
                blocks, dict, row_index, raw_size: hdr.raw_size, block_size: hdr.block_size,
                checksummed: hdr.checksummed, version: hdr.version, _owned_arena: Vec::new(), _owned_col_slices: Vec::new(),
            });
        }
    }
    Ok(Frame {
        blocks, dict, row_index, raw_size: hdr.raw_size, block_size: hdr.block_size,
        checksummed: hdr.checksummed, version: hdr.version, _owned_arena: stitch_arena, _owned_col_slices: owned_col_slices,
    })
}

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

fn compute_cold_group_req_mask(
    canon_s0: &[u8],
    is_v2: bool,
    stream0_buf: &[u8],
    s0_ranges: &[(usize, usize)],
    slots: usize,
    targets: Option<&[ColTarget<'_>]>,
) -> u64 {
    let Some(t_list) = targets else { return u64::MAX };
    if slots == 0 || s0_ranges.is_empty() { return u64::MAX; }
    if t_list.is_empty() { return 0; }
    if is_v2 && !canon_s0.is_empty() && stream0_buf.is_empty() {
        return columns::resolve_stream0_required_slots(canon_s0, t_list).unwrap_or(u64::MAX);
    }
    let mut mask = 0u64;
    for bi in [0usize, s0_ranges.len() - 1] {
        let (st, en) = s0_ranges[bi];
        let raw = &stream0_buf[st..en];
        let s0_hdr = if is_v2 && !canon_s0.is_empty() && (raw.is_empty() || raw.len() == 6 || raw.first() == Some(&0)) {
            canon_s0
        } else if is_v2 && raw.len() >= 4 {
            &raw[..raw.len() - 4]
        } else {
            raw
        };
        match columns::resolve_stream0_required_slots(s0_hdr, t_list) {
            Some(m) => mask |= m,
            None => return u64::MAX,
        }
        if s0_ranges.len() == 1 { break; }
    }
    mask
}

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
            let (p, h) = load_cold_v6_prefix(fd, file_off, src.len())?;
            (Some(p), h)
        }
    } else {
        let (p, h) = load_cold_v6_prefix(fd, file_off, src.len())?;
        if h.raw_size > max_output {
            return Err(Error::OutputLimit { requested: h.raw_size as u64, limit: max_output });
        }
        (Some(p), h)
    };
    let prefix: &[u8] = prefix_owned.as_deref().unwrap_or(&src[..hdr.payload_start]);
    let (total_v6_payload, entries) = parse_v6_group_dir(prefix, &hdr)?;
    let dict = decode_header_dict(prefix, &hdr)?;
    let (infos, row_index) = parse_v6_block_infos_and_ri(prefix, &hdr)?;
    let num_groups = entries.len();
    type GroupLoaded = (
        Vec<u8>, bool, bool, (usize, usize), usize, Vec<(usize, usize)>, usize,
        Vec<Option<(bool, usize, usize, Vec<(usize, usize)>)>>, Option<(bool, usize, Vec<(usize, usize)>)>,
    );
    let mut group_results: Vec<Option<GroupLoaded>> = (0..num_groups).map(|_| None).collect();
    let results_ptr = group_results.as_mut_ptr() as usize;
    let next_g = AtomicUsize::new(0);
    let workers = threads.clamp(1, 16).min(num_groups.max(1));

    crate::pool::parallel_try_run(workers, |_| {
        loop {
            let gi = next_g.fetch_add(1, Ordering::Relaxed);
            if gi >= num_groups { return Ok(()); }
            let entry = &entries[gi];
            let g_rel = entry.payload_rel_offset as usize;
            let g_end_rel = entries.get(gi + 1).map(|e| e.payload_rel_offset as usize).unwrap_or(total_v6_payload as usize);
            let g_len = g_end_rel - g_rel;
            let g_abs = hdr.payload_start + g_rel;
            let g_disk = file_off + g_abs as u64;
            let (bc, sc) = (entry.block_count as usize, entry.slot_count as usize);
            let est_hdr_s0 = (18 + bc * 12 + (sc + 2) * bc * 4 + 4096).min(g_len);
            let mut g_arena: Vec<u8> = Vec::new();

            let (
                s0_in_src, is_v2, canon_range, s0_base_off, s0_total, data_offset, gh_slots,
                s0_ranges, mut col_ranges_by_slot, col_totals, rr_total, rr_ranges,
            ) = {
                let (gh, in_src, base_off) = if res.is_range_resident(g_abs, g_abs + est_hdr_s0) {
                    let gh0 = ParsedGroupHeader::parse(&src[g_abs..g_abs + g_len], entry.start_block)?;
                    let s0_end = gh0.data_offset + gh0.stream_total_len(0);
                    if res.is_range_resident(g_abs, g_abs + s0_end) {
                        (gh0, true, g_abs + gh0.data_offset)
                    } else {
                        g_arena.resize(s0_end, 0);
                        pread_at_exact(fd, g_disk, &mut g_arena).map_err(|e| Error::Backend(e.to_string()))?;
                        let gh_parsed = ParsedGroupHeader::parse(&g_arena, entry.start_block)?;
                        let doff = gh_parsed.data_offset;
                        (gh_parsed, false, doff)
                    }
                } else {
                    g_arena.resize(g_len.min(65_536), 0);
                    pread_at_exact(fd, g_disk, &mut g_arena).map_err(|e| Error::Backend(e.to_string()))?;
                    if g_arena.len() >= 18 && (u16::from_le_bytes([g_arena[14], g_arena[15]]) & GROUP_V2_FLAG) != 0 {
                        let canon_len = u16::from_le_bytes([g_arena[16], g_arena[17]]) as usize;
                        let sls = (u16::from_le_bytes([g_arena[14], g_arena[15]]) & 0x1fff) as usize;
                        let cnt = u16::from_le_bytes([g_arena[12], g_arena[13]]) as usize;
                        let n_st = if sls > 0 { sls + 2 } else { 1 };
                        let need_hdr = (18 + cnt * 12 + n_st * cnt * 4 + canon_len).min(g_len);
                        if need_hdr > g_arena.len() {
                            let old = g_arena.len();
                            g_arena.resize(need_hdr, 0);
                            pread_at_exact(fd, g_disk + old as u64, &mut g_arena[old..]).map_err(|e| Error::Backend(e.to_string()))?;
                        }
                    }
                    let (s0_tot0, data_off) = {
                        let gh0 = ParsedGroupHeader::parse(&g_arena, entry.start_block)?;
                        (gh0.stream_total_len(0), gh0.data_offset)
                    };
                    let s0_end = data_off + s0_tot0;
                    if s0_end > g_arena.len() {
                        let old = g_arena.len();
                        g_arena.resize(s0_end, 0);
                        pread_at_exact(fd, g_disk + old as u64, &mut g_arena[old..]).map_err(|e| Error::Backend(e.to_string()))?;
                    }
                    let gh_parsed = ParsedGroupHeader::parse(&g_arena, entry.start_block)?;
                    (gh_parsed, false, data_off)
                };
                let (s0_tot, doff, cnt, sls, c_len) = (gh.stream_total_len(0), gh.data_offset, gh.count, gh.slots, gh.canon_s0.len());
                let c_range = if in_src { (g_abs + doff - c_len, g_abs + doff) } else { (doff - c_len, doff) };
                let s0_r: Vec<(usize, usize)> = (0..cnt).map(|bi| gh.stream_slice_range(0, bi)).collect();
                let mut col_r = Vec::with_capacity(sls);
                let mut col_t = Vec::with_capacity(sls);
                for c in 0..sls {
                    col_t.push(gh.stream_total_len(c + 1));
                    col_r.push((0..cnt).map(|bi| gh.stream_slice_range(c + 1, bi)).collect());
                }
                let (rr_t, rr_r) = if gh.is_v2 && sls > 0 {
                    let rt = gh.stream_total_len(sls + 1);
                    let rr = if rt > 0 { (0..cnt).map(|bi| gh.stream_slice_range(sls + 1, bi)).collect() } else { Vec::new() };
                    (rt, rr)
                } else {
                    (0, Vec::new())
                };
                (in_src, gh.is_v2, c_range, base_off, s0_tot, doff, sls, s0_r, col_r, col_t, rr_t, rr_r)
            };

            let (canon_slice, s0_slice) = if s0_in_src {
                (&src[canon_range.0..canon_range.1], &src[s0_base_off..s0_base_off + s0_total])
            } else {
                (&g_arena[canon_range.0..canon_range.1], &g_arena[s0_base_off..s0_base_off + s0_total])
            };
            let req_mask = compute_cold_group_req_mask(canon_slice, is_v2, s0_slice, &s0_ranges, gh_slots, opt_targets.as_deref());

            let mut col_rel_offs = vec![0usize; gh_slots];
            let mut cur_off = data_offset + s0_total;
            for c in 0..gh_slots {
                col_rel_offs[c] = cur_off;
                cur_off += col_totals[c];
            }
            let rr_rel_off = cur_off;

            let mut col_info: Vec<Option<(bool, usize, usize, Vec<(usize, usize)>)>> = vec![None; gh_slots];
            let mut c = 0usize;
            while c < gh_slots {
                if ((req_mask >> c) & 1) == 0 {
                    c += 1;
                    continue;
                }
                let c_abs_st = g_abs + col_rel_offs[c];
                if res.is_range_resident(c_abs_st, c_abs_st + col_totals[c]) {
                    col_info[c] = Some((true, c_abs_st, col_totals[c], std::mem::take(&mut col_ranges_by_slot[c])));
                    c += 1;
                    continue;
                }
                let run_start = c;
                let mut run_bytes = col_totals[c];
                c += 1;
                while c < gh_slots && ((req_mask >> c) & 1) != 0 {
                    let nxt_st = g_abs + col_rel_offs[c];
                    if res.is_range_resident(nxt_st, nxt_st + col_totals[c]) { break; }
                    run_bytes += col_totals[c];
                    c += 1;
                }
                let arena_run_base = g_arena.len();
                g_arena.resize(arena_run_base + run_bytes, 0);
                let disk_start = g_disk + col_rel_offs[run_start] as u64;
                let mut off = 0usize;
                while off < run_bytes {
                    let take = (run_bytes - off).min(CHUNK_BYTES);
                    pread_at_exact(fd, disk_start + off as u64, &mut g_arena[arena_run_base + off..arena_run_base + off + take])
                        .map_err(|e| Error::Backend(e.to_string()))?;
                    off += take;
                }
                let mut rel = 0usize;
                for slot in run_start..c {
                    let len = col_totals[slot];
                    col_info[slot] = Some((false, arena_run_base + rel, len, std::mem::take(&mut col_ranges_by_slot[slot])));
                    rel += len;
                }
            }

            let rr_info = if rr_total > 0 {
                let rr_abs_st = g_abs + rr_rel_off;
                if res.is_range_resident(rr_abs_st, rr_abs_st + rr_total) {
                    Some((true, rr_abs_st, rr_ranges))
                } else {
                    let rr_base = g_arena.len();
                    g_arena.resize(rr_base + rr_total, 0);
                    pread_at_exact(fd, g_disk + rr_rel_off as u64, &mut g_arena[rr_base..rr_base + rr_total])
                        .map_err(|e| Error::Backend(e.to_string()))?;
                    Some((false, rr_base, rr_ranges))
                }
            } else {
                None
            };

            let slot_ptr = unsafe { &mut *(results_ptr as *mut Option<GroupLoaded>).add(gi) };
            *slot_ptr = Some((g_arena, s0_in_src, is_v2, canon_range, s0_base_off, s0_ranges, gh_slots, col_info, rr_info));
        }
    })?;

    let mut stitch_bytes = 0usize;
    for r in &group_results {
        let (_, _, _, c_rng, _, ref s0_r, _, _, ref rr_opt) = *r.as_ref().unwrap();
        if let Some((_, _, ref rr_r)) = *rr_opt {
            for (bi, &(rr_st, rr_en)) in rr_r.iter().enumerate() {
                if rr_en > rr_st {
                    let (s0_st, s0_en) = s0_r[bi];
                    let s0_l = if s0_st == s0_en { c_rng.1 - c_rng.0 } else { s0_en - s0_st };
                    stitch_bytes += s0_l + (rr_en - rr_st);
                }
            }
        }
    }
    let total_arena_bytes: usize = group_results.iter().map(|r| r.as_ref().unwrap().0.len()).sum::<usize>() + stitch_bytes + num_groups * 16_384;
    let mut flat_arena: Vec<u8> = Vec::with_capacity(total_arena_bytes);
    let mut group_arena_bases: Vec<usize> = Vec::with_capacity(num_groups);
    for r in &group_results {
        group_arena_bases.push(flat_arena.len());
        flat_arena.extend_from_slice(&r.as_ref().unwrap().0);
    }

    let total_col_refs: usize = entries.iter().map(|e| (e.block_count as usize) * (e.slot_count as usize)).sum();
    let mut owned_col_slices: Vec<&'static [u8]> = Vec::with_capacity(total_col_refs);
    let mut blocks: Vec<Block<'a>> = Vec::with_capacity(hdr.count);
    let mut deferred_payloads: Vec<(usize, usize, usize)> = Vec::new();

    for (gi, entry) in entries.iter().enumerate() {
        let sb = entry.start_block as usize;
        let ga_base = group_arena_bases[gi];
        let (_, s0_in_src, is_v2, canon_range, s0_base_off, ref s0_ranges, slots, ref col_info, ref rr_info) =
            *group_results[gi].as_ref().unwrap();
        let mut rc_cache: Vec<(u16, usize, usize)> = Vec::new();
        for (bi, &(s0_st, s0_en)) in s0_ranges.iter().enumerate() {
            let blk_idx = sb + bi;
            let mut info = infos[blk_idx];
            let (s0_hdr_in_src, s0_hdr_st, s0_hdr_en) = if s0_st == s0_en && is_v2 {
                if s0_in_src { (true, canon_range.0, canon_range.1) } else { (false, ga_base + canon_range.0, ga_base + canon_range.1) }
            } else {
                let (rc_opt, tpl_tail) = if s0_in_src {
                    let body = ParsedGroupHeader::unwrap_verified_slice(&src[s0_base_off + s0_st..s0_base_off + s0_en], is_v2, blk_idx)?;
                    ((is_v2 && body.len() == 2).then(|| u16::from_le_bytes([body[0], body[1]])), (is_v2 && body.first() == Some(&0)).then(|| body[1..].to_vec()))
                } else {
                    let st = ga_base + s0_base_off + s0_st;
                    let body = ParsedGroupHeader::unwrap_verified_slice(&flat_arena[st..st + (s0_en - s0_st)], is_v2, blk_idx)?;
                    ((is_v2 && body.len() == 2).then(|| u16::from_le_bytes([body[0], body[1]])), (is_v2 && body.first() == Some(&0)).then(|| body[1..].to_vec()))
                };
                if let Some(rc) = rc_opt {
                    let (st, en) = if let Some(&(_, st, en)) = rc_cache.iter().find(|&&(r, _, _)| r == rc) {
                        (st, en)
                    } else {
                        let (st, en) = if s0_in_src {
                            columns::expand_v6_compact_s0(&src[canon_range.0..canon_range.1], rc as usize, &mut flat_arena)
                        } else {
                            let c_vec = flat_arena[ga_base + canon_range.0..ga_base + canon_range.1].to_vec();
                            columns::expand_v6_compact_s0(&c_vec, rc as usize, &mut flat_arena)
                        }.ok_or(Error::InvalidFrame("bad v6 compact s0"))?;
                        rc_cache.push((rc, st, en));
                        (st, en)
                    };
                    (false, st, en)
                } else if let Some(sel_tail) = tpl_tail {
                    let (st, en) = if s0_in_src {
                        columns::expand_v6_tpl_s0(&src[canon_range.0..canon_range.1], &sel_tail, &mut flat_arena)
                    } else {
                        let c_vec = flat_arena[ga_base + canon_range.0..ga_base + canon_range.1].to_vec();
                        columns::expand_v6_tpl_s0(&c_vec, &sel_tail, &mut flat_arena)
                    }.ok_or(Error::InvalidFrame("bad v6 tpl s0"))?;
                    (false, st, en)
                } else if s0_in_src {
                    let st = s0_base_off + s0_st;
                    (true, st, st + (s0_en - s0_st) - usize::from(is_v2) * 4)
                } else {
                    let st = ga_base + s0_base_off + s0_st;
                    (false, st, st + (s0_en - s0_st) - usize::from(is_v2) * 4)
                }
            };
            info.payload_offset = s0_hdr_st;

            let mut stitched = None;
            if let Some((rr_in_src, rr_base, ref rr_ranges)) = *rr_info {
                let (rr_st, rr_en) = rr_ranges[bi];
                if rr_en > rr_st {
                    let rr_body: Vec<u8> = if rr_in_src {
                        ParsedGroupHeader::unwrap_verified_slice(&src[rr_base + rr_st..rr_base + rr_en], is_v2, blk_idx)?.to_vec()
                    } else {
                        let off = ga_base + rr_base + rr_st;
                        ParsedGroupHeader::unwrap_verified_slice(&flat_arena[off..off + (rr_en - rr_st)], is_v2, blk_idx)?.to_vec()
                    };
                    let st = flat_arena.len();
                    if s0_hdr_in_src { flat_arena.extend_from_slice(&src[s0_hdr_st..s0_hdr_en]); } else { flat_arena.extend_from_within(s0_hdr_st..s0_hdr_en); }
                    flat_arena.extend_from_slice(&rr_body);
                    stitched = Some((st, flat_arena.len()));
                }
            }
            let s0_slice: &'a [u8] = if let Some((st, en)) = stitched {
                deferred_payloads.push((blocks.len(), st, en));
                &[]
            } else if s0_hdr_in_src {
                &src[s0_hdr_st..s0_hdr_en]
            } else {
                deferred_payloads.push((blocks.len(), s0_hdr_st, s0_hdr_en));
                &[]
            };

            let col_slices: &'a [&'a [u8]] = if slots > 0 {
                let base_idx = owned_col_slices.len();
                for s in 0..slots {
                    let c_slice: &'static [u8] = match col_info[s] {
                        Some((in_src, base_off, _, ref c_ranges)) => {
                            let (c_st, c_en) = c_ranges[bi];
                            if in_src {
                                unsafe { std::mem::transmute(&src[base_off + c_st..base_off + c_en]) }
                            } else {
                                unsafe { std::slice::from_raw_parts(flat_arena.as_ptr().add(ga_base + base_off + c_st), c_en - c_st) }
                            }
                        }
                        None => &[],
                    };
                    owned_col_slices.push(c_slice);
                }
                unsafe { std::slice::from_raw_parts(owned_col_slices.as_ptr().add(base_idx) as *const &'a [u8], slots) }
            } else {
                &[]
            };
            blocks.push(Block { info, payload: s0_slice, col_slices });
        }
    }
    let arena_ptr = flat_arena.as_ptr();
    for (b_idx, st, en) in deferred_payloads {
        blocks[b_idx].payload = unsafe { std::slice::from_raw_parts(arena_ptr.add(st), en - st) };
    }

    Ok(Frame {
        blocks, dict, row_index, raw_size: hdr.raw_size, block_size: hdr.block_size,
        checksummed: hdr.checksummed, version: hdr.version, _owned_arena: flat_arena, _owned_col_slices: owned_col_slices,
    })
}

pub(crate) fn decode_v6_direct_into(src: &[u8], out: &mut [u8], options: &DecodeOptions) -> Result<()> {
    let hdr = parse_v1_to_v5_header(src, options.max_output)?;
    parse_v6_frame(src, &hdr)?.decode_into(out, options)
}
