//! Compressed-domain query execution: partial HTTP-Range tile fetch, pyramid resize, crop,
//! grayscale (chroma-skip), subband blur/sharpen, brightness/contrast, sepia, rotate & flip.

use crate::checksum::crc32c;
use crate::color::{merge_planes_rgb, upsample_2x2_bilinear};
use crate::format::VimgIndex;
use crate::range::{coalesce_ranges, RangeSource, RangeStats};
use crate::subband::{decode_detail_level, decode_ll_band};
use crate::wavelet::{level_dims, WaveletTile};
use crate::{Error, Result};

/// Declarative compressed-domain image transform query.
#[derive(Clone, Debug)]
pub struct TransformQuery {
    /// Optional crop rectangle `(x, y, w, h)` in full-resolution source pixel coordinates.
    pub crop: Option<(u32, u32, u32, u32)>,
    /// Optional target output width (preserves aspect ratio if `target_height` is `None`).
    pub target_width: Option<u32>,
    /// Optional target output height (preserves aspect ratio if `target_width` is `None`).
    pub target_height: Option<u32>,
    /// Optional explicit pyramid level (`0` = coarsest, `dwt_levels` = full). Auto-selected if `None`.
    pub force_level: Option<u8>,
    /// Skip `Co`/`Cg` chroma ranges completely over the wire and output grayscale (`1` channel or `3` if RGB requested).
    pub grayscale: bool,
    /// Subband-domain blur (`0.0` = none, `1.0` = strong low-pass attenuation of detail bands).
    pub blur: f32,
    /// Subband-domain sharpen (`0.0` = none, `1.0..=2.0` = detail band boost).
    pub sharpen: f32,
    /// Brightness delta applied directly to `LL` luma (`-128..=128`).
    pub brightness: i16,
    /// Contrast multiplier applied directly in wavelet space (`1.0` = identity).
    pub contrast: f32,
    /// Warm sepia tint applied directly on `Co`/`Cg` planes.
    pub sepia: bool,
    /// Clockwise 90-degree rotations (`0`, `1`=90°, `2`=180°, `3`=270°).
    pub rotate_cw: u8,
    pub flip_h: bool,
    pub flip_v: bool,
    /// Verify per-tile hardware `CRC32C` checksums (default `true`).
    pub verify_crc: bool,
    /// If `true`, salvage corrupt tiles as zeroed subbands instead of aborting decode.
    pub recover_corrupt: bool,
    /// Coalesce nearby Range gaps `<= coalesce_gap` bytes (default `8192`).
    pub coalesce_gap: u64,
}

impl Default for TransformQuery {
    fn default() -> Self {
        Self {
            crop: None,
            target_width: None,
            target_height: None,
            force_level: None,
            grayscale: false,
            blur: 0.0,
            sharpen: 0.0,
            brightness: 0,
            contrast: 1.0,
            sepia: false,
            rotate_cw: 0,
            flip_h: false,
            flip_v: false,
            verify_crc: true,
            recover_corrupt: false,
            coalesce_gap: 2048,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub decoded_level: u8,
    pub tiles_fetched: usize,
    pub tiles_total: usize,
    pub stats: RangeStats,
}

/// Computes target `(out_w, out_h)` and optimal minimum pyramid level `0..=dwt_levels`.
fn select_pyramid_level(
    idx: &VimgIndex,
    crop_w: u32,
    crop_h: u32,
    query: &TransformQuery,
) -> (usize, Option<(u32, u32)>) {
    let max_lvl = idx.header.dwt_levels as usize;
    let target_dims = match (query.target_width, query.target_height) {
        (Some(tw), Some(th)) => Some((tw.max(1), th.max(1))),
        (Some(tw), None) => {
            let th = ((crop_h as u64 * tw as u64 + (crop_w as u64 / 2)) / crop_w.max(1) as u64).max(1) as u32;
            Some((tw.max(1), th))
        }
        (None, Some(th)) => {
            let tw = ((crop_w as u64 * th as u64 + (crop_h as u64 / 2)) / crop_h.max(1) as u64).max(1) as u32;
            Some((tw, th.max(1)))
        }
        (None, None) => None,
    };

    if let Some(fl) = query.force_level {
        return ((fl as usize).min(max_lvl), target_dims);
    }

    let Some((tw, th)) = target_dims else {
        return (max_lvl, None);
    };

    // Pick coarsest pyramid level whose downsampled crop dimensions are >= (tw, th).
    for lvl in 0..=max_lvl {
        let steps = max_lvl - lvl;
        let (lw, lh) = level_dims(crop_w as usize, crop_h as usize, steps);
        if lw >= tw as usize && lh >= th as usize {
            return (lvl, Some((tw, th)));
        }
    }
    (max_lvl, Some((tw, th)))
}

/// Computes the exact coalesced byte ranges needed beyond `prefix_end` to satisfy `query`.
/// Returns `(coalesced_ranges, target_level, tiles_fetched, tiles_total)`.
pub fn plan_query_ranges(
    idx: &VimgIndex,
    query: &TransformQuery,
    prefix_end: u64,
) -> Result<(Vec<(u64, u64)>, u8, usize, usize)> {
    let hdr = &idx.header;
    let (cx, cy, cw, ch) = match query.crop {
        Some((x, y, w, h)) => {
            if w == 0 || h == 0 || x >= hdr.width || y >= hdr.height {
                return Err(Error::InvalidInput("crop rectangle outside image bounds"));
            }
            (x, y, w.min(hdr.width - x), h.min(hdr.height - y))
        }
        None => (0, 0, hdr.width, hdr.height),
    };
    let (target_lvl, _) = select_pyramid_level(idx, cw, ch, query);
    let ts = hdr.tile_size as usize;
    let tx0 = (cx as usize) / ts;
    let ty0 = (cy as usize) / ts;
    let tx1 = ((cx as usize + cw as usize - 1) / ts).min(hdr.tiles_x as usize - 1);
    let ty1 = ((cy as usize + ch as usize - 1) / ts).min(hdr.tiles_y as usize - 1);
    let tiles_fetched = (tx1 - tx0 + 1) * (ty1 - ty0 + 1);

    let fetch_chroma = hdr.channels >= 3 && !query.grayscale && !query.sepia;
    let mut planes = vec![0usize];
    if fetch_chroma {
        planes.push(1);
        planes.push(2);
    }
    if hdr.channels == 4 {
        planes.push(3);
    }

    let mut net_ranges = Vec::new();
    for lvl in 0..=target_lvl {
        for &plane in &planes {
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    let tid = ty * (hdr.tiles_x as usize) + tx;
                    let e = idx.entry(lvl, plane, tid);
                    let end = e.offset + e.len as u64;
                    if end > prefix_end {
                        net_ranges.push((e.offset, end));
                    }
                }
            }
        }
    }
    let effective_gap = query.coalesce_gap.min(hdr.total_bytes / 32);
    let (coalesced, _) = coalesce_ranges(&net_ranges, effective_gap);
    Ok((coalesced, target_lvl as u8, tiles_fetched, hdr.n_tiles()))
}

/// Executes a compressed-domain query directly over any `RangeSource` (local file, slice, or HTTP).
pub fn decode_from_source(source: &dyn RangeSource, query: &TransformQuery) -> Result<DecodedImage> {
    let mut stats = RangeStats::default();
    let (idx, prefix) = VimgIndex::read_from_source_with_prefix(source, &mut stats)?;
    let hdr = &idx.header;

    let (cx, cy, cw, ch) = match query.crop {
        Some((x, y, w, h)) => {
            if w == 0 || h == 0 || x >= hdr.width || y >= hdr.height {
                return Err(Error::InvalidInput("crop rectangle outside image bounds"));
            }
            let cw = w.min(hdr.width - x);
            let ch = h.min(hdr.height - y);
            (x, y, cw, ch)
        }
        None => (0, 0, hdr.width, hdr.height),
    };

    let (target_lvl, target_resize) = select_pyramid_level(&idx, cw, ch, query);
    let max_lvl = hdr.dwt_levels as usize;
    let steps_skipped = max_lvl - target_lvl;

    // Determine exact intersecting tile grid `[tx0..=tx1, ty0..=ty1]`.
    let ts = hdr.tile_size as usize;
    let tx0 = (cx as usize) / ts;
    let ty0 = (cy as usize) / ts;
    let tx1 = ((cx as usize + cw as usize - 1) / ts).min(hdr.tiles_x as usize - 1);
    let ty1 = ((cy as usize + ch as usize - 1) / ts).min(hdr.tiles_y as usize - 1);
    let sel_tx = tx1 - tx0 + 1;
    let sel_ty = ty1 - ty0 + 1;
    let tiles_fetched = sel_tx * sel_ty;

    // Which planes must be fetched? Grayscale or sepia skips chroma completely if grayscale!
    let fetch_chroma = hdr.channels >= 3 && !query.grayscale && !query.sepia;
    let fetch_alpha = hdr.channels == 4;
    let mut planes_to_fetch = vec![0usize];
    if fetch_chroma {
        planes_to_fetch.push(1);
        planes_to_fetch.push(2);
    }
    if fetch_alpha {
        planes_to_fetch.push(3);
    }

    // Build exact byte-range list strictly for `(0..=target_lvl, planes_to_fetch, selected_tiles)`.
    // Any chunk already inside the fetched header `prefix` is reused with 0 extra network bytes!
    let cap = (target_lvl + 1) * planes_to_fetch.len() * tiles_fetched;
    let mut chunk_keys = Vec::with_capacity(cap);
    let mut mapping = vec![(0usize, 0usize); cap];
    let mut net_ranges = Vec::with_capacity(cap);
    let mut net_slots = Vec::with_capacity(cap);
    let prefix_end = prefix.len() as u64;

    for lvl in 0..=target_lvl {
        for &plane in &planes_to_fetch {
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    let k = chunk_keys.len();
                    let tid = ty * (hdr.tiles_x as usize) + tx;
                    let e = idx.entry(lvl, plane, tid);
                    chunk_keys.push((lvl, plane, tx - tx0, ty - ty0, e));
                    let end = e.offset + e.len as u64;
                    if end <= prefix_end {
                        mapping[k] = (0, e.offset as usize);
                    } else {
                        net_slots.push(k);
                        net_ranges.push((e.offset, end));
                    }
                }
            }
        }
    }

    let mut fetched = vec![prefix];
    if !net_ranges.is_empty() {
        let effective_gap = query.coalesce_gap.min(hdr.total_bytes / 32);
        let (coalesced, net_map) = coalesce_ranges(&net_ranges, effective_gap);
        let spans = source.read_ranges(&coalesced)?;
        stats.requests += coalesced.len();
        stats.bytes_fetched += coalesced.iter().map(|&(a, b)| b - a).sum::<u64>();
        fetched.extend(spans);
        for (&k, (ci, rel)) in net_slots.iter().zip(net_map.into_iter()) {
            mapping[k] = (ci + 1, rel);
        }
    }

    // Compute per-level detail gains for subband-domain blur / sharpen / contrast!
    let detail_gains: Vec<f32> = (0..target_lvl)
        .map(|k| {
            let mut g = query.contrast.clamp(0.1, 4.0);
            if query.blur > 0.0 {
                let b = query.blur.clamp(0.0, 1.0);
                // Stronger attenuation on finer detail levels simulates true Gaussian low-pass!
                let frac = ((k + 1) as f32) / (target_lvl.max(1) as f32);
                g *= (1.0 - b * (0.45 + 0.55 * frac)).max(0.0);
            }
            if query.sharpen > 0.0 {
                let s = query.sharpen.clamp(0.0, 2.0);
                let frac = ((k + 1) as f32) / (target_lvl.max(1) as f32);
                g *= 1.0 + s * (0.35 + 0.65 * frac);
            }
            g
        })
        .collect();

    // Precompute decoded pixel canvas dimensions for the selected tile bounding box at `target_lvl`.
    let mut col_x = vec![0usize; sel_tx + 1];
    for ix in 0..sel_tx {
        let (tw, _) = hdr.tile_plane_dims(tx0 + ix, ty0, 0);
        col_x[ix + 1] = col_x[ix] + level_dims(tw, 1, steps_skipped).0;
    }
    let mut row_y = vec![0usize; sel_ty + 1];
    for iy in 0..sel_ty {
        let (_, th) = hdr.tile_plane_dims(tx0, ty0 + iy, 0);
        row_y[iy + 1] = row_y[iy] + level_dims(1, th, steps_skipped).1;
    }
    let box_w = col_x[sel_tx];
    let box_h = row_y[sel_ty];

    let mut y_canvas = vec![0i16; box_w * box_h];
    let mut co_canvas = if fetch_chroma || query.sepia { Some(vec![0i16; box_w * box_h]) } else { None };
    let mut cg_canvas = if fetch_chroma || query.sepia { Some(vec![0i16; box_w * box_h]) } else { None };
    let mut a_canvas = if fetch_alpha { Some(vec![255i16; box_w * box_h]) } else { None };

    // Index fetched slices by `(lvl, plane_slot, ltx, lty)` for O(1) lookup.
    let stride_ty = sel_tx;
    let stride_plane = sel_ty * stride_ty;
    let stride_lvl = planes_to_fetch.len() * stride_plane;

    let mut jobs = Vec::with_capacity(planes_to_fetch.len() * sel_ty * sel_tx);
    for (p_slot, &plane) in planes_to_fetch.iter().enumerate() {
        for lty in 0..sel_ty {
            for ltx in 0..sel_tx {
                jobs.push((p_slot, plane, ltx, lty));
            }
        }
    }

    let decode_one = |&(p_slot, plane, ltx, lty): &(usize, usize, usize, usize)| -> Result<(usize, usize, usize, Vec<i16>)> {
        let (pw_full, ph_full) = hdr.tile_plane_dims(tx0 + ltx, ty0 + lty, plane);
        let (ll_w, ll_h) = level_dims(pw_full, ph_full, max_lvl);

        // Level 0 (LL)
        let k0 = p_slot * stride_plane + lty * stride_ty + ltx;
        let e0 = chunk_keys[k0].4;
        let (ci0, off0) = mapping[k0];
        let blob0 = &fetched[ci0][off0..off0 + e0.len as usize];
        let mut ll = match verify_chunk(blob0, e0.crc32c, hdr.has_crc() && query.verify_crc) {
            Ok(()) => decode_ll_band(blob0, ll_w, ll_h, hdr.quality, plane)?,
            Err(_) if query.recover_corrupt => vec![if plane == 0 { 128 } else { 0 }; ll_w * ll_h],
            Err(err) => return Err(err),
        };

        // Direct LL-domain brightness / contrast on luma plane (zero pixel pass!).
        if plane == 0 && (query.brightness != 0 || (query.contrast - 1.0).abs() > 1e-4) {
            let c = query.contrast.clamp(0.1, 4.0);
            let b = query.brightness as f32;
            for v in &mut ll {
                *v = (((*v as f32 - 128.0) * c) + 128.0 + b).round() as i32;
            }
        }

        // Detail levels 1..=target_lvl
        let mut details = Vec::with_capacity(target_lvl);
        let mut out_dims = Vec::with_capacity(max_lvl);
        let (mut dw, mut dh) = (pw_full, ph_full);
        for _ in 0..max_lvl {
            out_dims.push((dw, dh));
            dw = dw.div_ceil(2);
            dh = dh.div_ceil(2);
        }
        out_dims.reverse();

        for lvl in 1..=target_lvl {
            let k = lvl * stride_lvl + p_slot * stride_plane + lty * stride_ty + ltx;
            let e = chunk_keys[k].4;
            let (ci, off) = mapping[k];
            let blob = &fetched[ci][off..off + e.len as usize];
            let (ow, oh) = out_dims[lvl - 1];
            let d = match verify_chunk(blob, e.crc32c, hdr.has_crc() && query.verify_crc) {
                Ok(()) => decode_detail_level(blob, ow, oh, hdr.quality, plane, lvl - 1, max_lvl)?,
                Err(_) if query.recover_corrupt => {
                    let lw = ow.div_ceil(2);
                    let hw = ow / 2;
                    let lh_r = oh.div_ceil(2);
                    let hh_r = oh / 2;
                    crate::wavelet::DetailBands {
                        out_w: ow,
                        out_h: oh,
                        lh: vec![0; lw * hh_r],
                        hl: vec![0; hw * lh_r],
                        hh: vec![0; hw * hh_r],
                    }
                }
                Err(err) => return Err(err),
            };
            details.push(d);
        }

        let wt = WaveletTile { full_w: pw_full, full_h: ph_full, ll_w, ll_h, ll, details };
        let gains_slice = if plane == 0 { &detail_gains[..] } else { &[] };
        let (mut tile_px, tw_dec, th_dec) = wt.reconstruct(target_lvl, gains_slice);

        let dst_tw = col_x[ltx + 1] - col_x[ltx];
        let dst_th = row_y[lty + 1] - row_y[lty];
        if tw_dec != dst_tw || th_dec != dst_th {
            tile_px = upsample_2x2_bilinear(&tile_px, tw_dec, th_dec, dst_tw, dst_th);
        }
        Ok((plane, ltx, lty, tile_px))
    };

    #[cfg(not(target_arch = "wasm32"))]
    let decoded_tiles: Vec<(usize, usize, usize, Vec<i16>)> = {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(jobs.len());
        if n_threads <= 1 || jobs.len() < 4 {
            jobs.iter().map(decode_one).collect::<Result<Vec<_>>>()?
        } else {
            let chunk_sz = jobs.len().div_ceil(n_threads);
            let mut results: Vec<Result<Vec<(usize, usize, usize, Vec<i16>)>>> = Vec::new();
            std::thread::scope(|s| {
                let handles: Vec<_> = jobs
                    .chunks(chunk_sz)
                    .map(|slice| {
                        let f = &decode_one;
                        s.spawn(move || slice.iter().map(f).collect::<Result<Vec<_>>>())
                    })
                    .collect();
                for h in handles {
                    results.push(h.join().expect("vimg decode worker panicked"));
                }
            });
            let mut flat = Vec::with_capacity(jobs.len());
            for part in results {
                flat.extend(part?);
            }
            flat
        }
    };

    #[cfg(target_arch = "wasm32")]
    let decoded_tiles: Vec<(usize, usize, usize, Vec<i16>)> =
        jobs.iter().map(decode_one).collect::<Result<Vec<_>>>()?;

    for (plane, ltx, lty, tile_px) in decoded_tiles {
        let dst_tw = col_x[ltx + 1] - col_x[ltx];
        let dst_th = row_y[lty + 1] - row_y[lty];
        let dst_plane = match plane {
            0 => &mut y_canvas,
            1 => co_canvas.as_mut().unwrap(),
            2 => cg_canvas.as_mut().unwrap(),
            _ => a_canvas.as_mut().unwrap(),
        };
        let ox = col_x[ltx];
        let oy = row_y[lty];
        for y in 0..dst_th {
            dst_plane[(oy + y) * box_w + ox..(oy + y) * box_w + ox + dst_tw]
                .copy_from_slice(&tile_px[y * dst_tw..(y + 1) * dst_tw]);
        }
    }

    // Apply warm sepia tint directly in YCoCg-R chroma space if requested (0 chroma fetch!).
    if query.sepia {
        if let (Some(co), Some(cg)) = (co_canvas.as_mut(), cg_canvas.as_mut()) {
            co.fill(34);
            cg.fill(6);
        }
    }

    // Sub-tile exact crop trimming in decoded pyramid coordinates.
    let sub_x0 = ((cx as usize) - tx0 * ts) >> steps_skipped;
    let sub_y0 = ((cy as usize) - ty0 * ts) >> steps_skipped;
    let sub_w = (level_dims(cw as usize, ch as usize, steps_skipped).0).min(box_w.saturating_sub(sub_x0)).max(1);
    let sub_h = (level_dims(cw as usize, ch as usize, steps_skipped).1).min(box_h.saturating_sub(sub_y0)).max(1);

    let out_ch = if query.grayscale && !query.sepia {
        if fetch_alpha { 4 } else { 1 }
    } else if query.sepia {
        if fetch_alpha { 4 } else { 3 }
    } else {
        hdr.channels as usize
    };

    let mut cur_w = sub_w;
    let mut cur_h = sub_h;
    let mut pixels = if sub_x0 == 0 && sub_y0 == 0 && sub_w == box_w && sub_h == box_h {
        merge_planes_rgb(
            &y_canvas,
            co_canvas.as_deref(),
            cg_canvas.as_deref(),
            a_canvas.as_deref(),
            box_w,
            box_h,
            out_ch,
        )
    } else {
        let yc = crop_plane_i16(&y_canvas, box_w, sub_x0, sub_y0, sub_w, sub_h);
        let coc = co_canvas.as_deref().map(|p| crop_plane_i16(p, box_w, sub_x0, sub_y0, sub_w, sub_h));
        let cgc = cg_canvas.as_deref().map(|p| crop_plane_i16(p, box_w, sub_x0, sub_y0, sub_w, sub_h));
        let ac = a_canvas.as_deref().map(|p| crop_plane_i16(p, box_w, sub_x0, sub_y0, sub_w, sub_h));
        merge_planes_rgb(&yc, coc.as_deref(), cgc.as_deref(), ac.as_deref(), sub_w, sub_h, out_ch)
    };

    // Final exact resize (only operates on the already-shrunk smallest pyramid level!).
    if let Some((tw, th)) = target_resize {
        if tw as usize != cur_w || th as usize != cur_h {
            pixels = resize_bilinear_u8(&pixels, cur_w, cur_h, tw as usize, th as usize, out_ch);
            cur_w = tw as usize;
            cur_h = th as usize;
        }
    }

    // Geometric orientation (flip / 90°/180°/270° rotate).
    if query.flip_h || query.flip_v || (query.rotate_cw & 3) != 0 {
        let (np, nw, nh) = apply_geom(&pixels, cur_w, cur_h, out_ch, query.rotate_cw & 3, query.flip_h, query.flip_v);
        pixels = np;
        cur_w = nw;
        cur_h = nh;
    }

    Ok(DecodedImage {
        pixels,
        width: cur_w as u32,
        height: cur_h as u32,
        channels: out_ch as u8,
        decoded_level: target_lvl as u8,
        tiles_fetched,
        tiles_total: hdr.n_tiles(),
        stats,
    })
}

fn verify_chunk(blob: &[u8], expected_crc: u32, check: bool) -> Result<()> {
    if check {
        let actual = crc32c(blob);
        if actual != expected_crc {
            return Err(Error::ChecksumMismatch { expected: expected_crc, actual });
        }
    }
    Ok(())
}

fn crop_plane_i16(src: &[i16], stride: usize, x0: usize, y0: usize, w: usize, h: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(w * h);
    for y in y0..y0 + h {
        out.extend_from_slice(&src[y * stride + x0..y * stride + x0 + w]);
    }
    out
}

fn resize_bilinear_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, ch: usize) -> Vec<u8> {
    let mut dst = vec![0u8; dw * dh * ch];
    let sx_scale = sw as f32 / dw as f32;
    let sy_scale = sh as f32 / dh as f32;
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sy_scale - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sx_scale - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f32;
            let w00 = (1.0 - wx) * (1.0 - wy);
            let w10 = wx * (1.0 - wy);
            let w01 = (1.0 - wx) * wy;
            let w11 = wx * wy;
            let o00 = (y0 * sw + x0) * ch;
            let o10 = (y0 * sw + x1) * ch;
            let o01 = (y1 * sw + x0) * ch;
            let o11 = (y1 * sw + x1) * ch;
            let od = (y * dw + x) * ch;
            for c in 0..ch {
                let v = src[o00 + c] as f32 * w00
                    + src[o10 + c] as f32 * w10
                    + src[o01 + c] as f32 * w01
                    + src[o11 + c] as f32 * w11;
                dst[od + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

fn apply_geom(
    src: &[u8],
    w: usize,
    h: usize,
    ch: usize,
    rot_cw: u8,
    flip_h: bool,
    flip_v: bool,
) -> (Vec<u8>, usize, usize) {
    let (dw, dh) = if (rot_cw & 1) == 1 { (h, w) } else { (w, h) };
    let mut dst = vec![0u8; dw * dh * ch];
    for y in 0..h {
        let sy = if flip_v { h - 1 - y } else { y };
        for x in 0..w {
            let sx = if flip_h { w - 1 - x } else { x };
            let (dx, dy) = match rot_cw {
                1 => (h - 1 - sy, sx),
                2 => (w - 1 - sx, h - 1 - sy),
                3 => (sy, w - 1 - sx),
                _ => (sx, sy),
            };
            let si = (y * w + x) * ch;
            let di = (dy * dw + dx) * ch;
            dst[di..di + ch].copy_from_slice(&src[si..si + ch]);
        }
    }
    (dst, dw, dh)
}
