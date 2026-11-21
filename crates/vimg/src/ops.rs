//! Compressed-domain query execution: partial HTTP-Range tile fetch, pyramid resize, crop,
//! grayscale (chroma-skip), subband blur/sharpen, brightness/contrast, sepia, rotate & flip.

use crate::checksum::crc32c;
use crate::color::{merge_planes_rgb, merge_planes_u16, upsample_2x2_bilinear};
use crate::entropy::{decode_spatial_palette, CODEC_SPATIAL_PALETTE};
use crate::format::{VimgIndex, MAX_IMAGE_PIXELS, MAX_RESIZE_DIM, MAX_RESIZE_PIXELS};
use crate::meta::VimgMetadata;
use crate::range::{coalesce_ranges, RangeSource, RangeStats};
use crate::subband::{decode_detail_level_bd, decode_ll_band_bd};
use crate::wavelet::{level_dims, WaveletScratch, WaveletTile};
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
    /// Chroma saturation multiplier applied directly on `Co`/`Cg` planes (`0.0` = gray, `1.0` = identity, `2.0` = vivid).
    pub saturation: f32,
    /// Color warmth / white-balance temperature shift on `Co` (`-1.0` = cool blue, `0.0` = neutral, `+1.0` = warm).
    pub temperature: f32,
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
    /// Coalesce nearby Range gaps `<= coalesce_gap` bytes (default `2048`).
    pub coalesce_gap: u64,
    /// Load optional trailing TLV metadata (`CICP`, `EXIF`, `XMP`) when present (default `true`).
    /// Set to `false` for partial remote thumbnails/crops to avoid an extra trailing Range fetch.
    pub load_metadata: bool,
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
            saturation: 1.0,
            temperature: 0.0,
            sepia: false,
            rotate_cw: 0,
            flip_h: false,
            flip_v: false,
            verify_crc: true,
            recover_corrupt: false,
            coalesce_gap: 2048,
            load_metadata: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodedImage {
    /// Packed 8-bit pixels (for 10-bit/12-bit HDR, contains an 8-bit shifted preview; see `pixels_u16`).
    pub pixels: Vec<u8>,
    /// Full-precision 10-bit (`0..=1023`) or 12-bit (`0..=4095`) pixels when `bit_depth > 8`.
    pub pixels_u16: Option<Vec<u16>>,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub bit_depth: u8,
    pub decoded_level: u8,
    pub tiles_fetched: usize,
    pub tiles_total: usize,
    pub stats: RangeStats,
    /// Embedded ICC color profile (if present in the archive).
    pub icc_profile: Option<Vec<u8>>,
    /// Embedded trailing TLV metadata (`CICP`, `EXIF`, `XMP`) when present and `load_metadata` is true.
    pub metadata: Option<VimgMetadata>,
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
#[allow(clippy::type_complexity)]
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
    let bit_depth = hdr.bit_depth();
    let max_sample = hdr.max_sample() as i16;
    let bd_shift = bit_depth.saturating_sub(8);
    let mid_luma = 1i32 << (bit_depth - 1);
    let icc_profile = idx.read_icc_from_source(source, &prefix, query.verify_crc, &mut stats)?;
    let metadata = if query.load_metadata && idx.meta_wire_len > 0 {
        idx.read_metadata_from_source(source, query.verify_crc, &mut stats)?
    } else {
        None
    };

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
    if let Some((tw, th)) = target_resize {
        if tw > MAX_RESIZE_DIM || th > MAX_RESIZE_DIM || (tw as u64) * (th as u64) > MAX_RESIZE_PIXELS {
            return Err(Error::InvalidInput("target resize dimensions exceed safety limits"));
        }
    }
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
        if spans.len() != coalesced.len()
            || spans.iter().zip(coalesced.iter()).any(|(sp, &(a, b))| sp.len() as u64 != b - a)
        {
            return Err(Error::Truncated);
        }
        stats.requests += coalesced.len();
        stats.bytes_fetched += coalesced.iter().map(|&(a, b)| b - a).sum::<u64>();
        fetched.extend(spans);
        for (&k, (ci, rel)) in net_slots.iter().zip(net_map) {
            mapping[k] = (ci + 1, rel);
        }
    }

    // Compute per-level detail gains for subband-domain blur / sharpen / contrast!
    let detail_gains = build_detail_gains(target_lvl, query);

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
    let _ = box_w
        .checked_mul(box_h)
        .filter(|&n| (n as u64) <= MAX_IMAGE_PIXELS)
        .ok_or(Error::InvalidFormat("decoded tile canvas exceeds safety limits"))?;

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

    // Index fetched slices by `(lvl, plane_slot, ltx, lty)` for O(1) lookup.
    let stride_ty = sel_tx;
    let stride_plane = sel_ty * stride_ty;
    let stride_lvl = planes_to_fetch.len() * stride_plane;

    enum TilePixels {
        U8(Vec<u8>),
        U16(Vec<u16>),
    }

    // Decode all planes of one spatial tile `(ltx, lty)` in L1 cache and merge straight to packed RGB/RGBA!
    // Reuses per-worker `WaveletScratch` across tiles to eliminate ~1,500 heap allocations per frame.
    let decode_tile = |tile_idx: usize, wt_scratch: &mut WaveletScratch| -> Result<(usize, usize, TilePixels)> {
        let lty = tile_idx / sel_tx;
        let ltx = tile_idx % sel_tx;
        let dst_tw = col_x[ltx + 1] - col_x[ltx];
        let dst_th = row_y[lty + 1] - row_y[lty];

        let mut y_tile: Option<Vec<i16>> = None;
        let mut co_tile: Option<Vec<i16>> = None;
        let mut cg_tile: Option<Vec<i16>> = None;
        let mut a_tile: Option<Vec<i16>> = None;

        for (p_slot, &plane) in planes_to_fetch.iter().enumerate() {
            let (pw_full, ph_full) = hdr.tile_plane_dims(tx0 + ltx, ty0 + lty, plane);

            // Fast zero-IDWT spatial Palette+RLE shortcut for lossless UI/screenshot/solid-alpha tiles!
            if target_lvl == max_lvl && max_lvl >= 1 && query.blur <= 0.0 && query.sharpen <= 0.0 {
                let kf = max_lvl * stride_lvl + p_slot * stride_plane + lty * stride_ty + ltx;
                let ef = chunk_keys[kf].4;
                let (cif, offf) = mapping[kf];
                let blobf = &fetched[cif][offf..offf + ef.len as usize];
                if blobf.first() == Some(&CODEC_SPATIAL_PALETTE) {
                    let mut tile_px = match verify_chunk(blobf, ef.crc32c, hdr.has_crc() && query.verify_crc) {
                        Ok(()) => decode_spatial_palette(blobf, pw_full, ph_full)?,
                        Err(_) if query.recover_corrupt => vec![if plane == 0 { mid_luma as i16 } else { 0 }; pw_full * ph_full],
                        Err(err) => return Err(err),
                    };
                    if plane == 0 && (query.brightness != 0 || (query.contrast - 1.0).abs() > 1e-4) {
                        let c = query.contrast.clamp(0.1, 4.0);
                        let b = (query.brightness as f32) * ((1u32 << bd_shift) as f32);
                        let mid_f = mid_luma as f32;
                        for v in &mut tile_px {
                            *v = ((((*v as f32 - mid_f) * c) + mid_f + b).round() as i32)
                                .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        }
                    }
                    if pw_full != dst_tw || ph_full != dst_th {
                        tile_px = upsample_2x2_bilinear(&tile_px, pw_full, ph_full, dst_tw, dst_th);
                    }
                    match plane {
                        0 => y_tile = Some(tile_px),
                        1 => co_tile = Some(tile_px),
                        2 => cg_tile = Some(tile_px),
                        _ => a_tile = Some(tile_px),
                    }
                    continue;
                }
            }

            let (ll_w, ll_h) = level_dims(pw_full, ph_full, max_lvl);

            let k0 = p_slot * stride_plane + lty * stride_ty + ltx;
            let e0 = chunk_keys[k0].4;
            let (ci0, off0) = mapping[k0];
            let blob0 = &fetched[ci0][off0..off0 + e0.len as usize];
            let mut ll = match verify_chunk(blob0, e0.crc32c, hdr.has_crc() && query.verify_crc) {
                Ok(()) => decode_ll_band_bd(blob0, ll_w, ll_h, hdr.quality, plane, bit_depth)?,
                Err(_) if query.recover_corrupt => vec![if plane == 0 { mid_luma } else { 0 }; ll_w * ll_h],
                Err(err) => return Err(err),
            };

            if plane == 0 && (query.brightness != 0 || (query.contrast - 1.0).abs() > 1e-4) {
                let c = query.contrast.clamp(0.1, 4.0);
                let b = (query.brightness as f32) * ((1u32 << bd_shift) as f32);
                let mid_f = mid_luma as f32;
                for v in &mut ll {
                    *v = (((*v as f32 - mid_f) * c) + mid_f + b).round() as i32;
                }
            }

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
                    Ok(()) => decode_detail_level_bd(blob, ow, oh, hdr.quality, plane, lvl - 1, hdr.quant_dwt_levels(), bit_depth)?,
                    Err(_) if query.recover_corrupt => crate::wavelet::DetailBands::all_zero(ow, oh),
                    Err(err) => return Err(err),
                };
                details.push(d);
            }

            let wt = WaveletTile { full_w: pw_full, full_h: ph_full, ll_w, ll_h, ll, details };
            let gains_slice = if plane == 0 { &detail_gains[..] } else { &[] };
            let (mut tile_px, tw_dec, th_dec) =
                wt.reconstruct_mode_with(target_lvl, gains_slice, hdr.is_dwt_97(), wt_scratch);
            if tw_dec != dst_tw || th_dec != dst_th {
                tile_px = upsample_2x2_bilinear(&tile_px, tw_dec, th_dec, dst_tw, dst_th);
            }
            match plane {
                0 => y_tile = Some(tile_px),
                1 => co_tile = Some(tile_px),
                2 => cg_tile = Some(tile_px),
                _ => a_tile = Some(tile_px),
            }
        }

        apply_color_adjust(&mut co_tile, &mut cg_tile, dst_tw * dst_th, bd_shift, query);

        let px = if bit_depth == 8 {
            TilePixels::U8(merge_planes_rgb(
                y_tile.as_deref().unwrap(),
                co_tile.as_deref(),
                cg_tile.as_deref(),
                a_tile.as_deref(),
                dst_tw,
                dst_th,
                out_ch,
            ))
        } else {
            TilePixels::U16(merge_planes_u16(
                y_tile.as_deref().unwrap(),
                co_tile.as_deref(),
                cg_tile.as_deref(),
                a_tile.as_deref(),
                dst_tw,
                dst_th,
                out_ch,
                max_sample,
            ))
        };
        Ok((ltx, lty, px))
    };

    #[cfg(not(target_arch = "wasm32"))]
    let decoded_tiles: Vec<(usize, usize, TilePixels)> = {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(tiles_fetched);
        if n_threads <= 1 || tiles_fetched < 2 {
            let mut scratch = WaveletScratch::default();
            (0..tiles_fetched).map(|i| decode_tile(i, &mut scratch)).collect::<Result<Vec<_>>>()?
        } else {
            let next_idx = AtomicUsize::new(0);
            type TileBatch = Result<Vec<(usize, usize, TilePixels)>>;
            let mut results: Vec<TileBatch> = Vec::with_capacity(n_threads);
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..n_threads)
                    .map(|_| {
                        let f = &decode_tile;
                        let counter = &next_idx;
                        s.spawn(move || {
                            let mut scratch = WaveletScratch::default();
                            let mut local = Vec::new();
                            loop {
                                let idx = counter.fetch_add(1, Ordering::Relaxed);
                                if idx >= tiles_fetched {
                                    break;
                                }
                                local.push(f(idx, &mut scratch)?);
                            }
                            Ok(local)
                        })
                    })
                    .collect();
                for h in handles {
                    results.push(h.join().expect("vimg decode worker panicked"));
                }
            });
            let mut flat = Vec::with_capacity(tiles_fetched);
            for part in results {
                flat.extend(part?);
            }
            flat
        }
    };

    #[cfg(target_arch = "wasm32")]
    let decoded_tiles: Vec<(usize, usize, TilePixels)> = {
        let mut scratch = WaveletScratch::default();
        (0..tiles_fetched).map(|i| decode_tile(i, &mut scratch)).collect::<Result<Vec<_>>>()?
    };

    let mut cur_w = sub_w;
    let mut cur_h = sub_h;
    let sub_x1 = sub_x0 + sub_w;
    let sub_y1 = sub_y0 + sub_h;

    if bit_depth == 8 {
        let mut pixels = vec![0u8; sub_w * sub_h * out_ch];
        for (ltx, lty, tile_px) in decoded_tiles {
            let TilePixels::U8(tile_rgb) = tile_px else { unreachable!() };
            let tx_px0 = col_x[ltx];
            let ty_px0 = row_y[lty];
            let dst_tw = col_x[ltx + 1] - tx_px0;
            let dst_th = row_y[lty + 1] - ty_px0;
            let ix0 = tx_px0.max(sub_x0);
            let iy0 = ty_px0.max(sub_y0);
            let ix1 = (tx_px0 + dst_tw).min(sub_x1);
            let iy1 = (ty_px0 + dst_th).min(sub_y1);
            if ix0 >= ix1 || iy0 >= iy1 {
                continue;
            }
            let copy_cols = (ix1 - ix0) * out_ch;
            let sx_off = (ix0 - tx_px0) * out_ch;
            let dx_off = (ix0 - sub_x0) * out_ch;
            for y in iy0..iy1 {
                let src_row = (y - ty_px0) * dst_tw * out_ch + sx_off;
                let dst_row = (y - sub_y0) * sub_w * out_ch + dx_off;
                pixels[dst_row..dst_row + copy_cols]
                    .copy_from_slice(&tile_rgb[src_row..src_row + copy_cols]);
            }
        }
        if let Some((tw, th)) = target_resize {
            if tw as usize != cur_w || th as usize != cur_h {
                pixels = resize_bicubic_u8(&pixels, cur_w, cur_h, tw as usize, th as usize, out_ch);
                cur_w = tw as usize;
                cur_h = th as usize;
            }
        }
        if query.flip_h || query.flip_v || (query.rotate_cw & 3) != 0 {
            let (np, nw, nh) = apply_geom(&pixels, cur_w, cur_h, out_ch, query.rotate_cw & 3, query.flip_h, query.flip_v);
            pixels = np;
            cur_w = nw;
            cur_h = nh;
        }
        Ok(DecodedImage {
            pixels,
            pixels_u16: None,
            width: cur_w as u32,
            height: cur_h as u32,
            channels: out_ch as u8,
            bit_depth,
            decoded_level: target_lvl as u8,
            tiles_fetched,
            tiles_total: hdr.n_tiles(),
            stats,
            icc_profile,
            metadata,
        })
    } else {
        let mut pixels16 = vec![0u16; sub_w * sub_h * out_ch];
        for (ltx, lty, tile_px) in decoded_tiles {
            let TilePixels::U16(tile_rgb) = tile_px else { unreachable!() };
            let tx_px0 = col_x[ltx];
            let ty_px0 = row_y[lty];
            let dst_tw = col_x[ltx + 1] - tx_px0;
            let dst_th = row_y[lty + 1] - ty_px0;
            let ix0 = tx_px0.max(sub_x0);
            let iy0 = ty_px0.max(sub_y0);
            let ix1 = (tx_px0 + dst_tw).min(sub_x1);
            let iy1 = (ty_px0 + dst_th).min(sub_y1);
            if ix0 >= ix1 || iy0 >= iy1 {
                continue;
            }
            let copy_cols = (ix1 - ix0) * out_ch;
            let sx_off = (ix0 - tx_px0) * out_ch;
            let dx_off = (ix0 - sub_x0) * out_ch;
            for y in iy0..iy1 {
                let src_row = (y - ty_px0) * dst_tw * out_ch + sx_off;
                let dst_row = (y - sub_y0) * sub_w * out_ch + dx_off;
                pixels16[dst_row..dst_row + copy_cols]
                    .copy_from_slice(&tile_rgb[src_row..src_row + copy_cols]);
            }
        }
        if let Some((tw, th)) = target_resize {
            if tw as usize != cur_w || th as usize != cur_h {
                pixels16 = resize_bilinear_u16(&pixels16, cur_w, cur_h, tw as usize, th as usize, out_ch, max_sample as f32);
                cur_w = tw as usize;
                cur_h = th as usize;
            }
        }
        if query.flip_h || query.flip_v || (query.rotate_cw & 3) != 0 {
            let (np, nw, nh) = apply_geom(&pixels16, cur_w, cur_h, out_ch, query.rotate_cw & 3, query.flip_h, query.flip_v);
            pixels16 = np;
            cur_w = nw;
            cur_h = nh;
        }
        let preview_u8: Vec<u8> = pixels16.iter().map(|&v| (v >> bd_shift).min(255) as u8).collect();
        Ok(DecodedImage {
            pixels: preview_u8,
            pixels_u16: Some(pixels16),
            width: cur_w as u32,
            height: cur_h as u32,
            channels: out_ch as u8,
            bit_depth,
            decoded_level: target_lvl as u8,
            tiles_fetched,
            tiles_total: hdr.n_tiles(),
            stats,
            icc_profile,
            metadata,
        })
    }
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

pub(crate) fn build_detail_gains(target_lvl: usize, query: &TransformQuery) -> Vec<f32> {
    (0..target_lvl)
        .map(|k| {
            let mut g = query.contrast.clamp(0.1, 4.0);
            if query.blur > 0.0 {
                let b = query.blur.clamp(0.0, 1.0);
                let dist = (target_lvl - 1 - k) as f32;
                let atten = (1.0 - b * (0.95 / (1.0 + dist * 0.75))).clamp(0.0, 1.0);
                g *= atten;
            }
            if query.sharpen > 0.0 {
                let sh = query.sharpen.clamp(0.0, 2.5);
                let boost = if k + 1 == target_lvl { 1.0 + sh * 0.85 } else { 1.0 + sh * 0.35 };
                g *= boost;
            }
            g
        })
        .collect()
}

pub(crate) fn apply_color_adjust(
    co_tile: &mut Option<Vec<i16>>,
    cg_tile: &mut Option<Vec<i16>>,
    len: usize,
    bd_shift: u8,
    query: &TransformQuery,
) {
    if query.sepia {
        let co_sep = 34i16 << bd_shift;
        let cg_sep = 6i16 << bd_shift;
        *co_tile = Some(vec![co_sep; len]);
        *cg_tile = Some(vec![cg_sep; len]);
        return;
    }
    let sat = query.saturation.clamp(0.0, 3.0);
    let temp = query.temperature.clamp(-1.0, 1.0);
    let do_sat = (sat - 1.0).abs() > 1e-4;
    let do_temp = temp.abs() > 1e-4;
    if !do_sat && !do_temp {
        return;
    }
    let sat_q8 = (sat * 256.0).round() as i32;
    let temp_co = (temp * 48.0 * ((1u32 << bd_shift) as f32)).round() as i32;
    if let Some(co) = co_tile.as_mut() {
        for v in co.iter_mut() {
            let mut val = *v as i32;
            if do_sat {
                val = (val * sat_q8 + 128) >> 8;
            }
            if do_temp {
                val += temp_co;
            }
            *v = val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
    if do_sat {
        if let Some(cg) = cg_tile.as_mut() {
            for v in cg.iter_mut() {
                let val = ((*v as i32) * sat_q8 + 128) >> 8;
                *v = val.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }
}

#[inline(always)]
fn catmull_rom_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

/// Crisp separable 2-pass Catmull-Rom bicubic resize for sharp, artifact-free thumbnails.
pub(crate) fn resize_bicubic_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, ch: usize) -> Vec<u8> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    // Pass 1: horizontal `sw x sh -> dw x sh` into f32 intermediate
    let mut tmp = vec![0.0f32; dw * sh * ch];
    let sx_scale = sw as f32 / dw as f32;
    let sw_max = (sw - 1) as isize;
    let mut x_taps = Vec::with_capacity(dw);
    for x in 0..dw {
        let fx = (x as f32 + 0.5) * sx_scale - 0.5;
        let ix = fx.floor() as isize;
        let t = fx - ix as f32;
        let w = catmull_rom_weights(t);
        let idx = [
            (ix - 1).clamp(0, sw_max) as usize,
            ix.clamp(0, sw_max) as usize,
            (ix + 1).clamp(0, sw_max) as usize,
            (ix + 2).clamp(0, sw_max) as usize,
        ];
        x_taps.push((idx, w));
    }
    for y in 0..sh {
        let srow = y * sw * ch;
        let drow = y * dw * ch;
        for (x, &(idx, w)) in x_taps.iter().enumerate() {
            let o0 = srow + idx[0] * ch;
            let o1 = srow + idx[1] * ch;
            let o2 = srow + idx[2] * ch;
            let o3 = srow + idx[3] * ch;
            let od = drow + x * ch;
            for c in 0..ch {
                tmp[od + c] = src[o0 + c] as f32 * w[0]
                    + src[o1 + c] as f32 * w[1]
                    + src[o2 + c] as f32 * w[2]
                    + src[o3 + c] as f32 * w[3];
            }
        }
    }

    // Pass 2: vertical `dw x sh -> dw x dh`
    let mut dst = vec![0u8; dw * dh * ch];
    let sy_scale = sh as f32 / dh as f32;
    let sh_max = (sh - 1) as isize;
    let row_elems = dw * ch;
    for y in 0..dh {
        let fy = (y as f32 + 0.5) * sy_scale - 0.5;
        let iy = fy.floor() as isize;
        let t = fy - iy as f32;
        let w = catmull_rom_weights(t);
        let r0 = ((iy - 1).clamp(0, sh_max) as usize) * row_elems;
        let r1 = (iy.clamp(0, sh_max) as usize) * row_elems;
        let r2 = ((iy + 1).clamp(0, sh_max) as usize) * row_elems;
        let r3 = ((iy + 2).clamp(0, sh_max) as usize) * row_elems;
        let rd = y * row_elems;
        for i in 0..row_elems {
            let v = tmp[r0 + i] * w[0] + tmp[r1 + i] * w[1] + tmp[r2 + i] * w[2] + tmp[r3 + i] * w[3];
            dst[rd + i] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    dst
}

pub(crate) fn resize_bilinear_u16(
    src: &[u16],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
    ch: usize,
    max_val: f32,
) -> Vec<u16> {
    let mut dst = vec![0u16; dw * dh * ch];
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
                dst[od + c] = v.round().clamp(0.0, max_val) as u16;
            }
        }
    }
    dst
}

pub(crate) fn apply_geom_pub<T: Copy + Default>(
    src: &[T],
    w: usize,
    h: usize,
    ch: usize,
    rot_cw: u8,
    flip_h: bool,
    flip_v: bool,
) -> (Vec<T>, usize, usize) {
    apply_geom(src, w, h, ch, rot_cw, flip_h, flip_v)
}

fn apply_geom<T: Copy + Default>(
    src: &[T],
    w: usize,
    h: usize,
    ch: usize,
    rot_cw: u8,
    flip_h: bool,
    flip_v: bool,
) -> (Vec<T>, usize, usize) {
    let (dw, dh) = if (rot_cw & 1) == 1 { (h, w) } else { (w, h) };
    let mut dst = vec![T::default(); dw * dh * ch];
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
