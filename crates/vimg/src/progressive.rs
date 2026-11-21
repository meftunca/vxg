//! Stateful progressive wavelet decoder (`L0 -> L1 -> L2 -> L3`) over any `RangeSource`.
//!
//! Fetches and decodes each pyramid level strictly ONCE, caching reconstructed `LL` and detail
//! bands so advancing from `1/8` preview to `1/4`, `1/2`, and `1/1` full resolution never
//! re-downloads or re-decodes earlier levels.

use crate::checksum::crc32c;
use crate::color::{
    merge_planes_rgb, merge_planes_u16, merge_planes_u16_i32, upsample_2x2_bilinear,
    upsample_2x2_bilinear_i32,
};
use crate::entropy::{decode_spatial_palette, CODEC_SPATIAL_PALETTE};
use crate::format::VimgIndex;
use crate::ops::{
    apply_color_adjust, apply_color_adjust_i32, apply_geom_pub, build_detail_gains,
    resize_bicubic_u8, resize_bilinear_u16, DecodedImage, TransformQuery,
};
use crate::range::{coalesce_ranges, RangeSource, RangeStats};
use crate::subband::{decode_detail_level_bd, decode_ll_band_bd};
use crate::wavelet::{level_dims, DetailBands, WaveletScratch, WaveletTile};
use crate::{Error, Result};

struct CachedTilePlane {
    pw_full: usize,
    ph_full: usize,
    ll_w: usize,
    ll_h: usize,
    ll: Vec<i32>,
    details: Vec<DetailBands>,
    spatial_override: Option<Vec<i16>>,
}

/// Stateful progressive `VIMG1` decoder: call [`Self::open`] once, then call [`Self::next_level`]
/// repeatedly to progressively sharpen from `L0 (1/8)` up to full resolution without re-fetching
/// or re-decoding previous pyramid levels.
pub struct ProgressiveDecoder {
    index: VimgIndex,
    prefix: Vec<u8>,
    query: TransformQuery,
    stats: RangeStats,
    cx: u32,
    cy: u32,
    cw: u32,
    ch: u32,
    tx0: usize,
    ty0: usize,
    sel_tx: usize,
    sel_ty: usize,
    planes: Vec<usize>,
    cached: Vec<CachedTilePlane>,
    icc_profile: Option<Vec<u8>>,
    next_lvl: usize,
    max_lvl: usize,
}

impl ProgressiveDecoder {
    /// Reads only the header + directory prefix from `source` and prepares progressive streaming.
    pub fn open(source: &dyn RangeSource, query: TransformQuery) -> Result<Self> {
        let mut stats = RangeStats::default();
        let (index, prefix) = VimgIndex::read_from_source_with_prefix(source, &mut stats)?;
        let hdr = &index.header;

        let (cx, cy, cw, ch) = match query.crop {
            Some((x, y, w, h)) => {
                if w == 0 || h == 0 || x >= hdr.width || y >= hdr.height {
                    return Err(Error::InvalidInput("crop rectangle outside image bounds"));
                }
                (x, y, w.min(hdr.width - x), h.min(hdr.height - y))
            }
            None => (0, 0, hdr.width, hdr.height),
        };

        let icc_profile = index.read_icc_from_source(source, &prefix, query.verify_crc, &mut stats)?;
        let ts = hdr.tile_size as usize;
        let tx0 = (cx as usize) / ts;
        let ty0 = (cy as usize) / ts;
        let tx1 = ((cx as usize + cw as usize - 1) / ts).min(hdr.tiles_x as usize - 1);
        let ty1 = ((cy as usize + ch as usize - 1) / ts).min(hdr.tiles_y as usize - 1);
        let sel_tx = tx1 - tx0 + 1;
        let sel_ty = ty1 - ty0 + 1;

        let fetch_chroma = hdr.channels >= 3 && !query.grayscale && !query.sepia;
        let mut planes = vec![0usize];
        if fetch_chroma {
            planes.push(1);
            planes.push(2);
        }
        if hdr.channels == 4 {
            planes.push(3);
        }

        let max_lvl = hdr.dwt_levels as usize;
        let mut cached = Vec::with_capacity(sel_tx * sel_ty * planes.len());
        for lty in 0..sel_ty {
            for ltx in 0..sel_tx {
                for &plane in &planes {
                    let (pw_full, ph_full) = hdr.tile_plane_dims(tx0 + ltx, ty0 + lty, plane);
                    let (ll_w, ll_h) = level_dims(pw_full, ph_full, max_lvl);
                    cached.push(CachedTilePlane {
                        pw_full,
                        ph_full,
                        ll_w,
                        ll_h,
                        ll: Vec::new(),
                        details: Vec::with_capacity(max_lvl),
                        spatial_override: None,
                    });
                }
            }
        }

        Ok(Self {
            index,
            prefix,
            query,
            stats,
            cx,
            cy,
            cw,
            ch,
            tx0,
            ty0,
            sel_tx,
            sel_ty,
            planes,
            cached,
            icc_profile,
            next_lvl: 0,
            max_lvl,
        })
    }

    pub fn total_levels(&self) -> usize {
        self.max_lvl + 1
    }

    pub fn current_level(&self) -> Option<u8> {
        self.next_lvl.checked_sub(1).map(|v| v as u8)
    }

    pub fn stats(&self) -> RangeStats {
        self.stats
    }

    /// Fetches and decodes strictly the NEXT pyramid level (`0 -> 1 -> ... -> dwt_levels`),
    /// returning `Ok(None)` once full resolution has already been produced.
    pub fn next_level(&mut self, source: &dyn RangeSource) -> Result<Option<DecodedImage>> {
        if self.next_lvl > self.max_lvl {
            return Ok(None);
        }
        let lvl = self.next_lvl;
        let hdr = &self.index.header;
        let bit_depth = hdr.bit_depth();
        let max_sample = hdr.max_sample() as i32;
        let bd_shift = bit_depth.saturating_sub(8);
        let mid_luma = 1i32 << (bit_depth - 1);
        let tiles_fetched = self.sel_tx * self.sel_ty;
        let n_planes = self.planes.len();
        let prefix_end = self.prefix.len() as u64;

        // Collect strictly the chunks for THIS single level `lvl`!
        let mut keys = Vec::with_capacity(tiles_fetched * n_planes);
        let mut mapping = vec![(0usize, 0usize); tiles_fetched * n_planes];
        let mut net_ranges = Vec::new();
        let mut net_slots = Vec::new();

        for (p_slot, &plane) in self.planes.iter().enumerate() {
            for lty in 0..self.sel_ty {
                for ltx in 0..self.sel_tx {
                    let k = keys.len();
                    let tid = (self.ty0 + lty) * (hdr.tiles_x as usize) + (self.tx0 + ltx);
                    let e = self.index.entry(lvl, plane, tid);
                    keys.push((ltx, lty, p_slot, plane, e));
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

        let mut fetched: Vec<Vec<u8>> = Vec::new();
        if !net_ranges.is_empty() {
            let effective_gap = self.query.coalesce_gap.min(hdr.total_bytes / 32);
            let (coalesced, net_map) = coalesce_ranges(&net_ranges, effective_gap);
            let spans = source.read_ranges(&coalesced)?;
            if spans.len() != coalesced.len()
                || spans.iter().zip(coalesced.iter()).any(|(sp, &(a, b))| sp.len() as u64 != b - a)
            {
                return Err(Error::Truncated);
            }
            self.stats.requests += coalesced.len();
            self.stats.bytes_fetched += coalesced.iter().map(|&(a, b)| b - a).sum::<u64>();
            fetched = spans;
            for (&k, (ci, rel)) in net_slots.iter().zip(net_map) {
                mapping[k] = (ci + 1, rel);
            }
        }

        // Decode newly fetched chunks into the persistent cache
        for (k, &(ltx, lty, p_slot, plane, e)) in keys.iter().enumerate() {
            let (ci, off) = mapping[k];
            let blob = if ci == 0 {
                &self.prefix[off..off + e.len as usize]
            } else {
                &fetched[ci - 1][off..off + e.len as usize]
            };
            let c_idx = (lty * self.sel_tx + ltx) * n_planes + p_slot;
            let slot = &mut self.cached[c_idx];
            let crc_ok = if hdr.has_crc() && self.query.verify_crc {
                let actual = crc32c(blob);
                if actual != e.crc32c {
                    if self.query.recover_corrupt {
                        false
                    } else {
                        return Err(Error::ChecksumMismatch { expected: e.crc32c, actual });
                    }
                } else {
                    true
                }
            } else {
                true
            };

            if lvl == 0 {
                let mut ll = if crc_ok {
                    decode_ll_band_bd(blob, slot.ll_w, slot.ll_h, hdr.quality, plane, bit_depth)?
                } else {
                    vec![if plane == 0 { mid_luma } else { 0 }; slot.ll_w * slot.ll_h]
                };
                if plane == 0 && (self.query.brightness != 0 || (self.query.contrast - 1.0).abs() > 1e-4) {
                    let c = self.query.contrast.clamp(0.1, 4.0);
                    let b = (self.query.brightness as f32) * ((1u32 << bd_shift) as f32);
                    let mid_f = mid_luma as f32;
                    for v in &mut ll {
                        *v = (((*v as f32 - mid_f) * c) + mid_f + b).round() as i32;
                    }
                }
                slot.ll = ll;
            } else if lvl == self.max_lvl && blob.first() == Some(&CODEC_SPATIAL_PALETTE) {
                let mut sp = if crc_ok {
                    decode_spatial_palette(blob, slot.pw_full, slot.ph_full)?
                } else {
                    vec![if plane == 0 { mid_luma as i16 } else { 0 }; slot.pw_full * slot.ph_full]
                };
                if plane == 0 && (self.query.brightness != 0 || (self.query.contrast - 1.0).abs() > 1e-4) {
                    let c = self.query.contrast.clamp(0.1, 4.0);
                    let b = (self.query.brightness as f32) * ((1u32 << bd_shift) as f32);
                    let mid_f = mid_luma as f32;
                    for v in &mut sp {
                        *v = ((((*v as f32 - mid_f) * c) + mid_f + b).round() as i32)
                            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    }
                }
                slot.spatial_override = Some(sp);
            } else {
                let (ow, oh) = level_dims(slot.pw_full, slot.ph_full, self.max_lvl - lvl);
                let d = if crc_ok {
                    decode_detail_level_bd(blob, ow, oh, hdr.quality, plane, lvl - 1, hdr.quant_dwt_levels(), bit_depth)?
                } else {
                    DetailBands::all_zero(ow, oh)
                };
                slot.details.push(d);
            }
        }

        // Reconstruct current level `lvl` from cached bands
        let steps_skipped = self.max_lvl - lvl;
        let mut col_x = vec![0usize; self.sel_tx + 1];
        for ltx in 0..self.sel_tx {
            let (tw_full, _) = hdr.tile_plane_dims(self.tx0 + ltx, self.ty0, 0);
            col_x[ltx + 1] = col_x[ltx] + level_dims(tw_full, 1, steps_skipped).0;
        }
        let mut row_y = vec![0usize; self.sel_ty + 1];
        for lty in 0..self.sel_ty {
            let (_, th_full) = hdr.tile_plane_dims(self.tx0, self.ty0 + lty, 0);
            row_y[lty + 1] = row_y[lty] + level_dims(1, th_full, steps_skipped).1;
        }
        let grid_w = col_x[self.sel_tx];
        let grid_h = row_y[self.sel_ty];

        let ts = hdr.tile_size as usize;
        let sub_x0 = ((self.cx as usize) - self.tx0 * ts) >> steps_skipped;
        let sub_y0 = ((self.cy as usize) - self.ty0 * ts) >> steps_skipped;
        let (sub_w, sub_h) = level_dims(self.cw as usize, self.ch as usize, steps_skipped);
        let sub_w = sub_w.min(grid_w.saturating_sub(sub_x0)).max(1);
        let sub_h = sub_h.min(grid_h.saturating_sub(sub_y0)).max(1);
        let sub_x1 = sub_x0 + sub_w;
        let sub_y1 = sub_y0 + sub_h;

        let out_ch = if self.query.grayscale {
            if hdr.channels == 4 { 4 } else { 1 }
        } else if self.query.sepia {
            if hdr.channels == 4 { 4 } else { 3 }
        } else {
            hdr.channels as usize
        };
        let detail_gains = build_detail_gains(lvl, &self.query);

        let mut scratch = WaveletScratch::default();
        let mut cur_w = sub_w;
        let mut cur_h = sub_h;
        let metadata = if lvl == self.max_lvl && self.query.load_metadata && self.index.meta_wire_len > 0 {
            self.index.read_metadata_from_source(source, self.query.verify_crc, &mut self.stats)?
        } else {
            None
        };

        let result = if bit_depth == 8 {
            let mut pixels = vec![0u8; sub_w * sub_h * out_ch];
            for lty in 0..self.sel_ty {
                for ltx in 0..self.sel_tx {
                    let tx_px0 = col_x[ltx];
                    let ty_px0 = row_y[lty];
                    let dst_tw = col_x[ltx + 1] - tx_px0;
                    let dst_th = row_y[lty + 1] - ty_px0;
                    let (y_t, mut co_t, mut cg_t, a_t) = self.reconstruct_tile_planes(
                        ltx, lty, dst_tw, dst_th, lvl, &detail_gains, &mut scratch,
                    );
                    apply_color_adjust(&mut co_t, &mut cg_t, dst_tw * dst_th, bd_shift, &self.query);
                    let tile_rgb = merge_planes_rgb(
                        &y_t, co_t.as_deref(), cg_t.as_deref(), a_t.as_deref(), dst_tw, dst_th, out_ch,
                    );
                    blit_tile(&tile_rgb, dst_tw, dst_th, tx_px0, ty_px0, sub_x0, sub_y0, sub_x1, sub_y1, sub_w, out_ch, &mut pixels);
                }
            }
            if let (Some(tw), Some(th)) = (self.query.target_width, self.query.target_height) {
                if tw as usize != cur_w || th as usize != cur_h {
                    pixels = resize_bicubic_u8(&pixels, cur_w, cur_h, tw as usize, th as usize, out_ch);
                    cur_w = tw as usize;
                    cur_h = th as usize;
                }
            }
            if self.query.flip_h || self.query.flip_v || (self.query.rotate_cw & 3) != 0 {
                let (np, nw, nh) = apply_geom_pub(
                    &pixels, cur_w, cur_h, out_ch, self.query.rotate_cw & 3, self.query.flip_h, self.query.flip_v,
                );
                pixels = np;
                cur_w = nw;
                cur_h = nh;
            }
            DecodedImage {
                pixels,
                pixels_u16: None,
                width: cur_w as u32,
                height: cur_h as u32,
                channels: out_ch as u8,
                bit_depth,
                decoded_level: lvl as u8,
                tiles_fetched,
                tiles_total: hdr.n_tiles(),
                stats: self.stats,
                icc_profile: self.icc_profile.clone(),
                metadata,
            }
        } else {
            let mut pixels16 = vec![0u16; sub_w * sub_h * out_ch];
            for lty in 0..self.sel_ty {
                for ltx in 0..self.sel_tx {
                    let tx_px0 = col_x[ltx];
                    let ty_px0 = row_y[lty];
                    let dst_tw = col_x[ltx + 1] - tx_px0;
                    let dst_th = row_y[lty + 1] - ty_px0;
                    let tile_rgb = if bit_depth == 16 {
                        let (y_t, mut co_t, mut cg_t, a_t) = self.reconstruct_tile_planes_i32(
                            ltx, lty, dst_tw, dst_th, lvl, &detail_gains, &mut scratch,
                        );
                        apply_color_adjust_i32(&mut co_t, &mut cg_t, dst_tw * dst_th, bd_shift, &self.query);
                        merge_planes_u16_i32(
                            &y_t, co_t.as_deref(), cg_t.as_deref(), a_t.as_deref(), dst_tw, dst_th, out_ch, max_sample,
                        )
                    } else {
                        let (y_t, mut co_t, mut cg_t, a_t) = self.reconstruct_tile_planes(
                            ltx, lty, dst_tw, dst_th, lvl, &detail_gains, &mut scratch,
                        );
                        apply_color_adjust(&mut co_t, &mut cg_t, dst_tw * dst_th, bd_shift, &self.query);
                        merge_planes_u16(
                            &y_t, co_t.as_deref(), cg_t.as_deref(), a_t.as_deref(), dst_tw, dst_th, out_ch, max_sample as i16,
                        )
                    };
                    blit_tile(&tile_rgb, dst_tw, dst_th, tx_px0, ty_px0, sub_x0, sub_y0, sub_x1, sub_y1, sub_w, out_ch, &mut pixels16);
                }
            }
            if let (Some(tw), Some(th)) = (self.query.target_width, self.query.target_height) {
                if tw as usize != cur_w || th as usize != cur_h {
                    pixels16 = resize_bilinear_u16(&pixels16, cur_w, cur_h, tw as usize, th as usize, out_ch, max_sample as f32);
                    cur_w = tw as usize;
                    cur_h = th as usize;
                }
            }
            if self.query.flip_h || self.query.flip_v || (self.query.rotate_cw & 3) != 0 {
                let (np, nw, nh) = apply_geom_pub(
                    &pixels16, cur_w, cur_h, out_ch, self.query.rotate_cw & 3, self.query.flip_h, self.query.flip_v,
                );
                pixels16 = np;
                cur_w = nw;
                cur_h = nh;
            }
            let preview_u8: Vec<u8> = pixels16.iter().map(|&v| (v >> bd_shift).min(255) as u8).collect();
            DecodedImage {
                pixels: preview_u8,
                pixels_u16: Some(pixels16),
                width: cur_w as u32,
                height: cur_h as u32,
                channels: out_ch as u8,
                bit_depth,
                decoded_level: lvl as u8,
                tiles_fetched,
                tiles_total: hdr.n_tiles(),
                stats: self.stats,
                icc_profile: self.icc_profile.clone(),
                metadata,
            }
        };

        self.next_lvl += 1;
        Ok(Some(result))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reconstruct_tile_planes(
        &self,
        ltx: usize,
        lty: usize,
        dst_tw: usize,
        dst_th: usize,
        lvl: usize,
        detail_gains: &[f32],
        scratch: &mut WaveletScratch,
    ) -> (Vec<i16>, Option<Vec<i16>>, Option<Vec<i16>>, Option<Vec<i16>>) {
        let hdr = &self.index.header;
        let n_planes = self.planes.len();
        let mut y_t = None;
        let mut co_t = None;
        let mut cg_t = None;
        let mut a_t = None;

        for (p_slot, &plane) in self.planes.iter().enumerate() {
            let c_idx = (lty * self.sel_tx + ltx) * n_planes + p_slot;
            let slot = &self.cached[c_idx];
            let mut tile_px = if lvl == self.max_lvl && self.query.blur <= 0.0 && self.query.sharpen <= 0.0 {
                if let Some(ref sp) = slot.spatial_override {
                    sp.clone()
                } else {
                    let wt = WaveletTile {
                        full_w: slot.pw_full,
                        full_h: slot.ph_full,
                        ll_w: slot.ll_w,
                        ll_h: slot.ll_h,
                        ll: slot.ll.clone(),
                        details: slot.details.clone(),
                    };
                    let g = if plane == 0 { detail_gains } else { &[] };
                    wt.reconstruct_mode_with(lvl, g, hdr.is_dwt_97(), scratch).0
                }
            } else {
                let wt = WaveletTile {
                    full_w: slot.pw_full,
                    full_h: slot.ph_full,
                    ll_w: slot.ll_w,
                    ll_h: slot.ll_h,
                    ll: slot.ll.clone(),
                    details: slot.details.clone(),
                };
                let g = if plane == 0 { detail_gains } else { &[] };
                wt.reconstruct_mode_with(lvl, g, hdr.is_dwt_97(), scratch).0
            };
            let (tw_dec, th_dec) = level_dims(slot.pw_full, slot.ph_full, self.max_lvl - lvl);
            if tw_dec != dst_tw || th_dec != dst_th {
                tile_px = upsample_2x2_bilinear(&tile_px, tw_dec, th_dec, dst_tw, dst_th);
            }
            match plane {
                0 => y_t = Some(tile_px),
                1 => co_t = Some(tile_px),
                2 => cg_t = Some(tile_px),
                _ => a_t = Some(tile_px),
            }
        }
        (y_t.unwrap(), co_t, cg_t, a_t)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reconstruct_tile_planes_i32(
        &self,
        ltx: usize,
        lty: usize,
        dst_tw: usize,
        dst_th: usize,
        lvl: usize,
        detail_gains: &[f32],
        scratch: &mut WaveletScratch,
    ) -> (Vec<i32>, Option<Vec<i32>>, Option<Vec<i32>>, Option<Vec<i32>>) {
        let hdr = &self.index.header;
        let n_planes = self.planes.len();
        let mut y_t = None;
        let mut co_t = None;
        let mut cg_t = None;
        let mut a_t = None;

        for (p_slot, &plane) in self.planes.iter().enumerate() {
            let c_idx = (lty * self.sel_tx + ltx) * n_planes + p_slot;
            let slot = &self.cached[c_idx];
            let wt = WaveletTile {
                full_w: slot.pw_full,
                full_h: slot.ph_full,
                ll_w: slot.ll_w,
                ll_h: slot.ll_h,
                ll: slot.ll.clone(),
                details: slot.details.clone(),
            };
            let g = if plane == 0 { detail_gains } else { &[] };
            let mut tile_px = wt.reconstruct_i32_mode_with(lvl, g, hdr.is_dwt_97(), scratch).0;
            let (tw_dec, th_dec) = level_dims(slot.pw_full, slot.ph_full, self.max_lvl - lvl);
            if tw_dec != dst_tw || th_dec != dst_th {
                tile_px = upsample_2x2_bilinear_i32(&tile_px, tw_dec, th_dec, dst_tw, dst_th);
            }
            match plane {
                0 => y_t = Some(tile_px),
                1 => co_t = Some(tile_px),
                2 => cg_t = Some(tile_px),
                _ => a_t = Some(tile_px),
            }
        }
        (y_t.unwrap(), co_t, cg_t, a_t)
    }
}

#[allow(clippy::too_many_arguments)]
fn blit_tile<T: Copy>(
    tile: &[T],
    dst_tw: usize,
    dst_th: usize,
    tx_px0: usize,
    ty_px0: usize,
    sub_x0: usize,
    sub_y0: usize,
    sub_x1: usize,
    sub_y1: usize,
    sub_w: usize,
    out_ch: usize,
    dst: &mut [T],
) {
    let ix0 = tx_px0.max(sub_x0);
    let iy0 = ty_px0.max(sub_y0);
    let ix1 = (tx_px0 + dst_tw).min(sub_x1);
    let iy1 = (ty_px0 + dst_th).min(sub_y1);
    if ix0 >= ix1 || iy0 >= iy1 {
        return;
    }
    let copy_cols = (ix1 - ix0) * out_ch;
    let sx_off = (ix0 - tx_px0) * out_ch;
    let dx_off = (ix0 - sub_x0) * out_ch;
    for y in iy0..iy1 {
        let src_row = (y - ty_px0) * dst_tw * out_ch + sx_off;
        let dst_row = (y - sub_y0) * sub_w * out_ch + dx_off;
        dst[dst_row..dst_row + copy_cols].copy_from_slice(&tile[src_row..src_row + copy_cols]);
    }
}
