//! Wavelet subband perceptual quantization, MED `LL` prediction, and 4×4 oriented micro-block
//! 2D locality scanning (`LH` row-major, `HL` col-major, `HH` row-major).

use crate::entropy::{pack_stream, unpack_stream};
use crate::wavelet::{DetailBands, WaveletTile};
use crate::Result;

#[inline]
fn base_quant_scale(quality: u8) -> f32 {
    let q = (quality.clamp(1, 99)) as f32;
    if q >= 50.0 {
        ((100.0 - q) / 50.0).powf(1.55) * 18.0 + 1.0
    } else {
        19.0 + ((50.0 - q) / 49.0).powf(1.6) * 56.0
    }
}

/// Perceptual quantization step (`1` when `quality == 100`, i.e., 100% bit-exact lossless).
/// Uses true 2D wavelet synthesis-norm scaling across pyramid levels so every level spends
/// equal marginal bits per dB of MSE reduction.
#[inline]
pub fn quant_step(quality: u8, plane: usize, detail_idx: usize, dwt_levels: usize, is_hh: bool) -> i32 {
    if quality >= 100 {
        return 1;
    }
    let base = base_quant_scale(quality);
    let finest_dist = dwt_levels.saturating_sub(1).saturating_sub(detail_idx);
    let level_mul = match finest_dist {
        0 => 2.20,
        1 => 1.08,
        2 => 0.54,
        _ => 0.28,
    };
    let diag_mul = if is_hh { 1.50 } else { 1.0 };
    let plane_mul = match plane {
        0 => 1.0,  // Y luma
        3 => 1.0,  // Alpha
        _ => 1.65, // Co / Cg chroma
    };
    (base * level_mul * diag_mul * plane_mul).round().max(1.0) as i32
}

/// Quantization step for `LL` (Level 0) residuals aligned with the 2D synthesis-norm curve.
#[inline]
pub fn ll_quant_step(quality: u8, plane: usize) -> i32 {
    if quality >= 90 || plane == 3 {
        return 1;
    }
    let base = base_quant_scale(quality);
    let plane_mul = if plane == 0 { 0.28 } else { 0.50 };
    (base * plane_mul).round().max(1.0) as i32
}

/// JPEG 2000-style Laplacian MMSE dead-zone quantizer: `q = floor(|c| / step)`.
#[inline(always)]
fn quant_deadzone(c: i32, step: i32) -> i32 {
    if step <= 1 {
        return c;
    }
    let q = c.abs() / step;
    if c < 0 { -q } else { q }
}

/// Laplacian MMSE reconstruction centroid: `sign(q) * (|q| + 0.375) * step`.
#[inline(always)]
fn dequant(q: i32, step: i32) -> i32 {
    if step <= 1 || q == 0 {
        return q;
    }
    let mag = q.abs() * step + ((step * 3) >> 3);
    if q < 0 { -mag } else { mag }
}

/// JPEG-LS Median Edge Detector (MED) predictor for 2D `LL` grid.
#[inline(always)]
fn med_predict(left: i32, top: i32, top_left: i32) -> i32 {
    let mn = left.min(top);
    let mx = left.max(top);
    if top_left >= mx {
        mn
    } else if top_left <= mn {
        mx
    } else {
        left + top - top_left
    }
}

/// Scans a 2D band into 4×4 micro-blocks (`transpose = true` scans column-major for `HL` vertical edges!).
fn scan_band_4x4(src: &[i32], w: usize, h: usize, step: i32, transpose: bool, dst: &mut Vec<i32>) {
    if !transpose {
        let mut by = 0;
        while by < h {
            let bh = (h - by).min(4);
            let mut bx = 0;
            while bx < w {
                let bw = (w - bx).min(4);
                for dy in 0..bh {
                    let row = (by + dy) * w + bx;
                    for dx in 0..bw {
                        dst.push(quant_deadzone(src[row + dx], step));
                    }
                }
                bx += 4;
            }
            by += 4;
        }
    } else {
        let mut bx = 0;
        while bx < w {
            let bw = (w - bx).min(4);
            let mut by = 0;
            while by < h {
                let bh = (h - by).min(4);
                for dx in 0..bw {
                    for dy in 0..bh {
                        dst.push(quant_deadzone(src[(by + dy) * w + bx + dx], step));
                    }
                }
                by += 4;
            }
            bx += 4;
        }
    }
}

fn unscan_band_4x4(src: &[i32], w: usize, h: usize, step: i32, transpose: bool) -> Vec<i32> {
    let mut dst = vec![0i32; w * h];
    let mut p = 0usize;
    if !transpose {
        let mut by = 0;
        while by < h {
            let bh = (h - by).min(4);
            let mut bx = 0;
            while bx < w {
                let bw = (w - bx).min(4);
                for dy in 0..bh {
                    let row = (by + dy) * w + bx;
                    for dx in 0..bw {
                        dst[row + dx] = dequant(src[p], step);
                        p += 1;
                    }
                }
                bx += 4;
            }
            by += 4;
        }
    } else {
        let mut bx = 0;
        while bx < w {
            let bw = (w - bx).min(4);
            let mut by = 0;
            while by < h {
                let bh = (h - by).min(4);
                for dx in 0..bw {
                    for dy in 0..bh {
                        dst[(by + dy) * w + bx + dx] = dequant(src[p], step);
                        p += 1;
                    }
                }
                by += 4;
            }
            bx += 4;
        }
    }
    dst
}

/// Encodes Level 0 (`LL`) subband using MED 2D DPCM + context-adaptive Hybrid-Golomb FSE.
pub fn encode_ll_band(ll: &[i32], w: usize, h: usize, quality: u8, plane: usize) -> Vec<u8> {
    let step = ll_quant_step(quality, plane);
    let mut residuals = vec![0i32; w * h];
    let mut rec = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pred = match (x, y) {
                (0, 0) => if plane == 0 { 128 } else { 0 },
                (x, 0) => rec[x - 1],
                (0, y) => rec[(y - 1) * w],
                (_, _) => med_predict(rec[idx - 1], rec[idx - w], rec[idx - w - 1]),
            };
            let diff = ll[idx] - pred;
            let q = if step <= 1 { diff } else { quant_deadzone(diff, step) };
            residuals[idx] = q;
            rec[idx] = pred + if step <= 1 { q } else { dequant(q, step) };
        }
    }
    let mut out = Vec::new();
    pack_stream(&residuals, &mut out);
    out
}

/// Decodes Level 0 (`LL`) subband back into `(w * h)` coefficients.
pub fn decode_ll_band(payload: &[u8], w: usize, h: usize, quality: u8, plane: usize) -> Result<Vec<i32>> {
    let step = ll_quant_step(quality, plane);
    let mut residuals = vec![0i32; w * h];
    unpack_stream(payload, &mut residuals)?;
    let mut ll = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pred = match (x, y) {
                (0, 0) => if plane == 0 { 128 } else { 0 },
                (x, 0) => ll[x - 1],
                (0, y) => ll[(y - 1) * w],
                (_, _) => med_predict(ll[idx - 1], ll[idx - w], ll[idx - w - 1]),
            };
            let q = residuals[idx];
            ll[idx] = pred + if step <= 1 { q } else { dequant(q, step) };
        }
    }
    Ok(ll)
}

/// Encodes one pyramid detail level (`LH + HL + HH`) using 4×4 oriented scanning + context FSE.
pub fn encode_detail_level(
    d: &DetailBands,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
) -> Vec<u8> {
    let step_lh = quant_step(quality, plane, detail_idx, dwt_levels, false);
    let step_hh = quant_step(quality, plane, detail_idx, dwt_levels, true);
    let lw = d.out_w.div_ceil(2);
    let hw = d.out_w / 2;
    let lh_rows = d.out_h.div_ceil(2);
    let hh_rows = d.out_h / 2;
    let mut qbuf = Vec::with_capacity(d.lh.len() + d.hl.len() + d.hh.len());
    scan_band_4x4(&d.lh, lw, hh_rows, step_lh, false, &mut qbuf);
    scan_band_4x4(&d.hl, hw, lh_rows, step_lh, true, &mut qbuf);
    scan_band_4x4(&d.hh, hw, hh_rows, step_hh, false, &mut qbuf);
    let mut out = Vec::new();
    pack_stream(&qbuf, &mut out);
    out
}

/// Decodes one pyramid detail level (`LH + HL + HH`) back into `DetailBands`.
pub fn decode_detail_level(
    payload: &[u8],
    out_w: usize,
    out_h: usize,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
) -> Result<DetailBands> {
    let lw = out_w.div_ceil(2);
    let hw = out_w / 2;
    let lh_rows = out_h.div_ceil(2);
    let hh_rows = out_h / 2;
    let n_lh = lw * hh_rows;
    let n_hl = hw * lh_rows;
    let n_hh = hw * hh_rows;

    let mut qbuf = vec![0i32; n_lh + n_hl + n_hh];
    let is_hybrid = unpack_stream(payload, &mut qbuf)?;

    let step_lh = quant_step(quality, plane, detail_idx, dwt_levels, false);
    let step_hh = quant_step(quality, plane, detail_idx, dwt_levels, true);
    let (lh, hl, hh) = if is_hybrid {
        (
            unscan_band_4x4(&qbuf[..n_lh], lw, hh_rows, step_lh, false),
            unscan_band_4x4(&qbuf[n_lh..n_lh + n_hl], hw, lh_rows, step_lh, true),
            unscan_band_4x4(&qbuf[n_lh + n_hl..], hw, hh_rows, step_hh, false),
        )
    } else {
        (
            qbuf[..n_lh].iter().map(|&q| dequant(q, step_lh)).collect(),
            qbuf[n_lh..n_lh + n_hl].iter().map(|&q| dequant(q, step_lh)).collect(),
            qbuf[n_lh + n_hl..].iter().map(|&q| dequant(q, step_hh)).collect(),
        )
    };
    Ok(DetailBands { out_w, out_h, lh, hl, hh })
}

/// Encodes all pyramid levels (`0..=dwt_levels`) of a single `WaveletTile`.
pub fn encode_wavelet_tile_levels(wt: &WaveletTile, quality: u8, plane: usize) -> Vec<Vec<u8>> {
    let dwt_levels = wt.details.len();
    let mut out = Vec::with_capacity(dwt_levels + 1);
    out.push(encode_ll_band(&wt.ll, wt.ll_w, wt.ll_h, quality, plane));
    for (k, d) in wt.details.iter().enumerate() {
        out.push(encode_detail_level(d, quality, plane, k, dwt_levels));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subband_lossless_roundtrip_is_bit_exact() {
        let w = 64;
        let h = 64;
        let mut src = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                src.push(((x * 3 + y * 5) % 256) as i16);
            }
        }
        let wt = WaveletTile::forward(&src, w, h, 3);
        let chunks = encode_wavelet_tile_levels(&wt, 100, 0);
        let ll = decode_ll_band(&chunks[0], wt.ll_w, wt.ll_h, 100, 0).unwrap();
        let mut details = Vec::new();
        for (k, d) in wt.details.iter().enumerate() {
            details.push(decode_detail_level(&chunks[k + 1], d.out_w, d.out_h, 100, 0, k, 3).unwrap());
        }
        let wt2 = WaveletTile { full_w: w, full_h: h, ll_w: wt.ll_w, ll_h: wt.ll_h, ll, details };
        let (rec, rw, rh) = wt2.reconstruct(3, &[]);
        assert_eq!((rw, rh), (w, h));
        assert_eq!(rec, src);
    }
}
