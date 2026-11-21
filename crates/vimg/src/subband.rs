//! Wavelet subband perceptual quantization, MED `LL` prediction, and single-pass 2D
//! context-adaptive subband encoding/decoding.

use crate::entropy::{
    decode_2d_bands, encode_2d_ac, encode_blk4_ac_with_aq, try_encode_spatial_palette,
};
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
/// Scales cleanly with `bit_depth` (`8`, `10`, or `12`).
#[inline]
pub fn quant_step_bd(
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
    is_hh: bool,
    bit_depth: u8,
) -> i32 {
    if quality >= 100 {
        return 1;
    }
    let scale = 1i32 << bit_depth.saturating_sub(8);
    quant_step(quality, plane, detail_idx, dwt_levels, is_hh) * scale
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
        _ => 1.65, // YCoCg-R Co / Cg chroma
    };
    (base * level_mul * diag_mul * plane_mul).round().max(1.0) as i32
}

/// Quantization step for `LL` (Level 0) residuals aligned with `bit_depth` (`8`, `10`, `12`).
#[inline]
pub fn ll_quant_step_bd(quality: u8, plane: usize, bit_depth: u8) -> i32 {
    if quality >= 90 || plane == 3 {
        return 1;
    }
    let scale = 1i32 << bit_depth.saturating_sub(8);
    ll_quant_step(quality, plane) * scale
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

/// Encodes Level 0 (`LL`) subband using MED 2D DPCM + 2D adaptive binary range coding.
pub fn encode_ll_band(ll: &[i32], w: usize, h: usize, quality: u8, plane: usize) -> Vec<u8> {
    encode_ll_band_bd(ll, w, h, quality, plane, 8)
}

pub fn encode_ll_band_bd(
    ll: &[i32],
    w: usize,
    h: usize,
    quality: u8,
    plane: usize,
    bit_depth: u8,
) -> Vec<u8> {
    let step = ll_quant_step_bd(quality, plane, bit_depth);
    let mid = 1i32 << (bit_depth.clamp(8, 16) - 1);
    let mut residuals = vec![0i32; w * h];
    let mut rec = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pred = match (x, y) {
                (0, 0) => if plane == 0 { mid } else { 0 },
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
    encode_2d_ac(&[(&residuals, w, h)], &mut out);
    out
}

/// Decodes Level 0 (`LL`) subband back into `(w * h)` coefficients.
pub fn decode_ll_band(payload: &[u8], w: usize, h: usize, quality: u8, plane: usize) -> Result<Vec<i32>> {
    decode_ll_band_bd(payload, w, h, quality, plane, 8)
}

pub fn decode_ll_band_bd(
    payload: &[u8],
    w: usize,
    h: usize,
    quality: u8,
    plane: usize,
    bit_depth: u8,
) -> Result<Vec<i32>> {
    let step = ll_quant_step_bd(quality, plane, bit_depth);
    let mid = 1i32 << (bit_depth.clamp(8, 16) - 1);
    let mut residuals = vec![0i32; w * h];
    decode_2d_bands(payload, &mut [(&mut residuals, w, h)], &[])?;
    let mut ll = vec![0i32; w * h];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let pred = match (x, y) {
                (0, 0) => if plane == 0 { mid } else { 0 },
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

/// Encodes one pyramid detail level (`LH + HL + HH`) into a single compressed chunk.
pub fn encode_detail_level(
    d: &DetailBands,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
) -> Vec<u8> {
    encode_detail_level_bd(d, quality, plane, detail_idx, dwt_levels, 8)
}

/// Computes a tile-adaptive psychovisual quantization multiplier in fixed-point `1/16` (`16 = 1.0×`).
/// Smooth sky/skin tiles get finer quantization (`0.75×–0.875×`) to eliminate banding & ringing;
/// chaotic/noisy high-frequency tiles get contrast-masked coarser quantization (`1.19×–1.50×`).
#[inline]
fn compute_aq_mul_q4(d: &DetailBands, plane: usize, bit_depth: u8) -> u8 {
    if plane == 3 {
        return 16; // Always keep alpha crisp
    }
    let total = d.lh.len() + d.hl.len() + d.hh.len();
    if total < 64 {
        return 16;
    }
    let mut sum: u64 = 0;
    for &v in &d.lh {
        sum += v.unsigned_abs() as u64;
    }
    for &v in &d.hl {
        sum += v.unsigned_abs() as u64;
    }
    for &v in &d.hh {
        sum += v.unsigned_abs() as u64;
    }
    let mad_8bit = (sum / (total as u64)) >> bit_depth.saturating_sub(8);
    if mad_8bit <= 2 {
        12 // 0.75× — razor-sharp smooth gradients, zero banding
    } else if mad_8bit <= 5 {
        14 // 0.875×
    } else if mad_8bit <= 12 {
        16 // 1.00× — standard
    } else if mad_8bit <= 22 {
        19 // 1.1875× — contrast masking on moderate texture
    } else if mad_8bit <= 36 {
        22 // 1.375× — contrast masking on busy foliage/night texture
    } else {
        24 // 1.50× — strong contrast masking on chaotic high-frequency grain
    }
}

pub fn encode_detail_level_bd(
    d: &DetailBands,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
    bit_depth: u8,
) -> Vec<u8> {
    encode_detail_level_full(d, quality, plane, detail_idx, dwt_levels, bit_depth, false)
}

pub fn encode_detail_level_full(
    d: &DetailBands,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
    bit_depth: u8,
    adaptive_quant: bool,
) -> Vec<u8> {
    let lw = d.out_w.div_ceil(2);
    let hw = d.out_w / 2;
    let lh_rows = d.out_h.div_ceil(2);
    let hh_rows = d.out_h / 2;
    let mut out = Vec::new();

    if quality >= 100 {
        encode_2d_ac(&[(&d.lh, lw, hh_rows), (&d.hl, hw, lh_rows), (&d.hh, hw, hh_rows)], &mut out);
        return out;
    }

    let base_lh = quant_step_bd(quality, plane, detail_idx, dwt_levels, false, bit_depth);
    let base_hh = quant_step_bd(quality, plane, detail_idx, dwt_levels, true, bit_depth);
    let q_mul_q4 = if adaptive_quant { compute_aq_mul_q4(d, plane, bit_depth) } else { 16 };
    let step_lh = if q_mul_q4 != 16 && base_lh > 1 {
        ((base_lh * q_mul_q4 as i32 + 8) >> 4).max(1)
    } else {
        base_lh
    };
    let step_hh = if q_mul_q4 != 16 && base_hh > 1 {
        ((base_hh * q_mul_q4 as i32 + 8) >> 4).max(1)
    } else {
        base_hh
    };

    let qlh: Vec<i32> = d.lh.iter().map(|&c| quant_deadzone(c, step_lh)).collect();
    let qhl: Vec<i32> = d.hl.iter().map(|&c| quant_deadzone(c, step_lh)).collect();
    let qhh: Vec<i32> = d.hh.iter().map(|&c| quant_deadzone(c, step_hh)).collect();

    encode_blk4_ac_with_aq(&[(&qlh, lw, hh_rows), (&qhl, hw, lh_rows), (&qhh, hw, hh_rows)], q_mul_q4, &mut out);
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
    decode_detail_level_bd(payload, out_w, out_h, quality, plane, detail_idx, dwt_levels, 8)
}

#[allow(clippy::too_many_arguments)]
pub fn decode_detail_level_bd(
    payload: &[u8],
    out_w: usize,
    out_h: usize,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
    bit_depth: u8,
) -> Result<DetailBands> {
    // Fast zero-allocation path for all-zero detail chunks (`CODEC_ALL_ZERO`)!
    if payload.len() == 1 && payload[0] == 0 {
        return Ok(DetailBands::all_zero(out_w, out_h));
    }

    let lw = out_w.div_ceil(2);
    let hw = out_w / 2;
    let lh_rows = out_h.div_ceil(2);
    let hh_rows = out_h / 2;

    let mut lh = vec![0i32; lw * hh_rows];
    let mut hl = vec![0i32; hw * lh_rows];
    let mut hh = vec![0i32; hw * hh_rows];

    let step_lh = quant_step_bd(quality, plane, detail_idx, dwt_levels, false, bit_depth);
    let step_hh = quant_step_bd(quality, plane, detail_idx, dwt_levels, true, bit_depth);
    let steps = if quality < 100 { [step_lh, step_lh, step_hh] } else { [1, 1, 1] };

    decode_2d_bands(
        payload,
        &mut [(&mut lh, lw, hh_rows), (&mut hl, hw, lh_rows), (&mut hh, hw, hh_rows)],
        &steps,
    )?;

    Ok(DetailBands { out_w, out_h, lh, hl, hh })
}

/// Encodes all pyramid levels (`0..=dwt_levels`) of a single `WaveletTile`.
pub fn encode_wavelet_tile_levels(wt: &WaveletTile, quality: u8, plane: usize) -> Vec<Vec<u8>> {
    encode_wavelet_tile_levels_bd(wt, quality, plane, 8)
}

pub fn encode_wavelet_tile_levels_bd(
    wt: &WaveletTile,
    quality: u8,
    plane: usize,
    bit_depth: u8,
) -> Vec<Vec<u8>> {
    encode_wavelet_tile_levels_full(wt, None, quality, plane, bit_depth, false)
}

pub fn encode_wavelet_tile_levels_full(
    wt: &WaveletTile,
    spatial_plane: Option<&[i16]>,
    quality: u8,
    plane: usize,
    bit_depth: u8,
    adaptive_quant: bool,
) -> Vec<Vec<u8>> {
    let dwt_levels = wt.details.len();
    let mut out = Vec::with_capacity(dwt_levels + 1);
    out.push(encode_ll_band_bd(&wt.ll, wt.ll_w, wt.ll_h, quality, plane, bit_depth));
    for (k, d) in wt.details.iter().enumerate() {
        out.push(encode_detail_level_full(d, quality, plane, k, dwt_levels, bit_depth, adaptive_quant));
    }
    // In lossless mode, if the raw spatial tile has <=32 unique values (UI/screenshots/solid alpha),
    // store the finest chunk directly via `CODEC_SPATIAL_PALETTE` whenever it beats wavelet details!
    // Keeps L0..L(n-1) pyramid intact for thumbnails/resize while enabling zero-IDWT full decode!
    if quality >= 100 && dwt_levels >= 1 {
        if let Some(sp) = spatial_plane {
            if let Some(sp_blob) = try_encode_spatial_palette(sp, wt.full_w, wt.full_h) {
                let detail_total: usize = out[1..=dwt_levels].iter().map(Vec::len).sum();
                if sp_blob.len() + (dwt_levels - 1) < detail_total {
                    // Replace smeared multi-level wavelet details with 1-byte CODEC_ALL_ZERO on
                    // intermediate levels (L0 LL stays intact for previews) + exact zero-IDWT
                    // CODEC_SPATIAL_PALETTE at the finest level!
                    for slot in &mut out[1..dwt_levels] {
                        *slot = vec![0u8]; // CODEC_ALL_ZERO
                    }
                    out[dwt_levels] = sp_blob;
                }
            }
        }
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
