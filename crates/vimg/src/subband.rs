//! Wavelet subband perceptual quantization, MED `LL` prediction, Zero-Run-Length (`ZRL`)
//! tokenization, and pure-Rust `tANS / FSE` tile-plane payload coding.

use crate::fse::{fse_decode, fse_encode_min};
use crate::wavelet::{DetailBands, WaveletTile};
use crate::{Error, Result};

const CODEC_ALL_ZERO: u8 = 0;
const CODEC_RAW_ZRL: u8 = 1;
const CODEC_FSE_ZRL: u8 = 2;

#[inline(always)]
fn zz_encode(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

#[inline(always)]
fn zz_decode(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

#[inline(always)]
fn put_uleb(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            return;
        }
    }
}

#[inline(always)]
fn get_uleb(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let mut v = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        let b = *buf.get(*pos).ok_or(Error::Truncated)?;
        *pos += 1;
        v |= u32::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            return Ok(v);
        }
        shift += 7;
    }
    Err(Error::InvalidFormat("uleb32 overflow"))
}

/// Perceptual quantization step (`1` when `quality == 100`, i.e., 100% bit-exact lossless).
/// `detail_idx`: `0` = coarsest detail level (`L1`), `dwt_levels - 1` = finest detail level (`L_max`).
#[inline]
pub fn quant_step(quality: u8, plane: usize, detail_idx: usize, dwt_levels: usize, is_hh: bool) -> i32 {
    if quality >= 100 {
        return 1;
    }
    // Map quality 1..=99 -> base scale roughly 0.8 (q=95) .. 36.0 (q=10)
    let q = (quality.clamp(1, 99)) as f32;
    let base = if q >= 50.0 {
        ((100.0 - q) / 50.0).powf(1.45) * 14.0 + 1.0
    } else {
        15.0 + ((50.0 - q) / 49.0).powf(1.6) * 48.0
    };
    // Finer detail levels are visually less sensitive -> coarser step.
    let finest_dist = dwt_levels.saturating_sub(1).saturating_sub(detail_idx);
    let level_mul = match finest_dist {
        0 => 1.85,
        1 => 1.15,
        2 => 0.72,
        _ => 0.50,
    };
    let diag_mul = if is_hh { 1.45 } else { 1.0 };
    let plane_mul = match plane {
        0 => 1.0,  // Y luma
        3 => 1.0,  // Alpha
        _ => 1.55, // Co / Cg chroma
    };
    (base * level_mul * diag_mul * plane_mul).round().max(1.0) as i32
}

/// Quantization step for `LL` (Level 0) residuals (`1` for lossless and high quality >= 65).
#[inline]
pub fn ll_quant_step(quality: u8, plane: usize) -> i32 {
    if quality >= 80 || plane == 3 {
        return 1;
    }
    let factor = ((80 - quality) as f32 / 80.0) * if plane == 0 { 4.0 } else { 8.0 };
    (1.0 + factor).round().max(1.0) as i32
}

#[inline(always)]
fn quant_deadzone(c: i32, step: i32) -> i32 {
    if step <= 1 {
        return c;
    }
    let a = c.abs();
    // Slight dead-zone suppresses high-frequency sensor noise to exact zero.
    let dz = (step * 3) >> 2;
    if a < dz {
        0
    } else {
        let q = (a + (step >> 2)) / step;
        if c < 0 { -q } else { q }
    }
}

#[inline(always)]
fn dequant(q: i32, step: i32) -> i32 {
    if step <= 1 || q == 0 {
        return q;
    }
    let mag = q.abs() * step + (step >> 2);
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

/// Tokenizes signed integer sequence into `(zrl_tokens, extra_uleb_bytes)`.
/// - `0..=63`: zero run of length `1..=64`
/// - `64..=253`: non-zero zigzag magnitude `1..=190`
/// - `254`: long zero run (`>= 65`, length in `extra`)
/// - `255`: large zigzag magnitude (`>= 191`, value in `extra`)
fn tokenize_zrl(vals: &[i32], tokens: &mut Vec<u8>, extra: &mut Vec<u8>) -> bool {
    tokens.clear();
    extra.clear();
    let mut i = 0usize;
    let n = vals.len();
    let mut any_nz = false;
    while i < n {
        if vals[i] == 0 {
            let start = i;
            while i < n && vals[i] == 0 {
                i += 1;
            }
            let run = i - start;
            if run <= 64 {
                tokens.push((run - 1) as u8);
            } else {
                tokens.push(254);
                put_uleb(extra, run as u32);
            }
        } else {
            any_nz = true;
            let z = zz_encode(vals[i]);
            if z <= 190 {
                tokens.push((63 + z) as u8);
            } else {
                tokens.push(255);
                put_uleb(extra, z);
            }
            i += 1;
        }
    }
    any_nz
}

fn detokenize_zrl(tokens: &[u8], extra: &[u8], out: &mut [i32]) -> Result<()> {
    let mut o = 0usize;
    let mut ep = 0usize;
    let n = out.len();
    for &t in tokens {
        if t < 64 {
            let run = (t as usize) + 1;
            if o + run > n {
                return Err(Error::InvalidFormat("zrl zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
        } else if t < 254 {
            if o >= n {
                return Err(Error::InvalidFormat("zrl coeff overflow"));
            }
            out[o] = zz_decode((t - 63) as u32);
            o += 1;
        } else if t == 254 {
            let run = get_uleb(extra, &mut ep)? as usize;
            if run == 0 || o + run > n {
                return Err(Error::InvalidFormat("zrl long zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
        } else {
            if o >= n {
                return Err(Error::InvalidFormat("zrl large coeff overflow"));
            }
            let z = get_uleb(extra, &mut ep)?;
            out[o] = zz_decode(z);
            o += 1;
        }
    }
    if o != n || ep != extra.len() {
        return Err(Error::InvalidFormat("zrl token/extra length mismatch"));
    }
    Ok(())
}

fn pack_zrl_stream(vals: &[i32], out: &mut Vec<u8>) {
    out.clear();
    let mut tokens = Vec::with_capacity(vals.len());
    let mut extra = Vec::new();
    let any_nz = tokenize_zrl(vals, &mut tokens, &mut extra);
    if !any_nz {
        out.push(CODEC_ALL_ZERO);
        return;
    }
    let mut fse_buf = Vec::new();
    let use_fse = fse_encode_min(&tokens, &mut fse_buf, tokens.len().saturating_sub(2));
    if use_fse {
        out.push(CODEC_FSE_ZRL);
        put_uleb(out, tokens.len() as u32);
        put_uleb(out, fse_buf.len() as u32);
        out.extend_from_slice(&fse_buf);
    } else {
        out.push(CODEC_RAW_ZRL);
        put_uleb(out, tokens.len() as u32);
        out.extend_from_slice(&tokens);
    }
    out.extend_from_slice(&extra);
}

fn unpack_zrl_stream(payload: &[u8], out: &mut [i32]) -> Result<()> {
    if payload.is_empty() {
        return Err(Error::Truncated);
    }
    match payload[0] {
        CODEC_ALL_ZERO => {
            if payload.len() != 1 {
                return Err(Error::InvalidFormat("trailing bytes in all-zero band"));
            }
            out.fill(0);
            Ok(())
        }
        CODEC_RAW_ZRL => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let tok_end = pos.checked_add(tok_len).ok_or(Error::InvalidFormat("tok len overflow"))?;
            let tokens = payload.get(pos..tok_end).ok_or(Error::Truncated)?;
            let extra = &payload[tok_end..];
            detokenize_zrl(tokens, extra, out)
        }
        CODEC_FSE_ZRL => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let fse_len = get_uleb(payload, &mut pos)? as usize;
            let fse_end = pos.checked_add(fse_len).ok_or(Error::InvalidFormat("fse len overflow"))?;
            let fse_blob = payload.get(pos..fse_end).ok_or(Error::Truncated)?;
            let extra = &payload[fse_end..];
            let mut tokens = vec![0u8; tok_len];
            fse_decode(fse_blob, &mut tokens)?;
            detokenize_zrl(&tokens, extra, out)
        }
        _ => Err(Error::InvalidFormat("unknown subband codec tag")),
    }
}

/// Encodes Level 0 (`LL`) subband using MED 2D DPCM + ZRL + FSE.
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
    pack_zrl_stream(&residuals, &mut out);
    out
}

/// Decodes Level 0 (`LL`) subband back into `(w * h)` coefficients.
pub fn decode_ll_band(payload: &[u8], w: usize, h: usize, quality: u8, plane: usize) -> Result<Vec<i32>> {
    let step = ll_quant_step(quality, plane);
    let mut residuals = vec![0i32; w * h];
    unpack_zrl_stream(payload, &mut residuals)?;
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

/// Encodes one pyramid detail level (`LH + HL + HH`) into a single compressed chunk.
pub fn encode_detail_level(
    d: &DetailBands,
    quality: u8,
    plane: usize,
    detail_idx: usize,
    dwt_levels: usize,
) -> Vec<u8> {
    let step_lh = quant_step(quality, plane, detail_idx, dwt_levels, false);
    let step_hh = quant_step(quality, plane, detail_idx, dwt_levels, true);
    let total = d.lh.len() + d.hl.len() + d.hh.len();
    let mut qbuf = Vec::with_capacity(total);
    for &c in &d.lh {
        qbuf.push(quant_deadzone(c, step_lh));
    }
    for &c in &d.hl {
        qbuf.push(quant_deadzone(c, step_lh));
    }
    for &c in &d.hh {
        qbuf.push(quant_deadzone(c, step_hh));
    }
    let mut out = Vec::new();
    pack_zrl_stream(&qbuf, &mut out);
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
    unpack_zrl_stream(payload, &mut qbuf)?;

    let step_lh = quant_step(quality, plane, detail_idx, dwt_levels, false);
    let step_hh = quant_step(quality, plane, detail_idx, dwt_levels, true);
    let mut lh = Vec::with_capacity(n_lh);
    for &q in &qbuf[..n_lh] {
        lh.push(dequant(q, step_lh));
    }
    let mut hl = Vec::with_capacity(n_hl);
    for &q in &qbuf[n_lh..n_lh + n_hl] {
        hl.push(dequant(q, step_lh));
    }
    let mut hh = Vec::with_capacity(n_hh);
    for &q in &qbuf[n_lh + n_hl..] {
        hh.push(dequant(q, step_hh));
    }
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
