//! Wavelet subband perceptual quantization, MED `LL` prediction, 4×4 micro-block 2D locality
//! scanning, compact 32-symbol Hybrid-Golomb + bit-packed extra-bits tokenization, and pure-Rust
//! `tANS / FSE` tile-plane payload coding.

use crate::fse::{fse_decode, fse_encode_min};
use crate::wavelet::{DetailBands, WaveletTile};
use crate::{Error, Result};

const CODEC_ALL_ZERO: u8 = 0;
const CODEC_RAW_ZRL: u8 = 1;
const CODEC_FSE_ZRL: u8 = 2;
const CODEC_HYBRID_FSE: u8 = 3;
const CODEC_HYBRID_RAW: u8 = 4;

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

struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self { buf: Vec::new(), acc: 0, nbits: 0 }
    }

    #[inline(always)]
    fn write(&mut self, val: u32, bits: u32) {
        self.acc |= (val as u64) << self.nbits;
        self.nbits += bits;
        while self.nbits >= 8 {
            self.buf.push((self.acc & 0xff) as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.buf.push((self.acc & 0xff) as u8);
        }
        self.buf
    }
}

struct BitReader<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, acc: 0, nbits: 0 }
    }

    #[inline(always)]
    fn read(&mut self, bits: u32) -> Result<u32> {
        while self.nbits < bits {
            let b = *self.buf.get(self.pos).ok_or(Error::Truncated)?;
            self.pos += 1;
            self.acc |= (b as u64) << self.nbits;
            self.nbits += 8;
        }
        let mask = (1u64 << bits) - 1;
        let v = (self.acc & mask) as u32;
        self.acc >>= bits;
        self.nbits -= bits;
        Ok(v)
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
    let q = (quality.clamp(1, 99)) as f32;
    let base = if q >= 50.0 {
        ((100.0 - q) / 50.0).powf(1.55) * 18.0 + 1.0
    } else {
        19.0 + ((50.0 - q) / 49.0).powf(1.6) * 56.0
    };
    // 2D biorthogonal synthesis L2 norm halves each finer level -> step doubles each finer level!
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

/// Quantization step for `LL` (Level 0) residuals (`1` for lossless & high quality >= 88).
#[inline]
pub fn ll_quant_step(quality: u8, plane: usize) -> i32 {
    if quality >= 88 || plane == 3 {
        return 1;
    }
    let factor = ((88 - quality) as f32 / 88.0) * if plane == 0 { 4.2 } else { 8.5 };
    (1.0 + factor).round().max(1.0) as i32
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

/// Compact 32-symbol Hybrid-Golomb tokenizer + bit-packed extra bits + trailing-zero elision:
/// - `0..=3`: zero run `1..=4` (0 extra bits)
/// - `4..=14`: zero run `3 + (1<<k) .. 2 + (2<<k)` (`k = t - 3` extra bits)
/// - `15`: long zero run `4099..=69634` (`16` extra bits)
/// - `16..=19`: `+1, -1, +2, -2` (0 extra bits)
/// - `20..=30`: `±(1 + (1<<k) .. (2<<k))` (`k = t - 19` mag bits + `1` sign bit)
/// - `31`: large value `±(4097..=69632)` (`16` mag bits + `1` sign bit)
fn tokenize_hybrid(vals: &[i32], tokens: &mut Vec<u8>) -> Option<Vec<u8>> {
    tokens.clear();
    let mut end = vals.len();
    while end > 0 && vals[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut bw = BitWriter::new();
    let mut i = 0usize;
    while i < end {
        if vals[i] == 0 {
            let start = i;
            while i < end && vals[i] == 0 {
                i += 1;
            }
            let mut run = i - start;
            while run > 0 {
                if run <= 4 {
                    tokens.push((run - 1) as u8);
                    break;
                } else if run <= 4098 {
                    let rem = (run - 3) as u32;
                    let k = 31 - rem.leading_zeros();
                    let base = 3 + (1usize << k);
                    tokens.push((3 + k) as u8);
                    bw.write((run - base) as u32, k);
                    break;
                } else {
                    let chunk = run.min(69634);
                    tokens.push(15);
                    bw.write((chunk - 4099) as u32, 16);
                    run -= chunk;
                }
            }
        } else {
            let v = vals[i];
            let a = v.unsigned_abs();
            let sign = u32::from(v < 0);
            if a == 1 {
                tokens.push(16 + sign as u8);
            } else if a == 2 {
                tokens.push(18 + sign as u8);
            } else if a <= 4096 {
                let rem = a - 1;
                let k = 31 - rem.leading_zeros();
                let base = 1 + (1u32 << k);
                tokens.push((19 + k) as u8);
                bw.write(((a - base) << 1) | sign, k + 1);
            } else {
                let clamped = a.min(69632);
                tokens.push(31);
                bw.write(((clamped - 4097) << 1) | sign, 17);
            }
            i += 1;
        }
    }
    Some(bw.finish())
}

fn detokenize_hybrid(tokens: &[u8], extra: &[u8], out: &mut [i32]) -> Result<()> {
    let mut br = BitReader::new(extra);
    let mut o = 0usize;
    let n = out.len();
    for &t in tokens {
        if t < 4 {
            let run = (t as usize) + 1;
            if o + run > n {
                return Err(Error::InvalidFormat("hybrid zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
        } else if t < 15 {
            let k = (t - 3) as u32;
            let run = 3 + (1usize << k) + br.read(k)? as usize;
            if o + run > n {
                return Err(Error::InvalidFormat("hybrid zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
        } else if t == 15 {
            let run = 4099 + br.read(16)? as usize;
            if o + run > n {
                return Err(Error::InvalidFormat("hybrid long zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
        } else if t < 20 {
            if o >= n {
                return Err(Error::InvalidFormat("hybrid coeff overflow"));
            }
            out[o] = match t {
                16 => 1,
                17 => -1,
                18 => 2,
                _ => -2,
            };
            o += 1;
        } else if t < 31 {
            if o >= n {
                return Err(Error::InvalidFormat("hybrid coeff overflow"));
            }
            let k = (t - 19) as u32;
            let raw = br.read(k + 1)?;
            let sign = raw & 1;
            let mag = (1 + (1u32 << k) + (raw >> 1)) as i32;
            out[o] = if sign != 0 { -mag } else { mag };
            o += 1;
        } else if t == 31 {
            if o >= n {
                return Err(Error::InvalidFormat("hybrid large coeff overflow"));
            }
            let raw = br.read(17)?;
            let sign = raw & 1;
            let mag = (4097 + (raw >> 1)) as i32;
            out[o] = if sign != 0 { -mag } else { mag };
            o += 1;
        } else {
            return Err(Error::InvalidFormat("invalid hybrid token"));
        }
    }
    if o < n {
        out[o..n].fill(0);
    }
    Ok(())
}

fn detokenize_legacy_zrl(tokens: &[u8], extra: &[u8], out: &mut [i32]) -> Result<()> {
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
            out[o] = zz_decode(get_uleb(extra, &mut ep)?);
            o += 1;
        }
    }
    if o != n || ep != extra.len() {
        return Err(Error::InvalidFormat("zrl token/extra length mismatch"));
    }
    Ok(())
}

fn pack_stream(vals: &[i32], out: &mut Vec<u8>) {
    out.clear();
    let mut tokens = Vec::with_capacity(vals.len());
    let Some(extra) = tokenize_hybrid(vals, &mut tokens) else {
        out.push(CODEC_ALL_ZERO);
        return;
    };
    let mut fse_buf = Vec::new();
    let use_fse = fse_encode_min(&tokens, &mut fse_buf, tokens.len().saturating_sub(2));
    if use_fse {
        out.push(CODEC_HYBRID_FSE);
        put_uleb(out, tokens.len() as u32);
        put_uleb(out, fse_buf.len() as u32);
        out.extend_from_slice(&fse_buf);
    } else {
        out.push(CODEC_HYBRID_RAW);
        put_uleb(out, tokens.len() as u32);
        out.extend_from_slice(&tokens);
    }
    out.extend_from_slice(&extra);
}

fn unpack_stream(payload: &[u8], out: &mut [i32]) -> Result<bool> {
    if payload.is_empty() {
        return Err(Error::Truncated);
    }
    match payload[0] {
        CODEC_ALL_ZERO => {
            if payload.len() != 1 {
                return Err(Error::InvalidFormat("trailing bytes in all-zero band"));
            }
            out.fill(0);
            Ok(true)
        }
        CODEC_HYBRID_RAW => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let tok_end = pos.checked_add(tok_len).ok_or(Error::InvalidFormat("tok len overflow"))?;
            let tokens = payload.get(pos..tok_end).ok_or(Error::Truncated)?;
            detokenize_hybrid(tokens, &payload[tok_end..], out)?;
            Ok(true)
        }
        CODEC_HYBRID_FSE => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let fse_len = get_uleb(payload, &mut pos)? as usize;
            let fse_end = pos.checked_add(fse_len).ok_or(Error::InvalidFormat("fse len overflow"))?;
            let fse_blob = payload.get(pos..fse_end).ok_or(Error::Truncated)?;
            let mut tokens = vec![0u8; tok_len];
            fse_decode(fse_blob, &mut tokens)?;
            detokenize_hybrid(&tokens, &payload[fse_end..], out)?;
            Ok(true)
        }
        CODEC_RAW_ZRL => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let tok_end = pos.checked_add(tok_len).ok_or(Error::InvalidFormat("tok len overflow"))?;
            let tokens = payload.get(pos..tok_end).ok_or(Error::Truncated)?;
            detokenize_legacy_zrl(tokens, &payload[tok_end..], out)?;
            Ok(false)
        }
        CODEC_FSE_ZRL => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let fse_len = get_uleb(payload, &mut pos)? as usize;
            let fse_end = pos.checked_add(fse_len).ok_or(Error::InvalidFormat("fse len overflow"))?;
            let fse_blob = payload.get(pos..fse_end).ok_or(Error::Truncated)?;
            let mut tokens = vec![0u8; tok_len];
            fse_decode(fse_blob, &mut tokens)?;
            detokenize_legacy_zrl(&tokens, &payload[fse_end..], out)?;
            Ok(false)
        }
        _ => Err(Error::InvalidFormat("unknown subband codec tag")),
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

/// Encodes Level 0 (`LL`) subband using MED 2D DPCM + compact 32-symbol Hybrid-Golomb FSE.
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

/// Encodes one pyramid detail level (`LH + HL + HH`) using 4×4 oriented scanning + Hybrid-Golomb FSE.
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
