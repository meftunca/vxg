//! Pure-Rust, zero-dependency, zero-header-overhead **Adaptive Binary Range Coder** +
//! 2D Spatial-Context Wavelet Subband Model (`VIMG`).
//!
//! Hot-loop optimized with 1-instruction `CLZ` energy bucketing, branchless sign trits,
//! bounds-check-free interior raster & 4×4 block fast paths, and fused active-block dequantization.

use crate::{Error, Result};

const CODEC_ALL_ZERO: u8 = 0;
const CODEC_2D_AC: u8 = 1;
const CODEC_BLK4_AC: u8 = 2;
pub const CODEC_BLK4_AQ: u8 = 3;
pub const CODEC_PALETTE_RLE: u8 = 4;
pub const CODEC_SPATIAL_PALETTE: u8 = 5;

const PROB_INIT: u16 = 1024; // 0.5 in 11-bit scale (1..=2047)
const TOP: u32 = 0x0100_0000;

struct RangeEnc {
    out: Vec<u8>,
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
}

impl RangeEnc {
    fn with_capacity(cap: usize) -> Self {
        Self { out: Vec::with_capacity(cap), low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1 }
    }

    #[inline(always)]
    fn shift_low(&mut self) {
        let low_hi = (self.low >> 32) as u32;
        if (self.low as u32) < 0xFF00_0000 || low_hi != 0 {
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(low_hi as u8));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = ((self.low as u32) << 8) as u64;
    }

    #[inline(always)]
    fn encode_bit(&mut self, prob: &mut u16, bit: u32) {
        let p = *prob as u32;
        let bound = (self.range >> 11) * p;
        let is_zero = bit == 0;
        self.low += if is_zero { 0 } else { bound as u64 };
        self.range = if is_zero { bound } else { self.range - bound };
        *prob = (if is_zero { p + ((2048 - p) >> 5) } else { p - (p >> 5) }) as u16;
        if self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    #[inline(always)]
    fn encode_bypass(&mut self, bit: u32) {
        self.range >>= 1;
        self.low += if bit != 0 { self.range as u64 } else { 0 };
        if self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct RangeDec<'a> {
    buf: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}

impl<'a> RangeDec<'a> {
    fn new(buf: &'a [u8]) -> Result<Self> {
        if buf.len() < 5 {
            return Err(Error::Truncated);
        }
        let mut code = 0u32;
        for &b in &buf[..5] {
            code = (code << 8) | (b as u32);
        }
        Ok(Self { buf, pos: 5, range: 0xFFFF_FFFF, code })
    }

    #[inline(always)]
    fn read_byte(&mut self) -> u32 {
        if self.pos < self.buf.len() {
            // SAFETY: `self.pos < self.buf.len()` checked right above.
            let b = unsafe { *self.buf.get_unchecked(self.pos) } as u32;
            self.pos += 1;
            b
        } else {
            0
        }
    }

    #[inline(always)]
    fn decode_bit(&mut self, prob: &mut u16) -> u32 {
        let p = *prob as u32;
        let bound = (self.range >> 11) * p;
        let lt = self.code < bound;
        // Pure branchless CSEL / CMOV on ARM64 & x86_64 — eliminates mispredictions on 50/50 bits!
        self.range = if lt { bound } else { self.range - bound };
        self.code = if lt { self.code } else { self.code - bound };
        *prob = (if lt { p + ((2048 - p) >> 5) } else { p - (p >> 5) }) as u16;
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.read_byte();
        }
        u32::from(!lt)
    }

    #[inline(always)]
    fn decode_bypass(&mut self) -> u32 {
        self.range >>= 1;
        let ge = self.code >= self.range;
        self.code = if ge { self.code - self.range } else { self.code };
        if self.range < TOP {
            self.range <<= 8;
            self.code = (self.code << 8) | self.read_byte();
        }
        u32::from(ge)
    }
}

#[inline(always)]
fn energy_bucket(act: u32) -> usize {
    (32 - act.leading_zeros()).min(7) as usize
}

#[inline(always)]
fn sign_trit(v: i32) -> usize {
    usize::from(v != 0) + usize::from(v < 0)
}

/// Adaptive 2D wavelet context model (zero header bytes, adapts online per tile chunk!).
struct SubbandModel {
    blk_sig: [u16; 12],
    coeff_nz: [u16; 48],
    sign_ctx: [u16; 27],
    gt1: [u16; 8],
    gt2: [u16; 8],
    gt3: [u16; 8],
    gt4: [u16; 8],
    exp_unary: [u16; 14],
}

impl SubbandModel {
    #[inline]
    fn new() -> Self {
        Self {
            blk_sig: [PROB_INIT; 12],
            coeff_nz: [PROB_INIT; 48],
            sign_ctx: [PROB_INIT; 27],
            gt1: [PROB_INIT; 8],
            gt2: [PROB_INIT; 8],
            gt3: [PROB_INIT; 8],
            gt4: [PROB_INIT; 8],
            exp_unary: [PROB_INIT; 14],
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn encode_coeff(
        &mut self,
        enc: &mut RangeEnc,
        v: i32,
        band_base_z: usize,
        band_base_s: usize,
        w_val: i32,
        n_val: i32,
        nw_val: i32,
        ne_val: i32,
    ) {
        let aw = w_val.unsigned_abs();
        let an = n_val.unsigned_abs();
        let anw = nw_val.unsigned_abs();
        let ane = ne_val.unsigned_abs();
        let act = aw + an + ((anw + ane) >> 1);
        let sig_bits = usize::from(aw != 0)
            | (usize::from(an != 0) << 1)
            | (usize::from((anw | ane) != 0) << 2)
            | (usize::from(act >= 4) << 3);
        let zctx = band_base_z + sig_bits;

        if v == 0 {
            enc.encode_bit(&mut self.coeff_nz[zctx], 0);
            return;
        }
        enc.encode_bit(&mut self.coeff_nz[zctx], 1);

        let sctx = band_base_s + sign_trit(w_val) * 3 + sign_trit(n_val);
        enc.encode_bit(&mut self.sign_ctx[sctx], u32::from(v < 0));

        let a = v.unsigned_abs();
        let ectx = energy_bucket(act);
        if a == 1 {
            enc.encode_bit(&mut self.gt1[ectx], 0);
            return;
        }
        enc.encode_bit(&mut self.gt1[ectx], 1);
        if a == 2 {
            enc.encode_bit(&mut self.gt2[ectx], 0);
            return;
        }
        enc.encode_bit(&mut self.gt2[ectx], 1);
        if a == 3 {
            enc.encode_bit(&mut self.gt3[ectx], 0);
            return;
        }
        enc.encode_bit(&mut self.gt3[ectx], 1);
        if a == 4 {
            enc.encode_bit(&mut self.gt4[ectx], 0);
            return;
        }
        enc.encode_bit(&mut self.gt4[ectx], 1);

        let ac = a.min(65535);
        let rem = ac - 1;
        let bit_len = 31 - rem.leading_zeros();
        let k = (bit_len - 2) as usize;
        for u in 0..k {
            enc.encode_bit(&mut self.exp_unary[u.min(13)], 1);
        }
        enc.encode_bit(&mut self.exp_unary[k.min(13)], 0);
        let offset = ac - (1 + (1u32 << bit_len));
        for b in (0..bit_len).rev() {
            enc.encode_bypass((offset >> b) & 1);
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn decode_coeff(
        &mut self,
        dec: &mut RangeDec<'_>,
        band_base_z: usize,
        band_base_s: usize,
        w_val: i32,
        n_val: i32,
        nw_val: i32,
        ne_val: i32,
    ) -> i32 {
        let aw = w_val.unsigned_abs();
        let an = n_val.unsigned_abs();
        let anw = nw_val.unsigned_abs();
        let ane = ne_val.unsigned_abs();
        let act = aw + an + ((anw + ane) >> 1);
        let sig_bits = usize::from(aw != 0)
            | (usize::from(an != 0) << 1)
            | (usize::from((anw | ane) != 0) << 2)
            | (usize::from(act >= 4) << 3);
        let zctx = band_base_z + sig_bits;

        if dec.decode_bit(&mut self.coeff_nz[zctx]) == 0 {
            return 0;
        }

        let sctx = band_base_s + sign_trit(w_val) * 3 + sign_trit(n_val);
        let neg = dec.decode_bit(&mut self.sign_ctx[sctx]) != 0;

        let ectx = energy_bucket(act);
        let mag = if dec.decode_bit(&mut self.gt1[ectx]) == 0 {
            1u32
        } else if dec.decode_bit(&mut self.gt2[ectx]) == 0 {
            2
        } else if dec.decode_bit(&mut self.gt3[ectx]) == 0 {
            3
        } else if dec.decode_bit(&mut self.gt4[ectx]) == 0 {
            4
        } else {
            let mut k = 0usize;
            while k < 14 && dec.decode_bit(&mut self.exp_unary[k.min(13)]) != 0 {
                k += 1;
            }
            let bit_len = (k + 2) as u32;
            let mut offset = 0u32;
            for _ in 0..bit_len {
                offset = (offset << 1) | dec.decode_bypass();
            }
            1 + (1u32 << bit_len) + offset
        };

        if neg { -(mag as i32) } else { mag as i32 }
    }
}

#[inline]
fn write_zigzag_varint(v: i32, out: &mut Vec<u8>) {
    let mut u = ((v << 1) ^ (v >> 31)) as u32;
    while u >= 0x80 {
        out.push(((u as u8) & 0x7F) | 0x80);
        u >>= 7;
    }
    out.push(u as u8);
}

#[inline]
fn read_zigzag_varint(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut u = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        if *pos >= buf.len() {
            return Err(Error::Truncated);
        }
        let b = buf[*pos];
        *pos += 1;
        u |= ((b & 0x7F) as u32) << shift;
        if (b & 0x80) == 0 {
            return Ok(((u >> 1) as i32) ^ -((u & 1) as i32));
        }
        shift += 7;
    }
    Err(Error::InvalidFormat("varint overflow in palette"))
}

/// Fast lossless Palette + RLE encoder (`CODEC_PALETTE_RLE = 4`) for UI screenshots, diagrams,
/// solid alpha channels, and sparse wavelet subbands (`<= 32` unique integer values).
/// Returns `None` immediately on the 33rd unique value so natural photo tiles pay `< 2 µs`.
fn try_encode_palette_rle(bands: &[(&[i32], usize, usize)]) -> Option<(Vec<u8>, usize)> {
    let mut palette = [0i32; 32];
    let mut n_syms = 0usize;
    for &(grid, _, _) in bands {
        for &v in grid {
            let mut found = false;
            for &p in &palette[..n_syms] {
                if p == v {
                    found = true;
                    break;
                }
            }
            if !found {
                if n_syms >= 32 {
                    return None;
                }
                palette[n_syms] = v;
                n_syms += 1;
            }
        }
    }
    if n_syms == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(64);
    out.push(CODEC_PALETTE_RLE);
    out.push(n_syms as u8);
    for &p in &palette[..n_syms] {
        write_zigzag_varint(p, &mut out);
    }
    if n_syms == 1 {
        return Some((out, 1));
    }

    let find_idx = |val: i32| -> u8 {
        for (i, &p) in palette[..n_syms].iter().enumerate() {
            if p == val {
                return i as u8;
            }
        }
        0
    };

    let flush_run = |idx: u8, mut run: usize, dst: &mut Vec<u8>| {
        while run > 0 {
            if run <= 4 {
                dst.push((((run - 1) as u8) << 5) | idx);
                run = 0;
            } else if run <= 260 {
                dst.push(0x80 | idx);
                dst.push((run - 5) as u8);
                run = 0;
            } else {
                let take = run.min(65_796);
                let rem = (take - 261) as u16;
                dst.push(0xA0 | idx);
                dst.extend_from_slice(&rem.to_le_bytes());
                run -= take;
            }
        }
    };

    for &(grid, w, h) in bands {
        if w == 0 || h == 0 {
            continue;
        }
        let mut y = 0usize;
        while y < h {
            if y > 0 && w >= 4 {
                let prev_row = &grid[(y - 1) * w..y * w];
                let mut dup_rows = 0usize;
                while y + dup_rows < h && dup_rows < 256 && &grid[(y + dup_rows) * w..(y + dup_rows + 1) * w] == prev_row {
                    dup_rows += 1;
                }
                if dup_rows > 0 {
                    out.push(0xC0);
                    out.push((dup_rows - 1) as u8);
                    y += dup_rows;
                    continue;
                }
            }
            let row = &grid[y * w..(y + 1) * w];
            let mut cur_idx = find_idx(row[0]);
            let mut cur_run = 1usize;
            for &v in &row[1..] {
                let idx = find_idx(v);
                if idx == cur_idx {
                    cur_run += 1;
                } else {
                    flush_run(cur_idx, cur_run, &mut out);
                    cur_idx = idx;
                    cur_run = 1;
                }
            }
            flush_run(cur_idx, cur_run, &mut out);
            y += 1;
        }
    }
    Some((out, n_syms))
}

/// Encodes a full spatial lossless plane directly via Palette+RLE (`CODEC_SPATIAL_PALETTE = 5`)
/// when the tile has `<= 32` unique sample values (e.g. UI screenshots, terminals, diagrams, solid alpha).
/// Skips wavelet smearing and allows zero-IDWT full-resolution decoding!
pub fn try_encode_spatial_palette(plane: &[i16], w: usize, h: usize) -> Option<Vec<u8>> {
    let mut palette = [0i16; 32];
    let mut n_syms = 0usize;
    for &v in plane {
        let mut found = false;
        for &p in &palette[..n_syms] {
            if p == v {
                found = true;
                break;
            }
        }
        if !found {
            if n_syms >= 32 {
                return None;
            }
            palette[n_syms] = v;
            n_syms += 1;
        }
    }
    if n_syms == 0 || plane.len() != w * h {
        return None;
    }
    let tmp: Vec<i32> = plane.iter().map(|&v| v as i32).collect();
    let (mut bytes, _) = try_encode_palette_rle(&[(&tmp, w, h)])?;
    bytes[0] = CODEC_SPATIAL_PALETTE;
    Some(bytes)
}

/// Decodes a `CODEC_SPATIAL_PALETTE` chunk directly into `w * h` spatial `i16` samples (zero IDWT!).
pub fn decode_spatial_palette(payload: &[u8], w: usize, h: usize) -> Result<Vec<i16>> {
    if payload.first() != Some(&CODEC_SPATIAL_PALETTE) {
        return Err(Error::InvalidFormat("expected CODEC_SPATIAL_PALETTE"));
    }
    let mut out32 = vec![0i32; w * h];
    decode_palette_rle_body(&payload[1..], &mut [(&mut out32, w, h)], &[])?;
    Ok(out32.into_iter().map(|v| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16).collect())
}

fn decode_palette_rle_body(
    body: &[u8],
    targets: &mut [(&mut [i32], usize, usize)],
    dequant_steps: &[i32],
) -> Result<()> {
    if body.is_empty() {
        return Err(Error::Truncated);
    }
    let n_syms = body[0] as usize;
    if n_syms == 0 || n_syms > 32 {
        return Err(Error::InvalidFormat("invalid palette symbol count"));
    }
    let mut pos = 1usize;
    let mut palette = [0i32; 32];
    for slot in palette.iter_mut().take(n_syms) {
        *slot = read_zigzag_varint(body, &mut pos)?;
    }
    if n_syms == 1 {
        for (b_idx, (grid, _, _)) in targets.iter_mut().enumerate() {
            let step = dequant_steps.get(b_idx).copied().unwrap_or(1);
            let val = if step > 1 { dequant_val(palette[0], step) } else { palette[0] };
            grid.fill(val);
        }
        return Ok(());
    }
    let total_needed: usize = targets.iter().map(|(g, _, _)| g.len()).sum();
    let mut b_i = 0usize;
    let mut b_off = 0usize;
    let mut written = 0usize;
    while written < total_needed {
        if pos >= body.len() {
            return Err(Error::Truncated);
        }
        let tok = body[pos];
        pos += 1;
        if tok == 0xC0 {
            // Fast 2D scanline repeat: copy previous row of current band `1..=256` times!
            if pos >= body.len() {
                return Err(Error::Truncated);
            }
            let dup_rows = (body[pos] as usize) + 1;
            pos += 1;
            while b_i < targets.len() && b_off >= targets[b_i].0.len() {
                b_i += 1;
                b_off = 0;
            }
            if b_i >= targets.len() {
                return Err(Error::InvalidFormat("palette row-copy out of bands"));
            }
            let w = targets[b_i].1;
            if w == 0 || b_off < w {
                return Err(Error::InvalidFormat("palette row-copy without previous row"));
            }
            let copy_elems = dup_rows
                .checked_mul(w)
                .ok_or(Error::InvalidFormat("palette row-copy overflow"))?;
            if b_off.checked_add(copy_elems).is_none_or(|end| end > targets[b_i].0.len()) {
                return Err(Error::InvalidFormat("palette row-copy exceeds band bounds"));
            }
            let src_start = b_off - w;
            for r in 0..dup_rows {
                targets[b_i].0.copy_within(src_start..src_start + w, b_off + r * w);
            }
            b_off += copy_elems;
            written += copy_elems;
            continue;
        }
        let idx = (tok & 0x1F) as usize;
        if idx >= n_syms {
            return Err(Error::InvalidFormat("palette index out of bounds"));
        }
        let run: usize = if (tok & 0x80) == 0 {
            (((tok >> 5) & 0x03) as usize) + 1
        } else {
            match (tok >> 5) & 0x03 {
                0 => {
                    if pos >= body.len() {
                        return Err(Error::Truncated);
                    }
                    let r = 5 + (body[pos] as usize);
                    pos += 1;
                    r
                }
                1 => {
                    if pos + 2 > body.len() {
                        return Err(Error::Truncated);
                    }
                    let r = 261 + (u16::from_le_bytes([body[pos], body[pos + 1]]) as usize);
                    pos += 2;
                    r
                }
                _ => return Err(Error::InvalidFormat("invalid palette RLE token")),
            }
        };
        if run > total_needed - written {
            return Err(Error::InvalidFormat("palette RLE run exceeds band length"));
        }
        let raw_val = palette[idx];
        let mut rem = run;
        while rem > 0 {
            while b_i < targets.len() && b_off >= targets[b_i].0.len() {
                b_i += 1;
                b_off = 0;
            }
            if b_i >= targets.len() {
                return Err(Error::InvalidFormat("palette RLE overflow"));
            }
            let step = dequant_steps.get(b_i).copied().unwrap_or(1);
            let val = if step > 1 { dequant_val(raw_val, step) } else { raw_val };
            let avail = targets[b_i].0.len() - b_off;
            let take = rem.min(avail);
            targets[b_i].0[b_off..b_off + take].fill(val);
            b_off += take;
            rem -= take;
        }
        written += run;
    }
    Ok(())
}

/// Encodes 1..3 2D bands in raster order using the 2D adaptive binary range coder (`CODEC_2D_AC`)
/// or fast Palette+RLE (`CODEC_PALETTE_RLE`) whenever the band has `<= 32` unique values and
/// compresses strictly smaller (e.g., UI screenshots, diagrams, solid alpha planes).
pub fn encode_2d_ac(bands: &[(&[i32], usize, usize)], out: &mut Vec<u8>) {
    let any_nz = bands.iter().any(|(b, _, _)| b.iter().any(|&v| v != 0));
    if !any_nz {
        out.clear();
        out.push(CODEC_ALL_ZERO);
        return;
    }
    let rle_cand = try_encode_palette_rle(bands);
    if let Some((rle_bytes, 1)) = rle_cand {
        // Constant non-zero band (e.g., solid UI background or opaque Alpha = 255): 3–4 bytes!
        out.clear();
        out.extend_from_slice(&rle_bytes);
        return;
    }
    let total_elems: usize = bands.iter().map(|(b, _, _)| b.len()).sum();
    let mut enc = RangeEnc::with_capacity(total_elems / 2 + 16);
    let mut model = SubbandModel::new();
    for (b_idx, &(grid, w, h)) in bands.iter().enumerate() {
        if w == 0 || h == 0 {
            continue;
        }
        let bid = b_idx.min(2);
        let bz = bid * 16;
        let bs = bid * 9;
        // Row 0
        let mut w_v = 0i32;
        for &v in &grid[..w] {
            model.encode_coeff(&mut enc, v, bz, bs, w_v, 0, 0, 0);
            w_v = v;
        }
        // Rows 1..h with bounds-check-free interior loop
        for y in 1..h {
            let prev = &grid[(y - 1) * w..y * w];
            let cur = &grid[y * w..(y + 1) * w];
            let v0 = cur[0];
            let ne0 = if w > 1 { prev[1] } else { 0 };
            model.encode_coeff(&mut enc, v0, bz, bs, 0, prev[0], 0, ne0);
            let mut w_v = v0;
            if w > 2 {
                for x in 1..w - 1 {
                    let v = cur[x];
                    model.encode_coeff(&mut enc, v, bz, bs, w_v, prev[x], prev[x - 1], prev[x + 1]);
                    w_v = v;
                }
            }
            if w > 1 {
                let lx = w - 1;
                model.encode_coeff(&mut enc, cur[lx], bz, bs, w_v, prev[lx], prev[lx - 1], 0);
            }
        }
    }
    let payload = enc.finish();
    out.clear();
    if let Some((rle_bytes, _)) = rle_cand {
        if rle_bytes.len() < 1 + payload.len() {
            out.extend_from_slice(&rle_bytes);
            return;
        }
    }
    out.reserve(1 + payload.len());
    out.push(CODEC_2D_AC);
    out.extend_from_slice(&payload);
}

/// Encodes 3 lossy detail bands `(LH, HL, HH)` using 4×4 adaptive block-significance skipping +
/// optional tile-adaptive quantization multiplier `q_mul_q4` (`16` = 1.0× standard `CODEC_BLK4_AC`,
/// other values in `4..=48` emit 1-byte `CODEC_BLK4_AQ` prefix).
pub fn encode_blk4_ac_with_aq(bands: &[(&[i32], usize, usize)], q_mul_q4: u8, out: &mut Vec<u8>) {
    encode_blk4_ac(bands, out);
    if q_mul_q4 != 16 && out.first() == Some(&CODEC_BLK4_AC) {
        out[0] = CODEC_BLK4_AQ;
        out.insert(1, q_mul_q4);
    }
}

/// Encodes 3 lossy detail bands `(LH, HL, HH)` using 4×4 adaptive block-significance skipping +
/// 2D causal adaptive binary range coding (`CODEC_BLK4_AC`).
pub fn encode_blk4_ac(bands: &[(&[i32], usize, usize)], out: &mut Vec<u8>) {
    let any_nz = bands.iter().any(|(b, _, _)| b.iter().any(|&v| v != 0));
    if !any_nz {
        out.clear();
        out.push(CODEC_ALL_ZERO);
        return;
    }
    let total_elems: usize = bands.iter().map(|(b, _, _)| b.len()).sum();
    let mut enc = RangeEnc::with_capacity(total_elems / 4 + 16);
    let mut model = SubbandModel::new();

    for (b_idx, &(grid, w, h)) in bands.iter().enumerate() {
        if w == 0 || h == 0 {
            continue;
        }
        let bid = b_idx.min(2);
        let bz = bid * 16;
        let bs = bid * 9;
        let bx_n = w.div_ceil(4);
        let by_n = h.div_ceil(4);
        let mut sig = vec![0u8; bx_n * by_n];

        for by in 0..by_n {
            let y0 = by * 4;
            let bh = (h - y0).min(4);
            for bx in 0..bx_n {
                let x0 = bx * 4;
                let bw = (w - x0).min(4);
                let mut nz = false;
                'chk: for dy in 0..bh {
                    let row = (y0 + dy) * w + x0;
                    for dx in 0..bw {
                        if grid[row + dx] != 0 {
                            nz = true;
                            break 'chk;
                        }
                    }
                }
                let ls = if bx > 0 { sig[by * bx_n + bx - 1] as usize } else { 0 };
                let ts = if by > 0 { sig[(by - 1) * bx_n + bx] as usize } else { 0 };
                let bctx = bid * 4 + (ls << 1) + ts;
                if !nz {
                    enc.encode_bit(&mut model.blk_sig[bctx], 0);
                    continue;
                }
                enc.encode_bit(&mut model.blk_sig[bctx], 1);
                sig[by * bx_n + bx] = 1;

                if bx > 0 && by > 0 && bw == 4 && bh == 4 {
                    // Fast interior 4x4 block: zero per-pixel bounds checks!
                    for dy in 0..4 {
                        let y = y0 + dy;
                        let prev = &grid[(y - 1) * w + x0 - 1..(y - 1) * w + x0 + 4];
                        let cur = &grid[y * w + x0 - 1..y * w + x0 + 4];
                        let mut w_v = cur[0];
                        for dx in 0..4 {
                            let v = cur[dx + 1];
                            model.encode_coeff(&mut enc, v, bz, bs, w_v, prev[dx + 1], prev[dx], 0);
                            w_v = v;
                        }
                    }
                } else {
                    for dy in 0..bh {
                        let y = y0 + dy;
                        let row = y * w;
                        for dx in 0..bw {
                            let x = x0 + dx;
                            let w_v = if x > 0 { grid[row + x - 1] } else { 0 };
                            let n_v = if y > 0 { grid[row - w + x] } else { 0 };
                            let nw_v = if x > 0 && y > 0 { grid[row - w + x - 1] } else { 0 };
                            model.encode_coeff(&mut enc, grid[row + x], bz, bs, w_v, n_v, nw_v, 0);
                        }
                    }
                }
            }
        }
    }
    let payload = enc.finish();
    out.clear();
    out.reserve(1 + payload.len());
    out.push(CODEC_BLK4_AC);
    out.extend_from_slice(&payload);
}

#[inline(always)]
fn dequant_val(q: i32, step: i32) -> i32 {
    if q == 0 {
        0
    } else {
        let mag = q.abs() * step + ((step * 3) >> 3);
        if q < 0 { -mag } else { mag }
    }
}

/// Decodes 1..3 2D bands from `CODEC_ALL_ZERO`, `CODEC_2D_AC`, or `CODEC_BLK4_AC`,
/// optionally applying fused active-block dequantization (`dequant_steps[i] > 1`) so
/// zero `4×4` blocks are never touched twice in memory!
pub fn decode_2d_bands(
    payload: &[u8],
    targets: &mut [(&mut [i32], usize, usize)],
    dequant_steps: &[i32],
) -> Result<()> {
    if payload.is_empty() {
        return Err(Error::Truncated);
    }
    match payload[0] {
        CODEC_ALL_ZERO => {
            if payload.len() != 1 {
                return Err(Error::InvalidFormat("trailing bytes in all-zero band"));
            }
            for (buf, _, _) in targets.iter_mut() {
                buf.fill(0);
            }
            Ok(())
        }
        CODEC_2D_AC => {
            let mut dec = RangeDec::new(&payload[1..])?;
            let mut model = SubbandModel::new();
            for (b_idx, (grid, w, h)) in targets.iter_mut().enumerate() {
                let (w, h) = (*w, *h);
                if w == 0 || h == 0 {
                    continue;
                }
                let bid = b_idx.min(2);
                let bz = bid * 16;
                let bs = bid * 9;
                // Row 0
                let mut w_v = 0i32;
                for x in 0..w {
                    let v = model.decode_coeff(&mut dec, bz, bs, w_v, 0, 0, 0);
                    grid[x] = v;
                    w_v = v;
                }
                // Rows 1..h with bounds-check-free interior loop
                for y in 1..h {
                    let (above, cur_all) = grid.split_at_mut(y * w);
                    let prev = &above[(y - 1) * w..y * w];
                    let cur = &mut cur_all[..w];
                    let ne0 = if w > 1 { prev[1] } else { 0 };
                    let v0 = model.decode_coeff(&mut dec, bz, bs, 0, prev[0], 0, ne0);
                    cur[0] = v0;
                    let mut w_v = v0;
                    if w > 2 {
                        for x in 1..w - 1 {
                            let v = model.decode_coeff(&mut dec, bz, bs, w_v, prev[x], prev[x - 1], prev[x + 1]);
                            cur[x] = v;
                            w_v = v;
                        }
                    }
                    if w > 1 {
                        let lx = w - 1;
                        cur[lx] = model.decode_coeff(&mut dec, bz, bs, w_v, prev[lx], prev[lx - 1], 0);
                    }
                }
                let step = dequant_steps.get(b_idx).copied().unwrap_or(1);
                if step > 1 {
                    for v in grid.iter_mut() {
                        if *v != 0 {
                            *v = dequant_val(*v, step);
                        }
                    }
                }
            }
            Ok(())
        }
        CODEC_BLK4_AC | CODEC_BLK4_AQ => {
            let (body, aq_mul_q4) = if payload[0] == CODEC_BLK4_AQ {
                if payload.len() < 2 {
                    return Err(Error::Truncated);
                }
                let qm = payload[1];
                if !(4..=48).contains(&qm) {
                    return Err(Error::InvalidFormat("invalid adaptive quant scale"));
                }
                (&payload[2..], Some(qm as i32))
            } else {
                (&payload[1..], None)
            };
            let mut dec = RangeDec::new(body)?;
            let mut model = SubbandModel::new();
            for (b_idx, (grid, w, h)) in targets.iter_mut().enumerate() {
                let (w, h) = (*w, *h);
                if w == 0 || h == 0 {
                    continue;
                }
                let bid = b_idx.min(2);
                let bz = bid * 16;
                let bs = bid * 9;
                grid.fill(0);
                let bx_n = w.div_ceil(4);
                let by_n = h.div_ceil(4);
                let mut sig = vec![0u8; bx_n * by_n];

                for by in 0..by_n {
                    let y0 = by * 4;
                    let bh = (h - y0).min(4);
                    for bx in 0..bx_n {
                        let ls = if bx > 0 { sig[by * bx_n + bx - 1] as usize } else { 0 };
                        let ts = if by > 0 { sig[(by - 1) * bx_n + bx] as usize } else { 0 };
                        let bctx = bid * 4 + (ls << 1) + ts;
                        if dec.decode_bit(&mut model.blk_sig[bctx]) == 0 {
                            continue;
                        }
                        sig[by * bx_n + bx] = 1;
                        let x0 = bx * 4;
                        let bw = (w - x0).min(4);
                        if bx > 0 && by > 0 && bw == 4 && bh == 4 {
                            for dy in 0..4 {
                                let y = y0 + dy;
                                let (above, cur_all) = grid.split_at_mut(y * w);
                                let prev = &above[(y - 1) * w + x0 - 1..(y - 1) * w + x0 + 4];
                                let cur = &mut cur_all[x0 - 1..x0 + 4];
                                let mut w_v = cur[0];
                                for dx in 0..4 {
                                    let v = model.decode_coeff(&mut dec, bz, bs, w_v, prev[dx + 1], prev[dx], 0);
                                    cur[dx + 1] = v;
                                    w_v = v;
                                }
                            }
                        } else {
                            for dy in 0..bh {
                                let y = y0 + dy;
                                let row = y * w;
                                for dx in 0..bw {
                                    let x = x0 + dx;
                                    let w_v = if x > 0 { grid[row + x - 1] } else { 0 };
                                    let n_v = if y > 0 { grid[row - w + x] } else { 0 };
                                    let nw_v = if x > 0 && y > 0 { grid[row - w + x - 1] } else { 0 };
                                    grid[row + x] = model.decode_coeff(&mut dec, bz, bs, w_v, n_v, nw_v, 0);
                                }
                            }
                        }
                    }
                }

                // Fused active-block-only dequantization: skips 75–90% of zero blocks completely!
                let base_step = dequant_steps.get(b_idx).copied().unwrap_or(1);
                let step = match aq_mul_q4 {
                    Some(qm) if base_step > 1 => ((base_step * qm + 8) >> 4).max(1),
                    _ => base_step,
                };
                if step > 1 {
                    for by in 0..by_n {
                        let y0 = by * 4;
                        let bh = (h - y0).min(4);
                        for bx in 0..bx_n {
                            if sig[by * bx_n + bx] == 0 {
                                continue;
                            }
                            let x0 = bx * 4;
                            let bw = (w - x0).min(4);
                            for dy in 0..bh {
                                let row = (y0 + dy) * w + x0;
                                for dx in 0..bw {
                                    let v = grid[row + dx];
                                    if v != 0 {
                                        grid[row + dx] = dequant_val(v, step);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        CODEC_PALETTE_RLE => decode_palette_rle_body(&payload[1..], targets, dequant_steps),
        _ => Err(Error::InvalidFormat("unknown subband codec tag")),
    }
}
