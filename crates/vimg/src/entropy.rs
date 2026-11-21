//! Subband entropy tokenization and context-adaptive `tANS / FSE` stream packing:
//! - `CODEC_ALL_ZERO (0)`: 1-byte all-zero subband
//! - `CODEC_RAW_ZRL (1)` / `CODEC_FSE_ZRL (2)`: legacy v0.1 ZRL streams (decode-compatible)
//! - `CODEC_HYBRID_FSE (3)` / `CODEC_HYBRID_RAW (4)`: 1-context 32-symbol Hybrid-Golomb
//! - `CODEC_HYBRID_CTX (5)`: 2-context (`after-zero` 14-sym + `after-nz` 30-sym) and
//!   3-context (`after-zero` 14-sym + `after-|prev|==1` 30-sym + `after-|prev|>=2` 30-sym)
//!   adaptive FSE with separated sign/magnitude bit-packing and trailing-zero elision.

use crate::fse::{fse_decode, fse_encode_min};
use crate::{Error, Result};

const CODEC_ALL_ZERO: u8 = 0;
const CODEC_RAW_ZRL: u8 = 1;
const CODEC_FSE_ZRL: u8 = 2;
const CODEC_HYBRID_FSE: u8 = 3;
const CODEC_HYBRID_RAW: u8 = 4;
const CODEC_HYBRID_CTX: u8 = 5;

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

#[inline(always)]
fn push_zero_run(run_total: usize, dst: &mut Vec<u8>, bw: &mut BitWriter) {
    let mut run = run_total;
    while run > 0 {
        if run <= 4 {
            dst.push((run - 1) as u8);
            break;
        } else if run <= 4098 {
            let rem = (run - 3) as u32;
            let k = 31 - rem.leading_zeros();
            let base = 3 + (1usize << k);
            dst.push((3 + k) as u8);
            bw.write((run - base) as u32, k);
            break;
        } else {
            let chunk = run.min(69634);
            dst.push(15);
            bw.write((chunk - 4099) as u32, 16);
            run -= chunk;
        }
    }
}

#[inline(always)]
fn read_zero_run(t: u8, br: &mut BitReader<'_>) -> Result<usize> {
    if t < 4 {
        Ok((t as usize) + 1)
    } else if t < 15 {
        let k = (t - 3) as u32;
        Ok(3 + (1usize << k) + br.read(k)? as usize)
    } else {
        Ok(4099 + br.read(16)? as usize)
    }
}

#[inline(always)]
fn encode_mag_class(a: u32, sign: u32, bw: &mut BitWriter) -> u8 {
    if a == 1 {
        bw.write(sign, 1);
        0
    } else if a == 2 {
        bw.write(sign, 1);
        1
    } else if a <= 4096 {
        let rem = a - 1;
        let k = 31 - rem.leading_zeros();
        let base = 1 + (1u32 << k);
        bw.write(((a - base) << 1) | sign, k + 1);
        (1 + k) as u8
    } else {
        let clamped = a.min(69632);
        bw.write(((clamped - 4097) << 1) | sign, 17);
        13
    }
}

#[inline(always)]
fn decode_mag_class(m_cls: u8, br: &mut BitReader<'_>) -> Result<(i32, u32)> {
    match m_cls {
        0 => {
            let sign = br.read(1)?;
            Ok((if sign != 0 { -1 } else { 1 }, 1))
        }
        1 => {
            let sign = br.read(1)?;
            Ok((if sign != 0 { -2 } else { 2 }, 2))
        }
        2..=12 => {
            let k = (m_cls - 1) as u32;
            let raw = br.read(k + 1)?;
            let sign = raw & 1;
            let mag = 1 + (1u32 << k) + (raw >> 1);
            Ok((if sign != 0 { -(mag as i32) } else { mag as i32 }, mag))
        }
        13 => {
            let raw = br.read(17)?;
            let sign = raw & 1;
            let mag = 4097 + (raw >> 1);
            Ok((if sign != 0 { -(mag as i32) } else { mag as i32 }, mag))
        }
        _ => Err(Error::InvalidFormat("invalid magnitude class")),
    }
}

/// Tokenizes `vals[0..end]` into 2 or 3 context-separated token streams:
/// - `s[0]` (`ctx == 0`, immediately after a zero-run): **guaranteed non-zero**, alphabet `0..=13`!
/// - `s[1]` (`ctx == 1`, after `|prev| == 1` or start): zero-runs `0..=15` OR mag `16..=29`.
/// - `s[2]` (`ctx == 2`, after `|prev| >= 2`, only if `n_ctx == 3`): high-energy cluster stream.
fn tokenize_ctx(vals: &[i32], end: usize, n_ctx: usize) -> ([Vec<u8>; 3], Vec<u8>) {
    let mut s = [Vec::new(), Vec::new(), Vec::new()];
    let mut bw = BitWriter::new();
    let mut ctx = 1usize;
    let mut i = 0usize;
    while i < end {
        if vals[i] == 0 {
            let start = i;
            while i < end && vals[i] == 0 {
                i += 1;
            }
            let run = i - start;
            if run <= 69634 {
                push_zero_run(run, &mut s[ctx], &mut bw);
                ctx = 0;
            } else {
                // Rare >69634 run: emit first chunk in `ctx`, keep remaining chunks in `ctx = 1`.
                let mut rem = run;
                while rem > 0 {
                    let chunk = rem.min(69634);
                    push_zero_run(chunk, &mut s[ctx], &mut bw);
                    rem -= chunk;
                    ctx = if rem > 0 { 1 } else { 0 };
                }
            }
        } else {
            let v = vals[i];
            let a = v.unsigned_abs();
            let sign = u32::from(v < 0);
            let m_cls = encode_mag_class(a, sign, &mut bw);
            if ctx == 0 {
                s[0].push(m_cls);
            } else {
                s[ctx].push(16 + m_cls);
            }
            ctx = if n_ctx == 3 && a >= 2 { 2 } else { 1 };
            i += 1;
        }
    }
    (s, bw.finish())
}

fn pack_ctx_candidate(streams: &[Vec<u8>], extra: &[u8], out: &mut Vec<u8>) {
    out.clear();
    let n_ctx = streams.len();
    out.push(CODEC_HYBRID_CTX);
    let flags_pos = out.len();
    out.push((n_ctx as u8) << 4);
    let mut fse_mask = 0u8;
    let mut fse_buf = Vec::new();
    for (k, st) in streams.iter().enumerate() {
        put_uleb(out, st.len() as u32);
        if st.is_empty() {
            continue;
        }
        fse_buf.clear();
        if fse_encode_min(st, &mut fse_buf, st.len().saturating_sub(2)) {
            fse_mask |= 1u8 << k;
            put_uleb(out, fse_buf.len() as u32);
            out.extend_from_slice(&fse_buf);
        } else {
            out.extend_from_slice(st);
        }
    }
    out[flags_pos] |= fse_mask;
    out.extend_from_slice(extra);
}

fn unpack_ctx_stream(payload: &[u8], out: &mut [i32]) -> Result<()> {
    if payload.len() < 2 {
        return Err(Error::Truncated);
    }
    let flags = payload[1];
    let n_ctx = ((flags >> 4) & 0x0f) as usize;
    if n_ctx != 2 && n_ctx != 3 {
        return Err(Error::InvalidFormat("invalid hybrid ctx count"));
    }
    let fse_mask = flags & 0x07;
    let mut pos = 2usize;
    let mut s = [Vec::new(), Vec::new(), Vec::new()];
    for k in 0..n_ctx {
        let tok_len = get_uleb(payload, &mut pos)? as usize;
        if tok_len == 0 {
            continue;
        }
        if ((fse_mask >> k) & 1) != 0 {
            let fse_len = get_uleb(payload, &mut pos)? as usize;
            let fse_end = pos.checked_add(fse_len).ok_or(Error::InvalidFormat("fse len overflow"))?;
            let blob = payload.get(pos..fse_end).ok_or(Error::Truncated)?;
            s[k] = vec![0u8; tok_len];
            fse_decode(blob, &mut s[k])?;
            pos = fse_end;
        } else {
            let tok_end = pos.checked_add(tok_len).ok_or(Error::InvalidFormat("tok len overflow"))?;
            let slice = payload.get(pos..tok_end).ok_or(Error::Truncated)?;
            s[k].extend_from_slice(slice);
            pos = tok_end;
        }
    }
    let mut br = BitReader::new(&payload[pos..]);
    let mut p = [0usize; 3];
    let mut ctx = 1usize;
    let mut o = 0usize;
    let n = out.len();
    while p[ctx] < s[ctx].len() {
        let t = s[ctx][p[ctx]];
        p[ctx] += 1;
        if ctx == 0 {
            if o >= n {
                return Err(Error::InvalidFormat("ctx coeff overflow"));
            }
            let (val, mag) = decode_mag_class(t, &mut br)?;
            out[o] = val;
            o += 1;
            ctx = if n_ctx == 3 && mag >= 2 { 2 } else { 1 };
        } else if t < 16 {
            let run = read_zero_run(t, &mut br)?;
            if o + run > n {
                return Err(Error::InvalidFormat("ctx zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
            ctx = if run == 69634 && p[0] == s[0].len() { 1 } else { 0 };
        } else if t < 30 {
            if o >= n {
                return Err(Error::InvalidFormat("ctx coeff overflow"));
            }
            let (val, mag) = decode_mag_class(t - 16, &mut br)?;
            out[o] = val;
            o += 1;
            ctx = if n_ctx == 3 && mag >= 2 { 2 } else { 1 };
        } else {
            return Err(Error::InvalidFormat("invalid ctx token"));
        }
    }
    if p[0] != s[0].len() || p[1] != s[1].len() || p[2] != s[2].len() {
        return Err(Error::InvalidFormat("unconsumed ctx token stream"));
    }
    if o < n {
        out[o..n].fill(0);
    }
    Ok(())
}

fn tokenize_hybrid_1ctx(vals: &[i32], end: usize, tokens: &mut Vec<u8>) -> Vec<u8> {
    tokens.clear();
    let mut bw = BitWriter::new();
    let mut i = 0usize;
    while i < end {
        if vals[i] == 0 {
            let start = i;
            while i < end && vals[i] == 0 {
                i += 1;
            }
            push_zero_run(i - start, tokens, &mut bw);
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
    bw.finish()
}

fn detokenize_hybrid_1ctx(tokens: &[u8], extra: &[u8], out: &mut [i32]) -> Result<()> {
    let mut br = BitReader::new(extra);
    let mut o = 0usize;
    let n = out.len();
    for &t in tokens {
        if t < 16 {
            let run = read_zero_run(t, &mut br)?;
            if o + run > n {
                return Err(Error::InvalidFormat("hybrid zero run overflow"));
            }
            out[o..o + run].fill(0);
            o += run;
        } else if t < 20 {
            if o >= n {
                return Err(Error::InvalidFormat("hybrid coeff overflow"));
            }
            out[o] = match t { 16 => 1, 17 => -1, 18 => 2, _ => -2 };
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

/// Packs a scanned coefficient stream using the strictly smallest of 1-ctx, 2-ctx, or 3-ctx FSE.
pub fn pack_stream(vals: &[i32], out: &mut Vec<u8>) {
    out.clear();
    let mut end = vals.len();
    while end > 0 && vals[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        out.push(CODEC_ALL_ZERO);
        return;
    }

    // Candidate A: 1-context compact 32-symbol Hybrid-Golomb (best for tiny sparse tiles)
    let mut tokens = Vec::with_capacity(end);
    let extra1 = tokenize_hybrid_1ctx(vals, end, &mut tokens);
    let mut fse_buf = Vec::new();
    if fse_encode_min(&tokens, &mut fse_buf, tokens.len().saturating_sub(2)) {
        out.push(CODEC_HYBRID_FSE);
        put_uleb(out, tokens.len() as u32);
        put_uleb(out, fse_buf.len() as u32);
        out.extend_from_slice(&fse_buf);
    } else {
        out.push(CODEC_HYBRID_RAW);
        put_uleb(out, tokens.len() as u32);
        out.extend_from_slice(&tokens);
    }
    out.extend_from_slice(&extra1);

    // Candidate B: 2-context adaptive FSE (`after-zero` 14-sym + `after-nz` 30-sym)
    if tokens.len() >= 40 {
        let mut cand = Vec::with_capacity(out.len());
        let (s2, ex2) = tokenize_ctx(vals, end, 2);
        pack_ctx_candidate(&s2[..2], &ex2, &mut cand);
        if cand.len() < out.len() {
            core::mem::swap(out, &mut cand);
        }
        // Candidate C: 3-context variance-clustered FSE (`after-zero`, `|prev|==1`, `|prev|>=2`)
        if tokens.len() >= 128 {
            let (s3, ex3) = tokenize_ctx(vals, end, 3);
            if s3[2].len() >= 24 {
                pack_ctx_candidate(&s3[..3], &ex3, &mut cand);
                if cand.len() < out.len() {
                    core::mem::swap(out, &mut cand);
                }
            }
        }
    }
}

/// Unpacks any v0.1 or v0.2 subband stream into `out`. Returns `true` if 4×4 oriented scan applies.
pub fn unpack_stream(payload: &[u8], out: &mut [i32]) -> Result<bool> {
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
        CODEC_HYBRID_CTX => {
            unpack_ctx_stream(payload, out)?;
            Ok(true)
        }
        CODEC_HYBRID_RAW => {
            let mut pos = 1usize;
            let tok_len = get_uleb(payload, &mut pos)? as usize;
            let tok_end = pos.checked_add(tok_len).ok_or(Error::InvalidFormat("tok len overflow"))?;
            let tokens = payload.get(pos..tok_end).ok_or(Error::Truncated)?;
            detokenize_hybrid_1ctx(tokens, &payload[tok_end..], out)?;
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
            detokenize_hybrid_1ctx(&tokens, &payload[fse_end..], out)?;
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
