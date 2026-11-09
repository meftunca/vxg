//! Pure-Rust, zero-C-dependency (`wasm32`-compatible) 2-way interleaved `tANS / FSE`
//! (Finite State Entropy) byte-stream encoder and decoder for `BLOCK_DICT_LZ`.
//!
//! Competes strictly by exact byte count against canonical Huffman and raw streams,
//! capturing sub-bit-per-symbol entropy (`< 1.0 bit/sym`) on skewed `off_codes`,
//! `toks`, `len_ext`, and low-entropy `lits` distributions.

use crate::{Error, Result};

use super::pack::BitWriter;

const FSE_RLE_MAGIC: u8 = 0x00;
const FSE_TANS_BASE: u8 = 0x80; // 0x80 => L=8 (256), 0x81 => L=9 (512), 0x82 => L=10 (1024)
const MAX_TABLE_LOG: u32 = 10;
const MAX_STATES: usize = 1 << MAX_TABLE_LOG;

#[derive(Clone, Copy, Default)]
struct EncSym {
    count: u16,
    nb_lo: u8,
    thresh: u16,
    occ_base: u16,
}

struct HdrBitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> HdrBitReader<'a> {
    #[inline(always)]
    fn new(bytes: &'a [u8], start_byte: usize) -> Self {
        Self {
            bytes,
            bit_pos: start_byte * 8,
        }
    }

    #[inline(always)]
    fn read(&mut self, bits: u32) -> Result<u32> {
        if bits == 0 {
            return Ok(0);
        }
        let next = self
            .bit_pos
            .checked_add(bits as usize)
            .ok_or(Error::SizeOverflow)?;
        if next > self.bytes.len() * 8 {
            return Err(Error::InvalidFrame("fse header truncated"));
        }
        let b_idx = self.bit_pos >> 3;
        let shift = self.bit_pos & 7;
        let raw = if b_idx + 4 <= self.bytes.len() {
            u32::from_le_bytes(self.bytes[b_idx..b_idx + 4].try_into().unwrap())
        } else {
            let mut tmp = [0u8; 4];
            let rem = self.bytes.len() - b_idx;
            tmp[..rem].copy_from_slice(&self.bytes[b_idx..]);
            u32::from_le_bytes(tmp)
        };
        self.bit_pos = next;
        Ok((raw >> shift) & ((1u32 << bits) - 1))
    }

    #[inline(always)]
    fn finish_byte_aligned(&self) -> Result<usize> {
        let rem = self.bit_pos & 7;
        if rem != 0 {
            let b = self.bytes[self.bit_pos >> 3];
            if (b >> rem) != 0 {
                return Err(Error::InvalidFrame("fse header padding"));
            }
        }
        Ok(self.bit_pos.div_ceil(8))
    }
}

struct RevBitReader<'a> {
    bytes: &'a [u8],
    bit_cursor: usize,
}

impl<'a> RevBitReader<'a> {
    #[inline(always)]
    fn init(bytes: &'a [u8]) -> Result<Self> {
        let &last = bytes
            .last()
            .ok_or(Error::InvalidFrame("fse empty bitstream"))?;
        if last == 0 {
            return Err(Error::InvalidFrame("fse missing sentinel"));
        }
        let sentinel_bit = 7 - (last.leading_zeros() as usize);
        let bit_cursor = (bytes.len() - 1) * 8 + sentinel_bit;
        Ok(Self { bytes, bit_cursor })
    }

    #[inline(always)]
    fn pop(&mut self, bits: u32) -> Result<usize> {
        let b = bits as usize;
        if b > self.bit_cursor {
            return Err(Error::InvalidFrame("fse bitstream underflow"));
        }
        self.bit_cursor -= b;
        if b == 0 {
            return Ok(0);
        }
        let byte_idx = self.bit_cursor >> 3;
        let shift = self.bit_cursor & 7;
        let raw = if byte_idx + 8 <= self.bytes.len() {
            unsafe {
                self.bytes
                    .as_ptr()
                    .add(byte_idx)
                    .cast::<u64>()
                    .read_unaligned()
                    .to_le()
            }
        } else {
            let mut tmp = [0u8; 8];
            let rem = self.bytes.len() - byte_idx;
            tmp[..rem].copy_from_slice(&self.bytes[byte_idx..]);
            u64::from_le_bytes(tmp)
        };
        Ok(((raw >> shift) & ((1u64 << bits) - 1)) as usize)
    }
}

fn normalize_counts(
    hist: &[u32; 256],
    total: usize,
    max_sym: usize,
    table_log: u32,
    norm: &mut [u16; 256],
) {
    let n_states = 1usize << table_log;
    norm.fill(0);
    let mut sum = 0usize;
    let mut best_sym = 0usize;
    let mut best_freq = 0u32;

    for s in 0..=max_sym {
        let f = hist[s];
        if f == 0 {
            continue;
        }
        if f > best_freq {
            best_freq = f;
            best_sym = s;
        }
        let scaled = ((f as u64 * n_states as u64 + (total as u64 / 2)) / total as u64).max(1);
        let c = (scaled as usize).min(n_states - 1) as u16;
        norm[s] = c;
        sum += c as usize;
    }

    while sum > n_states {
        let mut pick = usize::MAX;
        let mut best_excess = i64::MIN;
        for s in 0..=max_sym {
            let c = norm[s] as i64;
            if c > 1 {
                let excess = c * (total as i64) - (hist[s] as i64) * (n_states as i64);
                if excess > best_excess {
                    best_excess = excess;
                    pick = s;
                }
            }
        }
        norm[pick] -= 1;
        sum -= 1;
    }

    if sum < n_states {
        norm[best_sym] += (n_states - sum) as u16;
    }
    debug_assert_eq!(norm[..=max_sym].iter().map(|&x| x as usize).sum::<usize>(), n_states);
}

#[inline]
fn spread_step(n_states: usize) -> usize {
    (n_states >> 1) + (n_states >> 3) + 3
}

/// Encode `src` into `out` using 2-way interleaved `tANS / FSE` (or 2-byte RLE for constant streams).
/// Returns `true` iff `src.len() >= min_src` and the encoded representation is strictly `<= max_out`.
pub(super) fn fse_encode_min(
    src: &[u8],
    out: &mut Vec<u8>,
    max_out: usize,
    min_src: usize,
) -> bool {
    let m = src.len();
    out.clear();
    if m < min_src || m < 16 || max_out < 2 {
        return false;
    }

    let mut hist = [0u32; 256];
    for &b in src {
        hist[b as usize] += 1;
    }

    let mut distinct = 0usize;
    let mut max_sym = 0usize;
    let mut max_freq = 0u32;
    for (s, &f) in hist.iter().enumerate() {
        if f > 0 {
            distinct += 1;
            max_sym = s;
            if f > max_freq {
                max_freq = f;
            }
        }
    }

    if distinct == 1 {
        out.extend_from_slice(&[FSE_RLE_MAGIC, max_sym as u8]);
        return out.len() <= max_out;
    }

    // Fast bailout when flat distribution cannot possibly beat Huffman or raw.
    if (max_freq as usize) * 16 < m && distinct > 64 {
        return false;
    }

    let table_log: u32 = if m < 256 && distinct <= 32 {
        8
    } else if m < 1024 || distinct <= 48 {
        9
    } else {
        10
    };
    let n_states = 1usize << table_log;
    if distinct > n_states / 2 {
        return false;
    }

    let mut norm = [0u16; 256];
    normalize_counts(&hist, m, max_sym, table_log, &mut norm);

    // Serialize header: `[magic, max_sym, presence_bmap, bitpacked_counts]`.
    let m_syms = max_sym + 1;
    let bmap_bytes = m_syms.div_ceil(8);
    if 2 + bmap_bytes >= max_out {
        return false;
    }
    out.reserve((m / 2).min(max_out));
    out.push(FSE_TANS_BASE + (table_log - 8) as u8);
    out.push(max_sym as u8);
    let bmap_start = out.len();
    out.resize(bmap_start + bmap_bytes, 0);
    for s in 0..m_syms {
        if norm[s] > 0 {
            out[bmap_start + (s >> 3)] |= 1u8 << (s & 7);
        }
    }

    {
        let mut hbw = BitWriter::new(out);
        for &c in &norm[..m_syms] {
            if c == 0 {
                continue;
            }
            let v = (c - 1) as u32;
            if v == 0 {
                hbw.put(0, 1);
            } else if v <= 16 {
                hbw.put(0b01, 2); // bit0 = 1, bit1 = 0
                hbw.put(v - 1, 4);
            } else {
                hbw.put(0b11, 2); // bit0 = 1, bit1 = 1
                hbw.put(v, table_log);
            }
        }
        hbw.finish();
    }
    if out.len() + 2 > max_out {
        out.clear();
        return false;
    }

    // Build spread occurrence -> state lookup table.
    let mut syms = [EncSym::default(); 256];
    let mut occ_cursor = 0u16;
    for s in 0..m_syms {
        let c = norm[s];
        if c == 0 {
            continue;
        }
        let k = 31 - (c as u32).leading_zeros();
        let nb_hi = table_log - k;
        let nb_lo = nb_hi - 1;
        let thresh = (((c as usize) << nb_hi) - n_states) as u16;
        syms[s] = EncSym {
            count: c,
            nb_lo: nb_lo as u8,
            thresh,
            occ_base: occ_cursor,
        };
        occ_cursor += c;
    }

    let mut table_sym = [0u8; MAX_STATES];
    let step = spread_step(n_states);
    let mask = n_states - 1;
    let mut pos = 0usize;
    for (s, &c) in norm[..m_syms].iter().enumerate() {
        for _ in 0..c {
            table_sym[pos] = s as u8;
            pos = (pos + step) & mask;
        }
    }

    let mut enc_state_for_occ = [0u16; MAX_STATES];
    let mut cur_occ = [0u16; 256];
    for (state, &sym) in table_sym[..n_states].iter().enumerate() {
        let s = sym as usize;
        let idx = syms[s].occ_base + cur_occ[s];
        cur_occ[s] += 1;
        enc_state_for_occ[idx as usize] = state as u16;
    }

    // Backwards 2-way interleaved tANS encoding (`states[0]` = even, `states[1]` = odd).
    let mut states = [0usize; 2];
    states[(m - 1) & 1] = enc_state_for_occ[syms[src[m - 1] as usize].occ_base as usize] as usize;
    states[(m - 2) & 1] = enc_state_for_occ[syms[src[m - 2] as usize].occ_base as usize] as usize;

    {
        let mut bw = BitWriter::new(out);
        let mut i = m - 2;
        while i > 0 {
            i -= 1;
            let sym_info = unsafe { *syms.get_unchecked(src[i] as usize) };
            let st = &mut states[i & 1];
            let nb = sym_info.nb_lo as u32 + u32::from(*st >= sym_info.thresh as usize);
            let y = *st + n_states;
            let x = y >> nb;
            let bits = (y as u32) & ((1u32 << nb) - 1);
            bw.put(bits, nb);
            let idx = sym_info.occ_base as usize + (x - sym_info.count as usize);
            *st = unsafe { *enc_state_for_occ.get_unchecked(idx) as usize };
        }
        bw.put(states[1] as u32, table_log);
        bw.put(states[0] as u32, table_log);
        bw.put(1, 1);
        bw.finish();
    }

    if out.len() <= max_out {
        true
    } else {
        out.clear();
        false
    }
}

/// Decode an `FSE` (or 2-byte RLE) compressed byte stream `blob` into `out`.
pub(super) fn fse_decode(blob: &[u8], out: &mut [u8]) -> Result<()> {
    let m = out.len();
    if m == 0 || blob.is_empty() {
        return Err(Error::InvalidFrame("fse empty stream"));
    }
    let magic = blob[0];
    if magic == FSE_RLE_MAGIC {
        if blob.len() != 2 {
            return Err(Error::InvalidFrame("fse rle length"));
        }
        out.fill(blob[1]);
        return Ok(());
    }
    if !(FSE_TANS_BASE..=FSE_TANS_BASE + 2).contains(&magic) || m < 2 {
        return Err(Error::InvalidFrame("fse header magic"));
    }
    let table_log = 8 + (magic - FSE_TANS_BASE) as u32;
    let n_states = 1usize << table_log;
    let max_sym = *blob.get(1).ok_or(Error::Truncated)? as usize;
    let m_syms = max_sym + 1;
    let bmap_bytes = m_syms.div_ceil(8);
    let bmap_end = 2 + bmap_bytes;
    let bmap = blob.get(2..bmap_end).ok_or(Error::Truncated)?;
    if (m_syms & 7) != 0 && (bmap[bmap_bytes - 1] >> (m_syms & 7)) != 0 {
        return Err(Error::InvalidFrame("fse bmap padding"));
    }

    let mut hbr = HdrBitReader::new(blob, bmap_end);
    let mut norm = [0u16; 256];
    let mut sum = 0usize;
    let mut distinct = 0usize;
    for s in 0..m_syms {
        if ((bmap[s >> 3] >> (s & 7)) & 1) == 0 {
            continue;
        }
        let v = if hbr.read(1)? == 0 {
            0u32
        } else if hbr.read(1)? == 0 {
            1 + hbr.read(4)?
        } else {
            hbr.read(table_log)?
        };
        let c = (v as usize) + 1;
        if c >= n_states {
            return Err(Error::InvalidFrame("fse symbol count overflow"));
        }
        norm[s] = c as u16;
        sum += c;
        distinct += 1;
        if sum > n_states {
            return Err(Error::InvalidFrame("fse norm sum overflow"));
        }
    }
    if distinct < 2 || sum != n_states || norm[max_sym] == 0 {
        return Err(Error::InvalidFrame("fse invalid norm table"));
    }
    let stream_start = hbr.finish_byte_aligned()?;
    let stream = blob.get(stream_start..).ok_or(Error::Truncated)?;

    // Build `dec_table[0..n_states]`: `(base_state: u16) | (nb: u8 << 16) | (sym: u8 << 24)`.
    let mut table_sym = [0u8; MAX_STATES];
    let step = spread_step(n_states);
    let mask = n_states - 1;
    let mut pos = 0usize;
    for (s, &c) in norm[..m_syms].iter().enumerate() {
        for _ in 0..c {
            table_sym[pos] = s as u8;
            pos = (pos + step) & mask;
        }
    }

    let mut next_occ = norm;
    let mut dec_table = [0u32; MAX_STATES];
    for state in 0..n_states {
        let s = table_sym[state] as usize;
        let x = next_occ[s] as u32;
        next_occ[s] = (x + 1) as u16;
        let nb = table_log - (31 - x.leading_zeros());
        let base = ((x as usize) << nb) - n_states;
        dec_table[state] = (base as u32) | (nb << 16) | ((s as u32) << 24);
    }

    let mut rbr = RevBitReader::init(stream)?;
    let mut st0 = rbr.pop(table_log)?;
    let mut st1 = rbr.pop(table_log)?;

    let body = m - 2;
    let pairs = body / 2;
    let mut idx = 0usize;
    for _ in 0..pairs {
        let e0 = unsafe { *dec_table.get_unchecked(st0) };
        let e1 = unsafe { *dec_table.get_unchecked(st1) };
        out[idx] = (e0 >> 24) as u8;
        out[idx + 1] = (e1 >> 24) as u8;
        let add0 = rbr.pop((e0 >> 16) & 0xff)?;
        let add1 = rbr.pop((e1 >> 16) & 0xff)?;
        st0 = ((e0 & 0xffff) as usize) + add0;
        st1 = ((e1 & 0xffff) as usize) + add1;
        idx += 2;
    }
    if (body & 1) != 0 {
        let e0 = unsafe { *dec_table.get_unchecked(st0) };
        out[idx] = (e0 >> 24) as u8;
        let add0 = rbr.pop((e0 >> 16) & 0xff)?;
        st0 = ((e0 & 0xffff) as usize) + add0;
        idx += 1;
    }

    let states = [st0, st1];
    out[idx] = (unsafe { *dec_table.get_unchecked(states[(m - 2) & 1]) } >> 24) as u8;
    out[idx + 1] = (unsafe { *dec_table.get_unchecked(states[(m - 1) & 1]) } >> 24) as u8;

    if rbr.bit_cursor != 0 {
        return Err(Error::InvalidFrame("fse trailing bits"));
    }
    Ok(())
}
