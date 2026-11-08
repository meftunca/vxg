use std::cell::RefCell;

use crate::{columns::huf_decode, Error, Result};

use super::pack::{
    DICT_LZ_MAGIC, FLAG_HUF_EXT, FLAG_HUF_LITS, FLAG_HUF_OFFS, FLAG_HUF_TOKS,
};

struct DecScratch {
    lits: Vec<u8>,
    toks: Vec<u8>,
    offs: Vec<u8>,
    ext: Vec<u8>,
}

impl DecScratch {
    const fn new() -> Self {
        Self {
            lits: Vec::new(),
            toks: Vec::new(),
            offs: Vec::new(),
            ext: Vec::new(),
        }
    }
}

thread_local! {
    static DEC_SCRATCH: RefCell<DecScratch> = const { RefCell::new(DecScratch::new()) };
}

#[inline(always)]
fn unpack_stream<'a, 'b: 'a>(
    cur: &mut WireCursor<'b>,
    huf: bool,
    raw_n: usize,
    scratch: &'a mut Vec<u8>,
) -> Result<&'a [u8]> {
    if huf {
        if raw_n == 0 {
            return Err(Error::InvalidFrame("dict_lz empty huf stream"));
        }
        let clen = cur.uleb()?;
        let blob = cur.take(clen)?;
        if scratch.len() < raw_n {
            scratch.resize(raw_n, 0);
        }
        let buf = &mut scratch[..raw_n];
        huf_decode(blob, buf)?;
        Ok(buf)
    } else {
        cur.take(raw_n)
    }
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> WireCursor<'a> {
    #[inline(always)]
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline(always)]
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    #[inline(always)]
    fn byte(&mut self) -> Result<u8> {
        let &b = self.bytes.get(self.pos).ok_or(Error::Truncated)?;
        self.pos += 1;
        Ok(b)
    }

    #[inline(always)]
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::SizeOverflow)?;
        let s = self.bytes.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(s)
    }

    #[inline(always)]
    fn uleb(&mut self) -> Result<usize> {
        let mut val = 0usize;
        let mut shift = 0u32;
        for _ in 0..5 {
            let b = self.byte()?;
            val |= ((b & 0x7f) as usize) << shift;
            if (b & 0x80) == 0 {
                return Ok(val);
            }
            shift += 7;
        }
        Err(Error::InvalidFrame("dict_lz uleb overflow"))
    }
}

#[inline(always)]
fn read_ext_len(ext: &[u8], pos: &mut usize) -> Result<usize> {
    let &first = ext
        .get(*pos)
        .ok_or(Error::InvalidFrame("dict_lz ext underflow"))?;
    *pos += 1;
    if first < 255 {
        return Ok(first as usize);
    }
    let mut val = 0usize;
    let mut shift = 0u32;
    for _ in 0..5 {
        let &b = ext
            .get(*pos)
            .ok_or(Error::InvalidFrame("dict_lz ext underflow"))?;
        *pos += 1;
        val |= ((b & 0x7f) as usize) << shift;
        if (b & 0x80) == 0 {
            return val.checked_add(255).ok_or(Error::SizeOverflow);
        }
        shift += 7;
    }
    Err(Error::InvalidFrame("dict_lz ext overflow"))
}

#[inline(always)]
fn read_bits_checked(buf: &[u8], bit_pos: &mut usize, bits: u32) -> Result<usize> {
    if bits == 0 {
        return Ok(0);
    }
    let next_bp = (*bit_pos)
        .checked_add(bits as usize)
        .ok_or(Error::SizeOverflow)?;
    if next_bp > buf.len() * 8 {
        return Err(Error::InvalidFrame("dict_lz off_bits underflow"));
    }
    let byte_idx = *bit_pos >> 3;
    let shift = *bit_pos & 7;
    let raw = if byte_idx + 8 <= buf.len() {
        u64::from_le_bytes(buf[byte_idx..byte_idx + 8].try_into().unwrap())
    } else {
        let mut tmp = [0u8; 8];
        let rem = buf.len() - byte_idx;
        tmp[..rem].copy_from_slice(&buf[byte_idx..]);
        u64::from_le_bytes(tmp)
    };
    *bit_pos = next_bp;
    Ok(((raw >> shift) & ((1u64 << bits) - 1)) as usize)
}

#[inline(always)]
unsafe fn copy16(src: *const u8, dst: *mut u8) {
    let v = unsafe { src.cast::<u128>().read_unaligned() };
    unsafe { dst.cast::<u128>().write_unaligned(v) };
}

#[inline(always)]
unsafe fn copy_wild16(src: *const u8, dst: *mut u8, len: usize) {
    unsafe {
        copy16(src, dst);
        if len > 16 {
            let mut copied = 16usize;
            while copied < len {
                copy16(src.add(copied), dst.add(copied));
                copied += 16;
            }
        }
    }
}

#[inline(always)]
fn copy_overlap_slice(out: &mut [u8], src_start: usize, dst_pos: usize, len: usize) {
    let period = dst_pos - src_start;
    if period == 1 {
        let b = out[src_start];
        out[dst_pos..dst_pos + len].fill(b);
        return;
    }
    out.copy_within(src_start..src_start + period, dst_pos);
    let mut written = period;
    while written < len {
        let step = written.min(len - written);
        out.copy_within(dst_pos..dst_pos + step, dst_pos + written);
        written += step;
    }
}

#[inline(always)]
fn copy_match_general(
    dict: &[u8],
    out: &mut [u8],
    dst_pos: usize,
    offset: usize,
    match_len: usize,
) -> Result<()> {
    let d_len = dict.len();
    if offset == 0 || offset > d_len + dst_pos {
        return Err(Error::InvalidFrame("dict_lz invalid match offset"));
    }
    let end = dst_pos
        .checked_add(match_len)
        .ok_or(Error::SizeOverflow)?;
    if end > out.len() {
        return Err(Error::InvalidFrame("dict_lz match exceeds output"));
    }

    if offset <= dst_pos {
        let src_start = dst_pos - offset;
        if offset >= match_len {
            out.copy_within(src_start..src_start + match_len, dst_pos);
        } else {
            copy_overlap_slice(out, src_start, dst_pos, match_len);
        }
        return Ok(());
    }

    let dict_back = offset - dst_pos;
    let d_start = d_len - dict_back;
    if match_len <= dict_back {
        out[dst_pos..end].copy_from_slice(&dict[d_start..d_start + match_len]);
    } else {
        out[dst_pos..dst_pos + dict_back].copy_from_slice(&dict[d_start..d_len]);
        let rem = match_len - dict_back;
        let out_dst = dst_pos + dict_back;
        if offset >= rem {
            out.copy_within(0..rem, out_dst);
        } else {
            copy_overlap_slice(out, 0, out_dst, rem);
        }
    }
    Ok(())
}

#[inline(always)]
unsafe fn decode_offset_fast(
    oc: u8,
    r0: &mut usize,
    r1: &mut usize,
    r2: &mut usize,
    bits_ptr: *const u8,
    bit_pos: &mut usize,
) -> usize {
    match oc {
        0 => *r0,
        1 => {
            let o = *r1;
            *r1 = *r0;
            *r0 = o;
            o
        }
        2 => {
            let o = *r2;
            *r2 = *r1;
            *r1 = *r0;
            *r0 = o;
            o
        }
        _ => {
            let b = (oc - 3) as u32;
            let raw64 = unsafe { bits_ptr.add(*bit_pos >> 3).cast::<u64>().read_unaligned().to_le() };
            let shift = *bit_pos & 7;
            *bit_pos += b as usize;
            let o = (1usize << b) | (((raw64 >> shift) & ((1u64 << b) - 1)) as usize);
            *r2 = *r1;
            *r1 = *r0;
            *r0 = o;
            o
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn execute_sequences(
    dict: &[u8],
    out: &mut [u8],
    lits_slice: &[u8],
    toks_slice: &[u8],
    offs_slice: &[u8],
    ext_slice: &[u8],
    off_bits: &[u8],
) -> Result<()> {
    let raw_len = out.len();
    let seq_count = toks_slice.len();
    let lit_total = lits_slice.len();
    let ext_raw_len = ext_slice.len();
    let bits_byte_len = off_bits.len();

    let (mut r0, mut r1, mut r2) = (1usize, 4usize, 8usize);
    let (mut dst_pos, mut lit_pos, mut ext_pos, mut bit_pos) = (0usize, 0usize, 0usize, 0usize);

    let out_ptr = out.as_mut_ptr();
    let lit_ptr = lits_slice.as_ptr();
    let ext_ptr = ext_slice.as_ptr();
    let bits_ptr = off_bits.as_ptr();
    let (d_ptr, d_len) = (dict.as_ptr(), dict.len());

    let safe_dst_short = raw_len.saturating_sub(48);
    let safe_lit_short = lit_total.saturating_sub(16);
    let safe_dst_med = raw_len.saturating_sub(560);
    let safe_lit_med = lit_total.saturating_sub(288);
    let safe_bit_bytes = bits_byte_len.saturating_sub(8);
    let can_fast_short = raw_len >= 48 && lit_total >= 16 && bits_byte_len >= 8;
    let can_fast_med = raw_len >= 560 && lit_total >= 288 && bits_byte_len >= 8;

    for seq_idx in 0..seq_count {
        let tok = unsafe { *toks_slice.get_unchecked(seq_idx) };
        let oc = unsafe { *offs_slice.get_unchecked(seq_idx) };
        let ll_code = (tok & 0x0f) as usize;
        let ml_code = (tok >> 4) as usize;

        if can_fast_short
            & (ll_code < 15)
            & (ml_code < 15)
            & (dst_pos <= safe_dst_short)
            & (lit_pos <= safe_lit_short)
            & ((bit_pos >> 3) <= safe_bit_bytes)
            & (oc <= 23)
        {
            let next_dst = dst_pos + ll_code;
            let offset = unsafe {
                decode_offset_fast(oc, &mut r0, &mut r1, &mut r2, bits_ptr, &mut bit_pos)
            };
            let match_len = ml_code + 3;
            if offset <= next_dst {
                unsafe {
                    copy16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos));
                    lit_pos += ll_code;
                    let msrc = out_ptr.add(next_dst - offset);
                    let mdst = out_ptr.add(next_dst);
                    if offset >= 16 {
                        copy16(msrc, mdst);
                        copy16(msrc.add(16), mdst.add(16));
                    } else if offset >= match_len {
                        copy16(msrc, mdst);
                    } else if offset == 1 {
                        core::ptr::write_bytes(mdst, *msrc, match_len);
                    } else {
                        for i in 0..match_len {
                            *mdst.add(i) = *msrc.add(i);
                        }
                    }
                    dst_pos = next_dst + match_len;
                }
                continue;
            }
            let dict_back = offset - next_dst;
            if dict_back <= d_len && dict_back >= 32 {
                unsafe {
                    copy16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos));
                    lit_pos += ll_code;
                    let msrc = d_ptr.add(d_len - dict_back);
                    let mdst = out_ptr.add(next_dst);
                    copy16(msrc, mdst);
                    copy16(msrc.add(16), mdst.add(16));
                    dst_pos = next_dst + match_len;
                }
                continue;
            }
            unsafe { copy16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos)) };
            lit_pos += ll_code;
            copy_match_general(dict, out, next_dst, offset, match_len)?;
            dst_pos = next_dst + match_len;
            continue;
        }

        if can_fast_med
            & (dst_pos <= safe_dst_med)
            & (lit_pos <= safe_lit_med)
            & ((bit_pos >> 3) <= safe_bit_bytes)
            & (oc <= 23)
            & (ext_pos + 2 <= ext_raw_len)
        {
            let mut ea = 0usize;
            let lit_len = if ll_code == 15 {
                let e = unsafe { *ext_ptr.add(ext_pos) };
                if e == 255 { usize::MAX } else { ea = 1; 15 + e as usize }
            } else {
                ll_code
            };
            let match_len = if (lit_len != usize::MAX) & (ml_code == 15) {
                let e = unsafe { *ext_ptr.add(ext_pos + ea) };
                if e == 255 { usize::MAX } else { ea += 1; 18 + e as usize }
            } else {
                ml_code + 3
            };
            if (lit_len != usize::MAX) & (match_len != usize::MAX) {
                ext_pos += ea;
                let next_dst = dst_pos + lit_len;
                let offset = unsafe {
                    decode_offset_fast(oc, &mut r0, &mut r1, &mut r2, bits_ptr, &mut bit_pos)
                };
                if offset <= next_dst {
                    unsafe {
                        copy_wild16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), lit_len);
                        lit_pos += lit_len;
                        let msrc = out_ptr.add(next_dst - offset);
                        let mdst = out_ptr.add(next_dst);
                        if offset >= 16 || offset >= match_len {
                            copy_wild16(msrc, mdst, match_len);
                        } else {
                            copy_overlap_slice(out, next_dst - offset, next_dst, match_len);
                        }
                        dst_pos = next_dst + match_len;
                    }
                    continue;
                }
                let dict_back = offset - next_dst;
                if dict_back <= d_len && dict_back >= match_len + 16 {
                    unsafe {
                        copy_wild16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), lit_len);
                        lit_pos += lit_len;
                        copy_wild16(d_ptr.add(d_len - dict_back), out_ptr.add(next_dst), match_len);
                        dst_pos = next_dst + match_len;
                    }
                    continue;
                }
                unsafe { copy_wild16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), lit_len) };
                lit_pos += lit_len;
                copy_match_general(dict, out, next_dst, offset, match_len)?;
                dst_pos = next_dst + match_len;
                continue;
            }
        }

        // General checked path.
        let mut lit_len = ll_code;
        if ll_code == 15 {
            lit_len = lit_len
                .checked_add(read_ext_len(ext_slice, &mut ext_pos)?)
                .ok_or(Error::SizeOverflow)?;
        }
        let mut match_len = ml_code + 3;
        if ml_code == 15 {
            match_len = match_len
                .checked_add(read_ext_len(ext_slice, &mut ext_pos)?)
                .ok_or(Error::SizeOverflow)?;
        }

        let lit_end = lit_pos.checked_add(lit_len).ok_or(Error::SizeOverflow)?;
        let next_dst = dst_pos.checked_add(lit_len).ok_or(Error::SizeOverflow)?;
        if lit_end > lit_total || next_dst > raw_len {
            return Err(Error::InvalidFrame("dict_lz literal bounds"));
        }
        if lit_len > 0 {
            out[dst_pos..next_dst].copy_from_slice(&lits_slice[lit_pos..lit_end]);
            lit_pos = lit_end;
        }

        let offset = match oc {
            0 => r0,
            1 => {
                let o = r1;
                r1 = r0;
                r0 = o;
                o
            }
            2 => {
                let o = r2;
                r2 = r1;
                r1 = r0;
                r0 = o;
                o
            }
            3..=23 => {
                let b = u32::from(oc - 3);
                let extra = read_bits_checked(off_bits, &mut bit_pos, b)?;
                let o = (1usize << b) | extra;
                r2 = r1;
                r1 = r0;
                r0 = o;
                o
            }
            _ => return Err(Error::InvalidFrame("dict_lz invalid off_code")),
        };

        copy_match_general(dict, out, next_dst, offset, match_len)?;
        dst_pos = next_dst + match_len;
    }

    if ext_pos != ext_raw_len {
        return Err(Error::InvalidFrame("dict_lz ext trailing"));
    }
    if bit_pos.div_ceil(8) != bits_byte_len {
        return Err(Error::InvalidFrame("dict_lz off_bits trailing"));
    }
    if (bit_pos & 7) != 0 {
        let tail_mask = !((1u8 << (bit_pos & 7)) - 1);
        if (off_bits[bits_byte_len - 1] & tail_mask) != 0 {
            return Err(Error::InvalidFrame("dict_lz off_bits padding"));
        }
    }

    let rem_lits = lit_total - lit_pos;
    if dst_pos.checked_add(rem_lits) != Some(raw_len) {
        return Err(Error::InvalidFrame("dict_lz output length mismatch"));
    }
    if rem_lits > 0 {
        out[dst_pos..raw_len].copy_from_slice(&lits_slice[lit_pos..lit_total]);
    }
    Ok(())
}

pub(crate) fn decode_dict_block(payload: &[u8], dict: &[u8], out: &mut [u8]) -> Result<()> {
    let raw_len = out.len();
    let mut cur = WireCursor::new(payload);
    let flags = cur.byte()?;
    if (flags & 0x80) == 0 || (flags & 0x70) != 0 {
        return Err(Error::InvalidFrame("dict_lz header flags"));
    }
    let seq_count = cur.uleb()?;
    let lit_total = cur.uleb()?;
    if lit_total > raw_len || seq_count > raw_len / 3 {
        return Err(Error::InvalidFrame("dict_lz header counts"));
    }

    if seq_count == 0 {
        if lit_total != raw_len || flags != (DICT_LZ_MAGIC | FLAG_HUF_LITS) {
            return Err(Error::InvalidFrame("dict_lz literal-only header"));
        }
        let clen = cur.uleb()?;
        let blob = cur.take(clen)?;
        if cur.remaining() != 0 {
            return Err(Error::InvalidFrame("dict_lz trailing bytes"));
        }
        return huf_decode(blob, out);
    }

    // Zero-TLS, zero-Huffman fast path for ultra-compact blocks (e.g. long dictionary matches).
    if (flags & 0x0f) == 0 {
        let lits_slice = cur.take(lit_total)?;
        let toks_slice = cur.take(seq_count)?;
        let offs_slice = cur.take(seq_count)?;
        let ext_raw_len = cur.uleb()?;
        if ext_raw_len > raw_len {
            return Err(Error::InvalidFrame("dict_lz ext length"));
        }
        let ext_slice = cur.take(ext_raw_len)?;
        let bits_byte_len = cur.uleb()?;
        if bits_byte_len != cur.remaining() {
            return Err(Error::InvalidFrame("dict_lz off_bits length"));
        }
        let off_bits = cur.take(bits_byte_len)?;
        return execute_sequences(
            dict, out, lits_slice, toks_slice, offs_slice, ext_slice, off_bits,
        );
    }

    DEC_SCRATCH.with(|cell| {
        let DecScratch {
            lits,
            toks,
            offs,
            ext,
        } = &mut *cell.borrow_mut();

        let lits_slice = unpack_stream(&mut cur, (flags & FLAG_HUF_LITS) != 0, lit_total, lits)?;
        let toks_slice = unpack_stream(&mut cur, (flags & FLAG_HUF_TOKS) != 0, seq_count, toks)?;
        let offs_slice = unpack_stream(&mut cur, (flags & FLAG_HUF_OFFS) != 0, seq_count, offs)?;
        let ext_raw_len = cur.uleb()?;
        if ext_raw_len > raw_len {
            return Err(Error::InvalidFrame("dict_lz ext length"));
        }
        let ext_slice = unpack_stream(&mut cur, (flags & FLAG_HUF_EXT) != 0, ext_raw_len, ext)?;
        let bits_byte_len = cur.uleb()?;
        if bits_byte_len != cur.remaining() {
            return Err(Error::InvalidFrame("dict_lz off_bits length"));
        }
        let off_bits = cur.take(bits_byte_len)?;

        execute_sequences(
            dict, out, lits_slice, toks_slice, offs_slice, ext_slice, off_bits,
        )
    })
}

pub(crate) fn validate_dict_block(payload: &[u8], dict: &[u8], raw_len: usize) -> Result<()> {
    let mut tmp = vec![0u8; raw_len];
    decode_dict_block(payload, dict, &mut tmp)
}
