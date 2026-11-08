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
            for i in 0..match_len {
                out[dst_pos + i] = out[src_start + i];
            }
        }
        return Ok(());
    }

    // Match starts inside prefix dictionary `dict`.
    let dict_back = offset - dst_pos;
    let d_start = d_len - dict_back;
    if match_len <= dict_back {
        out[dst_pos..end].copy_from_slice(&dict[d_start..d_start + match_len]);
    } else {
        out[dst_pos..dst_pos + dict_back].copy_from_slice(&dict[d_start..d_len]);
        let rem = match_len - dict_back;
        let out_dst = dst_pos + dict_back;
        if offset >= rem {
            // Note: `out_dst == offset`, so `0..rem` does not overlap `out_dst..out_dst + rem`!
            out.copy_within(0..rem, out_dst);
        } else {
            for i in 0..rem {
                out[out_dst + i] = out[i];
            }
        }
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

    DEC_SCRATCH.with(|cell| {
        let DecScratch {
            lits,
            toks,
            offs,
            ext,
        } = &mut *cell.borrow_mut();

        let lits_slice: &[u8] = if (flags & FLAG_HUF_LITS) != 0 {
            if lit_total == 0 {
                return Err(Error::InvalidFrame("dict_lz empty huf lits"));
            }
            let clen = cur.uleb()?;
            let blob = cur.take(clen)?;
            lits.resize(lit_total, 0);
            huf_decode(blob, &mut lits[..lit_total])?;
            &lits[..lit_total]
        } else {
            cur.take(lit_total)?
        };

        let toks_slice: &[u8] = if (flags & FLAG_HUF_TOKS) != 0 {
            let clen = cur.uleb()?;
            let blob = cur.take(clen)?;
            toks.resize(seq_count, 0);
            huf_decode(blob, &mut toks[..seq_count])?;
            &toks[..seq_count]
        } else {
            cur.take(seq_count)?
        };

        let offs_slice: &[u8] = if (flags & FLAG_HUF_OFFS) != 0 {
            let clen = cur.uleb()?;
            let blob = cur.take(clen)?;
            offs.resize(seq_count, 0);
            huf_decode(blob, &mut offs[..seq_count])?;
            &offs[..seq_count]
        } else {
            cur.take(seq_count)?
        };

        let ext_raw_len = cur.uleb()?;
        if ext_raw_len > raw_len {
            return Err(Error::InvalidFrame("dict_lz ext length"));
        }
        let ext_slice: &[u8] = if (flags & FLAG_HUF_EXT) != 0 {
            if ext_raw_len == 0 {
                return Err(Error::InvalidFrame("dict_lz empty huf ext"));
            }
            let clen = cur.uleb()?;
            let blob = cur.take(clen)?;
            ext.resize(ext_raw_len, 0);
            huf_decode(blob, &mut ext[..ext_raw_len])?;
            &ext[..ext_raw_len]
        } else {
            cur.take(ext_raw_len)?
        };

        let bits_byte_len = cur.uleb()?;
        if bits_byte_len != cur.remaining() {
            return Err(Error::InvalidFrame("dict_lz off_bits length"));
        }
        let off_bits = cur.take(bits_byte_len)?;

        let mut rep = [1usize, 4, 8];
        let mut dst_pos = 0usize;
        let mut lit_pos = 0usize;
        let mut ext_pos = 0usize;
        let mut bit_pos = 0usize;

        let out_ptr = out.as_mut_ptr();
        let lit_ptr = lits_slice.as_ptr();
        let bits_ptr = off_bits.as_ptr();
        let safe_dst = raw_len.saturating_sub(48);
        let safe_lit = lit_total.saturating_sub(16);
        let safe_bit_bytes = bits_byte_len.saturating_sub(8);
        let can_fast_base = raw_len >= 48 && lit_total >= 16 && bits_byte_len >= 8;

        for seq_idx in 0..seq_count {
            let tok = unsafe { *toks_slice.get_unchecked(seq_idx) };
            let oc = unsafe { *offs_slice.get_unchecked(seq_idx) };
            let ll_code = (tok & 0x0f) as usize;
            let ml_code = (tok >> 4) as usize;

            // Fast inline path: no length extensions, sufficient literal/output/bitstream margin.
            if can_fast_base
                & (ll_code < 15)
                & (ml_code < 15)
                & (dst_pos <= safe_dst)
                & (lit_pos <= safe_lit)
                & ((bit_pos >> 3) <= safe_bit_bytes)
                & (oc <= 23)
            {
                let next_dst = dst_pos + ll_code;
                let offset = match oc {
                    0 => rep[0],
                    1 => {
                        let o = rep[1];
                        rep[1] = rep[0];
                        rep[0] = o;
                        o
                    }
                    2 => {
                        let o = rep[2];
                        rep[2] = rep[1];
                        rep[1] = rep[0];
                        rep[0] = o;
                        o
                    }
                    _ => {
                        let b = (oc - 3) as u32;
                        let byte_idx = bit_pos >> 3;
                        let shift = bit_pos & 7;
                        let raw64 = unsafe {
                            bits_ptr.add(byte_idx).cast::<u64>().read_unaligned().to_le()
                        };
                        bit_pos += b as usize;
                        let extra = ((raw64 >> shift) & ((1u64 << b) - 1)) as usize;
                        let o = (1usize << b) | extra;
                        rep[2] = rep[1];
                        rep[1] = rep[0];
                        rep[0] = o;
                        o
                    }
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
                        } else {
                            for i in 0..match_len {
                                *mdst.add(i) = *msrc.add(i);
                            }
                        }
                        dst_pos = next_dst + match_len;
                    }
                    continue;
                }
                // Offset reaches into `dict` (`offset > next_dst`): inline SIMD copy when wholly inside `dict`.
                let dict_back = offset - next_dst;
                if dict_back <= dict.len() && dict_back >= 32 {
                    unsafe {
                        copy16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos));
                        lit_pos += ll_code;
                        let msrc = dict.as_ptr().add(dict.len() - dict_back);
                        let mdst = out_ptr.add(next_dst);
                        copy16(msrc, mdst);
                        copy16(msrc.add(16), mdst.add(16));
                        dst_pos = next_dst + match_len;
                    }
                    continue;
                }
                unsafe {
                    copy16(lit_ptr.add(lit_pos), out_ptr.add(dst_pos));
                }
                lit_pos += ll_code;
                copy_match_general(dict, out, next_dst, offset, match_len)?;
                dst_pos = next_dst + match_len;
                continue;
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
                0 => rep[0],
                1 => {
                    let o = rep[1];
                    rep[1] = rep[0];
                    rep[0] = o;
                    o
                }
                2 => {
                    let o = rep[2];
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = o;
                    o
                }
                3..=23 => {
                    let b = u32::from(oc - 3);
                    let extra = read_bits_checked(off_bits, &mut bit_pos, b)?;
                    let o = (1usize << b) | extra;
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = o;
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
    })
}

pub(crate) fn validate_dict_block(payload: &[u8], dict: &[u8], raw_len: usize) -> Result<()> {
    let mut tmp = vec![0u8; raw_len];
    decode_dict_block(payload, dict, &mut tmp)
}
