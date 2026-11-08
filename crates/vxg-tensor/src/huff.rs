//! Zero-dependency Canonical Huffman codec (`14`-bit single-lookup `L1` table) used by
//! `EC-Turbo` tensor streams (`entropy_pack`) and `VXLZ` metadata compression (`meta_lz`).

use crate::format_io::Cur;
use crate::{Error, Result};

const MAX_BITS: usize = 14;
const LUT_SIZE: usize = 1 << MAX_BITS;

#[inline(always)]
fn bit_rev(mut c: u32, len: u8) -> u16 {
    let mut r = 0u32;
    for _ in 0..len {
        r = (r << 1) | (c & 1);
        c >>= 1;
    }
    r as u16
}

fn build_huff_lengths(counts: &[u64; 256]) -> [u8; 256] {
    let mut syms: Vec<(u64, usize)> = counts
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, c)| c > 0)
        .map(|(i, c)| (c, i))
        .collect();
    let mut lens = [0u8; 256];
    if syms.is_empty() {
        return lens;
    }
    if syms.len() == 1 {
        lens[syms[0].1] = 1;
        return lens;
    }
    syms.sort_by_key(|&(c, s)| (c, s));
    let m = syms.len();
    let mut nodes_w: Vec<u64> = syms.iter().map(|&(c, _)| c).collect();
    nodes_w.reserve(m);
    let mut parent = vec![usize::MAX; 2 * m];
    let (mut i, mut j) = (0usize, m);
    for _ in 0..(m - 1) {
        let mut pop_min = |nw: &[u64]| {
            if j >= nw.len() || (i < m && nw[i] <= nw[j]) {
                let x = i;
                i += 1;
                x
            } else {
                let x = j;
                j += 1;
                x
            }
        };
        let (a, b) = (pop_min(&nodes_w), pop_min(&nodes_w));
        let p = nodes_w.len();
        nodes_w.push(nodes_w[a] + nodes_w[b]);
        parent[a] = p;
        parent[b] = p;
    }
    let mut bl_count = [0i32; MAX_BITS + 1];
    let mut clamped = false;
    for (idx, &(_, sym)) in syms.iter().enumerate() {
        let mut d = 0usize;
        let mut cur = idx;
        while parent[cur] != usize::MAX {
            d += 1;
            cur = parent[cur];
        }
        if d > MAX_BITS {
            d = MAX_BITS;
            clamped = true;
        }
        lens[sym] = d as u8;
        bl_count[d] += 1;
    }
    if clamped {
        let mut kraft: i32 = (1..=MAX_BITS).map(|b| bl_count[b] << (MAX_BITS - b)).sum();
        while kraft > LUT_SIZE as i32 {
            let mut b = MAX_BITS - 1;
            while b > 0 && bl_count[b] == 0 {
                b -= 1;
            }
            bl_count[b] -= 1;
            bl_count[b + 1] += 2;
            bl_count[MAX_BITS] -= 1;
            kraft -= 1;
        }
        let mut ptr = 0usize;
        for bits in (1..=MAX_BITS).rev() {
            for _ in 0..(bl_count[bits] as usize) {
                lens[syms[ptr].1] = bits as u8;
                ptr += 1;
            }
        }
    }
    lens
}

pub(crate) fn huff_encode(data: &[u8], out: &mut Vec<u8>) {
    if data.is_empty() {
        out.push(0);
        return;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let active = counts.iter().filter(|&&c| c > 0).count();
    if active == 1 {
        let sym = counts.iter().position(|&c| c > 0).unwrap() as u8;
        out.push(1);
        out.push(sym);
        return;
    }
    let lens = build_huff_lengths(&counts);
    let mut bl = [0u32; MAX_BITS + 1];
    for &l in &lens {
        if l > 0 {
            bl[l as usize] += 1;
        }
    }
    let mut next_code = [0u32; MAX_BITS + 1];
    let mut code = 0u32;
    for bits in 1..=MAX_BITS {
        code = (code + bl[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = [0u16; 256];
    for s in 0..256 {
        let l = lens[s];
        if l > 0 {
            codes[s] = bit_rev(next_code[l as usize], l);
            next_code[l as usize] += 1;
        }
    }

    out.push(2);
    for pair in lens.chunks_exact(2) {
        out.push(pair[0] | (pair[1] << 4));
    }
    let len_pos = out.len();
    out.extend_from_slice(&0u32.to_le_bytes());
    let stream_start = out.len();

    let mut bitbuf = 0u64;
    let mut nbits = 0u32;
    for &b in data {
        let s = b as usize;
        bitbuf |= (codes[s] as u64) << nbits;
        nbits += lens[s] as u32;
        if nbits >= 32 {
            out.extend_from_slice(&(bitbuf as u32).to_le_bytes());
            bitbuf >>= 32;
            nbits -= 32;
        }
    }
    while nbits > 0 {
        out.push((bitbuf & 0xff) as u8);
        bitbuf >>= 8;
        nbits = nbits.saturating_sub(8);
    }
    let byte_len = (out.len() - stream_start) as u32;
    out[len_pos..len_pos + 4].copy_from_slice(&byte_len.to_le_bytes());
}

pub(crate) fn huff_decode(r: &mut Cur<'_>, out_len: usize) -> Result<Vec<u8>> {
    let mode = r.u8()?;
    if out_len == 0 || mode == 0 {
        return Ok(vec![0u8; out_len]);
    }
    if mode == 1 {
        let sym = r.u8()?;
        return Ok(vec![sym; out_len]);
    }
    if mode != 2 {
        return Err(Error::Format("invalid Huffman stream mode".into()));
    }
    let packed_lens = r.take(128)?;
    let mut lens = [0u8; 256];
    let mut bl = [0u32; MAX_BITS + 1];
    for (i, &p) in packed_lens.iter().enumerate() {
        for (k, l) in [p & 0x0f, p >> 4].into_iter().enumerate() {
            if l as usize > MAX_BITS {
                return Err(Error::Format("invalid Huffman code length".into()));
            }
            lens[i * 2 + k] = l;
            bl[l as usize] += (l > 0) as u32;
        }
    }
    let kraft: u32 = (1..=MAX_BITS).map(|b| bl[b] << (MAX_BITS - b)).sum();
    if kraft > LUT_SIZE as u32 {
        return Err(Error::Format("corrupt Huffman Kraft sum".into()));
    }
    let mut next_code = [0u32; MAX_BITS + 1];
    let mut code = 0u32;
    for bits in 1..=MAX_BITS {
        code = (code + bl[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut lut = vec![0u16; LUT_SIZE];
    for s in 0..256 {
        let l = lens[s];
        if l > 0 {
            let rev = bit_rev(next_code[l as usize], l) as usize;
            next_code[l as usize] += 1;
            let entry = (s as u16) | ((l as u16) << 8);
            for idx in (rev..LUT_SIZE).step_by(1usize << l) {
                lut[idx] = entry;
            }
        }
    }
    let byte_len = r.u32()? as usize;
    let bs = r.take(byte_len)?;
    let mut out = vec![0u8; out_len];
    let mut bitbuf = 0u64;
    let mut nbits = 0u32;
    let mut bpos = 0usize;

    for slot in &mut out {
        if nbits < MAX_BITS as u32 {
            while nbits <= 56 && bpos < bs.len() {
                bitbuf |= (bs[bpos] as u64) << nbits;
                nbits += 8;
                bpos += 1;
            }
        }
        let entry = lut[(bitbuf as usize) & (LUT_SIZE - 1)];
        let len = (entry >> 8) as u32;
        if len == 0 {
            return Err(Error::Format("invalid Huffman codeword".into()));
        }
        *slot = entry as u8;
        bitbuf >>= len;
        nbits = nbits.saturating_sub(len);
    }
    Ok(out)
}
