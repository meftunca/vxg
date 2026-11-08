use crate::columns::huf_encode_min;

pub(crate) const DICT_LZ_MAGIC: u8 = 0x80;
pub(crate) const FLAG_HUF_LITS: u8 = 0x01;
pub(crate) const FLAG_HUF_TOKS: u8 = 0x02;
pub(crate) const FLAG_HUF_OFFS: u8 = 0x04;
pub(crate) const FLAG_HUF_EXT: u8 = 0x08;

#[inline(always)]
pub(crate) fn put_uleb(out: &mut Vec<u8>, mut v: usize) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

#[inline(always)]
pub(crate) fn uleb_len(mut v: usize) -> usize {
    let mut n = 1usize;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

#[inline(always)]
pub(super) fn push_ext_len(ext: &mut Vec<u8>, rem: usize) {
    if rem < 255 {
        ext.push(rem as u8);
    } else {
        ext.push(255);
        put_uleb(ext, rem - 255);
    }
}

pub(super) struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    nb: u32,
}

impl<'a> BitWriter<'a> {
    #[inline(always)]
    pub(super) fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            nb: 0,
        }
    }

    #[inline(always)]
    pub(super) fn put(&mut self, val: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        self.acc |= u64::from(val) << self.nb;
        self.nb += bits;
        if self.nb >= 32 {
            self.out.extend_from_slice(&(self.acc as u32).to_le_bytes());
            self.acc >>= 32;
            self.nb -= 32;
        }
    }

    #[inline(always)]
    pub(super) fn finish(self) {
        let mut acc = self.acc;
        let mut nb = self.nb;
        while nb >= 8 {
            self.out.push(acc as u8);
            acc >>= 8;
            nb -= 8;
        }
        if nb > 0 {
            self.out.push(acc as u8);
        }
    }
}

pub(super) struct StreamBufs {
    pub(super) lits: Vec<u8>,
    pub(super) toks: Vec<u8>,
    pub(super) off_codes: Vec<u8>,
    pub(super) len_ext: Vec<u8>,
    pub(super) off_bits: Vec<u8>,
    pub(super) huf_lits: Vec<u8>,
    pub(super) huf_toks: Vec<u8>,
    pub(super) huf_offs: Vec<u8>,
    pub(super) huf_ext: Vec<u8>,
}

impl StreamBufs {
    pub(super) fn new() -> Self {
        Self {
            lits: Vec::new(),
            toks: Vec::new(),
            off_codes: Vec::new(),
            len_ext: Vec::new(),
            off_bits: Vec::new(),
            huf_lits: Vec::new(),
            huf_toks: Vec::new(),
            huf_offs: Vec::new(),
            huf_ext: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn reset_for(&mut self, n: usize) {
        self.lits.clear();
        self.toks.clear();
        self.off_codes.clear();
        self.len_ext.clear();
        self.off_bits.clear();
        self.lits.reserve(n / 2);
        self.toks.reserve(n / 8);
        self.off_codes.reserve(n / 8);
    }
}

pub(super) fn serialize_streams(
    chunk: &[u8],
    anchor: usize,
    sb: &mut StreamBufs,
    out: &mut Vec<u8>,
    max_len: usize,
) -> Option<(usize, usize)> {
    let n = chunk.len();
    let seq_count = sb.toks.len();
    let out_start = out.len();

    if seq_count == 0 {
        sb.huf_lits.clear();
        let hdr_budget = 1 + 1 + uleb_len(n) + 4;
        if max_len <= hdr_budget {
            return None;
        }
        if !huf_encode_min(chunk, &mut sb.huf_lits, max_len - hdr_budget, 32) {
            return None;
        }
        let est = 1 + 1 + uleb_len(n) + uleb_len(sb.huf_lits.len()) + sb.huf_lits.len();
        if est >= max_len {
            return None;
        }
        out.reserve(est);
        out.push(DICT_LZ_MAGIC | FLAG_HUF_LITS);
        put_uleb(out, 0);
        put_uleb(out, n);
        put_uleb(out, sb.huf_lits.len());
        out.extend_from_slice(&sb.huf_lits);
        return Some((out_start, out.len() - out_start));
    }

    if anchor < n {
        sb.lits.extend_from_slice(&chunk[anchor..n]);
    }
    let lit_total = sb.lits.len();
    let ext_raw_len = sb.len_ext.len();

    let mut flags = DICT_LZ_MAGIC;

    sb.huf_lits.clear();
    let use_huf_lits = lit_total >= 32
        && huf_encode_min(&sb.lits, &mut sb.huf_lits, lit_total.saturating_sub(2), 32)
        && uleb_len(sb.huf_lits.len()) + sb.huf_lits.len() < lit_total;
    if use_huf_lits {
        flags |= FLAG_HUF_LITS;
    }

    sb.huf_toks.clear();
    let use_huf_toks = seq_count >= 32
        && huf_encode_min(&sb.toks, &mut sb.huf_toks, seq_count.saturating_sub(2), 32)
        && uleb_len(sb.huf_toks.len()) + sb.huf_toks.len() < seq_count;
    if use_huf_toks {
        flags |= FLAG_HUF_TOKS;
    }

    sb.huf_offs.clear();
    let use_huf_offs = seq_count >= 32
        && huf_encode_min(
            &sb.off_codes,
            &mut sb.huf_offs,
            seq_count.saturating_sub(2),
            32,
        )
        && uleb_len(sb.huf_offs.len()) + sb.huf_offs.len() < seq_count;
    if use_huf_offs {
        flags |= FLAG_HUF_OFFS;
    }

    sb.huf_ext.clear();
    let use_huf_ext = ext_raw_len >= 48
        && huf_encode_min(
            &sb.len_ext,
            &mut sb.huf_ext,
            ext_raw_len.saturating_sub(2),
            48,
        )
        && uleb_len(sb.huf_ext.len()) + sb.huf_ext.len() < ext_raw_len;
    if use_huf_ext {
        flags |= FLAG_HUF_EXT;
    }

    let lits_wire = if use_huf_lits {
        uleb_len(sb.huf_lits.len()) + sb.huf_lits.len()
    } else {
        lit_total
    };
    let toks_wire = if use_huf_toks {
        uleb_len(sb.huf_toks.len()) + sb.huf_toks.len()
    } else {
        seq_count
    };
    let offs_wire = if use_huf_offs {
        uleb_len(sb.huf_offs.len()) + sb.huf_offs.len()
    } else {
        seq_count
    };
    let ext_wire = uleb_len(ext_raw_len)
        + if use_huf_ext {
            uleb_len(sb.huf_ext.len()) + sb.huf_ext.len()
        } else {
            ext_raw_len
        };
    let bits_wire = uleb_len(sb.off_bits.len()) + sb.off_bits.len();
    let total_wire = 1
        + uleb_len(seq_count)
        + uleb_len(lit_total)
        + lits_wire
        + toks_wire
        + offs_wire
        + ext_wire
        + bits_wire;

    if total_wire >= max_len {
        return None;
    }

    out.reserve(total_wire);
    out.push(flags);
    put_uleb(out, seq_count);
    put_uleb(out, lit_total);

    if use_huf_lits {
        put_uleb(out, sb.huf_lits.len());
        out.extend_from_slice(&sb.huf_lits);
    } else {
        out.extend_from_slice(&sb.lits);
    }
    if use_huf_toks {
        put_uleb(out, sb.huf_toks.len());
        out.extend_from_slice(&sb.huf_toks);
    } else {
        out.extend_from_slice(&sb.toks);
    }
    if use_huf_offs {
        put_uleb(out, sb.huf_offs.len());
        out.extend_from_slice(&sb.huf_offs);
    } else {
        out.extend_from_slice(&sb.off_codes);
    }
    put_uleb(out, ext_raw_len);
    if use_huf_ext {
        put_uleb(out, sb.huf_ext.len());
        out.extend_from_slice(&sb.huf_ext);
    } else {
        out.extend_from_slice(&sb.len_ext);
    }
    put_uleb(out, sb.off_bits.len());
    out.extend_from_slice(&sb.off_bits);

    debug_assert_eq!(out.len() - out_start, total_wire);
    Some((out_start, total_wire))
}
