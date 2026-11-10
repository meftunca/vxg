use crate::columns::huf_encode_min;

use super::fse::fse_encode_min;

pub(crate) const DICT_LZ_MAGIC: u8 = 0x80;
pub(crate) const FLAG_HUF_LITS: u8 = 0x01;
pub(crate) const FLAG_HUF_TOKS: u8 = 0x02;
pub(crate) const FLAG_HUF_OFFS: u8 = 0x04;
pub(crate) const FLAG_HUF_EXT: u8 = 0x08;
pub(crate) const FLAG_FSE: u8 = 0x10;

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum StreamMode {
    Raw,
    Huf,
    Fse,
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
    pub(super) fse_lits: Vec<u8>,
    pub(super) fse_toks: Vec<u8>,
    pub(super) fse_offs: Vec<u8>,
    pub(super) fse_ext: Vec<u8>,
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
            fse_lits: Vec::new(),
            fse_toks: Vec::new(),
            fse_offs: Vec::new(),
            fse_ext: Vec::new(),
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

#[inline]
#[allow(clippy::too_many_arguments)]
fn pick_stream_mode(
    raw: &[u8],
    huf_buf: &mut Vec<u8>,
    fse_buf: &mut Vec<u8>,
    huf_min: usize,
    huf_margin: usize,
    try_fse: bool,
    fse_min: usize,
    fse_margin_floor: usize,
    fse_rel_shift: u32,
) -> (StreamMode, usize, usize) {
    let raw_len = raw.len();
    huf_buf.clear();
    let has_huf = raw_len >= huf_min
        && huf_encode_min(raw, huf_buf, raw_len.saturating_sub(huf_margin + 2), huf_min)
        && uleb_len(huf_buf.len()) + huf_buf.len() + huf_margin < raw_len;
    let non_fse_mode = if has_huf { StreamMode::Huf } else { StreamMode::Raw };
    let non_fse_wire = if has_huf {
        uleb_len(huf_buf.len()) + huf_buf.len()
    } else {
        raw_len
    };

    fse_buf.clear();
    let fse_margin = if fse_rel_shift == 0 {
        fse_margin_floor
    } else {
        (non_fse_wire >> fse_rel_shift).max(fse_margin_floor)
    };
    if try_fse && raw_len >= fse_min && non_fse_wire > fse_margin + 3 {
        let fse_cap = non_fse_wire - fse_margin - 2;
        if fse_encode_min(raw, fse_buf, fse_cap, fse_min) {
            let fw = uleb_len(fse_buf.len()) + fse_buf.len();
            if fw + fse_margin <= non_fse_wire {
                return (StreamMode::Fse, fw, non_fse_wire);
            }
        }
    }
    (non_fse_mode, non_fse_wire, non_fse_wire)
}

#[inline]
fn emit_chosen_stream(
    out: &mut Vec<u8>,
    mode: StreamMode,
    raw: &[u8],
    huf: &[u8],
    fse: &[u8],
) {
    match mode {
        StreamMode::Raw => out.extend_from_slice(raw),
        StreamMode::Huf => {
            put_uleb(out, huf.len());
            out.extend_from_slice(huf);
        }
        StreamMode::Fse => {
            put_uleb(out, fse.len());
            out.extend_from_slice(fse);
        }
    }
}

pub(super) fn serialize_streams(
    chunk: &[u8],
    anchor: usize,
    sb: &mut StreamBufs,
    out: &mut Vec<u8>,
    max_len: usize,
    level: u8,
) -> Option<(usize, usize)> {
    let n = chunk.len();
    let seq_count = sb.toks.len();
    let out_start = out.len();

    if seq_count == 0 {
        sb.huf_lits.clear();
        sb.fse_lits.clear();
        let hdr_budget = 1 + 1 + uleb_len(n) + 4;
        if max_len <= hdr_budget {
            return None;
        }
        let has_huf = huf_encode_min(chunk, &mut sb.huf_lits, max_len - hdr_budget, 32);
        let huf_est = if has_huf {
            1 + 1 + uleb_len(n) + uleb_len(sb.huf_lits.len()) + sb.huf_lits.len()
        } else {
            usize::MAX
        };
        let fse_cap = huf_est.min(max_len).saturating_sub(hdr_budget + 2);
        let has_fse = level >= 2
            && fse_cap >= 2
            && fse_encode_min(chunk, &mut sb.fse_lits, fse_cap, 32);
        if has_fse {
            let est = 1 + 1 + 1 + uleb_len(n) + uleb_len(sb.fse_lits.len()) + sb.fse_lits.len();
            if est < huf_est && est < max_len {
                out.reserve(est);
                out.push(DICT_LZ_MAGIC | FLAG_FSE);
                out.push(FLAG_HUF_LITS);
                put_uleb(out, 0);
                put_uleb(out, n);
                put_uleb(out, sb.fse_lits.len());
                out.extend_from_slice(&sb.fse_lits);
                return Some((out_start, out.len() - out_start));
            }
        }
        if huf_est >= max_len {
            return None;
        }
        out.reserve(huf_est);
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
    let (fse_margin, fse_rel_shift, min_block_fse) = if level >= 6 {
        (1usize, 0u32, 2usize)
    } else if level == 5 {
        (2usize, 7u32, 4usize)
    } else {
        (4usize, 6u32, 8usize)
    };
    let try_fse_seq = level >= 2;
    let try_fse_lits = level >= 5;

    let (mut m_lits, mut w_lits, nf_lits) = pick_stream_mode(
        &sb.lits,
        &mut sb.huf_lits,
        &mut sb.fse_lits,
        96,
        8,
        try_fse_lits,
        96,
        fse_margin + 2,
        fse_rel_shift,
    );
    let (mut m_toks, mut w_toks, nf_toks) = pick_stream_mode(
        &sb.toks,
        &mut sb.huf_toks,
        &mut sb.fse_toks,
        96,
        8,
        try_fse_seq,
        48,
        fse_margin,
        fse_rel_shift,
    );
    let (mut m_offs, mut w_offs, nf_offs) = pick_stream_mode(
        &sb.off_codes,
        &mut sb.huf_offs,
        &mut sb.fse_offs,
        96,
        8,
        try_fse_seq,
        32,
        fse_margin,
        fse_rel_shift,
    );
    let (mut m_ext, mut w_ext, nf_ext) = pick_stream_mode(
        &sb.len_ext,
        &mut sb.huf_ext,
        &mut sb.fse_ext,
        256,
        12,
        try_fse_seq,
        32,
        fse_margin,
        fse_rel_shift,
    );

    let fse_saved =
        (nf_lits - w_lits) + (nf_toks - w_toks) + (nf_offs - w_offs) + (nf_ext - w_ext);
    if fse_saved < min_block_fse {
        if m_lits == StreamMode::Fse {
            m_lits = if !sb.huf_lits.is_empty() && nf_lits < lit_total {
                StreamMode::Huf
            } else {
                StreamMode::Raw
            };
            w_lits = nf_lits;
        }
        if m_toks == StreamMode::Fse {
            m_toks = if !sb.huf_toks.is_empty() && nf_toks < seq_count {
                StreamMode::Huf
            } else {
                StreamMode::Raw
            };
            w_toks = nf_toks;
        }
        if m_offs == StreamMode::Fse {
            m_offs = if !sb.huf_offs.is_empty() && nf_offs < seq_count {
                StreamMode::Huf
            } else {
                StreamMode::Raw
            };
            w_offs = nf_offs;
        }
        if m_ext == StreamMode::Fse {
            m_ext = if !sb.huf_ext.is_empty() && nf_ext < ext_raw_len {
                StreamMode::Huf
            } else {
                StreamMode::Raw
            };
            w_ext = nf_ext;
        }
    }

    let mut flags = DICT_LZ_MAGIC;
    let mut fse_mask = 0u8;
    for (mode, bit) in [
        (m_lits, FLAG_HUF_LITS),
        (m_toks, FLAG_HUF_TOKS),
        (m_offs, FLAG_HUF_OFFS),
        (m_ext, FLAG_HUF_EXT),
    ] {
        match mode {
            StreamMode::Huf => flags |= bit,
            StreamMode::Fse => fse_mask |= bit,
            StreamMode::Raw => {}
        }
    }
    if fse_mask != 0 {
        flags |= FLAG_FSE;
    }

    let hdr_bytes = 1 + usize::from(fse_mask != 0);
    let ext_wire = uleb_len(ext_raw_len) + w_ext;
    let bits_wire = uleb_len(sb.off_bits.len()) + sb.off_bits.len();
    let total_wire = hdr_bytes
        + uleb_len(seq_count)
        + uleb_len(lit_total)
        + w_lits
        + w_toks
        + w_offs
        + ext_wire
        + bits_wire;

    if total_wire >= max_len {
        return None;
    }

    out.reserve(total_wire);
    out.push(flags);
    if fse_mask != 0 {
        out.push(fse_mask);
    }
    put_uleb(out, seq_count);
    put_uleb(out, lit_total);

    emit_chosen_stream(out, m_lits, &sb.lits, &sb.huf_lits, &sb.fse_lits);
    emit_chosen_stream(out, m_toks, &sb.toks, &sb.huf_toks, &sb.fse_toks);
    emit_chosen_stream(out, m_offs, &sb.off_codes, &sb.huf_offs, &sb.fse_offs);
    put_uleb(out, ext_raw_len);
    emit_chosen_stream(out, m_ext, &sb.len_ext, &sb.huf_ext, &sb.fse_ext);
    put_uleb(out, sb.off_bits.len());
    out.extend_from_slice(&sb.off_bits);

    debug_assert_eq!(out.len() - out_start, total_wire);
    Some((out_start, total_wire))
}
