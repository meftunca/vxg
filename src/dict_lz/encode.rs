use super::{
    dict_tag4, dict_tag6, mul4, mul6,
    pack::{push_ext_len, serialize_streams, BitWriter, StreamBufs},
    read_u32, read_u64, PreparedDict, DICT_HT4_BITS, DICT_POS_MASK, LOCAL_HT4_BITS,
    LOCAL_HT4_SIZE, LOCAL_HT6_BITS, LOCAL_HT6_SIZE,
};

const MAX_OFFSET: usize = (1 << 21) - 1;
const MASK6: u64 = 0x0000_ffff_ffff_ffff;
const LOCAL_HT4_SHIFT: u32 = (32 - LOCAL_HT4_BITS) as u32;
const LOCAL_HT6_SHIFT: u32 = (64 - LOCAL_HT6_BITS) as u32;
const DICT_HT4_SHIFT: u32 = (32 - DICT_HT4_BITS) as u32;

pub(crate) struct DictLzScratch {
    local_ht4: Box<[[u16; 2]; LOCAL_HT4_SIZE]>,
    local_ht6: Box<[[u16; 2]; LOCAL_HT6_SIZE]>,
    sb: StreamBufs,
}

impl DictLzScratch {
    pub(crate) fn new() -> Self {
        Self {
            local_ht4: Box::new([[0u16; 2]; LOCAL_HT4_SIZE]),
            local_ht6: Box::new([[0u16; 2]; LOCAL_HT6_SIZE]),
            sb: StreamBufs::new(),
        }
    }
}

#[inline(always)]
fn match_len_slices(src: &[u8], mut a: usize, dst: &[u8], mut b: usize, limit: usize) -> usize {
    let start = b;
    let end = b + limit;
    while b + 8 <= end {
        let diff = read_u64(src, a) ^ read_u64(dst, b);
        if diff != 0 {
            return (b - start) + (diff.trailing_zeros() as usize >> 3);
        }
        a += 8;
        b += 8;
    }
    while b < end && unsafe { *src.get_unchecked(a) == *dst.get_unchecked(b) } {
        a += 1;
        b += 1;
    }
    b - start
}

#[inline(always)]
fn insert_local(
    chunk: &[u8],
    pos: usize,
    local_ht4: &mut [[u16; 2]; LOCAL_HT4_SIZE],
    local_ht6: &mut [[u16; 2]; LOCAL_HT6_SIZE],
) {
    let w8 = read_u64(chunk, pos);
    let p16 = pos as u16;
    unsafe {
        let b6 = local_ht6.get_unchecked_mut((mul6(w8) >> LOCAL_HT6_SHIFT) as usize);
        *b6 = [p16, b6[0]];
        let b4 = local_ht4.get_unchecked_mut((mul4(w8 as u32) >> LOCAL_HT4_SHIFT) as usize);
        *b4 = [p16, b4[0]];
    }
}

#[inline(always)]
fn resolve_cand<const SMALL: bool>(pos: usize, raw_u16: u16) -> Option<usize> {
    if SMALL {
        Some(raw_u16 as usize)
    } else {
        let mut cand = (pos & !0xffff) | (raw_u16 as usize);
        if cand >= pos {
            if pos >= 65536 {
                cand -= 65536;
            } else {
                return None;
            }
        }
        if pos - cand <= MAX_OFFSET {
            Some(cand)
        } else {
            None
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn emit_sequence(
    chunk: &[u8],
    anchor: usize,
    pos: usize,
    ml: usize,
    offset: usize,
    rep_idx: u8,
    rep: &mut [usize; 3],
    lits: &mut Vec<u8>,
    toks: &mut Vec<u8>,
    off_codes: &mut Vec<u8>,
    len_ext: &mut Vec<u8>,
    bw: &mut BitWriter<'_>,
) {
    let lit_len = pos - anchor;
    if lit_len > 0 {
        lits.extend_from_slice(&chunk[anchor..pos]);
    }
    let ll_code = lit_len.min(15);
    let ml_sub = ml - 3;
    let ml_code = ml_sub.min(15);
    toks.push((ll_code as u8) | ((ml_code as u8) << 4));
    if ll_code == 15 {
        push_ext_len(len_ext, lit_len - 15);
    }
    if ml_code == 15 {
        push_ext_len(len_ext, ml_sub - 15);
    }
    match rep_idx {
        0 => off_codes.push(0),
        1 => {
            off_codes.push(1);
            rep.swap(0, 1);
        }
        2 => {
            off_codes.push(2);
            let r2 = rep[2];
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = r2;
        }
        _ => {
            let bits = 31 - (offset as u32).leading_zeros();
            off_codes.push((3 + bits) as u8);
            if bits > 0 {
                bw.put((offset as u32) & ((1u32 << bits) - 1), bits);
            }
            rep[2] = rep[1];
            rep[1] = rep[0];
            rep[0] = offset;
        }
    }
}

#[inline(always)]
fn off_pen(off: usize) -> i32 {
    if off <= 256 {
        0
    } else if off <= 4096 {
        2
    } else if off <= 32768 {
        4
    } else {
        6
    }
}

#[inline(always)]
fn probe_dict6(
    d: &PreparedDict,
    chunk: &[u8],
    pos: usize,
    rem: usize,
    w6: u64,
    m6: u64,
    mut best_len: usize,
    mut best_off: usize,
    mut best_sc: i32,
) -> (usize, usize, i32) {
    let d_len = d.raw.len();
    let t6 = dict_tag6(m6);
    let b6 = unsafe { *d.ht6.get_unchecked((m6 >> d.ht6_shift) as usize) };
    let e0 = b6[0];
    if e0 != 0 {
        if (e0 ^ t6) <= DICT_POS_MASK {
            let d_cand = ((e0 & DICT_POS_MASK) - 1) as usize;
            let dict_back = d_len - d_cand;
            let off = pos + dict_back;
            let d_lim = rem.min(dict_back);
            if off <= MAX_OFFSET
                && d_lim >= 6
                && d_cand + 8 <= d_len
                && (read_u64(&d.raw, d_cand) & MASK6) == w6
            {
                let ml = 6 + match_len_slices(&d.raw, d_cand + 6, chunk, pos + 6, d_lim - 6);
                let sc = (ml as i32) * 4 - off_pen(off);
                if sc > best_sc {
                    best_len = ml;
                    best_off = off;
                    best_sc = sc;
                    if best_len >= 256 {
                        return (best_len, best_off, best_sc);
                    }
                }
            }
        }
        let e1 = b6[1];
        if e1 != 0 && (e1 ^ t6) <= DICT_POS_MASK {
            let d_cand = ((e1 & DICT_POS_MASK) - 1) as usize;
            let dict_back = d_len - d_cand;
            let off = pos + dict_back;
            let d_lim = rem.min(dict_back);
            if off <= MAX_OFFSET
                && d_lim >= 6
                && d_cand + 8 <= d_len
                && (read_u64(&d.raw, d_cand) & MASK6) == w6
            {
                let ml = 6 + match_len_slices(&d.raw, d_cand + 6, chunk, pos + 6, d_lim - 6);
                let sc = (ml as i32) * 4 - off_pen(off);
                if sc > best_sc {
                    best_len = ml;
                    best_off = off;
                    best_sc = sc;
                }
            }
        }
    }
    (best_len, best_off, best_sc)
}

#[inline(always)]
fn probe_dict4(
    d: &PreparedDict,
    chunk: &[u8],
    pos: usize,
    rem: usize,
    w4: u32,
    p4: u32,
    mut best_len: usize,
    mut best_off: usize,
    mut best_sc: i32,
) -> (usize, usize, i32) {
    let d_len = d.raw.len();
    let t4 = dict_tag4(w4);
    let b4 = unsafe { *d.ht4.get_unchecked((p4 >> DICT_HT4_SHIFT) as usize) };
    let e0 = b4[0];
    if e0 != 0 {
        if (e0 ^ t4) <= DICT_POS_MASK {
            let d_cand = ((e0 & DICT_POS_MASK) - 1) as usize;
            let dict_back = d_len - d_cand;
            let off = pos + dict_back;
            let d_lim = rem.min(dict_back);
            if off <= MAX_OFFSET
                && d_lim >= 4
                && d_cand + 4 <= d_len
                && read_u32(&d.raw, d_cand) == w4
            {
                let ml = 4 + match_len_slices(&d.raw, d_cand + 4, chunk, pos + 4, d_lim - 4);
                if ml >= 5 || off <= 16384 {
                    let sc = (ml as i32) * 4 - off_pen(off);
                    if sc > best_sc {
                        best_len = ml;
                        best_off = off;
                        best_sc = sc;
                    }
                }
            }
        }
        let e1 = b4[1];
        if e1 != 0 && best_len < 6 && (e1 ^ t4) <= DICT_POS_MASK {
            let d_cand = ((e1 & DICT_POS_MASK) - 1) as usize;
            let dict_back = d_len - d_cand;
            let off = pos + dict_back;
            let d_lim = rem.min(dict_back);
            if off <= MAX_OFFSET
                && d_lim >= 4
                && d_cand + 4 <= d_len
                && read_u32(&d.raw, d_cand) == w4
            {
                let ml = 4 + match_len_slices(&d.raw, d_cand + 4, chunk, pos + 4, d_lim - 4);
                if ml >= 5 || off <= 16384 {
                    let sc = (ml as i32) * 4 - off_pen(off);
                    if sc > best_sc {
                        best_len = ml;
                        best_off = off;
                        best_sc = sc;
                    }
                }
            }
        }
    }
    (best_len, best_off, best_sc)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn probe_pos<const SMALL: bool, const NEED_HT4: bool>(
    chunk: &[u8],
    n: usize,
    pos: usize,
    anchor: usize,
    r0: usize,
    dict: Option<&PreparedDict>,
    local_ht4: &[[u16; 2]; LOCAL_HT4_SIZE],
    local_ht6: &[[u16; 2]; LOCAL_HT6_SIZE],
    w8: u64,
    m6: u64,
    p4: u32,
) -> (usize, usize, u8, i32) {
    let rem = n - pos;
    let w6 = w8 & MASK6;
    let w4 = w8 as u32;
    let mut ml = 0usize;
    let mut off = 0usize;
    let mut rep_idx = 255u8;
    let mut best_sc = 0i32;

    if pos > anchor && r0 <= pos && (read_u32(chunk, pos - r0) & 0x00ff_ffff) == (w4 & 0x00ff_ffff)
    {
        ml = 3 + match_len_slices(chunk, pos - r0 + 3, chunk, pos + 3, rem - 3);
        off = r0;
        rep_idx = 0;
        best_sc = (ml as i32) * 4 + 3;
        if ml >= 48 {
            return (ml, off, rep_idx, best_sc);
        }
    }

    if pos > 0 {
        let b6 = unsafe { *local_ht6.get_unchecked((m6 >> LOCAL_HT6_SHIFT) as usize) };
        for raw6 in b6 {
            if let Some(cand6) = resolve_cand::<SMALL>(pos, raw6) {
                if (read_u64(chunk, cand6) & MASK6) == w6 {
                    let c_off = pos - cand6;
                    let c_ml = 6 + match_len_slices(chunk, cand6 + 6, chunk, pos + 6, rem - 6);
                    let sc = (c_ml as i32) * 4 - off_pen(c_off);
                    if sc > best_sc {
                        ml = c_ml;
                        off = c_off;
                        rep_idx = 255;
                        best_sc = sc;
                        if ml >= 48 {
                            return (ml, off, rep_idx, best_sc);
                        }
                    }
                }
            }
            if raw6 == 0 {
                break;
            }
        }
    }

    if ml < 32 {
        if let Some(d) = dict {
            let (d_ml, d_off, d_sc) = probe_dict6(d, chunk, pos, rem, w6, m6, ml, off, best_sc);
            if d_sc > best_sc {
                ml = d_ml;
                off = d_off;
                rep_idx = 255;
                best_sc = d_sc;
                if ml >= 48 {
                    return (ml, off, rep_idx, best_sc);
                }
            }
        }
    }

    if NEED_HT4 && ml < 6 {
        if pos > 0 {
            let b4 = unsafe { *local_ht4.get_unchecked((p4 >> LOCAL_HT4_SHIFT) as usize) };
            for raw4 in b4 {
                if let Some(cand4) = resolve_cand::<SMALL>(pos, raw4) {
                    if read_u32(chunk, cand4) == w4 {
                        let c_off = pos - cand4;
                        let c_ml = 4 + match_len_slices(chunk, cand4 + 4, chunk, pos + 4, rem - 4);
                        if c_ml >= 5 || c_off <= 16384 {
                            let sc = (c_ml as i32) * 4 - off_pen(c_off);
                            if sc > best_sc {
                                ml = c_ml;
                                off = c_off;
                                rep_idx = 255;
                                best_sc = sc;
                            }
                        }
                    }
                }
                if raw4 == 0 {
                    break;
                }
            }
        }
        if ml < 6 {
            if let Some(d) = dict {
                let (d_ml, d_off, d_sc) = probe_dict4(d, chunk, pos, rem, w4, p4, ml, off, best_sc);
                if d_sc > best_sc {
                    ml = d_ml;
                    off = d_off;
                    rep_idx = 255;
                    best_sc = d_sc;
                }
            }
        }
    }

    (ml, off, rep_idx, best_sc)
}

fn encode_loop<const SMALL: bool>(
    chunk: &[u8],
    dict: Option<&PreparedDict>,
    sc: &mut DictLzScratch,
) -> usize {
    let n = chunk.len();
    sc.local_ht4.fill([0u16; 2]);
    sc.local_ht6.fill([0u16; 2]);
    let d_slice: &[u8] = dict.map_or(&[], |d| d.raw.as_slice());
    let d_len = d_slice.len();
    let limit = n.saturating_sub(8);

    sc.sb.reset_for(n);
    insert_local(chunk, 0, &mut sc.local_ht4, &mut sc.local_ht6);

    let mut rep = [1usize, 4, 8];
    let mut pos = if dict.is_some() { 0usize } else { 1usize };
    let mut anchor = 0usize;
    let mut misses = 0usize;

    let StreamBufs {
        lits,
        toks,
        off_codes,
        len_ext,
        off_bits,
        ..
    } = &mut sc.sb;
    let mut bw = BitWriter::new(off_bits);

    while pos <= limit {
        let w8 = read_u64(chunk, pos);
        let m6 = mul6(w8);
        let p4 = mul4(w8 as u32);

        let (mut ml, mut off, mut rep_idx, best_sc) = probe_pos::<SMALL, true>(
            chunk,
            n,
            pos,
            anchor,
            rep[0],
            dict,
            &sc.local_ht4,
            &sc.local_ht6,
            w8,
            m6,
            p4,
        );

        if pos > 0 {
            let p16 = pos as u16;
            unsafe {
                let b6 = sc
                    .local_ht6
                    .get_unchecked_mut((m6 >> LOCAL_HT6_SHIFT) as usize);
                *b6 = [p16, b6[0]];
                let b4 = sc
                    .local_ht4
                    .get_unchecked_mut((p4 >> LOCAL_HT4_SHIFT) as usize);
                *b4 = [p16, b4[0]];
            }
        }

        if ml == 0 {
            misses += 1;
            let shift = if pos >= 2048 && lits.len() + (pos - anchor) > (pos * 3) / 4 {
                3
            } else {
                4
            };
            pos += 1 + (misses >> shift);
            continue;
        }
        misses = 0;

        // Surgical 1-step lazy check on short/medium matches (`ml <= 18`); skips `ht4` when `ml >= 5`!
        let noisy = pos >= 2048 && lits.len() + (pos - anchor) > (pos * 3) / 4;
        if !noisy && ml <= 18 && pos + 1 <= limit {
            let p1 = pos + 1;
            let wn = read_u64(chunk, p1);
            let m6n = mul6(wn);
            let p4n = mul4(wn as u32);
            let (n_ml, n_off, n_rep, n_sc) = if ml == 4 {
                probe_pos::<SMALL, true>(
                    chunk, n, p1, anchor, rep[0], dict, &sc.local_ht4, &sc.local_ht6, wn, m6n, p4n,
                )
            } else {
                probe_pos::<SMALL, false>(
                    chunk, n, p1, anchor, rep[0], dict, &sc.local_ht4, &sc.local_ht6, wn, m6n, p4n,
                )
            };
            if n_ml > ml && n_sc > best_sc + 3 {
                insert_local(chunk, p1, &mut sc.local_ht4, &mut sc.local_ht6);
                pos = p1;
                ml = n_ml;
                off = n_off;
                rep_idx = n_rep;
            }
        }

        // Backward match extension across skipped literal bytes.
        if off <= pos {
            while pos > anchor
                && pos > off
                && unsafe { *chunk.get_unchecked(pos - 1) == *chunk.get_unchecked(pos - 1 - off) }
            {
                pos -= 1;
                ml += 1;
            }
        } else if d_len + pos > off {
            let mut d_src = d_len + pos - off;
            while pos > anchor
                && d_src > 0
                && unsafe { *chunk.get_unchecked(pos - 1) == *d_slice.get_unchecked(d_src - 1) }
            {
                pos -= 1;
                d_src -= 1;
                ml += 1;
            }
        }

        if rep_idx == 255 {
            rep_idx = if off == rep[0] && pos > anchor {
                0
            } else if off == rep[1] {
                1
            } else if off == rep[2] {
                2
            } else {
                255
            };
        }

        emit_sequence(
            chunk, anchor, pos, ml, off, rep_idx, &mut rep, lits, toks, off_codes, len_ext, &mut bw,
        );

        let match_end = pos + ml;
        let idx_lim = match_end.min(limit + 1);
        if ml <= 12 {
            let mut q = pos + 1;
            while q < idx_lim {
                insert_local(chunk, q, &mut sc.local_ht4, &mut sc.local_ht6);
                q += 1;
            }
        } else if ml <= 32 {
            let mut q = pos + 2;
            while q < idx_lim {
                insert_local(chunk, q, &mut sc.local_ht4, &mut sc.local_ht6);
                q += 2;
            }
            if idx_lim > pos + 1 {
                insert_local(chunk, idx_lim - 1, &mut sc.local_ht4, &mut sc.local_ht6);
            }
        } else if idx_lim > pos + 3 {
            insert_local(chunk, pos + 2, &mut sc.local_ht4, &mut sc.local_ht6);
            insert_local(chunk, idx_lim - 2, &mut sc.local_ht4, &mut sc.local_ht6);
            insert_local(chunk, idx_lim - 1, &mut sc.local_ht4, &mut sc.local_ht6);
        }

        pos = match_end;
        anchor = pos;
    }
    bw.finish();
    anchor
}

pub(super) fn encode_into(
    chunk: &[u8],
    dict: Option<&PreparedDict>,
    sc: &mut DictLzScratch,
    out: &mut Vec<u8>,
    max_len: usize,
) -> Option<(usize, usize)> {
    let anchor = if chunk.len() <= 65536 {
        encode_loop::<true>(chunk, dict, sc)
    } else {
        encode_loop::<false>(chunk, dict, sc)
    };
    serialize_streams(chunk, anchor, &mut sc.sb, out, max_len)
}
