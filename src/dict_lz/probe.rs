use super::{
    dict_tag4, dict_tag6, read_u32, read_u64, PreparedDict, DICT_HT4_BITS, DICT_POS_MASK,
    LOCAL_HT4_BITS, LOCAL_HT4_SIZE, LOCAL_HT6_BITS, LOCAL_HT6_SIZE,
};

pub(super) const MAX_OFFSET: usize = (1 << 21) - 1;
pub(super) const MASK6: u64 = 0x0000_ffff_ffff_ffff;
pub(super) const LOCAL_HT4_SHIFT: u32 = (32 - LOCAL_HT4_BITS) as u32;
pub(super) const LOCAL_HT6_SHIFT: u32 = (64 - LOCAL_HT6_BITS) as u32;
const DICT_HT4_SHIFT: u32 = (32 - DICT_HT4_BITS) as u32;

#[inline(always)]
pub(super) fn match_len_slices(
    src: &[u8],
    mut a: usize,
    dst: &[u8],
    mut b: usize,
    limit: usize,
) -> usize {
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
#[allow(clippy::too_many_arguments)]
fn probe_dict6<const DWAYS: usize>(
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
    for &entry in b6.iter().take(DWAYS) {
        if entry == 0 {
            break;
        }
        if (entry ^ t6) <= DICT_POS_MASK {
            let d_cand = ((entry & DICT_POS_MASK) - 1) as usize;
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
    }
    (best_len, best_off, best_sc)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn probe_dict4<const DWAYS: usize>(
    d: &PreparedDict,
    chunk: &[u8],
    pos: usize,
    rem: usize,
    w4: u32,
    p4: u32,
    max_off4: usize,
    mut best_len: usize,
    mut best_off: usize,
    mut best_sc: i32,
) -> (usize, usize, i32) {
    let d_len = d.raw.len();
    let t4 = dict_tag4(w4);
    let b4 = unsafe { *d.ht4.get_unchecked((p4 >> DICT_HT4_SHIFT) as usize) };
    for &entry in b4.iter().take(DWAYS) {
        if entry == 0 || best_len >= 6 {
            break;
        }
        if (entry ^ t4) <= DICT_POS_MASK {
            let d_cand = ((entry & DICT_POS_MASK) - 1) as usize;
            let dict_back = d_len - d_cand;
            let off = pos + dict_back;
            let d_lim = rem.min(dict_back);
            if off <= MAX_OFFSET
                && d_lim >= 4
                && d_cand + 4 <= d_len
                && read_u32(&d.raw, d_cand) == w4
            {
                let ml = 4 + match_len_slices(&d.raw, d_cand + 4, chunk, pos + 4, d_lim - 4);
                if ml >= 5 || off <= max_off4 {
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
fn eq3_at(buf: &[u8], idx: usize, w4_low24: u32) -> bool {
    if idx + 4 <= buf.len() {
        (read_u32(buf, idx) & 0x00ff_ffff) == w4_low24
    } else {
        unsafe {
            let b0 = *buf.get_unchecked(idx) as u32;
            let b1 = (*buf.get_unchecked(idx + 1) as u32) << 8;
            let b2 = (*buf.get_unchecked(idx + 2) as u32) << 16;
            (b0 | b1 | b2) == w4_low24
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn probe_pos<
    const SMALL: bool,
    const WAYS: usize,
    const DWAYS: usize,
    const NEED_HT4: bool,
>(
    chunk: &[u8],
    n: usize,
    pos: usize,
    anchor: usize,
    rep: &[usize; 3],
    dict: Option<&PreparedDict>,
    local_ht4: &[[u16; WAYS]; LOCAL_HT4_SIZE],
    local_ht6: &[[u16; WAYS]; LOCAL_HT6_SIZE],
    w8: u64,
    m6: u64,
    p4: u32,
) -> (usize, usize, u8, i32) {
    let rem = n - pos;
    let w6 = w8 & MASK6;
    let w4 = w8 as u32;
    let w3 = w4 & 0x00ff_ffff;
    let mut ml = 0usize;
    let mut off = 0usize;
    let mut rep_idx = 255u8;
    let mut best_sc = 0i32;

    let d_slice: &[u8] = dict.map_or(&[], |d| d.raw.as_slice());
    let d_len = d_slice.len();
    let rep_lim = if WAYS <= 2 { 1 } else { 3 };
    for (ri, &r) in rep.iter().take(rep_lim).enumerate() {
        if (ri == 0 && pos == anchor) || r == 0 || r > MAX_OFFSET {
            continue;
        }
        let c_ml = if r <= pos {
            if (read_u32(chunk, pos - r) & 0x00ff_ffff) != w3 {
                continue;
            }
            3 + match_len_slices(chunk, pos - r + 3, chunk, pos + 3, rem - 3)
        } else if d_len + pos >= r {
            let d_cand = d_len + pos - r;
            let d_lim = rem.min(d_len - d_cand);
            if d_lim < 3 || !eq3_at(d_slice, d_cand, w3) {
                continue;
            }
            3 + match_len_slices(d_slice, d_cand + 3, chunk, pos + 3, d_lim - 3)
        } else {
            continue;
        };
        let sc = (c_ml as i32) * 4 + 3 - (ri as i32);
        if sc > best_sc {
            ml = c_ml;
            off = r;
            rep_idx = ri as u8;
            best_sc = sc;
            if ml >= 64 {
                return (ml, off, rep_idx, best_sc);
            }
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
                        if ml >= 64 {
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

    let dict6_gate = if WAYS >= 8 { 96 } else { 48 };
    if ml < dict6_gate {
        if let Some(d) = dict {
            let (d_ml, d_off, d_sc) =
                probe_dict6::<DWAYS>(d, chunk, pos, rem, w6, m6, ml, off, best_sc);
            if d_sc > best_sc {
                ml = d_ml;
                off = d_off;
                rep_idx = 255;
                best_sc = d_sc;
                if ml >= 64 {
                    return (ml, off, rep_idx, best_sc);
                }
            }
        }
    }

    let max_off4 = if WAYS >= 8 {
        65_535
    } else if WAYS >= 4 {
        32_768
    } else {
        16_384
    };
    let ht4_gate = if WAYS >= 4 { 8 } else { 6 };
    if NEED_HT4 && ml < ht4_gate {
        if pos > 0 {
            let b4 = unsafe { *local_ht4.get_unchecked((p4 >> LOCAL_HT4_SHIFT) as usize) };
            for raw4 in b4 {
                if let Some(cand4) = resolve_cand::<SMALL>(pos, raw4) {
                    if read_u32(chunk, cand4) == w4 {
                        let c_off = pos - cand4;
                        let c_ml = 4 + match_len_slices(chunk, cand4 + 4, chunk, pos + 4, rem - 4);
                        if c_ml >= 5 || c_off <= max_off4 {
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
                let (d_ml, d_off, d_sc) =
                    probe_dict4::<DWAYS>(d, chunk, pos, rem, w4, p4, max_off4, ml, off, best_sc);
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
