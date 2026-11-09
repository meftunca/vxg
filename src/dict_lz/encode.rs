use super::{
    mul4, mul6,
    pack::{push_ext_len, serialize_streams, BitWriter, StreamBufs},
    probe::{probe_pos, LOCAL_HT4_SHIFT, LOCAL_HT6_SHIFT},
    read_u64, PreparedDict, LOCAL_HT4_SIZE, LOCAL_HT6_SIZE,
};

type FastTables = (
    Box<[[u16; 2]; LOCAL_HT4_SIZE]>,
    Box<[[u16; 2]; LOCAL_HT6_SIZE]>,
);
type DeepTables = (
    Box<[[u16; 8]; LOCAL_HT4_SIZE]>,
    Box<[[u16; 8]; LOCAL_HT6_SIZE]>,
);

pub(crate) struct DictLzScratch {
    local_ht4: Box<[[u16; 4]; LOCAL_HT4_SIZE]>,
    local_ht6: Box<[[u16; 4]; LOCAL_HT6_SIZE]>,
    fast_tables: Option<FastTables>,
    deep_tables: Option<DeepTables>,
    sb: StreamBufs,
}

impl DictLzScratch {
    pub(crate) fn new() -> Self {
        Self {
            local_ht4: Box::new([[0u16; 4]; LOCAL_HT4_SIZE]),
            local_ht6: Box::new([[0u16; 4]; LOCAL_HT6_SIZE]),
            fast_tables: None,
            deep_tables: None,
            sb: StreamBufs::new(),
        }
    }
}

#[inline(always)]
fn push_way<const WAYS: usize>(b: &mut [u16; WAYS], val: u16) {
    let mut i = WAYS - 1;
    while i > 0 {
        b[i] = b[i - 1];
        i -= 1;
    }
    b[0] = val;
}

#[inline(always)]
fn insert_local<const WAYS: usize>(
    chunk: &[u8],
    pos: usize,
    local_ht4: &mut [[u16; WAYS]; LOCAL_HT4_SIZE],
    local_ht6: &mut [[u16; WAYS]; LOCAL_HT6_SIZE],
) {
    let w8 = read_u64(chunk, pos);
    let p16 = pos as u16;
    unsafe {
        push_way(
            local_ht6.get_unchecked_mut((mul6(w8) >> LOCAL_HT6_SHIFT) as usize),
            p16,
        );
        push_way(
            local_ht4.get_unchecked_mut((mul4(w8 as u32) >> LOCAL_HT4_SHIFT) as usize),
            p16,
        );
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

#[allow(clippy::too_many_arguments)]
fn encode_loop<const SMALL: bool, const WAYS: usize, const DWAYS: usize>(
    chunk: &[u8],
    dict: Option<&PreparedDict>,
    local_ht4: &mut [[u16; WAYS]; LOCAL_HT4_SIZE],
    local_ht6: &mut [[u16; WAYS]; LOCAL_HT6_SIZE],
    sb: &mut StreamBufs,
    level: u8,
) -> usize {
    let n = chunk.len();
    local_ht4.fill([0u16; WAYS]);
    local_ht6.fill([0u16; WAYS]);
    let d_slice: &[u8] = dict.map_or(&[], |d| d.raw.as_slice());
    let d_len = d_slice.len();
    let limit = n.saturating_sub(8);

    sb.reset_for(n);
    insert_local(chunk, 0, local_ht4, local_ht6);

    let mut rep = [1usize, 4, 8];
    let mut pos = if dict.is_some() { 0usize } else { 1usize };
    let mut anchor = 0usize;
    let mut misses = 0usize;
    let lazy_max: usize = match level {
        0 | 1 => 0,
        2 => 14,
        3 => 24,
        4 | 5 => 32,
        _ => 48,
    };
    let lazy_ht4_max: usize = if level >= 6 { 16 } else { 8 };

    let StreamBufs {
        lits,
        toks,
        off_codes,
        len_ext,
        off_bits,
        ..
    } = sb;
    let mut bw = BitWriter::new(off_bits);

    while pos <= limit {
        let w8 = read_u64(chunk, pos);
        let m6 = mul6(w8);
        let p4 = mul4(w8 as u32);

        let (mut ml, mut off, mut rep_idx, mut best_sc) = probe_pos::<SMALL, WAYS, DWAYS, true>(
            chunk, n, pos, anchor, &rep, dict, local_ht4, local_ht6, w8, m6, p4,
        );

        if pos > 0 {
            let p16 = pos as u16;
            unsafe {
                push_way(
                    local_ht6.get_unchecked_mut((m6 >> LOCAL_HT6_SHIFT) as usize),
                    p16,
                );
                push_way(
                    local_ht4.get_unchecked_mut((p4 >> LOCAL_HT4_SHIFT) as usize),
                    p16,
                );
            }
        }

        if ml == 0 {
            if level >= 7 {
                pos += 1;
            } else {
                misses += 1;
                let shift = if pos >= 2048 && lits.len() + (pos - anchor) > (pos * 3) / 4 {
                    3
                } else if level >= 4 {
                    5
                } else {
                    4
                };
                pos += 1 + (misses >> shift);
            }
            continue;
        }
        misses = 0;

        let noisy = level < 6 && pos >= 2048 && lits.len() + (pos - anchor) > (pos * 3) / 4;
        if !noisy && ml <= lazy_max && pos + 1 <= limit {
            let p1 = pos + 1;
            let wn = read_u64(chunk, p1);
            let m6n = mul6(wn);
            let p4n = mul4(wn as u32);
            let (n_ml, n_off, n_rep, n_sc) = if ml <= lazy_ht4_max {
                probe_pos::<SMALL, WAYS, DWAYS, true>(
                    chunk, n, p1, anchor, &rep, dict, local_ht4, local_ht6, wn, m6n, p4n,
                )
            } else {
                probe_pos::<SMALL, WAYS, DWAYS, false>(
                    chunk, n, p1, anchor, &rep, dict, local_ht4, local_ht6, wn, m6n, p4n,
                )
            };
            if n_ml > ml && n_sc > best_sc + 2 {
                insert_local(chunk, p1, local_ht4, local_ht6);
                pos = p1;
                ml = n_ml;
                off = n_off;
                rep_idx = n_rep;
                best_sc = n_sc;

                if level >= 7 && ml <= lazy_max && pos + 1 <= limit {
                    let p2 = pos + 1;
                    let w2 = read_u64(chunk, p2);
                    let (m2, o2, r2, s2) = probe_pos::<SMALL, WAYS, DWAYS, true>(
                        chunk,
                        n,
                        p2,
                        anchor,
                        &rep,
                        dict,
                        local_ht4,
                        local_ht6,
                        w2,
                        mul6(w2),
                        mul4(w2 as u32),
                    );
                    if m2 > ml && s2 > best_sc + 2 {
                        insert_local(chunk, p2, local_ht4, local_ht6);
                        pos = p2;
                        ml = m2;
                        off = o2;
                        rep_idx = r2;
                    }
                }
            }
        }

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
        let step = if level >= 6 || ml <= 16 {
            1
        } else if ml <= 48 {
            2
        } else {
            4
        };
        let mut q = pos + step;
        while q < idx_lim {
            insert_local(chunk, q, local_ht4, local_ht6);
            q += step;
        }
        if step > 1 && idx_lim > pos + 1 {
            insert_local(chunk, idx_lim - 1, local_ht4, local_ht6);
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
    level: u8,
) -> Option<(usize, usize)> {
    let small = chunk.len() <= 65536;
    let anchor = if level <= 2 {
        let (ht4, ht6) = sc.fast_tables.get_or_insert_with(|| {
            (
                Box::new([[0u16; 2]; LOCAL_HT4_SIZE]),
                Box::new([[0u16; 2]; LOCAL_HT6_SIZE]),
            )
        });
        if small {
            encode_loop::<true, 2, 2>(chunk, dict, ht4, ht6, &mut sc.sb, level)
        } else {
            encode_loop::<false, 2, 2>(chunk, dict, ht4, ht6, &mut sc.sb, level)
        }
    } else if level <= 5 {
        if small {
            encode_loop::<true, 4, 4>(
                chunk,
                dict,
                &mut sc.local_ht4,
                &mut sc.local_ht6,
                &mut sc.sb,
                level,
            )
        } else {
            encode_loop::<false, 4, 4>(
                chunk,
                dict,
                &mut sc.local_ht4,
                &mut sc.local_ht6,
                &mut sc.sb,
                level,
            )
        }
    } else {
        let (ht4, ht6) = sc.deep_tables.get_or_insert_with(|| {
            (
                Box::new([[0u16; 8]; LOCAL_HT4_SIZE]),
                Box::new([[0u16; 8]; LOCAL_HT6_SIZE]),
            )
        });
        if small {
            encode_loop::<true, 8, 4>(chunk, dict, ht4, ht6, &mut sc.sb, level)
        } else {
            encode_loop::<false, 8, 4>(chunk, dict, ht4, ht6, &mut sc.sb, level)
        }
    };
    serialize_streams(chunk, anchor, &mut sc.sb, out, max_len, level)
}
