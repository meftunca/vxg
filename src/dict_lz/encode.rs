use super::{
    hash4, hash6, insert_pos,
    pack::{push_ext_len, serialize_streams, BitWriter, StreamBufs},
    read_u32, read_u64, PreparedDict, HT4_SIZE, HT6_SIZE,
};

pub(crate) struct DictLzScratch {
    cached_dict_id: u64,
    cached_dict_len: usize,
    window: Vec<u8>,
    ht4: Box<[[u32; 4]; HT4_SIZE]>,
    ht6: Box<[[u32; 2]; HT6_SIZE]>,
    sb: StreamBufs,
}

impl DictLzScratch {
    pub(crate) fn new() -> Self {
        Self {
            cached_dict_id: 0,
            cached_dict_len: 0,
            window: Vec::new(),
            ht4: Box::new([[0u32; 4]; HT4_SIZE]),
            ht6: Box::new([[0u32; 2]; HT6_SIZE]),
            sb: StreamBufs::new(),
        }
    }
}

#[inline(always)]
fn match_len_at(win: &[u8], mut a: usize, mut b: usize, limit: usize) -> usize {
    let start = b;
    let end = b + limit;
    while b + 8 <= end {
        let diff = read_u64(win, a) ^ read_u64(win, b);
        if diff != 0 {
            return (b - start) + (diff.trailing_zeros() as usize >> 3);
        }
        a += 8;
        b += 8;
    }
    while b < end && unsafe { *win.get_unchecked(a) == *win.get_unchecked(b) } {
        a += 1;
        b += 1;
    }
    b - start
}

#[derive(Clone, Copy)]
struct Match {
    len: usize,
    offset: usize,
    rep_idx: u8,
    score: i32,
}

#[inline(always)]
fn off_penalty(off: usize) -> i32 {
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
fn find_best_match(
    win: &[u8],
    vpos: usize,
    total_len: usize,
    rep: &[usize; 3],
    ht4: &[[u32; 4]; HT4_SIZE],
    ht6: &[[u32; 2]; HT6_SIZE],
    allow_rep0: bool,
) -> Match {
    let rem = total_len - vpos;
    let mut best = Match {
        len: 0,
        offset: 0,
        rep_idx: 255,
        score: 0,
    };
    if rem < 8 {
        return best;
    }
    let w8 = read_u64(win, vpos);
    let w6 = w8 & 0x0000_ffff_ffff_ffff;
    let w4 = w8 as u32;
    let w3 = w4 & 0x00ff_ffff;

    // 1. Probe repeat offsets (`len >= 3`).
    for (r, &roff) in rep.iter().enumerate() {
        if (r == 0 && !allow_rep0) || roff == 0 || roff > vpos {
            continue;
        }
        let cand = vpos - roff;
        if (read_u32(win, cand) & 0x00ff_ffff) == w3 {
            let ml = 3 + match_len_at(win, cand + 3, vpos + 3, rem - 3);
            let sc = (ml as i32) * 4 + 3 - i32::from(r > 0);
            if sc > best.score {
                best = Match {
                    len: ml,
                    offset: roff,
                    rep_idx: r as u8,
                    score: sc,
                };
                if ml >= 96 {
                    return best;
                }
            }
        }
    }

    // 2. Probe 2-way 6-byte hash table (`1` cache line).
    let b6 = unsafe { ht6.get_unchecked(hash6(w8)) };
    for &raw_p in b6 {
        if raw_p == 0 {
            break;
        }
        let cand = (raw_p - 1) as usize;
        if cand >= vpos {
            continue;
        }
        if best.len >= 6
            && unsafe { *win.get_unchecked(cand + best.len) != *win.get_unchecked(vpos + best.len) }
        {
            continue;
        }
        if (read_u64(win, cand) & 0x0000_ffff_ffff_ffff) == w6 {
            let off = vpos - cand;
            let ml = 6 + match_len_at(win, cand + 6, vpos + 6, rem - 6);
            let sc = (ml as i32) * 4 - off_penalty(off);
            if sc > best.score {
                best = Match {
                    len: ml,
                    offset: off,
                    rep_idx: 255,
                    score: sc,
                };
                if ml >= 96 {
                    return best;
                }
            }
        }
    }

    // 3. Probe 4-way 4-byte hash table (`1` cache line, skipped when `best.len >= 16`).
    if best.len < 16 {
        let b4 = unsafe { ht4.get_unchecked(hash4(w4)) };
        for &raw_p in b4 {
            if raw_p == 0 {
                break;
            }
            let cand = (raw_p - 1) as usize;
            if cand >= vpos {
                continue;
            }
            if best.len >= 4
                && unsafe {
                    *win.get_unchecked(cand + best.len) != *win.get_unchecked(vpos + best.len)
                }
            {
                continue;
            }
            if read_u32(win, cand) == w4 {
                let off = vpos - cand;
                let ml = 4 + match_len_at(win, cand + 4, vpos + 4, rem - 4);
                if ml == 4 && off > 16384 {
                    continue;
                }
                let sc = (ml as i32) * 4 - off_penalty(off);
                if sc > best.score {
                    best = Match {
                        len: ml,
                        offset: off,
                        rep_idx: 255,
                        score: sc,
                    };
                    if ml >= 96 {
                        return best;
                    }
                }
            }
        }
    }

    best
}

pub(super) fn encode_into(
    chunk: &[u8],
    dict: Option<&PreparedDict>,
    sc: &mut DictLzScratch,
    out: &mut Vec<u8>,
    max_len: usize,
) -> Option<(usize, usize)> {
    let n = chunk.len();
    let d_len = match dict {
        Some(d) => {
            if sc.cached_dict_id != d.id || sc.cached_dict_len != d.raw.len() {
                sc.window.clear();
                sc.window.reserve(d.raw.len() + n + 16);
                sc.window.extend_from_slice(&d.raw);
                sc.cached_dict_id = d.id;
                sc.cached_dict_len = d.raw.len();
            }
            sc.window.truncate(d.raw.len());
            sc.window.extend_from_slice(chunk);
            sc.ht4.copy_from_slice(&*d.ht4);
            sc.ht6.copy_from_slice(&*d.ht6);
            d.raw.len()
        }
        None => {
            sc.cached_dict_id = 0;
            sc.cached_dict_len = 0;
            sc.window.clear();
            sc.window.extend_from_slice(chunk);
            sc.ht4.fill([0; 4]);
            sc.ht6.fill([0; 2]);
            0
        }
    };

    let total_len = d_len + n;
    sc.sb.reset_for(n);
    let mut rep = [1usize, 4, 8];
    let mut pos = 0usize;
    let mut anchor = 0usize;
    let mut misses = 0usize;
    let mut bw = BitWriter::new(&mut sc.sb.off_bits);

    while pos + 8 <= n {
        let vpos = d_len + pos;
        let mut m = find_best_match(
            &sc.window,
            vpos,
            total_len,
            &rep,
            &sc.ht4,
            &sc.ht6,
            pos > anchor,
        );
        insert_pos(&sc.window, vpos, &mut sc.ht4, &mut sc.ht6);

        if m.len == 0 {
            misses += 1;
            let shift = if pos >= 2048 && sc.sb.lits.len() + (pos - anchor) > (pos * 3) / 4 {
                3
            } else {
                5
            };
            let step = 1 + (misses >> shift);
            for s in 1..step.min(n - 7 - pos) {
                if (s & 1) == 0 && shift == 5 {
                    insert_pos(&sc.window, vpos + s, &mut sc.ht4, &mut sc.ht6);
                }
            }
            pos += step;
            continue;
        }
        misses = 0;

        // Adaptive lazy evaluation (skipped on noisy/incompressible blocks where literals > 75%).
        let noisy = pos >= 2048 && sc.sb.lits.len() + (pos - anchor) > (pos * 3) / 4;
        let max_lazy = if noisy { 0 } else { 2 };
        let mut lazy_steps = 0usize;
        while m.len < 28 && lazy_steps < max_lazy && pos + 9 <= n {
            let next_vpos = d_len + pos + 1;
            let m_next = find_best_match(
                &sc.window,
                next_vpos,
                total_len,
                &rep,
                &sc.ht4,
                &sc.ht6,
                true,
            );
            if m_next.len > m.len && m_next.score > m.score + 3 {
                pos += 1;
                insert_pos(&sc.window, next_vpos, &mut sc.ht4, &mut sc.ht6);
                m = m_next;
                lazy_steps += 1;
            } else {
                break;
            }
        }

        // Backward match extension across skipped literal bytes or unindexed positions.
        while pos > anchor
            && d_len + pos > m.offset
            && unsafe {
                *sc.window.get_unchecked(d_len + pos - 1)
                    == *sc.window.get_unchecked(d_len + pos - 1 - m.offset)
            }
        {
            pos -= 1;
            m.len += 1;
        }

        let lit_len = pos - anchor;
        if lit_len > 0 {
            sc.sb.lits.extend_from_slice(&chunk[anchor..pos]);
        }
        let ll_code = lit_len.min(15);
        let ml_sub = m.len - 3;
        let ml_code = ml_sub.min(15);
        sc.sb.toks.push((ll_code as u8) | ((ml_code as u8) << 4));
        if ll_code == 15 {
            push_ext_len(&mut sc.sb.len_ext, lit_len - 15);
        }
        if ml_code == 15 {
            push_ext_len(&mut sc.sb.len_ext, ml_sub - 15);
        }

        match m.rep_idx {
            0 => sc.sb.off_codes.push(0),
            1 => {
                sc.sb.off_codes.push(1);
                rep.swap(0, 1);
            }
            2 => {
                sc.sb.off_codes.push(2);
                let r2 = rep[2];
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = r2;
            }
            _ => {
                let off = m.offset;
                let bits = 31 - (off as u32).leading_zeros();
                sc.sb.off_codes.push((3 + bits) as u8);
                if bits > 0 {
                    bw.put((off as u32) & ((1u32 << bits) - 1), bits);
                }
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = off;
            }
        }

        let match_end = pos + m.len;
        let idx_limit = match_end.min(n.saturating_sub(8));
        if m.len <= 16 {
            let mut q = pos + 1;
            while q < idx_limit {
                insert_pos(&sc.window, d_len + q, &mut sc.ht4, &mut sc.ht6);
                q += 1;
            }
        } else if m.len <= 64 {
            let mut q = pos + 2;
            while q < idx_limit {
                insert_pos(&sc.window, d_len + q, &mut sc.ht4, &mut sc.ht6);
                q += 2;
            }
        } else if idx_limit > pos + 2 {
            insert_pos(&sc.window, d_len + idx_limit - 2, &mut sc.ht4, &mut sc.ht6);
            insert_pos(&sc.window, d_len + idx_limit - 1, &mut sc.ht4, &mut sc.ht6);
        }

        pos = match_end;
        anchor = pos;
    }
    bw.finish();

    serialize_streams(chunk, anchor, &mut sc.sb, out, max_len)
}
