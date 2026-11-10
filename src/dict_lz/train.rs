use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Mutex;

use crate::{columns, pool::parallel_run_checked};

use super::{
    build_dict_tables, encode_dict_block_into, encode_wire_dict, read_u64,
    train_macro::try_macroscopic_dict, PreparedDict, MIN_DICT_RAW,
};

pub(super) const SKETCH6_BITS: usize = 15;
pub(super) const SKETCH6_SIZE: usize = 1 << SKETCH6_BITS; // 32,768 x u8 = 32 KiB (L1D resident)
pub(super) const SKETCH16_BITS: usize = 16;
pub(super) const SKETCH16_SIZE: usize = 1 << SKETCH16_BITS; // 65,536 x u8 = 64 KiB
pub(super) const DENSE_BLK: usize = 256;
const SEG_LEN: usize = 64;
const SEG_STRIDE: usize = 32;
const MAX_CANDIDATES: usize = 4096;

pub(super) struct TrainScratch {
    pub(super) freq6: Box<[u8; SKETCH6_SIZE]>,
    pub(super) seen6: Box<[u8; SKETCH6_SIZE]>,
    pub(super) touched6: Vec<u16>,
    pub(super) freq16: Box<[u8; SKETCH16_SIZE]>,
    pub(super) seen16: Box<[u8; SKETCH16_SIZE]>,
    pub(super) touched16: Vec<u16>,
}

impl TrainScratch {
    fn new() -> Self {
        Self {
            freq6: vec![0u8; SKETCH6_SIZE].into_boxed_slice().try_into().unwrap(),
            seen6: vec![0u8; SKETCH6_SIZE].into_boxed_slice().try_into().unwrap(),
            touched6: Vec::with_capacity(16_384),
            freq16: vec![0u8; SKETCH16_SIZE].into_boxed_slice().try_into().unwrap(),
            seen16: vec![0u8; SKETCH16_SIZE].into_boxed_slice().try_into().unwrap(),
            touched16: Vec::with_capacity(8_192),
        }
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        for &idx in &self.touched6 {
            unsafe {
                *self.freq6.get_unchecked_mut(idx as usize) = 0;
                *self.seen6.get_unchecked_mut(idx as usize) = 0;
            }
        }
        self.touched6.clear();
        for &idx in &self.touched16 {
            unsafe {
                *self.freq16.get_unchecked_mut(idx as usize) = 0;
                *self.seen16.get_unchecked_mut(idx as usize) = 0;
            }
        }
        self.touched16.clear();
    }
}

thread_local! {
    pub(super) static TRAIN_SCRATCH: RefCell<TrainScratch> = RefCell::new(TrainScratch::new());
}

#[inline(always)]
fn hash6(data: &[u8], pos: usize) -> usize {
    let w = read_u64(data, pos) & 0x0000_ffff_ffff_ffff;
    (w.wrapping_mul(0x9e37_79b1_85eb_ca87) >> (64 - SKETCH6_BITS)) as usize
}

/// Fast-reject random or high-entropy encrypted inputs.
#[inline]
fn probe_looks_random(probe: &[u8]) -> bool {
    if probe.len() < 256 {
        return false;
    }
    let mut bmap = [0u64; 4];
    for &b in &probe[..256] {
        bmap[(b >> 6) as usize] |= 1u64 << (b & 63);
    }
    let distinct =
        bmap[0].count_ones() + bmap[1].count_ones() + bmap[2].count_ones() + bmap[3].count_ones();
    distinct >= 196
}

fn scan_shingles_range(
    sc: &mut TrainScratch,
    input: &[u8],
    windows: usize,
    win_len: usize,
    usable_span: usize,
    step: usize,
    w_lo: usize,
    w_hi: usize,
) -> usize {
    let n = input.len();
    let mut cross = 0usize;
    for w_idx in w_lo..w_hi {
        let tag = (w_idx + 1) as u8;
        let w_start = if windows > 1 {
            (usable_span * w_idx) / (windows - 1)
        } else {
            0
        };
        let slice = &input[w_start..(w_start + win_len).min(n)];
        let mut p = 0usize;
        while p + 8 <= slice.len() {
            let h = hash6(slice, p);
            let sw = unsafe { sc.seen6.get_unchecked_mut(h) };
            if *sw != tag {
                *sw = tag;
                let prev = unsafe { *sc.freq6.get_unchecked(h) };
                unsafe { *sc.freq6.get_unchecked_mut(h) = prev.saturating_add(1) };
                if prev == 0 {
                    sc.touched6.push(h as u16);
                } else if prev == 1 {
                    cross += 1;
                }
            }
            p += step;
        }
    }
    cross
}

fn score_candidates_range(
    freq6: &[u8; SKETCH6_SIZE],
    input: &[u8],
    windows: usize,
    win_len: usize,
    usable_span: usize,
    step: usize,
    min_score: u32,
    w_lo: usize,
    w_hi: usize,
    out: &mut Vec<(u32, usize)>,
) {
    let n = input.len();
    for w_idx in w_lo..w_hi {
        let w_start = if windows > 1 {
            (usable_span * w_idx) / (windows - 1)
        } else {
            0
        };
        let w_end = (w_start + win_len).min(n);
        let mut s = w_start;
        while s + SEG_LEN <= w_end && out.len() < MAX_CANDIDATES {
            let seg = &input[s..s + SEG_LEN];
            let mut score = 0u32;
            let mut p = 0usize;
            while p + 8 <= SEG_LEN {
                let c = unsafe { *freq6.get_unchecked(hash6(seg, p)) };
                if c >= 2 {
                    score += u32::from(c);
                }
                p += step;
            }
            if score >= min_score {
                out.push((score, s));
            }
            s += SEG_STRIDE;
        }
    }
}

fn train_raw_dict(
    input: &[u8],
    block_size: usize,
    is_structured: bool,
    level: u8,
    workers: usize,
) -> Option<Vec<u8>> {
    let n = input.len();
    let count = n.div_ceil(block_size);
    if count < 3 || n < 24_576 {
        return None;
    }
    let mid = n / 2;
    if probe_looks_random(&input[mid..n.min(mid + 256)]) {
        return None;
    }

    TRAIN_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.clear();

        if !is_structured {
            if let Some(macro_raw) = try_macroscopic_dict(&mut sc, input, count, workers) {
                return Some(macro_raw);
            }
        }

        let max_target = if level >= 8 {
            114_688
        } else if level >= 6 {
            65_536
        } else {
            32_768
        };
        let div = if level >= 6 { 6 } else { 8 };
        let target_len = (n / div).clamp(4096, max_target);
        let max_wins = if level >= 6 { 96 } else { 64 };
        let windows = count.clamp(4, max_wins);
        let win_len = (block_size / 4).clamp(1024, 2048);
        let usable_span = n.saturating_sub(win_len);
        let step = if level >= 6 || n <= 8 * 1024 * 1024 {
            1usize
        } else {
            2usize
        };
        let min_score = if step == 1 { 18u32 } else { 10u32 };
        let min_hits = if step == 1 { 9usize } else { 5usize };

        let w_threads = workers.min(8).min(windows / 4);
        let mut cross_shingles;
        if w_threads <= 1 {
            cross_shingles =
                scan_shingles_range(&mut sc, input, windows, win_len, usable_span, step, 0, windows);
        } else {
            let partials: Vec<Mutex<Vec<(u16, u8)>>> =
                (0..w_threads).map(|_| Mutex::new(Vec::new())).collect();
            let c0 = Mutex::new(0usize);
            let sc0_ptr = AtomicPtr::new(&mut *sc as *mut TrainScratch);
            let _ = parallel_run_checked(w_threads, |wid| {
                let w_lo = (windows * wid) / w_threads;
                let w_hi = (windows * (wid + 1)) / w_threads;
                if wid == 0 {
                    let sc0 = unsafe { &mut *sc0_ptr.load(Ordering::Relaxed) };
                    let c = scan_shingles_range(
                        sc0, input, windows, win_len, usable_span, step, w_lo, w_hi,
                    );
                    *c0.lock().unwrap() = c;
                } else {
                    TRAIN_SCRATCH.with(|wcell| {
                        let mut lsc = wcell.borrow_mut();
                        lsc.clear();
                        let _ = scan_shingles_range(
                            &mut lsc, input, windows, win_len, usable_span, step, w_lo, w_hi,
                        );
                        let mut pairs = Vec::with_capacity(lsc.touched6.len());
                        for &idx in &lsc.touched6 {
                            pairs.push((idx, unsafe { *lsc.freq6.get_unchecked(idx as usize) }));
                        }
                        lsc.clear();
                        *partials[wid].lock().unwrap() = pairs;
                    });
                }
            });
            cross_shingles = *c0.lock().unwrap();
            for slot in partials.into_iter().skip(1) {
                for (idx, add) in slot.into_inner().unwrap() {
                    let prev = unsafe { *sc.freq6.get_unchecked(idx as usize) };
                    let next = prev.saturating_add(add);
                    unsafe { *sc.freq6.get_unchecked_mut(idx as usize) = next };
                    if prev == 0 {
                        sc.touched6.push(idx);
                    }
                    if prev < 2 && next >= 2 {
                        cross_shingles += 1;
                    }
                }
            }
        }

        if cross_shingles < 48 {
            return None;
        }

        let mut candidates: Vec<(u32, usize)> = Vec::with_capacity(MAX_CANDIDATES.min(windows * 64));
        if w_threads <= 1 {
            score_candidates_range(
                &sc.freq6,
                input,
                windows,
                win_len,
                usable_span,
                step,
                min_score,
                0,
                windows,
                &mut candidates,
            );
        } else {
            let freq_ref: &[u8; SKETCH6_SIZE] = &sc.freq6;
            let cands_slots: Vec<Mutex<Vec<(u32, usize)>>> =
                (0..w_threads).map(|_| Mutex::new(Vec::new())).collect();
            let _ = parallel_run_checked(w_threads, |wid| {
                let w_lo = (windows * wid) / w_threads;
                let w_hi = (windows * (wid + 1)) / w_threads;
                let mut loc = Vec::with_capacity((w_hi - w_lo) * 32);
                score_candidates_range(
                    freq_ref,
                    input,
                    windows,
                    win_len,
                    usable_span,
                    step,
                    min_score,
                    w_lo,
                    w_hi,
                    &mut loc,
                );
                *cands_slots[wid].lock().unwrap() = loc;
            });
            for slot in cands_slots {
                let loc = slot.into_inner().unwrap();
                let rem = MAX_CANDIDATES - candidates.len();
                if rem == 0 {
                    break;
                }
                candidates.extend_from_slice(&loc[..loc.len().min(rem)]);
            }
        }

        if candidates.is_empty() {
            return None;
        }
        candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let mut selected: Vec<(usize, usize)> = Vec::with_capacity(target_len / SEG_LEN + 8);
        let mut total_bytes = 0usize;
        for &(_init_score, start) in &candidates {
            if total_bytes >= target_len {
                break;
            }
            let seg = &input[start..start + SEG_LEN];
            let (mut live_score, mut live_hits) = (0u32, 0usize);
            let mut p = 0usize;
            while p + 8 <= SEG_LEN {
                let c = unsafe { *sc.freq6.get_unchecked(hash6(seg, p)) };
                if c >= 2 {
                    live_score += u32::from(c);
                    live_hits += 1;
                }
                p += step;
            }
            if live_hits < min_hits || live_score < min_score {
                continue;
            }
            let base_end = start + SEG_LEN;
            let max_ext = (base_end + 48).min(n);
            let mut end = base_end;
            for (off, &b) in input[base_end..max_ext].iter().enumerate() {
                if matches!(b, b'\n' | b',' | b'}' | b'"' | b' ') {
                    end = base_end + off + 1;
                    if b == b'\n' {
                        break;
                    }
                }
            }
            let take = (end - start).min(target_len - total_bytes);
            if take < 24 {
                break;
            }
            let chosen = &input[start..start + take];
            let mut z = 0usize;
            while z + 8 <= chosen.len() {
                unsafe { *sc.freq6.get_unchecked_mut(hash6(chosen, z)) = 0 };
                z += step;
            }
            selected.push((start, start + take));
            total_bytes += take;
        }

        if total_bytes < MIN_DICT_RAW {
            return None;
        }
        let mut raw = Vec::with_capacity(total_bytes);
        for &(s, e) in selected.iter().rev() {
            raw.extend_from_slice(&input[s..e]);
        }
        Some(raw)
    })
}

pub(super) fn build_prepared_dict(
    input: &[u8],
    block_size: usize,
    level: u8,
    workers: usize,
) -> Option<PreparedDict> {
    let workers = workers.max(1);
    let count = input.len().div_ceil(block_size);
    if count < 3 || input.len() < 24_576 {
        return None;
    }
    let p0 = count / 2;
    let st0 = p0 * block_size;
    let chunk0 = &input[st0..(st0 + block_size).min(input.len())];
    let carry0 = st0 > 0 && input[st0 - 1] != b'\n';

    let mut tmp_col = Vec::with_capacity(block_size);
    let structured0 = columns::encode_structured_only_into(chunk0, carry0, &mut tmp_col)
        .ok()
        .flatten()
        .map(|(_, l)| l);

    let strongly_tabular = structured0.is_some_and(|l| l * 4 < chunk0.len());
    let raw = train_raw_dict(input, block_size, strongly_tabular, level, workers)?;
    let (ht4, ht6, ht6_shift) = build_dict_tables(&raw);

    let mut dict = PreparedDict {
        raw,
        wire_payload: Vec::new(),
        prefer_over_grammar: false,
        try_structured: false,
        ht4,
        ht6,
        ht6_shift,
    };

    let mut pre_wire: Option<Vec<u8>> = None;
    let (col_sum, dlz_sum, any_structured, pilot_bytes) = if let Some(len0) = structured0 {
        let p1 = count / 3;
        if workers <= 1 {
            let mut tmp_dlz = Vec::with_capacity(block_size);
            let dlz0 = encode_dict_block_into(chunk0, Some(&dict), &mut tmp_dlz, len0, level)
                .map_or(len0, |(_, l)| l);
            let (mut cs, mut ds, mut pb) = (len0, dlz0, chunk0.len());
            if p1 != p0 {
                let st1 = p1 * block_size;
                let chunk1 = &input[st1..(st1 + block_size).min(input.len())];
                let carry1 = st1 > 0 && input[st1 - 1] != b'\n';
                tmp_col.clear();
                let len1 = columns::encode_structured_only_into(chunk1, carry1, &mut tmp_col)
                    .ok()
                    .flatten()
                    .map_or(chunk1.len(), |(_, l)| l);
                tmp_dlz.clear();
                let dlz1 = encode_dict_block_into(chunk1, Some(&dict), &mut tmp_dlz, len1, level)
                    .map_or(len1, |(_, l)| l);
                cs += len1;
                ds += dlz1;
                pb += chunk1.len();
            }
            (cs, ds, true, pb)
        } else {
            let has_p1 = p1 != p0;
            let n_tasks = (1 + usize::from(has_p1) + 1).min(workers);
            let r0 = Mutex::new(len0);
            let r1 = Mutex::new((0usize, 0usize, 0usize));
            let rw = Mutex::new(None);
            let dict_ref = &dict;
            let _ = parallel_run_checked(n_tasks, |wid| {
                if wid == 0 {
                    let mut buf = Vec::with_capacity(block_size);
                    let dlz0 = encode_dict_block_into(chunk0, Some(dict_ref), &mut buf, len0, level)
                        .map_or(len0, |(_, l)| l);
                    *r0.lock().unwrap() = dlz0;
                } else if wid == 1 && has_p1 {
                    let st1 = p1 * block_size;
                    let chunk1 = &input[st1..(st1 + block_size).min(input.len())];
                    let carry1 = st1 > 0 && input[st1 - 1] != b'\n';
                    let mut cbuf = Vec::with_capacity(block_size);
                    let len1 = columns::encode_structured_only_into(chunk1, carry1, &mut cbuf)
                        .ok()
                        .flatten()
                        .map_or(chunk1.len(), |(_, l)| l);
                    cbuf.clear();
                    let dlz1 =
                        encode_dict_block_into(chunk1, Some(dict_ref), &mut cbuf, len1, level)
                            .map_or(len1, |(_, l)| l);
                    *r1.lock().unwrap() = (len1, dlz1, chunk1.len());
                } else {
                    *rw.lock().unwrap() = Some(encode_wire_dict(&dict_ref.raw, level));
                }
            });
            pre_wire = rw.into_inner().unwrap();
            let dlz0 = r0.into_inner().unwrap();
            let (l1, d1, b1) = r1.into_inner().unwrap();
            (len0 + l1, dlz0 + d1, true, chunk0.len() + b1)
        }
    } else if workers <= 1 {
        let sub = chunk0;
        tmp_col.clear();
        let col_len = columns::encode_grammar_block_into_with_carry(sub, carry0, &mut tmp_col)
            .ok()
            .flatten()
            .map_or(sub.len(), |(_, l)| l);
        let mut tmp_dlz = Vec::with_capacity(block_size);
        let dlz_len = encode_dict_block_into(sub, Some(&dict), &mut tmp_dlz, col_len, level)
            .map_or(col_len, |(_, l)| l);
        (col_len, dlz_len, false, sub.len())
    } else {
        let sub = chunk0;
        let n_tasks = 3.min(workers);
        let rc = Mutex::new(sub.len());
        let rd = Mutex::new(None);
        let rw = Mutex::new(None);
        let dict_ref = &dict;
        let _ = parallel_run_checked(n_tasks, |wid| {
            if wid == 0 {
                let mut cbuf = Vec::with_capacity(block_size);
                let cl = columns::encode_grammar_block_into_with_carry(sub, carry0, &mut cbuf)
                    .ok()
                    .flatten()
                    .map_or(sub.len(), |(_, l)| l);
                *rc.lock().unwrap() = cl;
            } else if wid == 1 {
                let mut dbuf = Vec::with_capacity(block_size);
                *rd.lock().unwrap() =
                    encode_dict_block_into(sub, Some(dict_ref), &mut dbuf, sub.len(), level)
                        .map(|(_, l)| l);
            } else {
                *rw.lock().unwrap() = Some(encode_wire_dict(&dict_ref.raw, level));
            }
        });
        pre_wire = rw.into_inner().unwrap();
        let col_len = rc.into_inner().unwrap();
        let dlz_len = rd
            .into_inner()
            .unwrap()
            .filter(|&l| l < col_len)
            .unwrap_or(col_len);
        (col_len, dlz_len, false, sub.len())
    };

    let est_dict_wire = dict.raw.len() / 2;
    let est_amort = (est_dict_wire * pilot_bytes) / input.len().max(1) + 6;
    let any_fast_lz_win = !any_structured && dlz_sum + est_amort < col_sum;
    if dlz_sum >= col_sum && !any_fast_lz_win {
        return None;
    }

    dict.wire_payload = pre_wire.unwrap_or_else(|| encode_wire_dict(&dict.raw, level));
    let exact_amort = (dict.wire_payload.len() * pilot_bytes) / input.len().max(1) + 6;
    if dlz_sum + exact_amort >= col_sum && !any_fast_lz_win {
        return None;
    }

    if !any_structured {
        dict.prefer_over_grammar = true;
        dict.try_structured = false;
    } else if dlz_sum + exact_amort < col_sum {
        dict.prefer_over_grammar = true;
        dict.try_structured = col_sum * 100 < dlz_sum * 125;
    } else {
        dict.prefer_over_grammar = false;
        dict.try_structured = true;
    }
    Some(dict)
}
