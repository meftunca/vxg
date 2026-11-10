use std::sync::Mutex;

use crate::pool::parallel_run_checked;

use super::{
    read_u64,
    train::{TrainScratch, DENSE_BLK, SKETCH16_BITS, TRAIN_SCRATCH},
    MAX_DICT_RAW,
};

#[inline(always)]
fn sketch_hash16(data: &[u8], pos: usize) -> usize {
    let a = read_u64(data, pos).wrapping_mul(0x9e37_79b1_85eb_ca87);
    let b = read_u64(data, pos + 8).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    ((a ^ b.rotate_left(23)).wrapping_mul(0x1656_67b1_9e37_79f9) >> (64 - SKETCH16_BITS)) as usize
}

/// 100% shift-invariant SWAR 1-in-16 byte anchor (`((b ^ (b >> 3)) & 15) == 11`),
/// yielding ~16 anchors per 256-byte block while keeping `sketch16` load `<= 0.5`.
#[inline(always)]
fn anchor_mask16(w: u64) -> u64 {
    let x = w ^ ((w >> 3) & 0x0f0f_0f0f_0f0f_0f0f);
    x & (x >> 1) & (!x >> 2) & (x >> 3) & 0x0101_0101_0101_0101
}

#[inline(always)]
fn bump_sketch16(sc: &mut TrainScratch, h: usize, tag: u8) {
    let sw = unsafe { sc.seen16.get_unchecked_mut(h) };
    if *sw != tag {
        *sw = tag;
        let f = unsafe { sc.freq16.get_unchecked_mut(h) };
        let prev = *f;
        if prev == 0 {
            sc.touched16.push(h as u16);
        }
        *f = prev.saturating_add(1);
    }
}

#[inline(always)]
fn mark_used16(sc: &mut TrainScratch, h: usize) {
    let f = unsafe { sc.freq16.get_unchecked_mut(h) };
    if *f != 0 {
        *f |= 0x80;
    }
}

fn scan_macro_range(
    sc: &mut TrainScratch,
    input: &[u8],
    windows: usize,
    win_len: usize,
    usable_span: usize,
    w_lo: usize,
    w_hi: usize,
    fallback: bool,
) -> usize {
    let n = input.len();
    let mut anc = 0usize;
    for w_idx in w_lo..w_hi {
        let tag = (w_idx + 1) as u8;
        let w_start = (usable_span * w_idx) / (windows - 1);
        if fallback {
            let slice = &input[w_start..(w_start + 4096).min(n)];
            let mut p = 0usize;
            while p + 16 <= slice.len() {
                bump_sketch16(sc, sketch_hash16(slice, p), tag);
                p += 16;
            }
        } else {
            let slice = &input[w_start..(w_start + win_len).min(n)];
            let mut p = 0usize;
            while p + 24 <= slice.len() {
                let mut m = anchor_mask16(read_u64(slice, p));
                while m != 0 {
                    let k = (m.trailing_zeros() >> 3) as usize;
                    m &= m - 1;
                    anc += 1;
                    bump_sketch16(sc, sketch_hash16(slice, p + k), tag);
                }
                p += 8;
            }
        }
    }
    anc
}

fn populate_macro_sketch(
    sc: &mut TrainScratch,
    input: &[u8],
    windows: usize,
    win_len: usize,
    usable_span: usize,
    workers: usize,
    fallback: bool,
) -> usize {
    let w_threads = workers.min(8).min(windows / 8);
    if w_threads <= 1 {
        return scan_macro_range(sc, input, windows, win_len, usable_span, 0, windows, fallback);
    }
    let partials: Vec<Mutex<(usize, Vec<(u16, u8)>)>> = (0..w_threads)
        .map(|_| Mutex::new((0, Vec::new())))
        .collect();
    let sc0_anc = Mutex::new(0usize);
    let sc0_ptr = std::sync::atomic::AtomicPtr::new(sc as *mut TrainScratch);

    let _ = parallel_run_checked(w_threads, |wid| {
        let w_lo = (windows * wid) / w_threads;
        let w_hi = (windows * (wid + 1)) / w_threads;
        if wid == 0 {
            let sc0 = unsafe { &mut *sc0_ptr.load(std::sync::atomic::Ordering::Relaxed) };
            let a = scan_macro_range(sc0, input, windows, win_len, usable_span, w_lo, w_hi, fallback);
            *sc0_anc.lock().unwrap() = a;
        } else {
            TRAIN_SCRATCH.with(|cell| {
                let mut lsc = cell.borrow_mut();
                lsc.clear();
                let a = scan_macro_range(&mut lsc, input, windows, win_len, usable_span, w_lo, w_hi, fallback);
                let mut pairs = Vec::with_capacity(lsc.touched16.len());
                for &idx in &lsc.touched16 {
                    pairs.push((idx, unsafe { *lsc.freq16.get_unchecked(idx as usize) }));
                }
                lsc.clear();
                *partials[wid].lock().unwrap() = (a, pairs);
            });
        }
    });

    let mut total_anc = *sc0_anc.lock().unwrap();
    for slot in partials.into_iter().skip(1) {
        let (a, pairs) = slot.into_inner().unwrap();
        total_anc += a;
        for (idx, add) in pairs {
            let f = unsafe { sc.freq16.get_unchecked_mut(idx as usize) };
            if *f == 0 {
                sc.touched16.push(idx);
            }
            *f = f.saturating_add(add);
        }
    }
    total_anc
}

/// Try macroscopic verbatim corpus extraction (up to `256 KiB`, e.g. periodic source/docs/templates).
/// Returns `Some(raw_dict)` ONLY when substantial multi-KiB verbatim sections repeat across windows.
pub(super) fn try_macroscopic_dict(
    sc: &mut TrainScratch,
    input: &[u8],
    count: usize,
    workers: usize,
) -> Option<Vec<u8>> {
    let n = input.len();
    if n < 4 * 1024 * 1024 || count < 32 {
        return None;
    }
    let windows = count.clamp(32, 127);
    let win_len = 8192.min(n / windows);
    let usable_span = n.saturating_sub(win_len);
    let total_anc = populate_macro_sketch(sc, input, windows, win_len, usable_span, workers, false);
    let fallback = total_anc < windows * 8;
    if fallback {
        for &idx in &sc.touched16 {
            unsafe {
                *sc.freq16.get_unchecked_mut(idx as usize) = 0;
                *sc.seen16.get_unchecked_mut(idx as usize) = 0;
            }
        }
        sc.touched16.clear();
        let _ = populate_macro_sketch(sc, input, windows, win_len, usable_span, workers, true);
    }

    let min_freq = 2u8;
    let min_anc = if fallback { 6 } else { 4 };
    let dense_budget = (n / 16).clamp(32_768, MAX_DICT_RAW);
    let prefix_scan = n.min(384 * 1024);

    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut total_bytes = 0usize;
    let mut s = 0usize;
    while s + DENSE_BLK <= prefix_scan && total_bytes + DENSE_BLK <= dense_budget {
        let seg = &input[s..s + DENSE_BLK];
        let (mut anc, mut init_hits, mut live_hits) = (0usize, 0usize, 0usize);
        if fallback {
            let mut p = 0usize;
            while p + 16 <= seg.len() {
                anc += 1;
                let raw_c = unsafe { *sc.freq16.get_unchecked(sketch_hash16(seg, p)) };
                if (raw_c & 0x7f) >= min_freq {
                    init_hits += 1;
                    if (raw_c & 0x80) == 0 {
                        live_hits += 1;
                    }
                }
                p += 16;
            }
        } else {
            let mut p = 0usize;
            while p + 24 <= seg.len() {
                let mut m = anchor_mask16(read_u64(seg, p));
                while m != 0 {
                    let k = (m.trailing_zeros() >> 3) as usize;
                    m &= m - 1;
                    anc += 1;
                    let raw_c = unsafe { *sc.freq16.get_unchecked(sketch_hash16(seg, p + k)) };
                    if (raw_c & 0x7f) >= min_freq {
                        init_hits += 1;
                        if (raw_c & 0x80) == 0 {
                            live_hits += 1;
                        }
                    }
                }
                p += 8;
            }
        }

        if anc >= min_anc && init_hits * 3 >= anc * 2 && live_hits >= 1 {
            if fallback {
                let mut p = 0usize;
                while p + 16 <= seg.len() {
                    mark_used16(sc, sketch_hash16(seg, p));
                    p += 16;
                }
            } else {
                let mut p = 0usize;
                while p + 24 <= seg.len() {
                    let mut m = anchor_mask16(read_u64(seg, p));
                    while m != 0 {
                        let k = (m.trailing_zeros() >> 3) as usize;
                        m &= m - 1;
                        mark_used16(sc, sketch_hash16(seg, p + k));
                    }
                    p += 8;
                }
            }
            if let Some(last) = spans.last_mut() {
                if s <= last.1 + 2048 && total_bytes + (s + DENSE_BLK - last.1) <= dense_budget {
                    total_bytes += s + DENSE_BLK - last.1;
                    last.1 = s + DENSE_BLK;
                    s += DENSE_BLK;
                    continue;
                }
            }
            spans.push((s, s + DENSE_BLK));
            total_bytes += DENSE_BLK;
        }
        s += DENSE_BLK;
        if s >= 16_384 && s < 16_640 && total_bytes < 2048 {
            if prefix_scan > 81_920 {
                s = 65_536;
            } else {
                break;
            }
        } else if s >= 81_920 && s < 82_176 && total_bytes < 8192 {
            break;
        }
    }

    if total_bytes <= 32_768 {
        return None;
    }
    if spans[0].0 <= 4096 && n >= 256 {
        let head = read_u64(input, 0);
        let mut p = 32_768usize;
        let search_lim = dense_budget.min(n - 128);
        while p <= search_lim {
            if read_u64(input, p) == head && input[p..p + 128] == input[..128] {
                return Some(input[..p].to_vec());
            }
            p += 1;
        }
    }
    let mut raw = Vec::with_capacity(total_bytes);
    for &(st, en) in &spans {
        raw.extend_from_slice(&input[st..en]);
    }
    Some(raw)
}
