use std::cell::RefCell;

use crate::columns;

use super::{
    build_dict_tables, encode_dict_block_into, encode_wire_dict, read_u64, PreparedDict,
    MAX_DICT_RAW, MIN_DICT_RAW,
};

const SKETCH6_BITS: usize = 15;
const SKETCH6_SIZE: usize = 1 << SKETCH6_BITS; // 32,768 x u8 = 32 KiB (L1D resident)
const SKETCH16_BITS: usize = 16;
const SKETCH16_SIZE: usize = 1 << SKETCH16_BITS; // 65,536 x u8 = 64 KiB
const DENSE_BLK: usize = 256;
const SEG_LEN: usize = 64;
const SEG_STRIDE: usize = 32;
const MAX_CANDIDATES: usize = 4096;

struct TrainScratch {
    freq6: Box<[u8; SKETCH6_SIZE]>,
    seen6: Box<[u8; SKETCH6_SIZE]>,
    touched6: Vec<u16>,
    freq16: Box<[u8; SKETCH16_SIZE]>,
    seen16: Box<[u8; SKETCH16_SIZE]>,
    touched16: Vec<u16>,
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
    fn clear(&mut self) {
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
    static TRAIN_SCRATCH: RefCell<TrainScratch> = RefCell::new(TrainScratch::new());
}

#[inline(always)]
fn hash6(data: &[u8], pos: usize) -> usize {
    let w = read_u64(data, pos) & 0x0000_ffff_ffff_ffff;
    (w.wrapping_mul(0x9e37_79b1_85eb_ca87) >> (64 - SKETCH6_BITS)) as usize
}

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

/// Try macroscopic verbatim corpus extraction (up to `256 KiB`, e.g. periodic source/docs/templates).
/// Returns `Some(raw_dict)` ONLY when substantial multi-KiB verbatim sections repeat across windows.
fn try_macroscopic_dict(sc: &mut TrainScratch, input: &[u8], count: usize) -> Option<Vec<u8>> {
    let n = input.len();
    if n < 4 * 1024 * 1024 || count < 32 {
        return None;
    }
    let windows = count.clamp(32, 127);
    let win_len = 8192.min(n / windows);
    let usable_span = n.saturating_sub(win_len);
    let mut total_anc = 0usize;
    for w_idx in 0..windows {
        let tag = (w_idx + 1) as u8;
        let w_start = (usable_span * w_idx) / (windows - 1);
        let slice = &input[w_start..(w_start + win_len).min(n)];
        let mut p = 0usize;
        while p + 24 <= slice.len() {
            let mut m = anchor_mask16(read_u64(slice, p));
            while m != 0 {
                let k = (m.trailing_zeros() >> 3) as usize;
                m &= m - 1;
                total_anc += 1;
                bump_sketch16(sc, sketch_hash16(slice, p + k), tag);
            }
            p += 8;
        }
    }
    let fallback = total_anc < windows * 8;
    if fallback {
        for &idx in &sc.touched16 {
            unsafe {
                *sc.freq16.get_unchecked_mut(idx as usize) = 0;
                *sc.seen16.get_unchecked_mut(idx as usize) = 0;
            }
        }
        sc.touched16.clear();
        for w_idx in 0..windows {
            let tag = (w_idx + 1) as u8;
            let w_start = (usable_span * w_idx) / (windows - 1);
            let slice = &input[w_start..(w_start + 4096).min(n)];
            let mut p = 0usize;
            while p + 16 <= slice.len() {
                bump_sketch16(sc, sketch_hash16(slice, p), tag);
                p += 16;
            }
        }
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
                    unsafe { *sc.freq16.get_unchecked_mut(sketch_hash16(seg, p)) |= 0x80 };
                    p += 16;
                }
            } else {
                let mut p = 0usize;
                while p + 24 <= seg.len() {
                    let mut m = anchor_mask16(read_u64(seg, p));
                    while m != 0 {
                        let k = (m.trailing_zeros() >> 3) as usize;
                        m &= m - 1;
                        unsafe { *sc.freq16.get_unchecked_mut(sketch_hash16(seg, p + k)) |= 0x80 };
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

fn train_raw_dict(
    input: &[u8],
    block_size: usize,
    is_structured: bool,
    level: u8,
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
            if let Some(macro_raw) = try_macroscopic_dict(&mut sc, input, count) {
                return Some(macro_raw);
            }
        }

        // FastCover 6-byte shingle vocabulary discovery (`<= 32 KiB` default, `<= 112 KiB` high level).
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
        let mut cross_shingles = 0usize;

        for w_idx in 0..windows {
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
                        cross_shingles += 1;
                    }
                }
                p += step;
            }
        }

        if cross_shingles < 48 {
            return None;
        }

        let mut candidates: Vec<(u32, usize)> = Vec::with_capacity(MAX_CANDIDATES.min(windows * 64));
        for w_idx in 0..windows {
            let w_start = if windows > 1 {
                (usable_span * w_idx) / (windows - 1)
            } else {
                0
            };
            let w_end = (w_start + win_len).min(n);
            let mut s = w_start;
            while s + SEG_LEN <= w_end && candidates.len() < MAX_CANDIDATES {
                let seg = &input[s..s + SEG_LEN];
                let mut score = 0u32;
                let mut p = 0usize;
                while p + 8 <= SEG_LEN {
                    let c = unsafe { *sc.freq6.get_unchecked(hash6(seg, p)) };
                    if c >= 2 {
                        score += u32::from(c);
                    }
                    p += step;
                }
                if score >= min_score {
                    candidates.push((score, s));
                }
                s += SEG_STRIDE;
            }
        }
        if candidates.is_empty() {
            return None;
        }
        candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));

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
) -> Option<PreparedDict> {
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
    let raw = train_raw_dict(input, block_size, strongly_tabular, level)?;
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

    let mut tmp_dlz = Vec::with_capacity(block_size);
    let (col_sum, dlz_sum, any_structured, pilot_bytes) = if let Some(len0) = structured0 {
        let dlz0 = encode_dict_block_into(chunk0, Some(&dict), &mut tmp_dlz, len0, level)
            .map_or(len0, |(_, l)| l);
        let (mut cs, mut ds, mut pb) = (len0, dlz0, chunk0.len());
        let p1 = count / 3;
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
        let sub = chunk0;
        tmp_col.clear();
        let col_len = columns::encode_grammar_block_into_with_carry(sub, carry0, &mut tmp_col)
            .ok()
            .flatten()
            .map_or(sub.len(), |(_, l)| l);
        tmp_dlz.clear();
        let dlz_len = encode_dict_block_into(sub, Some(&dict), &mut tmp_dlz, col_len, level)
            .map_or(col_len, |(_, l)| l);
        (col_len, dlz_len, false, sub.len())
    };

    let est_dict_wire = dict.raw.len() / 2;
    let est_amort = (est_dict_wire * pilot_bytes) / input.len().max(1) + 6;
    let any_fast_lz_win = !any_structured && dlz_sum + est_amort < col_sum;
    if dlz_sum >= col_sum && !any_fast_lz_win {
        return None;
    }

    dict.wire_payload = encode_wire_dict(&dict.raw, level);
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
