use std::sync::atomic::AtomicU64;

use crate::columns;

use super::{
    encode_dict_block_into, encode_wire_dict, index_dict_tables, next_id, PreparedDict, HT4_SIZE,
    HT6_SIZE, MAX_DICT_RAW, MIN_DICT_RAW,
};

const SKETCH_BITS: usize = 14;
const SKETCH_SIZE: usize = 1 << SKETCH_BITS;
const SEG_LEN: usize = 64;
const SEG_STRIDE: usize = 32;
const MAX_CANDIDATES: usize = 4096;

#[inline(always)]
fn hash6(data: &[u8], pos: usize) -> usize {
    let w = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) & 0x0000_ffff_ffff_ffff;
    (w.wrapping_mul(0x9e37_79b1_85eb_ca87) >> (64 - SKETCH_BITS)) as usize
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
    let distinct = bmap[0].count_ones()
        + bmap[1].count_ones()
        + bmap[2].count_ones()
        + bmap[3].count_ones();
    distinct >= 196
}

fn train_raw_dict(input: &[u8], block_size: usize) -> Option<Vec<u8>> {
    let n = input.len();
    let count = n.div_ceil(block_size);
    if count < 3 || n < 24_576 {
        return None;
    }
    let mid = n / 2;
    if probe_looks_random(&input[mid..n.min(mid + 256)]) {
        return None;
    }

    let target_len = (n / 8).clamp(4096, MAX_DICT_RAW);
    let windows = count.clamp(4, 64);
    let win_len = (block_size / 4).clamp(1024, 2048);

    // Pass 1: Count cross-window 6-byte shingle occurrences across evenly spaced windows.
    let mut freq = vec![0u16; SKETCH_SIZE];
    let mut seen_win = vec![0u8; SKETCH_SIZE];
    let usable_span = n.saturating_sub(win_len);
    for w_idx in 0..windows {
        let tag = (w_idx + 1) as u8;
        let w_start = if windows > 1 {
            (usable_span * w_idx) / (windows - 1)
        } else {
            0
        };
        let slice = &input[w_start..(w_start + win_len).min(n)];
        if slice.len() < 16 {
            continue;
        }
        let mut p = 0usize;
        while p + 8 <= slice.len() {
            let h = hash6(slice, p);
            if seen_win[h] != tag {
                seen_win[h] = tag;
                freq[h] = freq[h].saturating_add(1);
            }
            p += 2;
        }
    }

    let cross_shingles = freq.iter().filter(|&&c| c >= 2).count();
    if cross_shingles < 48 {
        return None;
    }

    // Pass 2: Score candidate segments uniformly across all sampled windows.
    let mut candidates: Vec<(u32, usize, usize)> = Vec::with_capacity(MAX_CANDIDATES.min(windows * 64));
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
                let c = freq[hash6(seg, p)];
                if c >= 2 {
                    score += u32::from(c);
                }
                p += 2;
            }
            if score >= 10 {
                candidates.push((score, s, SEG_LEN));
            }
            s += SEG_STRIDE;
        }
    }
    if candidates.is_empty() {
        return None;
    }

    candidates.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    // Greedily pick uncovered segments (FastCover-style shingle zeroing to maximize vocabulary
    // diversity across the 32 KiB prefix dictionary).
    let mut selected: Vec<(usize, usize)> = Vec::with_capacity(target_len / SEG_LEN + 8);
    let mut total_bytes = 0usize;
    for &(_init_score, start, base_len) in &candidates {
        if total_bytes >= target_len {
            break;
        }
        let seg = &input[start..start + base_len];
        let mut live_score = 0u32;
        let mut live_hits = 0usize;
        let mut p = 0usize;
        while p + 8 <= base_len {
            let c = freq[hash6(seg, p)];
            if c >= 2 {
                live_score += u32::from(c);
                live_hits += 1;
            }
            p += 2;
        }
        if live_hits < 5 || live_score < 10 {
            continue;
        }
        let base_end = start + base_len;
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
            freq[hash6(chosen, z)] = 0;
            z += 2;
        }
        selected.push((start, start + take));
        total_bytes += take;
    }

    if total_bytes < MIN_DICT_RAW {
        return None;
    }

    // Place highest-scoring segments at the END of `raw` (smallest distance from block offset 0).
    let mut raw = Vec::with_capacity(total_bytes);
    for &(s, e) in selected.iter().rev() {
        raw.extend_from_slice(&input[s..e]);
    }
    Some(raw)
}

pub(super) fn build_prepared_dict(
    input: &[u8],
    block_size: usize,
    _id_gen: &AtomicU64,
) -> Option<PreparedDict> {
    let raw = train_raw_dict(input, block_size)?;
    let mut ht4 = Box::new([[0u32; 4]; HT4_SIZE]);
    let mut ht6 = Box::new([[0u32; 2]; HT6_SIZE]);
    index_dict_tables(&raw, &mut ht4, &mut ht6);

    // Defer wire dictionary compression until pilot blocks confirm `DICT_LZ` wins!
    let mut dict = PreparedDict {
        id: next_id(),
        raw,
        wire_payload: Vec::new(),
        prefer_over_grammar: false,
        try_structured: false,
        ht4,
        ht6,
    };

    let count = input.len().div_ceil(block_size);
    let p0 = count / 3;
    let p1 = ((count * 2) / 3).max(p0 + 1).min(count - 1);
    let pilots = if p0 == p1 { &[p0][..] } else { &[p0, p1][..] };

    let est_dict_wire = dict.raw.len() / 2;
    let est_amort = est_dict_wire.div_ceil(count) + 6;
    let mut tmp_col = Vec::with_capacity(block_size);
    let mut tmp_dlz = Vec::with_capacity(block_size);
    let mut col_sum = 0usize;
    let mut dlz_sum = 0usize;
    let mut any_structured = false;
    let mut any_fast_lz_win = false;

    for &idx in pilots {
        let st = idx * block_size;
        let en = (st + block_size).min(input.len());
        let chunk = &input[st..en];
        let carry = st > 0 && input[st - 1] != b'\n';
        tmp_col.clear();
        let (col_len, is_struct) =
            if let Ok(Some((_off, len))) = columns::encode_structured_only_into(chunk, carry, &mut tmp_col) {
                (len, true)
            } else {
                tmp_col.clear();
                match columns::encode_grammar_block_into_with_carry(chunk, carry, &mut tmp_col) {
                    Ok(Some((_off, len))) => (len, false),
                    _ => (chunk.len(), false),
                }
            };
        if is_struct {
            any_structured = true;
        }
        tmp_dlz.clear();
        let dlz_len = match encode_dict_block_into(chunk, Some(&dict), &mut tmp_dlz, col_len) {
            Some((_off, len)) => len,
            None => col_len,
        };
        if !is_struct && dlz_len + est_amort < col_len {
            any_fast_lz_win = true;
        }
        col_sum += col_len;
        dlz_sum += dlz_len;
    }

    if dlz_sum >= col_sum && !any_fast_lz_win {
        return None;
    }

    dict.wire_payload = encode_wire_dict(&dict.raw);
    let exact_amort = (dict.wire_payload.len().div_ceil(count) + 6) * pilots.len();
    if dlz_sum + exact_amort >= col_sum && !any_fast_lz_win {
        return None;
    }

    if !any_structured {
        dict.prefer_over_grammar = true;
        dict.try_structured = false;
    } else if dlz_sum + exact_amort < col_sum {
        dict.prefer_over_grammar = true;
        // Only run per-block structured probes if `COLUMN` came within 8% of `DICT_LZ` on pilots.
        dict.try_structured = col_sum * 100 < dlz_sum * 108;
    } else {
        dict.prefer_over_grammar = false;
        dict.try_structured = true;
    }
    Some(dict)
}
