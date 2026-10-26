#[inline(always)]
fn parse_decimal(value: &[u8]) -> Option<u64> {
    let len = value.len();
    if len == 0 {
        return None;
    }
    let ptr = value.as_ptr();
    let b0 = unsafe { *ptr };
    if len == 1 {
        let d0 = b0.wrapping_sub(b'0');
        return if d0 <= 9 { Some(d0 as u64) } else { None };
    }
    if b0 == b'0' {
        return None;
    }
    unsafe {
        if len <= 8 {
            if len >= 4 {
                let d_hi = (ptr.cast::<u32>().read_unaligned() ^ 0x3030_3030) as u64;
                let d_lo = (ptr.add(len - 4).cast::<u32>().read_unaligned() ^ 0x3030_3030) as u64;
                let d = ((d_hi << ((8 - len) << 3)) & 0xffff_ffff) | (d_lo << 32);
                if ((d.wrapping_add(0x0606_0606_0606_0606)) | d) & 0xf0f0_f0f0_f0f0_f0f0 != 0 {
                    return None;
                }
                let p2 =
                    ((d & 0x00ff_00ff_00ff_00ff) * 10) + ((d >> 8) & 0x00ff_00ff_00ff_00ff);
                let p4 = ((p2 & 0x0000_ffff_0000_ffff) * 100)
                    + ((p2 >> 16) & 0x0000_ffff_0000_ffff);
                return Some((p4 & 0xffff_ffff) * 10_000 + (p4 >> 32));
            }
            let d_hi = (ptr.cast::<u16>().read_unaligned() ^ 0x3030) as u32;
            let d_lo = (ptr.add(len - 2).cast::<u16>().read_unaligned() ^ 0x3030) as u32;
            let d = ((d_hi << ((4 - len) << 3)) & 0xffff) | (d_lo << 16);
            if ((d.wrapping_add(0x0606_0606)) | d) & 0xf0f0_f0f0 != 0 {
                return None;
            }
            let p2 = ((d & 0x00ff_00ff) * 10) + ((d >> 8) & 0x00ff_00ff);
            return Some(((p2 & 0xffff) * 100 + (p2 >> 16)) as u64);
        }
        if len <= 19 {
            let mut result = parse_dec8_at(ptr)?;
            let mut pos = 8usize;
            if pos + 8 <= len {
                result = result * 100_000_000 + parse_dec8_at(ptr.add(pos))?;
                pos += 8;
            }
            if pos + 4 <= len {
                result = result * 10_000 + parse_dec4_at(ptr.add(pos))?;
                pos += 4;
            }
            while pos < len {
                let d = (*ptr.add(pos)).wrapping_sub(b'0');
                if d > 9 {
                    return None;
                }
                result = result * 10 + (d as u64);
                pos += 1;
            }
            return Some(result);
        }
    }
    let mut result = 0u64;
    for &byte in value {
        let d = byte.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        result = result.checked_mul(10)?.checked_add(u64::from(d))?;
    }
    Some(result)
}

#[inline(always)]
fn raw_model_size(values: &[&[u8]]) -> usize {
    let mut size = 1usize;
    for value in values {
        let len = value.len();
        size += if len < 0x80 { 1 + len } else { uleb_len(len) + len };
    }
    size
}

fn pack_raw_model(values: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw_model_size(values));
    out.push(MODEL_RAW);
    for value in values {
        put_bytes(&mut out, value);
    }
    out
}

#[inline(always)]
fn hash_slice(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    let h = unsafe {
        if len >= 16 {
            let a = ptr.cast::<u64>().read_unaligned();
            let b = ptr.add(len - 8).cast::<u64>().read_unaligned();
            let c = ptr.add((len >> 1) - 4).cast::<u64>().read_unaligned();
            let d = ptr.add(((len * 3) >> 2) - 4).cast::<u64>().read_unaligned();
            a ^ b.rotate_left(17) ^ c.rotate_left(33) ^ d.rotate_left(49) ^ (len as u64)
        } else if len >= 8 {
            let a = ptr.cast::<u64>().read_unaligned();
            let b = ptr.add(len - 8).cast::<u64>().read_unaligned();
            a ^ b.rotate_left(31) ^ (len as u64)
        } else if len >= 4 {
            let a = ptr.cast::<u32>().read_unaligned() as u64;
            let b = ptr.add(len - 4).cast::<u32>().read_unaligned() as u64;
            a ^ (b << 32) ^ (len as u64)
        } else if len > 0 {
            let a = *ptr as u64;
            let b = *ptr.add(len >> 1) as u64;
            let c = *ptr.add(len - 1) as u64;
            a | (b << 8) | (c << 16) | ((len as u64) << 24)
        } else {
            0
        }
    };
    // Fold the high product bits down: callers index with `hash & mask`, and the low
    // bits of a multiplicative hash only depend on the low input bits.
    let m = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    m ^ (m >> 29)
}

#[derive(Default)]
struct ModelScratch {
    parsed_u64: Vec<u64>,
    seg_tokens: Vec<u8>,
    seg_decimals: Vec<u64>,
    seg_pads: Vec<u16>,
    epoch_prefixes: Vec<u8>,
    epoch_flags: Vec<u8>,
    epoch_explicit: Vec<u64>,
    front_buf: Vec<u8>,
    front_lz: Vec<u8>,
    raw_lens: Vec<u8>,
    raw_stream: Vec<u8>,
    raw_lz: Vec<u8>,
}

thread_local! {
    static MODEL_SCRATCH: std::cell::RefCell<ModelScratch> =
        std::cell::RefCell::new(ModelScratch::default());
}

#[inline(always)]
fn common_prefix_suffix(prev: &[u8], cur: &[u8]) -> (usize, usize) {
    let max_lcp = prev.len().min(cur.len());
    let mut lcp = 0usize;
    let p_ptr = prev.as_ptr();
    let c_ptr = cur.as_ptr();
    let mut mismatched = false;
    while lcp + 8 <= max_lcp {
        let a = unsafe { p_ptr.add(lcp).cast::<u64>().read_unaligned().to_le() };
        let b = unsafe { c_ptr.add(lcp).cast::<u64>().read_unaligned().to_le() };
        let diff = a ^ b;
        if diff == 0 {
            lcp += 8;
        } else {
            lcp += (diff.trailing_zeros() >> 3) as usize;
            mismatched = true;
            break;
        }
    }
    if !mismatched {
        while lcp < max_lcp && unsafe { *p_ptr.add(lcp) == *c_ptr.add(lcp) } {
            lcp += 1;
        }
    }
    let max_lcs = (prev.len() - lcp).min(cur.len() - lcp);
    let mut lcs = 0usize;
    let prev_len = prev.len();
    let cur_len = cur.len();
    let mut lcs_mismatched = false;
    while lcs + 8 <= max_lcs {
        let a = unsafe { p_ptr.add(prev_len - lcs - 8).cast::<u64>().read_unaligned().to_le() };
        let b = unsafe { c_ptr.add(cur_len - lcs - 8).cast::<u64>().read_unaligned().to_le() };
        let diff = a ^ b;
        if diff == 0 {
            lcs += 8;
        } else {
            lcs += (diff.leading_zeros() >> 3) as usize;
            lcs_mismatched = true;
            break;
        }
    }
    if !lcs_mismatched {
        while lcs < max_lcs
            && unsafe { *p_ptr.add(prev_len - 1 - lcs) == *c_ptr.add(cur_len - 1 - lcs) }
        {
            lcs += 1;
        }
    }
    (lcp, lcs)
}

#[inline(always)]
fn dict_words_and_hash(bytes: &[u8]) -> (u64, u64, u64) {
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    let (w0, w1, raw_h) = unsafe {
        if len > 16 {
            let a = ptr.cast::<u64>().read_unaligned();
            let b = ptr.add(len - 8).cast::<u64>().read_unaligned();
            let c = ptr.add((len >> 1) - 4).cast::<u64>().read_unaligned();
            let d = ptr.add(((len * 3) >> 2) - 4).cast::<u64>().read_unaligned();
            (
                a,
                b,
                a ^ b.rotate_left(17) ^ c.rotate_left(33) ^ d.rotate_left(49) ^ (len as u64),
            )
        } else if len >= 8 {
            let a = ptr.cast::<u64>().read_unaligned();
            let b = ptr.add(len - 8).cast::<u64>().read_unaligned();
            (a, b, a ^ b.rotate_left(31) ^ (len as u64))
        } else if len >= 4 {
            let a = ptr.cast::<u32>().read_unaligned() as u64;
            let b = ptr.add(len - 4).cast::<u32>().read_unaligned() as u64;
            let w0 = a | (b << 32);
            (w0, 0, w0 ^ (len as u64))
        } else if len > 0 {
            let a = *ptr as u64;
            let b = *ptr.add(len >> 1) as u64;
            let c = *ptr.add(len - 1) as u64;
            let w0 = a | (b << 8) | (c << 16);
            (w0, 0, w0 | ((len as u64) << 24))
        } else {
            (0, 0, 0)
        }
    };
    let m = raw_h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (w0, w1, m ^ (m >> 29))
}

fn pack_dictionary_model_mode(
    values: &[&[u8]],
    limit: usize,
    allow_front: bool,
) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 3 >= limit {
        return None;
    }
    let first = values[0];
    let mut first_diff = 1usize;
    while first_diff < n && slice_eq_fast(values[first_diff], first) {
        first_diff += 1;
    }
    if first_diff == n {
        // 1-entry constant column (u == 1, width == 0):
        let size = 1 + 1 + uleb_len(first.len()) + first.len() + 1;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DICTIONARY);
        out.push(1);
        put_bytes(&mut out, first);
        out.push(0);
        return Some(out);
    }

    // At least 2 distinct entries exist (`first` and `second`), so `u >= 2`, `min_width >= 1`:
    let second = values[first_diff];
    let mut dict_bytes_sum =
        uleb_len(first.len()) + first.len() + uleb_len(second.len()) + second.len();
    let min_second_cost = if allow_front {
        3
    } else {
        uleb_len(second.len()) + second.len()
    };
    if 3 + uleb_len(first.len()) + first.len() + min_second_cost + (n + 7) / 8 >= limit {
        return None;
    }

    // Fast open-addressed hash table dynamically sized to `(n.next_power_of_two() * 2).clamp(32, 512)`
    // so small columns (`n <= 64`) don't zero a 1 KB table.
    let ht_cap = (n.next_power_of_two() * 2).clamp(32, 512);
    let ht_mask = ht_cap - 1;
    let mut ht_hashes = [core::mem::MaybeUninit::<u64>::uninit(); 512];
    let mut ht_slots_uninit = [core::mem::MaybeUninit::<u16>::uninit(); 512];
    let ht_slots: &mut [u16] = unsafe {
        let ptr = ht_slots_uninit.as_mut_ptr().cast::<u16>();
        let s = std::slice::from_raw_parts_mut(ptr, ht_cap);
        s.fill(u16::MAX);
        s
    };
    let mut unique = [core::mem::MaybeUninit::<&[u8]>::uninit(); 256];
    let mut unique_w0 = [core::mem::MaybeUninit::<u64>::uninit(); 256];
    let mut unique_w1 = [core::mem::MaybeUninit::<u64>::uninit(); 256];
    let mut unique_lens = [core::mem::MaybeUninit::<u32>::uninit(); 256];
    let (w0_0, w1_0, h0) = dict_words_and_hash(first);
    let (w0_1, w1_1, h1) = dict_words_and_hash(second);
    unique[0].write(first);
    unique_w0[0].write(w0_0);
    unique_w1[0].write(w1_0);
    unique_lens[0].write(first.len() as u32);
    unique[1].write(second);
    unique_w0[1].write(w0_1);
    unique_w1[1].write(w1_1);
    unique_lens[1].write(second.len() as u32);
    let pos0 = (h0 as usize) & ht_mask;
    ht_hashes[pos0].write(h0);
    ht_slots[pos0] = 0;
    let mut pos1 = (h1 as usize) & ht_mask;
    if pos1 == pos0 {
        pos1 = (pos1 + 1) & ht_mask;
    }
    ht_hashes[pos1].write(h1);
    ht_slots[pos1] = 1;
    let mut unique_len = 2usize;

    let mut raw_ids_inline = [core::mem::MaybeUninit::<u8>::uninit(); 2048];
    let mut raw_ids_overflow: Vec<u8> = if n > 2048 {
        let mut v = Vec::with_capacity(n);
        v.resize(first_diff, 0);
        v.push(1);
        v
    } else {
        unsafe {
            std::ptr::write_bytes(raw_ids_inline.as_mut_ptr().cast::<u8>(), 0, first_diff);
        }
        raw_ids_inline[first_diff].write(1);
        Vec::new()
    };
    let mut overflowed_small_ht = false;
    let mut last_w0 = w0_1;
    let mut last_w1 = w1_1;
    let mut last_len = second.len() as u32;
    let mut last_id = 1u8;

    for (row_idx, &val) in values.iter().enumerate().skip(first_diff + 1) {
        let vlen = val.len();
        let (w0, w1, h) = dict_words_and_hash(val);
        let id = if vlen <= 16 && vlen as u32 == last_len && w0 == last_w0 && w1 == last_w1 {
            last_id
        } else {
            let mut pos = (h as usize) & ht_mask;
            let found_id = loop {
                let slot_id = unsafe { *ht_slots.get_unchecked(pos) };
                if slot_id == u16::MAX {
                    let new_id = unique_len;
                    dict_bytes_sum += uleb_len(vlen) + vlen;
                    let u = new_id + 1;
                    let min_width = bit_width(new_id as u64);
                    let min_packed = (n * (min_width as usize) + 7) / 8;
                    let est_dict = if allow_front && new_id < 256 {
                        uleb_len(first.len()) + first.len() + (u - 1) * 3
                    } else {
                        dict_bytes_sum
                    };
                    if 1 + uleb_len(u) + est_dict + 1 + min_packed >= limit {
                        return None;
                    }
                    if new_id >= 256 {
                        overflowed_small_ht = true;
                        break 0;
                    }
                    unique[new_id].write(val);
                    unique_w0[new_id].write(w0);
                    unique_w1[new_id].write(w1);
                    unique_lens[new_id].write(vlen as u32);
                    unique_len = u;
                    ht_hashes[pos].write(h);
                    unsafe {
                        *ht_slots.get_unchecked_mut(pos) = new_id as u16;
                    }
                    break new_id as u8;
                }
                let idx = slot_id as usize;
                let s_len = unsafe { unique_lens.get_unchecked(idx).assume_init() };
                if s_len == vlen as u32 {
                    let s_w0 = unsafe { unique_w0.get_unchecked(idx).assume_init() };
                    let s_w1 = unsafe { unique_w1.get_unchecked(idx).assume_init() };
                    if s_w0 == w0 && s_w1 == w1 {
                        if vlen <= 16 {
                            break slot_id as u8;
                        }
                        let slot_hash = unsafe { ht_hashes.get_unchecked(pos).assume_init() };
                        if slot_hash == h {
                            let slot_slice = unsafe { unique.get_unchecked(idx).assume_init() };
                            if slice_eq_fast(
                                unsafe { slot_slice.get_unchecked(8..vlen - 8) },
                                unsafe { val.get_unchecked(8..vlen - 8) },
                            ) {
                                break slot_id as u8;
                            }
                        }
                    }
                }
                pos = (pos + 1) & ht_mask;
            };
            last_w0 = w0;
            last_w1 = w1;
            last_len = vlen as u32;
            last_id = found_id;
            found_id
        };
        if overflowed_small_ht {
            break;
        }
        if n <= 2048 {
            raw_ids_inline[row_idx].write(id);
        } else {
            raw_ids_overflow.push(id);
        }
    }

    if !overflowed_small_ht {
        let u = unique_len;
        if u == n && (!allow_front || n > 256 || first.len() < 8) {
            return None;
        }
        let width = bit_width((u - 1) as u64);
        let packed_len = bit_bytes(n, width).ok()?;
        let unique_init: &[&[u8]] =
            unsafe { std::slice::from_raw_parts(unique.as_ptr().cast::<&[u8]>(), u) };
        let raw_ids_slice: &mut [u8] = if n <= 2048 {
            unsafe { std::slice::from_raw_parts_mut(raw_ids_inline.as_mut_ptr().cast::<u8>(), n) }
        } else {
            &mut raw_ids_overflow[..]
        };
        let already_sorted = (1..u).all(|i| unique_init[i - 1] < unique_init[i]);
        if already_sorted {
            return build_front_dict_packet(
                u,
                n,
                width,
                packed_len,
                dict_bytes_sum,
                unique_init,
                raw_ids_slice,
                limit,
                allow_front,
            );
        }
        #[inline(always)]
        fn prefix_key_be(s: &[u8]) -> u64 {
            let len = s.len();
            if len >= 8 {
                unsafe { s.as_ptr().cast::<u64>().read_unaligned().to_be() }
            } else {
                unsafe { (read_u128_partial(s.as_ptr(), len) as u64).swap_bytes() }
            }
        }
        let mut order_uninit = [core::mem::MaybeUninit::<(u64, u8)>::uninit(); 256];
        let order_slice: &mut [(u64, u8)] = unsafe {
            let ptr = order_uninit.as_mut_ptr().cast::<(u64, u8)>();
            for i in 0..u {
                let s = *unique_init.get_unchecked(i);
                *ptr.add(i) = (prefix_key_be(s), i as u8);
            }
            std::slice::from_raw_parts_mut(ptr, u)
        };
        order_slice.sort_unstable_by(|&(ka, a), &(kb, b)| {
            if ka != kb {
                ka.cmp(&kb)
            } else {
                unsafe {
                    let sa = *unique_init.get_unchecked(a as usize);
                    let sb = *unique_init.get_unchecked(b as usize);
                    if sa.len() >= 8 && sb.len() >= 8 {
                        sa.get_unchecked(8..).cmp(sb.get_unchecked(8..))
                    } else {
                        sa.cmp(sb)
                    }
                }
            }
        });
        let mut sorted_unique_uninit = [core::mem::MaybeUninit::<&[u8]>::uninit(); 256];
        let mut rank_of_uninit = [core::mem::MaybeUninit::<u8>::uninit(); 256];
        let sorted_ptr = sorted_unique_uninit.as_mut_ptr().cast::<&[u8]>();
        let rank_of_ptr = rank_of_uninit.as_mut_ptr().cast::<u8>();
        for (rank, &(_, orig_id)) in order_slice.iter().enumerate() {
            unsafe {
                *sorted_ptr.add(rank) = *unique_init.get_unchecked(orig_id as usize);
                *rank_of_ptr.add(orig_id as usize) = rank as u8;
            }
        }
        let sorted_entries: &[&[u8]] =
            unsafe { std::slice::from_raw_parts(sorted_ptr, u) };
        for id in raw_ids_slice.iter_mut() {
            *id = unsafe { *rank_of_ptr.add(*id as usize) };
        }

        return build_front_dict_packet(
            u,
            n,
            width,
            packed_len,
            dict_bytes_sum,
            sorted_entries,
            raw_ids_slice,
            limit,
            allow_front,
        );
    }

    // Fallback for dictionaries with > 256 unique entries (`MODEL_DICTIONARY` only):
    if n <= 256 {
        return None;
    }
    let min_packed_len = (n * 9 + 7) >> 3;
    if 4 + min_packed_len >= limit {
        return None;
    }
    // Hash dedup (no full sort): exact size check first, then sort only the `u`
    // unique entries and map each row to its rank via the dedup id.
    let cap = (n * 2).next_power_of_two();
    let mask = cap - 1;
    let hshift = 64 - cap.trailing_zeros();
    let mut slots: Vec<u32> = vec![u32::MAX; cap];
    let mut uniq: Vec<&[u8]> = Vec::with_capacity(n.min(1024));
    let mut row_ids: Vec<u32> = Vec::with_capacity(n);
    let mut plain_dict_bytes = 0usize;
    for &val in values {
        // Multiplicative hash: take the high bits (low bits only see low input bits).
        let mut pos = (hash_slice(val) >> hshift) as usize;
        let id = loop {
            let s = slots[pos];
            if s == u32::MAX {
                let id = uniq.len() as u32;
                slots[pos] = id;
                uniq.push(val);
                plain_dict_bytes += uleb_len(val.len()) + val.len();
                if 3 + plain_dict_bytes + min_packed_len >= limit {
                    return None;
                }
                break id;
            }
            if slice_eq_fast(uniq[s as usize], val) {
                break s;
            }
            pos = (pos + 1) & mask;
        };
        row_ids.push(id);
    }
    let u = uniq.len();
    if u == n {
        return None;
    }
    let width = bit_width((u - 1) as u64);
    let packed_len = bit_bytes(n, width).ok()?;
    let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
    if size >= limit {
        return None;
    }
    let mut order: Vec<u32> = (0..u as u32).collect();
    order.sort_unstable_by(|&a, &b| uniq[a as usize].cmp(uniq[b as usize]));
    let mut rank = vec![0u32; u];
    for (r, &id) in order.iter().enumerate() {
        rank[id as usize] = r as u32;
    }
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_DICTIONARY);
    put_uleb(&mut out, u);
    for &id in &order {
        put_bytes(&mut out, uniq[id as usize]);
    }
    out.push(width);
    push_bit_values(
        &mut out,
        n,
        width,
        row_ids.iter().map(|&id| rank[id as usize] as u64),
    );
    Some(out)
}

