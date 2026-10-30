#[inline(never)]
fn refill_json_cache(data: &[u8], w: usize, cache: &mut (usize, u64, u64)) -> bool {
    let base = w * 64;
    let len = data.len();
    if base + 64 <= len {
        let q = unsafe { json_quote_mask64(data.as_ptr().add(base)) };
        cache.0 = w;
        cache.1 = q;
        true
    } else if base < len {
        let rem = len - base;
        let mut tail = [0u8; 64];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(base), tail.as_mut_ptr(), rem);
            let q = json_quote_mask64(tail.as_ptr());
            cache.0 = w;
            cache.1 = q;
        }
        true
    } else {
        false
    }
}

#[inline(never)]
fn next_json_string_slow(
    data: &[u8],
    mut from: usize,
    cache: &mut (usize, u64, u64),
) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    loop {
        if from >= len {
            return None;
        }
        let w = from >> 6;
        if cache.0 != w && !refill_json_cache(data, w, cache) {
            return None;
        }
        let m = cache.1 >> (from & 63);
        if m != 0 {
            let end = from + m.trailing_zeros() as usize;
            if end >= len {
                return None;
            }
            let hit = unsafe { *ptr.add(end) };
            if hit == b'"' {
                return Some(end + 1);
            }
            if hit < 0x20 || end + 1 >= len || unsafe { *ptr.add(end + 1) } < 0x20 {
                return None;
            }
            from = end + 2;
        } else {
            from = (w + 1) * 64;
        }
    }
}

#[inline(always)]
fn scan_json_string_cached(
    data: &[u8],
    pos: usize,
    cache: &mut (usize, u64, u64),
) -> Option<usize> {
    let from = pos + 1;
    let w = from >> 6;
    if cache.0 != w && !refill_json_cache(data, w, cache) {
        return None;
    }
    let m = cache.1 >> (from & 63);
    if m != 0 {
        let end = from + m.trailing_zeros() as usize;
        if end < data.len() && unsafe { *data.as_ptr().add(end) } == b'"' {
            return Some(end + 1);
        }
        return next_json_string_slow(data, end, cache);
    }
    next_json_string_slow(data, (w + 1) * 64, cache)
}

/// Delimited (CSV/TSV/PSV) row matcher at `data[start..]`. With on-demand 64-byte
/// structural bitmasks each field boundary is one `trailing_zeros`; whenever the fast
/// path cannot *prove* success (escape, control byte, long value, any mismatch) it
/// defers to the scalar reference matcher, so results are identical by construction.
#[inline(always)]
fn try_match_delimited_stream<'a>(
    data: &'a [u8],
    start: usize,
    segments: &[&'a [u8]],
    delim: u8,
    cache: &mut (usize, u64, u64),
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let len = data.len();
    let ptr = data.as_ptr();
    'fast: {
        let mut pos = start;
        for slot in 0..slots {
            if pos >= len {
                break 'fast;
            }
            let first = unsafe { *ptr.add(pos) };
            let val_end = if first == b'"' {
                let Some(end) = next_delim_bit::<true>(data, delim, pos + 1, cache) else {
                    break 'fast;
                };
                if end >= len || unsafe { *ptr.add(end) } != b'"' {
                    break 'fast;
                }
                end + 1
            } else {
                let Some(end) = next_delim_bit::<false>(data, delim, pos, cache) else {
                    break 'fast;
                };
                if end == pos || end > len {
                    break 'fast;
                }
                end
            };
            unsafe {
                *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
            }
            pos = val_end;
            if slot + 1 < slots {
                if pos >= len || unsafe { *ptr.add(pos) } != delim {
                    break 'fast;
                }
                pos += 1;
            } else {
                if pos - start > 4096 {
                    break 'fast;
                }
                let last_seg = unsafe { *segments.get_unchecked(slots) };
                let next_end = pos + last_seg.len();
                if next_end > len
                    || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, last_seg)
                {
                    break 'fast;
                }
                pos = next_end;
            }
        }
        return Some(pos - start);
    }
    try_match_delimited_scalar(unsafe { data.get_unchecked(start..) }, segments, delim, out_values)
}

/// Scalar reference matcher for delimited rows (also the fallback of the bitmask path).
#[inline(never)]
fn try_match_delimited_scalar<'a>(
    rem: &'a [u8],
    segments: &[&'a [u8]],
    delim: u8,
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let relaxed = delim_is_relaxed(delim);
    let slots = segments.len() - 1;
    let rem_len = rem.len();
    let rem_ptr = rem.as_ptr();
    let mut pos = 0usize;
    for slot in 0..slots {
        if pos >= rem_len {
            return None;
        }
        let first = unsafe { *rem_ptr.add(pos) };
        let val_end = if first == b'"' {
            scan_string(rem, pos)?
        } else {
            let mut end = pos;
            while end < rem_len {
                let b = unsafe { *rem_ptr.add(end) };
                if b == delim
                    || b == b'"'
                    || if relaxed {
                        b < 0x20 || b == 0x7f
                    } else {
                        b.wrapping_sub(0x21) >= 0x5e
                    }
                {
                    break;
                }
                end += 1;
            }
            if end == pos {
                return None;
            }
            end
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
        }
        pos = val_end;
        if slot + 1 < slots {
            if pos >= rem_len || unsafe { *rem_ptr.add(pos) } != delim {
                return None;
            }
            pos += 1;
        } else {
            if pos > 4096 {
                return None;
            }
            let last_seg = unsafe { *segments.get_unchecked(slots) };
            let next_end = pos + last_seg.len();
            if next_end > rem_len
                || !slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, last_seg)
            {
                return None;
            }
            pos = next_end;
        }
    }
    Some(pos)
}

#[inline(always)]
fn try_match_logfmt_stream<'a>(
    data: &'a [u8],
    start: usize,
    segments: &[&'a [u8]],
    cache: &mut (usize, u64, u64),
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let len = data.len();
    let ptr = data.as_ptr();
    let seg0 = segments[0];
    if start + seg0.len() > len
        || !slice_eq_fast(unsafe { data.get_unchecked(start..start + seg0.len()) }, seg0)
    {
        return None;
    }
    'fast: {
        let mut pos = start + seg0.len();
        for slot in 0..slots {
            if pos >= len {
                break 'fast;
            }
            let first = unsafe { *ptr.add(pos) };
            let val_end = if first == b'"' {
                let Some(end) = next_delim_bit::<true>(data, b' ', pos + 1, cache) else {
                    break 'fast;
                };
                if end >= len || unsafe { *ptr.add(end) } != b'"' {
                    break 'fast;
                }
                end + 1
            } else {
                let Some(end) = next_delim_bit::<false>(data, b' ', pos, cache) else {
                    break 'fast;
                };
                if end == pos || end > len {
                    break 'fast;
                }
                end
            };
            unsafe {
                *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
            }
            pos = val_end;
            let next_seg = unsafe { *segments.get_unchecked(slot + 1) };
            let next_end = pos + next_seg.len();
            if next_end > len
                || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
            {
                break 'fast;
            }
            pos = next_end;
        }
        let last_len = segments[slots].len();
        if pos - start - last_len <= 4096 {
            return Some(pos - start);
        }
    }
    try_match_logfmt_scalar(unsafe { data.get_unchecked(start..) }, segments, out_values)
}

#[inline(never)]
fn try_match_logfmt_scalar<'a>(
    rem: &'a [u8],
    segments: &[&'a [u8]],
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let rem_len = rem.len();
    let rem_ptr = rem.as_ptr();
    let seg0 = segments[0];
    if rem_len < seg0.len() || !slice_eq_fast(&rem[..seg0.len()], seg0) {
        return None;
    }
    let mut pos = seg0.len();
    for slot in 0..slots {
        if pos >= rem_len {
            return None;
        }
        let first = unsafe { *rem_ptr.add(pos) };
        let val_end = if first == b'"' {
            scan_string(rem, pos)?
        } else {
            let mut end = pos;
            while end < rem_len {
                let b = unsafe { *rem_ptr.add(end) };
                if b <= b' ' || b == b'"' || b == 0x7f {
                    break;
                }
                end += 1;
            }
            if end == pos {
                return None;
            }
            end
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
        }
        pos = val_end;
        let next_seg = unsafe { *segments.get_unchecked(slot + 1) };
        let next_end = pos + next_seg.len();
        if next_end > rem_len
            || !slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, next_seg)
        {
            return None;
        }
        pos = next_end;
    }
    let last_len = segments[slots].len();
    (pos - last_len <= 4096).then_some(pos)
}

thread_local! {
    static ENCODE_VEC_POOL: std::cell::RefCell<Vec<Vec<&'static [u8]>>> =
        std::cell::RefCell::new(Vec::with_capacity(128));
}

/// Smallest capacity across a template's column vectors.
///
/// Columns come from a pool and can have different capacities, so the fast row loops must bound
/// their unchecked writes by the minimum over *all* slots, not by slot 0 alone (that was a heap
/// buffer overflow when slot 0 was larger than another slot).
#[inline(always)]
fn min_col_capacity(values: &[Vec<&[u8]>]) -> usize {
    values.iter().map(Vec::capacity).min().unwrap_or(0)
}

#[inline(always)]
fn take_col_vec<'a>(est_rows: usize) -> Vec<&'a [u8]> {
    let v = ENCODE_VEC_POOL.with(|cell| cell.borrow_mut().pop());
    match v {
        Some(mut vec_static) => {
            vec_static.clear();
            vec_static.reserve(est_rows);
            unsafe { core::mem::transmute::<Vec<&'static [u8]>, Vec<&'a [u8]>>(vec_static) }
        }
        None => Vec::with_capacity(est_rows),
    }
}

impl<'a> Drop for Template<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        ENCODE_VEC_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                for mut col in core::mem::take(&mut self.values) {
                    if pool.len() < 128 && col.capacity() >= 16 {
                        col.clear();
                        let vec_static = unsafe {
                            core::mem::transmute::<Vec<&'a [u8]>, Vec<&'static [u8]>>(col)
                        };
                        pool.push(vec_static);
                    }
                }
            }
        });
    }
}

#[inline(always)]
fn try_match_clean_stream<'a>(
    data: &'a [u8],
    start: usize,
    templates: &[Template<'a>],
    clean_nl_flags: &[bool],
    shared_segs: &[[u8; 16]; 16],
    max_shared: &[u8; 16],
    mut cur_tpl: usize,
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<(usize, usize)> {
    let tpl_count = templates.len();
    if !clean_nl_flags[cur_tpl] {
        return None;
    }
    let len = data.len();
    let ptr = data.as_ptr();
    if tpl_count == 1 {
        let segs = &templates[0].segments;
        let slots = segs.len() - 1;
        let seg0 = segs[0];
        if start + seg0.len() > len
            || !slice_eq_fast(unsafe { data.get_unchecked(start..start + seg0.len()) }, seg0)
        {
            return None;
        }
        let mut pos = start + seg0.len();
        for slot in 0..slots {
            if pos >= len {
                return None;
            }
            let first = unsafe { *ptr.add(pos) };
            let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
            let val_end = match first {
                b'"' => scan_string(data, pos)?,
                b'{' | b'[' => scan_compound(data, pos)?,
                _ if first <= b',' || first == b'}' || first == b']' => return None,
                _ => scan_primitive_end(ptr, len, pos),
            };
            unsafe {
                *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
            }
            pos = val_end;
            let next_end = pos + next_seg.len();
            if next_end > len
                || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
            {
                return None;
            }
            pos = next_end;
        }
        return Some((0, pos - start));
    }

    let seg0 = templates[cur_tpl].segments[0];
    let first_rem_byte = unsafe { *ptr.add(start) };
    if start + seg0.len() > len
        || first_rem_byte != unsafe { *seg0.as_ptr() }
        || !slice_eq_fast(unsafe { data.get_unchecked(start..start + seg0.len()) }, seg0)
    {
        if tpl_count > 16 {
            return None;
        }
        let mut found = None;
        for alt in 0..tpl_count {
            if alt != cur_tpl && clean_nl_flags[alt] && shared_segs[cur_tpl][alt] == 0 {
                let alt_seg0 = templates[alt].segments[0];
                if start + alt_seg0.len() <= len
                    && first_rem_byte == unsafe { *alt_seg0.as_ptr() }
                    && slice_eq_fast(
                        unsafe { data.get_unchecked(start..start + alt_seg0.len()) },
                        alt_seg0,
                    )
                {
                    found = Some(alt);
                    break;
                }
            }
        }
        cur_tpl = found?;
    }
    let mut segs = &templates[cur_tpl].segments;
    let mut slots = segs.len() - 1;
    let mut pos = start + segs[0].len();
    let mut slot = 0usize;
    loop {
        if slot >= slots {
            return Some((cur_tpl, pos - start));
        }
        if cur_tpl < 16 && slot >= max_shared[cur_tpl] as usize {
            while slot < slots {
                if pos >= len {
                    return None;
                }
                let first = unsafe { *ptr.add(pos) };
                let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
                let val_end = match first {
                    b'"' => scan_string(data, pos)?,
                    b'{' | b'[' => scan_compound(data, pos)?,
                    _ if first <= b',' || first == b'}' || first == b']' => return None,
                    _ => scan_primitive_end(ptr, len, pos),
                };
                unsafe {
                    *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
                }
                pos = val_end;
                let next_end = pos + next_seg.len();
                if next_end > len
                    || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
                {
                    return None;
                }
                pos = next_end;
                slot += 1;
            }
            return Some((cur_tpl, pos - start));
        }
        if pos >= len {
            return None;
        }
        let first = unsafe { *ptr.add(pos) };
        let val_end = match first {
            b'"' => scan_string(data, pos)?,
            b'{' | b'[' => scan_compound(data, pos)?,
            _ if first <= b',' || first == b'}' || first == b']' => return None,
            _ => scan_primitive_end(ptr, len, pos),
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
        }
        pos = val_end;
        let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
        let next_end = pos + next_seg.len();
        if next_end <= len
            && slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
        {
            pos = next_end;
            slot += 1;
            continue;
        }
        if tpl_count > 16 {
            return None;
        }
        let req_shared = (slot + 1) as u8;
        let mut switched = false;
        for alt in 0..tpl_count {
            if alt != cur_tpl && clean_nl_flags[alt] && shared_segs[cur_tpl][alt] == req_shared {
                let alt_seg = templates[alt].segments[slot + 1];
                let alt_end = pos + alt_seg.len();
                if alt_end <= len
                    && slice_eq_fast(unsafe { data.get_unchecked(pos..alt_end) }, alt_seg)
                {
                    cur_tpl = alt;
                    segs = &templates[cur_tpl].segments;
                    slots = segs.len() - 1;
                    pos = alt_end;
                    slot += 1;
                    switched = true;
                    break;
                }
            }
        }
        if !switched {
            return None;
        }
    }
}

thread_local! {
    static ENCODE_ROW_POOL: std::cell::RefCell<Vec<Row<'static>>> =
        std::cell::RefCell::new(Vec::with_capacity(512));
}

#[inline(always)]
fn take_row_vec<'a>(est_rows: usize) -> Vec<Row<'a>> {
    let mut vec_static = ENCODE_ROW_POOL.with(|cell| core::mem::take(&mut *cell.borrow_mut()));
    vec_static.clear();
    vec_static.reserve(est_rows);
    unsafe { core::mem::transmute::<Vec<Row<'static>>, Vec<Row<'a>>>(vec_static) }
}

#[inline(always)]
fn return_row_vec<'a>(mut rows: Vec<Row<'a>>) {
    if rows.capacity() >= 64 {
        rows.clear();
        let vec_static = unsafe { core::mem::transmute::<Vec<Row<'a>>, Vec<Row<'static>>>(rows) };
        ENCODE_ROW_POOL.with(|cell| {
            let mut pool = cell.borrow_mut();
            if vec_static.capacity() > pool.capacity() {
                *pool = vec_static;
            }
        });
    }
}

