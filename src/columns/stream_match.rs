#[inline(always)]
fn scan_json_string_reg(
    data: &[u8],
    pos: usize,
    cw: &mut usize,
    cq: &mut u64,
    cp: &mut u64,
    cache: &mut (usize, u64, u64),
) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    let from = pos + 1;
    let w = from >> 6;
    if *cw != w {
        let base = w * 64;
        if base + 64 <= len {
            *cw = w;
            (*cq, *cp) = unsafe { json_struct_masks64(ptr.add(base)) };
        } else {
            if !refill_json_cache(data, w, cache) { return None; }
            (*cw, *cq, *cp) = *cache;
        }
    }
    let m = *cq >> (from & 63);
    let slow_from = if m != 0 {
        let end = from + m.trailing_zeros() as usize;
        if end < len && unsafe { *ptr.add(end) } == b'"' { return Some(end + 1); }
        end
    } else {
        let nw = w + 1;
        let nbase = nw * 64;
        if nbase + 64 <= len {
            let (nq, np) = unsafe { json_struct_masks64(ptr.add(nbase)) };
            (*cw, *cq, *cp) = (nw, nq, np);
            if nq != 0 {
                let end = nbase + nq.trailing_zeros() as usize;
                if unsafe { *ptr.add(end) } == b'"' { return Some(end + 1); }
                end
            } else {
                (nw + 1) * 64
            }
        } else {
            nbase
        }
    };
    *cache = (*cw, *cq, *cp);
    let res = next_json_string_slow(data, slow_from, cache);
    (*cw, *cq, *cp) = *cache;
    res
}

#[inline(always)]
fn scan_json_prim_reg(
    data: &[u8],
    pos: usize,
    cw: &mut usize,
    cq: &mut u64,
    cp: &mut u64,
    cache: &mut (usize, u64, u64),
) -> usize {
    let len = data.len();
    let from = pos + 1;
    if from >= len {
        return len;
    }
    let ptr = data.as_ptr();
    let w = from >> 6;
    if *cw != w {
        let base = w * 64;
        if base + 64 <= len {
            *cw = w;
            (*cq, *cp) = unsafe { json_struct_masks64(ptr.add(base)) };
        } else if refill_json_cache(data, w, cache) {
            (*cw, *cq, *cp) = *cache;
        } else {
            return len;
        }
    }
    let m = *cp >> (from & 63);
    if m != 0 {
        return (from + m.trailing_zeros() as usize).min(len);
    }
    let nw = w + 1;
    let nbase = nw * 64;
    if nbase + 64 <= len {
        let (nq, np) = unsafe { json_struct_masks64(ptr.add(nbase)) };
        (*cw, *cq, *cp) = (nw, nq, np);
        if np != 0 {
            return nbase + np.trailing_zeros() as usize;
        }
        return scan_primitive_end(ptr, len, nbase + 63);
    }
    scan_primitive_end(ptr, len, pos)
}

/// Delimited (CSV/TSV/PSV) row matcher at `data[start..]`. With on-demand 64-byte
/// structural bitmasks each field boundary is one `trailing_zeros`; whenever the fast
/// path cannot *prove* success (escape, control byte, long value, any mismatch) it
/// defers to the scalar reference matcher, so results are identical by construction.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn try_match_delimited_stream<'a>(
    data: &'a [u8],
    start: usize,
    segments: &[&'a [u8]],
    delim: u8,
    cache: &mut (usize, u64, u64),
    col_ptrs: &[*mut &'a [u8]; MAX_SLOTS],
    row_idx: usize,
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let len = data.len();
    let ptr = data.as_ptr();
    let last_seg = unsafe { *segments.get_unchecked(slots) };
    let last_b0 = if last_seg.len() == 1 { Some(last_seg[0]) } else { None };
    let (mut cw, mut cq, mut cu) = *cache;
    'fast: {
        let mut pos = start;
        for slot in 0..slots {
            if pos >= len { break 'fast; }
            let first = unsafe { *ptr.add(pos) };
            let val_end = if first == b'"' {
                let fp = pos + 1;
                let w = fp >> 6;
                if cw != w {
                    let base = w * 64;
                    if base + 64 <= len {
                        (cq, cu) = unsafe { delimited_masks64(ptr.add(base), delim) };
                        cw = w;
                    } else {
                        if !refill_delim_cache(data, w, delim, cache) { break 'fast; }
                        (cw, cq, cu) = *cache;
                    }
                }
                let m = cq >> (fp & 63);
                let end = if m != 0 {
                    fp + m.trailing_zeros() as usize
                } else {
                    let nw = w + 1;
                    let nbase = nw * 64;
                    if nbase + 64 <= len {
                        (cq, cu) = unsafe { delimited_masks64(ptr.add(nbase), delim) };
                        cw = nw;
                        if cq != 0 {
                            nbase + cq.trailing_zeros() as usize
                        } else {
                            *cache = (cw, cq, cu);
                            let Some(e) = next_delim_bit_slow(data, delim, nw, true, cache) else { break 'fast; };
                            (cw, cq, cu) = *cache;
                            e
                        }
                    } else {
                        *cache = (cw, cq, cu);
                        let Some(e) = next_delim_bit_slow(data, delim, w, true, cache) else { break 'fast; };
                        (cw, cq, cu) = *cache;
                        e
                    }
                };
                if end >= len || unsafe { *ptr.add(end) } != b'"' { break 'fast; }
                end + 1
            } else {
                let w = pos >> 6;
                if cw != w {
                    let base = w * 64;
                    if base + 64 <= len {
                        (cq, cu) = unsafe { delimited_masks64(ptr.add(base), delim) };
                        cw = w;
                    } else {
                        if !refill_delim_cache(data, w, delim, cache) { break 'fast; }
                        (cw, cq, cu) = *cache;
                    }
                }
                let m = cu >> (pos & 63);
                let end = if m != 0 {
                    pos + m.trailing_zeros() as usize
                } else {
                    let nw = w + 1;
                    let nbase = nw * 64;
                    if nbase + 64 <= len {
                        (cq, cu) = unsafe { delimited_masks64(ptr.add(nbase), delim) };
                        cw = nw;
                        if cu != 0 {
                            nbase + cu.trailing_zeros() as usize
                        } else {
                            *cache = (cw, cq, cu);
                            let Some(e) = next_delim_bit_slow(data, delim, nw, false, cache) else { break 'fast; };
                            (cw, cq, cu) = *cache;
                            e
                        }
                    } else {
                        *cache = (cw, cq, cu);
                        let Some(e) = next_delim_bit_slow(data, delim, w, false, cache) else { break 'fast; };
                        (cw, cq, cu) = *cache;
                        e
                    }
                };
                if end == pos || end > len { break 'fast; }
                end
            };
            unsafe { *col_ptrs.get_unchecked(slot).add(row_idx) = data.get_unchecked(pos..val_end); }
            pos = val_end;
            if slot + 1 < slots {
                if pos >= len || unsafe { *ptr.add(pos) } != delim { break 'fast; }
                pos += 1;
            } else if let Some(lb) = last_b0 {
                if pos - start > 4096 || pos >= len || unsafe { *ptr.add(pos) } != lb { break 'fast; }
                pos += 1;
            } else {
                let next_end = pos + last_seg.len();
                if pos - start > 4096 || next_end > len || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, last_seg) {
                    break 'fast;
                }
                pos = next_end;
            }
        }
        *cache = (cw, cq, cu);
        return Some(pos - start);
    }
    *cache = (cw, cq, cu);
    let line_len = try_match_delimited_scalar(unsafe { data.get_unchecked(start..) }, segments, delim, out_values)?;
    for slot in 0..slots {
        unsafe { *col_ptrs.get_unchecked(slot).add(row_idx) = *out_values.get_unchecked(slot); }
    }
    Some(line_len)
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

