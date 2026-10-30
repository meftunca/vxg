#[inline(always)]
fn find_byte(needle: u8, data: &[u8]) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut pos = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        if len >= 16 {
            let vn = vdupq_n_u8(needle);
            while pos + 32 <= len {
                let c0 = vceqq_u8(vld1q_u8(ptr.add(pos)), vn);
                let c1 = vceqq_u8(vld1q_u8(ptr.add(pos + 16)), vn);
                let any = vorrq_u8(c0, c1);
                if vmaxvq_u8(any) != 0 {
                    let m0 = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c0), 4)),
                        0,
                    );
                    if m0 != 0 {
                        return Some(pos + (m0.trailing_zeros() as usize >> 2));
                    }
                    let m1 = vget_lane_u64(
                        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c1), 4)),
                        0,
                    );
                    return Some(pos + 16 + (m1.trailing_zeros() as usize >> 2));
                }
                pos += 32;
            }
            if pos + 16 <= len {
                let c0 = vceqq_u8(vld1q_u8(ptr.add(pos)), vn);
                let m0 = vget_lane_u64(
                    vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c0), 4)),
                    0,
                );
                if m0 != 0 {
                    return Some(pos + (m0.trailing_zeros() as usize >> 2));
                }
                pos += 16;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        if len >= 16 {
            let vn = _mm_set1_epi8(needle as i8);
            while pos + 32 <= len {
                let c0 = _mm_cmpeq_epi8(_mm_loadu_si128(ptr.add(pos).cast::<__m128i>()), vn);
                let c1 = _mm_cmpeq_epi8(_mm_loadu_si128(ptr.add(pos + 16).cast::<__m128i>()), vn);
                let m = (_mm_movemask_epi8(c0) as u32) | ((_mm_movemask_epi8(c1) as u32) << 16);
                if m != 0 {
                    return Some(pos + m.trailing_zeros() as usize);
                }
                pos += 32;
            }
            if pos + 16 <= len {
                let c0 = _mm_cmpeq_epi8(_mm_loadu_si128(ptr.add(pos).cast::<__m128i>()), vn);
                let m0 = _mm_movemask_epi8(c0) as u32;
                if m0 != 0 {
                    return Some(pos + m0.trailing_zeros() as usize);
                }
                pos += 16;
            }
        }
    }
    let splat = (needle as u64) * 0x0101_0101_0101_0101;
    while pos + 8 <= len {
        let word = unsafe { ptr.add(pos).cast::<u64>().read_unaligned() ^ splat };
        let z = word.wrapping_sub(0x0101_0101_0101_0101) & !word & 0x8080_8080_8080_8080;
        if z != 0 {
            return Some(pos + (z.trailing_zeros() as usize >> 3));
        }
        pos += 8;
    }
    if pos + 4 <= len {
        let word = unsafe { ptr.add(pos).cast::<u32>().read_unaligned() ^ (splat as u32) };
        let z = word.wrapping_sub(0x0101_0101) & !word & 0x8080_8080;
        if z != 0 {
            return Some(pos + (z.trailing_zeros() as usize >> 3));
        }
        pos += 4;
    }
    while pos < len {
        if unsafe { *ptr.add(pos) } == needle {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

#[inline(always)]
unsafe fn eq16_unaligned(a: *const u8, b: *const u8) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let va = _mm_loadu_si128(a.cast::<__m128i>());
        let vb = _mm_loadu_si128(b.cast::<__m128i>());
        _mm_movemask_epi8(_mm_cmpeq_epi8(va, vb)) == 0xffff
    }
    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        a.cast::<u128>().read_unaligned() == b.cast::<u128>().read_unaligned()
    }
}

#[inline(always)]
fn slice_eq_fast(a: &[u8], b: &[u8]) -> bool {
    let len = a.len();
    if len != b.len() {
        return false;
    }
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    unsafe {
        if len >= 8 {
            if len <= 16 {
                let tail = len - 8;
                return ((pa.cast::<u64>().read_unaligned() ^ pb.cast::<u64>().read_unaligned())
                    | (pa.add(tail).cast::<u64>().read_unaligned()
                        ^ pb.add(tail).cast::<u64>().read_unaligned()))
                    == 0;
            }
            if pa.cast::<u64>().read_unaligned() != pb.cast::<u64>().read_unaligned() {
                return false;
            }
            if len <= 32 {
                if pa.add(8).cast::<u64>().read_unaligned()
                    != pb.add(8).cast::<u64>().read_unaligned()
                {
                    return false;
                }
                let tail = len - 16;
                return eq16_unaligned(pa.add(tail), pb.add(tail));
            }
            let mut off = 8usize;
            while off + 16 < len {
                if !eq16_unaligned(pa.add(off), pb.add(off)) {
                    return false;
                }
                off += 16;
            }
            let tail = len - 16;
            return eq16_unaligned(pa.add(tail), pb.add(tail));
        }
        if len >= 4 {
            let tail = len - 4;
            return ((pa.cast::<u32>().read_unaligned() ^ pb.cast::<u32>().read_unaligned())
                | (pa.add(tail).cast::<u32>().read_unaligned()
                    ^ pb.add(tail).cast::<u32>().read_unaligned()))
                == 0;
        }
        if len >= 2 {
            return pa.cast::<u16>().read_unaligned() == pb.cast::<u16>().read_unaligned()
                && *pa.add(len - 1) == *pb.add(len - 1);
        }
        len == 0 || *pa == *pb
    }
}

/// Speculatively match `record` against a known `template` by comparing literal
/// segments and scanning values between them.
///
/// Safety / Equivalence invariant:
/// In any valid top-level JSON object record produced by `parse_record`, every
/// literal segment `segments[s]` for `s > 0` contains the `:` after the key of
/// slot `s`, and `segments[slots]` ends with the closing `}` (plus any trailing
/// newline). When `record` has no irregular whitespace outside strings (i.e.
/// `segments[0]` starts with `{`, every internal segment starts with `,` and ends
/// with `:`, and the final segment starts with `}` and ends at `record.len()`),
/// matching all `slots + 1` literal segments and scanning valid values between
/// them is 100% equivalent to `parse_record(record)` producing identical
/// segments. We verify this clean-segment property once via `template_is_clean`.
#[inline(always)]
fn template_is_clean(segments: &[&[u8]]) -> bool {
    let slots = segments.len().saturating_sub(1);
    if slots == 0 {
        return false;
    }
    let first = segments[0];
    if first.first() != Some(&b'{') || first.last() != Some(&b':') {
        return false;
    }
    if scan_string(first, 1) != Some(first.len() - 1) {
        return false;
    }
    for seg in &segments[1..slots] {
        if seg.first() != Some(&b',') || seg.last() != Some(&b':') {
            return false;
        }
        if scan_string(seg, 1) != Some(seg.len() - 1) {
            return false;
        }
    }
    let last = segments[slots];
    match last {
        b"}" | b"}\n" | b"}\r\n" => true,
        _ => false,
    }
}

#[inline(always)]
fn template_is_clean_multiline_json(segments: &[&[u8]]) -> bool {
    let slots = segments.len().saturating_sub(1);
    if slots == 0 {
        return false;
    }
    let first = segments[0];
    let mut pos = 0usize;
    skip_space(first, &mut pos);
    if first.get(pos) != Some(&b'{') {
        return false;
    }
    pos += 1;
    skip_space(first, &mut pos);
    let Some(end_key) = scan_string(first, pos) else {
        return false;
    };
    pos = end_key;
    skip_space(first, &mut pos);
    if first.get(pos) != Some(&b':') {
        return false;
    }
    pos += 1;
    skip_space(first, &mut pos);
    if pos != first.len() {
        return false;
    }
    for seg in &segments[1..slots] {
        let mut p = 0usize;
        if seg.first() != Some(&b',') {
            return false;
        }
        p += 1;
        skip_space(seg, &mut p);
        let Some(ek) = scan_string(seg, p) else {
            return false;
        };
        p = ek;
        skip_space(seg, &mut p);
        if seg.get(p) != Some(&b':') {
            return false;
        }
        p += 1;
        skip_space(seg, &mut p);
        if p != seg.len() {
            return false;
        }
    }
    let last = segments[slots];
    let mut lp = 0usize;
    skip_space(last, &mut lp);
    if last.get(lp) != Some(&b'}') {
        return false;
    }
    lp += 1;
    skip_space(last, &mut lp);
    if last.get(lp) == Some(&b',') {
        lp += 1;
        skip_space(last, &mut lp);
    }
    lp == last.len() && last.ends_with(b"\n")
}

#[inline(never)]
fn scan_multiline_json_object_body(rem: &[u8], start: usize) -> Option<usize> {
    let limit = rem.len().min(4096);
    let mut pos = start;
    while pos < limit && matches!(rem[pos], b' ' | b'\t' | b'\r' | b'\n') {
        pos += 1;
    }
    if pos >= limit || (rem[pos] != b'"' && rem[pos] != b'}') {
        return None;
    }
    let mut depth = 1usize;
    while pos < limit {
        match rem[pos] {
            b'"' => pos = scan_string(rem, pos)?,
            b'{' | b'[' => {
                depth += 1;
                pos += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                let close = rem[pos];
                pos += 1;
                if depth == 0 {
                    if close != b'}' {
                        return None;
                    }
                    break;
                }
            }
            _ => pos += 1,
        }
    }
    if depth != 0 {
        return None;
    }
    if pos < rem.len() && rem[pos] == b',' {
        pos += 1;
    }
    while pos < rem.len() && matches!(rem[pos], b' ' | b'\t' | b'\r') {
        pos += 1;
    }
    if pos < rem.len() {
        if rem[pos] != b'\n' {
            return None;
        }
        pos += 1;
    }
    Some(pos)
}

#[inline(always)]
fn scan_multiline_json_object(rem: &[u8], first_line_len: usize) -> Option<usize> {
    if first_line_len < 2 || first_line_len > 16 {
        return None;
    }
    let b0 = rem[0];
    if b0 != b'{' && b0 != b' ' && b0 != b'\t' {
        return None;
    }
    let mut pos = 0usize;
    while pos < first_line_len && matches!(rem[pos], b' ' | b'\t') {
        pos += 1;
    }
    if pos >= first_line_len || rem[pos] != b'{' {
        return None;
    }
    pos += 1;
    while pos < first_line_len && matches!(rem[pos], b' ' | b'\t' | b'\r') {
        pos += 1;
    }
    if pos + 1 != first_line_len || rem[pos] != b'\n' {
        return None;
    }
    scan_multiline_json_object_body(rem, first_line_len)
}

#[inline(never)]
fn parse_clf_or_bracketed_log_spans(
    data: &[u8],
    spans: &mut [(u32, u32); MAX_SLOTS],
) -> Option<usize> {
    let mut body_len = data.len();
    if body_len == 0 || data[body_len - 1] != b'\n' {
        return None;
    }
    body_len -= 1;
    if body_len > 0 && data[body_len - 1] == b'\r' {
        body_len -= 1;
    }
    if body_len < 16 || body_len > 4096 {
        return None;
    }
    let body = &data[..body_len];
    let mut pos = 0usize;
    let mut count = 0usize;
    let mut has_structured_token = false;
    while pos < body_len {
        let b = body[pos];
        if b == b'[' {
            let mut end = pos + 1;
            while end < body_len && body[end] != b']' {
                if body[end] < 0x20 || body[end] >= 0x7f || body[end] == b'[' {
                    return None;
                }
                end += 1;
            }
            if end >= body_len || body[end] != b']' {
                return None;
            }
            if end > pos + 1 {
                if count >= MAX_SLOTS {
                    return None;
                }
                spans[count] = ((pos + 1) as u32, end as u32);
                count += 1;
                has_structured_token = true;
            }
            pos = end + 1;
        } else if b == b'"' {
            let end = scan_string(body, pos)?;
            if count >= MAX_SLOTS {
                return None;
            }
            spans[count] = (pos as u32, end as u32);
            count += 1;
            has_structured_token = true;
            pos = end;
        } else {
            let mut end = pos;
            let mut eq_pos = None;
            while end < body_len && body[end] > b' ' && body[end] < 0x7f {
                if matches!(body[end], b'"' | b'[' | b']' | b'{' | b'}') {
                    return None;
                }
                if body[end] == b'=' && eq_pos.is_none() {
                    eq_pos = Some(end);
                }
                end += 1;
            }
            if end == pos {
                return None;
            }
            if let Some(eq) = eq_pos {
                if eq > pos
                    && body[pos..eq].iter().all(|&c| is_logfmt_key_byte(c))
                    && eq + 1 <= end
                {
                    let val_start = eq + 1;
                    let val_end = if val_start < body_len && body[val_start] == b'"' {
                        scan_string(body, val_start)?
                    } else {
                        if val_start == end {
                            return None;
                        }
                        end
                    };
                    if count >= MAX_SLOTS {
                        return None;
                    }
                    spans[count] = (val_start as u32, val_end as u32);
                    count += 1;
                    has_structured_token = true;
                    pos = val_end;
                } else {
                    if count >= MAX_SLOTS {
                        return None;
                    }
                    spans[count] = (pos as u32, end as u32);
                    count += 1;
                    pos = end;
                }
            } else {
                if count >= MAX_SLOTS {
                    return None;
                }
                spans[count] = (pos as u32, end as u32);
                count += 1;
                pos = end;
            }
        }
        if pos == body_len {
            break;
        }
        if body[pos] != b' ' {
            return None;
        }
        pos += 1;
    }
    (has_structured_token && count >= 4).then_some(count)
}

#[inline(always)]
fn try_match_clean_template<'a>(
    record: &'a [u8],
    segments: &[&'a [u8]],
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> bool {
    let slots = segments.len() - 1;
    let seg0 = segments[0];
    if record.len() < seg0.len() || !slice_eq_fast(&record[..seg0.len()], seg0) {
        return false;
    }
    let mut pos = seg0.len();
    for slot in 0..slots {
        let val_end = match scan_value(record, pos) {
            Some(end) => end,
            None => return false,
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = record.get_unchecked(pos..val_end);
        }
        pos = val_end;
        let next_seg = unsafe { *segments.get_unchecked(slot + 1) };
        let next_end = pos + next_seg.len();
        if next_end > record.len() || !slice_eq_fast(&record[pos..next_end], next_seg) {
            return false;
        }
        pos = next_end;
    }
    pos == record.len()
}

#[inline(always)]
fn scan_delimited_value(data: &[u8], pos: usize, delim: u8) -> Option<usize> {
    if pos >= data.len() {
        return None;
    }
    let first = data[pos];
    if first == b'"' {
        return scan_string(data, pos);
    }
    if first == delim || first <= b' ' || first >= 0x7f {
        return None;
    }
    let relaxed = delim == b',' || delim == b'\t';
    let len = data.len();
    let ptr = data.as_ptr();
    let mut end = pos + 1;
    while end < len {
        let b = unsafe { *ptr.add(end) };
        if b == delim || b == b'\n' || b == b'\r' {
            break;
        }
        if b == b'"' || b < b' ' && b != b'\t' || b == 0x7f {
            return None;
        }
        // CSV / TSV fields may hold spaces, brackets and UTF-8 text (user agents, names); only
        // space-delimited records need the strict token alphabet.
        if !relaxed
            && (b <= b' '
                || b >= 0x7f
                || matches!(b, b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'))
        {
            return None;
        }
        end += 1;
    }
    Some(end)
}

