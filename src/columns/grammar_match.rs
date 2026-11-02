fn grammar_from_block<'a>(
    data: &'a [u8],
) -> Result<(Vec<Template<'a>>, Vec<Row<'a>>, usize, usize)> {
    let (templates, rows, template_rows, raw_rows, _raw_bytes) =
        grammar_from_block_mode(data, false)?;
    Ok((templates, rows, template_rows, raw_rows))
}

#[inline(always)]
fn template_is_clean_clf(segments: &[&[u8]]) -> bool {
    let slots = segments.len().saturating_sub(1);
    if !(4..=MAX_SLOTS).contains(&slots) || !segments[0].is_empty() {
        return false;
    }
    let mut prev_bracket = false;
    for &seg in &segments[1..slots] {
        let Some(rest) = (if prev_bracket {
            seg.strip_prefix(b"] ")
        } else {
            seg.strip_prefix(b" ")
        }) else {
            return false;
        };
        if rest.is_empty() {
            prev_bracket = false;
        } else if rest == b"[" {
            prev_bracket = true;
        } else if rest.len() >= 2
            && rest.last() == Some(&b'=')
            && rest[..rest.len() - 1]
                .iter()
                .all(|&b| is_logfmt_key_byte(b))
        {
            prev_bracket = false;
        } else {
            return false;
        }
    }
    if prev_bracket {
        matches!(segments[slots], b"]\n" | b"]\r\n")
    } else {
        matches!(segments[slots], b"\n" | b"\r\n")
    }
}

const CLF_IP_OK: [bool; 256] = {
    let mut t = [false; 256];
    let mut b = 0x21usize;
    while b < 0x7f {
        t[b] = !matches!(
            b as u8,
            b'=' | b',' | b'\t' | b'|' | b'"' | b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'
        );
        b += 1;
    }
    t
};

const CLF_TOK_OK: [bool; 256] = {
    let mut t = [false; 256];
    let mut b = 0x21usize;
    while b < 0x7f {
        t[b] = !matches!(b as u8, b'"' | b'[' | b']' | b'{' | b'}' | b'=');
        b += 1;
    }
    t
};

#[inline(always)]
unsafe fn scan_clf_bracket(ptr: *const u8, len: usize, pos: usize) -> Option<usize> {
    let mut end = pos;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v_close = vdupq_n_u8(b']');
        let v_open = vdupq_n_u8(b'[');
        let v_20 = vdupq_n_u8(0x20);
        let v_5f = vdupq_n_u8(0x5f);
        while end + 16 <= len {
            let c = vld1q_u8(ptr.add(end));
            let bad = vorrq_u8(vcgeq_u8(vsubq_u8(c, v_20), v_5f), vceqq_u8(c, v_open));
            let stop = vorrq_u8(vceqq_u8(c, v_close), bad);
            let m = vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(stop))));
            if m != 0 {
                let idx = (m.trailing_zeros() >> 2) as usize;
                let hit = end + idx;
                return (*ptr.add(hit) == b']' && hit > pos).then_some(hit);
            }
            end += 16;
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v_close = _mm_set1_epi8(b']' as i8);
        let v_open = _mm_set1_epi8(b'[' as i8);
        let v_1f = _mm_set1_epi8(0x1f);
        let v_7f = _mm_set1_epi8(0x7f);
        while end + 16 <= len {
            let c = _mm_loadu_si128(ptr.add(end).cast::<__m128i>());
            let ctrl = _mm_or_si128(_mm_cmpeq_epi8(_mm_max_epu8(c, v_1f), v_1f), _mm_cmpeq_epi8(_mm_min_epu8(c, v_7f), v_7f));
            let stop = _mm_or_si128(_mm_cmpeq_epi8(c, v_close), _mm_or_si128(_mm_cmpeq_epi8(c, v_open), ctrl));
            let m = _mm_movemask_epi8(stop) as u32;
            if m != 0 {
                let hit = end + m.trailing_zeros() as usize;
                return (*ptr.add(hit) == b']' && hit > pos).then_some(hit);
            }
            end += 16;
        }
    }
    while end < len {
        let b = unsafe { *ptr.add(end) };
        if b == b']' {
            return (end > pos).then_some(end);
        }
        if b < 0x20 || b >= 0x7f || b == b'[' {
            return None;
        }
        end += 1;
    }
    None
}

#[inline(always)]
fn try_match_clf_stream<'a>(
    data: &'a [u8],
    start: usize,
    segments: &[&'a [u8]],
    quote_cache: &mut (usize, u64, u64),
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let len = data.len();
    let ptr = data.as_ptr();
    if start >= len || !unsafe { *ptr.add(start) }.is_ascii_digit() {
        return None;
    }
    let mut end0 = start + 1;
    while end0 < len && CLF_IP_OK[unsafe { *ptr.add(end0) } as usize] {
        end0 += 1;
    }
    if end0 - start < 7 {
        return None;
    }
    unsafe { *out_values.get_unchecked_mut(0) = data.get_unchecked(start..end0); }
    let mut pos = end0;
    let mut has_structured = false;
    let (mut cw, mut cq) = (quote_cache.0, quote_cache.1);
    for slot in 1..slots {
        let seg = unsafe { *segments.get_unchecked(slot) };
        let next_pos = pos + seg.len();
        if next_pos >= len || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_pos) }, seg) {
            return None;
        }
        pos = next_pos;
        let seg_last = unsafe { *seg.get_unchecked(seg.len() - 1) };
        if seg_last == b'[' {
            let end = unsafe { scan_clf_bracket(ptr, len, pos)? };
            has_structured = true;
            unsafe { *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..end); }
            pos = end;
        } else if seg_last == b'=' {
            let mut end = pos;
            while end < len {
                let b = unsafe { *ptr.add(end) };
                if !CLF_TOK_OK[b as usize] && b != b'=' {
                    if b <= b' ' || b >= 0x7f { break; }
                    return None;
                }
                end += 1;
            }
            if end == pos { return None; }
            has_structured = true;
            unsafe { *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..end); }
            pos = end;
        } else if unsafe { *ptr.add(pos) } == b'"' {
            let end = scan_json_string_reg(data, pos, &mut cw, &mut cq, quote_cache)?;
            has_structured = true;
            unsafe { *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..end); }
            pos = end;
        } else {
            let mut end = pos;
            while end < len && CLF_TOK_OK[unsafe { *ptr.add(end) } as usize] {
                end += 1;
            }
            if end == pos || (end < len && { let b = unsafe { *ptr.add(end) }; b > b' ' && b < 0x7f }) {
                return None;
            }
            unsafe { *out_values.get_unchecked_mut(slot) = data.get_unchecked(pos..end); }
            pos = end;
        }
    }
    quote_cache.0 = cw;
    quote_cache.1 = cq;
    let last_seg = unsafe { *segments.get_unchecked(slots) };
    let next_end = pos + last_seg.len();
    if !has_structured
        || next_end > len
        || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, last_seg)
    {
        return None;
    }
    let tail_nl = if last_seg.ends_with(b"\r\n") { 2 } else { 1 };
    let body_len = (next_end - start) - tail_nl;
    (16..=4096).contains(&body_len).then_some(next_end - start)
}

#[inline(always)]
fn refresh_col_ptrs<'a>(
    cols: &mut [Vec<&'a [u8]>],
    slots: usize,
    row_idx: usize,
    grow: bool,
    ptrs: &mut [*mut &'a [u8]; MAX_SLOTS],
) -> usize {
    let mut min_cap = usize::MAX;
    for slot in 0..slots {
        unsafe {
            let col = cols.get_unchecked_mut(slot);
            if grow {
                col.set_len(row_idx);
                col.reserve(256);
            }
            min_cap = min_cap.min(col.capacity());
            *ptrs.get_unchecked_mut(slot) = col.as_mut_ptr();
        }
    }
    min_cap
}

#[inline(never)]
fn match_single_delim_or_logfmt_stream<'a>(
    data: &'a [u8],
    tpl: &mut Template<'a>,
    delim_opt: Option<u8>,
    space_kind: u8,
    line_start: &mut usize,
    row_idx: &mut usize,
    col_cap: &mut usize,
    delim_mask_cache: &mut (usize, u64, u64),
    val_buf: &mut [&'a [u8]; MAX_SLOTS],
) {
    let segs = &tpl.segments;
    let slots = tpl.values.len();
    let total_len = data.len();
    let mut cur_line_start = *line_start;
    let mut cur_row_idx = *row_idx;
    let mut col_ptrs = [core::ptr::null_mut(); MAX_SLOTS];
    let mut cur_col_cap = refresh_col_ptrs(&mut tpl.values, slots, cur_row_idx, false, &mut col_ptrs);
    macro_rules! push_vals {
        () => {
            if cur_row_idx == cur_col_cap {
                cur_col_cap = refresh_col_ptrs(&mut tpl.values, slots, cur_row_idx, true, &mut col_ptrs);
            }
            for slot in 0..slots {
                unsafe { *col_ptrs.get_unchecked(slot).add(cur_row_idx) = *val_buf.get_unchecked(slot); }
            }
            cur_row_idx += 1;
        };
    }
    if let Some(d) = delim_opt {
        while cur_line_start < total_len {
            let Some(line_len) = try_match_delimited_stream(data, cur_line_start, segs, d, delim_mask_cache, val_buf) else { break; };
            cur_line_start += line_len;
            push_vals!();
        }
    } else if space_kind == 2 {
        while cur_line_start < total_len {
            let Some(line_len) = try_match_clf_stream(data, cur_line_start, segs, delim_mask_cache, val_buf) else { break; };
            cur_line_start += line_len;
            push_vals!();
        }
    } else {
        while cur_line_start < total_len {
            let Some(line_len) = try_match_logfmt_stream(data, cur_line_start, segs, delim_mask_cache, val_buf) else { break; };
            cur_line_start += line_len;
            push_vals!();
        }
    }
    *line_start = cur_line_start;
    *row_idx = cur_row_idx;
    *col_cap = cur_col_cap;
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn match_multi_clean_stream<'a>(
    data: &'a [u8],
    templates: &mut [Template<'a>],
    clean_nl_flags: &[bool],
    shared_segs: &[[u8; 16]; 16],
    max_shared: &[u8; 16],
    last_template: &mut usize,
    line_start: &mut usize,
    template_rows: &mut usize,
    rows: &mut Vec<Row<'a>>,
    val_buf: &mut [&'a [u8]; MAX_SLOTS],
) -> bool {
    let total_len = data.len();
    let mut cur_line_start = *line_start;
    let mut cur_last_tpl = *last_template;
    let mut matched_any = false;
    while cur_line_start < total_len {
        let Some((matched_idx, line_len)) = try_match_clean_stream(
            data,
            cur_line_start,
            templates,
            clean_nl_flags,
            shared_segs,
            max_shared,
            cur_last_tpl,
            val_buf,
        ) else {
            break;
        };
        cur_line_start += line_len;
        let tpl = unsafe { templates.get_unchecked_mut(matched_idx) };
        let slots = tpl.values.len();
        if slots != 0 {
            cur_last_tpl = matched_idx;
            let row_idx = unsafe { tpl.values.get_unchecked(0).len() };
            if row_idx < min_col_capacity(&tpl.values) {
                for slot in 0..slots {
                    unsafe {
                        let col = tpl.values.get_unchecked_mut(slot);
                        *col.as_mut_ptr().add(row_idx) = *val_buf.get_unchecked(slot);
                        col.set_len(row_idx + 1);
                    }
                }
            } else {
                for slot in 0..slots {
                    unsafe {
                        tpl.values
                            .get_unchecked_mut(slot)
                            .push(*val_buf.get_unchecked(slot));
                    }
                }
            }
        }
        rows.push(Row::Template(matched_idx));
        *template_rows += 1;
        matched_any = true;
    }
    *line_start = cur_line_start;
    *last_template = cur_last_tpl;
    matched_any
}

#[inline(never)]
fn match_single_json_clean_stream<'a>(
    data: &'a [u8],
    tpl: &mut Template<'a>,
    line_start: &mut usize,
    row_idx: &mut usize,
    col_cap: &mut usize,
) {
    let segs = &tpl.segments;
    let slots = tpl.values.len();
    let seg0 = segs[0];
    let seg0_len = seg0.len();
    let total_len = data.len();
    let data_ptr = data.as_ptr();
    let mut cur_line_start = *line_start;
    let mut cur_row_idx = *row_idx;
    let mut col_ptrs = [core::ptr::null_mut(); MAX_SLOTS];
    let mut cur_col_cap = refresh_col_ptrs(&mut tpl.values, slots, cur_row_idx, false, &mut col_ptrs);
    while cur_line_start + seg0_len <= total_len {
        if !slice_eq_fast(unsafe { data.get_unchecked(cur_line_start..cur_line_start + seg0_len) }, seg0) {
            break;
        }
        if cur_row_idx == cur_col_cap {
            cur_col_cap = refresh_col_ptrs(&mut tpl.values, slots, cur_row_idx, true, &mut col_ptrs);
        }
        let mut pos = cur_line_start + seg0_len;
        let mut ok = true;
        for slot in 0..slots {
            if pos >= total_len { ok = false; break; }
            let first = unsafe { *data_ptr.add(pos) };
            let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
            let val_end = match first {
                b'"' => match scan_string(data, pos) { Some(e) => e, None => { ok = false; break; } },
                b'{' | b'[' => match scan_compound(data, pos) { Some(e) => e, None => { ok = false; break; } },
                _ if first <= b',' || first == b'}' || first == b']' => { ok = false; break; }
                _ => scan_primitive_end(data_ptr, total_len, pos),
            };
            unsafe { *col_ptrs.get_unchecked(slot).add(cur_row_idx) = data.get_unchecked(pos..val_end); }
            pos = val_end;
            let next_end = pos + next_seg.len();
            if next_end > total_len || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg) {
                ok = false;
                break;
            }
            pos = next_end;
        }
        if !ok { break; }
        cur_line_start = pos;
        cur_row_idx += 1;
    }
    *line_start = cur_line_start;
    *row_idx = cur_row_idx;
    *col_cap = cur_col_cap;
}

#[inline(never)]
fn match_single_json_qmask_stream<'a>(
    data: &'a [u8],
    tpl: &mut Template<'a>,
    line_start: &mut usize,
    row_idx: &mut usize,
    col_cap: &mut usize,
    delim_mask_cache: &mut (usize, u64, u64),
) {
    let segs = &tpl.segments;
    let slots = tpl.values.len();
    let seg0 = segs[0];
    let seg0_len = seg0.len();
    let total_len = data.len();
    let data_ptr = data.as_ptr();
    let mut cur_line_start = *line_start;
    let mut cur_row_idx = *row_idx;
    let mut col_ptrs = [core::ptr::null_mut(); MAX_SLOTS];
    let mut cur_col_cap = refresh_col_ptrs(&mut tpl.values, slots, cur_row_idx, false, &mut col_ptrs);
    let (mut cw, mut cq) = (delim_mask_cache.0, delim_mask_cache.1);
    while cur_line_start + seg0_len <= total_len {
        if !slice_eq_fast(unsafe { data.get_unchecked(cur_line_start..cur_line_start + seg0_len) }, seg0) {
            break;
        }
        if cur_row_idx == cur_col_cap {
            cur_col_cap = refresh_col_ptrs(&mut tpl.values, slots, cur_row_idx, true, &mut col_ptrs);
        }
        let mut pos = cur_line_start + seg0_len;
        let mut ok = true;
        for slot in 0..slots {
            if pos >= total_len { ok = false; break; }
            let first = unsafe { *data_ptr.add(pos) };
            let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
            let val_end = match first {
                b'"' => match scan_json_string_reg(data, pos, &mut cw, &mut cq, delim_mask_cache) {
                    Some(e) => e,
                    None => { ok = false; break; }
                },
                b'{' | b'[' => match scan_compound(data, pos) { Some(e) => e, None => { ok = false; break; } },
                _ if first <= b',' || first == b'}' || first == b']' => { ok = false; break; }
                _ => scan_primitive_end(data_ptr, total_len, pos),
            };
            unsafe { *col_ptrs.get_unchecked(slot).add(cur_row_idx) = data.get_unchecked(pos..val_end); }
            pos = val_end;
            let next_end = pos + next_seg.len();
            if next_end > total_len || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg) {
                ok = false;
                break;
            }
            pos = next_end;
        }
        if !ok { break; }
        cur_line_start = pos;
        cur_row_idx += 1;
    }
    (delim_mask_cache.0, delim_mask_cache.1) = (cw, cq);
    *line_start = cur_line_start;
    *row_idx = cur_row_idx;
    *col_cap = cur_col_cap;
}

