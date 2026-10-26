fn grammar_from_block<'a>(
    data: &'a [u8],
) -> Result<(Vec<Template<'a>>, Vec<Row<'a>>, usize, usize)> {
    let (templates, rows, template_rows, raw_rows, _raw_bytes) =
        grammar_from_block_mode(data, false)?;
    Ok((templates, rows, template_rows, raw_rows))
}

#[inline(never)]
fn match_single_delim_or_logfmt_stream<'a>(
    data: &'a [u8],
    tpl: &mut Template<'a>,
    delim_opt: Option<u8>,
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
    let mut cur_col_cap = *col_cap;
    if let Some(d) = delim_opt {
        while cur_line_start < total_len {
            let Some(line_len) = try_match_delimited_stream(
                data,
                cur_line_start,
                segs,
                d,
                delim_mask_cache,
                val_buf,
            ) else {
                break;
            };
            cur_line_start += line_len;
            if cur_row_idx == cur_col_cap {
                for slot in 0..slots {
                    unsafe {
                        let col = tpl.values.get_unchecked_mut(slot);
                        col.set_len(cur_row_idx);
                        col.reserve(256);
                    }
                }
                cur_col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
            }
            for slot in 0..slots {
                unsafe {
                    *tpl.values
                        .get_unchecked_mut(slot)
                        .as_mut_ptr()
                        .add(cur_row_idx) = *val_buf.get_unchecked(slot);
                }
            }
            cur_row_idx += 1;
        }
    } else {
        while cur_line_start < total_len {
            let Some(line_len) = try_match_logfmt_stream(
                data,
                cur_line_start,
                segs,
                delim_mask_cache,
                val_buf,
            ) else {
                break;
            };
            cur_line_start += line_len;
            if cur_row_idx == cur_col_cap {
                for slot in 0..slots {
                    unsafe {
                        let col = tpl.values.get_unchecked_mut(slot);
                        col.set_len(cur_row_idx);
                        col.reserve(256);
                    }
                }
                cur_col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
            }
            for slot in 0..slots {
                unsafe {
                    *tpl.values
                        .get_unchecked_mut(slot)
                        .as_mut_ptr()
                        .add(cur_row_idx) = *val_buf.get_unchecked(slot);
                }
            }
            cur_row_idx += 1;
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
            if row_idx < unsafe { tpl.values.get_unchecked(0).capacity() } {
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
    let mut cur_col_cap = *col_cap;
    while cur_line_start + seg0_len <= total_len {
        if !slice_eq_fast(
            unsafe { data.get_unchecked(cur_line_start..cur_line_start + seg0_len) },
            seg0,
        ) {
            break;
        }
        if cur_row_idx == cur_col_cap {
            for slot in 0..slots {
                unsafe {
                    let col = tpl.values.get_unchecked_mut(slot);
                    col.set_len(cur_row_idx);
                    col.reserve(256);
                }
            }
            cur_col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
        }
        let mut pos = cur_line_start + seg0_len;
        let mut ok = true;
        for slot in 0..slots {
            if pos >= total_len {
                ok = false;
                break;
            }
            let first = unsafe { *data_ptr.add(pos) };
            let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
            let val_end = match first {
                b'"' => match scan_string(data, pos) {
                    Some(e) => e,
                    None => {
                        ok = false;
                        break;
                    }
                },
                b'{' | b'[' => match scan_compound(data, pos) {
                    Some(e) => e,
                    None => {
                        ok = false;
                        break;
                    }
                },
                _ if first <= b',' || first == b'}' || first == b']' => {
                    ok = false;
                    break;
                }
                _ => scan_primitive_end(data_ptr, total_len, pos),
            };
            unsafe {
                *tpl.values
                    .get_unchecked_mut(slot)
                    .as_mut_ptr()
                    .add(cur_row_idx) = data.get_unchecked(pos..val_end);
            }
            pos = val_end;
            let next_end = pos + next_seg.len();
            if next_end > total_len
                || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
            {
                ok = false;
                break;
            }
            pos = next_end;
        }
        if !ok {
            break;
        }
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
    let mut cur_col_cap = *col_cap;
    while cur_line_start + seg0_len <= total_len {
        if !slice_eq_fast(
            unsafe { data.get_unchecked(cur_line_start..cur_line_start + seg0_len) },
            seg0,
        ) {
            break;
        }
        if cur_row_idx == cur_col_cap {
            for slot in 0..slots {
                unsafe {
                    let col = tpl.values.get_unchecked_mut(slot);
                    col.set_len(cur_row_idx);
                    col.reserve(256);
                }
            }
            cur_col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
        }
        let mut pos = cur_line_start + seg0_len;
        let mut ok = true;
        for slot in 0..slots {
            if pos >= total_len {
                ok = false;
                break;
            }
            let first = unsafe { *data_ptr.add(pos) };
            let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
            let val_end = match first {
                b'"' => match scan_json_string_cached(data, pos, delim_mask_cache) {
                    Some(e) => e,
                    None => {
                        ok = false;
                        break;
                    }
                },
                b'{' | b'[' => match scan_compound(data, pos) {
                    Some(e) => e,
                    None => {
                        ok = false;
                        break;
                    }
                },
                _ if first <= b',' || first == b'}' || first == b']' => {
                    ok = false;
                    break;
                }
                _ => scan_primitive_end(data_ptr, total_len, pos),
            };
            unsafe {
                *tpl.values
                    .get_unchecked_mut(slot)
                    .as_mut_ptr()
                    .add(cur_row_idx) = data.get_unchecked(pos..val_end);
            }
            pos = val_end;
            let next_end = pos + next_seg.len();
            if next_end > total_len
                || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
            {
                ok = false;
                break;
            }
            pos = next_end;
        }
        if !ok {
            break;
        }
        cur_line_start = pos;
        cur_row_idx += 1;
    }
    *line_start = cur_line_start;
    *row_idx = cur_row_idx;
    *col_cap = cur_col_cap;
}

