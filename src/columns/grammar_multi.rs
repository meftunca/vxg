struct MultiMatchScratch {
    seg_matchers: [[core::mem::MaybeUninit<SegMatcher>; MAX_SLOTS + 1]; 16],
}

impl Default for MultiMatchScratch {
    fn default() -> Self {
        Self {
            seg_matchers: [[core::mem::MaybeUninit::uninit(); MAX_SLOTS + 1]; 16],
        }
    }
}

std::thread_local! {
    static MULTI_MATCH_TLS: core::cell::RefCell<MultiMatchScratch> =
        core::cell::RefCell::new(MultiMatchScratch::default());
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn try_match_clean_stream_many<'a>(
    data: &'a [u8],
    start: usize,
    templates: &[Template<'a>],
    clean_nl_flags: &[bool],
    cur_tpl: usize,
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<(usize, usize)> {
    if !clean_nl_flags[cur_tpl] {
        return None;
    }
    let len = data.len();
    let ptr = data.as_ptr();
    let segs = &templates[cur_tpl].segments;
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
        if next_end > len
            || !slice_eq_fast(unsafe { data.get_unchecked(pos..next_end) }, next_seg)
        {
            return None;
        }
        pos = next_end;
    }
    Some((cur_tpl, pos - start))
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
    delim_mask_cache: &mut (usize, u64, u64),
    val_buf: &mut [&'a [u8]; MAX_SLOTS],
) -> bool {
    let total_len = data.len();
    let mut cur_line_start = *line_start;
    let mut cur_last_tpl = *last_template;
    if cur_line_start >= total_len || !clean_nl_flags[cur_last_tpl] {
        return false;
    }
    let tpl_count = templates.len();
    if tpl_count > 16 {
        let mut matched_any = false;
        while cur_line_start < total_len {
            let Some((matched_idx, line_len)) = try_match_clean_stream_many(
                data,
                cur_line_start,
                templates,
                clean_nl_flags,
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
                for slot in 0..slots {
                    unsafe {
                        tpl.values
                            .get_unchecked_mut(slot)
                            .push(*val_buf.get_unchecked(slot));
                    }
                }
            }
            rows.push(Row::Template(matched_idx));
            *template_rows += 1;
            matched_any = true;
        }
        *line_start = cur_line_start;
        *last_template = cur_last_tpl;
        return matched_any;
    }

    let ptr = data.as_ptr();
    let first_rem_byte = unsafe { *ptr.add(cur_line_start) };
    let seg0_cur = templates[cur_last_tpl].segments[0];
    let seg0_ok = if cur_line_start + seg0_cur.len() <= total_len
        && first_rem_byte == unsafe { *seg0_cur.as_ptr() }
        && slice_eq_fast(
            unsafe { data.get_unchecked(cur_line_start..cur_line_start + seg0_cur.len()) },
            seg0_cur,
        ) {
        true
    } else {
        let mut any_alt = false;
        for alt in 0..tpl_count {
            if alt != cur_last_tpl && clean_nl_flags[alt] && shared_segs[cur_last_tpl][alt] == 0 {
                let alt_seg0 = templates[alt].segments[0];
                if cur_line_start + alt_seg0.len() <= total_len
                    && first_rem_byte == unsafe { *alt_seg0.as_ptr() }
                    && slice_eq_fast(
                        unsafe {
                            data.get_unchecked(cur_line_start..cur_line_start + alt_seg0.len())
                        },
                        alt_seg0,
                    )
                {
                    any_alt = true;
                    break;
                }
            }
        }
        any_alt
    };
    if !seg0_ok {
        return false;
    }

    MULTI_MATCH_TLS.with(|cell| {
        let mut tls = cell.borrow_mut();
        let mut slots_arr = [0usize; 16];
        let mut row_idx_arr = [0usize; 16];
        let mut start_idx_arr = [0usize; 16];
        let mut col_cap_arr = [0usize; 16];
        let mut col_ptrs = [[core::ptr::null_mut::<&'a [u8]>(); MAX_SLOTS]; 16];

        for t in 0..tpl_count {
            if clean_nl_flags[t] {
                init_seg_matchers(&templates[t].segments, &mut tls.seg_matchers[t]);
                let slots = templates[t].values.len();
                slots_arr[t] = slots;
                if slots != 0 {
                    let r_idx = unsafe { templates[t].values.get_unchecked(0).len() };
                    row_idx_arr[t] = r_idx;
                    start_idx_arr[t] = r_idx;
                    col_cap_arr[t] = refresh_col_ptrs(
                        &mut templates[t].values,
                        slots,
                        r_idx,
                        false,
                        &mut col_ptrs[t],
                    );
                }
            }
        }

        let (mut cw, mut cq, mut cp) = *delim_mask_cache;
        let mut matched_count = 0usize;

        'row_loop: while cur_line_start < total_len {
            let mut cur_tpl = cur_last_tpl;
            if !clean_nl_flags[cur_tpl] {
                break;
            }
            let sm0 = unsafe { tls.seg_matchers[cur_tpl].get_unchecked(0).assume_init_ref() };
            if unsafe {
                !sm0.matches(
                    data,
                    ptr,
                    total_len,
                    cur_line_start,
                    templates.get_unchecked(cur_tpl).segments.get_unchecked(0),
                )
            } {
                let mut found = usize::MAX;
                for alt in 0..tpl_count {
                    if alt != cur_tpl && clean_nl_flags[alt] && shared_segs[cur_tpl][alt] == 0 {
                        let alt_sm0 =
                            unsafe { tls.seg_matchers[alt].get_unchecked(0).assume_init_ref() };
                        if unsafe {
                            alt_sm0.matches(
                                data,
                                ptr,
                                total_len,
                                cur_line_start,
                                templates.get_unchecked(alt).segments.get_unchecked(0),
                            )
                        } {
                            found = alt;
                            break;
                        }
                    }
                }
                if found == usize::MAX {
                    break 'row_loop;
                }
                cur_tpl = found;
            }

            let mut segs: &[&'a [u8]] = &unsafe { templates.get_unchecked(cur_tpl) }.segments;
            let mut slots = segs.len() - 1;
            let mut pos = cur_line_start + unsafe { segs.get_unchecked(0) }.len();
            let mut slot = 0usize;

            loop {
                if slot >= slots {
                    break;
                }
                if slot >= max_shared[cur_tpl] as usize {
                    let sms = &tls.seg_matchers[cur_tpl];
                    while slot < slots {
                        if pos >= total_len {
                            break 'row_loop;
                        }
                        let first = unsafe { *ptr.add(pos) };
                        let val_end = match first {
                            b'"' => match scan_json_string_reg(
                                data,
                                pos,
                                &mut cw,
                                &mut cq,
                                &mut cp,
                                delim_mask_cache,
                            ) {
                                Some(e) => e,
                                None => break 'row_loop,
                            },
                            b'{' | b'[' => match scan_compound(data, pos) {
                                Some(e) => e,
                                None => break 'row_loop,
                            },
                            _ if first <= b',' || first == b'}' || first == b']' => {
                                break 'row_loop;
                            }
                            _ => scan_json_prim_reg(
                                data,
                                pos,
                                &mut cw,
                                &mut cq,
                                &mut cp,
                                delim_mask_cache,
                            ),
                        };
                        unsafe {
                            *val_buf.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
                        }
                        pos = val_end;
                        let sm = unsafe { sms.get_unchecked(slot + 1).assume_init_ref() };
                        if unsafe {
                            !sm.matches(data, ptr, total_len, pos, segs.get_unchecked(slot + 1))
                        } {
                            break 'row_loop;
                        }
                        pos += sm.len;
                        slot += 1;
                    }
                    break;
                }

                if pos >= total_len {
                    break 'row_loop;
                }
                let first = unsafe { *ptr.add(pos) };
                let val_end = match first {
                    b'"' => match scan_json_string_reg(
                        data,
                        pos,
                        &mut cw,
                        &mut cq,
                        &mut cp,
                        delim_mask_cache,
                    ) {
                        Some(e) => e,
                        None => break 'row_loop,
                    },
                    b'{' | b'[' => match scan_compound(data, pos) {
                        Some(e) => e,
                        None => break 'row_loop,
                    },
                    _ if first <= b',' || first == b'}' || first == b']' => break 'row_loop,
                    _ => scan_json_prim_reg(data, pos, &mut cw, &mut cq, &mut cp, delim_mask_cache),
                };
                unsafe {
                    *val_buf.get_unchecked_mut(slot) = data.get_unchecked(pos..val_end);
                }
                pos = val_end;
                let sm = unsafe {
                    tls.seg_matchers[cur_tpl]
                        .get_unchecked(slot + 1)
                        .assume_init_ref()
                };
                if unsafe { !sm.matches(data, ptr, total_len, pos, segs.get_unchecked(slot + 1)) } {
                    let req_shared = (slot + 1) as u8;
                    let mut switched = false;
                    for alt in 0..tpl_count {
                        if alt != cur_tpl
                            && clean_nl_flags[alt]
                            && shared_segs[cur_tpl][alt] == req_shared
                        {
                            let alt_sm = unsafe {
                                tls.seg_matchers[alt]
                                    .get_unchecked(slot + 1)
                                    .assume_init_ref()
                            };
                            let alt_segs = &unsafe { templates.get_unchecked(alt) }.segments;
                            if unsafe {
                                alt_sm.matches(
                                    data,
                                    ptr,
                                    total_len,
                                    pos,
                                    alt_segs.get_unchecked(slot + 1),
                                )
                            } {
                                cur_tpl = alt;
                                segs = alt_segs;
                                slots = segs.len() - 1;
                                pos += alt_sm.len;
                                slot += 1;
                                switched = true;
                                break;
                            }
                        }
                    }
                    if !switched {
                        break 'row_loop;
                    }
                } else {
                    pos += sm.len;
                    slot += 1;
                }
            }

            cur_line_start = pos;
            let val_slots = slots_arr[cur_tpl];
            if val_slots != 0 {
                cur_last_tpl = cur_tpl;
                let r_idx = row_idx_arr[cur_tpl];
                if r_idx == col_cap_arr[cur_tpl] {
                    let tpl_mut = unsafe { templates.get_unchecked_mut(cur_tpl) };
                    col_cap_arr[cur_tpl] = refresh_col_ptrs(
                        &mut tpl_mut.values,
                        val_slots,
                        r_idx,
                        true,
                        &mut col_ptrs[cur_tpl],
                    );
                    // Rebind `segs` safety: `cur_tpl` drops before next iteration.
                }
                let ptrs = &col_ptrs[cur_tpl];
                for s in 0..val_slots {
                    unsafe {
                        *ptrs.get_unchecked(s).add(r_idx) = *val_buf.get_unchecked(s);
                    }
                }
                row_idx_arr[cur_tpl] = r_idx + 1;
            }
            rows.push(Row::Template(cur_tpl));
            matched_count += 1;
        }

        *delim_mask_cache = (cw, cq, cp);
        if matched_count == 0 {
            return false;
        }
        for t in 0..tpl_count {
            let final_len = row_idx_arr[t];
            if final_len > start_idx_arr[t] {
                let tpl = unsafe { templates.get_unchecked_mut(t) };
                for s in 0..slots_arr[t] {
                    unsafe {
                        tpl.values.get_unchecked_mut(s).set_len(final_len);
                    }
                }
            }
        }
        *template_rows += matched_count;
        *line_start = cur_line_start;
        *last_template = cur_last_tpl;
        true
    })
}
