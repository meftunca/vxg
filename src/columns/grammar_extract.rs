fn grammar_from_block_mode<'a>(
    data: &'a [u8],
    allow_early_abort: bool,
) -> Result<(Vec<Template<'a>>, Vec<Row<'a>>, usize, usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        unsafe fn grammar_from_block_mode_x86_v3<'a>(
            data: &'a [u8],
            allow_early_abort: bool,
        ) -> Result<(Vec<Template<'a>>, Vec<Row<'a>>, usize, usize, usize)> {
            grammar_from_block_mode_impl(data, allow_early_abort)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            return unsafe { grammar_from_block_mode_x86_v3(data, allow_early_abort) };
        }
    }
    grammar_from_block_mode_impl(data, allow_early_abort)
}

#[inline(always)]
fn grammar_from_block_mode_impl<'a>(
    data: &'a [u8],
    allow_early_abort: bool,
) -> Result<(Vec<Template<'a>>, Vec<Row<'a>>, usize, usize, usize)> {
    let first_nl = find_byte(b'\n', data).map_or(data.len(), |i| i + 1);
    let est_rows = (data.len() / first_nl.max(24).min(64)).max(16) + 16;
    let mut templates = Vec::<Template<'a>>::with_capacity(8);
    let mut clean_flags = Vec::<bool>::with_capacity(8);
    let mut clean_nl_flags = Vec::<bool>::with_capacity(8);
    let mut delim_flags = Vec::<Option<u8>>::with_capacity(8);
    let mut logfmt_flags = Vec::<bool>::with_capacity(8);
    let mut shared_segs = [[0u8; 16]; 16];
    let mut max_shared = [0u8; 16];
    let mut has_json_tpl = false;
    let mut rows = take_row_vec(est_rows);
    let mut raw_history: [(usize, &'a [u8]); 16] = [(usize::MAX, &[]); 16];
    let mut raw_hist_pos = 0usize;
    let mut template_rows = 0usize;
    let mut raw_rows = 0usize;
    let mut raw_bytes = 0usize;
    let mut last_template = 0usize;
    let mut val_buf_uninit = core::mem::MaybeUninit::<[&'a [u8]; MAX_SLOTS]>::uninit();
    let val_buf: &mut [&'a [u8]; MAX_SLOTS] = unsafe { val_buf_uninit.assume_init_mut() };
    let mut spans_buf_uninit = core::mem::MaybeUninit::<[(u32, u32); MAX_SLOTS]>::uninit();
    let spans_buf: &mut [(u32, u32); MAX_SLOTS] = unsafe { spans_buf_uninit.assume_init_mut() };
    let mut overflow_spans = Vec::new();
    let mut line_start = 0usize;
    let total_len = data.len();
    let mut delim_mask_cache = (usize::MAX, 0u64, 0u64);
    let mut delim_cache_for: Option<u8> = None;
    while line_start < total_len {
        let rem = &data[line_start..];

        // Single-pass stream fast path: match directly on `data` without scanning for `\n` first,
        // and switch templates mid-line on shared prefix segments without re-scanning `0..pos`.
        if templates.len() == 1 && clean_nl_flags[0] && !templates[0].values.is_empty() {
            let tpl = unsafe { templates.get_unchecked_mut(0) };
            let slots = tpl.values.len();
            let mut row_idx = unsafe { tpl.values.get_unchecked(0).len() };
            let mut col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
            let start_row_idx = row_idx;
            let use_qmask = slots >= 5
                && (0..slots)
                    .filter(|&s| {
                        tpl.values[s]
                            .first()
                            .is_some_and(|v| v.first() == Some(&b'"'))
                    })
                    .count()
                    >= 4;
            if use_qmask {
                if delim_cache_for != Some(0) {
                    delim_mask_cache = (usize::MAX, 0, 0);
                    delim_cache_for = Some(0);
                }
                match_single_json_qmask_stream(
                    data,
                    tpl,
                    &mut line_start,
                    &mut row_idx,
                    &mut col_cap,
                    &mut delim_mask_cache,
                );
            } else {
                match_single_json_clean_stream(
                    data,
                    tpl,
                    &mut line_start,
                    &mut row_idx,
                    &mut col_cap,
                );
            }
            if row_idx > start_row_idx {
                let added = row_idx - start_row_idx;
                for slot in 0..slots {
                    unsafe {
                        tpl.values.get_unchecked_mut(slot).set_len(row_idx);
                    }
                }
                template_rows += added;
                rows.resize(rows.len() + added, Row::Template(0));
                continue;
            }
        } else if templates.len() == 1 && (delim_flags[0].is_some() || logfmt_flags[0]) {
            let tpl = unsafe { templates.get_unchecked_mut(0) };
            let slots = tpl.values.len();
            let delim_opt = delim_flags[0];
            let target_delim = delim_opt.unwrap_or(b' ');
            if delim_cache_for != Some(target_delim) {
                delim_mask_cache = (usize::MAX, 0, 0);
                delim_cache_for = Some(target_delim);
            }
            let mut row_idx = unsafe { tpl.values.get_unchecked(0).len() };
            let mut col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
            let start_row_idx = row_idx;
            match_single_delim_or_logfmt_stream(
                data,
                tpl,
                delim_opt,
                &mut line_start,
                &mut row_idx,
                &mut col_cap,
                &mut delim_mask_cache,
                val_buf,
            );
            if row_idx > start_row_idx {
                let added = row_idx - start_row_idx;
                for slot in 0..slots {
                    unsafe {
                        tpl.values.get_unchecked_mut(slot).set_len(row_idx);
                    }
                }
                template_rows += added;
                rows.resize(rows.len() + added, Row::Template(0));
                continue;
            }
        } else if !templates.is_empty() {
            if match_multi_clean_stream(
                data,
                &mut templates,
                &clean_nl_flags,
                &shared_segs,
                &max_shared,
                &mut last_template,
                &mut line_start,
                &mut template_rows,
                &mut rows,
                val_buf,
            ) {
                continue;
            }
        }

        let line_len = match find_byte(b'\n', rem) {
            Some(idx) => {
                let first_len = idx + 1;
                scan_multiline_json_object(rem, first_len).unwrap_or(first_len)
            }
            None => rem.len(),
        };
        let record = &rem[..line_len];
        line_start += line_len;

        if !templates.is_empty() {
            let mut matched_other = None;
            for (idx, tpl) in templates.iter().enumerate() {
                if clean_flags[idx]
                    && !clean_nl_flags[idx]
                    && try_match_clean_template(record, &tpl.segments, val_buf)
                {
                    matched_other = Some(idx);
                    break;
                }
            }
            if let Some(idx) = matched_other {
                last_template = idx;
                let tpl = &mut templates[idx];
                let slots = tpl.values.len();
                for slot in 0..slots {
                    unsafe {
                        tpl.values
                            .get_unchecked_mut(slot)
                            .push(*val_buf.get_unchecked(slot));
                    }
                }
                rows.push(Row::Template(idx));
                template_rows += 1;
                continue;
            }
        }

        if let Some(slot_count) =
            parse_record_spans(record, spans_buf, &mut overflow_spans)
        {
            if slot_count > MAX_SLOTS {
                if !allow_early_abort {
                    return_row_vec(rows);
                    return Err(invalid("column template bounds"));
                }
                let r_idx = rows.len();
                raw_history[raw_hist_pos & 15] = (r_idx, record);
                raw_hist_pos += 1;
                rows.push(Row::Raw(record));
                raw_rows += 1;
                raw_bytes += record.len();
            } else {
                let spans = &spans_buf[..slot_count];
                let existing = templates.iter().position(|template| {
                    if template.segments.len() != slot_count + 1 {
                        return false;
                    }
                    let mut prev = 0usize;
                    for (slot, &(start, end)) in spans.iter().enumerate() {
                        if !slice_eq_fast(template.segments[slot], &record[prev..start as usize]) {
                            return false;
                        }
                        prev = end as usize;
                    }
                    slice_eq_fast(template.segments[slot_count], &record[prev..])
                });
                if existing.is_none() && allow_early_abort && templates.len() >= MAX_TEMPLATES {
                    let r_idx = rows.len();
                    raw_history[raw_hist_pos & 15] = (r_idx, record);
                    raw_hist_pos += 1;
                    rows.push(Row::Raw(record));
                    raw_rows += 1;
                    raw_bytes += record.len();
                } else {
                    let index = existing.unwrap_or_else(|| {
                        if templates.len() == 1
                            && rows.len() == 1
                            && !clean_nl_flags[0]
                            && !has_json_tpl
                        {
                            let first_slice = &data[..first_nl];
                            rows[0] = Row::Raw(first_slice);
                            raw_history[0] = (0, first_slice);
                            raw_hist_pos = 1;
                            template_rows -= 1;
                            raw_rows += 1;
                            raw_bytes += first_slice.len();
                            templates.clear();
                            clean_flags.clear();
                            clean_nl_flags.clear();
                            delim_flags.clear();
                            logfmt_flags.clear();
                            max_shared = [0u8; 16];
                            has_json_tpl = false;
                        }
                        let mut segments = Vec::with_capacity(slot_count + 1);
                        let mut prev = 0usize;
                        for &(start, end) in spans {
                            segments.push(&record[prev..start as usize]);
                            prev = end as usize;
                        }
                        segments.push(&record[prev..]);
                        let is_single_clean = template_is_clean(&segments);
                        let is_clean_nl = is_single_clean
                            && matches!(segments.last().copied(), Some(b"}\n" | b"}\r\n"));
                        let is_clean =
                            is_single_clean || template_is_clean_multiline_json(&segments);
                        let delim_opt = delimited_template_delim(&segments);
                        let is_logfmt = delim_opt.is_none() && template_is_clean_logfmt(&segments);
                        if is_clean
                            || segments[0].first() == Some(&b'{')
                            || segments.iter().any(|s| s.contains(&b'['))
                        {
                            has_json_tpl = true;
                        }
                        let new_idx = templates.len();
                        if new_idx < 16 {
                            let mut new_max = 0u8;
                            for prev_idx in 0..new_idx {
                                let min_segs =
                                    templates[prev_idx].segments.len().min(segments.len());
                                let mut common = 0usize;
                                while common < min_segs
                                    && slice_eq_fast(
                                        templates[prev_idx].segments[common],
                                        segments[common],
                                    )
                                {
                                    common += 1;
                                }
                                let c = common as u8;
                                shared_segs[prev_idx][new_idx] = c;
                                shared_segs[new_idx][prev_idx] = c;
                                if c > max_shared[prev_idx] {
                                    max_shared[prev_idx] = c;
                                }
                                if c > new_max {
                                    new_max = c;
                                }
                            }
                            max_shared[new_idx] = new_max;
                        }
                        clean_flags.push(is_clean);
                        clean_nl_flags.push(is_clean_nl);
                        delim_flags.push(delim_opt);
                        logfmt_flags.push(is_logfmt);
                        templates.push(Template {
                            values: (0..slot_count)
                                .map(|_| take_col_vec(est_rows))
                                .collect(),
                            segments,
                        });
                        new_idx
                    });
                    if templates.len() > MAX_TEMPLATES {
                        return_row_vec(rows);
                        return Err(invalid("column template bounds"));
                    }
                    last_template = index;
                    let tpl = &mut templates[index];
                    for (slot, &(start, end)) in spans.iter().enumerate() {
                        tpl.values[slot].push(&record[start as usize..end as usize]);
                    }
                    rows.push(Row::Template(index));
                    template_rows += 1;
                }
            }
        } else if allow_early_abort
            && record.len() >= 2
            && record.len() <= 32
            && record.ends_with(b"\n")
        {
            if let Some(idx) = templates
                .iter()
                .position(|t| t.values.is_empty() && slice_eq_fast(t.segments[0], record))
            {
                rows.push(Row::Template(idx));
                template_rows += 1;
            } else if templates.len() < 3
                && raw_history
                    .iter()
                    .any(|&(r_idx, prev)| r_idx != usize::MAX && slice_eq_fast(prev, record))
            {
                let new_idx = templates.len();
                let mut new_max = 0u8;
                for prev_idx in 0..new_idx {
                    let common =
                        u8::from(slice_eq_fast(templates[prev_idx].segments[0], record));
                    shared_segs[prev_idx][new_idx] = common;
                    shared_segs[new_idx][prev_idx] = common;
                    if common > max_shared[prev_idx] {
                        max_shared[prev_idx] = common;
                    }
                    if common > new_max {
                        new_max = common;
                    }
                }
                max_shared[new_idx] = new_max;
                clean_flags.push(false);
                clean_nl_flags.push(true);
                delim_flags.push(None);
                logfmt_flags.push(false);
                templates.push(Template {
                    values: Vec::new(),
                    segments: vec![record],
                });
                for entry in &mut raw_history {
                    if entry.0 != usize::MAX && slice_eq_fast(entry.1, record) {
                        rows[entry.0] = Row::Template(new_idx);
                        entry.0 = usize::MAX;
                        raw_rows -= 1;
                        raw_bytes -= record.len();
                        template_rows += 1;
                    }
                }
                if templates.len() == 1 {
                    last_template = new_idx;
                }
                rows.push(Row::Template(new_idx));
                template_rows += 1;
            } else {
                let r_idx = rows.len();
                raw_history[raw_hist_pos & 15] = (r_idx, record);
                raw_hist_pos += 1;
                rows.push(Row::Raw(record));
                raw_rows += 1;
                raw_bytes += record.len();
            }
        } else {
            let r_idx = rows.len();
            raw_history[raw_hist_pos & 15] = (r_idx, record);
            raw_hist_pos += 1;
            rows.push(Row::Raw(record));
            raw_rows += 1;
            raw_bytes += record.len();
        }
        if allow_early_abort && line_start >= 1024 {
            let has_param_tpl = templates.iter().any(|t| !t.values.is_empty());
            if (!has_param_tpl
                && (template_rows == 0
                    || (line_start >= 2048 && raw_bytes > (line_start * 3) / 4)))
                || (line_start >= 4096 && (template_rows < 2 || raw_bytes > (line_start * 3) / 4))
                || (line_start >= 2048 && !has_json_tpl && raw_bytes * 2 > line_start)
            {
                templates.clear();
                return Ok((templates, rows, 0, raw_rows, raw_bytes));
            }
        }
    }
    if allow_early_abort && !(templates.len() == 1 && !templates[0].values.is_empty() && template_rows >= 2) {
        let mut tpl_counts = [0usize; MAX_TEMPLATES];
        if templates.iter().all(|t| !t.values.is_empty()) {
            for (i, t) in templates.iter().enumerate() {
                tpl_counts[i] = t.values[0].len();
            }
        } else if templates.len() == 1 {
            tpl_counts[0] = template_rows;
        } else {
            for r in &rows {
                if let Row::Template(idx) = *r {
                    tpl_counts[idx] += 1;
                }
            }
        }
        let max_tpl_rows = templates
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.values.is_empty())
            .map(|(i, _)| tpl_counts[i])
            .max()
            .unwrap_or(0);
        let should_prune = templates.iter().enumerate().any(|(i, t)| {
            let r = tpl_counts[i];
            if t.values.is_empty() {
                return r < 8;
            }
            let min_keep = if max_tpl_rows >= 32 && r < max_tpl_rows {
                if t.values.len() >= 16 && (raw_rows > 0 || templates.len() > 2) {
                    12
                } else {
                    6
                }
            } else if max_tpl_rows >= 16 {
                4
            } else {
                2
            };
            r < min_keep
        });
        if should_prune {
            let mut remap = [usize::MAX; MAX_TEMPLATES];
            let mut demoted_cursor = [0usize; MAX_TEMPLATES];
            let mut new_templates = Vec::with_capacity(templates.len());
            let old_tpl_count = templates.len();
            let mut old_templates: Vec<Option<Template<'a>>> =
                templates.into_iter().map(Some).collect();
            for old_idx in 0..old_templates.len() {
                let keep = {
                    let tpl = old_templates[old_idx].as_ref().unwrap();
                    let r = tpl_counts[old_idx];
                    if tpl.values.is_empty() {
                        r >= 8
                    } else {
                        let min_keep = if max_tpl_rows >= 32 && r < max_tpl_rows {
                            if tpl.values.len() >= 16 && (raw_rows > 0 || old_tpl_count > 2) {
                                12
                            } else {
                                6
                            }
                        } else if max_tpl_rows >= 16 {
                            4
                        } else {
                            2
                        };
                        r >= min_keep
                    }
                };
                if keep {
                    let new_idx = new_templates.len();
                    remap[old_idx] = new_idx;
                    new_templates.push(old_templates[old_idx].take().unwrap());
                }
            }
            templates = new_templates;
            for r in &mut rows {
                if let Row::Template(old_idx) = *r {
                    let mapped = remap[old_idx];
                    if mapped == usize::MAX {
                        let tpl = old_templates[old_idx].as_ref().unwrap();
                        let raw_slice = if tpl.values.is_empty() {
                            tpl.segments[0]
                        } else {
                            let k = demoted_cursor[old_idx];
                            demoted_cursor[old_idx] = k + 1;
                            let val0 = tpl.values[0][k];
                            let seg0_len = tpl.segments[0].len();
                            let last_slot = tpl.values.len() - 1;
                            let val_last = tpl.values[last_slot][k];
                            let seg_last_len = tpl.segments[last_slot + 1].len();
                            let start_ptr = unsafe { val0.as_ptr().sub(seg0_len) };
                            let end_ptr =
                                unsafe { val_last.as_ptr().add(val_last.len() + seg_last_len) };
                            unsafe {
                                std::slice::from_raw_parts(
                                    start_ptr,
                                    end_ptr.offset_from(start_ptr) as usize,
                                )
                            }
                        };
                        *r = Row::Raw(raw_slice);
                        template_rows -= 1;
                        raw_rows += 1;
                        raw_bytes += raw_slice.len();
                    } else {
                        *r = Row::Template(mapped);
                    }
                }
            }
        }
    }
    Ok((templates, rows, template_rows, raw_rows, raw_bytes))
}

fn raw_packet(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + uleb_len(data.len()) + 1);
    out.push(RAW_PACKET);
    put_uleb(&mut out, data.len());
    out.extend_from_slice(data);
    out
}

#[inline(always)]
unsafe fn parse_dec4_at(ptr: *const u8) -> Option<u64> {
    let w = unsafe { ptr.cast::<u32>().read_unaligned().to_le() };
    let d = w ^ 0x3030_3030;
    if ((d | d.wrapping_add(0x0606_0606)) & 0xf0f0_f0f0) != 0 {
        return None;
    }
    let p2 = (d & 0x00ff_00ff) * 10 + ((d >> 8) & 0x00ff_00ff);
    Some(((p2 & 0xffff) * 100 + (p2 >> 16)) as u64)
}

#[inline(always)]
unsafe fn parse_dec8_at(ptr: *const u8) -> Option<u64> {
    let w = unsafe { ptr.cast::<u64>().read_unaligned().to_le() };
    let d = w ^ 0x3030_3030_3030_3030;
    if ((d | d.wrapping_add(0x0606_0606_0606_0606)) & 0xf0f0_f0f0_f0f0_f0f0) != 0 {
        return None;
    }
    let p2 = (d & 0x00ff_00ff_00ff_00ff) * 10 + ((d >> 8) & 0x00ff_00ff_00ff_00ff);
    let p4 = (p2 & 0x0000_ffff_0000_ffff) * 100 + ((p2 >> 16) & 0x0000_ffff_0000_ffff);
    Some((p4 & 0xffff_ffff) * 10_000 + (p4 >> 32))
}

