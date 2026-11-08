pub(crate) enum GrammarProjectOut {
    Empty,
    Spliced {
        payload: Vec<u8>,
        new_raw_len: usize,
        crc: u32,
    },
    Bytes(Vec<u8>),
    Fallback,
}

enum TemplateProjPlan {
    SkipAll,
    ConstJsonEmpty { crlf: bool },
    Slots {
        seg_bytes: Vec<u8>,
        seg_ranges: Vec<(usize, usize)>,
        slot_map: Vec<Option<usize>>,
        can_splice: bool,
    },
}

fn plan_template_projection(
    segs: &[&[u8]],
    tcols: &[&[u8]],
    cols: &[crate::ops::ColTarget<'_>],
) -> Result<Option<TemplateProjPlan>> {
    match cols[0] {
        crate::ops::ColTarget::JsonKey(first_k) => {
            if segs.last().is_some_and(|s| s.ends_with(b"\n"))
                && json_single_line_template_slot(segs, first_k.as_bytes()).is_some()
            {
                let crlf = segs.last().is_some_and(|s| s.ends_with(b"\r\n"));
                let mut present: Vec<(&str, usize)> = Vec::with_capacity(cols.len());
                for &col in cols {
                    let crate::ops::ColTarget::JsonKey(k) = col else {
                        unreachable!();
                    };
                    match json_single_line_template_slot(segs, k.as_bytes()) {
                        Some(Some(s)) => present.push((k, s)),
                        Some(None) => {}
                        None => return Ok(None),
                    }
                }
                if present.is_empty() {
                    return Ok(Some(TemplateProjPlan::ConstJsonEmpty { crlf }));
                }
                let mut seg_bytes = Vec::with_capacity(present.len() * 16 + 4);
                let mut seg_ranges = Vec::with_capacity(present.len() + 1);
                for (j, &(k, _)) in present.iter().enumerate() {
                    let st = seg_bytes.len();
                    seg_bytes.push(if j == 0 { b'{' } else { b',' });
                    crate::ops::encode_json_str(k.as_bytes(), &mut seg_bytes);
                    seg_bytes.push(b':');
                    seg_ranges.push((st, seg_bytes.len()));
                }
                let st = seg_bytes.len();
                seg_bytes.extend_from_slice(if crlf { b"}\r\n" } else { b"}\n" });
                seg_ranges.push((st, seg_bytes.len()));
                let slot_map = present.into_iter().map(|(_, s)| Some(s)).collect();
                Ok(Some(TemplateProjPlan::Slots {
                    seg_bytes,
                    seg_ranges,
                    slot_map,
                    can_splice: true,
                }))
            } else if !can_template_match_fast(segs, tcols, b"{")? {
                Ok(Some(TemplateProjPlan::SkipAll))
            } else {
                Ok(None)
            }
        }
        crate::ops::ColTarget::Delimited { delim, .. } => {
            if delim.is_ascii_digit() {
                return Ok(None);
            }
            if delimited_template_delim(segs) == Some(delim) {
                let crlf = segs.last().is_some_and(|s| s.ends_with(b"\r\n"));
                let m = cols.len();
                let mut seg_bytes = Vec::with_capacity(m + 2);
                let mut seg_ranges = Vec::with_capacity(m + 1);
                seg_ranges.push((0, 0));
                for _ in 1..m {
                    let st = seg_bytes.len();
                    seg_bytes.push(delim);
                    seg_ranges.push((st, st + 1));
                }
                let st = seg_bytes.len();
                seg_bytes.extend_from_slice(if crlf { b"\r\n" } else { b"\n" });
                seg_ranges.push((st, seg_bytes.len()));
                let mut slot_map = Vec::with_capacity(m);
                let mut can_splice = true;
                for &col in cols {
                    let crate::ops::ColTarget::Delimited { index, .. } = col else {
                        unreachable!();
                    };
                    if index < tcols.len() {
                        slot_map.push(Some(index));
                    } else {
                        slot_map.push(None);
                        can_splice = false;
                    }
                }
                Ok(Some(TemplateProjPlan::Slots {
                    seg_bytes,
                    seg_ranges,
                    slot_map,
                    can_splice,
                }))
            } else if !can_template_match_fast(segs, tcols, &[delim])? {
                Ok(Some(TemplateProjPlan::SkipAll))
            } else {
                Ok(None)
            }
        }
    }
}

fn prepare_projected_unbound<'a>(
    seg_bytes: &'a [u8],
    seg_ranges: &[(usize, usize)],
    slot_map: &[Option<usize>],
    tcols: &[&'a [u8]],
    ones: usize,
    hdr_raw_len: usize,
    scratch: &mut DecodeScratch,
) -> Result<UnboundTemplate> {
    let m = slot_map.len();
    let mut seg_refs: [&'a [u8]; MAX_SLOTS + 1] = [&[]; MAX_SLOTS + 1];
    for (j, &(s, e)) in seg_ranges.iter().enumerate() {
        seg_refs[j] = &seg_bytes[s..e];
    }
    let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn<'a>; MAX_SLOTS]>::uninit();
    let cols_arr: &mut [PreparedColumn<'a>; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
    let mut any_raw_stream = false;
    for (j, &s_opt) in slot_map.iter().enumerate() {
        if let Some(s) = s_opt {
            ensure_col_raw_arena(tcols[s], hdr_raw_len, &mut scratch.raw_arena);
            let pc = PreparedColumn::parse_and_validate(tcols[s], ones, scratch)?;
            if matches!(pc.stream, ColumnStream::Raw { .. })
                && matches!(pc.const_full, ConstValue::None)
            {
                any_raw_stream = true;
            }
            cols_arr[j] = pc;
        } else {
            cols_arr[j] = PreparedColumn {
                const_full: ConstValue::Slice(b""),
                const_lead: LeadValue::Empty,
                const_trail: b"",
                dict_slices: None,
                max_emit_len: 0,
                stream: ColumnStream::Raw {
                    cursor: Cursor::new(&[]),
                },
            };
        }
    }
    if any_raw_stream {
        ensure_full_raw_arena(hdr_raw_len, &mut scratch.raw_arena);
    }
    Ok(UnboundTemplate::build(&seg_refs[..=m], &cols_arr[..m], scratch))
}

#[inline]
fn emit_tpl_rows_all(tpl: &mut PreparedTemplate, n: usize, dst: &mut Vec<u8>) {
    let budget = tpl.row_budget.max(1024) + 64;
    dst.reserve(n * 32 + budget);
    for _ in 0..n {
        let pos = dst.len();
        dst.reserve(budget);
        unsafe {
            let sp = dst.as_mut_ptr().add(pos);
            let ep = tpl.emit_row(sp);
            dst.set_len(pos + ep.offset_from(sp) as usize);
        }
    }
}

fn build_spliced_grammar_packet(
    new_raw_len: usize,
    ones: usize,
    opt_tail_row: Option<&[u8]>,
    seg_bytes: &[u8],
    seg_ranges: &[(usize, usize)],
    slot_map: &[Option<usize>],
    tcols: &[&[u8]],
) -> Vec<u8> {
    let m = slot_map.len();
    let total_rows = ones + usize::from(opt_tail_row.is_some());
    let tail_cap = opt_tail_row.map_or(0, |t| 5 + t.len());
    let proj_segs: Vec<&[u8]> = seg_ranges.iter().map(|&(s, e)| &seg_bytes[s..e]).collect();
    let mut dict = proj_segs.clone();
    dict.sort_unstable();
    dict.dedup();
    let seg_width = bit_width(dict.len().saturating_sub(1) as u64);
    let col_sum: usize = slot_map.iter().map(|s| 5 + tcols[s.unwrap()].len()).sum();
    let mut out = Vec::with_capacity(64 + seg_bytes.len() + (total_rows / 8 + 4) + tail_cap + col_sum);
    out.push(GRAMMAR_PACKET);
    put_uleb(&mut out, new_raw_len);
    put_uleb(&mut out, 1);
    put_uleb(&mut out, dict.len());
    for &seg in &dict {
        put_bytes(&mut out, seg);
    }
    put_uleb(&mut out, m);
    out.push(seg_width);
    push_bit_values(
        &mut out,
        proj_segs.len(),
        seg_width,
        proj_segs.iter().map(|s| dict.binary_search(s).unwrap() as u64),
    );
    put_uleb(&mut out, total_rows);
    out.push(1);
    out.resize(out.len() + ones / 8, 0xff);
    if opt_tail_row.is_some() || (ones & 7) != 0 {
        out.push((1u8 << (ones & 7)) - 1);
    }
    if let Some(tail) = opt_tail_row {
        put_bytes(&mut out, tail);
    }
    for s_opt in slot_map {
        put_bytes(&mut out, tcols[s_opt.unwrap()]);
    }
    out
}
