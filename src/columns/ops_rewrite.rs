pub(crate) enum GrammarRewrite {
    Untouched,
    Rewritten { payload: Vec<u8>, new_raw_len: usize },
    Fallback,
}

enum SlotColRewrite {
    Untouched,
    Rewritten { model_bytes: Vec<u8>, byte_delta: isize },
    Fallback,
}

#[inline]
fn is_valid_slot_token(col: crate::ops::ColTarget<'_>, tok: &[u8]) -> bool {
    if tok.is_empty() || tok.contains(&b'\n') || tok.contains(&b'\r') {
        return false;
    }
    match col {
        crate::ops::ColTarget::JsonKey(_) => scan_value(tok, 0) == Some(tok.len()),
        crate::ops::ColTarget::Delimited { delim, .. } => {
            scan_delimited_value(tok, 0, delim) == Some(tok.len())
        }
    }
}

#[inline]
fn reset_scratch_to(scratch: &mut DecodeScratch, raw_mark: usize, slice_mark: usize) {
    scratch.u8_arena.clear();
    scratch.u16_arena.clear();
    scratch.u32_arena.clear();
    scratch.u64_arena.clear();
    scratch.piece_arena.clear();
    scratch.str_arena.clear();
    scratch.slice_arena.truncate(slice_mark);
    scratch.raw_arena.truncate(raw_mark);
    scratch.unbound_col_arena.clear();
    scratch.exec_col_arena.clear();
}

fn rewrite_single_slot_column<'a>(
    col_bytes: &'a [u8],
    ones: usize,
    hits_opt: Option<&[usize]>,
    target_col: crate::ops::ColTarget<'_>,
    op: &crate::ops::ColOp<'_>,
    scratch: &mut DecodeScratch,
) -> Result<SlotColRewrite> {
    let col = PreparedColumn::parse_and_validate(col_bytes, ones, scratch)?;
    let const_tok: Option<Vec<u8>> = match col.const_full {
        ConstValue::Slice(s) => Some(s.to_vec()),
        ConstValue::Inline { ref buf, len } => Some(buf[..len as usize].to_vec()),
        ConstValue::QuotedDecimal {
            prefix,
            ref buf,
            len,
        } => {
            let mut tmp = Vec::with_capacity(prefix.len() + len as usize + 2);
            tmp.push(b'"');
            tmp.extend_from_slice(prefix);
            tmp.extend_from_slice(&buf[..len as usize]);
            tmp.push(b'"');
            Some(tmp)
        }
        ConstValue::None => None,
    };

    if let Some(ct) = const_tok {
        let Some(new_tok) = crate::ops::apply_token_op(target_col, &ct, op) else {
            return Ok(SlotColRewrite::Untouched);
        };
        if !is_valid_slot_token(target_col, &new_tok) {
            return Ok(SlotColRewrite::Fallback);
        }
        let d_one = new_tok.len() as isize - ct.len() as isize;
        let (refs, delta) = if let Some(hits) = hits_opt {
            let mut r = vec![ct.as_slice(); ones];
            for &h in hits {
                r[h] = new_tok.as_slice();
            }
            (r, d_one * (hits.len() as isize))
        } else {
            (vec![new_tok.as_slice(); ones], d_one * (ones as isize))
        };
        let (model_bytes, _) = pack_model_v3(&refs);
        return Ok(SlotColRewrite::Rewritten {
            model_bytes,
            byte_delta: delta,
        });
    }

    if hits_opt.is_none() {
        if let ColumnStream::Dictionary { u8_offset, .. } = col.stream {
            if let Some((slice_off, entries)) = col.dict_slices {
                if entries <= 256 {
                    let mut orig_entries = Vec::with_capacity(entries);
                    let mut mapped_entries: Vec<Option<Vec<u8>>> = Vec::with_capacity(entries);
                    let mut any_changed = false;
                    for e in 0..entries {
                        let (ptr, len) = scratch.slice_arena[slice_off + e];
                        let eb = unsafe { std::slice::from_raw_parts(ptr, len) };
                        orig_entries.push(eb);
                        let m = crate::ops::apply_token_op(target_col, eb, op);
                        if let Some(ref nt) = m {
                            if !is_valid_slot_token(target_col, nt) {
                                return Ok(SlotColRewrite::Fallback);
                            }
                            any_changed = true;
                        }
                        mapped_entries.push(m);
                    }
                    if !any_changed {
                        return Ok(SlotColRewrite::Untouched);
                    }
                    let mut entry_refs = Vec::with_capacity(entries);
                    let mut entry_deltas = Vec::with_capacity(entries);
                    for e in 0..entries {
                        if let Some(ref nt) = mapped_entries[e] {
                            entry_refs.push(nt.as_slice());
                            entry_deltas.push(nt.len() as isize - orig_entries[e].len() as isize);
                        } else {
                            entry_refs.push(orig_entries[e]);
                            entry_deltas.push(0);
                        }
                    }
                    let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];
                    let mut refs = Vec::with_capacity(ones);
                    let mut delta = 0isize;
                    for &id in ids {
                        let idx = id as usize;
                        refs.push(entry_refs[idx]);
                        delta += entry_deltas[idx];
                    }
                    let (model_bytes, _) = pack_model_v3(&refs);
                    return Ok(SlotColRewrite::Rewritten {
                        model_bytes,
                        byte_delta: delta,
                    });
                }
            }
        }
    }

    let unbound = UnboundTemplate::build(&[b"", b""], &[col], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let budget = tpl.row_budget.max(512) + 64;
    let mut flat: Vec<u8> = Vec::with_capacity(ones * 16 + budget);
    let mut spans = Vec::with_capacity(ones);
    for _ in 0..ones {
        let s = flat.len();
        flat.reserve(budget);
        let written = unsafe {
            let sp = flat.as_mut_ptr().add(s);
            let ep = tpl.emit_row(sp);
            ep.offset_from(sp) as usize
        };
        unsafe {
            flat.set_len(s + written);
        }
        spans.push((s, s + written));
    }

    let mut mods: Vec<(usize, Vec<u8>)> = Vec::new();
    if let Some(hits) = hits_opt {
        for &t in hits {
            let (s, e) = spans[t];
            if let Some(nt) = crate::ops::apply_token_op(target_col, &flat[s..e], op) {
                if !is_valid_slot_token(target_col, &nt) {
                    return Ok(SlotColRewrite::Fallback);
                }
                mods.push((t, nt));
            }
        }
    } else {
        for (t, &(s, e)) in spans.iter().enumerate() {
            if let Some(nt) = crate::ops::apply_token_op(target_col, &flat[s..e], op) {
                if !is_valid_slot_token(target_col, &nt) {
                    return Ok(SlotColRewrite::Fallback);
                }
                mods.push((t, nt));
            }
        }
    }
    if mods.is_empty() {
        return Ok(SlotColRewrite::Untouched);
    }
    let mut refs: Vec<&[u8]> = spans.iter().map(|&(s, e)| &flat[s..e]).collect();
    let mut delta = 0isize;
    for (idx, nt) in &mods {
        let (s, e) = spans[*idx];
        delta += nt.len() as isize - (e - s) as isize;
        refs[*idx] = nt.as_slice();
    }
    let (model_bytes, _) = pack_model_v3(&refs);
    Ok(SlotColRewrite::Rewritten {
        model_bytes,
        byte_delta: delta,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rewrite_grammar_block(
    payload: &[u8],
    had_incoming_carry: bool,
    has_following_block: bool,
    override_first_raw: Option<&[u8]>,
    override_last_raw: Option<&[u8]>,
    where_pred: Option<&crate::ops::ColPred<'_>>,
    target_col: crate::ops::ColTarget<'_>,
    op: &crate::ops::ColOp<'_>,
) -> Result<GrammarRewrite> {
    let Some(mut hdr) = parse_grammar_for_ops(payload)? else {
        return Ok(GrammarRewrite::Fallback);
    };
    if hdr.row_count == 0 {
        return Ok(GrammarRewrite::Untouched);
    }
    if had_incoming_carry && !hdr.first_is_raw {
        return Ok(GrammarRewrite::Fallback);
    }
    let p_raw_rows_start = 1 + hdr.cursor.pos;

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.raw_arena.reserve(hdr.raw_len.saturating_mul(3) + 32768);

        let raw_slice_offset = {
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            read_grammar_raw_rows(&mut hdr.cursor, hdr.raw_count, raw_arena, slice_arena)?
        };
        let p_cols_start = 1 + hdr.cursor.pos;
        let raw_arena_mark = scratch.raw_arena.len();
        let slice_arena_mark = scratch.slice_arena.len();

        let mut raw_overrides: Vec<Option<Vec<u8>>> = vec![None; hdr.raw_count];
        let mut any_raw_changed = false;
        let mut raw_byte_delta = 0isize;
        for (r, ov_slot) in raw_overrides.iter_mut().enumerate() {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset + r];
            let orig = unsafe { std::slice::from_raw_parts(ptr, len) };
            let next_opt = if r == 0 && had_incoming_carry && hdr.first_is_raw {
                override_first_raw.map(|b| b.to_vec())
            } else if r + 1 == hdr.raw_count
                && hdr.last_is_raw
                && !orig.ends_with(b"\n")
                && has_following_block
            {
                override_last_raw.map(|b| b.to_vec())
            } else {
                crate::ops::apply_line_col_op(orig, where_pred, target_col, op)
            };
            if let Some(next_b) = next_opt {
                if next_b.is_empty() {
                    return Ok(GrammarRewrite::Fallback);
                }
                if next_b.as_slice() != orig {
                    raw_byte_delta += next_b.len() as isize - orig.len() as isize;
                    *ov_slot = Some(next_b);
                    any_raw_changed = true;
                }
            }
        }

        let mut tpl_cols: Vec<Vec<(usize, usize, &[u8])>> = Vec::with_capacity(hdr.template_count);
        for t in 0..hdr.template_count {
            let slots = hdr.slot_counts[t];
            let mut cols_meta = Vec::with_capacity(slots);
            for _ in 0..slots {
                let entry_start = 1 + hdr.cursor.pos;
                let len = hdr.cursor.uleb()?;
                let cb = hdr.cursor.take(len)?;
                let entry_end = 1 + hdr.cursor.pos;
                cols_meta.push((entry_start, entry_end, cb));
            }
            tpl_cols.push(cols_meta);
        }
        if !hdr.cursor.done() {
            return Err(invalid("column packet trailing bytes"));
        }

        let mut spliced_cols: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        let mut total_col_delta = 0isize;

        for (t, cols_meta) in tpl_cols.iter().enumerate() {
            let ones = hdr.tpl_row_counts[t];
            if ones == 0 {
                continue;
            }
            let col_slices: Vec<&[u8]> = cols_meta.iter().map(|&(_, _, b)| b).collect();
            let segs = hdr.segs(t);
            let target_slot = match resolve_template_slot(segs, &col_slices, target_col)? {
                SlotLookup::Fallback => return Ok(GrammarRewrite::Fallback),
                SlotLookup::Absent => continue,
                SlotLookup::Slot(s) => s,
            };
            let hits_opt = if let Some(wp) = where_pred {
                let pred_slot = match resolve_template_slot(segs, &col_slices, wp.col)? {
                    SlotLookup::Fallback => return Ok(GrammarRewrite::Fallback),
                    SlotLookup::Absent => continue,
                    SlotLookup::Slot(s) => s,
                };
                let (_, hits) =
                    eval_col_pred_on_slot(col_slices[pred_slot], ones, ones, wp, &mut scratch)?;
                reset_scratch_to(&mut scratch, raw_arena_mark, slice_arena_mark);
                if hits.is_empty() {
                    continue;
                }
                Some(hits)
            } else {
                None
            };

            let (entry_start, entry_end, target_bytes) = cols_meta[target_slot];
            let res = rewrite_single_slot_column(
                target_bytes,
                ones,
                hits_opt.as_deref(),
                target_col,
                op,
                &mut scratch,
            )?;
            reset_scratch_to(&mut scratch, raw_arena_mark, slice_arena_mark);

            match res {
                SlotColRewrite::Untouched => {}
                SlotColRewrite::Fallback => return Ok(GrammarRewrite::Fallback),
                SlotColRewrite::Rewritten {
                    model_bytes,
                    byte_delta,
                } => {
                    total_col_delta += byte_delta;
                    spliced_cols.push((entry_start, entry_end, model_bytes));
                }
            }
        }

        if !any_raw_changed && spliced_cols.is_empty() {
            return Ok(GrammarRewrite::Untouched);
        }

        let new_raw_isize = hdr.raw_len as isize + raw_byte_delta + total_col_delta;
        if new_raw_isize <= 0 || new_raw_isize > MAX_BLOCK_SIZE as isize {
            return Ok(GrammarRewrite::Fallback);
        }
        let new_raw_len = new_raw_isize as usize;

        let mut out = Vec::with_capacity(payload.len() + 128);
        out.push(GRAMMAR_PACKET);
        put_uleb(&mut out, new_raw_len);
        out.extend_from_slice(&payload[hdr.p_after_raw_len..p_raw_rows_start]);
        if !any_raw_changed {
            out.extend_from_slice(&payload[p_raw_rows_start..p_cols_start]);
        } else {
            for (r, ov_opt) in raw_overrides.iter().enumerate() {
                if let Some(nb) = ov_opt {
                    put_bytes(&mut out, nb);
                } else {
                    let (ptr, len) = scratch.slice_arena[raw_slice_offset + r];
                    put_bytes(&mut out, unsafe { std::slice::from_raw_parts(ptr, len) });
                }
            }
        }
        let mut copy_pos = p_cols_start;
        for (es, ee, model_bytes) in spliced_cols {
            out.extend_from_slice(&payload[copy_pos..es]);
            put_uleb(&mut out, model_bytes.len());
            out.extend_from_slice(&model_bytes);
            copy_pos = ee;
        }
        out.extend_from_slice(&payload[copy_pos..payload.len()]);

        if out.len() < 2 || out.len() >= new_raw_len {
            return Ok(GrammarRewrite::Fallback);
        }
        Ok(GrammarRewrite::Rewritten {
            payload: out,
            new_raw_len,
        })
    })
}
