fn unpack_grammar(cursor: &mut Cursor<'_>, strict: bool) -> Result<Vec<u8>> {
    // Peek the raw length so the output buffer is allocated exactly once and
    // the assembly writes straight into it; the inner decoder re-reads and
    // re-validates the same length from the packet.
    let mut probe = Cursor::new(cursor.bytes);
    probe.pos = cursor.pos;
    let raw_len = probe.uleb()?;
    if raw_len > MAX_BLOCK_SIZE {
        return Err(invalid("column raw length"));
    }
    let mut out = Vec::with_capacity(raw_len + 64);
    let sink = OutSink::new(out.as_mut_ptr(), raw_len);
    let pos = unpack_grammar_inner(cursor, strict, sink)?;
    if pos != raw_len {
        return Err(invalid("column output length"));
    }
    unsafe {
        out.set_len(pos);
    }
    Ok(out)
}

fn unpack_grammar_inner(
    cursor: &mut Cursor<'_>,
    strict: bool,
    mut sink: OutSink,
) -> Result<usize> {
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE {
        return Err(invalid("column raw length"));
    }
    let template_count = cursor.uleb()?;
    if template_count > MAX_TEMPLATES {
        return Err(invalid("column template count"));
    }
    let dictionary_len = cursor.uleb()?;
    if dictionary_len > template_count.saturating_mul(MAX_SLOTS + 1) {
        return Err(invalid("column segment dictionary count"));
    }
    let mut seg_dict_inline = [core::mem::MaybeUninit::<&[u8]>::uninit(); 128];
    let mut seg_dict_overflow: Vec<&[u8]> = if dictionary_len > 128 {
        vec![&[]; dictionary_len]
    } else {
        Vec::new()
    };
    let segment_dictionary: &mut [&[u8]] = if dictionary_len <= 128 {
        unsafe {
            std::slice::from_raw_parts_mut(
                seg_dict_inline.as_mut_ptr().cast::<&[u8]>(),
                dictionary_len,
            )
        }
    } else {
        &mut seg_dict_overflow[..dictionary_len]
    };
    for i in 0..dictionary_len {
        let len = cursor.uleb()?;
        let segment = cursor.take(len)?;
        if i > 0 && segment_dictionary[i - 1] >= segment {
            return Err(invalid("column segment dictionary ordering"));
        }
        segment_dictionary[i] = segment;
    }
    let mut slot_counts_inline = [0usize; 16];
    let mut slot_counts_overflow = if template_count > 16 {
        vec![0usize; template_count]
    } else {
        Vec::new()
    };
    let slot_counts: &mut [usize] = if template_count <= 16 {
        &mut slot_counts_inline[..template_count]
    } else {
        &mut slot_counts_overflow[..template_count]
    };
    let mut max_slots = 0usize;
    for i in 0..template_count {
        let slots = cursor.uleb()?;
        if slots > MAX_SLOTS {
            return Err(invalid("column slot count"));
        }
        if slots > max_slots {
            max_slots = slots;
        }
        slot_counts[i] = slots;
    }
    let segment_width = cursor.byte()?;
    if segment_width != bit_width(dictionary_len.saturating_sub(1) as u64) {
        return Err(invalid("column segment selector width"));
    }
    let use_inline_tpl = template_count <= 16 && max_slots < 32;
    let mut tpl_inline_uninit = core::mem::MaybeUninit::<[[&[u8]; 32]; 16]>::uninit();
    let tpl_inline: &mut [[&[u8]; 32]; 16] = unsafe { tpl_inline_uninit.assume_init_mut() };
    let mut tpl_packed_inline: [&[u8]; 16] = [&[]; 16];
    let mut tpl_overflow: Vec<Vec<&[u8]>> = if !use_inline_tpl {
        Vec::with_capacity(template_count)
    } else {
        Vec::new()
    };
    for tpl_idx in 0..template_count {
        let slots = slot_counts[tpl_idx];
        if dictionary_len == 0 {
            return Err(invalid("empty column segment dictionary"));
        }
        let packed_len = bit_bytes(slots + 1, segment_width)?;
        let packed_slice = cursor.take(packed_len)?;
        let mut reader =
            FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
        if use_inline_tpl {
            for s in 0..=slots {
                let index = reader.next_u64() as usize;
                if index >= dictionary_len {
                    return Err(invalid("column segment dictionary index"));
                }
                tpl_inline[tpl_idx][s] = segment_dictionary[index];
            }
            for prev_idx in 0..tpl_idx {
                if slot_counts[prev_idx] == slots && tpl_packed_inline[prev_idx] == packed_slice {
                    return Err(invalid("duplicate column template"));
                }
            }
            tpl_packed_inline[tpl_idx] = packed_slice;
        } else {
            let mut segs = Vec::with_capacity(slots + 1);
            for _ in 0..=slots {
                let index = reader.next_u64() as usize;
                if index >= dictionary_len {
                    return Err(invalid("column segment dictionary index"));
                }
                segs.push(segment_dictionary[index]);
            }
            if tpl_overflow.contains(&segs) {
                return Err(invalid("duplicate column template"));
            }
            tpl_overflow.push(segs);
        }
    }
    let row_count = cursor.uleb()?;
    if row_count > raw_len.saturating_add(1) {
        return Err(invalid("column row count"));
    }
    let selector_width = cursor.byte()?;
    if selector_width != bit_width(template_count as u64) {
        return Err(invalid("column row selector width"));
    }
    let selector_len = bit_bytes(row_count, selector_width)?;
    let selector_slice = cursor.take(selector_len)?;

    if !strict && selector_width <= 7 {
        return DECODE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            scratch.raw_arena.reserve(raw_len.saturating_mul(2) + 16384);

            fn read_raw_rows_into<'a>(
                cursor: &mut Cursor<'a>,
                raw_count: usize,
                raw_arena: &mut Vec<u8>,
                slice_arena: &mut Vec<(*const u8, usize)>,
            ) -> Result<usize> {
                let raw_slice_offset = slice_arena.len();
                if raw_count == 0 {
                    return Ok(raw_slice_offset);
                }
                slice_arena.reserve(raw_count);
                if cursor.bytes.get(cursor.pos) == Some(&0) {
                    cursor.pos += 1;
                    let lz_len = cursor.uleb()?;
                    let lz_bytes = cursor.take(lz_len)?;
                    let mut lz_cursor = Cursor::new(lz_bytes);
                    if lz_cursor.byte()? != FAST_LZ_PACKET {
                        return Err(invalid("column raw rows lz selector"));
                    }
                    let mut probe = Cursor::new(lz_cursor.bytes);
                    probe.pos = lz_cursor.pos;
                    let unpacked_len = probe.uleb()?;
                    if unpacked_len == 0 || unpacked_len > 65536 {
                        return Err(invalid("column raw rows lz length"));
                    }
                    let arena_start = raw_arena.len();
                    if arena_start + unpacked_len > raw_arena.capacity() {
                        return Err(invalid("column raw rows arena overflow"));
                    }
                    unsafe {
                        raw_arena.set_len(arena_start + unpacked_len);
                    }
                    unpack_fast_lz_into(
                        &mut lz_cursor,
                        &mut raw_arena[arena_start..arena_start + unpacked_len],
                    )?;
                    let unpacked: &'a [u8] = unsafe {
                        std::slice::from_raw_parts(raw_arena.as_ptr().add(arena_start), unpacked_len)
                    };
                    let mut sub = Cursor::new(unpacked);
                    let lens_len = sub.uleb()?;
                    let mut len_cursor = Cursor::new(sub.take(lens_len)?);
                    for _ in 0..raw_count {
                        let len = len_cursor.uleb()?;
                        let raw = sub.take(len)?;
                        if raw.is_empty() {
                            return Err(invalid("empty raw column row"));
                        }
                        slice_arena.push((raw.as_ptr(), len));
                    }
                    if !len_cursor.done() || !sub.done() {
                        return Err(invalid("column raw rows lz trailing bytes"));
                    }
                    Ok(raw_slice_offset)
                } else {
                    for _ in 0..raw_count {
                        let len = cursor.uleb()?;
                        let raw = cursor.take(len)?;
                        if raw.is_empty() {
                            return Err(invalid("empty raw column row"));
                        }
                        slice_arena.push((raw.as_ptr(), len));
                    }
                    Ok(raw_slice_offset)
                }
            }

            let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
            let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };

            // Fast path 1: Single-template block (`template_count == 1 && selector_width == 1`)
            if template_count == 1 && selector_width == 1 {
                let _ = FastBitReader::new_validated(selector_slice, row_count, 1)?;
                let mut ones = 0usize;
                for &b in selector_slice {
                    ones += b.count_ones() as usize;
                }
                let raw_count = row_count - ones;
                let DecodeScratch {
                    ref mut raw_arena,
                    ref mut slice_arena,
                    ..
                } = *scratch;
                let raw_slice_offset =
                    read_raw_rows_into(cursor, raw_count, raw_arena, slice_arena)?;

                let slots = slot_counts[0];
                for s in 0..slots {
                    let len = cursor.uleb()?;
                    let bytes = cursor.take(len)?;
                    cols[s] = PreparedColumn::parse_and_validate(bytes, ones, &mut scratch)?;
                }
                if !cursor.done() {
                    return Err(invalid("column packet trailing bytes"));
                }
                let tpl_segs = if use_inline_tpl {
                    &tpl_inline[0][..=slots]
                } else {
                    &tpl_overflow[0][..]
                };
                let unbound = UnboundTemplate::build(tpl_segs, &cols[..slots], &mut scratch);
                let unbound_len = scratch.unbound_col_arena.len();
                scratch.exec_col_arena.reserve(unbound_len);
                let mut tpl = unbound.bind(&mut scratch);

                let base = sink.base;
                let capacity = sink.capacity;
                let mut pos = 0usize;
                let mut tail_buf = core::mem::MaybeUninit::<[u8; 4096]>::uninit();

                if raw_count == 0 {
                    emit_template_rows(
                        &mut tpl,
                        row_count,
                        base,
                        &mut pos,
                        capacity,
                        &mut tail_buf,
                    )?;
                } else {
                    let raw_slices_ptr =
                        unsafe { scratch.slice_arena.as_ptr().add(raw_slice_offset) };
                    let mut raw_idx = 0usize;
                    let mut row_idx = 0usize;
                    while row_idx < row_count {
                        let byte = selector_slice[row_idx >> 3];
                        let bit = (byte >> (row_idx & 7)) & 1;
                        if bit == 0 {
                            let (raw_ptr, len) = unsafe { *raw_slices_ptr.add(raw_idx) };
                            raw_idx += 1;
                            if pos + len > capacity {
                                return Err(invalid("column output exceeds length"));
                            }
                            unsafe {
                                std::ptr::copy_nonoverlapping(raw_ptr, base.add(pos), len);
                            }
                            pos += len;
                            row_idx += 1;
                        } else {
                            let run_start = row_idx;
                            let bit_off = row_idx & 7;
                            if bit_off != 0 {
                                let shifted = byte >> bit_off;
                                let ones_in_byte = (!shifted).trailing_zeros() as usize;
                                let rem_in_byte = 8 - bit_off;
                                if ones_in_byte < rem_in_byte {
                                    row_idx = (row_idx + ones_in_byte).min(row_count);
                                    emit_template_rows(
                                        &mut tpl,
                                        row_idx - run_start,
                                        base,
                                        &mut pos,
                                        capacity,
                                        &mut tail_buf,
                                    )?;
                                    continue;
                                }
                                row_idx = (row_idx + rem_in_byte).min(row_count);
                            }
                            while row_idx + 8 <= row_count && selector_slice[row_idx >> 3] == 0xff
                            {
                                row_idx += 8;
                            }
                            if row_idx < row_count {
                                let b = selector_slice[row_idx >> 3];
                                let leading_ones = (!b).trailing_zeros() as usize;
                                row_idx = (row_idx + leading_ones).min(row_count);
                            }
                            emit_template_rows(
                                &mut tpl,
                                row_idx - run_start,
                                base,
                                &mut pos,
                                capacity,
                                &mut tail_buf,
                            )?;
                        }
                    }
                }

                if pos != raw_len {
                    return Err(invalid("column output length"));
                }
                return Ok(pos);
            }

            // Fast path 2: Multi-template or 0-template blocks (`template_count != 1 && selector_width <= 7`)
            let sel_u8_offset = unpack_and_validate_u8(
                selector_slice,
                row_count,
                selector_width,
                0,
                template_count + 1,
                &mut scratch.u8_arena,
            )?;
            let mut sel_counts = [0usize; 129];
            for &sel in &scratch.u8_arena[sel_u8_offset..sel_u8_offset + row_count] {
                sel_counts[sel as usize] += 1;
            }
            let raw_total = sel_counts[0];
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            let raw_slice_offset =
                read_raw_rows_into(cursor, raw_total, raw_arena, slice_arena)?;

            let mut unbound_inline = [UnboundTemplate::default(); 16];
            let mut unbound_overflow = if template_count > 16 {
                vec![UnboundTemplate::default(); template_count]
            } else {
                Vec::new()
            };
            let unbound_templates: &mut [UnboundTemplate] = if template_count <= 16 {
                &mut unbound_inline[..template_count]
            } else {
                &mut unbound_overflow[..template_count]
            };

            for tpl_idx in 0..template_count {
                let slots = slot_counts[tpl_idx];
                let count = sel_counts[tpl_idx + 1];
                for s in 0..slots {
                    let len = cursor.uleb()?;
                    let bytes = cursor.take(len)?;
                    cols[s] = PreparedColumn::parse_and_validate(bytes, count, &mut scratch)?;
                }
                let tpl_segs = if use_inline_tpl {
                    &tpl_inline[tpl_idx][..=slots]
                } else {
                    &tpl_overflow[tpl_idx][..]
                };
                unbound_templates[tpl_idx] =
                    UnboundTemplate::build(tpl_segs, &cols[..slots], &mut scratch);
            }
            if !cursor.done() {
                return Err(invalid("column packet trailing bytes"));
            }

            let total_unbound_cols = scratch.unbound_col_arena.len();
            scratch.exec_col_arena.reserve(total_unbound_cols);
            let mut prepared_inline_uninit =
                core::mem::MaybeUninit::<[PreparedTemplate; 16]>::uninit();
            let mut prepared_overflow = if template_count > 16 {
                vec![PreparedTemplate::empty(); template_count]
            } else {
                Vec::new()
            };
            let prepared_templates: &mut [PreparedTemplate] = if template_count <= 16 {
                unsafe {
                    std::slice::from_raw_parts_mut(
                        prepared_inline_uninit.as_mut_ptr().cast::<PreparedTemplate>(),
                        template_count,
                    )
                }
            } else {
                &mut prepared_overflow[..template_count]
            };
            for tpl_idx in 0..template_count {
                prepared_templates[tpl_idx] = unbound_templates[tpl_idx].bind(&mut scratch);
            }

            let base = sink.base;
            let capacity = sink.capacity;
            let end_cap = unsafe { base.add(capacity) };
            let mut dst = base;
            let mut raw_idx = 0usize;
            let mut tail_buf = core::mem::MaybeUninit::<[u8; 4096]>::uninit();
            let sel_ptr = unsafe { scratch.u8_arena.as_ptr().add(sel_u8_offset) };
            let raw_slices_ptr =
                unsafe { scratch.slice_arena.as_ptr().add(raw_slice_offset) };
            let tpl_ptr = prepared_templates.as_mut_ptr();

            for r in 0..row_count {
                let sel = unsafe { *sel_ptr.add(r) } as usize;
                if sel != 0 {
                    let tpl = unsafe { &mut *tpl_ptr.add(sel - 1) };
                    if unsafe { dst.add(tpl.row_budget) } <= end_cap {
                        dst = unsafe { tpl.emit_row(dst) };
                    } else {
                        let mut pos = unsafe { dst.offset_from(base) as usize };
                        emit_template_row_tail(
                            tpl,
                            tpl.row_budget,
                            base,
                            &mut pos,
                            capacity,
                            &mut tail_buf,
                        )?;
                        dst = unsafe { base.add(pos) };
                    }
                } else {
                    let (raw_ptr, len) = unsafe { *raw_slices_ptr.add(raw_idx) };
                    raw_idx += 1;
                    if unsafe { dst.add(len) } > end_cap {
                        return Err(invalid("column output exceeds length"));
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(raw_ptr, dst, len);
                        dst = dst.add(len);
                    }
                }
            }

            let pos = unsafe { dst.offset_from(base) as usize };
            if pos != raw_len {
                return Err(invalid("column output length"));
            }
            Ok(pos)
        });
    }

    let mut selector_reader =
        FastBitReader::new_validated(selector_slice, row_count, selector_width)?;
    let mut template_row_counts = vec![0usize; template_count];
    let mut rows = Vec::<Row<'_>>::with_capacity(row_count);
    let all_template_zero = false;
    for _ in 0..row_count {
        let selector = selector_reader.next_u64() as usize;
        if selector == 0 {
            let len = cursor.uleb()?;
            let raw = cursor.take(len)?;
            if raw.is_empty() {
                return Err(invalid("empty raw column row"));
            }
            rows.push(Row::Raw(raw));
        } else {
            let index = selector - 1;
            let count = template_row_counts
                .get_mut(index)
                .ok_or_else(|| invalid("column row template"))?;
            *count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
            rows.push(Row::Template(index));
        }
    }
    let templates: Vec<Vec<&[u8]>> = if use_inline_tpl {
        (0..template_count)
            .map(|idx| tpl_inline[idx][..=slot_counts[idx]].to_vec())
            .collect()
    } else {
        tpl_overflow
    };

    // Strict oracle path (`strict == true`): materializes `FlatValues` and verifies
    // full re-encode canonicality per column.
    if all_template_zero {
        rows.resize(row_count, Row::Template(0));
    }
    let column_counts: Vec<Vec<usize>> = templates
        .iter()
        .enumerate()
        .map(|(idx, template)| {
            vec![template_row_counts[idx]; template.len().saturating_sub(1)]
        })
        .collect();
    let mut values = Vec::with_capacity(template_count);
    for (template_index, counts) in column_counts.iter().enumerate() {
        let mut template_values = Vec::with_capacity(counts.len());
        for &count in counts {
            let len = cursor.uleb()?;
            let bytes = cursor.take(len)?;
            template_values.push(unpack_model_mode(bytes, count, strict)?);
            let _ = template_index;
        }
        values.push(template_values);
    }
    if !cursor.done() {
        return Err(invalid("column packet trailing bytes"));
    }
    let mut positions = templates
        .iter()
        .map(|template| vec![0usize; template.len().saturating_sub(1)])
        .collect::<Vec<_>>();
    for (template_index, template_values) in values.iter().enumerate() {
        for (slot, column) in template_values.iter().enumerate() {
            if column.len() != column_counts[template_index][slot] {
                return Err(invalid("column value count"));
            }
        }
    }
    for row in rows {
        match row {
            Row::Raw(raw) => sink.write(raw)?,
            Row::Template(index) => {
                let segments = &templates[index];
                sink.write(&segments[0])?;
                let column = &values[index];
                let cursor = &mut positions[index];
                let slots = column.len();
                let segment_heads = segments.as_ptr();
                let columns = column.as_ptr();
                let cursors = cursor.as_mut_ptr();
                for slot in 0..slots {
                    unsafe {
                        let position = *cursors.add(slot);
                        let values = &*columns.add(slot);
                        let &(start, len) = values.ranges.get_unchecked(position);
                        let start = start as usize;
                        sink.write(values.bytes.get_unchecked(start..start + len as usize))?;
                        *cursors.add(slot) = position + 1;
                        sink.write(&*segment_heads.add(slot + 1))?;
                    }
                }
            }
        }
        if sink.pos > raw_len {
            return Err(invalid("column output exceeds length"));
        }
    }
    if positions
        .iter()
        .zip(column_counts)
        .any(|(positions, counts)| positions != &counts)
    {
        return Err(invalid("column value consumption"));
    }
    if sink.pos != raw_len {
        return Err(invalid("column output length"));
    }
    Ok(sink.pos)
}

