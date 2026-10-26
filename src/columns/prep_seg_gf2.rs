impl<'a> PreparedColumn<'a> {
    #[inline(never)]
    fn parse_segment_dict(
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        if count < 2 {
            return Err(invalid("column segment dictionary count"));
        }
        let dictionary_len = cursor.uleb()?;
        if dictionary_len > MAX_SEGMENT_DICT {
            return Err(invalid("column segment dictionary size"));
        }
        let mut dict_v = core::mem::MaybeUninit::<[[u128; 2]; MAX_SEGMENT_DICT * 2]>::uninit();
        let mut dict_len = core::mem::MaybeUninit::<[u8; MAX_SEGMENT_DICT * 2]>::uninit();
        let dict_v_ptr = dict_v.as_mut_ptr().cast::<[u128; 2]>();
        let dict_len_ptr = dict_len.as_mut_ptr().cast::<u8>();
        let mut prev_entry: Option<&'a [u8]> = None;
        let mut max_dict_entry_len = 0usize;
        for i in 0..dictionary_len {
            let len = cursor.uleb()?;
            let entry = cursor.take(len)?;
            if entry.is_empty()
                || entry.len() > MAX_SEGMENT_LEN
                || entry.iter().any(|&byte| !segment_byte_ok(byte))
                || canonical_segment_decimal(entry).is_some()
            {
                return Err(invalid("column segment dictionary entry"));
            }
            if prev_entry.is_some_and(|last| last >= entry) {
                return Err(invalid("column segment dictionary ordering"));
            }
            prev_entry = Some(entry);
            if len > max_dict_entry_len {
                max_dict_entry_len = len;
            }
            let mut tmp_buf = [0u8; 32];
            tmp_buf[..len].copy_from_slice(entry);
            unsafe {
                *dict_v_ptr.add(i) = [
                    u128::from_le_bytes(tmp_buf[0..16].try_into().unwrap()),
                    u128::from_le_bytes(tmp_buf[16..32].try_into().unwrap()),
                ];
                *dict_len_ptr.add(i) = len as u8;
            }

            let mut sep_buf = [0u8; 32];
            sep_buf[0] = b';';
            sep_buf[1] = b' ';
            sep_buf[2..2 + len].copy_from_slice(entry);
            let sep_idx = MAX_SEGMENT_DICT + i;
            unsafe {
                *dict_v_ptr.add(sep_idx) = [
                    u128::from_le_bytes(sep_buf[0..16].try_into().unwrap()),
                    u128::from_le_bytes(sep_buf[16..32].try_into().unwrap()),
                ];
                *dict_len_ptr.add(sep_idx) = (len + 2) as u8;
            }
        }
        let width = cursor.byte()?;
        if width != bit_width((dictionary_len + 1) as u64) {
            return Err(invalid("column segment code width"));
        }
        let code_count = cursor.uleb()?;
        if code_count == 0 || code_count > count.saturating_mul(MAX_SEGMENTS_PER_VALUE + 1) {
            return Err(invalid("column segment code count"));
        }
        let code_len = bit_bytes(code_count, width)?;
        let code_slice = cursor.take(code_len)?;
        let code_u8_offset = unpack_and_validate_u8(
            code_slice,
            code_count,
            width,
            0,
            dictionary_len + 2,
            &mut scratch.u8_arena,
        )?;

        let decimal_count = cursor.uleb()?;
        if decimal_count > code_count {
            return Err(invalid("column segment decimal count"));
        }
        let decimal_width = cursor.byte()?;
        if decimal_width > 30 {
            return Err(invalid("column segment decimal width"));
        }
        let decimal_len = bit_bytes(decimal_count, decimal_width)?;
        let decimal_slice = cursor.take(decimal_len)?;
        let dec_u8_opt = if decimal_count > 0 && (1..=7).contains(&decimal_width) {
            Some(unpack_and_validate_u8(
                decimal_slice,
                decimal_count,
                decimal_width,
                0,
                256,
                &mut scratch.u8_arena,
            )?)
        } else {
            None
        };
        let dec_u16_opt = if dec_u8_opt.is_none() && decimal_count > 0 && (8..=14).contains(&decimal_width) {
            Some(unpack_u16_swar(
                decimal_slice,
                decimal_count,
                decimal_width,
                0,
                &mut scratch.u16_arena,
            )?)
        } else {
            None
        };
        let mut decimal_reader = if dec_u8_opt.is_none() && dec_u16_opt.is_none() {
            FastBitReader::new_validated(decimal_slice, decimal_count, decimal_width)?
        } else {
            FastBitReader::new_validated(&[], 0, 0)?
        };

        let pad_width = cursor.byte()?;
        if pad_width > 12 {
            return Err(invalid("column segment pad width"));
        }
        let pad_len = bit_bytes(count, pad_width)?;
        let pad_slice = cursor.take(pad_len)?;
        if !cursor.done() {
            return Err(invalid("column model trailing bytes"));
        }
        let pad_u8_opt = if pad_width <= 7 {
            Some(unpack_and_validate_u8(
                pad_slice,
                count,
                pad_width,
                0,
                256,
                &mut scratch.u8_arena,
            )?)
        } else {
            None
        };
        let mut pad_reader = if pad_u8_opt.is_none() {
            FastBitReader::new_validated(pad_slice, count, pad_width)?
        } else {
            FastBitReader::new_validated(&[], 0, 0)?
        };

        let end_code = dictionary_len as u8;
        let _decimal_code = dictionary_len as u8 + 1;
        let mut or_dec = 0u64;
        let mut or_pad = 0u64;
        let mut term_count = 0usize;
        let mut dec_used = 0usize;
        let mut seg_in_val = 0usize;
        let mut sep_offset = 0usize;
        let mut max_val_len = 0usize;

        let max_seg = max_dict_entry_len.max(if decimal_count > 0 {
            MAX_DECIMAL_SEGMENT_DIGITS
        } else {
            0
        });
        let max_total_bytes = code_count
            .saturating_add(1)
            .saturating_mul(max_seg + 2)
            .saturating_add(count.saturating_mul((1usize << pad_width) - 1))
            .saturating_add(96);
        let str_u8_offset = scratch.u8_arena.len();
        scratch.u8_arena.reserve(max_total_bytes);
        let len_u16_offset = scratch.u16_arena.len();
        scratch.u16_arena.reserve(count);

        let str_base = unsafe { scratch.u8_arena.as_mut_ptr().add(str_u8_offset) };
        let mut dst_ptr = str_base;
        let mut row_start_ptr = str_base;
        let len_dst = unsafe { scratch.u16_arena.as_mut_ptr().add(len_u16_offset) };
        let codes_ptr = unsafe { scratch.u8_arena.as_ptr().add(code_u8_offset) };
        let dec_u8_ptr = dec_u8_opt
            .map(|off| unsafe { scratch.u8_arena.as_ptr().add(off) })
            .unwrap_or(core::ptr::null());
        let dec_u16_ptr = dec_u16_opt
            .map(|off| unsafe { scratch.u16_arena.as_ptr().add(off) })
            .unwrap_or(core::ptr::null());
        let pads_u8_ptr = pad_u8_opt
            .map(|off| unsafe { scratch.u8_arena.as_ptr().add(off) })
            .unwrap_or(core::ptr::null());

        macro_rules! run_fast_segment_loop {
            ($read_dec:ident) => {{
                let mut row_start_code = 0usize;
                for i in 0..code_count {
                    let code = unsafe { *codes_ptr.add(i) };
                    if code < end_code {
                        let idx = (code as usize) + sep_offset;
                        sep_offset = MAX_SEGMENT_DICT;
                        unsafe {
                            let l = *dict_len_ptr.add(idx) as usize;
                            let [v0, v1] = *dict_v_ptr.add(idx);
                            dst_ptr.cast::<u128>().write_unaligned(v0);
                            dst_ptr.add(16).cast::<u128>().write_unaligned(v1);
                            dst_ptr = dst_ptr.add(l);
                        }
                    } else if code == end_code {
                        if i - row_start_code > MAX_SEGMENTS_PER_VALUE {
                            return Err(invalid("column segment count"));
                        }
                        row_start_code = i + 1;
                        if term_count >= count {
                            return Err(invalid("column segment terminator count"));
                        }
                        let p = unsafe { *pads_u8_ptr.add(term_count) as u64 };
                        or_pad |= p;
                        let pad = p as usize;
                        unsafe {
                            dst_ptr
                                .cast::<u128>()
                                .write_unaligned(0x2020_2020_2020_2020_2020_2020_2020_2020);
                            dst_ptr = dst_ptr.add(pad);
                        }
                        let row_len = unsafe { dst_ptr.offset_from(row_start_ptr) as usize };
                        if row_len > max_val_len {
                            max_val_len = row_len;
                        }
                        unsafe {
                            *len_dst.add(term_count) = row_len as u16;
                        }
                        row_start_ptr = dst_ptr;
                        term_count += 1;
                        sep_offset = 0;
                    } else {
                        if dec_used >= decimal_count {
                            return Err(invalid("column segment decimal usage"));
                        }
                        let d = unsafe { *$read_dec.add(dec_used) as u64 };
                        dec_used += 1;
                        or_dec |= d;
                        unsafe {
                            dst_ptr
                                .cast::<u16>()
                                .write_unaligned(u16::from_le_bytes(*b"; "));
                            dst_ptr = dst_ptr.add(sep_offset >> 5);
                            sep_offset = MAX_SEGMENT_DICT;
                            let entry = *DEC_10K.get_unchecked(d as usize);
                            dst_ptr.cast::<u32>().write_unaligned(entry as u32);
                            dst_ptr = dst_ptr.add((entry >> 32) as usize);
                        }
                    }
                }
                if code_count - row_start_code > MAX_SEGMENTS_PER_VALUE {
                    return Err(invalid("column segment count"));
                }
                seg_in_val = code_count - row_start_code;
            }};
        }

        if pad_width <= 4 && !pads_u8_ptr.is_null() && (!dec_u8_ptr.is_null() || decimal_count == 0) {
            run_fast_segment_loop!(dec_u8_ptr);
        } else if pad_width <= 4 && !pads_u8_ptr.is_null() && !dec_u16_ptr.is_null() && decimal_width <= 13 {
            run_fast_segment_loop!(dec_u16_ptr);
        } else {
            for i in 0..code_count {
                let code = unsafe { *codes_ptr.add(i) };
                if code < end_code {
                    seg_in_val += 1;
                    if seg_in_val > MAX_SEGMENTS_PER_VALUE {
                        return Err(invalid("column segment count"));
                    }
                    let idx = (code as usize) + sep_offset;
                    sep_offset = MAX_SEGMENT_DICT;
                    unsafe {
                        let l = *dict_len_ptr.add(idx) as usize;
                        let [v0, v1] = *dict_v_ptr.add(idx);
                        dst_ptr.cast::<u128>().write_unaligned(v0);
                        dst_ptr.add(16).cast::<u128>().write_unaligned(v1);
                        dst_ptr = dst_ptr.add(l);
                    }
                } else if code == end_code {
                    if term_count >= count {
                        return Err(invalid("column segment terminator count"));
                    }
                    let p = if !pads_u8_ptr.is_null() {
                        unsafe { *pads_u8_ptr.add(term_count) as u64 }
                    } else {
                        pad_reader.next_u64()
                    };
                    or_pad |= p;
                    let pad = p as usize;
                    if pad != 0 {
                        unsafe {
                            if pad <= 16 {
                                dst_ptr
                                    .cast::<u128>()
                                    .write_unaligned(0x2020_2020_2020_2020_2020_2020_2020_2020);
                            } else {
                                std::ptr::write_bytes(dst_ptr, b' ', pad);
                            }
                            dst_ptr = dst_ptr.add(pad);
                        }
                    }
                    let row_len = unsafe { dst_ptr.offset_from(row_start_ptr) as usize };
                    if row_len > max_val_len {
                        max_val_len = row_len;
                    }
                    unsafe {
                        *len_dst.add(term_count) = row_len as u16;
                    }
                    row_start_ptr = dst_ptr;
                    term_count += 1;
                    seg_in_val = 0;
                    sep_offset = 0;
                } else {
                    seg_in_val += 1;
                    if seg_in_val > MAX_SEGMENTS_PER_VALUE {
                        return Err(invalid("column segment count"));
                    }
                    if dec_used >= decimal_count {
                        return Err(invalid("column segment decimal usage"));
                    }
                    let d = if !dec_u8_ptr.is_null() {
                        unsafe { *dec_u8_ptr.add(dec_used) as u64 }
                    } else if !dec_u16_ptr.is_null() {
                        unsafe { *dec_u16_ptr.add(dec_used) as u64 }
                    } else {
                        decimal_reader.next_u64()
                    };
                    dec_used += 1;
                    if decimal_width == 30 && d > 999_999_999 {
                        return Err(invalid("column segment decimal range"));
                    }
                    or_dec |= d;
                    unsafe {
                        if sep_offset != 0 {
                            dst_ptr
                                .cast::<u16>()
                                .write_unaligned(u16::from_le_bytes(*b"; "));
                            dst_ptr = dst_ptr.add(2);
                        }
                        sep_offset = MAX_SEGMENT_DICT;
                        if d < 10_000 {
                            let entry = *DEC_10K.get_unchecked(d as usize);
                            dst_ptr.cast::<u32>().write_unaligned(entry as u32);
                            dst_ptr = dst_ptr.add((entry >> 32) as usize);
                        } else {
                            let l = write_decimal_fast(dst_ptr, d);
                            dst_ptr = dst_ptr.add(l);
                        }
                    }
                }
            }
        }
        if seg_in_val != 0 {
            return Err(invalid("column segment missing terminator"));
        }
        if term_count != count {
            return Err(invalid("column segment terminator count"));
        }
        if dec_used != decimal_count {
            return Err(invalid("column segment decimal usage"));
        }
        if bit_width(or_dec) != decimal_width {
            return Err(invalid("column segment decimal width"));
        }
        if bit_width(or_pad) != pad_width {
            return Err(invalid("column segment pad width"));
        }
        let actual_bytes = unsafe { dst_ptr.offset_from(str_base) as usize };
        unsafe {
            scratch.u8_arena.set_len(str_u8_offset + actual_bytes);
            scratch.u16_arena.set_len(len_u16_offset + count);
        }

        return Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::QuoteOnly,
            const_trail: b"\"",
            dict_slices: None,
            max_emit_len: max_val_len,
            stream: ColumnStream::SegmentDict {
                str_u8_offset,
                len_u16_offset,
            },
        });
    }

    #[inline(never)]
    fn parse_gf2_affine(
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        if count <= GF2_TRAIN_TRANSITIONS + 1 {
            return Err(invalid("column gf2 recurrence count"));
        }
        let prefix_len = cursor.uleb()?;
        let prefix = cursor.take(prefix_len)?;
        let first = cursor.u64()?;
        let mut coefficients = [0u64; GF2_COEFFICIENTS];
        for coeff in coefficients.iter_mut() {
            *coeff = cursor.u64()?;
        }
        if !cursor.done() {
            return Err(invalid("column model trailing bytes"));
        }
        let lut = build_gf2_nibble_lut(&coefficients);
        let out_u8_offset = scratch.u8_arena.len();
        scratch.u8_arena.reserve(count * 16);
        let out_ptr = unsafe { scratch.u8_arena.as_mut_ptr().add(out_u8_offset) };
        let mut word = first;
        for r in 0..count {
            let w_bytes = word.to_le_bytes();
            unsafe {
                write_hex16_from_8bytes(out_ptr.add(r * 16), w_bytes.as_ptr());
            }
            word = apply_affine_lut(&lut, word);
        }
        unsafe {
            scratch.u8_arena.set_len(out_u8_offset + count * 16);
        }
        Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::QuotePrefix(prefix),
            const_trail: b"\"",
            dict_slices: None,
            max_emit_len: 16,
            stream: ColumnStream::Copy16 {
                u8_offset: out_u8_offset,
            },
        })
    }
}

