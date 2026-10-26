impl<'a> PreparedColumn<'a> {
    fn parse_timestamp_or_periodic(
        tag: u8,
        bytes: &'a [u8],
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        match tag {
            MODEL_TIMESTAMP => {
                if count < 2 {
                    return Err(invalid("column timestamp count"));
                }
                let year = cursor.u16()?;
                if year > 9999 {
                    return Err(invalid("column timestamp year"));
                }
                let mut bases = [0u8; 5];
                let mut widths = [0u8; 5];
                for i in 0..5 {
                    let base = cursor.byte()?;
                    let width = cursor.byte()?;
                    if width > 6 {
                        return Err(invalid("column timestamp width"));
                    }
                    bases[i] = base;
                    widths[i] = width;
                }
                let min_bounds = [1u8, 1, 0, 0, 0];
                let max_bounds = [12u8, 31, 23, 59, 59];
                let mut slices: [&'a [u8]; 5] = [&[]; 5];
                for i in 0..5 {
                    if bases[i] < min_bounds[i] || bases[i] > max_bounds[i] {
                        return Err(invalid("column timestamp component"));
                    }
                    let packed_len = bit_bytes(count, widths[i])?;
                    slices[i] = cursor.take(packed_len)?;
                }
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                if (widths[0] | widths[1] | widths[2]) == 0 {
                    // Validate zero-width slices 0..3:
                    for i in 0..3 {
                        if !slices[i].is_empty() {
                            return Err(Error::Truncated);
                        }
                    }
                    if (widths[3] | widths[4]) == 0 {
                        if !slices[3].is_empty() || !slices[4].is_empty() {
                            return Err(Error::Truncated);
                        }
                        let mut buf = [0u8; 24];
                        let mut v = Vec::with_capacity(22);
                        v.push(b'"');
                        write_four_digits(&mut v, year);
                        v.push(b'-');
                        write_two_digits(&mut v, bases[0]);
                        v.push(b'-');
                        write_two_digits(&mut v, bases[1]);
                        v.push(b'T');
                        write_two_digits(&mut v, bases[2]);
                        v.push(b':');
                        write_two_digits(&mut v, bases[3]);
                        v.push(b':');
                        write_two_digits(&mut v, bases[4]);
                        v.extend_from_slice(b"Z\"");
                        buf[..22].copy_from_slice(&v);
                        return Ok(Self {
                            const_full: ConstValue::Inline { buf, len: 22 },
                            const_lead: LeadValue::Empty,
                            const_trail: b"",
                            dict_slices: None,
                            max_emit_len: 0,
                            stream: ColumnStream::Raw {
                                cursor: Cursor::new(&[]),
                            },
                        });
                    }
                    let min_u8_offset = unpack_and_validate_u8(
                        slices[3],
                        count,
                        widths[3],
                        bases[3],
                        (max_bounds[3] - bases[3]) as usize + 1,
                        &mut scratch.u8_arena,
                    )?;
                    let sec_u8_offset = unpack_and_validate_u8(
                        slices[4],
                        count,
                        widths[4],
                        bases[4],
                        (max_bounds[4] - bases[4]) as usize + 1,
                        &mut scratch.u8_arena,
                    )?;
                    let mut lead = [0u8; 16];
                    lead[0] = b'"';
                    lead[1] = b'0' + (year / 1000) as u8;
                    lead[2] = b'0' + (year / 100 % 10) as u8;
                    lead[3] = b'0' + (year / 10 % 10) as u8;
                    lead[4] = b'0' + (year % 10) as u8;
                    lead[5] = b'-';
                    lead[6] = b'0' + bases[0] / 10;
                    lead[7] = b'0' + bases[0] % 10;
                    lead[8] = b'-';
                    lead[9] = b'0' + bases[1] / 10;
                    lead[10] = b'0' + bases[1] % 10;
                    lead[11] = b'T';
                    lead[12] = b'0' + bases[2] / 10;
                    lead[13] = b'0' + bases[2] % 10;
                    lead[14] = b':';
                    return Ok(Self {
                        const_full: ConstValue::None,
                        const_lead: LeadValue::Inline { buf: lead, len: 15 },
                        const_trail: b"Z\"",
                        dict_slices: None,
                        max_emit_len: 5,
                        stream: ColumnStream::TimestampMinSec {
                            min_u8_offset,
                            sec_u8_offset,
                        },
                    });
                }
                let mut u8_offset = 0usize;
                for i in 0..5 {
                    let off = unpack_and_validate_u8(
                        slices[i],
                        count,
                        widths[i],
                        bases[i],
                        (max_bounds[i] - bases[i]) as usize + 1,
                        &mut scratch.u8_arena,
                    )?;
                    if i == 0 {
                        u8_offset = off;
                    }
                }
                let mut lead = [0u8; 16];
                lead[0] = b'"';
                lead[1] = b'0' + (year / 1000) as u8;
                lead[2] = b'0' + (year / 100 % 10) as u8;
                lead[3] = b'0' + (year / 10 % 10) as u8;
                lead[4] = b'0' + (year % 10) as u8;
                lead[5] = b'-';
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Inline { buf: lead, len: 6 },
                    const_trail: b"Z\"",
                    dict_slices: None,
                    max_emit_len: 14,
                    stream: ColumnStream::TimestampFull {
                        u8_offset,
                        stride: count,
                    },
                })
            }
            MODEL_PERIODIC_STRING => {
                if count < 2 {
                    return Err(invalid("column periodic count"));
                }
                let period = cursor.uleb()?;
                if !(1..=64).contains(&period) {
                    return Err(invalid("column periodic period"));
                }
                let body_start = cursor.pos;
                let mut max_len = 0usize;
                for _ in 0..count {
                    let len = cursor.uleb()?;
                    if len > MAX_BLOCK_SIZE {
                        return Err(invalid("column periodic length"));
                    }
                    let seed_len = len.min(period);
                    let _ = cursor.take(seed_len)?;
                    if len > max_len {
                        max_len = len;
                    }
                }
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let mut stream_cursor = Cursor::new(bytes);
                stream_cursor.pos = body_start;
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::QuoteOnly,
                    const_trail: b"\"",
                    dict_slices: None,
                    max_emit_len: max_len,
                    stream: ColumnStream::PeriodicString {
                        cursor: stream_cursor,
                        period,
                    },
                })
            }
            MODEL_PERIODIC_HEX => {
                if count < 2 {
                    return Err(invalid("column periodic hex count"));
                }
                let period = cursor.uleb()?;
                if !(1..=64).contains(&period) {
                    return Err(invalid("column periodic hex period"));
                }
                let body_start = cursor.pos;
                let mut max_len = 0usize;
                for _ in 0..count {
                    let len = cursor.uleb()?;
                    if len > MAX_BLOCK_SIZE {
                        return Err(invalid("column periodic hex length"));
                    }
                    let seed_len = len.min(period);
                    let packed_len = seed_len.div_ceil(2);
                    let packed = cursor.take(packed_len)?;
                    if (seed_len & 1) == 1 && (packed[packed_len - 1] & 15) != 0 {
                        return Err(invalid("nonzero periodic hex tail"));
                    }
                    if len > max_len {
                        max_len = len;
                    }
                }
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let mut stream_cursor = Cursor::new(bytes);
                stream_cursor.pos = body_start;
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::QuoteOnly,
                    const_trail: b"\"",
                    dict_slices: None,
                    max_emit_len: max_len,
                    stream: ColumnStream::PeriodicHex {
                        cursor: stream_cursor,
                        period,
                    },
                })
            }
            _ => Err(invalid("column model selector")),
        }
    }

    #[inline(never)]
    fn parse_epoch_counter(
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        if count < 2 {
            return Err(invalid("column epoch counter count"));
        }
        let lit_len = cursor.uleb()?;
        let literal = cursor.take(lit_len)?;
        let prefix_count = cursor.uleb()?;
        if prefix_count == 0 || prefix_count > MAX_EPOCH_PREFIXES {
            return Err(invalid("column epoch prefix count"));
        }
        let mut prefix_hex = [0u64; MAX_EPOCH_PREFIXES];
        let mut prev_val: Option<u32> = None;
        for i in 0..prefix_count {
            let b = cursor.take(4)?;
            let val = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            if prev_val.is_some_and(|last| last >= val) {
                return Err(invalid("column epoch prefix ordering"));
            }
            prev_val = Some(val);
            prefix_hex[i] = u32_to_hex8_le(val);
        }
        let width = cursor.byte()?;
        if width != bit_width((prefix_count - 1) as u64) {
            return Err(invalid("column epoch prefix width"));
        }
        let index_len = bit_bytes(count, width)?;
        let index_slice = cursor.take(index_len)?;
        let idx_u8_offset = unpack_and_validate_u8(
            index_slice,
            count,
            width,
            0,
            prefix_count,
            &mut scratch.u8_arena,
        )?;
        let flag_len = bit_bytes(count, 1)?;
        let flag_slice = cursor.take(flag_len)?;
        let _ = FastBitReader::new_validated(flag_slice, count, 1)?;
        if (flag_slice.first().copied().unwrap_or(0) & 1) == 0 {
            return Err(invalid("column epoch first explicit"));
        }
        let explicit_count = cursor.uleb()?;
        let mut ones_count = 0usize;
        for &b in flag_slice {
            ones_count += b.count_ones() as usize;
        }
        if explicit_count != ones_count {
            return Err(invalid("column epoch explicit count"));
        }
        let out_u8_offset = scratch.u8_arena.len();
        scratch.u8_arena.reserve(count * 16);
        let out_ptr = unsafe { scratch.u8_arena.as_mut_ptr().add(out_u8_offset) };
        let idx_ptr = unsafe { scratch.u8_arena.as_ptr().add(idx_u8_offset) };
        let mut prev_s = 0u32;
        let mut prev_s_hex = 0u64;
        for r in 0..count {
            let p_hex = unsafe { *prefix_hex.get_unchecked(*idx_ptr.add(r) as usize) };
            let flag = (unsafe { *flag_slice.get_unchecked(r >> 3) } >> (r & 7)) & 1;
            let s_hex = if flag != 0 {
                let v = cursor.uleb()?;
                if v > u32::MAX as usize {
                    return Err(invalid("column epoch suffix range"));
                }
                prev_s = v as u32;
                prev_s_hex = u32_to_hex8_le(prev_s);
                prev_s_hex
            } else {
                let next = prev_s
                    .checked_add(1)
                    .ok_or_else(|| invalid("column epoch suffix range"))?;
                prev_s = next;
                prev_s_hex = if (next & 0xff) != 0 {
                    let b3 = unsafe { *HEX_BYTE_LUT.get_unchecked((next & 0xff) as usize) } as u64;
                    (prev_s_hex & 0x0000_ffff_ffff_ffff) | (b3 << 48)
                } else {
                    u32_to_hex8_le(next)
                };
                prev_s_hex
            };
            unsafe {
                let dst = out_ptr.add(r * 16);
                dst.cast::<u64>().write_unaligned(p_hex);
                dst.add(8).cast::<u64>().write_unaligned(s_hex);
            }
        }
        if !cursor.done() {
            return Err(invalid("column model trailing bytes"));
        }
        unsafe {
            scratch.u8_arena.set_len(out_u8_offset + count * 16);
        }
        Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::QuotePrefix(literal),
            const_trail: b"\"",
            dict_slices: None,
            max_emit_len: 16,
            stream: ColumnStream::Copy16 {
                u8_offset: out_u8_offset,
            },
        })
    }
}

