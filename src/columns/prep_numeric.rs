#[inline(always)]
fn unpack_u16_plus_min_to_u64(
    packed_bytes: &[u8],
    count: usize,
    width: u8,
    min: u64,
    scratch: &mut DecodeScratch,
) -> Result<usize> {
    let u16_off = unpack_u16_swar(packed_bytes, count, width, 0, &mut scratch.u16_arena)?;
    let u64_off = scratch.u64_arena.len();
    scratch.u64_arena.reserve(count + 4);
    unsafe {
        let src = scratch.u16_arena.as_ptr().add(u16_off);
        let dst = scratch.u64_arena.as_mut_ptr().add(u64_off);
        let mut i = 0usize;
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            let vmin = vdupq_n_u64(min);
            while i + 4 <= count {
                let v16 = vld1_u16(src.add(i));
                let v32 = vmovl_u16(v16);
                vst1q_u64(dst.add(i), vaddq_u64(vmin, vmovl_u32(vget_low_u32(v32))));
                vst1q_u64(dst.add(i + 2), vaddq_u64(vmin, vmovl_high_u32(v32)));
                i += 4;
            }
        }
        while i < count {
            *dst.add(i) = min + (*src.add(i) as u64);
            i += 1;
        }
        scratch.u64_arena.set_len(u64_off + count);
    }
    scratch.u16_arena.truncate(u16_off);
    Ok(u64_off)
}

impl<'a> PreparedColumn<'a> {
    fn parse_numeric_or_hex(
        tag: u8,
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        match tag {
            MODEL_DECIMAL_AFFINE => {
                if count < 2 {
                    return Err(invalid("column affine count"));
                }
                let first = cursor.u64()?;
                let step_i64 = cursor.i64()?;
                let step = step_i64 as i128;
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let last = first as i128 + step * (count as i128 - 1);
                let last_u64 = u64::try_from(last).map_err(|_| invalid("column affine range"))?;
                if step_i64 == 0 {
                    let (buf, len) = format_inline_decimal(first);
                    return Ok(Self {
                        const_full: ConstValue::Inline { buf, len },
                        const_lead: LeadValue::Empty,
                        const_trail: b"",
                        dict_slices: None,
                        max_emit_len: 0,
                        stream: ColumnStream::Raw {
                            cursor: Cursor::new(&[]),
                        },
                    });
                }
                let stream = if first < 10_000 && last_u64 < 10_000 {
                    ColumnStream::DecimalAffineSmall {
                        current: first,
                        step: step_i64 as u64,
                    }
                } else {
                    ColumnStream::DecimalAffine {
                        current: first,
                        step: step_i64 as u64,
                    }
                };
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: None,
                    max_emit_len: 20,
                    stream,
                })
            }
            MODEL_DECIMAL_PACKED => {
                if count < 2 {
                    return Err(invalid("column packed decimal count"));
                }
                let min = cursor.u64()?;
                let width = cursor.byte()?;
                if width > 64 {
                    return Err(invalid("column packed decimal width"));
                }
                let packed_len = bit_bytes(count, width)?;
                let packed_bytes = cursor.take(packed_len)?;
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                if width == 0 {
                    let _ = FastBitReader::new_validated(packed_bytes, count, width)?;
                    let (buf, len) = format_inline_decimal(min);
                    return Ok(Self {
                        const_full: ConstValue::Inline { buf, len },
                        const_lead: LeadValue::Empty,
                        const_trail: b"",
                        dict_slices: None,
                        max_emit_len: 0,
                        stream: ColumnStream::Raw {
                            cursor: Cursor::new(&[]),
                        },
                    });
                }
                if width <= 14 && min.saturating_add((1u64 << width) - 1) < 10_000 {
                    let u16_offset = unpack_u16_swar(
                        packed_bytes,
                        count,
                        width,
                        min as u16,
                        &mut scratch.u16_arena,
                    )?;
                    return Ok(Self {
                        const_full: ConstValue::None,
                        const_lead: LeadValue::Empty,
                        const_trail: b"",
                        dict_slices: None,
                        max_emit_len: 4,
                        stream: ColumnStream::DecimalSmall { u16_offset },
                    });
                }
                if width <= 16 && min.checked_add((1u64 << width) - 1).is_some() {
                    let u64_offset =
                        unpack_u16_plus_min_to_u64(packed_bytes, count, width, min, scratch)?;
                    return Ok(Self {
                        const_full: ConstValue::None,
                        const_lead: LeadValue::Empty,
                        const_trail: b"",
                        dict_slices: None,
                        max_emit_len: 20,
                        stream: ColumnStream::DecimalValues { u64_offset },
                    });
                }
                let reader = FastBitReader::new_validated(packed_bytes, count, width)?;
                let max_possible_res = if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };
                if min.checked_add(max_possible_res).is_none() {
                    let mut check = reader;
                    for _ in 0..count {
                        if min.checked_add(check.next_u64()).is_none() {
                            return Err(Error::SizeOverflow);
                        }
                    }
                }
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: None,
                    max_emit_len: 20,
                    stream: ColumnStream::DecimalPacked { min, reader },
                })
            }
            MODEL_DECIMAL_SUFFIX => {
                if count < 2 {
                    return Err(invalid("column decimal suffix count"));
                }
                let prefix_len = cursor.uleb()?;
                let prefix = cursor.take(prefix_len)?;
                let min = cursor.u64()?;
                let width = cursor.byte()?;
                if width > 64 {
                    return Err(invalid("column decimal suffix width"));
                }
                let packed_len = bit_bytes(count, width)?;
                let packed_bytes = cursor.take(packed_len)?;
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                if width == 0 {
                    let _ = FastBitReader::new_validated(packed_bytes, count, width)?;
                    let (buf, len) = format_inline_decimal(min);
                    return Ok(Self {
                        const_full: ConstValue::QuotedDecimal { prefix, buf, len },
                        const_lead: LeadValue::Empty,
                        const_trail: b"",
                        dict_slices: None,
                        max_emit_len: 0,
                        stream: ColumnStream::Raw {
                            cursor: Cursor::new(&[]),
                        },
                    });
                }
                if width <= 14 && min.saturating_add((1u64 << width) - 1) < 10_000 {
                    let u16_offset = unpack_u16_swar(
                        packed_bytes,
                        count,
                        width,
                        min as u16,
                        &mut scratch.u16_arena,
                    )?;
                    return Ok(Self {
                        const_full: ConstValue::None,
                        const_lead: LeadValue::QuotePrefix(prefix),
                        const_trail: b"\"",
                        dict_slices: None,
                        max_emit_len: 4,
                        stream: ColumnStream::DecimalSmall { u16_offset },
                    });
                }
                if width <= 16 && min.checked_add((1u64 << width) - 1).is_some() {
                    let u64_offset =
                        unpack_u16_plus_min_to_u64(packed_bytes, count, width, min, scratch)?;
                    return Ok(Self {
                        const_full: ConstValue::None,
                        const_lead: LeadValue::QuotePrefix(prefix),
                        const_trail: b"\"",
                        dict_slices: None,
                        max_emit_len: 20,
                        stream: ColumnStream::DecimalValues { u64_offset },
                    });
                }
                let reader = FastBitReader::new_validated(packed_bytes, count, width)?;
                let max_possible_res = if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };
                if min.checked_add(max_possible_res).is_none() {
                    let mut check = reader;
                    for _ in 0..count {
                        if min.checked_add(check.next_u64()).is_none() {
                            return Err(Error::SizeOverflow);
                        }
                    }
                }
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::QuotePrefix(prefix),
                    const_trail: b"\"",
                    dict_slices: None,
                    max_emit_len: 20,
                    stream: ColumnStream::DecimalPacked { min, reader },
                })
            }
            MODEL_DECIMAL_DELTA => {
                if count < 2 {
                    return Err(invalid("column decimal delta count"));
                }
                let first = cursor.u64()?;
                let width = cursor.byte()?;
                if width > 64 {
                    return Err(invalid("column decimal delta width"));
                }
                let packed_len = bit_bytes(count - 1, width)?;
                let packed_bytes = cursor.take(packed_len)?;
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                if width == 0 {
                    let _ = FastBitReader::new_validated(packed_bytes, count - 1, width)?;
                    let (buf, len) = format_inline_decimal(first);
                    return Ok(Self {
                        const_full: ConstValue::Inline { buf, len },
                        const_lead: LeadValue::Empty,
                        const_trail: b"",
                        dict_slices: None,
                        max_emit_len: 0,
                        stream: ColumnStream::Raw {
                            cursor: Cursor::new(&[]),
                        },
                    });
                }
                if width <= 7 {
                    let zz_offset = unpack_and_validate_u8(
                        packed_bytes,
                        count - 1,
                        width,
                        0,
                        256,
                        &mut scratch.u8_arena,
                    )?;
                    let u16_offset = scratch.u16_arena.len();
                    scratch.u16_arena.reserve(count);
                    let mut prev = first as i64;
                    let mut or_all = first;
                    let zz_ptr = unsafe { scratch.u8_arena.as_ptr().add(zz_offset) };
                    let u16_dst = unsafe { scratch.u16_arena.as_mut_ptr().add(u16_offset) };
                    unsafe {
                        *u16_dst = first as u16;
                    }
                    let mut overflow_10k = first >= 10_000;
                    if !overflow_10k {
                        for i in 1..count {
                            let zz = unsafe { *zz_ptr.add(i - 1) } as i64;
                            let delta = (zz >> 1) ^ -(zz & 1);
                            let cur = prev + delta;
                            if (cur as u64) >= 10_000 {
                                overflow_10k = true;
                                break;
                            }
                            or_all |= cur as u64;
                            unsafe {
                                *u16_dst.add(i) = cur as u16;
                            }
                            prev = cur;
                        }
                    }
                    if !overflow_10k {
                        let _ = or_all;
                        scratch.u8_arena.truncate(zz_offset);
                        unsafe {
                            scratch.u16_arena.set_len(u16_offset + count);
                        }
                        return Ok(Self {
                            const_full: ConstValue::None,
                            const_lead: LeadValue::Empty,
                            const_trail: b"",
                            dict_slices: None,
                            max_emit_len: 4,
                            stream: ColumnStream::DecimalSmall { u16_offset },
                        });
                    }
                    scratch.u8_arena.truncate(zz_offset);
                }
                let mut reader = FastBitReader::new_validated(packed_bytes, count - 1, width)?;
                let u64_offset = scratch.u64_arena.len();
                scratch.u64_arena.reserve(count);
                scratch.u64_arena.push(first);
                let mut prev = first as i128;
                let mut or_all = first;
                for _ in 1..count {
                    let zz = reader.next_u64();
                    let delta = ((zz >> 1) as i64) ^ -((zz & 1) as i64);
                    let cur = prev + i128::from(delta);
                    if cur < 0 || cur > i128::from(u64::MAX) {
                        return Err(invalid("column decimal delta range"));
                    }
                    let u = cur as u64;
                    or_all |= u;
                    scratch.u64_arena.push(u);
                    prev = cur;
                }
                let stream = if or_all < 10_000 {
                    let u16_offset = scratch.u16_arena.len();
                    scratch.u16_arena.reserve(count);
                    for &v in &scratch.u64_arena[u64_offset..u64_offset + count] {
                        scratch.u16_arena.push(v as u16);
                    }
                    scratch.u64_arena.truncate(u64_offset);
                    ColumnStream::DecimalSmall { u16_offset }
                } else {
                    ColumnStream::DecimalValues { u64_offset }
                };
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: None,
                    max_emit_len: 20,
                    stream,
                })
            }
            MODEL_HEX_SUFFIX => {
                if count < 2 {
                    return Err(invalid("column hex count"));
                }
                let prefix_len = cursor.uleb()?;
                let prefix = cursor.take(prefix_len)?;
                let width_hdr = cursor.uleb()?;
                let (quoted, width) = if (1..=64).contains(&width_hdr) {
                    (true, width_hdr)
                } else if (68..=128).contains(&width_hdr) {
                    (false, width_hdr - 64)
                } else {
                    return Err(invalid("column hex width"));
                };
                let raw_total = count.checked_mul(width).ok_or(Error::SizeOverflow)?;
                let raw_slice = cursor.take(raw_total)?;
                if !cursor.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let stream = if width == 8 {
                    ColumnStream::HexSuffix8 {
                        ptr: raw_slice.as_ptr(),
                    }
                } else if width == 4 {
                    ColumnStream::HexSuffix4 {
                        ptr: raw_slice.as_ptr(),
                    }
                } else {
                    ColumnStream::HexSuffix {
                        cursor: Cursor::new(raw_slice),
                        width,
                    }
                };
                let (const_lead, const_trail): (LeadValue<'a>, &'a [u8]) = if quoted {
                    (LeadValue::QuotePrefix(prefix), b"\"")
                } else if prefix.is_empty() {
                    (LeadValue::Empty, b"")
                } else {
                    (LeadValue::Slice(prefix), b"")
                };
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead,
                    const_trail,
                    dict_slices: None,
                    max_emit_len: width * 2,
                    stream,
                })
            }
            _ => unreachable!(),
        }
    }
}

