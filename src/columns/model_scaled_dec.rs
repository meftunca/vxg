const MODEL_SCALED_DECIMAL: u8 = 16;

const POW10_U64_SMALL: [u64; 10] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
];

/// Bijective parser for `MODEL_SCALED_DECIMAL`: accepts canonical signed integers (`-123`)
/// and fixed-scale decimals (`19.99`, `-0.05`) with `scale` in `0..=9` that round-trip
/// byte-for-byte through [`write_scaled_i64_fast`]. Rejects `+`, `-0`, `-0.00`, and leading zeros.
#[inline(always)]
fn parse_exact_scaled_i64(b: &[u8]) -> Option<(i64, u8)> {
    let len = b.len();
    if len == 0 || len > 21 {
        return None;
    }
    let (neg, rest) = if b[0] == b'-' {
        (true, &b[1..])
    } else {
        (false, b)
    };
    if rest.is_empty() {
        return None;
    }
    let mut dot = usize::MAX;
    for (i, &c) in rest.iter().enumerate() {
        if c == b'.' {
            if dot != usize::MAX {
                return None;
            }
            dot = i;
        } else if !c.is_ascii_digit() {
            return None;
        }
    }
    let (int_b, frac_b) = if dot == usize::MAX {
        (rest, &[][..])
    } else {
        if dot == 0 || dot + 1 == rest.len() {
            return None;
        }
        (&rest[..dot], &rest[dot + 1..])
    };
    if int_b.len() > 1 && int_b[0] == b'0' {
        return None;
    }
    let scale = frac_b.len();
    if scale > 9 || int_b.len() + scale > 19 {
        return None;
    }
    let mut abs_u = 0u64;
    for &d in int_b {
        abs_u = abs_u.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    for &d in frac_b {
        abs_u = abs_u.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    if neg && abs_u == 0 {
        // `-0` / `-0.00` would re-format without `-`; leave to dictionary/raw for byte exactness.
        return None;
    }
    let val = if neg {
        if abs_u == (i64::MAX as u64) + 1 {
            i64::MIN
        } else if abs_u <= i64::MAX as u64 {
            -(abs_u as i64)
        } else {
            return None;
        }
    } else if abs_u <= i64::MAX as u64 {
        abs_u as i64
    } else {
        return None;
    };
    Some((val, scale as u8))
}

#[inline(always)]
unsafe fn write_scaled_i64_fast(mut dst: *mut u8, val: i64, scale: u8) -> usize {
    let start = dst;
    unsafe {
        let abs_u = if val < 0 {
            *dst = b'-';
            dst = dst.add(1);
            val.unsigned_abs()
        } else {
            val as u64
        };
        if scale == 0 {
            if abs_u < 10_000 {
                let entry = *DEC_10K.get_unchecked(abs_u as usize);
                dst.cast::<u32>().write_unaligned(entry as u32);
                dst = dst.add((entry >> 32) as usize);
            } else {
                dst = dst.add(write_decimal_fast(dst, abs_u));
            }
        } else {
            let p10 = *POW10_U64_SMALL.get_unchecked(scale as usize);
            let int_part = abs_u / p10;
            let frac_part = abs_u % p10;
            if int_part < 10_000 {
                let entry = *DEC_10K.get_unchecked(int_part as usize);
                dst.cast::<u32>().write_unaligned(entry as u32);
                dst = dst.add((entry >> 32) as usize);
            } else {
                dst = dst.add(write_decimal_fast(dst, int_part));
            }
            *dst = b'.';
            dst = dst.add(1);
            if scale == 2 {
                let pair = TWO_DIGITS
                    .as_ptr()
                    .add((frac_part as usize) * 2)
                    .cast::<u16>()
                    .read_unaligned();
                dst.cast::<u16>().write_unaligned(pair);
                dst = dst.add(2);
            } else {
                write_padded_digits(dst, scale as usize, frac_part);
                dst = dst.add(scale as usize);
            }
        }
        dst.offset_from(start) as usize
    }
}

fn pack_scaled_decimal_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || limit <= 11 {
        return None;
    }
    let (v0, scale) = parse_exact_scaled_i64(values[0])?;
    let (v1, s1) = parse_exact_scaled_i64(values[1])?;
    if s1 != scale {
        return None;
    }
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.parsed_u64.clear();
        sc.parsed_u64.reserve(n);
        let ptr = sc.parsed_u64.as_mut_ptr();
        unsafe {
            *ptr = v0 as u64;
            *ptr.add(1) = v1 as u64;
        }
        let mut min_v = v0.min(v1);
        let mut max_v = v0.max(v1);
        for (i, &s) in values.iter().enumerate().skip(2) {
            let (v, sc_i) = parse_exact_scaled_i64(s)?;
            if sc_i != scale {
                return None;
            }
            unsafe {
                *ptr.add(i) = v as u64;
            }
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        if scale == 0 && min_v >= 0 {
            return None;
        }
        unsafe {
            sc.parsed_u64.set_len(n);
        }
        let span128 = (max_v as i128) - (min_v as i128);
        let span = u64::try_from(span128).ok()?;
        let width = bit_width(span);
        let packed_len = bit_bytes(n, width).ok()?;
        let size = 11 + packed_len;
        if size >= limit {
            return None;
        }
        let mut out = take_model_buf(size);
        out.push(MODEL_SCALED_DECIMAL);
        out.push(scale);
        out.extend_from_slice(&min_v.to_le_bytes());
        out.push(width);
        let min_i128 = min_v as i128;
        push_bit_values(
            &mut out,
            n,
            width,
            sc.parsed_u64
                .iter()
                .map(|&raw| ((raw as i64 as i128) - min_i128) as u64),
        );
        Some(out)
    })
}

fn unpack_scaled_decimal_oracle(
    cursor: &mut Cursor<'_>,
    count: usize,
    values: &mut FlatValues,
) -> Result<()> {
    if count < 2 {
        return Err(invalid("column scaled decimal count"));
    }
    let scale = cursor.byte()?;
    if scale > 9 {
        return Err(invalid("column scaled decimal scale"));
    }
    let min_v = cursor.i64()?;
    if scale == 0 && min_v >= 0 {
        return Err(invalid("column scaled decimal noncanonical unsigned"));
    }
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column scaled decimal width"));
    }
    let packed_len = bit_bytes(count, width)?;
    let mut reader = BitReader::new(cursor.take(packed_len)?);
    let mut buf = [0u8; 24];
    for _ in 0..count {
        let res = reader.read(width)?;
        let val128 = (min_v as i128) + (res as i128);
        let val = i64::try_from(val128).map_err(|_| invalid("column scaled decimal overflow"))?;
        let n = unsafe { write_scaled_i64_fast(buf.as_mut_ptr(), val, scale) };
        values.push_slice(&buf[..n]);
    }
    reader.finish_zero_padded()?;
    Ok(())
}

impl<'a> PreparedColumn<'a> {
    #[inline(never)]
    fn parse_scaled_decimal(
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        if count < 2 {
            return Err(invalid("column scaled decimal count"));
        }
        let scale = cursor.byte()?;
        if scale > 9 {
            return Err(invalid("column scaled decimal scale"));
        }
        let min_v = cursor.i64()?;
        if scale == 0 && min_v >= 0 {
            return Err(invalid("column scaled decimal noncanonical unsigned"));
        }
        let width = cursor.byte()?;
        if width > 64 {
            return Err(invalid("column scaled decimal width"));
        }
        let packed_len = bit_bytes(count, width)?;
        let slice = cursor.take(packed_len)?;
        if !cursor.done() {
            return Err(invalid("column model trailing bytes"));
        }
        let max_r = if width == 0 {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
            return Err(invalid("column scaled decimal overflow"));
        }
        if width == 0 {
            let reader = BitReader::new(slice);
            reader.finish_zero_padded()?;
            let mut buf = [0u8; 24];
            let len = unsafe { write_scaled_i64_fast(buf.as_mut_ptr(), min_v, scale) } as u8;
            return Ok(Self {
                const_full: ConstValue::Inline { buf, len },
                const_lead: LeadValue::Empty,
                const_trail: b"",
                dict_slices: None,
                max_emit_len: 0,
                stream: ColumnStream::DecimalPacked {
                    min: 0,
                    reader: FastBitReader::new_validated(&[], 0, 0)?,
                },
            });
        }

        let mut reader = FastBitReader::new_validated(slice, count, width)?;
        let str_u8_offset = scratch.u8_arena.len();
        scratch.u8_arena.reserve(count * 24 + 48);
        let len_u16_offset = scratch.u16_arena.len();
        scratch.u16_arena.reserve(count);
        let str_base = unsafe { scratch.u8_arena.as_mut_ptr().add(str_u8_offset) };
        let len_base = unsafe { scratch.u16_arena.as_mut_ptr().add(len_u16_offset) };
        let mut dst = str_base;
        let mut max_val_len = 0usize;
        let min_i128 = min_v as i128;
        for i in 0..count {
            let r = reader.next_u64();
            let val = (min_i128 + (r as i128)) as i64;
            let written = unsafe {
                let w = write_scaled_i64_fast(dst, val, scale);
                *len_base.add(i) = w as u16;
                dst = dst.add(w);
                w
            };
            if written > max_val_len {
                max_val_len = written;
            }
        }
        let actual_bytes = unsafe { dst.offset_from(str_base) as usize };
        unsafe {
            std::ptr::write_bytes(dst, 0, 32);
            scratch.u8_arena.set_len(str_u8_offset + actual_bytes);
            scratch.u16_arena.set_len(len_u16_offset + count);
        }
        Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::Empty,
            const_trail: b"",
            dict_slices: None,
            max_emit_len: max_val_len,
            stream: ColumnStream::SegmentDict {
                str_u8_offset,
                len_u16_offset,
            },
        })
    }
}
