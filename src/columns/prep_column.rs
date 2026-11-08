#[derive(Clone, Copy)]
enum ColumnStream<'a> {
    Raw {
        cursor: Cursor<'a>,
    },
    Dictionary {
        piece_offset: usize,
        u8_offset: usize,
        max_len: usize,
    },
    DecimalAffineSmall {
        current: u64,
        step: u64,
    },
    DecimalAffine {
        current: u64,
        step: u64,
    },
    DecimalSmall {
        u16_offset: usize,
    },
    DecimalPacked {
        min: u64,
        reader: FastBitReader<'a>,
    },
    DecimalValues {
        u64_offset: usize,
    },
    HexSuffix4 {
        ptr: *const u8,
    },
    HexSuffix8 {
        ptr: *const u8,
    },
    HexSuffix {
        cursor: Cursor<'a>,
        width: usize,
    },
    TimestampMinSec {
        min_u8_offset: usize,
        sec_u8_offset: usize,
    },
    TimestampFull {
        u8_offset: usize,
        stride: usize,
    },
    Copy16 {
        u8_offset: usize,
        emit_len: u8,
        col_mask: [u64; 4],
    },
    Copy32 {
        u8_offset: usize,
        emit_len: u8,
        col_mask: [u64; 4],
    },
    SegmentDict {
        str_u8_offset: usize,
        len_u16_offset: usize,
    },
    PeriodicString {
        cursor: Cursor<'a>,
        period: usize,
    },
    PeriodicHex {
        cursor: Cursor<'a>,
        period: usize,
    },
}

#[derive(Clone, Copy)]
enum ConstValue<'a> {
    None,
    Slice(&'a [u8]),
    Inline { buf: [u8; 24], len: u8 },
    QuotedDecimal { prefix: &'a [u8], buf: [u8; 24], len: u8 },
}

const fn build_const_mask(bytes: &[u8]) -> [u64; 4] {
    let mut m = [0u64; 4];
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        m[(b >> 6) as usize] |= 1u64 << (b & 63);
        i += 1;
    }
    m
}

const HEX16_COL_MASK: [u64; 4] = build_const_mask(b"0123456789abcdef\"");

#[derive(Clone, Copy)]
enum LeadValue<'a> {
    Empty,
    QuoteOnly,
    QuotePrefix(&'a [u8]),
    Slice(&'a [u8]),
    Inline { buf: [u8; 16], len: u8 },
}

impl<'a> LeadValue<'a> {
    #[inline(always)]
    fn push_into(&self, dst: &mut Vec<u8>) {
        match *self {
            Self::Empty => {}
            Self::QuoteOnly => dst.push(b'"'),
            Self::QuotePrefix(prefix) => {
                dst.push(b'"');
                dst.extend_from_slice(prefix);
            }
            Self::Slice(s) => dst.extend_from_slice(s),
            Self::Inline { ref buf, len } => {
                dst.extend_from_slice(&buf[..len as usize]);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct PreparedColumn<'a> {
    const_full: ConstValue<'a>,
    const_lead: LeadValue<'a>,
    const_trail: &'a [u8],
    dict_slices: Option<(usize, usize)>,
    max_emit_len: usize,
    stream: ColumnStream<'a>,
}

#[inline(always)]
fn format_inline_decimal(val: u64) -> ([u8; 24], u8) {
    let mut buf = [0u8; 24];
    let len = unsafe { write_decimal_fast(buf.as_mut_ptr(), val) };
    (buf, len as u8)
}

#[inline(always)]
fn validate_raw_column(buf: &[u8], count: usize, err_msg: &'static str) -> Result<usize> {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let mut pos = 0usize;
    let mut max_len = 0usize;
    let mut rem = count;
    while rem >= 4 && pos + 516 <= len {
        unsafe {
            let b0 = *ptr.add(pos);
            if b0 >= 0x80 { break; }
            let l0 = b0 as usize;
            let p1 = pos + 1 + l0;
            let b1 = *ptr.add(p1);
            if b1 >= 0x80 { break; }
            let l1 = b1 as usize;
            let p2 = p1 + 1 + l1;
            let b2 = *ptr.add(p2);
            if b2 >= 0x80 { break; }
            let l2 = b2 as usize;
            let p3 = p2 + 1 + l2;
            let b3 = *ptr.add(p3);
            if b3 >= 0x80 { break; }
            let l3 = b3 as usize;
            pos = p3 + 1 + l3;
            max_len = max_len.max(l0.max(l1).max(l2.max(l3)));
            rem -= 4;
        }
    }
    for _ in 0..rem {
        if pos >= len {
            return Err(invalid("unexpected eof"));
        }
        let b0 = unsafe { *ptr.add(pos) };
        let rlen = if b0 < 0x80 {
            pos += 1;
            b0 as usize
        } else {
            let mut c = Cursor::new(buf);
            c.pos = pos;
            let v = c.uleb()?;
            pos = c.pos;
            v
        };
        if len - pos < rlen {
            return Err(invalid("unexpected eof"));
        }
        pos += rlen;
        if rlen > max_len {
            max_len = rlen;
        }
    }
    if pos != len {
        return Err(invalid(err_msg));
    }
    Ok(max_len)
}

impl<'a> PreparedColumn<'a> {
    fn parse_and_validate(
        bytes: &'a [u8],
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        if bytes.len() > u32::MAX as usize {
            return Err(Error::SizeOverflow);
        }
        let mut cursor = Cursor::new(bytes);
        let tag = cursor.byte()?;
        match tag {
            MODEL_RAW => {
                let body_start = cursor.pos;
                let max_len = validate_raw_column(&bytes[body_start..], count, "column model trailing bytes")?;
                let mut stream_cursor = Cursor::new(bytes);
                stream_cursor.pos = body_start;
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: None,
                    max_emit_len: max_len,
                    stream: ColumnStream::Raw {
                        cursor: stream_cursor,
                    },
                })
            }
            MODEL_LZ_RAW => {
                let mut probe = Cursor::new(cursor.bytes);
                probe.pos = cursor.pos;
                let raw_len = probe.uleb()?;
                if raw_len == 0 || raw_len > 65536 {
                    return Err(invalid("column lz raw length"));
                }
                let arena_start = scratch.raw_arena.len();
                if arena_start + raw_len > scratch.raw_arena.capacity() {
                    return Err(invalid("column lz raw arena overflow"));
                }
                unsafe {
                    scratch.raw_arena.set_len(arena_start + raw_len);
                }
                unpack_fast_lz_into(
                    &mut cursor,
                    &mut scratch.raw_arena[arena_start..arena_start + raw_len],
                )?;
                let unpacked: &'a [u8] = unsafe {
                    std::slice::from_raw_parts(
                        scratch.raw_arena.as_ptr().add(arena_start),
                        raw_len,
                    )
                };
                let max_len = validate_raw_column(unpacked, count, "column lz raw trailing bytes")?;
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: None,
                    max_emit_len: max_len,
                    stream: ColumnStream::Raw {
                        cursor: Cursor::new(unpacked),
                    },
                })
            }
            MODEL_DICTIONARY | MODEL_FRONT_DICT => {
                Self::parse_dict_or_front(tag, cursor, count, scratch)
            }
            MODEL_DECIMAL_AFFINE
            | MODEL_DECIMAL_PACKED
            | MODEL_DECIMAL_SUFFIX
            | MODEL_DECIMAL_DELTA
            | MODEL_HEX_SUFFIX => Self::parse_numeric_or_hex(tag, cursor, count, scratch),
            MODEL_TIMESTAMP | MODEL_PERIODIC_STRING | MODEL_PERIODIC_HEX => {
                Self::parse_timestamp_or_periodic(tag, bytes, cursor, count, scratch)
            }
            MODEL_EPOCH_COUNTER => Self::parse_epoch_counter(cursor, count, scratch),
            MODEL_SEGMENT_DICT => Self::parse_segment_dict(cursor, count, scratch),
            MODEL_GF2_AFFINE => Self::parse_gf2_affine(cursor, count, scratch),
            MODEL_PATTERN => Self::parse_pattern(cursor, count, scratch),
            MODEL_SCALED_DECIMAL => Self::parse_scaled_decimal(cursor, count, scratch),
            _ => Err(invalid("column model selector")),
        }
    }
}

#[inline(always)]
fn add_digit_range_to_mask(mask: &mut [u64; 4], width: usize, min: u64, max: u64) {
    let (mut lo, mut hi) = ([b'0'; PATTERN_MAX_DIGITS], [b'0'; PATTERN_MAX_DIGITS]);
    unsafe {
        write_padded_digits(lo.as_mut_ptr(), width, min);
        write_padded_digits(hi.as_mut_ptr(), width, max);
    }
    let mut p = 0usize;
    while p < width && lo[p] == hi[p] {
        mask[(lo[p] >> 6) as usize] |= 1u64 << (lo[p] & 63);
        p += 1;
    }
    if p < width {
        for b in lo[p]..=hi[p] {
            mask[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        if p + 1 < width {
            add_bytes_to_mask(mask, b"0123456789");
        }
    }
}

/// Extract the exact 256-bit byte alphabet of a `MODEL_PATTERN` column from its token header alone.
fn pattern_col_alphabet(c: &mut Cursor<'_>, mask: &mut [u64; 4]) -> Result<()> {
    let ntok = c.uleb()?;
    if !(1..=PATTERN_MAX_TOKENS).contains(&ntok) {
        return Err(invalid("column pattern tokens"));
    }
    for _ in 0..ntok {
        match c.byte()? {
            0 => {
                let width = c.byte()? as usize;
                let (min, radix) = (c.uleb()? as u64, c.uleb()? as u64);
                if !(1..=PATTERN_MAX_DIGITS).contains(&width) || radix == 0 {
                    return Err(invalid("column pattern digits"));
                }
                let max = min.checked_add(radix - 1).ok_or_else(|| invalid("column pattern range"))?;
                if max >= 10u64.pow(width as u32) {
                    return Err(invalid("column pattern width"));
                }
                add_digit_range_to_mask(mask, width, min, max);
            }
            1 => {
                let len = c.uleb()?;
                add_bytes_to_mask(mask, c.take(len)?);
            }
            2 => {
                let entries = c.uleb()?;
                if !(2..=PATTERN_MAX_DICT).contains(&entries) {
                    return Err(invalid("column pattern dictionary"));
                }
                for _ in 0..entries {
                    let len = c.uleb()?;
                    add_bytes_to_mask(mask, c.take(len)?);
                }
            }
            _ => return Err(invalid("column pattern token kind")),
        }
    }
    Ok(())
}

/// Flat column storage: all value bytes live in one buffer plus one range per
/// value. Used by the strict oracle decoder.
#[derive(Default)]
struct FlatValues {
    bytes: Vec<u8>,
    ranges: Vec<(u32, u32)>,
}

impl FlatValues {
    fn with_capacity(values: usize, bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes.min(1 << 20)),
            ranges: Vec::with_capacity(values),
        }
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    fn get(&self, index: usize) -> Option<&[u8]> {
        let &(offset, len) = self.ranges.get(index)?;
        let start = offset as usize;
        self.bytes.get(start..start.checked_add(len as usize)?)
    }

    #[inline]
    fn begin(&self) -> u32 {
        self.bytes.len() as u32
    }

    #[inline]
    fn finish(&mut self, start: u32) {
        let len = (self.bytes.len() as u32) - start;
        self.ranges.push((start, len));
    }

    #[inline]
    fn push_slice(&mut self, value: &[u8]) {
        let start = self.begin();
        copy_bytes(&mut self.bytes, value);
        self.finish(start);
    }

    fn materialize(&self) -> Vec<Vec<u8>> {
        (0..self.len())
            .map(|index| self.get(index).unwrap_or_default().to_vec())
            .collect()
    }
}

#[inline(always)]
fn write_raw_values_into(buf: &mut Vec<u8>, values: &[&[u8]], total_size: usize) {
    buf.clear();
    buf.reserve(total_size + 32);
    unsafe {
        let mut p = buf.as_mut_ptr();
        *p = MODEL_RAW;
        p = p.add(1);
        for &v in values {
            let len = v.len();
            if len < 0x80 {
                *p = len as u8;
                p = p.add(1);
            } else {
                let mut rem = len;
                loop {
                    let mut b = (rem & 0x7f) as u8;
                    rem >>= 7;
                    if rem != 0 {
                        b |= 0x80;
                    }
                    *p = b;
                    p = p.add(1);
                    if rem == 0 {
                        break;
                    }
                }
            }
            if len <= 16 {
                copy_short_exact(v.as_ptr(), p, len);
            } else if len <= 32 {
                copy16_unaligned(v.as_ptr(), p);
                copy16_unaligned(v.as_ptr().add(len - 16), p.add(len - 16));
            } else {
                std::ptr::copy_nonoverlapping(v.as_ptr(), p, len);
            }
            p = p.add(len);
        }
        buf.set_len(total_size);
    }
}

fn pack_raw_or_lz_model(values: &[&[u8]]) -> Vec<u8> {
    let total_size = raw_model_size(values);
    if !(64..=65536).contains(&(total_size - 1)) || total_size < 128 {
        return pack_raw_model(values);
    }
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let ModelScratch {
            raw_stream, raw_lz, ..
        } = &mut *sc;
        write_raw_values_into(raw_stream, values, total_size);
        raw_lz.clear();
        raw_lz.reserve(total_size / 2 + 64);
        if pack_fast_lz_into(&raw_stream[1..], raw_lz).is_some() && raw_lz.len() + 8 < total_size {
            let mut out = take_model_buf(raw_lz.len());
            out.extend_from_slice(raw_lz);
            out[0] = MODEL_LZ_RAW;
            out
        } else {
            let mut out = take_model_buf(total_size);
            out.extend_from_slice(raw_stream);
            out
        }
    })
}

fn pack_large_dict_fallback(
    values: &[&[u8]],
    dict_bytes_sum: usize,
    limit: usize,
) -> Option<Vec<u8>> {
    let n = values.len();
    if n <= 256 {
        return None;
    }
    let min_packed_len = (n * 9 + 7) >> 3;
    if 3 + dict_bytes_sum + min_packed_len >= limit {
        return None;
    }
    let cap = (n * 2).next_power_of_two();
    let mask = cap - 1;
    let hshift = 64 - cap.trailing_zeros();
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let ModelScratch {
            dict_slots,
            dict_uniq,
            dict_row_ids,
            dict_order,
            dict_rank,
            ..
        } = &mut *sc;
        dict_slots.clear();
        dict_slots.resize(cap, u32::MAX);
        dict_uniq.clear();
        dict_uniq.reserve(n.min(1024));
        dict_row_ids.clear();
        dict_row_ids.reserve(n);
        let mut plain_dict_bytes = 0usize;
        for &val in values {
            let mut pos = (hash_slice(val) >> hshift) as usize;
            let id = loop {
                let s = unsafe { *dict_slots.get_unchecked(pos) };
                if s == u32::MAX {
                    let id = dict_uniq.len() as u32;
                    unsafe {
                        *dict_slots.get_unchecked_mut(pos) = id;
                    }
                    dict_uniq.push((val.as_ptr(), val.len()));
                    plain_dict_bytes += uleb_len(val.len()) + val.len();
                    if 3 + plain_dict_bytes + min_packed_len >= limit {
                        return None;
                    }
                    break id;
                }
                let (up, ul) = unsafe { *dict_uniq.get_unchecked(s as usize) };
                if slice_eq_fast(unsafe { std::slice::from_raw_parts(up, ul) }, val) {
                    break s;
                }
                pos = (pos + 1) & mask;
            };
            dict_row_ids.push(id);
        }
        let u = dict_uniq.len();
        if u == n {
            return None;
        }
        let width = bit_width((u - 1) as u64);
        let packed_len = bit_bytes(n, width).ok()?;
        let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
        if size >= limit {
            return None;
        }
        dict_order.clear();
        dict_order.extend(0..u as u32);
        dict_order.sort_unstable_by(|&a, &b| unsafe {
            let (pa, la) = *dict_uniq.get_unchecked(a as usize);
            let (pb, lb) = *dict_uniq.get_unchecked(b as usize);
            std::slice::from_raw_parts(pa, la).cmp(std::slice::from_raw_parts(pb, lb))
        });
        dict_rank.clear();
        dict_rank.resize(u, 0);
        for (r, &id) in dict_order.iter().enumerate() {
            unsafe {
                *dict_rank.get_unchecked_mut(id as usize) = r as u32;
            }
        }
        let mut out = take_model_buf(size);
        out.push(MODEL_DICTIONARY);
        put_uleb(&mut out, u);
        for &id in &*dict_order {
            let (p, l) = unsafe { *dict_uniq.get_unchecked(id as usize) };
            put_bytes(&mut out, unsafe { std::slice::from_raw_parts(p, l) });
        }
        out.push(width);
        push_bit_values(
            &mut out,
            n,
            width,
            dict_row_ids
                .iter()
                .map(|&id| unsafe { *dict_rank.get_unchecked(id as usize) as u64 }),
        );
        Some(out)
    })
}

