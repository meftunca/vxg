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

#[derive(Clone, Copy)]
enum LeadValue<'a> {
    Empty,
    QuoteOnly,
    QuotePrefix(&'a [u8]),
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
    const_trail: &'static [u8],
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
                let mut max_len = 0usize;
                for _ in 0..count {
                    let len = cursor.uleb()?;
                    let _ = cursor.take(len)?;
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
                let mut sub = Cursor::new(unpacked);
                let mut max_len = 0usize;
                for _ in 0..count {
                    let len = sub.uleb()?;
                    let _ = sub.take(len)?;
                    if len > max_len {
                        max_len = len;
                    }
                }
                if !sub.done() {
                    return Err(invalid("column lz raw trailing bytes"));
                }
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
            _ => Err(invalid("column model selector")),
        }
    }
}

