//! P5D exact, block-local structured-column research packets.
//!
//! This module starts from raw bytes rather than a v2 command trace, serializes
//! every template, row selector, model and exception inside the same independent
//! block, then decodes the packet back to the exact bytes. It also provides the
//! v3 structured-column block payload (`encode_grammar_block`/`decode_block`);
//! its scalar rates are research numbers, never codec throughput claims.

use std::{fmt::Write, hint::black_box, time::Instant};

use crate::{Error, Result};

const RAW_PACKET: u8 = 0;
const GRAMMAR_PACKET: u8 = 1;
const STRIDE_PACKET: u8 = 2;
const FAST_LZ_PACKET: u8 = 3;
const MODEL_RAW: u8 = 0;
const MODEL_DICTIONARY: u8 = 1;
const MODEL_DECIMAL_AFFINE: u8 = 2;
const MODEL_HEX_SUFFIX: u8 = 3;
const MODEL_TIMESTAMP: u8 = 4;
const MODEL_DECIMAL_PACKED: u8 = 5;
const MODEL_DECIMAL_SUFFIX: u8 = 6;
const MODEL_PERIODIC_STRING: u8 = 7;
const MODEL_PERIODIC_HEX: u8 = 8;
const MODEL_GF2_AFFINE: u8 = 9;
const MODEL_EPOCH_COUNTER: u8 = 10;
const MODEL_SEGMENT_DICT: u8 = 11;
const MODEL_DECIMAL_DELTA: u8 = 12;
const MODEL_FRONT_DICT: u8 = 13;
const MODEL_LZ_RAW: u8 = 14;
const MAX_BLOCK_SIZE: usize = 1_048_576;
const MAX_TEMPLATES: usize = 4096;
const MAX_SLOTS: usize = 128;
const MAX_EPOCH_PREFIXES: usize = 64;
const MAX_SEGMENT_LEN: usize = 64;
const MAX_SEGMENT_DICT: usize = 64;
const MAX_SEGMENTS_PER_VALUE: usize = 1024;
const MAX_PAD: usize = 4096;
const MAX_DECIMAL_SEGMENT_DIGITS: usize = 9;
const HEX: &[u8; 16] = b"0123456789abcdef";

fn invalid(message: &'static str) -> Error {
    Error::InvalidFrame(message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelKind {
    Raw,
    Dictionary,
    DecimalAffine,
    HexSuffix,
    Timestamp,
    DecimalPacked,
    DecimalSuffix,
    PeriodicString,
    PeriodicHex,
    Gf2Affine,
    EpochCounter,
    SegmentDict,
    DecimalDelta,
}

impl ModelKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw-length-prefixed",
            Self::Dictionary => "canonical-byte-dictionary",
            Self::DecimalAffine => "canonical-decimal-affine",
            Self::HexSuffix => "quoted-fixed-width-lower-hex-suffix",
            Self::Timestamp => "quoted-canonical-iso8601-same-year",
            Self::DecimalPacked => "canonical-decimal-min-width-bitpack",
            Self::DecimalSuffix => "quoted-canonical-decimal-suffix-bitpack",
            Self::PeriodicString => "quoted-plain-byte-periodic-1-to-64",
            Self::PeriodicHex => "quoted-lower-hex-periodic-1-to-64",
            Self::Gf2Affine => "quoted-fixed-width-hex-gf2-affine-recurrence",
            Self::EpochCounter => "quoted-fixed-width-hex-epoch-counter",
            Self::SegmentDict => "quoted-semicolon-segment-dictionary",
            Self::DecimalDelta => "canonical-decimal-delta-bitpack",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Raw => 0,
            Self::Dictionary => 1,
            Self::DecimalAffine => 2,
            Self::HexSuffix => 3,
            Self::Timestamp => 4,
            Self::DecimalPacked => 5,
            Self::DecimalSuffix => 6,
            Self::PeriodicString => 7,
            Self::PeriodicHex => 8,
            Self::Gf2Affine => 9,
            Self::EpochCounter => 10,
            Self::SegmentDict => 11,
            Self::DecimalDelta => 12,
        }
    }
}

#[derive(Clone, Debug)]
struct Template<'a> {
    segments: Vec<&'a [u8]>,
    values: Vec<Vec<&'a [u8]>>,
}

#[derive(Clone, Copy, Debug)]
enum Row<'a> {
    Template(usize),
    Raw(&'a [u8]),
}

#[derive(Clone, Copy, Debug)]
struct ColumnStat {
    template: usize,
    slot: usize,
    kind: ModelKind,
    values: usize,
    payload_bytes: usize,
}

#[derive(Clone, Debug, Default)]
struct BlockStats {
    grammar: bool,
    templates: usize,
    template_rows: usize,
    raw_rows: usize,
    columns: Vec<ColumnStat>,
}

#[derive(Clone, Debug)]
struct PackedBlock {
    bytes: Vec<u8>,
    stats: BlockStats,
}

#[derive(Clone, Debug)]
struct PackedGrammar {
    bytes: Vec<u8>,
    stats: BlockStats,
}

/// A column-level decomposition. `template` and `slot` are local block shape
/// positions, never input field names or source-specific selectors.
#[derive(Clone, Debug)]
pub struct ColumnReport {
    pub block: usize,
    pub template: usize,
    pub slot: usize,
    pub model: &'static str,
    pub values: usize,
    pub payload_bytes: usize,
}

/// Complete P5D packet-size accounting. The 32-byte header plus a 32-byte
/// independent-block directory entry is charged for every block, matching the
/// target framing policy without claiming that this packet is a v3 frame.
#[derive(Debug)]
pub struct Evaluation {
    input_bytes: usize,
    block_size: usize,
    blocks: usize,
    grammar_blocks: usize,
    raw_blocks: usize,
    payload_bytes: usize,
    frame_directory_bytes: usize,
    template_count: usize,
    template_rows: usize,
    raw_rows: usize,
    model_counts: [usize; 13],
    model_bytes: [usize; 13],
    columns: Vec<ColumnReport>,
    pack_ns: Option<u128>,
    unpack_ns: Option<u128>,
}

#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pack_ns: u128,
    unpack_ns: u128,
}

#[derive(Clone, Copy)]
struct Timestamp {
    year: u16,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
}

#[derive(Clone, Copy)]
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline(always)]
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline(always)]
    fn byte(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.pos).ok_or(Error::Truncated)?;
        self.pos += 1;
        Ok(byte)
    }

    #[inline(always)]
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or(Error::SizeOverflow)?;
        let value = self.bytes.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(value)
    }

    #[inline(always)]
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().map_err(|_| Error::Truncated)?,
        ))
    }

    #[inline(always)]
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| Error::Truncated)?,
        ))
    }

    #[inline(always)]
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| Error::Truncated)?,
        ))
    }

    #[inline(always)]
    fn uleb(&mut self) -> Result<usize> {
        if let Some(&b0) = self.bytes.get(self.pos) {
            if b0 < 0x80 {
                self.pos += 1;
                return Ok(b0 as usize);
            }
            if let Some(&b1) = self.bytes.get(self.pos + 1) {
                if b1 < 0x80 && b1 != 0 {
                    self.pos += 2;
                    return Ok(((b0 & 0x7f) as usize) | ((b1 as usize) << 7));
                }
            }
        }
        self.uleb_slow()
    }

    #[cold]
    fn uleb_slow(&mut self) -> Result<usize> {
        let start = self.pos;
        let mut value = 0usize;
        for shift in (0..usize::BITS as usize).step_by(7) {
            let byte = self.byte()?;
            let part = (byte & 0x7f) as usize;
            value = value
                .checked_add(part.checked_shl(shift as u32).ok_or(Error::SizeOverflow)?)
                .ok_or(Error::SizeOverflow)?;
            if byte & 0x80 == 0 {
                if self.pos - start != uleb_len(value) {
                    return Err(invalid("noncanonical column uleb"));
                }
                return Ok(value);
            }
        }
        Err(invalid("column uleb overflow"))
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

fn uleb_len(mut value: usize) -> usize {
    let mut len = 1;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

fn put_uleb(out: &mut Vec<u8>, mut value: usize) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return;
        }
    }
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_uleb(out, value.len());
    out.extend_from_slice(value);
}

fn bit_width(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()) as u8
}

fn bit_bytes(count: usize, width: u8) -> Result<usize> {
    count
        .checked_mul(width as usize)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or(Error::SizeOverflow)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    fn read(&mut self, width: u8) -> Result<u64> {
        let bit = self.bit;
        let end = bit + width as usize;
        if end > self.bytes.len().saturating_mul(8) {
            if self.bytes.len().checked_mul(8).is_none() {
                return Err(Error::SizeOverflow);
            }
            return Err(Error::Truncated);
        }
        let index = bit >> 3;
        let offset = (bit & 7) as u32;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let value = if index + 8 <= self.bytes.len() {
            let chunk = u64::from_le_bytes(
                self.bytes[index..index + 8]
                    .try_into()
                    .map_err(|_| Error::Truncated)?,
            );
            (chunk >> offset) & mask
        } else {
            let mut chunk = 0u64;
            for (position, &byte) in self.bytes[index..].iter().enumerate() {
                chunk |= u64::from(byte) << (8 * position);
            }
            (chunk >> offset) & mask
        };
        self.bit = end;
        Ok(value)
    }

    fn finish_zero_padded(&self) -> Result<()> {
        let used = self.bit;
        let total = self.bytes.len().checked_mul(8).ok_or(Error::SizeOverflow)?;
        if used > total {
            return Err(Error::Truncated);
        }
        for bit in used..total {
            if (self.bytes[bit / 8] >> (bit & 7)) & 1 != 0 {
                return Err(invalid("nonzero column bit padding"));
            }
        }
        Ok(())
    }
}

/// Validated 64-bit register-buffered bit reader for the hot streaming decoder.
///
/// The caller validates `bytes.len() == bit_bytes(count, width)` and tail
/// zero-padding up front via `FastBitReader::new_validated`, so each `next_u64`
/// is a register shift/mask on the hot path.
#[derive(Clone, Copy)]
struct FastBitReader<'a> {
    bytes: &'a [u8],
    bit_end: usize,
    buf: u64,
    avail: u32,
    width: u8,
    mask: u64,
}

impl<'a> FastBitReader<'a> {
    #[inline(always)]
    fn new_validated(bytes: &'a [u8], count: usize, width: u8) -> Result<Self> {
        let expected_len = bit_bytes(count, width)?;
        if bytes.len() != expected_len {
            return Err(Error::Truncated);
        }
        let used_bits = count * (width as usize);
        let tail_bits = used_bits & 7;
        if tail_bits != 0 {
            let last = *bytes.last().ok_or(Error::Truncated)?;
            if (last >> tail_bits) != 0 {
                return Err(invalid("nonzero column bit padding"));
            }
        }
        let mask = if width == 0 {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Ok(Self {
            bytes,
            bit_end: 0,
            buf: 0,
            avail: if width == 0 { u32::MAX } else { 0 },
            width,
            mask,
        })
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let w = self.width as u32;
        if self.avail >= w {
            let val = self.buf & self.mask;
            self.buf = self.buf.wrapping_shr(w);
            self.avail -= w;
            return val;
        }
        self.refill_and_read()
    }

    #[inline(always)]
    fn refill_and_read(&mut self) -> u64 {
        let w = self.width as u32;
        let bit_pos = self.bit_end - (self.avail as usize);
        let byte_idx = bit_pos >> 3;
        let bit_off = (bit_pos & 7) as u32;
        if byte_idx + 8 <= self.bytes.len() && bit_off + w <= 64 {
            let raw = unsafe {
                self.bytes
                    .as_ptr()
                    .add(byte_idx)
                    .cast::<u64>()
                    .read_unaligned()
                    .to_le()
            };
            let shifted = raw >> bit_off;
            let val = shifted & self.mask;
            let total_avail = 64 - bit_off;
            let rem_avail = total_avail - w;
            self.buf = if w < 64 { shifted >> w } else { 0 };
            self.avail = rem_avail;
            self.bit_end = bit_pos + (total_avail as usize);
            val
        } else {
            let mut raw = 0u128;
            let rem = self.bytes.len().saturating_sub(byte_idx).min(16);
            for i in 0..rem {
                raw |= (unsafe { *self.bytes.get_unchecked(byte_idx + i) } as u128) << (8 * i);
            }
            let shifted = raw >> bit_off;
            let val = (shifted as u64) & self.mask;
            let total_avail = (rem as u32 * 8).saturating_sub(bit_off);
            let rem_avail = total_avail.saturating_sub(w).min(64);
            self.buf = (shifted >> w) as u64;
            self.avail = rem_avail;
            self.bit_end = bit_pos + (w as usize) + (rem_avail as usize);
            val
        }
    }
}

#[inline(always)]
fn push_bit_values(
    out: &mut Vec<u8>,
    count: usize,
    width: u8,
    values: impl IntoIterator<Item = u64>,
) {
    if width == 0 || count == 0 {
        return;
    }
    let w = width as u32;
    let needed = (count.saturating_mul(width as usize) + 7) / 8;
    let old_len = out.len();
    out.reserve(needed + 8);
    let mut dst = unsafe { out.as_mut_ptr().add(old_len) };
    if w == 8 {
        for value in values {
            unsafe {
                *dst = value as u8;
                dst = dst.add(1);
            }
        }
        unsafe {
            out.set_len(old_len + needed);
        }
        return;
    }
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let mut acc = 0u64;
    let mut bits = 0u32;
    let mut iter = values.into_iter();
    if w <= 7 {
        let w2 = w * 2;
        let w4 = w * 4;
        let w8 = w * 8;
        for _ in 0..(count / 8) {
            let v0 = iter.next().unwrap_or(0) & mask;
            let v1 = iter.next().unwrap_or(0) & mask;
            let v2 = iter.next().unwrap_or(0) & mask;
            let v3 = iter.next().unwrap_or(0) & mask;
            let v4 = iter.next().unwrap_or(0) & mask;
            let v5 = iter.next().unwrap_or(0) & mask;
            let v6 = iter.next().unwrap_or(0) & mask;
            let v7 = iter.next().unwrap_or(0) & mask;
            let p01 = v0 | (v1 << w);
            let p23 = v2 | (v3 << w);
            let p45 = v4 | (v5 << w);
            let p67 = v6 | (v7 << w);
            let p0123 = p01 | (p23 << w2);
            let p4567 = p45 | (p67 << w2);
            let chunk = p0123 | (p4567 << w4);
            acc |= chunk << bits;
            bits += w8;
            unsafe {
                dst.cast::<u64>().write_unaligned(acc.to_le());
                dst = dst.add((bits >> 3) as usize);
            }
            acc >>= bits & !7;
            bits &= 7;
        }
    } else if w <= 14 {
        let w2 = w * 2;
        let w4 = w * 4;
        for _ in 0..(count / 4) {
            let v0 = iter.next().unwrap_or(0) & mask;
            let v1 = iter.next().unwrap_or(0) & mask;
            let v2 = iter.next().unwrap_or(0) & mask;
            let v3 = iter.next().unwrap_or(0) & mask;
            let p01 = v0 | (v1 << w);
            let p23 = v2 | (v3 << w);
            let chunk = p01 | (p23 << w2);
            acc |= chunk << bits;
            bits += w4;
            unsafe {
                dst.cast::<u64>().write_unaligned(acc.to_le());
                dst = dst.add((bits >> 3) as usize);
            }
            acc >>= bits & !7;
            bits &= 7;
        }
    }
    for value in iter {
        let val = value & mask;
        acc |= val.wrapping_shl(bits);
        let new_bits = bits + w;
        if new_bits >= 64 {
            unsafe {
                dst.cast::<u64>().write_unaligned(acc.to_le());
                dst = dst.add(8);
            }
            let spill = 64 - bits;
            acc = if spill < 64 { val >> spill } else { 0 };
            bits = new_bits - 64;
        } else {
            bits = new_bits;
        }
    }
    if bits > 0 {
        let tail_bytes = ((bits + 7) >> 3) as usize;
        let le = acc.to_le_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(le.as_ptr(), dst, tail_bytes);
        }
    }
    unsafe {
        out.set_len(old_len + needed);
    }
}

/// Append `src` to `dst` without calling `memcpy` for the small pieces that
/// dominate structured columns (segments and values are usually 1..16 bytes).
///
/// Pieces up to 32 bytes are written with two overlapping fixed-width stores;
/// larger pieces fall back to the platform copy. `dst` is reserved with 16
/// bytes of slack so every store stays inside the allocation, and the length
/// advances to exactly `old_len + src.len()`.
#[inline(always)]
fn copy_bytes(dst: &mut Vec<u8>, src: &[u8]) {
    let len = src.len();
    dst.reserve(len + 16);
    let pos = dst.len();
    unsafe {
        let out = dst.as_mut_ptr().add(pos);
        let input = src.as_ptr();
        if len > 32 {
            std::ptr::copy_nonoverlapping(input, out, len);
        } else if len >= 16 {
            std::ptr::copy_nonoverlapping(input, out, 16);
            let tail = len - 16;
            std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 16);
        } else if len >= 8 {
            std::ptr::copy_nonoverlapping(input, out, 8);
            let tail = len - 8;
            std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 8);
        } else if len >= 4 {
            std::ptr::copy_nonoverlapping(input, out, 4);
            let tail = len - 4;
            std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 4);
        } else {
            let mut index = 0;
            while index < len {
                *out.add(index) = *input.add(index);
                index += 1;
            }
        }
        dst.set_len(pos + len);
    }
}

/// Append-only output cursor over one preallocated buffer.
///
/// Row assembly reserves the full block once and then only advances a position:
/// no `Vec` length or capacity work happens per segment or value. The capacity
/// check keeps one bound test per piece; the write itself is the same
/// overlapping-store scheme as [`copy_bytes`].
struct OutSink {
    base: *mut u8,
    pos: usize,
    capacity: usize,
}

impl OutSink {
    fn new(base: *mut u8, capacity: usize) -> Self {
        Self {
            base,
            pos: 0,
            capacity,
        }
    }

    #[inline(always)]
    fn write(&mut self, src: &[u8]) -> Result<()> {
        let len = src.len();
        if self.pos + len > self.capacity {
            return Err(invalid("column output exceeds length"));
        }
        unsafe {
            let out = self.base.add(self.pos);
            let input = src.as_ptr();
            // The overlapping stores need 16 bytes of headroom; the final
            // pieces of an exactly-sized slice take the exact copy instead.
            if len > 32 || self.pos + len + 16 > self.capacity {
                std::ptr::copy_nonoverlapping(input, out, len);
            } else if len >= 16 {
                std::ptr::copy_nonoverlapping(input, out, 16);
                let tail = len - 16;
                std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 16);
            } else if len >= 8 {
                std::ptr::copy_nonoverlapping(input, out, 8);
                let tail = len - 8;
                std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 8);
            } else if len >= 4 {
                std::ptr::copy_nonoverlapping(input, out, 4);
                let tail = len - 4;
                std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 4);
            } else {
                let mut index = 0;
                while index < len {
                    *out.add(index) = *input.add(index);
                    index += 1;
                }
            }
        }
        self.pos += len;
        Ok(())
    }
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

const fn two_digit_table() -> [u8; 200] {
    let mut table = [0u8; 200];
    let mut value = 0usize;
    while value < 100 {
        table[value * 2] = b'0' + (value / 10) as u8;
        table[value * 2 + 1] = b'0' + (value % 10) as u8;
        value += 1;
    }
    table
}

static TWO_DIGITS: [u8; 200] = two_digit_table();

/// Unpadded 0..=9999 table: low 32 bits store 1..=4 ASCII digits in LE order,
/// high 32 bits store the byte length (1..=4).
const fn build_dec_10k() -> [u64; 10_000] {
    let mut table = [0u64; 10_000];
    let mut v = 0usize;
    while v < 10_000 {
        let d3 = b'0' + (v / 1000) as u8;
        let d2 = b'0' + ((v / 100) % 10) as u8;
        let d1 = b'0' + ((v / 10) % 10) as u8;
        let d0 = b'0' + (v % 10) as u8;
        let (bytes, len) = if v < 10 {
            ([d0, 0, 0, 0], 1u64)
        } else if v < 100 {
            ([d1, d0, 0, 0], 2u64)
        } else if v < 1000 {
            ([d2, d1, d0, 0], 3u64)
        } else {
            ([d3, d2, d1, d0], 4u64)
        };
        let word = u32::from_le_bytes(bytes) as u64;
        table[v] = word | (len << 32);
        v += 1;
    }
    table
}

/// Zero-padded 4-digit ASCII table for 0..=9999 in LE byte order (`"0000"`..`"9999"`).
const fn build_dec_padded_10k() -> [u32; 10_000] {
    let mut table = [0u32; 10_000];
    let mut v = 0usize;
    while v < 10_000 {
        let d3 = b'0' + (v / 1000) as u8;
        let d2 = b'0' + ((v / 100) % 10) as u8;
        let d1 = b'0' + ((v / 10) % 10) as u8;
        let d0 = b'0' + (v % 10) as u8;
        table[v] = u32::from_le_bytes([d3, d2, d1, d0]);
        v += 1;
    }
    table
}

const fn build_hex_byte_lut() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut b = 0usize;
    while b < 256 {
        let hi = HEX[b >> 4];
        let lo = HEX[b & 15];
        table[b] = u16::from_le_bytes([hi, lo]);
        b += 1;
    }
    table
}

const fn build_hex_nibble_lut() -> [u8; 256] {
    let mut table = [0xffu8; 256];
    let mut b = b'0';
    while b <= b'9' {
        table[b as usize] = b - b'0';
        b += 1;
    }
    let mut b = b'a';
    while b <= b'f' {
        table[b as usize] = b - b'a' + 10;
        b += 1;
    }
    table
}

static DEC_10K: [u64; 10_000] = build_dec_10k();
static DEC_PADDED_10K: [u32; 10_000] = build_dec_padded_10k();
static HEX_BYTE_LUT: [u16; 256] = build_hex_byte_lut();
static HEX_NIBBLE_LUT: [u8; 256] = build_hex_nibble_lut();

#[inline(always)]
unsafe fn write_decimal_fast(dst: *mut u8, value: u64) -> usize {
    unsafe {
        if value < 10_000 {
            let entry = *DEC_10K.get_unchecked(value as usize);
            dst.cast::<u32>().write_unaligned(entry as u32);
            (entry >> 32) as usize
        } else if value < 100_000_000 {
            let hi = (value / 10_000) as usize;
            let lo = (value % 10_000) as usize;
            let hi_entry = *DEC_10K.get_unchecked(hi);
            let hi_len = (hi_entry >> 32) as usize;
            dst.cast::<u32>().write_unaligned(hi_entry as u32);
            let lo_digits = *DEC_PADDED_10K.get_unchecked(lo);
            dst.add(hi_len).cast::<u32>().write_unaligned(lo_digits);
            hi_len + 4
        } else {
            write_decimal_large(dst, value)
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn write_decimal_large(dst: *mut u8, mut value: u64) -> usize {
    let mut buffer = [0u8; 24];
    let mut position = buffer.len();
    while value >= 10_000 {
        let rem = (value % 10_000) as usize;
        value /= 10_000;
        position -= 4;
        unsafe {
            buffer
                .as_mut_ptr()
                .add(position)
                .cast::<u32>()
                .write_unaligned(*DEC_PADDED_10K.get_unchecked(rem));
        }
    }
    let hi_entry = unsafe { *DEC_10K.get_unchecked(value as usize) };
    let hi_len = (hi_entry >> 32) as usize;
    position -= hi_len;
    let hi_bytes = (hi_entry as u32).to_le_bytes();
    buffer[position..position + hi_len].copy_from_slice(&hi_bytes[..hi_len]);
    let total_len = buffer.len() - position;
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.as_ptr().add(position), dst, total_len);
    }
    total_len
}

#[inline(always)]
fn u32_to_hex8_le(val: u32) -> u64 {
    unsafe {
        let b0 = *HEX_BYTE_LUT.get_unchecked((val >> 24) as usize) as u64;
        let b1 = *HEX_BYTE_LUT.get_unchecked(((val >> 16) & 0xff) as usize) as u64;
        let b2 = *HEX_BYTE_LUT.get_unchecked(((val >> 8) & 0xff) as usize) as u64;
        let b3 = *HEX_BYTE_LUT.get_unchecked((val & 0xff) as usize) as u64;
        b0 | (b1 << 16) | (b2 << 32) | (b3 << 48)
    }
}

#[inline]
fn write_decimal(out: &mut Vec<u8>, value: u64) {
    out.reserve(24);
    unsafe {
        let len = write_decimal_fast(out.as_mut_ptr().add(out.len()), value);
        out.set_len(out.len() + len);
    }
}

#[inline]
fn push_decimal(values: &mut FlatValues, value: u64) {
    let start = values.begin();
    write_decimal(&mut values.bytes, value);
    values.finish(start);
}

fn write_two_digits(out: &mut Vec<u8>, value: u8) {
    out.push(b'0' + value / 10);
    out.push(b'0' + value % 10);
}

fn write_four_digits(out: &mut Vec<u8>, value: u16) {
    out.push(b'0' + (value / 1000) as u8);
    out.push(b'0' + (value / 100 % 10) as u8);
    out.push(b'0' + (value / 10 % 10) as u8);
    out.push(b'0' + (value % 10) as u8);
}

fn read_bytes(cursor: &mut Cursor<'_>) -> Result<Vec<u8>> {
    let len = cursor.uleb()?;
    Ok(cursor.take(len)?.to_vec())
}

#[inline(always)]
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

#[inline(always)]
fn skip_space(data: &[u8], pos: &mut usize) {
    while data.get(*pos).is_some_and(|byte| is_space(*byte)) {
        *pos += 1;
    }
}

#[inline(always)]
fn scan_string(data: &[u8], mut pos: usize) -> Option<usize> {
    if data.get(pos) != Some(&b'"') {
        return None;
    }
    pos += 1;
    let len = data.len();
    let ptr = data.as_ptr();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v_quote = vdupq_n_u8(b'"');
        let v_slash = vdupq_n_u8(b'\\');
        let v_ctrl = vdupq_n_u8(0x20);
        while pos + 32 <= len {
            let c0 = vld1q_u8(ptr.add(pos));
            let c1 = vld1q_u8(ptr.add(pos + 16));
            let s0 = vorrq_u8(
                vceqq_u8(c0, v_quote),
                vorrq_u8(vceqq_u8(c0, v_slash), vcltq_u8(c0, v_ctrl)),
            );
            let s1 = vorrq_u8(
                vceqq_u8(c1, v_quote),
                vorrq_u8(vceqq_u8(c1, v_slash), vcltq_u8(c1, v_ctrl)),
            );
            let m0 = vget_lane_u64(
                vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(s0), 4)),
                0,
            );
            let m1 = vget_lane_u64(
                vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(s1), 4)),
                0,
            );
            if (m0 | m1) != 0 {
                let tz = if m0 != 0 {
                    m0.trailing_zeros()
                } else {
                    64 + m1.trailing_zeros()
                };
                let idx = (tz >> 2) as usize;
                let hit = *ptr.add(pos + idx);
                if hit == b'"' {
                    return Some(pos + idx + 1);
                }
                if hit < 0x20 {
                    return None;
                }
                if pos + idx + 1 >= len || *ptr.add(pos + idx + 1) < 0x20 {
                    return None;
                }
                pos += idx + 2;
                continue;
            }
            pos += 32;
        }
        while pos + 16 <= len {
            let chunk = vld1q_u8(ptr.add(pos));
            let special = vorrq_u8(
                vceqq_u8(chunk, v_quote),
                vorrq_u8(vceqq_u8(chunk, v_slash), vcltq_u8(chunk, v_ctrl)),
            );
            let nibbles = vget_lane_u64(
                vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(special), 4)),
                0,
            );
            if nibbles != 0 {
                let idx = (nibbles.trailing_zeros() >> 2) as usize;
                let hit = *ptr.add(pos + idx);
                if hit == b'"' {
                    return Some(pos + idx + 1);
                }
                if hit < 0x20 {
                    return None;
                }
                if pos + idx + 1 >= len || *ptr.add(pos + idx + 1) < 0x20 {
                    return None;
                }
                pos += idx + 2;
                continue;
            }
            pos += 16;
        }
    }
    while pos < len {
        match data[pos] {
            b'"' => return Some(pos + 1),
            b'\\' => {
                if pos + 1 >= len || data[pos + 1] < 0x20 {
                    return None;
                }
                pos += 2;
            }
            byte if byte < 0x20 => return None,
            _ => pos += 1,
        }
    }
    None
}

fn scan_compound(data: &[u8], mut pos: usize) -> Option<usize> {
    let mut closes = [0u8; 64];
    let mut depth = match data.get(pos)? {
        b'{' => {
            closes[0] = b'}';
            1usize
        }
        b'[' => {
            closes[0] = b']';
            1usize
        }
        _ => return None,
    };
    pos += 1;
    let mut overflow = Vec::new();
    while depth > 0 {
        let expected = if depth <= 64 {
            closes[depth - 1]
        } else {
            *overflow.last().unwrap()
        };
        match *data.get(pos)? {
            b'"' => pos = scan_string(data, pos)?,
            b'{' => {
                if depth < 64 {
                    closes[depth] = b'}';
                } else {
                    overflow.push(b'}');
                }
                depth += 1;
                pos += 1;
            }
            b'[' => {
                if depth < 64 {
                    closes[depth] = b']';
                } else {
                    overflow.push(b']');
                }
                depth += 1;
                pos += 1;
            }
            byte if byte == expected => {
                if depth > 64 {
                    overflow.pop();
                }
                depth -= 1;
                pos += 1;
            }
            b'}' | b']' | b'\n' | b'\r' => return None,
            _ => pos += 1,
        }
    }
    Some(pos)
}

#[inline(always)]
fn scan_primitive_end(ptr: *const u8, len: usize, pos: usize) -> usize {
    let mut end = pos + 1;
    while end < len {
        let b = unsafe { *ptr.add(end) };
        if b <= b',' || b == b'}' || b == b']' {
            break;
        }
        end += 1;
    }
    end
}

#[inline(always)]
fn scan_value(data: &[u8], pos: usize) -> Option<usize> {
    let first = *data.get(pos)?;
    match first {
        b'"' => scan_string(data, pos),
        b'{' | b'[' => scan_compound(data, pos),
        _ if first <= b',' || first == b'}' || first == b']' => None,
        _ => Some(scan_primitive_end(data.as_ptr(), data.len(), pos)),
    }
}

#[inline(always)]
fn find_byte(needle: u8, data: &[u8]) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut pos = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        if len >= 16 {
            let vn = vdupq_n_u8(needle);
            while pos + 32 <= len {
                let c0 = vceqq_u8(vld1q_u8(ptr.add(pos)), vn);
                let c1 = vceqq_u8(vld1q_u8(ptr.add(pos + 16)), vn);
                let any = vorrq_u8(c0, c1);
                if vmaxvq_u8(any) != 0 {
                    let m0 = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c0), 4)), 0);
                    if m0 != 0 {
                        return Some(pos + (m0.trailing_zeros() as usize >> 2));
                    }
                    let m1 = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c1), 4)), 0);
                    return Some(pos + 16 + (m1.trailing_zeros() as usize >> 2));
                }
                pos += 32;
            }
            if pos + 16 <= len {
                let c0 = vceqq_u8(vld1q_u8(ptr.add(pos)), vn);
                let m0 = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(c0), 4)), 0);
                if m0 != 0 {
                    return Some(pos + (m0.trailing_zeros() as usize >> 2));
                }
                pos += 16;
            }
        }
    }
    let splat = (needle as u64) * 0x0101_0101_0101_0101;
    while pos + 8 <= len {
        let word = unsafe { ptr.add(pos).cast::<u64>().read_unaligned() ^ splat };
        let z = word.wrapping_sub(0x0101_0101_0101_0101) & !word & 0x8080_8080_8080_8080;
        if z != 0 {
            return Some(pos + (z.trailing_zeros() as usize >> 3));
        }
        pos += 8;
    }
    if pos + 4 <= len {
        let word = unsafe { ptr.add(pos).cast::<u32>().read_unaligned() ^ (splat as u32) };
        let z = word.wrapping_sub(0x0101_0101) & !word & 0x8080_8080;
        if z != 0 {
            return Some(pos + (z.trailing_zeros() as usize >> 3));
        }
        pos += 4;
    }
    while pos < len {
        if unsafe { *ptr.add(pos) } == needle {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

#[inline(always)]
fn slice_eq_fast(a: &[u8], b: &[u8]) -> bool {
    let len = a.len();
    if len != b.len() {
        return false;
    }
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    unsafe {
        if len >= 8 {
            if pa.cast::<u64>().read_unaligned() != pb.cast::<u64>().read_unaligned() {
                return false;
            }
            if len <= 16 {
                let tail = len - 8;
                return pa.add(tail).cast::<u64>().read_unaligned()
                    == pb.add(tail).cast::<u64>().read_unaligned();
            }
            if len <= 32 {
                if pa.add(8).cast::<u64>().read_unaligned()
                    != pb.add(8).cast::<u64>().read_unaligned()
                {
                    return false;
                }
                let tail = len - 16;
                return pa.add(tail).cast::<u128>().read_unaligned()
                    == pb.add(tail).cast::<u128>().read_unaligned();
            }
            let mut off = 8usize;
            while off + 16 < len {
                if pa.add(off).cast::<u128>().read_unaligned()
                    != pb.add(off).cast::<u128>().read_unaligned()
                {
                    return false;
                }
                off += 16;
            }
            let tail = len - 16;
            return pa.add(tail).cast::<u128>().read_unaligned()
                == pb.add(tail).cast::<u128>().read_unaligned();
        }
        if len >= 4 {
            let tail = len - 4;
            return pa.cast::<u32>().read_unaligned() == pb.cast::<u32>().read_unaligned()
                && pa.add(tail).cast::<u32>().read_unaligned()
                    == pb.add(tail).cast::<u32>().read_unaligned();
        }
        if len >= 2 {
            return pa.cast::<u16>().read_unaligned() == pb.cast::<u16>().read_unaligned()
                && *pa.add(len - 1) == *pb.add(len - 1);
        }
        len == 0 || *pa == *pb
    }
}

/// Speculatively match `record` against a known `template` by comparing literal
/// segments and scanning values between them.
///
/// Safety / Equivalence invariant:
/// In any valid top-level JSON object record produced by `parse_record`, every
/// literal segment `segments[s]` for `s > 0` contains the `:` after the key of
/// slot `s`, and `segments[slots]` ends with the closing `}` (plus any trailing
/// newline). When `record` has no irregular whitespace outside strings (i.e.
/// `segments[0]` starts with `{`, every internal segment starts with `,` and ends
/// with `:`, and the final segment starts with `}` and ends at `record.len()`),
/// matching all `slots + 1` literal segments and scanning valid values between
/// them is 100% equivalent to `parse_record(record)` producing identical
/// segments. We verify this clean-segment property once via `template_is_clean`.
#[inline(always)]
fn template_is_clean(segments: &[&[u8]]) -> bool {
    let slots = segments.len().saturating_sub(1);
    if slots == 0 {
        return false;
    }
    let first = segments[0];
    if first.first() != Some(&b'{') || first.last() != Some(&b':') {
        return false;
    }
    if scan_string(first, 1) != Some(first.len() - 1) {
        return false;
    }
    for seg in &segments[1..slots] {
        if seg.first() != Some(&b',') || seg.last() != Some(&b':') {
            return false;
        }
        if scan_string(seg, 1) != Some(seg.len() - 1) {
            return false;
        }
    }
    let last = segments[slots];
    match last {
        b"}" | b"}\n" | b"}\r\n" => true,
        _ => false,
    }
}

#[inline(always)]
fn try_match_clean_template<'a>(
    record: &'a [u8],
    segments: &[&'a [u8]],
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> bool {
    let slots = segments.len() - 1;
    let seg0 = segments[0];
    if record.len() < seg0.len() || !slice_eq_fast(&record[..seg0.len()], seg0) {
        return false;
    }
    let mut pos = seg0.len();
    for slot in 0..slots {
        let val_end = match scan_value(record, pos) {
            Some(end) => end,
            None => return false,
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = record.get_unchecked(pos..val_end);
        }
        pos = val_end;
        let next_seg = unsafe { *segments.get_unchecked(slot + 1) };
        let next_end = pos + next_seg.len();
        if next_end > record.len() || !slice_eq_fast(&record[pos..next_end], next_seg) {
            return false;
        }
        pos = next_end;
    }
    pos == record.len()
}

#[inline(always)]
fn scan_delimited_value(data: &[u8], pos: usize, delim: u8) -> Option<usize> {
    if pos >= data.len() {
        return None;
    }
    let first = data[pos];
    if first == b'"' {
        return scan_string(data, pos);
    }
    if first == delim || first <= b' ' || first >= 0x7f {
        return None;
    }
    let len = data.len();
    let ptr = data.as_ptr();
    let mut end = pos + 1;
    while end < len {
        let b = unsafe { *ptr.add(end) };
        if b == delim || b == b'\n' || b == b'\r' {
            break;
        }
        if b <= b' '
            || b >= 0x7f
            || matches!(b, b'"' | b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']')
        {
            return None;
        }
        end += 1;
    }
    Some(end)
}

fn parse_delimited_spans(
    data: &[u8],
    delim: u8,
    spans: &mut [(u32, u32); MAX_SLOTS],
) -> Option<usize> {
    let mut body_len = data.len();
    if body_len == 0 || data[body_len - 1] != b'\n' {
        return None;
    }
    body_len -= 1;
    if body_len > 0 && data[body_len - 1] == b'\r' {
        body_len -= 1;
    }
    if body_len == 0 || body_len > 4096 {
        return None;
    }
    let body = &data[..body_len];
    let mut pos = 0usize;
    let mut count = 0usize;
    loop {
        let start = pos;
        let end = scan_delimited_value(body, pos, delim)?;
        if count >= MAX_SLOTS {
            return None;
        }
        spans[count] = (start as u32, end as u32);
        count += 1;
        pos = end;
        if pos == body_len {
            break;
        }
        if body[pos] != delim {
            return None;
        }
        pos += 1;
    }
    (count >= 2).then_some(count)
}

#[inline(always)]
fn is_logfmt_key_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')
}

fn parse_logfmt_spans(
    data: &[u8],
    spans: &mut [(u32, u32); MAX_SLOTS],
) -> Option<usize> {
    let mut body_len = data.len();
    if body_len == 0 || data[body_len - 1] != b'\n' {
        return None;
    }
    body_len -= 1;
    if body_len > 0 && data[body_len - 1] == b'\r' {
        body_len -= 1;
    }
    if body_len == 0 || body_len > 4096 {
        return None;
    }
    let body = &data[..body_len];
    let mut pos = 0usize;
    let mut count = 0usize;
    while pos < body_len {
        let key_start = pos;
        while pos < body_len && is_logfmt_key_byte(body[pos]) {
            pos += 1;
        }
        if pos == key_start || pos >= body_len || body[pos] != b'=' {
            return None;
        }
        pos += 1;
        if pos >= body_len {
            return None;
        }
        let val_start = pos;
        let val_end = if body[pos] == b'"' {
            scan_string(body, pos)?
        } else {
            let mut e = pos;
            while e < body_len
                && body[e] > b' '
                && body[e] < 0x7f
                && !matches!(
                    body[e],
                    b'"' | b';' | b',' | b'(' | b')' | b'{' | b'}' | b'[' | b']'
                )
            {
                e += 1;
            }
            if e == pos {
                return None;
            }
            e
        };
        if count >= MAX_SLOTS {
            return None;
        }
        spans[count] = (val_start as u32, val_end as u32);
        count += 1;
        pos = val_end;
        if pos == body_len {
            break;
        }
        if body[pos] != b' ' {
            return None;
        }
        pos += 1;
    }
    (count >= 2).then_some(count)
}

/// Split top-level structured record values (JSON objects, CSV/TSV/pipe-delimited
/// rows, or `key=value` logfmt lines). Nested JSON arrays/objects stay exact
/// raw slot values; irregular lines safely return `None` (emitted as `RAW` rows).
fn parse_record_spans(
    data: &[u8],
    spans: &mut [(u32, u32); MAX_SLOTS],
    overflow: &mut Vec<(u32, u32)>,
) -> Option<usize> {
    let mut pos = 0;
    skip_space(data, &mut pos);
    let first = *data.get(pos)?;
    if first != b'{' {
        if pos == 0 {
            let len = data.len();
            if len < 4 || data[len - 1] != b'\n' || len > 4098 {
                return None;
            }
            if first == b'"' {
                let end = scan_string(data, 0)?;
                let d = *data.get(end)?;
                if matches!(d, b',' | b'\t' | b'|') {
                    return parse_delimited_spans(data, d, spans);
                }
                return None;
            }
            if first >= 0x7f
                || matches!(
                    first,
                    b'(' | b')' | b';' | b'}' | b'[' | b']' | b',' | b'\t' | b'|' | b'='
                )
            {
                return None;
            }
            let ptr = data.as_ptr();
            let mut i = 1usize;
            while i < len {
                let b = unsafe { *ptr.add(i) };
                if b <= b' '
                    || b >= 0x7f
                    || matches!(
                        b,
                        b'=' | b',' | b'\t' | b'|' | b'"' | b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'
                    )
                {
                    if b == b'=' {
                        if let Some(c) = parse_logfmt_spans(data, spans) {
                            return Some(c);
                        }
                        let mut j = i + 1;
                        while j < len {
                            let bj = unsafe { *ptr.add(j) };
                            if bj <= b' '
                                || bj >= 0x7f
                                || matches!(
                                    bj,
                                    b',' | b'\t' | b'|' | b'"' | b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'
                                )
                            {
                                if matches!(bj, b',' | b'\t' | b'|') {
                                    return parse_delimited_spans(data, bj, spans);
                                }
                                return None;
                            }
                            j += 1;
                        }
                        return None;
                    }
                    if matches!(b, b',' | b'\t' | b'|') {
                        return parse_delimited_spans(data, b, spans);
                    }
                    return None;
                }
                i += 1;
            }
        }
        return None;
    }
    pos += 1;
    let mut count = 0usize;
    overflow.clear();
    loop {
        skip_space(data, &mut pos);
        if data.get(pos) == Some(&b'}') {
            pos += 1;
            break;
        }
        pos = scan_string(data, pos)?;
        skip_space(data, &mut pos);
        if data.get(pos) != Some(&b':') {
            return None;
        }
        pos += 1;
        skip_space(data, &mut pos);
        let start = pos;
        let end = scan_value(data, pos)?;
        if count < MAX_SLOTS {
            spans[count] = (start as u32, end as u32);
        } else {
            overflow.push((start as u32, end as u32));
        }
        count += 1;
        pos = end;
        skip_space(data, &mut pos);
        match data.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => {
                pos += 1;
                break;
            }
            _ => return None,
        }
    }
    skip_space(data, &mut pos);
    if pos != data.len() {
        return None;
    }
    Some(count)
}

#[inline(always)]
fn delimited_template_delim(segments: &[&[u8]]) -> Option<u8> {
    let slots = segments.len().saturating_sub(1);
    if slots < 2 || !segments[0].is_empty() {
        return None;
    }
    if !matches!(segments[slots], b"\n" | b"\r\n") {
        return None;
    }
    let seg1 = segments[1];
    if seg1.len() != 1 {
        return None;
    }
    let d = seg1[0];
    if !matches!(d, b',' | b'\t' | b'|') {
        return None;
    }
    for &seg in &segments[2..slots] {
        if seg.len() != 1 || seg[0] != d {
            return None;
        }
    }
    Some(d)
}

#[inline(always)]
fn template_is_clean_logfmt(segments: &[&[u8]]) -> bool {
    let slots = segments.len().saturating_sub(1);
    if slots < 2 || !matches!(segments[slots], b"\n" | b"\r\n") {
        return false;
    }
    let seg0 = segments[0];
    if seg0.len() < 2
        || seg0.last() != Some(&b'=')
        || !seg0[..seg0.len() - 1].iter().all(|&b| is_logfmt_key_byte(b))
    {
        return false;
    }
    for &seg in &segments[1..slots] {
        if seg.len() < 3
            || seg[0] != b' '
            || seg.last() != Some(&b'=')
            || !seg[1..seg.len() - 1].iter().all(|&b| is_logfmt_key_byte(b))
        {
            return false;
        }
    }
    true
}

#[inline(always)]
fn try_match_delimited_stream<'a>(
    rem: &'a [u8],
    segments: &[&'a [u8]],
    delim: u8,
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let rem_len = rem.len();
    let rem_ptr = rem.as_ptr();
    let mut pos = 0usize;
    for slot in 0..slots {
        if pos >= rem_len {
            return None;
        }
        let first = unsafe { *rem_ptr.add(pos) };
        let val_end = if first == b'"' {
            scan_string(rem, pos)?
        } else {
            let mut end = pos;
            while end < rem_len {
                let b = unsafe { *rem_ptr.add(end) };
                if b == delim || b.wrapping_sub(0x21) >= 0x5e || b == b'"' {
                    break;
                }
                end += 1;
            }
            if end == pos {
                return None;
            }
            end
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
        }
        pos = val_end;
        if slot + 1 < slots {
            if pos >= rem_len || unsafe { *rem_ptr.add(pos) } != delim {
                return None;
            }
            pos += 1;
        } else {
            if pos > 4096 {
                return None;
            }
            let last_seg = unsafe { *segments.get_unchecked(slots) };
            let next_end = pos + last_seg.len();
            if next_end > rem_len
                || !slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, last_seg)
            {
                return None;
            }
            pos = next_end;
        }
    }
    Some(pos)
}

#[inline(always)]
fn try_match_logfmt_stream<'a>(
    rem: &'a [u8],
    segments: &[&'a [u8]],
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<usize> {
    let slots = segments.len() - 1;
    let rem_len = rem.len();
    let rem_ptr = rem.as_ptr();
    let seg0 = segments[0];
    if rem_len < seg0.len() || !slice_eq_fast(&rem[..seg0.len()], seg0) {
        return None;
    }
    let mut pos = seg0.len();
    for slot in 0..slots {
        if pos >= rem_len {
            return None;
        }
        let first = unsafe { *rem_ptr.add(pos) };
        let val_end = if first == b'"' {
            scan_string(rem, pos)?
        } else {
            let mut end = pos;
            while end < rem_len {
                let b = unsafe { *rem_ptr.add(end) };
                if b <= b' ' || b == b'"' || b == 0x7f {
                    break;
                }
                end += 1;
            }
            if end == pos {
                return None;
            }
            end
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
        }
        pos = val_end;
        let next_seg = unsafe { *segments.get_unchecked(slot + 1) };
        let next_end = pos + next_seg.len();
        if next_end > rem_len
            || !slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, next_seg)
        {
            return None;
        }
        pos = next_end;
    }
    let last_len = segments[slots].len();
    (pos - last_len <= 4096).then_some(pos)
}

thread_local! {
    static ENCODE_VEC_POOL: std::cell::RefCell<Vec<Vec<&'static [u8]>>> =
        std::cell::RefCell::new(Vec::with_capacity(128));
}

#[inline(always)]
fn take_col_vec<'a>(est_rows: usize) -> Vec<&'a [u8]> {
    let v = ENCODE_VEC_POOL.with(|cell| cell.borrow_mut().pop());
    match v {
        Some(mut vec_static) => {
            vec_static.clear();
            vec_static.reserve(est_rows);
            unsafe { core::mem::transmute::<Vec<&'static [u8]>, Vec<&'a [u8]>>(vec_static) }
        }
        None => Vec::with_capacity(est_rows),
    }
}

impl<'a> Drop for Template<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        ENCODE_VEC_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                for mut col in core::mem::take(&mut self.values) {
                    if pool.len() < 128 && col.capacity() >= 16 {
                        col.clear();
                        let vec_static = unsafe {
                            core::mem::transmute::<Vec<&'a [u8]>, Vec<&'static [u8]>>(col)
                        };
                        pool.push(vec_static);
                    }
                }
            }
        });
    }
}

#[inline(always)]
fn try_match_clean_stream<'a>(
    rem: &'a [u8],
    templates: &[Template<'a>],
    clean_nl_flags: &[bool],
    shared_segs: &[[u8; 16]; 16],
    max_shared: &[u8; 16],
    mut cur_tpl: usize,
    out_values: &mut [&'a [u8]; MAX_SLOTS],
) -> Option<(usize, usize)> {
    let tpl_count = templates.len();
    if !clean_nl_flags[cur_tpl] {
        return None;
    }
    let rem_len = rem.len();
    let rem_ptr = rem.as_ptr();
    if tpl_count == 1 {
        let segs = &templates[0].segments;
        let slots = segs.len() - 1;
        let seg0 = segs[0];
        if rem_len < seg0.len() || !slice_eq_fast(&rem[..seg0.len()], seg0) {
            return None;
        }
        let mut pos = seg0.len();
        for slot in 0..slots {
            if pos >= rem_len {
                return None;
            }
            let first = unsafe { *rem_ptr.add(pos) };
            let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
            let val_end = match first {
                b'"' => scan_string(rem, pos)?,
                b'{' | b'[' => scan_compound(rem, pos)?,
                _ if first <= b',' || first == b'}' || first == b']' => return None,
                _ => scan_primitive_end(rem_ptr, rem_len, pos),
            };
            unsafe {
                *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
            }
            pos = val_end;
            let next_end = pos + next_seg.len();
            if next_end > rem_len || !slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, next_seg) {
                return None;
            }
            pos = next_end;
        }
        return Some((0, pos));
    }

    let seg0 = templates[cur_tpl].segments[0];
    let first_rem_byte = unsafe { *rem_ptr };
    if rem_len < seg0.len()
        || first_rem_byte != unsafe { *seg0.as_ptr() }
        || !slice_eq_fast(unsafe { rem.get_unchecked(..seg0.len()) }, seg0)
    {
        if tpl_count > 16 {
            return None;
        }
        let mut found = None;
        for alt in 0..tpl_count {
            if alt != cur_tpl && clean_nl_flags[alt] && shared_segs[cur_tpl][alt] == 0 {
                let alt_seg0 = templates[alt].segments[0];
                if rem_len >= alt_seg0.len()
                    && first_rem_byte == unsafe { *alt_seg0.as_ptr() }
                    && slice_eq_fast(unsafe { rem.get_unchecked(..alt_seg0.len()) }, alt_seg0)
                {
                    found = Some(alt);
                    break;
                }
            }
        }
        cur_tpl = found?;
    }
    let mut segs = &templates[cur_tpl].segments;
    let mut slots = segs.len() - 1;
    let mut pos = segs[0].len();
    let mut slot = 0usize;
    loop {
        if slot >= slots {
            return Some((cur_tpl, pos));
        }
        if cur_tpl < 16 && slot >= max_shared[cur_tpl] as usize {
            while slot < slots {
                if pos >= rem_len {
                    return None;
                }
                let first = unsafe { *rem_ptr.add(pos) };
                let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
                let val_end = match first {
                    b'"' => scan_string(rem, pos)?,
                    b'{' | b'[' => scan_compound(rem, pos)?,
                    _ if first <= b',' || first == b'}' || first == b']' => return None,
                    _ => scan_primitive_end(rem_ptr, rem_len, pos),
                };
                unsafe {
                    *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
                }
                pos = val_end;
                let next_end = pos + next_seg.len();
                if next_end > rem_len
                    || !slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, next_seg)
                {
                    return None;
                }
                pos = next_end;
                slot += 1;
            }
            return Some((cur_tpl, pos));
        }
        if pos >= rem_len {
            return None;
        }
        let first = unsafe { *rem_ptr.add(pos) };
        let val_end = match first {
            b'"' => scan_string(rem, pos)?,
            b'{' | b'[' => scan_compound(rem, pos)?,
            _ if first <= b',' || first == b'}' || first == b']' => return None,
            _ => scan_primitive_end(rem_ptr, rem_len, pos),
        };
        unsafe {
            *out_values.get_unchecked_mut(slot) = rem.get_unchecked(pos..val_end);
        }
        pos = val_end;
        let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
        let next_end = pos + next_seg.len();
        if next_end <= rem_len && slice_eq_fast(unsafe { rem.get_unchecked(pos..next_end) }, next_seg) {
            pos = next_end;
            slot += 1;
            continue;
        }
        if tpl_count > 16 {
            return None;
        }
        let req_shared = (slot + 1) as u8;
        let mut switched = false;
        for alt in 0..tpl_count {
            if alt != cur_tpl && clean_nl_flags[alt] && shared_segs[cur_tpl][alt] == req_shared {
                let alt_seg = templates[alt].segments[slot + 1];
                let alt_end = pos + alt_seg.len();
                if alt_end <= rem_len && slice_eq_fast(unsafe { rem.get_unchecked(pos..alt_end) }, alt_seg) {
                    cur_tpl = alt;
                    segs = &templates[cur_tpl].segments;
                    slots = segs.len() - 1;
                    pos = alt_end;
                    slot += 1;
                    switched = true;
                    break;
                }
            }
        }
        if !switched {
            return None;
        }
    }
}

thread_local! {
    static ENCODE_ROW_POOL: std::cell::RefCell<Vec<Row<'static>>> =
        std::cell::RefCell::new(Vec::with_capacity(512));
}

#[inline(always)]
fn take_row_vec<'a>(est_rows: usize) -> Vec<Row<'a>> {
    let mut vec_static = ENCODE_ROW_POOL.with(|cell| core::mem::take(&mut *cell.borrow_mut()));
    vec_static.clear();
    vec_static.reserve(est_rows);
    unsafe { core::mem::transmute::<Vec<Row<'static>>, Vec<Row<'a>>>(vec_static) }
}

#[inline(always)]
fn return_row_vec<'a>(mut rows: Vec<Row<'a>>) {
    if rows.capacity() >= 64 {
        rows.clear();
        let vec_static = unsafe { core::mem::transmute::<Vec<Row<'a>>, Vec<Row<'static>>>(rows) };
        ENCODE_ROW_POOL.with(|cell| {
            let mut pool = cell.borrow_mut();
            if vec_static.capacity() > pool.capacity() {
                *pool = vec_static;
            }
        });
    }
}

fn grammar_from_block<'a>(
    data: &'a [u8],
) -> Result<(Vec<Template<'a>>, Vec<Row<'a>>, usize, usize)> {
    let (templates, rows, template_rows, raw_rows, _raw_bytes) =
        grammar_from_block_mode(data, false)?;
    Ok((templates, rows, template_rows, raw_rows))
}

fn grammar_from_block_mode<'a>(
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
    while line_start < total_len {
        let rem = &data[line_start..];

        // Single-pass stream fast path: match directly on `rem` without scanning for `\n` first,
        // and switch templates mid-line on shared prefix segments without re-scanning `0..pos`.
        if templates.len() == 1 && clean_nl_flags[0] && !templates[0].values.is_empty() {
            let tpl = unsafe { templates.get_unchecked_mut(0) };
            let segs = &tpl.segments;
            let slots = tpl.values.len();
            let seg0 = segs[0];
            let seg0_len = seg0.len();
            let mut row_idx = unsafe { tpl.values.get_unchecked(0).len() };
            let mut col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
            let start_row_idx = row_idx;
            while line_start + seg0_len <= total_len {
                let rem_inner = unsafe { data.get_unchecked(line_start..) };
                let rem_len = rem_inner.len();
                let rem_ptr = rem_inner.as_ptr();
                if !slice_eq_fast(unsafe { rem_inner.get_unchecked(..seg0_len) }, seg0) {
                    break;
                }
                if row_idx == col_cap {
                    for slot in 0..slots {
                        unsafe {
                            let col = tpl.values.get_unchecked_mut(slot);
                            col.set_len(row_idx);
                            col.reserve(256);
                        }
                    }
                    col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
                }
                let mut pos = seg0_len;
                let mut ok = true;
                for slot in 0..slots {
                    if pos >= rem_len {
                        ok = false;
                        break;
                    }
                    let first = unsafe { *rem_ptr.add(pos) };
                    let next_seg = unsafe { *segs.get_unchecked(slot + 1) };
                    let val_end = match first {
                        b'"' => match scan_string(rem_inner, pos) {
                            Some(e) => e,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        b'{' | b'[' => match scan_compound(rem_inner, pos) {
                            Some(e) => e,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        _ if first <= b',' || first == b'}' || first == b']' => {
                            ok = false;
                            break;
                        }
                        _ => scan_primitive_end(rem_ptr, rem_len, pos),
                    };
                    unsafe {
                        *tpl.values
                            .get_unchecked_mut(slot)
                            .as_mut_ptr()
                            .add(row_idx) = rem_inner.get_unchecked(pos..val_end);
                    }
                    pos = val_end;
                    let next_end = pos + next_seg.len();
                    if next_end > rem_len
                        || !slice_eq_fast(unsafe { rem_inner.get_unchecked(pos..next_end) }, next_seg)
                    {
                        ok = false;
                        break;
                    }
                    pos = next_end;
                }
                if !ok {
                    break;
                }
                line_start += pos;
                row_idx += 1;
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
            let segs = &tpl.segments;
            let slots = tpl.values.len();
            let delim_opt = delim_flags[0];
            let mut row_idx = unsafe { tpl.values.get_unchecked(0).len() };
            let mut col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
            let start_row_idx = row_idx;
            while line_start < total_len {
                let rem_inner = unsafe { data.get_unchecked(line_start..) };
                let line_len = if let Some(d) = delim_opt {
                    match try_match_delimited_stream(rem_inner, segs, d, val_buf) {
                        Some(l) => l,
                        None => break,
                    }
                } else {
                    match try_match_logfmt_stream(rem_inner, segs, val_buf) {
                        Some(l) => l,
                        None => break,
                    }
                };
                line_start += line_len;
                if row_idx == col_cap {
                    for slot in 0..slots {
                        unsafe {
                            let col = tpl.values.get_unchecked_mut(slot);
                            col.set_len(row_idx);
                            col.reserve(256);
                        }
                    }
                    col_cap = unsafe { tpl.values.get_unchecked(0).capacity() };
                }
                for slot in 0..slots {
                    unsafe {
                        *tpl.values
                            .get_unchecked_mut(slot)
                            .as_mut_ptr()
                            .add(row_idx) = *val_buf.get_unchecked(slot);
                    }
                }
                row_idx += 1;
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
        } else if !templates.is_empty() {
            let mut matched_any = false;
            while line_start < total_len {
                let rem_inner = unsafe { data.get_unchecked(line_start..) };
                let Some((matched_idx, line_len)) = try_match_clean_stream(
                    rem_inner,
                    &templates,
                    &clean_nl_flags,
                    &shared_segs,
                    &max_shared,
                    last_template,
                    val_buf,
                ) else {
                    break;
                };
                line_start += line_len;
                let tpl = unsafe { templates.get_unchecked_mut(matched_idx) };
                let slots = tpl.values.len();
                if slots != 0 {
                    last_template = matched_idx;
                    let row_idx = unsafe { tpl.values.get_unchecked(0).len() };
                    if row_idx < unsafe { tpl.values.get_unchecked(0).capacity() } {
                        for slot in 0..slots {
                            unsafe {
                                let col = tpl.values.get_unchecked_mut(slot);
                                *col.as_mut_ptr().add(row_idx) = *val_buf.get_unchecked(slot);
                                col.set_len(row_idx + 1);
                            }
                        }
                    } else {
                        for slot in 0..slots {
                            unsafe {
                                tpl.values
                                    .get_unchecked_mut(slot)
                                    .push(*val_buf.get_unchecked(slot));
                            }
                        }
                    }
                }
                rows.push(Row::Template(matched_idx));
                template_rows += 1;
                matched_any = true;
            }
            if matched_any {
                continue;
            }
        }

        let line_len = match find_byte(b'\n', rem) {
            Some(idx) => idx + 1,
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
                return_row_vec(rows);
                return Err(invalid("column template bounds"));
            }
            let spans = &spans_buf[..slot_count];
            let index = templates
                .iter()
                .position(|template| {
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
                })
                .unwrap_or_else(|| {
                    if templates.len() == 1 && rows.len() == 1 && !clean_nl_flags[0] {
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
                    let is_clean = template_is_clean(&segments);
                    let is_clean_nl =
                        is_clean && matches!(segments.last().copied(), Some(b"}\n" | b"}\r\n"));
                    let delim_opt = delimited_template_delim(&segments);
                    let is_logfmt = delim_opt.is_none() && template_is_clean_logfmt(&segments);
                    if is_clean || segments[0].first() == Some(&b'{') {
                        has_json_tpl = true;
                    }
                    let new_idx = templates.len();
                    if new_idx < 16 {
                        let mut new_max = 0u8;
                        for prev_idx in 0..new_idx {
                            let min_segs = templates[prev_idx].segments.len().min(segments.len());
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
        if allow_early_abort && line_start >= 2048 {
            if (line_start >= 4096 && (template_rows < 2 || raw_bytes > (line_start * 3) / 4))
                || (template_rows == 0 && !templates.iter().any(|t| !t.values.is_empty()))
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

#[inline(always)]
fn parse_decimal(value: &[u8]) -> Option<u64> {
    let len = value.len();
    if len == 0 {
        return None;
    }
    let ptr = value.as_ptr();
    let b0 = unsafe { *ptr };
    if len == 1 {
        let d0 = b0.wrapping_sub(b'0');
        return if d0 <= 9 { Some(d0 as u64) } else { None };
    }
    if b0 == b'0' {
        return None;
    }
    unsafe {
        if len <= 8 {
            if len >= 4 {
                let d_hi = (ptr.cast::<u32>().read_unaligned() ^ 0x3030_3030) as u64;
                let d_lo = (ptr.add(len - 4).cast::<u32>().read_unaligned() ^ 0x3030_3030) as u64;
                let d = ((d_hi << ((8 - len) << 3)) & 0xffff_ffff) | (d_lo << 32);
                if ((d.wrapping_add(0x0606_0606_0606_0606)) | d) & 0xf0f0_f0f0_f0f0_f0f0 != 0 {
                    return None;
                }
                let p2 =
                    ((d & 0x00ff_00ff_00ff_00ff) * 10) + ((d >> 8) & 0x00ff_00ff_00ff_00ff);
                let p4 = ((p2 & 0x0000_ffff_0000_ffff) * 100)
                    + ((p2 >> 16) & 0x0000_ffff_0000_ffff);
                return Some((p4 & 0xffff_ffff) * 10_000 + (p4 >> 32));
            }
            let d_hi = (ptr.cast::<u16>().read_unaligned() ^ 0x3030) as u32;
            let d_lo = (ptr.add(len - 2).cast::<u16>().read_unaligned() ^ 0x3030) as u32;
            let d = ((d_hi << ((4 - len) << 3)) & 0xffff) | (d_lo << 16);
            if ((d.wrapping_add(0x0606_0606)) | d) & 0xf0f0_f0f0 != 0 {
                return None;
            }
            let p2 = ((d & 0x00ff_00ff) * 10) + ((d >> 8) & 0x00ff_00ff);
            return Some(((p2 & 0xffff) * 100 + (p2 >> 16)) as u64);
        }
        if len <= 19 {
            let mut result = parse_dec8_at(ptr)?;
            let mut pos = 8usize;
            if pos + 8 <= len {
                result = result * 100_000_000 + parse_dec8_at(ptr.add(pos))?;
                pos += 8;
            }
            if pos + 4 <= len {
                result = result * 10_000 + parse_dec4_at(ptr.add(pos))?;
                pos += 4;
            }
            while pos < len {
                let d = (*ptr.add(pos)).wrapping_sub(b'0');
                if d > 9 {
                    return None;
                }
                result = result * 10 + (d as u64);
                pos += 1;
            }
            return Some(result);
        }
    }
    let mut result = 0u64;
    for &byte in value {
        let d = byte.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        result = result.checked_mul(10)?.checked_add(u64::from(d))?;
    }
    Some(result)
}

#[inline(always)]
fn raw_model_size(values: &[&[u8]]) -> usize {
    let mut size = 1usize;
    for value in values {
        let len = value.len();
        size += if len < 0x80 { 1 + len } else { uleb_len(len) + len };
    }
    size
}

fn pack_raw_model(values: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw_model_size(values));
    out.push(MODEL_RAW);
    for value in values {
        put_bytes(&mut out, value);
    }
    out
}

#[inline(always)]
fn hash_slice(bytes: &[u8]) -> u64 {
    let len = bytes.len();
    let ptr = bytes.as_ptr();
    let h = unsafe {
        if len >= 16 {
            let a = ptr.cast::<u64>().read_unaligned();
            let b = ptr.add(len - 8).cast::<u64>().read_unaligned();
            let c = ptr.add((len >> 1) - 4).cast::<u64>().read_unaligned();
            let d = ptr.add(((len * 3) >> 2) - 4).cast::<u64>().read_unaligned();
            a ^ b.rotate_left(17) ^ c.rotate_left(33) ^ d.rotate_left(49) ^ (len as u64)
        } else if len >= 8 {
            let a = ptr.cast::<u64>().read_unaligned();
            let b = ptr.add(len - 8).cast::<u64>().read_unaligned();
            a ^ b.rotate_left(31) ^ (len as u64)
        } else if len >= 4 {
            let a = ptr.cast::<u32>().read_unaligned() as u64;
            let b = ptr.add(len - 4).cast::<u32>().read_unaligned() as u64;
            a ^ (b << 32) ^ (len as u64)
        } else if len > 0 {
            let a = *ptr as u64;
            let b = *ptr.add(len >> 1) as u64;
            let c = *ptr.add(len - 1) as u64;
            a | (b << 8) | (c << 16) | ((len as u64) << 24)
        } else {
            0
        }
    };
    // Fold the high product bits down: callers index with `hash & mask`, and the low
    // bits of a multiplicative hash only depend on the low input bits.
    let m = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    m ^ (m >> 29)
}

#[derive(Default)]
struct ModelScratch {
    parsed_u64: Vec<u64>,
    seg_tokens: Vec<u8>,
    seg_decimals: Vec<u64>,
    seg_pads: Vec<u16>,
    epoch_prefixes: Vec<u8>,
    epoch_flags: Vec<u8>,
    epoch_explicit: Vec<u64>,
    front_buf: Vec<u8>,
    front_lz: Vec<u8>,
    raw_lens: Vec<u8>,
    raw_stream: Vec<u8>,
    raw_lz: Vec<u8>,
}

thread_local! {
    static MODEL_SCRATCH: std::cell::RefCell<ModelScratch> =
        std::cell::RefCell::new(ModelScratch::default());
}

#[inline(always)]
fn common_prefix_suffix(prev: &[u8], cur: &[u8]) -> (usize, usize) {
    let max_lcp = prev.len().min(cur.len());
    let mut lcp = 0usize;
    let p_ptr = prev.as_ptr();
    let c_ptr = cur.as_ptr();
    let mut mismatched = false;
    while lcp + 8 <= max_lcp {
        let a = unsafe { p_ptr.add(lcp).cast::<u64>().read_unaligned().to_le() };
        let b = unsafe { c_ptr.add(lcp).cast::<u64>().read_unaligned().to_le() };
        let diff = a ^ b;
        if diff == 0 {
            lcp += 8;
        } else {
            lcp += (diff.trailing_zeros() >> 3) as usize;
            mismatched = true;
            break;
        }
    }
    if !mismatched {
        while lcp < max_lcp && unsafe { *p_ptr.add(lcp) == *c_ptr.add(lcp) } {
            lcp += 1;
        }
    }
    let max_lcs = (prev.len() - lcp).min(cur.len() - lcp);
    let mut lcs = 0usize;
    let prev_len = prev.len();
    let cur_len = cur.len();
    let mut lcs_mismatched = false;
    while lcs + 8 <= max_lcs {
        let a = unsafe { p_ptr.add(prev_len - lcs - 8).cast::<u64>().read_unaligned().to_le() };
        let b = unsafe { c_ptr.add(cur_len - lcs - 8).cast::<u64>().read_unaligned().to_le() };
        let diff = a ^ b;
        if diff == 0 {
            lcs += 8;
        } else {
            lcs += (diff.leading_zeros() >> 3) as usize;
            lcs_mismatched = true;
            break;
        }
    }
    if !lcs_mismatched {
        while lcs < max_lcs
            && unsafe { *p_ptr.add(prev_len - 1 - lcs) == *c_ptr.add(cur_len - 1 - lcs) }
        {
            lcs += 1;
        }
    }
    (lcp, lcs)
}

fn pack_dictionary_model_mode(
    values: &[&[u8]],
    limit: usize,
    allow_front: bool,
) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 3 >= limit {
        return None;
    }
    let first = values[0];
    let mut first_diff = 1usize;
    while first_diff < n && slice_eq_fast(values[first_diff], first) {
        first_diff += 1;
    }
    if first_diff == n {
        // 1-entry constant column (u == 1, width == 0):
        let size = 1 + 1 + uleb_len(first.len()) + first.len() + 1;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DICTIONARY);
        out.push(1);
        put_bytes(&mut out, first);
        out.push(0);
        return Some(out);
    }

    // At least 2 distinct entries exist (`first` and `second`), so `u >= 2`, `min_width >= 1`:
    let second = values[first_diff];
    let mut dict_bytes_sum =
        uleb_len(first.len()) + first.len() + uleb_len(second.len()) + second.len();
    let min_second_cost = if allow_front {
        3
    } else {
        uleb_len(second.len()) + second.len()
    };
    if 3 + uleb_len(first.len()) + first.len() + min_second_cost + (n + 7) / 8 >= limit {
        return None;
    }

    // Fast open-addressed hash table dynamically sized to `(n.next_power_of_two() * 2).clamp(32, 512)`
    // so small columns (`n <= 64`) don't zero a 1 KB table.
    let ht_cap = (n.next_power_of_two() * 2).clamp(32, 512);
    let ht_mask = ht_cap - 1;
    let mut ht_hashes = [core::mem::MaybeUninit::<u64>::uninit(); 512];
    let mut ht_slots_uninit = [core::mem::MaybeUninit::<u16>::uninit(); 512];
    let ht_slots: &mut [u16] = unsafe {
        let ptr = ht_slots_uninit.as_mut_ptr().cast::<u16>();
        let s = std::slice::from_raw_parts_mut(ptr, ht_cap);
        s.fill(u16::MAX);
        s
    };
    let mut unique = [core::mem::MaybeUninit::<&[u8]>::uninit(); 256];
    unique[0].write(first);
    unique[1].write(second);
    let h0 = hash_slice(first);
    let pos0 = (h0 as usize) & ht_mask;
    ht_hashes[pos0].write(h0);
    ht_slots[pos0] = 0;
    let h1 = hash_slice(second);
    let mut pos1 = (h1 as usize) & ht_mask;
    if pos1 == pos0 {
        pos1 = (pos1 + 1) & ht_mask;
    }
    ht_hashes[pos1].write(h1);
    ht_slots[pos1] = 1;
    let mut unique_len = 2usize;

    let mut raw_ids_inline = [core::mem::MaybeUninit::<u8>::uninit(); 2048];
    let mut raw_ids_overflow: Vec<u8> = if n > 2048 {
        let mut v = Vec::with_capacity(n);
        v.resize(first_diff, 0);
        v.push(1);
        v
    } else {
        unsafe {
            std::ptr::write_bytes(raw_ids_inline.as_mut_ptr().cast::<u8>(), 0, first_diff);
        }
        raw_ids_inline[first_diff].write(1);
        Vec::new()
    };
    let mut overflowed_small_ht = false;

    for (row_idx, &val) in values.iter().enumerate().skip(first_diff + 1) {
        let h = hash_slice(val);
        let mut pos = (h as usize) & ht_mask;
        let id = loop {
            let slot_id = unsafe { *ht_slots.get_unchecked(pos) };
            if slot_id == u16::MAX {
                let new_id = unique_len;
                dict_bytes_sum += uleb_len(val.len()) + val.len();
                let u = new_id + 1;
                let min_width = bit_width(new_id as u64);
                let min_packed = (n * (min_width as usize) + 7) / 8;
                let est_dict = if allow_front && new_id < 256 {
                    uleb_len(first.len()) + first.len() + (u - 1) * 3
                } else {
                    dict_bytes_sum
                };
                if 1 + uleb_len(u) + est_dict + 1 + min_packed >= limit {
                    return None;
                }
                if new_id >= 256 {
                    overflowed_small_ht = true;
                    break 0;
                }
                unique[new_id].write(val);
                unique_len = u;
                ht_hashes[pos].write(h);
                unsafe {
                    *ht_slots.get_unchecked_mut(pos) = new_id as u16;
                }
                break new_id as u8;
            }
            let slot_hash = unsafe { ht_hashes[pos].assume_init() };
            let slot_slice = unsafe { unique[slot_id as usize].assume_init() };
            if slot_hash == h && slice_eq_fast(slot_slice, val) {
                break slot_id as u8;
            }
            pos = (pos + 1) & ht_mask;
        };
        if overflowed_small_ht {
            break;
        }
        if n <= 2048 {
            raw_ids_inline[row_idx].write(id);
        } else {
            raw_ids_overflow.push(id);
        }
    }

    if !overflowed_small_ht {
        let u = unique_len;
        if u == n && (!allow_front || n > 256 || first.len() < 8) {
            return None;
        }
        let width = bit_width((u - 1) as u64);
        let packed_len = bit_bytes(n, width).ok()?;
        let unique_init: &[&[u8]] =
            unsafe { std::slice::from_raw_parts(unique.as_ptr().cast::<&[u8]>(), u) };
        let raw_ids_slice: &mut [u8] = if n <= 2048 {
            unsafe { std::slice::from_raw_parts_mut(raw_ids_inline.as_mut_ptr().cast::<u8>(), n) }
        } else {
            &mut raw_ids_overflow[..]
        };
        let already_sorted = (1..u).all(|i| unique_init[i - 1] < unique_init[i]);
        if already_sorted {
            return build_front_dict_packet(
                u,
                n,
                width,
                packed_len,
                dict_bytes_sum,
                unique_init,
                raw_ids_slice,
                limit,
                allow_front,
            );
        }
        #[inline(always)]
        fn prefix_key_be(s: &[u8]) -> u64 {
            let len = s.len();
            if len >= 8 {
                unsafe { s.as_ptr().cast::<u64>().read_unaligned().to_be() }
            } else {
                unsafe { (read_u128_partial(s.as_ptr(), len) as u64).swap_bytes() }
            }
        }
        let mut order_uninit = [core::mem::MaybeUninit::<(u64, u8)>::uninit(); 256];
        let order_slice: &mut [(u64, u8)] = unsafe {
            let ptr = order_uninit.as_mut_ptr().cast::<(u64, u8)>();
            for i in 0..u {
                let s = *unique_init.get_unchecked(i);
                *ptr.add(i) = (prefix_key_be(s), i as u8);
            }
            std::slice::from_raw_parts_mut(ptr, u)
        };
        order_slice.sort_unstable_by(|&(ka, a), &(kb, b)| {
            if ka != kb {
                ka.cmp(&kb)
            } else {
                unsafe {
                    let sa = *unique_init.get_unchecked(a as usize);
                    let sb = *unique_init.get_unchecked(b as usize);
                    if sa.len() >= 8 && sb.len() >= 8 {
                        sa.get_unchecked(8..).cmp(sb.get_unchecked(8..))
                    } else {
                        sa.cmp(sb)
                    }
                }
            }
        });
        let mut sorted_unique_uninit = [core::mem::MaybeUninit::<&[u8]>::uninit(); 256];
        let mut rank_of_uninit = [core::mem::MaybeUninit::<u8>::uninit(); 256];
        let sorted_ptr = sorted_unique_uninit.as_mut_ptr().cast::<&[u8]>();
        let rank_of_ptr = rank_of_uninit.as_mut_ptr().cast::<u8>();
        for (rank, &(_, orig_id)) in order_slice.iter().enumerate() {
            unsafe {
                *sorted_ptr.add(rank) = *unique_init.get_unchecked(orig_id as usize);
                *rank_of_ptr.add(orig_id as usize) = rank as u8;
            }
        }
        let sorted_entries: &[&[u8]] =
            unsafe { std::slice::from_raw_parts(sorted_ptr, u) };
        for id in raw_ids_slice.iter_mut() {
            *id = unsafe { *rank_of_ptr.add(*id as usize) };
        }

        return build_front_dict_packet(
            u,
            n,
            width,
            packed_len,
            dict_bytes_sum,
            sorted_entries,
            raw_ids_slice,
            limit,
            allow_front,
        );
    }

    // Fallback for dictionaries with > 256 unique entries (`MODEL_DICTIONARY` only):
    if n <= 256 {
        return None;
    }
    let min_packed_len = (n * 9 + 7) >> 3;
    if 4 + min_packed_len >= limit {
        return None;
    }
    // Hash dedup (no full sort): exact size check first, then sort only the `u`
    // unique entries and map each row to its rank via the dedup id.
    let cap = (n * 2).next_power_of_two();
    let mask = cap - 1;
    let hshift = 64 - cap.trailing_zeros();
    let mut slots: Vec<u32> = vec![u32::MAX; cap];
    let mut uniq: Vec<&[u8]> = Vec::with_capacity(n.min(1024));
    let mut row_ids: Vec<u32> = Vec::with_capacity(n);
    let mut plain_dict_bytes = 0usize;
    for &val in values {
        // Multiplicative hash: take the high bits (low bits only see low input bits).
        let mut pos = (hash_slice(val) >> hshift) as usize;
        let id = loop {
            let s = slots[pos];
            if s == u32::MAX {
                let id = uniq.len() as u32;
                slots[pos] = id;
                uniq.push(val);
                plain_dict_bytes += uleb_len(val.len()) + val.len();
                if 3 + plain_dict_bytes + min_packed_len >= limit {
                    return None;
                }
                break id;
            }
            if slice_eq_fast(uniq[s as usize], val) {
                break s;
            }
            pos = (pos + 1) & mask;
        };
        row_ids.push(id);
    }
    let u = uniq.len();
    if u == n {
        return None;
    }
    let width = bit_width((u - 1) as u64);
    let packed_len = bit_bytes(n, width).ok()?;
    let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
    if size >= limit {
        return None;
    }
    let mut order: Vec<u32> = (0..u as u32).collect();
    order.sort_unstable_by(|&a, &b| uniq[a as usize].cmp(uniq[b as usize]));
    let mut rank = vec![0u32; u];
    for (r, &id) in order.iter().enumerate() {
        rank[id as usize] = r as u32;
    }
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_DICTIONARY);
    put_uleb(&mut out, u);
    for &id in &order {
        put_bytes(&mut out, uniq[id as usize]);
    }
    out.push(width);
    push_bit_values(
        &mut out,
        n,
        width,
        row_ids.iter().map(|&id| rank[id as usize] as u64),
    );
    Some(out)
}

#[inline(always)]
fn push_nibble_bytes(out: &mut Vec<u8>, s: &[u8], lut_ptr: *const u8) {
    let len = s.len();
    if len == 0 {
        return;
    }
    let needed = (len + 1) >> 1;
    let old_len = out.len();
    out.reserve(needed);
    unsafe {
        let src = s.as_ptr();
        let mut dst = out.as_mut_ptr().add(old_len);
        let mut j = 0usize;
        while j + 1 < len {
            let hi = *lut_ptr.add(*src.add(j) as usize);
            let lo = *lut_ptr.add(*src.add(j + 1) as usize);
            *dst = (hi << 4) | lo;
            dst = dst.add(1);
            j += 2;
        }
        if j < len {
            let hi = *lut_ptr.add(*src.add(j) as usize);
            *dst = hi << 4;
        }
        out.set_len(old_len + needed);
    }
}

#[inline(always)]
fn push_5bit_bytes(out: &mut Vec<u8>, s: &[u8], lut_ptr: *const u8) {
    let mut acc = 0u64;
    let mut bits = 0u32;
    for &b in s {
        let v = unsafe { *lut_ptr.add(b as usize) as u64 };
        acc |= v << bits;
        bits += 5;
        if bits >= 32 {
            out.extend_from_slice(&(acc as u32).to_le_bytes());
            acc >>= 32;
            bits -= 32;
        }
    }
    while bits > 0 {
        out.push(acc as u8);
        acc >>= 8;
        bits = bits.saturating_sub(8);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_front_dict_packet(
    u: usize,
    n: usize,
    width: u8,
    packed_len: usize,
    plain_dict_bytes: usize,
    entries: &[&[u8]],
    ranks: &[u8],
    limit: usize,
    allow_front: bool,
) -> Option<Vec<u8>> {
    if !allow_front || u < 2 || u > 256 {
        if u >= n {
            return None;
        }
        let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DICTIONARY);
        put_uleb(&mut out, u);
        for &entry in entries {
            put_bytes(&mut out, entry);
        }
        out.push(width);
        push_bit_values(&mut out, n, width, ranks.iter().map(|&r| r as u64));
        return Some(out);
    }

    // 1. Precalculate front+back delta dictionary stats and 256-bit char masks over `mid` in-place:
    const HEX_M0: u64 = 0x03ff_0000_0000_0000; // '0'..='9' (48..=57)
    const HEX_M1: u64 = 0x0000_007e_0000_0000; // 'a'..='f' (97..=102)
    let mut lcp_lcs_uninit = [core::mem::MaybeUninit::<(u32, u32)>::uninit(); 256];
    let mut is_hex_mid_uninit = [core::mem::MaybeUninit::<bool>::uninit(); 256];
    let lcp_lcs_ptr = lcp_lcs_uninit.as_mut_ptr().cast::<(u32, u32)>();
    let is_hex_mid_ptr = is_hex_mid_uninit.as_mut_ptr().cast::<bool>();
    let first = unsafe { *entries.get_unchecked(0) };
    let mut front_dict_bytes = uleb_len(first.len()) + first.len();
    let mut all_4bit_saved = 0usize;
    let mut all_5bit_saved = 0usize;
    let mut hex_nibble_bytes = uleb_len(first.len()) + first.len();
    let mut char_mask = [0u64; 4];
    let mut hex_mask = [0u64; 2];
    let mut prev = first;
    for i in 1..u {
        let cur = unsafe { *entries.get_unchecked(i) };
        let (lcp, lcs) = common_prefix_suffix(prev, cur);
        unsafe {
            *lcp_lcs_ptr.add(i) = (lcp as u32, lcs as u32);
        }
        let mid = &cur[lcp..cur.len() - lcs];
        let base_hdr = uleb_len(lcp) + uleb_len(lcs);
        front_dict_bytes += base_hdr + uleb_len(mid.len()) + mid.len();
        all_4bit_saved += mid.len() >> 1;
        all_5bit_saved += mid.len() - ((mid.len() * 5 + 7) >> 3);
        let mut m0 = 0u64;
        let mut m1 = 0u64;
        let mut m_hi = 0u8;
        for &b in mid {
            let bit = 1u64 << (b & 63);
            if b < 64 {
                m0 |= bit;
            } else {
                m1 |= bit;
            }
            m_hi |= b;
        }
        char_mask[0] |= m0;
        if (m_hi & 0x80) == 0 {
            char_mask[1] |= m1;
        } else {
            m1 = 0;
            for &b in mid {
                if b >= 64 {
                    char_mask[(b >> 6) as usize] |= 1u64 << (b & 63);
                }
            }
        }
        if mid.len() >= 2 && ((m0 & !HEX_M0) | (m1 & !HEX_M1) | ((m_hi & 0x80) as u64)) == 0 {
            unsafe {
                *is_hex_mid_ptr.add(i) = true;
            }
            hex_mask[0] |= m0;
            hex_mask[1] |= m1;
            hex_nibble_bytes += base_hdr + uleb_len(mid.len() << 1) + ((mid.len() + 1) >> 1);
        } else {
            unsafe {
                *is_hex_mid_ptr.add(i) = false;
            }
            hex_nibble_bytes += base_hdr + uleb_len((mid.len() << 1) | 1) + mid.len();
        }
        prev = cur;
    }

    // Choose best dictionary encoding (`dict_mode`: 0 = plain, 1 = front+back delta,
    // 2 = FAST_LZ front+back delta, 3 = 4-bit/5-bit palette-packed front+back delta):
    let mut dict_mode = if front_dict_bytes < plain_dict_bytes || u == n {
        1u8
    } else {
        0u8
    };
    let mut chosen_dict_bytes = if u == n {
        front_dict_bytes
    } else {
        front_dict_bytes.min(plain_dict_bytes)
    };
    let mut pal = [0u8; 32];
    let mut pal_len = 0usize;
    let mut pal_mixed = false;
    let all_pop = (char_mask[0].count_ones()
        + char_mask[1].count_ones()
        + char_mask[2].count_ones()
        + char_mask[3].count_ones()) as usize;
    if all_pop >= 1 && all_pop <= 32 {
        let saved = if all_pop <= 16 {
            all_4bit_saved
        } else {
            all_5bit_saved
        };
        if saved > all_pop + 4 {
            let packed_dict_bytes = 1 + all_pop + front_dict_bytes - saved;
            if packed_dict_bytes + 4 < chosen_dict_bytes {
                for word_idx in 0..4 {
                    let mut w = char_mask[word_idx];
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        pal[pal_len] = ((word_idx << 6) | bit) as u8;
                        pal_len += 1;
                    }
                }
                dict_mode = 3;
                chosen_dict_bytes = packed_dict_bytes;
            }
        }
    } else {
        let hex_pop = (hex_mask[0].count_ones() + hex_mask[1].count_ones()) as usize;
        if hex_pop >= 1 && hex_pop <= 16 {
            let nibble_dict_bytes = 1 + hex_pop + hex_nibble_bytes;
            if nibble_dict_bytes + 8 < chosen_dict_bytes {
                for word_idx in 0..2 {
                    let mut w = hex_mask[word_idx];
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        w &= w - 1;
                        pal[pal_len] = ((word_idx << 6) | bit) as u8;
                        pal_len += 1;
                    }
                }
                pal_mixed = true;
                dict_mode = 3;
                chosen_dict_bytes = nibble_dict_bytes;
            }
        }
    }

    // 2. Choose best index encoding (`idx_mode`: 0 = plain, 1 = repeat-prev, 2 = mode-exc, 3 = mode + non-mode repeat):
    let mut idx_mode = 0u8;
    let mut chosen_idx_bytes = packed_len;
    let mut mode_rank = 0u8;
    let mut exc1 = 0usize;
    let mut rem2 = 0usize;
    let mut exc3 = 0usize;
    let exc_bound = u - 1;
    let exc_width = bit_width(exc_bound.saturating_sub(1) as u64);

    if packed_len >= 24 && u < n {
        let mut freq_uninit = [core::mem::MaybeUninit::<u32>::uninit(); 256];
        let freq: &mut [u32] = unsafe {
            let ptr = freq_uninit.as_mut_ptr().cast::<u32>();
            let s = std::slice::from_raw_parts_mut(ptr, u);
            s.fill(0);
            s
        };
        let mut same_prev = 0usize;
        let mut prev_r = u8::MAX;
        for &r in ranks {
            unsafe {
                *freq.get_unchecked_mut(r as usize) += 1;
            }
            same_prev += (r == prev_r) as usize;
            prev_r = r;
        }
        let mut mode_count = 0usize;
        for (r, &c_u32) in freq.iter().enumerate() {
            let c = c_u32 as usize;
            if c > mode_count {
                mode_count = c;
                mode_rank = r as u8;
            }
        }
        let bm_n_bytes = (n + 7) / 8;
        exc1 = n - same_prev;
        let idx1_bytes = bm_n_bytes + (exc1 * (width as usize) + 7) / 8;
        rem2 = n - mode_count;
        let idx2_bytes =
            uleb_len(mode_rank as usize) + bm_n_bytes + (rem2 * (exc_width as usize) + 7) / 8;

        if idx1_bytes + 8 < chosen_idx_bytes {
            idx_mode = 1;
            chosen_idx_bytes = idx1_bytes;
        }
        if idx2_bytes + 8 < chosen_idx_bytes {
            idx_mode = 2;
            chosen_idx_bytes = idx2_bytes;
        }
        if exc_width >= 2 && rem2 * (exc_width as usize) > rem2 + 64 {
            let mut non_mode_rep = 0usize;
            let mut last_nm = u8::MAX;
            for &r in ranks {
                if r != mode_rank {
                    if r == last_nm {
                        non_mode_rep += 1;
                    } else {
                        last_nm = r;
                    }
                }
            }
            exc3 = rem2 - non_mode_rep;
            let idx3_bytes = uleb_len(mode_rank as usize)
                + bm_n_bytes
                + (rem2 + 7) / 8
                + (exc3 * (exc_width as usize) + 7) / 8;
            if idx3_bytes + 8 < chosen_idx_bytes {
                idx_mode = 3;
                chosen_idx_bytes = idx3_bytes;
            }
        }
    }

    let mut built_front_in_scratch = false;
    if dict_mode != 3
        && u >= 16
        && front_dict_bytes >= 512
        && front_dict_bytes <= 65536
        && 1 + uleb_len(u) + 1 + chosen_idx_bytes + 32 < limit
    {
        MODEL_SCRATCH.with(|cell| {
            let mut sc = cell.borrow_mut();
            let ModelScratch {
                front_buf,
                front_lz,
                ..
            } = &mut *sc;
            front_buf.clear();
            front_buf.reserve(front_dict_bytes);
            put_bytes(front_buf, first);
            for i in 1..u {
                let (lcp, lcs) = unsafe {
                    let p = *lcp_lcs_ptr.add(i);
                    (p.0 as usize, p.1 as usize)
                };
                let cur = unsafe { *entries.get_unchecked(i) };
                let mid = &cur[lcp..cur.len() - lcs];
                put_uleb(front_buf, lcp);
                put_uleb(front_buf, lcs);
                put_bytes(front_buf, mid);
            }
            built_front_in_scratch = true;
            front_lz.clear();
            front_lz.reserve(front_dict_bytes / 2);
            if pack_fast_lz_into(front_buf, front_lz).is_some() {
                let lz_total = uleb_len(front_lz.len()) + front_lz.len();
                if lz_total + 32 + (front_dict_bytes >> 3) < chosen_dict_bytes {
                    dict_mode = 2;
                    chosen_dict_bytes = lz_total;
                }
            }
        });
    }

    if dict_mode == 0 && idx_mode == 0 {
        if u >= n {
            return None;
        }
        let size = 1 + uleb_len(u) + plain_dict_bytes + 1 + packed_len;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DICTIONARY);
        put_uleb(&mut out, u);
        for &entry in entries {
            put_bytes(&mut out, entry);
        }
        out.push(width);
        push_bit_values(&mut out, n, width, ranks.iter().map(|&r| r as u64));
        return Some(out);
    }

    let size = 1 + uleb_len(u) + 1 + chosen_dict_bytes + chosen_idx_bytes;
    if size >= limit {
        return None;
    }
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_FRONT_DICT);
    put_uleb(&mut out, u);
    out.push(dict_mode | (idx_mode << 2));
    match dict_mode {
        0 => {
            for &entry in entries {
                put_bytes(&mut out, entry);
            }
        }
        1 => {
            if built_front_in_scratch {
                MODEL_SCRATCH.with(|cell| {
                    let sc = cell.borrow();
                    out.extend_from_slice(&sc.front_buf);
                });
            } else {
                put_bytes(&mut out, first);
                for i in 1..u {
                    let (lcp, lcs) = unsafe {
                        let p = *lcp_lcs_ptr.add(i);
                        (p.0 as usize, p.1 as usize)
                    };
                    let cur = unsafe { *entries.get_unchecked(i) };
                    let mid = &cur[lcp..cur.len() - lcs];
                    put_uleb(&mut out, lcp);
                    put_uleb(&mut out, lcs);
                    put_bytes(&mut out, mid);
                }
            }
        }
        2 => {
            MODEL_SCRATCH.with(|cell| {
                let sc = cell.borrow();
                put_uleb(&mut out, sc.front_lz.len());
                out.extend_from_slice(&sc.front_lz);
            });
        }
        3 => {
            let mut lut_uninit = [core::mem::MaybeUninit::<u8>::uninit(); 256];
            let lut_ptr = lut_uninit.as_mut_ptr().cast::<u8>();
            for (idx, &b) in pal[..pal_len].iter().enumerate() {
                unsafe {
                    *lut_ptr.add(b as usize) = idx as u8;
                }
            }
            out.push((pal_len as u8) | if pal_mixed { 0x80 } else { 0 });
            out.extend_from_slice(&pal[..pal_len]);
            put_bytes(&mut out, first);
            if !pal_mixed {
                for i in 1..u {
                    let (lcp, lcs) = unsafe {
                        let p = *lcp_lcs_ptr.add(i);
                        (p.0 as usize, p.1 as usize)
                    };
                    let cur = unsafe { *entries.get_unchecked(i) };
                    let mid = &cur[lcp..cur.len() - lcs];
                    put_uleb(&mut out, lcp);
                    put_uleb(&mut out, lcs);
                    put_uleb(&mut out, mid.len());
                    if pal_len <= 16 {
                        push_nibble_bytes(&mut out, mid, lut_ptr);
                    } else {
                        push_5bit_bytes(&mut out, mid, lut_ptr);
                    }
                }
            } else {
                for i in 1..u {
                    let (lcp, lcs) = unsafe {
                        let p = *lcp_lcs_ptr.add(i);
                        (p.0 as usize, p.1 as usize)
                    };
                    let cur = unsafe { *entries.get_unchecked(i) };
                    let mid = &cur[lcp..cur.len() - lcs];
                    put_uleb(&mut out, lcp);
                    put_uleb(&mut out, lcs);
                    if unsafe { *is_hex_mid_ptr.add(i) } {
                        put_uleb(&mut out, mid.len() << 1);
                        push_nibble_bytes(&mut out, mid, lut_ptr);
                    } else {
                        put_uleb(&mut out, (mid.len() << 1) | 1);
                        out.extend_from_slice(mid);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    match idx_mode {
        0 => {
            push_bit_values(&mut out, n, width, ranks.iter().map(|&r| r as u64));
        }
        1 => {
            push_bit_values(
                &mut out,
                n,
                1,
                (0..n).map(|i| u64::from(i > 0 && ranks[i] == ranks[i - 1])),
            );
            push_bit_values(
                &mut out,
                exc1,
                width,
                (0..n).filter_map(|i| {
                    if i == 0 || ranks[i] != ranks[i - 1] {
                        Some(ranks[i] as u64)
                    } else {
                        None
                    }
                }),
            );
        }
        2 => {
            put_uleb(&mut out, mode_rank as usize);
            push_bit_values(
                &mut out,
                n,
                1,
                ranks.iter().map(|&r| u64::from(r == mode_rank)),
            );
            push_bit_values(
                &mut out,
                rem2,
                exc_width,
                ranks.iter().filter_map(|&r| {
                    if r == mode_rank {
                        None
                    } else if r < mode_rank {
                        Some(r as u64)
                    } else {
                        Some((r - 1) as u64)
                    }
                }),
            );
        }
        3 => {
            put_uleb(&mut out, mode_rank as usize);
            push_bit_values(
                &mut out,
                n,
                1,
                ranks.iter().map(|&r| u64::from(r == mode_rank)),
            );
            let mut prev_nm = u8::MAX;
            push_bit_values(
                &mut out,
                rem2,
                1,
                ranks.iter().filter_map(|&r| {
                    if r == mode_rank {
                        None
                    } else if r == prev_nm {
                        Some(1u64)
                    } else {
                        prev_nm = r;
                        Some(0u64)
                    }
                }),
            );
            let mut prev_nm2 = u8::MAX;
            push_bit_values(
                &mut out,
                exc3,
                exc_width,
                ranks.iter().filter_map(|&r| {
                    if r == mode_rank || r == prev_nm2 {
                        None
                    } else {
                        prev_nm2 = r;
                        Some(if r < mode_rank {
                            r as u64
                        } else {
                            (r - 1) as u64
                        })
                    }
                }),
            );
        }
        _ => unreachable!(),
    }
    Some(out)
}

#[inline(always)]
fn hex_nibble(byte: u8) -> Option<u8> {
    let n = unsafe { *HEX_NIBBLE_LUT.get_unchecked(byte as usize) };
    if n != 0xff { Some(n) } else { None }
}

#[inline(always)]
fn parse_hex_u32_le(digits: &[u8]) -> Option<u32> {
    if digits.len() != 8 {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1_u8(digits.as_ptr());
        let d_dec = vsub_u8(v, vdup_n_u8(b'0'));
        let is_dec = vclt_u8(d_dec, vdup_n_u8(10));
        let is_hex = vclt_u8(vsub_u8(v, vdup_n_u8(b'a')), vdup_n_u8(6));
        let valid = vorr_u8(is_dec, is_hex);
        if vget_lane_u64(vreinterpret_u64_u8(valid), 0) != u64::MAX {
            return None;
        }
        let nib = vsub_u8(d_dec, vand_u8(is_hex, vdup_n_u8(39)));
        let nib16 = vreinterpret_u16_u8(nib);
        let paired = vorr_u16(vshl_n_u16(nib16, 4), vshr_n_u16(nib16, 8));
        let wide = vcombine_u16(paired, paired);
        return Some(vget_lane_u32(vreinterpret_u32_u8(vmovn_u16(wide)), 0));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut raw = [0u8; 4];
        let mut acc_err = 0u8;
        for i in 0..4 {
            let hi = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2] as usize) };
            let lo = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2 + 1] as usize) };
            acc_err |= hi | lo;
            raw[i] = (hi << 4) | lo;
        }
        if acc_err & 0xf0 != 0 {
            return None;
        }
        Some(u32::from_le_bytes(raw))
    }
}

#[inline(always)]
fn parse_hex_u64_le(digits: &[u8]) -> Option<u64> {
    if digits.len() != 16 {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1q_u8(digits.as_ptr());
        let d_dec = vsubq_u8(v, vdupq_n_u8(b'0'));
        let is_dec = vcltq_u8(d_dec, vdupq_n_u8(10));
        let is_hex = vcltq_u8(vsubq_u8(v, vdupq_n_u8(b'a')), vdupq_n_u8(6));
        let valid = vorrq_u8(is_dec, is_hex);
        let mask64 = vget_lane_u64(
            vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(valid), 4)),
            0,
        );
        if mask64 != u64::MAX {
            return None;
        }
        let nib = vsubq_u8(d_dec, vandq_u8(is_hex, vdupq_n_u8(39)));
        let nib16 = vreinterpretq_u16_u8(nib);
        let paired = vorrq_u16(vshlq_n_u16(nib16, 4), vshrq_n_u16(nib16, 8));
        return Some(vget_lane_u64(vreinterpret_u64_u8(vmovn_u16(paired)), 0));
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut raw = [0u8; 8];
        let mut acc_err = 0u8;
        for i in 0..8 {
            let hi = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2] as usize) };
            let lo = unsafe { *HEX_NIBBLE_LUT.get_unchecked(digits[i * 2 + 1] as usize) };
            acc_err |= hi | lo;
            raw[i] = (hi << 4) | lo;
        }
        if acc_err & 0xf0 != 0 {
            return None;
        }
        Some(u64::from_le_bytes(raw))
    }
}

#[inline(always)]
fn split_hex_suffix(value: &[u8]) -> Option<(&[u8], &[u8])> {
    if value.len() < 4 || value.first() != Some(&b'"') || value.last() != Some(&b'"') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    let mut start = inner.len();
    while start != 0 && hex_nibble(inner[start - 1]).is_some() {
        start -= 1;
    }
    let digits = &inner[start..];
    if digits.is_empty() || digits.len() % 2 != 0 {
        return None;
    }
    Some((&inner[..start], digits))
}

fn pack_hex_suffix_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let (prefix, first_digits) = split_hex_suffix(values[0])?;
    let raw_len = first_digits.len() / 2;
    if raw_len == 0 || raw_len > 64 {
        return None;
    }
    let size = 1 + uleb_len(prefix.len()) + prefix.len() + uleb_len(raw_len) + n * raw_len;
    if size >= limit {
        return None;
    }
    let expected_val_len = 2 + prefix.len() + first_digits.len();
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_HEX_SUFFIX);
    put_bytes(&mut out, prefix);
    put_uleb(&mut out, raw_len);
    let prefix_end = 1 + prefix.len();
    let qprefix = &values[0][..prefix_end];
    if raw_len == 4 {
        for &value in values {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            if prefix_end > 1 && hex_nibble(value[prefix_end - 1]).is_some() {
                return None;
            }
            let word = parse_hex_u32_le(&value[prefix_end..expected_val_len - 1])?;
            out.extend_from_slice(&word.to_le_bytes());
        }
    } else if raw_len == 8 {
        for &value in values {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            if prefix_end > 1 && hex_nibble(value[prefix_end - 1]).is_some() {
                return None;
            }
            let word = parse_hex_u64_le(&value[prefix_end..expected_val_len - 1])?;
            out.extend_from_slice(&word.to_le_bytes());
        }
    } else {
        for &value in values {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            if prefix_end > 1 && hex_nibble(value[prefix_end - 1]).is_some() {
                return None;
            }
            let digits = &value[prefix_end..expected_val_len - 1];
            for pair in digits.chunks_exact(2) {
                let hi = hex_nibble(pair[0])?;
                let lo = hex_nibble(pair[1])?;
                out.push((hi << 4) | lo);
            }
        }
    }
    Some(out)
}

fn pack_epoch_counter_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let (literal, first_digits) = split_hex_suffix(values[0])?;
    if first_digits.len() != 16 {
        return None;
    }
    let base_overhead = 1 + uleb_len(literal.len()) + literal.len() + 3 + ((n + 7) >> 3);
    if base_overhead + 5 >= limit {
        return None;
    }
    let expected_val_len = 2 + literal.len() + 16;
    let prefix_end = 1 + literal.len();
    let qprefix = &values[0][..prefix_end];
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.epoch_prefixes.clear();
        sc.epoch_prefixes.reserve(n);
        sc.epoch_flags.clear();
        sc.epoch_flags.reserve(n);
        sc.epoch_explicit.clear();
        let pref_ptr = sc.epoch_prefixes.as_mut_ptr();
        let flag_ptr = sc.epoch_flags.as_mut_ptr();
        let mut unique_prefixes = [0u32; MAX_EPOCH_PREFIXES];
        let mut unique_len = 0usize;
        let mut last_prefix = 0u32;
        let mut last_pid = u8::MAX;
        let mut previous = u64::MAX - 1;
        let mut explicit_bytes = 0usize;

        for (idx, &value) in values.iter().enumerate() {
            if value.len() != expected_val_len
                || value[expected_val_len - 1] != b'"'
                || !slice_eq_fast(&value[..prefix_end], qprefix)
            {
                return None;
            }
            let digits = &value[prefix_end..expected_val_len - 1];
            let word64 = parse_hex_u64_le(digits)?;
            let prefix = (word64 as u32).swap_bytes();
            let suffix = ((word64 >> 32) as u32).swap_bytes();
            let pid = if prefix == last_prefix && last_pid != u8::MAX {
                last_pid
            } else if let Some(pos) = unique_prefixes[..unique_len].iter().position(|&p| p == prefix) {
                last_prefix = prefix;
                last_pid = pos as u8;
                last_pid
            } else {
                if unique_len >= MAX_EPOCH_PREFIXES {
                    return None;
                }
                let id = unique_len as u8;
                unique_prefixes[unique_len] = prefix;
                unique_len += 1;
                last_prefix = prefix;
                last_pid = id;
                id
            };
            unsafe {
                *pref_ptr.add(idx) = pid;
            }
            let current = u64::from(suffix);
            if idx > 0 && previous + 1 == current {
                unsafe {
                    *flag_ptr.add(idx) = 0u8;
                }
            } else {
                unsafe {
                    *flag_ptr.add(idx) = 1u8;
                }
                sc.epoch_explicit.push(current);
                explicit_bytes += uleb_len(current as usize);
                if base_overhead + 4 * unique_len + explicit_bytes >= limit {
                    return None;
                }
                if idx == 7 && unique_len >= 7 && sc.epoch_explicit.len() == 8 {
                    return None;
                }
            }
            previous = current;
        }

        if unique_len == 0 {
            return None;
        }
        unsafe {
            sc.epoch_prefixes.set_len(n);
            sc.epoch_flags.set_len(n);
        }
        let mut order: [u8; MAX_EPOCH_PREFIXES] = core::array::from_fn(|i| i as u8);
        let order_slice = &mut order[..unique_len];
        order_slice.sort_unstable_by_key(|&id| unique_prefixes[id as usize]);
        let mut rank_of = [0u64; MAX_EPOCH_PREFIXES];
        for (rank, &orig_id) in order_slice.iter().enumerate() {
            rank_of[orig_id as usize] = rank as u64;
        }

        let width = bit_width((unique_len - 1) as u64);
        let size = 1
            + uleb_len(literal.len())
            + literal.len()
            + uleb_len(unique_len)
            + 4 * unique_len
            + 1
            + bit_bytes(n, width).ok()?
            + bit_bytes(n, 1).ok()?
            + uleb_len(sc.epoch_explicit.len())
            + sc.epoch_explicit
                .iter()
                .map(|value| uleb_len(*value as usize))
                .sum::<usize>();
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_EPOCH_COUNTER);
        put_bytes(&mut out, literal);
        put_uleb(&mut out, unique_len);
        for &orig_id in order_slice.iter() {
            out.extend_from_slice(&unique_prefixes[orig_id as usize].to_le_bytes());
        }
        out.push(width);
        push_bit_values(
            &mut out,
            n,
            width,
            sc.epoch_prefixes
                .iter()
                .map(|&pid| unsafe { *rank_of.get_unchecked(pid as usize) }),
        );
        push_bit_values(
            &mut out,
            n,
            1,
            sc.epoch_flags.iter().map(|&f| u64::from(f)),
        );
        put_uleb(&mut out, sc.epoch_explicit.len());
        for &value in &sc.epoch_explicit {
            put_uleb(&mut out, value as usize);
        }
        Some(out)
    })
}

#[inline(always)]
fn segment_byte_ok(byte: u8) -> bool {
    byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b' '
}

#[inline(always)]
fn canonical_segment_decimal(segment: &[u8]) -> Option<u64> {
    if segment.len() > MAX_DECIMAL_SEGMENT_DIGITS {
        return None;
    }
    parse_decimal(segment)
}

#[inline(always)]
fn validate_segment_slice(segment: &[u8]) -> bool {
    !segment.is_empty()
        && segment.len() <= MAX_SEGMENT_LEN
        && segment.iter().all(|&byte| segment_byte_ok(byte))
}

fn pack_segment_dictionary_model<'a>(values: &[&'a [u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 6 + (n + 7) / 8 >= limit {
        return None;
    }
    const HT_CAP: usize = 128;
    const HT_MASK: usize = HT_CAP - 1;
    let mut ht_hashes = [core::mem::MaybeUninit::<u64>::uninit(); HT_CAP];
    let mut ht_slots = [u8::MAX; HT_CAP];
    let mut dict_slices = [core::mem::MaybeUninit::<&'a [u8]>::uninit(); MAX_SEGMENT_DICT];
    let mut dict_len = 0usize;
    let mut dict_bytes_sum = 0usize;
    let mut max_decimal = 0u64;
    let mut max_pad = 0u64;

    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.seg_tokens.clear();
        sc.seg_tokens.reserve(n * (MAX_SEGMENTS_PER_VALUE + 1));
        sc.seg_decimals.clear();
        sc.seg_decimals.reserve(n * MAX_SEGMENTS_PER_VALUE);
        sc.seg_pads.clear();
        sc.seg_pads.reserve(n);

        for (row_idx, &value) in values.iter().enumerate() {
            let val_len = value.len();
            if val_len < 2 || value[0] != b'"' || value[val_len - 1] != b'"' {
                return None;
            }
            let inner = &value[1..val_len - 1];
            let mut end = inner.len();
            while end != 0 && inner[end - 1] == b' ' {
                end -= 1;
            }
            let pad = inner.len() - end;
            if pad > MAX_PAD {
                return None;
            }
            sc.seg_pads.push(pad as u16);
            if (pad as u64) > max_pad {
                max_pad = pad as u64;
            }
            let content = &inner[..end];
            if !content.is_empty() {
                let mut seg_count = 0usize;
                let mut start = 0usize;
                loop {
                    let is_last;
                    let seg = if start + 1 < content.len() {
                        if let Some(rel) = find_byte(b';', &content[start..content.len() - 1]) {
                            let pos = start + rel;
                            if content[pos + 1] != b' ' {
                                return None;
                            }
                            let s = &content[start..pos];
                            start = pos + 2;
                            is_last = false;
                            s
                        } else {
                            is_last = true;
                            &content[start..]
                        }
                    } else {
                        is_last = true;
                        &content[start..]
                    };
                    if seg.is_empty() || seg.len() > MAX_SEGMENT_LEN {
                        return None;
                    }
                    seg_count += 1;
                    if seg_count > MAX_SEGMENTS_PER_VALUE {
                        return None;
                    }
                    if seg[0].is_ascii_digit() {
                        if let Some(dec) = canonical_segment_decimal(seg) {
                            sc.seg_tokens.push(0xff);
                            sc.seg_decimals.push(dec);
                            if dec > max_decimal {
                                max_decimal = dec;
                            }
                            if is_last {
                                break;
                            }
                            continue;
                        }
                    }
                    let h = hash_slice(seg);
                    let mut pos = (h as usize) & HT_MASK;
                    let id = loop {
                        let slot = ht_slots[pos];
                        if slot == u8::MAX {
                            if dict_len >= MAX_SEGMENT_DICT || !validate_segment_slice(seg) {
                                return None;
                            }
                            let id = dict_len as u8;
                            dict_slices[dict_len].write(seg);
                            dict_len += 1;
                            dict_bytes_sum += 1 + seg.len();
                            ht_hashes[pos].write(h);
                            ht_slots[pos] = id;
                            break id;
                        }
                        let slot_hash = unsafe { ht_hashes[pos].assume_init() };
                        let slot_slice = unsafe { dict_slices[slot as usize].assume_init() };
                        if slot_hash == h && slice_eq_fast(slot_slice, seg) {
                            break slot;
                        }
                        pos = (pos + 1) & HT_MASK;
                    };
                    sc.seg_tokens.push(id);
                    if is_last {
                        break;
                    }
                }
            }
            sc.seg_tokens.push(0xfe);

            // Exact lower bound check on final SegmentDict size:
            let min_tokens = sc.seg_tokens.len() + (n - 1 - row_idx);
            let min_w = bit_width((dict_len + 1) as u64) as usize;
            let dec_w = bit_width(max_decimal) as usize;
            if 6 + dict_bytes_sum
                + (min_tokens * min_w + 7) / 8
                + (sc.seg_decimals.len() * dec_w + 7) / 8
                >= limit
            {
                return None;
            }
        }

        let tok_len = sc.seg_tokens.len();
        let dec_len = sc.seg_decimals.len();

        let width = bit_width((dict_len + 1) as u64);
        let decimal_width = bit_width(max_decimal);
        let pad_width = bit_width(max_pad);
        let size = 1
            + uleb_len(dict_len)
            + dict_bytes_sum
            + 1
            + uleb_len(tok_len)
            + bit_bytes(tok_len, width).ok()?
            + uleb_len(dec_len)
            + 1
            + bit_bytes(dec_len, decimal_width).ok()?
            + 1
            + bit_bytes(n, pad_width).ok()?;
        if size >= limit {
            return None;
        }

        let dict_init: &[&'a [u8]] = unsafe {
            std::slice::from_raw_parts(dict_slices.as_ptr().cast::<&'a [u8]>(), dict_len)
        };
        let mut order: [u8; MAX_SEGMENT_DICT] = core::array::from_fn(|i| i as u8);
        let order_slice = &mut order[..dict_len];
        order_slice.sort_unstable_by(|&a, &b| dict_init[a as usize].cmp(dict_init[b as usize]));
        let mut token_map = [0u64; 256];
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_SEGMENT_DICT);
        put_uleb(&mut out, dict_len);
        for (rank, &orig_id) in order_slice.iter().enumerate() {
            token_map[orig_id as usize] = rank as u64;
            put_bytes(&mut out, dict_init[orig_id as usize]);
        }
        token_map[0xfe] = dict_len as u64;
        token_map[0xff] = dict_len as u64 + 1;

        out.push(width);
        put_uleb(&mut out, tok_len);
        push_bit_values(
            &mut out,
            tok_len,
            width,
            sc.seg_tokens
                .iter()
                .map(|&tok| unsafe { *token_map.get_unchecked(tok as usize) }),
        );
        put_uleb(&mut out, dec_len);
        out.push(decimal_width);
        push_bit_values(
            &mut out,
            dec_len,
            decimal_width,
            sc.seg_decimals.iter().copied(),
        );
        out.push(pad_width);
        push_bit_values(
            &mut out,
            n,
            pad_width,
            sc.seg_pads.iter().map(|&p| u64::from(p)),
        );
        Some(out)
    })
}

#[inline(always)]
fn split_decimal_suffix(value: &[u8]) -> Option<(&[u8], u64)> {
    if value.len() < 4 || value.first() != Some(&b'"') || value.last() != Some(&b'"') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    let mut start = inner.len();
    while start != 0 && inner[start - 1].is_ascii_digit() {
        start -= 1;
    }
    let decimal = parse_decimal(&inner[start..])?;
    Some((&inner[..start], decimal))
}

#[inline(never)]
fn pack_decimal_suffix_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 {
        return None;
    }
    let v0 = values[0];
    let (prefix, first_val) = split_decimal_suffix(v0)?;
    if 1 + uleb_len(prefix.len()) + prefix.len() + 8 + 1 >= limit {
        return None;
    }
    let prefix_end = 1 + prefix.len();
    if prefix_end > 1 && v0[prefix_end - 1].is_ascii_digit() {
        return None;
    }
    let quoted_prefix = &v0[..prefix_end];
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.parsed_u64.clear();
        sc.parsed_u64.reserve(n);
        let parsed_ptr = sc.parsed_u64.as_mut_ptr();
        unsafe {
            *parsed_ptr = first_val;
        }
        let mut min = first_val;
        let mut max = first_val;
        if (8..=16).contains(&prefix_end) {
            let qp = quoted_prefix.as_ptr();
            let q0 = unsafe { (qp as *const u64).read_unaligned() };
            let q1 = unsafe { (qp.add(prefix_end - 8) as *const u64).read_unaligned() };
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2 || unsafe { *value.get_unchecked(val_len - 1) } != b'"' {
                    return None;
                }
                let vp = value.as_ptr();
                let w0 = unsafe { (vp as *const u64).read_unaligned() };
                let w1 = unsafe { (vp.add(prefix_end - 8) as *const u64).read_unaligned() };
                if (w0 ^ q0) | (w1 ^ q1) != 0 {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        } else if (32..=48).contains(&prefix_end) {
            let qp = quoted_prefix.as_ptr();
            let q0 = unsafe { (qp as *const u128).read_unaligned() };
            let q1 = unsafe { (qp.add(16) as *const u128).read_unaligned() };
            let q2 = unsafe { (qp.add(prefix_end - 16) as *const u128).read_unaligned() };
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2 || unsafe { *value.get_unchecked(val_len - 1) } != b'"' {
                    return None;
                }
                let vp = value.as_ptr();
                let w0 = unsafe { (vp as *const u128).read_unaligned() };
                let w1 = unsafe { (vp.add(16) as *const u128).read_unaligned() };
                let w2 = unsafe { (vp.add(prefix_end - 16) as *const u128).read_unaligned() };
                if (w0 ^ q0) | (w1 ^ q1) | (w2 ^ q2) != 0 {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        } else {
            for (idx, &value) in values.iter().enumerate().skip(1) {
                let val_len = value.len();
                if val_len < prefix_end + 2
                    || unsafe { *value.get_unchecked(val_len - 1) } != b'"'
                    || !slice_eq_fast(unsafe { value.get_unchecked(..prefix_end) }, quoted_prefix)
                {
                    return None;
                }
                let num = parse_decimal(unsafe { value.get_unchecked(prefix_end..val_len - 1) })?;
                if num < min {
                    min = num;
                }
                if num > max {
                    max = num;
                }
                unsafe {
                    *parsed_ptr.add(idx) = num;
                }
            }
        }
        unsafe {
            sc.parsed_u64.set_len(n);
        }
        let width = bit_width(max.checked_sub(min)?);
        let size = 1
            + uleb_len(prefix.len())
            + prefix.len()
            + 8
            + 1
            + bit_bytes(n, width).ok()?;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_DECIMAL_SUFFIX);
        put_bytes(&mut out, prefix);
        out.extend_from_slice(&min.to_le_bytes());
        out.push(width);
        push_bit_values(
            &mut out,
            n,
            width,
            sc.parsed_u64.iter().map(|&v| v - min),
        );
        Some(out)
    })
}

#[inline(always)]
fn plain_quoted(value: &[u8]) -> Option<&[u8]> {
    if value.len() < 2 || value.first() != Some(&b'"') || value.last() != Some(&b'"') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    if inner.iter().any(|&byte| byte == b'\\' || byte < 0x20) {
        return None;
    }
    Some(inner)
}

#[inline(always)]
fn periodic_at(content: &[u8], period: usize) -> bool {
    let len = content.len();
    len <= period
        || (content[period] == content[0]
            && content[len - 1] == content[len - 1 - period]
            && slice_eq_fast(&content[period..], &content[..len - period]))
}

fn pack_periodic_string_model<'a>(values: &[&'a [u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 2 + n >= limit {
        return None;
    }
    let v0 = values[0];
    let v1 = values[1];
    let v2 = values[n.min(3) - 1];
    if v0.len() < 2
        || v0[0] != b'"'
        || v0[v0.len() - 1] != b'"'
        || v1.len() < 2
        || v1[0] != b'"'
        || v1[v1.len() - 1] != b'"'
        || v2.len() < 2
        || v2[0] != b'"'
        || v2[v2.len() - 1] != b'"'
    {
        return None;
    }
    let inner0 = &v0[1..v0.len() - 1];
    let inner1 = &v1[1..v1.len() - 1];
    if !inner0.is_empty() && !inner1.is_empty() && 2 + 2 * n >= limit {
        return None;
    }
    let mut watch_inner: &'a [u8] = &v2[1..v2.len() - 1];
    let max_period = inner0
        .len()
        .max(inner1.len())
        .max(watch_inner.len())
        .saturating_sub(1)
        .min(64);

    // Since `size(period)` is monotonically non-decreasing for `period in 1..=64`,
    // the first valid `period` is guaranteed to be optimal, and if `size + rem >= limit`
    // at any row, no larger `period` can ever fit under `limit`!
    for period in 1..=max_period {
        if !periodic_at(inner0, period)
            || !periodic_at(inner1, period)
            || !periodic_at(watch_inner, period)
        {
            continue;
        }
        let mut size = 1 + uleb_len(period);
        let mut max_len = 0usize;
        let mut all_periodic = true;
        for (idx, &v) in values.iter().enumerate() {
            if v.len() < 2 || v[0] != b'"' || v[v.len() - 1] != b'"' {
                return None;
            }
            let content = &v[1..v.len() - 1];
            let len = content.len();
            if len > max_len {
                max_len = len;
            }
            size += uleb_len(len) + len.min(period);
            if size + (n - 1 - idx) >= limit {
                return None;
            }
            if !periodic_at(content, period) {
                watch_inner = content;
                all_periodic = false;
                break;
            }
        }
        if !all_periodic {
            continue;
        }
        if max_len == 0 || period > max_len || size >= limit {
            return None;
        }
        for &v in values {
            plain_quoted(v)?;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_PERIODIC_STRING);
        put_uleb(&mut out, period);
        for &v in values {
            let content = &v[1..v.len() - 1];
            put_uleb(&mut out, content.len());
            out.extend_from_slice(&content[..content.len().min(period)]);
        }
        return Some(out);
    }
    None
}

fn pack_nibbles(out: &mut Vec<u8>, nibbles: &[u8]) -> Option<()> {
    for pair in nibbles.chunks(2) {
        let high = hex_nibble(pair[0])?;
        let low = match pair.get(1) {
            Some(byte) => hex_nibble(*byte)?,
            None => 0,
        };
        out.push(high << 4 | low);
    }
    Some(())
}

fn pack_periodic_hex_model<'a>(values: &[&'a [u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 2 + n >= limit {
        return None;
    }
    let v0 = values[0];
    let v1 = values[1];
    let v2 = values[n.min(3) - 1];
    if v0.len() < 2
        || v0[0] != b'"'
        || v0[v0.len() - 1] != b'"'
        || v1.len() < 2
        || v1[0] != b'"'
        || v1[v1.len() - 1] != b'"'
        || v2.len() < 2
        || v2[0] != b'"'
        || v2[v2.len() - 1] != b'"'
    {
        return None;
    }
    let inner0 = &v0[1..v0.len() - 1];
    let inner1 = &v1[1..v1.len() - 1];
    if !inner0.is_empty() && !inner1.is_empty() && 2 + 2 * n >= limit {
        return None;
    }
    let mut watch_inner: &'a [u8] = &v2[1..v2.len() - 1];
    if (!inner0.is_empty() && hex_nibble(inner0[0]).is_none())
        || (!inner1.is_empty() && hex_nibble(inner1[0]).is_none())
        || (!watch_inner.is_empty() && hex_nibble(watch_inner[0]).is_none())
    {
        return None;
    }
    let max_period = inner0
        .len()
        .max(inner1.len())
        .max(watch_inner.len())
        .saturating_sub(1)
        .min(64);

    for period in 1..=max_period {
        if !periodic_at(inner0, period)
            || !periodic_at(inner1, period)
            || !periodic_at(watch_inner, period)
        {
            continue;
        }
        let mut size = 1 + uleb_len(period);
        let mut max_len = 0usize;
        let mut all_periodic = true;
        for (idx, &v) in values.iter().enumerate() {
            if v.len() < 2 || v[0] != b'"' || v[v.len() - 1] != b'"' {
                return None;
            }
            let content = &v[1..v.len() - 1];
            let len = content.len();
            if len > max_len {
                max_len = len;
            }
            size += uleb_len(len) + len.min(period).div_ceil(2);
            if size + (n - 1 - idx) >= limit {
                return None;
            }
            if !periodic_at(content, period) {
                watch_inner = content;
                all_periodic = false;
                break;
            }
        }
        if !all_periodic {
            continue;
        }
        if max_len == 0 || period > max_len || size >= limit {
            return None;
        }
        for &v in values {
            let content = plain_quoted(v)?;
            if !content[..content.len().min(period)]
                .iter()
                .all(|&byte| hex_nibble(byte).is_some())
            {
                return None;
            }
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_PERIODIC_HEX);
        put_uleb(&mut out, period);
        for &v in values {
            let content = &v[1..v.len() - 1];
            put_uleb(&mut out, content.len());
            pack_nibbles(&mut out, &content[..content.len().min(period)])?;
        }
        return Some(out);
    }
    None
}

const GF2_TRAIN_TRANSITIONS: usize = 96;
const GF2_COEFFICIENTS: usize = 65;

#[inline(always)]
fn apply_affine(coefficients: &[u64; GF2_COEFFICIENTS], input: u64) -> u64 {
    let mut output = coefficients[64];
    let mut w = input;
    while w != 0 {
        let bit = w.trailing_zeros() as usize;
        output ^= unsafe { *coefficients.get_unchecked(bit) };
        w &= w - 1;
    }
    output
}

#[inline(always)]
fn build_gf2_nibble_lut(coefficients: &[u64; GF2_COEFFICIENTS]) -> [[u64; 16]; 16] {
    let mut lut = [[0u64; 16]; 16];
    for g in 0..16 {
        let base = if g == 0 { coefficients[64] } else { 0 };
        let c0 = coefficients[g * 4];
        let c1 = coefficients[g * 4 + 1];
        let c2 = coefficients[g * 4 + 2];
        let c3 = coefficients[g * 4 + 3];
        let t0 = base;
        let t1 = t0 ^ c0;
        let t2 = t0 ^ c1;
        let t3 = t1 ^ c1;
        let t4 = t0 ^ c2;
        let t5 = t1 ^ c2;
        let t6 = t2 ^ c2;
        let t7 = t3 ^ c2;
        lut[g] = [
            t0,
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t0 ^ c3,
            t1 ^ c3,
            t2 ^ c3,
            t3 ^ c3,
            t4 ^ c3,
            t5 ^ c3,
            t6 ^ c3,
            t7 ^ c3,
        ];
    }
    lut
}

#[inline(always)]
fn apply_affine_lut(lut: &[[u64; 16]; 16], w: u64) -> u64 {
    lut[0][(w & 15) as usize]
        ^ lut[1][((w >> 4) & 15) as usize]
        ^ lut[2][((w >> 8) & 15) as usize]
        ^ lut[3][((w >> 12) & 15) as usize]
        ^ lut[4][((w >> 16) & 15) as usize]
        ^ lut[5][((w >> 20) & 15) as usize]
        ^ lut[6][((w >> 24) & 15) as usize]
        ^ lut[7][((w >> 28) & 15) as usize]
        ^ lut[8][((w >> 32) & 15) as usize]
        ^ lut[9][((w >> 36) & 15) as usize]
        ^ lut[10][((w >> 40) & 15) as usize]
        ^ lut[11][((w >> 44) & 15) as usize]
        ^ lut[12][((w >> 48) & 15) as usize]
        ^ lut[13][((w >> 52) & 15) as usize]
        ^ lut[14][((w >> 56) & 15) as usize]
        ^ lut[15][(w >> 60) as usize]
}

#[inline(always)]
fn solve_gf2_basis(train: &[u64]) -> Option<[u64; GF2_COEFFICIENTS]> {
    let x0 = train[0];
    let y0 = train[1];
    let mut basis_mask = [0u64; 64];
    let mut basis_out = [0u64; 64];
    let mut rank = 0usize;
    for pair in train.windows(2).skip(1).take(GF2_TRAIN_TRANSITIONS - 1) {
        let mut mask = pair[0] ^ x0;
        let mut output = pair[1] ^ y0;
        let mut inserted = false;
        while mask != 0 {
            let pivot = 63 - mask.leading_zeros() as usize;
            let pm = unsafe { *basis_mask.get_unchecked(pivot) };
            if pm != 0 {
                mask ^= pm;
                output ^= unsafe { *basis_out.get_unchecked(pivot) };
            } else {
                unsafe {
                    *basis_mask.get_unchecked_mut(pivot) = mask;
                    *basis_out.get_unchecked_mut(pivot) = output;
                }
                rank += 1;
                inserted = true;
                break;
            }
        }
        if !inserted && output != 0 {
            return None;
        }
    }
    if rank != 64 {
        return None;
    }
    let mut coefficients = [0u64; GF2_COEFFICIENTS];
    for bit in 0..64 {
        let mut val = basis_out[bit];
        let mut rem = basis_mask[bit] ^ (1u64 << bit);
        while rem != 0 {
            let k = rem.trailing_zeros() as usize;
            val ^= unsafe { *coefficients.get_unchecked(k) };
            rem &= rem - 1;
        }
        coefficients[bit] = val;
    }
    let mut val64 = y0;
    let mut rem64 = x0;
    while rem64 != 0 {
        let k = rem64.trailing_zeros() as usize;
        val64 ^= unsafe { *coefficients.get_unchecked(k) };
        rem64 &= rem64 - 1;
    }
    coefficients[64] = val64;
    Some(coefficients)
}

/// Learn one affine linear transform over GF(2) from a fixed prefix, then
/// require it to predict a nonempty suffix exactly. This is a model class over
/// arbitrary 64-bit values; it stores the full 65-word transform and has no
/// source, field-name, seed or fixed-shift dependency.
#[allow(dead_code)]
fn learn_affine_recurrence(values: &[u64]) -> Option<[u64; GF2_COEFFICIENTS]> {
    if values.len() <= GF2_TRAIN_TRANSITIONS + 1 {
        return None;
    }
    let coefficients = solve_gf2_basis(&values[..GF2_TRAIN_TRANSITIONS + 1])?;
    let lut = build_gf2_nibble_lut(&coefficients);
    if values
        .windows(2)
        .skip(GF2_TRAIN_TRANSITIONS)
        .any(|pair| apply_affine_lut(&lut, pair[0]) != pair[1])
    {
        return None;
    }
    Some(coefficients)
}

fn pack_gf2_affine_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    // The transform plus the fixed prefix absolutely cost more than this
    // bound, so a hopeless column never pays for the parse or the solver.
    if values.len() <= GF2_TRAIN_TRANSITIONS + 1 {
        return None;
    }
    if 1 + uleb_len(0) + 8 + 8 * GF2_COEFFICIENTS >= limit {
        return None;
    }
    // A deterministic GF(2) recurrence with a duplicate in `values[..8]` enters a cycle
    // of length <= 7, so its transition differences span at most 13 dimensions (< 64)
    // and `solve_gf2_basis` is guaranteed to return `None`.
    for i in 1..8 {
        for j in 0..i {
            if slice_eq_fast(values[i], values[j]) {
                return None;
            }
        }
    }
    let (prefix, first_digits) = split_hex_suffix(values[0])?;
    if first_digits.len() != 16 {
        return None;
    }
    let size = 1 + uleb_len(prefix.len()) + prefix.len() + 8 + 8 * GF2_COEFFICIENTS;
    if size >= limit {
        return None;
    }
    let expected_val_len = 2 + prefix.len() + 16;
    let prefix_end = 1 + prefix.len();
    let mut train = [0u64; GF2_TRAIN_TRANSITIONS + 1];
    for (i, &value) in values[..GF2_TRAIN_TRANSITIONS + 1].iter().enumerate() {
        if value.len() != expected_val_len
            || value[0] != b'"'
            || value[expected_val_len - 1] != b'"'
            || !slice_eq_fast(&value[1..prefix_end], prefix)
        {
            return None;
        }
        train[i] = parse_hex_u64_le(&value[prefix_end..expected_val_len - 1])?;
    }
    let coefficients = solve_gf2_basis(&train)?;
    let lut = build_gf2_nibble_lut(&coefficients);
    let mut prev = train[GF2_TRAIN_TRANSITIONS];
    for &value in &values[GF2_TRAIN_TRANSITIONS + 1..] {
        if value.len() != expected_val_len
            || value[0] != b'"'
            || value[expected_val_len - 1] != b'"'
            || !slice_eq_fast(&value[1..prefix_end], prefix)
        {
            return None;
        }
        let word = parse_hex_u64_le(&value[prefix_end..expected_val_len - 1])?;
        if apply_affine_lut(&lut, prev) != word {
            return None;
        }
        prev = word;
    }
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_GF2_AFFINE);
    put_bytes(&mut out, prefix);
    out.extend_from_slice(&train[0].to_le_bytes());
    for coefficient in coefficients {
        out.extend_from_slice(&coefficient.to_le_bytes());
    }
    Some(out)
}

#[inline(always)]
fn digits(value: &[u8]) -> Option<u16> {
    if value.len() != 2 {
        return None;
    }
    let d0 = value[0].wrapping_sub(b'0');
    let d1 = value[1].wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 {
        return None;
    }
    Some(u16::from(d0) * 10 + u16::from(d1))
}

#[inline(always)]
fn parse_timestamp(value: &[u8]) -> Option<Timestamp> {
    if value.len() != 22
        || value[0] != b'"'
        || value[21] != b'"'
        || value[5] != b'-'
        || value[8] != b'-'
        || value[11] != b'T'
        || value[14] != b':'
        || value[17] != b':'
        || value[20] != b'Z'
    {
        return None;
    }
    let y0 = value[1].wrapping_sub(b'0');
    let y1 = value[2].wrapping_sub(b'0');
    let y2 = value[3].wrapping_sub(b'0');
    let y3 = value[4].wrapping_sub(b'0');
    if y0 > 9 || y1 > 9 || y2 > 9 || y3 > 9 {
        return None;
    }
    let year =
        u16::from(y0) * 1000 + u16::from(y1) * 100 + u16::from(y2) * 10 + u16::from(y3);
    let month = digits(&value[6..8])? as u8;
    let day = digits(&value[9..11])? as u8;
    let hour = digits(&value[12..14])? as u8;
    let minute = digits(&value[15..17])? as u8;
    let second = digits(&value[18..20])? as u8;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour >= 24
        || minute >= 60
        || second >= 60
    {
        return None;
    }
    Some(Timestamp {
        year,
        month,
        day,
        hour,
        minute,
        second,
    })
}

#[inline(never)]
fn pack_timestamp_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 13 >= limit {
        return None;
    }
    let v0 = values[0];
    let first = parse_timestamp(v0)?;
    let year = first.year;
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.parsed_u64.clear();
        sc.parsed_u64.reserve(n);
        let parsed_ptr = sc.parsed_u64.as_mut_ptr();
        let pack_ts = |ts: Timestamp| -> u64 {
            (ts.month as u64)
                | ((ts.day as u64) << 8)
                | ((ts.hour as u64) << 16)
                | ((ts.minute as u64) << 24)
                | ((ts.second as u64) << 32)
        };
        let mut prev_w0 = unsafe { v0.as_ptr().cast::<u64>().read_unaligned() };
        let first_w0_low48 = prev_w0 & 0x0000_ffff_ffff_ffff;
        let mut prev_w6 = unsafe { v0.as_ptr().add(6).cast::<u64>().read_unaligned() };
        let mut prev_w14 = unsafe { v0.as_ptr().add(14).cast::<u64>().read_unaligned() };
        let mut prev_packed = pack_ts(first);
        unsafe {
            *parsed_ptr = prev_packed;
        }
        #[cfg(target_arch = "aarch64")]
        let (mins, maxs) = unsafe {
            use core::arch::aarch64::*;
            let mut min_v = vcreate_u8(prev_packed);
            let mut max_v = min_v;
            for (idx, &val) in values.iter().enumerate().skip(1) {
                if val.len() != 22 {
                    return None;
                }
                let ptr = val.as_ptr();
                let w0 = ptr.cast::<u64>().read_unaligned();
                let w6 = ptr.add(6).cast::<u64>().read_unaligned();
                let w14 = ptr.add(14).cast::<u64>().read_unaligned();
                if w6 == prev_w6 && w0 == prev_w0 {
                    if w14 == prev_w14 {
                        *parsed_ptr.add(idx) = prev_packed;
                        continue;
                    }
                    let d14 = w14 ^ 0x225a_3030_3a30_303a;
                    if ((d14.wrapping_add(0x0000_0606_0006_0600)) | d14) & 0xffff_f0f0_fff0_f0ff
                        != 0
                    {
                        return None;
                    }
                    let p14 = d14.wrapping_mul(10).wrapping_add(d14 >> 8);
                    let minute = (p14 >> 8) & 0xff;
                    let second = (p14 >> 32) & 0xff;
                    if minute >= 60 || second >= 60 {
                        return None;
                    }
                    prev_w14 = w14;
                    prev_packed = (prev_packed & 0x00ff_ffff) | (minute << 24) | (second << 32);
                    let pv = vcreate_u8(prev_packed);
                    min_v = vmin_u8(min_v, pv);
                    max_v = vmax_u8(max_v, pv);
                    *parsed_ptr.add(idx) = prev_packed;
                    continue;
                }
                if (w0 & 0x0000_ffff_ffff_ffff) != first_w0_low48 {
                    return None;
                }
                let d6 = w6 ^ 0x3030_5430_302d_3030;
                let d14 = w14 ^ 0x225a_3030_3a30_303a;
                if (((d6.wrapping_add(0x0606_0006_0600_0606)) | d6) & 0xf0f0_fff0_f0ff_f0f0)
                    | (((d14.wrapping_add(0x0000_0606_0006_0600)) | d14) & 0xffff_f0f0_fff0_f0ff)
                    != 0
                {
                    return None;
                }
                let p6 = d6.wrapping_mul(10).wrapping_add(d6 >> 8);
                let p14 = d14.wrapping_mul(10).wrapping_add(d14 >> 8);
                let month = p6 & 0xff;
                let day = (p6 >> 24) & 0xff;
                let hour = (p6 >> 48) & 0xff;
                let minute = (p14 >> 8) & 0xff;
                let second = (p14 >> 32) & 0xff;
                if month.wrapping_sub(1) >= 12
                    || day.wrapping_sub(1) >= 31
                    || hour >= 24
                    || minute >= 60
                    || second >= 60
                {
                    return None;
                }
                prev_w0 = w0;
                prev_w6 = w6;
                prev_w14 = w14;
                prev_packed =
                    month | (day << 8) | (hour << 16) | (minute << 24) | (second << 32);
                let pv = vcreate_u8(prev_packed);
                min_v = vmin_u8(min_v, pv);
                max_v = vmax_u8(max_v, pv);
                *parsed_ptr.add(idx) = prev_packed;
            }
            let min_b = vget_lane_u64(vreinterpret_u64_u8(min_v), 0).to_le_bytes();
            let max_b = vget_lane_u64(vreinterpret_u64_u8(max_v), 0).to_le_bytes();
            (
                [min_b[0], min_b[1], min_b[2], min_b[3], min_b[4]],
                [max_b[0], max_b[1], max_b[2], max_b[3], max_b[4]],
            )
        };
        #[cfg(not(target_arch = "aarch64"))]
        let (mins, maxs) = {
            let mut mins = [first.month, first.day, first.hour, first.minute, first.second];
            let mut maxs = mins;
            for (idx, &val) in values.iter().enumerate().skip(1) {
                if val.len() != 22 {
                    return None;
                }
                let ptr = val.as_ptr();
                let w0 = unsafe { ptr.cast::<u64>().read_unaligned() };
                let w6 = unsafe { ptr.add(6).cast::<u64>().read_unaligned() };
                let w14 = unsafe { ptr.add(14).cast::<u64>().read_unaligned() };
                if w0 == prev_w0 && w6 == prev_w6 && w14 == prev_w14 {
                    unsafe {
                        *parsed_ptr.add(idx) = prev_packed;
                    }
                    continue;
                }
                let ts = parse_timestamp(val)?;
                if ts.year != year {
                    return None;
                }
                let c = [ts.month, ts.day, ts.hour, ts.minute, ts.second];
                for i in 0..5 {
                    if c[i] < mins[i] {
                        mins[i] = c[i];
                    }
                    if c[i] > maxs[i] {
                        maxs[i] = c[i];
                    }
                }
                prev_w0 = w0;
                prev_w6 = w6;
                prev_w14 = w14;
                prev_packed = pack_ts(ts);
                unsafe {
                    *parsed_ptr.add(idx) = prev_packed;
                }
            }
            (mins, maxs)
        };
        unsafe {
            sc.parsed_u64.set_len(n);
        }
        let mut plans = [(0u8, 0u8); 5];
        let mut bit_count = 0usize;
        for i in 0..5 {
            let w = bit_width((maxs[i] - mins[i]) as u64);
            plans[i] = (mins[i], w);
            bit_count += bit_bytes(n, w).ok()?;
        }
        let size = 1 + 2 + 10 + bit_count;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_TIMESTAMP);
        out.extend_from_slice(&year.to_le_bytes());
        for (base, width) in &plans {
            out.extend_from_slice(&[*base, *width]);
        }
        for (comp_idx, &(base, width)) in plans.iter().enumerate() {
            if width != 0 {
                let shift = (comp_idx * 8) as u32;
                let base_u64 = base as u64;
                push_bit_values(
                    &mut out,
                    n,
                    width,
                    sc.parsed_u64
                        .iter()
                        .map(|&w| ((w >> shift) & 0xff) - base_u64),
                );
            }
        }
        Some(out)
    })
}

/// Canonical tie-breaking priority rank from the specification:
/// earlier model in the race wins ties (`candidate.len() < best.len()`).
#[inline(always)]
const fn canonical_model_rank(kind: ModelKind) -> u8 {
    match kind {
        ModelKind::Raw => 0,
        ModelKind::Dictionary => 1,
        ModelKind::DecimalAffine => 2,
        ModelKind::DecimalPacked => 3,
        ModelKind::HexSuffix => 4,
        ModelKind::DecimalSuffix => 5,
        ModelKind::PeriodicString => 6,
        ModelKind::PeriodicHex => 7,
        ModelKind::Gf2Affine => 8,
        ModelKind::Timestamp => 9,
        ModelKind::EpochCounter => 10,
        ModelKind::SegmentDict => 11,
        ModelKind::DecimalDelta => 12,
    }
}

#[inline(always)]
fn effective_limit(best_len: usize, best_kind: ModelKind, candidate_kind: ModelKind) -> usize {
    // In canonical order, `candidate_kind` beats `best_kind` iff:
    // - `len < best_len` (when `rank(candidate) > rank(best)`), OR
    // - `len <= best_len`, i.e., `len < best_len + 1` (when `rank(candidate) < rank(best)`).
    best_len + usize::from(canonical_model_rank(candidate_kind) < canonical_model_rank(best_kind))
}

fn pack_model(values: &[&[u8]]) -> (Vec<u8>, ModelKind) {
    pack_model_mode(values, false)
}

fn pack_model_v3(values: &[&[u8]]) -> (Vec<u8>, ModelKind) {
    pack_model_mode(values, true)
}

fn pack_model_mode(values: &[&[u8]], allow_v3_ext: bool) -> (Vec<u8>, ModelKind) {
    let n = values.len();
    if n < 2 {
        let raw = pack_raw_model(values);
        if allow_v3_ext && raw.len() >= 128 {
            let mut lz = Vec::with_capacity(raw.len() / 2);
            if pack_fast_lz_into(&raw[1..], &mut lz).is_some() && lz.len() + 8 < raw.len() {
                lz[0] = MODEL_LZ_RAW;
                return (lz, ModelKind::Raw);
            }
        }
        return (raw, ModelKind::Raw);
    }

    let first_slice = values[0];
    let min_raw_size = 1 + n + first_slice.len();
    let mut raw_size_opt: Option<usize> = if n < 256 {
        Some(raw_model_size(values))
    } else {
        None
    };
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_len = raw_size_opt.unwrap_or(usize::MAX);
    let mut best_kind = ModelKind::Raw;

    macro_rules! get_raw_size {
        () => {
            *raw_size_opt.get_or_insert_with(|| raw_model_size(values))
        };
    }

    macro_rules! try_model {
        (|$limit:ident| $call:expr, $kind:expr) => {{
            let $limit = if best_bytes.is_some() {
                effective_limit(best_len, best_kind, $kind)
            } else if let Some(rs) = raw_size_opt {
                rs
            } else {
                usize::MAX
            };
            if let Some(bytes) = $call {
                let len = bytes.len();
                if best_bytes.is_some() || len < min_raw_size || len < get_raw_size!() {
                    best_len = len;
                    best_kind = $kind;
                    best_bytes = Some(bytes);
                }
            }
        }};
    }

    let is_quoted = first_slice.len() >= 2
        && first_slice[0] == b'"'
        && first_slice[first_slice.len() - 1] == b'"';

    if !is_quoted {
        // Non-quoted values cannot match any of the 8 quoted-only models
        // (`HexSuffix`, `DecimalSuffix`, `PeriodicString`, `PeriodicHex`,
        // `Gf2Affine`, `Timestamp`, `EpochCounter`, `SegmentDict`).
        // Check if it's a canonical decimal column in a single pass!
        let mut skip_dictionary = false;
        if let Some(first_num) = parse_decimal(first_slice) {
            if let Some(second_num) = parse_decimal(values[1]) {
                MODEL_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    sc.parsed_u64.clear();
                    sc.parsed_u64.reserve(n);
                    let parsed_ptr = sc.parsed_u64.as_mut_ptr();
                    unsafe {
                        *parsed_ptr = first_num;
                        *parsed_ptr.add(1) = second_num;
                    }
                    let mut all_decimal = true;
                    let mut min = first_num.min(second_num);
                    let mut max = first_num.max(second_num);
                    let step128 = second_num as i128 - first_num as i128;
                    let mut step64 = 0i64;
                    let mut affine_ok = true;
                    let mut delta_ok = true;
                    let mut max_zigzag = 0u64;
                    if let Ok(s64) = i64::try_from(step128) {
                        step64 = s64;
                        max_zigzag = ((s64 << 1) ^ (s64 >> 63)) as u64;
                    } else {
                        affine_ok = false;
                        delta_ok = false;
                    }
                    let mut prev = second_num;

                    for (idx, &val) in values.iter().enumerate().skip(2) {
                        if let Some(num) = parse_decimal(val) {
                            unsafe {
                                *parsed_ptr.add(idx) = num;
                            }
                            if num < min {
                                min = num;
                            }
                            if num > max {
                                max = num;
                            }
                            if ((num | prev) >> 63) == 0 {
                                let delta = (num as i64).wrapping_sub(prev as i64);
                                if delta != step64 {
                                    affine_ok = false;
                                    if delta_ok {
                                        let zz = ((delta << 1) ^ (delta >> 63)) as u64;
                                        if zz > max_zigzag {
                                            max_zigzag = zz;
                                        }
                                    }
                                }
                            } else {
                                let diff128 = num as i128 - prev as i128;
                                if diff128 != step128 {
                                    affine_ok = false;
                                }
                                if delta_ok {
                                    if let Ok(delta) = i64::try_from(diff128) {
                                        let zz = ((delta << 1) ^ (delta >> 63)) as u64;
                                        if zz > max_zigzag {
                                            max_zigzag = zz;
                                        }
                                    } else {
                                        delta_ok = false;
                                    }
                                }
                            }
                            prev = num;
                        } else {
                            all_decimal = false;
                            break;
                        }
                    }

                    if all_decimal {
                        unsafe {
                            sc.parsed_u64.set_len(n);
                        }
                        if affine_ok {
                            if step128 != 0 {
                                // Arithmetic progression with non-zero step has `n` distinct values (`u == n`),
                                // so `Dictionary` is guaranteed to return `None`.
                                skip_dictionary = true;
                            }
                            let lim = if best_bytes.is_some() {
                                effective_limit(best_len, best_kind, ModelKind::DecimalAffine)
                            } else if 17 < min_raw_size {
                                usize::MAX
                            } else {
                                get_raw_size!()
                            };
                            if 17 < lim {
                                let step = step64;
                                let mut out = Vec::with_capacity(17);
                                out.push(MODEL_DECIMAL_AFFINE);
                                out.extend_from_slice(&first_num.to_le_bytes());
                                out.extend_from_slice(&step.to_le_bytes());
                                best_len = 17;
                                best_kind = ModelKind::DecimalAffine;
                                best_bytes = Some(out);
                            }
                        }
                        // When `affine_ok && step128 != 0 && n >= 49`, `DecimalAffine` (17 bytes, rank 2)
                        // cannot be beaten by `DecimalPacked` (`>= 17` bytes, rank 3) or `DecimalDelta` (`>= 17` bytes, rank 12).
                        if !(affine_ok && step128 != 0 && n >= 49 && best_kind == ModelKind::DecimalAffine) {
                            let packed_width = bit_width(max - min);
                            if let Ok(packed_bits) = bit_bytes(n, packed_width) {
                                let packed_size = 10 + packed_bits;
                                let delta_width = bit_width(max_zigzag);
                                let delta_size = if delta_ok {
                                    bit_bytes(n - 1, delta_width).ok().map(|b| 10 + b)
                                } else {
                                    None
                                };
                                if delta_size.is_some_and(|ds| ds < packed_size) {
                                    let ds = delta_size.unwrap();
                                    let lim = if best_bytes.is_some() {
                                        effective_limit(best_len, best_kind, ModelKind::DecimalDelta)
                                    } else if ds < min_raw_size {
                                        usize::MAX
                                    } else {
                                        get_raw_size!()
                                    };
                                    if ds < lim {
                                        let mut out = Vec::with_capacity(ds);
                                        out.push(MODEL_DECIMAL_DELTA);
                                        out.extend_from_slice(&first_num.to_le_bytes());
                                        out.push(delta_width);
                                        push_bit_values(
                                            &mut out,
                                            n - 1,
                                            delta_width,
                                            sc.parsed_u64.windows(2).map(|pair| {
                                                let delta = (pair[1] as i128 - pair[0] as i128) as i64;
                                                ((delta << 1) ^ (delta >> 63)) as u64
                                            }),
                                        );
                                        best_len = ds;
                                        best_kind = ModelKind::DecimalDelta;
                                        best_bytes = Some(out);
                                    }
                                } else {
                                    let lim = if best_bytes.is_some() {
                                        effective_limit(best_len, best_kind, ModelKind::DecimalPacked)
                                    } else if packed_size < min_raw_size {
                                        usize::MAX
                                    } else {
                                        get_raw_size!()
                                    };
                                    if packed_size < lim {
                                        let mut out = Vec::with_capacity(packed_size);
                                        out.push(MODEL_DECIMAL_PACKED);
                                        out.extend_from_slice(&min.to_le_bytes());
                                        out.push(packed_width);
                                        push_bit_values(
                                            &mut out,
                                            n,
                                            packed_width,
                                            sc.parsed_u64.iter().map(|&v| v - min),
                                        );
                                        best_len = packed_size;
                                        best_kind = ModelKind::DecimalPacked;
                                        best_bytes = Some(out);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
        // Now check Dictionary with the tight `best_len`:
        if !skip_dictionary {
            try_model!(
                |limit| pack_dictionary_model_mode(values, limit, allow_v3_ext),
                ModelKind::Dictionary
            );
        }
        if allow_v3_ext && best_kind == ModelKind::Raw {
            let raw = pack_raw_model(values);
            if (64..=65536).contains(&(raw.len() - 1)) && raw.len() >= 128 {
                let mut lz = Vec::with_capacity(raw.len());
                if pack_fast_lz_into(&raw[1..], &mut lz).is_some() && lz.len() + 8 < raw.len() {
                    lz[0] = MODEL_LZ_RAW;
                    return (lz, ModelKind::Raw);
                }
            }
            return (raw, best_kind);
        }
        let _ = best_len;
        return (
            best_bytes.unwrap_or_else(|| pack_raw_model(values)),
            best_kind,
        );
    }

    // Quoted value path (`first_slice` starts and ends with `"`):
    let inner0 = &first_slice[1..first_slice.len() - 1];
    let last_inner_byte = inner0.last().copied().unwrap_or(0);

    // 1. Timestamp (only possible when `first_slice.len() == 22` and ends with `Z"`):
    if first_slice.len() == 22 && last_inner_byte == b'Z' {
        try_model!(
            |limit| pack_timestamp_model(values, limit),
            ModelKind::Timestamp
        );
        if best_kind == ModelKind::Timestamp && (allow_v3_ext || best_len <= 25) {
            return (best_bytes.unwrap(), best_kind);
        }
    }

    // 1b. For small v3 columns (`n <= 64`) or when the first two rows are identical (`values[0] == values[1]`,
    // e.g. constant strings like `"eu-central-1"`), run `Dictionary` first so `best_len` is tight before
    // testing `DecimalSuffix` / `HexSuffix` / `SegmentDict`.
    let mut dict_already_tried = false;
    if (allow_v3_ext && n <= 64) || slice_eq_fast(first_slice, values[1]) {
        dict_already_tried = true;
        try_model!(
            |limit| pack_dictionary_model_mode(values, limit, allow_v3_ext),
            ModelKind::Dictionary
        );
        if best_kind == ModelKind::Dictionary
            && best_len <= 3 + uleb_len(first_slice.len()) + first_slice.len()
        {
            return (best_bytes.unwrap(), best_kind);
        }
    }

    // 2. Trailing decimal digit -> `DecimalSuffix`:
    if last_inner_byte.is_ascii_digit() {
        try_model!(
            |limit| pack_decimal_suffix_model(values, limit),
            ModelKind::DecimalSuffix
        );
    }

    // 3. Multi-byte lowercase hex suffix -> `EpochCounter`, `Gf2Affine`, `HexSuffix`:
    let has_hex_tail = hex_nibble(last_inner_byte).is_some();
    let has_multi_hex_tail = has_hex_tail
        && inner0.len() >= 8
        && hex_nibble(inner0[inner0.len() - 2]).is_some()
        && hex_nibble(inner0[inner0.len() - 3]).is_some()
        && hex_nibble(inner0[inner0.len() - 4]).is_some();
    if has_multi_hex_tail {
        let has_early_dup = inner0.len() < 16
            || (1..values.len().min(8)).any(|i| (0..i).any(|j| slice_eq_fast(values[i], values[j])));
        if has_early_dup && allow_v3_ext && !dict_already_tried {
            dict_already_tried = true;
            try_model!(
                |limit| pack_dictionary_model_mode(values, limit, true),
                ModelKind::Dictionary
            );
        }
        if inner0.len() >= 16 {
            try_model!(
                |limit| pack_epoch_counter_model(values, limit),
                ModelKind::EpochCounter
            );
            try_model!(
                |limit| pack_gf2_affine_model(values, limit),
                ModelKind::Gf2Affine
            );
        }
        if !has_early_dup && allow_v3_ext && !dict_already_tried {
            dict_already_tried = true;
            try_model!(
                |limit| pack_dictionary_model_mode(values, limit, true),
                ModelKind::Dictionary
            );
        }
        try_model!(
            |limit| pack_hex_suffix_model(values, limit),
            ModelKind::HexSuffix
        );
    }

    // 4. Every valid SegmentDict value consists exclusively of bytes satisfying `segment_byte_ok(b) || b == b';'`.
    // Also skip SegmentDict when `DecimalSuffix` already matched and `n >= 4 * (first_slice.len() + 8)` with
    // a `"; "` separator before the trailing decimal, where `SegmentDict` is mathematically guaranteed >= `best_len`.
    let dec_suffix_dominates_seg = best_kind == ModelKind::DecimalSuffix
        && n >= 4 * (first_slice.len() + 8)
        && {
            let mut s = inner0.len();
            while s != 0 && inner0[s - 1].is_ascii_digit() {
                s -= 1;
            }
            s == 0 || inner0[..s].ends_with(b"; ")
        };
    let seg_chars_ok = !dec_suffix_dominates_seg
        && (!allow_v3_ext || best_len > 6 + (n + 7) / 8)
        && inner0.iter().all(|&b| segment_byte_ok(b) || b == b';');
    let looks_like_segment_dict =
        seg_chars_ok && (last_inner_byte == b' ' || find_byte(b';', inner0).is_some());
    if looks_like_segment_dict {
        try_model!(
            |limit| pack_segment_dictionary_model(values, limit),
            ModelKind::SegmentDict
        );
    }

    // 5. Dictionary model (skip if already tried or if SegmentDict/DecimalSuffix already succeeded and the first 8 rows are all distinct):
    let skip_dict_after_seg = (best_kind == ModelKind::SegmentDict
        || (best_kind == ModelKind::DecimalSuffix && allow_v3_ext))
        && values.len() >= 8
        && (1..8).all(|i| (0..i).all(|j| !slice_eq_fast(values[i], values[j])));
    if !dict_already_tried && !skip_dict_after_seg {
        try_model!(
            |limit| pack_dictionary_model_mode(values, limit, allow_v3_ext),
            ModelKind::Dictionary
        );
    }

    // 5b. Short hex tail fallback (now aborts in O(1) via `size >= limit` when `Dictionary` won):
    if !allow_v3_ext && has_hex_tail && !has_multi_hex_tail {
        if inner0.len() >= 16 {
            try_model!(
                |limit| pack_epoch_counter_model(values, limit),
                ModelKind::EpochCounter
            );
            try_model!(
                |limit| pack_gf2_affine_model(values, limit),
                ModelKind::Gf2Affine
            );
        }
        try_model!(
            |limit| pack_hex_suffix_model(values, limit),
            ModelKind::HexSuffix
        );
    }

    // 6. SegmentDict fallback when `inner0` had no `;` or trailing space:
    if !looks_like_segment_dict
        && seg_chars_ok
        && (!allow_v3_ext
            || best_kind == ModelKind::Raw
            || values[1..n.min(8)]
                .iter()
                .any(|v| v.len() >= 3 && (v[v.len() - 2] == b' ' || find_byte(b';', v).is_some())))
    {
        try_model!(
            |limit| pack_segment_dictionary_model(values, limit),
            ModelKind::SegmentDict
        );
    }

    // 7. PeriodicString & PeriodicHex:
    if !allow_v3_ext || best_kind == ModelKind::Raw {
        try_model!(
            |limit| pack_periodic_string_model(values, limit),
            ModelKind::PeriodicString
        );
        try_model!(
            |limit| pack_periodic_hex_model(values, limit),
            ModelKind::PeriodicHex
        );
    }

    if allow_v3_ext && best_kind == ModelKind::Raw {
        let raw = pack_raw_model(values);
        if (64..=65536).contains(&(raw.len() - 1)) && raw.len() >= 128 {
            let mut lz = Vec::with_capacity(raw.len());
            if pack_fast_lz_into(&raw[1..], &mut lz).is_some() && lz.len() + 8 < raw.len() {
                lz[0] = MODEL_LZ_RAW;
                return (lz, ModelKind::Raw);
            }
        }
        return (raw, best_kind);
    }

    let _ = best_len;
    (
        best_bytes.unwrap_or_else(|| pack_raw_model(values)),
        best_kind,
    )
}

/// Encode owned values (test/oracle helper): borrows each value before entering the model race.
fn pack_model_values(values: &[Vec<u8>]) -> (Vec<u8>, ModelKind) {
    let refs = values.iter().map(|value| value.as_slice()).collect::<Vec<_>>();
    pack_model(&refs)
}

#[cfg(test)]
fn unpack_model(bytes: &[u8], count: usize) -> Result<Vec<Vec<u8>>> {
    Ok(unpack_model_mode(bytes, count, true)?.materialize())
}

fn unpack_model_mode(bytes: &[u8], count: usize, strict: bool) -> Result<FlatValues> {
    // Value offsets are u32 ranges into this payload, so the whole payload must
    // fit in u32; every push below is then infallible by construction.
    if bytes.len() > u32::MAX as usize {
        return Err(Error::SizeOverflow);
    }
    let mut cursor = Cursor::new(bytes);
    let tag = cursor.byte()?;
    let per_value = match tag {
        MODEL_HEX_SUFFIX | MODEL_EPOCH_COUNTER | MODEL_PERIODIC_HEX | MODEL_PERIODIC_STRING
        | MODEL_GF2_AFFINE => 24,
        MODEL_SEGMENT_DICT => 48,
        _ => 8,
    };
    let mut values = FlatValues::with_capacity(count, count.saturating_mul(per_value));
    match tag {
        MODEL_RAW => {
            for _ in 0..count {
                let len = cursor.uleb()?;
                let value = cursor.take(len)?;
                values.push_slice(value);
            }
        }
        MODEL_LZ_RAW => {
            let mut probe = Cursor::new(cursor.bytes);
            probe.pos = cursor.pos;
            let raw_len = probe.uleb()?;
            if raw_len == 0 || raw_len > 65536 {
                return Err(invalid("column lz raw length"));
            }
            let mut unpacked = vec![0u8; raw_len];
            unpack_fast_lz_into(&mut cursor, &mut unpacked)?;
            let mut sub = Cursor::new(&unpacked);
            for _ in 0..count {
                let len = sub.uleb()?;
                let value = sub.take(len)?;
                values.push_slice(value);
            }
            if !sub.done() {
                return Err(invalid("column lz raw trailing bytes"));
            }
        }
        MODEL_DICTIONARY | MODEL_FRONT_DICT => {
            let entries_len = cursor.uleb()?;
            if entries_len == 0
                || (tag == MODEL_DICTIONARY && entries_len >= count)
                || entries_len > count
            {
                return Err(invalid("column dictionary cardinality"));
            }
            let mut entries = Vec::with_capacity(entries_len);
            if tag == MODEL_FRONT_DICT {
                if entries_len < 2 || entries_len > 256 {
                    return Err(invalid("column front dict cardinality"));
                }
                let mode_byte = cursor.byte()?;
                let dict_mode = mode_byte & 0x03;
                let idx_mode = mode_byte >> 2;
                if idx_mode > 3 {
                    return Err(invalid("column front dict mode"));
                }
                let width = bit_width((entries_len - 1) as u64);
                fn read_front_entries(
                    dcur: &mut Cursor<'_>,
                    entries_len: usize,
                    entries: &mut Vec<Vec<u8>>,
                ) -> Result<()> {
                    let first = read_bytes(dcur)?;
                    entries.push(first);
                    for _ in 1..entries_len {
                        let lcp = dcur.uleb()?;
                        let lcs = dcur.uleb()?;
                        let mid = read_bytes(dcur)?;
                        let prev: &Vec<u8> = entries.last().unwrap();
                        if lcp + lcs > prev.len() {
                            return Err(invalid("column front dictionary bounds"));
                        }
                        let mut entry = Vec::with_capacity(lcp + mid.len() + lcs);
                        entry.extend_from_slice(&prev[..lcp]);
                        entry.extend_from_slice(&mid);
                        entry.extend_from_slice(&prev[prev.len() - lcs..]);
                        if prev >= &entry {
                            return Err(invalid("column dictionary ordering"));
                        }
                        entries.push(entry);
                    }
                    Ok(())
                }
                if dict_mode == 0 {
                    for _ in 0..entries_len {
                        let entry = read_bytes(&mut cursor)?;
                        if entries.last().is_some_and(|last: &Vec<u8>| last >= &entry) {
                            return Err(invalid("column dictionary ordering"));
                        }
                        entries.push(entry);
                    }
                } else if dict_mode == 1 {
                    read_front_entries(&mut cursor, entries_len, &mut entries)?;
                } else if dict_mode == 2 {
                    let lz_len = cursor.uleb()?;
                    let lz_bytes = cursor.take(lz_len)?;
                    let mut lzc = Cursor::new(lz_bytes);
                    if lzc.byte()? != FAST_LZ_PACKET {
                        return Err(invalid("column front dict lz selector"));
                    }
                    let mut probe = Cursor::new(lzc.bytes);
                    probe.pos = lzc.pos;
                    let raw_len = probe.uleb()?;
                    if raw_len == 0 || raw_len > 65536 {
                        return Err(invalid("column front dict lz length"));
                    }
                    let mut unpacked_dict = vec![0u8; raw_len];
                    unpack_fast_lz_into(&mut lzc, &mut unpacked_dict)?;
                    let mut dcur = Cursor::new(&unpacked_dict);
                    read_front_entries(&mut dcur, entries_len, &mut entries)?;
                    if !dcur.done() {
                        return Err(invalid("column front dict lz trailing"));
                    }
                } else {
                    let (pal, pal_len, pal_mixed) = read_front_dict_palette(&mut cursor)?;
                    let first = read_bytes(&mut cursor)?;
                    entries.push(first);
                    for _ in 1..entries_len {
                        let lcp = cursor.uleb()?;
                        let lcs = cursor.uleb()?;
                        let raw_hdr = cursor.uleb()?;
                        let (mid_len, is_raw) = if pal_mixed {
                            (raw_hdr >> 1, (raw_hdr & 1) != 0)
                        } else {
                            (raw_hdr, false)
                        };
                        if mid_len > 65536 {
                            return Err(invalid("column front dict entry length"));
                        }
                        let packed_mid = if is_raw {
                            cursor.take(mid_len)?
                        } else if pal_len <= 16 {
                            cursor.take((mid_len + 1) >> 1)?
                        } else {
                            cursor.take((mid_len * 5 + 7) >> 3)?
                        };
                        let prev: &Vec<u8> = entries.last().unwrap();
                        if lcp + lcs > prev.len() {
                            return Err(invalid("column front dictionary bounds"));
                        }
                        let total_len = lcp + mid_len + lcs;
                        let mut entry = vec![0u8; total_len];
                        entry[..lcp].copy_from_slice(&prev[..lcp]);
                        if is_raw {
                            entry[lcp..lcp + mid_len].copy_from_slice(packed_mid);
                        } else if mid_len > 0 {
                            unpack_nibble_bytes_into(
                                packed_mid,
                                mid_len,
                                &pal,
                                pal_len,
                                unsafe { entry.as_mut_ptr().add(lcp) },
                            )?;
                        }
                        entry[lcp + mid_len..].copy_from_slice(&prev[prev.len() - lcs..]);
                        if prev >= &entry {
                            return Err(invalid("column dictionary ordering"));
                        }
                        entries.push(entry);
                    }
                }
                let mut u8_arena = Vec::with_capacity(count + 16);
                let u8_offset = unpack_front_dict_indices(
                    &mut cursor,
                    count,
                    entries_len,
                    width,
                    idx_mode,
                    &mut u8_arena,
                )?;
                for i in 0..count {
                    let idx = u8_arena[u8_offset + i] as usize;
                    let entry = entries
                        .get(idx)
                        .ok_or_else(|| invalid("column dictionary index"))?;
                    values.push_slice(entry);
                }
            } else {
                for _ in 0..entries_len {
                    let entry = read_bytes(&mut cursor)?;
                    if entries.last().is_some_and(|last: &Vec<u8>| last >= &entry) {
                        return Err(invalid("column dictionary ordering"));
                    }
                    entries.push(entry);
                }
                let width = cursor.byte()?;
                if width != bit_width((entries_len - 1) as u64) {
                    return Err(invalid("column dictionary width"));
                }
                let packed_len = bit_bytes(count, width)?;
                let mut reader = BitReader::new(cursor.take(packed_len)?);
                for _ in 0..count {
                    let index =
                        usize::try_from(reader.read(width)?).map_err(|_| Error::SizeOverflow)?;
                    let entry = entries
                        .get(index)
                        .ok_or_else(|| invalid("column dictionary index"))?;
                    values.push_slice(entry);
                }
                reader.finish_zero_padded()?;
            }
        }
        MODEL_DECIMAL_AFFINE => {
            if count < 2 {
                return Err(invalid("column affine count"));
            }
            let first = cursor.u64()?;
            let step = cursor.i64()? as i128;
            for index in 0..count {
                let value = first as i128 + step * index as i128;
                let value = u64::try_from(value).map_err(|_| invalid("column affine range"))?;
                push_decimal(&mut values, value);
            }
        }
        MODEL_HEX_SUFFIX => {
            if count < 2 {
                return Err(invalid("column hex count"));
            }
            let prefix = read_bytes(&mut cursor)?;
            let width = cursor.uleb()?;
            if width == 0 || width > 64 {
                return Err(invalid("column hex width"));
            }
            for _ in 0..count {
                let raw = cursor.take(width)?;
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&prefix);
                for &byte in raw {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                values.bytes.push(b'"');
                values.finish(start);
            }
        }
        MODEL_TIMESTAMP => {
            if count < 2 {
                return Err(invalid("column timestamp count"));
            }
            let year = cursor.u16()?;
            if year > 9999 {
                return Err(invalid("column timestamp year"));
            }
            let plans = (0..5)
                .map(|_| {
                    let base = cursor.byte()?;
                    let width = cursor.byte()?;
                    if width > 6 {
                        return Err(invalid("column timestamp width"));
                    }
                    Ok((base, width))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut components = Vec::with_capacity(5);
            for (base, width) in &plans {
                let packed_len = bit_bytes(count, *width)?;
                let mut reader = BitReader::new(cursor.take(packed_len)?);
                let values = (0..count)
                    .map(|_| {
                        let value = reader.read(*width)?;
                        u8::try_from(value + *base as u64).map_err(|_| invalid("column timestamp range"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                reader.finish_zero_padded()?;
                components.push(values);
            }
            for index in 0..count {
                let (month, day, hour, minute, second) = (
                    components[0][index],
                    components[1][index],
                    components[2][index],
                    components[3][index],
                    components[4][index],
                );
                if !(1..=12).contains(&month)
                    || !(1..=31).contains(&day)
                    || hour >= 24
                    || minute >= 60
                    || second >= 60
                {
                    return Err(invalid("column timestamp component"));
                }
                let start = values.begin();
                values.bytes.push(b'"');
                write_four_digits(&mut values.bytes, year);
                values.bytes.push(b'-');
                write_two_digits(&mut values.bytes, month);
                values.bytes.push(b'-');
                write_two_digits(&mut values.bytes, day);
                values.bytes.push(b'T');
                write_two_digits(&mut values.bytes, hour);
                values.bytes.push(b':');
                write_two_digits(&mut values.bytes, minute);
                values.bytes.push(b':');
                write_two_digits(&mut values.bytes, second);
                values.bytes.extend_from_slice(b"Z\"");
                values.finish(start);
            }
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
            let mut reader = BitReader::new(cursor.take(packed_len)?);
            for _ in 0..count {
                let residual = reader.read(width)?;
                let value = min.checked_add(residual).ok_or(Error::SizeOverflow)?;
                push_decimal(&mut values, value);
            }
            reader.finish_zero_padded()?;
        }
        MODEL_DECIMAL_SUFFIX => {
            if count < 2 {
                return Err(invalid("column decimal suffix count"));
            }
            let prefix = read_bytes(&mut cursor)?;
            let min = cursor.u64()?;
            let width = cursor.byte()?;
            if width > 64 {
                return Err(invalid("column decimal suffix width"));
            }
            let packed_len = bit_bytes(count, width)?;
            let mut reader = BitReader::new(cursor.take(packed_len)?);
            for _ in 0..count {
                let residual = reader.read(width)?;
                let value = min.checked_add(residual).ok_or(Error::SizeOverflow)?;
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&prefix);
                write_decimal(&mut values.bytes, value);
                values.bytes.push(b'"');
                values.finish(start);
            }
            reader.finish_zero_padded()?;
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
            let mut reader = BitReader::new(cursor.take(packed_len)?);
            push_decimal(&mut values, first);
            let mut previous = first as i128;
            for _ in 1..count {
                let zigzag = reader.read(width)?;
                let delta = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                let current = previous + i128::from(delta);
                let current =
                    u64::try_from(current).map_err(|_| invalid("column decimal delta range"))?;
                push_decimal(&mut values, current);
                previous = i128::from(current);
            }
            reader.finish_zero_padded()?;
        }
        MODEL_PERIODIC_STRING => {
            if count < 2 {
                return Err(invalid("column periodic count"));
            }
            let period = cursor.uleb()?;
            if !(1..=64).contains(&period) {
                return Err(invalid("column periodic period"));
            }
            for _ in 0..count {
                let len = cursor.uleb()?;
                if len > MAX_BLOCK_SIZE {
                    return Err(invalid("column periodic length"));
                }
                let seed_len = len.min(period);
                let seed = cursor.take(seed_len)?;
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(seed);
                for index in seed_len..len {
                    let byte = values.bytes[start as usize + 1 + index % period];
                    values.bytes.push(byte);
                }
                values.bytes.push(b'"');
                values.finish(start);
            }
        }
        MODEL_PERIODIC_HEX => {
            if count < 2 {
                return Err(invalid("column periodic hex count"));
            }
            let period = cursor.uleb()?;
            if !(1..=64).contains(&period) {
                return Err(invalid("column periodic hex period"));
            }
            for _ in 0..count {
                let len = cursor.uleb()?;
                if len > MAX_BLOCK_SIZE {
                    return Err(invalid("column periodic hex length"));
                }
                let seed_len = len.min(period);
                let packed_len = seed_len.div_ceil(2);
                let packed = cursor.take(packed_len)?;
                let start = values.begin();
                values.bytes.push(b'"');
                for (index, &byte) in packed.iter().enumerate() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    if index * 2 + 1 < seed_len {
                        values.bytes.push(HEX[(byte & 15) as usize]);
                    } else if byte & 15 != 0 {
                        return Err(invalid("nonzero periodic hex tail"));
                    }
                }
                for index in seed_len..len {
                    let byte = values.bytes[start as usize + 1 + index % period];
                    values.bytes.push(byte);
                }
                values.bytes.push(b'"');
                values.finish(start);
            }
        }
        MODEL_GF2_AFFINE => {
            if count <= GF2_TRAIN_TRANSITIONS + 1 {
                return Err(invalid("column gf2 recurrence count"));
            }
            let prefix = read_bytes(&mut cursor)?;
            let first = cursor.u64()?;
            let mut coefficients = [0u64; GF2_COEFFICIENTS];
            for coefficient in &mut coefficients {
                *coefficient = cursor.u64()?;
            }
            let mut word = first;
            for _ in 0..count {
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&prefix);
                for byte in word.to_le_bytes() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                values.bytes.push(b'"');
                values.finish(start);
                word = apply_affine(&coefficients, word);
            }
        }
        MODEL_EPOCH_COUNTER => {
            if count < 2 {
                return Err(invalid("column epoch counter count"));
            }
            let literal = read_bytes(&mut cursor)?;
            let prefix_count = cursor.uleb()?;
            if prefix_count == 0 || prefix_count > MAX_EPOCH_PREFIXES {
                return Err(invalid("column epoch prefix count"));
            }
            let mut prefixes = Vec::with_capacity(prefix_count);
            for _ in 0..prefix_count {
                let bytes = cursor.take(4)?;
                let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                if prefixes.last().is_some_and(|last: &u32| *last >= value) {
                    return Err(invalid("column epoch prefix ordering"));
                }
                prefixes.push(value);
            }
            let width = cursor.byte()?;
            if width != bit_width((prefix_count - 1) as u64) {
                return Err(invalid("column epoch prefix width"));
            }
            let index_len = bit_bytes(count, width)?;
            let mut index_reader = BitReader::new(cursor.take(index_len)?);
            let prefix_indices = (0..count)
                .map(|_| {
                    let index = usize::try_from(index_reader.read(width)?)
                        .map_err(|_| Error::SizeOverflow)?;
                    prefixes
                        .get(index)
                        .copied()
                        .ok_or_else(|| invalid("column epoch prefix index"))
                })
                .collect::<Result<Vec<_>>>()?;
            index_reader.finish_zero_padded()?;
            let flag_len = bit_bytes(count, 1)?;
            let mut flag_reader = BitReader::new(cursor.take(flag_len)?);
            let flags = (0..count)
                .map(|_| flag_reader.read(1))
                .collect::<Result<Vec<_>>>()?;
            flag_reader.finish_zero_padded()?;
            if flags.first() != Some(&1) {
                return Err(invalid("column epoch first explicit"));
            }
            let explicit_count = cursor.uleb()?;
            if explicit_count != flags.iter().filter(|&&flag| flag == 1).count() {
                return Err(invalid("column epoch explicit count"));
            }
            let mut explicit = Vec::with_capacity(explicit_count);
            for _ in 0..explicit_count {
                let value = cursor.uleb()?;
                if value > u32::MAX as usize {
                    return Err(invalid("column epoch suffix range"));
                }
                explicit.push(value as u32);
            }
            let mut explicit = explicit.into_iter();
            let mut previous = 0u64;
            for row in 0..count {
                let suffix = if flags[row] == 1 {
                    u64::from(
                        explicit
                            .next()
                            .ok_or_else(|| invalid("column epoch explicit order"))?,
                    )
                } else {
                    previous
                        .checked_add(1)
                        .ok_or_else(|| invalid("column epoch sequential"))?
                };
                if suffix > u64::from(u32::MAX) {
                    return Err(invalid("column epoch suffix range"));
                }
                let prefix = prefix_indices[row];
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&literal);
                for byte in prefix.to_be_bytes() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                for byte in (suffix as u32).to_be_bytes() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                values.bytes.push(b'"');
                values.finish(start);
                previous = suffix;
            }
        }
        MODEL_SEGMENT_DICT => {
            if count < 2 {
                return Err(invalid("column segment dictionary count"));
            }
            let dictionary_len = cursor.uleb()?;
            if dictionary_len > MAX_SEGMENT_DICT {
                return Err(invalid("column segment dictionary size"));
            }
            let mut dictionary = Vec::with_capacity(dictionary_len);
            for _ in 0..dictionary_len {
                let entry = read_bytes(&mut cursor)?;
                if entry.is_empty()
                    || entry.len() > MAX_SEGMENT_LEN
                    || entry.iter().any(|&byte| !segment_byte_ok(byte))
                    || canonical_segment_decimal(&entry).is_some()
                {
                    return Err(invalid("column segment dictionary entry"));
                }
                if dictionary.last().is_some_and(|last: &Vec<u8>| last >= &entry) {
                    return Err(invalid("column segment dictionary ordering"));
                }
                dictionary.push(entry);
            }
            let width = cursor.byte()?;
            if width != bit_width((dictionary_len + 1) as u64) {
                return Err(invalid("column segment code width"));
            }
            let code_count = cursor.uleb()?;
            if code_count == 0
                || code_count > count.saturating_mul(MAX_SEGMENTS_PER_VALUE + 1)
            {
                return Err(invalid("column segment code count"));
            }
            let code_len = bit_bytes(code_count, width)?;
            let mut code_reader = BitReader::new(cursor.take(code_len)?);
            let codes = (0..code_count)
                .map(|_| code_reader.read(width))
                .collect::<Result<Vec<_>>>()?;
            code_reader.finish_zero_padded()?;
            let decimal_count = cursor.uleb()?;
            if decimal_count > code_count {
                return Err(invalid("column segment decimal count"));
            }
            let decimal_width = cursor.byte()?;
            if decimal_width > 30 {
                return Err(invalid("column segment decimal width"));
            }
            let decimal_len = bit_bytes(decimal_count, decimal_width)?;
            let mut decimal_reader = BitReader::new(cursor.take(decimal_len)?);
            let decimals = (0..decimal_count)
                .map(|_| decimal_reader.read(decimal_width))
                .collect::<Result<Vec<_>>>()?;
            decimal_reader.finish_zero_padded()?;
            if decimals.iter().any(|&value| value > 999_999_999) {
                return Err(invalid("column segment decimal range"));
            }
            if bit_width(decimals.iter().copied().max().unwrap_or(0)) != decimal_width {
                return Err(invalid("column segment decimal width"));
            }
            let pad_width = cursor.byte()?;
            if pad_width > 12 {
                return Err(invalid("column segment pad width"));
            }
            let pad_len = bit_bytes(count, pad_width)?;
            let mut pad_reader = BitReader::new(cursor.take(pad_len)?);
            let pads = (0..count)
                .map(|_| pad_reader.read(pad_width))
                .collect::<Result<Vec<_>>>()?;
            pad_reader.finish_zero_padded()?;
            if pads.iter().any(|&pad| pad > MAX_PAD as u64) {
                return Err(invalid("column segment pad"));
            }
            if bit_width(pads.iter().copied().max().unwrap_or(0)) != pad_width {
                return Err(invalid("column segment pad width"));
            }
            let end_code = dictionary_len as u64;
            let decimal_code = dictionary_len as u64 + 1;
            if codes.iter().any(|&code| code > decimal_code) {
                return Err(invalid("column segment code"));
            }
            if codes.iter().filter(|&&code| code == end_code).count() != count {
                return Err(invalid("column segment terminator count"));
            }
            if codes.iter().filter(|&&code| code == decimal_code).count() != decimals.len() {
                return Err(invalid("column segment decimal usage"));
            }
            let mut decimals = decimals.into_iter();
            let mut pads = pads.into_iter();
            let mut start = values.begin();
            let mut open = false;
            let mut segment_count = 0usize;
            for code in codes {
                if code == end_code {
                    if !open {
                        values.bytes.push(b'"');
                    }
                    let pad = pads.next().ok_or_else(|| invalid("column segment pad order"))?;
                    values.bytes.extend(std::iter::repeat(b' ').take(pad as usize));
                    values.bytes.push(b'"');
                    values.finish(start);
                    open = false;
                    segment_count = 0;
                    start = values.begin();
                } else {
                    if open {
                        values.bytes.extend_from_slice(b"; ");
                    } else {
                        values.bytes.push(b'"');
                        open = true;
                    }
                    if code == decimal_code {
                        let decimal = decimals
                            .next()
                            .ok_or_else(|| invalid("column segment decimal order"))?;
                        write_decimal(&mut values.bytes, decimal);
                    } else {
                        let entry = dictionary
                            .get(code as usize)
                            .ok_or_else(|| invalid("column segment index"))?;
                        values.bytes.extend_from_slice(entry);
                    }
                    segment_count += 1;
                    if segment_count > MAX_SEGMENTS_PER_VALUE {
                        return Err(invalid("column segment count"));
                    }
                }
            }
            if open {
                return Err(invalid("column segment missing terminator"));
            }
        }
        _ => return Err(invalid("column model selector")),
    }
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    if strict {
        let owned = values.materialize();
        let (canonical, _) = pack_model_values(&owned);
        if canonical != bytes {
            return Err(invalid("noncanonical column model"));
        }
    }
    Ok(values)
}

#[inline(always)]
fn has_structured_line_near(
    data: &[u8],
    probe_base: usize,
    skip_first_partial: bool,
    max_lines: usize,
    hits: &mut usize,
    prev_slots: &mut Option<usize>,
    probe_spans: &mut [(u32, u32); MAX_SLOTS],
    probe_overflow: &mut Vec<(u32, u32)>,
) -> bool {
    let mut pos = if skip_first_partial {
        let Some(nl) = find_byte(b'\n', &data[probe_base..data.len().min(probe_base + 4096)])
        else {
            return false;
        };
        probe_base + nl + 1
    } else {
        probe_base
    };
    for _ in 0..max_lines {
        if pos >= data.len() {
            break;
        }
        let Some(nl) = find_byte(b'\n', &data[pos..data.len().min(pos + 4096)]) else {
            break;
        };
        let end = pos + nl + 1;
        let line = &data[pos..end];
        if let Some(slots) = parse_record_spans(line, probe_spans, probe_overflow) {
            let mut p = 0usize;
            skip_space(line, &mut p);
            if line.get(p) == Some(&b'{') {
                return true;
            }
            if *prev_slots == Some(slots) {
                *hits += 1;
                if *hits >= 2 {
                    return true;
                }
            } else {
                *prev_slots = Some(slots);
                if *hits == 0 {
                    *hits = 1;
                }
            }
        }
        pos = end;
    }
    false
}

fn pack_grammar_into(data: &[u8], out: &mut Vec<u8>) -> Result<bool> {
    if data.len() >= 256 {
        let sample_end = data.len().min(64);
        for &b in &data[..sample_end] {
            if b < b'\t' || (b >= 0x0e && b < b' ') {
                return Ok(false);
            }
        }
        let first_looks_json = data[0] == b'{'
            && find_byte(b'\n', &data[..data.len().min(4096)]).is_some_and(|nl| {
                nl >= 2
                    && (data[nl - 1] == b'}'
                        || (nl >= 3 && data[nl - 1] == b'\r' && data[nl - 2] == b'}'))
            });
        if !first_looks_json {
            let mut probe_spans = [(0u32, 0u32); MAX_SLOTS];
            let mut probe_overflow = Vec::new();
            let mut hits = 0usize;
            let mut prev_slots = None;
            let has_start = has_structured_line_near(
                data,
                0,
                false,
                8,
                &mut hits,
                &mut prev_slots,
                &mut probe_spans,
                &mut probe_overflow,
            );
            if !has_start {
                let mid = data.len() / 2;
                let try_mid = if hits > 0 {
                    data.len() >= 4096
                } else {
                    data.len() >= 4096
                        && find_byte(b'\n', &data[mid..data.len().min(mid + 4096)])
                            .is_some_and(|nl| data.get(mid + nl + 1) == Some(&b'{'))
                };
                if !try_mid
                    || !has_structured_line_near(
                        data,
                        mid,
                        true,
                        6,
                        &mut hits,
                        &mut prev_slots,
                        &mut probe_spans,
                        &mut probe_overflow,
                    )
                {
                    return Ok(false);
                }
            }
        }
    }
    let (templates, rows, template_rows, raw_rows, raw_bytes) = grammar_from_block_mode(data, true)?;
    if templates.is_empty()
        || !templates.iter().any(|t| !t.values.is_empty())
        || template_rows < 2
        || (data.len() >= 4096 && raw_bytes * 3 > data.len() * 2)
    {
        return_row_vec(rows);
        return Ok(false);
    }
    out.push(GRAMMAR_PACKET);
    put_uleb(out, data.len());
    put_uleb(out, templates.len());
    let total_segments: usize = templates.iter().map(|t| t.segments.len()).sum();
    let mut seg_dict_inline = [core::mem::MaybeUninit::<&[u8]>::uninit(); 128];
    let mut seg_dict_overflow = Vec::new();
    let segment_dictionary: &mut [&[u8]] = if total_segments <= 128 {
        let mut idx = 0usize;
        for template in &templates {
            for &seg in &template.segments {
                seg_dict_inline[idx].write(seg);
                idx += 1;
            }
        }
        let slice: &mut [&[u8]] = unsafe {
            std::slice::from_raw_parts_mut(seg_dict_inline.as_mut_ptr().cast::<&[u8]>(), total_segments)
        };
        slice.sort_unstable();
        let mut w = 0usize;
        for r in 0..slice.len() {
            if w == 0 || slice[r] != slice[w - 1] {
                slice[w] = slice[r];
                w += 1;
            }
        }
        &mut slice[..w]
    } else {
        seg_dict_overflow.reserve(total_segments);
        for template in &templates {
            seg_dict_overflow.extend_from_slice(&template.segments);
        }
        seg_dict_overflow.sort_unstable();
        seg_dict_overflow.dedup();
        &mut seg_dict_overflow[..]
    };
    put_uleb(out, segment_dictionary.len());
    for segment in &*segment_dictionary {
        put_bytes(out, segment);
    }
    for template in &templates {
        let slots = template.values.len();
        if template.segments.len() != slots + 1 || slots > MAX_SLOTS {
            return_row_vec(rows);
            return Err(invalid("column template shape"));
        }
        put_uleb(out, slots);
    }
    let segment_width = bit_width(segment_dictionary.len().saturating_sub(1) as u64);
    out.push(segment_width);
    for template in &templates {
        push_bit_values(
            out,
            template.segments.len(),
            segment_width,
            template
                .segments
                .iter()
                .map(|segment| segment_dictionary.binary_search(segment).unwrap() as u64),
        );
    }
    put_uleb(out, rows.len());
    let selector_width = bit_width(templates.len() as u64);
    out.push(selector_width);
    if templates.len() == 1 && raw_rows == 0 {
        let full_bytes = rows.len() / 8;
        let rem_bits = rows.len() & 7;
        out.resize(out.len() + full_bytes, 0xff);
        if rem_bits != 0 {
            out.push((1u8 << rem_bits) - 1);
        }
    } else {
        push_bit_values(
            out,
            rows.len(),
            selector_width,
            rows.iter().map(|row| match row {
                Row::Template(index) => (index + 1) as u64,
                Row::Raw(_) => 0,
            }),
        );
        if raw_rows > 0 {
            if selector_width <= 7 && raw_bytes >= 64 {
                let wrote_lz = MODEL_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    let ModelScratch {
                        raw_lens,
                        raw_stream,
                        raw_lz,
                        ..
                    } = &mut *sc;
                    raw_lens.clear();
                    raw_lens.reserve(raw_rows * 2);
                    raw_stream.clear();
                    raw_stream.reserve(4 + raw_rows * 2 + raw_bytes);
                    for row in &rows {
                        if let Row::Raw(bytes) = row {
                            put_uleb(raw_lens, bytes.len());
                            raw_stream.extend_from_slice(bytes);
                        }
                    }
                    let plain_len = raw_lens.len() + raw_bytes;
                    let hdr_len = uleb_len(raw_lens.len()) + raw_lens.len();
                    let total_stream_len = hdr_len + raw_bytes;
                    raw_lz.clear();
                    if (64..=65536).contains(&total_stream_len) {
                        raw_stream.resize(total_stream_len, 0);
                        raw_stream.copy_within(0..raw_bytes, hdr_len);
                        let mut hdr_pos = 0usize;
                        let mut v = raw_lens.len();
                        while v >= 0x80 {
                            raw_stream[hdr_pos] = ((v as u8) & 0x7f) | 0x80;
                            hdr_pos += 1;
                            v >>= 7;
                        }
                        raw_stream[hdr_pos] = v as u8;
                        hdr_pos += 1;
                        raw_stream[hdr_pos..hdr_len].copy_from_slice(raw_lens);
                        raw_lz.reserve(total_stream_len / 2);
                        if pack_fast_lz_into(raw_stream, raw_lz).is_some()
                            && 1 + uleb_len(raw_lz.len()) + raw_lz.len() + 4 < plain_len
                        {
                            out.push(0);
                            put_uleb(out, raw_lz.len());
                            out.extend_from_slice(raw_lz);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                });
                if !wrote_lz {
                    for row in &rows {
                        if let Row::Raw(bytes) = row {
                            put_bytes(out, bytes);
                        }
                    }
                }
            } else {
                for row in &rows {
                    if let Row::Raw(bytes) = row {
                        put_bytes(out, bytes);
                    }
                }
            }
        }
    }
    return_row_vec(rows);
    for template in &templates {
        for column in &template.values {
            let (model, _kind) = pack_model_v3(column);
            let _ = uleb_len(model.len())
                .checked_add(model.len())
                .ok_or(Error::SizeOverflow)?;
            put_uleb(out, model.len());
            out.extend_from_slice(&model);
        }
    }
    Ok(true)
}

fn pack_grammar(data: &[u8]) -> Result<PackedGrammar> {
    let (templates, rows, template_rows, raw_rows) = grammar_from_block(data)?;
    let mut out = Vec::with_capacity(4096);
    out.push(GRAMMAR_PACKET);
    put_uleb(&mut out, data.len());
    put_uleb(&mut out, templates.len());
    let total_segments: usize = templates.iter().map(|t| t.segments.len()).sum();
    let mut segment_dictionary = Vec::with_capacity(total_segments);
    for template in &templates {
        segment_dictionary.extend_from_slice(&template.segments);
    }
    segment_dictionary.sort_unstable();
    segment_dictionary.dedup();
    put_uleb(&mut out, segment_dictionary.len());
    for segment in &segment_dictionary {
        put_bytes(&mut out, segment);
    }
    for template in &templates {
        let slots = template.values.len();
        if template.segments.len() != slots + 1 || slots > MAX_SLOTS {
            return_row_vec(rows);
            return Err(invalid("column template shape"));
        }
        put_uleb(&mut out, slots);
    }
    let segment_width = bit_width(segment_dictionary.len().saturating_sub(1) as u64);
    out.push(segment_width);
    for template in &templates {
        push_bit_values(
            &mut out,
            template.segments.len(),
            segment_width,
            template
                .segments
                .iter()
                .map(|segment| segment_dictionary.binary_search(segment).unwrap() as u64),
        );
    }
    put_uleb(&mut out, rows.len());
    let selector_width = bit_width(templates.len() as u64);
    out.push(selector_width);
    push_bit_values(
        &mut out,
        rows.len(),
        selector_width,
        rows.iter().map(|row| match row {
            Row::Template(index) => (index + 1) as u64,
            Row::Raw(_) => 0,
        }),
    );
    for row in &rows {
        if let Row::Raw(bytes) = row {
            put_bytes(&mut out, bytes);
        }
    }
    return_row_vec(rows);
    let mut stats = BlockStats {
        grammar: true,
        templates: templates.len(),
        template_rows,
        raw_rows,
        columns: Vec::with_capacity(total_segments),
    };
    for (template_index, template) in templates.iter().enumerate() {
        for (slot, column) in template.values.iter().enumerate() {
            let (model, kind) = pack_model(column);
            let payload_bytes = uleb_len(model.len())
                .checked_add(model.len())
                .ok_or(Error::SizeOverflow)?;
            put_uleb(&mut out, model.len());
            out.extend_from_slice(&model);
            stats.columns.push(ColumnStat {
                template: template_index,
                slot,
                kind,
                values: column.len(),
                payload_bytes,
            });
        }
    }
    Ok(PackedGrammar { bytes: out, stats })
}

/// Preloaded 64-byte piece for branchless unaligned `u128` stores during row assembly.
///
/// Size is exactly 64 bytes (`1 << 6`) so indexing `entries[idx]` compiles to a bit-shift `idx << 6`.
#[derive(Clone, Copy)]
#[repr(C)]
struct FastPiece {
    v0: u128,
    v1: u128,
    v2: u128,
    len: usize,
    ptr: *const u8,
}

#[inline(always)]
unsafe fn read_u128_partial(ptr: *const u8, len: usize) -> u128 {
    unsafe {
        if len >= 16 {
            ptr.cast::<u128>().read_unaligned()
        } else if len >= 8 {
            let lo = ptr.cast::<u64>().read_unaligned() as u128;
            let hi = ptr.add(len - 8).cast::<u64>().read_unaligned() as u128;
            lo | ((hi >> ((16 - len) * 8)) << 64)
        } else if len >= 4 {
            let lo = ptr.cast::<u32>().read_unaligned() as u64;
            let hi = ptr.add(len - 4).cast::<u32>().read_unaligned() as u64;
            (lo | ((hi >> ((8 - len) * 8)) << 32)) as u128
        } else if len > 0 {
            let b0 = *ptr as u64;
            let b1 = *ptr.add(len >> 1) as u64;
            let b2 = *ptr.add(len - 1) as u64;
            (b0 | (b1 << ((len >> 1) * 8)) | (b2 << ((len - 1) * 8))) as u128
        } else {
            0
        }
    }
}

impl FastPiece {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            v0: 0,
            v1: 0,
            v2: 0,
            len: 0,
            ptr: core::ptr::null(),
        }
    }

    #[inline(always)]
    fn from_slice(slice: &[u8]) -> Self {
        let len = slice.len();
        let ptr = slice.as_ptr();
        unsafe {
            let v0 = read_u128_partial(ptr, len);
            let v1 = if len > 16 {
                read_u128_partial(ptr.add(16), len - 16)
            } else {
                0
            };
            let v2 = if len > 32 {
                read_u128_partial(ptr.add(32), len - 32)
            } else {
                0
            };
            Self {
                v0,
                v1,
                v2,
                len,
                ptr,
            }
        }
    }

    #[inline(always)]
    unsafe fn emit_16(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_32(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(16).cast::<u128>().write_unaligned(self.v1);
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_48_exact(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(16).cast::<u128>().write_unaligned(self.v1);
            dst.add(32).cast::<u128>().write_unaligned(self.v2);
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_48(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(16).cast::<u128>().write_unaligned(self.v1);
            dst.add(32).cast::<u128>().write_unaligned(self.v2);
            if self.len > 48 {
                Self::emit_tail_cold(self.ptr, dst, self.len);
            }
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_tail_cold(src: *const u8, dst: *mut u8, len: usize) {
        unsafe {
            if len <= 160 {
                let mut off = 48usize;
                while off + 16 < len {
                    dst.add(off)
                        .cast::<u128>()
                        .write_unaligned(src.add(off).cast::<u128>().read_unaligned());
                    off += 16;
                }
                dst.add(len - 16)
                    .cast::<u128>()
                    .write_unaligned(src.add(len - 16).cast::<u128>().read_unaligned());
            } else {
                std::ptr::copy_nonoverlapping(src.add(48), dst.add(48), len - 48);
            }
        }
    }
}

/// Reusable per-thread scratch arenas for zero-allocation block decoding.
#[derive(Default)]
struct DecodeScratch {
    u8_arena: Vec<u8>,
    u16_arena: Vec<u16>,
    u32_arena: Vec<u32>,
    u64_arena: Vec<u64>,
    piece_arena: Vec<FastPiece>,
    str_arena: Vec<u8>,
    slice_arena: Vec<(*const u8, usize)>,
    raw_arena: Vec<u8>,
    unbound_col_arena: Vec<UnboundExecColumn<'static>>,
    exec_col_arena: Vec<ExecColumn<'static>>,
}

impl DecodeScratch {
    #[inline(always)]
    fn clear(&mut self) {
        self.u8_arena.clear();
        self.u16_arena.clear();
        self.u32_arena.clear();
        self.u64_arena.clear();
        self.piece_arena.clear();
        self.str_arena.clear();
        self.slice_arena.clear();
        self.raw_arena.clear();
        self.unbound_col_arena.clear();
        self.exec_col_arena.clear();
    }
}

std::thread_local! {
    static DECODE_SCRATCH: std::cell::RefCell<DecodeScratch> = std::cell::RefCell::new(DecodeScratch {
        u8_arena: Vec::with_capacity(16384),
        u16_arena: Vec::with_capacity(4096),
        u32_arena: Vec::with_capacity(4096),
        u64_arena: Vec::with_capacity(4096),
        piece_arena: Vec::with_capacity(512),
        str_arena: Vec::with_capacity(4096),
        slice_arena: Vec::with_capacity(512),
        raw_arena: Vec::with_capacity(262144),
        unbound_col_arena: Vec::with_capacity(256),
        exec_col_arena: Vec::with_capacity(256),
    });
}

#[inline(always)]
fn read_front_dict_palette(cursor: &mut Cursor<'_>) -> Result<([u8; 32], u8, bool)> {
    let hdr = cursor.byte()?;
    let pal_mixed = (hdr & 0x80) != 0;
    let pal_len = hdr & 0x7f;
    if pal_len == 0 || pal_len > if pal_mixed { 16 } else { 32 } {
        return Err(invalid("column front dict palette length"));
    }
    let pal_slice = cursor.take(pal_len as usize)?;
    for w in pal_slice.windows(2) {
        if w[0] >= w[1] {
            return Err(invalid("column front dict palette ordering"));
        }
    }
    let mut pal = [0u8; 32];
    pal[..pal_len as usize].copy_from_slice(pal_slice);
    Ok((pal, pal_len, pal_mixed))
}

#[inline(always)]
fn unpack_nibble_bytes_into(
    src: &[u8],
    char_len: usize,
    pal: &[u8; 32],
    pal_len: u8,
    dst: *mut u8,
) -> Result<()> {
    if pal_len > 16 {
        let mut j = 0usize;
        let mut s_idx = 0usize;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            let v_pal = uint8x16x2_t(vld1q_u8(pal.as_ptr()), vld1q_u8(pal.as_ptr().add(16)));
            while j + 8 <= char_len && s_idx + 8 <= src.len() {
                let w = u64::from_le(src.as_ptr().add(s_idx).cast::<u64>().read_unaligned());
                let idx_u64 = (w & 0x1f)
                    | (((w >> 5) & 0x1f) << 8)
                    | (((w >> 10) & 0x1f) << 16)
                    | (((w >> 15) & 0x1f) << 24)
                    | (((w >> 20) & 0x1f) << 32)
                    | (((w >> 25) & 0x1f) << 40)
                    | (((w >> 30) & 0x1f) << 48)
                    | (((w >> 35) & 0x1f) << 56);
                let v_idx = vreinterpret_u8_u64(vdup_n_u64(idx_u64));
                if pal_len < 32 && vmaxv_u8(v_idx) >= pal_len {
                    return Err(invalid("column front dict 5bit out of bounds"));
                }
                let ascii = vqtbl2_u8(v_pal, v_idx);
                vst1_u8(dst.add(j), ascii);
                j += 8;
                s_idx += 5;
            }
        }
        let mut acc = 0u64;
        let mut bits = 0u32;
        while j < char_len {
            if bits < 5 {
                if s_idx + 4 <= src.len() {
                    let w = u32::from_le(unsafe {
                        src.as_ptr().add(s_idx).cast::<u32>().read_unaligned()
                    }) as u64;
                    acc |= w << bits;
                    bits += 32;
                    s_idx += 4;
                } else {
                    while s_idx < src.len() {
                        acc |= (src[s_idx] as u64) << bits;
                        bits += 8;
                        s_idx += 1;
                    }
                }
            }
            let idx = (acc & 0x1f) as u8;
            if idx >= pal_len {
                return Err(invalid("column front dict 5bit out of bounds"));
            }
            unsafe {
                *dst.add(j) = *pal.get_unchecked(idx as usize);
            }
            acc >>= 5;
            bits -= 5;
            j += 1;
        }
        if acc != 0 {
            return Err(invalid("column front dict 5bit padding"));
        }
        return Ok(());
    }
    let mut j = 0usize;
    let mut b_idx = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v_pal = vld1q_u8(pal.as_ptr());
        while j + 16 <= char_len {
            let v8 = vld1_u8(src.as_ptr().add(b_idx));
            let hi = vshr_n_u8(v8, 4);
            let lo = vand_u8(v8, vdup_n_u8(0x0f));
            let zipped = vcombine_u8(vzip1_u8(hi, lo), vzip2_u8(hi, lo));
            if pal_len < 16 && vmaxvq_u8(zipped) >= pal_len {
                return Err(invalid("column front dict nibble out of bounds"));
            }
            let ascii = vqtbl1q_u8(v_pal, zipped);
            vst1q_u8(dst.add(j), ascii);
            j += 16;
            b_idx += 8;
        }
        if j + 8 <= char_len {
            let w4 = src.as_ptr().add(b_idx).cast::<u32>().read_unaligned();
            let v8 = vreinterpret_u8_u32(vdup_n_u32(w4));
            let hi = vshr_n_u8(v8, 4);
            let lo = vand_u8(v8, vdup_n_u8(0x0f));
            let zipped = vzip1_u8(hi, lo);
            if pal_len < 16 && vmaxv_u8(zipped) >= pal_len {
                return Err(invalid("column front dict nibble out of bounds"));
            }
            let ascii = vqtbl1_u8(v_pal, zipped);
            vst1_u8(dst.add(j), ascii);
            j += 8;
            b_idx += 4;
        }
    }
    while j + 1 < char_len {
        let b = unsafe { *src.get_unchecked(b_idx) };
        let hi = b >> 4;
        let lo = b & 0x0f;
        if hi >= pal_len || lo >= pal_len {
            return Err(invalid("column front dict nibble out of bounds"));
        }
        unsafe {
            *dst.add(j) = *pal.get_unchecked(hi as usize);
            *dst.add(j + 1) = *pal.get_unchecked(lo as usize);
        }
        j += 2;
        b_idx += 1;
    }
    if j < char_len {
        let b = unsafe { *src.get_unchecked(b_idx) };
        let hi = b >> 4;
        let lo = b & 0x0f;
        if hi >= pal_len || lo != 0 {
            return Err(invalid("column front dict nibble padding"));
        }
        unsafe {
            *dst.add(j) = *pal.get_unchecked(hi as usize);
        }
    }
    Ok(())
}

/// Unpack `count` `width`-bit integers (`width in 0..=7`) into `out`, adding `base`
/// to each value, and simultaneously verify that every raw value is `< bound`
/// (`1 <= bound <= 128`) and tail padding bits are zero.
///
/// Processes 8 rows (`width` bytes) per iteration using 64-bit SWAR with zero branches.
#[inline(always)]
fn unpack_and_validate_u8(
    bytes: &[u8],
    count: usize,
    width: u8,
    base: u8,
    bound: usize,
    out: &mut Vec<u8>,
) -> Result<usize> {
    let expected_len = bit_bytes(count, width)?;
    if bytes.len() != expected_len {
        return Err(Error::Truncated);
    }
    let used_bits = count * (width as usize);
    let tail_bits = used_bits & 7;
    if tail_bits != 0 {
        let last = *bytes.last().ok_or(Error::Truncated)?;
        if (last >> tail_bits) != 0 {
            return Err(invalid("nonzero column bit padding"));
        }
    }
    let start = out.len();
    out.reserve(count + 8);
    if width == 0 {
        unsafe {
            std::ptr::write_bytes(out.as_mut_ptr().add(start), base, count);
            out.set_len(start + count);
        }
        return Ok(start);
    }
    if width == 8 {
        if bound < 256 {
            let b_max = bound as u8;
            let mut i = 0usize;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                while i + 16 <= count {
                    let v = vld1q_u8(bytes.as_ptr().add(i));
                    if vmaxvq_u8(v) >= b_max {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    i += 16;
                }
            }
            while i < count {
                if bytes[i] >= b_max {
                    return Err(invalid("column bit index out of bounds"));
                }
                i += 1;
            }
        }
        unsafe {
            let dst = out.as_mut_ptr().add(start);
            if base == 0 {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, count);
            } else {
                for (i, &b) in bytes.iter().enumerate() {
                    *dst.add(i) = b.wrapping_add(base);
                }
            }
            out.set_len(start + count);
        }
        return Ok(start);
    }
    let w = width as u32;
    let m = (1u64 << w) - 1;
    let check_bound = bound < (1usize << w);
    let bias = ((128usize.saturating_sub(bound)) as u64).wrapping_mul(0x0101_0101_0101_0101);
    let base_splat = (base as u64).wrapping_mul(0x0101_0101_0101_0101);
    let mut bad_bits = 0u64;
    let mut row = 0usize;
    let mut byte_pos = 0usize;
    let step = width as usize;
    let src_ptr = bytes.as_ptr();
    let dst_ptr = unsafe { out.as_mut_ptr().add(start) };

    while row < count {
        let v = if byte_pos + 8 <= bytes.len() {
            u64::from_le(unsafe { src_ptr.add(byte_pos).cast::<u64>().read_unaligned() })
        } else {
            u64::from_le(unsafe {
                read_u128_partial(src_ptr.add(byte_pos), bytes.len() - byte_pos) as u64
            })
        };
        let u8x8 = (v & m)
            | (((v >> w) & m) << 8)
            | (((v >> (2 * w)) & m) << 16)
            | (((v >> (3 * w)) & m) << 24)
            | (((v >> (4 * w)) & m) << 32)
            | (((v >> (5 * w)) & m) << 40)
            | (((v >> (6 * w)) & m) << 48)
            | (((v >> (7 * w)) & m) << 56);
        if check_bound {
            bad_bits |= u8x8.wrapping_add(bias);
        }
        let out_word = if base == 0 {
            u8x8
        } else {
            u8x8.wrapping_add(base_splat)
        };
        unsafe {
            dst_ptr.add(row).cast::<u64>().write_unaligned(out_word.to_le());
        }
        row += 8;
        byte_pos += step;
    }
    if check_bound && (bad_bits & 0x8080_8080_8080_8080) != 0 {
        return Err(invalid("column bit index out of bounds"));
    }
    unsafe {
        out.set_len(start + count);
    }
    Ok(start)
}

fn unpack_front_dict_indices(
    cursor: &mut Cursor<'_>,
    count: usize,
    entries_len: usize,
    width: u8,
    idx_mode: u8,
    u8_arena: &mut Vec<u8>,
) -> Result<usize> {
    match idx_mode {
        0 => {
            let packed_len = bit_bytes(count, width)?;
            let packed_bytes = cursor.take(packed_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            unpack_and_validate_u8(packed_bytes, count, width, 0, entries_len, u8_arena)
        }
        1 => {
            let bm_len = bit_bytes(count, 1)?;
            let bm_bytes = cursor.take(bm_len)?;
            let _ = FastBitReader::new_validated(bm_bytes, count, 1)?;
            if count > 0 && (bm_bytes[0] & 1) != 0 {
                return Err(invalid("column front dict repeat bit0"));
            }
            let mut ones = 0usize;
            for &b in bm_bytes {
                ones += b.count_ones() as usize;
            }
            let exc_count = count - ones;
            let exc_len = bit_bytes(exc_count, width)?;
            let exc_bytes = cursor.take(exc_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let dst_offset = u8_arena.len();
            u8_arena.reserve(count + exc_count + 16);
            unsafe {
                u8_arena.set_len(dst_offset + count);
            }
            let exc_offset =
                unpack_and_validate_u8(exc_bytes, exc_count, width, 0, entries_len, u8_arena)?;
            unsafe {
                let exc_ptr = u8_arena.as_ptr().add(exc_offset);
                let dst_ptr = u8_arena.as_mut_ptr().add(dst_offset);
                let mut exc_idx = 0usize;
                let full_bytes = count >> 3;
                for byte_idx in 0..full_bytes {
                    let b = *bm_bytes.get_unchecked(byte_idx);
                    let d = dst_ptr.add(byte_idx << 3);
                    if b == 0xff {
                        let v =
                            (*exc_ptr.add(exc_idx - 1) as u64).wrapping_mul(0x0101_0101_0101_0101);
                        d.cast::<u64>().write_unaligned(v);
                    } else if b == 0x00 {
                        d.cast::<u64>()
                            .write_unaligned(exc_ptr.add(exc_idx).cast::<u64>().read_unaligned());
                        exc_idx += 8;
                    } else {
                        let nb = (!b) as usize;
                        exc_idx += nb & 1;
                        *d = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 1) & 1;
                        *d.add(1) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 2) & 1;
                        *d.add(2) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 3) & 1;
                        *d.add(3) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 4) & 1;
                        *d.add(4) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 5) & 1;
                        *d.add(5) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 6) & 1;
                        *d.add(6) = *exc_ptr.add(exc_idx - 1);
                        exc_idx += (nb >> 7) & 1;
                        *d.add(7) = *exc_ptr.add(exc_idx - 1);
                    }
                }
                let rem = count & 7;
                if rem != 0 {
                    let nb = (!*bm_bytes.get_unchecked(full_bytes)) as usize;
                    let d = dst_ptr.add(full_bytes << 3);
                    for bit in 0..rem {
                        exc_idx += (nb >> bit) & 1;
                        *d.add(bit) = *exc_ptr.add(exc_idx - 1);
                    }
                }
                u8_arena.set_len(dst_offset + count);
            }
            Ok(dst_offset)
        }
        2 => {
            let mode_rank = cursor.uleb()?;
            if mode_rank >= entries_len {
                return Err(invalid("column front dict mode rank"));
            }
            let bm_len = bit_bytes(count, 1)?;
            let bm_bytes = cursor.take(bm_len)?;
            let _ = FastBitReader::new_validated(bm_bytes, count, 1)?;
            let mut ones = 0usize;
            for &b in bm_bytes {
                ones += b.count_ones() as usize;
            }
            let exc_count = count - ones;
            let exc_bound = entries_len - 1;
            let exc_width = bit_width(exc_bound.saturating_sub(1) as u64);
            let exc_len = bit_bytes(exc_count, exc_width)?;
            let exc_bytes = cursor.take(exc_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let dst_offset = u8_arena.len();
            u8_arena.reserve(count + exc_count + 16);
            let mr = mode_rank as u8;
            unsafe {
                std::ptr::write_bytes(u8_arena.as_mut_ptr().add(dst_offset), mr, count);
                u8_arena.set_len(dst_offset + count);
            }
            let exc_offset = unpack_and_validate_u8(
                exc_bytes,
                exc_count,
                exc_width,
                0,
                exc_bound.max(1),
                u8_arena,
            )?;
            unsafe {
                let exc_ptr = u8_arena.as_ptr().add(exc_offset);
                let dst_ptr = u8_arena.as_mut_ptr().add(dst_offset);
                let mut exc_idx = 0usize;
                let full_words = count >> 6;
                for word_idx in 0..full_words {
                    let base = word_idx << 6;
                    let w = u64::from_le(
                        bm_bytes
                            .as_ptr()
                            .add(word_idx << 3)
                            .cast::<u64>()
                            .read_unaligned(),
                    );
                    let mut zeros = !w;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let v = *exc_ptr.add(exc_idx);
                        exc_idx += 1;
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                for byte_idx in (full_words << 3)..bm_bytes.len() {
                    let b = *bm_bytes.get_unchecked(byte_idx);
                    let base = byte_idx << 3;
                    let rem = (count - base).min(8);
                    let mask = ((1u16 << rem) - 1) as u8;
                    let mut zeros = (!b) & mask;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let v = *exc_ptr.add(exc_idx);
                        exc_idx += 1;
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                u8_arena.set_len(dst_offset + count);
            }
            Ok(dst_offset)
        }
        3 => {
            let mode_rank = cursor.uleb()?;
            if mode_rank >= entries_len {
                return Err(invalid("column front dict mode rank"));
            }
            let bm_len = bit_bytes(count, 1)?;
            let bm_bytes = cursor.take(bm_len)?;
            let _ = FastBitReader::new_validated(bm_bytes, count, 1)?;
            let mut mode_ones = 0usize;
            for &b in bm_bytes {
                mode_ones += b.count_ones() as usize;
            }
            let rem_count = count - mode_ones;
            let rep_len = bit_bytes(rem_count, 1)?;
            let rep_bytes = cursor.take(rep_len)?;
            let _ = FastBitReader::new_validated(rep_bytes, rem_count, 1)?;
            if rem_count > 0 && (rep_bytes[0] & 1) != 0 {
                return Err(invalid("column front dict repeat bit0"));
            }
            let mut rep_ones = 0usize;
            for &b in rep_bytes {
                rep_ones += b.count_ones() as usize;
            }
            let exc_count = rem_count - rep_ones;
            let exc_bound = entries_len - 1;
            let exc_width = bit_width(exc_bound.saturating_sub(1) as u64);
            let exc_len = bit_bytes(exc_count, exc_width)?;
            let exc_bytes = cursor.take(exc_len)?;
            if !cursor.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let dst_offset = u8_arena.len();
            u8_arena.reserve(count + exc_count + 16);
            let mr = mode_rank as u8;
            unsafe {
                std::ptr::write_bytes(u8_arena.as_mut_ptr().add(dst_offset), mr, count);
                u8_arena.set_len(dst_offset + count);
            }
            let exc_offset = unpack_and_validate_u8(
                exc_bytes,
                exc_count,
                exc_width,
                0,
                exc_bound.max(1),
                u8_arena,
            )?;
            unsafe {
                let exc_ptr = u8_arena.as_ptr().add(exc_offset);
                let dst_ptr = u8_arena.as_mut_ptr().add(dst_offset);
                let mut nm_idx = 0usize;
                let mut exc_idx = 0usize;
                let full_words = count >> 6;
                for word_idx in 0..full_words {
                    let base = word_idx << 6;
                    let w = u64::from_le(
                        bm_bytes
                            .as_ptr()
                            .add(word_idx << 3)
                            .cast::<u64>()
                            .read_unaligned(),
                    );
                    let mut zeros = !w;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let is_rep = (*rep_bytes.get_unchecked(nm_idx >> 3) >> (nm_idx & 7)) & 1;
                        nm_idx += 1;
                        exc_idx += (is_rep ^ 1) as usize;
                        let v = *exc_ptr.add(exc_idx - 1);
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                for byte_idx in (full_words << 3)..bm_bytes.len() {
                    let b = *bm_bytes.get_unchecked(byte_idx);
                    let base = byte_idx << 3;
                    let rem = (count - base).min(8);
                    let mask = ((1u16 << rem) - 1) as u8;
                    let mut zeros = (!b) & mask;
                    while zeros != 0 {
                        let bit = zeros.trailing_zeros() as usize;
                        zeros &= zeros - 1;
                        let is_rep = (*rep_bytes.get_unchecked(nm_idx >> 3) >> (nm_idx & 7)) & 1;
                        nm_idx += 1;
                        exc_idx += (is_rep ^ 1) as usize;
                        let v = *exc_ptr.add(exc_idx - 1);
                        *dst_ptr.add(base + bit) = v + u8::from(v >= mr);
                    }
                }
                u8_arena.set_len(dst_offset + count);
            }
            Ok(dst_offset)
        }
        _ => Err(invalid("column front dict index mode")),
    }
}

/// Unpack `count` `width`-bit integers (`width in 1..=14`) into `out`, adding `min`
/// to each value (where `min + (1 << width) - 1 < 65536`), and validate tail padding bits.
///
/// Processes 4 `u16` values per iteration using a single 64-bit load and SWAR add.
#[inline(always)]
fn unpack_u16_swar(
    bytes: &[u8],
    count: usize,
    width: u8,
    min: u16,
    out: &mut Vec<u16>,
) -> Result<usize> {
    let expected_len = bit_bytes(count, width)?;
    if bytes.len() != expected_len {
        return Err(Error::Truncated);
    }
    let used_bits = count * (width as usize);
    let tail_bits = used_bits & 7;
    if tail_bits != 0 {
        let last = *bytes.last().ok_or(Error::Truncated)?;
        if (last >> tail_bits) != 0 {
            return Err(invalid("nonzero column bit padding"));
        }
    }
    let start = out.len();
    out.reserve(count + 8);
    let w = width as u32;
    let m = (1u64 << w) - 1;
    let src_ptr = bytes.as_ptr();
    let dst_ptr = unsafe { out.as_mut_ptr().add(start) };

    if width <= 7 {
        let step = width as usize;
        let mut row = 0usize;
        let mut byte_pos = 0usize;
        #[cfg(target_arch = "aarch64")]
        let v_min = unsafe { core::arch::aarch64::vdupq_n_u16(min) };
        #[cfg(not(target_arch = "aarch64"))]
        let min_splat = (min as u64).wrapping_mul(0x0001_0001_0001_0001);

        while row < count {
            let v = if byte_pos + 8 <= bytes.len() {
                u64::from_le(unsafe { src_ptr.add(byte_pos).cast::<u64>().read_unaligned() })
            } else {
                u64::from_le(unsafe {
                    read_u128_partial(src_ptr.add(byte_pos), bytes.len() - byte_pos) as u64
                })
            };
            let u8x8 = (v & m)
                | (((v >> w) & m) << 8)
                | (((v >> (2 * w)) & m) << 16)
                | (((v >> (3 * w)) & m) << 24)
                | (((v >> (4 * w)) & m) << 32)
                | (((v >> (5 * w)) & m) << 40)
                | (((v >> (6 * w)) & m) << 48)
                | (((v >> (7 * w)) & m) << 56);
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let v8 = vreinterpret_u8_u64(vdup_n_u64(u8x8));
                let v16 = vaddq_u16(vmovl_u8(v8), v_min);
                vst1q_u16(dst_ptr.add(row), v16);
            }
            #[cfg(not(target_arch = "aarch64"))]
            unsafe {
                let lo = (u8x8 & 0xff)
                    | (((u8x8 >> 8) & 0xff) << 16)
                    | (((u8x8 >> 16) & 0xff) << 32)
                    | (((u8x8 >> 24) & 0xff) << 48);
                let hi = ((u8x8 >> 32) & 0xff)
                    | (((u8x8 >> 40) & 0xff) << 16)
                    | (((u8x8 >> 48) & 0xff) << 32)
                    | (((u8x8 >> 56) & 0xff) << 48);
                dst_ptr
                    .add(row)
                    .cast::<u64>()
                    .write_unaligned(lo.wrapping_add(min_splat).to_le());
                dst_ptr
                    .add(row + 4)
                    .cast::<u64>()
                    .write_unaligned(hi.wrapping_add(min_splat).to_le());
            }
            row += 8;
            byte_pos += step;
        }
        unsafe {
            out.set_len(start + count);
        }
        return Ok(start);
    }

    let min_splat = (min as u64).wrapping_mul(0x0001_0001_0001_0001);
    let step_bits = (width as usize) * 4;
    let mut row = 0usize;
    let mut bit_pos = 0usize;

    while row < count {
        let byte_pos = bit_pos >> 3;
        let bit_off = (bit_pos & 7) as u32;
        let raw = if byte_pos + 8 <= bytes.len() {
            u64::from_le(unsafe { src_ptr.add(byte_pos).cast::<u64>().read_unaligned() })
        } else {
            u64::from_le(unsafe {
                read_u128_partial(src_ptr.add(byte_pos), bytes.len() - byte_pos) as u64
            })
        };
        let v = raw >> bit_off;
        let u16x4 = (v & m)
            | (((v >> w) & m) << 16)
            | (((v >> (2 * w)) & m) << 32)
            | (((v >> (3 * w)) & m) << 48);
        let out_word = u16x4.wrapping_add(min_splat);
        unsafe {
            dst_ptr.add(row).cast::<u64>().write_unaligned(out_word.to_le());
        }
        row += 4;
        bit_pos += step_bits;
    }
    unsafe {
        out.set_len(start + count);
    }
    Ok(start)
}

#[inline(always)]
unsafe fn write_hex8_from_4bytes(dst: *mut u8, src: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vreinterpret_u8_u32(vdup_n_u32(src.cast::<u32>().read_unaligned()));
        let hi = vshr_n_u8(v, 4);
        let lo = vand_u8(v, vdup_n_u8(0x0f));
        let zipped = vzip1_u8(hi, lo);
        let hex_lut = vld1q_u8(HEX.as_ptr());
        let ascii = vqtbl1_u8(hex_lut, zipped);
        vst1_u8(dst, ascii);
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        let b0 = *HEX_BYTE_LUT.get_unchecked(*src as usize) as u64;
        let b1 = *HEX_BYTE_LUT.get_unchecked(*src.add(1) as usize) as u64;
        let b2 = *HEX_BYTE_LUT.get_unchecked(*src.add(2) as usize) as u64;
        let b3 = *HEX_BYTE_LUT.get_unchecked(*src.add(3) as usize) as u64;
        let lo = b0 | (b1 << 16) | (b2 << 32) | (b3 << 48);
        dst.cast::<u64>().write_unaligned(lo);
    }
}

#[inline(always)]
unsafe fn write_hex16_from_8bytes(dst: *mut u8, src: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v = vld1_u8(src);
        let hi = vshr_n_u8(v, 4);
        let lo = vand_u8(v, vdup_n_u8(0x0f));
        let zipped = vcombine_u8(vzip1_u8(hi, lo), vzip2_u8(hi, lo));
        let hex_lut = vld1q_u8(HEX.as_ptr());
        let ascii = vqtbl1q_u8(hex_lut, zipped);
        vst1q_u8(dst, ascii);
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        write_hex8_from_4bytes(dst, src);
        write_hex8_from_4bytes(dst.add(8), src.add(4));
    }
}

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
                let entries_len = cursor.uleb()?;
                if entries_len == 0
                    || (tag == MODEL_DICTIONARY && entries_len >= count)
                    || entries_len > count
                {
                    return Err(invalid("column dictionary cardinality"));
                }
                let slice_offset = scratch.slice_arena.len();
                scratch.slice_arena.reserve(entries_len);
                let mut max_len = 0usize;
                let mut first_entry: &'a [u8] = &[];
                let u8_offset = if tag == MODEL_FRONT_DICT {
                    if entries_len < 2 || entries_len > 256 {
                        return Err(invalid("column front dict cardinality"));
                    }
                    let mode_byte = cursor.byte()?;
                    let dict_mode = mode_byte & 0x03;
                    let idx_mode = mode_byte >> 2;
                    if idx_mode > 3 {
                        return Err(invalid("column front dict mode"));
                    }
                    let width = bit_width((entries_len - 1) as u64);
                    let mut lz_dict_cursor = Cursor::new(&[]);
                    let use_lz_cursor = if dict_mode == 2 {
                        let lz_len = cursor.uleb()?;
                        let lz_bytes = cursor.take(lz_len)?;
                        let mut lzc = Cursor::new(lz_bytes);
                        if lzc.byte()? != FAST_LZ_PACKET {
                            return Err(invalid("column front dict lz selector"));
                        }
                        let mut probe = Cursor::new(lzc.bytes);
                        probe.pos = lzc.pos;
                        let raw_len = probe.uleb()?;
                        if raw_len == 0 || raw_len > 65536 {
                            return Err(invalid("column front dict lz length"));
                        }
                        let arena_start = scratch.raw_arena.len();
                        if arena_start + raw_len > scratch.raw_arena.capacity() {
                            return Err(invalid("column front dict lz arena overflow"));
                        }
                        unsafe {
                            scratch.raw_arena.set_len(arena_start + raw_len);
                        }
                        unpack_fast_lz_into(
                            &mut lzc,
                            &mut scratch.raw_arena[arena_start..arena_start + raw_len],
                        )?;
                        let unpacked_slice: &'a [u8] = unsafe {
                            std::slice::from_raw_parts(
                                scratch.raw_arena.as_ptr().add(arena_start),
                                raw_len,
                            )
                        };
                        lz_dict_cursor = Cursor::new(unpacked_slice);
                        true
                    } else {
                        false
                    };
                    scratch.slice_arena.reserve(entries_len);
                    if dict_mode == 0 {
                        let mut prev_entry: Option<&'a [u8]> = None;
                        for _ in 0..entries_len {
                            let len = cursor.uleb()?;
                            let entry = cursor.take(len)?;
                            if prev_entry.is_some_and(|last| last >= entry) {
                                return Err(invalid("column dictionary ordering"));
                            }
                            prev_entry = Some(entry);
                            if len > max_len {
                                max_len = len;
                            }
                            scratch.slice_arena.push((entry.as_ptr(), len));
                        }
                    } else if dict_mode == 3 {
                        let (pal, pal_len, pal_mixed) = read_front_dict_palette(&mut cursor)?;
                        let len0 = cursor.uleb()?;
                        let raw_first = cursor.take(len0)?;
                        let first_start = scratch.raw_arena.len();
                        if first_start + len0 + 32 > scratch.raw_arena.capacity() {
                            return Err(invalid("column front dictionary arena overflow"));
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                raw_first.as_ptr(),
                                scratch.raw_arena.as_mut_ptr().add(first_start),
                                len0,
                            );
                            scratch.raw_arena.set_len(first_start + len0);
                            first_entry = std::slice::from_raw_parts(
                                scratch.raw_arena.as_ptr().add(first_start),
                                len0,
                            );
                        }
                        max_len = len0;
                        scratch.slice_arena.push((first_entry.as_ptr(), len0));
                        let mut prev_entry: &'a [u8] = first_entry;
                        #[cfg(target_arch = "aarch64")]
                        let v_pal = unsafe { core::arch::aarch64::vld1q_u8(pal.as_ptr()) };
                        for _ in 1..entries_len {
                            let (lcp, lcs, raw_hdr) = if cursor.pos + 3 <= cursor.bytes.len() {
                                let p = unsafe { cursor.bytes.as_ptr().add(cursor.pos) };
                                let b0 = unsafe { *p };
                                let b1 = unsafe { *p.add(1) };
                                let b2 = unsafe { *p.add(2) };
                                if (b0 | b1 | b2) < 0x80 {
                                    cursor.pos += 3;
                                    (b0 as usize, b1 as usize, b2 as usize)
                                } else {
                                    (cursor.uleb()?, cursor.uleb()?, cursor.uleb()?)
                                }
                            } else {
                                (cursor.uleb()?, cursor.uleb()?, cursor.uleb()?)
                            };
                            let (mid_len, is_raw) = if pal_mixed {
                                (raw_hdr >> 1, (raw_hdr & 1) != 0)
                            } else {
                                (raw_hdr, false)
                            };
                            if mid_len > 65536 {
                                return Err(invalid("column front dict entry length"));
                            }
                            let packed_mid = if is_raw {
                                cursor.take(mid_len)?
                            } else if pal_len <= 16 {
                                cursor.take((mid_len + 1) >> 1)?
                            } else {
                                cursor.take((mid_len * 5 + 7) >> 3)?
                            };
                            if lcp + lcs > prev_entry.len() {
                                return Err(invalid("column front dictionary bounds"));
                            }
                            let total_len = lcp + mid_len + lcs;
                            let arena_start = scratch.raw_arena.len();
                            if arena_start + total_len + 32 > scratch.raw_arena.capacity() {
                                return Err(invalid("column front dictionary arena overflow"));
                            }
                            #[cfg(target_arch = "aarch64")]
                            let handled_fast = if !is_raw
                                && pal_len <= 16
                                && lcp <= 32
                                && mid_len <= 32
                                && lcs <= 16
                            {
                                unsafe {
                                    use core::arch::aarch64::*;
                                    let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                    let p_ptr = prev_entry.as_ptr();
                                    let lcs_v0 = p_ptr
                                        .add(prev_entry.len() - lcs)
                                        .cast::<u128>()
                                        .read_unaligned();
                                    dst.cast::<u128>()
                                        .write_unaligned(p_ptr.cast::<u128>().read_unaligned());
                                    if lcp > 16 {
                                        dst.add(16).cast::<u128>().write_unaligned(
                                            p_ptr.add(16).cast::<u128>().read_unaligned(),
                                        );
                                    }
                                    if mid_len > 0 {
                                        let packed_len = packed_mid.len();
                                        if (mid_len & 1) != 0
                                            && (*packed_mid.get_unchecked(packed_len - 1) & 0x0f)
                                                != 0
                                        {
                                            return Err(invalid("column front dict nibble padding"));
                                        }
                                        let raw128 =
                                            read_u128_partial(packed_mid.as_ptr(), packed_len);
                                        let raw64 = raw128 as u64;
                                        let v8 = vreinterpret_u8_u64(vdup_n_u64(raw64.to_le()));
                                        let hi = vshr_n_u8(v8, 4);
                                        let lo = vand_u8(v8, vdup_n_u8(0x0f));
                                        let zipped =
                                            vcombine_u8(vzip1_u8(hi, lo), vzip2_u8(hi, lo));
                                        if pal_len < 16 && vmaxvq_u8(zipped) >= pal_len {
                                            return Err(invalid(
                                                "column front dict nibble out of bounds",
                                            ));
                                        }
                                        vst1q_u8(dst.add(lcp), vqtbl1q_u8(v_pal, zipped));
                                        if mid_len > 16 {
                                            let raw64_hi = (raw128 >> 64) as u64;
                                            let v8_hi =
                                                vreinterpret_u8_u64(vdup_n_u64(raw64_hi.to_le()));
                                            let hi2 = vshr_n_u8(v8_hi, 4);
                                            let lo2 = vand_u8(v8_hi, vdup_n_u8(0x0f));
                                            let zipped2 =
                                                vcombine_u8(vzip1_u8(hi2, lo2), vzip2_u8(hi2, lo2));
                                            if pal_len < 16 && vmaxvq_u8(zipped2) >= pal_len {
                                                return Err(invalid(
                                                    "column front dict nibble out of bounds",
                                                ));
                                            }
                                            vst1q_u8(
                                                dst.add(lcp + 16),
                                                vqtbl1q_u8(v_pal, zipped2),
                                            );
                                        }
                                    }
                                    dst.add(lcp + mid_len)
                                        .cast::<u128>()
                                        .write_unaligned(lcs_v0);
                                    scratch.raw_arena.set_len(arena_start + total_len);
                                }
                                true
                            } else {
                                false
                            };
                            #[cfg(not(target_arch = "aarch64"))]
                            let handled_fast = false;
                            if !handled_fast {
                                unsafe {
                                    let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                    if lcp > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr(),
                                            dst,
                                            lcp,
                                        );
                                    }
                                    if mid_len > 0 {
                                        if is_raw {
                                            std::ptr::copy_nonoverlapping(
                                                packed_mid.as_ptr(),
                                                dst.add(lcp),
                                                mid_len,
                                            );
                                        } else {
                                            unpack_nibble_bytes_into(
                                                packed_mid,
                                                mid_len,
                                                &pal,
                                                pal_len,
                                                dst.add(lcp),
                                            )?;
                                        }
                                    }
                                    if lcs > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr().add(prev_entry.len() - lcs),
                                            dst.add(lcp + mid_len),
                                            lcs,
                                        );
                                    }
                                    scratch.raw_arena.set_len(arena_start + total_len);
                                }
                            }
                            let entry: &'a [u8] = unsafe {
                                std::slice::from_raw_parts(
                                    scratch.raw_arena.as_ptr().add(arena_start),
                                    total_len,
                                )
                            };
                            let ord_ok = if lcp < prev_entry.len() && lcp < total_len {
                                let a = unsafe { *prev_entry.as_ptr().add(lcp) };
                                let b = unsafe { *entry.as_ptr().add(lcp) };
                                if a != b {
                                    a < b
                                } else {
                                    prev_entry[lcp + 1..] < entry[lcp + 1..]
                                }
                            } else {
                                prev_entry.len() < total_len
                            };
                            if !ord_ok {
                                return Err(invalid("column dictionary ordering"));
                            }
                            prev_entry = entry;
                            if total_len > max_len {
                                max_len = total_len;
                            }
                            scratch.slice_arena.push((entry.as_ptr(), total_len));
                        }
                    } else {
                        let dcur = if use_lz_cursor {
                            &mut lz_dict_cursor
                        } else {
                            &mut cursor
                        };
                        let len0 = dcur.uleb()?;
                        let raw_first = dcur.take(len0)?;
                        let first_start = scratch.raw_arena.len();
                        if first_start + len0 + 32 > scratch.raw_arena.capacity() {
                            return Err(invalid("column front dictionary arena overflow"));
                        }
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                raw_first.as_ptr(),
                                scratch.raw_arena.as_mut_ptr().add(first_start),
                                len0,
                            );
                            scratch.raw_arena.set_len(first_start + len0);
                            first_entry = std::slice::from_raw_parts(
                                scratch.raw_arena.as_ptr().add(first_start),
                                len0,
                            );
                        }
                        max_len = len0;
                        scratch.slice_arena.push((first_entry.as_ptr(), len0));
                        let mut prev_entry: &'a [u8] = first_entry;
                        for _ in 1..entries_len {
                            let (lcp, lcs, mid_len) = if dcur.pos + 3 <= dcur.bytes.len() {
                                let p = unsafe { dcur.bytes.as_ptr().add(dcur.pos) };
                                let b0 = unsafe { *p };
                                let b1 = unsafe { *p.add(1) };
                                let b2 = unsafe { *p.add(2) };
                                if (b0 | b1 | b2) < 0x80 {
                                    dcur.pos += 3;
                                    (b0 as usize, b1 as usize, b2 as usize)
                                } else {
                                    (dcur.uleb()?, dcur.uleb()?, dcur.uleb()?)
                                }
                            } else {
                                (dcur.uleb()?, dcur.uleb()?, dcur.uleb()?)
                            };
                            let mid = dcur.take(mid_len)?;
                            if lcp + lcs > prev_entry.len() {
                                return Err(invalid("column front dictionary bounds"));
                            }
                            let total_len = lcp + mid_len + lcs;
                            let arena_start = scratch.raw_arena.len();
                            if arena_start + total_len + 32 > scratch.raw_arena.capacity() {
                                return Err(invalid("column front dictionary arena overflow"));
                            }
                            unsafe {
                                let dst = scratch.raw_arena.as_mut_ptr().add(arena_start);
                                if lcp <= 32 && mid_len <= 32 && lcs <= 16 {
                                    let p_ptr = prev_entry.as_ptr();
                                    let lcs_v0 = p_ptr
                                        .add(prev_entry.len() - lcs)
                                        .cast::<u128>()
                                        .read_unaligned();
                                    dst.cast::<u128>()
                                        .write_unaligned(p_ptr.cast::<u128>().read_unaligned());
                                    if lcp > 16 {
                                        dst.add(16).cast::<u128>().write_unaligned(
                                            p_ptr.add(16).cast::<u128>().read_unaligned(),
                                        );
                                    }
                                    if mid_len > 0 {
                                        let m_ptr = mid.as_ptr();
                                        dst.add(lcp)
                                            .cast::<u128>()
                                            .write_unaligned(read_u128_partial(m_ptr, mid_len));
                                        if mid_len > 16 {
                                            dst.add(lcp + 16).cast::<u128>().write_unaligned(
                                                read_u128_partial(m_ptr.add(16), mid_len - 16),
                                            );
                                        }
                                    }
                                    dst.add(lcp + mid_len)
                                        .cast::<u128>()
                                        .write_unaligned(lcs_v0);
                                } else {
                                    if lcp > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr(),
                                            dst,
                                            lcp,
                                        );
                                    }
                                    if mid_len > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            mid.as_ptr(),
                                            dst.add(lcp),
                                            mid_len,
                                        );
                                    }
                                    if lcs > 0 {
                                        std::ptr::copy_nonoverlapping(
                                            prev_entry.as_ptr().add(prev_entry.len() - lcs),
                                            dst.add(lcp + mid_len),
                                            lcs,
                                        );
                                    }
                                }
                                scratch.raw_arena.set_len(arena_start + total_len);
                            }
                            let entry: &'a [u8] = unsafe {
                                std::slice::from_raw_parts(
                                    scratch.raw_arena.as_ptr().add(arena_start),
                                    total_len,
                                )
                            };
                            let ord_ok = if lcp < prev_entry.len() && lcp < total_len {
                                let a = unsafe { *prev_entry.as_ptr().add(lcp) };
                                let b = unsafe { *entry.as_ptr().add(lcp) };
                                if a != b {
                                    a < b
                                } else {
                                    prev_entry[lcp + 1..] < entry[lcp + 1..]
                                }
                            } else {
                                prev_entry.len() < total_len
                            };
                            if !ord_ok {
                                return Err(invalid("column dictionary ordering"));
                            }
                            prev_entry = entry;
                            if total_len > max_len {
                                max_len = total_len;
                            }
                            scratch.slice_arena.push((entry.as_ptr(), total_len));
                        }
                        if use_lz_cursor && !dcur.done() {
                            return Err(invalid("column front dict lz trailing"));
                        }
                    }
                    unpack_front_dict_indices(
                        &mut cursor,
                        count,
                        entries_len,
                        width,
                        idx_mode,
                        &mut scratch.u8_arena,
                    )?
                } else {
                    scratch.slice_arena.reserve(entries_len);
                    let mut prev_entry: Option<&'a [u8]> = None;
                    for i in 0..entries_len {
                        let len = cursor.uleb()?;
                        let entry = cursor.take(len)?;
                        if prev_entry.is_some_and(|last| last >= entry) {
                            return Err(invalid("column dictionary ordering"));
                        }
                        prev_entry = Some(entry);
                        if i == 0 {
                            first_entry = entry;
                        }
                        if len > max_len {
                            max_len = len;
                        }
                        scratch.slice_arena.push((entry.as_ptr(), len));
                    }
                    let width = cursor.byte()?;
                    if width != bit_width((entries_len - 1) as u64) {
                        return Err(invalid("column dictionary width"));
                    }
                    let packed_len = bit_bytes(count, width)?;
                    let packed_bytes = cursor.take(packed_len)?;
                    if !cursor.done() {
                        return Err(invalid("column model trailing bytes"));
                    }
                    if entries_len == 1 {
                        let _ = FastBitReader::new_validated(packed_bytes, count, width)?;
                        scratch.slice_arena.truncate(slice_offset);
                        return Ok(Self {
                            const_full: ConstValue::Slice(first_entry),
                            const_lead: LeadValue::Empty,
                            const_trail: b"",
                            dict_slices: None,
                            max_emit_len: 0,
                            stream: ColumnStream::Raw {
                                cursor: Cursor::new(&[]),
                            },
                        });
                    }
                    unpack_and_validate_u8(
                        packed_bytes,
                        count,
                        width,
                        0,
                        entries_len,
                        &mut scratch.u8_arena,
                    )?
                };
                let piece_offset = scratch.piece_arena.len();
                scratch.piece_arena.reserve(entries_len);
                unsafe {
                    scratch.piece_arena.set_len(piece_offset + entries_len);
                }
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::Empty,
                    const_trail: b"",
                    dict_slices: Some((slice_offset, entries_len)),
                    max_emit_len: max_len,
                    stream: ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        max_len,
                    },
                })
            }
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
                let width = cursor.uleb()?;
                if width == 0 || width > 64 {
                    return Err(invalid("column hex width"));
                }
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
                Ok(Self {
                    const_full: ConstValue::None,
                    const_lead: LeadValue::QuotePrefix(prefix),
                    const_trail: b"\"",
                    dict_slices: None,
                    max_emit_len: width * 2,
                    stream,
                })
            }
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
            MODEL_EPOCH_COUNTER => Self::parse_epoch_counter(cursor, count, scratch),
            MODEL_SEGMENT_DICT => Self::parse_segment_dict(cursor, count, scratch),
            MODEL_GF2_AFFINE => Self::parse_gf2_affine(cursor, count, scratch),
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
        let mut dict_v0 = core::mem::MaybeUninit::<[u128; MAX_SEGMENT_DICT * 2]>::uninit();
        let mut dict_v1 = core::mem::MaybeUninit::<[u128; MAX_SEGMENT_DICT * 2]>::uninit();
        let mut dict_len = core::mem::MaybeUninit::<[u8; MAX_SEGMENT_DICT * 2]>::uninit();
        let dict_v0_ptr = dict_v0.as_mut_ptr().cast::<u128>();
        let dict_v1_ptr = dict_v1.as_mut_ptr().cast::<u128>();
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
                *dict_v0_ptr.add(i) = u128::from_le_bytes(tmp_buf[0..16].try_into().unwrap());
                *dict_v1_ptr.add(i) = u128::from_le_bytes(tmp_buf[16..32].try_into().unwrap());
                *dict_len_ptr.add(i) = len as u8;
            }

            let mut sep_buf = [0u8; 32];
            sep_buf[0] = b';';
            sep_buf[1] = b' ';
            sep_buf[2..2 + len].copy_from_slice(entry);
            let sep_idx = MAX_SEGMENT_DICT + i;
            unsafe {
                *dict_v0_ptr.add(sep_idx) =
                    u128::from_le_bytes(sep_buf[0..16].try_into().unwrap());
                *dict_v1_ptr.add(sep_idx) =
                    u128::from_le_bytes(sep_buf[16..32].try_into().unwrap());
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
            .saturating_sub(count)
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

        if !dec_u8_ptr.is_null() && !pads_u8_ptr.is_null() {
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
                        dst_ptr
                            .cast::<u128>()
                            .write_unaligned(*dict_v0_ptr.add(idx));
                        dst_ptr
                            .add(16)
                            .cast::<u128>()
                            .write_unaligned(*dict_v1_ptr.add(idx));
                        dst_ptr = dst_ptr.add(l);
                    }
                } else if code == end_code {
                    if term_count >= count {
                        return Err(invalid("column segment terminator count"));
                    }
                    let p = unsafe { *pads_u8_ptr.add(term_count) as u64 };
                    or_pad |= p;
                    let pad = p as usize;
                    if pad != 0 {
                        unsafe {
                            dst_ptr
                                .cast::<u128>()
                                .write_unaligned(0x2020_2020_2020_2020_2020_2020_2020_2020);
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
                    let d = unsafe { *dec_u8_ptr.add(dec_used) as u64 };
                    dec_used += 1;
                    or_dec |= d;
                    unsafe {
                        if sep_offset != 0 {
                            dst_ptr
                                .cast::<u16>()
                                .write_unaligned(u16::from_le_bytes(*b"; "));
                            dst_ptr = dst_ptr.add(2);
                        }
                        sep_offset = MAX_SEGMENT_DICT;
                        let entry = *DEC_10K.get_unchecked(d as usize);
                        dst_ptr.cast::<u32>().write_unaligned(entry as u32);
                        dst_ptr = dst_ptr.add((entry >> 32) as usize);
                    }
                }
            }
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
                        dst_ptr
                            .cast::<u128>()
                            .write_unaligned(*dict_v0_ptr.add(idx));
                        dst_ptr
                            .add(16)
                            .cast::<u128>()
                            .write_unaligned(*dict_v1_ptr.add(idx));
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

        Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::QuoteOnly,
            const_trail: b"\"",
            dict_slices: None,
            max_emit_len: max_val_len,
            stream: ColumnStream::SegmentDict {
                str_u8_offset,
                len_u16_offset,
            },
        })
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

#[derive(Clone, Copy)]
enum BoundStream<'a> {
    Raw {
        cursor: Cursor<'a>,
    },
    Dictionary16 {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    Dictionary32 {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    Dictionary48 {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    DictionaryPair32 {
        entries0: *const FastPiece,
        entries1: *const FastPiece,
        idx_ptr0: *const u8,
        idx_delta: isize,
    },
    DictionaryPair48 {
        entries0: *const FastPiece,
        entries1: *const FastPiece,
        idx_ptr0: *const u8,
        idx_delta: isize,
    },
    DictionaryQuad32 {
        entries0: *const FastPiece,
        idx_ptr0: *const u8,
        p_off: [u16; 3],
        u_off: [u16; 3],
    },
    DictionaryQuad48 {
        entries0: *const FastPiece,
        idx_ptr0: *const u8,
        p_off: [u16; 3],
        u_off: [u16; 3],
    },
    DictionaryLong {
        entries: *const FastPiece,
        idx_ptr: *const u8,
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
        val_ptr: *const u16,
    },
    DecimalPacked {
        min: u64,
        reader: FastBitReader<'a>,
    },
    DecimalValues {
        val_ptr: *const u64,
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
        min_ptr: *const u8,
        sec_ptr: *const u8,
    },
    TimestampFull {
        ptr: *const u8,
        stride: usize,
    },
    Copy16 {
        ptr: *const u8,
    },
    SegmentDict {
        str_ptr: *const u8,
        len_ptr: *const u16,
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
struct ExecColumn<'a> {
    stream: BoundStream<'a>,
    after: FastPiece,
}

impl<'a> ExecColumn<'a> {
    #[inline(always)]
    unsafe fn emit(&mut self, mut dst: *mut u8) -> *mut u8 {
        unsafe {
            match &mut self.stream {
                BoundStream::Dictionary16 { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    return (*entries.add(idx)).emit_16(dst);
                }
                BoundStream::Dictionary32 { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    return (*entries.add(idx)).emit_32(dst);
                }
                BoundStream::Dictionary48 { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    return (*entries.add(idx)).emit_48_exact(dst);
                }
                BoundStream::DictionaryPair32 {
                    entries0,
                    entries1,
                    idx_ptr0,
                    idx_delta,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let idx0 = *p as usize;
                    let idx1 = *p.wrapping_offset(*idx_delta) as usize;
                    let p0 = &*entries0.add(idx0);
                    let p1 = &*entries1.add(idx1);
                    dst = p0.emit_32(dst);
                    return p1.emit_32(dst);
                }
                BoundStream::DictionaryPair48 {
                    entries0,
                    entries1,
                    idx_ptr0,
                    idx_delta,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let idx0 = *p as usize;
                    let idx1 = *p.wrapping_offset(*idx_delta) as usize;
                    let p0 = &*entries0.add(idx0);
                    let p1 = &*entries1.add(idx1);
                    dst = p0.emit_48_exact(dst);
                    return p1.emit_48_exact(dst);
                }
                BoundStream::DictionaryQuad32 {
                    entries0,
                    idx_ptr0,
                    p_off,
                    u_off,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let e0 = *entries0;
                    let i0 = *p as usize;
                    let i1 = *p.add(u_off[0] as usize) as usize;
                    let i2 = *p.add(u_off[1] as usize) as usize;
                    let i3 = *p.add(u_off[2] as usize) as usize;
                    dst = (*e0.add(i0)).emit_32(dst);
                    dst = (*e0.add(p_off[0] as usize + i1)).emit_32(dst);
                    dst = (*e0.add(p_off[1] as usize + i2)).emit_32(dst);
                    return (*e0.add(p_off[2] as usize + i3)).emit_32(dst);
                }
                BoundStream::DictionaryQuad48 {
                    entries0,
                    idx_ptr0,
                    p_off,
                    u_off,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let e0 = *entries0;
                    let i0 = *p as usize;
                    let i1 = *p.add(u_off[0] as usize) as usize;
                    let i2 = *p.add(u_off[1] as usize) as usize;
                    let i3 = *p.add(u_off[2] as usize) as usize;
                    dst = (*e0.add(i0)).emit_48_exact(dst);
                    dst = (*e0.add(p_off[0] as usize + i1)).emit_48_exact(dst);
                    dst = (*e0.add(p_off[1] as usize + i2)).emit_48_exact(dst);
                    return (*e0.add(p_off[2] as usize + i3)).emit_48_exact(dst);
                }
                BoundStream::DictionaryLong { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    dst = (*entries.add(idx)).emit_48(dst);
                }
                BoundStream::DecimalSmall { val_ptr } => {
                    let val = **val_ptr as usize;
                    *val_ptr = val_ptr.add(1);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                }
                BoundStream::DecimalAffineSmall { current, step } => {
                    let val = *current as usize;
                    *current = current.wrapping_add(*step);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                }
                BoundStream::DecimalValues { val_ptr } => {
                    let val = **val_ptr;
                    *val_ptr = val_ptr.add(1);
                    let len = write_decimal_fast(dst, val);
                    dst = dst.add(len);
                }
                BoundStream::DecimalAffine { current, step } => {
                    let val = *current;
                    *current = current.wrapping_add(*step);
                    let len = write_decimal_fast(dst, val);
                    dst = dst.add(len);
                }
                BoundStream::DecimalPacked { min, reader } => {
                    let val = *min + reader.next_u64();
                    let len = write_decimal_fast(dst, val);
                    dst = dst.add(len);
                }
                BoundStream::TimestampMinSec { min_ptr, sec_ptr } => {
                    let min = **min_ptr as usize;
                    *min_ptr = min_ptr.add(1);
                    let sec = **sec_ptr as usize;
                    *sec_ptr = sec_ptr.add(1);
                    let min_16 =
                        TWO_DIGITS.as_ptr().add(min * 2).cast::<u16>().read_unaligned() as u64;
                    let sec_16 =
                        TWO_DIGITS.as_ptr().add(sec * 2).cast::<u16>().read_unaligned() as u64;
                    let word = min_16 | ((b':' as u64) << 16) | (sec_16 << 24);
                    dst.cast::<u64>().write_unaligned(word);
                    dst = dst.add(5);
                }
                BoundStream::TimestampFull { ptr, stride } => {
                    let p = *ptr;
                    *ptr = p.add(1);
                    let s = *stride;
                    let m = *p as usize;
                    let d = *p.add(s) as usize;
                    let h = *p.add(s * 2) as usize;
                    let min = *p.add(s * 3) as usize;
                    let sec = *p.add(s * 4) as usize;

                    let m_16 =
                        TWO_DIGITS.as_ptr().add(m * 2).cast::<u16>().read_unaligned() as u64;
                    let d_16 =
                        TWO_DIGITS.as_ptr().add(d * 2).cast::<u16>().read_unaligned() as u64;
                    let h_16 =
                        TWO_DIGITS.as_ptr().add(h * 2).cast::<u16>().read_unaligned() as u64;
                    let min_16 =
                        TWO_DIGITS.as_ptr().add(min * 2).cast::<u16>().read_unaligned() as u64;
                    let sec_16 =
                        TWO_DIGITS.as_ptr().add(sec * 2).cast::<u16>().read_unaligned() as u64;

                    let w0 = m_16
                        | ((b'-' as u64) << 16)
                        | (d_16 << 24)
                        | ((b'T' as u64) << 40)
                        | (h_16 << 48);
                    let w1 = h_16
                        | ((b':' as u64) << 16)
                        | (min_16 << 24)
                        | ((b':' as u64) << 40)
                        | (sec_16 << 48);
                    dst.cast::<u64>().write_unaligned(w0);
                    dst.add(6).cast::<u64>().write_unaligned(w1);
                    dst = dst.add(14);
                }
                BoundStream::Copy16 { ptr } => {
                    let p = *ptr;
                    *ptr = p.add(16);
                    dst.cast::<u128>()
                        .write_unaligned(p.cast::<u128>().read_unaligned());
                    dst = dst.add(16);
                }
                BoundStream::HexSuffix4 { ptr } => {
                    write_hex8_from_4bytes(dst, *ptr);
                    *ptr = ptr.add(4);
                    dst = dst.add(8);
                }
                BoundStream::HexSuffix8 { ptr } => {
                    write_hex16_from_8bytes(dst, *ptr);
                    *ptr = ptr.add(8);
                    dst = dst.add(16);
                }
                BoundStream::HexSuffix { cursor, width } => {
                    let w = *width;
                    let raw = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + w);
                    cursor.pos += w;
                    for (i, &b) in raw.iter().enumerate() {
                        let pair = *HEX_BYTE_LUT.get_unchecked(b as usize);
                        dst.add(i * 2).cast::<u16>().write_unaligned(pair);
                    }
                    dst = dst.add(w * 2);
                }
                BoundStream::SegmentDict { str_ptr, len_ptr } => {
                    let len = **len_ptr as usize;
                    *len_ptr = len_ptr.add(1);
                    let p = *str_ptr;
                    *str_ptr = p.add(len);
                    dst.cast::<u128>()
                        .write_unaligned(p.cast::<u128>().read_unaligned());
                    dst.add(16)
                        .cast::<u128>()
                        .write_unaligned(p.add(16).cast::<u128>().read_unaligned());
                    dst.add(32)
                        .cast::<u128>()
                        .write_unaligned(p.add(32).cast::<u128>().read_unaligned());
                    if len > 48 {
                        dst.add(48)
                            .cast::<u128>()
                            .write_unaligned(p.add(48).cast::<u128>().read_unaligned());
                        dst.add(64)
                            .cast::<u128>()
                            .write_unaligned(p.add(64).cast::<u128>().read_unaligned());
                        if len > 80 {
                            FastPiece::emit_tail_cold(p, dst, len);
                        }
                    }
                    dst = dst.add(len);
                }
                BoundStream::Raw { cursor } => {
                    let len = cursor.uleb().unwrap_unchecked();
                    let val = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + len);
                    cursor.pos += len;
                    std::ptr::copy_nonoverlapping(val.as_ptr(), dst, len);
                    dst = dst.add(len);
                }
                BoundStream::PeriodicString { cursor, period } => {
                    let len = cursor.uleb().unwrap_unchecked();
                    let p = *period;
                    let seed_len = len.min(p);
                    let seed = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + seed_len);
                    cursor.pos += seed_len;
                    std::ptr::copy_nonoverlapping(seed.as_ptr(), dst, seed_len);
                    for idx in seed_len..len {
                        *dst.add(idx) = *seed.get_unchecked(idx % p);
                    }
                    dst = dst.add(len);
                }
                BoundStream::PeriodicHex { cursor, period } => {
                    let len = cursor.uleb().unwrap_unchecked();
                    let p = *period;
                    let seed_len = len.min(p);
                    let packed_len = seed_len.div_ceil(2);
                    let packed = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + packed_len);
                    cursor.pos += packed_len;
                    for (idx, &byte) in packed.iter().enumerate() {
                        *dst.add(idx * 2) = HEX[(byte >> 4) as usize];
                        if idx * 2 + 1 < seed_len {
                            *dst.add(idx * 2 + 1) = HEX[(byte & 15) as usize];
                        }
                    }
                    for idx in seed_len..len {
                        *dst.add(idx) = *dst.add(idx % p);
                    }
                    dst = dst.add(len);
                }
            }
            if self.after.len == 0 {
                dst
            } else if self.after.len <= 16 {
                self.after.emit_16(dst)
            } else if self.after.len <= 32 {
                self.after.emit_32(dst)
            } else {
                self.after.emit_48(dst)
            }
        }
    }
}

#[derive(Clone, Copy)]
struct UnboundExecColumn<'a> {
    stream: ColumnStream<'a>,
    after_range: (usize, usize),
}

#[derive(Clone, Copy, Default)]
struct UnboundTemplate {
    head_range: (usize, usize),
    col_start: usize,
    col_end: usize,
    row_budget: usize,
}

#[derive(Clone, Copy)]
struct PreparedTemplate {
    head: FastPiece,
    col_ptr: *mut ExecColumn<'static>,
    col_len: usize,
    row_budget: usize,
}

impl PreparedTemplate {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            head: FastPiece::empty(),
            col_ptr: std::ptr::null_mut(),
            col_len: 0,
            row_budget: 0,
        }
    }
}

impl UnboundTemplate {
    fn build<'a>(
        segments: &[&'a [u8]],
        cols: &[PreparedColumn<'a>],
        scratch: &mut DecodeScratch,
    ) -> Self {
        // Step 1: Fold constant (`width == 0` / `step == 0`) columns directly into
        // `scratch.str_arena`, and absorb `const_lead` / `const_trail` for active columns.
        let mut seg_ranges = [(0usize, 0usize); MAX_SLOTS + 1];
        let seg0_start = scratch.str_arena.len();
        scratch.str_arena.extend_from_slice(segments[0]);
        seg_ranges[0] = (seg0_start, scratch.str_arena.len());
        let mut active_len = 0usize;
        let mut active_cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
        let active_cols: &mut [PreparedColumn; MAX_SLOTS] =
            unsafe { active_cols_uninit.assume_init_mut() };

        for (slot, &col) in cols.iter().enumerate() {
            let right_seg = segments[slot + 1];
            match col.const_full {
                ConstValue::None => {
                    col.const_lead.push_into(&mut scratch.str_arena);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                    let next_start = scratch.str_arena.len();
                    scratch.str_arena.extend_from_slice(col.const_trail);
                    scratch.str_arena.extend_from_slice(right_seg);
                    active_cols[active_len] = col;
                    active_len += 1;
                    seg_ranges[active_len] = (next_start, scratch.str_arena.len());
                }
                ConstValue::Slice(s) => {
                    scratch.str_arena.extend_from_slice(s);
                    scratch.str_arena.extend_from_slice(right_seg);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                }
                ConstValue::Inline { ref buf, len } => {
                    scratch.str_arena.extend_from_slice(&buf[..len as usize]);
                    scratch.str_arena.extend_from_slice(right_seg);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                }
                ConstValue::QuotedDecimal {
                    prefix,
                    ref buf,
                    len,
                } => {
                    scratch.str_arena.push(b'"');
                    scratch.str_arena.extend_from_slice(prefix);
                    scratch.str_arena.extend_from_slice(&buf[..len as usize]);
                    scratch.str_arena.push(b'"');
                    scratch.str_arena.extend_from_slice(right_seg);
                    seg_ranges[active_len].1 = scratch.str_arena.len();
                }
            }
        }

        let slots = active_len;

        // Step 2: Build `FastPiece`s for `MODEL_DICTIONARY` / `MODEL_FRONT_DICT` entries,
        // absorbing adjacent literal segments when 100% of the segment can be eliminated.
        for slot in 0..slots {
            if let Some((slice_off, slice_len)) = active_cols[slot].dict_slices {
                let max_entry = active_cols[slot].max_emit_len;
                let left_range = seg_ranges[slot];
                let left_len_raw = left_range.1 - left_range.0;
                let absorb_left = left_len_raw > 0 && left_len_raw + max_entry <= 48;
                let left_len = if absorb_left { left_len_raw } else { 0 };

                let right_range = seg_ranges[slot + 1];
                let right_len_raw = right_range.1 - right_range.0;
                let cur_len = left_len + max_entry;
                let right_len = if right_len_raw == 0 || cur_len >= 48 {
                    0
                } else if slot + 1 < slots && active_cols[slot + 1].dict_slices.is_some() {
                    let next_max = active_cols[slot + 1].max_emit_len;
                    if cur_len <= 32
                        && next_max <= 32
                        && cur_len + right_len_raw + next_max <= 64
                    {
                        if cur_len + right_len_raw <= 32 {
                            right_len_raw
                        } else if right_len_raw + next_max <= 32 {
                            0
                        } else {
                            32 - cur_len
                        }
                    } else if cur_len + right_len_raw <= 48 {
                        right_len_raw
                    } else if next_max <= 48 && cur_len + right_len_raw + next_max <= 96 {
                        48 - cur_len
                    } else {
                        0
                    }
                } else if cur_len + right_len_raw <= 48 {
                    right_len_raw
                } else {
                    0
                };

                let (piece_offset, u8_offset) = match active_cols[slot].stream {
                    ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        ..
                    } => (piece_offset, u8_offset),
                    _ => unreachable!(),
                };

                if left_len > 0 || right_len > 0 {
                    let mut buf = [0u8; 96];
                    if left_len > 0 {
                        let left_bytes = &scratch.str_arena[left_range.0..left_range.1];
                        buf[..left_len].copy_from_slice(left_bytes);
                        seg_ranges[slot] = (left_range.0, left_range.0);
                    }
                    let (r0, r1, r2) = if right_len > 0 {
                        let right_ptr =
                            unsafe { scratch.str_arena.as_ptr().add(right_range.0) };
                        seg_ranges[slot + 1].0 = right_range.0 + right_len;
                        unsafe {
                            (
                                read_u128_partial(right_ptr, right_len),
                                if right_len > 16 {
                                    read_u128_partial(right_ptr.add(16), right_len - 16)
                                } else {
                                    0
                                },
                                if right_len > 32 {
                                    read_u128_partial(right_ptr.add(32), right_len - 32)
                                } else {
                                    0
                                },
                            )
                        }
                    } else {
                        (0, 0, 0)
                    };
                    let max_fused = left_len + max_entry + right_len;
                    let piece_ptr = unsafe { scratch.piece_arena.as_mut_ptr().add(piece_offset) };
                    let slice_ptr = unsafe { scratch.slice_arena.as_ptr().add(slice_off) };
                    let b_ptr = buf.as_mut_ptr();
                    for idx in 0..slice_len {
                        let (ptr, len) = unsafe { *slice_ptr.add(idx) };
                        let total = left_len + len + right_len;
                        unsafe {
                            let dst_m = b_ptr.add(left_len);
                            dst_m
                                .cast::<u128>()
                                .write_unaligned(read_u128_partial(ptr, len));
                            if len > 16 {
                                dst_m.add(16).cast::<u128>().write_unaligned(
                                    read_u128_partial(ptr.add(16), len - 16),
                                );
                                if len > 32 {
                                    dst_m.add(32).cast::<u128>().write_unaligned(
                                        read_u128_partial(ptr.add(32), len - 32),
                                    );
                                }
                            }
                            if right_len > 0 {
                                let dst_r = dst_m.add(len);
                                dst_r.cast::<u128>().write_unaligned(r0);
                                if right_len > 16 {
                                    dst_r.add(16).cast::<u128>().write_unaligned(r1);
                                    if right_len > 32 {
                                        dst_r.add(32).cast::<u128>().write_unaligned(r2);
                                    }
                                }
                            }
                            *piece_ptr.add(idx) = FastPiece {
                                v0: b_ptr.cast::<u128>().read_unaligned(),
                                v1: b_ptr.add(16).cast::<u128>().read_unaligned(),
                                v2: b_ptr.add(32).cast::<u128>().read_unaligned(),
                                ptr: std::ptr::null(),
                                len: total,
                            };
                        }
                    }
                    active_cols[slot].max_emit_len = max_fused;
                    active_cols[slot].stream = ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        max_len: max_fused,
                    };
                } else {
                    let piece_ptr = unsafe { scratch.piece_arena.as_mut_ptr().add(piece_offset) };
                    let slice_ptr = unsafe { scratch.slice_arena.as_ptr().add(slice_off) };
                    for idx in 0..slice_len {
                        let (ptr, len) = unsafe { *slice_ptr.add(idx) };
                        let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                        unsafe {
                            *piece_ptr.add(idx) = FastPiece::from_slice(s);
                        }
                    }
                }
            }
        }

        let head_range = seg_ranges[0];
        let mut row_budget = (head_range.1 - head_range.0) + 96;
        let col_start = scratch.unbound_col_arena.len();
        scratch.unbound_col_arena.reserve(slots);
        for slot in 0..slots {
            let col = active_cols[slot];
            let after_range = seg_ranges[slot + 1];
            row_budget = row_budget
                .saturating_add(col.max_emit_len)
                .saturating_add(after_range.1 - after_range.0);
            let unbound_col: UnboundExecColumn<'static> = unsafe {
                core::mem::transmute(UnboundExecColumn {
                    stream: col.stream,
                    after_range,
                })
            };
            scratch.unbound_col_arena.push(unbound_col);
        }
        let col_end = scratch.unbound_col_arena.len();

        Self {
            head_range,
            col_start,
            col_end,
            row_budget,
        }
    }

    /// Bind all arena offsets to raw pointers once all `DecodeScratch` arenas are frozen.
    fn bind(self, scratch: &mut DecodeScratch) -> PreparedTemplate {
        let head = FastPiece::from_slice(&scratch.str_arena[self.head_range.0..self.head_range.1]);
        let u8_base = scratch.u8_arena.as_ptr();
        let u16_base = scratch.u16_arena.as_ptr();
        let u64_base = scratch.u64_arena.as_ptr();
        let piece_base = scratch.piece_arena.as_ptr();
        let exec_start = scratch.exec_col_arena.len();
        let mut i = self.col_start;
        while i < self.col_end {
            let col = scratch.unbound_col_arena[i];
            if col.after_range.0 == col.after_range.1 {
                if let ColumnStream::Dictionary {
                    piece_offset: p0,
                    u8_offset: u0,
                    max_len: m0,
                } = col.stream
                {
                    if i + 3 < self.col_end
                        && scratch.unbound_col_arena[i + 1].after_range.0
                            == scratch.unbound_col_arena[i + 1].after_range.1
                        && scratch.unbound_col_arena[i + 2].after_range.0
                            == scratch.unbound_col_arena[i + 2].after_range.1
                        && scratch.unbound_col_arena[i + 3].after_range.0
                            == scratch.unbound_col_arena[i + 3].after_range.1
                    {
                        if let (
                            ColumnStream::Dictionary {
                                piece_offset: p1,
                                u8_offset: u1,
                                max_len: m1,
                            },
                            ColumnStream::Dictionary {
                                piece_offset: p2,
                                u8_offset: u2,
                                max_len: m2,
                            },
                            ColumnStream::Dictionary {
                                piece_offset: p3,
                                u8_offset: u3,
                                max_len: m3,
                            },
                        ) = (
                            scratch.unbound_col_arena[i + 1].stream,
                            scratch.unbound_col_arena[i + 2].stream,
                            scratch.unbound_col_arena[i + 3].stream,
                        ) {
                            let dp1 = p1.wrapping_sub(p0);
                            let dp2 = p2.wrapping_sub(p0);
                            let dp3 = p3.wrapping_sub(p0);
                            let du1 = u1.wrapping_sub(u0);
                            let du2 = u2.wrapping_sub(u0);
                            let du3 = u3.wrapping_sub(u0);
                            let max4 = m0.max(m1).max(m2).max(m3);
                            if (dp1 | dp2 | dp3 | du1 | du2 | du3) <= 65535 && max4 <= 48 {
                                let p_off = [dp1 as u16, dp2 as u16, dp3 as u16];
                                let u_off = [du1 as u16, du2 as u16, du3 as u16];
                                let stream = unsafe {
                                    if max4 <= 32 {
                                        BoundStream::DictionaryQuad32 {
                                            entries0: piece_base.add(p0),
                                            idx_ptr0: u8_base.add(u0),
                                            p_off,
                                            u_off,
                                        }
                                    } else {
                                        BoundStream::DictionaryQuad48 {
                                            entries0: piece_base.add(p0),
                                            idx_ptr0: u8_base.add(u0),
                                            p_off,
                                            u_off,
                                        }
                                    }
                                };
                                scratch.exec_col_arena.push(ExecColumn {
                                    stream,
                                    after: FastPiece::empty(),
                                });
                                i += 4;
                                continue;
                            }
                        }
                    }
                    if i + 1 < self.col_end
                        && scratch.unbound_col_arena[i + 1].after_range.0
                            == scratch.unbound_col_arena[i + 1].after_range.1
                    {
                        if let ColumnStream::Dictionary {
                            piece_offset: p1,
                            u8_offset: u1,
                            max_len: m1,
                        } = scratch.unbound_col_arena[i + 1].stream
                        {
                            let idx_delta = (u1 as isize).wrapping_sub(u0 as isize);
                            if m0 <= 32 && m1 <= 32 {
                                scratch.exec_col_arena.push(ExecColumn {
                                    stream: unsafe {
                                        BoundStream::DictionaryPair32 {
                                            entries0: piece_base.add(p0),
                                            entries1: piece_base.add(p1),
                                            idx_ptr0: u8_base.add(u0),
                                            idx_delta,
                                        }
                                    },
                                    after: FastPiece::empty(),
                                });
                                i += 2;
                                continue;
                            } else if m0 <= 48 && m1 <= 48 {
                                scratch.exec_col_arena.push(ExecColumn {
                                    stream: unsafe {
                                        BoundStream::DictionaryPair48 {
                                            entries0: piece_base.add(p0),
                                            entries1: piece_base.add(p1),
                                            idx_ptr0: u8_base.add(u0),
                                            idx_delta,
                                        }
                                    },
                                    after: FastPiece::empty(),
                                });
                                i += 2;
                                continue;
                            }
                        }
                    }
                }
            }
            let after = if col.after_range.0 == col.after_range.1 {
                FastPiece::empty()
            } else {
                FastPiece::from_slice(&scratch.str_arena[col.after_range.0..col.after_range.1])
            };
            let bound = unsafe {
                match col.stream {
                    ColumnStream::Raw { cursor } => BoundStream::Raw { cursor },
                    ColumnStream::Dictionary {
                        piece_offset,
                        u8_offset,
                        max_len,
                    } => {
                        if after.len == 0 && max_len <= 16 {
                            BoundStream::Dictionary16 {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        } else if after.len == 0 && max_len <= 32 {
                            BoundStream::Dictionary32 {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        } else if after.len == 0 && max_len <= 48 {
                            BoundStream::Dictionary48 {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        } else {
                            BoundStream::DictionaryLong {
                                entries: piece_base.add(piece_offset),
                                idx_ptr: u8_base.add(u8_offset),
                            }
                        }
                    }
                    ColumnStream::DecimalAffineSmall { current, step } => {
                        BoundStream::DecimalAffineSmall { current, step }
                    }
                    ColumnStream::DecimalAffine { current, step } => {
                        BoundStream::DecimalAffine { current, step }
                    }
                    ColumnStream::DecimalSmall { u16_offset } => BoundStream::DecimalSmall {
                        val_ptr: u16_base.add(u16_offset),
                    },
                    ColumnStream::DecimalPacked { min, reader } => {
                        BoundStream::DecimalPacked { min, reader }
                    }
                    ColumnStream::DecimalValues { u64_offset } => BoundStream::DecimalValues {
                        val_ptr: u64_base.add(u64_offset),
                    },
                    ColumnStream::HexSuffix4 { ptr } => BoundStream::HexSuffix4 { ptr },
                    ColumnStream::HexSuffix8 { ptr } => BoundStream::HexSuffix8 { ptr },
                    ColumnStream::HexSuffix { cursor, width } => {
                        BoundStream::HexSuffix { cursor, width }
                    }
                    ColumnStream::TimestampMinSec {
                        min_u8_offset,
                        sec_u8_offset,
                    } => BoundStream::TimestampMinSec {
                        min_ptr: u8_base.add(min_u8_offset),
                        sec_ptr: u8_base.add(sec_u8_offset),
                    },
                    ColumnStream::TimestampFull { u8_offset, stride } => {
                        BoundStream::TimestampFull {
                            ptr: u8_base.add(u8_offset),
                            stride,
                        }
                    }
                    ColumnStream::Copy16 { u8_offset } => BoundStream::Copy16 {
                        ptr: u8_base.add(u8_offset),
                    },
                    ColumnStream::SegmentDict {
                        str_u8_offset,
                        len_u16_offset,
                    } => BoundStream::SegmentDict {
                        str_ptr: u8_base.add(str_u8_offset),
                        len_ptr: u16_base.add(len_u16_offset),
                    },
                    ColumnStream::PeriodicString { cursor, period } => {
                        BoundStream::PeriodicString { cursor, period }
                    }
                    ColumnStream::PeriodicHex { cursor, period } => {
                        BoundStream::PeriodicHex { cursor, period }
                    }
                }
            };
            scratch.exec_col_arena.push(ExecColumn {
                stream: bound,
                after,
            });
            i += 1;
        }
        let exec_end = scratch.exec_col_arena.len();
        PreparedTemplate {
            head,
            col_ptr: unsafe { scratch.exec_col_arena.as_mut_ptr().add(exec_start) },
            col_len: exec_end - exec_start,
            row_budget: self.row_budget,
        }
    }
}

impl PreparedTemplate {
    #[inline(always)]
    unsafe fn emit_row(&mut self, mut dst: *mut u8) -> *mut u8 {
        unsafe {
            if self.head.len > 0 {
                if self.head.len <= 16 {
                    dst = self.head.emit_16(dst);
                } else if self.head.len <= 32 {
                    dst = self.head.emit_32(dst);
                } else {
                    dst = self.head.emit_48(dst);
                }
            }
            let slots = self.col_len;
            let col_ptr = self.col_ptr;
            for slot in 0..slots {
                dst = (*col_ptr.add(slot)).emit(dst);
            }
            dst
        }
    }
}

#[inline(always)]
fn emit_template_rows(
    tpl: &mut PreparedTemplate,
    mut remaining: usize,
    base: *mut u8,
    pos: &mut usize,
    capacity: usize,
    tail_buf: &mut core::mem::MaybeUninit<[u8; 4096]>,
) -> Result<()> {
    let budget = tpl.row_budget;
    while remaining > 0 {
        let avail = capacity.saturating_sub(*pos);
        let safe_rows = if budget > 0 {
            (avail / budget).min(remaining)
        } else {
            remaining
        };
        if safe_rows > 0 {
            unsafe {
                let mut dst = base.add(*pos);
                for _ in 0..safe_rows {
                    dst = tpl.emit_row(dst);
                }
                *pos = dst.offset_from(base) as usize;
            }
            remaining -= safe_rows;
            if remaining == 0 {
                break;
            }
            if *pos + budget <= capacity {
                continue;
            }
        }
        // Fallback when remaining capacity < `budget` (the last 1..2 rows of the block):
        if budget <= 4096 {
            let written = unsafe {
                let start_ptr = tail_buf.as_mut_ptr().cast::<u8>();
                let end_ptr = tpl.emit_row(start_ptr);
                end_ptr.offset_from(start_ptr) as usize
            };
            if *pos + written > capacity {
                return Err(invalid("column output exceeds length"));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tail_buf.as_ptr().cast::<u8>(),
                    base.add(*pos),
                    written,
                );
            }
            *pos += written;
        } else {
            let mut heap_row = vec![0u8; budget];
            let written = unsafe {
                let start_ptr = heap_row.as_mut_ptr();
                let end_ptr = tpl.emit_row(start_ptr);
                end_ptr.offset_from(start_ptr) as usize
            };
            if *pos + written > capacity {
                return Err(invalid("column output exceeds length"));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(heap_row.as_ptr(), base.add(*pos), written);
            }
            *pos += written;
        }
        remaining -= 1;
    }
    Ok(())
}

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
                        if tpl.col_len != 0 {
                            dst = unsafe { tpl.emit_row(dst) };
                        } else if tpl.head.len <= 16 {
                            unsafe {
                                dst.cast::<u128>().write_unaligned(tpl.head.v0);
                                dst = dst.add(tpl.head.len);
                            }
                        } else {
                            dst = unsafe { tpl.emit_row(dst) };
                        }
                    } else {
                        let mut pos = unsafe { dst.offset_from(base) as usize };
                        emit_template_rows(tpl, 1, base, &mut pos, capacity, &mut tail_buf)?;
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

#[derive(Default)]
struct StrideScratch {
    planes: [Vec<u8>; 8],
    super_mask_buf: Vec<u64>,
    mask_buf: Vec<u64>,
    nz_buf: Vec<u8>,
    lz_dict4: Vec<u32>,
    lz_dict7: Vec<u16>,
    lz_tokens: Vec<u8>,
    lz_offsets: Vec<u8>,
    lz_ext: Vec<u8>,
    lz_lits: Vec<u8>,
    lz_meta_raw: Vec<u8>,
    lz_meta_comp: Vec<u8>,
    lz_lits_comp: Vec<u8>,
    lz_lits_huf: Vec<u8>,
    lz_meta_huf: Vec<u8>,
    lz_sub_lits: Vec<u8>,
    lz_sub_tokens: Vec<u8>,
    lz_sub_offsets: Vec<u8>,
    lz_sub_ext: Vec<u8>,
}

thread_local! {
    static STRIDE_SCRATCH: std::cell::RefCell<StrideScratch> =
        std::cell::RefCell::new(StrideScratch::default());
    /// Plane buffers for `STRIDE_PACKET` mode 2 (delta planes + nested `FAST_LZ`); kept separate
    /// from `STRIDE_SCRATCH` because the nested `FAST_LZ` coder borrows that scratch itself.
    static STRIDE_PLANES2: std::cell::RefCell<[Vec<u8>; 8]> =
        std::cell::RefCell::new(Default::default());
    static STRIDE_COMBINED: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------------------------
// Canonical Huffman (order-0) entropy stage for FAST_LZ literal / meta streams.
//
// Blob format:
//   kind 1: `sym`                                   (every byte equals `sym`)
//   kind 2: `max_sym`, code-length nibbles for symbols `0..=max_sym` (low nibble first;
//           0..=11 = length, 12/13/14/15 = run of 3/6/12/24 zero lengths), ULEB sizes of
//           streams 0..=2, then 4 LSB-first bitstreams; stream `k` holds output quarter `k`
//           (`q = ceil(n/4)`). Codes are canonical, length-limited to `HUF_BITS`, and must form
//           a complete prefix code; every stream must be consumed to its last byte.
// ---------------------------------------------------------------------------------------------
const HUF_BITS: u32 = 11;
const HUF_KIND_RLE: u8 = 1;
const HUF_KIND_HUF: u8 = 2;
/// Streams shorter than this stay raw: per-blob table setup would cost more than it saves.
const HUF_MIN: usize = 160;
/// Streams of at least this many symbols are split into 4 interleaved bitstreams.
const HUF_SPLIT4: usize = 1024;

fn huf_lengths(freq: &[u32; 256]) -> Option<[u8; 256]> {
    let mut leaves = [(0u32, 0u16); 256];
    let mut k = 0usize;
    for (s, &c) in freq.iter().enumerate() {
        if c > 0 {
            leaves[k] = (c, s as u16);
            k += 1;
        }
    }
    if k < 2 {
        return None;
    }
    // Leaves were gathered in ascending symbol order, so a stable LSD radix sort on
    // the count alone yields exactly the `(count, sym)` order (skips empty passes).
    {
        let mut tmp = [(0u32, 0u16); 256];
        let max_c = leaves[..k].iter().map(|l| l.0).max().unwrap_or(0);
        let mut in_tmp = false;
        let mut shift = 0u32;
        while shift < 32 && (shift == 0 || (max_c >> shift) != 0) {
            let (src, dst) = if in_tmp {
                (&tmp[..k], &mut leaves[..k])
            } else {
                (&leaves[..k], &mut tmp[..k])
            };
            let mut hist = [0u16; 257];
            for l in src {
                hist[((l.0 >> shift) & 0xff) as usize + 1] += 1;
            }
            if hist[1..].iter().any(|&h| h as usize == k) {
                shift += 8;
                continue;
            }
            for i in 1..257 {
                hist[i] += hist[i - 1];
            }
            for l in src {
                let d = ((l.0 >> shift) & 0xff) as usize;
                dst[hist[d] as usize] = *l;
                hist[d] += 1;
            }
            in_tmp = !in_tmp;
            shift += 8;
        }
        if in_tmp {
            leaves[..k].copy_from_slice(&tmp[..k]);
        }
    }
    debug_assert!(leaves[..k].windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)));
    let mut shift = 0u32;
    loop {
        let mut q1_w = [0u64; 256];
        let mut q1_id = [0u16; 256];
        let mut parent = [0u16; 512];
        let mut i0 = 0usize;
        let mut i1 = 0usize;
        let mut n1 = 0usize;
        let mut next = 256u16;
        macro_rules! pop_min {
            () => {{
                let w0 = if i0 < k {
                    ((leaves[i0].0.wrapping_add((1u32 << shift) - 1) >> shift).max(1)) as u64
                } else {
                    u64::MAX
                };
                let w1 = if i1 < n1 { q1_w[i1] } else { u64::MAX };
                if w0 <= w1 {
                    let id = leaves[i0].1;
                    i0 += 1;
                    (w0, id)
                } else {
                    let id = q1_id[i1];
                    i1 += 1;
                    (w1, id)
                }
            }};
        }
        for _ in 0..(k - 1) {
            let (wa, a) = pop_min!();
            let (wb, b) = pop_min!();
            parent[a as usize] = next;
            parent[b as usize] = next;
            q1_w[n1] = wa + wb;
            q1_id[n1] = next;
            n1 += 1;
            next += 1;
        }
        let root = (next - 1) as usize;
        let mut depth = [0u8; 512];
        for node in (256..root).rev() {
            depth[node] = depth[parent[node] as usize] + 1;
        }
        let mut lens = [0u8; 256];
        let mut max_len = 0u8;
        for &(_, s) in &leaves[..k] {
            let l = depth[parent[s as usize] as usize] + 1;
            lens[s as usize] = l;
            if l > max_len {
                max_len = l;
            }
        }
        if u32::from(max_len) <= HUF_BITS {
            return Some(lens);
        }
        shift += 1;
    }
}

/// Canonical codes (bit-reversed for LSB-first emission), ordered by `(len, sym)`.
fn huf_codes(lens: &[u8; 256], max_sym: usize) -> [u16; 256] {
    let mut bl_count = [0u16; 16];
    for &l in &lens[..=max_sym] {
        bl_count[l as usize] += 1;
    }
    bl_count[0] = 0;
    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    let mut codes = [0u16; 256];
    for s in 0..=max_sym {
        let l = lens[s] as usize;
        if l > 0 {
            let c = next_code[l];
            next_code[l] += 1;
            codes[s] = c.reverse_bits() >> (16 - l);
        }
    }
    codes
}

/// Append a Huffman blob for `src` to `out` if it is strictly smaller than `limit` bytes.
fn huf_encode(src: &[u8], out: &mut Vec<u8>, limit: usize) -> bool {
    let n = src.len();
    if n < HUF_MIN {
        return false;
    }
    let mut freq = [0u32; 256];
    let ptr = src.as_ptr();
    if n >= 512 {
        // 4-way unaliased histogram avoids store-to-load forwarding stalls on zero runs.
        let mut f1 = [0u32; 256];
        let mut f2 = [0u32; 256];
        let mut f3 = [0u32; 256];
        let chunks8 = n / 8;
        for i in 0..chunks8 {
            unsafe {
                let w = ptr.add(i * 8).cast::<u64>().read_unaligned().to_le();
                freq[(w & 0xff) as usize] += 1;
                f1[((w >> 8) & 0xff) as usize] += 1;
                f2[((w >> 16) & 0xff) as usize] += 1;
                f3[((w >> 24) & 0xff) as usize] += 1;
                freq[((w >> 32) & 0xff) as usize] += 1;
                f1[((w >> 40) & 0xff) as usize] += 1;
                f2[((w >> 48) & 0xff) as usize] += 1;
                f3[(w >> 56) as usize] += 1;
            }
        }
        for &b in &src[chunks8 * 8..] {
            freq[b as usize] += 1;
        }
        for s in 0..256 {
            freq[s] += f1[s] + f2[s] + f3[s];
        }
    } else {
        let chunks4 = n / 4;
        for i in 0..chunks4 {
            unsafe {
                let p = ptr.add(i * 4);
                freq[*p as usize] += 1;
                freq[*p.add(1) as usize] += 1;
                freq[*p.add(2) as usize] += 1;
                freq[*p.add(3) as usize] += 1;
            }
        }
        for &b in &src[chunks4 * 4..] {
            freq[b as usize] += 1;
        }
    }
    let mut distinct = 0usize;
    let mut max_sym = 0usize;
    let mut max_freq = 0u32;
    for (s, &c) in freq.iter().enumerate() {
        if c > 0 {
            distinct += 1;
            max_sym = s;
            if c > max_freq {
                max_freq = c;
            }
        }
    }
    if distinct == 1 {
        if 2 >= limit {
            return false;
        }
        out.push(HUF_KIND_RLE);
        out.push(src[0]);
        return true;
    }
    // Adaptive Entropy Gate: if the alphabet is wide and flat, skip Huffman tree & bitstream build.
    if distinct >= 200 && (max_freq as usize) * 28 < n {
        return false;
    }
    let Some(lens) = huf_lengths(&freq) else {
        return false;
    };
    let mut bits = 0u64;
    for s in 0..=max_sym {
        bits += freq[s] as u64 * lens[s] as u64;
    }
    // Table nibbles on stack.
    let mut nibbles = [0u8; 256];
    let mut nib_len = 0usize;
    let mut s = 0usize;
    while s <= max_sym {
        if lens[s] != 0 {
            nibbles[nib_len] = lens[s];
            nib_len += 1;
            s += 1;
            continue;
        }
        let mut run = 0usize;
        while s + run <= max_sym && lens[s + run] == 0 {
            run += 1;
        }
        s += run;
        for (code, size) in [(15u8, 24usize), (14, 12), (13, 6), (12, 3)] {
            while run >= size {
                nibbles[nib_len] = code;
                nib_len += 1;
                run -= size;
            }
        }
        while run > 0 {
            nibbles[nib_len] = 0;
            nib_len += 1;
            run -= 1;
        }
    }
    let est = 2 + nib_len.div_ceil(2) + if n >= HUF_SPLIT4 { 9 } else { 0 } + (bits as usize).div_ceil(8) + 4;
    // Decode-cost gate: a Huffman stream costs a table build plus ~1.5 ns/byte on
    // decode versus a near-free raw copy, so demand at least 3% savings.
    if est >= limit || est * 100 > limit * 97 {
        return false;
    }
    let codes = huf_codes(&lens, max_sym);
    let mut packed_code = [0u32; 256];
    for s in 0..=max_sym {
        packed_code[s] = (codes[s] as u32) | ((lens[s] as u32) << 16);
    }
    let start = out.len();
    out.reserve(est + 32);
    out.push(HUF_KIND_HUF);
    out.push(max_sym as u8);
    for pair in nibbles[..nib_len].chunks(2) {
        out.push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
    }

    #[inline(always)]
    unsafe fn encode_stream_fast(
        part: &[u8],
        packed_code: &[u32; 256],
        out_ptr: *mut u8,
        mut out_pos: usize,
    ) -> usize {
        let mut acc = 0u64;
        let mut nb = 0u32;
        let p_in = part.as_ptr();
        let len5 = part.len() / 5 * 5;
        let mut i = 0usize;
        while i < len5 {
            unsafe {
                let pc0 = *packed_code.get_unchecked(*p_in.add(i) as usize);
                let pc1 = *packed_code.get_unchecked(*p_in.add(i + 1) as usize);
                let pc2 = *packed_code.get_unchecked(*p_in.add(i + 2) as usize);
                let pc3 = *packed_code.get_unchecked(*p_in.add(i + 3) as usize);
                let pc4 = *packed_code.get_unchecked(*p_in.add(i + 4) as usize);
                let l0 = pc0 >> 16;
                let l1 = pc1 >> 16;
                let l2 = pc2 >> 16;
                let l3 = pc3 >> 16;
                let l4 = pc4 >> 16;
                let l01 = l0 + l1;
                let l23 = l2 + l3;
                let l0123 = l01 + l23;
                let chunk_len = l0123 + l4;
                let c01 = u64::from(pc0 & 0xffff) | (u64::from(pc1 & 0xffff) << l0);
                let c23 = u64::from(pc2 & 0xffff) | (u64::from(pc3 & 0xffff) << l2);
                let c0123 = c01 | (c23 << l01);
                let chunk_bits = c0123 | (u64::from(pc4 & 0xffff) << l0123);
                acc |= chunk_bits << nb;
                nb += chunk_len;
                out_ptr.add(out_pos).cast::<u64>().write_unaligned(acc.to_le());
                out_pos += (nb >> 3) as usize;
                acc >>= nb & !7;
                nb &= 7;
            }
            i += 5;
        }
        while i < part.len() {
            let pc = unsafe { *packed_code.get_unchecked(*p_in.add(i) as usize) };
            acc |= u64::from(pc & 0xffff) << nb;
            nb += pc >> 16;
            i += 1;
        }
        while nb >= 8 {
            unsafe {
                *out_ptr.add(out_pos) = acc as u8;
            }
            out_pos += 1;
            acc >>= 8;
            nb -= 8;
        }
        if nb > 0 {
            unsafe {
                *out_ptr.add(out_pos) = acc as u8;
            }
            out_pos += 1;
        }
        out_pos
    }

    if n < HUF_SPLIT4 {
        let s_start = out.len();
        out.reserve((bits as usize).div_ceil(8) + 16);
        let new_len = unsafe { encode_stream_fast(src, &packed_code, out.as_mut_ptr(), s_start) };
        unsafe { out.set_len(new_len) };
    } else {
        let q = n.div_ceil(4);
        let mut stream_lens = [0usize; 4];
        let hdr_pos = out.len();
        out.reserve(9 + (bits as usize).div_ceil(8) + 32);
        let mut cur_pos = hdr_pos + 9;
        let out_ptr = out.as_mut_ptr();
        for k in 0..4 {
            let part = &src[(k * q).min(n)..((k + 1) * q).min(n)];
            let next_pos = unsafe { encode_stream_fast(part, &packed_code, out_ptr, cur_pos) };
            stream_lens[k] = next_pos - cur_pos;
            cur_pos = next_pos;
        }
        unsafe { out.set_len(cur_pos) };
        let mut uleb_hdr = [0u8; 9];
        let mut u_len = 0usize;
        for &sl in &stream_lens[..3] {
            let mut v = sl;
            loop {
                let mut b = (v & 0x7f) as u8;
                v >>= 7;
                if v != 0 {
                    b |= 0x80;
                }
                uleb_hdr[u_len] = b;
                u_len += 1;
                if v == 0 {
                    break;
                }
            }
        }
        out[hdr_pos..hdr_pos + u_len].copy_from_slice(&uleb_hdr[..u_len]);
        out.copy_within(hdr_pos + 9..cur_pos, hdr_pos + u_len);
        out.truncate(cur_pos - (9 - u_len));
    }
    if out.len() - start >= limit {
        out.truncate(start);
        return false;
    }
    true
}

#[inline(always)]
fn huf_decode_tail(
    src: &[u8],
    mut bp: usize,
    dst: *mut u8,
    mut idx: usize,
    end: usize,
    table: &[u16],
    mask: u64,
) -> bool {
    let slen = src.len();
    let sptr = src.as_ptr();
    unsafe {
        while idx < end {
            let byte_pos = bp >> 3;
            let b = if byte_pos + 8 <= slen {
                sptr.add(byte_pos).cast::<u64>().read_unaligned().to_le() >> (bp & 7)
            } else {
                let mut v = 0u64;
                let mut k = 0usize;
                while byte_pos + k < slen && k < 8 {
                    v |= (*sptr.add(byte_pos + k) as u64) << (k * 8);
                    k += 1;
                }
                v >> (bp & 7)
            };
            let e = *table.get_unchecked((b & mask) as usize);
            *dst.add(idx) = e as u8;
            bp += (e >> 8) as usize;
            idx += 1;
        }
    }
    bp <= slen * 8 && bp + 8 > slen * 8
}

fn huf_decode(blob: &[u8], dst: &mut [u8]) -> Result<()> {
    let n = dst.len();
    let mut cur = Cursor::new(blob);
    match cur.byte()? {
        HUF_KIND_RLE => {
            let sym = cur.byte()?;
            if !cur.done() {
                return Err(invalid("huffman trailing bytes"));
            }
            dst.fill(sym);
            return Ok(());
        }
        HUF_KIND_HUF => {}
        _ => return Err(invalid("huffman kind")),
    }
    let max_sym = cur.byte()? as usize;
    let mut lens = [0u8; 256];
    let mut bl_count = [0u16; 16];
    let mut s = 0usize;
    let mut pending: Option<u8> = None;
    while s <= max_sym {
        let nib = match pending.take() {
            Some(v) => v,
            None => {
                let b = cur.byte()?;
                pending = Some(b >> 4);
                b & 15
            }
        };
        match nib {
            0..=11 => {
                lens[s] = nib;
                bl_count[nib as usize] += 1;
                s += 1;
            }
            _ => {
                let run = 3usize << (nib - 12);
                if s + run > max_sym + 1 {
                    return Err(invalid("huffman length run"));
                }
                s += run;
            }
        }
    }
    if lens[max_sym] == 0 {
        return Err(invalid("huffman max symbol"));
    }
    bl_count[0] = 0;
    // Require a complete prefix code (Kraft sum == 2^max_len) from the per-length counts.
    let max_len = (1..16).rev().find(|&l| bl_count[l] != 0).unwrap_or(0) as u32;
    let mut kraft = 0u32;
    let mut used = 0usize;
    for l in 1..=max_len as usize {
        kraft += (bl_count[l] as u32) << (max_len as usize - l);
        used += bl_count[l] as usize;
    }
    if used < 2 || kraft != 1 << max_len {
        return Err(invalid("huffman incomplete code"));
    }
    let split4 = n >= HUF_SPLIT4;
    let (s0, s1, s2) = if split4 {
        (cur.uleb()?, cur.uleb()?, cur.uleb()?)
    } else {
        (blob.len() - cur.pos, 0, 0)
    };
    let rest = &blob[cur.pos..];
    let (a, rest) = rest.split_at(s0.min(rest.len()));
    let (b, rest) = rest.split_at(s1.min(rest.len()));
    let (c, d) = rest.split_at(s2.min(rest.len()));
    if a.len() != s0 || b.len() != s1 || c.len() != s2 {
        return Err(invalid("huffman stream sizes"));
    }
    let size = 1usize << max_len;
    let mut table_uninit = [std::mem::MaybeUninit::<u16>::uninit(); 1 << HUF_BITS];
    let tbl_ptr = table_uninit.as_mut_ptr().cast::<u16>();
    // Doubling build: at level `l` the table (size 2^l) is its lower half copied
    // once (contiguous memcpy), then every code of length `l` writes its single
    // bit-reversed slot. Low byte holds symbol (`e as u8`), high byte holds code length (`e >> 8`).
    {
        let mut start = [0u16; 17];
        for l in 1..16 {
            start[l + 1] = start[l] + bl_count[l];
        }
        let mut by_len = [0u8; 256];
        let mut fill = start;
        for (sym, &l) in lens[..=max_sym].iter().enumerate() {
            if l != 0 {
                by_len[fill[l as usize] as usize] = sym as u8;
                fill[l as usize] += 1;
            }
        }
        unsafe {
            *tbl_ptr = 0;
            let mut cur = 1usize;
            let mut code = 0u16;
            for l in 1..=max_len as usize {
                std::ptr::copy_nonoverlapping(tbl_ptr, tbl_ptr.add(cur), cur);
                cur <<= 1;
                code <<= 1;
                let l_hi = (l as u16) << 8;
                for &sym in &by_len[start[l] as usize..start[l + 1] as usize] {
                    let idx = (code.reverse_bits() >> (16 - l)) as usize;
                    *tbl_ptr.add(idx) = l_hi | (sym as u16);
                    code += 1;
                }
            }
        }
    }
    // SAFETY: `kraft == 1 << max_len == size` guarantees every slot in `0..size` was written above.
    let table = unsafe { std::slice::from_raw_parts(tbl_ptr, size) };
    let mask = (size - 1) as u64;
    if !split4 {
        let fast_lim = n.saturating_sub(5);
        let a_bit_lim = a.len().saturating_sub(8) * 8;
        let a_ptr = a.as_ptr();
        let p0 = dst.as_mut_ptr();
        let mut bp0 = 0usize;
        let mut i = 0usize;
        unsafe {
            while i < fast_lim && bp0 <= a_bit_lim {
                let mut b0 = a_ptr.add(bp0 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp0 & 7);
                macro_rules! step1 {
                    ($off:expr) => {{
                        let e0 = *table.get_unchecked((b0 & mask) as usize);
                        *p0.add(i + $off) = e0 as u8;
                        let l0 = (e0 >> 8) as usize;
                        bp0 += l0;
                        b0 >>= l0;
                    }};
                    ($off:expr, last) => {{
                        let e0 = *table.get_unchecked((b0 & mask) as usize);
                        *p0.add(i + $off) = e0 as u8;
                        bp0 += (e0 >> 8) as usize;
                    }};
                }
                step1!(0);
                step1!(1);
                step1!(2);
                step1!(3);
                step1!(4, last);
                i += 5;
            }
        }
        return if huf_decode_tail(a, bp0, p0, i, n, table, mask) {
            Ok(())
        } else {
            Err(invalid("huffman stream consumption"))
        };
    }
    let q = n.div_ceil(4);
    let lens4 = [
        q.min(n),
        q.min(n.saturating_sub(q)),
        q.min(n.saturating_sub(2 * q)),
        n.saturating_sub(3 * q),
    ];
    let base = dst.as_mut_ptr();
    let fast_lim = lens4[3].saturating_sub(5);
    let (a_bit_lim, b_bit_lim, c_bit_lim, d_bit_lim) = (
        a.len().saturating_sub(8) * 8,
        b.len().saturating_sub(8) * 8,
        c.len().saturating_sub(8) * 8,
        d.len().saturating_sub(8) * 8,
    );
    let (a_ptr, b_ptr, c_ptr, d_ptr) = (a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_ptr());
    let (mut bp0, mut bp1, mut bp2, mut bp3) = (0usize, 0usize, 0usize, 0usize);
    let mut i = 0usize;
    // SAFETY: stream k writes `dst[k*q .. k*q + lens4[k]]`, all inside `dst`.
    let (p0, p1, p2, p3) = unsafe {
        (
            base,
            base.add(q.min(n)),
            base.add((2 * q).min(n)),
            base.add((3 * q).min(n)),
        )
    };
    unsafe {
        while i < fast_lim
            && bp0 <= a_bit_lim
            && bp1 <= b_bit_lim
            && bp2 <= c_bit_lim
            && bp3 <= d_bit_lim
        {
            let mut b0 = a_ptr.add(bp0 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp0 & 7);
            let mut b1 = b_ptr.add(bp1 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp1 & 7);
            let mut b2 = c_ptr.add(bp2 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp2 & 7);
            let mut b3 = d_ptr.add(bp3 >> 3).cast::<u64>().read_unaligned().to_le() >> (bp3 & 7);
            macro_rules! step4 {
                ($off:expr) => {{
                    let e0 = *table.get_unchecked((b0 & mask) as usize);
                    let e1 = *table.get_unchecked((b1 & mask) as usize);
                    let e2 = *table.get_unchecked((b2 & mask) as usize);
                    let e3 = *table.get_unchecked((b3 & mask) as usize);
                    *p0.add(i + $off) = e0 as u8;
                    *p1.add(i + $off) = e1 as u8;
                    *p2.add(i + $off) = e2 as u8;
                    *p3.add(i + $off) = e3 as u8;
                    let l0 = (e0 >> 8) as usize;
                    let l1 = (e1 >> 8) as usize;
                    let l2 = (e2 >> 8) as usize;
                    let l3 = (e3 >> 8) as usize;
                    bp0 += l0;
                    bp1 += l1;
                    bp2 += l2;
                    bp3 += l3;
                    b0 >>= l0;
                    b1 >>= l1;
                    b2 >>= l2;
                    b3 >>= l3;
                }};
                ($off:expr, last) => {{
                    let e0 = *table.get_unchecked((b0 & mask) as usize);
                    let e1 = *table.get_unchecked((b1 & mask) as usize);
                    let e2 = *table.get_unchecked((b2 & mask) as usize);
                    let e3 = *table.get_unchecked((b3 & mask) as usize);
                    *p0.add(i + $off) = e0 as u8;
                    *p1.add(i + $off) = e1 as u8;
                    *p2.add(i + $off) = e2 as u8;
                    *p3.add(i + $off) = e3 as u8;
                    bp0 += (e0 >> 8) as usize;
                    bp1 += (e1 >> 8) as usize;
                    bp2 += (e2 >> 8) as usize;
                    bp3 += (e3 >> 8) as usize;
                }};
            }
            step4!(0);
            step4!(1);
            step4!(2);
            step4!(3);
            step4!(4, last);
            i += 5;
        }
    }
    let ok0 = huf_decode_tail(a, bp0, p0, i, lens4[0], table, mask);
    let ok1 = huf_decode_tail(b, bp1, p1, i, lens4[1], table, mask);
    let ok2 = huf_decode_tail(c, bp2, p2, i, lens4[2], table, mask);
    let ok3 = huf_decode_tail(d, bp3, p3, i, lens4[3], table, mask);
    if ok0 && ok1 && ok2 && ok3 {
        Ok(())
    } else {
        Err(invalid("huffman stream consumption"))
    }
}

/// `ULEB len` + Huffman blob when strictly smaller than `src`, else `ULEB src.len()` + raw bytes.
fn put_huf_or_raw(out: &mut Vec<u8>, src: &[u8]) {
    let mark = out.len();
    let reserve = uleb_len(src.len());
    out.resize(mark + reserve, 0);
    if huf_encode(src, out, src.len()) {
        let blob_len = out.len() - (mark + reserve);
        let mut hdr = [0u8; 5];
        let mut hlen = 0usize;
        let mut v = blob_len;
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
            }
            hdr[hlen] = b;
            hlen += 1;
            if v == 0 {
                break;
            }
        }
        out[mark..mark + hlen].copy_from_slice(&hdr[..hlen]);
        if hlen < reserve {
            out.copy_within(mark + reserve.., mark + hlen);
            out.truncate(out.len() - (reserve - hlen));
        }
    } else {
        out.truncate(mark);
        put_uleb(out, src.len());
        out.extend_from_slice(src);
    }
}

/// Inverse of [`put_huf_or_raw`]: `len == dst.len()` is raw, `len < dst.len()` is Huffman.
fn take_huf_or_raw(cursor: &mut Cursor<'_>, dst: &mut [u8]) -> Result<()> {
    let len = cursor.uleb()?;
    let bytes = cursor.take(len)?;
    if len == dst.len() {
        dst.copy_from_slice(bytes);
        Ok(())
    } else if len < dst.len() {
        huf_decode(bytes, dst)
    } else {
        Err(invalid("huffman substream length"))
    }
}

/// Fused inclusive prefix sum + optional Subtract-Green inverse + interleave:
/// `c_p[i] = first[p] + sum(planes[p][..=i])`, and if `sub_green`, `c_0 += c_1`, `c_2 += c_1`.
/// Each channel keeps its own SIMD-resident carry, so 3–4 independent dependency chains run
/// in parallel instead of one serial GPR<->SIMD round trip per 16 bytes.
fn prefix_interleave_planes(
    p_ptrs: [*const u8; 8],
    const_plane: [bool; 8],
    first: &[u8; 8],
    stride: usize,
    count: usize,
    sub_green: bool,
    dst: &mut [u8],
) {
    debug_assert!(dst.len() >= count * stride);
    if (0..stride).all(|p| const_plane[p]) {
        let g = first[1];
        let c0 = if sub_green { first[0].wrapping_add(g) } else { first[0] };
        let c1 = g;
        let c2 = if sub_green && stride >= 3 {
            first[2].wrapping_add(g)
        } else {
            first[2]
        };
        if stride == 4 {
            let px = u32::from_le_bytes([c0, c1, c2, first[3]]);
            let mut idx = 0usize;
            let dst_ptr = dst.as_mut_ptr().cast::<u32>();
            unsafe {
                while idx < count {
                    dst_ptr.add(idx).write_unaligned(px);
                    idx += 1;
                }
            }
        } else if stride == 2 {
            let px = u16::from_le_bytes([first[0], first[1]]);
            let mut idx = 0usize;
            let dst_ptr = dst.as_mut_ptr().cast::<u16>();
            unsafe {
                while idx < count {
                    dst_ptr.add(idx).write_unaligned(px);
                    idx += 1;
                }
            }
        } else if stride == 8 {
            let px = u64::from_le_bytes(*first);
            let mut idx = 0usize;
            let dst_ptr = dst.as_mut_ptr().cast::<u64>();
            unsafe {
                while idx < count {
                    dst_ptr.add(idx).write_unaligned(px);
                    idx += 1;
                }
            }
        } else {
            for idx in 0..count {
                dst[idx * 3] = c0;
                dst[idx * 3 + 1] = c1;
                dst[idx * 3 + 2] = c2;
            }
        }
        return;
    }

    let mut acc = *first;
    let mut idx = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let zero = vdupq_n_u8(0);
        macro_rules! scan {
            ($v:expr, $a:ident) => {{
                let mut s = $v;
                s = vaddq_u8(s, vextq_u8::<15>(zero, s));
                s = vaddq_u8(s, vextq_u8::<14>(zero, s));
                s = vaddq_u8(s, vextq_u8::<12>(zero, s));
                s = vaddq_u8(s, vextq_u8::<8>(zero, s));
                s = vaddq_u8(s, $a);
                $a = vdupq_laneq_u8::<15>(s);
                s
            }};
        }
        let dst_ptr = dst.as_mut_ptr();
        let mut a0 = vdupq_n_u8(acc[0]);
        let mut a1 = vdupq_n_u8(acc[1]);
        if stride == 4 {
            let mut a2 = vdupq_n_u8(acc[2]);
            let mut a3 = vdupq_n_u8(acc[3]);
            let (p0, p1, p2, p3) = (p_ptrs[0], p_ptrs[1], p_ptrs[2], p_ptrs[3]);
            if const_plane[3] && !const_plane[0] && !const_plane[1] && !const_plane[2] {
                if sub_green {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        vst4q_u8(
                            dst_ptr.add(idx * 4),
                            uint8x16x4_t(vaddq_u8(s0, s1), s1, vaddq_u8(s2, s1), a3),
                        );
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        vst4q_u8(dst_ptr.add(idx * 4), uint8x16x4_t(s0, s1, s2, a3));
                        idx += 16;
                    }
                }
            } else if !const_plane[0] && !const_plane[1] && !const_plane[2] && !const_plane[3] {
                if sub_green {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        let s3 = scan!(vld1q_u8(p3.add(idx)), a3);
                        vst4q_u8(
                            dst_ptr.add(idx * 4),
                            uint8x16x4_t(vaddq_u8(s0, s1), s1, vaddq_u8(s2, s1), s3),
                        );
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                        let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                        let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                        let s3 = scan!(vld1q_u8(p3.add(idx)), a3);
                        vst4q_u8(dst_ptr.add(idx * 4), uint8x16x4_t(s0, s1, s2, s3));
                        idx += 16;
                    }
                }
            }
            acc[0] = vgetq_lane_u8::<0>(a0);
            acc[1] = vgetq_lane_u8::<0>(a1);
            acc[2] = vgetq_lane_u8::<0>(a2);
            acc[3] = vgetq_lane_u8::<0>(a3);
        } else if stride == 3 && !const_plane[0] && !const_plane[1] && !const_plane[2] {
            let mut a2 = vdupq_n_u8(acc[2]);
            let (p0, p1, p2) = (p_ptrs[0], p_ptrs[1], p_ptrs[2]);
            if sub_green {
                while idx + 16 <= count {
                    let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                    let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                    let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                    vst3q_u8(
                        dst_ptr.add(idx * 3),
                        uint8x16x3_t(vaddq_u8(s0, s1), s1, vaddq_u8(s2, s1)),
                    );
                    idx += 16;
                }
            } else {
                while idx + 16 <= count {
                    let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                    let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                    let s2 = scan!(vld1q_u8(p2.add(idx)), a2);
                    vst3q_u8(dst_ptr.add(idx * 3), uint8x16x3_t(s0, s1, s2));
                    idx += 16;
                }
            }
            acc[0] = vgetq_lane_u8::<0>(a0);
            acc[1] = vgetq_lane_u8::<0>(a1);
            acc[2] = vgetq_lane_u8::<0>(a2);
        } else if stride == 2 && !const_plane[0] && !const_plane[1] {
            let (p0, p1) = (p_ptrs[0], p_ptrs[1]);
            while idx + 16 <= count {
                let s0 = scan!(vld1q_u8(p0.add(idx)), a0);
                let s1 = scan!(vld1q_u8(p1.add(idx)), a1);
                vst2q_u8(dst_ptr.add(idx * 2), uint8x16x2_t(s0, s1));
                idx += 16;
            }
            acc[0] = vgetq_lane_u8::<0>(a0);
            acc[1] = vgetq_lane_u8::<0>(a1);
        }
    }
    while idx < count {
        for p in 0..stride {
            if !const_plane[p] {
                acc[p] = acc[p].wrapping_add(unsafe { *p_ptrs[p].add(idx) });
            }
        }
        if sub_green {
            let g = acc[1];
            dst[idx * stride] = acc[0].wrapping_add(g);
            dst[idx * stride + 1] = g;
            dst[idx * stride + 2] = acc[2].wrapping_add(g);
            if stride == 4 {
                dst[idx * stride + 3] = acc[3];
            }
        } else if stride == 8 {
            unsafe {
                dst.as_mut_ptr()
                    .cast::<u64>()
                    .add(idx)
                    .write_unaligned(u64::from_le_bytes(acc));
            }
        } else {
            for p in 0..stride {
                dst[idx * stride + p] = acc[p];
            }
        }
        idx += 1;
    }
}

/// Detect and pack fixed-stride pixel / numeric / binary records (`stride in {2, 3, 4, 8}`)
/// using byte-plane transpose + optional Subtract-Green color decorrelation + horizontal delta.
///
/// Format of `STRIDE_PACKET` (`selector == 2`):
/// - `raw_len: ULEB`
/// - `prefix_len: ULEB`, followed by `prefix_bytes[..prefix_len]`
/// - `stride: u8` (`2`, `3`, `4`, or `8`), `mode: u8` (`1` = 2-level 64-bit word zero-mask,
///   `2` = per-plane `FAST_LZ`, `3` = Subtract-Green + per-plane `FAST_LZ`)
fn pack_stride_into(data: &[u8], out: &mut Vec<u8>) -> Option<()> {
    let n = data.len();
    if n < 256 || n > MAX_BLOCK_SIZE {
        return None;
    }
    // Quick check: detect BMP header (`BM` + data_offset at 10..14 and bpp at 28..30)
    // or test strides 2, 3, 4, and 8 on prefix_len = 0.
    let (prefix_len, stride) = if n >= 54 && data[0] == b'B' && data[1] == b'M' {
        let off = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
        let bpp = u16::from_le_bytes(data[28..30].try_into().unwrap());
        let s = match bpp {
            16 => 2usize,
            24 => 3usize,
            32 => 4usize,
            _ => 4usize,
        };
        if off <= 1024 && off + s * 64 <= n {
            (off, s)
        } else {
            (0usize, 4usize)
        }
    } else {
        // Quick 32-byte midpoint reject before scanning four 128-byte windows:
        let mid = (n / 2).saturating_sub(16);
        let quick = &data[mid..(mid + 32).min(n)];
        let mut q2 = 0usize;
        let mut q3 = 0usize;
        let mut q4 = 0usize;
        let mut q8 = 0usize;
        let mut q1 = 0usize;
        for i in 8..quick.len() {
            q2 += (quick[i] == quick[i - 2]) as usize;
            q3 += (quick[i] == quick[i - 3]) as usize;
            q4 += (quick[i] == quick[i - 4]) as usize;
            q8 += (quick[i] == quick[i - 8]) as usize;
            q1 += (quick[i] == quick[i - 1]) as usize;
        }
        let max_q = q2.max(q3).max(q4).max(q8);
        if max_q < 7 || max_q <= q1 + 2 {
            return None;
        }
        // Sample four 128-byte windows across the block at strides 2, 3, 4, and 8:
        let mut z2 = 0usize;
        let mut z3 = 0usize;
        let mut z4 = 0usize;
        let mut z8 = 0usize;
        let mut z1 = 0usize;
        let mut total = 0usize;
        let win = 128.min(n);
        for q in 0..4 {
            let start = ((n - win) * q) / 3;
            let sample = &data[start..start + win];
            for i in 8..sample.len() {
                z2 += (sample[i] == sample[i - 2]) as usize;
                z3 += (sample[i] == sample[i - 3]) as usize;
                z4 += (sample[i] == sample[i - 4]) as usize;
                z8 += (sample[i] == sample[i - 8]) as usize;
                z1 += (sample[i] == sample[i - 1]) as usize;
            }
            total += sample.len() - 8;
        }
        let max_z = z2.max(z3).max(z4).max(z8);
        // Require at least 45% byte-wise horizontal repetition and clear dominance over 1-byte RLE:
        if max_z * 100 < total * 45 || max_z <= z1 + total / 8 {
            return None;
        }
        let (even_s, even_z) = if z8 > z4 + total / 10 && z8 > z2 + total / 10 {
            (8usize, z8)
        } else if z2 > z4 + 6 {
            (2usize, z2)
        } else {
            (4usize, z4)
        };
        if z3 > even_z + 8 {
            (0usize, 3usize)
        } else {
            (0usize, even_s)
        }
    };

    let body = &data[prefix_len..];
    let count = body.len() / stride;
    if count < 64 {
        return None;
    }
    let tail = &body[count * stride..];

    // Sample 256 pixels across 4 windows to decide whether Subtract-Green reduces L1 residual energy
    // (only applicable to 3-channel RGB and 4-channel RGBA):
    let use_sub_green = if stride == 3 || stride == 4 {
        let mut raw_cost = 0usize;
        let mut sg_cost = 0usize;
        let swin = 64.min(count);
        for q in 0..4 {
            let base_px = ((count - swin) * q) / 3;
            for i in (base_px + 1)..(base_px + swin) {
                let o = i * stride;
                let po = (i - 1) * stride;
                let dr = body[o].wrapping_sub(body[po]);
                let dg = body[o + 1].wrapping_sub(body[po + 1]);
                let db = body[o + 2].wrapping_sub(body[po + 2]);
                let dr_sg = dr.wrapping_sub(dg);
                let db_sg = db.wrapping_sub(dg);
                raw_cost += (dr != 0) as usize
                    + (db != 0) as usize
                    + (dr.wrapping_add(1) > 2) as usize
                    + (db.wrapping_add(1) > 2) as usize;
                sg_cost += (dr_sg != 0) as usize
                    + (db_sg != 0) as usize
                    + (dr_sg.wrapping_add(1) > 2) as usize
                    + (db_sg.wrapping_add(1) > 2) as usize;
            }
        }
        sg_cost + 8 < raw_cost
    } else {
        false
    };

    let start_len = out.len();
    let mode_byte = if use_sub_green { 3u8 } else { 2u8 };
    out.push(STRIDE_PACKET);
    put_uleb(out, n);
    put_uleb(out, prefix_len);
    out.extend_from_slice(&data[..prefix_len]);
    out.push(stride as u8);
    out.push(mode_byte);

    let ok = STRIDE_COMBINED.with(|cell| {
        let mut comb = cell.borrow_mut();
        let total_cap = stride * count + 64;
        if comb.len() < total_cap {
            comb.resize(total_cap, 0);
        }

        let mut first_bytes = [0u8; 8];
        let mut all_zero = [false; 8];
        let body_ptr = body.as_ptr();
        let comb_ptr = comb.as_mut_ptr();

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            if stride == 4 && count >= 16 {
                let p0 = comb_ptr;
                let p1 = comb_ptr.add(count);
                let p2 = comb_ptr.add(2 * count);
                let p3 = comb_ptr.add(3 * count);
                let f0 = *body_ptr;
                let f1 = *body_ptr.add(1);
                let f2 = *body_ptr.add(2);
                let f3 = *body_ptr.add(3);
                first_bytes = if use_sub_green {
                    [f0.wrapping_sub(f1), f1, f2.wrapping_sub(f1), f3, 0, 0, 0, 0]
                } else {
                    [f0, f1, f2, f3, 0, 0, 0, 0]
                };
                let mut prev0 = vdupq_n_u8(first_bytes[0]);
                let mut prev1 = vdupq_n_u8(first_bytes[1]);
                let mut prev2 = vdupq_n_u8(first_bytes[2]);
                let mut prev3 = vdupq_n_u8(first_bytes[3]);
                let mut or0 = vdupq_n_u8(0);
                let mut or1 = vdupq_n_u8(0);
                let mut or2 = vdupq_n_u8(0);
                let mut or3 = vdupq_n_u8(0);
                let mut idx = 0usize;
                if use_sub_green {
                    while idx + 16 <= count {
                        let cur4 = vld4q_u8(body_ptr.add(idx * 4));
                        let c0 = vsubq_u8(cur4.0, cur4.1);
                        let c1 = cur4.1;
                        let c2 = vsubq_u8(cur4.2, cur4.1);
                        let c3 = cur4.3;
                        let lag0 = vextq_u8::<15>(prev0, c0);
                        let lag1 = vextq_u8::<15>(prev1, c1);
                        let lag2 = vextq_u8::<15>(prev2, c2);
                        let lag3 = vextq_u8::<15>(prev3, c3);
                        prev0 = c0;
                        prev1 = c1;
                        prev2 = c2;
                        prev3 = c3;
                        let d0 = vsubq_u8(c0, lag0);
                        let d1 = vsubq_u8(c1, lag1);
                        let d2 = vsubq_u8(c2, lag2);
                        let d3 = vsubq_u8(c3, lag3);
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        or3 = vorrq_u8(or3, d3);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        vst1q_u8(p3.add(idx), d3);
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let cur4 = vld4q_u8(body_ptr.add(idx * 4));
                        let lag0 = vextq_u8::<15>(prev0, cur4.0);
                        let lag1 = vextq_u8::<15>(prev1, cur4.1);
                        let lag2 = vextq_u8::<15>(prev2, cur4.2);
                        let lag3 = vextq_u8::<15>(prev3, cur4.3);
                        prev0 = cur4.0;
                        prev1 = cur4.1;
                        prev2 = cur4.2;
                        prev3 = cur4.3;
                        let d0 = vsubq_u8(cur4.0, lag0);
                        let d1 = vsubq_u8(cur4.1, lag1);
                        let d2 = vsubq_u8(cur4.2, lag2);
                        let d3 = vsubq_u8(cur4.3, lag3);
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        or3 = vorrq_u8(or3, d3);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        vst1q_u8(p3.add(idx), d3);
                        idx += 16;
                    }
                }
                let mut s_or = [
                    vmaxvq_u8(or0),
                    vmaxvq_u8(or1),
                    vmaxvq_u8(or2),
                    vmaxvq_u8(or3),
                ];
                let ptrs = [p0, p1, p2, p3];
                while idx < count {
                    let g = *body_ptr.add(idx * 4 + 1);
                    let pg = *body_ptr.add((idx - 1) * 4 + 1);
                    let dg = g.wrapping_sub(pg);
                    for p in 0..4 {
                        let cur = *body_ptr.add(idx * 4 + p);
                        let prev = *body_ptr.add((idx - 1) * 4 + p);
                        let d = cur.wrapping_sub(prev);
                        let v = if use_sub_green && (p == 0 || p == 2) {
                            d.wrapping_sub(dg)
                        } else {
                            d
                        };
                        *ptrs[p].add(idx) = v;
                        s_or[p] |= v;
                    }
                    idx += 1;
                }
                all_zero = [
                    s_or[0] == 0,
                    s_or[1] == 0,
                    s_or[2] == 0,
                    s_or[3] == 0,
                    false,
                    false,
                    false,
                    false,
                ];
            } else if stride == 3 && count >= 16 {
                let p0 = comb_ptr;
                let p1 = comb_ptr.add(count);
                let p2 = comb_ptr.add(2 * count);
                let f0 = *body_ptr;
                let f1 = *body_ptr.add(1);
                let f2 = *body_ptr.add(2);
                first_bytes = if use_sub_green {
                    [f0.wrapping_sub(f1), f1, f2.wrapping_sub(f1), 0, 0, 0, 0, 0]
                } else {
                    [f0, f1, f2, 0, 0, 0, 0, 0]
                };
                let mut prev0 = vdupq_n_u8(first_bytes[0]);
                let mut prev1 = vdupq_n_u8(first_bytes[1]);
                let mut prev2 = vdupq_n_u8(first_bytes[2]);
                let mut or0 = vdupq_n_u8(0);
                let mut or1 = vdupq_n_u8(0);
                let mut or2 = vdupq_n_u8(0);
                let mut idx = 0usize;
                if use_sub_green {
                    while idx + 16 <= count {
                        let cur3 = vld3q_u8(body_ptr.add(idx * 3));
                        let c0 = vsubq_u8(cur3.0, cur3.1);
                        let c1 = cur3.1;
                        let c2 = vsubq_u8(cur3.2, cur3.1);
                        let lag0 = vextq_u8::<15>(prev0, c0);
                        let lag1 = vextq_u8::<15>(prev1, c1);
                        let lag2 = vextq_u8::<15>(prev2, c2);
                        prev0 = c0;
                        prev1 = c1;
                        prev2 = c2;
                        let d0 = vsubq_u8(c0, lag0);
                        let d1 = vsubq_u8(c1, lag1);
                        let d2 = vsubq_u8(c2, lag2);
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        idx += 16;
                    }
                } else {
                    while idx + 16 <= count {
                        let cur3 = vld3q_u8(body_ptr.add(idx * 3));
                        let lag0 = vextq_u8::<15>(prev0, cur3.0);
                        let lag1 = vextq_u8::<15>(prev1, cur3.1);
                        let lag2 = vextq_u8::<15>(prev2, cur3.2);
                        prev0 = cur3.0;
                        prev1 = cur3.1;
                        prev2 = cur3.2;
                        let d0 = vsubq_u8(cur3.0, lag0);
                        let d1 = vsubq_u8(cur3.1, lag1);
                        let d2 = vsubq_u8(cur3.2, lag2);
                        or0 = vorrq_u8(or0, d0);
                        or1 = vorrq_u8(or1, d1);
                        or2 = vorrq_u8(or2, d2);
                        vst1q_u8(p0.add(idx), d0);
                        vst1q_u8(p1.add(idx), d1);
                        vst1q_u8(p2.add(idx), d2);
                        idx += 16;
                    }
                }
                let mut s_or = [vmaxvq_u8(or0), vmaxvq_u8(or1), vmaxvq_u8(or2)];
                let ptrs = [p0, p1, p2];
                while idx < count {
                    let g = *body_ptr.add(idx * 3 + 1);
                    let pg = *body_ptr.add((idx - 1) * 3 + 1);
                    let dg = g.wrapping_sub(pg);
                    for p in 0..3 {
                        let cur = *body_ptr.add(idx * 3 + p);
                        let prev = *body_ptr.add((idx - 1) * 3 + p);
                        let d = cur.wrapping_sub(prev);
                        let v = if use_sub_green && (p == 0 || p == 2) {
                            d.wrapping_sub(dg)
                        } else {
                            d
                        };
                        *ptrs[p].add(idx) = v;
                        s_or[p] |= v;
                    }
                    idx += 1;
                }
                all_zero = [
                    s_or[0] == 0,
                    s_or[1] == 0,
                    s_or[2] == 0,
                    false,
                    false,
                    false,
                    false,
                    false,
                ];
            } else if stride == 2 && count >= 16 {
                let p0 = comb_ptr;
                let p1 = comb_ptr.add(count);
                first_bytes[0] = *body_ptr;
                first_bytes[1] = *body_ptr.add(1);
                let mut prev0 = vdupq_n_u8(first_bytes[0]);
                let mut prev1 = vdupq_n_u8(first_bytes[1]);
                let mut or0 = vdupq_n_u8(0);
                let mut or1 = vdupq_n_u8(0);
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let cur2 = vld2q_u8(body_ptr.add(idx * 2));
                    let lag0 = vextq_u8::<15>(prev0, cur2.0);
                    let lag1 = vextq_u8::<15>(prev1, cur2.1);
                    prev0 = cur2.0;
                    prev1 = cur2.1;
                    let d0 = vsubq_u8(cur2.0, lag0);
                    let d1 = vsubq_u8(cur2.1, lag1);
                    or0 = vorrq_u8(or0, d0);
                    or1 = vorrq_u8(or1, d1);
                    vst1q_u8(p0.add(idx), d0);
                    vst1q_u8(p1.add(idx), d1);
                    idx += 16;
                }
                let mut s_or = [vmaxvq_u8(or0), vmaxvq_u8(or1)];
                let ptrs = [p0, p1];
                while idx < count {
                    for p in 0..2 {
                        let d = (*body_ptr.add(idx * 2 + p)).wrapping_sub(*body_ptr.add((idx - 1) * 2 + p));
                        *ptrs[p].add(idx) = d;
                        s_or[p] |= d;
                    }
                    idx += 1;
                }
                all_zero[0] = s_or[0] == 0;
                all_zero[1] = s_or[1] == 0;
            } else {
                for p in 0..stride {
                    first_bytes[p] = body[p];
                    let mut prev = first_bytes[p];
                    let mut s_or = 0u8;
                    let p_slice = &mut comb[p * count..(p + 1) * count];
                    p_slice[0] = 0;
                    for idx in 1..count {
                        let cur = body[idx * stride + p];
                        let d = cur.wrapping_sub(prev);
                        p_slice[idx] = d;
                        s_or |= d;
                        prev = cur;
                    }
                    all_zero[p] = s_or == 0;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for p in 0..stride {
                let f = body[p];
                first_bytes[p] = if use_sub_green && (p == 0 || p == 2) {
                    f.wrapping_sub(body[1])
                } else {
                    f
                };
                let mut prev = first_bytes[p];
                let mut s_or = 0u8;
                let p_slice = &mut comb[p * count..(p + 1) * count];
                p_slice[0] = 0;
                for idx in 1..count {
                    let cur = if use_sub_green && (p == 0 || p == 2) {
                        body[idx * stride + p].wrapping_sub(body[idx * stride + 1])
                    } else {
                        body[idx * stride + p]
                    };
                    let d = cur.wrapping_sub(prev);
                    p_slice[idx] = d;
                    s_or |= d;
                    prev = cur;
                }
                all_zero[p] = s_or == 0;
            }
        }

        encode_stride_planes(
            out,
            &comb[..stride * count],
            &first_bytes,
            &all_zero,
            stride,
            count,
            start_len + n - tail.len(),
        )
    });

    if ok {
        out.extend_from_slice(tail);
        if out.len() - start_len < n {
            return Some(());
        }
    }
    out.truncate(start_len);
    None
}

/// Append `stride` delta planes (first byte, kind, payload) from contiguous `comb[0..stride*count]` to `m`:
/// - `kind = 4`: constant plane (`all_zero[p] == true`, 0 payload bytes!)
/// - `kind = 3`: combined `FAST_LZ_PACKET` across `run_len >= 2` consecutive dense planes (zero-copy slice!)
/// - `kind = 2`: 2-level 64-bit hierarchical zero-mask (ultra-fast for sparse planes with `>= 88%` zero words)
/// - `kind = 1`: nested `FAST_LZ_PACKET`
/// - `kind = 0`: raw plane
/// Returns `false` once `m` reaches `limit`.
fn encode_stride_planes(
    m: &mut Vec<u8>,
    comb: &[u8],
    firsts: &[u8; 8],
    all_zero: &[bool; 8],
    stride: usize,
    count: usize,
    limit: usize,
) -> bool {
    let words = count.div_ceil(64);
    let full_words = count / 64;
    let super_words = words.div_ceil(64);
    let step = (full_words / 16).max(1);
    let mut sparse_cand = [false; 8];
    for p in 0..stride {
        if all_zero[p] {
            continue;
        }
        let p64_base = unsafe { comb.as_ptr().add(p * count).cast::<u64>() };
        let mut sparse_est = 0usize;
        let mut sw_i = 0usize;
        let mut sampled = 0usize;
        while sw_i < full_words {
            let u = unsafe {
                p64_base.add(sw_i * 8).read_unaligned()
                    | p64_base.add(sw_i * 8 + 1).read_unaligned()
                    | p64_base.add(sw_i * 8 + 2).read_unaligned()
                    | p64_base.add(sw_i * 8 + 3).read_unaligned()
                    | p64_base.add(sw_i * 8 + 4).read_unaligned()
                    | p64_base.add(sw_i * 8 + 5).read_unaligned()
                    | p64_base.add(sw_i * 8 + 6).read_unaligned()
                    | p64_base.add(sw_i * 8 + 7).read_unaligned()
            };
            sparse_est += (u != 0) as usize;
            sampled += 1;
            sw_i += step;
        }
        sparse_cand[p] = sampled > 0 && sparse_est * 8 <= sampled;
    }

    let mut lz = Vec::with_capacity(count * stride);
    let mut p = 0usize;
    while p < stride {
        m.push(firsts[p]);
        if all_zero[p] {
            m.push(4);
            if m.len() >= limit {
                return false;
            }
            p += 1;
            continue;
        }

        let plane = &comb[p * count..(p + 1) * count];

        if sparse_cand[p] {
            let p_base = plane.as_ptr();
            let mark = m.len();
            let used_kind2 = STRIDE_SCRATCH.with(|cell| {
                let mut sc = cell.borrow_mut();
                if sc.super_mask_buf.len() < super_words {
                    sc.super_mask_buf.resize(super_words, 0);
                }
                if sc.mask_buf.len() < words {
                    sc.mask_buf.resize(words, 0);
                }
                if sc.nz_buf.len() < count {
                    sc.nz_buf.resize(count, 0);
                }
                sc.super_mask_buf[..super_words].fill(0);
                let super_ptr = sc.super_mask_buf.as_mut_ptr();
                let mask_ptr = sc.mask_buf.as_mut_ptr();
                let nz_ptr = sc.nz_buf.as_mut_ptr();
                let mut nz_masks_len = 0usize;
                let mut nz_len = 0usize;
                for w in 0..words {
                    let (w0, w1, w2, w3, w4, w5, w6, w7) = if w < full_words {
                        let p64 = unsafe { p_base.add(w * 64).cast::<u64>() };
                        unsafe {
                            (
                                p64.add(0).read_unaligned().to_le(),
                                p64.add(1).read_unaligned().to_le(),
                                p64.add(2).read_unaligned().to_le(),
                                p64.add(3).read_unaligned().to_le(),
                                p64.add(4).read_unaligned().to_le(),
                                p64.add(5).read_unaligned().to_le(),
                                p64.add(6).read_unaligned().to_le(),
                                p64.add(7).read_unaligned().to_le(),
                            )
                        }
                    } else {
                        let mut tail_buf = [0u8; 64];
                        let rem = count - w * 64;
                        tail_buf[..rem].copy_from_slice(&plane[w * 64..count]);
                        let p64 = tail_buf.as_ptr().cast::<u64>();
                        unsafe {
                            (
                                p64.add(0).read_unaligned().to_le(),
                                p64.add(1).read_unaligned().to_le(),
                                p64.add(2).read_unaligned().to_le(),
                                p64.add(3).read_unaligned().to_le(),
                                p64.add(4).read_unaligned().to_le(),
                                p64.add(5).read_unaligned().to_le(),
                                p64.add(6).read_unaligned().to_le(),
                                p64.add(7).read_unaligned().to_le(),
                            )
                        }
                    };
                    if (w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7) == 0 {
                        continue;
                    }
                    let mut mask = 0u64;
                    let chunks = [w0, w1, w2, w3, w4, w5, w6, w7];
                    for (q, &chunk) in chunks.iter().enumerate() {
                        let mut c = chunk;
                        while c != 0 {
                            let b = (c.trailing_zeros() >> 3) as usize;
                            let v = (c >> (b * 8)) as u8;
                            mask |= 1u64 << (q * 8 + b);
                            unsafe {
                                *nz_ptr.add(nz_len) = v;
                            }
                            nz_len += 1;
                            c &= !(0xffu64 << (b * 8));
                        }
                    }
                    unsafe {
                        let sw = w >> 6;
                        let sb = w & 63;
                        let cur_sm = u64::from_le(*super_ptr.add(sw)) | (1u64 << sb);
                        *super_ptr.add(sw) = cur_sm.to_le();
                        *mask_ptr.add(nz_masks_len) = mask.to_le();
                    }
                    nz_masks_len += 1;
                }
                let k2_size = uleb_len(nz_masks_len)
                    + uleb_len(nz_len)
                    + super_words * 8
                    + nz_masks_len * 8
                    + nz_len;
                if k2_size * 12 < count {
                    m.push(2);
                    put_uleb(m, nz_masks_len);
                    put_uleb(m, nz_len);
                    let super_bytes = unsafe {
                        std::slice::from_raw_parts(super_ptr.cast::<u8>(), super_words * 8)
                    };
                    let mask_bytes = unsafe {
                        std::slice::from_raw_parts(mask_ptr.cast::<u8>(), nz_masks_len * 8)
                    };
                    let nz_bytes = unsafe { std::slice::from_raw_parts(nz_ptr, nz_len) };
                    m.extend_from_slice(super_bytes);
                    m.extend_from_slice(mask_bytes);
                    m.extend_from_slice(nz_bytes);
                    true
                } else {
                    false
                }
            });
            if used_kind2 {
                if m.len() >= limit {
                    return false;
                }
                p += 1;
                continue;
            }
            m.truncate(mark);
        }

        // Count consecutive dense planes starting at `p` (`p .. p + run_len`), already contiguous in `comb`!
        let mut run_len = 1usize;
        while p + run_len < stride && !all_zero[p + run_len] && !sparse_cand[p + run_len] {
            run_len += 1;
        }
        if run_len >= 2 && run_len * count <= MAX_BLOCK_SIZE {
            let comb_slice = &comb[p * count..(p + run_len) * count];
            lz.clear();
            if pack_fast_lz_into(comb_slice, &mut lz).is_some() && lz.len() + 2 < comb_slice.len() {
                m.push(3);
                m.push(run_len as u8);
                for k in 1..run_len {
                    m.push(firsts[p + k]);
                }
                put_uleb(m, lz.len());
                m.extend_from_slice(&lz);
                if m.len() >= limit {
                    return false;
                }
                p += run_len;
                continue;
            }
        }

        lz.clear();
        if pack_fast_lz_into(plane, &mut lz).is_some() && lz.len() < count {
            m.push(1);
            put_uleb(m, lz.len());
            m.extend_from_slice(&lz);
        } else {
            m.push(0);
            m.extend_from_slice(plane);
        }
        if m.len() >= limit {
            return false;
        }
        p += 1;
    }
    true
}

fn unpack_stride_into(cursor: &mut Cursor<'_>, out: &mut [u8]) -> Result<()> {
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE || raw_len != out.len() {
        return Err(invalid("stride raw length"));
    }
    let prefix_len = cursor.uleb()?;
    if prefix_len > raw_len {
        return Err(invalid("stride prefix length"));
    }
    let prefix = cursor.take(prefix_len)?;
    out[..prefix_len].copy_from_slice(prefix);

    let stride = cursor.byte()? as usize;
    let mode = cursor.byte()?;
    if !matches!(stride, 2 | 3 | 4 | 8) || !matches!(mode, 1..=3) || (mode == 3 && !matches!(stride, 3 | 4)) {
        return Err(invalid("stride header"));
    }
    let body_len = raw_len - prefix_len;
    let count = body_len / stride;
    if count < 64 {
        return Err(invalid("stride count"));
    }
    let tail_len = body_len - count * stride;
    let words = count.div_ceil(64);
    let super_words = words.div_ceil(64);
    let super_pad_bits = super_words * 64 - words;
    let pad_bits = words * 64 - count;
    let plane_cap = words * 64;

    if mode == 2 || mode == 3 {
        let sub_green = mode == 3;
        return STRIDE_PLANES2.with(|cell| {
            let mut planes = cell.borrow_mut();
            STRIDE_COMBINED.with(|comb_cell| {
                let mut combined = comb_cell.borrow_mut();
                let mut firsts = [0u8; 8];
                let mut const_plane = [false; 8];
                let mut p_ptrs = [std::ptr::null::<u8>(); 8];
                let mut p = 0usize;
                while p < stride {
                    firsts[p] = cursor.byte()?;
                    let kind = cursor.byte()?;
                    match kind {
                        4 => {
                            const_plane[p] = true;
                            p += 1;
                        }
                        3 => {
                            let run_len = cursor.byte()? as usize;
                            if run_len < 2 || p + run_len > stride {
                                return Err(invalid("stride combined run"));
                            }
                            for k in 1..run_len {
                                firsts[p + k] = cursor.byte()?;
                            }
                            let comb_len = run_len * count;
                            let len = cursor.uleb()?;
                            if len >= comb_len {
                                return Err(invalid("stride plane length"));
                            }
                            let mut sub = Cursor::new(cursor.take(len)?);
                            if sub.byte()? != FAST_LZ_PACKET {
                                return Err(invalid("stride plane packet"));
                            }
                            let req_end = (p + run_len) * count;
                            if combined.len() < req_end {
                                combined.resize(req_end, 0);
                            }
                            let dst = &mut combined[p * count..req_end];
                            unpack_fast_lz_into(&mut sub, dst)?;
                            let dst_ptr = dst.as_ptr();
                            for k in 0..run_len {
                                if dst[k * count] != 0 {
                                    return Err(invalid("stride plane first delta"));
                                }
                                p_ptrs[p + k] = unsafe { dst_ptr.add(k * count) };
                            }
                            p += run_len;
                        }
                        0 => {
                            if planes[p].len() < plane_cap {
                                planes[p].resize(plane_cap, 0);
                            }
                            let dst = &mut planes[p][..count];
                            dst.copy_from_slice(cursor.take(count)?);
                            if dst[0] != 0 {
                                return Err(invalid("stride plane first delta"));
                            }
                            p_ptrs[p] = planes[p].as_ptr();
                            p += 1;
                        }
                        1 => {
                            if planes[p].len() < plane_cap {
                                planes[p].resize(plane_cap, 0);
                            }
                            let dst = &mut planes[p][..count];
                            let len = cursor.uleb()?;
                            if len >= count {
                                return Err(invalid("stride plane length"));
                            }
                            let mut sub = Cursor::new(cursor.take(len)?);
                            if sub.byte()? != FAST_LZ_PACKET {
                                return Err(invalid("stride plane packet"));
                            }
                            unpack_fast_lz_into(&mut sub, dst)?;
                            if dst[0] != 0 {
                                return Err(invalid("stride plane first delta"));
                            }
                            p_ptrs[p] = planes[p].as_ptr();
                            p += 1;
                        }
                        2 => {
                            if planes[p].len() < plane_cap {
                                planes[p].resize(plane_cap, 0);
                            }
                            let nz_masks_len = cursor.uleb()?;
                            if nz_masks_len > words {
                                return Err(invalid("stride nonzero mask count"));
                            }
                            let nz_len = cursor.uleb()?;
                            if nz_len >= count || nz_len < nz_masks_len {
                                return Err(invalid("stride nonzero count"));
                            }
                            let super_slice = cursor.take(super_words * 8)?;
                            let mask_slice = cursor.take(nz_masks_len * 8)?;
                            let nz_slice = cursor.take(nz_len)?;
                            let plane_ptr = planes[p].as_mut_ptr();
                            let super_ptr = super_slice.as_ptr().cast::<u64>();
                            let mask_ptr = mask_slice.as_ptr().cast::<u64>();
                            let nz_ptr = nz_slice.as_ptr();
                            let mut nz_mask_idx = 0usize;
                            let mut nz_idx = 0usize;
                            for sw in 0..super_words {
                                let sm = u64::from_le(unsafe { super_ptr.add(sw).read_unaligned() });
                                if sw + 1 == super_words
                                    && super_pad_bits > 0
                                    && (sm >> (64 - super_pad_bits)) != 0
                                {
                                    return Err(invalid("stride supermask padding bits"));
                                }
                                let w_start = sw * 64;
                                let w_end = (w_start + 64).min(words);
                                let w_count = w_end - w_start;
                                if sm == 0 {
                                    unsafe {
                                        std::ptr::write_bytes(
                                            plane_ptr.add(w_start * 64),
                                            0,
                                            w_count * 64,
                                        );
                                    }
                                    continue;
                                }
                                let sm_pop = sm.count_ones() as usize;
                                if nz_mask_idx + sm_pop > nz_masks_len {
                                    return Err(invalid("stride nonzero mask underflow"));
                                }
                                let mut bw = 0usize;
                                while bw < w_count {
                                    let rem_sm = sm >> bw;
                                    if rem_sm == 0 {
                                        unsafe {
                                            std::ptr::write_bytes(
                                                plane_ptr.add((w_start + bw) * 64),
                                                0,
                                                (w_count - bw) * 64,
                                            );
                                        }
                                        break;
                                    }
                                    let z_words =
                                        (rem_sm.trailing_zeros() as usize).min(w_count - bw);
                                    if z_words > 0 {
                                        unsafe {
                                            std::ptr::write_bytes(
                                                plane_ptr.add((w_start + bw) * 64),
                                                0,
                                                z_words * 64,
                                            );
                                        }
                                        bw += z_words;
                                        if bw >= w_count {
                                            break;
                                        }
                                    }
                                    let w = w_start + bw;
                                    let base = w * 64;
                                    let mut m = u64::from_le(unsafe {
                                        mask_ptr.add(nz_mask_idx).read_unaligned()
                                    });
                                    nz_mask_idx += 1;
                                    if m == 0 {
                                        return Err(invalid("stride canonical zero mask"));
                                    }
                                    if w == 0 && (m & 1) != 0 {
                                        return Err(invalid("stride first byte bit"));
                                    }
                                    if w + 1 == words
                                        && pad_bits > 0
                                        && (m >> (64 - pad_bits)) != 0
                                    {
                                        return Err(invalid("stride padding bits"));
                                    }
                                    let pop = m.count_ones() as usize;
                                    if nz_idx + pop > nz_len {
                                        return Err(invalid("stride nonzero underflow"));
                                    }
                                    unsafe {
                                        std::ptr::write_bytes(plane_ptr.add(base), 0, 64);
                                    }
                                    while m != 0 {
                                        let bit = m.trailing_zeros() as usize;
                                        m &= m - 1;
                                        unsafe {
                                            let v = *nz_ptr.add(nz_idx);
                                            if v == 0 {
                                                return Err(invalid(
                                                    "stride canonical zero residual",
                                                ));
                                            }
                                            *plane_ptr.add(base + bit) = v;
                                        }
                                        nz_idx += 1;
                                    }
                                    bw += 1;
                                }
                            }
                            if nz_mask_idx != nz_masks_len || nz_idx != nz_len {
                                return Err(invalid("stride nonzero consumption"));
                            }
                            p_ptrs[p] = planes[p].as_ptr();
                            p += 1;
                        }
                        _ => return Err(invalid("stride plane kind")),
                    }
                }
                let tail = cursor.take(tail_len)?;
                if !cursor.done() {
                    return Err(invalid("stride trailing bytes"));
                }
                let px = &mut out[prefix_len..prefix_len + count * stride];
                prefix_interleave_planes(p_ptrs, const_plane, &firsts, stride, count, sub_green, px);
                out[prefix_len + count * stride..].copy_from_slice(tail);
                Ok(())
            })
        });
    }
    let words = count.div_ceil(64);
    let super_words = words.div_ceil(64);
    let super_pad_bits = super_words * 64 - words;
    let pad_bits = words * 64 - count;
    let plane_cap = words * 64;

    STRIDE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        for p in 0..stride {
            if scratch.planes[p].len() < plane_cap {
                scratch.planes[p].resize(plane_cap, 0);
            }
        }

        for p in 0..stride {
            let first_byte = cursor.byte()?;
            let nz_masks_len = cursor.uleb()?;
            if nz_masks_len > words {
                return Err(invalid("stride nonzero mask count"));
            }
            let nz_len = cursor.uleb()?;
            if nz_len >= count || nz_len < nz_masks_len {
                return Err(invalid("stride nonzero count"));
            }
            let super_slice = cursor.take(super_words * 8)?;
            let mask_slice = cursor.take(nz_masks_len * 8)?;
            let nz_slice = cursor.take(nz_len)?;
            let plane_ptr = scratch.planes[p].as_mut_ptr();
            let super_ptr = super_slice.as_ptr().cast::<u64>();
            let mask_ptr = mask_slice.as_ptr().cast::<u64>();
            let nz_ptr = nz_slice.as_ptr();
            let mut nz_mask_idx = 0usize;
            let mut nz_idx = 0usize;
            let mut acc = first_byte;

            for sw in 0..super_words {
                let sm = u64::from_le(unsafe { super_ptr.add(sw).read_unaligned() });
                if sw + 1 == super_words
                    && super_pad_bits > 0
                    && (sm >> (64 - super_pad_bits)) != 0
                {
                    return Err(invalid("stride supermask padding bits"));
                }
                let w_start = sw * 64;
                let w_end = (w_start + 64).min(words);
                let w_count = w_end - w_start;
                if sm == 0 {
                    unsafe {
                        std::ptr::write_bytes(plane_ptr.add(w_start * 64), acc, w_count * 64);
                    }
                    continue;
                }
                let sm_pop = sm.count_ones() as usize;
                if nz_mask_idx + sm_pop > nz_masks_len {
                    return Err(invalid("stride nonzero mask underflow"));
                }
                let mut bw = 0usize;
                while bw < w_count {
                    let rem_sm = sm >> bw;
                    if rem_sm == 0 {
                        unsafe {
                            std::ptr::write_bytes(
                                plane_ptr.add((w_start + bw) * 64),
                                acc,
                                (w_count - bw) * 64,
                            );
                        }
                        break;
                    }
                    let z_words = (rem_sm.trailing_zeros() as usize).min(w_count - bw);
                    if z_words > 0 {
                        unsafe {
                            std::ptr::write_bytes(
                                plane_ptr.add((w_start + bw) * 64),
                                acc,
                                z_words * 64,
                            );
                        }
                        bw += z_words;
                        if bw >= w_count {
                            break;
                        }
                    }
                    let w = w_start + bw;
                    let base = w * 64;
                    let mut m = u64::from_le(unsafe { mask_ptr.add(nz_mask_idx).read_unaligned() });
                    nz_mask_idx += 1;
                    if m == 0 {
                        return Err(invalid("stride canonical zero mask"));
                    }
                    if w == 0 && (m & 1) != 0 {
                        return Err(invalid("stride first byte bit"));
                    }
                    if w + 1 == words && pad_bits > 0 && (m >> (64 - pad_bits)) != 0 {
                        return Err(invalid("stride padding bits"));
                    }
                    let pop = m.count_ones() as usize;
                    if nz_idx + pop > nz_len {
                        return Err(invalid("stride nonzero underflow"));
                    }
                    if pop == 1 {
                        let bit = m.trailing_zeros() as usize;
                        let v = unsafe { *nz_ptr.add(nz_idx) };
                        if v == 0 {
                            return Err(invalid("stride canonical zero residual"));
                        }
                        nz_idx += 1;
                        unsafe {
                            if bit > 0 {
                                std::ptr::write_bytes(plane_ptr.add(base), acc, bit);
                            }
                            acc = acc.wrapping_add(v);
                            std::ptr::write_bytes(plane_ptr.add(base + bit), acc, 64 - bit);
                        }
                        bw += 1;
                        continue;
                    }
                    unsafe {
                        std::ptr::write_bytes(plane_ptr.add(base), 0, 64);
                    }
                    while m != 0 {
                        let bit = m.trailing_zeros() as usize;
                        m &= m - 1;
                        unsafe {
                            let v = *nz_ptr.add(nz_idx);
                            if v == 0 {
                                return Err(invalid("stride canonical zero residual"));
                            }
                            *plane_ptr.add(base + bit) = v;
                        }
                        nz_idx += 1;
                    }
                    // Fused 64-byte inclusive prefix sum:
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use core::arch::aarch64::*;
                        let zero = vdupq_n_u8(0);
                        for v_idx in 0..4 {
                            let vp = plane_ptr.add(base + v_idx * 16);
                            let mut s = vld1q_u8(vp);
                            s = vaddq_u8(s, vextq_u8::<15>(zero, s));
                            s = vaddq_u8(s, vextq_u8::<14>(zero, s));
                            s = vaddq_u8(s, vextq_u8::<12>(zero, s));
                            s = vaddq_u8(s, vextq_u8::<8>(zero, s));
                            s = vaddq_u8(s, vdupq_n_u8(acc));
                            acc = vgetq_lane_u8::<15>(s);
                            vst1q_u8(vp, s);
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    unsafe {
                        for b in 0..64 {
                            let bp = plane_ptr.add(base + b);
                            acc = acc.wrapping_add(*bp);
                            *bp = acc;
                        }
                    }
                    bw += 1;
                }
            }
            if nz_mask_idx != nz_masks_len || nz_idx != nz_len {
                return Err(invalid("stride nonzero consumption"));
            }
        }

        let tail = cursor.take(tail_len)?;
        if !cursor.done() {
            return Err(invalid("stride trailing bytes"));
        }

        let dst_body = &mut out[prefix_len..prefix_len + count * stride];
        let dst_ptr = dst_body.as_mut_ptr();
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use core::arch::aarch64::*;
            if stride == 4 {
                let p0 = scratch.planes[0].as_ptr();
                let p1 = scratch.planes[1].as_ptr();
                let p2 = scratch.planes[2].as_ptr();
                let p3 = scratch.planes[3].as_ptr();
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let v = uint8x16x4_t(
                        vld1q_u8(p0.add(idx)),
                        vld1q_u8(p1.add(idx)),
                        vld1q_u8(p2.add(idx)),
                        vld1q_u8(p3.add(idx)),
                    );
                    vst4q_u8(dst_ptr.add(idx * 4), v);
                    idx += 16;
                }
                while idx < count {
                    *dst_ptr.add(idx * 4) = *p0.add(idx);
                    *dst_ptr.add(idx * 4 + 1) = *p1.add(idx);
                    *dst_ptr.add(idx * 4 + 2) = *p2.add(idx);
                    *dst_ptr.add(idx * 4 + 3) = *p3.add(idx);
                    idx += 1;
                }
            } else if stride == 2 {
                let p0 = scratch.planes[0].as_ptr();
                let p1 = scratch.planes[1].as_ptr();
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let v = uint8x16x2_t(vld1q_u8(p0.add(idx)), vld1q_u8(p1.add(idx)));
                    vst2q_u8(dst_ptr.add(idx * 2), v);
                    idx += 16;
                }
                while idx < count {
                    *dst_ptr.add(idx * 2) = *p0.add(idx);
                    *dst_ptr.add(idx * 2 + 1) = *p1.add(idx);
                    idx += 1;
                }
            } else if stride == 8 {
                for idx in 0..count {
                    for p in 0..8 {
                        *dst_ptr.add(idx * 8 + p) = *scratch.planes[p].as_ptr().add(idx);
                    }
                }
            } else {
                let p0 = scratch.planes[0].as_ptr();
                let p1 = scratch.planes[1].as_ptr();
                let p2 = scratch.planes[2].as_ptr();
                let mut idx = 0usize;
                while idx + 16 <= count {
                    let v = uint8x16x3_t(
                        vld1q_u8(p0.add(idx)),
                        vld1q_u8(p1.add(idx)),
                        vld1q_u8(p2.add(idx)),
                    );
                    vst3q_u8(dst_ptr.add(idx * 3), v);
                    idx += 16;
                }
                while idx < count {
                    *dst_ptr.add(idx * 3) = *p0.add(idx);
                    *dst_ptr.add(idx * 3 + 1) = *p1.add(idx);
                    *dst_ptr.add(idx * 3 + 2) = *p2.add(idx);
                    idx += 1;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for idx in 0..count {
                for p in 0..stride {
                    dst_body[idx * stride + p] = scratch.planes[p][idx];
                }
            }
        }
        out[prefix_len + count * stride..].copy_from_slice(tail);
        Ok(())
    })
}

/// Pack a low-cardinality byte stream (e.g. Base64, hex/UUID, ASCII identifiers, XML/JSON literals)
/// using a 4-bit (<=16), 5-bit (<=32), or 6-bit (<=64) planar 16-lane bit-packed stream + raw escape stream.
fn pack_palette_stream(data: &[u8], out: &mut Vec<u8>) -> bool {
    let n = data.len();
    if n < 128 {
        return false;
    }
    // Fast 256-byte cardinality probe at start (and middle for large streams) to reject high-entropy binary data
    let s1 = n.min(256);
    let mut bmap = [0u64; 4];
    for &b in &data[..s1] {
        bmap[(b >> 6) as usize] |= 1u64 << (b & 63);
    }
    let sample_uniq =
        bmap[0].count_ones() + bmap[1].count_ones() + bmap[2].count_ones() + bmap[3].count_ones();
    if !(2..=76).contains(&sample_uniq) {
        return false;
    }
    if n >= 1024 {
        let mid = n / 2;
        let mut bmap2 = [0u64; 4];
        for &b in &data[mid..mid + 256] {
            bmap2[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        let mid_uniq = bmap2[0].count_ones()
            + bmap2[1].count_ones()
            + bmap2[2].count_ones()
            + bmap2[3].count_ones();
        if mid_uniq > 78 {
            return false;
        }
    }

    let mut counts = [0u32; 256];
    let chunks4 = n / 4;
    let ptr = data.as_ptr();
    for i in 0..chunks4 {
        unsafe {
            let p = ptr.add(i * 4);
            counts[*p as usize] += 1;
            counts[*p.add(1) as usize] += 1;
            counts[*p.add(2) as usize] += 1;
            counts[*p.add(3) as usize] += 1;
        }
    }
    for &b in &data[chunks4 * 4..] {
        counts[b as usize] += 1;
    }

    let mut syms = [(0u32, 0u8); 256];
    let mut uniq = 0usize;
    for b in 0..256 {
        let c = counts[b];
        if c > 0 {
            syms[uniq] = (c, b as u8);
            uniq += 1;
        }
    }
    if uniq < 2 {
        return false;
    }
    syms[..uniq].sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut best_bits = 0u8;
    let mut best_pal_len = 0usize;
    let mut best_escaped = 0usize;
    let mut best_total = usize::MAX;

    for bits in [4u8, 5, 6] {
        let cap = 1usize << bits;
        let (pal_len, escaped) = if uniq <= cap {
            (uniq, 0usize)
        } else {
            let keep = cap - 1;
            let mut top_sum = 0usize;
            for s in &syms[..keep] {
                top_sum += s.0 as usize;
            }
            (keep, n - top_sum)
        };
        if escaped * 8 > n {
            continue;
        }
        let packed_len = (n * (bits as usize)).div_ceil(8);
        let esc_hdr = if escaped > 0 { uleb_len(escaped) } else { 0 };
        let total = 2 + pal_len + esc_hdr + packed_len + escaped;
        if total < best_total {
            best_total = total;
            best_bits = bits;
            best_pal_len = pal_len;
            best_escaped = escaped;
        }
    }

    if best_bits == 0 || best_total + 16 + (n >> 5) >= n {
        return false;
    }

    let esc_code = if best_escaped > 0 {
        best_pal_len as u8
    } else {
        0
    };
    let mut map = [esc_code; 256];
    let mut pal = [0u8; 64];
    for (idx, &(_, sym)) in syms[..best_pal_len].iter().enumerate() {
        pal[idx] = sym;
        map[sym as usize] = idx as u8;
    }

    let packed_len = (n * (best_bits as usize)).div_ceil(8);
    out.clear();
    out.reserve(best_total + 16);
    out.push(best_bits | if best_escaped > 0 { 0x80 } else { 0 });
    out.push(best_pal_len as u8);
    out.extend_from_slice(&pal[..best_pal_len]);
    if best_escaped > 0 {
        put_uleb(out, best_escaped);
    }

    let packed_start = out.len();
    out.resize(packed_start + packed_len, 0);
    let chunks16 = n / 16;
    let rem = n % 16;

    const M4_64: u64 = 0x0f0f0f0f_0f0f0f0f;
    const M1_64: u64 = 0x01010101_01010101;
    const M2_64: u64 = 0x03030303_03030303;

    if best_escaped == 0 {
        let mut p_dst = unsafe { out.as_mut_ptr().add(packed_start) };
        for blk in 0..chunks16 {
            let base = blk * 16;
            let lo_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + i) } as usize]);
            let hi_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + 8 + i) } as usize]);
            let c_lo = u64::from_le_bytes(lo_arr);
            let c_hi = u64::from_le_bytes(hi_arr);
            let p4 = (c_lo & M4_64) | ((c_hi & M4_64) << 4);
            unsafe {
                p_dst.cast::<u64>().write_unaligned(p4.to_le());
                if best_bits == 4 {
                    p_dst = p_dst.add(8);
                } else if best_bits == 5 {
                    let q0 = ((c_lo >> 4) & M1_64) | (((c_hi >> 4) & M1_64) << 1);
                    let q1 = ((q0 as u32) & 0x03030303) | (((q0 >> 30) as u32) & 0x0c0c0c0c);
                    let q3 = ((q1 & 0x0f0f) | ((q1 >> 12) & 0xf0f0)) as u16;
                    p_dst.add(8).cast::<u16>().write_unaligned(q3.to_le());
                    p_dst = p_dst.add(10);
                } else {
                    let r0 = ((c_lo >> 4) & M2_64) | (((c_hi >> 4) & M2_64) << 2);
                    let p2 = ((r0 as u32) & 0x0f0f0f0f) | (((r0 >> 28) as u32) & 0xf0f0f0f0);
                    p_dst.add(8).cast::<u32>().write_unaligned(p2.to_le());
                    p_dst = p_dst.add(12);
                }
            }
        }
        if rem > 0 {
            let base = chunks16 * 16;
            let mut acc = 0u128;
            for i in 0..rem {
                let b = unsafe { *ptr.add(base + i) };
                acc |= (map[b as usize] as u128) << (i * (best_bits as usize));
            }
            let tail_bytes = (rem * (best_bits as usize)).div_ceil(8);
            let le = acc.to_le_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(le.as_ptr(), p_dst, tail_bytes);
            }
        }
    } else {
        let stride = (best_bits as usize) * 2;
        let esc_mask64 = M1_64 << best_bits;
        for blk in 0..chunks16 {
            let base = blk * 16;
            let lo_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + i) } as usize]);
            let hi_arr: [u8; 8] =
                std::array::from_fn(|i| map[unsafe { *ptr.add(base + 8 + i) } as usize]);
            let c_lo = u64::from_le_bytes(lo_arr);
            let c_hi = u64::from_le_bytes(hi_arr);
            let m_lo = (c_lo + M1_64) & esc_mask64;
            let m_hi = (c_hi + M1_64) & esc_mask64;
            if (m_lo | m_hi) != 0 {
                let mut ml = m_lo;
                while ml != 0 {
                    let lane = (ml.trailing_zeros() >> 3) as usize;
                    out.push(unsafe { *ptr.add(base + lane) });
                    ml &= ml - 1;
                }
                let mut mh = m_hi;
                while mh != 0 {
                    let lane = 8 + ((mh.trailing_zeros() >> 3) as usize);
                    out.push(unsafe { *ptr.add(base + lane) });
                    mh &= mh - 1;
                }
            }
            let p_dst = unsafe { out.as_mut_ptr().add(packed_start + blk * stride) };
            let p4 = (c_lo & M4_64) | ((c_hi & M4_64) << 4);
            unsafe {
                p_dst.cast::<u64>().write_unaligned(p4.to_le());
                if best_bits == 5 {
                    let q0 = ((c_lo >> 4) & M1_64) | (((c_hi >> 4) & M1_64) << 1);
                    let q1 = ((q0 as u32) & 0x03030303) | (((q0 >> 30) as u32) & 0x0c0c0c0c);
                    let q3 = ((q1 & 0x0f0f) | ((q1 >> 12) & 0xf0f0)) as u16;
                    p_dst.add(8).cast::<u16>().write_unaligned(q3.to_le());
                } else if best_bits == 6 {
                    let r0 = ((c_lo >> 4) & M2_64) | (((c_hi >> 4) & M2_64) << 2);
                    let p2 = ((r0 as u32) & 0x0f0f0f0f) | (((r0 >> 28) as u32) & 0xf0f0f0f0);
                    p_dst.add(8).cast::<u32>().write_unaligned(p2.to_le());
                }
            }
        }
        if rem > 0 {
            let base = chunks16 * 16;
            let mut acc = 0u128;
            for i in 0..rem {
                let b = unsafe { *ptr.add(base + i) };
                let code = map[b as usize];
                if code == esc_code {
                    out.push(b);
                }
                acc |= (code as u128) << (i * (best_bits as usize));
            }
            let tail_bytes = (rem * (best_bits as usize)).div_ceil(8);
            let le = acc.to_le_bytes();
            unsafe {
                let p_dst = out.as_mut_ptr().add(packed_start + chunks16 * stride);
                std::ptr::copy_nonoverlapping(le.as_ptr(), p_dst, tail_bytes);
            }
        }
    }
    true
}

fn unpack_palette_stream(src: &[u8], out: &mut [u8]) -> Result<()> {
    let n = out.len();
    let mut cursor = Cursor::new(src);
    let hdr = cursor.byte()?;
    let bits = hdr & 0x7f;
    let has_esc = (hdr & 0x80) != 0;
    if !matches!(bits, 4..=6) {
        return Err(invalid("palette bits"));
    }
    let cap = 1usize << bits;
    let pal_len = cursor.byte()? as usize;
    if pal_len == 0 || (has_esc && pal_len != cap - 1) || (!has_esc && pal_len > cap) {
        return Err(invalid("palette length"));
    }
    let pal_slice = cursor.take(pal_len)?;
    let esc_len = if has_esc { cursor.uleb()? } else { 0 };
    let packed_len = (n * (bits as usize)).div_ceil(8);
    let packed = cursor.take(packed_len)?;
    let esc_bytes = cursor.take(esc_len)?;
    if !cursor.done() {
        return Err(invalid("palette trailing bytes"));
    }

    let mut pal = [0u8; 64];
    pal[..pal_len].copy_from_slice(pal_slice);

    let chunks16 = n / 16;
    let rem = n % 16;
    let mut p_src = packed.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let mut esc_pos = 0usize;

    const M4_64: u64 = 0x0f0f0f0f_0f0f0f0f;
    const M1_64: u64 = 0x01010101_01010101;
    let esc_mask64: u64 = M1_64 << bits;

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "aarch64")]
    let (tbl1, tbl2, tbl4) = unsafe {
        let v0 = vld1q_u8(pal.as_ptr());
        let v1 = vld1q_u8(pal.as_ptr().add(16));
        let v2 = vld1q_u8(pal.as_ptr().add(32));
        let v3 = vld1q_u8(pal.as_ptr().add(48));
        (v0, uint8x16x2_t(v0, v1), uint8x16x4_t(v0, v1, v2, v3))
    };

    for blk in 0..chunks16 {
        let p4 = u64::from_le(unsafe { p_src.cast::<u64>().read_unaligned() });
        let mut idx_lo = p4 & M4_64;
        let mut idx_hi = (p4 >> 4) & M4_64;
        if bits == 4 {
            p_src = unsafe { p_src.add(8) };
        } else if bits == 5 {
            let q3 = u16::from_le(unsafe { p_src.add(8).cast::<u16>().read_unaligned() }) as u32;
            p_src = unsafe { p_src.add(10) };
            let q1 = (q3 & 0x0f0f) | ((q3 & 0xf0f0) << 12);
            let q0 = ((q1 & 0x03030303) as u64) | (((q1 & 0x0c0c0c0c) as u64) << 30);
            idx_lo |= (q0 & M1_64) << 4;
            idx_hi |= (q0 & 0x02020202_02020202) << 3;
        } else {
            let p2 = u32::from_le(unsafe { p_src.add(8).cast::<u32>().read_unaligned() });
            p_src = unsafe { p_src.add(12) };
            let r0 = ((p2 & 0x0f0f0f0f) as u64) | (((p2 & 0xf0f0f0f0) as u64) << 28);
            idx_lo |= (r0 & 0x03030303_03030303) << 4;
            idx_hi |= (r0 & 0x0c0c0c0c_0c0c0c0c) << 2;
        }

        let dst = unsafe { out_ptr.add(blk * 16) };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let idx_v =
                vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(idx_lo), vcreate_u64(idx_hi)));
            let syms_v = if bits == 4 {
                vqtbl1q_u8(tbl1, idx_v)
            } else if bits == 5 {
                vqtbl2q_u8(tbl2, idx_v)
            } else {
                vqtbl4q_u8(tbl4, idx_v)
            };
            vst1q_u8(dst, syms_v);
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            let lo_b = idx_lo.to_le_bytes();
            let hi_b = idx_hi.to_le_bytes();
            for i in 0..8 {
                *dst.add(i) = pal[(lo_b[i] as usize) & 63];
                *dst.add(8 + i) = pal[(hi_b[i] as usize) & 63];
            }
        }

        if has_esc {
            let mut m_lo = (idx_lo + M1_64) & esc_mask64;
            let mut m_hi = (idx_hi + M1_64) & esc_mask64;
            if (m_lo | m_hi) != 0 {
                while m_lo != 0 {
                    let lane = (m_lo.trailing_zeros() >> 3) as usize;
                    if esc_pos >= esc_len {
                        return Err(invalid("palette escape underflow"));
                    }
                    unsafe {
                        *dst.add(lane) = *esc_bytes.get_unchecked(esc_pos);
                    }
                    esc_pos += 1;
                    m_lo &= m_lo - 1;
                }
                while m_hi != 0 {
                    let lane = 8 + ((m_hi.trailing_zeros() >> 3) as usize);
                    if esc_pos >= esc_len {
                        return Err(invalid("palette escape underflow"));
                    }
                    unsafe {
                        *dst.add(lane) = *esc_bytes.get_unchecked(esc_pos);
                    }
                    esc_pos += 1;
                    m_hi &= m_hi - 1;
                }
            }
        }
    }

    if rem > 0 {
        let tail_bytes = (rem * (bits as usize)).div_ceil(8);
        let mut le = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(p_src, le.as_mut_ptr(), tail_bytes);
        }
        let acc = u128::from_le_bytes(le);
        let mask = (1u128 << bits) - 1;
        let esc_code = pal_len as u128;
        let base = chunks16 * 16;
        for i in 0..rem {
            let code = (acc >> (i * (bits as usize))) & mask;
            if has_esc && code == esc_code {
                if esc_pos >= esc_len {
                    return Err(invalid("palette escape underflow"));
                }
                out[base + i] = esc_bytes[esc_pos];
                esc_pos += 1;
            } else if (code as usize) < pal_len {
                out[base + i] = pal[code as usize];
            } else {
                return Err(invalid("palette symbol out of range"));
            }
        }
    }
    if esc_pos != esc_len {
        return Err(invalid("palette unused escapes"));
    }
    Ok(())
}

/// Core 4-stream LZ77 + repeat-offset match finder.
/// Returns `(seq_count, off_count, lit_total_len)`.
#[allow(clippy::too_many_arguments)]
fn pack_fast_lz_core(
    data: &[u8],
    use_dict7: bool,
    dict4_vec: &mut Vec<u32>,
    dict7_vec: &mut Vec<u16>,
    tokens_vec: &mut Vec<u8>,
    offsets_vec: &mut Vec<u8>,
    ext_vec: &mut Vec<u8>,
    lits_vec: &mut Vec<u8>,
    hash_len: u32,
) -> (usize, usize, usize) {
    match (hash_len, use_dict7) {
        (4, false) => pack_fast_lz_core_const::<4, false>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (4, true) => pack_fast_lz_core_const::<4, true>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (5, false) => pack_fast_lz_core_const::<5, false>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (5, true) => pack_fast_lz_core_const::<5, true>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (_, false) => pack_fast_lz_core_const::<6, false>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
        (_, true) => pack_fast_lz_core_const::<6, true>(
            data, dict4_vec, dict7_vec, tokens_vec, offsets_vec, ext_vec, lits_vec,
        ),
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn pack_fast_lz_core_const<const HASH_LEN: u32, const USE_DICT7: bool>(
    data: &[u8],
    dict4_vec: &mut Vec<u32>,
    dict7_vec: &mut Vec<u16>,
    tokens_vec: &mut Vec<u8>,
    offsets_vec: &mut Vec<u8>,
    ext_vec: &mut Vec<u8>,
    lits_vec: &mut Vec<u8>,
) -> (usize, usize, usize) {
    let n = data.len();
    let ptr = data.as_ptr();
    let (d4_len, shift4) = if n <= 512 {
        (256usize, 24u32)
    } else if n <= 2048 {
        (1024usize, 22u32)
    } else if n <= 8192 {
        (2048usize, 21u32)
    } else if n <= 24576 {
        (4096usize, 20u32)
    } else {
        (8192usize, 19u32)
    };
    let (d7_len, shift7) = (4096usize, 52u32);
    if dict4_vec.len() < d4_len {
        dict4_vec.resize(d4_len, 0);
    } else {
        dict4_vec[..d4_len].fill(0);
    }
    if USE_DICT7 {
        if dict7_vec.len() < d7_len {
            dict7_vec.resize(d7_len, 0);
        } else {
            dict7_vec[..d7_len].fill(0);
        }
    }

    let max_seqs = n / 4 + 16;
    if tokens_vec.len() < max_seqs {
        tokens_vec.resize(max_seqs, 0);
    }
    if offsets_vec.len() < max_seqs * 2 {
        offsets_vec.resize(max_seqs * 2, 0);
    }
    if lits_vec.len() < n + 64 {
        lits_vec.resize(n + 64, 0);
    }
    ext_vec.clear();

    let dict4 = dict4_vec.as_mut_ptr();
    let dict7 = dict7_vec.as_mut_ptr();
    let tok_ptr = tokens_vec.as_mut_ptr();
    let off_ptr = offsets_vec.as_mut_ptr().cast::<u16>();
    let lit_ptr = lits_vec.as_mut_ptr();
    let mut seq_count = 0usize;
    let mut off_count = 0usize;
    let mut lit_pos = 0usize;
    let mut rep_offset = 1usize;

    const { assert!(HASH_LEN >= 4 && HASH_LEN <= 6) };
    let m4: u64 = u64::MAX >> (64 - 8 * HASH_LEN);
    let hsh: u32 = 64 - 8 * HASH_LEN;
    #[inline(always)]
    fn hash4_fn<const HL: u32>(w: u64, hsh: u32, shift4: u32) -> usize {
        if HL > 4 {
            ((w << hsh).wrapping_mul(0xCF1B_BCDC_BB55_E8A5) >> (32 + shift4)) as usize
        } else {
            ((w as u32).wrapping_mul(0x9E37_79B1) >> shift4) as usize
        }
    }
    let hash4 = |w: u64| -> usize { hash4_fn::<HASH_LEN>(w, hsh, shift4) };
    let hash7 = |w: u64| -> usize {
        (((w & 0x00ff_ffff_ffff_ffff).wrapping_mul(0xCF1B_BCDC_BB55_E8A5)) >> shift7) as usize
    };
    #[inline(always)]
    unsafe fn match_len_from(
        ptr: *const u8,
        ip: usize,
        cand: usize,
        diff: u64,
        n: usize,
    ) -> usize {
        if diff != 0 {
            (diff.trailing_zeros() >> 3) as usize
        } else {
            let mut ml = 8usize;
            while ip + ml + 8 <= n {
                let a = unsafe { ptr.add(ip + ml).cast::<u64>().read_unaligned().to_le() };
                let b = unsafe { ptr.add(cand + ml).cast::<u64>().read_unaligned().to_le() };
                let d = a ^ b;
                if d == 0 {
                    ml += 8;
                } else {
                    ml += (d.trailing_zeros() >> 3) as usize;
                    break;
                }
            }
            while ip + ml < n && unsafe { *ptr.add(ip + ml) == *ptr.add(cand + ml) } {
                ml += 1;
            }
            ml
        }
    }

    let w0 = unsafe { ptr.cast::<u64>().read_unaligned().to_le() };
    unsafe {
        *dict4.add(hash4(w0)) = 0;
        if USE_DICT7 {
            *dict7.add(hash7(w0)) = 0;
        }
    }

    let mut anchor = 0usize;
    let mut ip = 1usize;
    let limit = n.saturating_sub(12);

    while ip < limit {
        let mut match_pos = 0usize;
        let mut best_ml = 0usize;
        let mut is_rep = false;
        let mut miss = 0usize;
        let matched_so_far = anchor - lit_pos;

        loop {
            if ip >= limit {
                break;
            }
            let w = unsafe { ptr.add(ip).cast::<u64>().read_unaligned().to_le() };
            let h4 = hash4(w);
            let pair4 = unsafe { *dict4.add(h4) };
            unsafe {
                *dict4.add(h4) = (pair4 << 16) | (ip as u32);
            }
            let cand0 = (pair4 & 0xffff) as usize;
            let cw0 = unsafe { ptr.add(cand0).cast::<u64>().read_unaligned().to_le() };
            let d0 = w ^ cw0;
            let hit0 = ((d0 & m4) == 0) & (cand0 < ip);
            let hit_rle = (w ^ (w >> 8)) as u32 == 0;
            let lit_cur_lits = ip - anchor;
            let mut c7 = 0usize;

            if miss >= 6 {
                if !(hit0 | hit_rle) {
                    if ip >= 8192 && matched_so_far < 128 && lit_cur_lits >= 2048 {
                        return (0, 0, n);
                    }
                    miss += 1;
                    let shift = if ip >= 2048 && matched_so_far < 64 && lit_cur_lits >= 1024 {
                        3
                    } else if seq_count < 2 || lit_cur_lits >= 512 {
                        4
                    } else {
                        5
                    };
                    ip += 1 + (miss >> shift);
                    continue;
                }
            } else {
                let hit = {
                    let c1 = (pair4 >> 16) as usize;
                    let cw1 = unsafe { ptr.add(c1).cast::<u64>().read_unaligned().to_le() };
                    let r_cand = ip.wrapping_sub(rep_offset);
                    let rw = unsafe {
                        ptr.add(if r_cand < ip { r_cand } else { 0 })
                            .cast::<u64>()
                            .read_unaligned()
                            .to_le()
                    };
                    let mut h = hit0
                        | ((((w ^ cw1) & m4) == 0) & (c1 < ip) & (miss < 4))
                        | (((w ^ rw) as u32 == 0) & (r_cand < ip) & (lit_cur_lits <= 4))
                        | hit_rle;
                    if USE_DICT7 {
                        let h7 = hash7(w);
                        c7 = unsafe { *dict7.add(h7) as usize };
                        unsafe {
                            *dict7.add(h7) = ip as u16;
                        }
                        let cw7 = unsafe { ptr.add(c7).cast::<u64>().read_unaligned().to_le() };
                        h |= (((w ^ cw7) & 0x00ff_ffff_ffff_ffff) == 0) & (c7 < ip);
                    }
                    h
                };
                if !hit {
                    if ip >= 8192 && matched_so_far < 128 && lit_cur_lits >= 2048 {
                        return (0, 0, n);
                    }
                    miss += 1;
                    let shift = if ip >= 2048 && matched_so_far < 64 && lit_cur_lits >= 1024 {
                        3
                    } else if seq_count < 2 || lit_cur_lits >= 512 {
                        4
                    } else {
                        5
                    };
                    ip += 1 + (miss >> shift);
                    continue;
                }
            }

            if hit0 {
                match_pos = cand0;
                best_ml = unsafe { match_len_from(ptr, ip, cand0, d0, n) };
            }

            let min_ml_4 = if lit_cur_lits < 15 { 4 } else { 5 };

            if best_ml < 16 && miss < 4 {
                let cand1 = (pair4 >> 16) as usize;
                if cand1 != cand0 && ip > cand1 {
                    let cw1 = unsafe { ptr.add(cand1).cast::<u64>().read_unaligned().to_le() };
                    let d1 = w ^ cw1;
                    if (d1 & m4) == 0 {
                        let ml1 = unsafe { match_len_from(ptr, ip, cand1, d1, n) };
                        if ml1 > best_ml {
                            best_ml = ml1;
                            match_pos = cand1;
                        }
                    }
                }
            }

            if USE_DICT7 && miss < 6 && best_ml < 8 && ip > c7 && c7 != match_pos {
                let cw7 = unsafe { ptr.add(c7).cast::<u64>().read_unaligned().to_le() };
                let d7 = w ^ cw7;
                if (d7 & 0x00ff_ffff_ffff_ffff) == 0 {
                    let ml7 = unsafe { match_len_from(ptr, ip, c7, d7, n) };
                    if ml7 > best_ml {
                        best_ml = ml7;
                        match_pos = c7;
                    }
                }
            }

            if ip > anchor
                && ip >= rep_offset
                && ((best_ml == 0 && lit_cur_lits <= 4)
                    || (best_ml > 0 && match_pos != ip - rep_offset))
            {
                let r_cand = ip - rep_offset;
                let rw = unsafe { ptr.add(r_cand).cast::<u64>().read_unaligned().to_le() };
                let r_diff = w ^ rw;
                if (r_diff & 0xffff_ffff) == 0 {
                    let rml = unsafe { match_len_from(ptr, ip, r_cand, r_diff, n) };
                    if best_ml == 0 || rml + 2 >= best_ml {
                        match_pos = r_cand;
                        best_ml = rml;
                        is_rep = true;
                    }
                }
            }

            if best_ml < 8 && hit_rle && ip + 1 < limit {
                let cw_rle = unsafe { ptr.add(ip + 1).cast::<u64>().read_unaligned().to_le() };
                let rle_ml = unsafe { match_len_from(ptr, ip + 1, ip, cw_rle ^ w, n) };
                if ip > anchor && unsafe { *ptr.add(ip - 1) == (w as u8) } {
                    let full_rle = rle_ml + 1;
                    if full_rle + 2 >= best_ml {
                        match_pos = ip - 1;
                        best_ml = full_rle;
                        is_rep = rep_offset == 1;
                    }
                } else if rle_ml >= 5 && rle_ml > best_ml + 1 {
                    ip += 1;
                    match_pos = ip - 1;
                    best_ml = rle_ml;
                    is_rep = rep_offset == 1;
                    break;
                }
            }

            let min_ml = if is_rep { 4 } else { min_ml_4 };
            if best_ml >= min_ml {
                break;
            }

            if ip >= 8192 && matched_so_far < 128 && lit_cur_lits >= 2048 {
                return (0, 0, n);
            }

            best_ml = 0;
            is_rep = false;
            miss += 1;
            let shift = if ip >= 2048 && matched_so_far < 64 && lit_cur_lits >= 1024 {
                3
            } else if seq_count < 2 || lit_cur_lits >= 512 {
                4
            } else {
                5
            };
            ip += 1 + (miss >> shift);
        }
        if ip >= limit {
            break;
        }

        let next_ip = ip + best_ml;

        if !is_rep {
            let orig_ip = ip;
            while ip > anchor
                && match_pos > 0
                && unsafe { *ptr.add(ip - 1) == *ptr.add(match_pos - 1) }
            {
                ip -= 1;
                match_pos -= 1;
            }
            if ip < orig_ip {
                let bw = unsafe { ptr.add(ip).cast::<u64>().read_unaligned().to_le() };
                let bh4 = hash4(bw);
                unsafe {
                    let prev = *dict4.add(bh4);
                    *dict4.add(bh4) = (prev << 16) | (ip as u32);
                    if USE_DICT7 {
                        *dict7.add(hash7(bw)) = ip as u16;
                    }
                }
            }
        }

        let lit_len = ip - anchor;
        let offset = ip - match_pos;
        let ml = next_ip - ip;
        if offset == rep_offset {
            is_rep = true;
        } else {
            rep_offset = offset;
        }

        if ml >= 6 && next_ip < limit {
            if ml >= 8 && offset >= 8 && ip + 2 < next_ip - 2 {
                let p = ip + 2;
                let pw = unsafe { ptr.add(p).cast::<u64>().read_unaligned().to_le() };
                let ph4 = hash4(pw);
                unsafe {
                    let prev = *dict4.add(ph4);
                    *dict4.add(ph4) = (prev << 16) | (p as u32);
                    if USE_DICT7 {
                        *dict7.add(hash7(pw)) = p as u16;
                    }
                }
            }
            let tail_p = next_ip - 2;
            let tw = unsafe { ptr.add(tail_p).cast::<u64>().read_unaligned().to_le() };
            let th4 = hash4(tw);
            unsafe {
                let prev = *dict4.add(th4);
                *dict4.add(th4) = (prev << 16) | (tail_p as u32);
                if USE_DICT7 {
                    *dict7.add(hash7(tw)) = tail_p as u16;
                }
            }
        }

        unsafe {
            if lit_len <= 32 && anchor + 32 <= n {
                let lsrc = ptr.add(anchor);
                let ldst = lit_ptr.add(lit_pos);
                ldst.cast::<u128>()
                    .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                ldst.add(16)
                    .cast::<u128>()
                    .write_unaligned(lsrc.add(16).cast::<u128>().read_unaligned());
                lit_pos += lit_len;
            } else if lit_len > 0 {
                std::ptr::copy_nonoverlapping(ptr.add(anchor), lit_ptr.add(lit_pos), lit_len);
                lit_pos += lit_len;
            }
        }

        let lit_code = lit_len.min(15) as u8;
        let ml_minus_4 = ml - 4;
        let ml_code = ml_minus_4.min(7) as u8;
        let tok = lit_code | (ml_code << 4) | ((is_rep as u8) << 7);
        unsafe {
            *tok_ptr.add(seq_count) = tok;
            if !is_rep {
                off_ptr
                    .add(off_count)
                    .write_unaligned((offset as u16).to_le());
                off_count += 1;
            }
        }
        seq_count += 1;

        if lit_code == 15 {
            let rem = lit_len - 15;
            if rem < 255 {
                ext_vec.push(rem as u8);
            } else {
                ext_vec.push(255);
                ext_vec.extend_from_slice(&((rem - 255) as u16).to_le_bytes());
            }
        }
        if ml_code == 7 {
            let rem = ml_minus_4 - 7;
            if rem < 255 {
                ext_vec.push(rem as u8);
            } else {
                ext_vec.push(255);
                ext_vec.extend_from_slice(&((rem - 255) as u16).to_le_bytes());
            }
        }

        ip = next_ip;
        anchor = ip;
    }

    if seq_count > 0 && anchor < n {
        let rem_lits = n - anchor;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr.add(anchor), lit_ptr.add(lit_pos), rem_lits);
        }
        lit_pos += rem_lits;
    }
    (seq_count, off_count, lit_pos)
}

/// Emit an entropy-only `FAST_LZ_PACKET` (`seq_count == 0`): the smaller of the NEON palette
/// stream (`comp_len > 0`, palette bytes) and a Huffman blob (`comp_len == 0`, ULEB length, blob).
fn emit_entropy_only(
    data: &[u8],
    tmp: &mut Vec<u8>,
    huf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> Option<()> {
    let n = data.len();
    let mut best_kind = 0u8;
    let mut best_len = n;
    huf.clear();
    if huf_encode(data, huf, best_len) && 1 + uleb_len(huf.len()) + huf.len() < best_len {
        best_kind = 2;
        best_len = 1 + uleb_len(huf.len()) + huf.len();
    } else if n >= 256 && pack_palette_stream(data, tmp) {
        best_kind = 1;
        best_len = uleb_len(tmp.len()) + tmp.len();
    }
    if best_kind == 0 || 1 + uleb_len(n) + 1 + best_len + 16 + (n >> 6) >= n {
        return None;
    }
    out.push(FAST_LZ_PACKET);
    put_uleb(out, n);
    put_uleb(out, 0);
    if best_kind == 1 {
        put_uleb(out, tmp.len());
        out.extend_from_slice(tmp);
    } else {
        put_uleb(out, 0);
        put_uleb(out, huf.len());
        out.extend_from_slice(huf);
    }
    Some(())
}

/// Fast 4-stream separated LZ77 + repeat-offset + NEON palette block compressor (`FAST_LZ_PACKET = 3`)
/// for non-schematic text, source code, mixed blocks, Base64 blobs, and general binary data.
fn pack_fast_lz_into(data: &[u8], out: &mut Vec<u8>) -> Option<()> {
    let n = data.len();
    if !(64..=65536).contains(&n) {
        return None;
    }
    let mut is_pure_base64 = false;
    if n >= 256 {
        const B64_M0: u64 = (1 << b'\t')
            | (1 << b'\n')
            | (1 << b'\r')
            | (1 << b'+')
            | (1 << b'-')
            | (0x7ff << b'/')
            | (1 << b'=');
        const B64_M1: u64 = (0x03ff_ffff << (b'A' - 64))
            | (1 << (b'_' - 64))
            | (0x03ff_ffff << (b'a' - 64));
        let mut bmap = [0u64; 4];
        for &b in &data[..256] {
            bmap[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        let u0 = bmap[0].count_ones()
            + bmap[1].count_ones()
            + bmap[2].count_ones()
            + bmap[3].count_ones();
        if u0 >= 196 {
            return None;
        }
        if n >= 4096
            && (52..=66).contains(&u0)
            && (bmap[0] & !B64_M0) == 0
            && (bmap[1] & !B64_M1) == 0
            && (bmap[2] | bmap[3]) == 0
        {
            let mid = n / 2;
            let mut bmap_m = [0u64; 4];
            for &b in &data[mid..mid + 256] {
                bmap_m[(b >> 6) as usize] |= 1u64 << (b & 63);
            }
            for &b in &data[n - 256..] {
                bmap_m[(b >> 6) as usize] |= 1u64 << (b & 63);
            }
            let um = bmap_m[0].count_ones()
                + bmap_m[1].count_ones()
                + bmap_m[2].count_ones()
                + bmap_m[3].count_ones();
            if (52..=66).contains(&um)
                && (bmap_m[0] & !B64_M0) == 0
                && (bmap_m[1] & !B64_M1) == 0
                && (bmap_m[2] | bmap_m[3]) == 0
            {
                is_pure_base64 = true;
            }
        }
    }

    STRIDE_SCRATCH.with(|cell| {
        let sc = &mut *cell.borrow_mut();
        if is_pure_base64 {
            return emit_entropy_only(data, &mut sc.lz_lits_comp, &mut sc.lz_lits_huf, out);
        }

        // Text-like blocks (almost no control/high bytes in a 256-byte probe) use a
        // 5-byte primary hash; binary / pixel data prefers 6 bytes.
        let probe = &data[..n.min(256)];
        let binary_bytes = probe
            .iter()
            .filter(|&&b| b >= 0x80 || b < 0x09 || (b > 0x0d && b < 0x20))
            .count();
        let hash_len = if binary_bytes * 32 <= probe.len() { 5 } else { 6 };
        let (seq_count, off_count, lit_total_len) = pack_fast_lz_core(
            data,
            n >= 2048,
            &mut sc.lz_dict4,
            &mut sc.lz_dict7,
            &mut sc.lz_tokens,
            &mut sc.lz_offsets,
            &mut sc.lz_ext,
            &mut sc.lz_lits,
            hash_len,
        );

        // Entropy-only block (palette or Huffman) when LZ found no matches:
        if seq_count == 0 {
            return emit_entropy_only(data, &mut sc.lz_lits_comp, &mut sc.lz_lits_huf, out);
        }

        let offs_byte_len = off_count * 2;
        let ext_len = sc.lz_ext.len();
        let meta_raw_len = seq_count + offs_byte_len + ext_len;

        let mut flags = 0u8;
        let mut meta_size = meta_raw_len;

        // Split-stream Huffman meta (flag 8): tokens | offset lo | offset hi | ext.
        if seq_count >= 160 {
            sc.lz_meta_huf.clear();
            put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_tokens[..seq_count]);
            if sc.lz_sub_lits.len() < 2 * off_count {
                sc.lz_sub_lits.resize(2 * off_count, 0);
            }
            let (lo_dst, hi_dst) = sc.lz_sub_lits[..2 * off_count].split_at_mut(off_count);
            let mut oi = 0usize;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let src_p = sc.lz_offsets.as_ptr();
                let lo_p = lo_dst.as_mut_ptr();
                let hi_p = hi_dst.as_mut_ptr();
                while oi + 16 <= off_count {
                    let pair = vld2q_u8(src_p.add(oi * 2));
                    vst1q_u8(lo_p.add(oi), pair.0);
                    vst1q_u8(hi_p.add(oi), pair.1);
                    oi += 16;
                }
            }
            while oi < off_count {
                lo_dst[oi] = sc.lz_offsets[2 * oi];
                hi_dst[oi] = sc.lz_offsets[2 * oi + 1];
                oi += 1;
            }
            put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_sub_lits[..off_count]);
            put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_sub_lits[off_count..2 * off_count]);
            put_huf_or_raw(&mut sc.lz_meta_huf, &sc.lz_ext);
            if sc.lz_meta_huf.len() + 8 < meta_size {
                meta_size = sc.lz_meta_huf.len();
                flags |= 8;
            }
        }

        // Only try secondary LZ on meta stream (flag 1) when split-stream Huffman wasn't used
        // or didn't compress meta by at least 12.5%:
        if (flags & 8 == 0 || meta_size + (meta_raw_len >> 3) >= meta_raw_len)
            && n >= 1024
            && meta_raw_len >= 256
            && seq_count >= 48
            && meta_raw_len + lit_total_len + (n >> 5) < n
        {
            sc.lz_meta_raw.clear();
            sc.lz_meta_raw.reserve(meta_raw_len);
            sc.lz_meta_raw.extend_from_slice(&sc.lz_tokens[..seq_count]);
            sc.lz_meta_raw
                .extend_from_slice(&sc.lz_offsets[..offs_byte_len]);
            sc.lz_meta_raw.extend_from_slice(&sc.lz_ext);

            let (m_seq, m_off, m_lit) = pack_fast_lz_core(
                &sc.lz_meta_raw,
                false,
                &mut sc.lz_dict4,
                &mut sc.lz_dict7,
                &mut sc.lz_sub_tokens,
                &mut sc.lz_sub_offsets,
                &mut sc.lz_sub_ext,
                &mut sc.lz_sub_lits,
                4,
            );
            if m_seq > 0 {
                let m_offs_bytes = m_off * 2;
                let m_ext_len = sc.lz_sub_ext.len();
                let m_total = uleb_len(m_seq)
                    + uleb_len(m_lit)
                    + uleb_len(m_off)
                    + uleb_len(m_ext_len)
                    + m_seq
                    + m_offs_bytes
                    + m_ext_len
                    + m_lit;
                let m_framed = uleb_len(m_total) + m_total;
                if m_total + 16 + (meta_raw_len >> 5) < meta_raw_len && m_framed + 8 < meta_size {
                    sc.lz_meta_comp.clear();
                    sc.lz_meta_comp.reserve(m_total);
                    put_uleb(&mut sc.lz_meta_comp, m_seq);
                    put_uleb(&mut sc.lz_meta_comp, m_lit);
                    put_uleb(&mut sc.lz_meta_comp, m_off);
                    put_uleb(&mut sc.lz_meta_comp, m_ext_len);
                    sc.lz_meta_comp
                        .extend_from_slice(&sc.lz_sub_tokens[..m_seq]);
                    sc.lz_meta_comp
                        .extend_from_slice(&sc.lz_sub_offsets[..m_offs_bytes]);
                    sc.lz_meta_comp.extend_from_slice(&sc.lz_sub_ext);
                    sc.lz_meta_comp.extend_from_slice(&sc.lz_sub_lits[..m_lit]);
                    meta_size = m_framed;
                    flags = (flags & !8) | 1;
                }
            }
        }

        // Literals: try Huffman (flag 4) first, falling back to NEON palette (flag 2) only if Huffman didn't compress.
        let mut lits_size = lit_total_len;
        if lit_total_len >= HUF_MIN {
            sc.lz_lits_huf.clear();
            if huf_encode(&sc.lz_lits[..lit_total_len], &mut sc.lz_lits_huf, lits_size)
                && uleb_len(sc.lz_lits_huf.len()) + sc.lz_lits_huf.len() < lits_size
            {
                flags |= 4;
                lits_size = uleb_len(sc.lz_lits_huf.len()) + sc.lz_lits_huf.len();
            }
        }
        if (flags & 4) == 0
            && lit_total_len >= 192
            && meta_size + lit_total_len + (n >> 6) < n
            && pack_palette_stream(&sc.lz_lits[..lit_total_len], &mut sc.lz_lits_comp)
            && uleb_len(sc.lz_lits_comp.len()) + sc.lz_lits_comp.len() < lits_size
        {
            flags |= 2;
            lits_size = uleb_len(sc.lz_lits_comp.len()) + sc.lz_lits_comp.len();
        }
        let lit_hdr = if flags != 0 {
            n + 1 + ((flags as usize) << 16) + lit_total_len
        } else {
            lit_total_len
        };

        let total_size = 1
            + uleb_len(n)
            + uleb_len(seq_count)
            + uleb_len(lit_hdr)
            + uleb_len(off_count)
            + uleb_len(ext_len)
            + meta_size
            + lits_size;
        if total_size + 16 + (n >> 6) >= n {
            return emit_entropy_only(data, &mut sc.lz_lits_comp, &mut sc.lz_lits_huf, out);
        }

        out.reserve(total_size);
        out.push(FAST_LZ_PACKET);
        put_uleb(out, n);
        put_uleb(out, seq_count);
        put_uleb(out, lit_hdr);
        put_uleb(out, off_count);
        put_uleb(out, ext_len);
        if (flags & 1) != 0 {
            put_uleb(out, sc.lz_meta_comp.len());
            out.extend_from_slice(&sc.lz_meta_comp);
        } else if (flags & 8) != 0 {
            out.extend_from_slice(&sc.lz_meta_huf);
        } else {
            out.extend_from_slice(&sc.lz_tokens[..seq_count]);
            out.extend_from_slice(&sc.lz_offsets[..offs_byte_len]);
            out.extend_from_slice(&sc.lz_ext);
        }
        if (flags & 2) != 0 {
            put_uleb(out, sc.lz_lits_comp.len());
            out.extend_from_slice(&sc.lz_lits_comp);
        } else if (flags & 4) != 0 {
            put_uleb(out, sc.lz_lits_huf.len());
            out.extend_from_slice(&sc.lz_lits_huf);
        } else {
            out.extend_from_slice(&sc.lz_lits[..lit_total_len]);
        }
        Some(())
    })
}

fn unpack_fast_lz_core(
    seq_count: usize,
    off_count: usize,
    toks: &[u8],
    offs_bytes: &[u8],
    ext_bytes: &[u8],
    lits: &[u8],
    out: &mut [u8],
) -> Result<()> {
    let raw_len = out.len();
    let lit_total_len = lits.len();
    let ext_len = ext_bytes.len();
    let out_ptr = out.as_mut_ptr();
    let lit_ptr = lits.as_ptr();
    let tok_ptr = toks.as_ptr();
    let off_ptr = offs_bytes.as_ptr().cast::<u16>();
    let mut dst_pos = 0usize;
    let mut lit_pos = 0usize;
    let mut off_idx = 0usize;
    let mut ext_pos = 0usize;
    let mut offset = 1usize;

    for seq_idx in 0..seq_count {
        let tok = unsafe { *tok_ptr.add(seq_idx) };
        let mut lit_len = (tok & 0x0f) as usize;
        let ml_code = ((tok >> 4) & 0x07) as usize;
        if (tok & 0x80) == 0 {
            if off_idx >= off_count {
                return Err(invalid("fast lz offset underflow"));
            }
            offset = u16::from_le(unsafe { off_ptr.add(off_idx).read_unaligned() }) as usize;
            off_idx += 1;
        }

        if lit_len < 15
            && dst_pos + 48 <= raw_len
            && lit_pos + 16 <= lit_total_len
            && offset <= dst_pos + lit_len
            && offset > 0
        {
            if ml_code < 7 {
                let match_len = ml_code + 4;
                if offset >= match_len {
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let msrc = out_ptr.add(dst_pos - offset);
                        let mdst = out_ptr.add(dst_pos);
                        mdst.cast::<u128>()
                            .write_unaligned(msrc.cast::<u128>().read_unaligned());
                        dst_pos += match_len;
                    }
                    continue;
                } else if offset == 1 {
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let b = *out_ptr.add(dst_pos - 1);
                        out_ptr
                            .add(dst_pos)
                            .cast::<u128>()
                            .write_unaligned(u128::from_le_bytes([b; 16]));
                        dst_pos += match_len;
                    }
                    continue;
                } else if offset == 4 {
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let p = out_ptr.add(dst_pos - 4).cast::<u32>().read_unaligned() as u128;
                        let v = p | (p << 32) | (p << 64) | (p << 96);
                        out_ptr.add(dst_pos).cast::<u128>().write_unaligned(v);
                        dst_pos += match_len;
                    }
                    continue;
                } else if offset == 8 {
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let p = out_ptr.add(dst_pos - 8).cast::<u64>().read_unaligned() as u128;
                        let v = p | (p << 64);
                        out_ptr.add(dst_pos).cast::<u128>().write_unaligned(v);
                        dst_pos += match_len;
                    }
                    continue;
                }
            } else if ext_pos < ext_len {
                let e = unsafe { *ext_bytes.get_unchecked(ext_pos) } as usize;
                let match_len = 11 + e;
                if match_len <= 32 && offset >= match_len {
                    ext_pos += 1;
                    unsafe {
                        let lsrc = lit_ptr.add(lit_pos);
                        let ldst = out_ptr.add(dst_pos);
                        ldst.cast::<u128>()
                            .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                        lit_pos += lit_len;
                        dst_pos += lit_len;

                        let msrc = out_ptr.add(dst_pos - offset);
                        let mdst = out_ptr.add(dst_pos);
                        mdst.cast::<u128>()
                            .write_unaligned(msrc.cast::<u128>().read_unaligned());
                        mdst.add(16)
                            .cast::<u128>()
                            .write_unaligned(msrc.add(16).cast::<u128>().read_unaligned());
                        dst_pos += match_len;
                    }
                    continue;
                }
            }
        }

        if offset == 0 {
            return Err(invalid("fast lz zero offset"));
        }
        if lit_len == 15 {
            if ext_pos >= ext_len {
                return Err(invalid("fast lz ext underflow"));
            }
            let e = ext_bytes[ext_pos] as usize;
            ext_pos += 1;
            if e < 255 {
                lit_len += e;
            } else {
                if ext_pos + 2 > ext_len {
                    return Err(invalid("fast lz ext underflow"));
                }
                let extra =
                    u16::from_le_bytes([ext_bytes[ext_pos], ext_bytes[ext_pos + 1]]) as usize;
                ext_pos += 2;
                lit_len += 255 + extra;
            }
        }
        let mut match_len = ml_code + 4;
        if ml_code == 7 {
            if ext_pos >= ext_len {
                return Err(invalid("fast lz ext underflow"));
            }
            let e = ext_bytes[ext_pos] as usize;
            ext_pos += 1;
            if e < 255 {
                match_len += e;
            } else {
                if ext_pos + 2 > ext_len {
                    return Err(invalid("fast lz ext underflow"));
                }
                let extra =
                    u16::from_le_bytes([ext_bytes[ext_pos], ext_bytes[ext_pos + 1]]) as usize;
                ext_pos += 2;
                match_len += 255 + extra;
            }
        }
        if dst_pos + lit_len + match_len + 32 <= raw_len
            && lit_pos + lit_len + 16 <= lit_total_len
            && offset <= dst_pos + lit_len
        {
            unsafe {
                let lsrc = lit_ptr.add(lit_pos);
                let ldst = out_ptr.add(dst_pos);
                ldst.cast::<u128>()
                    .write_unaligned(lsrc.cast::<u128>().read_unaligned());
                if lit_len > 16 {
                    // Slack verified above: dst has >= 32 spare bytes, lits >= 16.
                    let mut i = 16usize;
                    while i < lit_len {
                        ldst.add(i)
                            .cast::<u128>()
                            .write_unaligned(lsrc.add(i).cast::<u128>().read_unaligned());
                        i += 16;
                    }
                }
                lit_pos += lit_len;
                dst_pos += lit_len;

                let msrc = out_ptr.add(dst_pos - offset);
                let mdst = out_ptr.add(dst_pos);
                if offset >= 16 || match_len <= offset {
                    mdst.cast::<u128>()
                        .write_unaligned(msrc.cast::<u128>().read_unaligned());
                    if match_len > 16 {
                        mdst.add(16)
                            .cast::<u128>()
                            .write_unaligned(msrc.add(16).cast::<u128>().read_unaligned());
                        if match_len > 32 {
                            let mut i = 32usize;
                            while i < match_len {
                                mdst.add(i)
                                    .cast::<u128>()
                                    .write_unaligned(msrc.add(i).cast::<u128>().read_unaligned());
                                i += 16;
                            }
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 1 {
                    let b = *msrc;
                    let v = u128::from_le_bytes([b; 16]);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 4 {
                    let p = msrc.cast::<u32>().read_unaligned() as u128;
                    let v = p | (p << 32) | (p << 64) | (p << 96);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 8 {
                    let p = msrc.cast::<u64>().read_unaligned() as u128;
                    let v = p | (p << 64);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                } else if offset == 2 {
                    let p = msrc.cast::<u16>().read_unaligned() as u128;
                    let p32 = p | (p << 16);
                    let v = p32 | (p32 << 32) | (p32 << 64) | (p32 << 96);
                    mdst.cast::<u128>().write_unaligned(v);
                    if match_len > 16 {
                        let mut i = 16usize;
                        while i < match_len {
                            mdst.add(i).cast::<u128>().write_unaligned(v);
                            i += 16;
                        }
                    }
                    dst_pos += match_len;
                    continue;
                }
            }
            // Remaining overlap case (offset < 16, offset < match_len): periodic 16-byte copies.
            // Each store's first `s` bytes are valid; trailing bytes are rewritten by the next
            // store or fall inside the verified 32-byte slack.
            unsafe {
                let mdst = out_ptr.add(dst_pos);
                let step = (16 / offset) * offset;
                let mut c = 0usize;
                while c < match_len {
                    let s = if c + offset >= step { step } else { offset };
                    let v = mdst.add(c).sub(s).cast::<u128>().read_unaligned();
                    mdst.add(c).cast::<u128>().write_unaligned(v);
                    c += s;
                }
            }
            dst_pos += match_len;
            continue;
        }

        if lit_pos + lit_len > lit_total_len || dst_pos + lit_len + match_len > raw_len {
            return Err(invalid("fast lz bounds"));
        }
        if lit_len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), lit_len);
            }
            lit_pos += lit_len;
            dst_pos += lit_len;
        }
        if offset > dst_pos {
            return Err(invalid("fast lz match before start"));
        }
        if offset >= match_len {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_ptr.add(dst_pos - offset),
                    out_ptr.add(dst_pos),
                    match_len,
                );
            }
        } else if offset == 1 {
            let b = unsafe { *out_ptr.add(dst_pos - 1) };
            unsafe {
                std::ptr::write_bytes(out_ptr.add(dst_pos), b, match_len);
            }
        } else {
            let mut copied = 0usize;
            while copied < match_len {
                let avail = offset + copied;
                let take = avail.min(match_len - copied);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out_ptr.add(dst_pos - offset),
                        out_ptr.add(dst_pos + copied),
                        take,
                    );
                }
                copied += take;
            }
        }
        dst_pos += match_len;
    }

    let rem_lits = lit_total_len - lit_pos;
    if off_idx != off_count || ext_pos != ext_len || dst_pos + rem_lits != raw_len {
        return Err(invalid("fast lz final consumption"));
    }
    if rem_lits > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(lit_ptr.add(lit_pos), out_ptr.add(dst_pos), rem_lits);
        }
    }
    Ok(())
}

fn unpack_fast_lz_into(cursor: &mut Cursor<'_>, out: &mut [u8]) -> Result<()> {
    let raw_len = cursor.uleb()?;
    if raw_len == 0 || raw_len > 65536 || raw_len > MAX_BLOCK_SIZE || raw_len != out.len() {
        return Err(invalid("fast lz raw length"));
    }
    let seq_count = cursor.uleb()?;
    if seq_count == 0 {
        let comp_len = cursor.uleb()?;
        if comp_len == 0 {
            let huf_len = cursor.uleb()?;
            let huf = cursor.take(huf_len)?;
            if !cursor.done() {
                return Err(invalid("fast lz trailing bytes"));
            }
            return huf_decode(huf, out);
        }
        let comp = cursor.take(comp_len)?;
        if !cursor.done() {
            return Err(invalid("fast lz trailing bytes"));
        }
        return unpack_palette_stream(comp, out);
    }

    let lit_hdr = cursor.uleb()?;
    let (flags, lit_total_len) = if lit_hdr <= raw_len {
        (0usize, lit_hdr)
    } else {
        let rem = lit_hdr - (raw_len + 1);
        (rem >> 16, rem & 0xffff)
    };
    let off_count = cursor.uleb()?;
    let ext_len = cursor.uleb()?;
    if flags > 15
        || (flags & 9) == 9
        || (flags & 6) == 6
        || seq_count > raw_len / 4 + 16
        || off_count > seq_count
        || lit_total_len > raw_len
        || ext_len > seq_count * 6
    {
        return Err(invalid("fast lz header counts"));
    }

    if flags == 0 {
        let toks = cursor.take(seq_count)?;
        let offs_bytes = cursor.take(off_count * 2)?;
        let ext_bytes = cursor.take(ext_len)?;
        let lits = cursor.take(lit_total_len)?;
        if !cursor.done() {
            return Err(invalid("fast lz trailing bytes"));
        }
        return unpack_fast_lz_core(
            seq_count,
            off_count,
            toks,
            offs_bytes,
            ext_bytes,
            lits,
            out,
        );
    }

    STRIDE_SCRATCH.with(|cell| {
        let sc = &mut *cell.borrow_mut();
        let meta_raw_len = seq_count + off_count * 2 + ext_len;
        let (toks_slice, offs_slice, ext_slice) = if (flags & 1) != 0 {
            let meta_comp_len = cursor.uleb()?;
            let meta_comp = cursor.take(meta_comp_len)?;
            let mut mc = Cursor::new(meta_comp);
            let m_seq = mc.uleb()?;
            let m_lit = mc.uleb()?;
            let m_off = mc.uleb()?;
            let m_ext = mc.uleb()?;
            if m_seq == 0
                || m_seq > meta_raw_len / 4 + 16
                || m_off > m_seq
                || m_lit > meta_raw_len
                || m_ext > m_seq * 6
            {
                return Err(invalid("fast lz meta header"));
            }
            let m_toks = mc.take(m_seq)?;
            let m_offs = mc.take(m_off * 2)?;
            let m_exts = mc.take(m_ext)?;
            let m_lits = mc.take(m_lit)?;
            if !mc.done() {
                return Err(invalid("fast lz meta trailing bytes"));
            }
            if sc.lz_meta_raw.len() < meta_raw_len + 64 {
                sc.lz_meta_raw.resize(meta_raw_len + 64, 0);
            }
            unpack_fast_lz_core(
                m_seq,
                m_off,
                m_toks,
                m_offs,
                m_exts,
                m_lits,
                &mut sc.lz_meta_raw[..meta_raw_len],
            )?;
            let m_raw = &sc.lz_meta_raw[..meta_raw_len];
            (
                &m_raw[..seq_count],
                &m_raw[seq_count..seq_count + off_count * 2],
                &m_raw[seq_count + off_count * 2..],
            )
        } else if (flags & 8) != 0 {
            if sc.lz_meta_raw.len() < meta_raw_len + 64 {
                sc.lz_meta_raw.resize(meta_raw_len + 64, 0);
            }
            if sc.lz_sub_lits.len() < 2 * off_count {
                sc.lz_sub_lits.resize(2 * off_count, 0);
            }
            take_huf_or_raw(cursor, &mut sc.lz_meta_raw[..seq_count])?;
            let (lo, hi) = sc.lz_sub_lits[..2 * off_count].split_at_mut(off_count);
            take_huf_or_raw(cursor, lo)?;
            take_huf_or_raw(cursor, hi)?;
            let mut oi = 0usize;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                let lo_p = lo.as_ptr();
                let hi_p = hi.as_ptr();
                let dst_p = sc.lz_meta_raw.as_mut_ptr().add(seq_count);
                while oi + 16 <= off_count {
                    let v_lo = vld1q_u8(lo_p.add(oi));
                    let v_hi = vld1q_u8(hi_p.add(oi));
                    vst2q_u8(dst_p.add(oi * 2), uint8x16x2_t(v_lo, v_hi));
                    oi += 16;
                }
            }
            while oi < off_count {
                sc.lz_meta_raw[seq_count + 2 * oi] = lo[oi];
                sc.lz_meta_raw[seq_count + 2 * oi + 1] = hi[oi];
                oi += 1;
            }
            take_huf_or_raw(
                cursor,
                &mut sc.lz_meta_raw[seq_count + 2 * off_count..meta_raw_len],
            )?;
            let m_raw = &sc.lz_meta_raw[..meta_raw_len];
            (
                &m_raw[..seq_count],
                &m_raw[seq_count..seq_count + off_count * 2],
                &m_raw[seq_count + off_count * 2..],
            )
        } else {
            let t = cursor.take(seq_count)?;
            let o = cursor.take(off_count * 2)?;
            let e = cursor.take(ext_len)?;
            (t, o, e)
        };

        let lits_slice = if (flags & 2) != 0 {
            let lits_comp_len = cursor.uleb()?;
            let lits_comp = cursor.take(lits_comp_len)?;
            if sc.lz_lits.len() < lit_total_len + 64 {
                sc.lz_lits.resize(lit_total_len + 64, 0);
            }
            unpack_palette_stream(lits_comp, &mut sc.lz_lits[..lit_total_len])?;
            &sc.lz_lits[..lit_total_len]
        } else if (flags & 4) != 0 {
            let huf_len = cursor.uleb()?;
            let huf = cursor.take(huf_len)?;
            if sc.lz_lits.len() < lit_total_len + 64 {
                sc.lz_lits.resize(lit_total_len + 64, 0);
            }
            huf_decode(huf, &mut sc.lz_lits[..lit_total_len])?;
            &sc.lz_lits[..lit_total_len]
        } else {
            cursor.take(lit_total_len)?
        };
        if !cursor.done() {
            return Err(invalid("fast lz trailing bytes"));
        }

        unpack_fast_lz_core(
            seq_count,
            off_count,
            toks_slice,
            offs_slice,
            ext_slice,
            lits_slice,
            out,
        )
    })
}

fn unpack_block(bytes: &[u8]) -> Result<Vec<u8>> {
    unpack_block_mode(bytes, true)
}

fn unpack_block_mode(bytes: &[u8], strict: bool) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(bytes);
    let selector = cursor.byte()?;
    match selector {
        RAW_PACKET => {
            let len = cursor.uleb()?;
            if len > MAX_BLOCK_SIZE {
                return Err(invalid("raw column block length"));
            }
            let out = cursor.take(len)?.to_vec();
            if !cursor.done() {
                return Err(invalid("raw column block trailing bytes"));
            }
            Ok(out)
        }
        GRAMMAR_PACKET => unpack_grammar(&mut cursor, strict),
        STRIDE_PACKET => {
            let mut probe = Cursor::new(cursor.bytes);
            probe.pos = cursor.pos;
            let raw_len = probe.uleb()?;
            if raw_len > MAX_BLOCK_SIZE {
                return Err(invalid("stride raw length"));
            }
            let mut out = vec![0u8; raw_len];
            unpack_stride_into(&mut cursor, &mut out)?;
            Ok(out)
        }
        FAST_LZ_PACKET => {
            let mut probe = Cursor::new(cursor.bytes);
            probe.pos = cursor.pos;
            let raw_len = probe.uleb()?;
            if raw_len > MAX_BLOCK_SIZE {
                return Err(invalid("fast lz raw length"));
            }
            let mut out = vec![0u8; raw_len];
            unpack_fast_lz_into(&mut cursor, &mut out)?;
            Ok(out)
        }
        _ => Err(invalid("column packet selector")),
    }
}

/// Encode one independent block as a canonical structured packet.
///
/// Returns `None` when the grammar packet is not strictly smaller than the raw
/// packet; the caller then keeps its LZ/RAW payload. Every selector, dictionary,
/// table, escape and padding bit is already charged inside the returned packet.
pub fn encode_grammar_block(data: &[u8]) -> Result<Option<Vec<u8>>> {
    if data.is_empty() {
        return Ok(None);
    }
    if data.len() > MAX_BLOCK_SIZE {
        return Err(Error::InvalidOption("column block exceeds maximum"));
    }
    let raw_len = 1 + uleb_len(data.len()) + data.len();
    let mut out = Vec::with_capacity(4096);
    if pack_grammar_into(data, &mut out)? && out.len() < raw_len {
        return Ok(Some(out));
    }
    out.clear();
    if pack_stride_into(data, &mut out).is_some() && out.len() < raw_len {
        if out.len() * 4 > data.len() {
            let mut lz = Vec::with_capacity(out.len());
            if pack_fast_lz_into(data, &mut lz).is_some() && lz.len() < out.len() {
                return Ok(Some(lz));
            }
        }
        return Ok(Some(out));
    }
    out.clear();
    if pack_fast_lz_into(data, &mut out).is_some() && out.len() < raw_len {
        return Ok(Some(out));
    }
    Ok(None)
}

/// Encode one independent block directly into `out` if it compresses smaller than raw.
pub(crate) fn encode_grammar_block_into(
    data: &[u8],
    out: &mut Vec<u8>,
) -> Result<Option<(usize, usize)>> {
    if data.is_empty() {
        return Ok(None);
    }
    if data.len() > MAX_BLOCK_SIZE {
        return Err(Error::InvalidOption("column block exceeds maximum"));
    }
    let raw_len = 1 + uleb_len(data.len()) + data.len();
    let start = out.len();
    match pack_grammar_into(data, out) {
        Ok(true) => {
            let len = out.len() - start;
            if len < raw_len && (len * 4 < data.len() * 3 || data.len() < 512) {
                return Ok(Some((start, len)));
            }
            out.truncate(start);
        }
        Ok(false) => {
            out.truncate(start);
        }
        Err(err) => {
            out.truncate(start);
            return Err(err);
        }
    }
    if pack_stride_into(data, out).is_some() {
        let s_len = out.len() - start;
        if s_len < raw_len {
            if s_len * 4 > data.len() {
                let mut lz = Vec::with_capacity(s_len);
                if pack_fast_lz_into(data, &mut lz).is_some() && lz.len() < s_len {
                    out.truncate(start);
                    out.extend_from_slice(&lz);
                    return Ok(Some((start, lz.len())));
                }
            }
            return Ok(Some((start, s_len)));
        }
        out.truncate(start);
    }
    if pack_fast_lz_into(data, out).is_some() {
        let lz_len = out.len() - start;
        if lz_len < raw_len {
            return Ok(Some((start, lz_len)));
        }
        out.truncate(start);
    }
    Ok(None)
}

/// Decode and structurally validate one structured packet back to its exact
/// bytes. This is the production path: every structural rule, bound, count and
/// ordering is checked, but representation minimality (full re-encode equality)
/// is not recomputed; frame corruption is covered by the optional CRC32C.
pub fn decode_block(bytes: &[u8]) -> Result<Vec<u8>> {
    unpack_block_mode(bytes, false)
}

/// Decode one structured packet directly into the caller's exact-size buffer.
///
/// This is the production path used by the frame decoder: the row assembly
/// writes straight into `out`, so a structured block costs no intermediate
/// allocation and no second copy. `out` must be exactly the block raw size and
/// the same structural validation as [`decode_block`] applies.
pub fn decode_block_into(bytes: &[u8], out: &mut [u8]) -> Result<()> {
    let mut cursor = Cursor::new(bytes);
    let selector = cursor.byte()?;
    match selector {
        RAW_PACKET => {
            let len = cursor.uleb()?;
            if len > MAX_BLOCK_SIZE || len != out.len() {
                return Err(invalid("raw column block length"));
            }
            let data = cursor.take(len)?;
            out.copy_from_slice(data);
            if !cursor.done() {
                return Err(invalid("raw column block trailing bytes"));
            }
            Ok(())
        }
        GRAMMAR_PACKET => {
            let sink = OutSink::new(out.as_mut_ptr(), out.len());
            let pos = unpack_grammar_inner(&mut cursor, false, sink)?;
            if pos != out.len() {
                return Err(invalid("column output length"));
            }
            Ok(())
        }
        STRIDE_PACKET => unpack_stride_into(&mut cursor, out),
        FAST_LZ_PACKET => unpack_fast_lz_into(&mut cursor, out),
        _ => Err(invalid("column packet selector")),
    }
}

/// Strict oracle decode: additionally re-encodes every model and rejects a
/// payload that is not the exact charged minimum. Used by the research
/// evaluator and the canonicality tests.
pub fn decode_block_strict(bytes: &[u8]) -> Result<Vec<u8>> {
    unpack_block(bytes)
}

fn pack_block(data: &[u8]) -> Result<PackedBlock> {
    if data.len() > MAX_BLOCK_SIZE {
        return Err(Error::InvalidOption("column block exceeds maximum"));
    }
    let raw_len = 1 + uleb_len(data.len()) + data.len();
    let grammar = pack_grammar(data)?;
    if grammar.bytes.len() < raw_len {
        Ok(PackedBlock {
            bytes: grammar.bytes,
            stats: grammar.stats,
        })
    } else {
        Ok(PackedBlock {
            bytes: raw_packet(data),
            stats: BlockStats::default(),
        })
    }
}

fn check_block_size(block_size: usize) -> Result<()> {
    if !(4096..=MAX_BLOCK_SIZE).contains(&block_size) || block_size % 64 != 0 {
        return Err(Error::InvalidOption(
            "column block size must be 4096..1048576 and a multiple of 64",
        ));
    }
    Ok(())
}

/// Build each independent packet once and prove its exact reconstruction before
/// returning its complete directory-plus-payload accounting.
pub fn evaluate(input: &[u8], block_size: usize) -> Result<Evaluation> {
    check_block_size(block_size)?;
    let mut evaluation = Evaluation {
        input_bytes: input.len(),
        block_size,
        blocks: 0,
        grammar_blocks: 0,
        raw_blocks: 0,
        payload_bytes: 0,
        frame_directory_bytes: 32,
        template_count: 0,
        template_rows: 0,
        raw_rows: 0,
        model_counts: [0; 13],
        model_bytes: [0; 13],
        columns: Vec::new(),
        pack_ns: None,
        unpack_ns: None,
    };
    for (block, raw) in input.chunks(block_size).enumerate() {
        let packed = pack_block(raw)?;
        if unpack_block(&packed.bytes)? != raw {
            return Err(invalid("column evaluator roundtrip"));
        }
        evaluation.blocks += 1;
        evaluation.frame_directory_bytes = evaluation
            .frame_directory_bytes
            .checked_add(32)
            .ok_or(Error::SizeOverflow)?;
        evaluation.payload_bytes = evaluation
            .payload_bytes
            .checked_add(packed.bytes.len())
            .ok_or(Error::SizeOverflow)?;
        if packed.stats.grammar {
            evaluation.grammar_blocks += 1;
        } else {
            evaluation.raw_blocks += 1;
        }
        evaluation.template_count += packed.stats.templates;
        evaluation.template_rows += packed.stats.template_rows;
        evaluation.raw_rows += packed.stats.raw_rows;
        for column in packed.stats.columns {
            evaluation.model_counts[column.kind.index()] += 1;
            evaluation.model_bytes[column.kind.index()] = evaluation.model_bytes[column.kind.index()]
                .checked_add(column.payload_bytes)
                .ok_or(Error::SizeOverflow)?;
            evaluation.columns.push(ColumnReport {
                block,
                template: column.template,
                slot: column.slot,
                model: column.kind.name(),
                values: column.values,
                payload_bytes: column.payload_bytes,
            });
        }
    }
    Ok(evaluation)
}

/// Time complete packet construction and exact packet reconstruction separately.
/// Correctness is checked outside the timed loops so comparison scans are not
/// accidentally attributed to either scalar oracle rate.
pub fn time(input: &[u8], block_size: usize, iterations: usize) -> Result<Timing> {
    check_block_size(block_size)?;
    if iterations == 0 {
        return Err(Error::InvalidOption("column timing iterations must be positive"));
    }
    let chunks = input.chunks(block_size).collect::<Vec<_>>();
    let packets = chunks
        .iter()
        .map(|chunk| pack_block(chunk))
        .collect::<Result<Vec<_>>>()?;
    for (packet, raw) in packets.iter().zip(&chunks) {
        if unpack_block(&packet.bytes)? != *raw {
            return Err(invalid("timed column evaluator roundtrip"));
        }
    }
    let pack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        for raw in &chunks {
            checksum ^= pack_block(raw)?.bytes.len();
        }
    }
    black_box(checksum);
    let pack_ns = pack_start.elapsed().as_nanos();
    let unpack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        for packet in &packets {
            checksum ^= unpack_block(&packet.bytes)?.len();
        }
    }
    black_box(checksum);
    Ok(Timing {
        pack_ns,
        unpack_ns: unpack_start.elapsed().as_nanos(),
    })
}

impl Evaluation {
    pub fn apply_timing(&mut self, timing: &Timing) {
        self.pack_ns = Some(timing.pack_ns);
        self.unpack_ns = Some(timing.unpack_ns);
    }

    pub fn to_json(&self, iterations: usize) -> String {
        let total = self
            .frame_directory_bytes
            .checked_add(self.payload_bytes)
            .unwrap();
        let pack_rate = self
            .pack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.input_bytes as f64 * 1_000_000_000.0 / ns as f64);
        let unpack_rate = self
            .unpack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.input_bytes as f64 * 1_000_000_000.0 / ns as f64);
        let mut out = String::new();
        writeln!(out, "{{").unwrap();
        writeln!(out, "  \"schema\": \"vxg-p5d-structured-columns-r1\",").unwrap();
        writeln!(out, "  \"scope\": \"Complete independent block-local scalar research packet. It serializes templates, row selectors, model metadata and all values, proves exact decode, and charges a 32-byte header plus 32-byte directory entries. It is not a VXG v3 wire format or a codec GB/s benchmark.\",").unwrap();
        writeln!(out, "  \"selection_rule\": \"For each raw block, build RAW and grammar packets and select grammar only when it is strictly shorter. For every grammar column, choose the shortest of raw, canonical byte dictionary, canonical decimal affine, canonical decimal delta, quoted fixed-width lowercase-hex suffix, quoted hex epoch-counter, quoted semicolon segment dictionary, quoted same-year ISO-8601 and the retained periodic/GF(2) models; ties retain raw/earlier model order. No model reads a field name, source path, seed or external state.\",").unwrap();
        writeln!(out, "  \"input_bytes\": {},", self.input_bytes).unwrap();
        writeln!(out, "  \"block_size\": {},", self.block_size).unwrap();
        writeln!(out, "  \"blocks\": {},", self.blocks).unwrap();
        writeln!(out, "  \"grammar_blocks\": {},", self.grammar_blocks).unwrap();
        writeln!(out, "  \"raw_blocks\": {},", self.raw_blocks).unwrap();
        writeln!(out, "  \"payload_bytes\": {},", self.payload_bytes).unwrap();
        writeln!(out, "  \"frame_directory_bytes\": {},", self.frame_directory_bytes).unwrap();
        writeln!(out, "  \"prototype_total_frame_bytes\": {},", total).unwrap();
        writeln!(out, "  \"prototype_total_ratio\": {:.9},", total as f64 / self.input_bytes.max(1) as f64).unwrap();
        writeln!(out, "  \"template_count\": {},", self.template_count).unwrap();
        writeln!(out, "  \"template_rows\": {},", self.template_rows).unwrap();
        writeln!(out, "  \"raw_rows\": {},", self.raw_rows).unwrap();
        writeln!(out, "  \"timing_iterations\": {},", iterations).unwrap();
        writeln!(out, "  \"pack_elapsed_ns\": {},", self.pack_ns.map_or("null".to_owned(), |ns| ns.to_string())).unwrap();
        writeln!(out, "  \"unpack_elapsed_ns\": {},", self.unpack_ns.map_or("null".to_owned(), |ns| ns.to_string())).unwrap();
        writeln!(out, "  \"pack_input_bytes_per_second\": {},", pack_rate.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  \"unpack_input_bytes_per_second\": {},", unpack_rate.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  \"model_totals\": [").unwrap();
        let model_kinds = [
            ModelKind::Raw,
            ModelKind::Dictionary,
            ModelKind::DecimalAffine,
            ModelKind::HexSuffix,
            ModelKind::Timestamp,
            ModelKind::DecimalPacked,
            ModelKind::DecimalSuffix,
            ModelKind::PeriodicString,
            ModelKind::PeriodicHex,
            ModelKind::Gf2Affine,
            ModelKind::EpochCounter,
            ModelKind::SegmentDict,
            ModelKind::DecimalDelta,
        ];
        for (index, kind) in model_kinds.into_iter().enumerate() {
            writeln!(out, "    {{\"model\":\"{}\",\"columns\":{},\"payload_bytes\":{}}}{}", kind.name(), self.model_counts[kind.index()], self.model_bytes[kind.index()], if index + 1 == model_kinds.len() { "" } else { "," }).unwrap();
        }
        writeln!(out, "  ],").unwrap();
        writeln!(out, "  \"column_decomposition\": [").unwrap();
        for (index, column) in self.columns.iter().enumerate() {
            writeln!(out, "    {{\"block\":{},\"template\":{},\"slot\":{},\"model\":\"{}\",\"values\":{},\"payload_bytes\":{}}}{}", column.block, column.template, column.slot, column.model, column.values, column.payload_bytes, if index + 1 == self.columns.len() { "" } else { "," }).unwrap();
        }
        writeln!(out, "  ]").unwrap();
        writeln!(out, "}}").unwrap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let packed = pack_block(data).unwrap();
        assert_eq!(unpack_block(&packed.bytes).unwrap(), data);
    }

    #[test]
    fn exact_structured_records_select_and_roundtrip_models() {
        let data = (0..256)
            .flat_map(|index| {
                format!(
                    "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"kind\":\"{}\",\"request\":\"r-{:016x}\",\"id\":{index},\"tags\":[\"hot\",\"retry\"]}}\n",
                    index % 28 + 1,
                    index % 24,
                    if index & 1 == 0 { "open" } else { "close" },
                    index as u64 * 0x9e3779b97f4a7c15,
                )
            }
            .into_bytes())
            .collect::<Vec<_>>();
        roundtrip(&data);
        let grammar = pack_grammar(&data).unwrap();
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::Timestamp));
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::DecimalAffine));
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::HexSuffix));
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::Dictionary));
    }

    #[test]
    fn escaped_nested_and_partial_records_remain_exact() {
        let data = b"{\"a\":\"tab\\tquote\\\"slash\\\\line\\n\",\"list\":[1,{\"k\":\"v\"}]}\n{\"a\":7";
        roundtrip(data);
    }

    #[test]
    fn malformed_packets_and_noncanonical_model_are_rejected() {
        let data = b"{\"id\":1,\"kind\":\"x\"}\n{\"id\":2,\"kind\":\"x\"}\n";
        let grammar = pack_grammar(data).unwrap();
        assert!(unpack_block(&grammar.bytes[..grammar.bytes.len() - 1]).is_err());
        let mut raw = raw_packet(data);
        raw.extend_from_slice(b"x");
        assert!(unpack_block(&raw).is_err());
        let raw_model = pack_raw_model(&[
            b"repeated-canonical-dictionary-value".as_slice(),
            b"repeated-canonical-dictionary-value".as_slice(),
        ]);
        assert!(unpack_model(&raw_model, 2).is_err());
    }

    #[test]
    fn periodic_plain_strings_are_packed_and_canonical() {
        let values = [
            b"\"abcabcabcabc\"".to_vec(),
            b"\"xyzxyzxyzxyzxyz\"".to_vec(),
            b"\"012012012012\"".to_vec(),
        ];
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::PeriodicString);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut malformed = packed;
        malformed[1] = 0;
        assert!(unpack_model(&malformed, 3).is_err());

        let hex_values = [
            b"\"0123456789abcde0123456789abcde\"".to_vec(),
            b"\"fedcba987654321fedcba987654321\"".to_vec(),
        ];
        let (packed, kind) = pack_model_values(&hex_values);
        assert_eq!(kind, ModelKind::PeriodicHex);
        assert_eq!(unpack_model(&packed, hex_values.len()).unwrap(), hex_values);
    }

    #[test]
    fn production_decode_accepts_nonminimal_but_valid_models() {
        let values = [
            b"repeated-canonical-dictionary-value".to_vec(),
            b"repeated-canonical-dictionary-value".to_vec(),
        ];
        let refs = values.iter().map(|value| value.as_slice()).collect::<Vec<_>>();
        let raw_model = pack_raw_model(&refs);
        // Strict oracle: rejects because a smaller canonical model exists.
        assert!(unpack_model(&raw_model, 2).is_err());
        // Production decode: structural checks pass; exact bytes are recovered.
        let decoded = unpack_model_mode(&raw_model, 2, false).unwrap().materialize();
        assert_eq!(decoded, values);
    }

    #[test]
    fn learned_gf2_affine_recurrence_roundtrips_a_held_out_suffix() {
        fn xorshift_word(mut value: u64) -> u64 {
            value ^= value << 13;
            value ^= value >> 7;
            value ^ (value << 17)
        }
        fn quoted_word(word: u64) -> Vec<u8> {
            let mut value = b"\"r-".to_vec();
            for byte in word.to_le_bytes() {
                value.push(HEX[(byte >> 4) as usize]);
                value.push(HEX[(byte & 15) as usize]);
            }
            value.push(b'"');
            value
        }
        let mut coefficients = [0u64; GF2_COEFFICIENTS];
        for (bit, coefficient) in coefficients[..64].iter_mut().enumerate() {
            *coefficient = xorshift_word(1u64 << bit);
        }
        coefficients[64] = 0x7265_6375_7272_656e;
        let words = std::iter::successors(Some(0x1234_5678_9abc_def0u64), |word| {
            Some(apply_affine(&coefficients, *word))
        })
        .take(128)
        .collect::<Vec<_>>();
        let values = words.into_iter().map(quoted_word).collect::<Vec<_>>();
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::Gf2Affine);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
    }

    #[test]
    fn evaluation_charges_independent_directory_and_times() {
        let data = (0..1024)
            .flat_map(|index| format!("{{\"id\":{index},\"kind\":\"event\"}}\n").into_bytes())
            .collect::<Vec<_>>();
        let mut evaluation = evaluate(&data, 4096).unwrap();
        assert_eq!(evaluation.input_bytes, data.len());
        assert_eq!(evaluation.frame_directory_bytes, 32 + evaluation.blocks * 32);
        evaluation.apply_timing(&time(&data, 4096, 1).unwrap());
        assert!(evaluation.pack_ns.is_some());
        assert!(evaluation.unpack_ns.is_some());
    }

    #[test]
    fn epoch_counter_ids_are_packed_and_canonical() {
        let mut values = Vec::new();
        for row in 0..300u32 {
            let epoch = row / 128;
            let prefix = 0xf2f1_3f80u32.wrapping_add(epoch.wrapping_mul(0x9e37_79b9));
            let suffix = if row % 97 == 3 {
                row.wrapping_mul(0x2545_f491)
            } else {
                row % 128
            };
            values.push(format!("\"r-{prefix:08x}{suffix:08x}\"").into_bytes());
        }
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::EpochCounter);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut malformed = packed.clone();
        let first = malformed[5..9].to_vec();
        malformed[9..13].copy_from_slice(&first);
        assert!(unpack_model(&malformed, values.len()).is_err());
    }

    #[test]
    fn segment_dictionary_messages_are_packed_and_canonical() {
        let phrases = [
            "queue depth exceeded",
            "retry scheduled",
            "batch committed",
            "cache flush complete",
            "worker heartbeat ok",
            "shard rebalanced",
            "index merge done",
            "page evicted",
            "commit retried",
            "drop detected",
            "sync paused",
            "threshold crossed",
        ];
        let mut values = Vec::new();
        for row in 0..192usize {
            let mut inner = String::new();
            let phrase_count = 1 + row % 6;
            for slot in 0..phrase_count {
                if slot != 0 {
                    inner.push_str("; ");
                }
                let choice = (row * 7 + slot * 5) % 20;
                if choice >= phrases.len() {
                    inner.push_str(&format!("{}", (row * 31 + slot) % 4096));
                } else {
                    inner.push_str(phrases[choice]);
                }
            }
            let message_len = 8 + (row * 13) % 40;
            if inner.len() < message_len {
                inner.push_str(&" ".repeat(message_len - inner.len()));
            }
            values.push(format!("\"{inner}\"").into_bytes());
        }
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::SegmentDict);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut malformed = packed.clone();
        malformed.pop();
        assert!(unpack_model(&malformed, values.len()).is_err());
    }

    #[test]
    fn decimal_delta_model_packs_mixed_progressions_and_is_canonical() {
        let values = (0..256u64)
            .map(|row| (row * 2 + row % 3).to_string().into_bytes())
            .collect::<Vec<_>>();
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::DecimalDelta);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut trailing = packed.clone();
        trailing.push(0);
        assert!(unpack_model(&trailing, values.len()).is_err());
        let mut truncated = packed.clone();
        truncated.pop();
        assert!(unpack_model(&truncated, values.len()).is_err());
    }

    #[test]
    fn r2_style_blocks_roundtrip_with_new_models() {
        let event = (0..256u32)
            .flat_map(|row| {
                let epoch = row / 128;
                let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
                format!(
                    "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"event\":\"session.open\",\"region\":\"eu-1\",\"request_id\":\"r-{prefix:08x}{:08x}\",\"status\":\"ok\",\"sequence\":{row}}}\n",
                    row % 28 + 1,
                    row % 24,
                    row % 128,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&event);
        let bursty = (0..256u32)
            .flat_map(|row| {
                format!(
                    "{{\"burst\":{},\"record\":{row},\"tags\":[\"hot\",\"retry\"],\"message\":\"queue depth exceeded; retry scheduled; {}\",\"source\":\"worker-{}\"}}\n",
                    row % 7,
                    row % 4096,
                    row % 23,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&bursty);
    }

    #[test]
    fn csv_tsv_and_logfmt_blocks_compress_and_roundtrip() {
        let csv = (0..512u32)
            .flat_map(|row| {
                let epoch = row / 128;
                let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
                format!(
                    "\"2026-01-{:02}T{:02}:{:02}:00Z\",{row},\"session.open\",\"eu-1\",\"r-{prefix:08x}{:08x}\",\"worker-{}\"\n",
                    row % 28 + 1,
                    (row / 60) % 24,
                    row % 60,
                    row % 128,
                    row % 16,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&csv);
        let csv_packet = encode_grammar_block(&csv).unwrap().unwrap();
        let mut csv_out = vec![0u8; csv.len()];
        decode_block_into(&csv_packet, &mut csv_out).unwrap();
        assert_eq!(csv_out, csv);
        assert!(csv_packet.len() * 20 < csv.len());

        let tsv = (0..512u32)
            .flat_map(|row| {
                format!(
                    "{row}\t{}\t\"queue depth exceeded; retry scheduled; {}\"\t\"worker-{}\"\n",
                    100_000 + row * 5,
                    row % 128,
                    row % 8,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&tsv);
        let tsv_packet = encode_grammar_block(&tsv).unwrap().unwrap();
        let mut tsv_out = vec![0u8; tsv.len()];
        decode_block_into(&tsv_packet, &mut tsv_out).unwrap();
        assert_eq!(tsv_out, tsv);
        assert!(tsv_packet.len() * 20 < tsv.len());

        let logfmt = (0..512u32)
            .flat_map(|row| {
                let epoch = row / 128;
                let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
                format!(
                    "ts=\"2026-01-{:02}T{:02}:{:02}:00Z\" level=info seq={row} req_id=\"r-{prefix:08x}{:08x}\" msg=\"queue depth exceeded; retry scheduled; {}\" worker=\"worker-{}\"\n",
                    row % 28 + 1,
                    (row / 60) % 24,
                    row % 60,
                    row % 128,
                    row % 64,
                    row % 12,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&logfmt);
        let logfmt_packet = encode_grammar_block(&logfmt).unwrap().unwrap();
        let mut logfmt_out = vec![0u8; logfmt.len()];
        decode_block_into(&logfmt_packet, &mut logfmt_out).unwrap();
        assert_eq!(logfmt_out, logfmt);
        assert!(logfmt_packet.len() * 20 < logfmt.len());
    }

    #[test]
    fn huffman_roundtrips_and_rejects_malformed() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        let mut cases: Vec<Vec<u8>> = Vec::new();
        // Skewed (delta-like) residuals of many lengths, including non-multiples of 4.
        for &n in &[32usize, 33, 256, 257, 1001, 4096, 16_383, 65_536] {
            cases.push((0..n).map(|_| { let r = rnd() % 100; if r < 60 { 0 } else if r < 80 { 1 } else if r < 90 { 255 } else { (r * 7) as u8 } }).collect());
        }
        // Fibonacci-like frequencies force length limiting.
        let mut fib = Vec::new();
        let (mut a, mut b) = (1usize, 1usize);
        for sym in 0..24u8 {
            fib.extend(std::iter::repeat(sym).take(a));
            (a, b) = (b, a + b);
            if fib.len() > 60_000 { break; }
        }
        cases.push(fib);
        cases.push(vec![42u8; 5000]);
        for src in &cases {
            let mut blob = Vec::new();
            if !huf_encode(src, &mut blob, src.len()) {
                assert!(src.len() < HUF_MIN, "len {} should compress", src.len());
                continue;
            }
            assert!(blob.len() < src.len());
            let mut out = vec![0u8; src.len()];
            huf_decode(&blob, &mut out).unwrap();
            assert_eq!(&out, src);
            // Malformed: truncation, trailing byte, wrong output size.
            for cut in [1usize, 2, blob.len() / 2, blob.len() - 1] {
                if cut < blob.len() {
                    let mut o = vec![0u8; src.len()];
                    assert!(huf_decode(&blob[..cut], &mut o).is_err() || o != *src);
                }
            }
            let mut extra = blob.clone();
            extra.push(0);
            let mut o = vec![0u8; src.len()];
            assert!(huf_decode(&extra, &mut o).is_err());
        }
        // Incomplete code (single length-1 symbol plus nothing) is rejected.
        assert!(huf_decode(&[HUF_KIND_HUF, 1, 0x01, 0, 0, 0], &mut [0u8; 8]).is_err());
        // Uniform random bytes do not compress.
        let rand: Vec<u8> = (0..4096).map(|_| rnd() as u8).collect();
        assert!(!huf_encode(&rand, &mut Vec::new(), rand.len()));
    }

    #[test]
    fn stride_image_blocks_compress_roundtrip_and_reject_malformed() {
        // 1. 32-bit BMP (54-byte header + 16370 RGBA pixels + 2 tail bytes = 65536 bytes)
        let mut bmp32 = vec![0u8; 65536];
        bmp32[0] = b'B';
        bmp32[1] = b'M';
        bmp32[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp32[28..30].copy_from_slice(&32u16.to_le_bytes());
        for px in 0..16370usize {
            let base = 54 + px * 4;
            let band = (px / 256) as u8;
            let edge = if px % 256 == 0 { (px / 17) as u8 } else { 0 };
            bmp32[base] = band.wrapping_mul(13).wrapping_add(edge);
            bmp32[base + 1] = band.wrapping_mul(29);
            bmp32[base + 2] = band.wrapping_mul(47).wrapping_sub(edge);
            bmp32[base + 3] = 255;
        }
        bmp32[65534] = 0xaa;
        bmp32[65535] = 0x55;
        let pkt32 = encode_grammar_block(&bmp32).unwrap().unwrap();
        assert_eq!(pkt32[0], STRIDE_PACKET);
        assert!(pkt32.len() * 20 < bmp32.len());
        let mut out32 = vec![0u8; bmp32.len()];
        decode_block_into(&pkt32, &mut out32).unwrap();
        assert_eq!(out32, bmp32);
        assert_eq!(decode_block(&pkt32).unwrap(), bmp32);

        // 2. 24-bit raw RGB image (stride = 3, 21845 pixels + 1 tail byte = 65536 bytes)
        let mut rgb24 = vec![0u8; 65536];
        for px in 0..21845usize {
            let base = px * 3;
            let run = (px / 128) as u8;
            rgb24[base] = run.wrapping_mul(7);
            rgb24[base + 1] = run.wrapping_mul(19).wrapping_add((px % 128 == 63) as u8);
            rgb24[base + 2] = run.wrapping_mul(37);
        }
        rgb24[65535] = 0x7e;
        let pkt24 = encode_grammar_block(&rgb24).unwrap().unwrap();
        assert_eq!(pkt24[0], STRIDE_PACKET);
        assert!(pkt24.len() * 10 < rgb24.len());
        let mut out24 = vec![0u8; rgb24.len()];
        decode_block_into(&pkt24, &mut out24).unwrap();
        assert_eq!(out24, rgb24);

        // 3. Reject truncated and trailing-byte STRIDE_PACKETs
        assert!(decode_block_into(&pkt32[..pkt32.len() - 1], &mut out32).is_err());
        let mut trailing = pkt32.clone();
        trailing.push(0);
        assert!(decode_block_into(&trailing, &mut out32).is_err());
    }

    #[test]
    fn fast_lz_packet_compresses_prose_code_and_rejects_malformed() {
        let mut prose = Vec::with_capacity(65536);
        for i in 0..512u32 {
            prose.extend_from_slice(
                format!(
                    "pub fn compute_block_digest_{}(input: &[u8], seed: u64) -> Result<u64> {{\n    let mut acc = seed.wrapping_add({});\n    for &byte in input {{\n        acc = acc.rotate_left(5) ^ (byte as u64);\n    }}\n    Ok(acc)\n}}\n",
                    i % 16,
                    i * 31,
                )
                .as_bytes(),
            );
        }
        prose.truncate(65536);
        let pkt = encode_grammar_block(&prose).unwrap().unwrap();
        assert_eq!(pkt[0], FAST_LZ_PACKET);
        assert!(pkt.len() * 4 < prose.len());
        let mut out = vec![0u8; prose.len()];
        decode_block_into(&pkt, &mut out).unwrap();
        assert_eq!(out, prose);
        assert_eq!(decode_block(&pkt).unwrap(), prose);

        assert!(decode_block_into(&pkt[..pkt.len() - 1], &mut out).is_err());
        let mut trailing = pkt.clone();
        trailing.push(0);
        assert!(decode_block_into(&trailing, &mut out).is_err());
    }

    #[test]
    fn front_dict_palette_base32_and_mixed_raw_rows_roundtrip() {
        const BASE32: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
        let mut mixed = Vec::new();
        for row in 0..160usize {
            // Interleaved plain-text line and structured JSON line with Base32 IDs + IPv4 + unique trace IDs:
            mixed.extend_from_slice(
                b"IP Middleware: [10.42.0.15] -> /api/v1/orders (200 OK)\n",
            );
            let user_idx = if row % 5 < 3 { 0 } else { (row / 3) % 19 };
            let mut user_id = String::with_capacity(26);
            for k in 0..26usize {
                let c = BASE32[(user_idx * 7 + k * 13) & 31] as char;
                user_id.push(c);
            }
            let ip = format!("172.69.{}.{}", 10 + (row / 8) % 12, 100 + (row % 23));
            let trace = format!("{:032x}", (row as u128 + 1) * 0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834u128);
            mixed.extend_from_slice(
                format!(
                    "{{\"time\":\"2026-03-19T13:55:{:02}.{:06}Z\",\"level\":\"INFO\",\"user\":\"{user_id}\",\"ip\":\"{ip}\",\"trace\":\"{trace}\"}}\n",
                    row % 60,
                    (row * 4321) % 1_000_000,
                )
                .as_bytes(),
            );
            if row % 16 == 0 {
                mixed.extend_from_slice(
                    format!(
                        "SELECT id, company_id, status FROM orders WHERE company_id = '{user_id}' AND deleted_at IS NULL LIMIT {};\n",
                        10 + row,
                    )
                    .as_bytes(),
                );
            }
        }
        roundtrip(&mixed);
        let pkt = encode_grammar_block(&mixed).unwrap().unwrap();
        assert_eq!(pkt[0], GRAMMAR_PACKET);
        let mut out = vec![0u8; mixed.len()];
        decode_block_into(&pkt, &mut out).unwrap();
        assert_eq!(out, mixed);
        assert_eq!(decode_block(&pkt).unwrap(), mixed);
    }

    fn stride_roundtrip(data: &[u8], expect_stride: u8) {
        let mut pkt = Vec::new();
        pack_stride_into(data, &mut pkt).expect("stride packet");
        let mut c = Cursor::new(&pkt[1..]);
        assert_eq!(c.uleb().unwrap(), data.len());
        let prefix = c.uleb().unwrap();
        c.take(prefix).unwrap();
        assert_eq!(c.byte().unwrap(), expect_stride);
        let mut out = vec![0u8; data.len()];
        decode_block_into(&pkt, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn stride2_stride8_and_large_block_roundtrip() {
        // 16-bit PCM-like ramp (stride 2).
        let s2: Vec<u8> = (0..30000u32)
            .flat_map(|i| (((i / 8) as u16).wrapping_mul(3)).to_le_bytes())
            .collect();
        stride_roundtrip(&s2, 2);
        // 64-bit monotonically increasing timestamps (stride 8).
        let s8: Vec<u8> = (0..8000u64)
            .flat_map(|i| (1_700_000_000_000u64 + i * 1000).to_le_bytes())
            .collect();
        stride_roundtrip(&s8, 8);
        // > 64 KiB 32-bit block (stride 4).
        let s4: Vec<u8> = (0..60000u32)
            .flat_map(|i| (0x1000_0000u32 + (i / 16)).to_le_bytes())
            .collect();
        assert!(s4.len() > 65536);
        stride_roundtrip(&s4, 4);
    }
}
