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
const MODEL_PATTERN: u8 = 15;
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
    Pattern,
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
            Self::Pattern => "digit-field-literal-pattern-mixed-radix-delta",
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
            Self::Pattern => 13,
        }
    }
}

#[derive(Clone, Debug)]
struct Template<'a> {
    segments: Vec<&'a [u8]>,
    values: Vec<Vec<&'a [u8]>>,
}

thread_local! {
    static ENCODE_VEC_POOL: std::cell::RefCell<Vec<Vec<&'static [u8]>>> =
        std::cell::RefCell::new(Vec::with_capacity(128));
    static ENCODE_SEG_POOL: std::cell::RefCell<Vec<Vec<&'static [u8]>>> =
        std::cell::RefCell::new(Vec::with_capacity(16));
    static ENCODE_OUTER_POOL: std::cell::RefCell<Vec<Vec<Vec<&'static [u8]>>>> =
        std::cell::RefCell::new(Vec::with_capacity(16));
    static ENCODE_TPL_POOL: std::cell::RefCell<Vec<Vec<Template<'static>>>> =
        std::cell::RefCell::new(Vec::with_capacity(8));
    static COL_BUF_POOL: std::cell::RefCell<Vec<Vec<u8>>> =
        std::cell::RefCell::new(Vec::with_capacity(8));
}

#[inline(always)]
fn take_tpl_vec<'a>() -> Vec<Template<'a>> {
    match ENCODE_TPL_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut v) => {
            v.clear();
            unsafe { core::mem::transmute::<Vec<Template<'static>>, Vec<Template<'a>>>(v) }
        }
        None => Vec::with_capacity(8),
    }
}

#[inline(always)]
fn return_tpl_vec<'a>(mut v: Vec<Template<'a>>) {
    v.clear();
    if (4..=64).contains(&v.capacity()) {
        ENCODE_TPL_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                if pool.len() < 8 {
                    let vs = unsafe {
                        core::mem::transmute::<Vec<Template<'a>>, Vec<Template<'static>>>(v)
                    };
                    pool.push(vs);
                }
            }
        });
    }
}

#[inline(always)]
fn take_model_buf(cap: usize) -> Vec<u8> {
    match COL_BUF_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut buf) => {
            buf.clear();
            buf.reserve(cap);
            buf
        }
        None => Vec::with_capacity(cap.max(4096)),
    }
}

#[inline(always)]
fn put_model_buf(mut buf: Vec<u8>) {
    if (64..=131_072).contains(&buf.capacity()) {
        buf.clear();
        COL_BUF_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                if pool.len() < 8 {
                    pool.push(buf);
                }
            }
        });
    }
}

/// Smallest capacity across a template's column vectors.
///
/// Columns come from a pool and can have different capacities, so the fast row loops must bound
/// their unchecked writes by the minimum over *all* slots, not by slot 0 alone (that was a heap
/// buffer overflow when slot 0 was larger than another slot).
#[inline(always)]
fn min_col_capacity(values: &[Vec<&[u8]>]) -> usize {
    values.iter().map(Vec::capacity).min().unwrap_or(0)
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

#[inline(always)]
fn take_seg_vec<'a>(cap: usize) -> Vec<&'a [u8]> {
    match ENCODE_SEG_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut v) => {
            v.clear();
            v.reserve(cap);
            unsafe { core::mem::transmute::<Vec<&'static [u8]>, Vec<&'a [u8]>>(v) }
        }
        None => Vec::with_capacity(cap.max(32)),
    }
}

#[inline(always)]
fn take_outer_col_vec<'a>(slots: usize, est_rows: usize) -> Vec<Vec<&'a [u8]>> {
    let mut outer = match ENCODE_OUTER_POOL.with(|cell| cell.borrow_mut().pop()) {
        Some(mut v) => {
            v.clear();
            v.reserve(slots);
            unsafe { core::mem::transmute::<Vec<Vec<&'static [u8]>>, Vec<Vec<&'a [u8]>>>(v) }
        }
        None => Vec::with_capacity(slots.max(32)),
    };
    for _ in 0..slots {
        outer.push(take_col_vec(est_rows));
    }
    outer
}

#[inline(always)]
fn reserve_raw_arena_for_block(
    raw_arena: &mut Vec<u8>,
    cursor: &Cursor<'_>,
    raw_len: usize,
    raw_count: usize,
    single_slots: Option<usize>,
) -> Result<()> {
    let has_lz_raw = raw_count > 0 && cursor.bytes.get(cursor.pos) == Some(&0);
    let mut lz_dict_cols = 0usize;
    let mut need_raw = has_lz_raw;
    if let Some(slots) = single_slots.filter(|_| raw_count == 0) {
        let mut cprobe = *cursor;
        for _ in 0..slots {
            let len = cprobe.uleb()?;
            let cb = cprobe.take(len)?;
            if let Some(&t) = cb.first() {
                if matches!(
                    t,
                    MODEL_RAW | MODEL_LZ_RAW | MODEL_DICTIONARY | MODEL_FRONT_DICT | MODEL_PATTERN
                ) {
                    need_raw = true;
                    if t == MODEL_FRONT_DICT && cb.get(1).is_some_and(|&h| (h & 0x80) != 0) {
                        lz_dict_cols += 1;
                    }
                }
            }
        }
    } else {
        need_raw = true;
        lz_dict_cols = cursor.bytes[cursor.pos..]
            .windows(2)
            .filter(|w| w[0] == MODEL_FRONT_DICT && (w[1] & 0x80) != 0)
            .count();
    }
    if need_raw {
        let cap = if raw_len <= 65536 && lz_dict_cols == 0 && !has_lz_raw {
            98_304
        } else {
            raw_len.max(65536).saturating_mul(2 + lz_dict_cols) + 16384
        };
        raw_arena.reserve(cap);
    }
    Ok(())
}

impl<'a> Drop for Template<'a> {
    #[inline(always)]
    fn drop(&mut self) {
        let mut outer = core::mem::take(&mut self.values);
        ENCODE_VEC_POOL.with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                for mut col in outer.drain(..) {
                    if pool.len() < 128 && col.capacity() >= 16 {
                        col.clear();
                        let vec_static = unsafe {
                            core::mem::transmute::<Vec<&'a [u8]>, Vec<&'static [u8]>>(col)
                        };
                        pool.push(vec_static);
                    }
                }
            } else {
                outer.clear();
            }
        });
        if (8..=256).contains(&outer.capacity()) {
            ENCODE_OUTER_POOL.with(|cell| {
                if let Ok(mut pool) = cell.try_borrow_mut() {
                    if pool.len() < 16 {
                        let o_static = unsafe {
                            core::mem::transmute::<Vec<Vec<&'a [u8]>>, Vec<Vec<&'static [u8]>>>(outer)
                        };
                        pool.push(o_static);
                    }
                }
            });
        }
        let mut segs = core::mem::take(&mut self.segments);
        if (8..=256).contains(&segs.capacity()) {
            segs.clear();
            ENCODE_SEG_POOL.with(|cell| {
                if let Ok(mut pool) = cell.try_borrow_mut() {
                    if pool.len() < 16 {
                        let s_static = unsafe {
                            core::mem::transmute::<Vec<&'a [u8]>, Vec<&'static [u8]>>(segs)
                        };
                        pool.push(s_static);
                    }
                }
            });
        }
    }
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
    model_counts: [usize; 14],
    model_bytes: [usize; 14],
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


include!("columns/bit_io.rs");
include!("columns/lut_num.rs");
include!("columns/scan_spans.rs");
include!("columns/parse_records.rs");
include!("columns/stream_match.rs");
include!("columns/grammar_match.rs");
include!("columns/grammar_extract.rs");
include!("columns/model_dict.rs");
include!("columns/model_front_dict.rs");
include!("columns/model_pattern.rs");
include!("columns/model_hex_epoch.rs");
include!("columns/model_segment_dec.rs");
include!("columns/model_periodic_gf2.rs");
include!("columns/model_timestamp.rs");
include!("columns/model_select.rs");
include!("columns/model_unpack.rs");
include!("columns/model_unpack_ext.rs");
include!("columns/grammar_pack.rs");
include!("columns/front_dict_unpack.rs");
include!("columns/swar_unpack.rs");
include!("columns/prep_column.rs");
include!("columns/prep_dict.rs");
include!("columns/prep_numeric.rs");
include!("columns/prep_time_epoch.rs");
include!("columns/prep_seg_gf2.rs");
include!("columns/prep_pattern.rs");
include!("columns/exec_column.rs");
include!("columns/template_build.rs");
include!("columns/template_bind.rs");
include!("columns/grammar_unpack.rs");
include!("columns/huffman_enc.rs");
include!("columns/huffman_dec.rs");
include!("columns/stride_interleave.rs");
include!("columns/stride_pack.rs");
include!("columns/stride_neon.rs");
include!("columns/stride_sse2.rs");
include!("columns/stride_planes.rs");
include!("columns/stride_unpack.rs");
include!("columns/stride_mode1.rs");
include!("columns/palette.rs");
include!("columns/fast_lz_core.rs");
include!("columns/fast_lz_pack.rs");
include!("columns/fast_lz_unpack.rs");
include!("columns/block_eval.rs");
include!("columns/query.rs");
include!("columns/query_bigram.rs");
include!("columns/query_nibble.rs");
include!("columns/query_filter.rs");
include!("columns/query_precheck.rs");
include!("columns/query_search.rs");
include!("columns/query_numeric.rs");
include!("columns/ops_header.rs");
include!("columns/ops_pred.rs");
include!("columns/ops_rewrite.rs");
include!("columns/ops_numeric_rewrite.rs");
include!("columns/ops_search.rs");
include!("columns/ops_expr.rs");
include!("columns/ops_expr_filter.rs");
include!("columns/ops_agg.rs");
include!("columns/ops_group.rs");
include!("columns/ops_project_plan.rs");
include!("columns/ops_project.rs");
include!("columns/tests_models.rs");
include!("columns/tests_packets.rs");
