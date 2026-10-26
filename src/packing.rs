//! Scalar, lossless command-metadata packing oracles for format research.
//!
//! This module is bench-only. It consumes the command trace emitted by the
//! current v2 encoder, never changes a matcher decision, and is deliberately
//! not a v3 wire format. Every layout below roundtrips all v2 opcodes,
//! including pattern seeds, before its size is reported.

use std::{collections::BTreeMap, hint::black_box};

use crate::{
    analysis::{literal_blocks, MetadataCommand},
    block, Error, Frame, Result,
};

pub const GROUP_SIZES: [usize; 3] = [32, 64, 128];
pub(super) const FIELD_COUNT: usize = 6;
pub(super) const NUMERIC_FIELD_COUNT: usize = FIELD_COUNT - 1;
pub(super) const TOKEN_WIDTH: usize = 8;
pub(super) const TOKEN_DICTIONARY_LIMITS: [usize; 4] = [16, 32, 64, 128];

/// Layouts tested in P3. `Bitsliced` has the same field plan and bit count as
/// `Fixed`, but transposes each field into bit planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Fixed,
    Bitsliced,
    Int4,
    Ternary2Bit,
    TernaryBase3,
}

impl Layout {
    pub const ALL: [Self; 5] = [
        Self::Fixed,
        Self::Bitsliced,
        Self::Int4,
        Self::Ternary2Bit,
        Self::TernaryBase3,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed-width-exceptions",
            Self::Bitsliced => "bitsliced-exceptions",
            Self::Int4 => "int4-escape15",
            Self::Ternary2Bit => "ternary-classes-2bit",
            Self::TernaryBase3 => "ternary-classes-base3",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayoutResult {
    pub layout: Layout,
    pub group_size: usize,
    pub commands: usize,
    pub groups: usize,
    /// Bytes that replace a v2 LZ command/control/attribute stream. It includes
    /// the prototype's four-byte group-count header for every LZ block.
    pub metadata_bytes: usize,
    /// Largest temporary group buffer. The prototype deliberately does not
    /// materialize an entire decoded command tape in the timed unpack loop.
    pub max_group_bytes: usize,
    pub pack_ns: Option<u128>,
    pub unpack_ns: Option<u128>,
}

#[derive(Debug, Clone)]
pub struct Evaluation {
    pub input_bytes: usize,
    pub frame_bytes: usize,
    pub common_frame_envelope_bytes: usize,
    pub raw_passthrough_bytes: usize,
    pub literal_bytes: usize,
    pub current_v2_metadata_bytes: usize,
    pub lz_blocks: usize,
    pub raw_blocks: usize,
    pub results: Vec<LayoutResult>,
    pub distance_results: Vec<DistanceResult>,
    pub bounded_entropy: EntropyResult,
    pub adaptive_entropy: AdaptiveEntropyResult,
    pub literal_entropy: LiteralEntropyResult,
}

/// Exact, standalone distance-stream alternatives. These intentionally measure
/// distance payloads only; they are not eligible to claim a full-frame ratio
/// until P4 joins selector, length, literal and execution costs.
#[derive(Debug, Clone)]
pub struct DistanceResult {
    pub model: &'static str,
    pub group_size: Option<usize>,
    pub distances: usize,
    pub metadata_bytes: usize,
    pub max_scratch_bytes: usize,
    pub pack_ns: Option<u128>,
    pub unpack_ns: Option<u128>,
}

#[derive(Clone, Copy)]
pub(super) enum DistanceModel {
    Absolute,
    Delta,
    Xor,
    Cache(usize),
    Dictionary16,
}

impl DistanceModel {
    pub(super) const ALL: [Self; 6] = [
        Self::Absolute,
        Self::Delta,
        Self::Xor,
        Self::Cache(2),
        Self::Cache(4),
        Self::Cache(8),
    ];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Absolute => "absolute-u16-escape-u32",
            Self::Delta => "delta-zigzag-uleb",
            Self::Xor => "xor-uleb",
            Self::Cache(2) => "mru-cache-2",
            Self::Cache(4) => "mru-cache-4",
            Self::Cache(8) => "mru-cache-8",
            Self::Cache(_) => "mru-cache",
            Self::Dictionary16 => "group-dictionary-16",
        }
    }
}

pub(super) struct PackedDistances {
    pub(super) blocks: Vec<Vec<u8>>,
    pub(super) distances: usize,
    pub(super) metadata_bytes: usize,
    pub(super) max_scratch_bytes: usize,
}

/// A full logical-command prototype using per-block canonical Huffman fields
/// plus an 8-entry MRU distance selector. This is intentionally separate from
/// the grouped P3 layouts: it charges each codebook, bit length and escape.
#[derive(Debug, Clone)]
pub struct EntropyResult {
    pub commands: usize,
    pub metadata_bytes: usize,
    pub max_block_scratch_bytes: usize,
    pub pack_ns: Option<u128>,
    pub unpack_ns: Option<u128>,
}

/// Per-block fixed-vs-entropy selection. The pack loop deliberately builds
/// both payloads before making the choice, so selection work is charged.
#[derive(Debug, Clone)]
pub struct AdaptiveEntropyResult {
    pub commands: usize,
    pub metadata_bytes: usize,
    pub max_block_scratch_bytes: usize,
    pub entropy_blocks: usize,
    pub fixed_blocks: usize,
    pub pack_ns: Option<u128>,
    pub unpack_ns: Option<u128>,
}

/// Exact byte-literal entropy prototype. Each LZ block builds raw and
/// canonical-Huffman candidates, writes a selector, and retains the smaller
/// one. It replaces literal payload bytes only; current v2 command/prefix
/// bytes remain charged separately in its total-frame estimate.
#[derive(Debug, Clone)]
pub struct LiteralEntropyResult {
    pub literal_bytes: usize,
    pub encoded_bytes: usize,
    pub max_block_scratch_bytes: usize,
    pub dictionary_entries: usize,
    pub entropy_blocks: usize,
    pub dictionary_blocks: usize,
    pub raw_blocks: usize,
    pub pack_ns: Option<u128>,
    pub unpack_ns: Option<u128>,
}

pub(super) struct PackedEntropy {
    pub(super) blocks: Vec<Vec<u8>>,
    pub(super) commands: usize,
    pub(super) metadata_bytes: usize,
    pub(super) max_block_scratch_bytes: usize,
}

pub(super) struct PackedAdaptiveEntropy {
    pub(super) blocks: Vec<Vec<u8>>,
    pub(super) commands: usize,
    pub(super) metadata_bytes: usize,
    pub(super) max_block_scratch_bytes: usize,
    pub(super) entropy_blocks: usize,
    pub(super) fixed_blocks: usize,
}

pub(super) struct PackedLiteralEntropy {
    pub(super) blocks: Vec<Vec<u8>>,
    pub(super) literal_lengths: Vec<usize>,
    pub(super) literal_bytes: usize,
    pub(super) encoded_bytes: usize,
    pub(super) max_block_scratch_bytes: usize,
    pub(super) dictionary_entries: usize,
    pub(super) entropy_blocks: usize,
    pub(super) dictionary_blocks: usize,
    pub(super) raw_blocks: usize,
}

#[derive(Clone)]
pub(super) struct HuffmanCode {
    pub(super) symbol: u32,
    pub(super) len: u8,
}

#[derive(Clone)]
pub(super) struct HuffmanCodebook {
    pub(super) entries: Vec<HuffmanCode>,
    pub(super) by_symbol: BTreeMap<u32, (u32, u8)>,
    pub(super) by_code: BTreeMap<(u8, u32), u32>,
    pub(super) max_len: u8,
}

#[derive(Debug)]
pub struct PackedLayout {
    pub(super) blocks: Vec<Vec<u8>>,
    pub(super) commands: usize,
    pub(super) groups: usize,
    pub(super) metadata_bytes: usize,
    pub(super) max_group_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FieldPlan {
    pub(super) base: u32,
    pub(super) width: u8,
    pub(super) exception_count: u8,
}

#[derive(Default)]
pub(super) struct BitWriter {
    pub(super) bytes: Vec<u8>,
    pub(super) bit: u8,
}

pub(super) struct BitReader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) bit: usize,
}

pub(super) struct Cursor<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) pos: usize,
}

pub(super) fn invalid(message: &'static str) -> Error {
    Error::InvalidFrame(message)
}

pub(super) fn fields(command: MetadataCommand) -> [u32; FIELD_COUNT] {
    [
        command.op as u32,
        command.lit,
        command.run,
        command.distance,
        command.seed_lo,
        command.seed_hi,
    ]
}

pub(super) fn command(values: [u32; FIELD_COUNT]) -> Result<MetadataCommand> {
    let op = u8::try_from(values[0]).map_err(|_| invalid("packed opcode range"))?;
    if op > 7 {
        return Err(invalid("packed opcode range"));
    }
    Ok(MetadataCommand {
        op,
        lit: values[1],
        run: values[2],
        distance: values[3],
        seed_lo: values[4],
        seed_hi: values[5],
    })
}

pub(super) fn bits_to_bytes(bits: usize) -> Result<usize> {
    bits.checked_add(7)
        .map(|value| value / 8)
        .ok_or(Error::SizeOverflow)
}

pub(super) fn width(value: u32) -> u8 {
    (u32::BITS - value.leading_zeros()) as u8
}

pub(super) fn escape(width: u8) -> u32 {
    if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

pub(super) fn direct(residual: u32, width: u8) -> bool {
    width == 0 && residual == 0 || width != 0 && residual < escape(width)
}

pub(super) fn field_plan(values: &[u32]) -> Result<FieldPlan> {
    let base = *values.iter().min().ok_or_else(|| invalid("empty group"))?;
    let max = *values.iter().max().ok_or_else(|| invalid("empty group"))?;
    let max_width = width(max - base);
    let mut best = None;
    for candidate_width in 0..=max_width {
        let exceptions = values
            .iter()
            .filter(|&&value| !direct(value - base, candidate_width))
            .count();
        let data = bits_to_bytes(
            values
                .len()
                .checked_mul(candidate_width as usize)
                .ok_or(Error::SizeOverflow)?,
        )?;
        let exception_bytes = exceptions.checked_mul(5).ok_or(Error::SizeOverflow)?;
        let cost = data
            .checked_add(exception_bytes)
            .ok_or(Error::SizeOverflow)?;
        let candidate = (cost, exceptions, candidate_width);
        if best.map_or(true, |old| candidate < old) {
            best = Some(candidate);
        }
    }
    let (_, exceptions, chosen_width) = best.ok_or_else(|| invalid("empty group"))?;
    Ok(FieldPlan {
        base,
        width: chosen_width,
        exception_count: u8::try_from(exceptions).map_err(|_| Error::SizeOverflow)?,
    })
}

impl BitWriter {
    pub(super) fn push(&mut self, value: u32, width: u8) -> Result<()> {
        if width < 32 && value >= (1u32 << width) {
            return Err(invalid("packed value width"));
        }
        for index in 0..width {
            if self.bit == 0 {
                self.bytes.push(0);
            }
            if value & (1u32 << index) != 0 {
                *self.bytes.last_mut().unwrap() |= 1 << self.bit;
            }
            self.bit = (self.bit + 1) & 7;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Vec<u8> {
        self.bytes
    }

    pub(super) fn bit_len(&self) -> usize {
        if self.bytes.is_empty() {
            0
        } else if self.bit == 0 {
            self.bytes.len() * 8
        } else {
            (self.bytes.len() - 1) * 8 + self.bit as usize
        }
    }
}

impl<'a> BitReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    pub(super) fn read(&mut self, width: u8) -> Result<u32> {
        let end = self
            .bit
            .checked_add(width as usize)
            .ok_or(Error::SizeOverflow)?;
        if end > self.bytes.len().saturating_mul(8) {
            return Err(Error::Truncated);
        }
        let mut value = 0;
        for index in 0..width {
            let bit = self.bit + index as usize;
            value |= (((self.bytes[bit / 8] >> (bit & 7)) & 1) as u32) << index;
        }
        self.bit = end;
        Ok(value)
    }

    pub(super) fn finish_zero_padded(&self) -> Result<()> {
        for bit in self.bit..self.bytes.len().saturating_mul(8) {
            if self.bytes[bit / 8] & (1 << (bit & 7)) != 0 {
                return Err(invalid("non-canonical packed tail"));
            }
        }
        Ok(())
    }
}

impl<'a> Cursor<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub(super) fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(count).ok_or(Error::SizeOverflow)?;
        let out = self.bytes.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(out)
    }

    pub(super) fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(super) fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub(super) fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}


mod distances;
mod entropy;
mod eval;
mod group_fixed;
mod group_var;
mod literal_dict;
#[cfg(test)]
mod tests;

#[allow(unused_imports)]
use distances::*;
pub use distances::time_distance_models;
#[allow(unused_imports)]
use entropy::*;
pub use entropy::{time_adaptive_entropy, time_bounded_entropy};
pub use eval::{evaluate_frame, time_layouts};
use group_fixed::*;
#[allow(unused_imports)]
use group_var::*;
pub use group_var::{pack_blocks, unpack_blocks};
#[allow(unused_imports)]
use literal_dict::*;
pub use literal_dict::{time_frame_literal_entropy, time_literal_entropy};
