//! Scalar, lossless command-metadata packing oracles for format research.
//!
//! This module is bench-only. It consumes the command trace emitted by the
//! current v2 encoder, never changes a matcher decision, and is deliberately
//! not a v3 wire format. Every layout below roundtrips all v2 opcodes,
//! including pattern seeds, before its size is reported.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    fmt::Write,
    hint::black_box,
    time::Instant,
};

use crate::{
    analysis::{command_blocks, literal_blocks, MetadataCommand},
    block, frame, Error, Frame, Result,
};

pub const GROUP_SIZES: [usize; 3] = [32, 64, 128];
const FIELD_COUNT: usize = 6;
const NUMERIC_FIELD_COUNT: usize = FIELD_COUNT - 1;
const TOKEN_WIDTH: usize = 8;
const TOKEN_DICTIONARY_LIMITS: [usize; 4] = [16, 32, 64, 128];

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
enum DistanceModel {
    Absolute,
    Delta,
    Xor,
    Cache(usize),
    Dictionary16,
}

impl DistanceModel {
    const ALL: [Self; 6] = [
        Self::Absolute,
        Self::Delta,
        Self::Xor,
        Self::Cache(2),
        Self::Cache(4),
        Self::Cache(8),
    ];

    const fn name(self) -> &'static str {
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

struct PackedDistances {
    blocks: Vec<Vec<u8>>,
    distances: usize,
    metadata_bytes: usize,
    max_scratch_bytes: usize,
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

struct PackedEntropy {
    blocks: Vec<Vec<u8>>,
    commands: usize,
    metadata_bytes: usize,
    max_block_scratch_bytes: usize,
}

struct PackedAdaptiveEntropy {
    blocks: Vec<Vec<u8>>,
    commands: usize,
    metadata_bytes: usize,
    max_block_scratch_bytes: usize,
    entropy_blocks: usize,
    fixed_blocks: usize,
}

struct PackedLiteralEntropy {
    blocks: Vec<Vec<u8>>,
    literal_lengths: Vec<usize>,
    literal_bytes: usize,
    encoded_bytes: usize,
    max_block_scratch_bytes: usize,
    dictionary_entries: usize,
    entropy_blocks: usize,
    dictionary_blocks: usize,
    raw_blocks: usize,
}

#[derive(Clone)]
struct HuffmanCode {
    symbol: u32,
    len: u8,
}

#[derive(Clone)]
struct HuffmanCodebook {
    entries: Vec<HuffmanCode>,
    by_symbol: BTreeMap<u32, (u32, u8)>,
    by_code: BTreeMap<(u8, u32), u32>,
    max_len: u8,
}

#[derive(Debug)]
pub struct PackedLayout {
    blocks: Vec<Vec<u8>>,
    commands: usize,
    groups: usize,
    metadata_bytes: usize,
    max_group_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct FieldPlan {
    base: u32,
    width: u8,
    exception_count: u8,
}

#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit: u8,
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

fn invalid(message: &'static str) -> Error {
    Error::InvalidFrame(message)
}

fn fields(command: MetadataCommand) -> [u32; FIELD_COUNT] {
    [
        command.op as u32,
        command.lit,
        command.run,
        command.distance,
        command.seed_lo,
        command.seed_hi,
    ]
}

fn command(values: [u32; FIELD_COUNT]) -> Result<MetadataCommand> {
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

fn bits_to_bytes(bits: usize) -> Result<usize> {
    bits.checked_add(7)
        .map(|value| value / 8)
        .ok_or(Error::SizeOverflow)
}

fn width(value: u32) -> u8 {
    (u32::BITS - value.leading_zeros()) as u8
}

fn escape(width: u8) -> u32 {
    if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

fn direct(residual: u32, width: u8) -> bool {
    width == 0 && residual == 0 || width != 0 && residual < escape(width)
}

fn field_plan(values: &[u32]) -> Result<FieldPlan> {
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
    fn push(&mut self, value: u32, width: u8) -> Result<()> {
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

    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn bit_len(&self) -> usize {
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
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    fn read(&mut self, width: u8) -> Result<u32> {
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

    fn finish_zero_padded(&self) -> Result<()> {
        for bit in self.bit..self.bytes.len().saturating_mul(8) {
            if self.bytes[bit / 8] & (1 << (bit & 7)) != 0 {
                return Err(invalid("non-canonical packed tail"));
            }
        }
        Ok(())
    }
}

fn reverse_low_bits(value: u32, len: u8) -> u32 {
    if len == 32 {
        value.reverse_bits()
    } else {
        value.reverse_bits() >> (32 - len)
    }
}

impl HuffmanCodebook {
    fn from_lengths(mut lengths: Vec<(u32, u8)>) -> Result<Self> {
        if lengths.is_empty() {
            return Err(invalid("empty Huffman codebook"));
        }
        lengths.sort_unstable_by_key(|&(symbol, len)| (len, symbol));
        let mut entries = Vec::with_capacity(lengths.len());
        let mut by_symbol = BTreeMap::new();
        let mut by_code = BTreeMap::new();
        let mut code = 0u64;
        let mut previous_len = 0u8;
        for (symbol, len) in lengths {
            if len == 0 || len > 32 || by_symbol.contains_key(&symbol) {
                return Err(invalid("Huffman codebook"));
            }
            code <<= (len - previous_len) as u32;
            if code >= (1u64 << len) {
                return Err(invalid("oversubscribed Huffman codebook"));
            }
            let reversed = reverse_low_bits(code as u32, len);
            by_symbol.insert(symbol, (reversed, len));
            if by_code.insert((len, reversed), symbol).is_some() {
                return Err(invalid("duplicate Huffman code"));
            }
            entries.push(HuffmanCode {
                symbol,
                len,
            });
            code += 1;
            previous_len = len;
        }
        Ok(Self {
            max_len: previous_len,
            entries,
            by_symbol,
            by_code,
        })
    }

    fn from_values(values: &[u32]) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid("empty Huffman field"));
        }
        let mut counts = BTreeMap::new();
        for &value in values {
            *counts.entry(value).or_insert(0u64) += 1;
        }
        if counts.len() == 1 {
            return Self::from_lengths(vec![(*counts.keys().next().unwrap(), 1)]);
        }
        let mut children: Vec<Option<(usize, usize)>> = Vec::with_capacity(counts.len() * 2);
        let mut symbols = Vec::with_capacity(counts.len() * 2);
        let mut heap = BinaryHeap::new();
        for (index, (&symbol, &count)) in counts.iter().enumerate() {
            children.push(None);
            symbols.push(Some(symbol));
            heap.push(Reverse((count, symbol as u64, index)));
        }
        let mut ordinal = 1u64 << 32;
        while heap.len() > 1 {
            let Reverse((left_count, _, left)) = heap.pop().unwrap();
            let Reverse((right_count, _, right)) = heap.pop().unwrap();
            let index = children.len();
            children.push(Some((left, right)));
            symbols.push(None);
            heap.push(Reverse((
                left_count.checked_add(right_count).ok_or(Error::SizeOverflow)?,
                ordinal,
                index,
            )));
            ordinal = ordinal.checked_add(1).ok_or(Error::SizeOverflow)?;
        }
        let Reverse((_, _, root)) = heap.pop().unwrap();
        let mut lengths = Vec::with_capacity(counts.len());
        let mut pending = vec![(root, 0u8)];
        while let Some((node, depth)) = pending.pop() {
            if let Some(symbol) = symbols[node] {
                if depth == 0 || depth > 32 {
                    return Err(invalid("Huffman code length"));
                }
                lengths.push((symbol, depth));
            } else {
                let (left, right) = children[node].ok_or_else(|| invalid("Huffman tree"))?;
                let next = depth.checked_add(1).ok_or(Error::SizeOverflow)?;
                pending.push((left, next));
                pending.push((right, next));
            }
        }
        Self::from_lengths(lengths)
    }

    fn write_symbol(&self, writer: &mut BitWriter, symbol: u32) -> Result<()> {
        let &(reversed, len) = self
            .by_symbol
            .get(&symbol)
            .ok_or_else(|| invalid("Huffman symbol"))?;
        writer.push(reversed, len)
    }

    fn read_symbol(&self, reader: &mut BitReader<'_>) -> Result<u32> {
        let mut reversed = 0u32;
        for len in 1..=self.max_len {
            reversed |= reader.read(1)? << (len - 1);
            if let Some(&symbol) = self.by_code.get(&(len, reversed)) {
                return Ok(symbol);
            }
        }
        Err(invalid("Huffman code"))
    }
}

fn write_huffman_field(out: &mut Vec<u8>, values: &[u32]) -> Result<()> {
    let codebook = HuffmanCodebook::from_values(values)?;
    out.extend_from_slice(
        &u16::try_from(codebook.entries.len())
            .map_err(|_| Error::SizeOverflow)?
            .to_le_bytes(),
    );
    for entry in &codebook.entries {
        out.extend_from_slice(&entry.symbol.to_le_bytes());
        out.push(entry.len);
    }
    let mut writer = BitWriter::default();
    for &value in values {
        codebook.write_symbol(&mut writer, value)?;
    }
    let bit_len = writer.bit_len();
    out.extend_from_slice(
        &u32::try_from(bit_len)
            .map_err(|_| Error::SizeOverflow)?
            .to_le_bytes(),
    );
    out.extend_from_slice(&writer.finish());
    Ok(())
}

fn read_huffman_field(cursor: &mut Cursor<'_>, count: usize) -> Result<Vec<u32>> {
    let table_len = u16::from_le_bytes(cursor.take(2)?.try_into().unwrap()) as usize;
    if table_len == 0 || table_len > count {
        return Err(invalid("Huffman table length"));
    }
    let mut raw_entries = Vec::with_capacity(table_len);
    for _ in 0..table_len {
        raw_entries.push((cursor.u32()?, cursor.byte()?));
    }
    let codebook = HuffmanCodebook::from_lengths(raw_entries.clone())?;
    let canonical_entries: Vec<_> = codebook
        .entries
        .iter()
        .map(|entry| (entry.symbol, entry.len))
        .collect();
    if raw_entries != canonical_entries {
        return Err(invalid("non-canonical Huffman table"));
    }
    let bit_len = cursor.u32()? as usize;
    let bytes = bits_to_bytes(bit_len)?;
    let mut reader = BitReader::new(cursor.take(bytes)?);
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(codebook.read_symbol(&mut reader)?);
    }
    if reader.bit != bit_len {
        return Err(invalid("Huffman bit length"));
    }
    reader.finish_zero_padded()?;
    Ok(values)
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(count).ok_or(Error::SizeOverflow)?;
        let out = self.bytes.get(self.pos..end).ok_or(Error::Truncated)?;
        self.pos = end;
        Ok(out)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn done(&self) -> bool {
        self.pos == self.bytes.len()
    }
}

fn plans(commands: &[MetadataCommand]) -> Result<[FieldPlan; FIELD_COUNT]> {
    if commands.is_empty() || commands.len() > u8::MAX as usize {
        return Err(invalid("packed group count"));
    }
    let mut all = [[0u32; 128]; FIELD_COUNT];
    for (index, &command) in commands.iter().enumerate() {
        let values = fields(command);
        for field in 0..FIELD_COUNT {
            all[field][index] = values[field];
        }
    }
    let mut result = [FieldPlan {
        base: 0,
        width: 0,
        exception_count: 0,
    }; FIELD_COUNT];
    for field in 0..FIELD_COUNT {
        result[field] = field_plan(&all[field][..commands.len()])?;
    }
    Ok(result)
}

fn write_plan_header(
    out: &mut Vec<u8>,
    count: usize,
    plans: &[FieldPlan; FIELD_COUNT],
) -> Result<()> {
    out.push(u8::try_from(count).map_err(|_| Error::SizeOverflow)?);
    for plan in plans {
        out.extend_from_slice(&plan.base.to_le_bytes());
        out.push(plan.width);
        out.push(plan.exception_count);
    }
    Ok(())
}

fn read_plan_header(cursor: &mut Cursor<'_>) -> Result<(usize, [FieldPlan; FIELD_COUNT])> {
    let count = cursor.byte()? as usize;
    if count == 0 || count > 128 {
        return Err(invalid("packed group count"));
    }
    let mut plans = [FieldPlan {
        base: 0,
        width: 0,
        exception_count: 0,
    }; FIELD_COUNT];
    for plan in &mut plans {
        let base = cursor.u32()?;
        let width = cursor.byte()?;
        let exception_count = cursor.byte()?;
        if width > 32 || exception_count as usize > count {
            return Err(invalid("packed field plan"));
        }
        *plan = FieldPlan {
            base,
            width,
            exception_count,
        };
    }
    Ok((count, plans))
}

fn residuals(commands: &[MetadataCommand], field: usize, plan: FieldPlan) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(commands.len());
    for &command in commands {
        let value = fields(command)[field];
        out.push(
            value
                .checked_sub(plan.base)
                .ok_or_else(|| invalid("packed base"))?,
        );
    }
    Ok(out)
}

fn write_exceptions(out: &mut Vec<u8>, residuals: &[u32], plan: FieldPlan) -> Result<()> {
    let mut count = 0usize;
    for (index, &residual) in residuals.iter().enumerate() {
        if !direct(residual, plan.width) {
            out.push(u8::try_from(index).map_err(|_| Error::SizeOverflow)?);
            out.extend_from_slice(&residual.to_le_bytes());
            count += 1;
        }
    }
    if count != plan.exception_count as usize {
        return Err(invalid("packed exception count"));
    }
    Ok(())
}

fn write_fixed_field(out: &mut Vec<u8>, residuals: &[u32], plan: FieldPlan) -> Result<()> {
    let mut bits = BitWriter::default();
    for &residual in residuals {
        bits.push(
            if direct(residual, plan.width) {
                residual
            } else {
                escape(plan.width)
            },
            plan.width,
        )?;
    }
    out.extend_from_slice(&bits.finish());
    write_exceptions(out, residuals, plan)
}

fn write_bitsliced_field(out: &mut Vec<u8>, residuals: &[u32], plan: FieldPlan) -> Result<()> {
    let mut bits = BitWriter::default();
    let codes: Vec<u32> = residuals
        .iter()
        .map(|&residual| {
            if direct(residual, plan.width) {
                residual
            } else {
                escape(plan.width)
            }
        })
        .collect();
    for bit in 0..plan.width {
        for &code in &codes {
            bits.push((code >> bit) & 1, 1)?;
        }
    }
    out.extend_from_slice(&bits.finish());
    write_exceptions(out, residuals, plan)
}

fn read_exceptions(
    cursor: &mut Cursor<'_>,
    residuals: &mut [u32],
    plan: FieldPlan,
    escaped: &[usize],
) -> Result<()> {
    if plan.width != 0 && escaped.len() != plan.exception_count as usize {
        return Err(invalid("packed exception marker"));
    }
    let mut last = None;
    for ordinal in 0..plan.exception_count as usize {
        let index = cursor.byte()? as usize;
        let value = cursor.u32()?;
        if index >= residuals.len() || last.is_some_and(|previous| index <= previous) {
            return Err(invalid("packed exception index"));
        }
        if plan.width != 0 && escaped.get(ordinal).copied() != Some(index) {
            return Err(invalid("packed exception marker"));
        }
        if direct(value, plan.width) {
            return Err(invalid("non-canonical packed exception"));
        }
        residuals[index] = value;
        last = Some(index);
    }
    Ok(())
}

fn read_fixed_field(cursor: &mut Cursor<'_>, count: usize, plan: FieldPlan) -> Result<Vec<u32>> {
    let bytes = bits_to_bytes(
        count
            .checked_mul(plan.width as usize)
            .ok_or(Error::SizeOverflow)?,
    )?;
    let mut reader = BitReader::new(cursor.take(bytes)?);
    let mut residuals = Vec::with_capacity(count);
    let mut escaped = Vec::new();
    for index in 0..count {
        let code = reader.read(plan.width)?;
        if plan.width != 0 && code == escape(plan.width) {
            escaped.push(index);
        }
        residuals.push(if plan.width == 0 { 0 } else { code });
    }
    reader.finish_zero_padded()?;
    read_exceptions(cursor, &mut residuals, plan, &escaped)?;
    for residual in &mut residuals {
        *residual = plan
            .base
            .checked_add(*residual)
            .ok_or(Error::SizeOverflow)?;
    }
    Ok(residuals)
}

fn read_bitsliced_field(
    cursor: &mut Cursor<'_>,
    count: usize,
    plan: FieldPlan,
) -> Result<Vec<u32>> {
    let bytes = bits_to_bytes(
        count
            .checked_mul(plan.width as usize)
            .ok_or(Error::SizeOverflow)?,
    )?;
    let mut reader = BitReader::new(cursor.take(bytes)?);
    let mut residuals = vec![0u32; count];
    for bit in 0..plan.width {
        for residual in &mut residuals {
            *residual |= reader.read(1)? << bit;
        }
    }
    reader.finish_zero_padded()?;
    let escaped: Vec<usize> = if plan.width == 0 {
        Vec::new()
    } else {
        residuals
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value == escape(plan.width)).then_some(index))
            .collect()
    };
    read_exceptions(cursor, &mut residuals, plan, &escaped)?;
    for residual in &mut residuals {
        *residual = plan
            .base
            .checked_add(*residual)
            .ok_or(Error::SizeOverflow)?;
    }
    Ok(residuals)
}

fn commands_from_fields(
    values: [[u32; 128]; FIELD_COUNT],
    count: usize,
) -> Result<Vec<MetadataCommand>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(command(std::array::from_fn(|field| values[field][index]))?);
    }
    Ok(out)
}

fn pack_fixed_group(commands: &[MetadataCommand], bitsliced: bool) -> Result<Vec<u8>> {
    let plans = plans(commands)?;
    let mut out = Vec::new();
    write_plan_header(&mut out, commands.len(), &plans)?;
    for (field, plan) in plans.into_iter().enumerate() {
        let values = residuals(commands, field, plan)?;
        if bitsliced {
            write_bitsliced_field(&mut out, &values, plan)?;
        } else {
            write_fixed_field(&mut out, &values, plan)?;
        }
    }
    Ok(out)
}

fn unpack_fixed_group(cursor: &mut Cursor<'_>, bitsliced: bool) -> Result<Vec<MetadataCommand>> {
    let (count, plans) = read_plan_header(cursor)?;
    let mut values = [[0u32; 128]; FIELD_COUNT];
    for (field, plan) in plans.into_iter().enumerate() {
        let decoded = if bitsliced {
            read_bitsliced_field(cursor, count, plan)?
        } else {
            read_fixed_field(cursor, count, plan)?
        };
        values[field][..count].copy_from_slice(&decoded);
    }
    commands_from_fields(values, count)
}

fn bases(commands: &[MetadataCommand], fields_to_pack: &[usize]) -> Result<Vec<u32>> {
    fields_to_pack
        .iter()
        .map(|&field| {
            commands
                .iter()
                .map(|&command| fields(command)[field])
                .min()
                .ok_or_else(|| invalid("empty group"))
        })
        .collect()
}

fn write_int4_field(
    out: &mut Vec<u8>,
    commands: &[MetadataCommand],
    field: usize,
    base: u32,
) -> Result<()> {
    let mut nibbles = Vec::with_capacity(commands.len().div_ceil(2));
    let mut exceptions = Vec::new();
    for (index, &command) in commands.iter().enumerate() {
        let residual = fields(command)[field]
            .checked_sub(base)
            .ok_or_else(|| invalid("int4 base"))?;
        let code = if residual < 15 {
            residual as u8
        } else {
            exceptions.push((index, residual));
            15
        };
        if index & 1 == 0 {
            nibbles.push(code);
        } else {
            *nibbles.last_mut().unwrap() |= code << 4;
        }
    }
    out.extend_from_slice(&nibbles);
    for (index, residual) in exceptions {
        out.push(u8::try_from(index).map_err(|_| Error::SizeOverflow)?);
        out.extend_from_slice(&residual.to_le_bytes());
    }
    Ok(())
}

fn read_int4_field(cursor: &mut Cursor<'_>, count: usize, base: u32) -> Result<Vec<u32>> {
    let bytes = cursor.take(count.div_ceil(2))?;
    if count & 1 != 0 && bytes.last().unwrap() & 0xf0 != 0 {
        return Err(invalid("non-canonical int4 tail nibble"));
    }
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let code = (bytes[index / 2] >> if index & 1 == 0 { 0 } else { 4 }) & 15;
        let residual = if code < 15 {
            code as u32
        } else {
            let recorded = cursor.byte()? as usize;
            let residual = cursor.u32()?;
            if recorded != index || residual < 15 {
                return Err(invalid("non-canonical int4 exception"));
            }
            residual
        };
        out.push(base.checked_add(residual).ok_or(Error::SizeOverflow)?);
    }
    Ok(out)
}

fn pack_int4_group(commands: &[MetadataCommand]) -> Result<Vec<u8>> {
    if commands.is_empty() || commands.len() > 128 {
        return Err(invalid("packed group count"));
    }
    let fields_to_pack: Vec<usize> = (0..FIELD_COUNT).collect();
    let bases = bases(commands, &fields_to_pack)?;
    let mut out = Vec::new();
    out.push(commands.len() as u8);
    for base in &bases {
        out.extend_from_slice(&base.to_le_bytes());
    }
    for (&field, &base) in fields_to_pack.iter().zip(&bases) {
        write_int4_field(&mut out, commands, field, base)?;
    }
    Ok(out)
}

fn unpack_int4_group(cursor: &mut Cursor<'_>) -> Result<Vec<MetadataCommand>> {
    let count = cursor.byte()? as usize;
    if count == 0 || count > 128 {
        return Err(invalid("packed group count"));
    }
    let mut bases = [0u32; FIELD_COUNT];
    for base in &mut bases {
        *base = cursor.u32()?;
    }
    let mut values = [[0u32; 128]; FIELD_COUNT];
    for field in 0..FIELD_COUNT {
        let decoded = read_int4_field(cursor, count, bases[field])?;
        values[field][..count].copy_from_slice(&decoded);
    }
    commands_from_fields(values, count)
}

#[derive(Clone, Copy)]
enum Selector {
    TwoBit,
    Base3,
}

fn selector_class(op: u8) -> Result<u8> {
    match op {
        0 => Ok(0),
        1 | 2 | 7 => Ok(1),
        3..=6 => Ok(2),
        _ => Err(invalid("ternary opcode")),
    }
}

fn subcode(op: u8) -> Result<u8> {
    match op {
        0 => Ok(0),
        1 => Ok(0),
        2 => Ok(1),
        7 => Ok(2),
        3..=6 => Ok(op - 3),
        _ => Err(invalid("ternary opcode")),
    }
}

fn opcode(class: u8, subcode: u8) -> Result<u8> {
    match (class, subcode) {
        (0, 0) => Ok(0),
        (1, 0) => Ok(1),
        (1, 1) => Ok(2),
        (1, 2) => Ok(7),
        (2, 0..=3) => Ok(3 + subcode),
        _ => Err(invalid("non-canonical ternary subcode")),
    }
}

fn write_selectors(
    out: &mut Vec<u8>,
    commands: &[MetadataCommand],
    selector: Selector,
) -> Result<()> {
    match selector {
        Selector::TwoBit => {
            let mut bits = BitWriter::default();
            for &command in commands {
                bits.push(selector_class(command.op)? as u32, 2)?;
            }
            out.extend_from_slice(&bits.finish());
        }
        Selector::Base3 => {
            for group in commands.chunks(5) {
                let mut value = 0u8;
                let mut factor = 1u8;
                for &command in group {
                    value = value
                        .checked_add(selector_class(command.op)?.checked_mul(factor).unwrap())
                        .ok_or(Error::SizeOverflow)?;
                    factor = factor.checked_mul(3).ok_or(Error::SizeOverflow)?;
                }
                out.push(value);
            }
        }
    }
    Ok(())
}

fn read_selectors(cursor: &mut Cursor<'_>, count: usize, selector: Selector) -> Result<Vec<u8>> {
    match selector {
        Selector::TwoBit => {
            let bytes = bits_to_bytes(count.checked_mul(2).ok_or(Error::SizeOverflow)?)?;
            let mut reader = BitReader::new(cursor.take(bytes)?);
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let class = reader.read(2)? as u8;
                if class > 2 {
                    return Err(invalid("ternary selector range"));
                }
                out.push(class);
            }
            reader.finish_zero_padded()?;
            Ok(out)
        }
        Selector::Base3 => {
            let mut out = Vec::with_capacity(count);
            for byte_index in 0..count.div_ceil(5) {
                let mut value = cursor.byte()?;
                if value >= 243 {
                    return Err(invalid("ternary selector range"));
                }
                for trit in 0..5 {
                    let class = value % 3;
                    value /= 3;
                    let index = byte_index * 5 + trit;
                    if index < count {
                        out.push(class);
                    } else if class != 0 {
                        return Err(invalid("non-canonical ternary tail"));
                    }
                }
            }
            Ok(out)
        }
    }
}

fn pack_ternary_group(commands: &[MetadataCommand], selector: Selector) -> Result<Vec<u8>> {
    if commands.is_empty() || commands.len() > 128 {
        return Err(invalid("packed group count"));
    }
    let fields_to_pack: Vec<usize> = (1..FIELD_COUNT).collect();
    let bases = bases(commands, &fields_to_pack)?;
    let mut out = Vec::new();
    out.push(commands.len() as u8);
    for base in &bases {
        out.extend_from_slice(&base.to_le_bytes());
    }
    write_selectors(&mut out, commands, selector)?;
    let mut subcodes = BitWriter::default();
    for &command in commands {
        subcodes.push(subcode(command.op)? as u32, 2)?;
    }
    out.extend_from_slice(&subcodes.finish());
    for (&field, &base) in fields_to_pack.iter().zip(&bases) {
        write_int4_field(&mut out, commands, field, base)?;
    }
    Ok(out)
}

fn unpack_ternary_group(
    cursor: &mut Cursor<'_>,
    selector: Selector,
) -> Result<Vec<MetadataCommand>> {
    let count = cursor.byte()? as usize;
    if count == 0 || count > 128 {
        return Err(invalid("packed group count"));
    }
    let mut bases = [0u32; NUMERIC_FIELD_COUNT];
    for base in &mut bases {
        *base = cursor.u32()?;
    }
    let classes = read_selectors(cursor, count, selector)?;
    let subcode_bytes = bits_to_bytes(count.checked_mul(2).ok_or(Error::SizeOverflow)?)?;
    let mut subcodes = BitReader::new(cursor.take(subcode_bytes)?);
    let mut ops = Vec::with_capacity(count);
    for class in classes {
        ops.push(opcode(class, subcodes.read(2)? as u8)?);
    }
    subcodes.finish_zero_padded()?;
    let mut values = [[0u32; 128]; FIELD_COUNT];
    for (index, op) in ops.into_iter().enumerate() {
        values[0][index] = op as u32;
    }
    for field in 1..FIELD_COUNT {
        let decoded = read_int4_field(cursor, count, bases[field - 1])?;
        values[field][..count].copy_from_slice(&decoded);
    }
    commands_from_fields(values, count)
}

fn pack_group(layout: Layout, commands: &[MetadataCommand]) -> Result<Vec<u8>> {
    match layout {
        Layout::Fixed => pack_fixed_group(commands, false),
        Layout::Bitsliced => pack_fixed_group(commands, true),
        Layout::Int4 => pack_int4_group(commands),
        Layout::Ternary2Bit => pack_ternary_group(commands, Selector::TwoBit),
        Layout::TernaryBase3 => pack_ternary_group(commands, Selector::Base3),
    }
}

fn unpack_group(cursor: &mut Cursor<'_>, layout: Layout) -> Result<Vec<MetadataCommand>> {
    match layout {
        Layout::Fixed => unpack_fixed_group(cursor, false),
        Layout::Bitsliced => unpack_fixed_group(cursor, true),
        Layout::Int4 => unpack_int4_group(cursor),
        Layout::Ternary2Bit => unpack_ternary_group(cursor, Selector::TwoBit),
        Layout::TernaryBase3 => unpack_ternary_group(cursor, Selector::Base3),
    }
}

/// Pack every non-RAW block independently. The four-byte per-block group count
/// is charged to the candidate; RAW blocks retain an empty metadata stream.
pub fn pack_blocks(
    blocks: &[Vec<MetadataCommand>],
    layout: Layout,
    group_size: usize,
) -> Result<PackedLayout> {
    if !GROUP_SIZES.contains(&group_size) {
        return Err(Error::InvalidOption("group size must be 32, 64, or 128"));
    }
    let mut packed_blocks = Vec::with_capacity(blocks.len());
    let mut metadata_bytes = 0usize;
    let mut commands = 0usize;
    let mut groups = 0usize;
    let mut max_group_bytes = 0usize;
    for commands_in_block in blocks {
        if commands_in_block.is_empty() {
            packed_blocks.push(Vec::new());
            continue;
        }
        let group_count = commands_in_block.len().div_ceil(group_size);
        let mut packed = Vec::new();
        packed.extend_from_slice(
            &u32::try_from(group_count)
                .map_err(|_| Error::SizeOverflow)?
                .to_le_bytes(),
        );
        for group in commands_in_block.chunks(group_size) {
            let encoded = pack_group(layout, group)?;
            max_group_bytes = max_group_bytes.max(encoded.len());
            packed.extend_from_slice(&encoded);
            groups += 1;
        }
        metadata_bytes = metadata_bytes
            .checked_add(packed.len())
            .ok_or(Error::SizeOverflow)?;
        commands = commands
            .checked_add(commands_in_block.len())
            .ok_or(Error::SizeOverflow)?;
        packed_blocks.push(packed);
    }
    Ok(PackedLayout {
        blocks: packed_blocks,
        commands,
        groups,
        metadata_bytes,
        max_group_bytes,
    })
}

/// Decode only packed metadata. It intentionally does not time or pretend to
/// perform literal copying/match execution; P4 must include those costs.
pub fn unpack_blocks(packed: &PackedLayout, layout: Layout) -> Result<Vec<Vec<MetadataCommand>>> {
    let mut blocks = Vec::with_capacity(packed.blocks.len());
    for bytes in &packed.blocks {
        if bytes.is_empty() {
            blocks.push(Vec::new());
            continue;
        }
        let mut cursor = Cursor::new(bytes);
        let groups = cursor.u32()? as usize;
        if groups == 0 {
            return Err(invalid("packed block group count"));
        }
        let mut commands = Vec::new();
        for _ in 0..groups {
            commands.extend(unpack_group(&mut cursor, layout)?);
        }
        if !cursor.done() {
            return Err(invalid("trailing packed metadata"));
        }
        blocks.push(commands);
    }
    Ok(blocks)
}

const DISTANCE_TAG_ESCAPE: u32 = 8;
const DISTANCE_TAG_NONE: u32 = 9;

fn entropy_distance_tags(commands: &[MetadataCommand]) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut tags = Vec::with_capacity(commands.len());
    let mut escapes = Vec::new();
    let mut cache = Vec::with_capacity(8);
    for &command in commands {
        if is_match(command) {
            if command.distance == 0 {
                return Err(invalid("zero match distance"));
            }
            if let Some(index) = cache.iter().position(|&value| value == command.distance) {
                tags.push(index as u32);
            } else {
                tags.push(DISTANCE_TAG_ESCAPE);
                escapes.push(command.distance);
            }
            mru_touch(&mut cache, command.distance, 8);
        } else {
            tags.push(DISTANCE_TAG_NONE);
        }
    }
    Ok((tags, escapes))
}

fn pack_entropy_block(commands: &[MetadataCommand]) -> Result<Vec<u8>> {
    if commands.is_empty() {
        return Err(invalid("empty entropy block"));
    }
    let mut ops = Vec::with_capacity(commands.len());
    let mut literals = Vec::with_capacity(commands.len());
    let mut runs = Vec::with_capacity(commands.len());
    let mut seed_lo = Vec::with_capacity(commands.len());
    let mut seed_hi = Vec::with_capacity(commands.len());
    for &command in commands {
        ops.push(command.op as u32);
        literals.push(command.lit);
        runs.push(command.run);
        seed_lo.push(command.seed_lo);
        seed_hi.push(command.seed_hi);
    }
    let (tags, escapes) = entropy_distance_tags(commands)?;
    let mut out = Vec::new();
    out.extend_from_slice(
        &u32::try_from(commands.len())
            .map_err(|_| Error::SizeOverflow)?
            .to_le_bytes(),
    );
    for values in [&ops, &literals, &runs, &seed_lo, &seed_hi, &tags] {
        write_huffman_field(&mut out, values)?;
    }
    out.extend_from_slice(
        &u32::try_from(escapes.len())
            .map_err(|_| Error::SizeOverflow)?
            .to_le_bytes(),
    );
    for value in escapes {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

fn unpack_entropy_block(bytes: &[u8]) -> Result<Vec<MetadataCommand>> {
    let mut cursor = Cursor::new(bytes);
    let count = cursor.u32()? as usize;
    if count == 0 {
        return Err(invalid("entropy command count"));
    }
    let ops = read_huffman_field(&mut cursor, count)?;
    let literals = read_huffman_field(&mut cursor, count)?;
    let runs = read_huffman_field(&mut cursor, count)?;
    let seed_lo = read_huffman_field(&mut cursor, count)?;
    let seed_hi = read_huffman_field(&mut cursor, count)?;
    let tags = read_huffman_field(&mut cursor, count)?;
    let escape_count = cursor.u32()? as usize;
    let mut escapes = Vec::with_capacity(escape_count);
    for _ in 0..escape_count {
        escapes.push(cursor.u32()?);
    }
    if !cursor.done() {
        return Err(invalid("trailing entropy metadata"));
    }
    let mut commands = Vec::with_capacity(count);
    let mut cache = Vec::with_capacity(8);
    let mut escape_index = 0usize;
    for index in 0..count {
        let op = u8::try_from(ops[index]).map_err(|_| invalid("entropy opcode"))?;
        if op > 7 {
            return Err(invalid("entropy opcode"));
        }
        let distance = if is_match(MetadataCommand {
            op,
            lit: 0,
            run: 0,
            distance: 0,
            seed_lo: 0,
            seed_hi: 0,
        }) {
            match tags[index] {
                0..=7 => *cache
                    .get(tags[index] as usize)
                    .ok_or_else(|| invalid("entropy distance cache index"))?,
                DISTANCE_TAG_ESCAPE => {
                    let value = *escapes
                        .get(escape_index)
                        .ok_or_else(|| invalid("entropy distance escape"))?;
                    escape_index += 1;
                    if value == 0 {
                        return Err(invalid("zero match distance"));
                    }
                    value
                }
                _ => return Err(invalid("entropy distance tag")),
            }
        } else {
            if tags[index] != DISTANCE_TAG_NONE {
                return Err(invalid("entropy non-match distance tag"));
            }
            0
        };
        if is_match(MetadataCommand {
            op,
            lit: 0,
            run: 0,
            distance: 0,
            seed_lo: 0,
            seed_hi: 0,
        }) {
            mru_touch(&mut cache, distance, 8);
        }
        commands.push(MetadataCommand {
            op,
            lit: literals[index],
            run: runs[index],
            distance,
            seed_lo: seed_lo[index],
            seed_hi: seed_hi[index],
        });
    }
    if escape_index != escapes.len() {
        return Err(invalid("trailing entropy distance escape"));
    }
    Ok(commands)
}

fn pack_entropy_blocks(blocks: &[Vec<MetadataCommand>]) -> Result<PackedEntropy> {
    let mut packed_blocks = Vec::with_capacity(blocks.len());
    let mut commands = 0usize;
    let mut metadata_bytes = 0usize;
    let mut max_block_scratch_bytes = 0usize;
    for block in blocks {
        if block.is_empty() {
            packed_blocks.push(Vec::new());
            continue;
        }
        let packed = pack_entropy_block(block)?;
        commands = commands
            .checked_add(block.len())
            .ok_or(Error::SizeOverflow)?;
        metadata_bytes = metadata_bytes
            .checked_add(packed.len())
            .ok_or(Error::SizeOverflow)?;
        max_block_scratch_bytes = max_block_scratch_bytes.max(packed.len());
        packed_blocks.push(packed);
    }
    Ok(PackedEntropy {
        blocks: packed_blocks,
        commands,
        metadata_bytes,
        max_block_scratch_bytes,
    })
}

fn unpack_entropy_blocks(packed: &PackedEntropy) -> Result<Vec<Vec<MetadataCommand>>> {
    packed
        .blocks
        .iter()
        .map(|bytes| {
            if bytes.is_empty() {
                Ok(Vec::new())
            } else {
                unpack_entropy_block(bytes)
            }
        })
        .collect()
}

fn checked_entropy_result(blocks: &[Vec<MetadataCommand>]) -> Result<EntropyResult> {
    let packed = pack_entropy_blocks(blocks)?;
    if unpack_entropy_blocks(&packed)? != blocks {
        return Err(invalid("entropy command roundtrip"));
    }
    Ok(EntropyResult {
        commands: packed.commands,
        metadata_bytes: packed.metadata_bytes,
        max_block_scratch_bytes: packed.max_block_scratch_bytes,
        pack_ns: None,
        unpack_ns: None,
    })
}

/// Measure the full scalar entropy prototype. It includes canonical table
/// construction and parsing in the timed loops, but not output execution.
pub fn time_bounded_entropy(
    blocks: &[Vec<MetadataCommand>],
    iterations: usize,
) -> Result<EntropyResult> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let verified = checked_entropy_result(blocks)?;
    let pack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= pack_entropy_blocks(blocks)?.metadata_bytes;
    }
    black_box(checksum);
    let pack_elapsed = pack_start.elapsed();
    let packed = pack_entropy_blocks(blocks)?;
    let unpack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= unpack_entropy_blocks(&packed)?
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    }
    black_box(checksum);
    Ok(EntropyResult {
        pack_ns: Some(pack_elapsed.as_nanos()),
        unpack_ns: Some(unpack_start.elapsed().as_nanos()),
        ..verified
    })
}

fn pack_fixed_128_block(commands: &[MetadataCommand]) -> Result<Vec<u8>> {
    let group_count = commands.len().div_ceil(128);
    let mut out = Vec::new();
    out.extend_from_slice(
        &u32::try_from(group_count)
            .map_err(|_| Error::SizeOverflow)?
            .to_le_bytes(),
    );
    for group in commands.chunks(128) {
        out.extend_from_slice(&pack_fixed_group(group, false)?);
    }
    Ok(out)
}

fn unpack_fixed_128_block(bytes: &[u8]) -> Result<Vec<MetadataCommand>> {
    let mut cursor = Cursor::new(bytes);
    let group_count = cursor.u32()? as usize;
    if group_count == 0 {
        return Err(invalid("adaptive fixed group count"));
    }
    let mut commands = Vec::new();
    for _ in 0..group_count {
        commands.extend(unpack_fixed_group(&mut cursor, false)?);
    }
    if !cursor.done() {
        return Err(invalid("trailing adaptive fixed metadata"));
    }
    Ok(commands)
}

fn pack_adaptive_entropy_blocks(
    blocks: &[Vec<MetadataCommand>],
) -> Result<PackedAdaptiveEntropy> {
    let mut packed_blocks = Vec::with_capacity(blocks.len());
    let mut commands = 0usize;
    let mut metadata_bytes = 0usize;
    let mut max_block_scratch_bytes = 0usize;
    let mut entropy_blocks = 0usize;
    let mut fixed_blocks = 0usize;
    for block in blocks {
        if block.is_empty() {
            packed_blocks.push(Vec::new());
            continue;
        }
        let fixed = pack_fixed_128_block(block)?;
        let entropy = pack_entropy_block(block)?;
        let selection_scratch_bytes = fixed
            .len()
            .checked_add(entropy.len())
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::SizeOverflow)?;
        let (selector, payload) = if entropy.len() < fixed.len() {
            entropy_blocks += 1;
            (1u8, entropy)
        } else {
            fixed_blocks += 1;
            (0u8, fixed)
        };
        let mut packed = Vec::with_capacity(
            payload.len().checked_add(1).ok_or(Error::SizeOverflow)?,
        );
        packed.push(selector);
        packed.extend_from_slice(&payload);
        commands = commands
            .checked_add(block.len())
            .ok_or(Error::SizeOverflow)?;
        metadata_bytes = metadata_bytes
            .checked_add(packed.len())
            .ok_or(Error::SizeOverflow)?;
        // Both alternatives were materialized during selection, so their sum
        // is a conservative scratch peak for this scalar prototype.
        max_block_scratch_bytes = max_block_scratch_bytes.max(selection_scratch_bytes);
        packed_blocks.push(packed);
    }
    Ok(PackedAdaptiveEntropy {
        blocks: packed_blocks,
        commands,
        metadata_bytes,
        max_block_scratch_bytes,
        entropy_blocks,
        fixed_blocks,
    })
}

fn unpack_adaptive_entropy_blocks(
    packed: &PackedAdaptiveEntropy,
) -> Result<Vec<Vec<MetadataCommand>>> {
    packed
        .blocks
        .iter()
        .map(|bytes| {
            if bytes.is_empty() {
                return Ok(Vec::new());
            }
            let (&selector, payload) = bytes
                .split_first()
                .ok_or_else(|| invalid("adaptive selector"))?;
            match selector {
                0 => unpack_fixed_128_block(payload),
                1 => unpack_entropy_block(payload),
                _ => Err(invalid("adaptive selector")),
            }
        })
        .collect()
}

fn checked_adaptive_entropy_result(
    blocks: &[Vec<MetadataCommand>],
) -> Result<AdaptiveEntropyResult> {
    let packed = pack_adaptive_entropy_blocks(blocks)?;
    if unpack_adaptive_entropy_blocks(&packed)? != blocks {
        return Err(invalid("adaptive entropy command roundtrip"));
    }
    Ok(AdaptiveEntropyResult {
        commands: packed.commands,
        metadata_bytes: packed.metadata_bytes,
        max_block_scratch_bytes: packed.max_block_scratch_bytes,
        entropy_blocks: packed.entropy_blocks,
        fixed_blocks: packed.fixed_blocks,
        pack_ns: None,
        unpack_ns: None,
    })
}

/// Times both alternative builders plus selection. This is deliberately more
/// expensive than the selected decoder so mode analysis cannot be hidden.
pub fn time_adaptive_entropy(
    blocks: &[Vec<MetadataCommand>],
    iterations: usize,
) -> Result<AdaptiveEntropyResult> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let verified = checked_adaptive_entropy_result(blocks)?;
    let pack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= pack_adaptive_entropy_blocks(blocks)?.metadata_bytes;
    }
    black_box(checksum);
    let pack_elapsed = pack_start.elapsed();
    let packed = pack_adaptive_entropy_blocks(blocks)?;
    let unpack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= unpack_adaptive_entropy_blocks(&packed)?
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    }
    black_box(checksum);
    Ok(AdaptiveEntropyResult {
        pack_ns: Some(pack_elapsed.as_nanos()),
        unpack_ns: Some(unpack_start.elapsed().as_nanos()),
        ..verified
    })
}

/// Build one bounded, fixed-width token dictionary. Entries are sorted by
/// token value in the serialized table, so dictionary IDs are canonical even
/// though selection starts from frequency. A token reference covers exactly
/// eight literal bytes; non-token bytes are length-prefixed raw runs.
fn pack_token_dictionary(literals: &[u8], limit: usize) -> Result<Option<(Vec<u8>, usize)>> {
    if literals.len() < TOKEN_WIDTH * 2 || !TOKEN_DICTIONARY_LIMITS.contains(&limit) {
        return Ok(None);
    }
    let mut frequencies = BTreeMap::new();
    for start in 0..=literals.len() - TOKEN_WIDTH {
        let token = u64::from_le_bytes(
            literals[start..start + TOKEN_WIDTH].try_into().unwrap(),
        );
        let count = frequencies.entry(token).or_insert(0usize);
        *count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
    }
    let mut ranked = frequencies
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_token, left_count), (right_token, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_token.cmp(right_token))
    });
    ranked.truncate(limit);
    if ranked.is_empty() {
        return Ok(None);
    }
    let mut tokens = ranked.into_iter().map(|(token, _)| token).collect::<Vec<_>>();
    tokens.sort_unstable();
    let (mut packed, mut used) = encode_token_dictionary(literals, &tokens)?;
    if used.iter().any(|&used| !used) {
        tokens = tokens
            .into_iter()
            .zip(used)
            .filter_map(|(token, used)| used.then_some(token))
            .collect();
        if tokens.is_empty() {
            return Ok(None);
        }
        (packed, used) = encode_token_dictionary(literals, &tokens)?;
    }
    if used.iter().any(|&used| !used) {
        return Err(invalid("token dictionary selection"));
    }
    Ok(Some((packed, tokens.len())))
}

fn encode_token_dictionary(literals: &[u8], tokens: &[u64]) -> Result<(Vec<u8>, Vec<bool>)> {
    if tokens.is_empty() || tokens.len() > 128 || !tokens.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(invalid("token dictionary selection"));
    }
    let mut ids = BTreeMap::new();
    for (index, &token) in tokens.iter().enumerate() {
        ids.insert(token, u8::try_from(index).map_err(|_| Error::SizeOverflow)?);
    }
    let mut packed = Vec::with_capacity(
        2usize
            .checked_add(tokens.len().checked_mul(TOKEN_WIDTH).ok_or(Error::SizeOverflow)?)
            .and_then(|value| value.checked_add(literals.len()))
            .ok_or(Error::SizeOverflow)?,
    );
    packed.push(2);
    packed.push(u8::try_from(tokens.len()).map_err(|_| Error::SizeOverflow)?);
    for token in tokens {
        packed.extend_from_slice(&token.to_le_bytes());
    }
    let mut used = vec![false; tokens.len()];
    let mut pos = 0usize;
    while pos < literals.len() {
        let token = if pos + TOKEN_WIDTH <= literals.len() {
            let value = u64::from_le_bytes(
                literals[pos..pos + TOKEN_WIDTH].try_into().unwrap(),
            );
            ids.get(&value).copied()
        } else {
            None
        };
        if let Some(id) = token {
            packed.push(0x80 | id);
            used[id as usize] = true;
            pos += TOKEN_WIDTH;
            continue;
        }
        let start = pos;
        while pos < literals.len() && pos - start < 128 {
            let token = if pos + TOKEN_WIDTH <= literals.len() {
                let value = u64::from_le_bytes(
                    literals[pos..pos + TOKEN_WIDTH].try_into().unwrap(),
                );
                ids.contains_key(&value)
            } else {
                false
            };
            if token {
                break;
            }
            pos += 1;
        }
        let count = pos - start;
        if count == 0 {
            return Err(invalid("token literal progress"));
        }
        packed.push(u8::try_from(count - 1).map_err(|_| Error::SizeOverflow)?);
        packed.extend_from_slice(&literals[start..pos]);
    }
    Ok((packed, used))
}

fn unpack_token_dictionary(cursor: &mut Cursor<'_>, literal_len: usize) -> Result<Vec<u8>> {
    let count = cursor.byte()? as usize;
    if count == 0 || count > 128 {
        return Err(invalid("token dictionary count"));
    }
    let mut tokens = Vec::with_capacity(count);
    let mut previous = None;
    for _ in 0..count {
        let token = u64::from_le_bytes(cursor.take(TOKEN_WIDTH)?.try_into().unwrap());
        if previous.is_some_and(|value| token <= value) {
            return Err(invalid("non-canonical token dictionary"));
        }
        previous = Some(token);
        tokens.push(token);
    }
    let mut used = vec![false; count];
    let mut out = Vec::with_capacity(literal_len);
    while out.len() < literal_len {
        let tag = cursor.byte()?;
        if tag & 0x80 != 0 {
            let index = (tag & 0x7f) as usize;
            let token = *tokens
                .get(index)
                .ok_or_else(|| invalid("token dictionary reference"))?;
            let end = out
                .len()
                .checked_add(TOKEN_WIDTH)
                .ok_or(Error::SizeOverflow)?;
            if end > literal_len {
                return Err(invalid("token dictionary exceeds literal length"));
            }
            used[index] = true;
            out.extend_from_slice(&token.to_le_bytes());
        } else {
            let count = (tag as usize)
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?;
            let end = out
                .len()
                .checked_add(count)
                .ok_or(Error::SizeOverflow)?;
            if end > literal_len {
                return Err(invalid("token raw run exceeds literal length"));
            }
            out.extend_from_slice(cursor.take(count)?);
        }
    }
    if used.iter().any(|&used| !used) {
        return Err(invalid("unused token dictionary entry"));
    }
    if !cursor.done() {
        return Err(invalid("token literal trailing bytes"));
    }
    Ok(out)
}

fn pack_literal_entropy_blocks(blocks: &[Vec<u8>]) -> Result<PackedLiteralEntropy> {
    let mut packed_blocks = Vec::with_capacity(blocks.len());
    let mut literal_lengths = Vec::with_capacity(blocks.len());
    let mut literal_bytes = 0usize;
    let mut encoded_bytes = 0usize;
    let mut max_block_scratch_bytes = 0usize;
    let mut dictionary_entries = 0usize;
    let mut entropy_blocks = 0usize;
    let mut dictionary_blocks = 0usize;
    let mut raw_blocks = 0usize;
    for literals in blocks {
        let mut raw = Vec::with_capacity(
            literals
                .len()
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?,
        );
        raw.push(0);
        raw.extend_from_slice(literals);
        let entropy = if literals.is_empty() {
            None
        } else {
            let values = literals.iter().map(|&value| value as u32).collect::<Vec<_>>();
            let mut packed = Vec::new();
            packed.push(1);
            write_huffman_field(&mut packed, &values)?;
            Some(packed)
        };
        let mut candidates = vec![(0u8, raw, 0usize)];
        if let Some(entropy) = entropy {
            candidates.push((1, entropy, 0));
        }
        for limit in TOKEN_DICTIONARY_LIMITS {
            if let Some((dictionary, entries)) = pack_token_dictionary(literals, limit)? {
                candidates.push((2, dictionary, entries));
            }
        }
        let selection_scratch_bytes = candidates.iter().try_fold(0usize, |total, candidate| {
            total
                .checked_add(candidate.1.len())
                .ok_or(Error::SizeOverflow)
        })?;
        let mut candidates = candidates.into_iter();
        let mut chosen = candidates
            .next()
            .ok_or_else(|| invalid("literal candidate"))?;
        for candidate in candidates {
            if candidate.1.len() < chosen.1.len() {
                chosen = candidate;
            }
        }
        match chosen.0 {
            0 => raw_blocks += 1,
            1 => entropy_blocks += 1,
            2 => {
                dictionary_blocks += 1;
                dictionary_entries = dictionary_entries
                    .checked_add(chosen.2)
                    .ok_or(Error::SizeOverflow)?;
            }
            _ => return Err(invalid("literal candidate")),
        }
        literal_bytes = literal_bytes
            .checked_add(literals.len())
            .ok_or(Error::SizeOverflow)?;
        encoded_bytes = encoded_bytes
            .checked_add(chosen.1.len())
            .ok_or(Error::SizeOverflow)?;
        max_block_scratch_bytes = max_block_scratch_bytes.max(selection_scratch_bytes);
        literal_lengths.push(literals.len());
        packed_blocks.push(chosen.1);
    }
    Ok(PackedLiteralEntropy {
        blocks: packed_blocks,
        literal_lengths,
        literal_bytes,
        encoded_bytes,
        max_block_scratch_bytes,
        dictionary_entries,
        entropy_blocks,
        dictionary_blocks,
        raw_blocks,
    })
}

fn unpack_literal_entropy_blocks(packed: &PackedLiteralEntropy) -> Result<Vec<Vec<u8>>> {
    if packed.blocks.len() != packed.literal_lengths.len() {
        return Err(invalid("literal block lengths"));
    }
    packed
        .blocks
        .iter()
        .zip(&packed.literal_lengths)
        .map(|(bytes, &literal_len)| {
            let mut cursor = Cursor::new(bytes);
            let selector = cursor.byte()?;
            match selector {
                0 => {
                    let out = cursor.take(literal_len)?.to_vec();
                    if !cursor.done() {
                        return Err(invalid("literal raw trailing bytes"));
                    }
                    Ok(out)
                }
                1 => {
                    if literal_len == 0 {
                        return Err(invalid("literal empty entropy block"));
                    }
                    let values = read_huffman_field(&mut cursor, literal_len)?;
                    if !cursor.done() {
                        return Err(invalid("literal entropy trailing bytes"));
                    }
                    values
                        .into_iter()
                        .map(|value| {
                            u8::try_from(value)
                                .map_err(|_| invalid("literal entropy byte range"))
                        })
                        .collect()
                }
                2 => unpack_token_dictionary(&mut cursor, literal_len),
                _ => Err(invalid("literal selector")),
            }
        })
        .collect()
}

fn checked_literal_entropy_result(blocks: &[Vec<u8>]) -> Result<LiteralEntropyResult> {
    let packed = pack_literal_entropy_blocks(blocks)?;
    if unpack_literal_entropy_blocks(&packed)? != blocks {
        return Err(invalid("literal entropy roundtrip"));
    }
    Ok(LiteralEntropyResult {
        literal_bytes: packed.literal_bytes,
        encoded_bytes: packed.encoded_bytes,
        max_block_scratch_bytes: packed.max_block_scratch_bytes,
        dictionary_entries: packed.dictionary_entries,
        entropy_blocks: packed.entropy_blocks,
        dictionary_blocks: packed.dictionary_blocks,
        raw_blocks: packed.raw_blocks,
        pack_ns: None,
        unpack_ns: None,
    })
}

/// Times raw-versus-Huffman literal selection plus exact literal reconstruction.
/// The block's decoded literal length comes from the already validated command
/// stream, just as it would in a joined command/literal format.
pub fn time_literal_entropy(
    blocks: &[Vec<u8>],
    iterations: usize,
) -> Result<LiteralEntropyResult> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let verified = checked_literal_entropy_result(blocks)?;
    let pack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= pack_literal_entropy_blocks(blocks)?.encoded_bytes;
    }
    black_box(checksum);
    let pack_elapsed = pack_start.elapsed();
    let packed = pack_literal_entropy_blocks(blocks)?;
    let unpack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= unpack_literal_entropy_blocks(&packed)?
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    }
    black_box(checksum);
    Ok(LiteralEntropyResult {
        pack_ns: Some(pack_elapsed.as_nanos()),
        unpack_ns: Some(unpack_start.elapsed().as_nanos()),
        ..verified
    })
}

/// Frame convenience wrapper for the literal prototype timing. RAW blocks are
/// deliberately excluded because they remain raw passthrough payloads in the
/// complete-artifact accounting.
pub fn time_frame_literal_entropy(
    frame: &Frame<'_>,
    iterations: usize,
) -> Result<LiteralEntropyResult> {
    let blocks = literal_blocks(frame)?;
    let lz_blocks = frame
        .blocks
        .iter()
        .zip(blocks)
        .filter_map(|(block_info, literals)| {
            (block_info.info.kind != block::RAW).then_some(literals)
        })
        .collect::<Vec<_>>();
    time_literal_entropy(&lz_blocks, iterations)
}

fn checked_result(
    blocks: &[Vec<MetadataCommand>],
    layout: Layout,
    group_size: usize,
) -> Result<LayoutResult> {
    let packed = pack_blocks(blocks, layout, group_size)?;
    let decoded = unpack_blocks(&packed, layout)?;
    if decoded != blocks {
        return Err(invalid("packed command roundtrip"));
    }
    Ok(LayoutResult {
        layout,
        group_size,
        commands: packed.commands,
        groups: packed.groups,
        metadata_bytes: packed.metadata_bytes,
        max_group_bytes: packed.max_group_bytes,
        pack_ns: None,
        unpack_ns: None,
    })
}

/// Execute every exact P3 representation once, after which each result has
/// passed a packed-metadata roundtrip against the same command trace.
pub fn evaluate_frame(frame: &Frame<'_>) -> Result<Evaluation> {
    let blocks = command_blocks(frame)?;
    let extracted_literals = literal_blocks(frame)?;
    if extracted_literals.len() != blocks.len() {
        return Err(invalid("literal block count"));
    }
    let mut literal_bytes = 0usize;
    let mut lz_literal_blocks = Vec::new();
    let mut current_v2_metadata_bytes = 0usize;
    let mut raw_passthrough_bytes = 0usize;
    let mut lz_blocks = 0usize;
    let mut raw_blocks = 0usize;
    for (index, (block_info, commands)) in frame.blocks.iter().zip(&blocks).enumerate() {
        if block_info.info.kind == block::RAW {
            raw_blocks += 1;
            raw_passthrough_bytes = raw_passthrough_bytes
                .checked_add(block_info.payload.len())
                .ok_or(Error::SizeOverflow)?;
            continue;
        }
        lz_blocks += 1;
        let block_literals = commands.iter().try_fold(0usize, |total, command| {
            total
                .checked_add(command.lit as usize)
                .ok_or(Error::SizeOverflow)
        })?;
        literal_bytes = literal_bytes
            .checked_add(block_literals)
            .ok_or(Error::SizeOverflow)?;
        if extracted_literals[index].len() != block_literals {
            return Err(invalid("literal command accounting"));
        }
        lz_literal_blocks.push(extracted_literals[index].clone());
        let metadata = block_info
            .payload
            .len()
            .checked_sub(block::BLOCK_PREFIX)
            .and_then(|value| value.checked_sub(block_literals))
            .ok_or_else(|| invalid("v2 metadata accounting"))?;
        current_v2_metadata_bytes = current_v2_metadata_bytes
            .checked_add(metadata)
            .ok_or(Error::SizeOverflow)?;
    }
    let mut results = Vec::new();
    for layout in Layout::ALL {
        for group_size in GROUP_SIZES {
            results.push(checked_result(&blocks, layout, group_size)?);
        }
    }
    Ok(Evaluation {
        input_bytes: frame.raw_size(),
        frame_bytes: frame::HEADER_SIZE
            .checked_add(
                frame
                    .blocks
                    .len()
                    .checked_mul(frame::ENTRY_SIZE)
                    .ok_or(Error::SizeOverflow)?,
            )
            .and_then(|value| {
                value
                    .checked_add(raw_passthrough_bytes)?
                    .checked_add(literal_bytes)
            })
            .and_then(|value| value.checked_add(current_v2_metadata_bytes))
            .and_then(|value| value.checked_add(lz_blocks.checked_mul(block::BLOCK_PREFIX)?))
            .ok_or(Error::SizeOverflow)?,
        common_frame_envelope_bytes: frame::HEADER_SIZE
            .checked_add(
                frame
                    .blocks
                    .len()
                    .checked_mul(frame::ENTRY_SIZE)
                    .ok_or(Error::SizeOverflow)?,
            )
            .ok_or(Error::SizeOverflow)?,
        raw_passthrough_bytes,
        literal_bytes,
        current_v2_metadata_bytes,
        lz_blocks,
        raw_blocks,
        results,
        distance_results: checked_distance_results(&blocks)?,
        bounded_entropy: checked_entropy_result(&blocks)?,
        adaptive_entropy: checked_adaptive_entropy_result(&blocks)?,
        literal_entropy: checked_literal_entropy_result(&lz_literal_blocks)?,
    })
}

/// Measure scalar prototype packing/unpacking with no preparation hidden in
/// either loop. These are metadata bytes/s and commands/s—not full-codec GB/s.
pub fn time_layouts(
    blocks: &[Vec<MetadataCommand>],
    iterations: usize,
) -> Result<Vec<LayoutResult>> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let mut results = Vec::new();
    for layout in Layout::ALL {
        for group_size in GROUP_SIZES {
            let verified = checked_result(blocks, layout, group_size)?;
            let pack_start = Instant::now();
            let mut checksum = 0usize;
            for _ in 0..iterations {
                let packed = pack_blocks(blocks, layout, group_size)?;
                checksum ^= packed.metadata_bytes;
            }
            black_box(checksum);
            let pack_elapsed = pack_start.elapsed();
            let packed = pack_blocks(blocks, layout, group_size)?;
            let unpack_start = Instant::now();
            let mut decoded_commands = 0usize;
            for _ in 0..iterations {
                let decoded = unpack_blocks(&packed, layout)?;
                decoded_commands ^= decoded.iter().map(Vec::len).sum::<usize>();
            }
            black_box(decoded_commands);
            let unpack_elapsed = unpack_start.elapsed();
            results.push(LayoutResult {
                pack_ns: Some(pack_elapsed.as_nanos()),
                unpack_ns: Some(unpack_elapsed.as_nanos()),
                ..verified
            });
        }
    }
    Ok(results)
}

fn is_match(command: MetadataCommand) -> bool {
    matches!(command.op, 1 | 2 | 7)
}

fn distance_values(commands: &[MetadataCommand]) -> Result<Vec<u32>> {
    let mut values = Vec::new();
    for &command in commands {
        if is_match(command) {
            if command.distance == 0 {
                return Err(invalid("zero match distance"));
            }
            values.push(command.distance);
        }
    }
    Ok(values)
}

fn write_uleb(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 127) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 128;
        }
        out.push(byte);
        if value == 0 {
            return;
        }
    }
}

fn uleb_len(mut value: u64) -> usize {
    let mut bytes = 1;
    while value >= 128 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

fn read_uleb(cursor: &mut Cursor<'_>) -> Result<u64> {
    let mut value = 0u64;
    for index in 0..10usize {
        let byte = cursor.byte()?;
        let payload = (byte & 127) as u64;
        if index == 9 && payload > 1 {
            return Err(invalid("uleb overflow"));
        }
        value |= payload << (index * 7);
        if byte & 128 == 0 {
            if uleb_len(value) != index + 1 {
                return Err(invalid("non-canonical uleb"));
            }
            return Ok(value);
        }
    }
    Err(invalid("uleb overflow"))
}

fn write_absolute_distances(out: &mut Vec<u8>, values: &[u32]) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for &value in values {
        if value < u16::MAX as u32 {
            out.extend_from_slice(&(value as u16).to_le_bytes());
        } else {
            out.extend_from_slice(&u16::MAX.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

fn read_absolute_distances(cursor: &mut Cursor<'_>) -> Result<Vec<u32>> {
    let count = cursor.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let code = u16::from_le_bytes(cursor.take(2)?.try_into().unwrap()) as u32;
        let value = if code == u16::MAX as u32 {
            let escaped = cursor.u32()?;
            if escaped < u16::MAX as u32 {
                return Err(invalid("non-canonical absolute distance"));
            }
            escaped
        } else {
            code
        };
        if value == 0 {
            return Err(invalid("zero match distance"));
        }
        out.push(value);
    }
    Ok(out)
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn write_delta_distances(out: &mut Vec<u8>, values: &[u32]) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    let mut previous = 0i64;
    for &value in values {
        write_uleb(out, zigzag(value as i64 - previous));
        previous = value as i64;
    }
}

fn read_delta_distances(cursor: &mut Cursor<'_>) -> Result<Vec<u32>> {
    let count = cursor.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    let mut previous = 0i64;
    for _ in 0..count {
        let delta = unzigzag(read_uleb(cursor)?);
        let value = previous.checked_add(delta).ok_or(Error::SizeOverflow)?;
        if !(1..=u32::MAX as i64).contains(&value) {
            return Err(invalid("delta distance range"));
        }
        out.push(value as u32);
        previous = value;
    }
    Ok(out)
}

fn write_xor_distances(out: &mut Vec<u8>, values: &[u32]) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    let mut previous = 0;
    for &value in values {
        write_uleb(out, (value ^ previous) as u64);
        previous = value;
    }
}

fn read_xor_distances(cursor: &mut Cursor<'_>) -> Result<Vec<u32>> {
    let count = cursor.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    let mut previous = 0;
    for _ in 0..count {
        let residual = u32::try_from(read_uleb(cursor)?).map_err(|_| invalid("xor distance range"))?;
        let value = residual ^ previous;
        if value == 0 {
            return Err(invalid("zero match distance"));
        }
        out.push(value);
        previous = value;
    }
    Ok(out)
}

fn mru_touch(cache: &mut Vec<u32>, value: u32, capacity: usize) {
    if let Some(index) = cache.iter().position(|&candidate| candidate == value) {
        cache.remove(index);
    }
    cache.insert(0, value);
    cache.truncate(capacity);
}

fn write_cached_distances(out: &mut Vec<u8>, values: &[u32], capacity: usize) -> Result<()> {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    let mut cache = Vec::with_capacity(capacity);
    for &value in values {
        if let Some(index) = cache.iter().position(|&candidate| candidate == value) {
            out.push(index as u8);
        } else {
            out.push(u8::MAX);
            write_uleb(out, value as u64);
        }
        mru_touch(&mut cache, value, capacity);
    }
    Ok(())
}

fn read_cached_distances(cursor: &mut Cursor<'_>, capacity: usize) -> Result<Vec<u32>> {
    let count = cursor.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    let mut cache = Vec::with_capacity(capacity);
    for _ in 0..count {
        let tag = cursor.byte()? as usize;
        let value = if tag == u8::MAX as usize {
            u32::try_from(read_uleb(cursor)?).map_err(|_| invalid("cached distance range"))?
        } else {
            *cache.get(tag).ok_or_else(|| invalid("cached distance index"))?
        };
        if value == 0 {
            return Err(invalid("zero match distance"));
        }
        mru_touch(&mut cache, value, capacity);
        out.push(value);
    }
    Ok(out)
}

fn dictionary(values: &[u32]) -> Vec<u32> {
    let mut counts = BTreeMap::new();
    for &value in values {
        *counts.entry(value).or_insert(0usize) += 1;
    }
    let mut ranked: Vec<_> = counts.into_iter().collect();
    ranked.sort_by(|(left_value, left_count), (right_value, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_value.cmp(right_value))
    });
    ranked.into_iter().take(16).map(|(value, _)| value).collect()
}

fn write_dictionary_distances(out: &mut Vec<u8>, commands: &[MetadataCommand], group_size: usize) -> Result<usize> {
    let group_count = commands.len().div_ceil(group_size);
    out.extend_from_slice(&(group_count as u32).to_le_bytes());
    let mut maximum = 0usize;
    for group in commands.chunks(group_size) {
        let values = distance_values(group)?;
        let entries = dictionary(&values);
        out.push(u8::try_from(values.len()).map_err(|_| Error::SizeOverflow)?);
        out.push(entries.len() as u8);
        for &entry in &entries {
            out.extend_from_slice(&entry.to_le_bytes());
        }
        let group_start = out.len();
        for &value in &values {
            if let Some(index) = entries.iter().position(|&entry| entry == value) {
                out.push(index as u8);
            } else {
                out.push(u8::MAX);
                write_uleb(out, value as u64);
            }
        }
        maximum = maximum.max(out.len() - group_start + entries.len() * 4 + 2);
    }
    Ok(maximum)
}

fn read_dictionary_distances(cursor: &mut Cursor<'_>) -> Result<(Vec<u32>, usize)> {
    let group_count = cursor.u32()? as usize;
    let mut values = Vec::new();
    let mut maximum = 0usize;
    for _ in 0..group_count {
        let count = cursor.byte()? as usize;
        if count > 128 {
            return Err(invalid("dictionary group count"));
        }
        let entry_count = cursor.byte()? as usize;
        if entry_count > 16 {
            return Err(invalid("dictionary size"));
        }
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let entry = cursor.u32()?;
            if entry == 0 || entries.contains(&entry) {
                return Err(invalid("dictionary entry"));
            }
            entries.push(entry);
        }
        let group_start = cursor.pos;
        let mut group_values = Vec::with_capacity(count);
        for _ in 0..count {
            let tag = cursor.byte()? as usize;
            let value = if tag == u8::MAX as usize {
                u32::try_from(read_uleb(cursor)?).map_err(|_| invalid("dictionary distance range"))?
            } else {
                *entries.get(tag).ok_or_else(|| invalid("dictionary index"))?
            };
            if value == 0 {
                return Err(invalid("zero match distance"));
            }
            group_values.push(value);
        }
        if dictionary(&group_values) != entries {
            return Err(invalid("non-canonical dictionary"));
        }
        maximum = maximum.max(cursor.pos - group_start + entries.len() * 4 + 2);
        values.extend(group_values);
    }
    Ok((values, maximum))
}

fn pack_distance_blocks(
    blocks: &[Vec<MetadataCommand>],
    model: DistanceModel,
    group_size: usize,
) -> Result<PackedDistances> {
    let mut packed_blocks = Vec::with_capacity(blocks.len());
    let mut distances = 0usize;
    let mut metadata_bytes = 0usize;
    let mut max_scratch_bytes = 0usize;
    for commands in blocks {
        if commands.is_empty() {
            packed_blocks.push(Vec::new());
            continue;
        }
        let mut bytes = Vec::new();
        let block_distances = distance_values(commands)?;
        match model {
            DistanceModel::Absolute => write_absolute_distances(&mut bytes, &block_distances),
            DistanceModel::Delta => write_delta_distances(&mut bytes, &block_distances),
            DistanceModel::Xor => write_xor_distances(&mut bytes, &block_distances),
            DistanceModel::Cache(capacity) => {
                write_cached_distances(&mut bytes, &block_distances, capacity)?
            }
            DistanceModel::Dictionary16 => {
                max_scratch_bytes = max_scratch_bytes.max(write_dictionary_distances(
                    &mut bytes,
                    commands,
                    group_size,
                )?);
            }
        }
        if !matches!(model, DistanceModel::Dictionary16) {
            max_scratch_bytes = max_scratch_bytes.max(bytes.len());
        }
        distances = distances
            .checked_add(block_distances.len())
            .ok_or(Error::SizeOverflow)?;
        metadata_bytes = metadata_bytes
            .checked_add(bytes.len())
            .ok_or(Error::SizeOverflow)?;
        packed_blocks.push(bytes);
    }
    Ok(PackedDistances {
        blocks: packed_blocks,
        distances,
        metadata_bytes,
        max_scratch_bytes,
    })
}

fn unpack_distance_blocks(packed: &PackedDistances, model: DistanceModel) -> Result<Vec<Vec<u32>>> {
    let mut blocks = Vec::with_capacity(packed.blocks.len());
    for bytes in &packed.blocks {
        if bytes.is_empty() {
            blocks.push(Vec::new());
            continue;
        }
        let mut cursor = Cursor::new(bytes);
        let values = match model {
            DistanceModel::Absolute => read_absolute_distances(&mut cursor)?,
            DistanceModel::Delta => read_delta_distances(&mut cursor)?,
            DistanceModel::Xor => read_xor_distances(&mut cursor)?,
            DistanceModel::Cache(capacity) => read_cached_distances(&mut cursor, capacity)?,
            DistanceModel::Dictionary16 => read_dictionary_distances(&mut cursor)?.0,
        };
        if !cursor.done() {
            return Err(invalid("trailing packed distances"));
        }
        blocks.push(values);
    }
    Ok(blocks)
}

fn expected_distances(blocks: &[Vec<MetadataCommand>]) -> Result<Vec<Vec<u32>>> {
    blocks.iter().map(|commands| distance_values(commands)).collect()
}

fn checked_distance_result(
    blocks: &[Vec<MetadataCommand>],
    model: DistanceModel,
    group_size: usize,
) -> Result<DistanceResult> {
    let packed = pack_distance_blocks(blocks, model, group_size)?;
    if unpack_distance_blocks(&packed, model)? != expected_distances(blocks)? {
        return Err(invalid("packed distance roundtrip"));
    }
    Ok(DistanceResult {
        model: model.name(),
        group_size: matches!(model, DistanceModel::Dictionary16).then_some(group_size),
        distances: packed.distances,
        metadata_bytes: packed.metadata_bytes,
        max_scratch_bytes: packed.max_scratch_bytes,
        pack_ns: None,
        unpack_ns: None,
    })
}

fn checked_distance_results(blocks: &[Vec<MetadataCommand>]) -> Result<Vec<DistanceResult>> {
    let mut results = Vec::new();
    for model in DistanceModel::ALL {
        results.push(checked_distance_result(blocks, model, 32)?);
    }
    for group_size in GROUP_SIZES {
        results.push(checked_distance_result(
            blocks,
            DistanceModel::Dictionary16,
            group_size,
        )?);
    }
    Ok(results)
}

/// Timed counterpart of the exact distance tests. The denominator is only the
/// packed distance stream and never represents whole-codec bandwidth.
pub fn time_distance_models(
    blocks: &[Vec<MetadataCommand>],
    iterations: usize,
) -> Result<Vec<DistanceResult>> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let mut results = checked_distance_results(blocks)?;
    for result in &mut results {
        let model = match result.model {
            "absolute-u16-escape-u32" => DistanceModel::Absolute,
            "delta-zigzag-uleb" => DistanceModel::Delta,
            "xor-uleb" => DistanceModel::Xor,
            "mru-cache-2" => DistanceModel::Cache(2),
            "mru-cache-4" => DistanceModel::Cache(4),
            "mru-cache-8" => DistanceModel::Cache(8),
            "group-dictionary-16" => DistanceModel::Dictionary16,
            _ => return Err(invalid("distance timing model")),
        };
        let group_size = result.group_size.unwrap_or(32);
        let pack_start = Instant::now();
        let mut checksum = 0usize;
        for _ in 0..iterations {
            checksum ^= pack_distance_blocks(blocks, model, group_size)?.metadata_bytes;
        }
        black_box(checksum);
        let pack_elapsed = pack_start.elapsed();
        let packed = pack_distance_blocks(blocks, model, group_size)?;
        let unpack_start = Instant::now();
        let mut checksum = 0usize;
        for _ in 0..iterations {
            checksum ^= unpack_distance_blocks(&packed, model)?
                .iter()
                .map(Vec::len)
                .sum::<usize>();
        }
        black_box(checksum);
        result.pack_ns = Some(pack_elapsed.as_nanos());
        result.unpack_ns = Some(unpack_start.elapsed().as_nanos());
    }
    Ok(results)
}

impl Evaluation {
    pub fn apply_timings(&mut self, timings: &[LayoutResult]) -> Result<()> {
        for result in &mut self.results {
            let timing = timings
                .iter()
                .find(|candidate| {
                    candidate.layout == result.layout && candidate.group_size == result.group_size
                })
                .ok_or_else(|| invalid("missing packing timing"))?;
            result.pack_ns = timing.pack_ns;
            result.unpack_ns = timing.unpack_ns;
        }
        Ok(())
    }

    pub fn apply_distance_timings(&mut self, timings: &[DistanceResult]) -> Result<()> {
        for result in &mut self.distance_results {
            let timing = timings
                .iter()
                .find(|candidate| {
                    candidate.model == result.model && candidate.group_size == result.group_size
                })
                .ok_or_else(|| invalid("missing distance timing"))?;
            result.pack_ns = timing.pack_ns;
            result.unpack_ns = timing.unpack_ns;
        }
        Ok(())
    }

    pub fn apply_entropy_timing(&mut self, timing: &EntropyResult) {
        self.bounded_entropy.pack_ns = timing.pack_ns;
        self.bounded_entropy.unpack_ns = timing.unpack_ns;
    }

    pub fn apply_adaptive_entropy_timing(&mut self, timing: &AdaptiveEntropyResult) {
        self.adaptive_entropy.pack_ns = timing.pack_ns;
        self.adaptive_entropy.unpack_ns = timing.unpack_ns;
    }

    pub fn apply_literal_entropy_timing(&mut self, timing: &LiteralEntropyResult) {
        self.literal_entropy.pack_ns = timing.pack_ns;
        self.literal_entropy.unpack_ns = timing.unpack_ns;
    }

    /// Stable JSON with every denominator labelled. Candidate totals charge the
    /// retained literals, raw blocks and existing frame directory, but are
    /// explicitly prototype-container estimates rather than a v3 bitstream.
    pub fn to_json(&self, iterations: Option<usize>) -> String {
        let mut out = String::new();
        writeln!(out, "{{").unwrap();
        writeln!(out, "  \"schema\": 1,").unwrap();
        writeln!(out, "  \"scope\": \"P3 scalar lossless command-metadata prototypes; not a VXG v3 wire format and not a complete codec benchmark.\",").unwrap();
        writeln!(out, "  \"bit_order\": \"LSB-first within contiguous packed fields and planes; block and group state reset at every v2 independent block.\",").unwrap();
        writeln!(out, "  \"fixed_width_rule\": \"Per field: base=min; choose width 0..max residual width minimizing packed code bytes plus 5-byte indexed exceptions. Width zero has no code bits; exceptions are ordered (u8 index,u32 residual).\",").unwrap();
        writeln!(out, "  \"int4_rule\": \"Two nibbles per byte, direct residuals 0..14, 15 is an exact (u8 index,u32 residual) escape; an odd tail's high nibble must be zero.\",").unwrap();
        writeln!(out, "  \"ternary_rule\": \"Classes: LIT={{0}}; MATCH={{1,2,7}}; PATTERN={{3,4,5,6}}. Base-3 stores five trits per byte and rejects 243..255; 2-bit and base-3 variants retain exact opcode subcodes and all numeric payloads.\",").unwrap();
        writeln!(out, "  \"input_bytes\": {},", self.input_bytes).unwrap();
        writeln!(out, "  \"current_frame_bytes\": {},", self.frame_bytes).unwrap();
        writeln!(
            out,
            "  \"common_frame_envelope_bytes\": {},",
            self.common_frame_envelope_bytes
        )
        .unwrap();
        writeln!(
            out,
            "  \"raw_passthrough_bytes\": {},",
            self.raw_passthrough_bytes
        )
        .unwrap();
        writeln!(out, "  \"retained_literal_bytes\": {},", self.literal_bytes).unwrap();
        writeln!(
            out,
            "  \"current_v2_metadata_bytes_excluding_block_prefix_and_literals\": {},",
            self.current_v2_metadata_bytes
        )
        .unwrap();
        writeln!(out, "  \"lz_blocks\": {},", self.lz_blocks).unwrap();
        writeln!(out, "  \"raw_blocks\": {},", self.raw_blocks).unwrap();
        writeln!(
            out,
            "  \"timing_iterations\": {},",
            iterations.map_or("null".to_owned(), |value| value.to_string())
        )
        .unwrap();
        writeln!(out, "  \"results\": [").unwrap();
        for (index, result) in self.results.iter().enumerate() {
            let prototype_total = self
                .common_frame_envelope_bytes
                .checked_add(self.raw_passthrough_bytes)
                .and_then(|value| value.checked_add(self.literal_bytes))
                .and_then(|value| value.checked_add(result.metadata_bytes))
                .unwrap();
            let pack_commands_per_second = result
                .pack_ns
                .filter(|&ns| ns != 0)
                .map(|ns| result.commands as f64 * 1_000_000_000.0 / ns as f64);
            let unpack_commands_per_second = result
                .unpack_ns
                .filter(|&ns| ns != 0)
                .map(|ns| result.commands as f64 * 1_000_000_000.0 / ns as f64);
            let pack_metadata_bytes_per_second = result
                .pack_ns
                .filter(|&ns| ns != 0)
                .map(|ns| result.metadata_bytes as f64 * 1_000_000_000.0 / ns as f64);
            let unpack_metadata_bytes_per_second = result
                .unpack_ns
                .filter(|&ns| ns != 0)
                .map(|ns| result.metadata_bytes as f64 * 1_000_000_000.0 / ns as f64);
            writeln!(out, "    {{").unwrap();
            writeln!(out, "      \"layout\": \"{}\",", result.layout.name()).unwrap();
            writeln!(out, "      \"group_size\": {},", result.group_size).unwrap();
            writeln!(out, "      \"commands\": {},", result.commands).unwrap();
            writeln!(out, "      \"groups\": {},", result.groups).unwrap();
            writeln!(out, "      \"metadata_bytes\": {},", result.metadata_bytes).unwrap();
            writeln!(
                out,
                "      \"metadata_bytes_per_command\": {:.6},",
                result.metadata_bytes as f64 / result.commands.max(1) as f64
            )
            .unwrap();
            writeln!(
                out,
                "      \"max_group_scratch_bytes\": {},",
                result.max_group_bytes
            )
            .unwrap();
            writeln!(
                out,
                "      \"prototype_total_frame_bytes\": {},",
                prototype_total
            )
            .unwrap();
            writeln!(
                out,
                "      \"prototype_total_ratio\": {:.9},",
                prototype_total as f64 / self.input_bytes.max(1) as f64
            )
            .unwrap();
            writeln!(
                out,
                "      \"pack_elapsed_ns\": {},",
                result
                    .pack_ns
                    .map_or("null".to_owned(), |value| value.to_string())
            )
            .unwrap();
            writeln!(
                out,
                "      \"unpack_elapsed_ns\": {},",
                result
                    .unpack_ns
                    .map_or("null".to_owned(), |value| value.to_string())
            )
            .unwrap();
            writeln!(
                out,
                "      \"pack_commands_per_second\": {},",
                pack_commands_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))
            )
            .unwrap();
            writeln!(
                out,
                "      \"unpack_commands_per_second\": {},",
                unpack_commands_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))
            )
            .unwrap();
            writeln!(
                out,
                "      \"pack_metadata_bytes_per_second\": {},",
                pack_metadata_bytes_per_second
                    .map_or("null".to_owned(), |value| format!("{value:.3}"))
            )
            .unwrap();
            writeln!(
                out,
                "      \"unpack_metadata_bytes_per_second\": {}",
                unpack_metadata_bytes_per_second
                    .map_or("null".to_owned(), |value| format!("{value:.3}"))
            )
            .unwrap();
            writeln!(
                out,
                "    }}{}",
                if index + 1 == self.results.len() {
                    ""
                } else {
                    ","
                }
            )
            .unwrap();
        }
        writeln!(out, "  ],").unwrap();
        writeln!(out, "  \"distance_models\": [").unwrap();
        for (index, result) in self.distance_results.iter().enumerate() {
            let pack_values_per_second = result.pack_ns.filter(|&ns| ns != 0).map(|ns| {
                result.distances as f64 * 1_000_000_000.0 / ns as f64
            });
            let unpack_values_per_second = result.unpack_ns.filter(|&ns| ns != 0).map(|ns| {
                result.distances as f64 * 1_000_000_000.0 / ns as f64
            });
            writeln!(out, "    {{").unwrap();
            writeln!(out, "      \"model\": \"{}\",", result.model).unwrap();
            writeln!(out, "      \"group_size\": {},", result.group_size.map_or("null".to_owned(), |value| value.to_string())).unwrap();
            writeln!(out, "      \"distances\": {},", result.distances).unwrap();
            writeln!(out, "      \"metadata_bytes\": {},", result.metadata_bytes).unwrap();
            writeln!(out, "      \"metadata_bytes_per_distance\": {:.6},", result.metadata_bytes as f64 / result.distances.max(1) as f64).unwrap();
            writeln!(out, "      \"max_scratch_bytes\": {},", result.max_scratch_bytes).unwrap();
            writeln!(out, "      \"pack_elapsed_ns\": {},", result.pack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
            writeln!(out, "      \"unpack_elapsed_ns\": {},", result.unpack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
            writeln!(out, "      \"pack_distances_per_second\": {},", pack_values_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
            writeln!(out, "      \"unpack_distances_per_second\": {}", unpack_values_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
            writeln!(out, "    }}{}", if index + 1 == self.distance_results.len() { "" } else { "," }).unwrap();
        }
        writeln!(out, "  ],").unwrap();
        let entropy_total = self
            .common_frame_envelope_bytes
            .checked_add(self.raw_passthrough_bytes)
            .and_then(|value| value.checked_add(self.literal_bytes))
            .and_then(|value| value.checked_add(self.bounded_entropy.metadata_bytes))
            .unwrap();
        let entropy_pack_commands_per_second = self
            .bounded_entropy
            .pack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.bounded_entropy.commands as f64 * 1_000_000_000.0 / ns as f64);
        let entropy_unpack_commands_per_second = self
            .bounded_entropy
            .unpack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.bounded_entropy.commands as f64 * 1_000_000_000.0 / ns as f64);
        writeln!(out, "  \"bounded_entropy\": {{").unwrap();
        writeln!(out, "    \"scope\": \"Per-v2-block canonical Huffman tables for op/literal/run/seed/distance-tag fields plus MRU-8 distance escapes. All table, bit-length and u32 escape bytes are charged; output execution is not timed.\",").unwrap();
        writeln!(out, "    \"commands\": {},", self.bounded_entropy.commands).unwrap();
        writeln!(out, "    \"metadata_bytes\": {},", self.bounded_entropy.metadata_bytes).unwrap();
        writeln!(out, "    \"metadata_bytes_per_command\": {:.6},", self.bounded_entropy.metadata_bytes as f64 / self.bounded_entropy.commands.max(1) as f64).unwrap();
        writeln!(out, "    \"max_block_scratch_bytes\": {},", self.bounded_entropy.max_block_scratch_bytes).unwrap();
        writeln!(out, "    \"prototype_total_frame_bytes\": {},", entropy_total).unwrap();
        writeln!(out, "    \"prototype_total_ratio\": {:.9},", entropy_total as f64 / self.input_bytes.max(1) as f64).unwrap();
        writeln!(out, "    \"pack_elapsed_ns\": {},", self.bounded_entropy.pack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
        writeln!(out, "    \"unpack_elapsed_ns\": {},", self.bounded_entropy.unpack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
        writeln!(out, "    \"pack_commands_per_second\": {},", entropy_pack_commands_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "    \"unpack_commands_per_second\": {}", entropy_unpack_commands_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  }},").unwrap();
        let adaptive_total = self
            .common_frame_envelope_bytes
            .checked_add(self.raw_passthrough_bytes)
            .and_then(|value| value.checked_add(self.literal_bytes))
            .and_then(|value| value.checked_add(self.adaptive_entropy.metadata_bytes))
            .unwrap();
        let adaptive_pack_commands_per_second = self
            .adaptive_entropy
            .pack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.adaptive_entropy.commands as f64 * 1_000_000_000.0 / ns as f64);
        let adaptive_unpack_commands_per_second = self
            .adaptive_entropy
            .unpack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.adaptive_entropy.commands as f64 * 1_000_000_000.0 / ns as f64);
        writeln!(out, "  \"adaptive_entropy\": {{").unwrap();
        writeln!(out, "    \"scope\": \"Per block, build both fixed-128 and bounded entropy payloads, prepend a one-byte selector, and charge both builders to pack time. Output execution is not timed.\",").unwrap();
        writeln!(out, "    \"commands\": {},", self.adaptive_entropy.commands).unwrap();
        writeln!(out, "    \"metadata_bytes\": {},", self.adaptive_entropy.metadata_bytes).unwrap();
        writeln!(out, "    \"metadata_bytes_per_command\": {:.6},", self.adaptive_entropy.metadata_bytes as f64 / self.adaptive_entropy.commands.max(1) as f64).unwrap();
        writeln!(out, "    \"max_block_selection_scratch_bytes\": {},", self.adaptive_entropy.max_block_scratch_bytes).unwrap();
        writeln!(out, "    \"entropy_blocks\": {},", self.adaptive_entropy.entropy_blocks).unwrap();
        writeln!(out, "    \"fixed_blocks\": {},", self.adaptive_entropy.fixed_blocks).unwrap();
        writeln!(out, "    \"prototype_total_frame_bytes\": {},", adaptive_total).unwrap();
        writeln!(out, "    \"prototype_total_ratio\": {:.9},", adaptive_total as f64 / self.input_bytes.max(1) as f64).unwrap();
        writeln!(out, "    \"pack_elapsed_ns\": {},", self.adaptive_entropy.pack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
        writeln!(out, "    \"unpack_elapsed_ns\": {},", self.adaptive_entropy.unpack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
        writeln!(out, "    \"pack_commands_per_second\": {},", adaptive_pack_commands_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "    \"unpack_commands_per_second\": {}", adaptive_unpack_commands_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  }},").unwrap();
        let literal_entropy_total = self
            .frame_bytes
            .checked_sub(self.literal_bytes)
            .and_then(|value| value.checked_add(self.literal_entropy.encoded_bytes))
            .unwrap();
        let literal_entropy_pack_bytes_per_second = self
            .literal_entropy
            .pack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.literal_entropy.literal_bytes as f64 * 1_000_000_000.0 / ns as f64);
        let literal_entropy_unpack_bytes_per_second = self
            .literal_entropy
            .unpack_ns
            .filter(|&ns| ns != 0)
            .map(|ns| self.literal_entropy.literal_bytes as f64 * 1_000_000_000.0 / ns as f64);
        writeln!(out, "  \"literal_entropy\": {{").unwrap();
        writeln!(out, "    \"scope\": \"Per-LZ-block selector over raw, canonical-Huffman and deterministic 8-byte token dictionaries (16/32/64/128 entry limits). Every candidate payload is built for selection; selector, tables, token bytes, bit lengths and padding are charged. Complete total preserves current v2 control/attribute/prefix bytes and replaces literal payload bytes only.\",").unwrap();
        writeln!(out, "    \"literal_input_bytes\": {},", self.literal_entropy.literal_bytes).unwrap();
        writeln!(out, "    \"encoded_literal_bytes\": {},", self.literal_entropy.encoded_bytes).unwrap();
        writeln!(out, "    \"max_block_selection_scratch_bytes\": {},", self.literal_entropy.max_block_scratch_bytes).unwrap();
        writeln!(out, "    \"entropy_blocks\": {},", self.literal_entropy.entropy_blocks).unwrap();
        writeln!(out, "    \"dictionary_blocks\": {},", self.literal_entropy.dictionary_blocks).unwrap();
        writeln!(out, "    \"dictionary_entries\": {},", self.literal_entropy.dictionary_entries).unwrap();
        writeln!(out, "    \"raw_blocks\": {},", self.literal_entropy.raw_blocks).unwrap();
        writeln!(out, "    \"prototype_total_frame_bytes\": {},", literal_entropy_total).unwrap();
        writeln!(out, "    \"prototype_total_ratio\": {:.9},", literal_entropy_total as f64 / self.input_bytes.max(1) as f64).unwrap();
        writeln!(out, "    \"pack_elapsed_ns\": {},", self.literal_entropy.pack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
        writeln!(out, "    \"unpack_elapsed_ns\": {},", self.literal_entropy.unpack_ns.map_or("null".to_owned(), |value| value.to_string())).unwrap();
        writeln!(out, "    \"pack_literal_bytes_per_second\": {},", literal_entropy_pack_bytes_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "    \"unpack_literal_bytes_per_second\": {}", literal_entropy_unpack_bytes_per_second.map_or("null".to_owned(), |value| format!("{value:.3}"))).unwrap();
        writeln!(out, "  }}").unwrap();
        writeln!(out, "}}").unwrap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_with(field: usize, value: u32) -> MetadataCommand {
        let mut values = [0u32; FIELD_COUNT];
        values[0] = 7;
        values[field] = value;
        command(values).unwrap()
    }

    fn roundtrip(layout: Layout, commands: &[MetadataCommand]) {
        let packed = pack_blocks(&[commands.to_vec()], layout, 32).unwrap();
        assert_eq!(unpack_blocks(&packed, layout).unwrap(), [commands.to_vec()]);
    }

    #[test]
    fn every_required_width_roundtrips_fixed_and_bitsliced() {
        for bits in [0u8, 1, 4, 8, 16, 32] {
            let high = if bits == 0 {
                0
            } else if bits == 32 {
                u32::MAX
            } else {
                (1 << bits) - 1
            };
            let commands = [command_with(3, 0), command_with(3, high)];
            roundtrip(Layout::Fixed, &commands);
            roundtrip(Layout::Bitsliced, &commands);
        }
    }

    #[test]
    fn int4_escapes_and_odd_tail_are_exact() {
        let commands = [command_with(1, 0), command_with(1, 14), command_with(1, 15)];
        roundtrip(Layout::Int4, &commands);
        let mut bytes = pack_int4_group(&commands).unwrap();
        let nibble_start = 1 + FIELD_COUNT * 4;
        bytes[nibble_start + 1] |= 0x80;
        assert!(unpack_int4_group(&mut Cursor::new(&bytes)).is_err());
    }

    #[test]
    fn ternary_accepts_every_opcode_and_rejects_reserved_selector() {
        let commands: Vec<_> = (0..=7)
            .map(|op| MetadataCommand {
                op,
                lit: op as u32,
                run: 8 + op as u32,
                distance: 17 + op as u32,
                seed_lo: op as u32,
                seed_hi: 0,
            })
            .collect();
        roundtrip(Layout::Ternary2Bit, &commands);
        roundtrip(Layout::TernaryBase3, &commands);
        let mut bytes = pack_ternary_group(&commands[..5], Selector::Base3).unwrap();
        let selector_offset = 1 + NUMERIC_FIELD_COUNT * 4;
        bytes[selector_offset] = 243;
        assert!(unpack_ternary_group(&mut Cursor::new(&bytes), Selector::Base3).is_err());
    }

    #[test]
    fn bitslice_is_byte_identical_to_fixed_under_the_same_plan() {
        let commands: Vec<_> = (0..32)
            .map(|index| MetadataCommand {
                op: if index & 1 == 0 { 7 } else { 2 },
                lit: (index % 5) as u32,
                run: 8 + (index % 17) as u32,
                distance: 31 + (index * 37) as u32,
                seed_lo: 0,
                seed_hi: 0,
            })
            .collect();
        assert_eq!(
            pack_fixed_group(&commands, false).unwrap().len(),
            pack_fixed_group(&commands, true).unwrap().len()
        );
    }

    #[test]
    fn evaluation_roundtrips_an_actual_encoder_trace() {
        let data = (0..16_384)
            .flat_map(|index| {
                format!("{{\"event\":\"update\",\"id\":{}}}\n", index % 47).into_bytes()
            })
            .collect::<Vec<_>>();
        let encoded = crate::compress(
            &data,
            &crate::Options {
                block_size: 4096,
                checksum: false,
                legacy_v2: true,
                ..crate::Options::default()
            },
        )
        .unwrap();
        let frame = Frame::parse(&encoded, data.len()).unwrap();
        let evaluation = evaluate_frame(&frame).unwrap();
        assert_eq!(evaluation.frame_bytes, encoded.len());
        assert_eq!(
            evaluation.results.len(),
            Layout::ALL.len() * GROUP_SIZES.len()
        );
        assert!(evaluation.results.iter().all(|result| result.commands > 0));
        assert!(!evaluation.distance_results.is_empty());
    }

    #[test]
    fn distance_models_roundtrip_and_reject_noncanonical_uleb() {
        let commands = vec![
            MetadataCommand { op: 7, lit: 0, run: 8, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 2, lit: 1, run: 9, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 7, lit: 0, run: 16, distance: 130_000, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 0, lit: 4, run: 0, distance: 0, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 1, lit: 0, run: 12, distance: 65_535, seed_lo: 0, seed_hi: 0 },
        ];
        let blocks = vec![commands];
        for model in DistanceModel::ALL {
            let packed = pack_distance_blocks(&blocks, model, 32).unwrap();
            assert_eq!(
                unpack_distance_blocks(&packed, model).unwrap(),
                expected_distances(&blocks).unwrap()
            );
        }
        for group_size in GROUP_SIZES {
            let packed = pack_distance_blocks(&blocks, DistanceModel::Dictionary16, group_size).unwrap();
            assert_eq!(
                unpack_distance_blocks(&packed, DistanceModel::Dictionary16).unwrap(),
                expected_distances(&blocks).unwrap()
            );
        }
        assert!(read_uleb(&mut Cursor::new(&[0x80, 0])).is_err());
    }

    #[test]
    fn bounded_entropy_roundtrips_mixed_commands_and_rejects_table_reordering() {
        let commands = vec![
            MetadataCommand { op: 0, lit: 7, run: 0, distance: 0, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 7, lit: 0, run: 8, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 2, lit: 1, run: 9, distance: 31, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 3, lit: 0, run: 17, distance: 0, seed_lo: 0x0000_4241, seed_hi: 0 },
            MetadataCommand { op: 1, lit: 2, run: 33, distance: 130_000, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 6, lit: 0, run: 64, distance: 0, seed_lo: 0x0403_0201, seed_hi: 0x0807_0605 },
        ];
        let packed = pack_entropy_blocks(&[commands.clone()]).unwrap();
        assert_eq!(unpack_entropy_blocks(&packed).unwrap(), [commands]);
        let mut malformed = pack_entropy_block(&[MetadataCommand { op: 7, lit: 0, run: 8, distance: 1, seed_lo: 0, seed_hi: 0 }, MetadataCommand { op: 2, lit: 0, run: 8, distance: 1, seed_lo: 0, seed_hi: 0 }]).unwrap();
        // Command count (4 bytes), then the opcode table length (2 bytes). The
        // first table is canonical length/symbol order; swap its symbols only.
        malformed.swap(6, 11);
        assert!(unpack_entropy_block(&malformed).is_err());
    }

    #[test]
    fn adaptive_entropy_charges_and_roundtrips_both_block_modes() {
        let tiny = vec![
            MetadataCommand { op: 7, lit: 0, run: 8, distance: 17, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 2, lit: 0, run: 8, distance: 17, seed_lo: 0, seed_hi: 0 },
        ];
        let repeated = (0..2048)
            .map(|index| {
                let mut command = MetadataCommand {
                    op: 7,
                    lit: 0,
                    run: 16,
                    distance: 31,
                    seed_lo: 0,
                    seed_hi: 0,
                };
                // A few high residuals force fixed-width planes wide, while
                // the remaining commands keep every field highly skewed.
                // Repeat the same skewed distribution in each 128-command
                // fixed group. Fixed-width packing must pay the outlier width
                // for all commands in every group; the entropy codebooks pay
                // their tables once for the whole LZ block.
                match index % 128 {
                    0..=7 => command.op = 1,
                    8..=15 => command.lit = 1 << (index - 1),
                    16..=23 => command.run = 1 << (index - 9),
                    24..=31 => command.distance = 1 << (index - 17),
                    32..=39 => command.seed_lo = 1 << (index - 25),
                    40..=47 => command.seed_hi = 1 << (index - 33),
                    _ => {}
                }
                command
            })
            .collect::<Vec<_>>();
        let blocks = vec![tiny, repeated];
        let packed = pack_adaptive_entropy_blocks(&blocks).unwrap();
        assert_eq!(unpack_adaptive_entropy_blocks(&packed).unwrap(), blocks);
        assert!(packed.fixed_blocks > 0);
        assert!(packed.entropy_blocks > 0);
        assert!(packed.max_block_scratch_bytes > packed.metadata_bytes / 2);
    }

    #[test]
    fn literal_entropy_roundtrips_all_literal_modes_and_rejects_malformed_tables() {
        let tokens = (0..16)
            .map(|token| {
                std::array::from_fn(|index| (token * TOKEN_WIDTH + index) as u8)
            })
            .collect::<Vec<[u8; TOKEN_WIDTH]>>();
        let dictionary_friendly = (0..32)
            .flat_map(|_| tokens.iter().flat_map(|token| token.iter().copied()))
            .collect::<Vec<u8>>();
        let mut state = 0x6a09e667f3bcc909u64;
        let entropy_friendly = (0..512)
            .map(|_| {
                state = state
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(0x94d049bb133111eb);
                b"abcd"[(state >> 62) as usize]
            })
            .collect::<Vec<u8>>();
        let blocks = vec![
            Vec::new(),
            entropy_friendly,
            vec![b'a'; 512],
            dictionary_friendly,
            (0..64).collect::<Vec<u8>>(),
        ];
        let packed = pack_literal_entropy_blocks(&blocks).unwrap();
        assert_eq!(unpack_literal_entropy_blocks(&packed).unwrap(), blocks);
        assert!(packed.raw_blocks > 0);
        assert!(packed.entropy_blocks > 0);
        assert!(packed.dictionary_blocks > 0);

        let entropy_index = packed
            .blocks
            .iter()
            .position(|block| block.first() == Some(&1))
            .unwrap();
        let mut malformed = packed;
        malformed.blocks[entropy_index][0] = 2;
        assert!(unpack_literal_entropy_blocks(&malformed).is_err());

        let mut malformed = pack_literal_entropy_blocks(&blocks).unwrap();
        let dictionary_index = malformed
            .blocks
            .iter()
            .position(|block| block.first() == Some(&2))
            .unwrap();
        let first_token = malformed.blocks[dictionary_index][2..2 + TOKEN_WIDTH].to_vec();
        malformed.blocks[dictionary_index][2 + TOKEN_WIDTH..2 + TOKEN_WIDTH * 2]
            .copy_from_slice(&first_token);
        assert!(unpack_literal_entropy_blocks(&malformed).is_err());
    }

}
