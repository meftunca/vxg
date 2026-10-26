//! Bench-only P6B grouped-command decoder research.
//!
//! The packet in this module is deliberately not a VXG wire format. It takes
//! the already validated v2 logical command trace, serializes full 64-command
//! bit planes plus literals, and measures the scalar unpack/validation and
//! execution split without ever claiming a public VXG safe-decode result.

use crate::{
    analysis::{command_blocks, literal_blocks, MetadataCommand},
    block, frame, simd::Engine, Error, Frame, Result, Simd,
};

mod eval;
pub use eval::{evaluate_frame, Evaluation};

pub const GROUP_SIZE: usize = 64;
const FIELD_COUNT: usize = 6;
const BLOCK_MAGIC: &[u8; 4] = b"G641";
const BLOCK_HEADER: usize = 20;

const EMPTY: MetadataCommand = MetadataCommand {
    op: 0,
    lit: 0,
    run: 0,
    distance: 0,
    seed_lo: 0,
    seed_hi: 0,
};

fn invalid(message: &'static str) -> Error {
    Error::InvalidFrame(message)
}

fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(Error::SizeOverflow)
}

fn read_u16(bytes: &[u8], pos: &mut usize) -> Result<u16> {
    let end = checked_add(*pos, 2)?;
    let value = bytes.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u16::from_le_bytes(value.try_into().unwrap()))
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32> {
    let end = checked_add(*pos, 4)?;
    let value = bytes.get(*pos..end).ok_or(Error::Truncated)?;
    *pos = end;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn write_u16(out: &mut Vec<u8>, value: usize) -> Result<()> {
    out.extend_from_slice(&u16::try_from(value).map_err(|_| Error::SizeOverflow)?.to_le_bytes());
    Ok(())
}

fn write_u32(out: &mut Vec<u8>, value: usize) -> Result<()> {
    out.extend_from_slice(&u32::try_from(value).map_err(|_| Error::SizeOverflow)?.to_le_bytes());
    Ok(())
}

fn bit_width(value: u32) -> u8 {
    (u32::BITS - value.leading_zeros()) as u8
}

fn field(command: MetadataCommand, index: usize) -> u32 {
    match index {
        0 => command.op.into(),
        1 => command.lit,
        2 => command.run,
        3 => command.distance,
        4 => command.seed_lo,
        5 => command.seed_hi,
        _ => unreachable!("fixed group field"),
    }
}

fn command(values: [u32; FIELD_COUNT]) -> Result<MetadataCommand> {
    let op = u8::try_from(values[0]).map_err(|_| invalid("group opcode width"))?;
    Ok(MetadataCommand {
        op,
        lit: values[1],
        run: values[2],
        distance: values[3],
        seed_lo: values[4],
        seed_hi: values[5],
    })
}

#[derive(Clone, Copy)]
struct GroupPlan {
    base: u32,
    width: u8,
}

fn group_plans(commands: &[MetadataCommand]) -> Result<[GroupPlan; FIELD_COUNT]> {
    if commands.is_empty() || commands.len() > GROUP_SIZE {
        return Err(invalid("group command count"));
    }
    let mut plans = [GroupPlan { base: 0, width: 0 }; FIELD_COUNT];
    for (field_index, plan) in plans.iter_mut().enumerate() {
        let mut min = u32::MAX;
        let mut max = 0u32;
        for &command in commands {
            let value = field(command, field_index);
            min = min.min(value);
            max = max.max(value);
        }
        *plan = GroupPlan {
            base: min,
            width: bit_width(max - min),
        };
    }
    Ok(plans)
}

fn pack_group(out: &mut Vec<u8>, commands: &[MetadataCommand]) -> Result<()> {
    let plans = group_plans(commands)?;
    out.push(u8::try_from(commands.len()).map_err(|_| Error::SizeOverflow)?);
    for plan in plans {
        out.extend_from_slice(&plan.base.to_le_bytes());
        out.push(plan.width);
    }
    let plane_bytes = commands.len().div_ceil(8);
    for (field_index, plan) in plans.into_iter().enumerate() {
        for bit in 0..plan.width {
            let mut plane = vec![0u8; plane_bytes];
            for (lane, &command) in commands.iter().enumerate() {
                let residual = field(command, field_index)
                    .checked_sub(plan.base)
                    .ok_or_else(|| invalid("group base"))?;
                plane[lane / 8] |= u8::try_from((residual >> bit) & 1).unwrap() << (lane & 7);
            }
            out.extend_from_slice(&plane);
        }
    }
    Ok(())
}

/// Read exactly one group. `canonical` checks the exact minimum-base /
/// minimum-width encoding and zero tail bits; a validated replay can omit
/// those repeated checks without changing its untrusted input boundary.
fn read_group(
    bytes: &[u8],
    pos: &mut usize,
    expected_count: usize,
    canonical: bool,
) -> Result<[MetadataCommand; GROUP_SIZE]> {
    let count = *bytes.get(*pos).ok_or(Error::Truncated)? as usize;
    *pos = checked_add(*pos, 1)?;
    if count != expected_count || count == 0 || count > GROUP_SIZE {
        return Err(invalid("group command count"));
    }
    let mut plans = [GroupPlan { base: 0, width: 0 }; FIELD_COUNT];
    for plan in &mut plans {
        let base = read_u32(bytes, pos)?;
        let width = *bytes.get(*pos).ok_or(Error::Truncated)?;
        *pos = checked_add(*pos, 1)?;
        if width > 32 {
            return Err(invalid("group field width"));
        }
        *plan = GroupPlan { base, width };
    }
    let plane_bytes = count.div_ceil(8);
    let mut values = [[0u32; GROUP_SIZE]; FIELD_COUNT];
    for (field_index, plan) in plans.iter().copied().enumerate() {
        for bit in 0..plan.width {
            let end = checked_add(*pos, plane_bytes)?;
            let plane = bytes.get(*pos..end).ok_or(Error::Truncated)?;
            *pos = end;
            if canonical && count & 7 != 0 {
                let used = (1u8 << (count & 7)) - 1;
                if plane[plane_bytes - 1] & !used != 0 {
                    return Err(invalid("non-canonical group plane tail"));
                }
            }
            for lane in 0..count {
                values[field_index][lane] |= u32::from((plane[lane / 8] >> (lane & 7)) & 1) << bit;
            }
        }
        let mut min = u32::MAX;
        let mut max = 0u32;
        for value in &mut values[field_index][..count] {
            *value = plan.base.checked_add(*value).ok_or(Error::SizeOverflow)?;
            min = min.min(*value);
            max = max.max(*value);
        }
        if canonical && (min != plan.base || bit_width(max - min) != plan.width) {
            return Err(invalid("non-canonical group plan"));
        }
    }
    let mut commands = [EMPTY; GROUP_SIZE];
    for (lane, command_slot) in commands[..count].iter_mut().enumerate() {
        *command_slot = command(std::array::from_fn(|field| values[field][lane]))?;
    }
    Ok(commands)
}

#[derive(Clone, Copy)]
struct BlockHeader {
    raw_size: usize,
    commands: usize,
    groups: usize,
    literal_bytes: usize,
    groups_pos: usize,
}

fn read_block_header(bytes: &[u8]) -> Result<BlockHeader> {
    if bytes.len() < BLOCK_HEADER || bytes.get(..4) != Some(BLOCK_MAGIC) {
        return Err(invalid("group block magic"));
    }
    let mut pos = 4usize;
    let raw_size = read_u32(bytes, &mut pos)? as usize;
    let commands = read_u32(bytes, &mut pos)? as usize;
    let groups = read_u16(bytes, &mut pos)? as usize;
    let reserved = read_u16(bytes, &mut pos)?;
    let literal_bytes = read_u32(bytes, &mut pos)? as usize;
    if reserved != 0 || commands == 0 || commands > raw_size || groups == 0 {
        return Err(invalid("group block header"));
    }
    if groups != commands.div_ceil(GROUP_SIZE) {
        return Err(invalid("non-canonical group count"));
    }
    Ok(BlockHeader {
        raw_size,
        commands,
        groups,
        literal_bytes,
        groups_pos: pos,
    })
}

fn validate_command(
    command: MetadataCommand,
    raw_size: usize,
    output_pos: &mut usize,
    literal_pos: &mut usize,
    literal_bytes: usize,
    recent: &mut u32,
) -> Result<()> {
    if command.op > 7 || command.op == 7 && command.distance > u16::MAX as u32 {
        return Err(invalid("group opcode"));
    }
    let lit = command.lit as usize;
    let run = command.run as usize;
    if (command.op == 0 && (run != 0 || lit == 0)) || (command.op != 0 && run == 0) {
        return Err(invalid("group command progress"));
    }
    if matches!(command.op, 1 | 2 | 7) && run < 8 {
        return Err(invalid("group match below minmatch"));
    }
    let new_literals = checked_add(*literal_pos, lit)?;
    if new_literals > literal_bytes {
        return Err(Error::Truncated);
    }
    let match_pos = checked_add(*output_pos, lit)?;
    let next_pos = checked_add(match_pos, run)?;
    if next_pos > raw_size {
        return Err(invalid("group command exceeds output"));
    }
    match command.op {
        1 | 7 => {
            if command.distance == 0 || command.distance as usize > match_pos {
                return Err(invalid("group backreference before prefix"));
            }
            *recent = command.distance;
            if command.seed_lo != 0 || command.seed_hi != 0 {
                return Err(invalid("group match seed"));
            }
        }
        2 => {
            if *recent == 0 || command.distance != *recent || command.distance as usize > match_pos {
                return Err(invalid("group repeat distance"));
            }
            if command.seed_lo != 0 || command.seed_hi != 0 {
                return Err(invalid("group match seed"));
            }
        }
        3..=6 => {
            if command.distance != 0 {
                return Err(invalid("group pattern distance"));
            }
            let period = 1usize << (command.op - 3);
            let seed = command.seed_lo.to_le_bytes();
            if period < 4 && seed[period..].iter().any(|&byte| byte != 0) ||
                command.op < 6 && command.seed_hi != 0
            {
                return Err(invalid("group seed padding"));
            }
        }
        0 => {
            if command.distance != 0 || command.seed_lo != 0 || command.seed_hi != 0 {
                return Err(invalid("group literal fields"));
            }
        }
        _ => return Err(invalid("group opcode")),
    }
    *literal_pos = new_literals;
    *output_pos = next_pos;
    Ok(())
}

fn validate_grouped_block(bytes: &[u8]) -> Result<usize> {
    let header = read_block_header(bytes)?;
    let mut pos = header.groups_pos;
    let mut output_pos = 0usize;
    let mut literal_pos = 0usize;
    let mut recent = 0u32;
    let mut seen = 0usize;
    for group in 0..header.groups {
        let expected = (header.commands - seen).min(GROUP_SIZE);
        let commands = read_group(bytes, &mut pos, expected, true)?;
        for &command in &commands[..expected] {
            validate_command(
                command,
                header.raw_size,
                &mut output_pos,
                &mut literal_pos,
                header.literal_bytes,
                &mut recent,
            )?;
        }
        seen = checked_add(seen, expected)?;
        debug_assert!(group + 1 < header.groups || seen == header.commands);
    }
    let literal_end = checked_add(pos, header.literal_bytes)?;
    if seen != header.commands
        || output_pos != header.raw_size
        || literal_pos != header.literal_bytes
        || literal_end != bytes.len()
    {
        return Err(invalid("group block not consumed"));
    }
    Ok(pos)
}

fn pack_grouped_block(
    commands: &[MetadataCommand],
    literals: &[u8],
    raw_size: usize,
) -> Result<Vec<u8>> {
    if commands.is_empty() || commands.len() > raw_size {
        return Err(invalid("group source command count"));
    }
    let groups = commands.len().div_ceil(GROUP_SIZE);
    let mut out = Vec::new();
    out.extend_from_slice(BLOCK_MAGIC);
    write_u32(&mut out, raw_size)?;
    write_u32(&mut out, commands.len())?;
    write_u16(&mut out, groups)?;
    out.extend_from_slice(&0u16.to_le_bytes());
    write_u32(&mut out, literals.len())?;
    for group in commands.chunks(GROUP_SIZE) {
        pack_group(&mut out, group)?;
    }
    out.extend_from_slice(literals);
    let _ = validate_grouped_block(&out)?;
    Ok(out)
}

#[derive(Clone)]
enum PackedBlock {
    Raw { raw_size: usize, bytes: Vec<u8> },
    Grouped { raw_size: usize, bytes: Vec<u8> },
}

impl PackedBlock {
    fn raw_size(&self) -> usize {
        match self {
            Self::Raw { raw_size, .. } | Self::Grouped { raw_size, .. } => *raw_size,
        }
    }

    fn stored_bytes(&self) -> usize {
        match self {
            Self::Raw { bytes, .. } | Self::Grouped { bytes, .. } => bytes.len(),
        }
    }
}

/// A private, immutable proof that every grouped block has complete, canonical
/// metadata and a valid causal output schedule.
pub struct ValidatedGroupedFrame<'a> {
    frame: &'a PackedGroupedFrame,
    literal_starts: Vec<Option<usize>>,
}

pub struct PackedGroupedFrame {
    blocks: Vec<PackedBlock>,
    raw_size: usize,
    commands: usize,
    groups: usize,
    literal_bytes: usize,
    max_group_bytes: usize,
}

impl PackedGroupedFrame {
    pub fn from_frame(frame: &Frame<'_>) -> Result<Self> {
        let commands = command_blocks(frame)?;
        let literals = literal_blocks(frame)?;
        if commands.len() != frame.blocks.len() || literals.len() != frame.blocks.len() {
            return Err(invalid("group source block count"));
        }
        let mut blocks = Vec::with_capacity(frame.blocks.len());
        let mut command_count = 0usize;
        let mut group_count = 0usize;
        let mut literal_count = 0usize;
        let mut max_group_bytes = 0usize;
        for (index, source) in frame.blocks.iter().enumerate() {
            if source.info.kind == block::RAW {
                if source.payload.len() != source.info.raw_size {
                    return Err(invalid("group source RAW size"));
                }
                blocks.push(PackedBlock::Raw {
                    raw_size: source.info.raw_size,
                    bytes: source.payload.to_vec(),
                });
                continue;
            }
            let block_commands = &commands[index];
            let block_literals = &literals[index];
            let bytes = pack_grouped_block(block_commands, block_literals, source.info.raw_size)?;
            command_count = checked_add(command_count, block_commands.len())?;
            group_count = checked_add(group_count, block_commands.len().div_ceil(GROUP_SIZE))?;
            literal_count = checked_add(literal_count, block_literals.len())?;
            max_group_bytes = max_group_bytes.max(bytes.len().saturating_sub(BLOCK_HEADER + block_literals.len()));
            blocks.push(PackedBlock::Grouped {
                raw_size: source.info.raw_size,
                bytes,
            });
        }
        Ok(Self {
            blocks,
            raw_size: frame.raw_size(),
            commands: command_count,
            groups: group_count,
            literal_bytes: literal_count,
            max_group_bytes,
        })
    }

    pub fn validate(&self) -> Result<ValidatedGroupedFrame<'_>> {
        let mut raw_size = 0usize;
        let mut literal_starts = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            match block {
                PackedBlock::Raw { raw_size: expected, bytes } => {
                    if *expected != bytes.len() {
                        return Err(invalid("group RAW size"));
                    }
                    literal_starts.push(None);
                }
                PackedBlock::Grouped { raw_size: expected, bytes } => {
                    let literal_start = validate_grouped_block(bytes)?;
                    if read_block_header(bytes)?.raw_size != *expected {
                        return Err(invalid("group block raw size"));
                    }
                    literal_starts.push(Some(literal_start));
                }
            }
            raw_size = checked_add(raw_size, block.raw_size())?;
        }
        if raw_size != self.raw_size {
            return Err(invalid("group frame raw size"));
        }
        Ok(ValidatedGroupedFrame {
            frame: self,
            literal_starts,
        })
    }

    pub fn decode_into(&self, out: &mut [u8], engine: Engine) -> Result<()> {
        let validated = self.validate()?;
        validated.decode_into(out, engine)
    }

    pub fn synthetic_frame_bytes(&self) -> Result<usize> {
        let directory = self
            .blocks
            .len()
            .checked_mul(frame::ENTRY_SIZE)
            .ok_or(Error::SizeOverflow)?;
        let prefix = frame::HEADER_SIZE
            .checked_add(directory)
            .ok_or(Error::SizeOverflow)?;
        self.blocks
            .iter()
            .try_fold(prefix, |total, block| checked_add(total, block.stored_bytes()))
    }

    pub fn commands(&self) -> usize { self.commands }
    pub fn groups(&self) -> usize { self.groups }
    pub fn literal_bytes(&self) -> usize { self.literal_bytes }
    pub fn max_group_bytes(&self) -> usize { self.max_group_bytes }
    pub fn block_count(&self) -> usize { self.blocks.len() }
    pub fn raw_size(&self) -> usize { self.raw_size }
    pub fn lz_blocks(&self) -> usize {
        self.blocks.iter().filter(|block| matches!(block, PackedBlock::Grouped { .. })).count()
    }
}

