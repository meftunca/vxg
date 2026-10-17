//! Offline command-stream diagnostics for format research.
//!
//! This module is compiled only with the `bench` feature and is never called by
//! `compress` or a timed decode path. Its output describes an existing frame;
//! it does not change matcher decisions or wire bytes.
use std::{collections::BTreeMap, fmt::Write};

use crate::{
    block::{self, Command, Parser},
    Error, Frame, Result,
};

const GROUP: usize = 8;
const WIDTHS: usize = 65;

/// One lossless logical command, detached from the v1/v2 byte representation.
/// Command blocks preserve independent-block reset points for packing experiments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetadataCommand {
    pub op: u8,
    pub lit: u32,
    pub run: u32,
    pub distance: u32,
    /// Pattern opcodes carry up to eight seed bytes. Keeping the value split
    /// avoids a lossy "match-only" research trace and makes every prototype
    /// roundtrip every existing v2 opcode.
    pub seed_lo: u32,
    pub seed_hi: u32,
}

#[derive(Debug)]
pub struct CommandAnalysis {
    pub blocks: u64,
    pub raw_blocks: u64,
    pub raw_bytes: u64,
    pub commands: u64,
    pub match_commands: u64,
    pub matched_bytes: u64,
    pub literal_bytes: u64,
    pub op_counts: [u64; 8],
    pub literal_histogram: BTreeMap<usize, u64>,
    pub run_histogram: BTreeMap<usize, u64>,
    pub opcode_literal_run: BTreeMap<(u8, usize, usize), u64>,
    pub group_literal_widths: [u64; WIDTHS],
    pub group_run_widths: [u64; WIDTHS],
    pub group_distance_widths: [u64; WIDTHS],
    pub group_distance_delta_widths: [u64; WIDTHS],
    pub group_distance_xor_widths: [u64; WIDTHS],
    pub cache_hits_2: u64,
    pub cache_hits_4: u64,
    pub cache_hits_8: u64,
    pub nonoverlap_matches: u64,
    pub references_within_group: u64,
    pub min_distance: Option<usize>,
}

impl Default for CommandAnalysis {
    fn default() -> Self {
        Self {
            blocks: 0,
            raw_blocks: 0,
            raw_bytes: 0,
            commands: 0,
            match_commands: 0,
            matched_bytes: 0,
            literal_bytes: 0,
            op_counts: [0; 8],
            literal_histogram: BTreeMap::new(),
            run_histogram: BTreeMap::new(),
            opcode_literal_run: BTreeMap::new(),
            group_literal_widths: [0; WIDTHS],
            group_run_widths: [0; WIDTHS],
            group_distance_widths: [0; WIDTHS],
            group_distance_delta_widths: [0; WIDTHS],
            group_distance_xor_widths: [0; WIDTHS],
            cache_hits_2: 0,
            cache_hits_4: 0,
            cache_hits_8: 0,
            nonoverlap_matches: 0,
            references_within_group: 0,
            min_distance: None,
        }
    }
}

#[derive(Clone, Copy)]
struct GroupCommand {
    op: u8,
    lit: usize,
    run: usize,
    distance: usize,
}

fn width(value: usize) -> usize {
    (usize::BITS - value.leading_zeros()) as usize
}

fn span_width(values: impl Iterator<Item = usize>) -> usize {
    let mut min = usize::MAX;
    let mut max = 0;
    let mut seen = false;
    for value in values {
        min = min.min(value);
        max = max.max(value);
        seen = true;
    }
    if seen {
        width(max - min)
    } else {
        0
    }
}

fn cache_hit(cache: &mut Vec<usize>, distance: usize, capacity: usize) -> bool {
    let hit = cache.iter().position(|&value| value == distance);
    if let Some(index) = hit {
        cache.remove(index);
    }
    cache.insert(0, distance);
    cache.truncate(capacity);
    hit.is_some()
}

impl CommandAnalysis {
    fn record_group(&mut self, group: &[GroupCommand]) {
        if group.is_empty() {
            return;
        }
        self.group_literal_widths[span_width(group.iter().map(|command| command.lit))] += 1;
        self.group_run_widths[span_width(group.iter().map(|command| command.run))] += 1;
        let distances: Vec<usize> = group
            .iter()
            .filter(|command| matches!(command.op, 1 | 2 | 7))
            .map(|command| command.distance)
            .collect();
        if distances.is_empty() {
            return;
        }
        self.group_distance_widths[span_width(distances.iter().copied())] += 1;
        let deltas = distances.windows(2).map(|pair| pair[1].abs_diff(pair[0]));
        self.group_distance_delta_widths[span_width(deltas)] += 1;
        let xors = distances.windows(2).map(|pair| pair[1] ^ pair[0]);
        self.group_distance_xor_widths[span_width(xors)] += 1;
    }

    fn number_array(values: &[u64]) -> String {
        values
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn histogram_json(values: &BTreeMap<usize, u64>) -> String {
        values
            .iter()
            .map(|(value, count)| format!("\"{value}\":{count}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    fn joint_json(values: &BTreeMap<(u8, usize, usize), u64>) -> String {
        values
            .iter()
            .map(|((op, lit, run), count)| format!("\"{op}/{lit}/{run}\":{count}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// A stable, dependency-free JSON report for checked-in qualification data.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        let average_output = if self.commands == 0 {
            0.0
        } else {
            self.raw_bytes as f64 / self.commands as f64
        };
        let average_match = if self.match_commands == 0 {
            0.0
        } else {
            self.matched_bytes as f64 / self.match_commands as f64
        };
        writeln!(out, "{{").unwrap();
        writeln!(out, "  \"schema\": 1,").unwrap();
        writeln!(out, "  \"group_size\": {GROUP},").unwrap();
        writeln!(out, "  \"cache_policy\": \"Per block and capacity: look up every MATCH/REP/MATCH16 distance, remove a hit, insert at MRU front, truncate LRU tail.\",").unwrap();
        writeln!(out, "  \"blocks\": {},", self.blocks).unwrap();
        writeln!(out, "  \"raw_blocks\": {},", self.raw_blocks).unwrap();
        writeln!(out, "  \"raw_bytes\": {},", self.raw_bytes).unwrap();
        writeln!(out, "  \"commands\": {},", self.commands).unwrap();
        writeln!(out, "  \"match_commands\": {},", self.match_commands).unwrap();
        writeln!(out, "  \"matched_bytes\": {},", self.matched_bytes).unwrap();
        writeln!(out, "  \"literal_bytes\": {},", self.literal_bytes).unwrap();
        writeln!(
            out,
            "  \"average_output_bytes_per_command\": {average_output:.6},"
        )
        .unwrap();
        writeln!(out, "  \"average_match_bytes\": {average_match:.6},").unwrap();
        writeln!(
            out,
            "  \"op_counts\": [{}],",
            Self::number_array(&self.op_counts)
        )
        .unwrap();
        writeln!(
            out,
            "  \"literal_histogram\": {{{}}},",
            Self::histogram_json(&self.literal_histogram)
        )
        .unwrap();
        writeln!(
            out,
            "  \"run_histogram\": {{{}}},",
            Self::histogram_json(&self.run_histogram)
        )
        .unwrap();
        writeln!(
            out,
            "  \"opcode_literal_run\": {{{}}},",
            Self::joint_json(&self.opcode_literal_run)
        )
        .unwrap();
        writeln!(
            out,
            "  \"group_literal_residual_width_histogram\": [{}],",
            Self::number_array(&self.group_literal_widths)
        )
        .unwrap();
        writeln!(
            out,
            "  \"group_run_residual_width_histogram\": [{}],",
            Self::number_array(&self.group_run_widths)
        )
        .unwrap();
        writeln!(
            out,
            "  \"group_distance_residual_width_histogram\": [{}],",
            Self::number_array(&self.group_distance_widths)
        )
        .unwrap();
        writeln!(
            out,
            "  \"group_distance_delta_width_histogram\": [{}],",
            Self::number_array(&self.group_distance_delta_widths)
        )
        .unwrap();
        writeln!(
            out,
            "  \"group_distance_xor_width_histogram\": [{}],",
            Self::number_array(&self.group_distance_xor_widths)
        )
        .unwrap();
        writeln!(
            out,
            "  \"cache_hits\": {{\"2\":{},\"4\":{},\"8\":{}}},",
            self.cache_hits_2, self.cache_hits_4, self.cache_hits_8
        )
        .unwrap();
        writeln!(
            out,
            "  \"nonoverlap_matches\": {},",
            self.nonoverlap_matches
        )
        .unwrap();
        writeln!(
            out,
            "  \"references_within_8_command_group\": {},",
            self.references_within_group
        )
        .unwrap();
        writeln!(
            out,
            "  \"min_distance\": {}",
            self.min_distance
                .map_or("null".into(), |value| value.to_string())
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        out
    }
}

fn next_command(parser: &mut Parser<'_>) -> Result<Option<Command>> {
    parser.next()
}

/// Return every non-RAW command stream as exact logical commands. The command
/// parser validates each source stream and block-local state before returning.
/// This is an offline bench-only API; production encoding and decoding do not
/// allocate this representation.
pub fn command_blocks(frame: &Frame<'_>) -> Result<Vec<Vec<MetadataCommand>>> {
    let mut blocks = Vec::with_capacity(frame.blocks.len());
    for block in &frame.blocks {
        if block.info.kind == block::RAW {
            blocks.push(Vec::new());
            continue;
        }
        let mut parser = Parser::new(block.payload, block.info.raw_size, frame.version >= 2)?;
        let mut commands = Vec::new();
        while let Some(command) = next_command(&mut parser)? {
            commands.push(MetadataCommand {
                op: command.op,
                lit: u32::try_from(command.lit).map_err(|_| crate::Error::SizeOverflow)?,
                run: u32::try_from(command.run).map_err(|_| crate::Error::SizeOverflow)?,
                distance: u32::try_from(command.distance)
                    .map_err(|_| crate::Error::SizeOverflow)?,
                seed_lo: u32::from_le_bytes(command.seed[..4].try_into().unwrap()),
                seed_hi: u32::from_le_bytes(command.seed[4..].try_into().unwrap()),
            });
        }
        blocks.push(commands);
    }
    Ok(blocks)
}

/// Return the exact literal byte stream for every non-RAW v2 block. The
/// command parser verifies each literal boundary and complete stream
/// consumption before the detached copy is returned. RAW blocks contribute an
/// empty entry because their bytes remain a separately accounted passthrough
/// payload in packing experiments.
///
/// This is a bench-only extraction helper. It does not alter public framing,
/// matcher decisions or timed decode behavior.
pub fn literal_blocks(frame: &Frame<'_>) -> Result<Vec<Vec<u8>>> {
    let mut blocks = Vec::with_capacity(frame.blocks.len());
    for block in &frame.blocks {
        if block.info.kind == block::RAW {
            blocks.push(Vec::new());
            continue;
        }
        let mut parser = Parser::new(block.payload, block.info.raw_size, frame.version >= 2)?;
        let mut literals = Vec::with_capacity(parser.literals.len());
        while let Some(command) = next_command(&mut parser)? {
            let end = command
                .literal_start
                .checked_add(command.lit)
                .ok_or(Error::SizeOverflow)?;
            literals.extend_from_slice(
                parser
                    .literals
                    .get(command.literal_start..end)
                    .ok_or(Error::Truncated)?,
            );
        }
        if literals.len() != parser.literals.len() {
            return Err(Error::InvalidFrame("literal stream accounting"));
        }
        blocks.push(literals);
    }
    Ok(blocks)
}

/// Analyze a parsed frame. The parser still validates complete stream consumption.
pub fn analyze_frame(frame: &Frame<'_>) -> Result<CommandAnalysis> {
    let mut analysis = CommandAnalysis {
        blocks: frame.blocks.len() as u64,
        raw_bytes: frame.raw_size as u64,
        ..CommandAnalysis::default()
    };
    for block in &frame.blocks {
        if block.info.kind == block::RAW {
            analysis.raw_blocks += 1;
            continue;
        }
        let mut parser = Parser::new(block.payload, block.info.raw_size, frame.version >= 2)?;
        let mut group = Vec::with_capacity(GROUP);
        let mut cache2 = Vec::with_capacity(2);
        let mut cache4 = Vec::with_capacity(4);
        let mut cache8 = Vec::with_capacity(8);
        let mut group_output_start = 0;
        while let Some(command) = next_command(&mut parser)? {
            let match_pos = group_output_start
                + group
                    .iter()
                    .map(|item: &GroupCommand| item.lit + item.run)
                    .sum::<usize>()
                + command.lit;
            analysis.commands += 1;
            analysis.literal_bytes += command.lit as u64;
            analysis.op_counts[command.op as usize] += 1;
            *analysis.literal_histogram.entry(command.lit).or_default() += 1;
            *analysis.run_histogram.entry(command.run).or_default() += 1;
            *analysis
                .opcode_literal_run
                .entry((command.op, command.lit, command.run))
                .or_default() += 1;
            if matches!(command.op, 1 | 2 | 7) {
                analysis.match_commands += 1;
                analysis.matched_bytes += command.run as u64;
                analysis.nonoverlap_matches += u64::from(command.run <= command.distance);
                analysis.min_distance = Some(
                    analysis
                        .min_distance
                        .map_or(command.distance, |value| value.min(command.distance)),
                );
                analysis.references_within_group +=
                    u64::from(command.distance < match_pos.saturating_sub(group_output_start));
                analysis.cache_hits_2 += u64::from(cache_hit(&mut cache2, command.distance, 2));
                analysis.cache_hits_4 += u64::from(cache_hit(&mut cache4, command.distance, 4));
                analysis.cache_hits_8 += u64::from(cache_hit(&mut cache8, command.distance, 8));
            }
            group.push(GroupCommand {
                op: command.op,
                lit: command.lit,
                run: command.run,
                distance: command.distance,
            });
            if group.len() == GROUP {
                analysis.record_group(&group);
                group_output_start += group.iter().map(|item| item.lit + item.run).sum::<usize>();
                group.clear();
            }
        }
        if !group.is_empty() {
            analysis.record_group(&group);
        }
    }
    Ok(analysis)
}
