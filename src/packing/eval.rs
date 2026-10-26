use std::{fmt::Write, time::Instant};
use crate::{analysis::{command_blocks, literal_blocks}, block, frame, Frame};
use super::*;

pub(super) fn checked_result(
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

