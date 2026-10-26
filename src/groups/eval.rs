use std::{hint::black_box, time::Instant};

use super::*;

impl ValidatedGroupedFrame<'_> {
    /// Replay the immutable packet after `PackedGroupedFrame::validate` has
    /// proven every group/header/literal/output relation. This is an isolated
    /// research row and cannot stand in for `PackedGroupedFrame::decode_into`.
    pub fn decode_into(&self, out: &mut [u8], engine: Engine) -> Result<()> {
        if out.len() != self.frame.raw_size {
            return Err(Error::OutputSize {
                expected: self.frame.raw_size,
                actual: out.len(),
            });
        }
        let mut output_pos = 0usize;
        for (block, literal_start) in self.frame.blocks.iter().zip(&self.literal_starts) {
            let end = checked_add(output_pos, block.raw_size())?;
            let target = &mut out[output_pos..end];
            match block {
                PackedBlock::Raw { bytes, .. } => engine.copy(target, bytes),
                PackedBlock::Grouped { bytes, .. } => decode_validated_block(
                    bytes,
                    target,
                    literal_start.expect("validated grouped literal start"),
                    engine,
                )?,
            }
            output_pos = end;
        }
        Ok(())
    }
}

fn decode_validated_block(
    bytes: &[u8],
    out: &mut [u8],
    literal_start: usize,
    engine: Engine,
) -> Result<()> {
    let header = read_block_header(bytes)?;
    if out.len() != header.raw_size {
        return Err(Error::OutputSize {
            expected: header.raw_size,
            actual: out.len(),
        });
    }
    let mut pos = header.groups_pos;
    let mut output_pos = 0usize;
    let mut literal_pos = 0usize;
    let mut seen = 0usize;
    let literals_end = literal_start
        .checked_add(header.literal_bytes)
        .ok_or(Error::SizeOverflow)?;
    let literals = bytes
        .get(literal_start..literals_end)
        .ok_or(Error::Truncated)?;
    for _ in 0..header.groups {
        let count = (header.commands - seen).min(GROUP_SIZE);
        let commands = read_group(bytes, &mut pos, count, false)?;
        for &command in &commands[..count] {
            let lit = command.lit as usize;
            let run = command.run as usize;
            let match_pos = output_pos + lit;
            if lit != 0 {
                engine.copy(
                    &mut out[output_pos..match_pos],
                    &literals[literal_pos..literal_pos + lit],
                );
            }
            output_pos = match_pos;
            match command.op {
                1 | 2 | 7 => engine.repeat(out, output_pos, run, command.distance as usize),
                3..=6 => {
                    let mut seed = [0u8; 8];
                    seed[..4].copy_from_slice(&command.seed_lo.to_le_bytes());
                    seed[4..].copy_from_slice(&command.seed_hi.to_le_bytes());
                    engine.pattern(&mut out[output_pos..output_pos + run], &seed[..1usize << (command.op - 3)]);
                }
                0 => {}
                _ => return Err(invalid("group validated opcode")),
            }
            output_pos += run;
            literal_pos += lit;
        }
        seen += count;
    }
    if pos != literal_start
        || literals_end != bytes.len()
        || literal_pos != header.literal_bytes
        || output_pos != out.len()
    {
        return Err(invalid("group validated consumption"));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Evaluation {
    pub input_bytes: usize,
    pub source_frame_bytes: usize,
    pub synthetic_frame_bytes: usize,
    pub blocks: usize,
    pub lz_blocks: usize,
    pub commands: usize,
    pub groups: usize,
    pub literal_bytes: usize,
    pub max_group_bytes: usize,
    pub validation_ns: u128,
    pub replay_ns: u128,
    pub safe_decode_ns: u128,
}

impl Evaluation {
    pub fn to_json(&self, iterations: usize, simd: Simd) -> String {
        let rate = |elapsed: u128, numerator: usize| {
            if elapsed == 0 { 0.0 } else { numerator as f64 * 1_000_000_000.0 / elapsed as f64 }
        };
        format!(
            concat!(
                "{{\n",
                "  \"schema\": \"vxg-p6b-grouped-command-r1\",\n",
                "  \"scope\": \"Bench-only synthetic grouped-command packet. It is not a VXG v3 wire format or a public safe-decode qualification. Validation, replay and safe rows are separate.\",\n",
                "  \"group_size\": {group_size},\n",
                "  \"simd\": \"{simd:?}\",\n",
                "  \"timing_iterations\": {iterations},\n",
                "  \"input_bytes\": {input},\n",
                "  \"source_frame_bytes\": {source},\n",
                "  \"synthetic_frame_bytes\": {synthetic},\n",
                "  \"synthetic_ratio\": {ratio:.9},\n",
                "  \"blocks\": {blocks},\n",
                "  \"lz_blocks\": {lz_blocks},\n",
                "  \"commands\": {commands},\n",
                "  \"groups\": {groups},\n",
                "  \"literal_bytes\": {literals},\n",
                "  \"max_group_bytes\": {max_group},\n",
                "  \"validation_commands_per_second\": {validate_cps:.3},\n",
                "  \"replay_commands_per_second\": {replay_cps:.3},\n",
                "  \"safe_commands_per_second\": {safe_cps:.3},\n",
                "  \"replay_input_bytes_per_second\": {replay_bps:.3},\n",
                "  \"safe_input_bytes_per_second\": {safe_bps:.3}\n",
                "}}\n"
            ),
            group_size = GROUP_SIZE,
            simd = simd,
            iterations = iterations,
            input = self.input_bytes,
            source = self.source_frame_bytes,
            synthetic = self.synthetic_frame_bytes,
            ratio = self.synthetic_frame_bytes as f64 / self.input_bytes.max(1) as f64,
            blocks = self.blocks,
            lz_blocks = self.lz_blocks,
            commands = self.commands,
            groups = self.groups,
            literals = self.literal_bytes,
            max_group = self.max_group_bytes,
            validate_cps = rate(self.validation_ns, self.commands * iterations),
            replay_cps = rate(self.replay_ns, self.commands * iterations),
            safe_cps = rate(self.safe_decode_ns, self.commands * iterations),
            replay_bps = rate(self.replay_ns, self.input_bytes * iterations),
            safe_bps = rate(self.safe_decode_ns, self.input_bytes * iterations),
        )
    }
}

pub fn evaluate_frame(
    frame: &Frame<'_>,
    input: &[u8],
    iterations: usize,
    simd: Simd,
) -> Result<Evaluation> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let engine = Engine::new(simd)?;
    let packed = PackedGroupedFrame::from_frame(frame)?;
    let mut out = vec![0u8; input.len()];
    packed.decode_into(&mut out, engine)?;
    if out != input {
        return Err(invalid("grouped safe roundtrip"));
    }
    let validation_start = Instant::now();
    let mut validated_count = 0usize;
    for _ in 0..iterations {
        let validated = packed.validate()?;
        validated_count ^= validated.frame.commands;
    }
    black_box(validated_count);
    let validation_ns = validation_start.elapsed().as_nanos();
    let validated = packed.validate()?;
    let replay_start = Instant::now();
    let mut replay_check = 0u8;
    for _ in 0..iterations {
        validated.decode_into(&mut out, engine)?;
        replay_check ^= out[0];
    }
    black_box(replay_check);
    let replay_ns = replay_start.elapsed().as_nanos();
    let safe_start = Instant::now();
    let mut safe_check = 0u8;
    for _ in 0..iterations {
        packed.decode_into(&mut out, engine)?;
        safe_check ^= out[0];
    }
    black_box(safe_check);
    let safe_decode_ns = safe_start.elapsed().as_nanos();
    Ok(Evaluation {
        input_bytes: input.len(),
        source_frame_bytes: {
            let directory = frame
                .blocks
                .len()
                .checked_mul(frame::ENTRY_SIZE)
                .ok_or(Error::SizeOverflow)?;
            let prefix = frame::HEADER_SIZE
                .checked_add(directory)
                .ok_or(Error::SizeOverflow)?;
            frame
                .blocks
                .iter()
                .try_fold(prefix, |sum, block| checked_add(sum, block.payload.len()))?
        },
        synthetic_frame_bytes: packed.synthetic_frame_bytes()?,
        blocks: packed.block_count(),
        lz_blocks: packed.lz_blocks(),
        commands: packed.commands(),
        groups: packed.groups(),
        literal_bytes: packed.literal_bytes(),
        max_group_bytes: packed.max_group_bytes(),
        validation_ns,
        replay_ns,
        safe_decode_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_packet_roundtrips_mixed_commands() {
        let commands = [
            MetadataCommand { op: 0, lit: 4, run: 0, distance: 0, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 1, lit: 0, run: 8, distance: 4, seed_lo: 0, seed_hi: 0 },
            MetadataCommand { op: 3, lit: 2, run: 3, distance: 0, seed_lo: u32::from_le_bytes(*b"x\0\0\0"), seed_hi: 0 },
        ];
        let bytes = pack_grouped_block(&commands, b"abcdZZ", 17).unwrap();
        validate_grouped_block(&bytes).unwrap();
        let frame = PackedGroupedFrame {
            blocks: vec![PackedBlock::Grouped { raw_size: 17, bytes }],
            raw_size: 17,
            commands: commands.len(),
            groups: 1,
            literal_bytes: 6,
            max_group_bytes: 0,
        };
        let mut out = [0u8; 17];
        frame.decode_into(&mut out, Engine::new(Simd::Scalar).unwrap()).unwrap();
        assert_eq!(&out, b"abcdabcdabcdZZxxx");
    }

    #[test]
    fn grouped_packet_rejects_noncanonical_plan_and_truncation() {
        let command = MetadataCommand { op: 0, lit: 1, run: 0, distance: 0, seed_lo: 0, seed_hi: 0 };
        let bytes = pack_grouped_block(&[command], b"x", 1).unwrap();
        assert!(validate_grouped_block(&bytes[..bytes.len() - 1]).is_err());
        let mut noncanonical = bytes;
        noncanonical[21] = 1; // first field's serialized minimum base
        assert!(validate_grouped_block(&noncanonical).is_err());
    }
}
