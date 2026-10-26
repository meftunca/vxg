use std::time::Instant;
use super::*;

pub(super) const DISTANCE_TAG_ESCAPE: u32 = 8;
pub(super) const DISTANCE_TAG_NONE: u32 = 9;

pub(super) fn entropy_distance_tags(commands: &[MetadataCommand]) -> Result<(Vec<u32>, Vec<u32>)> {
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

pub(super) fn pack_entropy_block(commands: &[MetadataCommand]) -> Result<Vec<u8>> {
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

pub(super) fn unpack_entropy_block(bytes: &[u8]) -> Result<Vec<MetadataCommand>> {
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

pub(super) fn pack_entropy_blocks(blocks: &[Vec<MetadataCommand>]) -> Result<PackedEntropy> {
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

pub(super) fn unpack_entropy_blocks(packed: &PackedEntropy) -> Result<Vec<Vec<MetadataCommand>>> {
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

pub(super) fn checked_entropy_result(blocks: &[Vec<MetadataCommand>]) -> Result<EntropyResult> {
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

pub(super) fn pack_fixed_128_block(commands: &[MetadataCommand]) -> Result<Vec<u8>> {
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

pub(super) fn unpack_fixed_128_block(bytes: &[u8]) -> Result<Vec<MetadataCommand>> {
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

pub(super) fn pack_adaptive_entropy_blocks(
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

pub(super) fn unpack_adaptive_entropy_blocks(
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

pub(super) fn checked_adaptive_entropy_result(
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

