use std::time::Instant;
use super::*;

pub(super) fn is_match(command: MetadataCommand) -> bool {
    matches!(command.op, 1 | 2 | 7)
}

pub(super) fn distance_values(commands: &[MetadataCommand]) -> Result<Vec<u32>> {
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

pub(super) fn write_uleb(out: &mut Vec<u8>, mut value: u64) {
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

pub(super) fn uleb_len(mut value: u64) -> usize {
    let mut bytes = 1;
    while value >= 128 {
        value >>= 7;
        bytes += 1;
    }
    bytes
}

pub(super) fn read_uleb(cursor: &mut Cursor<'_>) -> Result<u64> {
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

pub(super) fn write_absolute_distances(out: &mut Vec<u8>, values: &[u32]) {
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

pub(super) fn read_absolute_distances(cursor: &mut Cursor<'_>) -> Result<Vec<u32>> {
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

pub(super) fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

pub(super) fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

pub(super) fn write_delta_distances(out: &mut Vec<u8>, values: &[u32]) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    let mut previous = 0i64;
    for &value in values {
        write_uleb(out, zigzag(value as i64 - previous));
        previous = value as i64;
    }
}

pub(super) fn read_delta_distances(cursor: &mut Cursor<'_>) -> Result<Vec<u32>> {
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

pub(super) fn write_xor_distances(out: &mut Vec<u8>, values: &[u32]) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    let mut previous = 0;
    for &value in values {
        write_uleb(out, (value ^ previous) as u64);
        previous = value;
    }
}

pub(super) fn read_xor_distances(cursor: &mut Cursor<'_>) -> Result<Vec<u32>> {
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

pub(super) fn mru_touch(cache: &mut Vec<u32>, value: u32, capacity: usize) {
    if let Some(index) = cache.iter().position(|&candidate| candidate == value) {
        cache.remove(index);
    }
    cache.insert(0, value);
    cache.truncate(capacity);
}

pub(super) fn write_cached_distances(out: &mut Vec<u8>, values: &[u32], capacity: usize) -> Result<()> {
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

pub(super) fn read_cached_distances(cursor: &mut Cursor<'_>, capacity: usize) -> Result<Vec<u32>> {
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

pub(super) fn dictionary(values: &[u32]) -> Vec<u32> {
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

pub(super) fn write_dictionary_distances(out: &mut Vec<u8>, commands: &[MetadataCommand], group_size: usize) -> Result<usize> {
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

pub(super) fn read_dictionary_distances(cursor: &mut Cursor<'_>) -> Result<(Vec<u32>, usize)> {
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

pub(super) fn pack_distance_blocks(
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

pub(super) fn unpack_distance_blocks(packed: &PackedDistances, model: DistanceModel) -> Result<Vec<Vec<u32>>> {
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

pub(super) fn expected_distances(blocks: &[Vec<MetadataCommand>]) -> Result<Vec<Vec<u32>>> {
    blocks.iter().map(|commands| distance_values(commands)).collect()
}

pub(super) fn checked_distance_result(
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

pub(super) fn checked_distance_results(blocks: &[Vec<MetadataCommand>]) -> Result<Vec<DistanceResult>> {
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

