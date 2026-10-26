use super::*;

pub(super) fn bases(commands: &[MetadataCommand], fields_to_pack: &[usize]) -> Result<Vec<u32>> {
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

pub(super) fn write_int4_field(
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

pub(super) fn read_int4_field(cursor: &mut Cursor<'_>, count: usize, base: u32) -> Result<Vec<u32>> {
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

pub(super) fn pack_int4_group(commands: &[MetadataCommand]) -> Result<Vec<u8>> {
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

pub(super) fn unpack_int4_group(cursor: &mut Cursor<'_>) -> Result<Vec<MetadataCommand>> {
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
pub(super) enum Selector {
    TwoBit,
    Base3,
}

pub(super) fn selector_class(op: u8) -> Result<u8> {
    match op {
        0 => Ok(0),
        1 | 2 | 7 => Ok(1),
        3..=6 => Ok(2),
        _ => Err(invalid("ternary opcode")),
    }
}

pub(super) fn subcode(op: u8) -> Result<u8> {
    match op {
        0 => Ok(0),
        1 => Ok(0),
        2 => Ok(1),
        7 => Ok(2),
        3..=6 => Ok(op - 3),
        _ => Err(invalid("ternary opcode")),
    }
}

pub(super) fn opcode(class: u8, subcode: u8) -> Result<u8> {
    match (class, subcode) {
        (0, 0) => Ok(0),
        (1, 0) => Ok(1),
        (1, 1) => Ok(2),
        (1, 2) => Ok(7),
        (2, 0..=3) => Ok(3 + subcode),
        _ => Err(invalid("non-canonical ternary subcode")),
    }
}

pub(super) fn write_selectors(
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

pub(super) fn read_selectors(cursor: &mut Cursor<'_>, count: usize, selector: Selector) -> Result<Vec<u8>> {
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

pub(super) fn pack_ternary_group(commands: &[MetadataCommand], selector: Selector) -> Result<Vec<u8>> {
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

pub(super) fn unpack_ternary_group(
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

pub(super) fn pack_group(layout: Layout, commands: &[MetadataCommand]) -> Result<Vec<u8>> {
    match layout {
        Layout::Fixed => pack_fixed_group(commands, false),
        Layout::Bitsliced => pack_fixed_group(commands, true),
        Layout::Int4 => pack_int4_group(commands),
        Layout::Ternary2Bit => pack_ternary_group(commands, Selector::TwoBit),
        Layout::TernaryBase3 => pack_ternary_group(commands, Selector::Base3),
    }
}

pub(super) fn unpack_group(cursor: &mut Cursor<'_>, layout: Layout) -> Result<Vec<MetadataCommand>> {
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

