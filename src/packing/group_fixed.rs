use std::{cmp::Reverse, collections::{BTreeMap, BinaryHeap}};
use super::*;

pub(super) fn plans(commands: &[MetadataCommand]) -> Result<[FieldPlan; FIELD_COUNT]> {
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

pub(super) fn write_plan_header(
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

pub(super) fn read_plan_header(cursor: &mut Cursor<'_>) -> Result<(usize, [FieldPlan; FIELD_COUNT])> {
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

pub(super) fn residuals(commands: &[MetadataCommand], field: usize, plan: FieldPlan) -> Result<Vec<u32>> {
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

pub(super) fn write_exceptions(out: &mut Vec<u8>, residuals: &[u32], plan: FieldPlan) -> Result<()> {
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

pub(super) fn write_fixed_field(out: &mut Vec<u8>, residuals: &[u32], plan: FieldPlan) -> Result<()> {
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

pub(super) fn write_bitsliced_field(out: &mut Vec<u8>, residuals: &[u32], plan: FieldPlan) -> Result<()> {
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

pub(super) fn read_exceptions(
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

pub(super) fn read_fixed_field(cursor: &mut Cursor<'_>, count: usize, plan: FieldPlan) -> Result<Vec<u32>> {
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

pub(super) fn read_bitsliced_field(
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

pub(super) fn commands_from_fields(
    values: [[u32; 128]; FIELD_COUNT],
    count: usize,
) -> Result<Vec<MetadataCommand>> {
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        out.push(command(std::array::from_fn(|field| values[field][index]))?);
    }
    Ok(out)
}

pub(super) fn pack_fixed_group(commands: &[MetadataCommand], bitsliced: bool) -> Result<Vec<u8>> {
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

pub(super) fn unpack_fixed_group(cursor: &mut Cursor<'_>, bitsliced: bool) -> Result<Vec<MetadataCommand>> {
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


pub(super) fn reverse_low_bits(value: u32, len: u8) -> u32 {
    if len == 32 {
        value.reverse_bits()
    } else {
        value.reverse_bits() >> (32 - len)
    }
}

impl HuffmanCodebook {
    pub(super) fn from_lengths(mut lengths: Vec<(u32, u8)>) -> Result<Self> {
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

    pub(super) fn from_values(values: &[u32]) -> Result<Self> {
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

    pub(super) fn write_symbol(&self, writer: &mut BitWriter, symbol: u32) -> Result<()> {
        let &(reversed, len) = self
            .by_symbol
            .get(&symbol)
            .ok_or_else(|| invalid("Huffman symbol"))?;
        writer.push(reversed, len)
    }

    pub(super) fn read_symbol(&self, reader: &mut BitReader<'_>) -> Result<u32> {
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

pub(super) fn write_huffman_field(out: &mut Vec<u8>, values: &[u32]) -> Result<()> {
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

pub(super) fn read_huffman_field(cursor: &mut Cursor<'_>, count: usize) -> Result<Vec<u32>> {
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

