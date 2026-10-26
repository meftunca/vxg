fn unpack_model_ext(
    tag: u8,
    mut cursor: &mut Cursor<'_>,
    count: usize,
    mut values: &mut FlatValues,
) -> Result<()> {
    match tag {
        MODEL_DECIMAL_DELTA => {
            if count < 2 {
                return Err(invalid("column decimal delta count"));
            }
            let first = cursor.u64()?;
            let width = cursor.byte()?;
            if width > 64 {
                return Err(invalid("column decimal delta width"));
            }
            let packed_len = bit_bytes(count - 1, width)?;
            let mut reader = BitReader::new(cursor.take(packed_len)?);
            push_decimal(values, first);
            let mut previous = first as i128;
            for _ in 1..count {
                let zigzag = reader.read(width)?;
                let delta = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                let current = previous + i128::from(delta);
                let current =
                    u64::try_from(current).map_err(|_| invalid("column decimal delta range"))?;
                push_decimal(&mut values, current);
                previous = i128::from(current);
            }
            reader.finish_zero_padded()?;
        }
        MODEL_PERIODIC_STRING => {
            if count < 2 {
                return Err(invalid("column periodic count"));
            }
            let period = cursor.uleb()?;
            if !(1..=64).contains(&period) {
                return Err(invalid("column periodic period"));
            }
            for _ in 0..count {
                let len = cursor.uleb()?;
                if len > MAX_BLOCK_SIZE {
                    return Err(invalid("column periodic length"));
                }
                let seed_len = len.min(period);
                let seed = cursor.take(seed_len)?;
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(seed);
                for index in seed_len..len {
                    let byte = values.bytes[start as usize + 1 + index % period];
                    values.bytes.push(byte);
                }
                values.bytes.push(b'"');
                values.finish(start);
            }
        }
        MODEL_PERIODIC_HEX => {
            if count < 2 {
                return Err(invalid("column periodic hex count"));
            }
            let period = cursor.uleb()?;
            if !(1..=64).contains(&period) {
                return Err(invalid("column periodic hex period"));
            }
            for _ in 0..count {
                let len = cursor.uleb()?;
                if len > MAX_BLOCK_SIZE {
                    return Err(invalid("column periodic hex length"));
                }
                let seed_len = len.min(period);
                let packed_len = seed_len.div_ceil(2);
                let packed = cursor.take(packed_len)?;
                let start = values.begin();
                values.bytes.push(b'"');
                for (index, &byte) in packed.iter().enumerate() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    if index * 2 + 1 < seed_len {
                        values.bytes.push(HEX[(byte & 15) as usize]);
                    } else if byte & 15 != 0 {
                        return Err(invalid("nonzero periodic hex tail"));
                    }
                }
                for index in seed_len..len {
                    let byte = values.bytes[start as usize + 1 + index % period];
                    values.bytes.push(byte);
                }
                values.bytes.push(b'"');
                values.finish(start);
            }
        }
        MODEL_GF2_AFFINE => {
            if count <= GF2_TRAIN_TRANSITIONS + 1 {
                return Err(invalid("column gf2 recurrence count"));
            }
            let prefix = read_bytes(&mut cursor)?;
            let first = cursor.u64()?;
            let mut coefficients = [0u64; GF2_COEFFICIENTS];
            for coefficient in &mut coefficients {
                *coefficient = cursor.u64()?;
            }
            let mut word = first;
            for _ in 0..count {
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&prefix);
                for byte in word.to_le_bytes() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                values.bytes.push(b'"');
                values.finish(start);
                word = apply_affine(&coefficients, word);
            }
        }
        MODEL_EPOCH_COUNTER => {
            if count < 2 {
                return Err(invalid("column epoch counter count"));
            }
            let literal = read_bytes(&mut cursor)?;
            let prefix_count = cursor.uleb()?;
            if prefix_count == 0 || prefix_count > MAX_EPOCH_PREFIXES {
                return Err(invalid("column epoch prefix count"));
            }
            let mut prefixes = Vec::with_capacity(prefix_count);
            for _ in 0..prefix_count {
                let bytes = cursor.take(4)?;
                let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                if prefixes.last().is_some_and(|last: &u32| *last >= value) {
                    return Err(invalid("column epoch prefix ordering"));
                }
                prefixes.push(value);
            }
            let width = cursor.byte()?;
            if width != bit_width((prefix_count - 1) as u64) {
                return Err(invalid("column epoch prefix width"));
            }
            let index_len = bit_bytes(count, width)?;
            let mut index_reader = BitReader::new(cursor.take(index_len)?);
            let prefix_indices = (0..count)
                .map(|_| {
                    let index = usize::try_from(index_reader.read(width)?)
                        .map_err(|_| Error::SizeOverflow)?;
                    prefixes
                        .get(index)
                        .copied()
                        .ok_or_else(|| invalid("column epoch prefix index"))
                })
                .collect::<Result<Vec<_>>>()?;
            index_reader.finish_zero_padded()?;
            let flag_len = bit_bytes(count, 1)?;
            let mut flag_reader = BitReader::new(cursor.take(flag_len)?);
            let flags = (0..count)
                .map(|_| flag_reader.read(1))
                .collect::<Result<Vec<_>>>()?;
            flag_reader.finish_zero_padded()?;
            if flags.first() != Some(&1) {
                return Err(invalid("column epoch first explicit"));
            }
            let explicit_count = cursor.uleb()?;
            if explicit_count != flags.iter().filter(|&&flag| flag == 1).count() {
                return Err(invalid("column epoch explicit count"));
            }
            let mut explicit = Vec::with_capacity(explicit_count);
            for _ in 0..explicit_count {
                let value = cursor.uleb()?;
                if value > u32::MAX as usize {
                    return Err(invalid("column epoch suffix range"));
                }
                explicit.push(value as u32);
            }
            let mut explicit = explicit.into_iter();
            let mut previous = 0u64;
            for row in 0..count {
                let suffix = if flags[row] == 1 {
                    u64::from(
                        explicit
                            .next()
                            .ok_or_else(|| invalid("column epoch explicit order"))?,
                    )
                } else {
                    previous
                        .checked_add(1)
                        .ok_or_else(|| invalid("column epoch sequential"))?
                };
                if suffix > u64::from(u32::MAX) {
                    return Err(invalid("column epoch suffix range"));
                }
                let prefix = prefix_indices[row];
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&literal);
                for byte in prefix.to_be_bytes() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                for byte in (suffix as u32).to_be_bytes() {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                values.bytes.push(b'"');
                values.finish(start);
                previous = suffix;
            }
        }
        MODEL_SEGMENT_DICT => {
            if count < 2 {
                return Err(invalid("column segment dictionary count"));
            }
            let dictionary_len = cursor.uleb()?;
            if dictionary_len > MAX_SEGMENT_DICT {
                return Err(invalid("column segment dictionary size"));
            }
            let mut dictionary = Vec::with_capacity(dictionary_len);
            for _ in 0..dictionary_len {
                let entry = read_bytes(&mut cursor)?;
                if entry.is_empty()
                    || entry.len() > MAX_SEGMENT_LEN
                    || entry.iter().any(|&byte| !segment_byte_ok(byte))
                    || canonical_segment_decimal(&entry).is_some()
                {
                    return Err(invalid("column segment dictionary entry"));
                }
                if dictionary.last().is_some_and(|last: &Vec<u8>| last >= &entry) {
                    return Err(invalid("column segment dictionary ordering"));
                }
                dictionary.push(entry);
            }
            let width = cursor.byte()?;
            if width != bit_width((dictionary_len + 1) as u64) {
                return Err(invalid("column segment code width"));
            }
            let code_count = cursor.uleb()?;
            if code_count == 0
                || code_count > count.saturating_mul(MAX_SEGMENTS_PER_VALUE + 1)
            {
                return Err(invalid("column segment code count"));
            }
            let code_len = bit_bytes(code_count, width)?;
            let mut code_reader = BitReader::new(cursor.take(code_len)?);
            let codes = (0..code_count)
                .map(|_| code_reader.read(width))
                .collect::<Result<Vec<_>>>()?;
            code_reader.finish_zero_padded()?;
            let decimal_count = cursor.uleb()?;
            if decimal_count > code_count {
                return Err(invalid("column segment decimal count"));
            }
            let decimal_width = cursor.byte()?;
            if decimal_width > 30 {
                return Err(invalid("column segment decimal width"));
            }
            let decimal_len = bit_bytes(decimal_count, decimal_width)?;
            let mut decimal_reader = BitReader::new(cursor.take(decimal_len)?);
            let decimals = (0..decimal_count)
                .map(|_| decimal_reader.read(decimal_width))
                .collect::<Result<Vec<_>>>()?;
            decimal_reader.finish_zero_padded()?;
            if decimals.iter().any(|&value| value > 999_999_999) {
                return Err(invalid("column segment decimal range"));
            }
            if bit_width(decimals.iter().copied().max().unwrap_or(0)) != decimal_width {
                return Err(invalid("column segment decimal width"));
            }
            let pad_width = cursor.byte()?;
            if pad_width > 12 {
                return Err(invalid("column segment pad width"));
            }
            let pad_len = bit_bytes(count, pad_width)?;
            let mut pad_reader = BitReader::new(cursor.take(pad_len)?);
            let pads = (0..count)
                .map(|_| pad_reader.read(pad_width))
                .collect::<Result<Vec<_>>>()?;
            pad_reader.finish_zero_padded()?;
            if pads.iter().any(|&pad| pad > MAX_PAD as u64) {
                return Err(invalid("column segment pad"));
            }
            if bit_width(pads.iter().copied().max().unwrap_or(0)) != pad_width {
                return Err(invalid("column segment pad width"));
            }
            let end_code = dictionary_len as u64;
            let decimal_code = dictionary_len as u64 + 1;
            if codes.iter().any(|&code| code > decimal_code) {
                return Err(invalid("column segment code"));
            }
            if codes.iter().filter(|&&code| code == end_code).count() != count {
                return Err(invalid("column segment terminator count"));
            }
            if codes.iter().filter(|&&code| code == decimal_code).count() != decimals.len() {
                return Err(invalid("column segment decimal usage"));
            }
            let mut decimals = decimals.into_iter();
            let mut pads = pads.into_iter();
            let mut start = values.begin();
            let mut open = false;
            let mut segment_count = 0usize;
            for code in codes {
                if code == end_code {
                    if !open {
                        values.bytes.push(b'"');
                    }
                    let pad = pads.next().ok_or_else(|| invalid("column segment pad order"))?;
                    values.bytes.extend(std::iter::repeat(b' ').take(pad as usize));
                    values.bytes.push(b'"');
                    values.finish(start);
                    open = false;
                    segment_count = 0;
                    start = values.begin();
                } else {
                    if open {
                        values.bytes.extend_from_slice(b"; ");
                    } else {
                        values.bytes.push(b'"');
                        open = true;
                    }
                    if code == decimal_code {
                        let decimal = decimals
                            .next()
                            .ok_or_else(|| invalid("column segment decimal order"))?;
                        write_decimal(&mut values.bytes, decimal);
                    } else {
                        let entry = dictionary
                            .get(code as usize)
                            .ok_or_else(|| invalid("column segment index"))?;
                        values.bytes.extend_from_slice(entry);
                    }
                    segment_count += 1;
                    if segment_count > MAX_SEGMENTS_PER_VALUE {
                        return Err(invalid("column segment count"));
                    }
                }
            }
            if open {
                return Err(invalid("column segment missing terminator"));
            }
        }
        _ => return Err(invalid("column model selector")),
    }
    Ok(())
}

