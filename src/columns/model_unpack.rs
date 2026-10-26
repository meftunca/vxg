#[cfg(test)]
fn unpack_model(bytes: &[u8], count: usize) -> Result<Vec<Vec<u8>>> {
    Ok(unpack_model_mode(bytes, count, true)?.materialize())
}

fn unpack_model_mode(bytes: &[u8], count: usize, strict: bool) -> Result<FlatValues> {
    // Value offsets are u32 ranges into this payload, so the whole payload must
    // fit in u32; every push below is then infallible by construction.
    if bytes.len() > u32::MAX as usize {
        return Err(Error::SizeOverflow);
    }
    let mut cursor = Cursor::new(bytes);
    let tag = cursor.byte()?;
    let per_value = match tag {
        MODEL_HEX_SUFFIX | MODEL_EPOCH_COUNTER | MODEL_PERIODIC_HEX | MODEL_PERIODIC_STRING
        | MODEL_GF2_AFFINE => 24,
        MODEL_SEGMENT_DICT => 48,
        _ => 8,
    };
    let mut values = FlatValues::with_capacity(count, count.saturating_mul(per_value));
    match tag {
        MODEL_RAW => {
            for _ in 0..count {
                let len = cursor.uleb()?;
                let value = cursor.take(len)?;
                values.push_slice(value);
            }
        }
        MODEL_LZ_RAW => {
            let mut probe = Cursor::new(cursor.bytes);
            probe.pos = cursor.pos;
            let raw_len = probe.uleb()?;
            if raw_len == 0 || raw_len > 65536 {
                return Err(invalid("column lz raw length"));
            }
            let mut unpacked = vec![0u8; raw_len];
            unpack_fast_lz_into(&mut cursor, &mut unpacked)?;
            let mut sub = Cursor::new(&unpacked);
            for _ in 0..count {
                let len = sub.uleb()?;
                let value = sub.take(len)?;
                values.push_slice(value);
            }
            if !sub.done() {
                return Err(invalid("column lz raw trailing bytes"));
            }
        }
        MODEL_DICTIONARY | MODEL_FRONT_DICT => {
            let entries_len = cursor.uleb()?;
            if entries_len == 0
                || (tag == MODEL_DICTIONARY && entries_len >= count)
                || entries_len > count
            {
                return Err(invalid("column dictionary cardinality"));
            }
            let mut entries = Vec::with_capacity(entries_len);
            if tag == MODEL_FRONT_DICT {
                if entries_len < 2 || entries_len > 256 {
                    return Err(invalid("column front dict cardinality"));
                }
                let mode_byte = cursor.byte()?;
                let dict_mode = mode_byte & 0x03;
                let idx_mode = mode_byte >> 2;
                if idx_mode > 3 {
                    return Err(invalid("column front dict mode"));
                }
                let width = bit_width((entries_len - 1) as u64);
                fn read_front_entries(
                    dcur: &mut Cursor<'_>,
                    entries_len: usize,
                    entries: &mut Vec<Vec<u8>>,
                ) -> Result<()> {
                    let first = read_bytes(dcur)?;
                    entries.push(first);
                    for _ in 1..entries_len {
                        let lcp = dcur.uleb()?;
                        let lcs = dcur.uleb()?;
                        let mid = read_bytes(dcur)?;
                        let prev: &Vec<u8> = entries.last().unwrap();
                        if lcp + lcs > prev.len() {
                            return Err(invalid("column front dictionary bounds"));
                        }
                        let mut entry = Vec::with_capacity(lcp + mid.len() + lcs);
                        entry.extend_from_slice(&prev[..lcp]);
                        entry.extend_from_slice(&mid);
                        entry.extend_from_slice(&prev[prev.len() - lcs..]);
                        if prev >= &entry {
                            return Err(invalid("column dictionary ordering"));
                        }
                        entries.push(entry);
                    }
                    Ok(())
                }
                if dict_mode == 0 {
                    for _ in 0..entries_len {
                        let entry = read_bytes(&mut cursor)?;
                        if entries.last().is_some_and(|last: &Vec<u8>| last >= &entry) {
                            return Err(invalid("column dictionary ordering"));
                        }
                        entries.push(entry);
                    }
                } else if dict_mode == 1 {
                    read_front_entries(&mut cursor, entries_len, &mut entries)?;
                } else if dict_mode == 2 {
                    let lz_len = cursor.uleb()?;
                    let lz_bytes = cursor.take(lz_len)?;
                    let mut lzc = Cursor::new(lz_bytes);
                    if lzc.byte()? != FAST_LZ_PACKET {
                        return Err(invalid("column front dict lz selector"));
                    }
                    let mut probe = Cursor::new(lzc.bytes);
                    probe.pos = lzc.pos;
                    let raw_len = probe.uleb()?;
                    if raw_len == 0 || raw_len > 65536 {
                        return Err(invalid("column front dict lz length"));
                    }
                    let mut unpacked_dict = vec![0u8; raw_len];
                    unpack_fast_lz_into(&mut lzc, &mut unpacked_dict)?;
                    let mut dcur = Cursor::new(&unpacked_dict);
                    read_front_entries(&mut dcur, entries_len, &mut entries)?;
                    if !dcur.done() {
                        return Err(invalid("column front dict lz trailing"));
                    }
                } else {
                    let (pal, pal_len, pal_mixed) = read_front_dict_palette(&mut cursor)?;
                    let first = read_bytes(&mut cursor)?;
                    entries.push(first);
                    for _ in 1..entries_len {
                        let lcp = cursor.uleb()?;
                        let lcs = cursor.uleb()?;
                        let raw_hdr = cursor.uleb()?;
                        let (mid_len, is_raw) = if pal_mixed {
                            (raw_hdr >> 1, (raw_hdr & 1) != 0)
                        } else {
                            (raw_hdr, false)
                        };
                        if mid_len > 65536 {
                            return Err(invalid("column front dict entry length"));
                        }
                        let packed_mid = if is_raw {
                            cursor.take(mid_len)?
                        } else if pal_len <= 16 {
                            cursor.take((mid_len + 1) >> 1)?
                        } else {
                            cursor.take((mid_len * 5 + 7) >> 3)?
                        };
                        let prev: &Vec<u8> = entries.last().unwrap();
                        if lcp + lcs > prev.len() {
                            return Err(invalid("column front dictionary bounds"));
                        }
                        let total_len = lcp + mid_len + lcs;
                        let mut entry = vec![0u8; total_len];
                        entry[..lcp].copy_from_slice(&prev[..lcp]);
                        if is_raw {
                            entry[lcp..lcp + mid_len].copy_from_slice(packed_mid);
                        } else if mid_len > 0 {
                            unpack_nibble_bytes_into(
                                packed_mid,
                                mid_len,
                                &pal,
                                pal_len,
                                unsafe { entry.as_mut_ptr().add(lcp) },
                            )?;
                        }
                        entry[lcp + mid_len..].copy_from_slice(&prev[prev.len() - lcs..]);
                        if prev >= &entry {
                            return Err(invalid("column dictionary ordering"));
                        }
                        entries.push(entry);
                    }
                }
                let mut u8_arena = Vec::with_capacity(count + 16);
                let u8_offset = unpack_front_dict_indices(
                    &mut cursor,
                    count,
                    entries_len,
                    width,
                    idx_mode,
                    &mut u8_arena,
                )?;
                for i in 0..count {
                    let idx = u8_arena[u8_offset + i] as usize;
                    let entry = entries
                        .get(idx)
                        .ok_or_else(|| invalid("column dictionary index"))?;
                    values.push_slice(entry);
                }
            } else {
                for _ in 0..entries_len {
                    let entry = read_bytes(&mut cursor)?;
                    if entries.last().is_some_and(|last: &Vec<u8>| last >= &entry) {
                        return Err(invalid("column dictionary ordering"));
                    }
                    entries.push(entry);
                }
                let width = cursor.byte()?;
                if width != bit_width((entries_len - 1) as u64) {
                    return Err(invalid("column dictionary width"));
                }
                let packed_len = bit_bytes(count, width)?;
                let mut reader = BitReader::new(cursor.take(packed_len)?);
                for _ in 0..count {
                    let index =
                        usize::try_from(reader.read(width)?).map_err(|_| Error::SizeOverflow)?;
                    let entry = entries
                        .get(index)
                        .ok_or_else(|| invalid("column dictionary index"))?;
                    values.push_slice(entry);
                }
                reader.finish_zero_padded()?;
            }
        }
        MODEL_DECIMAL_AFFINE => {
            if count < 2 {
                return Err(invalid("column affine count"));
            }
            let first = cursor.u64()?;
            let step = cursor.i64()? as i128;
            for index in 0..count {
                let value = first as i128 + step * index as i128;
                let value = u64::try_from(value).map_err(|_| invalid("column affine range"))?;
                push_decimal(&mut values, value);
            }
        }
        MODEL_HEX_SUFFIX => {
            if count < 2 {
                return Err(invalid("column hex count"));
            }
            let prefix = read_bytes(&mut cursor)?;
            let width = cursor.uleb()?;
            if width == 0 || width > 64 {
                return Err(invalid("column hex width"));
            }
            for _ in 0..count {
                let raw = cursor.take(width)?;
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&prefix);
                for &byte in raw {
                    values.bytes.push(HEX[(byte >> 4) as usize]);
                    values.bytes.push(HEX[(byte & 15) as usize]);
                }
                values.bytes.push(b'"');
                values.finish(start);
            }
        }
        MODEL_TIMESTAMP => {
            if count < 2 {
                return Err(invalid("column timestamp count"));
            }
            let year = cursor.u16()?;
            if year > 9999 {
                return Err(invalid("column timestamp year"));
            }
            let plans = (0..5)
                .map(|_| {
                    let base = cursor.byte()?;
                    let width = cursor.byte()?;
                    if width > 6 {
                        return Err(invalid("column timestamp width"));
                    }
                    Ok((base, width))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut components = Vec::with_capacity(5);
            for (base, width) in &plans {
                let packed_len = bit_bytes(count, *width)?;
                let mut reader = BitReader::new(cursor.take(packed_len)?);
                let values = (0..count)
                    .map(|_| {
                        let value = reader.read(*width)?;
                        u8::try_from(value + *base as u64).map_err(|_| invalid("column timestamp range"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                reader.finish_zero_padded()?;
                components.push(values);
            }
            for index in 0..count {
                let (month, day, hour, minute, second) = (
                    components[0][index],
                    components[1][index],
                    components[2][index],
                    components[3][index],
                    components[4][index],
                );
                if !(1..=12).contains(&month)
                    || !(1..=31).contains(&day)
                    || hour >= 24
                    || minute >= 60
                    || second >= 60
                {
                    return Err(invalid("column timestamp component"));
                }
                let start = values.begin();
                values.bytes.push(b'"');
                write_four_digits(&mut values.bytes, year);
                values.bytes.push(b'-');
                write_two_digits(&mut values.bytes, month);
                values.bytes.push(b'-');
                write_two_digits(&mut values.bytes, day);
                values.bytes.push(b'T');
                write_two_digits(&mut values.bytes, hour);
                values.bytes.push(b':');
                write_two_digits(&mut values.bytes, minute);
                values.bytes.push(b':');
                write_two_digits(&mut values.bytes, second);
                values.bytes.extend_from_slice(b"Z\"");
                values.finish(start);
            }
        }
        MODEL_DECIMAL_PACKED => {
            if count < 2 {
                return Err(invalid("column packed decimal count"));
            }
            let min = cursor.u64()?;
            let width = cursor.byte()?;
            if width > 64 {
                return Err(invalid("column packed decimal width"));
            }
            let packed_len = bit_bytes(count, width)?;
            let mut reader = BitReader::new(cursor.take(packed_len)?);
            for _ in 0..count {
                let residual = reader.read(width)?;
                let value = min.checked_add(residual).ok_or(Error::SizeOverflow)?;
                push_decimal(&mut values, value);
            }
            reader.finish_zero_padded()?;
        }
        MODEL_DECIMAL_SUFFIX => {
            if count < 2 {
                return Err(invalid("column decimal suffix count"));
            }
            let prefix = read_bytes(&mut cursor)?;
            let min = cursor.u64()?;
            let width = cursor.byte()?;
            if width > 64 {
                return Err(invalid("column decimal suffix width"));
            }
            let packed_len = bit_bytes(count, width)?;
            let mut reader = BitReader::new(cursor.take(packed_len)?);
            for _ in 0..count {
                let residual = reader.read(width)?;
                let value = min.checked_add(residual).ok_or(Error::SizeOverflow)?;
                let start = values.begin();
                values.bytes.push(b'"');
                values.bytes.extend_from_slice(&prefix);
                write_decimal(&mut values.bytes, value);
                values.bytes.push(b'"');
                values.finish(start);
            }
            reader.finish_zero_padded()?;
        }
        other => unpack_model_ext(other, &mut cursor, count, &mut values)?,
    }
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    if strict {
        let owned = values.materialize();
        let (canonical, _) = pack_model_values(&owned);
        if canonical != bytes {
            return Err(invalid("noncanonical column model"));
        }
    }
    Ok(values)
}

