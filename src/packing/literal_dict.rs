use std::{collections::BTreeMap, time::Instant};
use super::*;

pub(super) fn pack_token_dictionary(literals: &[u8], limit: usize) -> Result<Option<(Vec<u8>, usize)>> {
    if literals.len() < TOKEN_WIDTH * 2 || !TOKEN_DICTIONARY_LIMITS.contains(&limit) {
        return Ok(None);
    }
    let mut frequencies = BTreeMap::new();
    for start in 0..=literals.len() - TOKEN_WIDTH {
        let token = u64::from_le_bytes(
            literals[start..start + TOKEN_WIDTH].try_into().unwrap(),
        );
        let count = frequencies.entry(token).or_insert(0usize);
        *count = count.checked_add(1).ok_or(Error::SizeOverflow)?;
    }
    let mut ranked = frequencies
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_token, left_count), (right_token, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| left_token.cmp(right_token))
    });
    ranked.truncate(limit);
    if ranked.is_empty() {
        return Ok(None);
    }
    let mut tokens = ranked.into_iter().map(|(token, _)| token).collect::<Vec<_>>();
    tokens.sort_unstable();
    let (mut packed, mut used) = encode_token_dictionary(literals, &tokens)?;
    if used.iter().any(|&used| !used) {
        tokens = tokens
            .into_iter()
            .zip(used)
            .filter_map(|(token, used)| used.then_some(token))
            .collect();
        if tokens.is_empty() {
            return Ok(None);
        }
        (packed, used) = encode_token_dictionary(literals, &tokens)?;
    }
    if used.iter().any(|&used| !used) {
        return Err(invalid("token dictionary selection"));
    }
    Ok(Some((packed, tokens.len())))
}

pub(super) fn encode_token_dictionary(literals: &[u8], tokens: &[u64]) -> Result<(Vec<u8>, Vec<bool>)> {
    if tokens.is_empty() || tokens.len() > 128 || !tokens.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(invalid("token dictionary selection"));
    }
    let mut ids = BTreeMap::new();
    for (index, &token) in tokens.iter().enumerate() {
        ids.insert(token, u8::try_from(index).map_err(|_| Error::SizeOverflow)?);
    }
    let mut packed = Vec::with_capacity(
        2usize
            .checked_add(tokens.len().checked_mul(TOKEN_WIDTH).ok_or(Error::SizeOverflow)?)
            .and_then(|value| value.checked_add(literals.len()))
            .ok_or(Error::SizeOverflow)?,
    );
    packed.push(2);
    packed.push(u8::try_from(tokens.len()).map_err(|_| Error::SizeOverflow)?);
    for token in tokens {
        packed.extend_from_slice(&token.to_le_bytes());
    }
    let mut used = vec![false; tokens.len()];
    let mut pos = 0usize;
    while pos < literals.len() {
        let token = if pos + TOKEN_WIDTH <= literals.len() {
            let value = u64::from_le_bytes(
                literals[pos..pos + TOKEN_WIDTH].try_into().unwrap(),
            );
            ids.get(&value).copied()
        } else {
            None
        };
        if let Some(id) = token {
            packed.push(0x80 | id);
            used[id as usize] = true;
            pos += TOKEN_WIDTH;
            continue;
        }
        let start = pos;
        while pos < literals.len() && pos - start < 128 {
            let token = if pos + TOKEN_WIDTH <= literals.len() {
                let value = u64::from_le_bytes(
                    literals[pos..pos + TOKEN_WIDTH].try_into().unwrap(),
                );
                ids.contains_key(&value)
            } else {
                false
            };
            if token {
                break;
            }
            pos += 1;
        }
        let count = pos - start;
        if count == 0 {
            return Err(invalid("token literal progress"));
        }
        packed.push(u8::try_from(count - 1).map_err(|_| Error::SizeOverflow)?);
        packed.extend_from_slice(&literals[start..pos]);
    }
    Ok((packed, used))
}

pub(super) fn unpack_token_dictionary(cursor: &mut Cursor<'_>, literal_len: usize) -> Result<Vec<u8>> {
    let count = cursor.byte()? as usize;
    if count == 0 || count > 128 {
        return Err(invalid("token dictionary count"));
    }
    let mut tokens = Vec::with_capacity(count);
    let mut previous = None;
    for _ in 0..count {
        let token = u64::from_le_bytes(cursor.take(TOKEN_WIDTH)?.try_into().unwrap());
        if previous.is_some_and(|value| token <= value) {
            return Err(invalid("non-canonical token dictionary"));
        }
        previous = Some(token);
        tokens.push(token);
    }
    let mut used = vec![false; count];
    let mut out = Vec::with_capacity(literal_len);
    while out.len() < literal_len {
        let tag = cursor.byte()?;
        if tag & 0x80 != 0 {
            let index = (tag & 0x7f) as usize;
            let token = *tokens
                .get(index)
                .ok_or_else(|| invalid("token dictionary reference"))?;
            let end = out
                .len()
                .checked_add(TOKEN_WIDTH)
                .ok_or(Error::SizeOverflow)?;
            if end > literal_len {
                return Err(invalid("token dictionary exceeds literal length"));
            }
            used[index] = true;
            out.extend_from_slice(&token.to_le_bytes());
        } else {
            let count = (tag as usize)
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?;
            let end = out
                .len()
                .checked_add(count)
                .ok_or(Error::SizeOverflow)?;
            if end > literal_len {
                return Err(invalid("token raw run exceeds literal length"));
            }
            out.extend_from_slice(cursor.take(count)?);
        }
    }
    if used.iter().any(|&used| !used) {
        return Err(invalid("unused token dictionary entry"));
    }
    if !cursor.done() {
        return Err(invalid("token literal trailing bytes"));
    }
    Ok(out)
}

pub(super) fn pack_literal_entropy_blocks(blocks: &[Vec<u8>]) -> Result<PackedLiteralEntropy> {
    let mut packed_blocks = Vec::with_capacity(blocks.len());
    let mut literal_lengths = Vec::with_capacity(blocks.len());
    let mut literal_bytes = 0usize;
    let mut encoded_bytes = 0usize;
    let mut max_block_scratch_bytes = 0usize;
    let mut dictionary_entries = 0usize;
    let mut entropy_blocks = 0usize;
    let mut dictionary_blocks = 0usize;
    let mut raw_blocks = 0usize;
    for literals in blocks {
        let mut raw = Vec::with_capacity(
            literals
                .len()
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?,
        );
        raw.push(0);
        raw.extend_from_slice(literals);
        let entropy = if literals.is_empty() {
            None
        } else {
            let values = literals.iter().map(|&value| value as u32).collect::<Vec<_>>();
            let mut packed = Vec::new();
            packed.push(1);
            write_huffman_field(&mut packed, &values)?;
            Some(packed)
        };
        let mut candidates = vec![(0u8, raw, 0usize)];
        if let Some(entropy) = entropy {
            candidates.push((1, entropy, 0));
        }
        for limit in TOKEN_DICTIONARY_LIMITS {
            if let Some((dictionary, entries)) = pack_token_dictionary(literals, limit)? {
                candidates.push((2, dictionary, entries));
            }
        }
        let selection_scratch_bytes = candidates.iter().try_fold(0usize, |total, candidate| {
            total
                .checked_add(candidate.1.len())
                .ok_or(Error::SizeOverflow)
        })?;
        let mut candidates = candidates.into_iter();
        let mut chosen = candidates
            .next()
            .ok_or_else(|| invalid("literal candidate"))?;
        for candidate in candidates {
            if candidate.1.len() < chosen.1.len() {
                chosen = candidate;
            }
        }
        match chosen.0 {
            0 => raw_blocks += 1,
            1 => entropy_blocks += 1,
            2 => {
                dictionary_blocks += 1;
                dictionary_entries = dictionary_entries
                    .checked_add(chosen.2)
                    .ok_or(Error::SizeOverflow)?;
            }
            _ => return Err(invalid("literal candidate")),
        }
        literal_bytes = literal_bytes
            .checked_add(literals.len())
            .ok_or(Error::SizeOverflow)?;
        encoded_bytes = encoded_bytes
            .checked_add(chosen.1.len())
            .ok_or(Error::SizeOverflow)?;
        max_block_scratch_bytes = max_block_scratch_bytes.max(selection_scratch_bytes);
        literal_lengths.push(literals.len());
        packed_blocks.push(chosen.1);
    }
    Ok(PackedLiteralEntropy {
        blocks: packed_blocks,
        literal_lengths,
        literal_bytes,
        encoded_bytes,
        max_block_scratch_bytes,
        dictionary_entries,
        entropy_blocks,
        dictionary_blocks,
        raw_blocks,
    })
}

pub(super) fn unpack_literal_entropy_blocks(packed: &PackedLiteralEntropy) -> Result<Vec<Vec<u8>>> {
    if packed.blocks.len() != packed.literal_lengths.len() {
        return Err(invalid("literal block lengths"));
    }
    packed
        .blocks
        .iter()
        .zip(&packed.literal_lengths)
        .map(|(bytes, &literal_len)| {
            let mut cursor = Cursor::new(bytes);
            let selector = cursor.byte()?;
            match selector {
                0 => {
                    let out = cursor.take(literal_len)?.to_vec();
                    if !cursor.done() {
                        return Err(invalid("literal raw trailing bytes"));
                    }
                    Ok(out)
                }
                1 => {
                    if literal_len == 0 {
                        return Err(invalid("literal empty entropy block"));
                    }
                    let values = read_huffman_field(&mut cursor, literal_len)?;
                    if !cursor.done() {
                        return Err(invalid("literal entropy trailing bytes"));
                    }
                    values
                        .into_iter()
                        .map(|value| {
                            u8::try_from(value)
                                .map_err(|_| invalid("literal entropy byte range"))
                        })
                        .collect()
                }
                2 => unpack_token_dictionary(&mut cursor, literal_len),
                _ => Err(invalid("literal selector")),
            }
        })
        .collect()
}

pub(super) fn checked_literal_entropy_result(blocks: &[Vec<u8>]) -> Result<LiteralEntropyResult> {
    let packed = pack_literal_entropy_blocks(blocks)?;
    if unpack_literal_entropy_blocks(&packed)? != blocks {
        return Err(invalid("literal entropy roundtrip"));
    }
    Ok(LiteralEntropyResult {
        literal_bytes: packed.literal_bytes,
        encoded_bytes: packed.encoded_bytes,
        max_block_scratch_bytes: packed.max_block_scratch_bytes,
        dictionary_entries: packed.dictionary_entries,
        entropy_blocks: packed.entropy_blocks,
        dictionary_blocks: packed.dictionary_blocks,
        raw_blocks: packed.raw_blocks,
        pack_ns: None,
        unpack_ns: None,
    })
}

/// Times raw-versus-Huffman literal selection plus exact literal reconstruction.
/// The block's decoded literal length comes from the already validated command
/// stream, just as it would in a joined command/literal format.
pub fn time_literal_entropy(
    blocks: &[Vec<u8>],
    iterations: usize,
) -> Result<LiteralEntropyResult> {
    if iterations == 0 {
        return Err(Error::InvalidOption("iterations must be at least one"));
    }
    let verified = checked_literal_entropy_result(blocks)?;
    let pack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= pack_literal_entropy_blocks(blocks)?.encoded_bytes;
    }
    black_box(checksum);
    let pack_elapsed = pack_start.elapsed();
    let packed = pack_literal_entropy_blocks(blocks)?;
    let unpack_start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        checksum ^= unpack_literal_entropy_blocks(&packed)?
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    }
    black_box(checksum);
    Ok(LiteralEntropyResult {
        pack_ns: Some(pack_elapsed.as_nanos()),
        unpack_ns: Some(unpack_start.elapsed().as_nanos()),
        ..verified
    })
}

/// Frame convenience wrapper for the literal prototype timing. RAW blocks are
/// deliberately excluded because they remain raw passthrough payloads in the
/// complete-artifact accounting.
pub fn time_frame_literal_entropy(
    frame: &Frame<'_>,
    iterations: usize,
) -> Result<LiteralEntropyResult> {
    let blocks = literal_blocks(frame)?;
    let lz_blocks = frame
        .blocks
        .iter()
        .zip(blocks)
        .filter_map(|(block_info, literals)| {
            (block_info.info.kind != block::RAW).then_some(literals)
        })
        .collect::<Vec<_>>();
    time_literal_entropy(&lz_blocks, iterations)
}

