/// Parse a canonical unsigned decimal; `None` if it is not plain ASCII digits that fit in `u64`.
fn parse_u64_digits(b: &[u8]) -> Option<u64> {
    if b.is_empty() || b.len() > 20 || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if b.len() > 1 && b[0] == b'0' {
        // Not canonical (matches the line-based fallback, which rejects leading zeros).
        return None;
    }
    let mut v = 0u64;
    for &d in b {
        v = v.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    Some(v)
}

/// Column-aware numeric predicate for the template rows of a single-template block.
///
/// Only the target column is decoded to evaluate the predicate. Returns the target slot, its
/// parsed column and the matching template-row indices (at most `rem`). Returns `Ok(None)`
/// whenever the column shape is not handled; the caller then verifies rows one by one.
fn numeric_hits<'a>(
    nf: &crate::query::NumFilter,
    segs: &[&'a [u8]],
    col_slices: &[&'a [u8]],
    ones: usize,
    rem: usize,
    scratch: &mut DecodeScratch,
) -> Result<Option<(usize, PreparedColumn<'a>, Vec<usize>)>> {
    let slots = col_slices.len();
    let Some(target) = (0..slots).find(|&i| segs[i].ends_with(&nf.marker)) else {
        return Ok(None);
    };
    let col = PreparedColumn::parse_and_validate(col_slices[target], ones, scratch)?;
    if !matches!(col.const_lead, LeadValue::Empty) {
        return Ok(None);
    }
    let mut hits: Vec<usize> = Vec::new();
    match col.const_full {
        ConstValue::Inline { ref buf, len } => {
            let Some(v) = parse_u64_digits(&buf[..len as usize]) else {
                return Ok(None);
            };
            if nf.cmp.test(v) {
                hits.extend(0..ones.min(rem));
            }
        }
        ConstValue::Slice(s) => {
            let Some(v) = parse_u64_digits(s) else {
                return Ok(None);
            };
            if nf.cmp.test(v) {
                hits.extend(0..ones.min(rem));
            }
        }
        ConstValue::QuotedDecimal { .. } => return Ok(None),
        ConstValue::None => match col.stream {
            ColumnStream::DecimalSmall { u16_offset } => {
                for t in 0..ones {
                    if nf.cmp.test(u64::from(scratch.u16_arena[u16_offset + t])) {
                        hits.push(t);
                        if hits.len() >= rem {
                            break;
                        }
                    }
                }
            }
            ColumnStream::Dictionary { u8_offset, .. } => {
                // Evaluate the predicate once per dictionary entry, then scan the id array.
                let Some((slice_off, entries)) = col.dict_slices else {
                    return Ok(None);
                };
                let mut valid = [false; 256];
                for e in 0..entries.min(256) {
                    let (ptr, len) = scratch.slice_arena[slice_off + e];
                    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
                    valid[e] = parse_u64_digits(bytes).is_some_and(|v| nf.cmp.test(v));
                }
                let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];
                for (t, &id) in ids.iter().enumerate() {
                    if valid[id as usize] {
                        hits.push(t);
                        if hits.len() >= rem {
                            break;
                        }
                    }
                }
            }
            ColumnStream::DecimalValues { u64_offset } => {
                for t in 0..ones {
                    if nf.cmp.test(scratch.u64_arena[u64_offset + t]) {
                        hits.push(t);
                        if hits.len() >= rem {
                            break;
                        }
                    }
                }
            }
            ColumnStream::DecimalAffineSmall { current, step }
            | ColumnStream::DecimalAffine { current, step } => {
                for t in 0..ones {
                    let v = current.wrapping_add(step.wrapping_mul(t as u64));
                    if nf.cmp.test(v) {
                        hits.push(t);
                        if hits.len() >= rem {
                            break;
                        }
                    }
                }
            }
            ColumnStream::DecimalPacked { min, mut reader } => {
                for t in 0..ones {
                    if nf.cmp.test(min.wrapping_add(reader.next_u64())) {
                        hits.push(t);
                        if hits.len() >= rem {
                            break;
                        }
                    }
                }
            }
            _ => return Ok(None),
        },
    }
    Ok(Some((target, col, hits)))
}
