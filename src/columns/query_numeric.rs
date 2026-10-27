/// Parse a canonical unsigned decimal; `None` if it is not plain ASCII digits that fit in `u64`.
fn parse_u64_digits(b: &[u8]) -> Option<u64> {
    if b.is_empty() || b.len() > 20 || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut v = 0u64;
    for &d in b {
        v = v.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    Some(v)
}

/// Column-aware numeric filter for a single-template block with no raw rows.
///
/// Only the target column is decoded to evaluate the predicate; the other columns are decoded
/// (and rows materialized) only when at least one row matches. Returns `Ok(None)` whenever the
/// block shape is not handled so the caller can use the line-based fallback.
#[allow(clippy::too_many_arguments)]
fn numeric_scan_single_template<'a>(
    nf: &crate::query::NumFilter,
    segs: &[&'a [u8]],
    col_slices: &[&'a [u8]],
    ones: usize,
    scratch: &mut DecodeScratch,
    block_index: usize,
    base_row: usize,
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
) -> Result<Option<usize>> {
    let slots = col_slices.len();
    let Some(target) = (0..slots).find(|&i| segs[i].ends_with(&nf.marker)) else {
        return Ok(None);
    };
    let rem = limit.saturating_sub(out.len());
    if rem == 0 {
        return Ok(Some(ones));
    }
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
    if hits.is_empty() {
        return Ok(Some(ones));
    }
    let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
    let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
    for s in 0..slots {
        cols[s] = if s == target {
            col
        } else {
            PreparedColumn::parse_and_validate(col_slices[s], ones, scratch)?
        };
    }
    let unbound = UnboundTemplate::build(segs, &cols[..slots], scratch);
    let unbound_len = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(unbound_len);
    let mut tpl = unbound.bind(scratch);
    let mut row_buf = vec![0u8; tpl.row_budget.max(4096) + 64];
    let mut cur = 0usize;
    for &t in &hits {
        unsafe {
            tpl.skip_rows(t - cur);
            let sp = row_buf.as_mut_ptr();
            let ep = tpl.emit_row(sp);
            let w = ep.offset_from(sp) as usize;
            cur = t + 1;
            out.push(crate::query::RowMatch {
                block_index,
                row_index: t,
                global_row_index: base_row + t,
                bytes: row_buf[..w].to_vec(),
            });
        }
    }
    Ok(Some(ones))
}

