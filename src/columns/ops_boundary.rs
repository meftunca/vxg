pub(crate) struct FastGrammarBoundary<'a> {
    pub row_count: usize,
    pub first_raw: Option<std::borrow::Cow<'a, [u8]>>,
    pub last_unterminated_raw: Option<std::borrow::Cow<'a, [u8]>>,
}

pub(crate) struct IndepGrammarBoundary<'a> {
    pub row_count: usize,
    pub first_is_raw: bool,
    pub first_raw: Option<std::borrow::Cow<'a, [u8]>>,
    pub last_unterminated_raw: Option<std::borrow::Cow<'a, [u8]>>,
}

/// Block-independent boundary inspector for parallel `ops` and `query` pre-passes.
/// Does not depend on `had_incoming_carry`: always extracts `first_raw` when `first_is_raw`
/// (zero-alloc borrowed slice on >99.9% of blocks), allowing all `nb` blocks to be inspected
/// concurrently across worker threads.
pub(crate) fn peek_grammar_boundary_indep<'a>(
    payload: &'a [u8],
) -> Result<Option<IndepGrammarBoundary<'a>>> {
    use std::borrow::Cow;

    if payload.first() != Some(&GRAMMAR_PACKET) {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&payload[1..]);
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE {
        return Err(invalid("column raw length"));
    }
    let template_count = cursor.uleb()?;
    if template_count > MAX_TEMPLATES {
        return Err(invalid("column template count"));
    }
    if template_count == 0 || template_count > 16 {
        return Ok(None);
    }
    let dictionary_len = cursor.uleb()?;
    if dictionary_len == 0 || dictionary_len > template_count.saturating_mul(MAX_SLOTS + 1) {
        return Err(invalid("column segment dictionary count"));
    }
    let mut seg_ends_nl_inline = [false; 128];
    let mut seg_ends_nl_overflow = if dictionary_len > 128 {
        vec![false; dictionary_len]
    } else {
        Vec::new()
    };
    let seg_ends_nl: &mut [bool] = if dictionary_len <= 128 {
        &mut seg_ends_nl_inline[..dictionary_len]
    } else {
        &mut seg_ends_nl_overflow[..dictionary_len]
    };
    let mut prev_seg: Option<&[u8]> = None;
    for flag in seg_ends_nl.iter_mut() {
        let len = cursor.uleb()?;
        let seg = cursor.take(len)?;
        if prev_seg.is_some_and(|p| p >= seg) {
            return Err(invalid("column segment dictionary ordering"));
        }
        prev_seg = Some(seg);
        *flag = seg.last() == Some(&b'\n');
    }
    let mut slot_counts = [0usize; 16];
    for slot_out in slot_counts.iter_mut().take(template_count) {
        let slots = cursor.uleb()?;
        if slots > MAX_SLOTS {
            return Err(invalid("column slot count"));
        }
        *slot_out = slots;
    }
    let segment_width = cursor.byte()?;
    if segment_width != bit_width(dictionary_len.saturating_sub(1) as u64) {
        return Err(invalid("column segment selector width"));
    }
    let mut packed_slices = [&[][..]; 16];
    for t_idx in 0..template_count {
        let slots = slot_counts[t_idx];
        let packed_len = bit_bytes(slots + 1, segment_width)?;
        let packed_slice = cursor.take(packed_len)?;
        for prev in 0..t_idx {
            if slot_counts[prev] == slots && packed_slices[prev] == packed_slice {
                return Err(invalid("duplicate column template"));
            }
        }
        packed_slices[t_idx] = packed_slice;
        let mut rdr = FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
        rdr.skip(slots);
        let last_idx = rdr.next_u64() as usize;
        if last_idx >= dictionary_len {
            return Err(invalid("column segment dictionary index"));
        }
        if !seg_ends_nl[last_idx] {
            return Ok(None);
        }
    }
    let row_count = cursor.uleb()?;
    if row_count > raw_len.saturating_add(1) {
        return Err(invalid("column row count"));
    }
    let selector_width = cursor.byte()?;
    if selector_width != bit_width(template_count as u64) {
        return Err(invalid("column row selector width"));
    }
    let selector_len = bit_bytes(row_count, selector_width)?;
    let selector_slice = cursor.take(selector_len)?;
    if row_count == 0 {
        let _ = FastBitReader::new_validated(selector_slice, 0, selector_width)?;
        return Ok(Some(IndepGrammarBoundary {
            row_count: 0,
            first_is_raw: false,
            first_raw: None,
            last_unterminated_raw: None,
        }));
    }
    let (raw_count, first_is_raw, last_is_raw) = if template_count == 1 {
        let _ = FastBitReader::new_validated(selector_slice, row_count, 1)?;
        let ones: usize = selector_slice.iter().map(|b| b.count_ones() as usize).sum();
        let f_raw = (selector_slice[0] & 1) == 0;
        let l_raw = ((selector_slice[(row_count - 1) >> 3] >> ((row_count - 1) & 7)) & 1) == 0;
        (row_count - ones, f_raw, l_raw)
    } else {
        let mut rdr = FastBitReader::new_validated(selector_slice, row_count, selector_width)?;
        let mut r_cnt = 0usize;
        let mut f_raw = false;
        let mut l_raw = false;
        for i in 0..row_count {
            let sel = rdr.next_u64() as usize;
            if sel > template_count {
                return Err(invalid("column row selector index"));
            }
            if sel == 0 {
                r_cnt += 1;
                if i == 0 {
                    f_raw = true;
                }
                if i + 1 == row_count {
                    l_raw = true;
                }
            }
        }
        (r_cnt, f_raw, l_raw)
    };
    if (!first_is_raw && !last_is_raw) || raw_count == 0 {
        return Ok(Some(IndepGrammarBoundary {
            row_count,
            first_is_raw,
            first_raw: None,
            last_unterminated_raw: None,
        }));
    }
    if cursor.bytes.get(cursor.pos) != Some(&0) {
        let mut first_raw = None;
        let mut last_unterminated_raw = None;
        for i in 0..raw_count {
            let len = cursor.uleb()?;
            let raw = cursor.take(len)?;
            if raw.is_empty() {
                return Err(invalid("empty raw column row"));
            }
            if i == 0 && first_is_raw {
                first_raw = Some(Cow::Borrowed(raw));
                if !last_is_raw {
                    break;
                }
            }
            if i + 1 == raw_count && last_is_raw && !raw.ends_with(b"\n") {
                last_unterminated_raw = Some(Cow::Borrowed(raw));
            }
        }
        return Ok(Some(IndepGrammarBoundary {
            row_count,
            first_is_raw,
            first_raw,
            last_unterminated_raw,
        }));
    }
    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.raw_arena.reserve(raw_len.saturating_mul(2) + 16384);
        let off = {
            let DecodeScratch {
                ref mut raw_arena,
                ref mut slice_arena,
                ..
            } = *scratch;
            read_grammar_raw_rows(&mut cursor, raw_count, raw_arena, slice_arena)?
        };
        let first_raw = if first_is_raw {
            let (p, l) = scratch.slice_arena[off];
            Some(Cow::Owned(unsafe { std::slice::from_raw_parts(p, l) }.to_vec()))
        } else {
            None
        };
        let last_unterminated_raw = if last_is_raw {
            let (p, l) = scratch.slice_arena[off + raw_count - 1];
            let s = unsafe { std::slice::from_raw_parts(p, l) };
            (!s.ends_with(b"\n")).then(|| Cow::Owned(s.to_vec()))
        } else {
            None
        };
        Ok(Some(IndepGrammarBoundary {
            row_count,
            first_is_raw,
            first_raw,
            last_unterminated_raw,
        }))
    })
}

/// Zero-alloc $O(\text{header})$ boundary inspector for sequential / 1-thread `ops` and `query`.
#[inline]
pub(crate) fn peek_grammar_boundary_fast<'a>(
    payload: &'a [u8],
    had_incoming_carry: bool,
) -> Result<Option<FastGrammarBoundary<'a>>> {
    let Some(indep) = peek_grammar_boundary_indep(payload)? else {
        return Ok(None);
    };
    if had_incoming_carry && !indep.first_is_raw && indep.row_count > 0 {
        return Ok(None);
    }
    Ok(Some(FastGrammarBoundary {
        row_count: indep.row_count,
        first_raw: if had_incoming_carry {
            indep.first_raw
        } else {
            None
        },
        last_unterminated_raw: indep.last_unterminated_raw,
    }))
}
