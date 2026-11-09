// Per-column compressed-domain unpackers pushing `GRAMMAR_PACKET` column models
// (`MODEL_DECIMAL_PACKED`, `MODEL_SCALED_DECIMAL`, `MODEL_DECIMAL_DELTA`, `MODEL_DECIMAL_AFFINE`,
// `MODEL_DICTIONARY`, `MODEL_FRONT_DICT`, and general `PreparedColumn` streams) directly into
// `ColBuilder` vectors without ASCII formatting or per-row text re-parsing.

#[derive(Clone, Copy)]
enum PreParsedDictTok {
    Null,
    Bool(bool),
    I64(i64),
    Dec(crate::ops::DecNum),
    Utf8 { off: u32, len: u32 },
}

fn classify_dict_tok(tok: &[u8], is_json: bool, utf8_arena: &mut Vec<u8>) -> PreParsedDictTok {
    use crate::ops::{decode_delim_str, parse_canonical_u64, unescape_json_str, DecNum};
    if is_json {
        if tok == b"null" {
            return PreParsedDictTok::Null;
        }
        if tok == b"true" {
            return PreParsedDictTok::Bool(true);
        }
        if tok == b"false" {
            return PreParsedDictTok::Bool(false);
        }
        if tok.first() == Some(&b'"') {
            if let Some(unesc) = unescape_json_str(tok) {
                let off = utf8_arena.len() as u32;
                utf8_arena.extend_from_slice(&unesc);
                return PreParsedDictTok::Utf8 {
                    off,
                    len: (utf8_arena.len() as u32) - off,
                };
            }
        }
        if let Some(u) = parse_canonical_u64(tok) {
            return match i64::try_from(u) {
                Ok(iv) => PreParsedDictTok::I64(iv),
                Err(_) => PreParsedDictTok::Dec(DecNum::from_u64(u)),
            };
        }
        if let Some(d) = DecNum::parse(tok) {
            return PreParsedDictTok::Dec(d);
        }
        let off = utf8_arena.len() as u32;
        utf8_arena.extend_from_slice(tok);
        PreParsedDictTok::Utf8 {
            off,
            len: (utf8_arena.len() as u32) - off,
        }
    } else {
        if tok.is_empty() {
            return PreParsedDictTok::Utf8 { off: 0, len: 0 };
        }
        let is_quoted = tok.first() == Some(&b'"');
        let Some(dec) = decode_delim_str(tok) else {
            let off = utf8_arena.len() as u32;
            utf8_arena.extend_from_slice(tok);
            return PreParsedDictTok::Utf8 {
                off,
                len: (utf8_arena.len() as u32) - off,
            };
        };
        if !is_quoted {
            if let Some(u) = parse_canonical_u64(&dec) {
                return match i64::try_from(u) {
                    Ok(iv) => PreParsedDictTok::I64(iv),
                    Err(_) => PreParsedDictTok::Dec(DecNum::from_u64(u)),
                };
            }
            if let Some(d) = DecNum::parse(&dec) {
                return PreParsedDictTok::Dec(d);
            }
        }
        let off = utf8_arena.len() as u32;
        utf8_arena.extend_from_slice(&dec);
        PreParsedDictTok::Utf8 {
            off,
            len: (utf8_arena.len() as u32) - off,
        }
    }
}

#[inline(always)]
fn push_preparsed_dict_tok(
    sink: &mut crate::arrow_builder::ColBuilder,
    entry: PreParsedDictTok,
    utf8_arena: &[u8],
) {
    match entry {
        PreParsedDictTok::Null => sink.push_null(),
        PreParsedDictTok::Bool(b) => sink.push_bool(b),
        PreParsedDictTok::I64(v) => sink.push_i64(v),
        PreParsedDictTok::Dec(d) => sink.push_dec(d),
        PreParsedDictTok::Utf8 { off, len } => {
            let s = off as usize;
            sink.push_utf8_bytes(&utf8_arena[s..s + len as usize]);
        }
    }
}

fn try_push_decimal_packed_arrow(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    digit_delim: bool,
    sink: &mut crate::arrow_builder::ColBuilder,
) -> Result<bool> {
    if digit_delim || ones < 2 || col_bytes.first() != Some(&MODEL_DECIMAL_PACKED) {
        return Ok(false);
    }
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(true);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let min = cursor.u64()?;
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column packed decimal width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed_bytes = cursor.take(packed_len)?;
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let _ = FastBitReader::new_validated(packed_bytes, ones, width)?;
    let max_r = if width == 0 {
        0
    } else if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let Some(hi) = min.checked_add(max_r) else {
        return Ok(false);
    };
    if hi > i64::MAX as u64 {
        return Ok(false);
    }
    let min_i64 = min as i64;
    if width == 0 {
        BitReader::new(packed_bytes).finish_zero_padded()?;
        sink.push_i64_repeat(min_i64, pop);
        return Ok(true);
    }
    sink.reserve_rows(pop);
    match mask {
        TplMask::Empty => {}
        TplMask::All => {
            if width <= 16 {
                let mut buf = [0u16; 64];
                let mut row = 0usize;
                while row < ones {
                    let cnt = (ones - row).min(64);
                    unpack_u16_chunk64(packed_bytes, row, cnt, width, &mut buf);
                    for &r16 in &buf[..cnt] {
                        sink.push_i64(min_i64 + i64::from(r16));
                    }
                    row += cnt;
                }
            } else if width <= 32 {
                let mut buf = [0u32; 64];
                let mut row = 0usize;
                while row < ones {
                    let cnt = (ones - row).min(64);
                    unpack_u32_chunk64(packed_bytes, row, cnt, width, &mut buf);
                    for &r32 in &buf[..cnt] {
                        sink.push_i64(min_i64 + i64::from(r32));
                    }
                    row += cnt;
                }
            } else {
                let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
                for _ in 0..ones {
                    sink.push_i64(min_i64 + (rdr.next_u64() as i64));
                }
            }
        }
        TplMask::Bits(bs, _) => {
            let mut buf = [0u16; 64];
            for (wi, &w0) in bs.words().iter().enumerate() {
                let mut w = w0;
                if w == 0 {
                    continue;
                }
                let base = wi << 6;
                if width <= 16 && w.count_ones() >= 16 {
                    let cnt = (ones - base).min(64);
                    unpack_u16_chunk64(packed_bytes, base, cnt, width, &mut buf);
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        sink.push_i64(min_i64 + i64::from(buf[bit]));
                        w &= w - 1;
                    }
                } else {
                    while w != 0 {
                        let row = base + w.trailing_zeros() as usize;
                        let r = read_packed_bits_at(packed_bytes, row * width as usize, width);
                        sink.push_i64(min_i64 + (r as i64));
                        w &= w - 1;
                    }
                }
            }
        }
    }
    Ok(true)
}

fn try_push_scaled_dec_arrow(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    digit_delim: bool,
    sink: &mut crate::arrow_builder::ColBuilder,
) -> Result<bool> {
    use crate::ops::DecNum;
    if digit_delim || ones < 2 || col_bytes.first() != Some(&MODEL_SCALED_DECIMAL) {
        return Ok(false);
    }
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(true);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let scale = cursor.byte()?;
    if scale > 9 {
        return Err(invalid("column scaled decimal scale"));
    }
    let min_v = cursor.i64()?;
    if scale == 0 && min_v >= 0 {
        return Err(invalid("column scaled decimal noncanonical unsigned"));
    }
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column scaled decimal width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed_bytes = cursor.take(packed_len)?;
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let _ = FastBitReader::new_validated(packed_bytes, ones, width)?;
    let max_r = if width == 0 {
        0
    } else if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if (min_v as i128) + (max_r as i128) > (i64::MAX as i128) {
        return Err(invalid("column scaled decimal overflow"));
    }
    if width == 0 {
        BitReader::new(packed_bytes).finish_zero_padded()?;
        if scale == 0 {
            sink.push_i64_repeat(min_v, pop);
        } else {
            sink.push_dec_repeat(DecNum::new(min_v as i128, scale), pop);
        }
        return Ok(true);
    }
    sink.reserve_rows(pop);
    let min_i128 = min_v as i128;
    match mask {
        TplMask::Empty => {}
        TplMask::All => {
            if width <= 16 {
                let mut buf = [0u16; 64];
                let mut row = 0usize;
                while row < ones {
                    let cnt = (ones - row).min(64);
                    unpack_u16_chunk64(packed_bytes, row, cnt, width, &mut buf);
                    if scale == 0 {
                        for &r16 in &buf[..cnt] {
                            sink.push_i64((min_i128 + i128::from(r16)) as i64);
                        }
                    } else {
                        for &r16 in &buf[..cnt] {
                            sink.push_dec(DecNum::new(min_i128 + i128::from(r16), scale));
                        }
                    }
                    row += cnt;
                }
            } else {
                let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
                if scale == 0 {
                    for _ in 0..ones {
                        sink.push_i64((min_i128 + (rdr.next_u64() as i128)) as i64);
                    }
                } else {
                    for _ in 0..ones {
                        sink.push_dec(DecNum::new(min_i128 + (rdr.next_u64() as i128), scale));
                    }
                }
            }
        }
        TplMask::Bits(bs, _) => {
            for (wi, &w0) in bs.words().iter().enumerate() {
                let mut w = w0;
                let base = wi << 6;
                while w != 0 {
                    let row = base + w.trailing_zeros() as usize;
                    let r = read_packed_bits_at(packed_bytes, row * width as usize, width);
                    if scale == 0 {
                        sink.push_i64((min_i128 + (r as i128)) as i64);
                    } else {
                        sink.push_dec(DecNum::new(min_i128 + (r as i128), scale));
                    }
                    w &= w - 1;
                }
            }
        }
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn push_slot_column_masked(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    target_col: crate::ops::ColTarget<'_>,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
    sink: &mut crate::arrow_builder::ColBuilder,
) -> Result<()> {
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(());
    }
    let digit_delim = matches!(
        target_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    );
    if try_push_decimal_packed_arrow(col_bytes, ones, mask, digit_delim, sink)?
        || try_push_scaled_dec_arrow(col_bytes, ones, mask, digit_delim, sink)?
    {
        return Ok(());
    }

    reset_scratch_to(scratch, raw_mark, slice_mark);
    ensure_col_raw_arena(col_bytes, hdr_raw_len, &mut scratch.raw_arena);
    let pc = PreparedColumn::parse_and_validate(col_bytes, ones, scratch)?;

    match pc.const_full {
        ConstValue::Slice(s) => {
            sink.push_raw_token_repeat(s, pop);
            return Ok(());
        }
        ConstValue::Inline { ref buf, len } => {
            sink.push_raw_token_repeat(&buf[..len as usize], pop);
            return Ok(());
        }
        ConstValue::QuotedDecimal {
            prefix,
            ref buf,
            len,
        } => {
            let mut full = Vec::with_capacity(prefix.len() + len as usize + 2);
            full.push(b'"');
            full.extend_from_slice(prefix);
            full.extend_from_slice(&buf[..len as usize]);
            full.push(b'"');
            sink.push_raw_token_repeat(&full, pop);
            return Ok(());
        }
        ConstValue::None => {}
    }

    if !digit_delim && matches!(pc.const_lead, LeadValue::Empty) && pc.const_trail.is_empty() {
        match pc.stream {
            ColumnStream::DecimalSmall { u16_offset } => {
                sink.reserve_rows(pop);
                match mask {
                    TplMask::All => {
                        for &v in &scratch.u16_arena[u16_offset..u16_offset + ones] {
                            sink.push_i64(i64::from(v));
                        }
                    }
                    TplMask::Bits(bs, _) => {
                        for (wi, &w0) in bs.words().iter().enumerate() {
                            let mut w = w0;
                            let base = wi << 6;
                            while w != 0 {
                                let row = base + w.trailing_zeros() as usize;
                                sink.push_i64(i64::from(scratch.u16_arena[u16_offset + row]));
                                w &= w - 1;
                            }
                        }
                    }
                    TplMask::Empty => {}
                }
                return Ok(());
            }
            ColumnStream::DecimalValues { u64_offset } => {
                sink.reserve_rows(pop);
                let push_u64 = |sink: &mut crate::arrow_builder::ColBuilder, v: u64| match i64::try_from(v) {
                    Ok(iv) => sink.push_i64(iv),
                    Err(_) => sink.push_dec(crate::ops::DecNum::from_u64(v)),
                };
                match mask {
                    TplMask::All => {
                        for &v in &scratch.u64_arena[u64_offset..u64_offset + ones] {
                            push_u64(sink, v);
                        }
                    }
                    TplMask::Bits(bs, _) => {
                        for (wi, &w0) in bs.words().iter().enumerate() {
                            let mut w = w0;
                            let base = wi << 6;
                            while w != 0 {
                                let row = base + w.trailing_zeros() as usize;
                                push_u64(sink, scratch.u64_arena[u64_offset + row]);
                                w &= w - 1;
                            }
                        }
                    }
                    TplMask::Empty => {}
                }
                return Ok(());
            }
            ColumnStream::DecimalAffineSmall { current, step }
            | ColumnStream::DecimalAffine { current, step } => {
                sink.reserve_rows(pop);
                let push_u64 = |sink: &mut crate::arrow_builder::ColBuilder, v: u64| match i64::try_from(v) {
                    Ok(iv) => sink.push_i64(iv),
                    Err(_) => sink.push_dec(crate::ops::DecNum::from_u64(v)),
                };
                match mask {
                    TplMask::All => {
                        let mut cur = current;
                        for _ in 0..ones {
                            push_u64(sink, cur);
                            cur = cur.wrapping_add(step);
                        }
                    }
                    TplMask::Bits(bs, _) => {
                        for (wi, &w0) in bs.words().iter().enumerate() {
                            let mut w = w0;
                            let base = wi << 6;
                            while w != 0 {
                                let row = base + w.trailing_zeros() as usize;
                                push_u64(sink, current.wrapping_add(step.wrapping_mul(row as u64)));
                                w &= w - 1;
                            }
                        }
                    }
                    TplMask::Empty => {}
                }
                return Ok(());
            }
            _ => {}
        }
    }

    if !digit_delim {
        if let ColumnStream::Dictionary { u8_offset, .. } = pc.stream {
            if let Some((slice_off, entries)) = pc.dict_slices {
                if entries <= 256 {
                    let mut utf8_arena = Vec::with_capacity(entries * 16);
                    let mut lut = [PreParsedDictTok::Null; 256];
                    for (e, slot) in lut.iter_mut().enumerate().take(entries) {
                        let (ptr, len) = scratch.slice_arena[slice_off + e];
                        let s = unsafe { std::slice::from_raw_parts(ptr, len) };
                        *slot = classify_dict_tok(s, sink.is_json, &mut utf8_arena);
                    }
                    sink.reserve_rows(pop);
                    match mask {
                        TplMask::All => {
                            for &idx in &scratch.u8_arena[u8_offset..u8_offset + ones] {
                                push_preparsed_dict_tok(sink, lut[idx as usize], &utf8_arena);
                            }
                        }
                        TplMask::Bits(bs, _) => {
                            for (wi, &w0) in bs.words().iter().enumerate() {
                                let mut w = w0;
                                let base = wi << 6;
                                while w != 0 {
                                    let row = base + w.trailing_zeros() as usize;
                                    let idx = scratch.u8_arena[u8_offset + row] as usize;
                                    push_preparsed_dict_tok(sink, lut[idx], &utf8_arena);
                                    w &= w - 1;
                                }
                            }
                        }
                        TplMask::Empty => {}
                    }
                    return Ok(());
                }
            }
        }
    }

    if matches!(pc.stream, ColumnStream::Raw { .. }) {
        ensure_full_raw_arena(hdr_raw_len, &mut scratch.raw_arena);
    }
    let unbound = UnboundTemplate::build(&[b"", b""], &[pc], scratch);
    let n_exec = scratch.unbound_col_arena.len();
    scratch.exec_col_arena.reserve(n_exec);
    let mut tpl = unbound.bind(scratch);
    let mut val_buf = vec![0u8; tpl.row_budget.max(512) + 64];
    sink.reserve_rows(pop);
    match mask {
        TplMask::All => {
            for _ in 0..ones {
                let written = unsafe {
                    let sp = val_buf.as_mut_ptr();
                    let ep = tpl.emit_row(sp);
                    ep.offset_from(sp) as usize
                };
                sink.push_raw_token(&val_buf[..written]);
            }
        }
        TplMask::Bits(bs, _) => {
            let mut cur_t = 0usize;
            for (wi, &w0) in bs.words().iter().enumerate() {
                let mut w = w0;
                let base = wi << 6;
                while w != 0 {
                    let ti = base + w.trailing_zeros() as usize;
                    let written = unsafe {
                        tpl.skip_rows(ti - cur_t);
                        let sp = val_buf.as_mut_ptr();
                        let ep = tpl.emit_row(sp);
                        ep.offset_from(sp) as usize
                    };
                    cur_t = ti + 1;
                    sink.push_raw_token(&val_buf[..written]);
                    w &= w - 1;
                }
            }
        }
        TplMask::Empty => {}
    }
    Ok(())
}
