struct OpsGrammarHeader<'a> {
    raw_len: usize,
    p_after_raw_len: usize,
    template_count: usize,
    slot_counts: [usize; 16],
    tpl0_delim: Option<u8>,
    tpl0_segs: [&'a [u8]; MAX_SLOTS + 1],
    extra_tpl_segs: Vec<Vec<&'a [u8]>>,
    row_count: usize,
    selector_width: u8,
    selector_slice: &'a [u8],
    tpl_row_counts: [usize; 16],
    raw_count: usize,
    first_is_raw: bool,
    last_is_raw: bool,
    cursor: Cursor<'a>,
}

#[inline(always)]
fn popcount_bytes(slice: &[u8]) -> usize {
    let mut chunks = slice.chunks_exact(8);
    let mut total = 0usize;
    for c in &mut chunks {
        total += u64::from_ne_bytes(c.try_into().unwrap()).count_ones() as usize;
    }
    for &b in chunks.remainder() {
        total += b.count_ones() as usize;
    }
    total
}

impl<'a> OpsGrammarHeader<'a> {
    #[inline]
    fn segs(&self, tpl_idx: usize) -> &[&'a [u8]] {
        let slots = self.slot_counts[tpl_idx];
        if tpl_idx == 0 {
            &self.tpl0_segs[..=slots]
        } else {
            &self.extra_tpl_segs[tpl_idx - 1][..=slots]
        }
    }
}

fn parse_grammar_for_ops<'a>(payload: &'a [u8]) -> Result<Option<OpsGrammarHeader<'a>>> {
    if payload.first() != Some(&GRAMMAR_PACKET) {
        return Ok(None);
    }
    let mut cursor = Cursor::new(&payload[1..]);
    let raw_len = cursor.uleb()?;
    if raw_len > MAX_BLOCK_SIZE {
        return Err(invalid("column raw length"));
    }
    let p_after_raw_len = 1 + cursor.pos;
    let (template_count, slot_counts, tpl0_delim, tpl0_segs, extra_tpl_segs) =
        if let Some(hit) = try_match_cached_grammar_wire(&payload[p_after_raw_len..], &mut cursor) {
            hit
        } else {
            let start_pos = cursor.pos;
            let template_count = cursor.uleb()?;
            if template_count > MAX_TEMPLATES {
                return Err(invalid("column template count"));
            }
            if template_count == 0 || template_count > 16 {
                return Ok(None);
            }
            let dictionary_len = cursor.uleb()?;
            if dictionary_len == 0 || dictionary_len > template_count.saturating_mul(MAX_SLOTS + 1)
            {
                return Err(invalid("column segment dictionary count"));
            }
            let mut seg_dict_inline = [&[][..]; 128];
            let mut seg_dict_overflow = if dictionary_len > 128 {
                vec![&[][..]; dictionary_len]
            } else {
                Vec::new()
            };
            let seg_dict: &mut [&'a [u8]] = if dictionary_len <= 128 {
                &mut seg_dict_inline[..dictionary_len]
            } else {
                &mut seg_dict_overflow[..dictionary_len]
            };
            for i in 0..dictionary_len {
                let len = cursor.uleb()?;
                let seg = cursor.take(len)?;
                if i > 0 && seg_dict[i - 1] >= seg {
                    return Err(invalid("column segment dictionary ordering"));
                }
                seg_dict[i] = seg;
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
            let mut tpl0_segs = [&[][..]; MAX_SLOTS + 1];
            let mut extra_tpl_segs = Vec::with_capacity(template_count.saturating_sub(1));
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
                let mut reader =
                    FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
                if t_idx == 0 {
                    for s in 0..=slots {
                        let idx = reader.next_u64() as usize;
                        if idx >= dictionary_len {
                            return Err(invalid("column segment dictionary index"));
                        }
                        tpl0_segs[s] = seg_dict[idx];
                    }
                    if !tpl0_segs[slots].ends_with(b"\n") {
                        return Ok(None);
                    }
                } else {
                    let mut segs = Vec::with_capacity(slots + 1);
                    for _ in 0..=slots {
                        let idx = reader.next_u64() as usize;
                        if idx >= dictionary_len {
                            return Err(invalid("column segment dictionary index"));
                        }
                        segs.push(seg_dict[idx]);
                    }
                    if !segs[slots].ends_with(b"\n") {
                        return Ok(None);
                    }
                    extra_tpl_segs.push(segs);
                }
            }
            let wire_len = cursor.pos - start_pos;
            let tpl0_delim = delimited_template_delim(&tpl0_segs[..=slot_counts[0]]);
            store_cached_grammar_wire(
                &payload[p_after_raw_len..p_after_raw_len + wire_len],
                template_count,
                &slot_counts,
                tpl0_delim,
                &tpl0_segs,
                &extra_tpl_segs,
            );
            (
                template_count,
                slot_counts,
                tpl0_delim,
                tpl0_segs,
                extra_tpl_segs,
            )
        };
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
    let mut tpl_row_counts = [0usize; 16];
    let (raw_count, first_is_raw, last_is_raw) = if row_count == 0 {
        let _ = FastBitReader::new_validated(selector_slice, 0, selector_width)?;
        (0, false, false)
    } else if template_count == 1 {
        let _ = FastBitReader::new_validated(selector_slice, row_count, 1)?;
        let ones = popcount_bytes(selector_slice);
        tpl_row_counts[0] = ones;
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
            } else {
                tpl_row_counts[sel - 1] += 1;
            }
        }
        (r_cnt, f_raw, l_raw)
    };
    Ok(Some(OpsGrammarHeader {
        raw_len,
        p_after_raw_len,
        template_count,
        slot_counts,
        tpl0_delim,
        tpl0_segs,
        extra_tpl_segs,
        row_count,
        selector_width,
        selector_slice,
        tpl_row_counts,
        raw_count,
        first_is_raw,
        last_is_raw,
        cursor,
    }))
}

fn nth_zero_selector(
    selector_slice: &[u8],
    row_count: usize,
    selector_width: u8,
    target_zero: usize,
) -> usize {
    if selector_width == 1 {
        return nth_selector_bit(selector_slice, row_count, false, target_zero);
    }
    let mut rdr = FastBitReader::new_validated(selector_slice, row_count, selector_width).unwrap();
    let mut z = 0usize;
    for r in 0..row_count {
        if rdr.next_u64() == 0 {
            if z == target_zero {
                return r;
            }
            z += 1;
        }
    }
    row_count
}

#[allow(clippy::too_many_arguments)]
fn advance_search_carry(
    hdr: &OpsGrammarHeader<'_>,
    raw_slice_offset: usize,
    block_index: usize,
    base_row: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    pred: &crate::ops::ColPred<'_>,
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
    scratch: &DecodeScratch,
) -> (
    usize,
    usize,
    Option<(usize, usize, usize, Vec<u8>)>,
    Option<usize>,
) {
    let mut raw_start = 0usize;
    if let Some((c_blk, c_row, c_glob, mut c_bytes)) = carry.take() {
        if hdr.first_is_raw && hdr.raw_count > 0 {
            let (ptr, len) = scratch.slice_arena[raw_slice_offset];
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            c_bytes.extend_from_slice(slice);
            raw_start = 1;
            if !c_bytes.ends_with(b"\n") && hdr.row_count == 1 {
                *carry = Some((c_blk, c_row, c_glob, c_bytes));
                return (1, 1, None, Some(0));
            }
        }
        if crate::ops::line_matches_col_pred(&c_bytes, pred) && out.len() < limit {
            out.push(crate::query::RowMatch {
                block_index: c_blk,
                row_index: c_row,
                global_row_index: c_glob,
                bytes: c_bytes,
            });
        }
    }
    let mut raw_end = hdr.raw_count;
    let mut next_carry = None;
    if hdr.last_is_raw && raw_end > raw_start {
        let (ptr, len) = scratch.slice_arena[raw_slice_offset + hdr.raw_count - 1];
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        if !slice.ends_with(b"\n") {
            raw_end = hdr.raw_count - 1;
            next_carry = Some((
                block_index,
                hdr.row_count - 1,
                base_row + hdr.row_count - 1,
                slice.to_vec(),
            ));
        }
    }
    (raw_start, raw_end, next_carry, None)
}

#[allow(clippy::too_many_arguments)]
fn finish_raw_only_search(
    hdr: &OpsGrammarHeader<'_>,
    raw_slice_offset: usize,
    block_index: usize,
    global_row_base: usize,
    carry: &mut Option<(usize, usize, usize, Vec<u8>)>,
    pred: &crate::ops::ColPred<'_>,
    limit: usize,
    out: &mut Vec<crate::query::RowMatch>,
    scratch: &DecodeScratch,
) -> Result<Option<usize>> {
    let had_incoming_carry = carry.is_some();
    let base_row = global_row_base - usize::from(had_incoming_carry);
    let (raw_start, raw_end, next_carry, early_ret) = advance_search_carry(
        hdr,
        raw_slice_offset,
        block_index,
        base_row,
        carry,
        pred,
        limit,
        out,
        scratch,
    );
    if let Some(ret) = early_ret {
        return Ok(Some(ret));
    }
    let new_rows = hdr.row_count - usize::from(had_incoming_carry);
    for r_idx in raw_start..raw_end {
        if out.len() >= limit {
            break;
        }
        let (ptr, len) = scratch.slice_arena[raw_slice_offset + r_idx];
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        if crate::ops::line_matches_col_pred(slice, pred) {
            let row_idx =
                nth_zero_selector(hdr.selector_slice, hdr.row_count, hdr.selector_width, r_idx);
            out.push(crate::query::RowMatch {
                block_index,
                row_index: row_idx,
                global_row_index: base_row + row_idx,
                bytes: slice.to_vec(),
            });
        }
    }
    *carry = next_carry;
    Ok(Some(new_rows))
}

struct TplColSlices<'a> {
    inline: [&'a [u8]; 64],
    overflow: Vec<&'a [u8]>,
    starts: [u16; 17],
}

impl<'a> TplColSlices<'a> {
    #[inline(always)]
    fn get(&self, t: usize) -> &[&'a [u8]] {
        let s = self.starts[t] as usize;
        let e = self.starts[t + 1] as usize;
        if self.overflow.is_empty() {
            &self.inline[s..e]
        } else {
            &self.overflow[s..e]
        }
    }
}

fn read_all_template_col_slices<'a>(hdr: &mut OpsGrammarHeader<'a>) -> Result<TplColSlices<'a>> {
    let total_slots: usize = hdr.slot_counts[..hdr.template_count].iter().sum();
    let mut out = TplColSlices {
        inline: [&[]; 64],
        overflow: if total_slots > 64 { Vec::with_capacity(total_slots) } else { Vec::new() },
        starts: [0u16; 17],
    };
    let mut pos = 0usize;
    for t in 0..hdr.template_count {
        out.starts[t] = pos as u16;
        for _ in 0..hdr.slot_counts[t] {
            let len = hdr.cursor.uleb()?;
            let s = hdr.cursor.take(len)?;
            if total_slots <= 64 { out.inline[pos] = s; } else { out.overflow.push(s); }
            pos += 1;
        }
    }
    out.starts[hdr.template_count] = pos as u16;
    if !hdr.cursor.done() {
        return Err(invalid("column packet trailing bytes"));
    }
    Ok(out)
}

#[inline(always)]
fn ensure_raw_rows_arena(hdr: &OpsGrammarHeader<'_>, raw_arena: &mut Vec<u8>) {
    ensure_proj_raw_arena(hdr, 0, raw_arena);
}

#[inline(always)]
fn ensure_proj_raw_arena(hdr: &OpsGrammarHeader<'_>, proj_cols: usize, raw_arena: &mut Vec<u8>) {
    let need = hdr
        .raw_len
        .max(65_536)
        .saturating_mul(6 + proj_cols.saturating_mul(2))
        .saturating_add(262_144);
    if raw_arena.capacity() < need {
        debug_assert_eq!(raw_arena.len(), 0);
        raw_arena.reserve(need.saturating_sub(raw_arena.len()));
    }
}

#[inline(always)]
fn ensure_full_raw_arena(raw_len: usize, raw_arena: &mut Vec<u8>) {
    if raw_arena.is_empty() {
        let need = raw_len.max(65_536).saturating_mul(6) + 262_144;
        if raw_arena.capacity() < need {
            raw_arena.reserve(need);
        }
    }
}

#[inline(always)]
fn ensure_col_raw_arena(col_bytes: &[u8], raw_len: usize, raw_arena: &mut Vec<u8>) {
    if raw_arena.is_empty() && col_bytes.first().is_some() {
        ensure_full_raw_arena(raw_len, raw_arena);
    }
}

pub(crate) fn split_grammar_packet_streams(
    payload: &[u8],
) -> Option<(usize, Vec<(usize, usize)>)> {
    let mut hdr = parse_grammar_for_ops(payload).ok()??;
    for _ in 0..hdr.raw_count {
        let len = hdr.cursor.uleb().ok()?;
        let _ = hdr.cursor.take(len).ok()?;
    }
    let stream0_len = 1 + hdr.cursor.pos;
    let total_slots: usize = hdr.slot_counts[..hdr.template_count].iter().sum();
    if total_slots == 0 || total_slots > 64 {
        return None;
    }
    let mut spans = Vec::with_capacity(total_slots);
    for _ in 0..total_slots {
        let start = 1 + hdr.cursor.pos;
        let len = hdr.cursor.uleb().ok()?;
        let _ = hdr.cursor.take(len).ok()?;
        let end = 1 + hdr.cursor.pos;
        spans.push((start, end));
    }
    if !hdr.cursor.done() {
        return None;
    }
    Some((stream0_len, spans))
}

pub(crate) fn resolve_stream0_required_slots(
    stream0: &[u8],
    targets: &[crate::ops::ColTarget<'_>],
) -> Option<u64> {
    let hdr = parse_grammar_for_ops(stream0).ok()??;
    let total_slots: usize = hdr.slot_counts[..hdr.template_count].iter().sum();
    if total_slots == 0 || total_slots > 64 {
        return None;
    }
    let mut mask = 0u64;
    let mut slot_base = 0usize;
    for t in 0..hdr.template_count {
        let segs = hdr.segs(t);
        let slots = hdr.slot_counts[t];
        if !segs.last().is_some_and(|s| s.ends_with(b"\n")) {
            return None;
        }
        let tpl_delim = if t == 0 {
            hdr.tpl0_delim
        } else {
            delimited_template_delim(segs)
        };
        for &col in targets {
            match col {
                crate::ops::ColTarget::JsonKey(key) => {
                    let found = json_single_line_template_slot(segs, key.as_bytes())?;
                    if let Some(s) = found {
                        mask |= 1u64 << (slot_base + s);
                    }
                }
                crate::ops::ColTarget::Delimited { delim, index } => {
                    if tpl_delim != Some(delim) {
                        return None;
                    }
                    if index < slots {
                        mask |= 1u64 << (slot_base + index);
                    }
                }
            }
        }
        slot_base += slots;
    }
    Some(mask)
}
