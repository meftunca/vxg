enum TplFilter {
    Never,
    Always,
    DictSingle {
        u8_offset: usize,
        valid_ids: Box<[bool; 256]>,
        single_id: Option<u8>,
    },
    CheckAll,
    /// Exact template-row indices already selected by a column-level predicate.
    Hits(Vec<usize>),
}

fn classify_template_filter(
    unbound: &UnboundTemplate,
    active_dict_lens: &[usize],
    scratch: &DecodeScratch,
    needle: &[u8],
) -> TplFilter {
    if needle.is_empty() {
        return TplFilter::Always;
    }
    let head_slice = &scratch.str_arena[unbound.head_range.0..unbound.head_range.1];
    if contains_subslice(head_slice, needle) {
        return TplFilter::Always;
    }
    let active_cols = &scratch.unbound_col_arena[unbound.col_start..unbound.col_end];
    let mut non_dict_col_mask = [0u64; 4];
    let mut static_mask = [0u64; 4];
    add_bytes_to_mask(&mut static_mask, head_slice);

    let seg_overlaps = |seg: &[u8]| -> bool {
        if seg.is_empty() {
            return false;
        }
        let max_p = (needle.len() - 1).min(seg.len());
        for p in 1..=max_p {
            if seg.ends_with(&needle[..p]) || seg.starts_with(&needle[needle.len() - p..]) {
                return true;
            }
        }
        contains_subslice(needle, seg)
    };
    let mut overlaps_any_seg = seg_overlaps(head_slice);
    for (idx, ucol) in active_cols.iter().enumerate() {
        let after_slice = &scratch.str_arena[ucol.after_range.0..ucol.after_range.1];
        if contains_subslice(after_slice, needle) {
            return TplFilter::Always;
        }
        if !overlaps_any_seg
            && ((idx + 1 < active_cols.len() && after_slice.is_empty()) || seg_overlaps(after_slice))
        {
            overlaps_any_seg = true;
        }
        add_bytes_to_mask(&mut static_mask, after_slice);
    }

    for ucol in active_cols {
        match ucol.stream {
            ColumnStream::Dictionary { .. } => {}
            ColumnStream::DecimalAffineSmall { .. }
            | ColumnStream::DecimalAffine { .. }
            | ColumnStream::DecimalSmall { .. }
            | ColumnStream::DecimalPacked { .. }
            | ColumnStream::DecimalValues { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"-0123456789\"");
            }
            ColumnStream::TimestampMinSec { .. } | ColumnStream::TimestampFull { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"0123456789-:T\"");
            }
            ColumnStream::HexSuffix4 { .. }
            | ColumnStream::HexSuffix8 { .. }
            | ColumnStream::HexSuffix { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"0123456789abcdef");
            }
            ColumnStream::PeriodicHex { .. } => {
                add_bytes_to_mask(&mut non_dict_col_mask, b"0123456789abcdef\"");
            }
            ColumnStream::Copy16 { col_mask, .. } | ColumnStream::Copy32 { col_mask, .. } => {
                for i in 0..4 {
                    non_dict_col_mask[i] |= col_mask[i];
                }
            }
            ColumnStream::Raw { cursor } => {
                let raw_slice = &cursor.bytes[cursor.pos..];
                if overlaps_any_seg || contains_subslice(raw_slice, needle) {
                    let mut c = cursor;
                    while !c.done() {
                        let Ok(len) = c.uleb() else { break };
                        let Ok(val) = c.take(len) else { break };
                        add_bytes_to_mask(&mut non_dict_col_mask, val);
                    }
                }
            }
            ColumnStream::SegmentDict { str_u8_offset, .. } => {
                let seg_slice = &scratch.u8_arena[str_u8_offset..];
                if overlaps_any_seg || contains_subslice(seg_slice, needle) {
                    add_bytes_to_mask(&mut non_dict_col_mask, seg_slice);
                    add_bytes_to_mask(&mut non_dict_col_mask, b"\"");
                }
            }
            ColumnStream::PeriodicString { .. } => {
                non_dict_col_mask = [u64::MAX; 4];
            }
        }
    }

    let mut total_mask = [
        static_mask[0] | non_dict_col_mask[0],
        static_mask[1] | non_dict_col_mask[1],
        static_mask[2] | non_dict_col_mask[2],
        static_mask[3] | non_dict_col_mask[3],
    ];
    for (idx, ucol) in active_cols.iter().enumerate() {
        if let ColumnStream::Dictionary { piece_offset, .. } = ucol.stream {
            let dlen = active_dict_lens[idx];
            for e in 0..dlen {
                let piece = unsafe { scratch.piece_arena[piece_offset + e].as_slice() };
                add_bytes_to_mask(&mut total_mask, piece);
            }
        }
    }

    for &b in needle {
        if !mask_has_byte(&total_mask, b) {
            return TplFilter::Never;
        }
    }

    let mut best_filter: Option<(usize, TplFilter)> = None;
    for k in 0..needle.len() {
        let b = needle[k];
        if mask_has_byte(&non_dict_col_mask, b) {
            continue;
        }
        let mut aligns_static = piece_can_align_at(head_slice, needle, k);
        if !aligns_static {
            for ucol in active_cols {
                let after_slice = &scratch.str_arena[ucol.after_range.0..ucol.after_range.1];
                if piece_can_align_at(after_slice, needle, k) {
                    aligns_static = true;
                    break;
                }
            }
        }
        if aligns_static {
            continue;
        }

        // `needle[k]` cannot come from any static segment or non-dictionary column.
        // It MUST come from a `Dictionary` column whose entry aligns with `needle` at `k`.
        let mut matching_cols = 0usize;
        let mut candidate_u8_off = 0usize;
        let mut candidate_dlen = 0usize;
        let mut candidate_ids = [false; 256];
        let mut candidate_count = 0usize;
        let mut last_id = 0u8;

        for (idx, ucol) in active_cols.iter().enumerate() {
            if let ColumnStream::Dictionary {
                piece_offset,
                u8_offset,
                ..
            } = ucol.stream
            {
                let dlen = active_dict_lens[idx].min(256);
                let mut col_matches = 0usize;
                let mut local_ids = [false; 256];
                let mut local_last = 0u8;
                for e in 0..dlen {
                    let piece = unsafe { scratch.piece_arena[piece_offset + e].as_slice() };
                    if piece_can_align_at(piece, needle, k) {
                        local_ids[e] = true;
                        local_last = e as u8;
                        col_matches += 1;
                    }
                }
                if col_matches > 0 {
                    matching_cols += 1;
                    if matching_cols == 1 {
                        candidate_u8_off = u8_offset;
                        candidate_dlen = dlen;
                        candidate_ids = local_ids;
                        candidate_count = col_matches;
                        last_id = local_last;
                    }
                }
            }
        }

        if matching_cols == 0 {
            return TplFilter::Never;
        }
        if matching_cols == 1 && candidate_count < candidate_dlen {
            let is_better = best_filter
                .as_ref()
                .is_none_or(|(prev_cnt, _)| candidate_count < *prev_cnt);
            if is_better {
                let single_id = (candidate_count == 1).then_some(last_id);
                best_filter = Some((
                    candidate_count,
                    TplFilter::DictSingle {
                        u8_offset: candidate_u8_off,
                        valid_ids: Box::new(candidate_ids),
                        single_id,
                    },
                ));
                if candidate_count == 1 {
                    break;
                }
            }
        }
    }

    best_filter.map_or(TplFilter::CheckAll, |(_, f)| f)
}

fn huf_alphabet_mask(blob: &[u8], mask: &mut [u64; 4]) -> bool {
    let mut cur = Cursor::new(blob);
    let Ok(kind) = cur.byte() else { return false };
    match kind {
        HUF_KIND_RLE => {
            let Ok(sym) = cur.byte() else { return false };
            mask[(sym >> 6) as usize] |= 1u64 << (sym & 63);
            true
        }
        HUF_KIND_HUF => {
            let Ok(max_sym) = cur.byte() else { return false };
            let max_sym = max_sym as usize;
            const NIB_STEP: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 6, 12, 24];
            const NIB_LEN: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0];
            let mut s = 0usize;
            while s <= max_sym {
                let Ok(b) = cur.byte() else { return false };
                let b = b as usize;
                let n0 = b & 15;
                if NIB_LEN[n0] != 0 {
                    mask[s >> 6] |= 1u64 << (s & 63);
                }
                s += NIB_STEP[n0] as usize;
                if s <= max_sym {
                    let n1 = b >> 4;
                    if NIB_LEN[n1] != 0 {
                        mask[s >> 6] |= 1u64 << (s & 63);
                    }
                    s += NIB_STEP[n1] as usize;
                }
            }
            true
        }
        _ => false,
    }
}

fn palette_alphabet_mask(src: &[u8], n: usize, mask: &mut [u64; 4]) -> bool {
    let mut cursor = Cursor::new(src);
    let Ok(hdr) = cursor.byte() else { return false };
    let bits = hdr & 0x7f;
    let has_esc = (hdr & 0x80) != 0;
    if !matches!(bits, 4..=6) {
        return false;
    }
    let Ok(pal_len) = cursor.byte() else { return false };
    let Ok(pal_slice) = cursor.take(pal_len as usize) else {
        return false;
    };
    add_bytes_to_mask(mask, pal_slice);
    if has_esc {
        let Ok(esc_len) = cursor.uleb() else { return false };
        let packed_len = (n * (bits as usize)).div_ceil(8);
        let Ok(_) = cursor.take(packed_len) else { return false };
        let Ok(esc_bytes) = cursor.take(esc_len) else {
            return false;
        };
        add_bytes_to_mask(mask, esc_bytes);
    }
    true
}

fn fast_lz_alphabet_mask(mut cursor: Cursor<'_>, mask: &mut [u64; 4]) -> bool {
    let Ok(raw_len) = cursor.uleb() else { return false };
    let Ok(seq_count) = cursor.uleb() else { return false };
    if seq_count == 0 {
        let Ok(comp_len) = cursor.uleb() else { return false };
        if comp_len == 0 {
            let Ok(huf_len) = cursor.uleb() else { return false };
            let Ok(huf) = cursor.take(huf_len) else { return false };
            return huf_alphabet_mask(huf, mask);
        }
        let Ok(comp) = cursor.take(comp_len) else { return false };
        return palette_alphabet_mask(comp, raw_len, mask);
    }
    let Ok(lit_hdr) = cursor.uleb() else { return false };
    let (flags, lit_total_len) = if lit_hdr <= raw_len {
        (0usize, lit_hdr)
    } else {
        let rem = lit_hdr - (raw_len + 1);
        (rem >> 16, rem & 0xffff)
    };
    let Ok(off_count) = cursor.uleb() else { return false };
    let Ok(ext_len) = cursor.uleb() else { return false };
    if (flags & 1) != 0 {
        let Ok(m_len) = cursor.uleb() else { return false };
        let Ok(_) = cursor.take(m_len) else { return false };
    } else if (flags & 8) != 0 {
        for _ in 0..4 {
            let Ok(l) = cursor.uleb() else { return false };
            let Ok(_) = cursor.take(l) else { return false };
        }
    } else {
        let Ok(_) = cursor.take(seq_count + off_count * 2 + ext_len) else {
            return false;
        };
    }
    if (flags & 2) != 0 {
        let Ok(l) = cursor.uleb() else { return false };
        let Ok(comp) = cursor.take(l) else { return false };
        palette_alphabet_mask(comp, lit_total_len, mask)
    } else if (flags & 4) != 0 {
        let Ok(l) = cursor.uleb() else { return false };
        let Ok(huf) = cursor.take(l) else { return false };
        huf_alphabet_mask(huf, mask)
    } else {
        let Ok(lits) = cursor.take(lit_total_len) else {
            return false;
        };
        add_bytes_to_mask(mask, lits);
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemplatePreCheck {
    Never,
    Always,
    SingleSlot(usize),
    CheckTemplate,
}

fn precheck_template_substr(
    tpl_segs: &[&[u8]],
    col_slices: &[&[u8]],
    ones: usize,
    needle: &[u8],
) -> Result<TemplatePreCheck> {
    if needle.is_empty() {
        return Ok(TemplatePreCheck::Always);
    }
    let slots = col_slices.len();
    let mut overlaps_any_seg = (1..slots).any(|s| tpl_segs[s].is_empty());
    for &seg in tpl_segs {
        if contains_subslice(seg, needle) {
            return Ok(TemplatePreCheck::Always);
        }
        if !overlaps_any_seg && !seg.is_empty() {
            let max_p = (needle.len() - 1).min(seg.len());
            for p in 1..=max_p {
                if seg.ends_with(&needle[..p]) || seg.starts_with(&needle[needle.len() - p..]) {
                    overlaps_any_seg = true;
                    break;
                }
            }
            if !overlaps_any_seg && contains_subslice(needle, seg) {
                overlaps_any_seg = true;
            }
        }
    }

    if !overlaps_any_seg {
        let mut match_count = 0usize;
        let mut first_slot = 0usize;
        for (s, &cb) in col_slices.iter().enumerate() {
            if col_can_contain_isolated(cb, ones, needle)? {
                match_count += 1;
                if match_count == 1 {
                    first_slot = s;
                } else {
                    return Ok(TemplatePreCheck::CheckTemplate);
                }
            }
        }
        return Ok(match match_count {
            0 => TemplatePreCheck::Never,
            1 => TemplatePreCheck::SingleSlot(first_slot),
            _ => TemplatePreCheck::CheckTemplate,
        });
    }

    let mut st = BigramState::new();
    for s in 0..slots {
        st.feed_static(tpl_segs[s]);
        feed_col_into_bigram_state(col_slices[s], &mut st)?;
        if !st.exact && st.byte_mask == [u64::MAX; 4] {
            return Ok(TemplatePreCheck::CheckTemplate);
        }
    }
    st.feed_static(tpl_segs[slots]);
    if !st.matches_needle(needle) {
        return Ok(TemplatePreCheck::Never);
    }
    Ok(TemplatePreCheck::CheckTemplate)
}

/// Fast pre-unpack check on `tpl_segs` and raw column headers (`col_slices`) that rejects
/// blocks in $O(1)$ time without unpacking column streams when `needle` cannot match.
#[inline]
fn can_template_match_fast(
    tpl_segs: &[&[u8]],
    col_slices: &[&[u8]],
    needle: &[u8],
) -> Result<bool> {
    Ok(!matches!(
        precheck_template_substr(tpl_segs, col_slices, usize::MAX, needle)?,
        TemplatePreCheck::Never
    ))
}

/// Find the row index (`0..row_count`) of the `nth` occurrence of `target_bit` (`true` = 1, `false` = 0)
/// inside a 1-bit packed `selector_slice`.
fn nth_selector_bit(selector_slice: &[u8], row_count: usize, target_one: bool, mut nth: usize) -> usize {
    for (byte_idx, &b) in selector_slice.iter().enumerate() {
        let base_row = byte_idx * 8;
        let valid_bits = row_count.saturating_sub(base_row).min(8);
        let mask = if valid_bits == 8 { 0xffu8 } else { (1u8 << valid_bits) - 1 };
        let word = (if target_one { b } else { !b }) & mask;
        let ones = word.count_ones() as usize;
        if nth < ones {
            for bit in 0..valid_bits {
                if ((word >> bit) & 1) != 0 {
                    if nth == 0 {
                        return base_row + bit;
                    }
                    nth -= 1;
                }
            }
        }
        nth -= ones;
    }
    row_count
}
