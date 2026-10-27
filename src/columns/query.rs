impl<'a> FastBitReader<'a> {
    #[inline(always)]
    fn skip(&mut self, n: usize) {
        if self.width == 0 || n == 0 {
            return;
        }
        let cur_bit = self.bit_end - (self.avail as usize);
        self.bit_end = cur_bit + n * (self.width as usize);
        self.avail = 0;
        self.buf = 0;
    }
}

impl FastPiece {
    #[inline(always)]
    unsafe fn as_slice(&self) -> &[u8] {
        unsafe {
            if self.len <= 48 {
                std::slice::from_raw_parts((&raw const self.v0).cast::<u8>(), self.len)
            } else {
                std::slice::from_raw_parts(self.ptr, self.len)
            }
        }
    }
}

impl<'a> ExecColumn<'a> {
    #[inline(always)]
    unsafe fn skip_rows(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        unsafe {
            match &mut self.stream {
                BoundStream::Dictionary16 { idx_ptr, .. }
                | BoundStream::Dictionary32 { idx_ptr, .. }
                | BoundStream::Dictionary48 { idx_ptr, .. }
                | BoundStream::DictionaryLong { idx_ptr, .. }
                | BoundStream::DictionaryPair32 {
                    idx_ptr0: idx_ptr, ..
                }
                | BoundStream::DictionaryPair48 {
                    idx_ptr0: idx_ptr, ..
                }
                | BoundStream::DictionaryQuad32 {
                    idx_ptr0: idx_ptr, ..
                }
                | BoundStream::DictionaryQuad48 {
                    idx_ptr0: idx_ptr, ..
                } => {
                    *idx_ptr = idx_ptr.add(n);
                }
                BoundStream::DecimalAffineSmall16 { current, step }
                | BoundStream::DecimalAffineSmall32 { current, step }
                | BoundStream::DecimalAffineSmall48 { current, step }
                | BoundStream::DecimalAffineSmall { current, step }
                | BoundStream::DecimalAffine { current, step } => {
                    *current = current.wrapping_add(step.wrapping_mul(n as u64));
                }
                BoundStream::DecimalSmall16 { val_ptr }
                | BoundStream::DecimalSmall32 { val_ptr }
                | BoundStream::DecimalSmall48 { val_ptr }
                | BoundStream::DecimalSmall { val_ptr } => {
                    *val_ptr = val_ptr.add(n);
                }
                BoundStream::DecimalValues { val_ptr } => {
                    *val_ptr = val_ptr.add(n);
                }
                BoundStream::DecimalPacked { reader, .. } => {
                    reader.skip(n);
                }
                BoundStream::TimestampMinSec16 { min_ptr, sec_ptr }
                | BoundStream::TimestampMinSec32 { min_ptr, sec_ptr }
                | BoundStream::TimestampMinSec { min_ptr, sec_ptr } => {
                    *min_ptr = min_ptr.add(n);
                    *sec_ptr = sec_ptr.add(n);
                }
                BoundStream::TimestampFull { ptr, .. } => {
                    *ptr = ptr.add(n);
                }
                BoundStream::Copy16After16 { ptr }
                | BoundStream::Copy16After32 { ptr }
                | BoundStream::Copy16 { ptr } => {
                    *ptr = ptr.add(n * 16);
                }
                BoundStream::HexSuffix4 { ptr } => {
                    *ptr = ptr.add(n * 4);
                }
                BoundStream::HexSuffix8 { ptr } => {
                    *ptr = ptr.add(n * 8);
                }
                BoundStream::HexSuffix { cursor, width } => {
                    cursor.pos += n * (*width);
                }
                BoundStream::SegmentDict { str_ptr, len_ptr } => {
                    let mut byte_sum = 0usize;
                    for i in 0..n {
                        byte_sum += (*len_ptr.add(i)) as usize;
                    }
                    *len_ptr = len_ptr.add(n);
                    *str_ptr = str_ptr.add(byte_sum);
                }
                BoundStream::Raw { cursor } => {
                    for _ in 0..n {
                        let len = cursor.uleb().unwrap_unchecked();
                        cursor.pos += len;
                    }
                }
                BoundStream::PeriodicString { cursor, period } => {
                    let p = *period;
                    for _ in 0..n {
                        let len = cursor.uleb().unwrap_unchecked();
                        cursor.pos += len.min(p);
                    }
                }
                BoundStream::PeriodicHex { cursor, period } => {
                    let p = *period;
                    for _ in 0..n {
                        let len = cursor.uleb().unwrap_unchecked();
                        cursor.pos += len.min(p).div_ceil(2);
                    }
                }
            }
        }
    }
}

impl PreparedTemplate {
    #[inline(always)]
    unsafe fn skip_rows(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        unsafe {
            for slot in 0..self.col_len {
                (*self.col_ptr.add(slot)).skip_rows(n);
            }
        }
    }
}

#[inline(always)]
fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    let nlen = needle.len();
    if nlen == 0 {
        return true;
    }
    let hlen = haystack.len();
    if hlen < nlen {
        return false;
    }
    let first = needle[0];
    let last = needle[nlen - 1];
    let limit = hlen - nlen;
    let hptr = haystack.as_ptr();
    let nptr = needle.as_ptr();
    let mut i = 0usize;
    while i <= limit {
        let Some(rel) = find_byte(first, &haystack[i..=limit]) else {
            return false;
        };
        i += rel;
        unsafe {
            if *hptr.add(i + nlen - 1) == last
                && std::slice::from_raw_parts(hptr.add(i), nlen)
                    == std::slice::from_raw_parts(nptr, nlen)
            {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Checks if `piece` can align with `needle` such that `needle[k]` lands on some byte of `piece`.
#[inline(always)]
fn piece_can_align_at(piece: &[u8], needle: &[u8], k: usize) -> bool {
    let target = needle[k];
    let plen = piece.len();
    let nlen = needle.len();
    for j in 0..plen {
        if piece[j] == target {
            let left = k.min(j);
            let right = (nlen - k).min(plen - j);
            if piece[j - left..j + right] == needle[k - left..k + right] {
                return true;
            }
        }
    }
    false
}

#[inline(always)]
fn add_bytes_to_mask(mask: &mut [u64; 4], bytes: &[u8]) {
    for &b in bytes {
        mask[(b >> 6) as usize] |= 1u64 << (b & 63);
    }
}

#[inline(always)]
fn mask_has_byte(mask: &[u64; 4], b: u8) -> bool {
    (mask[(b >> 6) as usize] & (1u64 << (b & 63))) != 0
}

/// Fast $O(1)$ header inspection returning `Some((row_count, ends_with_nl))` for `GRAMMAR_PACKET`.
pub(crate) fn grammar_block_row_meta(payload: &[u8]) -> Result<Option<(usize, bool)>> {
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
    let dictionary_len = cursor.uleb()?;
    if dictionary_len > template_count.saturating_mul(MAX_SLOTS + 1) {
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
    for flag in seg_ends_nl.iter_mut() {
        let len = cursor.uleb()?;
        let seg = cursor.take(len)?;
        *flag = seg.last() == Some(&b'\n');
    }
    let mut slot_counts_inline = [0usize; 16];
    let mut slot_counts_overflow = if template_count > 16 {
        vec![0usize; template_count]
    } else {
        Vec::new()
    };
    let slot_counts: &mut [usize] = if template_count <= 16 {
        &mut slot_counts_inline[..template_count]
    } else {
        &mut slot_counts_overflow[..template_count]
    };
    for s in slot_counts.iter_mut() {
        let slots = cursor.uleb()?;
        if slots > MAX_SLOTS {
            return Err(invalid("column slot count"));
        }
        *s = slots;
    }
    let segment_width = cursor.byte()?;
    if segment_width != bit_width(dictionary_len.saturating_sub(1) as u64) {
        return Err(invalid("column segment selector width"));
    }
    let mut tpl_ends_nl = [true; MAX_TEMPLATES];
    for (t_idx, &slots) in slot_counts.iter().enumerate() {
        if dictionary_len == 0 {
            return Err(invalid("empty column segment dictionary"));
        }
        let packed_len = bit_bytes(slots + 1, segment_width)?;
        let packed_slice = cursor.take(packed_len)?;
        let mut rdr = FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
        rdr.skip(slots);
        let last_seg_idx = rdr.next_u64() as usize;
        if last_seg_idx >= dictionary_len {
            return Err(invalid("column segment dictionary index"));
        }
        tpl_ends_nl[t_idx] = seg_ends_nl[last_seg_idx];
    }
    let row_count = cursor.uleb()?;
    if row_count > raw_len.saturating_add(1) {
        return Err(invalid("column row count"));
    }
    if row_count == 0 {
        return Ok(Some((0, true)));
    }
    let selector_width = cursor.byte()?;
    if selector_width != bit_width(template_count as u64) {
        return Err(invalid("column row selector width"));
    }
    let selector_len = bit_bytes(row_count, selector_width)?;
    let selector_slice = cursor.take(selector_len)?;
    let mut sel_rdr = FastBitReader::new_validated(selector_slice, row_count, selector_width)?;
    sel_rdr.skip(row_count - 1);
    let last_sel = sel_rdr.next_u64() as usize;
    if last_sel > 0 {
        if last_sel > template_count {
            return Err(invalid("column row selector index"));
        }
        return Ok(Some((row_count, tpl_ends_nl[last_sel - 1])));
    }
    if cursor.bytes.get(cursor.pos) == Some(&0) {
        cursor.pos += 1;
        let lz_len = cursor.uleb()?;
        let lz_bytes = cursor.take(lz_len)?;
        let mut lz_cursor = Cursor::new(lz_bytes);
        if lz_cursor.byte()? != FAST_LZ_PACKET {
            return Err(invalid("column raw rows lz selector"));
        }
        let mut probe = Cursor::new(lz_cursor.bytes);
        probe.pos = lz_cursor.pos;
        let unpacked_len = probe.uleb()?;
        if unpacked_len == 0 || unpacked_len > 65536 {
            return Err(invalid("column raw rows lz length"));
        }
        let ends_nl = DECODE_SCRATCH.with(|cell| -> Result<bool> {
            let mut scratch = cell.borrow_mut();
            scratch.raw_arena.clear();
            scratch.raw_arena.reserve(unpacked_len + 64);
            unsafe {
                scratch.raw_arena.set_len(unpacked_len);
            }
            unpack_fast_lz_into(&mut lz_cursor, &mut scratch.raw_arena[..unpacked_len])?;
            Ok(scratch.raw_arena[unpacked_len - 1] == b'\n')
        })?;
        Ok(Some((row_count, ends_nl)))
    } else {
        let raw_count = if selector_width == 1 {
            let ones: usize = selector_slice.iter().map(|b| b.count_ones() as usize).sum();
            row_count.saturating_sub(ones)
        } else {
            let mut r = FastBitReader::new_validated(selector_slice, row_count, selector_width)?;
            let mut c = 0usize;
            for _ in 0..row_count {
                if r.next_u64() == 0 {
                    c += 1;
                }
            }
            c
        };
        let mut last_byte = b'\n';
        for _ in 0..raw_count {
            let len = cursor.uleb()?;
            let raw = cursor.take(len)?;
            last_byte = *raw.last().ok_or_else(|| invalid("empty raw column row"))?;
        }
        Ok(Some((row_count, last_byte == b'\n')))
    }
}

/// Fast $O(1)$ header inspection returning `Some(row_count)` for `GRAMMAR_PACKET` payloads.
pub(crate) fn grammar_block_row_count(payload: &[u8]) -> Result<Option<usize>> {
    Ok(grammar_block_row_meta(payload)?.map(|(c, _)| c))
}

fn read_grammar_raw_rows<'a>(
    cursor: &mut Cursor<'a>,
    raw_count: usize,
    raw_arena: &mut Vec<u8>,
    slice_arena: &mut Vec<(*const u8, usize)>,
) -> Result<usize> {
    let raw_slice_offset = slice_arena.len();
    if raw_count == 0 {
        return Ok(raw_slice_offset);
    }
    slice_arena.reserve(raw_count);
    if cursor.bytes.get(cursor.pos) == Some(&0) {
        cursor.pos += 1;
        let lz_len = cursor.uleb()?;
        let lz_bytes = cursor.take(lz_len)?;
        let mut lz_cursor = Cursor::new(lz_bytes);
        if lz_cursor.byte()? != FAST_LZ_PACKET {
            return Err(invalid("column raw rows lz selector"));
        }
        let mut probe = Cursor::new(lz_cursor.bytes);
        probe.pos = lz_cursor.pos;
        let unpacked_len = probe.uleb()?;
        if unpacked_len == 0 || unpacked_len > 65536 {
            return Err(invalid("column raw rows lz length"));
        }
        let arena_start = raw_arena.len();
        if arena_start + unpacked_len > raw_arena.capacity() {
            return Err(invalid("column raw rows arena overflow"));
        }
        unsafe {
            raw_arena.set_len(arena_start + unpacked_len);
        }
        unpack_fast_lz_into(
            &mut lz_cursor,
            &mut raw_arena[arena_start..arena_start + unpacked_len],
        )?;
        let unpacked: &'a [u8] = unsafe {
            std::slice::from_raw_parts(raw_arena.as_ptr().add(arena_start), unpacked_len)
        };
        let mut sub = Cursor::new(unpacked);
        let lens_len = sub.uleb()?;
        let mut len_cursor = Cursor::new(sub.take(lens_len)?);
        for _ in 0..raw_count {
            let len = len_cursor.uleb()?;
            let raw = sub.take(len)?;
            if raw.is_empty() {
                return Err(invalid("empty raw column row"));
            }
            slice_arena.push((raw.as_ptr(), len));
        }
        if !len_cursor.done() || !sub.done() {
            return Err(invalid("column raw rows lz trailing bytes"));
        }
        Ok(raw_slice_offset)
    } else {
        for _ in 0..raw_count {
            let len = cursor.uleb()?;
            let raw = cursor.take(len)?;
            if raw.is_empty() {
                return Err(invalid("empty raw column row"));
            }
            slice_arena.push((raw.as_ptr(), len));
        }
        Ok(raw_slice_offset)
    }
}

/// Unpack only rows `[start_row .. end_row.min(row_count))` from a `GRAMMAR_PACKET` payload
/// using $O(1)$ column stream skipping (`PreparedTemplate::skip_rows`).
pub(crate) fn unpack_grammar_row_range(
    payload: &[u8],
    start_row: usize,
    end_row: usize,
    out: &mut Vec<u8>,
) -> Result<Option<usize>> {
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
    let dictionary_len = cursor.uleb()?;
    if dictionary_len > template_count.saturating_mul(MAX_SLOTS + 1) {
        return Err(invalid("column segment dictionary count"));
    }
    let mut seg_dict = vec![&[][..]; dictionary_len];
    for i in 0..dictionary_len {
        let len = cursor.uleb()?;
        let seg = cursor.take(len)?;
        if i > 0 && seg_dict[i - 1] >= seg {
            return Err(invalid("column segment dictionary ordering"));
        }
        seg_dict[i] = seg;
    }
    let mut slot_counts = vec![0usize; template_count];
    for s in &mut slot_counts {
        let slots = cursor.uleb()?;
        if slots > MAX_SLOTS {
            return Err(invalid("column slot count"));
        }
        *s = slots;
    }
    let segment_width = cursor.byte()?;
    if segment_width != bit_width(dictionary_len.saturating_sub(1) as u64) {
        return Err(invalid("column segment selector width"));
    }
    let mut tpl_segs = Vec::with_capacity(template_count);
    for &slots in &slot_counts {
        if dictionary_len == 0 {
            return Err(invalid("empty column segment dictionary"));
        }
        let packed_len = bit_bytes(slots + 1, segment_width)?;
        let packed_slice = cursor.take(packed_len)?;
        let mut reader = FastBitReader::new_validated(packed_slice, slots + 1, segment_width)?;
        let mut segs = Vec::with_capacity(slots + 1);
        for _ in 0..=slots {
            let idx = reader.next_u64() as usize;
            if idx >= dictionary_len {
                return Err(invalid("column segment dictionary index"));
            }
            segs.push(seg_dict[idx]);
        }
        tpl_segs.push(segs);
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

    let clamped_end = end_row.min(row_count);
    if start_row >= clamped_end {
        return Ok(Some(0));
    }
    if selector_width > 7 {
        let full = decode_block(payload)?;
        let mut cur = 0usize;
        for (idx, line) in full.split_inclusive(|&b| b == b'\n').enumerate() {
            if idx >= start_row && idx < clamped_end {
                out.extend_from_slice(line);
                cur += 1;
            }
        }
        return Ok(Some(cur));
    }

    DECODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.raw_arena.reserve(raw_len.saturating_mul(2) + 16384);
        let sel_u8_offset = unpack_and_validate_u8(
            selector_slice,
            row_count,
            selector_width,
            0,
            template_count + 1,
            &mut scratch.u8_arena,
        )?;
        let mut sel_counts = [0usize; 129];
        let mut skip_counts = [0usize; 129];
        for (r, &sel) in scratch.u8_arena[sel_u8_offset..sel_u8_offset + row_count]
            .iter()
            .enumerate()
        {
            sel_counts[sel as usize] += 1;
            if r < start_row {
                skip_counts[sel as usize] += 1;
            }
        }
        let raw_total = sel_counts[0];
        let DecodeScratch {
            ref mut raw_arena,
            ref mut slice_arena,
            ..
        } = *scratch;
        let raw_slice_offset =
            read_grammar_raw_rows(&mut cursor, raw_total, raw_arena, slice_arena)?;

        let mut cols_uninit = core::mem::MaybeUninit::<[PreparedColumn; MAX_SLOTS]>::uninit();
        let cols: &mut [PreparedColumn; MAX_SLOTS] = unsafe { cols_uninit.assume_init_mut() };
        let mut unbound_templates = vec![UnboundTemplate::default(); template_count];
        for tpl_idx in 0..template_count {
            let slots = slot_counts[tpl_idx];
            let count = sel_counts[tpl_idx + 1];
            for s in 0..slots {
                let len = cursor.uleb()?;
                let bytes = cursor.take(len)?;
                cols[s] = PreparedColumn::parse_and_validate(bytes, count, &mut scratch)?;
            }
            unbound_templates[tpl_idx] =
                UnboundTemplate::build(&tpl_segs[tpl_idx], &cols[..slots], &mut scratch);
        }
        if !cursor.done() {
            return Err(invalid("column packet trailing bytes"));
        }
        let total_unbound_cols = scratch.unbound_col_arena.len();
        scratch.exec_col_arena.reserve(total_unbound_cols);
        let mut prepared_templates = Vec::with_capacity(template_count);
        for tpl_idx in 0..template_count {
            let mut pt = unbound_templates[tpl_idx].bind(&mut scratch);
            unsafe {
                pt.skip_rows(skip_counts[tpl_idx + 1]);
            }
            prepared_templates.push(pt);
        }

        let mut raw_idx = skip_counts[0];
        let mut row_buf = vec![0u8; 4096];
        for r in start_row..clamped_end {
            let sel = scratch.u8_arena[sel_u8_offset + r] as usize;
            if sel != 0 {
                let tpl = &mut prepared_templates[sel - 1];
                if tpl.row_budget > row_buf.len() {
                    row_buf.resize(tpl.row_budget + 64, 0);
                }
                let written = unsafe {
                    let start_p = row_buf.as_mut_ptr();
                    let end_p = tpl.emit_row(start_p);
                    end_p.offset_from(start_p) as usize
                };
                out.extend_from_slice(&row_buf[..written]);
            } else {
                let (raw_ptr, len) = scratch.slice_arena[raw_slice_offset + raw_idx];
                raw_idx += 1;
                let s = unsafe { std::slice::from_raw_parts(raw_ptr, len) };
                out.extend_from_slice(s);
            }
        }
        Ok(Some(clamped_end - start_row))
    })
}
