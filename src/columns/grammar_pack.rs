#[inline(always)]
fn has_structured_line_near(
    data: &[u8],
    probe_base: usize,
    skip_first_partial: bool,
    max_lines: usize,
    hits: &mut usize,
    prev_slots: &mut Option<usize>,
    probe_spans: &mut [(u32, u32); MAX_SLOTS],
    probe_overflow: &mut Vec<(u32, u32)>,
) -> bool {
    let mut pos = if skip_first_partial {
        let Some(nl) = find_byte(b'\n', &data[probe_base..data.len().min(probe_base + 4096)])
        else {
            return false;
        };
        probe_base + nl + 1
    } else {
        probe_base
    };
    for _ in 0..max_lines {
        if pos >= data.len() {
            break;
        }
        let Some(nl) = find_byte(b'\n', &data[pos..data.len().min(pos + 4096)]) else {
            break;
        };
        let first_len = nl + 1;
        let rec_len = scan_multiline_json_object(&data[pos..], first_len).unwrap_or(first_len);
        let end = pos + rec_len;
        let line = &data[pos..end];
        if let Some(slots) = parse_record_spans(line, probe_spans, probe_overflow) {
            let mut p = 0usize;
            skip_space(line, &mut p);
            if line.get(p) == Some(&b'{') {
                return true;
            }
            if *prev_slots == Some(slots) {
                *hits += 1;
                if *hits >= 2 {
                    return true;
                }
            } else {
                *prev_slots = Some(slots);
                if *hits == 0 {
                    *hits = 1;
                }
            }
        }
        pos = end;
    }
    false
}

fn pack_grammar_into(data: &[u8], out: &mut Vec<u8>) -> Result<bool> {
    if data.len() >= 256 {
        let sample_end = data.len().min(64);
        for &b in &data[..sample_end] {
            if b < b'\t' || (b >= 0x0e && b < b' ') {
                return Ok(false);
            }
        }
        let first_looks_json = data[0] == b'{'
            && find_byte(b'\n', &data[..data.len().min(4096)]).is_some_and(|nl| {
                nl >= 2
                    && (data[nl - 1] == b'}'
                        || (nl >= 3 && data[nl - 1] == b'\r' && data[nl - 2] == b'}'))
            });
        if !first_looks_json {
            let mut probe_spans = [(0u32, 0u32); MAX_SLOTS];
            let mut probe_overflow = Vec::new();
            let mut hits = 0usize;
            let mut prev_slots = None;
            let has_start = has_structured_line_near(
                data,
                0,
                false,
                8,
                &mut hits,
                &mut prev_slots,
                &mut probe_spans,
                &mut probe_overflow,
            );
            if !has_start {
                let mid = data.len() / 2;
                let try_mid = if hits > 0 {
                    data.len() >= 4096
                } else {
                    data.len() >= 4096
                        && find_byte(b'\n', &data[mid..data.len().min(mid + 4096)])
                            .is_some_and(|nl| data.get(mid + nl + 1) == Some(&b'{'))
                };
                if !try_mid
                    || !has_structured_line_near(
                        data,
                        mid,
                        true,
                        6,
                        &mut hits,
                        &mut prev_slots,
                        &mut probe_spans,
                        &mut probe_overflow,
                    )
                {
                    return Ok(false);
                }
            }
        }
    }
    let (templates, rows, template_rows, raw_rows, raw_bytes) = grammar_from_block_mode(data, true)?;
    if templates.is_empty()
        || !templates.iter().any(|t| !t.values.is_empty())
        || template_rows < 2
        || (data.len() >= 4096
            && (raw_bytes * 3 > data.len() * 2
                || (templates.len() > 2 && raw_rows > template_rows * 2 && raw_bytes * 2 > data.len())))
    {
        return_row_vec(rows);
        return Ok(false);
    }
    out.push(GRAMMAR_PACKET);
    put_uleb(out, data.len());
    put_uleb(out, templates.len());
    let total_segments: usize = templates.iter().map(|t| t.segments.len()).sum();
    let mut seg_dict_inline = [core::mem::MaybeUninit::<&[u8]>::uninit(); 128];
    let mut seg_dict_overflow = Vec::new();
    let segment_dictionary: &mut [&[u8]] = if total_segments <= 128 {
        let mut idx = 0usize;
        for template in &templates {
            for &seg in &template.segments {
                seg_dict_inline[idx].write(seg);
                idx += 1;
            }
        }
        let slice: &mut [&[u8]] = unsafe {
            std::slice::from_raw_parts_mut(seg_dict_inline.as_mut_ptr().cast::<&[u8]>(), total_segments)
        };
        slice.sort_unstable();
        let mut w = 0usize;
        for r in 0..slice.len() {
            if w == 0 || slice[r] != slice[w - 1] {
                slice[w] = slice[r];
                w += 1;
            }
        }
        &mut slice[..w]
    } else {
        seg_dict_overflow.reserve(total_segments);
        for template in &templates {
            seg_dict_overflow.extend_from_slice(&template.segments);
        }
        seg_dict_overflow.sort_unstable();
        seg_dict_overflow.dedup();
        &mut seg_dict_overflow[..]
    };
    put_uleb(out, segment_dictionary.len());
    for segment in &*segment_dictionary {
        put_bytes(out, segment);
    }
    for template in &templates {
        let slots = template.values.len();
        if template.segments.len() != slots + 1 || slots > MAX_SLOTS {
            return_row_vec(rows);
            return Err(invalid("column template shape"));
        }
        put_uleb(out, slots);
    }
    let segment_width = bit_width(segment_dictionary.len().saturating_sub(1) as u64);
    out.push(segment_width);
    for template in &templates {
        push_bit_values(
            out,
            template.segments.len(),
            segment_width,
            template
                .segments
                .iter()
                .map(|segment| segment_dictionary.binary_search(segment).unwrap() as u64),
        );
    }
    put_uleb(out, rows.len());
    let selector_width = bit_width(templates.len() as u64);
    out.push(selector_width);
    if templates.len() == 1 && raw_rows == 0 {
        let full_bytes = rows.len() / 8;
        let rem_bits = rows.len() & 7;
        out.resize(out.len() + full_bytes, 0xff);
        if rem_bits != 0 {
            out.push((1u8 << rem_bits) - 1);
        }
    } else {
        push_bit_values(
            out,
            rows.len(),
            selector_width,
            rows.iter().map(|row| match row {
                Row::Template(index) => (index + 1) as u64,
                Row::Raw(_) => 0,
            }),
        );
        if raw_rows > 0 {
            if selector_width <= 7 && raw_bytes >= 64 {
                let wrote_lz = MODEL_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    let ModelScratch {
                        raw_lens,
                        raw_stream,
                        raw_lz,
                        ..
                    } = &mut *sc;
                    raw_lens.clear();
                    raw_lens.reserve(raw_rows * 2);
                    raw_stream.clear();
                    raw_stream.reserve(4 + raw_rows * 2 + raw_bytes);
                    for row in &rows {
                        if let Row::Raw(bytes) = row {
                            put_uleb(raw_lens, bytes.len());
                            raw_stream.extend_from_slice(bytes);
                        }
                    }
                    let plain_len = raw_lens.len() + raw_bytes;
                    let hdr_len = uleb_len(raw_lens.len()) + raw_lens.len();
                    let total_stream_len = hdr_len + raw_bytes;
                    raw_lz.clear();
                    if (64..=65536).contains(&total_stream_len) {
                        raw_stream.resize(total_stream_len, 0);
                        raw_stream.copy_within(0..raw_bytes, hdr_len);
                        let mut hdr_pos = 0usize;
                        let mut v = raw_lens.len();
                        while v >= 0x80 {
                            raw_stream[hdr_pos] = ((v as u8) & 0x7f) | 0x80;
                            hdr_pos += 1;
                            v >>= 7;
                        }
                        raw_stream[hdr_pos] = v as u8;
                        hdr_pos += 1;
                        raw_stream[hdr_pos..hdr_len].copy_from_slice(raw_lens);
                        raw_lz.reserve(total_stream_len / 2);
                        if pack_fast_lz_into(raw_stream, raw_lz).is_some()
                            && 1 + uleb_len(raw_lz.len()) + raw_lz.len() + 4 < plain_len
                        {
                            out.push(0);
                            put_uleb(out, raw_lz.len());
                            out.extend_from_slice(raw_lz);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                });
                if !wrote_lz {
                    for row in &rows {
                        if let Row::Raw(bytes) = row {
                            put_bytes(out, bytes);
                        }
                    }
                }
            } else {
                for row in &rows {
                    if let Row::Raw(bytes) = row {
                        put_bytes(out, bytes);
                    }
                }
            }
        }
    }
    return_row_vec(rows);
    for template in &templates {
        for column in &template.values {
            let (model, _kind) = pack_model_v3(column);
            let _ = uleb_len(model.len())
                .checked_add(model.len())
                .ok_or(Error::SizeOverflow)?;
            put_uleb(out, model.len());
            out.extend_from_slice(&model);
        }
    }
    Ok(true)
}

fn pack_grammar(data: &[u8]) -> Result<PackedGrammar> {
    let (templates, rows, template_rows, raw_rows) = grammar_from_block(data)?;
    let mut out = Vec::with_capacity(4096);
    out.push(GRAMMAR_PACKET);
    put_uleb(&mut out, data.len());
    put_uleb(&mut out, templates.len());
    let total_segments: usize = templates.iter().map(|t| t.segments.len()).sum();
    let mut segment_dictionary = Vec::with_capacity(total_segments);
    for template in &templates {
        segment_dictionary.extend_from_slice(&template.segments);
    }
    segment_dictionary.sort_unstable();
    segment_dictionary.dedup();
    put_uleb(&mut out, segment_dictionary.len());
    for segment in &segment_dictionary {
        put_bytes(&mut out, segment);
    }
    for template in &templates {
        let slots = template.values.len();
        if template.segments.len() != slots + 1 || slots > MAX_SLOTS {
            return_row_vec(rows);
            return Err(invalid("column template shape"));
        }
        put_uleb(&mut out, slots);
    }
    let segment_width = bit_width(segment_dictionary.len().saturating_sub(1) as u64);
    out.push(segment_width);
    for template in &templates {
        push_bit_values(
            &mut out,
            template.segments.len(),
            segment_width,
            template
                .segments
                .iter()
                .map(|segment| segment_dictionary.binary_search(segment).unwrap() as u64),
        );
    }
    put_uleb(&mut out, rows.len());
    let selector_width = bit_width(templates.len() as u64);
    out.push(selector_width);
    push_bit_values(
        &mut out,
        rows.len(),
        selector_width,
        rows.iter().map(|row| match row {
            Row::Template(index) => (index + 1) as u64,
            Row::Raw(_) => 0,
        }),
    );
    for row in &rows {
        if let Row::Raw(bytes) = row {
            put_bytes(&mut out, bytes);
        }
    }
    return_row_vec(rows);
    let mut stats = BlockStats {
        grammar: true,
        templates: templates.len(),
        template_rows,
        raw_rows,
        columns: Vec::with_capacity(total_segments),
    };
    for (template_index, template) in templates.iter().enumerate() {
        for (slot, column) in template.values.iter().enumerate() {
            let (model, kind) = pack_model(column);
            let payload_bytes = uleb_len(model.len())
                .checked_add(model.len())
                .ok_or(Error::SizeOverflow)?;
            put_uleb(&mut out, model.len());
            out.extend_from_slice(&model);
            stats.columns.push(ColumnStat {
                template: template_index,
                slot,
                kind,
                values: column.len(),
                payload_bytes,
            });
        }
    }
    Ok(PackedGrammar { bytes: out, stats })
}

/// Preloaded 64-byte piece for branchless unaligned `u128` stores during row assembly.
///
/// Size is exactly 64 bytes (`1 << 6`) so indexing `entries[idx]` compiles to a bit-shift `idx << 6`.
#[derive(Clone, Copy)]
#[repr(C)]
struct FastPiece {
    v0: u128,
    v1: u128,
    v2: u128,
    len: usize,
    ptr: *const u8,
}

#[inline(always)]
unsafe fn read_u128_partial(ptr: *const u8, len: usize) -> u128 {
    unsafe {
        if len >= 16 {
            ptr.cast::<u128>().read_unaligned()
        } else if len >= 8 {
            let lo = ptr.cast::<u64>().read_unaligned() as u128;
            let hi = ptr.add(len - 8).cast::<u64>().read_unaligned() as u128;
            lo | ((hi >> ((16 - len) * 8)) << 64)
        } else if len >= 4 {
            let lo = ptr.cast::<u32>().read_unaligned() as u64;
            let hi = ptr.add(len - 4).cast::<u32>().read_unaligned() as u64;
            (lo | ((hi >> ((8 - len) * 8)) << 32)) as u128
        } else if len > 0 {
            let b0 = *ptr as u64;
            let b1 = *ptr.add(len >> 1) as u64;
            let b2 = *ptr.add(len - 1) as u64;
            (b0 | (b1 << ((len >> 1) * 8)) | (b2 << ((len - 1) * 8))) as u128
        } else {
            0
        }
    }
}

impl FastPiece {
    #[inline(always)]
    const fn empty() -> Self {
        Self {
            v0: 0,
            v1: 0,
            v2: 0,
            len: 0,
            ptr: core::ptr::null(),
        }
    }

    #[inline(always)]
    fn from_slice(slice: &[u8]) -> Self {
        let len = slice.len();
        let ptr = slice.as_ptr();
        unsafe {
            let v0 = read_u128_partial(ptr, len);
            let v1 = if len > 16 {
                read_u128_partial(ptr.add(16), len - 16)
            } else {
                0
            };
            let v2 = if len > 32 {
                read_u128_partial(ptr.add(32), len - 32)
            } else {
                0
            };
            Self {
                v0,
                v1,
                v2,
                len,
                ptr,
            }
        }
    }

    #[inline(always)]
    unsafe fn emit_16(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_32(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(16).cast::<u128>().write_unaligned(self.v1);
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_48_exact(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(16).cast::<u128>().write_unaligned(self.v1);
            dst.add(32).cast::<u128>().write_unaligned(self.v2);
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_48(&self, dst: *mut u8) -> *mut u8 {
        unsafe {
            dst.cast::<u128>().write_unaligned(self.v0);
            dst.add(16).cast::<u128>().write_unaligned(self.v1);
            dst.add(32).cast::<u128>().write_unaligned(self.v2);
            if self.len > 48 {
                Self::emit_tail_cold(self.ptr, dst, self.len);
            }
            dst.add(self.len)
        }
    }

    #[inline(always)]
    unsafe fn emit_tail_cold(src: *const u8, dst: *mut u8, len: usize) {
        unsafe {
            if len <= 160 {
                let mut off = 48usize;
                while off + 16 < len {
                    dst.add(off)
                        .cast::<u128>()
                        .write_unaligned(src.add(off).cast::<u128>().read_unaligned());
                    off += 16;
                }
                dst.add(len - 16)
                    .cast::<u128>()
                    .write_unaligned(src.add(len - 16).cast::<u128>().read_unaligned());
            } else {
                std::ptr::copy_nonoverlapping(src.add(48), dst.add(48), len - 48);
            }
        }
    }
}

/// Reusable per-thread scratch arenas for zero-allocation block decoding.
#[derive(Default)]
struct DecodeScratch {
    u8_arena: Vec<u8>,
    u16_arena: Vec<u16>,
    u32_arena: Vec<u32>,
    u64_arena: Vec<u64>,
    piece_arena: Vec<FastPiece>,
    str_arena: Vec<u8>,
    slice_arena: Vec<(*const u8, usize)>,
    raw_arena: Vec<u8>,
    unbound_col_arena: Vec<UnboundExecColumn<'static>>,
    exec_col_arena: Vec<ExecColumn<'static>>,
}

impl DecodeScratch {
    #[inline(always)]
    fn clear(&mut self) {
        self.u8_arena.clear();
        self.u16_arena.clear();
        self.u32_arena.clear();
        self.u64_arena.clear();
        self.piece_arena.clear();
        self.str_arena.clear();
        self.slice_arena.clear();
        self.raw_arena.clear();
        self.unbound_col_arena.clear();
        self.exec_col_arena.clear();
    }
}

std::thread_local! {
    static DECODE_SCRATCH: std::cell::RefCell<DecodeScratch> = std::cell::RefCell::new(DecodeScratch {
        u8_arena: Vec::with_capacity(16384),
        u16_arena: Vec::with_capacity(4096),
        u32_arena: Vec::with_capacity(4096),
        u64_arena: Vec::with_capacity(4096),
        piece_arena: Vec::with_capacity(512),
        str_arena: Vec::with_capacity(4096),
        slice_arena: Vec::with_capacity(512),
        raw_arena: Vec::with_capacity(262144),
        unbound_col_arena: Vec::with_capacity(256),
        exec_col_arena: Vec::with_capacity(256),
    });
}

