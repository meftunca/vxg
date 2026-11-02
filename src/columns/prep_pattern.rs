// Fast production decoder and zero-unpack alphabet inspector for `MODEL_PATTERN`.

#[derive(Clone, Copy)]
enum FastPatTok<'a> {
    Digits { width: u8, min: u64, radix: u64, div: FastDiv62 },
    Lit(&'a [u8]),
    Dict { slice_off: u32, len: u8, fixed_width: Option<u16>, max_len: usize, div: FastDiv62 },
}

#[inline(always)]
unsafe fn write_padded_digits(dst: *mut u8, width: usize, mut val: u64) {
    unsafe {
        match width {
            2 => {
                let p = TWO_DIGITS.as_ptr().add((val as usize) * 2).cast::<u16>().read_unaligned();
                dst.cast::<u16>().write_unaligned(p);
            }
            4 => dst.cast::<u32>().write_unaligned(*DEC_PADDED_10K.get_unchecked(val as usize)),
            3 => {
                let q = ((val as u32) * 41) >> 12;
                *dst = b'0' + q as u8;
                let r = (val as usize) - (q as usize) * 100;
                let p = TWO_DIGITS.as_ptr().add(r * 2).cast::<u16>().read_unaligned();
                dst.add(1).cast::<u16>().write_unaligned(p);
            }
            1 => *dst = b'0' + (val as u8),
            6 => {
                let hi = (val / 10_000) as usize;
                let lo = (val % 10_000) as usize;
                let p = TWO_DIGITS.as_ptr().add(hi * 2).cast::<u16>().read_unaligned();
                dst.cast::<u16>().write_unaligned(p);
                dst.add(2).cast::<u32>().write_unaligned(*DEC_PADDED_10K.get_unchecked(lo));
            }
            _ => {
                let mut end = width;
                while end >= 4 {
                    end -= 4;
                    let rem = (val % 10_000) as usize;
                    val /= 10_000;
                    dst.add(end).cast::<u32>().write_unaligned(*DEC_PADDED_10K.get_unchecked(rem));
                }
                while end > 0 {
                    end -= 1;
                    *dst.add(end) = b'0' + (val % 10) as u8;
                    val /= 10;
                }
            }
        }
    }
}

#[inline(always)]
fn render_const_tokens_into(toks: &[FastPatTok<'_>], mut dst: *mut u8) -> usize {
    let start = dst;
    unsafe {
        for &tok in toks {
            match tok {
                FastPatTok::Lit(s) => {
                    std::ptr::copy_nonoverlapping(s.as_ptr(), dst, s.len());
                    dst = dst.add(s.len());
                }
                FastPatTok::Digits { width, min, .. } => {
                    let w = width as usize;
                    write_padded_digits(dst, w, min);
                    dst = dst.add(w);
                }
                FastPatTok::Dict { .. } => unreachable!(),
            }
        }
        dst.offset_from(start) as usize
    }
}

fn render_affix_slice<'a>(
    toks: &[FastPatTok<'a>],
    total_len: usize,
    raw_arena: &mut Vec<u8>,
) -> Result<&'a [u8]> {
    if total_len == 0 {
        return Ok(b"");
    }
    if toks.len() == 1 {
        if let FastPatTok::Lit(s) = toks[0] {
            return Ok(s);
        }
    }
    let st = raw_arena.len();
    if st + total_len > raw_arena.capacity() {
        return Err(invalid("column output exceeds length"));
    }
    unsafe {
        render_const_tokens_into(toks, raw_arena.as_mut_ptr().add(st));
        raw_arena.set_len(st + total_len);
        Ok(std::slice::from_raw_parts(raw_arena.as_ptr().add(st), total_len))
    }
}

impl<'a> PreparedColumn<'a> {
    #[inline(never)]
    fn parse_pattern(cursor: Cursor<'a>, count: usize, scratch: &mut DecodeScratch) -> Result<Self> {
        let slice_base = scratch.slice_arena.len();
        let res = Self::parse_pattern_inner(cursor, count, scratch);
        scratch.slice_arena.truncate(slice_base);
        res
    }

    fn parse_pattern_inner(mut cursor: Cursor<'a>, count: usize, scratch: &mut DecodeScratch) -> Result<Self> {
        if count < 2 {
            return Err(invalid("column pattern count"));
        }
        let orig_cursor = cursor;
        let ntok = cursor.uleb()?;
        if !(1..=PATTERN_MAX_TOKENS).contains(&ntok) {
            return Err(invalid("column pattern tokens"));
        }
        let mut toks = [FastPatTok::Lit(&[]); PATTERN_MAX_TOKENS];
        let (mut product, mut max_row_len) = (1u64, 0usize);

        for t in 0..ntok {
            let tok = match cursor.byte()? {
                0 => {
                    let width = cursor.byte()?;
                    let w_usize = width as usize;
                    let (min, radix) = (cursor.uleb()? as u64, cursor.uleb()? as u64);
                    if !(1..=PATTERN_MAX_DIGITS).contains(&w_usize) || radix == 0 {
                        return Err(invalid("column pattern digits"));
                    }
                    let max = min.checked_add(radix - 1).ok_or_else(|| invalid("column pattern range"))?;
                    if max >= 10u64.pow(width as u32) {
                        return Err(invalid("column pattern width"));
                    }
                    product = product.checked_mul(radix).ok_or_else(|| invalid("column pattern product"))?;
                    max_row_len = max_row_len.saturating_add(w_usize);
                    FastPatTok::Digits { width, min, radix, div: FastDiv62::new(radix) }
                }
                1 => {
                    let len = cursor.uleb()?;
                    if len > MAX_BLOCK_SIZE {
                        return Err(invalid("column pattern literal"));
                    }
                    max_row_len = max_row_len.saturating_add(len);
                    FastPatTok::Lit(cursor.take(len)?)
                }
                2 => {
                    let entries = cursor.uleb()?;
                    if !(2..=PATTERN_MAX_DICT).contains(&entries) {
                        return Err(invalid("column pattern dictionary"));
                    }
                    let off = scratch.slice_arena.len();
                    scratch.slice_arena.reserve(entries);
                    let mut prev: Option<&'a [u8]> = None;
                    let (mut max_e, mut min_e) = (0usize, usize::MAX);
                    for _ in 0..entries {
                        let len = cursor.uleb()?;
                        let e = cursor.take(len)?;
                        if prev.is_some_and(|l| l >= e) {
                            return Err(invalid("column pattern dictionary order"));
                        }
                        prev = Some(e);
                        max_e = max_e.max(len);
                        min_e = min_e.min(len);
                        scratch.slice_arena.push((e.as_ptr(), len));
                    }
                    product = product.checked_mul(entries as u64).ok_or_else(|| invalid("column pattern product"))?;
                    max_row_len = max_row_len.saturating_add(max_e);
                    let fixed_width = (min_e == max_e && max_e <= u16::MAX as usize).then_some(max_e as u16);
                    FastPatTok::Dict {
                        slice_off: off as u32,
                        len: entries as u8,
                        fixed_width,
                        max_len: max_e,
                        div: FastDiv62::new(entries as u64),
                    }
                }
                _ => return Err(invalid("column pattern token kind")),
            };
            if product > PATTERN_MAX_PRODUCT {
                return Err(invalid("column pattern product"));
            }
            toks[t] = tok;
        }
        if product == 1 {
            return Err(invalid("column pattern degenerate"));
        }
        if max_row_len > 2048 {
            return Self::parse_pattern_huge(orig_cursor, count, scratch);
        }

        let n0 = cursor.uleb()? as u64;
        if n0 >= product {
            return Err(invalid("column pattern first value"));
        }
        let w = cursor.byte()?;
        if w > 64 {
            return Err(invalid("column pattern width"));
        }
        let exc_count = cursor.uleb()?;
        if exc_count > count - 1 {
            return Err(invalid("column pattern exceptions"));
        }
        let mut exc_cursor = cursor;
        let mut val_row = 0usize;
        for k in 0..exc_count {
            let step = cursor.uleb()?;
            val_row = val_row.checked_add(step).ok_or_else(|| invalid("column pattern exception row"))?;
            if (k > 0 && step == 0) || val_row == 0 || val_row >= count {
                return Err(invalid("column pattern exception row"));
            }
            let z = cursor.uleb()? as u64;
            if w < 64 && z >> w == 0 {
                return Err(invalid("column pattern exception value"));
            }
        }
        let packed = cursor.take(bit_bytes(count - 1, w)?)?;
        if !cursor.done() {
            return Err(invalid("column model trailing bytes"));
        }
        let mut reader = FastBitReader::new_validated(packed, count - 1, w)?;

        let is_var = |tok: FastPatTok<'_>| match tok {
            FastPatTok::Digits { radix, .. } => radix > 1,
            FastPatTok::Lit(_) => false,
            FastPatTok::Dict { .. } => true,
        };
        let act_start = (0..ntok).find(|&t| is_var(toks[t])).unwrap();
        let act_end = (0..ntok).rfind(|&t| is_var(toks[t])).unwrap() + 1;
        let const_tok_len = |s: &[FastPatTok<'_>]| -> usize {
            s.iter().map(|t| match *t {
                FastPatTok::Lit(b) => b.len(),
                FastPatTok::Digits { width, .. } => width as usize,
                FastPatTok::Dict { .. } => 0,
            }).sum()
        };

        let lead_len = const_tok_len(&toks[..act_start]);
        let const_lead = if lead_len == 0 {
            LeadValue::Empty
        } else if act_start > 1 && lead_len <= 16 {
            let mut buf = [0u8; 16];
            render_const_tokens_into(&toks[..act_start], buf.as_mut_ptr());
            LeadValue::Inline { buf, len: lead_len as u8 }
        } else {
            LeadValue::Slice(render_affix_slice(&toks[..act_start], lead_len, &mut scratch.raw_arena)?)
        };
        let trail_len = const_tok_len(&toks[act_end..ntok]);
        let const_trail = render_affix_slice(&toks[act_end..ntok], trail_len, &mut scratch.raw_arena)?;

        let act_toks = &toks[act_start..act_end];
        let act_len = act_toks.len();
        let (mut fixed_len, mut max_act_len) = (Some(0usize), 0usize);
        for &tok in act_toks {
            match tok {
                FastPatTok::Digits { width, .. } => {
                    fixed_len = fixed_len.map(|a| a + width as usize);
                    max_act_len += width as usize;
                }
                FastPatTok::Lit(s) => {
                    fixed_len = fixed_len.map(|a| a + s.len());
                    max_act_len += s.len();
                }
                FastPatTok::Dict { fixed_width, max_len, .. } => {
                    fixed_len = fixed_len.zip(fixed_width).map(|(a, fw)| a + fw as usize);
                    max_act_len += max_len;
                }
            }
        }

        let (mut next_exc_row, mut next_exc_z, mut exc_left) = if exc_count > 0 {
            (exc_cursor.uleb().unwrap(), exc_cursor.uleb().unwrap() as u64, exc_count - 1)
        } else {
            (usize::MAX, 0u64, 0usize)
        };

        // Fast path 1: Fixed-width active span in `1..=32` bytes -> `Copy16` or `Copy32`.
        if let Some(flen) = fixed_len.filter(|&l| (1..=32).contains(&l)) {
            let stride = if flen <= 16 { 16usize } else { 32usize };
            let mut col_mask = [0u64; 4];
            let mut tok_offs = [0u8; PATTERN_MAX_TOKENS];
            let mut cur_bytes = [0u8; 32];
            let mut off = 0usize;
            for (i, &tok) in act_toks.iter().enumerate() {
                tok_offs[i] = off as u8;
                match tok {
                    FastPatTok::Digits { width, min, radix, .. } => {
                        let w_u = width as usize;
                        add_digit_range_to_mask(&mut col_mask, w_u, min, min + radix - 1);
                        if radix == 1 {
                            unsafe { write_padded_digits(cur_bytes.as_mut_ptr().add(off), w_u, min); }
                        }
                        off += w_u;
                    }
                    FastPatTok::Lit(s) => {
                        add_bytes_to_mask(&mut col_mask, s);
                        cur_bytes[off..off + s.len()].copy_from_slice(s);
                        off += s.len();
                    }
                    FastPatTok::Dict { slice_off, len, fixed_width, .. } => {
                        for e in 0..len as usize {
                            let (p, l) = scratch.slice_arena[slice_off as usize + e];
                            add_bytes_to_mask(&mut col_mask, unsafe { std::slice::from_raw_parts(p, l) });
                        }
                        off += fixed_width.unwrap() as usize;
                    }
                }
            }

            let tail_digits = match act_toks[act_len - 1] {
                FastPatTok::Digits { width, min, radix, .. } => {
                    Some((radix, width as usize, min, tok_offs[act_len - 1] as usize))
                }
                _ => None,
            };

            let render_fixed = |n: u64, buf: &mut [u8; 32], slices: &[(*const u8, usize)]| -> u64 {
                let (mut rem, mut last_f) = (n, 0u64);
                for i in (0..act_len).rev() {
                    match act_toks[i] {
                        FastPatTok::Digits { width, min, radix, div } if radix > 1 => {
                            let (q, f) = div.div_rem(rem);
                            rem = q;
                            if i + 1 == act_len {
                                last_f = f;
                            }
                            unsafe {
                                write_padded_digits(buf.as_mut_ptr().add(tok_offs[i] as usize), width as usize, min + f);
                            }
                        }
                        FastPatTok::Dict { slice_off, div, .. } => {
                            let (q, f) = div.div_rem(rem);
                            rem = q;
                            unsafe {
                                let (p, l) = *slices.get_unchecked(slice_off as usize + f as usize);
                                copy_short_exact(p, buf.as_mut_ptr().add(tok_offs[i] as usize), l);
                            }
                        }
                        _ => {}
                    }
                }
                last_f
            };

            #[inline(always)]
            unsafe fn store_fixed(dst: *mut u8, buf: &[u8; 32], stride: usize) {
                unsafe {
                    copy16_unaligned(buf.as_ptr(), dst);
                    if stride == 32 {
                        copy16_unaligned(buf.as_ptr().add(16), dst.add(16));
                    }
                }
            }

            let out_u8_offset = scratch.u8_arena.len();
            scratch.u8_arena.reserve(count * stride);
            let out_ptr = unsafe { scratch.u8_arena.as_mut_ptr().add(out_u8_offset) };
            let mut cur = n0;
            let mut last_field = render_fixed(cur, &mut cur_bytes, &scratch.slice_arena);
            unsafe { store_fixed(out_ptr, &cur_bytes, stride); }

            for r in 1..count {
                let packed_z = reader.next_u64();
                let z = if r == next_exc_row {
                    if packed_z != 0 {
                        return Err(invalid("column pattern exception padding"));
                    }
                    let ez = next_exc_z;
                    if exc_left > 0 {
                        exc_left -= 1;
                        next_exc_row += unsafe { exc_cursor.uleb().unwrap_unchecked() };
                        next_exc_z = unsafe { exc_cursor.uleb().unwrap_unchecked() } as u64;
                    } else {
                        next_exc_row = usize::MAX;
                    }
                    ez
                } else {
                    packed_z
                };
                if z == 0 {
                    unsafe { store_fixed(out_ptr.add(r * stride), &cur_bytes, stride); }
                    continue;
                }
                let delta = ((z >> 1) as i64) ^ -((z & 1) as i64);
                if let Some((radix, width, min, toff)) = tail_digits {
                    if delta > 0 && (delta as u64) < radix - last_field {
                        let d_u = delta as u64;
                        last_field += d_u;
                        cur += d_u;
                        unsafe {
                            write_padded_digits(cur_bytes.as_mut_ptr().add(toff), width, min + last_field);
                            store_fixed(out_ptr.add(r * stride), &cur_bytes, stride);
                        }
                        continue;
                    }
                }
                let next = (cur as i64).checked_add(delta).ok_or_else(|| invalid("column pattern delta"))?;
                if next < 0 || next as u64 >= product {
                    return Err(invalid("column pattern range"));
                }
                cur = next as u64;
                last_field = render_fixed(cur, &mut cur_bytes, &scratch.slice_arena);
                unsafe { store_fixed(out_ptr.add(r * stride), &cur_bytes, stride); }
            }

            unsafe { scratch.u8_arena.set_len(out_u8_offset + count * stride); }
            let stream = if stride == 16 {
                ColumnStream::Copy16 { u8_offset: out_u8_offset, emit_len: flen as u8, col_mask }
            } else {
                ColumnStream::Copy32 { u8_offset: out_u8_offset, emit_len: flen as u8, col_mask }
            };
            return Ok(Self {
                const_full: ConstValue::None,
                const_lead,
                const_trail,
                dict_slices: None,
                max_emit_len: flen,
                stream,
            });
        }

        // Fast path 2: Variable-width or `33..=2048` active span -> `ColumnStream::SegmentDict`.
        let str_u8_offset = scratch.u8_arena.len();
        scratch.u8_arena.reserve(count.saturating_mul(max_act_len).saturating_add(96));
        let len_u16_offset = scratch.u16_arena.len();
        scratch.u16_arena.reserve(count);

        let str_base = unsafe { scratch.u8_arena.as_mut_ptr().add(str_u8_offset) };
        let len_base = unsafe { scratch.u16_arena.as_mut_ptr().add(len_u16_offset) };
        let mut dst_ptr = str_base;
        let mut vals = [0u64; PATTERN_MAX_TOKENS];

        let render_var = |n: u64, vals: &mut [u64; PATTERN_MAX_TOKENS], mut dst: *mut u8, slices: &[(*const u8, usize)]| -> (*mut u8, usize) {
            let mut rem = n;
            for i in (0..act_len).rev() {
                match act_toks[i] {
                    FastPatTok::Digits { radix, div, .. } if radix > 1 => {
                        let (q, r) = div.div_rem(rem);
                        vals[i] = r;
                        rem = q;
                    }
                    FastPatTok::Digits { .. } => vals[i] = 0,
                    FastPatTok::Dict { div, .. } => {
                        let (q, r) = div.div_rem(rem);
                        vals[i] = r;
                        rem = q;
                    }
                    FastPatTok::Lit(_) => {}
                }
            }
            let row_start = dst;
            unsafe {
                for i in 0..act_len {
                    match act_toks[i] {
                        FastPatTok::Lit(s) => {
                            copy_short_exact(s.as_ptr(), dst, s.len());
                            dst = dst.add(s.len());
                        }
                        FastPatTok::Digits { width, min, .. } => {
                            let w_u = width as usize;
                            write_padded_digits(dst, w_u, min + vals[i]);
                            dst = dst.add(w_u);
                        }
                        FastPatTok::Dict { slice_off, .. } => {
                            let (p, l) = *slices.get_unchecked(slice_off as usize + vals[i] as usize);
                            copy_short_exact(p, dst, l);
                            dst = dst.add(l);
                        }
                    }
                }
                (dst, dst.offset_from(row_start) as usize)
            }
        };

        let mut cur = n0;
        let (next_dst, mut prev_len) = render_var(cur, &mut vals, dst_ptr, &scratch.slice_arena);
        let (mut prev_ptr, mut max_val_len) = (dst_ptr, prev_len);
        dst_ptr = next_dst;
        unsafe { *len_base = prev_len as u16; }

        for r in 1..count {
            let packed_z = reader.next_u64();
            let z = if r == next_exc_row {
                if packed_z != 0 {
                    return Err(invalid("column pattern exception padding"));
                }
                let ez = next_exc_z;
                if exc_left > 0 {
                    exc_left -= 1;
                    next_exc_row += unsafe { exc_cursor.uleb().unwrap_unchecked() };
                    next_exc_z = unsafe { exc_cursor.uleb().unwrap_unchecked() } as u64;
                } else {
                    next_exc_row = usize::MAX;
                }
                ez
            } else {
                packed_z
            };
            if z == 0 {
                unsafe {
                    copy_short_exact(prev_ptr, dst_ptr, prev_len);
                    dst_ptr = dst_ptr.add(prev_len);
                    *len_base.add(r) = prev_len as u16;
                }
                continue;
            }
            let delta = ((z >> 1) as i64) ^ -((z & 1) as i64);
            let next = (cur as i64).checked_add(delta).ok_or_else(|| invalid("column pattern delta"))?;
            if next < 0 || next as u64 >= product {
                return Err(invalid("column pattern range"));
            }
            cur = next as u64;
            prev_ptr = dst_ptr;
            let (ndst, rlen) = render_var(cur, &mut vals, dst_ptr, &scratch.slice_arena);
            dst_ptr = ndst;
            prev_len = rlen;
            max_val_len = max_val_len.max(rlen);
            unsafe { *len_base.add(r) = rlen as u16; }
        }

        let actual_bytes = unsafe { dst_ptr.offset_from(str_base) as usize };
        unsafe {
            std::ptr::write_bytes(dst_ptr, 0, 32);
            scratch.u8_arena.set_len(str_u8_offset + actual_bytes);
            scratch.u16_arena.set_len(len_u16_offset + count);
        }
        Ok(Self {
            const_full: ConstValue::None,
            const_lead,
            const_trail,
            dict_slices: None,
            max_emit_len: max_val_len,
            stream: ColumnStream::SegmentDict { str_u8_offset, len_u16_offset },
        })
    }

    #[cold]
    #[inline(never)]
    fn parse_pattern_huge(mut cursor: Cursor<'a>, count: usize, scratch: &mut DecodeScratch) -> Result<Self> {
        let (start, cap) = (scratch.raw_arena.len(), scratch.raw_arena.capacity());
        let (mut pos, mut max_len, mut overflow) = (start, 0usize, false);
        pattern_decode(&mut cursor, count, &mut |row| {
            if overflow {
                return;
            }
            if pos + uleb_len(row.len()) + row.len() + 32 > cap {
                overflow = true;
                return;
            }
            unsafe { scratch.raw_arena.set_len(pos); }
            put_uleb(&mut scratch.raw_arena, row.len());
            scratch.raw_arena.extend_from_slice(row);
            pos = scratch.raw_arena.len();
            max_len = max_len.max(row.len());
        })?;
        if overflow || !cursor.done() {
            unsafe { scratch.raw_arena.set_len(start); }
            return Err(invalid(if overflow { "column pattern arena overflow" } else { "column model trailing bytes" }));
        }
        let bytes: &'a [u8] = unsafe { std::slice::from_raw_parts(scratch.raw_arena.as_ptr().add(start), pos - start) };
        Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::Empty,
            const_trail: b"",
            dict_slices: None,
            max_emit_len: max_len,
            stream: ColumnStream::Raw { cursor: Cursor::new(bytes) },
        })
    }
}
