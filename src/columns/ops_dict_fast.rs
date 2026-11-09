#[inline(always)]
fn compact_even_bits_u64(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    ((x | (x >> 16)) & 0xffff_ffff) as u32
}

#[inline(always)]
fn read_packed_u64_word(packed: &[u8], byte_off: usize) -> u64 {
    if byte_off >= packed.len() {
        return 0;
    }
    let rem = packed.len() - byte_off;
    if rem >= 8 {
        unsafe { packed.as_ptr().add(byte_off).cast::<u64>().read_unaligned().to_le() }
    } else {
        let mut w = 0u64;
        for i in 0..rem {
            w |= u64::from(packed[byte_off + i]) << (8 * i);
        }
        w
    }
}

/// Fast zero-arena `MODEL_DICTIONARY` (`entries <= 16`) masked evaluator (`SWAR` for `1..=2`-bit ids).
fn try_eval_dict_masked(
    col_bytes: &[u8],
    ones: usize,
    pred: &crate::ops::ColPred<'_>,
    active: &mut TplMask,
) -> Result<bool> {
    if ones == 0 || col_bytes.first() != Some(&MODEL_DICTIONARY) {
        return Ok(false);
    }
    let mut c = Cursor::new(&col_bytes[1..]);
    let entries_len = c.uleb()?;
    if entries_len == 0 || entries_len >= ones {
        return Err(invalid("column dictionary cardinality"));
    }
    if entries_len > 16 {
        return Ok(false);
    }
    let mut prev_entry: Option<&[u8]> = None;
    let mut lut_bits = 0u16;
    for i in 0..entries_len {
        let len = c.uleb()?;
        let entry = c.take(len)?;
        if prev_entry.is_some_and(|last| last >= entry) {
            return Err(invalid("column dictionary ordering"));
        }
        prev_entry = Some(entry);
        if crate::ops::eval_token_pred(pred.col, entry, pred.pred) {
            lut_bits |= 1u16 << i;
        }
    }
    let width = c.byte()?;
    if width != bit_width((entries_len - 1) as u64) {
        return Err(invalid("column dictionary width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed = c.take(packed_len)?;
    if !c.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let _ = FastBitReader::new_validated(packed, ones, width)?;

    let full_mask = if entries_len == 16 { u16::MAX } else { (1u16 << entries_len) - 1 };
    let needs_bounds_check = width > 0 && entries_len < (1usize << width);
    let wlen = ones.div_ceil(64);

    if !needs_bounds_check {
        if lut_bits == 0 {
            *active = TplMask::Empty;
            return Ok(true);
        }
        if lut_bits == full_mask {
            return Ok(true);
        }
    }

    let eval_word = |wi: usize, cnt: usize, buf: &mut [u16; 64]| -> Result<u64> {
        let valid_w_mask = if cnt == 64 { u64::MAX } else { (1u64 << cnt) - 1 };
        match width {
            0 => Ok(if (lut_bits & 1) != 0 { valid_w_mask } else { 0 }),
            1 => {
                let raw = read_packed_u64_word(packed, wi * 8);
                let out = if lut_bits == 0b10 { raw } else { !raw };
                Ok(out & valid_w_mask)
            }
            2 => {
                let lo = read_packed_u64_word(packed, wi * 16);
                let hi = read_packed_u64_word(packed, wi * 16 + 8);
                if entries_len == 3 {
                    let bad_lo = lo & (lo >> 1) & 0x5555_5555_5555_5555;
                    let bad_hi = hi & (hi >> 1) & 0x5555_5555_5555_5555;
                    if (bad_lo | bad_hi) != 0 {
                        return Err(invalid("column bit index out of bounds"));
                    }
                }
                let map_half = |x: u64| -> u32 {
                    let ev = x & 0x5555_5555_5555_5555;
                    let od = (x >> 1) & 0x5555_5555_5555_5555;
                    let mut m = 0u64;
                    if (lut_bits & 1) != 0 { m |= !ev & !od; }
                    if (lut_bits & 2) != 0 { m |= ev & !od; }
                    if (lut_bits & 4) != 0 { m |= !ev & od; }
                    if (lut_bits & 8) != 0 { m |= ev & od; }
                    compact_even_bits_u64(m)
                };
                let w = u64::from(map_half(lo)) | (u64::from(map_half(hi)) << 32);
                Ok(w & valid_w_mask)
            }
            _ => {
                unpack_u16_chunk64(packed, wi << 6, cnt, width, buf);
                let mut w = 0u64;
                for (bit, &idx) in buf[..cnt].iter().enumerate() {
                    if (idx as usize) >= entries_len {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    w |= u64::from((lut_bits >> idx) & 1) << bit;
                }
                Ok(w)
            }
        }
    };

    let mut buf = [0u16; 64];
    match *active {
        TplMask::Empty => {
            if needs_bounds_check {
                for wi in 0..wlen {
                    let _ = eval_word(wi, (ones - (wi << 6)).min(64), &mut buf)?;
                }
            }
        }
        TplMask::All => {
            let mut bs = RowBitset::all_zeros(ones);
            let ws = bs.words_mut();
            let mut pop = 0usize;
            for (wi, slot) in ws.iter_mut().enumerate().take(wlen) {
                let w = eval_word(wi, (ones - (wi << 6)).min(64), &mut buf)?;
                *slot = w;
                pop += w.count_ones() as usize;
            }
            *active = if lut_bits == 0 {
                TplMask::Empty
            } else if lut_bits == full_mask {
                TplMask::All
            } else {
                TplMask::from_pop(bs, pop, ones)
            };
        }
        TplMask::Bits(ref mut bs, ref mut cur_pop) => {
            let ws = bs.words_mut();
            let mut new_pop = 0usize;
            for (wi, slot) in ws.iter_mut().enumerate().take(wlen) {
                let w0 = *slot;
                if w0 == 0 && !needs_bounds_check {
                    continue;
                }
                let w = w0 & eval_word(wi, (ones - (wi << 6)).min(64), &mut buf)?;
                *slot = w;
                new_pop += w.count_ones() as usize;
            }
            if new_pop == 0 {
                *active = TplMask::Empty;
            } else {
                *cur_pop = new_pop;
            }
        }
    }
    Ok(true)
}
