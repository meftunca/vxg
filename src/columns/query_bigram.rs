/// Incremental $O(1)$-amortized bit-selector cursor replacing repeated $O(N)$ `nth_selector_bit`
/// scans when materializing multiple hits in a block.
struct SelectorBitCursor<'a> {
    bytes: &'a [u8],
    row_count: usize,
    target_one: bool,
    byte_idx: usize,
    rem_word: u8,
    cur_nth: usize,
}

impl<'a> SelectorBitCursor<'a> {
    #[inline(always)]
    fn load_word(bytes: &[u8], row_count: usize, target_one: bool, byte_idx: usize) -> u8 {
        if byte_idx >= bytes.len() {
            return 0;
        }
        let base_row = byte_idx * 8;
        let valid_bits = row_count.saturating_sub(base_row).min(8);
        if valid_bits == 0 {
            return 0;
        }
        let mask = if valid_bits == 8 { 0xffu8 } else { (1u8 << valid_bits) - 1 };
        let b = bytes[byte_idx];
        (if target_one { b } else { !b }) & mask
    }

    #[inline(always)]
    fn new(bytes: &'a [u8], row_count: usize, target_one: bool) -> Self {
        let rem_word = Self::load_word(bytes, row_count, target_one, 0);
        Self {
            bytes,
            row_count,
            target_one,
            byte_idx: 0,
            rem_word,
            cur_nth: 0,
        }
    }

    #[inline(always)]
    fn advance_to(&mut self, target_nth: usize) -> usize {
        if target_nth < self.cur_nth {
            *self = Self::new(self.bytes, self.row_count, self.target_one);
        }
        let mut skip = target_nth - self.cur_nth;
        loop {
            let ones = self.rem_word.count_ones() as usize;
            if skip < ones {
                let mut w = self.rem_word;
                for _ in 0..skip {
                    w &= w - 1;
                }
                let bit = w.trailing_zeros() as usize;
                self.rem_word = w & (w - 1);
                self.cur_nth = target_nth + 1;
                return self.byte_idx * 8 + bit;
            }
            skip -= ones;
            self.byte_idx += 1;
            if self.byte_idx >= self.bytes.len() {
                self.rem_word = 0;
                self.cur_nth = target_nth;
                return self.row_count;
            }
            self.rem_word =
                Self::load_word(self.bytes, self.row_count, self.target_one, self.byte_idx);
        }
    }
}

#[inline(always)]
fn decimal_digits_u64(v: u64) -> usize {
    if v < 10_000 {
        1 + usize::from(v >= 10) + usize::from(v >= 100) + usize::from(v >= 1_000)
    } else if v < 100_000_000 {
        5 + usize::from(v >= 100_000) + usize::from(v >= 1_000_000) + usize::from(v >= 10_000_000)
    } else {
        const POW10: [u64; 12] = [
            1_000_000_000,
            10_000_000_000,
            100_000_000_000,
            1_000_000_000_000,
            10_000_000_000_000,
            100_000_000_000_000,
            1_000_000_000_000_000,
            10_000_000_000_000_000,
            100_000_000_000_000_000,
            1_000_000_000_000_000_000,
            10_000_000_000_000_000_000,
            u64::MAX,
        ];
        let mut d = 9usize;
        for &p in &POW10[..11] {
            if v >= p {
                d += 1;
            } else {
                break;
            }
        }
        d
    }
}

#[inline(always)]
fn format_padded_u64_stack(mut v: u64, width: usize, out: &mut [u8; 24]) {
    for i in (0..width).rev() {
        out[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
}

#[inline(always)]
fn fixed_decimal_lcp_matches(lo: u64, hi: u64, width: usize, needle: &[u8]) -> bool {
    if needle.len() > width || width > 20 {
        return needle.len() <= width;
    }
    let mut lo_buf = [0u8; 24];
    let mut hi_buf = [0u8; 24];
    format_padded_u64_stack(lo, width, &mut lo_buf);
    format_padded_u64_stack(hi, width, &mut hi_buf);
    let mut lcp = 0usize;
    while lcp < width && lo_buf[lcp] == hi_buf[lcp] {
        lcp += 1;
    }
    let max_start = width - needle.len();
    'start_loop: for start in 0..=max_start {
        for (i, &b) in needle.iter().enumerate() {
            let p = start + i;
            if p < lcp {
                if b != lo_buf[p] {
                    continue 'start_loop;
                }
            } else if p == lcp && (b < lo_buf[lcp] || b > hi_buf[lcp]) {
                continue 'start_loop;
            }
        }
        return true;
    }
    false
}

/// Exact zero-false-negative check whether any integer in `lo..=hi` (formatted either zero-padded
/// to `padded_width` or as canonical unsigned decimal) can contain `needle` as a substring.
#[inline]
fn decimal_range_can_contain(
    lo: u64,
    hi: u64,
    padded_width: Option<usize>,
    needle: &[u8],
) -> bool {
    if needle.is_empty() {
        return true;
    }
    if lo > hi || !needle.iter().all(u8::is_ascii_digit) {
        return false;
    }
    if let Some(w) = padded_width {
        if decimal_digits_u64(hi) > w {
            return true;
        }
        return fixed_decimal_lcp_matches(lo, hi, w, needle);
    }
    let d_lo = decimal_digits_u64(lo);
    let d_hi = decimal_digits_u64(hi);
    if needle.len() > d_hi {
        return false;
    }
    if d_lo == d_hi {
        return fixed_decimal_lcp_matches(lo, hi, d_hi, needle);
    }
    const POW10_20: [u64; 20] = [
        1,
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
        1_000_000_000,
        10_000_000_000,
        100_000_000_000,
        1_000_000_000_000,
        10_000_000_000_000,
        100_000_000_000_000,
        1_000_000_000_000_000,
        10_000_000_000_000_000,
        100_000_000_000_000_000,
        1_000_000_000_000_000_000,
        10_000_000_000_000_000_000,
    ];
    let top_lo = POW10_20[d_hi - 1];
    if fixed_decimal_lcp_matches(top_lo, hi, d_hi, needle) {
        return true;
    }
    // Check numbers of length `<= d_hi - 1` (`lo ..= top_lo - 1`):
    if needle.len() >= d_hi {
        return false;
    }
    if needle.len() < d_hi - 1 {
        return true;
    }
    // `needle.len() == d_hi - 1`: must match an entire canonical `(d_hi - 1)`-digit number.
    (d_hi == 2 && lo == 0 && needle == b"0") || needle[0] != b'0'
}

/// Exact 22-position (`"YYYY-MM-DDTHH:MM:SSZ"`) bitmask filter for `MODEL_TIMESTAMP`.
fn timestamp_can_contain(mut c: Cursor<'_>, needle: &[u8]) -> Result<bool> {
    if needle.is_empty() {
        return Ok(true);
    }
    if needle.len() > 22 || needle.iter().any(|&b| b >= 128) {
        return Ok(false);
    }
    let year = c.u16()?;
    if year > 9999 {
        return Err(invalid("column timestamp year"));
    }
    let mut bases = [0u8; 5];
    let mut widths = [0u8; 5];
    for i in 0..5 {
        bases[i] = c.byte()?;
        widths[i] = c.byte()?;
        if widths[i] > 6 {
            return Err(invalid("column timestamp width"));
        }
    }
    const MAX_BOUNDS: [u8; 5] = [12, 31, 23, 59, 59];
    let mut pm = [0u128; 22];
    pm[0] = 1u128 << b'"';
    let mut y = year;
    for p in (1..=4).rev() {
        pm[p] = 1u128 << (b'0' + (y % 10) as u8);
        y /= 10;
    }
    pm[5] = 1u128 << b'-';
    pm[8] = 1u128 << b'-';
    pm[11] = 1u128 << b'T';
    pm[14] = 1u128 << b':';
    pm[17] = 1u128 << b':';
    pm[20] = 1u128 << b'Z';
    pm[21] = 1u128 << b'"';
    const COMP_POS: [usize; 5] = [6, 9, 12, 15, 18];
    for i in 0..5 {
        let lo = bases[i];
        let span = if widths[i] == 0 { 0u16 } else { (1u16 << widths[i]) - 1 };
        let hi = (lo as u16 + span).min(MAX_BOUNDS[i] as u16) as u8;
        if lo > hi {
            return Ok(true);
        }
        let mut tens = 0u128;
        for d in (lo / 10)..=(hi / 10) {
            tens |= 1u128 << (b'0' + d);
        }
        let mut ones_m = 0u128;
        if hi - lo >= 9 {
            for d in 0..=9u8 {
                ones_m |= 1u128 << (b'0' + d);
            }
        } else {
            for v in lo..=hi {
                ones_m |= 1u128 << (b'0' + (v % 10));
            }
        }
        pm[COMP_POS[i]] = tens;
        pm[COMP_POS[i] + 1] = ones_m;
    }
    let max_start = 22 - needle.len();
    'start: for start in 0..=max_start {
        for (k, &b) in needle.iter().enumerate() {
            if (pm[start + k] & (1u128 << b)) == 0 {
                continue 'start;
            }
        }
        return Ok(true);
    }
    Ok(false)
}

/// Reconstructs `MODEL_FRONT_DICT` entries (`dict_mode` `0`, `1`, `3`) on a stack double-buffer
/// (falling back to heap only if an entry exceeds 256 bytes), calling `visit(entry)` on each entry
/// and stopping early with `Ok(true)` if `visit` returns `true`. Returns `Ok(None)` for `dict_mode == 2`.
fn front_dict_visit_entries(
    c: &mut Cursor<'_>,
    entries_len: usize,
    dict_mode: u8,
    mut visit: impl FnMut(&[u8]) -> bool,
) -> Result<Option<bool>> {
    if dict_mode == 0 {
        for _ in 0..entries_len {
            let len = c.uleb()?;
            let entry = c.take(len)?;
            if visit(entry) {
                return Ok(Some(true));
            }
        }
        return Ok(Some(false));
    }
    if dict_mode != 1 && dict_mode != 3 {
        return Ok(None);
    }
    let (pal, pal_len, pal_mixed) = if dict_mode == 3 {
        read_front_dict_palette(c)?
    } else {
        ([0u8; 32], 0, false)
    };
    let len0 = c.uleb()?;
    let raw_first = c.take(len0)?;
    if visit(raw_first) {
        return Ok(Some(true));
    }
    let mut stack_a = [0u8; 256];
    let mut stack_b = [0u8; 256];
    let mut heap_prev: Vec<u8> = Vec::new();
    let mut heap_cur: Vec<u8> = Vec::new();
    let mut prev_in_a = true;
    let mut prev_len = len0;
    if len0 <= 256 {
        stack_a[..len0].copy_from_slice(raw_first);
    } else {
        heap_prev.extend_from_slice(raw_first);
    }

    for _ in 1..entries_len {
        let lcp = c.uleb()?;
        let lcs = c.uleb()?;
        let raw_hdr = c.uleb()?;
        let (mid_len, is_raw) = if dict_mode == 1 {
            (raw_hdr, true)
        } else if pal_mixed {
            (raw_hdr >> 1, (raw_hdr & 1) != 0)
        } else {
            (raw_hdr, false)
        };
        let packed_mid = if is_raw {
            c.take(mid_len)?
        } else if pal_len <= 16 {
            c.take((mid_len + 1) >> 1)?
        } else {
            c.take((mid_len * 5 + 7) >> 3)?
        };
        if lcp + lcs > prev_len || mid_len > 65536 {
            return Err(invalid("column front dictionary bounds"));
        }
        let total_len = lcp + mid_len + lcs;
        if prev_len <= 256 && total_len <= 256 && heap_prev.is_empty() {
            let (prev_s, cur_s): (&[u8], &mut [u8]) = if prev_in_a {
                (&stack_a[..prev_len], &mut stack_b[..total_len])
            } else {
                (&stack_b[..prev_len], &mut stack_a[..total_len])
            };
            cur_s[..lcp].copy_from_slice(&prev_s[..lcp]);
            if mid_len > 0 {
                if is_raw {
                    cur_s[lcp..lcp + mid_len].copy_from_slice(packed_mid);
                } else {
                    unpack_nibble_bytes_into(
                        packed_mid,
                        mid_len,
                        &pal,
                        pal_len,
                        cur_s[lcp..].as_mut_ptr(),
                    )?;
                }
            }
            cur_s[lcp + mid_len..].copy_from_slice(&prev_s[prev_len - lcs..]);
            if visit(cur_s) {
                return Ok(Some(true));
            }
            prev_in_a = !prev_in_a;
            prev_len = total_len;
        } else {
            if heap_prev.is_empty() {
                let ps = if prev_in_a { &stack_a[..prev_len] } else { &stack_b[..prev_len] };
                heap_prev.extend_from_slice(ps);
            }
            heap_cur.resize(total_len, 0);
            heap_cur[..lcp].copy_from_slice(&heap_prev[..lcp]);
            if mid_len > 0 {
                if is_raw {
                    heap_cur[lcp..lcp + mid_len].copy_from_slice(packed_mid);
                } else {
                    unpack_nibble_bytes_into(
                        packed_mid,
                        mid_len,
                        &pal,
                        pal_len,
                        heap_cur[lcp..].as_mut_ptr(),
                    )?;
                }
            }
            heap_cur[lcp + mid_len..].copy_from_slice(&heap_prev[prev_len - lcs..]);
            if visit(&heap_cur) {
                return Ok(Some(true));
            }
            std::mem::swap(&mut heap_prev, &mut heap_cur);
            prev_len = total_len;
        }
    }
    Ok(Some(false))
}
