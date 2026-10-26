const fn two_digit_table() -> [u8; 200] {
    let mut table = [0u8; 200];
    let mut value = 0usize;
    while value < 100 {
        table[value * 2] = b'0' + (value / 10) as u8;
        table[value * 2 + 1] = b'0' + (value % 10) as u8;
        value += 1;
    }
    table
}

static TWO_DIGITS: [u8; 200] = two_digit_table();

/// Unpadded 0..=9999 table: low 32 bits store 1..=4 ASCII digits in LE order,
/// high 32 bits store the byte length (1..=4).
const fn build_dec_10k() -> [u64; 10_000] {
    let mut table = [0u64; 10_000];
    let mut v = 0usize;
    while v < 10_000 {
        let d3 = b'0' + (v / 1000) as u8;
        let d2 = b'0' + ((v / 100) % 10) as u8;
        let d1 = b'0' + ((v / 10) % 10) as u8;
        let d0 = b'0' + (v % 10) as u8;
        let (bytes, len) = if v < 10 {
            ([d0, 0, 0, 0], 1u64)
        } else if v < 100 {
            ([d1, d0, 0, 0], 2u64)
        } else if v < 1000 {
            ([d2, d1, d0, 0], 3u64)
        } else {
            ([d3, d2, d1, d0], 4u64)
        };
        let word = u32::from_le_bytes(bytes) as u64;
        table[v] = word | (len << 32);
        v += 1;
    }
    table
}

/// Zero-padded 4-digit ASCII table for 0..=9999 in LE byte order (`"0000"`..`"9999"`).
const fn build_dec_padded_10k() -> [u32; 10_000] {
    let mut table = [0u32; 10_000];
    let mut v = 0usize;
    while v < 10_000 {
        let d3 = b'0' + (v / 1000) as u8;
        let d2 = b'0' + ((v / 100) % 10) as u8;
        let d1 = b'0' + ((v / 10) % 10) as u8;
        let d0 = b'0' + (v % 10) as u8;
        table[v] = u32::from_le_bytes([d3, d2, d1, d0]);
        v += 1;
    }
    table
}

const fn build_hex_byte_lut() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut b = 0usize;
    while b < 256 {
        let hi = HEX[b >> 4];
        let lo = HEX[b & 15];
        table[b] = u16::from_le_bytes([hi, lo]);
        b += 1;
    }
    table
}

const fn build_hex_nibble_lut() -> [u8; 256] {
    let mut table = [0xffu8; 256];
    let mut b = b'0';
    while b <= b'9' {
        table[b as usize] = b - b'0';
        b += 1;
    }
    let mut b = b'a';
    while b <= b'f' {
        table[b as usize] = b - b'a' + 10;
        b += 1;
    }
    table
}

static DEC_10K: [u64; 10_000] = build_dec_10k();
static DEC_PADDED_10K: [u32; 10_000] = build_dec_padded_10k();
static HEX_BYTE_LUT: [u16; 256] = build_hex_byte_lut();
static HEX_NIBBLE_LUT: [u8; 256] = build_hex_nibble_lut();

#[inline(always)]
unsafe fn write_decimal_fast(dst: *mut u8, value: u64) -> usize {
    unsafe {
        if value < 10_000 {
            let entry = *DEC_10K.get_unchecked(value as usize);
            dst.cast::<u32>().write_unaligned(entry as u32);
            (entry >> 32) as usize
        } else if value < 100_000_000 {
            let hi = (value / 10_000) as usize;
            let lo = (value % 10_000) as usize;
            let hi_entry = *DEC_10K.get_unchecked(hi);
            let hi_len = (hi_entry >> 32) as usize;
            dst.cast::<u32>().write_unaligned(hi_entry as u32);
            let lo_digits = *DEC_PADDED_10K.get_unchecked(lo);
            dst.add(hi_len).cast::<u32>().write_unaligned(lo_digits);
            hi_len + 4
        } else {
            write_decimal_large(dst, value)
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn write_decimal_large(dst: *mut u8, mut value: u64) -> usize {
    let mut buffer = [0u8; 24];
    let mut position = buffer.len();
    while value >= 10_000 {
        let rem = (value % 10_000) as usize;
        value /= 10_000;
        position -= 4;
        unsafe {
            buffer
                .as_mut_ptr()
                .add(position)
                .cast::<u32>()
                .write_unaligned(*DEC_PADDED_10K.get_unchecked(rem));
        }
    }
    let hi_entry = unsafe { *DEC_10K.get_unchecked(value as usize) };
    let hi_len = (hi_entry >> 32) as usize;
    position -= hi_len;
    let hi_bytes = (hi_entry as u32).to_le_bytes();
    buffer[position..position + hi_len].copy_from_slice(&hi_bytes[..hi_len]);
    let total_len = buffer.len() - position;
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.as_ptr().add(position), dst, total_len);
    }
    total_len
}

#[inline(always)]
fn u32_to_hex8_le(val: u32) -> u64 {
    unsafe {
        let b0 = *HEX_BYTE_LUT.get_unchecked((val >> 24) as usize) as u64;
        let b1 = *HEX_BYTE_LUT.get_unchecked(((val >> 16) & 0xff) as usize) as u64;
        let b2 = *HEX_BYTE_LUT.get_unchecked(((val >> 8) & 0xff) as usize) as u64;
        let b3 = *HEX_BYTE_LUT.get_unchecked((val & 0xff) as usize) as u64;
        b0 | (b1 << 16) | (b2 << 32) | (b3 << 48)
    }
}

#[inline]
fn write_decimal(out: &mut Vec<u8>, value: u64) {
    out.reserve(24);
    unsafe {
        let len = write_decimal_fast(out.as_mut_ptr().add(out.len()), value);
        out.set_len(out.len() + len);
    }
}

#[inline]
fn push_decimal(values: &mut FlatValues, value: u64) {
    let start = values.begin();
    write_decimal(&mut values.bytes, value);
    values.finish(start);
}

fn write_two_digits(out: &mut Vec<u8>, value: u8) {
    out.push(b'0' + value / 10);
    out.push(b'0' + value % 10);
}

fn write_four_digits(out: &mut Vec<u8>, value: u16) {
    out.push(b'0' + (value / 1000) as u8);
    out.push(b'0' + (value / 100 % 10) as u8);
    out.push(b'0' + (value / 10 % 10) as u8);
    out.push(b'0' + (value % 10) as u8);
}

fn read_bytes(cursor: &mut Cursor<'_>) -> Result<Vec<u8>> {
    let len = cursor.uleb()?;
    Ok(cursor.take(len)?.to_vec())
}

#[inline(always)]
fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\r' | b'\n')
}

#[inline(always)]
fn skip_space(data: &[u8], pos: &mut usize) {
    while data.get(*pos).is_some_and(|byte| is_space(*byte)) {
        *pos += 1;
    }
}

#[inline(always)]
fn scan_string(data: &[u8], mut pos: usize) -> Option<usize> {
    if data.get(pos) != Some(&b'"') {
        return None;
    }
    pos += 1;
    let len = data.len();
    #[allow(unused_variables)]
    let ptr = data.as_ptr();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let v_quote = vdupq_n_u8(b'"');
        let v_slash = vdupq_n_u8(b'\\');
        let v_ctrl = vdupq_n_u8(0x20);
        // Most structured values close within one vector. Inspect the first
        // 16 bytes before loading/scanning bytes beyond their closing quote.
        while pos + 16 <= len {
            let chunk = vld1q_u8(ptr.add(pos));
            let special = vorrq_u8(
                vceqq_u8(chunk, v_quote),
                vorrq_u8(vceqq_u8(chunk, v_slash), vcltq_u8(chunk, v_ctrl)),
            );
            let mut nibbles = vget_lane_u64(
                vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(special), 4)),
                0,
            );
            if nibbles != 0 {
                let mut cur = pos;
                loop {
                    let rel = (nibbles.trailing_zeros() >> 2) as usize;
                    let hit_pos = cur + rel;
                    let hit = *ptr.add(hit_pos);
                    if hit == b'"' {
                        return Some(hit_pos + 1);
                    }
                    if hit < 0x20 {
                        return None;
                    }
                    if hit_pos + 1 >= len || *ptr.add(hit_pos + 1) < 0x20 {
                        return None;
                    }
                    cur = hit_pos + 2;
                    let consumed = rel + 2;
                    if cur - pos >= 16 || consumed >= 16 {
                        break;
                    }
                    nibbles >>= consumed << 2;
                    if nibbles == 0 {
                        cur = pos + 16;
                        break;
                    }
                }
                pos = cur;
                continue;
            }
            pos += 16;
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v_quote = _mm_set1_epi8(b'"' as i8);
        let v_slash = _mm_set1_epi8(b'\\' as i8);
        let v_ctrl_max = _mm_set1_epi8(0x1f);
        while pos + 16 <= len {
            let c0 = _mm_loadu_si128(ptr.add(pos).cast::<__m128i>());
            let s0 = _mm_or_si128(
                _mm_cmpeq_epi8(c0, v_quote),
                _mm_or_si128(
                    _mm_cmpeq_epi8(c0, v_slash),
                    _mm_cmpeq_epi8(_mm_max_epu8(c0, v_ctrl_max), v_ctrl_max),
                ),
            );
            let mut m = _mm_movemask_epi8(s0) as u32;
            if m != 0 {
                let mut cur = pos;
                loop {
                    let rel = m.trailing_zeros() as usize;
                    let hit_pos = cur + rel;
                    let hit = *ptr.add(hit_pos);
                    if hit == b'"' {
                        return Some(hit_pos + 1);
                    }
                    if hit < 0x20 {
                        return None;
                    }
                    if hit_pos + 1 >= len || *ptr.add(hit_pos + 1) < 0x20 {
                        return None;
                    }
                    cur = hit_pos + 2;
                    let consumed = rel + 2;
                    if cur - pos >= 16 || consumed >= 16 {
                        break;
                    }
                    m >>= consumed;
                    if m == 0 {
                        cur = pos + 16;
                        break;
                    }
                }
                pos = cur;
                continue;
            }
            pos += 16;
        }
    }
    while pos < len {
        match data[pos] {
            b'"' => return Some(pos + 1),
            b'\\' => {
                if pos + 1 >= len || data[pos + 1] < 0x20 {
                    return None;
                }
                pos += 2;
            }
            byte if byte < 0x20 => return None,
            _ => pos += 1,
        }
    }
    None
}

#[inline(always)]
fn scan_compound(data: &[u8], mut pos: usize) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    let mut close_bits = 0u64;
    let first = *data.get(pos)?;
    let mut depth = match first {
        b'{' => {
            if pos + 1 < len && unsafe { *ptr.add(pos + 1) } == b'}' {
                return Some(pos + 2);
            }
            1usize
        }
        b'[' => {
            if pos + 1 < len && unsafe { *ptr.add(pos + 1) } == b']' {
                return Some(pos + 2);
            }
            close_bits = 1;
            1usize
        }
        _ => return None,
    };
    pos += 1;
    let mut overflow = Vec::new();
    while depth > 0 {
        if pos >= len {
            return None;
        }
        let expected = if depth <= 64 {
            if ((close_bits >> (depth - 1)) & 1) == 0 {
                b'}'
            } else {
                b']'
            }
        } else {
            *overflow.last().unwrap()
        };
        let b = unsafe { *ptr.add(pos) };
        if b > b'"' && b < b'[' {
            pos += 1;
            continue;
        }
        match b {
            b'"' => pos = scan_string(data, pos)?,
            b'{' => {
                if depth < 64 {
                    close_bits &= !(1u64 << depth);
                } else {
                    overflow.push(b'}');
                }
                depth += 1;
                pos += 1;
            }
            b'[' => {
                if depth < 64 {
                    close_bits |= 1u64 << depth;
                } else {
                    overflow.push(b']');
                }
                depth += 1;
                pos += 1;
            }
            byte if byte == expected => {
                if depth > 64 {
                    overflow.pop();
                }
                depth -= 1;
                pos += 1;
            }
            b'}' | b']' | b'\n' | b'\r' => return None,
            _ => pos += 1,
        }
    }
    Some(pos)
}

#[inline(always)]
fn scan_primitive_end(ptr: *const u8, len: usize, pos: usize) -> usize {
    let mut end = pos + 1;
    while end < len {
        let b = unsafe { *ptr.add(end) };
        if b <= b',' || b == b'}' || b == b']' {
            break;
        }
        end += 1;
    }
    end
}

#[inline(always)]
fn scan_value(data: &[u8], pos: usize) -> Option<usize> {
    let first = *data.get(pos)?;
    match first {
        b'"' => scan_string(data, pos),
        b'{' | b'[' => scan_compound(data, pos),
        _ if first <= b',' || first == b'}' || first == b']' => None,
        _ => Some(scan_primitive_end(data.as_ptr(), data.len(), pos)),
    }
}

