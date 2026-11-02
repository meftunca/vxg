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
            let v32 = value as u32;
            let hi = (v32 / 10_000) as usize;
            let lo = (v32 - (hi as u32) * 10_000) as usize;
            let hi_entry = *DEC_10K.get_unchecked(hi);
            let hi_len = (hi_entry >> 32) as usize;
            dst.cast::<u32>().write_unaligned(hi_entry as u32);
            dst.add(hi_len)
                .cast::<u32>()
                .write_unaligned(*DEC_PADDED_10K.get_unchecked(lo));
            hi_len + 4
        } else if value < 1_000_000_000_000 {
            let hi = (value / 100_000_000) as usize;
            let lo8 = value % 100_000_000;
            let hi_entry = *DEC_10K.get_unchecked(hi);
            let hi_len = (hi_entry >> 32) as usize;
            dst.cast::<u32>().write_unaligned(hi_entry as u32);
            let lo_u64 = (*DEC_PADDED_10K.get_unchecked((lo8 / 10_000) as usize) as u64)
                | ((*DEC_PADDED_10K.get_unchecked((lo8 % 10_000) as usize) as u64) << 32);
            dst.add(hi_len).cast::<u64>().write_unaligned(lo_u64);
            hi_len + 8
        } else {
            write_decimal_large(dst, value)
        }
    }
}

#[cold]
#[inline(never)]
unsafe fn write_decimal_large(dst: *mut u8, value: u64) -> usize {
    unsafe {
        if value < 10_000_000_000_000_000 {
            let top8 = value / 100_000_000;
            let bot8 = value % 100_000_000;
            let hi_entry = *DEC_10K.get_unchecked((top8 / 10_000) as usize);
            let hi_len = (hi_entry >> 32) as usize;
            dst.cast::<u32>().write_unaligned(hi_entry as u32);
            dst.add(hi_len)
                .cast::<u32>()
                .write_unaligned(*DEC_PADDED_10K.get_unchecked((top8 % 10_000) as usize));
            let bot_u64 = (*DEC_PADDED_10K.get_unchecked((bot8 / 10_000) as usize) as u64)
                | ((*DEC_PADDED_10K.get_unchecked((bot8 % 10_000) as usize) as u64) << 32);
            dst.add(hi_len + 4).cast::<u64>().write_unaligned(bot_u64);
            hi_len + 12
        } else {
            let top4 = (value / 10_000_000_000_000_000) as usize;
            let rem16 = value % 10_000_000_000_000_000;
            let mid8 = rem16 / 100_000_000;
            let bot8 = rem16 % 100_000_000;
            let hi_entry = *DEC_10K.get_unchecked(top4);
            let hi_len = (hi_entry >> 32) as usize;
            dst.cast::<u32>().write_unaligned(hi_entry as u32);
            let mid_u64 = (*DEC_PADDED_10K.get_unchecked((mid8 / 10_000) as usize) as u64)
                | ((*DEC_PADDED_10K.get_unchecked((mid8 % 10_000) as usize) as u64) << 32);
            let bot_u64 = (*DEC_PADDED_10K.get_unchecked((bot8 / 10_000) as usize) as u64)
                | ((*DEC_PADDED_10K.get_unchecked((bot8 % 10_000) as usize) as u64) << 32);
            dst.add(hi_len).cast::<u64>().write_unaligned(mid_u64);
            dst.add(hi_len + 8).cast::<u64>().write_unaligned(bot_u64);
            hi_len + 16
        }
    }
}

/// Exact division-free `n / d` and `n % d` for `0 <= n < 2^62` and `2 <= d <= 2^62`.
#[derive(Clone, Copy)]
struct FastDiv62 {
    mul: u64,
    shr: u32,
    d: u64,
}

impl FastDiv62 {
    #[inline(always)]
    fn new(d: u64) -> Self {
        if d <= 1 {
            Self { mul: 0, shr: 0, d }
        } else if d.is_power_of_two() {
            Self { mul: 1u64 << 63, shr: d.trailing_zeros() - 1, d }
        } else {
            let k = 63 - d.leading_zeros();
            let mul = (((1u128 << (63 + k)) + (d as u128 - 1)) / (d as u128)) as u64;
            Self { mul, shr: k - 1, d }
        }
    }

    #[inline(always)]
    fn div_rem(self, n: u64) -> (u64, u64) {
        let q = (((n as u128 * self.mul as u128) >> 64) as u64) >> self.shr;
        (q, n - q * self.d)
    }
}

/// Exact-bounds inline copy without overread or overwrite (avoids `memcpy` calls on short tokens).
#[inline(always)]
unsafe fn copy_short_exact(src: *const u8, dst: *mut u8, len: usize) {
    unsafe {
        if len == 1 {
            *dst = *src;
        } else if len <= 4 {
            if len >= 2 {
                let a = src.cast::<u16>().read_unaligned();
                let b = src.add(len - 2).cast::<u16>().read_unaligned();
                dst.cast::<u16>().write_unaligned(a);
                dst.add(len - 2).cast::<u16>().write_unaligned(b);
            }
        } else if len <= 8 {
            let a = src.cast::<u32>().read_unaligned();
            let b = src.add(len - 4).cast::<u32>().read_unaligned();
            dst.cast::<u32>().write_unaligned(a);
            dst.add(len - 4).cast::<u32>().write_unaligned(b);
        } else if len <= 16 {
            let a = src.cast::<u64>().read_unaligned();
            let b = src.add(len - 8).cast::<u64>().read_unaligned();
            dst.cast::<u64>().write_unaligned(a);
            dst.add(len - 8).cast::<u64>().write_unaligned(b);
        } else {
            std::ptr::copy_nonoverlapping(src, dst, len);
        }
    }
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

#[inline(never)]
fn refill_json_cache(data: &[u8], w: usize, cache: &mut (usize, u64, u64)) -> bool {
    let base = w * 64;
    let len = data.len();
    if base + 64 <= len {
        let (q, p) = unsafe { json_struct_masks64(data.as_ptr().add(base)) };
        *cache = (w, q, p);
        true
    } else if base < len {
        let rem = len - base;
        let mut tail = [0u8; 64];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(base), tail.as_mut_ptr(), rem);
            let (q, p) = json_struct_masks64(tail.as_ptr());
            *cache = (w, q, p);
        }
        true
    } else {
        false
    }
}

#[inline(never)]
fn next_json_string_slow(
    data: &[u8],
    mut from: usize,
    cache: &mut (usize, u64, u64),
) -> Option<usize> {
    let len = data.len();
    let ptr = data.as_ptr();
    loop {
        if from >= len {
            return None;
        }
        let w = from >> 6;
        if cache.0 != w && !refill_json_cache(data, w, cache) {
            return None;
        }
        let m = cache.1 >> (from & 63);
        if m != 0 {
            let end = from + m.trailing_zeros() as usize;
            if end >= len {
                return None;
            }
            let hit = unsafe { *ptr.add(end) };
            if hit == b'"' {
                return Some(end + 1);
            }
            if hit < 0x20 || end + 1 >= len || unsafe { *ptr.add(end + 1) } < 0x20 {
                return None;
            }
            from = end + 2;
        } else {
            from = (w + 1) * 64;
        }
    }
}

