fn uleb_len(mut value: usize) -> usize {
    let mut len = 1;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

fn put_uleb(out: &mut Vec<u8>, mut value: usize) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return;
        }
    }
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_uleb(out, value.len());
    out.extend_from_slice(value);
}

fn bit_width(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()) as u8
}

fn bit_bytes(count: usize, width: u8) -> Result<usize> {
    count
        .checked_mul(width as usize)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
        .ok_or(Error::SizeOverflow)
}

struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    fn read(&mut self, width: u8) -> Result<u64> {
        let bit = self.bit;
        let end = bit + width as usize;
        if end > self.bytes.len().saturating_mul(8) {
            if self.bytes.len().checked_mul(8).is_none() {
                return Err(Error::SizeOverflow);
            }
            return Err(Error::Truncated);
        }
        let index = bit >> 3;
        let offset = (bit & 7) as u32;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let value = if index + 8 <= self.bytes.len() {
            let chunk = u64::from_le_bytes(
                self.bytes[index..index + 8]
                    .try_into()
                    .map_err(|_| Error::Truncated)?,
            );
            (chunk >> offset) & mask
        } else {
            let mut chunk = 0u64;
            for (position, &byte) in self.bytes[index..].iter().enumerate() {
                chunk |= u64::from(byte) << (8 * position);
            }
            (chunk >> offset) & mask
        };
        self.bit = end;
        Ok(value)
    }

    fn finish_zero_padded(&self) -> Result<()> {
        let used = self.bit;
        let total = self.bytes.len().checked_mul(8).ok_or(Error::SizeOverflow)?;
        if used > total {
            return Err(Error::Truncated);
        }
        for bit in used..total {
            if (self.bytes[bit / 8] >> (bit & 7)) & 1 != 0 {
                return Err(invalid("nonzero column bit padding"));
            }
        }
        Ok(())
    }
}

/// Validated 64-bit register-buffered bit reader for the hot streaming decoder.
///
/// The caller validates `bytes.len() == bit_bytes(count, width)` and tail
/// zero-padding up front via `FastBitReader::new_validated`, so each `next_u64`
/// is a register shift/mask on the hot path.
#[derive(Clone, Copy)]
struct FastBitReader<'a> {
    bytes: &'a [u8],
    bit_end: usize,
    buf: u64,
    avail: u32,
    width: u8,
    mask: u64,
}

impl<'a> FastBitReader<'a> {
    #[inline(always)]
    fn new_validated(bytes: &'a [u8], count: usize, width: u8) -> Result<Self> {
        let expected_len = bit_bytes(count, width)?;
        if bytes.len() != expected_len {
            return Err(Error::Truncated);
        }
        let used_bits = count * (width as usize);
        let tail_bits = used_bits & 7;
        if tail_bits != 0 {
            let last = *bytes.last().ok_or(Error::Truncated)?;
            if (last >> tail_bits) != 0 {
                return Err(invalid("nonzero column bit padding"));
            }
        }
        let mask = if width == 0 {
            0
        } else if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        Ok(Self {
            bytes,
            bit_end: 0,
            buf: 0,
            avail: if width == 0 { u32::MAX } else { 0 },
            width,
            mask,
        })
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let w = self.width as u32;
        if self.avail >= w {
            let val = self.buf & self.mask;
            self.buf = self.buf.wrapping_shr(w);
            self.avail -= w;
            return val;
        }
        self.refill_and_read()
    }

    #[inline(always)]
    fn refill_and_read(&mut self) -> u64 {
        let w = self.width as u32;
        let bit_pos = self.bit_end - (self.avail as usize);
        let byte_idx = bit_pos >> 3;
        let bit_off = (bit_pos & 7) as u32;
        if byte_idx + 8 <= self.bytes.len() && bit_off + w <= 64 {
            let raw = unsafe {
                self.bytes
                    .as_ptr()
                    .add(byte_idx)
                    .cast::<u64>()
                    .read_unaligned()
                    .to_le()
            };
            let shifted = raw >> bit_off;
            let val = shifted & self.mask;
            let total_avail = 64 - bit_off;
            let rem_avail = total_avail - w;
            self.buf = if w < 64 { shifted >> w } else { 0 };
            self.avail = rem_avail;
            self.bit_end = bit_pos + (total_avail as usize);
            val
        } else {
            let mut raw = 0u128;
            let rem = self.bytes.len().saturating_sub(byte_idx).min(16);
            for i in 0..rem {
                raw |= (unsafe { *self.bytes.get_unchecked(byte_idx + i) } as u128) << (8 * i);
            }
            let shifted = raw >> bit_off;
            let val = (shifted as u64) & self.mask;
            let total_avail = (rem as u32 * 8).saturating_sub(bit_off);
            let rem_avail = total_avail.saturating_sub(w).min(64);
            self.buf = (shifted >> w) as u64;
            self.avail = rem_avail;
            self.bit_end = bit_pos + (w as usize) + (rem_avail as usize);
            val
        }
    }
}

#[inline(always)]
fn push_bit_values(
    out: &mut Vec<u8>,
    count: usize,
    width: u8,
    values: impl IntoIterator<Item = u64>,
) {
    if width == 0 || count == 0 {
        return;
    }
    let w = width as u32;
    let needed = (count.saturating_mul(width as usize) + 7) / 8;
    let old_len = out.len();
    out.reserve(needed + 8);
    let mut dst = unsafe { out.as_mut_ptr().add(old_len) };
    if w == 8 {
        for value in values {
            unsafe {
                *dst = value as u8;
                dst = dst.add(1);
            }
        }
        unsafe {
            out.set_len(old_len + needed);
        }
        return;
    }
    let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    let mut acc = 0u64;
    let mut bits = 0u32;
    let mut iter = values.into_iter();
    if w <= 7 {
        let w2 = w * 2;
        let w4 = w * 4;
        let w8 = w * 8;
        for _ in 0..(count / 8) {
            let v0 = iter.next().unwrap_or(0) & mask;
            let v1 = iter.next().unwrap_or(0) & mask;
            let v2 = iter.next().unwrap_or(0) & mask;
            let v3 = iter.next().unwrap_or(0) & mask;
            let v4 = iter.next().unwrap_or(0) & mask;
            let v5 = iter.next().unwrap_or(0) & mask;
            let v6 = iter.next().unwrap_or(0) & mask;
            let v7 = iter.next().unwrap_or(0) & mask;
            let p01 = v0 | (v1 << w);
            let p23 = v2 | (v3 << w);
            let p45 = v4 | (v5 << w);
            let p67 = v6 | (v7 << w);
            let p0123 = p01 | (p23 << w2);
            let p4567 = p45 | (p67 << w2);
            let chunk = p0123 | (p4567 << w4);
            acc |= chunk << bits;
            bits += w8;
            unsafe {
                dst.cast::<u64>().write_unaligned(acc.to_le());
                dst = dst.add((bits >> 3) as usize);
            }
            acc >>= bits & !7;
            bits &= 7;
        }
    } else if w <= 14 {
        let w2 = w * 2;
        let w4 = w * 4;
        for _ in 0..(count / 4) {
            let v0 = iter.next().unwrap_or(0) & mask;
            let v1 = iter.next().unwrap_or(0) & mask;
            let v2 = iter.next().unwrap_or(0) & mask;
            let v3 = iter.next().unwrap_or(0) & mask;
            let p01 = v0 | (v1 << w);
            let p23 = v2 | (v3 << w);
            let chunk = p01 | (p23 << w2);
            acc |= chunk << bits;
            bits += w4;
            unsafe {
                dst.cast::<u64>().write_unaligned(acc.to_le());
                dst = dst.add((bits >> 3) as usize);
            }
            acc >>= bits & !7;
            bits &= 7;
        }
    }
    for value in iter {
        let val = value & mask;
        acc |= val.wrapping_shl(bits);
        let new_bits = bits + w;
        if new_bits >= 64 {
            unsafe {
                dst.cast::<u64>().write_unaligned(acc.to_le());
                dst = dst.add(8);
            }
            let spill = 64 - bits;
            acc = if spill < 64 { val >> spill } else { 0 };
            bits = new_bits - 64;
        } else {
            bits = new_bits;
        }
    }
    if bits > 0 {
        let tail_bytes = ((bits + 7) >> 3) as usize;
        let le = acc.to_le_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(le.as_ptr(), dst, tail_bytes);
        }
    }
    unsafe {
        out.set_len(old_len + needed);
    }
}

/// Append `src` to `dst` without calling `memcpy` for the small pieces that
/// dominate structured columns (segments and values are usually 1..16 bytes).
///
/// Pieces up to 32 bytes are written with two overlapping fixed-width stores;
/// larger pieces fall back to the platform copy. `dst` is reserved with 16
/// bytes of slack so every store stays inside the allocation, and the length
/// advances to exactly `old_len + src.len()`.
#[inline(always)]
fn copy_bytes(dst: &mut Vec<u8>, src: &[u8]) {
    let len = src.len();
    dst.reserve(len + 16);
    let pos = dst.len();
    unsafe {
        let out = dst.as_mut_ptr().add(pos);
        let input = src.as_ptr();
        if len > 32 {
            std::ptr::copy_nonoverlapping(input, out, len);
        } else if len >= 16 {
            std::ptr::copy_nonoverlapping(input, out, 16);
            let tail = len - 16;
            std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 16);
        } else if len >= 8 {
            std::ptr::copy_nonoverlapping(input, out, 8);
            let tail = len - 8;
            std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 8);
        } else if len >= 4 {
            std::ptr::copy_nonoverlapping(input, out, 4);
            let tail = len - 4;
            std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 4);
        } else {
            let mut index = 0;
            while index < len {
                *out.add(index) = *input.add(index);
                index += 1;
            }
        }
        dst.set_len(pos + len);
    }
}

/// Append-only output cursor over one preallocated buffer.
///
/// Row assembly reserves the full block once and then only advances a position:
/// no `Vec` length or capacity work happens per segment or value. The capacity
/// check keeps one bound test per piece; the write itself is the same
/// overlapping-store scheme as [`copy_bytes`].
struct OutSink {
    base: *mut u8,
    pos: usize,
    capacity: usize,
}

impl OutSink {
    fn new(base: *mut u8, capacity: usize) -> Self {
        Self {
            base,
            pos: 0,
            capacity,
        }
    }

    #[inline(always)]
    fn write(&mut self, src: &[u8]) -> Result<()> {
        let len = src.len();
        if self.pos + len > self.capacity {
            return Err(invalid("column output exceeds length"));
        }
        unsafe {
            let out = self.base.add(self.pos);
            let input = src.as_ptr();
            // The overlapping stores need 16 bytes of headroom; the final
            // pieces of an exactly-sized slice take the exact copy instead.
            if len > 32 || self.pos + len + 16 > self.capacity {
                std::ptr::copy_nonoverlapping(input, out, len);
            } else if len >= 16 {
                std::ptr::copy_nonoverlapping(input, out, 16);
                let tail = len - 16;
                std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 16);
            } else if len >= 8 {
                std::ptr::copy_nonoverlapping(input, out, 8);
                let tail = len - 8;
                std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 8);
            } else if len >= 4 {
                std::ptr::copy_nonoverlapping(input, out, 4);
                let tail = len - 4;
                std::ptr::copy_nonoverlapping(input.add(tail), out.add(tail), 4);
            } else {
                let mut index = 0;
                while index < len {
                    *out.add(index) = *input.add(index);
                    index += 1;
                }
            }
        }
        self.pos += len;
        Ok(())
    }
}

/// Flat column storage: all value bytes live in one buffer plus one range per
/// value. Used by the strict oracle decoder.
#[derive(Default)]
struct FlatValues {
    bytes: Vec<u8>,
    ranges: Vec<(u32, u32)>,
}

impl FlatValues {
    fn with_capacity(values: usize, bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes.min(1 << 20)),
            ranges: Vec::with_capacity(values),
        }
    }

    fn len(&self) -> usize {
        self.ranges.len()
    }

    fn get(&self, index: usize) -> Option<&[u8]> {
        let &(offset, len) = self.ranges.get(index)?;
        let start = offset as usize;
        self.bytes.get(start..start.checked_add(len as usize)?)
    }

    #[inline]
    fn begin(&self) -> u32 {
        self.bytes.len() as u32
    }

    #[inline]
    fn finish(&mut self, start: u32) {
        let len = (self.bytes.len() as u32) - start;
        self.ranges.push((start, len));
    }

    #[inline]
    fn push_slice(&mut self, value: &[u8]) {
        let start = self.begin();
        copy_bytes(&mut self.bytes, value);
        self.finish(start);
    }

    fn materialize(&self) -> Vec<Vec<u8>> {
        (0..self.len())
            .map(|index| self.get(index).unwrap_or_default().to_vec())
            .collect()
    }
}

