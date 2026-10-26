#[derive(Clone, Copy)]
enum BoundStream<'a> {
    Raw {
        cursor: Cursor<'a>,
    },
    Dictionary16 {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    Dictionary32 {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    Dictionary48 {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    DictionaryPair32 {
        entries0: *const FastPiece,
        entries1: *const FastPiece,
        idx_ptr0: *const u8,
        idx_delta: isize,
    },
    DictionaryPair48 {
        entries0: *const FastPiece,
        entries1: *const FastPiece,
        idx_ptr0: *const u8,
        idx_delta: isize,
    },
    DictionaryQuad32 {
        entries0: *const FastPiece,
        idx_ptr0: *const u8,
        p_off: [u16; 3],
        u_off: [u16; 3],
    },
    DictionaryQuad48 {
        entries0: *const FastPiece,
        idx_ptr0: *const u8,
        p_off: [u16; 3],
        u_off: [u16; 3],
    },
    DictionaryLong {
        entries: *const FastPiece,
        idx_ptr: *const u8,
    },
    DecimalAffineSmall16 {
        current: u64,
        step: u64,
    },
    DecimalAffineSmall32 {
        current: u64,
        step: u64,
    },
    DecimalAffineSmall48 {
        current: u64,
        step: u64,
    },
    DecimalAffineSmall {
        current: u64,
        step: u64,
    },
    DecimalAffine {
        current: u64,
        step: u64,
    },
    DecimalSmall16 {
        val_ptr: *const u16,
    },
    DecimalSmall32 {
        val_ptr: *const u16,
    },
    DecimalSmall48 {
        val_ptr: *const u16,
    },
    DecimalSmall {
        val_ptr: *const u16,
    },
    DecimalPacked {
        min: u64,
        reader: FastBitReader<'a>,
    },
    DecimalValues {
        val_ptr: *const u64,
    },
    HexSuffix4 {
        ptr: *const u8,
    },
    HexSuffix8 {
        ptr: *const u8,
    },
    HexSuffix {
        cursor: Cursor<'a>,
        width: usize,
    },
    TimestampMinSec16 {
        min_ptr: *const u8,
        sec_ptr: *const u8,
    },
    TimestampMinSec32 {
        min_ptr: *const u8,
        sec_ptr: *const u8,
    },
    TimestampMinSec {
        min_ptr: *const u8,
        sec_ptr: *const u8,
    },
    TimestampFull {
        ptr: *const u8,
        stride: usize,
    },
    Copy16After16 {
        ptr: *const u8,
    },
    Copy16After32 {
        ptr: *const u8,
    },
    Copy16 {
        ptr: *const u8,
    },
    SegmentDict {
        str_ptr: *const u8,
        len_ptr: *const u16,
    },
    PeriodicString {
        cursor: Cursor<'a>,
        period: usize,
    },
    PeriodicHex {
        cursor: Cursor<'a>,
        period: usize,
    },
}

#[derive(Clone, Copy)]
struct ExecColumn<'a> {
    stream: BoundStream<'a>,
    after: FastPiece,
}

impl<'a> ExecColumn<'a> {
    #[inline(always)]
    unsafe fn emit(&mut self, mut dst: *mut u8) -> *mut u8 {
        unsafe {
            match &mut self.stream {
                BoundStream::Dictionary16 { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    return (*entries.add(idx)).emit_16(dst);
                }
                BoundStream::Dictionary32 { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    return (*entries.add(idx)).emit_32(dst);
                }
                BoundStream::Dictionary48 { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    return (*entries.add(idx)).emit_48_exact(dst);
                }
                BoundStream::DictionaryPair32 {
                    entries0,
                    entries1,
                    idx_ptr0,
                    idx_delta,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let idx0 = *p as usize;
                    let idx1 = *p.wrapping_offset(*idx_delta) as usize;
                    let p0 = &*entries0.add(idx0);
                    let p1 = &*entries1.add(idx1);
                    dst = p0.emit_32(dst);
                    return p1.emit_32(dst);
                }
                BoundStream::DictionaryPair48 {
                    entries0,
                    entries1,
                    idx_ptr0,
                    idx_delta,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let idx0 = *p as usize;
                    let idx1 = *p.wrapping_offset(*idx_delta) as usize;
                    let p0 = &*entries0.add(idx0);
                    let p1 = &*entries1.add(idx1);
                    dst = p0.emit_48_exact(dst);
                    return p1.emit_48_exact(dst);
                }
                BoundStream::DictionaryQuad32 {
                    entries0,
                    idx_ptr0,
                    p_off,
                    u_off,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let e0 = *entries0;
                    let i0 = *p as usize;
                    let i1 = *p.add(u_off[0] as usize) as usize;
                    let i2 = *p.add(u_off[1] as usize) as usize;
                    let i3 = *p.add(u_off[2] as usize) as usize;
                    dst = (*e0.add(i0)).emit_32(dst);
                    dst = (*e0.add(p_off[0] as usize + i1)).emit_32(dst);
                    dst = (*e0.add(p_off[1] as usize + i2)).emit_32(dst);
                    return (*e0.add(p_off[2] as usize + i3)).emit_32(dst);
                }
                BoundStream::DictionaryQuad48 {
                    entries0,
                    idx_ptr0,
                    p_off,
                    u_off,
                } => {
                    let p = *idx_ptr0;
                    *idx_ptr0 = p.add(1);
                    let e0 = *entries0;
                    let i0 = *p as usize;
                    let i1 = *p.add(u_off[0] as usize) as usize;
                    let i2 = *p.add(u_off[1] as usize) as usize;
                    let i3 = *p.add(u_off[2] as usize) as usize;
                    dst = (*e0.add(i0)).emit_48_exact(dst);
                    dst = (*e0.add(p_off[0] as usize + i1)).emit_48_exact(dst);
                    dst = (*e0.add(p_off[1] as usize + i2)).emit_48_exact(dst);
                    return (*e0.add(p_off[2] as usize + i3)).emit_48_exact(dst);
                }
                BoundStream::DictionaryLong { entries, idx_ptr } => {
                    let idx = **idx_ptr as usize;
                    *idx_ptr = idx_ptr.add(1);
                    dst = (*entries.add(idx)).emit_48(dst);
                }
                BoundStream::DecimalSmall16 { val_ptr } => {
                    let val = **val_ptr as usize;
                    *val_ptr = val_ptr.add(1);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                    return self.after.emit_16(dst);
                }
                BoundStream::DecimalSmall32 { val_ptr } => {
                    let val = **val_ptr as usize;
                    *val_ptr = val_ptr.add(1);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                    return self.after.emit_32(dst);
                }
                BoundStream::DecimalSmall48 { val_ptr } => {
                    let val = **val_ptr as usize;
                    *val_ptr = val_ptr.add(1);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                    return self.after.emit_48_exact(dst);
                }
                BoundStream::DecimalSmall { val_ptr } => {
                    let val = **val_ptr as usize;
                    *val_ptr = val_ptr.add(1);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                }
                BoundStream::DecimalAffineSmall16 { current, step } => {
                    let val = *current as usize;
                    *current = current.wrapping_add(*step);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                    return self.after.emit_16(dst);
                }
                BoundStream::DecimalAffineSmall32 { current, step } => {
                    let val = *current as usize;
                    *current = current.wrapping_add(*step);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                    return self.after.emit_32(dst);
                }
                BoundStream::DecimalAffineSmall48 { current, step } => {
                    let val = *current as usize;
                    *current = current.wrapping_add(*step);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                    return self.after.emit_48_exact(dst);
                }
                BoundStream::DecimalAffineSmall { current, step } => {
                    let val = *current as usize;
                    *current = current.wrapping_add(*step);
                    let entry = *DEC_10K.get_unchecked(val);
                    dst.cast::<u32>().write_unaligned(entry as u32);
                    dst = dst.add((entry >> 32) as usize);
                }
                BoundStream::DecimalValues { val_ptr } => {
                    let val = **val_ptr;
                    *val_ptr = val_ptr.add(1);
                    let len = write_decimal_fast(dst, val);
                    dst = dst.add(len);
                }
                BoundStream::DecimalAffine { current, step } => {
                    let val = *current;
                    *current = current.wrapping_add(*step);
                    let len = write_decimal_fast(dst, val);
                    dst = dst.add(len);
                    if (1..=16).contains(&self.after.len) {
                        return self.after.emit_16(dst);
                    }
                }
                BoundStream::DecimalPacked { min, reader } => {
                    let val = *min + reader.next_u64();
                    let len = write_decimal_fast(dst, val);
                    dst = dst.add(len);
                }
                BoundStream::TimestampMinSec16 { min_ptr, sec_ptr } => {
                    let min = **min_ptr as usize;
                    *min_ptr = min_ptr.add(1);
                    let sec = **sec_ptr as usize;
                    *sec_ptr = sec_ptr.add(1);
                    let min_16 =
                        TWO_DIGITS.as_ptr().add(min * 2).cast::<u16>().read_unaligned() as u64;
                    let sec_16 =
                        TWO_DIGITS.as_ptr().add(sec * 2).cast::<u16>().read_unaligned() as u64;
                    let word = min_16 | ((b':' as u64) << 16) | (sec_16 << 24);
                    dst.cast::<u64>().write_unaligned(word);
                    dst = dst.add(5);
                    return self.after.emit_16(dst);
                }
                BoundStream::TimestampMinSec32 { min_ptr, sec_ptr } => {
                    let min = **min_ptr as usize;
                    *min_ptr = min_ptr.add(1);
                    let sec = **sec_ptr as usize;
                    *sec_ptr = sec_ptr.add(1);
                    let min_16 =
                        TWO_DIGITS.as_ptr().add(min * 2).cast::<u16>().read_unaligned() as u64;
                    let sec_16 =
                        TWO_DIGITS.as_ptr().add(sec * 2).cast::<u16>().read_unaligned() as u64;
                    let word = min_16 | ((b':' as u64) << 16) | (sec_16 << 24);
                    dst.cast::<u64>().write_unaligned(word);
                    dst = dst.add(5);
                    return self.after.emit_32(dst);
                }
                BoundStream::TimestampMinSec { min_ptr, sec_ptr } => {
                    let min = **min_ptr as usize;
                    *min_ptr = min_ptr.add(1);
                    let sec = **sec_ptr as usize;
                    *sec_ptr = sec_ptr.add(1);
                    let min_16 =
                        TWO_DIGITS.as_ptr().add(min * 2).cast::<u16>().read_unaligned() as u64;
                    let sec_16 =
                        TWO_DIGITS.as_ptr().add(sec * 2).cast::<u16>().read_unaligned() as u64;
                    let word = min_16 | ((b':' as u64) << 16) | (sec_16 << 24);
                    dst.cast::<u64>().write_unaligned(word);
                    dst = dst.add(5);
                }
                BoundStream::TimestampFull { ptr, stride } => {
                    let p = *ptr;
                    *ptr = p.add(1);
                    let s = *stride;
                    let m = *p as usize;
                    let d = *p.add(s) as usize;
                    let h = *p.add(s * 2) as usize;
                    let min = *p.add(s * 3) as usize;
                    let sec = *p.add(s * 4) as usize;

                    let m_16 =
                        TWO_DIGITS.as_ptr().add(m * 2).cast::<u16>().read_unaligned() as u64;
                    let d_16 =
                        TWO_DIGITS.as_ptr().add(d * 2).cast::<u16>().read_unaligned() as u64;
                    let h_16 =
                        TWO_DIGITS.as_ptr().add(h * 2).cast::<u16>().read_unaligned() as u64;
                    let min_16 =
                        TWO_DIGITS.as_ptr().add(min * 2).cast::<u16>().read_unaligned() as u64;
                    let sec_16 =
                        TWO_DIGITS.as_ptr().add(sec * 2).cast::<u16>().read_unaligned() as u64;

                    let w0 = m_16
                        | ((b'-' as u64) << 16)
                        | (d_16 << 24)
                        | ((b'T' as u64) << 40)
                        | (h_16 << 48);
                    let w1 = h_16
                        | ((b':' as u64) << 16)
                        | (min_16 << 24)
                        | ((b':' as u64) << 40)
                        | (sec_16 << 48);
                    dst.cast::<u64>().write_unaligned(w0);
                    dst.add(6).cast::<u64>().write_unaligned(w1);
                    dst = dst.add(14);
                    if (1..=16).contains(&self.after.len) {
                        return self.after.emit_16(dst);
                    }
                }
                BoundStream::Copy16After16 { ptr } => {
                    let p = *ptr;
                    *ptr = p.add(16);
                    dst.cast::<u128>()
                        .write_unaligned(p.cast::<u128>().read_unaligned());
                    dst = dst.add(16);
                    return self.after.emit_16(dst);
                }
                BoundStream::Copy16After32 { ptr } => {
                    let p = *ptr;
                    *ptr = p.add(16);
                    dst.cast::<u128>()
                        .write_unaligned(p.cast::<u128>().read_unaligned());
                    dst = dst.add(16);
                    return self.after.emit_32(dst);
                }
                BoundStream::Copy16 { ptr } => {
                    let p = *ptr;
                    *ptr = p.add(16);
                    dst.cast::<u128>()
                        .write_unaligned(p.cast::<u128>().read_unaligned());
                    dst = dst.add(16);
                }
                BoundStream::HexSuffix4 { ptr } => {
                    write_hex8_from_4bytes(dst, *ptr);
                    *ptr = ptr.add(4);
                    dst = dst.add(8);
                }
                BoundStream::HexSuffix8 { ptr } => {
                    write_hex16_from_8bytes(dst, *ptr);
                    *ptr = ptr.add(8);
                    dst = dst.add(16);
                }
                BoundStream::HexSuffix { cursor, width } => {
                    let w = *width;
                    let raw = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + w);
                    cursor.pos += w;
                    for (i, &b) in raw.iter().enumerate() {
                        let pair = *HEX_BYTE_LUT.get_unchecked(b as usize);
                        dst.add(i * 2).cast::<u16>().write_unaligned(pair);
                    }
                    dst = dst.add(w * 2);
                }
                BoundStream::SegmentDict { str_ptr, len_ptr } => {
                    let len = **len_ptr as usize;
                    *len_ptr = len_ptr.add(1);
                    let p = *str_ptr;
                    *str_ptr = p.add(len);
                    dst.cast::<u128>()
                        .write_unaligned(p.cast::<u128>().read_unaligned());
                    dst.add(16)
                        .cast::<u128>()
                        .write_unaligned(p.add(16).cast::<u128>().read_unaligned());
                    dst.add(32)
                        .cast::<u128>()
                        .write_unaligned(p.add(32).cast::<u128>().read_unaligned());
                    dst.add(48)
                        .cast::<u128>()
                        .write_unaligned(p.add(48).cast::<u128>().read_unaligned());
                    dst.add(64)
                        .cast::<u128>()
                        .write_unaligned(p.add(64).cast::<u128>().read_unaligned());
                    if len > 80 {
                        FastPiece::emit_tail_cold(p, dst, len);
                    }
                    dst = dst.add(len);
                    if (1..=16).contains(&self.after.len) {
                        return self.after.emit_16(dst);
                    }
                }
                BoundStream::Raw { cursor } => {
                    let len = cursor.uleb().unwrap_unchecked();
                    let val = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + len);
                    cursor.pos += len;
                    std::ptr::copy_nonoverlapping(val.as_ptr(), dst, len);
                    dst = dst.add(len);
                }
                BoundStream::PeriodicString { cursor, period } => {
                    let len = cursor.uleb().unwrap_unchecked();
                    let p = *period;
                    let seed_len = len.min(p);
                    let seed = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + seed_len);
                    cursor.pos += seed_len;
                    std::ptr::copy_nonoverlapping(seed.as_ptr(), dst, seed_len);
                    for idx in seed_len..len {
                        *dst.add(idx) = *seed.get_unchecked(idx % p);
                    }
                    dst = dst.add(len);
                }
                BoundStream::PeriodicHex { cursor, period } => {
                    let len = cursor.uleb().unwrap_unchecked();
                    let p = *period;
                    let seed_len = len.min(p);
                    let packed_len = seed_len.div_ceil(2);
                    let packed = cursor.bytes.get_unchecked(cursor.pos..cursor.pos + packed_len);
                    cursor.pos += packed_len;
                    for (idx, &byte) in packed.iter().enumerate() {
                        *dst.add(idx * 2) = HEX[(byte >> 4) as usize];
                        if idx * 2 + 1 < seed_len {
                            *dst.add(idx * 2 + 1) = HEX[(byte & 15) as usize];
                        }
                    }
                    for idx in seed_len..len {
                        *dst.add(idx) = *dst.add(idx % p);
                    }
                    dst = dst.add(len);
                }
            }
            if self.after.len == 0 {
                dst
            } else if self.after.len <= 16 {
                self.after.emit_16(dst)
            } else if self.after.len <= 32 {
                self.after.emit_32(dst)
            } else {
                self.after.emit_48(dst)
            }
        }
    }
}

