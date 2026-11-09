// High-throughput SWAR / little-endian chunked bit-unpackers (`u16` and `u32`, 64-element blocks).

#[inline(always)]
fn unpack_u16_chunk64(
    packed: &[u8],
    start_row: usize,
    cnt: usize,
    width: u8,
    out: &mut [u16; 64],
) {
    if cnt == 0 {
        return;
    }
    if width == 0 {
        out[..cnt].fill(0);
        return;
    }
    #[cfg(target_endian = "little")]
    if width == 16 {
        let byte_off = start_row * 2;
        let byte_len = cnt * 2;
        if byte_off + byte_len <= packed.len() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    packed.as_ptr().add(byte_off),
                    out.as_mut_ptr().cast::<u8>(),
                    byte_len,
                );
            }
            return;
        }
    }
    if width == 8 && start_row + cnt <= packed.len() {
        let src = &packed[start_row..start_row + cnt];
        for (dst, &b) in out[..cnt].iter_mut().zip(src.iter()) {
            *dst = u16::from(b);
        }
        return;
    }
    let w = width as usize;
    let p_len = packed.len();
    let p_ptr = packed.as_ptr();
    let mut i = 0usize;
    if width <= 7 {
        let m = (1u16 << w) - 1;
        let (w2, w3, w4, w5, w6, w7) = (w * 2, w * 3, w * 4, w * 5, w * 6, w * 7);
        while i + 8 <= cnt {
            let bit_pos = (start_row + i) * w;
            let byte_idx = bit_pos >> 3;
            if byte_idx + 8 > p_len {
                break;
            }
            let x = unsafe { p_ptr.add(byte_idx).cast::<u64>().read_unaligned().to_le() }
                >> (bit_pos & 7);
            out[i] = (x as u16) & m;
            out[i + 1] = ((x >> w) as u16) & m;
            out[i + 2] = ((x >> w2) as u16) & m;
            out[i + 3] = ((x >> w3) as u16) & m;
            out[i + 4] = ((x >> w4) as u16) & m;
            out[i + 5] = ((x >> w5) as u16) & m;
            out[i + 6] = ((x >> w6) as u16) & m;
            out[i + 7] = ((x >> w7) as u16) & m;
            i += 8;
        }
    } else if width <= 15 {
        let m = (1u16 << w) - 1;
        let (w2, w3) = (w * 2, w * 3);
        while i + 4 <= cnt {
            let bit_pos = (start_row + i) * w;
            let byte_idx = bit_pos >> 3;
            if byte_idx + 16 > p_len {
                break;
            }
            let raw = unsafe { p_ptr.add(byte_idx).cast::<u128>().read_unaligned().to_le() };
            let x = (raw >> (bit_pos & 7)) as u64;
            out[i] = (x as u16) & m;
            out[i + 1] = ((x >> w) as u16) & m;
            out[i + 2] = ((x >> w2) as u16) & m;
            out[i + 3] = ((x >> w3) as u16) & m;
            i += 4;
        }
    }
    while i < cnt {
        out[i] = read_packed_bits_at(packed, (start_row + i) * w, width) as u16;
        i += 1;
    }
}

#[inline(always)]
fn unpack_u32_chunk64(
    packed: &[u8],
    start_row: usize,
    cnt: usize,
    width: u8,
    out: &mut [u32; 64],
) {
    if cnt == 0 {
        return;
    }
    if width == 0 {
        out[..cnt].fill(0);
        return;
    }
    if width <= 16 {
        let mut tmp = [0u16; 64];
        unpack_u16_chunk64(packed, start_row, cnt, width, &mut tmp);
        for i in 0..cnt {
            out[i] = u32::from(tmp[i]);
        }
        return;
    }
    #[cfg(target_endian = "little")]
    if width == 32 {
        let byte_off = start_row * 4;
        let byte_len = cnt * 4;
        if byte_off + byte_len <= packed.len() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    packed.as_ptr().add(byte_off),
                    out.as_mut_ptr().cast::<u8>(),
                    byte_len,
                );
            }
            return;
        }
    }
    let w = width as usize;
    let p_len = packed.len();
    let p_ptr = packed.as_ptr();
    let mut i = 0usize;
    if width <= 31 {
        let m = (1u32 << w) - 1;
        while i + 2 <= cnt {
            let bit_pos = (start_row + i) * w;
            let byte_idx = bit_pos >> 3;
            if byte_idx + 16 > p_len {
                break;
            }
            let raw = unsafe { p_ptr.add(byte_idx).cast::<u128>().read_unaligned().to_le() };
            let x = (raw >> (bit_pos & 7)) as u64;
            out[i] = (x as u32) & m;
            out[i + 1] = ((x >> w) as u32) & m;
            i += 2;
        }
    }
    while i < cnt {
        out[i] = read_packed_bits_at(packed, (start_row + i) * w, width) as u32;
        i += 1;
    }
}
