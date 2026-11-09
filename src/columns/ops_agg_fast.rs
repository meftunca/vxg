/// Fast zero-arena residual aggregator for `MODEL_DECIMAL_PACKED` columns (`SUM`/`MIN`/`MAX`/`COUNT`).
/// Factors out `min` (`sum(min + r_i) = pop * min + sum(r_i)`) and probes only surviving `RowBitset`
/// positions on filtered queries instead of unpacking the entire column.
fn try_agg_decimal_packed_fast(
    col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    target_col: crate::ops::ColTarget<'_>,
    stats: &mut crate::ops::AggStats,
) -> Result<bool> {
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(true);
    }
    if matches!(
        target_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
        return Ok(false);
    }
    if ones < 2 || col_bytes.first() != Some(&MODEL_DECIMAL_PACKED) {
        return Ok(false);
    }
    let mut cursor = Cursor::new(&col_bytes[1..]);
    let min = cursor.u64()?;
    let width = cursor.byte()?;
    if width > 64 {
        return Err(invalid("column packed decimal width"));
    }
    let packed_len = bit_bytes(ones, width)?;
    let packed_bytes = cursor.take(packed_len)?;
    if !cursor.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let _ = FastBitReader::new_validated(packed_bytes, ones, width)?;
    let max_r = if width == 0 {
        0
    } else if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    if min.checked_add(max_r).is_none() {
        return Ok(false);
    }
    if width == 0 {
        stats.add_const_num(min, pop as u64);
        return Ok(true);
    }

    match mask {
        TplMask::Empty => unreachable!(),
        TplMask::All => {
            if width <= 16 {
                let mut buf = [0u16; 64];
                let mut r_sum = 0u64;
                let mut r_lo = u16::MAX;
                let mut r_hi = 0u16;
                let mut start_row = 0usize;
                while start_row < ones {
                    let cnt = (ones - start_row).min(64);
                    unpack_u16_chunk64(packed_bytes, start_row, cnt, width, &mut buf);
                    #[cfg(target_arch = "aarch64")]
                    if cnt == 64 {
                        unsafe {
                            use core::arch::aarch64::*;
                            let p = buf.as_ptr();
                            for g in 0..8usize {
                                let v = vld1q_u16(p.add(g * 8));
                                r_sum += u64::from(vaddlvq_u16(v));
                                r_lo = r_lo.min(vminvq_u16(v));
                                r_hi = r_hi.max(vmaxvq_u16(v));
                            }
                        }
                        start_row += 64;
                        continue;
                    }
                    for &v in &buf[..cnt] {
                        r_sum += u64::from(v);
                        r_lo = r_lo.min(v);
                        r_hi = r_hi.max(v);
                    }
                    start_row += cnt;
                }
                let lo64 = min + u64::from(r_lo);
                let hi64 = min + u64::from(r_hi);
                stats.row_count += ones as u64;
                stats.num_count += ones as u64;
                stats.sum += u128::from(min) * (ones as u128) + u128::from(r_sum);
                stats.min = Some(stats.min.map_or(lo64, |m| m.min(lo64)));
                stats.max = Some(stats.max.map_or(hi64, |m| m.max(hi64)));
                Ok(true)
            } else {
                let mut rdr = FastBitReader::new_validated(packed_bytes, ones, width)?;
                let mut r_sum = 0u128;
                let mut r_lo = u64::MAX;
                let mut r_hi = 0u64;
                for _ in 0..ones {
                    let r = rdr.next_u64();
                    r_sum += u128::from(r);
                    r_lo = r_lo.min(r);
                    r_hi = r_hi.max(r);
                }
                let lo64 = min + r_lo;
                let hi64 = min + r_hi;
                stats.row_count += ones as u64;
                stats.num_count += ones as u64;
                stats.sum += u128::from(min) * (ones as u128) + r_sum;
                stats.min = Some(stats.min.map_or(lo64, |m| m.min(lo64)));
                stats.max = Some(stats.max.map_or(hi64, |m| m.max(hi64)));
                Ok(true)
            }
        }
        TplMask::Bits(bs, _) => {
            let mut buf = [0u16; 64];
            let mut r_sum = 0u128;
            let mut r_lo = u64::MAX;
            let mut r_hi = 0u64;
            for (wi, &w0) in bs.words().iter().enumerate() {
                if w0 == 0 {
                    continue;
                }
                let start_row = wi << 6;
                let cnt = (ones - start_row).min(64);
                let mut w = w0;
                if w0.count_ones() <= 20 || width > 16 {
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let r = read_packed_bits_at(
                            packed_bytes,
                            (start_row + bit) * (width as usize),
                            width,
                        );
                        r_sum += u128::from(r);
                        r_lo = r_lo.min(r);
                        r_hi = r_hi.max(r);
                        w &= w - 1;
                    }
                } else {
                    unpack_u16_chunk64(packed_bytes, start_row, cnt, width, &mut buf);
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let r = u64::from(buf[bit]);
                        r_sum += u128::from(r);
                        r_lo = r_lo.min(r);
                        r_hi = r_hi.max(r);
                        w &= w - 1;
                    }
                }
            }
            let lo64 = min + r_lo;
            let hi64 = min + r_hi;
            stats.row_count += pop as u64;
            stats.num_count += pop as u64;
            stats.sum += u128::from(min) * (pop as u128) + r_sum;
            stats.min = Some(stats.min.map_or(lo64, |m| m.min(lo64)));
            stats.max = Some(stats.max.map_or(hi64, |m| m.max(hi64)));
            Ok(true)
        }
    }
}
