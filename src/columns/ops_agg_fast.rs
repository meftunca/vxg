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

/// Fast zero-alloc stack-bucket `GROUP BY` for narrow `MODEL_DECIMAL_PACKED` group columns (`width <= 8`).
#[allow(clippy::too_many_arguments)]
fn try_group_by_decimal_packed_fast<'a>(
    g_col_bytes: &'a [u8],
    a_slot_bytes: Option<(&'a [u8], crate::ops::ColTarget<'_>)>,
    ones: usize,
    mask: &TplMask,
    group_col: crate::ops::ColTarget<'_>,
    out_map: &mut crate::ops::agg::GroupMap,
    hdr_raw_len: usize,
    raw_mark: usize,
    slice_mark: usize,
    scratch: &mut DecodeScratch,
) -> Result<bool> {
    let pop = mask.popcount(ones);
    if pop == 0 {
        return Ok(true);
    }
    if matches!(
        group_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
        return Ok(false);
    }
    if ones < 2 || g_col_bytes.first() != Some(&MODEL_DECIMAL_PACKED) {
        return Ok(false);
    }
    let mut g_cur = Cursor::new(&g_col_bytes[1..]);
    let g_min = g_cur.u64()?;
    let g_width = g_cur.byte()?;
    if g_width > 64 {
        return Err(invalid("column packed decimal width"));
    }
    if g_width > 8 {
        return Ok(false);
    }
    let g_packed_len = bit_bytes(ones, g_width)?;
    let g_packed = g_cur.take(g_packed_len)?;
    if !g_cur.done() {
        return Err(invalid("column model trailing bytes"));
    }
    let _ = FastBitReader::new_validated(g_packed, ones, g_width)?;
    let g_max_r = if g_width == 0 { 0 } else { (1u64 << g_width) - 1 };
    if g_min.checked_add(g_max_r).is_none() {
        return Ok(false);
    }

    if g_width == 0 {
        let mut st = crate::ops::AggStats::default();
        if let Some((a_bytes, acol)) = a_slot_bytes {
            aggregate_slot_with_mask(
                a_bytes, ones, mask, acol, &mut st, hdr_raw_len, raw_mark, slice_mark, scratch,
            )?;
        } else {
            st.add_rows_only(pop as u64);
        }
        out_map.merge_u64_key(group_col, g_min, &st);
        return Ok(true);
    }

    let entries = 1usize << g_width;
    let mut g_buf = [0u16; 64];

    if let Some((a_bytes, acol)) = a_slot_bytes {
        let mut inline_b = [crate::ops::AggStats::default(); 16];
        let mut big_b = if entries > 16 {
            vec![crate::ops::AggStats::default(); entries]
        } else {
            Vec::new()
        };
        let buckets: &mut [crate::ops::AggStats] = if entries <= 16 {
            &mut inline_b[..entries]
        } else {
            &mut big_b[..entries]
        };

        let a_digit_delim = matches!(
            acol,
            crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
        );
        let fast_a_packed = if !a_digit_delim && a_bytes.first() == Some(&MODEL_DECIMAL_PACKED) {
            let mut ac = Cursor::new(&a_bytes[1..]);
            let a_min = ac.u64()?;
            let a_width = ac.byte()?;
            if a_width > 64 {
                return Err(invalid("column packed decimal width"));
            }
            let a_len = bit_bytes(ones, a_width)?;
            let a_pk = ac.take(a_len)?;
            if !ac.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(a_pk, ones, a_width)?;
            let a_max_r = if a_width == 0 {
                0
            } else if a_width == 64 {
                u64::MAX
            } else {
                (1u64 << a_width) - 1
            };
            if a_width <= 16 && a_min.checked_add(a_max_r).is_some() {
                Some((a_min, a_width, a_pk))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((a_min, a_width, a_packed)) = fast_a_packed {
            let mut a_buf = [0u16; 64];
            match mask {
                TplMask::All => {
                    let mut start_row = 0usize;
                    while start_row < ones {
                        let cnt = (ones - start_row).min(64);
                        unpack_u16_chunk64(g_packed, start_row, cnt, g_width, &mut g_buf);
                        if a_width == 0 {
                            for &gd in &g_buf[..cnt] {
                                buckets[gd as usize].add_num(a_min);
                            }
                        } else {
                            unpack_u16_chunk64(a_packed, start_row, cnt, a_width, &mut a_buf);
                            for (&gd, &ad) in g_buf[..cnt].iter().zip(a_buf[..cnt].iter()) {
                                buckets[gd as usize].add_num(a_min + u64::from(ad));
                            }
                        }
                        start_row += cnt;
                    }
                }
                TplMask::Bits(bs, _) => {
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        if w0 == 0 {
                            continue;
                        }
                        let start_row = wi << 6;
                        let cnt = (ones - start_row).min(64);
                        let mut w = w0;
                        if w0.count_ones() <= 16 {
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                let idx = start_row + bit;
                                let gd = read_packed_bits_at(
                                    g_packed,
                                    idx * (g_width as usize),
                                    g_width,
                                ) as usize;
                                let av = if a_width == 0 {
                                    a_min
                                } else {
                                    a_min
                                        + read_packed_bits_at(
                                            a_packed,
                                            idx * (a_width as usize),
                                            a_width,
                                        )
                                };
                                buckets[gd].add_num(av);
                                w &= w - 1;
                            }
                        } else {
                            unpack_u16_chunk64(g_packed, start_row, cnt, g_width, &mut g_buf);
                            if a_width > 0 {
                                unpack_u16_chunk64(a_packed, start_row, cnt, a_width, &mut a_buf);
                            }
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                let av = if a_width == 0 {
                                    a_min
                                } else {
                                    a_min + u64::from(a_buf[bit])
                                };
                                buckets[g_buf[bit] as usize].add_num(av);
                                w &= w - 1;
                            }
                        }
                    }
                }
                TplMask::Empty => unreachable!(),
            }
        } else {
            let a_view = build_slot_num_view(a_bytes, ones, acol, hdr_raw_len, scratch)?;
            match mask {
                TplMask::All => {
                    let mut start_row = 0usize;
                    while start_row < ones {
                        let cnt = (ones - start_row).min(64);
                        unpack_u16_chunk64(g_packed, start_row, cnt, g_width, &mut g_buf);
                        for (i, &gd) in g_buf[..cnt].iter().enumerate() {
                            add_parsed_slot_num(
                                &mut buckets[gd as usize],
                                a_view.get(start_row + i, scratch),
                                1,
                            );
                        }
                        start_row += cnt;
                    }
                }
                TplMask::Bits(bs, _) => {
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        if w0 == 0 {
                            continue;
                        }
                        let start_row = wi << 6;
                        let mut w = w0;
                        while w != 0 {
                            let bit = w.trailing_zeros() as usize;
                            let idx = start_row + bit;
                            let gd = read_packed_bits_at(
                                g_packed,
                                idx * (g_width as usize),
                                g_width,
                            ) as usize;
                            add_parsed_slot_num(&mut buckets[gd], a_view.get(idx, scratch), 1);
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => unreachable!(),
            }
            reset_scratch_to(scratch, raw_mark, slice_mark);
        }

        for (d, b) in buckets.iter().enumerate() {
            if b.row_count > 0 {
                out_map.merge_u64_key(group_col, g_min + d as u64, b);
            }
        }
        return Ok(true);
    }

    let mut freq_small = [0u32; 16];
    let mut freq_big = [0u32; 256];
    let freq: &mut [u32] = if entries <= 16 {
        &mut freq_small[..entries]
    } else {
        &mut freq_big[..entries]
    };
    match mask {
        TplMask::All => {
            let mut start_row = 0usize;
            while start_row < ones {
                let cnt = (ones - start_row).min(64);
                unpack_u16_chunk64(g_packed, start_row, cnt, g_width, &mut g_buf);
                for &gd in &g_buf[..cnt] {
                    freq[gd as usize] += 1;
                }
                start_row += cnt;
            }
        }
        TplMask::Bits(bs, _) => {
            for (wi, &w0) in bs.words().iter().enumerate() {
                if w0 == 0 {
                    continue;
                }
                let start_row = wi << 6;
                let cnt = (ones - start_row).min(64);
                let mut w = w0;
                if w0.count_ones() <= 20 {
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let gd = read_packed_bits_at(
                            g_packed,
                            (start_row + bit) * (g_width as usize),
                            g_width,
                        ) as usize;
                        freq[gd] += 1;
                        w &= w - 1;
                    }
                } else {
                    unpack_u16_chunk64(g_packed, start_row, cnt, g_width, &mut g_buf);
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        freq[g_buf[bit] as usize] += 1;
                        w &= w - 1;
                    }
                }
            }
        }
        TplMask::Empty => unreachable!(),
    }
    for (d, &c) in freq.iter().enumerate() {
        if c > 0 {
            let mut st = crate::ops::AggStats::default();
            st.add_rows_only(u64::from(c));
            out_map.merge_u64_key(group_col, g_min + d as u64, &st);
        }
    }
    Ok(true)
}
