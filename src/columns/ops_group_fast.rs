// Fused compressed-domain `GROUP BY` kernels (`PACKED`, `SCALED_DECIMAL`, `DELTA`, `AFFINE`, `SQL`).

#[allow(clippy::too_many_arguments)]
fn accum_u32_residuals(
    g_packed: &[u8],
    g_width: u8,
    entries: usize,
    needs_bounds: bool,
    a_packed: &[u8],
    a_width: u8,
    ones: usize,
    mask: &TplMask,
    soa: &mut SoaU32,
) -> Result<()> {
    let (cnt_out, r_sum, r_min, r_max) = soa.slices_mut();
    if a_width == 0 {
        accum_group_counts(g_packed, g_width, entries, needs_bounds, ones, mask, cnt_out)?;
        for i in 0..entries {
            if cnt_out[i] > 0 {
                r_min[i] = 0;
            }
        }
        return Ok(());
    }
    match mask {
        TplMask::Empty => Ok(()),
        TplMask::All if g_width == 2 && a_width <= 16 => {
            let (mut c_b, mut s_b, mut lo_b, mut hi_b) =
                ([0u32; 4], [0u64; 4], [u32::MAX; 4], [0u32; 4]);
            let mut a_buf = [0u16; 64];
            let mut start_row = 0usize;
            while start_row < ones {
                let n = (ones - start_row).min(64);
                let byte_off = (start_row >> 6) * 16;
                let lo = read_packed_u64_word(g_packed, byte_off);
                let hi = read_packed_u64_word(g_packed, byte_off + 8);
                if needs_bounds
                    && entries == 3
                    && ((lo & (lo >> 1) | hi & (hi >> 1)) & 0x5555_5555_5555_5555) != 0
                {
                    return Err(invalid("column bit index out of bounds"));
                }
                unpack_u16_chunk64(a_packed, start_row, n, a_width, &mut a_buf);
                let mut gw = lo;
                for &av in &a_buf[..n.min(32)] {
                    let id = (gw & 3) as usize;
                    gw >>= 2;
                    let r = u32::from(av);
                    c_b[id] += 1;
                    s_b[id] += u64::from(r);
                    lo_b[id] = lo_b[id].min(r);
                    hi_b[id] = hi_b[id].max(r);
                }
                if n > 32 {
                    let mut gw2 = hi;
                    for &av in &a_buf[32..n] {
                        let id = (gw2 & 3) as usize;
                        gw2 >>= 2;
                        let r = u32::from(av);
                        c_b[id] += 1;
                        s_b[id] += u64::from(r);
                        lo_b[id] = lo_b[id].min(r);
                        hi_b[id] = hi_b[id].max(r);
                    }
                }
                start_row += n;
            }
            for i in 0..entries.min(4) {
                if c_b[i] > 0 {
                    cnt_out[i] += c_b[i];
                    r_sum[i] += s_b[i];
                    r_min[i] = r_min[i].min(lo_b[i]);
                    r_max[i] = r_max[i].max(hi_b[i]);
                }
            }
            Ok(())
        }
        TplMask::All if a_width <= 16 => {
            let (mut g_buf, mut a_buf) = ([0u16; 64], [0u16; 64]);
            let mut start_row = 0usize;
            while start_row < ones {
                let n = (ones - start_row).min(64);
                unpack_u16_chunk64(g_packed, start_row, n, g_width, &mut g_buf);
                if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                    return Err(invalid("column bit index out of bounds"));
                }
                unpack_u16_chunk64(a_packed, start_row, n, a_width, &mut a_buf);
                for (&gd, &av) in g_buf[..n].iter().zip(a_buf[..n].iter()) {
                    let (id, r) = (gd as usize, u32::from(av));
                    cnt_out[id] += 1;
                    r_sum[id] += u64::from(r);
                    r_min[id] = r_min[id].min(r);
                    r_max[id] = r_max[id].max(r);
                }
                start_row += n;
            }
            Ok(())
        }
        TplMask::All => {
            let (mut g_buf, mut a_buf) = ([0u16; 64], [0u32; 64]);
            let mut start_row = 0usize;
            while start_row < ones {
                let n = (ones - start_row).min(64);
                unpack_u16_chunk64(g_packed, start_row, n, g_width, &mut g_buf);
                if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                    return Err(invalid("column bit index out of bounds"));
                }
                unpack_u32_chunk64(a_packed, start_row, n, a_width, &mut a_buf);
                for (&gd, &r) in g_buf[..n].iter().zip(a_buf[..n].iter()) {
                    let id = gd as usize;
                    cnt_out[id] += 1;
                    r_sum[id] += u64::from(r);
                    r_min[id] = r_min[id].min(r);
                    r_max[id] = r_max[id].max(r);
                }
                start_row += n;
            }
            Ok(())
        }
        TplMask::Bits(bs, _) => {
            let (mut g_buf, mut a_buf) = ([0u16; 64], [0u32; 64]);
            for (wi, &w0) in bs.words().iter().enumerate() {
                if w0 == 0 && !needs_bounds {
                    continue;
                }
                let start_row = wi << 6;
                let n = (ones - start_row).min(64);
                if !needs_bounds && w0.count_ones() <= 16 {
                    let mut w = w0;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let idx = start_row + bit;
                        let id = if g_width == 0 {
                            0
                        } else {
                            read_packed_bits_at(g_packed, idx * g_width as usize, g_width) as usize
                        };
                        let r = read_packed_bits_at(a_packed, idx * a_width as usize, a_width) as u32;
                        cnt_out[id] += 1;
                        r_sum[id] += u64::from(r);
                        r_min[id] = r_min[id].min(r);
                        r_max[id] = r_max[id].max(r);
                        w &= w - 1;
                    }
                } else {
                    unpack_u16_chunk64(g_packed, start_row, n, g_width, &mut g_buf);
                    if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    if w0 == 0 {
                        continue;
                    }
                    unpack_u32_chunk64(a_packed, start_row, n, a_width, &mut a_buf);
                    let mut w = w0;
                    while w != 0 {
                        let bit = w.trailing_zeros() as usize;
                        let (id, r) = (g_buf[bit] as usize, a_buf[bit]);
                        cnt_out[id] += 1;
                        r_sum[id] += u64::from(r);
                        r_min[id] = r_min[id].min(r);
                        r_max[id] = r_max[id].max(r);
                        w &= w - 1;
                    }
                }
            }
            Ok(())
        }
    }
}

fn try_group_by_packed_or_dict_fast(
    g_col_bytes: &[u8],
    a_slot_bytes: Option<(&[u8], crate::ops::ColTarget<'_>)>,
    ones: usize,
    mask: &TplMask,
    group_col: crate::ops::ColTarget<'_>,
    out_map: &mut crate::ops::agg::GroupMap,
) -> Result<bool> {
    if matches!(
        group_col,
        crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()
    ) {
        return Ok(false);
    }
    let Some((backing, g_width, g_packed, needs_bounds)) = parse_group_backing(g_col_bytes, ones)?
    else {
        return Ok(false);
    };
    let entries = backing.entries();

    let Some((a_bytes, acol)) = a_slot_bytes else {
        let (mut c_s, mut c_b) = ([0u32; 16], [0u32; 256]);
        let cnt = if entries <= 16 { &mut c_s[..entries] } else { &mut c_b[..entries] };
        accum_group_counts(g_packed, g_width, entries, needs_bounds, ones, mask, cnt)?;
        backing.flush_rows_only(cnt, group_col, out_map);
        return Ok(true);
    };
    if matches!(acol, crate::ops::ColTarget::Delimited { delim, .. } if delim.is_ascii_digit()) {
        return Ok(false);
    }

    match a_bytes.first().copied() {
        Some(MODEL_DECIMAL_PACKED) => {
            let mut ac = Cursor::new(&a_bytes[1..]);
            let a_min = ac.u64()?;
            let a_width = ac.byte()?;
            if a_width > 64 {
                return Err(invalid("column packed decimal width"));
            }
            let a_packed = ac.take(bit_bytes(ones, a_width)?)?;
            if !ac.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(a_packed, ones, a_width)?;
            let a_max_r = if a_width == 0 {
                0
            } else if a_width == 64 {
                u64::MAX
            } else {
                (1u64 << a_width) - 1
            };
            if a_width > 32 || a_min.checked_add(a_max_r).is_none() {
                return Ok(false);
            }
            let mut soa = SoaU32::new(entries);
            accum_u32_residuals(
                g_packed, g_width, entries, needs_bounds, a_packed, a_width, ones, mask, &mut soa,
            )?;
            backing.flush_u32_residuals(&soa, a_min, group_col, out_map);
            Ok(true)
        }
        Some(MODEL_SCALED_DECIMAL) => {
            let mut ac = Cursor::new(&a_bytes[1..]);
            let scale = ac.byte()?;
            if scale > 9 {
                return Err(invalid("column scaled decimal scale"));
            }
            let min_v = ac.i64()?;
            if scale == 0 && min_v >= 0 {
                return Err(invalid("column scaled decimal noncanonical unsigned"));
            }
            let a_width = ac.byte()?;
            if a_width > 64 {
                return Err(invalid("column scaled decimal width"));
            }
            let a_packed = ac.take(bit_bytes(ones, a_width)?)?;
            if !ac.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(a_packed, ones, a_width)?;
            if a_width == 0 {
                BitReader::new(a_packed).finish_zero_padded()?;
            }
            let max_r = if a_width == 0 {
                0
            } else if a_width == 64 {
                u64::MAX
            } else {
                (1u64 << a_width) - 1
            };
            if i128::from(min_v) + i128::from(max_r) > i128::from(i64::MAX) {
                return Err(invalid("column scaled decimal overflow"));
            }
            if a_width > 32 {
                return Ok(false);
            }
            let mut soa = SoaU32::new(entries);
            accum_u32_residuals(
                g_packed, g_width, entries, needs_bounds, a_packed, a_width, ones, mask, &mut soa,
            )?;
            backing.flush_scaled_residuals(&soa, min_v, scale, group_col, out_map);
            Ok(true)
        }
        Some(MODEL_DECIMAL_AFFINE) => {
            let mut ac = Cursor::new(&a_bytes[1..]);
            let first = ac.u64()?;
            let step_i64 = ac.i64()?;
            if !ac.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let last_i128 = i128::from(first) + i128::from(step_i64) * ((ones - 1) as i128);
            if u64::try_from(last_i128).is_err() {
                return Err(invalid("column affine decimal overflow"));
            }
            let step_u64 = step_i64 as u64;
            let mut soa = SoaU64::new(entries);
            let (cnt, v_sum, v_min, v_max) = soa.slices_mut();
            let mut g_buf = [0u16; 64];
            let mut row = 0usize;
            while row < ones {
                let n = (ones - row).min(64);
                if let TplMask::Bits(bs, _) = mask {
                    if bs.words()[row >> 6] == 0 && !needs_bounds {
                        row += n;
                        continue;
                    }
                }
                unpack_u16_chunk64(g_packed, row, n, g_width, &mut g_buf);
                if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                    return Err(invalid("column bit index out of bounds"));
                }
                match mask {
                    TplMask::All => {
                        for j in 0..n {
                            let id = g_buf[j] as usize;
                            let v = first.wrapping_add(step_u64.wrapping_mul((row + j) as u64));
                            cnt[id] += 1;
                            v_sum[id] += u128::from(v);
                            v_min[id] = v_min[id].min(v);
                            v_max[id] = v_max[id].max(v);
                        }
                    }
                    TplMask::Bits(bs, _) => {
                        let mut w = bs.words()[row >> 6];
                        while w != 0 {
                            let bit = w.trailing_zeros() as usize;
                            let id = g_buf[bit] as usize;
                            let v = first.wrapping_add(step_u64.wrapping_mul((row + bit) as u64));
                            cnt[id] += 1;
                            v_sum[id] += u128::from(v);
                            v_min[id] = v_min[id].min(v);
                            v_max[id] = v_max[id].max(v);
                            w &= w - 1;
                        }
                    }
                    TplMask::Empty => {}
                }
                row += n;
            }
            backing.flush_u64_vals(&soa, group_col, out_map);
            Ok(true)
        }
        Some(MODEL_DECIMAL_DELTA) => {
            let mut ac = Cursor::new(&a_bytes[1..]);
            let first = ac.u64()?;
            let w_hdr = ac.byte()?;
            let mut soa = SoaU64::new(entries);
            let (cnt, v_sum, v_min, v_max) = soa.slices_mut();
            if (w_hdr & PFOR_DELTA_FLAG) != 0 {
                let mut pfor_vals = Vec::with_capacity(ones);
                let _ = decode_pfor_delta_into(first, w_hdr & 0x7f, &mut ac, ones, &mut pfor_vals)?;
                if !ac.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let mut g_buf = [0u16; 64];
                let mut row = 0usize;
                while row < ones {
                    let n = (ones - row).min(64);
                    if let TplMask::Bits(bs, _) = mask {
                        if bs.words()[row >> 6] == 0 && !needs_bounds {
                            row += n;
                            continue;
                        }
                    }
                    unpack_u16_chunk64(g_packed, row, n, g_width, &mut g_buf);
                    if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    match mask {
                        TplMask::All => {
                            for (&gd, &v) in g_buf[..n].iter().zip(pfor_vals[row..row + n].iter()) {
                                let id = gd as usize;
                                cnt[id] += 1;
                                v_sum[id] += u128::from(v);
                                v_min[id] = v_min[id].min(v);
                                v_max[id] = v_max[id].max(v);
                            }
                        }
                        TplMask::Bits(bs, _) => {
                            let mut w = bs.words()[row >> 6];
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                let (id, v) = (g_buf[bit] as usize, pfor_vals[row + bit]);
                                cnt[id] += 1;
                                v_sum[id] += u128::from(v);
                                v_min[id] = v_min[id].min(v);
                                v_max[id] = v_max[id].max(v);
                                w &= w - 1;
                            }
                        }
                        TplMask::Empty => {}
                    }
                    row += n;
                }
            } else {
                let d_width = w_hdr;
                if d_width > 64 {
                    return Err(invalid("column delta decimal width"));
                }
                if d_width > 32 || first > (1u64 << 62) {
                    return Ok(false);
                }
                let steps = ones - 1;
                let d_packed = ac.take(bit_bytes(steps, d_width)?)?;
                if !ac.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let _ = FastBitReader::new_validated(d_packed, steps, d_width)?;
                let (mut cur, mut neg_or) = (first, first);
                let (mut g_buf, mut z_buf, mut v_buf) = ([0u16; 64], [0u32; 64], [0u64; 64]);
                let (mut step_pos, mut row) = (0usize, 0usize);
                while row < ones {
                    let n = (ones - row).min(64);
                    unpack_u16_chunk64(g_packed, row, n, g_width, &mut g_buf);
                    if needs_bounds && g_buf[..n].iter().any(|&id| (id as usize) >= entries) {
                        return Err(invalid("column bit index out of bounds"));
                    }
                    let start_j = if row == 0 {
                        v_buf[0] = cur;
                        1
                    } else {
                        0
                    };
                    let step_cnt = n - start_j;
                    if step_cnt > 0 {
                        unpack_u32_chunk64(d_packed, step_pos, step_cnt, d_width, &mut z_buf);
                        for j in 0..step_cnt {
                            let z = z_buf[j] as i64;
                            let d = (z >> 1) ^ -(z & 1);
                            cur = (cur as i64).wrapping_add(d) as u64;
                            neg_or |= cur;
                            v_buf[start_j + j] = cur;
                        }
                        step_pos += step_cnt;
                    }
                    match mask {
                        TplMask::All => {
                            for j in 0..n {
                                let (id, v) = (g_buf[j] as usize, v_buf[j]);
                                cnt[id] += 1;
                                v_sum[id] += u128::from(v);
                                v_min[id] = v_min[id].min(v);
                                v_max[id] = v_max[id].max(v);
                            }
                        }
                        TplMask::Bits(bs, _) => {
                            let mut w = bs.words()[row >> 6];
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                let (id, v) = (g_buf[bit] as usize, v_buf[bit]);
                                cnt[id] += 1;
                                v_sum[id] += u128::from(v);
                                v_min[id] = v_min[id].min(v);
                                v_max[id] = v_max[id].max(v);
                                w &= w - 1;
                            }
                        }
                        TplMask::Empty => {}
                    }
                    row += n;
                }
                if (neg_or >> 63) != 0 {
                    return Err(invalid("column delta decimal range"));
                }
            }
            backing.flush_u64_vals(&soa, group_col, out_map);
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Zero-per-row-alloc block reducer for single-group-column `MultiGroupMap` SQL queries.
fn try_multi_group_single_fast(
    g_view: &SqlGroupSlotView<'_>,
    a_views: &[SlotNumView],
    ones: usize,
    mask: &TplMask,
    scratch: &DecodeScratch,
    out_map: &mut crate::ops::multi_agg::MultiGroupMap,
) -> bool {
    let n_agg = a_views.len();
    match *g_view {
        SqlGroupSlotView::Const(ref opt_k) => {
            let Some(ref k) = *opt_k else {
                return true;
            };
            let mut row_cnt = 0u64;
            let mut aggs = vec![crate::ops::AggStats::default(); n_agg];
            let mut visit = |idx: usize| {
                row_cnt += 1;
                for (st, av) in aggs.iter_mut().zip(a_views.iter()) {
                    add_parsed_slot_num(st, av.get(idx, scratch), 1);
                }
            };
            match mask {
                TplMask::All => (0..ones).for_each(&mut visit),
                TplMask::Bits(bs, _) => {
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        let mut w = w0;
                        let base = wi << 6;
                        while w != 0 {
                            visit(base + w.trailing_zeros() as usize);
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => {}
            }
            out_map.add_entry(vec![k.clone()], row_cnt, &aggs);
            true
        }
        SqlGroupSlotView::Dict { u8_offset, ref keys } => {
            let entries = keys.len();
            let mut row_cnts = vec![0u64; entries];
            let mut aggs = vec![crate::ops::AggStats::default(); entries * n_agg];
            let ids = &scratch.u8_arena[u8_offset..u8_offset + ones];
            let mut visit = |idx: usize| {
                let b = ids[idx] as usize;
                row_cnts[b] += 1;
                let base = b * n_agg;
                for (j, av) in a_views.iter().enumerate() {
                    add_parsed_slot_num(&mut aggs[base + j], av.get(idx, scratch), 1);
                }
            };
            match mask {
                TplMask::All => (0..ones).for_each(&mut visit),
                TplMask::Bits(bs, _) => {
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        let mut w = w0;
                        let base = wi << 6;
                        while w != 0 {
                            visit(base + w.trailing_zeros() as usize);
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => {}
            }
            for (e, opt_k) in keys.iter().enumerate() {
                if row_cnts[e] > 0 {
                    if let Some(ref k) = *opt_k {
                        out_map.add_entry(
                            vec![k.clone()],
                            row_cnts[e],
                            &aggs[e * n_agg..(e + 1) * n_agg],
                        );
                    }
                }
            }
            true
        }
        SqlGroupSlotView::U64 {
            view: SlotNumView::U16 { u16_offset },
            is_str,
        } => {
            let vals = &scratch.u16_arena[u16_offset..u16_offset + ones];
            let (mut lo, mut hi) = (u16::MAX, 0u16);
            for &v in vals {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if hi < lo || usize::from(hi - lo) >= 256 {
                return false;
            }
            let span = usize::from(hi - lo) + 1;
            let mut row_cnts = vec![0u64; span];
            let mut aggs = vec![crate::ops::AggStats::default(); span * n_agg];
            let mut visit = |idx: usize| {
                let b = usize::from(vals[idx] - lo);
                row_cnts[b] += 1;
                let base = b * n_agg;
                for (j, av) in a_views.iter().enumerate() {
                    add_parsed_slot_num(&mut aggs[base + j], av.get(idx, scratch), 1);
                }
            };
            match mask {
                TplMask::All => (0..ones).for_each(&mut visit),
                TplMask::Bits(bs, _) => {
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        let mut w = w0;
                        let base = wi << 6;
                        while w != 0 {
                            visit(base + w.trailing_zeros() as usize);
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => {}
            }
            for (d, &rc) in row_cnts.iter().enumerate() {
                if rc > 0 {
                    let k = fmt_u64_group_key(u64::from(lo) + d as u64, is_str);
                    out_map.add_entry(vec![k], rc, &aggs[d * n_agg..(d + 1) * n_agg]);
                }
            }
            true
        }
        _ => false,
    }
}
