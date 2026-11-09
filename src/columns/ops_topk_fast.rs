// Zero-arena SIMD/SWAR Top-K scanners for `MODEL_DECIMAL_PACKED`, `MODEL_DECIMAL_AFFINE`,
// and unpatched `MODEL_DECIMAL_DELTA` columns.

#[inline(always)]
fn max_u16_chunk(buf: &[u16; 64], cnt: usize) -> u16 {
    #[cfg(target_arch = "aarch64")]
    if cnt == 64 {
        unsafe {
            use core::arch::aarch64::*;
            let p = buf.as_ptr();
            let m0 = vmaxq_u16(vld1q_u16(p), vld1q_u16(p.add(8)));
            let m1 = vmaxq_u16(vld1q_u16(p.add(16)), vld1q_u16(p.add(24)));
            let m2 = vmaxq_u16(vld1q_u16(p.add(32)), vld1q_u16(p.add(40)));
            let m3 = vmaxq_u16(vld1q_u16(p.add(48)), vld1q_u16(p.add(56)));
            return vmaxvq_u16(vmaxq_u16(vmaxq_u16(m0, m1), vmaxq_u16(m2, m3)));
        }
    }
    let mut hi = 0u16;
    for &v in &buf[..cnt] {
        hi = hi.max(v);
    }
    hi
}

#[inline(always)]
fn min_u16_chunk(buf: &[u16; 64], cnt: usize) -> u16 {
    #[cfg(target_arch = "aarch64")]
    if cnt == 64 {
        unsafe {
            use core::arch::aarch64::*;
            let p = buf.as_ptr();
            let m0 = vminq_u16(vld1q_u16(p), vld1q_u16(p.add(8)));
            let m1 = vminq_u16(vld1q_u16(p.add(16)), vld1q_u16(p.add(24)));
            let m2 = vminq_u16(vld1q_u16(p.add(32)), vld1q_u16(p.add(40)));
            let m3 = vminq_u16(vld1q_u16(p.add(48)), vld1q_u16(p.add(56)));
            return vminvq_u16(vminq_u16(vminq_u16(m0, m1), vminq_u16(m2, m3)));
        }
    }
    let mut lo = u16::MAX;
    for &v in &buf[..cnt] {
        lo = lo.min(v);
    }
    lo
}

#[allow(clippy::too_many_arguments)]
fn try_scan_decimal_topk(
    order_col_bytes: &[u8],
    ones: usize,
    mask: &TplMask,
    row_map: &mut TplRowMap<'_>,
    base_row: usize,
    digit_delim: bool,
    shared_bound: Option<&std::sync::atomic::AtomicU64>,
    outer: &crate::ops::top_k::TopKHeap,
    cands: &mut CandTracker,
) -> Result<bool> {
    use crate::ops::top_k::SortKeyRef;
    use crate::ops::SortDir;

    if digit_delim || ones < 2 {
        return Ok(false);
    }
    match order_col_bytes.first().copied() {
        Some(MODEL_DECIMAL_AFFINE) => {
            let mut c = Cursor::new(&order_col_bytes[1..]);
            let first = c.u64()?;
            let step_i64 = c.i64()?;
            if !c.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let last_i128 = i128::from(first) + i128::from(step_i64) * ((ones - 1) as i128);
            let Ok(last) = u64::try_from(last_i128) else {
                return Err(invalid("column affine decimal overflow"));
            };
            let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                return Ok(true);
            };
            if !num_cmp_overlaps_bounds(cmp, first.min(last), first.max(last)) {
                return Ok(true);
            }
            let step_u64 = step_i64 as u64;
            match mask {
                TplMask::All => {
                    for ti in 0..ones {
                        let v = first.wrapping_add(step_u64.wrapping_mul(ti as u64));
                        if cmp.test(v) {
                            let b_row = row_map.row_of(ti);
                            cands.try_push(SortKeyRef::Num(v), ti, b_row, base_row + b_row, outer);
                        }
                    }
                }
                TplMask::Bits(bs, _) => {
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        let mut w = w0;
                        let base = wi << 6;
                        while w != 0 {
                            let ti = base + w.trailing_zeros() as usize;
                            let v = first.wrapping_add(step_u64.wrapping_mul(ti as u64));
                            if cmp.test(v) {
                                let b_row = row_map.row_of(ti);
                                cands.try_push(
                                    SortKeyRef::Num(v),
                                    ti,
                                    b_row,
                                    base_row + b_row,
                                    outer,
                                );
                            }
                            w &= w - 1;
                        }
                    }
                }
                TplMask::Empty => {}
            }
            Ok(true)
        }
        Some(MODEL_DECIMAL_PACKED) => {
            let mut c = Cursor::new(&order_col_bytes[1..]);
            let min = c.u64()?;
            let width = c.byte()?;
            if width > 64 {
                return Err(invalid("column packed decimal width"));
            }
            let packed = c.take(bit_bytes(ones, width)?)?;
            if !c.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(packed, ones, width)?;
            let max_r = if width == 0 {
                0
            } else if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            let Some(max_val) = min.checked_add(max_r) else {
                return Ok(false);
            };
            let Some(init_cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                return Ok(true);
            };
            if !num_cmp_overlaps_bounds(init_cmp, min, max_val) {
                return Ok(true);
            }
            if width == 0 {
                push_const_key_cands(
                    SortKeyRef::Num(min),
                    ones,
                    mask,
                    row_map,
                    base_row,
                    outer,
                    cands,
                );
                return Ok(true);
            }
            if width > 32 {
                return Ok(false);
            }
            match (outer.dir(), init_cmp, mask) {
                (SortDir::Desc, crate::query::NumCmp::Ge(mut lo), TplMask::All) if width <= 16 => {
                    let mut r_buf = [0u16; 64];
                    let mut row = 0usize;
                    while row < ones && lo <= max_val {
                        let cnt = (ones - row).min(64);
                        let need_r = lo.saturating_sub(min) as u16;
                        unpack_u16_chunk64(packed, row, cnt, width, &mut r_buf);
                        if max_u16_chunk(&r_buf, cnt) >= need_r {
                            for (j, &r) in r_buf[..cnt].iter().enumerate() {
                                let v = min + u64::from(r);
                                if v >= lo {
                                    let ti = row + j;
                                    let b_row = row_map.row_of(ti);
                                    if cands.try_push(
                                        SortKeyRef::Num(v),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    ) {
                                        if let Some(nb) = cands.tight_num_bound() {
                                            lo = lo.max(nb);
                                        }
                                    }
                                }
                            }
                        }
                        row += cnt;
                    }
                }
                (SortDir::Asc, crate::query::NumCmp::Le(mut hi), TplMask::All) if width <= 16 => {
                    let mut r_buf = [0u16; 64];
                    let mut row = 0usize;
                    while row < ones && hi >= min {
                        let cnt = (ones - row).min(64);
                        let allow_r = (hi - min).min(u64::from(u16::MAX)) as u16;
                        unpack_u16_chunk64(packed, row, cnt, width, &mut r_buf);
                        if min_u16_chunk(&r_buf, cnt) <= allow_r {
                            for (j, &r) in r_buf[..cnt].iter().enumerate() {
                                let v = min + u64::from(r);
                                if v <= hi {
                                    let ti = row + j;
                                    let b_row = row_map.row_of(ti);
                                    if cands.try_push(
                                        SortKeyRef::Num(v),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    ) {
                                        if let Some(nb) = cands.tight_num_bound() {
                                            hi = hi.min(nb);
                                        }
                                    }
                                }
                            }
                        }
                        row += cnt;
                    }
                }
                (_, _, TplMask::All) => {
                    let mut r_buf = [0u32; 64];
                    let mut cmp = init_cmp;
                    let mut row = 0usize;
                    while row < ones && num_cmp_overlaps_bounds(cmp, min, max_val) {
                        let cnt = (ones - row).min(64);
                        unpack_u32_chunk64(packed, row, cnt, width, &mut r_buf);
                        let (mut c_lo, mut c_hi) = (u32::MAX, 0u32);
                        for &r in &r_buf[..cnt] {
                            c_lo = c_lo.min(r);
                            c_hi = c_hi.max(r);
                        }
                        if num_cmp_overlaps_bounds(cmp, min + u64::from(c_lo), min + u64::from(c_hi))
                        {
                            for (j, &r) in r_buf[..cnt].iter().enumerate() {
                                let v = min + u64::from(r);
                                if cmp.test(v) {
                                    let ti = row + j;
                                    let b_row = row_map.row_of(ti);
                                    if cands.try_push(
                                        SortKeyRef::Num(v),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    ) {
                                        if let Some(nb) = cands.tight_num_bound() {
                                            cmp = match outer.dir() {
                                                SortDir::Desc => crate::query::NumCmp::Ge(nb),
                                                SortDir::Asc => crate::query::NumCmp::Le(nb),
                                            };
                                        }
                                    }
                                }
                            }
                        }
                        row += cnt;
                    }
                }
                (_, _, TplMask::Bits(bs, _)) => {
                    let mut r_buf = [0u32; 64];
                    let mut cmp = init_cmp;
                    for (wi, &w0) in bs.words().iter().enumerate() {
                        if w0 == 0 {
                            continue;
                        }
                        let row = wi << 6;
                        let cnt = (ones - row).min(64);
                        if w0.count_ones() <= 16 {
                            let mut w = w0;
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                let ti = row + bit;
                                let v =
                                    min + read_packed_bits_at(packed, ti * width as usize, width);
                                if cmp.test(v) {
                                    let b_row = row_map.row_of(ti);
                                    if cands.try_push(
                                        SortKeyRef::Num(v),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    ) {
                                        if let Some(nb) = cands.tight_num_bound() {
                                            cmp = match outer.dir() {
                                                SortDir::Desc => crate::query::NumCmp::Ge(nb),
                                                SortDir::Asc => crate::query::NumCmp::Le(nb),
                                            };
                                        }
                                    }
                                }
                                w &= w - 1;
                            }
                        } else {
                            unpack_u32_chunk64(packed, row, cnt, width, &mut r_buf);
                            let mut w = w0;
                            while w != 0 {
                                let bit = w.trailing_zeros() as usize;
                                let ti = row + bit;
                                let v = min + u64::from(r_buf[bit]);
                                if cmp.test(v) {
                                    let b_row = row_map.row_of(ti);
                                    if cands.try_push(
                                        SortKeyRef::Num(v),
                                        ti,
                                        b_row,
                                        base_row + b_row,
                                        outer,
                                    ) {
                                        if let Some(nb) = cands.tight_num_bound() {
                                            cmp = match outer.dir() {
                                                SortDir::Desc => crate::query::NumCmp::Ge(nb),
                                                SortDir::Asc => crate::query::NumCmp::Le(nb),
                                            };
                                        }
                                    }
                                }
                                w &= w - 1;
                            }
                        }
                    }
                }
                (_, _, TplMask::Empty) => {}
            }
            Ok(true)
        }
        Some(MODEL_DECIMAL_DELTA) => {
            if ones < 2 {
                return Ok(false);
            }
            let mut c = Cursor::new(&order_col_bytes[1..]);
            let first = c.u64()?;
            let w_hdr = c.byte()?;
            if (w_hdr & PFOR_DELTA_FLAG) != 0 {
                let bw = w_hdr & 0x7f;
                if bw >= 64 {
                    return Err(invalid("column pfor delta width"));
                }
                let min_val = read_uleb64(&mut c)?;
                let span = read_uleb64(&mut c)?;
                let max_val = min_val
                    .checked_add(span)
                    .ok_or_else(|| invalid("column pfor delta bounds"))?;
                if first < min_val || first > max_val {
                    return Err(invalid("column pfor delta range"));
                }
                let steps = ones - 1;
                let exc_count = c.uleb()?;
                if exc_count == 0 || exc_count > steps {
                    return Err(invalid("column pfor delta exception count"));
                }
                if bw > 0 {
                    let packed = c.take(bit_bytes(steps, bw)?)?;
                    let _ = FastBitReader::new_validated(packed, steps, bw)?;
                }
                let min_exc = if bw == 0 { 1u64 } else { 1u64 << bw };
                let mut prev_idx: Option<usize> = None;
                for _ in 0..exc_count {
                    let gap = c.uleb()?;
                    let idx = match prev_idx {
                        None => gap,
                        Some(p) => p
                            .checked_add(1)
                            .and_then(|v| v.checked_add(gap))
                            .ok_or_else(|| invalid("column pfor delta exception index"))?,
                    };
                    if idx >= steps {
                        return Err(invalid("column pfor delta exception index"));
                    }
                    let patch_zz = read_uleb64(&mut c)?;
                    if patch_zz < min_exc {
                        return Err(invalid("column pfor delta noncanonical exception"));
                    }
                    prev_idx = Some(idx);
                }
                if !c.done() {
                    return Err(invalid("column model trailing bytes"));
                }
                let Some(init_cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                    return Ok(true);
                };
                if !num_cmp_overlaps_bounds(init_cmp, min_val, max_val) {
                    return Ok(true);
                }
                return Ok(false);
            }
            let d_width = w_hdr;
            if d_width > 64 {
                return Err(invalid("column delta decimal width"));
            }
            if d_width > 32 || first > (1u64 << 62) {
                return Ok(false);
            }
            let steps = ones - 1;
            let d_packed = c.take(bit_bytes(steps, d_width)?)?;
            if !c.done() {
                return Err(invalid("column model trailing bytes"));
            }
            let _ = FastBitReader::new_validated(d_packed, steps, d_width)?;
            if d_width == 0 {
                if let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) {
                    if cmp.test(first) {
                        push_const_key_cands(
                            SortKeyRef::Num(first),
                            ones,
                            mask,
                            row_map,
                            base_row,
                            outer,
                            cands,
                        );
                    }
                }
                return Ok(true);
            }
            let max_drop = (steps as u64) * (1u64 << (d_width - 1));
            let max_rise = (steps as u64) * (((1u64 << d_width) - 1) >> 1);
            if first >= max_drop {
                let Some(cmp) = outer.num_accept_cmp(shared_bound, base_row) else {
                    return Ok(true);
                };
                if !num_cmp_overlaps_bounds(cmp, first - max_drop, first + max_rise) {
                    return Ok(true);
                }
            }
            let mut cmp_opt = outer.num_accept_cmp(shared_bound, base_row);
            let (mut cur, mut neg_or) = (first, first);
            let (mut z_buf, mut v_buf) = ([0u32; 64], [0u64; 64]);
            let (mut step_pos, mut row) = (0usize, 0usize);
            while row < ones {
                let n = (ones - row).min(64);
                let (mut c_lo, mut c_hi);
                let start_j = if row == 0 {
                    v_buf[0] = cur;
                    c_lo = cur;
                    c_hi = cur;
                    1
                } else {
                    c_lo = u64::MAX;
                    c_hi = 0u64;
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
                        c_lo = c_lo.min(cur);
                        c_hi = c_hi.max(cur);
                        v_buf[start_j + j] = cur;
                    }
                    step_pos += step_cnt;
                }
                if (neg_or >> 63) != 0 {
                    return Err(invalid("column delta decimal range"));
                }
                if let Some(ref mut cmp) = cmp_opt {
                    if num_cmp_overlaps_bounds(*cmp, c_lo, c_hi) {
                        match mask {
                            TplMask::All => {
                                for (j, &v) in v_buf[..n].iter().enumerate() {
                                    if cmp.test(v) {
                                        let ti = row + j;
                                        let b_row = row_map.row_of(ti);
                                        if cands.try_push(
                                            SortKeyRef::Num(v),
                                            ti,
                                            b_row,
                                            base_row + b_row,
                                            outer,
                                        ) {
                                            if let Some(nb) = cands.tight_num_bound() {
                                                *cmp = match outer.dir() {
                                                    SortDir::Desc => crate::query::NumCmp::Ge(nb),
                                                    SortDir::Asc => crate::query::NumCmp::Le(nb),
                                                };
                                            }
                                        }
                                    }
                                }
                            }
                            TplMask::Bits(bs, _) => {
                                let mut w = bs.words()[row >> 6];
                                while w != 0 {
                                    let bit = w.trailing_zeros() as usize;
                                    let v = v_buf[bit];
                                    if cmp.test(v) {
                                        let ti = row + bit;
                                        let b_row = row_map.row_of(ti);
                                        if cands.try_push(
                                            SortKeyRef::Num(v),
                                            ti,
                                            b_row,
                                            base_row + b_row,
                                            outer,
                                        ) {
                                            if let Some(nb) = cands.tight_num_bound() {
                                                *cmp = match outer.dir() {
                                                    SortDir::Desc => crate::query::NumCmp::Ge(nb),
                                                    SortDir::Asc => crate::query::NumCmp::Le(nb),
                                                };
                                            }
                                        }
                                    }
                                    w &= w - 1;
                                }
                            }
                            TplMask::Empty => {}
                        }
                    }
                }
                row += n;
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
