#[inline(always)]
fn digits(value: &[u8]) -> Option<u16> {
    if value.len() != 2 {
        return None;
    }
    let d0 = value[0].wrapping_sub(b'0');
    let d1 = value[1].wrapping_sub(b'0');
    if d0 > 9 || d1 > 9 {
        return None;
    }
    Some(u16::from(d0) * 10 + u16::from(d1))
}

#[inline(always)]
fn parse_timestamp(value: &[u8]) -> Option<Timestamp> {
    if value.len() != 22
        || value[0] != b'"'
        || value[21] != b'"'
        || value[5] != b'-'
        || value[8] != b'-'
        || value[11] != b'T'
        || value[14] != b':'
        || value[17] != b':'
        || value[20] != b'Z'
    {
        return None;
    }
    let y0 = value[1].wrapping_sub(b'0');
    let y1 = value[2].wrapping_sub(b'0');
    let y2 = value[3].wrapping_sub(b'0');
    let y3 = value[4].wrapping_sub(b'0');
    if y0 > 9 || y1 > 9 || y2 > 9 || y3 > 9 {
        return None;
    }
    let year =
        u16::from(y0) * 1000 + u16::from(y1) * 100 + u16::from(y2) * 10 + u16::from(y3);
    let month = digits(&value[6..8])? as u8;
    let day = digits(&value[9..11])? as u8;
    let hour = digits(&value[12..14])? as u8;
    let minute = digits(&value[15..17])? as u8;
    let second = digits(&value[18..20])? as u8;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour >= 24
        || minute >= 60
        || second >= 60
    {
        return None;
    }
    Some(Timestamp {
        year,
        month,
        day,
        hour,
        minute,
        second,
    })
}

#[inline(always)]
fn pack_timestamp_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 13 >= limit {
        return None;
    }
    let v0 = values[0];
    let first = parse_timestamp(v0)?;
    let year = first.year;
    MODEL_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        sc.parsed_u64.clear();
        sc.parsed_u64.reserve(n);
        let parsed_ptr = sc.parsed_u64.as_mut_ptr();
        let pack_ts = |ts: Timestamp| -> u64 {
            (ts.month as u64)
                | ((ts.day as u64) << 8)
                | ((ts.hour as u64) << 16)
                | ((ts.minute as u64) << 24)
                | ((ts.second as u64) << 32)
        };
        let mut prev_w0 = unsafe { v0.as_ptr().cast::<u64>().read_unaligned() };
        #[allow(unused_variables)]
        let first_w0_low48 = prev_w0 & 0x0000_ffff_ffff_ffff;
        let mut prev_w6 = unsafe { v0.as_ptr().add(6).cast::<u64>().read_unaligned() };
        let mut prev_w14 = unsafe { v0.as_ptr().add(14).cast::<u64>().read_unaligned() };
        let mut prev_packed = pack_ts(first);
        unsafe {
            *parsed_ptr = prev_packed;
        }
        #[cfg(target_arch = "aarch64")]
        let (mins, maxs) = unsafe {
            use core::arch::aarch64::*;
            let mut min_v = vcreate_u8(prev_packed);
            let mut max_v = min_v;
            for (idx, &val) in values.iter().enumerate().skip(1) {
                if val.len() != 22 {
                    return None;
                }
                let ptr = val.as_ptr();
                let w0 = ptr.cast::<u64>().read_unaligned();
                let w6 = ptr.add(6).cast::<u64>().read_unaligned();
                let w14 = ptr.add(14).cast::<u64>().read_unaligned();
                if w6 == prev_w6 && w0 == prev_w0 {
                    if w14 == prev_w14 {
                        *parsed_ptr.add(idx) = prev_packed;
                        continue;
                    }
                    let d14 = w14 ^ 0x225a_3030_3a30_303a;
                    if ((d14.wrapping_add(0x0000_0606_0006_0600)) | d14) & 0xffff_f0f0_fff0_f0ff
                        != 0
                    {
                        return None;
                    }
                    let p14 = d14.wrapping_mul(10).wrapping_add(d14 >> 8);
                    let minute = (p14 >> 8) & 0xff;
                    let second = (p14 >> 32) & 0xff;
                    if minute >= 60 || second >= 60 {
                        return None;
                    }
                    prev_w14 = w14;
                    prev_packed = (prev_packed & 0x00ff_ffff) | (minute << 24) | (second << 32);
                    let pv = vcreate_u8(prev_packed);
                    min_v = vmin_u8(min_v, pv);
                    max_v = vmax_u8(max_v, pv);
                    *parsed_ptr.add(idx) = prev_packed;
                    continue;
                }
                if (w0 & 0x0000_ffff_ffff_ffff) != first_w0_low48 {
                    return None;
                }
                let d6 = w6 ^ 0x3030_5430_302d_3030;
                let d14 = w14 ^ 0x225a_3030_3a30_303a;
                if (((d6.wrapping_add(0x0606_0006_0600_0606)) | d6) & 0xf0f0_fff0_f0ff_f0f0)
                    | (((d14.wrapping_add(0x0000_0606_0006_0600)) | d14) & 0xffff_f0f0_fff0_f0ff)
                    != 0
                {
                    return None;
                }
                let p6 = d6.wrapping_mul(10).wrapping_add(d6 >> 8);
                let p14 = d14.wrapping_mul(10).wrapping_add(d14 >> 8);
                let month = p6 & 0xff;
                let day = (p6 >> 24) & 0xff;
                let hour = (p6 >> 48) & 0xff;
                let minute = (p14 >> 8) & 0xff;
                let second = (p14 >> 32) & 0xff;
                if month.wrapping_sub(1) >= 12
                    || day.wrapping_sub(1) >= 31
                    || hour >= 24
                    || minute >= 60
                    || second >= 60
                {
                    return None;
                }
                prev_w0 = w0;
                prev_w6 = w6;
                prev_w14 = w14;
                prev_packed =
                    month | (day << 8) | (hour << 16) | (minute << 24) | (second << 32);
                let pv = vcreate_u8(prev_packed);
                min_v = vmin_u8(min_v, pv);
                max_v = vmax_u8(max_v, pv);
                *parsed_ptr.add(idx) = prev_packed;
            }
            let min_b = vget_lane_u64(vreinterpret_u64_u8(min_v), 0).to_le_bytes();
            let max_b = vget_lane_u64(vreinterpret_u64_u8(max_v), 0).to_le_bytes();
            (
                [min_b[0], min_b[1], min_b[2], min_b[3], min_b[4]],
                [max_b[0], max_b[1], max_b[2], max_b[3], max_b[4]],
            )
        };
        #[cfg(target_arch = "x86_64")]
        let (mins, maxs) = unsafe {
            use core::arch::x86_64::*;
            let mut min_v = _mm_cvtsi64_si128(prev_packed as i64);
            let mut max_v = min_v;
            for (idx, &val) in values.iter().enumerate().skip(1) {
                if val.len() != 22 {
                    return None;
                }
                let ptr = val.as_ptr();
                let w0 = ptr.cast::<u64>().read_unaligned();
                let w6 = ptr.add(6).cast::<u64>().read_unaligned();
                let w14 = ptr.add(14).cast::<u64>().read_unaligned();
                if w6 == prev_w6 && w0 == prev_w0 {
                    if w14 == prev_w14 {
                        *parsed_ptr.add(idx) = prev_packed;
                        continue;
                    }
                    let d14 = w14 ^ 0x225a_3030_3a30_303a;
                    if ((d14.wrapping_add(0x0000_0606_0006_0600)) | d14) & 0xffff_f0f0_fff0_f0ff
                        != 0
                    {
                        return None;
                    }
                    let p14 = d14.wrapping_mul(10).wrapping_add(d14 >> 8);
                    let minute = (p14 >> 8) & 0xff;
                    let second = (p14 >> 32) & 0xff;
                    if minute >= 60 || second >= 60 {
                        return None;
                    }
                    prev_w14 = w14;
                    prev_packed = (prev_packed & 0x00ff_ffff) | (minute << 24) | (second << 32);
                    let pv = _mm_cvtsi64_si128(prev_packed as i64);
                    min_v = _mm_min_epu8(min_v, pv);
                    max_v = _mm_max_epu8(max_v, pv);
                    *parsed_ptr.add(idx) = prev_packed;
                    continue;
                }
                if (w0 & 0x0000_ffff_ffff_ffff) != first_w0_low48 {
                    return None;
                }
                let d6 = w6 ^ 0x3030_5430_302d_3030;
                let d14 = w14 ^ 0x225a_3030_3a30_303a;
                if (((d6.wrapping_add(0x0606_0006_0600_0606)) | d6) & 0xf0f0_fff0_f0ff_f0f0)
                    | (((d14.wrapping_add(0x0000_0606_0006_0600)) | d14) & 0xffff_f0f0_fff0_f0ff)
                    != 0
                {
                    return None;
                }
                let p6 = d6.wrapping_mul(10).wrapping_add(d6 >> 8);
                let p14 = d14.wrapping_mul(10).wrapping_add(d14 >> 8);
                let month = p6 & 0xff;
                let day = (p6 >> 24) & 0xff;
                let hour = (p6 >> 48) & 0xff;
                let minute = (p14 >> 8) & 0xff;
                let second = (p14 >> 32) & 0xff;
                if month.wrapping_sub(1) >= 12
                    || day.wrapping_sub(1) >= 31
                    || hour >= 24
                    || minute >= 60
                    || second >= 60
                {
                    return None;
                }
                prev_w0 = w0;
                prev_w6 = w6;
                prev_w14 = w14;
                prev_packed =
                    month | (day << 8) | (hour << 16) | (minute << 24) | (second << 32);
                let pv = _mm_cvtsi64_si128(prev_packed as i64);
                min_v = _mm_min_epu8(min_v, pv);
                max_v = _mm_max_epu8(max_v, pv);
                *parsed_ptr.add(idx) = prev_packed;
            }
            let min_b = (_mm_cvtsi128_si64(min_v) as u64).to_le_bytes();
            let max_b = (_mm_cvtsi128_si64(max_v) as u64).to_le_bytes();
            (
                [min_b[0], min_b[1], min_b[2], min_b[3], min_b[4]],
                [max_b[0], max_b[1], max_b[2], max_b[3], max_b[4]],
            )
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let (mins, maxs) = {
            let mut mins = [first.month, first.day, first.hour, first.minute, first.second];
            let mut maxs = mins;
            for (idx, &val) in values.iter().enumerate().skip(1) {
                if val.len() != 22 {
                    return None;
                }
                let ptr = val.as_ptr();
                let w0 = unsafe { ptr.cast::<u64>().read_unaligned() };
                let w6 = unsafe { ptr.add(6).cast::<u64>().read_unaligned() };
                let w14 = unsafe { ptr.add(14).cast::<u64>().read_unaligned() };
                if w0 == prev_w0 && w6 == prev_w6 && w14 == prev_w14 {
                    unsafe {
                        *parsed_ptr.add(idx) = prev_packed;
                    }
                    continue;
                }
                let ts = parse_timestamp(val)?;
                if ts.year != year {
                    return None;
                }
                let c = [ts.month, ts.day, ts.hour, ts.minute, ts.second];
                for i in 0..5 {
                    if c[i] < mins[i] {
                        mins[i] = c[i];
                    }
                    if c[i] > maxs[i] {
                        maxs[i] = c[i];
                    }
                }
                prev_w0 = w0;
                prev_w6 = w6;
                prev_w14 = w14;
                prev_packed = pack_ts(ts);
                unsafe {
                    *parsed_ptr.add(idx) = prev_packed;
                }
            }
            (mins, maxs)
        };
        unsafe {
            sc.parsed_u64.set_len(n);
        }
        let mut plans = [(0u8, 0u8); 5];
        let mut bit_count = 0usize;
        for i in 0..5 {
            let w = bit_width((maxs[i] - mins[i]) as u64);
            plans[i] = (mins[i], w);
            bit_count += bit_bytes(n, w).ok()?;
        }
        let size = 1 + 2 + 10 + bit_count;
        if size >= limit {
            return None;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_TIMESTAMP);
        out.extend_from_slice(&year.to_le_bytes());
        for (base, width) in &plans {
            out.extend_from_slice(&[*base, *width]);
        }
        for (comp_idx, &(base, width)) in plans.iter().enumerate() {
            if width != 0 {
                let shift = (comp_idx * 8) as u32;
                let base_u64 = base as u64;
                push_bit_values(
                    &mut out,
                    n,
                    width,
                    sc.parsed_u64
                        .iter()
                        .map(|&w| ((w >> shift) & 0xff) - base_u64),
                );
            }
        }
        Some(out)
    })
}

/// Canonical tie-breaking priority rank from the specification:
/// earlier model in the race wins ties (`candidate.len() < best.len()`).
#[inline(always)]
const fn canonical_model_rank(kind: ModelKind) -> u8 {
    match kind {
        ModelKind::Raw => 0,
        ModelKind::Dictionary => 1,
        ModelKind::DecimalAffine => 2,
        ModelKind::DecimalPacked => 3,
        ModelKind::HexSuffix => 4,
        ModelKind::DecimalSuffix => 5,
        ModelKind::PeriodicString => 6,
        ModelKind::PeriodicHex => 7,
        ModelKind::Gf2Affine => 8,
        ModelKind::Timestamp => 9,
        ModelKind::EpochCounter => 10,
        ModelKind::SegmentDict => 11,
        ModelKind::DecimalDelta => 12,
    }
}

#[inline(always)]
fn effective_limit(best_len: usize, best_kind: ModelKind, candidate_kind: ModelKind) -> usize {
    // In canonical order, `candidate_kind` beats `best_kind` iff:
    // - `len < best_len` (when `rank(candidate) > rank(best)`), OR
    // - `len <= best_len`, i.e., `len < best_len + 1` (when `rank(candidate) < rank(best)`).
    best_len + usize::from(canonical_model_rank(candidate_kind) < canonical_model_rank(best_kind))
}

fn pack_model(values: &[&[u8]]) -> (Vec<u8>, ModelKind) {
    pack_model_mode(values, false)
}

fn pack_model_v3(values: &[&[u8]]) -> (Vec<u8>, ModelKind) {
    pack_model_mode(values, true)
}

