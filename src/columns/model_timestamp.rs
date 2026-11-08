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
fn parse_timestamp_core20(core: &[u8]) -> Option<Timestamp> {
    if core.len() != 20
        || core[4] != b'-'
        || core[7] != b'-'
        || core[10] != b'T'
        || core[13] != b':'
        || core[16] != b':'
        || core[19] != b'Z'
    {
        return None;
    }
    let y0 = core[0].wrapping_sub(b'0');
    let y1 = core[1].wrapping_sub(b'0');
    let y2 = core[2].wrapping_sub(b'0');
    let y3 = core[3].wrapping_sub(b'0');
    if y0 > 9 || y1 > 9 || y2 > 9 || y3 > 9 {
        return None;
    }
    let year =
        u16::from(y0) * 1000 + u16::from(y1) * 100 + u16::from(y2) * 10 + u16::from(y3);
    let month = digits(&core[5..7])? as u8;
    let day = digits(&core[8..10])? as u8;
    let hour = digits(&core[11..13])? as u8;
    let minute = digits(&core[14..16])? as u8;
    let second = digits(&core[17..19])? as u8;
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
fn parse_timestamp(value: &[u8]) -> Option<Timestamp> {
    if value.len() != 22 || value[0] != b'"' || value[21] != b'"' {
        return None;
    }
    parse_timestamp_core20(&value[1..21])
}

#[inline(always)]
fn pack_timestamp_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    pack_timestamp_model_ext(values, limit, false)
}

#[inline(always)]
fn pack_timestamp_model_ext(values: &[&[u8]], limit: usize, unquoted: bool) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 13 >= limit {
        return None;
    }
    let req_len = if unquoted { 20 } else { 22 };
    let v0 = values[0];
    if v0.len() != req_len {
        return None;
    }
    let first = if unquoted {
        parse_timestamp_core20(v0)?
    } else {
        parse_timestamp(v0)?
    };
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
        #[inline(always)]
        unsafe fn load_ts_words(ptr: *const u8, unquoted: bool) -> (u64, u64, u64) {
            let w0 = ptr.cast::<u64>().read_unaligned();
            if unquoted {
                let w6 = ptr.add(5).cast::<u64>().read_unaligned();
                let w14 =
                    (ptr.add(12).cast::<u64>().read_unaligned() >> 8) | 0x2200_0000_0000_0000;
                (w0, w6, w14)
            } else {
                let w6 = ptr.add(6).cast::<u64>().read_unaligned();
                let w14 = ptr.add(14).cast::<u64>().read_unaligned();
                (w0, w6, w14)
            }
        }
        let (mut prev_w0, mut prev_w6, mut prev_w14) =
            unsafe { load_ts_words(v0.as_ptr(), unquoted) };
        #[allow(unused_variables)]
        let w0_prefix_mask: u64 = if unquoted {
            0x0000_00ff_ffff_ffff
        } else {
            0x0000_ffff_ffff_ffff
        };
        #[allow(unused_variables)]
        let first_w0_low48 = prev_w0 & w0_prefix_mask;
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
                if val.len() != req_len {
                    return None;
                }
                let (w0, w6, w14) = load_ts_words(val.as_ptr(), unquoted);
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
                if (w0 & w0_prefix_mask) != first_w0_low48 {
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
                if val.len() != req_len {
                    return None;
                }
                let (w0, w6, w14) = load_ts_words(val.as_ptr(), unquoted);
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
                if (w0 & w0_prefix_mask) != first_w0_low48 {
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
                if val.len() != req_len {
                    return None;
                }
                let (w0, w6, w14) = unsafe { load_ts_words(val.as_ptr(), unquoted) };
                if w0 == prev_w0 && w6 == prev_w6 && w14 == prev_w14 {
                    unsafe {
                        *parsed_ptr.add(idx) = prev_packed;
                    }
                    continue;
                }
                let ts = if unquoted {
                    parse_timestamp_core20(val)?
                } else {
                    parse_timestamp(val)?
                };
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
        let wire_year = if unquoted { year | 0x8000 } else { year };
        let mut out = take_model_buf(size);
        out.push(MODEL_TIMESTAMP);
        out.extend_from_slice(&wire_year.to_le_bytes());
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
        ModelKind::Pattern => 13,
        ModelKind::ScaledDecimal => 14,
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


/// Detect whether `line0` is an identifier header row (`VendorID,pickup_ts,...`) followed by
/// a typed data row `line1` (`2,2024-01-01T00:57:55Z,...`) on `,`, `\t`, or `|` delimited text.
fn looks_like_delimited_header(data: &[u8], first_nl: usize) -> bool {
    if first_nl == 0 || first_nl >= data.len() {
        return false;
    }
    fn trim_nl(s: &[u8]) -> &[u8] {
        s.strip_suffix(b"\r\n")
            .or_else(|| s.strip_suffix(b"\n"))
            .unwrap_or(s)
    }
    let line0 = trim_nl(&data[..first_nl]);
    if line0.is_empty() || matches!(line0[0], b'{' | b'[' | b'#') {
        return false;
    }
    let rem = &data[first_nl..];
    let second_len = find_byte(b'\n', rem).map_or(rem.len(), |i| i + 1);
    let line1 = trim_nl(&rem[..second_len]);
    if line1.is_empty() {
        return false;
    }
    for delim in [b',', b'\t', b'|'] {
        if !line0.contains(&delim) {
            continue;
        }
        let mut it0 = line0.split(|&b| b == delim);
        let mut it1 = line1.split(|&b| b == delim);
        let mut cols = 0usize;
        let mut all_hdr_ident = true;
        let mut any_data_num = false;
        loop {
            match (it0.next(), it1.next()) {
                (Some(f0), Some(f1)) => {
                    cols += 1;
                    let f0_t = f0.strip_prefix(b"\"").and_then(|s| s.strip_suffix(b"\"")).unwrap_or(f0);
                    if f0_t.is_empty()
                        || !(f0_t[0].is_ascii_alphabetic() || f0_t[0] == b'_')
                        || !f0_t.iter().all(|&b| {
                            b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b' ' | b'.' | b'/')
                        })
                    {
                        all_hdr_ident = false;
                        break;
                    }
                    let f1_t = f1.strip_prefix(b"\"").and_then(|s| s.strip_suffix(b"\"")).unwrap_or(f1);
                    if f1_t.first().is_some_and(|&b| b.is_ascii_digit() || b == b'-') {
                        any_data_num = true;
                    }
                }
                (None, None) => break,
                _ => {
                    all_hdr_ident = false;
                    break;
                }
            }
        }
        if cols >= 2 && all_hdr_ident && any_data_num {
            return true;
        }
    }
    false
}
