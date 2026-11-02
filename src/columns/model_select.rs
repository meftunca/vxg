fn pack_model_mode(values: &[&[u8]], allow_v3_ext: bool) -> (Vec<u8>, ModelKind) {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2,ssse3,sse4.1,popcnt,bmi1,bmi2")]
        unsafe fn pack_model_mode_x86_v3(
            values: &[&[u8]],
            allow_v3_ext: bool,
        ) -> (Vec<u8>, ModelKind) {
            pack_model_mode_impl(values, allow_v3_ext)
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
            && std::arch::is_x86_feature_detected!("popcnt")
            && std::arch::is_x86_feature_detected!("bmi1")
            && std::arch::is_x86_feature_detected!("bmi2")
        {
            return unsafe { pack_model_mode_x86_v3(values, allow_v3_ext) };
        }
    }
    pack_model_mode_impl(values, allow_v3_ext)
}

#[inline(always)]
fn probe_early16(values: &[&[u8]]) -> (usize, bool) {
    let n = values.len();
    let m = n.min(16);
    let mut fp = [0u64; 16];
    for (i, &v) in values[..m].iter().enumerate() {
        let len = v.len();
        fp[i] = if len >= 8 {
            unsafe {
                let p = v.as_ptr();
                p.cast::<u64>().read_unaligned()
                    ^ p.add(len - 8).cast::<u64>().read_unaligned().rotate_left(29)
                    ^ (len as u64)
            }
        } else if len >= 4 {
            unsafe {
                let p = v.as_ptr();
                (p.cast::<u32>().read_unaligned() as u64)
                    | ((p.add(len - 4).cast::<u32>().read_unaligned() as u64) << 32)
                    ^ (len as u64)
            }
        } else if len > 0 {
            (v[0] as u64) | ((v[len >> 1] as u64) << 8) | ((v[len - 1] as u64) << 16) | ((len as u64) << 24)
        } else {
            0
        };
    }
    let mut dups = 0usize;
    let mut first_diff: Option<&[u8]> = None;
    for i in 1..m {
        let f = fp[i];
        let mut matched = false;
        for j in 0..i {
            if fp[j] == f && slice_eq_fast(values[i], values[j]) {
                dups += 1;
                if j == 0 {
                    matched = true;
                }
                break;
            }
        }
        if first_diff.is_none() && !matched && (f != fp[0] || !slice_eq_fast(values[i], values[0])) {
            first_diff = Some(values[i]);
        }
    }
    if first_diff.is_none() && n > m && !slice_eq_fast(values[n - 1], values[0]) {
        first_diff = Some(values[n - 1]);
    }
    let same_pat_lits = first_diff.is_some_and(|dv| pattern_r1_same_literals(values[0], dv));
    (dups, same_pat_lits)
}

#[inline(always)]
fn pack_model_mode_impl(values: &[&[u8]], allow_v3_ext: bool) -> (Vec<u8>, ModelKind) {
    let n = values.len();
    if n < 2 {
        if allow_v3_ext {
            return (pack_raw_or_lz_model(values), ModelKind::Raw);
        }
        return (pack_raw_model(values), ModelKind::Raw);
    }

    let first_slice = values[0];
    let min_raw_size = 1 + n + first_slice.len();
    let mut raw_size_opt: Option<usize> = if n < 256 {
        Some(raw_model_size(values))
    } else {
        None
    };
    let mut best_bytes: Option<Vec<u8>> = None;
    let mut best_len = raw_size_opt.unwrap_or(usize::MAX);
    let mut best_kind = ModelKind::Raw;

    macro_rules! get_raw_size {
        () => {
            *raw_size_opt.get_or_insert_with(|| raw_model_size(values))
        };
    }

    macro_rules! try_model {
        (|$limit:ident| $call:expr, $kind:expr) => {{
            let $limit = if best_bytes.is_some() {
                effective_limit(best_len, best_kind, $kind)
            } else if let Some(rs) = raw_size_opt {
                rs
            } else {
                usize::MAX
            };
            if let Some(bytes) = $call {
                let len = bytes.len();
                if best_bytes.is_some() || len < min_raw_size || len < get_raw_size!() {
                    best_len = len;
                    best_kind = $kind;
                    if let Some(old) = best_bytes.replace(bytes) {
                        put_model_buf(old);
                    }
                } else {
                    put_model_buf(bytes);
                }
            }
        }};
    }

    let is_quoted = first_slice.len() >= 2
        && first_slice[0] == b'"'
        && first_slice[first_slice.len() - 1] == b'"';
    let (dups16, same_pat_lits) = probe_early16(values);

    if !is_quoted {
        // Non-quoted values cannot match any of the 8 quoted-only models
        // (`HexSuffix`, `DecimalSuffix`, `PeriodicString`, `PeriodicHex`,
        // `Gf2Affine`, `Timestamp`, `EpochCounter`, `SegmentDict`).
        // Check if it's a canonical decimal column in a single pass!
        let mut skip_dictionary = false;
        let mut is_all_decimal = false;
        if let Some(first_num) = parse_decimal(first_slice) {
            if let Some(second_num) = parse_decimal(values[1]) {
                MODEL_SCRATCH.with(|cell| {
                    let mut sc = cell.borrow_mut();
                    sc.parsed_u64.clear();
                    sc.parsed_u64.reserve(n);
                    let parsed_ptr = sc.parsed_u64.as_mut_ptr();
                    unsafe {
                        *parsed_ptr = first_num;
                        *parsed_ptr.add(1) = second_num;
                    }
                    let mut all_decimal = true;
                    let mut min = first_num.min(second_num);
                    let mut max = first_num.max(second_num);
                    let step128 = second_num as i128 - first_num as i128;
                    let mut step64 = 0i64;
                    let mut affine_ok = true;
                    let mut delta_ok = true;
                    let mut max_zigzag = 0u64;
                    if let Ok(s64) = i64::try_from(step128) {
                        step64 = s64;
                        max_zigzag = ((s64 << 1) ^ (s64 >> 63)) as u64;
                    } else {
                        affine_ok = false;
                        delta_ok = false;
                    }
                    let mut prev = second_num;
                    let mut hi_bits = first_num | second_num;
                    let mut diff_xor = 0i64;

                    for (idx, &val) in values.iter().enumerate().skip(2) {
                        if let Some(num) = parse_decimal(val) {
                            unsafe {
                                *parsed_ptr.add(idx) = num;
                            }
                            min = min.min(num);
                            max = max.max(num);
                            hi_bits |= num;
                            let delta = (num as i64).wrapping_sub(prev as i64);
                            if delta != step64 {
                                diff_xor = 1;
                                let zz = ((delta << 1) ^ (delta >> 63)) as u64;
                                max_zigzag = max_zigzag.max(zz);
                            }
                            prev = num;
                        } else {
                            all_decimal = false;
                            break;
                        }
                    }

                    if all_decimal {
                        is_all_decimal = true;
                        unsafe {
                            sc.parsed_u64.set_len(n);
                        }
                        if (hi_bits >> 63) == 0 {
                            if diff_xor != 0 {
                                affine_ok = false;
                            }
                        } else {
                            max_zigzag = if delta_ok {
                                ((step64 << 1) ^ (step64 >> 63)) as u64
                            } else {
                                0
                            };
                            let mut p = second_num;
                            for &num in &sc.parsed_u64[2..n] {
                                let diff128 = num as i128 - p as i128;
                                if diff128 != step128 {
                                    affine_ok = false;
                                }
                                if delta_ok {
                                    if let Ok(delta) = i64::try_from(diff128) {
                                        let zz = ((delta << 1) ^ (delta >> 63)) as u64;
                                        max_zigzag = max_zigzag.max(zz);
                                    } else {
                                        delta_ok = false;
                                    }
                                }
                                p = num;
                            }
                        }
                        if affine_ok {
                            if step128 != 0 {
                                // Arithmetic progression with non-zero step has `n` distinct values (`u == n`),
                                // so `Dictionary` is guaranteed to return `None`.
                                skip_dictionary = true;
                            }
                            let lim = if best_bytes.is_some() {
                                effective_limit(best_len, best_kind, ModelKind::DecimalAffine)
                            } else if 17 < min_raw_size {
                                usize::MAX
                            } else {
                                get_raw_size!()
                            };
                            if 17 < lim {
                                let step = step64;
                                let mut out = take_model_buf(17);
                                out.push(MODEL_DECIMAL_AFFINE);
                                out.extend_from_slice(&first_num.to_le_bytes());
                                out.extend_from_slice(&step.to_le_bytes());
                                best_len = 17;
                                best_kind = ModelKind::DecimalAffine;
                                if let Some(old) = best_bytes.replace(out) {
                                    put_model_buf(old);
                                }
                            }
                        }
                        // When `affine_ok && step128 != 0 && n >= 49`, `DecimalAffine` (17 bytes, rank 2)
                        // cannot be beaten by `DecimalPacked` (`>= 17` bytes, rank 3) or `DecimalDelta` (`>= 17` bytes, rank 12).
                        if !(affine_ok && step128 != 0 && n >= 49 && best_kind == ModelKind::DecimalAffine) {
                            let packed_width = bit_width(max - min);
                            if let Ok(packed_bits) = bit_bytes(n, packed_width) {
                                let packed_size = 10 + packed_bits;
                                let delta_width = bit_width(max_zigzag);
                                let delta_size = if delta_ok {
                                    bit_bytes(n - 1, delta_width).ok().map(|b| 10 + b)
                                } else {
                                    None
                                };
                                if delta_size.is_some_and(|ds| ds < packed_size) {
                                    let ds = delta_size.unwrap();
                                    let lim = if best_bytes.is_some() {
                                        effective_limit(best_len, best_kind, ModelKind::DecimalDelta)
                                    } else if ds < min_raw_size {
                                        usize::MAX
                                    } else {
                                        get_raw_size!()
                                    };
                                    if ds < lim {
                                        let mut out = take_model_buf(ds);
                                        out.push(MODEL_DECIMAL_DELTA);
                                        out.extend_from_slice(&first_num.to_le_bytes());
                                        out.push(delta_width);
                                        push_bit_values(
                                            &mut out,
                                            n - 1,
                                            delta_width,
                                            sc.parsed_u64.windows(2).map(|pair| {
                                                let delta = (pair[1] as i128 - pair[0] as i128) as i64;
                                                ((delta << 1) ^ (delta >> 63)) as u64
                                            }),
                                        );
                                        best_len = ds;
                                        best_kind = ModelKind::DecimalDelta;
                                        if let Some(old) = best_bytes.replace(out) {
                                            put_model_buf(old);
                                        }
                                    }
                                } else {
                                    let lim = if best_bytes.is_some() {
                                        effective_limit(best_len, best_kind, ModelKind::DecimalPacked)
                                    } else if packed_size < min_raw_size {
                                        usize::MAX
                                    } else {
                                        get_raw_size!()
                                    };
                                    if packed_size < lim {
                                        let mut out = take_model_buf(packed_size);
                                        out.push(MODEL_DECIMAL_PACKED);
                                        out.extend_from_slice(&min.to_le_bytes());
                                        out.push(packed_width);
                                        push_bit_values(
                                            &mut out,
                                            n,
                                            packed_width,
                                            sc.parsed_u64.iter().map(|&v| v - min),
                                        );
                                        best_len = packed_size;
                                        best_kind = ModelKind::DecimalPacked;
                                        if let Some(old) = best_bytes.replace(out) {
                                            put_model_buf(old);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
        if is_all_decimal && best_bytes.is_some() && allow_v3_ext && dups16 < 8 && n > 64 {
            skip_dictionary = true;
        }
        let unquoted_const_hint = !skip_dictionary
            && (n <= 64
                || (dups16 >= 2 && !same_pat_lits)
                || (slice_eq_fast(first_slice, values[1]) && slice_eq_fast(first_slice, values[n - 1])));
        if unquoted_const_hint {
            try_model!(|limit| pack_dictionary_model_mode(values, limit, allow_v3_ext), ModelKind::Dictionary);
        }
        // Unquoted digit/literal patterns (CLF / CSV / TSV timestamps, IPv4, versions, ...) and
        // unquoted fixed-width lowercase hex ids (`8201e2bd73ab4876`). Plain decimals are left to
        // the decimal models.
        let unquoted_hex_tail = allow_v3_ext
            && !is_all_decimal
            && first_slice.len() >= 8
            && hex_nibble(first_slice[first_slice.len() - 1]).is_some()
            && hex_nibble(first_slice[first_slice.len() - 2]).is_some()
            && hex_nibble(first_slice[first_slice.len() - 3]).is_some()
            && hex_nibble(first_slice[first_slice.len() - 4]).is_some();
        let unquoted_hex_early_dup = unquoted_hex_tail && dups16 > 0;
        if unquoted_hex_tail && !unquoted_hex_early_dup && !unquoted_const_hint {
            try_model!(|limit| pack_hex_suffix_model_mode(values, limit, false), ModelKind::HexSuffix);
            if best_kind == ModelKind::HexSuffix {
                skip_dictionary = true;
            }
        }
        if allow_v3_ext
            && !is_all_decimal
            && best_kind != ModelKind::HexSuffix
            && !(best_kind == ModelKind::Dictionary && dups16 >= 3 && !same_pat_lits)
        {
            try_model!(|limit| pack_pattern_model(values, limit), ModelKind::Pattern);
            if best_kind == ModelKind::Pattern && dups16 == 0 && n >= 16 {
                skip_dictionary = true;
            }
        }
        if !skip_dictionary && !unquoted_const_hint {
            if best_bytes.is_none() && dups16 == 0 {
                let _ = get_raw_size!();
            }
            try_model!(|limit| pack_dictionary_model_mode(values, limit, allow_v3_ext), ModelKind::Dictionary);
        }
        if unquoted_hex_tail && (unquoted_hex_early_dup || unquoted_const_hint) {
            try_model!(|limit| pack_hex_suffix_model_mode(values, limit, false), ModelKind::HexSuffix);
        }
        if allow_v3_ext && best_kind == ModelKind::Raw {
            return (pack_raw_or_lz_model(values), ModelKind::Raw);
        }
        let _ = best_len;
        return (best_bytes.unwrap_or_else(|| pack_raw_model(values)), best_kind);
    }

    // Quoted value path (`first_slice` starts and ends with `"`):
    let inner0 = &first_slice[1..first_slice.len() - 1];
    let last_inner_byte = inner0.last().copied().unwrap_or(0);

    // 1. Timestamp (only possible when `first_slice.len() == 22` and ends with `Z"`):
    if first_slice.len() == 22 && last_inner_byte == b'Z' {
        try_model!(|limit| pack_timestamp_model(values, limit), ModelKind::Timestamp);
        if best_kind == ModelKind::Timestamp && (allow_v3_ext || best_len <= 25) {
            return (best_bytes.unwrap(), best_kind);
        }
    }

    // 1b. For small v3 columns (`n <= 64`) or when the first two rows are identical (`values[0] == values[1]`,
    // e.g. constant strings like `"eu-central-1"`), run `Dictionary` first so `best_len` is tight before
    // testing `DecimalSuffix` / `HexSuffix` / `SegmentDict`.
    let mut dict_already_tried = false;
    if (allow_v3_ext && (n <= 64 || (dups16 >= 2 && !same_pat_lits)))
        || (slice_eq_fast(first_slice, values[1]) && slice_eq_fast(first_slice, values[n - 1]))
    {
        dict_already_tried = true;
        try_model!(|limit| pack_dictionary_model_mode(values, limit, allow_v3_ext), ModelKind::Dictionary);
        if best_kind == ModelKind::Dictionary
            && best_len <= 3 + uleb_len(first_slice.len()) + first_slice.len()
        {
            return (best_bytes.unwrap(), best_kind);
        }
    }

    // 2. Trailing decimal digit -> `DecimalSuffix`:
    if last_inner_byte.is_ascii_digit()
        && values.get(1).is_some_and(|v1| v1.len() >= 2 && v1[v1.len() - 2].is_ascii_digit())
    {
        try_model!(|limit| pack_decimal_suffix_model(values, limit), ModelKind::DecimalSuffix);
    }

    // 3. Multi-byte lowercase hex suffix -> `EpochCounter`, `Gf2Affine`, `HexSuffix`:
    let has_hex_tail = hex_nibble(last_inner_byte).is_some();
    let has_multi_hex_tail = has_hex_tail
        && inner0.len() >= 8
        && hex_nibble(inner0[inner0.len() - 2]).is_some()
        && hex_nibble(inner0[inner0.len() - 3]).is_some()
        && hex_nibble(inner0[inner0.len() - 4]).is_some();

    // 2b. Digit-field / literal pattern (timestamps in any layout, IPv4, versions, `u012336`-style ids).
    if allow_v3_ext
        && !(best_kind == ModelKind::Dictionary && dups16 >= 3 && !same_pat_lits)
        && !(has_multi_hex_tail && inner0.len() >= 16 && dups16 == 0 && !same_pat_lits)
    {
        try_model!(|limit| pack_pattern_model(values, limit), ModelKind::Pattern);
    }

    if has_multi_hex_tail {
        let has_early_dup = inner0.len() < 16 || dups16 > 0;
        if has_early_dup && allow_v3_ext && !dict_already_tried {
            dict_already_tried = true;
            try_model!(|limit| pack_dictionary_model_mode(values, limit, true), ModelKind::Dictionary);
        }
        try_model!(|limit| pack_hex_suffix_model(values, limit), ModelKind::HexSuffix);
        if inner0.len() >= 16 {
            try_model!(|limit| pack_epoch_counter_model(values, limit), ModelKind::EpochCounter);
            if best_kind == ModelKind::HexSuffix {
                if let Some((prefix, first_digits)) = split_hex_suffix(first_slice) {
                    let hdr_len = 1 + uleb_len(prefix.len()) + prefix.len() + 1;
                    let raw64_opt = first_digits
                        .len()
                        .eq(&16)
                        .then(|| best_bytes.as_deref())
                        .flatten()
                        .filter(|b| b.len() == hdr_len + n * 8);
                    if let Some(hb) = raw64_opt {
                        let lim = effective_limit(best_len, best_kind, ModelKind::Gf2Affine);
                        if let Some(gf2_out) = pack_gf2_affine_from_raw64(prefix, &hb[hdr_len..], n, lim) {
                            best_len = gf2_out.len();
                            best_kind = ModelKind::Gf2Affine;
                            if let Some(old) = best_bytes.replace(gf2_out) {
                                put_model_buf(old);
                            }
                        }
                    } else {
                        try_model!(|limit| pack_gf2_affine_model(values, limit), ModelKind::Gf2Affine);
                    }
                }
            } else {
                try_model!(|limit| pack_gf2_affine_model(values, limit), ModelKind::Gf2Affine);
            }
        }
        if matches!(best_kind, ModelKind::HexSuffix | ModelKind::EpochCounter | ModelKind::Gf2Affine)
            && dups16 == 0
            && n >= 16
        {
            dict_already_tried = true;
        }
        if !has_early_dup && allow_v3_ext && !dict_already_tried {
            dict_already_tried = true;
            try_model!(|limit| pack_dictionary_model_mode(values, limit, true), ModelKind::Dictionary);
        }
    }

    // 4. Every valid SegmentDict value consists exclusively of bytes satisfying `segment_byte_ok(b) || b == b';'`.
    // Also skip SegmentDict when `DecimalSuffix` already matched and `n >= 4 * (first_slice.len() + 8)` with
    // a `"; "` separator before the trailing decimal, where `SegmentDict` is mathematically guaranteed >= `best_len`.
    let dec_suffix_dominates_seg = best_kind == ModelKind::DecimalSuffix
        && n >= 4 * (first_slice.len() + 8)
        && {
            let mut s = inner0.len();
            while s != 0 && inner0[s - 1].is_ascii_digit() {
                s -= 1;
            }
            s == 0 || inner0[..s].ends_with(b"; ")
        };
    let can_try_seg = !dec_suffix_dominates_seg && (!allow_v3_ext || best_len > 6 + (n + 7) / 8);
    let mut seg_chars_opt: Option<bool> = None;
    let looks_like_segment_dict = can_try_seg
        && (last_inner_byte == b' ' || find_byte(b';', inner0).is_some())
        && *seg_chars_opt.insert(inner0.iter().all(|&b| segment_byte_ok(b) || b == b';'));
    if looks_like_segment_dict {
        try_model!(
            |limit| pack_segment_dictionary_model(values, limit),
            ModelKind::SegmentDict
        );
    }

    // 5. Dictionary model (skip if already tried or if SegmentDict/DecimalSuffix/Pattern already succeeded and first 16 rows are distinct):
    let skip_dict_after_seg = (best_kind == ModelKind::SegmentDict
        || (matches!(best_kind, ModelKind::DecimalSuffix | ModelKind::Pattern) && allow_v3_ext))
        && n >= 8
        && dups16 == 0;
    if !dict_already_tried && !skip_dict_after_seg {
        if best_bytes.is_none() && dups16 == 0 {
            let _ = get_raw_size!();
        }
        try_model!(
            |limit| pack_dictionary_model_mode(values, limit, allow_v3_ext),
            ModelKind::Dictionary
        );
    }

    // 5b. Short hex tail fallback (now aborts in O(1) via `size >= limit` when `Dictionary` won):
    if !allow_v3_ext && has_hex_tail && !has_multi_hex_tail {
        if inner0.len() >= 16 {
            try_model!(
                |limit| pack_epoch_counter_model(values, limit),
                ModelKind::EpochCounter
            );
            try_model!(
                |limit| pack_gf2_affine_model(values, limit),
                ModelKind::Gf2Affine
            );
        }
        try_model!(
            |limit| pack_hex_suffix_model(values, limit),
            ModelKind::HexSuffix
        );
    }

    // 6. SegmentDict fallback when `inner0` had no `;` or trailing space:
    if !looks_like_segment_dict
        && can_try_seg
        && (!allow_v3_ext
            || best_kind == ModelKind::Raw
            || values[1..n.min(8)]
                .iter()
                .any(|v| v.len() >= 3 && (v[v.len() - 2] == b' ' || find_byte(b';', v).is_some())))
        && seg_chars_opt.unwrap_or_else(|| inner0.iter().all(|&b| segment_byte_ok(b) || b == b';'))
    {
        try_model!(
            |limit| pack_segment_dictionary_model(values, limit),
            ModelKind::SegmentDict
        );
    }

    // 7. PeriodicString & PeriodicHex:
    if !allow_v3_ext || best_kind == ModelKind::Raw {
        try_model!(
            |limit| pack_periodic_string_model(values, limit),
            ModelKind::PeriodicString
        );
        try_model!(
            |limit| pack_periodic_hex_model(values, limit),
            ModelKind::PeriodicHex
        );
    }

    if allow_v3_ext && best_kind == ModelKind::Raw {
        return (pack_raw_or_lz_model(values), ModelKind::Raw);
    }

    let _ = best_len;
    (
        best_bytes.unwrap_or_else(|| pack_raw_model(values)),
        best_kind,
    )
}

/// Encode owned values (test/oracle helper): borrows each value before entering the model race.
fn pack_model_values(values: &[Vec<u8>]) -> (Vec<u8>, ModelKind) {
    let refs = values.iter().map(|value| value.as_slice()).collect::<Vec<_>>();
    pack_model(&refs)
}

