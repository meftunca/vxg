#[inline(always)]
fn plain_quoted(value: &[u8]) -> Option<&[u8]> {
    if value.len() < 2 || value.first() != Some(&b'"') || value.last() != Some(&b'"') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    if inner.iter().any(|&byte| byte == b'\\' || byte < 0x20) {
        return None;
    }
    Some(inner)
}

#[inline(always)]
fn periodic_at(content: &[u8], period: usize) -> bool {
    let len = content.len();
    len <= period
        || (content[period] == content[0]
            && content[len - 1] == content[len - 1 - period]
            && slice_eq_fast(&content[period..], &content[..len - period]))
}

fn pack_periodic_string_model<'a>(values: &[&'a [u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 2 + n >= limit {
        return None;
    }
    let v0 = values[0];
    let v1 = values[1];
    let v2 = values[n.min(3) - 1];
    if v0.len() < 2
        || v0[0] != b'"'
        || v0[v0.len() - 1] != b'"'
        || v1.len() < 2
        || v1[0] != b'"'
        || v1[v1.len() - 1] != b'"'
        || v2.len() < 2
        || v2[0] != b'"'
        || v2[v2.len() - 1] != b'"'
    {
        return None;
    }
    let inner0 = &v0[1..v0.len() - 1];
    let inner1 = &v1[1..v1.len() - 1];
    if !inner0.is_empty() && !inner1.is_empty() && 2 + 2 * n >= limit {
        return None;
    }
    let mut watch_inner: &'a [u8] = &v2[1..v2.len() - 1];
    let max_period = inner0
        .len()
        .max(inner1.len())
        .max(watch_inner.len())
        .saturating_sub(1)
        .min(64);

    // Since `size(period)` is monotonically non-decreasing for `period in 1..=64`,
    // the first valid `period` is guaranteed to be optimal, and if `size + rem >= limit`
    // at any row, no larger `period` can ever fit under `limit`!
    for period in 1..=max_period {
        if !periodic_at(inner0, period)
            || !periodic_at(inner1, period)
            || !periodic_at(watch_inner, period)
        {
            continue;
        }
        let mut size = 1 + uleb_len(period);
        let mut max_len = 0usize;
        let mut all_periodic = true;
        for (idx, &v) in values.iter().enumerate() {
            if v.len() < 2 || v[0] != b'"' || v[v.len() - 1] != b'"' {
                return None;
            }
            let content = &v[1..v.len() - 1];
            let len = content.len();
            if len > max_len {
                max_len = len;
            }
            size += uleb_len(len) + len.min(period);
            if size + (n - 1 - idx) >= limit {
                return None;
            }
            if !periodic_at(content, period) {
                watch_inner = content;
                all_periodic = false;
                break;
            }
        }
        if !all_periodic {
            continue;
        }
        if max_len == 0 || period > max_len || size >= limit {
            return None;
        }
        for &v in values {
            plain_quoted(v)?;
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_PERIODIC_STRING);
        put_uleb(&mut out, period);
        for &v in values {
            let content = &v[1..v.len() - 1];
            put_uleb(&mut out, content.len());
            out.extend_from_slice(&content[..content.len().min(period)]);
        }
        return Some(out);
    }
    None
}

fn pack_nibbles(out: &mut Vec<u8>, nibbles: &[u8]) -> Option<()> {
    for pair in nibbles.chunks(2) {
        let high = hex_nibble(pair[0])?;
        let low = match pair.get(1) {
            Some(byte) => hex_nibble(*byte)?,
            None => 0,
        };
        out.push(high << 4 | low);
    }
    Some(())
}

fn pack_periodic_hex_model<'a>(values: &[&'a [u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || 2 + n >= limit {
        return None;
    }
    let v0 = values[0];
    let v1 = values[1];
    let v2 = values[n.min(3) - 1];
    if v0.len() < 2
        || v0[0] != b'"'
        || v0[v0.len() - 1] != b'"'
        || v1.len() < 2
        || v1[0] != b'"'
        || v1[v1.len() - 1] != b'"'
        || v2.len() < 2
        || v2[0] != b'"'
        || v2[v2.len() - 1] != b'"'
    {
        return None;
    }
    let inner0 = &v0[1..v0.len() - 1];
    let inner1 = &v1[1..v1.len() - 1];
    if !inner0.is_empty() && !inner1.is_empty() && 2 + 2 * n >= limit {
        return None;
    }
    let mut watch_inner: &'a [u8] = &v2[1..v2.len() - 1];
    if (!inner0.is_empty() && hex_nibble(inner0[0]).is_none())
        || (!inner1.is_empty() && hex_nibble(inner1[0]).is_none())
        || (!watch_inner.is_empty() && hex_nibble(watch_inner[0]).is_none())
    {
        return None;
    }
    let max_period = inner0
        .len()
        .max(inner1.len())
        .max(watch_inner.len())
        .saturating_sub(1)
        .min(64);

    for period in 1..=max_period {
        if !periodic_at(inner0, period)
            || !periodic_at(inner1, period)
            || !periodic_at(watch_inner, period)
        {
            continue;
        }
        let mut size = 1 + uleb_len(period);
        let mut max_len = 0usize;
        let mut all_periodic = true;
        for (idx, &v) in values.iter().enumerate() {
            if v.len() < 2 || v[0] != b'"' || v[v.len() - 1] != b'"' {
                return None;
            }
            let content = &v[1..v.len() - 1];
            let len = content.len();
            if len > max_len {
                max_len = len;
            }
            size += uleb_len(len) + len.min(period).div_ceil(2);
            if size + (n - 1 - idx) >= limit {
                return None;
            }
            if !periodic_at(content, period) {
                watch_inner = content;
                all_periodic = false;
                break;
            }
        }
        if !all_periodic {
            continue;
        }
        if max_len == 0 || period > max_len || size >= limit {
            return None;
        }
        for &v in values {
            let content = plain_quoted(v)?;
            if !content[..content.len().min(period)]
                .iter()
                .all(|&byte| hex_nibble(byte).is_some())
            {
                return None;
            }
        }
        let mut out = Vec::with_capacity(size);
        out.push(MODEL_PERIODIC_HEX);
        put_uleb(&mut out, period);
        for &v in values {
            let content = &v[1..v.len() - 1];
            put_uleb(&mut out, content.len());
            pack_nibbles(&mut out, &content[..content.len().min(period)])?;
        }
        return Some(out);
    }
    None
}

const GF2_TRAIN_TRANSITIONS: usize = 96;
const GF2_COEFFICIENTS: usize = 65;

#[inline(always)]
fn apply_affine(coefficients: &[u64; GF2_COEFFICIENTS], input: u64) -> u64 {
    let mut output = coefficients[64];
    let mut w = input;
    while w != 0 {
        let bit = w.trailing_zeros() as usize;
        output ^= unsafe { *coefficients.get_unchecked(bit) };
        w &= w - 1;
    }
    output
}

#[inline(always)]
fn build_gf2_nibble_lut(coefficients: &[u64; GF2_COEFFICIENTS]) -> [[u64; 16]; 16] {
    let mut lut = [[0u64; 16]; 16];
    for g in 0..16 {
        let base = if g == 0 { coefficients[64] } else { 0 };
        let c0 = coefficients[g * 4];
        let c1 = coefficients[g * 4 + 1];
        let c2 = coefficients[g * 4 + 2];
        let c3 = coefficients[g * 4 + 3];
        let t0 = base;
        let t1 = t0 ^ c0;
        let t2 = t0 ^ c1;
        let t3 = t1 ^ c1;
        let t4 = t0 ^ c2;
        let t5 = t1 ^ c2;
        let t6 = t2 ^ c2;
        let t7 = t3 ^ c2;
        lut[g] = [
            t0,
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t0 ^ c3,
            t1 ^ c3,
            t2 ^ c3,
            t3 ^ c3,
            t4 ^ c3,
            t5 ^ c3,
            t6 ^ c3,
            t7 ^ c3,
        ];
    }
    lut
}

#[inline(always)]
fn apply_affine_lut(lut: &[[u64; 16]; 16], w: u64) -> u64 {
    lut[0][(w & 15) as usize]
        ^ lut[1][((w >> 4) & 15) as usize]
        ^ lut[2][((w >> 8) & 15) as usize]
        ^ lut[3][((w >> 12) & 15) as usize]
        ^ lut[4][((w >> 16) & 15) as usize]
        ^ lut[5][((w >> 20) & 15) as usize]
        ^ lut[6][((w >> 24) & 15) as usize]
        ^ lut[7][((w >> 28) & 15) as usize]
        ^ lut[8][((w >> 32) & 15) as usize]
        ^ lut[9][((w >> 36) & 15) as usize]
        ^ lut[10][((w >> 40) & 15) as usize]
        ^ lut[11][((w >> 44) & 15) as usize]
        ^ lut[12][((w >> 48) & 15) as usize]
        ^ lut[13][((w >> 52) & 15) as usize]
        ^ lut[14][((w >> 56) & 15) as usize]
        ^ lut[15][(w >> 60) as usize]
}

#[inline(always)]
fn solve_gf2_basis(train: &[u64]) -> Option<[u64; GF2_COEFFICIENTS]> {
    let x0 = train[0];
    let y0 = train[1];
    let mut basis_mask = [0u64; 64];
    let mut basis_out = [0u64; 64];
    let mut rank = 0usize;
    for pair in train.windows(2).skip(1).take(GF2_TRAIN_TRANSITIONS - 1) {
        let mut mask = pair[0] ^ x0;
        let mut output = pair[1] ^ y0;
        let mut inserted = false;
        while mask != 0 {
            let pivot = 63 - mask.leading_zeros() as usize;
            let pm = unsafe { *basis_mask.get_unchecked(pivot) };
            if pm != 0 {
                mask ^= pm;
                output ^= unsafe { *basis_out.get_unchecked(pivot) };
            } else {
                unsafe {
                    *basis_mask.get_unchecked_mut(pivot) = mask;
                    *basis_out.get_unchecked_mut(pivot) = output;
                }
                rank += 1;
                inserted = true;
                break;
            }
        }
        if !inserted && output != 0 {
            return None;
        }
    }
    if rank != 64 {
        return None;
    }
    let mut coefficients = [0u64; GF2_COEFFICIENTS];
    for bit in 0..64 {
        let mut val = basis_out[bit];
        let mut rem = basis_mask[bit] ^ (1u64 << bit);
        while rem != 0 {
            let k = rem.trailing_zeros() as usize;
            val ^= unsafe { *coefficients.get_unchecked(k) };
            rem &= rem - 1;
        }
        coefficients[bit] = val;
    }
    let mut val64 = y0;
    let mut rem64 = x0;
    while rem64 != 0 {
        let k = rem64.trailing_zeros() as usize;
        val64 ^= unsafe { *coefficients.get_unchecked(k) };
        rem64 &= rem64 - 1;
    }
    coefficients[64] = val64;
    Some(coefficients)
}

/// Learn one affine linear transform over GF(2) from a fixed prefix, then
/// require it to predict a nonempty suffix exactly. This is a model class over
/// arbitrary 64-bit values; it stores the full 65-word transform and has no
/// source, field-name, seed or fixed-shift dependency.
#[allow(dead_code)]
fn learn_affine_recurrence(values: &[u64]) -> Option<[u64; GF2_COEFFICIENTS]> {
    if values.len() <= GF2_TRAIN_TRANSITIONS + 1 {
        return None;
    }
    let coefficients = solve_gf2_basis(&values[..GF2_TRAIN_TRANSITIONS + 1])?;
    let lut = build_gf2_nibble_lut(&coefficients);
    if values
        .windows(2)
        .skip(GF2_TRAIN_TRANSITIONS)
        .any(|pair| apply_affine_lut(&lut, pair[0]) != pair[1])
    {
        return None;
    }
    Some(coefficients)
}

fn pack_gf2_affine_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    // The transform plus the fixed prefix absolutely cost more than this
    // bound, so a hopeless column never pays for the parse or the solver.
    if values.len() <= GF2_TRAIN_TRANSITIONS + 1 {
        return None;
    }
    if 1 + uleb_len(0) + 8 + 8 * GF2_COEFFICIENTS >= limit {
        return None;
    }
    // A deterministic GF(2) recurrence with a duplicate in `values[..8]` enters a cycle
    // of length <= 7, so its transition differences span at most 13 dimensions (< 64)
    // and `solve_gf2_basis` is guaranteed to return `None`.
    for i in 1..8 {
        for j in 0..i {
            if slice_eq_fast(values[i], values[j]) {
                return None;
            }
        }
    }
    let (prefix, first_digits) = split_hex_suffix(values[0])?;
    if first_digits.len() != 16 {
        return None;
    }
    let size = 1 + uleb_len(prefix.len()) + prefix.len() + 8 + 8 * GF2_COEFFICIENTS;
    if size >= limit {
        return None;
    }
    let expected_val_len = 2 + prefix.len() + 16;
    let prefix_end = 1 + prefix.len();
    let mut train = [0u64; GF2_TRAIN_TRANSITIONS + 1];
    for (i, &value) in values[..GF2_TRAIN_TRANSITIONS + 1].iter().enumerate() {
        if value.len() != expected_val_len
            || value[0] != b'"'
            || value[expected_val_len - 1] != b'"'
            || !slice_eq_fast(&value[1..prefix_end], prefix)
        {
            return None;
        }
        train[i] = parse_hex_u64_le(&value[prefix_end..expected_val_len - 1])?;
    }
    let coefficients = solve_gf2_basis(&train)?;
    let lut = build_gf2_nibble_lut(&coefficients);
    let mut prev = train[GF2_TRAIN_TRANSITIONS];
    for &value in &values[GF2_TRAIN_TRANSITIONS + 1..] {
        if value.len() != expected_val_len
            || value[0] != b'"'
            || value[expected_val_len - 1] != b'"'
            || !slice_eq_fast(&value[1..prefix_end], prefix)
        {
            return None;
        }
        let word = parse_hex_u64_le(&value[prefix_end..expected_val_len - 1])?;
        if apply_affine_lut(&lut, prev) != word {
            return None;
        }
        prev = word;
    }
    let mut out = Vec::with_capacity(size);
    out.push(MODEL_GF2_AFFINE);
    put_bytes(&mut out, prefix);
    out.extend_from_slice(&train[0].to_le_bytes());
    for coefficient in coefficients {
        out.extend_from_slice(&coefficient.to_le_bytes());
    }
    Some(out)
}

