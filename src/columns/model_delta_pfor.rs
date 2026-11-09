// Exception-patched row-delta numeric column model (`PFOR-Delta`, `MODEL_DECIMAL_DELTA` with `width_hdr = 0x80 | base_width`).
//
// Cumulative counters, metrics, and multi-partition time-series (such as `datasets/covid-19`)
// have small row-to-row steps (`0..63`) on >95% of rows, punctuated by rare massive resets at
// entity/country boundaries (e.g., `30,000,000 -> 0`). Unpatched delta coding is forced to use
// `bit_width(max_zigzag)` (`~26` bits) for every row. `PFOR-Delta` packs the `n - 1` bulk steps
// in `base_width` (`0..=63`) bits and stores only the outlier steps as `(gap: uleb, zigzag: uleb64)`
// exceptions alongside exact `[min_val, min_val + span]` block zone-map bounds.

const PFOR_DELTA_FLAG: u8 = 0x80;

#[inline(always)]
fn uleb64_len(mut v: u64) -> usize {
    let mut len = 1usize;
    while v >= 0x80 {
        v >>= 7;
        len += 1;
    }
    len
}

#[inline(always)]
fn put_uleb64(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            return;
        }
    }
}

#[inline(always)]
fn read_uleb64(c: &mut Cursor<'_>) -> Result<u64> {
    let mut v = 0u64;
    let mut shift = 0u32;
    for _ in 0..10 {
        let b = c.byte()?;
        if shift == 63 && b > 1 {
            return Err(invalid("column uleb64 overflow"));
        }
        v |= u64::from(b & 0x7f) << shift;
        if (b & 0x80) == 0 {
            if shift > 0 && b == 0 {
                return Err(invalid("non-canonical column uleb64"));
            }
            return Ok(v);
        }
        shift += 7;
    }
    Err(invalid("column uleb64 overflow"))
}

/// Extract exact `(min, max)` bounds from `MODEL_DECIMAL_PACKED`, `MODEL_DECIMAL_AFFINE`, or
/// patched `MODEL_DECIMAL_DELTA` (`PFOR-Delta`) headers in `O(1)` without unpacking values.
#[inline(always)]
fn decimal_header_bounds(col_bytes: &[u8], ones: usize) -> Option<(u64, u64)> {
    let mut c = Cursor::new(col_bytes);
    let tag = c.byte().ok()?;
    match tag {
        MODEL_DECIMAL_PACKED => {
            let min = c.u64().ok()?;
            let width = c.byte().ok()?;
            if width > 64 {
                return None;
            }
            let span = if width == 0 {
                0
            } else if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            };
            Some((min, min.saturating_add(span)))
        }
        MODEL_DECIMAL_AFFINE => {
            let first = c.u64().ok()?;
            let step = c.i64().ok()?;
            let last_i128 = first as i128 + (step as i128) * ((ones.saturating_sub(1)) as i128);
            let last = u64::try_from(last_i128).ok()?;
            Some((first.min(last), first.max(last)))
        }
        MODEL_DECIMAL_DELTA => {
            let _first = c.u64().ok()?;
            let w_hdr = c.byte().ok()?;
            if (w_hdr & PFOR_DELTA_FLAG) == 0 {
                return None;
            }
            if (w_hdr & 0x7f) >= 64 {
                return None;
            }
            let min_val = read_uleb64(&mut c).ok()?;
            let span = read_uleb64(&mut c).ok()?;
            let max_val = min_val.checked_add(span)?;
            Some((min_val, max_val))
        }
        _ => None,
    }
}

/// Try packing `vals` using exception-patched `PFOR-Delta`. Returns `Some(buf)` only when it
/// saves both `>= 16` bytes and `>= 10%` over `packed_size` (keeping `MODEL_DECIMAL_PACKED`'s
/// direct SIMD query path untouched on random/non-cumulative columns) and beats `strict_limit`.
fn try_pack_pfor_delta(
    vals: &[u64],
    min_val: u64,
    max_val: u64,
    max_zigzag: u64,
    packed_size: usize,
    unpatched_delta_size: Option<usize>,
    strict_limit: usize,
) -> Option<Vec<u8>> {
    let n = vals.len();
    if n < 16 {
        return None;
    }
    let steps = n - 1;
    let max_w = bit_width(max_zigzag);
    if max_w <= 2 {
        return None;
    }

    // Single linear pass: 65-bin bit-width histogram + exact uleb64 exception byte totals per bin.
    let mut hist = [0u32; 65];
    let mut val_bytes = [0u32; 65];
    for pair in vals.windows(2) {
        let d = (pair[1] as i128 - pair[0] as i128) as i64;
        let zz = ((d << 1) ^ (d >> 63)) as u64;
        let w = bit_width(zz) as usize;
        hist[w] += 1;
        val_bytes[w] += uleb64_len(zz) as u32;
    }

    let span = max_val - min_val;
    let fixed_hdr = 10 + uleb64_len(min_val) + uleb64_len(span);
    let max_allowed_by_packed = (packed_size * 9 / 10).min(packed_size.saturating_sub(16));
    let mut ceiling = strict_limit.min(max_allowed_by_packed);
    if let Some(uds) = unpatched_delta_size {
        ceiling = ceiling.min(uds.saturating_sub(8));
    }
    if fixed_hdr + 2 >= ceiling {
        return None;
    }

    // Sweep candidate base widths `0..max_w` (`<= 64` iterations, <100 ns) from high to low.
    let mut exc_cnt = 0usize;
    let mut exc_vbytes = 0usize;
    let mut best_bw: Option<u8> = None;
    let mut best_est = ceiling;

    for bw in (0..max_w.min(64)).rev() {
        let next_bin = (bw + 1) as usize;
        exc_cnt += hist[next_bin] as usize;
        exc_vbytes += val_bytes[next_bin] as usize;
        if exc_cnt == 0 {
            continue;
        }
        // Keep exception count bounded (`<= steps / 2`) so decode stays fast.
        if exc_cnt * 2 > steps {
            break;
        }
        let base_bytes = (steps * (bw as usize) + 7) >> 3;
        // Conservative 1.5B/gap estimate refined to exact size right after choosing `best_bw`.
        let est = fixed_hdr + uleb_len(exc_cnt) + base_bytes + exc_vbytes + exc_cnt;
        if est < best_est {
            best_est = est;
            best_bw = Some(bw);
        }
    }

    let bw = best_bw?;
    let thresh = if bw == 0 { 0u64 } else { (1u64 << bw) - 1 };

    // Compute exact encoded byte length including exact index-gap uleb lengths.
    let mut exact_exc = 0usize;
    let mut exact_exc_bytes = 0usize;
    let mut prev_idx: Option<usize> = None;
    for (idx, pair) in vals.windows(2).enumerate() {
        let d = (pair[1] as i128 - pair[0] as i128) as i64;
        let zz = ((d << 1) ^ (d >> 63)) as u64;
        if zz > thresh {
            let gap = match prev_idx {
                None => idx,
                Some(p) => idx - p - 1,
            };
            prev_idx = Some(idx);
            exact_exc += 1;
            exact_exc_bytes += uleb_len(gap) + uleb64_len(zz);
        }
    }
    if exact_exc == 0 {
        return None;
    }
    let base_bytes = bit_bytes(steps, bw).ok()?;
    let total_len = fixed_hdr + uleb_len(exact_exc) + base_bytes + exact_exc_bytes;
    if total_len >= ceiling {
        return None;
    }

    let mut out = take_model_buf(total_len);
    out.push(MODEL_DECIMAL_DELTA);
    out.extend_from_slice(&vals[0].to_le_bytes());
    out.push(PFOR_DELTA_FLAG | bw);
    put_uleb64(&mut out, min_val);
    put_uleb64(&mut out, span);
    put_uleb(&mut out, exact_exc);
    if bw > 0 {
        push_bit_values(
            &mut out,
            steps,
            bw,
            vals.windows(2).map(|pair| {
                let d = (pair[1] as i128 - pair[0] as i128) as i64;
                let zz = ((d << 1) ^ (d >> 63)) as u64;
                if zz > thresh { 0 } else { zz }
            }),
        );
    }
    let mut last_idx: Option<usize> = None;
    for (idx, pair) in vals.windows(2).enumerate() {
        let d = (pair[1] as i128 - pair[0] as i128) as i64;
        let zz = ((d << 1) ^ (d >> 63)) as u64;
        if zz > thresh {
            let gap = match last_idx {
                None => idx,
                Some(p) => idx - p - 1,
            };
            last_idx = Some(idx);
            put_uleb(&mut out, gap);
            put_uleb64(&mut out, zz);
        }
    }
    debug_assert_eq!(out.len(), total_len);
    Some(out)
}

/// Shared selection helper for `MODEL_DECIMAL_DELTA` (unpatched or `PFOR-Delta`) vs `MODEL_DECIMAL_PACKED`.
#[allow(clippy::too_many_arguments)]
fn select_packed_or_delta_column(
    vals: &[u64],
    first_num: u64,
    min_val: u64,
    max_val: u64,
    delta_ok: bool,
    max_zigzag: u64,
    min_raw_size: usize,
    raw_size_fn: impl FnOnce() -> usize,
    best_bytes: &mut Option<Vec<u8>>,
    best_len: &mut usize,
    best_kind: &mut ModelKind,
) {
    let n = vals.len();
    let packed_width = bit_width(max_val - min_val);
    let Ok(packed_bits) = bit_bytes(n, packed_width) else {
        return;
    };
    let packed_size = 10 + packed_bits;
    let delta_width = bit_width(max_zigzag);
    let delta_size = if delta_ok {
        bit_bytes(n - 1, delta_width).ok().map(|b| 10 + b)
    } else {
        None
    };

    let mut raw_cached: Option<usize> = None;
    let mut raw_fn = Some(raw_size_fn);
    let mut get_raw = || -> usize {
        if let Some(r) = raw_cached {
            r
        } else {
            let r = (raw_fn.take().unwrap())();
            raw_cached = Some(r);
            r
        }
    };

    if delta_ok {
        let pfor_lim = if best_bytes.is_some() {
            effective_limit(*best_len, *best_kind, ModelKind::DecimalDelta)
        } else if packed_size < min_raw_size {
            usize::MAX
        } else {
            get_raw()
        };
        if let Some(pfor_buf) = try_pack_pfor_delta(
            vals,
            min_val,
            max_val,
            max_zigzag,
            packed_size,
            delta_size,
            pfor_lim,
        ) {
            *best_len = pfor_buf.len();
            *best_kind = ModelKind::DecimalDelta;
            if let Some(old) = best_bytes.replace(pfor_buf) {
                put_model_buf(old);
            }
            return;
        }
    }

    if delta_size.is_some_and(|ds| ds < packed_size) {
        let ds = delta_size.unwrap();
        let lim = if best_bytes.is_some() {
            effective_limit(*best_len, *best_kind, ModelKind::DecimalDelta)
        } else if ds < min_raw_size {
            usize::MAX
        } else {
            get_raw()
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
                vals.windows(2).map(|pair| {
                    let delta = (pair[1] as i128 - pair[0] as i128) as i64;
                    ((delta << 1) ^ (delta >> 63)) as u64
                }),
            );
            *best_len = ds;
            *best_kind = ModelKind::DecimalDelta;
            if let Some(old) = best_bytes.replace(out) {
                put_model_buf(old);
            }
        }
    } else {
        let lim = if best_bytes.is_some() {
            effective_limit(*best_len, *best_kind, ModelKind::DecimalPacked)
        } else if packed_size < min_raw_size {
            usize::MAX
        } else {
            get_raw()
        };
        if packed_size < lim {
            let mut out = take_model_buf(packed_size);
            out.push(MODEL_DECIMAL_PACKED);
            out.extend_from_slice(&min_val.to_le_bytes());
            out.push(packed_width);
            push_bit_values(&mut out, n, packed_width, vals.iter().map(|&v| v - min_val));
            *best_len = packed_size;
            *best_kind = ModelKind::DecimalPacked;
            if let Some(old) = best_bytes.replace(out) {
                put_model_buf(old);
            }
        }
    }
}

/// Validated `PFOR-Delta` decoder writing reconstructed `u64` values into `dst_out[..count]`
/// and returning `or_all` (`< 10_000` check for `DecimalSmall`). Enforces `[min_val, max_val]`
/// zone-map bounds on every decoded row so corrupted streams can never disagree with zone-maps.
fn decode_pfor_delta_into(
    first: u64,
    base_width: u8,
    cursor: &mut Cursor<'_>,
    count: usize,
    dst: &mut Vec<u64>,
) -> Result<u64> {
    if count < 2 {
        return Err(invalid("column pfor delta count"));
    }
    if base_width >= 64 {
        return Err(invalid("column pfor delta width"));
    }
    let min_val = read_uleb64(cursor)?;
    let span = read_uleb64(cursor)?;
    let max_val = min_val
        .checked_add(span)
        .ok_or_else(|| invalid("column pfor delta bounds"))?;
    if first < min_val || first > max_val {
        return Err(invalid("column pfor delta range"));
    }
    let steps = count - 1;
    let exc_count = cursor.uleb()?;
    if exc_count == 0 || exc_count > steps {
        return Err(invalid("column pfor delta exception count"));
    }

    let base_off = dst.len();
    dst.reserve(count);
    dst.push(first);
    if base_width == 0 {
        dst.resize(base_off + count, 0);
    } else {
        let packed_len = match bit_bytes(steps, base_width) {
            Ok(l) => l,
            Err(e) => {
                dst.truncate(base_off);
                return Err(e);
            }
        };
        let packed_bytes = match cursor.take(packed_len) {
            Ok(b) => b,
            Err(e) => {
                dst.truncate(base_off);
                return Err(e);
            }
        };
        let mut reader = match FastBitReader::new_validated(packed_bytes, steps, base_width) {
            Ok(r) => r,
            Err(e) => {
                dst.truncate(base_off);
                return Err(e);
            }
        };
        if base_width <= 16 {
            let mut z_buf = [0u16; 64];
            let mut s = 0usize;
            while s < steps {
                let cnt = (steps - s).min(64);
                unpack_u16_chunk64(packed_bytes, s, cnt, base_width, &mut z_buf);
                for &z in &z_buf[..cnt] {
                    dst.push(u64::from(z));
                }
                s += cnt;
            }
        } else {
            for _ in 0..steps {
                dst.push(reader.next_u64());
            }
        }
    }

    let min_exc = if base_width == 0 { 1u64 } else { 1u64 << base_width };
    let mut prev_idx: Option<usize> = None;
    for _ in 0..exc_count {
        let gap = match cursor.uleb() {
            Ok(g) => g,
            Err(e) => {
                dst.truncate(base_off);
                return Err(e);
            }
        };
        let idx = match prev_idx {
            None => gap,
            Some(p) => match p.checked_add(1).and_then(|v| v.checked_add(gap)) {
                Some(v) => v,
                None => {
                    dst.truncate(base_off);
                    return Err(invalid("column pfor delta exception index"));
                }
            },
        };
        if idx >= steps {
            dst.truncate(base_off);
            return Err(invalid("column pfor delta exception index"));
        }
        let patch_zz = match read_uleb64(cursor) {
            Ok(z) => z,
            Err(e) => {
                dst.truncate(base_off);
                return Err(e);
            }
        };
        if patch_zz < min_exc || dst[base_off + 1 + idx] != 0 {
            dst.truncate(base_off);
            return Err(invalid("column pfor delta noncanonical exception"));
        }
        dst[base_off + 1 + idx] = patch_zz;
        prev_idx = Some(idx);
    }

    let mut prev = first as i128;
    let mut or_all = first;
    for slot in &mut dst[base_off + 1..base_off + count] {
        let zz = *slot;
        let delta = ((zz >> 1) as i64) ^ -((zz & 1) as i64);
        let cur = prev + i128::from(delta);
        if cur < i128::from(min_val) || cur > i128::from(max_val) {
            dst.truncate(base_off);
            return Err(invalid("column pfor delta range"));
        }
        let u = cur as u64;
        *slot = u;
        or_all |= u;
        prev = cur;
    }
    Ok(or_all)
}
