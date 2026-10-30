// MODEL_PATTERN: fixed-shape strings made of digit fields and literal / small-dictionary text
// (timestamps in any layout, IPv4 addresses, versions, dates, ids like `u012336`).
//
// A value is split into maximal digit runs and maximal non-digit runs. All rows must have the same
// token shape and the same width for every digit run. Non-digit tokens are either one constant
// literal or a small sorted dictionary. Every row is then mapped to one mixed-radix integer
//   N = ((f0 * R1 + f1) * R2 + f2) ...   (f_i = digit value - min, or dictionary index)
// whose row-to-row deltas are tiny for time-ordered data (consecutive seconds are consecutive
// integers even across minute and hour carries), so they bit-pack to a few bits per row. Rare large
// deltas (day or month carries) are stored as explicit exceptions.
//
// Payload (after the tag byte):
//   uleb token_count (1..=32)
//   per token: 0 = digits: byte width (1..=18), uleb min, uleb radix
//              1 = literal: uleb len, bytes
//              2 = dictionary: uleb entries (2..=64), entries as uleb len + bytes, strictly ascending
//   uleb N0 (row 0), byte W (0..=64), uleb exception_count,
//   per exception: uleb (row - previous exception row; the first one is relative to row 0), uleb zigzag
//   packed zigzag deltas for rows 1..count, W bits each (exception rows hold 0), zero padded.

const PATTERN_MAX_TOKENS: usize = 32;
const PATTERN_MAX_DIGITS: usize = 18;
const PATTERN_MAX_DICT: usize = 64;
const PATTERN_MAX_PRODUCT: u64 = 1 << 62;

#[derive(Clone, Copy)]
struct PatSpan {
    digit: bool,
    start: u32,
    end: u32,
}

/// Split `v` into digit / non-digit runs. `None` if it has too many tokens or a digit run is too wide.
fn pattern_split(v: &[u8], out: &mut [PatSpan; PATTERN_MAX_TOKENS]) -> Option<usize> {
    let mut n = 0usize;
    let mut i = 0usize;
    while i < v.len() {
        let digit = v[i].is_ascii_digit();
        let start = i;
        while i < v.len() && v[i].is_ascii_digit() == digit {
            i += 1;
        }
        if n == PATTERN_MAX_TOKENS || (digit && i - start > PATTERN_MAX_DIGITS) {
            return None;
        }
        out[n] = PatSpan {
            digit,
            start: start as u32,
            end: i as u32,
        };
        n += 1;
    }
    Some(n)
}

#[inline(always)]
fn pattern_digits_value(d: &[u8]) -> u64 {
    d.iter().fold(0u64, |a, &b| a * 10 + u64::from(b - b'0'))
}

enum PatTok<'a> {
    Digits { width: usize, min: u64, radix: u64 },
    Lit(&'a [u8]),
    Dict(Vec<&'a [u8]>),
}

fn pack_pattern_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || limit < 16 {
        return None;
    }
    let mut first = [PatSpan { digit: false, start: 0, end: 0 }; PATTERN_MAX_TOKENS];
    let ntok = pattern_split(values[0], &mut first)?;
    if !first[..ntok].iter().any(|s| s.digit) {
        return None;
    }
    // Pass 1: shape check, digit ranges, non-digit distinct sets.
    let mut mins = [u64::MAX; PATTERN_MAX_TOKENS];
    let mut maxs = [0u64; PATTERN_MAX_TOKENS];
    let mut distinct: Vec<Vec<&[u8]>> = vec![Vec::new(); ntok];
    let mut cur = first;
    for (r, &v) in values.iter().enumerate() {
        if r > 0 && pattern_split(v, &mut cur)? != ntok {
            return None;
        }
        for t in 0..ntok {
            let (s, c) = (first[t], cur[t]);
            if s.digit != c.digit {
                return None;
            }
            let part = &v[c.start as usize..c.end as usize];
            if s.digit {
                if c.end - c.start != s.end - s.start {
                    return None;
                }
                let x = pattern_digits_value(part);
                mins[t] = mins[t].min(x);
                maxs[t] = maxs[t].max(x);
            } else {
                let set = &mut distinct[t];
                if !set.iter().any(|e| *e == part) {
                    if set.len() == PATTERN_MAX_DICT {
                        return None;
                    }
                    set.push(part);
                }
            }
        }
    }
    let mut toks: Vec<PatTok> = Vec::with_capacity(ntok);
    let mut product = 1u64;
    for t in 0..ntok {
        if first[t].digit {
            let radix = maxs[t] - mins[t] + 1;
            product = product.checked_mul(radix)?;
            toks.push(PatTok::Digits {
                width: (first[t].end - first[t].start) as usize,
                min: mins[t],
                radix,
            });
        } else if distinct[t].len() == 1 {
            toks.push(PatTok::Lit(distinct[t][0]));
        } else {
            let mut set = core::mem::take(&mut distinct[t]);
            set.sort_unstable();
            product = product.checked_mul(set.len() as u64)?;
            toks.push(PatTok::Dict(set));
        }
        if product > PATTERN_MAX_PRODUCT {
            return None;
        }
    }
    if product == 1 {
        return None;
    }
    // Pass 2: mixed-radix integer per row, zigzag deltas.
    let mut nums: Vec<u64> = Vec::with_capacity(n);
    for &v in values {
        pattern_split(v, &mut cur)?;
        let mut acc = 0u64;
        for t in 0..ntok {
            let part = &v[cur[t].start as usize..cur[t].end as usize];
            match &toks[t] {
                PatTok::Digits { min, radix, .. } => {
                    acc = acc * radix + (pattern_digits_value(part) - min);
                }
                PatTok::Dict(set) => {
                    let idx = set.binary_search(&part).ok()?;
                    acc = acc * set.len() as u64 + idx as u64;
                }
                PatTok::Lit(_) => {}
            }
        }
        nums.push(acc);
    }
    let zz = |d: i64| -> u64 { ((d << 1) ^ (d >> 63)) as u64 };
    let deltas: Vec<u64> = (1..n).map(|i| zz(nums[i] as i64 - nums[i - 1] as i64)).collect();
    let mut hist = [0usize; 65];
    for &z in &deltas {
        hist[(64 - z.leading_zeros()) as usize] += 1;
    }
    // Pick the width minimizing packed bytes + ~6 bytes per exception.
    let mut best_w = 64usize;
    let mut best_cost = usize::MAX;
    for w in 0..=64usize {
        let exc: usize = hist[w + 1..].iter().sum();
        let cost = (deltas.len() * w).div_ceil(8) + exc * 6;
        if cost < best_cost {
            best_cost = cost;
            best_w = w;
        }
    }
    let w = best_w as u8;
    let is_exc = |z: u64| best_w < 64 && z >> best_w != 0;
    let mut out = Vec::with_capacity(64 + deltas.len() * best_w / 8);
    out.push(MODEL_PATTERN);
    put_uleb(&mut out, ntok);
    for tok in &toks {
        match tok {
            PatTok::Digits { width, min, radix } => {
                out.push(0);
                out.push(*width as u8);
                put_uleb(&mut out, *min as usize);
                put_uleb(&mut out, *radix as usize);
            }
            PatTok::Lit(l) => {
                out.push(1);
                put_bytes(&mut out, l);
            }
            PatTok::Dict(set) => {
                out.push(2);
                put_uleb(&mut out, set.len());
                for e in set {
                    put_bytes(&mut out, e);
                }
            }
        }
    }
    put_uleb(&mut out, nums[0] as usize);
    out.push(w);
    let excs: Vec<(usize, u64)> = deltas
        .iter()
        .enumerate()
        .filter(|(_, &z)| is_exc(z))
        .map(|(i, &z)| (i + 1, z))
        .collect();
    put_uleb(&mut out, excs.len());
    let mut prev_row = 0usize;
    for &(row, z) in &excs {
        put_uleb(&mut out, row - prev_row);
        put_uleb(&mut out, z as usize);
        prev_row = row;
    }
    push_bit_values(
        &mut out,
        deltas.len(),
        w,
        deltas.iter().map(|&z| if is_exc(z) { 0 } else { z }),
    );
    (out.len() < limit).then_some(out)
}

/// Decode a `MODEL_PATTERN` payload (cursor positioned after the tag byte), calling `sink` once per
/// row. Shared by the strict oracle decoder and the production column parser.
fn pattern_decode(
    cursor: &mut Cursor<'_>,
    count: usize,
    sink: &mut dyn FnMut(&[u8]),
) -> Result<()> {
    if count < 2 {
        return Err(invalid("column pattern count"));
    }
    let ntok = cursor.uleb()?;
    if !(1..=PATTERN_MAX_TOKENS).contains(&ntok) {
        return Err(invalid("column pattern tokens"));
    }
    let mut toks: Vec<PatTok> = Vec::with_capacity(ntok);
    let mut product = 1u64;
    for _ in 0..ntok {
        match cursor.byte()? {
            0 => {
                let width = cursor.byte()? as usize;
                let min = cursor.uleb()? as u64;
                let radix = cursor.uleb()? as u64;
                if !(1..=PATTERN_MAX_DIGITS).contains(&width) || radix == 0 {
                    return Err(invalid("column pattern digits"));
                }
                let max = min.checked_add(radix - 1).ok_or_else(|| invalid("column pattern range"))?;
                if max >= 10u64.pow(width as u32) {
                    return Err(invalid("column pattern width"));
                }
                product = product.checked_mul(radix).ok_or_else(|| invalid("column pattern product"))?;
                toks.push(PatTok::Digits { width, min, radix });
            }
            1 => {
                let len = cursor.uleb()?;
                if len > MAX_BLOCK_SIZE {
                    return Err(invalid("column pattern literal"));
                }
                toks.push(PatTok::Lit(cursor.take(len)?));
            }
            2 => {
                let entries = cursor.uleb()?;
                if !(2..=PATTERN_MAX_DICT).contains(&entries) {
                    return Err(invalid("column pattern dictionary"));
                }
                let mut set: Vec<&[u8]> = Vec::with_capacity(entries);
                for _ in 0..entries {
                    let len = cursor.uleb()?;
                    let e = cursor.take(len)?;
                    if set.last().is_some_and(|l| *l >= e) {
                        return Err(invalid("column pattern dictionary order"));
                    }
                    set.push(e);
                }
                product = product.checked_mul(entries as u64).ok_or_else(|| invalid("column pattern product"))?;
                toks.push(PatTok::Dict(set));
            }
            _ => return Err(invalid("column pattern token kind")),
        }
        if product > PATTERN_MAX_PRODUCT {
            return Err(invalid("column pattern product"));
        }
    }
    if product == 1 {
        return Err(invalid("column pattern degenerate"));
    }
    let n0 = cursor.uleb()? as u64;
    if n0 >= product {
        return Err(invalid("column pattern first value"));
    }
    let w = cursor.byte()?;
    if w > 64 {
        return Err(invalid("column pattern width"));
    }
    let exc_count = cursor.uleb()?;
    if exc_count > count - 1 {
        return Err(invalid("column pattern exceptions"));
    }
    let mut excs: Vec<(usize, u64)> = Vec::with_capacity(exc_count);
    let mut row = 0usize;
    for k in 0..exc_count {
        let step = cursor.uleb()?;
        row = row.checked_add(step).ok_or_else(|| invalid("column pattern exception row"))?;
        if (k > 0 && step == 0) || row == 0 || row >= count {
            return Err(invalid("column pattern exception row"));
        }
        let z = cursor.uleb()? as u64;
        if w < 64 && z >> w == 0 {
            return Err(invalid("column pattern exception value"));
        }
        excs.push((row, z));
    }
    let packed = cursor.take(bit_bytes(count - 1, w)?)?;
    let mut reader = BitReader::new(packed);
    let mut next_exc = 0usize;
    let mut cur = n0;
    let mut vals = [0u64; PATTERN_MAX_TOKENS];
    let mut row_buf: Vec<u8> = Vec::with_capacity(64);
    // Fast path: when the last variable token is a digit field, small non-negative deltas only touch
    // that field, so only its digits are re-rendered in place.
    let mut tail = None; // (radix, width, min, bytes of trailing literals)
    {
        let mut trail = 0usize;
        for tok in toks.iter().rev() {
            match tok {
                PatTok::Lit(l) => trail += l.len(),
                PatTok::Digits { width, min, radix } => {
                    tail = Some((*radix, *width, *min, trail));
                    break;
                }
                PatTok::Dict(_) => break,
            }
        }
    }
    let render = |n: u64, vals: &mut [u64; PATTERN_MAX_TOKENS], row_buf: &mut Vec<u8>| {
        let mut rem = n;
        for t in (0..toks.len()).rev() {
            match &toks[t] {
                PatTok::Digits { radix, .. } => {
                    vals[t] = rem % radix;
                    rem /= radix;
                }
                PatTok::Dict(set) => {
                    vals[t] = rem % set.len() as u64;
                    rem /= set.len() as u64;
                }
                PatTok::Lit(_) => {}
            }
        }
        row_buf.clear();
        for (t, tok) in toks.iter().enumerate() {
            match tok {
                PatTok::Lit(l) => row_buf.extend_from_slice(l),
                PatTok::Dict(set) => row_buf.extend_from_slice(set[vals[t] as usize]),
                PatTok::Digits { width, min, .. } => {
                    let mut x = min + vals[t];
                    let at = row_buf.len();
                    row_buf.resize(at + width, b'0');
                    for k in (0..*width).rev() {
                        row_buf[at + k] = b'0' + (x % 10) as u8;
                        x /= 10;
                    }
                }
            }
        }
    };
    let last_tok = toks.len();
    let _ = last_tok;
    let mut last_field = 0u64; // value of the last digit field (valid after a render)
    let sync_last = |vals: &[u64; PATTERN_MAX_TOKENS], toks: &[PatTok]| -> u64 {
        match toks.iter().rposition(|t| !matches!(t, PatTok::Lit(_))) {
            Some(i) => vals[i],
            None => 0,
        }
    };
    render(cur, &mut vals, &mut row_buf);
    last_field = sync_last(&vals, &toks).max(last_field);
    sink(&row_buf);
    for r in 1..count {
        let packed_z = reader.read(w)?;
        let z = if next_exc < excs.len() && excs[next_exc].0 == r {
            if packed_z != 0 {
                return Err(invalid("column pattern exception padding"));
            }
            next_exc += 1;
            excs[next_exc - 1].1
        } else {
            packed_z
        };
        let delta = ((z >> 1) as i64) ^ -((z & 1) as i64);
        if let Some((radix, width, min, trail)) = tail {
            if delta >= 0 && (delta as u64) < radix - last_field {
                if delta != 0 {
                    last_field += delta as u64;
                    cur += delta as u64;
                    let mut x = min + last_field;
                    let end = row_buf.len() - trail;
                    for k in 1..=width {
                        row_buf[end - k] = b'0' + (x % 10) as u8;
                        x /= 10;
                    }
                }
                sink(&row_buf);
                continue;
            }
        }
        let next = (cur as i64)
            .checked_add(delta)
            .ok_or_else(|| invalid("column pattern delta"))?;
        if next < 0 || next as u64 >= product {
            return Err(invalid("column pattern range"));
        }
        cur = next as u64;
        render(cur, &mut vals, &mut row_buf);
        last_field = sync_last(&vals, &toks);
        sink(&row_buf);
    }
    reader.finish_zero_padded()?;
    Ok(())
}

impl<'a> PreparedColumn<'a> {
    /// Production parse for `MODEL_PATTERN`: materialize the rows once as length-prefixed values in the
    /// decode arena and expose them through the RAW stream (the same trick as wide dictionaries).
    fn parse_pattern(
        mut cursor: Cursor<'a>,
        count: usize,
        scratch: &mut DecodeScratch,
    ) -> Result<Self> {
        let mut rows: Vec<u8> = Vec::with_capacity(count * 24);
        let mut max_len = 0usize;
        pattern_decode(&mut cursor, count, &mut |row| {
            put_uleb(&mut rows, row.len());
            rows.extend_from_slice(row);
            max_len = max_len.max(row.len());
        })?;
        if !cursor.done() {
            return Err(invalid("column model trailing bytes"));
        }
        let start = scratch.raw_arena.len();
        if start + rows.len() + 32 > scratch.raw_arena.capacity() {
            return Err(invalid("column pattern arena overflow"));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                rows.as_ptr(),
                scratch.raw_arena.as_mut_ptr().add(start),
                rows.len(),
            );
            scratch.raw_arena.set_len(start + rows.len());
        }
        let bytes: &'a [u8] = unsafe {
            std::slice::from_raw_parts(scratch.raw_arena.as_ptr().add(start), rows.len())
        };
        Ok(Self {
            const_full: ConstValue::None,
            const_lead: LeadValue::Empty,
            const_trail: b"",
            dict_slices: None,
            max_emit_len: max_len,
            stream: ColumnStream::Raw {
                cursor: Cursor::new(bytes),
            },
        })
    }
}
