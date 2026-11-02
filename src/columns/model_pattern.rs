// MODEL_PATTERN: fixed-shape strings made of digit fields and literal / small-dictionary text
// (timestamps in any layout, IPv4 addresses, versions, dates, ids like `u012336`).
//
// Every value is split into maximal digit runs and maximal non-digit runs. Rows share the same
// token shape and digit widths; non-digit tokens are either constant literals or sorted <=64-entry
// dictionaries. Each row maps to a mixed-radix integer `N = ((f0 * R1 + f1) * R2 + f2) ...` whose
// zigzag row-to-row deltas bit-pack tightly with explicit outlier exceptions.

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

#[derive(Default)]
struct PatternScratch {
    dig_vals: Vec<u64>,
    dict_ids: Vec<u8>,
    deltas: Vec<u64>,
    excs: Vec<(usize, u64)>,
}

std::thread_local! {
    static PATTERN_SCRATCH: core::cell::RefCell<PatternScratch> =
        core::cell::RefCell::new(PatternScratch::default());
}

#[inline(always)]
fn pattern_split(v: &[u8], out: &mut [PatSpan; PATTERN_MAX_TOKENS]) -> Option<usize> {
    let (mut n, mut i, len, ptr) = (0usize, 0usize, v.len(), v.as_ptr());
    while i < len {
        let digit = unsafe { (*ptr.add(i)).is_ascii_digit() };
        let start = i;
        i += 1;
        while i < len && unsafe { (*ptr.add(i)).is_ascii_digit() } == digit {
            i += 1;
        }
        if n == PATTERN_MAX_TOKENS || (digit && i - start > PATTERN_MAX_DIGITS) {
            return None;
        }
        out[n] = PatSpan { digit, start: start as u32, end: i as u32 };
        n += 1;
    }
    Some(n)
}

#[inline(always)]
fn pattern_digits_value(d: &[u8]) -> u64 {
    unsafe {
        let p = d.as_ptr();
        match d.len() {
            2 => ((*p - b'0') as u64) * 10 + ((*p.add(1) - b'0') as u64),
            4 => {
                let raw = u32::from_le(p.cast::<u32>().read_unaligned()) - 0x3030_3030;
                let pair = (raw & 0x00ff_00ff) * 10 + ((raw >> 8) & 0x00ff_00ff);
                ((pair & 0xffff) * 100 + (pair >> 16)) as u64
            }
            3 => ((*p - b'0') as u64) * 100 + ((*p.add(1) - b'0') as u64) * 10 + ((*p.add(2) - b'0') as u64),
            1 => (*p - b'0') as u64,
            6 => {
                let hi = ((*p - b'0') as u64) * 10 + ((*p.add(1) - b'0') as u64);
                let raw4 = u32::from_le(p.add(2).cast::<u32>().read_unaligned()) - 0x3030_3030;
                let pair = (raw4 & 0x00ff_00ff) * 10 + ((raw4 >> 8) & 0x00ff_00ff);
                hi * 10_000 + (((pair & 0xffff) * 100 + (pair >> 16)) as u64)
            }
            _ => d.iter().fold(0u64, |a, &b| a * 10 + u64::from(b - b'0')),
        }
    }
}

/// Gather a fixed-length `4..=32` byte slice into two `u128` words using only in-bounds loads.
#[inline(always)]
fn load_pat_words(v: &[u8]) -> (u128, u128) {
    let (len, p) = (v.len(), v.as_ptr());
    unsafe {
        if len >= 16 {
            let lo = u128::from_le(p.cast::<u128>().read_unaligned());
            let hi = if len > 16 { u128::from_le(p.add(len - 16).cast::<u128>().read_unaligned()) } else { 0 };
            (lo, hi)
        } else if len >= 8 {
            let lo64 = u64::from_le(p.cast::<u64>().read_unaligned()) as u128;
            let hi64 = u64::from_le(p.add(len - 8).cast::<u64>().read_unaligned()) as u128;
            (lo64 | (hi64 << 64), 0)
        } else {
            let lo32 = u32::from_le(p.cast::<u32>().read_unaligned()) as u128;
            let hi32 = u32::from_le(p.add(len - 4).cast::<u32>().read_unaligned()) as u128;
            (lo32 | (hi32 << 32), 0)
        }
    }
}

enum PatTok<'a> {
    Digits { width: usize, min: u64, radix: u64 },
    Lit(&'a [u8]),
    Dict(Vec<&'a [u8]>),
}

/// True when `v0` and `v1` split into the same pattern shape AND share identical non-digit literals.
#[inline(always)]
fn pattern_r1_same_literals(v0: &[u8], v1: &[u8]) -> bool {
    let mut s0 = [PatSpan { digit: false, start: 0, end: 0 }; PATTERN_MAX_TOKENS];
    let mut s1 = [PatSpan { digit: false, start: 0, end: 0 }; PATTERN_MAX_TOKENS];
    let Some(n0) = pattern_split(v0, &mut s0) else { return false };
    if v1.len() != v0.len() || pattern_split(v1, &mut s1) != Some(n0) {
        return false;
    }
    let mut any_dig = false;
    for t in 0..n0 {
        let (a, b) = (s0[t], s1[t]);
        if a.digit != b.digit || a.start != b.start || a.end != b.end {
            return false;
        }
        if a.digit {
            any_dig = true;
        } else if &v0[a.start as usize..a.end as usize] != &v1[b.start as usize..b.end as usize] {
            return false;
        }
    }
    any_dig
}

fn pack_pattern_model(values: &[&[u8]], limit: usize) -> Option<Vec<u8>> {
    let n = values.len();
    if n < 2 || limit < 16 {
        return None;
    }
    let v0 = values[0];
    let mut first = [PatSpan { digit: false, start: 0, end: 0 }; PATTERN_MAX_TOKENS];
    let ntok = pattern_split(v0, &mut first)?;
    let (mut ndig, mut nlit) = (0usize, 0usize);
    let mut dig_spans = [(0u32, 0u32); PATTERN_MAX_TOKENS];
    let mut lit0 = [&[][..]; PATTERN_MAX_TOKENS];
    for s in &first[..ntok] {
        if s.digit {
            dig_spans[ndig] = (s.start, s.end);
            ndig += 1;
        } else {
            lit0[nlit] = &v0[s.start as usize..s.end as usize];
            nlit += 1;
        }
    }
    if ndig == 0 {
        return None;
    }
    // Stack-only shape rejection on row 1 (~15 ns, zero heap allocations on non-pattern columns).
    let mut cur = [PatSpan { digit: false, start: 0, end: 0 }; PATTERN_MAX_TOKENS];
    if pattern_split(values[1], &mut cur)? != ntok {
        return None;
    }
    for t in 0..ntok {
        let (s, c) = (first[t], cur[t]);
        if s.digit != c.digit || (s.digit && c.end - c.start != s.end - s.start) {
            return None;
        }
    }

    let len0 = v0.len();
    let use_swar = (4..=32).contains(&len0);
    let (exp_lo, exp_hi, check_lo, check_hi, dig_lo, dig_hi, dig06_lo, dig06_hi, dig10_lo, dig10_hi) =
        if use_swar {
            let (mut lit_arr, mut dig_arr, mut exp_arr) = ([0u8; 32], [0u8; 32], [0u8; 32]);
            for s in &first[..ntok] {
                for idx in s.start as usize..s.end as usize {
                    if s.digit {
                        dig_arr[idx] = 0xff;
                        exp_arr[idx] = b'0';
                    } else {
                        lit_arr[idx] = 0xff;
                        exp_arr[idx] = v0[idx];
                    }
                }
            }
            let (l_lo, l_hi) = load_pat_words(&lit_arr[..len0]);
            let (d_lo, d_hi) = load_pat_words(&dig_arr[..len0]);
            let (e_lo, e_hi) = load_pat_words(&exp_arr[..len0]);
            const M_F0: u128 = 0xf0f0_f0f0_f0f0_f0f0_f0f0_f0f0_f0f0_f0f0;
            const M_06: u128 = 0x0606_0606_0606_0606_0606_0606_0606_0606;
            const M_10: u128 = 0x1010_1010_1010_1010_1010_1010_1010_1010;
            (e_lo, e_hi, l_lo | (d_lo & M_F0), l_hi | (d_hi & M_F0), d_lo, d_hi, d_lo & M_06, d_hi & M_06, d_lo & M_10, d_hi & M_10)
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        };

    PATTERN_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let total_dig = n * ndig;
        sc.dig_vals.clear();
        sc.dig_vals.reserve(total_dig);
        unsafe {
            sc.dig_vals.set_len(total_dig);
        }
        let mut has_dict_lit = false;

        let mut mins = [u64::MAX; PATTERN_MAX_TOKENS];
        let mut maxs = [0u64; PATTERN_MAX_TOKENS];
        let mut dict_extra: [Vec<&[u8]>; PATTERN_MAX_TOKENS] = core::array::from_fn(|_| Vec::new());

        for (r, &v) in values.iter().enumerate() {
            let swar_hit = if r == 0 {
                true
            } else if use_swar && v.len() == len0 {
                let (v_lo, v_hi) = load_pat_words(v);
                let diff_lo = v_lo ^ exp_lo;
                let ok_lo = (diff_lo & check_lo) == 0
                    && (((diff_lo & dig_lo).wrapping_add(dig06_lo)) & dig10_lo) == 0;
                if len0 > 16 {
                    let diff_hi = v_hi ^ exp_hi;
                    ok_lo && (diff_hi & check_hi) == 0 && (((diff_hi & dig_hi).wrapping_add(dig06_hi)) & dig10_hi) == 0
                } else {
                    ok_lo
                }
            } else {
                false
            };

            let dig_base = r * ndig;
            if swar_hit {
                for d in 0..ndig {
                    let (st, en) = dig_spans[d];
                    let x = pattern_digits_value(unsafe { v.get_unchecked(st as usize..en as usize) });
                    mins[d] = mins[d].min(x);
                    maxs[d] = maxs[d].max(x);
                    unsafe { *sc.dig_vals.get_unchecked_mut(dig_base + d) = x; }
                }
            } else {
                if pattern_split(v, &mut cur)? != ntok {
                    return None;
                }
                let lit_base = r * nlit;
                let (mut d, mut l) = (0usize, 0usize);
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
                        mins[d] = mins[d].min(x);
                        maxs[d] = maxs[d].max(x);
                        unsafe { *sc.dig_vals.get_unchecked_mut(dig_base + d) = x; }
                        d += 1;
                    } else {
                        if part != lit0[l] {
                            if !has_dict_lit {
                                sc.dict_ids.clear();
                                sc.dict_ids.resize(n * nlit, 0);
                                has_dict_lit = true;
                            }
                            let extra = &mut dict_extra[l];
                            let id = if let Some(pos) = extra.iter().position(|e| *e == part) {
                                (pos + 1) as u8
                            } else {
                                if extra.len() + 1 == PATTERN_MAX_DICT {
                                    return None;
                                }
                                extra.push(part);
                                extra.len() as u8
                            };
                            sc.dict_ids[lit_base + l] = id;
                        }
                        l += 1;
                    }
                }
            }
        }

        let mut toks: Vec<PatTok> = Vec::with_capacity(ntok);
        let mut remaps = [[0u8; PATTERN_MAX_DICT]; PATTERN_MAX_TOKENS];
        let mut product = 1u64;
        let (mut d, mut l) = (0usize, 0usize);
        for t in 0..ntok {
            if first[t].digit {
                let radix = maxs[d] - mins[d] + 1;
                product = product.checked_mul(radix)?;
                toks.push(PatTok::Digits { width: (first[t].end - first[t].start) as usize, min: mins[d], radix });
                d += 1;
            } else if dict_extra[l].is_empty() {
                toks.push(PatTok::Lit(lit0[l]));
                l += 1;
            } else {
                let extra = &dict_extra[l];
                let card = 1 + extra.len();
                let mut pairs: [(&[u8], u8); PATTERN_MAX_DICT] = [(&[], 0); PATTERN_MAX_DICT];
                pairs[0] = (lit0[l], 0);
                for (k, &s) in extra.iter().enumerate() {
                    pairs[k + 1] = (s, (k + 1) as u8);
                }
                pairs[..card].sort_unstable_by(|a, b| a.0.cmp(b.0));
                let mut sorted_set = Vec::with_capacity(card);
                for (rank, &(s, ins_id)) in pairs[..card].iter().enumerate() {
                    remaps[l][ins_id as usize] = rank as u8;
                    sorted_set.push(s);
                }
                product = product.checked_mul(card as u64)?;
                toks.push(PatTok::Dict(sorted_set));
                l += 1;
            }
            if product > PATTERN_MAX_PRODUCT {
                return None;
            }
        }
        if product == 1 {
            return None;
        }

        #[derive(Clone, Copy)]
        enum StepKind {
            Dig { idx: u8, min: u64, radix: u64 },
            Dict { idx: u8, radix: u64 },
        }
        let mut steps = [StepKind::Dig { idx: 0, min: 0, radix: 1 }; PATTERN_MAX_TOKENS];
        let (mut nsteps, mut d_i, mut l_i) = (0usize, 0usize, 0usize);
        for tok in &toks {
            match tok {
                PatTok::Digits { min, radix, .. } => {
                    if *radix > 1 {
                        steps[nsteps] = StepKind::Dig { idx: d_i as u8, min: *min, radix: *radix };
                        nsteps += 1;
                    }
                    d_i += 1;
                }
                PatTok::Lit(_) => l_i += 1,
                PatTok::Dict(set) => {
                    steps[nsteps] = StepKind::Dict { idx: l_i as u8, radix: set.len() as u64 };
                    nsteps += 1;
                    l_i += 1;
                }
            }
        }

        let PatternScratch { ref dig_vals, ref dict_ids, ref mut deltas, ref mut excs } = *sc;
        let zz = |delta: i64| -> u64 { ((delta << 1) ^ (delta >> 63)) as u64 };
        deltas.clear();
        deltas.reserve(n - 1);
        let mut hist = [0usize; 65];
        let (mut n0, mut prev_num) = (0u64, 0u64);
        let active_steps = &steps[..nsteps];
        if !has_dict_lit {
            for r in 0..n {
                let dig_row = unsafe { dig_vals.get_unchecked(r * ndig..(r + 1) * ndig) };
                let mut acc = 0u64;
                for &st in active_steps {
                    if let StepKind::Dig { idx, min, radix } = st {
                        acc = acc * radix + (unsafe { *dig_row.get_unchecked(idx as usize) } - min);
                    }
                }
                if r == 0 {
                    n0 = acc;
                } else {
                    let z = zz(acc as i64 - prev_num as i64);
                    deltas.push(z);
                    hist[(64 - z.leading_zeros()) as usize] += 1;
                }
                prev_num = acc;
            }
        } else {
            for r in 0..n {
                let dig_row = &dig_vals[r * ndig..(r + 1) * ndig];
                let lit_row = &dict_ids[r * nlit..(r + 1) * nlit];
                let mut acc = 0u64;
                for &st in active_steps {
                    match st {
                        StepKind::Dig { idx, min, radix } => {
                            acc = acc * radix + (unsafe { *dig_row.get_unchecked(idx as usize) } - min);
                        }
                        StepKind::Dict { idx, radix } => {
                            let ins = unsafe { *lit_row.get_unchecked(idx as usize) } as usize;
                            acc = acc * radix + remaps[idx as usize][ins] as u64;
                        }
                    }
                }
                if r == 0 {
                    n0 = acc;
                } else {
                    let z = zz(acc as i64 - prev_num as i64);
                    deltas.push(z);
                    hist[(64 - z.leading_zeros()) as usize] += 1;
                }
                prev_num = acc;
            }
        }

        let delta_len = deltas.len();
        let (mut best_w, mut best_cost, mut exc_rem) = (64usize, usize::MAX, delta_len);
        for w in 0..=64usize {
            exc_rem -= hist[w];
            let cost = (delta_len * w).div_ceil(8) + exc_rem * 6;
            if cost < best_cost {
                best_cost = cost;
                best_w = w;
            }
        }
        if 4 + (delta_len * best_w).div_ceil(8) >= limit {
            return None;
        }

        let w = best_w as u8;
        let is_exc = |z: u64| best_w < 64 && z >> best_w != 0;
        let mut out = take_model_buf(64 + (delta_len * best_w).div_ceil(8));
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
        put_uleb(&mut out, n0 as usize);
        out.push(w);
        excs.clear();
        for (i, &z) in deltas.iter().enumerate() {
            if is_exc(z) {
                excs.push((i + 1, z));
            }
        }
        put_uleb(&mut out, excs.len());
        let mut prev_row = 0usize;
        for &(row, z) in excs.iter() {
            put_uleb(&mut out, row - prev_row);
            put_uleb(&mut out, z as usize);
            prev_row = row;
        }
        if out.len() + (delta_len * best_w).div_ceil(8) >= limit {
            return None;
        }
        push_bit_values(&mut out, delta_len, w, deltas.iter().map(|&z| if is_exc(z) { 0 } else { z }));
        (out.len() < limit).then_some(out)
    })
}

/// Decode a `MODEL_PATTERN` payload (cursor positioned after the tag byte), calling `sink` once per
/// row. Used by the strict oracle decoder and as the fallback for huge active spans (`> 2048` B).
fn pattern_decode(cursor: &mut Cursor<'_>, count: usize, sink: &mut dyn FnMut(&[u8])) -> Result<()> {
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
    let (mut next_exc, mut cur) = (0usize, n0);
    let mut vals = [0u64; PATTERN_MAX_TOKENS];
    let mut row_buf: Vec<u8> = Vec::with_capacity(64);
    let render = |n: u64, vals: &mut [u64; PATTERN_MAX_TOKENS], row_buf: &mut Vec<u8>| {
        let mut rem = n;
        for t in (0..toks.len()).rev() {
            match &toks[t] {
                PatTok::Digits { radix, .. } => { vals[t] = rem % radix; rem /= radix; }
                PatTok::Dict(set) => { vals[t] = rem % set.len() as u64; rem /= set.len() as u64; }
                PatTok::Lit(_) => {}
            }
        }
        row_buf.clear();
        for (t, tok) in toks.iter().enumerate() {
            match tok {
                PatTok::Lit(l) => row_buf.extend_from_slice(l),
                PatTok::Dict(set) => row_buf.extend_from_slice(set[vals[t] as usize]),
                PatTok::Digits { width, min, .. } => {
                    let (mut x, at) = (min + vals[t], row_buf.len());
                    row_buf.resize(at + width, b'0');
                    for k in (0..*width).rev() {
                        row_buf[at + k] = b'0' + (x % 10) as u8;
                        x /= 10;
                    }
                }
            }
        }
    };
    render(cur, &mut vals, &mut row_buf);
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
        let next = (cur as i64).checked_add(delta).ok_or_else(|| invalid("column pattern delta"))?;
        if next < 0 || next as u64 >= product {
            return Err(invalid("column pattern range"));
        }
        cur = next as u64;
        render(cur, &mut vals, &mut row_buf);
        sink(&row_buf);
    }
    reader.finish_zero_padded()?;
    Ok(())
}
