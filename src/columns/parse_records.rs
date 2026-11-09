fn parse_delimited_spans(
    data: &[u8],
    delim: u8,
    spans: &mut [(u32, u32); MAX_SLOTS],
) -> Option<usize> {
    let mut body_len = data.len();
    if body_len == 0 || data[body_len - 1] != b'\n' {
        return None;
    }
    body_len -= 1;
    if body_len > 0 && data[body_len - 1] == b'\r' {
        body_len -= 1;
    }
    if body_len == 0 || body_len > 4096 {
        return None;
    }
    let body = &data[..body_len];
    let relaxed = delim_is_relaxed(delim);
    let mut pos = 0usize;
    let mut count = 0usize;
    let mut non_empty = 0usize;
    loop {
        let start = pos;
        let end = scan_delimited_value(body, pos, delim)?;
        if end > start {
            non_empty += 1;
        }
        if count >= MAX_SLOTS {
            return None;
        }
        spans[count] = (start as u32, end as u32);
        count += 1;
        pos = end;
        if pos == body_len {
            break;
        }
        if body[pos] != delim {
            return None;
        }
        pos += 1;
        if pos == body_len {
            if !relaxed || count >= MAX_SLOTS {
                return None;
            }
            spans[count] = (pos as u32, pos as u32);
            count += 1;
            break;
        }
    }
    (count >= 2 && non_empty >= 1 && (count >= 3 || non_empty == count)).then_some(count)
}

#[inline(always)]
fn is_logfmt_key_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')
}

fn parse_logfmt_spans(
    data: &[u8],
    spans: &mut [(u32, u32); MAX_SLOTS],
) -> Option<usize> {
    let mut body_len = data.len();
    if body_len == 0 || data[body_len - 1] != b'\n' {
        return None;
    }
    body_len -= 1;
    if body_len > 0 && data[body_len - 1] == b'\r' {
        body_len -= 1;
    }
    if body_len == 0 || body_len > 4096 {
        return None;
    }
    let body = &data[..body_len];
    let mut pos = 0usize;
    let mut count = 0usize;
    while pos < body_len {
        let key_start = pos;
        while pos < body_len && is_logfmt_key_byte(body[pos]) {
            pos += 1;
        }
        if pos == key_start || pos >= body_len || body[pos] != b'=' {
            return None;
        }
        pos += 1;
        if pos >= body_len {
            return None;
        }
        let val_start = pos;
        let val_end = if body[pos] == b'"' {
            scan_string(body, pos)?
        } else {
            let mut e = pos;
            while e < body_len
                && body[e] > b' '
                && body[e] < 0x7f
                && !matches!(
                    body[e],
                    b'"' | b';' | b',' | b'(' | b')' | b'{' | b'}' | b'[' | b']'
                )
            {
                e += 1;
            }
            if e == pos {
                return None;
            }
            e
        };
        if count >= MAX_SLOTS {
            return None;
        }
        spans[count] = (val_start as u32, val_end as u32);
        count += 1;
        pos = val_end;
        if pos == body_len {
            break;
        }
        if body[pos] != b' ' {
            return None;
        }
        pos += 1;
    }
    (count >= 2).then_some(count)
}

/// Split top-level structured record values (JSON objects, CSV/TSV/pipe-delimited
/// rows, or `key=value` logfmt lines). Nested JSON arrays/objects stay exact
/// raw slot values; irregular lines safely return `None` (emitted as `RAW` rows).
fn parse_record_spans(
    data: &[u8],
    spans: &mut [(u32, u32); MAX_SLOTS],
    overflow: &mut Vec<(u32, u32)>,
) -> Option<usize> {
    let mut pos = 0;
    skip_space(data, &mut pos);
    let first = *data.get(pos)?;
    if first != b'{' {
        if pos == 0 {
            let len = data.len();
            if len < 4 || data[len - 1] != b'\n' || len > 4098 {
                return None;
            }
            if first == b'"' {
                let end = scan_string(data, 0)?;
                let d = *data.get(end)?;
                if matches!(d, b',' | b'\t' | b'|') {
                    return parse_delimited_spans(data, d, spans);
                }
                return None;
            }
            if matches!(first, b',' | b'\t') {
                return parse_delimited_spans(data, first, spans);
            }
            if first == 0x7f || matches!(first, b'}' | b']' | b'|' | b'=') {
                return None;
            }
            if first > 0x7f || matches!(first, b'(' | b')' | b';' | b'[') {
                let lim = len.min(128);
                for &bj in &data[1..lim] {
                    if bj == b'\t' {
                        return parse_delimited_spans(data, b'\t', spans);
                    }
                    if bj == b',' {
                        return parse_delimited_spans(data, b',', spans).filter(|&c| c >= 3);
                    }
                    if bj < b' ' || bj == b'"' || bj == 0x7f {
                        break;
                    }
                }
                return None;
            }
            let ptr = data.as_ptr();
            let mut i = 1usize;
            while i < len {
                let b = unsafe { *ptr.add(i) };
                if b <= b' '
                    || b >= 0x7f
                    || matches!(
                        b,
                        b'=' | b',' | b'\t' | b'|' | b'"' | b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'
                    )
                {
                    if b == b'=' {
                        if let Some(c) = parse_logfmt_spans(data, spans) {
                            return Some(c);
                        }
                        let mut j = i + 1;
                        while j < len {
                            let bj = unsafe { *ptr.add(j) };
                            if bj <= b' '
                                || bj >= 0x7f
                                || matches!(
                                    bj,
                                    b',' | b'\t' | b'|' | b'"' | b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'
                                )
                            {
                                if matches!(bj, b',' | b'\t' | b'|') {
                                    return parse_delimited_spans(data, bj, spans);
                                }
                                return None;
                            }
                            j += 1;
                        }
                        return None;
                    }
                    if matches!(b, b',' | b'\t' | b'|') {
                        return parse_delimited_spans(data, b, spans);
                    }
                    if b == b' ' && first.is_ascii_digit() && i >= 7 {
                        if let Some(c) = parse_clf_or_bracketed_log_spans(data, spans) {
                            return Some(c);
                        }
                    }
                    if b == b' ' || b > 0x7f || matches!(b, b'(' | b')' | b';' | b'[') {
                        let lim = len.min(i + 128);
                        for &bj in &data[i + 1..lim] {
                            if bj == b'\t' {
                                return parse_delimited_spans(data, b'\t', spans);
                            }
                            if bj == b',' {
                                if data.contains(&b'{') {
                                    return None;
                                }
                                return parse_delimited_spans(data, b',', spans).filter(|&c| c >= 3);
                            }
                            if bj < b' ' || matches!(bj, b'"' | b'{' | b'}' | 0x7f) {
                                break;
                            }
                        }
                    }
                    return None;
                }
                i += 1;
            }
        }
        return None;
    }
    parse_json_record_spans(data, pos, spans, overflow)
}

#[inline(always)]
fn delimited_template_delim(segments: &[&[u8]]) -> Option<u8> {
    let slots = segments.len().saturating_sub(1);
    if slots < 2 || !segments[0].is_empty() {
        return None;
    }
    if !matches!(segments[slots], b"\n" | b"\r\n") {
        return None;
    }
    let seg1 = segments[1];
    if seg1.len() != 1 {
        return None;
    }
    let d = seg1[0];
    if !matches!(d, b',' | b'\t' | b'|') {
        return None;
    }
    for &seg in &segments[2..slots] {
        if seg.len() != 1 || seg[0] != d {
            return None;
        }
    }
    Some(d)
}

#[inline(always)]
fn template_is_clean_logfmt(segments: &[&[u8]]) -> bool {
    let slots = segments.len().saturating_sub(1);
    if slots < 2 || !matches!(segments[slots], b"\n" | b"\r\n") {
        return false;
    }
    let seg0 = segments[0];
    if seg0.len() < 2
        || seg0.last() != Some(&b'=')
        || !seg0[..seg0.len() - 1].iter().all(|&b| is_logfmt_key_byte(b))
    {
        return false;
    }
    for &seg in &segments[1..slots] {
        if seg.len() < 3
            || seg[0] != b' '
            || seg.last() != Some(&b'=')
            || !seg[1..seg.len() - 1].iter().all(|&b| is_logfmt_key_byte(b))
        {
            return false;
        }
    }
    true
}

/// CSV / TSV unquoted fields may hold spaces, brackets and UTF-8; every matcher and mask must use
/// the same stop set (see `scan_delimited_value`).
#[inline(always)]
const fn delim_is_relaxed(delim: u8) -> bool {
    delim == b',' || delim == b'\t'
}

/// Structural bitmasks for the 64 bytes at `p` (bit `i` = byte `p[i]`):
/// - `q`: bytes that terminate a quoted value scan (`"`, `\`, `< 0x20`), exactly the
///   stop set of [`scan_string`];
/// - `u`: bytes that terminate an unquoted delimited value (`delim`, `"`, or outside
///   `0x21..=0x7e`), exactly the stop set of [`try_match_delimited_scalar`].
///
/// # Safety
/// `p..p + 64` must be readable.
#[inline(always)]
unsafe fn delimited_masks64(p: *const u8, delim: u8) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let v_quote = vdupq_n_u8(b'"');
        let v_slash = vdupq_n_u8(b'\\');
        let v_ctrl = vdupq_n_u8(0x20);
        let v_delim = vdupq_n_u8(delim);
        let (q, u) = if delim_is_relaxed(delim) {
            let v_7f = vdupq_n_u8(0x7f);
            macro_rules! lane_rel {
                ($k:expr) => {{
                    let c = vld1q_u8(p.add($k * 16));
                    let q_base = vorrq_u8(vceqq_u8(c, v_quote), vcltq_u8(c, v_ctrl));
                    let q = vorrq_u8(q_base, vceqq_u8(c, v_slash));
                    let u = vorrq_u8(q_base, vorrq_u8(vceqq_u8(c, v_delim), vceqq_u8(c, v_7f)));
                    (vandq_u8(q, w), vandq_u8(u, w))
                }};
            }
            let (q0, u0) = lane_rel!(0);
            let (q1, u1) = lane_rel!(1);
            let (q2, u2) = lane_rel!(2);
            let (q3, u3) = lane_rel!(3);
            (vpaddq_u8(vpaddq_u8(q0, q1), vpaddq_u8(q2, q3)), vpaddq_u8(vpaddq_u8(u0, u1), vpaddq_u8(u2, u3)))
        } else {
            let v_21 = vdupq_n_u8(0x21);
            let v_5e = vdupq_n_u8(0x5e);
            macro_rules! lane_st {
                ($k:expr) => {{
                    let c = vld1q_u8(p.add($k * 16));
                    let is_q = vceqq_u8(c, v_quote);
                    let q = vorrq_u8(is_q, vorrq_u8(vceqq_u8(c, v_slash), vcltq_u8(c, v_ctrl)));
                    let u = vorrq_u8(is_q, vorrq_u8(vceqq_u8(c, v_delim), vcgeq_u8(vsubq_u8(c, v_21), v_5e)));
                    (vandq_u8(q, w), vandq_u8(u, w))
                }};
            }
            let (q0, u0) = lane_st!(0);
            let (q1, u1) = lane_st!(1);
            let (q2, u2) = lane_st!(2);
            let (q3, u3) = lane_st!(3);
            (vpaddq_u8(vpaddq_u8(q0, q1), vpaddq_u8(q2, q3)), vpaddq_u8(vpaddq_u8(u0, u1), vpaddq_u8(u2, u3)))
        };
        let r = vreinterpretq_u64_u8(vpaddq_u8(q, u));
        (vgetq_lane_u64::<0>(r), vgetq_lane_u64::<1>(r))
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v_quote = _mm_set1_epi8(b'"' as i8);
        let v_slash = _mm_set1_epi8(b'\\' as i8);
        let v_1f = _mm_set1_epi8(0x1f);
        let v_delim = _mm_set1_epi8(delim as i8);
        let v_21 = _mm_set1_epi8(0x21);
        let v_5d = _mm_set1_epi8(0x5d);
        let relaxed = delim_is_relaxed(delim);
        let mut qm = 0u64;
        let mut um = 0u64;
        for k in 0..4 {
            let c = _mm_loadu_si128(p.add(k * 16).cast::<__m128i>());
            let is_q = _mm_cmpeq_epi8(c, v_quote);
            let ctrl = _mm_cmpeq_epi8(_mm_max_epu8(c, v_1f), v_1f);
            let q = _mm_or_si128(is_q, _mm_or_si128(_mm_cmpeq_epi8(c, v_slash), ctrl));
            let x = _mm_sub_epi8(c, v_21);
            let in_range = _mm_cmpeq_epi8(_mm_min_epu8(x, v_5d), x);
            let stop = _mm_or_si128(is_q, _mm_cmpeq_epi8(c, v_delim));
            let u_bits = if relaxed {
                let ctrl_or_del = _mm_or_si128(ctrl, _mm_cmpeq_epi8(c, _mm_set1_epi8(0x7f)));
                (_mm_movemask_epi8(_mm_or_si128(stop, ctrl_or_del)) as u32) & 0xffff
            } else {
                (_mm_movemask_epi8(stop) as u32 | !(_mm_movemask_epi8(in_range) as u32)) & 0xffff
            };
            qm |= (_mm_movemask_epi8(q) as u32 as u64) << (k * 16);
            um |= (u_bits as u64) << (k * 16);
        }
        (qm, um)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let mut qm = 0u64;
        let mut um = 0u64;
        for i in 0..64 {
            let b = unsafe { *p.add(i) };
            if b == b'"' || b == b'\\' || b < 0x20 {
                qm |= 1 << i;
            }
            let stop = if delim_is_relaxed(delim) {
                b == delim || b == b'"' || b < 0x20 || b == 0x7f
            } else {
                b == delim || b == b'"' || b.wrapping_sub(0x21) >= 0x5e
            };
            if stop {
                um |= 1 << i;
            }
        }
        (qm, um)
    }
}

#[inline(never)]
fn refill_delim_cache(
    data: &[u8],
    w: usize,
    delim: u8,
    cache: &mut (usize, u64, u64),
) -> bool {
    let base = w * 64;
    let len = data.len();
    if base + 64 <= len {
        let (q, u) = unsafe { delimited_masks64(data.as_ptr().add(base), delim) };
        *cache = (w, q, u);
        true
    } else if base < len {
        let rem = len - base;
        let mut tail = [0u8; 64];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(base), tail.as_mut_ptr(), rem);
            let (q, u) = delimited_masks64(tail.as_ptr(), delim);
            *cache = (w, q, u);
        }
        true
    } else {
        false
    }
}

#[inline(never)]
fn next_delim_bit_slow(
    data: &[u8],
    delim: u8,
    mut w: usize,
    is_q: bool,
    cache: &mut (usize, u64, u64),
) -> Option<usize> {
    for _ in 0..4 {
        w += 1;
        if !refill_delim_cache(data, w, delim, cache) {
            return None;
        }
        let mask = if is_q { cache.1 } else { cache.2 };
        if mask != 0 {
            return Some(w * 64 + mask.trailing_zeros() as usize);
        }
    }
    None
}

#[inline(always)]
fn next_delim_bit<const IS_Q: bool>(
    data: &[u8],
    delim: u8,
    from: usize,
    cache: &mut (usize, u64, u64),
) -> Option<usize> {
    let w = from >> 6;
    if cache.0 != w && !refill_delim_cache(data, w, delim, cache) {
        return None;
    }
    let mask = if IS_Q { cache.1 } else { cache.2 };
    let m = mask >> (from & 63);
    if m != 0 {
        return Some(from + m.trailing_zeros() as usize);
    }
    next_delim_bit_slow(data, delim, w, IS_Q, cache)
}

/// Structural bitmasks `(quote_mask, prim_stop_mask)` for 64 bytes of JSON at `p`:
/// - `quote_mask` bit `i` = 1 iff `p[i]` stops a JSON string scan (`"`, `\`, `< 0x20`).
/// - `prim_stop_mask` bit `i` = 1 iff `p[i]` stops a JSON primitive (`<= ','`, `'}'`, `']'`).
///
/// # Safety
/// `p..p + 64` must be readable.
#[inline(always)]
unsafe fn json_struct_masks64(p: *const u8) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let v_quote = vdupq_n_u8(b'"');
        let v_slash = vdupq_n_u8(b'\\');
        let v_ctrl = vdupq_n_u8(0x20);
        let v_dash = vdupq_n_u8(b'-');
        let v_rbrace = vdupq_n_u8(b'}');
        let v_rbrack = vdupq_n_u8(b']');
        macro_rules! lane {
            ($k:expr) => {{
                let c = vld1q_u8(p.add($k * 16));
                let q = vorrq_u8(
                    vceqq_u8(c, v_quote),
                    vorrq_u8(vceqq_u8(c, v_slash), vcltq_u8(c, v_ctrl)),
                );
                let pr = vorrq_u8(
                    vcltq_u8(c, v_dash),
                    vorrq_u8(vceqq_u8(c, v_rbrace), vceqq_u8(c, v_rbrack)),
                );
                (vandq_u8(q, w), vandq_u8(pr, w))
            }};
        }
        let (q0, p0) = lane!(0);
        let (q1, p1) = lane!(1);
        let (q2, p2) = lane!(2);
        let (q3, p3) = lane!(3);
        let q_half = vpaddq_u8(vpaddq_u8(q0, q1), vpaddq_u8(q2, q3));
        let p_half = vpaddq_u8(vpaddq_u8(p0, p1), vpaddq_u8(p2, p3));
        let both = vreinterpretq_u64_u8(vpaddq_u8(q_half, p_half));
        (vgetq_lane_u64::<0>(both), vgetq_lane_u64::<1>(both))
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v_quote = _mm_set1_epi8(b'"' as i8);
        let v_slash = _mm_set1_epi8(b'\\' as i8);
        let v_1f = _mm_set1_epi8(0x1f);
        let v_comma = _mm_set1_epi8(b',' as i8);
        let v_rbrace = _mm_set1_epi8(b'}' as i8);
        let v_rbrack = _mm_set1_epi8(b']' as i8);
        let mut qm = 0u64;
        let mut pm = 0u64;
        for k in 0..4 {
            let c = _mm_loadu_si128(p.add(k * 16).cast::<__m128i>());
            let ctrl = _mm_cmpeq_epi8(_mm_max_epu8(c, v_1f), v_1f);
            let q = _mm_or_si128(
                _mm_cmpeq_epi8(c, v_quote),
                _mm_or_si128(_mm_cmpeq_epi8(c, v_slash), ctrl),
            );
            let pr = _mm_or_si128(
                _mm_cmpeq_epi8(_mm_max_epu8(c, v_comma), v_comma),
                _mm_or_si128(_mm_cmpeq_epi8(c, v_rbrace), _mm_cmpeq_epi8(c, v_rbrack)),
            );
            qm |= (_mm_movemask_epi8(q) as u32 as u64) << (k * 16);
            pm |= (_mm_movemask_epi8(pr) as u32 as u64) << (k * 16);
        }
        (qm, pm)
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let mut qm = 0u64;
        let mut pm = 0u64;
        for i in 0..64 {
            let b = unsafe { *p.add(i) };
            if b == b'"' || b == b'\\' || b < 0x20 {
                qm |= 1 << i;
            }
            if b <= b',' || b == b'}' || b == b']' {
                pm |= 1 << i;
            }
        }
        (qm, pm)
    }
}

thread_local! {
    static ENCODE_ROW_POOL: std::cell::RefCell<Vec<Row<'static>>> =
        std::cell::RefCell::new(Vec::with_capacity(512));
}

#[inline(always)]
fn take_row_vec<'a>(est_rows: usize) -> Vec<Row<'a>> {
    let mut vec_static = ENCODE_ROW_POOL.with(|cell| core::mem::take(&mut *cell.borrow_mut()));
    vec_static.clear();
    vec_static.reserve(est_rows);
    unsafe { core::mem::transmute::<Vec<Row<'static>>, Vec<Row<'a>>>(vec_static) }
}

#[inline(always)]
fn return_row_vec<'a>(mut rows: Vec<Row<'a>>) {
    if rows.capacity() >= 64 {
        rows.clear();
        let vec_static = unsafe { core::mem::transmute::<Vec<Row<'a>>, Vec<Row<'static>>>(rows) };
        ENCODE_ROW_POOL.with(|cell| {
            let mut pool = cell.borrow_mut();
            if vec_static.capacity() > pool.capacity() {
                *pool = vec_static;
            }
        });
    }
}

