//! Column addressing, predicates, transformations, and exact single-line reference
//! semantics shared by the compressed-domain engine and per-line fallbacks.

use std::borrow::Cow;

use crate::query::NumCmp;

/// Identifies a structured column inside flat JSON/NDJSON objects or delimited (CSV/TSV/PSV) rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColTarget<'a> {
    /// First top-level `"key":` field inside a flat JSON object record.
    JsonKey(&'a str),
    /// Zero-based column `index` in a `delim`-separated record (`b','`, `b'\t'`, or `b'|'`).
    Delimited { delim: u8, index: usize },
}

/// Byte-level string predicate evaluated against the logical (unquoted/unescaped) string value of a column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrCmp<'a> {
    Eq(&'a [u8]),
    Ne(&'a [u8]),
    Prefix(&'a [u8]),
    Suffix(&'a [u8]),
    Contains(&'a [u8]),
}

impl StrCmp<'_> {
    #[inline]
    pub fn test(self, val: &[u8]) -> bool {
        match self {
            StrCmp::Eq(x) => val == x,
            StrCmp::Ne(x) => val != x,
            StrCmp::Prefix(p) => val.starts_with(p),
            StrCmp::Suffix(s) => val.ends_with(s),
            StrCmp::Contains(n) => crate::columns::contains_subslice(val, n),
        }
    }
}

impl<'a> StrCmp<'a> {
    #[inline]
    pub(crate) fn required_slice(self) -> &'a [u8] {
        match self {
            StrCmp::Eq(s) | StrCmp::Prefix(s) | StrCmp::Suffix(s) | StrCmp::Contains(s) => s,
            StrCmp::Ne(_) => &[],
        }
    }
}

/// Column predicate: either a string comparison or an unsigned-integer comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValuePred<'a> {
    Str(StrCmp<'a>),
    Num(NumCmp),
}

/// A column-addressed predicate (`WHERE col <op> val`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColPred<'a> {
    pub col: ColTarget<'a>,
    pub pred: ValuePred<'a>,
}

impl<'a> ColPred<'a> {
    #[inline]
    pub const fn json_str(key: &'a str, cmp: StrCmp<'a>) -> Self {
        Self {
            col: ColTarget::JsonKey(key),
            pred: ValuePred::Str(cmp),
        }
    }

    #[inline]
    pub const fn json_num(key: &'a str, cmp: NumCmp) -> Self {
        Self {
            col: ColTarget::JsonKey(key),
            pred: ValuePred::Num(cmp),
        }
    }

    #[inline]
    pub const fn delimited_str(delim: u8, index: usize, cmp: StrCmp<'a>) -> Self {
        Self {
            col: ColTarget::Delimited { delim, index },
            pred: ValuePred::Str(cmp),
        }
    }

    #[inline]
    pub const fn delimited_num(delim: u8, index: usize, cmp: NumCmp) -> Self {
        Self {
            col: ColTarget::Delimited { delim, index },
            pred: ValuePred::Num(cmp),
        }
    }
}

/// Mutation / mapping operation applied to a target column.
pub enum ColOp<'a> {
    /// Set the column to a logical string value (automatically JSON-quoted/escaped for `JsonKey`,
    /// or quoted when needed for `Delimited`).
    SetStr(&'a [u8]),
    /// Set the column's raw wire token directly (e.g. `b"0"`, `b"null"`, `b"true"`, `b"\"ok\""`).
    SetRaw(&'a [u8]),
    /// Transform logical string values `s -> f(s)`; non-string tokens (e.g. JSON `null`/`123`) are left unchanged.
    MapStr(&'a (dyn Fn(&[u8]) -> Vec<u8> + Sync)),
    /// Transform canonical unsigned integer values `v -> f(v)`; non-integer tokens are left unchanged.
    MapU64(&'a (dyn Fn(u64) -> u64 + Sync)),
}

#[inline]
pub(crate) fn parse_canonical_u64(b: &[u8]) -> Option<u64> {
    if b.is_empty() || b.len() > 20 || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if b.len() > 1 && b[0] == b'0' {
        return None;
    }
    let mut v = 0u64;
    for &d in b {
        v = v.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    Some(v)
}

#[inline]
fn hex4_val(b: &[u8]) -> Option<u16> {
    if b.len() < 4 {
        return None;
    }
    let mut v = 0u16;
    for &c in &b[..4] {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return None,
        };
        v = (v << 4) | u16::from(d);
    }
    Some(v)
}

/// Unescape a raw JSON string token `"..."` into its logical UTF-8/byte sequence.
/// Returns `None` if `raw_tok` is not a valid JSON string literal.
#[inline]
pub fn unescape_json_str(raw_tok: &[u8]) -> Option<Cow<'_, [u8]>> {
    if raw_tok.len() < 2 || raw_tok[0] != b'"' || raw_tok[raw_tok.len() - 1] != b'"' {
        return None;
    }
    let inner = &raw_tok[1..raw_tok.len() - 1];
    if !inner.iter().any(|&b| b == b'\\' || b == b'"' || b < 0x20) {
        return Some(Cow::Borrowed(inner));
    }
    let mut out = Vec::with_capacity(inner.len());
    let mut i = 0usize;
    while i < inner.len() {
        let b = inner[i];
        if b < 0x20 || b == b'"' {
            return None;
        }
        if b != b'\\' {
            out.push(b);
            i += 1;
            continue;
        }
        i += 1;
        let esc = *inner.get(i)?;
        i += 1;
        match esc {
            b'"' => out.push(b'"'),
            b'\\' => out.push(b'\\'),
            b'/' => out.push(b'/'),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'u' => {
                let code = hex4_val(inner.get(i..i + 4)?)?;
                i += 4;
                let ch = if (0xd800..=0xdbff).contains(&code) {
                    if inner.get(i..i + 2) != Some(b"\\u") {
                        return None;
                    }
                    let lo = hex4_val(inner.get(i + 2..i + 6)?)?;
                    if !(0xdc00..=0xdfff).contains(&lo) {
                        return None;
                    }
                    i += 6;
                    let cp = 0x10000 + (((u32::from(code) - 0xd800) << 10) | (u32::from(lo) - 0xdc00));
                    char::from_u32(cp)?
                } else {
                    char::from_u32(u32::from(code))?
                };
                let mut tmp = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
            }
            _ => return None,
        }
    }
    Some(Cow::Owned(out))
}

/// Encode `s` as a canonical JSON string literal `"..."` into `out`.
pub fn encode_json_str(s: &[u8], out: &mut Vec<u8>) {
    out.reserve(s.len() + 2);
    out.push(b'"');
    if !s.iter().any(|&b| b < 0x20 || b == b'"' || b == b'\\') {
        out.extend_from_slice(s);
        out.push(b'"');
        return;
    }
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in s {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                out.extend_from_slice(&[b'\\', b'u', b'0', b'0', HEX[(b >> 4) as usize], HEX[(b & 15) as usize]]);
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

#[inline]
pub fn decode_delim_str(raw_tok: &[u8]) -> Option<Cow<'_, [u8]>> {
    if raw_tok.len() >= 2 && raw_tok[0] == b'"' && raw_tok[raw_tok.len() - 1] == b'"' {
        unescape_json_str(raw_tok)
    } else {
        Some(Cow::Borrowed(raw_tok))
    }
}

pub fn encode_delim_str(s: &[u8], delim: u8, out: &mut Vec<u8>) {
    let relaxed = delim == b',' || delim == b'\t';
    let needs_quotes = s.is_empty()
        || s[0] <= b' '
        || s[0] >= 0x7f
        || s.iter().any(|&b| {
            b == delim
                || b == b'"'
                || b == 0x7f
                || if relaxed {
                    b < 0x20
                } else {
                    b <= b' ' || b >= 0x7f || matches!(b, b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']')
                }
        });
    if needs_quotes {
        encode_json_str(s, out);
    } else {
        out.extend_from_slice(s);
    }
}

#[inline]
pub fn format_col_str(col: ColTarget<'_>, s: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    match col {
        ColTarget::JsonKey(_) => encode_json_str(s, &mut out),
        ColTarget::Delimited { delim, .. } => encode_delim_str(s, delim, &mut out),
    }
    out
}

/// Evaluate `pred` on a raw column wire token `raw_tok`.
#[inline]
pub fn eval_token_pred(col: ColTarget<'_>, raw_tok: &[u8], pred: ValuePred<'_>) -> bool {
    match pred {
        ValuePred::Num(cmp) => parse_canonical_u64(raw_tok).is_some_and(|v| cmp.test(v)),
        ValuePred::Str(sc) => match col {
            ColTarget::JsonKey(_) => unescape_json_str(raw_tok).is_some_and(|s| sc.test(&s)),
            ColTarget::Delimited { .. } => decode_delim_str(raw_tok).is_some_and(|s| sc.test(&s)),
        },
    }
}

/// Apply `op` to a raw column wire token `raw_tok`. Returns `Some(new_raw_tok)` only if the
/// operation applies and produces a byte sequence different from `raw_tok`.
pub fn apply_token_op(col: ColTarget<'_>, raw_tok: &[u8], op: &ColOp<'_>) -> Option<Vec<u8>> {
    let next = match op {
        ColOp::SetStr(s) => format_col_str(col, s),
        ColOp::SetRaw(r) => r.to_vec(),
        ColOp::MapStr(f) => {
            let cur = match col {
                ColTarget::JsonKey(_) => unescape_json_str(raw_tok)?,
                ColTarget::Delimited { .. } => decode_delim_str(raw_tok)?,
            };
            let mapped = f(&cur);
            format_col_str(col, &mapped)
        }
        ColOp::MapU64(f) => {
            let v = parse_canonical_u64(raw_tok)?;
            let nv = f(v);
            if nv == v {
                return None;
            }
            nv.to_string().into_bytes()
        }
    };
    (next.as_slice() != raw_tok).then_some(next)
}

#[inline]
fn skip_json_ws(data: &[u8], pos: &mut usize) {
    while *pos < data.len() && matches!(data[*pos], b' ' | b'\t' | b'\r' | b'\n') {
        *pos += 1;
    }
}

#[inline]
pub(crate) fn scan_json_str_end(data: &[u8], start: usize) -> Option<usize> {
    if data.get(start) != Some(&b'"') {
        return None;
    }
    let mut i = start + 1;
    while i < data.len() {
        let b = data[i];
        if b == b'"' {
            return Some(i + 1);
        }
        if b < 0x20 {
            return None;
        }
        if b == b'\\' {
            i += 1;
            if i >= data.len() || data[i] < 0x20 {
                return None;
            }
        }
        i += 1;
    }
    None
}

pub(crate) fn scan_json_val_end(data: &[u8], start: usize) -> Option<usize> {
    let first = *data.get(start)?;
    match first {
        b'"' => scan_json_str_end(data, start),
        b'{' | b'[' => {
            let mut stack = vec![if first == b'{' { b'}' } else { b']' }];
            let mut i = start + 1;
            while i < data.len() {
                match data[i] {
                    b'"' => {
                        i = scan_json_str_end(data, i)?;
                        continue;
                    }
                    b'{' => stack.push(b'}'),
                    b'[' => stack.push(b']'),
                    b'}' | b']' => {
                        if stack.pop() != Some(data[i]) {
                            return None;
                        }
                        if stack.is_empty() {
                            return Some(i + 1);
                        }
                    }
                    b'\n' | b'\r' => return None,
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ if first <= b',' || first == b'}' || first == b']' => None,
        _ => {
            let mut i = start + 1;
            while i < data.len() {
                let b = data[i];
                if b <= b',' || b == b'}' || b == b']' {
                    break;
                }
                i += 1;
            }
            Some(i)
        }
    }
}

/// Locate the byte span `(start, end)` of the first top-level `"key":` value in a valid
/// top-level JSON object record `line`.
pub fn find_json_key_span(line: &[u8], key: &[u8]) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    skip_json_ws(line, &mut pos);
    if line.get(pos) != Some(&b'{') {
        return None;
    }
    pos += 1;
    let mut found: Option<(usize, usize)> = None;
    loop {
        skip_json_ws(line, &mut pos);
        if line.get(pos) == Some(&b'}') {
            pos += 1;
            break;
        }
        let k_start = pos;
        let k_end = scan_json_str_end(line, pos)?;
        pos = k_end;
        skip_json_ws(line, &mut pos);
        if line.get(pos) != Some(&b':') {
            return None;
        }
        pos += 1;
        skip_json_ws(line, &mut pos);
        let v_start = pos;
        let v_end = scan_json_val_end(line, pos)?;
        if found.is_none() {
            if let Some(k_decoded) = unescape_json_str(&line[k_start..k_end]) {
                if k_decoded.as_ref() == key {
                    found = Some((v_start, v_end));
                }
            }
        }
        pos = v_end;
        skip_json_ws(line, &mut pos);
        match line.get(pos) {
            Some(b',') => pos += 1,
            Some(b'}') => {
                pos += 1;
                break;
            }
            _ => return None,
        }
    }
    skip_json_ws(line, &mut pos);
    if line.get(pos) == Some(&b',') {
        pos += 1;
        skip_json_ws(line, &mut pos);
    }
    (pos == line.len()).then_some(found).flatten()
}

/// Locate the byte span `(start, end)` of zero-based field `index` in a `delim`-delimited record `line`
/// (requiring at least 2 fields and the same quoting/byte rules as `parse_delimited_spans`).
pub fn find_delimited_col_span(line: &[u8], delim: u8, index: usize) -> Option<(usize, usize)> {
    let mut body_len = line.len();
    if body_len > 0 && line[body_len - 1] == b'\n' {
        body_len -= 1;
        if body_len > 0 && line[body_len - 1] == b'\r' {
            body_len -= 1;
        }
    }
    if body_len == 0 {
        return None;
    }
    let body = &line[..body_len];
    let relaxed = delim == b',' || delim == b'\t';
    let mut pos = 0usize;
    let mut count = 0usize;
    let mut found = None;
    loop {
        if pos >= body_len {
            return None;
        }
        let first = body[pos];
        let end = if first == b'"' {
            scan_json_str_end(body, pos)?
        } else {
            if first == delim || first <= b' ' || first >= 0x7f {
                return None;
            }
            let mut e = pos + 1;
            while e < body_len {
                let b = body[e];
                if b == delim {
                    break;
                }
                if b == b'"' || (b < b' ' && b != b'\t') || b == 0x7f {
                    return None;
                }
                if !relaxed
                    && (b <= b' '
                        || b >= 0x7f
                        || matches!(b, b'(' | b')' | b';' | b'{' | b'}' | b'[' | b']'))
                {
                    return None;
                }
                e += 1;
            }
            e
        };
        if count == index {
            found = Some((pos, end));
        }
        count += 1;
        pos = end;
        if pos == body_len {
            break;
        }
        if body[pos] != delim {
            return None;
        }
        pos += 1;
    }
    (count >= 2).then_some(found).flatten()
}

#[inline]
pub fn find_line_col_span(line: &[u8], col: ColTarget<'_>) -> Option<(usize, usize)> {
    match col {
        ColTarget::JsonKey(k) => find_json_key_span(line, k.as_bytes()),
        ColTarget::Delimited { delim, index } => find_delimited_col_span(line, delim, index),
    }
}

#[inline]
pub fn line_matches_col_pred(line: &[u8], pred: &ColPred<'_>) -> bool {
    let Some((s, e)) = find_line_col_span(line, pred.col) else {
        return false;
    };
    eval_token_pred(pred.col, &line[s..e], pred.pred)
}

pub fn apply_line_col_op(
    line: &[u8],
    where_pred: Option<&ColPred<'_>>,
    target_col: ColTarget<'_>,
    op: &ColOp<'_>,
) -> Option<Vec<u8>> {
    if let Some(p) = where_pred {
        if !line_matches_col_pred(line, p) {
            return None;
        }
    }
    let (s, e) = find_line_col_span(line, target_col)?;
    let new_tok = apply_token_op(target_col, &line[s..e], op)?;
    let mut out = Vec::with_capacity(line.len() - (e - s) + new_tok.len());
    out.extend_from_slice(&line[..s]);
    out.extend_from_slice(&new_tok);
    out.extend_from_slice(&line[e..]);
    Some(out)
}
