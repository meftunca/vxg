//! Numeric predicate support for [`super::filter_json_u64`].
//!
//! Scope: flat JSON records whose `"key":` value is an unsigned integer written as
//! plain decimal digits (no sign, no fraction, no exponent). The compressed-domain fast
//! path and the line-based fallback implement the same semantics.

/// Comparison applied to an unsigned integer field. `Range` is inclusive on both ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumCmp {
    Eq(u64),
    Ge(u64),
    Le(u64),
    Range(u64, u64),
}

impl NumCmp {
    #[inline(always)]
    pub fn test(self, v: u64) -> bool {
        match self {
            NumCmp::Eq(x) => v == x,
            NumCmp::Ge(x) => v >= x,
            NumCmp::Le(x) => v <= x,
            NumCmp::Range(lo, hi) => v >= lo && v <= hi,
        }
    }
}

/// A numeric filter on one JSON key. `marker` is the exact byte string `"key":`.
#[derive(Clone, Debug)]
pub(crate) struct NumFilter {
    pub marker: Vec<u8>,
    pub cmp: NumCmp,
}

impl NumFilter {
    pub(crate) fn new(key: &str, cmp: NumCmp) -> Self {
        let mut marker = Vec::with_capacity(key.len() + 3);
        marker.push(b'"');
        marker.extend_from_slice(key.as_bytes());
        marker.extend_from_slice(b"\":");
        Self { marker, cmp }
    }
}

/// Parse the unsigned integer that follows the first `"key":` in `line`.
/// Returns `None` if the key is absent or the value is not a plain unsigned integer.
pub(crate) fn line_u64_after_marker(line: &[u8], marker: &[u8]) -> Option<u64> {
    let pos = line.windows(marker.len()).position(|w| w == marker)?;
    let rest = &line[pos + marker.len()..];
    let digits = rest.iter().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || digits > 20 {
        return None;
    }
    if let Some(&next) = rest.get(digits) {
        if matches!(next, b'.' | b'e' | b'E') {
            return None;
        }
    }
    if digits > 1 && rest[0] == b'0' {
        return None;
    }
    let mut v: u64 = 0;
    for &d in &rest[..digits] {
        v = v.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
    }
    Some(v)
}

pub(crate) fn line_numeric_match(line: &[u8], f: &NumFilter) -> bool {
    line_u64_after_marker(line, &f.marker).is_some_and(|v| f.cmp.test(v))
}
