//! Exact base-10 signed fixed-point numbers ([`DecNum`]) and comparisons ([`DecCmp`])
//! for compressed-domain predicates, aggregations, and `ORDER BY` without IEEE-754 drift.

use core::cmp::Ordering;
use core::fmt;

use crate::query::NumCmp;

pub(crate) const POW10_I128: [i128; 20] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
    10_000_000_000,
    100_000_000_000,
    1_000_000_000_000,
    10_000_000_000_000,
    100_000_000_000_000,
    1_000_000_000_000_000,
    10_000_000_000_000_000,
    100_000_000_000_000_000,
    1_000_000_000_000_000_000,
    10_000_000_000_000_000_000,
];

/// Canonical exact base-10 signed decimal (`value = mantissa * 10^(-scale)`).
///
/// Normalized on construction so trailing fractional zeros are stripped (`19.50` -> `195 * 10^-1`,
/// `-0.00` -> `0 * 10^0`), making `PartialEq`, `Eq`, and `Ord` exact mathematical relations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DecNum {
    pub mantissa: i128,
    pub scale: u8,
}

impl DecNum {
    pub const ZERO: Self = Self {
        mantissa: 0,
        scale: 0,
    };

    #[inline]
    pub fn new(mut mantissa: i128, mut scale: u8) -> Self {
        if mantissa == 0 {
            return Self::ZERO;
        }
        while scale > 0 && mantissa % 10 == 0 {
            mantissa /= 10;
            scale -= 1;
        }
        Self { mantissa, scale }
    }

    #[inline]
    pub const fn from_u64(v: u64) -> Self {
        Self {
            mantissa: v as i128,
            scale: 0,
        }
    }

    #[inline]
    pub const fn from_i64(v: i64) -> Self {
        Self {
            mantissa: v as i128,
            scale: 0,
        }
    }

    /// Parse a canonical signed integer or fixed-point decimal (`-123`, `0`, `19.99`, `-0.05`)
    /// with integer magnitude `<= u64::MAX` and `0..=9` fractional digits.
    #[inline]
    pub fn parse(b: &[u8]) -> Option<Self> {
        if b.is_empty() {
            return None;
        }
        let (neg, rest) = if b[0] == b'-' {
            (true, &b[1..])
        } else {
            (false, b)
        };
        if rest.is_empty() {
            return None;
        }
        let dot_pos = rest.iter().position(|&c| c == b'.');
        let (int_bytes, frac_bytes) = match dot_pos {
            Some(p) => (&rest[..p], Some(&rest[p + 1..])),
            None => (rest, None),
        };
        if int_bytes.is_empty() || int_bytes.len() > 20 {
            return None;
        }
        if int_bytes.len() > 1 && int_bytes[0] == b'0' {
            return None;
        }
        let mut int_val = 0u64;
        for &d in int_bytes {
            if !d.is_ascii_digit() {
                return None;
            }
            int_val = int_val.checked_mul(10)?.checked_add(u64::from(d - b'0'))?;
        }
        let (scale, frac_val) = match frac_bytes {
            None => (0u8, 0u64),
            Some(fb) => {
                if fb.is_empty() || fb.len() > 9 {
                    return None;
                }
                let mut fv = 0u64;
                for &d in fb {
                    if !d.is_ascii_digit() {
                        return None;
                    }
                    fv = fv * 10 + u64::from(d - b'0');
                }
                (fb.len() as u8, fv)
            }
        };
        let abs_m = (int_val as i128) * POW10_I128[scale as usize] + (frac_val as i128);
        let signed_m = if neg { -abs_m } else { abs_m };
        Some(Self::new(signed_m, scale))
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        if self.scale == 0 {
            self.mantissa as f64
        } else {
            (self.mantissa as f64) / (POW10_I128[self.scale as usize] as f64)
        }
    }
}

impl Ord for DecNum {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        if self.scale == other.scale {
            return self.mantissa.cmp(&other.mantissa);
        }
        if self.scale < other.scale {
            let diff = (other.scale - self.scale) as usize;
            if let Some(p) = POW10_I128.get(diff) {
                if let Some(lhs) = self.mantissa.checked_mul(*p) {
                    return lhs.cmp(&other.mantissa);
                }
            }
            if self.mantissa >= 0 {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        } else {
            let diff = (self.scale - other.scale) as usize;
            if let Some(p) = POW10_I128.get(diff) {
                if let Some(rhs) = other.mantissa.checked_mul(*p) {
                    return self.mantissa.cmp(&rhs);
                }
            }
            if other.mantissa >= 0 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
    }
}

impl PartialOrd for DecNum {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Format an unnormalized `(mantissa, scale)` pair in base 10 (`scale` fractional digits).
pub fn format_scaled_i128(mantissa: i128, scale: u8) -> String {
    if scale == 0 {
        return mantissa.to_string();
    }
    let neg = mantissa < 0;
    let abs_u = mantissa.unsigned_abs();
    let p10 = POW10_I128[scale.min(19) as usize] as u128;
    let int_part = abs_u / p10;
    let frac_part = abs_u % p10;
    let sign = if neg { "-" } else { "" };
    format!(
        "{sign}{int_part}.{frac_part:0width$}",
        width = scale as usize
    )
}

impl fmt::Display for DecNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_scaled_i128(self.mantissa, self.scale))
    }
}

/// Exact comparison predicate over signed integers and fixed-point decimals ([`DecNum`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecCmp {
    Eq(DecNum),
    Ne(DecNum),
    Gt(DecNum),
    Ge(DecNum),
    Lt(DecNum),
    Le(DecNum),
    Range(DecNum, DecNum),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum U64DomainCmp {
    AlwaysFalse,
    AlwaysTrue,
    Cmp(NumCmp),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScaledResidualCmp {
    AlwaysFalse,
    AlwaysTrue,
    Interval { lo: u64, hi: u64, is_neq: bool },
}

/// Project `v >= d` (`is_ge = true`) or `v <= d` (`is_ge = false`) onto integer `m = v * 10^col_scale`.
#[inline]
fn bound_on_scaled_int(d: DecNum, col_scale: u8, strict: bool, is_lower: bool) -> Option<i128> {
    if d.scale <= col_scale {
        let mul = POW10_I128[(col_scale - d.scale) as usize];
        let exact = d.mantissa.checked_mul(mul)?;
        if strict {
            if is_lower {
                exact.checked_add(1)
            } else {
                exact.checked_sub(1)
            }
        } else {
            Some(exact)
        }
    } else {
        // `d` is normalized so `d.mantissa` is not divisible by `10^(d.scale - col_scale)`:
        // `q < d * 10^col_scale < q + 1`.
        let div = POW10_I128[(d.scale - col_scale).min(19) as usize];
        let q = d.mantissa.div_euclid(div);
        if is_lower {
            q.checked_add(1)
        } else {
            Some(q)
        }
    }
}

impl DecCmp {
    #[inline]
    pub fn test(self, v: DecNum) -> bool {
        match self {
            Self::Eq(x) => v == x,
            Self::Ne(x) => v != x,
            Self::Gt(x) => v > x,
            Self::Ge(x) => v >= x,
            Self::Lt(x) => v < x,
            Self::Le(x) => v <= x,
            Self::Range(lo, hi) => v >= lo && v <= hi,
        }
    }

    #[inline]
    pub fn overlaps_bounds(self, lo: DecNum, hi: DecNum) -> bool {
        match self {
            Self::Eq(x) => x >= lo && x <= hi,
            Self::Ne(x) => !(lo == x && hi == x),
            Self::Gt(x) => hi > x,
            Self::Ge(x) => hi >= x,
            Self::Lt(x) => lo < x,
            Self::Le(x) => lo <= x,
            Self::Range(a, b) => a <= b && hi >= a && lo <= b,
        }
    }

    #[inline]
    pub fn always_true_bounds(self, lo: DecNum, hi: DecNum) -> bool {
        match self {
            Self::Eq(x) => lo == x && hi == x,
            Self::Ne(x) => x < lo || x > hi,
            Self::Gt(x) => lo > x,
            Self::Ge(x) => lo >= x,
            Self::Lt(x) => hi < x,
            Self::Le(x) => hi <= x,
            Self::Range(a, b) => a <= b && lo >= a && hi <= b,
        }
    }

    /// Map `self` onto the `u64` integer domain (`col_scale == 0`, `[0..=u64::MAX]`).
    #[inline]
    pub(crate) fn to_u64_domain(self) -> U64DomainCmp {
        match self.to_scaled_i64_residual_i128(0, 0, u64::MAX) {
            ScaledResidualCmp::AlwaysFalse => U64DomainCmp::AlwaysFalse,
            ScaledResidualCmp::AlwaysTrue => U64DomainCmp::AlwaysTrue,
            ScaledResidualCmp::Interval { lo, hi, is_neq } => {
                if is_neq {
                    U64DomainCmp::Cmp(NumCmp::Ne(lo))
                } else if lo == hi {
                    U64DomainCmp::Cmp(NumCmp::Eq(lo))
                } else if lo == 0 {
                    U64DomainCmp::Cmp(NumCmp::Le(hi))
                } else if hi == u64::MAX {
                    U64DomainCmp::Cmp(NumCmp::Ge(lo))
                } else {
                    U64DomainCmp::Cmp(NumCmp::Range(lo, hi))
                }
            }
        }
    }

    /// Map `self` onto residual `r in [0..=max_r]` where `val = (min_base + r) * 10^(-col_scale)`.
    #[inline]
    pub(crate) fn to_scaled_i64_residual_i128(
        self,
        col_scale: u8,
        min_base: i128,
        max_r: u64,
    ) -> ScaledResidualCmp {
        let hi_base = min_base + i128::from(max_r);
        match self {
            Self::Eq(d) => {
                if d.scale > col_scale {
                    return ScaledResidualCmp::AlwaysFalse;
                }
                let mul = POW10_I128[(col_scale - d.scale) as usize];
                let Some(t) = d.mantissa.checked_mul(mul) else {
                    return ScaledResidualCmp::AlwaysFalse;
                };
                if t < min_base || t > hi_base {
                    ScaledResidualCmp::AlwaysFalse
                } else if max_r == 0 {
                    ScaledResidualCmp::AlwaysTrue
                } else {
                    let r = (t - min_base) as u64;
                    ScaledResidualCmp::Interval {
                        lo: r,
                        hi: r,
                        is_neq: false,
                    }
                }
            }
            Self::Ne(d) => {
                if d.scale > col_scale {
                    return ScaledResidualCmp::AlwaysTrue;
                }
                let mul = POW10_I128[(col_scale - d.scale) as usize];
                let Some(t) = d.mantissa.checked_mul(mul) else {
                    return ScaledResidualCmp::AlwaysTrue;
                };
                if t < min_base || t > hi_base {
                    ScaledResidualCmp::AlwaysTrue
                } else if max_r == 0 {
                    ScaledResidualCmp::AlwaysFalse
                } else {
                    let r = (t - min_base) as u64;
                    ScaledResidualCmp::Interval {
                        lo: r,
                        hi: r,
                        is_neq: true,
                    }
                }
            }
            Self::Gt(d) => Self::range_residual(
                bound_on_scaled_int(d, col_scale, true, true),
                Some(i128::MAX),
                min_base,
                hi_base,
                max_r,
            ),
            Self::Ge(d) => Self::range_residual(
                bound_on_scaled_int(d, col_scale, false, true),
                Some(i128::MAX),
                min_base,
                hi_base,
                max_r,
            ),
            Self::Lt(d) => Self::range_residual(
                Some(i128::MIN),
                bound_on_scaled_int(d, col_scale, true, false),
                min_base,
                hi_base,
                max_r,
            ),
            Self::Le(d) => Self::range_residual(
                Some(i128::MIN),
                bound_on_scaled_int(d, col_scale, false, false),
                min_base,
                hi_base,
                max_r,
            ),
            Self::Range(a, b) => {
                if a > b {
                    return ScaledResidualCmp::AlwaysFalse;
                }
                Self::range_residual(
                    bound_on_scaled_int(a, col_scale, false, true),
                    bound_on_scaled_int(b, col_scale, false, false),
                    min_base,
                    hi_base,
                    max_r,
                )
            }
        }
    }

    #[inline]
    fn range_residual(
        lo_opt: Option<i128>,
        hi_opt: Option<i128>,
        min_base: i128,
        hi_base: i128,
        max_r: u64,
    ) -> ScaledResidualCmp {
        let Some(lo_i) = lo_opt else {
            return ScaledResidualCmp::AlwaysFalse;
        };
        let Some(hi_i) = hi_opt else {
            return ScaledResidualCmp::AlwaysFalse;
        };
        if lo_i > hi_i || lo_i > hi_base || hi_i < min_base {
            return ScaledResidualCmp::AlwaysFalse;
        }
        if lo_i <= min_base && hi_i >= hi_base {
            return ScaledResidualCmp::AlwaysTrue;
        }
        let r_lo = if lo_i <= min_base {
            0
        } else {
            (lo_i - min_base) as u64
        };
        let r_hi = if hi_i >= hi_base {
            max_r
        } else {
            (hi_i - min_base) as u64
        };
        ScaledResidualCmp::Interval {
            lo: r_lo,
            hi: r_hi,
            is_neq: false,
        }
    }
}
