//! Exact IEEE-754 `f16` / `bf16` <-> `f32` conversions and numerical error telemetry (`SQNR`,
//! cosine similarity, RMSE, max absolute error).

/// Converts an IEEE-754 binary16 (`f16`) raw `u16` bit pattern to `f32`.
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16 -> normalized f32
        let mut m = mant;
        let mut e = 127u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        f32::from_bits(sign | (e << 23) | (m << 13))
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Converts `f32` to IEEE-754 binary16 (`f16`) raw `u16` bits using round-to-nearest-even.
#[inline(always)]
pub fn f32_to_f16(val: f32) -> u16 {
    let b = val.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x007f_ffff;

    if exp == 255 {
        if mant != 0 {
            return sign | 0x7e00 | ((mant >> 13) as u16);
        }
        return sign | 0x7c00;
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    // Exponent -25 includes values above the midpoint between zero and
    // the smallest f16 subnormal; let nearest-even rounding decide them.
    if unbiased < -25 {
        return sign;
    }
    if unbiased < -14 {
        let shift = (-14 - unbiased) as u32;
        let full_mant = mant | 0x0080_0000;
        let round_shift = 13 + shift;
        let half = 1u32 << (round_shift - 1);
        let mask = (1u32 << round_shift) - 1;
        let rem = full_mant & mask;
        let mut m16 = (full_mant >> round_shift) as u16;
        if rem > half || (rem == half && (m16 & 1) != 0) {
            m16 += 1;
        }
        return sign | m16;
    }
    let e16 = ((unbiased + 15) as u16) << 10;
    let rem = mant & 0x1fff;
    let mut m16 = (mant >> 13) as u16;
    if rem > 0x1000 || (rem == 0x1000 && (m16 & 1) != 0) {
        m16 += 1;
        if m16 == 0x0400 {
            return sign | (e16 + 0x0400);
        }
    }
    sign | e16 | m16
}

/// Converts a `bfloat16` (`bf16`) raw `u16` bit pattern to `f32`.
#[inline(always)]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_subnormal_rounding_boundaries_both_signs() {
        let midpoint = 2.0f32.powi(-25);
        let cases = [
            (0.0, 0),
            (f32::from_bits(1), 0),
            (f32::from_bits(midpoint.to_bits() - 1), 0),
            (midpoint, 0),
            (f32::from_bits(midpoint.to_bits() + 1), 1),
            (0.75 * 2.0f32.powi(-24), 1),
            (2.0f32.powi(-24), 1),
            (3.0 * midpoint, 2),
            (5.0 * midpoint, 2),
            (f16_to_f32(0x03ff) + midpoint, 0x0400),
        ];
        for (value, expected) in cases {
            assert_eq!(f32_to_f16(value), expected, "{value:e}");
            assert_eq!(f32_to_f16(-value), 0x8000 | expected, "-{value:e}");
        }
    }
}

/// Converts `f32` to `bfloat16` (`bf16`) raw `u16` bits using round-to-nearest-even.
#[inline(always)]
pub fn f32_to_bf16(val: f32) -> u16 {
    let b = val.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return ((b >> 16) as u16) | 0x0040;
    }
    let lsb = (b >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    (b.wrapping_add(rounding_bias) >> 16) as u16
}

/// Numerical fidelity telemetry comparing an approximation `approx` against reference `reference`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ErrorMetrics {
    /// Signal-to-Quantization-Noise Ratio in decibels (`10 * log10(sum(r^2) / sum((r-a)^2))`).
    pub sqnr_db: f64,
    /// Cosine similarity in `[-1.0, 1.0]`.
    pub cosine_sim: f64,
    /// Root-Mean-Square Error.
    pub rmse: f64,
    /// Maximum absolute elementwise difference `max_i |r_i - a_i|`.
    pub max_abs_err: f32,
}

impl ErrorMetrics {
    pub fn compute(reference: &[f32], approx: &[f32]) -> Self {
        assert_eq!(reference.len(), approx.len());
        if reference.is_empty() {
            return Self {
                sqnr_db: f64::INFINITY,
                cosine_sim: 1.0,
                rmse: 0.0,
                max_abs_err: 0.0,
            };
        }
        let mut sig_pow = 0.0f64;
        let mut app_pow = 0.0f64;
        let mut dot = 0.0f64;
        let mut err_pow = 0.0f64;
        let mut max_abs = 0.0f32;

        for (&r, &a) in reference.iter().zip(approx.iter()) {
            let rf = r as f64;
            let af = a as f64;
            let diff = rf - af;
            sig_pow += rf * rf;
            app_pow += af * af;
            dot += rf * af;
            err_pow += diff * diff;
            let ad = (r - a).abs();
            if ad > max_abs {
                max_abs = ad;
            }
        }

        let sqnr_db = if err_pow == 0.0 {
            f64::INFINITY
        } else if sig_pow == 0.0 {
            0.0
        } else {
            10.0 * (sig_pow / err_pow).log10()
        };
        let denom = (sig_pow * app_pow).sqrt();
        let cosine_sim = if denom == 0.0 {
            1.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        };
        let rmse = (err_pow / reference.len() as f64).sqrt();

        Self {
            sqnr_db,
            cosine_sim,
            rmse,
            max_abs_err: max_abs,
        }
    }
}

/// Fits a degree-3 polynomial `P(u) = (a0 + a2*u^2) + u*(a1 + a3*u^2)` where
/// `u = (q as f32) - 0.5*(k - 1)` to `emp[0..k]` (`k = 8` or `16`), returning `([a0, a1, a2, a3], centroids)`.
/// Using exact IEEE-754 `mul_add` ensures CPU register LUTs and GPU `float4` FMA evaluation match to 0 ULP!
pub fn fit_cubic_codebook(emp: &[f32]) -> ([f32; 4], [f32; 16]) {
    let k = emp.len();
    let half_span = 0.5 * ((k.saturating_sub(1)) as f64);
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];

    for (q, &y_f32) in emp.iter().enumerate() {
        let u = (q as f64) - half_span;
        let basis = [1.0, u, u * u, u * u * u];
        let y = y_f32 as f64;
        for r in 0..4 {
            atb[r] += basis[r] * y;
            for c in 0..4 {
                ata[r][c] += basis[r] * basis[c];
            }
        }
    }

    // 4x4 Gaussian elimination with partial pivoting
    let mut aug = [[0.0f64; 5]; 4];
    for r in 0..4 {
        for c in 0..4 {
            aug[r][c] = ata[r][c];
        }
        aug[r][4] = atb[r];
    }
    for col in 0..4 {
        let mut piv = col;
        for r in (col + 1)..4 {
            if aug[r][col].abs() > aug[piv][col].abs() {
                piv = r;
            }
        }
        aug.swap(col, piv);
        let diag = aug[col][col];
        if diag.abs() > 1e-18 {
            for c in col..5 {
                aug[col][c] /= diag;
            }
            for r in 0..4 {
                if r != col {
                    let f = aug[r][col];
                    for c in col..5 {
                        aug[r][c] -= f * aug[col][c];
                    }
                }
            }
        }
    }

    let poly = [
        aug[0][4] as f32,
        aug[1][4] as f32,
        aug[2][4] as f32,
        aug[3][4] as f32,
    ];
    let [a0, a1, a2, a3] = poly;
    let center = 0.5 * ((k.saturating_sub(1)) as f32);
    let mut cents = [0.0f32; 16];
    for (q, slot) in cents.iter_mut().take(k).enumerate() {
        let u = (q as f32) - center;
        let u2 = u * u;
        *slot = (a3.mul_add(u2, a1)).mul_add(u, a2.mul_add(u2, a0));
    }
    (poly, cents)
}
