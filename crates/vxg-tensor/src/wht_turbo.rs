//! Outlier-Free Sign-Flipped Walsh-Hadamard Vector Quantization Codec (`VXT-Turbo`):
//! - Applies deterministic pseudo-random `+-1` column sign flips (`D`) + blockwise `FWHT_32` (`H_32`)
//!   so heavy-tailed neural channels become i.i.d. isotropic Gaussian vectors (`0` sparse outliers!).
//! - `Turbo2` (`bits = 2`, `2.125 bpw`, `~7.53x` vs 16-bit): `Orth4D-16` (`~10.03 dB` SQNR).
//! - `Turbo2.5` (`bits = 5`, `2.625 bpw`, `~6.10x` vs 16-bit): `Orth4D-64` (`~12.98 dB` SQNR).
//! - `Turbo3` (`bits = 3`, `3.125 bpw`, `~5.12x` vs 16-bit): `Orth2D-16` (`~15.36 dB` SQNR).
//! - `EC-Turbo3` (`bits = 4`, `~2.82 bpw` on disk, `3.125 bpw` on GPU): Entropy-Constrained
//!   `EC-Orth2D-16` with `(0, 0)` origin deadzone (`~14.97 dB` SQNR) + Canonical Huffman container.

use crate::fp16::f16_to_f32;
use crate::wht2::fwht32_inplace;
use crate::wht_turbo_enc::encode_turbo_planes;
use crate::wht_turbo_gptq::encode_turbo_gptq_planes;
use crate::wht_turbo_tables::{
    ec_orth2d_tables, orth2d_tables, orth4d64_tables, orth4d_tables, unpack_t25_lane,
};
pub use crate::wht_turbo_tables::{EC_ORTH2D_16_F16, ORTH2D_16_F16, ORTH4D_16_F16, ORTH4D_64_F16};
use crate::{Error, Result};

/// Deterministic 32-bit sign bit mask (`0x0000_0000` = `+1.0`, `0x8000_0000` = `-1.0`) matching MSL!
#[inline(always)]
pub fn turbo_col_sign_mask(col: u32) -> u32 {
    let mut h = col ^ 0x9e37_79b9;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    (h & 1) << 31
}

#[inline(always)]
pub(crate) fn apply_col_sign(v: f32, col: usize) -> f32 {
    if v == 0.0 {
        0.0
    } else {
        f32::from_bits(v.to_bits() ^ turbo_col_sign_mask(col as u32))
    }
}

/// Applies column sign flips + blockwise `FWHT_32` to `x` (`cols` -> `padded_cols`).
pub fn turbo_fwht32_padded(x: &[f32], padded_cols: usize) -> Vec<f32> {
    debug_assert_eq!(padded_cols % 32, 0);
    let mut out = vec![0.0f32; padded_cols];
    for (c, &v) in x.iter().enumerate() {
        out[c] = apply_col_sign(v, c);
    }
    for blk in out.chunks_exact_mut(32) {
        let arr: &mut [f32; 32] = blk.try_into().unwrap();
        fwht32_inplace(arr);
    }
    out
}

/// Outlier-free `4-row`-tiled Sign-Flipped Hadamard VQ compressed matrix (`Turbo2`, `Turbo2.5`,
/// `Turbo3`, and `EC-Turbo3`).
#[derive(Clone, Debug)]
pub struct TurboWhtMatrix {
    pub bits: u8,
    pub rows: usize,
    pub cols: usize,
    pub padded_rows: usize,
    pub padded_cols: usize,
    pub disk_bytes: usize,
    pub scales_f16: Vec<u16>,
    pub packed_words: Vec<u32>,
}

impl TurboWhtMatrix {
    #[inline]
    pub fn words_per_tile_step(bits: u8) -> usize {
        match bits {
            2 => 64,
            5 => 80,
            _ => 96,
        }
    }

    pub fn encode(bits: u8, rows: usize, cols: usize, weights: &[f32]) -> Result<Self> {
        let default_lam = match bits {
            4 => 0.050,
            5 => 0.060,
            _ => 0.0,
        };
        Self::encode_with_lambda(bits, rows, cols, weights, |_| default_lam)
    }

    pub fn encode_with_lambda<F>(
        bits: u8,
        rows: usize,
        cols: usize,
        weights: &[f32],
        row_lambda: F,
    ) -> Result<Self>
    where
        F: Fn(usize) -> f32 + Sync,
    {
        Self::encode_with_hessian(bits, rows, cols, weights, None, row_lambda)
    }

    pub fn encode_with_hessian<F>(
        bits: u8,
        rows: usize,
        cols: usize,
        weights: &[f32],
        h_rot: Option<&[f32]>,
        row_lambda: F,
    ) -> Result<Self>
    where
        F: Fn(usize) -> f32 + Sync,
    {
        Self::encode_with_ch_amp(bits, rows, cols, weights, h_rot, None, row_lambda)
    }

    pub fn encode_with_ch_amp<F>(
        bits: u8,
        rows: usize,
        cols: usize,
        weights: &[f32],
        h_rot: Option<&[f32]>,
        ch_amp: Option<&[f32]>,
        row_lambda: F,
    ) -> Result<Self>
    where
        F: Fn(usize) -> f32 + Sync,
    {
        if !matches!(bits, 2 | 3 | 4 | 5) {
            return Err(Error::Format(
                "TurboWhtMatrix bits must be 2, 3, 4 (EC-Turbo3), or 5 (Turbo2.5)".into(),
            ));
        }
        if rows == 0 || cols == 0 || weights.len() != rows.saturating_mul(cols) {
            return Err(Error::InvalidShape("invalid rows*cols for TurboWhtMatrix"));
        }
        let padded_rows = rows.div_ceil(4) * 4;
        let padded_cols = cols.div_ceil(256) * 256;
        let h_checked = h_rot.filter(|h| h.len() == padded_cols);
        let amp_checked = ch_amp.filter(|a| a.len() == cols);
        let wps = Self::words_per_tile_step(bits);
        let (scales_f16, packed_words) = if let Some(amp) = amp_checked.filter(|_| matches!(bits, 3 | 4)) {
            encode_turbo_gptq_planes(
                bits,
                rows,
                cols,
                padded_cols,
                wps,
                weights,
                h_checked,
                amp,
                row_lambda,
            )
        } else {
            encode_turbo_planes(
                bits,
                rows,
                cols,
                padded_cols,
                wps,
                weights,
                h_checked,
                row_lambda,
            )
        };
        Ok(Self {
            bits,
            rows,
            cols,
            padded_rows,
            padded_cols,
            disk_bytes: 0,
            scales_f16,
            packed_words,
        })
    }

    pub fn raw_bytes(&self) -> usize {
        self.scales_f16.len() * 2 + self.packed_words.len() * 4
    }

    pub fn compressed_bytes(&self) -> usize {
        if self.disk_bytes > 0 {
            self.disk_bytes
        } else {
            self.raw_bytes()
        }
    }

    pub fn bits_per_weight(&self) -> f64 {
        (self.compressed_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Concatenates multiple compatible (`rows % 4 == 0`, matching `bits` & `cols`) matrices vertically.
    pub fn concat_vertical(parts: &[&Self]) -> Option<Self> {
        let first = *parts.first()?;
        if parts.len() < 2 {
            return None;
        }
        let mut total_rows = 0usize;
        for (i, m) in parts.iter().enumerate() {
            if m.bits != first.bits || m.cols != first.cols || m.padded_cols != first.padded_cols {
                return None;
            }
            if i + 1 < parts.len() && m.rows % 4 != 0 {
                return None;
            }
            total_rows += m.rows;
        }
        let padded_rows = parts.iter().map(|m| m.padded_rows).sum();
        let disk_bytes = if parts.iter().all(|m| m.disk_bytes > 0) {
            parts.iter().map(|m| m.disk_bytes).sum()
        } else {
            0
        };
        let mut scales_f16 = Vec::with_capacity(parts.iter().map(|m| m.scales_f16.len()).sum());
        let mut packed_words = Vec::with_capacity(parts.iter().map(|m| m.packed_words.len()).sum());
        for m in parts {
            scales_f16.extend_from_slice(&m.scales_f16);
            packed_words.extend_from_slice(&m.packed_words);
        }
        Some(Self {
            bits: first.bits,
            rows: total_rows,
            cols: first.cols,
            padded_rows,
            padded_cols: first.padded_cols,
            disk_bytes,
            scales_f16,
            packed_words,
        })
    }

    pub fn decode_row(&self, r: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.cols);
        let tile = r / 4;
        let sr = r % 4;
        let steps = self.padded_cols / 256;
        let wps = Self::words_per_tile_step(self.bits);
        let sc_tile = &self.scales_f16[tile * steps * 8..(tile + 1) * steps * 8];
        let pw_tile = &self.packed_words[tile * steps * wps..(tile + 1) * steps * wps];
        let mut rbuf = vec![0.0f32; self.padded_cols];

        if self.bits == 2 {
            let (_, _, lut256) = orth4d_tables();
            for step in 0..steps {
                let sc0 = f16_to_f32(sc_tile[step * 8 + sr]);
                let sc1 = f16_to_f32(sc_tile[step * 8 + 4 + sr]);
                let p64 = &pw_tile[step * 64..(step + 1) * 64];
                for lane in 0..32 {
                    let sc = if lane < 16 { sc0 } else { sc1 };
                    let w = p64[lane * 2 + (sr >> 1)];
                    let c16 = ((w >> ((sr & 1) * 16)) & 0xffff) as usize;
                    let v0 = lut256[c16 & 0xff];
                    let v1 = lut256[c16 >> 8];
                    let base = step * 256 + lane * 8;
                    for k in 0..4 {
                        rbuf[base + k] = sc * v0[k];
                        rbuf[base + 4 + k] = sc * v1[k];
                    }
                }
            }
        } else if self.bits == 5 {
            let (_, _, lut1024) = orth4d64_tables();
            for step in 0..steps {
                let sc0 = f16_to_f32(sc_tile[step * 8 + sr]);
                let sc1 = f16_to_f32(sc_tile[step * 8 + 4 + sr]);
                let p80 = &pw_tile[step * 80..(step + 1) * 80];
                for lane in 0..32 {
                    let sc = if lane < 16 { sc0 } else { sc1 };
                    let [c_lo, c_hi] = unpack_t25_lane(p80, lane)[sr];
                    let v0 = lut1024[c_lo];
                    let v1 = lut1024[c_hi];
                    let base = step * 256 + lane * 8;
                    for k in 0..4 {
                        rbuf[base + k] = sc * v0[k];
                        rbuf[base + 4 + k] = sc * v1[k];
                    }
                }
            }
        } else {
            let (_, _, lut64) = if self.bits == 4 {
                ec_orth2d_tables()
            } else {
                orth2d_tables()
            };
            for step in 0..steps {
                let sc0 = f16_to_f32(sc_tile[step * 8 + sr]);
                let sc1 = f16_to_f32(sc_tile[step * 8 + 4 + sr]);
                let p96 = &pw_tile[step * 96..(step + 1) * 96];
                for lane in 0..32 {
                    let sc = if lane < 16 { sc0 } else { sc1 };
                    let iw = p96[lane * 2 + (sr >> 1)];
                    let idx16 = ((iw >> ((sr & 1) * 16)) & 0xffff) as usize;
                    let sgn8 = ((p96[64 + lane] >> (sr * 8)) & 0xff) as usize;
                    let base = step * 256 + lane * 8;
                    for p in 0..4 {
                        let code6 = ((idx16 >> (p * 4)) & 0x0f) | (((sgn8 >> (p * 2)) & 3) << 4);
                        let v2 = lut64[code6];
                        rbuf[base + p * 2] = sc * v2[0];
                        rbuf[base + p * 2 + 1] = sc * v2[1];
                    }
                }
            }
        }

        for blk in rbuf.chunks_exact_mut(32) {
            let arr: &mut [f32; 32] = blk.try_into().unwrap();
            fwht32_inplace(arr);
        }
        for (c, slot) in dst.iter_mut().enumerate() {
            *slot = apply_col_sign(rbuf[c], c);
        }
    }

    pub fn gemv_cpu(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        let x_rot = turbo_fwht32_padded(x, self.padded_cols);
        let row_tiles = self.padded_rows / 4;
        let workers = threads.clamp(1, 256).min(row_tiles.max(1));

        let (_, _, lut256) = orth4d_tables();
        let (_, _, lut1024) = orth4d64_tables();
        let (_, _, lut64) = if self.bits == 4 {
            ec_orth2d_tables()
        } else {
            orth2d_tables()
        };

        let run_tiles = |t0: usize, y_slice: &mut [f32]| {
            let steps = self.padded_cols / 256;
            let wps = Self::words_per_tile_step(self.bits);
            for (dt, y4) in y_slice.chunks_mut(4).enumerate() {
                let tile = t0 + dt;
                let sc_tile = &self.scales_f16[tile * steps * 8..(tile + 1) * steps * 8];
                let pw_tile = &self.packed_words[tile * steps * wps..(tile + 1) * steps * wps];
                let mut acc = [0.0f32; 4];

                if self.bits == 2 {
                    for step in 0..steps {
                        let p64 = &pw_tile[step * 64..(step + 1) * 64];
                        for gh in 0..2 {
                            let sc = [
                                f16_to_f32(sc_tile[step * 8 + gh * 4]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 1]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 2]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 3]),
                            ];
                            for sub in 0..16 {
                                let lane = gh * 16 + sub;
                                let xv = &x_rot[step * 256 + lane * 8..step * 256 + lane * 8 + 8];
                                let (w0, w1) = (p64[lane * 2], p64[lane * 2 + 1]);
                                let c16s = [
                                    (w0 & 0xffff) as usize,
                                    (w0 >> 16) as usize,
                                    (w1 & 0xffff) as usize,
                                    (w1 >> 16) as usize,
                                ];
                                for sr in 0..4 {
                                    let (v0, v1) = (lut256[c16s[sr] & 0xff], lut256[c16s[sr] >> 8]);
                                    let d = v0[0].mul_add(xv[0], v0[1].mul_add(xv[1], v0[2].mul_add(xv[2], v0[3] * xv[3])))
                                        + v1[0].mul_add(xv[4], v1[1].mul_add(xv[5], v1[2].mul_add(xv[6], v1[3] * xv[7])));
                                    acc[sr] = sc[sr].mul_add(d, acc[sr]);
                                }
                            }
                        }
                    }
                } else if self.bits == 5 {
                    for step in 0..steps {
                        let p80 = &pw_tile[step * 80..(step + 1) * 80];
                        for gh in 0..2 {
                            let sc = [
                                f16_to_f32(sc_tile[step * 8 + gh * 4]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 1]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 2]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 3]),
                            ];
                            for sub in 0..16 {
                                let lane = gh * 16 + sub;
                                let xv = &x_rot[step * 256 + lane * 8..step * 256 + lane * 8 + 8];
                                let cc = unpack_t25_lane(p80, lane);
                                for sr in 0..4 {
                                    let (v0, v1) = (lut1024[cc[sr][0]], lut1024[cc[sr][1]]);
                                    let d = v0[0].mul_add(xv[0], v0[1].mul_add(xv[1], v0[2].mul_add(xv[2], v0[3] * xv[3])))
                                        + v1[0].mul_add(xv[4], v1[1].mul_add(xv[5], v1[2].mul_add(xv[6], v1[3] * xv[7])));
                                    acc[sr] = sc[sr].mul_add(d, acc[sr]);
                                }
                            }
                        }
                    }
                } else {
                    for step in 0..steps {
                        let p96 = &pw_tile[step * 96..(step + 1) * 96];
                        for gh in 0..2 {
                            let sc = [
                                f16_to_f32(sc_tile[step * 8 + gh * 4]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 1]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 2]),
                                f16_to_f32(sc_tile[step * 8 + gh * 4 + 3]),
                            ];
                            for sub in 0..16 {
                                let lane = gh * 16 + sub;
                                let xv = &x_rot[step * 256 + lane * 8..step * 256 + lane * 8 + 8];
                                let (w0, w1, sw) = (p96[lane * 2], p96[lane * 2 + 1], p96[64 + lane]);
                                let i16s = [
                                    (w0 & 0xffff) as usize,
                                    (w0 >> 16) as usize,
                                    (w1 & 0xffff) as usize,
                                    (w1 >> 16) as usize,
                                ];
                                for sr in 0..4 {
                                    let idx16 = i16s[sr];
                                    let sgn8 = ((sw >> (sr * 8)) & 0xff) as usize;
                                    let mut d = 0.0f32;
                                    for p in 0..4 {
                                        let c6 = ((idx16 >> (p * 4)) & 0x0f) | (((sgn8 >> (p * 2)) & 3) << 4);
                                        let v2 = lut64[c6];
                                        d = v2[0].mul_add(xv[p * 2], v2[1].mul_add(xv[p * 2 + 1], d));
                                    }
                                    acc[sr] = sc[sr].mul_add(d, acc[sr]);
                                }
                            }
                        }
                    }
                }
                for (sr, slot) in y4.iter_mut().enumerate() {
                    *slot = acc[sr];
                }
            }
        };

        if workers <= 1 || row_tiles < 8 {
            run_tiles(0, y);
        } else {
            let tpw = row_tiles.div_ceil(workers);
            std::thread::scope(|s| {
                for (ci, yc) in y.chunks_mut(tpw * 4).enumerate() {
                    s.spawn(move || run_tiles(ci * tpw, yc));
                }
            });
        }
    }
}
