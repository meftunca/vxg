//! Fast Walsh-Hadamard 2-Bit Incoherence Codec (`VXT-WHT2+Outlier`, `~2.40 bpw`, `~6.7x` vs 16-bit):
//! Decomposes `W = W_sparse + R_bulk * H_32` where super-weight outliers are extracted into an
//! exact `FP16` CSR stream BEFORE blockwise `32x32` Walsh-Hadamard rotation (`H^T = H`, `H^2 = I`),
//! turning heavy-tailed neural weights into i.i.d. Gaussian coordinates quantized with an exact
//! 4-centroid (`2`-bit) cubic Lloyd-Max codebook.

use crate::fp16::{f16_to_f32, f32_to_f16, fit_cubic_codebook};
use crate::{Error, Result};

const INV_SQRT32: f32 = 0.176_776_695_296_636_89; // 1.0 / sqrt(32.0)

/// In-place normalized orthogonal 32-point Fast Walsh-Hadamard Transform (`H = H^T`, `H * H = I`).
#[inline]
pub fn fwht32_inplace(x: &mut [f32; 32]) {
    let mut len = 1usize;
    while len < 32 {
        let step = len << 1;
        let mut i = 0usize;
        while i < 32 {
            for j in 0..len {
                let u = x[i + j];
                let v = x[i + j + len];
                x[i + j] = u + v;
                x[i + j + len] = u - v;
            }
            i += step;
        }
        len = step;
    }
    for v in x.iter_mut() {
        *v *= INV_SQRT32;
    }
}

/// Zero-pads `x` (`cols` elements) to `padded_cols` (multiple of 32) and applies blockwise `FWHT_32`.
pub fn fwht32_padded(x: &[f32], padded_cols: usize) -> Vec<f32> {
    debug_assert_eq!(padded_cols % 32, 0);
    let mut out = vec![0.0f32; padded_cols];
    out[..x.len()].copy_from_slice(x);
    for blk in out.chunks_exact_mut(32) {
        let arr: &mut [f32; 32] = blk.try_into().unwrap();
        fwht32_inplace(arr);
    }
    out
}

/// Compressed-domain Walsh-Hadamard rotated `2`-bit Lloyd-Max + sparse `FP16` CSR matrix.
#[derive(Clone, Debug)]
pub struct Wht2Matrix {
    pub rows: usize,
    pub cols: usize,
    pub padded_cols: usize,
    /// `4` learned centroids stored in `[0..4]` (remaining `12` zeroed for uniform struct ABI).
    pub centroids: [f32; 16],
    /// `1` `f16` scale per 128-weight rotated group (`rows * (padded_cols / 128)` entries).
    pub scales_f16: Vec<u16>,
    /// Coalesced `u32` words (`16` `u32`s per 256-weight superblock = `8` `u32`s per 128-group).
    pub packed_words: Vec<u32>,
    /// CSR row offsets (`rows + 1` entries) into `outliers_packed` (in unrotated column space).
    pub outlier_row_ptr: Vec<u32>,
    /// Packed `(col_idx as u32) | ((val_f16 as u32) << 16)` when `cols <= 65536`.
    pub outliers_packed: Vec<u32>,
}

impl Wht2Matrix {
    pub fn encode(rows: usize, cols: usize, weights: &[f32]) -> Result<Self> {
        if rows == 0 || cols == 0 || weights.len() != rows.saturating_mul(cols) {
            return Err(Error::InvalidShape(
                "invalid rows*cols for Wht2Matrix::encode",
            ));
        }
        let padded_cols = cols.div_ceil(256) * 256;
        let groups_per_row = padded_cols / 128;
        let total_groups = rows * groups_per_row;
        let total_numel = rows * cols;
        let allow_outliers = cols <= 65536;

        // Pass 1: Estimate raw 128-group trimmed RMS to gate pre-rotation super-weight spikes
        let mut raw_rms = vec![0.0f32; total_groups];
        for r in 0..rows {
            let row_w = &weights[r * cols..(r + 1) * cols];
            for g in 0..groups_per_row {
                let c0 = g * 128;
                let valid = (cols.saturating_sub(c0)).min(128);
                if valid == 0 {
                    continue;
                }
                let s = &row_w[c0..c0 + valid];
                let mut sq = 0.0f32;
                for &v in s {
                    sq = v.mul_add(v, sq);
                }
                let r0 = (sq / (valid as f32)).sqrt();
                if r0 == 0.0 {
                    continue;
                }
                let cut = 2.8f32 * r0;
                let (mut csq, mut cn) = (0.0f32, 0usize);
                for &v in s {
                    if v.abs() <= cut {
                        csq = v.mul_add(v, csq);
                        cn += 1;
                    }
                }
                raw_rms[r * groups_per_row + g] = if cn >= 8 {
                    (csq / (cn as f32)).sqrt().max(r0 * 0.15)
                } else {
                    r0
                };
            }
        }

        // Calibrate global pre-rotation outlier threshold (~0.85% = 0.27 bpw -> total ~2.40 bpw)
        let sample_target = 65_536usize;
        let mut stride = (total_numel / sample_target).max(1);
        while stride > 1 && gcd_usize(stride, cols) != 1 {
            stride += 1;
        }
        let out_thresh = if allow_outliers {
            let mut score_samp = Vec::with_capacity(sample_target.min(total_numel) + 128);
            let mut ic = 0usize;
            while ic < total_numel {
                let r = ic / cols;
                let c = ic % cols;
                let sc = raw_rms[r * groups_per_row + (c / 128)];
                score_samp.push(if sc > 0.0 {
                    weights[ic].abs() / sc.sqrt()
                } else {
                    0.0
                });
                ic += stride;
            }
            score_samp.sort_unstable_by(f32::total_cmp);
            let qi = (((1.0 - 0.0085f64) * (score_samp.len() as f64)) as usize)
                .min(score_samp.len().saturating_sub(1));
            score_samp
                .get(qi)
                .copied()
                .unwrap_or(f32::INFINITY)
                .max(1e-7)
        } else {
            f32::INFINITY
        };

        // Pass 2: Extract pre-rotation FP16 CSR outliers & rotate clean rows via FWHT_32
        let mut outlier_row_ptr = Vec::with_capacity(rows + 1);
        outlier_row_ptr.push(0u32);
        let mut outliers_packed = Vec::with_capacity(total_numel / 110);

        let mut scales_f16 = vec![0u16; total_groups];
        let mut group_scales_f32 = vec![0.0f32; total_groups];
        // Sample rotated normalized coordinates during row sweep to avoid storing full `rows*padded_cols`!
        let row_sample_stride = (rows / 256).max(1);
        let mut rot_samples = Vec::with_capacity(65_536);
        let mut row_buf = vec![0.0f32; padded_cols];

        for r in 0..rows {
            let row_w = &weights[r * cols..(r + 1) * cols];
            row_buf[..cols].copy_from_slice(row_w);
            row_buf[cols..].fill(0.0);

            if allow_outliers {
                for g in 0..groups_per_row {
                    let sc = raw_rms[r * groups_per_row + g];
                    if sc == 0.0 {
                        continue;
                    }
                    let g_th = (out_thresh * sc.sqrt()).max(2.95 * sc);
                    let c0 = g * 128;
                    let valid = (cols.saturating_sub(c0)).min(128);
                    for i in 0..valid {
                        let col = c0 + i;
                        let w = row_buf[col];
                        if w.abs() > g_th {
                            let wf16 = f32_to_f16(w);
                            outliers_packed.push((col as u32) | ((wf16 as u32) << 16));
                            row_buf[col] = w - f16_to_f32(wf16);
                        }
                    }
                }
            }
            outlier_row_ptr.push(outliers_packed.len() as u32);

            for blk in row_buf.chunks_exact_mut(32) {
                let arr: &mut [f32; 32] = blk.try_into().unwrap();
                fwht32_inplace(arr);
            }

            let take_sample = (r % row_sample_stride == 0) && rot_samples.len() < 65_536;
            for g in 0..groups_per_row {
                let g_idx = r * groups_per_row + g;
                let s = &row_buf[g * 128..(g + 1) * 128];
                let mut sq = 0.0f32;
                for &v in s {
                    sq = v.mul_add(v, sq);
                }
                let rms = (sq * (1.0 / 128.0)).sqrt();
                let sf16 = f32_to_f16(rms);
                let sc = f16_to_f32(sf16);
                scales_f16[g_idx] = sf16;
                group_scales_f32[g_idx] = sc;
                if take_sample && sc > 0.0 {
                    let inv = 1.0 / sc;
                    for &v in s.iter().step_by(4) {
                        if rot_samples.len() < 65_536 {
                            rot_samples.push((v * inv).clamp(-6.0, 6.0));
                        }
                    }
                }
            }
        }

        let centroids = train_lloyd_max_4(&mut rot_samples);
        let mids = [
            0.5 * (centroids[0] + centroids[1]),
            0.5 * (centroids[1] + centroids[2]),
            0.5 * (centroids[2] + centroids[3]),
        ];

        // Pass 3: Re-rotate clean rows in parallel & pack 2-bit codes (zero extra full-matrix RAM!)
        let words_per_row = (padded_cols / 256) * 16;
        let mut packed_words = vec![0u32; rows * words_per_row];
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(rows.max(1));

        let pack_chunk = |r0: usize, pw_slice: &mut [u32]| {
            let mut rbuf = vec![0.0f32; padded_cols];
            for (dr, dst_row) in pw_slice.chunks_exact_mut(words_per_row).enumerate() {
                let r = r0 + dr;
                rbuf[..cols].copy_from_slice(&weights[r * cols..(r + 1) * cols]);
                rbuf[cols..].fill(0.0);
                let os = outlier_row_ptr[r] as usize;
                let oe = outlier_row_ptr[r + 1] as usize;
                for &pk in &outliers_packed[os..oe] {
                    let col = (pk & 0xffff) as usize;
                    let val = f16_to_f32((pk >> 16) as u16);
                    rbuf[col] -= val;
                }
                for blk in rbuf.chunks_exact_mut(32) {
                    let arr: &mut [f32; 32] = blk.try_into().unwrap();
                    fwht32_inplace(arr);
                }
                for step in 0..(padded_cols / 256) {
                    let mut u16s = [0u16; 32];
                    for lane in 0..32 {
                        let g = step * 2 + (lane / 16);
                        let sc = group_scales_f32[r * groups_per_row + g];
                        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                        let base = step * 256 + lane * 8;
                        let mut code16 = 0u16;
                        for k in 0..8 {
                            let q = if sc == 0.0 {
                                1u16
                            } else {
                                let z = rbuf[base + k] * inv;
                                let mut c = 0u16;
                                if z >= mids[1] {
                                    c = if z >= mids[2] { 3 } else { 2 };
                                } else if z >= mids[0] {
                                    c = 1;
                                }
                                c
                            };
                            code16 |= q << (k * 2);
                        }
                        u16s[lane] = code16;
                    }
                    let dst16 = &mut dst_row[step * 16..step * 16 + 16];
                    for w in 0..16 {
                        dst16[w] = (u16s[w * 2] as u32) | ((u16s[w * 2 + 1] as u32) << 16);
                    }
                }
            }
        };

        if workers <= 1 || rows < 32 {
            pack_chunk(0, &mut packed_words);
        } else {
            let rpw = rows.div_ceil(workers);
            std::thread::scope(|s| {
                for (ci, ch) in packed_words.chunks_mut(rpw * words_per_row).enumerate() {
                    s.spawn(move || pack_chunk(ci * rpw, ch));
                }
            });
        }

        Ok(Self {
            rows,
            cols,
            padded_cols,
            centroids,
            scales_f16,
            packed_words,
            outlier_row_ptr,
            outliers_packed,
        })
    }

    pub fn compressed_bytes(&self) -> usize {
        16 + self.scales_f16.len() * 2
            + self.packed_words.len() * 4
            + self.outlier_row_ptr.len() * 4
            + self.outliers_packed.len() * 4
    }

    pub fn bits_per_weight(&self) -> f64 {
        (self.compressed_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Reconstructs a single row `r` (`dst.len() == self.cols`) in `O(padded_cols)` via `FWHT_32`.
    pub fn decode_row(&self, r: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.cols);
        let groups_per_row = self.padded_cols / 128;
        let words_per_row = (self.padded_cols / 256) * 16;
        let row_pw = &self.packed_words[r * words_per_row..(r + 1) * words_per_row];
        let mut rbuf = vec![0.0f32; self.padded_cols];

        for step in 0..(self.padded_cols / 256) {
            let p16 = &row_pw[step * 16..step * 16 + 16];
            for lane in 0..32 {
                let g = step * 2 + (lane / 16);
                let sc = f16_to_f32(self.scales_f16[r * groups_per_row + g]);
                let code16 = ((p16[lane / 2] >> ((lane % 2) * 16)) & 0xffff) as usize;
                let base = step * 256 + lane * 8;
                for k in 0..8 {
                    rbuf[base + k] = sc * self.centroids[(code16 >> (k * 2)) & 3];
                }
            }
        }
        for blk in rbuf.chunks_exact_mut(32) {
            let arr: &mut [f32; 32] = blk.try_into().unwrap();
            fwht32_inplace(arr);
        }
        dst.copy_from_slice(&rbuf[..self.cols]);
        let os = self.outlier_row_ptr[r] as usize;
        let oe = self.outlier_row_ptr[r + 1] as usize;
        for &pk in &self.outliers_packed[os..oe] {
            let col = (pk & 0xffff) as usize;
            if col < self.cols {
                dst[col] += f16_to_f32((pk >> 16) as u16);
            }
        }
    }

    pub fn decode_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            self.decode_row(r, &mut out[r * self.cols..(r + 1) * self.cols]);
        }
        out
    }

    /// Fused compressed-domain CPU `GEMV`: rotates `x` once via `FWHT_32`, then computes dot
    /// products directly against 2-bit codes + sparse `FP16` CSR outliers on raw `x`.
    pub fn gemv_cpu(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        let mut x_raw = vec![0.0f32; self.padded_cols];
        x_raw[..self.cols].copy_from_slice(x);
        let x_rot = fwht32_padded(x, self.padded_cols);

        let workers = threads.clamp(1, 256).min(self.rows.max(1));
        if workers <= 1 || self.rows < 16 {
            self.gemv_rows(&x_raw, &x_rot, y, 0);
        } else {
            let chunk = self.rows.div_ceil(workers);
            std::thread::scope(|s| {
                for (ci, yc) in y.chunks_mut(chunk).enumerate() {
                    let xr = &x_raw;
                    let xh = &x_rot;
                    s.spawn(move || self.gemv_rows(xr, xh, yc, ci * chunk));
                }
            });
        }
    }

    fn gemv_rows(&self, x_raw: &[f32], x_rot: &[f32], y_slice: &mut [f32], r0: usize) {
        let groups_per_row = self.padded_cols / 128;
        let words_per_row = (self.padded_cols / 256) * 16;
        let c = self.centroids;

        for (dr, out_y) in y_slice.iter_mut().enumerate() {
            let r = r0 + dr;
            let row_pw = &self.packed_words[r * words_per_row..(r + 1) * words_per_row];
            let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

            for step in 0..(self.padded_cols / 256) {
                let p16 = &row_pw[step * 16..step * 16 + 16];
                for half_g in 0..2 {
                    let g = step * 2 + half_g;
                    let sc = f16_to_f32(self.scales_f16[r * groups_per_row + g]);
                    if sc == 0.0 {
                        continue;
                    }
                    let slut = [sc * c[0], sc * c[1], sc * c[2], sc * c[3]];
                    for sub in 0..16 {
                        let lane = half_g * 16 + sub;
                        let code16 = ((p16[lane / 2] >> ((lane % 2) * 16)) & 0xffff) as usize;
                        let xv = &x_rot[step * 256 + lane * 8..step * 256 + lane * 8 + 8];
                        a0 = slut[code16 & 3].mul_add(xv[0], a0);
                        a1 = slut[(code16 >> 2) & 3].mul_add(xv[1], a1);
                        a2 = slut[(code16 >> 4) & 3].mul_add(xv[2], a2);
                        a3 = slut[(code16 >> 6) & 3].mul_add(xv[3], a3);
                        a0 = slut[(code16 >> 8) & 3].mul_add(xv[4], a0);
                        a1 = slut[(code16 >> 10) & 3].mul_add(xv[5], a1);
                        a2 = slut[(code16 >> 12) & 3].mul_add(xv[6], a2);
                        a3 = slut[(code16 >> 14) & 3].mul_add(xv[7], a3);
                    }
                }
            }
            let mut sum = (a0 + a1) + (a2 + a3);
            let os = self.outlier_row_ptr[r] as usize;
            let oe = self.outlier_row_ptr[r + 1] as usize;
            for &pk in &self.outliers_packed[os..oe] {
                let col = (pk & 0xffff) as usize;
                let val = f16_to_f32((pk >> 16) as u16);
                sum = val.mul_add(x_raw[col], sum);
            }
            *out_y = sum;
        }
    }
}

fn train_lloyd_max_4(samples: &mut [f32]) -> [f32; 16] {
    if samples.is_empty() {
        let (_, p) = fit_cubic_codebook(&[-1.51, -0.45, 0.45, 1.51]);
        return p;
    }
    samples.sort_unstable_by(f32::total_cmp);
    let n = samples.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0f64);
    let mut run = 0.0f64;
    for &v in samples.iter() {
        run += v as f64;
        prefix.push(run);
    }
    let mut c = [
        samples[n / 8],
        samples[(3 * n) / 8],
        samples[(5 * n) / 8],
        samples[(7 * n) / 8],
    ];
    for i in 1..4 {
        if c[i] <= c[i - 1] {
            c[i] = c[i - 1] + 1e-4;
        }
    }
    for _ in 0..24 {
        let b1 = samples.partition_point(|&v| v < 0.5 * (c[0] + c[1]));
        let b2 = samples.partition_point(|&v| v < 0.5 * (c[1] + c[2]));
        let b3 = samples.partition_point(|&v| v < 0.5 * (c[2] + c[3]));
        let bounds = [0, b1, b2, b3, n];
        for i in 0..4 {
            if bounds[i + 1] > bounds[i] {
                c[i] = ((prefix[bounds[i + 1]] - prefix[bounds[i]])
                    / ((bounds[i + 1] - bounds[i]) as f64)) as f32;
            }
        }
    }
    let (_, poly_cents) = fit_cubic_codebook(&c);
    poly_cents
}

#[inline]
fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
