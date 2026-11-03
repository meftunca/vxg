//! Mode 2 (`Cb4Outlier` & `Cb3Outlier`): Non-uniform 1D Lloyd-Max learned codebook (`3` or `4`
//! bits/weight) + adaptive sparse `FP16` outlier CSR stream, plus uniform `Q4_0` (`4.50 bpw`)
//! reference codec and fused compressed-domain CPU `GEMV` kernels.

use crate::fp16::{f16_to_f32, f32_to_f16, fit_cubic_codebook};
use crate::{Error, Result};

/// Compressed-domain non-uniform codebook + sparse `FP16` outlier matrix (`bits = 3` or `4`).
#[derive(Clone, Debug)]
pub struct CodebookMatrix {
    pub bits: u8,
    pub rows: usize,
    pub cols: usize,
    pub padded_cols: usize,
    pub centroids: [f32; 16],
    /// `1` `f16` scale per 128-weight group (`rows * (padded_cols / 128)` entries).
    pub scales_f16: Vec<u16>,
    /// Planar bit-packed `u32` masks per 32-weight tile (`3` `u32`s/tile for 3-bit, `4` for 4-bit).
    pub packed_words: Vec<u32>,
    /// CSR row offsets (`rows + 1` entries) into `outliers_packed`.
    pub outlier_row_ptr: Vec<u32>,
    /// Packed `(col_idx as u32) | ((val_f16 as u32) << 16)` when `cols <= 65536`.
    pub outliers_packed: Vec<u32>,
}

impl CodebookMatrix {
    /// Encodes `weights` (`rows * cols` `f32`) into a `3`-bit or `4`-bit non-uniform Lloyd-Max
    /// codebook + sparse `FP16` outlier matrix.
    pub fn encode(bits: u8, rows: usize, cols: usize, weights: &[f32]) -> Result<Self> {
        if !matches!(bits, 3 | 4) {
            return Err(Error::Unsupported(
                "CodebookMatrix supports 3 or 4 bits".into(),
            ));
        }
        if rows == 0 || cols == 0 || weights.len() != rows.saturating_mul(cols) {
            return Err(Error::InvalidShape(
                "invalid rows*cols for CodebookMatrix::encode",
            ));
        }
        let k_cent = 1usize << bits;
        let padded_cols = cols.div_ceil(256) * 256;
        let groups_per_row = padded_cols / 128;
        let total_groups = rows * groups_per_row;
        let total_tiles = total_groups * 4;
        let allow_outliers = cols <= 65536;
        let train_gate = if bits == 3 { 3.25f32 } else { 3.55f32 };

        let mut scales_f16 = vec![0u16; total_groups];

        // Pass 1: Robust 2-step trimmed RMS group scale (per 128 weights)
        let mut group_scales_f32 = vec![0.0f32; total_groups];

        for r in 0..rows {
            let row_w = &weights[r * cols..(r + 1) * cols];
            for g in 0..groups_per_row {
                let g_idx = r * groups_per_row + g;
                let c_start = g * 128;
                let valid = (cols.saturating_sub(c_start)).min(128);
                if valid == 0 {
                    continue;
                }
                let slice = &row_w[c_start..c_start + valid];

                let mut sum_sq0 = 0.0f32;
                for &v in slice {
                    sum_sq0 = v.mul_add(v, sum_sq0);
                }
                let rms0 = (sum_sq0 / (valid as f32)).sqrt();
                if rms0 == 0.0 {
                    continue;
                }

                let trim_cut = 2.8f32 * rms0;
                let mut clean_sq = 0.0f32;
                let mut clean_n = 0usize;
                for &v in slice {
                    if v.abs() <= trim_cut {
                        clean_sq = v.mul_add(v, clean_sq);
                        clean_n += 1;
                    }
                }
                let rms_clean = if clean_n >= 8 {
                    (clean_sq / (clean_n as f32)).sqrt().max(rms0 * 0.15)
                } else {
                    rms0
                };

                let sf16 = f32_to_f16(rms_clean);
                scales_f16[g_idx] = sf16;
                group_scales_f32[g_idx] = f16_to_f32(sf16);
            }
        }

        // Pass 2: Sample bulk RMS-normalized weights & fit cubic Lloyd-Max centroids
        let total_numel = rows * cols;
        let sample_target = 65_536usize;
        let mut stride = (total_numel / sample_target).max(1);
        while stride > 1 && gcd_usize(stride, cols) != 1 {
            stride += 1;
        }
        let mut samples = Vec::with_capacity(sample_target.min(total_numel) + 128);

        let mut idx_cursor = 0usize;
        while idx_cursor < total_numel {
            let r = idx_cursor / cols;
            let c = idx_cursor % cols;
            let sc = group_scales_f32[r * groups_per_row + (c / 128)];
            if sc > 0.0 {
                let z = weights[idx_cursor] / sc;
                if !allow_outliers || z.abs() <= train_gate {
                    samples.push(z.clamp(-8.0, 8.0));
                }
            } else {
                samples.push(0.0);
            }
            idx_cursor += stride;
        }

        let centroids = train_lloyd_max_1d(&mut samples, k_cent);
        let mut midpoints = [0.0f32; 15];
        for i in 0..(k_cent - 1) {
            midpoints[i] = 0.5 * (centroids[i] + centroids[i + 1]);
        }

        // Pass 3: Quantize & pack all 128-weight groups (multi-threaded over rows!)
        let words_per_tile = bits as usize;
        let mut packed_words = vec![0u32; total_tiles * words_per_tile];
        let words_per_row = groups_per_row * 4 * words_per_tile;
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(rows.max(1));

        let encode_rows = |r_start: usize, pw_slice: &mut [u32]| {
            for (dr, row_pw) in pw_slice.chunks_exact_mut(words_per_row).enumerate() {
                let r = r_start + dr;
                let row_w = &weights[r * cols..(r + 1) * cols];
                if bits == 4 {
                    for g in 0..groups_per_row {
                        let g_idx = r * groups_per_row + g;
                        let sc = group_scales_f32[g_idx];
                        let inv_sc = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                        for sub in 0..4 {
                            let base_col = g * 128 + sub * 32;
                            let off = (g * 4 + sub) * 4;
                            for w in 0..4 {
                                let mut word = 0u32;
                                for k in 0..8 {
                                    let col = base_col + w * 8 + k;
                                    let q = if col >= cols || sc == 0.0 {
                                        0usize
                                    } else {
                                        nearest_centroid(row_w[col] * inv_sc, &midpoints[..15])
                                    };
                                    word |= (q as u32) << (k * 4);
                                }
                                row_pw[off + w] = word;
                            }
                        }
                    }
                } else {
                    // Coalesced 256-weight blocks (2 groups of 128 = 24 u32s: 32 ushort lo2 + 32 uchar hi1)
                    let steps = groups_per_row / 2;
                    for step in 0..steps {
                        let mut lo2_arr = [0u16; 32];
                        let mut hi1_arr = [0u8; 32];
                        for lane in 0..32 {
                            let g = step * 2 + (lane / 16);
                            let g_idx = r * groups_per_row + g;
                            let sc = group_scales_f32[g_idx];
                            let inv_sc = if sc > 0.0 { 1.0 / sc } else { 0.0 };
                            let base_col = step * 256 + lane * 8;
                            let mut lo2 = 0u16;
                            let mut hi1 = 0u8;
                            for k in 0..8 {
                                let col = base_col + k;
                                let q = if col >= cols || sc == 0.0 {
                                    0usize
                                } else {
                                    nearest_centroid(row_w[col] * inv_sc, &midpoints[..7])
                                };
                                lo2 |= ((q as u16) & 3) << (k * 2);
                                hi1 |= (((q as u8) >> 2) & 1) << k;
                            }
                            lo2_arr[lane] = lo2;
                            hi1_arr[lane] = hi1;
                        }
                        let dst24 = &mut row_pw[step * 24..step * 24 + 24];
                        for w in 0..16 {
                            dst24[w] =
                                (lo2_arr[w * 2] as u32) | ((lo2_arr[w * 2 + 1] as u32) << 16);
                        }
                        for w in 0..8 {
                            dst24[16 + w] = u32::from_le_bytes([
                                hi1_arr[w * 4],
                                hi1_arr[w * 4 + 1],
                                hi1_arr[w * 4 + 2],
                                hi1_arr[w * 4 + 3],
                            ]);
                        }
                    }
                }
            }
        };

        if workers <= 1 || rows < 32 {
            encode_rows(0, &mut packed_words);
        } else {
            let rows_per_worker = rows.div_ceil(workers);
            let pw_chunk_len = rows_per_worker * words_per_row;
            std::thread::scope(|s| {
                for (ci, pw_chunk) in packed_words.chunks_mut(pw_chunk_len).enumerate() {
                    let r_start = ci * rows_per_worker;
                    s.spawn(move || encode_rows(r_start, pw_chunk));
                }
            });
        }

        // Pass 4: L2-optimal quantile residual outlier selection (`w - sc * centroids[q]`)
        let mut outlier_row_ptr = Vec::with_capacity(rows + 1);
        outlier_row_ptr.push(0u32);
        let mut outliers_packed: Vec<u32> = Vec::with_capacity(total_numel / 110);
        if allow_outliers {
            let target_frac = if bits == 3 { 0.0080f64 } else { 0.0085f64 };
            let min_z_res = if bits == 3 { 0.55f32 } else { 0.36f32 };
            let mut err_samples = Vec::with_capacity(sample_target.min(total_numel) + 128);
            let mut ec = 0usize;
            while ec < total_numel {
                let r = ec / cols;
                let c = ec % cols;
                let sc = group_scales_f32[r * groups_per_row + (c / 128)];
                if sc > 0.0 {
                    let q = nearest_centroid(weights[ec] / sc, &midpoints[..k_cent - 1]);
                    let res = (weights[ec] - sc * centroids[q]).abs();
                    err_samples.push(res / sc.sqrt());
                } else {
                    err_samples.push(0.0);
                }
                ec += stride;
            }
            err_samples.sort_unstable_by(f32::total_cmp);
            let q_idx = (((1.0 - target_frac) * (err_samples.len() as f64)) as usize)
                .min(err_samples.len().saturating_sub(1));
            let err_thresh = err_samples
                .get(q_idx)
                .copied()
                .unwrap_or(f32::INFINITY)
                .max(1e-7);

            for r in 0..rows {
                let row_w = &weights[r * cols..(r + 1) * cols];
                for g in 0..groups_per_row {
                    let sc = group_scales_f32[r * groups_per_row + g];
                    if sc == 0.0 {
                        continue;
                    }
                    let inv_sc = 1.0 / sc;
                    let group_res_thresh = (err_thresh * sc.sqrt()).max(min_z_res * sc);
                    let c_start = g * 128;
                    let valid = (cols.saturating_sub(c_start)).min(128);
                    for i in 0..valid {
                        let col = c_start + i;
                        let w = row_w[col];
                        let q = nearest_centroid(w * inv_sc, &midpoints[..k_cent - 1]);
                        let residual = w - sc * centroids[q];
                        if residual.abs() > group_res_thresh {
                            let rf16 = f32_to_f16(residual) as u32;
                            outliers_packed.push((col as u32) | (rf16 << 16));
                        }
                    }
                }
                outlier_row_ptr.push(outliers_packed.len() as u32);
            }
        } else {
            outlier_row_ptr.resize(rows + 1, 0u32);
        }

        Ok(Self {
            bits,
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
        64 + self.scales_f16.len() * 2
            + self.packed_words.len() * 4
            + self.outlier_row_ptr.len() * 4
            + self.outliers_packed.len() * 4
    }

    pub fn bits_per_weight(&self) -> f64 {
        (self.compressed_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    pub fn outlier_count(&self) -> usize {
        self.outliers_packed.len()
    }

    /// Reconstructs a single row `r` (`dst.len() == self.cols`) in `O(cols)`.
    pub fn decode_row(&self, r: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.cols);
        let groups_per_row = self.padded_cols / 128;
        let wpt = self.bits as usize;
        if self.bits == 4 {
            for g in 0..groups_per_row {
                let g_idx = r * groups_per_row + g;
                let sc = f16_to_f32(self.scales_f16[g_idx]);
                for sub in 0..4 {
                    let col_base = g * 128 + sub * 32;
                    if col_base >= self.cols {
                        break;
                    }
                    let valid = (self.cols - col_base).min(32);
                    let tile_idx = g_idx * 4 + sub;
                    let cw = &self.packed_words[tile_idx * wpt..(tile_idx + 1) * wpt];
                    for l in 0..valid {
                        let q = ((cw[l / 8] >> ((l % 8) * 4)) & 0x0f) as usize;
                        dst[col_base + l] = sc * self.centroids[q];
                    }
                }
            }
        } else {
            let row_pw = &self.packed_words[r * groups_per_row * 12..(r + 1) * groups_per_row * 12];
            for step in 0..(groups_per_row / 2) {
                let p24 = &row_pw[step * 24..step * 24 + 24];
                for lane in 0..32 {
                    let base_col = step * 256 + lane * 8;
                    if base_col >= self.cols {
                        break;
                    }
                    let valid = (self.cols - base_col).min(8);
                    let g = step * 2 + (lane / 16);
                    let sc = f16_to_f32(self.scales_f16[r * groups_per_row + g]);
                    let lo2 = ((p24[lane / 2] >> ((lane % 2) * 16)) & 0xffff) as u16;
                    let hi1 = (p24[16 + lane / 4].to_le_bytes())[lane % 4];
                    for k in 0..valid {
                        let q =
                            (((lo2 >> (k * 2)) & 3) as usize) | ((((hi1 >> k) & 1) as usize) << 2);
                        dst[base_col + k] = sc * self.centroids[q];
                    }
                }
            }
        }
        let os = self.outlier_row_ptr[r] as usize;
        let oe = self.outlier_row_ptr[r + 1] as usize;
        for &packed in &self.outliers_packed[os..oe] {
            let col = (packed & 0xffff) as usize;
            if col < self.cols {
                dst[col] += f16_to_f32((packed >> 16) as u16);
            }
        }
    }

    /// Reconstructs approximated `f32` matrix (`rows * cols`) for `SQNR (dB)` verification.
    pub fn decode_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            self.decode_row(r, &mut out[r * self.cols..(r + 1) * self.cols]);
        }
        out
    }

    /// Fused compressed-domain multi-threaded CPU `GEMV` (`y = W · x`) with pre-scaled register LUT
    /// + sparse `FP16` CSR outlier accumulation and zero weight decompression buffer.
    pub fn gemv_cpu(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);

        let mut x_pad = vec![0.0f32; self.padded_cols];
        x_pad[..self.cols].copy_from_slice(x);

        let workers = threads.clamp(1, 256).min(self.rows.max(1));
        if workers <= 1 || self.rows < 16 {
            self.gemv_rows_range(&x_pad, y, 0);
        } else {
            let chunk = self.rows.div_ceil(workers);
            std::thread::scope(|s| {
                for (ci, y_chunk) in y.chunks_mut(chunk).enumerate() {
                    let r0 = ci * chunk;
                    let xp = &x_pad;
                    s.spawn(move || self.gemv_rows_range(xp, y_chunk, r0));
                }
            });
        }
    }

    fn gemv_rows_range(&self, x_pad: &[f32], y_slice: &mut [f32], row_start: usize) {
        let groups_per_row = self.padded_cols / 128;
        let cents = self.centroids;

        if self.bits == 4 {
            for (dr, out_y) in y_slice.iter_mut().enumerate() {
                let r = row_start + dr;
                let (mut acc0, mut acc1, mut acc2, mut acc3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

                for g in 0..groups_per_row {
                    let g_idx = r * groups_per_row + g;
                    let sc = f16_to_f32(self.scales_f16[g_idx]);
                    if sc == 0.0 {
                        continue;
                    }
                    let mut slut = [0.0f32; 16];
                    for k in 0..16 {
                        slut[k] = sc * cents[k];
                    }
                    for sub in 0..4 {
                        let tile_idx = g_idx * 4 + sub;
                        let cw = &self.packed_words[tile_idx * 4..tile_idx * 4 + 4];
                        let base_x = g * 128 + sub * 32;
                        for (w, &word) in cw.iter().enumerate() {
                            let xv = &x_pad[base_x + w * 8..base_x + w * 8 + 8];
                            acc0 = slut[(word & 0x0f) as usize].mul_add(xv[0], acc0);
                            acc1 = slut[((word >> 4) & 0x0f) as usize].mul_add(xv[1], acc1);
                            acc2 = slut[((word >> 8) & 0x0f) as usize].mul_add(xv[2], acc2);
                            acc3 = slut[((word >> 12) & 0x0f) as usize].mul_add(xv[3], acc3);
                            acc0 = slut[((word >> 16) & 0x0f) as usize].mul_add(xv[4], acc0);
                            acc1 = slut[((word >> 20) & 0x0f) as usize].mul_add(xv[5], acc1);
                            acc2 = slut[((word >> 24) & 0x0f) as usize].mul_add(xv[6], acc2);
                            acc3 = slut[((word >> 28) & 0x0f) as usize].mul_add(xv[7], acc3);
                        }
                    }
                }
                let mut sum = (acc0 + acc1) + (acc2 + acc3);
                let os = self.outlier_row_ptr[r] as usize;
                let oe = self.outlier_row_ptr[r + 1] as usize;
                for &packed in &self.outliers_packed[os..oe] {
                    let col = (packed & 0xffff) as usize;
                    let val = f16_to_f32((packed >> 16) as u16);
                    sum = val.mul_add(x_pad[col], sum);
                }
                *out_y = sum;
            }
        } else {
            for (dr, out_y) in y_slice.iter_mut().enumerate() {
                let r = row_start + dr;
                let (mut acc0, mut acc1, mut acc2, mut acc3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let row_pw =
                    &self.packed_words[r * groups_per_row * 12..(r + 1) * groups_per_row * 12];

                for step in 0..(groups_per_row / 2) {
                    let p24 = &row_pw[step * 24..step * 24 + 24];
                    for half_grp in 0..2 {
                        let g = step * 2 + half_grp;
                        let sc = f16_to_f32(self.scales_f16[r * groups_per_row + g]);
                        if sc == 0.0 {
                            continue;
                        }
                        let mut slut = [0.0f32; 8];
                        for k in 0..8 {
                            slut[k] = sc * cents[k];
                        }
                        for sub_lane in 0..16 {
                            let lane = half_grp * 16 + sub_lane;
                            let lo2 = ((p24[lane / 2] >> ((lane % 2) * 16)) & 0xffff) as usize;
                            let hi1 = (p24[16 + lane / 4].to_le_bytes())[lane % 4] as usize;
                            let xv = &x_pad[step * 256 + lane * 8..step * 256 + lane * 8 + 8];
                            acc0 = slut[(lo2 & 3) | ((hi1 & 1) << 2)].mul_add(xv[0], acc0);
                            acc1 = slut[((lo2 >> 2) & 3) | (((hi1 >> 1) & 1) << 2)]
                                .mul_add(xv[1], acc1);
                            acc2 = slut[((lo2 >> 4) & 3) | (((hi1 >> 2) & 1) << 2)]
                                .mul_add(xv[2], acc2);
                            acc3 = slut[((lo2 >> 6) & 3) | (((hi1 >> 3) & 1) << 2)]
                                .mul_add(xv[3], acc3);
                            acc0 = slut[((lo2 >> 8) & 3) | (((hi1 >> 4) & 1) << 2)]
                                .mul_add(xv[4], acc0);
                            acc1 = slut[((lo2 >> 10) & 3) | (((hi1 >> 5) & 1) << 2)]
                                .mul_add(xv[5], acc1);
                            acc2 = slut[((lo2 >> 12) & 3) | (((hi1 >> 6) & 1) << 2)]
                                .mul_add(xv[6], acc2);
                            acc3 = slut[((lo2 >> 14) & 3) | (((hi1 >> 7) & 1) << 2)]
                                .mul_add(xv[7], acc3);
                        }
                    }
                }
                let mut sum = (acc0 + acc1) + (acc2 + acc3);
                let os = self.outlier_row_ptr[r] as usize;
                let oe = self.outlier_row_ptr[r + 1] as usize;
                for &packed in &self.outliers_packed[os..oe] {
                    let col = (packed & 0xffff) as usize;
                    let val = f16_to_f32((packed >> 16) as u16);
                    sum = val.mul_add(x_pad[col], sum);
                }
                *out_y = sum;
            }
        }
    }
}

/// Fast exact 1D Lloyd-Max centroid optimizer using sorted sample + $O(K \log N)$ prefix-sum updates.
fn train_lloyd_max_1d(samples: &mut [f32], k: usize) -> [f32; 16] {
    let mut cents = [0.0f32; 16];
    if samples.is_empty() {
        for (i, slot) in cents.iter_mut().take(k).enumerate() {
            *slot = -1.0 + (2.0 * (i as f32)) / ((k - 1).max(1) as f32);
        }
        return cents;
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0f64);
    let mut run = 0.0f64;
    for &v in samples.iter() {
        run += v as f64;
        prefix.push(run);
    }

    for (i, slot) in cents.iter_mut().take(k).enumerate() {
        let q_idx = ((2 * i + 1) * n) / (2 * k);
        *slot = samples[q_idx.min(n - 1)];
    }
    // Ensure strict initial monotonicity if degenerate
    for i in 1..k {
        if cents[i] <= cents[i - 1] {
            cents[i] = (cents[i - 1] + 1e-4).min(1.0);
        }
    }

    for _iter in 0..20 {
        let mut bounds = [0usize; 17];
        bounds[0] = 0;
        bounds[k] = n;
        for b in 1..k {
            let mid = 0.5 * (cents[b - 1] + cents[b]);
            bounds[b] = samples.partition_point(|&v| v < mid);
        }
        for c in 0..k {
            let lo = bounds[c];
            let hi = bounds[c + 1];
            if hi > lo {
                cents[c] = ((prefix[hi] - prefix[lo]) / ((hi - lo) as f64)) as f32;
            }
        }
    }
    let (_, poly_cents) = fit_cubic_codebook(&cents[..k]);
    poly_cents
}

#[inline(always)]
fn nearest_centroid(val: f32, mids: &[f32]) -> usize {
    if mids.len() == 15 {
        let mut base = 0usize;
        if val >= mids[7] {
            base = 8;
        }
        if val >= mids[base + 3] {
            base += 4;
        }
        if val >= mids[base + 1] {
            base += 2;
        }
        if val >= mids[base] {
            base += 1;
        }
        base
    } else {
        // len == 7 (3-bit)
        let mut base = 0usize;
        if val >= mids[3] {
            base = 4;
        }
        if val >= mids[base + 1] {
            base += 2;
        }
        if val >= mids[base] {
            base += 1;
        }
        base
    }
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

pub use crate::q4_0::Q4_0Matrix;
