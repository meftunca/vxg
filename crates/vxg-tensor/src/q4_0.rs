//! Standard `GGUF Q4_0` (`32` weights/block, `1` `f16` scale + `16` nibble bytes = `4.50 bpw`)
//! reference codec and multi-threaded CPU `GEMV` kernel for head-to-head benchmarking.

use crate::fp16::{f16_to_f32, f32_to_f16};

/// Standard `GGUF Q4_0` (`32` weights/block = `4.50 bits/weight`) reference matrix.
#[derive(Clone, Debug)]
pub struct Q4_0Matrix {
    pub rows: usize,
    pub cols: usize,
    pub padded_cols: usize,
    pub scales_f16: Vec<u16>,
    pub qs_words: Vec<u32>,
}

impl Q4_0Matrix {
    pub fn encode(rows: usize, cols: usize, weights: &[f32]) -> Self {
        let padded_cols = cols.div_ceil(32) * 32;
        let blocks_per_row = padded_cols / 32;
        let total_blocks = rows * blocks_per_row;
        let mut scales_f16 = vec![0u16; total_blocks];
        let mut qs_words = vec![0u32; total_blocks * 4];

        for r in 0..rows {
            let row_w = &weights[r * cols..(r + 1) * cols];
            for b in 0..blocks_per_row {
                let blk_idx = r * blocks_per_row + b;
                let c0 = b * 32;
                let mut amax = 0.0f32;
                let mut vals = [0.0f32; 32];
                for l in 0..32 {
                    if c0 + l < cols {
                        let v = row_w[c0 + l];
                        vals[l] = v;
                        if v.abs() > amax {
                            amax = v.abs();
                        }
                    }
                }
                let d = amax / 7.0;
                let df16 = f32_to_f16(d);
                scales_f16[blk_idx] = df16;
                let d_eff = f16_to_f32(df16);
                let id = if d_eff != 0.0 { 1.0 / d_eff } else { 0.0 };

                let mut bytes = [0u8; 16];
                for j in 0..16 {
                    let q0 = ((vals[j] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
                    let q1 = ((vals[j + 16] * id + 8.5).floor() as i32).clamp(0, 15) as u8;
                    bytes[j] = q0 | (q1 << 4);
                }
                for w in 0..4 {
                    qs_words[blk_idx * 4 + w] = u32::from_le_bytes([
                        bytes[w * 4],
                        bytes[w * 4 + 1],
                        bytes[w * 4 + 2],
                        bytes[w * 4 + 3],
                    ]);
                }
            }
        }
        Self {
            rows,
            cols,
            padded_cols,
            scales_f16,
            qs_words,
        }
    }

    pub fn compressed_bytes(&self) -> usize {
        self.scales_f16.len() * 2 + self.qs_words.len() * 4
    }

    pub fn bits_per_weight(&self) -> f64 {
        (self.compressed_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    pub fn decode_row(&self, r: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.cols);
        let bpr = self.padded_cols / 32;
        for b in 0..bpr {
            let bi = r * bpr + b;
            let d = f16_to_f32(self.scales_f16[bi]);
            let c0 = b * 32;
            for w in 0..4 {
                let bytes = self.qs_words[bi * 4 + w].to_le_bytes();
                for k in 0..4 {
                    let j = w * 4 + k;
                    if c0 + j < self.cols {
                        dst[c0 + j] = (((bytes[k] & 0x0f) as i32) - 8) as f32 * d;
                    }
                    if c0 + j + 16 < self.cols {
                        dst[c0 + j + 16] = (((bytes[k] >> 4) as i32) - 8) as f32 * d;
                    }
                }
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

    pub fn gemv_cpu(&self, x: &[f32], y: &mut [f32], threads: usize) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        let mut x_pad = vec![0.0f32; self.padded_cols];
        x_pad[..self.cols].copy_from_slice(x);
        let bpr = self.padded_cols / 32;
        let workers = threads.clamp(1, 256).min(self.rows.max(1));
        let chunk = self.rows.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, y_chunk) in y.chunks_mut(chunk).enumerate() {
                let r0 = ci * chunk;
                let xp = &x_pad;
                s.spawn(move || {
                    for (dr, out) in y_chunk.iter_mut().enumerate() {
                        let r = r0 + dr;
                        let mut sum = 0.0f32;
                        for b in 0..bpr {
                            let bi = r * bpr + b;
                            let d = f16_to_f32(self.scales_f16[bi]);
                            let xv = &xp[b * 32..b * 32 + 32];
                            let mut blk_acc = 0.0f32;
                            for w in 0..4 {
                                let bytes = self.qs_words[bi * 4 + w].to_le_bytes();
                                let j = w * 4;
                                for k in 0..4 {
                                    let q0 = ((bytes[k] & 0x0f) as i32 - 8) as f32;
                                    let q1 = ((bytes[k] >> 4) as i32 - 8) as f32;
                                    blk_acc = q0.mul_add(xv[j + k], blk_acc);
                                    blk_acc = q1.mul_add(xv[j + k + 16], blk_acc);
                                }
                            }
                            sum = d.mul_add(blk_acc, sum);
                        }
                        *out = sum;
                    }
                });
            }
        });
    }
}
