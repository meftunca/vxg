//! Mode 1 (`LosslessBp16`): 100% bit-exact lossless bit-plane + Top-7 exponent LUT compression for
//! `BF16` (`~11.3 bits/weight`) and `FP16` (`~14.2 bits/weight`) neural weights, with coalesced
//! superblock planes (`uchar[32]` per bit-plane) for fused GPU (`simdgroup`) and CPU `GEMV`.

use crate::fp16::{bf16_to_f32, f16_to_f32};
use crate::{DType, Error, Result};

#[derive(Clone, Debug)]
pub struct LosslessMatrix {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    pub padded_cols: usize,
    pub lut: [u8; 8],
    pub base_bytes: Vec<u8>,
    pub code_words: Vec<u32>,
    pub sb_esc_offsets: Vec<u32>,
    pub esc_bytes: Vec<u8>,
}

#[inline]
fn pack_32_bytes_to_u32s(src: &[u8; 32], dst8: &mut [u32]) {
    for w in 0..8 {
        dst8[w] = u32::from_le_bytes([src[w * 4], src[w * 4 + 1], src[w * 4 + 2], src[w * 4 + 3]]);
    }
}

#[inline]
fn read_sb_plane_byte(sb_cw: &[u32], plane: usize, lane: usize) -> u8 {
    let word = sb_cw[plane * 8 + (lane >> 2)];
    ((word >> ((lane & 3) * 8)) & 0xff) as u8
}

impl LosslessMatrix {
    /// Encodes `words` (`rows * cols` raw `u16` values in `BF16` or `F16`) losslessly.
    pub fn encode(dtype: DType, rows: usize, cols: usize, words: &[u16]) -> Result<Self> {
        if rows == 0 || cols == 0 || words.len() != rows.saturating_mul(cols) {
            return Err(Error::InvalidShape(
                "invalid rows*cols for LosslessMatrix::encode",
            ));
        }
        let padded_cols = cols.div_ceil(256) * 256;
        let sbs_per_row = padded_cols / 256;
        let total_sbs = rows * sbs_per_row;
        let total_padded = rows * padded_cols;

        let mut freq = [0u64; 256];
        match dtype {
            DType::BF16 => {
                for &w in words {
                    freq[((w >> 7) & 0xff) as usize] += 1;
                }
            }
            DType::F16 => {
                for &w in words {
                    freq[((w >> 10) & 0x1f) as usize] += 1;
                }
            }
        }
        if padded_cols > cols {
            freq[0] += ((padded_cols - cols) * rows) as u64;
        }

        let max_sym = match dtype {
            DType::BF16 => 256usize,
            DType::F16 => 32usize,
        };
        let mut sym_counts: Vec<(u8, u64)> = (0..max_sym).map(|s| (s as u8, freq[s])).collect();
        sym_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut lut = [0u8; 8];
        let mut rev = [u8::MAX; 256];
        for (code, &(sym, _)) in sym_counts.iter().take(7).enumerate() {
            lut[code] = sym;
            rev[sym as usize] = code as u8;
        }

        let words_per_sb = match dtype {
            DType::BF16 => 24usize,
            DType::F16 => 48usize,
        };
        let mut base_bytes = vec![0u8; total_padded];
        let mut code_words = vec![0u32; total_sbs * words_per_sb];
        let mut sb_esc_offsets = Vec::with_capacity(total_sbs);
        let mut esc_bytes = Vec::with_capacity(total_padded / 32);

        for r in 0..rows {
            let row_src = &words[r * cols..(r + 1) * cols];
            for sb in 0..sbs_per_row {
                let sb_idx = r * sbs_per_row + sb;
                let sb_esc_base = u32::try_from(esc_bytes.len())
                    .map_err(|_| Error::Format("escape stream > 4 GiB".into()))?;
                sb_esc_offsets.push(sb_esc_base);
                let base_sb = &mut base_bytes[sb_idx * 256..(sb_idx + 1) * 256];
                let sb_cw = &mut code_words[sb_idx * words_per_sb..(sb_idx + 1) * words_per_sb];

                match dtype {
                    DType::BF16 => {
                        let (mut p0, mut p1, mut p2) = ([0u8; 32], [0u8; 32], [0u8; 32]);
                        for lane in 0..32 {
                            let (mut b0, mut b1, mut b2) = (0u8, 0u8, 0u8);
                            for k in 0..8 {
                                let off = lane * 8 + k;
                                let c = sb * 256 + off;
                                let w = if c < cols { row_src[c] } else { 0u16 };
                                base_sb[off] = (((w >> 8) & 0x80) | (w & 0x7f)) as u8;
                                let exp = ((w >> 7) & 0xff) as u8;
                                let code = rev[exp as usize];
                                if code < 7 {
                                    b0 |= (code & 1) << k;
                                    b1 |= ((code >> 1) & 1) << k;
                                    b2 |= ((code >> 2) & 1) << k;
                                } else {
                                    b0 |= 1u8 << k;
                                    b1 |= 1u8 << k;
                                    b2 |= 1u8 << k;
                                    esc_bytes.push(exp);
                                }
                            }
                            p0[lane] = b0;
                            p1[lane] = b1;
                            p2[lane] = b2;
                        }
                        pack_32_bytes_to_u32s(&p0, &mut sb_cw[0..8]);
                        pack_32_bytes_to_u32s(&p1, &mut sb_cw[8..16]);
                        pack_32_bytes_to_u32s(&p2, &mut sb_cw[16..24]);
                    }
                    DType::F16 => {
                        let (mut ph0, mut ph1, mut psm) = ([0u8; 32], [0u8; 32], [0u8; 32]);
                        let (mut pc0, mut pc1, mut pc2) = ([0u8; 32], [0u8; 32], [0u8; 32]);
                        for lane in 0..32 {
                            let (mut h0, mut h1, mut sm) = (0u8, 0u8, 0u8);
                            let (mut c0, mut c1, mut c2) = (0u8, 0u8, 0u8);
                            for k in 0..8 {
                                let off = lane * 8 + k;
                                let c = sb * 256 + off;
                                let w = if c < cols { row_src[c] } else { 0u16 };
                                base_sb[off] = (w & 0xff) as u8;
                                h0 |= (((w >> 8) & 1) as u8) << k;
                                h1 |= (((w >> 9) & 1) as u8) << k;
                                sm |= (((w >> 15) & 1) as u8) << k;
                                let exp = ((w >> 10) & 0x1f) as u8;
                                let code = rev[exp as usize];
                                if code < 7 {
                                    c0 |= (code & 1) << k;
                                    c1 |= ((code >> 1) & 1) << k;
                                    c2 |= ((code >> 2) & 1) << k;
                                } else {
                                    c0 |= 1u8 << k;
                                    c1 |= 1u8 << k;
                                    c2 |= 1u8 << k;
                                    esc_bytes.push(exp);
                                }
                            }
                            ph0[lane] = h0;
                            ph1[lane] = h1;
                            psm[lane] = sm;
                            pc0[lane] = c0;
                            pc1[lane] = c1;
                            pc2[lane] = c2;
                        }
                        pack_32_bytes_to_u32s(&ph0, &mut sb_cw[0..8]);
                        pack_32_bytes_to_u32s(&ph1, &mut sb_cw[8..16]);
                        pack_32_bytes_to_u32s(&psm, &mut sb_cw[16..24]);
                        pack_32_bytes_to_u32s(&pc0, &mut sb_cw[24..32]);
                        pack_32_bytes_to_u32s(&pc1, &mut sb_cw[32..40]);
                        pack_32_bytes_to_u32s(&pc2, &mut sb_cw[40..48]);
                    }
                }
            }
        }

        Ok(Self {
            dtype,
            rows,
            cols,
            padded_cols,
            lut,
            base_bytes,
            code_words,
            sb_esc_offsets,
            esc_bytes,
        })
    }

    pub fn compressed_bytes(&self) -> usize {
        8 + self.base_bytes.len()
            + self.code_words.len() * 4
            + self.sb_esc_offsets.len() * 4
            + self.esc_bytes.len()
    }

    pub fn bits_per_weight(&self) -> f64 {
        (self.compressed_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Reconstructs exact original `u16` words (`100%` bit-for-bit identical to input).
    pub fn decode_u16(&self) -> Vec<u16> {
        let mut out = vec![0u16; self.rows * self.cols];
        let sbs_per_row = self.padded_cols / 256;
        let wpsb = match self.dtype {
            DType::BF16 => 24,
            DType::F16 => 48,
        };

        for r in 0..self.rows {
            let dst_row = &mut out[r * self.cols..(r + 1) * self.cols];
            for sb in 0..sbs_per_row {
                let sb_idx = r * sbs_per_row + sb;
                let mut esc_pos = self.sb_esc_offsets[sb_idx] as usize;
                let base_sb = &self.base_bytes[sb_idx * 256..(sb_idx + 1) * 256];
                let sb_cw = &self.code_words[sb_idx * wpsb..(sb_idx + 1) * wpsb];

                for lane in 0..32 {
                    let col_base = sb * 256 + lane * 8;
                    if col_base >= self.cols {
                        break;
                    }
                    let valid = (self.cols - col_base).min(8);
                    match self.dtype {
                        DType::BF16 => {
                            let b0 = read_sb_plane_byte(sb_cw, 0, lane);
                            let b1 = read_sb_plane_byte(sb_cw, 1, lane);
                            let b2 = read_sb_plane_byte(sb_cw, 2, lane);
                            for k in 0..valid {
                                let code = (((b0 >> k) & 1)
                                    | (((b1 >> k) & 1) << 1)
                                    | (((b2 >> k) & 1) << 2))
                                    as usize;
                                let exp = if code == 7 {
                                    let e = self.esc_bytes[esc_pos];
                                    esc_pos += 1;
                                    e
                                } else {
                                    self.lut[code]
                                };
                                let sm = base_sb[lane * 8 + k] as u16;
                                dst_row[col_base + k] =
                                    ((sm & 0x80) << 8) | ((exp as u16) << 7) | (sm & 0x7f);
                            }
                        }
                        DType::F16 => {
                            let h0 = read_sb_plane_byte(sb_cw, 0, lane);
                            let h1 = read_sb_plane_byte(sb_cw, 1, lane);
                            let sm = read_sb_plane_byte(sb_cw, 2, lane);
                            let c0 = read_sb_plane_byte(sb_cw, 3, lane);
                            let c1 = read_sb_plane_byte(sb_cw, 4, lane);
                            let c2 = read_sb_plane_byte(sb_cw, 5, lane);
                            for k in 0..valid {
                                let code = (((c0 >> k) & 1)
                                    | (((c1 >> k) & 1) << 1)
                                    | (((c2 >> k) & 1) << 2))
                                    as usize;
                                let exp = if code == 7 {
                                    let e = self.esc_bytes[esc_pos];
                                    esc_pos += 1;
                                    e
                                } else {
                                    self.lut[code]
                                };
                                let hi = ((((sm >> k) & 1) as u16) << 15)
                                    | ((exp as u16) << 10)
                                    | ((((h1 >> k) & 1) as u16) << 9)
                                    | ((((h0 >> k) & 1) as u16) << 8);
                                dst_row[col_base + k] = hi | (base_sb[lane * 8 + k] as u16);
                            }
                        }
                    }
                }
            }
        }
        out
    }

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
                    let row_start = ci * chunk;
                    let xp = &x_pad;
                    s.spawn(move || self.gemv_rows_range(xp, y_chunk, row_start));
                }
            });
        }
    }

    fn gemv_rows_range(&self, x_pad: &[f32], y_slice: &mut [f32], row_start: usize) {
        let sbs_per_row = self.padded_cols / 256;
        let lut = self.lut;

        match self.dtype {
            DType::BF16 => {
                let mut exp_f32_lut = [0u32; 8];
                for i in 0..7 {
                    exp_f32_lut[i] = (lut[i] as u32) << 23;
                }
                for (dr, out_y) in y_slice.iter_mut().enumerate() {
                    let r = row_start + dr;
                    let (mut acc0, mut acc1, mut acc2, mut acc3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    for sb in 0..sbs_per_row {
                        let sb_idx = r * sbs_per_row + sb;
                        let mut esc_pos = self.sb_esc_offsets[sb_idx] as usize;
                        let base_sb = &self.base_bytes[sb_idx * 256..(sb_idx + 1) * 256];
                        let sb_cw = &self.code_words[sb_idx * 24..(sb_idx + 1) * 24];

                        for w in 0..8 {
                            let b0w = sb_cw[w].to_le_bytes();
                            let b1w = sb_cw[8 + w].to_le_bytes();
                            let b2w = sb_cw[16 + w].to_le_bytes();
                            let esc_w = sb_cw[w] & sb_cw[8 + w] & sb_cw[16 + w];
                            for sub in 0..4 {
                                let lane = w * 4 + sub;
                                let (b0, b1, b2) = (b0w[sub], b1w[sub], b2w[sub]);
                                let bm = &base_sb[lane * 8..lane * 8 + 8];
                                let xv = &x_pad[sb * 256 + lane * 8..sb * 256 + lane * 8 + 8];
                                if esc_w == 0 {
                                    for q in 0..2 {
                                        let k = q * 4;
                                        let c0 = (((b0 >> k) & 1)
                                            | (((b1 >> k) & 1) << 1)
                                            | (((b2 >> k) & 1) << 2))
                                            as usize;
                                        let c1 = (((b0 >> (k + 1)) & 1)
                                            | (((b1 >> (k + 1)) & 1) << 1)
                                            | (((b2 >> (k + 1)) & 1) << 2))
                                            as usize;
                                        let c2 = (((b0 >> (k + 2)) & 1)
                                            | (((b1 >> (k + 2)) & 1) << 1)
                                            | (((b2 >> (k + 2)) & 1) << 2))
                                            as usize;
                                        let c3 = (((b0 >> (k + 3)) & 1)
                                            | (((b1 >> (k + 3)) & 1) << 1)
                                            | (((b2 >> (k + 3)) & 1) << 2))
                                            as usize;
                                        acc0 = reconstruct_bf16_fast(bm[k], exp_f32_lut[c0])
                                            .mul_add(xv[k], acc0);
                                        acc1 = reconstruct_bf16_fast(bm[k + 1], exp_f32_lut[c1])
                                            .mul_add(xv[k + 1], acc1);
                                        acc2 = reconstruct_bf16_fast(bm[k + 2], exp_f32_lut[c2])
                                            .mul_add(xv[k + 2], acc2);
                                        acc3 = reconstruct_bf16_fast(bm[k + 3], exp_f32_lut[c3])
                                            .mul_add(xv[k + 3], acc3);
                                    }
                                } else {
                                    for k in 0..8 {
                                        let code = (((b0 >> k) & 1)
                                            | (((b1 >> k) & 1) << 1)
                                            | (((b2 >> k) & 1) << 2))
                                            as usize;
                                        let eb = if code == 7 {
                                            let e = self.esc_bytes[esc_pos] as u32;
                                            esc_pos += 1;
                                            e << 23
                                        } else {
                                            exp_f32_lut[code]
                                        };
                                        acc0 =
                                            reconstruct_bf16_fast(bm[k], eb).mul_add(xv[k], acc0);
                                    }
                                }
                            }
                        }
                    }
                    *out_y = (acc0 + acc1) + (acc2 + acc3);
                }
            }
            DType::F16 => {
                for (dr, out_y) in y_slice.iter_mut().enumerate() {
                    let r = row_start + dr;
                    let (mut acc0, mut acc1) = (0.0f32, 0.0f32);
                    for sb in 0..sbs_per_row {
                        let sb_idx = r * sbs_per_row + sb;
                        let mut esc_pos = self.sb_esc_offsets[sb_idx] as usize;
                        let base_sb = &self.base_bytes[sb_idx * 256..(sb_idx + 1) * 256];
                        let sb_cw = &self.code_words[sb_idx * 48..(sb_idx + 1) * 48];
                        for lane in 0..32 {
                            let h0 = read_sb_plane_byte(sb_cw, 0, lane);
                            let h1 = read_sb_plane_byte(sb_cw, 1, lane);
                            let sm = read_sb_plane_byte(sb_cw, 2, lane);
                            let c0 = read_sb_plane_byte(sb_cw, 3, lane);
                            let c1 = read_sb_plane_byte(sb_cw, 4, lane);
                            let c2 = read_sb_plane_byte(sb_cw, 5, lane);
                            let xv = &x_pad[sb * 256 + lane * 8..sb * 256 + lane * 8 + 8];
                            for k in 0..8 {
                                let code = (((c0 >> k) & 1)
                                    | (((c1 >> k) & 1) << 1)
                                    | (((c2 >> k) & 1) << 2))
                                    as usize;
                                let exp = if code == 7 {
                                    let e = self.esc_bytes[esc_pos];
                                    esc_pos += 1;
                                    e
                                } else {
                                    lut[code]
                                };
                                let bits16 = ((((sm >> k) & 1) as u16) << 15)
                                    | ((exp as u16) << 10)
                                    | ((((h1 >> k) & 1) as u16) << 9)
                                    | ((((h0 >> k) & 1) as u16) << 8)
                                    | (base_sb[lane * 8 + k] as u16);
                                let wf = f16_to_f32(bits16);
                                if (k & 1) == 0 {
                                    acc0 = wf.mul_add(xv[k], acc0);
                                } else {
                                    acc1 = wf.mul_add(xv[k], acc1);
                                }
                            }
                        }
                    }
                    *out_y = acc0 + acc1;
                }
            }
        }
    }
}

#[inline(always)]
fn reconstruct_bf16_fast(sm: u8, exp_shifted_23: u32) -> f32 {
    let sign = ((sm as u32) & 0x80) << 24;
    let mant = ((sm as u32) & 0x7f) << 16;
    f32::from_bits(sign | exp_shifted_23 | mant)
}

/// Baseline uncompressed 16-bit (`BF16`/`FP16`) CPU `GEMV` for head-to-head benchmarking.
pub fn gemv_raw16_cpu(
    dtype: DType,
    words: &[u16],
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
    threads: usize,
) {
    assert_eq!(words.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    let workers = threads.clamp(1, 256).min(rows.max(1));
    let chunk = rows.div_ceil(workers);
    std::thread::scope(|s| {
        for (ci, y_chunk) in y.chunks_mut(chunk).enumerate() {
            let r0 = ci * chunk;
            s.spawn(move || {
                for (dr, out) in y_chunk.iter_mut().enumerate() {
                    let row = &words[(r0 + dr) * cols..(r0 + dr + 1) * cols];
                    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let n4 = cols / 4;
                    match dtype {
                        DType::BF16 => {
                            for i in 0..n4 {
                                let k = i * 4;
                                a0 = bf16_to_f32(row[k]).mul_add(x[k], a0);
                                a1 = bf16_to_f32(row[k + 1]).mul_add(x[k + 1], a1);
                                a2 = bf16_to_f32(row[k + 2]).mul_add(x[k + 2], a2);
                                a3 = bf16_to_f32(row[k + 3]).mul_add(x[k + 3], a3);
                            }
                            for k in (n4 * 4)..cols {
                                a0 = bf16_to_f32(row[k]).mul_add(x[k], a0);
                            }
                        }
                        DType::F16 => {
                            for i in 0..n4 {
                                let k = i * 4;
                                a0 = f16_to_f32(row[k]).mul_add(x[k], a0);
                                a1 = f16_to_f32(row[k + 1]).mul_add(x[k + 1], a1);
                                a2 = f16_to_f32(row[k + 2]).mul_add(x[k + 2], a2);
                                a3 = f16_to_f32(row[k + 3]).mul_add(x[k + 3], a3);
                            }
                            for k in (n4 * 4)..cols {
                                a0 = f16_to_f32(row[k]).mul_add(x[k], a0);
                            }
                        }
                    }
                    *out = (a0 + a1) + (a2 + a3);
                }
            });
        }
    });
}
