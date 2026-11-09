//! `Qwen3_5VisionModel` (`ViT` + 2D Axial `RoPE` + `2x2` `PatchMerger`) encoder and 3D `M-RoPE`
//! multimodal position/attention helpers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::format::PackedTensor;
pub use crate::vision_img::{apply_mrope_neox, compute_mrope_positions, step_full_gqa_mrope};
use crate::vision_img::{load_image_source, preprocess_image_bytes, PreprocessedImage, VisionConfig};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct EncodedVisionImage {
    pub embeds: Vec<f32>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub llm_tokens: usize,
}

struct VisionBlock {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    qkv_w: PackedTensor,
    qkv_b: Vec<f32>,
    proj_w: PackedTensor,
    proj_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    fc1_w: PackedTensor,
    fc1_b: Vec<f32>,
    fc2_w: PackedTensor,
    fc2_b: Vec<f32>,
}

pub struct VisionEncoder {
    pub cfg: VisionConfig,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    pos_embed: Vec<f32>,
    blocks: Vec<VisionBlock>,
    merger_norm_w: Vec<f32>,
    merger_norm_b: Vec<f32>,
    merger_fc1_w: PackedTensor,
    merger_fc1_b: Vec<f32>,
    merger_fc2_w: PackedTensor,
    merger_fc2_b: Vec<f32>,
}

#[inline]
fn gelu_pytorch_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

#[inline]
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

#[inline]
fn gelu_exact(x: f32) -> f32 {
    const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    0.5 * x * (1.0 + erf_approx(x * INV_SQRT2))
}

fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], dim: usize, eps: f32, out: &mut [f32]) {
    let inv_d = 1.0 / (dim as f32);
    for (src, dst) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
        let mean = src.iter().copied().sum::<f32>() * inv_d;
        let mut var = 0.0f32;
        for &v in src {
            let d = v - mean;
            var = d.mul_add(d, var);
        }
        let inv_std = 1.0 / (var * inv_d + eps).sqrt();
        for i in 0..dim {
            dst[i] = ((src[i] - mean) * inv_std).mul_add(w[i], b[i]);
        }
    }
}

fn decode_packed_par(t: &PackedTensor, threads: usize, buf: &mut Vec<f32>) {
    let (rows, cols) = (t.rows(), t.cols());
    buf.resize(rows * cols, 0.0);
    if let PackedTensor::RawF32 { data, .. } = t {
        buf.copy_from_slice(data);
        return;
    }
    let workers = threads.clamp(1, 16).min(rows.max(1));
    if workers <= 1 || rows < 64 {
        for r in 0..rows {
            t.decode_row(r, &mut buf[r * cols..(r + 1) * cols]);
        }
        return;
    }
    let chunk_rows = rows.div_ceil(workers);
    std::thread::scope(|s| {
        for (ci, chunk) in buf.chunks_mut(chunk_rows * cols).enumerate() {
            let r0 = ci * chunk_rows;
            let n_r = chunk.len() / cols;
            s.spawn(move || {
                for dr in 0..n_r {
                    t.decode_row(r0 + dr, &mut chunk[dr * cols..(dr + 1) * cols]);
                }
            });
        }
    });
}

#[cfg(vxt_has_metal)]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// Computes `y[n, m] = sum_k x[n, k] * w[m, k] + bias[m]` for `x: [n_rows, k_dim]`, `w: [m_dim, k_dim]`.
#[cfg_attr(vxt_has_metal, allow(unused_variables, unreachable_code))]
fn dense_linear_par(
    x: &[f32],
    n_rows: usize,
    k_dim: usize,
    w: &[f32],
    bias: &[f32],
    m_dim: usize,
    threads: usize,
    y: &mut [f32],
) {
    #[cfg(vxt_has_metal)]
    {
        if n_rows > 0 && m_dim > 0 && k_dim > 0 {
            for r in 0..n_rows {
                y[r * m_dim..(r + 1) * m_dim].copy_from_slice(bias);
            }
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    112,
                    n_rows as i32,
                    m_dim as i32,
                    k_dim as i32,
                    1.0,
                    x.as_ptr(),
                    k_dim as i32,
                    w.as_ptr(),
                    k_dim as i32,
                    1.0,
                    y.as_mut_ptr(),
                    m_dim as i32,
                );
            }
        }
        return;
    }
    let workers = threads.clamp(1, 16).min(n_rows.max(1));
    if workers <= 1 || n_rows < 8 {
        for r in 0..n_rows {
            let xr = &x[r * k_dim..(r + 1) * k_dim];
            let yr = &mut y[r * m_dim..(r + 1) * m_dim];
            for m in 0..m_dim {
                let wm = &w[m * k_dim..(m + 1) * k_dim];
                let mut acc = bias[m];
                for k in 0..k_dim {
                    acc = xr[k].mul_add(wm[k], acc);
                }
                yr[m] = acc;
            }
        }
        return;
    }
    let next_r = AtomicUsize::new(0);
    let step = 4usize;
    let ptr_addr = y.as_mut_ptr() as usize;
    std::thread::scope(|s| {
        for _ in 0..workers {
            let next_ref = &next_r;
            s.spawn(move || loop {
                let r0 = next_ref.fetch_add(step, Ordering::Relaxed);
                if r0 >= n_rows {
                    break;
                }
                let r_end = (r0 + step).min(n_rows);
                for r in r0..r_end {
                    let xr = &x[r * k_dim..(r + 1) * k_dim];
                    let yr = unsafe {
                        std::slice::from_raw_parts_mut(
                            (ptr_addr as *mut f32).add(r * m_dim),
                            m_dim,
                        )
                    };
                    for m in 0..m_dim {
                        let wm = &w[m * k_dim..(m + 1) * k_dim];
                        let mut acc = bias[m];
                        for k in 0..k_dim {
                            acc = xr[k].mul_add(wm[k], acc);
                        }
                        yr[m] = acc;
                    }
                }
            });
        }
    });
}

impl VisionEncoder {
    pub fn from_tensors(cfg: VisionConfig, map: &mut HashMap<String, PackedTensor>) -> Result<Self> {
        fn take_mat(m: &mut HashMap<String, PackedTensor>, name: &str) -> Result<PackedTensor> {
            m.remove(name)
                .ok_or_else(|| Error::Format(format!("missing vision tensor '{name}' in VXT")))
        }
        fn take_vec(m: &mut HashMap<String, PackedTensor>, name: &str) -> Result<Vec<f32>> {
            Ok(take_mat(m, name)?.decode_f32())
        }

        let patch_w = take_mat(map, "vis.patch_embed.proj.weight")?.decode_f32();
        let patch_b = take_vec(map, "vis.patch_embed.proj.bias")?;
        let pos_embed = take_mat(map, "vis.pos_embed.weight")?.decode_f32();
        if patch_w.len() != cfg.hidden_size * cfg.patch_dim() {
            return Err(Error::InvalidShape("vis.patch_embed.proj.weight shape mismatch"));
        }
        if pos_embed.len() != cfg.num_position_embeddings * cfg.hidden_size {
            return Err(Error::InvalidShape("vis.pos_embed.weight shape mismatch"));
        }

        let mut blocks = Vec::with_capacity(cfg.depth);
        for il in 0..cfg.depth {
            let p = format!("vis.blocks.{il}");
            blocks.push(VisionBlock {
                norm1_w: take_vec(map, &format!("{p}.norm1.weight"))?,
                norm1_b: take_vec(map, &format!("{p}.norm1.bias"))?,
                qkv_w: take_mat(map, &format!("{p}.attn.qkv.weight"))?,
                qkv_b: take_vec(map, &format!("{p}.attn.qkv.bias"))?,
                proj_w: take_mat(map, &format!("{p}.attn.proj.weight"))?,
                proj_b: take_vec(map, &format!("{p}.attn.proj.bias"))?,
                norm2_w: take_vec(map, &format!("{p}.norm2.weight"))?,
                norm2_b: take_vec(map, &format!("{p}.norm2.bias"))?,
                fc1_w: take_mat(map, &format!("{p}.mlp.linear_fc1.weight"))?,
                fc1_b: take_vec(map, &format!("{p}.mlp.linear_fc1.bias"))?,
                fc2_w: take_mat(map, &format!("{p}.mlp.linear_fc2.weight"))?,
                fc2_b: take_vec(map, &format!("{p}.mlp.linear_fc2.bias"))?,
            });
        }

        Ok(Self {
            patch_w,
            patch_b,
            pos_embed,
            blocks,
            merger_norm_w: take_vec(map, "vis.merger.norm.weight")?,
            merger_norm_b: take_vec(map, "vis.merger.norm.bias")?,
            merger_fc1_w: take_mat(map, "vis.merger.linear_fc1.weight")?,
            merger_fc1_b: take_vec(map, "vis.merger.linear_fc1.bias")?,
            merger_fc2_w: take_mat(map, "vis.merger.linear_fc2.weight")?,
            merger_fc2_b: take_vec(map, "vis.merger.linear_fc2.bias")?,
            cfg,
        })
    }

    pub fn encode_source(&self, src: &str, threads: usize) -> Result<EncodedVisionImage> {
        let bytes = load_image_source(src)?;
        let pre = preprocess_image_bytes(&bytes, &self.cfg)?;
        self.encode_preprocessed(&pre, threads)
    }

    pub fn encode_preprocessed(
        &self,
        img: &PreprocessedImage,
        threads: usize,
    ) -> Result<EncodedVisionImage> {
        let c = &self.cfg;
        let seq = img.num_patches();
        let d = c.hidden_size;
        let m = c.spatial_merge_size;
        let m_unit = c.merge_unit();
        if seq == 0 || seq % m_unit != 0 {
            return Err(Error::InvalidShape("vision patch count not divisible by merge_unit"));
        }

        let mut hidden = vec![0.0f32; seq * d];
        dense_linear_par(
            &img.patches,
            seq,
            c.patch_dim(),
            &self.patch_w,
            &self.patch_b,
            d,
            threads,
            &mut hidden,
        );

        // Add bilinear-interpolated `pos_embed` (`align_corners = true`) + build 2D axial RoPE
        let side = (c.num_position_embeddings as f64).sqrt().round() as usize;
        let side_max = side.saturating_sub(1) as f32;
        let gh_denom = img.grid_h.saturating_sub(1).max(1) as f32;
        let gw_denom = img.grid_w.saturating_sub(1).max(1) as f32;
        let blocks_w = img.grid_w / m;
        let spatial_dim = c.head_dim / 2;
        let quarter = spatial_dim / 2;
        let inv_freq: Vec<f32> = (0..quarter)
            .map(|k| c.rope_theta.powf(-(2.0 * (k as f32)) / (spatial_dim as f32)))
            .collect();

        let mut rope_cos = vec![0.0f32; seq * spatial_dim];
        let mut rope_sin = vec![0.0f32; seq * spatial_dim];
        let spatial_patches = img.grid_h * img.grid_w;

        for idx in 0..seq {
            let within = idx % spatial_patches;
            let in_col = within % m;
            let in_row = (within / m) % m;
            let b_col = (within / m_unit) % blocks_w;
            let b_row = within / (m_unit * blocks_w);
            let row = b_row * m + in_row;
            let col = b_col * m + in_col;

            let sh = if img.grid_h > 1 { (row as f32) * side_max / gh_denom } else { 0.0 };
            let sw = if img.grid_w > 1 { (col as f32) * side_max / gw_denom } else { 0.0 };
            let h0 = (sh.floor() as usize).min(side - 1);
            let h1 = (h0 + 1).min(side - 1);
            let w0 = (sw.floor() as usize).min(side - 1);
            let w1 = (w0 + 1).min(side - 1);
            let dh = sh - (h0 as f32);
            let dw = sw - (w0 as f32);
            let taps = [
                ((1.0 - dh) * (1.0 - dw), h0 * side + w0),
                ((1.0 - dh) * dw, h0 * side + w1),
                (dh * (1.0 - dw), h1 * side + w0),
                (dh * dw, h1 * side + w1),
            ];
            let hs = &mut hidden[idx * d..(idx + 1) * d];
            for &(wt, pidx) in &taps {
                if wt > 0.0 {
                    let pe = &self.pos_embed[pidx * d..(pidx + 1) * d];
                    for j in 0..d {
                        hs[j] = wt.mul_add(pe[j], hs[j]);
                    }
                }
            }

            let cs = &mut rope_cos[idx * spatial_dim..(idx + 1) * spatial_dim];
            let sn = &mut rope_sin[idx * spatial_dim..(idx + 1) * spatial_dim];
            for k in 0..quarter {
                let th_r = (row as f32) * inv_freq[k];
                let th_c = (col as f32) * inv_freq[k];
                let (sr, cr) = th_r.sin_cos();
                let (sc, cc) = th_c.sin_cos();
                cs[k] = cr;
                sn[k] = sr;
                cs[quarter + k] = cc;
                sn[quarter + k] = sc;
            }
        }

        let mut normed = vec![0.0f32; seq * d];
        let mut qkv = vec![0.0f32; seq * 3 * d];
        let mut attn_cat = vec![0.0f32; seq * d];
        let mut proj_out = vec![0.0f32; seq * d];
        let mut fc1_out = vec![0.0f32; seq * c.intermediate_size];
        let mut w_buf = Vec::new();
        let nh = c.num_heads;
        let hd = c.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        for blk in &self.blocks {
            layer_norm_rows(&hidden, &blk.norm1_w, &blk.norm1_b, d, 1e-6, &mut normed);
            decode_packed_par(&blk.qkv_w, threads, &mut w_buf);
            dense_linear_par(&normed, seq, d, &w_buf, &blk.qkv_b, 3 * d, threads, &mut qkv);

            // Apply 2D axial vision RoPE to Q (`0..d`) and K (`d..2*d`)
            for s in 0..seq {
                let cs = &rope_cos[s * spatial_dim..(s + 1) * spatial_dim];
                let sn = &rope_sin[s * spatial_dim..(s + 1) * spatial_dim];
                let row_qkv = &mut qkv[s * 3 * d..(s + 1) * 3 * d];
                for q_or_k in 0..2 {
                    let base = q_or_k * d;
                    for h in 0..nh {
                        let hv = &mut row_qkv[base + h * hd..base + (h + 1) * hd];
                        for i in 0..spatial_dim {
                            let (x0, x1) = (hv[i], hv[i + spatial_dim]);
                            hv[i] = x0 * cs[i] - x1 * sn[i];
                            hv[i + spatial_dim] = x1 * cs[i] + x0 * sn[i];
                        }
                    }
                }
            }

            // Bidirectional multi-head self-attention parallelized across `nh` heads
            let attn_ptr = attn_cat.as_mut_ptr() as usize;
            let qkv_ref = &qkv;
            let next_h = AtomicUsize::new(0);
            let h_workers = threads.clamp(1, 16).min(nh);
            std::thread::scope(|sp| {
                for _ in 0..h_workers {
                    let nh_ref = &next_h;
                    sp.spawn(move || {
                        let mut scores = vec![0.0f32; seq];
                        loop {
                            let h = nh_ref.fetch_add(1, Ordering::Relaxed);
                            if h >= nh {
                                break;
                            }
                            let q_off = h * hd;
                            let k_off = d + h * hd;
                            let v_off = 2 * d + h * hd;
                            for qi in 0..seq {
                                let qh = &qkv_ref[qi * 3 * d + q_off..qi * 3 * d + q_off + hd];
                                let mut max_s = f32::NEG_INFINITY;
                                for ki in 0..seq {
                                    let kh = &qkv_ref[ki * 3 * d + k_off..ki * 3 * d + k_off + hd];
                                    let mut dot = 0.0f32;
                                    for j in 0..hd {
                                        dot = qh[j].mul_add(kh[j], dot);
                                    }
                                    let sv = dot * scale;
                                    scores[ki] = sv;
                                    if sv > max_s {
                                        max_s = sv;
                                    }
                                }
                                let mut sum_e = 0.0f32;
                                for sv in &mut scores {
                                    *sv = (*sv - max_s).exp();
                                    sum_e += *sv;
                                }
                                let inv_s = 1.0 / sum_e;
                                let out_h = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        (attn_ptr as *mut f32).add(qi * d + q_off),
                                        hd,
                                    )
                                };
                                out_h.fill(0.0);
                                for ki in 0..seq {
                                    let wt = scores[ki] * inv_s;
                                    let vh = &qkv_ref[ki * 3 * d + v_off..ki * 3 * d + v_off + hd];
                                    for j in 0..hd {
                                        out_h[j] = wt.mul_add(vh[j], out_h[j]);
                                    }
                                }
                            }
                        }
                    });
                }
            });

            decode_packed_par(&blk.proj_w, threads, &mut w_buf);
            dense_linear_par(&attn_cat, seq, d, &w_buf, &blk.proj_b, d, threads, &mut proj_out);
            for (h_v, &p_v) in hidden.iter_mut().zip(&proj_out) {
                *h_v += p_v;
            }

            layer_norm_rows(&hidden, &blk.norm2_w, &blk.norm2_b, d, 1e-6, &mut normed);
            decode_packed_par(&blk.fc1_w, threads, &mut w_buf);
            dense_linear_par(
                &normed,
                seq,
                d,
                &w_buf,
                &blk.fc1_b,
                c.intermediate_size,
                threads,
                &mut fc1_out,
            );
            for v in &mut fc1_out {
                *v = gelu_pytorch_tanh(*v);
            }
            decode_packed_par(&blk.fc2_w, threads, &mut w_buf);
            dense_linear_par(
                &fc1_out,
                seq,
                c.intermediate_size,
                &w_buf,
                &blk.fc2_b,
                d,
                threads,
                &mut proj_out,
            );
            for (h_v, &p_v) in hidden.iter_mut().zip(&proj_out) {
                *h_v += p_v;
            }
        }

        // `PatchMerger`: LayerNorm(hidden_size) per patch -> reshape [seq/4, 4*d] -> fc1 -> GELU -> fc2
        layer_norm_rows(
            &hidden,
            &self.merger_norm_w,
            &self.merger_norm_b,
            d,
            1e-6,
            &mut normed,
        );
        let llm_tokens = seq / m_unit;
        let merged_dim = d * m_unit;
        let mut m_fc1 = vec![0.0f32; llm_tokens * merged_dim];
        decode_packed_par(&self.merger_fc1_w, threads, &mut w_buf);
        dense_linear_par(
            &normed,
            llm_tokens,
            merged_dim,
            &w_buf,
            &self.merger_fc1_b,
            merged_dim,
            threads,
            &mut m_fc1,
        );
        for v in &mut m_fc1 {
            *v = gelu_exact(*v);
        }
        let mut embeds = vec![0.0f32; llm_tokens * c.out_hidden_size];
        decode_packed_par(&self.merger_fc2_w, threads, &mut w_buf);
        dense_linear_par(
            &m_fc1,
            llm_tokens,
            merged_dim,
            &w_buf,
            &self.merger_fc2_b,
            c.out_hidden_size,
            threads,
            &mut embeds,
        );

        Ok(EncodedVisionImage {
            embeds,
            grid_t: img.grid_t,
            grid_h: img.grid_h,
            grid_w: img.grid_w,
            llm_tokens,
        })
    }
}
