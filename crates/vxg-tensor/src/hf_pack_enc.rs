//! Per-tensor quantization & calibration adapter helpers for HuggingFace `.safetensors` -> `.vxt`
//! model packing (`hf_pack`).

use std::collections::HashMap;

use crate::cb4_calib::{cb4_col_amp, cb4_target_outlier_frac, Cb4CalibPlan, Cb4TensorSource};
use crate::cb4_opt::encode_cb4_optimized;
use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::format::{PackedTensor, VxtEntry};
use crate::gguf_meta::ModelConfig;
use crate::hf_meta::HfTransform;
use crate::llm_calib::{turbo_ch_amp, TurboCalibPlan};
use crate::llm_pack::LlmPackMode;
use crate::lossless::LosslessMatrix;
use crate::safetensors::{SafeTensorMeta, SafeTensorsView};
use crate::wht2::Wht2Matrix;
use crate::wht_turbo::TurboWhtMatrix;
use crate::Result;

pub(crate) struct HfIndexedTensor {
    pub dst_name: String,
    pub shard_idx: usize,
    pub meta_idx: usize,
    pub tf: HfTransform,
}

pub(crate) struct HfCalibAdapter<'a> {
    pub views: &'a [SafeTensorsView<'a>],
    pub items: &'a [HfIndexedTensor],
    pub by_name: HashMap<&'a str, usize>,
    pub norm_plus_one: bool,
}

impl HfCalibAdapter<'_> {
    pub(crate) fn extract_transformed(&self, idx: usize) -> Result<Vec<f32>> {
        let it = &self.items[idx];
        let st = &self.views[it.shard_idx];
        let meta = &st.tensors[it.meta_idx];
        let mut data = st.extract_f32(meta)?;
        match it.tf {
            HfTransform::NormWeight if self.norm_plus_one => {
                for x in &mut data {
                    *x += 1.0;
                }
            }
            HfTransform::NegExp => {
                for x in &mut data {
                    *x = -x.exp();
                }
            }
            _ => {}
        }
        Ok(data)
    }
}

impl Cb4TensorSource for HfCalibAdapter<'_> {
    fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn writer_tensors(&self, _n_layer: usize) -> Vec<String> {
        self.items
            .iter()
            .filter(|it| {
                it.dst_name.ends_with(".ffn_down.weight")
                    || it.dst_name.ends_with(".ssm_out.weight")
                    || it.dst_name.ends_with(".attn_output.weight")
            })
            .map(|it| it.dst_name.clone())
            .collect()
    }

    fn matrix_dims(&self, name: &str) -> Option<(usize, usize)> {
        let &idx = self.by_name.get(name)?;
        let it = &self.items[idx];
        item_matrix_dims(&self.views[it.shard_idx].tensors[it.meta_idx], it.tf)
    }

    fn extract_f32(&self, name: &str) -> Option<Vec<f32>> {
        let &idx = self.by_name.get(name)?;
        self.extract_transformed(idx).ok()
    }
}

pub(crate) fn item_matrix_dims(meta: &SafeTensorMeta, tf: HfTransform) -> Option<(usize, usize)> {
    if matches!(tf, HfTransform::Flatten2D) {
        if meta.shape.len() < 2 {
            return None;
        }
        let rows = meta.shape[0];
        let cols: usize = meta.shape[1..].iter().copied().product();
        return (rows > 0 && cols > 0).then_some((rows, cols));
    }
    meta.matrix_dims()
}

fn non_empty(v: &[f32]) -> Option<&[f32]> {
    (!v.is_empty()).then_some(v)
}

pub(crate) fn encode_hf_item(
    adapter: &HfCalibAdapter<'_>,
    idx: usize,
    mode: LlmPackMode,
    cfg: &ModelConfig,
    cb4_plan: &Cb4CalibPlan,
    turbo_plan: &TurboCalibPlan,
) -> Result<VxtEntry> {
    let it = &adapter.items[idx];
    let st = &adapter.views[it.shard_idx];
    let meta: &SafeTensorMeta = &st.tensors[it.meta_idx];
    let layer_idx = it
        .dst_name
        .strip_prefix("blk.")
        .and_then(|s| s.split_once('.'))
        .and_then(|(i, _)| i.parse::<usize>().ok());

    if !matches!(it.tf, HfTransform::Identity | HfTransform::Flatten2D) {
        let mut data = adapter.extract_transformed(idx)?;
        if mode == LlmPackMode::Turbo {
            if let Some(il) = layer_idx {
                let sc = if it.dst_name.ends_with(".attn_norm.weight") {
                    turbo_plan.attn_smooth.get(il)
                } else if it.dst_name.ends_with(".post_attention_norm.weight")
                    || it.dst_name.ends_with(".ffn_norm.weight")
                {
                    turbo_plan.post_smooth.get(il)
                } else {
                    None
                };
                if let Some(scales) = sc {
                    for &(c, g) in scales {
                        if c < data.len() {
                            data[c] /= g;
                        }
                    }
                }
            }
        }
        return Ok(VxtEntry {
            name: it.dst_name.clone(),
            tensor: PackedTensor::RawF32 {
                rows: 1,
                cols: data.len().max(1),
                data,
            },
        });
    }

    let Some((rows, cols)) = item_matrix_dims(meta, it.tf) else {
        let data = st.extract_f32(meta)?;
        return Ok(VxtEntry {
            name: it.dst_name.clone(),
            tensor: PackedTensor::RawF32 {
                rows: 1,
                cols: data.len().max(1),
                data,
            },
        });
    };

    if cols < 64 || rows < 16 {
        let data = st.extract_f32(meta)?;
        return Ok(VxtEntry {
            name: it.dst_name.clone(),
            tensor: PackedTensor::RawF32 { rows, cols, data },
        });
    }

    let col_scales: &[(usize, f32)] = if mode == LlmPackMode::Turbo {
        if let Some(il) = layer_idx {
            if it.dst_name.ends_with(".attn_qkv.weight")
                || it.dst_name.ends_with(".attn_gate.weight")
                || it.dst_name.ends_with(".ssm_alpha.weight")
                || it.dst_name.ends_with(".ssm_beta.weight")
                || it.dst_name.ends_with(".attn_q.weight")
                || it.dst_name.ends_with(".attn_k.weight")
                || it.dst_name.ends_with(".attn_v.weight")
            {
                turbo_plan.attn_smooth.get(il).map_or(&[], Vec::as_slice)
            } else if it.dst_name.ends_with(".ffn_gate.weight")
                || it.dst_name.ends_with(".ffn_up.weight")
            {
                turbo_plan.post_smooth.get(il).map_or(&[], Vec::as_slice)
            } else {
                &[]
            }
        } else {
            &[]
        }
    } else {
        &[]
    };

    let force_lossless = rows < 64
        || it.dst_name == "vis.pos_embed.weight"
        || it.dst_name == "vis.patch_embed.proj.weight"
        || it.dst_name.ends_with(".ssm_alpha.weight")
        || it.dst_name.ends_with(".ssm_beta.weight");
    let effective_mode = if force_lossless { LlmPackMode::Lossless } else { mode };

    let tensor = match effective_mode {
        LlmPackMode::Lossless => {
            if col_scales.is_empty() {
                if let Some((dt, words)) = st.extract_fp16_words(meta)? {
                    PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                } else {
                    PackedTensor::RawF32 {
                        rows,
                        cols,
                        data: st.extract_f32(meta)?,
                    }
                }
            } else {
                let mut f32s = st.extract_f32(meta)?;
                for &(c, g) in col_scales {
                    if c < cols {
                        for r in 0..rows {
                            f32s[r * cols + c] *= g;
                        }
                    }
                }
                PackedTensor::RawF32 { rows, cols, data: f32s }
            }
        }
        LlmPackMode::Cb4 => {
            let f32s = st.extract_f32(meta)?;
            if it.dst_name.starts_with("vis.") {
                PackedTensor::Codebook(encode_cb4_optimized(rows, cols, &f32s, None, 0.0085)?)
            } else {
                let is_rec = layer_idx.is_some_and(|il| cfg.is_recurrent_layer(il));
                let col_amp = cb4_col_amp(cb4_plan, &it.dst_name, layer_idx, cols);
                let frac = cb4_target_outlier_frac(
                    &it.dst_name,
                    layer_idx,
                    cfg.n_layer,
                    is_rec,
                    cb4_plan.has_output_weight,
                );
                PackedTensor::Codebook(encode_cb4_optimized(rows, cols, &f32s, col_amp, frac)?)
            }
        }
        LlmPackMode::Cb3 => {
            PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &st.extract_f32(meta)?)?)
        }
        LlmPackMode::Wht2 => {
            let f32s = st.extract_f32(meta)?;
            let is_wht2 = rows <= 65_536
                && (it.dst_name.contains(".ffn_")
                    || it.dst_name.ends_with(".attn_q.weight")
                    || it.dst_name.ends_with(".attn_output.weight"));
            if is_wht2 {
                PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &f32s)?)
            } else {
                PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &f32s)?)
            }
        }
        LlmPackMode::Turbo => {
            let mut f32s = st.extract_f32(meta)?;
            for &(c, g) in col_scales {
                if c < cols {
                    for r in 0..rows {
                        f32s[r * cols + c] *= g;
                    }
                }
            }
            let inv_cols = 1.0 / (cols.max(1) as f64);
            let row_ms: Vec<f32> = (0..rows)
                .map(|r| {
                    let row = &f32s[r * cols..(r + 1) * cols];
                    let s2: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    ((s2 * inv_cols) as f32).max(1e-12)
                })
                .collect();
            let mut sorted = row_ms.clone();
            sorted.sort_by(f32::total_cmp);
            let mat_ms = sorted.get(rows / 2).copied().unwrap_or(1e-6).max(1e-12);
            let ch_amp = turbo_ch_amp(turbo_plan, &it.dst_name, layer_idx, cols);

            if it.dst_name.starts_with("vis.") {
                PackedTensor::TurboWht(TurboWhtMatrix::encode_with_hessian(
                    4,
                    rows,
                    cols,
                    &f32s,
                    None,
                    |r| if row_ms[r] > 3.2 * mat_ms { 0.0 } else { 0.030 },
                )?)
            } else if it.dst_name == "token_embd.weight" || it.dst_name == "output.weight" {
                let h = if it.dst_name == "token_embd.weight" {
                    &turbo_plan.h_embd
                } else {
                    &turbo_plan.h_lm_head
                };
                let head_bits = if turbo_plan.ch_plan.is_some() { 3 } else { 4 };
                let lam = if head_bits == 3 {
                    0.0
                } else if it.dst_name == "token_embd.weight" {
                    0.052
                } else {
                    0.040
                };
                PackedTensor::TurboWht(TurboWhtMatrix::encode_with_ch_amp(
                    head_bits,
                    rows,
                    cols,
                    &f32s,
                    non_empty(h),
                    ch_amp,
                    |r| if row_ms[r] > 3.0 * mat_ms { 0.0 } else { lam },
                )?)
            } else {
                let n_lay = cfg.n_layer.max(1);
                let h_rot = layer_idx.and_then(|il| {
                    if it.dst_name.ends_with(".attn_qkv.weight")
                        || it.dst_name.ends_with(".attn_gate.weight")
                        || it.dst_name.ends_with(".attn_q.weight")
                        || it.dst_name.ends_with(".attn_k.weight")
                        || it.dst_name.ends_with(".attn_v.weight")
                    {
                        turbo_plan.h_attn.get(il).and_then(|v| non_empty(v))
                    } else if it.dst_name.ends_with(".ssm_out.weight") {
                        turbo_plan.h_ssm_out.get(il).and_then(|v| non_empty(v))
                    } else if it.dst_name.ends_with(".ffn_gate.weight")
                        || it.dst_name.ends_with(".ffn_up.weight")
                    {
                        turbo_plan.h_ffn_gu.get(il).and_then(|v| non_empty(v))
                    } else {
                        None
                    }
                });
                let is_edge = layer_idx.is_some_and(|il| il <= 1 || il + 2 >= n_lay);
                let is_anchor = is_edge || layer_idx.is_some_and(|il| il % 4 == 3);
                let base_lam: f32 = if it.dst_name.ends_with(".attn_q.weight")
                    || it.dst_name.ends_with(".attn_k.weight")
                    || it.dst_name.ends_with(".attn_v.weight")
                    || it.dst_name.ends_with(".attn_output.weight")
                    || it.dst_name.ends_with(".ssm_out.weight")
                {
                    0.026
                } else if it.dst_name.ends_with(".attn_qkv.weight")
                    || it.dst_name.ends_with(".attn_gate.weight")
                {
                    if is_edge { 0.036 } else { 0.044 }
                } else if it.dst_name.ends_with(".ffn_down.weight") {
                    if is_anchor { 0.038 } else { 0.052 }
                } else if it.dst_name.ends_with(".ffn_gate.weight")
                    || it.dst_name.ends_with(".ffn_up.weight")
                {
                    if is_anchor { 0.045 } else { 0.074 }
                } else {
                    0.040
                };
                PackedTensor::TurboWht(TurboWhtMatrix::encode_with_ch_amp(
                    4,
                    rows,
                    cols,
                    &f32s,
                    h_rot,
                    ch_amp,
                    |r| {
                        if row_ms[r] > 3.2 * mat_ms {
                            0.0
                        } else {
                            base_lam * (mat_ms / row_ms[r]).clamp(0.45, 1.50)
                        }
                    },
                )?)
            }
        }
        LlmPackMode::Q4_0 => {
            PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &st.extract_f32(meta)?))
        }
    };

    Ok(VxtEntry {
        name: it.dst_name.clone(),
        tensor,
    })
}
