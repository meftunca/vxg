//! Per-step CPU / per-matrix-Metal execution, weight holders, multimodal embedding splicing,
//! and decoded-raw archive diagnostics extracted from [`crate::llm`] (`<= 600` lines).

use std::collections::HashMap;

use crate::format::PackedTensor;
use crate::gguf_meta::ModelConfig;
use crate::llm::{LlmModel, LlmState};
use crate::llm_ops::{rms_norm, silu, step_deltanet_recurrent, step_full_gqa};
use crate::metal::{GpuPackedTensor, MetalEngine};
use crate::vision::{step_full_gqa_mrope, EncodedVisionImage};
use crate::{Error, Result};

pub(crate) struct MatWeight {
    pub(crate) cpu: PackedTensor,
    pub(crate) gpu: Option<GpuPackedTensor>,
}

impl MatWeight {
    pub(crate) fn cpu_only(cpu: PackedTensor) -> Self {
        Self { cpu, gpu: None }
    }
    pub(crate) fn upload_gpu(&mut self, eng: &MetalEngine) -> Result<()> {
        if self.gpu.is_none() {
            self.gpu = Some(eng.upload_packed(&self.cpu)?);
        }
        Ok(())
    }
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.cpu.rows()
    }
    #[inline]
    pub(crate) fn forward(
        &self,
        x: &[f32],
        y: &mut [f32],
        metal: Option<&mut MetalEngine>,
        threads: usize,
    ) -> Result<()> {
        if let (Some(eng), Some(g)) = (metal, self.gpu.as_ref()) {
            let _ = g.run_gemv(eng, x, y, 1)?;
        } else {
            self.cpu.gemv_cpu(x, y, threads);
        }
        Ok(())
    }
}

pub(crate) fn forward_multi(
    x: &[f32],
    jobs: &mut [(&MatWeight, &mut [f32])],
    metal: Option<&mut MetalEngine>,
    threads: usize,
) -> Result<()> {
    if let Some(eng) = metal {
        if jobs.iter().all(|(w, _)| w.gpu.is_some()) {
            let mut gpu_jobs: Vec<(&GpuPackedTensor, &mut [f32])> = jobs
                .iter_mut()
                .map(|(w, y)| (w.gpu.as_ref().unwrap(), &mut **y))
                .collect();
            let _ = eng.run_multi_same_x(x, &mut gpu_jobs)?;
            return Ok(());
        }
    }
    for (w, y) in jobs.iter_mut() {
        w.cpu.gemv_cpu(x, y, threads);
    }
    Ok(())
}

pub(crate) enum LayerAttn {
    LinearDeltaNet {
        wqkv: MatWeight,
        wqkv_gate: MatWeight,
        ssm_alpha: MatWeight,
        ssm_beta: MatWeight,
        ssm_out: MatWeight,
        ssm_conv1d: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_dt_bias: Vec<f32>,
        ssm_norm: Vec<f32>,
    },
    FullGqa {
        wq: MatWeight,
        wk: MatWeight,
        wv: MatWeight,
        wo: MatWeight,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        q_bias: Option<Vec<f32>>,
        k_bias: Option<Vec<f32>>,
        v_bias: Option<Vec<f32>>,
        gated_q: bool,
    },
}

pub(crate) struct DecoderLayer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) post_attn_norm: Vec<f32>,
    pub(crate) attn: LayerAttn,
    pub(crate) ffn_gate: MatWeight,
    pub(crate) ffn_up: MatWeight,
    pub(crate) ffn_down: MatWeight,
}

pub(crate) fn extract_decoder_layers(
    cfg: &ModelConfig,
    map: &mut HashMap<String, PackedTensor>,
) -> Result<Vec<DecoderLayer>> {
    let take_mat = |m: &mut HashMap<String, PackedTensor>, name: &str| -> Result<MatWeight> {
        let t = m
            .remove(name)
            .ok_or_else(|| Error::Format(format!("missing model tensor '{name}'")))?;
        Ok(MatWeight::cpu_only(t))
    };
    let take_vec = |m: &mut HashMap<String, PackedTensor>, name: &str| -> Result<Vec<f32>> {
        let t = m
            .remove(name)
            .ok_or_else(|| Error::Format(format!("missing model vector '{name}'")))?;
        Ok(t.decode_f32())
    };
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for il in 0..cfg.n_layer {
        let pfx = format!("blk.{il}");
        let attn_norm = take_vec(map, &format!("{pfx}.attn_norm.weight"))?;
        let post_attn_norm = map
            .remove(&format!("{pfx}.post_attention_norm.weight"))
            .or_else(|| map.remove(&format!("{pfx}.ffn_norm.weight")))
            .ok_or_else(|| Error::Format(format!("missing post-attn norm in layer {il}")))?
            .decode_f32();
        let attn = if cfg.is_recurrent_layer(il) {
            LayerAttn::LinearDeltaNet {
                wqkv: take_mat(map, &format!("{pfx}.attn_qkv.weight"))?,
                wqkv_gate: take_mat(map, &format!("{pfx}.attn_gate.weight"))?,
                ssm_alpha: take_mat(map, &format!("{pfx}.ssm_alpha.weight"))?,
                ssm_beta: take_mat(map, &format!("{pfx}.ssm_beta.weight"))?,
                ssm_out: take_mat(map, &format!("{pfx}.ssm_out.weight"))?,
                ssm_conv1d: take_vec(map, &format!("{pfx}.ssm_conv1d.weight"))?,
                ssm_a: take_vec(map, &format!("{pfx}.ssm_a"))?,
                ssm_dt_bias: take_vec(map, &format!("{pfx}.ssm_dt.bias"))?,
                ssm_norm: take_vec(map, &format!("{pfx}.ssm_norm.weight"))?,
            }
        } else {
            let wq = take_mat(map, &format!("{pfx}.attn_q.weight"))?;
            let wk = take_mat(map, &format!("{pfx}.attn_k.weight"))?;
            let wv = take_mat(map, &format!("{pfx}.attn_v.weight"))?;
            let wo = take_mat(map, &format!("{pfx}.attn_output.weight"))?;
            let gated_q = wq.rows() == cfg.n_head * cfg.head_dim * 2;
            LayerAttn::FullGqa {
                wq,
                wk,
                wv,
                wo,
                q_norm: map.remove(&format!("{pfx}.attn_q_norm.weight")).map(|t| t.decode_f32()),
                k_norm: map.remove(&format!("{pfx}.attn_k_norm.weight")).map(|t| t.decode_f32()),
                q_bias: map.remove(&format!("{pfx}.attn_q.bias")).map(|t| t.decode_f32()),
                k_bias: map.remove(&format!("{pfx}.attn_k.bias")).map(|t| t.decode_f32()),
                v_bias: map.remove(&format!("{pfx}.attn_v.bias")).map(|t| t.decode_f32()),
                gated_q,
            }
        };
        layers.push(DecoderLayer {
            attn_norm,
            post_attn_norm,
            attn,
            ffn_gate: take_mat(map, &format!("{pfx}.ffn_gate.weight"))?,
            ffn_up: take_mat(map, &format!("{pfx}.ffn_up.weight"))?,
            ffn_down: take_mat(map, &format!("{pfx}.ffn_down.weight"))?,
        });
    }
    Ok(layers)
}

pub(crate) fn build_multimodal_embeds(
    token_embd: &PackedTensor,
    n_vocab: usize,
    n_embd: usize,
    tok_ids: &[u32],
    images: &[&EncodedVisionImage],
    image_token_id: u32,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; tok_ids.len() * n_embd];
    let mut i = 0usize;
    let mut img_idx = 0usize;
    while i < tok_ids.len() {
        let tid = tok_ids[i];
        if tid == image_token_id {
            let img = *images.get(img_idx).ok_or_else(|| {
                Error::Format("more <|image_pad|> spans than encoded images".into())
            })?;
            if img.embeds.len() != img.llm_tokens * n_embd {
                return Err(Error::InvalidShape(
                    "EncodedVisionImage out_hidden_size != language n_embd",
                ));
            }
            let n_tok = img.llm_tokens;
            if i + n_tok > tok_ids.len()
                || tok_ids[i..i + n_tok].iter().any(|&t| t != image_token_id)
            {
                return Err(Error::Format(
                    "<|image_pad|> span length mismatch against EncodedVisionImage".into(),
                ));
            }
            out[i * n_embd..(i + n_tok) * n_embd].copy_from_slice(&img.embeds);
            img_idx += n_tok.min(1);
            i += n_tok;
        } else {
            if (tid as usize) >= n_vocab {
                return Err(Error::Format(format!("token id {tid} out of vocab {n_vocab}")));
            }
            token_embd.decode_row(tid as usize, &mut out[i * n_embd..(i + 1) * n_embd]);
            i += 1;
        }
    }
    if img_idx != images.len() {
        return Err(Error::Format(
            "fewer <|image_pad|> spans than provided images".into(),
        ));
    }
    Ok(out)
}

pub(crate) fn forward_embed_step(
    model: &mut LlmModel,
    state: &mut LlmState,
    cur_in: &[f32],
    pos_thw: [u32; 3],
    mrope_section: Option<[usize; 3]>,
    mut logits_out: Option<&mut [f32]>,
) -> Result<()> {
    let dim = model.cfg.n_embd;
    let eps = model.cfg.rms_norm_eps;
    let threads = model.cpu_threads;

    let mut cur = cur_in.to_vec();
    let mut normed = vec![0.0f32; dim];
    let mut attn_out = vec![0.0f32; dim];
    let mut ffn_gate_buf = vec![0.0f32; model.cfg.n_ff];
    let mut ffn_up_buf = vec![0.0f32; model.cfg.n_ff];
    let mut ffn_out = vec![0.0f32; dim];

    for il in 0..model.cfg.n_layer {
        let layer = &model.layers[il];
        rms_norm(&cur, &layer.attn_norm, eps, &mut normed);

        match &layer.attn {
            LayerAttn::LinearDeltaNet {
                wqkv,
                wqkv_gate,
                ssm_alpha,
                ssm_beta,
                ssm_out,
                ssm_conv1d,
                ssm_a,
                ssm_dt_bias,
                ssm_norm,
            } => {
                let d_inner = model.cfg.ssm_inner_size;
                let n_vh = model.cfg.ssm_dt_rank;
                let mut qkv_mixed = vec![0.0f32; wqkv.rows()];
                let mut z = vec![0.0f32; d_inner];
                let mut alpha_raw = vec![0.0f32; n_vh];
                let mut beta_raw = vec![0.0f32; n_vh];
                let mut y_concat = vec![0.0f32; d_inner];

                forward_multi(
                    &normed,
                    &mut [
                        (wqkv, &mut qkv_mixed),
                        (wqkv_gate, &mut z),
                        (ssm_alpha, &mut alpha_raw),
                        (ssm_beta, &mut beta_raw),
                    ],
                    model.metal.as_mut(),
                    threads,
                )?;
                step_deltanet_recurrent(
                    &qkv_mixed,
                    &z,
                    &alpha_raw,
                    &beta_raw,
                    ssm_conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    &mut state.conv_states[il],
                    &mut state.ssm_states[il],
                    model.cfg.ssm_group_count,
                    n_vh,
                    model.cfg.ssm_state_size,
                    d_inner / n_vh,
                    model.cfg.ssm_conv_kernel,
                    eps,
                    &mut y_concat,
                );
                ssm_out.forward(&y_concat, &mut attn_out, model.metal.as_mut(), threads)?;
            }
            LayerAttn::FullGqa {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                q_bias,
                k_bias,
                v_bias,
                gated_q,
            } => {
                let (nh, nkv, hd) = (model.cfg.n_head, model.cfg.n_head_kv, model.cfg.head_dim);
                let mut q_raw = vec![0.0f32; wq.rows()];
                let mut k_cur = vec![0.0f32; nkv * hd];
                let mut v_cur = vec![0.0f32; nkv * hd];
                let mut att_cat = vec![0.0f32; nh * hd];

                forward_multi(
                    &normed,
                    &mut [(wq, &mut q_raw), (wk, &mut k_cur), (wv, &mut v_cur)],
                    model.metal.as_mut(),
                    threads,
                )?;
                for (dst, bias) in [
                    (&mut q_raw[..], q_bias),
                    (&mut k_cur[..], k_bias),
                    (&mut v_cur[..], v_bias),
                ] {
                    if let Some(b) = bias {
                        for (v, &bv) in dst.iter_mut().zip(b) {
                            *v += bv;
                        }
                    }
                }
                if let Some(msec) = mrope_section {
                    step_full_gqa_mrope(
                        &q_raw,
                        &mut k_cur,
                        &v_cur,
                        q_norm.as_deref(),
                        k_norm.as_deref(),
                        *gated_q,
                        nh,
                        nkv,
                        hd,
                        state.pos,
                        pos_thw,
                        msec,
                        model.cfg.rope_dim,
                        model.cfg.rope_freq_base,
                        eps,
                        &mut state.k_caches[il],
                        &mut state.v_caches[il],
                        &mut att_cat,
                    );
                } else if model.cfg.rope_neox && pos_thw[0] as usize != state.pos {
                    step_full_gqa_mrope(
                        &q_raw,
                        &mut k_cur,
                        &v_cur,
                        q_norm.as_deref(),
                        k_norm.as_deref(),
                        *gated_q,
                        nh,
                        nkv,
                        hd,
                        state.pos,
                        pos_thw,
                        [0, 0, 0],
                        model.cfg.rope_dim,
                        model.cfg.rope_freq_base,
                        eps,
                        &mut state.k_caches[il],
                        &mut state.v_caches[il],
                        &mut att_cat,
                    );
                } else {
                    step_full_gqa(
                        &q_raw,
                        &mut k_cur,
                        &v_cur,
                        q_norm.as_deref(),
                        k_norm.as_deref(),
                        *gated_q,
                        nh,
                        nkv,
                        hd,
                        state.pos,
                        model.cfg.rope_dim,
                        model.cfg.rope_freq_base,
                        model.cfg.rope_neox,
                        eps,
                        &mut state.k_caches[il],
                        &mut state.v_caches[il],
                        &mut att_cat,
                    );
                }
                wo.forward(&att_cat, &mut attn_out, model.metal.as_mut(), threads)?;
            }
        }

        for i in 0..dim {
            cur[i] += attn_out[i];
        }
        rms_norm(&cur, &layer.post_attn_norm, eps, &mut normed);

        forward_multi(
            &normed,
            &mut [
                (&layer.ffn_gate, &mut ffn_gate_buf),
                (&layer.ffn_up, &mut ffn_up_buf),
            ],
            model.metal.as_mut(),
            threads,
        )?;
        for i in 0..model.cfg.n_ff {
            ffn_gate_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];
        }
        layer
            .ffn_down
            .forward(&ffn_gate_buf, &mut ffn_out, model.metal.as_mut(), threads)?;
        for i in 0..dim {
            cur[i] += ffn_out[i];
        }
    }

    state.pos += 1;
    if let Some(logits) = logits_out.as_deref_mut() {
        rms_norm(&cur, &model.output_norm, eps, &mut normed);
        let head = model.output_head.as_ref().unwrap_or(&model.token_embd);
        head.forward(&normed, logits, model.metal.as_mut(), threads)?;
    }
    Ok(())
}

// Fingerprint is FNV-1a over exact reconstructed LE f32 bits, not a quality metric.
pub(crate) fn diagnostic_raw_tensor(t: PackedTensor) -> (PackedTensor, serde_json::Value) {
    let (rows, cols) = (t.rows(), t.cols());
    let kind = format!("{:?}", std::mem::discriminant(&t));
    let data = t.decode_f32();
    let mut hash = 0xcbf29ce484222325u64;
    for x in &data {
        for b in x.to_le_bytes() {
            hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    let audit = serde_json::json!({
        "rows": rows,
        "cols": cols,
        "elements": data.len(),
        "source_kind": kind,
        "decoded_f32_fnv1a": format!("{hash:016x}"),
        "finite": data.iter().all(|x| x.is_finite()),
    });
    (PackedTensor::RawF32 { rows, cols, data }, audit)
}

#[cfg(test)]
mod decoded_diagnostic_tests {
    use super::*;
    #[test]
    fn turbo_materialization_preserves_exact_decode_bits_and_raw_route() {
        let w: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.31).sin()).collect();
        let t = PackedTensor::TurboWht(
            crate::wht_turbo::TurboWhtMatrix::encode(4, 4, 64, &w).unwrap(),
        );
        let expected: Vec<_> = t.decode_f32().iter().map(|x| x.to_bits()).collect();
        let (raw, audit) = diagnostic_raw_tensor(t);
        assert!(matches!(raw, PackedTensor::RawF32 { rows: 4, cols: 64, .. }));
        assert_eq!(
            expected,
            raw.decode_f32().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(audit["finite"], true);
    }
    #[test]
    fn raw_materialization_preserves_exact_decode_bits() {
        let input = PackedTensor::RawF32 {
            rows: 1,
            cols: 4,
            data: vec![-0.0, 1.25, -2.5, f32::from_bits(1)],
        };
        let expected: Vec<_> = input.decode_f32().iter().map(|x| x.to_bits()).collect();
        let (raw, audit) = diagnostic_raw_tensor(input);
        assert!(matches!(raw, PackedTensor::RawF32 { .. }));
        assert_eq!(
            expected,
            raw.decode_f32().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(audit["elements"], 4);
    }
}
