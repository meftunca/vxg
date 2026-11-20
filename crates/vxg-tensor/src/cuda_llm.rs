//! 100% VRAM-resident NVIDIA CUDA LLM runner (`CudaTurboRunner` / `CudaTurboState`)
//! for `VXT-Turbo` (`E8`, `bits = 7`) and `VXT-Turbo-HQ` (`E8-HQ`, `bits = 8`) archives.
//!
//! Executes embedding decode, RMSNorm + fused FWHT-32, Gated-DeltaNet recurrence,
//! Gated-GQA with dynamic VRAM KV cache, SwiGLU/GeGLU FFN, LM-Head, and GPU argmax
//! on a single `cudaStream_t` with zero per-layer PCIe transfers.

use crate::cuda::CudaEngine;
use crate::format::PackedTensor;
use crate::gguf_meta::ModelConfig;
use crate::llm::DecoderLayer;
#[cfg(vxt_has_cuda)]
use crate::llm::LayerAttn;
#[cfg(not(vxt_has_cuda))]
use crate::Error;
use crate::Result;

#[cfg(vxt_has_cuda)]
use std::ffi::c_void;

#[cfg(vxt_has_cuda)]
#[repr(C)]
struct CudaMatSpecC {
    kind: i32,
    rows: u32,
    cols: u32,
    padded_cols: u32,
    scales_f16: *const u16,
    n_scales: usize,
    packed_words: *const u32,
    n_words: usize,
    f32_data: *const f32,
}

#[cfg(vxt_has_cuda)]
#[repr(C)]
struct CudaLlmCfgC {
    n_vocab: u32,
    n_embd: u32,
    n_layer: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    n_ff: u32,
    rope_dim: u32,
    rope_freq_base: f32,
    rope_neox: u32,
    rms_norm_eps: f32,
    embed_scale: f32,
    ffn_gelu: u32,
    ssm_conv_kernel: u32,
    ssm_state_size: u32,
    ssm_group_count: u32,
    ssm_dt_rank: u32,
    ssm_inner_size: u32,
}

#[cfg(vxt_has_cuda)]
unsafe extern "C" {
    fn vxt_cuda_llm_create(
        cfg: *const CudaLlmCfgC,
        token_embd: *const CudaMatSpecC,
        output_norm: *const f32,
        output_head: *const CudaMatSpecC,
    ) -> *mut c_void;
    fn vxt_cuda_llm_destroy(ptr: *mut c_void);
    fn vxt_cuda_llm_set_layer_deltanet(
        runner: *mut c_void, il: u32, attn_norm: *const f32, post_attn_norm: *const f32,
        attn_post_norm: *const f32, ffn_post_norm: *const f32, wqkv: *const CudaMatSpecC,
        wqkv_gate: *const CudaMatSpecC, ssm_alpha: *const CudaMatSpecC, ssm_beta: *const CudaMatSpecC,
        ssm_out: *const CudaMatSpecC, ssm_conv1d: *const f32, n_conv1d: usize, ssm_a: *const f32,
        ssm_dt_bias: *const f32, ssm_norm: *const f32, ffn_gate: *const CudaMatSpecC,
        ffn_up: *const CudaMatSpecC, ffn_down: *const CudaMatSpecC,
    ) -> i32;
    fn vxt_cuda_llm_set_layer_gqa(
        runner: *mut c_void, il: u32, attn_norm: *const f32, post_attn_norm: *const f32,
        attn_post_norm: *const f32, ffn_post_norm: *const f32, wq: *const CudaMatSpecC,
        wk: *const CudaMatSpecC, wv: *const CudaMatSpecC, wo: *const CudaMatSpecC,
        q_norm: *const f32, k_norm: *const f32, q_bias: *const f32, n_q_bias: usize,
        k_bias: *const f32, n_k_bias: usize, v_bias: *const f32, n_v_bias: usize, gated_q: u32,
        ffn_gate: *const CudaMatSpecC, ffn_up: *const CudaMatSpecC, ffn_down: *const CudaMatSpecC,
    ) -> i32;
    fn vxt_cuda_llm_state_create(runner: *const c_void) -> *mut c_void;
    fn vxt_cuda_llm_state_destroy(st: *mut c_void);
    fn vxt_cuda_llm_step(
        runner: *const c_void, st: *mut c_void, token_id: u32, pos: u32, logits: *mut f32,
    ) -> i32;
    fn vxt_cuda_llm_prefill(
        runner: *const c_void, st: *mut c_void, tokens: *const u32, n_tokens: usize,
        start_pos: u32, logits_host: *mut f32,
    ) -> i32;
    fn vxt_cuda_llm_init_seen(
        runner: *const c_void, st: *mut c_void, tok_ids: *const u32, n_toks: usize,
    ) -> i32;
    fn vxt_cuda_llm_step_greedy(
        runner: *const c_void, st: *mut c_void, token_id: u32, pos: u32,
        presence_pen: f32, out_next_token: *mut u32,
    ) -> i32;
    fn vxt_cuda_llm_mark_seen(runner: *const c_void, st: *mut c_void, tok_id: u32) -> i32;
    fn vxt_cuda_llm_checkpoint(runner: *const c_void, st: *mut c_void, restore: u32) -> i32;
    fn vxt_cuda_llm_replay_trunk(
        runner: *const c_void, st: *mut c_void, tokens: *const u32, n_tokens: usize, start_pos: u32,
    ) -> i32;
    fn vxt_cuda_llm_kv_bytes(st: *const c_void) -> usize;
}

pub struct CudaTurboState {
    #[cfg(vxt_has_cuda)]
    ptr: *mut c_void,
    #[cfg(not(vxt_has_cuda))]
    _priv: (),
}

#[cfg(vxt_has_cuda)]
unsafe impl Send for CudaTurboState {}

#[cfg(vxt_has_cuda)]
impl Drop for CudaTurboState {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { vxt_cuda_llm_state_destroy(self.ptr) };
        }
    }
}

impl CudaTurboState {
    pub fn kv_bytes(&self) -> usize {
        #[cfg(vxt_has_cuda)] { unsafe { vxt_cuda_llm_kv_bytes(self.ptr) } }
        #[cfg(not(vxt_has_cuda))] { 0 }
    }
}

pub struct CudaTurboRunner {
    #[cfg(vxt_has_cuda)]
    ptr: *mut c_void,
    #[cfg(vxt_has_cuda)]
    n_vocab: usize,
    #[cfg(vxt_has_cuda)]
    device_name: String,
    #[cfg(not(vxt_has_cuda))]
    _priv: (),
}

#[cfg(vxt_has_cuda)]
unsafe impl Send for CudaTurboRunner {}
#[cfg(vxt_has_cuda)]
unsafe impl Sync for CudaTurboRunner {}

#[cfg(vxt_has_cuda)]
impl Drop for CudaTurboRunner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { vxt_cuda_llm_destroy(self.ptr) };
        }
    }
}

#[cfg(vxt_has_cuda)]
fn is_turbo_e8(t: &PackedTensor) -> bool {
    matches!(t, PackedTensor::TurboWht(m) if m.bits == 7 || m.bits == 8)
}

#[cfg(vxt_has_cuda)]
struct OwnedMatSpec {
    spec: CudaMatSpecC,
    _f32_backing: Vec<f32>,
}

#[cfg(vxt_has_cuda)]
fn make_mat_spec(t: &PackedTensor, allow_f32_decode: bool) -> Option<OwnedMatSpec> {
    match t {
        PackedTensor::TurboWht(m) if m.bits == 7 || m.bits == 8 => Some(OwnedMatSpec {
            spec: CudaMatSpecC {
                kind: m.bits as i32, rows: m.rows as u32, cols: m.cols as u32,
                padded_cols: m.padded_cols as u32, scales_f16: m.scales_f16.as_ptr(),
                n_scales: m.scales_f16.len(), packed_words: m.packed_words.as_ptr(),
                n_words: m.packed_words.len(), f32_data: std::ptr::null(),
            },
            _f32_backing: Vec::new(),
        }),
        PackedTensor::RawF32 { rows, cols, data } if allow_f32_decode => Some(OwnedMatSpec {
            spec: CudaMatSpecC {
                kind: 1, rows: *rows as u32, cols: *cols as u32, padded_cols: *cols as u32,
                scales_f16: std::ptr::null(), n_scales: 0, packed_words: std::ptr::null(),
                n_words: 0, f32_data: data.as_ptr(),
            },
            _f32_backing: Vec::new(),
        }),
        other if allow_f32_decode => {
            let (rows, cols) = (other.rows(), other.cols());
            let decoded = other.decode_f32();
            let ptr = decoded.as_ptr();
            Some(OwnedMatSpec {
                spec: CudaMatSpecC {
                    kind: 1, rows: rows as u32, cols: cols as u32, padded_cols: cols as u32,
                    scales_f16: std::ptr::null(), n_scales: 0, packed_words: std::ptr::null(),
                    n_words: 0, f32_data: ptr,
                },
                _f32_backing: decoded,
            })
        }
        _ => None,
    }
}

#[cfg(vxt_has_cuda)]
fn is_cuda_turbo_compatible(
    cfg: &ModelConfig,
    token_embd: &PackedTensor,
    output_head: Option<&PackedTensor>,
    layers: &[DecoderLayer],
) -> bool {
    if !is_turbo_e8(token_embd) && !matches!(token_embd, PackedTensor::RawF32 { .. }) {
        return false;
    }
    if let Some(h) = output_head {
        if !is_turbo_e8(h) && !matches!(h, PackedTensor::RawF32 { .. }) {
            return false;
        }
    }
    for ly in layers {
        if ly.ffn_moe.is_some()
            || !is_turbo_e8(&ly.ffn_gate.cpu)
            || !is_turbo_e8(&ly.ffn_up.cpu)
            || !is_turbo_e8(&ly.ffn_down.cpu)
        {
            return false;
        }
        match &ly.attn {
            LayerAttn::LinearDeltaNet { wqkv, wqkv_gate, ssm_out, .. } => {
                let dv = cfg.ssm_inner_size.checked_div(cfg.ssm_dt_rank).unwrap_or(0);
                if dv == 0 || dv % 32 != 0 || dv > 512 || cfg.ssm_state_size == 0
                    || cfg.ssm_state_size > 512 || cfg.ssm_group_count == 0
                    || cfg.ssm_dt_rank % cfg.ssm_group_count != 0
                    || !is_turbo_e8(&wqkv.cpu) || !is_turbo_e8(&wqkv_gate.cpu)
                    || !is_turbo_e8(&ssm_out.cpu)
                {
                    return false;
                }
            }
            LayerAttn::FullGqa { wq, wk, wv, wo, .. } => {
                if cfg.head_dim == 0 || cfg.head_dim % 32 != 0 || cfg.head_dim > 512
                    || cfg.n_head_kv == 0 || cfg.n_head % cfg.n_head_kv != 0
                    || !is_turbo_e8(&wq.cpu) || !is_turbo_e8(&wk.cpu)
                    || !is_turbo_e8(&wv.cpu) || !is_turbo_e8(&wo.cpu)
                {
                    return false;
                }
            }
        }
    }
    true
}

impl CudaTurboRunner {
    pub fn device_name(&self) -> &str {
        #[cfg(vxt_has_cuda)] { &self.device_name }
        #[cfg(not(vxt_has_cuda))] { "" }
    }

    pub(crate) fn try_from_layers(
        eng: &CudaEngine,
        cfg: &ModelConfig,
        token_embd: &PackedTensor,
        output_norm: &[f32],
        output_head: Option<&PackedTensor>,
        layers: &[DecoderLayer],
    ) -> Result<Option<Self>> {
        #[cfg(vxt_has_cuda)]
        {
            if !is_cuda_turbo_compatible(cfg, token_embd, output_head, layers) {
                return Ok(None);
            }
            let c_cfg = CudaLlmCfgC {
                n_vocab: cfg.n_vocab as u32, n_embd: cfg.n_embd as u32, n_layer: cfg.n_layer as u32,
                n_head: cfg.n_head as u32, n_head_kv: cfg.n_head_kv as u32, head_dim: cfg.head_dim as u32,
                n_ff: cfg.n_ff as u32, rope_dim: cfg.rope_dim as u32, rope_freq_base: cfg.rope_freq_base,
                rope_neox: u32::from(cfg.rope_neox), rms_norm_eps: cfg.rms_norm_eps,
                embed_scale: cfg.embed_scale, ffn_gelu: u32::from(cfg.ffn_gelu),
                ssm_conv_kernel: cfg.ssm_conv_kernel as u32, ssm_state_size: cfg.ssm_state_size as u32,
                ssm_group_count: cfg.ssm_group_count as u32, ssm_dt_rank: cfg.ssm_dt_rank as u32,
                ssm_inner_size: cfg.ssm_inner_size as u32,
            };
            let emb = make_mat_spec(token_embd, true)
                .ok_or_else(|| crate::Error::Backend("CUDA runner: unsupported embedding".into()))?;
            let head_opt = match output_head {
                Some(h) => Some(make_mat_spec(h, true)
                    .ok_or_else(|| crate::Error::Backend("CUDA runner: unsupported head".into()))?),
                None => None,
            };
            let head_ptr = head_opt.as_ref().map_or(std::ptr::null(), |m| &m.spec);
            let ptr = unsafe { vxt_cuda_llm_create(&c_cfg, &emb.spec, output_norm.as_ptr(), head_ptr) };
            if ptr.is_null() {
                return Err(crate::Error::Backend("vxt_cuda_llm_create failed".into()));
            }
            let runner = Self { ptr, n_vocab: cfg.n_vocab, device_name: eng.device_name().to_string() };
            let opt_ptr = |v: &Option<Vec<f32>>| v.as_ref().map_or(std::ptr::null(), |x| x.as_ptr());
            let opt_len = |v: &Option<Vec<f32>>| v.as_ref().map_or(0, Vec::len);

            for (il, ly) in layers.iter().enumerate() {
                let fg = make_mat_spec(&ly.ffn_gate.cpu, false).unwrap();
                let fu = make_mat_spec(&ly.ffn_up.cpu, false).unwrap();
                let fd = make_mat_spec(&ly.ffn_down.cpu, false).unwrap();
                let rc = match &ly.attn {
                    LayerAttn::LinearDeltaNet {
                        wqkv, wqkv_gate, ssm_alpha, ssm_beta, ssm_out,
                        ssm_conv1d, ssm_a, ssm_dt_bias, ssm_norm,
                    } => {
                        let m_qkv = make_mat_spec(&wqkv.cpu, false).unwrap();
                        let m_gate = make_mat_spec(&wqkv_gate.cpu, false).unwrap();
                        let m_alpha = make_mat_spec(&ssm_alpha.cpu, true).unwrap();
                        let m_beta = make_mat_spec(&ssm_beta.cpu, true).unwrap();
                        let m_out = make_mat_spec(&ssm_out.cpu, false).unwrap();
                        unsafe {
                            vxt_cuda_llm_set_layer_deltanet(
                                runner.ptr, il as u32, ly.attn_norm.as_ptr(), ly.post_attn_norm.as_ptr(),
                                opt_ptr(&ly.attn_post_norm), opt_ptr(&ly.ffn_post_norm),
                                &m_qkv.spec, &m_gate.spec, &m_alpha.spec, &m_beta.spec, &m_out.spec,
                                ssm_conv1d.as_ptr(), ssm_conv1d.len(), ssm_a.as_ptr(),
                                ssm_dt_bias.as_ptr(), ssm_norm.as_ptr(), &fg.spec, &fu.spec, &fd.spec,
                            )
                        }
                    }
                    LayerAttn::FullGqa {
                        wq, wk, wv, wo, q_norm, k_norm, q_bias, k_bias, v_bias, gated_q,
                    } => {
                        let m_q = make_mat_spec(&wq.cpu, false).unwrap();
                        let m_k = make_mat_spec(&wk.cpu, false).unwrap();
                        let m_v = make_mat_spec(&wv.cpu, false).unwrap();
                        let m_o = make_mat_spec(&wo.cpu, false).unwrap();
                        unsafe {
                            vxt_cuda_llm_set_layer_gqa(
                                runner.ptr, il as u32, ly.attn_norm.as_ptr(), ly.post_attn_norm.as_ptr(),
                                opt_ptr(&ly.attn_post_norm), opt_ptr(&ly.ffn_post_norm),
                                &m_q.spec, &m_k.spec, &m_v.spec, &m_o.spec,
                                opt_ptr(q_norm), opt_ptr(k_norm), opt_ptr(q_bias), opt_len(q_bias),
                                opt_ptr(k_bias), opt_len(k_bias), opt_ptr(v_bias), opt_len(v_bias),
                                u32::from(*gated_q), &fg.spec, &fu.spec, &fd.spec,
                            )
                        }
                    }
                };
                if rc != 0 {
                    return Err(crate::Error::Backend(format!("CUDA layer {il} upload failed (rc={rc})")));
                }
            }
            Ok(Some(runner))
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (eng, cfg, token_embd, output_norm, output_head, layers);
            Ok(None)
        }
    }

    pub fn new_state(&self) -> Result<CudaTurboState> {
        #[cfg(vxt_has_cuda)]
        {
            let ptr = unsafe { vxt_cuda_llm_state_create(self.ptr) };
            if ptr.is_null() {
                return Err(crate::Error::Backend("vxt_cuda_llm_state_create failed".into()));
            }
            Ok(CudaTurboState { ptr })
        }
        #[cfg(not(vxt_has_cuda))]
        {
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }

    pub fn prefill_tokens(
        &self, state: &mut CudaTurboState, pos: &mut usize, prompt_ids: &[u32], logits_out: &mut [f32],
    ) -> Result<()> {
        #[cfg(vxt_has_cuda)]
        {
            if prompt_ids.is_empty() {
                return Ok(());
            }
            if logits_out.len() != self.n_vocab {
                return Err(crate::Error::InvalidShape("CUDA prefill logits_out len != n_vocab"));
            }
            let rc = unsafe {
                vxt_cuda_llm_prefill(
                    self.ptr, state.ptr, prompt_ids.as_ptr(), prompt_ids.len(),
                    *pos as u32, logits_out.as_mut_ptr(),
                )
            };
            if rc != 0 {
                return Err(crate::Error::Backend(format!("vxt_cuda_llm_prefill failed (rc={rc})")));
            }
            *pos += prompt_ids.len();
            Ok(())
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, pos, prompt_ids, logits_out);
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }

    pub fn step_single(
        &self, state: &mut CudaTurboState, pos: u32, token_id: u32, logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        #[cfg(vxt_has_cuda)]
        {
            let out_ptr = match logits_out {
                Some(buf) => {
                    if buf.len() != self.n_vocab {
                        return Err(crate::Error::InvalidShape("CUDA step logits_out len != n_vocab"));
                    }
                    buf.as_mut_ptr()
                }
                None => std::ptr::null_mut(),
            };
            let rc = unsafe { vxt_cuda_llm_step(self.ptr, state.ptr, token_id, pos, out_ptr) };
            if rc != 0 {
                return Err(crate::Error::Backend(format!("vxt_cuda_llm_step failed (rc={rc})")));
            }
            Ok(())
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, pos, token_id, logits_out);
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }

    pub(crate) fn step_greedy_single(
        &self, state: &mut CudaTurboState, pos: u32, token_id: u32, presence_pen: f32,
    ) -> Result<u32> {
        #[cfg(vxt_has_cuda)]
        {
            let mut next = 0u32;
            let rc = unsafe {
                vxt_cuda_llm_step_greedy(self.ptr, state.ptr, token_id, pos, presence_pen, &mut next)
            };
            if rc != 0 {
                return Err(crate::Error::Backend(format!("vxt_cuda_llm_step_greedy failed (rc={rc})")));
            }
            Ok(next)
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, pos, token_id, presence_pen);
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }

    pub(crate) fn checkpoint(&self, state: &mut CudaTurboState, restore: bool) -> Result<()> {
        #[cfg(vxt_has_cuda)]
        {
            let rc = unsafe { vxt_cuda_llm_checkpoint(self.ptr, state.ptr, u32::from(restore)) };
            if rc != 0 {
                return Err(crate::Error::Backend(format!("vxt_cuda_llm_checkpoint failed (rc={rc})")));
            }
            Ok(())
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, restore);
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }

    pub(crate) fn replay_trunk(
        &self, state: &mut CudaTurboState, tok_ids: &[u32], pos_base: u32,
    ) -> Result<()> {
        #[cfg(vxt_has_cuda)]
        {
            if tok_ids.is_empty() {
                return Ok(());
            }
            let rc = unsafe {
                vxt_cuda_llm_replay_trunk(self.ptr, state.ptr, tok_ids.as_ptr(), tok_ids.len(), pos_base)
            };
            if rc != 0 {
                return Err(crate::Error::Backend(format!("vxt_cuda_llm_replay_trunk failed (rc={rc})")));
            }
            Ok(())
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, tok_ids, pos_base);
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }

    pub(crate) fn init_seen(&self, state: &mut CudaTurboState, seen_ids: &[u32]) {
        #[cfg(vxt_has_cuda)]
        {
            let _ = unsafe { vxt_cuda_llm_init_seen(self.ptr, state.ptr, seen_ids.as_ptr(), seen_ids.len()) };
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, seen_ids);
        }
    }

    #[allow(dead_code)]
    pub(crate) fn mark_seen(&self, state: &mut CudaTurboState, tok_id: u32) {
        #[cfg(vxt_has_cuda)]
        {
            let _ = unsafe { vxt_cuda_llm_mark_seen(self.ptr, state.ptr, tok_id) };
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, tok_id);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_fast<F: FnMut(u32) -> bool>(
        &self, state: &mut CudaTurboState, pos: &mut usize, first_tok: u32,
        max_new_tokens: usize, presence_pen: f32, stop_ids: &[u32], mut on_token: F,
    ) -> Result<Vec<u32>> {
        #[cfg(vxt_has_cuda)]
        {
            let mut gen_ids = Vec::with_capacity(max_new_tokens);
            if max_new_tokens == 0 || stop_ids.contains(&first_tok) {
                return Ok(gen_ids);
            }
            gen_ids.push(first_tok);
            if !on_token(first_tok) || max_new_tokens == 1 {
                return Ok(gen_ids);
            }
            if presence_pen > 0.0 {
                let rc = unsafe { vxt_cuda_llm_init_seen(self.ptr, state.ptr, &first_tok, 1) };
                if rc != 0 {
                    return Err(crate::Error::Backend(format!("vxt_cuda_llm_init_seen failed ({rc})")));
                }
            }
            let mut cur_tok = first_tok;
            while gen_ids.len() < max_new_tokens {
                let mut next_tok = 0u32;
                let rc = unsafe {
                    vxt_cuda_llm_step_greedy(
                        self.ptr, state.ptr, cur_tok, *pos as u32, presence_pen, &mut next_tok,
                    )
                };
                if rc != 0 {
                    return Err(crate::Error::Backend(format!("vxt_cuda_llm_step_greedy failed ({rc})")));
                }
                *pos += 1;
                if stop_ids.contains(&next_tok) {
                    break;
                }
                gen_ids.push(next_tok);
                if !on_token(next_tok) {
                    break;
                }
                cur_tok = next_tok;
            }
            Ok(gen_ids)
        }
        #[cfg(not(vxt_has_cuda))]
        {
            let _ = (state, pos, first_tok, max_new_tokens, presence_pen, stop_ids, &mut on_token);
            Err(Error::Unsupported("CUDA unavailable".into()))
        }
    }
}

#[cfg(all(test, vxt_has_cuda))]
mod tests {
    use std::collections::HashMap;
    use super::*;
    use crate::format::{write_vxt_model, VxtEntry};
    use crate::gguf_meta::GgufVal;
    use crate::llm::LlmModel;
    use crate::tokenizer::BpeTokenizer;
    use crate::wht_turbo::TurboWhtMatrix;

    fn mk_turbo(rows: usize, cols: usize, seed: u64, bits: u8) -> PackedTensor {
        let mut s = seed ^ 0x9e3779b97f4a7c15;
        let mut data = vec![0.0f32; rows * cols];
        for v in &mut data {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (((s >> 40) as f32) / ((1u32 << 24) as f32) - 0.5) * 0.25;
        }
        PackedTensor::TurboWht(TurboWhtMatrix::encode(bits, rows, cols, &data).unwrap())
    }

    fn mk_vec(n: usize, val: f32) -> PackedTensor {
        PackedTensor::RawF32 { rows: 1, cols: n, data: vec![val; n] }
    }

    #[test]
    fn cuda_turbo_runner_matches_cpu_hybrid_qwen35() {
        let cfg = ModelConfig {
            arch: "qwen35".into(), model_name: "mini-qwen35".into(), n_vocab: 64, n_embd: 256, n_layer: 2,
            n_head: 4, n_head_kv: 2, head_dim: 64, n_ff: 512,
            rope_dim: 32, rope_freq_base: 10000.0, rope_neox: true,
            rms_norm_eps: 1e-6, embed_scale: 1.0, ffn_gelu: false,
            ssm_conv_kernel: 4, ssm_state_size: 64, ssm_group_count: 2,
            ssm_dt_rank: 4, ssm_inner_size: 256, full_attn_interval: 2,
            bos_token_id: Some(1), eos_token_id: 2,
        };
        let e = |name: &str, tensor: PackedTensor| VxtEntry { name: name.into(), tensor };
        let tensors = vec![
            e("token_embd.weight", mk_turbo(64, 256, 1, 8)),
            e("output_norm.weight", mk_vec(256, 1.0)),
            e("blk.0.attn_norm.weight", mk_vec(256, 1.0)),
            e("blk.0.post_attention_norm.weight", mk_vec(256, 1.0)),
            e("blk.0.attn_qkv.weight", mk_turbo(512, 256, 10, 7)),
            e("blk.0.attn_gate.weight", mk_turbo(256, 256, 11, 7)),
            e("blk.0.ssm_alpha.weight", PackedTensor::RawF32 { rows: 4, cols: 256, data: vec![0.01; 1024] }),
            e("blk.0.ssm_beta.weight", PackedTensor::RawF32 { rows: 4, cols: 256, data: vec![-0.02; 1024] }),
            e("blk.0.ssm_out.weight", mk_turbo(256, 256, 12, 7)),
            e("blk.0.ssm_conv1d.weight", mk_vec(512 * 4, 0.25)),
            e("blk.0.ssm_a", mk_vec(4, -0.5)),
            e("blk.0.ssm_dt.bias", mk_vec(4, 0.1)),
            e("blk.0.ssm_norm.weight", mk_vec(64, 1.0)),
            e("blk.0.ffn_gate.weight", mk_turbo(512, 256, 13, 7)),
            e("blk.0.ffn_up.weight", mk_turbo(512, 256, 14, 7)),
            e("blk.0.ffn_down.weight", mk_turbo(256, 512, 15, 7)),
            e("blk.1.attn_norm.weight", mk_vec(256, 1.0)),
            e("blk.1.post_attention_norm.weight", mk_vec(256, 1.0)),
            e("blk.1.attn_q.weight", mk_turbo(512, 256, 20, 8)),
            e("blk.1.attn_k.weight", mk_turbo(128, 256, 21, 8)),
            e("blk.1.attn_v.weight", mk_turbo(128, 256, 22, 8)),
            e("blk.1.attn_output.weight", mk_turbo(256, 256, 23, 8)),
            e("blk.1.attn_q_norm.weight", mk_vec(64, 1.0)),
            e("blk.1.attn_k_norm.weight", mk_vec(64, 1.0)),
            e("blk.1.ffn_gate.weight", mk_turbo(512, 256, 24, 8)),
            e("blk.1.ffn_up.weight", mk_turbo(512, 256, 25, 8)),
            e("blk.1.ffn_down.weight", mk_turbo(256, 512, 26, 8)),
        ];
        let check_case = |label: &str, cfg: ModelConfig, tensors: Vec<VxtEntry>| {
            let mut kv = HashMap::new();
            kv.insert("tokenizer.ggml.tokens".into(), GgufVal::ArrStr((0..64).map(|i| format!("t{i}")).collect()));
            kv.insert("tokenizer.ggml.merges".into(), GgufVal::ArrStr(Vec::new()));
            let mut meta = vec![("model.config".to_string(), cfg.to_kv_bytes())];
            meta.extend(BpeTokenizer::to_vxt_meta_entries(&kv).unwrap());
            let bytes = write_vxt_model(&meta, &tensors).unwrap();
            let mut cpu_m = LlmModel::from_vxt_bytes(&bytes, false, 2).unwrap();
            let mut gpu_m = LlmModel::from_vxt_bytes(&bytes, true, 2).unwrap();
            assert!(gpu_m.backend_label().contains("VXT-Turbo Resident"), "{label}: got {}", gpu_m.backend_label());
            let (mut st_cpu, mut st_gpu) = (cpu_m.new_state(), gpu_m.new_state());
            let (mut l_cpu, mut l_gpu) = (vec![0.0f32; 64], vec![0.0f32; 64]);
            let prompt = [1u32, 5, 12, 33];
            cpu_m.prefill(&mut st_cpu, &prompt, &mut l_cpu).unwrap();
            gpu_m.prefill(&mut st_gpu, &prompt, &mut l_gpu).unwrap();
            let max_err = l_cpu.iter().zip(&l_gpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("[CUDA-TEST {label}] max_err={max_err:.6} cpu={:?} gpu={:?}", &l_cpu[..4], &l_gpu[..4]);
            (cpu_m, gpu_m, st_cpu, st_gpu, l_cpu, l_gpu, max_err)
        };
        let mut c0 = cfg.clone(); c0.n_layer = 0;
        let (_, _, _, _, _, _, e0) = check_case("0-layer-e8hq-emb", c0, vec![
            e("token_embd.weight", mk_turbo(64, 256, 1, 8)),
            e("output_norm.weight", mk_vec(256, 1.0)),
        ]);
        let mut c1d = cfg.clone(); c1d.n_layer = 1; c1d.full_attn_interval = 4;
        let (_, _, _, _, _, _, e1d) = check_case("1-layer-deltanet", c1d, tensors[..16].to_vec());
        let mut c1g = cfg.clone(); c1g.n_layer = 1; c1g.full_attn_interval = 1;
        let mut tg = vec![e("token_embd.weight", mk_turbo(64, 256, 1, 8)), e("output_norm.weight", mk_vec(256, 1.0))];
        for t in &tensors[16..] {
            tg.push(VxtEntry { name: t.name.replace("blk.1.", "blk.0."), tensor: t.tensor.clone() });
        }
        let (_, _, _, _, _, _, e1g) = check_case("1-layer-gqa", c1g, tg);
        let (mut cpu_m, gpu_m, mut st_cpu, mut st_gpu, mut l_cpu, l_gpu, max_err) =
            check_case("2-layer-hybrid", cfg, tensors);
        assert!(e0 < 1e-3 && e1d < 1e-3 && e1g < 1e-3 && max_err < 1e-3, "e0={e0} e1d={e1d} e1g={e1g} hyb={max_err}");

        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        let first_tok = argmax(&l_cpu);
        assert_eq!(first_tok, argmax(&l_gpu));

        let mut cpu_gen = vec![first_tok];
        let mut cur = first_tok;
        for _ in 1..6 {
            cpu_m.forward_step(&mut st_cpu, cur, Some(&mut l_cpu)).unwrap();
            cur = argmax(&l_cpu);
            cpu_gen.push(cur);
        }
        let gpu_gen = gpu_m
            .generate_greedy_turbo(&mut st_gpu, &[1, 5, 12, 33], first_tok, 6, false, 0.0, &[], |_| true)
            .unwrap();
        assert_eq!(cpu_gen, gpu_gen, "greedy token mismatch CPU vs CUDA resident");
    }
}
