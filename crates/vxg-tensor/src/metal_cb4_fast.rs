//! High-Throughput Bit-Exact `CB4+Outlier` All-GPU LLM Runner (`MetalCb4FastRunner`):
//! - Single 256B-aligned Metal weight arena (`NR = 2` step-major layers + `NR = 4` tile-major vocab)
//! - 32-bit SRAM `c16` codebook LUT (`0-ULP` bit-identical to cubic polynomial evaluation)
//! - Fused `NR = 2` `[Gate; Up] + SwiGLU`, write-only `delta` + `Add+RMSNorm`, and `M = 1` GQA
//! - 64-TG split-state `FP32` DeltaNet + batched `M <= 32` prefill (`NR = 2 x BM = 8` GEMM)
//! - 2-Deep `8`-step pipelined `MTLCommandBuffer` streaming with exact `FP32` checkpoint recovery

use crate::format::PackedTensor;
use crate::gguf_meta::ModelConfig;
use crate::llm::{DecoderLayer, LayerAttn};
use crate::metal::MetalEngine;
use crate::tokenizer::BpeTokenizer;
use crate::{Error, Result};

fn is_cb4_mat(t: &PackedTensor, expected_cols: usize, align_rows32: bool) -> bool {
    let PackedTensor::Codebook(cb) = t else {
        return false;
    };
    cb.bits == 4
        && cb.cols == expected_cols
        && cb.padded_cols >= expected_cols
        && cb.padded_cols % 256 == 0
        && (!align_rows32 || cb.rows % 32 == 0)
}

#[cfg_attr(not(vxt_has_metal), allow(dead_code))]
pub(crate) fn is_cb4_fast_compatible(
    cfg: &ModelConfig,
    token_embd: &PackedTensor,
    output_head: Option<&PackedTensor>,
    layers: &[DecoderLayer],
) -> bool {
    if std::env::var_os("VXG_CB4_ROWS4").is_some()
        || std::env::var_os("VXG_CB4_GENERIC").is_some()
    {
        return false;
    }
    if cfg.n_embd == 0 || cfg.n_ff == 0 || cfg.n_vocab == 0 || layers.len() != cfg.n_layer {
        return false;
    }
    if !is_cb4_mat(token_embd, cfg.n_embd, false) || token_embd.rows() != cfg.n_vocab {
        return false;
    }
    if let Some(h) = output_head {
        if !is_cb4_mat(h, cfg.n_embd, false) || h.rows() != cfg.n_vocab {
            return false;
        }
    }
    for ly in layers {
        if !is_cb4_mat(&ly.ffn_gate.cpu, cfg.n_embd, true)
            || !is_cb4_mat(&ly.ffn_up.cpu, cfg.n_embd, true)
            || !is_cb4_mat(&ly.ffn_down.cpu, cfg.n_ff, false)
            || ly.ffn_gate.cpu.rows() != cfg.n_ff
            || ly.ffn_up.cpu.rows() != cfg.n_ff
            || ly.ffn_down.cpu.rows() != cfg.n_embd
        {
            return false;
        }
        match &ly.attn {
            LayerAttn::LinearDeltaNet {
                wqkv,
                wqkv_gate,
                ssm_alpha,
                ssm_beta,
                ssm_out,
                ..
            } => {
                let dv = cfg.ssm_inner_size.checked_div(cfg.ssm_dt_rank).unwrap_or(0);
                let conv_ch = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
                if dv == 0
                    || dv > 512
                    || cfg.ssm_state_size == 0
                    || cfg.ssm_state_size > 512
                    || cfg.ssm_dt_rank == 0
                    || cfg.ssm_group_count == 0
                {
                    return false;
                }
                if !is_cb4_mat(&wqkv.cpu, cfg.n_embd, true)
                    || !is_cb4_mat(&wqkv_gate.cpu, cfg.n_embd, true)
                    || !is_cb4_mat(&ssm_out.cpu, cfg.ssm_inner_size, false)
                    || wqkv.cpu.rows() != conv_ch
                    || wqkv_gate.cpu.rows() != cfg.ssm_inner_size
                    || ssm_out.cpu.rows() != cfg.n_embd
                    || ssm_alpha.cpu.rows() != cfg.ssm_dt_rank
                    || ssm_beta.cpu.rows() != cfg.ssm_dt_rank
                    || ssm_alpha.cpu.cols() != cfg.n_embd
                    || ssm_beta.cpu.cols() != cfg.n_embd
                {
                    return false;
                }
            }
            LayerAttn::FullGqa { wq, wk, wv, wo, .. } => {
                let kv_dim = cfg.n_head_kv * cfg.head_dim;
                let att_dim = cfg.n_head * cfg.head_dim;
                if cfg.head_dim == 0
                    || cfg.head_dim > 512
                    || cfg.n_head_kv == 0
                    || cfg.n_head < cfg.n_head_kv
                {
                    return false;
                }
                if !is_cb4_mat(&wq.cpu, cfg.n_embd, true)
                    || !is_cb4_mat(&wk.cpu, cfg.n_embd, true)
                    || !is_cb4_mat(&wv.cpu, cfg.n_embd, false)
                    || !is_cb4_mat(&wo.cpu, att_dim, false)
                    || wk.cpu.rows() != kv_dim
                    || wv.cpu.rows() != kv_dim
                    || wo.cpu.rows() != cfg.n_embd
                {
                    return false;
                }
            }
        }
    }
    true
}

pub(crate) struct MetalCb4FastRunner {
    #[cfg(vxt_has_metal)]
    inner: crate::metal_cb4_fast_ffi::Cb4FastRunnerInner,
    #[cfg_attr(not(vxt_has_metal), allow(dead_code))]
    n_vocab: usize,
}

pub(crate) struct MetalCb4FastState {
    #[cfg(vxt_has_metal)]
    inner: crate::metal_cb4_fast_ffi::Cb4FastStateInner,
}

impl MetalCb4FastRunner {
    pub fn try_from_layers(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        token_embd: &PackedTensor,
        output_norm: &[f32],
        output_head: Option<&PackedTensor>,
        layers: &[DecoderLayer],
    ) -> Result<Option<Self>> {
        #[cfg(vxt_has_metal)]
        {
            let Some(inner) = crate::metal_cb4_fast_ffi::Cb4FastRunnerInner::try_create(
                eng,
                cfg,
                token_embd,
                output_norm,
                output_head,
                layers,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(Self {
                inner,
                n_vocab: cfg.n_vocab,
            }))
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (eng, cfg, token_embd, output_norm, output_head, layers);
            Ok(None)
        }
    }

    pub fn new_state(&self) -> Result<MetalCb4FastState> {
        #[cfg(vxt_has_metal)]
        {
            Ok(MetalCb4FastState {
                inner: self.inner.new_state()?,
            })
        }
        #[cfg(not(vxt_has_metal))]
        {
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }

    pub fn step_single(
        &self,
        state: &mut MetalCb4FastState,
        pos: u32,
        rope_delta: isize,
        tok_id: u32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            let mode = u32::from(logits_out.is_some());
            let rpos = ((pos as isize) + rope_delta).max(0) as u32;
            self.inner
                .step_batch(&mut state.inner, &[tok_id], pos, rpos, mode, logits_out)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (state, pos, rope_delta, tok_id, logits_out);
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }

    pub fn prefill_tokens(
        &self,
        state: &mut MetalCb4FastState,
        pos: &mut usize,
        rope_delta: isize,
        prompt_ids: &[u32],
        logits_out: &mut [f32],
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            let mut off = 0usize;
            while off < prompt_ids.len() {
                let m = (prompt_ids.len() - off).min(32);
                let is_last = off + m == prompt_ids.len();
                let mode = if is_last { 2u32 } else { 0u32 };
                let rpos = ((*pos as isize) + rope_delta).max(0) as u32;
                self.inner.step_batch(
                    &mut state.inner,
                    &prompt_ids[off..off + m],
                    *pos as u32,
                    rpos,
                    mode,
                    is_last.then_some(&mut *logits_out),
                )?;
                *pos += m;
                off += m;
            }
            Ok(())
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (state, pos, rope_delta, prompt_ids, logits_out);
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prefill_embeds_mrope(
        &self,
        state: &mut MetalCb4FastState,
        pos: &mut usize,
        embeds: &[f32],
        n_embd: usize,
        mrope_pos: &[[u32; 4]],
        mrope_section: [usize; 3],
        logits_out: &mut [f32],
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            let mut off = 0usize;
            let total = mrope_pos.len();
            while off < total {
                let m = (total - off).min(32);
                let is_last = off + m == total;
                let mode = if is_last { 2u32 } else { 0u32 };
                self.inner.step_embeds_batch(
                    &mut state.inner,
                    &embeds[off * n_embd..(off + m) * n_embd],
                    &mrope_pos[off..off + m],
                    mrope_section,
                    *pos as u32,
                    mode,
                    is_last.then_some(&mut *logits_out),
                )?;
                *pos += m;
                off += m;
            }
            Ok(())
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (state, pos, embeds, n_embd, mrope_pos, mrope_section, logits_out);
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_fast<F: FnMut(u32, &BpeTokenizer) -> bool>(
        &self,
        state: &mut MetalCb4FastState,
        pos: &mut usize,
        rope_delta: isize,
        cb4_failed: &mut bool,
        tokenizer: &BpeTokenizer,
        first_tok: u32,
        max_new_tokens: usize,
        stop_ids: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>> {
        #[cfg(vxt_has_metal)]
        {
            let is_stop = |t: u32| stop_ids.contains(&t);
            let mut gen_ids = Vec::with_capacity(max_new_tokens);
            if max_new_tokens == 0 || is_stop(first_tok) {
                return Ok(gen_ids);
            }
            if first_tok as usize >= self.n_vocab {
                return Err(Error::Format("CB4 greedy token outside vocabulary".into()));
            }
            gen_ids.push(first_tok);
            if !on_token(first_tok, tokenizer) || max_new_tokens == 1 {
                return Ok(gen_ids);
            }

            const BURST: usize = 8;
            let st_in = &mut state.inner;
            let start_pos = *pos;
            self.inner.checkpoint(st_in, false)?;
            let mut pending = std::collections::VecDeque::new();
            let mut submitted = 0usize;
            let mut consumed = 0usize;
            let total = max_new_tokens - 1;
            self.inner.ring_set(0, first_tok);

            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                    loop {
                        while pending.len() < 2 && submitted < total {
                            let count = (total - submitted).min(BURST);
                            let slot = (submitted & 255) as u32;
                            let p_base = start_pos + submitted;
                            let r_base = ((p_base as isize) + rope_delta).max(0) as u32;
                            let cmd = self.inner.enqueue_async(
                                st_in,
                                slot,
                                count as u32,
                                p_base as u32,
                                r_base,
                            )?;
                            pending.push_back((cmd, slot, count));
                            submitted += count;
                        }
                        let Some((cmd, slot, count)) = pending.pop_front() else {
                            return Ok(());
                        };
                        self.inner.wait_cmd(cmd)?;
                        for k in 0..count {
                            let s = (slot + k as u32) & 255;
                            if self.inner.ring_get_err(s) != 0 {
                                return Err(Error::Backend(
                                    "nonfinite greedy logits; state invalidated".into(),
                                ));
                            }
                            let tok = self.inner.ring_get_out(s);
                            if tok as usize >= self.n_vocab {
                                return Err(Error::Format(
                                    "CB4 greedy token outside vocabulary".into(),
                                ));
                            }
                            consumed = gen_ids.len();
                            if is_stop(tok) {
                                return Ok(());
                            }
                            gen_ids.push(tok);
                            if !on_token(tok, tokenizer) {
                                return Ok(());
                            }
                        }
                    }
                }));

            let (result, panic) = match outcome {
                Ok(r) => (r, None),
                Err(payload) => (Ok(()), Some(payload)),
            };

            let mut drain_error = None;
            while let Some((cmd, _, _)) = pending.pop_front() {
                if let Err(error) = self.inner.wait_cmd(cmd) {
                    if drain_error.is_none() {
                        drain_error = Some(error);
                    }
                }
            }

            if result.is_err() || drain_error.is_some() {
                *cb4_failed = true;
                let _ = self.inner.checkpoint(st_in, true);
                *pos = start_pos;
                if let Some(payload) = panic {
                    std::panic::resume_unwind(payload);
                }
                result?;
                return Err(drain_error.unwrap());
            }

            let recovery = (|| -> Result<()> {
                if consumed != submitted || panic.is_some() {
                    self.inner.checkpoint(st_in, true)?;
                    *pos = start_pos;
                    let mut off = 0usize;
                    while off < consumed {
                        let m = (consumed - off).min(32);
                        let rpos = ((*pos as isize) + rope_delta).max(0) as u32;
                        self.inner.step_batch(st_in, &gen_ids[off..off + m], *pos as u32, rpos, 0, None)?;
                        *pos += m;
                        off += m;
                    }
                } else {
                    *pos = start_pos + consumed;
                }
                Ok(())
            })();

            if let Some(payload) = panic {
                std::panic::resume_unwind(payload);
            }
            if let Err(err) = recovery {
                *cb4_failed = true;
                return Err(err);
            }
            Ok(gen_ids)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (
                state,
                pos,
                rope_delta,
                cb4_failed,
                tokenizer,
                first_tok,
                max_new_tokens,
                stop_ids,
                &mut on_token,
            );
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }
}
