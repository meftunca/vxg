//! Ultra-Fused All-GPU Pipelined & Batched `Turbo` LLM Runner (`MetalTurboRunner`):
//! - Serial `MTLComputeCommandEncoder` (`0` manual buffer barriers, `0` standalone `fwht32` launches)
//! - Vertically concatenated sibling projections (`[wqkv; gate]`, `[wq; wk; wv]`, `[ffn_gate; ffn_up]`)
//! - Pre-rotated `FP16` `[ssm_alpha; ssm_beta]` + `FP16` `DeltaNet` `ssm_states` & `GQA` KV cache
//! - 2-Stage Hadamard MMSE Sign-Sketch + Exact Block-Verified `LM Head` (`0` extra `.vxt` bytes)
//! - 2-Deep Zero-Bubble Pipelined `MTLCommandBuffer` Streaming + Lossless N-gram `M = 4` `GEMM` Bursts

use crate::format::PackedTensor;
use crate::gguf_meta::ModelConfig;
use crate::llm::{DecoderLayer, LayerAttn};
use crate::metal::MetalEngine;
#[cfg(not(vxt_has_metal))]
use crate::Error;
use crate::Result;

#[cfg_attr(not(vxt_has_metal), allow(dead_code))]
pub(crate) enum TurboAttnCpuSpec<'a> {
    LinearDeltaNet {
        wqkv: &'a PackedTensor,
        wqkv_gate: &'a PackedTensor,
        ssm_alpha: &'a PackedTensor,
        ssm_beta: &'a PackedTensor,
        ssm_out: &'a PackedTensor,
        ssm_conv1d: &'a [f32],
        ssm_a: &'a [f32],
        ssm_dt_bias: &'a [f32],
        ssm_norm: &'a [f32],
    },
    FullGqa {
        wq: &'a PackedTensor,
        wk: &'a PackedTensor,
        wv: &'a PackedTensor,
        wo: &'a PackedTensor,
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        q_bias: Option<&'a [f32]>,
        k_bias: Option<&'a [f32]>,
        v_bias: Option<&'a [f32]>,
        gated_q: bool,
    },
}

#[cfg_attr(not(vxt_has_metal), allow(dead_code))]
pub(crate) struct TurboLayerCpuSpec<'a> {
    pub attn_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub attn: TurboAttnCpuSpec<'a>,
    pub ffn_gate: &'a PackedTensor,
    pub ffn_up: &'a PackedTensor,
    pub ffn_down: &'a PackedTensor,
}

#[cfg_attr(not(vxt_has_metal), allow(dead_code))]
pub(crate) fn is_turbo_compatible(
    cfg: &ModelConfig,
    token_embd: &PackedTensor,
    output_head: Option<&PackedTensor>,
    layers: &[TurboLayerCpuSpec<'_>],
) -> bool {
    if cfg.ffn_gelu || (cfg.embed_scale - 1.0).abs() > 1e-6 {
        return false;
    }
    if !matches!(token_embd, PackedTensor::TurboWht(_)) {
        return false;
    }
    if let Some(h) = output_head {
        if !matches!(h, PackedTensor::TurboWht(_)) {
            return false;
        }
    }
    for ly in layers {
        let (PackedTensor::TurboWht(fg), PackedTensor::TurboWht(fu), PackedTensor::TurboWht(_)) =
            (ly.ffn_gate, ly.ffn_up, ly.ffn_down)
        else {
            return false;
        };
        if fg.bits != fu.bits || fg.rows % 4 != 0 {
            return false;
        }
        match &ly.attn {
            TurboAttnCpuSpec::LinearDeltaNet {
                wqkv,
                wqkv_gate,
                ssm_out,
                ..
            } => {
                let dv = cfg.ssm_inner_size.checked_div(cfg.ssm_dt_rank).unwrap_or(0);
                if dv == 0 || dv % 32 != 0 || dv > 512 || cfg.ssm_state_size > 512 {
                    return false;
                }
                if !matches!(
                    (wqkv, wqkv_gate, ssm_out),
                    (
                        PackedTensor::TurboWht(_),
                        PackedTensor::TurboWht(_),
                        PackedTensor::TurboWht(_)
                    )
                ) {
                    return false;
                }
            }
            TurboAttnCpuSpec::FullGqa { wq, wk, wv, wo, .. } => {
                if cfg.head_dim == 0 || cfg.head_dim % 32 != 0 || cfg.head_dim > 512 {
                    return false;
                }
                let (
                    PackedTensor::TurboWht(mq),
                    PackedTensor::TurboWht(mk),
                    PackedTensor::TurboWht(mv),
                    PackedTensor::TurboWht(_),
                ) = (wq, wk, wv, wo)
                else {
                    return false;
                };
                if mk.bits != mv.bits || mq.rows % 4 != 0 || mk.rows % 4 != 0 {
                    return false;
                }
            }
        }
    }
    true
}

pub struct MetalTurboState {
    #[cfg(vxt_has_metal)]
    inner: crate::metal_turbo_ffi::TurboStateInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

pub struct MetalTurboRunner {
    #[cfg(vxt_has_metal)]
    inner: crate::metal_turbo_ffi::TurboRunnerInner,
    #[cfg(vxt_has_metal)]
    ngram_spec: std::sync::atomic::AtomicBool,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl MetalTurboRunner {
    pub fn set_ngram_spec(&self, enable: bool) {
        #[cfg(vxt_has_metal)]
        self.ngram_spec
            .store(enable, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(vxt_has_metal))]
        let _ = enable;
    }

    pub(crate) fn try_from_layers(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        token_embd: &PackedTensor,
        output_norm: &[f32],
        output_head: Option<&PackedTensor>,
        layers: &[DecoderLayer],
    ) -> Result<Option<Self>> {
        if cfg.ffn_gelu
            || (cfg.embed_scale - 1.0).abs() > 1e-6
            || layers
                .iter()
                .any(|l| l.attn_post_norm.is_some() || l.ffn_post_norm.is_some())
        {
            return Ok(None);
        }
        let mut specs = Vec::with_capacity(layers.len());
        for ly in layers {
            let attn = match &ly.attn {
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
                } => TurboAttnCpuSpec::LinearDeltaNet {
                    wqkv: &wqkv.cpu,
                    wqkv_gate: &wqkv_gate.cpu,
                    ssm_alpha: &ssm_alpha.cpu,
                    ssm_beta: &ssm_beta.cpu,
                    ssm_out: &ssm_out.cpu,
                    ssm_conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                },
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
                } => TurboAttnCpuSpec::FullGqa {
                    wq: &wq.cpu,
                    wk: &wk.cpu,
                    wv: &wv.cpu,
                    wo: &wo.cpu,
                    q_norm: q_norm.as_deref(),
                    k_norm: k_norm.as_deref(),
                    q_bias: q_bias.as_deref(),
                    k_bias: k_bias.as_deref(),
                    v_bias: v_bias.as_deref(),
                    gated_q: *gated_q,
                },
            };
            specs.push(TurboLayerCpuSpec {
                attn_norm: &ly.attn_norm,
                post_attn_norm: &ly.post_attn_norm,
                attn,
                ffn_gate: &ly.ffn_gate.cpu,
                ffn_up: &ly.ffn_up.cpu,
                ffn_down: &ly.ffn_down.cpu,
            });
        }
        #[cfg(vxt_has_metal)]
        {
            Ok(crate::metal_turbo_ffi::TurboRunnerInner::try_new(
                eng,
                cfg,
                token_embd,
                output_norm,
                output_head,
                &specs,
            )?
            .map(|inner| Self {
                inner,
                ngram_spec: std::sync::atomic::AtomicBool::new(false),
            }))
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (eng, cfg, token_embd, output_norm, output_head, specs);
            Ok(None)
        }
    }

    pub fn new_state(&self) -> Result<MetalTurboState> {
        #[cfg(vxt_has_metal)]
        {
            Ok(MetalTurboState {
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
        state: &mut MetalTurboState,
        pos: u32,
        tok_id: u32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            let mode = u32::from(logits_out.is_some());
            self.inner.step_batch(
                &mut state.inner,
                &[tok_id],
                pos,
                false,
                mode,
                logits_out,
                None,
            )
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (state, pos, tok_id, logits_out);
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }

    pub fn prefill_tokens(
        &self,
        state: &mut MetalTurboState,
        pos: &mut usize,
        prompt_ids: &[u32],
        logits_out: &mut [f32],
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            let mut off = 0usize;
            while off < prompt_ids.len() {
                let rem = prompt_ids.len() - off;
                let m = rem.min(4);
                let is_last = off + m == prompt_ids.len();
                let mode = if is_last { 4u32 } else { 0u32 };
                self.inner.step_batch(
                    &mut state.inner,
                    &prompt_ids[off..off + m],
                    *pos as u32,
                    false,
                    mode,
                    is_last.then_some(&mut *logits_out),
                    None,
                )?;
                *pos += m;
                off += m;
            }
            Ok(())
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (state, pos, prompt_ids, logits_out);
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_fast<F: FnMut(u32) -> bool>(
        &self,
        state: &mut MetalTurboState,
        pos: &mut usize,
        prompt_ids: &[u32],
        first_tok: u32,
        max_new_tokens: usize,
        use_sketch: bool,
        presence_pen: f32,
        stop_ids: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>> {
        #[cfg(vxt_has_metal)]
        {
            struct SeenGuard<'a>(
                &'a crate::metal_turbo_ffi::TurboRunnerInner,
                &'a mut crate::metal_turbo_ffi::TurboStateInner,
            );
            impl Drop for SeenGuard<'_> {
                fn drop(&mut self) {
                    self.0.init_seen(self.1, &[], 0.0);
                }
            }
            let is_stop = |t: u32| stop_ids.contains(&t);
            let mut gen_ids = Vec::with_capacity(max_new_tokens);
            if max_new_tokens == 0 || is_stop(first_tok) {
                return Ok(gen_ids);
            }
            gen_ids.push(first_tok);
            if !on_token(first_tok) || max_new_tokens == 1 {
                return Ok(gen_ids);
            }

            self.inner
                .init_seen(&mut state.inner, &[first_tok], presence_pen);
            let guard = SeenGuard(&self.inner, &mut state.inner);
            let st_in = &mut *guard.1;

            // `first_tok` is emitted but unconsumed. Rolling checkpoints every 8 steps bound
            // EOS/cancellation replay to <= 7 trunk-only (`compute_mode = 0`) steps in 1 CmdBuf,
            // while optional lossless 3-gram Prompt-Lookup / Output N-gram runs `M = 4` bursts.
            const BURST: usize = 4;
            const CKPT_WINDOW: usize = 8;
            let start_pos = *pos;
            self.inner.generation_checkpoint(st_in, false)?;
            let mut ckpt_consumed = 0usize;
            let mut pending = std::collections::VecDeque::new();
            let mut submitted = 0usize;
            let mut consumed = 0usize;
            let total = max_new_tokens - 1;
            let use_ngram = self
                .ngram_spec
                .load(std::sync::atomic::Ordering::Relaxed)
                && total >= 4;
            let mut ngram =
                use_ngram.then(|| crate::turbo_ngram::TurboNgramTable::new(prompt_ids, first_tok));
            let mut spec_streak = 0u32;
            let mut spec_stats = [0usize; 2];
            let head_mode = if use_sketch { 2u32 } else { 3u32 };
            self.inner.ring_set(0, first_tok);
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                    loop {
                        if consumed >= total {
                            return Ok(());
                        }
                        if pending.is_empty() && submitted == consumed {
                            if consumed.saturating_sub(ckpt_consumed) >= CKPT_WINDOW {
                                self.inner.generation_checkpoint(st_in, false)?;
                                ckpt_consumed = consumed;
                            }
                            if let Some(ng) = ngram.as_mut() {
                                if total - consumed >= 4 {
                                    ng.sync_generated(&gen_ids);
                                    if let Some((d, dkey)) = ng.draft3(spec_streak > 0, stop_ids) {
                                        let prev = consumed;
                                        let cur_tok = gen_ids[prev];
                                        let mut out4 = [0u32; 4];
                                        self.inner.step_batch(
                                            st_in,
                                            &[cur_tok, d[0], d[1], d[2]],
                                            (start_pos + prev) as u32,
                                            true,
                                            head_mode,
                                            None,
                                            Some(&mut out4),
                                        )?;
                                        ckpt_consumed = prev + 1;
                                        submitted = prev + 4;
                                        let mut acc = 0usize;
                                        for k in 0..4 {
                                            let pred = out4[k];
                                            acc += 1;
                                            consumed = gen_ids.len();
                                            if is_stop(pred) {
                                                self.inner.rollback_deltanet(st_in, acc as u32, 4)?;
                                                submitted = consumed;
                                                return Ok(());
                                            }
                                            gen_ids.push(pred);
                                            if presence_pen > 0.0 {
                                                self.inner.mark_seen(st_in, pred);
                                            }
                                            if !on_token(pred) || consumed >= total {
                                                self.inner.rollback_deltanet(st_in, acc as u32, 4)?;
                                                submitted = consumed;
                                                return Ok(());
                                            }
                                            if k < 3 && pred != d[k] {
                                                break;
                                            }
                                        }
                                        if acc < 4 {
                                            self.inner.rollback_deltanet(st_in, acc as u32, 4)?;
                                        }
                                        spec_stats[0] += 1;
                                        spec_stats[1] += acc;
                                        submitted = consumed;
                                        if acc == 1 {
                                            ng.invalidate(dkey);
                                            spec_streak = 0;
                                        } else {
                                            spec_streak = spec_streak.saturating_add(1);
                                        }
                                        self.inner
                                            .ring_set((submitted & 255) as u32, *gen_ids.last().unwrap());
                                        continue;
                                    }
                                    spec_streak = 0;
                                }
                            }
                        }
                        // Copy-heavy output: if the current suffix already has an N-gram hit, let the
                        // pipeline drain after the in-flight burst so the next step can be an M=4 draft.
                        let ngram_hint = pending.len() == 1
                            && ngram.as_mut().is_some_and(|ng| {
                                ng.sync_generated(&gen_ids);
                                ng.draft3(spec_streak > 0, stop_ids).is_some()
                            });
                        while pending.len() < 2
                            && submitted < total
                            && !(ngram_hint && !pending.is_empty())
                            && (pending.is_empty()
                                || submitted.saturating_sub(ckpt_consumed) < CKPT_WINDOW)
                        {
                            let rem_win = CKPT_WINDOW
                                .saturating_sub(submitted.saturating_sub(ckpt_consumed))
                                .max(BURST);
                            let count = (total - submitted).min(BURST).min(rem_win);
                            let slot = (submitted & 255) as u32;
                            let cmd = self.inner.enqueue_async(
                                st_in,
                                slot,
                                count as u32,
                                (start_pos + submitted) as u32,
                                use_sketch,
                            )?;
                            pending.push_back((cmd, slot, count));
                            submitted += count;
                        }
                        let Some((cmd, slot, count)) = pending.pop_front() else {
                            return Ok(());
                        };
                        self.inner.wait_cmd(cmd)?;
                        for k in 0..count {
                            let tok = self.inner.ring_get((slot + k as u32) & 255);
                            consumed = gen_ids.len();
                            if is_stop(tok) {
                                return Ok(());
                            }
                            gen_ids.push(tok);
                            if !on_token(tok) {
                                return Ok(());
                            }
                        }
                    }
                }));
            let (result, panic) = match outcome {
                Ok(result) => (result, None),
                Err(payload) => (Ok(()), Some(payload)),
            };
            // A submitted command owns scratch/ring/state buffers until drained,
            // including when another command or enqueue failed. Never restore early.
            let mut drain_error = None;
            while let Some((cmd, _, _)) = pending.pop_front() {
                if let Err(error) = self.inner.wait_cmd(cmd) {
                    if drain_error.is_none() {
                        drain_error = Some(error);
                    }
                }
            }
            if std::env::var_os("VXT_SPEC_STATS").is_some() {
                eprintln!(
                    "[VXT_SPEC] m4_steps={} m4_tokens={} total_tokens={} mean_accept={:.2}",
                    spec_stats[0], spec_stats[1], gen_ids.len(),
                    spec_stats[1] as f64 / spec_stats[0].max(1) as f64
                );
            }
            let recovery = (|| -> Result<()> {
                if consumed != submitted
                    || result.is_err()
                    || drain_error.is_some()
                    || panic.is_some()
                {
                    self.inner.generation_checkpoint(st_in, true)?;
                    let replay_beg = ckpt_consumed.min(consumed);
                    self.inner.replay_trunk(
                        st_in,
                        &gen_ids[replay_beg..consumed],
                        (start_pos + replay_beg) as u32,
                    )?;
                    *pos = start_pos + consumed;
                } else {
                    *pos = start_pos + consumed;
                }
                Ok(())
            })();
            if let Some(payload) = panic {
                std::panic::resume_unwind(payload);
            }
            recovery?;
            result?;
            if let Some(error) = drain_error {
                return Err(error);
            }
            Ok(gen_ids)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (
                state,
                pos,
                prompt_ids,
                first_tok,
                max_new_tokens,
                use_sketch,
                presence_pen,
                stop_ids,
                &mut on_token,
            );
            Err(Error::Unsupported("Metal unavailable".into()))
        }
    }
}
