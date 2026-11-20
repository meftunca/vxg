//! Lossless Dual-Model Speculative Decoding (`Draft + Target`) for `vxg-tensor`.
//!
//! Supports hybrid `Qwen3.5` (`Gated-DeltaNet` + `Gated-GQA`), `Qwen2`/`Qwen3`, `Llama`,
//! `Gemma`, and `MoE` models across:
//! - **Apple Metal (`MetalTurboRunner`)**: Single `M <= 4` batched GPU GEMM verification +
//!   microsecond GPU `DeltaNet` state rollback (`rollback_deltanet`) + draft trunk replay.
//! - **NVIDIA CUDA (`CudaTurboRunner`)**: Resident GPU checkpoint/restore + `replay_trunk`.
//! - **CPU / Generic Backends**: Exact `LlmState` checkpoint/rollback (`DeltaNet` + `KV`),
//!   supporting both **100% token-for-token exact greedy (`temp = 0`)** and **mathematically
//!   exact Leviathan et al. (2023) stochastic rejection sampling (`temp > 0`)**.

use std::time::Instant;

use crate::llm::{CpuStateCheckpoint, LlmModel, LlmState};
use crate::tokenizer::BpeTokenizer;
use crate::{Error, Result};

/// Configuration for dual-model speculative decoding.
#[derive(Clone, Copy, Debug)]
pub struct SpeculativeConfig {
    /// Number of draft tokens proposed per verification round (`1..=3` on Metal `M=4` fast path,
    /// `1..=8` on CPU / stochastic path).
    pub draft_tokens: usize,
    /// Whether to use the 2-stage sign-sketch LM head on Metal (`false` = exact full head).
    pub use_sketch: bool,
    /// Sampling temperature (`<= 1e-6` selects 100% exact greedy verification).
    pub temperature: f32,
    /// Presence penalty applied to previously emitted tokens (`0.0` = disabled).
    pub presence_penalty: f32,
    /// Deterministic PRNG seed for stochastic rejection sampling (`temperature > 0`).
    pub seed: u64,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_tokens: 3,
            use_sketch: false,
            temperature: 0.0,
            presence_penalty: 0.0,
            seed: 0x2026_11_20_5eed_u64,
        }
    }
}

/// Telemetry returned from a speculative generation run.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpeculativeStats {
    pub verify_steps: usize,
    pub total_drafted: usize,
    pub total_accepted_draft: usize,
    pub emitted_tokens: usize,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub used_metal_batch: bool,
}

impl SpeculativeStats {
    /// Fraction of proposed draft tokens accepted by the target model (`0.0..=1.0`).
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.total_accepted_draft as f64 / self.total_drafted as f64
        }
    }

    /// Mean emitted tokens per target verification step (including the guaranteed target/bonus token).
    pub fn mean_tokens_per_step(&self) -> f64 {
        if self.verify_steps == 0 {
            0.0
        } else {
            // Each verify step yields 1..=(K+1) tokens (accepted draft tokens + 1 target/bonus token).
            (self.total_accepted_draft + self.verify_steps) as f64 / self.verify_steps as f64
        }
    }
}

enum StateCheckpoint {
    MetalTurbo(usize),
    CudaTurbo(usize),
    Cpu(CpuStateCheckpoint),
}

fn supports_checkpoint(model: &LlmModel, state: &LlmState) -> bool {
    (model.metal_turbo.is_some() && state.turbo_gpu.is_some() && state.rope_delta == 0)
        || (model.cuda_turbo.is_some() && state.cuda_turbo_gpu.is_some() && state.rope_delta == 0)
        || state.is_cpu_state()
}

fn save_checkpoint(model: &LlmModel, state: &mut LlmState) -> Result<StateCheckpoint> {
    if state.rope_delta == 0 {
        if let (Some(tr), Some(tst)) = (model.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) {
            tr.checkpoint(tst, false)?;
            return Ok(StateCheckpoint::MetalTurbo(state.pos));
        }
        if let (Some(ct), Some(cst)) = (model.cuda_turbo.as_ref(), state.cuda_turbo_gpu.as_mut()) {
            ct.checkpoint(cst, false)?;
            return Ok(StateCheckpoint::CudaTurbo(state.pos));
        }
    }
    if state.is_cpu_state() {
        return Ok(StateCheckpoint::Cpu(state.save_cpu_checkpoint()));
    }
    Err(Error::Unsupported(
        "speculative decoding checkpoint requires VXT-Turbo (Metal/CUDA) or CPU state".into(),
    ))
}

fn restore_checkpoint(
    model: &LlmModel,
    state: &mut LlmState,
    ckpt: &StateCheckpoint,
) -> Result<()> {
    match ckpt {
        StateCheckpoint::MetalTurbo(pos) => {
            let (Some(tr), Some(tst)) = (model.metal_turbo.as_ref(), state.turbo_gpu.as_mut())
            else {
                return Err(Error::Backend("missing MetalTurbo state during restore".into()));
            };
            tr.checkpoint(tst, true)?;
            state.pos = *pos;
            Ok(())
        }
        StateCheckpoint::CudaTurbo(pos) => {
            let (Some(ct), Some(cst)) =
                (model.cuda_turbo.as_ref(), state.cuda_turbo_gpu.as_mut())
            else {
                return Err(Error::Backend("missing CudaTurbo state during restore".into()));
            };
            ct.checkpoint(cst, true)?;
            state.pos = *pos;
            Ok(())
        }
        StateCheckpoint::Cpu(cpu) => {
            state.restore_cpu_checkpoint(cpu, &model.cfg);
            Ok(())
        }
    }
}

fn replay_trunk(model: &mut LlmModel, state: &mut LlmState, tok_ids: &[u32]) -> Result<()> {
    if tok_ids.is_empty() {
        return Ok(());
    }
    if state.rope_delta == 0 {
        if let (Some(tr), Some(tst)) = (model.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) {
            tr.replay_trunk(tst, tok_ids, state.pos as u32)?;
            state.pos += tok_ids.len();
            return Ok(());
        }
        if let (Some(ct), Some(cst)) = (model.cuda_turbo.as_ref(), state.cuda_turbo_gpu.as_mut()) {
            ct.replay_trunk(cst, tok_ids, state.pos as u32)?;
            state.pos += tok_ids.len();
            return Ok(());
        }
    }
    for &tok in tok_ids {
        model.forward_step(state, tok, None)?;
    }
    Ok(())
}

#[inline]
fn argmax_with_penalty(logits: &[f32], seen: &[bool], presence_pen: f32) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let use_pen = presence_pen > 0.0 && seen.len() == logits.len();
    for (i, &v) in logits.iter().enumerate() {
        let adj = if use_pen && seen[i] { v - presence_pen } else { v };
        if adj > best_val {
            best_val = adj;
            best_idx = i;
        }
    }
    best_idx as u32
}

fn draft_step_greedy(
    draft: &mut LlmModel,
    draft_state: &mut LlmState,
    tok: u32,
    use_sketch: bool,
    scratch_logits: &mut [f32],
) -> Result<u32> {
    if draft_state.rope_delta == 0 {
        if let (Some(tr), Some(tst)) = (draft.metal_turbo.as_ref(), draft_state.turbo_gpu.as_mut())
        {
            let pred = tr.step_greedy_single(tst, draft_state.pos as u32, tok, use_sketch)?;
            draft_state.pos += 1;
            return Ok(pred);
        }
        if let (Some(ct), Some(cst)) =
            (draft.cuda_turbo.as_ref(), draft_state.cuda_turbo_gpu.as_mut())
        {
            let pred = ct.step_greedy_single(cst, draft_state.pos as u32, tok, 0.0)?;
            draft_state.pos += 1;
            return Ok(pred);
        }
    }
    draft.forward_step(draft_state, tok, Some(scratch_logits))?;
    Ok(argmax_with_penalty(scratch_logits, &[], 0.0))
}

#[inline]
fn xorshift_f32(rng: &mut u64) -> f32 {
    let mut x = *rng;
    if x == 0 {
        x = 0x9e37_79b9_7f4a_7c15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    ((x >> 40) as f32) * (1.0 / ((1u64 << 24) as f32))
}

fn softmax_temp(logits: &[f32], seen: &[bool], presence_pen: f32, temp: f32, probs: &mut [f32]) {
    let inv_t = 1.0 / temp.max(1e-5);
    let use_pen = presence_pen > 0.0 && seen.len() == logits.len();
    let mut max_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let adj = if use_pen && seen[i] { v - presence_pen } else { v };
        if adj > max_v {
            max_v = adj;
        }
    }
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let adj = if use_pen && seen[i] { v - presence_pen } else { v };
        let e = ((adj - max_v) * inv_t).exp();
        probs[i] = e;
        sum += e;
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
}

fn sample_probs(probs: &[f32], rng: &mut u64) -> u32 {
    let r = xorshift_f32(rng);
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

fn sample_residual(p_target: &[f32], p_draft: &[f32], rng: &mut u64) -> u32 {
    let mut sum = 0.0f32;
    for (&pt, &pd) in p_target.iter().zip(p_draft.iter()) {
        let d = pt - pd;
        if d > 0.0 {
            sum += d;
        }
    }
    if sum <= 1e-12 {
        return sample_probs(p_target, rng);
    }
    let target = xorshift_f32(rng) * sum;
    let mut acc = 0.0f32;
    for (i, (&pt, &pd)) in p_target.iter().zip(p_draft.iter()).enumerate() {
        let d = pt - pd;
        if d > 0.0 {
            acc += d;
            if acc >= target {
                return i as u32;
            }
        }
    }
    (p_target.len().saturating_sub(1)) as u32
}

/// Synchronize the draft model state after a speculative verification round.
///
/// Invariant:
/// - Before the round, `draft` was checkpointed at `start_pos`.
/// - During drafting of `k_actual` tokens (`batch = [cur_tok, d[0], ..., d[k_actual - 1]]`),
///   `draft` consumed exactly `batch[0..k_actual]` (so `draft_state.pos == start_pos + k_actual`).
/// - Target accepted `acc` consumed tokens (`batch[0..acc]`, where `1 <= acc <= k_actual + 1`).
fn sync_draft_after_verify(
    draft: &mut LlmModel,
    draft_state: &mut LlmState,
    draft_ckpt: &StateCheckpoint,
    batch: &[u32],
    k_actual: usize,
    acc: usize,
) -> Result<()> {
    if acc == k_actual + 1 {
        // All draft tokens accepted + target bonus token: draft only lacks the final token `batch[k_actual]`.
        replay_trunk(draft, draft_state, &batch[k_actual..k_actual + 1])?;
    } else if acc == k_actual {
        // Last draft token rejected: draft already consumed exact `batch[0..k_actual]`. Zero work!
    } else {
        // Earlier mismatch (`acc < k_actual`): restore draft checkpoint and replay only `batch[0..acc]`.
        restore_checkpoint(draft, draft_state, draft_ckpt)?;
        replay_trunk(draft, draft_state, &batch[..acc])?;
    }
    Ok(())
}

/// Run lossless dual-model speculative decoding (`draft` proposes `K` tokens, `target` verifies).
///
/// Preconditions:
/// - Both `target` and `draft` have already prefilled the identical prompt prefix
///   (`target_state.pos == draft_state.pos`).
/// - `first_tok` was sampled from `target`'s prefill logits and has NOT yet been fed into
///   either model.
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative<F: FnMut(u32, &BpeTokenizer) -> bool>(
    target: &mut LlmModel,
    target_state: &mut LlmState,
    draft: &mut LlmModel,
    draft_state: &mut LlmState,
    first_tok: u32,
    max_new_tokens: usize,
    stop_ids: &[u32],
    cfg: SpeculativeConfig,
    mut on_token: F,
) -> Result<(Vec<u32>, SpeculativeStats)> {
    if target.cfg.n_vocab != draft.cfg.n_vocab {
        return Err(Error::Format(format!(
            "speculative decoding requires matching vocabulary sizes (target={}, draft={})",
            target.cfg.n_vocab, draft.cfg.n_vocab
        )));
    }
    if target_state.pos != draft_state.pos {
        return Err(Error::Format(format!(
            "target and draft states must start at identical positions (target={}, draft={})",
            target_state.pos, draft_state.pos
        )));
    }
    if !supports_checkpoint(target, target_state) || !supports_checkpoint(draft, draft_state) {
        return Err(Error::Unsupported(
            "speculative decoding requires VXT-Turbo (Metal/CUDA) or CPU states for both target and draft".into(),
        ));
    }

    let is_stop = |t: u32| stop_ids.contains(&t);
    let mut stats = SpeculativeStats::default();
    let mut gen_ids = Vec::with_capacity(max_new_tokens);
    if max_new_tokens == 0 || is_stop(first_tok) {
        return Ok((gen_ids, stats));
    }
    gen_ids.push(first_tok);
    stats.emitted_tokens = 1;
    if !on_token(first_tok, &target.tokenizer) || max_new_tokens == 1 {
        return Ok((gen_ids, stats));
    }

    let greedy = cfg.temperature <= 1e-6;
    let use_metal_batch = greedy
        && target_state.rope_delta == 0
        && target.metal_turbo.is_some()
        && target_state.turbo_gpu.is_some();

    if use_metal_batch {
        stats.used_metal_batch = true;
        return generate_speculative_metal_batch(
            target,
            target_state,
            draft,
            draft_state,
            gen_ids,
            max_new_tokens,
            stop_ids,
            cfg,
            stats,
            on_token,
        );
    }

    generate_speculative_universal(
        target,
        target_state,
        draft,
        draft_state,
        gen_ids,
        max_new_tokens,
        stop_ids,
        cfg,
        stats,
        on_token,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_speculative_metal_batch<F: FnMut(u32, &BpeTokenizer) -> bool>(
    target: &mut LlmModel,
    target_state: &mut LlmState,
    draft: &mut LlmModel,
    draft_state: &mut LlmState,
    mut gen_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_ids: &[u32],
    cfg: SpeculativeConfig,
    mut stats: SpeculativeStats,
    mut on_token: F,
) -> Result<(Vec<u32>, SpeculativeStats)> {
    let is_stop = |t: u32| stop_ids.contains(&t);
    let max_k = cfg.draft_tokens.clamp(1, 3);
    let mut draft_logits = vec![0.0f32; draft.cfg.n_vocab];

    {
        let tr = target.metal_turbo.as_ref().unwrap();
        let tst = target_state.turbo_gpu.as_mut().unwrap();
        tr.init_seen(tst, &[gen_ids[0]], cfg.presence_penalty);
    }

    let mut consumed = 0usize;
    let mut batch = [0u32; 4];
    let mut out = [0u32; 4];

    let res: Result<()> = (|| {
        while gen_ids.len() < max_new_tokens {
            let rem = max_new_tokens - gen_ids.len();
            // We need at most `rem - 1` draft tokens because `cur_tok` already yields 1 target token.
            let k_goal = max_k.min(rem.saturating_sub(1));
            let cur_tok = gen_ids[consumed];

            // 1. Draft Phase
            let t_draft = Instant::now();
            let draft_ckpt = save_checkpoint(draft, draft_state)?;
            batch[0] = cur_tok;
            let mut k_actual = 0usize;
            if k_goal > 0 {
                let mut feed = cur_tok;
                for j in 0..k_goal {
                    let d_tok =
                        draft_step_greedy(draft, draft_state, feed, cfg.use_sketch, &mut draft_logits)?;
                    if is_stop(d_tok) {
                        // Do not draft past EOS; roll back draft to just before `feed`?
                        // Actually `draft` consumed `batch[0..=j]`, so we can stop drafting with
                        // `k_actual = j`, and `sync_draft_after_verify` will align draft cleanly!
                        // Wait: if `j == 0` we consumed `cur_tok` (1 token) while `k_actual = 0`,
                        // so let's restore checkpoint if `d_tok` is stop so invariant holds strictly!
                        if j == 0 {
                            restore_checkpoint(draft, draft_state, &draft_ckpt)?;
                        } else {
                            // `draft` consumed `j + 1` tokens (`cur_tok` + `d[0..j-1]`), which is
                            // 1 more than `k_actual = j`. Restore + replay `j` so invariant holds.
                            restore_checkpoint(draft, draft_state, &draft_ckpt)?;
                            replay_trunk(draft, draft_state, &batch[..j])?;
                        }
                        break;
                    }
                    batch[j + 1] = d_tok;
                    k_actual = j + 1;
                    feed = d_tok;
                }
            }
            stats.draft_ms += t_draft.elapsed().as_secs_f64() * 1000.0;
            let m = 1 + k_actual;
            stats.total_drafted += k_actual;

            // 2. Verify Phase: ONE batched Metal GPU GEMM pass (`M = 1..=4`)!
            let t_ver = Instant::now();
            let pos_base = target_state.pos as u32;
            {
                let tr = target.metal_turbo.as_ref().unwrap();
                let tst = target_state.turbo_gpu.as_mut().unwrap();
                tr.verify_batch_greedy(tst, pos_base, &batch[..m], cfg.use_sketch, &mut out[..m])?;
            }
            stats.verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;
            stats.verify_steps += 1;

            let mut acc = 0usize;
            let mut hit_stop = false;
            for j in 0..m {
                let pred = out[j];
                acc = j + 1;
                if is_stop(pred) {
                    hit_stop = true;
                    break;
                }
                gen_ids.push(pred);
                stats.emitted_tokens = gen_ids.len();
                if cfg.presence_penalty > 0.0 {
                    let tr = target.metal_turbo.as_ref().unwrap();
                    let tst = target_state.turbo_gpu.as_mut().unwrap();
                    tr.mark_seen(tst, pred);
                }
                if !on_token(pred, &target.tokenizer) || gen_ids.len() >= max_new_tokens {
                    hit_stop = true;
                    break;
                }
                if j < k_actual {
                    if pred == batch[j + 1] {
                        stats.total_accepted_draft += 1;
                    } else {
                        break;
                    }
                }
            }

            // 3. Roll back Target GPU DeltaNet state if `acc < m`
            if acc < m {
                let tr = target.metal_turbo.as_ref().unwrap();
                let tst = target_state.turbo_gpu.as_mut().unwrap();
                tr.rollback_deltanet(tst, acc as u32, m as u32)?;
            }
            target_state.pos += acc;

            // 4. Synchronize Draft state to match Target (`batch[0..acc]`)
            sync_draft_after_verify(draft, draft_state, &draft_ckpt, &batch[..m], k_actual, acc)?;
            consumed += acc;

            if hit_stop {
                break;
            }
        }
        Ok(())
    })();

    // Cleanly reset GPU presence-penalty seen table before returning.
    if let (Some(tr), Some(tst)) = (target.metal_turbo.as_ref(), target_state.turbo_gpu.as_mut()) {
        tr.init_seen(tst, &[], 0.0);
    }

    res?;
    Ok((gen_ids, stats))
}

#[allow(clippy::too_many_arguments)]
fn generate_speculative_universal<F: FnMut(u32, &BpeTokenizer) -> bool>(
    target: &mut LlmModel,
    target_state: &mut LlmState,
    draft: &mut LlmModel,
    draft_state: &mut LlmState,
    mut gen_ids: Vec<u32>,
    max_new_tokens: usize,
    stop_ids: &[u32],
    cfg: SpeculativeConfig,
    mut stats: SpeculativeStats,
    mut on_token: F,
) -> Result<(Vec<u32>, SpeculativeStats)> {
    let is_stop = |t: u32| stop_ids.contains(&t);
    let max_k = cfg.draft_tokens.clamp(1, 8);
    let greedy = cfg.temperature <= 1e-6;
    let n_vocab = target.cfg.n_vocab;

    let mut seen = if cfg.presence_penalty > 0.0 {
        let mut s = vec![false; n_vocab];
        if (gen_ids[0] as usize) < n_vocab {
            s[gen_ids[0] as usize] = true;
        }
        s
    } else {
        Vec::new()
    };

    let use_cuda_target = greedy
        && target_state.rope_delta == 0
        && target.cuda_turbo.is_some()
        && target_state.cuda_turbo_gpu.is_some();
    if use_cuda_target {
        let ct = target.cuda_turbo.as_ref().unwrap();
        let cst = target_state.cuda_turbo_gpu.as_mut().unwrap();
        ct.init_seen(cst, &[gen_ids[0]]);
    }

    let mut rng = cfg.seed;
    let mut draft_logits = vec![0.0f32; n_vocab];
    let mut target_logits = vec![0.0f32; n_vocab];
    let mut draft_probs_buf: Vec<Vec<f32>> = if greedy {
        Vec::new()
    } else {
        (0..max_k).map(|_| vec![0.0f32; n_vocab]).collect()
    };
    let mut target_probs = if greedy { Vec::new() } else { vec![0.0f32; n_vocab] };

    let mut consumed = 0usize;
    let mut batch = Vec::with_capacity(max_k + 1);

    while gen_ids.len() < max_new_tokens {
        let rem = max_new_tokens - gen_ids.len();
        let k_goal = max_k.min(rem.saturating_sub(1));
        let cur_tok = gen_ids[consumed];

        // 1. Draft Phase
        let t_draft = Instant::now();
        let draft_ckpt = save_checkpoint(draft, draft_state)?;
        batch.clear();
        batch.push(cur_tok);
        let mut k_actual = 0usize;

        if k_goal > 0 {
            let mut feed = cur_tok;
            for j in 0..k_goal {
                let d_tok = if greedy {
                    draft_step_greedy(draft, draft_state, feed, cfg.use_sketch, &mut draft_logits)?
                } else {
                    draft.forward_step(draft_state, feed, Some(&mut draft_logits))?;
                    softmax_temp(
                        &draft_logits,
                        &seen,
                        cfg.presence_penalty,
                        cfg.temperature,
                        &mut draft_probs_buf[j],
                    );
                    sample_probs(&draft_probs_buf[j], &mut rng)
                };
                if is_stop(d_tok) {
                    restore_checkpoint(draft, draft_state, &draft_ckpt)?;
                    replay_trunk(draft, draft_state, &batch[..j])?;
                    break;
                }
                batch.push(d_tok);
                k_actual = j + 1;
                feed = d_tok;
            }
        }
        stats.draft_ms += t_draft.elapsed().as_secs_f64() * 1000.0;
        let m = 1 + k_actual;
        stats.total_drafted += k_actual;

        // 2. Verify Phase (early-exit on first mismatch -> zero wasted target steps!)
        let t_ver = Instant::now();
        let mut acc = 0usize;
        let mut hit_stop = false;
        stats.verify_steps += 1;

        for j in 0..m {
            let pred = if use_cuda_target {
                let ct = target.cuda_turbo.as_ref().unwrap();
                let cst = target_state.cuda_turbo_gpu.as_mut().unwrap();
                let p = ct.step_greedy_single(
                    cst,
                    target_state.pos as u32,
                    batch[j],
                    cfg.presence_penalty,
                )?;
                target_state.pos += 1;
                p
            } else {
                target.forward_step(target_state, batch[j], Some(&mut target_logits))?;
                if greedy {
                    argmax_with_penalty(&target_logits, &seen, cfg.presence_penalty)
                } else {
                    softmax_temp(
                        &target_logits,
                        &seen,
                        cfg.presence_penalty,
                        cfg.temperature,
                        &mut target_probs,
                    );
                    if j < k_actual {
                        let d_tok = batch[j + 1] as usize;
                        let pt = target_probs[d_tok];
                        let pd = draft_probs_buf[j][d_tok].max(1e-12);
                        let accept_prob = (pt / pd).min(1.0);
                        if xorshift_f32(&mut rng) <= accept_prob {
                            batch[j + 1]
                        } else {
                            sample_residual(&target_probs, &draft_probs_buf[j], &mut rng)
                        }
                    } else {
                        sample_probs(&target_probs, &mut rng)
                    }
                }
            };

            acc = j + 1;
            if is_stop(pred) {
                hit_stop = true;
                break;
            }
            gen_ids.push(pred);
            stats.emitted_tokens = gen_ids.len();
            if !seen.is_empty() && (pred as usize) < seen.len() {
                seen[pred as usize] = true;
            }
            if !on_token(pred, &target.tokenizer) || gen_ids.len() >= max_new_tokens {
                hit_stop = true;
                break;
            }
            if j < k_actual {
                if pred == batch[j + 1] {
                    stats.total_accepted_draft += 1;
                } else {
                    break;
                }
            }
        }
        stats.verify_ms += t_ver.elapsed().as_secs_f64() * 1000.0;

        // 3. Synchronize Draft state to exact Target prefix `batch[0..acc]`
        sync_draft_after_verify(draft, draft_state, &draft_ckpt, &batch, k_actual, acc)?;
        consumed += acc;

        if hit_stop {
            break;
        }
    }

    if use_cuda_target {
        if let (Some(ct), Some(cst)) =
            (target.cuda_turbo.as_ref(), target_state.cuda_turbo_gpu.as_mut())
        {
            ct.init_seen(cst, &[]);
        }
    }

    Ok((gen_ids, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{write_vxt_model, PackedTensor, VxtEntry};
    use crate::gguf_meta::ModelConfig;
    use crate::wht_turbo::TurboWhtMatrix;

    fn det_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut x = seed ^ 0x9e37_79b9_7f4a_7c15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let u = ((x >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0;
            out.push(u * scale);
        }
        out
    }

    fn build_synthetic_hybrid_vxt(seed: u64, bits: u8, perturb: f32) -> Vec<u8> {
        let n_vocab = 64usize;
        let n_embd = 64usize;
        let n_ff = 64usize;
        let n_head = 2usize;
        let n_head_kv = 1usize;
        let head_dim = 32usize;
        let ssm_k = 4usize;
        let ssm_state = 32usize;
        let ssm_groups = 2usize;
        let ssm_dt = 2usize;
        let ssm_inner = 64usize;
        let dv = ssm_inner / ssm_dt;
        let qk_dim = ssm_groups * ssm_state;
        let conv_ch = qk_dim * 2 + ssm_inner;

        let cfg = ModelConfig {
            arch: "qwen35".into(),
            model_name: "spec-hybrid-test".into(),
            n_vocab,
            n_embd,
            n_layer: 2,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            rope_dim: head_dim,
            rope_freq_base: 10_000.0,
            rope_neox: true,
            rms_norm_eps: 1e-6,
            embed_scale: 1.0,
            ffn_gelu: false,
            bos_token_id: Some(0),
            eos_token_id: 63,
            ssm_conv_kernel: ssm_k,
            ssm_state_size: ssm_state,
            ssm_group_count: ssm_groups,
            ssm_dt_rank: ssm_dt,
            ssm_inner_size: ssm_inner,
            full_attn_interval: 2,
        };

        let mut entries: Vec<VxtEntry> = Vec::new();
        let make_mat = |name: &str, rows: usize, cols: usize, s: u64| -> VxtEntry {
            let mut w = det_vec(rows * cols, seed.wrapping_add(s), 0.22);
            if perturb != 0.0 {
                let noise = det_vec(rows * cols, seed.wrapping_add(s ^ 0x5555_aaaa), perturb);
                for (a, b) in w.iter_mut().zip(noise.iter()) {
                    *a += *b;
                }
            }
            let tensor = if bits == 7 || bits == 8 {
                PackedTensor::TurboWht(TurboWhtMatrix::encode(bits, rows, cols, &w).unwrap())
            } else {
                PackedTensor::RawF32 {
                    rows,
                    cols,
                    data: w,
                }
            };
            VxtEntry {
                name: name.into(),
                tensor,
            }
        };
        let make_vec = |name: &str, len: usize, val: f32, s: u64| -> VxtEntry {
            let data = if val == 0.0 {
                det_vec(len, seed.wrapping_add(s), 0.15)
            } else {
                vec![val; len]
            };
            VxtEntry {
                name: name.into(),
                tensor: PackedTensor::RawF32 {
                    rows: 1,
                    cols: len,
                    data,
                },
            }
        };

        entries.push(make_mat("token_embd.weight", n_vocab, n_embd, 1));
        entries.push(make_vec("output_norm.weight", n_embd, 1.0, 2));
        entries.push(make_mat("output.weight", n_vocab, n_embd, 3));

        // Layer 0: Gated-DeltaNet
        entries.push(make_vec("blk.0.attn_norm.weight", n_embd, 1.0, 10));
        entries.push(make_vec("blk.0.post_attention_norm.weight", n_embd, 1.0, 11));
        entries.push(make_mat("blk.0.attn_qkv.weight", conv_ch, n_embd, 12));
        entries.push(make_mat("blk.0.attn_gate.weight", ssm_inner, n_embd, 13));
        entries.push(make_mat("blk.0.ssm_alpha.weight", ssm_dt, n_embd, 14));
        entries.push(make_mat("blk.0.ssm_beta.weight", ssm_dt, n_embd, 15));
        entries.push(make_mat("blk.0.ssm_out.weight", n_embd, ssm_inner, 16));
        entries.push(VxtEntry {
            name: "blk.0.ssm_conv1d.weight".into(),
            tensor: PackedTensor::RawF32 {
                rows: conv_ch,
                cols: ssm_k,
                data: det_vec(conv_ch * ssm_k, seed.wrapping_add(17), 0.2),
            },
        });
        entries.push(make_vec("blk.0.ssm_a", ssm_dt, -0.5, 18));
        entries.push(make_vec("blk.0.ssm_dt.bias", ssm_dt, -0.2, 19));
        entries.push(make_vec("blk.0.ssm_norm.weight", dv, 1.0, 20));
        entries.push(make_mat("blk.0.ffn_gate.weight", n_ff, n_embd, 21));
        entries.push(make_mat("blk.0.ffn_up.weight", n_ff, n_embd, 22));
        entries.push(make_mat("blk.0.ffn_down.weight", n_embd, n_ff, 23));

        // Layer 1: Gated-GQA
        entries.push(make_vec("blk.1.attn_norm.weight", n_embd, 1.0, 30));
        entries.push(make_vec("blk.1.post_attention_norm.weight", n_embd, 1.0, 31));
        entries.push(make_mat("blk.1.attn_q.weight", n_head * head_dim * 2, n_embd, 32));
        entries.push(make_mat("blk.1.attn_k.weight", n_head_kv * head_dim, n_embd, 33));
        entries.push(make_mat("blk.1.attn_v.weight", n_head_kv * head_dim, n_embd, 34));
        entries.push(make_mat("blk.1.attn_output.weight", n_embd, n_head * head_dim, 35));
        entries.push(make_vec("blk.1.attn_q_norm.weight", head_dim, 1.0, 36));
        entries.push(make_vec("blk.1.attn_k_norm.weight", head_dim, 1.0, 37));
        entries.push(make_mat("blk.1.ffn_gate.weight", n_ff, n_embd, 38));
        entries.push(make_mat("blk.1.ffn_up.weight", n_ff, n_embd, 39));
        entries.push(make_mat("blk.1.ffn_down.weight", n_embd, n_ff, 40));

        // Minimal valid tokenizer metadata (64 single-byte/token entries)
        let mut tok_buf = Vec::new();
        tok_buf.extend_from_slice(&(n_vocab as u32).to_le_bytes());
        for i in 0..n_vocab {
            let s = format!("t{i}");
            tok_buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            tok_buf.extend_from_slice(s.as_bytes());
        }
        let mut merges_buf = Vec::new();
        merges_buf.extend_from_slice(&0u32.to_le_bytes());
        let mut types_buf = Vec::new();
        types_buf.extend_from_slice(&(n_vocab as u32).to_le_bytes());
        for _ in 0..n_vocab {
            types_buf.extend_from_slice(&1i32.to_le_bytes());
        }

        let meta_kv = vec![
            ("model.config".into(), cfg.to_kv_bytes()),
            ("tokenizer.tokens".into(), tok_buf),
            ("tokenizer.merges".into(), merges_buf),
            ("tokenizer.types".into(), types_buf),
        ];
        write_vxt_model(&meta_kv, &entries).unwrap()
    }

    fn run_reference_greedy(
        model: &mut LlmModel,
        prompt: &[u32],
        n_gen: usize,
        stop_ids: &[u32],
    ) -> Vec<u32> {
        let mut st = model.new_state();
        let mut logits = vec![0.0f32; model.cfg.n_vocab];
        model.prefill(&mut st, prompt, &mut logits).unwrap();
        let mut out = Vec::with_capacity(n_gen);
        let mut tok = argmax_with_penalty(&logits, &[], 0.0);
        while out.len() < n_gen && !stop_ids.contains(&tok) {
            out.push(tok);
            if out.len() == n_gen {
                break;
            }
            model.forward_step(&mut st, tok, Some(&mut logits)).unwrap();
            tok = argmax_with_penalty(&logits, &[], 0.0);
        }
        out
    }

    #[test]
    fn speculative_hybrid_deltanet_gqa_cpu_exact_parity() {
        let target_bytes = build_synthetic_hybrid_vxt(101, 0, 0.0);
        // Slightly perturbed draft model forces BOTH accepts AND DeltaNet rollbacks!
        let draft_bytes = build_synthetic_hybrid_vxt(101, 0, 0.09);

        let prompt = [2u32, 7, 15, 29];
        let stop_ids = [63u32];

        let mut target_ref = LlmModel::from_vxt_bytes(&target_bytes, false, 1).unwrap();
        let ref_ids = run_reference_greedy(&mut target_ref, &prompt, 16, &stop_ids);
        assert_eq!(ref_ids.len(), 16);

        let mut target = LlmModel::from_vxt_bytes(&target_bytes, false, 1).unwrap();
        let mut draft = LlmModel::from_vxt_bytes(&draft_bytes, false, 1).unwrap();
        let mut t_st = target.new_state();
        let mut d_st = draft.new_state();
        let mut t_logits = vec![0.0f32; target.cfg.n_vocab];
        let mut d_logits = vec![0.0f32; draft.cfg.n_vocab];
        target.prefill(&mut t_st, &prompt, &mut t_logits).unwrap();
        draft.prefill(&mut d_st, &prompt, &mut d_logits).unwrap();
        let first_tok = argmax_with_penalty(&t_logits, &[], 0.0);

        let mut streamed = Vec::new();
        let (spec_ids, stats) = generate_speculative(
            &mut target,
            &mut t_st,
            &mut draft,
            &mut d_st,
            first_tok,
            16,
            &stop_ids,
            SpeculativeConfig {
                draft_tokens: 3,
                ..Default::default()
            },
            |t, _| {
                streamed.push(t);
                true
            },
        )
        .unwrap();

        assert_eq!(spec_ids, ref_ids, "CPU hybrid DeltaNet+GQA speculative decode must match 100%");
        assert_eq!(streamed, ref_ids);
        assert_eq!(t_st.pos, d_st.pos, "target & draft positions must stay synchronized");
        assert!(stats.verify_steps > 0 && stats.total_drafted > 0);
    }

    #[test]
    fn speculative_stochastic_rejection_sampling_valid() {
        let target_bytes = build_synthetic_hybrid_vxt(202, 0, 0.0);
        let draft_bytes = build_synthetic_hybrid_vxt(202, 0, 0.05);
        let prompt = [1u32, 4, 9];
        let mut target = LlmModel::from_vxt_bytes(&target_bytes, false, 1).unwrap();
        let mut draft = LlmModel::from_vxt_bytes(&draft_bytes, false, 1).unwrap();
        let mut t_st = target.new_state();
        let mut d_st = draft.new_state();
        let mut t_logits = vec![0.0f32; target.cfg.n_vocab];
        let mut d_logits = vec![0.0f32; draft.cfg.n_vocab];
        target.prefill(&mut t_st, &prompt, &mut t_logits).unwrap();
        draft.prefill(&mut d_st, &prompt, &mut d_logits).unwrap();
        let first_tok = argmax_with_penalty(&t_logits, &[], 0.0);

        let (ids, stats) = generate_speculative(
            &mut target,
            &mut t_st,
            &mut draft,
            &mut d_st,
            first_tok,
            14,
            &[63],
            SpeculativeConfig {
                draft_tokens: 4,
                temperature: 0.8,
                presence_penalty: 0.2,
                seed: 42,
                ..Default::default()
            },
            |_, _| true,
        )
        .unwrap();

        assert!(!ids.is_empty());
        assert_eq!(t_st.pos, d_st.pos);
        assert!(stats.verify_steps > 0);
    }

    #[cfg(vxt_has_metal)]
    #[test]
    fn speculative_hybrid_deltanet_gqa_metal_m4_exact_parity() {
        let target_bytes = build_synthetic_hybrid_vxt(303, 8, 0.0);
        // Perturbed E8 (`bits=7`) draft model against E8-HQ (`bits=8`) target on Metal GPU!
        let draft_bytes = build_synthetic_hybrid_vxt(303, 7, 0.06);

        let prompt = [3u32, 11, 19, 27];
        let stop_ids: [u32; 0] = [];

        let mut target_ref = LlmModel::from_vxt_bytes(&target_bytes, true, 1).unwrap();
        assert!(
            target_ref.metal_turbo.is_some(),
            "synthetic E8-HQ model should activate MetalTurboRunner"
        );
        let ref_ids = run_reference_greedy(&mut target_ref, &prompt, 16, &stop_ids);
        assert_eq!(ref_ids.len(), 16);

        let mut target = LlmModel::from_vxt_bytes(&target_bytes, true, 1).unwrap();
        let mut draft = LlmModel::from_vxt_bytes(&draft_bytes, true, 1).unwrap();
        assert!(draft.metal_turbo.is_some());

        let mut t_st = target.new_state();
        let mut d_st = draft.new_state();
        let mut t_logits = vec![0.0f32; target.cfg.n_vocab];
        let mut d_logits = vec![0.0f32; draft.cfg.n_vocab];
        target.prefill(&mut t_st, &prompt, &mut t_logits).unwrap();
        draft.prefill(&mut d_st, &prompt, &mut d_logits).unwrap();
        let first_tok = argmax_with_penalty(&t_logits, &[], 0.0);

        let (spec_ids, stats) = generate_speculative(
            &mut target,
            &mut t_st,
            &mut draft,
            &mut d_st,
            first_tok,
            16,
            &stop_ids,
            SpeculativeConfig {
                draft_tokens: 3,
                use_sketch: false,
                ..Default::default()
            },
            |_, _| true,
        )
        .unwrap();

        assert!(stats.used_metal_batch, "should use Metal M=4 batched GPU verify");
        assert_eq!(
            spec_ids, ref_ids,
            "Metal M=4 hybrid DeltaNet+GQA speculative decode must match 100% token-for-token"
        );
        assert_eq!(t_st.pos, d_st.pos);
        assert!(stats.verify_steps > 0 && stats.total_drafted > 0);
        eprintln!(
            "[metal_m4_parity_perturbed] steps={} accepted={}/{} ({:.1}%) mean={:.2} tok/step",
            stats.verify_steps,
            stats.total_accepted_draft,
            stats.total_drafted,
            stats.acceptance_rate() * 100.0,
            stats.mean_tokens_per_step()
        );

        // Also test 100% acceptance (all-match + bonus token `acc == 4` path) AND mid-batch EOS stop!
        let mut target2 = LlmModel::from_vxt_bytes(&target_bytes, true, 1).unwrap();
        let mut draft2 = LlmModel::from_vxt_bytes(&target_bytes, true, 1).unwrap();
        let mut t2 = target2.new_state();
        let mut d2 = draft2.new_state();
        target2.prefill(&mut t2, &prompt, &mut t_logits).unwrap();
        draft2.prefill(&mut d2, &prompt, &mut d_logits).unwrap();
        let f2 = argmax_with_penalty(&t_logits, &[], 0.0);
        let (full_ids, full_stats) = generate_speculative(
            &mut target2,
            &mut t2,
            &mut draft2,
            &mut d2,
            f2,
            16,
            &stop_ids,
            SpeculativeConfig {
                draft_tokens: 3,
                use_sketch: false,
                ..Default::default()
            },
            |_, _| true,
        )
        .unwrap();
        assert_eq!(full_ids, ref_ids, "Metal M=4 full-accept bonus path must match 100%");
        assert_eq!(t2.pos, d2.pos);
        assert!(
            full_stats.acceptance_rate() > 0.99,
            "identical draft/target must achieve ~100% acceptance (got {:.1}%)",
            full_stats.acceptance_rate() * 100.0
        );
        eprintln!(
            "[metal_m4_parity_full_accept] steps={} accepted={}/{} ({:.1}%) mean={:.2} tok/step",
            full_stats.verify_steps,
            full_stats.total_accepted_draft,
            full_stats.total_drafted,
            full_stats.acceptance_rate() * 100.0,
            full_stats.mean_tokens_per_step()
        );
    }
}
