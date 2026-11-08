//! Stateful multi-turn LLM conversation engine (`ChatSession`) with exact incremental `KV` +
//! `DeltaNet` prefix caching, `ChatML` / `Llama-3` template formatting, instant atomic abort,
//! and interactive terminal REPL (`vxg-tensor chat`).

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::llm::{sample_logits, LlmModel, LlmState};
use crate::llm_ops::{
    has_severe_loop, sample_logits_top_k_p, strip_think_block, TokenPenaltyTracker,
};
use crate::tokenizer::BpeTokenizer;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct ChatParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub presence_penalty: Option<f32>,
    pub enable_thinking: bool,
}

impl Default for ChatParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.0,
            presence_penalty: Some(0.0),
            enable_thinking: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TurnMetrics {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub prefill_tps: f64,
    pub decode_tps: f64,
    pub stream_gbs: f64,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub new_prefill_tokens: usize,
    pub generated_tokens: usize,
    pub finish_reason: String,
    pub decode_path: String,
}

#[derive(Clone, Debug)]
pub struct ModelSummary {
    pub model_path: String,
    pub model_name: String,
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub compressed_bytes: usize,
    pub backend: String,
    pub cb4_fast: bool,
    pub turbo_fast: bool,
    pub load_ms: f64,
}

/// Formats a multi-turn conversation so historical turns stay prefix-identical across turns.
pub fn format_chat_messages(
    tokenizer: &BpeTokenizer,
    arch: &str,
    messages: &[ChatMessage],
    enable_thinking: bool,
) -> Vec<u32> {
    format_chat_messages_cached(tokenizer, arch, messages, enable_thinking, &HashMap::new())
}

fn format_chat_messages_cached(
    tokenizer: &BpeTokenizer,
    arch: &str,
    messages: &[ChatMessage],
    enable_thinking: bool,
    asst_cache: &HashMap<String, (bool, Vec<u32>)>,
) -> Vec<u32> {
    let is_qwen_think = matches!(arch, "qwen3" | "qwen35");
    if tokenizer.token_id("<|im_start|>").is_some() && tokenizer.token_id("<|im_end|>").is_some() {
        let mut out = Vec::with_capacity(messages.len() * 32 + 16);
        let open_hdr = |think: bool| match (is_qwen_think, think) {
            (true, true) => "<|im_start|>assistant\n<think>\n",
            (true, false) => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            (false, _) => "<|im_start|>assistant\n",
        };
        let close_ids = tokenizer.encode("<|im_end|>\n");
        for msg in messages {
            let role = match msg.role.as_str() {
                "developer" | "system" => "system",
                "assistant" => "assistant",
                _ => "user",
            };
            if role == "assistant" {
                let stripped = strip_think_block(&msg.content);
                if stripped.len() == msg.content.len() {
                    if let Some((false, exact_ids)) = asst_cache.get(&msg.content) {
                        out.extend(tokenizer.encode(open_hdr(false)));
                        out.extend_from_slice(exact_ids);
                        out.extend_from_slice(&close_ids);
                        continue;
                    }
                }
                out.extend(tokenizer.encode(open_hdr(false)));
                out.extend(tokenizer.encode(stripped));
                out.extend_from_slice(&close_ids);
            } else {
                out.extend(tokenizer.encode(&format!("<|im_start|>{role}\n{}<|im_end|>\n", msg.content)));
            }
        }
        out.extend(tokenizer.encode(open_hdr(enable_thinking)));
        return out;
    }

    if tokenizer.token_id("<|start_header_id|>").is_some()
        && tokenizer.token_id("<|eot_id|>").is_some()
    {
        let mut out = tokenizer.encode("<|begin_of_text|>");
        let eot_ids = tokenizer.encode("<|eot_id|>");
        let asst_hdr = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n");
        for msg in messages {
            let role = if msg.role == "developer" { "system" } else { msg.role.as_str() };
            if role == "assistant" {
                if let Some((_, exact_ids)) = asst_cache.get(&msg.content) {
                    out.extend_from_slice(&asst_hdr);
                    out.extend_from_slice(exact_ids);
                    out.extend_from_slice(&eot_ids);
                    continue;
                }
            }
            out.extend(tokenizer.encode(&format!(
                "<|start_header_id|>{role}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.content.trim()
            )));
        }
        out.extend_from_slice(&asst_hdr);
        return out;
    }

    let mut buf = String::new();
    for msg in messages {
        buf.push_str(&msg.role);
        buf.push_str(": ");
        buf.push_str(&msg.content);
        buf.push('\n');
    }
    buf.push_str("assistant: ");
    tokenizer.encode(&buf)
}

fn flush_chunk<F: FnMut(&str) -> bool>(
    tok: &BpeTokenizer, ids: &[u32], start: &mut usize, full: &mut String, aborted: &mut bool, cb: &mut F,
) -> bool {
    let chunk = tok.decode(&ids[*start..]);
    if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
        full.push_str(&chunk);
        *start = ids.len();
        if !cb(&chunk) {
            *aborted = true;
            return false;
        }
    }
    true
}

pub struct ChatSession {
    pub model: LlmModel,
    pub state: LlmState,
    pub summary: ModelSummary,
    consumed_tokens: Vec<u32>,
    asst_cache: HashMap<String, (bool, Vec<u32>)>,
    logits: Vec<f32>,
    cb4_greedy_ready: bool,
    rng: u64,
}

impl ChatSession {
    pub fn load(path: &str, use_gpu: bool, threads: usize) -> Result<Self> {
        let t0 = Instant::now();
        let mut model = LlmModel::load_file(path, use_gpu, threads)?;
        let cb4_greedy_ready = model.enable_cb4_greedy().is_ok();
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let state = model.new_state();
        let n_vocab = model.cfg.n_vocab;
        let summary = ModelSummary {
            model_path: path.to_owned(),
            model_name: model.cfg.model_name.clone(),
            arch: model.cfg.arch.clone(),
            n_layer: model.cfg.n_layer,
            n_embd: model.cfg.n_embd,
            n_vocab,
            compressed_bytes: model.total_compressed_bytes,
            backend: model.backend_label(),
            cb4_fast: model.has_cb4_fast_decode() && cb4_greedy_ready,
            turbo_fast: model.has_turbo_fast_decode(),
            load_ms,
        };
        Ok(Self {
            model,
            state,
            summary,
            consumed_tokens: Vec::with_capacity(4096),
            asst_cache: HashMap::new(),
            logits: vec![0.0f32; n_vocab],
            cb4_greedy_ready,
            rng: 0x2026_11_08_5eed_u64,
        })
    }

    pub fn reset(&mut self) {
        self.state = self.model.new_state();
        self.consumed_tokens.clear();
        self.asst_cache.clear();
    }

    pub fn cached_token_count(&self) -> usize {
        self.consumed_tokens.len()
    }

    pub fn generate_chat<F: FnMut(&str) -> bool>(
        &mut self,
        messages: &[ChatMessage],
        params: &ChatParams,
        cancel: &AtomicBool,
        mut on_text: F,
    ) -> Result<(String, TurnMetrics)> {
        if messages.is_empty() {
            return Err(Error::Format("messages cannot be empty".into()));
        }
        cancel.store(false, Ordering::Relaxed);
        let max_tokens = params.max_tokens.clamp(1, 8192);
        let temp = params.temperature.max(0.0);
        let is_qwen_think = matches!(self.model.cfg.arch.as_str(), "qwen3" | "qwen35");

        let full_ids = format_chat_messages_cached(
            &self.model.tokenizer,
            &self.model.cfg.arch,
            messages,
            params.enable_thinking,
            &self.asst_cache,
        );
        if full_ids.is_empty() {
            return Err(Error::Format("chat prompt tokenized to 0 tokens".into()));
        }

        if self.consumed_tokens.len() != self.state.pos
            || !full_ids.starts_with(&self.consumed_tokens)
            || full_ids.len() <= self.consumed_tokens.len()
        {
            self.reset();
        }

        let cached_tokens = self.consumed_tokens.len();
        let delta_ids = &full_ids[cached_tokens..];
        let t_pre = Instant::now();
        if let Err(e) = self.model.prefill(&mut self.state, delta_ids, &mut self.logits) {
            self.reset();
            self.model.prefill(&mut self.state, &full_ids, &mut self.logits)?;
            let _ = e;
        }
        let prefill_s = t_pre.elapsed().as_secs_f64().max(1e-9);
        let actual_cached = if self.state.pos == full_ids.len() { cached_tokens } else { 0 };
        let new_prefill_tokens = full_ids.len().saturating_sub(actual_cached);
        self.consumed_tokens.clear();
        self.consumed_tokens.extend_from_slice(&full_ids);

        let eos_id = self.model.cfg.eos_token_id;
        let im_end_id = self.model.tokenizer.token_id("<|im_end|>").unwrap_or(eos_id);
        let eot_id = self.model.tokenizer.token_id("<|eot_id|>").unwrap_or(
            self.model.tokenizer.token_id("<|endoftext|>").unwrap_or(eos_id),
        );
        let think_end_id = self.model.tokenizer.token_id("</think>");
        let stop_ids = [eos_id, im_end_id, eot_id];

        let explicit_zero_pen = params.presence_penalty == Some(0.0);
        let use_cb4_greedy = self.cb4_greedy_ready
            && !params.enable_thinking
            && temp <= 1e-6
            && explicit_zero_pen
            && self.logits.iter().all(|x| x.is_finite());

        let presence_pen = if use_cb4_greedy || explicit_zero_pen {
            0.0
        } else {
            params.presence_penalty.unwrap_or(if is_qwen_think { 0.5 } else { 0.0 })
        };
        let freq_pen = 0.0f32;
        let top_k = if is_qwen_think { 20 } else { 40 };
        let top_p = if params.enable_thinking { 0.95 } else if is_qwen_think { 0.85 } else { 1.0 };

        let mut full_text = String::new();
        if is_qwen_think && params.enable_thinking {
            let prefix = "<think>\n";
            full_text.push_str(prefix);
            if !on_text(prefix) {
                let m = TurnMetrics {
                    prefill_ms: prefill_s * 1000.0,
                    prefill_tps: (new_prefill_tokens as f64) / prefill_s,
                    prompt_tokens: full_ids.len(),
                    cached_tokens: actual_cached,
                    new_prefill_tokens,
                    finish_reason: "stop".into(),
                    decode_path: "aborted".into(),
                    ..TurnMetrics::default()
                };
                return Ok((full_text, m));
            }
        }

        let (pos_before_dec, mut aborted, mut hit_stop, mut dec_start_tok) = (self.state.pos, false, false, 0usize);
        let t_dec: Instant;
        let mut t_last: Option<Instant> = None;
        let decode_path: String;

        let gen_ids = if use_cb4_greedy {
            decode_path = "cb4-fast-greedy".into();
            let first_id = sample_logits(&self.logits, 0.0, &mut self.rng);
            t_dec = Instant::now();
            let mut streamed = Vec::with_capacity(max_tokens);
            let res = self.model.generate_greedy_cb4(
                &mut self.state,
                first_id,
                max_tokens,
                &stop_ids,
                |tok, tokenizer| {
                    if cancel.load(Ordering::Relaxed) {
                        aborted = true;
                        return false;
                    }
                    streamed.push(tok);
                    t_last = Some(Instant::now());
                    flush_chunk(tokenizer, &streamed, &mut dec_start_tok, &mut full_text, &mut aborted, &mut on_text)
                        && !has_severe_loop(&streamed)
                },
            );
            match res {
                Ok(ids) => {
                    hit_stop = ids.len() < max_tokens && !aborted;
                    ids
                }
                Err(e) => {
                    self.reset();
                    return Err(e);
                }
            }
        } else if temp <= 1e-6 && explicit_zero_pen && self.model.has_turbo_fast_decode() {
            decode_path = "turbo-fast-greedy".into();
            let first_id = sample_logits(&self.logits, 0.0, &mut self.rng);
            t_dec = Instant::now();
            let mut streamed = Vec::with_capacity(max_tokens);
            let tok_ref = &self.model.tokenizer;
            let ids = self.model.generate_greedy_turbo(
                &mut self.state,
                delta_ids,
                first_id,
                max_tokens,
                true,
                presence_pen,
                &stop_ids,
                |tok| {
                    if cancel.load(Ordering::Relaxed) {
                        aborted = true;
                        return false;
                    }
                    streamed.push(tok);
                    flush_chunk(tok_ref, &streamed, &mut dec_start_tok, &mut full_text, &mut aborted, &mut on_text)
                        && !has_severe_loop(&streamed)
                },
            )?;
            hit_stop = ids.len() < max_tokens && !aborted;
            ids
        } else {
            decode_path = if temp <= 1e-6 { "forward-greedy".into() } else { "forward-sampled".into() };
            t_dec = Instant::now();
            let mut exempt = stop_ids.to_vec();
            exempt.extend(think_end_id);
            let mut tracker = TokenPenaltyTracker::new(self.model.cfg.n_vocab, &exempt);
            let mut in_think = is_qwen_think && params.enable_thinking;
            let think_budget = (max_tokens / 2).clamp(256, 1536);
            let close_nl = if in_think { self.model.tokenizer.encode("\n</think>\n\n") } else { Vec::new() };
            let close_after = if in_think { self.model.tokenizer.encode("</think>\n\n") } else { Vec::new() };
            let mut ids = Vec::with_capacity(max_tokens);

            while ids.len() < max_tokens {
                if cancel.load(Ordering::Relaxed) { aborted = true; break; }
                let ended_nl = full_text.ends_with('\n');
                let over = ids.len() >= think_budget && (ended_nl || ids.len() >= think_budget + 48);
                if in_think && !close_nl.is_empty() && (over || has_severe_loop(&ids)) && ids.len() + close_nl.len() < max_tokens {
                    let seq = if ended_nl { &close_after } else { &close_nl };
                    for (k, &ctok) in seq.iter().enumerate() {
                        ids.push(ctok);
                        let last = k + 1 == seq.len();
                        self.model.forward_step(&mut self.state, ctok, last.then_some(&mut self.logits))?;
                    }
                    if !flush_chunk(&self.model.tokenizer, &ids, &mut dec_start_tok, &mut full_text, &mut aborted, &mut on_text) {
                        break;
                    }
                    in_think = false;
                    tracker.clear_counts();
                }

                if !explicit_zero_pen {
                    tracker.apply(&mut self.logits, presence_pen, freq_pen);
                }
                let next_id = sample_logits_top_k_p(&self.logits, temp, top_k, top_p, &mut self.rng);
                if stop_ids.contains(&next_id) {
                    hit_stop = true;
                    break;
                }
                if in_think && Some(next_id) == think_end_id {
                    in_think = false;
                    tracker.clear_counts();
                } else {
                    tracker.record(next_id, &self.model.tokenizer);
                }
                ids.push(next_id);
                if !flush_chunk(&self.model.tokenizer, &ids, &mut dec_start_tok, &mut full_text, &mut aborted, &mut on_text) {
                    break;
                }
                if !in_think && !explicit_zero_pen && has_severe_loop(&ids) {
                    hit_stop = true;
                    break;
                }
                if ids.len() < max_tokens {
                    self.model.forward_step(&mut self.state, next_id, Some(&mut self.logits))?;
                }
            }
            ids
        };

        if dec_start_tok < gen_ids.len() {
            let tail = self.model.tokenizer.decode(&gen_ids[dec_start_tok..]);
            if !tail.is_empty() {
                full_text.push_str(&tail);
                let _ = on_text(&tail);
            }
        }

        let stepped = self.state.pos.saturating_sub(pos_before_dec).min(gen_ids.len());
        self.consumed_tokens.extend_from_slice(&gen_ids[..stepped]);
        if !full_text.is_empty() && !gen_ids.is_empty() {
            self.asst_cache
                .insert(full_text.clone(), (params.enable_thinking, gen_ids.clone()));
        }
        if self.consumed_tokens.len() != self.state.pos {
            self.reset();
        }

        let decode_s = t_last.map_or_else(|| t_dec.elapsed().as_secs_f64(), |t| t.duration_since(t_dec).as_secs_f64()).max(1e-9);
        let dec_steps = gen_ids.len().saturating_sub(1).max(1);
        let decode_tps = if gen_ids.len() > 1 {
            (dec_steps as f64) / decode_s
        } else {
            1.0 / prefill_s
        };
        let stream_gbs = (decode_tps * (self.summary.compressed_bytes as f64)) / 1e9;
        let finish_reason = if aborted || hit_stop || gen_ids.len() < max_tokens { "stop" } else { "length" };
        let metrics = TurnMetrics {
            prefill_ms: prefill_s * 1000.0,
            decode_ms: decode_s * 1000.0,
            prefill_tps: (new_prefill_tokens.max(1) as f64) / prefill_s,
            decode_tps,
            stream_gbs,
            prompt_tokens: full_ids.len(),
            cached_tokens: actual_cached,
            new_prefill_tokens,
            generated_tokens: gen_ids.len(),
            finish_reason: finish_reason.into(),
            decode_path,
        };
        Ok((full_text, metrics))
    }
}

pub fn cmd_chat_llm(args: &[String]) -> Result<()> {
    let Some(model_path) = args.first() else {
        return Err(Error::Format(
            "usage: vxg-tensor chat <model.vxt | model.gguf> [--temp T] [--max-tokens N] [--think] [--cpu]".into(),
        ));
    };
    let mut params = ChatParams { max_tokens: 1024, ..ChatParams::default() };
    let mut explicit_temp = false;
    let mut explicit_max = false;
    let mut use_gpu = true;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--temp" => {
                i += 1;
                explicit_temp = true;
                params.temperature = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                if params.temperature > 0.0 {
                    params.presence_penalty = None;
                }
            }
            "--max-tokens" | "-n" => {
                i += 1;
                explicit_max = true;
                params.max_tokens = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1024);
            }
            "--think" => params.enable_thinking = true,
            "--cpu" => use_gpu = false,
            other => return Err(Error::Format(format!("unexpected flag '{other}'"))),
        }
        i += 1;
    }
    if params.enable_thinking {
        if !explicit_temp {
            params.temperature = 0.6;
        }
        params.presence_penalty = None;
        if !explicit_max && params.max_tokens < 2048 {
            params.max_tokens = 2048;
        }
    }

    let mut session = ChatSession::load(model_path, use_gpu, threads)?;
    println!(
        "Loaded '{}' ({}, {:.2} MiB) on {} in {:.0} ms\nCommands: /clear (reset context), /think (toggle CoT), /exit",
        session.summary.model_name,
        session.summary.arch,
        session.summary.compressed_bytes as f64 / (1024.0 * 1024.0),
        session.summary.backend,
        session.summary.load_ms
    );

    let cancel = AtomicBool::new(false);
    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        print!("\nYou > ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| Error::Io(e.to_string()))? == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "/exit" | "/quit" => break,
            "/clear" => {
                session.reset();
                history.clear();
                println!("[context cleared]");
                continue;
            }
            "/think" => {
                params.enable_thinking = !params.enable_thinking;
                if params.enable_thinking {
                    if !explicit_temp { params.temperature = 0.6; }
                    params.presence_penalty = None;
                    params.max_tokens = params.max_tokens.max(2048);
                }
                println!("[thinking = {}]", params.enable_thinking);
                continue;
            }
            _ => {}
        }
        history.push(ChatMessage { role: "user".into(), content: trimmed.to_owned() });
        print!("VXT > ");
        let _ = io::stdout().flush();
        let (reply, m) = session.generate_chat(&history, &params, &cancel, |chunk| {
            print!("{chunk}");
            let _ = io::stdout().flush();
            true
        })?;
        println!(
            "\n  [prefill {:.1} ms ({} cached + {} new) | {:.1} tok/s ({:.1} GB/s) | {} tok]",
            m.prefill_ms,
            m.cached_tokens,
            m.new_prefill_tokens,
            m.decode_tps,
            m.stream_gbs,
            m.generated_tokens
        );
        history.push(ChatMessage { role: "assistant".into(), content: reply });
    }
    Ok(())
}
