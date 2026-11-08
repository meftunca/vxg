//! CLI subcommand implementations for full-model `.vxt` packing (`pack-model`), compressed-domain
//! Apple Metal / CPU text generation (`run`), and head-to-head logit fidelity evaluation (`eval`).

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use crate::llm::{compare_logits, sample_logits, LlmModel};
use crate::llm_ops::{has_severe_loop, sample_logits_top_k_p, TokenPenaltyTracker};
use crate::llm_pack::{pack_gguf_model_to_vxt_with_entropy_scale, LlmPackMode};
use crate::mmap::TensorBytes;
use crate::{Error, Result};

pub fn cmd_pack_model(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(Error::Format(
            "usage: vxg-tensor pack-model <in.gguf | hf_model_dir | model.safetensors> <out.vxt> [--mode lossless|cb4|cb3|wht2|turbo|q4_0] [--threads T] [--turbo-entropy-scale S]".into(),
        ));
    }
    let in_path = &args[0];
    let out_path = &args[1];
    let mut mode = LlmPackMode::Lossless;
    let mut entropy_scale = None;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                let s = args
                    .get(i)
                    .ok_or_else(|| Error::Format("missing value for --mode".into()))?;
                mode = LlmPackMode::parse(s)?;
            }
            "--turbo-entropy-scale" => {
                i += 1;
                entropy_scale = Some(args.get(i).and_then(|s| s.parse::<f32>().ok())
                    .ok_or_else(|| Error::Format("invalid --turbo-entropy-scale".into()))?);
            }
            "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Format("invalid --threads".into()))?;
            }
            other => return Err(Error::Format(format!("unexpected flag '{other}'"))),
        }
        i += 1;
    }

    if entropy_scale.is_some() && mode != LlmPackMode::Turbo {
        return Err(Error::Format("--turbo-entropy-scale requires --mode turbo".into()));
    }
    let scale = entropy_scale.unwrap_or(1.0);
    if !scale.is_finite() || scale < 0.0 {
        return Err(Error::Format("--turbo-entropy-scale must be finite and nonnegative".into()));
    }
    let t0 = Instant::now();
    let in_p = Path::new(in_path);
    let is_hf = !in_path.to_ascii_lowercase().ends_with(".gguf")
        && (in_p.is_dir() || crate::hf_pack::resolve_hf_model_dir(in_p).is_some());
    let (cfg, in_len, archive) = if is_hf {
        if entropy_scale.is_some() {
            return Err(Error::Format(
                "--turbo-entropy-scale is only supported for GGUF inputs".into(),
            ));
        }
        let (cfg, raw_bytes, archive) = crate::hf_pack::pack_hf_model_to_vxt(in_p, mode, threads)?;
        (cfg, raw_bytes as usize, archive)
    } else {
        let mapped = TensorBytes::open(in_path)?;
        let in_bytes = mapped.as_slice();
        let (cfg, archive) =
            pack_gguf_model_to_vxt_with_entropy_scale(in_bytes, mode, threads, scale)?;
        (cfg, in_bytes.len(), archive)
    };
    fs::write(Path::new(out_path), &archive).map_err(|e| Error::Io(e.to_string()))?;

    let in_mib = in_len as f64 / (1024.0 * 1024.0);
    let out_mib = archive.len() as f64 / (1024.0 * 1024.0);
    let ratio = (in_len as f64) / (archive.len().max(1) as f64);
    println!(
        "Packed LLM '{}' (arch={}, {} layers, vocab={}) [{}] -> {}\n  {:.2} MiB -> {:.2} MiB ({:.2}x compression) in {:.2}s",
        cfg.model_name,
        cfg.arch,
        cfg.n_layer,
        cfg.n_vocab,
        mode.name(),
        out_path,
        in_mib,
        out_mib,
        ratio,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn cmd_run_llm(args: &[String]) -> Result<()> {
    if args.is_empty() {
        return Err(Error::Format(
            "usage: vxg-tensor run <model.vxt | model.gguf | model.safetensors> --prompt \"...\" [--max-tokens N] [--temp T] [--cpu] [--raw] [--think] [--reference-decode] [--cb4-greedy --temp 0 --presence-penalty 0]".into(),
        ));
    }
    let model_path = &args[0];
    let mut prompt =
        String::from("Explain why memory bandwidth limits LLM inference in 3 short bullet points.");
    let mut max_tokens = 64usize;
    let mut temp = 0.0f32;
    let mut presence_penalty: Option<f32> = None;
    let mut use_gpu = true;
    let mut raw_prompt = false;
    let mut enable_thinking = false;
    let mut exact_head = false;
    let mut reference_decode = false;
    let mut cb4_greedy = false;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => {
                i += 1;
                prompt = args
                    .get(i)
                    .cloned()
                    .ok_or_else(|| Error::Format("missing --prompt string".into()))?;
            }
            "--max-tokens" | "-n" => {
                i += 1;
                max_tokens = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Format("invalid --max-tokens".into()))?;
            }
            "--temp" => {
                i += 1;
                temp = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Format("invalid --temp".into()))?;
            }
            "--presence-penalty" => {
                i += 1;
                presence_penalty = Some(
                    args.get(i)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| Error::Format("invalid --presence-penalty".into()))?,
                );
            }
            "--threads" => {
                i += 1;
                threads = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Format("invalid --threads".into()))?;
            }
            "--cpu" => use_gpu = false,
            "--raw" => raw_prompt = true,
            "--think" => enable_thinking = true,
            "--exact-head" => exact_head = true,
            "--reference-decode" => reference_decode = true,
            "--cb4-greedy" => cb4_greedy = true,
            other => return Err(Error::Format(format!("unexpected flag '{other}'"))),
        }
        i += 1;
    }

    if cb4_greedy && (!use_gpu || temp != 0.0 || presence_penalty != Some(0.0) || reference_decode || exact_head) {
        return Err(Error::Unsupported("--cb4-greedy requires GPU, --temp 0, explicit --presence-penalty 0, and no alternate decode flags".into()));
    }
    let t_load = Instant::now();
    let mut model = LlmModel::load_file(model_path, use_gpu, threads)?;
    if cb4_greedy { model.enable_cb4_greedy()?; }
    let load_s = t_load.elapsed().as_secs_f64();

    let prompt_ids = if raw_prompt {
        model.tokenizer.encode(&prompt)
    } else {
        model
            .tokenizer
            .encode_chat(&prompt, &model.cfg.arch, enable_thinking)
    };
    if prompt_ids.is_empty() {
        return Err(Error::Format("prompt tokenized to 0 tokens".into()));
    }

    println!(
        "Loaded '{}' ({}, {:.2} MiB) on {} in {:.2}s | prompt={} tokens",
        model.cfg.model_name,
        model.cfg.arch,
        model.total_compressed_bytes as f64 / (1024.0 * 1024.0),
        model.backend_label(),
        load_s,
        prompt_ids.len()
    );
    println!("---");

    let mut state = model.new_state();
    let mut logits = vec![0.0f32; model.cfg.n_vocab];

    let t_prefill = Instant::now();
    model.prefill(&mut state, &prompt_ids, &mut logits)?;
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    if cb4_greedy && !logits.iter().all(|x| x.is_finite()) {
        return Err(Error::Format("CB4 greedy requires finite prefill logits".into()));
    }

    let eos_id = model.cfg.eos_token_id;
    let im_end_id = model.tokenizer.token_id("<|im_end|>").unwrap_or(eos_id);
    let eot_id = model.tokenizer.token_id("<|endoftext|>").unwrap_or(eos_id);
    let think_end_id = model.tokenizer.token_id("</think>");
    let is_qwen3 = model.cfg.arch.starts_with("qwen3");

    let explicit_zero_pen = presence_penalty == Some(0.0) || cb4_greedy;
    let presence_pen = if explicit_zero_pen {
        0.0
    } else {
        presence_penalty.unwrap_or(if is_qwen3 { 0.5 } else { 0.0 })
    };
    let freq_pen = 0.0f32;
    let top_k = if is_qwen3 { 20 } else { 40 };
    let top_p = if enable_thinking { 0.95 } else if is_qwen3 { 0.85 } else { 1.0 };

    let mut exempt = vec![eos_id, im_end_id, eot_id];
    if let Some(te) = think_end_id {
        exempt.push(te);
    }
    let mut tracker = TokenPenaltyTracker::new(model.cfg.n_vocab, &exempt);

    let mut rng = 0x2026_11_03_5eed_u64;
    let mut dec_start_tok = 0usize;
    let t_decode: Instant;
    println!("Decode path: {} | presence_penalty={presence_pen} max_tokens={max_tokens}",
        if cb4_greedy { "cb4-greedy-full-head" } else if temp <= 1e-6 && model.has_turbo_fast_decode() && !reference_decode {
            if exact_head { "turbo-fast-exact-head" } else { "turbo-fast-sketch-head" }
        } else { "forward-step" });
    if enable_thinking && !raw_prompt && is_qwen3 {
        print!("<think>\n");
        let _ = io::stdout().flush();
    }

    let gen_ids = if cb4_greedy {
        let first_id = sample_logits(&logits, 0.0, &mut rng);
        t_decode = Instant::now();
        let mut streamed_ids = Vec::with_capacity(max_tokens);
        model.generate_greedy_cb4(&mut state, first_id, max_tokens, &[eos_id, im_end_id, eot_id], |tok, tokenizer| {
            streamed_ids.push(tok);
            let chunk = tokenizer.decode(&streamed_ids[dec_start_tok..]);
            if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                print!("{chunk}");
                let _ = io::stdout().flush();
                dec_start_tok = streamed_ids.len();
            }
            true
        })?
    } else if temp <= 1e-6 && model.has_turbo_fast_decode() && !reference_decode {
        let first_id = sample_logits(&logits, 0.0, &mut rng);
        t_decode = Instant::now();
        let mut streamed_ids = Vec::with_capacity(max_tokens);
        model.generate_greedy_turbo(
            &mut state,
            &prompt_ids,
            first_id,
            max_tokens,
            !exact_head,
            presence_pen,
            &[eos_id, im_end_id, eot_id],
            |tok| {
                streamed_ids.push(tok);
                let chunk = model.tokenizer.decode(&streamed_ids[dec_start_tok..]);
                if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                    print!("{chunk}");
                    let _ = io::stdout().flush();
                    dec_start_tok = streamed_ids.len();
                }
                true
            },
        )?
    } else {
        t_decode = Instant::now();
        let mut ids = Vec::with_capacity(max_tokens);
        let mut in_think = enable_thinking && !raw_prompt && is_qwen3;
        let think_budget = (max_tokens / 2).clamp(256, 1536);
        let close_nl = model.tokenizer.encode("\n</think>\n\n");
        let close_after_nl = model.tokenizer.encode("</think>\n\n");
        let mut last_ended_nl = false;

        while ids.len() < max_tokens {
            let force_close = in_think
                && ((ids.len() >= think_budget && (last_ended_nl || ids.len() >= think_budget + 48))
                    || has_severe_loop(&ids));
            if force_close && ids.len() + close_nl.len() < max_tokens {
                let seq = if last_ended_nl { &close_after_nl } else { &close_nl };
                for (k, &ctok) in seq.iter().enumerate() {
                    ids.push(ctok);
                    let last = k + 1 == seq.len();
                    model.forward_step(&mut state, ctok, if last { Some(&mut logits) } else { None })?;
                }
                let chunk = model.tokenizer.decode(&ids[dec_start_tok..]);
                if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                    print!("{chunk}");
                    let _ = io::stdout().flush();
                    dec_start_tok = ids.len();
                }
                in_think = false;
                tracker.clear_counts();
            }

            if !explicit_zero_pen {
                tracker.apply(&mut logits, presence_pen, freq_pen);
            }
            let next_id = sample_logits_top_k_p(&logits, temp, top_k, top_p, &mut rng);
            if next_id == eos_id || next_id == im_end_id || next_id == eot_id {
                break;
            }
            if in_think && Some(next_id) == think_end_id {
                in_think = false;
                tracker.clear_counts();
            } else {
                tracker.record(next_id, &model.tokenizer);
            }
            ids.push(next_id);
            let chunk = model.tokenizer.decode(&ids[dec_start_tok..]);
            if !chunk.is_empty() && !chunk.ends_with('\u{FFFD}') {
                last_ended_nl = chunk.ends_with('\n');
                print!("{chunk}");
                let _ = io::stdout().flush();
                dec_start_tok = ids.len();
            }
            if !in_think && !explicit_zero_pen && has_severe_loop(&ids) {
                break;
            }
            if ids.len() < max_tokens {
                model.forward_step(&mut state, next_id, Some(&mut logits))?;
            }
        }
        ids
    };
    if dec_start_tok < gen_ids.len() {
        print!("{}", model.tokenizer.decode(&gen_ids[dec_start_tok..]));
    }
    println!("\n---");

    let decode_steps = gen_ids.len().saturating_sub(1).max(1);
    let decode_s = t_decode.elapsed().as_secs_f64().max(1e-9);
    let prefill_tok_s = (prompt_ids.len() as f64) / prefill_s.max(1e-9);
    let decode_tok_s = if gen_ids.len() > 1 {
        (decode_steps as f64) / decode_s
    } else {
        1.0 / prefill_s.max(1e-9)
    };
    let eq_bw_gbs = (model.total_compressed_bytes as f64 * decode_tok_s) / 1e9;

    println!(
        "Stats: prefill {:.1} tok/s ({} tok in {:.2}s) | decode {:.2} tok/s ({} tok, {:.1} GB/s compressed stream)",
        prefill_tok_s,
        prompt_ids.len(),
        prefill_s,
        decode_tok_s,
        gen_ids.len(),
        eq_bw_gbs
    );
    println!("Timing: load_s={load_s:.6} prefill_s={prefill_s:.6} decode_s={decode_s:.6} generated={} state_pos={}", gen_ids.len(), state.pos);
    Ok(())
}

pub fn cmd_eval_llm(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(Error::Format(
            "usage: vxg-tensor eval <ref.gguf | ref.vxt> <test.vxt> [--prompt \"...\"] [--steps N] [--cpu]".into(),
        ));
    }
    let ref_path = &args[0];
    let test_path = &args[1];
    let mut prompt = String::from("Türkiye'nin başkenti Ankara'dır. Yapay zeka modellerinde bellek bant genişliği neden kritiktir?");
    let mut steps = 8usize;
    let mut use_gpu = true;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" | "-p" => {
                i += 1;
                prompt = args.get(i).cloned().unwrap_or(prompt);
            }
            "--steps" | "-n" => {
                i += 1;
                steps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(steps);
            }
            "--cpu" => use_gpu = false,
            other => return Err(Error::Format(format!("unexpected flag '{other}'"))),
        }
        i += 1;
    }

    // Run teacher-forced rollout on reference model first, then free its RAM before loading test!
    let (prompt_len, vocab_sz, ref_bytes, ref_trace) = {
        let mut ref_m = LlmModel::load_file(ref_path, use_gpu, threads)?;
        let ids = ref_m.tokenizer.encode_chat(&prompt, &ref_m.cfg.arch, false);
        let mut st = ref_m.new_state();
        let mut logits = vec![0.0f32; ref_m.cfg.n_vocab];
        for (idx, &tid) in ids.iter().enumerate() {
            let last = idx + 1 == ids.len();
            ref_m.forward_step(&mut st, tid, last.then_some(logits.as_mut_slice()))?;
        }
        let mut trace: Vec<(u32, Vec<f32>)> = Vec::with_capacity(steps);
        let mut rng = 0u64;
        for s in 0..steps {
            let next = sample_logits(&logits, 0.0, &mut rng);
            trace.push((next, logits.clone()));
            if s + 1 < steps {
                ref_m.forward_step(&mut st, next, Some(&mut logits))?;
            }
        }
        (
            ids.len(),
            ref_m.cfg.n_vocab,
            ref_m.total_compressed_bytes,
            (ids, trace),
        )
    };

    let mut test_m = LlmModel::load_file(test_path, use_gpu, threads)?;
    let mut st = test_m.new_state();
    let mut logits = vec![0.0f32; vocab_sz];
    for (idx, &tid) in ref_trace.0.iter().enumerate() {
        let last = idx + 1 == prompt_len;
        test_m.forward_step(&mut st, tid, last.then_some(logits.as_mut_slice()))?;
    }

    let mut sum_kl = 0.0f64;
    let mut sum_cos = 0.0f64;
    let mut sum_rmse = 0.0f64;
    let mut top1_hits = 0usize;

    let t_dec = Instant::now();
    for (s, (teacher_tok, ref_l)) in ref_trace.1.iter().enumerate() {
        let m = compare_logits(ref_l, &logits);
        sum_kl += m.kl_div;
        sum_cos += m.cosine_sim;
        sum_rmse += m.rmse;
        if m.top1_match {
            top1_hits += 1;
        }
        if s + 1 < steps {
            test_m.forward_step(&mut st, *teacher_tok, Some(&mut logits))?;
        }
    }
    let dec_tok_s =
        ((steps.saturating_sub(1).max(1)) as f64) / t_dec.elapsed().as_secs_f64().max(1e-9);
    let n = steps as f64;
    println!(
        "=== VXT END-TO-END LOGIT EVAL ({} steps, prompt={} tok) ===\n  Ref : {} ({:.2} MiB)\n  Test: {} ({:.2} MiB, {:.2}x smaller) on {}\n  Mean KL(P_ref || P_vxt): {:.6} nats | Mean Logit CosSim: {:.6} | RMSE: {:.5} | Top-1 Match: {}/{} ({:.1}%) | Decode: {:.2} tok/s",
        steps,
        prompt_len,
        ref_path,
        ref_bytes as f64 / (1024.0 * 1024.0),
        test_path,
        test_m.total_compressed_bytes as f64 / (1024.0 * 1024.0),
        (ref_bytes as f64) / (test_m.total_compressed_bytes.max(1) as f64),
        test_m.backend_label(),
        sum_kl / n,
        sum_cos / n,
        sum_rmse / n,
        top1_hits,
        steps,
        (top1_hits as f64 * 100.0) / n,
        dec_tok_s
    );
    Ok(())
}
