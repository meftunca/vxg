//! CLI subcommand implementations for full-model `.vxt` packing (`pack-model`), compressed-domain
//! Apple Metal / CPU text generation (`run`), and head-to-head logit fidelity evaluation (`eval`).

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use crate::llm::{compare_logits, sample_logits, LlmModel};
use crate::llm_pack::{pack_gguf_model_to_vxt, LlmPackMode};
use crate::mmap::TensorBytes;
use crate::{Error, Result};

pub fn cmd_pack_model(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        return Err(Error::Format(
            "usage: vxg-tensor pack-model <in.gguf> <out.vxt> [--mode lossless|cb4|cb3|wht2|q4_0] [--threads T]".into(),
        ));
    }
    let in_path = &args[0];
    let out_path = &args[1];
    let mut mode = LlmPackMode::Cb4;
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

    let t0 = Instant::now();
    let mapped = TensorBytes::open(in_path)?;
    let in_bytes = mapped.as_slice();
    let (cfg, archive) = pack_gguf_model_to_vxt(in_bytes, mode, threads)?;
    fs::write(Path::new(out_path), &archive).map_err(|e| Error::Io(e.to_string()))?;

    let in_mib = in_bytes.len() as f64 / (1024.0 * 1024.0);
    let out_mib = archive.len() as f64 / (1024.0 * 1024.0);
    let ratio = (in_bytes.len() as f64) / (archive.len().max(1) as f64);
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
            "usage: vxg-tensor run <model.vxt | model.gguf> --prompt \"...\" [--max-tokens N] [--temp T] [--cpu] [--raw] [--think]".into(),
        ));
    }
    let model_path = &args[0];
    let mut prompt =
        String::from("Explain why memory bandwidth limits LLM inference in 3 short bullet points.");
    let mut max_tokens = 64usize;
    let mut temp = 0.0f32;
    let mut use_gpu = true;
    let mut raw_prompt = false;
    let mut enable_thinking = false;
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
            other => return Err(Error::Format(format!("unexpected flag '{other}'"))),
        }
        i += 1;
    }

    let t_load = Instant::now();
    let mut model = LlmModel::load_file(model_path, use_gpu, threads)?;
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
    for (idx, &tid) in prompt_ids.iter().enumerate() {
        let last = idx + 1 == prompt_ids.len();
        model.forward_step(&mut state, tid, last.then_some(logits.as_mut_slice()))?;
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();

    let eos_id = model.cfg.eos_token_id;
    let im_end_id = model.tokenizer.token_id("<|im_end|>").unwrap_or(eos_id);
    let eot_id = model.tokenizer.token_id("<|endoftext|>").unwrap_or(eos_id);

    let mut rng = 0x2026_11_03_5eed_u64;
    let mut gen_ids = Vec::with_capacity(max_tokens);
    let mut printed_bytes = 0usize;
    let t_decode = Instant::now();

    for step in 0..max_tokens {
        let next_id = sample_logits(&logits, temp, &mut rng);
        if next_id == eos_id || next_id == im_end_id || next_id == eot_id {
            break;
        }
        gen_ids.push(next_id);
        let decoded = model.tokenizer.decode(&gen_ids);
        if decoded.len() > printed_bytes && !decoded.ends_with('\u{FFFD}') {
            print!("{}", &decoded[printed_bytes..]);
            let _ = io::stdout().flush();
            printed_bytes = decoded.len();
        }
        if step + 1 < max_tokens {
            model.forward_step(&mut state, next_id, Some(&mut logits))?;
        }
    }
    let full = model.tokenizer.decode(&gen_ids);
    if full.len() > printed_bytes {
        print!("{}", &full[printed_bytes..]);
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
