//! Sequential real-model teacher forcing with forward-only timing, independent source weights.
use std::time::Instant;
use vxg_tensor::llm::{compare_logits, sample_logits, LlmModel};
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: qualify-model SOURCE.gguf|SOURCE.safetensors TEST.vxt CONTEXT_TOKENS STEPS (Metal required)"
                .into(),
        );
    }
    let context: usize = args[3].parse()?;
    let steps: usize = args[4].parse()?;
    if !(1..=2048).contains(&context) || !(2..=256).contains(&steps) {
        return Err("context must be 1..2048 and steps 2..256".into());
    }
    println!("GATE max_rmse<=1e-4 min_cos>=0.99999 top1_all=true finite_logits=true");
    let text="Lossless compression preserves every original weight bit. Memory bandwidth affects inference speed. Türkiye'nin başkenti Ankara'dır. The quick brown fox jumps over the lazy dog. ".repeat(100);
    let (ids, trace, vocab, config, token_ids) = {
        let t = Instant::now();
        let mut m = LlmModel::load_file(&args[1], true, 2)?;
        let load = t.elapsed().as_secs_f64();
        let backend = m.backend_label();
        if !backend.contains("1-CmdBuf Fused") {
            return Err(format!("required fused Metal backend, got {backend}").into());
        }
        let token_ids = m.tokenizer.encode_chat(&text, &m.cfg.arch, false);
        let mut ids = token_ids.clone();
        ids.truncate(context);
        if ids.len() != context {
            return Err("insufficient tokenized context".into());
        }
        let mut state = m.new_state();
        let mut logits = vec![0.; m.cfg.n_vocab];
        let t = Instant::now();
        for (i, &tok) in ids.iter().enumerate() {
            m.forward_step(
                &mut state,
                tok,
                (i + 1 == ids.len()).then_some(logits.as_mut_slice()),
            )?;
        }
        let prefill = t.elapsed().as_secs_f64();
        let mut trace = Vec::with_capacity(steps);
        let mut forward_s = 0.;
        let mut rng = 0;
        for i in 0..steps {
            if !logits.iter().all(|v| v.is_finite()) {
                return Err("nonfinite source logits".into());
            }
            let next = sample_logits(&logits, 0., &mut rng);
            trace.push((next, logits.clone()));
            if i + 1 < steps {
                let t = Instant::now();
                m.forward_step(&mut state, next, Some(&mut logits))?;
                forward_s += t.elapsed().as_secs_f64();
            }
        }
        println!("REF backend={backend} load_s={load:.6} prefill_tokens={} prefill_s={prefill:.6} prefill_tps={:.6} decode_forwards={} decode_forward_s={forward_s:.6} decode_tps={:.6}",ids.len(),ids.len() as f64/prefill,steps-1,(steps-1) as f64/forward_s);
        (ids, trace, m.cfg.n_vocab, m.cfg.to_kv_bytes(), token_ids)
    };
    let t = Instant::now();
    let mut m = LlmModel::load_file(&args[2], true, 2)?;
    let load = t.elapsed().as_secs_f64();
    let backend = m.backend_label();
    if !backend.contains("1-CmdBuf Fused") {
        return Err(format!("required fused Metal backend, got {backend}").into());
    }
    if m.cfg.n_vocab != vocab {
        return Err("vocabulary mismatch".into());
    }
    if m.cfg.to_kv_bytes() != config {
        return Err("model configuration mismatch".into());
    }
    if m.tokenizer.encode_chat(&text, &m.cfg.arch, false) != token_ids {
        return Err("tokenization mismatch on qualification text".into());
    }
    println!("CONFIG exact_serialized_match=true TOKENIZER qualification_text_ids_match=true (not exhaustive tokenizer equivalence)");
    let mut state = m.new_state();
    let mut logits = vec![0.; vocab];
    let t = Instant::now();
    for (i, &tok) in ids.iter().enumerate() {
        m.forward_step(
            &mut state,
            tok,
            (i + 1 == ids.len()).then_some(logits.as_mut_slice()),
        )?;
    }
    let prefill = t.elapsed().as_secs_f64();
    let (mut max_abs, mut max_rmse, mut max_kl, mut min_cos, mut hits, mut bit_mismatches) =
        (0f32, 0f64, 0f64, 1f64, 0usize, 0usize);
    let mut forward_s = 0.;
    for (i, (teacher, reference)) in trace.iter().enumerate() {
        if !logits.iter().all(|v| v.is_finite()) {
            return Err("nonfinite test logits".into());
        }
        let c = compare_logits(reference, &logits);
        max_rmse = max_rmse.max(c.rmse);
        max_kl = max_kl.max(c.kl_div);
        min_cos = min_cos.min(c.cosine_sim);
        hits += usize::from(c.top1_match);
        for (a, b) in reference.iter().zip(&logits) {
            max_abs = max_abs.max((a - b).abs());
            bit_mismatches += usize::from(a.to_bits() != b.to_bits());
        }
        println!(
            "STEP {i} teacher={teacher} top1={} rmse={:.9e} kl={:.9e} cos={:.12}",
            c.top1_match, c.rmse, c.kl_div, c.cosine_sim
        );
        if i + 1 < steps {
            let t = Instant::now();
            m.forward_step(&mut state, *teacher, Some(&mut logits))?;
            forward_s += t.elapsed().as_secs_f64();
        }
    }
    println!("TEST backend={backend} load_s={load:.6} prefill_tokens={} prefill_s={prefill:.6} prefill_tps={:.6} decode_forwards={} decode_forward_s={forward_s:.6} decode_tps={:.6}",ids.len(),ids.len() as f64/prefill,steps-1,(steps-1) as f64/forward_s);
    println!("RESULT steps={steps} top1={hits}/{steps} max_abs={max_abs:.9e} max_rmse={max_rmse:.9e} max_kl={max_kl:.9e} min_cos={min_cos:.12} logit_bit_mismatches={bit_mismatches}");
    if !parity_passes(hits, steps, max_rmse, min_cos) {
        return Err("numerical parity gate failed".into());
    }
    Ok(())
}

fn parity_passes(hits: usize, steps: usize, rmse: f64, cosine: f64) -> bool {
    steps > 0
        && hits == steps
        && rmse.is_finite()
        && cosine.is_finite()
        && rmse <= 1e-4
        && cosine >= 0.99999
}
#[cfg(test)]
mod tests {
    #[test]
    fn parity_gate_rejects_nonfinite_and_bad_results() {
        for (hits, steps, rmse, cos) in [
            (1, 2, 0., 1.),
            (0, 0, 0., 1.),
            (2, 2, 1.01e-4, 1.),
            (2, 2, 0., 0.99),
            (2, 2, f64::NAN, 1.),
            (2, 2, 0., f64::NAN),
            (2, 2, f64::INFINITY, 1.),
        ] {
            assert!(!super::parity_passes(hits, steps, rmse, cos));
        }
        assert!(super::parity_passes(32, 32, 1e-4, 0.99999));
    }
}
