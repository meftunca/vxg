use std::collections::HashMap;

use vxg_tensor::codebook::CodebookMatrix;
use vxg_tensor::format::{write_vxt_model, PackedTensor, VxtEntry};
use vxg_tensor::gguf_meta::{GgufVal, ModelConfig};
use vxg_tensor::llm::{compare_logits, sample_logits, LlmModel};
use vxg_tensor::metal::MetalEngine;
use vxg_tensor::tokenizer::BpeTokenizer;
use vxg_tensor::wht2::Wht2Matrix;
use vxg_tensor::wht_turbo::TurboWhtMatrix;

fn pseudo_vec(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut st = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            st ^= st >> 12;
            st ^= st << 25;
            st ^= st >> 27;
            let u = ((st.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / ((1u32 << 24) as f32);
            let v = (u * 2.0 - 1.0) * scale;
            if i % 97 == 7 {
                v * 4.5
            } else {
                v
            }
        })
        .collect()
}

fn build_synthetic_qwen35_vxt(mode: u8) -> Vec<u8> {
    let cfg = ModelConfig {
        arch: "qwen35".into(),
        model_name: "synth-qwen35".into(),
        n_layer: 2,
        n_embd: 256,
        n_ff: 512,
        n_vocab: 256,
        n_head: 4,
        n_head_kv: 2,
        head_dim: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 1_000_000.0,
        rope_dim: 32,
        rope_neox: true,
        ffn_gelu: false,
        embed_scale: 1.0,
        full_attn_interval: 2, // layer 0 = LinearDeltaNet (dk=128!), layer 1 = FullGqa (gated_q)!
        ssm_conv_kernel: 4,
        ssm_state_size: 128,
        ssm_group_count: 2,
        ssm_dt_rank: 2,
        ssm_inner_size: 256,
        eos_token_id: 2,
        bos_token_id: Some(1),
    };

    let mut kv = HashMap::new();
    let tokens: Vec<String> = (0..256).map(|i| format!("tok{i}")).collect();
    kv.insert("tokenizer.ggml.tokens".into(), GgufVal::ArrStr(tokens));
    kv.insert("tokenizer.ggml.merges".into(), GgufVal::ArrStr(Vec::new()));
    let mut meta = vec![("model.config".to_owned(), cfg.to_kv_bytes())];
    meta.extend(BpeTokenizer::to_vxt_meta_entries(&kv).unwrap());

    let mk_mat = |name: &str, rows: usize, cols: usize, seed: u64| -> VxtEntry {
        let data = pseudo_vec(seed, rows * cols, 0.08);
        let tensor = if mode >= 4 {
            let dtype = if mode % 2 == 0 { vxg_tensor::DType::BF16 } else { vxg_tensor::DType::F16 };
            let words: Vec<u16> = data.iter().map(|&v| if dtype == vxg_tensor::DType::BF16 { (v.to_bits() >> 16) as u16 } else { vxg_tensor::fp16::f32_to_f16(v) }).collect();
            if mode < 6 {
                PackedTensor::Lossless(vxg_tensor::lossless::LosslessMatrix::encode(dtype, rows, cols, &words).unwrap())
            } else {
                let data = words.iter().map(|&w| if dtype == vxg_tensor::DType::BF16 { f32::from_bits((w as u32) << 16) } else { vxg_tensor::fp16::f16_to_f32(w) }).collect();
                PackedTensor::RawF32 { rows, cols, data }
            }
        } else if rows < 64 {
            PackedTensor::RawF32 { rows, cols, data }
        } else if mode == 3 {
            PackedTensor::TurboWht(
                TurboWhtMatrix::encode_with_lambda(4, rows, cols, &data, |r| {
                    if name.contains("token_embd") && r >= 128 {
                        0.18
                    } else {
                        0.02
                    }
                })
                .unwrap(),
            )
        } else if mode == 2 {
            let bits = if name.contains("token_embd") {
                2
            } else if name.contains("ffn_") {
                5
            } else {
                3
            };
            PackedTensor::TurboWht(TurboWhtMatrix::encode(bits, rows, cols, &data).unwrap())
        } else if mode == 1 {
            PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &data).unwrap())
        } else {
            PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, &data).unwrap())
        };
        VxtEntry {
            name: name.into(),
            tensor,
        }
    };
    let mk_vec = |name: &str, len: usize, val_base: f32, seed: u64| -> VxtEntry {
        let mut data = pseudo_vec(seed, len, 0.15);
        for v in &mut data {
            *v += val_base;
        }
        VxtEntry {
            name: name.into(),
            tensor: PackedTensor::RawF32 {
                rows: 1,
                cols: len,
                data,
            },
        }
    };

    let conv_ch = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
    let entries = vec![
        mk_mat("token_embd.weight", cfg.n_vocab, cfg.n_embd, 10),
        mk_vec("output_norm.weight", cfg.n_embd, 1.0, 11),
        mk_vec("blk.0.attn_norm.weight", cfg.n_embd, 1.0, 100),
        mk_vec("blk.0.post_attention_norm.weight", cfg.n_embd, 1.0, 101),
        mk_mat("blk.0.attn_qkv.weight", conv_ch, cfg.n_embd, 102),
        mk_mat(
            "blk.0.attn_gate.weight",
            cfg.ssm_inner_size,
            cfg.n_embd,
            103,
        ),
        mk_mat("blk.0.ssm_alpha.weight", cfg.ssm_dt_rank, cfg.n_embd, 104),
        mk_mat("blk.0.ssm_beta.weight", cfg.ssm_dt_rank, cfg.n_embd, 105),
        mk_mat("blk.0.ssm_out.weight", cfg.n_embd, cfg.ssm_inner_size, 106),
        mk_vec(
            "blk.0.ssm_conv1d.weight",
            conv_ch * cfg.ssm_conv_kernel,
            0.0,
            107,
        ),
        mk_vec("blk.0.ssm_a", cfg.ssm_dt_rank, -0.5, 108),
        mk_vec("blk.0.ssm_dt.bias", cfg.ssm_dt_rank, 0.1, 109),
        mk_vec(
            "blk.0.ssm_norm.weight",
            cfg.ssm_inner_size / cfg.ssm_dt_rank,
            1.0,
            110,
        ),
        mk_mat("blk.0.ffn_gate.weight", cfg.n_ff, cfg.n_embd, 111),
        mk_mat("blk.0.ffn_up.weight", cfg.n_ff, cfg.n_embd, 112),
        mk_mat("blk.0.ffn_down.weight", cfg.n_embd, cfg.n_ff, 113),
        mk_vec("blk.1.attn_norm.weight", cfg.n_embd, 1.0, 200),
        mk_vec("blk.1.post_attention_norm.weight", cfg.n_embd, 1.0, 201),
        mk_mat(
            "blk.1.attn_q.weight",
            cfg.n_head * cfg.head_dim * 2,
            cfg.n_embd,
            202,
        ),
        mk_mat(
            "blk.1.attn_k.weight",
            cfg.n_head_kv * cfg.head_dim,
            cfg.n_embd,
            203,
        ),
        mk_mat(
            "blk.1.attn_v.weight",
            cfg.n_head_kv * cfg.head_dim,
            cfg.n_embd,
            204,
        ),
        mk_mat(
            "blk.1.attn_output.weight",
            cfg.n_embd,
            cfg.n_head * cfg.head_dim,
            205,
        ),
        mk_vec("blk.1.attn_q_norm.weight", cfg.head_dim, 1.0, 206),
        mk_vec("blk.1.attn_k_norm.weight", cfg.head_dim, 1.0, 207),
        mk_mat("blk.1.ffn_gate.weight", cfg.n_ff, cfg.n_embd, 208),
        mk_mat("blk.1.ffn_up.weight", cfg.n_ff, cfg.n_embd, 209),
        mk_mat("blk.1.ffn_down.weight", cfg.n_embd, cfg.n_ff, 210),
    ];

    write_vxt_model(&meta, &entries).unwrap()
}

#[test]
fn hybrid_qwen35_1cmdbuf_metal_gpu_matches_cpu_across_cb4_and_wht2() {
    for mode in [0u8, 1u8] {
        let vxt = build_synthetic_qwen35_vxt(mode);
        let mut cpu_model = LlmModel::from_vxt_bytes(&vxt, false, 2).unwrap();
        let mut st_cpu = cpu_model.new_state();
        let mut logits_cpu = vec![0.0f32; cpu_model.cfg.n_vocab];

        if MetalEngine::new().is_ok() {
            let mut gpu_model = LlmModel::from_vxt_bytes(&vxt, true, 2).unwrap();
            assert!(gpu_model.backend_label().contains("1-CmdBuf Fused"));
            let mut st_gpu = gpu_model.new_state();
            let mut logits_gpu = vec![0.0f32; gpu_model.cfg.n_vocab];

            for (step, &tok) in [3u32, 17, 42, 99, 128, 7, 63, 211, 19, 88]
                .iter()
                .enumerate()
            {
                cpu_model
                    .forward_step(&mut st_cpu, tok, Some(&mut logits_cpu))
                    .unwrap();
                gpu_model
                    .forward_step(&mut st_gpu, tok, Some(&mut logits_gpu))
                    .unwrap();
                let m = compare_logits(&logits_cpu, &logits_gpu);
                assert!(
                    m.cosine_sim > 0.999_99 && m.rmse < 1e-4 && m.top1_match,
                    "mode={mode} step={step}: cos={:.8} rmse={:.3e} top1={}",
                    m.cosine_sim,
                    m.rmse,
                    m.top1_match
                );
            }
        } else {
            cpu_model
                .forward_step(&mut st_cpu, 3, Some(&mut logits_cpu))
                .unwrap();
        }
    }
}

#[test]
fn hybrid_qwen35_turbo_pipelined_gpu_matches_cpu_and_batched_decode() {
    for mode in [2u8, 3u8] {
        let vxt = build_synthetic_qwen35_vxt(mode);
        let mut cpu_model = LlmModel::from_vxt_bytes(&vxt, false, 2).unwrap();
        let mut st_cpu = cpu_model.new_state();
        let mut logits_cpu = vec![0.0f32; cpu_model.cfg.n_vocab];

        #[cfg(target_os = "macos")]
        let _ = MetalEngine::new().expect("Metal shader library must compile cleanly on macOS");
        if MetalEngine::new().is_err() {
            cpu_model
                .forward_step(&mut st_cpu, 5, Some(&mut logits_cpu))
                .unwrap();
            continue;
        }

        let mut gpu_model = LlmModel::from_vxt_bytes(&vxt, true, 2).unwrap();
        assert!(gpu_model.has_turbo_fast_decode());
        assert!(gpu_model.backend_label().contains("VXT-Turbo Pipelined"));

        // 1. Step-by-step CPU vs FP16-State / Scaled-FP16-Dot Turbo GPU parity
        let mut st_gpu = gpu_model.new_state();
        let mut logits_gpu = vec![0.0f32; gpu_model.cfg.n_vocab];
        let seq = [5u32, 11, 29, 77, 140, 5, 11, 29, 83, 199];
        for (step, &tok) in seq.iter().enumerate() {
            cpu_model
                .forward_step(&mut st_cpu, tok, Some(&mut logits_cpu))
                .unwrap();
            gpu_model
                .forward_step(&mut st_gpu, tok, Some(&mut logits_gpu))
                .unwrap();
            let m = compare_logits(&logits_cpu, &logits_gpu);
            assert!(
                m.cosine_sim > 0.9995 && m.rmse < 5e-3 && m.top1_match,
                "mode={mode} turbo step={step}: cos={:.6} rmse={:.3e} top1={}",
                m.cosine_sim,
                m.rmse,
                m.top1_match
            );
        }

        // 2. Batched M=4 `prefill` + `generate_greedy_turbo` (both exact & sketch + N-gram rollback!)
        let mut st_ref = gpu_model.new_state();
        let mut logits_ref = vec![0.0f32; gpu_model.cfg.n_vocab];
        for (idx, &tok) in seq.iter().enumerate() {
            let last = idx + 1 == seq.len();
            gpu_model
                .forward_step(&mut st_ref, tok, last.then_some(&mut logits_ref))
                .unwrap();
        }
        let logits_prefill_ref = logits_ref.clone();
        let mut rng = 0u64;
        let first_tok = sample_logits(&logits_ref, 0.0, &mut rng);
        let mut ref_gen = vec![first_tok];
        let mut cur = first_tok;
        for _ in 1..12 {
            gpu_model
                .forward_step(&mut st_ref, cur, Some(&mut logits_ref))
                .unwrap();
            cur = sample_logits(&logits_ref, 0.0, &mut rng);
            ref_gen.push(cur);
        }

        for use_sketch in [false, true] {
            let mut st_fast = gpu_model.new_state();
            let mut logits_fast = vec![0.0f32; gpu_model.cfg.n_vocab];
            gpu_model
                .prefill(&mut st_fast, &seq, &mut logits_fast)
                .unwrap();
            let mp = compare_logits(&logits_prefill_ref, &logits_fast);
            assert!(
                mp.cosine_sim > 0.9999 && mp.top1_match,
                "mode={mode} prefill mismatch: cos={:.6} rmse={:.4e} top1={}",
                mp.cosine_sim,
                mp.rmse,
                mp.top1_match
            );

            let fast_first = sample_logits(&logits_fast, 0.0, &mut rng);
            let fast_gen = gpu_model
                .generate_greedy_turbo(
                    &mut st_fast,
                    &seq,
                    fast_first,
                    12,
                    use_sketch,
                    0.0,
                    &[u32::MAX],
                    |_| true,
                )
                .unwrap();
            assert_eq!(fast_gen, ref_gen, "mode={mode} mismatch for use_sketch={use_sketch}");
        }
    }
}

// Synthetic recurrent+attention qualification. Raw reference weights are built directly
// from the source words, without decoding or re-encoding a packed archive.
#[test]
fn hybrid_lossless_long_context_matches_independent_raw_weights() {
    for mode in [4u8, 5] {
        let compressed = build_synthetic_qwen35_vxt(mode);
        let raw = build_synthetic_qwen35_vxt(mode + 2);
        let mut reference = LlmModel::from_vxt_bytes(&raw, false, 2).unwrap();
        let mut cpu = LlmModel::from_vxt_bytes(&compressed, false, 2).unwrap();
        let mut ref_state = reference.new_state();
        let mut cpu_state = cpu.new_state();
        let mut a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        let require_gpu = std::env::var_os("VXG_REQUIRE_METAL").is_some();
        let mut gpu = if MetalEngine::new().is_ok() {
            let m = LlmModel::from_vxt_bytes(&compressed, true, 2).unwrap();
            assert!(m.backend_label().contains("1-CmdBuf Fused"));
            let state = m.new_state();
            Some((m, state))
        } else {
            assert!(!require_gpu, "VXG_REQUIRE_METAL set but Metal unavailable");
            eprintln!("Metal unavailable; CPU-only qualification");
            None
        };
        let mut g = vec![0.0; 256];
        let mut max_rmse = 0.0f64;
        let mut max_cpu_rmse = 0.0f64;
        let mut max_cpu_abs = 0.0f32;
        let mut max_gpu_abs = 0.0f32;
        for step in 0..256 {
            let tok = ((step * 73 + 19) % 256) as u32;
            reference.forward_step(&mut ref_state, tok, Some(&mut a)).unwrap();
            cpu.forward_step(&mut cpu_state, tok, Some(&mut b)).unwrap();
            assert!(a.iter().all(|v| v.is_finite()));
            assert!(b.iter().all(|v| v.is_finite()));
            // Raw and BP16 GEMV kernels accumulate in different orders; exact source
            // weights do not imply bit-identical floating-point inference.
            max_cpu_abs = max_cpu_abs.max(a.iter().zip(&b).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max));
            let c = compare_logits(&a, &b);
            max_cpu_rmse = max_cpu_rmse.max(c.rmse);
            assert!(c.rmse < 1e-4 && c.cosine_sim > 0.99999 && c.top1_match, "mode={mode} CPU step={step} rmse={} cos={} top1={}", c.rmse, c.cosine_sim, c.top1_match);
            if let Some((m, st)) = gpu.as_mut() {
                m.forward_step(st, tok, Some(&mut g)).unwrap();
                assert!(g.iter().all(|v| v.is_finite()));
                max_gpu_abs = max_gpu_abs.max(a.iter().zip(&g).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max));
                let metrics = compare_logits(&a, &g);
                max_rmse = max_rmse.max(metrics.rmse);
                assert!(metrics.rmse < 1e-4 && metrics.cosine_sim > 0.99999 && metrics.top1_match, "mode={mode} GPU step={step} rmse={} cos={} top1={}", metrics.rmse, metrics.cosine_sim, metrics.top1_match);
            }
        }
        eprintln!("PASS synthetic mode={mode} steps=256 CPU raw-reference max_rmse={max_cpu_rmse:e}; CPU max_abs={max_cpu_abs:e}; GPU={} max_rmse={max_rmse:e} max_abs={max_gpu_abs:e}", gpu.is_some());
    }
}

// This is explicitly a device test: an unavailable Metal device is a failure.
#[test]
#[cfg(target_os = "macos")]
fn turbo_stop_state_matches_sequential_gpu_replay() {
    MetalEngine::new().expect("Turbo stop-state test requires a real Metal device");
    let bytes = build_synthetic_qwen35_vxt(3);
    let mut model = LlmModel::from_vxt_bytes(&bytes, true, 2).unwrap();
    assert!(model.has_turbo_fast_decode());
    let prompt = [5, 11, 29, 77, 140];
    let mut rng = 0;
    for sketch in [false, true] {
        for &(cap, cancel) in &[(0, 0), (1, 0), (40, 1), (8, 0), (9, 0), (10, 0),
            (17, 0), (40, 8), (40, 9), (40, 16), (40, 17), (270, 257)] {
            let mut fast = model.new_state();
            let mut reference = model.new_state();
            let mut a = vec![0.0; model.cfg.n_vocab];
            let mut b = a.clone();
            model.prefill(&mut fast, &prompt, &mut a).unwrap();
            model.prefill(&mut reference, &prompt, &mut b).unwrap();
            let first = sample_logits(&a, 0.0, &mut rng);
            let mut calls = 0;
            let generated = model.generate_greedy_turbo(&mut fast, &prompt, first,
                cap, sketch, 0.7, &[], |_| { calls += 1; cancel == 0 || calls < cancel }).unwrap();
            let consumed = generated.len().saturating_sub(1);
            assert_eq!(fast.pos, prompt.len() + consumed, "cap={cap} cancel={cancel}");
            for &token in generated.iter().take(consumed) {
                model.forward_step(&mut reference, token, Some(&mut b)).unwrap();
            }
            // Continuing through several tokens checks recurrent state, valid KV
            // prefix, overwritten speculative suffix, and presence-penalty reset.
            for token in [37, 93, 201] {
                model.forward_step(&mut fast, token, Some(&mut a)).unwrap();
                model.forward_step(&mut reference, token, Some(&mut b)).unwrap();
                assert_eq!(a, b, "continuation cap={cap} cancel={cancel} sketch={sketch}");
            }
            let first = sample_logits(&a, 0.0, &mut rng);
            let x = model.generate_greedy_turbo(&mut fast, &[], first, 18, sketch, 0.0, &[], |_| true).unwrap();
            let y = model.generate_greedy_turbo(&mut reference, &[], first, 18, sketch, 0.0, &[], |_| true).unwrap();
            assert_eq!(x, y);
        }
        // Initial KV capacity is 1024. At position 810, the second pending
        // eight-step command crosses its +200 growth reserve and must fence
        // writes from the first command before copying the old cache.
        let long_prompt: Vec<u32> = prompt.iter().copied().cycle().take(810).collect();
        let mut fast = model.new_state();
        let mut reference = model.new_state();
        let mut a = vec![0.0; model.cfg.n_vocab];
        let mut b = a.clone();
        model.prefill(&mut fast, &long_prompt, &mut a).unwrap();
        model.prefill(&mut reference, &long_prompt, &mut b).unwrap();
        let first = sample_logits(&a, 0.0, &mut rng);
        let mut calls = 0;
        let generated = model.generate_greedy_turbo(&mut fast, &long_prompt, first, 40,
            sketch, 0.0, &[], |_| { calls += 1; calls < 17 }).unwrap();
        assert_eq!(fast.pos, 826);
        for &token in generated.iter().take(16) {
            model.forward_step(&mut reference, token, Some(&mut b)).unwrap();
        }
        model.forward_step(&mut fast, 37, Some(&mut a)).unwrap();
        model.forward_step(&mut reference, 37, Some(&mut b)).unwrap();
        assert_eq!(a, b, "KV growth with pending commands");
        // A panicking callback must drain queued work before unwinding, and
        // leave the same accepted prefix available to a caller that catches it.
        let mut fast = model.new_state();
        let mut reference = model.new_state();
        let mut a = vec![0.0; model.cfg.n_vocab];
        let mut b = a.clone();
        model.prefill(&mut fast, &prompt, &mut a).unwrap();
        model.prefill(&mut reference, &prompt, &mut b).unwrap();
        let first = sample_logits(&a, 0.0, &mut rng);
        let mut delivered = Vec::new();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.generate_greedy_turbo(&mut fast, &prompt, first, 40, sketch, 0.7, &[], |token| {
                delivered.push(token);
                assert!(delivered.len() < 9, "intentional callback panic");
                true
            }).unwrap();
        }));
        assert!(panic.is_err());
        assert_eq!(fast.pos, prompt.len() + 8);
        for &token in delivered.iter().take(8) {
            model.forward_step(&mut reference, token, Some(&mut b)).unwrap();
        }
        model.forward_step(&mut fast, 37, Some(&mut a)).unwrap();
        model.forward_step(&mut reference, 37, Some(&mut b)).unwrap();
        assert_eq!(a, b, "callback panic continuation");
        // Select actual predicted IDs as stop IDs, including first-token stop.
        let mut trace_state = model.new_state();
        let mut logits = vec![0.0; model.cfg.n_vocab];
        model.prefill(&mut trace_state, &prompt, &mut logits).unwrap();
        let first = sample_logits(&logits, 0.0, &mut rng);
        let trace = model.generate_greedy_turbo(&mut trace_state, &prompt, first, 40,
            sketch, 0.0, &[], |_| true).unwrap();
        for index in [0, 1, 7, 8, 15, 16, 32] {
            let stop = trace[index];
            let boundary = trace.iter().position(|&t| t == stop).unwrap();
            let mut fast = model.new_state();
            let mut reference = model.new_state();
            model.prefill(&mut fast, &prompt, &mut logits).unwrap();
            model.prefill(&mut reference, &prompt, &mut logits).unwrap();
            let out = model.generate_greedy_turbo(&mut fast, &prompt, first, 40,
                sketch, 0.0, &[stop], |_| true).unwrap();
            assert_eq!(out, trace[..boundary]);
            assert_eq!(fast.pos, prompt.len() + boundary);
            let mut other = logits.clone();
            for &token in &out { model.forward_step(&mut reference, token, Some(&mut other)).unwrap(); }
            for token in [37, 93, 201] {
                model.forward_step(&mut fast, token, Some(&mut logits)).unwrap();
                model.forward_step(&mut reference, token, Some(&mut other)).unwrap();
                assert_eq!(logits, other, "EOS boundary={boundary} sketch={sketch}");
            }
        }
    }
}

#[test]
#[ignore = "requires retained production artifact via VXG_TURBO_STOP_MODEL and Metal"]
#[cfg(target_os = "macos")]
fn turbo_real_model_stop_and_resume() {
    MetalEngine::new().expect("real-model stop test requires Metal");
    let path = std::env::var("VXG_TURBO_STOP_MODEL").expect("set retained Turbo model path");
    let mut model = LlmModel::load_file(&path, true, 2).unwrap();
    assert!(model.has_turbo_fast_decode());
    let prompt = model.tokenizer.encode_chat("What is 17 times 23? Show a short calculation.", &model.cfg.arch, false);
    let stop = [model.cfg.eos_token_id,
        model.tokenizer.token_id("<|im_end|>").unwrap_or(model.cfg.eos_token_id),
        model.tokenizer.token_id("<|endoftext|>").unwrap_or(model.cfg.eos_token_id)];
    let mut fast = model.new_state();
    let mut reference = model.new_state();
    let mut a = vec![0.0; model.cfg.n_vocab];
    let mut b = a.clone();
    model.prefill(&mut fast, &prompt, &mut a).unwrap();
    model.prefill(&mut reference, &prompt, &mut b).unwrap();
    let first = sample_logits(&a, 0.0, &mut 0);
    let generated = model.generate_greedy_turbo(&mut fast, &prompt, first, 128,
        true, 0.0, &stop, |_| true).unwrap();
    assert!(generated.len() < 128, "expected retained early-EOS reproduction");
    assert_eq!(fast.pos, prompt.len() + generated.len());
    eprintln!("real prompt={} emitted={} pos={} text={}", prompt.len(), generated.len(), fast.pos, model.tokenizer.decode(&generated));
    for &token in &generated { model.forward_step(&mut reference, token, Some(&mut b)).unwrap(); }
    for token in [first, 37, 93] {
        model.forward_step(&mut fast, token, Some(&mut a)).unwrap();
        model.forward_step(&mut reference, token, Some(&mut b)).unwrap();
        assert_eq!(a, b, "real continuation logits");
    }
    let first = sample_logits(&a, 0.0, &mut 0);
    let x = model.generate_greedy_turbo(&mut fast, &[], first, 20, true, 0.0, &[], |_| true).unwrap();
    let y = model.generate_greedy_turbo(&mut reference, &[], first, 20, true, 0.0, &[], |_| true).unwrap();
    assert_eq!(x, y, "real resumed generated sequence");
    let mut timing = model.new_state();
    model.prefill(&mut timing, &prompt, &mut a).unwrap();
    let first = sample_logits(&a, 0.0, &mut 0);
    let start = std::time::Instant::now();
    let capped = model.generate_greedy_turbo(&mut timing, &prompt, first, 128,
        true, 0.0, &[], |_| true).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(capped.len(), 128);
    assert_eq!(timing.pos, prompt.len() + 127);
    eprintln!("capped128 diagnostic elapsed={elapsed:.6}s processed127_tps={:.2} pos={}", 127.0 / elapsed, timing.pos);
}

// Explicit device requirement: unavailable Metal fails rather than skips.
#[test]
#[cfg(target_os = "macos")]
fn cb4_greedy_stop_state_matches_sequential_gpu_replay() {
    MetalEngine::new().expect("CB4 greedy test requires a real Metal device");
    let bytes = build_synthetic_qwen35_vxt(0);
    let mut model = LlmModel::from_vxt_bytes(&bytes, true, 2).unwrap();
    model.enable_cb4_greedy().unwrap();
    let prompt = [5, 11, 29, 77, 140];
    let mut rng = 0;
    for &(cap, cancel) in &[(0,0), (1,0), (8,0), (17,0), (40,1), (40,8), (40,17)] {
        let mut fast = model.new_state();
        let mut reference = model.new_state();
        let mut a = vec![0.; model.cfg.n_vocab];
        let mut b = a.clone();
        model.prefill(&mut fast, &prompt, &mut a).unwrap();
        model.prefill(&mut reference, &prompt, &mut b).unwrap();
        let first = sample_logits(&a, 0., &mut rng);
        let mut calls = 0;
        let emitted = model.generate_greedy_cb4(&mut fast, first, cap, &[], |_, _| {
            calls += 1; cancel == 0 || calls < cancel
        }).unwrap();
        assert_eq!(calls, emitted.len());
        let consumed = emitted.len().saturating_sub(1);
        assert_eq!(fast.pos, prompt.len() + consumed);
        for (i, &token) in emitted.iter().enumerate() {
            assert_eq!(token, sample_logits(&b, 0., &mut rng), "token {i}");
            if i < consumed { model.forward_step(&mut reference, token, Some(&mut b)).unwrap(); }
        }
        let pending = emitted.last().copied().unwrap_or(first);
        for _ in 0..3 {
            model.forward_step(&mut fast, pending, Some(&mut a)).unwrap();
            model.forward_step(&mut reference, pending, Some(&mut b)).unwrap();
            assert_eq!(a, b, "continuation cap={cap} cancel={cancel}");
        }
        let pos = fast.pos;
        let stopped = model.generate_greedy_cb4(&mut fast, first, 10, &[first], |_, _| panic!("EOS callback")).unwrap();
        assert!(stopped.is_empty()); assert_eq!(fast.pos, pos);
    }
    let mut trace_state = model.new_state();
    let mut logits = vec![0.; model.cfg.n_vocab];
    model.prefill(&mut trace_state, &prompt, &mut logits).unwrap();
    let first = sample_logits(&logits, 0., &mut rng);
    let trace = model.generate_greedy_cb4(&mut trace_state, first, 12, &[], |_, _| true).unwrap();
    for &stop in &trace {
        let mut stopped = model.new_state();
        let mut replay = model.new_state();
        model.prefill(&mut stopped, &prompt, &mut logits).unwrap();
        model.prefill(&mut replay, &prompt, &mut logits).unwrap();
        let emitted = model.generate_greedy_cb4(&mut stopped, first, 20, &[stop], |_, _| true).unwrap();
        let count = trace.iter().position(|&t| t == stop).unwrap();
        assert_eq!(emitted, trace[..count]);
        assert_eq!(stopped.pos, prompt.len() + count);
        for token in emitted { model.forward_step(&mut replay, token, Some(&mut logits)).unwrap(); }
        let mut actual = logits.clone();
        model.forward_step(&mut stopped, stop, Some(&mut actual)).unwrap();
        model.forward_step(&mut replay, stop, Some(&mut logits)).unwrap();
        assert_eq!(actual, logits);
    }
    let mut panicked = model.new_state();
    let mut replay = model.new_state();
    model.prefill(&mut panicked, &prompt, &mut logits).unwrap();
    model.prefill(&mut replay, &prompt, &mut logits).unwrap();
    let mut visible = Vec::new();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.generate_greedy_cb4(&mut panicked, first, 20, &[], |t, _| {
            visible.push(t); if visible.len() == 8 { panic!("deliberate callback panic"); } true
        }).unwrap();
    }));
    assert!(panic.is_err());
    assert_eq!(panicked.pos, prompt.len() + visible.len() - 1);
    for &token in visible.iter().take(visible.len() - 1) {
        model.forward_step(&mut replay, token, Some(&mut logits)).unwrap();
    }
    let pending = *visible.last().unwrap();
    let mut actual = logits.clone();
    model.forward_step(&mut panicked, pending, Some(&mut actual)).unwrap();
    model.forward_step(&mut replay, pending, Some(&mut logits)).unwrap();
    assert_eq!(actual, logits);
    let mut cpu = LlmModel::from_vxt_bytes(&bytes, false, 2).unwrap();
    assert!(cpu.enable_cb4_greedy().is_err());
}

#[test]
#[cfg(target_os = "macos")]
fn cb4_greedy_nonfinite_device_error_invalidates_state() {
    MetalEngine::new().expect("CB4 failure test requires a real Metal device");
    let bytes = build_synthetic_qwen35_vxt(0);
    let (meta, mut entries) = vxg_tensor::format::read_vxt_model(&bytes).unwrap();
    entries.push(VxtEntry { name: "output.weight".into(), tensor: PackedTensor::RawF32 {
        rows: 256, cols: 256, data: vec![f32::NAN; 256 * 256],
    }});
    let bytes = write_vxt_model(&meta, &entries).unwrap();
    let mut model = LlmModel::from_vxt_bytes(&bytes, true, 2).unwrap();
    model.enable_cb4_greedy().unwrap();
    let mut state = model.new_state();
    assert!(model.generate_greedy_cb4(&mut state, 5, 2, &[], |_, _| true).is_err());
    assert_eq!(state.pos, 0);
    let mut logits = vec![0.; 256];
    assert!(model.forward_step(&mut state, 5, Some(&mut logits)).is_err());
    assert!(model.prefill(&mut state, &[5], &mut logits).is_err());
    assert!(model.generate_greedy_cb4(&mut state, 5, 2, &[], |_, _| true).is_err());
}

// Experimental pipeline selection mutates process environment. Qualification
// tests are ignored by default and must run alone with --test-threads=1.
#[cfg(target_os = "macos")]
struct Cb4ExperimentalEnvGuard(Option<std::ffi::OsString>);
#[cfg(target_os = "macos")]
impl Cb4ExperimentalEnvGuard {
    fn capture() -> Self { Self(std::env::var_os("VXG_CB4_ROWS4")) }
}
#[cfg(target_os = "macos")]
impl Drop for Cb4ExperimentalEnvGuard {
    fn drop(&mut self) {
        if let Some(value)=&self.0 { std::env::set_var("VXG_CB4_ROWS4",value); }
        else { std::env::remove_var("VXG_CB4_ROWS4"); }
    }
}

// Run this opt-in comparison alone (--test-threads=1): the environment selects
// a pipeline once at context creation; existing contexts must remain independent.
#[test]
#[ignore = "requires isolated actual Metal qualification with --ignored --test-threads=1"]
#[cfg(target_os = "macos")]
fn cb4_rows4_matches_rows2_bits_and_synthetic_logits() {
    let _env = Cb4ExperimentalEnvGuard::capture();
    std::env::remove_var("VXG_CB4_ROWS4");
    let mut baseline = MetalEngine::new().expect("rows4 comparison requires actual Metal");
    let bytes = build_synthetic_qwen35_vxt(0);
    let mut a = LlmModel::from_vxt_bytes(&bytes, true, 2).unwrap();
    std::env::set_var("VXG_CB4_ROWS4", "1");
    let mut candidate = MetalEngine::new().expect("rows4 comparison requires actual Metal");
    let mut b = LlmModel::from_vxt_bytes(&bytes, true, 2).unwrap();
    // Odd row counts exercise masked SIMD groups; unaligned logical columns
    // exercise zero padding. Duplicate sparse columns preserve reduction order.
    for rows in [1, 3, 4, 5, 7, 9, 17] {
        for cols in [1, 129, 257, 513] {
            let mut cb = CodebookMatrix::encode(4, rows, cols, &pseudo_vec(93, rows*cols, 0.1)).unwrap();
            cb.outlier_row_ptr = (0..=rows).map(|r| (r*3) as u32).collect();
            cb.outliers_packed = (0..rows).flat_map(|_| [
                (vxg_tensor::fp16::f32_to_f16(0.25) as u32) << 16,
                (vxg_tensor::fp16::f32_to_f16(-0.125) as u32) << 16,
                ((vxg_tensor::fp16::f32_to_f16(0.5) as u32) << 16) | (cols-1) as u32,
            ]).collect();
            let tensor = PackedTensor::Codebook(cb);
            let ga = baseline.upload_packed(&tensor).unwrap();
            let gb = candidate.upload_packed(&tensor).unwrap();
            let x = pseudo_vec(41, cols, 0.5);
            let mut ya = vec![f32::NAN; rows]; let mut yb = ya.clone();
            baseline.run_multi_same_x(&x, &mut [(&ga, &mut ya)]).unwrap();
            candidate.run_multi_same_x(&x, &mut [(&gb, &mut yb)]).unwrap();
            assert!(ya.iter().chain(&yb).all(|x| x.is_finite()));
            assert_eq!(ya.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                yb.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "rows={rows} cols={cols}");
        }
    }
    let mut sa = a.new_state(); let mut sb = b.new_state();
    let mut la = vec![0.; a.cfg.n_vocab]; let mut lb = la.clone();
    for token in [5,11,29,77,140].into_iter().chain((0..32).map(|i| (i*37+11)%256)) {
        a.forward_step(&mut sa, token, Some(&mut la)).unwrap();
        b.forward_step(&mut sb, token, Some(&mut lb)).unwrap();
        assert!(la.iter().chain(&lb).all(|x| x.is_finite()));
        assert_eq!(la.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            lb.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "synthetic pos={}",sa.pos);
    }
    assert_eq!(sa.pos, sb.pos);
}

#[test]
#[ignore = "requires retained CB4 artifact in VXG_CB4_ROWS_MODEL"]
#[cfg(target_os = "macos")]
fn cb4_rows4_real_model_32_steps_full_logit_bits() {
    let _env = Cb4ExperimentalEnvGuard::capture();
    let path = std::env::var("VXG_CB4_ROWS_MODEL").expect("set retained artifact path");
    std::env::remove_var("VXG_CB4_ROWS4");
    let mut a = LlmModel::load_file(&path, true, 2).unwrap();
    assert!(a.backend_label().contains("1-CmdBuf Fused"));
    let ids = a.tokenizer.encode_chat("What is 17 times 23? Show a short calculation.", &a.cfg.arch, false);
    let vocab = a.cfg.n_vocab;
    let mut sa = a.new_state();
    let mut logits = vec![0.; vocab];
    a.prefill(&mut sa, &ids, &mut logits).unwrap();
    assert!(logits.iter().all(|x| x.is_finite()));
    let mut expected = vec![logits.iter().map(|x|x.to_bits()).collect::<Vec<_>>()];
    let mut tokens = Vec::new();
    for _ in 0..32 {
        let token = sample_logits(&logits, 0., &mut 0); tokens.push(token);
        a.forward_step(&mut sa, token, Some(&mut logits)).unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        expected.push(logits.iter().map(|x|x.to_bits()).collect());
    }
    let final_pos = sa.pos;
    drop(sa); drop(a); // Never retain two production models on the host.
    std::env::set_var("VXG_CB4_ROWS4", "1");
    let mut b = LlmModel::load_file(&path, true, 2).unwrap();
    std::env::remove_var("VXG_CB4_ROWS4");
    assert!(b.backend_label().contains("1-CmdBuf Fused"));
    let mut sb = b.new_state();
    b.prefill(&mut sb, &ids, &mut logits).unwrap();
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(expected[0],logits.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
    for (step,&token) in tokens.iter().enumerate() {
        b.forward_step(&mut sb, token, Some(&mut logits)).unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        assert_eq!(expected[step+1],logits.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),"real step={step}");
    }
    assert_eq!(final_pos,sb.pos);
    eprintln!("rows4 real full-logit bits equal: prompt={} generation_steps=32 vocab={vocab}",ids.len());
}
