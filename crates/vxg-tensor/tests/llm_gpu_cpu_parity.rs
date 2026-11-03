use std::collections::HashMap;

use vxg_tensor::codebook::CodebookMatrix;
use vxg_tensor::format::{write_vxt_model, PackedTensor, VxtEntry};
use vxg_tensor::gguf_meta::{GgufVal, ModelConfig};
use vxg_tensor::llm::{compare_logits, LlmModel};
use vxg_tensor::metal::MetalEngine;
use vxg_tensor::tokenizer::BpeTokenizer;
use vxg_tensor::wht2::Wht2Matrix;

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

fn build_synthetic_qwen35_vxt(use_wht2: bool) -> Vec<u8> {
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
        let tensor = if rows < 64 {
            PackedTensor::RawF32 { rows, cols, data }
        } else if use_wht2 {
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
    for use_wht2 in [false, true] {
        let vxt = build_synthetic_qwen35_vxt(use_wht2);
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
                    "wht2={use_wht2} step={step}: cos={:.8} rmse={:.3e} top1={}",
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
