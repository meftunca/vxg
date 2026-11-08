use std::fs;
use std::path::PathBuf;

use serde_json::json;
use vxg_tensor::bench::generate_realistic_llm_weights;
use vxg_tensor::hf_pack::{inspect_hf_model_dir, pack_hf_model_to_vxt, resolve_hf_model_dir};
use vxg_tensor::llm::LlmModel;
use vxg_tensor::llm_pack::LlmPackMode;
use vxg_tensor::safetensors::write_safetensors_u16;
use vxg_tensor::DType;

fn build_sharded_hf_fixture(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();

    // 4-layer miniature hybrid Qwen3.5 (layers 0,1,2 recurrent DeltaNet, layer 3 full Gated-GQA)
    let cfg = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 264,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "eos_token_id": 256,
            "tie_word_embeddings": true,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }
    });
    fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();

    let mut vocab = serde_json::Map::new();
    for b in 0u32..256 {
        vocab.insert(format!("t{b}"), json!(b));
    }
    let tok = json!({
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [["t72", "t105"], "t65 t66"]
        },
        "added_tokens": [
            {"id": 256, "content": "<|endoftext|>", "special": true},
            {"id": 257, "content": "<|im_start|>", "special": true},
            {"id": 258, "content": "<|im_end|>", "special": true},
            {"id": 259, "content": "<think>", "special": false},
            {"id": 260, "content": "</think>", "special": false}
        ]
    });
    fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok).unwrap()).unwrap();

    let mk_mat = |r: usize, c: usize, seed: u64| -> Vec<u16> {
        let (_, f32s) = generate_realistic_llm_weights(DType::BF16, r, c.max(256), seed);
        f32s.into_iter()
            .take(r * c)
            .map(|v| vxg_tensor::fp16::f32_to_bf16(v * 0.35))
            .collect()
    };
    let zero_bf16 = |n: usize| -> Vec<u16> { vec![0u16; n] };
    let one_bf16 = |n: usize| -> Vec<u16> { vec![0x3F80u16; n] };

    // Shard 1: embeddings + layers 0..2 + excluded vision tensor
    let mut s1: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
        ("model.language_model.embed_tokens.weight".into(), vec![264, 256], mk_mat(264, 256, 1)),
        ("model.visual.patch_embed.weight".into(), vec![64, 64], mk_mat(64, 64, 999)),
    ];
    for il in 0..2 {
        let p = format!("model.language_model.layers.{il}");
        s1.push((format!("{p}.input_layernorm.weight"), vec![256], zero_bf16(256)));
        s1.push((format!("{p}.post_attention_layernorm.weight"), vec![256], zero_bf16(256)));
        s1.push((format!("{p}.mlp.gate_proj.weight"), vec![512, 256], mk_mat(512, 256, 10 + il as u64)));
        s1.push((format!("{p}.mlp.up_proj.weight"), vec![512, 256], mk_mat(512, 256, 20 + il as u64)));
        s1.push((format!("{p}.mlp.down_proj.weight"), vec![256, 512], mk_mat(256, 512, 30 + il as u64)));
        s1.push((format!("{p}.linear_attn.in_proj_qkv.weight"), vec![768, 256], mk_mat(768, 256, 40 + il as u64)));
        s1.push((format!("{p}.linear_attn.in_proj_z.weight"), vec![256, 256], mk_mat(256, 256, 50 + il as u64)));
        s1.push((format!("{p}.linear_attn.in_proj_a.weight"), vec![4, 256], mk_mat(4, 256, 60 + il as u64)));
        s1.push((format!("{p}.linear_attn.in_proj_b.weight"), vec![4, 256], mk_mat(4, 256, 70 + il as u64)));
        s1.push((format!("{p}.linear_attn.out_proj.weight"), vec![256, 256], mk_mat(256, 256, 80 + il as u64)));
        s1.push((format!("{p}.linear_attn.conv1d.weight"), vec![768, 1, 4], mk_mat(768, 4, 90 + il as u64)));
        s1.push((format!("{p}.linear_attn.norm.weight"), vec![64], one_bf16(64)));
        s1.push((format!("{p}.linear_attn.A_log"), vec![4], one_bf16(4)));
        s1.push((format!("{p}.linear_attn.dt_bias"), vec![4], zero_bf16(4)));
    }

    // Shard 2: layer 2 (recurrent) + layer 3 (full Gated-GQA) + final norm
    let mut s2: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
    {
        let p = "model.language_model.layers.2";
        s2.push((format!("{p}.input_layernorm.weight"), vec![256], zero_bf16(256)));
        s2.push((format!("{p}.post_attention_layernorm.weight"), vec![256], zero_bf16(256)));
        s2.push((format!("{p}.mlp.gate_proj.weight"), vec![512, 256], mk_mat(512, 256, 112)));
        s2.push((format!("{p}.mlp.up_proj.weight"), vec![512, 256], mk_mat(512, 256, 122)));
        s2.push((format!("{p}.mlp.down_proj.weight"), vec![256, 512], mk_mat(256, 512, 132)));
        s2.push((format!("{p}.linear_attn.in_proj_qkv.weight"), vec![768, 256], mk_mat(768, 256, 142)));
        s2.push((format!("{p}.linear_attn.in_proj_z.weight"), vec![256, 256], mk_mat(256, 256, 152)));
        s2.push((format!("{p}.linear_attn.in_proj_a.weight"), vec![4, 256], mk_mat(4, 256, 162)));
        s2.push((format!("{p}.linear_attn.in_proj_b.weight"), vec![4, 256], mk_mat(4, 256, 172)));
        s2.push((format!("{p}.linear_attn.out_proj.weight"), vec![256, 256], mk_mat(256, 256, 182)));
        s2.push((format!("{p}.linear_attn.conv1d.weight"), vec![768, 1, 4], mk_mat(768, 4, 192)));
        s2.push((format!("{p}.linear_attn.norm.weight"), vec![64], one_bf16(64)));
        s2.push((format!("{p}.linear_attn.A_log"), vec![4], one_bf16(4)));
        s2.push((format!("{p}.linear_attn.dt_bias"), vec![4], zero_bf16(4)));
    }
    {
        let p = "model.language_model.layers.3";
        s2.push((format!("{p}.input_layernorm.weight"), vec![256], zero_bf16(256)));
        s2.push((format!("{p}.post_attention_layernorm.weight"), vec![256], zero_bf16(256)));
        s2.push((format!("{p}.mlp.gate_proj.weight"), vec![512, 256], mk_mat(512, 256, 213)));
        s2.push((format!("{p}.mlp.up_proj.weight"), vec![512, 256], mk_mat(512, 256, 223)));
        s2.push((format!("{p}.mlp.down_proj.weight"), vec![256, 512], mk_mat(256, 512, 233)));
        s2.push((format!("{p}.self_attn.q_proj.weight"), vec![512, 256], mk_mat(512, 256, 243)));
        s2.push((format!("{p}.self_attn.k_proj.weight"), vec![128, 256], mk_mat(128, 256, 253)));
        s2.push((format!("{p}.self_attn.v_proj.weight"), vec![128, 256], mk_mat(128, 256, 263)));
        s2.push((format!("{p}.self_attn.o_proj.weight"), vec![256, 256], mk_mat(256, 256, 273)));
        s2.push((format!("{p}.self_attn.q_norm.weight"), vec![64], zero_bf16(64)));
        s2.push((format!("{p}.self_attn.k_norm.weight"), vec![64], zero_bf16(64)));
    }
    s2.push(("model.language_model.norm.weight".into(), vec![256], zero_bf16(256)));

    let write_shard = |fname: &str, items: &[(String, Vec<usize>, Vec<u16>)]| {
        let refs: Vec<(&str, DType, &[usize], &[u16])> = items
            .iter()
            .map(|(n, sh, w)| (n.as_str(), DType::BF16, sh.as_slice(), w.as_slice()))
            .collect();
        fs::write(dir.join(fname), write_safetensors_u16(&refs)).unwrap();
    };
    write_shard("model-00001-of-00002.safetensors", &s1);
    write_shard("model-00002-of-00002.safetensors", &s2);

    let mut wmap = serde_json::Map::new();
    for (n, _, _) in &s1 {
        wmap.insert(n.clone(), json!("model-00001-of-00002.safetensors"));
    }
    for (n, _, _) in &s2 {
        wmap.insert(n.clone(), json!("model-00002-of-00002.safetensors"));
    }
    fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&json!({"weight_map": wmap})).unwrap(),
    )
    .unwrap();
}

#[test]
fn hf_sharded_dir_packs_lossless_and_cb4_v2_and_runs_on_metal() {
    let dir = std::env::temp_dir().join("vxt_hf_sharded_test_fixture");
    build_sharded_hf_fixture(&dir);

    let hf = resolve_hf_model_dir(&dir).expect("should detect sharded HF model dir");
    assert_eq!(hf.shard_paths.len(), 2);
    let summary = inspect_hf_model_dir(&hf).unwrap();
    assert_eq!(summary.cfg.arch, "qwen35");
    assert_eq!(summary.shard_count, 2);

    let (cfg_ll, in_bytes, vxt_ll) = pack_hf_model_to_vxt(&dir, LlmPackMode::Lossless, 2).unwrap();
    assert_eq!(cfg_ll.arch, "qwen35");
    assert!(in_bytes > vxt_ll.len() as u64);

    // Also verify selecting a single shard inside a sharded HF folder resolves the whole model
    let (_, _, vxt_cb4) =
        pack_hf_model_to_vxt(&dir.join("model-00001-of-00002.safetensors"), LlmPackMode::Cb4, 2)
            .unwrap();
    assert!(vxt_cb4.len() < vxt_ll.len());

    let mut m_ll = LlmModel::from_vxt_bytes(&vxt_ll, true, 2).unwrap();
    let mut m_cb4 = LlmModel::from_vxt_bytes(&vxt_cb4, true, 2).unwrap();
    if vxg_tensor::metal::MetalEngine::new().is_ok() {
        assert!(m_cb4.backend_label().contains("CB4-Fast"));
    }

    let mut st_ll = m_ll.new_state();
    let mut st_cb4 = m_cb4.new_state();
    let mut log_ll = vec![0.0f32; cfg_ll.n_vocab];
    let mut log_cb4 = vec![0.0f32; cfg_ll.n_vocab];
    m_ll.prefill(&mut st_ll, &[1, 2, 3, 4], &mut log_ll).unwrap();
    m_cb4.prefill(&mut st_cb4, &[1, 2, 3, 4], &mut log_cb4).unwrap();

    assert!(log_ll.iter().all(|x| x.is_finite()));
    assert!(log_cb4.iter().all(|x| x.is_finite()));
    let mut m_ll_cpu = LlmModel::from_vxt_bytes(&vxt_ll, false, 2).unwrap();
    let mut m_cb4_cpu = LlmModel::from_vxt_bytes(&vxt_cb4, false, 2).unwrap();
    let mut log_ll_cpu = vec![0.0f32; cfg_ll.n_vocab];
    let mut log_cb4_cpu = vec![0.0f32; cfg_ll.n_vocab];
    let mut st_ll_cpu = m_ll_cpu.new_state();
    let mut st_cb4_cpu = m_cb4_cpu.new_state();
    m_ll_cpu.prefill(&mut st_ll_cpu, &[1, 2, 3, 4], &mut log_ll_cpu).unwrap();
    m_cb4_cpu.prefill(&mut st_cb4_cpu, &[1, 2, 3, 4], &mut log_cb4_cpu).unwrap();

    assert!(vxg_tensor::llm::compare_logits(&log_ll_cpu, &log_ll).cosine_sim > 0.999999);
    assert!(vxg_tensor::llm::compare_logits(&log_cb4_cpu, &log_cb4).cosine_sim > 0.999999);
    let metrics = vxg_tensor::llm::compare_logits(&log_ll, &log_cb4);
    assert!(metrics.cosine_sim > 0.94, "cos_sim={}", metrics.cosine_sim);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore]
fn real_qwen35_hf_dir_packs_cb4_v2_and_runs_cb4_fast() {
    let hf_dir = PathBuf::from("/private/tmp/vxg-real-qualification-20261104/native-source");
    if !hf_dir.is_dir() {
        return;
    }
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let (cfg, raw_bytes, vxt_bytes) =
        pack_hf_model_to_vxt(&hf_dir, LlmPackMode::Cb4, threads).unwrap();
    assert_eq!(cfg.arch, "qwen35");
    assert_eq!(cfg.n_vocab, 248320);
    let out_mib = vxt_bytes.len() as f64 / (1024.0 * 1024.0);
    assert!(out_mib < 1010.0, "expected < 1 GiB CB4-v2 archive, got {out_mib:.2} MiB");
    assert!(raw_bytes > 4_000_000_000);

    let mut model = LlmModel::from_vxt_bytes(&vxt_bytes, true, threads).unwrap();
    assert!(model.backend_label().contains("CB4-Fast"));
    model.enable_cb4_greedy().unwrap();

    let prompt = model
        .tokenizer
        .encode_chat("2 + 2 kaç eder? Sadece sayıyı yaz.", "qwen35", false);
    let mut state = model.new_state();
    let mut logits = vec![0.0f32; cfg.n_vocab];
    model.prefill(&mut state, &prompt, &mut logits).unwrap();
    let mut rng = 1u64;
    let first = vxg_tensor::llm::sample_logits(&logits, 0.0, &mut rng);
    let im_end = model.tokenizer.token_id("<|im_end|>").unwrap_or(cfg.eos_token_id);
    let gen = model
        .generate_greedy_cb4(&mut state, first, 16, &[cfg.eos_token_id, im_end], |_, _| true)
        .unwrap();
    let text = model.tokenizer.decode(&gen);
    eprintln!("HF -> CB4-v2 ({out_mib:.2} MiB) reply: {text:?}");
    assert!(text.contains('4'), "unexpected reply: {text:?}");
}
