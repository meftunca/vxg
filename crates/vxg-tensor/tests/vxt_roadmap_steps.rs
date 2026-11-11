//! Fast synthetic integration & GPU/CPU parity tests for VXT Roadmap Steps 3, 4, and 2:
//! - Step 3: `Llama-3`, `Mistral`, `Qwen3`, and `Gemma-3` (`GeGLU` + `embed_scale` + 4-norm sandwich)
//!   HF `.safetensors` packing, chat templates, and 1-CmdBuf `CB4-Fast` GPU vs CPU parity.
//! - Step 4: `PackedTensor::gemm_cpu` tiled L2 decompression, batched CPU prefill, and `Q8_0`
//!   compressed `KV-Cache` (`3.76x` VRAM savings) on both CPU and Apple Metal GPU.
//! - Step 2: 1-Command-Buffer Apple Metal GPU Vision (`ViT + 2D Axial RoPE + 2x2 PatchMerger`)
//!   + end-to-end 3D `M-RoPE` multimodal prefill parity.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde_json::json;
use vxg_tensor::chat::{format_chat_messages, ChatMessage};
use vxg_tensor::codebook::CodebookMatrix;
use vxg_tensor::format::{write_vxt_model, PackedTensor, VxtEntry};
use vxg_tensor::fp16::f32_to_bf16;
use vxg_tensor::gguf_meta::{GgufVal, ModelConfig};
use vxg_tensor::hf_pack::pack_hf_model_to_vxt;
use vxg_tensor::llm::{compare_logits, KvCacheMode, LlmModel};
use vxg_tensor::llm_pack::LlmPackMode;
use vxg_tensor::lossless::LosslessMatrix;
use vxg_tensor::metal::MetalEngine;
use vxg_tensor::q4_0::Q4_0Matrix;
use vxg_tensor::safetensors::write_safetensors_u16;
use vxg_tensor::tokenizer::BpeTokenizer;
use vxg_tensor::vision::VisionEncoder;
use vxg_tensor::vision_img::{PreprocessedImage, VisionConfig, DEFAULT_MAX_PIXELS};
use vxg_tensor::wht_turbo::TurboWhtMatrix;
use vxg_tensor::DType;

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..n)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = ((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / ((1u32 << 24) as f32);
            let v = (u * 2.0 - 1.0) * scale;
            if i % 113 == 5 { v * 3.5 } else { v }
        })
        .collect()
}

fn det_bf16(seed: u64, n: usize, scale: f32) -> Vec<u16> {
    det_f32(seed, n, scale).into_iter().map(f32_to_bf16).collect()
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (xf, yf) = (x as f64, y as f64);
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

fn build_mini_tokenizer_json(n_vocab: usize, added: &[(&str, usize)]) -> Vec<u8> {
    let mut vocab = serde_json::Map::new();
    for i in 0..n_vocab {
        vocab.insert(format!("t{i}"), json!(i));
    }
    let added_arr: Vec<_> = added
        .iter()
        .map(|&(tok, id)| {
            vocab.insert(tok.to_owned(), json!(id));
            json!({
                "id": id,
                "content": tok,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            })
        })
        .collect();
    serde_json::to_vec(&json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": added_arr
    }))
    .unwrap()
}

fn write_synthetic_hf_dir(dir: &Path, model_type: &str) {
    fs::create_dir_all(dir).unwrap();
    let (n_layer, n_embd, n_ff, n_vocab, n_head, n_kv, hd) = (2usize, 256, 512, 256, 4, 2, 64);
    let is_gemma = model_type == "gemma3";
    let cfg = json!({
        "model_type": model_type,
        "num_hidden_layers": n_layer,
        "hidden_size": n_embd,
        "intermediate_size": n_ff,
        "vocab_size": n_vocab,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv,
        "head_dim": hd,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": if is_gemma { "gelu_pytorch_tanh" } else { "silu" },
        "tie_word_embeddings": is_gemma,
        "eos_token_id": 2,
        "bos_token_id": 1
    });
    fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let sp: &[(&str, usize)] = match model_type {
        "llama" => &[("<|begin_of_text|>", 250), ("<|start_header_id|>", 251), ("<|end_header_id|>", 252), ("<|eot_id|>", 253)],
        "mistral" => &[("<s>", 1), ("</s>", 2), ("[INST]", 250), ("[/INST]", 251)],
        "gemma3" => &[("<bos>", 1), ("<eos>", 2), ("<start_of_turn>", 250), ("<end_of_turn>", 251)],
        _ => &[("<|im_start|>", 250), ("<|im_end|>", 251), ("<think>", 252), ("</think>", 253)],
    };
    fs::write(dir.join("tokenizer.json"), build_mini_tokenizer_json(n_vocab, sp)).unwrap();

    let norm_val = if is_gemma { 0.05f32 } else { 1.05f32 };
    let mk_norm = |seed: u64, len: usize| -> Vec<u16> {
        det_f32(seed, len, 0.15).into_iter().map(|d| f32_to_bf16(norm_val + d)).collect()
    };

    let mut owned: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
    owned.push(("model.embed_tokens.weight".into(), vec![n_vocab, n_embd], det_bf16(10, n_vocab * n_embd, 0.04)));
    owned.push(("model.norm.weight".into(), vec![n_embd], mk_norm(11, n_embd)));
    if !is_gemma {
        owned.push(("lm_head.weight".into(), vec![n_vocab, n_embd], det_bf16(12, n_vocab * n_embd, 0.04)));
    }
    for il in 0..n_layer {
        let s0 = 100 + (il as u64) * 30;
        let p = format!("model.layers.{il}");
        owned.push((format!("{p}.input_layernorm.weight"), vec![n_embd], mk_norm(s0, n_embd)));
        owned.push((format!("{p}.post_attention_layernorm.weight"), vec![n_embd], mk_norm(s0 + 1, n_embd)));
        if is_gemma {
            owned.push((format!("{p}.pre_feedforward_layernorm.weight"), vec![n_embd], mk_norm(s0 + 2, n_embd)));
            owned.push((format!("{p}.post_feedforward_layernorm.weight"), vec![n_embd], mk_norm(s0 + 3, n_embd)));
        }
        owned.push((format!("{p}.self_attn.q_proj.weight"), vec![n_head * hd, n_embd], det_bf16(s0 + 4, n_head * hd * n_embd, 0.05)));
        owned.push((format!("{p}.self_attn.k_proj.weight"), vec![n_kv * hd, n_embd], det_bf16(s0 + 5, n_kv * hd * n_embd, 0.05)));
        owned.push((format!("{p}.self_attn.v_proj.weight"), vec![n_kv * hd, n_embd], det_bf16(s0 + 6, n_kv * hd * n_embd, 0.05)));
        owned.push((format!("{p}.self_attn.o_proj.weight"), vec![n_embd, n_head * hd], det_bf16(s0 + 7, n_embd * n_head * hd, 0.05)));
        if model_type == "qwen3" || is_gemma {
            owned.push((format!("{p}.self_attn.q_norm.weight"), vec![hd], mk_norm(s0 + 8, hd)));
            owned.push((format!("{p}.self_attn.k_norm.weight"), vec![hd], mk_norm(s0 + 9, hd)));
        }
        owned.push((format!("{p}.mlp.gate_proj.weight"), vec![n_ff, n_embd], det_bf16(s0 + 10, n_ff * n_embd, 0.05)));
        owned.push((format!("{p}.mlp.up_proj.weight"), vec![n_ff, n_embd], det_bf16(s0 + 11, n_ff * n_embd, 0.05)));
        owned.push((format!("{p}.mlp.down_proj.weight"), vec![n_embd, n_ff], det_bf16(s0 + 12, n_embd * n_ff, 0.05)));
    }
    let refs: Vec<(&str, DType, &[usize], &[u16])> = owned
        .iter()
        .map(|(n, sh, w)| (n.as_str(), DType::BF16, sh.as_slice(), w.as_slice()))
        .collect();
    fs::write(dir.join("model.safetensors"), write_safetensors_u16(&refs)).unwrap();
}

#[test]
fn step3_hf_families_llama_mistral_qwen3_gemma3_and_chat_templates() {
    let tmp = std::env::temp_dir().join(format!("vxt_step3_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let gpu_ok = MetalEngine::new().is_ok();

    for family in ["llama", "mistral", "qwen3", "gemma3"] {
        let mdir = tmp.join(family);
        write_synthetic_hf_dir(&mdir, family);
        let (cfg, _raw_bytes, vxt_bytes) = pack_hf_model_to_vxt(&mdir, LlmPackMode::Cb4, 2).unwrap();
        if family == "gemma3" {
            assert!(cfg.ffn_gelu, "gemma3 must enable GeGLU");
            assert!((cfg.embed_scale - 16.0).abs() < 1e-5, "gemma3 embed_scale must equal sqrt(256)=16");
        } else {
            assert!(!cfg.ffn_gelu);
            assert!((cfg.embed_scale - 1.0).abs() < 1e-6);
        }

        let mut cpu_m = LlmModel::from_vxt_bytes(&vxt_bytes, false, 2).unwrap();
        let msgs = vec![
            ChatMessage::text("system", "hi"),
            ChatMessage::text("user", "hello"),
        ];
        let chat_ids = format_chat_messages(&cpu_m.tokenizer, &cpu_m.cfg.arch, &msgs, false);
        match family {
            "llama" => assert!(chat_ids.contains(&251) && chat_ids.contains(&253)),
            "mistral" => assert!(chat_ids.contains(&250) && chat_ids.contains(&251)),
            "gemma3" => assert!(chat_ids.contains(&250) && chat_ids.contains(&251)),
            "qwen3" => assert!(chat_ids.contains(&250) && chat_ids.contains(&251)),
            _ => unreachable!(),
        }

        if gpu_ok {
            let mut gpu_m = LlmModel::from_vxt_bytes(&vxt_bytes, true, 2).unwrap();
            assert!(gpu_m.has_cb4_fast_decode(), "{family} must use 1-CmdBuf CB4-Fast runner");
            let prompt = [3u32, 9, 17, 25, 42, 63];
            let mut st_c = cpu_m.new_state();
            let mut st_g = gpu_m.new_state();
            let mut lc = vec![0.0f32; cfg.n_vocab];
            let mut lg = vec![0.0f32; cfg.n_vocab];
            cpu_m.prefill(&mut st_c, &prompt, &mut lc).unwrap();
            gpu_m.prefill(&mut st_g, &prompt, &mut lg).unwrap();
            let m0 = compare_logits(&lc, &lg);
            assert!(m0.cosine_sim > 0.99999 && m0.top1_match, "{family} prefill parity: {m0:?}");

            cpu_m.forward_step(&mut st_c, 77, Some(&mut lc)).unwrap();
            gpu_m.forward_step(&mut st_g, 77, Some(&mut lg)).unwrap();
            let m1 = compare_logits(&lc, &lg);
            assert!(m1.cosine_sim > 0.99999 && m1.top1_match, "{family} decode parity: {m1:?}");
        }
    }
    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn step4_tiled_gemm_prefill_and_q8_kv_cache_cpu_and_metal() {
    let (rows, cols, batch_m) = (96usize, 256usize, 11usize);
    let w = det_f32(901, rows * cols, 0.08);
    let xb = det_f32(902, batch_m * cols, 0.5);
    let variants = [
        PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, &w).unwrap()),
        PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &w).unwrap()),
        PackedTensor::TurboWht(TurboWhtMatrix::encode(3, rows, cols, &w).unwrap()),
        PackedTensor::Lossless(LosslessMatrix::encode(DType::BF16, rows, cols, &det_bf16(901, rows * cols, 0.08)).unwrap()),
        PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &w)),
    ];
    for (vi, pt) in variants.iter().enumerate() {
        let mut ref_y = vec![0.0f32; batch_m * rows];
        for m in 0..batch_m {
            pt.gemv_cpu(&xb[m * cols..(m + 1) * cols], &mut ref_y[m * rows..(m + 1) * rows], 2);
        }
        let mut got_y = vec![0.0f32; batch_m * rows];
        pt.gemm_cpu(&xb, batch_m, &mut got_y, 2);
        assert!(cosine_sim(&ref_y, &got_y) > 0.999_999, "variant {vi} gemm_cpu mismatch");
    }

    let tmp = std::env::temp_dir().join(format!("vxt_step4_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    write_synthetic_hf_dir(&tmp, "qwen3");
    let (cfg, _, vxt_bytes) = pack_hf_model_to_vxt(&tmp, LlmPackMode::Cb4, 2).unwrap();
    let _ = fs::remove_dir_all(&tmp);

    let prompt: Vec<u32> = (0..13).map(|i| (i * 7 + 3) as u32).collect();
    let mut cpu_m = LlmModel::from_vxt_bytes(&vxt_bytes, false, 2).unwrap();
    let mut st_step = cpu_m.new_state();
    let mut st_tile = cpu_m.new_state();
    let mut st_q8 = cpu_m.new_state_with_kv(KvCacheMode::Q8_0);
    assert!(
        cpu_m.kv_cache_bytes(&st_tile) > cpu_m.kv_cache_bytes(&st_q8) * 35 / 10,
        "CPU Q8_0 KV-cache must shrink >= 3.5x ({} vs {})",
        cpu_m.kv_cache_bytes(&st_tile),
        cpu_m.kv_cache_bytes(&st_q8)
    );

    let (mut l_step, mut l_tile, mut l_q8) = (
        vec![0.0f32; cfg.n_vocab],
        vec![0.0f32; cfg.n_vocab],
        vec![0.0f32; cfg.n_vocab],
    );
    for (idx, &t) in prompt.iter().enumerate() {
        let last = idx + 1 == prompt.len();
        cpu_m.forward_step(&mut st_step, t, last.then_some(&mut l_step)).unwrap();
    }
    cpu_m.prefill(&mut st_tile, &prompt, &mut l_tile).unwrap();
    cpu_m.prefill(&mut st_q8, &prompt, &mut l_q8).unwrap();
    let mt = compare_logits(&l_step, &l_tile);
    assert!(mt.cosine_sim > 0.999_999 && mt.top1_match, "CPU tiled prefill mismatch: {mt:?}");
    let mq_cpu = compare_logits(&l_tile, &l_q8);
    assert!(mq_cpu.cosine_sim > 0.999 && mq_cpu.top1_match, "CPU Q8_0 prefill mismatch: {mq_cpu:?}");

    cpu_m.forward_step(&mut st_tile, 41, Some(&mut l_tile)).unwrap();
    cpu_m.forward_step(&mut st_q8, 41, Some(&mut l_q8)).unwrap();
    let mq_cpu_dec = compare_logits(&l_tile, &l_q8);
    assert!(mq_cpu_dec.cosine_sim > 0.999 && mq_cpu_dec.top1_match, "CPU Q8_0 decode mismatch: {mq_cpu_dec:?}");

    if MetalEngine::new().is_ok() {
        let mut gpu_m = LlmModel::from_vxt_bytes(&vxt_bytes, true, 2).unwrap();
        let mut gst_f32 = gpu_m.new_state_with_kv(KvCacheMode::Fp32);
        let mut gst_q8 = gpu_m.new_state_with_kv(KvCacheMode::Q8_0);
        let (b_f32, b_q8) = (gpu_m.kv_cache_bytes(&gst_f32), gpu_m.kv_cache_bytes(&gst_q8));
        assert!(b_f32 > b_q8 * 35 / 10, "Metal Q8_0 KV-cache must shrink >= 3.5x ({b_f32} vs {b_q8})");

        let (mut gf, mut gq) = (vec![0.0f32; cfg.n_vocab], vec![0.0f32; cfg.n_vocab]);
        gpu_m.prefill(&mut gst_f32, &prompt, &mut gf).unwrap();
        gpu_m.prefill(&mut gst_q8, &prompt, &mut gq).unwrap();
        let mp = compare_logits(&gf, &gq);
        assert!(mp.cosine_sim > 0.999 && mp.top1_match, "Metal Q8_0 prefill mismatch: {mp:?}");

        gpu_m.forward_step(&mut gst_f32, 41, Some(&mut gf)).unwrap();
        gpu_m.forward_step(&mut gst_q8, 41, Some(&mut gq)).unwrap();
        let md = compare_logits(&gf, &gq);
        assert!(md.cosine_sim > 0.999 && md.top1_match, "Metal Q8_0 decode mismatch: {md:?}");

        let cross = compare_logits(&l_q8, &gq);
        assert!(cross.cosine_sim > 0.9999 && cross.top1_match, "CPU Q8_0 vs Metal Q8_0: {cross:?}");
    }
}

#[test]
fn step2_vision_1cmdbuf_metal_gpu_and_multimodal_mrope_parity() {
    let vcfg = VisionConfig {
        depth: 2,
        hidden_size: 128,
        intermediate_size: 256,
        num_heads: 4,
        head_dim: 32,
        in_channels: 3,
        patch_size: 4,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 256,
        num_position_embeddings: 64,
        rope_theta: 10000.0,
        mrope_section: [6, 5, 5],
        vision_start_token_id: 250,
        vision_end_token_id: 251,
        image_token_id: 252,
        min_pixels: 64,
        max_pixels: DEFAULT_MAX_PIXELS,
    };
    let d = vcfg.hidden_size;
    let pdim = vcfg.patch_dim();
    let mdim = d * vcfg.merge_unit();
    let pack_cb4 = |r: usize, c: usize, seed: u64| -> PackedTensor {
        PackedTensor::Codebook(CodebookMatrix::encode(4, r, c, &det_f32(seed, r * c, 0.06)).unwrap())
    };
    let raw_vec = |n: usize, seed: u64, bias: f32| -> PackedTensor {
        let data: Vec<f32> = det_f32(seed, n, 0.05).into_iter().map(|v| v + bias).collect();
        PackedTensor::RawF32 { rows: 1, cols: n, data }
    };

    let mut vis_entries = vec![
        ("vis.patch_embed.proj.weight".to_owned(), PackedTensor::RawF32 { rows: d, cols: pdim, data: det_f32(1, d * pdim, 0.04) }),
        ("vis.patch_embed.proj.bias".to_owned(), raw_vec(d, 2, 0.0)),
        ("vis.pos_embed.weight".to_owned(), PackedTensor::RawF32 { rows: 64, cols: d, data: det_f32(3, 64 * d, 0.02) }),
        ("vis.merger.norm.weight".to_owned(), raw_vec(d, 4, 1.0)),
        ("vis.merger.norm.bias".to_owned(), raw_vec(d, 5, 0.0)),
        ("vis.merger.linear_fc1.weight".to_owned(), pack_cb4(mdim, mdim, 6)),
        ("vis.merger.linear_fc1.bias".to_owned(), raw_vec(mdim, 7, 0.0)),
        ("vis.merger.linear_fc2.weight".to_owned(), pack_cb4(vcfg.out_hidden_size, mdim, 8)),
        ("vis.merger.linear_fc2.bias".to_owned(), raw_vec(vcfg.out_hidden_size, 9, 0.0)),
    ];
    for il in 0..vcfg.depth {
        let s = 50 + (il as u64) * 20;
        let p = format!("vis.blocks.{il}");
        vis_entries.extend([
            (format!("{p}.norm1.weight"), raw_vec(d, s, 1.0)),
            (format!("{p}.norm1.bias"), raw_vec(d, s + 1, 0.0)),
            (format!("{p}.attn.qkv.weight"), pack_cb4(3 * d, d, s + 2)),
            (format!("{p}.attn.qkv.bias"), raw_vec(3 * d, s + 3, 0.0)),
            (format!("{p}.attn.proj.weight"), pack_cb4(d, d, s + 4)),
            (format!("{p}.attn.proj.bias"), raw_vec(d, s + 5, 0.0)),
            (format!("{p}.norm2.weight"), raw_vec(d, s + 6, 1.0)),
            (format!("{p}.norm2.bias"), raw_vec(d, s + 7, 0.0)),
            (format!("{p}.mlp.linear_fc1.weight"), pack_cb4(vcfg.intermediate_size, d, s + 8)),
            (format!("{p}.mlp.linear_fc1.bias"), raw_vec(vcfg.intermediate_size, s + 9, 0.0)),
            (format!("{p}.mlp.linear_fc2.weight"), pack_cb4(d, vcfg.intermediate_size, s + 10)),
            (format!("{p}.mlp.linear_fc2.bias"), raw_vec(d, s + 11, 0.0)),
        ]);
    }

    let mut map_cpu: HashMap<String, PackedTensor> = vis_entries.iter().cloned().collect();
    let vis_cpu = VisionEncoder::from_tensors(vcfg.clone(), &mut map_cpu).unwrap();
    let pre = PreprocessedImage {
        patches: det_f32(777, 16 * pdim, 0.35),
        grid_t: 1,
        grid_h: 4,
        grid_w: 4,
        resized_h: 16,
        resized_w: 16,
    };
    let enc_cpu = vis_cpu.encode_preprocessed(&pre, 2).unwrap();
    assert_eq!(enc_cpu.llm_tokens, 4);

    if MetalEngine::new().is_err() {
        return;
    }

    // Build full VLM `.vxt` archive (`CB4` language trunk + `VisionEncoder`) and verify Metal GPU ViT + M-RoPE
    let lcfg = ModelConfig {
        arch: "qwen35".into(),
        model_name: "synth-vlm".into(),
        n_layer: 1,
        n_embd: 256,
        n_ff: 512,
        n_vocab: 256,
        n_head: 4,
        n_head_kv: 2,
        head_dim: 64,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10000.0,
        rope_dim: 32,
        rope_neox: true,
        ffn_gelu: false,
        embed_scale: 1.0,
        full_attn_interval: 1,
        ssm_conv_kernel: 0,
        ssm_state_size: 0,
        ssm_group_count: 0,
        ssm_dt_rank: 0,
        ssm_inner_size: 0,
        eos_token_id: 2,
        bos_token_id: Some(1),
    };
    let mut kv = HashMap::new();
    let toks: Vec<String> = (0..256).map(|i| format!("tok{i}")).collect();
    kv.insert("tokenizer.ggml.tokens".into(), GgufVal::ArrStr(toks));
    kv.insert("tokenizer.ggml.merges".into(), GgufVal::ArrStr(Vec::new()));
    let mut meta = vec![
        ("model.config".to_owned(), lcfg.to_kv_bytes()),
        ("vision.config".to_owned(), vcfg.to_json_bytes()),
    ];
    meta.extend(BpeTokenizer::to_vxt_meta_entries(&kv).unwrap());

    let mut all_entries: Vec<VxtEntry> = vis_entries
        .into_iter()
        .map(|(name, tensor)| VxtEntry { name, tensor })
        .collect();
    all_entries.extend([
        VxtEntry { name: "token_embd.weight".into(), tensor: pack_cb4(256, 256, 300) },
        VxtEntry { name: "output_norm.weight".into(), tensor: raw_vec(256, 301, 1.0) },
        VxtEntry { name: "blk.0.attn_norm.weight".into(), tensor: raw_vec(256, 302, 1.0) },
        VxtEntry { name: "blk.0.post_attention_norm.weight".into(), tensor: raw_vec(256, 303, 1.0) },
        VxtEntry { name: "blk.0.attn_q.weight".into(), tensor: pack_cb4(512, 256, 304) },
        VxtEntry { name: "blk.0.attn_k.weight".into(), tensor: pack_cb4(128, 256, 305) },
        VxtEntry { name: "blk.0.attn_v.weight".into(), tensor: pack_cb4(128, 256, 306) },
        VxtEntry { name: "blk.0.attn_output.weight".into(), tensor: pack_cb4(256, 256, 307) },
        VxtEntry { name: "blk.0.attn_q_norm.weight".into(), tensor: raw_vec(64, 308, 1.0) },
        VxtEntry { name: "blk.0.attn_k_norm.weight".into(), tensor: raw_vec(64, 309, 1.0) },
        VxtEntry { name: "blk.0.ffn_gate.weight".into(), tensor: pack_cb4(512, 256, 310) },
        VxtEntry { name: "blk.0.ffn_up.weight".into(), tensor: pack_cb4(512, 256, 311) },
        VxtEntry { name: "blk.0.ffn_down.weight".into(), tensor: pack_cb4(256, 512, 312) },
    ]);
    let vxt_bytes = write_vxt_model(&meta, &all_entries).unwrap();

    let mut vlm_cpu = LlmModel::from_vxt_bytes(&vxt_bytes, false, 2).unwrap();
    let mut vlm_gpu = LlmModel::from_vxt_bytes(&vxt_bytes, true, 2).unwrap();
    assert!(
        vlm_gpu.backend_label().contains("Vision (Metal GPU)"),
        "expected Metal GPU vision label, got: {}",
        vlm_gpu.backend_label()
    );

    // Encode synthetic tiny PNG data URI through both CPU and 1-CmdBuf Metal GPU VisionEncoder
    let mut img_buf = image::RgbImage::new(32, 32);
    for (x, y, px) in img_buf.enumerate_pixels_mut() {
        *px = image::Rgb([(x * 7) as u8, (y * 7) as u8, ((x + y) * 3) as u8]);
    }
    let png_path = std::env::temp_dir().join(format!("vxt_vis_test_{}.png", std::process::id()));
    img_buf.save(&png_path).unwrap();
    let png_str = png_path.to_str().unwrap();
    let img_c = vlm_cpu.encode_image_source(png_str).unwrap();
    let img_g = vlm_gpu.encode_image_source(png_str).unwrap();
    let _ = fs::remove_file(&png_path);

    let vis_cos = cosine_sim(&img_c.embeds, &img_g.embeds);
    assert!(vis_cos > 0.9999, "1-CmdBuf Metal Vision vs CPU cosine_sim={vis_cos}");

    let mut mm_toks = vec![10u32, 250];
    mm_toks.resize(mm_toks.len() + img_g.llm_tokens, 252);
    mm_toks.extend([251, 33, 44]);

    let (mut st_c, mut st_g) = (vlm_cpu.new_state(), vlm_gpu.new_state());
    let (mut lc, mut lg) = (vec![0.0f32; 256], vec![0.0f32; 256]);
    vlm_cpu.prefill_multimodal(&mut st_c, &mm_toks, &[&img_c], &mut lc).unwrap();
    vlm_gpu.prefill_multimodal(&mut st_g, &mm_toks, &[&img_g], &mut lg).unwrap();
    let mm_cmp = compare_logits(&lc, &lg);
    assert!(mm_cmp.cosine_sim > 0.9999 && mm_cmp.top1_match, "Multimodal 3D M-RoPE parity: {mm_cmp:?}");
}
