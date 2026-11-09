use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use image::{ImageBuffer, ImageFormat, Rgb};
use serde_json::json;
use vxg_tensor::bench::generate_realistic_llm_weights;
use vxg_tensor::chat::{ChatMessage, ChatParams, ChatSession};
use vxg_tensor::hf_pack::pack_hf_model_to_vxt;
use vxg_tensor::llm::{compare_logits, LlmModel};
use vxg_tensor::llm_pack::LlmPackMode;
use vxg_tensor::safetensors::write_safetensors_u16;
use vxg_tensor::server::EmbeddedServer;
use vxg_tensor::DType;

fn make_test_png_data_uri(w: u32, h: u32) -> String {
    let img = ImageBuffer::from_fn(w, h, |x, y| {
        if x < w / 2 {
            Rgb([220u8, (y * 3) as u8, 40u8])
        } else {
            Rgb([30u8, 140u8, (x * 2) as u8])
        }
    });
    let mut png_bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png_bytes), ImageFormat::Png)
        .unwrap();
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut b64 = String::from("data:image/png;base64,");
    for chunk in png_bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        b64.push(B64[((n >> 18) & 63) as usize] as char);
        b64.push(B64[((n >> 12) & 63) as usize] as char);
        b64.push(if chunk.len() > 1 { B64[((n >> 6) & 63) as usize] as char } else { '=' });
        b64.push(if chunk.len() > 2 { B64[(n & 63) as usize] as char } else { '=' });
    }
    b64
}

fn build_vlm_9b_style_fixture(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).unwrap();

    // Miniature Qwen3.5-9B-style VLM:
    // - untied lm_head.weight (tie_word_embeddings = false)
    // - asymmetric DeltaNet heads (n_vh = 4, n_kh = 2 -> group ratio 2:1, matching 9B's 32:16!)
    // - full Qwen3_5VisionModel (5D Conv3d patch_embed, pos_embed, 2 ViT blocks, 2x2 merger)
    // - 3D M-RoPE (rope_dim = 16, mrope_section = [3, 3, 2])
    let cfg = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "vision_start_token_id": 261,
        "vision_end_token_id": 262,
        "image_token_id": 263,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 264,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 2,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "eos_token_id": 256,
            "tie_word_embeddings": false,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [3, 3, 2]
            }
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 256,
            "num_position_embeddings": 64
        }
    });
    fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();

    let mut vocab = serde_json::Map::new();
    for b in 0u32..256 {
        vocab.insert(format!("t{b}"), json!(b));
    }
    let tok = json!({
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "added_tokens": [
            {"id": 256, "content": "<|endoftext|>", "special": true},
            {"id": 257, "content": "<|im_start|>", "special": true},
            {"id": 258, "content": "<|im_end|>", "special": true},
            {"id": 259, "content": "<think>", "special": false},
            {"id": 260, "content": "</think>", "special": false},
            {"id": 261, "content": "<|vision_start|>", "special": true},
            {"id": 262, "content": "<|vision_end|>", "special": true},
            {"id": 263, "content": "<|image_pad|>", "special": true}
        ]
    });
    fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok).unwrap()).unwrap();

    let mk = |r: usize, c: usize, seed: u64| -> Vec<u16> {
        let (_, f32s) = generate_realistic_llm_weights(DType::BF16, r.max(16), c.max(256), seed);
        f32s.into_iter()
            .take(r * c)
            .map(|v| vxg_tensor::fp16::f32_to_bf16(v * 0.25))
            .collect()
    };
    let z16 = |n: usize| vec![0u16; n];
    let o16 = |n: usize| vec![0x3F80u16; n];

    let mut items: Vec<(String, Vec<usize>, Vec<u16>)> = vec![
        ("model.language_model.embed_tokens.weight".into(), vec![264, 256], mk(264, 256, 1)),
        ("lm_head.weight".into(), vec![264, 256], mk(264, 256, 2)),
        ("model.language_model.norm.weight".into(), vec![256], z16(256)),
        // Layer 0: asymmetric DeltaNet (qkv = (2*2*64) + (4*64) = 512)
        ("model.language_model.layers.0.input_layernorm.weight".into(), vec![256], z16(256)),
        ("model.language_model.layers.0.post_attention_layernorm.weight".into(), vec![256], z16(256)),
        ("model.language_model.layers.0.mlp.gate_proj.weight".into(), vec![512, 256], mk(512, 256, 10)),
        ("model.language_model.layers.0.mlp.up_proj.weight".into(), vec![512, 256], mk(512, 256, 11)),
        ("model.language_model.layers.0.mlp.down_proj.weight".into(), vec![256, 512], mk(256, 512, 12)),
        ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight".into(), vec![512, 256], mk(512, 256, 13)),
        ("model.language_model.layers.0.linear_attn.in_proj_z.weight".into(), vec![256, 256], mk(256, 256, 14)),
        ("model.language_model.layers.0.linear_attn.in_proj_a.weight".into(), vec![4, 256], mk(4, 256, 15)),
        ("model.language_model.layers.0.linear_attn.in_proj_b.weight".into(), vec![4, 256], mk(4, 256, 16)),
        ("model.language_model.layers.0.linear_attn.out_proj.weight".into(), vec![256, 256], mk(256, 256, 17)),
        ("model.language_model.layers.0.linear_attn.conv1d.weight".into(), vec![512, 1, 4], mk(512, 4, 18)),
        ("model.language_model.layers.0.linear_attn.norm.weight".into(), vec![64], o16(64)),
        ("model.language_model.layers.0.linear_attn.A_log".into(), vec![4], o16(4)),
        ("model.language_model.layers.0.linear_attn.dt_bias".into(), vec![4], z16(4)),
        // Layer 1: Full Gated-GQA with 3D M-RoPE
        ("model.language_model.layers.1.input_layernorm.weight".into(), vec![256], z16(256)),
        ("model.language_model.layers.1.post_attention_layernorm.weight".into(), vec![256], z16(256)),
        ("model.language_model.layers.1.mlp.gate_proj.weight".into(), vec![512, 256], mk(512, 256, 20)),
        ("model.language_model.layers.1.mlp.up_proj.weight".into(), vec![512, 256], mk(512, 256, 21)),
        ("model.language_model.layers.1.mlp.down_proj.weight".into(), vec![256, 512], mk(256, 512, 22)),
        ("model.language_model.layers.1.self_attn.q_proj.weight".into(), vec![512, 256], mk(512, 256, 23)),
        ("model.language_model.layers.1.self_attn.k_proj.weight".into(), vec![128, 256], mk(128, 256, 24)),
        ("model.language_model.layers.1.self_attn.v_proj.weight".into(), vec![128, 256], mk(128, 256, 25)),
        ("model.language_model.layers.1.self_attn.o_proj.weight".into(), vec![256, 256], mk(256, 256, 26)),
        ("model.language_model.layers.1.self_attn.q_norm.weight".into(), vec![64], z16(64)),
        ("model.language_model.layers.1.self_attn.k_norm.weight".into(), vec![64], z16(64)),
        // Vision encoder (including 5D Conv3d patch_embed [64, 3, 2, 16, 16] -> [64, 1536])
        ("model.visual.patch_embed.proj.weight".into(), vec![64, 3, 2, 16, 16], mk(64, 1536, 30)),
        ("model.visual.patch_embed.proj.bias".into(), vec![64], z16(64)),
        ("model.visual.pos_embed.weight".into(), vec![64, 64], mk(64, 64, 31)),
        ("model.visual.merger.norm.weight".into(), vec![64], o16(64)),
        ("model.visual.merger.norm.bias".into(), vec![64], z16(64)),
        ("model.visual.merger.linear_fc1.weight".into(), vec![256, 256], mk(256, 256, 32)),
        ("model.visual.merger.linear_fc1.bias".into(), vec![256], z16(256)),
        ("model.visual.merger.linear_fc2.weight".into(), vec![256, 256], mk(256, 256, 33)),
        ("model.visual.merger.linear_fc2.bias".into(), vec![256], z16(256)),
    ];
    for b in 0..2 {
        let p = format!("model.visual.blocks.{b}");
        let s = 40 + (b as u64) * 10;
        items.push((format!("{p}.norm1.weight"), vec![64], o16(64)));
        items.push((format!("{p}.norm1.bias"), vec![64], z16(64)));
        items.push((format!("{p}.attn.qkv.weight"), vec![192, 64], mk(192, 64, s + 1)));
        items.push((format!("{p}.attn.qkv.bias"), vec![192], z16(192)));
        items.push((format!("{p}.attn.proj.weight"), vec![64, 64], mk(64, 64, s + 2)));
        items.push((format!("{p}.attn.proj.bias"), vec![64], z16(64)));
        items.push((format!("{p}.norm2.weight"), vec![64], o16(64)));
        items.push((format!("{p}.norm2.bias"), vec![64], z16(64)));
        items.push((format!("{p}.mlp.linear_fc1.weight"), vec![128, 64], mk(128, 64, s + 3)));
        items.push((format!("{p}.mlp.linear_fc1.bias"), vec![128], z16(128)));
        items.push((format!("{p}.mlp.linear_fc2.weight"), vec![64, 128], mk(64, 128, s + 4)));
        items.push((format!("{p}.mlp.linear_fc2.bias"), vec![64], z16(64)));
    }

    let refs: Vec<(&str, DType, &[usize], &[u16])> = items
        .iter()
        .map(|(n, sh, w)| (n.as_str(), DType::BF16, sh.as_slice(), w.as_slice()))
        .collect();
    fs::write(dir.join("model.safetensors"), write_safetensors_u16(&refs)).unwrap();
}

#[test]
fn vlm_mrope_cpu_metal_parity_and_openai_vision_caching() {
    let dir = std::env::temp_dir().join("vxt_vlm_mrope_test_fixture");
    build_vlm_9b_style_fixture(&dir);
    let data_uri = make_test_png_data_uri(96, 64);

    let (cfg, _, vxt_cb4) = pack_hf_model_to_vxt(&dir, LlmPackMode::Cb4, 2).unwrap();
    let vxt_path = dir.join("vlm_cb4.vxt");
    fs::write(&vxt_path, &vxt_cb4).unwrap();

    let mut m_cpu = LlmModel::from_vxt_bytes(&vxt_cb4, false, 2).unwrap();
    let mut m_gpu = LlmModel::from_vxt_bytes(&vxt_cb4, true, 2).unwrap();
    assert!(m_cpu.has_vision() && m_gpu.has_vision());
    if vxg_tensor::metal::MetalEngine::new().is_ok() {
        assert!(m_gpu.backend_label().contains("CB4-Fast"));
        assert!(m_gpu.backend_label().contains("Vision"));
    }

    let enc = m_cpu.encode_image_source(&data_uri).unwrap();
    assert_eq!((enc.grid_h, enc.grid_w, enc.llm_tokens), (4, 6, 6));
    let mut prompt = vec![257u32, 10, 261];
    prompt.resize(prompt.len() + enc.llm_tokens, 263);
    prompt.extend_from_slice(&[262, 11, 12, 258, 257]);

    let (mut st_cpu, mut st_gpu) = (m_cpu.new_state(), m_gpu.new_state());
    let (mut log_cpu, mut log_gpu) = (vec![0.0f32; cfg.n_vocab], vec![0.0f32; cfg.n_vocab]);
    m_cpu.prefill_multimodal(&mut st_cpu, &prompt, &[&enc], &mut log_cpu).unwrap();
    m_gpu.prefill_multimodal(&mut st_gpu, &prompt, &[&enc], &mut log_gpu).unwrap();
    assert_eq!(st_cpu.rope_delta, -3);
    assert_eq!(st_gpu.rope_delta, -3);
    let pre_cmp = compare_logits(&log_cpu, &log_gpu);
    assert!(pre_cmp.cosine_sim > 0.99999, "prefill cos_sim={}", pre_cmp.cosine_sim);

    // Verify post-prefill text decode step preserves exact rope_delta on both CPU & Metal CB4-Fast!
    m_cpu.forward_step(&mut st_cpu, 42, Some(&mut log_cpu)).unwrap();
    m_gpu.forward_step(&mut st_gpu, 42, Some(&mut log_gpu)).unwrap();
    let dec_cmp = compare_logits(&log_cpu, &log_gpu);
    assert!(dec_cmp.cosine_sim > 0.99999, "decode cos_sim={}", dec_cmp.cosine_sim);

    // Verify ChatSession multi-turn Vision -> text follow-up hits incremental prefix cache!
    let mut sess = ChatSession::load(vxt_path.to_str().unwrap(), true, 2).unwrap();
    assert!(sess.summary.vision);
    let cancel = AtomicBool::new(false);
    let params = ChatParams { max_tokens: 4, temperature: 0.0, ..ChatParams::default() };
    let mut hist = vec![ChatMessage {
        role: "user".into(),
        content: "t72".into(),
        images: vec![data_uri.clone()],
    }];
    let (r1, m1) = sess.generate_chat(&hist, &params, &cancel, |_| true).unwrap();
    assert_eq!(m1.cached_tokens, 0);
    hist.push(ChatMessage::text("assistant", r1));
    hist.push(ChatMessage::text("user", "t65"));
    let (_, m2) = sess.generate_chat(&hist, &params, &cancel, |_| true).unwrap();
    assert!(m2.cached_tokens > 0, "expected Vision Turn-2 prefix cache hit, got 0");

    // Verify OpenAI /v1/chat/completions multimodal content array (image_url + text)!
    let srv = EmbeddedServer::start("127.0.0.1", 0).unwrap();
    srv.handle().load_model(vxt_path.to_string_lossy().into_owned(), true, 2).unwrap();
    let body = serde_json::to_vec(&json!({
        "model": "vxt-local",
        "max_tokens": 3,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": "t72"}
            ]
        }]
    }))
    .unwrap();
    let mut conn = TcpStream::connect(("127.0.0.1", srv.handle().port())).unwrap();
    write!(
        conn,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .unwrap();
    conn.write_all(&body).unwrap();
    let mut resp = String::new();
    conn.read_to_string(&mut resp).unwrap();
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "resp={resp}");
    assert!(resp.contains("\"chat.completion\""));

    let _ = fs::remove_dir_all(&dir);
}
