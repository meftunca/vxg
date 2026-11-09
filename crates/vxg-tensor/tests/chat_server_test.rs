use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;

use serde_json::{json, Value};
use vxg_tensor::llm_ops::{strip_think_block, TokenPenaltyTracker};
use vxg_tensor::server::EmbeddedServer;
use vxg_tensor::tokenizer::BpeTokenizer;

fn raw_http_request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect EmbeddedServer");
    let payload = body.unwrap_or("");
    let req = if body.is_some() {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        )
    } else {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        )
    };
    stream.write_all(req.as_bytes()).expect("write request");
    let mut resp = String::new();
    stream.read_to_string(&mut resp).expect("read response");
    let (head, body_out) = resp.split_once("\r\n\r\n").expect("HTTP header separator");
    let status_line = head.lines().next().unwrap_or("");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    (code, head.to_owned(), body_out.to_owned())
}

#[test]
fn embedded_server_http_endpoints_and_cors() {
    let srv = EmbeddedServer::start("127.0.0.1", 0).expect("start ephemeral server");
    let port = srv.handle().port();
    assert!(port > 0);

    let (code_opt, head_opt, _) = raw_http_request(port, "OPTIONS", "/v1/chat/completions", None);
    assert_eq!(code_opt, 204);
    assert!(head_opt.to_ascii_lowercase().contains("access-control-allow-origin"));

    let (code_status, _, body_status) = raw_http_request(port, "GET", "/v1/vxt/status", None);
    assert_eq!(code_status, 200);
    let st: Value = serde_json::from_str(&body_status).expect("valid status json");
    assert_eq!(st["ok"], true);
    assert_eq!(st["port"], port);
    assert!(st["model"].is_null());

    let (code_models, _, body_models) = raw_http_request(port, "GET", "/v1/models", None);
    assert_eq!(code_models, 200);
    let md: Value = serde_json::from_str(&body_models).expect("valid models json");
    assert_eq!(md["object"], "list");

    let no_model_req = json!({
        "model": "vxt-local",
        "stream": false,
        "messages": [{"role": "user", "content": "Merhaba"}]
    })
    .to_string();
    let (code_err, _, body_err) =
        raw_http_request(port, "POST", "/v1/chat/completions", Some(&no_model_req));
    assert_eq!(code_err, 400);
    assert!(body_err.contains("No model loaded"));
}

#[test]
fn anti_loop_dry_tracker_and_think_history_stripping() {
    assert_eq!(
        strip_think_block("<think>\nstep 1\n</think>\n\nÖZET: Tamam."),
        "ÖZET: Tamam."
    );
    assert_eq!(strip_think_block("<think>\naborted mid-thought"), "");
    assert_eq!(strip_think_block("Düz yanıt."), "Düz yanıt.");

    let vocab = vec![
        "0".to_owned(),
        "4".to_owned(),
        "bellek".to_owned(),
        "bant".to_owned(),
        "genisligi".to_owned(),
        "darbogaz".to_owned(),
        "teshis".to_owned(),
        "adimi".to_owned(),
        "<|im_start|>".to_owned(),
        "<|im_end|>".to_owned(),
        "<think>".to_owned(),
        "</think>".to_owned(),
    ];
    let merges = vec!["b e".to_owned()];
    let tok = BpeTokenizer::from_vocab_and_merges(vocab.clone(), &merges, None).unwrap();
    assert_eq!(tok.encode("<think>"), vec![10]);
    assert_eq!(tok.encode("</think>"), vec![11]);

    let mut tracker = TokenPenaltyTracker::new(vocab.len(), &[8, 9, 10, 11]);
    // Record digits (`0`, `4`) and words (`2..=7`), then replay 5-token prefix `2,3,4,5,6` -> next was `7`
    for &id in &[0u32, 4, 0, 0, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6] {
        tracker.record(id, &tok);
    }
    let mut logits = vec![10.0f32; vocab.len()];
    tracker.apply(&mut logits, 0.5, 0.0);
    // Digit `0` must be completely exempt from presence_penalty (`10.0` untouched)
    assert!((logits[0] - 10.0).abs() < 1e-6);
    // Word `2` ("bellek") gets flat presence_penalty (`-0.5`)
    assert!((logits[2] - 9.5).abs() < 1e-6);
    // Token `7` ("adimi") gets BOTH word presence_penalty (`-0.5`) AND DRY 5-gram clone penalty (`-0.85`)
    assert!(logits[7] < 8.7, "expected DRY suffix-clone penalty on token 7, got {}", logits[7]);

    // Repeated single-digit numbers (vocab[0]="0", vocab[1]="4") must NEVER incur DRY suffix penalties
    let mut num_tracker = TokenPenaltyTracker::new(vocab.len(), &[8, 9, 10, 11]);
    for &id in &[2u32, 1, 0, 0, 0, 1, 0, 2, 1, 0, 0, 0, 1] {
        num_tracker.record(id, &tok);
    }
    let mut num_logits = vec![10.0f32; vocab.len()];
    num_tracker.apply(&mut num_logits, 0.5, 0.0);
    assert!((num_logits[0] - 10.0).abs() < 1e-6, "digit token '0' must never be penalized by DRY");
    assert!((num_logits[1] - 10.0).abs() < 1e-6, "digit token '4' must never be penalized by DRY");
}

#[test]
#[ignore]
fn real_qwen35_cb4_v2_openai_sse_and_prefix_cache() {
    let vxt_path =
        "/private/tmp/vxg-real-qualification-20261104/cb4-v2-20261108/Qwen3.5-2B-cb4-v2.vxt";
    if !Path::new(vxt_path).is_file() {
        eprintln!("Skipping real model test; {vxt_path} not present");
        return;
    }

    let srv = EmbeddedServer::start("127.0.0.1", 0).expect("start server");
    let port = srv.handle().port();
    let sum = srv
        .handle()
        .load_model(vxt_path.to_owned(), true, 4)
        .expect("load Qwen3.5-2B-cb4-v2.vxt");
    assert!(sum.cb4_fast);

    // Turn 1: OpenAI SSE streaming (`stream: true`)
    let turn1_body = json!({
        "model": sum.model_name,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": [
            {"role": "system", "content": "Kısa ve net cevap ver."},
            {"role": "user", "content": "Türkiye'nin başkenti neresidir?"}
        ]
    })
    .to_string();
    let (code1, head1, sse_body) =
        raw_http_request(port, "POST", "/v1/chat/completions", Some(&turn1_body));
    assert_eq!(code1, 200);
    assert!(head1.to_ascii_lowercase().contains("text/event-stream"));
    assert!(sse_body.contains("data: [DONE]"));

    let mut assembled = String::new();
    for line in sse_body.lines() {
        let Some(rest) = line.strip_prefix("data: ") else { continue };
        if rest == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(rest).expect("valid SSE JSON chunk");
        if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
            assembled.push_str(delta);
        }
    }
    assert!(
        assembled.contains("Ankara"),
        "expected 'Ankara' in streamed reply, got: {assembled}"
    );

    // Turn 2: Multi-turn follow-up (`stream: false`), verifying exact KV + DeltaNet prefix cache hit!
    let turn2_body = json!({
        "model": sum.model_name,
        "stream": false,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "Kısa ve net cevap ver."},
            {"role": "user", "content": "Türkiye'nin başkenti neresidir?"},
            {"role": "assistant", "content": assembled},
            {"role": "user", "content": "Peki bu şehrin hangi coğrafi bölgede yer aldığını ve önemini iki cümleyle açıkla."}
        ]
    })
    .to_string();
    let (code2, _, json_body2) =
        raw_http_request(port, "POST", "/v1/chat/completions", Some(&turn2_body));
    assert_eq!(code2, 200);
    let resp2: Value = serde_json::from_str(&json_body2).expect("valid Turn 2 JSON");
    let reply2 = resp2["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        reply2.contains("Anadolu") || reply2.contains("Ankara"),
        "expected contextual Ankara/İç Anadolu answer in Turn 2 reply, got: {reply2}"
    );

    let snap = srv.handle().snapshot();
    let m = snap.last_metrics.expect("Turn 2 metrics recorded");
    eprintln!(
        "[Turn 2 Verified] reply={reply2:?} | cached={} + new={} | prefill={:.2} ms | decode={:.1} tok/s ({:.1} GB/s)",
        m.cached_tokens, m.new_prefill_tokens, m.prefill_ms, m.decode_tps, m.stream_gbs
    );
    assert!(
        m.cached_tokens >= 20,
        "expected Turn 2 incremental prefix cache hit >= 20 tokens, got {}",
        m.cached_tokens
    );
    assert!(m.decode_tps > 80.0, "expected Metal CB4-Fast decode speed > 80 tok/s, got {}", m.decode_tps);
}
