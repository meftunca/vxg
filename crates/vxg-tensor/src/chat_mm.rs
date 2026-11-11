//! Multimodal Vision (`VLM`) prompt formatting, per-session ViT feature caching, and
//! interactive terminal REPL (`vxg-tensor chat`) extracted from [`crate::chat`] (`<= 600` lines).

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crate::chat::{format_chat_messages_cached, ChatMessage, ChatParams, ChatSession};
use crate::llm::LlmModel;
use crate::llm_ops::strip_think_block;
use crate::tokenizer::BpeTokenizer;
use crate::vision::EncodedVisionImage;
use crate::vision_img::VisionConfig;
use crate::{Error, Result};

#[inline]
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h = (h ^ (b as u64)).wrapping_mul(0x100000001b3);
    }
    h
}

pub(crate) fn format_gemma_or_mistral(
    tokenizer: &BpeTokenizer,
    messages: &[ChatMessage],
    asst_cache: &HashMap<String, (bool, Vec<u32>)>,
) -> Option<Vec<u32>> {
    if tokenizer.token_id("<start_of_turn>").is_some()
        && tokenizer.token_id("<end_of_turn>").is_some()
    {
        let mut out = if tokenizer.token_id("<bos>").is_some() {
            tokenizer.encode("<bos>")
        } else {
            Vec::new()
        };
        let eot_ids = tokenizer.encode("<end_of_turn>\n");
        let model_hdr = tokenizer.encode("<start_of_turn>model\n");
        let mut pending_sys = String::new();
        for msg in messages {
            if matches!(msg.role.as_str(), "system" | "developer") {
                if !pending_sys.is_empty() {
                    pending_sys.push_str("\n\n");
                }
                pending_sys.push_str(msg.content.trim());
                continue;
            }
            if msg.role == "assistant" {
                if let Some((_, exact_ids)) = asst_cache.get(&msg.content) {
                    out.extend_from_slice(&model_hdr);
                    out.extend_from_slice(exact_ids);
                    out.extend_from_slice(&eot_ids);
                    continue;
                }
                out.extend(tokenizer.encode(&format!(
                    "<start_of_turn>model\n{}<end_of_turn>\n",
                    msg.content.trim()
                )));
            } else {
                let body = if pending_sys.is_empty() {
                    msg.content.trim().to_owned()
                } else {
                    let s = format!("{pending_sys}\n\n{}", msg.content.trim());
                    pending_sys.clear();
                    s
                };
                out.extend(tokenizer.encode(&format!("<start_of_turn>user\n{body}<end_of_turn>\n")));
            }
        }
        if !pending_sys.is_empty() {
            out.extend(tokenizer.encode(&format!(
                "<start_of_turn>user\n{pending_sys}<end_of_turn>\n"
            )));
        }
        out.extend_from_slice(&model_hdr);
        return Some(out);
    }

    if tokenizer.token_id("[INST]").is_some() && tokenizer.token_id("[/INST]").is_some() {
        let mut out = if tokenizer.token_id("<s>").is_some() {
            tokenizer.encode("<s>")
        } else {
            Vec::new()
        };
        let eos_s = tokenizer.encode("</s>");
        let mut pending_sys = String::new();
        for msg in messages {
            if matches!(msg.role.as_str(), "system" | "developer") {
                if !pending_sys.is_empty() {
                    pending_sys.push_str("\n\n");
                }
                pending_sys.push_str(msg.content.trim());
                continue;
            }
            if msg.role == "assistant" {
                if let Some((_, exact_ids)) = asst_cache.get(&msg.content) {
                    out.extend_from_slice(exact_ids);
                } else {
                    out.extend(tokenizer.encode(msg.content.trim()));
                }
                out.extend_from_slice(&eos_s);
            } else {
                let body = if pending_sys.is_empty() {
                    msg.content.trim().to_owned()
                } else {
                    let s = format!("{pending_sys}\n\n{}", msg.content.trim());
                    pending_sys.clear();
                    s
                };
                out.extend(tokenizer.encode(&format!("[INST] {body} [/INST]")));
            }
        }
        if !pending_sys.is_empty() {
            out.extend(tokenizer.encode(&format!("[INST] {pending_sys} [/INST]")));
        }
        return Some(out);
    }
    None
}

pub(crate) fn count_image_spans(ids: &[u32], image_token_id: u32) -> usize {
    let mut spans = 0usize;
    let mut in_span = false;
    for &id in ids {
        if id == image_token_id {
            if !in_span {
                spans += 1;
                in_span = true;
            }
        } else {
            in_span = false;
        }
    }
    spans
}

pub(crate) fn build_single_turn_vlm_prompt(
    tokenizer: &BpeTokenizer,
    arch: &str,
    vcfg: &VisionConfig,
    prompt: &str,
    images: &[EncodedVisionImage],
    enable_thinking: bool,
) -> Vec<u32> {
    let is_qwen_think = matches!(arch, "qwen3" | "qwen35");
    let mut out = tokenizer.encode("<|im_start|>user\n");
    for img in images {
        out.push(vcfg.vision_start_token_id);
        out.resize(out.len() + img.llm_tokens, vcfg.image_token_id);
        out.push(vcfg.vision_end_token_id);
    }
    out.extend(tokenizer.encode(&format!("{prompt}<|im_end|>\n")));
    let open_hdr = match (is_qwen_think, enable_thinking) {
        (true, true) => "<|im_start|>assistant\n<think>\n",
        (true, false) => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        (false, _) => "<|im_start|>assistant\n",
    };
    out.extend(tokenizer.encode(open_hdr));
    out
}

pub(crate) fn format_chat_multimodal(
    model: &LlmModel,
    messages: &[ChatMessage],
    enable_thinking: bool,
    asst_cache: &HashMap<String, (bool, Vec<u32>)>,
    img_cache: &mut HashMap<u64, Arc<EncodedVisionImage>>,
) -> Result<(Vec<u32>, Vec<Arc<EncodedVisionImage>>)> {
    if messages.iter().all(|m| m.images.is_empty()) {
        let ids = format_chat_messages_cached(
            &model.tokenizer,
            &model.cfg.arch,
            messages,
            enable_thinking,
            asst_cache,
        );
        return Ok((ids, Vec::new()));
    }
    let vcfg = model.vision_config().cloned().ok_or_else(|| {
        Error::Unsupported(
            "Loaded VXT model has no Vision encoder; load a VLM .vxt (e.g. Qwen3.5-9B / 2B)".into(),
        )
    })?;
    let tok = &model.tokenizer;
    let is_qwen_think = matches!(model.cfg.arch.as_str(), "qwen3" | "qwen35");
    let open_hdr = |think: bool| match (is_qwen_think, think) {
        (true, true) => "<|im_start|>assistant\n<think>\n",
        (true, false) => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        (false, _) => "<|im_start|>assistant\n",
    };
    let close_ids = tok.encode("<|im_end|>\n");

    let mut out = Vec::with_capacity(messages.len() * 64 + 256);
    let mut encoded_imgs = Vec::new();

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
                    out.extend(tok.encode(open_hdr(false)));
                    out.extend_from_slice(exact_ids);
                    out.extend_from_slice(&close_ids);
                    continue;
                }
            }
            out.extend(tok.encode(open_hdr(false)));
            out.extend(tok.encode(stripped));
            out.extend_from_slice(&close_ids);
        } else if msg.images.is_empty() {
            out.extend(tok.encode(&format!("<|im_start|>{role}\n{}<|im_end|>\n", msg.content)));
        } else {
            out.extend(tok.encode(&format!("<|im_start|>{role}\n")));
            for src in &msg.images {
                let key = fnv1a64(src.as_bytes());
                let img = if let Some(cached) = img_cache.get(&key) {
                    Arc::clone(cached)
                } else {
                    let enc = Arc::new(model.encode_image_source(src)?);
                    img_cache.insert(key, Arc::clone(&enc));
                    enc
                };
                out.push(vcfg.vision_start_token_id);
                out.resize(out.len() + img.llm_tokens, vcfg.image_token_id);
                out.push(vcfg.vision_end_token_id);
                encoded_imgs.push(img);
            }
            out.extend(tok.encode(&format!("{}<|im_end|>\n", msg.content)));
        }
    }
    out.extend(tok.encode(open_hdr(enable_thinking)));
    Ok((out, encoded_imgs))
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
        "Loaded '{}' ({}, {:.2} MiB) on {} in {:.0} ms\nCommands: /image <path>, /clear, /think, /exit",
        session.summary.model_name,
        session.summary.arch,
        session.summary.compressed_bytes as f64 / (1024.0 * 1024.0),
        session.summary.backend,
        session.summary.load_ms
    );

    let cancel = AtomicBool::new(false);
    let stdin = io::stdin();
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut pending_images: Vec<String> = Vec::new();
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
        if let Some(img_path) = trimmed.strip_prefix("/image ") {
            let p = img_path.trim().to_owned();
            if !p.is_empty() {
                pending_images.push(p.clone());
                println!("[attached image: {p} ({} pending)]", pending_images.len());
            }
            continue;
        }
        match trimmed {
            "/exit" | "/quit" => break,
            "/clear" => {
                session.reset();
                history.clear();
                pending_images.clear();
                println!("[context cleared]");
                continue;
            }
            "/think" => {
                params.enable_thinking = !params.enable_thinking;
                if params.enable_thinking {
                    if !explicit_temp {
                        params.temperature = 0.6;
                    }
                    params.presence_penalty = None;
                    params.max_tokens = params.max_tokens.max(2048);
                }
                println!("[thinking = {}]", params.enable_thinking);
                continue;
            }
            _ => {}
        }
        history.push(ChatMessage {
            role: "user".into(),
            content: trimmed.to_owned(),
            images: std::mem::take(&mut pending_images),
        });
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
        history.push(ChatMessage::text("assistant", reply));
    }
    Ok(())
}
