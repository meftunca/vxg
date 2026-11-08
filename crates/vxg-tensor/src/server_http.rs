//! HTTP/1.1 + CORS + OpenAI-compatible `/v1/models` & `/v1/chat/completions` SSE handler
//! backing [`crate::server::EmbeddedServer`].

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::chat::{ChatMessage, ChatParams};
use crate::server::{snapshot_to_json, EmbeddedServerHandle, StreamEvent, WorkerJob};

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn write_json(stream: &mut TcpStream, status: &str, body: &Value) -> std::io::Result<()> {
    let payload = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    let hdr = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(hdr.as_bytes())?;
    stream.write_all(&payload)?;
    stream.flush()
}

fn read_http_request(stream: &TcpStream) -> std::io::Result<Option<(String, String, Vec<u8>)>> {
    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    if reader.read_line(&mut req_line)? == 0 {
        return Ok(None);
    }
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_uppercase();
    let raw_path = parts.next().unwrap_or("/").to_owned();
    let path = raw_path.split('?').next().unwrap_or("/").to_owned();

    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
        }
    }

    let mut body = Vec::new();
    if headers
        .get("transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    {
        loop {
            let mut sz_line = String::new();
            if reader.read_line(&mut sz_line)? == 0 {
                break;
            }
            let hex = sz_line.trim().split(';').next().unwrap_or("0");
            let n = usize::from_str_radix(hex, 16).unwrap_or(0);
            if n == 0 {
                let mut trail = String::new();
                let _ = reader.read_line(&mut trail);
                break;
            }
            let start = body.len();
            body.resize(start + n, 0);
            reader.read_exact(&mut body[start..])?;
            let mut crlf = [0u8; 2];
            let _ = reader.read_exact(&mut crlf);
        }
    } else if let Some(cl) = headers.get("content-length").and_then(|v| v.parse::<usize>().ok()) {
        body.resize(cl, 0);
        reader.read_exact(&mut body)?;
    }
    Ok(Some((method, path, body)))
}

fn extract_content_text(v: &Value) -> String {
    if let Some(s) = v.as_str() {
        return s.to_owned();
    }
    if let Some(arr) = v.as_array() {
        let mut out = String::new();
        for part in arr {
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                out.push_str(t);
            } else if let Some(s) = part.as_str() {
                out.push_str(s);
            }
        }
        return out;
    }
    String::new()
}

pub(crate) fn handle_http_conn(
    mut stream: TcpStream,
    handle: EmbeddedServerHandle,
) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true);
    let Some((method, path, body)) = read_http_request(&stream)? else {
        return Ok(());
    };

    if method == "OPTIONS" {
        let hdr = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        stream.write_all(hdr.as_bytes())?;
        return stream.flush();
    }

    match (method.as_str(), path.as_str()) {
        ("GET", "/health" | "/v1/vxt/status") => {
            write_json(&mut stream, "200 OK", &snapshot_to_json(&handle.snapshot()))
        }
        ("GET", "/v1/models") => {
            let snap = handle.snapshot();
            let model_id = snap
                .model
                .as_ref()
                .map(|m| m.model_name.clone())
                .unwrap_or_else(|| "vxt-local".into());
            let resp = json!({
                "object": "list",
                "data": [{
                    "id": model_id,
                    "object": "model",
                    "created": unix_secs(),
                    "owned_by": "vxg-tensor"
                }]
            });
            write_json(&mut stream, "200 OK", &resp)
        }
        ("POST", "/v1/vxt/abort") => {
            handle.abort();
            write_json(&mut stream, "200 OK", &json!({ "ok": true }))
        }
        ("POST", "/v1/vxt/reset") => {
            let _ = handle.reset_cache();
            write_json(&mut stream, "200 OK", &json!({ "ok": true }))
        }
        ("POST", "/v1/vxt/load") => {
            let req: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
            let Some(model_path) = req.get("path").and_then(Value::as_str) else {
                return write_json(
                    &mut stream,
                    "400 Bad Request",
                    &json!({"error":{"message":"missing 'path' field"}}),
                );
            };
            let use_gpu = req.get("use_gpu").and_then(Value::as_bool).unwrap_or(true);
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            match handle.load_model(model_path.to_owned(), use_gpu, threads) {
                Ok(_) => write_json(&mut stream, "200 OK", &snapshot_to_json(&handle.snapshot())),
                Err(e) => write_json(
                    &mut stream,
                    "500 Internal Server Error",
                    &json!({"error":{"message":e.to_string()}}),
                ),
            }
        }
        ("POST", "/v1/chat/completions") => handle_chat_completions(&mut stream, &handle, &body),
        _ => write_json(
            &mut stream,
            "404 Not Found",
            &json!({ "error": { "message": format!("unknown route {method} {path}") } }),
        ),
    }
}

fn handle_chat_completions(
    stream: &mut TcpStream,
    handle: &EmbeddedServerHandle,
    body: &[u8],
) -> std::io::Result<()> {
    let req: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return write_json(
                stream,
                "400 Bad Request",
                &json!({ "error": { "message": format!("invalid JSON: {e}") } }),
            )
        }
    };

    let mut messages = Vec::new();
    if let Some(arr) = req.get("messages").and_then(Value::as_array) {
        for item in arr {
            let role = item.get("role").and_then(Value::as_str).unwrap_or("user").to_owned();
            let content = item.get("content").map(extract_content_text).unwrap_or_default();
            messages.push(ChatMessage { role, content });
        }
    }
    if messages.is_empty() {
        return write_json(
            stream,
            "400 Bad Request",
            &json!({ "error": { "message": "messages array cannot be empty" } }),
        );
    }
    if handle.snapshot().model.is_none() {
        return write_json(
            stream,
            "400 Bad Request",
            &json!({ "error": { "message": "No model loaded in VXT server. Load a .vxt or .gguf model first." } }),
        );
    }

    let is_stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let enable_thinking = req.get("enable_thinking").and_then(Value::as_bool).unwrap_or(false);
    let temp = req
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(if enable_thinking { 0.6 } else { 0.0 }) as f32;
    let max_tokens = req
        .get("max_completion_tokens")
        .or_else(|| req.get("max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(if enable_thinking { 2048 } else { 1024 }) as usize;
    let presence_penalty = req
        .get("presence_penalty")
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .or_else(|| if temp <= 1e-6 && !enable_thinking { Some(0.0) } else { None });

    let model_id = handle
        .snapshot()
        .model
        .map(|m| m.model_name)
        .or_else(|| req.get("model").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| "vxt-local".into());

    let (ev_tx, ev_rx) = mpsc::channel();
    if let Err(e) = handle.send_job(WorkerJob::Chat {
        messages,
        params: ChatParams {
            max_tokens,
            temperature: temp,
            presence_penalty,
            enable_thinking,
        },
        events: ev_tx,
    }) {
        return write_json(
            stream,
            "500 Internal Server Error",
            &json!({ "error": { "message": e.to_string() } }),
        );
    }

    let created = unix_secs();
    let req_id = format!("chatcmpl-vxt-{created}");

    if !is_stream {
        let mut last_err = None;
        let mut done = None;
        while let Ok(ev) = ev_rx.recv() {
            match ev {
                StreamEvent::Chunk(_) => {}
                StreamEvent::Done { full_text, metrics } => {
                    done = Some((full_text, metrics));
                    break;
                }
                StreamEvent::Err(e) => {
                    last_err = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = last_err {
            return write_json(
                stream,
                "500 Internal Server Error",
                &json!({ "error": { "message": e } }),
            );
        }
        let (text, m) = done.unwrap_or_default();
        return write_json(
            stream,
            "200 OK",
            &json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": text },
                    "finish_reason": m.finish_reason,
                }],
                "usage": {
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.generated_tokens,
                    "total_tokens": m.prompt_tokens + m.generated_tokens,
                },
            }),
        );
    }

    let sse_hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache, no-transform\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: *\r\n\r\n";
    stream.write_all(sse_hdr.as_bytes())?;
    let init_chunk = json!({
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{ "index": 0, "delta": { "role": "assistant", "content": "" }, "finish_reason": null }]
    });
    write!(stream, "data: {init_chunk}\n\n")?;
    stream.flush()?;

    while let Ok(ev) = ev_rx.recv() {
        match ev {
            StreamEvent::Chunk(delta) => {
                let c = json!({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{ "index": 0, "delta": { "content": delta }, "finish_reason": null }]
                });
                if write!(stream, "data: {c}\n\n").and_then(|_| stream.flush()).is_err() {
                    handle.abort();
                    return Ok(());
                }
            }
            StreamEvent::Done { metrics, .. } => {
                let end_chunk = json!({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{ "index": 0, "delta": {}, "finish_reason": metrics.finish_reason }],
                    "usage": {
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.generated_tokens,
                        "total_tokens": metrics.prompt_tokens + metrics.generated_tokens,
                    }
                });
                let _ = write!(stream, "data: {end_chunk}\n\ndata: [DONE]\n\n");
                let _ = stream.flush();
                break;
            }
            StreamEvent::Err(e) => {
                let err_chunk = json!({ "error": { "message": e } });
                let _ = write!(stream, "data: {err_chunk}\n\ndata: [DONE]\n\n");
                let _ = stream.flush();
                break;
            }
        }
    }
    Ok(())
}
