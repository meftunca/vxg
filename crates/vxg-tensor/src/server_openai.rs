//! OpenAI API production extensions for `vxg-tensor serve`:
//! - `tools` / `tool_choice` prompt injection and `<tool_call>` -> OpenAI `tool_calls` extraction
//! - `response_format` (`json_object` / `json_schema`) structured JSON mode
//! - `stop` sequence parsing & trimming
//! - Prometheus `/metrics` exposition formatter

use serde_json::{json, Value};

use crate::chat::ChatMessage;
use crate::server::ServerSnapshot;

/// Parses OpenAI `stop` field (`"str"` or `["s1", "s2"]`, up to 8 non-empty sequences).
pub(crate) fn parse_stop_sequences(req: &Value) -> Vec<String> {
    let Some(v) = req.get("stop") else {
        return Vec::new();
    };
    if let Some(s) = v.as_str() {
        return if s.is_empty() { Vec::new() } else { vec![s.to_owned()] };
    }
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .filter_map(Value::as_str)
            .filter(|s| !s.is_empty())
            .take(8)
            .map(str::to_owned)
            .collect();
    }
    Vec::new()
}

/// Trims `text` at the earliest occurrence of any `stop` sequence; returns `(trimmed, hit_stop)`.
pub(crate) fn trim_at_stop_sequences(text: &str, stops: &[String]) -> (String, bool) {
    let mut cut: Option<usize> = None;
    for s in stops {
        if let Some(idx) = text.find(s.as_str()) {
            cut = Some(cut.map_or(idx, |c| c.min(idx)));
        }
    }
    match cut {
        Some(idx) => (text[..idx].to_owned(), true),
        None => (text.to_owned(), false),
    }
}

/// Injects OpenAI `tools` definitions and/or `response_format` JSON constraints into `messages`.
/// Returns `true` if `tools` were supplied (enabling `<tool_call>` parsing on the output).
pub(crate) fn apply_openai_request_controls(req: &Value, messages: &mut Vec<ChatMessage>) -> bool {
    let mut additions = Vec::new();
    let mut has_tools = false;

    if let Some(tools) = req.get("tools").and_then(Value::as_array) {
        let defs: Vec<&Value> = tools.iter().filter(|t| t.is_object()).collect();
        if !defs.is_empty() {
            has_tools = true;
            let tools_json = serde_json::to_string(&defs).unwrap_or_else(|_| "[]".into());
            additions.push(format!(
                "You may call one or more functions to assist with the user query.\n\
                 You are provided with function signatures within <tools></tools> XML tags:\n\
                 <tools>\n{tools_json}\n</tools>\n\
                 For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:\n\
                 <tool_call>\n{{\"name\": \"<function-name>\", \"arguments\": {{<args-json-object>}}}}\n</tool_call>"
            ));
        }
    }

    if let Some(rf) = req.get("response_format").and_then(Value::as_object) {
        let rtype = rf.get("type").and_then(Value::as_str).unwrap_or("");
        if rtype == "json_object" {
            additions.push(
                "Respond ONLY with a single valid JSON object. Do not include markdown fences or extra prose."
                    .to_owned(),
            );
        } else if rtype == "json_schema" {
            let schema = rf
                .get("json_schema")
                .and_then(|s| serde_json::to_string(s).ok())
                .unwrap_or_else(|| "{}".into());
            additions.push(format!(
                "Respond ONLY with valid JSON conforming strictly to this JSON Schema (no markdown fences, no commentary):\n{schema}"
            ));
        }
    }

    if !additions.is_empty() {
        let block = additions.join("\n\n");
        if let Some(first) = messages.first_mut() {
            if matches!(first.role.as_str(), "system" | "developer") {
                if !first.content.is_empty() {
                    first.content.push_str("\n\n");
                }
                first.content.push_str(&block);
                return has_tools;
            }
        }
        messages.insert(0, ChatMessage::text("system", block));
    }
    has_tools
}

/// Extracts Hermes/Qwen `<tool_call>{...}</tool_call>` blocks into OpenAI `tool_calls` JSON objects.
/// Returns `(remaining_text, tool_calls)`.
pub(crate) fn extract_tool_calls(text: &str, req_salt: u64) -> (String, Vec<Value>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut cursor = 0usize;

    while let Some(rel_start) = text[cursor..].find(OPEN) {
        let start = cursor + rel_start;
        clean.push_str(&text[cursor..start]);
        let body_start = start + OPEN.len();
        let Some(rel_end) = text[body_start..].find(CLOSE) else {
            clean.push_str(&text[start..]);
            cursor = text.len();
            break;
        };
        let body_end = body_start + rel_end;
        let raw = text[body_start..body_end].trim();
        if let Ok(parsed) = serde_json::from_str::<Value>(raw) {
            if let Some(name) = parsed.get("name").and_then(Value::as_str) {
                let args = match parsed.get("arguments") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".into()),
                    None => "{}".into(),
                };
                calls.push(json!({
                    "id": format!("call_vxt_{}_{}", req_salt, calls.len()),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    }
                }));
            } else {
                clean.push_str(&text[start..body_end + CLOSE.len()]);
            }
        } else {
            clean.push_str(&text[start..body_end + CLOSE.len()]);
        }
        cursor = body_end + CLOSE.len();
    }
    if cursor < text.len() {
        clean.push_str(&text[cursor..]);
    }
    (clean.trim().to_owned(), calls)
}

/// Formats server health & throughput counters in Prometheus text exposition (`0.0.4`) format.
pub(crate) fn format_prometheus_metrics(
    snap: &ServerSnapshot,
    requests_total: u64,
    queue_depth: usize,
    tokens_generated_total: u64,
    prompt_tokens_total: u64,
    cached_tokens_total: u64,
) -> String {
    let loaded = usize::from(snap.model.is_some());
    let busy = usize::from(snap.busy);
    let (dec_tps, pre_tps) = snap
        .last_metrics
        .as_ref()
        .map_or((0.0, 0.0), |m| (m.decode_tps, m.prefill_tps));
    format!(
        "# HELP vxt_model_loaded Whether a VXT model is currently loaded (0 or 1).\n\
         # TYPE vxt_model_loaded gauge\n\
         vxt_model_loaded {loaded}\n\
         # HELP vxt_worker_busy Whether the inference worker is currently generating.\n\
         # TYPE vxt_worker_busy gauge\n\
         vxt_worker_busy {busy}\n\
         # HELP vxt_queue_depth Number of queued or in-flight inference requests.\n\
         # TYPE vxt_queue_depth gauge\n\
         vxt_queue_depth {queue_depth}\n\
         # HELP vxt_kv_cached_tokens Current prefix-cached tokens resident in KV/DeltaNet state.\n\
         # TYPE vxt_kv_cached_tokens gauge\n\
         vxt_kv_cached_tokens {}\n\
         # HELP vxt_requests_total Total inference requests completed.\n\
         # TYPE vxt_requests_total counter\n\
         vxt_requests_total {requests_total}\n\
         # HELP vxt_prompt_tokens_total Total prompt tokens processed.\n\
         # TYPE vxt_prompt_tokens_total counter\n\
         vxt_prompt_tokens_total {prompt_tokens_total}\n\
         # HELP vxt_cached_tokens_total Total prompt tokens served from prefix cache.\n\
         # TYPE vxt_cached_tokens_total counter\n\
         vxt_cached_tokens_total {cached_tokens_total}\n\
         # HELP vxt_tokens_generated_total Total completion tokens generated.\n\
         # TYPE vxt_tokens_generated_total counter\n\
         vxt_tokens_generated_total {tokens_generated_total}\n\
         # HELP vxt_last_prefill_tps Last turn prefill throughput in tokens/sec.\n\
         # TYPE vxt_last_prefill_tps gauge\n\
         vxt_last_prefill_tps {pre_tps:.2}\n\
         # HELP vxt_last_decode_tps Last turn decode throughput in tokens/sec.\n\
         # TYPE vxt_last_decode_tps gauge\n\
         vxt_last_decode_tps {dec_tps:.2}\n",
        snap.cached_tokens
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        apply_openai_request_controls, extract_tool_calls, format_prometheus_metrics,
        parse_stop_sequences, trim_at_stop_sequences,
    };
    use crate::chat::ChatMessage;
    use crate::server::ServerSnapshot;

    #[test]
    fn openai_tools_json_stop_and_metrics_work() {
        let req = json!({
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
            "response_format": {"type": "json_object"},
            "stop": ["END", "STOP"]
        });
        let mut msgs = vec![ChatMessage::text("user", "Weather in Istanbul?")];
        assert!(apply_openai_request_controls(&req, &mut msgs));
        assert_eq!(msgs[0].role, "system");
        assert!(msgs[0].content.contains("<tools>"));
        assert!(msgs[0].content.contains("valid JSON object"));

        let stops = parse_stop_sequences(&req);
        assert_eq!(stops, vec!["END", "STOP"]);
        assert_eq!(trim_at_stop_sequences("hello STOP world END", &stops), ("hello ".into(), true));

        let raw = "Thinking...\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Istanbul\"}}\n</tool_call>";
        let (clean, calls) = extract_tool_calls(raw, 42);
        assert_eq!(clean, "Thinking...");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Istanbul\"}");

        let prom = format_prometheus_metrics(&ServerSnapshot::default(), 5, 1, 256, 128, 64);
        assert!(prom.contains("vxt_requests_total 5"));
        assert!(prom.contains("vxt_tokens_generated_total 256"));
    }
}
