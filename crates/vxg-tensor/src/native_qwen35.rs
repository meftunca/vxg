//! Native, lossless Qwen3.5-2B text-trunk import. Archive weights retain source
//! names, shapes, dtypes and bits; runtime-only adaptations never enter the archive.
use crate::format::{write_vxt_model, PackedTensor, VxtEntry};
use crate::gguf_meta::ModelConfig;
use crate::lossless::LosslessMatrix;
use crate::safetensors::{SafeDType, SafeTensorsView};
use crate::tokenizer::BpeTokenizer;
use crate::{DType, Error, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

pub const MANIFEST: &str = "native.qwen35.manifest";
const CONFIG: &str = "native.config.json";
const TOKENIZER: &str = "native.tokenizer.json";
const PREFIX: &str = "model.language_model.";
fn bad(s: impl Into<String>) -> Error {
    Error::Format(s.into())
}
fn parse(b: &[u8]) -> Result<Value> {
    serde_json::from_slice(b).map_err(|e| bad(format!("native JSON: {e}")))
}
fn usize_at(v: &Value, k: &str) -> Result<usize> {
    v[k].as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| bad(format!("invalid native config {k}")))
}
fn require(v: &Value, k: &str, expected: Value) -> Result<()> {
    if v[k] != expected {
        return Err(Error::Unsupported(format!(
            "native Qwen3.5-2B requires {k}={expected}"
        )));
    }
    Ok(())
}

/// Deliberately narrow architecture contract, rather than guessing on other Qwen variants.
pub fn config(bytes: &[u8]) -> Result<ModelConfig> {
    let root = parse(bytes)?;
    let t = &root["text_config"];
    let r = &t["rope_parameters"];
    require(&root, "model_type", json!("qwen3_5"))?;
    require(
        &root,
        "architectures",
        json!(["Qwen3_5ForConditionalGeneration"]),
    )?;
    require(&root, "tie_word_embeddings", json!(true))?;
    for (k, v) in [
        ("model_type", json!("qwen3_5_text")),
        ("tie_word_embeddings", json!(true)),
        ("attention_bias", json!(false)),
        ("attn_output_gate", json!(true)),
        ("hidden_act", json!("silu")),
        ("hidden_size", json!(2048)),
        ("intermediate_size", json!(6144)),
        ("num_hidden_layers", json!(24)),
        ("num_attention_heads", json!(8)),
        ("num_key_value_heads", json!(2)),
        ("head_dim", json!(256)),
        ("linear_num_key_heads", json!(16)),
        ("linear_num_value_heads", json!(16)),
        ("linear_key_head_dim", json!(128)),
        ("linear_value_head_dim", json!(128)),
        ("linear_conv_kernel_dim", json!(4)),
        ("vocab_size", json!(248320)),
        ("full_attention_interval", json!(4)),
        ("eos_token_id", json!(248044)),
        ("mlp_only_layers", json!([])),
    ] {
        require(t, k, v)?;
    }
    require(r, "rope_type", json!("default"))?;
    require(r, "partial_rotary_factor", json!(0.25))?;
    require(r, "rope_theta", json!(10000000))?;
    require(r, "mrope_interleaved", json!(true))?;
    require(r, "mrope_section", json!([11, 11, 10]))?;
    require(t, "rms_norm_eps", json!(0.000001))?;
    let expected: Vec<_> = (0..24)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    require(t, "layer_types", json!(expected))?;
    Ok(ModelConfig {
        arch: "qwen35".into(),
        model_name: "Qwen3.5-2B native text trunk".into(),
        n_layer: usize_at(t, "num_hidden_layers")?,
        n_embd: 2048,
        n_ff: 6144,
        n_vocab: 248320,
        n_head: 8,
        n_head_kv: 2,
        head_dim: 256,
        rms_norm_eps: 1e-6,
        rope_freq_base: 10_000_000.0,
        rope_dim: 64,
        rope_neox: true,
        ffn_gelu: false,
        embed_scale: 1.0,
        full_attn_interval: 4,
        ssm_conv_kernel: 4,
        ssm_state_size: 128,
        ssm_group_count: 16,
        ssm_dt_rank: 16,
        ssm_inner_size: 2048,
        eos_token_id: 248044,
        bos_token_id: None,
    })
}

/// Source tensor contract for every supported language parameter (no optional omissions).
fn expected() -> BTreeMap<String, (String, Vec<usize>, &'static str)> {
    let mut out = BTreeMap::new();
    let mut add = |src: String, dst: String, shape: Vec<usize>, dt| {
        out.insert(format!("{PREFIX}{src}"), (dst, shape, dt));
    };
    add("embed_tokens.weight".into(), "token_embd.weight".into(), vec![248320, 2048], "BF16");
    add("norm.weight".into(), "output_norm.weight".into(), vec![2048], "BF16");
    for i in 0..24 {
        let mut layer = |src: &str, dst: &str, shape: Vec<usize>, dt| {
            add(format!("layers.{i}.{src}"), format!("blk.{i}.{dst}"), shape, dt)
        };
        layer("input_layernorm.weight", "attn_norm.weight", vec![2048], "BF16");
        layer("post_attention_layernorm.weight", "post_attention_norm.weight", vec![2048], "BF16");
        for (s, d, sh) in [
            ("down", "down", vec![2048, 6144]),
            ("gate", "gate", vec![6144, 2048]),
            ("up", "up", vec![6144, 2048]),
        ] {
            layer(&format!("mlp.{s}_proj.weight"), &format!("ffn_{d}.weight"), sh, "BF16");
        }
        if (i + 1) % 4 == 0 {
            for (s, d, sh) in [
                ("q_proj", "q", vec![4096, 2048]),
                ("k_proj", "k", vec![512, 2048]),
                ("v_proj", "v", vec![512, 2048]),
                ("o_proj", "output", vec![2048, 2048]),
                ("q_norm", "q_norm", vec![256]),
                ("k_norm", "k_norm", vec![256]),
            ] {
                layer(&format!("self_attn.{s}.weight"), &format!("attn_{d}.weight"), sh, "BF16");
            }
        } else {
            for (s, d, sh, dt) in [
                ("in_proj_qkv.weight", "attn_qkv.weight", vec![6144, 2048], "BF16"),
                ("in_proj_z.weight", "attn_gate.weight", vec![2048, 2048], "BF16"),
                ("in_proj_a.weight", "ssm_alpha.weight", vec![16, 2048], "BF16"),
                ("in_proj_b.weight", "ssm_beta.weight", vec![16, 2048], "BF16"),
                ("out_proj.weight", "ssm_out.weight", vec![2048, 2048], "BF16"),
                ("conv1d.weight", "ssm_conv1d.weight", vec![6144, 1, 4], "BF16"),
                ("norm.weight", "ssm_norm.weight", vec![128], "F32"),
                ("A_log", "ssm_a", vec![16], "F32"),
                ("dt_bias", "ssm_dt.bias", vec![16], "BF16"),
            ] {
                layer(&format!("linear_attn.{s}"), d, sh, dt);
            }
        }
    }
    out
}
fn dtype(d: SafeDType) -> &'static str {
    match d {
        SafeDType::BF16 => "BF16",
        SafeDType::F32 => "F32",
        SafeDType::F16 => "F16",
        SafeDType::Other => "unsupported",
    }
}
fn dims(shape: &[usize]) -> (usize, usize) {
    let c = *shape.last().unwrap();
    (shape[..shape.len() - 1].iter().product(), c)
}
fn descriptor(shape: &[usize], dt: &str) -> Value {
    json!({"shape":shape,"dtype":dt})
}

fn validate_source(st: &SafeTensorsView<'_>) -> Result<Value> {
    let mut wanted = expected();
    let mut seen = HashSet::new();
    let mut retained = serde_json::Map::new();
    let mut excluded = serde_json::Map::new();
    for m in &st.tensors {
        if !seen.insert(&m.name) {
            return Err(bad(format!("duplicate source tensor {}", m.name)));
        }
        let n = m
            .shape
            .iter()
            .try_fold(1usize, |n, &x| n.checked_mul(x))
            .ok_or_else(|| bad("source shape overflow"))?;
        if n == 0 {
            return Err(bad("empty source tensor"));
        }
        let width = match m.dtype {
            SafeDType::BF16 => 2,
            SafeDType::F32 => 4,
            _ => {
                return Err(Error::Unsupported(format!(
                    "unsupported source dtype for {}",
                    m.name
                )))
            }
        };
        if n.checked_mul(width) != Some(st.tensor_bytes(m).len()) {
            return Err(bad("source byte length mismatch"));
        }
        if m.name.starts_with("model.visual.") || m.name.starts_with("mtp.") {
            excluded.insert(m.name.clone(), descriptor(&m.shape, dtype(m.dtype)));
            continue;
        }
        let (_, sh, dt) = wanted
            .remove(&m.name)
            .ok_or_else(|| bad(format!("unsupported native tensor {}", m.name)))?;
        if m.shape != sh || dtype(m.dtype) != dt {
            return Err(bad(format!(
                "native tensor contract mismatch {}: {:?}/{} expected {:?}/{dt}",
                m.name,
                m.shape,
                dtype(m.dtype),
                sh
            )));
        }
        retained.insert(m.name.clone(), descriptor(&m.shape, dt));
    }
    if !wanted.is_empty() {
        return Err(bad(format!(
            "missing native tensors: {:?}",
            wanted.keys().collect::<Vec<_>>()
        )));
    }
    Ok(
        json!({"schema":1,"scope":"qwen3.5-2b-language-only","tensors":retained,"excluded":excluded}),
    )
}

/// Preserve original BF16 bits through the codec and original F32 bits as raw F32.
pub fn pack(
    source: &[u8],
    config_json: &[u8],
    tokenizer_json: &[u8],
) -> Result<(ModelConfig, Vec<u8>)> {
    let cfg = config(config_json)?;
    tokenizer(tokenizer_json, cfg.n_vocab)?;
    let st = SafeTensorsView::parse(source)?;
    let manifest = validate_source(&st)?;
    let mut entries = Vec::new();
    for m in &st.tensors {
        if !m.name.starts_with(PREFIX) {
            continue;
        }
        let tensor = encode_tensor(&st, m)?;
        entries.push(VxtEntry {
            name: m.name.clone(),
            tensor,
        });
    }
    let meta = vec![
        (
            MANIFEST.into(),
            serde_json::to_vec(&manifest).map_err(|e| bad(e.to_string()))?,
        ),
        (CONFIG.into(), config_json.to_vec()),
        (TOKENIZER.into(), tokenizer_json.to_vec()),
    ];
    Ok((cfg, write_vxt_model(&meta, &entries)?))
}

pub fn tokenizer(bytes: &[u8], n_vocab: usize) -> Result<BpeTokenizer> {
    let v = parse(bytes)?;
    require(&v["model"], "type", json!("BPE"))?;
    let contract = parse(include_bytes!("native_qwen35_tokenizer_contract.json"))?;
    for (k, expected) in contract.as_object().unwrap() {
        if k == "model" {
            for (key, value) in expected.as_object().unwrap() {
                require(&v["model"], key, value.clone())?;
            }
        } else {
            require(&v, k, expected.clone())?;
        }
    }
    let pattern = v["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        .as_str()
        .ok_or_else(|| bad("native tokenizer regex missing"))?;
    let vocabulary = v["model"]["vocab"]
        .as_object()
        .ok_or_else(|| bad("missing native tokenizer vocab"))?;
    let mut tokens = vec![String::new(); n_vocab];
    let mut assigned = HashSet::new();
    let mut types = vec![1i32; n_vocab];
    let mut put = |tok: &str, id: u64| -> Result<()> {
        let i = usize::try_from(id).map_err(|_| bad("token ID overflow"))?;
        if i >= n_vocab || !assigned.insert(i) {
            return Err(bad("duplicate/out of range native token ID"));
        }
        tokens[i] = tok.into();
        Ok(())
    };
    for (tok, id) in vocabulary {
        put(tok, id.as_u64().ok_or_else(|| bad("invalid token ID"))?)?;
    }
    for t in v["added_tokens"]
        .as_array()
        .ok_or_else(|| bad("missing native added_tokens"))?
    {
        for flag in ["single_word", "lstrip", "rstrip", "normalized"] {
            require(t, flag, json!(false))?;
        }
        if !t["special"].is_boolean() {
            return Err(bad("invalid added token special flag"));
        }
        let id = t["id"]
            .as_u64()
            .ok_or_else(|| bad("invalid added token ID"))?;
        let content = t["content"]
            .as_str()
            .ok_or_else(|| bad("invalid added token"))?;
        if !content.is_ascii() || content.is_empty() {
            return Err(Error::Unsupported(
                "native added tokens must be nonempty ASCII (normalization bypass contract)".into(),
            ));
        }
        put(content, id)?;
        // All HF added tokens are extracted before BPE, including special=false.
        // This marker controls extraction only; decode does not skip special IDs.
        types[id as usize] = 3;
    }
    if tokens.get(248044).map(String::as_str) != Some("<|endoftext|>")
        || tokens.get(248045).map(String::as_str) != Some("<|im_start|>")
        || tokens.get(248046).map(String::as_str) != Some("<|im_end|>")
    {
        return Err(bad("native tokenizer EOS/ChatML IDs mismatch"));
    }
    // Official vocabulary has reserved padding IDs without token strings. These
    // remain empty; they are still real embedding rows and are never reassigned.
    let mut merges = Vec::new();
    for m in v["model"]["merges"]
        .as_array()
        .ok_or_else(|| bad("missing native merges"))?
    {
        let line = m
            .as_str()
            .ok_or_else(|| bad("expected native tokenizer merge string"))?;
        if line.split(' ').count() != 2 {
            return Err(bad("invalid native merge string"));
        }
        merges.push(line.to_owned());
    }
    BpeTokenizer::from_vocab_and_merges(tokens, &merges, Some(&types))?.with_native_pattern(pattern)
}

fn adapt(entries: Vec<VxtEntry>, manifest: &Value) -> Result<HashMap<String, PackedTensor>> {
    require(manifest, "schema", json!(1))?;
    require(manifest, "scope", json!("qwen3.5-2b-language-only"))?;
    let desc = manifest["tensors"]
        .as_object()
        .ok_or_else(|| bad("missing native shape manifest"))?;
    let mut wanted = expected();
    if desc.len() != wanted.len() {
        return Err(bad("native manifest tensor count mismatch"));
    }
    let mut map = HashMap::new();
    for e in entries {
        let (dst, shape, dt) = wanted.remove(&e.name).ok_or_else(|| {
            bad(format!(
                "unexpected/duplicate native archive tensor {}",
                e.name
            ))
        })?;
        if desc.get(&e.name) != Some(&descriptor(&shape, dt))
            || (e.tensor.rows(), e.tensor.cols()) != dims(&shape)
        {
            return Err(bad(format!("native archive shape mismatch {}", e.name)));
        }
        let valid = match &e.tensor {
            PackedTensor::Lossless(x) => dt == "BF16" && x.dtype == DType::BF16,
            PackedTensor::RawF32 { .. } => dt == "F32",
            _ => false,
        };
        if !valid {
            return Err(bad(format!(
                "native archive dtype/codec mismatch {}",
                e.name
            )));
        }
        let mut tensor = e.tensor;
        if e.name.ends_with(".A_log")
            || (e.name.ends_with("norm.weight") && !e.name.ends_with("linear_attn.norm.weight"))
        {
            let mut data = tensor.decode_f32();
            for x in &mut data {
                *x = if e.name.ends_with(".A_log") {
                    -x.exp()
                } else {
                    *x + 1.0
                };
            }
            tensor = PackedTensor::RawF32 {
                rows: tensor.rows(),
                cols: tensor.cols(),
                data,
            };
        } else if e.name.ends_with("linear_attn.conv1d.weight") {
            // Native [channels,1,kernel] is contiguous; no bit reordering needed.
            // Archive dimensions flatten to [channels,kernel] already.
        }
        map.insert(dst, tensor);
    }
    if !wanted.is_empty() {
        return Err(bad("native archive missing tensors"));
    }
    Ok(map)
}

pub fn load_archive(
    meta: &[(String, Vec<u8>)],
    entries: Vec<VxtEntry>,
) -> Result<(ModelConfig, BpeTokenizer, HashMap<String, PackedTensor>)> {
    let get = |key: &str| -> Result<&[u8]> {
        let mut found = meta.iter().filter(|(k, _)| k == key);
        let b = found.next().ok_or_else(|| bad(format!("missing {key}")))?;
        if found.next().is_some() {
            return Err(bad(format!("duplicate {key}")));
        }
        Ok(&b.1)
    };
    let cfg = config(get(CONFIG)?)?;
    let tok = tokenizer(get(TOKENIZER)?, cfg.n_vocab)?;
    let map = adapt(entries, &parse(get(MANIFEST)?)?)?;
    Ok((cfg, tok, map))
}

/// Direct source reference deliberately bypasses compression and archive decoding.
pub fn load_source(
    source: &[u8],
    config_json: &[u8],
    tokenizer_json: &[u8],
) -> Result<(ModelConfig, BpeTokenizer, HashMap<String, PackedTensor>)> {
    let cfg = config(config_json)?;
    let tok = tokenizer(tokenizer_json, cfg.n_vocab)?;
    let st = SafeTensorsView::parse(source)?;
    let manifest = validate_source(&st)?;
    let expected = expected();
    let mut map = HashMap::new();
    for m in &st.tensors {
        let Some((dst, _, _)) = expected.get(&m.name) else {
            continue;
        };
        let mut data = st.extract_f32(m)?;
        if m.name.ends_with(".A_log") {
            for x in &mut data {
                *x = -x.exp();
            }
        } else if m.name.ends_with("norm.weight") && !m.name.ends_with("linear_attn.norm.weight") {
            for x in &mut data {
                *x += 1.0;
            }
        }
        let (rows, cols) = dims(&m.shape);
        map.insert(dst.clone(), PackedTensor::RawF32 { rows, cols, data });
    }
    debug_assert_eq!(manifest["tensors"].as_object().unwrap().len(), map.len());
    Ok((cfg, tok, map))
}

fn encode_tensor(
    st: &SafeTensorsView<'_>,
    m: &crate::safetensors::SafeTensorMeta,
) -> Result<PackedTensor> {
    let (rows, cols) = dims(&m.shape);
    match st.extract_fp16_words(m)? {
        Some((dt, words)) => Ok(PackedTensor::Lossless(LosslessMatrix::encode(
            dt, rows, cols, &words,
        )?)),
        None if m.dtype == SafeDType::F32 => Ok(PackedTensor::RawF32 {
            rows,
            cols,
            data: st.extract_f32(m)?,
        }),
        _ => Err(Error::Unsupported("native tensor dtype".into())),
    }
}

/// Independently compare decoded archive payloads to every retained source bit.
/// Also checks original names, dimensions, native shape/dtype manifest and raw artifacts.
pub fn verify(
    source: &[u8],
    config_json: &[u8],
    tokenizer_json: &[u8],
    archive: &[u8],
) -> Result<usize> {
    let cfg = config(config_json)?;
    tokenizer(tokenizer_json, cfg.n_vocab)?;
    let st = SafeTensorsView::parse(source)?;
    let manifest = validate_source(&st)?;
    let (meta, entries) = crate::format::read_vxt_model(archive)?;
    let get = |key: &str| -> Result<&[u8]> {
        let found: Vec<_> = meta.iter().filter(|(k, _)| k == key).collect();
        if found.len() != 1 {
            return Err(bad(format!("missing/duplicate metadata {key}")));
        }
        Ok(found[0].1.as_slice())
    };
    if get(CONFIG)? != config_json
        || get(TOKENIZER)? != tokenizer_json
        || parse(get(MANIFEST)?)? != manifest
    {
        return Err(bad(
            "native original config/tokenizer/shape manifest mismatch",
        ));
    }
    let mut archived: HashMap<_, _> = HashMap::new();
    for e in entries {
        if archived.insert(e.name, e.tensor).is_some() {
            return Err(bad("duplicate archive tensor"));
        }
    }
    let mut count = 0;
    for m in &st.tensors {
        if !m.name.starts_with(PREFIX) {
            continue;
        }
        let t = archived
            .remove(&m.name)
            .ok_or_else(|| bad(format!("missing archive tensor {}", m.name)))?;
        if (t.rows(), t.cols()) != dims(&m.shape) {
            return Err(bad(format!("archive dimensions differ {}", m.name)));
        }
        let exact = match (&t, m.dtype) {
            (PackedTensor::Lossless(x), SafeDType::BF16) if x.dtype == DType::BF16 => {
                let words = x.decode_u16();
                let raw = st.tensor_bytes(m);
                words.len() * 2 == raw.len()
                    && words
                        .iter()
                        .zip(raw.chunks_exact(2))
                        .all(|(&a, b)| a == u16::from_le_bytes([b[0], b[1]]))
            }
            (PackedTensor::RawF32 { data, .. }, SafeDType::F32) => {
                let raw = st.tensor_bytes(m);
                data.len() * 4 == raw.len()
                    && data
                        .iter()
                        .zip(raw.chunks_exact(4))
                        .all(|(a, b)| a.to_bits() == u32::from_le_bytes(b.try_into().unwrap()))
            }
            _ => false,
        };
        if !exact {
            return Err(bad(format!("source/archive bits differ {}", m.name)));
        }
        count += 1;
    }
    if !archived.is_empty() {
        return Err(bad("unexpected native archive tensors"));
    }
    Ok(count)
}

#[cfg(test)]
#[path = "native_qwen35_tests.rs"]
mod tests;
