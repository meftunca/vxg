//! HuggingFace directory discovery (`config.json`, `tokenizer.json`, single or sharded
//! `model.safetensors.index.json`), hyperparameter/BPE metadata parsing, and tensor name mapping.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::gguf_meta::{encode_string_list, ModelConfig};
use crate::mmap::TensorBytes;
use crate::safetensors::SafeTensorsView;
use crate::tokenizer::BpeTokenizer;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct HfSourcePaths {
    pub dir: PathBuf,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub shard_paths: Vec<PathBuf>,
}

impl HfSourcePaths {
    pub fn total_bytes(&self) -> u64 {
        self.shard_paths
            .iter()
            .filter_map(|p| fs::metadata(p).ok().map(|m| m.len()))
            .sum()
    }
}

pub struct HfInspectTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub raw_bytes: usize,
}

pub struct HfInspectSummary {
    pub cfg: ModelConfig,
    pub shard_count: usize,
    pub total_bytes: u64,
    pub tensor_count: usize,
    pub preview: Vec<HfInspectTensor>,
}

pub fn resolve_hf_model_dir(path: &Path) -> Option<HfSourcePaths> {
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        let fname = path.file_name()?.to_str()?;
        let is_hf_member = fname == "config.json"
            || fname == "tokenizer.json"
            || fname == "model.safetensors.index.json"
            || fname.to_ascii_lowercase().ends_with(".safetensors");
        if !is_hf_member {
            return None;
        }
        path.parent()?.to_path_buf()
    };

    let config_path = dir.join("config.json");
    let tokenizer_path = dir.join("tokenizer.json");
    if !config_path.is_file() || !tokenizer_path.is_file() {
        return None;
    }

    let mut shard_paths = Vec::new();
    let idx_path = dir.join("model.safetensors.index.json");
    if idx_path.is_file() {
        if let Ok(bytes) = fs::read(&idx_path) {
            if let Ok(val) = serde_json::from_slice::<Value>(&bytes) {
                if let Some(wmap) = val.get("weight_map").and_then(Value::as_object) {
                    let mut names = BTreeSet::new();
                    for v in wmap.values() {
                        if let Some(s) = v.as_str() {
                            names.insert(s.to_owned());
                        }
                    }
                    let cands: Vec<PathBuf> = names.into_iter().map(|n| dir.join(n)).collect();
                    if !cands.is_empty() && cands.iter().all(|p| p.is_file()) {
                        shard_paths = cands;
                    }
                }
            }
        }
    }

    if shard_paths.is_empty() {
        let single = dir.join("model.safetensors");
        if single.is_file() {
            shard_paths.push(single);
        } else if path.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            shard_paths.push(path.to_path_buf());
        } else if let Ok(rd) = fs::read_dir(&dir) {
            let mut found: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file()
                        && p.extension()
                            .and_then(|e| e.to_str())
                            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
                })
                .collect();
            found.sort();
            shard_paths = found;
        }
    }

    (!shard_paths.is_empty()).then_some(HfSourcePaths {
        dir,
        config_path,
        tokenizer_path,
        shard_paths,
    })
}

fn json_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().and_then(|x| usize::try_from(x).ok())
}

fn json_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key)?.as_f64().map(|x| x as f32)
}

fn json_first_u32(v: &Value, key: &str) -> Option<u32> {
    let f = v.get(key)?;
    f.as_u64()
        .and_then(|id| u32::try_from(id).ok())
        .or_else(|| f.as_array()?.first()?.as_u64().and_then(|id| u32::try_from(id).ok()))
}

pub fn parse_hf_config(bytes: &[u8], fallback_name: &str) -> Result<(ModelConfig, bool, bool)> {
    let root: Value =
        serde_json::from_slice(bytes).map_err(|e| Error::Format(format!("HF config.json: {e}")))?;
    let t = if root.get("text_config").is_some_and(Value::is_object) {
        &root["text_config"]
    } else {
        &root
    };
    let raw_type = t
        .get("model_type")
        .or_else(|| root.get("model_type"))
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Format("missing model_type in HF config.json".into()))?;

    let (arch, norm_plus_one) = match raw_type {
        "qwen3_5" | "qwen3_5_text" => ("qwen35", true),
        "qwen3" => ("qwen3", false),
        "qwen2" => ("qwen2", false),
        "llama" | "mistral" => ("llama", false),
        other => {
            return Err(Error::Unsupported(format!(
                "unsupported HF model_type '{other}' (expected qwen3_5, qwen3, qwen2, llama)"
            )))
        }
    };

    let req = |k: &str| -> Result<usize> {
        json_usize(t, k).ok_or_else(|| Error::Format(format!("missing {k} in HF config.json")))
    };
    let (n_layer, n_embd, n_ff, n_vocab, n_head) = (
        req("num_hidden_layers")?,
        req("hidden_size")?,
        req("intermediate_size")?,
        req("vocab_size")?,
        req("num_attention_heads")?,
    );
    let n_head_kv = json_usize(t, "num_key_value_heads").unwrap_or(n_head);
    let head_dim = json_usize(t, "head_dim").unwrap_or(n_embd / n_head.max(1));
    let rms_norm_eps = json_f32(t, "rms_norm_eps").unwrap_or(1e-6);

    let r = t.get("rope_parameters").or_else(|| t.get("rope_scaling")).unwrap_or(&Value::Null);
    let default_theta = if arch == "qwen35" { 10_000_000.0 } else { 10_000.0 };
    let rope_freq_base = json_f32(t, "rope_theta")
        .or_else(|| json_f32(r, "rope_theta"))
        .unwrap_or(default_theta);
    let partial_rot = json_f32(t, "partial_rotary_factor")
        .or_else(|| json_f32(r, "partial_rotary_factor"))
        .unwrap_or(1.0);
    let rope_dim = ((head_dim as f32 * partial_rot).round() as usize).clamp(2, head_dim);

    let (full_attn_interval, ssm_conv_kernel, ssm_state_size, ssm_group_count, ssm_dt_rank, ssm_inner_size) =
        if arch == "qwen35" {
            let fi = json_usize(t, "full_attention_interval").unwrap_or(4);
            let ck = json_usize(t, "linear_conv_kernel_dim").unwrap_or(4);
            let ss = json_usize(t, "linear_key_head_dim").unwrap_or(128);
            let gc = json_usize(t, "linear_num_key_heads").unwrap_or(16);
            let dr = json_usize(t, "linear_num_value_heads").unwrap_or(16);
            let vd = json_usize(t, "linear_value_head_dim").unwrap_or(ss);
            (fi, ck, ss, gc, dr, dr * vd)
        } else {
            (0, 0, 0, 0, 0, 0)
        };

    let eos_token_id = json_first_u32(t, "eos_token_id")
        .or_else(|| json_first_u32(&root, "eos_token_id"))
        .unwrap_or(0);
    let bos_token_id = json_first_u32(t, "bos_token_id")
        .or_else(|| json_first_u32(&root, "bos_token_id"));
    let tie = t
        .get("tie_word_embeddings")
        .or_else(|| root.get("tie_word_embeddings"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    Ok((
        ModelConfig {
            arch: arch.to_owned(),
            model_name: fallback_name.to_owned(),
            n_layer,
            n_embd,
            n_ff,
            n_vocab,
            n_head,
            n_head_kv,
            head_dim,
            rms_norm_eps,
            rope_freq_base,
            rope_dim,
            rope_neox: true,
            full_attn_interval,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_dt_rank,
            ssm_inner_size,
            eos_token_id,
            bos_token_id,
        },
        norm_plus_one,
        tie,
    ))
}

pub(crate) fn parse_hf_tokenizer_meta(bytes: &[u8], min_vocab: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let v: Value = serde_json::from_slice(bytes)
        .map_err(|e| Error::Format(format!("HF tokenizer.json: {e}")))?;
    let vocab_obj = v["model"]["vocab"]
        .as_object()
        .ok_or_else(|| Error::Format("missing model.vocab in HF tokenizer.json".into()))?;
    let mut max_id = min_vocab.saturating_sub(1);
    for id_v in vocab_obj.values() {
        if let Some(id) = id_v.as_u64().and_then(|x| usize::try_from(x).ok()) {
            max_id = max_id.max(id);
        }
    }
    let empty_arr = Vec::new();
    let added = v["added_tokens"].as_array().unwrap_or(&empty_arr);
    for item in added {
        if let Some(id) = item["id"].as_u64().and_then(|x| usize::try_from(x).ok()) {
            max_id = max_id.max(id);
        }
    }
    let n_vocab = (max_id + 1).max(min_vocab);
    let mut tokens = vec![String::new(); n_vocab];
    let mut types = vec![1i32; n_vocab];
    for (tok, id_v) in vocab_obj {
        let id = id_v
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| Error::Format("invalid token ID in HF tokenizer.json".into()))?;
        tokens[id] = tok.clone();
    }
    for item in added {
        if let (Some(id), Some(content)) = (
            item["id"].as_u64().and_then(|x| usize::try_from(x).ok()),
            item["content"].as_str(),
        ) {
            tokens[id] = content.to_owned();
            types[id] = 3;
        }
    }

    let mut merges = Vec::new();
    if let Some(m_arr) = v["model"]["merges"].as_array() {
        merges.reserve(m_arr.len());
        for m in m_arr {
            if let Some(s) = m.as_str() {
                merges.push(s.to_owned());
            } else if let Some(pair) = m.as_array() {
                if pair.len() == 2 {
                    if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                        merges.push(format!("{a} {b}"));
                    }
                }
            }
        }
    }
    BpeTokenizer::from_vocab_and_merges(tokens.clone(), &merges, Some(&types))?;

    let mut type_bytes = Vec::with_capacity(types.len() * 4);
    for &t in &types {
        type_bytes.extend_from_slice(&t.to_le_bytes());
    }
    Ok(vec![
        ("tokenizer.tokens".to_owned(), encode_string_list(&tokens)),
        ("tokenizer.merges".to_owned(), encode_string_list(&merges)),
        ("tokenizer.types".to_owned(), type_bytes),
    ])
}

#[derive(Clone, Copy)]
pub(crate) enum HfTransform {
    Identity,
    NormWeight,
    NegExp,
    Squeeze1D,
}

pub(crate) fn map_hf_tensor_name(
    src: &str,
    tie_word_embeddings: bool,
    n_layer: usize,
) -> Option<(String, HfTransform)> {
    if src.starts_with("model.visual.") || src.starts_with("mtp.") || src.starts_with("visual.") {
        return None;
    }
    if src == "lm_head.weight" {
        return (!tie_word_embeddings).then(|| ("output.weight".into(), HfTransform::Identity));
    }
    let body = src
        .strip_prefix("model.language_model.")
        .or_else(|| src.strip_prefix("model."))
        .unwrap_or(src);

    match body {
        "embed_tokens.weight" => return Some(("token_embd.weight".into(), HfTransform::Identity)),
        "norm.weight" => return Some(("output_norm.weight".into(), HfTransform::NormWeight)),
        "lm_head.weight" => {
            return (!tie_word_embeddings).then(|| ("output.weight".into(), HfTransform::Identity))
        }
        _ => {}
    }

    let (idx_s, suffix) = body.strip_prefix("layers.")?.split_once('.')?;
    let il: usize = idx_s.parse().ok()?;
    if il >= n_layer {
        return None;
    }
    let (dst_suf, tf) = match suffix {
        "input_layernorm.weight" => ("attn_norm.weight", HfTransform::NormWeight),
        "post_attention_layernorm.weight" => ("post_attention_norm.weight", HfTransform::NormWeight),
        "mlp.gate_proj.weight" => ("ffn_gate.weight", HfTransform::Identity),
        "mlp.up_proj.weight" => ("ffn_up.weight", HfTransform::Identity),
        "mlp.down_proj.weight" => ("ffn_down.weight", HfTransform::Identity),
        "self_attn.q_proj.weight" => ("attn_q.weight", HfTransform::Identity),
        "self_attn.q_proj.bias" => ("attn_q.bias", HfTransform::Identity),
        "self_attn.k_proj.weight" => ("attn_k.weight", HfTransform::Identity),
        "self_attn.k_proj.bias" => ("attn_k.bias", HfTransform::Identity),
        "self_attn.v_proj.weight" => ("attn_v.weight", HfTransform::Identity),
        "self_attn.v_proj.bias" => ("attn_v.bias", HfTransform::Identity),
        "self_attn.o_proj.weight" => ("attn_output.weight", HfTransform::Identity),
        "self_attn.q_norm.weight" => ("attn_q_norm.weight", HfTransform::NormWeight),
        "self_attn.k_norm.weight" => ("attn_k_norm.weight", HfTransform::NormWeight),
        "linear_attn.in_proj_qkv.weight" => ("attn_qkv.weight", HfTransform::Identity),
        "linear_attn.in_proj_z.weight" => ("attn_gate.weight", HfTransform::Identity),
        "linear_attn.in_proj_a.weight" => ("ssm_alpha.weight", HfTransform::Identity),
        "linear_attn.in_proj_b.weight" => ("ssm_beta.weight", HfTransform::Identity),
        "linear_attn.out_proj.weight" => ("ssm_out.weight", HfTransform::Identity),
        "linear_attn.conv1d.weight" => ("ssm_conv1d.weight", HfTransform::Squeeze1D),
        "linear_attn.norm.weight" => ("ssm_norm.weight", HfTransform::Identity),
        "linear_attn.A_log" => ("ssm_a", HfTransform::NegExp),
        "linear_attn.dt_bias" => ("ssm_dt.bias", HfTransform::Identity),
        _ => return None,
    };
    Some((format!("blk.{il}.{dst_suf}"), tf))
}

pub fn inspect_hf_model_dir(hf: &HfSourcePaths) -> Result<HfInspectSummary> {
    let dir_name = hf.dir.file_name().and_then(|s| s.to_str()).unwrap_or("hf-model");
    let cfg_bytes = fs::read(&hf.config_path)
        .map_err(|e| Error::Io(format!("{}: {e}", hf.config_path.display())))?;
    let (cfg, _, _) = parse_hf_config(&cfg_bytes, dir_name)?;

    let mut tensor_count = 0usize;
    let mut preview = Vec::new();
    for sp in &hf.shard_paths {
        let mapped = TensorBytes::open(sp)?;
        let st = SafeTensorsView::parse(mapped.as_slice())?;
        tensor_count += st.tensors.len();
        for t in &st.tensors {
            if preview.len() >= 80 {
                break;
            }
            preview.push(HfInspectTensor {
                name: t.name.clone(),
                shape: t.shape.clone(),
                dtype: format!("{:?}", t.dtype),
                raw_bytes: t.data_end.saturating_sub(t.data_start),
            });
        }
    }
    Ok(HfInspectSummary {
        cfg,
        shard_count: hf.shard_paths.len(),
        total_bytes: hf.total_bytes(),
        tensor_count,
        preview,
    })
}
