//! GGUF v2/v3 metadata KV table parser and `.vxt` model configuration / tokenizer serializer.

use std::collections::HashMap;

use crate::{Error, Result};

#[derive(Clone, Debug)]
pub enum GgufVal {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
}

impl GgufVal {
    pub fn as_usize(&self) -> Option<usize> {
        match *self {
            Self::U64(v) => usize::try_from(v).ok(),
            Self::I64(v) => usize::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F64(v) => Some(v as f32),
            Self::U64(v) => Some(v as f32),
            Self::I64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Architecture-agnostic LLM hyperparameter descriptor supporting `qwen35` (hybrid Gated-DeltaNet
/// + Gated-GQA) as well as dense `qwen3`, `qwen2`, and `llama` Transformer models.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub arch: String,
    pub model_name: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    pub rope_neox: bool,
    /// `0` = all layers are standard full GQA; `4` (`qwen35`) = every 4th layer (`3,7,11,...`) is full Gated-GQA and others are Recurrent Gated-DeltaNet.
    pub full_attn_interval: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_dt_rank: usize,
    pub ssm_inner_size: usize,
    pub eos_token_id: u32,
    pub bos_token_id: Option<u32>,
}

impl ModelConfig {
    #[inline]
    pub fn is_recurrent_layer(&self, il: usize) -> bool {
        self.full_attn_interval > 0 && ((il + 1) % self.full_attn_interval != 0)
    }

    pub fn from_gguf_kv(kv: &HashMap<String, GgufVal>, fallback_vocab: usize) -> Result<Self> {
        let arch = kv
            .get("general.architecture")
            .and_then(GgufVal::as_str)
            .ok_or_else(|| Error::Format("missing general.architecture in GGUF".into()))?
            .to_owned();
        let model_name = kv
            .get("general.name")
            .and_then(GgufVal::as_str)
            .unwrap_or(&arch)
            .to_owned();

        let get_u = |suffix: &str| -> Option<usize> {
            kv.get(&format!("{arch}.{suffix}"))
                .and_then(GgufVal::as_usize)
        };
        let get_f = |suffix: &str| -> Option<f32> {
            kv.get(&format!("{arch}.{suffix}"))
                .and_then(GgufVal::as_f32)
        };

        let n_layer = get_u("block_count")
            .ok_or_else(|| Error::Format(format!("missing {arch}.block_count")))?;
        let n_embd = get_u("embedding_length")
            .ok_or_else(|| Error::Format(format!("missing {arch}.embedding_length")))?;
        let n_ff = get_u("feed_forward_length")
            .ok_or_else(|| Error::Format(format!("missing {arch}.feed_forward_length")))?;
        let n_head = get_u("attention.head_count")
            .ok_or_else(|| Error::Format(format!("missing {arch}.attention.head_count")))?;
        let n_head_kv = get_u("attention.head_count_kv").unwrap_or(n_head);
        let head_dim = get_u("attention.key_length").unwrap_or(n_embd / n_head.max(1));
        let rms_norm_eps = get_f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let rope_freq_base = get_f("rope.freq_base").unwrap_or(10_000.0);
        let rope_dim = get_u("rope.dimension_count").unwrap_or(head_dim);
        let rope_neox = matches!(arch.as_str(), "qwen2" | "qwen3" | "qwen35");

        let full_attn_interval = get_u("full_attention_interval").unwrap_or(0);
        let ssm_conv_kernel = get_u("ssm.conv_kernel").unwrap_or(0);
        let ssm_state_size = get_u("ssm.state_size").unwrap_or(0);
        let ssm_group_count = get_u("ssm.group_count").unwrap_or(0);
        let ssm_dt_rank = get_u("ssm.time_step_rank").unwrap_or(0);
        let ssm_inner_size = get_u("ssm.inner_size").unwrap_or(0);

        let n_vocab = match kv.get("tokenizer.ggml.tokens") {
            Some(GgufVal::ArrStr(v)) => v.len(),
            _ => get_u("vocab_size").unwrap_or(fallback_vocab),
        };
        let eos_token_id = kv
            .get("tokenizer.ggml.eos_token_id")
            .and_then(GgufVal::as_usize)
            .unwrap_or(0) as u32;
        let bos_token_id = kv
            .get("tokenizer.ggml.bos_token_id")
            .and_then(GgufVal::as_usize)
            .map(|v| v as u32);

        Ok(Self {
            arch,
            model_name,
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
            rope_neox,
            full_attn_interval,
            ssm_conv_kernel,
            ssm_state_size,
            ssm_group_count,
            ssm_dt_rank,
            ssm_inner_size,
            eos_token_id,
            bos_token_id,
        })
    }

    pub fn to_kv_bytes(&self) -> Vec<u8> {
        let s = format!(
            "arch={}\nname={}\nn_layer={}\nn_embd={}\nn_ff={}\nn_vocab={}\nn_head={}\nn_head_kv={}\nhead_dim={}\neps={:.12e}\nrope_base={:.6}\nrope_dim={}\nrope_neox={}\nfull_int={}\nssm_conv={}\nssm_state={}\nssm_grp={}\nssm_dt={}\nssm_in={}\neos={}\nbos={}\n",
            self.arch,
            self.model_name.replace('\n', " "),
            self.n_layer,
            self.n_embd,
            self.n_ff,
            self.n_vocab,
            self.n_head,
            self.n_head_kv,
            self.head_dim,
            self.rms_norm_eps,
            self.rope_freq_base,
            self.rope_dim,
            self.rope_neox as u8,
            self.full_attn_interval,
            self.ssm_conv_kernel,
            self.ssm_state_size,
            self.ssm_group_count,
            self.ssm_dt_rank,
            self.ssm_inner_size,
            self.eos_token_id,
            self.bos_token_id.map(|v| v.to_string()).unwrap_or_else(|| "none".into()),
        );
        s.into_bytes()
    }

    pub fn from_kv_bytes(bytes: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| Error::Format("invalid UTF-8 in VXT model.config".into()))?;
        let mut m = HashMap::new();
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                m.insert(k, v);
            }
        }
        let p_usize = |k: &str| -> Result<usize> {
            m.get(k)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| Error::Format(format!("missing/invalid {k} in VXT model.config")))
        };
        let p_f32 = |k: &str| -> Result<f32> {
            m.get(k)
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| Error::Format(format!("missing/invalid {k} in VXT model.config")))
        };
        Ok(Self {
            arch: m.get("arch").copied().unwrap_or("unknown").to_owned(),
            model_name: m.get("name").copied().unwrap_or("model").to_owned(),
            n_layer: p_usize("n_layer")?,
            n_embd: p_usize("n_embd")?,
            n_ff: p_usize("n_ff")?,
            n_vocab: p_usize("n_vocab")?,
            n_head: p_usize("n_head")?,
            n_head_kv: p_usize("n_head_kv")?,
            head_dim: p_usize("head_dim")?,
            rms_norm_eps: p_f32("eps")?,
            rope_freq_base: p_f32("rope_base")?,
            rope_dim: p_usize("rope_dim")?,
            rope_neox: p_usize("rope_neox")? != 0,
            full_attn_interval: p_usize("full_int")?,
            ssm_conv_kernel: p_usize("ssm_conv")?,
            ssm_state_size: p_usize("ssm_state")?,
            ssm_group_count: p_usize("ssm_grp")?,
            ssm_dt_rank: p_usize("ssm_dt")?,
            ssm_inner_size: p_usize("ssm_in")?,
            eos_token_id: p_usize("eos")? as u32,
            bos_token_id: m.get("bos").and_then(|v| v.parse().ok()),
        })
    }
}

/// Parses the complete GGUF v2/v3 metadata KV table from `bytes`.
pub fn parse_gguf_kv(bytes: &[u8]) -> Result<HashMap<String, GgufVal>> {
    if bytes.len() < 24 || &bytes[..4] != b"GGUF" {
        return Err(Error::Format("not a GGUF file".into()));
    }
    let mut r = KvCur { b: bytes, pos: 4 };
    let ver = r.u32()?;
    if !(2..=3).contains(&ver) {
        return Err(Error::Unsupported(format!(
            "unsupported GGUF version {ver}"
        )));
    }
    let _tensor_count = r.u64()?;
    let kv_count =
        usize::try_from(r.u64()?).map_err(|_| Error::Format("GGUF kv_count overflow".into()))?;

    let mut out = HashMap::with_capacity(kv_count.min(1024));
    for _ in 0..kv_count {
        let key = r.str()?;
        let vt = r.u32()?;
        if let Some(val) = r.read_val(vt)? {
            out.insert(key, val);
        }
    }
    Ok(out)
}

pub fn encode_string_list(items: &[String]) -> Vec<u8> {
    let total_bytes: usize = items.iter().map(|s| 4 + s.len()).sum::<usize>() + 4;
    let mut out = Vec::with_capacity(total_bytes);
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for s in items {
        let b = s.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

pub fn decode_string_list(bytes: &[u8]) -> Result<Vec<String>> {
    let mut r = KvCur { b: bytes, pos: 0 };
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n.min(524_288));
    for _ in 0..n {
        let len = r.u32()? as usize;
        let s = std::str::from_utf8(r.take(len)?)
            .map_err(|_| Error::Format("invalid UTF-8 in VXT string list".into()))?
            .to_owned();
        out.push(s);
    }
    Ok(out)
}

struct KvCur<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> KvCur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Format("GGUF KV overflow".into()))?;
        if end > self.b.len() {
            return Err(Error::Format("truncated GGUF KV section".into()));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        let s = self.take(4)?;
        Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    fn str(&mut self) -> Result<String> {
        let n = usize::try_from(self.u64()?)
            .map_err(|_| Error::Format("GGUF string overflow".into()))?;
        std::str::from_utf8(self.take(n)?)
            .map(str::to_owned)
            .map_err(|_| Error::Format("invalid UTF-8 in GGUF KV".into()))
    }

    fn read_val(&mut self, vt: u32) -> Result<Option<GgufVal>> {
        Ok(match vt {
            0 => Some(GgufVal::U64(self.take(1)?[0] as u64)),
            1 => Some(GgufVal::I64(self.take(1)?[0] as i8 as i64)),
            2 => {
                let s = self.take(2)?;
                Some(GgufVal::U64(u16::from_le_bytes([s[0], s[1]]) as u64))
            }
            3 => {
                let s = self.take(2)?;
                Some(GgufVal::I64(i16::from_le_bytes([s[0], s[1]]) as i64))
            }
            4 => Some(GgufVal::U64(self.u32()? as u64)),
            5 => Some(GgufVal::I64(self.i32()? as i64)),
            6 => {
                let s = self.take(4)?;
                Some(GgufVal::F64(
                    f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64
                ))
            }
            7 => Some(GgufVal::Bool(self.take(1)?[0] != 0)),
            8 => Some(GgufVal::Str(self.str()?)),
            9 => {
                let et = self.u32()?;
                let cnt = usize::try_from(self.u64()?)
                    .map_err(|_| Error::Format("GGUF array overflow".into()))?;
                if et == 8 {
                    let mut v = Vec::with_capacity(cnt.min(524_288));
                    for _ in 0..cnt {
                        v.push(self.str()?);
                    }
                    Some(GgufVal::ArrStr(v))
                } else if et == 5 || et == 4 {
                    let mut v = Vec::with_capacity(cnt.min(524_288));
                    for _ in 0..cnt {
                        v.push(self.i32()?);
                    }
                    Some(GgufVal::ArrI32(v))
                } else {
                    for _ in 0..cnt {
                        let _ = self.read_val(et)?;
                    }
                    None
                }
            }
            10 => Some(GgufVal::U64(self.u64()?)),
            11 => {
                let s = self.take(8)?;
                Some(GgufVal::I64(i64::from_le_bytes(s.try_into().unwrap())))
            }
            12 => {
                let s = self.take(8)?;
                Some(GgufVal::F64(f64::from_le_bytes(s.try_into().unwrap())))
            }
            other => return Err(Error::Format(format!("unknown GGUF KV type {other}"))),
        })
    }
}
