//! Full-LLM GGUF -> `.vxt` (`VXT1`) parallel model packer (`pack-model`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::format::{write_vxt_model, PackedTensor, VxtEntry};
use crate::fp16::f32_to_bf16;
use crate::gguf::{GgufTensorMeta, GgufView};
use crate::gguf_meta::{parse_gguf_kv, ModelConfig};
use crate::lossless::LosslessMatrix;
use crate::tokenizer::BpeTokenizer;
use crate::wht2::Wht2Matrix;
use crate::{DType, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmPackMode {
    Lossless,
    Cb4,
    Cb3,
    Wht2,
    Q4_0,
}

impl LlmPackMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "lossless" => Ok(Self::Lossless),
            "cb4" => Ok(Self::Cb4),
            "cb3" => Ok(Self::Cb3),
            "wht2" => Ok(Self::Wht2),
            "q4_0" => Ok(Self::Q4_0),
            other => Err(Error::Format(format!(
                "unknown --mode '{other}' (expected lossless|cb4|cb3|wht2|q4_0)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Lossless => "lossless",
            Self::Cb4 => "cb4",
            Self::Cb3 => "cb3",
            Self::Wht2 => "wht2",
            Self::Q4_0 => "q4_0",
        }
    }
}

fn should_skip_layer(name: &str, n_layer: usize) -> bool {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((idx_s, _)) = rest.split_once('.') {
            if let Ok(il) = idx_s.parse::<usize>() {
                return il >= n_layer;
            }
        }
    }
    false
}

pub fn encode_gguf_tensor(
    gg: &GgufView<'_>,
    meta: &GgufTensorMeta,
    mode: LlmPackMode,
) -> Result<PackedTensor> {
    let dims = meta.matrix_dims();
    // 1D norms/biases, conv1d kernels (`cols == 4`), or tiny control projections (`rows < 64`)
    let Some((rows, cols)) = dims else {
        let data = gg.extract_f32(meta)?;
        return Ok(PackedTensor::RawF32 {
            rows: 1,
            cols: data.len().max(1),
            data,
        });
    };

    if cols < 64 || rows < 16 {
        let data = gg.extract_f32(meta)?;
        return Ok(PackedTensor::RawF32 { rows, cols, data });
    }

    // Keep tiny recurrent control gates (`rows < 64`, e.g., `16 x 2048` `ssm_alpha`/`ssm_beta`) lossless!
    let effective_mode = if rows < 64 {
        LlmPackMode::Lossless
    } else {
        mode
    };

    match effective_mode {
        LlmPackMode::Lossless => {
            if let Some((dt, words)) = gg.extract_fp16_words(meta)? {
                Ok(PackedTensor::Lossless(LosslessMatrix::encode(
                    dt, rows, cols, &words,
                )?))
            } else {
                let f32s = gg.extract_f32(meta)?;
                let words: Vec<u16> = f32s.iter().map(|&v| f32_to_bf16(v)).collect();
                Ok(PackedTensor::Lossless(LosslessMatrix::encode(
                    DType::BF16,
                    rows,
                    cols,
                    &words,
                )?))
            }
        }
        LlmPackMode::Cb4 => {
            let f32s = gg.extract_f32(meta)?;
            Ok(PackedTensor::Codebook(CodebookMatrix::encode(
                4, rows, cols, &f32s,
            )?))
        }
        LlmPackMode::Cb3 => {
            let f32s = gg.extract_f32(meta)?;
            Ok(PackedTensor::Codebook(CodebookMatrix::encode(
                3, rows, cols, &f32s,
            )?))
        }
        LlmPackMode::Wht2 => {
            let f32s = gg.extract_f32(meta)?;
            // Hybrid 2-bit incoherence schedule: apply ~2.31 bpw Walsh-Hadamard 2-bit (`WHT2`) to
            // memoryless FFN (`ffn_gate`/`ffn_up`/`ffn_down`) & full-attention `attn_q`/`attn_output`,
            // while keeping tied `token_embd`, KV heads, and recurrent DeltaNet gates (`attn_qkv`,
            // `attn_gate`, `ssm_out`) at 3-bit Lloyd-Max + CSR outliers (`CB3`, ~3.38 bpw) so
            // recurrent state transitions never compound quantization drift!
            let is_wht2_target = rows <= 65_536
                && (meta.name.contains(".ffn_")
                    || meta.name.ends_with(".attn_q.weight")
                    || meta.name.ends_with(".attn_output.weight"));
            if is_wht2_target {
                Ok(PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &f32s)?))
            } else {
                Ok(PackedTensor::Codebook(CodebookMatrix::encode(
                    3, rows, cols, &f32s,
                )?))
            }
        }
        LlmPackMode::Q4_0 => {
            let f32s = gg.extract_f32(meta)?;
            Ok(PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &f32s)))
        }
    }
}

/// Packs a complete LLM GGUF file (config + BPE tokenizer + all trunk weights) into a `.vxt` file.
pub fn pack_gguf_model_to_vxt(
    gguf_bytes: &[u8],
    mode: LlmPackMode,
    threads: usize,
) -> Result<(ModelConfig, Vec<u8>)> {
    let kv = parse_gguf_kv(gguf_bytes)?;
    let gg = GgufView::parse(gguf_bytes)?;
    let fallback_vocab = gg
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.matrix_dims().map(|(r, _)| r))
        .unwrap_or(0);
    let cfg = ModelConfig::from_gguf_kv(&kv, fallback_vocab)?;

    let mut meta_kv = vec![
        ("model.config".to_owned(), cfg.to_kv_bytes()),
        (
            "model.pack_mode".to_owned(),
            mode.name().as_bytes().to_vec(),
        ),
    ];
    meta_kv.extend(BpeTokenizer::to_vxt_meta_entries(&kv)?);

    let active_metas: Vec<&GgufTensorMeta> = gg
        .tensors
        .iter()
        .filter(|m| !should_skip_layer(&m.name, cfg.n_layer))
        .collect();

    let n = active_metas.len();
    let slots: Vec<Mutex<Option<VxtEntry>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let first_err: Mutex<Option<Error>> = Mutex::new(None);
    let next_idx = AtomicUsize::new(0);
    let workers = threads.clamp(1, 32).min(n.max(1));

    std::thread::scope(|s| {
        for _ in 0..workers {
            let gg_ref = &gg;
            let metas_ref = &active_metas;
            let slots_ref = &slots;
            let err_ref = &first_err;
            let idx_ref = &next_idx;
            s.spawn(move || loop {
                if err_ref.lock().unwrap().is_some() {
                    break;
                }
                let idx = idx_ref.fetch_add(1, Ordering::Relaxed);
                if idx >= n {
                    break;
                }
                let meta = metas_ref[idx];
                match encode_gguf_tensor(gg_ref, meta, mode) {
                    Ok(tensor) => {
                        *slots_ref[idx].lock().unwrap() = Some(VxtEntry {
                            name: meta.name.clone(),
                            tensor,
                        });
                    }
                    Err(e) => {
                        let mut g = err_ref.lock().unwrap();
                        if g.is_none() {
                            *g = Some(e);
                        }
                        break;
                    }
                }
            });
        }
    });

    if let Some(e) = first_err.into_inner().unwrap() {
        return Err(e);
    }

    let mut entries = Vec::with_capacity(n);
    for slot in slots {
        if let Some(e) = slot.into_inner().unwrap() {
            entries.push(e);
        }
    }

    let archive = write_vxt_model(&meta_kv, &entries)?;
    Ok((cfg, archive))
}
