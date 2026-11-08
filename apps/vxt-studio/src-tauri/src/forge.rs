//! Tensor inspection, local model discovery, synthetic `.safetensors` demo generation, and
//! `.gguf` / `.safetensors` -> `.vxt` conversion helpers for `VXT Studio`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use vxg_tensor::bench::generate_realistic_llm_weights;
use vxg_tensor::codebook::{CodebookMatrix, Q4_0Matrix};
use vxg_tensor::format::{read_vxt_model, write_vxt_archive, PackedTensor, VxtEntry};
use vxg_tensor::gguf::GgufView;
use vxg_tensor::gguf_meta::parse_gguf_kv;
use vxg_tensor::hf_pack::{inspect_hf_model_dir, pack_hf_model_to_vxt, resolve_hf_model_dir};
use vxg_tensor::llm_pack::{pack_gguf_model_to_vxt_with_entropy_scale, LlmPackMode};
use vxg_tensor::lossless::LosslessMatrix;
use vxg_tensor::mmap::TensorBytes;
use vxg_tensor::safetensors::{write_safetensors_u16, SafeTensorsView};
use vxg_tensor::wht2::Wht2Matrix;
use vxg_tensor::wht_turbo::TurboWhtMatrix;
use vxg_tensor::{DType, Error, Result};

#[derive(Clone, Serialize)]
pub struct DiscoveredFile {
    pub path: String,
    pub name: String,
    pub kind: String,
    pub size_mib: f64,
    pub runnable_hint: bool,
}

#[derive(Clone, Serialize)]
pub struct TensorRowInfo {
    pub index: usize,
    pub name: String,
    pub shape: String,
    pub codec_or_dtype: String,
    pub bpw: f64,
    pub size_kib: f64,
}

#[derive(Clone, Serialize)]
pub struct InspectReport {
    pub path: String,
    pub file_name: String,
    pub format: String,
    pub size_bytes: usize,
    pub size_mib: f64,
    pub tensor_count: usize,
    pub metadata_count: usize,
    pub runnable_llm: bool,
    pub arch_hint: Option<String>,
    pub tensors: Vec<TensorRowInfo>,
}

#[derive(Clone, Serialize)]
pub struct ConvertReport {
    pub input_path: String,
    pub output_path: String,
    pub mode: String,
    pub runnable_llm: bool,
    pub tensor_count: usize,
    pub input_mib: f64,
    pub output_mib: f64,
    pub ratio: f64,
    pub elapsed_s: f64,
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

pub fn default_demo_vxt_path() -> Option<String> {
    let candidates = [
        "/private/tmp/vxg-real-qualification-20261104/cb4-v2-20261108/Qwen3.5-2B-cb4-v2.vxt",
        "/private/tmp/vxg-real-qualification-20261104/cb4-baseline-20261104/Qwen3.5-2B-cb4.vxt",
    ];
    candidates
        .into_iter()
        .find(|p| Path::new(p).is_file())
        .map(str::to_owned)
}

pub fn discover_local_files() -> Vec<DiscoveredFile> {
    let mut dirs: Vec<PathBuf> = vec![
        PathBuf::from("/private/tmp/vxg-real-qualification-20261104/cb4-v2-20261108"),
        PathBuf::from("/private/tmp/vxg-real-qualification-20261104/cb4-baseline-20261104"),
        PathBuf::from("/private/tmp/vxg-real-qualification-20261104/turbo-baseline"),
        PathBuf::from("/private/tmp/vxg-real-qualification-20261104"),
        PathBuf::from("/private/tmp/vxt-studio-demo"),
    ];
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(Path::new(&home).join("Models"));
        dirs.push(Path::new(&home).join("Downloads"));
        let hf_hub = Path::new(&home).join(".cache/huggingface/hub");
        if let Ok(repos) = fs::read_dir(&hf_hub) {
            for repo in repos.flatten() {
                let snaps = repo.path().join("snapshots");
                if snaps.is_dir() {
                    dirs.push(snaps);
                }
            }
        }
    }

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if let Some(hf) = resolve_hf_model_dir(&p) {
                    let full = p.to_string_lossy().into_owned();
                    if seen.insert(full.clone()) {
                        let raw_name = p
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let name = p
                            .parent()
                            .and_then(|s| s.parent())
                            .and_then(|r| r.file_name())
                            .and_then(|r| r.to_str())
                            .and_then(|r| r.strip_prefix("models--"))
                            .map(|r| r.replace("--", "/"))
                            .unwrap_or(raw_name);
                        out.push(DiscoveredFile {
                            path: full,
                            name,
                            kind: "HF-DIR".into(),
                            size_mib: round2(hf.total_bytes() as f64 / (1024.0 * 1024.0)),
                            runnable_hint: true,
                        });
                    }
                }
                continue;
            }
            if !p.is_file() {
                continue;
            }
            let Some(ext) = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) else {
                continue;
            };
            if !matches!(ext.as_str(), "vxt" | "gguf" | "safetensors") {
                continue;
            }
            let full = p.to_string_lossy().into_owned();
            if !seen.insert(full.clone()) {
                continue;
            }
            let sz = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let runnable_hint = matches!(ext.as_str(), "vxt" | "gguf") && sz > 32 * 1024 * 1024;
            out.push(DiscoveredFile {
                path: full,
                name,
                kind: ext.to_uppercase(),
                size_mib: round2(sz as f64 / (1024.0 * 1024.0)),
                runnable_hint,
            });
        }
    }
    out.sort_by(|a, b| {
        let a_pref = a.name.contains("cb4-v2");
        let b_pref = b.name.contains("cb4-v2");
        b_pref.cmp(&a_pref).then_with(|| a.name.cmp(&b.name))
    });
    out
}

pub fn inspect_tensor_file(path: &str) -> Result<InspectReport> {
    let in_p = Path::new(path);
    let lower = path.to_ascii_lowercase();
    if in_p.is_dir()
        || (!lower.ends_with(".vxt") && !lower.ends_with(".gguf") && resolve_hf_model_dir(in_p).is_some())
    {
        let hf = resolve_hf_model_dir(in_p).ok_or_else(|| {
            Error::Format(
                "Klasörde config.json, tokenizer.json ve .safetensors bulunamadı".into(),
            )
        })?;
        let sum = inspect_hf_model_dir(&hf)?;
        let file_name = hf
            .dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_owned());
        let tensors = sum
            .preview
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let numel = t.shape.iter().copied().product::<usize>().max(1);
                TensorRowInfo {
                    index: i,
                    name: t.name,
                    shape: format!("{:?}", t.shape),
                    codec_or_dtype: t.dtype,
                    bpw: round2((t.raw_bytes as f64 * 8.0) / (numel as f64)),
                    size_kib: round2(t.raw_bytes as f64 / 1024.0),
                }
            })
            .collect();
        return Ok(InspectReport {
            path: path.to_owned(),
            file_name,
            format: format!("HuggingFace ({} shard)", sum.shard_count),
            size_bytes: sum.total_bytes as usize,
            size_mib: round2(sum.total_bytes as f64 / (1024.0 * 1024.0)),
            tensor_count: sum.tensor_count,
            metadata_count: 2,
            runnable_llm: true,
            arch_hint: Some(sum.cfg.arch),
            tensors,
        });
    }

    let mapped = TensorBytes::open(path)?;
    let bytes = mapped.as_slice();
    let file_name = Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned());
    let size_bytes = bytes.len();
    let size_mib = round2(size_bytes as f64 / (1024.0 * 1024.0));

    if bytes.starts_with(b"VXT1") {
        let (meta, entries) = read_vxt_model(bytes)?;
        let runnable_llm = meta.iter().any(|(k, _)| k == "model.config" || k == "native.qwen35.manifest");
        let arch_hint = meta
            .iter()
            .find(|(k, _)| k == "model.config")
            .and_then(|(_, v)| std::str::from_utf8(v).ok())
            .and_then(|s| s.lines().find(|l| l.starts_with("arch=")))
            .map(|l| l.trim_start_matches("arch=").to_owned());
        let tensors = entries
            .iter()
            .take(80)
            .enumerate()
            .map(|(i, e)| TensorRowInfo {
                index: i,
                name: e.name.clone(),
                shape: format!("{} × {}", e.tensor.rows(), e.tensor.cols()),
                codec_or_dtype: e.tensor.mode_name().to_owned(),
                bpw: round2(e.tensor.bits_per_weight()),
                size_kib: round2(e.tensor.compressed_bytes() as f64 / 1024.0),
            })
            .collect();
        return Ok(InspectReport {
            path: path.to_owned(),
            file_name,
            format: "VXT1".into(),
            size_bytes,
            size_mib,
            tensor_count: entries.len(),
            metadata_count: meta.len(),
            runnable_llm,
            arch_hint,
            tensors,
        });
    }

    if bytes.starts_with(b"GGUF") {
        let gg = GgufView::parse(bytes)?;
        let kv = parse_gguf_kv(bytes).unwrap_or_default();
        let arch_hint = kv.get("general.architecture").and_then(|v| v.as_str()).map(str::to_owned);
        let runnable_llm = arch_hint.is_some() && kv.contains_key("tokenizer.ggml.tokens");
        let tensors = gg
            .tensors
            .iter()
            .take(80)
            .enumerate()
            .map(|(i, t)| {
                let raw_numel = t.numel();
                let raw_b = t.ggml_type.byte_size(raw_numel).unwrap_or(0);
                let numel = raw_numel.max(1);
                TensorRowInfo {
                    index: i,
                    name: t.name.clone(),
                    shape: format!("{:?}", t.ne),
                    codec_or_dtype: format!("{:?}", t.ggml_type),
                    bpw: round2((raw_b as f64 * 8.0) / (numel as f64)),
                    size_kib: round2(raw_b as f64 / 1024.0),
                }
            })
            .collect();
        return Ok(InspectReport {
            path: path.to_owned(),
            file_name,
            format: format!("GGUF v{}", gg.version),
            size_bytes,
            size_mib,
            tensor_count: gg.tensors.len(),
            metadata_count: kv.len(),
            runnable_llm,
            arch_hint,
            tensors,
        });
    }

    let st = SafeTensorsView::parse(bytes)?;
    let tensors = st
        .tensors
        .iter()
        .take(80)
        .enumerate()
        .map(|(i, t)| {
            let raw_b = t.data_end.saturating_sub(t.data_start);
            let numel = t.numel().max(1);
            TensorRowInfo {
                index: i,
                name: t.name.clone(),
                shape: format!("{:?}", t.shape),
                codec_or_dtype: format!("{:?}", t.dtype),
                bpw: round2((raw_b as f64 * 8.0) / (numel as f64)),
                size_kib: round2(raw_b as f64 / 1024.0),
            }
        })
        .collect();
    Ok(InspectReport {
        path: path.to_owned(),
        file_name,
        format: "SafeTensors".into(),
        size_bytes,
        size_mib,
        tensor_count: st.tensors.len(),
        metadata_count: 0,
        runnable_llm: false,
        arch_hint: None,
        tensors,
    })
}

pub fn convert_tensor_to_vxt(
    input_path: &str,
    output_path: &str,
    mode: &str,
    pack_as_llm: bool,
) -> Result<ConvertReport> {
    let t0 = Instant::now();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    if let Some(parent) = Path::new(output_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| Error::Io(e.to_string()))?;
        }
    }

    let in_p = Path::new(input_path);
    if (pack_as_llm || in_p.is_dir())
        && !input_path.to_ascii_lowercase().ends_with(".gguf")
        && resolve_hf_model_dir(in_p).is_some()
    {
        let llm_mode = LlmPackMode::parse(mode)?;
        let (_cfg, raw_bytes, archive) = pack_hf_model_to_vxt(in_p, llm_mode, threads)?;
        let (_, entries) = read_vxt_model(&archive)?;
        fs::write(output_path, &archive).map_err(|e| Error::Io(e.to_string()))?;
        let in_mib = raw_bytes as f64 / (1024.0 * 1024.0);
        let out_mib = archive.len() as f64 / (1024.0 * 1024.0);
        return Ok(ConvertReport {
            input_path: input_path.to_owned(),
            output_path: output_path.to_owned(),
            mode: llm_mode.name().to_owned(),
            runnable_llm: true,
            tensor_count: entries.len(),
            input_mib: round2(in_mib),
            output_mib: round2(out_mib),
            ratio: round2(in_mib / out_mib.max(1e-9)),
            elapsed_s: round2(t0.elapsed().as_secs_f64()),
        });
    }

    let mapped = TensorBytes::open(input_path)?;
    let bytes = mapped.as_slice();

    if pack_as_llm && bytes.starts_with(b"GGUF") {
        let llm_mode = LlmPackMode::parse(mode)?;
        let (_cfg, archive) =
            pack_gguf_model_to_vxt_with_entropy_scale(bytes, llm_mode, threads, 1.0)?;
        let (_, entries) = read_vxt_model(&archive)?;
        fs::write(output_path, &archive).map_err(|e| Error::Io(e.to_string()))?;
        let in_mib = bytes.len() as f64 / (1024.0 * 1024.0);
        let out_mib = archive.len() as f64 / (1024.0 * 1024.0);
        return Ok(ConvertReport {
            input_path: input_path.to_owned(),
            output_path: output_path.to_owned(),
            mode: llm_mode.name().to_owned(),
            runnable_llm: true,
            tensor_count: entries.len(),
            input_mib: round2(in_mib),
            output_mib: round2(out_mib),
            ratio: round2(in_mib / out_mib.max(1e-9)),
            elapsed_s: round2(t0.elapsed().as_secs_f64()),
        });
    }

    let mut entries = Vec::new();
    let encode_f32 = |m: &str, rows: usize, cols: usize, f32s: &[f32]| -> Result<PackedTensor> {
        Ok(match m {
            "cb4" => PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, f32s)?),
            "cb3" => PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, f32s)?),
            "wht2" => PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, f32s)?),
            "turbo" | "turbo2" => PackedTensor::TurboWht(TurboWhtMatrix::encode(2, rows, cols, f32s)?),
            "turbo3" => PackedTensor::TurboWht(TurboWhtMatrix::encode(3, rows, cols, f32s)?),
            "q4_0" => PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, f32s)),
            _ => return Err(Error::Format(format!("unsupported tensor pack mode '{m}'"))),
        })
    };

    if bytes.starts_with(b"GGUF") {
        let gg = GgufView::parse(bytes)?;
        for meta in &gg.tensors {
            let (rows, cols) = if mode == "lossless" {
                meta.matrix_dims().unwrap_or((1, meta.numel()))
            } else {
                let Some(d) = meta.matrix_dims() else { continue };
                if d.0 < 16 || d.1 < 32 { continue; }
                d
            };
            if rows == 0 || cols == 0 {
                continue;
            }
            let tensor = if mode == "lossless" {
                if let Some((dt, words)) = gg.extract_fp16_words(meta)? {
                    PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                } else {
                    PackedTensor::RawF32 { rows, cols, data: gg.extract_f32(meta)? }
                }
            } else {
                encode_f32(mode, rows, cols, &gg.extract_f32(meta)?)?
            };
            entries.push(VxtEntry { name: meta.name.clone(), tensor });
        }
    } else {
        let st = SafeTensorsView::parse(bytes)?;
        for meta in &st.tensors {
            let (rows, cols) = if mode == "lossless" {
                meta.matrix_dims().unwrap_or((1, meta.numel()))
            } else {
                let Some(d) = meta.matrix_dims() else { continue };
                if d.0 < 16 || d.1 < 32 { continue; }
                d
            };
            if rows == 0 || cols == 0 {
                continue;
            }
            let tensor = if mode == "lossless" {
                if let Some((dt, words)) = st.extract_fp16_words(meta)? {
                    PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                } else {
                    PackedTensor::RawF32 { rows, cols, data: st.extract_f32(meta)? }
                }
            } else {
                encode_f32(mode, rows, cols, &st.extract_f32(meta)?)?
            };
            entries.push(VxtEntry { name: meta.name.clone(), tensor });
        }
    }

    if entries.is_empty() {
        return Err(Error::Format("no packable 2D tensors found in input file".into()));
    }
    let archive = write_vxt_archive(&entries)?;
    fs::write(output_path, &archive).map_err(|e| Error::Io(e.to_string()))?;
    let in_mib = bytes.len() as f64 / (1024.0 * 1024.0);
    let out_mib = archive.len() as f64 / (1024.0 * 1024.0);
    Ok(ConvertReport {
        input_path: input_path.to_owned(),
        output_path: output_path.to_owned(),
        mode: mode.to_uppercase(),
        runnable_llm: false,
        tensor_count: entries.len(),
        input_mib: round2(in_mib),
        output_mib: round2(out_mib),
        ratio: round2(in_mib / out_mib.max(1e-9)),
        elapsed_s: round2(t0.elapsed().as_secs_f64()),
    })
}

pub fn create_demo_safetensors() -> Result<InspectReport> {
    let dir = Path::new("/private/tmp/vxt-studio-demo");
    fs::create_dir_all(dir).map_err(|e| Error::Io(e.to_string()))?;
    let out_path = dir.join("demo-transformer-block-bf16.safetensors");

    let specs = [
        ("blk.0.attn_qkv.weight", 1536usize, 1024usize, 0x101u64),
        ("blk.0.attn_output.weight", 1024, 1024, 0x202),
        ("blk.0.ffn_gate.weight", 2048, 1024, 0x303),
        ("blk.0.ffn_up.weight", 2048, 1024, 0x404),
        ("blk.0.ffn_down.weight", 1024, 2048, 0x505),
    ];
    let mut shapes = Vec::new();
    let mut buffers = Vec::new();
    for &(_, r, c, seed) in &specs {
        let (words, _) = generate_realistic_llm_weights(DType::BF16, r, c, seed);
        shapes.push([r, c]);
        buffers.push(words);
    }
    let mut refs = Vec::new();
    for i in 0..specs.len() {
        refs.push((specs[i].0, DType::BF16, &shapes[i][..], buffers[i].as_slice()));
    }
    let raw = write_safetensors_u16(&refs);
    let out_str = out_path.to_string_lossy().into_owned();
    fs::write(&out_path, &raw).map_err(|e| Error::Io(e.to_string()))?;
    inspect_tensor_file(&out_str)
}
