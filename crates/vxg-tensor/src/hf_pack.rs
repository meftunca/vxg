//! Direct HuggingFace model folder (`.safetensors` + `config.json` + `tokenizer.json`, single or
//! sharded `model.safetensors.index.json`) -> runnable `.vxt` (`VXT1`) parallel LLM packer.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::cb4_calib::{
    build_cb4_calib_plan_from_source, cb4_col_amp, cb4_target_outlier_frac, Cb4CalibPlan,
    Cb4TensorSource,
};
use crate::cb4_opt::encode_cb4_optimized;
use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::format::{write_vxt_model, PackedTensor, VxtEntry};
use crate::gguf_meta::ModelConfig;
use crate::hf_meta::{
    map_hf_tensor_name, map_hf_vision_tensor_name, parse_hf_tokenizer_meta, HfTransform,
};
pub use crate::hf_meta::{
    inspect_hf_model_dir, parse_hf_config, parse_hf_vision_config, resolve_hf_model_dir,
    HfInspectSummary, HfInspectTensor, HfSourcePaths,
};
use crate::llm_pack::LlmPackMode;
use crate::lossless::LosslessMatrix;
use crate::mmap::TensorBytes;
use crate::safetensors::{SafeTensorMeta, SafeTensorsView};
use crate::wht2::Wht2Matrix;
use crate::wht_turbo::TurboWhtMatrix;
use crate::{Error, Result};

struct HfIndexedTensor {
    dst_name: String,
    shard_idx: usize,
    meta_idx: usize,
    tf: HfTransform,
}

struct HfCalibAdapter<'a> {
    views: &'a [SafeTensorsView<'a>],
    items: &'a [HfIndexedTensor],
    by_name: HashMap<&'a str, usize>,
    norm_plus_one: bool,
}

impl HfCalibAdapter<'_> {
    fn extract_transformed(&self, idx: usize) -> Result<Vec<f32>> {
        let it = &self.items[idx];
        let st = &self.views[it.shard_idx];
        let meta = &st.tensors[it.meta_idx];
        let mut data = st.extract_f32(meta)?;
        match it.tf {
            HfTransform::NormWeight if self.norm_plus_one => {
                for x in &mut data {
                    *x += 1.0;
                }
            }
            HfTransform::NegExp => {
                for x in &mut data {
                    *x = -x.exp();
                }
            }
            _ => {}
        }
        Ok(data)
    }
}

impl Cb4TensorSource for HfCalibAdapter<'_> {
    fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    fn writer_tensors(&self, _n_layer: usize) -> Vec<String> {
        self.items
            .iter()
            .filter(|it| {
                it.dst_name.ends_with(".ffn_down.weight")
                    || it.dst_name.ends_with(".ssm_out.weight")
                    || it.dst_name.ends_with(".attn_output.weight")
            })
            .map(|it| it.dst_name.clone())
            .collect()
    }

    fn matrix_dims(&self, name: &str) -> Option<(usize, usize)> {
        let &idx = self.by_name.get(name)?;
        let it = &self.items[idx];
        item_matrix_dims(&self.views[it.shard_idx].tensors[it.meta_idx], it.tf)
    }

    fn extract_f32(&self, name: &str) -> Option<Vec<f32>> {
        let &idx = self.by_name.get(name)?;
        self.extract_transformed(idx).ok()
    }
}

fn item_matrix_dims(meta: &SafeTensorMeta, tf: HfTransform) -> Option<(usize, usize)> {
    if matches!(tf, HfTransform::Flatten2D) {
        if meta.shape.len() < 2 {
            return None;
        }
        let rows = meta.shape[0];
        let cols: usize = meta.shape[1..].iter().copied().product();
        return (rows > 0 && cols > 0).then_some((rows, cols));
    }
    meta.matrix_dims()
}

fn encode_hf_item(
    adapter: &HfCalibAdapter<'_>,
    idx: usize,
    mode: LlmPackMode,
    cfg: &ModelConfig,
    cb4_plan: &Cb4CalibPlan,
) -> Result<VxtEntry> {
    let it = &adapter.items[idx];
    let st = &adapter.views[it.shard_idx];
    let meta: &SafeTensorMeta = &st.tensors[it.meta_idx];

    if !matches!(it.tf, HfTransform::Identity | HfTransform::Flatten2D) {
        let data = adapter.extract_transformed(idx)?;
        return Ok(VxtEntry {
            name: it.dst_name.clone(),
            tensor: PackedTensor::RawF32 {
                rows: 1,
                cols: data.len().max(1),
                data,
            },
        });
    }

    let Some((rows, cols)) = item_matrix_dims(meta, it.tf) else {
        let data = st.extract_f32(meta)?;
        return Ok(VxtEntry {
            name: it.dst_name.clone(),
            tensor: PackedTensor::RawF32 {
                rows: 1,
                cols: data.len().max(1),
                data,
            },
        });
    };

    if cols < 64 || rows < 16 {
        let data = st.extract_f32(meta)?;
        return Ok(VxtEntry {
            name: it.dst_name.clone(),
            tensor: PackedTensor::RawF32 { rows, cols, data },
        });
    }

    let force_lossless = rows < 64
        || it.dst_name == "vis.pos_embed.weight"
        || it.dst_name == "vis.patch_embed.proj.weight"
        || it.dst_name.ends_with(".ssm_alpha.weight")
        || it.dst_name.ends_with(".ssm_beta.weight");
    let effective_mode = if force_lossless { LlmPackMode::Lossless } else { mode };
    let layer_idx = it
        .dst_name
        .strip_prefix("blk.")
        .and_then(|s| s.split_once('.'))
        .and_then(|(i, _)| i.parse::<usize>().ok());

    let tensor = match effective_mode {
        LlmPackMode::Lossless => {
            if let Some((dt, words)) = st.extract_fp16_words(meta)? {
                PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
            } else {
                PackedTensor::RawF32 {
                    rows,
                    cols,
                    data: st.extract_f32(meta)?,
                }
            }
        }
        LlmPackMode::Cb4 => {
            let f32s = st.extract_f32(meta)?;
            if it.dst_name.starts_with("vis.") {
                PackedTensor::Codebook(encode_cb4_optimized(rows, cols, &f32s, None, 0.0085)?)
            } else {
                let is_rec = layer_idx.is_some_and(|il| cfg.is_recurrent_layer(il));
                let col_amp = cb4_col_amp(cb4_plan, &it.dst_name, layer_idx, cols);
                let frac = cb4_target_outlier_frac(
                    &it.dst_name,
                    layer_idx,
                    cfg.n_layer,
                    is_rec,
                    cb4_plan.has_output_weight,
                );
                PackedTensor::Codebook(encode_cb4_optimized(rows, cols, &f32s, col_amp, frac)?)
            }
        }
        LlmPackMode::Cb3 => {
            PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &st.extract_f32(meta)?)?)
        }
        LlmPackMode::Wht2 => {
            let f32s = st.extract_f32(meta)?;
            let is_wht2 = rows <= 65_536
                && (it.dst_name.contains(".ffn_")
                    || it.dst_name.ends_with(".attn_q.weight")
                    || it.dst_name.ends_with(".attn_output.weight"));
            if is_wht2 {
                PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &f32s)?)
            } else {
                PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &f32s)?)
            }
        }
        LlmPackMode::Turbo => {
            let f32s = st.extract_f32(meta)?;
            let inv_cols = 1.0 / (cols.max(1) as f64);
            let row_ms: Vec<f32> = (0..rows)
                .map(|r| {
                    let row = &f32s[r * cols..(r + 1) * cols];
                    let s2: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    ((s2 * inv_cols) as f32).max(1e-12)
                })
                .collect();
            let mut sorted = row_ms.clone();
            sorted.sort_by(f32::total_cmp);
            let mat_ms = sorted.get(rows / 2).copied().unwrap_or(1e-6).max(1e-12);
            PackedTensor::TurboWht(TurboWhtMatrix::encode_with_hessian(
                4,
                rows,
                cols,
                &f32s,
                None,
                |r| if row_ms[r] > 3.2 * mat_ms { 0.0 } else { 0.045 },
            )?)
        }
        LlmPackMode::Q4_0 => {
            PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &st.extract_f32(meta)?))
        }
    };

    Ok(VxtEntry {
        name: it.dst_name.clone(),
        tensor,
    })
}

pub fn pack_hf_model_to_vxt(
    path: &Path,
    mode: LlmPackMode,
    threads: usize,
) -> Result<(ModelConfig, u64, Vec<u8>)> {
    let hf = resolve_hf_model_dir(path).ok_or_else(|| {
        Error::Format(format!(
            "'{}' is not a valid HuggingFace model directory (expected config.json + tokenizer.json + .safetensors)",
            path.display()
        ))
    })?;
    let dir_name = hf.dir.file_name().and_then(|s| s.to_str()).unwrap_or("hf-model");
    let cfg_bytes = fs::read(&hf.config_path)
        .map_err(|e| Error::Io(format!("{}: {e}", hf.config_path.display())))?;
    let tok_bytes = fs::read(&hf.tokenizer_path)
        .map_err(|e| Error::Io(format!("{}: {e}", hf.tokenizer_path.display())))?;

    let (mut cfg, norm_plus_one, tie_word_embeddings) = parse_hf_config(&cfg_bytes, dir_name)?;
    let preproc_bytes = fs::read(hf.dir.join("preprocessor_config.json")).ok();
    let vision_cfg = parse_hf_vision_config(&cfg_bytes, preproc_bytes.as_deref());

    let mut mapped_shards = Vec::with_capacity(hf.shard_paths.len());
    for sp in &hf.shard_paths {
        mapped_shards.push(TensorBytes::open(sp)?);
    }
    let views: Vec<SafeTensorsView<'_>> = mapped_shards
        .iter()
        .map(|m| SafeTensorsView::parse(m.as_slice()))
        .collect::<Result<Vec<_>>>()?;

    let mut items = Vec::new();
    let mut seen_dst = BTreeSet::new();
    for (si, st) in views.iter().enumerate() {
        for (mi, meta) in st.tensors.iter().enumerate() {
            let mapped = map_hf_tensor_name(&meta.name, tie_word_embeddings, cfg.n_layer).or_else(
                || {
                    vision_cfg
                        .as_ref()
                        .and_then(|vc| map_hf_vision_tensor_name(&meta.name, vc.depth))
                },
            );
            if let Some((dst_name, tf)) = mapped {
                if seen_dst.insert(dst_name.clone()) {
                    if dst_name == "token_embd.weight" {
                        if let Some((rows, _)) = meta.matrix_dims() {
                            cfg.n_vocab = rows;
                        }
                    }
                    items.push(HfIndexedTensor {
                        dst_name,
                        shard_idx: si,
                        meta_idx: mi,
                        tf,
                    });
                }
            }
        }
    }

    if !seen_dst.contains("token_embd.weight") || !seen_dst.contains("output_norm.weight") {
        return Err(Error::Format(
            "HuggingFace safetensors missing required language trunk weights (embed_tokens / norm)"
                .into(),
        ));
    }

    let tok_meta = parse_hf_tokenizer_meta(&tok_bytes, cfg.n_vocab)?;
    let mut meta_kv = vec![
        ("model.config".to_owned(), cfg.to_kv_bytes()),
        ("model.pack_mode".to_owned(), mode.name().as_bytes().to_vec()),
    ];
    if let Some(vcfg) = vision_cfg {
        if seen_dst.contains("vis.patch_embed.proj.weight") {
            meta_kv.push(("vision.config".to_owned(), vcfg.to_json_bytes()));
        }
    }
    meta_kv.extend(tok_meta);

    let by_name: HashMap<&str, usize> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (it.dst_name.as_str(), i))
        .collect();
    let adapter = HfCalibAdapter {
        views: &views,
        items: &items,
        by_name,
        norm_plus_one,
    };
    let cb4_plan = if mode == LlmPackMode::Cb4 {
        build_cb4_calib_plan_from_source(&adapter, &cfg)
    } else {
        Cb4CalibPlan::default()
    };

    let n = items.len();
    let mut slots: Vec<Mutex<Option<VxtEntry>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let mut regular_indices = Vec::with_capacity(n);
    for (idx, it) in items.iter().enumerate() {
        let numel = views[it.shard_idx].tensors[it.meta_idx].numel();
        if numel >= 128_000_000 {
            let entry = encode_hf_item(&adapter, idx, mode, &cfg, &cb4_plan)?;
            *slots[idx].get_mut().unwrap() = Some(entry);
        } else {
            regular_indices.push(idx);
        }
    }

    let reg_len = regular_indices.len();
    let workers = threads.clamp(1, 6).min(reg_len.max(1));
    let first_err: Mutex<Option<Error>> = Mutex::new(None);
    let next_idx = AtomicUsize::new(0);

    std::thread::scope(|s| {
        for _ in 0..workers {
            let ad_ref = &adapter;
            let cfg_ref = &cfg;
            let cb4_ref = &cb4_plan;
            let slots_ref = &slots;
            let err_ref = &first_err;
            let idx_ref = &next_idx;
            let reg_ref = &regular_indices;
            s.spawn(move || loop {
                if err_ref.lock().unwrap().is_some() {
                    break;
                }
                let k = idx_ref.fetch_add(1, Ordering::Relaxed);
                if k >= reg_len {
                    break;
                }
                let idx = reg_ref[k];
                match encode_hf_item(ad_ref, idx, mode, cfg_ref, cb4_ref) {
                    Ok(entry) => *slots_ref[idx].lock().unwrap() = Some(entry),
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
    let entries: Vec<VxtEntry> = slots
        .into_iter()
        .filter_map(|s| s.into_inner().unwrap())
        .collect();
    let archive = write_vxt_model(&meta_kv, &entries)?;
    Ok((cfg, hf.total_bytes(), archive))
}
