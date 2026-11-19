//! Low-RAM streaming `.vxt` file packers (`pack_gguf_model_to_vxt_file`, `pack_hf_model_to_vxt_file`)
//! and user-friendly quality presets (`--preset q3|q4|q6|q8|lossless`).
//! Streams each calibrated layer batch directly to disk via [`VxtStreamWriter`] so `27B`/`70B+`
//! models pack comfortably within bounded memory (`~4–8 GB` peak RSS instead of holding the full archive).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::cb4_calib::{build_cb4_calib_plan, build_cb4_calib_plan_from_source, Cb4CalibPlan};
use crate::format::VxtEntry;
use crate::format_stream::VxtStreamWriter;
use crate::gguf::{GgufTensorMeta, GgufView};
use crate::gguf_meta::{parse_gguf_kv, ModelConfig};
use crate::hf_meta::{
    map_hf_tensor_name, map_hf_vision_tensor_name, parse_hf_config, parse_hf_tokenizer_meta,
    parse_hf_vision_config, resolve_hf_model_dir,
};
use crate::hf_pack::PackProgressFn;
use crate::hf_pack_enc::{encode_hf_item, HfCalibAdapter, HfIndexedTensor};
use crate::llm_calib::{run_turbo_calibration, run_turbo_calibration_from_source, TurboCalibPlan};
use crate::llm_pack::{encode_gguf_tensor_with_layers, should_skip_layer, LlmPackMode};
use crate::mmap::TensorBytes;
use crate::safetensors::SafeTensorsView;
use crate::tokenizer::BpeTokenizer;
use crate::turbo_linalg::effective_cpu_threads;
use crate::turbo_seq_calib::run_sequential_turbo_pack_streaming;
use crate::{Error, Result};

/// Production quality/size presets calibrated on `Qwen3.5-9B` & `0.8B` Pareto frontiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackPreset {
    /// `~3.05 bpw` (`VXT-Turbo v5`, `7-bit E8`): beats `GGUF UD-IQ3_XXS` & `MLX 4-bit` Wiki Top-1.
    Q3,
    /// `~4.50 bpw` (`VXT-Turbo-HQ`, `s=1.85`): beats `GGUF Q4_K_M` while `~0.9 bpw` smaller.
    Q4,
    /// `~6.27 bpw` (`VXT-Turbo-HQ`, `s=0.75`): beats `GGUF Q6_K` (`-32%` TR KLD) while `~0.43 GB` smaller.
    Q6,
    /// `~7.71 bpw` (`VXT-Turbo-HQ`, `s=0.25`): beats `GGUF Q8_0` (`-56%` TR KLD) while `~0.89 GB` smaller.
    Q8,
    /// Bit-exact lossless BF16/FP16 archive (`~11–13 bpw`).
    Lossless,
}

impl PackPreset {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "q3" | "turbo" => Ok(Self::Q3),
            "q4" => Ok(Self::Q4),
            "q6" | "hq" | "turbo-hq" => Ok(Self::Q6),
            "q8" => Ok(Self::Q8),
            "lossless" | "fp16" | "bf16" => Ok(Self::Lossless),
            _ => Err(Error::Format(format!(
                "unknown --preset '{s}' (expected q3|q4|q6|q8|lossless)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q3 => "q3",
            Self::Q4 => "q4",
            Self::Q6 => "q6",
            Self::Q8 => "q8",
            Self::Lossless => "lossless",
        }
    }

    pub fn to_mode_and_scale(self) -> (LlmPackMode, f32) {
        match self {
            Self::Q3 => (LlmPackMode::Turbo, 1.0),
            Self::Q4 => (LlmPackMode::TurboHq, 1.85),
            Self::Q6 => (LlmPackMode::TurboHq, 0.75),
            Self::Q8 => (LlmPackMode::TurboHq, 0.25),
            Self::Lossless => (LlmPackMode::Lossless, 1.0),
        }
    }
}

/// Streams a GGUF model directly into a `.vxt` file on disk (`VXT1`), flushing each layer's
/// encoded tensors immediately so peak memory stays bounded even for `27B`/`70B+` models.
#[allow(clippy::too_many_arguments)]
pub fn pack_gguf_model_to_vxt_file(
    gguf_bytes: &[u8],
    out_path: &Path,
    mode: LlmPackMode,
    threads: usize,
    turbo_entropy_scale: f32,
    calib_text: Option<&str>,
    no_calib: bool,
    progress: Option<PackProgressFn<'_>>,
) -> Result<(ModelConfig, u64)> {
    if !turbo_entropy_scale.is_finite() || turbo_entropy_scale < 0.0 {
        return Err(Error::Format("turbo entropy scale must be finite and nonnegative".into()));
    }
    if !mode.is_turbo() && turbo_entropy_scale != 1.0 {
        return Err(Error::Format("turbo entropy scale requires turbo mode".into()));
    }
    let kv = parse_gguf_kv(gguf_bytes)?;
    let gg = GgufView::parse(gguf_bytes)?;
    let fallback_vocab = gg
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.matrix_dims().map(|(r, _)| r))
        .unwrap_or(0);
    let cfg = ModelConfig::from_gguf_kv(&kv, fallback_vocab)?;

    let active_metas: Vec<&GgufTensorMeta> = gg
        .tensors
        .iter()
        .filter(|m| !should_skip_layer(&m.name, cfg.n_layer))
        .collect();
    let active_names: HashSet<&str> = active_metas.iter().map(|m| m.name.as_str()).collect();
    let eff_t = effective_cpu_threads(threads);
    let use_seq = !no_calib
        && mode.is_turbo()
        && cfg.n_layer >= 4
        && cfg.n_embd >= 256
        && cfg.n_vocab >= 1024;
    let is_hq = mode == LlmPackMode::TurboHq;

    let mut meta_kv = vec![
        ("model.config".to_owned(), cfg.to_kv_bytes()),
        ("model.pack_mode".to_owned(), mode.name().as_bytes().to_vec()),
    ];
    if mode.is_turbo() && turbo_entropy_scale != 1.0 {
        meta_kv.push(("model.turbo_entropy_scale".to_owned(), turbo_entropy_scale.to_string().into_bytes()));
    }
    if use_seq {
        let ver: &[u8] = if is_hq { b"v5-e8-hq-gptq" } else { b"v5-e8-rd-gptq" };
        meta_kv.push(("model.turbo_version".to_owned(), ver.to_vec()));
    }
    let tok_entries = BpeTokenizer::to_vxt_meta_entries(&kv)?;
    meta_kv.extend(tok_entries.iter().cloned());

    let file = File::create(out_path)
        .map_err(|e| Error::Io(format!("{}: {e}", out_path.display())))?;
    let mut writer = VxtStreamWriter::new(file, active_metas.len(), &meta_kv, eff_t)?;
    let mut written_names: HashSet<String> = HashSet::with_capacity(active_metas.len());

    let smooth_plan = if use_seq {
        let tok = BpeTokenizer::from_vxt_meta(&tok_entries)?;
        run_sequential_turbo_pack_streaming(
            &gg,
            &cfg,
            &tok,
            calib_text,
            turbo_entropy_scale,
            is_hq,
            eff_t,
            progress,
            &mut |batch| {
                let mut entries = Vec::with_capacity(batch.len());
                for (name, tensor) in batch {
                    if active_names.contains(name.as_str()) && written_names.insert(name.clone()) {
                        entries.push(VxtEntry { name, tensor });
                    }
                }
                writer.write_batch_parallel(entries)
            },
        )?;
        TurboCalibPlan::default()
    } else if mode == LlmPackMode::Turbo {
        run_turbo_calibration(&gg, &cfg)
    } else {
        TurboCalibPlan::default()
    };

    let cb4_plan = if mode == LlmPackMode::Cb4 {
        build_cb4_calib_plan(&gg, &cfg)
    } else {
        Cb4CalibPlan::default()
    };

    let remaining: Vec<&GgufTensorMeta> = active_metas
        .into_iter()
        .filter(|m| !written_names.contains(&m.name))
        .collect();
    let rem_n = remaining.len();
    let batch_sz = (eff_t * 2).clamp(4, 32);
    let mut done_rem = 0usize;

    for chunk in remaining.chunks(batch_sz) {
        let cn = chunk.len();
        let workers = eff_t.min(cn.max(1));
        let slots: Vec<Mutex<Option<VxtEntry>>> = (0..cn).map(|_| Mutex::new(None)).collect();
        let first_err: Mutex<Option<Error>> = Mutex::new(None);
        let next_idx = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..workers {
                let (gg_r, ch_r, sl_r, er_r, ix_r, pl_r, c4_r) =
                    (&gg, chunk, &slots, &first_err, &next_idx, &smooth_plan, &cb4_plan);
                s.spawn(move || loop {
                    if er_r.lock().unwrap().is_some() { break; }
                    let i = ix_r.fetch_add(1, Ordering::Relaxed);
                    if i >= cn { break; }
                    let meta = ch_r[i];
                    match encode_gguf_tensor_with_layers(
                        gg_r, meta, mode, cfg.n_layer, cfg.full_attn_interval, pl_r, c4_r, turbo_entropy_scale,
                    ) {
                        Ok(tensor) => {
                            *sl_r[i].lock().unwrap() = Some(VxtEntry { name: meta.name.clone(), tensor });
                        }
                        Err(e) => {
                            let mut g = er_r.lock().unwrap();
                            if g.is_none() { *g = Some(e); }
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
            .map(|m| m.into_inner().unwrap().unwrap())
            .collect();
        writer.write_batch_parallel(entries)?;
        done_rem += cn;
        if let Some(cb) = progress {
            if !cb(done_rem, rem_n.max(1), "Kalan tensörler diske yazılıyor...") {
                return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
            }
        }
    }

    let total_bytes = writer.finish()?;
    Ok((cfg, total_bytes))
}

/// Streams a HuggingFace safetensors model directory directly into a `.vxt` file on disk (`VXT1`).
#[allow(clippy::too_many_arguments)]
pub fn pack_hf_model_to_vxt_file(
    path: &Path,
    out_path: &Path,
    mode: LlmPackMode,
    threads: usize,
    turbo_entropy_scale: f32,
    calib_text: Option<&str>,
    no_calib: bool,
    progress: Option<PackProgressFn<'_>>,
) -> Result<(ModelConfig, u64, u64)> {
    if !turbo_entropy_scale.is_finite() || turbo_entropy_scale < 0.0 {
        return Err(Error::Format("turbo entropy scale must be finite and nonnegative".into()));
    }
    if !mode.is_turbo() && turbo_entropy_scale != 1.0 {
        return Err(Error::Format("turbo entropy scale requires turbo mode".into()));
    }
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
    let mut raw_bytes = 0u64;
    for (si, st) in views.iter().enumerate() {
        for (mi, meta) in st.tensors.iter().enumerate() {
            let mapped = map_hf_tensor_name(&meta.name, &cfg.arch, tie_word_embeddings, cfg.n_layer)
                .or_else(|| vision_cfg.as_ref().and_then(|vc| map_hf_vision_tensor_name(&meta.name, vc.depth)));
            if let Some((dst_name, tf)) = mapped {
                if seen_dst.insert(dst_name.clone()) {
                    raw_bytes += (meta.data_end - meta.data_start) as u64;
                    if dst_name == "token_embd.weight" {
                        if let Some((rows, _)) = meta.matrix_dims() { cfg.n_vocab = rows; }
                    }
                    items.push(HfIndexedTensor { dst_name, shard_idx: si, meta_idx: mi, tf });
                }
            }
        }
    }
    if !seen_dst.contains("token_embd.weight") || !seen_dst.contains("output_norm.weight") {
        return Err(Error::Format("HuggingFace safetensors missing required language trunk weights".into()));
    }

    let tok_meta = parse_hf_tokenizer_meta(&tok_bytes, cfg.n_vocab)?;
    let eff_t = effective_cpu_threads(threads);
    let use_seq = !no_calib && mode.is_turbo() && cfg.n_layer >= 4 && cfg.n_embd >= 256 && cfg.n_vocab >= 1024;
    let is_hq = mode == LlmPackMode::TurboHq;

    let mut meta_kv = vec![
        ("model.config".to_owned(), cfg.to_kv_bytes()),
        ("model.pack_mode".to_owned(), mode.name().as_bytes().to_vec()),
    ];
    if let Some(vcfg) = vision_cfg {
        if seen_dst.contains("vis.patch_embed.proj.weight") {
            meta_kv.push(("vision.config".to_owned(), vcfg.to_json_bytes()));
        }
    }
    if mode.is_turbo() && turbo_entropy_scale != 1.0 {
        meta_kv.push(("model.turbo_entropy_scale".to_owned(), turbo_entropy_scale.to_string().into_bytes()));
    }
    if use_seq {
        let ver: &[u8] = if is_hq { b"v5-e8-hq-gptq" } else { b"v5-e8-rd-gptq" };
        meta_kv.push(("model.turbo_version".to_owned(), ver.to_vec()));
    }
    meta_kv.extend(tok_meta.iter().cloned());

    let by_name: HashMap<&str, usize> = items.iter().enumerate().map(|(i, it)| (it.dst_name.as_str(), i)).collect();
    let adapter = HfCalibAdapter { views: &views, items: &items, by_name, norm_plus_one };
    let cb4_plan = if mode == LlmPackMode::Cb4 { build_cb4_calib_plan_from_source(&adapter, &cfg) } else { Cb4CalibPlan::default() };

    let file = File::create(out_path)
        .map_err(|e| Error::Io(format!("{}: {e}", out_path.display())))?;
    let mut writer = VxtStreamWriter::new(file, items.len(), &meta_kv, eff_t)?;
    let mut written_names: HashSet<String> = HashSet::with_capacity(items.len());

    let turbo_plan = if use_seq {
        let tok = BpeTokenizer::from_vxt_meta(&tok_meta)?;
        run_sequential_turbo_pack_streaming(
            &adapter,
            &cfg,
            &tok,
            calib_text,
            turbo_entropy_scale,
            is_hq,
            eff_t,
            progress,
            &mut |batch| {
                let mut entries = Vec::with_capacity(batch.len());
                for (name, tensor) in batch {
                    if seen_dst.contains(name.as_str()) && written_names.insert(name.clone()) {
                        entries.push(VxtEntry { name, tensor });
                    }
                }
                writer.write_batch_parallel(entries)
            },
        )?;
        TurboCalibPlan::default()
    } else if mode == LlmPackMode::Turbo {
        run_turbo_calibration_from_source(&adapter, &cfg)
    } else {
        TurboCalibPlan::default()
    };

    let rem_indices: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, it)| !written_names.contains(&it.dst_name))
        .map(|(idx, _)| idx)
        .collect();
    let batch_sz = (eff_t * 2).clamp(4, 24);
    for chunk in rem_indices.chunks(batch_sz) {
        let cn = chunk.len();
        let workers = eff_t.min(6).min(cn.max(1));
        let slots: Vec<Mutex<Option<VxtEntry>>> = (0..cn).map(|_| Mutex::new(None)).collect();
        let first_err: Mutex<Option<Error>> = Mutex::new(None);
        let next_idx = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..workers {
                let (ad_r, ch_r, sl_r, er_r, ix_r, cf_r, c4_r, tb_r) =
                    (&adapter, chunk, &slots, &first_err, &next_idx, &cfg, &cb4_plan, &turbo_plan);
                s.spawn(move || loop {
                    if er_r.lock().unwrap().is_some() { break; }
                    let i = ix_r.fetch_add(1, Ordering::Relaxed);
                    if i >= cn { break; }
                    match encode_hf_item(ad_r, ch_r[i], mode, cf_r, c4_r, tb_r) {
                        Ok(entry) => { *sl_r[i].lock().unwrap() = Some(entry); }
                        Err(e) => {
                            let mut g = er_r.lock().unwrap();
                            if g.is_none() { *g = Some(e); }
                            break;
                        }
                    }
                });
            }
        });
        if let Some(e) = first_err.into_inner().unwrap() {
            return Err(e);
        }
        let entries: Vec<VxtEntry> = slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
        writer.write_batch_parallel(entries)?;
    }

    let out_bytes = writer.finish()?;
    Ok((cfg, raw_bytes, out_bytes))
}

#[cfg(test)]
mod tests {
    use super::PackPreset;
    use crate::llm_pack::LlmPackMode;

    #[test]
    fn pack_presets_map_to_pareto_points() {
        assert_eq!(PackPreset::parse("q3").unwrap().to_mode_and_scale(), (LlmPackMode::Turbo, 1.0));
        assert_eq!(PackPreset::parse("q4").unwrap().to_mode_and_scale(), (LlmPackMode::TurboHq, 1.85));
        assert_eq!(PackPreset::parse("q6").unwrap().to_mode_and_scale(), (LlmPackMode::TurboHq, 0.75));
        assert_eq!(PackPreset::parse("q8").unwrap().to_mode_and_scale(), (LlmPackMode::TurboHq, 0.25));
        assert_eq!(PackPreset::parse("lossless").unwrap().to_mode_and_scale(), (LlmPackMode::Lossless, 1.0));
        assert!(PackPreset::parse("invalid").is_err());
    }
}
