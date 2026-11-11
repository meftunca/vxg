//! Direct HuggingFace model folder (`.safetensors` + `config.json` + `tokenizer.json`, single or
//! sharded `model.safetensors.index.json`) -> runnable `.vxt` (`VXT1`) parallel LLM packer.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::cb4_calib::{build_cb4_calib_plan_from_source, Cb4CalibPlan};
use crate::format::{write_vxt_model, VxtEntry};
use crate::gguf_meta::ModelConfig;
use crate::hf_meta::{
    map_hf_tensor_name, map_hf_vision_tensor_name, parse_hf_tokenizer_meta,
};
pub use crate::hf_meta::{
    inspect_hf_model_dir, parse_hf_config, parse_hf_vision_config, resolve_hf_model_dir,
    HfInspectSummary, HfInspectTensor, HfSourcePaths,
};
use crate::hf_pack_enc::{encode_hf_item, HfCalibAdapter, HfIndexedTensor};
use crate::llm_calib::{run_turbo_calibration_from_source, TurboCalibPlan};
use crate::llm_pack::LlmPackMode;
use crate::mmap::TensorBytes;
use crate::safetensors::SafeTensorsView;
use crate::{Error, Result};

pub type PackProgressFn<'a> = &'a (dyn Fn(usize, usize, &str) -> bool + Sync);

pub fn pack_hf_model_to_vxt(
    path: &Path,
    mode: LlmPackMode,
    threads: usize,
) -> Result<(ModelConfig, u64, Vec<u8>)> {
    pack_hf_model_to_vxt_with_progress(path, mode, threads, None)
}

pub fn pack_hf_model_to_vxt_with_progress(
    path: &Path,
    mode: LlmPackMode,
    threads: usize,
    progress: Option<PackProgressFn<'_>>,
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
            let mapped =
                map_hf_tensor_name(&meta.name, &cfg.arch, tie_word_embeddings, cfg.n_layer)
                    .or_else(|| {
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
    let turbo_plan = if mode == LlmPackMode::Turbo {
        run_turbo_calibration_from_source(&adapter, &cfg)
    } else {
        TurboCalibPlan::default()
    };

    let n = items.len();
    let done_cnt = AtomicUsize::new(0);
    if let Some(cb) = progress {
        if !cb(0, n, "Kalibrasyon tamamlandı, tensörler sıkıştırılıyor...") {
            return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
        }
    }
    let mut slots: Vec<Mutex<Option<VxtEntry>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let mut regular_indices = Vec::with_capacity(n);
    for (idx, it) in items.iter().enumerate() {
        let numel = views[it.shard_idx].tensors[it.meta_idx].numel();
        if numel >= 128_000_000 {
            if let Some(cb) = progress {
                if !cb(done_cnt.load(Ordering::Relaxed), n, &it.dst_name) {
                    return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
                }
            }
            let entry = encode_hf_item(&adapter, idx, mode, &cfg, &cb4_plan, &turbo_plan)?;
            *slots[idx].get_mut().unwrap() = Some(entry);
            let done = done_cnt.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(cb) = progress {
                if !cb(done, n, &it.dst_name) {
                    return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
                }
            }
        } else {
            regular_indices.push(idx);
        }
    }

    let reg_len = regular_indices.len();
    let workers = threads.clamp(1, 6).min(reg_len.max(1));
    let first_err: Mutex<Option<Error>> = Mutex::new(None);
    let next_idx = AtomicUsize::new(0);
    let done_ref = &done_cnt;

    std::thread::scope(|s| {
        for _ in 0..workers {
            let ad_ref = &adapter;
            let cfg_ref = &cfg;
            let cb4_ref = &cb4_plan;
            let turbo_ref = &turbo_plan;
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
                match encode_hf_item(ad_ref, idx, mode, cfg_ref, cb4_ref, turbo_ref) {
                    Ok(entry) => {
                        let tname = entry.name.clone();
                        *slots_ref[idx].lock().unwrap() = Some(entry);
                        let done = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(cb) = progress {
                            if !cb(done, n, &tname) {
                                let mut g = err_ref.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(Error::Io(
                                        "işlem kullanıcı tarafından iptal edildi".into(),
                                    ));
                                }
                                break;
                            }
                        }
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
    let entries: Vec<VxtEntry> = slots
        .into_iter()
        .filter_map(|s| s.into_inner().unwrap())
        .collect();
    let archive = write_vxt_model(&meta_kv, &entries)?;
    Ok((cfg, hf.total_bytes(), archive))
}
