//! Full-LLM GGUF -> `.vxt` (`VXT1`) parallel model packer (`pack-model`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::cb4_calib::{
    build_cb4_calib_plan, cb4_col_amp, cb4_target_outlier_frac, Cb4CalibPlan,
};
use crate::cb4_opt::encode_cb4_optimized;
use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::format::{write_vxt_model, PackedTensor, VxtEntry};
use crate::gguf::{GgufTensorMeta, GgufView};
use crate::gguf_meta::{parse_gguf_kv, ModelConfig};
use crate::llm_calib::{run_turbo_calibration, turbo_ch_amp, TurboCalibPlan};
use crate::lossless::LosslessMatrix;
use crate::tokenizer::BpeTokenizer;
use crate::wht2::Wht2Matrix;
use crate::wht_turbo::TurboWhtMatrix;
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmPackMode {
    Lossless,
    Cb4,
    Cb3,
    Wht2,
    Turbo,
    Q4_0,
}

impl LlmPackMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "lossless" => Ok(Self::Lossless),
            "cb4" => Ok(Self::Cb4),
            "cb3" => Ok(Self::Cb3),
            "wht2" => Ok(Self::Wht2),
            "turbo" | "turbo2" => Ok(Self::Turbo),
            "q4_0" => Ok(Self::Q4_0),
            other => Err(Error::Format(format!(
                "unknown --mode '{other}' (expected lossless|cb4|cb3|wht2|turbo|q4_0)"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Lossless => "lossless",
            Self::Cb4 => "cb4",
            Self::Cb3 => "cb3",
            Self::Wht2 => "wht2",
            Self::Turbo => "turbo",
            Self::Q4_0 => "q4_0",
        }
    }
}

const EMBD_LAM: f32 = 0.052;
const GU_LAM: f32 = 0.074;
const DOWN_LAM: f32 = 0.052;

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
    encode_gguf_tensor_with_layers(
        gg,
        meta,
        mode,
        0,
        0,
        &TurboCalibPlan::default(),
        &Cb4CalibPlan::default(),
        1.0,
    )
}

fn non_empty_slice(v: &[f32]) -> Option<&[f32]> {
    (!v.is_empty()).then_some(v)
}

#[allow(clippy::too_many_arguments)]
fn encode_gguf_tensor_with_layers(
    gg: &GgufView<'_>,
    meta: &GgufTensorMeta,
    mode: LlmPackMode,
    n_layer: usize,
    full_attn_interval: usize,
    plan: &TurboCalibPlan,
    cb4_plan: &Cb4CalibPlan,
    turbo_entropy_scale: f32,
) -> Result<PackedTensor> {
    let layer_idx = meta
        .name
        .strip_prefix("blk.")
        .and_then(|s| s.split_once('.'))
        .and_then(|(idx, _)| idx.parse::<usize>().ok());
    let dims = meta.matrix_dims();
    // 1D norms/biases, conv1d kernels (`cols == 4`), or tiny control projections (`rows < 64`)
    let Some((rows, cols)) = dims else {
        let mut data = gg.extract_f32(meta)?;
        if mode == LlmPackMode::Turbo {
            if let Some(il) = layer_idx {
                let scales = if meta.name.ends_with(".attn_norm.weight") {
                    plan.attn_smooth.get(il)
                } else if meta.name.ends_with(".post_attention_norm.weight")
                    || meta.name.ends_with(".ffn_norm.weight")
                {
                    plan.post_smooth.get(il)
                } else {
                    None
                };
                if let Some(sc) = scales {
                    for &(c, g) in sc {
                        if c < data.len() {
                            data[c] /= g;
                        }
                    }
                }
            }
        }
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

    let col_scales: &[(usize, f32)] = if mode == LlmPackMode::Turbo {
        if let Some(il) = layer_idx {
            if meta.name.ends_with(".attn_qkv.weight")
                || meta.name.ends_with(".attn_gate.weight")
                || meta.name.ends_with(".ssm_alpha.weight")
                || meta.name.ends_with(".ssm_beta.weight")
                || meta.name.ends_with(".attn_q.weight")
                || meta.name.ends_with(".attn_k.weight")
                || meta.name.ends_with(".attn_v.weight")
            {
                plan.attn_smooth.get(il).map_or(&[], Vec::as_slice)
            } else if meta.name.ends_with(".ffn_gate.weight")
                || meta.name.ends_with(".ffn_up.weight")
            {
                plan.post_smooth.get(il).map_or(&[], Vec::as_slice)
            } else {
                &[]
            }
        } else {
            &[]
        }
    } else {
        &[]
    };

    // Keep tiny recurrent control gates (`rows < 64`, e.g., `16 x 2048` `ssm_alpha`/`ssm_beta`) lossless!
    let effective_mode = if rows < 64 {
        LlmPackMode::Lossless
    } else {
        mode
    };

    match effective_mode {
        LlmPackMode::Lossless => {
            if col_scales.is_empty() {
                if let Some((dt, words)) = gg.extract_fp16_words(meta)? {
                    return Ok(PackedTensor::Lossless(LosslessMatrix::encode(
                        dt, rows, cols, &words,
                    )?));
                }
            }
            let mut f32s = gg.extract_f32(meta)?;
            for &(c, g) in col_scales {
                if c < cols {
                    for r in 0..rows {
                        f32s[r * cols + c] *= g;
                    }
                }
            }
            // F32 and dequantized sources need all 32 bits. Narrowing to BF16 here
            // silently changed weights even when the caller requested lossless.
            Ok(PackedTensor::RawF32 { rows, cols, data: f32s })
        }
        LlmPackMode::Cb4 => {
            let f32s = gg.extract_f32(meta)?;
            let is_rec = layer_idx.is_some_and(|il| {
                full_attn_interval > 0 && ((il + 1) % full_attn_interval != 0)
            });
            let col_amp = cb4_col_amp(cb4_plan, &meta.name, layer_idx, cols);
            let frac = cb4_target_outlier_frac(
                &meta.name,
                layer_idx,
                n_layer,
                is_rec,
                cb4_plan.has_output_weight,
            );
            Ok(PackedTensor::Codebook(encode_cb4_optimized(
                rows, cols, &f32s, col_amp, frac,
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
        LlmPackMode::Turbo => {
            let mut f32s = gg.extract_f32(meta)?;
            for &(c, g) in col_scales {
                if c < cols {
                    for r in 0..rows {
                        f32s[r * cols + c] *= g;
                    }
                }
            }
            let n_lay = if n_layer > 0 { n_layer } else { 24 };
            let inv_cols = 1.0 / (cols.max(1) as f64);
            let row_ms: Vec<f32> = (0..rows)
                .map(|r| {
                    let row = &f32s[r * cols..(r + 1) * cols];
                    let s2: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    ((s2 * inv_cols) as f32).max(1e-12)
                })
                .collect();
            let mat_ms = {
                let mut sorted = row_ms.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                sorted.get(rows / 2).copied().unwrap_or(1e-6).max(1e-12)
            };

            let ch_amp = turbo_ch_amp(plan, &meta.name, layer_idx, cols);
            if meta.name == "token_embd.weight" || meta.name == "output.weight" {
                let h = if meta.name == "token_embd.weight" { &plan.h_embd } else { &plan.h_lm_head };
                let head_bits = if plan.ch_plan.is_some() { 3 } else { 4 };
                let lam = if head_bits == 3 {
                    0.0
                } else if meta.name == "token_embd.weight" {
                    EMBD_LAM
                } else {
                    0.040
                };
                Ok(PackedTensor::TurboWht(TurboWhtMatrix::encode_with_ch_amp(
                    head_bits,
                    rows,
                    cols,
                    &f32s,
                    non_empty_slice(h),
                    ch_amp,
                    |r| turbo_entropy_scale * if row_ms[r] > 3.0 * mat_ms { 0.0 } else { lam },
                )?))
            } else {
                let h_rot = layer_idx.and_then(|il| {
                    if meta.name.ends_with(".attn_qkv.weight")
                        || meta.name.ends_with(".attn_gate.weight")
                        || meta.name.ends_with(".attn_q.weight")
                        || meta.name.ends_with(".attn_k.weight")
                        || meta.name.ends_with(".attn_v.weight")
                    {
                        plan.h_attn.get(il).and_then(|v| non_empty_slice(v))
                    } else if meta.name.ends_with(".ssm_out.weight") {
                        plan.h_ssm_out.get(il).and_then(|v| non_empty_slice(v))
                    } else if meta.name.ends_with(".ffn_gate.weight")
                        || meta.name.ends_with(".ffn_up.weight")
                    {
                        plan.h_ffn_gu.get(il).and_then(|v| non_empty_slice(v))
                    } else {
                        None
                    }
                });
                let is_gqa_attn = meta.name.ends_with(".attn_q.weight")
                    || meta.name.ends_with(".attn_k.weight")
                    || meta.name.ends_with(".attn_v.weight")
                    || meta.name.ends_with(".attn_output.weight");
                let is_ssm_out = meta.name.ends_with(".ssm_out.weight");
                let is_qkv_gate = meta.name.ends_with(".attn_qkv.weight")
                    || meta.name.ends_with(".attn_gate.weight");
                let is_fg_fu = meta.name.ends_with(".ffn_gate.weight")
                    || meta.name.ends_with(".ffn_up.weight");
                let is_ffn_down = meta.name.ends_with(".ffn_down.weight");
                let is_edge = layer_idx.is_some_and(|il| il <= 1 || il + 2 >= n_lay);
                let is_anchor = is_edge || layer_idx.is_some_and(|il| il % 4 == 3);
                let base_lam: f32 = if is_gqa_attn || is_ssm_out {
                    0.026
                } else if is_qkv_gate {
                    if is_edge { 0.036 } else { 0.044 }
                } else if is_ffn_down {
                    if is_anchor { 0.038 } else { DOWN_LAM }
                } else if is_fg_fu {
                    if is_anchor { 0.045 } else { GU_LAM }
                } else {
                    0.040
                };
                Ok(PackedTensor::TurboWht(TurboWhtMatrix::encode_with_ch_amp(
                    4,
                    rows,
                    cols,
                    &f32s,
                    h_rot,
                    ch_amp,
                    |r| {
                        if row_ms[r] > 3.2 * mat_ms {
                            0.0
                        } else {
                            turbo_entropy_scale * (base_lam * (mat_ms / row_ms[r]).clamp(0.45, 1.50))
                        }
                    },
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
    pack_gguf_model_to_vxt_with_entropy_scale(gguf_bytes, mode, threads, 1.0)
}

/// Explicit diagnostic entropy multiplier; the default wrapper preserves archive bytes.
pub fn pack_gguf_model_to_vxt_with_entropy_scale(
    gguf_bytes: &[u8], mode: LlmPackMode, threads: usize, turbo_entropy_scale: f32,
) -> Result<(ModelConfig, Vec<u8>)> {
    pack_with_activation(gguf_bytes, mode, threads, turbo_entropy_scale, None, None)
}

pub fn pack_gguf_model_to_vxt_with_progress(
    gguf_bytes: &[u8],
    mode: LlmPackMode,
    threads: usize,
    progress: Option<&(dyn Fn(usize, usize, &str) -> bool + Sync)>,
) -> Result<(ModelConfig, Vec<u8>)> {
    pack_with_activation(gguf_bytes, mode, threads, 1.0, None, progress)
}

pub fn turbo_activation_manifest(bytes: &[u8]) -> Result<serde_json::Value> {
    let gg=GgufView::parse(bytes)?; let kv=parse_gguf_kv(bytes)?;
    let vocab=gg.tensors.iter().find(|t|t.name=="token_embd.weight").and_then(|t|t.matrix_dims()).map_or(0,|x|x.0);
    let cfg=ModelConfig::from_gguf_kv(&kv,vocab)?;
    if cfg.n_embd<64 || cfg.n_layer==0 { return Err(Error::Format("activation diagnostic requires a supported nonempty plan".into())); }
    Ok(crate::llm_calib::activation_manifest(bytes,&cfg,&run_turbo_calibration(&gg,&cfg)))
}

pub fn pack_gguf_model_with_activation_profile(bytes: &[u8], threads: usize, profile: &serde_json::Value) -> Result<(ModelConfig,Vec<u8>)> {
    pack_with_activation(bytes,LlmPackMode::Turbo,threads,1.,Some(profile),None)
}
fn pack_with_activation(gguf_bytes: &[u8], mode: LlmPackMode, threads: usize, turbo_entropy_scale: f32, activation: Option<&serde_json::Value>, progress: Option<&(dyn Fn(usize, usize, &str) -> bool + Sync)>) -> Result<(ModelConfig,Vec<u8>)> {
    if !turbo_entropy_scale.is_finite() || turbo_entropy_scale < 0.0 {
        return Err(Error::Format("turbo entropy scale must be finite and nonnegative".into()));
    }
    if mode != LlmPackMode::Turbo && turbo_entropy_scale != 1.0 {
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

    let mut meta_kv = vec![
        ("model.config".to_owned(), cfg.to_kv_bytes()),
        (
            "model.pack_mode".to_owned(),
            mode.name().as_bytes().to_vec(),
        ),
    ];
    if mode == LlmPackMode::Turbo && turbo_entropy_scale != 1.0 {
        meta_kv.push(("model.turbo_entropy_scale".to_owned(), turbo_entropy_scale.to_string().into_bytes()));
    }
    meta_kv.extend(BpeTokenizer::to_vxt_meta_entries(&kv)?);

    let active_metas: Vec<&GgufTensorMeta> = gg
        .tensors
        .iter()
        .filter(|m| !should_skip_layer(&m.name, cfg.n_layer))
        .collect();

    let n = active_metas.len();
    let workers = threads.clamp(1, 32).min(n.max(1));
    let mut smooth_plan = if mode == LlmPackMode::Turbo {
        run_turbo_calibration(&gg, &cfg)
    } else {
        TurboCalibPlan::default()
    };
    let cb4_plan = if mode == LlmPackMode::Cb4 {
        build_cb4_calib_plan(&gg, &cfg)
    } else {
        Cb4CalibPlan::default()
    };

    if let Some(profile)=activation {
        let manifest=crate::llm_calib::activation_manifest(gguf_bytes,&cfg,&smooth_plan);
        crate::llm_calib::apply_activation_profile(profile,&manifest,&mut smooth_plan)?;
        meta_kv.push(("diagnostic.activation_profile_fnv".into(),crate::llm_calib::fingerprint(&serde_json::to_vec(profile).map_err(|e|Error::Format(e.to_string()))?).into_bytes()));
    }
    let slots: Vec<Mutex<Option<VxtEntry>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let first_err: Mutex<Option<Error>> = Mutex::new(None);
    let next_idx = AtomicUsize::new(0);
    let done_cnt = AtomicUsize::new(0);
    if let Some(cb) = progress {
        if !cb(0, n, "GGUF kalibrasyonu tamamlandı, sıkıştırılıyor...") {
            return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
        }
    }

    std::thread::scope(|s| {
        for _ in 0..workers {
            let gg_ref = &gg;
            let metas_ref = &active_metas;
            let slots_ref = &slots;
            let err_ref = &first_err;
            let idx_ref = &next_idx;
            let done_ref = &done_cnt;
            let plan_ref = &smooth_plan;
            let cb4_ref = &cb4_plan;
            s.spawn(move || loop {
                if err_ref.lock().unwrap().is_some() {
                    break;
                }
                let idx = idx_ref.fetch_add(1, Ordering::Relaxed);
                if idx >= n {
                    break;
                }
                let meta = metas_ref[idx];
                match encode_gguf_tensor_with_layers(
                    gg_ref,
                    meta,
                    mode,
                    cfg.n_layer,
                    cfg.full_attn_interval,
                    plan_ref,
                    cb4_ref,
                    turbo_entropy_scale,
                ) {
                    Ok(tensor) => {
                        *slots_ref[idx].lock().unwrap() = Some(VxtEntry {
                            name: meta.name.clone(),
                            tensor,
                        });
                        let done = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                        if let Some(cb) = progress {
                            if !cb(done, n, &meta.name) {
                                let mut g = err_ref.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
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

    let mut entries = Vec::with_capacity(n);
    for slot in slots {
        if let Some(e) = slot.into_inner().unwrap() {
            entries.push(e);
        }
    }

    let archive = write_vxt_model(&meta_kv, &entries)?;
    Ok((cfg, archive))
}

#[cfg(test)]
mod entropy_ablation_tests {
    use super::*;
    fn matrix(name: &str) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes()); b.extend(1u64.to_le_bytes()); b.extend(0u64.to_le_bytes());
        b.extend((name.len() as u64).to_le_bytes()); b.extend(name.as_bytes());
        b.extend(2u32.to_le_bytes()); b.extend(64u64.to_le_bytes()); b.extend(64u64.to_le_bytes());
        b.extend(0u32.to_le_bytes()); b.extend(0u64.to_le_bytes());
        b.resize((b.len()+31)/32*32,0);
        for i in 0..4096 { b.extend((((i*73%199) as f32-99.)/83.).to_le_bytes()); }
        b
    }
    #[test]
    fn scale_zero_all_roles_and_default_identity() {
        for name in ["token_embd.weight", "output.weight", "blk.5.attn_q.weight", "blk.5.ssm_out.weight", "blk.5.attn_qkv.weight", "blk.5.attn_gate.weight", "blk.5.ffn_gate.weight", "blk.5.ffn_up.weight", "blk.5.ffn_down.weight", "blk.0.ffn_down.weight", "blk.5.other.weight"] {
            let bytes=matrix(name); let gg=GgufView::parse(&bytes).unwrap(); let meta=&gg.tensors[0];
            let old=encode_gguf_tensor(&gg,meta,LlmPackMode::Turbo).unwrap();
            let one=encode_gguf_tensor_with_layers(&gg,meta,LlmPackMode::Turbo,0,0,&TurboCalibPlan::default(),&Cb4CalibPlan::default(),1.).unwrap();
            assert_eq!(old.decode_f32(),one.decode_f32(),"{name}");
            // Independent pre-ablation lambda expression, including row energy adaptation.
            let w=gg.extract_f32(meta).unwrap();
            let ms:Vec<f32>=w.chunks(64).map(|r| (r.iter().map(|&v|(v as f64)*(v as f64)).sum::<f64>()/64.) as f32).collect();
            let mut sorted=ms.clone();sorted.sort_by(|a,b|a.total_cmp(b));let median=sorted[32];
            let special=name=="token_embd.weight" || name=="output.weight";
            let base=if name=="token_embd.weight" {0.052} else if special {0.040}
                else if name.contains("attn_q.weight") || name.contains("ssm_out") {0.026}
                else if name.contains("attn_qkv") || name.contains("attn_gate") {0.044}
                else if name.contains("ffn_down") {if name.contains("blk.0."){0.038}else{0.052}}
                else if name.contains("ffn_gate") || name.contains("ffn_up") {0.074} else {0.040};
            let legacy=TurboWhtMatrix::encode_with_hessian(4,64,64,&w,None,|r| {
                if ms[r]>(if special {3.0}else{3.2})*median {0.}
                else if special {base} else {base*(median/ms[r]).clamp(0.45,1.50)}
            }).unwrap();
            let serialized=|t| write_vxt_model(&[],&[VxtEntry{name:name.into(),tensor:t}]).unwrap();
            assert_eq!(serialized(one),serialized(PackedTensor::TurboWht(legacy)),"legacy bytes {name}");
            let zero=encode_gguf_tensor_with_layers(&gg,meta,LlmPackMode::Turbo,0,0,&TurboCalibPlan::default(),&Cb4CalibPlan::default(),0.).unwrap();
            let direct=TurboWhtMatrix::encode_with_hessian(4,64,64,&gg.extract_f32(meta).unwrap(),None,|_|0.).unwrap();
            assert_eq!(zero.decode_f32(),PackedTensor::TurboWht(direct).decode_f32(),"{name}");
        }
    }
    #[test]
    fn invalid_entropy_scales_rejected_before_input() {
        for scale in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,-1.] {
            assert!(pack_gguf_model_to_vxt_with_entropy_scale(&[],LlmPackMode::Turbo,1,scale).is_err());
        }
        assert!(pack_gguf_model_to_vxt_with_entropy_scale(&[],LlmPackMode::Lossless,1,0.).is_err());
    }
}
