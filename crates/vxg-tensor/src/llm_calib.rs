//! Default analytical Sink-SmoothQuant planner plus explicit observed-profile diagnostic.
//! The optional profile replaces only block-isotropic importance, never smoothing.
//! - Identifies true residual attention-sink channels (`writer_rms / median > 1.8`) across all
//!   residual writers (`ffn_down`, `ssm_out`, `attn_output`) and migrates sink outlier scale (`<= 4x`)
//!   into reader columns without disturbing normal semantic channels.
//! - Computes domain-agnostic 32-column Hadamard block energy weights (`[0.65, 2.0]`, mean `1.0`)
//!   from `RMSNorm` gains (`attn_norm`, `post_attention_norm`, `ssm_norm`, `output_norm`), preserving
//!   exact intra-block Rademacher isotropy with zero calibration-prompt overfitting.

use crate::cb4_calib::{
    build_cb4_calib_plan_from_source, cb4_col_amp, Cb4CalibPlan, Cb4TensorSource,
};
use crate::gguf::GgufView;
use crate::gguf_meta::ModelConfig;
use crate::wht_turbo::TurboWhtMatrix;
use crate::Result;

#[derive(Default)]
pub(crate) struct TurboCalibPlan {
    pub attn_smooth: Vec<Vec<(usize, f32)>>,
    pub post_smooth: Vec<Vec<(usize, f32)>>,
    pub h_attn: Vec<Vec<f32>>,
    pub h_ssm_out: Vec<Vec<f32>>,
    pub h_attn_out: Vec<Vec<f32>>,
    pub h_ffn_gu: Vec<Vec<f32>>,
    pub h_ffn_down: Vec<Vec<f32>>,
    pub h_embd: Vec<f32>,
    pub h_lm_head: Vec<f32>,
    pub ch_plan: Option<Cb4CalibPlan>,
}

pub(crate) fn turbo_ch_amp<'a>(
    plan: &'a TurboCalibPlan,
    name: &str,
    layer_idx: Option<usize>,
    cols: usize,
) -> Option<&'a [f32]> {
    plan.ch_plan
        .as_ref()
        .and_then(|cp| cb4_col_amp(cp, name, layer_idx, cols))
}

fn block_isotropic_hessian(ch_amp: &[f32]) -> Vec<f32> {
    let cols = ch_amp.len();
    let padded_cols = cols.div_ceil(256) * 256;
    if cols < 32 {
        return vec![1.0f32; padded_cols];
    }
    let active_cols = cols.div_ceil(32) * 32;
    let n_blks = active_cols / 32;
    let mut blk_e = vec![0.0f64; n_blks];
    for (b, slot) in blk_e.iter_mut().enumerate() {
        let c0 = b * 32;
        let c1 = (c0 + 32).min(cols);
        let mut s2 = 0.0f64;
        for &a in &ch_amp[c0..c1] {
            s2 += (a as f64) * (a as f64);
        }
        *slot = s2 * (1.0 / 32.0);
    }
    let tot: f64 = blk_e.iter().sum();
    if tot <= 1e-30 {
        return vec![1.0f32; padded_cols];
    }
    let inv_mean = (n_blks as f64) / tot;
    let mut h = vec![1.0f32; padded_cols];
    for (b, &be) in blk_e.iter().enumerate() {
        let reg = (0.45 + 0.55 * (be * inv_mean)).clamp(0.65, 2.0) as f32;
        h[b * 32..(b + 1) * 32].fill(reg);
    }
    let h_sum: f64 = h[..active_cols].iter().map(|&v| v as f64).sum();
    if h_sum > 0.0 {
        let scale = (active_cols as f64 / h_sum) as f32;
        for v in &mut h[..active_cols] {
            *v *= scale;
        }
    }
    h
}

pub(crate) fn run_turbo_calibration(gg: &GgufView<'_>, cfg: &ModelConfig) -> TurboCalibPlan {
    run_turbo_calibration_from_source(gg, cfg)
}

pub(crate) fn run_turbo_calibration_from_source(
    src: &dyn Cb4TensorSource,
    cfg: &ModelConfig,
) -> TurboCalibPlan {
    let n_embd = cfg.n_embd;
    let n_layer = cfg.n_layer;
    if n_embd < 64 || n_layer == 0 {
        return TurboCalibPlan::default();
    }
    let mut plan = TurboCalibPlan {
        attn_smooth: vec![Vec::new(); n_layer],
        post_smooth: vec![Vec::new(); n_layer],
        h_attn: vec![Vec::new(); n_layer],
        h_ssm_out: vec![Vec::new(); n_layer],
        h_attn_out: vec![Vec::new(); n_layer],
        h_ffn_gu: vec![Vec::new(); n_layer],
        h_ffn_down: vec![Vec::new(); n_layer],
        h_embd: Vec::new(),
        h_lm_head: Vec::new(),
        ch_plan: None,
    };

    let mut ch_e = vec![0.0f64; n_embd];
    let mut count = 0usize;
    for wname in src.writer_tensors(n_layer) {
        let Some((rows, cols)) = src.matrix_dims(&wname) else {
            continue;
        };
        if rows != n_embd || cols < 64 {
            continue;
        }
        let Some(w) = src.extract_f32(&wname) else {
            continue;
        };
        let ns = cols.div_ceil(8).max(1) as f64;
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let mut s2 = 0.0f64;
            for &v in row.iter().step_by(8) {
                s2 += (v as f64) * (v as f64);
            }
            ch_e[r] += s2 / ns;
        }
        count += 1;
    }
    if count == 0 {
        return plan;
    }

    let rms: Vec<f32> = ch_e
        .iter()
        .map(|&e| ((e / count as f64).sqrt()) as f32)
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[n_embd / 2].max(1e-12);
    let w_rel: Vec<f32> = rms.iter().map(|&v| (v / med).clamp(0.5, 8.0)).collect();

    let eval_norm =
        |norm_name: &str, _hi_scale: f32, has_res: bool| -> (Vec<(usize, f32)>, Vec<f32>) {
            let Some(nw) = src.extract_f32(norm_name) else {
                return (Vec::new(), Vec::new());
            };
            if nw.len() != n_embd {
                return (Vec::new(), Vec::new());
            }
            let mut ns = nw.iter().map(|v| v.abs()).collect::<Vec<_>>();
            ns.sort_by(|a, b| a.total_cmp(b));
            let n_med = ns[n_embd / 2].max(1e-12);

            // For weight-only Hadamard+GPTQ quantization, scaling reader weight columns (`W[:, c] *= g`)
            // violates AM-GM optimality by smearing `g^2` variance across the entire 32-col block; leave
            // column scales unmutated and let intra-block Cholesky GPTQ cancel spikes directly.
            let mut amp = vec![0.0f32; n_embd];
            for c in 0..n_embd {
                let r_mul = if has_res { w_rel[c] } else { 1.0 };
                amp[c] = (nw[c].abs() / n_med) * r_mul;
            }
            (Vec::new(), block_isotropic_hessian(&amp))
        };

    for il in 0..n_layer {
        let pfx = format!("blk.{il}");
        let (as_sc, ah) = eval_norm(&format!("{pfx}.attn_norm.weight"), 4.0, il >= 1);
        plan.attn_smooth[il] = as_sc;
        plan.h_attn[il] = ah;

        let pn_name = if src.has_tensor(&format!("{pfx}.post_attention_norm.weight")) {
            format!("{pfx}.post_attention_norm.weight")
        } else {
            format!("{pfx}.ffn_norm.weight")
        };
        let (ps_sc, ph) = eval_norm(&pn_name, 4.0, true);
        plan.post_smooth[il] = ps_sc;
        plan.h_ffn_gu[il] = ph;

        if cfg.is_recurrent_layer(il) && cfg.ssm_inner_size >= 64 && cfg.ssm_dt_rank > 0 {
            if let Some(sw) = src.extract_f32(&format!("{pfx}.ssm_norm.weight")) {
                let dv = cfg.ssm_inner_size / cfg.ssm_dt_rank;
                if sw.len() == dv && dv >= 32 {
                    let mut full = Vec::with_capacity(cfg.ssm_inner_size);
                    for _ in 0..cfg.ssm_dt_rank {
                        full.extend(sw.iter().map(|v| v.abs()));
                    }
                    plan.h_ssm_out[il] = block_isotropic_hessian(&full);
                }
            }
        }
    }

    let (_, h_out) = eval_norm("output_norm.weight", 1.0, true);
    if !h_out.is_empty() {
        if src.has_tensor("output.weight") {
            plan.h_lm_head = h_out;
        } else {
            plan.h_embd = h_out.iter().map(|&hv| 0.35 + 0.65 * hv).collect();
        }
    }

    let ch_plan = build_cb4_calib_plan_from_source(src, cfg);
    if !ch_plan.amp_embd.is_empty() && plan.h_embd.is_empty() {
        plan.h_embd = block_isotropic_hessian(&ch_plan.amp_embd);
    }
    for il in 0..n_layer {
        if let Some(a) = ch_plan.amp_attn_out.get(il).filter(|v| !v.is_empty()) {
            plan.h_attn_out[il] = block_isotropic_hessian(a);
        }
        if let Some(d) = ch_plan.amp_ffn_down.get(il).filter(|v| !v.is_empty()) {
            plan.h_ffn_down[il] = block_isotropic_hessian(d);
        }
    }
    plan.ch_plan = Some(ch_plan);
    plan
}

// Explicit diagnostic profile. Values are second moments in x/g coordinates;
// block isotropy is retained, never a claim of a measured full Hessian.
pub(crate) fn fingerprint(bytes: &[u8]) -> String {
    format!(
        "{:016x}",
        bytes
            .iter()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(*b))
                .wrapping_mul(0x100000001b3))
    )
}
pub(crate) fn activation_manifest(
    bytes: &[u8],
    cfg: &ModelConfig,
    p: &TurboCalibPlan,
) -> serde_json::Value {
    let mut groups = serde_json::Map::new();
    for i in 0..cfg.n_layer {
        for (family, h, scales) in [
            ("attn", &p.h_attn[i], &p.attn_smooth[i]),
            ("ffn_gu", &p.h_ffn_gu[i], &p.post_smooth[i]),
            ("ssm_out", &p.h_ssm_out[i], &Vec::new()),
        ] {
            if !h.is_empty() {
                let cols = if family == "ssm_out" {
                    cfg.ssm_inner_size
                } else {
                    cfg.n_embd
                };
                groups.insert(
                    format!("{family}.{i}"),
                    serde_json::json!({"cols":cols,"scales":scales}),
                );
            }
        }
    }
    serde_json::json!({"version":1,"source_fnv":fingerprint(bytes),"coordinate":"original-linear-input-div-exported-g;block32-isotropic","policy":"0.45+0.55*relative-energy;clamp[0.65,2];active-mean1;padding1","groups":groups})
}
fn observed_h(m: &[f64], cols: usize, samples: u64) -> crate::Result<Vec<f32>> {
    if samples == 0
        || cols < 32
        || m.len() != cols.div_ceil(32)
        || m.iter().any(|x| !x.is_finite() || *x < 0.)
        || (!m.iter().sum::<f64>().is_finite() || m.iter().sum::<f64>() <= 1e-30)
    {
        return Err(crate::Error::Format(
            "invalid activation samples/shape/moments".into(),
        ));
    }
    let mean = m.iter().sum::<f64>() / m.len() as f64;
    let mut h = vec![1.; cols.div_ceil(256) * 256];
    for (b, x) in m.iter().enumerate() {
        h[b * 32..(b + 1) * 32].fill((0.45 + 0.55 * x / mean).clamp(0.65, 2.) as f32);
    }
    let n = cols.div_ceil(32) * 32;
    let scale = (n as f64 / h[..n].iter().map(|x| *x as f64).sum::<f64>()) as f32;
    for x in &mut h[..n] {
        *x *= scale;
    }
    Ok(h)
}
pub(crate) fn apply_activation_profile(
    profile: &serde_json::Value,
    manifest: &serde_json::Value,
    p: &mut TurboCalibPlan,
) -> crate::Result<()> {
    let bad =
        || crate::Error::Format("activation profile identity/version/coverage mismatch".into());
    if profile.get("manifest") != Some(manifest) {
        return Err(bad());
    }
    let groups = manifest["groups"].as_object().ok_or_else(bad)?;
    let stats = profile["stats"].as_object().ok_or_else(bad)?;
    if groups.is_empty() || groups.len() != stats.len() {
        return Err(bad());
    }
    let mut common_samples = None;
    for (key, g) in groups {
        let st = stats.get(key).ok_or_else(bad)?;
        let samples = st["samples"].as_u64().ok_or_else(bad)?;
        if st["cols"] != g["cols"]
            || st["calls"].as_u64() != Some(samples)
            || samples > 256
            || common_samples.is_some_and(|n| n != samples)
        {
            return Err(bad());
        }
        common_samples = Some(samples);
        let vals = st["block_second_moment"].as_array().ok_or_else(bad)?;
        let moments: Vec<f64> = vals
            .iter()
            .map(|x| x.as_f64().ok_or_else(bad))
            .collect::<crate::Result<_>>()?;
        let h = observed_h(
            &moments,
            g["cols"].as_u64().ok_or_else(bad)? as usize,
            samples,
        )?;
        let (family, i) = key.split_once('.').ok_or_else(bad)?;
        let i: usize = i.parse().map_err(|_| bad())?;
        match family {
            "attn" => p.h_attn[i] = h,
            "ffn_gu" => p.h_ffn_gu[i] = h,
            "ssm_out" => p.h_ssm_out[i] = h,
            _ => return Err(bad()),
        }
    }
    Ok(())
}

pub(crate) const EMBD_LAM: f32 = 0.052;
pub(crate) const GU_LAM: f32 = 0.074;
pub(crate) const DOWN_LAM: f32 = 0.052;

#[inline]
pub(crate) fn non_empty_slice(v: &[f32]) -> Option<&[f32]> {
    (!v.is_empty()).then_some(v)
}

/// Returns `(bits, base_lambda)` per tensor role for `VXT-Turbo v3` (`~2.97 bpw` on Qwen3.5-9B).
pub(crate) fn turbo_v3_tensor_spec(
    name: &str,
    layer_idx: Option<usize>,
    n_layer: usize,
    has_output_weight: bool,
) -> (u8, f32) {
    let n_lay = if n_layer > 0 { n_layer } else { 24 };
    if name == "token_embd.weight" {
        return if has_output_weight { (4, 0.024) } else { (6, 0.0) };
    }
    if name == "output.weight" {
        return (6, 0.0);
    }
    let is_kv_or_out = name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight")
        || name.ends_with(".ssm_out.weight");
    if is_kv_or_out {
        return (6, 0.0);
    }
    if name.ends_with(".attn_q.weight") {
        return if layer_idx.is_some_and(|il| il <= 3 || il + 2 >= n_lay) { (6, 0.0) } else { (3, 0.0) };
    }
    let is_boundary = layer_idx.is_some_and(|il| il == 0 || il + 2 >= n_lay);
    if name.ends_with(".attn_qkv.weight") {
        if is_boundary {
            (6, 0.0)
        } else if layer_idx.is_some_and(|il| il <= 2 || il + 4 >= n_lay) {
            (3, 0.0)
        } else {
            (4, 0.016)
        }
    } else if name.ends_with(".attn_gate.weight") {
        if is_boundary { (6, 0.0) } else { (4, 0.020) }
    } else if name.ends_with(".ffn_down.weight") {
        if layer_idx.is_some_and(|il| il <= 1 || il + 2 >= n_lay || il % 4 == 3) {
            (3, 0.0)
        } else {
            (4, 0.018)
        }
    } else if name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight") {
        if layer_idx.is_some_and(|il| il <= 1 || il + 2 >= n_lay || il % 4 == 3) {
            (4, 0.016)
        } else if layer_idx.is_some_and(|il| 5 <= il && il + 5 < n_lay && matches!(il % 4, 1 | 2)) {
            (5, 0.012)
        } else {
            (4, 0.024)
        }
    } else {
        (3, 0.0)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_turbo_calibrated_tensor(
    name: &str,
    layer_idx: Option<usize>,
    rows: usize,
    cols: usize,
    f32s: &[f32],
    n_layer: usize,
    plan: &TurboCalibPlan,
    turbo_entropy_scale: f32,
) -> Result<TurboWhtMatrix> {
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
    let ch_amp = turbo_ch_amp(plan, name, layer_idx, cols);
    let is_calibrated = plan.ch_plan.is_some();

    if name == "token_embd.weight" || name == "output.weight" {
        let h = if name == "token_embd.weight" { &plan.h_embd } else { &plan.h_lm_head };
        let head_bits = if is_calibrated { 6 } else { 4 };
        let lam = if head_bits == 6 {
            0.0
        } else if name == "token_embd.weight" {
            EMBD_LAM
        } else {
            0.040
        };
        return TurboWhtMatrix::encode_with_ch_amp(head_bits, rows, cols, f32s, non_empty_slice(h), ch_amp, |r| {
            turbo_entropy_scale * if row_ms[r] > 3.0 * mat_ms { 0.0 } else { lam }
        });
    }

    let h_rot = layer_idx.and_then(|il| {
        if name.ends_with(".attn_qkv.weight") || name.ends_with(".attn_gate.weight")
            || name.ends_with(".attn_q.weight") || name.ends_with(".attn_k.weight") || name.ends_with(".attn_v.weight")
        {
            plan.h_attn.get(il).and_then(|v| non_empty_slice(v))
        } else if name.ends_with(".ssm_out.weight") {
            plan.h_ssm_out.get(il).and_then(|v| non_empty_slice(v))
        } else if name.ends_with(".attn_output.weight") {
            plan.h_attn_out.get(il).and_then(|v| non_empty_slice(v))
        } else if name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight") {
            plan.h_ffn_gu.get(il).and_then(|v| non_empty_slice(v))
        } else if name.ends_with(".ffn_down.weight") {
            plan.h_ffn_down.get(il).and_then(|v| non_empty_slice(v))
        } else {
            None
        }
    });

    let is_kv_or_out = name.ends_with(".attn_k.weight") || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight") || name.ends_with(".ssm_out.weight");
    let is_q = name.ends_with(".attn_q.weight");
    let is_qkv_gate = name.ends_with(".attn_qkv.weight") || name.ends_with(".attn_gate.weight");
    let is_fg_fu = name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight");
    let is_ffn_down = name.ends_with(".ffn_down.weight");
    let is_edge = layer_idx.is_some_and(|il| il <= 1 || il + 2 >= n_lay);
    let is_boundary = layer_idx.is_some_and(|il| il == 0 || il + 2 >= n_lay);
    let is_anchor = is_edge || layer_idx.is_some_and(|il| il % 4 == 3);

    if is_calibrated {
        let (bits, base_lam): (u8, f32) = if is_kv_or_out {
            (6, 0.0)
        } else if is_q {
            if is_edge { (6, 0.0) } else { (3, 0.0) }
        } else if is_qkv_gate {
            if is_boundary { (6, 0.0) } else { (3, 0.0) }
        } else if is_ffn_down {
            if is_anchor { (3, 0.0) } else { (4, 0.014) }
        } else if is_fg_fu {
            (4, if is_anchor { 0.014 } else { 0.020 })
        } else {
            (3, 0.0)
        };
        TurboWhtMatrix::encode_with_ch_amp(bits, rows, cols, f32s, h_rot, ch_amp, |r| {
            if bits != 4 || row_ms[r] > 3.0 * mat_ms {
                0.0
            } else {
                turbo_entropy_scale * base_lam * (mat_ms / row_ms[r]).clamp(0.50, 1.00)
            }
        })
    } else {
        let base_lam: f32 = if is_kv_or_out || is_q {
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
        TurboWhtMatrix::encode_with_ch_amp(4, rows, cols, f32s, h_rot, ch_amp, |r| {
            if row_ms[r] > 3.2 * mat_ms {
                0.0
            } else {
                turbo_entropy_scale * (base_lam * (mat_ms / row_ms[r]).clamp(0.45, 1.50))
            }
        })
    }
}
#[cfg(test)]
mod activation_tests {
    use super::*;
    #[test]
    fn normalization_and_padding() {
        let h = observed_h(&[1., 9.], 64, 1).unwrap();
        assert_eq!(h.len(), 256);
        assert!(h[0] < h[32]);
        assert!((h[..64].iter().sum::<f32>() / 64. - 1.).abs() < 1e-6);
        assert_eq!(h[64], 1.);
    }
    #[test]
    fn rejects_invalid() {
        for (v, n) in [
            (vec![0.], 1),
            (vec![f64::NAN], 1),
            (vec![1.], 0),
            (vec![-1.], 1),
        ] {
            assert!(observed_h(&v, 32, n).is_err());
        }
    }
    #[test]
    fn profile_rejects_shape_counts_and_provenance_mismatch() {
        let m = serde_json::json!({"version":1,"source_fnv":"source","groups":{"attn.0":{"cols":32,"scales":[[0,2.0]]}}});
        let valid = serde_json::json!({"manifest":m,"stats":{"attn.0":{"cols":32,"calls":8,"samples":8,"block_second_moment":[1.0]}}});
        let plan = || TurboCalibPlan {
            h_attn: vec![vec![1.; 256]],
            ..TurboCalibPlan::default()
        };
        assert!(apply_activation_profile(&valid, &m, &mut plan()).is_ok());
        for field in ["source_fnv", "version", "groups"] {
            let mut v = valid.clone();
            v["manifest"][field] = serde_json::Value::Null;
            assert!(apply_activation_profile(&v, &m, &mut plan()).is_err());
        }
        for (field, val) in [
            ("cols", serde_json::json!(64)),
            ("calls", serde_json::json!(9)),
            ("samples", serde_json::json!(0)),
            ("block_second_moment", serde_json::json!([1., 2.])),
        ] {
            let mut v = valid.clone();
            v["stats"]["attn.0"][field] = val;
            assert!(apply_activation_profile(&v, &m, &mut plan()).is_err());
        }
    }
    #[test]
    fn rejects_identity_and_missing_coverage() {
        let m = serde_json::json!({"groups":{"attn.0":{"cols":32}}});
        assert!(apply_activation_profile(
            &serde_json::json!({}),
            &m,
            &mut TurboCalibPlan::default()
        )
        .is_err());
        assert!(apply_activation_profile(
            &serde_json::json!({"manifest":m,"stats":{}}),
            &m,
            &mut TurboCalibPlan::default()
        )
        .is_err());
    }
}
