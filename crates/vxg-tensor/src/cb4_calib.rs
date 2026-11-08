//! Zero-shot analytical `CB4-v2` column-amplitude (`col_amp`) & sensitivity-budget planner:
//! - Extracts residual-writer channel RMS (`ffn_down`, `ssm_out`, `attn_output`) to detect
//!   super-weight attention-sink channels (e.g., `Qwen3.5-2B` ch `1201` at `3.38x` RMS) without
//!   touching any evaluation or calibration prompts.
//! - Propagates `RMSNorm` gains (`attn_norm`, `post_attention_norm`, `output_norm`) and 2-hop
//!   row RMS (`SwiGLU` `ffn_gate * ffn_up` -> `ffn_down`, `DeltaNet` `ssm_norm * attn_gate` ->
//!   `ssm_out`, `GQA` `attn_v` -> `attn_output`) into mean-`1.0` per-column RMS weights.

use crate::gguf::{GgufTensorMeta, GgufView};
use crate::gguf_meta::ModelConfig;

#[derive(Default)]
pub(crate) struct Cb4CalibPlan {
    pub amp_attn: Vec<Vec<f32>>,
    pub amp_post: Vec<Vec<f32>>,
    pub amp_ffn_down: Vec<Vec<f32>>,
    pub amp_ssm_out: Vec<Vec<f32>>,
    pub amp_attn_out: Vec<Vec<f32>>,
    pub amp_embd: Vec<f32>,
    pub amp_lm_head: Vec<f32>,
    pub has_output_weight: bool,
}

fn find_tensor<'a>(gg: &'a GgufView<'a>, name: &str) -> Option<&'a GgufTensorMeta> {
    gg.tensors.iter().find(|m| m.name == name)
}

fn skip_layer(name: &str, n_layer: usize) -> bool {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .and_then(|(s, _)| s.parse::<usize>().ok())
        .is_some_and(|il| il >= n_layer)
}

pub(crate) fn normalize_amp(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let sum0: f64 = v
        .iter()
        .map(|&x| if x.is_finite() && x > 0.0 { x as f64 } else { 0.0 })
        .sum();
    if sum0 <= 1e-30 {
        v.fill(1.0);
        return;
    }
    let inv0 = (v.len() as f64) / sum0;
    for x in v.iter_mut() {
        let s = if x.is_finite() && *x > 0.0 {
            ((*x as f64) * inv0) as f32
        } else {
            1.0
        };
        *x = s.clamp(0.35, 6.0);
    }
    let sum1: f64 = v.iter().map(|&x| x as f64).sum();
    if sum1 > 1e-30 {
        let inv1 = ((v.len() as f64) / sum1) as f32;
        for x in v.iter_mut() {
            *x *= inv1;
        }
    }
}

fn weighted_row_rms(row: &[f32], in_amp: &[f32]) -> f32 {
    let cols = row.len();
    if cols == 0 {
        return 1.0;
    }
    let step = if cols >= 512 { 2 } else { 1 };
    let mut s2 = 0.0f64;
    let mut cnt = 0usize;
    let mut c = 0usize;
    while c < cols {
        let a = in_amp.get(c).copied().unwrap_or(1.0) as f64;
        let prod = (row[c] as f64) * a;
        s2 += prod * prod;
        cnt += 1;
        c += step;
    }
    ((s2 / (cnt.max(1) as f64)).sqrt() as f32).max(1e-12)
}

pub(crate) trait Cb4TensorSource: Sync {
    fn has_tensor(&self, name: &str) -> bool;
    fn writer_tensors(&self, n_layer: usize) -> Vec<String>;
    fn matrix_dims(&self, name: &str) -> Option<(usize, usize)>;
    fn extract_f32(&self, name: &str) -> Option<Vec<f32>>;
}

impl Cb4TensorSource for GgufView<'_> {
    fn has_tensor(&self, name: &str) -> bool {
        find_tensor(self, name).is_some()
    }

    fn writer_tensors(&self, n_layer: usize) -> Vec<String> {
        self.tensors
            .iter()
            .filter(|m| {
                !skip_layer(&m.name, n_layer)
                    && (m.name.ends_with(".ffn_down.weight")
                        || m.name.ends_with(".ssm_out.weight")
                        || m.name.ends_with(".attn_output.weight"))
            })
            .map(|m| m.name.clone())
            .collect()
    }

    fn matrix_dims(&self, name: &str) -> Option<(usize, usize)> {
        find_tensor(self, name).and_then(|m| m.matrix_dims())
    }

    fn extract_f32(&self, name: &str) -> Option<Vec<f32>> {
        find_tensor(self, name).and_then(|m| GgufView::extract_f32(self, m).ok())
    }
}

pub(crate) fn build_cb4_calib_plan(gg: &GgufView<'_>, cfg: &ModelConfig) -> Cb4CalibPlan {
    build_cb4_calib_plan_from_source(gg, cfg)
}

pub(crate) fn build_cb4_calib_plan_from_source(
    src: &dyn Cb4TensorSource,
    cfg: &ModelConfig,
) -> Cb4CalibPlan {
    let (n_embd, n_layer) = (cfg.n_embd, cfg.n_layer);
    if n_embd < 64 || n_layer == 0 {
        return Cb4CalibPlan::default();
    }
    let has_output_weight = src.has_tensor("output.weight");
    let mut ch_e = vec![0.0f64; n_embd];
    let mut writer_cnt = 0usize;
    for wname in src.writer_tensors(n_layer) {
        let Some((rows, cols)) = src.matrix_dims(&wname) else {
            continue;
        };
        if rows != n_embd || cols < 64 {
            continue;
        };
        let Some(w) = src.extract_f32(&wname) else {
            continue;
        };
        let ns = cols.div_ceil(8).max(1) as f64;
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            let s2: f64 = row.iter().step_by(8).map(|&v| (v as f64) * (v as f64)).sum();
            ch_e[r] += s2 / ns;
        }
        writer_cnt += 1;
    }

    let w_rel: Vec<f32> = if writer_cnt > 0 {
        let rms: Vec<f32> = ch_e
            .iter()
            .map(|&e| ((e / writer_cnt as f64).sqrt()) as f32)
            .collect();
        let mut sorted = rms.clone();
        sorted.sort_by(f32::total_cmp);
        let med = sorted[n_embd / 2].max(1e-12);
        rms.iter().map(|&v| (v / med).clamp(0.5, 8.0)).collect()
    } else {
        vec![1.0; n_embd]
    };

    let eval_norm_amp = |norm_name: &str, has_res: bool| -> Vec<f32> {
        let Some(nw) = src.extract_f32(norm_name) else {
            return vec![1.0; n_embd];
        };
        if nw.len() != n_embd {
            return vec![1.0; n_embd];
        }
        let mut amp: Vec<f32> = (0..n_embd)
            .map(|c| nw[c].abs() * if has_res { w_rel[c] } else { 1.0 })
            .collect();
        normalize_amp(&mut amp);
        amp
    };

    type LayerAmps = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let mut layer_slots: Vec<LayerAmps> = (0..n_layer)
        .map(|_| (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()))
        .collect();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4)
        .min(n_layer);
    let chunk = n_layer.div_ceil(workers);

    std::thread::scope(|s| {
        for (ci, slot_chunk) in layer_slots.chunks_mut(chunk).enumerate() {
            let il_base = ci * chunk;
            s.spawn(move || {
                for (di, slot) in slot_chunk.iter_mut().enumerate() {
                    let il = il_base + di;
                    let pfx = format!("blk.{il}");
                    let a_attn = eval_norm_amp(&format!("{pfx}.attn_norm.weight"), il >= 1);
                    let pn_name = if src.has_tensor(&format!("{pfx}.post_attention_norm.weight")) {
                        format!("{pfx}.post_attention_norm.weight")
                    } else {
                        format!("{pfx}.ffn_norm.weight")
                    };
                    let a_post = eval_norm_amp(&pn_name, true);

                    let mut a_down = Vec::new();
                    let g_name = format!("{pfx}.ffn_gate.weight");
                    let u_name = format!("{pfx}.ffn_up.weight");
                    if src.matrix_dims(&g_name) == Some((cfg.n_ff, n_embd))
                        && src.matrix_dims(&u_name) == Some((cfg.n_ff, n_embd))
                    {
                        if let (Some(gw), Some(uw)) =
                            (src.extract_f32(&g_name), src.extract_f32(&u_name))
                        {
                            a_down = (0..cfg.n_ff)
                                .map(|k| {
                                    weighted_row_rms(&gw[k * n_embd..(k + 1) * n_embd], &a_post)
                                        * weighted_row_rms(
                                            &uw[k * n_embd..(k + 1) * n_embd],
                                            &a_post,
                                        )
                                })
                                .collect();
                            normalize_amp(&mut a_down);
                        }
                    }

                    let mut a_ssm = Vec::new();
                    let mut a_ao = Vec::new();
                    if cfg.is_recurrent_layer(il) && cfg.ssm_inner_size >= 64 && cfg.ssm_dt_rank > 0
                    {
                        let dv = cfg.ssm_inner_size / cfg.ssm_dt_rank;
                        let s_name = format!("{pfx}.ssm_norm.weight");
                        let ag_name = format!("{pfx}.attn_gate.weight");
                        if src.matrix_dims(&ag_name) == Some((cfg.ssm_inner_size, n_embd)) {
                            if let (Some(sw), Some(agw)) =
                                (src.extract_f32(&s_name), src.extract_f32(&ag_name))
                            {
                                if sw.len() == dv {
                                    a_ssm = (0..cfg.ssm_inner_size)
                                        .map(|k| {
                                            sw[k % dv].abs()
                                                * weighted_row_rms(
                                                    &agw[k * n_embd..(k + 1) * n_embd],
                                                    &a_attn,
                                                )
                                        })
                                        .collect();
                                    normalize_amp(&mut a_ssm);
                                }
                            }
                        }
                    } else if cfg.n_head > 0 && cfg.n_head_kv > 0 && cfg.head_dim > 0 {
                        let (nh, nkv, hd) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim);
                        let kv_mul = (nh / nkv).max(1);
                        let v_name = format!("{pfx}.attn_v.weight");
                        if src.matrix_dims(&v_name) == Some((nkv * hd, n_embd)) {
                            if let Some(vw) = src.extract_f32(&v_name) {
                                a_ao = (0..nh * hd)
                                    .map(|k| {
                                        let (h, d) = (k / hd, k % hd);
                                        let vr = (h / kv_mul).min(nkv - 1) * hd + d;
                                        weighted_row_rms(
                                            &vw[vr * n_embd..(vr + 1) * n_embd],
                                            &a_attn,
                                        )
                                    })
                                    .collect();
                                normalize_amp(&mut a_ao);
                            }
                        }
                    }
                    *slot = (a_attn, a_post, a_down, a_ssm, a_ao);
                }
            });
        }
    });

    let mut plan = Cb4CalibPlan {
        amp_attn: Vec::with_capacity(n_layer),
        amp_post: Vec::with_capacity(n_layer),
        amp_ffn_down: Vec::with_capacity(n_layer),
        amp_ssm_out: Vec::with_capacity(n_layer),
        amp_attn_out: Vec::with_capacity(n_layer),
        amp_embd: Vec::new(),
        amp_lm_head: Vec::new(),
        has_output_weight,
    };
    for (a, p, d, s, o) in layer_slots {
        plan.amp_attn.push(a);
        plan.amp_post.push(p);
        plan.amp_ffn_down.push(d);
        plan.amp_ssm_out.push(s);
        plan.amp_attn_out.push(o);
    }

    let lm_amp = eval_norm_amp("output_norm.weight", true);
    if has_output_weight {
        plan.amp_lm_head = lm_amp;
        plan.amp_embd = vec![1.0; n_embd];
    } else {
        let mut tied: Vec<f32> = lm_amp.iter().map(|&a| 0.30 + 0.70 * a).collect();
        normalize_amp(&mut tied);
        plan.amp_embd = tied;
    }
    plan
}

pub(crate) fn cb4_col_amp<'a>(
    plan: &'a Cb4CalibPlan,
    name: &str,
    layer_idx: Option<usize>,
    cols: usize,
) -> Option<&'a [f32]> {
    let check = |s: &'a [f32]| (s.len() == cols).then_some(s);
    if name == "token_embd.weight" {
        return check(&plan.amp_embd);
    }
    if name == "output.weight" {
        return check(&plan.amp_lm_head);
    }
    let il = layer_idx?;
    if name.ends_with(".attn_qkv.weight")
        || name.ends_with(".attn_gate.weight")
        || name.ends_with(".attn_q.weight")
        || name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
    {
        plan.amp_attn.get(il).and_then(|v| check(v))
    } else if name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight") {
        plan.amp_post.get(il).and_then(|v| check(v))
    } else if name.ends_with(".ffn_down.weight") {
        plan.amp_ffn_down.get(il).and_then(|v| check(v))
    } else if name.ends_with(".ssm_out.weight") {
        plan.amp_ssm_out.get(il).and_then(|v| check(v))
    } else if name.ends_with(".attn_output.weight") {
        plan.amp_attn_out.get(il).and_then(|v| check(v))
    } else {
        None
    }
}

pub(crate) fn cb4_target_outlier_frac(
    name: &str,
    layer_idx: Option<usize>,
    n_layer: usize,
    is_recurrent: bool,
    has_output_weight: bool,
) -> f64 {
    if n_layer == 0 {
        return 0.0074;
    }
    if name == "token_embd.weight" {
        return if has_output_weight { 0.0056 } else { 0.0070 };
    }
    if name == "output.weight" {
        return 0.0092;
    }
    let is_edge = layer_idx.is_some_and(|il| il <= 1 || il + 2 >= n_layer);
    let is_anchor = is_edge || !is_recurrent;
    if name.ends_with(".attn_k.weight") || name.ends_with(".attn_v.weight") {
        0.0132
    } else if name.ends_with(".attn_output.weight") {
        0.0112
    } else if name.ends_with(".ssm_out.weight") {
        if is_edge { 0.0112 } else { 0.0100 }
    } else if name.ends_with(".ffn_down.weight") {
        if is_anchor { 0.0102 } else { 0.0086 }
    } else if name.ends_with(".attn_q.weight") {
        0.0094
    } else if name.ends_with(".attn_qkv.weight") {
        if is_edge { 0.0086 } else { 0.0078 }
    } else if name.ends_with(".ffn_up.weight") {
        if is_edge { 0.0076 } else { 0.00685 }
    } else if name.ends_with(".attn_gate.weight") {
        if is_edge { 0.0064 } else { 0.0056 }
    } else if name.ends_with(".ffn_gate.weight") {
        if is_edge { 0.0060 } else { 0.0051 }
    } else {
        0.0074
    }
}
