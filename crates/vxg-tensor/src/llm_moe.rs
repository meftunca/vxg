//! Mixture-of-Experts (`MoE`) FFN layer support for `Qwen3-MoE`, `Mixtral`, and `DeepSeek-MoE`.
//! Supports sparse Top-K softmax routing, per-expert SwiGLU/GeGLU (`gate`, `up`, `down`), and
//! optional gated shared experts (`shexp`), reusing `MatWeight` for both CPU and per-matrix Metal GEMV.

use std::collections::HashMap;

use crate::cuda::CudaEngine;
use crate::format::PackedTensor;
use crate::llm_cpu::MatWeight;
use crate::llm_ops::{sigmoid, silu};
use crate::llm_prefill::gelu_tanh_f32;
use crate::metal::MetalEngine;
use crate::{Error, Result};

pub(crate) struct MoeExpert {
    pub(crate) gate: MatWeight,
    pub(crate) up: MatWeight,
    pub(crate) down: MatWeight,
}

pub(crate) struct MoeShared {
    pub(crate) gate: MatWeight,
    pub(crate) up: MatWeight,
    pub(crate) down: MatWeight,
    pub(crate) router_gate: Option<Vec<f32>>,
}

pub(crate) struct MoeFfn {
    pub(crate) router: MatWeight,
    pub(crate) experts: Vec<MoeExpert>,
    pub(crate) shared: Option<Box<MoeShared>>,
    pub(crate) top_k: usize,
    pub(crate) norm_topk_prob: bool,
}

impl MoeFfn {
    /// Extracts a per-layer MoE FFN from `map` if `blk.{il}.ffn_gate_inp.weight` is present.
    pub(crate) fn extract(
        pfx: &str,
        default_top_k: usize,
        map: &mut HashMap<String, PackedTensor>,
    ) -> Result<Option<Self>> {
        let Some(router_t) = map.remove(&format!("{pfx}.ffn_gate_inp.weight")) else {
            return Ok(None);
        };
        let n_experts = router_t.rows();
        let router = MatWeight::cpu_only(router_t);
        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let g = map
                .remove(&format!("{pfx}.ffn_gate.{e}.weight"))
                .ok_or_else(|| Error::Format(format!("missing MoE expert {pfx}.ffn_gate.{e}.weight")))?;
            let u = map
                .remove(&format!("{pfx}.ffn_up.{e}.weight"))
                .ok_or_else(|| Error::Format(format!("missing MoE expert {pfx}.ffn_up.{e}.weight")))?;
            let d = map
                .remove(&format!("{pfx}.ffn_down.{e}.weight"))
                .ok_or_else(|| Error::Format(format!("missing MoE expert {pfx}.ffn_down.{e}.weight")))?;
            experts.push(MoeExpert {
                gate: MatWeight::cpu_only(g),
                up: MatWeight::cpu_only(u),
                down: MatWeight::cpu_only(d),
            });
        }
        let shared = if let (Some(sg), Some(su), Some(sd)) = (
            map.remove(&format!("{pfx}.ffn_gate_shexp.weight")),
            map.remove(&format!("{pfx}.ffn_up_shexp.weight")),
            map.remove(&format!("{pfx}.ffn_down_shexp.weight")),
        ) {
            let rg = map
                .remove(&format!("{pfx}.ffn_gate_inp_shexp.weight"))
                .map(|t| t.decode_f32());
            Some(Box::new(MoeShared {
                gate: MatWeight::cpu_only(sg),
                up: MatWeight::cpu_only(su),
                down: MatWeight::cpu_only(sd),
                router_gate: rg,
            }))
        } else {
            None
        };
        let top_k = if default_top_k > 0 {
            default_top_k
        } else if n_experts >= 64 {
            8
        } else {
            2
        }
        .min(n_experts.max(1));
        Ok(Some(Self {
            router,
            experts,
            shared,
            top_k,
            norm_topk_prob: true,
        }))
    }

    pub(crate) fn upload_cuda(&mut self, eng: &CudaEngine) -> Result<()> {
        self.router.upload_cuda(eng)?;
        for exp in &mut self.experts {
            exp.gate.upload_cuda(eng)?;
            exp.up.upload_cuda(eng)?;
            exp.down.upload_cuda(eng)?;
        }
        if let Some(sh) = self.shared.as_deref_mut() {
            sh.gate.upload_cuda(eng)?;
            sh.up.upload_cuda(eng)?;
            sh.down.upload_cuda(eng)?;
        }
        Ok(())
    }

    /// Runs sparse Top-K MoE forward for a single hidden state vector `x[dim]` -> `out[dim]`.
    pub(crate) fn forward(
        &self,
        x: &[f32],
        out: &mut [f32],
        use_gelu: bool,
        mut metal: Option<&mut MetalEngine>,
        cuda: Option<&CudaEngine>,
        threads: usize,
    ) -> Result<()> {
        let n_exp = self.experts.len();
        let dim = out.len();
        out.fill(0.0);
        if n_exp == 0 {
            return Ok(());
        }
        let mut r_logits = vec![0.0f32; n_exp];
        self.router.forward(x, &mut r_logits, metal.as_deref_mut(), cuda, threads)?;

        let max_l = r_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = vec![0.0f32; n_exp];
        let mut sum_p = 0.0f32;
        for i in 0..n_exp {
            let e = (r_logits[i] - max_l).exp();
            probs[i] = e;
            sum_p += e;
        }
        let inv_s = 1.0 / sum_p.max(1e-12);
        for p in &mut probs {
            *p *= inv_s;
        }

        let mut order: Vec<usize> = (0..n_exp).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then_with(|| a.cmp(&b)));
        let k = self.top_k.min(n_exp);
        let top_sum: f32 = if self.norm_topk_prob {
            order[..k].iter().map(|&i| probs[i]).sum::<f32>().max(1e-12)
        } else {
            1.0
        };

        let mut exp_out = vec![0.0f32; dim];
        for &ei in &order[..k] {
            let w = probs[ei] / top_sum;
            let exp = &self.experts[ei];
            let n_ff = exp.gate.rows();
            let (mut g_buf, mut u_buf) = (vec![0.0f32; n_ff], vec![0.0f32; n_ff]);
            exp.gate.forward(x, &mut g_buf, metal.as_deref_mut(), cuda, threads)?;
            exp.up.forward(x, &mut u_buf, metal.as_deref_mut(), cuda, threads)?;
            if use_gelu {
                for j in 0..n_ff { g_buf[j] = gelu_tanh_f32(g_buf[j]) * u_buf[j]; }
            } else {
                for j in 0..n_ff { g_buf[j] = silu(g_buf[j]) * u_buf[j]; }
            }
            exp.down.forward(&g_buf, &mut exp_out, metal.as_deref_mut(), cuda, threads)?;
            for j in 0..dim {
                out[j] = w.mul_add(exp_out[j], out[j]);
            }
        }

        if let Some(sh) = self.shared.as_deref() {
            let n_ff = sh.gate.rows();
            let (mut g_buf, mut u_buf) = (vec![0.0f32; n_ff], vec![0.0f32; n_ff]);
            sh.gate.forward(x, &mut g_buf, metal.as_deref_mut(), cuda, threads)?;
            sh.up.forward(x, &mut u_buf, metal.as_deref_mut(), cuda, threads)?;
            if use_gelu {
                for j in 0..n_ff { g_buf[j] = gelu_tanh_f32(g_buf[j]) * u_buf[j]; }
            } else {
                for j in 0..n_ff { g_buf[j] = silu(g_buf[j]) * u_buf[j]; }
            }
            sh.down.forward(&g_buf, &mut exp_out, metal.as_deref_mut(), cuda, threads)?;
            let scale = if let Some(rg) = sh.router_gate.as_deref() {
                let mut dot = 0.0f32;
                for (a, b) in x.iter().zip(rg.iter()) {
                    dot = a.mul_add(*b, dot);
                }
                sigmoid(dot)
            } else {
                1.0
            };
            for j in 0..dim {
                out[j] = scale.mul_add(exp_out[j], out[j]);
            }
        }
        Ok(())
    }
}

/// Maps HuggingFace MoE layer weight suffixes (`mlp.gate.*`, `mlp.experts.{e}.*`,
/// `block_sparse_moe.*`, `mlp.shared_expert.*`) to canonical `.vxt` / GGUF tensor names.
pub(crate) fn map_hf_moe_suffix(il: usize, suffix: &str) -> Option<String> {
    match suffix {
        "mlp.gate.weight" | "block_sparse_moe.gate.weight" => {
            return Some(format!("blk.{il}.ffn_gate_inp.weight"));
        }
        "mlp.shared_expert.gate_proj.weight" => {
            return Some(format!("blk.{il}.ffn_gate_shexp.weight"));
        }
        "mlp.shared_expert.up_proj.weight" => {
            return Some(format!("blk.{il}.ffn_up_shexp.weight"));
        }
        "mlp.shared_expert.down_proj.weight" => {
            return Some(format!("blk.{il}.ffn_down_shexp.weight"));
        }
        "mlp.shared_expert_gate.weight" => {
            return Some(format!("blk.{il}.ffn_gate_inp_shexp.weight"));
        }
        _ => {}
    }
    let rest = suffix
        .strip_prefix("mlp.experts.")
        .or_else(|| suffix.strip_prefix("block_sparse_moe.experts."))?;
    let (e_str, proj) = rest.split_once('.')?;
    let e: usize = e_str.parse().ok()?;
    let mapped = match proj {
        "gate_proj.weight" | "w1.weight" => "ffn_gate",
        "up_proj.weight" | "w3.weight" => "ffn_up",
        "down_proj.weight" | "w2.weight" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{il}.{mapped}.{e}.weight"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{map_hf_moe_suffix, MoeFfn};
    use crate::format::PackedTensor;
    use crate::llm_ops::silu;

    #[test]
    fn moe_topk_routing_and_hf_name_mapping_are_exact() {
        assert_eq!(map_hf_moe_suffix(3, "mlp.gate.weight").as_deref(), Some("blk.3.ffn_gate_inp.weight"));
        assert_eq!(map_hf_moe_suffix(3, "mlp.experts.7.gate_proj.weight").as_deref(), Some("blk.3.ffn_gate.7.weight"));
        assert_eq!(map_hf_moe_suffix(1, "block_sparse_moe.experts.2.w2.weight").as_deref(), Some("blk.1.ffn_down.2.weight"));
        assert_eq!(map_hf_moe_suffix(0, "mlp.shared_expert.up_proj.weight").as_deref(), Some("blk.0.ffn_up_shexp.weight"));

        let mut map = HashMap::new();
        // 3 experts, dim=2, n_ff=2, top_k=2; router selects experts 0 and 2 strongly.
        map.insert("blk.0.ffn_gate_inp.weight".into(), PackedTensor::RawF32 {
            rows: 3, cols: 2, data: vec![5.0, 0.0, -5.0, 0.0, 4.0, 0.0],
        });
        for (e, scale) in [(0usize, 1.0f32), (1, 100.0), (2, 2.0)] {
            map.insert(format!("blk.0.ffn_gate.{e}.weight"), PackedTensor::RawF32 { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] });
            map.insert(format!("blk.0.ffn_up.{e}.weight"), PackedTensor::RawF32 { rows: 2, cols: 2, data: vec![scale, 0.0, 0.0, scale] });
            map.insert(format!("blk.0.ffn_down.{e}.weight"), PackedTensor::RawF32 { rows: 2, cols: 2, data: vec![1.0, 0.0, 0.0, 1.0] });
        }
        let moe = MoeFfn::extract("blk.0", 2, &mut map).unwrap().unwrap();
        let x = [1.0f32, 0.5];
        let mut out = [0.0f32; 2];
        moe.forward(&x, &mut out, false, None, None, 1).unwrap();

        let p0 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let p2 = 1.0 - p0;
        let expected0 = (p0 * 1.0 + p2 * 2.0) * silu(1.0);
        let expected1 = (p0 * 1.0 + p2 * 2.0) * silu(0.5) * 0.5;
        assert!((out[0] - expected0).abs() < 1e-4, "got={} exp={}", out[0], expected0);
        assert!((out[1] - expected1).abs() < 1e-4, "got={} exp={}", out[1], expected1);
    }
}
