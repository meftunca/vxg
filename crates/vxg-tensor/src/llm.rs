//! Compressed-domain autoregressive LLM inference engine (`qwen35` hybrid Gated-DeltaNet +
//! Gated-GQA + SwiGLU, plus standard `qwen3` / `qwen2` / `llama` GQA Transformers) executing
//! directly on `.vxt` compressed weights in Apple Metal GPU Unified Memory or multi-threaded CPU.

use std::collections::HashMap;

use crate::format::{read_vxt_model, PackedTensor};
use crate::gguf::GgufView;
use crate::gguf_meta::{parse_gguf_kv, ModelConfig};
use crate::llm_ops::{
    alloc_cpu_state_vecs, rms_norm, silu, step_deltanet_recurrent, step_full_gqa,
};
pub use crate::llm_ops::{compare_logits, sample_logits, LogitEvalMetrics};
use crate::llm_pack::{encode_gguf_tensor, LlmPackMode};
use crate::metal::{GpuPackedTensor, MetalEngine};
use crate::metal_llm::{MetalLlmAttnSpec, MetalLlmLayerSpec, MetalLlmRunner, MetalLlmState};
use crate::mmap::TensorBytes;
use crate::tokenizer::BpeTokenizer;
use crate::{Error, Result};

struct MatWeight {
    cpu: PackedTensor,
    gpu: Option<GpuPackedTensor>,
}

impl MatWeight {
    fn from_packed(cpu: PackedTensor, metal: Option<&MetalEngine>) -> Result<Self> {
        let gpu = match metal {
            Some(eng) => Some(eng.upload_packed(&cpu)?),
            None => None,
        };
        Ok(Self { cpu, gpu })
    }

    #[inline]
    fn rows(&self) -> usize {
        self.cpu.rows()
    }

    #[inline]
    fn forward(
        &self,
        x: &[f32],
        y: &mut [f32],
        metal: Option<&mut MetalEngine>,
        threads: usize,
    ) -> Result<()> {
        if let (Some(eng), Some(g)) = (metal, self.gpu.as_ref()) {
            let _ = g.run_gemv(eng, x, y, 1)?;
        } else {
            self.cpu.gemv_cpu(x, y, threads);
        }
        Ok(())
    }
}

fn forward_multi(
    x: &[f32],
    jobs: &mut [(&MatWeight, &mut [f32])],
    metal: Option<&mut MetalEngine>,
    threads: usize,
) -> Result<()> {
    if let Some(eng) = metal {
        if jobs.iter().all(|(w, _)| w.gpu.is_some()) {
            let mut gpu_jobs: Vec<(&GpuPackedTensor, &mut [f32])> = jobs
                .iter_mut()
                .map(|(w, y)| (w.gpu.as_ref().unwrap(), &mut **y))
                .collect();
            let _ = eng.run_multi_same_x(x, &mut gpu_jobs)?;
            return Ok(());
        }
    }
    for (w, y) in jobs.iter_mut() {
        w.cpu.gemv_cpu(x, y, threads);
    }
    Ok(())
}

enum LayerAttn {
    LinearDeltaNet {
        wqkv: MatWeight,
        wqkv_gate: MatWeight,
        ssm_alpha: MatWeight,
        ssm_beta: MatWeight,
        ssm_out: MatWeight,
        ssm_conv1d: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_dt_bias: Vec<f32>,
        ssm_norm: Vec<f32>,
    },
    FullGqa {
        wq: MatWeight,
        wk: MatWeight,
        wv: MatWeight,
        wo: MatWeight,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        q_bias: Option<Vec<f32>>,
        k_bias: Option<Vec<f32>>,
        v_bias: Option<Vec<f32>>,
        gated_q: bool,
    },
}

struct DecoderLayer {
    attn_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    attn: LayerAttn,
    ffn_gate: MatWeight,
    ffn_up: MatWeight,
    ffn_down: MatWeight,
}

pub struct LlmState {
    pub pos: usize,
    gpu: Option<MetalLlmState>,
    k_caches: Vec<Vec<f32>>,
    v_caches: Vec<Vec<f32>>,
    conv_states: Vec<Vec<f32>>,
    ssm_states: Vec<Vec<f32>>,
}

pub struct LlmModel {
    pub cfg: ModelConfig,
    pub tokenizer: BpeTokenizer,
    pub total_compressed_bytes: usize,
    metal: Option<MetalEngine>,
    metal_llm: Option<MetalLlmRunner>,
    cpu_threads: usize,
    token_embd: MatWeight,
    output_norm: Vec<f32>,
    output_head: Option<MatWeight>,
    layers: Vec<DecoderLayer>,
}

fn try_build_metal_llm(
    eng: &MetalEngine,
    cfg: &ModelConfig,
    layers: &[DecoderLayer],
    output_norm: &[f32],
    head: &MatWeight,
) -> Option<MetalLlmRunner> {
    let head_gpu = head.gpu.as_ref()?;
    let mut specs = Vec::with_capacity(layers.len());
    for ly in layers {
        let attn = match &ly.attn {
            LayerAttn::LinearDeltaNet {
                wqkv,
                wqkv_gate,
                ssm_alpha,
                ssm_beta,
                ssm_out,
                ssm_conv1d,
                ssm_a,
                ssm_dt_bias,
                ssm_norm,
            } => MetalLlmAttnSpec::LinearDeltaNet {
                wqkv: wqkv.gpu.as_ref()?,
                wqkv_gate: wqkv_gate.gpu.as_ref()?,
                ssm_alpha: ssm_alpha.gpu.as_ref()?,
                ssm_beta: ssm_beta.gpu.as_ref()?,
                ssm_out: ssm_out.gpu.as_ref()?,
                ssm_conv1d,
                ssm_a,
                ssm_dt_bias,
                ssm_norm,
            },
            LayerAttn::FullGqa {
                wq,
                wk,
                wv,
                wo,
                q_norm,
                k_norm,
                q_bias,
                k_bias,
                v_bias,
                gated_q,
            } => MetalLlmAttnSpec::FullGqa {
                wq: wq.gpu.as_ref()?,
                wk: wk.gpu.as_ref()?,
                wv: wv.gpu.as_ref()?,
                wo: wo.gpu.as_ref()?,
                q_norm: q_norm.as_deref(),
                k_norm: k_norm.as_deref(),
                q_bias: q_bias.as_deref(),
                k_bias: k_bias.as_deref(),
                v_bias: v_bias.as_deref(),
                gated_q: *gated_q,
            },
        };
        specs.push(MetalLlmLayerSpec {
            attn_norm: &ly.attn_norm,
            post_attn_norm: &ly.post_attn_norm,
            attn,
            ffn_gate: ly.ffn_gate.gpu.as_ref()?,
            ffn_up: ly.ffn_up.gpu.as_ref()?,
            ffn_down: ly.ffn_down.gpu.as_ref()?,
        });
    }
    MetalLlmRunner::new(eng, cfg, &specs, output_norm, head_gpu).ok()
}

impl LlmModel {
    pub fn load_file(path: &str, use_gpu: bool, cpu_threads: usize) -> Result<Self> {
        let mapped = TensorBytes::open(path)?;
        let bytes = mapped.as_slice();
        if bytes.starts_with(b"VXT1") {
            Self::from_vxt_bytes(bytes, use_gpu, cpu_threads)
        } else if bytes.starts_with(b"GGUF") {
            Self::from_gguf_bytes(bytes, use_gpu, cpu_threads)
        } else {
            Err(Error::Unsupported("expected .vxt or .gguf LLM file".into()))
        }
    }

    pub fn from_vxt_bytes(bytes: &[u8], use_gpu: bool, cpu_threads: usize) -> Result<Self> {
        let (meta_kv, entries) = read_vxt_model(bytes)?;
        let cfg_bytes = meta_kv
            .iter()
            .find(|(k, _)| k == "model.config")
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| Error::Format("missing model.config in .vxt (use pack-model)".into()))?;
        let cfg = ModelConfig::from_kv_bytes(cfg_bytes)?;
        let tokenizer = BpeTokenizer::from_vxt_meta(&meta_kv)?;
        let map: HashMap<String, PackedTensor> =
            entries.into_iter().map(|e| (e.name, e.tensor)).collect();
        Self::assemble(cfg, tokenizer, map, bytes.len(), use_gpu, cpu_threads)
    }

    pub fn from_gguf_bytes(bytes: &[u8], use_gpu: bool, cpu_threads: usize) -> Result<Self> {
        let kv = parse_gguf_kv(bytes)?;
        let gg = GgufView::parse(bytes)?;
        let fallback_vocab = gg
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.matrix_dims().map(|(r, _)| r))
            .unwrap_or(0);
        let cfg = ModelConfig::from_gguf_kv(&kv, fallback_vocab)?;
        let tokenizer = BpeTokenizer::from_gguf_kv(&kv)?;
        let mut map = HashMap::with_capacity(gg.tensors.len());
        for m in &gg.tensors {
            if let Some(rest) = m.name.strip_prefix("blk.") {
                if let Some((is, _)) = rest.split_once('.') {
                    if is.parse::<usize>().is_ok_and(|il| il >= cfg.n_layer) {
                        continue;
                    }
                }
            }
            map.insert(
                m.name.clone(),
                encode_gguf_tensor(&gg, m, LlmPackMode::Lossless)?,
            );
        }
        Self::assemble(cfg, tokenizer, map, bytes.len(), use_gpu, cpu_threads)
    }

    fn assemble(
        cfg: ModelConfig,
        tokenizer: BpeTokenizer,
        mut map: HashMap<String, PackedTensor>,
        archive_bytes: usize,
        use_gpu: bool,
        cpu_threads: usize,
    ) -> Result<Self> {
        let metal = if use_gpu {
            MetalEngine::new().ok()
        } else {
            None
        };
        let m_ref = metal.as_ref();

        let take_mat = |map: &mut HashMap<String, PackedTensor>, name: &str| -> Result<MatWeight> {
            let t = map
                .remove(name)
                .ok_or_else(|| Error::Format(format!("missing model tensor '{name}'")))?;
            MatWeight::from_packed(t, m_ref)
        };
        let take_vec = |map: &mut HashMap<String, PackedTensor>, name: &str| -> Result<Vec<f32>> {
            let t = map
                .remove(name)
                .ok_or_else(|| Error::Format(format!("missing model vector '{name}'")))?;
            Ok(t.decode_f32())
        };

        let token_embd = take_mat(&mut map, "token_embd.weight")?;
        let output_norm = take_vec(&mut map, "output_norm.weight")?;
        let output_head = match map.remove("output.weight") {
            Some(t) => Some(MatWeight::from_packed(t, m_ref)?),
            None => None,
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let pfx = format!("blk.{il}");
            let attn_norm = take_vec(&mut map, &format!("{pfx}.attn_norm.weight"))?;
            let post_attn_norm = map
                .remove(&format!("{pfx}.post_attention_norm.weight"))
                .or_else(|| map.remove(&format!("{pfx}.ffn_norm.weight")))
                .ok_or_else(|| Error::Format(format!("missing post-attn norm in layer {il}")))?
                .decode_f32();

            let attn = if cfg.is_recurrent_layer(il) {
                LayerAttn::LinearDeltaNet {
                    wqkv: take_mat(&mut map, &format!("{pfx}.attn_qkv.weight"))?,
                    wqkv_gate: take_mat(&mut map, &format!("{pfx}.attn_gate.weight"))?,
                    ssm_alpha: take_mat(&mut map, &format!("{pfx}.ssm_alpha.weight"))?,
                    ssm_beta: take_mat(&mut map, &format!("{pfx}.ssm_beta.weight"))?,
                    ssm_out: take_mat(&mut map, &format!("{pfx}.ssm_out.weight"))?,
                    ssm_conv1d: take_vec(&mut map, &format!("{pfx}.ssm_conv1d.weight"))?,
                    ssm_a: take_vec(&mut map, &format!("{pfx}.ssm_a"))?,
                    ssm_dt_bias: take_vec(&mut map, &format!("{pfx}.ssm_dt.bias"))?,
                    ssm_norm: take_vec(&mut map, &format!("{pfx}.ssm_norm.weight"))?,
                }
            } else {
                let wq = take_mat(&mut map, &format!("{pfx}.attn_q.weight"))?;
                let wk = take_mat(&mut map, &format!("{pfx}.attn_k.weight"))?;
                let wv = take_mat(&mut map, &format!("{pfx}.attn_v.weight"))?;
                let wo = take_mat(&mut map, &format!("{pfx}.attn_output.weight"))?;
                let gated_q = wq.rows() == cfg.n_head * cfg.head_dim * 2;
                LayerAttn::FullGqa {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm: map
                        .remove(&format!("{pfx}.attn_q_norm.weight"))
                        .map(|t| t.decode_f32()),
                    k_norm: map
                        .remove(&format!("{pfx}.attn_k_norm.weight"))
                        .map(|t| t.decode_f32()),
                    q_bias: map
                        .remove(&format!("{pfx}.attn_q.bias"))
                        .map(|t| t.decode_f32()),
                    k_bias: map
                        .remove(&format!("{pfx}.attn_k.bias"))
                        .map(|t| t.decode_f32()),
                    v_bias: map
                        .remove(&format!("{pfx}.attn_v.bias"))
                        .map(|t| t.decode_f32()),
                    gated_q,
                }
            };

            layers.push(DecoderLayer {
                attn_norm,
                post_attn_norm,
                attn,
                ffn_gate: take_mat(&mut map, &format!("{pfx}.ffn_gate.weight"))?,
                ffn_up: take_mat(&mut map, &format!("{pfx}.ffn_up.weight"))?,
                ffn_down: take_mat(&mut map, &format!("{pfx}.ffn_down.weight"))?,
            });
        }

        let head_ref = output_head.as_ref().unwrap_or(&token_embd);
        let metal_llm = metal
            .as_ref()
            .and_then(|eng| try_build_metal_llm(eng, &cfg, &layers, &output_norm, head_ref));

        Ok(Self {
            cfg,
            tokenizer,
            total_compressed_bytes: archive_bytes,
            metal,
            metal_llm,
            cpu_threads: cpu_threads.max(1),
            token_embd,
            output_norm,
            output_head,
            layers,
        })
    }

    pub fn backend_label(&self) -> String {
        match (&self.metal, &self.metal_llm) {
            (Some(m), Some(_)) => format!("Apple Metal GPU ({}, 1-CmdBuf Fused)", m.device_name()),
            (Some(m), None) => format!("Apple Metal GPU ({})", m.device_name()),
            (None, _) => format!("CPU ({} threads)", self.cpu_threads),
        }
    }

    pub fn new_state(&self) -> LlmState {
        if let Some(runner) = &self.metal_llm {
            if let Ok(gpu_st) = runner.new_state() {
                return LlmState {
                    pos: 0,
                    gpu: Some(gpu_st),
                    k_caches: Vec::new(),
                    v_caches: Vec::new(),
                    conv_states: Vec::new(),
                    ssm_states: Vec::new(),
                };
            }
        }
        let (k_caches, v_caches, conv_states, ssm_states) = alloc_cpu_state_vecs(&self.cfg);
        LlmState {
            pos: 0,
            gpu: None,
            k_caches,
            v_caches,
            conv_states,
            ssm_states,
        }
    }

    pub fn forward_step(
        &mut self,
        state: &mut LlmState,
        token_id: u32,
        mut logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let dim = self.cfg.n_embd;
        let mut cur = vec![0.0f32; dim];
        self.token_embd.cpu.decode_row(token_id as usize, &mut cur);

        if let (Some(runner), Some(gpu_st)) = (self.metal_llm.as_mut(), state.gpu.as_mut()) {
            runner.step(gpu_st, &cur, state.pos, logits_out)?;
            state.pos += 1;
            return Ok(());
        }

        let eps = self.cfg.rms_norm_eps;
        let threads = self.cpu_threads;

        let mut normed = vec![0.0f32; dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut ffn_gate_buf = vec![0.0f32; self.cfg.n_ff];
        let mut ffn_up_buf = vec![0.0f32; self.cfg.n_ff];
        let mut ffn_out = vec![0.0f32; dim];

        for il in 0..self.cfg.n_layer {
            let layer = &self.layers[il];
            rms_norm(&cur, &layer.attn_norm, eps, &mut normed);

            match &layer.attn {
                LayerAttn::LinearDeltaNet {
                    wqkv,
                    wqkv_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_out,
                    ssm_conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                } => {
                    let d_inner = self.cfg.ssm_inner_size;
                    let n_vh = self.cfg.ssm_dt_rank;
                    let mut qkv_mixed = vec![0.0f32; wqkv.rows()];
                    let mut z = vec![0.0f32; d_inner];
                    let mut alpha_raw = vec![0.0f32; n_vh];
                    let mut beta_raw = vec![0.0f32; n_vh];
                    let mut y_concat = vec![0.0f32; d_inner];

                    forward_multi(
                        &normed,
                        &mut [
                            (wqkv, &mut qkv_mixed),
                            (wqkv_gate, &mut z),
                            (ssm_alpha, &mut alpha_raw),
                            (ssm_beta, &mut beta_raw),
                        ],
                        self.metal.as_mut(),
                        threads,
                    )?;
                    step_deltanet_recurrent(
                        &qkv_mixed,
                        &z,
                        &alpha_raw,
                        &beta_raw,
                        ssm_conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                        &mut state.conv_states[il],
                        &mut state.ssm_states[il],
                        self.cfg.ssm_group_count,
                        n_vh,
                        self.cfg.ssm_state_size,
                        d_inner / n_vh,
                        self.cfg.ssm_conv_kernel,
                        eps,
                        &mut y_concat,
                    );
                    ssm_out.forward(&y_concat, &mut attn_out, self.metal.as_mut(), threads)?;
                }
                LayerAttn::FullGqa {
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    q_bias,
                    k_bias,
                    v_bias,
                    gated_q,
                } => {
                    let (nh, nkv, hd) = (self.cfg.n_head, self.cfg.n_head_kv, self.cfg.head_dim);
                    let mut q_raw = vec![0.0f32; wq.rows()];
                    let mut k_cur = vec![0.0f32; nkv * hd];
                    let mut v_cur = vec![0.0f32; nkv * hd];
                    let mut att_cat = vec![0.0f32; nh * hd];

                    forward_multi(
                        &normed,
                        &mut [(wq, &mut q_raw), (wk, &mut k_cur), (wv, &mut v_cur)],
                        self.metal.as_mut(),
                        threads,
                    )?;
                    for (dst, bias) in [
                        (&mut q_raw[..], q_bias),
                        (&mut k_cur[..], k_bias),
                        (&mut v_cur[..], v_bias),
                    ] {
                        if let Some(b) = bias {
                            for (v, &bv) in dst.iter_mut().zip(b) {
                                *v += bv;
                            }
                        }
                    }
                    step_full_gqa(
                        &q_raw,
                        &mut k_cur,
                        &v_cur,
                        q_norm.as_deref(),
                        k_norm.as_deref(),
                        *gated_q,
                        nh,
                        nkv,
                        hd,
                        state.pos,
                        self.cfg.rope_dim,
                        self.cfg.rope_freq_base,
                        self.cfg.rope_neox,
                        eps,
                        &mut state.k_caches[il],
                        &mut state.v_caches[il],
                        &mut att_cat,
                    );
                    wo.forward(&att_cat, &mut attn_out, self.metal.as_mut(), threads)?;
                }
            }

            for i in 0..dim {
                cur[i] += attn_out[i];
            }
            rms_norm(&cur, &layer.post_attn_norm, eps, &mut normed);

            forward_multi(
                &normed,
                &mut [
                    (&layer.ffn_gate, &mut ffn_gate_buf),
                    (&layer.ffn_up, &mut ffn_up_buf),
                ],
                self.metal.as_mut(),
                threads,
            )?;
            for i in 0..self.cfg.n_ff {
                ffn_gate_buf[i] = silu(ffn_gate_buf[i]) * ffn_up_buf[i];
            }
            layer
                .ffn_down
                .forward(&ffn_gate_buf, &mut ffn_out, self.metal.as_mut(), threads)?;
            for i in 0..dim {
                cur[i] += ffn_out[i];
            }
        }

        state.pos += 1;
        if let Some(logits) = logits_out.as_deref_mut() {
            rms_norm(&cur, &self.output_norm, eps, &mut normed);
            let head = self.output_head.as_ref().unwrap_or(&self.token_embd);
            head.forward(&normed, logits, self.metal.as_mut(), threads)?;
        }
        Ok(())
    }
}
