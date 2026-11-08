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
use crate::metal::{GpuPackedTensor, MetalEngine};
use crate::metal_cb4_fast::{MetalCb4FastRunner, MetalCb4FastState};
use crate::metal_llm::{MetalLlmRunner, MetalLlmState};
use crate::metal_llm_turbo::{MetalTurboRunner, MetalTurboState};
use crate::mmap::TensorBytes;
use crate::tokenizer::BpeTokenizer;
use crate::{Error, Result};

pub(crate) struct MatWeight {
    pub(crate) cpu: PackedTensor,
    pub(crate) gpu: Option<GpuPackedTensor>,
}

impl MatWeight {
    fn cpu_only(cpu: PackedTensor) -> Self {
        Self { cpu, gpu: None }
    }
    pub(crate) fn upload_gpu(&mut self, eng: &MetalEngine) -> Result<()> {
        if self.gpu.is_none() {
            self.gpu = Some(eng.upload_packed(&self.cpu)?);
        }
        Ok(())
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

pub(crate) enum LayerAttn {
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

pub(crate) struct DecoderLayer {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) post_attn_norm: Vec<f32>,
    pub(crate) attn: LayerAttn,
    pub(crate) ffn_gate: MatWeight,
    pub(crate) ffn_up: MatWeight,
    pub(crate) ffn_down: MatWeight,
}

pub struct LlmState {
    pub pos: usize,
    cb4_failed: bool,
    turbo_gpu: Option<MetalTurboState>,
    cb4_fast_gpu: Option<MetalCb4FastState>,
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
    metal_turbo: Option<MetalTurboRunner>,
    metal_cb4_fast: Option<MetalCb4FastRunner>,
    metal_llm: Option<MetalLlmRunner>,
    cpu_threads: usize,
    cb4_greedy_enabled: bool,
    token_embd: MatWeight,
    output_norm: Vec<f32>,
    output_head: Option<MatWeight>,
    layers: Vec<DecoderLayer>,
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
            let dir = std::path::Path::new(path).parent().unwrap_or_else(|| std::path::Path::new("."));
            let config = std::fs::read(dir.join("config.json")).map_err(|e| Error::Io(format!("native config.json: {e}")))?;
            let tokenizer = std::fs::read(dir.join("tokenizer.json")).map_err(|e| Error::Io(format!("native tokenizer.json: {e}")))?;
            let (cfg, tok, map) = crate::native_qwen35::load_source(bytes, &config, &tokenizer)?;
            Self::assemble(cfg, tok, map, bytes.len(), use_gpu, cpu_threads)
        }
    }

    pub fn from_vxt_bytes(bytes: &[u8], use_gpu: bool, cpu_threads: usize) -> Result<Self> {
        let (meta_kv, entries) = read_vxt_model(bytes)?;
        if meta_kv.iter().any(|(k, _)| k == crate::native_qwen35::MANIFEST) {
            let (cfg, tokenizer, map) = crate::native_qwen35::load_archive(&meta_kv, entries)?;
            return Self::assemble(cfg, tokenizer, map, bytes.len(), use_gpu, cpu_threads);
        }
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

    /// Explicit diagnostic: reconstruct every packed tensor before backend selection.
    /// Config/tokenizer and smoothing vectors come from the archive unchanged.
    pub fn load_vxt_decoded_raw_diagnostic(path: &str, use_gpu: bool, cpu_threads: usize) -> Result<(Self, serde_json::Value)> {
        let mapped = TensorBytes::open(path)?;
        let bytes = mapped.as_slice();
        let (meta, entries) = read_vxt_model(bytes)?;
        if meta.iter().any(|(k, _)| k == crate::native_qwen35::MANIFEST) {
            return Err(Error::Unsupported("decoded diagnostic requires standard pack-model archive".into()));
        }
        let cfg = ModelConfig::from_kv_bytes(&meta.iter().find(|(k, _)| k == "model.config")
            .ok_or_else(|| Error::Format("missing model.config".into()))?.1)?;
        let tokenizer = BpeTokenizer::from_vxt_meta(&meta)?;
        let mut map = HashMap::new();
        let mut inventory = Vec::new();
        for e in entries {
            let (raw, audit) = diagnostic_raw_tensor(e.tensor);
            inventory.push(serde_json::json!({"name":e.name,"decoded":audit}));
            map.insert(e.name, raw);
        }
        inventory.sort_by(|a,b| a["name"].as_str().cmp(&b["name"].as_str()));
        let model = Self::assemble(cfg, tokenizer, map, bytes.len(), use_gpu, cpu_threads)?;
        Ok((model, serde_json::json!(inventory)))
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
                // Keep the reference independent of the packer: otherwise a shared
                // conversion bug can make a lossy archive appear exact in eval.
                {
                    let data = gg.extract_f32(m)?;
                    let (rows, cols) = m.matrix_dims().unwrap_or((1, data.len().max(1)));
                    PackedTensor::RawF32 { rows, cols, data }
                },
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
        let take_mat = |map: &mut HashMap<String, PackedTensor>, name: &str| -> Result<MatWeight> {
            let t = map
                .remove(name)
                .ok_or_else(|| Error::Format(format!("missing model tensor '{name}'")))?;
            Ok(MatWeight::cpu_only(t))
        };
        let take_vec = |map: &mut HashMap<String, PackedTensor>, name: &str| -> Result<Vec<f32>> {
            let t = map
                .remove(name)
                .ok_or_else(|| Error::Format(format!("missing model vector '{name}'")))?;
            Ok(t.decode_f32())
        };

        let mut token_embd = take_mat(&mut map, "token_embd.weight")?;
        let output_norm = take_vec(&mut map, "output_norm.weight")?;
        let mut output_head = map.remove("output.weight").map(MatWeight::cpu_only);

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

        let mut metal_turbo = None;
        let mut metal_cb4_fast = None;
        let mut metal_llm = None;
        if let Some(eng) = metal.as_ref() {
            metal_turbo = MetalTurboRunner::try_from_layers(
                eng,
                &cfg,
                &token_embd.cpu,
                &output_norm,
                output_head.as_ref().map(|h| &h.cpu),
                &layers,
            )?;
            if metal_turbo.is_none() {
                metal_cb4_fast = MetalCb4FastRunner::try_from_layers(
                    eng,
                    &cfg,
                    &token_embd.cpu,
                    &output_norm,
                    output_head.as_ref().map(|h| &h.cpu),
                    &layers,
                )?;
            }
            if metal_turbo.is_none() && metal_cb4_fast.is_none() {
                metal_llm = MetalLlmRunner::upload_and_build(
                    eng,
                    &cfg,
                    &mut token_embd,
                    &mut output_head,
                    &mut layers,
                    &output_norm,
                )?;
            }
        }

        Ok(Self {
            cfg,
            tokenizer,
            total_compressed_bytes: archive_bytes,
            metal,
            metal_turbo,
            metal_cb4_fast,
            metal_llm,
            cpu_threads: cpu_threads.max(1),
            cb4_greedy_enabled: false,
            token_embd,
            output_norm,
            output_head,
            layers,
        })
    }

    pub fn backend_label(&self) -> String {
        match (
            &self.metal,
            &self.metal_turbo,
            &self.metal_cb4_fast,
            &self.metal_llm,
        ) {
            (Some(m), Some(_), _, _) => {
                format!("Apple Metal GPU ({}, VXT-Turbo Pipelined)", m.device_name())
            }
            (Some(m), None, Some(_), _) => {
                format!(
                    "Apple Metal GPU ({}, 1-CmdBuf Fused, CB4-Fast Pipelined)",
                    m.device_name()
                )
            }
            (Some(m), None, None, Some(_)) => {
                format!("Apple Metal GPU ({}, 1-CmdBuf Fused)", m.device_name())
            }
            (Some(m), None, None, None) => format!("Apple Metal GPU ({})", m.device_name()),
            (None, _, _, _) => format!("CPU ({} threads)", self.cpu_threads),
        }
    }

    pub fn has_turbo_fast_decode(&self) -> bool {
        self.metal_turbo.is_some()
    }

    pub fn has_cb4_fast_decode(&self) -> bool {
        self.metal_cb4_fast.is_some()
    }

    pub fn new_state(&self) -> LlmState {
        if let Some(tr) = &self.metal_turbo {
            if let Ok(tst) = tr.new_state() {
                return LlmState {
                    pos: 0,
                    cb4_failed: false,
                    turbo_gpu: Some(tst),
                    cb4_fast_gpu: None,
                    gpu: None,
                    k_caches: Vec::new(),
                    v_caches: Vec::new(),
                    conv_states: Vec::new(),
                    ssm_states: Vec::new(),
                };
            }
        }
        if let Some(cf) = &self.metal_cb4_fast {
            if let Ok(cst) = cf.new_state() {
                return LlmState {
                    pos: 0,
                    cb4_failed: false,
                    turbo_gpu: None,
                    cb4_fast_gpu: Some(cst),
                    gpu: None,
                    k_caches: Vec::new(),
                    v_caches: Vec::new(),
                    conv_states: Vec::new(),
                    ssm_states: Vec::new(),
                };
            }
        }
        if let Some(runner) = &self.metal_llm {
            if let Ok(gpu_st) = runner.new_state() {
                return LlmState {
                    pos: 0,
                    cb4_failed: false,
                    turbo_gpu: None,
                    cb4_fast_gpu: None,
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
            cb4_failed: false,
            turbo_gpu: None,
            cb4_fast_gpu: None,
            gpu: None,
            k_caches,
            v_caches,
            conv_states,
            ssm_states,
        }
    }

    pub fn prefill(
        &mut self,
        state: &mut LlmState,
        prompt_ids: &[u32],
        logits_out: &mut [f32],
    ) -> Result<()> {
        if state.cb4_failed { return Err(Error::Unsupported("CB4 device failure invalidated this state; create a new state".into())); }
        if let (Some(tr), Some(tst)) = (self.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) {
            return tr.prefill_tokens(tst, &mut state.pos, prompt_ids, logits_out);
        }
        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            return cf.prefill_tokens(cst, &mut state.pos, prompt_ids, logits_out);
        }
        for (idx, &tid) in prompt_ids.iter().enumerate() {
            let last = idx + 1 == prompt_ids.len();
            self.forward_step(state, tid, last.then_some(&mut *logits_out))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_turbo<F: FnMut(u32) -> bool>(
        &self,
        state: &mut LlmState,
        prompt_ids: &[u32],
        first_tok: u32,
        max_new_tokens: usize,
        use_sketch: bool,
        presence_pen: f32,
        stop_ids: &[u32],
        on_token: F,
    ) -> Result<Vec<u32>> {
        if state.cb4_failed { return Err(Error::Unsupported("CB4 device failure invalidated this state; create a new state".into())); }
        let (Some(tr), Some(tst)) = (self.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) else {
            return Err(Error::Unsupported("Turbo fast decode unavailable".into()));
        };
        tr.generate_greedy_fast(
            tst,
            &mut state.pos,
            prompt_ids,
            first_tok,
            max_new_tokens,
            use_sketch,
            presence_pen,
            stop_ids,
            on_token,
        )
    }

    /// Opt in to exact CB4 embedding decode and full-head GPU greedy selection.
    pub fn enable_cb4_greedy(&mut self) -> Result<()> {
        let PackedTensor::Codebook(embedding) = &self.token_embd.cpu else {
            return Err(Error::Unsupported("CB4 greedy requires a four-bit Codebook embedding".into()));
        };
        if embedding.bits != 4 {
            return Err(Error::Unsupported("CB4 greedy requires a four-bit Codebook embedding".into()));
        }
        if self.metal_cb4_fast.is_some() {
            self.cb4_greedy_enabled = true;
            return Ok(());
        }
        let gpu = self.token_embd.gpu.as_ref().ok_or_else(|| Error::Unsupported("CB4 greedy requires an uploaded Metal embedding".into()))?;
        let runner = self.metal_llm.as_mut().ok_or_else(|| Error::Unsupported("CB4 greedy requires the generic Metal runner".into()))?;
        runner.enable_cb4_greedy(gpu, &embedding.centroids)?;
        self.cb4_greedy_enabled = true;
        Ok(())
    }

    /// `first_tok` is predicted but unconsumed. On cap/cancellation the last
    /// emitted token remains pending; stop tokens are never emitted or consumed.
    /// A device-step error invalidates the state: discard it and create a new one.
    pub fn generate_greedy_cb4<F: FnMut(u32, &BpeTokenizer) -> bool>(
        &mut self, state: &mut LlmState, first_tok: u32,
        max_new_tokens: usize, stop_ids: &[u32], mut on_token: F,
    ) -> Result<Vec<u32>> {
        if state.cb4_failed { return Err(Error::Unsupported("CB4 device failure invalidated this state; create a new state".into())); }
        if !self.cb4_greedy_enabled {
            return Err(Error::Unsupported("CB4 greedy must be explicitly enabled".into()));
        }
        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            return cf.generate_greedy_fast(
                cst,
                &mut state.pos,
                &mut state.cb4_failed,
                &self.tokenizer,
                first_tok,
                max_new_tokens,
                stop_ids,
                on_token,
            );
        }
        let (Some(runner), Some(gpu)) = (self.metal_llm.as_mut(), state.gpu.as_mut()) else {
            return Err(Error::Unsupported("CB4 greedy requires Metal model and state".into()));
        };
        let mut emitted = Vec::with_capacity(max_new_tokens);
        let mut next = first_tok;
        while emitted.len() < max_new_tokens && !stop_ids.contains(&next) {
            if next as usize >= self.cfg.n_vocab {
                return Err(Error::Format("CB4 greedy token outside vocabulary".into()));
            }
            emitted.push(next);
            if !on_token(next, &self.tokenizer) || emitted.len() == max_new_tokens { break; }
            let predicted = match runner.greedy_step(gpu, next, state.pos) {
                Ok(token) => token,
                Err(error) => { state.cb4_failed = true; return Err(error); }
            };
            state.pos += 1;
            next = predicted;
        }
        Ok(emitted)
    }

    pub fn forward_step(
        &mut self,
        state: &mut LlmState,
        token_id: u32,
        mut logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        if state.cb4_failed { return Err(Error::Unsupported("CB4 device failure invalidated this state; create a new state".into())); }
        if let (Some(tr), Some(tst)) = (self.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) {
            tr.step_single(tst, state.pos as u32, token_id, logits_out)?;
            state.pos += 1;
            return Ok(());
        }
        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            cf.step_single(cst, state.pos as u32, token_id, logits_out)?;
            state.pos += 1;
            return Ok(());
        }
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

// Fingerprint is FNV-1a over exact reconstructed LE f32 bits, not a quality metric.
fn diagnostic_raw_tensor(t: PackedTensor) -> (PackedTensor, serde_json::Value) {
    let (rows, cols) = (t.rows(), t.cols());
    let kind = format!("{:?}", std::mem::discriminant(&t));
    let data = t.decode_f32();
    let mut hash = 0xcbf29ce484222325u64;
    for x in &data { for b in x.to_le_bytes() { hash = (hash ^ b as u64).wrapping_mul(0x100000001b3); } }
    let audit = serde_json::json!({"rows":rows,"cols":cols,"elements":data.len(),"source_kind":kind,"decoded_f32_fnv1a":format!("{hash:016x}"),"finite":data.iter().all(|x|x.is_finite())});
    (PackedTensor::RawF32 {rows, cols, data}, audit)
}
#[cfg(test)]
mod decoded_diagnostic_tests {
    use super::*;
    #[test]
    fn turbo_materialization_preserves_exact_decode_bits_and_raw_route() {
        let w:Vec<f32>=(0..256).map(|i| ((i as f32)*0.31).sin()).collect();
        let t=PackedTensor::TurboWht(crate::wht_turbo::TurboWhtMatrix::encode(4,4,64,&w).unwrap());
        let expected:Vec<_>=t.decode_f32().iter().map(|x|x.to_bits()).collect();
        let (raw,audit)=diagnostic_raw_tensor(t);
        assert!(matches!(raw,PackedTensor::RawF32{rows:4,cols:64,..}));
        assert_eq!(expected,raw.decode_f32().iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        assert_eq!(audit["finite"],true);
    }
    #[test]
    fn raw_materialization_preserves_exact_decode_bits() {
        let input = PackedTensor::RawF32 {rows:1,cols:4,data:vec![-0.0,1.25,-2.5,f32::from_bits(1)]};
        let expected:Vec<_> = input.decode_f32().iter().map(|x|x.to_bits()).collect();
        let (raw,audit)=diagnostic_raw_tensor(input);
        assert!(matches!(raw,PackedTensor::RawF32{..}));
        assert_eq!(expected,raw.decode_f32().iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        assert_eq!(audit["elements"],4);
    }
}
