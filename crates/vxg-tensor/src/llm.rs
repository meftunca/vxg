//! Compressed-domain autoregressive LLM + Multimodal VLM inference engine (`qwen35` hybrid
//! Gated-DeltaNet + Gated-GQA + SwiGLU + `Qwen3_5VisionModel` 3D `M-RoPE`, plus standard
//! `qwen3` / `qwen2` / `llama` GQA Transformers) executing directly on `.vxt` compressed weights.

use std::collections::HashMap;

use crate::format::{read_vxt_model, PackedTensor};
use crate::gguf::GgufView;
use crate::gguf_meta::{parse_gguf_kv, ModelConfig};
use crate::llm_cpu::{
    build_multimodal_embeds, diagnostic_raw_tensor, extract_decoder_layers, forward_embed_step,
};
pub(crate) use crate::llm_cpu::{DecoderLayer, LayerAttn, MatWeight};
use crate::llm_ops::alloc_cpu_state_vecs;
pub use crate::llm_ops::{compare_logits, sample_logits, LogitEvalMetrics};
use crate::llm_prefill::{cpu_kv_cache_bytes, prefill_cpu_tiled};
pub use crate::llm_prefill::{KvCacheMode, KvQ8Block};
use crate::metal::MetalEngine;
use crate::metal_cb4_fast::{MetalCb4FastRunner, MetalCb4FastState};
use crate::metal_llm::{MetalLlmRunner, MetalLlmState};
use crate::metal_llm_turbo::{MetalTurboRunner, MetalTurboState};
use crate::mmap::TensorBytes;
use crate::tokenizer::BpeTokenizer;
use crate::vision::{compute_mrope_positions, EncodedVisionImage, VisionEncoder};
use crate::vision_img::VisionConfig;
use crate::{Error, Result};

pub struct LlmState {
    pub pos: usize,
    pub rope_delta: isize,
    pub kv_mode: KvCacheMode,
    cb4_failed: bool,
    turbo_gpu: Option<MetalTurboState>,
    cb4_fast_gpu: Option<MetalCb4FastState>,
    gpu: Option<MetalLlmState>,
    pub(crate) k_caches: Vec<Vec<f32>>,
    pub(crate) v_caches: Vec<Vec<f32>>,
    pub(crate) k_q8: Vec<Vec<KvQ8Block>>,
    pub(crate) v_q8: Vec<Vec<KvQ8Block>>,
    pub(crate) conv_states: Vec<Vec<f32>>,
    pub(crate) ssm_states: Vec<Vec<f32>>,
}

pub struct LlmModel {
    pub cfg: ModelConfig,
    pub tokenizer: BpeTokenizer,
    pub total_compressed_bytes: usize,
    pub(crate) metal: Option<MetalEngine>,
    metal_turbo: Option<MetalTurboRunner>,
    metal_cb4_fast: Option<MetalCb4FastRunner>,
    metal_llm: Option<MetalLlmRunner>,
    pub(crate) cpu_threads: usize,
    cb4_greedy_enabled: bool,
    pub(crate) token_embd: MatWeight,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) output_head: Option<MatWeight>,
    pub(crate) layers: Vec<DecoderLayer>,
    vision: Option<VisionEncoder>,
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
            let dir = std::path::Path::new(path)
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            let config = std::fs::read(dir.join("config.json"))
                .map_err(|e| Error::Io(format!("native config.json: {e}")))?;
            let tokenizer = std::fs::read(dir.join("tokenizer.json"))
                .map_err(|e| Error::Io(format!("native tokenizer.json: {e}")))?;
            let (cfg, tok, map) = crate::native_qwen35::load_source(bytes, &config, &tokenizer)?;
            Self::assemble(cfg, tok, None, map, bytes.len(), use_gpu, cpu_threads)
        }
    }

    pub fn from_vxt_bytes(bytes: &[u8], use_gpu: bool, cpu_threads: usize) -> Result<Self> {
        let (meta_kv, entries) = read_vxt_model(bytes)?;
        if meta_kv.iter().any(|(k, _)| k == crate::native_qwen35::MANIFEST) {
            let (cfg, tokenizer, map) = crate::native_qwen35::load_archive(&meta_kv, entries)?;
            return Self::assemble(cfg, tokenizer, None, map, bytes.len(), use_gpu, cpu_threads);
        }
        let cfg_bytes = meta_kv
            .iter()
            .find(|(k, _)| k == "model.config")
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| Error::Format("missing model.config in .vxt (use pack-model)".into()))?;
        let cfg = ModelConfig::from_kv_bytes(cfg_bytes)?;
        let vision_cfg = meta_kv
            .iter()
            .find(|(k, _)| k == "vision.config")
            .map(|(_, v)| VisionConfig::from_json_bytes(v))
            .transpose()?;
        let tokenizer = BpeTokenizer::from_vxt_meta(&meta_kv)?;
        let map: HashMap<String, PackedTensor> =
            entries.into_iter().map(|e| (e.name, e.tensor)).collect();
        Self::assemble(cfg, tokenizer, vision_cfg, map, bytes.len(), use_gpu, cpu_threads)
    }

    /// Explicit diagnostic: reconstruct every packed tensor before backend selection.
    pub fn load_vxt_decoded_raw_diagnostic(
        path: &str,
        use_gpu: bool,
        cpu_threads: usize,
    ) -> Result<(Self, serde_json::Value)> {
        let mapped = TensorBytes::open(path)?;
        let bytes = mapped.as_slice();
        let (meta, entries) = read_vxt_model(bytes)?;
        if meta.iter().any(|(k, _)| k == crate::native_qwen35::MANIFEST) {
            return Err(Error::Unsupported(
                "decoded diagnostic requires standard pack-model archive".into(),
            ));
        }
        let cfg = ModelConfig::from_kv_bytes(
            &meta
                .iter()
                .find(|(k, _)| k == "model.config")
                .ok_or_else(|| Error::Format("missing model.config".into()))?
                .1,
        )?;
        let vision_cfg = meta
            .iter()
            .find(|(k, _)| k == "vision.config")
            .map(|(_, v)| VisionConfig::from_json_bytes(v))
            .transpose()?;
        let tokenizer = BpeTokenizer::from_vxt_meta(&meta)?;
        let mut map = HashMap::new();
        let mut inventory = Vec::new();
        for e in entries {
            let (raw, audit) = diagnostic_raw_tensor(e.tensor);
            inventory.push(serde_json::json!({"name": e.name, "decoded": audit}));
            map.insert(e.name, raw);
        }
        inventory.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        let model = Self::assemble(cfg, tokenizer, vision_cfg, map, bytes.len(), use_gpu, cpu_threads)?;
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
            let data = gg.extract_f32(m)?;
            let (rows, cols) = m.matrix_dims().unwrap_or((1, data.len().max(1)));
            map.insert(m.name.clone(), PackedTensor::RawF32 { rows, cols, data });
        }
        Self::assemble(cfg, tokenizer, None, map, bytes.len(), use_gpu, cpu_threads)
    }

    fn assemble(
        cfg: ModelConfig,
        tokenizer: BpeTokenizer,
        vision_cfg: Option<VisionConfig>,
        mut map: HashMap<String, PackedTensor>,
        archive_bytes: usize,
        use_gpu: bool,
        cpu_threads: usize,
    ) -> Result<Self> {
        let mut vision = match vision_cfg {
            Some(vc) if map.contains_key("vis.patch_embed.proj.weight") => {
                Some(VisionEncoder::from_tensors(vc, &mut map)?)
            }
            _ => None,
        };
        let metal = if use_gpu { MetalEngine::new().ok() } else { None };
        let threads = cpu_threads.max(1);
        if let (Some(eng), Some(vis)) = (metal.as_ref(), vision.as_mut()) {
            let _ = vis.upload_gpu(eng, threads);
        }
        let mut token_embd = MatWeight::cpu_only(
            map.remove("token_embd.weight")
                .ok_or_else(|| Error::Format("missing model tensor 'token_embd.weight'".into()))?,
        );
        let output_norm = map
            .remove("output_norm.weight")
            .ok_or_else(|| Error::Format("missing model vector 'output_norm.weight'".into()))?
            .decode_f32();
        let mut output_head = map.remove("output.weight").map(MatWeight::cpu_only);
        let mut layers = extract_decoder_layers(&cfg, &mut map)?;

        let (mut metal_turbo, mut metal_cb4_fast, mut metal_llm) = (None, None, None);
        if let Some(eng) = metal.as_ref() {
            let head_cpu = output_head.as_ref().map(|h| &h.cpu);
            metal_turbo = MetalTurboRunner::try_from_layers(
                eng, &cfg, &token_embd.cpu, &output_norm, head_cpu, &layers,
            )?;
            if metal_turbo.is_none() {
                metal_cb4_fast = MetalCb4FastRunner::try_from_layers(
                    eng, &cfg, &token_embd.cpu, &output_norm, head_cpu, &layers,
                )?;
            }
            if metal_turbo.is_none() && metal_cb4_fast.is_none() {
                metal_llm = MetalLlmRunner::upload_and_build(
                    eng, &cfg, &mut token_embd, &mut output_head, &mut layers, &output_norm,
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
            cpu_threads: threads,
            cb4_greedy_enabled: false,
            token_embd,
            output_norm,
            output_head,
            layers,
            vision,
        })
    }

    pub fn backend_label(&self) -> String {
        let base = match (&self.metal, &self.metal_turbo, &self.metal_cb4_fast, &self.metal_llm) {
            (Some(m), Some(_), _, _) => {
                format!("Apple Metal GPU ({}, VXT-Turbo Pipelined)", m.device_name())
            }
            (Some(m), None, Some(_), _) => {
                format!("Apple Metal GPU ({}, 1-CmdBuf Fused, CB4-Fast Pipelined)", m.device_name())
            }
            (Some(m), None, None, Some(_)) => {
                format!("Apple Metal GPU ({}, 1-CmdBuf Fused)", m.device_name())
            }
            (Some(m), None, None, None) => format!("Apple Metal GPU ({})", m.device_name()),
            (None, _, _, _) => format!("CPU ({} threads)", self.cpu_threads),
        };
        if let Some(vis) = self.vision.as_ref() {
            if vis.has_gpu() {
                format!("{base} + Vision (Metal GPU)")
            } else {
                format!("{base} + Vision")
            }
        } else {
            base
        }
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn vision_config(&self) -> Option<&VisionConfig> {
        self.vision.as_ref().map(|v| &v.cfg)
    }

    pub fn encode_image_source(&self, src: &str) -> Result<EncodedVisionImage> {
        let v = self.vision.as_ref().ok_or_else(|| {
            Error::Unsupported("loaded VXT model does not contain a vision encoder".into())
        })?;
        v.encode_source(src, self.cpu_threads)
    }

    pub fn has_turbo_fast_decode(&self) -> bool {
        self.metal_turbo.is_some()
    }

    pub fn has_cb4_fast_decode(&self) -> bool {
        self.metal_cb4_fast.is_some()
    }

    pub fn new_state(&self) -> LlmState {
        self.new_state_with_kv(KvCacheMode::Fp32)
    }

    pub fn new_state_with_kv(&self, kv_mode: KvCacheMode) -> LlmState {
        let want_q8 = kv_mode == KvCacheMode::Q8_0 && self.cfg.head_dim % 32 == 0;
        let turbo_gpu = self.metal_turbo.as_ref().and_then(|tr| tr.new_state().ok());
        let cb4_fast_gpu = if turbo_gpu.is_none() {
            self.metal_cb4_fast.as_ref().and_then(|cf| cf.new_state(want_q8).ok())
        } else {
            None
        };
        let gpu = if turbo_gpu.is_none() && cb4_fast_gpu.is_none() && !want_q8 {
            self.metal_llm.as_ref().and_then(|r| r.new_state().ok())
        } else {
            None
        };
        let is_cpu = turbo_gpu.is_none() && cb4_fast_gpu.is_none() && gpu.is_none();
        let (mut k_caches, mut v_caches, conv_states, ssm_states) = if is_cpu {
            alloc_cpu_state_vecs(&self.cfg)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        let (mut k_q8, mut v_q8) = (Vec::new(), Vec::new());
        if is_cpu && want_q8 {
            let nl = self.cfg.n_layer;
            let blks = self.cfg.n_head_kv * (self.cfg.head_dim / 32) * 256;
            k_q8.reserve_exact(nl);
            v_q8.reserve_exact(nl);
            for il in 0..nl {
                let cap = if self.cfg.is_recurrent_layer(il) { 0 } else { blks };
                k_q8.push(Vec::with_capacity(cap));
                v_q8.push(Vec::with_capacity(cap));
                k_caches[il] = Vec::new();
                v_caches[il] = Vec::new();
            }
        }
        LlmState {
            pos: 0,
            rope_delta: 0,
            kv_mode: if want_q8 { KvCacheMode::Q8_0 } else { KvCacheMode::Fp32 },
            cb4_failed: false,
            turbo_gpu,
            cb4_fast_gpu,
            gpu,
            k_caches,
            v_caches,
            k_q8,
            v_q8,
            conv_states,
            ssm_states,
        }
    }

    pub fn kv_cache_bytes(&self, state: &LlmState) -> usize {
        if let Some(cst) = state.cb4_fast_gpu.as_ref() {
            cst.kv_bytes()
        } else {
            cpu_kv_cache_bytes(state)
        }
    }

    pub fn prefill(
        &mut self,
        state: &mut LlmState,
        prompt_ids: &[u32],
        logits_out: &mut [f32],
    ) -> Result<()> {
        if state.cb4_failed {
            return Err(Error::Unsupported(
                "CB4 device failure invalidated this state; create a new state".into(),
            ));
        }
        if state.rope_delta == 0 {
            if let (Some(tr), Some(tst)) = (self.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) {
                return tr.prefill_tokens(tst, &mut state.pos, prompt_ids, logits_out);
            }
        }
        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            return cf.prefill_tokens(cst, &mut state.pos, state.rope_delta, prompt_ids, logits_out);
        }
        prefill_cpu_tiled(self, state, prompt_ids, logits_out)
    }

    pub fn prefill_multimodal(
        &mut self,
        state: &mut LlmState,
        tok_ids: &[u32],
        images: &[&EncodedVisionImage],
        logits_out: &mut [f32],
    ) -> Result<()> {
        if images.is_empty() {
            return self.prefill(state, tok_ids, logits_out);
        }
        if state.cb4_failed {
            return Err(Error::Unsupported(
                "CB4 device failure invalidated this state; create a new state".into(),
            ));
        }
        let vcfg = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::Unsupported("model has no vision encoder".into()))?
            .cfg
            .clone();
        let start_rope = ((state.pos as isize) + state.rope_delta).max(0) as usize;
        let (mrope_pos, new_delta) =
            compute_mrope_positions(tok_ids, images, &vcfg, state.pos, start_rope)?;
        let dim = self.cfg.n_embd;
        let embeds = build_multimodal_embeds(
            &self.token_embd.cpu, self.cfg.n_vocab, dim, self.cfg.embed_scale,
            tok_ids, images, vcfg.image_token_id,
        )?;

        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            cf.prefill_embeds_mrope(
                cst, &mut state.pos, &embeds, dim, &mrope_pos, vcfg.mrope_section, logits_out,
            )?;
            state.rope_delta = new_delta;
            return Ok(());
        }

        if state.gpu.is_some() || state.turbo_gpu.is_some() {
            if state.pos == 0 {
                state.gpu = None;
                state.turbo_gpu = None;
                let (k, v, c, s) = alloc_cpu_state_vecs(&self.cfg);
                state.k_caches = k;
                state.v_caches = v;
                state.conv_states = c;
                state.ssm_states = s;
            } else {
                return Err(Error::Unsupported(
                    "multimodal mid-turn transition requires CB4-Fast or CPU state".into(),
                ));
            }
        }
        for (idx, p) in mrope_pos.iter().enumerate() {
            let last = idx + 1 == mrope_pos.len();
            forward_embed_step(
                self,
                state,
                &embeds[idx * dim..(idx + 1) * dim],
                [p[0], p[1], p[2]],
                Some(vcfg.mrope_section),
                last.then_some(&mut *logits_out),
            )?;
        }
        state.rope_delta = new_delta;
        Ok(())
    }

    pub fn set_turbo_ngram(&self, enable: bool) {
        if let Some(tr) = self.metal_turbo.as_ref() {
            tr.set_ngram_spec(enable);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_turbo<F: FnMut(u32) -> bool>(
        &self, state: &mut LlmState, prompt_ids: &[u32], first_tok: u32,
        max_new_tokens: usize, use_sketch: bool, presence_pen: f32, stop_ids: &[u32], on_token: F,
    ) -> Result<Vec<u32>> {
        if state.cb4_failed {
            return Err(Error::Unsupported("CB4 device failure invalidated this state; create a new state".into()));
        }
        let (Some(tr), Some(tst)) = (self.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) else {
            return Err(Error::Unsupported("Turbo fast decode unavailable".into()));
        };
        tr.generate_greedy_fast(tst, &mut state.pos, prompt_ids, first_tok, max_new_tokens, use_sketch, presence_pen, stop_ids, on_token)
    }

    /// Opt in to exact CB4 embedding decode and full-head GPU greedy selection.
    pub fn enable_cb4_greedy(&mut self) -> Result<()> {
        let PackedTensor::Codebook(embedding) = &self.token_embd.cpu else {
            return Err(Error::Unsupported(
                "CB4 greedy requires a four-bit Codebook embedding".into(),
            ));
        };
        if embedding.bits != 4 {
            return Err(Error::Unsupported(
                "CB4 greedy requires a four-bit Codebook embedding".into(),
            ));
        }
        if self.metal_cb4_fast.is_some() {
            self.cb4_greedy_enabled = true;
            return Ok(());
        }
        let gpu = self.token_embd.gpu.as_ref().ok_or_else(|| {
            Error::Unsupported("CB4 greedy requires an uploaded Metal embedding".into())
        })?;
        let runner = self.metal_llm.as_mut().ok_or_else(|| {
            Error::Unsupported("CB4 greedy requires the generic Metal runner".into())
        })?;
        runner.enable_cb4_greedy(gpu, &embedding.centroids)?;
        self.cb4_greedy_enabled = true;
        Ok(())
    }

    pub fn generate_greedy_cb4<F: FnMut(u32, &BpeTokenizer) -> bool>(
        &mut self,
        state: &mut LlmState,
        first_tok: u32,
        max_new_tokens: usize,
        stop_ids: &[u32],
        mut on_token: F,
    ) -> Result<Vec<u32>> {
        if state.cb4_failed {
            return Err(Error::Unsupported(
                "CB4 device failure invalidated this state; create a new state".into(),
            ));
        }
        if !self.cb4_greedy_enabled {
            return Err(Error::Unsupported("CB4 greedy must be explicitly enabled".into()));
        }
        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            return cf.generate_greedy_fast(
                cst,
                &mut state.pos,
                state.rope_delta,
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
            if !on_token(next, &self.tokenizer) || emitted.len() == max_new_tokens {
                break;
            }
            let predicted = match runner.greedy_step(gpu, next, state.pos) {
                Ok(token) => token,
                Err(error) => {
                    state.cb4_failed = true;
                    return Err(error);
                }
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
        logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        if state.cb4_failed {
            return Err(Error::Unsupported(
                "CB4 device failure invalidated this state; create a new state".into(),
            ));
        }
        if state.rope_delta == 0 {
            if let (Some(tr), Some(tst)) = (self.metal_turbo.as_ref(), state.turbo_gpu.as_mut()) {
                tr.step_single(tst, state.pos as u32, token_id, logits_out)?;
                state.pos += 1;
                return Ok(());
            }
        }
        if let (Some(cf), Some(cst)) = (self.metal_cb4_fast.as_ref(), state.cb4_fast_gpu.as_mut()) {
            cf.step_single(cst, state.pos as u32, state.rope_delta, token_id, logits_out)?;
            state.pos += 1;
            return Ok(());
        }
        let dim = self.cfg.n_embd;
        let mut cur = vec![0.0f32; dim];
        self.token_embd.cpu.decode_row(token_id as usize, &mut cur);
        if (self.cfg.embed_scale - 1.0).abs() > 1e-7 {
            for v in cur.iter_mut() {
                *v *= self.cfg.embed_scale;
            }
        }

        if state.rope_delta == 0 {
            if let (Some(runner), Some(gpu_st)) = (self.metal_llm.as_mut(), state.gpu.as_mut()) {
                runner.step(gpu_st, &cur, state.pos, logits_out)?;
                state.pos += 1;
                return Ok(());
            }
        }
        let rpos = ((state.pos as isize) + state.rope_delta).max(0) as u32;
        forward_embed_step(self, state, &cur, [rpos, rpos, rpos], None, logits_out)
    }
}
