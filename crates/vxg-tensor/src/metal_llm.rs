//! RAII 1-Command-Buffer All-GPU LLM forward-pass runner (`MetalLlmRunner` & `MetalLlmState`).

use crate::gguf_meta::ModelConfig;
use crate::llm::{DecoderLayer, LayerAttn, MatWeight};
use crate::metal::{GpuPackedTensor, MetalEngine};
#[cfg(not(vxt_has_metal))]
use crate::Error;
use crate::Result;

#[cfg_attr(not(vxt_has_metal), allow(dead_code))]
pub(crate) enum MetalLlmAttnSpec<'a> {
    LinearDeltaNet {
        wqkv: &'a GpuPackedTensor,
        wqkv_gate: &'a GpuPackedTensor,
        ssm_alpha: &'a GpuPackedTensor,
        ssm_beta: &'a GpuPackedTensor,
        ssm_out: &'a GpuPackedTensor,
        ssm_conv1d: &'a [f32],
        ssm_a: &'a [f32],
        ssm_dt_bias: &'a [f32],
        ssm_norm: &'a [f32],
    },
    FullGqa {
        wq: &'a GpuPackedTensor,
        wk: &'a GpuPackedTensor,
        wv: &'a GpuPackedTensor,
        wo: &'a GpuPackedTensor,
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        q_bias: Option<&'a [f32]>,
        k_bias: Option<&'a [f32]>,
        v_bias: Option<&'a [f32]>,
        gated_q: bool,
    },
}

#[cfg_attr(not(vxt_has_metal), allow(dead_code))]
pub(crate) struct MetalLlmLayerSpec<'a> {
    pub attn_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub attn: MetalLlmAttnSpec<'a>,
    pub ffn_gate: &'a GpuPackedTensor,
    pub ffn_up: &'a GpuPackedTensor,
    pub ffn_down: &'a GpuPackedTensor,
}

#[cfg(vxt_has_metal)]
mod imp {
    use core::ffi::{c_char, c_void};
    use core::ptr::{null_mut, NonNull};

    use super::{MetalLlmAttnSpec, MetalLlmLayerSpec};
    use crate::gguf_meta::ModelConfig;
    use crate::metal::{GpuPackedTensor, MetalEngine};
    use crate::metal_ffi::{cbuf_to_str, GpuBuf, VxtMetalOp};
    use crate::{Error, Result};

    #[repr(C)]
    struct VxtLlmConfig {
        n_layer: u32,
        n_embd: u32,
        n_ff: u32,
        n_vocab: u32,
        n_head: u32,
        n_head_kv: u32,
        head_dim: u32,
        rope_dim: u32,
        rope_neox: u32,
        rope_freq_base: f32,
        rms_norm_eps: f32,
        ssm_conv_kernel: u32,
        ssm_state_size: u32,
        ssm_group_count: u32,
        ssm_inner_size: u32,
        ssm_dt_rank: u32,
    }

    #[repr(C)]
    struct VxtLlmLayerDesc {
        is_recurrent: u32,
        gated_q: u32,
        attn_norm_buf: *mut c_void,
        post_attn_norm_buf: *mut c_void,
        op_a0: VxtMetalOp,
        op_a1: VxtMetalOp,
        op_a2: VxtMetalOp,
        op_a3: VxtMetalOp,
        op_out: VxtMetalOp,
        vec0_buf: *mut c_void,
        vec1_buf: *mut c_void,
        vec2_buf: *mut c_void,
        vec3_buf: *mut c_void,
        vec4_buf: *mut c_void,
        op_ffn_gate: VxtMetalOp,
        op_ffn_up: VxtMetalOp,
        op_ffn_down: VxtMetalOp,
    }

    extern "C" {
        fn vxt_metal_llm_create(
            gemv_ctx: *mut c_void,
            cfg: *const VxtLlmConfig,
            layers: *const VxtLlmLayerDesc,
            output_norm_buf: *mut c_void,
            output_head_op: *const VxtMetalOp,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> *mut c_void;
        fn vxt_metal_llm_destroy(runner: *mut c_void);
        fn vxt_metal_llm_state_create(
            runner: *mut c_void,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> *mut c_void;
        fn vxt_metal_llm_state_destroy(state: *mut c_void);
        fn vxt_metal_llm_enable_cb4_greedy(runner: *mut c_void, embedding: *const VxtMetalOp,
            centroids: *const f32, err: *mut c_char, cap: usize) -> i32;
        fn vxt_metal_llm_greedy_step(runner: *mut c_void, state: *mut c_void,
            token: u32, pos: u32, next: *mut u32, err: *mut c_char, cap: usize) -> i32;
        fn vxt_metal_llm_step(
            runner: *mut c_void,
            state: *mut c_void,
            cur_init: *const f32,
            pos: u32,
            compute_logits: u32,
            logits_out: *mut f32,
            err_out: *mut c_char,
            err_cap: usize,
        ) -> i32;
    }

    fn zero_op() -> VxtMetalOp {
        VxtMetalOp {
            kind: 0,
            sub: 0,
            rows: 0,
            cols: 0,
            lut8: [0; 8],
            poly16: [0.0; 16],
            b0: null_mut(),
            b1: null_mut(),
            b2: null_mut(),
            b3: null_mut(),
            y_dst: null_mut(),
        }
    }

    pub struct RunnerInner {
        pub(super) raw: NonNull<c_void>,
        _vec_bufs: Vec<GpuBuf>,
    }

    unsafe impl Send for RunnerInner {}

    impl Drop for RunnerInner {
        fn drop(&mut self) {
            unsafe { vxt_metal_llm_destroy(self.raw.as_ptr()) };
        }
    }

    pub struct StateInner(NonNull<c_void>);

    unsafe impl Send for StateInner {}

    impl Drop for StateInner {
        fn drop(&mut self) {
            unsafe { vxt_metal_llm_state_destroy(self.0.as_ptr()) };
        }
    }

    impl RunnerInner {
        pub fn new(
            eng: &MetalEngine,
            cfg: &ModelConfig,
            layers: &[MetalLlmLayerSpec<'_>],
            output_norm: &[f32],
            output_head: &GpuPackedTensor,
        ) -> Result<Self> {
            let mut vec_bufs = Vec::with_capacity(layers.len() * 6 + 2);
            fn up(eng: &MetalEngine, bufs: &mut Vec<GpuBuf>, s: &[f32]) -> Result<*mut c_void> {
                let b = eng.upload_f32_buf(s)?;
                let p = b.as_ptr();
                bufs.push(b);
                Ok(p)
            }
            fn up_opt(
                eng: &MetalEngine,
                bufs: &mut Vec<GpuBuf>,
                opt: Option<&[f32]>,
            ) -> Result<*mut c_void> {
                match opt {
                    Some(s) => up(eng, bufs, s),
                    None => Ok(null_mut()),
                }
            }

            let mut descs = Vec::with_capacity(layers.len());
            for ly in layers {
                let attn_norm_buf = up(eng, &mut vec_bufs, ly.attn_norm)?;
                let post_attn_norm_buf = up(eng, &mut vec_bufs, ly.post_attn_norm)?;
                let desc = match &ly.attn {
                    MetalLlmAttnSpec::LinearDeltaNet {
                        wqkv,
                        wqkv_gate,
                        ssm_alpha,
                        ssm_beta,
                        ssm_out,
                        ssm_conv1d,
                        ssm_a,
                        ssm_dt_bias,
                        ssm_norm,
                    } => VxtLlmLayerDesc {
                        is_recurrent: 1,
                        gated_q: 0,
                        attn_norm_buf,
                        post_attn_norm_buf,
                        op_a0: wqkv.descriptor_op(),
                        op_a1: wqkv_gate.descriptor_op(),
                        op_a2: ssm_alpha.descriptor_op(),
                        op_a3: ssm_beta.descriptor_op(),
                        op_out: ssm_out.descriptor_op(),
                        vec0_buf: up(eng, &mut vec_bufs, ssm_conv1d)?,
                        vec1_buf: up(eng, &mut vec_bufs, ssm_a)?,
                        vec2_buf: up(eng, &mut vec_bufs, ssm_dt_bias)?,
                        vec3_buf: up(eng, &mut vec_bufs, ssm_norm)?,
                        vec4_buf: null_mut(),
                        op_ffn_gate: ly.ffn_gate.descriptor_op(),
                        op_ffn_up: ly.ffn_up.descriptor_op(),
                        op_ffn_down: ly.ffn_down.descriptor_op(),
                    },
                    MetalLlmAttnSpec::FullGqa {
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
                    } => VxtLlmLayerDesc {
                        is_recurrent: 0,
                        gated_q: *gated_q as u32,
                        attn_norm_buf,
                        post_attn_norm_buf,
                        op_a0: wq.descriptor_op(),
                        op_a1: wk.descriptor_op(),
                        op_a2: wv.descriptor_op(),
                        op_a3: zero_op(),
                        op_out: wo.descriptor_op(),
                        vec0_buf: up_opt(eng, &mut vec_bufs, *q_norm)?,
                        vec1_buf: up_opt(eng, &mut vec_bufs, *k_norm)?,
                        vec2_buf: up_opt(eng, &mut vec_bufs, *q_bias)?,
                        vec3_buf: up_opt(eng, &mut vec_bufs, *k_bias)?,
                        vec4_buf: up_opt(eng, &mut vec_bufs, *v_bias)?,
                        op_ffn_gate: ly.ffn_gate.descriptor_op(),
                        op_ffn_up: ly.ffn_up.descriptor_op(),
                        op_ffn_down: ly.ffn_down.descriptor_op(),
                    },
                };
                descs.push(desc);
            }

            let out_norm_buf = up(eng, &mut vec_bufs, output_norm)?;
            let out_head_op = output_head.descriptor_op();
            let c_cfg = VxtLlmConfig {
                n_layer: cfg.n_layer as u32,
                n_embd: cfg.n_embd as u32,
                n_ff: cfg.n_ff as u32,
                n_vocab: cfg.n_vocab as u32,
                n_head: cfg.n_head as u32,
                n_head_kv: cfg.n_head_kv as u32,
                head_dim: cfg.head_dim as u32,
                rope_dim: cfg.rope_dim as u32,
                rope_neox: cfg.rope_neox as u32,
                rope_freq_base: cfg.rope_freq_base,
                rms_norm_eps: cfg.rms_norm_eps,
                ssm_conv_kernel: cfg.ssm_conv_kernel as u32,
                ssm_state_size: cfg.ssm_state_size as u32,
                ssm_group_count: cfg.ssm_group_count as u32,
                ssm_inner_size: cfg.ssm_inner_size as u32,
                ssm_dt_rank: cfg.ssm_dt_rank as u32,
            };

            let mut err_buf = [0u8; 512];
            let raw = unsafe {
                vxt_metal_llm_create(
                    eng.raw_ctx().as_ptr(),
                    &c_cfg,
                    descs.as_ptr(),
                    out_norm_buf,
                    &out_head_op,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len(),
                )
            };
            let Some(nn) = NonNull::new(raw) else {
                return Err(Error::Backend(cbuf_to_str(&err_buf)));
            };
            Ok(Self {
                raw: nn,
                _vec_bufs: vec_bufs,
            })
        }

        pub fn enable_cb4_greedy(&mut self, embedding: &GpuPackedTensor, centroids: &[f32; 16]) -> Result<()> {
            let op = embedding.descriptor_op();
            let mut err = [0u8; 512];
            let rc = unsafe { vxt_metal_llm_enable_cb4_greedy(self.raw.as_ptr(), &op,
                centroids.as_ptr(), err.as_mut_ptr().cast(), err.len()) };
            if rc != 0 { return Err(Error::Backend(cbuf_to_str(&err))); }
            Ok(())
        }

        pub fn greedy_step(&mut self, state: &mut StateInner, token: u32, pos: usize) -> Result<u32> {
            let pos = u32::try_from(pos).map_err(|_| Error::Backend("Metal position exceeds u32".into()))?;
            let mut next = 0;
            let mut err = [0u8; 512];
            let rc = unsafe { vxt_metal_llm_greedy_step(self.raw.as_ptr(), state.0.as_ptr(), token,
                pos, &mut next, err.as_mut_ptr().cast(), err.len()) };
            if rc != 0 { return Err(Error::Backend(cbuf_to_str(&err))); }
            Ok(next)
        }

        pub fn new_state(&self) -> Result<StateInner> {
            let mut err_buf = [0u8; 512];
            let raw = unsafe {
                vxt_metal_llm_state_create(
                    self.raw.as_ptr(),
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len(),
                )
            };
            NonNull::new(raw)
                .map(StateInner)
                .ok_or_else(|| Error::Backend(cbuf_to_str(&err_buf)))
        }

        pub fn step(
            &mut self,
            state: &mut StateInner,
            cur_init: &[f32],
            pos: usize,
            logits_out: Option<&mut [f32]>,
        ) -> Result<()> {
            let (comp, l_ptr) = match logits_out {
                Some(s) => (1u32, s.as_mut_ptr()),
                None => (0u32, null_mut()),
            };
            let mut err_buf = [0u8; 512];
            let rc = unsafe {
                vxt_metal_llm_step(
                    self.raw.as_ptr(),
                    state.0.as_ptr(),
                    cur_init.as_ptr(),
                    pos as u32,
                    comp,
                    l_ptr,
                    err_buf.as_mut_ptr() as *mut c_char,
                    err_buf.len(),
                )
            };
            if rc != 0 {
                return Err(Error::Backend(cbuf_to_str(&err_buf)));
            }
            Ok(())
        }
    }
}

pub(crate) struct MetalLlmState {
    #[cfg(vxt_has_metal)]
    inner: imp::StateInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

pub(crate) struct MetalLlmRunner {
    #[cfg(vxt_has_metal)]
    inner: imp::RunnerInner,
    #[cfg(not(vxt_has_metal))]
    _priv: (),
}

impl MetalLlmRunner {
    #[cfg(all(test, vxt_has_metal))]
    pub(crate) fn raw_ptr(&self) -> *mut core::ffi::c_void {
        self.inner.raw.as_ptr()
    }

    pub fn new(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        layers: &[MetalLlmLayerSpec<'_>],
        output_norm: &[f32],
        output_head: &GpuPackedTensor,
    ) -> Result<Self> {
        #[cfg(vxt_has_metal)]
        {
            Ok(Self {
                inner: imp::RunnerInner::new(eng, cfg, layers, output_norm, output_head)?,
            })
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (eng, cfg, layers, output_norm, output_head);
            Err(Error::Unsupported("Metal LLM runner unavailable".into()))
        }
    }

    pub fn enable_cb4_greedy(&mut self, embedding: &GpuPackedTensor, centroids: &[f32; 16]) -> Result<()> {
        #[cfg(vxt_has_metal)] { self.inner.enable_cb4_greedy(embedding, centroids) }
        #[cfg(not(vxt_has_metal))] { let _ = (embedding, centroids); Err(Error::Unsupported("Metal unavailable".into())) }
    }

    pub fn greedy_step(&mut self, state: &mut MetalLlmState, token: u32, pos: usize) -> Result<u32> {
        #[cfg(vxt_has_metal)] { self.inner.greedy_step(&mut state.inner, token, pos) }
        #[cfg(not(vxt_has_metal))] { let _ = (state, token, pos); Err(Error::Unsupported("Metal unavailable".into())) }
    }

    pub fn new_state(&self) -> Result<MetalLlmState> {
        #[cfg(vxt_has_metal)]
        {
            Ok(MetalLlmState {
                inner: self.inner.new_state()?,
            })
        }
        #[cfg(not(vxt_has_metal))]
        {
            Err(Error::Unsupported("Metal LLM runner unavailable".into()))
        }
    }

    pub fn step(
        &mut self,
        state: &mut MetalLlmState,
        cur_init: &[f32],
        pos: usize,
        logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        #[cfg(vxt_has_metal)]
        {
            self.inner.step(&mut state.inner, cur_init, pos, logits_out)
        }
        #[cfg(not(vxt_has_metal))]
        {
            let _ = (state, cur_init, pos, logits_out);
            Err(Error::Unsupported("Metal LLM runner unavailable".into()))
        }
    }

    pub(crate) fn upload_and_build(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        token_embd: &mut MatWeight,
        output_head: &mut Option<MatWeight>,
        layers: &mut [DecoderLayer],
        output_norm: &[f32],
    ) -> Result<Option<Self>> {
        token_embd.upload_gpu(eng)?;
        if let Some(h) = output_head.as_mut() {
            h.upload_gpu(eng)?;
        }
        for ly in layers.iter_mut() {
            ly.ffn_gate.upload_gpu(eng)?;
            ly.ffn_up.upload_gpu(eng)?;
            ly.ffn_down.upload_gpu(eng)?;
            match &mut ly.attn {
                LayerAttn::LinearDeltaNet {
                    wqkv,
                    wqkv_gate,
                    ssm_alpha,
                    ssm_beta,
                    ssm_out,
                    ..
                } => {
                    wqkv.upload_gpu(eng)?;
                    wqkv_gate.upload_gpu(eng)?;
                    ssm_alpha.upload_gpu(eng)?;
                    ssm_beta.upload_gpu(eng)?;
                    ssm_out.upload_gpu(eng)?;
                }
                LayerAttn::FullGqa { wq, wk, wv, wo, .. } => {
                    wq.upload_gpu(eng)?;
                    wk.upload_gpu(eng)?;
                    wv.upload_gpu(eng)?;
                    wo.upload_gpu(eng)?;
                }
            }
        }
        let head_ref = output_head.as_ref().unwrap_or(token_embd);
        Ok(Self::try_from_layers(eng, cfg, layers, output_norm, head_ref))
    }

    pub(crate) fn try_from_layers(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        layers: &[DecoderLayer],
        output_norm: &[f32],
        head: &MatWeight,
    ) -> Option<Self> {
        if cfg.ffn_gelu
            || (cfg.embed_scale - 1.0).abs() > 1e-6
            || layers
                .iter()
                .any(|l| l.attn_post_norm.is_some() || l.ffn_post_norm.is_some())
        {
            return None;
        }
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
        Self::new(eng, cfg, &specs, output_norm, head_gpu).ok()
    }
}
