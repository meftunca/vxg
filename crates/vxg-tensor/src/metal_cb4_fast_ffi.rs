//! Low-level Objective-C++ FFI bindings for `MetalCb4FastRunner` (`metal_cb4_fast.mm`).

#![cfg(vxt_has_metal)]

use core::ffi::{c_char, c_void};
use core::ptr::{null, null_mut, NonNull};

use crate::codebook::CodebookMatrix;
use crate::format::PackedTensor;
use crate::fp16::fit_cubic_codebook;
use crate::gguf_meta::ModelConfig;
use crate::llm::{DecoderLayer, LayerAttn};
use crate::metal::MetalEngine;
use crate::metal_cb4_fast::is_cb4_fast_compatible;
use crate::metal_ffi::cbuf_to_str;
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
#[derive(Clone, Copy)]
struct VxtCb4FastSubMat {
    rows: u32,
    cols: u32,
    padded_cols: u32,
    nnz: u32,
    poly: [f32; 4],
    scales_f16: *const u16,
    packed_words: *const u32,
    outlier_row_ptr: *const u32,
    outliers_packed: *const u32,
}

impl VxtCb4FastSubMat {
    fn zero() -> Self {
        Self {
            rows: 0,
            cols: 0,
            padded_cols: 0,
            nnz: 0,
            poly: [0.0; 4],
            scales_f16: null(),
            packed_words: null(),
            outlier_row_ptr: null(),
            outliers_packed: null(),
        }
    }

    fn from_cb4(cb: &CodebookMatrix) -> Self {
        let (poly, _) = fit_cubic_codebook(&cb.centroids[..16]);
        Self {
            rows: cb.rows as u32,
            cols: cb.cols as u32,
            padded_cols: cb.padded_cols as u32,
            nnz: cb.outliers_packed.len() as u32,
            poly,
            scales_f16: cb.scales_f16.as_ptr(),
            packed_words: cb.packed_words.as_ptr(),
            outlier_row_ptr: cb.outlier_row_ptr.as_ptr(),
            outliers_packed: if cb.outliers_packed.is_empty() {
                null()
            } else {
                cb.outliers_packed.as_ptr()
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VxtCb4FastConcatMat {
    n_subs: u32,
    subs: [VxtCb4FastSubMat; 3],
}

impl VxtCb4FastConcatMat {
    fn from_subs(list: &[&CodebookMatrix]) -> Self {
        let mut subs = [VxtCb4FastSubMat::zero(); 3];
        for (i, cb) in list.iter().take(3).enumerate() {
            subs[i] = VxtCb4FastSubMat::from_cb4(cb);
        }
        Self {
            n_subs: list.len() as u32,
            subs,
        }
    }
}

#[repr(C)]
struct VxtCb4FastLayerUpload {
    is_recurrent: u32,
    gated_q: u32,
    ab_is_strided32: u32,
    _pad: u32,
    attn_norm: *const f32,
    post_attn_norm: *const f32,
    attn_post_norm: *const f32,
    ffn_post_norm: *const f32,
    w_in: VxtCb4FastConcatMat,
    w_out: VxtCb4FastConcatMat,
    w_ffn_gu: VxtCb4FastConcatMat,
    w_ffn_dn: VxtCb4FastConcatMat,
    ab_weights_pad: *const f32,
    vec0: *const f32,
    vec0_len: u32,
    vec1: *const f32,
    vec1_len: u32,
    vec2: *const f32,
    vec2_len: u32,
    vec3: *const f32,
    vec3_len: u32,
    vec4: *const f32,
    vec4_len: u32,
}

extern "C" {
    fn vxt_metal_cb4_fast_create(
        gemv_ctx: *mut c_void,
        cfg: *const VxtLlmConfig,
        embed_scale: f32,
        ffn_gelu: u32,
        embed_in: *const VxtCb4FastConcatMat,
        embed_centroids: *const f32,
        has_untied_head: u32,
        head_in: *const VxtCb4FastConcatMat,
        output_norm: *const f32,
        layers_in: *const VxtCb4FastLayerUpload,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_cb4_fast_destroy(runner: *mut c_void);
    fn vxt_metal_cb4_fast_state_create(
        runner: *mut c_void,
        kv_q8: u32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_cb4_fast_state_destroy(state: *mut c_void);
    fn vxt_metal_cb4_fast_state_kv_bytes(state: *mut c_void) -> usize;
    fn vxt_metal_cb4_fast_step_batch(
        runner: *mut c_void,
        state: *mut c_void,
        tok_ids: *const u32,
        batch_m: u32,
        pos_base: u32,
        rope_pos_base: u32,
        compute_mode: u32,
        logits_out: *mut f32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> i32;
    fn vxt_metal_cb4_fast_step_embeds_batch(
        runner: *mut c_void,
        state: *mut c_void,
        embeds_in: *const f32,
        mrope_pos4: *const u32,
        mrope_s1: u32,
        mrope_s2: u32,
        batch_m: u32,
        pos_base: u32,
        compute_mode: u32,
        logits_out: *mut f32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> i32;
    fn vxt_metal_cb4_fast_checkpoint(
        runner: *mut c_void,
        state: *mut c_void,
        restore: u32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> i32;
    fn vxt_metal_cb4_fast_ring_set_token(runner: *mut c_void, slot: u32, tok_id: u32);
    fn vxt_metal_cb4_fast_ring_get_out(runner: *mut c_void, slot: u32) -> u32;
    fn vxt_metal_cb4_fast_ring_get_err(runner: *mut c_void, slot: u32) -> u32;
    fn vxt_metal_cb4_fast_enqueue_async(
        runner: *mut c_void,
        state: *mut c_void,
        base_slot: u32,
        step_count: u32,
        pos_base: u32,
        rope_pos_base: u32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_cb4_fast_wait_cmd(raw_cmd: *mut c_void, err_out: *mut c_char, err_cap: usize)
        -> i32;
}

pub(crate) struct Cb4FastRunnerInner {
    raw: NonNull<c_void>,
}

unsafe impl Send for Cb4FastRunnerInner {}

impl Drop for Cb4FastRunnerInner {
    fn drop(&mut self) {
        unsafe { vxt_metal_cb4_fast_destroy(self.raw.as_ptr()) };
    }
}

pub(crate) struct Cb4FastStateInner(NonNull<c_void>);

unsafe impl Send for Cb4FastStateInner {}

impl Drop for Cb4FastStateInner {
    fn drop(&mut self) {
        unsafe { vxt_metal_cb4_fast_state_destroy(self.0.as_ptr()) };
    }
}

impl Cb4FastStateInner {
    pub(crate) fn kv_bytes(&self) -> usize {
        unsafe { vxt_metal_cb4_fast_state_kv_bytes(self.0.as_ptr()) }
    }
}

fn opt_slice_ptr(opt: Option<&[f32]>) -> (*const f32, u32) {
    match opt {
        Some(s) if !s.is_empty() => (s.as_ptr(), s.len() as u32),
        _ => (null(), 0),
    }
}

fn build_ab_padded(
    ssm_alpha: &PackedTensor,
    ssm_beta: &PackedTensor,
    n_vh: usize,
    n_embd: usize,
    padded_embd: usize,
) -> (Vec<f32>, u32) {
    let is_strided32 = u32::from(matches!(
        (ssm_alpha, ssm_beta),
        (PackedTensor::RawF32 { .. }, PackedTensor::RawF32 { .. })
    ));
    let a = ssm_alpha.decode_f32();
    let b = ssm_beta.decode_f32();
    let mut out = vec![0.0f32; 2 * n_vh * padded_embd];
    for r in 0..n_vh {
        out[r * padded_embd..r * padded_embd + n_embd]
            .copy_from_slice(&a[r * n_embd..(r + 1) * n_embd]);
        out[(n_vh + r) * padded_embd..(n_vh + r) * padded_embd + n_embd]
            .copy_from_slice(&b[r * n_embd..(r + 1) * n_embd]);
    }
    (out, is_strided32)
}

impl Cb4FastRunnerInner {
    pub(crate) fn try_create(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        token_embd: &PackedTensor,
        output_norm: &[f32],
        output_head: Option<&PackedTensor>,
        layers: &[DecoderLayer],
    ) -> Result<Option<Self>> {
        if !is_cb4_fast_compatible(cfg, token_embd, output_head, layers) {
            return Ok(None);
        }
        let PackedTensor::Codebook(emb_cb) = token_embd else {
            return Ok(None);
        };
        let embed_desc = VxtCb4FastConcatMat::from_subs(&[emb_cb]);
        let (has_untied_head, head_desc) = match output_head {
            Some(PackedTensor::Codebook(h_cb)) => (1u32, VxtCb4FastConcatMat::from_subs(&[h_cb])),
            None => (0u32, embed_desc),
            _ => return Ok(None),
        };

        let padded_embd = (cfg.n_embd + 255) & !255;
        let mut ab_keep: Vec<Vec<f32>> = Vec::with_capacity(layers.len());
        let mut layer_uploads = Vec::with_capacity(layers.len());

        for ly in layers {
            let (PackedTensor::Codebook(fg), PackedTensor::Codebook(fu), PackedTensor::Codebook(fd)) =
                (&ly.ffn_gate.cpu, &ly.ffn_up.cpu, &ly.ffn_down.cpu)
            else {
                return Ok(None);
            };
            let w_ffn_gu = VxtCb4FastConcatMat::from_subs(&[fg, fu]);
            let w_ffn_dn = VxtCb4FastConcatMat::from_subs(&[fd]);
            let (apn_p, _) = opt_slice_ptr(ly.attn_post_norm.as_deref());
            let (fpn_p, _) = opt_slice_ptr(ly.ffn_post_norm.as_deref());

            match &ly.attn {
                LayerAttn::LinearDeltaNet {
                    wqkv, wqkv_gate, ssm_alpha, ssm_beta, ssm_out,
                    ssm_conv1d, ssm_a, ssm_dt_bias, ssm_norm,
                } => {
                    let (PackedTensor::Codebook(mq), PackedTensor::Codebook(mg), PackedTensor::Codebook(mo)) =
                        (&wqkv.cpu, &wqkv_gate.cpu, &ssm_out.cpu)
                    else {
                        return Ok(None);
                    };
                    let (ab_vec, ab_is_strided32) = build_ab_padded(
                        &ssm_alpha.cpu, &ssm_beta.cpu, cfg.ssm_dt_rank, cfg.n_embd, padded_embd,
                    );
                    let ab_ptr = ab_vec.as_ptr();
                    ab_keep.push(ab_vec);

                    layer_uploads.push(VxtCb4FastLayerUpload {
                        is_recurrent: 1, gated_q: 0, ab_is_strided32, _pad: 0,
                        attn_norm: ly.attn_norm.as_ptr(),
                        post_attn_norm: ly.post_attn_norm.as_ptr(),
                        attn_post_norm: apn_p,
                        ffn_post_norm: fpn_p,
                        w_in: VxtCb4FastConcatMat::from_subs(&[mq, mg]),
                        w_out: VxtCb4FastConcatMat::from_subs(&[mo]),
                        w_ffn_gu, w_ffn_dn, ab_weights_pad: ab_ptr,
                        vec0: ssm_conv1d.as_ptr(), vec0_len: ssm_conv1d.len() as u32,
                        vec1: ssm_a.as_ptr(), vec1_len: ssm_a.len() as u32,
                        vec2: ssm_dt_bias.as_ptr(), vec2_len: ssm_dt_bias.len() as u32,
                        vec3: ssm_norm.as_ptr(), vec3_len: ssm_norm.len() as u32,
                        vec4: null(), vec4_len: 0,
                    });
                }
                LayerAttn::FullGqa {
                    wq, wk, wv, wo, q_norm, k_norm, q_bias, k_bias, v_bias, gated_q,
                } => {
                    let (
                        PackedTensor::Codebook(mq), PackedTensor::Codebook(mk),
                        PackedTensor::Codebook(mv), PackedTensor::Codebook(mo),
                    ) = (&wq.cpu, &wk.cpu, &wv.cpu, &wo.cpu)
                    else {
                        return Ok(None);
                    };
                    let (v0, l0) = opt_slice_ptr(q_norm.as_deref());
                    let (v1, l1) = opt_slice_ptr(k_norm.as_deref());
                    let (v2, l2) = opt_slice_ptr(q_bias.as_deref());
                    let (v3, l3) = opt_slice_ptr(k_bias.as_deref());
                    let (v4, l4) = opt_slice_ptr(v_bias.as_deref());

                    layer_uploads.push(VxtCb4FastLayerUpload {
                        is_recurrent: 0, gated_q: u32::from(*gated_q), ab_is_strided32: 0, _pad: 0,
                        attn_norm: ly.attn_norm.as_ptr(),
                        post_attn_norm: ly.post_attn_norm.as_ptr(),
                        attn_post_norm: apn_p,
                        ffn_post_norm: fpn_p,
                        w_in: VxtCb4FastConcatMat::from_subs(&[mq, mk, mv]),
                        w_out: VxtCb4FastConcatMat::from_subs(&[mo]),
                        w_ffn_gu, w_ffn_dn, ab_weights_pad: null(),
                        vec0: v0, vec0_len: l0, vec1: v1, vec1_len: l1,
                        vec2: v2, vec2_len: l2, vec3: v3, vec3_len: l3,
                        vec4: v4, vec4_len: l4,
                    });
                }
            }
        }

        let c_cfg = VxtLlmConfig {
            n_layer: cfg.n_layer as u32, n_embd: cfg.n_embd as u32, n_ff: cfg.n_ff as u32,
            n_vocab: cfg.n_vocab as u32, n_head: cfg.n_head as u32, n_head_kv: cfg.n_head_kv as u32,
            head_dim: cfg.head_dim as u32, rope_dim: cfg.rope_dim as u32,
            rope_neox: u32::from(cfg.rope_neox), rope_freq_base: cfg.rope_freq_base,
            rms_norm_eps: cfg.rms_norm_eps, ssm_conv_kernel: cfg.ssm_conv_kernel as u32,
            ssm_state_size: cfg.ssm_state_size as u32, ssm_group_count: cfg.ssm_group_count as u32,
            ssm_inner_size: cfg.ssm_inner_size as u32, ssm_dt_rank: cfg.ssm_dt_rank as u32,
        };

        let mut err_buf = [0u8; 512];
        let raw = unsafe {
            vxt_metal_cb4_fast_create(
                eng.raw_ctx().as_ptr(),
                &c_cfg,
                cfg.embed_scale,
                u32::from(cfg.ffn_gelu),
                &embed_desc,
                emb_cb.centroids.as_ptr(),
                has_untied_head,
                &head_desc,
                output_norm.as_ptr(),
                layer_uploads.as_ptr(),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        drop(ab_keep);
        let Some(nn) = NonNull::new(raw) else {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        };
        Ok(Some(Self { raw: nn }))
    }

    pub(crate) fn new_state(&self, kv_q8: bool) -> Result<Cb4FastStateInner> {
        let mut err_buf = [0u8; 512];
        let ptr = unsafe {
            vxt_metal_cb4_fast_state_create(
                self.raw.as_ptr(),
                u32::from(kv_q8),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        let Some(nn) = NonNull::new(ptr) else {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        };
        Ok(Cb4FastStateInner(nn))
    }

    pub(crate) fn step_batch(
        &self,
        state: &mut Cb4FastStateInner,
        tok_ids: &[u32],
        pos_base: u32,
        rope_pos_base: u32,
        compute_mode: u32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let l_ptr = logits_out.map_or(null_mut(), |s| s.as_mut_ptr());
        let rc = unsafe {
            vxt_metal_cb4_fast_step_batch(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                tok_ids.as_ptr(),
                tok_ids.len() as u32,
                pos_base,
                rope_pos_base,
                compute_mode,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_embeds_batch(
        &self,
        state: &mut Cb4FastStateInner,
        embeds: &[f32],
        mrope_pos: &[[u32; 4]],
        mrope_section: [usize; 3],
        pos_base: u32,
        compute_mode: u32,
        logits_out: Option<&mut [f32]>,
    ) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let l_ptr = logits_out.map_or(null_mut(), |s| s.as_mut_ptr());
        let rc = unsafe {
            vxt_metal_cb4_fast_step_embeds_batch(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                embeds.as_ptr(),
                mrope_pos.as_ptr() as *const u32,
                mrope_section[1] as u32,
                mrope_section[2] as u32,
                mrope_pos.len() as u32,
                pos_base,
                compute_mode,
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

    pub(crate) fn checkpoint(&self, state: &mut Cb4FastStateInner, restore: bool) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_cb4_fast_checkpoint(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                u32::from(restore),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        if rc != 0 {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(())
    }

    pub(crate) fn ring_set(&self, slot: u32, tok_id: u32) {
        unsafe { vxt_metal_cb4_fast_ring_set_token(self.raw.as_ptr(), slot, tok_id) };
    }

    pub(crate) fn ring_get_out(&self, slot: u32) -> u32 {
        unsafe { vxt_metal_cb4_fast_ring_get_out(self.raw.as_ptr(), slot) }
    }

    pub(crate) fn ring_get_err(&self, slot: u32) -> u32 {
        unsafe { vxt_metal_cb4_fast_ring_get_err(self.raw.as_ptr(), slot) }
    }

    pub(crate) fn enqueue_async(
        &self,
        state: &mut Cb4FastStateInner,
        base_slot: u32,
        step_count: u32,
        pos_base: u32,
        rope_pos_base: u32,
    ) -> Result<NonNull<c_void>> {
        let mut err_buf = [0u8; 512];
        let raw = unsafe {
            vxt_metal_cb4_fast_enqueue_async(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                base_slot,
                step_count,
                pos_base,
                rope_pos_base,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        NonNull::new(raw).ok_or_else(|| Error::Backend(cbuf_to_str(&err_buf)))
    }

    pub(crate) fn wait_cmd(&self, cmd: NonNull<c_void>) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_cb4_fast_wait_cmd(
                cmd.as_ptr(),
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
