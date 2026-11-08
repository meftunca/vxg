//! Low-level Objective-C++ FFI bindings for `MetalTurboRunner` (`metal_llm_turbo.mm`).

#![cfg(vxt_has_metal)]

use core::ffi::{c_char, c_void};
use core::ptr::{null_mut, NonNull};

use crate::format::PackedTensor;
use crate::fp16::f32_to_f16;
use crate::gguf_meta::ModelConfig;
use crate::metal::MetalEngine;
use crate::metal_ffi::{cbuf_to_str, GpuBuf};
use crate::metal_llm_turbo::{is_turbo_compatible, TurboAttnCpuSpec, TurboLayerCpuSpec};
use crate::wht_turbo::{turbo_fwht32_padded, TurboWhtMatrix};
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
struct VxtTurboMat {
    bits: u32,
    rows: u32,
    cols: u32,
    padded_cols: u32,
    scales_buf: *mut c_void,
    packed_buf: *mut c_void,
}

impl VxtTurboMat {
    fn zero() -> Self {
        Self {
            bits: 0,
            rows: 0,
            cols: 0,
            padded_cols: 0,
            scales_buf: null_mut(),
            packed_buf: null_mut(),
        }
    }
}

#[repr(C)]
struct VxtTurboLayerDesc {
    is_recurrent: u32,
    gated_q: u32,
    has_split_gate: u32,
    gqa_flags: u32,
    attn_norm_buf: *mut c_void,
    post_attn_norm_buf: *mut c_void,
    w_qkv: VxtTurboMat,
    w_gate_sep: VxtTurboMat,
    ab_rot_f16_buf: *mut c_void,
    w_out: VxtTurboMat,
    vec0_buf: *mut c_void,
    vec1_buf: *mut c_void,
    vec2_buf: *mut c_void,
    vec3_buf: *mut c_void,
    vec4_buf: *mut c_void,
    w_ffn_gu: VxtTurboMat,
    w_ffn_down: VxtTurboMat,
}

extern "C" {
    fn vxt_metal_turbo_create(
        gemv_ctx: *mut c_void,
        cfg: *const VxtLlmConfig,
        embed_mat: *const VxtTurboMat,
        layers: *const VxtTurboLayerDesc,
        output_norm_buf: *mut c_void,
        output_head_mat: *const VxtTurboMat,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_turbo_destroy(runner: *mut c_void);
    fn vxt_metal_turbo_state_create(
        runner: *mut c_void,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_turbo_state_destroy(state: *mut c_void);
    fn vxt_metal_turbo_step_batch(
        runner: *mut c_void,
        state: *mut c_void,
        tok_ids: *const u32,
        batch_m: u32,
        pos_base: u32,
        save_backup: u32,
        compute_mode: u32,
        logits_out: *mut f32,
        argmax_ids_out: *mut u32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_generation_checkpoint(
        runner: *mut c_void,
        state: *mut c_void,
        restore: u32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_ring_set_token(runner: *mut c_void, slot: u32, tok_id: u32);
    fn vxt_metal_turbo_ring_get_out(runner: *mut c_void, slot: u32) -> u32;
    fn vxt_metal_turbo_init_seen(
        state: *mut c_void,
        seen_ids: *const u32,
        n_seen: usize,
        presence_pen: f32,
    );
    fn vxt_metal_turbo_step_async_enqueue(
        runner: *mut c_void,
        state: *mut c_void,
        base_slot: u32,
        step_count: u32,
        pos_base: u32,
        use_sketch: u32,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_turbo_wait_cmd(raw_cmd: *mut c_void, err_out: *mut c_char, err_cap: usize) -> i32;
}

pub(crate) struct TurboRunnerInner {
    raw: NonNull<c_void>,
}

unsafe impl Send for TurboRunnerInner {}

impl Drop for TurboRunnerInner {
    fn drop(&mut self) {
        unsafe { vxt_metal_turbo_destroy(self.raw.as_ptr()) };
    }
}

pub(crate) struct TurboStateInner(NonNull<c_void>);

unsafe impl Send for TurboStateInner {}

impl Drop for TurboStateInner {
    fn drop(&mut self) {
        unsafe { vxt_metal_turbo_state_destroy(self.0.as_ptr()) };
    }
}

fn interleave_ffn_gu(fg: &TurboWhtMatrix, fu: &TurboWhtMatrix) -> Option<TurboWhtMatrix> {
    if fg.rows == fu.rows
        && fg.rows % 32 == 0
        && fg.bits == fu.bits
        && fg.cols == fu.cols
        && fg.padded_cols == fu.padded_cols
    {
        let n_blks = fg.rows / 32;
        let steps = fg.padded_cols / 256;
        let wps = TurboWhtMatrix::words_per_tile_step(fg.bits);
        let sc_blk = 8 * steps * 8;
        let pw_blk = 8 * steps * wps;
        let mut scales_f16 = Vec::with_capacity(fg.scales_f16.len() * 2);
        let mut packed_words = Vec::with_capacity(fg.packed_words.len() * 2);
        for b in 0..n_blks {
            scales_f16.extend_from_slice(&fg.scales_f16[b * sc_blk..(b + 1) * sc_blk]);
            scales_f16.extend_from_slice(&fu.scales_f16[b * sc_blk..(b + 1) * sc_blk]);
            packed_words.extend_from_slice(&fg.packed_words[b * pw_blk..(b + 1) * pw_blk]);
            packed_words.extend_from_slice(&fu.packed_words[b * pw_blk..(b + 1) * pw_blk]);
        }
        Some(TurboWhtMatrix {
            bits: fg.bits,
            rows: fg.rows * 2,
            cols: fg.cols,
            padded_rows: fg.padded_rows * 2,
            padded_cols: fg.padded_cols,
            disk_bytes: 0,
            scales_f16,
            packed_words,
        })
    } else {
        TurboWhtMatrix::concat_vertical(&[fg, fu])
    }
}

fn upload_tmat(
    ctx: NonNull<c_void>,
    pool: &mut Vec<GpuBuf>,
    m: &TurboWhtMatrix,
) -> Result<VxtTurboMat> {
    let sc = GpuBuf::upload(ctx, &m.scales_f16)?;
    let pw = GpuBuf::upload(ctx, &m.packed_words)?;
    let desc = VxtTurboMat {
        bits: m.bits as u32,
        rows: m.rows as u32,
        cols: m.cols as u32,
        padded_cols: m.padded_cols as u32,
        scales_buf: sc.as_ptr(),
        packed_buf: pw.as_ptr(),
    };
    pool.push(sc);
    pool.push(pw);
    Ok(desc)
}

fn upload_ab_rot_f16(
    ctx: NonNull<c_void>,
    pool: &mut Vec<GpuBuf>,
    alpha: &PackedTensor,
    beta: &PackedTensor,
    n_vh: usize,
    dim: usize,
) -> Result<*mut c_void> {
    let padded_cols = dim.div_ceil(256) * 256;
    let a_f32 = alpha.decode_f32();
    let b_f32 = beta.decode_f32();
    let mut rot_f16 = Vec::with_capacity(2 * n_vh * padded_cols);
    for src in [&a_f32[..], &b_f32[..]] {
        for r in 0..n_vh {
            let row = &src[r * dim..(r + 1) * dim];
            let r_rot = turbo_fwht32_padded(row, padded_cols);
            rot_f16.extend(r_rot.into_iter().map(f32_to_f16));
        }
    }
    let b = GpuBuf::upload(ctx, &rot_f16)?;
    let p = b.as_ptr();
    pool.push(b);
    Ok(p)
}

impl TurboRunnerInner {
    pub fn try_new(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        token_embd: &PackedTensor,
        output_norm: &[f32],
        output_head: Option<&PackedTensor>,
        layers: &[TurboLayerCpuSpec<'_>],
    ) -> Result<Option<Self>> {
        if !is_turbo_compatible(cfg, token_embd, output_head, layers) {
            return Ok(None);
        }
        let ctx = eng.raw_ctx();
        let mut bufs = Vec::with_capacity(layers.len() * 14 + 8);

        let up_f32 = |pool: &mut Vec<GpuBuf>, s: &[f32]| -> Result<*mut c_void> {
            let b = GpuBuf::upload(ctx, s)?;
            let p = b.as_ptr();
            pool.push(b);
            Ok(p)
        };
        let up_opt = |pool: &mut Vec<GpuBuf>, opt: Option<&[f32]>| -> Result<*mut c_void> {
            match opt {
                Some(s) => up_f32(pool, s),
                None => Ok(null_mut()),
            }
        };

        let PackedTensor::TurboWht(emb_m) = token_embd else {
            return Ok(None);
        };
        let embed_desc = upload_tmat(ctx, &mut bufs, emb_m)?;
        let head_desc = match output_head {
            Some(PackedTensor::TurboWht(hm)) => upload_tmat(ctx, &mut bufs, hm)?,
            None => embed_desc,
            _ => return Ok(None),
        };
        let output_norm_buf = up_f32(&mut bufs, output_norm)?;

        let mut descs = Vec::with_capacity(layers.len());
        for ly in layers {
            let attn_norm_buf = up_f32(&mut bufs, ly.attn_norm)?;
            let post_attn_norm_buf = up_f32(&mut bufs, ly.post_attn_norm)?;
            let (
                PackedTensor::TurboWht(fg),
                PackedTensor::TurboWht(fu),
                PackedTensor::TurboWht(fd),
            ) = (ly.ffn_gate, ly.ffn_up, ly.ffn_down)
            else {
                return Ok(None);
            };
            let Some(gu_cat) = interleave_ffn_gu(fg, fu) else {
                return Ok(None);
            };
            let w_ffn_gu = upload_tmat(ctx, &mut bufs, &gu_cat)?;
            let w_ffn_down = upload_tmat(ctx, &mut bufs, fd)?;

            match &ly.attn {
                TurboAttnCpuSpec::LinearDeltaNet {
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
                    let (
                        PackedTensor::TurboWht(mq),
                        PackedTensor::TurboWht(mg),
                        PackedTensor::TurboWht(mo),
                    ) = (wqkv, wqkv_gate, ssm_out)
                    else {
                        return Ok(None);
                    };
                    let (has_split_gate, w_qkv, w_gate_sep) =
                        if let Some(cat) = TurboWhtMatrix::concat_vertical(&[mq, mg]) {
                            (
                                0u32,
                                upload_tmat(ctx, &mut bufs, &cat)?,
                                VxtTurboMat::zero(),
                            )
                        } else {
                            (
                                1u32,
                                upload_tmat(ctx, &mut bufs, mq)?,
                                upload_tmat(ctx, &mut bufs, mg)?,
                            )
                        };
                    let ab_rot_f16_buf = upload_ab_rot_f16(
                        ctx,
                        &mut bufs,
                        ssm_alpha,
                        ssm_beta,
                        cfg.ssm_dt_rank,
                        cfg.n_embd,
                    )?;
                    descs.push(VxtTurboLayerDesc {
                        is_recurrent: 1,
                        gated_q: 0,
                        has_split_gate,
                        gqa_flags: 0,
                        attn_norm_buf,
                        post_attn_norm_buf,
                        w_qkv,
                        w_gate_sep,
                        ab_rot_f16_buf,
                        w_out: upload_tmat(ctx, &mut bufs, mo)?,
                        vec0_buf: up_f32(&mut bufs, ssm_conv1d)?,
                        vec1_buf: up_f32(&mut bufs, ssm_a)?,
                        vec2_buf: up_f32(&mut bufs, ssm_dt_bias)?,
                        vec3_buf: up_f32(&mut bufs, ssm_norm)?,
                        vec4_buf: null_mut(),
                        w_ffn_gu,
                        w_ffn_down,
                    });
                }
                TurboAttnCpuSpec::FullGqa {
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
                    let (
                        PackedTensor::TurboWht(mq),
                        PackedTensor::TurboWht(mk),
                        PackedTensor::TurboWht(mv),
                        PackedTensor::TurboWht(mo),
                    ) = (wq, wk, wv, wo)
                    else {
                        return Ok(None);
                    };
                    let (has_split_gate, w_qkv, w_gate_sep) =
                        if let Some(qkv_cat) = TurboWhtMatrix::concat_vertical(&[mq, mk, mv]) {
                            (
                                0u32,
                                upload_tmat(ctx, &mut bufs, &qkv_cat)?,
                                VxtTurboMat::zero(),
                            )
                        } else if let Some(kv_cat) = TurboWhtMatrix::concat_vertical(&[mk, mv]) {
                            (
                                1u32,
                                upload_tmat(ctx, &mut bufs, mq)?,
                                upload_tmat(ctx, &mut bufs, &kv_cat)?,
                            )
                        } else {
                            return Ok(None);
                        };
                    let flags = (*gated_q as u32)
                        | ((cfg.rope_neox as u32) << 1)
                        | ((q_norm.is_some() as u32) << 2)
                        | ((k_norm.is_some() as u32) << 3)
                        | ((q_bias.is_some() as u32) << 4)
                        | ((k_bias.is_some() as u32) << 5)
                        | ((v_bias.is_some() as u32) << 6);
                    descs.push(VxtTurboLayerDesc {
                        is_recurrent: 0,
                        gated_q: *gated_q as u32,
                        has_split_gate,
                        gqa_flags: flags,
                        attn_norm_buf,
                        post_attn_norm_buf,
                        w_qkv,
                        w_gate_sep,
                        ab_rot_f16_buf: null_mut(),
                        w_out: upload_tmat(ctx, &mut bufs, mo)?,
                        vec0_buf: up_opt(&mut bufs, *q_norm)?,
                        vec1_buf: up_opt(&mut bufs, *k_norm)?,
                        vec2_buf: up_opt(&mut bufs, *q_bias)?,
                        vec3_buf: up_opt(&mut bufs, *k_bias)?,
                        vec4_buf: up_opt(&mut bufs, *v_bias)?,
                        w_ffn_gu,
                        w_ffn_down,
                    });
                }
            }
        }

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
        let ptr = unsafe {
            vxt_metal_turbo_create(
                ctx.as_ptr(),
                &c_cfg,
                &embed_desc,
                descs.as_ptr(),
                output_norm_buf,
                &head_desc,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        let Some(raw) = NonNull::new(ptr) else {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        };
        drop(bufs);
        Ok(Some(Self { raw }))
    }

    pub fn new_state(&self) -> Result<TurboStateInner> {
        let mut err_buf = [0u8; 512];
        let ptr = unsafe {
            vxt_metal_turbo_state_create(
                self.raw.as_ptr(),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        NonNull::new(ptr)
            .map(TurboStateInner)
            .ok_or_else(|| Error::Backend(cbuf_to_str(&err_buf)))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_batch(
        &self,
        state: &mut TurboStateInner,
        tok_ids: &[u32],
        pos_base: u32,
        save_backup: bool,
        compute_mode: u32,
        logits_out: Option<&mut [f32]>,
        argmax_out: Option<&mut [u32]>,
    ) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let l_ptr = logits_out.map_or(null_mut(), |s| s.as_mut_ptr());
        let a_ptr = argmax_out.map_or(null_mut(), |s| s.as_mut_ptr());
        let rc = unsafe {
            vxt_metal_turbo_step_batch(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                tok_ids.as_ptr(),
                tok_ids.len() as u32,
                pos_base,
                save_backup as u32,
                compute_mode,
                l_ptr,
                a_ptr,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        if rc != 0 {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(())
    }

    pub fn generation_checkpoint(&self, state: &mut TurboStateInner, restore: bool) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_turbo_generation_checkpoint(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                restore as u32,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        if rc != 0 {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(())
    }

    pub fn enqueue_async(
        &self,
        state: &mut TurboStateInner,
        base_slot: u32,
        step_count: u32,
        pos_base: u32,
        use_sketch: bool,
    ) -> Result<*mut c_void> {
        let mut err_buf = [0u8; 512];
        let cmd = unsafe {
            vxt_metal_turbo_step_async_enqueue(
                self.raw.as_ptr(),
                state.0.as_ptr(),
                base_slot,
                step_count,
                pos_base,
                use_sketch as u32,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        if cmd.is_null() {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(cmd)
    }

    pub fn wait_cmd(&self, cmd: *mut c_void) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_turbo_wait_cmd(cmd, err_buf.as_mut_ptr() as *mut c_char, err_buf.len())
        };
        if rc != 0 {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(())
    }

    #[inline]
    pub fn ring_set(&self, slot: u32, tok: u32) {
        unsafe { vxt_metal_turbo_ring_set_token(self.raw.as_ptr(), slot, tok) };
    }

    #[inline]
    pub fn ring_get(&self, slot: u32) -> u32 {
        unsafe { vxt_metal_turbo_ring_get_out(self.raw.as_ptr(), slot) }
    }

    #[inline]
    pub fn init_seen(&self, state: &mut TurboStateInner, seen_ids: &[u32], presence_pen: f32) {
        unsafe {
            vxt_metal_turbo_init_seen(
                state.0.as_ptr(),
                seen_ids.as_ptr(),
                seen_ids.len(),
                presence_pen,
            )
        };
    }
}
