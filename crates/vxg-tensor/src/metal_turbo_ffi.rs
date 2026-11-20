//! Low-level Objective-C++ FFI bindings for `MetalTurboRunner` (`metal_llm_turbo.mm`).

#![cfg(vxt_has_metal)]

use core::ffi::{c_char, c_void};
use core::ptr::{null_mut, NonNull};

use crate::format::PackedTensor;
use crate::fp16::f32_to_f16;
use crate::gguf_meta::ModelConfig;
use crate::llm::{DecoderLayer, LayerAttn, MatWeight};
use crate::metal::MetalEngine;
use crate::metal_ffi::cbuf_to_str;
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
struct VxtArenaMatC {
    scales_off: usize,
    packed_off: usize,
    rows: u32,
    cols: u32,
    padded_cols: u32,
    bits: u32,
}

impl VxtArenaMatC {
    fn zero() -> Self {
        Self {
            scales_off: 0,
            packed_off: 0,
            rows: 0,
            cols: 0,
            padded_cols: 0,
            bits: 0,
        }
    }
}

#[repr(C)]
struct VxtArenaLayerDescC {
    is_recurrent: u32,
    gated_q: u32,
    gqa_flags: u32,
    has_split_gate: u32,
    attn_norm_off: usize,
    post_attn_norm_off: usize,
    w_qkv: VxtArenaMatC,
    w_gate_sep: VxtArenaMatC,
    ab_rot_f16_off: usize,
    w_out: VxtArenaMatC,
    vec0_off: usize,
    vec1_off: usize,
    vec2_off: usize,
    vec3_off: usize,
    vec4_off: usize,
    w_ffn_gu: VxtArenaMatC,
    w_ffn_down: VxtArenaMatC,
}

extern "C" {
    fn vxt_metal_turbo_arena_begin(
        gemv_ctx: *mut c_void, cfg: *const VxtLlmConfig, total_arena_bytes: usize,
        out_arena_ptr: *mut *mut c_void, err_out: *mut c_char, err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_turbo_arena_finalize(
        runner: *mut c_void, embed_mat: *const VxtArenaMatC, output_head_mat: *const VxtArenaMatC,
        output_norm_off: usize, layers: *const VxtArenaLayerDescC,
        err_out: *mut c_char, err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_destroy(runner: *mut c_void);
    fn vxt_metal_turbo_state_create(runner: *mut c_void, err_out: *mut c_char, err_cap: usize) -> *mut c_void;
    fn vxt_metal_turbo_state_destroy(state: *mut c_void);
    fn vxt_metal_turbo_step_batch(
        runner: *mut c_void, state: *mut c_void, tok_ids: *const u32, batch_m: u32,
        pos_base: u32, save_backup: u32, compute_mode: u32, logits_out: *mut f32,
        argmax_ids_out: *mut u32, err_out: *mut c_char, err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_rollback_deltanet(
        runner: *mut c_void, state: *mut c_void, accepted_m: u32, batch_m: u32,
        err_out: *mut c_char, err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_generation_checkpoint(
        runner: *mut c_void, state: *mut c_void, restore: u32, err_out: *mut c_char, err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_replay_trunk(
        runner: *mut c_void, state: *mut c_void, tok_ids: *const u32, step_count: u32,
        pos_base: u32, err_out: *mut c_char, err_cap: usize,
    ) -> i32;
    fn vxt_metal_turbo_ring_set_token(runner: *mut c_void, slot: u32, tok_id: u32);
    fn vxt_metal_turbo_ring_get_out(runner: *mut c_void, slot: u32) -> u32;
    fn vxt_metal_turbo_init_seen(state: *mut c_void, seen_ids: *const u32, n_seen: usize, presence_pen: f32);
    fn vxt_metal_turbo_mark_seen(state: *mut c_void, tok_id: u32);
    fn vxt_metal_turbo_step_async_enqueue(
        runner: *mut c_void, state: *mut c_void, base_slot: u32, step_count: u32,
        pos_base: u32, use_sketch: u32, err_out: *mut c_char, err_cap: usize,
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
        let sps = TurboWhtMatrix::scales_per_tile_step(fg.bits);
        let wps = TurboWhtMatrix::words_per_tile_step(fg.bits);
        let sc_blk = 8 * steps * sps;
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

fn build_ab_rot_f16(
    alpha: &PackedTensor,
    beta: &PackedTensor,
    n_vh: usize,
    dim: usize,
) -> Vec<u16> {
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
    rot_f16
}

fn align256(len: usize) -> usize {
    if len == 0 {
        0
    } else {
        ((len + 255) & !255).max(256)
    }
}

fn mat_arena_bytes(m: &TurboWhtMatrix) -> usize {
    align256(m.scales_f16.len() * 2) + align256(m.packed_words.len() * 4)
}

fn concat_compatible(a: &TurboWhtMatrix, b: &TurboWhtMatrix) -> bool {
    a.bits == b.bits
        && a.cols == b.cols
        && a.padded_cols == b.padded_cols
        && a.rows % 4 == 0
        && b.rows % 4 == 0
}

fn can_concat3(a: &TurboWhtMatrix, b: &TurboWhtMatrix, c: &TurboWhtMatrix) -> bool {
    concat_compatible(a, b) && concat_compatible(a, c)
}

fn check_layers_turbo_compatible(
    cfg: &ModelConfig,
    token_embd: &PackedTensor,
    output_head: Option<&PackedTensor>,
    layers: &[DecoderLayer],
) -> bool {
    if cfg.ffn_gelu || (cfg.embed_scale - 1.0).abs() > 1e-6 {
        return false;
    }
    if !matches!(token_embd, PackedTensor::TurboWht(_)) {
        return false;
    }
    if let Some(h) = output_head {
        if !matches!(h, PackedTensor::TurboWht(_)) {
            return false;
        }
    }
    for ly in layers {
        if ly.attn_post_norm.is_some() || ly.ffn_post_norm.is_some() {
            return false;
        }
        let (
            PackedTensor::TurboWht(fg),
            PackedTensor::TurboWht(fu),
            PackedTensor::TurboWht(_),
        ) = (&ly.ffn_gate.cpu, &ly.ffn_up.cpu, &ly.ffn_down.cpu)
        else {
            return false;
        };
        if !concat_compatible(fg, fu) {
            return false;
        }
        match &ly.attn {
            LayerAttn::LinearDeltaNet {
                wqkv,
                wqkv_gate,
                ssm_out,
                ..
            } => {
                let dv = cfg.ssm_inner_size.checked_div(cfg.ssm_dt_rank).unwrap_or(0);
                if dv == 0 || dv % 32 != 0 || dv > 512 || cfg.ssm_state_size > 512 {
                    return false;
                }
                if !matches!(
                    (&wqkv.cpu, &wqkv_gate.cpu, &ssm_out.cpu),
                    (
                        PackedTensor::TurboWht(_),
                        PackedTensor::TurboWht(_),
                        PackedTensor::TurboWht(_)
                    )
                ) {
                    return false;
                }
            }
            LayerAttn::FullGqa { wq, wk, wv, wo, .. } => {
                if cfg.head_dim == 0 || cfg.head_dim % 32 != 0 || cfg.head_dim > 512 {
                    return false;
                }
                let (
                    PackedTensor::TurboWht(mq),
                    PackedTensor::TurboWht(mk),
                    PackedTensor::TurboWht(mv),
                    PackedTensor::TurboWht(_),
                ) = (&wq.cpu, &wk.cpu, &wv.cpu, &wo.cpu)
                else {
                    return false;
                };
                if !concat_compatible(mk, mv) || mq.rows % 4 != 0 {
                    return false;
                }
            }
        }
    }
    true
}

struct ArenaWriter {
    base: *mut u8,
    cursor: usize,
    cap: usize,
}

impl ArenaWriter {
    fn write_slice<T>(&mut self, s: &[T]) -> usize {
        let len = std::mem::size_of_val(s);
        if len == 0 {
            return 0;
        }
        let step = align256(len);
        assert!(
            self.cursor + step <= self.cap,
            "Metal Turbo arena overflow: {} + {} > {}",
            self.cursor,
            step,
            self.cap
        );
        let off = self.cursor;
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr() as *const u8, self.base.add(off), len);
        }
        self.cursor += step;
        off
    }

    fn write_opt<T>(&mut self, opt: Option<&[T]>) -> usize {
        match opt {
            Some(s) => self.write_slice(s),
            None => 0,
        }
    }

    fn write_mat(&mut self, m: &TurboWhtMatrix) -> VxtArenaMatC {
        let scales_off = self.write_slice(&m.scales_f16);
        let packed_off = self.write_slice(&m.packed_words);
        VxtArenaMatC {
            scales_off,
            packed_off,
            rows: m.rows as u32,
            cols: m.cols as u32,
            padded_cols: m.padded_cols as u32,
            bits: m.bits as u32,
        }
    }
}

fn empty_mat_weight() -> MatWeight {
    MatWeight::cpu_only(PackedTensor::RawF32 {
        rows: 0,
        cols: 0,
        data: Vec::new(),
    })
}

impl TurboRunnerInner {
    /// Streams CPU weights directly into a single Metal `weight_arena_buf` layer-by-layer,
    /// draining & freeing each CPU layer as it is copied so physical RAM stays flat (~1× model
    /// size instead of ~2.5–3× peak during load).
    pub fn try_new_draining(
        eng: &MetalEngine,
        cfg: &ModelConfig,
        token_embd: &mut MatWeight,
        output_norm: &[f32],
        output_head: &mut Option<MatWeight>,
        layers: &mut Vec<DecoderLayer>,
    ) -> Result<Option<Self>> {
        let head_ref = output_head.as_ref().map(|h| &h.cpu);
        if !check_layers_turbo_compatible(cfg, &token_embd.cpu, head_ref, layers) {
            return Ok(None);
        }

        // Pass 1: Compute exact total arena bytes without allocating any temporary matrices.
        let mut total_bytes: usize = 256;
        let PackedTensor::TurboWht(emb_m) = &token_embd.cpu else {
            return Ok(None);
        };
        total_bytes += mat_arena_bytes(emb_m);
        if let Some(PackedTensor::TurboWht(hm)) = head_ref {
            total_bytes += mat_arena_bytes(hm);
        }
        total_bytes += align256(std::mem::size_of_val(output_norm));
        let ab_padded_cols = cfg.n_embd.div_ceil(256) * 256;
        let ab_bytes = align256(2 * cfg.ssm_dt_rank * ab_padded_cols * 2);

        for ly in layers.iter() {
            total_bytes += align256(std::mem::size_of_val(ly.attn_norm.as_slice()));
            total_bytes += align256(std::mem::size_of_val(ly.post_attn_norm.as_slice()));
            let (
                PackedTensor::TurboWht(fg),
                PackedTensor::TurboWht(fu),
                PackedTensor::TurboWht(fd),
            ) = (&ly.ffn_gate.cpu, &ly.ffn_up.cpu, &ly.ffn_down.cpu)
            else {
                return Ok(None);
            };
            total_bytes += align256((fg.scales_f16.len() + fu.scales_f16.len()) * 2);
            total_bytes += align256((fg.packed_words.len() + fu.packed_words.len()) * 4);
            total_bytes += mat_arena_bytes(fd);

            match &ly.attn {
                LayerAttn::LinearDeltaNet {
                    wqkv,
                    wqkv_gate,
                    ssm_out,
                    ssm_conv1d,
                    ssm_a,
                    ssm_dt_bias,
                    ssm_norm,
                    ..
                } => {
                    let (
                        PackedTensor::TurboWht(mq),
                        PackedTensor::TurboWht(mg),
                        PackedTensor::TurboWht(mo),
                    ) = (&wqkv.cpu, &wqkv_gate.cpu, &ssm_out.cpu)
                    else {
                        return Ok(None);
                    };
                    if concat_compatible(mq, mg) {
                        total_bytes += align256((mq.scales_f16.len() + mg.scales_f16.len()) * 2);
                        total_bytes += align256((mq.packed_words.len() + mg.packed_words.len()) * 4);
                    } else {
                        total_bytes += mat_arena_bytes(mq) + mat_arena_bytes(mg);
                    }
                    total_bytes += ab_bytes;
                    total_bytes += mat_arena_bytes(mo);
                    total_bytes += align256(std::mem::size_of_val(ssm_conv1d.as_slice()));
                    total_bytes += align256(std::mem::size_of_val(ssm_a.as_slice()));
                    total_bytes += align256(std::mem::size_of_val(ssm_dt_bias.as_slice()));
                    total_bytes += align256(std::mem::size_of_val(ssm_norm.as_slice()));
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
                    ..
                } => {
                    let (
                        PackedTensor::TurboWht(mq),
                        PackedTensor::TurboWht(mk),
                        PackedTensor::TurboWht(mv),
                        PackedTensor::TurboWht(mo),
                    ) = (&wq.cpu, &wk.cpu, &wv.cpu, &wo.cpu)
                    else {
                        return Ok(None);
                    };
                    if can_concat3(mq, mk, mv) {
                        total_bytes += align256(
                            (mq.scales_f16.len() + mk.scales_f16.len() + mv.scales_f16.len()) * 2,
                        );
                        total_bytes += align256(
                            (mq.packed_words.len() + mk.packed_words.len() + mv.packed_words.len())
                                * 4,
                        );
                    } else {
                        total_bytes += mat_arena_bytes(mq);
                        total_bytes += align256((mk.scales_f16.len() + mv.scales_f16.len()) * 2);
                        total_bytes += align256((mk.packed_words.len() + mv.packed_words.len()) * 4);
                    }
                    total_bytes += mat_arena_bytes(mo);
                    for opt in [q_norm, k_norm, q_bias, k_bias, v_bias] {
                        if let Some(v) = opt.as_ref() {
                            total_bytes += align256(std::mem::size_of_val(v.as_slice()));
                        }
                    }
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

        let mut arena_ptr: *mut c_void = null_mut();
        let mut err_buf = [0u8; 512];
        let raw_ptr = unsafe {
            vxt_metal_turbo_arena_begin(
                eng.raw_ctx().as_ptr(),
                &c_cfg,
                total_bytes,
                &mut arena_ptr,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        let Some(raw) = NonNull::new(raw_ptr) else {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        };
        // Wrap immediately in RAII so any error cleanly destroys `raw`.
        let runner = Self { raw };
        let mut w = ArenaWriter {
            base: arena_ptr as *mut u8,
            cursor: 256,
            cap: total_bytes,
        };

        // Pass 2: Drain & copy weights directly into `weight_arena_buf` (1 layer at a time!).
        let embed_am = {
            let emb_owned = std::mem::replace(token_embd, empty_mat_weight());
            let PackedTensor::TurboWht(emb_m) = emb_owned.cpu else {
                unreachable!()
            };
            w.write_mat(&emb_m)
        };
        let head_am = match output_head.take() {
            Some(head_owned) => {
                let PackedTensor::TurboWht(hm) = head_owned.cpu else {
                    unreachable!()
                };
                w.write_mat(&hm)
            }
            None => embed_am,
        };
        let output_norm_off = w.write_slice(output_norm);

        let mut arena_layers = Vec::with_capacity(layers.len());
        for ly in layers.drain(..) {
            let attn_norm_off = w.write_slice(&ly.attn_norm);
            let post_attn_norm_off = w.write_slice(&ly.post_attn_norm);
            let (
                PackedTensor::TurboWht(fg),
                PackedTensor::TurboWht(fu),
                PackedTensor::TurboWht(fd),
            ) = (ly.ffn_gate.cpu, ly.ffn_up.cpu, ly.ffn_down.cpu)
            else {
                unreachable!()
            };
            let gu_cat = interleave_ffn_gu(&fg, &fu)
                .ok_or_else(|| Error::Backend("ffn_gu interleave failed".into()))?;
            drop(fg);
            drop(fu);
            let w_ffn_gu = w.write_mat(&gu_cat);
            drop(gu_cat);
            let w_ffn_down = w.write_mat(&fd);
            drop(fd);

            match ly.attn {
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
                    let (
                        PackedTensor::TurboWht(mq),
                        PackedTensor::TurboWht(mg),
                        PackedTensor::TurboWht(mo),
                    ) = (wqkv.cpu, wqkv_gate.cpu, ssm_out.cpu)
                    else {
                        unreachable!()
                    };
                    let (has_split_gate, w_qkv, w_gate_sep) =
                        if let Some(cat) = TurboWhtMatrix::concat_vertical(&[&mq, &mg]) {
                            drop(mq);
                            drop(mg);
                            let am = w.write_mat(&cat);
                            drop(cat);
                            (0u32, am, VxtArenaMatC::zero())
                        } else {
                            let aq = w.write_mat(&mq);
                            drop(mq);
                            let ag = w.write_mat(&mg);
                            drop(mg);
                            (1u32, aq, ag)
                        };
                    let ab = build_ab_rot_f16(&ssm_alpha.cpu, &ssm_beta.cpu, cfg.ssm_dt_rank, cfg.n_embd);
                    drop(ssm_alpha);
                    drop(ssm_beta);
                    let ab_rot_f16_off = w.write_slice(&ab);
                    drop(ab);
                    let w_out = w.write_mat(&mo);
                    drop(mo);
                    arena_layers.push(VxtArenaLayerDescC {
                        is_recurrent: 1,
                        gated_q: 0,
                        gqa_flags: 0,
                        has_split_gate,
                        attn_norm_off,
                        post_attn_norm_off,
                        w_qkv,
                        w_gate_sep,
                        ab_rot_f16_off,
                        w_out,
                        vec0_off: w.write_slice(&ssm_conv1d),
                        vec1_off: w.write_slice(&ssm_a),
                        vec2_off: w.write_slice(&ssm_dt_bias),
                        vec3_off: w.write_slice(&ssm_norm),
                        vec4_off: 0,
                        w_ffn_gu,
                        w_ffn_down,
                    });
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
                    let (
                        PackedTensor::TurboWht(mq),
                        PackedTensor::TurboWht(mk),
                        PackedTensor::TurboWht(mv),
                        PackedTensor::TurboWht(mo),
                    ) = (wq.cpu, wk.cpu, wv.cpu, wo.cpu)
                    else {
                        unreachable!()
                    };
                    let (has_split_gate, w_qkv, w_gate_sep) =
                        if let Some(qkv_cat) = TurboWhtMatrix::concat_vertical(&[&mq, &mk, &mv]) {
                            drop(mq);
                            drop(mk);
                            drop(mv);
                            let am = w.write_mat(&qkv_cat);
                            drop(qkv_cat);
                            (0u32, am, VxtArenaMatC::zero())
                        } else {
                            let aq = w.write_mat(&mq);
                            drop(mq);
                            let kv_cat = TurboWhtMatrix::concat_vertical(&[&mk, &mv])
                                .ok_or_else(|| Error::Backend("gqa kv concat failed".into()))?;
                            drop(mk);
                            drop(mv);
                            let akv = w.write_mat(&kv_cat);
                            drop(kv_cat);
                            (1u32, aq, akv)
                        };
                    let w_out = w.write_mat(&mo);
                    drop(mo);
                    let flags = (gated_q as u32)
                        | ((cfg.rope_neox as u32) << 1)
                        | ((q_norm.is_some() as u32) << 2)
                        | ((k_norm.is_some() as u32) << 3)
                        | ((q_bias.is_some() as u32) << 4)
                        | ((k_bias.is_some() as u32) << 5)
                        | ((v_bias.is_some() as u32) << 6);
                    arena_layers.push(VxtArenaLayerDescC {
                        is_recurrent: 0,
                        gated_q: gated_q as u32,
                        gqa_flags: flags,
                        has_split_gate,
                        attn_norm_off,
                        post_attn_norm_off,
                        w_qkv,
                        w_gate_sep,
                        ab_rot_f16_off: 0,
                        w_out,
                        vec0_off: w.write_opt(q_norm.as_deref()),
                        vec1_off: w.write_opt(k_norm.as_deref()),
                        vec2_off: w.write_opt(q_bias.as_deref()),
                        vec3_off: w.write_opt(k_bias.as_deref()),
                        vec4_off: w.write_opt(v_bias.as_deref()),
                        w_ffn_gu,
                        w_ffn_down,
                    });
                }
            }
        }
        layers.shrink_to_fit();

        let rc = unsafe {
            vxt_metal_turbo_arena_finalize(
                runner.raw.as_ptr(),
                &embed_am,
                &head_am,
                output_norm_off,
                arena_layers.as_ptr(),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        if rc != 0 {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(Some(runner))
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

    pub fn rollback_deltanet(&self, state: &mut TurboStateInner, accepted_m: u32, batch_m: u32) -> Result<()> {
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_turbo_rollback_deltanet(
                self.raw.as_ptr(), state.0.as_ptr(), accepted_m, batch_m,
                err_buf.as_mut_ptr() as *mut c_char, err_buf.len(),
            )
        };
        if rc != 0 { return Err(Error::Backend(cbuf_to_str(&err_buf))); }
        Ok(())
    }

    pub fn replay_trunk(&self, state: &mut TurboStateInner, tok_ids: &[u32], pos_base: u32) -> Result<()> {
        if tok_ids.is_empty() { return Ok(()); }
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_turbo_replay_trunk(
                self.raw.as_ptr(), state.0.as_ptr(), tok_ids.as_ptr(),
                tok_ids.len() as u32, pos_base, err_buf.as_mut_ptr() as *mut c_char, err_buf.len(),
            )
        };
        if rc != 0 { return Err(Error::Backend(cbuf_to_str(&err_buf))); }
        Ok(())
    }

    #[inline]
    pub fn mark_seen(&self, state: &mut TurboStateInner, tok_id: u32) {
        unsafe { vxt_metal_turbo_mark_seen(state.0.as_ptr(), tok_id) };
    }

    #[inline]
    pub fn init_seen(&self, state: &mut TurboStateInner, seen_ids: &[u32], presence_pen: f32) {
        unsafe { vxt_metal_turbo_init_seen(state.0.as_ptr(), seen_ids.as_ptr(), seen_ids.len(), presence_pen) };
    }
}
