//! Low-level Objective-C++ FFI bindings for `MetalTurboRunner` (`metal_llm_turbo.mm`).

#![cfg(vxt_has_metal)]

use core::ffi::{c_char, c_void};
use core::ptr::{null_mut, NonNull};

use crate::format::PackedTensor;
use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::gguf_meta::ModelConfig;
use crate::llm::{DecoderLayer, LayerAttn, MatWeight};
use crate::metal::MetalEngine;
use crate::metal_ffi::cbuf_to_str;
use crate::turbo_e8_hq::decode_e8_hq_words;
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

fn arena_wps(bits: u8) -> usize {
    match bits {
        9 => 224,
        10 => 192,
        other => TurboWhtMatrix::words_per_tile_step(other),
    }
}

fn mat_arena_bytes_bits(m: &TurboWhtMatrix, target_bits: u8) -> usize {
    let pw_len = (m.padded_rows / 4) * (m.padded_cols / 256) * arena_wps(target_bits);
    align256(m.scales_f16.len() * 2) + align256(pw_len * 4)
}

fn cat_arena_bytes_bits(mats: &[&TurboWhtMatrix], target_bits: u8) -> usize {
    let sc_len: usize = mats.iter().map(|m| m.scales_f16.len()).sum();
    let pw_len: usize = mats
        .iter()
        .map(|m| (m.padded_rows / 4) * (m.padded_cols / 256) * arena_wps(target_bits))
        .sum();
    align256(sc_len * 2) + align256(pw_len * 4)
}

#[inline(always)]
fn e8_hq_unpack_u8(w0: u32, w1: u32) -> ([u8; 8], u8) {
    let b0 = w0.to_le_bytes();
    let b1 = w1.to_le_bytes();
    let par = (b0[0] ^ b0[1] ^ b0[2] ^ b0[3] ^ b1[0] ^ b1[1] ^ b1[2]) & 1;
    let u7 = ((b1[3] & 0x7f) << 1) | par;
    let c = (b1[3] >> 7) & 1;
    ([b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2], u7], c)
}

/// Inspects an `E8-HQ` (`bits == 8`) matrix and selects the narrowest GPU arena bit-width
/// (`10` = 6-bit `E8-HQ6` `-25%` RAM, `9` = 7-bit `E8-HQ7` `-12.5%` RAM, or `8` = full 8-bit)
/// that fits `>= 99.5%` of `32`-column blocks bit-exactly without rescaling.
fn choose_e8_arena_bits(m: &TurboWhtMatrix) -> u8 {
    if m.bits != 8 || std::env::var_os("VXT_NO_HQ_COMPACT").is_some() {
        return m.bits;
    }
    let total_ts = (m.padded_rows / 4) * (m.padded_cols / 256);
    let total_blocks = total_ts * 32;
    if total_blocks == 0 {
        return 8;
    }
    let max_over = (total_blocks * 5) / 1000;
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4)
        .min(total_ts.max(1));
    let chunk_ts = total_ts.div_ceil(n_threads);
    let (over6, over7) = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let ts0 = tid * chunk_ts;
            let ts1 = (ts0 + chunk_ts).min(total_ts);
            if ts0 >= ts1 {
                continue;
            }
            let pw = &m.packed_words[ts0 * 256..ts1 * 256];
            handles.push(s.spawn(move || {
                let (mut o6, mut o7) = (0usize, 0usize);
                for ts_pw in pw.chunks_exact(256) {
                    for b8 in 0..8usize {
                        for sr in 0..4usize {
                            let (mut lo, mut hi) = (255u8, 0u8);
                            for s4 in 0..4usize {
                                let idx = ((b8 * 4 + s4) * 4 + sr) * 2;
                                let (u, _) = e8_hq_unpack_u8(ts_pw[idx], ts_pw[idx + 1]);
                                for &v in &u {
                                    lo = lo.min(v);
                                    hi = hi.max(v);
                                }
                            }
                            if lo < 96 || hi > 159 {
                                o6 += 1;
                                if lo < 64 || hi > 191 {
                                    o7 += 1;
                                }
                            }
                        }
                    }
                }
                (o6, o7)
            }));
        }
        let (mut t6, mut t7) = (0usize, 0usize);
        for h in handles {
            let (a, b) = h.join().unwrap_or((total_blocks, total_blocks));
            t6 += a;
            t7 += b;
        }
        (t6, t7)
    });
    if over6 <= max_over {
        10
    } else if over7 <= max_over {
        9
    } else {
        8
    }
}

#[inline]
fn merge_arena_bits(a: u8, b: u8) -> u8 {
    // Among E8-HQ variants (8 = 8-bit widest, 9 = 7-bit, 10 = 6-bit), `min` picks the widest/safest.
    if a >= 8 && b >= 8 {
        a.min(b)
    } else {
        a
    }
}

#[inline]
fn quantize_e8_narrow(y: &[f32; 8], max_u: u8, zero_off: f32) -> ([u8; 8], u8) {
    let mut best_err = f32::INFINITY;
    let mut best_u = [0u8; 8];
    let mut best_c = 0u8;
    let max_i = max_u as i32;
    for c in 0u8..=1u8 {
        let bias = if c != 0 { -zero_off + 0.5 } else { -zero_off };
        let mut u = [0u8; 8];
        let mut par = 0u8;
        let mut err = 0.0f32;
        let mut best_flip_idx = 0usize;
        let mut best_flip_u = 0u8;
        let mut best_flip_pen = f32::INFINITY;
        for i in 0..8 {
            let yf = y[i] - bias;
            let ub = (yf.round() as i32).clamp(0, max_i) as u8;
            u[i] = ub;
            par ^= ub & 1;
            let d0 = yf - (ub as f32);
            let e0 = d0 * d0;
            err += e0;
            let alt = if ub == 0 {
                1
            } else if ub == max_u {
                max_u - 1
            } else if yf >= (ub as f32) {
                ub + 1
            } else {
                ub - 1
            };
            let d1 = yf - (alt as f32);
            let pen = d1 * d1 - e0;
            if pen < best_flip_pen {
                best_flip_pen = pen;
                best_flip_idx = i;
                best_flip_u = alt;
            }
        }
        if par != 0 {
            u[best_flip_idx] = best_flip_u;
            err += best_flip_pen;
        }
        if err < best_err {
            best_err = err;
            best_u = u;
            best_c = c;
        }
    }
    (best_u, best_c)
}

#[inline(always)]
fn pack_hq7_row28(v: &[u8; 8], c: u8) -> (u32, u32) {
    let a = (v[0] as u32)
        | ((v[1] as u32) << 7)
        | ((v[2] as u32) << 14)
        | ((v[3] as u32) << 21);
    let tail7 = (((v[7] >> 1) & 0x3f) as u32) | (((c & 1) as u32) << 6);
    let b = (v[4] as u32)
        | ((v[5] as u32) << 7)
        | ((v[6] as u32) << 14)
        | (tail7 << 21);
    (a, b)
}

#[inline(always)]
fn pack_hq6_row24(v: &[u8; 8], c: u8) -> (u32, u32) {
    let a = (v[0] as u32)
        | ((v[1] as u32) << 6)
        | ((v[2] as u32) << 12)
        | ((v[3] as u32) << 18);
    let tail6 = (((v[7] >> 1) & 0x1f) as u32) | (((c & 1) as u32) << 5);
    let b = (v[4] as u32)
        | ((v[5] as u32) << 6)
        | ((v[6] as u32) << 12)
        | (tail6 << 18);
    (a, b)
}

fn repack_e8_hq_narrow_slice(
    scales_in: &[u16],
    pw_in: &[u32],
    scales_out: &mut [u16],
    pw_out: &mut [u32],
    target_bits: u8,
) {
    let out_wps = arena_wps(target_bits);
    let total_ts = pw_in.len() / 256;
    let (lo_b, hi_b, max_u, zero_off) = if target_bits == 9 {
        (64u8, 191u8, 127u8, 64.0f32)
    } else {
        (96u8, 159u8, 63u8, 32.0f32)
    };
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 8))
        .unwrap_or(4)
        .min(total_ts.max(1));
    let chunk_ts = total_ts.div_ceil(n_threads);
    std::thread::scope(|s| {
        let mut sc_rest = scales_out;
        let mut pw_rest = pw_out;
        for tid in 0..n_threads {
            let ts0 = tid * chunk_ts;
            let ts1 = (ts0 + chunk_ts).min(total_ts);
            if ts0 >= ts1 {
                break;
            }
            let cnt = ts1 - ts0;
            let (sc_dst, sc_tail) = sc_rest.split_at_mut(cnt * 32);
            let (pw_dst, pw_tail) = pw_rest.split_at_mut(cnt * out_wps);
            sc_rest = sc_tail;
            pw_rest = pw_tail;
            let sc_src = &scales_in[ts0 * 32..ts1 * 32];
            let pw_src = &pw_in[ts0 * 256..ts1 * 256];
            s.spawn(move || {
                let mut ab_lanes = [[(0u32, 0u32); 4]; 32];
                for t in 0..cnt {
                    let s_in = &sc_src[t * 32..(t + 1) * 32];
                    let s_out = &mut sc_dst[t * 32..(t + 1) * 32];
                    let p_in = &pw_src[t * 256..(t + 1) * 256];
                    let p_out = &mut pw_dst[t * out_wps..(t + 1) * out_wps];

                    for b8 in 0..8usize {
                        for sr in 0..4usize {
                            let (mut lo, mut hi) = (255u8, 0u8);
                            let mut us = [([0u8; 8], 0u8); 4];
                            for s4 in 0..4usize {
                                let idx = ((b8 * 4 + s4) * 4 + sr) * 2;
                                let uc = e8_hq_unpack_u8(p_in[idx], p_in[idx + 1]);
                                for &v in &uc.0 {
                                    lo = lo.min(v);
                                    hi = hi.max(v);
                                }
                                us[s4] = uc;
                            }
                            if lo >= lo_b && hi <= hi_b {
                                s_out[b8 * 4 + sr] = s_in[b8 * 4 + sr];
                                for s4 in 0..4usize {
                                    let (u, c) = us[s4];
                                    let v = [
                                        u[0] - lo_b, u[1] - lo_b, u[2] - lo_b, u[3] - lo_b,
                                        u[4] - lo_b, u[5] - lo_b, u[6] - lo_b, u[7] - lo_b,
                                    ];
                                    ab_lanes[b8 * 4 + s4][sr] = if target_bits == 9 {
                                        pack_hq7_row28(&v, c)
                                    } else {
                                        pack_hq6_row24(&v, c)
                                    };
                                }
                            } else {
                                let sc = f16_to_f32(s_in[b8 * 4 + sr]);
                                let mult = if target_bits == 9 || (lo >= 64 && hi <= 191) {
                                    2.0f32
                                } else {
                                    4.0f32
                                };
                                let new_sf = f32_to_f16(sc * mult);
                                let new_sc = f16_to_f32(new_sf);
                                let ratio = if new_sc > 0.0 { sc / new_sc } else { 0.0 };
                                s_out[b8 * 4 + sr] = new_sf;
                                for s4 in 0..4usize {
                                    let idx = ((b8 * 4 + s4) * 4 + sr) * 2;
                                    let q = decode_e8_hq_words(p_in[idx], p_in[idx + 1]);
                                    let mut y = [0.0f32; 8];
                                    for j in 0..8 {
                                        y[j] = q[j] * ratio;
                                    }
                                    let (v, c) = quantize_e8_narrow(&y, max_u, zero_off);
                                    ab_lanes[b8 * 4 + s4][sr] = if target_bits == 9 {
                                        pack_hq7_row28(&v, c)
                                    } else {
                                        pack_hq6_row24(&v, c)
                                    };
                                }
                            }
                        }
                    }

                    if target_bits == 9 {
                        for lane in 0..32usize {
                            let [(a0, b0), (a1, b1), (a2, b2), (a3, b3)] = ab_lanes[lane];
                            p_out[lane * 4] = a0 | ((b0 & 0x0f) << 28);
                            p_out[lane * 4 + 1] = (b0 >> 4) | ((a1 & 0xff) << 24);
                            p_out[lane * 4 + 2] = (a1 >> 8) | ((b1 & 0x0fff) << 20);
                            p_out[lane * 4 + 3] = (b1 >> 12) | ((a2 & 0xffff) << 16);
                            p_out[128 + lane * 2] = (a2 >> 16) | ((b2 & 0x000f_ffff) << 12);
                            p_out[128 + lane * 2 + 1] = (b2 >> 20) | ((a3 & 0x00ff_ffff) << 8);
                            p_out[192 + lane] = (a3 >> 24) | (b3 << 4);
                        }
                    } else {
                        for lane in 0..32usize {
                            let [(a0, b0), (a1, b1), (a2, b2), (a3, b3)] = ab_lanes[lane];
                            p_out[lane * 4] = a0 | ((b0 & 0xff) << 24);
                            p_out[lane * 4 + 1] = (b0 >> 8) | ((a1 & 0xffff) << 16);
                            p_out[lane * 4 + 2] = (a1 >> 16) | (b1 << 8);
                            p_out[lane * 4 + 3] = a2 | ((b2 & 0xff) << 24);
                            p_out[128 + lane * 2] = (b2 >> 8) | ((a3 & 0xffff) << 16);
                            p_out[128 + lane * 2 + 1] = (a3 >> 16) | (b3 << 8);
                        }
                    }
                }
            });
        }
    });
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

    fn alloc_slice_mut<T>(&mut self, count: usize) -> (usize, &mut [T]) {
        let len = count * std::mem::size_of::<T>();
        if len == 0 {
            return (0, &mut []);
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
        self.cursor += step;
        let ptr = unsafe { self.base.add(off) as *mut T };
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, count) };
        (off, slice)
    }

    fn write_mat_bits(&mut self, m: &TurboWhtMatrix, target_bits: u8) -> VxtArenaMatC {
        if m.bits == 8 && (target_bits == 9 || target_bits == 10) {
            let out_pw_len = (m.padded_rows / 4) * (m.padded_cols / 256) * arena_wps(target_bits);
            let (scales_off, sc_out) = self.alloc_slice_mut::<u16>(m.scales_f16.len());
            let sc_ptr = sc_out.as_mut_ptr();
            let sc_len = sc_out.len();
            let (packed_off, pw_out) = self.alloc_slice_mut::<u32>(out_pw_len);
            let sc_out_slice = unsafe { std::slice::from_raw_parts_mut(sc_ptr, sc_len) };
            repack_e8_hq_narrow_slice(
                &m.scales_f16,
                &m.packed_words,
                sc_out_slice,
                pw_out,
                target_bits,
            );
            return VxtArenaMatC {
                scales_off,
                packed_off,
                rows: m.rows as u32,
                cols: m.cols as u32,
                padded_cols: m.padded_cols as u32,
                bits: target_bits as u32,
            };
        }
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

struct LayerArenaPlan {
    ffn_gu_bits: u8,
    ffn_down_bits: u8,
    qkv_bits: u8,
    gate_sep_bits: u8,
    out_bits: u8,
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
    /// size instead of ~2.5–3× peak during load). Automatically compacts eligible `E8-HQ`
    /// (`8-bit`) matrices into `E8-HQ7` (`7-bit`, `-12.5%` GPU RAM) or `E8-HQ6` (`6-bit`, `-25%`).
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

        // Pass 1: Choose optimal per-matrix GPU bit-width (`E8-HQ8/7/6`) and compute exact arena bytes.
        let mut total_bytes: usize = 256;
        let PackedTensor::TurboWht(emb_m) = &token_embd.cpu else {
            return Ok(None);
        };
        let emb_bits = choose_e8_arena_bits(emb_m);
        total_bytes += mat_arena_bytes_bits(emb_m, emb_bits);
        let head_bits = if let Some(PackedTensor::TurboWht(hm)) = head_ref {
            let hb = choose_e8_arena_bits(hm);
            total_bytes += mat_arena_bytes_bits(hm, hb);
            hb
        } else {
            emb_bits
        };
        total_bytes += align256(std::mem::size_of_val(output_norm));
        let ab_padded_cols = cfg.n_embd.div_ceil(256) * 256;
        let ab_bytes = align256(2 * cfg.ssm_dt_rank * ab_padded_cols * 2);

        let mut plans = Vec::with_capacity(layers.len());
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
            let ffn_gu_bits = merge_arena_bits(choose_e8_arena_bits(fg), choose_e8_arena_bits(fu));
            let ffn_down_bits = choose_e8_arena_bits(fd);
            total_bytes += cat_arena_bytes_bits(&[fg, fu], ffn_gu_bits);
            total_bytes += mat_arena_bytes_bits(fd, ffn_down_bits);

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
                    let bq = choose_e8_arena_bits(mq);
                    let bg = choose_e8_arena_bits(mg);
                    let out_bits = choose_e8_arena_bits(mo);
                    let (qkv_bits, gate_sep_bits) = if concat_compatible(mq, mg) {
                        let mb = merge_arena_bits(bq, bg);
                        total_bytes += cat_arena_bytes_bits(&[mq, mg], mb);
                        (mb, mb)
                    } else {
                        total_bytes += mat_arena_bytes_bits(mq, bq) + mat_arena_bytes_bits(mg, bg);
                        (bq, bg)
                    };
                    total_bytes += ab_bytes;
                    total_bytes += mat_arena_bytes_bits(mo, out_bits);
                    total_bytes += align256(std::mem::size_of_val(ssm_conv1d.as_slice()));
                    total_bytes += align256(std::mem::size_of_val(ssm_a.as_slice()));
                    total_bytes += align256(std::mem::size_of_val(ssm_dt_bias.as_slice()));
                    total_bytes += align256(std::mem::size_of_val(ssm_norm.as_slice()));
                    plans.push(LayerArenaPlan {
                        ffn_gu_bits,
                        ffn_down_bits,
                        qkv_bits,
                        gate_sep_bits,
                        out_bits,
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
                    let bq = choose_e8_arena_bits(mq);
                    let bk = choose_e8_arena_bits(mk);
                    let bv = choose_e8_arena_bits(mv);
                    let out_bits = choose_e8_arena_bits(mo);
                    let (qkv_bits, gate_sep_bits) = if can_concat3(mq, mk, mv) {
                        let mb = merge_arena_bits(bq, merge_arena_bits(bk, bv));
                        total_bytes += cat_arena_bytes_bits(&[mq, mk, mv], mb);
                        (mb, mb)
                    } else {
                        let bkv = merge_arena_bits(bk, bv);
                        total_bytes += mat_arena_bytes_bits(mq, bq);
                        total_bytes += cat_arena_bytes_bits(&[mk, mv], bkv);
                        (bq, bkv)
                    };
                    total_bytes += mat_arena_bytes_bits(mo, out_bits);
                    for opt in [q_norm, k_norm, q_bias, k_bias, v_bias] {
                        if let Some(v) = opt.as_ref() {
                            total_bytes += align256(std::mem::size_of_val(v.as_slice()));
                        }
                    }
                    plans.push(LayerArenaPlan {
                        ffn_gu_bits,
                        ffn_down_bits,
                        qkv_bits,
                        gate_sep_bits,
                        out_bits,
                    });
                }
            }
        }

        if std::env::var_os("VXT_LOG_ARENA").is_some() {
            let (mut n8, mut n9, mut n10) = (0usize, 0usize, 0usize);
            for b in [emb_bits, head_bits]
                .into_iter()
                .chain(plans.iter().flat_map(|p| [p.ffn_gu_bits, p.ffn_down_bits, p.qkv_bits, p.out_bits]))
            {
                match b {
                    8 => n8 += 1,
                    9 => n9 += 1,
                    10 => n10 += 1,
                    _ => {}
                }
            }
            eprintln!(
                "[VXT-ARENA] weight_arena={:.2} MiB (HQ8={n8}, HQ7={n9}, HQ6={n10}, emb=b{emb_bits}, head=b{head_bits})",
                (total_bytes as f64) / (1024.0 * 1024.0)
            );
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

        // Pass 2: Drain & compact weights directly into `weight_arena_buf` (1 layer at a time!).
        let embed_am = {
            let emb_owned = std::mem::replace(token_embd, empty_mat_weight());
            let PackedTensor::TurboWht(emb_m) = emb_owned.cpu else {
                unreachable!()
            };
            w.write_mat_bits(&emb_m, emb_bits)
        };
        let head_am = match output_head.take() {
            Some(head_owned) => {
                let PackedTensor::TurboWht(hm) = head_owned.cpu else {
                    unreachable!()
                };
                w.write_mat_bits(&hm, head_bits)
            }
            None => embed_am,
        };
        let output_norm_off = w.write_slice(output_norm);

        let mut arena_layers = Vec::with_capacity(layers.len());
        for (ly, plan) in layers.drain(..).zip(plans.into_iter()) {
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
            let w_ffn_gu = w.write_mat_bits(&gu_cat, plan.ffn_gu_bits);
            drop(gu_cat);
            let w_ffn_down = w.write_mat_bits(&fd, plan.ffn_down_bits);
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
                            let am = w.write_mat_bits(&cat, plan.qkv_bits);
                            drop(cat);
                            (0u32, am, VxtArenaMatC::zero())
                        } else {
                            let aq = w.write_mat_bits(&mq, plan.qkv_bits);
                            drop(mq);
                            let ag = w.write_mat_bits(&mg, plan.gate_sep_bits);
                            drop(mg);
                            (1u32, aq, ag)
                        };
                    let ab = build_ab_rot_f16(&ssm_alpha.cpu, &ssm_beta.cpu, cfg.ssm_dt_rank, cfg.n_embd);
                    drop(ssm_alpha);
                    drop(ssm_beta);
                    let ab_rot_f16_off = w.write_slice(&ab);
                    drop(ab);
                    let w_out = w.write_mat_bits(&mo, plan.out_bits);
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
                            let am = w.write_mat_bits(&qkv_cat, plan.qkv_bits);
                            drop(qkv_cat);
                            (0u32, am, VxtArenaMatC::zero())
                        } else {
                            let aq = w.write_mat_bits(&mq, plan.qkv_bits);
                            drop(mq);
                            let kv_cat = TurboWhtMatrix::concat_vertical(&[&mk, &mv])
                                .ok_or_else(|| Error::Backend("gqa kv concat failed".into()))?;
                            drop(mk);
                            drop(mv);
                            let akv = w.write_mat_bits(&kv_cat, plan.gate_sep_bits);
                            drop(kv_cat);
                            (1u32, aq, akv)
                        };
                    let w_out = w.write_mat_bits(&mo, plan.out_bits);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turbo_e8_hq::pack_e8_hq_words;

    fn unpack_hq7_row_ref(p224: &[u32], lane: usize, sr: usize) -> (u32, u32) {
        let w0 = p224[lane * 4];
        let w1 = p224[lane * 4 + 1];
        let w2 = p224[lane * 4 + 2];
        let w3 = p224[lane * 4 + 3];
        let w4 = p224[128 + lane * 2];
        let w5 = p224[128 + lane * 2 + 1];
        let w6 = p224[192 + lane];
        match sr {
            0 => (w0 & 0x0fff_ffff, (w0 >> 28) | ((w1 & 0x00ff_ffff) << 4)),
            1 => ((w1 >> 24) | ((w2 & 0x000f_ffff) << 8), (w2 >> 20) | ((w3 & 0x0000_ffff) << 12)),
            2 => ((w3 >> 16) | ((w4 & 0x0000_0fff) << 16), (w4 >> 12) | ((w5 & 0x0000_00ff) << 20)),
            _ => ((w5 >> 8) | ((w6 & 0x0000_000f) << 24), w6 >> 4),
        }
    }

    fn decode_hq7_ref(a: u32, b: u32) -> [f32; 8] {
        let p = a ^ (b & 0x001f_ffff);
        let p = p ^ (p >> 14);
        let par = (p ^ (p >> 7)) & 1;
        let u7 = ((b >> 20) & 0x7e) | par;
        let bias = if (b & 0x0800_0000) != 0 { -63.5f32 } else { -64.0f32 };
        [
            ((a & 127) as f32) + bias,
            (((a >> 7) & 127) as f32) + bias,
            (((a >> 14) & 127) as f32) + bias,
            (((a >> 21) & 127) as f32) + bias,
            ((b & 127) as f32) + bias,
            (((b >> 7) & 127) as f32) + bias,
            (((b >> 14) & 127) as f32) + bias,
            (u7 as f32) + bias,
        ]
    }

    fn unpack_hq6_row_ref(p192: &[u32], lane: usize, sr: usize) -> (u32, u32) {
        let w0 = p192[lane * 4];
        let w1 = p192[lane * 4 + 1];
        let w2 = p192[lane * 4 + 2];
        let w3 = p192[lane * 4 + 3];
        let w4 = p192[128 + lane * 2];
        let w5 = p192[128 + lane * 2 + 1];
        match sr {
            0 => (w0 & 0x00ff_ffff, (w0 >> 24) | ((w1 & 0x0000_ffff) << 8)),
            1 => ((w1 >> 16) | ((w2 & 0x0000_00ff) << 16), w2 >> 8),
            2 => (w3 & 0x00ff_ffff, (w3 >> 24) | ((w4 & 0x0000_ffff) << 8)),
            _ => ((w4 >> 16) | ((w5 & 0x0000_00ff) << 16), w5 >> 8),
        }
    }

    fn decode_hq6_ref(a: u32, b: u32) -> [f32; 8] {
        let p = a ^ (b & 0x0003_ffff);
        let p = p ^ (p >> 12);
        let par = (p ^ (p >> 6)) & 1;
        let u7 = ((b >> 17) & 0x3e) | par;
        let bias = if (b & 0x0080_0000) != 0 { -31.5f32 } else { -32.0f32 };
        [
            ((a & 63) as f32) + bias,
            (((a >> 6) & 63) as f32) + bias,
            (((a >> 12) & 63) as f32) + bias,
            (((a >> 18) & 63) as f32) + bias,
            ((b & 63) as f32) + bias,
            (((b >> 6) & 63) as f32) + bias,
            (((b >> 12) & 63) as f32) + bias,
            (u7 as f32) + bias,
        ]
    }

    #[test]
    fn e8_hq7_and_hq6_repack_lossless_and_outlier_parity() {
        let mut scales = vec![f32_to_f16(0.25); 32];
        let mut pw8 = vec![0u32; 256];
        let mut rng = 0x1234_5678u64;
        let mut next = || -> u32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 32) as u32
        };

        // Test 1: 100% lossless HQ7 (`u in 64..=191`)
        for lane in 0..32 {
            for sr in 0..4 {
                let mut u = [0u8; 8];
                let mut par = 0u8;
                for j in 0..7 {
                    u[j] = 64 + (next() % 128) as u8;
                    par ^= u[j] & 1;
                }
                let mut u7 = 64 + (next() % 128) as u8;
                if (u7 & 1) != par {
                    u7 = if u7 < 191 { u7 + 1 } else { u7 - 1 };
                }
                u[7] = u7;
                let c = (next() & 1) as u8;
                let [w0, w1] = pack_e8_hq_words(&u, c);
                let idx = (lane * 4 + sr) * 2;
                pw8[idx] = w0;
                pw8[idx + 1] = w1;
            }
        }
        let mut sc7 = vec![0u16; 32];
        let mut pw7 = vec![0u32; 224];
        repack_e8_hq_narrow_slice(&scales, &pw8, &mut sc7, &mut pw7, 9);
        assert_eq!(sc7, scales);
        for lane in 0..32 {
            for sr in 0..4 {
                let idx = (lane * 4 + sr) * 2;
                let ref_q = decode_e8_hq_words(pw8[idx], pw8[idx + 1]);
                let (a, b) = unpack_hq7_row_ref(&pw7, lane, sr);
                let got_q = decode_hq7_ref(a, b);
                assert_eq!(got_q, ref_q, "HQ7 bit-exact mismatch at lane={lane} sr={sr}");
            }
        }

        // Test 2: 100% lossless HQ6 (`u in 96..=159`)
        for lane in 0..32 {
            for sr in 0..4 {
                let mut u = [0u8; 8];
                let mut par = 0u8;
                for j in 0..7 {
                    u[j] = 96 + (next() % 64) as u8;
                    par ^= u[j] & 1;
                }
                let mut u7 = 96 + (next() % 64) as u8;
                if (u7 & 1) != par {
                    u7 = if u7 < 159 { u7 + 1 } else { u7 - 1 };
                }
                u[7] = u7;
                let c = (next() & 1) as u8;
                let [w0, w1] = pack_e8_hq_words(&u, c);
                let idx = (lane * 4 + sr) * 2;
                pw8[idx] = w0;
                pw8[idx + 1] = w1;
            }
        }
        let mut sc6 = vec![0u16; 32];
        let mut pw6 = vec![0u32; 192];
        repack_e8_hq_narrow_slice(&scales, &pw8, &mut sc6, &mut pw6, 10);
        assert_eq!(sc6, scales);
        for lane in 0..32 {
            for sr in 0..4 {
                let idx = (lane * 4 + sr) * 2;
                let ref_q = decode_e8_hq_words(pw8[idx], pw8[idx + 1]);
                let (a, b) = unpack_hq6_row_ref(&pw6, lane, sr);
                let got_q = decode_hq6_ref(a, b);
                assert_eq!(got_q, ref_q, "HQ6 bit-exact mismatch at lane={lane} sr={sr}");
            }
        }

        // Test 3: Outlier block (`u = 250`, outside 7-bit) rescales cleanly (`sc *= 2`) with small error
        scales[0] = f32_to_f16(0.5);
        let outlier_u = [250, 10, 128, 128, 200, 50, 128, 128];
        let [w0, w1] = pack_e8_hq_words(&outlier_u, 1);
        pw8[0] = w0;
        pw8[1] = w1;
        repack_e8_hq_narrow_slice(&scales, &pw8, &mut sc7, &mut pw7, 9);
        assert!((f16_to_f32(sc7[0]) - 1.0).abs() < 1e-3);
        let ref_q = decode_e8_hq_words(w0, w1);
        let (a, b) = unpack_hq7_row_ref(&pw7, 0, 0);
        let got_q = decode_hq7_ref(a, b);
        for j in 0..8 {
            let err = (f16_to_f32(scales[0]) * ref_q[j] - f16_to_f32(sc7[0]) * got_q[j]).abs();
            assert!(err <= 1.0, "outlier rescaling error too high at j={j}: {err}");
        }
    }
}
