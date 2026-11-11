//! Multi-Token Tiled Compressed `GEMM` (`PackedTensor::gemm_cpu`), `Q8_0` Compressed `KV-Cache`
//! (`KvCacheMode::Q8_0`, `34 B / 32` floats = `8.50 bpw`), and Batched Tiled CPU Prefill.

use crate::format::PackedTensor;
use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::llm::{DecoderLayer, LayerAttn, LlmModel, LlmState};
use crate::llm_ops::{apply_rope, rms_norm, silu, step_deltanet_recurrent, step_full_gqa};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum KvCacheMode {
    #[default]
    Fp32,
    Q8_0,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct KvQ8Block {
    pub d: u16,
    pub qs: [i8; 32],
}

const _: () = assert!(core::mem::size_of::<KvQ8Block>() == 34);

impl KvQ8Block {
    #[inline]
    pub fn quantize(src: &[f32]) -> Self {
        debug_assert_eq!(src.len(), 32);
        let mut amax = 0.0f32;
        for &v in src {
            amax = amax.max(v.abs());
        }
        let dh = f32_to_f16(amax / 127.0);
        let df = f16_to_f32(dh);
        let inv = if df > 0.0 { 1.0 / df } else { 0.0 };
        let mut qs = [0i8; 32];
        for i in 0..32 {
            qs[i] = (src[i] * inv).round_ties_even().clamp(-127.0, 127.0) as i8;
        }
        Self { d: dh, qs }
    }

    #[inline]
    pub fn dot_f32(&self, q: &[f32]) -> f32 {
        let sc = f16_to_f32(self.d);
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        for i in 0..16 {
            s0 = (self.qs[2 * i] as f32).mul_add(q[2 * i], s0);
            s1 = (self.qs[2 * i + 1] as f32).mul_add(q[2 * i + 1], s1);
        }
        (s0 + s1) * sc
    }

    #[inline]
    pub fn axpy_f32(&self, weight: f32, out: &mut [f32]) {
        let ws = weight * f16_to_f32(self.d);
        for i in 0..32 {
            out[i] = (self.qs[i] as f32).mul_add(ws, out[i]);
        }
    }
}

#[inline]
pub(crate) fn gelu_tanh_f32(g: f32) -> f32 {
    let z = 0.797_884_6f32 * (g + 0.044_715 * g * g * g);
    let e2 = (-2.0 * z.abs()).exp();
    let th = ((1.0 - e2) / (1.0 + e2)).copysign(z);
    0.5 * g * (1.0 + th)
}

#[inline]
pub(crate) fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq = v.mul_add(v, sum_sq);
    }
    let scale = 1.0 / (sum_sq / (n as f32) + eps).sqrt();
    for (xv, &wv) in x.iter_mut().zip(weight.iter()) {
        *xv *= scale * wv;
    }
}

#[cfg(vxt_has_metal)]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[inline]
fn dense_tile_gemm(
    x_batch: &[f32],
    batch_m: usize,
    cols: usize,
    w_tile: &[f32],
    br: usize,
    y_dst: &mut [f32],
    ldc: usize,
    r_off: usize,
) {
    #[cfg(vxt_has_metal)]
    {
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                batch_m as i32,
                br as i32,
                cols as i32,
                1.0,
                x_batch.as_ptr(),
                cols as i32,
                w_tile.as_ptr(),
                cols as i32,
                0.0,
                y_dst.as_mut_ptr().add(r_off),
                ldc as i32,
            );
        }
        return;
    }
    #[cfg(not(vxt_has_metal))]
    {
        for m in 0..batch_m {
            let xm = &x_batch[m * cols..(m + 1) * cols];
            let ym = &mut y_dst[m * ldc + r_off..m * ldc + r_off + br];
            for r in 0..br {
                let wr = &w_tile[r * cols..(r + 1) * cols];
                let mut s = 0.0f32;
                for (&wv, &xv) in wr.iter().zip(xm.iter()) {
                    s = wv.mul_add(xv, s);
                }
                ym[r] = s;
            }
        }
    }
}

impl PackedTensor {
    /// Multi-token tiled compressed `GEMM`: computes `y_batch[m, r] = dot(W[r, :], x_batch[m, :])`
    /// for `m in 0..batch_m`, decoding each `32`-row compressed tile only once across all `batch_m`
    /// tokens and dispatching dense tile math to Apple AMX (`cblas_sgemm`) / SIMD FMA.
    pub fn gemm_cpu(&self, x_batch: &[f32], batch_m: usize, y_batch: &mut [f32], threads: usize) {
        let rows = self.rows();
        let cols = self.cols();
        if batch_m == 0 || rows == 0 {
            return;
        }
        assert_eq!(x_batch.len(), batch_m * cols);
        assert_eq!(y_batch.len(), batch_m * rows);
        if batch_m == 1 {
            self.gemv_cpu(x_batch, y_batch, threads);
            return;
        }
        if let Self::RawF32 { data, .. } = self {
            dense_tile_gemm(x_batch, batch_m, cols, data, rows, y_batch, rows, 0);
            return;
        }

        const TILE_R: usize = 32;
        let workers = threads.clamp(1, 16).min(rows.div_ceil(TILE_R).max(1));
        if workers <= 1 {
            let mut w_tile = vec![0.0f32; TILE_R.min(rows) * cols];
            let mut r0 = 0usize;
            while r0 < rows {
                let br = (rows - r0).min(TILE_R);
                for dr in 0..br {
                    self.decode_row(r0 + dr, &mut w_tile[dr * cols..(dr + 1) * cols]);
                }
                dense_tile_gemm(
                    x_batch,
                    batch_m,
                    cols,
                    &w_tile[..br * cols],
                    br,
                    y_batch,
                    rows,
                    r0,
                );
                r0 += br;
            }
            return;
        }

        let tiles_total = rows.div_ceil(TILE_R);
        let tiles_per_worker = tiles_total.div_ceil(workers);
        let rows_per_worker = tiles_per_worker * TILE_R;
        let mut partial: Vec<Vec<f32>> = (0..workers)
            .filter_map(|wi| {
                let r_beg = wi * rows_per_worker;
                if r_beg >= rows {
                    return None;
                }
                let w_rows = (rows - r_beg).min(rows_per_worker);
                Some(vec![0.0f32; batch_m * w_rows])
            })
            .collect();

        std::thread::scope(|s| {
            for (wi, part_y) in partial.iter_mut().enumerate() {
                let r_beg = wi * rows_per_worker;
                let w_rows = (rows - r_beg).min(rows_per_worker);
                s.spawn(move || {
                    let mut w_tile = vec![0.0f32; TILE_R.min(w_rows) * cols];
                    let mut loc_r = 0usize;
                    while loc_r < w_rows {
                        let br = (w_rows - loc_r).min(TILE_R);
                        for dr in 0..br {
                            self.decode_row(
                                r_beg + loc_r + dr,
                                &mut w_tile[dr * cols..(dr + 1) * cols],
                            );
                        }
                        dense_tile_gemm(
                            x_batch,
                            batch_m,
                            cols,
                            &w_tile[..br * cols],
                            br,
                            part_y,
                            w_rows,
                            loc_r,
                        );
                        loc_r += br;
                    }
                });
            }
        });

        for (wi, part_y) in partial.iter().enumerate() {
            let r_beg = wi * rows_per_worker;
            let w_rows = (rows - r_beg).min(rows_per_worker);
            for m in 0..batch_m {
                y_batch[m * rows + r_beg..m * rows + r_beg + w_rows]
                    .copy_from_slice(&part_y[m * w_rows..(m + 1) * w_rows]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_full_gqa_q8(
    q_raw: &[f32],
    k_cur: &mut [f32],
    v_cur: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    gated_q: bool,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    pos: usize,
    rope_pos: usize,
    rope_dim: usize,
    rope_freq_base: f32,
    rope_neox: bool,
    eps: f32,
    k_q8: &mut Vec<KvQ8Block>,
    v_q8: &mut Vec<KvQ8Block>,
    attn_out: &mut [f32],
) {
    debug_assert_eq!(head_dim % 32, 0);
    let blks_per_head = head_dim / 32;
    let kv_blk_stride = n_head_kv * blks_per_head;

    let mut q_heads = vec![0.0f32; n_head * head_dim];
    let mut q_gates = vec![0.0f32; if gated_q { n_head * head_dim } else { 0 }];

    for h in 0..n_head {
        let q_dst = &mut q_heads[h * head_dim..(h + 1) * head_dim];
        if gated_q {
            let src = &q_raw[h * 2 * head_dim..(h + 1) * 2 * head_dim];
            q_dst.copy_from_slice(&src[..head_dim]);
            q_gates[h * head_dim..(h + 1) * head_dim].copy_from_slice(&src[head_dim..]);
        } else {
            q_dst.copy_from_slice(&q_raw[h * head_dim..(h + 1) * head_dim]);
        }
        if let Some(qn) = q_norm {
            let tmp = q_dst.to_vec();
            rms_norm(&tmp, qn, eps, q_dst);
        }
        apply_rope(q_dst, rope_pos, rope_dim, rope_freq_base, rope_neox);
    }

    for kh in 0..n_head_kv {
        let ks = &mut k_cur[kh * head_dim..(kh + 1) * head_dim];
        if let Some(kn) = k_norm {
            let tmp = ks.to_vec();
            rms_norm(&tmp, kn, eps, ks);
        }
        apply_rope(ks, rope_pos, rope_dim, rope_freq_base, rope_neox);
    }

    let need_blks = (pos + 1) * kv_blk_stride;
    let zero_blk = KvQ8Block { d: 0, qs: [0; 32] };
    if k_q8.len() < need_blks {
        k_q8.resize(need_blks, zero_blk);
        v_q8.resize(need_blks, zero_blk);
    }
    let dst_base = pos * kv_blk_stride;
    for kh in 0..n_head_kv {
        for b in 0..blks_per_head {
            let off = kh * head_dim + b * 32;
            k_q8[dst_base + kh * blks_per_head + b] = KvQ8Block::quantize(&k_cur[off..off + 32]);
            v_q8[dst_base + kh * blks_per_head + b] = KvQ8Block::quantize(&v_cur[off..off + 32]);
        }
    }

    let gqa_mul = n_head / n_head_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let seq_len = pos + 1;
    let mut scores = vec![0.0f32; seq_len];

    for h in 0..n_head {
        let kv_h = h / gqa_mul;
        let qh = &q_heads[h * head_dim..(h + 1) * head_dim];
        let mut max_s = f32::NEG_INFINITY;
        for (t, sc) in scores.iter_mut().enumerate() {
            let kb = &k_q8[t * kv_blk_stride + kv_h * blks_per_head..][..blks_per_head];
            let mut dot = 0.0f32;
            for b in 0..blks_per_head {
                dot += kb[b].dot_f32(&qh[b * 32..(b + 1) * 32]);
            }
            let s = dot * scale;
            *sc = s;
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_exp = 0.0f32;
        for sc in &mut scores {
            let e = (*sc - max_s).exp();
            *sc = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        let out_h = &mut attn_out[h * head_dim..(h + 1) * head_dim];
        out_h.fill(0.0);
        for (t, &e) in scores.iter().enumerate() {
            let w = e * inv_sum;
            let vb = &v_q8[t * kv_blk_stride + kv_h * blks_per_head..][..blks_per_head];
            for b in 0..blks_per_head {
                vb[b].axpy_f32(w, &mut out_h[b * 32..(b + 1) * 32]);
            }
        }
        if gated_q {
            let gh = &q_gates[h * head_dim..(h + 1) * head_dim];
            for (o, &g) in out_h.iter_mut().zip(gh.iter()) {
                *o *= 1.0 / (1.0 + (-g).exp());
            }
        }
    }
}

pub(crate) fn cpu_kv_cache_bytes(state: &LlmState) -> usize {
    let f32_bytes: usize = state
        .k_caches
        .iter()
        .chain(state.v_caches.iter())
        .map(|v| v.capacity() * 4)
        .sum();
    let q8_bytes: usize = state
        .k_q8
        .iter()
        .chain(state.v_q8.iter())
        .map(|v| v.capacity() * core::mem::size_of::<KvQ8Block>())
        .sum();
    f32_bytes + q8_bytes
}

pub(crate) fn prefill_cpu_tiled(
    model: &mut LlmModel,
    state: &mut LlmState,
    prompt_ids: &[u32],
    logits_out: &mut [f32],
) -> Result<()> {
    if prompt_ids.is_empty() {
        return Ok(());
    }
    if prompt_ids.len() == 1 || model.metal.is_some() {
        for (idx, &tid) in prompt_ids.iter().enumerate() {
            let last = idx + 1 == prompt_ids.len();
            model.forward_step(state, tid, last.then_some(&mut *logits_out))?;
        }
        return Ok(());
    }

    let dim = model.cfg.n_embd;
    let n_ff = model.cfg.n_ff;
    let eps = model.cfg.rms_norm_eps;
    let threads = model.cpu_threads;
    let use_q8 = state.kv_mode == KvCacheMode::Q8_0 && model.cfg.head_dim % 32 == 0;
    const CHUNK_M: usize = 32;

    let mut off = 0usize;
    while off < prompt_ids.len() {
        let m = (prompt_ids.len() - off).min(CHUNK_M);
        let is_last_chunk = off + m == prompt_ids.len();
        let mut cur = vec![0.0f32; m * dim];
        for (ti, &tid) in prompt_ids[off..off + m].iter().enumerate() {
            let row = &mut cur[ti * dim..(ti + 1) * dim];
            model.token_embd.cpu.decode_row(tid as usize, row);
            if (model.cfg.embed_scale - 1.0).abs() > 1e-7 {
                for v in row.iter_mut() {
                    *v *= model.cfg.embed_scale;
                }
            }
        }

        let mut normed = vec![0.0f32; m * dim];
        let mut attn_out = vec![0.0f32; m * dim];
        let mut ffn_gate_b = vec![0.0f32; m * n_ff];
        let mut ffn_up_b = vec![0.0f32; m * n_ff];
        let mut ffn_out = vec![0.0f32; m * dim];

        for il in 0..model.cfg.n_layer {
            let layer: &DecoderLayer = &model.layers[il];
            for ti in 0..m {
                rms_norm(
                    &cur[ti * dim..(ti + 1) * dim],
                    &layer.attn_norm,
                    eps,
                    &mut normed[ti * dim..(ti + 1) * dim],
                );
            }

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
                    let qkv_r = wqkv.rows();
                    let d_in = model.cfg.ssm_inner_size;
                    let n_vh = model.cfg.ssm_dt_rank;
                    let mut qkv_b = vec![0.0f32; m * qkv_r];
                    let mut z_b = vec![0.0f32; m * d_in];
                    let mut a_b = vec![0.0f32; m * n_vh];
                    let mut b_b = vec![0.0f32; m * n_vh];
                    let mut y_b = vec![0.0f32; m * d_in];

                    wqkv.cpu.gemm_cpu(&normed, m, &mut qkv_b, threads);
                    wqkv_gate.cpu.gemm_cpu(&normed, m, &mut z_b, threads);
                    ssm_alpha.cpu.gemm_cpu(&normed, m, &mut a_b, threads);
                    ssm_beta.cpu.gemm_cpu(&normed, m, &mut b_b, threads);

                    for ti in 0..m {
                        step_deltanet_recurrent(
                            &qkv_b[ti * qkv_r..(ti + 1) * qkv_r], &z_b[ti * d_in..(ti + 1) * d_in],
                            &a_b[ti * n_vh..(ti + 1) * n_vh], &b_b[ti * n_vh..(ti + 1) * n_vh],
                            ssm_conv1d, ssm_a, ssm_dt_bias, ssm_norm,
                            &mut state.conv_states[il], &mut state.ssm_states[il],
                            model.cfg.ssm_group_count, n_vh, model.cfg.ssm_state_size,
                            d_in / n_vh, model.cfg.ssm_conv_kernel, eps,
                            &mut y_b[ti * d_in..(ti + 1) * d_in],
                        );
                    }
                    ssm_out.cpu.gemm_cpu(&y_b, m, &mut attn_out, threads);
                }
                LayerAttn::FullGqa {
                    wq, wk, wv, wo, q_norm, k_norm, q_bias, k_bias, v_bias, gated_q,
                } => {
                    let (nh, nkv, hd) = (model.cfg.n_head, model.cfg.n_head_kv, model.cfg.head_dim);
                    let (qr, kvd, attd) = (wq.rows(), nkv * hd, nh * hd);
                    let mut q_b = vec![0.0f32; m * qr];
                    let mut k_b = vec![0.0f32; m * kvd];
                    let mut v_b = vec![0.0f32; m * kvd];
                    let mut att_b = vec![0.0f32; m * attd];

                    wq.cpu.gemm_cpu(&normed, m, &mut q_b, threads);
                    wk.cpu.gemm_cpu(&normed, m, &mut k_b, threads);
                    wv.cpu.gemm_cpu(&normed, m, &mut v_b, threads);

                    for ti in 0..m {
                        if let Some(b) = q_bias {
                            for (v, &bv) in q_b[ti * qr..(ti + 1) * qr].iter_mut().zip(b) { *v += bv; }
                        }
                        if let Some(b) = k_bias {
                            for (v, &bv) in k_b[ti * kvd..(ti + 1) * kvd].iter_mut().zip(b) { *v += bv; }
                        }
                        if let Some(b) = v_bias {
                            for (v, &bv) in v_b[ti * kvd..(ti + 1) * kvd].iter_mut().zip(b) { *v += bv; }
                        }
                        let pos_t = state.pos + ti;
                        let rpos_t = ((pos_t as isize) + state.rope_delta).max(0) as usize;
                        if use_q8 {
                            step_full_gqa_q8(
                                &q_b[ti * qr..(ti + 1) * qr], &mut k_b[ti * kvd..(ti + 1) * kvd],
                                &v_b[ti * kvd..(ti + 1) * kvd], q_norm.as_deref(), k_norm.as_deref(),
                                *gated_q, nh, nkv, hd, pos_t, rpos_t,
                                model.cfg.rope_dim, model.cfg.rope_freq_base, model.cfg.rope_neox, eps,
                                &mut state.k_q8[il], &mut state.v_q8[il],
                                &mut att_b[ti * attd..(ti + 1) * attd],
                            );
                        } else {
                            step_full_gqa(
                                &q_b[ti * qr..(ti + 1) * qr], &mut k_b[ti * kvd..(ti + 1) * kvd],
                                &v_b[ti * kvd..(ti + 1) * kvd], q_norm.as_deref(), k_norm.as_deref(),
                                *gated_q, nh, nkv, hd, pos_t,
                                model.cfg.rope_dim, model.cfg.rope_freq_base, model.cfg.rope_neox, eps,
                                &mut state.k_caches[il], &mut state.v_caches[il],
                                &mut att_b[ti * attd..(ti + 1) * attd],
                            );
                        }
                    }
                    wo.cpu.gemm_cpu(&att_b, m, &mut attn_out, threads);
                }
            }

            if let Some(apn) = layer.attn_post_norm.as_deref() {
                for ti in 0..m {
                    rms_norm_inplace(&mut attn_out[ti * dim..(ti + 1) * dim], apn, eps);
                }
            }
            for (c, &d) in cur.iter_mut().zip(attn_out.iter()) {
                *c += d;
            }
            for ti in 0..m {
                rms_norm(
                    &cur[ti * dim..(ti + 1) * dim],
                    &layer.post_attn_norm,
                    eps,
                    &mut normed[ti * dim..(ti + 1) * dim],
                );
            }

            layer.ffn_gate.cpu.gemm_cpu(&normed, m, &mut ffn_gate_b, threads);
            layer.ffn_up.cpu.gemm_cpu(&normed, m, &mut ffn_up_b, threads);
            if model.cfg.ffn_gelu {
                for (g, &u) in ffn_gate_b.iter_mut().zip(ffn_up_b.iter()) {
                    *g = gelu_tanh_f32(*g) * u;
                }
            } else {
                for (g, &u) in ffn_gate_b.iter_mut().zip(ffn_up_b.iter()) {
                    *g = silu(*g) * u;
                }
            }
            layer.ffn_down.cpu.gemm_cpu(&ffn_gate_b, m, &mut ffn_out, threads);

            if let Some(fpn) = layer.ffn_post_norm.as_deref() {
                for ti in 0..m {
                    rms_norm_inplace(&mut ffn_out[ti * dim..(ti + 1) * dim], fpn, eps);
                }
            }
            for (c, &d) in cur.iter_mut().zip(ffn_out.iter()) {
                *c += d;
            }
        }

        state.pos += m;
        if is_last_chunk {
            let mut last_norm = vec![0.0f32; dim];
            rms_norm(
                &cur[(m - 1) * dim..m * dim],
                &model.output_norm,
                eps,
                &mut last_norm,
            );
            let head = model.output_head.as_ref().unwrap_or(&model.token_embd);
            head.cpu.gemv_cpu(&last_norm, logits_out, threads);
        }
        off += m;
    }
    Ok(())
}
