//! Fast head-outer L1-resident SIMD calibration sequence runners (DeltaNet + GQA).
//! Numerically identical to `llm_ops::{step_deltanet_recurrent, step_full_gqa}` (`< 1e-5`).

use crate::gguf_meta::ModelConfig;
use crate::llm_ops::{apply_rope, l2_norm_scaled, rms_norm_inplace, sigmoid, silu};

#[inline(always)]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0usize;
    while i + 8 <= n {
        s0 = a[i].mul_add(b[i], s0);
        s1 = a[i + 1].mul_add(b[i + 1], s1);
        s2 = a[i + 2].mul_add(b[i + 2], s2);
        s3 = a[i + 3].mul_add(b[i + 3], s3);
        s4 = a[i + 4].mul_add(b[i + 4], s4);
        s5 = a[i + 5].mul_add(b[i + 5], s5);
        s6 = a[i + 6].mul_add(b[i + 6], s6);
        s7 = a[i + 7].mul_add(b[i + 7], s7);
        i += 8;
    }
    let mut s = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    while i < n {
        s = a[i].mul_add(b[i], s);
        i += 1;
    }
    s
}

/// Processes `4` rows of DeltaNet state `sh[jv..jv+4, 0..dk]` simultaneously so `k` and `q` stay in registers.
#[inline(always)]
fn deltanet_4rows(
    sh: &mut [f32], dk: usize, decay: f32, beta: f32,
    v4: &[f32], k: &[f32], q: &[f32], o4: &mut [f32],
) {
    let (r01, r23) = sh.split_at_mut(2 * dk);
    let (r0, r1) = r01.split_at_mut(dk);
    let (r2, r3) = r23.split_at_mut(dk);
    let (mut sk0, mut sk1, mut sk2, mut sk3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    let mut i = 0usize;
    while i + 4 <= dk {
        for l in 0..4 {
            let kv = k[i + l];
            sk0[l] = r0[i + l].mul_add(kv, sk0[l]);
            sk1[l] = r1[i + l].mul_add(kv, sk1[l]);
            sk2[l] = r2[i + l].mul_add(kv, sk2[l]);
            sk3[l] = r3[i + l].mul_add(kv, sk3[l]);
        }
        i += 4;
    }
    while i < dk {
        let kv = k[i];
        sk0[0] = r0[i].mul_add(kv, sk0[0]);
        sk1[0] = r1[i].mul_add(kv, sk1[0]);
        sk2[0] = r2[i].mul_add(kv, sk2[0]);
        sk3[0] = r3[i].mul_add(kv, sk3[0]);
        i += 1;
    }
    let h4 = |a: [f32; 4]| (a[0] + a[2]) + (a[1] + a[3]);
    let d0 = (v4[0] - h4(sk0) * decay) * beta;
    let d1 = (v4[1] - h4(sk1) * decay) * beta;
    let d2 = (v4[2] - h4(sk2) * decay) * beta;
    let d3 = (v4[3] - h4(sk3) * decay) * beta;
    let (mut o0, mut o1, mut o2, mut o3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    i = 0;
    while i + 4 <= dk {
        for l in 0..4 {
            let (kv, qv) = (k[i + l], q[i + l]);
            let n0 = d0.mul_add(kv, r0[i + l] * decay); r0[i + l] = n0; o0[l] = n0.mul_add(qv, o0[l]);
            let n1 = d1.mul_add(kv, r1[i + l] * decay); r1[i + l] = n1; o1[l] = n1.mul_add(qv, o1[l]);
            let n2 = d2.mul_add(kv, r2[i + l] * decay); r2[i + l] = n2; o2[l] = n2.mul_add(qv, o2[l]);
            let n3 = d3.mul_add(kv, r3[i + l] * decay); r3[i + l] = n3; o3[l] = n3.mul_add(qv, o3[l]);
        }
        i += 4;
    }
    while i < dk {
        let (kv, qv) = (k[i], q[i]);
        let n0 = d0.mul_add(kv, r0[i] * decay); r0[i] = n0; o0[0] = n0.mul_add(qv, o0[0]);
        let n1 = d1.mul_add(kv, r1[i] * decay); r1[i] = n1; o1[0] = n1.mul_add(qv, o1[0]);
        let n2 = d2.mul_add(kv, r2[i] * decay); r2[i] = n2; o2[0] = n2.mul_add(qv, o2[0]);
        let n3 = d3.mul_add(kv, r3[i] * decay); r3[i] = n3; o3[0] = n3.mul_add(qv, o3[0]);
        i += 1;
    }
    o4[0] = h4(o0);
    o4[1] = h4(o1);
    o4[2] = h4(o2);
    o4[3] = h4(o3);
}

#[inline(always)]
fn deltanet_row_update_dot(row: &mut [f32], decay: f32, d_jv: f32, k: &[f32], q: &[f32]) -> f32 {
    let n = row.len();
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut i = 0usize;
    while i + 4 <= n {
        let r0 = d_jv.mul_add(k[i], row[i] * decay); row[i] = r0; s0 = r0.mul_add(q[i], s0);
        let r1 = d_jv.mul_add(k[i + 1], row[i + 1] * decay); row[i + 1] = r1; s1 = r1.mul_add(q[i + 1], s1);
        let r2 = d_jv.mul_add(k[i + 2], row[i + 2] * decay); row[i + 2] = r2; s2 = r2.mul_add(q[i + 2], s2);
        let r3 = d_jv.mul_add(k[i + 3], row[i + 3] * decay); row[i + 3] = r3; s3 = r3.mul_add(q[i + 3], s3);
        i += 4;
    }
    let mut s = (s0 + s2) + (s1 + s3);
    while i < n {
        let r = d_jv.mul_add(k[i], row[i] * decay);
        row[i] = r;
        s = r.mul_add(q[i], s);
        i += 1;
    }
    s
}

/// Head-outer L1-resident SIMD DeltaNet runner (numerically identical to `step_deltanet_recurrent`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_deltanet_calib_fast(
    cfg: &ModelConfig, n_seq: usize, seq_len: usize, qkv: &[f32], z: &[f32],
    alpha: &[f32], beta: &[f32], conv1d: &[f32], ssm_a: &[f32], dt_bias: &[f32], ssm_norm: &[f32],
    threads: usize,
) -> Vec<f32> {
    let (d_inner, n_vh, n_kh, dk) = (cfg.ssm_inner_size, cfg.ssm_dt_rank, cfg.ssm_group_count, cfg.ssm_state_size);
    let dv = d_inner / n_vh.max(1);
    let qk_dim = n_kh * dk;
    let conv_ch = d_inner + 2 * qk_dim;
    let (k_sz, hist, eps) = (cfg.ssm_conv_kernel, cfg.ssm_conv_kernel.saturating_sub(1), cfg.rms_norm_eps);
    let q_scale = 1.0 / (dk as f32).sqrt();
    let mut out = vec![0.0f32; n_seq * seq_len * d_inner];
    let workers = threads.clamp(1, 32).min(n_seq.max(1));
    let spw = n_seq.div_ceil(workers);

    std::thread::scope(|s| {
        for (wi, out_blk) in out.chunks_mut(spw * seq_len * d_inner).enumerate() {
            let si0 = wi * spw;
            let n_loc = out_blk.len() / (seq_len * d_inner).max(1);
            s.spawn(move || {
                let mut conv_st = vec![0.0f32; hist * conv_ch];
                let mut conv_silu = vec![0.0f32; seq_len * conv_ch];
                let mut sh = vec![0.0f32; dv * dk];
                let (mut q_h, mut k_h, mut o_h) = (vec![0.0f32; dk], vec![0.0f32; dk], vec![0.0f32; dv]);
                for dsi in 0..n_loc {
                    let si = si0 + dsi;
                    let out_s = &mut out_blk[dsi * seq_len * d_inner..(dsi + 1) * seq_len * d_inner];
                    conv_st.fill(0.0);
                    for p in 0..seq_len {
                        let t = si * seq_len + p;
                        let qkv_t = &qkv[t * conv_ch..(t + 1) * conv_ch];
                        let dst_c = &mut conv_silu[p * conv_ch..(p + 1) * conv_ch];
                        for c in 0..conv_ch {
                            let w = &conv1d[c * k_sz..(c + 1) * k_sz];
                            let mut acc = 0.0f32;
                            for k in 0..hist { acc = conv_st[k * conv_ch + c].mul_add(w[k], acc); }
                            acc = qkv_t[c].mul_add(w[hist], acc);
                            for k in 0..hist.saturating_sub(1) { conv_st[k * conv_ch + c] = conv_st[(k + 1) * conv_ch + c]; }
                            if hist > 0 { conv_st[(hist - 1) * conv_ch + c] = qkv_t[c]; }
                            dst_c[c] = silu(acc);
                        }
                    }
                    for h in 0..n_vh {
                        let hk = h / (n_vh / n_kh).max(1);
                        let (sa, dtb) = (ssm_a[h], dt_bias[h]);
                        sh.fill(0.0);
                        for p in 0..seq_len {
                            let t = si * seq_len + p;
                            let cs = &conv_silu[p * conv_ch..(p + 1) * conv_ch];
                            l2_norm_scaled(&cs[hk * dk..(hk + 1) * dk], eps, q_scale, &mut q_h);
                            l2_norm_scaled(&cs[qk_dim + hk * dk..qk_dim + (hk + 1) * dk], eps, 1.0, &mut k_h);
                            let v_h = &cs[2 * qk_dim + h * dv..2 * qk_dim + (h + 1) * dv];
                            let b_val = sigmoid(beta[t * n_vh + h]);
                            let x_sp = alpha[t * n_vh + h] + dtb;
                            let sp = if x_sp > 20.0 { x_sp } else if x_sp < -20.0 { x_sp.exp() } else { (1.0 + x_sp.exp()).ln() };
                            let decay = (sp * sa).exp();
                            let mut jv = 0usize;
                            while jv + 4 <= dv {
                                deltanet_4rows(
                                    &mut sh[jv * dk..(jv + 4) * dk], dk, decay, b_val,
                                    &v_h[jv..jv + 4], &k_h, &q_h, &mut o_h[jv..jv + 4],
                                );
                                jv += 4;
                            }
                            while jv < dv {
                                let row = &mut sh[jv * dk..(jv + 1) * dk];
                                let sk = dot8(row, &k_h) * decay;
                                let d_jv = (v_h[jv] - sk) * b_val;
                                o_h[jv] = deltanet_row_update_dot(row, decay, d_jv, &k_h, &q_h);
                                jv += 1;
                            }
                            rms_norm_inplace(&mut o_h, ssm_norm, eps);
                            let z_h = &z[t * d_inner + h * dv..t * d_inner + (h + 1) * dv];
                            let dst_h = &mut out_s[p * d_inner + h * dv..p * d_inner + (h + 1) * dv];
                            for j in 0..dv { dst_h[j] = o_h[j] * silu(z_h[j]); }
                        }
                    }
                }
            });
        }
    });
    out
}

/// Zero-alloc SIMD GQA calibration runner (numerically identical to `step_full_gqa`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gqa_calib_fast(
    cfg: &ModelConfig, n_seq: usize, seq_len: usize, gated_q: bool,
    q: &[f32], k: &[f32], v: &[f32], q_norm: Option<&[f32]>, k_norm: Option<&[f32]>, threads: usize,
) -> Vec<f32> {
    let (nh, nkv, hd, eps) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.rms_norm_eps);
    let q_dim = q.len() / (n_seq * seq_len).max(1);
    let (kv_dim, att_dim) = (nkv * hd, nh * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; n_seq * seq_len * att_dim];
    let workers = threads.clamp(1, 32).min(n_seq.max(1));
    let spw = n_seq.div_ceil(workers);

    std::thread::scope(|s| {
        for (wi, out_blk) in out.chunks_mut(spw * seq_len * att_dim).enumerate() {
            let si0 = wi * spw;
            let n_loc = out_blk.len() / (seq_len * att_dim).max(1);
            s.spawn(move || {
                let mut kc = vec![0.0f32; seq_len * kv_dim];
                let mut vc = vec![0.0f32; seq_len * kv_dim];
                let mut q_cur = vec![0.0f32; nh * hd];
                let mut q_gate = vec![0.0f32; if gated_q { nh * hd } else { 0 }];
                let mut scores = vec![0.0f32; seq_len];
                for dsi in 0..n_loc {
                    let si = si0 + dsi;
                    let out_s = &mut out_blk[dsi * seq_len * att_dim..(dsi + 1) * seq_len * att_dim];
                    for p in 0..seq_len {
                        let t = si * seq_len + p;
                        let q_raw = &q[t * q_dim..(t + 1) * q_dim];
                        if gated_q {
                            for h in 0..nh {
                                q_cur[h * hd..(h + 1) * hd].copy_from_slice(&q_raw[h * 2 * hd..h * 2 * hd + hd]);
                                q_gate[h * hd..(h + 1) * hd].copy_from_slice(&q_raw[h * 2 * hd + hd..(h + 1) * 2 * hd]);
                            }
                        } else {
                            q_cur.copy_from_slice(q_raw);
                        }
                        let k_slot = &mut kc[p * kv_dim..(p + 1) * kv_dim];
                        k_slot.copy_from_slice(&k[t * kv_dim..(t + 1) * kv_dim]);
                        vc[p * kv_dim..(p + 1) * kv_dim].copy_from_slice(&v[t * kv_dim..(t + 1) * kv_dim]);
                        if let Some(qn) = q_norm { for h in 0..nh { rms_norm_inplace(&mut q_cur[h * hd..(h + 1) * hd], qn, eps); } }
                        if let Some(kn) = k_norm { for h in 0..nkv { rms_norm_inplace(&mut k_slot[h * hd..(h + 1) * hd], kn, eps); } }
                        for h in 0..nh { apply_rope(&mut q_cur[h * hd..(h + 1) * hd], p, cfg.rope_dim, cfg.rope_freq_base, cfg.rope_neox); }
                        for h in 0..nkv { apply_rope(&mut k_slot[h * hd..(h + 1) * hd], p, cfg.rope_dim, cfg.rope_freq_base, cfg.rope_neox); }

                        let slen = p + 1;
                        let att_p = &mut out_s[p * att_dim..(p + 1) * att_dim];
                        for h in 0..nh {
                            let kv_h = h / (nh / nkv);
                            let qh = &q_cur[h * hd..(h + 1) * hd];
                            let mut max_s = f32::NEG_INFINITY;
                            for tt in 0..slen {
                                let sv = dot8(qh, &kc[tt * kv_dim + kv_h * hd..tt * kv_dim + (kv_h + 1) * hd]) * scale;
                                scores[tt] = sv;
                                if sv > max_s { max_s = sv; }
                            }
                            let mut sum_exp = 0.0f32;
                            for sv in &mut scores[..slen] { *sv = (*sv - max_s).exp(); sum_exp += *sv; }
                            let inv_sum = 1.0 / sum_exp;
                            let out_h = &mut att_p[h * hd..(h + 1) * hd];
                            out_h.fill(0.0);
                            for tt in 0..slen {
                                let w = scores[tt] * inv_sum;
                                let vh = &vc[tt * kv_dim + kv_h * hd..tt * kv_dim + (kv_h + 1) * hd];
                                for d in 0..hd { out_h[d] = w.mul_add(vh[d], out_h[d]); }
                            }
                            if gated_q {
                                let gh = &q_gate[h * hd..(h + 1) * hd];
                                for d in 0..hd { out_h[d] *= sigmoid(gh[d]); }
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::run_deltanet_calib_fast;
    use crate::gguf_meta::ModelConfig;
    use crate::llm_ops::step_deltanet_recurrent;

    #[test]
    fn deltanet_calib_fast_matches_step_recurrent() {
        let (n_seq, seq_len, n_kh, n_vh, dk, dv, k_sz) = (2usize, 8, 2, 4, 32, 32, 4);
        let d_inner = n_vh * dv;
        let conv_ch = d_inner + 2 * n_kh * dk;
        let cfg = ModelConfig {
            arch: "qwen35".into(), model_name: "t".into(), n_layer: 1, n_embd: 128, n_ff: 128,
            n_head: 4, n_head_kv: 2, head_dim: 32, n_vocab: 64, rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0, rope_dim: 32, rope_neox: false, full_attn_interval: 4,
            ssm_conv_kernel: k_sz, ssm_state_size: dk, ssm_group_count: n_kh, ssm_dt_rank: n_vh,
            ssm_inner_size: d_inner, ffn_gelu: false, embed_scale: 1.0, bos_token_id: Some(1), eos_token_id: 2,
        };
        let gen = |n: usize, seed: usize| -> Vec<f32> {
            (0..n).map(|i| ((((i + seed) * 37) % 101) as f32 - 50.0) * 0.015).collect()
        };
        let (qkv, z) = (gen(n_seq * seq_len * conv_ch, 1), gen(n_seq * seq_len * d_inner, 2));
        let (alpha, beta) = (gen(n_seq * seq_len * n_vh, 3), gen(n_seq * seq_len * n_vh, 4));
        let (conv1d, ssm_a) = (gen(conv_ch * k_sz, 5), vec![-0.5f32; n_vh]);
        let (dt_bias, ssm_norm) = (gen(n_vh, 6), vec![1.0f32; dv]);
        let got = run_deltanet_calib_fast(&cfg, n_seq, seq_len, &qkv, &z, &alpha, &beta, &conv1d, &ssm_a, &dt_bias, &ssm_norm, 2);
        let mut ref_out = vec![0.0f32; n_seq * seq_len * d_inner];
        for si in 0..n_seq {
            let (mut cs, mut ss) = (vec![0.0f32; (k_sz - 1) * conv_ch], vec![0.0f32; n_vh * dv * dk]);
            for p in 0..seq_len {
                let t = si * seq_len + p;
                step_deltanet_recurrent(
                    &qkv[t * conv_ch..(t + 1) * conv_ch], &z[t * d_inner..(t + 1) * d_inner],
                    &alpha[t * n_vh..(t + 1) * n_vh], &beta[t * n_vh..(t + 1) * n_vh],
                    &conv1d, &ssm_a, &dt_bias, &ssm_norm, &mut cs, &mut ss,
                    n_kh, n_vh, dk, dv, k_sz, 1e-6, &mut ref_out[t * d_inner..(t + 1) * d_inner],
                );
            }
        }
        let max_err = got.iter().zip(&ref_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "deltanet fast parity max_err={max_err}");
    }
}
