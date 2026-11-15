//! **Hamle 2 + Hamle 3** sequential layer-by-layer Teacher (`BF16`) vs Student calibration
//! (`VXT-Turbo v3`): Super-Block `LDL-GPTQ`, Teacher drift cancellation, 2-sweep regularized
//! scales, RMSNorm / `ssm_alpha` / `ssm_beta` alignment, and Top-32 full-vocab `output_norm` KLD.

use std::collections::HashMap;

use crate::cb4_calib::Cb4TensorSource;
use crate::format::PackedTensor;
use crate::gguf_meta::ModelConfig;
use crate::llm_calib::turbo_v3_tensor_spec;
use crate::llm_ops::{rms_norm, rms_norm_inplace, silu, step_deltanet_recurrent, step_full_gqa};
use crate::llm_prefill::gelu_tanh_f32;
use crate::tokenizer::BpeTokenizer;
use crate::turbo_calib_corpus::build_calib_sequences;
use crate::turbo_full_gptq::encode_turbo_full_gptq;
use crate::turbo_head_kld::{
    align_rmsnorm_weights, align_tiny_proj_f16, optimize_output_norm_kld, TeacherHeadStats,
};
use crate::turbo_linalg::{effective_cpu_threads, sgemm_nt, SuperBlockHessian};
use crate::wht_turbo::TurboWhtMatrix;
use crate::{Error, Result};

fn pad_col(c: usize) -> usize {
    c.div_ceil(256) * 256
}

fn fetch_vec(src: &dyn Cb4TensorSource, name: &str) -> Result<Vec<f32>> {
    src.extract_f32(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}' during turbo v3 calibration")))
}

fn fetch_rows(src: &dyn Cb4TensorSource, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
    src.extract_rows_f32(name, r0, r1)
        .ok_or_else(|| Error::Format(format!("missing row slice {r0}..{r1} of '{name}'")))
}

fn batch_rms_norm(x: &[f32], w: &[f32], eps: f32, n_tok: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tok * dim];
    for t in 0..n_tok {
        rms_norm(&x[t * dim..(t + 1) * dim], w, eps, &mut out[t * dim..(t + 1) * dim]);
    }
    out
}

fn hidden_cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (fx, fy) = (x as f64, y as f64);
        ab += fx * fy;
        aa += fx * fx;
        bb += fy * fy;
    }
    if aa <= 1e-30 || bb <= 1e-30 { 0.0 } else { (ab / (aa.sqrt() * bb.sqrt())) as f32 }
}

fn build_embd_ch_amp(src: &dyn Cb4TensorSource, dim: usize) -> Vec<f32> {
    let a0 = src.extract_f32("blk.0.attn_norm.weight").filter(|v| v.len() == dim);
    let on = src.extract_f32("output_norm.weight").filter(|v| v.len() == dim);
    let mut raw: Vec<f32> = (0..dim)
        .map(|c| {
            let va = a0.as_ref().map_or(1.0, |v| v[c].abs());
            let vo = on.as_ref().map_or(1.0, |v| v[c].abs());
            (0.65 * va + 0.35 * vo).max(1e-4)
        })
        .collect();
    let rms0 = (raw.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dim.max(1) as f64).sqrt().max(1e-12);
    for v in &mut raw {
        *v = ((*v as f64 / rms0) as f32).clamp(0.25, 4.0);
    }
    let rms1 = (raw.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dim.max(1) as f64).sqrt().max(1e-12);
    let inv = (1.0 / rms1) as f32;
    for v in &mut raw {
        *v *= inv;
    }
    raw
}

#[allow(clippy::too_many_arguments)]
fn encode_token_embd_chunked(
    src: &dyn Cb4TensorSource, rows: usize, cols: usize, bits: u8,
    base_lam: f32, turbo_entropy_scale: f32, ch_amp: &[f32], eff_threads: usize,
) -> Result<TurboWhtMatrix> {
    const CHUNK: usize = 4096;
    let (padded_rows, padded_cols) = (rows.div_ceil(4) * 4, pad_col(cols));
    let (steps, wps) = (padded_cols / 256, TurboWhtMatrix::words_per_tile_step(bits));
    let mut scales_f16 = Vec::with_capacity((padded_rows / 4) * steps * 8);
    let mut packed_words = Vec::with_capacity((padded_rows / 4) * steps * wps);
    let inv_cols = 1.0 / (cols.max(1) as f64);
    let mut ref_mat_ms = None;

    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + CHUNK).min(rows);
        let rc = r1 - r0;
        let chunk = fetch_rows(src, "token_embd.weight", r0, r1)?;
        let row_ms: Vec<f32> = (0..rc)
            .map(|r| {
                let s2: f64 = chunk[r * cols..(r + 1) * cols].iter().map(|&v| (v as f64).powi(2)).sum();
                ((s2 * inv_cols) as f32).max(1e-12)
            })
            .collect();
        let mat_ms = *ref_mat_ms.get_or_insert_with(|| {
            let mut s = row_ms.clone();
            s.sort_by(|a, b| a.total_cmp(b));
            s.get(rc / 2).copied().unwrap_or(1e-6).max(1e-12)
        });
        let mat = TurboWhtMatrix::encode_with_ch_amp_threads(
            bits, rc, cols, &chunk, None, Some(ch_amp),
            |r| turbo_entropy_scale * if row_ms[r] > 3.0 * mat_ms { 0.0 } else { base_lam },
            eff_threads,
        )?;
        scales_f16.extend_from_slice(&mat.scales_f16);
        packed_words.extend_from_slice(&mat.packed_words);
        r0 = r1;
    }
    Ok(TurboWhtMatrix { bits, rows, cols, padded_rows, padded_cols, disk_bytes: 0, scales_f16, packed_words })
}

#[allow(clippy::too_many_arguments)]
fn pack_proj_with_teacher(
    src: &dyn Cb4TensorSource, name: &str, il: usize, cfg: &ModelConfig, has_output_weight: bool,
    hess: &SuperBlockHessian, x_ref: &[f32], x_stu: &[f32], n_tok: usize, turbo_entropy_scale: f32, eff_threads: usize,
) -> Result<(PackedTensor, Vec<f32>, Vec<f32>)> {
    let (rows, cols) = src.matrix_dims(name).ok_or_else(|| Error::Format(format!("missing projection '{name}'")))?;
    let (bits, base_lam) = turbo_v3_tensor_spec(name, Some(il), cfg.n_layer, has_output_weight);
    let (mut y_ref, mut tmp) = (vec![0.0f32; n_tok * rows], Vec::new());
    let mut hook = |r0: usize, r1: usize, w_chunk: &[f32]| {
        let rc = r1 - r0;
        tmp.resize(n_tok * rc, 0.0);
        sgemm_nt(n_tok, rc, cols, 1.0, x_ref, cols, w_chunk, cols, 0.0, &mut tmp, rc, eff_threads);
        for t in 0..n_tok { y_ref[t * rows + r0..t * rows + r1].copy_from_slice(&tmp[t * rc..(t + 1) * rc]); }
    };
    let (mat, y_stu) = encode_turbo_full_gptq(
        bits, rows, cols, |r0, r1| fetch_rows(src, name, r0, r1),
        hess, Some(x_ref), x_stu, n_tok, base_lam * turbo_entropy_scale, true, Some(&mut hook), eff_threads,
    )?;
    Ok((PackedTensor::TurboWht(mat), y_stu, y_ref))
}

#[allow(clippy::too_many_arguments)]
fn run_deltanet_sequences(
    cfg: &ModelConfig, n_seq: usize, seq_len: usize, qkv: &[f32], z: &[f32],
    alpha: &[f32], beta: &[f32], conv1d: &[f32], ssm_a: &[f32], dt_bias: &[f32], ssm_norm: &[f32],
) -> Vec<f32> {
    let (d_inner, n_vh) = (cfg.ssm_inner_size, cfg.ssm_dt_rank);
    let conv_ch = d_inner + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
    let conv_hist = cfg.ssm_conv_kernel.saturating_sub(1) * conv_ch;
    let dv = d_inner / n_vh.max(1);
    let ssm_len = n_vh * dv * cfg.ssm_state_size;
    let mut out = vec![0.0f32; n_seq * seq_len * d_inner];

    std::thread::scope(|s| {
        for (si, out_s) in out.chunks_exact_mut(seq_len * d_inner).enumerate() {
            s.spawn(move || {
                let (mut conv_st, mut ssm_st) = (vec![0.0f32; conv_hist], vec![0.0f32; ssm_len]);
                for p in 0..seq_len {
                    let t = si * seq_len + p;
                    step_deltanet_recurrent(
                        &qkv[t * conv_ch..(t + 1) * conv_ch], &z[t * d_inner..(t + 1) * d_inner],
                        &alpha[t * n_vh..(t + 1) * n_vh], &beta[t * n_vh..(t + 1) * n_vh],
                        conv1d, ssm_a, dt_bias, ssm_norm, &mut conv_st, &mut ssm_st,
                        cfg.ssm_group_count, n_vh, cfg.ssm_state_size, dv, cfg.ssm_conv_kernel,
                        cfg.rms_norm_eps, &mut out_s[p * d_inner..(p + 1) * d_inner],
                    );
                }
            });
        }
    });
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gqa_sequences(
    cfg: &ModelConfig, n_seq: usize, seq_len: usize, gated_q: bool,
    q: &[f32], k: &[f32], v: &[f32], q_norm: Option<&[f32]>, k_norm: Option<&[f32]>,
) -> Vec<f32> {
    let (nh, nkv, hd) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim);
    let q_dim = q.len() / (n_seq * seq_len).max(1);
    let (kv_dim, att_dim) = (nkv * hd, nh * hd);
    let mut out = vec![0.0f32; n_seq * seq_len * att_dim];

    std::thread::scope(|s| {
        for (si, out_s) in out.chunks_exact_mut(seq_len * att_dim).enumerate() {
            s.spawn(move || {
                let (mut kc, mut vc) = (Vec::with_capacity(seq_len * kv_dim), Vec::with_capacity(seq_len * kv_dim));
                let (mut k_tmp, mut v_tmp) = (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim]);
                for p in 0..seq_len {
                    let t = si * seq_len + p;
                    k_tmp.copy_from_slice(&k[t * kv_dim..(t + 1) * kv_dim]);
                    v_tmp.copy_from_slice(&v[t * kv_dim..(t + 1) * kv_dim]);
                    step_full_gqa(
                        &q[t * q_dim..(t + 1) * q_dim], &mut k_tmp, &mut v_tmp, q_norm, k_norm, gated_q,
                        nh, nkv, hd, p, cfg.rope_dim, cfg.rope_freq_base, cfg.rope_neox, cfg.rms_norm_eps,
                        &mut kc, &mut vc, &mut out_s[p * att_dim..(p + 1) * att_dim],
                    );
                }
            });
        }
    });
    out
}

fn add_bias_rows(buf: &mut [f32], bias: Option<&[f32]>, n_tok: usize, cols: usize) {
    if let Some(b) = bias {
        for t in 0..n_tok {
            let row = &mut buf[t * cols..(t + 1) * cols];
            for c in 0..cols.min(b.len()) { row[c] += b[c]; }
        }
    }
}

/// Runs sequential Teacher-vs-Student layer calibration and returns prepacked `.vxt` tensors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sequential_turbo_pack(
    src: &dyn Cb4TensorSource, cfg: &ModelConfig, tok: &BpeTokenizer, calib_text: Option<&str>,
    turbo_entropy_scale: f32, threads: usize, progress: Option<&(dyn Fn(usize, usize, &str) -> bool + Sync)>,
) -> Result<HashMap<String, PackedTensor>> {
    let eff_threads = effective_cpu_threads(threads);
    let seqs = build_calib_sequences(tok, calib_text, 128, 16);
    if seqs.is_empty() {
        return Err(Error::Format("sequential turbo calibration yielded zero sequences".into()));
    }
    let (n_seq, seq_len) = (seqs.len(), seqs[0].len());
    let (n_tok, dim, eps, dim_pad) = (n_seq * seq_len, cfg.n_embd, cfg.rms_norm_eps, pad_col(cfg.n_embd));
    let has_output_weight = src.has_tensor("output.weight");
    let total_steps = cfg.n_layer + 2;

    if let Some(cb) = progress {
        if !cb(0, total_steps, "Turbo v3 sequential kalibrasyon başlatılıyor...") {
            return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
        }
    }

    let (embd_rows, embd_cols) = src
        .matrix_dims("token_embd.weight")
        .ok_or_else(|| Error::Format("missing token_embd.weight".into()))?;
    if embd_cols != dim {
        return Err(Error::InvalidShape("token_embd.weight cols != n_embd"));
    }

    let mut row_cache: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut cur_ref = vec![0.0f32; n_tok * dim];
    for (si, seq) in seqs.iter().enumerate() {
        for (p, &tid) in seq.iter().enumerate() {
            let uid = (tid as usize).min(embd_rows - 1);
            let row = row_cache
                .entry(uid as u32)
                .or_insert_with(|| fetch_rows(src, "token_embd.weight", uid, uid + 1).unwrap_or_else(|_| vec![0.0; dim]));
            let dst = &mut cur_ref[(si * seq_len + p) * dim..(si * seq_len + p + 1) * dim];
            for c in 0..dim { dst[c] = row[c] * cfg.embed_scale; }
        }
    }

    let (emb_bits, emb_lam) = turbo_v3_tensor_spec("token_embd.weight", None, cfg.n_layer, has_output_weight);
    let ch_amp_embd = build_embd_ch_amp(src, dim);
    let emb_mat = encode_token_embd_chunked(src, embd_rows, dim, emb_bits, emb_lam, turbo_entropy_scale, &ch_amp_embd, eff_threads)?;
    let mut cur_stu = vec![0.0f32; n_tok * dim];
    for (si, seq) in seqs.iter().enumerate() {
        for (p, &tid) in seq.iter().enumerate() {
            let uid = (tid as usize).min(embd_rows - 1);
            let dst = &mut cur_stu[(si * seq_len + p) * dim..(si * seq_len + p + 1) * dim];
            emb_mat.decode_row(uid, dst);
            if (cfg.embed_scale - 1.0).abs() > 1e-7 {
                for v in dst { *v *= cfg.embed_scale; }
            }
        }
    }

    let mut out_map: HashMap<String, PackedTensor> = HashMap::new();
    out_map.insert("token_embd.weight".to_owned(), PackedTensor::TurboWht(emb_mat));

    for il in 0..cfg.n_layer {
        if let Some(cb) = progress {
            let msg = format!("Turbo v3 katman kalibrasyonu {}/{} (cos={:.4})", il + 1, cfg.n_layer, hidden_cos(&cur_stu, &cur_ref));
            if !cb(il + 1, total_steps, &msg) {
                return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
            }
        }
        let pfx = format!("blk.{il}");
        let an_name = format!("{pfx}.attn_norm.weight");
        let an_orig = fetch_vec(src, &an_name)?;
        let normed_ref = batch_rms_norm(&cur_ref, &an_orig, eps, n_tok, dim);
        let an_stu = align_rmsnorm_weights(&cur_stu, &normed_ref, &an_orig, eps, n_tok, dim);
        let normed_stu = batch_rms_norm(&cur_stu, &an_stu, eps, n_tok, dim);
        out_map.insert(an_name, PackedTensor::RawF32 { rows: 1, cols: dim, data: an_stu });

        let hess_attn = SuperBlockHessian::build(&normed_stu, Some(&normed_ref), n_tok, dim, dim_pad, eff_threads);
        let (mut attn_out_stu, mut attn_out_ref) = if cfg.is_recurrent_layer(il) {
            let (qkv_name, gate_name) = (format!("{pfx}.attn_qkv.weight"), format!("{pfx}.attn_gate.weight"));
            let (pt_qkv, qkv_s, qkv_r) = pack_proj_with_teacher(src, &qkv_name, il, cfg, has_output_weight, &hess_attn, &normed_ref, &normed_stu, n_tok, turbo_entropy_scale, eff_threads)?;
            let (pt_z, z_s, z_r) = pack_proj_with_teacher(src, &gate_name, il, cfg, has_output_weight, &hess_attn, &normed_ref, &normed_stu, n_tok, turbo_entropy_scale, eff_threads)?;
            out_map.insert(qkv_name, pt_qkv);
            out_map.insert(gate_name, pt_z);

            let (a_name, b_name) = (format!("{pfx}.ssm_alpha.weight"), format!("{pfx}.ssm_beta.weight"));
            let (ar, ac) = src.matrix_dims(&a_name).ok_or_else(|| Error::Format(format!("missing {a_name}")))?;
            let (br, bc) = src.matrix_dims(&b_name).ok_or_else(|| Error::Format(format!("missing {b_name}")))?;
            let (pt_a, a_s, a_r) = align_tiny_proj_f16(&fetch_vec(src, &a_name)?, ar, ac, &normed_stu, &normed_ref, n_tok, eff_threads)?;
            let (pt_b, b_s, b_r) = align_tiny_proj_f16(&fetch_vec(src, &b_name)?, br, bc, &normed_stu, &normed_ref, n_tok, eff_threads)?;
            out_map.insert(a_name, pt_a);
            out_map.insert(b_name, pt_b);

            let conv1d = fetch_vec(src, &format!("{pfx}.ssm_conv1d.weight"))?;
            let ssm_a = fetch_vec(src, &format!("{pfx}.ssm_a"))?;
            let dt_bias = fetch_vec(src, &format!("{pfx}.ssm_dt.bias"))?;
            let ssm_norm = fetch_vec(src, &format!("{pfx}.ssm_norm.weight"))?;
            let y_r = run_deltanet_sequences(cfg, n_seq, seq_len, &qkv_r, &z_r, &a_r, &b_r, &conv1d, &ssm_a, &dt_bias, &ssm_norm);
            let y_s = run_deltanet_sequences(cfg, n_seq, seq_len, &qkv_s, &z_s, &a_s, &b_s, &conv1d, &ssm_a, &dt_bias, &ssm_norm);
            let d_in = cfg.ssm_inner_size;
            let hess_o = SuperBlockHessian::build(&y_s, Some(&y_r), n_tok, d_in, pad_col(d_in), eff_threads);
            let sout_name = format!("{pfx}.ssm_out.weight");
            let (pt_o, os, or_) = pack_proj_with_teacher(src, &sout_name, il, cfg, has_output_weight, &hess_o, &y_r, &y_s, n_tok, turbo_entropy_scale, eff_threads)?;
            out_map.insert(sout_name, pt_o);
            (os, or_)
        } else {
            let (qn, kn, vn) = (format!("{pfx}.attn_q.weight"), format!("{pfx}.attn_k.weight"), format!("{pfx}.attn_v.weight"));
            let (pt_q, mut qs, mut qr) = pack_proj_with_teacher(src, &qn, il, cfg, has_output_weight, &hess_attn, &normed_ref, &normed_stu, n_tok, turbo_entropy_scale, eff_threads)?;
            let (pt_k, mut ks, mut kr) = pack_proj_with_teacher(src, &kn, il, cfg, has_output_weight, &hess_attn, &normed_ref, &normed_stu, n_tok, turbo_entropy_scale, eff_threads)?;
            let (pt_v, mut vs, mut vr) = pack_proj_with_teacher(src, &vn, il, cfg, has_output_weight, &hess_attn, &normed_ref, &normed_stu, n_tok, turbo_entropy_scale, eff_threads)?;
            let q_rows = pt_q.rows();
            out_map.insert(qn, pt_q);
            out_map.insert(kn, pt_k);
            out_map.insert(vn, pt_v);

            let (qb, kb, vb) = (src.extract_f32(&format!("{pfx}.attn_q.bias")), src.extract_f32(&format!("{pfx}.attn_k.bias")), src.extract_f32(&format!("{pfx}.attn_v.bias")));
            add_bias_rows(&mut qs, qb.as_deref(), n_tok, q_rows);
            add_bias_rows(&mut qr, qb.as_deref(), n_tok, q_rows);
            let kv_dim = cfg.n_head_kv * cfg.head_dim;
            add_bias_rows(&mut ks, kb.as_deref(), n_tok, kv_dim);
            add_bias_rows(&mut kr, kb.as_deref(), n_tok, kv_dim);
            add_bias_rows(&mut vs, vb.as_deref(), n_tok, kv_dim);
            add_bias_rows(&mut vr, vb.as_deref(), n_tok, kv_dim);

            let q_norm = src.extract_f32(&format!("{pfx}.attn_q_norm.weight"));
            let k_norm = src.extract_f32(&format!("{pfx}.attn_k_norm.weight"));
            let gated_q = q_rows == cfg.n_head * cfg.head_dim * 2;
            let ar = run_gqa_sequences(cfg, n_seq, seq_len, gated_q, &qr, &kr, &vr, q_norm.as_deref(), k_norm.as_deref());
            let as_ = run_gqa_sequences(cfg, n_seq, seq_len, gated_q, &qs, &ks, &vs, q_norm.as_deref(), k_norm.as_deref());
            let att_dim = cfg.n_head * cfg.head_dim;
            let hess_o = SuperBlockHessian::build(&as_, Some(&ar), n_tok, att_dim, pad_col(att_dim), eff_threads);
            let oname = format!("{pfx}.attn_output.weight");
            let (pt_o, os, or_) = pack_proj_with_teacher(src, &oname, il, cfg, has_output_weight, &hess_o, &ar, &as_, n_tok, turbo_entropy_scale, eff_threads)?;
            out_map.insert(oname, pt_o);
            (os, or_)
        };

        if let Some(apn) = src.extract_f32(&format!("{pfx}.attn_post_norm.weight")) {
            for t in 0..n_tok {
                rms_norm_inplace(&mut attn_out_ref[t * dim..(t + 1) * dim], &apn, eps);
                rms_norm_inplace(&mut attn_out_stu[t * dim..(t + 1) * dim], &apn, eps);
            }
        }
        for i in 0..(n_tok * dim) {
            cur_ref[i] += attn_out_ref[i];
            cur_stu[i] += attn_out_stu[i];
        }

        let post_name = if src.has_tensor(&format!("{pfx}.post_attention_norm.weight")) {
            format!("{pfx}.post_attention_norm.weight")
        } else {
            format!("{pfx}.ffn_norm.weight")
        };
        let fn_orig = fetch_vec(src, &post_name)?;
        let fn_ref = batch_rms_norm(&cur_ref, &fn_orig, eps, n_tok, dim);
        let fn_stu_w = align_rmsnorm_weights(&cur_stu, &fn_ref, &fn_orig, eps, n_tok, dim);
        let fn_stu = batch_rms_norm(&cur_stu, &fn_stu_w, eps, n_tok, dim);
        out_map.insert(post_name, PackedTensor::RawF32 { rows: 1, cols: dim, data: fn_stu_w });

        let hess_ffn = SuperBlockHessian::build(&fn_stu, Some(&fn_ref), n_tok, dim, dim_pad, eff_threads);
        let (fg_name, fu_name, fd_name) = (format!("{pfx}.ffn_gate.weight"), format!("{pfx}.ffn_up.weight"), format!("{pfx}.ffn_down.weight"));
        let (pt_g, mut g_s, mut g_r) = pack_proj_with_teacher(src, &fg_name, il, cfg, has_output_weight, &hess_ffn, &fn_ref, &fn_stu, n_tok, turbo_entropy_scale, eff_threads)?;
        let (pt_u, u_s, u_r) = pack_proj_with_teacher(src, &fu_name, il, cfg, has_output_weight, &hess_ffn, &fn_ref, &fn_stu, n_tok, turbo_entropy_scale, eff_threads)?;
        out_map.insert(fg_name, pt_g);
        out_map.insert(fu_name, pt_u);

        for i in 0..g_r.len() {
            g_r[i] = if cfg.ffn_gelu { gelu_tanh_f32(g_r[i]) } else { silu(g_r[i]) } * u_r[i];
            g_s[i] = if cfg.ffn_gelu { gelu_tanh_f32(g_s[i]) } else { silu(g_s[i]) } * u_s[i];
        }
        let hess_down = SuperBlockHessian::build(&g_s, Some(&g_r), n_tok, cfg.n_ff, pad_col(cfg.n_ff), eff_threads);
        let (pt_d, mut ffn_s, mut ffn_r) = pack_proj_with_teacher(src, &fd_name, il, cfg, has_output_weight, &hess_down, &g_r, &g_s, n_tok, turbo_entropy_scale, eff_threads)?;
        out_map.insert(fd_name, pt_d);

        if let Some(fpn) = src.extract_f32(&format!("{pfx}.ffn_post_norm.weight")) {
            for t in 0..n_tok {
                rms_norm_inplace(&mut ffn_r[t * dim..(t + 1) * dim], &fpn, eps);
                rms_norm_inplace(&mut ffn_s[t * dim..(t + 1) * dim], &fpn, eps);
            }
        }
        for i in 0..(n_tok * dim) {
            cur_ref[i] += ffn_r[i];
            cur_stu[i] += ffn_s[i];
        }
    }

    if let Some(cb) = progress {
        let msg = format!("Turbo v3 LM-Head KLD optimizasyonu... (cos={:.4})", hidden_cos(&cur_stu, &cur_ref));
        if !cb(cfg.n_layer + 1, total_steps, &msg) {
            return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
        }
    }
    let out_norm_orig = fetch_vec(src, "output_norm.weight")?;
    let fn_ref = batch_rms_norm(&cur_ref, &out_norm_orig, eps, n_tok, dim);
    let out_norm_init = align_rmsnorm_weights(&cur_stu, &fn_ref, &out_norm_orig, eps, n_tok, dim);
    let fn_stu = batch_rms_norm(&cur_stu, &out_norm_init, eps, n_tok, dim);

    let out_norm_final = if has_output_weight {
        let (out_rows, out_cols) = src
            .matrix_dims("output.weight")
            .ok_or_else(|| Error::Format("missing output.weight".into()))?;
        let hess_head = SuperBlockHessian::build(&fn_stu, Some(&fn_ref), n_tok, dim, dim_pad, eff_threads);
        let mut stats = TeacherHeadStats::new(&fn_ref, n_tok, dim, out_rows);
        let (bits_h, lam_h) = turbo_v3_tensor_spec("output.weight", None, cfg.n_layer, has_output_weight);
        let mut hook = |r0: usize, r1: usize, wc: &[f32]| stats.observe_chunk(r0, r1, wc, eff_threads);
        let (out_mat, _) = encode_turbo_full_gptq(
            bits_h, out_rows, out_cols, |r0, r1| fetch_rows(src, "output.weight", r0, r1),
            &hess_head, Some(&fn_ref), &fn_stu, n_tok, lam_h * turbo_entropy_scale, false, Some(&mut hook), eff_threads,
        )?;
        let opt = optimize_output_norm_kld(&cur_stu, &out_norm_orig, &out_norm_init, eps, &stats, &|vid, dst| out_mat.decode_row(vid, dst), dim, eff_threads);
        out_map.insert("output.weight".to_owned(), PackedTensor::TurboWht(out_mat));
        opt
    } else {
        let mut stats = TeacherHeadStats::new(&fn_ref, n_tok, dim, embd_rows);
        for r0 in (0..embd_rows).step_by(512) {
            let r1 = (r0 + 512).min(embd_rows);
            stats.observe_chunk(r0, r1, &fetch_rows(src, "token_embd.weight", r0, r1)?, eff_threads);
        }
        let emb_ref = out_map.get("token_embd.weight");
        optimize_output_norm_kld(
            &cur_stu, &out_norm_orig, &out_norm_init, eps, &stats,
            &|vid, dst| { if let Some(pt) = emb_ref { pt.decode_row(vid, dst); } },
            dim, eff_threads,
        )
    };
    out_map.insert("output_norm.weight".to_owned(), PackedTensor::RawF32 { rows: 1, cols: dim, data: out_norm_final });
    Ok(out_map)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use super::run_sequential_turbo_pack;
    use crate::cb4_calib::Cb4TensorSource;
    use crate::format::PackedTensor;
    use crate::gguf_meta::ModelConfig;
    use crate::tokenizer::BpeTokenizer;

    struct MockHybSource(HashMap<String, (Option<(usize, usize)>, Vec<f32>)>);
    impl Cb4TensorSource for MockHybSource {
        fn has_tensor(&self, name: &str) -> bool { self.0.contains_key(name) }
        fn writer_tensors(&self, _: usize) -> Vec<String> { Vec::new() }
        fn matrix_dims(&self, name: &str) -> Option<(usize, usize)> { self.0.get(name).and_then(|x| x.0) }
        fn extract_f32(&self, name: &str) -> Option<Vec<f32>> { self.0.get(name).map(|x| x.1.clone()) }
    }

    #[test]
    fn sequential_turbo_pack_runs_end_to_end_on_hybrid_model() {
        let (dim, ff, voc) = (256usize, 256usize, 256usize);
        let cfg = ModelConfig {
            arch: "qwen35".into(), model_name: "mock".into(), n_layer: 2, n_embd: dim, n_ff: ff,
            n_vocab: voc, n_head: 4, n_head_kv: 2, head_dim: 64, rms_norm_eps: 1e-6,
            rope_freq_base: 10000.0, rope_dim: 64, rope_neox: true, ffn_gelu: false, embed_scale: 1.0,
            full_attn_interval: 2, ssm_conv_kernel: 4, ssm_state_size: 16, ssm_group_count: 2,
            ssm_dt_rank: 4, ssm_inner_size: dim, eos_token_id: 0, bos_token_id: None,
        };
        let mut s = 0x9988_7766_5544_3322u64;
        let mut mk = |n: usize, sd: f32| -> Vec<f32> {
            (0..n).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((s >> 40) as f32) / 16777216.0 - 0.5) * sd
            }).collect()
        };
        let mut map = HashMap::new();
        map.insert("token_embd.weight".into(), (Some((voc, dim)), mk(voc * dim, 0.08)));
        map.insert("output.weight".into(), (Some((voc, dim)), mk(voc * dim, 0.08)));
        map.insert("output_norm.weight".into(), (None, vec![1.0; dim]));
        let conv_ch = dim + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
        for il in 0..2 {
            let p = format!("blk.{il}");
            map.insert(format!("{p}.attn_norm.weight"), (None, vec![1.0; dim]));
            map.insert(format!("{p}.post_attention_norm.weight"), (None, vec![1.0; dim]));
            map.insert(format!("{p}.ffn_gate.weight"), (Some((ff, dim)), mk(ff * dim, 0.06)));
            map.insert(format!("{p}.ffn_up.weight"), (Some((ff, dim)), mk(ff * dim, 0.06)));
            map.insert(format!("{p}.ffn_down.weight"), (Some((dim, ff)), mk(dim * ff, 0.06)));
            if cfg.is_recurrent_layer(il) {
                map.insert(format!("{p}.attn_qkv.weight"), (Some((conv_ch, dim)), mk(conv_ch * dim, 0.06)));
                map.insert(format!("{p}.attn_gate.weight"), (Some((dim, dim)), mk(dim * dim, 0.06)));
                map.insert(format!("{p}.ssm_alpha.weight"), (Some((cfg.ssm_dt_rank, dim)), mk(cfg.ssm_dt_rank * dim, 0.05)));
                map.insert(format!("{p}.ssm_beta.weight"), (Some((cfg.ssm_dt_rank, dim)), mk(cfg.ssm_dt_rank * dim, 0.05)));
                map.insert(format!("{p}.ssm_out.weight"), (Some((dim, dim)), mk(dim * dim, 0.06)));
                map.insert(format!("{p}.ssm_conv1d.weight"), (None, vec![0.25; conv_ch * cfg.ssm_conv_kernel]));
                map.insert(format!("{p}.ssm_a"), (None, vec![-0.5; cfg.ssm_dt_rank]));
                map.insert(format!("{p}.ssm_dt.bias"), (None, vec![0.1; cfg.ssm_dt_rank]));
                map.insert(format!("{p}.ssm_norm.weight"), (None, vec![1.0; dim / cfg.ssm_dt_rank]));
            } else {
                map.insert(format!("{p}.attn_q.weight"), (Some((dim, dim)), mk(dim * dim, 0.06)));
                map.insert(format!("{p}.attn_k.weight"), (Some((128, dim)), mk(128 * dim, 0.06)));
                map.insert(format!("{p}.attn_v.weight"), (Some((128, dim)), mk(128 * dim, 0.06)));
                map.insert(format!("{p}.attn_output.weight"), (Some((dim, dim)), mk(dim * dim, 0.06)));
            }
        }
        let mut vocab = vec!["<unk>".into(), " ".into()];
        for b in 33u8..126 { vocab.push((b as char).to_string()); }
        while vocab.len() < voc { vocab.push(format!("t{}", vocab.len())); }
        let tok = BpeTokenizer::from_vocab_and_merges(vocab, &[], None).unwrap();
        let res = run_sequential_turbo_pack(&MockHybSource(map), &cfg, &tok, None, 1.0, 2, None).unwrap();
        assert!(res.contains_key("output.weight"));
        assert!(res.contains_key("blk.0.ssm_alpha.weight"));
        assert!(res.contains_key("blk.1.attn_q.weight"));
        for pt in res.values() {
            if let PackedTensor::TurboWht(m) = pt {
                if matches!(m.bits, 4 | 5) {
                    assert!(m.scales_f16.iter().all(|&v| (v & 0x003f) == 0));
                }
            }
        }
    }
}
