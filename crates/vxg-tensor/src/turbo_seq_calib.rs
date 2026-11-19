//! Clean-Teacher `VXT-Turbo v4` (`bits = 7` `VXT-Turbo-E8`) sequential calibration:
//! - Runs a single unmutated Teacher (`BF16/FP16`) forward pass over `4,096` disjoint multi-domain
//!   tokens (`16 x 256` with zero self-repetition and zero evaluation n-gram overlap).
//! - Measures exact sub-layer residual write sensitivity `rho_l = ||delta_h_l||_F / ||h_l + delta_h_l||_F`
//!   to allocate Gosset `E8` lattice density `beta` dynamically across depths and roles.
//! - Preserves all `RMSNorm` weights (`RawF32`) and recurrent `ssm_alpha`/`ssm_beta` gates (`Lossless`)
//!   100% unmodified while encoding trunk/head projections via `256`-col `SuperBlockHessian` `LDL-GPTQ`.

use std::collections::HashMap;

use crate::cb4_calib::Cb4TensorSource;
use crate::format::PackedTensor;
use crate::gguf_meta::ModelConfig;
use crate::llm_calib::turbo_v4_tensor_beta;
use crate::llm_ops::{rms_norm, rms_norm_inplace, silu};
use crate::llm_prefill::gelu_tanh_f32;
use crate::tokenizer::BpeTokenizer;
use crate::turbo_calib_corpus::build_calib_sequences;
use crate::turbo_calib_ops::{run_deltanet_calib_fast, run_gqa_calib_fast};
use crate::turbo_e8::encode_e8_superblock_gptq;
use crate::turbo_e8_hq::encode_e8_hq_superblock_gptq;
use crate::turbo_linalg::{effective_cpu_threads, sgemm_nt, SuperBlockHessian};
use crate::wht_turbo::TurboWhtMatrix;
use crate::{Error, Result};

fn pad_col(c: usize) -> usize {
    c.div_ceil(256) * 256
}

fn fetch_vec(src: &dyn Cb4TensorSource, name: &str) -> Result<Vec<f32>> {
    src.extract_f32(name)
        .ok_or_else(|| Error::Format(format!("missing tensor '{name}' during turbo v4 calibration")))
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

fn vec_rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let s2: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (s2 / (x.len() as f64)).sqrt() as f32
}

fn vec_rms_sum(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() {
        return 0.0;
    }
    let s2: f64 = a.iter().zip(b).map(|(&x, &y)| ((x + y) as f64).powi(2)).sum();
    (s2 / (a.len() as f64)).sqrt() as f32
}

fn stream_proj_f32(
    src: &dyn Cb4TensorSource,
    name: &str,
    x_in: &[f32],
    n_tok: usize,
    eff_threads: usize,
) -> Result<(usize, Vec<f32>)> {
    let (rows, cols) = src
        .matrix_dims(name)
        .ok_or_else(|| Error::Format(format!("missing projection '{name}'")))?;
    let chunk_rows = if cols > 4096 { 256usize } else { 512usize };
    let mut y_out = vec![0.0f32; n_tok * rows];
    let mut tmp = Vec::new();
    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + chunk_rows).min(rows);
        let rc = r1 - r0;
        let w_chunk = fetch_rows(src, name, r0, r1)?;
        tmp.resize(n_tok * rc, 0.0);
        sgemm_nt(n_tok, rc, cols, 1.0, x_in, cols, &w_chunk, cols, 0.0, &mut tmp, rc, eff_threads);
        for t in 0..n_tok {
            y_out[t * rows + r0..t * rows + r1].copy_from_slice(&tmp[t * rc..(t + 1) * rc]);
        }
        r0 = r1;
    }
    Ok((rows, y_out))
}

const HQ_BETA_MUL: f32 = 6.20;

fn pack_e8_proj(
    src: &dyn Cb4TensorSource,
    name: &str,
    hess: &SuperBlockHessian,
    beta: f32,
    hq: bool,
    eff_threads: usize,
) -> Result<PackedTensor> {
    let (rows, cols) = src
        .matrix_dims(name)
        .ok_or_else(|| Error::Format(format!("missing projection '{name}'")))?;
    let mat = if hq {
        encode_e8_hq_superblock_gptq(
            rows, cols, |r0, r1| fetch_rows(src, name, r0, r1), hess, beta * HQ_BETA_MUL, None, eff_threads,
        )?
    } else {
        encode_e8_superblock_gptq(
            rows, cols, |r0, r1| fetch_rows(src, name, r0, r1), hess, beta, None, eff_threads,
        )?
    };
    Ok(PackedTensor::TurboWht(mat))
}

fn build_embd_ch_amp(src: &dyn Cb4TensorSource, dim: usize) -> Vec<f32> {
    let a0 = src.extract_f32("blk.0.attn_norm.weight").filter(|v| v.len() == dim);
    let on = src.extract_f32("output_norm.weight").filter(|v| v.len() == dim);
    let mut raw: Vec<f32> = (0..dim)
        .map(|c| {
            let va = a0.as_ref().map_or(1.0, |v| v[c].abs());
            let vo = on.as_ref().map_or(1.0, |v| v[c].abs());
            (0.75 * va + 0.25 * vo).max(1e-4)
        })
        .collect();
    let rms0 = (raw.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / dim.max(1) as f64)
        .sqrt()
        .max(1e-12);
    for v in &mut raw {
        *v = ((*v as f64 / rms0) as f32).clamp(0.25, 4.0);
    }
    let rms1 = (raw.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / dim.max(1) as f64)
        .sqrt()
        .max(1e-12);
    let inv = (1.0 / rms1) as f32;
    for v in &mut raw {
        *v *= inv;
    }
    raw
}

fn encode_untied_embd_e8(
    src: &dyn Cb4TensorSource,
    rows: usize,
    cols: usize,
    base_beta: f32,
    ch_amp: &[f32],
    hq: bool,
    eff_threads: usize,
) -> Result<TurboWhtMatrix> {
    const CHUNK: usize = 4096;
    let (padded_rows, padded_cols) = (rows.div_ceil(4) * 4, pad_col(cols));
    let steps = padded_cols / 256;
    let bits = if hq { 8u8 } else { 7u8 };
    let wps = TurboWhtMatrix::words_per_tile_step(bits);
    let eff_beta = if hq { (base_beta * HQ_BETA_MUL).clamp(1.20, 48.0) } else { base_beta.clamp(0.90, 3.40) };
    let mut scales_f16 = Vec::with_capacity((padded_rows / 4) * steps * 32);
    let mut packed_words = Vec::with_capacity((padded_rows / 4) * steps * wps);
    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + CHUNK).min(rows);
        let rc = r1 - r0;
        let chunk = fetch_rows(src, "token_embd.weight", r0, r1)?;
        let mat = TurboWhtMatrix::encode_with_ch_amp_threads(
            bits, rc, cols, &chunk, None, Some(ch_amp), |_| eff_beta, eff_threads,
        )?;
        scales_f16.extend_from_slice(&mat.scales_f16);
        packed_words.extend_from_slice(&mat.packed_words);
        r0 = r1;
    }
    Ok(TurboWhtMatrix { bits, rows, cols, padded_rows, padded_cols, disk_bytes: 0, scales_f16, packed_words })
}

fn add_bias_rows(buf: &mut [f32], bias: Option<&[f32]>, n_tok: usize, cols: usize) {
    if let Some(b) = bias {
        for t in 0..n_tok {
            let row = &mut buf[t * cols..(t + 1) * cols];
            for c in 0..cols.min(b.len()) {
                row[c] += b[c];
            }
        }
    }
}

/// Runs Clean-Teacher `VXT-Turbo v5` (`bits = 7` `E8`) / `VXT-Turbo-HQ` (`bits = 8` `E8-HQ`) calibration
/// and collects all packed tensors into a `HashMap` (for in-memory callers and tests).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sequential_turbo_pack(
    src: &dyn Cb4TensorSource,
    cfg: &ModelConfig,
    tok: &BpeTokenizer,
    calib_text: Option<&str>,
    turbo_entropy_scale: f32,
    hq: bool,
    threads: usize,
    progress: Option<&(dyn Fn(usize, usize, &str) -> bool + Sync)>,
) -> Result<HashMap<String, PackedTensor>> {
    let mut out_map = HashMap::new();
    run_sequential_turbo_pack_streaming(
        src,
        cfg,
        tok,
        calib_text,
        turbo_entropy_scale,
        hq,
        threads,
        progress,
        &mut |batch| {
            for (k, v) in batch {
                out_map.insert(k, v);
            }
            Ok(())
        },
    )?;
    Ok(out_map)
}

/// Streaming variant of [`run_sequential_turbo_pack`]: invokes `on_batch` after `token_embd`, after
/// each transformer layer's attention & FFN projections, and after `output.weight`, allowing callers
/// to flush compressed sections directly to disk and keep peak RAM bounded even for `27B`/`70B+` models.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_sequential_turbo_pack_streaming(
    src: &dyn Cb4TensorSource,
    cfg: &ModelConfig,
    tok: &BpeTokenizer,
    calib_text: Option<&str>,
    turbo_entropy_scale: f32,
    hq: bool,
    threads: usize,
    progress: Option<&(dyn Fn(usize, usize, &str) -> bool + Sync)>,
    on_batch: &mut dyn FnMut(Vec<(String, PackedTensor)>) -> Result<()>,
) -> Result<()> {
    let eff_threads = effective_cpu_threads(threads);
    let (req_len, req_seqs) = if cfg.n_vocab >= 1024 { (256usize, 16usize) } else { (64usize, 4usize) };
    let seqs = build_calib_sequences(tok, calib_text, req_len, req_seqs);
    if seqs.is_empty() {
        return Err(Error::Format("sequential turbo calibration yielded zero sequences".into()));
    }
    let (n_seq, seq_len) = (seqs.len(), seqs[0].len());
    let (n_tok, dim, eps, dim_pad) = (n_seq * seq_len, cfg.n_embd, cfg.rms_norm_eps, pad_col(cfg.n_embd));
    let has_output_weight = src.has_tensor("output.weight");
    let ent_scale = turbo_entropy_scale.clamp(0.15, 3.00);
    let total_steps = cfg.n_layer + 2;
    let label = if hq { "Turbo-HQ E8" } else { "Turbo v5 E8" };

    if let Some(cb) = progress {
        if !cb(0, total_steps, &format!("{label} kalibrasyonu başlatılıyor...")) {
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
            for c in 0..dim {
                dst[c] = row[c] * cfg.embed_scale;
            }
        }
    }
    drop(row_cache);

    if has_output_weight {
        let emb_beta = turbo_v4_tensor_beta("token_embd.weight", None, cfg.n_layer, false, true) / ent_scale;
        let ch_amp_embd = build_embd_ch_amp(src, dim);
        let emb_mat = encode_untied_embd_e8(src, embd_rows, dim, emb_beta, &ch_amp_embd, hq, eff_threads)?;
        on_batch(vec![("token_embd.weight".to_owned(), PackedTensor::TurboWht(emb_mat))])?;
    }

    let (mut ema_attn, mut ema_ffn) = (0.0f32, 0.0f32);

    for il in 0..cfg.n_layer {
        let is_rec = cfg.is_recurrent_layer(il);
        if let Some(cb) = progress {
            let msg = format!("{label} katman kalibrasyonu {}/{} (rms={:.3})", il + 1, cfg.n_layer, vec_rms(&cur_ref));
            if !cb(il + 1, total_steps, &msg) {
                return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
            }
        }
        let pfx = format!("blk.{il}");
        let an_orig = fetch_vec(src, &format!("{pfx}.attn_norm.weight"))?;
        let normed_ref = batch_rms_norm(&cur_ref, &an_orig, eps, n_tok, dim);
        let hess_attn = SuperBlockHessian::build(&normed_ref, None, n_tok, dim, dim_pad, eff_threads);

        let attn_out_ref = if is_rec {
            let (qkv_name, gate_name) = (format!("{pfx}.attn_qkv.weight"), format!("{pfx}.attn_gate.weight"));
            let sout_name = format!("{pfx}.ssm_out.weight");
            let (_, qkv_r) = stream_proj_f32(src, &qkv_name, &normed_ref, n_tok, eff_threads)?;
            let (_, z_r) = stream_proj_f32(src, &gate_name, &normed_ref, n_tok, eff_threads)?;
            let (_, a_r) = stream_proj_f32(src, &format!("{pfx}.ssm_alpha.weight"), &normed_ref, n_tok, eff_threads)?;
            let (_, b_r) = stream_proj_f32(src, &format!("{pfx}.ssm_beta.weight"), &normed_ref, n_tok, eff_threads)?;
            let conv1d = fetch_vec(src, &format!("{pfx}.ssm_conv1d.weight"))?;
            let ssm_a = fetch_vec(src, &format!("{pfx}.ssm_a"))?;
            let dt_bias = fetch_vec(src, &format!("{pfx}.ssm_dt.bias"))?;
            let ssm_norm = fetch_vec(src, &format!("{pfx}.ssm_norm.weight"))?;
            let y_r = run_deltanet_calib_fast(cfg, n_seq, seq_len, &qkv_r, &z_r, &a_r, &b_r, &conv1d, &ssm_a, &dt_bias, &ssm_norm, eff_threads);
            let (_, mut or_) = stream_proj_f32(src, &sout_name, &y_r, n_tok, eff_threads)?;
            if let Some(apn) = src.extract_f32(&format!("{pfx}.attn_post_norm.weight")) {
                for t in 0..n_tok {
                    rms_norm_inplace(&mut or_[t * dim..(t + 1) * dim], &apn, eps);
                }
            }
            let rho = vec_rms(&or_) / vec_rms_sum(&cur_ref, &or_).max(1e-12);
            let gain = if il == 0 { 1.0 } else { (rho / ema_attn.max(1e-6)).powf(0.35).clamp(0.90, 1.22) };
            ema_attn = if il == 0 { rho } else { 0.72 * ema_attn + 0.28 * rho };

            let b_qkv = turbo_v4_tensor_beta(&qkv_name, Some(il), cfg.n_layer, true, has_output_weight) * gain / ent_scale;
            let b_gate = turbo_v4_tensor_beta(&gate_name, Some(il), cfg.n_layer, true, has_output_weight) * gain / ent_scale;
            let b_out = turbo_v4_tensor_beta(&sout_name, Some(il), cfg.n_layer, true, has_output_weight) * gain / ent_scale;
            let mut batch = Vec::with_capacity(3);
            batch.push((qkv_name.clone(), pack_e8_proj(src, &qkv_name, &hess_attn, b_qkv, hq, eff_threads)?));
            batch.push((gate_name.clone(), pack_e8_proj(src, &gate_name, &hess_attn, b_gate, hq, eff_threads)?));
            let d_in = cfg.ssm_inner_size;
            let hess_o = SuperBlockHessian::build(&y_r, None, n_tok, d_in, pad_col(d_in), eff_threads);
            batch.push((sout_name.clone(), pack_e8_proj(src, &sout_name, &hess_o, b_out, hq, eff_threads)?));
            on_batch(batch)?;
            or_
        } else {
            let (qn, kn, vn) = (format!("{pfx}.attn_q.weight"), format!("{pfx}.attn_k.weight"), format!("{pfx}.attn_v.weight"));
            let oname = format!("{pfx}.attn_output.weight");
            let (q_rows, mut qr) = stream_proj_f32(src, &qn, &normed_ref, n_tok, eff_threads)?;
            let (_, mut kr) = stream_proj_f32(src, &kn, &normed_ref, n_tok, eff_threads)?;
            let (_, mut vr) = stream_proj_f32(src, &vn, &normed_ref, n_tok, eff_threads)?;
            let kv_dim = cfg.n_head_kv * cfg.head_dim;
            add_bias_rows(&mut qr, src.extract_f32(&format!("{pfx}.attn_q.bias")).as_deref(), n_tok, q_rows);
            add_bias_rows(&mut kr, src.extract_f32(&format!("{pfx}.attn_k.bias")).as_deref(), n_tok, kv_dim);
            add_bias_rows(&mut vr, src.extract_f32(&format!("{pfx}.attn_v.bias")).as_deref(), n_tok, kv_dim);
            let q_norm = src.extract_f32(&format!("{pfx}.attn_q_norm.weight"));
            let k_norm = src.extract_f32(&format!("{pfx}.attn_k_norm.weight"));
            let gated_q = q_rows == cfg.n_head * cfg.head_dim * 2;
            let ar = run_gqa_calib_fast(cfg, n_seq, seq_len, gated_q, &qr, &kr, &vr, q_norm.as_deref(), k_norm.as_deref(), eff_threads);
            let (_, mut or_) = stream_proj_f32(src, &oname, &ar, n_tok, eff_threads)?;
            if let Some(apn) = src.extract_f32(&format!("{pfx}.attn_post_norm.weight")) {
                for t in 0..n_tok {
                    rms_norm_inplace(&mut or_[t * dim..(t + 1) * dim], &apn, eps);
                }
            }
            let rho = vec_rms(&or_) / vec_rms_sum(&cur_ref, &or_).max(1e-12);
            let gain = if il == 0 { 1.0 } else { (rho / ema_attn.max(1e-6)).powf(0.35).clamp(0.90, 1.22) };
            ema_attn = if il == 0 { rho } else { 0.72 * ema_attn + 0.28 * rho };

            let b_q = turbo_v4_tensor_beta(&qn, Some(il), cfg.n_layer, false, has_output_weight) * gain / ent_scale;
            let b_k = turbo_v4_tensor_beta(&kn, Some(il), cfg.n_layer, false, has_output_weight) * gain / ent_scale;
            let b_v = turbo_v4_tensor_beta(&vn, Some(il), cfg.n_layer, false, has_output_weight) * gain / ent_scale;
            let b_o = turbo_v4_tensor_beta(&oname, Some(il), cfg.n_layer, false, has_output_weight) * gain / ent_scale;
            let mut batch = Vec::with_capacity(4);
            batch.push((qn.clone(), pack_e8_proj(src, &qn, &hess_attn, b_q, hq, eff_threads)?));
            batch.push((kn.clone(), pack_e8_proj(src, &kn, &hess_attn, b_k, hq, eff_threads)?));
            batch.push((vn.clone(), pack_e8_proj(src, &vn, &hess_attn, b_v, hq, eff_threads)?));
            let att_dim = cfg.n_head * cfg.head_dim;
            let hess_o = SuperBlockHessian::build(&ar, None, n_tok, att_dim, pad_col(att_dim), eff_threads);
            batch.push((oname.clone(), pack_e8_proj(src, &oname, &hess_o, b_o, hq, eff_threads)?));
            on_batch(batch)?;
            or_
        };

        for i in 0..(n_tok * dim) {
            cur_ref[i] += attn_out_ref[i];
        }

        let post_name = if src.has_tensor(&format!("{pfx}.post_attention_norm.weight")) {
            format!("{pfx}.post_attention_norm.weight")
        } else {
            format!("{pfx}.ffn_norm.weight")
        };
        let fn_orig = fetch_vec(src, &post_name)?;
        let fn_ref = batch_rms_norm(&cur_ref, &fn_orig, eps, n_tok, dim);
        let hess_ffn = SuperBlockHessian::build(&fn_ref, None, n_tok, dim, dim_pad, eff_threads);

        let (fg_name, fu_name, fd_name) = (
            format!("{pfx}.ffn_gate.weight"),
            format!("{pfx}.ffn_up.weight"),
            format!("{pfx}.ffn_down.weight"),
        );
        let (_, mut g_r) = stream_proj_f32(src, &fg_name, &fn_ref, n_tok, eff_threads)?;
        let (_, u_r) = stream_proj_f32(src, &fu_name, &fn_ref, n_tok, eff_threads)?;
        for i in 0..g_r.len() {
            g_r[i] = if cfg.ffn_gelu { gelu_tanh_f32(g_r[i]) } else { silu(g_r[i]) } * u_r[i];
        }
        let (_, mut ffn_r) = stream_proj_f32(src, &fd_name, &g_r, n_tok, eff_threads)?;
        if let Some(fpn) = src.extract_f32(&format!("{pfx}.ffn_post_norm.weight")) {
            for t in 0..n_tok {
                rms_norm_inplace(&mut ffn_r[t * dim..(t + 1) * dim], &fpn, eps);
            }
        }
        let rho_f = vec_rms(&ffn_r) / vec_rms_sum(&cur_ref, &ffn_r).max(1e-12);
        let gain_f = if il == 0 { 1.0 } else { (rho_f / ema_ffn.max(1e-6)).powf(0.35).clamp(0.90, 1.22) };
        ema_ffn = if il == 0 { rho_f } else { 0.72 * ema_ffn + 0.28 * rho_f };

        let b_g = turbo_v4_tensor_beta(&fg_name, Some(il), cfg.n_layer, is_rec, has_output_weight) * gain_f / ent_scale;
        let b_u = turbo_v4_tensor_beta(&fu_name, Some(il), cfg.n_layer, is_rec, has_output_weight) * gain_f / ent_scale;
        let b_d = turbo_v4_tensor_beta(&fd_name, Some(il), cfg.n_layer, is_rec, has_output_weight) * gain_f / ent_scale;
        let mut ffn_batch = Vec::with_capacity(3);
        ffn_batch.push((fg_name.clone(), pack_e8_proj(src, &fg_name, &hess_ffn, b_g, hq, eff_threads)?));
        ffn_batch.push((fu_name.clone(), pack_e8_proj(src, &fu_name, &hess_ffn, b_u, hq, eff_threads)?));
        let hess_down = SuperBlockHessian::build(&g_r, None, n_tok, cfg.n_ff, pad_col(cfg.n_ff), eff_threads);
        ffn_batch.push((fd_name.clone(), pack_e8_proj(src, &fd_name, &hess_down, b_d, hq, eff_threads)?));
        on_batch(ffn_batch)?;

        for i in 0..(n_tok * dim) {
            cur_ref[i] += ffn_r[i];
        }
    }

    if let Some(cb) = progress {
        let msg = format!("{label} LM-Head kodlanıyor... (rms={:.3})", vec_rms(&cur_ref));
        if !cb(cfg.n_layer + 1, total_steps, &msg) {
            return Err(Error::Io("işlem kullanıcı tarafından iptal edildi".into()));
        }
    }
    let out_norm_orig = fetch_vec(src, "output_norm.weight")?;
    let fn_ref = batch_rms_norm(&cur_ref, &out_norm_orig, eps, n_tok, dim);
    let hess_head = SuperBlockHessian::build(&fn_ref, None, n_tok, dim, dim_pad, eff_threads);

    if has_output_weight {
        let b_h = turbo_v4_tensor_beta("output.weight", None, cfg.n_layer, false, true) / ent_scale;
        on_batch(vec![("output.weight".to_owned(), pack_e8_proj(src, "output.weight", &hess_head, b_h, hq, eff_threads)?)])?;
    } else {
        let b_e = turbo_v4_tensor_beta("token_embd.weight", None, cfg.n_layer, false, false) / ent_scale;
        on_batch(vec![("token_embd.weight".to_owned(), pack_e8_proj(src, "token_embd.weight", &hess_head, b_e, hq, eff_threads)?)])?;
    }
    Ok(())
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
        let res = run_sequential_turbo_pack(&MockHybSource(map), &cfg, &tok, None, 1.0, false, 2, None).unwrap();
        assert!(res.contains_key("token_embd.weight"));
        assert!(res.contains_key("output.weight"));
        assert!(!res.contains_key("blk.0.ssm_alpha.weight"));
        assert!(res.contains_key("blk.1.attn_q.weight"));
        for pt in res.values() {
            let PackedTensor::TurboWht(m) = pt else { panic!("expected TurboWht") };
            assert_eq!(m.bits, 7);
            assert!(m.scales_f16.iter().all(|&v| (v & 0x003f) == 0));
        }
    }
}
