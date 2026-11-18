//! **Hamle 3** output/norm calibration: closed-form RMSNorm ridge alignment, `ssm_alpha`/`ssm_beta`
//! CG ridge alignment (`F16` `Lossless`), and streaming Top-32 + stratified tail KLD Adam optimizer
//! for `output_norm.weight`.

use std::collections::{BTreeSet, HashMap};

use crate::format::PackedTensor;
use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::lossless::LosslessMatrix;
use crate::turbo_linalg::sgemm_nt;
use crate::{DType, Result};

/// Closed-form moment-matching channelwise alignment for RMSNorm weights (`[0.85, 1.18] * w_orig[c]`).
pub(crate) fn align_rmsnorm_weights(
    x_stu: &[f32], y_norm_ref: &[f32], w_orig: &[f32], eps: f32, n_tok: usize, dim: usize,
) -> Vec<f32> {
    if n_tok == 0 || dim == 0 || x_stu.len() != n_tok * dim || y_norm_ref.len() != n_tok * dim || w_orig.len() != dim {
        return w_orig.to_vec();
    }
    let (mut sum_u2, mut sum_y2, mut sum_uy) = (vec![0.0f64; dim], vec![0.0f64; dim], vec![0.0f64; dim]);
    let inv_dim = 1.0 / (dim as f32);
    for t in 0..n_tok {
        let (xr, yr) = (&x_stu[t * dim..(t + 1) * dim], &y_norm_ref[t * dim..(t + 1) * dim]);
        let sq: f32 = xr.iter().map(|&v| v * v).sum();
        let scale = 1.0 / (sq * inv_dim + eps).sqrt();
        for c in 0..dim {
            let (u, y) = ((xr[c] * scale) as f64, yr[c] as f64);
            sum_u2[c] += u * u;
            sum_y2[c] += y * y;
            sum_uy[c] += u * y;
        }
    }
    (0..dim)
        .map(|c| {
            let w0 = w_orig[c];
            let wf = w0 as f64;
            if !w0.is_finite() || w0.abs() <= 1e-6 || sum_u2[c] <= 1e-12 || sum_uy[c] * wf <= 0.0 {
                return w0;
            }
            let g_var = wf.signum() * (sum_y2[c] / sum_u2[c]).sqrt();
            let g = 0.70 * g_var + 0.30 * wf;
            ((g / wf).clamp(0.85, 1.18) * wf) as f32
        })
        .collect()
}

/// Winsorized, held-out-validated 2-step CG ridge alignment for tiny `ssm_alpha`/`ssm_beta` (`Lossless(F16)`).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn align_tiny_proj_f16(
    w_orig: &[f32], rows: usize, cols: usize, x_stu: &[f32], x_ref: &[f32], n_tok: usize, threads: usize,
) -> Result<(PackedTensor, Vec<f32>, Vec<f32>)> {
    let mut y_ref = vec![0.0f32; n_tok * rows];
    sgemm_nt(n_tok, rows, cols, 1.0, x_ref, cols, w_orig, cols, 0.0, &mut y_ref, rows, threads);

    let words_orig: Vec<u16> = w_orig.iter().map(|&v| f32_to_f16(v)).collect();
    let w_orig_f: Vec<f32> = words_orig.iter().map(|&u| f16_to_f32(u)).collect();
    let mut y_orig_f16 = vec![0.0f32; n_tok * rows];
    sgemm_nt(n_tok, rows, cols, 1.0, x_stu, cols, &w_orig_f, cols, 0.0, &mut y_orig_f16, rows, threads);

    let tr_idx: Vec<usize> = (0..n_tok).filter(|&t| t % 4 != 3).collect();
    let n_tr = tr_idx.len();
    let n_val = n_tok - n_tr;
    if n_val == 0 || n_tr == 0 {
        return Ok((PackedTensor::Lossless(LosslessMatrix::encode(DType::F16, rows, cols, &words_orig)?), y_orig_f16, y_ref));
    }

    let mut y_init = vec![0.0f32; n_tok * rows];
    sgemm_nt(n_tok, rows, cols, 1.0, x_stu, cols, w_orig, cols, 0.0, &mut y_init, rows, threads);

    let inv_c = 1.0 / (cols.max(1) as f64);
    let rms_tr: Vec<f64> = tr_idx.iter().map(|&t| {
        let s2: f64 = x_stu[t * cols..(t + 1) * cols].iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s2 * inv_c).sqrt()
    }).collect();
    let mut sorted = rms_tr.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let cap = 4.0 * sorted.get(n_tr / 2).copied().unwrap_or(1.0).max(1e-12);

    let (mut x_tr, mut x_tr_t, mut e_tr_t) = (vec![0.0f32; n_tr * cols], vec![0.0f32; cols * n_tr], vec![0.0f32; rows * n_tr]);
    let mut mean_x2 = 0.0f64;
    for (ti, &t) in tr_idx.iter().enumerate() {
        let tw = if rms_tr[ti] > cap { (cap / rms_tr[ti]) as f32 } else { 1.0 };
        for c in 0..cols {
            let v = x_stu[t * cols + c] * tw;
            x_tr[ti * cols + c] = v;
            x_tr_t[c * n_tr + ti] = v;
            mean_x2 += (v as f64) * (v as f64);
        }
        for r in 0..rows {
            e_tr_t[r * n_tr + ti] = (y_ref[t * rows + r] - y_init[t * rows + r]) * tw;
        }
    }
    mean_x2 /= (n_tr * cols.max(1)) as f64;
    let inv_n = 1.0 / (n_tr as f32);
    let ridge = (0.50 * mean_x2).max(1e-8) as f32;

    let mut r_cg = vec![0.0f32; rows * cols];
    sgemm_nt(rows, cols, n_tr, inv_n, &e_tr_t, n_tr, &x_tr_t, n_tr, 0.0, &mut r_cg, cols, threads);
    let (mut p_cg, mut dw) = (r_cg.clone(), vec![0.0f32; rows * cols]);
    let (mut px, mut ap) = (vec![0.0f32; rows * n_tr], vec![0.0f32; rows * cols]);
    let mut rz_old: Vec<f64> = (0..rows)
        .map(|r| r_cg[r * cols..(r + 1) * cols].iter().map(|&v| (v as f64) * (v as f64)).sum())
        .collect();

    for _ in 0..2 {
        sgemm_nt(rows, n_tr, cols, 1.0, &p_cg, cols, &x_tr, cols, 0.0, &mut px, n_tr, threads);
        sgemm_nt(rows, cols, n_tr, inv_n, &px, n_tr, &x_tr_t, n_tr, 0.0, &mut ap, cols, threads);
        for i in 0..(rows * cols) { ap[i] += ridge * p_cg[i]; }
        for r in 0..rows {
            let off = r * cols;
            let p_ap: f64 = (0..cols).map(|c| (p_cg[off + c] as f64) * (ap[off + c] as f64)).sum();
            if p_ap <= 1e-24 || rz_old[r] <= 1e-24 { continue; }
            let alpha = (rz_old[r] / p_ap) as f32;
            let mut rz_new = 0.0f64;
            for c in 0..cols {
                dw[off + c] += alpha * p_cg[off + c];
                r_cg[off + c] -= alpha * ap[off + c];
                let rv = r_cg[off + c] as f64;
                rz_new += rv * rv;
            }
            let beta = (rz_new / rz_old[r]) as f32;
            for c in 0..cols { p_cg[off + c] = r_cg[off + c] + beta * p_cg[off + c]; }
            rz_old[r] = rz_new;
        }
    }

    let mut words_cand = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let off = r * cols;
        let wn = w_orig[off..off + cols].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let dn = dw[off..off + cols].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
        let sc = if dn > 0.08 * wn && dn > 1e-12 { (0.08 * wn / dn) as f32 } else { 1.0 };
        for c in 0..cols { words_cand.push(f32_to_f16(w_orig[off + c] + sc * dw[off + c])); }
    }
    let w_cand_f: Vec<f32> = words_cand.iter().map(|&u| f16_to_f32(u)).collect();
    let mut y_cand = vec![0.0f32; n_tok * rows];
    sgemm_nt(n_tok, rows, cols, 1.0, x_stu, cols, &w_cand_f, cols, 0.0, &mut y_cand, rows, threads);

    let (mut use_words, mut y_stu) = (words_orig, y_orig_f16);
    for r in 0..rows {
        let (mut vc, mut vo) = (0.0f64, 0.0f64);
        for t in (0..n_tok).filter(|&t| t % 4 == 3) {
            let yr = y_ref[t * rows + r] as f64;
            vc += (yr - y_cand[t * rows + r] as f64).powi(2);
            vo += (yr - y_stu[t * rows + r] as f64).powi(2);
        }
        if vc < vo {
            use_words[r * cols..(r + 1) * cols].copy_from_slice(&words_cand[r * cols..(r + 1) * cols]);
            for t in 0..n_tok { y_stu[t * rows + r] = y_cand[t * rows + r]; }
        }
    }
    Ok((PackedTensor::Lossless(LosslessMatrix::encode(DType::F16, rows, cols, &use_words)?), y_stu, y_ref))
}

const TOP_K: usize = 32;

pub(crate) struct TeacherHeadStats {
    pub(crate) t_idx: Vec<usize>,
    u_ref_sub: Vec<f32>,
    n_vocab: usize,
    dim: usize,
    max_l: Vec<f32>,
    sum_exp: Vec<f64>,
    top: Vec<Vec<(f32, u32)>>,
}

impl TeacherHeadStats {
    pub(crate) fn new(final_norm_ref: &[f32], n_tok: usize, dim: usize, n_vocab: usize) -> Self {
        let m = 512.min(n_tok.max(1));
        let t_idx: Vec<usize> = (0..m).map(|i| i * n_tok / m).collect();
        let mut u_ref_sub = Vec::with_capacity(m * dim);
        for &t in &t_idx {
            u_ref_sub.extend_from_slice(&final_norm_ref[t * dim..(t + 1) * dim]);
        }
        Self {
            t_idx,
            u_ref_sub,
            n_vocab,
            dim,
            max_l: vec![f32::NEG_INFINITY; m],
            sum_exp: vec![0.0; m],
            top: vec![Vec::with_capacity(TOP_K + 1); m],
        }
    }

    pub(crate) fn observe_chunk(&mut self, r0: usize, r1: usize, w_chunk: &[f32], threads: usize) {
        let (m, rc) = (self.t_idx.len(), r1 - r0);
        if m == 0 || rc == 0 {
            return;
        }
        let mut logits = vec![0.0f32; m * rc];
        sgemm_nt(m, rc, self.dim, 1.0, &self.u_ref_sub, self.dim, w_chunk, self.dim, 0.0, &mut logits, rc, threads);
        for i in 0..m {
            let row = &logits[i * rc..(i + 1) * rc];
            let cmax = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let new_max = self.max_l[i].max(cmax);
            let mut se = if self.sum_exp[i] > 0.0 {
                self.sum_exp[i] * ((self.max_l[i] - new_max) as f64).exp()
            } else {
                0.0
            };
            let tp = &mut self.top[i];
            let mut thresh = if tp.len() >= TOP_K { tp[TOP_K - 1].0 } else { f32::NEG_INFINITY };
            for (dr, &l) in row.iter().enumerate() {
                se += ((l - new_max) as f64).exp();
                if l > thresh {
                    let vid = (r0 + dr) as u32;
                    let pos = tp.partition_point(|x| x.0 > l);
                    tp.insert(pos, (l, vid));
                    if tp.len() > TOP_K {
                        tp.pop();
                        thresh = tp[TOP_K - 1].0;
                    }
                }
            }
            self.max_l[i] = new_max;
            self.sum_exp[i] = se;
        }
    }
}

/// Runs `15` Adam steps minimizing sampled full-vocab Teacher KLD over `output_norm.weight`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn optimize_output_norm_kld(
    cur_stu: &[f32],
    out_norm_orig: &[f32],
    out_norm_init: &[f32],
    eps: f32,
    stats: &TeacherHeadStats,
    decode_row: &(dyn Fn(usize, &mut [f32]) + Sync),
    dim: usize,
    threads: usize,
) -> Vec<f32> {
    let m = stats.t_idx.len();
    if m == 0 || stats.n_vocab <= TOP_K || dim == 0 {
        return out_norm_init.to_vec();
    }
    let mut tail_set: BTreeSet<u32> = BTreeSet::new();
    let tail_cap = 2048usize.min(stats.n_vocab);
    for k in 0..stats.n_vocab {
        if tail_set.len() >= tail_cap {
            break;
        }
        tail_set.insert(((k.wrapping_mul(248309) + 17) % stats.n_vocab) as u32);
    }
    let mut active_set: BTreeSet<u32> = stats.top.iter().flat_map(|v| v.iter().map(|&(_, id)| id)).collect();
    active_set.extend(tail_set.iter().copied());
    let active_ids: Vec<u32> = active_set.into_iter().collect();
    let n_act = active_ids.len();
    let is_tail: Vec<bool> = active_ids.iter().map(|id| tail_set.contains(id)).collect();
    let vid_to_loc: HashMap<u32, usize> = active_ids.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    let mut w_act = vec![0.0f32; n_act * dim];
    let w_p = threads.clamp(1, 32).min(n_act.max(1));
    let rpw = n_act.div_ceil(w_p);
    std::thread::scope(|s| {
        for (ci, chunk) in w_act.chunks_mut(rpw * dim).enumerate() {
            let i0 = ci * rpw;
            let ids = &active_ids;
            s.spawn(move || {
                for (dr, dst) in chunk.chunks_exact_mut(dim).enumerate() {
                    decode_row(ids[i0 + dr] as usize, dst);
                }
            });
        }
    });

    let mut u_stu = vec![0.0f32; m * dim];
    let inv_dim = 1.0 / (dim as f32);
    for (i, &t) in stats.t_idx.iter().enumerate() {
        let xr = &cur_stu[t * dim..(t + 1) * dim];
        let sq: f32 = xr.iter().map(|&v| v * v).sum();
        let sc = 1.0 / (sq * inv_dim + eps).sqrt();
        for c in 0..dim {
            u_stu[i * dim + c] = xr[c] * sc;
        }
    }

    let mut p_top: Vec<Vec<(usize, f32)>> = Vec::with_capacity(m);
    let mut p_tail = Vec::with_capacity(m);
    let mut is_top_row = vec![false; m * n_act];
    for i in 0..m {
        let log_z = (stats.max_l[i] as f64) + stats.sum_exp[i].max(1e-30).ln();
        let mut pts = Vec::with_capacity(stats.top[i].len());
        let mut sum_top = 0.0f64;
        for &(l, vid) in &stats.top[i] {
            let p = ((l as f64) - log_z).exp();
            sum_top += p;
            if let Some(&loc) = vid_to_loc.get(&vid) {
                is_top_row[i * n_act + loc] = true;
                pts.push((loc, p as f32));
            }
        }
        p_tail.push((1.0 - sum_top).max(0.0) as f32);
        p_top.push(pts);
    }

    let row_bg_stats: Vec<(usize, f32, f32)> = (0..m)
        .map(|i| {
            let tmask = &is_top_row[i * n_act..(i + 1) * n_act];
            let b_i = (0..n_act).filter(|&s| !tmask[s] && is_tail[s]).count();
            let tail_cnt_i = stats.n_vocab.saturating_sub(p_top[i].len());
            let w_bg_i = if b_i > 0 { (tail_cnt_i as f32 / b_i as f32).max(1.0) } else { 1.0 };
            (b_i, w_bg_i.ln(), p_tail[i] / (b_i.max(1) as f32))
        })
        .collect();

    let (mut g, mut best_g) = (out_norm_init.to_vec(), out_norm_init.to_vec());
    let (mut m1, mut m2) = (vec![0.0f32; dim], vec![0.0f32; dim]);
    let mut best_loss = f64::INFINITY;
    let (mut xn, mut logits, mut d_logits, mut grad_xn) = (
        vec![0.0f32; m * dim],
        vec![0.0f32; m * n_act],
        vec![0.0f32; m * n_act],
        vec![0.0f32; m * dim],
    );
    let mut w_act_t = vec![0.0f32; dim * n_act];
    for s in 0..n_act {
        for c in 0..dim {
            w_act_t[c * n_act + s] = w_act[s * dim + c];
        }
    }

    let m_val = m / 4;
    let m_tr = m - m_val;
    if m_val == 0 || m_tr == 0 {
        return out_norm_init.to_vec();
    }
    let reg = 0.08f32;
    let inv_m_tr = 1.0 / (m_tr as f32);
    for step in 0..16 {
        for i in 0..m {
            for c in 0..dim {
                xn[i * dim + c] = u_stu[i * dim + c] * g[c];
            }
        }
        sgemm_nt(m, n_act, dim, 1.0, &xn, dim, &w_act, dim, 0.0, &mut logits, n_act, threads);
        let mut loss_val = 0.0f64;
        for i in 0..m {
            let is_tr = i % 4 != 3;
            let lrow = &logits[i * n_act..(i + 1) * n_act];
            let drow = &mut d_logits[i * n_act..(i + 1) * n_act];
            let tmask = &is_top_row[i * n_act..(i + 1) * n_act];
            let (_b_i, ln_w_bg_i, p_bg_tgt) = row_bg_stats[i];
            let mut lmax = f32::NEG_INFINITY;
            for s in 0..n_act {
                if tmask[s] || is_tail[s] {
                    let adj = lrow[s] + if tmask[s] { 0.0 } else { ln_w_bg_i };
                    drow[s] = adj;
                    lmax = lmax.max(adj);
                } else {
                    drow[s] = 0.0;
                }
            }
            let mut se = 0.0f64;
            for s in 0..n_act {
                if tmask[s] || is_tail[s] {
                    se += ((drow[s] - lmax) as f64).exp();
                }
            }
            let inv_se = (1.0 / se.max(1e-30)) as f32;
            let mut q_bg = 0.0f64;
            for s in 0..n_act {
                if !tmask[s] && !is_tail[s] {
                    drow[s] = 0.0;
                    continue;
                }
                let q = ((drow[s] - lmax).exp()) * inv_se;
                if tmask[s] {
                    drow[s] = q * inv_m_tr;
                } else {
                    q_bg += q as f64;
                    drow[s] = (q - p_bg_tgt) * inv_m_tr;
                }
            }
            if is_tr {
                for &(loc, pt) in &p_top[i] {
                    drow[loc] -= pt * inv_m_tr;
                }
            } else {
                for &(loc, pt) in &p_top[i] {
                    let ps = ((lrow[loc] - lmax).exp() * inv_se).max(1e-20);
                    loss_val += (pt as f64) * ((pt.max(1e-20) / ps) as f64).ln();
                }
                if p_tail[i] > 1e-12 {
                    loss_val += (p_tail[i] as f64) * ((p_tail[i] as f64) / q_bg.max(1e-20)).ln();
                }
                drow.fill(0.0);
            }
        }
        let val_kl = loss_val / (m_val as f64);
        if val_kl < best_loss {
            best_loss = val_kl;
            best_g.copy_from_slice(&g);
        }
        if step == 15 {
            break;
        }
        sgemm_nt(m, dim, n_act, 1.0, &d_logits, n_act, &w_act_t, n_act, 0.0, &mut grad_xn, dim, threads);
        let t_f = (step + 1) as i32;
        let bc1 = 1.0 - 0.9f32.powi(t_f);
        let bc2 = 1.0 - 0.999f32.powi(t_f);
        for c in 0..dim {
            let mut gc = 2.0 * reg * (g[c] - out_norm_init[c]);
            for i in 0..m {
                gc += grad_xn[i * dim + c] * u_stu[i * dim + c];
            }
            m1[c] = 0.9 * m1[c] + 0.1 * gc;
            m2[c] = 0.999 * m2[c] + 0.001 * gc * gc;
            let upd = 0.015 * (m1[c] / bc1) / ((m2[c] / bc2).sqrt() + 1e-8);
            g[c] -= upd;
            let w0 = out_norm_orig[c];
            if w0.is_finite() && w0.abs() > 1e-6 {
                g[c] = (g[c] / w0).clamp(0.85, 1.18) * w0;
            }
        }
    }
    best_g
}

#[cfg(test)]
mod tests {
    use super::{align_rmsnorm_weights, optimize_output_norm_kld, TeacherHeadStats};
    use crate::turbo_linalg::sgemm_nt;

    #[test]
    fn norm_ridge_and_kld_adam_strictly_reduce_head_kl() {
        let (n_tok, dim, n_vocab) = (64usize, 64usize, 4096usize);
        let mut rng = 0xabcd_1234_5678_90efu64;
        let mut nxt = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((rng >> 40) as f32) / 16777216.0 - 0.5) * 2.0
        };
        let w_norm: Vec<f32> = (0..dim).map(|_| 1.0 + 0.2 * nxt()).collect();
        let x_ref: Vec<f32> = (0..n_tok * dim).map(|_| nxt()).collect();
        let x_stu: Vec<f32> = x_ref.iter().enumerate().map(|(i, &v)| v * (if i % 2 == 0 { 1.15 } else { 0.85 }) + 0.08 * nxt()).collect();
        let mut y_ref = vec![0.0f32; n_tok * dim];
        for t in 0..n_tok {
            crate::llm_ops::rms_norm(&x_ref[t * dim..(t + 1) * dim], &w_norm, 1e-6, &mut y_ref[t * dim..(t + 1) * dim]);
        }
        let g_init = align_rmsnorm_weights(&x_stu, &y_ref, &w_norm, 1e-6, n_tok, dim);
        let head_w: Vec<f32> = (0..n_vocab * dim).map(|_| 0.15 * nxt()).collect();
        let mut stats = TeacherHeadStats::new(&y_ref, n_tok, dim, n_vocab);
        stats.observe_chunk(0, n_vocab, &head_w, 1);

        let g_opt = optimize_output_norm_kld(&x_stu, &w_norm, &g_init, 1e-6, &stats, &|r, dst| dst.copy_from_slice(&head_w[r * dim..(r + 1) * dim]), dim, 1);
        let eval_kl = |gw: &[f32]| -> f64 {
            let mut y_s = vec![0.0f32; n_tok * dim];
            for t in 0..n_tok {
                crate::llm_ops::rms_norm(&x_stu[t * dim..(t + 1) * dim], gw, 1e-6, &mut y_s[t * dim..(t + 1) * dim]);
            }
            let (mut lr, mut ls) = (vec![0.0; n_tok * n_vocab], vec![0.0; n_tok * n_vocab]);
            sgemm_nt(n_tok, n_vocab, dim, 1.0, &y_ref, dim, &head_w, dim, 0.0, &mut lr, n_vocab, 1);
            sgemm_nt(n_tok, n_vocab, dim, 1.0, &y_s, dim, &head_w, dim, 0.0, &mut ls, n_vocab, 1);
            let mut tot = 0.0f64;
            for t in 0..n_tok {
                let (mr, ms) = (
                    lr[t * n_vocab..(t + 1) * n_vocab].iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    ls[t * n_vocab..(t + 1) * n_vocab].iter().copied().fold(f32::NEG_INFINITY, f32::max),
                );
                let zer: f64 = lr[t * n_vocab..(t + 1) * n_vocab].iter().map(|&v| ((v - mr) as f64).exp()).sum();
                let zes: f64 = ls[t * n_vocab..(t + 1) * n_vocab].iter().map(|&v| ((v - ms) as f64).exp()).sum();
                for v in 0..n_vocab {
                    let p = ((lr[t * n_vocab + v] - mr) as f64).exp() / zer;
                    let q = (((ls[t * n_vocab + v] - ms) as f64).exp() / zes).max(1e-20);
                    tot += p * (p.max(1e-20) / q).ln();
                }
            }
            tot / n_tok as f64
        };
        let kl_orig = eval_kl(&w_norm);
        let kl_init = eval_kl(&g_init);
        let kl_opt = eval_kl(&g_opt);
        assert!(kl_opt < kl_orig, "KLD did not strictly improve vs orig: {kl_opt} >= {kl_orig}");
        assert!(kl_opt <= kl_init + 1e-6, "Adam step degraded KLD vs init: {kl_opt} > {kl_init}");
    }
}
