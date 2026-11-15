//! Chunked Lazy-Batch Super-Block 2D/4D `LDL-GPTQ` encoder with **Hamle 3** 2-sweep
//! regularized Teacher-aligned `scales_f16` coordinate descent (`VXT-Turbo v3`).

use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::turbo_linalg::{rotate_rows_fwht32, sgemm_nt, SuperBlockHessian};
use crate::wht_turbo::TurboWhtMatrix;
use crate::wht_turbo_enc::{rademacher_sgn2, rademacher_sgn4};
use crate::wht_turbo_gptq_4d::{
    flush_2d_step_words, pack_2d_half_step, pack_t25_step_words, round_f16_12bit,
    search_2d_gptq_pair, search_4d_gptq_quad,
};
use crate::wht_turbo_tables::{
    ec_orth2d_tables, orth2d64_tables, orth2d_tables, orth4d64_nz_mask, orth4d64_tables,
    EC_BITS_2D_16, EC_BITS_4D_64,
};
use crate::{Error, Result};

/// Solves one `128`-col GPTQ panel row (32 `4D` quads for `bits = 5` or 64 `2D` pairs otherwise).
#[allow(clippy::too_many_arguments)]
#[inline]
fn solve_group128_gptq(
    bits: u8, lc0: usize, bw: usize, alpha: &[f32], m: &[f32], h_diag: &[f32],
    work_grp: &[f32], wt_grp: &[f32], lam: f32, row_w2: f32, cb2: &[[f32; 2]],
    cb64: &[[f32; 4]; 64], ec_cost: &[f32], pidx_out: &mut [u8], psgn_out: &mut [u8], u_out: &mut [f32; 128],
) -> u16 {
    let is_t25 = bits == 5;
    let is_12b = matches!(bits, 4 | 5);
    let mut v_tgt = [0.0f32; 128];
    for k in 0..128 {
        let mut s = work_grp[k];
        let mk = &m[(lc0 + k) * bw + lc0..];
        for j in 0..k { s += mk[j] * wt_grp[j]; }
        v_tgt[k] = s;
    }

    let (mut e0, mut h_sum, mut a_sum) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..128 {
        let hk = h_diag[lc0 + k];
        e0 += hk * work_grp[k] * work_grp[k];
        h_sum += hk;
        a_sum += alpha[lc0 + k];
    }
    let a_mean = (a_sum * (1.0 / 128.0)).max(1e-6);
    let raw_init = f32_to_f16((e0 / h_sum.max(1e-12)).sqrt());
    let mut sf16 = if is_12b { round_f16_12bit(raw_init) } else { raw_init };
    let mut sc = f16_to_f32(sf16);
    u_out.fill(0.0);
    if sc <= 0.0 { return 0; }

    for _it in 0..2 {
        let inv_sc = 1.0 / sc;
        let (mut wk_cur, mut wt_u) = ([0.0f32; 128], [0.0f32; 128]);
        for k in 0..128 {
            wk_cur[k] = work_grp[k] * inv_sc;
            wt_u[k] = wt_grp[k] * inv_sc;
        }
        let eff_hl = if is_12b && lam > 0.0 {
            0.5 * lam * a_mean * (row_w2 / (sc * sc).max(row_w2 * 0.40)).min(2.0)
        } else { 0.0 };
        let mut pen_u = [0.0f32; 64];
        for c in 0..ec_cost.len().min(64) { pen_u[c] = 2.0 * eff_hl * ec_cost[c]; }

        if is_t25 {
            for q in 0..32 {
                let (i, g) = (q * 4, lc0 + q * 4);
                let a = [alpha[g], alpha[g + 1], alpha[g + 2], alpha[g + 3]];
                let m10 = m[(g + 1) * bw + g];
                let m2 = [m[(g + 2) * bw + g], m[(g + 2) * bw + g + 1]];
                let m3 = [m[(g + 3) * bw + g], m[(g + 3) * bw + g + 1], m[(g + 3) * bw + g + 2]];
                let (c6, s4, uq) = search_4d_gptq_quad(
                    cb64, &pen_u, [wk_cur[i], wk_cur[i + 1], wk_cur[i + 2], wk_cur[i + 3]],
                    [wt_u[i], wt_u[i + 1], wt_u[i + 2]], a, m10, m2, m3,
                );
                pidx_out[q] = c6; psgn_out[q] = s4;
                u_out[i..i + 4].copy_from_slice(&uq);
                let (eu0, eu1, eu2, eu3) = (wt_u[i] - uq[0], wt_u[i + 1] - uq[1], wt_u[i + 2] - uq[2], wt_u[i + 3] - uq[3]);
                for r in (i + 4)..128 {
                    let mr = &m[(lc0 + r) * bw + g..];
                    wk_cur[r] += mr[0] * eu0 + mr[1] * eu1 + mr[2] * eu2 + mr[3] * eu3;
                }
            }
        } else {
            for p2 in 0..64 {
                let (i0, i1, g) = (p2 * 2, p2 * 2 + 1, lc0 + p2 * 2);
                let (c4, s2, u0, u1) = search_2d_gptq_pair(
                    cb2, &pen_u, wk_cur[i0], wk_cur[i1], wt_u[i0], alpha[g], alpha[g + 1], m[(g + 1) * bw + g],
                );
                pidx_out[p2] = c4; psgn_out[p2] = s2; u_out[i0] = u0; u_out[i1] = u1;
                let (eu0, eu1) = (wt_u[i0] - u0, wt_u[i1] - u1);
                for r in (i1 + 1)..128 {
                    let mr = &m[(lc0 + r) * bw + g..];
                    wk_cur[r] += mr[0] * eu0 + mr[1] * eu1;
                }
            }
        }

        let (mut num, mut den) = (0.0f32, 0.0f32);
        for k in 0..128 {
            let mut vc = u_out[k];
            let mk = &m[(lc0 + k) * bw + lc0..];
            for j in 0..k { vc += mk[j] * u_out[j]; }
            let ak = alpha[lc0 + k];
            num += ak * v_tgt[k] * vc;
            den += ak * vc * vc;
        }
        if den > 0.0 && num > 0.0 {
            let raw = f32_to_f16(num / den);
            sf16 = if is_12b { round_f16_12bit(raw) } else { raw };
            sc = f16_to_f32(sf16);
            if sc > 0.0 { continue; }
        }
        u_out.fill(0.0);
        pidx_out[..if is_t25 { 32 } else { 64 }].fill(0);
        return 0;
    }
    sf16
}

#[allow(clippy::too_many_arguments)]
fn cd_scales_chunk(
    bits: u8, rc: usize, n_groups: usize, n_tok: usize, y_tgt: &[f32],
    q_groups: &[Vec<f32>], tok_w: &[f32], sf16_mat: &mut [u16], workers: usize,
) {
    let is_12b = matches!(bits, 4 | 5);
    let w2_tok: Vec<f32> = tok_w.iter().map(|&w| w * w).collect();
    let cd_worker = |rb0: usize, sf_slice: &mut [u16]| {
        let n_rows_b = sf_slice.len() / n_groups;
        let (mut res, mut qq_tr, mut base_sc) = (vec![0.0f32; n_tok], vec![0.0f32; n_groups], vec![0.0f32; n_groups]);
        let mut sf_init = vec![0u16; n_groups];
        for dr in 0..n_rows_b {
            let r = rb0 + dr;
            let sf_row = &mut sf_slice[dr * n_groups..(dr + 1) * n_groups];
            sf_init.copy_from_slice(sf_row);
            res.copy_from_slice(&y_tgt[r * n_tok..(r + 1) * n_tok]);
            for g in 0..n_groups {
                let sc0 = f16_to_f32(sf_row[g]);
                base_sc[g] = sc0;
                let qrow = &q_groups[g][r * n_tok..(r + 1) * n_tok];
                let mut q2 = 0.0f32;
                for t in 0..n_tok {
                    let qv = qrow[t];
                    res[t] -= sc0 * qv;
                    if t % 4 != 3 { q2 += w2_tok[t] * qv * qv; }
                }
                qq_tr[g] = q2;
            }
            let val_init: f64 = (0..n_tok).filter(|&t| t % 4 == 3)
                .map(|t| (w2_tok[t] as f64) * (res[t] as f64).powi(2)).sum();
            for _sw in 0..2 {
                for g in 0..n_groups {
                    let cur_u16 = sf_row[g];
                    if cur_u16 == 0 || qq_tr[g] <= 1e-20 { continue; }
                    let (old_sc, b_sc, qrow) = (f16_to_f32(cur_u16), base_sc[g], &q_groups[g][r * n_tok..(r + 1) * n_tok]);
                    let mut rq_tr = 0.0f32;
                    for t in 0..n_tok {
                        if t % 4 != 3 { rq_tr += w2_tok[t] * res[t] * qrow[t]; }
                    }
                    let prior_w = 0.30 * qq_tr[g];
                    let s_opt = (old_sc * qq_tr[g] + rq_tr + prior_w * b_sc) / (qq_tr[g] + prior_w);
                    let cand = s_opt.clamp(0.85 * b_sc, 1.15 * b_sc);
                    let cand_u16 = if is_12b { round_f16_12bit(f32_to_f16(cand)) } else { f32_to_f16(cand) };
                    if cand_u16 != 0 && cand_u16 != cur_u16 {
                        let new_sc = f16_to_f32(cand_u16);
                        let delta = new_sc - old_sc;
                        let d_obj = (delta as f64) * (-2.0 * (rq_tr as f64) + (delta as f64) * (qq_tr[g] as f64))
                            + (prior_w as f64) * (((new_sc - b_sc) as f64).powi(2) - ((old_sc - b_sc) as f64).powi(2));
                        if d_obj < 0.0 {
                            sf_row[g] = cand_u16;
                            for t in 0..n_tok { res[t] -= delta * qrow[t]; }
                        }
                    }
                }
            }
            let val_new: f64 = (0..n_tok).filter(|&t| t % 4 == 3)
                .map(|t| (w2_tok[t] as f64) * (res[t] as f64).powi(2)).sum();
            if !(val_new < val_init) { sf_row.copy_from_slice(&sf_init); }
        }
    };
    let w_cd = workers.min(rc.max(1));
    if w_cd > 1 && rc >= 8 {
        let rpw = rc.div_ceil(w_cd);
        std::thread::scope(|s| {
            for (ci, chunk) in sf16_mat[..rc * n_groups].chunks_mut(rpw * n_groups).enumerate() {
                s.spawn(move || cd_worker(ci * rpw, chunk));
            }
        });
    } else {
        cd_worker(0, &mut sf16_mat[..rc * n_groups]);
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_chunk_tiles(
    bits: u8, tile0: usize, n_tiles_chunk: usize, rc: usize, steps: usize, n_groups: usize, wps: usize,
    sf16_mat: &[u16], pidx_all: &[u8], psgn_all: &[u8], scales_f16: &mut [u16], packed_words: &mut [u32],
) {
    let is_t25 = bits == 5;
    let is_ec2 = bits == 4;
    let is_t4 = bits == 6;
    for tl in 0..n_tiles_chunk {
        let g_tile = tile0 + tl;
        for step in 0..steps {
            let ts = g_tile * steps + step;
            let sc8 = &mut scales_f16[ts * 8..(ts + 1) * 8];
            let pw_step = &mut packed_words[ts * wps..(ts + 1) * wps];
            if is_t25 {
                let mut c10_t = [[0u16; 64]; 4];
                for sr in 0..4 {
                    for k in 0..64 {
                        c10_t[sr][k] = (rademacher_sgn4(ts, k >> 1, sr, k & 1) as u16) << 6;
                    }
                    let r = tl * 4 + sr;
                    if r >= rc {
                        continue;
                    }
                    for gh in 0..2 {
                        let g = step * 2 + gh;
                        let sf = sf16_mat[r * n_groups + g];
                        sc8[gh * 4 + sr] = sf;
                        if sf > 0 {
                            let off = (r * n_groups + g) * 64;
                            for bi in 0..32 {
                                let k = gh * 32 + bi;
                                let best = pidx_all[off + bi] as usize;
                                let nz4 = orth4d64_nz_mask(best);
                                let rs4 = rademacher_sgn4(ts, k >> 1, sr, k & 1);
                                let s4 = (psgn_all[off + bi] & nz4) | (rs4 & !nz4);
                                c10_t[sr][k] = (best as u16) | ((s4 as u16) << 6);
                            }
                        }
                    }
                }
                pack_t25_step_words(&c10_t, pw_step);
            } else {
                let mut idx_r = [[0u16; 32]; 4];
                let mut sgn_r = [[0u8; 32]; 4];
                if is_ec2 {
                    for sr in 0..4 {
                        for lane in 0..32 {
                            let mut sb = 0u8;
                            for pr in 0..4 {
                                sb |= rademacher_sgn2(ts, lane, sr, pr) << (pr * 2);
                            }
                            sgn_r[sr][lane] = sb;
                        }
                    }
                }
                for sr in 0..4 {
                    let r = tl * 4 + sr;
                    if r >= rc {
                        continue;
                    }
                    for gh in 0..2 {
                        let g = step * 2 + gh;
                        let sf = sf16_mat[r * n_groups + g];
                        sc8[gh * 4 + sr] = sf;
                        if sf > 0 {
                            let off = (r * n_groups + g) * 64;
                            let pi: &[u8; 64] = pidx_all[off..off + 64].try_into().unwrap();
                            let ps: &[u8; 64] = psgn_all[off..off + 64].try_into().unwrap();
                            pack_2d_half_step(bits, ts, sr, gh, pi, ps, &mut idx_r, &mut sgn_r, pw_step);
                        }
                    }
                }
                if !is_t4 {
                    flush_2d_step_words(&idx_r, &sgn_r, pw_step);
                }
            }
        }
    }
}

/// Quantizes `[rows, cols]` using Super-Block Lazy-Batch 2D/4D `LDL-GPTQ` + Teacher-drift target
/// + **Hamle 3** 2-sweep regularized Teacher-aligned `scales_f16` coordinate descent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_turbo_full_gptq<F>(
    bits: u8, rows: usize, cols: usize, mut fetch_rows: F, hess: &SuperBlockHessian,
    x_ref: Option<&[f32]>, x_stu: &[f32], n_tok: usize, base_lam: f32,
    compute_y_stu: bool, mut logit_hook: Option<&mut dyn FnMut(usize, usize, &[f32])>, threads: usize,
) -> Result<(TurboWhtMatrix, Vec<f32>)>
where
    F: FnMut(usize, usize) -> Result<Vec<f32>>,
{
    if !matches!(bits, 3 | 4 | 5 | 6) {
        return Err(Error::Format("encode_turbo_full_gptq requires bits in 3..=6".into()));
    }
    let padded_rows = rows.div_ceil(4) * 4;
    let padded_cols = cols.div_ceil(256) * 256;
    let (row_tiles, steps, n_groups) = (padded_rows / 4, padded_cols / 256, padded_cols / 128);
    let wps = TurboWhtMatrix::words_per_tile_step(bits);

    let (cb2_16, _, _) = if bits == 4 { ec_orth2d_tables() } else { orth2d_tables() };
    let (cb2_64, _, _) = orth2d64_tables();
    let cb2: &[[f32; 2]] = if bits == 6 { &cb2_64 } else { &cb2_16 };
    let (cb64, _, _) = orth4d64_tables();
    let ec_cost: Vec<f32> = match bits {
        5 => EC_BITS_4D_64.to_vec(),
        4 => EC_BITS_2D_16.to_vec(),
        _ => Vec::new(),
    };

    let mut scales_f16 = vec![0u16; row_tiles * steps * 8];
    let mut packed_words = vec![0u32; row_tiles * steps * wps];
    let mut y_stu = if compute_y_stu { vec![0.0f32; n_tok * rows] } else { Vec::new() };

    let x_blend_rot = if let Some(xr) = x_ref {
        let blended: Vec<f32> = xr.iter().zip(x_stu).map(|(&r, &s)| 0.70 * r + 0.30 * s).collect();
        rotate_rows_fwht32(&blended, n_tok, cols, padded_cols, threads)
    } else {
        hess.x_stu_rot.clone()
    };

    let chunk_rows = if cols > 4096 { 256usize } else { 512usize };
    let workers = threads.clamp(1, 32);

    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + chunk_rows).min(rows);
        let rc = r1 - r0;
        let rc_pad = rc.div_ceil(4) * 4;
        let w_chunk = fetch_rows(r0, r1)?;
        if w_chunk.len() != rc * cols {
            return Err(Error::InvalidShape("fetch_rows returned wrong element count"));
        }
        if let Some(hook) = logit_hook.as_deref_mut() { hook(r0, r1, &w_chunk); }

        let (mut row_ms, mut row_w2) = (vec![1.0f32; rc], vec![1.0f32; rc]);
        let inv_cols = 1.0 / (cols.max(1) as f64);
        for r in 0..rc {
            let (mut ms, mut w2) = (0.0f64, 0.0f64);
            for (c, &v) in w_chunk[r * cols..(r + 1) * cols].iter().enumerate() {
                let (vv, ca) = ((v as f64) * (v as f64), hess.ch_amp[c] as f64);
                ms += ca * ca * vv;
                w2 += vv;
            }
            row_ms[r] = ((ms * inv_cols) as f32).max(1e-12);
            row_w2[r] = ((w2 * inv_cols) as f32).max(1e-12);
        }
        let mut sorted_ms = row_ms.clone();
        sorted_ms.sort_by(|a, b| a.total_cmp(b));
        let mat_ms = sorted_ms.get(rc / 2).copied().unwrap_or(1e-6).max(1e-12);
        let row_lams: Vec<f32> = (0..rc)
            .map(|r| if !matches!(bits, 4 | 5) || base_lam <= 0.0 || row_ms[r] > 3.0 * mat_ms {
                0.0
            } else {
                (if bits == 5 { 0.5 * base_lam } else { base_lam }) * (mat_ms / row_ms[r]).clamp(0.50, 1.00)
            })
            .collect();

        let w_rot = rotate_rows_fwht32(&w_chunk, rc, cols, padded_cols, threads);
        let mut w_target = w_rot.clone();
        for sb in &hess.blocks {
            if let Some(ref bd) = sb.b_drift {
                sgemm_nt(
                    rc, sb.bw, sb.bw, 1.0, &w_rot[sb.c0..], padded_cols, bd, sb.bw,
                    1.0, &mut w_target[sb.c0..], padded_cols, threads,
                );
            }
        }
        let mut y_tgt = vec![0.0f32; rc * n_tok];
        sgemm_nt(rc, n_tok, padded_cols, 1.0, &w_rot, padded_cols, &x_blend_rot, padded_cols, 0.0, &mut y_tgt, n_tok, threads);

        let mut work = w_target.clone();
        let mut sf16_mat = vec![0u16; rc * n_groups];
        let mut q_groups: Vec<Vec<f32>> = (0..n_groups).map(|_| vec![0.0f32; rc * n_tok]).collect();
        let (mut pidx_all, mut psgn_all) = (vec![0u8; rc * n_groups * 64], vec![0u8; rc * n_groups * 64]);
        let (mut u_panel, mut err_panel) = (vec![0.0f32; rc * 128], vec![0.0f32; rc * 128]);

        for sb in &hess.blocks {
            for p in 0..(sb.bw / 128) {
                let (lc0, c0) = (p * 128, sb.c0 + p * 128);
                let g = c0 / 128;
                let w_p = workers.min(rc.max(1));
                let rpw = rc.div_ceil(w_p);
                std::thread::scope(|s| {
                    let (wt_ref, wk_ref, rl_ref, rw2_ref, ec_ref) = (&w_target, &work, &row_lams, &row_w2, &ec_cost);
                    for (ci, ((((sf_b, up_b), ep_b), pi_b), ps_b)) in sf16_mat
                        .chunks_mut(rpw * n_groups)
                        .zip(u_panel.chunks_mut(rpw * 128))
                        .zip(err_panel.chunks_mut(rpw * 128))
                        .zip(pidx_all.chunks_mut(rpw * n_groups * 64))
                        .zip(psgn_all.chunks_mut(rpw * n_groups * 64))
                        .enumerate()
                    {
                        let (rb0, nrb) = (ci * rpw, up_b.len() / 128);
                        s.spawn(move || {
                            let mut u_row = [0.0f32; 128];
                            for dr in 0..nrb {
                                let r = rb0 + dr;
                                let off = (dr * n_groups + g) * 64;
                                let sf = solve_group128_gptq(
                                    bits, lc0, sb.bw, &sb.alpha, &sb.m, &sb.h_diag,
                                    &wk_ref[r * padded_cols + c0..], &wt_ref[r * padded_cols + c0..],
                                    rl_ref[r], rw2_ref[r], cb2, &cb64, ec_ref,
                                    &mut pi_b[off..off + 64], &mut ps_b[off..off + 64], &mut u_row,
                                );
                                sf_b[dr * n_groups + g] = sf;
                                let sc = f16_to_f32(sf);
                                up_b[dr * 128..(dr + 1) * 128].copy_from_slice(&u_row);
                                let wt = &wt_ref[r * padded_cols + c0..r * padded_cols + c0 + 128];
                                for k in 0..128 { ep_b[dr * 128 + k] = wt[k] - sc * u_row[k]; }
                            }
                        });
                    }
                });

                if lc0 + 128 < sb.bw {
                    sgemm_nt(
                        rc, sb.bw - (lc0 + 128), 128, 1.0, &err_panel, 128,
                        &sb.m[(lc0 + 128) * sb.bw + lc0..], sb.bw, 1.0, &mut work[c0 + 128..], padded_cols, threads,
                    );
                }
                sgemm_nt(rc, n_tok, 128, 1.0, &u_panel, 128, &hess.x_stu_rot[c0..], padded_cols, 0.0, &mut q_groups[g], n_tok, threads);
            }
        }

        if n_tok >= 64 {
            cd_scales_chunk(bits, rc, n_groups, n_tok, &y_tgt, &q_groups, &hess.tok_w, &mut sf16_mat, workers);
        }
        if compute_y_stu {
            for r in 0..rc {
                let mut acc = vec![0.0f32; n_tok];
                for g in 0..n_groups {
                    let sc = f16_to_f32(sf16_mat[r * n_groups + g]);
                    if sc == 0.0 { continue; }
                    let qrow = &q_groups[g][r * n_tok..(r + 1) * n_tok];
                    for t in 0..n_tok { acc[t] += sc * qrow[t]; }
                }
                for t in 0..n_tok { y_stu[t * rows + (r0 + r)] = acc[t]; }
            }
        }
        pack_chunk_tiles(
            bits, r0 / 4, rc_pad / 4, rc, steps, n_groups, wps, &sf16_mat, &pidx_all, &psgn_all,
            &mut scales_f16, &mut packed_words,
        );
        r0 = r1;
    }

    Ok((TurboWhtMatrix { bits, rows, cols, padded_rows, padded_cols, disk_bytes: 0, scales_f16, packed_words }, y_stu))
}

#[cfg(test)]
mod tests {
    use super::encode_turbo_full_gptq;
    use crate::format::{read_vxt_model, write_vxt_model, PackedTensor, VxtEntry};
    use crate::turbo_linalg::{sgemm_nt, SuperBlockHessian};
    use crate::wht_turbo::TurboWhtMatrix;

    fn make_norm_gen(mut rng: u64) -> impl FnMut() -> f32 {
        move || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (((rng >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }
    }

    #[test]
    fn full_gptq_beats_intra32_on_correlated_drift_and_preserves_sc_mode2() {
        let (rows, cols, n_tr, n_te) = (64usize, 512usize, 256usize, 128usize);
        let mut next_norm = make_norm_gen(0xfeed_face_2026_1115u64);
        let weights: Vec<f32> = (0..rows * cols).map(|_| 0.045 * next_norm()).collect();
        let make_x = |nt: usize, sink: bool, nn: &mut dyn FnMut() -> f32| -> (Vec<f32>, Vec<f32>) {
            let (mut xr, mut xs) = (vec![0.0f32; nt * cols], vec![0.0f32; nt * cols]);
            for t in 0..nt {
                let f = if sink && t == 0 { 40.0 } else if t % 5 == 0 { 2.5 } else { 0.8 };
                let (s1, s2) = (nn() * 3.0, nn() * 2.0);
                for c in 0..cols {
                    let b = f * (s1 * (0.04 * (c as f32)).sin() + s2 * (c % 37 == 3) as i32 as f32 + 0.35 * nn());
                    xr[t * cols + c] = b;
                    xs[t * cols + c] = 0.88 * b + 0.06 * nn() + if c % 32 == 5 { 0.4 * s1 } else { 0.0 };
                }
                for c in 0..cols { xs[t * cols + c] += 0.15 * xr[t * cols + (c ^ 1)]; }
            }
            (xr, xs)
        };
        let (xr_tr, xs_tr) = make_x(n_tr, true, &mut next_norm);
        let (xr_te, xs_te) = make_x(n_te, false, &mut next_norm);
        let hess = SuperBlockHessian::build(&xs_tr, Some(&xr_tr), n_tr, cols, cols, 2);

        for &bits in &[4u8, 5, 6] {
            let lam = if bits == 6 { 0.0 } else { 0.012 };
            let (mat, y_tr) = encode_turbo_full_gptq(
                bits, rows, cols, |r0, r1| Ok(weights[r0 * cols..r1 * cols].to_vec()),
                &hess, Some(&xr_tr), &xs_tr, n_tr, lam, true, None, 2,
            ).unwrap();
            if matches!(bits, 4 | 5) {
                assert!(mat.scales_f16.iter().all(|&v| (v & 0x003f) == 0), "bits={bits} broke sc_mode==2");
            }
            let dec = PackedTensor::TurboWht(mat.clone()).decode_f32();
            let mut y_gemv = vec![0.0f32; n_tr * rows];
            sgemm_nt(n_tr, rows, cols, 1.0, &xs_tr, cols, &dec, cols, 0.0, &mut y_gemv, rows, 2);
            let rel_err: f64 = y_tr.iter().zip(&y_gemv).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>().sqrt()
                / y_gemv.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt().max(1e-9);
            assert!(rel_err < 2e-3, "bits={bits} y_stu mismatch rel_err={rel_err}");

            let base = TurboWhtMatrix::encode_with_ch_amp(bits, rows, cols, &weights, None, Some(&hess.ch_amp), |_| lam).unwrap();
            let dec_b = PackedTensor::TurboWht(base).decode_f32();
            let (mut yr_te, mut yb_te, mut yn_te) = (vec![0.0; n_te * rows], vec![0.0; n_te * rows], vec![0.0; n_te * rows]);
            sgemm_nt(n_te, rows, cols, 1.0, &xr_te, cols, &weights, cols, 0.0, &mut yr_te, rows, 1);
            sgemm_nt(n_te, rows, cols, 1.0, &xs_te, cols, &dec_b, cols, 0.0, &mut yb_te, rows, 1);
            sgemm_nt(n_te, rows, cols, 1.0, &xs_te, cols, &dec, cols, 0.0, &mut yn_te, rows, 1);
            let sig: f64 = yr_te.iter().map(|&v| (v as f64).powi(2)).sum();
            let eb: f64 = yr_te.iter().zip(&yb_te).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let en: f64 = yr_te.iter().zip(&yn_te).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let (sq_b, sq_n) = (10.0 * (sig / eb).log10(), 10.0 * (sig / en).log10());
            assert!(sq_n > sq_b + 1.5, "bits={bits}: new {sq_n:.2} dB <= base {sq_b:.2} dB + 1.5");
        }
    }

    #[test]
    fn tiny_activation_and_sink_token_no_zero_collapse_and_ec_roundtrip() {
        let (rows, cols, n_tok) = (64usize, 256usize, 128usize);
        let mut next_norm = make_norm_gen(0x1357_2468_abcd_ef01u64);
        let weights: Vec<f32> = (0..rows * cols).map(|_| 0.045 * next_norm()).collect();
        let mut xs = vec![0.0f32; n_tok * cols];
        for t in 0..n_tok {
            let amp = if t == 0 { 40.0 * 0.05 } else { 0.05 };
            for c in 0..cols { xs[t * cols + c] = amp * next_norm(); }
        }
        let xr = xs.clone();
        let hess = SuperBlockHessian::build(&xs, Some(&xr), n_tok, cols, cols, 2);
        let (mat, _) = encode_turbo_full_gptq(
            4, rows, cols, |r0, r1| Ok(weights[r0 * cols..r1 * cols].to_vec()),
            &hess, Some(&xr), &xs, n_tok, 0.018, false, None, 2,
        ).unwrap();
        let blob = write_vxt_model(&[], &[VxtEntry { name: "w".into(), tensor: PackedTensor::TurboWht(mat) }]).unwrap();
        let (_, entries) = read_vxt_model(&blob).unwrap();
        let dec = entries[0].tensor.decode_f32();
        let w_n2: f64 = weights.iter().map(|&v| (v as f64).powi(2)).sum();
        let d_n2: f64 = dec.iter().map(|&v| (v as f64).powi(2)).sum();
        let e_n2: f64 = weights.iter().zip(&dec).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let rel = (e_n2 / w_n2).sqrt();
        let rms_ratio = (d_n2 / w_n2).sqrt();
        assert!(rel < 0.28, "tiny-activation weight collapse: rel={rel:.4}");
        assert!((0.93..=1.06).contains(&rms_ratio), "tiny-activation rms_ratio out of range: {rms_ratio:.4}");
    }
}
