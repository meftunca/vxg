//! High-precision 8D Gosset Lattice (`E8-HQ`, `bits = 8`) Vector Quantizer:
//! - `8-bit` coordinates (`u_j in 0..=255`, `sum(u_j) & 1 == 0`, coset `c in 0..=1`,
//!   `q_j in [-128.0, +127.5]`) packed into `[u32; 2]` (`uint2` = `8` bytes, zero-LUT SIMD).
//! - Exact `16-bit f16` `32`-column sub-scales (`~65 dB` scale precision ceiling).
//! - Reuses full-width `SB_COLS = 4096` `act_order` `SuperBlockHessian` `LDL-GPTQ` and
//!   per-block Schur-complement Lagrangian water-filling (`Lagr(alpha)`).

use crate::format_io::Cur;
use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::huff::{huff_decode, huff_encode};
use crate::turbo_linalg::{rotate_rows_fwht32, sgemm_nt, Block32Chol, SuperBlockHessian};
use crate::wht2::fwht32_inplace;
use crate::wht_turbo::{apply_col_sign, TurboWhtMatrix};
use crate::wht_turbo_gptq::HadamardBlockChol;
use crate::{Error, Result};

const LAGR_P_HQ: f32 = 0.72;

/// Packs an `8D` `E8-HQ` vector (`u_j in 0..=255`, even parity, `c in 0..=1`) into `[w0, w1]`.
#[inline(always)]
pub(crate) fn pack_e8_hq_words(u: &[u8; 8], c: u8) -> [u32; 2] {
    debug_assert_eq!((u[0] ^ u[1] ^ u[2] ^ u[3] ^ u[4] ^ u[5] ^ u[6] ^ u[7]) & 1, 0);
    let w0 = u32::from_le_bytes([u[0], u[1], u[2], u[3]]);
    let w1 = u32::from_le_bytes([u[4], u[5], u[6], ((u[7] >> 1) & 0x7f) | ((c & 1) << 7)]);
    [w0, w1]
}

/// Decodes packed `E8-HQ` `[w0, w1]` into `8` `f32` coordinates in `[-128.0, +127.5]`.
#[inline(always)]
pub(crate) fn decode_e8_hq_words(w0: u32, w1: u32) -> [f32; 8] {
    let b0 = w0.to_le_bytes();
    let b1 = w1.to_le_bytes();
    let par = (b0[0] ^ b0[1] ^ b0[2] ^ b0[3] ^ b1[0] ^ b1[1] ^ b1[2]) & 1;
    let u7 = ((b1[3] & 0x7f) << 1) | par;
    let bias = if (b1[3] & 0x80) != 0 { -127.5f32 } else { -128.0f32 };
    [
        (b0[0] as f32) + bias, (b0[1] as f32) + bias,
        (b0[2] as f32) + bias, (b0[3] as f32) + bias,
        (b1[0] as f32) + bias, (b1[1] as f32) + bias,
        (b1[2] as f32) + bias, (u7 as f32) + bias,
    ]
}

/// Computes `<decode_e8_hq_words(w0, w1), xv[0..8]>` without lookup tables.
#[inline(always)]
pub(crate) fn dot_e8_hq_words(w0: u32, w1: u32, xv: &[f32]) -> f32 {
    let q = decode_e8_hq_words(w0, w1);
    let s0 = q[0].mul_add(xv[0], q[1].mul_add(xv[1], q[2].mul_add(xv[2], q[3] * xv[3])));
    let s1 = q[4].mul_add(xv[4], q[5].mul_add(xv[5], q[6].mul_add(xv[6], q[7] * xv[7])));
    s0 + s1
}

#[inline(always)]
fn opp_parity_u8(ub: u8, yf: f32) -> u8 {
    if ub == 0 { 1 } else if ub == 255 { 254 } else if yf >= (ub as f32) { ub + 1 } else { ub - 1 }
}

/// Solves one `8D` `E8-HQ` Gosset lattice vector under an `8x8` unit lower-triangular `LDL^T` metric.
/// Returns `(q, words, weighted_loss, residual_d)` where `residual_d[i] = y_final[i] - q[i]`.
#[inline]
pub(crate) fn solve_e8_hq_8d_gptq(
    wk8: &[f32; 8], wb8: &[f32; 8], al8: &[f32; 8], m8: &[[f32; 8]; 8],
) -> ([f32; 8], [u32; 2], f32, [f32; 8]) {
    let (mut best_loss, mut best_q, mut best_w, mut best_d) = (f32::INFINITY, [0.0f32; 8], [0u32; 2], [0.0f32; 8]);
    for c in 0u8..=1u8 {
        let bias = if c != 0 { -127.5f32 } else { -128.0f32 };
        let (mut u_sel, mut q_sel, mut alt_u) = ([128u8; 8], [0.0f32; 8], [128u8; 8]);
        let (mut y_arr, mut e_sel, mut d_sel, mut par_sum) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], 0u8);
        for i in 0..8 {
            let mut yi = wk8[i];
            for j in 0..i { yi = m8[i][j].mul_add(e_sel[j], yi); }
            y_arr[i] = yi;
            let yf = yi - bias;
            let ub = (yf.round() as i32).clamp(0, 255) as u8;
            let q = (ub as f32) + bias;
            u_sel[i] = ub;
            q_sel[i] = q;
            e_sel[i] = wb8[i] - q;
            d_sel[i] = yi - q;
            alt_u[i] = opp_parity_u8(ub, yf);
            par_sum ^= ub & 1;
        }
        let (tot_loss, d_fin) = if par_sum != 0 {
            let mut prefix_fl = [0.0f32; 8];
            let mut p_acc = 0.0f32;
            let (mut best_i0, mut best_pen0) = (7usize, f32::INFINITY);
            for i in 0..8 {
                prefix_fl[i] = p_acc;
                let d0 = d_sel[i];
                let l0 = al8[i] * d0 * d0;
                p_acc += l0;
                let df = y_arr[i] - ((alt_u[i] as f32) + bias);
                let pen = al8[i] * df * df - l0;
                if pen < best_pen0 { best_pen0 = pen; best_i0 = i; }
            }
            let (mut best_fl, mut best_fu, mut best_fq, mut best_fd) = (best_loss, u_sel, q_sel, d_sel);
            for step in 0..8 {
                let idx_f = if step == 0 { best_i0 } else {
                    let cand = 7 - (step - 1);
                    if cand == best_i0 { continue; }
                    cand
                };
                let mut fl = prefix_fl[idx_f];
                if fl >= best_fl { continue; }
                let uf = alt_u[idx_f];
                let qf = (uf as f32) + bias;
                let df = y_arr[idx_f] - qf;
                fl = al8[idx_f].mul_add(df * df, fl);
                if fl >= best_fl { continue; }
                let (mut tu, mut tq, mut td, mut dq) = (u_sel, q_sel, d_sel, [0.0f32; 8]);
                tu[idx_f] = uf;
                tq[idx_f] = qf;
                td[idx_f] = df;
                dq[idx_f] = q_sel[idx_f] - qf;
                let mut ok = true;
                for i in (idx_f + 1)..8 {
                    let mut yi = y_arr[i];
                    for j in idx_f..i { yi = m8[i][j].mul_add(dq[j], yi); }
                    let yf = yi - bias;
                    let mut ub = (yf.round() as i32).clamp(0, 255) as u8;
                    if i == 7 {
                        let req = (tu[0] ^ tu[1] ^ tu[2] ^ tu[3] ^ tu[4] ^ tu[5] ^ tu[6]) & 1;
                        if (ub & 1) != req { ub = opp_parity_u8(ub, yf); }
                    }
                    let qi = (ub as f32) + bias;
                    tu[i] = ub;
                    tq[i] = qi;
                    dq[i] = q_sel[i] - qi;
                    let di = yi - qi;
                    td[i] = di;
                    fl = al8[i].mul_add(di * di, fl);
                    if fl >= best_fl { ok = false; break; }
                }
                if ok && fl < best_fl { best_fl = fl; best_fu = tu; best_fq = tq; best_fd = td; }
            }
            if best_fl < best_loss { u_sel = best_fu; q_sel = best_fq; }
            (best_fl, best_fd)
        } else {
            let mut fl = 0.0f32;
            for i in 0..8 { fl = al8[i].mul_add(d_sel[i] * d_sel[i], fl); }
            (fl, d_sel)
        };
        if tot_loss < best_loss {
            best_loss = tot_loss;
            best_q = q_sel;
            best_w = pack_e8_hq_words(&u_sel, c);
            best_d = d_fin;
        }
    }
    (best_q, best_w, best_loss, best_d)
}

/// Solves one `32`-column `FWHT_32` block using L1-resident `Block32Chol` (`4` `E8-HQ` vectors + `f16` scale).
#[inline]
pub(crate) fn solve_block32_e8_hq_chol(
    ch: &Block32Chol, work_blk: &[f32], wt_blk: &[f32], beta: f32,
    w_out: &mut [[u32; 2]; 4], wq_out: &mut [f32; 32],
) -> u16 {
    let mut v_tgt = [0.0f32; 32];
    let (mut e0, mut h_sum, mut max_abs) = (0.0f32, 0.0f32, 0.0f32);
    for k in 0..32 {
        let mut s = work_blk[k];
        let mk = &ch.m32[k];
        for j in 0..k { s = mk[j].mul_add(wt_blk[j], s); }
        v_tgt[k] = s;
        let wk = work_blk[k];
        let hk = ch.hd32[k];
        e0 = (hk * wk).mul_add(wk, e0);
        h_sum += hk;
        let av = wk.abs();
        if av > max_abs { max_abs = av; }
    }
    let rms = (e0 / h_sum.max(1e-12)).sqrt();
    let sc_init = (rms / beta.max(1.0)).max(max_abs / 126.5);
    let mut best_sf = f32_to_f16(sc_init);
    let sc0 = f16_to_f32(best_sf);
    wq_out.fill(0.0);
    w_out.fill(pack_e8_hq_words(&[128; 8], 0));
    if sc0 <= 0.0 { return 0; }

    let eval_given_sc = |sc: f32| -> (f32, [f32; 32], [[u32; 2]; 4], [f32; 32]) {
        let inv_sc = 1.0 / sc;
        let (mut wk, mut wb) = ([0.0f32; 32], [0.0f32; 32]);
        for k in 0..32 { wk[k] = work_blk[k] * inv_sc; wb[k] = wt_blk[k] * inv_sc; }
        let (mut q32, mut d32, mut words, mut raw_loss) = ([0.0f32; 32], [0.0f32; 32], [[0u32; 2]; 4], 0.0f32);
        for q8 in 0..4 {
            let i0 = q8 * 8;
            let (oq, ow, l8, od) = solve_e8_hq_8d_gptq(
                wk[i0..i0 + 8].try_into().unwrap(),
                wb[i0..i0 + 8].try_into().unwrap(),
                &ch.al8[q8], &ch.m8[q8],
            );
            q32[i0..i0 + 8].copy_from_slice(&oq);
            d32[i0..i0 + 8].copy_from_slice(&od);
            words[q8] = ow;
            raw_loss += l8;
            if i0 + 8 < 32 {
                let mut e8 = [0.0f32; 8];
                for j in 0..8 { e8[j] = wb[i0 + j] - oq[j]; }
                for r in (i0 + 8)..32 {
                    let mr = &ch.m32[r][i0..i0 + 8];
                    let a0 = mr[0].mul_add(e8[0], mr[1].mul_add(e8[1], mr[2].mul_add(e8[2], mr[3] * e8[3])));
                    let a1 = mr[4].mul_add(e8[4], mr[5].mul_add(e8[5], mr[6].mul_add(e8[6], mr[7] * e8[7])));
                    wk[r] += a0 + a1;
                }
            }
        }
        let mut vc = [0.0f32; 32];
        for k in 0..32 { vc[k] = v_tgt[k].mul_add(inv_sc, -d32[k]); }
        (sc * sc * raw_loss, q32, words, vc)
    };

    let (best_loss, best_q, best_w, best_vc) = eval_given_sc(sc0);
    let mut best_sc = sc0;
    // Closed-form optimal LS f16 scale given quantized lattice coords `best_q` (zero extra lattice passes!).
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for k in 0..32 {
        let ak = ch.al32[k];
        num = ak.mul_add(v_tgt[k] * best_vc[k], num);
        den = ak.mul_add(best_vc[k] * best_vc[k], den);
    }
    if den > 0.0 && num > 0.0 {
        let sf_cand = f32_to_f16((num / den).clamp(0.92 * sc0, 1.08 * sc0));
        let sc1 = f16_to_f32(sf_cand);
        if sc1 > 0.0 && sf_cand != best_sf {
            let mut loss1 = 0.0f32;
            for k in 0..32 {
                let d = v_tgt[k] - sc1 * best_vc[k];
                loss1 = ch.al32[k].mul_add(d * d, loss1);
            }
            if loss1 < best_loss {
                best_sf = sf_cand;
                best_sc = sc1;
            }
        }
    }
    for k in 0..32 { wq_out[k] = best_sc * best_q[k]; }
    *w_out = best_w;
    best_sf
}

/// Encodes `[rows, cols]` into `TurboWhtMatrix` (`bits = 8`, `VXT-Turbo-HQ`) via full-width
/// `act_order` `SuperBlockHessian` `LDL-GPTQ` and `Lagr(alpha)` water-filling.
pub(crate) fn encode_e8_hq_superblock_gptq<F>(
    rows: usize, cols: usize, mut fetch_rows: F, hess: &SuperBlockHessian,
    base_beta: f32, mut on_chunk: Option<&mut dyn FnMut(usize, usize, &[f32])>, threads: usize,
) -> Result<TurboWhtMatrix>
where
    F: FnMut(usize, usize) -> Result<Vec<f32>>,
{
    let (padded_rows, padded_cols) = (rows.div_ceil(4) * 4, cols.div_ceil(256) * 256);
    let (row_tiles, steps, n_blks32) = (padded_rows / 4, padded_cols / 256, padded_cols / 32);
    let mut scales_f16 = vec![0u16; row_tiles * steps * 32];
    let zpair = pack_e8_hq_words(&[128; 8], 0);
    let mut packed_words = vec![0u32; row_tiles * steps * 256];
    for ch in packed_words.chunks_exact_mut(2) { ch[0] = zpair[0]; ch[1] = zpair[1]; }
    let chunk_rows = if cols > 4096 { 512usize } else { 1024usize };
    let workers = threads.clamp(1, 32);
    let eff_base_beta = base_beta.clamp(1.20, 48.0);

    let mut blk_sqrt_m = vec![1.0f32; n_blks32];
    let mut sb_chols: Vec<Vec<Block32Chol>> = Vec::with_capacity(hess.blocks.len());
    let mut max_bw = 0usize;
    for sb in &hess.blocks {
        max_bw = max_bw.max(sb.bw);
        let nb = sb.bw / 32;
        let mut chols = Vec::with_capacity(nb);
        for (sbi, &orig_b) in sb.blk_ord.iter().enumerate() {
            let g32 = sb.c0 / 32 + orig_b;
            let lc0 = sbi * 32;
            let chol = Block32Chol::from_sb(lc0, sb.bw, &sb.alpha, &sb.m, &sb.h_diag);
            blk_sqrt_m[g32] = (chol.al_sum * (1.0 / 32.0)).max(1e-6).sqrt();
            chols.push(chol);
        }
        sb_chols.push(chols);
    }
    let valid_blks = cols.div_ceil(32).min(n_blks32).max(1);

    let mut blk_betas = vec![eff_base_beta; chunk_rows * n_blks32];
    let mut sf_mat = vec![0u16; chunk_rows * n_blks32];
    let mut w8_mat = vec![[0u32; 2]; chunk_rows * n_blks32 * 4];

    let mut r0 = 0usize;
    while r0 < rows {
        let r1 = (r0 + chunk_rows).min(rows);
        let rc = r1 - r0;
        let w_chunk = fetch_rows(r0, r1)?;
        if w_chunk.len() != rc * cols {
            return Err(Error::InvalidShape("fetch_rows returned wrong element count"));
        }
        if let Some(cb) = on_chunk.as_deref_mut() { cb(r0, r1, &w_chunk); }

        let w_rot = rotate_rows_fwht32(&w_chunk, rc, cols, padded_cols, threads);
        let w_p = workers.min(rc.max(1));
        let rpw = rc.div_ceil(w_p);
        let (bb_sl, sf_sl, w8_sl) = (
            &mut blk_betas[..rc * n_blks32],
            &mut sf_mat[..rc * n_blks32],
            &mut w8_mat[..rc * n_blks32 * 4],
        );

        std::thread::scope(|s| {
            let (wr_ref, bsm_ref, h_ref, ch_ref) = (&w_rot, &blk_sqrt_m, hess, &sb_chols);
            for (ci, ((bb_b, sf_b), w8_b)) in bb_sl
                .chunks_mut(rpw * n_blks32)
                .zip(sf_sl.chunks_mut(rpw * n_blks32))
                .zip(w8_sl.chunks_mut(rpw * n_blks32 * 4))
                .enumerate()
            {
                let rb0 = ci * rpw;
                let nrb = bb_b.len() / n_blks32;
                s.spawn(move || {
                    for dr in 0..nrb {
                        let r = rb0 + dr;
                        let dst = &mut bb_b[dr * n_blks32..(dr + 1) * n_blks32];
                        let row_w = &wr_ref[r * padded_cols..(r + 1) * padded_cols];
                        let mut log_sum = 0.0f64;
                        for g32 in 0..valid_blks {
                            let sl = &row_w[g32 * 32..(g32 + 1) * 32];
                            let rms = (sl.iter().map(|&v| v * v).sum::<f32>() * (1.0 / 32.0)).max(1e-12).sqrt();
                            let sc = (bsm_ref[g32] * rms).max(1e-12).powf(LAGR_P_HQ);
                            dst[g32] = sc;
                            log_sum += (sc as f64).ln();
                        }
                        let inv_gm = (-(log_sum / valid_blks as f64)).exp() as f32;
                        for g32 in 0..valid_blks {
                            dst[g32] = (eff_base_beta * (dst[g32] * inv_gm).clamp(0.48, 2.15)).clamp(1.20, 48.0);
                        }
                        for g32 in valid_blks..n_blks32 { dst[g32] = eff_base_beta; }
                    }
                    let mut w_sb = vec![0.0f32; nrb * max_bw];
                    let mut work_sb = vec![0.0f32; nrb * max_bw];
                    let mut err_panel = vec![0.0f32; nrb * 32];
                    let (mut w_out, mut wq_out) = ([[0u32; 2]; 4], [0.0f32; 32]);
                    for (sb_idx, sb) in h_ref.blocks.iter().enumerate() {
                        let (bw, nb) = (sb.bw, sb.bw / 32);
                        for dr in 0..nrb {
                            let src_row = &wr_ref[(rb0 + dr) * padded_cols..(rb0 + dr + 1) * padded_cols];
                            let dst_row = &mut w_sb[dr * bw..(dr + 1) * bw];
                            for (sbi, &orig_b) in sb.blk_ord.iter().enumerate() {
                                let src_c0 = sb.c0 + orig_b * 32;
                                dst_row[sbi * 32..(sbi + 1) * 32].copy_from_slice(&src_row[src_c0..src_c0 + 32]);
                            }
                        }
                        work_sb[..nrb * bw].copy_from_slice(&w_sb[..nrb * bw]);
                        let chols = &ch_ref[sb_idx];
                        for sbi in 0..nb {
                            let lc0 = sbi * 32;
                            let orig_g32 = sb.c0 / 32 + sb.blk_ord[sbi];
                            let chol = &chols[sbi];
                            for dr in 0..nrb {
                                let wt = &w_sb[dr * bw + lc0..dr * bw + lc0 + 32];
                                let wk = &work_sb[dr * bw + lc0..dr * bw + lc0 + 32];
                                let sf = solve_block32_e8_hq_chol(
                                    chol, wk, wt, bb_b[dr * n_blks32 + orig_g32], &mut w_out, &mut wq_out,
                                );
                                sf_b[dr * n_blks32 + orig_g32] = sf;
                                let off4 = (dr * n_blks32 + orig_g32) * 4;
                                w8_b[off4..off4 + 4].copy_from_slice(&w_out);
                                for k in 0..32 { err_panel[dr * 32 + k] = wt[k] - wq_out[k]; }
                            }
                            if lc0 + 32 < bw {
                                sgemm_nt(
                                    nrb, bw - (lc0 + 32), 32, 1.0, &err_panel, 32,
                                    &chol.m_tail, 32, 1.0, &mut work_sb[lc0 + 32..], bw, 1,
                                );
                            }
                        }
                    }
                });
            }
        });

        let tile0 = r0 / 4;
        for tl in 0..rc.div_ceil(4) {
            let g_tile = tile0 + tl;
            for step in 0..steps {
                let ts = g_tile * steps + step;
                let sc32 = &mut scales_f16[ts * 32..(ts + 1) * 32];
                let pw256 = &mut packed_words[ts * 256..(ts + 1) * 256];
                for sr in 0..4 {
                    let r = tl * 4 + sr;
                    if r >= rc { continue; }
                    for b8 in 0..8 {
                        let g32 = step * 8 + b8;
                        sc32[b8 * 4 + sr] = sf_mat[r * n_blks32 + g32];
                        let off4 = (r * n_blks32 + g32) * 4;
                        for sub in 0..4 {
                            let idx = ((b8 * 4 + sub) * 4 + sr) * 2;
                            pw256[idx] = w8_mat[off4 + sub][0];
                            pw256[idx + 1] = w8_mat[off4 + sub][1];
                        }
                    }
                }
            }
        }
        r0 = r1;
    }
    Ok(TurboWhtMatrix { bits: 8, rows, cols, padded_rows, padded_cols, disk_bytes: 0, scales_f16, packed_words })
}

/// Encodes `[rows, cols]` into `bits = 8` using analytical `32x32` `HadamardBlockChol` (`encode` / untied `token_embd`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_e8_hq_planes<F>(
    rows: usize, cols: usize, padded_cols: usize, weights: &[f32],
    h_rot: Option<&[f32]>, ch_amp: Option<&[f32]>, row_beta: F, threads: usize,
) -> (Vec<u16>, Vec<u32>)
where
    F: Fn(usize) -> f32 + Sync,
{
    let padded_rows = rows.div_ceil(4) * 4;
    let (row_tiles, steps, n_blks32) = (padded_rows / 4, padded_cols / 256, padded_cols / 32);
    let mut blk_chols: Vec<Block32Chol> = Vec::with_capacity(n_blks32);
    for b in 0..n_blks32 {
        let c0 = b * 32;
        let amps = ch_amp.map_or(&[][..], |a| if c0 < a.len() { &a[c0..a.len().min(c0 + 32)] } else { &[] });
        let chol = HadamardBlockChol::from_amps(amps);
        let blk_sc = h_rot.and_then(|hr| hr.get(c0).copied()).unwrap_or(chol.mean_d).max(1e-6);
        let ratio = blk_sc / chol.mean_d.max(1e-12);
        let mut spec = chol.d_unrot;
        for v in &mut spec { *v *= ratio; }
        fwht32_inplace(&mut spec);
        let inv_sq32 = 1.0 / 32.0f64.sqrt();
        let (mut s, mut bm, mut bal, mut bhd) = ([0.0f64; 1024], [0.0f32; 1024], [1.0f32; 32], [1.0f32; 32]);
        for u in 0..32 {
            for v in 0..32 { s[u * 32 + v] = (spec[u ^ v] as f64) * inv_sq32; }
            s[u * 32 + u] = s[u * 32 + u].max(1e-8);
            bhd[u] = s[u * 32 + u] as f32;
        }
        for k in (0..32).rev() {
            let ak = s[k * 32 + k].max(1e-8);
            bal[k] = ak as f32;
            bm[k * 32 + k] = 1.0;
            for j in 0..k { bm[k * 32 + j] = (s[k * 32 + j] / ak) as f32; }
            for u in 0..k {
                let mu = bm[k * 32 + u] as f64;
                for v in 0..=u { s[u * 32 + v] -= ak * mu * (bm[k * 32 + v] as f64); }
            }
        }
        blk_chols.push(Block32Chol::from_sb(0, 32, &bal, &bm, &bhd));
    }

    let mut scales_f16 = vec![0u16; row_tiles * steps * 32];
    let zpair = pack_e8_hq_words(&[128; 8], 0);
    let mut packed_words = vec![0u32; row_tiles * steps * 256];
    for ch in packed_words.chunks_exact_mut(2) { ch[0] = zpair[0]; ch[1] = zpair[1]; }
    let workers = threads.max(1).min(row_tiles.max(1));

    let encode_slice = |t0: usize, sc_sl: &mut [u16], pw_sl: &mut [u32]| {
        let mut r_rot = [0.0f32; 32];
        let (mut w_out, mut wq_out) = ([[0u32; 2]; 4], [0.0f32; 32]);
        for (dt, (sc_tile, pw_tile)) in sc_sl
            .chunks_exact_mut(steps * 32)
            .zip(pw_sl.chunks_exact_mut(steps * 256))
            .enumerate()
        {
            let tile = t0 + dt;
            for sr in 0..4 {
                let r = tile * 4 + sr;
                if r >= rows { continue; }
                let beta = row_beta(r).clamp(1.20, 48.0);
                let src = &weights[r * cols..(r + 1) * cols];
                for step in 0..steps {
                    let sc32 = &mut sc_tile[step * 32..(step + 1) * 32];
                    let pw256 = &mut pw_tile[step * 256..(step + 1) * 256];
                    for b8 in 0..8 {
                        let g32 = step * 8 + b8;
                        let c0 = g32 * 32;
                        r_rot.fill(0.0);
                        for j in 0..32 {
                            if c0 + j < cols { r_rot[j] = apply_col_sign(src[c0 + j], c0 + j); }
                        }
                        fwht32_inplace(&mut r_rot);
                        sc32[b8 * 4 + sr] = solve_block32_e8_hq_chol(
                            &blk_chols[g32], &r_rot, &r_rot, beta, &mut w_out, &mut wq_out,
                        );
                        for sub in 0..4 {
                            let idx = ((b8 * 4 + sub) * 4 + sr) * 2;
                            pw256[idx] = w_out[sub][0];
                            pw256[idx + 1] = w_out[sub][1];
                        }
                    }
                }
            }
        }
    };

    if workers <= 1 || row_tiles < 8 {
        encode_slice(0, &mut scales_f16, &mut packed_words);
    } else {
        let tpw = row_tiles.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, (sc_c, pw_c)) in scales_f16
                .chunks_mut(tpw * steps * 32)
                .zip(packed_words.chunks_mut(tpw * steps * 256))
                .enumerate()
            {
                s.spawn(move || encode_slice(ci * tpw, sc_c, pw_c));
            }
        });
    }
    (scales_f16, packed_words)
}

/// Canonical Huffman encoder for `bits == 8` (`E8-HQ`): `c`-conditioned `u0..u6` + `tail` (`u7>>1 | c<<7`).
pub(crate) fn pack_e8_hq_ec(total_ts: usize, pw: &[u32], out: &mut Vec<u8>) {
    let n_vec = total_ts * 128;
    let mut tail_syms = Vec::with_capacity(n_vec);
    let mut coord_c0 = Vec::with_capacity(n_vec * 4);
    let mut coord_c1 = Vec::with_capacity(n_vec * 4);
    for ch in pw[..n_vec * 2].chunks_exact(2) {
        let b0 = ch[0].to_le_bytes();
        let b1 = ch[1].to_le_bytes();
        tail_syms.push(b1[3]);
        let dst = if (b1[3] & 0x80) == 0 { &mut coord_c0 } else { &mut coord_c1 };
        dst.extend_from_slice(&[b0[0], b0[1], b0[2], b0[3], b1[0], b1[1], b1[2]]);
    }
    huff_encode(&tail_syms, out);
    huff_encode(&coord_c0, out);
    huff_encode(&coord_c1, out);
}

/// Canonical Huffman decoder for `bits == 8` (`E8-HQ`).
pub(crate) fn unpack_e8_hq_ec(total_ts: usize, r: &mut Cur<'_>, pw_out: &mut [u32]) -> Result<()> {
    let n_vec = total_ts * 128;
    let tail_syms = huff_decode(r, n_vec)?;
    let n_c0 = tail_syms.iter().filter(|&&b| (b & 0x80) == 0).count();
    let n_c1 = n_vec - n_c0;
    let coord_c0 = huff_decode(r, n_c0 * 7)?;
    let coord_c1 = huff_decode(r, n_c1 * 7)?;
    let (mut i0, mut i1) = (0usize, 0usize);
    for (k, ch) in pw_out[..n_vec * 2].chunks_exact_mut(2).enumerate() {
        let tb = tail_syms[k];
        if (tb & 0x80) == 0 {
            ch[0] = u32::from_le_bytes([coord_c0[i0], coord_c0[i0 + 1], coord_c0[i0 + 2], coord_c0[i0 + 3]]);
            ch[1] = u32::from_le_bytes([coord_c0[i0 + 4], coord_c0[i0 + 5], coord_c0[i0 + 6], tb]);
            i0 += 7;
        } else {
            ch[0] = u32::from_le_bytes([coord_c1[i1], coord_c1[i1 + 1], coord_c1[i1 + 2], coord_c1[i1 + 3]]);
            ch[1] = u32::from_le_bytes([coord_c1[i1 + 4], coord_c1[i1 + 5], coord_c1[i1 + 6], tb]);
            i1 += 7;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{decode_e8_hq_words, encode_e8_hq_superblock_gptq, pack_e8_hq_words};
    use crate::format::{read_vxt_model, write_vxt_model, PackedTensor, VxtEntry};
    use crate::turbo_linalg::{sgemm_nt, SuperBlockHessian};

    #[test]
    fn e8_hq_pack_decode_parity_and_ec_roundtrip() {
        for c in 0u8..=1 {
            let shift = if c != 0 { 0.5f32 } else { 0.0f32 };
            let u = [0u8, 255, 19, 240, 128, 127, 63, 190];
            let [w0, w1] = pack_e8_hq_words(&u, c);
            let q = decode_e8_hq_words(w0, w1);
            for j in 0..8 {
                assert!((q[j] - ((u[j] as f32 - 128.0) + shift)).abs() < 1e-6);
            }
        }
        let (rows, cols, n_tok) = (128usize, 256usize, 256usize);
        let mut s = 0x717c_c1b7_2722_0a95u64;
        let mut rnd = || -> f32 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f32) / 8388608.0 - 1.0
        };
        let x: Vec<f32> = (0..n_tok * cols).map(|_| rnd()).collect();
        let w: Vec<f32> = (0..rows * cols).map(|_| rnd() * 0.05).collect();
        let hess = SuperBlockHessian::build(&x, None, n_tok, cols, cols, 2);
        let m = encode_e8_hq_superblock_gptq(rows, cols, |r0, r1| Ok(w[r0 * cols..r1 * cols].to_vec()), &hess, 8.0, None, 2).unwrap();
        let bytes = write_vxt_model(&[], &[VxtEntry { name: "hq".into(), tensor: PackedTensor::TurboWht(m.clone()) }]).unwrap();
        let (_, entries) = read_vxt_model(&bytes).unwrap();
        let PackedTensor::TurboWht(dec) = &entries[0].tensor else { panic!("expected TurboWht") };
        assert_eq!(dec.scales_f16, m.scales_f16);
        assert_eq!(dec.packed_words, m.packed_words);
        assert!(dec.bits_per_weight() < 5.75, "hq bpw={:.4}", dec.bits_per_weight());

        let w_dec = entries[0].tensor.decode_f32();
        let (mut y_ref, mut y_q) = (vec![0.0f32; n_tok * rows], vec![0.0f32; n_tok * rows]);
        sgemm_nt(n_tok, rows, cols, 1.0, &x, cols, &w, cols, 0.0, &mut y_ref, rows, 2);
        sgemm_nt(n_tok, rows, cols, 1.0, &x, cols, &w_dec, cols, 0.0, &mut y_q, rows, 2);
        let p_sig: f64 = y_ref.iter().map(|&v| (v as f64).powi(2)).sum();
        let p_err: f64 = y_ref.iter().zip(&y_q).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let sqnr = 10.0 * (p_sig / p_err).log10();
        assert!(sqnr > 31.0, "hq sqnr={sqnr:.2} dB at {:.3} bpw", dec.bits_per_weight());
    }
}
