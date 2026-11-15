//! Shared 2D/4D Cholesky GPTQ tuple search, `Turbo2.5` (`bits = 5`) `Orth4D-64` group quantizer,
//! and tile-step word packers shared between `wht_turbo_gptq` and `turbo_full_gptq`.

use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::wht2::fwht32_inplace;
use crate::wht_turbo_enc::{rademacher_sgn2, rademacher_sgn4};
use crate::wht_turbo_gptq::HadamardBlockChol;
use crate::wht_turbo_tables::{ec_orth2d_nz_mask, orth4d64_nz_mask, orth4d64_tables, EC_BITS_4D_64};

#[inline(always)]
pub(crate) fn round_f16_12bit(raw_f16: u16) -> u16 {
    if raw_f16 == 0 {
        0
    } else {
        (raw_f16.min(0x7bdf) + 32) & 0xffc0
    }
}

#[inline(always)]
fn eval_2d_cost(
    c: usize,
    cu: [f32; 2],
    base_pen: f32,
    wk0: f32,
    wk1: f32,
    w0: f32,
    a0: f32,
    a1: f32,
    m10: f32,
    neg0: bool,
) -> (f32, u8, f32, f32) {
    let u0 = if neg0 { -cu[0] } else { cu[0] };
    let d0 = wk0 - u0;
    let t1 = wk1 + m10 * (w0 - u0);
    let neg1 = cu[1] > 0.0 && t1.is_sign_negative();
    let u1 = if neg1 { -cu[1] } else { cu[1] };
    let d1 = t1 - u1;
    let cost = a0 * d0 * d0 + base_pen + a1 * d1 * d1;
    let _ = c;
    let s2 = (neg0 as u8) | ((neg1 as u8) << 1);
    (cost, s2, u0, u1)
}

/// Exact branch-and-bound sequential 2D GPTQ tuple search over `(wk0, wk1)` given unit LDL rows.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn search_2d_gptq_pair(
    cb2: &[[f32; 2]],
    pen_u: &[f32; 64],
    wk0: f32,
    wk1: f32,
    w0: f32,
    a0: f32,
    a1: f32,
    m10: f32,
) -> (u8, u8, f32, f32) {
    let n_cb = cb2.len();
    let abs0 = wk0.abs();
    let abs1 = wk1.abs();

    // Tight initial bound from unconstrained greedy centroid.
    let mut seed_c = 0usize;
    let mut seed_score = f32::INFINITY;
    for c in 0..n_cb {
        let d0 = abs0 - cb2[c][0];
        let d1 = abs1 - cb2[c][1];
        let sc = pen_u[c] + a0 * d0 * d0 + a1 * d1 * d1;
        if sc < seed_score {
            seed_score = sc;
            seed_c = c;
        }
    }

    let s0_seed = cb2[seed_c][0] > 0.0 && wk0.is_sign_negative();
    let (mut best_cost, mut best_s2, bu0, bu1) =
        eval_2d_cost(seed_c, cb2[seed_c], pen_u[seed_c], wk0, wk1, w0, a0, a1, m10, s0_seed);
    let mut best_c = seed_c;
    let mut best_u = (bu0, bu1);

    for c in 0..n_cb {
        let cu = cb2[c];
        let base_pen = pen_u[c];
        let d0_min = abs0 - cu[0];
        let c0 = a0 * d0_min * d0_min + base_pen;
        if c0 >= best_cost {
            continue;
        }
        let s0_pref = cu[0] > 0.0 && wk0.is_sign_negative();
        let u0 = if s0_pref { -cu[0] } else { cu[0] };
        let t1 = wk1 + m10 * (w0 - u0);
        let neg1 = cu[1] > 0.0 && t1.is_sign_negative();
        let u1 = if neg1 { -cu[1] } else { cu[1] };
        let d1 = t1 - u1;
        let cost = c0 + a1 * d1 * d1;
        if cost < best_cost {
            best_cost = cost;
            best_c = c;
            best_s2 = (s0_pref as u8) | ((neg1 as u8) << 1);
            best_u = (u0, u1);
        }

        if cu[0] > 0.0 {
            let d0_alt = abs0 + cu[0];
            let c0_alt = a0 * d0_alt * d0_alt + base_pen;
            if c0_alt < best_cost {
                let (cost_alt, s2_alt, u0_a, u1_a) =
                    eval_2d_cost(c, cu, base_pen, wk0, wk1, w0, a0, a1, m10, !s0_pref);
                if cost_alt < best_cost {
                    best_cost = cost_alt;
                    best_c = c;
                    best_s2 = s2_alt;
                    best_u = (u0_a, u1_a);
                }
            }
        }
    }

    (best_c as u8, best_s2, best_u.0, best_u.1)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn eval_4d_pref(
    cu: [f32; 4],
    base_pen: f32,
    wk: [f32; 4],
    w_pre: [f32; 3],
    a: [f32; 4],
    m10: f32,
    m2: [f32; 2],
    m3: [f32; 3],
) -> (f32, u8, [f32; 4]) {
    let s0 = cu[0] > 0.0 && wk[0].is_sign_negative();
    let u0 = if s0 { -cu[0] } else { cu[0] };
    let e0 = w_pre[0] - u0;
    let t1 = wk[1] + m10 * e0;
    let s1 = cu[1] > 0.0 && t1.is_sign_negative();
    let u1 = if s1 { -cu[1] } else { cu[1] };
    let e1 = w_pre[1] - u1;
    let t2 = wk[2] + m2[0] * e0 + m2[1] * e1;
    let s2 = cu[2] > 0.0 && t2.is_sign_negative();
    let u2 = if s2 { -cu[2] } else { cu[2] };
    let e2 = w_pre[2] - u2;
    let t3 = wk[3] + m3[0] * e0 + m3[1] * e1 + m3[2] * e2;
    let s3 = cu[3] > 0.0 && t3.is_sign_negative();
    let u3 = if s3 { -cu[3] } else { cu[3] };
    let cost = base_pen
        + a[0] * (wk[0] - u0).powi(2)
        + a[1] * (t1 - u1).powi(2)
        + a[2] * (t2 - u2).powi(2)
        + a[3] * (t3 - u3).powi(2);
    let s4 = (s0 as u8) | ((s1 as u8) << 1) | ((s2 as u8) << 2) | ((s3 as u8) << 3);
    (cost, s4, [u0, u1, u2, u3])
}

/// Exact branch-and-bound sequential 4D GPTQ search for `ORTH4D_64_F16` (`bits = 5`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn search_4d_gptq_quad(
    cb64: &[[f32; 4]; 64],
    pen_u: &[f32; 64],
    wk: [f32; 4],
    w_pre: [f32; 3],
    a: [f32; 4],
    m10: f32,
    m2: [f32; 2],
    m3: [f32; 3],
) -> (u8, u8, [f32; 4]) {
    let abs_wk = [wk[0].abs(), wk[1].abs(), wk[2].abs(), wk[3].abs()];
    let mut seed_c = 0usize;
    let mut seed_score = f32::INFINITY;
    for c in 0..64 {
        let cu = cb64[c];
        let sc = pen_u[c]
            + a[0] * (abs_wk[0] - cu[0]).powi(2)
            + a[1] * (abs_wk[1] - cu[1]).powi(2)
            + a[2] * (abs_wk[2] - cu[2]).powi(2)
            + a[3] * (abs_wk[3] - cu[3]).powi(2);
        if sc < seed_score {
            seed_score = sc;
            seed_c = c;
        }
    }

    let (mut best_cost, mut best_s4, mut best_u) =
        eval_4d_pref(cb64[seed_c], pen_u[seed_c], wk, w_pre, a, m10, m2, m3);
    let mut best_c = seed_c;
    // Also seed origin (c = 0) when zeroed penalty makes origin strong.
    if seed_c != 0 {
        let (c0_val, s0_val, u0_val) =
            eval_4d_pref(cb64[0], pen_u[0], wk, w_pre, a, m10, m2, m3);
        if c0_val < best_cost {
            best_cost = c0_val;
            best_c = 0;
            best_s4 = s0_val;
            best_u = u0_val;
        }
    }

    for c in 0..64 {
        let cu = cb64[c];
        let pen = pen_u[c];
        let min0 = pen + a[0] * (abs_wk[0] - cu[0]).powi(2);
        if min0 >= best_cost {
            continue;
        }
        let s0p = cu[0] > 0.0 && wk[0].is_sign_negative();
        let n0_passes = if cu[0] > 0.0 { 2 } else { 1 };
        for p0 in 0..n0_passes {
            let s0 = if p0 == 0 { s0p } else { !s0p };
            let u0 = if s0 { -cu[0] } else { cu[0] };
            let c0 = pen + a[0] * (wk[0] - u0).powi(2);
            if c0 >= best_cost {
                break;
            }
            let e0 = w_pre[0] - u0;
            let t1 = wk[1] + m10 * e0;
            let s1p = cu[1] > 0.0 && t1.is_sign_negative();
            let n1_passes = if cu[1] > 0.0 { 2 } else { 1 };
            for p1 in 0..n1_passes {
                let s1 = if p1 == 0 { s1p } else { !s1p };
                let u1 = if s1 { -cu[1] } else { cu[1] };
                let c1 = c0 + a[1] * (t1 - u1).powi(2);
                if c1 >= best_cost {
                    break;
                }
                let e1 = w_pre[1] - u1;
                let t2 = wk[2] + m2[0] * e0 + m2[1] * e1;
                let s2p = cu[2] > 0.0 && t2.is_sign_negative();
                let n2_passes = if cu[2] > 0.0 { 2 } else { 1 };
                for p2 in 0..n2_passes {
                    let s2 = if p2 == 0 { s2p } else { !s2p };
                    let u2 = if s2 { -cu[2] } else { cu[2] };
                    let c2 = c1 + a[2] * (t2 - u2).powi(2);
                    if c2 >= best_cost {
                        break;
                    }
                    let e2 = w_pre[2] - u2;
                    let t3 = wk[3] + m3[0] * e0 + m3[1] * e1 + m3[2] * e2;
                    let s3 = cu[3] > 0.0 && t3.is_sign_negative();
                    let u3 = if s3 { -cu[3] } else { cu[3] };
                    let cost = c2 + a[3] * (t3 - u3).powi(2);
                    if cost < best_cost {
                        best_cost = cost;
                        best_c = c;
                        best_s4 = (s0 as u8)
                            | ((s1 as u8) << 1)
                            | ((s2 as u8) << 2)
                            | ((s3 as u8) << 3);
                        best_u = [u0, u1, u2, u3];
                    }
                }
            }
        }
    }

    (best_c as u8, best_s4, best_u)
}

#[allow(clippy::too_many_arguments)]
fn quantize_group128_gptq_4d_once(
    g_rot: &[f32],
    g_unrot_ds: &[f32],
    blks: &[HadamardBlockChol; 4],
    sc: f32,
    eff_hl: f32,
    cb64: &[[f32; 4]; 64],
    quad_idx: &mut [u8; 32],
    quad_sgn: &mut [u8; 32],
) -> (f32, f32, f32) {
    let inv_sc = 1.0 / sc;
    let mut pen_u = [0.0f32; 64];
    for c in 0..64 {
        pen_u[c] = 2.0 * eff_hl * EC_BITS_4D_64[c];
    }
    let (mut dwc, mut dcc, mut bit_sum) = (0.0f32, 0.0f32, 0.0f32);

    for b in 0..4 {
        let chol = &blks[b];
        let w_raw = &g_rot[b * 32..(b + 1) * 32];
        let mut w_u = [0.0f32; 32];
        for j in 0..32 {
            w_u[j] = w_raw[j] * inv_sc;
        }
        let mut work = w_u;
        let mut u_blk = [0.0f32; 32];

        for q in 0..8 {
            let i0 = q * 4;
            let (i1, i2, i3) = (i0 + 1, i0 + 2, i0 + 3);
            let (best_c, best_s4, best_u) = search_4d_gptq_quad(
                cb64,
                &pen_u,
                [work[i0], work[i1], work[i2], work[i3]],
                [w_u[i0], w_u[i1], w_u[i2]],
                [chol.alpha[i0], chol.alpha[i1], chol.alpha[i2], chol.alpha[i3]],
                chol.m[i1][i0],
                [chol.m[i2][i0], chol.m[i2][i1]],
                [chol.m[i3][i0], chol.m[i3][i1], chol.m[i3][i2]],
            );
            quad_idx[b * 8 + q] = best_c;
            quad_sgn[b * 8 + q] = best_s4;
            u_blk[i0..i0 + 4].copy_from_slice(&best_u);
            bit_sum += EC_BITS_4D_64[best_c as usize];

            let e0 = w_u[i0] - best_u[0];
            let e1 = w_u[i1] - best_u[1];
            let e2 = w_u[i2] - best_u[2];
            let e3 = w_u[i3] - best_u[3];
            for r in (i3 + 1)..32 {
                let mr = &chol.m[r];
                work[r] += mr[i0] * e0 + mr[i1] * e1 + mr[i2] * e2 + mr[i3] * e3;
            }
        }

        fwht32_inplace(&mut u_blk);
        let w_ds = &g_unrot_ds[b * 32..(b + 1) * 32];
        for j in 0..32 {
            let dj = chol.d_unrot[j];
            let uj = u_blk[j];
            dwc = (dj * w_ds[j]).mul_add(uj, dwc);
            dcc = (dj * uj).mul_add(uj, dcc);
        }
    }

    (dwc, dcc, bit_sum)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_group128_gptq_t25(
    ts: usize,
    sr: usize,
    gh: usize,
    lam: f32,
    rms: f32,
    g_rot: &[f32],
    g_ds: &[f32],
    blks: &[HadamardBlockChol; 4],
    codes10_row: &mut [u16; 64],
) -> u16 {
    let (cb64, _, _) = orth4d64_tables();
    let (mut e0, mut h_sum) = (0.0f32, 0.0f32);
    for b in 0..4 {
        for j in 0..32 {
            let dj = blks[b].d_unrot[j];
            let wv = g_ds[b * 32 + j];
            e0 = (dj * wv).mul_add(wv, e0);
            h_sum += dj;
        }
    }
    let h_mean = (h_sum * (1.0 / 128.0)).max(0.1);
    let mut sf16 = round_f16_12bit(f32_to_f16((e0 / h_sum.max(1.0)).sqrt()));
    let mut sc = f16_to_f32(sf16);
    let mut quad_idx = [0u8; 32];
    let mut quad_sgn = [0u8; 32];

    if sc > 0.0 {
        for _it in 0..2 {
            let eff_hl =
                0.5 * lam * h_mean * (rms / (sc * sc).max(rms * 0.02)).min(16.0);
            let (dwc, dcc, _) = quantize_group128_gptq_4d_once(
                g_rot,
                g_ds,
                blks,
                sc,
                eff_hl,
                &cb64,
                &mut quad_idx,
                &mut quad_sgn,
            );
            if dcc > 0.0 && dwc > 0.0 {
                sf16 = round_f16_12bit(f32_to_f16(dwc / dcc));
                sc = f16_to_f32(sf16);
                if sc == 0.0 {
                    quad_idx.fill(0);
                    break;
                }
            } else {
                sf16 = 0;
                sc = 0.0;
                quad_idx.fill(0);
                break;
            }
        }

        if sf16 > 0 && lam > 0.0 {
            let abs_lam = lam * rms;
            let mut best_j = f32::INFINITY;
            let mut best_u16 = sf16;
            let mut best_qidx = quad_idx;
            let mut best_qsgn = quad_sgn;
            for delta in [-64i32, 0, 64] {
                let cand_u16 = ((sf16 as i32 + delta).clamp(64, 0x7bc0) as u16) & 0xffc0;
                let c_sc = f16_to_f32(cand_u16);
                if c_sc <= 0.0 {
                    continue;
                }
                let eff_hl =
                    0.5 * lam * h_mean * (rms / (c_sc * c_sc).max(rms * 0.02)).min(16.0);
                let mut c_idx = [0u8; 32];
                let mut c_sgn = [0u8; 32];
                let (dwc, dcc, bit_sum) = quantize_group128_gptq_4d_once(
                    g_rot,
                    g_ds,
                    blks,
                    c_sc,
                    eff_hl,
                    &cb64,
                    &mut c_idx,
                    &mut c_sgn,
                );
                let opt_u16 = if dcc > 0.0 && dwc > 0.0 {
                    round_f16_12bit(f32_to_f16(dwc / dcc))
                } else {
                    cand_u16
                };
                let opt_sc = f16_to_f32(opt_u16);
                let err_sum = (e0 - 2.0 * opt_sc * dwc + opt_sc * opt_sc * dcc).max(0.0);
                let j = err_sum + abs_lam * bit_sum;
                if j < best_j {
                    best_j = j;
                    best_u16 = opt_u16;
                    best_qidx = c_idx;
                    best_qsgn = c_sgn;
                }
            }
            sf16 = best_u16;
            sc = f16_to_f32(sf16);
            quad_idx = best_qidx;
            quad_sgn = best_qsgn;
        }

        if sc > 0.0 {
            for bi in 0..32 {
                let k = gh * 32 + bi;
                let best = quad_idx[bi] as usize;
                let nz4 = orth4d64_nz_mask(best);
                let rs4 = rademacher_sgn4(ts, k >> 1, sr, k & 1);
                let s4 = (quad_sgn[bi] & nz4) | (rs4 & !nz4);
                codes10_row[k] = (best as u16) | ((s4 as u16) << 6);
            }
        }
    }
    sf16
}

/// Packs a single `4`-row x `256`-col tile step for `bits = 5` (`Turbo2.5`, `80` u32 words).
#[inline]
pub(crate) fn pack_t25_step_words(codes10_r: &[[u16; 64]; 4], pw_step: &mut [u32]) {
    pw_step[64..80].fill(0);
    for lane in 0..32 {
        let c00 = codes10_r[0][lane * 2] as u32;
        let c01 = codes10_r[0][lane * 2 + 1] as u32;
        let c10 = codes10_r[1][lane * 2] as u32;
        let c11 = codes10_r[1][lane * 2 + 1] as u32;
        let c20 = codes10_r[2][lane * 2] as u32;
        let c21 = codes10_r[2][lane * 2 + 1] as u32;
        let c30 = codes10_r[3][lane * 2] as u32;
        let c31 = codes10_r[3][lane * 2 + 1] as u32;
        pw_step[lane * 2] = c00 | (c01 << 10) | (c10 << 20) | ((c30 >> 8) << 30);
        pw_step[lane * 2 + 1] = c11 | (c20 << 10) | (c21 << 20) | ((c31 >> 8) << 30);
        let h16 = (c30 & 0xff) | ((c31 & 0xff) << 8);
        pw_step[64 + (lane >> 1)] |= h16 << ((lane & 1) * 16);
    }
}

/// Packs one `128`-col half-step of `bits = 3, 4, 6` into step buffers.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn pack_2d_half_step(
    bits: u8,
    ts: usize,
    sr: usize,
    gh: usize,
    pair_idx: &[u8; 64],
    pair_sgn: &[u8; 64],
    idx_r: &mut [[u16; 32]; 4],
    sgn_r: &mut [[u8; 32]; 4],
    pw_step: &mut [u32],
) {
    if bits == 6 {
        for sub in 0..16 {
            let lane = gh * 16 + sub;
            let mut word = 0u32;
            for pr in 0..4 {
                let p = sub * 4 + pr;
                let c8 = ((pair_idx[p] as u32) & 0x3f) | (((pair_sgn[p] as u32) & 3) << 6);
                word |= c8 << (pr * 8);
            }
            pw_step[lane * 4 + sr] = word;
        }
    } else {
        let is_ec = bits == 4;
        for sub in 0..16 {
            let lane = gh * 16 + sub;
            let p = sub * 4;
            idx_r[sr][lane] = (pair_idx[p] as u16)
                | ((pair_idx[p + 1] as u16) << 4)
                | ((pair_idx[p + 2] as u16) << 8)
                | ((pair_idx[p + 3] as u16) << 12);
            let mut sb = 0u8;
            for pr in 0..4 {
                let raw_s2 = pair_sgn[p + pr];
                let s2 = if is_ec {
                    let nz2 = ec_orth2d_nz_mask(pair_idx[p + pr] as usize);
                    let rs2 = rademacher_sgn2(ts, lane, sr, pr);
                    (raw_s2 & nz2) | (rs2 & !nz2)
                } else {
                    raw_s2
                };
                sb |= s2 << (pr * 2);
            }
            sgn_r[sr][lane] = sb;
        }
    }
}

/// Flushes packed `bits = 3 | 4` words across the 32 lanes of a tile step.
#[inline]
pub(crate) fn flush_2d_step_words(idx_r: &[[u16; 32]; 4], sgn_r: &[[u8; 32]; 4], pw_step: &mut [u32]) {
    for lane in 0..32 {
        pw_step[lane * 2] = (idx_r[0][lane] as u32) | ((idx_r[1][lane] as u32) << 16);
        pw_step[lane * 2 + 1] = (idx_r[2][lane] as u32) | ((idx_r[3][lane] as u32) << 16);
        pw_step[64 + lane] = (sgn_r[0][lane] as u32)
            | ((sgn_r[1][lane] as u32) << 8)
            | ((sgn_r[2][lane] as u32) << 16)
            | ((sgn_r[3][lane] as u32) << 24);
    }
}

#[cfg(test)]
mod tests {
    use crate::wht_turbo::TurboWhtMatrix;

    #[test]
    fn turbo25_4d_gptq_improves_anisotropic_activation_sqnr() {
        let (rows, cols) = (64usize, 256usize);
        let mut weights = vec![0.0f32; rows * cols];
        let mut rng = 0x2525_cafe_2026_1115u64;
        let mut next_norm = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (((rng >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        for w in &mut weights {
            *w = 0.045 * next_norm();
        }
        let mut ch_amp = vec![1.0f32; cols];
        for c in 0..cols {
            ch_amp[c] = if c % 32 == 7 { 5.2 } else if c % 16 == 3 { 2.4 } else { 0.65 };
        }

        let base = TurboWhtMatrix::encode_with_hessian(5, rows, cols, &weights, None, |_| 0.015).unwrap();
        let gptq = TurboWhtMatrix::encode_with_ch_amp(5, rows, cols, &weights, None, Some(&ch_amp), |_| 0.015).unwrap();

        // Check decode_row / gemv_cpu parity and sc_mode == 2 invariant!
        assert!(gptq.scales_f16.iter().all(|&v| (v & 0x003f) == 0));

        let (mut err_b, mut err_g, mut sig) = (0.0f64, 0.0f64, 0.0f64);
        let (mut y_ref, mut y_b, mut y_g) = (vec![0.0f32; rows], vec![0.0f32; rows], vec![0.0f32; rows]);
        let dec_g = crate::format::PackedTensor::TurboWht(gptq.clone()).decode_f32();
        for _ in 0..32 {
            let x: Vec<f32> = (0..cols).map(|c| ch_amp[c] * next_norm()).collect();
            for r in 0..rows {
                y_ref[r] = weights[r * cols..(r + 1) * cols].iter().zip(&x).map(|(w, xv)| w * xv).sum();
            }
            base.gemv_cpu(&x, &mut y_b, 1);
            gptq.gemv_cpu(&x, &mut y_g, 1);
            for r in 0..rows {
                let y_dec: f32 = dec_g[r * cols..(r + 1) * cols].iter().zip(&x).map(|(w, xv)| w * xv).sum();
                assert!((y_dec - y_g[r]).abs() < 1e-4, "decode vs gemv_cpu mismatch");
                let rv = y_ref[r] as f64;
                sig += rv * rv;
                err_b += (rv - y_b[r] as f64).powi(2);
                err_g += (rv - y_g[r] as f64).powi(2);
            }
        }
        let sqnr_b = 10.0 * (sig / err_b).log10();
        let sqnr_g = 10.0 * (sig / err_g).log10();
        assert!(
            sqnr_g > sqnr_b + 0.8,
            "expected Turbo2.5 4D GPTQ SQNR ({sqnr_g:.2} dB) > baseline ({sqnr_b:.2} dB) + 0.8 dB"
        );
    }
}
