//! Intra-block `32x32` Hadamard-Cholesky (`GPTQ` / `OBQ`) vector quantizer for `VXT-Turbo`
//! (`Turbo3` `bits = 3` and `EC-Turbo3` `bits = 4`).
//!
//! Because `TurboWhtMatrix` rotates every 32-column block by `Q_b = H_32 * S_b`, an anisotropic
//! input channel second-moment diagonal `D_b = diag(a_{32b}^2 .. a_{32b+31}^2)` becomes a dense
//! `32x32` positive-definite Toeplitz-XOR Hessian `A_b = H_32 D_b H_32` in the Hadamard domain.
//! Factoring `A_b = M_b^T diag(alpha_b) M_b` with unit lower-triangular `M_b` yields exact sequential
//! `2D` codebook error feedback (`work[k] += M_b[k, j] * (w_rot[j] - z_hat[j])`) with **zero**
//! change to the `VXT1` wire format or Metal/CPU decoders.

use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::wht2::fwht32_inplace;
use crate::wht_turbo::apply_col_sign;
use crate::wht_turbo_enc::rademacher_sgn2;
use crate::wht_turbo_tables::{
    ec_orth2d_nz_mask, ec_orth2d_tables, orth2d_tables, EC_BITS_2D_16,
};

const SQRT32: f64 = 5.656_854_249_492_381;

#[derive(Clone)]
struct HadamardBlockChol {
    /// Regularized unrotated channel weights `d_j` (`mean = block_mean`).
    d_unrot: [f32; 32],
    /// Diagonal entries `alpha_0 .. alpha_31` of `A_b = M^T diag(alpha) M`.
    alpha: [f32; 32],
    /// Unit lower-triangular `M` (`m[i][j]` active for `j < i`).
    m: [[f32; 32]; 32],
    /// Mean of `d_unrot` (`trace(A_b) / 32`).
    mean_d: f32,
}

impl HadamardBlockChol {
    fn from_amps(amps: &[f32]) -> Self {
        let mut raw_d = [1.0f64; 32];
        let mut sum = 0.0f64;
        for j in 0..32 {
            let a = amps.get(j).copied().unwrap_or(1.0).max(1e-4) as f64;
            let v = a * a;
            raw_d[j] = v;
            sum += v;
        }
        let mean = (sum * (1.0 / 32.0)).max(1e-12);
        let mut d_reg = [0.0f64; 32];
        let mut d_f32 = [0.0f32; 32];
        let mut reg_sum = 0.0f64;
        for j in 0..32 {
            let c = raw_d[j].clamp(0.25 * mean, 9.0 * mean);
            let r = 0.90 * c + 0.10 * mean;
            d_reg[j] = r;
            reg_sum += r;
        }
        let norm = (32.0 * mean) / reg_sum.max(1e-30);
        for j in 0..32 {
            d_reg[j] *= norm;
            d_f32[j] = d_reg[j] as f32;
        }

        // Toeplitz-XOR identity: (H_32 diag(d) H_32)[u, v] = fwht32(d)[u ^ v] / sqrt(32).
        let mut spec = [0.0f32; 32];
        spec.copy_from_slice(&d_f32);
        fwht32_inplace(&mut spec);
        let mut s = [[0.0f64; 32]; 32];
        for u in 0..32 {
            for v in 0..32 {
                s[u][v] = (spec[u ^ v] as f64) / SQRT32;
            }
            s[u][u] = s[u][u].max(1e-9 * mean);
        }

        // Reverse-order LDL^T: A = M^T diag(alpha) M with unit lower-triangular M.
        let mut alpha = [1.0f32; 32];
        let mut m = [[0.0f32; 32]; 32];
        for k in (0..32).rev() {
            let ak = s[k][k].max(1e-12 * mean);
            alpha[k] = ak as f32;
            m[k][k] = 1.0;
            let inv_ak = 1.0 / ak;
            let mut mk = [0.0f64; 32];
            for j in 0..k {
                mk[j] = s[k][j] * inv_ak;
                m[k][j] = mk[j] as f32;
            }
            for u in 0..k {
                let a_mku = ak * mk[u];
                for v in 0..=u {
                    s[u][v] -= a_mku * mk[v];
                    s[v][u] = s[u][v];
                }
            }
        }

        Self {
            d_unrot: d_f32,
            alpha,
            m,
            mean_d: mean as f32,
        }
    }
}

#[inline(always)]
fn round_f16_12bit(raw_f16: u16) -> u16 {
    if raw_f16 == 0 {
        0
    } else {
        (raw_f16.min(0x7bdf) + 32) & 0xffc0
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_group128_gptq(
    g_rot: &[f32],
    g_unrot_ds: &[f32],
    blks: &[HadamardBlockChol; 4],
    sc: f32,
    eff_hl: f32,
    cb2: &[[f32; 2]; 16],
    ec_cost16: &[f32; 16],
    pair_idx: &mut [u8; 64],
    pair_sgn: &mut [u8; 64],
) -> (f32, f32, f32) {
    let pen_scale = 2.0 * sc * sc * eff_hl;
    let (mut dwc, mut dcc, mut bit_sum) = (0.0f32, 0.0f32, 0.0f32);

    for b in 0..4 {
        let chol = &blks[b];
        let w = &g_rot[b * 32..(b + 1) * 32];
        let mut work = [0.0f32; 32];
        work.copy_from_slice(w);
        let mut u_blk = [0.0f32; 32];

        for p in 0..16 {
            let i0 = p * 2;
            let i1 = i0 + 1;
            let (a0, a1, m10) = (chol.alpha[i0], chol.alpha[i1], chol.m[i1][i0]);
            let (wk0, wk1, w0) = (work[i0], work[i1], w[i0]);

            let mut best_c = 0usize;
            let mut best_s2 = 0u8;
            let mut best_z = (0.0f32, 0.0f32);
            let mut best_u = (0.0f32, 0.0f32);
            let mut best_cost = f32::INFINITY;

            for c in 0..16 {
                let (cu0, cu1) = (cb2[c][0], cb2[c][1]);
                let (m0, m1) = (sc * cu0, sc * cu1);
                let base_pen = pen_scale * ec_cost16[c];

                let s0_cands: &[bool] = if cu0 == 0.0 { &[false] } else { &[false, true] };
                for &neg0 in s0_cands {
                    let z0 = if neg0 { -m0 } else { m0 };
                    let d0 = wk0 - z0;
                    let c0 = a0 * d0 * d0 + base_pen;
                    if c0 >= best_cost {
                        continue;
                    }
                    let t1 = wk1 + m10 * (w0 - z0);
                    let neg1 = cu1 > 0.0 && t1.is_sign_negative();
                    let z1 = if neg1 { -m1 } else { m1 };
                    let d1 = t1 - z1;
                    let cost = c0 + a1 * d1 * d1;
                    if cost < best_cost {
                        best_cost = cost;
                        best_c = c;
                        best_s2 = (neg0 as u8) | ((neg1 as u8) << 1);
                        best_z = (z0, z1);
                        best_u = (
                            if neg0 { -cu0 } else { cu0 },
                            if neg1 { -cu1 } else { cu1 },
                        );
                    }
                }
            }

            pair_idx[b * 16 + p] = best_c as u8;
            pair_sgn[b * 16 + p] = best_s2;
            u_blk[i0] = best_u.0;
            u_blk[i1] = best_u.1;
            bit_sum += ec_cost16[best_c];

            let e0 = w[i0] - best_z.0;
            let e1 = w[i1] - best_z.1;
            for r in (i1 + 1)..32 {
                work[r] = chol.m[r][i0].mul_add(e0, chol.m[r][i1].mul_add(e1, work[r]));
            }
        }

        // Transform unit codebook vector u_blk back to sign-flipped channel basis to accumulate
        // exact d_unrot-weighted LSQ numerator/denominator in O(32 log 32).
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
pub(crate) fn encode_turbo_gptq_planes<F>(
    bits: u8,
    rows: usize,
    cols: usize,
    padded_cols: usize,
    wps: usize,
    weights: &[f32],
    h_rot: Option<&[f32]>,
    ch_amp: &[f32],
    row_lambda: F,
) -> (Vec<u16>, Vec<u32>)
where
    F: Fn(usize) -> f32 + Sync,
{
    debug_assert!(matches!(bits, 3 | 4));
    let is_ec = bits == 4;
    let padded_rows = rows.div_ceil(4) * 4;
    let row_tiles = padded_rows / 4;
    let steps = padded_cols / 256;
    let n_blks32 = padded_cols / 32;

    let mut blk_chols = Vec::with_capacity(n_blks32);
    for b in 0..n_blks32 {
        let c0 = b * 32;
        let slice = if c0 < ch_amp.len() {
            &ch_amp[c0..ch_amp.len().min(c0 + 32)]
        } else {
            &[]
        };
        let mut chol = HadamardBlockChol::from_amps(slice);
        if let Some(hr) = h_rot {
            let blk_scale = hr.get(c0).copied().unwrap_or(1.0).max(0.1);
            let ratio = blk_scale / chol.mean_d.max(1e-12);
            for j in 0..32 {
                chol.d_unrot[j] *= ratio;
                chol.alpha[j] *= ratio;
            }
            chol.mean_d = blk_scale;
        }
        blk_chols.push(chol);
    }

    let (cb2, _, _) = if is_ec { ec_orth2d_tables() } else { orth2d_tables() };
    let ec_cost16 = if is_ec { EC_BITS_2D_16 } else { [0.0; 16] };

    let mut scales_f16 = vec![0u16; row_tiles * steps * 8];
    let mut packed_words = vec![0u32; row_tiles * steps * wps];
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(row_tiles.max(1));

    let encode_tiles = |t0: usize, sc_slice: &mut [u16], pw_slice: &mut [u32]| {
        let mut r_rot = [
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
        ];
        let mut r_ds = [
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
        ];

        for (dt, (sc_tile, pw_tile)) in sc_slice
            .chunks_exact_mut(steps * 8)
            .zip(pw_slice.chunks_exact_mut(steps * wps))
            .enumerate()
        {
            let tile = t0 + dt;
            let mut row_ms = [1.0f32; 4];
            for sr in 0..4 {
                let r = tile * 4 + sr;
                r_rot[sr].fill(0.0);
                r_ds[sr].fill(0.0);
                if r < rows {
                    let src = &weights[r * cols..(r + 1) * cols];
                    let mut wh2 = 0.0f64;
                    for (c, &v) in src.iter().enumerate() {
                        let ds = apply_col_sign(v, c);
                        r_ds[sr][c] = ds;
                        r_rot[sr][c] = ds;
                        let dj = blk_chols[c / 32].d_unrot[c % 32] as f64;
                        wh2 += dj * (v as f64) * (v as f64);
                    }
                    for blk in r_rot[sr].chunks_exact_mut(32) {
                        let arr: &mut [f32; 32] = blk.try_into().unwrap();
                        fwht32_inplace(arr);
                    }
                    row_ms[sr] = ((wh2 / cols.max(1) as f64) as f32).max(1e-12);
                }
            }

            for step in 0..steps {
                let ts = tile * steps + step;
                let sc8 = &mut sc_tile[step * 8..(step + 1) * 8];
                let pw_step = &mut pw_tile[step * wps..(step + 1) * wps];
                let mut idx_r = [[0u16; 32]; 4];
                let mut sgn_r = [[0u8; 32]; 4];
                if is_ec {
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
                    let r = tile * 4 + sr;
                    if r >= rows {
                        continue;
                    }
                    let lam = row_lambda(r);
                    let rms = row_ms[sr];
                    for gh in 0..2 {
                        let g_off = step * 256 + gh * 128;
                        let b0 = g_off / 32;
                        let blks: &[HadamardBlockChol; 4] =
                            (&blk_chols[b0..b0 + 4]).try_into().unwrap();
                        let g_rot = &r_rot[sr][g_off..g_off + 128];
                        let g_ds = &r_ds[sr][g_off..g_off + 128];

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
                        let raw_init = f32_to_f16((e0 / h_sum.max(1.0)).sqrt());
                        let mut sf16 = if is_ec { round_f16_12bit(raw_init) } else { raw_init };
                        let mut sc = f16_to_f32(sf16);
                        let mut pair_idx = [0u8; 64];
                        let mut pair_sgn = [0u8; 64];

                        if sc > 0.0 {
                            for _it in 0..2 {
                                let eff_hl = if is_ec {
                                    0.5 * lam * (rms / (sc * sc).max(rms * 0.015)).min(16.0 * h_mean)
                                } else {
                                    0.5 * lam * h_mean
                                };
                                let (dwc, dcc, _) = quantize_group128_gptq(
                                    g_rot,
                                    g_ds,
                                    blks,
                                    sc,
                                    eff_hl,
                                    &cb2,
                                    &ec_cost16,
                                    &mut pair_idx,
                                    &mut pair_sgn,
                                );
                                if dcc > 0.0 && dwc > 0.0 {
                                    let raw_new = f32_to_f16(dwc / dcc);
                                    sf16 = if is_ec { round_f16_12bit(raw_new) } else { raw_new };
                                    sc = f16_to_f32(sf16);
                                    if sc == 0.0 {
                                        pair_idx.fill(0);
                                        break;
                                    }
                                } else {
                                    sf16 = 0;
                                    sc = 0.0;
                                    pair_idx.fill(0);
                                    break;
                                }
                            }

                            if is_ec && sf16 > 0 && lam > 0.0 {
                                let abs_lam = lam * rms;
                                let mut best_j = e0 + abs_lam * (64.0 * ec_cost16[0] + 2.0);
                                let mut best_u16 = 0u16;
                                let mut best_pidx = [0u8; 64];
                                let mut best_psgn = [0u8; 64];
                                for delta in [-128i32, -64, 0, 64, 128] {
                                    let cand_u16 =
                                        ((sf16 as i32 + delta).clamp(64, 0x7bc0) as u16) & 0xffc0;
                                    let c_sc = f16_to_f32(cand_u16);
                                    if c_sc <= 0.0 {
                                        continue;
                                    }
                                    let eff_hl = 0.5
                                        * lam
                                        * (rms / (c_sc * c_sc).max(rms * 0.015)).min(16.0 * h_mean);
                                    let mut c_idx = [0u8; 64];
                                    let mut c_sgn = [0u8; 64];
                                    let (dwc, dcc, bit_sum) = quantize_group128_gptq(
                                        g_rot, g_ds, blks, c_sc, eff_hl, &cb2, &ec_cost16,
                                        &mut c_idx, &mut c_sgn,
                                    );
                                    let opt_u16 = if dcc > 0.0 && dwc > 0.0 {
                                        round_f16_12bit(f32_to_f16(dwc / dcc))
                                    } else {
                                        0
                                    };
                                    let opt_sc = f16_to_f32(opt_u16);
                                    let err_sum =
                                        (e0 - 2.0 * opt_sc * dwc + opt_sc * opt_sc * dcc).max(0.0);
                                    let j = err_sum + abs_lam * bit_sum;
                                    if j < best_j {
                                        best_j = j;
                                        best_u16 = opt_u16;
                                        best_pidx = c_idx;
                                        best_psgn = c_sgn;
                                    }
                                }
                                sf16 = best_u16;
                                sc = f16_to_f32(sf16);
                                pair_idx = best_pidx;
                                pair_sgn = best_psgn;
                            }

                            if sc > 0.0 {
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
                        sc8[gh * 4 + sr] = sf16;
                    }
                }

                for lane in 0..32 {
                    pw_step[lane * 2] = (idx_r[0][lane] as u32) | ((idx_r[1][lane] as u32) << 16);
                    pw_step[lane * 2 + 1] =
                        (idx_r[2][lane] as u32) | ((idx_r[3][lane] as u32) << 16);
                    pw_step[64 + lane] = (sgn_r[0][lane] as u32)
                        | ((sgn_r[1][lane] as u32) << 8)
                        | ((sgn_r[2][lane] as u32) << 16)
                        | ((sgn_r[3][lane] as u32) << 24);
                }
            }
        }
    };

    if workers <= 1 || row_tiles < 8 {
        encode_tiles(0, &mut scales_f16, &mut packed_words);
    } else {
        let tpw = row_tiles.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, (sc_ch, pw_ch)) in scales_f16
                .chunks_mut(tpw * steps * 8)
                .zip(packed_words.chunks_mut(tpw * steps * wps))
                .enumerate()
            {
                s.spawn(move || encode_tiles(ci * tpw, sc_ch, pw_ch));
            }
        });
    }

    (scales_f16, packed_words)
}

#[cfg(test)]
mod tests {
    use crate::wht_turbo::TurboWhtMatrix;

    #[test]
    fn gptq_cholesky_improves_anisotropic_activation_sqnr() {
        let (rows, cols) = (64usize, 256usize);
        let mut weights = vec![0.0f32; rows * cols];
        let mut rng = 0x5eed_cafe_2026_1111u64;
        let mut next_norm = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (((rng >> 11) as f64) / ((1u64 << 53) as f64)).max(1e-12);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        };
        for w in &mut weights {
            *w = 0.04 * next_norm();
        }
        // Realistic intra-block spiky RMSNorm + attention-sink channel amplitudes
        let mut ch_amp = vec![1.0f32; cols];
        for c in 0..cols {
            ch_amp[c] = if c % 32 == 3 {
                4.5
            } else if c % 16 == 7 {
                2.4
            } else {
                0.7
            };
        }

        let base = TurboWhtMatrix::encode_with_hessian(4, rows, cols, &weights, None, |_| 0.045)
            .unwrap();
        let gptq = TurboWhtMatrix::encode_with_ch_amp(
            4,
            rows,
            cols,
            &weights,
            None,
            Some(&ch_amp),
            |_| 0.045,
        )
        .unwrap();

        let (mut err_base, mut err_gptq, mut sig_e) = (0.0f64, 0.0f64, 0.0f64);
        let mut y_ref = vec![0.0f32; rows];
        let mut y_b = vec![0.0f32; rows];
        let mut y_g = vec![0.0f32; rows];
        for _ in 0..32 {
            let x: Vec<f32> = (0..cols).map(|c| ch_amp[c] * next_norm()).collect();
            for r in 0..rows {
                y_ref[r] = weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(&x)
                    .map(|(w, xv)| w * xv)
                    .sum();
            }
            base.gemv_cpu(&x, &mut y_b, 1);
            gptq.gemv_cpu(&x, &mut y_g, 1);
            for r in 0..rows {
                let ref_v = y_ref[r] as f64;
                sig_e += ref_v * ref_v;
                err_base += (ref_v - y_b[r] as f64).powi(2);
                err_gptq += (ref_v - y_g[r] as f64).powi(2);
            }
        }
        let sqnr_base = 10.0 * (sig_e / err_base).log10();
        let sqnr_gptq = 10.0 * (sig_e / err_gptq).log10();
        assert!(
            sqnr_gptq > sqnr_base + 1.0,
            "expected GPTQ Cholesky SQNR ({sqnr_gptq:.2} dB) > baseline ({sqnr_base:.2} dB) + 1.0 dB"
        );
    }
}
