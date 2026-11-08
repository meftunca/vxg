//! Parallel tile encoders for `TurboWhtMatrix` (`Turbo2`, `Turbo2.5`, `Turbo3`, `EC-Turbo3`)
//! with per-row reverse water-filling (`row_ms / sc^2`) and 12-bit quantized scale search.

use crate::fp16::{f16_to_f32, f32_to_f16};
use crate::wht2::fwht32_inplace;
use crate::wht_turbo::apply_col_sign;
use crate::wht_turbo_tables::{
    ec_orth2d_nz_mask, ec_orth2d_tables, orth2d_tables, orth4d64_nz_mask, orth4d64_tables,
    orth4d_tables, EC_BITS_2D_16, EC_BITS_4D_64,
};

#[inline(always)]
pub(crate) fn rademacher_sgn2(ts: usize, lane: usize, sr: usize, pr: usize) -> u8 {
    let mut h = (ts as u32).wrapping_mul(0x9e37_79b9)
        ^ ((lane as u32) << 6)
        ^ ((sr as u32) << 2)
        ^ (pr as u32);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    ((h >> 15) & 3) as u8
}

#[inline(always)]
pub(crate) fn rademacher_sgn4(ts: usize, lane: usize, sr: usize, half_k: usize) -> u8 {
    rademacher_sgn2(ts, lane, sr, half_k * 2) | (rademacher_sgn2(ts, lane, sr, half_k * 2 + 1) << 2)
}

#[inline(always)]
fn round_f16_12bit(raw_f16: u16) -> u16 {
    if raw_f16 == 0 {
        0
    } else {
        (raw_f16.min(0x7bdf) + 32) & 0xffc0
    }
}

pub(crate) fn encode_turbo_planes<F>(
    bits: u8,
    rows: usize,
    cols: usize,
    padded_cols: usize,
    wps: usize,
    weights: &[f32],
    h_rot: Option<&[f32]>,
    row_lambda: F,
) -> (Vec<u16>, Vec<u32>)
where
    F: Fn(usize) -> f32 + Sync,
{
    let padded_rows = rows.div_ceil(4) * 4;
    let row_tiles = padded_rows / 4;
    let steps = padded_cols / 256;
    let mut scales_f16 = vec![0u16; row_tiles * steps * 8];
    let mut packed_words = vec![0u32; row_tiles * steps * wps];

    let (cb4, n2_4, _) = orth4d_tables();
    let (cb64, n2_64, _) = orth4d64_tables();
    let (cb2, _, _) = if bits == 4 {
        ec_orth2d_tables()
    } else {
        orth2d_tables()
    };
    let mut cb2_hsq = [[0.0f32; 2]; 16];
    for i in 0..16 {
        cb2_hsq[i] = [0.5 * cb2[i][0] * cb2[i][0], 0.5 * cb2[i][1] * cb2[i][1]];
    }
    let ec_cost16 = if bits == 4 { EC_BITS_2D_16 } else { [0.0; 16] };
    let ec_cost64 = if bits == 5 { EC_BITS_4D_64 } else { [0.0; 64] };
    let ones128 = [1.0f32; 128];

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(row_tiles.max(1));

    let encode_tiles = |t0: usize, sc_slice: &mut [u16], pw_slice: &mut [u32]| {
        let mut rbufs = [
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
            vec![0.0f32; padded_cols],
        ];
        let mut codes_r = [[0u8; 64]; 4];
        let mut codes10_r = [[0u16; 64]; 4];

        for (dt, (sc_tile, pw_tile)) in sc_slice
            .chunks_exact_mut(steps * 8)
            .zip(pw_slice.chunks_exact_mut(steps * wps))
            .enumerate()
        {
            let tile = t0 + dt;
            let mut row_ms = [1.0f32; 4];
            for sr in 0..4 {
                let r = tile * 4 + sr;
                let rb = &mut rbufs[sr];
                rb.fill(0.0);
                if r < rows {
                    let src = &weights[r * cols..(r + 1) * cols];
                    let mut w2_sum = 0.0f64;
                    for (c, &v) in src.iter().enumerate() {
                        w2_sum += (v as f64) * (v as f64);
                        rb[c] = apply_col_sign(v, c);
                    }
                    for blk in rb.chunks_exact_mut(32) {
                        let arr: &mut [f32; 32] = blk.try_into().unwrap();
                        fwht32_inplace(arr);
                    }
                    if let Some(hr) = h_rot {
                        let mut wh2 = 0.0f64;
                        for (k, &rv) in rb.iter().enumerate() {
                            wh2 += (hr[k] as f64) * (rv as f64) * (rv as f64);
                        }
                        row_ms[sr] = ((wh2 / cols.max(1) as f64) as f32).max(1e-12);
                    } else {
                        row_ms[sr] = ((w2_sum / cols.max(1) as f64) as f32).max(1e-12);
                    }
                }
            }

            for step in 0..steps {
                let ts = tile * steps + step;
                let sc8 = &mut sc_tile[step * 8..(step + 1) * 8];
                let pw_step = &mut pw_tile[step * wps..(step + 1) * wps];

                if bits == 2 {
                    for sr in 0..4 {
                        let r = tile * 4 + sr;
                        codes_r[sr].fill(0);
                        if r >= rows {
                            continue;
                        }
                        for gh in 0..2 {
                            let g_in = &rbufs[sr][step * 256 + gh * 128..step * 256 + (gh + 1) * 128];
                            let mut sf16 = round_f16_12bit(f32_to_f16(
                                (g_in.iter().map(|v| v * v).sum::<f32>() * (1.0 / 128.0)).sqrt(),
                            ));
                            let mut sc = f16_to_f32(sf16);
                            let pen = 1.0 + row_lambda(r);
                            if sc > 0.0 {
                                for _it in 0..2 {
                                    let inv = 1.0 / sc;
                                    let (mut dwc, mut dcc) = (0.0f32, 0.0f32);
                                    for (bi, c4) in g_in.chunks_exact(4).enumerate() {
                                        let a = [
                                            c4[0].abs() * inv,
                                            c4[1].abs() * inv,
                                            c4[2].abs() * inv,
                                            c4[3].abs() * inv,
                                        ];
                                        let (mut best, mut bs) = (0usize, f32::NEG_INFINITY);
                                        for i in 0..16 {
                                            let s = a[0] * cb4[i][0]
                                                + a[1] * cb4[i][1]
                                                + a[2] * cb4[i][2]
                                                + a[3] * cb4[i][3]
                                                - pen * n2_4[i];
                                            if s > bs {
                                                bs = s;
                                                best = i;
                                            }
                                        }
                                        let sgn = ((c4[0].is_sign_negative() as u8) << 4)
                                            | ((c4[1].is_sign_negative() as u8) << 5)
                                            | ((c4[2].is_sign_negative() as u8) << 6)
                                            | ((c4[3].is_sign_negative() as u8) << 7);
                                        codes_r[sr][gh * 32 + bi] = (best as u8) | sgn;
                                        for k in 0..4 {
                                            let u = cb4[best][k];
                                            dwc += c4[k].abs() * u;
                                            dcc += u * u;
                                        }
                                    }
                                    if dcc > 0.0 {
                                        sf16 = round_f16_12bit(f32_to_f16(dwc / dcc));
                                        sc = f16_to_f32(sf16);
                                        if sc == 0.0 {
                                            break;
                                        }
                                    }
                                }
                            }
                            sc8[gh * 4 + sr] = sf16;
                        }
                    }
                    for lane in 0..32 {
                        let c0 = (codes_r[0][lane * 2] as u32) | ((codes_r[0][lane * 2 + 1] as u32) << 8);
                        let c1 = (codes_r[1][lane * 2] as u32) | ((codes_r[1][lane * 2 + 1] as u32) << 8);
                        let c2 = (codes_r[2][lane * 2] as u32) | ((codes_r[2][lane * 2 + 1] as u32) << 8);
                        let c3 = (codes_r[3][lane * 2] as u32) | ((codes_r[3][lane * 2 + 1] as u32) << 8);
                        pw_step[lane * 2] = c0 | (c1 << 16);
                        pw_step[lane * 2 + 1] = c2 | (c3 << 16);
                    }
                } else if bits == 5 {
                    for sr in 0..4 {
                        for k in 0..64 {
                            codes10_r[sr][k] = (rademacher_sgn4(ts, k >> 1, sr, k & 1) as u16) << 6;
                        }
                        let r = tile * 4 + sr;
                        if r >= rows {
                            continue;
                        }
                        let lam = row_lambda(r);
                        let rms = row_ms[sr];
                        for gh in 0..2 {
                            let g_in = &rbufs[sr][step * 256 + gh * 128..step * 256 + (gh + 1) * 128];
                            let e0: f32 = g_in.iter().map(|v| v * v).sum();
                            let mut sf16 = round_f16_12bit(f32_to_f16((e0 * (1.0 / 128.0)).sqrt()));
                            let mut sc = f16_to_f32(sf16);
                            let mut grp_best = [0u8; 32];
                            if sc > 0.0 {
                                for _it in 0..2 {
                                    let inv = 1.0 / sc;
                                    let eff_hl = 0.5
                                        * lam
                                        * (rms / (sc * sc).max(rms * 0.02)).min(16.0);
                                    let (mut dwc, mut dcc) = (0.0f32, 0.0f32);
                                    for (bi, c4) in g_in.chunks_exact(4).enumerate() {
                                        let a = [
                                            c4[0].abs() * inv,
                                            c4[1].abs() * inv,
                                            c4[2].abs() * inv,
                                            c4[3].abs() * inv,
                                        ];
                                        let (mut best, mut bs) = (0usize, f32::NEG_INFINITY);
                                        for i in 0..64 {
                                            let s = a[0] * cb64[i][0]
                                                + a[1] * cb64[i][1]
                                                + a[2] * cb64[i][2]
                                                + a[3] * cb64[i][3]
                                                - n2_64[i]
                                                - eff_hl * ec_cost64[i];
                                            if s > bs {
                                                bs = s;
                                                best = i;
                                            }
                                        }
                                        grp_best[bi] = best as u8;
                                        for k in 0..4 {
                                            let u = cb64[best][k];
                                            dwc += c4[k].abs() * u;
                                            dcc += u * u;
                                        }
                                    }
                                    if dcc > 0.0 {
                                        sf16 = round_f16_12bit(f32_to_f16(dwc / dcc));
                                        sc = f16_to_f32(sf16);
                                        if sc == 0.0 {
                                            grp_best.fill(0);
                                            break;
                                        }
                                    } else {
                                        sf16 = 0;
                                        sc = 0.0;
                                        grp_best.fill(0);
                                        break;
                                    }
                                }
                                if sf16 > 0 && lam > 0.0 {
                                    let abs_lam = lam * rms;
                                    let mut best_j = e0 + abs_lam * (32.0 * ec_cost64[0] + 2.0);
                                    let mut best_u16 = 0u16;
                                    let mut best_idx = [0u8; 32];
                                    for delta in [-128i32, -64, 0, 64, 128] {
                                        let cand_u16 = ((sf16 as i32 + delta).clamp(64, 0x7bc0) as u16) & 0xffc0;
                                        let c_sc = f16_to_f32(cand_u16);
                                        if c_sc <= 0.0 {
                                            continue;
                                        }
                                        let inv = 1.0 / c_sc;
                                        let eff_hl = 0.5
                                            * lam
                                            * (rms / (c_sc * c_sc).max(rms * 0.02)).min(16.0);
                                        let mut cand_idx = [0u8; 32];
                                        let (mut dwc, mut dcc, mut bit_sum) = (0.0f32, 0.0f32, 0.0f32);
                                        for (bi, c4) in g_in.chunks_exact(4).enumerate() {
                                            let a = [
                                                c4[0].abs() * inv,
                                                c4[1].abs() * inv,
                                                c4[2].abs() * inv,
                                                c4[3].abs() * inv,
                                            ];
                                            let (mut best, mut bs) = (0usize, f32::NEG_INFINITY);
                                            for i in 0..64 {
                                                let s = a[0] * cb64[i][0]
                                                    + a[1] * cb64[i][1]
                                                    + a[2] * cb64[i][2]
                                                    + a[3] * cb64[i][3]
                                                    - n2_64[i]
                                                    - eff_hl * ec_cost64[i];
                                                if s > bs {
                                                    bs = s;
                                                    best = i;
                                                }
                                            }
                                            cand_idx[bi] = best as u8;
                                            for d in 0..4 {
                                                let u = cb64[best][d];
                                                dwc += c4[d].abs() * u;
                                                dcc += u * u;
                                            }
                                            bit_sum += ec_cost64[best];
                                        }
                                        let opt_u16 = if dcc > 0.0 {
                                            round_f16_12bit(f32_to_f16(dwc / dcc))
                                        } else {
                                            0
                                        };
                                        let opt_sc = f16_to_f32(opt_u16);
                                        let err_sum = (e0 - 2.0 * opt_sc * dwc + opt_sc * opt_sc * dcc).max(0.0);
                                        let j = err_sum + abs_lam * bit_sum;
                                        if j < best_j {
                                            best_j = j;
                                            best_u16 = opt_u16;
                                            best_idx = cand_idx;
                                        }
                                    }
                                    sf16 = best_u16;
                                    sc = f16_to_f32(sf16);
                                    grp_best = best_idx;
                                }
                                if sc > 0.0 {
                                    for (bi, c4) in g_in.chunks_exact(4).enumerate() {
                                        let best = grp_best[bi] as usize;
                                        let k = gh * 32 + bi;
                                        let nz4 = orth4d64_nz_mask(best);
                                        let rs4 = rademacher_sgn4(ts, k >> 1, sr, k & 1);
                                        let raw_sgn = (c4[0].is_sign_negative() as u8)
                                            | ((c4[1].is_sign_negative() as u8) << 1)
                                            | ((c4[2].is_sign_negative() as u8) << 2)
                                            | ((c4[3].is_sign_negative() as u8) << 3);
                                        let s4 = (raw_sgn & nz4) | (rs4 & !nz4);
                                        codes10_r[sr][k] = (best as u16) | ((s4 as u16) << 6);
                                    }
                                }
                            }
                            sc8[gh * 4 + sr] = sf16;
                        }
                    }
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
                } else {
                    let is_ec = bits == 4;
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
                            let g_in = &rbufs[sr][g_off..g_off + 128];
                            let g_h: &[f32] = match h_rot {
                                Some(hr) => &hr[g_off..g_off + 128],
                                None => &ones128,
                            };
                            let (mut e0, mut h_sum) = (0.0f32, 0.0f32);
                            for u in 0..128 {
                                let hv = g_h[u];
                                e0 += hv * g_in[u] * g_in[u];
                                h_sum += hv;
                            }
                            let h_mean = (h_sum * (1.0 / 128.0)).max(0.1);
                            let raw_init = f32_to_f16((e0 / h_sum.max(1.0)).sqrt());
                            let mut sf16 = if is_ec { round_f16_12bit(raw_init) } else { raw_init };
                            let mut sc = f16_to_f32(sf16);
                            let mut pair_idx = [0u8; 64];
                            if sc > 0.0 {
                                for _it in 0..2 {
                                    let inv = 1.0 / sc;
                                    let eff_hl = if is_ec {
                                        0.5 * lam * (rms / (sc * sc).max(rms * 0.015)).min(16.0 * h_mean)
                                    } else {
                                        0.5 * lam * h_mean
                                    };
                                    let (mut dwc, mut dcc) = (0.0f32, 0.0f32);
                                    for (pi, c2) in g_in.chunks_exact(2).enumerate() {
                                        let (h0, h1) = (g_h[pi * 2], g_h[pi * 2 + 1]);
                                        let (a0, a1) = (h0 * c2[0].abs() * inv, h1 * c2[1].abs() * inv);
                                        let (mut best, mut bs) = (0usize, f32::NEG_INFINITY);
                                        for i in 0..16 {
                                            let s = a0 * cb2[i][0]
                                                + a1 * cb2[i][1]
                                                - h0 * cb2_hsq[i][0]
                                                - h1 * cb2_hsq[i][1]
                                                - eff_hl * ec_cost16[i];
                                            if s > bs {
                                                bs = s;
                                                best = i;
                                            }
                                        }
                                        pair_idx[pi] = best as u8;
                                        let (u0, u1) = (cb2[best][0], cb2[best][1]);
                                        dwc += h0 * c2[0].abs() * u0 + h1 * c2[1].abs() * u1;
                                        dcc += h0 * u0 * u0 + h1 * u1 * u1;
                                    }
                                    if dcc > 0.0 {
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
                                    for delta in [-128i32, -64, 0, 64, 128] {
                                        let cand_u16 = ((sf16 as i32 + delta).clamp(64, 0x7bc0) as u16) & 0xffc0;
                                        let c_sc = f16_to_f32(cand_u16);
                                        if c_sc <= 0.0 {
                                            continue;
                                        }
                                        let inv = 1.0 / c_sc;
                                        let eff_hl = 0.5
                                            * lam
                                            * (rms / (c_sc * c_sc).max(rms * 0.015)).min(16.0 * h_mean);
                                        let mut cand_idx = [0u8; 64];
                                        let (mut dwc, mut dcc, mut bit_sum) = (0.0f32, 0.0f32, 0.0f32);
                                        for (pi, c2) in g_in.chunks_exact(2).enumerate() {
                                            let (h0, h1) = (g_h[pi * 2], g_h[pi * 2 + 1]);
                                            let (a0, a1) = (h0 * c2[0].abs() * inv, h1 * c2[1].abs() * inv);
                                            let (mut best, mut bs) = (0usize, f32::NEG_INFINITY);
                                            for i in 0..16 {
                                                let s = a0 * cb2[i][0]
                                                    + a1 * cb2[i][1]
                                                    - h0 * cb2_hsq[i][0]
                                                    - h1 * cb2_hsq[i][1]
                                                    - eff_hl * ec_cost16[i];
                                                if s > bs {
                                                    bs = s;
                                                    best = i;
                                                }
                                            }
                                            cand_idx[pi] = best as u8;
                                            let (u0, u1) = (cb2[best][0], cb2[best][1]);
                                            dwc += h0 * c2[0].abs() * u0 + h1 * c2[1].abs() * u1;
                                            dcc += h0 * u0 * u0 + h1 * u1 * u1;
                                            bit_sum += ec_cost16[best];
                                        }
                                        let opt_u16 = if dcc > 0.0 {
                                            round_f16_12bit(f32_to_f16(dwc / dcc))
                                        } else {
                                            0
                                        };
                                        let opt_sc = f16_to_f32(opt_u16);
                                        let err_sum = (e0 - 2.0 * opt_sc * dwc + opt_sc * opt_sc * dcc).max(0.0);
                                        let j = err_sum + abs_lam * bit_sum;
                                        if j < best_j {
                                            best_j = j;
                                            best_u16 = opt_u16;
                                            best_pidx = cand_idx;
                                        }
                                    }
                                    sf16 = best_u16;
                                    sc = f16_to_f32(sf16);
                                    pair_idx = best_pidx;
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
                                            let base = sub * 8 + pr * 2;
                                            let raw_s2 = (g_in[base].is_sign_negative() as u8)
                                                | ((g_in[base + 1].is_sign_negative() as u8) << 1);
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
                        pw_step[lane * 2 + 1] = (idx_r[2][lane] as u32) | ((idx_r[3][lane] as u32) << 16);
                        pw_step[64 + lane] = (sgn_r[0][lane] as u32)
                            | ((sgn_r[1][lane] as u32) << 8)
                            | ((sgn_r[2][lane] as u32) << 16)
                            | ((sgn_r[3][lane] as u32) << 24);
                    }
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
