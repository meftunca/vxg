//! `CB4-v2` Activation-Aware Joint Scale / Codebook / Sparse-Outlier Matrix Encoder:
//! - Truncated non-outlier bulk training with population-weighted (`n_q`) cubic Lloyd-Max (`0-ULP`
//!   fixed-point compatible with GPU `fit_cubic_codebook`).
//! - 2-round EM group-scale (`SCALE_GRID` + weighted LSQ) & global cubic codebook co-optimization.
//! - Coprime-quantile L2 outlier ranking (`|w - s_g * c_q| * col_amp[c]`) + discrete `FP16`
//!   (`+-1 ULP`) non-outlier scale search + exhaustive 16-code `FP16`-residual rounding on outliers.

use crate::codebook::CodebookMatrix;
use crate::fp16::{f16_to_f32, f32_to_f16, fit_cubic_codebook};
use crate::{Error, Result};

const SCALE_GRID: [f32; 9] = [0.78, 0.84, 0.90, 0.95, 1.00, 1.05, 1.11, 1.18, 1.26];

#[inline(always)]
fn nearest16(val: f32, mids: &[f32; 15]) -> usize {
    let mut base = if val >= mids[7] { 8usize } else { 0usize };
    if val >= mids[base + 3] { base += 4; }
    if val >= mids[base + 1] { base += 2; }
    if val >= mids[base] { base += 1; }
    base
}

#[inline]
fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn fit_cubic_weighted_16(emp: &[f32; 16], counts: &[f64; 16]) -> [f32; 16] {
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];
    for q in 0..16 {
        let (w, u, y) = (counts[q].max(1.0), (q as f64) - 7.5, emp[q] as f64);
        let b = [1.0, u, u * u, u * u * u];
        for r in 0..4 {
            atb[r] += w * b[r] * y;
            for c in 0..4 { ata[r][c] += w * b[r] * b[c]; }
        }
    }
    let mut aug = [[0.0f64; 5]; 4];
    for r in 0..4 {
        aug[r][..4].copy_from_slice(&ata[r]);
        aug[r][4] = atb[r];
    }
    for col in 0..4 {
        let mut piv = col;
        for r in (col + 1)..4 {
            if aug[r][col].abs() > aug[piv][col].abs() { piv = r; }
        }
        aug.swap(col, piv);
        let diag = aug[col][col];
        if diag.abs() > 1e-18 {
            for c in col..5 { aug[col][c] /= diag; }
            for r in 0..4 {
                if r != col {
                    let f = aug[r][col];
                    for c in col..5 { aug[r][c] -= f * aug[col][c]; }
                }
            }
        }
    }
    let (a0, a1, a2, a3) = (aug[0][4] as f32, aug[1][4] as f32, aug[2][4] as f32, aug[3][4] as f32);
    let mut out = [0.0f32; 16];
    for (q, slot) in out.iter_mut().enumerate() {
        let u = (q as f32) - 7.5;
        let u2 = u * u;
        *slot = (a3.mul_add(u2, a1)).mul_add(u, a2.mul_add(u2, a0));
    }
    out
}

fn train_lloyd_cubic_bulk(samples: &mut [f32]) -> ([f32; 16], [f32; 15]) {
    let mut cents = [0.0f32; 16];
    samples.sort_by(f32::total_cmp);
    let n = samples.len();
    if n < 16 || (samples[n - 1] - samples[0]).abs() <= 1e-12 {
        for (i, slot) in cents.iter_mut().enumerate() { *slot = -1.0 + (2.0 * (i as f32)) / 15.0; }
    } else {
        let mut pref = Vec::with_capacity(n + 1);
        pref.push(0.0f64);
        let mut run = 0.0f64;
        for &v in samples.iter() { run += v as f64; pref.push(run); }
        for (i, slot) in cents.iter_mut().enumerate() { *slot = samples[((2 * i + 1) * n) / 32]; }
        for i in 1..16 { if cents[i] <= cents[i - 1] { cents[i] = cents[i - 1] + 1e-4; } }
        for _ in 0..28 {
            let mut bounds = [0usize; 17];
            bounds[16] = n;
            for b in 1..16 { bounds[b] = samples.partition_point(|&v| v < 0.5 * (cents[b - 1] + cents[b])); }
            let (mut counts, mut raw) = ([1.0f64; 16], cents);
            for c in 0..16 {
                let (lo, hi) = (bounds[c], bounds[c + 1]);
                if hi > lo {
                    counts[c] = (hi - lo) as f64;
                    raw[c] = ((pref[hi] - pref[lo]) / counts[c]) as f32;
                }
            }
            cents = fit_cubic_weighted_16(&raw, &counts);
            for i in 1..16 { if cents[i] <= cents[i - 1] { cents[i] = cents[i - 1] + 1e-4; } }
        }
    }
    let (_, c_out) = fit_cubic_codebook(&cents);
    let mut mids = [0.0f32; 15];
    for i in 0..15 { mids[i] = 0.5 * (c_out[i] + c_out[i + 1]); }
    (c_out, mids)
}

#[inline]
fn par_row_chunks<T: Send, F: Fn(usize, &mut [T]) + Sync>(
    buf: &mut [T],
    rows: usize,
    elems_per_row: usize,
    workers: usize,
    f: F,
) {
    if workers <= 1 {
        f(0, buf);
    } else {
        let rpw = rows.div_ceil(workers);
        std::thread::scope(|s| {
            for (ci, ch) in buf.chunks_mut(rpw * elems_per_row).enumerate() {
                let f_ref = &f;
                s.spawn(move || f_ref(ci * rpw, ch));
            }
        });
    }
}

pub fn encode_cb4_optimized(
    rows: usize,
    cols: usize,
    weights: &[f32],
    col_amp: Option<&[f32]>,
    target_outlier_frac: f64,
) -> Result<CodebookMatrix> {
    if rows == 0 || cols == 0 || weights.len() != rows.saturating_mul(cols) {
        return Err(Error::InvalidShape(
            "invalid rows*cols for encode_cb4_optimized",
        ));
    }
    let padded_cols = cols.div_ceil(256) * 256;
    let gpr = padded_cols / 128;
    let total_groups = rows * gpr;
    let total_numel = rows * cols;
    let allow_outliers = cols <= 65_536;
    let eff_frac = if allow_outliers {
        target_outlier_frac.clamp(0.0, 0.05)
    } else {
        0.0
    };

    let unit_amp;
    let amp: &[f32] = match col_amp {
        Some(s) if s.len() == cols => s,
        _ => {
            unit_amp = vec![1.0f32; cols];
            &unit_amp
        }
    };
    let amp_sq: Vec<f64> = amp.iter().map(|&a| (a as f64) * (a as f64)).collect();

    let workers = if rows >= 128 && total_numel >= 262_144 {
        std::thread::available_parallelism()
            .map(|n| n.get().clamp(1, 10))
            .unwrap_or(4)
            .min(rows)
    } else {
        1
    };

    // Pass 1: Initial trimmed-RMS group scales
    let mut sc_f32 = vec![0.0f32; total_groups];
    par_row_chunks(&mut sc_f32, rows, gpr, workers, |r0, sc_slice| {
        for (dr, row_sc) in sc_slice.chunks_exact_mut(gpr).enumerate() {
            let row_w = &weights[(r0 + dr) * cols..(r0 + dr + 1) * cols];
            for (g, slot) in row_sc.iter_mut().enumerate() {
                let c0 = g * 128;
                let valid = cols.saturating_sub(c0).min(128);
                if valid == 0 {
                    continue;
                }
                let sl = &row_w[c0..c0 + valid];
                let mut sq0 = 0.0f32;
                for &v in sl {
                    if v.is_finite() {
                        sq0 = v.mul_add(v, sq0);
                    }
                }
                let rms0 = (sq0 / (valid as f32)).sqrt();
                if rms0 == 0.0 {
                    continue;
                }
                let cut = 2.8 * rms0;
                let (mut csq, mut cn) = (0.0f32, 0usize);
                for &v in sl {
                    if v.is_finite() && v.abs() <= cut {
                        csq = v.mul_add(v, csq);
                        cn += 1;
                    }
                }
                let rc = if cn >= 8 {
                    (csq / (cn as f32)).sqrt().max(rms0 * 0.15)
                } else {
                    rms0
                };
                *slot = f16_to_f32(f32_to_f16(rc));
            }
        }
    });

    // Pass 2: 2-Round Truncated-Bulk Cubic Lloyd-Max + Bulk Group-Scale EM
    let sample_target = 131_072usize;
    let mut stride = (total_numel / sample_target).max(1);
    while stride > 1 && gcd_usize(stride, cols) != 1 {
        stride += 1;
    }

    let z_bulk_cut = if eff_frac > 0.0 {
        let mut abs_z = Vec::with_capacity(sample_target.min(total_numel) + 128);
        let mut idx = 0usize;
        while idx < total_numel {
            let (r, c) = (idx / cols, idx % cols);
            let sc = sc_f32[r * gpr + c / 128];
            let v = weights[idx];
            abs_z.push(if sc > 0.0 && v.is_finite() {
                (v / sc).abs().min(16.0)
            } else {
                0.0
            });
            idx += stride;
        }
        let kth = (((1.0 - eff_frac * 0.85) * (abs_z.len() as f64)) as usize)
            .min(abs_z.len().saturating_sub(1));
        let (_, &mut zc, _) = abs_z.select_nth_unstable_by(kth, f32::total_cmp);
        zc.clamp(1.6, 8.0)
    } else {
        8.0f32
    };

    let gather_bulk = |sc_vec: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(sample_target.min(total_numel) + 128);
        let mut idx = 0usize;
        while idx < total_numel {
            let (r, c) = (idx / cols, idx % cols);
            let sc = sc_vec[r * gpr + c / 128];
            let v = weights[idx];
            if sc > 0.0 && v.is_finite() {
                let z = v / sc;
                if z.abs() <= z_bulk_cut {
                    out.push(z.clamp(-8.0, 8.0));
                }
            } else {
                out.push(0.0);
            }
            idx += stride;
        }
        out
    };

    let mut samp = gather_bulk(&sc_f32);
    let (mut cents, mut mids) = train_lloyd_cubic_bulk(&mut samp);

    let opt_scales_bulk = |sc_vec: &mut [f32], c_lut: &[f32; 16], m_lut: &[f32; 15]| {
        par_row_chunks(sc_vec, rows, gpr, workers, |r0, sc_slice| {
            for (dr, row_sc) in sc_slice.chunks_exact_mut(gpr).enumerate() {
                let row_w = &weights[(r0 + dr) * cols..(r0 + dr + 1) * cols];
                for (g, slot) in row_sc.iter_mut().enumerate() {
                    let sc0 = *slot;
                    if sc0 <= 0.0 {
                        continue;
                    }
                    let c0 = g * 128;
                    let valid = cols.saturating_sub(c0).min(128);
                    let max_w = z_bulk_cut * sc0;
                    let (mut best_err, mut best_s) = (f64::INFINITY, sc0);
                    for &mul in &SCALE_GRID {
                        let sc = sc0 * mul;
                        let inv_sc = 1.0 / sc;
                        let (mut num, mut den) = (0.0f64, 0.0f64);
                        for i in 0..valid {
                            let w = row_w[c0 + i];
                            if w.is_finite() && w.abs() <= max_w {
                                let cw2 = amp_sq[c0 + i];
                                let cq = c_lut[nearest16(w * inv_sc, m_lut)] as f64;
                                num += cw2 * (w as f64) * cq;
                                den += cw2 * cq * cq;
                            }
                        }
                        if den <= 1e-24 {
                            continue;
                        }
                        let cs = f16_to_f32(f32_to_f16((num / den) as f32));
                        if !cs.is_finite() || cs <= 0.0 {
                            continue;
                        }
                        let inv_cs = 1.0 / cs;
                        let mut err = 0.0f64;
                        for i in 0..valid {
                            let w = row_w[c0 + i];
                            if w.is_finite() && w.abs() <= max_w {
                                let d = (w - cs * c_lut[nearest16(w * inv_cs, m_lut)]) as f64;
                                err += amp_sq[c0 + i] * (d * d);
                            }
                        }
                        if err < best_err {
                            best_err = err;
                            best_s = cs;
                        }
                    }
                    *slot = best_s;
                }
            }
        });
    };

    opt_scales_bulk(&mut sc_f32, &cents, &mids);
    let mut samp2 = gather_bulk(&sc_f32);
    let (c2, m2) = train_lloyd_cubic_bulk(&mut samp2);
    cents = c2;
    mids = m2;
    opt_scales_bulk(&mut sc_f32, &cents, &mids);

    // Pass 3: Sink-Row-Capped Coprime-Quantile Threshold + Final Joint Discrete Optimization
    let cap_sc = {
        let g_step = (total_groups / 65_536).max(1);
        let mut pos_sc: Vec<f32> = sc_f32.iter().step_by(g_step).copied().filter(|&s| s > 0.0).collect();
        if pos_sc.is_empty() {
            1.0f32
        } else {
            let mid = pos_sc.len() / 2;
            let (_, &mut med, _) = pos_sc.select_nth_unstable_by(mid, f32::total_cmp);
            (1.45 * med).max(1e-12)
        }
    };
    let w_thresh = if eff_frac > 0.0 {
        let err_target = 1_048_576usize;
        let mut e_stride = (total_numel / err_target).max(1);
        while e_stride > 1 && gcd_usize(e_stride, cols) != 1 { e_stride += 1; }
        let mut errs = Vec::with_capacity(err_target.min(total_numel) + 128);
        let mut idx = 0usize;
        while idx < total_numel {
            let (r, c) = (idx / cols, idx % cols);
            let sc = sc_f32[r * gpr + c / 128];
            let w = weights[idx];
            if w.is_finite() && sc > 0.0 {
                let damp = if sc > cap_sc { cap_sc / sc } else { 1.0 };
                let q = nearest16(w / sc, &mids);
                errs.push((w - sc * cents[q]).abs() * damp * amp[c]);
            } else {
                errs.push(0.0);
            }
            idx += e_stride;
        }
        let kth = (((1.0 - eff_frac) * (errs.len() as f64)) as usize).min(errs.len().saturating_sub(1));
        let (_, &mut th, _) = errs.select_nth_unstable_by(kth, f32::total_cmp);
        th.max(1e-8)
    } else {
        f32::INFINITY
    };

    let words_per_row = gpr * 16;
    let mut scales_f16 = vec![0u16; total_groups];
    let mut packed_words = vec![0u32; rows * words_per_row];
    let mut row_out_counts = vec![0u32; rows];

    let finalize_chunk =
        |r0: usize, sf_slice: &mut [u16], pw_slice: &mut [u32], rc_slice: &mut [u32]| -> Vec<u32> {
            let n_rows = rc_slice.len();
            let est_cap = (((n_rows * cols) as f64) * (eff_frac * 1.15)) as usize + 64;
            let mut out_vec = Vec::with_capacity(est_cap);
            let mut q_grp = [0u8; 128];
            let mut is_out = [false; 128];
            let mut grp_cand = [(0.0f32, 0u8); 128];

            for dr in 0..n_rows {
                let r = r0 + dr;
                let row_w = &weights[r * cols..(r + 1) * cols];
                let row_sf = &mut sf_slice[dr * gpr..(dr + 1) * gpr];
                let row_pw = &mut pw_slice[dr * words_per_row..(dr + 1) * words_per_row];
                let out_before = out_vec.len();

                for g in 0..gpr {
                    let sc0 = sc_f32[r * gpr + g];
                    let c0 = g * 128;
                    let valid = cols.saturating_sub(c0).min(128);
                    q_grp.fill(0);
                    if sc0 <= 0.0 || valid == 0 {
                        continue;
                    }
                    is_out.fill(false);
                    if eff_frac > 0.0 {
                        let inv0 = 1.0 / sc0;
                        let damp = if sc0 > cap_sc { cap_sc / sc0 } else { 1.0 };
                        let mut nc = 0usize;
                        for i in 0..valid {
                            let w = row_w[c0 + i];
                            if w.is_finite() {
                                let score = (w - sc0 * cents[nearest16(w * inv0, &mids)]).abs() * damp * amp[c0 + i];
                                if score > w_thresh {
                                    grp_cand[nc] = (score, i as u8);
                                    nc += 1;
                                }
                            }
                        }
                        if nc > 8 {
                            grp_cand[..nc].select_nth_unstable_by(nc - 8, |a, b| a.0.total_cmp(&b.0));
                            for &(_, idx_i) in &grp_cand[nc - 8..nc] { is_out[idx_i as usize] = true; }
                        } else {
                            for &(_, idx_i) in &grp_cand[..nc] { is_out[idx_i as usize] = true; }
                        }
                    }

                    let mut best_sf16 = f32_to_f16(sc0);
                    let mut best_err = f64::INFINITY;
                    for &mul in &SCALE_GRID {
                        let sc = sc0 * mul;
                        let inv_sc = 1.0 / sc;
                        let (mut num, mut den) = (0.0f64, 0.0f64);
                        for i in 0..valid {
                            if !is_out[i] {
                                let w = row_w[c0 + i];
                                if w.is_finite() {
                                    let cw2 = amp_sq[c0 + i];
                                    let cq = cents[nearest16(w * inv_sc, &mids)] as f64;
                                    num += cw2 * (w as f64) * cq;
                                    den += cw2 * cq * cq;
                                }
                            }
                        }
                        let base_h = if den > 1e-24 {
                            f32_to_f16((num / den) as f32)
                        } else {
                            f32_to_f16(sc)
                        };
                        for dh in -1i16..=1i16 {
                            let ch = base_h.saturating_add_signed(dh);
                            if (ch & 0x8000) != 0 {
                                continue;
                            }
                            let cs = f16_to_f32(ch);
                            if !cs.is_finite() || cs <= 0.0 {
                                continue;
                            }
                            let inv_cs = 1.0 / cs;
                            let mut err = 0.0f64;
                            for i in 0..valid {
                                if !is_out[i] {
                                    let w = row_w[c0 + i];
                                    if w.is_finite() {
                                        let d =
                                            (w - cs * cents[nearest16(w * inv_cs, &mids)]) as f64;
                                        err += amp_sq[c0 + i] * (d * d);
                                    }
                                }
                            }
                            if err < best_err {
                                best_err = err;
                                best_sf16 = ch;
                            }
                        }
                    }

                    row_sf[g] = best_sf16;
                    let final_sc = f16_to_f32(best_sf16);
                    let inv_fsc = if final_sc > 0.0 { 1.0 / final_sc } else { 0.0 };
                    for i in 0..valid {
                        let w = row_w[c0 + i];
                        if !w.is_finite() || final_sc <= 0.0 {
                            q_grp[i] = 0;
                        } else if !is_out[i] {
                            q_grp[i] = nearest16(w * inv_fsc, &mids) as u8;
                        } else {
                            let (mut best_q, mut best_rf16, mut best_ae) = (0u8, 0u16, f32::INFINITY);
                            for q in 0..16 {
                                let base = final_sc * cents[q];
                                let rf16 = f32_to_f16(w - base);
                                let ae = (w - (base + f16_to_f32(rf16))).abs();
                                if ae < best_ae {
                                    best_ae = ae;
                                    best_q = q as u8;
                                    best_rf16 = rf16;
                                }
                            }
                            q_grp[i] = best_q;
                            if (best_rf16 & 0x7fff) != 0 {
                                out_vec.push(((c0 + i) as u32) | ((best_rf16 as u32) << 16));
                            }
                        }
                    }

                    for sub in 0..4 {
                        let off = (g * 4 + sub) * 4;
                        let qb = sub * 32;
                        for w_idx in 0..4 {
                            let mut word = 0u32;
                            for k in 0..8 {
                                word |= (q_grp[qb + w_idx * 8 + k] as u32) << (k * 4);
                            }
                            row_pw[off + w_idx] = word;
                        }
                    }
                }
                rc_slice[dr] = (out_vec.len() - out_before) as u32;
            }
            out_vec
        };

    let chunk_outs: Vec<Vec<u32>> = if workers <= 1 {
        vec![finalize_chunk(0, &mut scales_f16, &mut packed_words, &mut row_out_counts)]
    } else {
        let rpw = rows.div_ceil(workers);
        std::thread::scope(|s| {
            let handles: Vec<_> = scales_f16
                .chunks_mut(rpw * gpr)
                .enumerate()
                .zip(packed_words.chunks_mut(rpw * words_per_row).zip(row_out_counts.chunks_mut(rpw)))
                .map(|((ci, sf_c), (pw_c, rc_c))| s.spawn(move || finalize_chunk(ci * rpw, sf_c, pw_c, rc_c)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };

    let mut outliers_packed = Vec::with_capacity(chunk_outs.iter().map(Vec::len).sum());
    for co in chunk_outs {
        outliers_packed.extend_from_slice(&co);
    }
    let mut outlier_row_ptr = Vec::with_capacity(rows + 1);
    outlier_row_ptr.push(0u32);
    let mut running = 0u32;
    for &cnt in &row_out_counts {
        running = running
            .checked_add(cnt)
            .ok_or_else(|| Error::Format("CB4 outlier CSR u32 overflow".into()))?;
        outlier_row_ptr.push(running);
    }
    Ok(CodebookMatrix {
        bits: 4,
        rows,
        cols,
        padded_cols,
        centroids: cents,
        scales_f16,
        packed_words,
        outlier_row_ptr,
        outliers_packed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::generate_realistic_llm_weights;
    use crate::cb4_calib::normalize_amp;
    use crate::codebook::Q4_0Matrix;
    use crate::DType;

    fn act_sqnr(ref_w: &[f32], dec: &[f32], amp: &[f32], cols: usize) -> f64 {
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (i, (&rv, &av)) in ref_w.iter().zip(dec).enumerate() {
            let wt = (amp[i % cols] as f64).powi(2);
            sig += wt * (rv as f64).powi(2);
            err += wt * ((rv - av) as f64).powi(2);
        }
        10.0 * (sig / err).log10()
    }

    #[test]
    fn cubic_codebook_is_zero_ulp_fixed_point_and_handles_sink_columns() {
        let (rows, cols) = (544usize, 512usize);
        let (_, weights) = generate_realistic_llm_weights(DType::BF16, rows, cols, 20261108);
        let mut amp = vec![1.0f32; cols];
        for c in [17, 145, 273, 401] {
            amp[c] = 5.0;
        }
        normalize_amp(&mut amp);

        let cb_amp = encode_cb4_optimized(rows, cols, &weights, Some(&amp), 0.0074).unwrap();
        let cb_uni = encode_cb4_optimized(rows, cols, &weights, None, 0.0074).unwrap();
        let q4 = Q4_0Matrix::encode(rows, cols, &weights);
        let (_, gpu_cents) = fit_cubic_codebook(&cb_amp.centroids);
        for q in 0..16 {
            let ulp = (cb_amp.centroids[q].to_bits() as i64 - gpu_cents[q].to_bits() as i64).abs();
            assert!(ulp <= 1, "expected <= 1 ULP on q={q}, got {ulp}");
        }
        let s_amp = act_sqnr(&weights, &cb_amp.decode_f32(), &amp, cols);
        let s_uni = act_sqnr(&weights, &cb_uni.decode_f32(), &amp, cols);
        let s_q4 = act_sqnr(&weights, &q4.decode_f32(), &amp, cols);
        assert!(s_amp > s_uni + 0.15 && s_amp > s_q4 + 4.0, "{s_amp:.2} vs {s_uni:.2} vs {s_q4:.2}");
    }
}
