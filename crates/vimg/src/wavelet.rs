//! Reversible integer Le Gall `CDF 5/3` wavelet lifting (1D & multi-level 2D pyramid).
//!
//! Works for any tile dimension `w >= 1, h >= 1` (both even and odd sizes) with symmetric
//! boundary extension, guaranteeing exact bit-for-bit lossless reconstruction (`step = 1`)
//! and native zero-overhead resolution pyramids (`L0 = 1/8, L1 = 1/4, L2 = 1/2, L3 = 1/1`).

/// 1D forward integer `CDF 5/3` lifting: splits `data[0..n]` in-place into `[low(0..nl), high(0..nh)]`.
#[inline]
pub fn dwt53_1d_forward(data: &mut [i32], scratch: &mut [i32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);

    for i in 0..nl {
        low[i] = data[i * 2];
    }
    for i in 0..nh {
        high[i] = data[i * 2 + 1];
    }

    // Predict step: d[i] = odd[i] - ((even[i] + even[min(i+1, nl-1)]) >> 1)
    for i in 0..nh {
        let l = low[i];
        let r = low[(i + 1).min(nl - 1)];
        high[i] -= (l + r) >> 1;
    }

    // Update step: s[i] = even[i] + ((d_left + d_right + 2) >> 2)
    for i in 0..nl {
        let dl = if i == 0 { high[0] } else { high[i - 1] };
        let dr = if i < nh { high[i] } else { high[nh - 1] };
        low[i] += (dl + dr + 2) >> 2;
    }

    data[..nl].copy_from_slice(low);
    data[nl..n].copy_from_slice(high);
}

/// Exact 1D inverse integer `CDF 5/3` lifting: reconstructs `[low(0..nl), high(0..nh)]` in-place.
#[inline]
pub fn dwt53_1d_inverse(data: &mut [i32], scratch: &mut [i32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);
    low.copy_from_slice(&data[..nl]);
    high.copy_from_slice(&data[nl..n]);

    // Undo Update
    for i in 0..nl {
        let dl = if i == 0 { high[0] } else { high[i - 1] };
        let dr = if i < nh { high[i] } else { high[nh - 1] };
        low[i] -= (dl + dr + 2) >> 2;
    }

    // Undo Predict
    for i in 0..nh {
        let l = low[i];
        let r = low[(i + 1).min(nl - 1)];
        high[i] += (l + r) >> 1;
    }

    for i in 0..nl {
        data[i * 2] = low[i];
    }
    for i in 0..nh {
        data[i * 2 + 1] = high[i];
    }
}

/// Dimensions `(w, h)` of the LL approximation after `steps` downsampling steps.
#[inline]
pub fn level_dims(w: usize, h: usize, steps: usize) -> (usize, usize) {
    let mut cw = w;
    let mut ch = h;
    for _ in 0..steps {
        cw = cw.div_ceil(2);
        ch = ch.div_ceil(2);
    }
    (cw, ch)
}

/// 1-level 2D forward `CDF 5/3` on the top-left `(w, h)` region inside a buffer of row `stride`.
pub fn dwt53_2d_forward_step(buf: &mut [i32], stride: usize, w: usize, h: usize) {
    let mut scratch = vec![0i32; w.max(h)];
    if w > 1 {
        for y in 0..h {
            let row = &mut buf[y * stride..y * stride + w];
            dwt53_1d_forward(row, &mut scratch);
        }
    }
    if h > 1 {
        let mut col = vec![0i32; h];
        for x in 0..w {
            for y in 0..h {
                col[y] = buf[y * stride + x];
            }
            dwt53_1d_forward(&mut col, &mut scratch);
            for y in 0..h {
                buf[y * stride + x] = col[y];
            }
        }
    }
}

/// 1-level 2D inverse `CDF 5/3` on the top-left `(w, h)` region inside a buffer of row `stride`.
pub fn dwt53_2d_inverse_step(buf: &mut [i32], stride: usize, w: usize, h: usize) {
    let mut scratch = vec![0i32; w.max(h)];
    if h > 1 {
        let mut col = vec![0i32; h];
        for x in 0..w {
            for y in 0..h {
                col[y] = buf[y * stride + x];
            }
            dwt53_1d_inverse(&mut col, &mut scratch);
            for y in 0..h {
                buf[y * stride + x] = col[y];
            }
        }
    }
    if w > 1 {
        for y in 0..h {
            let row = &mut buf[y * stride..y * stride + w];
            dwt53_1d_inverse(row, &mut scratch);
        }
    }
}

/// Decomposed multi-level wavelet tile for one color plane.
/// - `ll`: coarsest Level 0 approximation `(ll_w, ll_h)`.
/// - `details[k]` (`k = 0..n_dwt_levels`, coarsest -> finest): `(lh, hl, hh)` subbands needed to
///   reconstruct resolution level `k + 1`.
#[derive(Clone, Debug)]
pub struct WaveletTile {
    pub full_w: usize,
    pub full_h: usize,
    pub ll_w: usize,
    pub ll_h: usize,
    pub ll: Vec<i32>,
    pub details: Vec<DetailBands>,
}

#[derive(Clone, Debug)]
pub struct DetailBands {
    pub out_w: usize,
    pub out_h: usize,
    pub lh: Vec<i32>, // size: lw * hh
    pub hl: Vec<i32>, // size: hw * lh
    pub hh: Vec<i32>, // size: hw * hh
}

impl WaveletTile {
    /// Forward `n_dwt` levels of 2D `CDF 5/3` decomposition.
    pub fn forward(plane: &[i16], w: usize, h: usize, n_dwt: usize) -> Self {
        let mut buf: Vec<i32> = plane.iter().map(|&v| v as i32).collect();
        let mut dims = Vec::with_capacity(n_dwt);
        let (mut cw, mut ch) = (w, h);
        for _ in 0..n_dwt {
            dims.push((cw, ch));
            dwt53_2d_forward_step(&mut buf, w, cw, ch);
            cw = cw.div_ceil(2);
            ch = ch.div_ceil(2);
        }
        let (ll_w, ll_h) = (cw, ch);
        let mut ll = Vec::with_capacity(ll_w * ll_h);
        for y in 0..ll_h {
            ll.extend_from_slice(&buf[y * w..y * w + ll_w]);
        }

        // Extract details in coarsest -> finest order so Level 1..=n_dwt stream progressively!
        let mut details = Vec::with_capacity(n_dwt);
        for &(ow, oh) in dims.iter().rev() {
            let lw = ow.div_ceil(2);
            let hw = ow / 2;
            let lh_rows = oh.div_ceil(2);
            let hh_rows = oh / 2;

            let mut lh = Vec::with_capacity(lw * hh_rows);
            for y in lh_rows..oh {
                lh.extend_from_slice(&buf[y * w..y * w + lw]);
            }
            let mut hl = Vec::with_capacity(hw * lh_rows);
            for y in 0..lh_rows {
                hl.extend_from_slice(&buf[y * w + lw..y * w + ow]);
            }
            let mut hh = Vec::with_capacity(hw * hh_rows);
            for y in lh_rows..oh {
                hh.extend_from_slice(&buf[y * w + lw..y * w + ow]);
            }
            details.push(DetailBands { out_w: ow, out_h: oh, lh, hl, hh });
        }

        Self { full_w: w, full_h: h, ll_w, ll_h, ll, details }
    }

    /// Reconstructs the tile up to `target_level` (`0` = coarsest `LL`, `n_dwt` = full resolution).
    /// `detail_gains[k]` (`k = 0..target_level`) scales `(lh, hl, hh)` directly in subband domain
    /// (`1.0` = identity, `< 1.0` = blur, `> 1.0` = sharpen).
    pub fn reconstruct(
        &self,
        target_level: usize,
        detail_gains: &[f32],
    ) -> (Vec<i16>, usize, usize) {
        let lvl = target_level.min(self.details.len());
        if lvl == 0 {
            let out: Vec<i16> = self.ll.iter().map(|&v| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16).collect();
            return (out, self.ll_w, self.ll_h);
        }
        let (final_w, final_h) = (self.details[lvl - 1].out_w, self.details[lvl - 1].out_h);
        let mut buf = vec![0i32; final_w * final_h];
        for y in 0..self.ll_h {
            buf[y * final_w..y * final_w + self.ll_w]
                .copy_from_slice(&self.ll[y * self.ll_w..(y + 1) * self.ll_w]);
        }

        for k in 0..lvl {
            let d = &self.details[k];
            let gain = detail_gains.get(k).copied().unwrap_or(1.0);
            let lw = d.out_w.div_ceil(2);
            let hw = d.out_w / 2;
            let lh_rows = d.out_h.div_ceil(2);
            let hh_rows = d.out_h / 2;

            if (gain - 1.0).abs() < 1e-6 {
                for y in 0..hh_rows {
                    let row = &mut buf[(lh_rows + y) * final_w..(lh_rows + y) * final_w + lw];
                    row.copy_from_slice(&d.lh[y * lw..(y + 1) * lw]);
                }
                for y in 0..lh_rows {
                    let row = &mut buf[y * final_w + lw..y * final_w + d.out_w];
                    row.copy_from_slice(&d.hl[y * hw..(y + 1) * hw]);
                }
                for y in 0..hh_rows {
                    let row = &mut buf[(lh_rows + y) * final_w + lw..(lh_rows + y) * final_w + d.out_w];
                    row.copy_from_slice(&d.hh[y * hw..(y + 1) * hw]);
                }
            } else if gain <= 1e-6 {
                for y in 0..hh_rows {
                    buf[(lh_rows + y) * final_w..(lh_rows + y) * final_w + d.out_w].fill(0);
                }
                for y in 0..lh_rows {
                    buf[y * final_w + lw..y * final_w + d.out_w].fill(0);
                }
            } else {
                let g_q8 = (gain * 256.0).round() as i32;
                for y in 0..hh_rows {
                    let dst = &mut buf[(lh_rows + y) * final_w..(lh_rows + y) * final_w + lw];
                    let src = &d.lh[y * lw..(y + 1) * lw];
                    for (dv, &sv) in dst.iter_mut().zip(src) {
                        *dv = (sv * g_q8 + 128) >> 8;
                    }
                }
                for y in 0..lh_rows {
                    let dst = &mut buf[y * final_w + lw..y * final_w + d.out_w];
                    let src = &d.hl[y * hw..(y + 1) * hw];
                    for (dv, &sv) in dst.iter_mut().zip(src) {
                        *dv = (sv * g_q8 + 128) >> 8;
                    }
                }
                for y in 0..hh_rows {
                    let dst = &mut buf[(lh_rows + y) * final_w + lw..(lh_rows + y) * final_w + d.out_w];
                    let src = &d.hh[y * hw..(y + 1) * hw];
                    for (dv, &sv) in dst.iter_mut().zip(src) {
                        *dv = (sv * g_q8 + 128) >> 8;
                    }
                }
            }

            dwt53_2d_inverse_step(&mut buf, final_w, d.out_w, d.out_h);
        }

        let out: Vec<i16> = buf.into_iter().map(|v| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16).collect();
        (out, final_w, final_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf53_2d_is_bit_exact_for_even_and_odd_tiles() {
        for (w, h, lvls) in [(128, 128, 3), (64, 64, 3), (37, 53, 3), (13, 7, 2), (1, 1, 3)] {
            let mut src = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    src.push((((x * 17 + y * 31) % 256) as i16) - 128);
                }
            }
            let wt = WaveletTile::forward(&src, w, h, lvls);
            let (rec, rw, rh) = wt.reconstruct(lvls, &[]);
            assert_eq!((rw, rh), (w, h));
            assert_eq!(rec, src, "mismatch on {w}x{h} lvls={lvls}");
        }
    }
}
