//! Wavelet lifting transforms (`CDF 5/3` reversible integer for lossless, `CDF 9/7`
//! biorthogonal lifting for high-efficiency lossy) + multi-level 2D pyramid.
//!
//! Both transforms work for any tile dimension `w >= 1, h >= 1` (even and odd sizes) with
//! symmetric boundary extension and exact unity DC gain so every pyramid level
//! (`L0 = 1/8, L1 = 1/4, L2 = 1/2, L3 = 1/1`) has 1:1 pixel brightness without rescaling.

use crate::simd_dwt;

const A97: f32 = -1.586_134_3;
const B97: f32 = -0.052_980_12;
const G97: f32 = 0.882_911_1;
const D97: f32 = 0.443_506_87;
const K97: f32 = 1.230_174_1;
const INV_K97: f32 = 1.0 / K97;

/// 1D forward DC-unity `CDF 9/7` lifting: splits `data[0..n]` in-place into `[low(0..nl), high(0..nh)]`.
#[inline]
pub fn dwt97_1d_forward(data: &mut [f32], scratch: &mut [f32]) {
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

    for i in 0..nh {
        let l = low[i];
        let r = low[(i + 1).min(nl - 1)];
        high[i] += A97 * (l + r);
    }
    for i in 0..nl {
        let dl = if i == 0 { high[0] } else { high[i - 1] };
        let dr = if i < nh { high[i] } else { high[nh - 1] };
        low[i] += B97 * (dl + dr);
    }
    for i in 0..nh {
        let l = low[i];
        let r = low[(i + 1).min(nl - 1)];
        high[i] += G97 * (l + r);
    }
    for i in 0..nl {
        let dl = if i == 0 { high[0] } else { high[i - 1] };
        let dr = if i < nh { high[i] } else { high[nh - 1] };
        low[i] += D97 * (dl + dr);
    }
    // Exact unity DC gain: low /= K97, high *= K97 (det = 1, flat regions stay unchanged!)
    for v in low.iter_mut() {
        *v *= INV_K97;
    }
    for v in high.iter_mut() {
        *v *= K97;
    }

    data[..nl].copy_from_slice(low);
    data[nl..n].copy_from_slice(high);
}

/// 1D inverse DC-unity `CDF 9/7` lifting: reconstructs `[low(0..nl), high(0..nh)]` in-place.
#[inline]
pub fn dwt97_1d_inverse(data: &mut [f32], scratch: &mut [f32]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let nl = n.div_ceil(2);
    let nh = n / 2;
    let (low, high) = scratch[..n].split_at_mut(nl);

    for i in 0..nl {
        low[i] = data[i] * K97;
    }
    for i in 0..nh {
        high[i] = data[nl + i] * INV_K97;
    }

    for i in 0..nl {
        let dl = if i == 0 { high[0] } else { high[i - 1] };
        let dr = if i < nh { high[i] } else { high[nh - 1] };
        low[i] -= D97 * (dl + dr);
    }
    for i in 0..nh {
        let l = low[i];
        let r = low[(i + 1).min(nl - 1)];
        high[i] -= G97 * (l + r);
    }
    for i in 0..nl {
        let dl = if i == 0 { high[0] } else { high[i - 1] };
        let dr = if i < nh { high[i] } else { high[nh - 1] };
        low[i] -= B97 * (dl + dr);
    }
    for i in 0..nh {
        let l = low[i];
        let r = low[(i + 1).min(nl - 1)];
        high[i] -= A97 * (l + r);
    }

    for i in 0..nl {
        data[i * 2] = low[i];
    }
    for i in 0..nh {
        data[i * 2 + 1] = high[i];
    }
}

pub fn dwt97_2d_forward_step_with(
    buf: &mut [f32],
    stride: usize,
    w: usize,
    h: usize,
    row_scratch: &mut [f32],
    vert_tmp: &mut [f32],
) {
    if w > 1 {
        for y in 0..h {
            simd_dwt::dwt97_1d_forward_simd(&mut buf[y * stride..y * stride + w], row_scratch);
        }
    }
    if h > 1 {
        simd_dwt::dwt97_2d_vertical_forward(buf, stride, w, h, vert_tmp);
    }
}

pub fn dwt97_2d_inverse_step_with(
    buf: &mut [f32],
    stride: usize,
    w: usize,
    h: usize,
    row_scratch: &mut [f32],
    vert_tmp: &mut [f32],
) {
    if h > 1 {
        simd_dwt::dwt97_2d_vertical_inverse(buf, stride, w, h, vert_tmp);
    }
    if w > 1 {
        for y in 0..h {
            simd_dwt::dwt97_1d_inverse_simd(&mut buf[y * stride..y * stride + w], row_scratch);
        }
    }
}

pub fn dwt97_2d_forward_step(buf: &mut [f32], stride: usize, w: usize, h: usize) {
    let mut row_scratch = vec![0.0f32; w.max(h)];
    let mut vert_tmp = vec![0.0f32; w * h];
    dwt97_2d_forward_step_with(buf, stride, w, h, &mut row_scratch, &mut vert_tmp);
}

pub fn dwt97_2d_inverse_step(buf: &mut [f32], stride: usize, w: usize, h: usize) {
    let mut row_scratch = vec![0.0f32; w.max(h)];
    let mut vert_tmp = vec![0.0f32; w * h];
    dwt97_2d_inverse_step_with(buf, stride, w, h, &mut row_scratch, &mut vert_tmp);
}

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

/// 1-level 2D forward `CDF 5/3` using reusable scratch buffers and SIMD vertical lifting.
pub fn dwt53_2d_forward_step_with(
    buf: &mut [i32],
    stride: usize,
    w: usize,
    h: usize,
    row_scratch: &mut [i32],
    vert_tmp: &mut [i32],
) {
    if w > 1 {
        for y in 0..h {
            simd_dwt::dwt53_1d_forward_simd(&mut buf[y * stride..y * stride + w], row_scratch);
        }
    }
    if h > 1 {
        simd_dwt::dwt53_2d_vertical_forward(buf, stride, w, h, vert_tmp);
    }
}

/// 1-level 2D inverse `CDF 5/3` using reusable scratch buffers and SIMD horizontal + vertical lifting.
pub fn dwt53_2d_inverse_step_with(
    buf: &mut [i32],
    stride: usize,
    w: usize,
    h: usize,
    row_scratch: &mut [i32],
    vert_tmp: &mut [i32],
) {
    if h > 1 {
        simd_dwt::dwt53_2d_vertical_inverse(buf, stride, w, h, vert_tmp);
    }
    if w > 1 {
        for y in 0..h {
            simd_dwt::dwt53_1d_inverse_simd(&mut buf[y * stride..y * stride + w], row_scratch);
        }
    }
}

/// 1-level 2D forward `CDF 5/3` on the top-left `(w, h)` region inside a buffer of row `stride`.
pub fn dwt53_2d_forward_step(buf: &mut [i32], stride: usize, w: usize, h: usize) {
    let mut row_scratch = vec![0i32; w.max(h)];
    let mut vert_tmp = vec![0i32; w * h];
    dwt53_2d_forward_step_with(buf, stride, w, h, &mut row_scratch, &mut vert_tmp);
}

/// 1-level 2D inverse `CDF 5/3` on the top-left `(w, h)` region inside a buffer of row `stride`.
pub fn dwt53_2d_inverse_step(buf: &mut [i32], stride: usize, w: usize, h: usize) {
    let mut row_scratch = vec![0i32; w.max(h)];
    let mut vert_tmp = vec![0i32; w * h];
    dwt53_2d_inverse_step_with(buf, stride, w, h, &mut row_scratch, &mut vert_tmp);
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
    /// Forward `n_dwt` levels of 2D `CDF 5/3` decomposition (bit-exact lossless).
    pub fn forward(plane: &[i16], w: usize, h: usize, n_dwt: usize) -> Self {
        Self::forward_mode(plane, w, h, n_dwt, false)
    }

    /// Forward `n_dwt` levels of 2D wavelet decomposition (`use_97 = true` for lossy `CDF 9/7`,
    /// `false` for bit-exact integer `CDF 5/3`).
    pub fn forward_mode(plane: &[i16], w: usize, h: usize, n_dwt: usize, use_97: bool) -> Self {
        let mut dims = Vec::with_capacity(n_dwt);
        let (mut cw, mut ch) = (w, h);
        let buf: Vec<i32> = if use_97 {
            let mut fbuf: Vec<f32> = plane.iter().map(|&v| v as f32).collect();
            let mut row_s = vec![0.0f32; w.max(h)];
            let mut vert_t = vec![0.0f32; w * h];
            for _ in 0..n_dwt {
                dims.push((cw, ch));
                dwt97_2d_forward_step_with(&mut fbuf, w, cw, ch, &mut row_s, &mut vert_t);
                cw = cw.div_ceil(2);
                ch = ch.div_ceil(2);
            }
            simd_dwt::f32_to_i32_round_slice(&fbuf)
        } else {
            let mut ibuf: Vec<i32> = plane.iter().map(|&v| v as i32).collect();
            let mut row_s = vec![0i32; w.max(h)];
            let mut vert_t = vec![0i32; w * h];
            for _ in 0..n_dwt {
                dims.push((cw, ch));
                dwt53_2d_forward_step_with(&mut ibuf, w, cw, ch, &mut row_s, &mut vert_t);
                cw = cw.div_ceil(2);
                ch = ch.div_ceil(2);
            }
            ibuf
        };
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

    /// Reconstructs the tile using `CDF 5/3` up to `target_level`.
    pub fn reconstruct(
        &self,
        target_level: usize,
        detail_gains: &[f32],
    ) -> (Vec<i16>, usize, usize) {
        self.reconstruct_mode(target_level, detail_gains, false)
    }

    /// Reconstructs the tile up to `target_level` (`0` = coarsest `LL`, `n_dwt` = full resolution).
    /// `detail_gains[k]` (`k = 0..target_level`) scales `(lh, hl, hh)` directly in subband domain
    /// (`1.0` = identity, `< 1.0` = blur, `> 1.0` = sharpen).
    pub fn reconstruct_mode(
        &self,
        target_level: usize,
        detail_gains: &[f32],
        use_97: bool,
    ) -> (Vec<i16>, usize, usize) {
        let lvl = target_level.min(self.details.len());
        if lvl == 0 {
            let out: Vec<i16> = self.ll.iter().map(|&v| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16).collect();
            return (out, self.ll_w, self.ll_h);
        }
        let (final_w, final_h) = (self.details[lvl - 1].out_w, self.details[lvl - 1].out_h);
        if use_97 {
            let mut fbuf = vec![0.0f32; final_w * final_h];
            let mut row_s = vec![0.0f32; final_w.max(final_h)];
            let mut vert_t = vec![0.0f32; final_w * final_h];
            for y in 0..self.ll_h {
                for x in 0..self.ll_w {
                    fbuf[y * final_w + x] = self.ll[y * self.ll_w + x] as f32;
                }
            }
            for k in 0..lvl {
                let d = &self.details[k];
                let gain = detail_gains.get(k).copied().unwrap_or(1.0);
                let lw = d.out_w.div_ceil(2);
                let hw = d.out_w / 2;
                let lh_rows = d.out_h.div_ceil(2);
                let hh_rows = d.out_h / 2;
                for y in 0..hh_rows {
                    let dst = &mut fbuf[(lh_rows + y) * final_w..(lh_rows + y) * final_w + lw];
                    for (dv, &sv) in dst.iter_mut().zip(&d.lh[y * lw..(y + 1) * lw]) {
                        *dv = sv as f32 * gain;
                    }
                }
                for y in 0..lh_rows {
                    let dst = &mut fbuf[y * final_w + lw..y * final_w + d.out_w];
                    for (dv, &sv) in dst.iter_mut().zip(&d.hl[y * hw..(y + 1) * hw]) {
                        *dv = sv as f32 * gain;
                    }
                }
                for y in 0..hh_rows {
                    let dst = &mut fbuf[(lh_rows + y) * final_w + lw..(lh_rows + y) * final_w + d.out_w];
                    for (dv, &sv) in dst.iter_mut().zip(&d.hh[y * hw..(y + 1) * hw]) {
                        *dv = sv as f32 * gain;
                    }
                }
                dwt97_2d_inverse_step_with(&mut fbuf, final_w, d.out_w, d.out_h, &mut row_s, &mut vert_t);
            }
            let out = simd_dwt::f32_to_i16_round_slice(&fbuf);
            return (out, final_w, final_h);
        }
        let mut buf = vec![0i32; final_w * final_h];
        let mut row_s = vec![0i32; final_w.max(final_h)];
        let mut vert_t = vec![0i32; final_w * final_h];
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

            dwt53_2d_inverse_step_with(&mut buf, final_w, d.out_w, d.out_h, &mut row_s, &mut vert_t);
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

    #[test]
    fn cdf97_2d_has_unity_dc_and_high_accuracy() {
        for (w, h, lvls) in [(128, 128, 3), (37, 53, 3)] {
            // Flat constant tile must stay exact across all levels (unity DC gain)!
            let flat = vec![128i16; w * h];
            let wt_f = WaveletTile::forward_mode(&flat, w, h, lvls, true);
            for &v in &wt_f.ll {
                assert!((v - 128).abs() <= 1, "DC drift in LL: {v}");
            }
            let (rec_f, _, _) = wt_f.reconstruct_mode(lvls, &[], true);
            assert!(rec_f.iter().all(|&v| (v - 128).abs() <= 1));

            let src: Vec<i16> = (0..w * h).map(|i| (((i * 19 + 7) % 256) as i16) - 128).collect();
            let wt = WaveletTile::forward_mode(&src, w, h, lvls, true);
            let (rec, rw, rh) = wt.reconstruct_mode(lvls, &[], true);
            assert_eq!((rw, rh), (w, h));
            let max_err = src.iter().zip(rec.iter()).map(|(&a, &b)| (a - b).abs()).max().unwrap_or(0);
            assert!(max_err <= 2, "CDF 9/7 roundtrip max_err={max_err} on {w}x{h}");
        }
    }
}
