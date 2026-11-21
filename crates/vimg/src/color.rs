//! Reversible integer `YCoCg-R` lifting color transform (`4:4:4` lossless & `4:2:0` lossy)
//! with ARMv8 NEON / x86_64 AVX2 SIMD acceleration.

use crate::simd;

/// Forward reversible `YCoCg-R` lifting (Malvar & Sullivan): exact integer inverse for all RGB.
#[inline(always)]
pub fn rgb_to_ycocg_r(r: u8, g: u8, b: u8) -> (i16, i16, i16) {
    let r = r as i16;
    let g = g as i16;
    let b = b as i16;
    let co = r - b;
    let t = b + (co >> 1);
    let cg = g - t;
    let y = t + (cg >> 1);
    (y, co, cg)
}

/// Exact inverse of `rgb_to_ycocg_r` with clamping for lossy quantization tolerance.
#[inline(always)]
pub fn ycocg_r_to_rgb(y: i16, co: i16, cg: i16) -> (u8, u8, u8) {
    let t = y - (cg >> 1);
    let g = cg + t;
    let b = t - (co >> 1);
    let r = b + co;
    (r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)
}

/// Converts packed RGB (`channels == 3`) or RGBA (`channels == 4`) into planar `(Y, Co, Cg, Option<A>)`.
pub fn split_planes_444(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> (Vec<i16>, Vec<i16>, Vec<i16>, Option<Vec<i16>>) {
    let n = width * height;
    let mut y_p = vec![0i16; n];
    let mut co_p = vec![0i16; n];
    let mut cg_p = vec![0i16; n];
    if channels == 1 {
        for i in 0..n {
            y_p[i] = pixels[i] as i16;
        }
        return (y_p, co_p, cg_p, None);
    }
    let mut a_p = if channels == 4 { Some(vec![255i16; n]) } else { None };
    if let Some(a_vec) = a_p.as_mut() {
        for i in 0..n {
            let o = i * 4;
            let (y, co, cg) = rgb_to_ycocg_r(pixels[o], pixels[o + 1], pixels[o + 2]);
            y_p[i] = y;
            co_p[i] = co;
            cg_p[i] = cg;
            a_vec[i] = pixels[o + 3] as i16;
        }
    } else {
        simd::split_rgb_ycocg_r(pixels, &mut y_p, &mut co_p, &mut cg_p);
    }
    (y_p, co_p, cg_p, a_p)
}

/// 2x2 box downsample for `4:2:0` chroma planes (`cw = width.div_ceil(2)`, `ch = height.div_ceil(2)`).
pub fn downsample_2x2(src: &[i16], width: usize, height: usize) -> (Vec<i16>, usize, usize) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut dst = vec![0i16; cw * ch];
    for cy in 0..ch {
        let y0 = cy * 2;
        let y1 = (y0 + 1).min(height - 1);
        let row0 = &src[y0 * width..(y0 + 1) * width];
        let row1 = &src[y1 * width..(y1 + 1) * width];
        let out_row = &mut dst[cy * cw..(cy + 1) * cw];
        simd::downsample_2x2_row(row0, row1, out_row);
    }
    (dst, cw, ch)
}

/// Fast bilinear upsample from `4:2:0` chroma `(cw, ch)` back to `(width, height)`.
pub fn upsample_2x2_bilinear(
    src: &[i16],
    cw: usize,
    ch: usize,
    width: usize,
    height: usize,
) -> Vec<i16> {
    let mut dst = vec![0i16; width * height];
    if cw == 0 || ch == 0 {
        return dst;
    }
    for y in 0..height {
        let cy0 = (y / 2).min(ch - 1);
        let cy1 = if (y & 1) == 0 {
            cy0.saturating_sub(1)
        } else {
            (cy0 + 1).min(ch - 1)
        };
        let r0 = &src[cy0 * cw..(cy0 + 1) * cw];
        let r1 = &src[cy1 * cw..(cy1 + 1) * cw];
        let out_row = &mut dst[y * width..(y + 1) * width];
        simd::upsample_2x2_bilinear_row(r0, r1, out_row);
    }
    dst
}

/// Merges decoded `YCoCg-R` (or grayscale `Y`-only) planes back into packed RGB / RGBA bytes.
pub fn merge_planes_rgb(
    y_p: &[i16],
    co_p: Option<&[i16]>,
    cg_p: Option<&[i16]>,
    a_p: Option<&[i16]>,
    width: usize,
    height: usize,
    out_channels: usize,
) -> Vec<u8> {
    let n = width * height;
    let mut out = vec![0u8; n * out_channels];
    match (co_p, cg_p, out_channels) {
        (None, _, 1) => {
            for i in 0..n {
                out[i] = y_p[i].clamp(0, 255) as u8;
            }
        }
        (None, _, 3) => {
            for (i, &yv) in y_p.iter().enumerate().take(n) {
                let g = yv.clamp(0, 255) as u8;
                let o = i * 3;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
            }
        }
        (None, _, 4) => {
            for i in 0..n {
                let g = y_p[i].clamp(0, 255) as u8;
                let a = a_p.map_or(255u8, |ap| ap[i].clamp(0, 255) as u8);
                let o = i * 4;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
                out[o + 3] = a;
            }
        }
        (Some(co), Some(cg), 3) => {
            simd::merge_ycocg_r_rgb(y_p, co, cg, &mut out);
        }
        (Some(co), Some(cg), 4) => {
            for i in 0..n {
                let (r, g, b) = ycocg_r_to_rgb(y_p[i], co[i], cg[i]);
                let a = a_p.map_or(255u8, |ap| ap[i].clamp(0, 255) as u8);
                let o = i * 4;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
                out[o + 3] = a;
            }
        }
        _ => {
            for i in 0..n {
                out[i] = y_p[i].clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// Forward reversible `YCoCg-R` lifting for 10-bit / 12-bit HDR (`0..=4095`, fits safely in `i16`).
#[inline(always)]
pub fn rgb_to_ycocg_r_u16(r: u16, g: u16, b: u16) -> (i16, i16, i16) {
    let r = r as i16;
    let g = g as i16;
    let b = b as i16;
    let co = r - b;
    let t = b + (co >> 1);
    let cg = g - t;
    let y = t + (cg >> 1);
    (y, co, cg)
}

/// Exact inverse of `rgb_to_ycocg_r_u16` clamped to `0..=max_val`.
#[inline(always)]
pub fn ycocg_r_to_rgb_u16(y: i16, co: i16, cg: i16, max_val: i16) -> (u16, u16, u16) {
    let t = y - (cg >> 1);
    let g = cg + t;
    let b = t - (co >> 1);
    let r = b + co;
    (r.clamp(0, max_val) as u16, g.clamp(0, max_val) as u16, b.clamp(0, max_val) as u16)
}

/// Converts packed 10-bit or 12-bit `u16` RGB / RGBA / Gray pixels into planar `(Y, Co, Cg, Option<A>)`.
pub fn split_planes_444_u16(
    pixels: &[u16],
    width: usize,
    height: usize,
    channels: usize,
    max_val: u16,
) -> (Vec<i16>, Vec<i16>, Vec<i16>, Option<Vec<i16>>) {
    let n = width * height;
    let mut y_p = vec![0i16; n];
    let mut co_p = vec![0i16; n];
    let mut cg_p = vec![0i16; n];
    if channels == 1 {
        for i in 0..n {
            y_p[i] = pixels[i].min(max_val) as i16;
        }
        return (y_p, co_p, cg_p, None);
    }
    let mut a_p = if channels == 4 { Some(vec![max_val as i16; n]) } else { None };
    for i in 0..n {
        let o = i * channels;
        let (y, co, cg) = rgb_to_ycocg_r_u16(
            pixels[o].min(max_val),
            pixels[o + 1].min(max_val),
            pixels[o + 2].min(max_val),
        );
        y_p[i] = y;
        co_p[i] = co;
        cg_p[i] = cg;
        if let Some(a_vec) = a_p.as_mut() {
            a_vec[i] = pixels[o + 3].min(max_val) as i16;
        }
    }
    (y_p, co_p, cg_p, a_p)
}

/// Merges decoded `YCoCg-R` (or grayscale `Y`-only) planes back into packed 10-bit/12-bit `u16` RGB/RGBA.
#[allow(clippy::too_many_arguments)]
pub fn merge_planes_u16(
    y_p: &[i16],
    co_p: Option<&[i16]>,
    cg_p: Option<&[i16]>,
    a_p: Option<&[i16]>,
    width: usize,
    height: usize,
    out_channels: usize,
    max_val: i16,
) -> Vec<u16> {
    let n = width * height;
    let mut out = vec![0u16; n * out_channels];
    match (co_p, cg_p, out_channels) {
        (None, _, 1) => {
            for i in 0..n {
                out[i] = y_p[i].clamp(0, max_val) as u16;
            }
        }
        (None, _, 3) => {
            for (i, &yv) in y_p.iter().enumerate().take(n) {
                let g = yv.clamp(0, max_val) as u16;
                let o = i * 3;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
            }
        }
        (None, _, 4) => {
            for i in 0..n {
                let g = y_p[i].clamp(0, max_val) as u16;
                let a = a_p.map_or(max_val as u16, |ap| ap[i].clamp(0, max_val) as u16);
                let o = i * 4;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
                out[o + 3] = a;
            }
        }
        (Some(co), Some(cg), 3) => {
            for i in 0..n {
                let (r, g, b) = ycocg_r_to_rgb_u16(y_p[i], co[i], cg[i], max_val);
                let o = i * 3;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
            }
        }
        (Some(co), Some(cg), 4) => {
            for i in 0..n {
                let (r, g, b) = ycocg_r_to_rgb_u16(y_p[i], co[i], cg[i], max_val);
                let a = a_p.map_or(max_val as u16, |ap| ap[i].clamp(0, max_val) as u16);
                let o = i * 4;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
                out[o + 3] = a;
            }
        }
        _ => {
            for i in 0..n {
                out[i] = y_p[i].clamp(0, max_val) as u16;
            }
        }
    }
    out
}

/// Forward reversible `YCoCg-R` lifting in `i32` for full 16-bit HDR (`0..=65535`, zero overflow).
#[inline(always)]
pub fn rgb_to_ycocg_r_i32(r: u16, g: u16, b: u16) -> (i32, i32, i32) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;
    let co = r - b;
    let t = b + (co >> 1);
    let cg = g - t;
    let y = t + (cg >> 1);
    (y, co, cg)
}

/// Exact inverse of `rgb_to_ycocg_r_i32` clamped to `0..=max_val` (`max_val` up to `65535`).
#[inline(always)]
pub fn ycocg_r_to_rgb_i32(y: i32, co: i32, cg: i32, max_val: i32) -> (u16, u16, u16) {
    let t = y - (cg >> 1);
    let g = cg + t;
    let b = t - (co >> 1);
    let r = b + co;
    (r.clamp(0, max_val) as u16, g.clamp(0, max_val) as u16, b.clamp(0, max_val) as u16)
}

/// 2x2 box downsample in `i32` for 16-bit `4:2:0` chroma planes.
pub fn downsample_2x2_i32(src: &[i32], width: usize, height: usize) -> (Vec<i32>, usize, usize) {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut dst = vec![0i32; cw * ch];
    for cy in 0..ch {
        let y0 = cy * 2;
        let y1 = (y0 + 1).min(height - 1);
        let row0 = &src[y0 * width..(y0 + 1) * width];
        let row1 = &src[y1 * width..(y1 + 1) * width];
        let out_row = &mut dst[cy * cw..(cy + 1) * cw];
        for (cx, dv) in out_row.iter_mut().enumerate() {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(width - 1);
            *dv = (row0[x0] + row0[x1] + row1[x0] + row1[x1] + 2) >> 2;
        }
    }
    (dst, cw, ch)
}

/// Bilinear 2x2 upsample in `i32` for 16-bit `4:2:0` chroma planes (zero `i16` overflow).
pub fn upsample_2x2_bilinear_i32(
    src: &[i32],
    cw: usize,
    ch: usize,
    width: usize,
    height: usize,
) -> Vec<i32> {
    let mut dst = vec![0i32; width * height];
    if cw == 0 || ch == 0 {
        return dst;
    }
    for y in 0..height {
        let cy0 = (y / 2).min(ch - 1);
        let cy1 = if (y & 1) == 0 {
            cy0.saturating_sub(1)
        } else {
            (cy0 + 1).min(ch - 1)
        };
        let r0 = &src[cy0 * cw..(cy0 + 1) * cw];
        let r1 = &src[cy1 * cw..(cy1 + 1) * cw];
        let out_row = &mut dst[y * width..(y + 1) * width];
        for (x, dv) in out_row.iter_mut().enumerate() {
            let cx0 = (x / 2).min(cw - 1);
            let cx1 = if (x & 1) == 0 {
                cx0.saturating_sub(1)
            } else {
                (cx0 + 1).min(cw - 1)
            };
            let v = 9 * r0[cx0] + 3 * r0[cx1] + 3 * r1[cx0] + r1[cx1] + 8;
            *dv = v >> 4;
        }
    }
    dst
}

/// Merges decoded 16-bit `i32` `YCoCg-R` (or grayscale `Y`-only) planes into packed `u16` RGB/RGBA.
#[allow(clippy::too_many_arguments)]
pub fn merge_planes_u16_i32(
    y_p: &[i32],
    co_p: Option<&[i32]>,
    cg_p: Option<&[i32]>,
    a_p: Option<&[i32]>,
    width: usize,
    height: usize,
    out_channels: usize,
    max_val: i32,
) -> Vec<u16> {
    let n = width * height;
    let mut out = vec![0u16; n * out_channels];
    match (co_p, cg_p, out_channels) {
        (None, _, 1) => {
            for i in 0..n {
                out[i] = y_p[i].clamp(0, max_val) as u16;
            }
        }
        (None, _, 3) => {
            for (i, &yv) in y_p.iter().enumerate().take(n) {
                let g = yv.clamp(0, max_val) as u16;
                let o = i * 3;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
            }
        }
        (None, _, 4) => {
            for i in 0..n {
                let g = y_p[i].clamp(0, max_val) as u16;
                let a = a_p.map_or(max_val as u16, |ap| ap[i].clamp(0, max_val) as u16);
                let o = i * 4;
                out[o] = g;
                out[o + 1] = g;
                out[o + 2] = g;
                out[o + 3] = a;
            }
        }
        (Some(co), Some(cg), 3) => {
            for i in 0..n {
                let (r, g, b) = ycocg_r_to_rgb_i32(y_p[i], co[i], cg[i], max_val);
                let o = i * 3;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
            }
        }
        (Some(co), Some(cg), 4) => {
            for i in 0..n {
                let (r, g, b) = ycocg_r_to_rgb_i32(y_p[i], co[i], cg[i], max_val);
                let a = a_p.map_or(max_val as u16, |ap| ap[i].clamp(0, max_val) as u16);
                let o = i * 4;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
                out[o + 3] = a;
            }
        }
        _ => {
            for i in 0..n {
                out[i] = y_p[i].clamp(0, max_val) as u16;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ycocg_r_is_bit_exact_across_all_boundaries() {
        for r in (0..=255).step_by(5) {
            for g in (0..=255).step_by(5) {
                for b in (0..=255).step_by(5) {
                    let (y, co, cg) = rgb_to_ycocg_r(r as u8, g as u8, b as u8);
                    assert_eq!(ycocg_r_to_rgb(y, co, cg), (r as u8, g as u8, b as u8));
                }
            }
        }
        // Verify 10-bit & 12-bit HDR YCoCg-R exactness
        for max_v in [1023u16, 4095u16] {
            let step = (max_v / 31).max(1) as usize;
            for r in (0..=max_v).step_by(step) {
                for g in (0..=max_v).step_by(step) {
                    for b in (0..=max_v).step_by(step) {
                        let (y, co, cg) = rgb_to_ycocg_r_u16(r, g, b);
                        assert_eq!(ycocg_r_to_rgb_u16(y, co, cg, max_v as i16), (r, g, b));
                    }
                }
            }
        }
    }
}
