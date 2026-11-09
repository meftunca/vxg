//! Multimodal Vision (`VLM`) configuration, pure-Rust Base64 / data-URI / file loading,
//! `Qwen2VL` / `Qwen3.5-VL` aspect-preserving `smart_resize`, bicubic resampling, and
//! block-major `2x2` spatial-merge patchification.

use std::fs;
use std::path::Path;

use image::imageops::FilterType;
use serde_json::{json, Value};

use crate::llm_ops::{rms_norm_inplace, sigmoid};
use crate::vision::EncodedVisionImage;
use crate::{Error, Result};

/// Default maximum pixel budget per image (`768 * 32 * 32 = 786,432` px -> `<= 768` ViT patches
/// = `<= 192` merged LLM tokens), overrideable via `VXG_MAX_VISION_PIXELS`.
pub const DEFAULT_MIN_PIXELS: usize = 4 * 32 * 32;
pub const DEFAULT_MAX_PIXELS: usize = 768 * 32 * 32;

#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub image_token_id: u32,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl VisionConfig {
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    pub fn merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    pub fn grid_factor(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    pub fn to_json_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "out_hidden_size": self.out_hidden_size,
            "num_position_embeddings": self.num_position_embeddings,
            "rope_theta": self.rope_theta,
            "mrope_section": self.mrope_section,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "image_token_id": self.image_token_id,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }))
        .unwrap_or_default()
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let v: Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::Format(format!("invalid vision.config JSON: {e}")))?;
        let req_u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(Value::as_u64)
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| Error::Format(format!("missing {k} in vision.config")))
        };
        let depth = req_u("depth")?;
        let hidden_size = req_u("hidden_size")?;
        let intermediate_size = req_u("intermediate_size")?;
        let num_heads = req_u("num_heads")?;
        let head_dim = v
            .get("head_dim")
            .and_then(Value::as_u64)
            .map(|x| x as usize)
            .unwrap_or_else(|| hidden_size / num_heads.max(1));
        let mut mrope = [11usize, 11, 10];
        if let Some(arr) = v.get("mrope_section").and_then(Value::as_array) {
            if arr.len() == 3 {
                for (i, item) in arr.iter().enumerate() {
                    if let Some(val) = item.as_u64() {
                        mrope[i] = val as usize;
                    }
                }
            }
        }
        Ok(Self {
            depth,
            hidden_size,
            intermediate_size,
            num_heads,
            head_dim,
            in_channels: req_u("in_channels")?,
            patch_size: req_u("patch_size")?,
            temporal_patch_size: req_u("temporal_patch_size")?,
            spatial_merge_size: req_u("spatial_merge_size")?,
            out_hidden_size: req_u("out_hidden_size")?,
            num_position_embeddings: req_u("num_position_embeddings")?,
            rope_theta: v
                .get("rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(10_000.0) as f32,
            mrope_section: mrope,
            vision_start_token_id: v
                .get("vision_start_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(248053) as u32,
            vision_end_token_id: v
                .get("vision_end_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(248054) as u32,
            image_token_id: v
                .get("image_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(248056) as u32,
            min_pixels: v
                .get("min_pixels")
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .unwrap_or(DEFAULT_MIN_PIXELS),
            max_pixels: v
                .get("max_pixels")
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .unwrap_or(DEFAULT_MAX_PIXELS),
        })
    }
}

#[derive(Clone, Debug)]
pub struct PreprocessedImage {
    pub patches: Vec<f32>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    pub resized_h: usize,
    pub resized_w: usize,
}

impl PreprocessedImage {
    pub fn num_patches(&self) -> usize {
        self.grid_t * self.grid_h * self.grid_w
    }
}

pub fn decode_base64(input: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len() * 3) / 4);
    let mut buf = 0u32;
    let mut bits = 0u8;
    for b in input.bytes() {
        if b.is_ascii_whitespace() || b == b'=' {
            continue;
        }
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(Error::Format("invalid base64 character in image payload".into())),
        };
        buf = (buf << 6) | (val as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xff) as u8);
        }
    }
    Ok(out)
}

/// Loads raw encoded image bytes (PNG/JPEG/WebP/BMP/GIF) from a `data:image/...;base64,...` URI,
/// a `file://` URL, a local filesystem path, or a raw base64 string.
pub fn load_image_source(src: &str) -> Result<Vec<u8>> {
    let trimmed = src.trim();
    if let Some(rest) = trimmed.strip_prefix("data:") {
        let (_, b64) = rest
            .split_once(",")
            .ok_or_else(|| Error::Format("malformed data: URI (missing comma)".into()))?;
        return decode_base64(b64);
    }
    let path_cand = trimmed.strip_prefix("file://").unwrap_or(trimmed);
    if Path::new(path_cand).is_file() {
        return fs::read(path_cand).map_err(|e| Error::Io(format!("{path_cand}: {e}")));
    }
    if trimmed.len() > 64 && !trimmed.contains('/') {
        return decode_base64(trimmed);
    }
    if trimmed.len() > 128 && !Path::new(path_cand).exists() {
        if let Ok(decoded) = decode_base64(trimmed) {
            if decoded.len() >= 16 {
                return Ok(decoded);
            }
        }
    }
    Err(Error::Io(format!(
        "image source not found or unsupported (expected local file path or data:image/...;base64,... URI): '{trimmed}'"
    )))
}

/// Exact port of HuggingFace `Qwen2VLImageProcessor.smart_resize`.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::InvalidShape("zero image dimension in smart_resize"));
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(Error::Format("image aspect ratio exceeds 200:1 limit".into()));
    }
    let mut h_bar = ((h / f).round() as usize).max(1) * factor;
    let mut w_bar = ((w / f).round() as usize).max(1) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / (max_pixels as f64)).sqrt();
        h_bar = (((h / beta / f).floor() as usize).max(1)) * factor;
        w_bar = (((w / beta / f).floor() as usize).max(1)) * factor;
    } else if h_bar * w_bar < min_pixels {
        let beta = ((min_pixels as f64) / (h * w)).sqrt();
        h_bar = (((h * beta / f).ceil() as usize).max(1)) * factor;
        w_bar = (((w * beta / f).ceil() as usize).max(1)) * factor;
    }
    Ok((h_bar, w_bar))
}

/// Decodes, resizes, normalizes (`(x / 255 - 0.5) / 0.5`), and patchifies an image in exact
/// `Qwen3.5-VL` block-major `[grid_h/m, grid_w/m, m, m, C, T, P, P]` order.
pub fn preprocess_image_bytes(bytes: &[u8], cfg: &VisionConfig) -> Result<PreprocessedImage> {
    let dyn_img = image::load_from_memory(bytes)
        .map_err(|e| Error::Format(format!("failed to decode image: {e}")))?;
    let (orig_w, orig_h) = (dyn_img.width() as usize, dyn_img.height() as usize);
    let max_px = std::env::var("VXG_MAX_VISION_PIXELS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(cfg.max_pixels)
        .max(cfg.min_pixels);
    let factor = cfg.grid_factor();
    let (resized_h, resized_w) = smart_resize(orig_h, orig_w, factor, cfg.min_pixels, max_px)?;

    let resized = if orig_h == resized_h && orig_w == resized_w {
        dyn_img.to_rgb8()
    } else {
        dyn_img
            .resize_exact(resized_w as u32, resized_h as u32, FilterType::CatmullRom)
            .to_rgb8()
    };
    let raw = resized.as_raw();

    let (p, m, t_rep, ch) = (
        cfg.patch_size,
        cfg.spatial_merge_size,
        cfg.temporal_patch_size,
        cfg.in_channels,
    );
    if ch != 3 {
        return Err(Error::Unsupported("only 3-channel RGB vision encoders supported".into()));
    }
    let grid_h = resized_h / p;
    let grid_w = resized_w / p;
    let blocks_h = grid_h / m;
    let blocks_w = grid_w / m;
    let patch_dim = cfg.patch_dim();
    let mut patches = vec![0.0f32; grid_h * grid_w * patch_dim];

    // Normalize `(u8 / 255.0 - 0.5) / 0.5 == u8 * (2.0 / 255.0) - 1.0`
    const SCALE: f32 = 2.0 / 255.0;
    let mut patch_idx = 0usize;
    for bh in 0..blocks_h {
        for bw in 0..blocks_w {
            for mh in 0..m {
                for mw in 0..m {
                    let py0 = (bh * m + mh) * p;
                    let px0 = (bw * m + mw) * p;
                    let dst = &mut patches[patch_idx * patch_dim..(patch_idx + 1) * patch_dim];
                    let mut k = 0usize;
                    for c in 0..3 {
                        for _t in 0..t_rep {
                            for dy in 0..p {
                                let row_off = ((py0 + dy) * resized_w + px0) * 3 + c;
                                for dx in 0..p {
                                    dst[k] = (raw[row_off + dx * 3] as f32).mul_add(SCALE, -1.0);
                                    k += 1;
                                }
                            }
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }

    Ok(PreprocessedImage {
        patches,
        grid_t: 1,
        grid_h,
        grid_w,
        resized_h,
        resized_w,
    })
}

/// Computes 3D `(pos_t, pos_h, pos_w, 0)` `M-RoPE` coordinates for `tok_ids` starting at logical
/// RoPE position `start_rope_pos = (kv_pos_before as isize + prev_rope_delta) as usize`.
pub fn compute_mrope_positions(
    tok_ids: &[u32],
    images: &[&EncodedVisionImage],
    cfg: &VisionConfig,
    kv_pos_before: usize,
    start_rope_pos: usize,
) -> Result<(Vec<[u32; 4]>, isize)> {
    let mut out = Vec::with_capacity(tok_ids.len());
    let mut cur = start_rope_pos;
    let mut img_idx = 0usize;
    let mut i = 0usize;
    let m = cfg.spatial_merge_size.max(1);

    while i < tok_ids.len() {
        if tok_ids[i] == cfg.image_token_id {
            let img = *images.get(img_idx).ok_or_else(|| {
                Error::Format("more <|image_pad|> spans in prompt than encoded images".into())
            })?;
            let hm = img.grid_h / m;
            let wm = img.grid_w / m;
            let n_tok = img.grid_t * hm * wm;
            if i + n_tok > tok_ids.len()
                || tok_ids[i..i + n_tok].iter().any(|&t| t != cfg.image_token_id)
            {
                return Err(Error::Format(
                    "<|image_pad|> token count mismatch against EncodedVisionImage".into(),
                ));
            }
            for t in 0..img.grid_t {
                for rh in 0..hm {
                    for rw in 0..wm {
                        out.push([(cur + t) as u32, (cur + rh) as u32, (cur + rw) as u32, 0]);
                    }
                }
            }
            cur += img.grid_t.max(hm).max(wm);
            img_idx += 1;
            i += n_tok;
        } else {
            let p = cur as u32;
            out.push([p, p, p, 0]);
            cur += 1;
            i += 1;
        }
    }
    if img_idx != images.len() {
        return Err(Error::Format(
            "fewer <|image_pad|> spans in prompt than provided images".into(),
        ));
    }
    let total_kv = (kv_pos_before + tok_ids.len()) as isize;
    Ok((out, (cur as isize) - total_kv))
}

pub fn apply_mrope_neox(
    vec: &mut [f32],
    pos_thw: [u32; 3],
    rope_dim: usize,
    freq_base: f32,
    mrope_section: [usize; 3],
) {
    let rdim = rope_dim.min(vec.len());
    let half = rdim / 2;
    let h_lim = mrope_section[1] * 3;
    let w_lim = mrope_section[2] * 3;
    for i in 0..half {
        let axis = if i < h_lim && i % 3 == 1 {
            1
        } else if i < w_lim && i % 3 == 2 {
            2
        } else {
            0
        };
        let freq = freq_base.powf(-(2.0 * (i as f32)) / (rdim as f32));
        let (sin_t, cos_t) = ((pos_thw[axis] as f32) * freq).sin_cos();
        let (x0, x1) = (vec[i], vec[i + half]);
        vec[i] = x0 * cos_t - x1 * sin_t;
        vec[i + half] = x0 * sin_t + x1 * cos_t;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn step_full_gqa_mrope(
    q_raw: &[f32],
    k_cur: &mut [f32],
    v_cur: &[f32],
    q_norm: Option<&[f32]>,
    k_norm: Option<&[f32]>,
    gated_q: bool,
    nh: usize,
    nkv: usize,
    hd: usize,
    kv_pos: usize,
    pos_thw: [u32; 3],
    mrope_section: [usize; 3],
    rope_dim: usize,
    rope_freq_base: f32,
    eps: f32,
    k_cache: &mut Vec<f32>,
    v_cache: &mut Vec<f32>,
    att_cat: &mut [f32],
) {
    let mut q_cur = vec![0.0f32; nh * hd];
    let mut q_gate = vec![0.0f32; if gated_q { nh * hd } else { 0 }];
    if gated_q {
        for h in 0..nh {
            q_cur[h * hd..(h + 1) * hd].copy_from_slice(&q_raw[h * 2 * hd..h * 2 * hd + hd]);
            q_gate[h * hd..(h + 1) * hd]
                .copy_from_slice(&q_raw[h * 2 * hd + hd..(h + 1) * 2 * hd]);
        }
    } else {
        q_cur.copy_from_slice(q_raw);
    }
    if let Some(qn) = q_norm {
        for h in 0..nh {
            rms_norm_inplace(&mut q_cur[h * hd..(h + 1) * hd], qn, eps);
        }
    }
    if let Some(kn) = k_norm {
        for h in 0..nkv {
            rms_norm_inplace(&mut k_cur[h * hd..(h + 1) * hd], kn, eps);
        }
    }
    for h in 0..nh {
        apply_mrope_neox(&mut q_cur[h * hd..(h + 1) * hd], pos_thw, rope_dim, rope_freq_base, mrope_section);
    }
    for h in 0..nkv {
        apply_mrope_neox(&mut k_cur[h * hd..(h + 1) * hd], pos_thw, rope_dim, rope_freq_base, mrope_section);
    }
    k_cache.extend_from_slice(k_cur);
    v_cache.extend_from_slice(v_cur);
    let seq_len = kv_pos + 1;
    let kv_stride = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];
    for h in 0..nh {
        let kv_h = h / (nh / nkv);
        let qh = &q_cur[h * hd..(h + 1) * hd];
        let mut max_s = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let kh = &k_cache[t * kv_stride + kv_h * hd..t * kv_stride + (kv_h + 1) * hd];
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot = qh[d].mul_add(kh[d], dot);
            }
            let s = dot * scale;
            scores[t] = s;
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        let out_h = &mut att_cat[h * hd..(h + 1) * hd];
        out_h.fill(0.0);
        for t in 0..seq_len {
            let w = scores[t] * inv_sum;
            let vh = &v_cache[t * kv_stride + kv_h * hd..t * kv_stride + (kv_h + 1) * hd];
            for d in 0..hd {
                out_h[d] = w.mul_add(vh[d], out_h[d]);
            }
        }
        if gated_q {
            let gh = &q_gate[h * hd..(h + 1) * hd];
            for d in 0..hd {
                out_h[d] *= sigmoid(gh[d]);
            }
        }
    }
}
