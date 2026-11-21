//! Panic-safe C ABI (`libvimg.a` / `libvimg.dylib` / `.so` / `.dll`) and zero-WASI `wasm32` exports.

use core::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    decode_from_source, decode_query, encode, format::HEADER_BYTES, plan_query_ranges,
    EncodeOptions, SparseRangeSource, TransformQuery, VimgHeader, VimgIndex,
};

const VERSION_CSTR: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

#[repr(C)]
pub struct VimgBuf {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub decoded_level: u8,
    pub status: i32,
}

impl VimgBuf {
    fn err(code: i32) -> Self {
        Self {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            width: 0,
            height: 0,
            channels: 0,
            decoded_level: 0,
            status: code,
        }
    }

    fn from_vec(mut v: Vec<u8>, width: u32, height: u32, channels: u8, decoded_level: u8) -> Self {
        let ptr = v.as_mut_ptr();
        let len = v.len();
        let cap = v.capacity();
        core::mem::forget(v);
        Self { ptr, len, cap, width, height, channels, decoded_level, status: 0 }
    }
}

#[no_mangle]
pub extern "C" fn vimg_version_string() -> *const c_char {
    VERSION_CSTR.as_ptr().cast::<c_char>()
}

#[no_mangle]
pub unsafe extern "C" fn vimg_free(buf: VimgBuf) {
    if !buf.ptr.is_null() && buf.cap > 0 {
        let _ = Vec::from_raw_parts(buf.ptr, buf.len, buf.cap);
    }
}

#[no_mangle]
pub unsafe extern "C" fn vimg_encode_alloc(
    pixels: *const u8,
    pixels_len: usize,
    width: u32,
    height: u32,
    channels: u8,
    quality: u8,
    tile_size: u32,
    dwt_levels: u8,
) -> VimgBuf {
    if pixels.is_null() {
        return VimgBuf::err(-1);
    }
    let slice = core::slice::from_raw_parts(pixels, pixels_len);
    let opts = EncodeOptions {
        quality,
        tile_size: if tile_size == 0 { 128 } else { tile_size },
        dwt_levels,
        chroma_420: quality < 100,
        checksum: true,
        threads: 0,
    };
    match catch_unwind(AssertUnwindSafe(|| encode(slice, width, height, channels, &opts))) {
        Ok(Ok(bytes)) => VimgBuf::from_vec(bytes, width, height, channels, dwt_levels),
        Ok(Err(_)) => VimgBuf::err(-2),
        Err(_) => VimgBuf::err(-7),
    }
}

#[no_mangle]
pub unsafe extern "C" fn vimg_decode_query_alloc(
    vimg_ptr: *const u8,
    vimg_len: usize,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
    target_w: u32,
    target_h: u32,
    grayscale: u8,
    blur: f32,
    sharpen: f32,
    brightness: i16,
    contrast: f32,
    rotate_cw: u8,
) -> VimgBuf {
    if vimg_ptr.is_null() {
        return VimgBuf::err(-1);
    }
    let slice = core::slice::from_raw_parts(vimg_ptr, vimg_len);
    let query = TransformQuery {
        crop: if crop_w > 0 && crop_h > 0 { Some((crop_x, crop_y, crop_w, crop_h)) } else { None },
        target_width: (target_w > 0).then_some(target_w),
        target_height: (target_h > 0).then_some(target_h),
        grayscale: grayscale != 0,
        blur,
        sharpen,
        brightness,
        contrast: if contrast <= 0.0 { 1.0 } else { contrast },
        rotate_cw,
        ..TransformQuery::default()
    };
    match catch_unwind(AssertUnwindSafe(|| decode_query(slice, &query))) {
        Ok(Ok(img)) => VimgBuf::from_vec(img.pixels, img.width, img.height, img.channels, img.decoded_level),
        Ok(Err(_)) => VimgBuf::err(-2),
        Err(_) => VimgBuf::err(-7),
    }
}

// ============================================================================
// Zero-WASI WebAssembly HTTP-Range Streaming Exports (no wasm-bindgen needed)
// ============================================================================
// Metadata slot layout (`[u32; 12]`):
// [0]=status (0=ok) [1]=width [2]=height [3]=level [4]=tiles_fetched [5]=tiles_total
// [6]=out_ptr [7]=out_len [8]=dir_need [9]=total_bytes [10]=tile_size [11]=dwt_levels
static mut WASM_META: [u32; 12] = [0; 12];

#[no_mangle]
pub extern "C" fn vimg_wasm_meta_ptr() -> *const u32 {
    core::ptr::addr_of!(WASM_META) as *const u32
}

#[no_mangle]
pub extern "C" fn vimg_wasm_alloc(len: usize) -> *mut u8 {
    let mut v = Vec::<u8>::with_capacity(len);
    let ptr = v.as_mut_ptr();
    core::mem::forget(v);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn vimg_wasm_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let _ = Vec::from_raw_parts(ptr, 0, len);
    }
}

#[no_mangle]
pub unsafe extern "C" fn vimg_wasm_parse_header(ptr: *const u8, len: usize) -> i32 {
    let meta = &mut *core::ptr::addr_of_mut!(WASM_META);
    meta.fill(0);
    if ptr.is_null() || len < HEADER_BYTES {
        meta[0] = 1;
        return -1;
    }
    let slice = core::slice::from_raw_parts(ptr, len);
    match VimgHeader::parse(slice) {
        Ok(h) => {
            meta[1] = h.width;
            meta[2] = h.height;
            meta[5] = h.n_tiles() as u32;
            meta[8] = (HEADER_BYTES + h.dir_bytes as usize) as u32;
            meta[9] = h.total_bytes as u32;
            meta[10] = h.tile_size;
            meta[11] = h.dwt_levels as u32;
            0
        }
        Err(_) => {
            meta[0] = 2;
            -2
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vimg_wasm_plan(
    prefix_ptr: *const u8,
    prefix_len: usize,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
    target_w: u32,
    target_h: u32,
    grayscale: u8,
    sepia: u8,
) -> i32 {
    let meta = &mut *core::ptr::addr_of_mut!(WASM_META);
    meta.fill(0);
    if prefix_ptr.is_null() {
        meta[0] = 1;
        return -1;
    }
    let prefix = core::slice::from_raw_parts(prefix_ptr, prefix_len);
    let Ok(idx) = VimgIndex::parse_prefix(prefix) else {
        meta[0] = 2;
        return -2;
    };
    let q = TransformQuery {
        crop: if crop_w > 0 && crop_h > 0 { Some((crop_x, crop_y, crop_w, crop_h)) } else { None },
        target_width: (target_w > 0).then_some(target_w),
        target_height: (target_h > 0).then_some(target_h),
        grayscale: grayscale != 0,
        sepia: sepia != 0,
        ..TransformQuery::default()
    };
    match plan_query_ranges(&idx, &q, prefix_len as u64) {
        Ok((ranges, lvl, tf, tt)) => {
            let mut flat: Vec<u32> = Vec::with_capacity(ranges.len() * 2);
            for (s, e) in ranges {
                flat.push(s as u32);
                flat.push(e as u32);
            }
            let out_len = flat.len() * 4;
            let mut bytes = Vec::<u8>::with_capacity(out_len);
            for v in flat {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let ptr = bytes.as_mut_ptr();
            let cap = bytes.capacity();
            core::mem::forget(bytes);
            meta[1] = idx.header.width;
            meta[2] = idx.header.height;
            meta[3] = lvl as u32;
            meta[4] = tf as u32;
            meta[5] = tt as u32;
            meta[6] = ptr as u32;
            meta[7] = cap as u32;
            meta[8] = (HEADER_BYTES + idx.header.dir_bytes as usize) as u32;
            meta[9] = idx.header.total_bytes as u32;
            meta[10] = idx.header.tile_size;
            meta[11] = idx.header.dwt_levels as u32;
            0
        }
        Err(_) => {
            meta[0] = 3;
            -3
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn vimg_wasm_decode_sparse(
    prefix_ptr: *const u8,
    prefix_len: usize,
    span_offsets_ptr: *const u32,
    span_lens_ptr: *const u32,
    n_spans: usize,
    spans_data_ptr: *const u8,
    spans_data_len: usize,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
    target_w: u32,
    target_h: u32,
    grayscale: u8,
    sepia: u8,
    blur: f32,
    sharpen: f32,
    brightness: i16,
    contrast: f32,
    rotate_cw: u8,
) -> i32 {
    let meta = &mut *core::ptr::addr_of_mut!(WASM_META);
    meta.fill(0);
    if prefix_ptr.is_null() {
        meta[0] = 1;
        return -1;
    }
    let prefix = core::slice::from_raw_parts(prefix_ptr, prefix_len);
    let Ok(hdr) = VimgHeader::parse(prefix) else {
        meta[0] = 2;
        return -2;
    };
    let mut spans: Vec<(u64, &[u8])> = Vec::with_capacity(n_spans + 1);
    spans.push((0, prefix));
    if n_spans > 0 && !span_offsets_ptr.is_null() && !span_lens_ptr.is_null() && !spans_data_ptr.is_null() {
        let offs = core::slice::from_raw_parts(span_offsets_ptr, n_spans);
        let lens = core::slice::from_raw_parts(span_lens_ptr, n_spans);
        let data = core::slice::from_raw_parts(spans_data_ptr, spans_data_len);
        let mut cursor = 0usize;
        for i in 0..n_spans {
            let l = lens[i] as usize;
            if cursor + l > data.len() {
                meta[0] = 4;
                return -4;
            }
            spans.push((offs[i] as u64, &data[cursor..cursor + l]));
            cursor += l;
        }
    }
    let src = SparseRangeSource { total_len: hdr.total_bytes, spans };
    let q = TransformQuery {
        crop: if crop_w > 0 && crop_h > 0 { Some((crop_x, crop_y, crop_w, crop_h)) } else { None },
        target_width: (target_w > 0).then_some(target_w),
        target_height: (target_h > 0).then_some(target_h),
        grayscale: grayscale != 0,
        sepia: sepia != 0,
        blur,
        sharpen,
        brightness,
        contrast: if contrast <= 0.0 { 1.0 } else { contrast },
        rotate_cw,
        ..TransformQuery::default()
    };
    match decode_from_source(&src, &q) {
        Ok(img) => {
            let n_px = (img.width as usize) * (img.height as usize);
            let mut rgba = Vec::with_capacity(n_px * 4);
            match img.channels {
                1 => {
                    for &g in &img.pixels {
                        rgba.extend_from_slice(&[g, g, g, 255]);
                    }
                }
                3 => {
                    for c in img.pixels.chunks_exact(3) {
                        rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
                    }
                }
                4 => rgba = img.pixels,
                _ => {
                    meta[0] = 5;
                    return -5;
                }
            }
            let ptr = rgba.as_mut_ptr();
            let cap = rgba.capacity();
            core::mem::forget(rgba);
            meta[1] = img.width;
            meta[2] = img.height;
            meta[3] = img.decoded_level as u32;
            meta[4] = img.tiles_fetched as u32;
            meta[5] = img.tiles_total as u32;
            meta[6] = ptr as u32;
            meta[7] = cap as u32;
            meta[9] = hdr.total_bytes as u32;
            meta[10] = hdr.tile_size;
            meta[11] = hdr.dwt_levels as u32;
            0
        }
        Err(_) => {
            meta[0] = 6;
            -6
        }
    }
}

/// Encodes browser `ImageData` (`RGBA8`, `width * height * 4` bytes) into a `.vimg` buffer in WASM.
/// Auto-detects opaque images and packs them as 3-channel RGB for maximum compression ratio.
/// On success returns `0` with `WASM_META[6] = ptr`, `WASM_META[7] = len` (`cap == len`).
#[no_mangle]
pub unsafe extern "C" fn vimg_wasm_encode_rgba(
    rgba_ptr: *const u8,
    width: u32,
    height: u32,
    quality: u8,
    tile_size: u32,
) -> i32 {
    let meta = &mut *core::ptr::addr_of_mut!(WASM_META);
    meta.fill(0);
    if rgba_ptr.is_null() || width == 0 || height == 0 {
        meta[0] = 1;
        return -1;
    }
    let n_px = (width as usize).saturating_mul(height as usize);
    let rgba = core::slice::from_raw_parts(rgba_ptr, n_px * 4);
    let has_alpha = rgba.chunks_exact(4).any(|p| p[3] != 255);
    let (pixels, ch): (Vec<u8>, u8) = if has_alpha {
        (rgba.to_vec(), 4)
    } else {
        let mut rgb = Vec::with_capacity(n_px * 3);
        for p in rgba.chunks_exact(4) {
            rgb.extend_from_slice(&[p[0], p[1], p[2]]);
        }
        (rgb, 3)
    };
    let opts = EncodeOptions {
        quality: quality.clamp(1, 100),
        tile_size: if tile_size == 0 { 256 } else { tile_size },
        dwt_levels: 3,
        chroma_420: quality < 100,
        checksum: true,
        threads: 1,
    };
    match encode(&pixels, width, height, ch, &opts) {
        Ok(mut encoded) => {
            encoded.shrink_to_fit();
            let ptr = encoded.as_mut_ptr();
            let len = encoded.len();
            core::mem::forget(encoded);
            meta[1] = width;
            meta[2] = height;
            meta[3] = ch as u32;
            meta[6] = ptr as u32;
            meta[7] = len as u32;
            meta[9] = len as u32;
            meta[10] = opts.tile_size;
            meta[11] = opts.dwt_levels as u32;
            0
        }
        Err(_) => {
            meta[0] = 2;
            -2
        }
    }
}
