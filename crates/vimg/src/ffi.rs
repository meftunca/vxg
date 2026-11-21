//! Panic-safe C ABI (`libvimg.a` / `libvimg.dylib` / `.so` / `.dll`) and zero-WASI `wasm32` exports.
#![allow(clippy::missing_safety_doc)]

use core::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    decode_from_source, decode_query, encode,
    format::{HEADER_BYTES, MAX_IMAGE_DIM, MAX_IMAGE_PIXELS},
    plan_query_ranges, EncodeOptions, RangeStats, SliceRangeSource, SparseRangeSource,
    TransformQuery, VimgHeader, VimgIndex,
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
        ..EncodeOptions::default()
    };
    match catch_unwind(AssertUnwindSafe(|| encode(slice, width, height, channels, &opts))) {
        Ok(Ok(bytes)) => VimgBuf::from_vec(bytes, width, height, channels, dwt_levels),
        Ok(Err(_)) => VimgBuf::err(-2),
        Err(_) => VimgBuf::err(-7),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VimgQueryC {
    pub crop_x: u32,
    pub crop_y: u32,
    pub crop_w: u32,
    pub crop_h: u32,
    pub target_w: u32,
    pub target_h: u32,
    pub grayscale: u8,
    pub sepia: u8,
    pub rotate_cw: u8,
    pub flip_h: u8,
    pub flip_v: u8,
    pub recover_corrupt: u8,
    pub brightness: i16,
    pub blur: f32,
    pub sharpen: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub temperature: f32,
}

#[no_mangle]
pub unsafe extern "C" fn vimg_decode_query_v2_alloc(
    vimg_ptr: *const u8,
    vimg_len: usize,
    query_ptr: *const VimgQueryC,
) -> VimgBuf {
    if vimg_ptr.is_null() {
        return VimgBuf::err(-1);
    }
    let slice = core::slice::from_raw_parts(vimg_ptr, vimg_len);
    let query = if query_ptr.is_null() {
        TransformQuery::default()
    } else {
        let q = *query_ptr;
        TransformQuery {
            crop: if q.crop_w > 0 && q.crop_h > 0 { Some((q.crop_x, q.crop_y, q.crop_w, q.crop_h)) } else { None },
            target_width: (q.target_w > 0).then_some(q.target_w),
            target_height: (q.target_h > 0).then_some(q.target_h),
            grayscale: q.grayscale != 0,
            sepia: q.sepia != 0,
            blur: q.blur,
            sharpen: q.sharpen,
            brightness: q.brightness,
            contrast: if q.contrast <= 0.0 { 1.0 } else { q.contrast },
            saturation: if q.saturation <= 0.0 { 1.0 } else { q.saturation },
            temperature: q.temperature,
            rotate_cw: q.rotate_cw,
            flip_h: q.flip_h != 0,
            flip_v: q.flip_v != 0,
            recover_corrupt: q.recover_corrupt != 0,
            ..TransformQuery::default()
        }
    };
    match catch_unwind(AssertUnwindSafe(|| decode_query(slice, &query))) {
        Ok(Ok(img)) => VimgBuf::from_vec(img.pixels, img.width, img.height, img.channels, img.decoded_level),
        Ok(Err(_)) => VimgBuf::err(-2),
        Err(_) => VimgBuf::err(-7),
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
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

/// Extracts the embedded ICC color profile (if present) from a `.vimg` buffer.
/// Returns `status = 0` with `ptr != NULL` if an ICC profile exists, `status = 1` if none exists,
/// or a negative error code on invalid/corrupt input.
#[no_mangle]
pub unsafe extern "C" fn vimg_extract_icc_alloc(
    vimg_ptr: *const u8,
    vimg_len: usize,
) -> VimgBuf {
    if vimg_ptr.is_null() {
        return VimgBuf::err(-1);
    }
    let slice = core::slice::from_raw_parts(vimg_ptr, vimg_len);
    match catch_unwind(AssertUnwindSafe(|| {
        let idx = VimgIndex::parse_prefix(slice)?;
        let src = SliceRangeSource(slice);
        let mut stats = RangeStats::default();
        idx.read_icc_from_source(&src, slice, true, &mut stats)
    })) {
        Ok(Ok(Some(icc))) => VimgBuf::from_vec(icc, 0, 0, 0, 0),
        Ok(Ok(None)) => VimgBuf::err(1),
        Ok(Err(_)) => VimgBuf::err(-2),
        Err(_) => VimgBuf::err(-7),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct VimgCicpC {
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: u8,
    pub present: u8,
}

/// Extracts CICP HDR color signaling (`BT.2020`, `PQ`, `HLG`, `Display P3`, `sRGB`) from a `.vimg` buffer.
/// Returns `0` on success (`out.present == 1` if CICP is present, `0` if absent) or `< 0` on error.
#[no_mangle]
pub unsafe extern "C" fn vimg_extract_cicp(
    vimg_ptr: *const u8,
    vimg_len: usize,
    out_ptr: *mut VimgCicpC,
) -> i32 {
    if vimg_ptr.is_null() || out_ptr.is_null() {
        return -1;
    }
    *out_ptr = VimgCicpC::default();
    let slice = core::slice::from_raw_parts(vimg_ptr, vimg_len);
    match catch_unwind(AssertUnwindSafe(|| {
        let idx = VimgIndex::parse_prefix(slice)?;
        let src = SliceRangeSource(slice);
        let mut stats = RangeStats::default();
        idx.read_metadata_from_source(&src, true, &mut stats)
    })) {
        Ok(Ok(Some(meta))) => {
            if let Some(c) = meta.cicp {
                *out_ptr = VimgCicpC {
                    color_primaries: c.color_primaries,
                    transfer_characteristics: c.transfer_characteristics,
                    matrix_coefficients: c.matrix_coefficients,
                    full_range: u8::from(c.full_range),
                    present: 1,
                };
            }
            0
        }
        Ok(Ok(None)) => 0,
        Ok(Err(_)) => -2,
        Err(_) => -7,
    }
}

/// Extracts the embedded raw EXIF payload (returns `status=0` + bytes, `status=1` if absent, `<0` on error).
#[no_mangle]
pub unsafe extern "C" fn vimg_extract_exif_alloc(vimg_ptr: *const u8, vimg_len: usize) -> VimgBuf {
    if vimg_ptr.is_null() {
        return VimgBuf::err(-1);
    }
    let slice = core::slice::from_raw_parts(vimg_ptr, vimg_len);
    match catch_unwind(AssertUnwindSafe(|| {
        let idx = VimgIndex::parse_prefix(slice)?;
        let src = SliceRangeSource(slice);
        let mut stats = RangeStats::default();
        idx.read_metadata_from_source(&src, true, &mut stats)
    })) {
        Ok(Ok(Some(meta))) => match meta.exif {
            Some(exif) => VimgBuf::from_vec(exif, 0, 0, 0, 0),
            None => VimgBuf::err(1),
        },
        Ok(Ok(None)) => VimgBuf::err(1),
        Ok(Err(_)) => VimgBuf::err(-2),
        Err(_) => VimgBuf::err(-7),
    }
}

/// Extracts the embedded raw XMP XML packet (returns `status=0` + bytes, `status=1` if absent, `<0` on error).
#[no_mangle]
pub unsafe extern "C" fn vimg_extract_xmp_alloc(vimg_ptr: *const u8, vimg_len: usize) -> VimgBuf {
    if vimg_ptr.is_null() {
        return VimgBuf::err(-1);
    }
    let slice = core::slice::from_raw_parts(vimg_ptr, vimg_len);
    match catch_unwind(AssertUnwindSafe(|| {
        let idx = VimgIndex::parse_prefix(slice)?;
        let src = SliceRangeSource(slice);
        let mut stats = RangeStats::default();
        idx.read_metadata_from_source(&src, true, &mut stats)
    })) {
        Ok(Ok(Some(meta))) => match meta.xmp {
            Some(xmp) => VimgBuf::from_vec(xmp, 0, 0, 0, 0),
            None => VimgBuf::err(1),
        },
        Ok(Ok(None)) => VimgBuf::err(1),
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

#[allow(clippy::too_many_arguments)]
unsafe fn wasm_plan_impl(
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
    force_level: i32,
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
        force_level: (force_level >= 0).then_some(force_level.min(255) as u8),
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
    wasm_plan_impl(prefix_ptr, prefix_len, crop_x, crop_y, crop_w, crop_h, target_w, target_h, grayscale, sepia, -1)
}

/// v0.3.0 range planner adding explicit `force_level` (`-1` = auto, `0..=dwt_levels` = progressive tier).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn vimg_wasm_plan_v3(
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
    force_level: i32,
) -> i32 {
    wasm_plan_impl(prefix_ptr, prefix_len, crop_x, crop_y, crop_w, crop_h, target_w, target_h, grayscale, sepia, force_level)
}

#[allow(clippy::too_many_arguments)]
unsafe fn wasm_decode_sparse_impl(
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
    saturation: f32,
    temperature: f32,
    rotate_cw: u8,
    force_level: i32,
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
            let Some(end) = cursor.checked_add(l) else {
                meta[0] = 4;
                return -4;
            };
            if end > data.len() {
                meta[0] = 4;
                return -4;
            }
            spans.push((offs[i] as u64, &data[cursor..end]));
            cursor = end;
        }
    }
    let src = SparseRangeSource { total_len: hdr.total_bytes, spans };
    let q = TransformQuery {
        crop: if crop_w > 0 && crop_h > 0 { Some((crop_x, crop_y, crop_w, crop_h)) } else { None },
        target_width: (target_w > 0).then_some(target_w),
        target_height: (target_h > 0).then_some(target_h),
        force_level: (force_level >= 0).then_some(force_level.min(255) as u8),
        grayscale: grayscale != 0,
        sepia: sepia != 0,
        blur,
        sharpen,
        brightness,
        contrast: if contrast <= 0.0 { 1.0 } else { contrast },
        saturation: if saturation < 0.0 { 1.0 } else { saturation },
        temperature,
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
                    for c in img.pixels.as_chunks::<3>().0 {
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
    wasm_decode_sparse_impl(
        prefix_ptr, prefix_len, span_offsets_ptr, span_lens_ptr, n_spans, spans_data_ptr, spans_data_len,
        crop_x, crop_y, crop_w, crop_h, target_w, target_h, grayscale, sepia, blur, sharpen, brightness,
        contrast, 1.0, 0.0, rotate_cw, -1,
    )
}

/// v0.3.0 sparse WASM decoder adding compressed-domain `saturation`, `temperature`, and `force_level`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn vimg_wasm_decode_sparse_v3(
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
    saturation: f32,
    temperature: f32,
    rotate_cw: u8,
    force_level: i32,
) -> i32 {
    wasm_decode_sparse_impl(
        prefix_ptr, prefix_len, span_offsets_ptr, span_lens_ptr, n_spans, spans_data_ptr, spans_data_len,
        crop_x, crop_y, crop_w, crop_h, target_w, target_h, grayscale, sepia, blur, sharpen, brightness,
        contrast, saturation, temperature, rotate_cw, force_level,
    )
}

unsafe fn wasm_encode_rgba_impl(
    rgba_ptr: *const u8,
    width: u32,
    height: u32,
    quality: u8,
    tile_size: u32,
    adaptive_quant: bool,
) -> i32 {
    let meta = &mut *core::ptr::addr_of_mut!(WASM_META);
    meta.fill(0);
    if rgba_ptr.is_null() || width == 0 || height == 0 || width > MAX_IMAGE_DIM || height > MAX_IMAGE_DIM {
        meta[0] = 1;
        return -1;
    }
    let Some(n_px_u64) = (width as u64).checked_mul(height as u64) else {
        meta[0] = 1;
        return -1;
    };
    if n_px_u64 > MAX_IMAGE_PIXELS {
        meta[0] = 1;
        return -1;
    }
    let Some(n_px) = usize::try_from(n_px_u64).ok() else {
        meta[0] = 1;
        return -1;
    };
    let Some(rgba_bytes) = n_px.checked_mul(4) else {
        meta[0] = 1;
        return -1;
    };
    let rgba = core::slice::from_raw_parts(rgba_ptr, rgba_bytes);
    let has_alpha = rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 255);
    let (pixels, ch): (Vec<u8>, u8) = if has_alpha {
        (rgba.to_vec(), 4)
    } else {
        let mut rgb = Vec::with_capacity(n_px * 3);
        for p in rgba.as_chunks::<4>().0 {
            rgb.extend_from_slice(&[p[0], p[1], p[2]]);
        }
        (rgb, 3)
    };
    let opts = EncodeOptions {
        quality: quality.clamp(1, 100),
        tile_size: if tile_size == 0 { 256 } else { tile_size },
        dwt_levels: 3,
        chroma_420: quality < 100,
        adaptive_quant,
        checksum: true,
        threads: 1,
        ..EncodeOptions::default()
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
    wasm_encode_rgba_impl(rgba_ptr, width, height, quality, tile_size, false)
}

/// v0.3.0 WASM encoder adding Tile-Adaptive Psychovisual Quantization (`adaptive_quant != 0`).
#[no_mangle]
pub unsafe extern "C" fn vimg_wasm_encode_rgba_v3(
    rgba_ptr: *const u8,
    width: u32,
    height: u32,
    quality: u8,
    tile_size: u32,
    adaptive_quant: u8,
) -> i32 {
    wasm_encode_rgba_impl(rgba_ptr, width, height, quality, tile_size, adaptive_quant != 0)
}
