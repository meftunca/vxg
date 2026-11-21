/*
 * VIMG v0.4.0 — HTTP/HTTPS Range-Native Hierarchical Wavelet Image & Animation Codec
 * SPDX-License-Identifier: MIT
 */

#ifndef VIMG_H
#define VIMG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint8_t *ptr;
    size_t len;
    size_t cap;
    uint32_t width;
    uint32_t height;
    uint8_t channels;
    uint8_t decoded_level;
    int32_t status;
} VimgBuf;

typedef struct {
    uint32_t crop_x;
    uint32_t crop_y;
    uint32_t crop_w;
    uint32_t crop_h;
    uint32_t target_w;
    uint32_t target_h;
    uint8_t grayscale;
    uint8_t sepia;
    uint8_t rotate_cw;
    uint8_t flip_h;
    uint8_t flip_v;
    uint8_t recover_corrupt;
    int16_t brightness;
    float blur;
    float sharpen;
    float contrast;
    float saturation;
    float temperature;
} VimgQueryC;

const char *vimg_version_string(void);
void vimg_free(VimgBuf buf);

VimgBuf vimg_encode_alloc(
    const uint8_t *pixels,
    size_t pixels_len,
    uint32_t width,
    uint32_t height,
    uint8_t channels,
    uint8_t quality,
    uint32_t tile_size,
    uint8_t dwt_levels
);

/* Legacy query API (kept for ABI compatibility; pass zeroed fields for defaults). */
VimgBuf vimg_decode_query_alloc(
    const uint8_t *vimg_ptr,
    size_t vimg_len,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_w,
    uint32_t crop_h,
    uint32_t target_w,
    uint32_t target_h,
    uint8_t grayscale,
    float blur,
    float sharpen,
    int16_t brightness,
    float contrast,
    uint8_t rotate_cw
);

/* Full-featured query API supporting sepia, flip_h, flip_v, and recover_corrupt.
 * Pass query == NULL for default full-resolution decode. */
VimgBuf vimg_decode_query_v2_alloc(
    const uint8_t *vimg_ptr,
    size_t vimg_len,
    const VimgQueryC *query
);

/* Extract embedded ICC color profile (returns status=0 + bytes, status=1 if absent, <0 on error). */
VimgBuf vimg_extract_icc_alloc(
    const uint8_t *vimg_ptr,
    size_t vimg_len
);

typedef struct {
    uint8_t color_primaries;
    uint8_t transfer_characteristics;
    uint8_t matrix_coefficients;
    uint8_t full_range;
    uint8_t present;
} VimgCicpC;

/* Extract CICP HDR color tags (returns 0 on success with out->present=1/0, <0 on error). */
int32_t vimg_extract_cicp(
    const uint8_t *vimg_ptr,
    size_t vimg_len,
    VimgCicpC *out_cicp
);

/* Extract embedded raw EXIF or XMP metadata (status=0 + bytes, status=1 if absent, <0 on error). */
VimgBuf vimg_extract_exif_alloc(const uint8_t *vimg_ptr, size_t vimg_len);
VimgBuf vimg_extract_xmp_alloc(const uint8_t *vimg_ptr, size_t vimg_len);

/* Zero-WASI WebAssembly / sparse HTTP-Range exports */
const uint32_t *vimg_wasm_meta_ptr(void);
uint8_t *vimg_wasm_alloc(size_t len);
void vimg_wasm_free(uint8_t *ptr, size_t len);
int32_t vimg_wasm_parse_header(const uint8_t *ptr, size_t len);
int32_t vimg_wasm_plan(
    const uint8_t *prefix_ptr,
    size_t prefix_len,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_w,
    uint32_t crop_h,
    uint32_t target_w,
    uint32_t target_h,
    uint8_t grayscale,
    uint8_t sepia
);
int32_t vimg_wasm_plan_v3(
    const uint8_t *prefix_ptr,
    size_t prefix_len,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_w,
    uint32_t crop_h,
    uint32_t target_w,
    uint32_t target_h,
    uint8_t grayscale,
    uint8_t sepia,
    int32_t force_level
);
int32_t vimg_wasm_decode_sparse(
    const uint8_t *prefix_ptr,
    size_t prefix_len,
    const uint32_t *span_offsets_ptr,
    const uint32_t *span_lens_ptr,
    size_t n_spans,
    const uint8_t *spans_data_ptr,
    size_t spans_data_len,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_w,
    uint32_t crop_h,
    uint32_t target_w,
    uint32_t target_h,
    uint8_t grayscale,
    uint8_t sepia,
    float blur,
    float sharpen,
    int16_t brightness,
    float contrast,
    uint8_t rotate_cw
);
int32_t vimg_wasm_decode_sparse_v3(
    const uint8_t *prefix_ptr,
    size_t prefix_len,
    const uint32_t *span_offsets_ptr,
    const uint32_t *span_lens_ptr,
    size_t n_spans,
    const uint8_t *spans_data_ptr,
    size_t spans_data_len,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_w,
    uint32_t crop_h,
    uint32_t target_w,
    uint32_t target_h,
    uint8_t grayscale,
    uint8_t sepia,
    float blur,
    float sharpen,
    int16_t brightness,
    float contrast,
    float saturation,
    float temperature,
    uint8_t rotate_cw,
    int32_t force_level
);
int32_t vimg_wasm_encode_rgba(
    const uint8_t *rgba_ptr,
    uint32_t width,
    uint32_t height,
    uint8_t quality,
    uint32_t tile_size
);
int32_t vimg_wasm_encode_rgba_v3(
    const uint8_t *rgba_ptr,
    uint32_t width,
    uint32_t height,
    uint8_t quality,
    uint32_t tile_size,
    uint8_t adaptive_quant
);

#ifdef __cplusplus
}
#endif

#endif /* VIMG_H */
