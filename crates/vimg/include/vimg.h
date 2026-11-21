/*
 * VIMG v0.1.0 — HTTP-Range-Native Hierarchical Wavelet Image Codec & Compressed-Domain Transform Engine
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

#ifdef __cplusplus
}
#endif

#endif /* VIMG_H */
