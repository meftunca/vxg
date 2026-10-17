/*
 * VXG v0.4.0 — Unified Adaptive Multi-Format Lossless Block Codec (Format v4)
 * Official C99 / C++11 Header for Static (`libvxg.a` / `vxg.lib`) and Dynamic
 * (`libvxg.dylib` / `libvxg.so` / `vxg.dll`) Library Integration.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef VXG_H
#define VXG_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T vxg_ssize_t;
#else
typedef ssize_t vxg_ssize_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Status and error codes returned by `vxg_*` functions when `< 0` */
#define VXG_OK                     ((vxg_ssize_t)0)
#define VXG_ERR_NULL_PTR           ((vxg_ssize_t)-1)
#define VXG_ERR_INVALID_OPTION     ((vxg_ssize_t)-2)
#define VXG_ERR_BUFFER_TOO_SMALL   ((vxg_ssize_t)-3)
#define VXG_ERR_CORRUPT_DATA       ((vxg_ssize_t)-4)
#define VXG_ERR_CHECKSUM_MISMATCH  ((vxg_ssize_t)-5)
#define VXG_ERR_UNSUPPORTED        ((vxg_ssize_t)-6)
#define VXG_ERR_INTERNAL           ((vxg_ssize_t)-7)

/**
 * Returns the null-terminated library version string (e.g. "0.4.0").
 */
const char *vxg_version_string(void);

/**
 * Returns a static null-terminated human-readable string for any `VXG_ERR_*` code.
 */
const char *vxg_error_message(vxg_ssize_t code);

/**
 * Returns the worst-case compressed frame size in bytes for `raw_len` input bytes.
 * Pass `block_size = 0` for the default 65536-byte (64 KiB) block size.
 */
vxg_ssize_t vxg_compress_bound(size_t raw_len, size_t block_size);

/**
 * O(1) header query: returns the exact uncompressed byte size of a VXG frame,
 * or a negative `VXG_ERR_*` code if the 32-byte header is truncated or invalid.
 */
vxg_ssize_t vxg_frame_raw_size(const uint8_t *src, size_t src_len);

/**
 * O(1) header query: returns the number of independent blocks in a VXG frame,
 * or a negative `VXG_ERR_*` code if invalid.
 */
vxg_ssize_t vxg_frame_block_count(const uint8_t *src, size_t src_len);

/**
 * O(1) header query: returns the uncompressed byte size of `block_index`,
 * or a negative `VXG_ERR_*` code if out of bounds or invalid.
 */
vxg_ssize_t vxg_frame_block_raw_size(const uint8_t *src, size_t src_len, size_t block_index);

/**
 * Compresses `src[0..src_len]` into caller-supplied `dst[0..dst_cap]`.
 * - `threads`: `0` = auto-detect all CPU cores, `1..256` = explicit worker count.
 * - `block_size`: `0` = default 65536 (64 KiB), or `256..65536`.
 * - `checksum`: non-zero enables hardware-accelerated CRC32C per block.
 * Returns exact compressed byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_compress_into(
    const uint8_t *src,
    size_t src_len,
    uint8_t *dst,
    size_t dst_cap,
    size_t threads,
    size_t block_size,
    int32_t checksum
);

/**
 * Zero-allocation decompression: decodes `src[0..src_len]` directly into
 * caller-supplied `dst[0..dst_cap]`.
 * - `threads`: `0` = auto-detect all CPU cores, `1..256` = explicit worker count.
 * - `verify_checksum`: non-zero verifies per-block CRC32C checksums.
 * Returns exact uncompressed byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_decompress_into(
    const uint8_t *src,
    size_t src_len,
    uint8_t *dst,
    size_t dst_cap,
    size_t threads,
    int32_t verify_checksum
);

/**
 * Zero-allocation O(1) random-access block extraction: decodes only `block_index`
 * directly into caller-supplied `dst[0..dst_cap]` without decoding other blocks.
 * Returns exact uncompressed byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_extract_block_into(
    const uint8_t *src,
    size_t src_len,
    size_t block_index,
    uint8_t *dst,
    size_t dst_cap,
    int32_t verify_checksum
);

/**
 * Convenience heap-allocating compression. Caller must free the returned buffer
 * using `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_compress_alloc(
    const uint8_t *src,
    size_t src_len,
    size_t threads,
    size_t block_size,
    int32_t checksum,
    size_t *out_len
);

/**
 * Convenience heap-allocating decompression. Caller must free the returned buffer
 * using `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_decompress_alloc(
    const uint8_t *src,
    size_t src_len,
    size_t threads,
    int32_t verify_checksum,
    size_t *out_len
);

/**
 * Frees a buffer allocated by `vxg_compress_alloc` or `vxg_decompress_alloc`.
 */
void vxg_free(uint8_t *ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* VXG_H */
