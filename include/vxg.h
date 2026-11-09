/*
 * VXG v0.5.2 — Unified Adaptive Multi-Format Lossless Block Codec & SQL Engine (Format v4/v5)
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
 * Returns the null-terminated library version string (e.g. "0.5.2").
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
 * Header + directory query: returns the exact compressed byte size of the first VXG frame
 * in `src[0..src_len]`, or a negative `VXG_ERR_*` code if truncated or invalid.
 */
vxg_ssize_t vxg_frame_compressed_size(const uint8_t *src, size_t src_len);

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
 * Allocates a zero-initialized buffer of `len` bytes owned by VXG (ideal for WebAssembly
 * hosts operating on `WebAssembly.Memory`). Free with `vxg_free(ptr, len)`.
 */
uint8_t *vxg_alloc(size_t len);

/**
 * Frees a buffer allocated by `vxg_alloc`, `vxg_compress_alloc`, or `vxg_decompress_alloc`.
 */
void vxg_free(uint8_t *ptr, size_t len);

/**
 * Compressed-domain structured record count across all blocks in a VXG frame.
 * On `GRAMMAR_PACKET` blocks this inspects only the grammar headers in O(1) time per block.
 * Returns total record count (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_frame_row_count(const uint8_t *frame_ptr, size_t frame_len);

/**
 * Compressed-domain random-access extraction of a single record `global_row_index`
 * directly into caller-supplied `dst[0..dst_cap]`.
 * Returns exact byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_extract_row_into(
    const uint8_t *frame_ptr,
    size_t frame_len,
    size_t global_row_index,
    uint8_t *dst,
    size_t dst_cap
);

/**
 * Compressed-domain range extraction of records `[start_row .. end_row)`
 * directly into caller-supplied `dst[0..dst_cap]`.
 * Returns exact byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_extract_row_range_into(
    const uint8_t *frame_ptr,
    size_t frame_len,
    size_t start_row,
    size_t end_row,
    uint8_t *dst,
    size_t dst_cap
);

/**
 * Compressed-domain predicate pushdown search for the first record containing
 * `needle[0..needle_len]`.
 * - If found: copies the matching record into `dst[0..dst_cap]`, stores its global
 *   row index in `*out_row_index` (if non-NULL), and returns the record byte length (`> 0`).
 * - If not found: stores `SIZE_MAX` in `*out_row_index` (if non-NULL) and returns `0`.
 * - On error: returns negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_find_first_into(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const uint8_t *needle,
    size_t needle_len,
    uint8_t *dst,
    size_t dst_cap,
    size_t *out_row_index
);

/**
 * Scalar aggregation statistics populated by `vxg_aggregate_col` (64 bytes, 8-byte aligned).
 */
typedef struct vxg_agg_stats_t {
    uint64_t row_count;   /* Total matching records (COUNT(*)) */
    uint64_t num_count;   /* Matching records with canonical u64 column value (COUNT(col)) */
    double   sum;         /* SUM(col) as IEEE-754 double (exact up to 2^53) */
    uint64_t sum_lo;      /* Low 64 bits of exact 128-bit integer SUM(col) */
    uint64_t sum_hi;      /* High 64 bits of exact 128-bit integer SUM(col) */
    uint64_t min_val;     /* MIN(col) when has_minmax != 0, else 0 */
    uint64_t max_val;     /* MAX(col) when has_minmax != 0, else 0 */
    int32_t  has_minmax;  /* 1 when num_count > 0, else 0 */
    int32_t  _reserved;   /* Zeroed padding */
} vxg_agg_stats_t;

/**
 * Zero-output-allocation integrity and CRC32C verification across single or concatenated
 * VXG frames using bounded ~512 KiB L2 scratch per worker thread.
 * Returns total verified uncompressed byte count (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_verify_frame(
    const uint8_t *src,
    size_t src_len,
    size_t threads,
    int32_t verify_checksum
);

/**
 * File-to-file compression using VXG's hybrid kernel I/O & mmap pipeline.
 * Pass `force != 0` to overwrite `output_path` if it already exists.
 * Returns compressed byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_compress_file(
    const char *input_path,
    const char *output_path,
    size_t threads,
    size_t block_size,
    int32_t checksum,
    int32_t force
);

/**
 * File-to-file decompression using zero-copy read-only `mmap` on `.vxg` inputs.
 * Returns uncompressed byte count written (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_decompress_file(
    const char *input_path,
    const char *output_path,
    size_t threads,
    int32_t verify_checksum,
    int32_t force
);

/**
 * Zero-copy `mmap` file verification with ~512 KiB L2 scratch and zero output allocation.
 * Returns verified uncompressed byte count (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_verify_file(
    const char *input_path,
    size_t threads,
    int32_t verify_checksum
);

/**
 * Compressed-domain `COUNT(*)` over compound boolean predicate `where_cstr`
 * (e.g. `"status>=500 && service==\"auth-gateway\""`, or `NULL`/`""` for all records).
 * Returns matching record count (`>= 0`), or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_count_where_str(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *where_cstr,
    size_t threads
);

/**
 * Compressed-domain scalar aggregation (`COUNT(*)`, `COUNT(col)`, `SUM`, `MIN`, `MAX`)
 * on `col_cstr` (`"latency_ms"`, `"csv:2"`, etc.) with optional `where_cstr`.
 * Returns `VXG_OK` (`0`) on success, or negative `VXG_ERR_*`.
 */
vxg_ssize_t vxg_aggregate_col(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *col_cstr,
    const char *where_cstr,
    size_t threads,
    vxg_agg_stats_t *out_stats
);

/**
 * Compressed-domain `SELECT` / `WHERE` / `ORDER BY` Top-K query returning a newly allocated
 * UTF-8/byte buffer of matching (or projected) records.
 * - `cols_cstr`: `"ts,service,latency_ms"` or `NULL` for full records.
 * - `where_cstr`: `"status>=500 && !contains:healthz"` or `NULL` for all records.
 * - `order_by_cstr`: `"latency_ms"` or `NULL` for storage order (`order_by_cstr` need not be in `cols_cstr`).
 * - `desc`: non-zero for `DESC`, `0` for `ASC` (`NULLS LAST`, stable row-index tie-breaking).
 * - `limit`: max records (`SIZE_MAX` for all matching records).
 * Free the returned buffer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_select_alloc(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *cols_cstr,
    const char *where_cstr,
    const char *order_by_cstr,
    int32_t desc,
    size_t limit,
    size_t threads,
    size_t *out_len
);

/**
 * Produces a new compressed `VXG1` frame directly in the compressed domain by projecting
 * `cols_cstr` (zero-reencode stream splicing when unfiltered) and/or filtering by `where_cstr`.
 * Free the returned buffer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_project_frame_alloc(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *cols_cstr,
    const char *where_cstr,
    size_t threads,
    size_t *out_len
);

/**
 * Compressed-domain `GROUP BY group_col_cstr` with optional `agg_col_cstr` (`SUM`/`MIN`/`MAX`)
 * and optional `where_cstr`, returning TSV lines (`key\trow_count\tnum_count\tsum\tmin\tmax\n`).
 * Free the returned buffer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_group_by_alloc(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *group_col_cstr,
    const char *agg_col_cstr,
    const char *where_cstr,
    size_t threads,
    size_t *out_len
);

/**
 * Executes a compressed-domain SQL `SELECT` query (`query_cstr`, supporting `WHERE`,
 * multi-key / multi-agg `GROUP BY`, `ORDER BY`, `LIMIT`, CSV header column names, and
 * dotted nested JSON paths) directly on `frame_ptr[0..frame_len]`.
 * - `format`: `0` = TSV, `1` = CSV, `2` = NDJSON.
 * - `no_header`: non-zero omits the header row on TSV/CSV output.
 * Free the returned buffer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_sql_alloc(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *query_cstr,
    int32_t format,
    int32_t no_header,
    size_t threads,
    size_t *out_len
);

/* ============================================================================
 * Apache Arrow C Data Interface & C Stream Interface (Zero-Copy Interop)
 * ============================================================================ */

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE           2
#define ARROW_FLAG_MAP_KEYS_SORTED    4

struct ArrowSchema {
    const char *format;
    const char *name;
    const char *metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema **children;
    struct ArrowSchema *dictionary;
    void (*release)(struct ArrowSchema *);
    void *private_data;
};

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void **buffers;
    struct ArrowArray **children;
    struct ArrowArray *dictionary;
    void (*release)(struct ArrowArray *);
    void *private_data;
};

#endif /* ARROW_C_DATA_INTERFACE */

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
    int (*get_schema)(struct ArrowArrayStream *, struct ArrowSchema *out);
    int (*get_next)(struct ArrowArrayStream *, struct ArrowArray *out);
    const char *(*get_last_error)(struct ArrowArrayStream *);
    void (*release)(struct ArrowArrayStream *);
    void *private_data;
};

#endif /* ARROW_C_STREAM_INTERFACE */

/**
 * Exports a compressed VXG frame (with optional compressed-domain SQL `query_cstr`, or
 * `NULL`/`""` for automatic CSV header / NDJSON schema discovery) into an Apache Arrow
 * C Data Interface `(ArrowSchema, ArrowArray)` pair with 64-byte-aligned buffers.
 * Caller/consumer releases ownership via `out_schema->release(out_schema)` and
 * `out_array->release(out_array)`. Returns `VXG_OK` (`0`) on success, or `< 0` error code.
 */
vxg_ssize_t vxg_arrow_export(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *query_cstr,
    size_t threads,
    struct ArrowSchema *out_schema,
    struct ArrowArray *out_array
);

/**
 * Exports a compressed VXG frame (with optional compressed-domain SQL `query_cstr`) into
 * an Apache Arrow C Stream Interface `ArrowArrayStream` (directly consumable via DuckDB,
 * Polars, PyArrow, DataFusion, and Arrow PyCapsule `__arrow_c_stream__`).
 * Returns `VXG_OK` (`0`) on success, or `< 0` error code.
 */
vxg_ssize_t vxg_arrow_export_stream(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *query_cstr,
    size_t threads,
    struct ArrowArrayStream *out_stream
);

/**
 * Exports a compressed VXG frame with compressed-domain column projection (`cols_cstr`, comma-
 * separated column names / `csv:N` targets, or `NULL`/`""`/`"*"` for automatic discovery) and
 * `WHERE` pushdown (`where_cstr`, accepting SQL `WHERE` or VXG filter syntax, or `NULL`/`""`)
 * into an Apache Arrow C Data Interface `(ArrowSchema, ArrowArray)` pair.
 * Returns `VXG_OK` (`0`) on success, or `< 0` error code.
 */
vxg_ssize_t vxg_arrow_query(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *cols_cstr,
    const char *where_cstr,
    size_t threads,
    struct ArrowSchema *out_schema,
    struct ArrowArray *out_array
);

/**
 * Exports a compressed VXG frame with compressed-domain column projection (`cols_cstr`) and
 * `WHERE` pushdown (`where_cstr`) into a windowed multi-batch Apache Arrow C Stream Interface
 * `ArrowArrayStream` (~64K rows per batch, preserving typed schema even on 0-match scans).
 * `frame_ptr[0..frame_len]` must remain valid until `out_stream->release(out_stream)` is called.
 * Returns `VXG_OK` (`0`) on success, or `< 0` error code.
 */
vxg_ssize_t vxg_arrow_query_stream(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *cols_cstr,
    const char *where_cstr,
    size_t threads,
    struct ArrowArrayStream *out_stream
);

/**
 * Exports a compressed VXG frame (with optional SQL `query_cstr`) into a newly allocated
 * Apache Arrow IPC Stream (`application/vnd.apache.arrow.stream`, MetadataVersion V5).
 * Free the returned buffer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
 */
uint8_t *vxg_arrow_ipc_alloc(
    const uint8_t *frame_ptr,
    size_t frame_len,
    const char *query_cstr,
    size_t threads,
    size_t *out_len
);

#ifdef __cplusplus
}
#endif

#endif /* VXG_H */
