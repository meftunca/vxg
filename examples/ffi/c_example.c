/*
 * Zero-allocation C integration example for VXG v4 (`libvxg.a` or `libvxg.dylib`/`.so`).
 *
 * Build & run (macOS static link):
 *   cargo build --release
 *   cc -O3 -Iinclude examples/ffi/c_example.c target/release/libvxg.a -lpthread -ldl -lm -o target/release/c_example
 *   ./target/release/c_example
 *
 * Build & run (Linux static link):
 *   cargo build --release
 *   cc -O3 -Iinclude examples/ffi/c_example.c target/release/libvxg.a -lpthread -ldl -lm -o target/release/c_example
 *   ./target/release/c_example
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vxg.h"

int main(void) {
    printf("VXG C ABI version: %s\n", vxg_version_string());

    /* 1. Prepare structured NDJSON telemetry payload in caller memory */
    const size_t rows = 1000;
    size_t raw_cap = rows * 128;
    uint8_t *raw = (uint8_t *)malloc(raw_cap);
    assert(raw != NULL);

    size_t raw_len = 0;
    for (size_t i = 0; i < rows; ++i) {
        int n = snprintf(
            (char *)(raw + raw_len),
            raw_cap - raw_len,
            "{\"ts\":\"2026-10-17T19:00:%02zuZ\",\"seq\":%zu,\"service\":\"auth-api\",\"status\":200}\n",
            i % 60,
            i
        );
        assert(n > 0);
        raw_len += (size_t)n;
    }

    /* 2. Preallocate compressed buffer using worst-case bound */
    vxg_ssize_t bound = vxg_compress_bound(raw_len, 0);
    assert(bound > 0);
    uint8_t *compressed = (uint8_t *)malloc((size_t)bound);
    assert(compressed != NULL);

    /* 3. Compress directly into `compressed` (threads=0 -> auto, block_size=0 -> 64 KiB, checksum=1) */
    vxg_ssize_t comp_len = vxg_compress_into(raw, raw_len, compressed, (size_t)bound, 0, 0, 1);
    if (comp_len < 0) {
        fprintf(stderr, "Compression failed: %s\n", vxg_error_message(comp_len));
        return 1;
    }
    printf(
        "Compressed %zu bytes -> %zd bytes (ratio: %.4f)\n",
        raw_len,
        comp_len,
        (double)comp_len / (double)raw_len
    );

    /* 4. O(1) header inspection and zero-allocation decompression into caller buffer */
    vxg_ssize_t expected_raw = vxg_frame_raw_size(compressed, (size_t)comp_len);
    vxg_ssize_t block_count = vxg_frame_block_count(compressed, (size_t)comp_len);
    assert(expected_raw == (vxg_ssize_t)raw_len);
    assert(block_count >= 1);

    uint8_t *restored = (uint8_t *)malloc((size_t)expected_raw);
    assert(restored != NULL);

    vxg_ssize_t dec_len = vxg_decompress_into(
        compressed,
        (size_t)comp_len,
        restored,
        (size_t)expected_raw,
        0,
        1
    );
    if (dec_len < 0) {
        fprintf(stderr, "Decompression failed: %s\n", vxg_error_message(dec_len));
        return 1;
    }
    assert((size_t)dec_len == raw_len);
    assert(memcmp(raw, restored, raw_len) == 0);

    /* 5. O(1) random-access single block extraction into caller buffer */
    vxg_ssize_t b0_size = vxg_frame_block_raw_size(compressed, (size_t)comp_len, 0);
    assert(b0_size > 0);
    uint8_t *block0 = (uint8_t *)malloc((size_t)b0_size);
    vxg_ssize_t b0_dec = vxg_extract_block_into(
        compressed,
        (size_t)comp_len,
        0,
        block0,
        (size_t)b0_size,
        1
    );
    assert(b0_dec == b0_size);
    assert(memcmp(raw, block0, (size_t)b0_size) == 0);

    printf("SUCCESS: Zero-copy C ABI roundtrip and O(1) block extraction verified (%zd blocks)!\n", block_count);

    free(block0);
    free(restored);
    free(compressed);
    free(raw);
    return 0;
}
