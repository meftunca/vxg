# VXG 0.4 — Unified Adaptive Multi-Format Lossless Block Codec (Format v4)

[![CI — Source Validation](https://github.com/meftunca/vxg/actions/workflows/ci.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/ci.yml)
[![CI — Cross-Platform Benchmarks](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml)
[![Release v0.4.0](https://img.shields.io/github/v/release/meftunca/vxg)](https://github.com/meftunca/vxg/releases/tag/v0.4.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**High-throughput Rust CPU (SIMD/SWAR) codec combining an adaptive structured-column engine (NDJSON / CSV / TSV / Logfmt), a 2D/Stride pixel engine (24-bpp & 32-bpp BMP), and a const-generic Fast-LZ + 4-stream canonical Huffman engine — with zero-overhead C ABI (`staticlib` / `cdylib`) and native Metal / CUDA / Vulkan sources.**

VXG `v4` inspects each independent 64 KiB block at runtime and routes it to the optimal lossless representation (`BLOCK_COLUMN`, `BLOCK_LZ`, or `BLOCK_RAW`). Every block is self-contained, enabling **$O(1)$ random-access block extraction** and **linear multi-core scaling** without cross-block decompression dependencies.

---

## Key Capabilities

| Area | Highlights |
| :--- | :--- |
| **Format v4 Wire Layout** | Compact 8-byte per-block directory (`payload_size \| kind << 30`, `crc32c`); full backward read compatibility with `v1`–`v3` frames |
| **Structured Column Engine** | Automatic schema & delimiter discovery for **NDJSON, CSV, TSV, and Logfmt**; specialized models for `Dictionary`, `DecimalDelta`, `DecimalSuffix`, `Timestamp` (SWAR+SIMD), `EpochCounter` (SIMD hex), `SegmentDictionary`, and `FrontCodedDict` |
| **Image & Stride Engine** | 24-bpp / 32-bpp BMP and multi-stride (`1, 2, 3, 4, 8`) byte-plane transpose + `Subtract-Green` (`R-G`, `B-G`) color decorrelation + 2-level 64-bit hierarchical zero-mask & per-plane `Fast-LZ` |
| **Prose & Source Code** | Const-generic `Fast-LZ` matcher paired with a 4-stream interleaved canonical Huffman coder |
| **Cross-Platform SIMD & SWAR** | Runtime-dispatched `NEON` (`aarch64`) and `SSE2` / `AVX2` / `AVX-512` (`x86_64`) fast paths, including 32-byte branchless string scanning and SIMD prefix-sum plane interleaving |
| **Integrity & Safety** | Hardware-accelerated CRC32C (`ARMv8 CRC` / `SSE4.2`); strict output-size bounds and deterministic rejection of truncated or malformed inputs |
| **Zero-Overhead Foreign FFI** | Ships as `rlib`, `staticlib` (`libvxg.a` / `vxg.lib`), and `cdylib` (`libvxg.dylib` / `.so` / `.dll`) with [`include/vxg.h`](include/vxg.h) and ready-to-run examples for **C/C++, Python, Go, Bun/Node.js, and C#** |

---

## Cross-Platform CI Benchmark Results (`macos-14`, `ubuntu-latest`, `windows-latest`)

All benchmarks below are executed automatically on GitHub Actions via [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) against native C **`liblz4`** (`lz4-c-a1`) and **`libzstd`** (`zstd-c-l1`) on `65,536`-byte independent blocks (`16 MiB` per dataset, single thread, checksums off).
- **Compression Ratio:** `compressed_bytes / raw_bytes` (**lower is smaller**).
- **Throughput:** Decimal `GB/s` (`10^9` uncompressed bytes/second; **higher is faster**).
- Full CSV and Markdown artifacts: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`reports/ci-v0.4.0/`](reports/ci-v0.4.0/).

### 1. macOS (`macos-14` / Apple Silicon `aarch64` NEON)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.909 GB/s** | 0.994 GB/s | 0.598 GB/s | **4.580 GB/s** | 4.582 GB/s | 1.507 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.182 GB/s** | 0.724 GB/s | 0.711 GB/s | **4.649 GB/s** | 2.967 GB/s | 1.738 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.013 GB/s** | 1.082 GB/s | 0.625 GB/s | **7.164 GB/s** | 5.235 GB/s | 1.854 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.288 GB/s** | 1.319 GB/s | 1.057 GB/s | **9.883 GB/s** | 5.622 GB/s | 2.203 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.643 GB/s** | 0.834 GB/s | 0.498 GB/s | **3.501 GB/s** | 1.900 GB/s | 1.097 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **0.832 GB/s** | 1.106 GB/s | 0.504 GB/s | **5.509 GB/s** | 3.879 GB/s | 1.426 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.450 GB/s** | 0.898 GB/s | 0.557 GB/s | **2.971 GB/s** | 2.323 GB/s | 1.119 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2889** | 0.4064 | 0.2933 | **0.253 GB/s** | 0.404 GB/s | 0.399 GB/s | **1.293 GB/s** | 2.592 GB/s | 1.169 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **4.117 GB/s** | 2.914 GB/s | 2.673 GB/s | **8.363 GB/s** | 13.012 GB/s | 3.238 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **4.171 GB/s** | 3.014 GB/s | 2.801 GB/s | **9.983 GB/s** | 6.762 GB/s | 5.538 GB/s |

### 2. Linux (`ubuntu-latest` / `x86_64` AVX2+SSE2)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.040 GB/s** | 1.003 GB/s | 0.578 GB/s | **3.640 GB/s** | 5.424 GB/s | 1.804 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.292 GB/s** | 1.190 GB/s | 0.790 GB/s | **2.806 GB/s** | 4.052 GB/s | 2.120 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.222 GB/s** | 1.448 GB/s | 0.810 GB/s | **3.884 GB/s** | 5.450 GB/s | 2.001 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.652 GB/s** | 1.892 GB/s | 1.075 GB/s | **3.885 GB/s** | 6.759 GB/s | 2.692 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.892 GB/s** | 1.076 GB/s | 0.614 GB/s | **2.301 GB/s** | 3.543 GB/s | 1.550 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.217 GB/s** | 1.510 GB/s | 0.814 GB/s | **4.751 GB/s** | 4.148 GB/s | 1.840 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.760 GB/s** | 1.101 GB/s | 0.572 GB/s | **2.227 GB/s** | 3.165 GB/s | 1.357 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2889** | 0.4064 | 0.2933 | **0.177 GB/s** | 0.518 GB/s | 0.285 GB/s | **1.023 GB/s** | 2.795 GB/s | 1.177 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **2.521 GB/s** | 6.405 GB/s | 3.761 GB/s | **9.652 GB/s** | 11.015 GB/s | 10.615 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **1.506 GB/s** | 5.880 GB/s | 3.915 GB/s | **4.532 GB/s** | 1.987 GB/s | 1.805 GB/s |

### 3. Windows (`windows-latest` / `x86_64` AVX2+SSE2)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.995 GB/s** | 0.898 GB/s | 0.386 GB/s | **3.764 GB/s** | 5.286 GB/s | 1.331 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.245 GB/s** | 0.785 GB/s | 0.562 GB/s | **3.223 GB/s** | 4.029 GB/s | 1.592 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.091 GB/s** | 0.896 GB/s | 0.577 GB/s | **4.260 GB/s** | 5.311 GB/s | 1.521 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.557 GB/s** | 1.043 GB/s | 0.541 GB/s | **4.428 GB/s** | 6.503 GB/s | 2.095 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.840 GB/s** | 0.731 GB/s | 0.406 GB/s | **2.971 GB/s** | 3.461 GB/s | 1.177 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.186 GB/s** | 0.913 GB/s | 0.507 GB/s | **4.830 GB/s** | 4.059 GB/s | 1.436 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.717 GB/s** | 0.801 GB/s | 0.345 GB/s | **2.275 GB/s** | 3.153 GB/s | 1.073 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2848** | 0.3993 | 0.2872 | **0.181 GB/s** | 0.449 GB/s | 0.268 GB/s | **1.042 GB/s** | 2.688 GB/s | 0.943 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **2.453 GB/s** | 2.191 GB/s | 1.800 GB/s | **9.536 GB/s** | 16.019 GB/s | 10.760 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **1.469 GB/s** | 2.060 GB/s | 1.853 GB/s | **4.536 GB/s** | 1.967 GB/s | 1.756 GB/s |

---

## Quick Start & CLI Usage (`vxg`)

Requires **Rust 1.89+**.

```sh
# Build CLI, staticlib (libvxg.a), and cdylib (libvxg.dylib / .so / .dll):
cargo build --release

# Inspect detected CPU SIMD & GPU backends:
./target/release/vxg hardware

# Compress and decompress using all available CPU threads (--threads auto):
./target/release/vxg compress input.ndjson input.vxg --threads auto
./target/release/vxg decompress input.vxg restored.ndjson --threads auto --force

# Verify frame structure and per-block CRC32C checksums without writing output:
./target/release/vxg verify input.vxg --threads auto

# Inspect frame header, overall ratio, and block-type breakdown (RAW / LZ / COLUMN):
./target/release/vxg info input.vxg --summary

# O(1) random-access single-block extraction (extracts block #0 without decoding other blocks):
./target/release/vxg block input.vxg 0 block0.bin --force

# Unix streaming pipeline via stdin / stdout ('-'):
cat input.ndjson | ./target/release/vxg c - - -q | ./target/release/vxg d - - -q > restored.ndjson
```

### Running the Multi-Format Benchmark Suite Locally

Requires `liblz4` and `libzstd` installed on your system (`brew install lz4 zstd` on macOS, `apt-get install liblz4-dev libzstd-dev` on Ubuntu/Debian, or `vcpkg install lz4 zstd` on Windows).

```sh
cargo run --release --features bench --bin vxg-bench -- \
  --suite v4 --size-mib 16 --samples 3 --seconds 0.05 \
  --csv reports/local-v4.csv --markdown reports/local-v4.md
```

---

## Using VXG from Other Languages (Static & Dynamic C ABI)

VXG compiles out of the box into both a **static archive** (`libvxg.a` / `vxg.lib`) and a **dynamic shared library** (`libvxg.dylib` / `libvxg.so` / `vxg.dll`), exposing a zero-allocation caller-buffer C ABI in [`include/vxg.h`](include/vxg.h). See **[`docs/FFI.md`](docs/FFI.md)** for full API documentation.

### Rust Library Usage
```rust
let input = b"{\"ts\":\"2026-10-17T19:00:00Z\",\"seq\":1,\"status\":\"ok\"}\n";
let frame = vxg::compress(input, &vxg::Options::default())?;
let output = vxg::decompress(&frame, &vxg::DecodeOptions::default())?;
assert_eq!(output, input);
```

### C / C++ Static or Dynamic Linking ([`examples/ffi/c_example.c`](examples/ffi/c_example.c))
```c
#include "vxg.h"

// 1. Compress into caller-provided buffer (threads=0 -> auto, block_size=0 -> 64 KiB, checksum=1):
vxg_ssize_t bound = vxg_compress_bound(raw_len, 0);
vxg_ssize_t c_len = vxg_compress_into(raw_ptr, raw_len, comp_buf, bound, 0, 0, 1);

// 2. O(1) header query + zero-allocation decode directly into caller buffer:
vxg_ssize_t raw_size = vxg_frame_raw_size(comp_buf, c_len);
vxg_ssize_t d_len = vxg_decompress_into(comp_buf, c_len, out_buf, raw_size, 0, 1);
```
```sh
cc -O3 -Iinclude examples/ffi/c_example.c target/release/libvxg.a -lpthread -ldl -lm -o c_example
```

### Python Zero-Copy `ctypes` + `memoryview` ([`examples/ffi/python_example.py`](examples/ffi/python_example.py))
```sh
python3 examples/ffi/python_example.py
```

### Go Zero-Copy `cgo` ([`examples/ffi/go_example.go`](examples/ffi/go_example.go))
```sh
CGO_CFLAGS="-I$(pwd)/include" CGO_LDFLAGS="$(pwd)/target/release/libvxg.a -lpthread -ldl -lm" \
  go run examples/ffi/go_example.go
```

### Bun / Node.js Zero-Copy `Uint8Array` FFI ([`examples/ffi/node_bun_example.mjs`](examples/ffi/node_bun_example.mjs))
```sh
bun run examples/ffi/node_bun_example.mjs
```

### C# / .NET 8+ Zero-Copy `Span<byte>` P/Invoke ([`examples/ffi/csharp_example.cs`](examples/ffi/csharp_example.cs))
Uses `[LibraryImport("vxg")]` and `fixed (byte* ...)` over `ReadOnlySpan<byte>` / `Span<byte>` for zero managed heap copies.

---

## Mathematical & Verification Scope

For an uncompressed input of $N$ bytes and block size $B$:

```text
encoded_size <= N + 32 + 32 * ceil(N / B)
CPU hash probes <= N
8 * accepted_matches <= matched_bytes <= N
```

- **Cross-checked SIMT & Model Verification:** `native/shaders/core.gpu` is validated across 110 encode/decode cases under ASan + UBSan (WG=8 and WG=64), 46 independent Python model cross-checks, 65,536 control words, 2,047 exhaustive small binary inputs, and 4,700 truncated prefix rejections.
- Detailed invariants and formal proof boundaries are documented in [`docs/PROOFS.md`](docs/PROOFS.md).

---

## Repository Layout

```text
src/                     Rust codec core (columns, stride/BMP, fast-LZ, Huffman, SIMD, CRC32C, C FFI)
src/bin/                 Production CLI (`vxg`) and multi-format benchmark harness (`vxg-bench`)
include/vxg.h            Official C99 / C++11 header for static & dynamic linking
examples/ffi/            Zero-copy integration examples for C, Python, Go, Bun/JS, and C#
native/shaders/core.gpu  Shared GPU kernel source (Metal, Vulkan GLSL, CUDA)
native/                  Metal, Vulkan, and CUDA host bridges
verification/            Independent Python reference model and C++ SIMT emulator
tests/                   Property tests, v3/v4 frame tests, and golden wire vectors
reports/ci-v0.4.0/       Cross-platform GitHub Actions benchmark outputs (macOS, Linux, Windows)
docs/                    Wire format, FFI guide, proofs, benchmarks, GPU architecture, and safety
```

[Wire Format](docs/FORMAT.md) · [FFI & Language Bindings](docs/FFI.md) · [Benchmarks](docs/BENCHMARKS.md) · [Proofs](docs/PROOFS.md) · [GPU Architecture](docs/GPU.md) · [Safety](docs/SAFETY.md) · [Validation](docs/VALIDATION.md)

---

## License

MIT License.
