# VXG 4.0.1 — Unified Adaptive Multi-Format Lossless Block Codec (Format v4)

[![CI — Source Validation](https://github.com/meftunca/vxg/actions/workflows/ci.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/ci.yml)
[![CI — Cross-Platform Benchmarks](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml)
[![Release v4.0.1](https://img.shields.io/github/v/release/meftunca/vxg)](https://github.com/meftunca/vxg/releases/tag/v4.0.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**High-throughput Rust CPU (SIMD/SWAR) codec combining an adaptive structured-column engine (NDJSON / CSV / TSV / Logfmt), a 2D/Stride pixel engine (24-bpp & 32-bpp BMP), and a const-generic Fast-LZ + 4-stream canonical Huffman engine — with zero-overhead C ABI (`staticlib` / `cdylib`) and native Metal / CUDA / Vulkan sources.**

VXG `v4` inspects each independent 64 KiB block at runtime and routes it to the optimal lossless representation (`BLOCK_COLUMN`, `BLOCK_LZ`, or `BLOCK_RAW`). Every block is self-contained, enabling **$O(1)$ random-access block extraction** and **linear multi-core scaling** without cross-block decompression dependencies.

---

## Key Capabilities

| Area | Highlights |
| :--- | :--- |
| **Format v4 Wire Layout** | Compact 8-byte per-block directory (`payload_size \| kind << 30`, `crc32c`); full backward read compatibility with `v1`–`v3` frames |
| **Structured Column Engine** | Automatic schema & delimiter discovery for **NDJSON, Pretty-Printed JSON, CSV, TSV, CLF/Bracketed Logs, and Logfmt**; specialized models for `Dictionary`, `DecimalDelta`, `DecimalSuffix`, `Timestamp` (SWAR+SIMD), `EpochCounter` (SIMD hex), `SegmentDictionary`, and `FrontCodedDict` (with 4-bit/5-bit nibble palettes and Fast-LZ dictionary streams) |
| **Image & Stride Engine** | 24-bpp / 32-bpp BMP and multi-stride (`1, 2, 3, 4, 8`) byte-plane transpose + `Subtract-Green` (`R-G`, `B-G`) color decorrelation + 2-level 64-bit hierarchical zero-mask & per-plane `Fast-LZ` |
| **Prose & Source Code** | Const-generic `Fast-LZ` matcher paired with a 4-stream interleaved canonical Huffman coder |
| **Cross-Platform SIMD & SWAR** | Runtime-dispatched `NEON` (`aarch64`) and `SSE2` / `AVX2` / `AVX-512` (`x86_64`) fast paths, including 32-byte branchless string scanning, 64-byte `FastPiece` unaligned `u128` row assembly, and SIMD prefix-sum plane interleaving |
| **Integrity & Safety** | Hardware-accelerated CRC32C (`ARMv8 CRC` / `SSE4.2`); strict output-size bounds and deterministic rejection of truncated or malformed inputs |
| **Zero-Allocation Rust & C ABI** | Zero-allocation `compress_into` / `decompress_into` / `extract_block_into` in Rust and C (`staticlib` `libvxg.a` / `vxg.lib`, `cdylib` `libvxg.dylib` / `.so` / `.dll`) with [`include/vxg.h`](include/vxg.h) and ready-to-run examples for **C/C++, Python, Go, Bun/Node.js, C#, and WebAssembly** |

---

## Local Apple Silicon Benchmark, Composite Index & Energy Highlights (1-Thread & 8-Thread, 16 MiB, 64 KiB Blocks)

Measured via `vxg-bench --suite v4 --size-mib 16 --samples 5 --seconds 0.08` against native C `liblz4` (`lz4-c-a1`) and `libzstd` (`zstd-c-l1`) on Apple Silicon (`aarch64` NEON):
- **VXG Performance Index (`VPI`):** $\text{VPI} = \frac{\sqrt{\text{Encode}_{\text{GB/s}} \cdot \text{Decode}_{\text{GB/s}}}}{\text{Ratio}}$ (higher is better).
- **Green Codec Index (`GCI`):** $\text{GCI} = \frac{\text{VPI}}{\sqrt{J_{\text{enc}} \cdot J_{\text{dec}}}}$ using per-process kernel hardware energy (`proc_pid_rusage` `RUSAGE_INFO_V6` `ri_energy_nj` in `J/GB`, see [`docs/ENERGY_BENCHMARKS.md`](docs/ENERGY_BENCHMARKS.md)).

| Dataset | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG 1T Enc / Dec (`GB/s`) | VXG 8T Enc / Dec (`GB/s`) | VXG 1T / 8T `VPI` (`vs LZ4`) | VXG 1T Enc / Dec (`J/GB`) | VXG 1T `GCI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | 1.704 / 9.420 | **11.122 / 47.196** | **98.4 / 562.4** (`5.20x` / `4.84x`) | 2.921 / 0.671 | **70.2** (`3.53x` / `10.80x`) |
| **`events.tsv`** | **0.0334** | 0.2223 | 0.0741 | 1.686 / 9.441 | **9.919 / 45.716** | **119.5 / 637.7** (`9.03x` / `8.28x`) | 2.952 / 0.653 | **86.0** (`8.27x` / `8.69x`) |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | 1.861 / 10.725 | **11.470 / 49.538** | **126.6 / 675.3** (`7.30x` / `7.52x`) | 2.683 / 0.590 | **100.6** (`5.47x` / `9.86x`) |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | 2.202 / 12.531 | **13.772 / 54.798** | **231.8 / 1212.0** (`7.61x` / `8.08x`) | 2.108 / **0.466** | **233.9** (`5.86x` / `10.73x`) |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | 1.322 / 5.583 | **6.190 / 30.762** | **61.1 / 310.2** (`5.28x` / `4.90x`) | 3.777 / **0.900** | **33.1** (`4.19x` / `8.95x`) |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | 1.756 / 8.807 | **10.532 / 44.595** | **158.4 / 872.9** (`7.35x` / `7.29x`) | 2.711 / **0.614** | **122.8** (`6.64x` / `15.54x`) |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | 1.154 / 4.523 | **7.789 / 27.442** | **47.6 / 304.5** (`4.02x` / `3.97x`) | 3.914 / 1.169 | **22.2** (`2.85x` / `8.54x`) |
| **`source-and-docs.txt`** | **0.2782** | 0.3830 | 0.2755 | **0.614 / 2.478** | **4.306 / 16.844** | **4.4 / 30.6** (`1.01x` / `1.02x`) | **6.887 / 1.718** | **1.3** (`0.68x` / `2.17x`) |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0091 | 9.533 / 14.246 | **36.922 / 63.932** | **2435.3 / 10153.2** (`2.09x` / `2.47x`) | 0.562 / 0.345 | **5530.4** (`1.22x` / `5.24x`) |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 8.793 / 14.846 | **35.666 / 64.509** | **9206.1 / 38648.9** (`19.42x` / `19.56x`) | 0.631 / 0.368 | **19116.6** (`13.67x` / `17.18x`) |
| **10-Dataset Geometric Mean** | **0.0250** | 0.1272 | 0.0607 | **2.094 / 8.211** | **11.750 / 41.545** | **165.9 / 883.8** (`5.27x` / `5.29x`) | **2.333 / 0.665** | **133.2** (`3.92x` / `8.59x`) |

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
let opts = vxg::Options::default();
let dec_opts = vxg::DecodeOptions::default();

// Allocating convenience API:
let frame = vxg::compress(input, &opts)?;
let output = vxg::decompress(&frame, &dec_opts)?;
assert_eq!(output, input);

// Zero-allocation caller-buffer API (single-thread and multi-thread):
let mut comp_buf = vec![0u8; vxg::compress_bound(input.len(), opts.block_size)?];
let c_len = vxg::compress_into(input, &mut comp_buf, &opts)?;
let mut out_buf = vec![0u8; vxg::frame_raw_size(&comp_buf[..c_len])?];
let d_len = vxg::decompress_into(&comp_buf[..c_len], &mut out_buf, &dec_opts)?;
assert_eq!(&out_buf[..d_len], input);
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

### WebAssembly (`wasm32-unknown-unknown` — Browser, Node.js, Bun, Cloudflare Workers) ([`examples/ffi/wasm_example.mjs`](examples/ffi/wasm_example.mjs))
Compiles to a standalone `vxg.wasm` module with **zero WASI imports** (`WebAssembly.instantiate(bytes, {})`) and zero `wasm-bindgen` dependencies:
```sh
rustup target add wasm32-unknown-unknown
cargo build --release --lib --target wasm32-unknown-unknown
node examples/ffi/wasm_example.mjs
```

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
src/                     Rust codec core (lib.rs, ffi.rs, simd.rs, crc32c.rs, gpu.rs, analysis.rs)
src/columns/             Modular structured-column, stride/BMP, Fast-LZ, and canonical Huffman submodules
src/frame/               Format v1–v4 frame encoder, parser, and zero-allocation parallel decoder
src/block/               Block LZ encoder and SIMD/SWAR decoder submodules
src/packing/             Command-metadata & entropy research submodules (--features bench)
src/groups/              Fixed-group command-layout evaluator submodules (--features bench)
src/bin/                 Production CLI (`vxg`) and multi-format benchmark harness (`vxg-bench`)
include/vxg.h            Official C99 / C++11 header for static & dynamic linking
examples/ffi/            Zero-copy integration examples for C, Python, Go, Bun/JS, C#, and WebAssembly
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
