# VXG 0.4.1 — Unified Adaptive Multi-Format Lossless Block Codec (Format v4)

[![CI — Source Validation](https://github.com/meftunca/vxg/actions/workflows/ci.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/ci.yml)
[![CI — Cross-Platform Benchmarks](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml)
[![Release v0.4.1](https://img.shields.io/github/v/release/meftunca/vxg)](https://github.com/meftunca/vxg/releases/tag/v0.4.1)
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
| **Zero-Allocation & Streaming API** | Zero-allocation `compress_into` / `decompress_into` / `extract_block_into` plus bounded-memory `StreamEncoder<W>` / `StreamDecoder<R>` (`std::io::Write` / `Read`) and multi-frame concatenated stream support in Rust and C (`staticlib` `libvxg.a` / `vxg.lib`, `cdylib` `libvxg.dylib` / `.so` / `.dll`) with [`include/vxg.h`](include/vxg.h) |

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
| **`source-and-docs.txt`** | **0.2698** | 0.3777 | 0.2713 | **0.592 / 2.468** | **4.306 / 16.844** | **4.5 / 31.6** (`1.02x` / `1.05x`) | **6.887 / 1.718** | **1.3** (`0.68x` / `2.17x`) |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0091 | 9.533 / 14.246 | **36.922 / 63.932** | **2435.3 / 10153.2** (`2.09x` / `2.47x`) | 0.562 / 0.345 | **5530.4** (`1.22x` / `5.24x`) |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 8.793 / 14.846 | **35.666 / 64.509** | **9206.1 / 38648.9** (`19.42x` / `19.56x`) | 0.631 / 0.368 | **19116.6** (`13.67x` / `17.18x`) |
| **10-Dataset Geometric Mean** | **0.0250** | 0.1272 | 0.0607 | **2.094 / 8.211** | **11.750 / 41.545** | **165.9 / 883.8** (`5.27x` / `5.29x`) | **2.333 / 0.665** | **133.2** (`3.92x` / `8.59x`) |

---

## Cross-Platform CI Benchmark Results — VXG `v0.4.1` (`macos-14`, `ubuntu-latest`, `windows-latest`)

All benchmarks below are executed automatically on GitHub Actions via [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) against native C **`liblz4`** (`lz4-c-a1`) and **`libzstd`** (`zstd-c-l1`) on `65,536`-byte independent blocks (`16 MiB` per dataset, single thread, checksums off).
- **Compression Ratio:** `compressed_bytes / raw_bytes` (**lower is smaller**).
- **Throughput:** Decimal `GB/s` (`10^9` uncompressed bytes/second; **higher is faster**).
- **VXG Performance Index (`VPI`):** `sqrt(Encode_GB/s * Decode_GB/s) / Ratio` (**higher is better**).
- Full CSV and Markdown artifacts: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`reports/ci-v0.4.1/`](reports/ci-v0.4.1/).

### 1. macOS (`macos-14` / Apple Silicon `aarch64` NEON)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.296 GB/s** | 0.982 GB/s | 0.730 GB/s | **5.954 GB/s** | 5.514 GB/s | 1.739 GB/s | **68.2** (`5.84x` / `5.09x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.187 GB/s** | 0.952 GB/s | 0.751 GB/s | **7.044 GB/s** | 4.026 GB/s | 1.715 GB/s | **86.6** (`9.83x` / `5.65x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.614 GB/s** | 1.205 GB/s | 0.770 GB/s | **8.613 GB/s** | 5.748 GB/s | 2.104 GB/s | **105.6** (`8.05x` / `5.77x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.618 GB/s** | 1.437 GB/s | 0.917 GB/s | **9.242 GB/s** | 6.288 GB/s | 2.811 GB/s | **170.6** (`8.50x` / `5.77x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.175 GB/s** | 0.873 GB/s | 0.616 GB/s | **4.980 GB/s** | 3.274 GB/s | 1.348 GB/s | **54.4** (`6.86x` / `6.22x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.281 GB/s** | 1.260 GB/s | 0.770 GB/s | **5.830 GB/s** | 4.055 GB/s | 1.597 GB/s | **110.1** (`7.52x` / `7.73x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.940 GB/s** | 0.985 GB/s | 0.671 GB/s | **3.503 GB/s** | 3.411 GB/s | 1.467 GB/s | **37.8** (`4.26x` / `4.56x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2702** | 0.3778 | 0.2714 | **0.374 GB/s** | 0.496 GB/s | 0.390 GB/s | **1.696 GB/s** | 3.200 GB/s | 1.084 GB/s | **2.9** (`0.88x` / `1.23x`) |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.423 GB/s** | 3.846 GB/s | 2.456 GB/s | **9.170 GB/s** | 14.402 GB/s | 2.719 GB/s | **1603.8** (`3.02x` / `5.66x`) |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **6.025 GB/s** | 4.357 GB/s | 3.669 GB/s | **9.611 GB/s** | 8.715 GB/s | 5.913 GB/s | **6131.5** (`22.59x` / `15.71x`) |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.554 GB/s** | 1.307 GB/s | 0.920 GB/s | **5.896 GB/s** | 5.205 GB/s | 1.990 GB/s | **121.4** (`5.91x` / `5.45x`) |

### 2. Linux (`ubuntu-latest` / `x86_64` AVX2+SSE2)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.169 GB/s** | 1.019 GB/s | 0.582 GB/s | **3.458 GB/s** | 5.541 GB/s | 1.858 GB/s | **49.4** (`4.14x` / `3.99x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.101 GB/s** | 1.213 GB/s | 0.799 GB/s | **2.699 GB/s** | 4.096 GB/s | 2.153 GB/s | **51.6** (`5.15x` / `2.92x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.489 GB/s** | 1.481 GB/s | 0.817 GB/s | **3.905 GB/s** | 5.551 GB/s | 2.063 GB/s | **68.3** (`4.78x` / `3.66x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.745 GB/s** | 1.952 GB/s | 1.091 GB/s | **3.599 GB/s** | 6.901 GB/s | 2.737 GB/s | **110.6** (`4.51x` / `3.47x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.935 GB/s** | 1.112 GB/s | 0.622 GB/s | **2.851 GB/s** | 3.549 GB/s | 1.575 GB/s | **36.7** (`3.94x` / `3.87x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.390 GB/s** | 1.554 GB/s | 0.822 GB/s | **4.107 GB/s** | 4.179 GB/s | 1.861 GB/s | **96.3** (`5.83x` / `6.06x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.837 GB/s** | 1.126 GB/s | 0.578 GB/s | **2.067 GB/s** | 3.158 GB/s | 1.376 GB/s | **27.4** (`3.00x` / `3.67x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2714** | 0.3786 | 0.2720 | **0.366 GB/s** | 0.551 GB/s | 0.306 GB/s | **1.369 GB/s** | 2.929 GB/s | 1.216 GB/s | **2.6** (`0.78x` / `1.16x`) |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.338 GB/s** | 6.793 GB/s | 4.102 GB/s | **9.654 GB/s** | 11.186 GB/s | 11.375 GB/s | **1634.7** (`2.63x` / `2.18x`) |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **4.601 GB/s** | 6.506 GB/s | 4.147 GB/s | **6.818 GB/s** | 1.993 GB/s | 1.830 GB/s | **4512.7** (`28.46x` / `19.55x`) |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.442 GB/s** | 1.671 GB/s | 0.956 GB/s | **3.513 GB/s** | 4.390 GB/s | 2.169 GB/s | **90.2** (`4.23x` / `3.80x`) |

### 3. Windows (`windows-latest` / `x86_64` AVX2+SSE2)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **2.183 GB/s** | 1.381 GB/s | 0.896 GB/s | **9.611 GB/s** | 9.539 GB/s | 2.619 GB/s | **112.5** (`6.18x` / `6.17x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **2.105 GB/s** | 1.212 GB/s | 0.989 GB/s | **10.302 GB/s** | 6.762 GB/s | 2.955 GB/s | **139.4** (`10.83x` / `6.04x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **2.278 GB/s** | 1.340 GB/s | 0.981 GB/s | **8.906 GB/s** | 8.571 GB/s | 2.843 GB/s | **127.6** (`7.56x` / `5.31x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.677 GB/s** | 1.490 GB/s | 1.155 GB/s | **14.200 GB/s** | 11.543 GB/s | 3.726 GB/s | **272.1** (`9.82x` / `7.12x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.521 GB/s** | 0.987 GB/s | 0.748 GB/s | **6.210 GB/s** | 4.672 GB/s | 2.018 GB/s | **69.1** (`6.86x` / `5.86x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **2.141 GB/s** | 1.268 GB/s | 0.838 GB/s | **9.216 GB/s** | 5.386 GB/s | 2.279 GB/s | **179.0** (`10.58x` / `10.09x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **1.165 GB/s** | 1.111 GB/s | 0.724 GB/s | **3.282 GB/s** | 3.963 GB/s | 1.710 GB/s | **40.7** (`4.01x` / `4.38x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2677** | 0.3720 | 0.2667 | **0.575 GB/s** | 0.735 GB/s | 0.514 GB/s | **2.508 GB/s** | 5.108 GB/s | 1.756 GB/s | **4.5** (`0.86x` / `1.26x`) |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **7.075 GB/s** | 3.121 GB/s | 2.749 GB/s | **14.442 GB/s** | 41.456 GB/s | 18.079 GB/s | **2112.5** (`2.60x` / `2.73x`) |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **6.639 GB/s** | 3.321 GB/s | 2.786 GB/s | **9.212 GB/s** | 4.583 GB/s | 3.950 GB/s | **6301.0** (`36.68x` / `22.67x`) |
| **Geometric Mean** | — | **0.0249** | 0.1268 | 0.0606 | **2.217 GB/s** | 1.429 GB/s | 1.060 GB/s | **7.753 GB/s** | 7.569 GB/s | 3.096 GB/s | **166.5** (`6.42x` / `5.56x`) |

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
