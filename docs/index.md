---
layout: home

hero:
  name: "VXG Codec"
  text: "Vectorized Execution Grammar"
  tagline: "5.1x smaller than LZ4, 2.4x smaller than Zstd, up to 54 GB/s parallel decode, and O(1) random block access across ARM64 NEON, x86_64 AVX2/AVX-512, and WebAssembly."
  actions:
    - theme: brand
      text: Get Started →
      link: /guide/getting-started
    - theme: alt
      text: Cross-Platform Benchmarks
      link: /BENCHMARKS
    - theme: alt
      text: View on GitHub
      link: https://github.com/meftunca/vxg

features:
  - title: "Grammar + Columnar + SIMD LZ"
    details: "Automatically discovers repeating record templates in CSV, TSV, logfmt, NDJSON, pretty JSON, and CLF logs while seamlessly handling prose/code (4-stream canonical Huffman + Fast-LZ) and 2D/strided media frames (SubGreen + sparse bitmask)."
  - title: "5.27x Higher VPI & 8.59x Higher Green Codec Index"
    details: "Across the 10-dataset v4 qualification suite (16 MiB), VXG achieves a 0.0250 geometric-mean compression ratio (40:1) at 2.09 GB/s encode and 8.21 GB/s decode (1T) — scaling to 11.75 GB/s encode and 41.55 GB/s decode on 8 threads."
  - title: "Zero-Allocation Caller Buffers & O(1) Random Access"
    details: "Every 64 KiB block is 100% self-contained with an 8-byte compact v4 directory entry. Inspect uncompressed sizes or extract any single block in O(1) time directly into caller-owned slices with zero per-block heap allocation."
  - title: "Universal Zero-Dependency Footprint"
    details: "Zero external runtime dependencies in default builds. Compiles simultaneously as a Rust library (rlib), static C archive (libvxg.a / vxg.lib), shared library (.so / .dylib / .dll), and dependency-free WebAssembly (vxg.wasm)."
---

## Why VXG?

Traditional byte-stream LZ compressors (`LZ4`, `Zstd`) scan inputs sequentially through a sliding window. On structured logs, telemetry tables, and columnar JSON streams, they repeatedly rediscover field names, delimiters, timestamps, and numeric formatting on every row.

**VXG (`VXG1` v4)** splits each independent block (`64 KiB` default) into a **Vectorized Execution Grammar (`GRAMMAR_PACKET`)**, **Strided Decorrelation Planes (`STRIDE_PACKET`)**, or **4-Stream Canonical Huffman + Fast-LZ (`FAST_LZ_PACKET`)**, selected strictly by exact byte budget:

| Metric (10-Dataset Geometric Mean, `16 MiB`) | **VXG v0.4** | **LZ4 (`1.10.0`)** | **Zstd (`-1`, `1.5.7`)** | **VXG Advantage** |
| :--- | :---: | :---: | :---: | :--- |
| **Compression Ratio** *(lower is better)* | **`0.0250` (`40.0:1`)** | `0.1272` (`7.9:1`) | `0.0607` (`16.5:1`) | **`5.09x` smaller than LZ4, `2.43x` smaller than Zstd-1** |
| **1-Thread Encode / Decode (`GB/s`)** | **`2.09 / 8.21 GB/s`** | `2.31 / 6.93 GB/s` | `1.57 / 2.67 GB/s` | **`1.18x` faster decode than LZ4, `3.08x` faster than Zstd-1** |
| **8-Thread Encode / Decode (`GB/s`)** | **`11.75 / 41.55 GB/s`** | `12.38 / 36.50 GB/s` | `8.77 / 16.04 GB/s` | **`1.14x` faster decode than LZ4, `2.59x` faster than Zstd-1** |
| **VXG Performance Index (`VPI`, 1T / 8T)** | **`165.9 / 883.8`** | `31.5 / 167.2` | `33.7 / 195.3` | **`5.27x` vs LZ4, `4.93x` vs Zstd-1** |
| **Decode Energy (`J/GB`, 1T)** *(lower is better)* | **`0.665 J/GB`** | `0.619 J/GB` | `1.641 J/GB` | **`2.47x` less decode energy than Zstd-1** |
| **Green Codec Index (`GCI`, 1T / 8T)** | **`133.2 / 728.7`** | `34.0 / 184.1` | `15.5 / 94.6` | **`3.92x` greener than LZ4, `8.59x` greener than Zstd-1** |

---

## Quick Start in 30 Seconds

### 1. Rust Library (`Cargo.toml`)

```rust
use vxg::{Options, StreamDecoder, StreamEncoder};

fn main() -> vxg::Result<()> {
    let raw = b"ts=2026-06-01T00:00:00Z level=INFO service=auth latency_us=120\n".repeat(1000);
    let opts = Options::default(); // VXG v4, 64 KiB blocks, CRC32C enabled

    // 1. One-shot compress & zero-allocation caller-buffer decompress
    let compressed = vxg::compress(&raw, &opts)?;
    let mut restored = vec![0u8; vxg::frame_raw_size(&compressed)?];
    vxg::decompress_into(&compressed, &mut restored, &opts)?;
    assert_eq!(restored, raw);

    // 2. O(1) random-access single-block extraction
    let block0_len = vxg::frame_block_raw_size(&compressed, 0)?;
    let mut block0 = vec![0u8; block0_len];
    vxg::extract_block_into(&compressed, 0, &mut block0, &opts)?;
    assert_eq!(&block0[..], &raw[..block0_len]);

    Ok(())
}
```

### 2. Command-Line Interface (`vxg`)

```sh
# Install or build from source
cargo build --release --bin vxg

# Compress with auto-detected CPU threads (-T0)
./target/release/vxg compress -T0 telemetry.ndjson -o telemetry.ndjson.vxg

# Inspect frame header, compression ratio, and per-block kinds
./target/release/vxg info --blocks telemetry.ndjson.vxg

# Extract only block #0 in O(1) time without decoding the rest of the file
./target/release/vxg block telemetry.ndjson.vxg 0 -o block0.ndjson

# Stream via stdin/stdout in Unix pipelines
cat access.log | ./target/release/vxg -c -T0 | ./target/release/vxg -d > access.restored.log
```
