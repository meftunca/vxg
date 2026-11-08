---
layout: home

hero:
  name: "VXG Codec"
  text: "Vectorized Execution Grammar"
  tagline: "Store raw NDJSON/CSV as-is (14x–21x faster ingest than Parquet ETL), 1.18x–1.32x smaller 64 KiB blocks than Zstd + Trained Dict on flat tabular records at LZ4-class speed, SQL on compressed files, and 100% byte-exact original recovery."
  actions:
    - theme: brand
      text: Try Live WASM Playground →
      link: /playground
    - theme: brand
      text: AI Tensor & Metal LLM (.vxt) →
      link: /TENSOR
    - theme: alt
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Real Corpora & DuckDB Benchmarks
      link: /CHIMERA_BENCHMARKS
    - theme: alt
      text: View on GitHub
      link: https://github.com/meftunca/vxg

features:
  - title: "Zero-ETL Lossless Archive + SQL"
    details: "Keep raw JSON/CSV verbatim without lossy Parquet ETL (14x–21x faster ingest), run SQL directly on compressed 64 KiB random-access blocks, and recover the original file byte-for-byte."
  - title: "Honest Real-Corpus Trade-Offs"
    details: "1.18x–1.32x smaller than Zstd + Trained Dict on flat tabular records (NYC TLC NDJSON/CSV); Zstd + Trained Dict leads on nested GitHub JSON, free-text logs, and prose (0.1205 vs 0.1484 geomean)."
  - title: "Calibrated vs DuckDB 2.0 + Parquet"
    details: "22x–540x faster than DuckDB on raw/.zst (0.21–3.84 ms vs 51–158 ms on 200k–300k in-cache rows), and competitive with or faster than prepared DuckDB + Parquet on CSV scans, Top-K, and projections."
  - title: "vxg-tensor (.vxt) AI Model Codec & Metal LLM"
    details: "Zero-dependency LLM weight compression (1.41x bit-exact Lossless-BF16, 3.62x CB4+Outlier beating Q4_0 in both size & KL, 4.66x–5.57x CB3/WHT2) with 105–137 tok/s 1-CmdBuf Apple Metal GPU inference."
---

## Why VXG?

Traditional byte-stream LZ compressors (`LZ4`, `Zstd`) scan inputs sequentially through a sliding window. On structured logs, telemetry tables, and columnar JSON streams, they repeatedly rediscover field names, delimiters, timestamps, and numeric formatting on every row.

**VXG (`VXG1` v4)** splits each independent block (`64 KiB` default) into a **Vectorized Execution Grammar (`GRAMMAR_PACKET`)**, **Strided Decorrelation Planes (`STRIDE_PACKET`)**, or **4-Stream Canonical Huffman + Fast-LZ (`FAST_LZ_PACKET`)**, selected strictly by exact byte budget. For public real-world corpora (`NYC TLC Taxi`, `GitHub Archive`, `Kodak` true-color photos), `Zstd + Trained Dictionary`, whole-file `zstd -19 --long`, and **DuckDB 2.0 + Parquet** head-to-head benchmarks, see **[Competitor Sweep & DuckDB](/CHIMERA_BENCHMARKS)**.

<CodecBenchmarkChart compact />

| Metric (`--suite v4` Synthetic Qualification Geomean, `16 MiB`) | **VXG v0.4** | **LZ4 (`1.10.0`)** | **Zstd (`-1`, `1.5.7`)** | **Trade-Off Notes** |
| :--- | :---: | :---: | :---: | :--- |
| **8-Dataset Structured/Text Ratio** *(excl. procedural BMP)* | **`0.0451` (`22.2:1`)** | `0.2103` (`4.8:1`) | `0.0957` (`10.4:1`) | **`4.66x` smaller than LZ4, `2.12x` smaller than Zstd-1** (`0.0845` Zstd+Dict-9) |
| **10-Dataset Ratio** *(incl. 2 procedural UI BMPs)* | **`0.0250` (`40.0:1`)** | `0.1272` (`7.9:1`) | `0.0607` (`16.5:1`) | **`5.09x` smaller than LZ4, `2.43x` smaller than Zstd-1** |
| **1-Thread Encode / Decode (`GB/s`)** | `2.09` / **`8.21 GB/s`** | **`2.31`** / `6.93 GB/s` | `1.57 / 2.67 GB/s` | **LZ4 encodes `1.10x` faster; VXG decodes `1.18x` faster** |
| **8-Thread Encode / Decode (`GB/s`)** | `11.75` / **`41.55 GB/s`** | **`12.38`** / `36.50 GB/s` | `8.77 / 16.04 GB/s` | **LZ4 encodes `1.05x` faster; VXG decodes `1.14x` faster** |
| **Decode Energy (`J/GB`, 1T)** *(lower is better)* | `0.665 J/GB` | **`0.619 J/GB`** | `1.641 J/GB` | **`2.47x` less decode energy than Zstd-1** |
| **Green Codec Index (`GCI`, 1T / 8T)** | **`133.2 / 728.7`** | `34.0 / 184.1` | `15.5 / 94.6` | **`3.92x` higher `GCI` than LZ4, `8.59x` higher than Zstd-1** |

### Cross-Platform GitHub Actions CI Summary (`1T`, `16 MiB` Geomean)

| CI Runner (`OS / Arch / SIMD`) | **VXG v4 Ratio** | **VXG v4 Encode / Decode** | **LZ4 Encode / Decode** | **Zstd-1 Encode / Decode** | **VXG `VPI` (`vs LZ4` / `Zstd-1`)** | **VXG `BCI` (`vs LZ4` / `Zstd-1`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **macOS (`macos-14` / `aarch64` NEON)** | **`0.0249`** | **`1.554 / 5.896 GB/s`** | `1.307 / 5.205 GB/s` | `0.920 / 1.990 GB/s` | **`121.4` (`5.91x` / `5.45x`)** | **`7.17` (`3.76` / `3.11`)** |
| **Linux (`ubuntu-latest` / `x86_64` AVX2)** | **`0.0249`** | `1.984` / **`6.822 GB/s`** | **`2.038`** / `5.994 GB/s` | `1.308 / 2.699 GB/s` | **`147.5` (`5.36x` / `4.76x`)** | **`8.16` (`4.58` / `3.88`)** |
| **Windows (`windows-latest` / `x86_64` AVX2)** | **`0.0249`** | **`2.295 / 8.362 GB/s`** | `1.520 / 7.519 GB/s` | `1.110 / 3.080 GB/s` | **`175.9` (`6.60x` / `5.76x`)** | **`9.17` (`4.48` / `3.84`)** |

### Compressed-Domain SQL Queries, `Top-K`, Analytics & Column Splicing

<CompressedOpsChart />

### Compressed-Domain AI Model Weights & Apple Metal GPU LLM Inference (`vxg-tensor` / `.vxt`)

`crates/vxg-tensor` brings VXG's compressed-domain philosophy to LLM weights (`GGUF` & `SafeTensors` -> `.vxt` `VXT1`), executing `Qwen3.5-2B` directly from compressed Unified Memory planes inside a single concurrent `MTLCommandBuffer` per token ([Full Specification & Benchmarks →](/TENSOR)):

<TensorBenchmarkChart compact />

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
# Build from source
cargo build --release --bin vxg

# Compress using all CPU cores (-t auto)
./target/release/vxg compress telemetry.ndjson telemetry.vxg -t auto -f

# Inspect frame header, compression ratio, and block-type summary
./target/release/vxg info telemetry.vxg --summary

# Compressed-domain SELECT + compound WHERE + ORDER BY Top-K + GROUP BY without full decode
./target/release/vxg query telemetry.vxg --select ts,service,latency_ms --where 'status==500' --order-by latency_ms --desc --limit 10 -t auto
./target/release/vxg agg telemetry.vxg --group-by service --col bytes --where 'status==500' --order-by sum --desc -t auto

# Zero-reencode VXG -> VXG column splicing & Unix stdin/stdout streaming
./target/release/vxg project telemetry.vxg slim.vxg --cols service,status,bytes -t auto -f
cat access.log | ./target/release/vxg c - - --stream -t auto -q | ./target/release/vxg d - - --stream -t auto -q > access.restored.log
```
