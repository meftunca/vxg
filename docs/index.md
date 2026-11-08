---
layout: home

hero:
  name: "VXG Codec v0.5.1"
  text: "Vectorized Execution Grammar"
  tagline: "Store raw NDJSON/CSV verbatim (2.4x–16.7x faster ingest than Parquet ETL), 1.17x–1.32x smaller 64 KiB blocks than Zstd + Trained Dict on flat tabular records, beat Zstd-9 & Zstd+Dict-3 on the 6-dataset real public geomean (0.1213, within 0.66% of Zstd+Dict-9), SQL on 40M+ compressed rows, and 100% byte-exact original recovery."
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
      text: Real Corpora & 40M DuckDB Benchmarks
      link: /CHIMERA_BENCHMARKS
    - theme: alt
      text: View on GitHub
      link: https://github.com/meftunca/vxg

features:
  - title: "Zero-ETL Lossless Archive + SQL (`vxg sql`)"
    details: "Keep raw JSON/CSV verbatim without lossy Parquet ETL, run SQL (`SELECT`, `WHERE`, `GROUP BY`, `ORDER BY Top-K`) directly on compressed blocks (CSV header names & dotted nested JSON), and recover the original file byte-for-byte."
  - title: "Honest Real-Corpus Trade-Offs (`Format v4/v5`)"
    details: "1.17x–1.32x smaller than Zstd + Trained Dict L9 on flat tabular records; pure-Rust L1D u16 + PreparedDict BLOCK_DICT_LZ brings the 6-dataset real geomean to 0.1213 — beating Zstd-9 (0.1419) and Zstd+Dict-3 (0.1407), within 0.66% of Zstd+Dict-9 (0.1205)."
  - title: "Calibrated vs DuckDB 2.0 + Parquet (Up to 40M Rows)"
    details: "Wins 15/15 in-cache 8T queries (0.16–1.32 ms) and 5/5 out-of-cache 8T queries on 40M-row (3.06 GB) NYC TLC CSV (17.5–44.4 ms), while DuckDB + Parquet leads 1T NDJSON and 10M-row NDJSON selective zone-map scans."
  - title: "vxg-tensor (.vxt) AI Model Codec & Metal LLM"
    details: "Zero-dependency LLM weight compression (1.41x bit-exact Lossless-BF16, 3.62x CB4+Outlier beating Q4_0 in both size & KL, 4.66x–5.57x CB3/WHT2) with 105–137 tok/s 1-CmdBuf Apple Metal GPU inference."
---

## Why VXG?

Traditional byte-stream LZ compressors (`LZ4`, `Zstd`) scan inputs sequentially through a sliding window. On structured logs, telemetry tables, and columnar JSON streams, they repeatedly rediscover field names, delimiters, timestamps, and numeric formatting on every row.

**VXG (`VXG1` v4/v5)** splits each independent block (`64 KiB` default, or `\n`-snapped `--record-aligned`) into a **Vectorized Execution Grammar (`GRAMMAR_PACKET`)**, **Self-Contained Frame Dictionary (`<= 256 KiB`) + 5-Stream Canonical Huffman (`BLOCK_DICT_LZ`)**, **Strided Decorrelation Planes (`STRIDE_PACKET`)**, or **Fast-LZ (`FAST_LZ_PACKET`)**, selected strictly by exact byte budget. For public real-world corpora (`NYC TLC Taxi`, `GitHub Archive`, `Kodak` true-color photos), `Zstd + Trained Dictionary`, whole-file `zstd -19 --long`, and **DuckDB 2.0 + Parquet (`200k` to `40M` rows)** head-to-head benchmarks, see **[Competitor Sweep & DuckDB](/CHIMERA_BENCHMARKS)**.

<CodecBenchmarkChart compact />

| Metric (`--suite v4` Synthetic Qualification Geomean, `16 MiB`) | **VXG v0.5.1** | **LZ4 (`1.10.0`)** | **Zstd (`-1`, `1.5.7`)** | **Trade-Off Notes** |
| :--- | :---: | :---: | :---: | :--- |
| **8-Dataset Structured/Text Ratio** *(excl. procedural BMP)* | **`0.0259` (`38.6:1`)** | `0.2107` (`4.7:1`) | `0.0959` (`10.4:1`) | **`8.14x` smaller than LZ4, `3.70x` smaller than Zstd-1** (`0.0846` Zstd+Dict-9; `source-and-docs.txt` `0.0036`) |
| **10-Dataset Ratio** *(incl. 2 procedural UI BMPs)* | **`0.0162` (`61.7:1`)** | `0.1286` (`7.8:1`) | `0.0615` (`16.3:1`) | **`7.94x` smaller than LZ4, `3.80x` smaller than Zstd-1** |
| **1-Thread Encode / Decode (`GB/s`, 10-set)** | `2.22` / **`10.53 GB/s`** | **`2.27`** / `6.94 GB/s` | `1.55 / 2.66 GB/s` | **Beats LZ4 on 8-set 1T Enc & Dec (`1.58 / 9.69` vs `1.51 / 5.79 GB/s`); `1.52x` faster 10-set decode** |
| **8-Thread Encode / Decode (`GB/s`, 10-set)** | `8.51` / **`54.97 GB/s`** | **`10.96`** / `32.11 GB/s` | `7.73 / 14.07 GB/s` | **LZ4 encodes `1.29x` faster; VXG decodes `1.71x` faster** (`94.5 GB/s` on `source-and-docs.txt`) |
| **Decode Energy (`J/GB`, 1T)** *(lower is better)* | **`0.533 J/GB`** | `0.634 J/GB` | `1.641 J/GB` | **`1.19x` less decode energy than LZ4, `3.08x` less than Zstd-1** |
| **Green Codec Index (`GCI`, 1T / 8T)** | **`275.5 / 1,343.8`** | `32.4 / 157.9` | `15.5 / 94.6` | **`8.51x` higher `GCI` than LZ4, `14.2x`–`17.8x` higher than Zstd-1** |

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

# Compress using all CPU cores (-t auto) with newline-aligned blocks (--record-aligned)
./target/release/vxg compress telemetry.ndjson telemetry.vxg --record-aligned -t auto -f

# Inspect frame header, compression ratio, and block-type summary
./target/release/vxg info telemetry.vxg --summary

# Compressed-domain SQL (`vxg sql`), SELECT, compound WHERE, ORDER BY Top-K & GROUP BY without full decode
./target/release/vxg sql telemetry.vxg "SELECT service, COUNT(*), SUM(bytes) WHERE status = 500 GROUP BY service ORDER BY SUM(bytes) DESC LIMIT 10" -t auto
./target/release/vxg query telemetry.vxg --select ts,service,latency_ms --where 'status==500' --order-by latency_ms --desc --limit 10 -t auto

# Zero-reencode VXG -> VXG column splicing & Unix stdin/stdout streaming
./target/release/vxg project telemetry.vxg slim.vxg --cols service,status,bytes -t auto -f
cat access.log | ./target/release/vxg c - - --stream -t auto -q | ./target/release/vxg d - - --stream -t auto -q > access.restored.log
```
