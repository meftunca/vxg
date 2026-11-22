---
layout: home

hero:
  name: "VXG v0.7.0"
  text: "Unified Compressed-Domain Engine"
  tagline: "Pure-Rust, zero-dependency compressed-domain computing across Data & SQL (vxg), Stateless Snapshot Redis (vxg-kv), E8 Lattice LLM/VLM Inference (vxt-turbo), and HTTP-Range Wavelet Imaging (vimg)."
  image:
    src: /logos/vxg.svg
    alt: VXG Logo
  actions:
    - theme: brand
      text: VXG SQL Playground →
      link: /playground
    - theme: brand
      text: VIMG Image Playground →
      link: /vimg-playground
    - theme: alt
      text: Stateless Redis (VXG-KV)
      link: /KV
    - theme: alt
      text: Image Codec (.vimg)
      link: /VIMG
    - theme: alt
      text: AI Tensor (.vxt)
      link: /TENSOR
    - theme: alt
      text: Get Started
      link: /guide/getting-started

features:
  - icon:
      src: /logos/vxg.svg
    title: "VXG Core v0.7.0 — Data, SQL & HTTP/1.1 Range"
    details: "Keep raw JSON/CSV verbatim (0.1171 6-dataset geomean, ahead of Zstd+Dict-9), run SQL (`EXPLAIN`, `DISTINCT`, `GROUP BY`, `HAVING`) or Arrow/DuckDB/Polars `read_vxg()` directly on local or remote HTTP Range blocks (15/15 vs DuckDB 2.0), and recover files byte-for-byte."
  - icon:
      src: /logos/vxg-kv.svg
    title: "VXG-KV v0.3.0 — Stateless Snapshot Redis (RESP2/RESP3)"
    details: "Zero-dep kqueue/epoll reactor beating Redis 8.10.3 11/11 (1.67M–1.85M ops/s, 1.93x less CPU) with 20x smaller snapshots, 7x less cold-pod RAM, AUTH/HELLO 3, JSON/Geo, BLPOP/BRPOP, XGROUP, O(1) DBSIZE, --wal-shards N, Time-Travel, and 26.8x faster VXG.SQL."
  - icon:
      src: /logos/vxt-turbo.svg
    title: "VXT-Turbo v0.6.0 — E8 Lattice AI Model Codec & GPU LLM"
    details: "E8/E8-HQ beats GGUF Q6_K (-0.43 GB, -32% TR KLD) and Q8_0 (-0.89 GB, -56% TR KLD) on Qwen3.5-9B; 963 tok/s Q3 / 796 tok/s Q6 on RTX 5090 CUDA Resident, 298 tok/s on Apple Metal, MoE + Vision 3D M-RoPE, and OpenAI SSE server."
  - icon:
      src: /logos/vimg.svg
    title: "VIMG v0.4.0 — HTTP-Range Wavelet Image & Animation Codec"
    details: "Zero-dep pure-Rust 8/10/12/16-bit HDR YCoCg-R + CDF 5/3 & 9/7 wavelet pyramid + VIMA1 animations. 1.00 ms UI Palette+RLE, zero-decode crop/downscale, ProgressiveDecoder, WebP Edge Proxy, and 277 KiB +simd128 WASM Studio."
---

## Why VXG?

Traditional byte-stream LZ compressors (`LZ4`, `Zstd`) scan inputs sequentially through a sliding window. On structured logs, telemetry tables, and columnar JSON streams, they repeatedly rediscover field names, delimiters, timestamps, and numeric formatting on every row.

**VXG (`VXG1` v4/v5)** splits each independent block (`64 KiB` default, or `\n`-snapped / row-bounded `--record-aligned`) into a **Vectorized Execution Grammar (`GRAMMAR_PACKET`, including exception-patched `PFOR-Delta` and multi-level `depth <= 4` nested JSON)**, **Self-Contained Frame Dictionary (`<= 256 KiB`) + `--level 1..9` 5-Stream Canonical Huffman / 2-Way `FSE` (`BLOCK_DICT_LZ`)**, **Strided Decorrelation Planes (`STRIDE_PACKET`)**, or **Fast-LZ (`FAST_LZ_PACKET`)**, selected strictly by exact byte budget — backed by 18 frozen `v1`–`v5` wire vectors, $O(1)$ per-block row-count indexing (`FLAG_ROW_INDEX`), and corrupt-frame partial recovery (`vxg recover`). For public real-world corpora (`NYC TLC Taxi`, `datasets/covid-19`, `GitHub Archive`, `Kodak` true-color photos, non-repeating `src/`), `Zstd + Trained Dictionary [32..256 KiB]`, whole-file `zstd -19 --long`, and **DuckDB 2.0 + Parquet (`1T` & `8T`, `200k` to `40M` rows)** head-to-head benchmarks, see **[Competitor Sweep & DuckDB](/CHIMERA_BENCHMARKS)**.

<CodecBenchmarkChart compact />

| Metric (`--suite v4` Synthetic Qualification Geomean, `16 MiB`) | **VXG v0.5** | **LZ4 (`1.10.0`)** | **Zstd (`-1`, `1.5.7`)** | **Trade-Off & Provenance Notes** |
| :--- | :---: | :---: | :---: | :--- |
| **8-Dataset Structured/Text Ratio** *(excl. procedural BMP)* | **`0.0259` (`38.6:1`)** | `0.2107` (`4.7:1`) | `0.0959` (`10.4:1`) | **Synthetic suite:** `source-and-docs.txt` (`0.0036`) repeats a `~190 KiB` seed; non-repeating `src/` is `0.1917` ([real corpora](/CHIMERA_BENCHMARKS)) |
| **10-Dataset Ratio** *(incl. 2 procedural UI BMPs)* | **`0.0162` (`61.7:1`)** | `0.1286` (`7.8:1`) | `0.0615` (`16.3:1`) | **`7.94x` smaller than LZ4, `3.80x` smaller than Zstd-1** (procedural UI BMPs; real camera `kodak.bmp` is `0.7985`) |
| **1-Thread Encode / Decode (`GB/s`, 10-set)** | `2.22` / **`10.53 GB/s`** | **`2.27`** / `6.94 GB/s` | `1.55 / 2.66 GB/s` | **Beats LZ4 on 8-set 1T Enc & Dec (`1.58 / 9.69` vs `1.51 / 5.79 GB/s`); `1.52x` faster 10-set decode** |
| **8-Thread Encode / Decode (`GB/s`, 10-set)** | `8.51` / **`54.97 GB/s`** | **`10.96`** / `32.11 GB/s` | `7.73 / 14.07 GB/s` | **LZ4 encodes `1.29x` faster; VXG decodes `1.71x` faster** (`94.5 GB/s` on periodic `source-and-docs.txt`) |
| **Decode Energy (`J/GB`, 1T)** *(lower is better)* | **`0.533 J/GB`** | `0.634 J/GB` | `1.641 J/GB` | **`1.19x` less decode energy than LZ4, `3.08x` less than Zstd-1** |
| **Green Codec Index (`GCI`, 1T / 8T)** | **`275.5 / 1,343.8`** | `32.4 / 157.9` | `15.5 / 94.6` | **`8.51x` higher `GCI` than LZ4, `14.2x`–`17.8x` higher than Zstd-1** |

### Historical `v0.4.1` Cross-Platform GitHub Actions CI Summary (`Format v4` Pre-`v5`, `1T`, `16 MiB` Geomean)

| CI Runner (`OS / Arch / SIMD`) | **VXG v4 Ratio** | **VXG v4 Encode / Decode** | **LZ4 Encode / Decode** | **Zstd-1 Encode / Decode** | **VXG `VPI` (`vs LZ4` / `Zstd-1`)** | **VXG `BCI` (`vs LZ4` / `Zstd-1`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **macOS (`macos-14` / `aarch64` NEON)** | **`0.0249`** | **`1.554 / 5.896 GB/s`** | `1.307 / 5.205 GB/s` | `0.920 / 1.990 GB/s` | **`121.4` (`5.91x` / `5.45x`)** | **`7.17` (`3.76` / `3.11`)** |
| **Linux (`ubuntu-latest` / `x86_64` AVX2)** | **`0.0249`** | `1.984` / **`6.822 GB/s`** | **`2.038`** / `5.994 GB/s` | `1.308 / 2.699 GB/s` | **`147.5` (`5.36x` / `4.76x`)** | **`8.16` (`4.58` / `3.88`)** |
| **Windows (`windows-latest` / `x86_64` AVX2)** | **`0.0249`** | **`2.295 / 8.362 GB/s`** | `1.520 / 7.519 GB/s` | `1.110 / 3.080 GB/s` | **`175.9` (`6.60x` / `5.76x`)** | **`9.17` (`4.48` / `3.84`)** |

### Compressed-Domain SQL Queries, `Top-K`, Analytics & Column Splicing

<CompressedOpsChart />

### Compressed-Domain AI Model Weights & Apple Metal / NVIDIA CUDA LLM Inference (`vxg-tensor` / `.vxt`)

`crates/vxg-tensor` brings VXG's compressed-domain philosophy to LLM/VLM/MoE weights (`GGUF` & `SafeTensors` -> `.vxt` `VXT1` via zero-OOM streaming `--preset q3|q4|q6|q8|lossless`), beating `GGUF Q6_K` and `Q8_0` at smaller sizes on `Qwen3.5-9B` and executing directly on **Apple Metal (`298 tok/s` `0.8B`, `176 tok/s` `2B`, `47.2 tok/s` `9B + Vision`)** and **NVIDIA CUDA VRAM-Resident (`962.86 tok/s` `Q3` / `796.28 tok/s` `Q6` on RTX 5090)** ([Full Specification & Benchmarks →](/TENSOR)):

<TensorBenchmarkChart compact />

---

## Quick Start in 30 Seconds

### 1. Rust Library (`Cargo.toml`)

```rust
use vxg::{DecodeOptions, Options};

fn main() -> vxg::Result<()> {
    let raw = b"ts=2026-06-01T00:00:00Z level=INFO service=auth latency_us=120\n".repeat(1000);
    let opts = Options::default(); // VXG v4/v5, 64 KiB blocks, level 3, CRC32C enabled
    let dec = DecodeOptions::default();

    // 1. One-shot compress & zero-allocation caller-buffer decompress
    let compressed = vxg::compress(&raw, &opts)?;
    let mut restored = vec![0u8; vxg::frame_raw_size(&compressed)?];
    vxg::decompress_into(&compressed, &mut restored, &dec)?;
    assert_eq!(restored, raw);

    // 2. O(1) random-access single-block extraction
    let block0_len = vxg::frame_block_raw_size(&compressed, 0)?;
    let mut block0 = vec![0u8; block0_len];
    vxg::extract_block_into(&compressed, 0, &mut block0, &dec)?;
    assert_eq!(&block0[..], &raw[..block0_len]);

    Ok(())
}
```

### 2. Command-Line Interface (`vxg`)

```sh
# Build from source
cargo build --release --bin vxg

# Compress using all CPU cores (-t auto) with newline-aligned blocks (--record-aligned)
./target/release/vxg compress telemetry.ndjson telemetry.vxg --record-aligned --level 3 -t auto -f

# Inspect frame header, compression ratio, and block-type summary
./target/release/vxg info telemetry.vxg --summary

# Compressed-domain SQL (`vxg sql`), Arrow IPC export (`vxg arrow`), and corrupt-frame block recovery
./target/release/vxg sql telemetry.vxg "SELECT service, COUNT(*), SUM(bytes) WHERE status = 500 GROUP BY service ORDER BY SUM(bytes) DESC LIMIT 10" -t auto
./target/release/vxg arrow telemetry.vxg telemetry.arrows --cols ts,service,status,bytes -t auto -f
./target/release/vxg recover telemetry.vxg salvaged.ndjson --report -f

# Zero-reencode VXG -> VXG column splicing & Unix stdin/stdout streaming
./target/release/vxg project telemetry.vxg slim.vxg --cols service,status,bytes -t auto -f
cat access.log | ./target/release/vxg c - - --stream -t auto -q | ./target/release/vxg d - - --stream -t auto -q > access.restored.log
```
