# VXG 0.4.7 — Unified Adaptive Multi-Format Lossless Block Codec & Compressed-Domain Engine (Format v4)

[![CI — Source Validation](https://github.com/meftunca/vxg/actions/workflows/ci.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/ci.yml)
[![CI — Cross-Platform Benchmarks](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml)
[![Release v0.4.7](https://img.shields.io/github/v/release/meftunca/vxg)](https://github.com/meftunca/vxg/releases/tag/v0.4.7)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Store raw NDJSON/CSV as-is (`14×–21×` faster ingest than Parquet ETL), `1.18×–1.32×` smaller `64 KiB` random-access blocks than `Zstd + Trained Dictionary` on flat tabular records at `LZ4`-class speed, run SQL (`SELECT`, `WHERE`, `ORDER BY Top-K`, `GROUP BY`, `SUM`, zero-reencode splicing) directly on compressed files, and recover the original file byte-for-byte.**

> **Scope on Real-World Data (`Section 1` & `Section 2`):**
> - **Where VXG wins:** Flat tabular records (`nyc-taxi.ndjson`, `nyc-taxi.csv` — `1.18×–1.32×` smaller than `Zstd + Trained Dict L9`, faster than `LZ4` on NDJSON, and **zero-ETL 100% byte-exact original file preservation** vs. irreversible Parquet schema conversion).
> - **Where competitors win:** Deeply nested JSON (`gharchive`), real free-text/HTTP logs (`apache-access.clf`, `macos-install.log`), and literary prose (`pg100-shakespeare.txt`), where `Zstd + Trained Dict L9` leads `64 KiB` block ratio (`0.1205` 6-dataset geomean vs `Zstd-9` `0.1419` vs `VXG` `0.1484`) and `LZ4` leads raw decode speed (`8.46` vs `7.86 GB/s`).

VXG `v4` inspects each independent `64 KiB` block at runtime and routes it by exact byte budget to `BLOCK_COLUMN` (grammar + typed column models), `BLOCK_LZ` (2D stride/pixel planes or 4-stream canonical Huffman + `Fast-LZ`), or `BLOCK_RAW`. Every block is self-contained, enabling **$O(1)$ random-access block extraction** and **multi-core scaling** without cross-block decompression dependencies.

---

## Key Capabilities

| Area | Highlights |
| :--- | :--- |
| **Format v4 Wire Layout** | Compact 8-byte per-block directory (`payload_size \| kind << 30`, `crc32c`); full backward read compatibility with `v1`–`v3` frames |
| **Structured Column Engine** | Automatic schema & delimiter discovery for **NDJSON, Pretty-Printed JSON, CSV, TSV, PSV, CLF/Bracketed Logs, and Logfmt**; 11 specialized models including `Dictionary`, `DecimalDelta`, `DecimalSuffix`, `Timestamp` (SWAR+SIMD), `HexSuffix` (quoted & unquoted SIMD hex), `EpochCounter`, `SegmentDictionary`, `FrontCodedDict`, and `Pattern` (`MODEL_PATTERN` SWAR + `Copy16`) |
| **Compressed-Domain SQL & Zero-ETL Archive** | Store raw NDJSON/CSV verbatim (`14×–21×` faster ingest than Parquet ETL), execute compound `WHERE` (`&&` / `\|\|` / `!`), `SELECT`, `ORDER BY` / `Top-K`, `SUM`/`MIN`/`MAX`/`AVG` + `GROUP BY`, `UPDATE`/`MAP`, and **zero-reencode `VXG -> VXG` column splicing** (`0.21–3.84 ms` on `200k–300k` in-cache rows — `22×–540×` faster than `DuckDB` on raw/`.zst`), and recover the original file byte-for-byte |
| **Image & Stride Engine** | 24-bpp / 32-bpp BMP and multi-stride (`1, 2, 3, 4, 8`) byte-plane transpose + `Subtract-Green` (`R-G`, `B-G`) color decorrelation + 2-level 64-bit hierarchical zero-mask & per-plane `Fast-LZ` |
| **Prose & Source Code** | Const-generic `Fast-LZ` matcher paired with a 4-stream interleaved canonical Huffman coder |
| **Cross-Platform SIMD & SWAR** | Runtime-dispatched `NEON` (`aarch64`) and `SSE2` / `AVX2` / `AVX-512` (`x86_64`) fast paths, including 32-byte branchless string scanning, 64-byte `FastPiece` unaligned `u128` row assembly, and SIMD prefix-sum plane interleaving |
| **Integrity & Safety** | Hardware-accelerated CRC32C (`ARMv8 CRC` / `SSE4.2`); strict output-size bounds and deterministic rejection of truncated or malformed inputs |
| **Zero-Copy `mmap` & Streaming Engine** | Zero-dependency cross-platform `MappedFile` / `MappedFileMut`, zero-output-alloc parallel `verify_frame` (`12+ GB/s`, `~64 KiB` L2 scratch/thread), and bounded-memory (`O(single_frame)`) multi-GB streaming across `query`, `agg`, `project`, `filter-frame`, `update`, and `map` (`FramePeeler` split-line stitching), plus `staticlib` / `cdylib` C ABI ([`include/vxg.h`](include/vxg.h)) |

---

## 1. Real-World Public & System Corpora (`64 KiB` Blocks vs. `Zstd + Trained Dict` & `zstd -19 --long`)

Measured via `vxg-bench` (`Apple Silicon aarch64 NEON`, `64 KiB` independent blocks, `liblz4 1.10.0`, `libzstd 1.5.7`, checksums off; fetched reproducibly via [`scripts/fetch_real_corpora.py`](scripts/fetch_real_corpora.py)).
- **`Zstd + Dict` (`zstd-dict-l3` / `zstd-dict-l9`):** Trains up to a `32 KiB` dictionary per dataset (`ZDICT_trainFromBuffer`) and charges `dict_bytes` in compressed size; `Enc`/`Dec` measure steady-state `ZSTD_CDict`/`ZSTD_DDict` throughput.
- **Whole-file `zstd -19 --long` (`128 MiB` window, single stream, `random access = NO`):** Reference column quantifying the exact compression-ratio cost of `64 KiB` block independence.
- **Reading the tables:** Bold cells mark **only the actual per-row winner** on each metric; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` compression) are excluded from speed bolding. Full `1T` & `8T` tables and parameter sweeps: **[`docs/CHIMERA_BENCHMARKS.md`](docs/CHIMERA_BENCHMARKS.md)**, [`reports/real-public-1t.md`](reports/real-public-1t.md), and [`reports/real-public-8t.md`](reports/real-public-8t.md).

| Real Public / OS Corpus (`8T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc / Dec (`GB/s`) | LZ4 (`a1`) Enc / Dec | Zstd-1 Enc / Dec | Zstd+D3 Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** *(NYC TLC, 200k trips)* | 52.3 MB | **0.0592** | 0.1899 | 0.0823 | 0.0783 | 0.0554 | **10.20 / 34.69** | 6.54 / 19.97 | 6.72 / 10.13 | 2.93 / 6.92 |
| **`nyc-taxi-2024-01.csv`** *(NYC TLC, 300k trips)* | 23.0 MB | **0.1807** | 0.4241 | 0.2194 | 0.2132 | 0.1617 | 1.34 / 3.10 | **1.59 / 4.63** | 0.90 / 1.72 | 0.48 / 1.77 |
| **`gharchive-2025-01-15.ndjson`** *(12k nested GitHub API)* | 50.3 MB | 0.1985 | 0.2256 | 0.1414 | **0.1150** | 0.0633 | 1.97 / 5.46 | **2.45 / 9.41** | 1.87 / 4.31 | 1.22 / 3.37 |
| **`apache-access.clf`** *(Elastic real HTTP CLF)* | 2.4 MB | 0.1198 | 0.1730 | 0.1062 | **0.0878** | 0.0535 | 0.62 / 1.62 | 0.94 / 1.46 | **1.05 / 1.97** | 0.51 / 1.23 |
| **`macos-install.log`** *(OS free-text multi-line log)* | 34.2 MB | 0.0948 | 0.1249 | 0.0770 | **0.0484** | 0.0278 | **8.80** / 22.43 | 8.59 / **25.44** | 6.14 / 11.23 | 2.18 / 8.15 |
| **`pg100-shakespeare.txt`** *(Gutenberg English prose)* | 5.6 MB | 0.4424 | 0.6223 | 0.3918 | **0.3745** | 0.3000 | **2.48** / 11.10 | 2.12 / **11.33** | 1.91 / 5.22 | 0.74 / 3.56 |
| **`kodak-truecolor-16.bmp`** *(16 real camera photos)* | 18.9 MB | 0.9256 | 0.8732 | 0.7494 | **0.7452** | 0.5656 | 3.58 / 19.61 | **1.62 / 6.24** | 1.36 / 2.48 | 0.41 / 3.29 |
| **Text & Structured Geomean (6)** | — | 0.1484 | 0.2500 | 0.1419 | **0.1205** | 0.0795 | 2.67 / 7.86 | **2.76 / 8.46** | 2.28 / 4.53 | 1.06 / 3.37 |

**Summary of Real-World Trade-Offs:**
- **Flat tabular records (`nyc-taxi.ndjson`, `nyc-taxi.csv`):** `VXG v4` achieves the smallest `64 KiB` blocks (`1.18×–1.32×` smaller than `Zstd + Trained Dict L9`, `2.35×–3.21×` smaller than `LZ4`, and within `6.9%–11.8%` of non-random-access whole-file `zstd -19 --long`). On `nyc-taxi.ndjson`, `VXG` leads `LZ4 (a1)` on both `1T` (`1.65 / 4.99` vs `1.29 / 4.36 GB/s`) and `8T` (`10.20 / 34.69` vs `6.54 / 19.97 GB/s`; `LZ4 a4` peaks at `41.55 GB/s` decode), while on `nyc-taxi.csv`, `LZ4` decodes `1.49×–1.97×` faster.
- **Why real logs & nested JSON (`apache-access.clf`, `macos-install.log`, `gharchive`, `shakespeare`) lose to `Zstd + Trained Dict`:** While synthetic `data.clf` compresses to `0.0980`, real `apache-access.clf` (`0.1198` vs `Zstd+Dict-9` `0.0878`) and `gharchive` (`0.1985` vs `0.1150`) contain high-entropy free-text URL paths, User-Agents, and nested JSON bodies that fall back to `Fast-LZ + 4-stream Huffman`, which has lower density than `Zstd`'s `FSE + optimal parsing + shared 32 KiB dictionary`. Adding a hybrid `BLOCK_ZSTD_DICT` fallback block type is scheduled for `v0.5` (`Section 4`).
- **Real camera photography (`kodak-truecolor-16.bmp`):** Optical sensor noise limits lossless compression across general-purpose codecs to `0.5656` (`zstd -19 --long`) – `0.7452` (`Zstd+Dict-9`) – `0.9256` (`VXG`, unbolded in throughput because `0.93` is near-raw storage), unlike procedural UI framebuffers (`Section 3`).

---

## 2. In-Cache Analytical Queries: `VXG` vs. `DuckDB 2.0.0` (`Raw`, `.zst`, `Parquet-Snappy`, `Parquet-Zstd`)

Measured standalone with zero background CPU contention (`libvxg.dylib` vs `duckdb 2.0.0`, warm OS cache, median of 5–7 runs, **100% result equality verified** via [`scripts/bench_vs_duckdb.py`](scripts/bench_vs_duckdb.py); full `1T` & `8T` tables in [`docs/CHIMERA_BENCHMARKS.md#duckdb`](docs/CHIMERA_BENCHMARKS.md#duckdb) and [`reports/duckdb-vs-vxg.json`](reports/duckdb-vs-vxg.json)).
- **Python + SQL parse/plan overhead calibration (`SELECT 1`):** `0.06 ms` ad-hoc / **`0.04 ms` prepared** at `1T`, and `0.09–0.12 ms` ad-hoc / **`0.06–0.07 ms` prepared** at `8T`. Parquet columns below report **`PREPARE / EXECUTE` (`prep`)** latencies so SQL parse/plan overhead is excluded.
- **Scale & cache boundary:** All three corpora (`200k–300k` rows, `23–56.5 MB` raw, `3–10 MB` compressed) fit in CPU/RAM cache. Out-of-cache `10M–40M+`-row multi-GB analytical scans (e.g., full-year NYC TLC) where DuckDB's vectorized engine excels are **not** claimed here and are scheduled for `v0.5`.

| Corpus & Storage / Query (`8T`, `SELECT 1` prep = `0.06–0.07 ms`) | **VXG `.vxg`** | DuckDB Raw | DuckDB `.zst` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson` (`52.3 MB` real, `200k` rows)** | **`3.10 MB` (`4.5 ms` enc, lossless=YES)** | `52.33 MB` (`0 ms`, YES) | `4.92 MB` (`32.9 ms`, YES) | `3.94 MB` (`84.3 ms` ETL — `18.6×` slower, **NO**) | **`2.96 MB`** (`93.6 ms` ETL — `20.7×` slower, **NO**) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | `1.38 ms` (`1T`: `3.84 ms`) | `57.23 ms` | `125.12 ms` | **`1.36 ms`** (`1T`: **`1.69 ms`**) | `1.64 ms` (`1T`: `2.36 ms`) |
| ↳ `SUM/MIN/MAX(total_cents)` *(all 200k rows)* | **`0.74 ms` (`270.9 M rows/s`)** | `54.15 ms` | `133.08 ms` | `1.22 ms` (`1T`: **`1.23 ms`**) | `1.38 ms` (`144.9 M rows/s`) |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `1.47 ms` (`136.5 M rows/s`) | `59.56 ms` | `124.31 ms` | **`1.08 ms`** (`185.2 M rows/s`) | `1.25 ms` (`160.0 M rows/s`) |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | `1.22 ms` (`1T`: `2.97 ms`) | `56.11 ms` | `119.83 ms` | **`1.02 ms`** (`1T`: **`1.25 ms`**) | `1.21 ms` (`1T`: `1.58 ms`) |
| ↳ `ORDER BY total_cents DESC LIMIT 10` | **`0.88 ms` (`228.0 M rows/s`)** | `58.12 ms` | `127.75 ms` | `2.26 ms` (`1T`: `3.17 ms`) | `3.48 ms` (`57.5 M rows/s`) |
| ↳ `SELECT 4 cols WHERE tip_cents >= 2500` | **`1.09 ms` (`184.1 M rows/s`)** | `61.24 ms` | `138.45 ms` | `3.19 ms` (`2.19 ms` ad-hoc) | `5.50 ms` (`3.83 ms` ad-hoc) |
| **`nyc-taxi-2024-01.csv` (`23.0 MB` real, `300k` rows)** | **`4.16 MB` (`6.6 ms` enc, lossless=YES)** | `23.00 MB` (`0 ms`, YES) | `5.01 MB` (`45.0 ms`, YES) | `5.86 MB` (`89.5 ms` ETL — `13.6×` slower, **NO**) | `4.39 MB` (`97.6 ms` ETL — `14.8×` slower, **NO**) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | **`0.38 ms` (`798.8 M rows/s`)** | `51.66 ms` | `120.59 ms` | `1.24 ms` (`1T`: **`1.84 ms`**) | `1.76 ms` (`170.5 M rows/s`) |
| ↳ `SUM/MIN/MAX(total_cents)` *(all 300k rows)* | **`0.32 ms` (`925.8 M rows/s`)** | `54.10 ms` | `117.85 ms` | `1.00 ms` (`300.0 M rows/s`) | `1.23 ms` (`243.9 M rows/s`) |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | **`0.21 ms` (`1453.7 M rows/s`)** | `51.22 ms` | `113.79 ms` | `1.11 ms` (`270.3 M rows/s`) | `1.24 ms` (`241.9 M rows/s`) |
| **`ndjson_varied.ndjson` (`56.5 MB` synthetic, `300k` rows)** | **`5.17 MB` (`8.7 ms` enc, lossless=YES)** | `56.50 MB` (`0 ms`, YES) | `9.86 MB` (`43.7 ms`, YES) | `9.70 MB` (`115.7 ms` ETL — `13.3×` slower, **NO**) | `5.56 MB` (`120.6 ms` ETL — `13.9×` slower, **NO**) |
| ↳ `COUNT(*) WHERE fraud & status=500 & lat>=150` | **`0.92 ms`** (`1T`: `3.51 ms`) | `61.84 ms` | `153.19 ms` | `0.94 ms` (`1T`: **`1.30 ms`**) | `1.01 ms` (`1T`: `1.60 ms`) |
| ↳ `GROUP BY service: COUNT(*), SUM(bytes)` | **`0.95 ms` (`316.4 M rows/s`)** | `63.86 ms` | `156.34 ms` | `1.74 ms` (`172.4 M rows/s`) | `2.27 ms` (`132.2 M rows/s`) |
| ↳ `ORDER BY latency_ms DESC LIMIT 10` | **`0.81 ms` (`372.4 M rows/s`)** | `63.98 ms` | `157.99 ms` | `2.53 ms` (`118.6 M rows/s`) | `3.49 ms` (`86.0 M rows/s`) |

---

## 3. Synthetic & Procedural Qualification Suites (`--suite v4` & Multi-Template Generators)

> **Important:** All datasets below (`--suite v4`, `scripts/gen_varied_ndjson.py`, `scripts/gen_format_bench.py`) are **synthetic / procedurally generated** qualification workloads. The two `.bmp` files in `--suite v4` are procedural flat-shaded UI/gradient framebuffers (`208:1`–`833:1`) and are separated from the 8 text/structured datasets below (`reports/synthetic-v4-{1t,8t}.md`, `reports/synthetic-multitpl-{1t,8t}.md`).

| Synthetic Suite (`64 KiB` blocks) | **VXG v4 Ratio** | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | `1T` VXG vs LZ4 (`Enc / Dec GB/s`) | `8T` VXG vs `LZ4 Sweep` (`Enc / Dec GB/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v4` Text & Structured Geomean (8 sets, `16 MiB`)** | **0.0451** | 0.2103 | 0.0857 | 0.0845 | 0.0080 | `1.27` / **`6.41`** vs **`1.38`** / `5.31` | `3.16 / 9.89` vs **`3.35 / 11.77`** |
| **`v4` Procedural UI BMP Geomean (2 sets, `16 MiB`)** | **0.0024** | 0.0178 | 0.0103 | 0.0111 | 0.0010 | `6.74 / 10.20` vs **`8.15 / 11.66`** | **`42.36 / 57.96`** vs `29.03 / 47.34` |
| **Synthetic Multi-Template (`ndjson_varied`, `56.5 MB`)** | **0.0914** | 0.2898 | 0.1480 | 0.1452 | 0.1198 | `0.93` / **`5.51`** vs **`1.05`** / `4.29` | `3.08 / 16.38` vs **`7.73 / 31.81`** |
| **Synthetic Delimited (`data.csv` / `data.clf`, `31 MB`)** | **0.1796 / 0.0980** | 0.4458 / 0.2843 | 0.2651 / 0.1504 | 0.2606 / 0.1454 | 0.2257 / 0.1225 | `0.65 / 2.14` · `0.59 / 2.45` (`LZ4` faster) | **`1.95`**`/5.13` · `1.85/5.62` (`LZ4` faster dec) |

---

## Cross-Platform CI Benchmark Results — VXG `v0.4.1` (`--suite v4` Synthetic Suite on `macos-14`, `ubuntu-latest`, `windows-latest`)

Executed automatically on GitHub Actions via [`.github/workflows/benchmarks.yml`](.github/workflows/benchmarks.yml) against native C **`liblz4`** (`lz4-c-a1`) and **`libzstd`** (`zstd-c-l1`) on `65,536`-byte independent blocks (`16 MiB` per synthetic `--suite v4` dataset, single thread, checksums off). Full artifacts: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`reports/ci-v0.4.1/`](reports/ci-v0.4.1/).

### 1. macOS (`macos-14` / Apple Silicon `aarch64` NEON)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.296 GB/s** | 0.982 GB/s | 0.730 GB/s | **5.954 GB/s** | 5.514 GB/s | 1.739 GB/s | **68.2** (`5.84x` / `5.09x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.187 GB/s** | 0.952 GB/s | 0.751 GB/s | **7.044 GB/s** | 4.026 GB/s | 1.715 GB/s | **86.6** (`9.83x` / `5.65x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.614 GB/s** | 1.205 GB/s | 0.770 GB/s | **8.613 GB/s** | 5.748 GB/s | 2.104 GB/s | **105.6** (`8.05x` / `5.77x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.618 GB/s** | 1.437 GB/s | 0.917 GB/s | **9.242 GB/s** | 6.288 GB/s | 2.811 GB/s | **170.6** (`8.50x` / `5.77x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.175 GB/s** | 0.873 GB/s | 0.616 GB/s | **4.980 GB/s** | 3.274 GB/s | 1.348 GB/s | **54.4** (`6.86x` / `6.22x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.281 GB/s** | 1.260 GB/s | 0.770 GB/s | **5.830 GB/s** | 4.055 GB/s | 1.597 GB/s | **110.1** (`7.52x` / `7.73x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.940 GB/s | **0.985 GB/s** | 0.671 GB/s | **3.503 GB/s** | 3.411 GB/s | 1.467 GB/s | **37.8** (`4.26x` / `4.56x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2702** | 0.3778 | 0.2714 | 0.374 GB/s | **0.496 GB/s** | 0.390 GB/s | 1.696 GB/s | **3.200 GB/s** | 1.084 GB/s | 2.9 (`0.88x` / `1.23x`) |
| **`ui-render-32bpp.bmp`** *(procedural)* | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.423 GB/s** | 3.846 GB/s | 2.456 GB/s | 9.170 GB/s | **14.402 GB/s** | 2.719 GB/s | **1603.8** (`3.02x` / `5.66x`) |
| **`rgb-frame-24bpp.bmp`** *(procedural)* | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **6.025 GB/s** | 4.357 GB/s | 3.669 GB/s | **9.611 GB/s** | 8.715 GB/s | 5.913 GB/s | **6131.5** (`22.59x` / `15.71x`) |
| **All-10 Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.554 GB/s** | 1.307 GB/s | 0.920 GB/s | **5.896 GB/s** | 5.205 GB/s | 1.990 GB/s | **121.4** (`5.91x` / `5.45x`) |

### 2. Linux (`ubuntu-latest` / `x86_64` AVX2+SSE2)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.706 GB/s** | 1.437 GB/s | 0.737 GB/s | **7.513 GB/s** | 7.170 GB/s | 2.395 GB/s | **87.9** (`5.46x` / `5.56x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.460 GB/s** | 1.281 GB/s | 1.034 GB/s | **6.709 GB/s** | 4.957 GB/s | 3.155 GB/s | **93.7** (`8.27x` / `3.84x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.885 GB/s** | 1.871 GB/s | 1.219 GB/s | **8.434 GB/s** | 7.626 GB/s | 3.201 GB/s | **113.0** (`6.00x` / `3.98x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 2.474 GB/s | **2.484 GB/s** | 1.650 GB/s | **10.906 GB/s** | 9.651 GB/s | 4.085 GB/s | **229.3** (`7.01x` / `4.79x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.377 GB/s** | 1.286 GB/s | 0.856 GB/s | **5.483 GB/s** | 4.587 GB/s | 2.113 GB/s | **61.8** (`5.42x` / `4.79x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 1.824 GB/s | **1.836 GB/s** | 1.145 GB/s | **7.681 GB/s** | 5.197 GB/s | 2.461 GB/s | **150.8** (`7.54x` / `7.00x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 1.032 GB/s | **1.300 GB/s** | 0.801 GB/s | 3.772 GB/s | **3.918 GB/s** | 1.792 GB/s | **41.1** (`3.76x` / `4.10x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2712** | 0.3785 | 0.2722 | 0.513 GB/s | **0.661 GB/s** | 0.416 GB/s | 2.205 GB/s | **3.970 GB/s** | 1.772 GB/s | 3.9 (`0.92x` / `1.24x`) |
| **`ui-render-32bpp.bmp`** *(procedural)* | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 8.370 GB/s | **9.805 GB/s** | 5.723 GB/s | 14.818 GB/s | **21.433 GB/s** | 4.797 GB/s | **2327.3** (`2.25x` / `4.05x`) |
| **`rgb-frame-24bpp.bmp`** *(procedural)* | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **7.311 GB/s** | 7.260 GB/s | 5.124 GB/s | **9.078 GB/s** | 2.881 GB/s | 2.622 GB/s | **6564.0** (`32.59x` / `21.37x`) |
| **All-10 Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | 1.984 GB/s | **2.038 GB/s** | 1.308 GB/s | **6.822 GB/s** | 5.994 GB/s | 2.699 GB/s | **147.5** (`5.36x` / `4.76x`) |

### 3. Windows (`windows-latest` / `x86_64` AVX2+SSE2)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **2.010 GB/s** | 1.364 GB/s | 0.872 GB/s | **8.915 GB/s** | 8.835 GB/s | 2.481 GB/s | **103.9** (`5.97x` / `5.94x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.893 GB/s** | 1.115 GB/s | 0.945 GB/s | **8.583 GB/s** | 6.385 GB/s | 2.917 GB/s | **120.7** (`10.06x` / `5.38x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.758 GB/s** | 1.312 GB/s | 1.045 GB/s | **9.922 GB/s** | 8.689 GB/s | 2.667 GB/s | **118.3** (`7.03x` / `4.93x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.748 GB/s** | 1.776 GB/s | 1.261 GB/s | **14.409 GB/s** | 11.171 GB/s | 3.658 GB/s | **277.7** (`9.33x` / `7.01x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.519 GB/s** | 1.132 GB/s | 0.760 GB/s | **6.405 GB/s** | 4.697 GB/s | 1.946 GB/s | **70.1** (`6.48x` / `6.01x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **2.018 GB/s** | 1.513 GB/s | 0.995 GB/s | **9.347 GB/s** | 4.840 GB/s | 2.339 GB/s | **174.9** (`9.98x` / `8.93x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 1.176 GB/s | **1.178 GB/s** | 0.733 GB/s | **4.472 GB/s** | 3.965 GB/s | 1.670 GB/s | **47.8** (`4.56x` / `5.16x`) |
| **`source-and-docs.txt`** | 16 MiB | 0.2678 | 0.3721 | **0.2672** | 0.576 GB/s | **0.735 GB/s** | 0.522 GB/s | 2.962 GB/s | **5.253 GB/s** | 1.805 GB/s | 4.9 (`0.92x` / `1.34x`) |
| **`ui-render-32bpp.bmp`** *(procedural)* | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **10.452 GB/s** | 3.523 GB/s | 3.133 GB/s | 17.003 GB/s | **47.484 GB/s** | 18.978 GB/s | **2785.9** (`3.02x` / `3.29x`) |
| **`rgb-frame-24bpp.bmp`** *(procedural)* | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **10.137 GB/s** | 3.554 GB/s | 2.892 GB/s | **11.329 GB/s** | 4.692 GB/s | 4.180 GB/s | **8634.5** (`48.02x` / `29.64x`) |
| **All-10 Geometric Mean** | — | **0.0249** | 0.1268 | 0.0606 | **2.295 GB/s** | 1.520 GB/s | 1.110 GB/s | **8.362 GB/s** | 7.519 GB/s | 3.080 GB/s | **175.9** (`6.60x` / `5.76x`) |

---

## Compressed-Domain Mutations, Column Splicing & Pushdown (`vxg-query-real`)

Measured on synthetic multi-template **`ndjson_varied.ndjson`** (`scripts/gen_varied_ndjson.py`, `56.50 MB` raw -> `5.17 MB` / `9.14%` VXG `v4`, `300,000` rows, `863` independent `64 KiB` blocks) and synthetic **`data.csv`** (`scripts/gen_format_bench.py`, `30.53 MB` -> `5.48 MB` / `17.96%`, `250,001` rows, Apple Silicon NEON, persistent worker pool, median of 5 runs via `vxg-query-real`). Every output is asserted 100% byte-for-byte identical to `decompress + raw op (+ compress)`:

| Workload (`300k` rows unless noted) | `1T` VXG (`ms` / `M rows/s`) | `1T` vs Naive Decode+Op | `8T` VXG (`ms` / `M rows/s`) | `8T` vs Naive Decode+Op |
| :--- | :---: | :---: | :---: | :---: |
| **`filter_column` (`level == "NOPE"`, 0-hit alphabet)** | `1.54 ms` (`194.8 M/s`) | `59.7×` | **`0.80 ms` (`375.0 M/s`)** | `103.4×` |
| **`filter_column` (`bytes >= 50000`, zone-map skip)** | `2.36 ms` (`127.1 M/s`) | `42.8×` | **`0.93 ms` (`322.6 M/s`)** | `100.6×` |
| **`top_k_by` (`ORDER BY latency_ms ASC LIMIT 10`)** | `1.45 ms` (`206.9 M/s`) | `63.2×` | **`0.58 ms` (`517.2 M/s`)** | `138.9×` |
| **`agg SUM/MIN/MAX(bytes)` (all `300k` rows)** | `1.70 ms` (`176.5 M/s`) | `62.6×` | **`0.57 ms` (`526.3 M/s`)** | `152.0×` |
| **`group_by(service) COUNT(*)` (`8` groups)** | `2.23 ms` (`134.5 M/s`) | `46.4×` | **`1.35 ms` (`222.2 M/s`)** | `79.3×` |
| **`group_by(service) SUM/MIN/MAX(bytes)`** | `4.25 ms` (`70.6 M/s`) | `40.3×` | **`1.00 ms` (`300.0 M/s`)** | `157.9×` |
| **`select [service,status,bytes]` (all `300k` -> `16.2 MB`)** | `8.61 ms` (`34.8 M/s`) | `15.0×` | **`3.07 ms` (`97.7 M/s`)** | `42.2×` |
| **`filter_frame` (`status == 500` -> `.vxg`)** | `13.25 ms` (`22.6 M/s`) | `8.0×` | **`2.18 ms` (`137.6 M/s`)** | `40.1×` |
| **`update_column WHERE latency_ms>=250 SET CRITICAL`** | `4.89 ms` (`61.3 M/s`) | `30.8×` | **`1.40 ms` (`214.3 M/s`)** | `66.6×` |
| **`map_column bytes -> bytes + 100` (1-col numeric splice)** | `4.86 ms` (`61.7 M/s`) | `32.3×` | **`1.66 ms` (`180.7 M/s`)** | `61.9×` |
| **`data.csv` `filter_col` / `update` / `map` (`250k` rows)** | `0.58 / 1.96 / 2.32 ms` (`431 / 128 / 108 M/s`) | `113.7× / 71.2× / 62.9×` | `2.48 / 2.89 / 3.25 ms` | `21.8×–36.5×` |

---

## 4. Known Issues (`v0.4.7`) & `v0.5` Roadmap

### Known Issues (`v0.4.7`)
1. **Unstructured / Real CLF Log Compression Ratio vs. `Zstd + Trained Dict`:** On real `apache-access.clf` (`0.1198` vs `0.0878`), `macos-install.log` (`0.0948` vs `0.0484`), and deeply nested `gharchive` (`0.1985` vs `0.1150`), free-text URL paths, User-Agents, and arbitrary JSON payloads fall back to `Fast-LZ + 4-stream Huffman`, which trails `Zstd`'s `FSE + optimal parsing + shared 32 KiB dictionary`.
2. **Compound `COUNT(*) WHERE` Non-Short-Circuit Unpacking (`1T` `3.84 ms` vs `SUM` `1.90 ms` on `nyc-taxi.ndjson`):** Multi-column `AND` expressions (`payment_type==1 && tip_cents>=1000 && distance_centimiles>=500`) currently unpack and evaluate every leaf `MODEL_DECIMAL_PACKED` column across all template rows independently before intersecting bitsets, rather than propagating the surviving `BitSet` mask from the first selective predicate into subsequent columns.
3. **Numeric `GROUP BY` Per-Block `HashMap` Overhead (`ops_group.rs`):** Low-cardinality integer columns encoded as `MODEL_DECIMAL_PACKED` (`GROUP BY payment_type`: `2.97 ms` `1T` / `1.22 ms` `8T` vs `DuckDB + Parquet` `1.25 / 1.02 ms`) allocate and merge a per-block `HashMap<u64, AggStats>` across `800+` blocks instead of using a zero-allocation stack bucket array like `MODEL_DICTIONARY` (`GROUP BY service`: `0.95 ms` `8T`).

### Planned for `v0.5` ([`docs/ROADMAP.md`](docs/ROADMAP.md))
- **`BLOCK_ZSTD_DICT` Hybrid Block Option:** Add a shared per-frame `32 KiB` dictionary + `Zstd` block route alongside `BLOCK_COLUMN` / `BLOCK_LZ` / `BLOCK_RAW` so unstructured logs, nested JSON, and prose never trail `Zstd + Trained Dict` in block ratio while flat tabular blocks retain compressed-domain SQL.
- **Short-Circuit `BitSet` Mask Propagation & Stack-Bucket Numeric `GROUP BY`:** Eliminate redundant column unpacking on selective `AND` clauses and remove per-block `HashMap` allocations on narrow `MODEL_DECIMAL_PACKED` group keys.
- **Out-of-Cache `40M+`-Row Scale Benchmark:** Evaluate streaming/`mmap` `VXG` vs. `DuckDB + Parquet` on full-year NYC TLC (`~40M` rows, multi-GB out-of-cache).

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

# Compressed-domain structured record random access, compound WHERE & SELECT projection:
./target/release/vxg query input.vxg --count
./target/release/vxg query input.vxg --row 42
./target/release/vxg query input.vxg --range 100:150
./target/release/vxg query input.vxg --select ts,service,status,latency_ms \
  --where 'status==500 && (service==fraud-engine || service==auth-gateway) && latency_ms>=150' \
  --limit 20 -t auto

# Compressed-domain GROUP BY / SUM / AVG analytics & zero-reencode VXG -> VXG column splicing:
./target/release/vxg agg input.vxg --group-by service --col bytes --where 'status==500' -t auto
./target/release/vxg project input.vxg slim.vxg --cols service,status,bytes,latency_ms -t auto -f

# Compressed-domain single-column mutations without full decode/re-encode:
./target/release/vxg update input.vxg patched.vxg --where status==500 --col level --set CRITICAL -f
./target/release/vxg map input.vxg upper.vxg --col service --upper -f

# Unix bounded-memory streaming pipeline via stdin / stdout ('-') or --stream:
cat input.ndjson | ./target/release/vxg c - - --stream -t auto -q \
  | ./target/release/vxg project - - --cols service,status,latency_ms --stream -t auto -q \
  | ./target/release/vxg query - --select service,latency_ms --where 'status==500' -t auto
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

// Compressed-domain structured record random access, analytics & projection:
let total_rows = vxg::frame_row_count(&frame)?;
let row0 = vxg::extract_row(&frame, 0, &dec_opts)?;
let where_ok = vxg::PredExpr::col(vxg::ColPred::json_str("status", vxg::StrCmp::Eq(b"ok")));
let stats = vxg::aggregate_column(&frame, Some(&where_ok), vxg::ColTarget::JsonKey("seq"), &dec_opts)?;
let cols = [vxg::ColTarget::JsonKey("ts"), vxg::ColTarget::JsonKey("status")];
let projected = vxg::select_columns(&frame, Some(&where_ok), &cols, 10, &dec_opts)?;
let top_rows = vxg::top_k_by(&frame, Some(&where_ok), vxg::ColTarget::JsonKey("seq"), vxg::SortDir::Desc, 5, &dec_opts)?;
let slim_frame = vxg::project_frame(&frame, None, &cols, &opts, &dec_opts)?;
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
src/                     Rust codec core (lib.rs, ffi.rs, stream.rs, simd.rs, crc32c.rs, gpu.rs)
src/columns/             Modular structured-column, query/analytics/splice kernels, stride/BMP & Fast-LZ
src/query/               Compressed-domain row seek, bigram/nibble search, and parallel pushdown
src/ops/                 Compressed-domain compound WHERE, agg/group_by, filter/map/update & column projection
src/frame/               Format v1–v4 frame encoder, parser, and zero-allocation parallel decoder
src/block/               Block LZ encoder and SIMD/SWAR decoder submodules
src/packing/             Command-metadata & entropy research submodules (--features bench)
src/groups/              Fixed-group command-layout evaluator submodules (--features bench)
src/bin/                 Production CLI (`vxg`) and benchmark harnesses (`vxg-bench`, `vxg-query-real`)
include/vxg.h            Official C99 / C++11 header for static & dynamic linking
examples/ffi/            Zero-copy integration examples for C, Python, Go, Bun/JS, C#, and WebAssembly
native/shaders/core.gpu  Shared GPU kernel source (Metal, Vulkan GLSL, CUDA)
native/                  Metal, Vulkan, and CUDA host bridges
verification/            Independent Python reference model and C++ SIMT emulator
tests/                   Property, stress, differential analytics/projection, and golden wire tests
reports/ci-v0.4.1/       Cross-platform GitHub Actions benchmark outputs (macOS, Linux, Windows)
docs/                    Wire format, FFI guide, proofs, benchmarks, GPU architecture, and safety
```

[Wire Format](docs/FORMAT.md) · [FFI & Language Bindings](docs/FFI.md) · [Benchmarks](docs/BENCHMARKS.md) · [Proofs](docs/PROOFS.md) · [GPU Architecture](docs/GPU.md) · [Safety](docs/SAFETY.md) · [Validation](docs/VALIDATION.md)

---

## License

MIT License.
