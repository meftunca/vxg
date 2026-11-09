# VXG 0.5.1 — Unified Adaptive Multi-Format Lossless Block Codec & SQL Engine (Format v4/v5)

[![CI — Source Validation](https://github.com/meftunca/vxg/actions/workflows/ci.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/ci.yml)
[![CI — Cross-Platform Benchmarks](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml)
[![Release v0.5.1](https://img.shields.io/github/v/release/meftunca/vxg)](https://github.com/meftunca/vxg/releases/tag/v0.5.1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Store raw NDJSON/CSV verbatim (`2.4×–16.7×` faster ingest than Parquet ETL), `1.17×–1.32×` smaller `64 KiB` random-access blocks than `Zstd + Trained Dictionary` on flat tabular records, beat `Zstd-9` and `Zstd+Dict-3` across the 6-dataset real public geomean (`0.1213`, within `0.66%` of `Zstd+Dict-9`), run SQL (`vxg sql` — `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY Top-K`, zero-reencode splicing) directly on compressed files up to `40M+` rows, and recover the original file byte-for-byte.**

> **Scope on Real-World Data (`Section 1`, `Section 2`, & `Section 3`):**
> - **Where VXG wins:** Flat tabular records (`nyc-taxi.ndjson`, `nyc-taxi.csv` — `1.17×–1.32×` smaller than `Zstd + Trained Dict L9`, faster `8T` encode & decode than `LZ4 (a1)` on NDJSON, and **zero-ETL 100% byte-exact original file preservation**), **all `15 / 15` in-cache `8T` analytical queries**, and **all `5 / 5` out-of-cache `8T` queries on `40,000,000`-row (`3.06 GB`) full-year NYC TLC CSV** against prepared `DuckDB 2.0 + Parquet`. On the 6-dataset real public geomean, `VXG v0.5.1` (`0.1213`) beats `LZ4` (`0.2500`), `Zstd-1` (`0.1620`), `Zstd-9` (`0.1419`), and `Zstd+Dict-3` (`0.1407`) — and beats `Zstd-9` + `Zstd+Dict-3` individually on `gharchive`, `apache-access.clf`, and `shakespeare` — using pure Rust (`0` C dependencies, `wasm32` compatible).
> - **Where competitors win:** On deeply nested JSON (`gharchive`: `0.1150` vs `0.1231`), free-text/HTTP logs (`apache-access.clf`: `0.0878` vs `0.1037`, `macos-install.log`: `0.0484` vs `0.0594`), and prose (`shakespeare`: `0.3745` vs `0.3884`), `Zstd + Trained Dict L9` (`FSE + optimal parsing`) leads `64 KiB` block ratio (`0.1205` vs `0.1213` geomean), `LZ4` leads raw throughput on CSV and non-tabular files (`5.72 / 22.86 GB/s` vs `2.17 / 14.46 GB/s` `8T` geomean), and `DuckDB + Parquet` leads `1T` NDJSON queries and selective `10M`-row (`2.62 GB`) NDJSON zone-map queries (`8.02–13.20 ms` vs `13.06–32.12 ms`).

VXG `v4/v5` inspects each independent `64 KiB` block at runtime and routes it by exact byte budget to `BLOCK_COLUMN` (grammar + typed column models), `BLOCK_DICT_LZ` (`v5` self-contained frame prefix dictionary + 5-stream canonical Huffman + 3-repcode LZ77), `BLOCK_LZ` (2D stride/pixel planes or `Fast-LZ`), or `BLOCK_RAW`. Every block is self-contained, enabling **$O(1)$ random-access block extraction** and **multi-core scaling** without cross-block decompression dependencies.

---

## Key Capabilities

| Area | Highlights |
| :--- | :--- |
| **Format v4/v5 Wire Layout** | Compact 8-byte per-block directory (`payload_size \| kind << 30`, `crc32c`); `v5` adds self-contained frame prefix dictionary (`FLAG_FRAME_DICT`, `<= 256 KiB`), `BLOCK_DICT_LZ` (`kind = 3`), and `\n`-snapped `--record-aligned` variable blocks (`FLAG_VAR_RAW`); 100% backward read compatibility with `v1`–`v4` frames |
| **Structured Column Engine** | Automatic schema & delimiter discovery for **NDJSON, 1st-Level Nested JSON (`actor.login`), Pretty-Printed JSON, CSV/TSV/PSV (with Row-0 header resolution), CLF/Bracketed Logs, and Logfmt**; 12 specialized models including `Dictionary`, `DecimalDelta`, `ScaledDecimal` (`tag 16` fixed-scale currency/floats), `Timestamp` (quoted & unquoted ISO-8601 SWAR+SIMD), `HexSuffix`, `EpochCounter`, `SegmentDictionary`, `FrontCodedDict`, and `Pattern` |
| **Compressed-Domain SQL (`vxg sql`) & Zero-ETL Archive** | Store raw NDJSON/CSV verbatim, execute `SELECT col, COUNT(*), SUM/MIN/MAX/AVG(col) ... WHERE ... GROUP BY ... ORDER BY ... LIMIT K` (`vxg sql`, `vxg::sql`, `vxg_sql_alloc` C ABI) with `DenseRange` group buckets and short-circuit `RowBitset` predicates (`0.16–1.32 ms` on `200k–300k` rows; `17.5–44.4 ms` on `40M` CSV rows), plus zero-reencode `VXG -> VXG` column splicing, and recover the original file byte-for-byte |
| **Image & Stride Engine** | 24-bpp / 32-bpp BMP and multi-stride (`1, 2, 3, 4, 8`) byte-plane transpose + `Subtract-Green` (`R-G`, `B-G`) color decorrelation + 2-level 64-bit hierarchical zero-mask & per-plane `Fast-LZ` |
| **Unstructured Logs, Prose & Nested JSON** | Pure-Rust FastCover + macroscopic SWAR `sketch16` frame dictionary trainer (`<= 256 KiB`, embedded in-frame) + L1D `u16` / register-tagged `PreparedDict` `BLOCK_DICT_LZ` (3 repcodes, backward extension, 5-stream canonical Huffman / zero-Huf raw fast path, SIMD `copy16` decode) |
| **Cross-Platform SIMD & SWAR** | Runtime-dispatched `NEON` (`aarch64`) and `SSE2` / `AVX2` / `AVX-512` (`x86_64`) fast paths, including 32-byte branchless string scanning, 64-byte `FastPiece` unaligned `u128` row assembly, and SIMD prefix-sum plane interleaving |
| **Integrity & Safety** | Hardware-accelerated CRC32C (`ARMv8 CRC` / `SSE4.2`); strict output-size bounds and deterministic rejection of truncated or malformed inputs |
| **Zero-Copy `mmap` & Streaming Engine** | Zero-dependency cross-platform `MappedFile` / `MappedFileMut`, zero-output-alloc parallel `verify_frame` (`12+ GB/s`, `~64 KiB` L2 scratch/thread), and bounded-memory (`O(single_frame)`) multi-GB streaming across `sql`, `query`, `agg`, `project`, `filter-frame`, `update`, and `map`, plus `staticlib` / `cdylib` C ABI ([`include/vxg.h`](include/vxg.h)) |

---

## 1. Real-World Public & System Corpora (`64 KiB` Blocks vs. `Zstd + Trained Dict` & `zstd -19 --long`)

Measured via `vxg-bench` (`Apple Silicon aarch64 NEON`, `64 KiB` independent blocks, `liblz4 1.10.0`, `libzstd 1.5.7`, checksums off; fetched reproducibly via [`scripts/fetch_real_corpora.py`](scripts/fetch_real_corpora.py)).
- **`Zstd + Dict` (`zstd-dict-l3` / `zstd-dict-l9`):** Trains up to a `32 KiB` external dictionary per dataset (`ZDICT_trainFromBuffer`), charging `dict_bytes` once; `Enc`/`Dec` measure steady-state `ZSTD_CDict`/`ZSTD_DDict` throughput (excluding `35–108 ms` `ZDICT` training). **`VXG v0.5.1`** trains and embeds its `<= 256 KiB` dictionary inside the `.vxg` frame itself, and `VXG Enc`/`Dec` **include** full dictionary training and header decoding time.
- **Whole-file `zstd -19 --long` (`128 MiB` window, single stream, `random access = NO`):** Reference column quantifying the exact compression-ratio cost of `64 KiB` block independence.
- **Reading the tables:** Bold cells mark **only the actual per-row winner** on each metric; codecs storing near-raw bytes (`ratio >= 0.90`) are excluded from speed bolding. Full `1T` & `8T` tables and parameter sweeps: **[`docs/CHIMERA_BENCHMARKS.md`](docs/CHIMERA_BENCHMARKS.md)**, [`reports/real-public-1t.md`](reports/real-public-1t.md), and [`reports/real-public-8t.md`](reports/real-public-8t.md).

| Real Public / OS Corpus (`8T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc / Dec (`GB/s`) | LZ4 (`a1`) Enc / Dec | Zstd-1 Enc / Dec | Zstd+D3 Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** *(NYC TLC, 200k trips)* | 52.3 MB | **0.0592** | 0.1899 | 0.0823 | 0.0783 | 0.0554 | **9.03 / 38.16** | 7.98 / 27.81 | 5.84 / 11.62 | 3.62 / 10.22 |
| **`nyc-taxi-2024-01.csv`** *(NYC TLC, 300k trips)* | 23.0 MB | **0.1829** | 0.4241 | 0.2194 | 0.2132 | 0.1617 | 3.43 / 13.68 | **3.77 / 18.74** | 2.94 / 7.15 | 1.60 / 4.99 |
| **`gharchive-2025-01-15.ndjson`** *(12k nested GitHub API)* | 50.3 MB | 0.1231 | 0.2256 | 0.1414 | **0.1150** | 0.0633 | 1.85 / 14.35 | **7.15 / 31.37** | 6.13 / 12.14 | 3.01 / 13.37 |
| **`apache-access.clf`** *(Elastic real HTTP CLF)* | 2.4 MB | 0.1037 | 0.1730 | 0.1062 | **0.0878** | 0.0535 | 1.00 / 12.94 | **6.20 / 16.06** | 4.86 / 9.15 | 2.41 / 7.99 |
| **`macos-install.log`** *(OS free-text multi-line log)* | 34.2 MB | 0.0594 | 0.1249 | 0.0770 | **0.0484** | 0.0278 | 3.97 / 17.47 | **11.57 / 43.96** | 7.16 / 18.83 | 4.20 / 17.68 |
| **`pg100-shakespeare.txt`** *(Gutenberg English prose)* | 5.6 MB | 0.3884 | 0.6223 | 0.3918 | **0.3745** | 0.3000 | 0.46 / 5.39 | **2.27 / 12.36** | 2.27 / 7.53 | 0.82 / 5.09 |
| **`kodak-truecolor-16.bmp`** *(16 real camera photos)* | 18.9 MB | 0.8256 | 0.8732 | 0.7494 | **0.7452** | 0.5656 | 0.87 / 13.91 | **5.27 / 21.29** | 2.97 / 5.80 | 0.62 / 5.86 |
| **Text & Structured Geomean (6)** | — | 0.1213 | 0.2500 | 0.1419 | **0.1205** | 0.0795 | 2.17 / 14.46 | **5.72 / 22.86** | 4.50 / 10.46 | 2.29 / 8.88 |

**Summary of Real-World Trade-Offs:**
- **Flat tabular records (`nyc-taxi.ndjson`, `nyc-taxi.csv`):** `VXG v0.5.1` achieves the smallest `64 KiB` blocks (`1.17×–1.32×` smaller than `Zstd + Trained Dict L9`, `2.32×–3.21×` smaller than `LZ4`, and within `6.9%–13.1%` of non-random-access whole-file `zstd -19 --long`). On `nyc-taxi.ndjson`, `VXG` beats `LZ4 (a1)` on both `8T` encode (`9.03` vs `7.98 GB/s`) and decode (`38.16` vs `27.81 GB/s`); on `nyc-taxi.csv`, `VXG` reaches `3.43 / 13.68 GB/s` (`8T`, beating `LZ4` `1T` encode `0.639` vs `0.598 GB/s`, while `LZ4` leads `8T` raw speed at `3.77 / 18.74 GB/s`).
- **Unstructured logs, nested JSON & prose (`gharchive`, `apache.clf`, `macos-install.log`, `shakespeare`):** `v0.5.1`'s L1D `u16` + register-tagged `PreparedDict` `BLOCK_DICT_LZ` engine brings **`gharchive` to `0.1231`** (`14.35 GB/s` `8T` decode — beating both `Zstd-9` `0.1414` and `Zstd+Dict-3` `0.1254` in ratio and decode speed) and **`pg100-shakespeare.txt` to `0.3884`** (beating both `Zstd-9` `0.3918` and `Zstd+Dict-3` `0.4005` on pure English prose!), pushing the **6-dataset real geomean to `0.1213`** at **`14.46 GB/s` `8T` decode** (`1.63×` faster than `Zstd+Dict-3`). `Zstd + Trained Dict L9` (`0.1205`) retains a `0.66%` geomean edge via FSE + optimal parsing, and `LZ4` remains faster in raw `GB/s` on unstructured text.

---

## 2. In-Cache Analytical Queries (`200k–300k` Rows): `VXG v0.5.1` vs. `DuckDB 2.0.0 + Parquet`

Measured standalone with zero background CPU contention (`libvxg.dylib` vs `duckdb 2.0.0`, warm OS cache, median of 5–7 runs, **100% result equality verified** via [`scripts/bench_vs_duckdb.py`](scripts/bench_vs_duckdb.py); full `1T` & `8T` tables in [`docs/CHIMERA_BENCHMARKS.md#duckdb`](docs/CHIMERA_BENCHMARKS.md#duckdb) and [`reports/duckdb-vs-vxg.json`](reports/duckdb-vs-vxg.json)). Parquet columns report **`PREPARE / EXECUTE` (`prep`)** latencies (`SELECT 1` overhead = `0.04–0.05 ms` `1T`, `0.06–0.07 ms` `8T`).

| Corpus & Storage / Query (`8T`, `SELECT 1` prep = `0.06–0.07 ms`) | **VXG `.vxg`** | DuckDB Raw | DuckDB `.zst` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson` (`52.3 MB` real, `200k` rows)** | **`3.10 MB` (`5.6 ms` enc, lossless=YES)** | `52.33 MB` (`0 ms`, YES) | `4.92 MB` (`34.5 ms`, YES) | `3.94 MB` (`84.7 ms` ETL — `15.0×` slower, **NO**) | **`2.96 MB`** (`94.2 ms` ETL — `16.7×` slower, **NO**) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | **`0.70 ms` (`286.5 M/s`)** (`1T`: `4.72 ms`) | `56.12 ms` | `124.47 ms` | `1.18 ms` (`1T`: **`1.50 ms`**) | `1.55 ms` (`1T`: `2.09 ms`) |
| ↳ `SUM/MIN/MAX(total_cents)` *(all 200k rows)* | **`0.35 ms` (`570.7 M rows/s`)** | `55.65 ms` | `122.41 ms` | `0.88 ms` (`1T`: **`1.12 ms`**) | `1.04 ms` (`192.0 M rows/s`) |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | **`0.66 ms` (`303.0 M rows/s`)** | `55.77 ms` | `123.05 ms` | `1.06 ms` (`1T`: **`1.33 ms`**) | `1.24 ms` (`160.9 M rows/s`) |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | **`0.70 ms` (`287.4 M/s`)** (`1T`: `3.09 ms`) | `55.76 ms` | `123.66 ms` | `1.07 ms` (`1T`: **`1.29 ms`**) | `1.20 ms` (`1T`: `1.72 ms`) |
| ↳ `ORDER BY total_cents DESC LIMIT 10` | **`1.19 ms` (`168.6 M rows/s`)** | `59.25 ms` | `131.19 ms` | `2.31 ms` (`1T`: **`2.90 ms`**) | `3.46 ms` (`57.9 M rows/s`) |
| ↳ `SELECT 4 cols WHERE tip_cents >= 2500` | **`0.97 ms` (`205.6 M rows/s`)** | `60.01 ms` | `133.02 ms` | `3.24 ms` (`2.60 ms` ad-hoc) | `5.61 ms` (`4.19 ms` ad-hoc) |
| **`nyc-taxi-2024-01.csv` (`23.0 MB` real, `300k` rows)** | **`4.21 MB` (`7.3 ms` enc, lossless=YES)** | `23.00 MB` (`0 ms`, YES) | `5.01 MB` (`46.8 ms`, YES) | `5.86 MB` (`94.8 ms` ETL — `13.0×` slower, **NO**) | `4.39 MB` (`99.5 ms` ETL — `13.7×` slower, **NO**) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | **`0.16 ms` (`1,906.8 M/s`)** (`1T`: **`0.90 ms`**) | `51.84 ms` | `119.60 ms` | `1.24 ms` (`1T`: `2.04 ms`) | `1.75 ms` (`171.6 M rows/s`) |
| ↳ `SUM/MIN/MAX(total_cents)` *(all 300k rows)* | **`0.21 ms` (`1,417.0 M rows/s`)** | `50.69 ms` | `116.01 ms` | `0.99 ms` (`302.7 M rows/s`) | `1.11 ms` (`270.2 M rows/s`) |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | **`0.22 ms` (`1,351.9 M rows/s`)** | `51.20 ms` | `114.35 ms` | `1.14 ms` (`262.8 M rows/s`) | `1.33 ms` (`226.2 M rows/s`) |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | **`0.42 ms` (`713.6 M rows/s`)** | `51.97 ms` | `119.65 ms` | `1.05 ms` (`1T`: **`1.68 ms`**) | `1.27 ms` (`236.7 M rows/s`) |
| **`ndjson_varied.ndjson` (`56.5 MB` synthetic, `300k` rows)** | **`5.17 MB` (`9.0 ms` enc, lossless=YES)** | `56.50 MB` (`0 ms`, YES) | `9.86 MB` (`50.8 ms`, YES) | `9.70 MB` (`118.1 ms` ETL — `13.1×` slower, **NO**) | `5.56 MB` (`127.2 ms` ETL — `14.1×` slower, **NO**) |
| ↳ `COUNT(*) WHERE fraud & status=500 & lat>=150` | **`0.90 ms`** (`1T`: `3.95 ms`) | `64.01 ms` | `153.99 ms` | `0.94 ms` (`1T`: **`1.23 ms`**) | `1.05 ms` (`1T`: `1.62 ms`) |
| ↳ `SUM/MIN/MAX(bytes)` *(all 300k rows)* | **`0.29 ms` (`1,023.6 M rows/s`)** | `59.37 ms` | `146.21 ms` | `1.32 ms` (`227.7 M rows/s`) | `1.91 ms` (`157.2 M rows/s`) |
| ↳ `GROUP BY service: COUNT(*), SUM(bytes)` | **`0.84 ms` (`358.9 M rows/s`)** | `64.07 ms` | `155.32 ms` | `1.73 ms` (`173.9 M rows/s`) | `2.30 ms` (`130.7 M rows/s`) |
| ↳ `ORDER BY latency_ms DESC LIMIT 10` | **`0.90 ms` (`333.7 M rows/s`)** | `63.44 ms` | `156.39 ms` | `2.42 ms` (`124.0 M rows/s`) | `2.74 ms` (`109.4 M rows/s`) |

---

## 3. Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Real 2024 NYC TLC Rows)

Measured over **all 12 months of unmodified 2024 NYC TLC Yellow Taxi trip data** (`41.17M` official rows; [`scripts/bench_scale_duckdb.py`](scripts/bench_scale_duckdb.py), [`reports/scale-duckdb-vs-vxg.md`](reports/scale-duckdb-vs-vxg.md), **100% numerical equality & lossless `vxg_verify_file` verified**):
- **`nyc-taxi-2024-40m.csv` (`40,000,000` rows, `3.06 GB` raw):** `VXG --record-aligned` (**`551.6 MB`**, **`1.56 s`** `8T` ingest, `lossless = YES`) is **smaller and `2.4×–2.6×` faster to ingest** than both `Parquet-Zstd` (`579.4 MB`, `3.80 s`, `lossless = NO`) and `Parquet-Snappy` (`773.2 MB`, `4.06 s`, `lossless = NO`), and **wins `5 / 5` `8T` queries**.
- **`nyc-taxi-2024-10m.ndjson` (`10,000,000` rows, `2.62 GB` raw):** `VXG --record-aligned` (`145.9 MB`, `1.02 s` `8T` ingest, `lossless = YES`) is `1.33×` smaller than `Parquet-Snappy` (`193.3 MB`) and matches `Parquet-Snappy` on full-scan `SUM` (`8.26` vs `8.20 ms`, beating `Parquet-Zstd` `9.69 ms`), while **`DuckDB + Parquet` leads** selective `COUNT WHERE`, `SUM WHERE`, `GROUP BY`, and `Top-10` due to `122k`-row row-group zone maps and stripped JSON key headers.

| Multi-GB Scale Query (`8T`, `SELECT 1` prep = `0.09 ms`) | `40M` CSV: **`VXG --record-aligned`** | `40M` CSV: `DuckDB Parquet-Snappy / Zstd` *(prep)* | `10M` NDJSON: `VXG --record-aligned` | `10M` NDJSON: **`DuckDB Parquet-Snappy / Zstd`** *(prep)* |
| :--- | :---: | :---: | :---: | :---: |
| **Archive Size & `8T` Ingest / ETL** | **`551.6 MB` (`1.56 s`, lossless=YES)** | `773.2 MB (4.06 s)` / `579.4 MB (3.80 s)` *(NO)* | `145.9 MB` (`1.02 s`, **lossless=YES**) | `193.3 MB (0.93 s)` / **`144.6 MB`** `(1.18 s)` *(NO)* |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | **`29.72 ms`** *(std: **`27.71 ms`**, `1,443 M/s`)* | `34.55 ms` / `57.44 ms` *(raw CSV: `1.26 s`)* | `32.12 ms` | **`9.36 ms`** / `12.43 ms` *(raw JSON: `0.42 s`)* |
| `SUM/MIN/MAX(total_cents)` *(full scan)* | **`18.41 ms` (`2,172.7 M rows/s`)** | `31.33 ms` / `48.85 ms` | `8.26 ms` (`1,210.7 M/s`) | **`8.20 ms`** / `9.69 ms` |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | **`18.59 ms` (`2,151.7 M rows/s`)** | `31.58 ms` / `36.91 ms` | `13.06 ms` | **`8.02 ms`** / `10.16 ms` |
| `GROUP BY payment_type: COUNT(*), SUM(total)` | **`44.39 ms` (`901.1 M rows/s`)** | `49.14 ms` / `48.98 ms` | `16.81 ms` | **`13.20 ms`** / `13.62 ms` |
| `ORDER BY total_cents DESC LIMIT 10` | **`17.53 ms` (`2,281.8 M rows/s`)** | `18.39 ms` / `22.74 ms` | `22.92 ms` | **`9.09 ms`** / `11.68 ms` |

---

## 4. Synthetic & Procedural Qualification Suites (`--suite v4` & Multi-Template Generators)

> **Important:** All datasets below (`--suite v4`, `scripts/gen_varied_ndjson.py`, `scripts/gen_format_bench.py`) are **synthetic / procedurally generated** qualification workloads. The two `.bmp` files in `--suite v4` are procedural flat-shaded UI/gradient framebuffers (`208:1`–`833:1`) and are separated from the 8 text/structured datasets below (`reports/synthetic-v4-{1t,8t}.md`, `reports/synthetic-multitpl-{1t,8t}.md`).

| Synthetic Suite (`64 KiB` blocks) | **VXG v0.5.1 Ratio** | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | `1T` VXG vs LZ4 (`Enc / Dec GB/s`) | `8T` VXG vs LZ4 (`a1`) (`Enc / Dec GB/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v4` Text & Structured Geomean (8 sets, `16 MiB`)** | **0.0259** | 0.2107 | 0.0859 | 0.0846 | 0.0081 | **`1.58` / `9.69`** vs `1.51` / `5.79` | `6.34` / **`50.84`** vs **`8.17`** / `28.13` |
| **`v4` Procedural UI BMP Geomean (2 sets, `16 MiB`)** | **0.0024** | 0.0178 | 0.0103 | 0.0111 | 0.0010 | `8.69` / **`14.69`** vs **`11.64`** / `14.30` | `27.61` / **`75.11`** vs **`35.50`** / `54.52` |
| **Synthetic Multi-Template (`ndjson_varied`, `56.5 MB`)** | **0.0914** | 0.2898 | 0.1480 | 0.1452 | 0.1198 | `1.04` / **`5.65`** vs **`1.17`** / `4.57` | **`6.20` / `40.29`** vs `5.98` / `25.44` |
| **Synthetic Delimited (`data.csv` / `data.clf`, `31 MB`)** | **0.1794 / 0.0980** | 0.4458 / 0.2843 | 0.2651 / 0.1504 | 0.2606 / 0.1454 | 0.2257 / 0.1225 | `0.67 / 2.36` · `0.61 / 2.53` (`LZ4` faster) | `3.78/14.73` · `3.61/`**`17.92`** (`LZ4 a1`: `17.25`) |

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

Measured on synthetic multi-template **`ndjson_varied.ndjson`** (`scripts/gen_varied_ndjson.py`, `56.50 MB` raw -> `5.17 MB` / `9.14%`, `300,000` rows, `863` independent `64 KiB` blocks) and synthetic **`data.csv`** (`scripts/gen_format_bench.py`, `30.53 MB` -> `5.48 MB` / `17.94%`, `250,001` rows, Apple Silicon NEON, persistent worker pool, median of 5 runs via `vxg-query-real`). Every output is asserted 100% byte-for-byte identical to `decompress + raw op (+ compress)`:

| Workload (`300k` rows unless noted) | `1T` VXG (`ms` / `M rows/s`) | `1T` vs Naive Decode+Op | `8T` VXG (`ms` / `M rows/s`) | `8T` vs Naive Decode+Op |
| :--- | :---: | :---: | :---: | :---: |
| **`filter_column` (`level == "NOPE"`, 0-hit alphabet)** | `1.62 ms` (`185.6 M/s`) | `50.8×` | **`0.75 ms` (`401.7 M/s`)** | `98.4×` |
| **`filter_column` (`bytes >= 50000`, zone-map skip)** | `1.81 ms` (`165.9 M/s`) | `52.3×` | **`0.80 ms` (`375.1 M/s`)** | `107.2×` |
| **`top_k_by` (`ORDER BY latency_ms ASC LIMIT 10`)** | `1.47 ms` (`204.5 M/s`) | `67.5×` | **`0.63 ms` (`479.3 M/s`)** | `140.5×` |
| **`agg SUM/MIN/MAX(bytes)` (all `300k` rows)** | `1.60 ms` (`187.5 M/s`) | `60.0×` | **`0.50 ms` (`595.2 M/s`)** | `171.8×` |
| **`group_by(service) COUNT(*)` (`8` groups)** | `2.14 ms` (`140.4 M/s`) | `45.2×` | **`0.38 ms` (`797.0 M/s`)** | `235.2×` |
| **`group_by(service) SUM/MIN/MAX(bytes)`** | `2.94 ms` (`101.9 M/s`) | `54.3×` | **`0.50 ms` (`596.3 M/s`)** | `298.4×` |
| **`select [service,status,bytes]` (all `300k` -> `16.2 MB`)** | `5.64 ms` (`53.2 M/s`) | `24.9×` | **`2.21 ms` (`135.7 M/s`)** | `59.5×` |
| **`project_frame` (`[service,status,bytes,lat]` splice -> `.vxg`)** | `6.34 ms` (`47.3 M/s`) | `26.2×` | **`2.00 ms` (`150.0 M/s`)** | `72.8×` |
| **`update_column WHERE latency_ms>=250 SET CRITICAL`** | `3.42 ms` (`87.7 M/s`) | `43.2×` | **`1.14 ms` (`263.5 M/s`)** | `81.9×` |
| **`map_column bytes -> bytes + 100` (1-col numeric splice)** | `4.33 ms` (`69.2 M/s`) | `35.7×` | **`1.49 ms` (`201.5 M/s`)** | `68.8×` |
| **`data.csv` `filter_col` / `update` / `map` (`250k` rows)** | `0.49 / 1.74 / 2.16 ms` (`514 / 144 / 115 M/s`) | `113.7× / 58.9× / 55.3×` | **`0.34 / 0.89 / 1.09 ms` (`739 / 281 / 229 M/s`)** | `60.9×–129.4×` |

---

## 5. Remaining Trade-Offs (`v0.5.1`) & Future Roadmap ([`docs/ROADMAP.md`](docs/ROADMAP.md))

1. **`10M+`-Row Selective NDJSON Queries vs. `DuckDB + Parquet`:** On `nyc-taxi-2024-10m.ndjson` (`2.62 GB`), `VXG --record-aligned` matches `Parquet-Snappy` on full-scan `SUM` (`8.26` vs `8.20 ms`) while remaining 100% byte-exact reversible, whereas `DuckDB + Parquet` leads selective `COUNT WHERE` (`9.36` vs `32.12 ms`) and `Top-10` (`9.09` vs `22.92 ms`) by pruning `122k`-row row-group zone maps in a single metadata table instead of inspecting `42,000` independent `64 KiB` block headers. Adding an optional macro-block zone-map summary footer is tracked in Stage 10.
2. **Deeply Nested (`>= 2`-Level) JSON & Optimal-Parsed FSE:** `v0.5.1` flattens 1st-level nested JSON objects (`actor.login`, `repo.name`) and pairs L1D `u16` + register-tagged `PreparedDict` matching with 5-stream canonical Huffman in `BLOCK_DICT_LZ` (`0.1213` 6-dataset real geomean vs `Zstd+Dict-9` `0.1205`). Closing the remaining `0.66%` gap on deeply nested JSON arrays/trees (`gharchive`) and literary prose without C dependencies requires multi-level array shredding and finite-state entropy (`FSE`).

---

## Quick Start & CLI Usage (`vxg`)

Requires **Rust 1.89+**.

```sh
# Build CLI, staticlib (libvxg.a), and cdylib (libvxg.dylib / .so / .dll):
cargo build --release

# Inspect detected CPU SIMD & GPU backends:
./target/release/vxg hardware

# Compress and decompress using all available CPU threads (--record-aligned snaps blocks to \n):
./target/release/vxg compress input.ndjson input.vxg --record-aligned --threads auto
./target/release/vxg decompress input.vxg restored.ndjson --threads auto --force

# Run SQL directly on compressed .vxg files (supports CSV header names & dotted nested JSON keys):
./target/release/vxg sql input.vxg \
  "SELECT service, COUNT(*), SUM(bytes), MAX(latency_ms) WHERE status = 500 GROUP BY service ORDER BY SUM(bytes) DESC LIMIT 10" \
  --format json -t auto

# Verify frame structure and per-block CRC32C checksums without writing output:
./target/release/vxg verify input.vxg --threads auto

# Inspect frame header, overall ratio, and block-type breakdown (RAW / LZ / COLUMN / DICT_LZ):
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
  | ./target/release/vxg sql - "SELECT service, COUNT(*) WHERE status = 500 GROUP BY service" --stream -t auto
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
let opts = vxg::Options { record_aligned: true, ..vxg::Options::default() };
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

// Compressed-domain SQL, structured record random access, analytics & projection:
let sql_q = vxg::sql::SqlQuery::parse("SELECT status, COUNT(*), SUM(seq) WHERE status = 'ok' GROUP BY status")?;
let tsv = vxg::sql::execute_sql(&frame, &sql_q, vxg::sql::SqlFormat::Tsv, true, &dec_opts)?;
let total_rows = vxg::frame_row_count(&frame)?;
let row0 = vxg::extract_row(&frame, 0, &dec_opts)?;
let cols = [vxg::ColTarget::JsonKey("ts"), vxg::ColTarget::JsonKey("status")];
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
src/                     Rust codec core (lib.rs, encode.rs, ffi.rs, stream.rs, simd.rs, crc32c.rs)
src/columns/             Modular structured-column, ScaledDecimal, query/splice kernels, stride/BMP & Fast-LZ
src/dict_lz/             Format v5 pure-Rust FastCover frame dictionary trainer & 5-stream `BLOCK_DICT_LZ`
src/sql/                 Zero-dependency SQL parser & compressed-domain execution engine (`vxg sql`)
src/query/               Compressed-domain row seek, bigram/nibble search, and parallel pushdown
src/ops/                 Compressed-domain compound WHERE, agg/group_by, filter/map/update & column projection
src/frame/               Format v1–v5 frame encoder, parser, `var_raw` splitter & zero-alloc parallel decoder
src/block/               Block LZ encoder and SIMD/SWAR decoder submodules
src/packing/             Command-metadata & entropy research submodules (--features bench)
src/groups/              Fixed-group command-layout evaluator submodules (--features bench)
src/bin/                 Production CLI (`vxg`) and benchmark harnesses (`vxg-bench`, `vxg-query-real`)
include/vxg.h            Official C99 / C++11 header for static & dynamic linking
examples/ffi/            Zero-copy integration examples for C, Python, Go, Bun/JS, C#, and WebAssembly
native/shaders/core.gpu  Shared GPU kernel source (Metal, Vulkan GLSL, CUDA)
native/                  Metal, Vulkan, and CUDA host bridges
verification/            Independent Python reference model and C++ SIMT emulator
crates/vxg-tensor/       Compressed-domain AI weight codec (.vxt), CB4-Fast Metal LLM/VLM (3D M-RoPE) & OpenAI SSE server
apps/vxt-studio/         Tauri v2 + React + shadcn/ui + AI SDK desktop app (Multimodal Vision Chat, Tensor Forge & API Hub)
tests/                   Property, stress, differential analytics/projection, and golden wire tests
reports/ci-v0.4.1/       Cross-platform GitHub Actions benchmark outputs (macOS, Linux, Windows)
docs/                    Wire format, tensor engine, FFI guide, proofs, benchmarks, GPU architecture, and safety
```

[Wire Format](docs/FORMAT.md) · [VXT Tensor & Metal LLM Engine](docs/TENSOR.md) · [FFI & Language Bindings](docs/FFI.md) · [Benchmarks](docs/BENCHMARKS.md) · [Proofs](docs/PROOFS.md) · [GPU Architecture](docs/GPU.md) · [Safety](docs/SAFETY.md) · [Validation](docs/VALIDATION.md)

---

## License

MIT License.
