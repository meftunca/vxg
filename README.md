# VXG 0.5.3 — Unified Adaptive Multi-Format Lossless Block Codec & SQL Engine (Format v4/v5)

[![CI — Source Validation](https://github.com/meftunca/vxg/actions/workflows/ci.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/ci.yml)
[![CI — Cross-Platform Benchmarks](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/meftunca/vxg/actions/workflows/benchmarks.yml)
[![Release v0.5.3](https://img.shields.io/github/v/release/meftunca/vxg)](https://github.com/meftunca/vxg/releases/tag/v0.5.3)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Store raw NDJSON/CSV verbatim (`1.2×–15.4×` faster ingest than Parquet ETL), achieve `1.17×–1.97×` smaller `64 KiB` random-access blocks than `Zstd + Trained Dictionary [32..256 KiB]` on flat tabular & cumulative time-series records, beat `Zstd+Dict-9 [32..256 KiB]` on the 6-dataset real public geomean (`0.1171` vs `0.1181`), run SQL (`vxg sql` — `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY Top-K`) or filtered zero-dependency Apache Arrow / DuckDB / Polars `read_vxg()` exports directly on compressed files up to `40M+` rows, and recover the original file byte-for-byte.**

> **Scope on Real-World Data (`Section 1`, `Section 2`, & `Section 3` — `1T` & `8T` reported side-by-side):**
> - **Where VXG wins:** Flat tabular and cumulative time-series records (`nyc-taxi.ndjson`, `nyc-taxi.csv`, `datasets/covid-19` — `1.17×–1.97×` smaller `64 KiB` blocks than `Zstd + Trained Dict L9 [32..256 KiB]`, faster `8T` encode & decode than `LZ4 (a1)` on NDJSON at `8.84 / 38.59 GB/s` vs `6.36 / 21.98 GB/s`, and **zero-ETL 100% byte-exact original file preservation**); the **6-dataset real public `64 KiB` geomean** (`0.1171`, ahead of `Zstd+Dict-9 [32..256K]` `0.1181`, `Zstd+Dict-3` `0.1385`, `Zstd-9` `0.1419`, `Zstd-1` `0.1620`, and `LZ4` `0.2500`); **in-cache SQL vs `DuckDB 2.0 + Parquet`** (**`15 / 15` on `8T` (`0.03–0.23 ms`) and `15 / 15` on `1T` (`0.12–1.07 ms`)** with `VXG --record-aligned`; fixed-`64 KiB` `VXG` wins `15 / 15` on `8T` and `11 / 15` on `1T`); and **multi-GB `10M` NDJSON (`5 / 5` `8T` at `1.31–4.31 ms`, `5 / 5` `1T` at `8.10–31.06 ms`) + `40M` CSV (`5 / 5` `8T` at `10.77–25.04 ms`, `4 / 5` `1T` at `80.52–156.39 ms`)** with `--record-aligned`, using pure Rust (`0` C dependencies, `wasm32` compatible).
> - **Where competitors win:** On individual deeply nested JSON (`gharchive`: `0.1072` vs `0.1181`), free-text/HTTP logs (`apache-access.clf`: `0.0896` vs `0.0974`, `macos-install.log`: `0.0452` vs `0.0548`), non-repeating source code (`vxg-src-nonrepeating.rs`: `0.1872` vs `0.1917`, though `VXG` beats `Zstd-9` `0.1939`), prose (`shakespeare`: `0.3729` vs `0.3780`), and camera BMP (`0.7363` vs `0.7985`), `Zstd + Trained Dict L9 [32..256K]` (`FSE + optimal parsing`) leads per-file `64 KiB` ratio; `LZ4 (a1)` leads raw `GB/s` on CSV and unstructured files (`4.95 / 20.52 GB/s` vs `1.52 / 15.35 GB/s` `8T` geomean); and `DuckDB + Parquet-Snappy` leads `1T` `40M` CSV `Top-10` (`29.80` vs `61.76 ms`), `4 / 15` `1T` in-cache queries on fixed-`64 KiB` NDJSON, and fixed-`64 KiB` (non-`--record-aligned`) `10M` NDJSON scans (`6.91–8.83 ms` vs `10.14–35.06 ms`).

VXG `v4/v5` inspects each independent block at runtime and routes it by exact byte budget to `BLOCK_COLUMN` (grammar + typed column models including exception-patched `PFOR-Delta` and multi-level `depth <= 4` nested JSON flattening), `BLOCK_DICT_LZ` (`v5` self-contained frame prefix dictionary + `--level 1..9` lazy/repcode matcher + 5-stream canonical Huffman / 2-way `FSE` `tANS`), `BLOCK_LZ` (2D stride/pixel planes or `Fast-LZ`), or `BLOCK_RAW`. Every block is self-contained, enabling **$O(1)$ random-access block & row extraction** (`FLAG_ROW_INDEX`), **corrupt-frame partial block recovery (`vxg recover`)**, and **multi-core scaling** without cross-block decompression dependencies.

---

## Key Capabilities

| Area | Highlights |
| :--- | :--- |
| **Format v4/v5 Wire Layout & Frozen Compatibility** | Compact 8-byte per-block directory (`payload_size \| kind << 30`, `crc32c`); `v5` adds self-contained frame prefix dictionary (`FLAG_FRAME_DICT`, `<= 256 KiB`), `BLOCK_DICT_LZ` (`kind = 3` with Huffman & `FLAG_FSE = 0x10`), `\n`-snapped / row-bounded `--record-aligned` blocks (`FLAG_VAR_RAW`), and optional per-block row-count index (`FLAG_ROW_INDEX = 0x0008` for $O(1)$ `frame_row_count` and $O(\log B)$ `extract_row`); **100% backward read compatibility with `v1`–`v5` frames** enforced by 18 frozen wire vectors in [`tests/vectors/`](tests/vectors/) |
| **Structured Column & `PFOR-Delta` Engine** | Automatic schema & delimiter discovery for **NDJSON, Multi-Level (`depth <= 4`) Nested JSON (`actor.login`, `payload.pull_request.head.sha`), Pretty-Printed JSON, nullable/headered CSV/TSV/PSV, CLF/Bracketed Logs, and Logfmt**; 12+ models including exception-patched **`PFOR-Delta`** (`0x80 \| base_width` in `DecimalDelta` & `ScaledDecimal` for cumulative time series with jumps), `Dictionary`, `Timestamp` (quoted & unquoted ISO-8601 SWAR+SIMD), `HexSuffix`, `EpochCounter`, `SegmentDictionary`, `FrontCodedDict`, and `Pattern` |
| **Compressed-Domain SQL (`vxg sql`) & Filtered Arrow / DuckDB / Polars Interop** | Store raw NDJSON/CSV verbatim, execute `SELECT col, COUNT(*), SUM/MIN/MAX/AVG(col) ... WHERE ... GROUP BY ... ORDER BY ... LIMIT K` (`vxg sql`, `vxg::sql`, `vxg_sql_alloc` C ABI) with zero-alloc `GroupAccumSoa` fused group buckets, dynamic `Top-K` zone-map/delta pruning, and atomic work-stealing (`0.03–0.23 ms` `8T` / `0.12–1.07 ms` `1T` on `200k–300k` rows; `1.31–4.31 ms` `8T` / `8.10–31.06 ms` `1T` on `10M` NDJSON; `10.77–25.04 ms` `8T` / `61.76–156.39 ms` `1T` on `40M` CSV), **filtered compressed-domain Apache Arrow IPC / C Data / C Stream pushdown (`vxg arrow --where`, `vxg_arrow_query_stream`, `read_vxg()` in `DuckDB` & `Polars`)**, and zero-reencode `VXG -> VXG` column splicing |
| **Image & Stride Engine** | 24-bpp / 32-bpp BMP and multi-stride (`1, 2, 3, 4, 8`) byte-plane transpose + `Subtract-Green` (`R-G`, `B-G`) color decorrelation + 2-level 64-bit hierarchical zero-mask & per-plane `Fast-LZ` |
| **Unstructured Logs, Prose, Parallel `FastCover` & `FSE (tANS)`** | Parallel `std::thread::scope` FastCover + macroscopic SWAR `sketch16` frame dictionary trainer (`<= 256 KiB`, embedded in-frame) + L1D `u16` / register-tagged `PreparedDict` `BLOCK_DICT_LZ` with configurable `--level 1..9` (`Options::level`, default `3`), 3 repcodes, +1/+2 lookahead lazy matching, and per-stream byte-budget competition between canonical Huffman and 2-way interleaved `FSE` (`tANS`, `table_log = 9`) |
| **Cross-Platform SIMD & SWAR** | Runtime-dispatched `NEON` (`aarch64`) and `SSE2` / `AVX2` / `AVX-512` (`x86_64`) fast paths, including 32-byte branchless string scanning, 64-byte `FastPiece` unaligned `u128` row assembly, and SIMD prefix-sum plane interleaving |
| **Integrity, Partial Recovery & CI Ratio Gate** | Hardware-accelerated CRC32C (`ARMv8 CRC` / `SSE4.2`); deterministic atomic rejection of corrupt frames by default (`1,200+` adversarial mutations tested) plus opt-in `vxg::recover_frame` (`vxg recover` / `vxg decompress --recover`) salvaging intact blocks; CI `<= 1%` compression-ratio regression gate |
| **Zero-Copy `mmap` & Streaming Engine** | Zero-dependency cross-platform `MappedFile` / `MappedFileMut`, zero-output-alloc parallel `verify_frame` (`12+ GB/s`, `~64 KiB` L2 scratch/thread), and bounded-memory (`O(single_frame)`) multi-GB streaming across `sql`, `query`, `agg`, `project`, `filter-frame`, `update`, and `map`, plus `staticlib` / `cdylib` C ABI ([`include/vxg.h`](include/vxg.h)) |

---

## 1. Real-World Public, Time-Series & Source Corpora (`64 KiB` Blocks vs. `Zstd+Dict [32..256 KiB]` & `zstd -19 --long`)

Measured sequentially on an uncontested host via `vxg-bench` (`Apple Silicon aarch64 NEON`, `64 KiB` independent blocks, `liblz4 1.10.0`, `libzstd 1.5.7`, checksums off; fetched reproducibly via [`scripts/fetch_real_corpora.py`](scripts/fetch_real_corpora.py)). Every table below is generated deterministically from [`reports/`](reports/) by [`scripts/render_tables.py`](scripts/render_tables.py) and drift-checked in CI.
- **Equal Dictionary Budget (`Zstd + Dict [32..256K]`):** `zstd-dict-l3` and `zstd-dict-l9` sweep **`32 KiB`, `64 KiB`, `112 KiB`, and `256 KiB`** per dataset via `ZDICT_trainFromBuffer` and report the **minimum total size (including `dict_bytes` charged once)**, matching `VXG v5`'s `<= 256 KiB` self-contained frame dictionary budget (`Enc`/`Dec` measure steady-state `ZSTD_CDict`/`ZSTD_DDict` speed, excluding `ZDICT` training; **`VXG v0.5.3` `Enc`/`Dec` include full dictionary training and header decoding**).
- **Whole-file `zstd -19 --long` (`128 MiB` window, single stream, `random access = NO`):** Reference column quantifying the exact compression-ratio cost of `64 KiB` block independence.
- **Reading the tables:** Bold cells mark **only the actual per-row winner** on each metric. Full `1T` & `8T` parameter sweeps: **[`docs/CHIMERA_BENCHMARKS.md`](docs/CHIMERA_BENCHMARKS.md)**, [`reports/real-public-1t.md`](reports/real-public-1t.md), [`reports/real-public-8t.md`](reports/real-public-8t.md), and [`reports/covid-19-and-src-8t.md`](reports/covid-19-and-src-8t.md).

<!-- BEGIN GENERATED TABLE: real-public-summary-8t -->
| Real Public / OS Corpus (`8T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc / Dec (`GB/s`) | LZ4 (`a1`) Enc / Dec | Zstd-1 Enc / Dec | Zstd+D3 Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** *(NYC TLC, 200k trips)* | 52.3 MB | **0.0593** | 0.1899 | 0.0823 | 0.0784 | 0.0554 | **8.84** / **38.59** | 6.36 / 21.98 | 4.46 / 11.51 | 3.49 / 8.62 |
| **`nyc-taxi-2024-01.csv`** *(NYC TLC, 300k trips)* | 23.0 MB | **0.1829** | 0.4241 | 0.2194 | 0.2141 | 0.1617 | 3.22 / 14.66 | **3.36** / **21.08** | 2.45 / 8.71 | 1.74 / 4.51 |
| **`gharchive-2025-01-15.ndjson`** *(12k nested GitHub API)* | 50.3 MB | 0.1181 | 0.2256 | 0.1414 | **0.1072** | 0.0633 | 0.95 / 15.31 | **6.13** / **23.29** | 4.60 / 11.13 | 2.49 / 11.56 |
| **`apache-access.clf`** *(Elastic real HTTP CLF)* | 2.4 MB | 0.0974 | 0.1730 | 0.1062 | **0.0896** | 0.0535 | 0.75 / 12.75 | **5.80** / **13.44** | 4.60 / 8.46 | 2.75 / 9.89 |
| **`macos-install.log`** *(OS free-text multi-line log)* | 34.2 MB | 0.0548 | 0.1249 | 0.0770 | **0.0452** | 0.0278 | 2.34 / 18.04 | 10.10 / **45.50** | **10.30** / 23.11 | 3.58 / 16.52 |
| **`pg100-shakespeare.txt`** *(Gutenberg English prose)* | 5.6 MB | 0.3780 | 0.6223 | 0.3918 | **0.3729** | 0.3000 | 0.26 / 6.57 | **1.91** / **11.32** | 1.84 / 5.19 | 0.92 / 4.05 |
| **`kodak-truecolor-16.bmp`** *(16 real camera photos)* | 18.9 MB | 0.7985 | 0.8732 | 0.7494 | **0.7363** | 0.5656 | 0.39 / 13.73 | **4.90** / **25.62** | 2.94 / 6.67 | 0.63 / 4.78 |
| **Text & Structured Geomean (6)** | — | **0.1171** | 0.2500 | 0.1419 | 0.1181 | 0.0795 | 1.52 / 15.35 | **4.95** / **20.52** | 4.04 / 10.21 | 2.27 / 8.17 |
<!-- END GENERATED TABLE: real-public-summary-8t -->

### Cumulative Numeric Time-Series (`datasets/covid-19`) & Non-Repeating Source Code (`0%` Repeat)

<!-- BEGIN GENERATED TABLE: covid-and-src -->
| Cumulative Time-Series & Non-Repeating Source (`64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | `1T` VXG Enc / Dec (`GB/s`) | `8T` VXG Enc / Dec (`GB/s`) | `8T` LZ4 (`a1`) Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** *(datasets/covid-19 cumulative time series)* | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1495 | 0.1066 | 0.49 / 1.96 | 2.08 / 11.05 | **5.06** / **19.27** |
| **`us_confirmed_33m.csv`** *(datasets/covid-19 US county cumulative prefix)* | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.0938 | 0.0619 | 0.76 / 3.22 | 4.38 / 24.66 | **8.32** / **24.68** |
| **`vxg-src-nonrepeating.rs`** *(All unique VXG src/*.rs files, 0% repeat)* | 2.9 MB | 0.1917 | 0.3333 | 0.2322 | 0.1939 | **0.1872** | 0.1342 | 0.08 / 1.30 | 0.38 / 7.26 | **4.21** / **13.04** |
<!-- END GENERATED TABLE: covid-and-src -->

**Summary of Real-World Trade-Offs:**
- **Flat tabular & cumulative time-series records (`nyc-taxi`, `datasets/covid-19`):** `VXG v0.5.3` achieves the smallest `64 KiB` blocks across every structured table (`1.17×–1.97×` smaller than `Zstd + Trained Dict L9 [32..256K]`). With exception-patched `PFOR-Delta`, `time-series-19-covid-combined.csv` drops from `0.1836` (`v0.5.1` `DICT_LZ` fallback) to **`0.0963`** (`131/131` `BLOCK_COLUMN` — beating `Zstd-9` `0.1415` by `31.9%` and even beating whole-file `zstd -19 --long` `0.1066`!), and `us_confirmed_33m.csv` drops to **`0.0456`** (`1.97×` smaller than `Zstd-9` `0.0897`). On `nyc-taxi.ndjson`, `VXG` beats `LZ4 (a1)` on both `8T` encode (`8.84` vs `6.36 GB/s`) and decode (`38.59` vs `21.98 GB/s`).
- **Unstructured logs, nested JSON, prose & non-repeating source code:** `v0.5.3` adds parallel `FastCover` dictionary training, 2-way interleaved `FSE` (`tANS`) stream competition (`FLAG_FSE = 0x10`) with a 2-pair 64-bit register backward bit-reader, and multi-level (`depth <= 4`) nested JSON flattening (`apache-access.clf` `0.1037 -> 0.0974` at default `-l 3`, `0.0932` at `-l 9`, beating `v0.5.0`'s `0.0990`; `gharchive` `0.1231 -> 0.1181`; `macos-install.log` `0.0594 -> 0.0548`; `shakespeare` `0.3884 -> 0.3780`), pushing the **6-dataset real geomean to `0.1171`** — **ahead of `Zstd+Dict-9 [32..256K]` (`0.1181`)** — at **`15.35 GB/s` `8T` decode** (`1.88×` faster than `Zstd+Dict-3` `8.17 GB/s`). On non-repeating `vxg-src-nonrepeating.rs` (`2.9 MB`), `VXG` (`0.1917`) beats `Zstd-9` (`0.1939`), while `Zstd+Dict-9 [32..256K]` (`0.1872`) leads by `2.3%`, and `LZ4` leads raw `GB/s` on unstructured inputs.

---

## 2. In-Cache Analytical Queries (`200k–300k` Rows, `1T` & `8T`): `VXG v0.5.3` vs. `DuckDB 2.0.0 + Parquet`

Measured standalone with zero background CPU contention (`libvxg.dylib` vs `duckdb 2.0.0`, warm OS cache, median of 5–7 runs, **100% result equality verified** via [`scripts/bench_vs_duckdb.py`](scripts/bench_vs_duckdb.py); raw artifact: [`reports/duckdb-vs-vxg.json`](reports/duckdb-vs-vxg.json)). Both **`1T` and `8T` latencies (`1T / 8T` in `ms`)** are shown side-by-side, and `DuckDB + Parquet` is tested at both default `ROW_GROUP_SIZE = 122880` and fine-grained `ROW_GROUP_SIZE = 16384` (`SELECT 1` `PREPARE/EXECUTE` overhead = `0.04 ms` `1T`, `0.09 ms` `8T`). With `v0.5.3` zero-alloc `GroupAccumSoa` fused group buckets, dynamic `Top-K` zone-map/delta pruning, and atomic work-stealing, **`VXG --record-aligned` wins `15 / 15` on `8T` (`0.03–0.23 ms`) and `15 / 15` on `1T` (`0.12–1.07 ms`)**, while fixed-`64 KiB` `VXG` wins `15 / 15` on `8T` and `11 / 15` on `1T`.

<!-- BEGIN GENERATED TABLE: duckdb-in-cache -->
| Corpus & Query (`1T / 8T` latency in `ms`, lower is faster) | `VXG` (`64K` std) | `VXG --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* | DuckDB Raw |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** *(52.3 MB, 200k rows)* | 3.10 MB (`6.8 ms`, YES) | **2.86 MB** (`29.4 ms`, YES) | 3.94 MB (`83.6 ms` ETL, NO) | 4.17 MB (`75.7 ms`, NO) | 2.96 MB (`92.0 ms` ETL, NO) | `52.33 MB` (YES) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 2.46 / 0.61 | **0.53** / **0.15** | 1.69 / 1.13 | 2.38 / 1.57 | 2.30 / 1.37 | 81.27 / 57.39 |
| ↳ `SUM/MIN/MAX(total_cents) [all 200k rows]` | 1.35 / 0.47 | **0.15** / **0.10** | 2.11 / 0.91 | 1.49 / 0.93 | 1.77 / 1.15 | 81.30 / 56.10 |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 1.73 / 0.44 | **0.28** / **0.09** | 1.18 / 0.98 | 1.47 / 0.91 | 1.46 / 1.22 | 81.51 / 57.32 |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | 2.08 / 0.49 | **0.52** / **0.10** | 1.22 / 0.98 | 1.56 / 0.86 | 1.57 / 1.14 | 84.65 / 55.83 |
| ↳ `ORDER BY total_cents DESC LIMIT 10` | 1.61 / 0.87 | **0.46** / **0.18** | 2.56 / 2.16 | 2.47 / 2.33 | 4.59 / 3.41 | 92.09 / 58.27 |
| ↳ `SELECT 4 cols WHERE tip_cents >= 2500` | 2.08 / 0.92 | **1.07** / **0.23** | 3.22 / 3.11 | 3.61 / 3.57 | 5.55 / 5.28 | 94.70 / 59.56 |
| **`nyc-taxi-2024-01.csv`** *(23.0 MB, 300k rows)* | 4.21 MB (`9.1 ms`, YES) | **4.19 MB** (`27.0 ms`, YES) | 5.86 MB (`105.3 ms` ETL, NO) | 6.19 MB (`88.9 ms`, NO) | 4.39 MB (`97.7 ms` ETL, NO) | `23.00 MB` (YES) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 0.90 / 0.19 | **0.80** / **0.13** | 1.92 / 1.21 | 2.64 / 1.27 | 2.80 / 1.57 | 77.86 / 51.57 |
| ↳ `SUM/MIN/MAX(total_cents) [all 300k rows]` | 0.40 / 0.18 | **0.35** / **0.16** | 1.45 / 0.97 | 1.81 / 0.92 | 1.91 / 1.10 | 79.03 / 50.90 |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 0.60 / 0.17 | **0.52** / **0.09** | 1.53 / 1.15 | 1.94 / 1.09 | 1.95 / 1.33 | 79.73 / 50.40 |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | 1.06 / 0.25 | **0.91** / **0.21** | 1.62 / 1.04 | 2.13 / 1.17 | 2.09 / 1.22 | 77.72 / 50.85 |
| **`ndjson_varied.ndjson`** *(56.5 MB, 300k rows)* | 5.17 MB (`8.8 ms`, YES) | **4.70 MB** (`41.6 ms`, YES) | 9.70 MB (`116.4 ms` ETL, NO) | 9.72 MB (`102.7 ms`, NO) | 5.56 MB (`121.0 ms` ETL, NO) | `56.50 MB` (YES) |
| ↳ `COUNT(*) WHERE fraud & status=500 & lat>=150` | 2.79 / 0.61 | **0.89** / **0.21** | 1.23 / 0.92 | 1.74 / 1.06 | 1.43 / 1.01 | 99.73 / 62.44 |
| ↳ `SUM/MIN/MAX(bytes) [all 300k rows]` | 1.17 / 0.32 | **0.12** / **0.03** | 2.35 / 1.24 | 2.57 / 0.94 | 3.61 / 1.80 | 88.87 / 58.29 |
| ↳ `GROUP BY service: COUNT(*), SUM(bytes)` | 2.60 / 0.54 | **0.83** / **0.16** | 2.97 / 1.73 | 3.31 / 1.57 | 4.25 / 2.25 | 100.75 / 62.54 |
| ↳ `ORDER BY latency_ms DESC LIMIT 10` | 1.65 / 0.81 | **0.44** / **0.17** | 3.15 / 2.06 | 2.55 / 1.59 | 3.88 / 2.38 | 101.63 / 62.16 |
| ↳ `SELECT 3 cols WHERE fraud & 500 & lat>=150` | 2.71 / 0.86 | **0.95** / **0.20** | 7.43 / 6.89 | 5.04 / 4.78 | 9.42 / 8.95 | 104.52 / 63.29 |
<!-- END GENERATED TABLE: duckdb-in-cache -->

---

## 3. Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Real 2024 NYC TLC Rows, `1T` & `8T`)

Measured over **all 12 months of unmodified 2024 NYC TLC Yellow Taxi trip data** (`41.17M` official rows; [`scripts/bench_scale_duckdb.py`](scripts/bench_scale_duckdb.py), [`reports/scale-duckdb-vs-vxg.md`](reports/scale-duckdb-vs-vxg.md), **100% numerical equality & lossless `vxg_verify_file` verified**):
- **`nyc-taxi-2024-40m.csv` (`40,000,000` rows, `3.06 GB` raw):** `VXG --record-aligned` (**`550.4 MB`**, **`2.93 s`** `8T` ingest, `lossless = YES`) is **smaller and `1.2×–1.7×` faster to ingest** than both `Parquet-Zstd` (`579.4 MB`, `3.48 s`, `lossless = NO`) and `Parquet-Snappy` (`773.1 MB`, `4.87 s`, `lossless = NO`), winning **`5 / 5` `8T` queries** (`10.77–25.04 ms` vs `Parquet-Snappy` `20.47–35.04 ms`; fixed-`64K` `VXG` wins `8T` `COUNT WHERE` at `23.54 ms`) and **`4 / 5` `1T` queries** (`80.52–156.39 ms` vs `133.96–172.73 ms`; `Parquet-Snappy` leads `1T` `Top-10` `29.80` vs `61.76 ms`).
- **`nyc-taxi-2024-10m.ndjson` (`10,000,000` rows, `2.62 GB` raw):** With thread-local JSON template/key memoization, zero-alloc `GroupAccumSoa` fused group buckets, dynamic `Top-K` zone-map/delta pruning, and row-bounded `--record-aligned` blocks (`FLAG_VAR_RAW`, bounded to `<= 1 MiB`), `VXG --record-aligned` (**`140.8 MB`**, **`0.61 s`** `8T` ingest, `lossless = YES`) is **smaller and faster to ingest than both `Parquet-Snappy` (`193.3 MB`, `0.77 s`) and `Parquet-Zstd` (`144.7 MB`, `0.95 s`)**, winning **`5 / 5` `8T` queries (`1.31–4.31 ms` vs `6.91–8.83 ms`, up to `7,642.3 M rows/s`)** and **`5 / 5` `1T` queries (`8.10–31.06 ms` vs `16.20–43.67 ms`)**. Fixed-`64 KiB` `VXG` (`~40,000` blocks of `~245` verbose JSON rows) remains slower than `Parquet` (`10.14–35.06 ms` `8T`), confirming that block count—rather than row count—governs verbose NDJSON scan overhead ([`reports/ndjson-scale-diagnostics.json`](reports/ndjson-scale-diagnostics.json)).

<!-- BEGIN GENERATED TABLE: duckdb-scale-40m-csv -->
| `nyc-taxi-2024-40m.csv` (`40M` rows, `3.06 GB` raw; `1T / 8T` in `ms`) | `VXG` (`64K` std) | `VXG --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Archive Size & `8T` Ingest / ETL** | 552.9 MB (`5.13 s`, YES) | **550.4 MB** (`2.93 s`, YES) | 773.1 MB (`4.87 s`, NO) | 818.8 MB (`4.74 s`, NO) | 579.4 MB (`3.48 s`, NO) |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 159.37 / **23.54** | **150.02** / 25.04 | 159.89 / 34.32 | 280.70 / 98.53 | 274.23 / 43.63 |
| `SUM/MIN/MAX(total_cents) [full scan]` | 90.08 / 15.02 | **80.52** / **13.10** | 148.07 / 25.85 | 214.38 / 73.95 | 204.03 / 36.20 |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 114.03 / 17.90 | **106.53** / **16.67** | 133.96 / 35.04 | 198.04 / 64.17 | 192.40 / 32.22 |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 174.01 / 37.30 | **156.39** / **23.25** | 172.73 / 32.85 | 258.14 / 76.95 | 233.14 / 40.84 |
| `ORDER BY total_cents DESC LIMIT 10` | 87.94 / 20.57 | 61.76 / **10.77** | **29.80** / 20.47 | 84.93 / 78.73 | 45.11 / 24.20 |
<!-- END GENERATED TABLE: duckdb-scale-40m-csv -->

<!-- BEGIN GENERATED TABLE: duckdb-scale-10m-ndjson -->
| `nyc-taxi-2024-10m.ndjson` (`10M` rows, `2.62 GB` raw; `1T / 8T` in `ms`) | `VXG` (`64K` std) | `VXG --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Archive Size & `8T` Ingest / ETL** | 152.6 MB (`2.65 s`, YES) | **140.8 MB** (`0.61 s`, YES) | 193.3 MB (`0.77 s`, NO) | 203.4 MB (`1.05 s`, NO) | 144.7 MB (`0.95 s`, NO) |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 127.03 / 21.87 | **29.18** / **3.93** | 40.07 / 8.83 | 65.78 / 23.17 | 68.72 / 12.08 |
| `SUM/MIN/MAX(total_cents) [full scan]` | 70.45 / 10.14 | **11.20** / **1.31** | 37.80 / 6.91 | 53.66 / 15.15 | 52.23 / 8.97 |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 91.13 / 12.60 | **15.93** / **2.09** | 35.05 / 6.94 | 50.69 / 16.19 | 49.93 / 9.37 |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 108.19 / 17.14 | **31.06** / **4.31** | 43.67 / 8.83 | 65.46 / 18.86 | 59.83 / 10.06 |
| `ORDER BY total_cents DESC LIMIT 10` | 84.93 / 35.06 | **8.10** / **1.63** | 16.20 / 8.79 | 22.68 / 20.79 | 26.17 / 11.78 |
<!-- END GENERATED TABLE: duckdb-scale-10m-ndjson -->

---

## 4. Synthetic & Procedural Qualification Suites (`--suite v4` & Multi-Template Generators)

> **Important (Synthetic Provenance & Periodic Seed Disclosure):** All datasets below (`--suite v4`, `scripts/gen_varied_ndjson.py`, `scripts/gen_format_bench.py`) are **synthetic / procedurally generated** qualification workloads (`reports/synthetic-v4-{1t,8t}.md`, `reports/synthetic-multitpl-{1t,8t}.md`). In particular:
> - **`source-and-docs.txt` in `--suite v4` (`0.0036` in `v5`, `0.0029` in whole-file `zstd -19 --long`)** repeats a `~190 KiB` 10-file source/docs seed to `16 MiB` via `fill_repeated` (`src/bin/bench_support/datasets.rs`), so the entire periodic seed fits inside `v5`'s `<= 256 KiB` frame dictionary. For **non-repeating real source code**, see **`vxg-src-nonrepeating.rs` (`0.1917`)** in Section 1 (`--suite v4-nonrep` in `vxg-bench` also replaces `source-and-docs.txt` with non-repeating workspace source).
> - The two `.bmp` files in `--suite v4` are procedural flat-shaded UI/gradient framebuffers (`208:1`–`833:1`) and are separated from the 8 text/structured datasets below; for real optical camera photographs, see `kodak-truecolor-16.bmp` (`0.7985`) in Section 1.

| Synthetic Suite (`64 KiB` blocks) | **VXG v0.5 Ratio** | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | `1T` VXG vs LZ4 (`Enc / Dec GB/s`) | `8T` VXG vs LZ4 (`a1`) (`Enc / Dec GB/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v4` Text & Structured Geomean (8 sets, `16 MiB`, incl. periodic `source-and-docs`)** | **0.0259** | 0.2107 | 0.0859 | 0.0846 | 0.0081 | **`1.58` / `9.69`** vs `1.51` / `5.79` | `6.34` / **`50.84`** vs **`8.17`** / `28.13` |
| **`v4` Procedural UI BMP Geomean (2 sets, `16 MiB`)** | **0.0024** | 0.0178 | 0.0103 | 0.0111 | 0.0010 | `8.69` / **`14.69`** vs **`11.64`** / `14.30` | `27.61` / **`75.11`** vs **`35.50`** / `54.52` |
| **Synthetic Multi-Template (`ndjson_varied`, `56.5 MB`)** | **0.0914** | 0.2898 | 0.1480 | 0.1452 | 0.1198 | `1.04` / **`5.65`** vs **`1.17`** / `4.57` | **`6.20` / `40.29`** vs `5.98` / `25.44` |
| **Synthetic Delimited (`data.csv` / `data.clf`, `31 MB`)** | **0.1794 / 0.0980** | 0.4458 / 0.2843 | 0.2651 / 0.1504 | 0.2606 / 0.1454 | 0.2257 / 0.1225 | `0.67 / 2.36` · `0.61 / 2.53` (`LZ4` faster) | `3.78/14.73` · `3.61/`**`17.92`** (`LZ4 a1`: `17.25`) |

---

## Historical Cross-Platform CI Baseline — VXG `v0.4.1` Pre-`v5` (`--suite v4` Synthetic Suite on `macos-14`, `ubuntu-latest`, `windows-latest`)

> **Version Note:** The three tables below archive the **`v0.4.1` (`Format v4`, pre-`FLAG_FRAME_DICT` / pre-`BLOCK_DICT_LZ`)** single-thread GitHub Actions baseline ([`reports/ci-v0.4.1/`](reports/ci-v0.4.1/)). In `v0.4.1`, `source-and-docs.txt` compressed to `0.2678–0.2712` via `64 KiB` block-local `FAST_LZ` because cross-block frame dictionaries (`Format v5`) did not yet exist.

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

## 5. Remaining Trade-Offs (`v0.5.2`) & Future Roadmap ([`docs/ROADMAP.md`](docs/ROADMAP.md))

1. **Single-Thread `GROUP BY` & Fixed-`64 KiB` Verbose NDJSON Scans vs. `DuckDB + Parquet`:** With `v0.5.2`'s thread-local JSON template memoization and row-bounded `--record-aligned` (`FLAG_VAR_RAW`, bounded to `<= 1 MiB`), `VXG --record-aligned` now beats `DuckDB + Parquet` on `5 / 5` `8T` and `4 / 5` `1T` `10M` NDJSON queries (`1.79–6.77 ms` `8T`), whereas fixed-`64 KiB` `VXG` (`~40,000` blocks of `~245` rows) still pays per-block header/column setup (`10.50–31.46 ms` `8T`), and `DuckDB + Parquet-Snappy` leads `1T` `GROUP BY` (`42.09` vs `51.07 ms` on `10M` NDJSON; `174.81` vs `242.90 ms` on `40M` CSV) and `1T` `40M` CSV `Top-10` (`26.95` vs `86.68 ms`).
2. **Deeply Nested (`>= 2`-Level) JSON, Free-Text Logs & Optimal-Parsed FSE:** `v0.5.2` leads the 6-dataset real public geomean (`0.1172` vs `Zstd+Dict-9 [32..256K]` `0.1181`), but `Zstd + Trained Dict L9` (`FSE + optimal parsing`) still leads individual `64 KiB` ratios on deeply nested JSON (`gharchive`: `0.1072` vs `0.1181`), HTTP/system logs (`0.0896` vs `0.0976`, `0.0452` vs `0.0548`), non-repeating source code (`0.1889` vs `0.1933`), and prose (`0.3729` vs `0.3786`), while `LZ4` leads raw `GB/s` on unstructured text. Multi-level array shredding and finite-state entropy (`FSE`) are tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Quick Start & CLI Usage (`vxg`)

Requires **Rust 1.89+**.

```sh
# Build CLI, staticlib (libvxg.a), and cdylib (libvxg.dylib / .so / .dll):
cargo build --release

# Inspect detected CPU SIMD & GPU backends:
./target/release/vxg hardware

# Compress and decompress (--record-aligned snaps blocks to \n; --level 1..9 tunes DICT_LZ):
./target/release/vxg compress input.ndjson input.vxg --record-aligned --level 3 --threads auto
./target/release/vxg decompress input.vxg restored.ndjson --threads auto --force

# Run SQL directly on compressed .vxg files (supports CSV header names & dotted nested JSON keys):
./target/release/vxg sql input.vxg \
  "SELECT service, COUNT(*), SUM(bytes), MAX(latency_ms) WHERE status = 500 GROUP BY service ORDER BY SUM(bytes) DESC LIMIT 10" \
  --format json -t auto

# Export projected compressed columns directly to zero-dependency Apache Arrow IPC Stream (.arrows):
./target/release/vxg arrow input.vxg projected.arrows --cols service,status,bytes,latency_ms -t auto -f

# Verify frame structure or salvage intact blocks from a corrupted/truncated .vxg archive:
./target/release/vxg verify input.vxg --threads auto
./target/release/vxg recover damaged.vxg salvaged.ndjson --report -f

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
  --suite v4-nonrep --size-mib 16 --samples 3 --seconds 0.05 \
  --csv reports/local-v4-nonrep.csv --markdown reports/local-v4-nonrep.md
```

---

## Using VXG from Other Languages (Static & Dynamic C ABI + Arrow / DuckDB)

VXG compiles out of the box into both a **static archive** (`libvxg.a` / `vxg.lib`) and a **dynamic shared library** (`libvxg.dylib` / `libvxg.so` / `vxg.dll`), exposing a zero-allocation caller-buffer C ABI plus Apache Arrow C Data (`ArrowSchema` / `ArrowArray`) and C Stream (`ArrowArrayStream`) exports in [`include/vxg.h`](include/vxg.h). See **[`docs/FFI.md`](docs/FFI.md)** for full API documentation.

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

// Compressed-domain SQL, Arrow export, structured record random access & projection:
let sql_q = vxg::sql::SqlQuery::parse("SELECT status, COUNT(*), SUM(seq) WHERE status = 'ok' GROUP BY status")?;
let tsv = vxg::sql::execute_sql(&frame, &sql_q, vxg::sql::SqlFormat::Tsv, true, &dec_opts)?;
let cols = [vxg::ColTarget::JsonKey("ts"), vxg::ColTarget::JsonKey("seq")];
let arrow_ipc = vxg::export_arrow_ipc(&frame, &cols, &dec_opts)?;
let total_rows = vxg::frame_row_count(&frame)?;
let row0 = vxg::extract_row(&frame, 0, &dec_opts)?;
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

### Python Zero-Copy `ctypes` & Zero-Dependency `DuckDB` `read_vxg()` Bridge ([`examples/ffi/python_example.py`](examples/ffi/python_example.py), [`examples/ffi/duckdb_read_vxg.py`](examples/ffi/duckdb_read_vxg.py))
```sh
python3 examples/ffi/python_example.py
python3 examples/ffi/duckdb_read_vxg.py   # Zero-pyarrow Arrow C Stream PyCapsule scan in DuckDB 2.0+
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
src/                     Rust codec core (lib.rs, encode.rs, arrow*.rs, stream.rs, simd.rs, crc32c.rs)
src/columns/             Modular structured-column, PFOR-Delta, ScaledDecimal, query/splice, stride/BMP & Fast-LZ
src/dict_lz/             Format v5 pure-Rust FastCover frame dictionary trainer & `--level 1..9` `BLOCK_DICT_LZ`
src/sql/                 Zero-dependency SQL parser & compressed-domain execution engine (`vxg sql`)
src/query/               Compressed-domain row seek, bigram/nibble search, and parallel pushdown
src/ops/                 Compressed-domain compound WHERE, agg/group_by, filter/map/update & column projection
src/frame/               Format v1–v5 frame encoder, parser, `var_raw` splitter, `recover.rs` & parallel decoder
src/block/               Block LZ encoder and SIMD/SWAR decoder submodules
src/packing/             Command-metadata & entropy research submodules (--features bench)
src/groups/              Fixed-group command-layout evaluator submodules (--features bench)
src/bin/                 Production CLI (`vxg`) and benchmark harnesses (`vxg-bench`, `vxg-query-real`)
include/vxg.h            Official C99 / C++11 header (codec, SQL, and Arrow C Data / C Stream ABI)
examples/ffi/            Zero-copy integration examples for C, Python, DuckDB/Arrow, Go, Bun/JS, C#, and WASM
native/shaders/core.gpu  Shared GPU kernel source (Metal, Vulkan GLSL, CUDA)
native/                  Metal, Vulkan, and CUDA host bridges
verification/            Independent Python reference model and C++ SIMT emulator
crates/vxg-tensor/       Compressed-domain AI weight codec (.vxt), CB4-Fast Metal LLM/VLM (3D M-RoPE) & OpenAI SSE server
apps/vxt-studio/         Tauri v2 + React + shadcn/ui + AI SDK desktop app (Multimodal Vision Chat, Tensor Forge & API Hub)
tests/                   Property, stress, differential SQL/Arrow/recovery, CI ratio gate & frozen `tests/vectors/`
reports/                 Current benchmark artifacts (`real-public-*`, `covid-19-and-src-*`, `duckdb-*`, `ci-v0.4.1/`)
archive/                 Archived historical milestone plans and early microbenchmark logs
docs/                    Wire format, tensor engine, FFI guide, proofs, benchmarks, GPU architecture, and safety
```

[Wire Format](docs/FORMAT.md) · [VXT Tensor & Metal LLM Engine](docs/TENSOR.md) · [FFI & Language Bindings](docs/FFI.md) · [Benchmarks](docs/BENCHMARKS.md) · [Proofs](docs/PROOFS.md) · [GPU Architecture](docs/GPU.md) · [Safety](docs/SAFETY.md) · [Validation](docs/VALIDATION.md)

---

## License

MIT License.
