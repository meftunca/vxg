# Competitor Parameter-Sweep, Trained-Dictionary, Whole-File & DuckDB Scale Benchmarks (`v0.5.3`)

This document evaluates `VXG v0.5.3` (`Format v4/v5`) under four strict methodology rules required for an honest, reproducible evaluation of a random-access structured codec and SQL engine. Every real-corpus and DuckDB table in this document is generated deterministically from [`reports/`](../reports/) by [`scripts/render_tables.py`](../scripts/render_tables.py) and verified against drift in CI.

1. **Strict Separation of Real-World Public Corpora vs. Synthetic/Procedural Corpora:** Publicly downloadable operational datasets (`NYC TLC Yellow Taxi`, `datasets/covid-19` cumulative time series, `GitHub Archive`, `Elastic Apache Access Log`, `Project Gutenberg Shakespeare`, `Kodak Lossless True Color` camera photographs, non-repeating `src/` Rust source code, and macOS `/var/log/install.log`) are reported separately from synthetic/procedural qualification generators (`--suite v4`, `scripts/gen_varied_ndjson.py`, `scripts/gen_format_bench.py`, and procedural 2D UI `.bmp` renders).
2. **Equalized Trained-Dictionary (`Zstd + Dict [32..256 KiB]`) & Whole-File (`zstd -19 --long`) Baselines:**
   - **Block-independent codecs (`64 KiB` blocks, `O(1)` random access = `YES`):** Native C `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`, **`zstd-dict-l3`**, **`zstd-dict-l9`**).
   - **Equal dictionary budget (`32 / 64 / 112 / 256 KiB` sweep):** `zstd-dict-*` trains across **`32 KiB`, `64 KiB`, `112 KiB`, and `256 KiB`** per dataset via `ZDICT_trainFromBuffer` (`src/baselines/zstd_ext.rs`) and reports the **best-scoring dictionary budget** per dataset (charging `dict_bytes` once in `encoded_size()`), matching `VXG v5`'s `<= 256 KiB` frame dictionary budget. `Enc` / `Dec` measure steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput (excluding `ZDICT` training), whereas **`VXG v5`** embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself (`FLAG_FRAME_DICT`), and **`VXG Enc` / `Dec` include full dictionary training and header decoding time**.
   - **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window / `windowLog = 27`, single stream) removes `64 KiB` block boundaries completely to quantify the exact compression-ratio cost of `64 KiB` random access.
3. **Head-to-Head Analytical Queries vs. `DuckDB 2.0.0` + `Parquet` (`1T` & `8T`, `ROW_GROUP_SIZE = 122880` & `16384`, In-Cache `200k–300k` & Out-of-Cache `10M–40M` Rows):** Compressed-domain queries are reported in **wall-clock latency (`ms`)** (`1T / 8T` side-by-side) — never by dividing raw file size by single-column scan time — calibrated with `SELECT 1` and `PREPARE / EXECUTE` (`prep`) statements, testing both DuckDB's default `ROW_GROUP_SIZE = 122880` (`~16–30 MiB` raw/group) and `ROW_GROUP_SIZE = 16384` (matching finer random-access granularity), alongside compressed storage (`MB`), ingest/ETL time, and **byte-exact lossless reversibility**.
4. **Honest Per-Row Bolding:** Bold cells mark **only the actual per-row winner** on each individual metric (lowest `64 KiB` block ratio, highest encode `GB/s`, highest decode `GB/s`, lowest `1T` query `ms`, lowest `8T` query `ms`). Codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` compression) are excluded from speed bolding.

---

## 1. Real-World Public, Time-Series & Source Corpora (`64 KiB` Independent Blocks)

Fetched reproducibly via [`scripts/fetch_real_corpora.py`](../scripts/fetch_real_corpora.py). Raw benchmark artifacts: [`reports/real-public-1t.md`](../reports/real-public-1t.md), [`reports/real-public-1t.csv`](../reports/real-public-1t.csv), [`reports/real-public-8t.md`](../reports/real-public-8t.md), [`reports/real-public-8t.csv`](../reports/real-public-8t.csv), [`reports/covid-19-and-src-1t.md`](../reports/covid-19-and-src-1t.md), and [`reports/covid-19-and-src-8t.md`](../reports/covid-19-and-src-8t.md).

### 1.1 Corpus Provenance

| Corpus | Size | Description & Source |
| :--- | :---: | :--- |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | First `300,000` real NYC TLC Yellow Taxi Jan 2024 trip records (`13` tabular columns, public CloudFront Parquet source exported verbatim to CSV) |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | First `200,000` real NYC TLC Yellow Taxi Jan 2024 trip records serialized as flat NDJSON |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | Unmodified [`datasets/covid-19`](https://github.com/datasets/covid-19) global country/province cumulative time-series CSV (`231,745` rows, nullable fields, large country-boundary jumps) |
| **`us_confirmed_33m.csv`** | 35.2 MB | First `33.6 MiB` of [`datasets/covid-19`](https://github.com/datasets/covid-19) `us_confirmed.csv` (`US` county-level daily cumulative counts) |
| **`vxg-src-nonrepeating.rs`** | 2.9 MB | Concatenation of every unique `.rs` file in `VXG`'s `src/` tree (`0%` repeated files; contrasts with `--suite v4`'s periodic `source-and-docs.txt` seed) |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | `12,000` unmodified GitHub Archive (`2025-01-15-15`) public API events (deeply nested, heterogeneous JSON trees + Markdown commit/PR bodies) |
| **`apache-access.clf`** | 2.4 MB | Unmodified Elastic `examples` real Apache HTTP Combined Log Format corpus |
| **`macos-install.log`** | 34.2 MB | Unmodified macOS `/var/log/install.log` multi-line free-text system log |
| **`pg100-shakespeare.txt`** | 5.6 MB | Unmodified Project Gutenberg `#100` (*Complete Works of William Shakespeare*, English prose/verse) |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | Uncompressed `3072×2048` 24-bpp mosaic of 16 real optical camera photographs (`kodim01`–`kodim16`, Kodak Lossless True Color Image Suite) |

### 1.2 Single-Thread Results (`threads = 1`, `block = 65536`, Apple Silicon NEON)

<!-- BEGIN GENERATED TABLE: real-public-full-1t -->
| Dataset (`1T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **1.36** | 1.33 | 1.02 | 0.55 | **6.26** | 4.47 | 2.11 | 1.65 |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 0.53 | **0.61** | 0.48 | 0.25 | 2.14 | **3.05** | 1.28 | 0.86 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.14 | **1.21** | 0.95 | 0.35 | 1.99 | **5.40** | 2.29 | 2.05 |
| **`apache-access.clf`** | 2.4 MB | 0.0974 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.18 | **1.49** | 1.13 | 0.56 | 1.87 | **5.88** | 2.62 | 2.14 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0277 | 0.31 | **2.09** | 1.44 | 0.76 | 2.49 | **7.29** | 3.36 | 3.41 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3780 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.04 | 0.38 | **0.39** | 0.16 | 0.99 | **2.82** | 1.13 | 0.71 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7985 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.05 | **0.83** | 0.52 | 0.13 | 1.83 | **4.08** | 1.03 | 0.97 |
| **Text & Structured Geomean (6)** | — | **0.1171** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 0.25 | **1.02** | 0.82 | 0.38 | 2.23 | **4.55** | 1.99 | 1.57 |
<!-- END GENERATED TABLE: real-public-full-1t -->

### 1.3 Multi-Thread Results (`threads = 8`, `block = 65536`, Apple Silicon NEON)

<!-- BEGIN GENERATED TABLE: real-public-full-8t -->
| Dataset (`8T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **8.84** | 6.36 | 4.46 | 3.49 | **38.59** | 21.98 | 11.51 | 8.62 |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 3.22 | **3.36** | 2.45 | 1.74 | 14.66 | **21.08** | 8.71 | 4.51 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.95 | **6.13** | 4.60 | 2.49 | 15.31 | **23.29** | 11.13 | 11.56 |
| **`apache-access.clf`** | 2.4 MB | 0.0974 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.75 | **5.80** | 4.60 | 2.75 | 12.75 | **13.44** | 8.46 | 9.89 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0278 | 2.34 | 10.10 | **10.30** | 3.58 | 18.04 | **45.50** | 23.11 | 16.52 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3780 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.26 | **1.91** | 1.84 | 0.92 | 6.57 | **11.32** | 5.19 | 4.05 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7985 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.39 | **4.90** | 2.94 | 0.63 | 13.73 | **25.62** | 6.67 | 4.78 |
| **Text & Structured Geomean (6)** | — | **0.1171** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 1.52 | **4.95** | 4.04 | 2.27 | 15.35 | **20.52** | 10.21 | 8.17 |
<!-- END GENERATED TABLE: real-public-full-8t -->

### 1.4 Cumulative Numeric Time-Series (`datasets/covid-19`) & Non-Repeating Source Code (`64 KiB` Blocks)

<!-- BEGIN GENERATED TABLE: covid-and-src -->
| Cumulative Time-Series & Non-Repeating Source (`64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | `1T` VXG Enc / Dec (`GB/s`) | `8T` VXG Enc / Dec (`GB/s`) | `8T` LZ4 (`a1`) Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** *(datasets/covid-19 cumulative time series)* | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1495 | 0.1066 | 0.49 / 1.96 | 2.08 / 11.05 | **5.06** / **19.27** |
| **`us_confirmed_33m.csv`** *(datasets/covid-19 US county cumulative prefix)* | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.0938 | 0.0619 | 0.76 / 3.22 | 4.38 / 24.66 | **8.32** / **24.68** |
| **`vxg-src-nonrepeating.rs`** *(All unique VXG src/*.rs files, 0% repeat)* | 2.9 MB | 0.1917 | 0.3333 | 0.2322 | 0.1939 | **0.1872** | 0.1342 | 0.08 / 1.30 | 0.38 / 7.26 | **4.21** / **13.04** |
<!-- END GENERATED TABLE: covid-and-src -->

### 1.5 Engineering Takeaways on Real-World Data

1. **Flat Tabular & Cumulative Time-Series Records (`nyc-taxi`, `datasets/covid-19`):**
   - On **`nyc-taxi-2024-01.ndjson`** (`52.3 MB`), `VXG v0.5.3` achieves **`0.0593` (`5.93%`)** — **`3.20×` smaller than `LZ4` (`0.1899`)**, **`1.39×` smaller than `Zstd-9` (`0.0823`)**, **`1.32×` smaller than `Zstd + Trained Dict L9 [32..256K]` (`0.0784`)**, and within **`7.0%` of non-random-access whole-file `zstd -19 --long` (`0.0554`)**, while encoding at **`8.84 GB/s`** and decoding at **`38.59 GB/s`** (`8T`) — faster than `LZ4 (a1)` (`6.36 / 21.98 GB/s`).
   - On **`time-series-19-covid-combined.csv`** (`8.6 MB`) and **`us_confirmed_33m.csv`** (`35.2 MB`), `VXG`'s exception-patched row-delta model (`PFOR-Delta`, `0x80 | base_width` in `MODEL_DECIMAL_DELTA` and `MODEL_SCALED_DECIMAL`) plus nullable CSV field support move `131 / 131` blocks from `BLOCK_DICT_LZ` (`0.1836` in `v0.5.1`) to `BLOCK_COLUMN` at **`0.0963`** (`31.9%` smaller than `Zstd-9` `0.1415`, and smaller than whole-file `zstd -19 --long` `0.1066`!) and **`0.0456`** on `us_confirmed` (`1.97×` smaller than `Zstd-9` `0.0897`).
2. **Unstructured Logs, Nested JSON, Prose & Non-Repeating Source (`BLOCK_DICT_LZ` + `FSE` in `Format v5`):**
   - `v0.5.3` combines a `4096`-entry `16`-bit full-tag local 4-byte match ring + `+1/+2` lazy lookahead (`--level 1..9`, default `3`), parallel `FastCover` dictionary training, 2-way interleaved `FSE` (`tANS`, `FLAG_FSE = 0x10`) with a 2-pair 64-bit register backward bit-reader, and multi-level (`depth <= 4`) nested JSON flattening: `apache-access.clf` reaches `0.0974` (`0.0932` at `-l 9`, beating `v0.5.0`'s `0.0990`), `gharchive` reaches `0.1181`, `macos-install.log` reaches `0.0548`, `pg100-shakespeare.txt` reaches `0.3780`, and non-repeating `vxg-src-nonrepeating.rs` (`2.9 MB`) improves to **`0.1917`** (beating `Zstd-9` `0.1939`).
   - Across the **6-dataset real public geomean**, **`VXG v0.5.3` (`0.1171`) beats `Zstd+Dict-9 [32..256K]` (`0.1181`)**, `Zstd+Dict-3` (`0.1385`), `Zstd-9` (`0.1419`), `Zstd-1` (`0.1620`), and `LZ4` (`0.2500`) at **`15.35 GB/s` `8T` decode** (`1.88×` faster than `Zstd+Dict-3` `8.17 GB/s`). On individual unstructured/nested files (`gharchive`, `apache.clf`, `macos-install.log`, `shakespeare`, `vxg-src-nonrepeating.rs`, `kodak.bmp`), `Zstd+Dict-9` retains per-file ratio wins via `FSE + optimal parsing`, and `LZ4` leads raw `GB/s`.

---

## 2. In-Cache Analytical Queries (`200k–300k` Rows, `1T` & `8T`): `VXG v0.5.3` vs. `DuckDB 2.0.0` + `Parquet` {#duckdb}

Measured standalone with zero background CPU contention (`libvxg.dylib` via `ctypes` vs. `duckdb 2.0.0` Python C++ extension, warm OS page cache, median of 5–7 runs, **100% result equality asserted on every query** via [`scripts/bench_vs_duckdb.py`](../scripts/bench_vs_duckdb.py); raw JSON artifact: [`reports/duckdb-vs-vxg.json`](../reports/duckdb-vs-vxg.json)). Parquet columns report **`PREPARE / EXECUTE` (`prep`)** latencies for both default `ROW_GROUP_SIZE = 122880` and fine-grained `ROW_GROUP_SIZE = 16384` (`SELECT 1` calibration = `0.04 ms` `1T`, `0.09 ms` `8T`).

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

- **In-Cache Summary (`1T` & `8T`):** With thread-local `GRAMMAR_TPL_WIRE_CACHE` + `JSON_TPL_SLOT_CACHE` memoization, zero-alloc `GroupAccumSoa` fused group buckets (`DenseRange` `<= 4096` + `DictKeys` SoA), dynamic `Top-K` zone-map/delta pruning, and short-circuit `RowBitset` predicates:
  - **`8` Threads (`8T`):** Both fixed-`64 KiB` `VXG` (`0.17–0.92 ms`) and `VXG --record-aligned` (`0.03–0.23 ms`) **win `15 / 15` queries** against prepared `DuckDB 2.0 + Parquet-Snappy/Zstd`.
  - **`1` Thread (`1T`):** `VXG --record-aligned` **wins `15 / 15` queries (`0.12–1.07 ms`)**, while fixed-`64 KiB` `VXG` **wins `11 / 15` queries** (`Parquet-Snappy` leads `4 / 6` `1T` queries on fixed-`64K` `nyc-taxi.ndjson`).

---

## 3. Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Real 2024 NYC TLC Rows, `1T` & `8T`) {#scale}

Evaluated over **all 12 months (`41,169,720` unmodified official rows) of 2024 NYC TLC Yellow Taxi trip records** via [`scripts/bench_scale_duckdb.py`](../scripts/bench_scale_duckdb.py) (raw artifacts: [`reports/scale-duckdb-vs-vxg.md`](../reports/scale-duckdb-vs-vxg.md), [`reports/scale-duckdb-vs-vxg.json`](../reports/scale-duckdb-vs-vxg.json), and empirical diagnostics in [`reports/ndjson-scale-diagnostics.json`](../reports/ndjson-scale-diagnostics.json)). Every query asserts 100% exact numerical equality between `VXG` and `DuckDB 2.0.0`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.

### 3.1 `nyc-taxi-2024-40m.csv` (`40,000,000` Real Rows, `3.06 GB` Raw)

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

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `313.51 ms` (`O(single_frame)` RAM across all `40M` rows).

### 3.2 `nyc-taxi-2024-10m.ndjson` (`10,000,000` Real Rows, `2.62 GB` Raw)

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

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `135.37 ms` (`O(single_frame)` RAM across all `10M` rows).
- **Empirical Root-Cause Diagnostics (`reports/ndjson-scale-diagnostics.json`):**
  1. **Zone-map selectivity:** On `nyc-taxi-2024-10m.ndjson`, `payment_type == 1` prunes `0.00%` of `64 KiB` blocks and `tip_cents >= 1000` prunes only `7.49%` (`7.41%` at `512 KiB`) because credit-card `$10+` tips occur every few dozen trips — confirming that a macro-block footer zone map alone would not solve verbose NDJSON scan cost.
  2. **Per-block JSON template re-parsing (`~85%` of `v0.5.1` `1T` latency) & `v0.5.3` Zero-Alloc `GroupAccumSoa`:** In `v0.5.1`, fixed-`64 KiB` NDJSON (`~245` rows/block across `~40,000` blocks) re-parsed identical `GRAMMAR_PACKET` JSON key templates up to `9×` per block. Memoizing `(tpl_hash, len) -> GrammarTpl` + 4-way `(tpl_hash, key) -> slot_idx`, bounding `--record-aligned` verbose NDJSON blocks to `target_block_rows = 4096` (`<= 1 MiB` raw), and fusing `GROUP BY` into zero-allocation `GroupAccumSoa` (`DenseRange` + `DictKeys`) with atomic work-stealing cut `COUNT WHERE` to **`29.18 ms` (`1T`) / `3.93 ms` (`8T`)** and `GROUP BY` to **`31.06 ms` (`1T`) / `4.31 ms` (`8T`)** — beating `DuckDB + Parquet-Snappy` on **`5 / 5` `8T` and `5 / 5` `1T` queries** while compressing smaller than both `Parquet-Snappy` (`140.8` vs `193.3 MB`) and `Parquet-Zstd` (`144.7 MB`).
  3. **Fine-grained Parquet (`ROW_GROUP_SIZE = 16384`):** Lowering Parquet's row-group size from `122,880` to `16,384` slows `DuckDB + Parquet-Snappy` by `2.1×–3.9×` on `8T` (`8.83 -> 23.17 ms` on `10M` NDJSON; `34.32 -> 98.53 ms` on `40M` CSV), confirming how sensitive columnar scan latency is to block/row-group count.

---

## 4. Synthetic & Procedural Qualification Corpora (`64 KiB` Blocks)

> **Provenance & Periodic Seed Disclosure:** All datasets in this section are **procedurally generated** for deterministic regression and format-coverage testing (`--suite v4`, `scripts/gen_varied_ndjson.py`, and `scripts/gen_format_bench.py`). Note that **`source-and-docs.txt` (`0.0036` in `VXG v5`, `0.0029` in `zstd -19 --long`)** repeats a `~190 KiB` 10-file source/docs seed to `16 MiB` via `fill_repeated` (`src/bin/bench_support/datasets.rs`), which fits inside `v5`'s `<= 256 KiB` frame dictionary; for **non-repeating real source code**, see **`vxg-src-nonrepeating.rs` (`0.1917`)** in Section 1.4 (`--suite v4-nonrep` replaces `source-and-docs.txt` with non-repeating workspace source). Raw artifacts: [`reports/synthetic-v4-1t.md`](../reports/synthetic-v4-1t.md), [`reports/synthetic-v4-8t.md`](../reports/synthetic-v4-8t.md), [`reports/synthetic-multitpl-1t.md`](../reports/synthetic-multitpl-1t.md), and [`reports/synthetic-multitpl-8t.md`](../reports/synthetic-multitpl-8t.md).

### 4.1 Synthetic Multi-Template & Delimited Suite (`gen_varied_ndjson.py` & `gen_format_bench.py`)

| Dataset (`64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | `1T` VXG Enc / Dec | `1T` LZ4 Enc / Dec | `8T` VXG Enc / Dec | `8T` LZ4 (`a1`) Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** *(3-template synthetic)* | 56.5 MB | **0.0914** | 0.2898 | 0.1480 | 0.1452 | 0.1198 | `1.035` / **`5.654`** | **`1.167`** / `4.567` | **`6.197` / `40.286`** | `5.980` / `25.444` |
| **`data.csv`** *(11-col synthetic + UA strings)* | 30.5 MB | **0.1794** | 0.4458 | 0.2651 | 0.2606 | 0.2257 | `0.671` / `2.357` | **`0.724` / `3.856`** | `3.778` / `14.732` | **`4.399` / `21.671`** |
| **`data.clf`** *(synthetic Apache CLF)* | 31.2 MB | **0.0980** | 0.2843 | 0.1504 | 0.1454 | 0.1225 | `0.608` / `2.525` | **`0.916` / `3.686`** | `3.612` / **`17.915`** | **`5.764`** / `17.246` |
| **3-Dataset Synthetic Geomean** | — | **0.1172** | 0.3324 | 0.1807 | 0.1765 | 0.1491 | `0.750` / `3.228` | **`0.918` / `4.019`** | `4.389` / **`21.989`** | **`5.333`** / `21.186` |

### 4.2 Synthetic `--suite v4` Qualification Suite (`16 MiB` per dataset, `8` Threads, Split Geomeans)

| Dataset (`16 MiB`, `8T`) | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc / Dec (`GB/s`) | `LZ4 Sweep` Enc / Dec | `Zstd+Dict Sweep` Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0868 | 0.0014 | 6.648 / **48.375** | **9.719** / 40.847 | 7.802 / 15.263 |
| **`events.tsv`** | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0680 | 0.0008 | 6.982 / **56.157** | **8.974** / 47.520 | 7.332 / 17.985 |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0731 | 0.0013 | 7.336 / **60.092** | **9.327** / 54.963 | 8.145 / 13.616 |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0544 | 0.0196 | 9.893 / **69.601** | **13.393** / 55.745 | 9.541 / 20.097 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.0902 | 0.0667 | 5.523 / 32.666 | **8.183 / 32.982** | 4.945 / 13.114 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0696 | 0.0517 | 6.465 / **48.273** | **11.105** / 42.341 | 7.320 / 17.102 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.0888 | 0.0642 | 4.997 / 26.342 | **7.728 / 34.165** | 5.208 / 11.764 |
| **`source-and-docs.txt`** *(periodic `~190 KiB` seed)* | **0.0036** | 0.4286 | 0.3133 | 0.2716 | 0.2007 | 0.0029 | **4.358 / 94.539** | 4.013 / 21.922 | 3.254 / 10.564 |
| **Text & Structured Geomean (8)** | **0.0259** | 0.2107 | 0.0959 | 0.0859 | 0.0846 | 0.0081 | 6.343 / **50.838** | **8.628** / 39.740 | 6.375 / 14.626 |
| **`ui-render-32bpp.bmp`** *(procedural UI)* | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0087 | 0.0010 | 27.868 / **76.950** | **36.387** / 66.873 | 35.765 / 26.612 |
| **`rgb-frame-24bpp.bmp`** *(procedural 24bpp)* | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0141 | 0.0010 | 27.348 / **73.312** | **34.627** / 44.447 | 27.465 / 33.933 |
| **Procedural 2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0111 | 0.0010 | 27.607 / **75.109** | **35.496** / 54.519 | 31.341 / 30.050 |

---

## Reproducing All Suites & Syncing Tables Locally

```sh
# 1. Fetch the real public & system corpora (/tmp/vxg-public-real):
/tmp/vxg-duckdb-venv/bin/python3 scripts/fetch_real_corpora.py /tmp/vxg-public-real

# 2. Run 64 KiB block + Zstd-Dict [32..256K] + whole-file zstd -19 --long benchmark:
cargo run --release --features bench --bin vxg-bench -- \
  --threads 8 --samples 5 --seconds 0.15 \
  --file /tmp/vxg-public-real/nyc-taxi-2024-01.csv \
  --file /tmp/vxg-public-real/nyc-taxi-2024-01.ndjson \
  --file /tmp/vxg-public-real/gharchive-2025-01-15.ndjson \
  --file /tmp/vxg-public-real/apache-access.clf \
  --file /tmp/vxg-public-real/macos-install.log \
  --file /tmp/vxg-public-real/pg100-shakespeare.txt \
  --file /tmp/vxg-public-real/kodak-truecolor-16.bmp \
  --csv reports/real-public-8t.csv --markdown reports/real-public-8t.md

# 3. Run in-cache (200k–300k rows) & out-of-cache (10M–40M rows) VXG vs DuckDB 2.0 + Parquet:
cargo build --release
/tmp/vxg-duckdb-venv/bin/python3 scripts/bench_vs_duckdb.py reports/duckdb-vs-vxg.json
/tmp/vxg-duckdb-venv/bin/python3 scripts/bench_scale_duckdb.py

# 4. Regenerate Markdown tables in README.md and docs/CHIMERA_BENCHMARKS.md from reports/:
python3 scripts/render_tables.py --write
python3 scripts/render_tables.py --check
```
