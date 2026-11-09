# Competitor Parameter-Sweep, Trained-Dictionary, Whole-File & DuckDB Scale Benchmarks (`v0.5.2`)

This document evaluates `VXG v0.5.2` (`Format v4/v5`) under four strict methodology rules required for an honest, reproducible evaluation of a random-access structured codec and SQL engine. Every real-corpus and DuckDB table in this document is generated deterministically from [`reports/`](../reports/) by [`scripts/render_tables.py`](../scripts/render_tables.py) and verified against drift in CI.

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
| **`vxg-src-nonrepeating.rs`** | 2.6 MB | Concatenation of every unique `.rs` file in `VXG`'s `src/` tree (`0%` repeated files; contrasts with `--suite v4`'s periodic `source-and-docs.txt` seed) |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | `12,000` unmodified GitHub Archive (`2025-01-15-15`) public API events (deeply nested, heterogeneous JSON trees + Markdown commit/PR bodies) |
| **`apache-access.clf`** | 2.4 MB | Unmodified Elastic `examples` real Apache HTTP Combined Log Format corpus |
| **`macos-install.log`** | 34.2 MB | Unmodified macOS `/var/log/install.log` multi-line free-text system log |
| **`pg100-shakespeare.txt`** | 5.6 MB | Unmodified Project Gutenberg `#100` (*Complete Works of William Shakespeare*, English prose/verse) |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | Uncompressed `3072×2048` 24-bpp mosaic of 16 real optical camera photographs (`kodim01`–`kodim16`, Kodak Lossless True Color Image Suite) |

### 1.2 Single-Thread Results (`threads = 1`, `block = 65536`, Apple Silicon NEON)

<!-- BEGIN GENERATED TABLE: real-public-full-1t -->
| Dataset (`1T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **1.40** | 1.38 | 1.06 | 0.56 | **5.83** | 4.66 | 2.13 | 1.80 |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 0.55 | **0.61** | 0.48 | 0.25 | 1.85 | **3.13** | 1.28 | 0.87 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.16 | **1.27** | 0.98 | 0.35 | 2.12 | **5.73** | 2.38 | 2.11 |
| **`apache-access.clf`** | 2.4 MB | 0.0976 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.20 | **1.59** | 1.19 | 0.56 | 2.14 | **5.69** | 2.64 | 2.23 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0277 | 0.37 | **2.15** | 1.47 | 0.79 | 2.60 | **7.65** | 3.35 | 3.57 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3786 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.04 | 0.39 | **0.41** | 0.16 | 0.96 | **2.86** | 1.15 | 0.70 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.06 | **0.87** | 0.54 | 0.13 | 1.89 | **4.28** | 1.04 | 0.94 |
| **Text & Structured Geomean (6)** | — | **0.1172** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 0.27 | **1.06** | 0.84 | 0.39 | 2.23 | **4.67** | 2.01 | 1.62 |
<!-- END GENERATED TABLE: real-public-full-1t -->

### 1.3 Multi-Thread Results (`threads = 8`, `block = 65536`, Apple Silicon NEON)

<!-- BEGIN GENERATED TABLE: real-public-full-8t -->
| Dataset (`8T`, `64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **8.79** | 7.49 | 7.09 | 3.88 | **43.55** | 28.92 | 15.02 | 12.60 |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 2.63 | **4.64** | 2.54 | 1.38 | 11.95 | **17.40** | 7.86 | 4.61 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 1.05 | 6.69 | **7.16** | 2.57 | 14.14 | **29.29** | 15.28 | 14.73 |
| **`apache-access.clf`** | 2.4 MB | 0.0976 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.71 | **6.42** | 6.33 | 2.61 | 12.44 | **13.51** | 11.67 | 9.46 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0278 | 2.56 | **11.77** | 7.39 | 3.92 | 16.90 | **37.25** | 22.93 | 23.41 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3786 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.27 | **2.74** | 1.95 | 0.90 | 7.01 | **15.71** | 5.69 | 3.59 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.38 | **4.43** | 3.30 | 0.93 | 12.63 | **21.02** | 7.30 | 6.86 |
| **Text & Structured Geomean (6)** | — | **0.1172** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 1.51 | **6.03** | 4.77 | 2.24 | 14.88 | **22.10** | 11.83 | 9.38 |
<!-- END GENERATED TABLE: real-public-full-8t -->

### 1.4 Cumulative Numeric Time-Series (`datasets/covid-19`) & Non-Repeating Source Code (`64 KiB` Blocks)

<!-- BEGIN GENERATED TABLE: covid-and-src -->
| Cumulative Time-Series & Non-Repeating Source (`64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-9 `[32..256K]` | `zstd -19 --long` *(whole)* | `1T` VXG Enc / Dec (`GB/s`) | `8T` VXG Enc / Dec (`GB/s`) | `8T` LZ4 (`a1`) Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** *(datasets/covid-19 cumulative time series)* | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1495 | 0.1066 | 0.52 / 1.60 | 1.90 / 10.65 | **5.19** / **24.12** |
| **`us_confirmed_33m.csv`** *(datasets/covid-19 US county cumulative prefix)* | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.0938 | 0.0619 | 0.80 / 2.54 | 4.87 / 17.37 | **8.95** / **32.16** |
| **`vxg-src-nonrepeating.rs`** *(All unique VXG src/*.rs files, 0% repeat)* | 2.6 MB | 0.1933 | 0.3350 | 0.2334 | 0.1954 | **0.1889** | 0.1355 | 0.09 / 1.30 | 0.42 / 7.52 | **3.49** / **10.71** |
<!-- END GENERATED TABLE: covid-and-src -->

### 1.5 Engineering Takeaways on Real-World Data

1. **Flat Tabular & Cumulative Time-Series Records (`nyc-taxi`, `datasets/covid-19`):**
   - On **`nyc-taxi-2024-01.ndjson`** (`52.3 MB`), `VXG v0.5.2` achieves **`0.0593` (`5.93%`)** — **`3.20×` smaller than `LZ4` (`0.1899`)**, **`1.39×` smaller than `Zstd-9` (`0.0823`)**, **`1.32×` smaller than `Zstd + Trained Dict L9 [32..256K]` (`0.0784`)**, and within **`7.0%` of non-random-access whole-file `zstd -19 --long` (`0.0554`)**, while encoding at **`8.79 GB/s`** and decoding at **`43.55 GB/s`** (`8T`) — faster than `LZ4 (a1)` (`7.49 / 28.92 GB/s`).
   - On **`time-series-19-covid-combined.csv`** (`8.6 MB`) and **`us_confirmed_33m.csv`** (`35.2 MB`), `v0.5.2`'s exception-patched row-delta model (`PFOR-Delta`, `0x80 | base_width` in `MODEL_DECIMAL_DELTA` and `MODEL_SCALED_DECIMAL`) plus nullable CSV field support move `131 / 131` blocks from `BLOCK_DICT_LZ` (`0.1836` in `v0.5.1`) to `BLOCK_COLUMN` at **`0.0963`** (`31.9%` smaller than `Zstd-9` `0.1415`, and smaller than whole-file `zstd -19 --long` `0.1066`!) and **`0.0456`** on `us_confirmed` (`1.97×` smaller than `Zstd-9` `0.0897`).
2. **Unstructured Logs, Nested JSON, Prose & Non-Repeating Source (`BLOCK_DICT_LZ` in `Format v5`):**
   - `v0.5.2` eliminates the `v0.5.0 -> v0.5.1` `DICT_LZ` ratio regressions via a `4096`-entry `16`-bit full-tag local 4-byte match ring + `+1/+2` lazy lookahead (`--level 1..9`, default `3`): `apache-access.clf` improves `0.1037 -> 0.0976` (`0.0934` at `-l 9`, beating `v0.5.0`'s `0.0990`), `gharchive` improves `0.1231 -> 0.1181`, `macos-install.log` improves `0.0594 -> 0.0548`, and `pg100-shakespeare.txt` improves `0.3884 -> 0.3786`.
   - Across the **6-dataset real public geomean**, **`VXG v0.5.2` (`0.1172`) beats `Zstd+Dict-9 [32..256K]` (`0.1181`)**, `Zstd+Dict-3` (`0.1385`), `Zstd-9` (`0.1419`), `Zstd-1` (`0.1620`), and `LZ4` (`0.2500`) at **`14.88 GB/s` `8T` decode** (`1.59×` faster than `Zstd+Dict-3`). On individual unstructured/nested files (`gharchive`, `apache.clf`, `macos-install.log`, `shakespeare`, `vxg-src-nonrepeating.rs`, `kodak.bmp`), `Zstd+Dict-9` retains per-file ratio wins via `FSE + optimal parsing`, and `LZ4` leads raw `GB/s`.

---

## 2. In-Cache Analytical Queries (`200k–300k` Rows, `1T` & `8T`): `VXG v0.5.2` vs. `DuckDB 2.0.0` + `Parquet` {#duckdb}

Measured standalone with zero background CPU contention (`libvxg.dylib` via `ctypes` vs. `duckdb 2.0.0` Python C++ extension, warm OS page cache, median of 5–7 runs, **100% result equality asserted on every query** via [`scripts/bench_vs_duckdb.py`](../scripts/bench_vs_duckdb.py); raw JSON artifact: [`reports/duckdb-vs-vxg.json`](../reports/duckdb-vs-vxg.json)). Parquet columns report **`PREPARE / EXECUTE` (`prep`)** latencies for both default `ROW_GROUP_SIZE = 122880` and fine-grained `ROW_GROUP_SIZE = 16384` (`SELECT 1` calibration = `0.04 ms` `1T`, `0.09 ms` `8T`).

<!-- BEGIN GENERATED TABLE: duckdb-in-cache -->
| Corpus & Query (`1T / 8T` latency in `ms`, lower is faster) | `VXG` (`64K` std) | `VXG --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* | DuckDB Raw |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.ndjson`** *(52.3 MB, 200k rows)* | 3.10 MB (`6.8 ms`, YES) | **2.86 MB** (`26.1 ms`, YES) | 3.94 MB (`82.0 ms` ETL, NO) | 4.17 MB (`76.1 ms`, NO) | 2.96 MB (`90.4 ms` ETL, NO) | `52.33 MB` (YES) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 2.44 / 0.42 | **0.56** / **0.11** | 1.26 / 0.98 | 1.96 / 1.00 | 2.01 / 1.30 | 84.16 / 56.36 |
| ↳ `SUM/MIN/MAX(total_cents) [all 200k rows]` | 1.37 / 0.27 | **0.26** / **0.05** | 1.05 / 0.84 | 1.33 / 0.75 | 1.33 / 1.02 | 80.23 / 54.52 |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 1.74 / 0.47 | **0.33** / **0.12** | 1.20 / 0.89 | 1.49 / 0.89 | 1.47 / 1.06 | 81.95 / 55.41 |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | 2.54 / 0.61 | **1.02** / **0.28** | 1.15 / 0.92 | 1.51 / 0.84 | 1.48 / 1.09 | 82.93 / 56.32 |
| ↳ `ORDER BY total_cents DESC LIMIT 10` | 1.66 / 0.80 | **0.53** / **0.20** | 2.70 / 2.09 | 2.51 / 1.70 | 4.82 / 3.32 | 90.11 / 59.18 |
| ↳ `SELECT 4 cols WHERE tip_cents >= 2500` | 2.13 / 0.82 | **1.22** / **0.26** | 2.93 / 2.98 | 3.49 / 3.54 | 5.15 / 5.17 | 97.30 / 59.73 |
| **`nyc-taxi-2024-01.csv`** *(23.0 MB, 300k rows)* | 4.21 MB (`9.0 ms`, YES) | **4.19 MB** (`20.3 ms`, YES) | 5.86 MB (`91.3 ms` ETL, NO) | 6.19 MB (`86.8 ms`, NO) | 4.39 MB (`97.8 ms` ETL, NO) | `23.00 MB` (YES) |
| ↳ `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 0.98 / 0.19 | **0.87** / **0.16** | 1.83 / 1.21 | 2.58 / 1.28 | 2.67 / 1.50 | 81.12 / 49.67 |
| ↳ `SUM/MIN/MAX(total_cents) [all 300k rows]` | 0.51 / 0.20 | **0.45** / **0.16** | 1.36 / 0.89 | 1.78 / 0.87 | 1.80 / 1.04 | 76.17 / 50.24 |
| ↳ `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 0.63 / 0.21 | **0.55** / **0.18** | 1.61 / 0.98 | 1.99 / 1.02 | 2.06 / 1.17 | 79.27 / 49.47 |
| ↳ `GROUP BY payment_type: COUNT(*), SUM(total)` | 1.74 / 0.42 | 1.65 / **0.36** | **1.55** / 0.97 | 2.11 / 0.97 | 2.09 / 1.18 | 78.16 / 50.44 |
| **`ndjson_varied.ndjson`** *(56.5 MB, 300k rows)* | 5.17 MB (`9.3 ms`, YES) | **4.70 MB** (`34.2 ms`, YES) | 9.70 MB (`114.2 ms` ETL, NO) | 9.72 MB (`104.1 ms`, NO) | 5.56 MB (`119.5 ms` ETL, NO) | `56.50 MB` (YES) |
| ↳ `COUNT(*) WHERE fraud & status=500 & lat>=150` | 2.77 / 0.69 | **0.91** / **0.26** | 1.09 / 0.86 | 1.73 / 1.03 | 1.53 / 0.97 | 101.07 / 62.08 |
| ↳ `SUM/MIN/MAX(bytes) [all 300k rows]` | 1.39 / 0.26 | **0.35** / **0.05** | 2.28 / 1.18 | 2.52 / 0.90 | 3.56 / 1.72 | 92.22 / 57.53 |
| ↳ `GROUP BY service: COUNT(*), SUM(bytes)` | 2.60 / 0.45 | **0.84** / **0.12** | 2.89 / 1.57 | 3.26 / 1.39 | 4.16 / 2.15 | 100.27 / 62.81 |
| ↳ `ORDER BY latency_ms DESC LIMIT 10` | 1.71 / 0.79 | **0.65** / **0.21** | 3.41 / 2.09 | 2.57 / 1.62 | 4.19 / 2.30 | 103.42 / 62.11 |
| ↳ `SELECT 3 cols WHERE fraud & 500 & lat>=150` | 2.84 / 0.79 | **0.99** / **0.24** | 7.06 / 7.05 | 4.92 / 4.94 | 9.25 / 9.24 | 104.61 / 64.44 |
<!-- END GENERATED TABLE: duckdb-in-cache -->

- **In-Cache Summary (`1T` & `8T`):** With thread-local `GRAMMAR_TPL_WIRE_CACHE` + `JSON_TPL_SLOT_CACHE` memoization, `DenseRange` (`span <= 4096`) numeric `GROUP BY` buckets, and short-circuit `RowBitset` predicates:
  - **`8` Threads (`8T`):** Both fixed-`64 KiB` `VXG` (`0.19–0.82 ms`) and `VXG --record-aligned` (`0.05–0.36 ms`) **win `15 / 15` queries** against prepared `DuckDB 2.0 + Parquet-Snappy/Zstd`.
  - **`1` Thread (`1T`):** `VXG --record-aligned` **wins `14 / 15` queries** (`Parquet-Snappy` wins `1T` CSV `GROUP BY` `1.55` vs `1.65 ms`), while fixed-`64 KiB` `VXG` **wins `9 / 15` queries** (`Parquet-Snappy` leads `5 / 6` `1T` queries on fixed-`64K` `nyc-taxi.ndjson` and `1T` CSV `GROUP BY`).

---

## 3. Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Real 2024 NYC TLC Rows, `1T` & `8T`) {#scale}

Evaluated over **all 12 months (`41,169,720` unmodified official rows) of 2024 NYC TLC Yellow Taxi trip records** via [`scripts/bench_scale_duckdb.py`](../scripts/bench_scale_duckdb.py) (raw artifacts: [`reports/scale-duckdb-vs-vxg.md`](../reports/scale-duckdb-vs-vxg.md), [`reports/scale-duckdb-vs-vxg.json`](../reports/scale-duckdb-vs-vxg.json), and empirical diagnostics in [`reports/ndjson-scale-diagnostics.json`](../reports/ndjson-scale-diagnostics.json)). Every query asserts 100% exact numerical equality between `VXG` and `DuckDB 2.0.0`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.

### 3.1 `nyc-taxi-2024-40m.csv` (`40,000,000` Real Rows, `3.06 GB` Raw)

<!-- BEGIN GENERATED TABLE: duckdb-scale-40m-csv -->
| `nyc-taxi-2024-40m.csv` (`40M` rows, `3.06 GB` raw; `1T / 8T` in `ms`) | `VXG` (`64K` std) | `VXG --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Archive Size & `8T` Ingest / ETL** | 552.9 MB (`5.01 s`, YES) | **550.2 MB** (`1.80 s`, YES) | 773.2 MB (`3.07 s`, NO) | 818.8 MB (`6.61 s`, NO) | 579.3 MB (`3.07 s`, NO) |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 161.89 / 23.24 | **154.32** / **21.05** | 161.30 / 30.67 | 253.91 / 84.90 | 274.92 / 41.88 |
| `SUM/MIN/MAX(total_cents) [full scan]` | 106.41 / 18.81 | **98.22** / **16.70** | 145.05 / 26.88 | 209.37 / 61.07 | 202.00 / 31.71 |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 116.61 / 16.00 | **111.97** / **14.94** | 134.07 / 25.22 | 200.60 / 61.60 | 188.64 / 30.62 |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 263.14 / 35.02 | 242.90 / 31.92 | **174.81** / **31.15** | 254.83 / 75.46 | 234.05 / 41.13 |
| `ORDER BY total_cents DESC LIMIT 10` | 94.59 / 22.86 | 86.68 / **16.15** | **26.95** / 18.21 | 79.96 / 77.10 | 42.03 / 21.16 |
<!-- END GENERATED TABLE: duckdb-scale-40m-csv -->

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `805.71 ms` (`O(single_frame)` RAM across all `40M` rows).

### 3.2 `nyc-taxi-2024-10m.ndjson` (`10,000,000` Real Rows, `2.62 GB` Raw)

<!-- BEGIN GENERATED TABLE: duckdb-scale-10m-ndjson -->
| `nyc-taxi-2024-10m.ndjson` (`10M` rows, `2.62 GB` raw; `1T / 8T` in `ms`) | `VXG` (`64K` std) | `VXG --record-aligned` | DuckDB `Parquet-Snappy` *(prep)* | DuckDB `Parquet-Snappy` *(RG=16k)* | DuckDB `Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Archive Size & `8T` Ingest / ETL** | 152.6 MB (`2.23 s`, YES) | **140.7 MB** (`0.48 s`, YES) | 193.3 MB (`0.77 s`, NO) | 203.4 MB (`0.96 s`, NO) | 144.7 MB (`0.92 s`, NO) |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 125.46 / 23.89 | **29.06** / **5.77** | 40.44 / 9.64 | 65.94 / 24.97 | 66.48 / 12.28 |
| `SUM/MIN/MAX(total_cents) [full scan]` | 68.95 / 10.50 | **13.51** / **2.01** | 36.01 / 7.01 | 50.54 / 14.58 | 51.67 / 8.78 |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 87.80 / 12.23 | **17.59** / **2.25** | 34.67 / 6.89 | 49.33 / 15.96 | 50.26 / 8.57 |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 128.28 / 17.40 | 51.07 / **6.77** | **42.09** / 8.14 | 62.47 / 17.38 | 58.68 / 9.86 |
| `ORDER BY total_cents DESC LIMIT 10` | 87.83 / 31.46 | **9.39** / **1.79** | 15.07 / 7.65 | 22.93 / 19.20 | 25.45 / 10.42 |
<!-- END GENERATED TABLE: duckdb-scale-10m-ndjson -->

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `253.09 ms` (`O(single_frame)` RAM across all `10M` rows).
- **Empirical Root-Cause Diagnostics (`reports/ndjson-scale-diagnostics.json`):**
  1. **Zone-map selectivity:** On `nyc-taxi-2024-10m.ndjson`, `payment_type == 1` prunes `0.00%` of `64 KiB` blocks and `tip_cents >= 1000` prunes only `7.49%` (`7.41%` at `512 KiB`) because credit-card `$10+` tips occur every few dozen trips — confirming that a macro-block footer zone map alone would not solve verbose NDJSON scan cost.
  2. **Per-block JSON template re-parsing (`~85%` of `v0.5.1` `1T` latency):** In `v0.5.1`, fixed-`64 KiB` NDJSON (`~245` rows/block across `~40,000` blocks) re-parsed identical `GRAMMAR_PACKET` JSON key templates up to `9×` per block. Memoizing `(tpl_hash, len) -> GrammarTpl` + 4-way `(tpl_hash, key) -> slot_idx` cut fixed-`64K` `1T` `COUNT WHERE` from `176.40 ms` to `125.46 ms`, and bounding `--record-aligned` verbose NDJSON blocks to `target_block_rows = 4096` (`<= 1 MiB` raw) cut `COUNT WHERE` to **`29.06 ms` (`1T`) / `5.77 ms` (`8T`)** — beating `DuckDB + Parquet-Snappy` on `5 / 5` `8T` and `4 / 5` `1T` queries while compressing smaller than both `Parquet-Snappy` (`140.7` vs `193.3 MB`) and `Parquet-Zstd` (`144.7 MB`).
  3. **Fine-grained Parquet (`ROW_GROUP_SIZE = 16384`):** Lowering Parquet's row-group size from `122,880` to `16,384` slows `DuckDB + Parquet-Snappy` by `2.1×–4.2×` on `8T` (`9.64 -> 24.97 ms` on `10M` NDJSON; `30.67 -> 84.90 ms` on `40M` CSV), confirming how sensitive columnar scan latency is to block/row-group count.

---

## 4. Synthetic & Procedural Qualification Corpora (`64 KiB` Blocks)

> **Provenance & Periodic Seed Disclosure:** All datasets in this section are **procedurally generated** for deterministic regression and format-coverage testing (`--suite v4`, `scripts/gen_varied_ndjson.py`, and `scripts/gen_format_bench.py`). Note that **`source-and-docs.txt` (`0.0036` in `VXG v5`, `0.0029` in `zstd -19 --long`)** repeats a `~190 KiB` 10-file source/docs seed to `16 MiB` via `fill_repeated` (`src/bin/bench_support/datasets.rs`), which fits inside `v5`'s `<= 256 KiB` frame dictionary; for **non-repeating real source code**, see **`vxg-src-nonrepeating.rs` (`0.1933`)** in Section 1.4 (`--suite v4-nonrep` replaces `source-and-docs.txt` with non-repeating workspace source). Raw artifacts: [`reports/synthetic-v4-1t.md`](../reports/synthetic-v4-1t.md), [`reports/synthetic-v4-8t.md`](../reports/synthetic-v4-8t.md), [`reports/synthetic-multitpl-1t.md`](../reports/synthetic-multitpl-1t.md), and [`reports/synthetic-multitpl-8t.md`](../reports/synthetic-multitpl-8t.md).

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
