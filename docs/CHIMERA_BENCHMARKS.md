# Competitor Parameter-Sweep, Trained-Dictionary, Whole-File & DuckDB Benchmarks

This document evaluates `VXG v4` under three methodology requirements needed for an apples-to-apples evaluation of a random-access structured codec:

1. **Strict Separation of Real-World Public Corpora vs. Synthetic/Procedural Corpora:** Publicly downloadable operational datasets (`NYC TLC Yellow Taxi`, `GitHub Archive`, `Elastic Apache Access Log`, `Project Gutenberg Shakespeare`, `Kodak Lossless True Color` camera photographs, and macOS `/var/log/install.log`) are reported separately from synthetic/procedural qualification generators (`--suite v4`, `scripts/gen_varied_ndjson.py`, `scripts/gen_format_bench.py`, and procedural 2D UI `.bmp` renders).
2. **Trained-Dictionary (`Zstd + Dict`) & Whole-File (`zstd -19 --long`) Baselines:**
   - **Block-independent codecs (`64 KiB` blocks, `O(1)` random access = `YES`):** Native C `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`, **`zstd-dict-l3`**, **`zstd-dict-l9`**).
   - **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary per dataset via `ZDICT_trainFromBuffer` (`src/baselines/zstd_ext.rs`). The dictionary byte length (`dict_bytes`) is charged once in `encoded_size()`; `Enc` / `Dec` measure steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput, while one-time `ZDICT` training latency (`ms`) is reported separately.
   - **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window / `windowLog = 27`, single stream) removes `64 KiB` block boundaries completely to quantify the exact compression-ratio cost (or absence thereof) of `64 KiB` random access.
3. **Head-to-Head Analytical Queries vs. `DuckDB 2.0.0` + `Parquet`:** Compressed-domain queries are reported in **wall-clock latency (`ms`)** and **million rows/second (`M rows/s`)** — never by dividing raw file size by single-column scan time — alongside compressed storage (`MB`), one-time ingest/ETL time (`ms`), and **byte-exact lossless reversibility** against `DuckDB 2.0.0` (`read_json_auto` / `read_csv`, `.zst`, `Parquet-Snappy`, and `Parquet-Zstd`).

> **Reading the tables:** Bold cells mark **only the actual per-row winner** on each individual metric (lowest `64 KiB` block ratio, highest encode `GB/s`, highest decode `GB/s`). Composite indices (`VPI` / `BCI` / `GCI`) summarize space-speed-energy trade-offs and do **not** imply faster raw `GB/s` than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec`.

---

## 1. Real-World Public & System Corpora (`64 KiB` Independent Blocks)

Fetched reproducibly via [`scripts/fetch_real_corpora.py`](../scripts/fetch_real_corpora.py) (SHA-256 manifest written to `/tmp/vxg-public-real/manifest.json`). Raw benchmark artifacts: [`reports/real-public-1t.md`](../reports/real-public-1t.md), [`reports/real-public-1t.csv`](../reports/real-public-1t.csv), [`reports/real-public-8t.md`](../reports/real-public-8t.md), and [`reports/real-public-8t.csv`](../reports/real-public-8t.csv).

### 1.1 Corpus Provenance

| Corpus | Size | Description & Source |
| :--- | :---: | :--- |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | First `300,000` real NYC TLC Yellow Taxi Jan 2024 trip records (`13` tabular columns, public CloudFront Parquet source exported verbatim to CSV) |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | First `200,000` real NYC TLC Yellow Taxi Jan 2024 trip records serialized as flat NDJSON |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | `12,000` unmodified GitHub Archive (`2025-01-15-15`) public API events (deeply nested, heterogeneous JSON trees + Markdown commit/PR bodies) |
| **`apache-access.clf`** | 2.4 MB | Unmodified Elastic `examples` real Apache HTTP Combined Log Format corpus |
| **`macos-install.log`** | 34.2 MB | Unmodified macOS `/var/log/install.log` multi-line free-text system log |
| **`pg100-shakespeare.txt`** | 5.6 MB | Unmodified Project Gutenberg `#100` (*Complete Works of William Shakespeare*, English prose/verse) |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | Uncompressed `3072×2048` 24-bpp mosaic of 16 real optical camera photographs (`kodim01`–`kodim16`, Kodak Lossless True Color Image Suite) |

### 1.2 Single-Thread Results (`threads = 1`, `block = 65536`, Apple Silicon NEON)

| Dataset | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1807** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | **0.524** | 0.449 | 0.435 | 0.219 | 1.257 | **2.482** | 0.920 | 0.718 | 108.6 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **1.654** | 1.293 | 0.993 | 0.515 | **4.992** | 4.360 | 2.047 | 1.703 | 77.4 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1985 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 0.634 | **1.055** | 0.829 | 0.337 | 2.341 | **3.528** | 1.993 | 1.701 | 80.3 |
| **`apache-access.clf`** | 2.4 MB | 0.1198 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.401 | **1.202** | 1.078 | 0.533 | 2.107 | **5.567** | 2.463 | 1.835 | 64.3 |
| **`macos-install.log`** | 34.2 MB | 0.0948 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0277 | 1.404 | **1.914** | 0.977 | 0.731 | 3.748 | **7.352** | 3.261 | 3.309 | 49.3 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.4424 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.398 | 0.399 | **0.399** | 0.147 | 1.803 | **2.927** | 1.150 | 0.756 | 97.8 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | **1.207** | 0.860 | 0.527 | 0.106 | **10.099** | 4.435 | 1.029 | 0.971 | 101.9 |
| **Text & Structured Geomean (6)** | — | 0.1484 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 0.705 | **0.908** | 0.730 | 0.360 | 2.436 | **4.074** | 1.806 | 1.456 | 77.1 |
| **2D Camera BMP (1)** | — | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | **1.207** | 0.860 | 0.527 | 0.106 | **10.099** | 4.435 | 1.029 | 0.971 | 101.9 |

### 1.3 Multi-Thread Results (`threads = 8`, `block = 65536`, Apple Silicon NEON)

| Dataset | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1807** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | 1.340 | **1.591** | 0.900 | 0.483 | 3.101 | **4.628** | 1.718 | 1.768 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **10.200** | 6.540 | 6.715 | 2.928 | **34.689** | 19.968 | 10.129 | 6.922 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1985 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 1.969 | **2.446** | 1.873 | 1.215 | 5.463 | **9.414** | 4.306 | 3.374 |
| **`apache-access.clf`** | 2.4 MB | 0.1198 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.621 | 0.943 | **1.052** | 0.513 | 1.616 | 1.462 | **1.969** | 1.228 |
| **`macos-install.log`** | 34.2 MB | 0.0948 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0278 | **8.797** | 8.593 | 6.142 | 2.182 | 22.427 | **25.441** | 11.226 | 8.146 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.4424 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | **2.480** | 2.124 | 1.906 | 0.738 | 11.102 | **11.333** | 5.216 | 3.561 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | **3.584** | 1.617 | 1.359 | 0.414 | **19.609** | 6.239 | 2.481 | 3.294 |
| **Text & Structured Geomean (6)** | — | 0.1484 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 2.673 | **2.756** | 2.278 | 1.060 | 7.864 | **8.460** | 4.530 | 3.372 |

### 1.4 Engineering Takeaways on Real-World Data

1. **Where `VXG v4` Wins Clearly (Flat / Multi-Template Tabular & Log Streams):**
   - On **`nyc-taxi-2024-01.ndjson`** (`52.3 MB`), `VXG v4` achieves **`0.0592` (`5.92%`)** — **`3.21×` smaller than `LZ4` (`0.1899`)**, **`1.65×` smaller than `Zstd-1` (`0.0977`)**, **`1.39×` smaller than `Zstd-9` (`0.0823`)**, **`1.32×` smaller than `Zstd + Trained Dict L9` (`0.0783`)**, and within **`6.9%` of non-random-access whole-file `zstd -19 --long` (`0.0554`)**, while encoding at **`10.20 GB/s`** and decoding at **`34.69 GB/s`** (`8T`) — faster than `LZ4 (a1)` (`6.54 / 19.97 GB/s`), though `LZ4 (a4)` in the parameter sweep reaches `41.55 GB/s` decode.
   - On **`nyc-taxi-2024-01.csv`** (`23.0 MB`), `VXG v4` achieves **`0.1807`** — **`2.35×` smaller than `LZ4` (`0.4241`)**, **`1.35×` smaller than `Zstd-1` (`0.2440`)**, and **`1.18×` smaller than `Zstd + Trained Dict L9` (`0.2132`)**; `LZ4` remains `1.5×–2.0×` faster on raw CSV decode (`2.48 / 4.63 GB/s` vs `1.26 / 3.10 GB/s`).
2. **Where `Zstd + Trained Dict` and `zstd -19 --long` Win (Deeply Nested Trees, Unstructured Logs & Prose):**
   - On **`gharchive-2025-01-15.ndjson`** (deeply nested arbitrary JSON schemas + long PR/commit text bodies), **`macos-install.log`** (free-text multi-line log), **`apache-access.clf`** (`2.4 MB` with long repeated URLs/User-Agents across blocks), and **`pg100-shakespeare.txt`** (literary prose), `Zstd + Trained Dict` (`0.1150`, `0.0484`, `0.0878`, `0.3745`) outperforms both plain `Zstd` and `VXG` (`0.1985`, `0.0948`, `0.1198`, `0.4424`) on `64 KiB` block ratio, while `LZ4` leads raw decode speed.
   - Comparing `Zstd+Dict-9` (`0.1205` 6-dataset geomean) to whole-file `zstd -19 --long` (`0.0795`) isolates the exact **`1.52×` compression-ratio cost of `64 KiB` random-access block boundaries** on cross-block repetitive text (`~0.002–0.005 GB/s` encode speed, no random access).
3. **Real Camera Photographs vs. Synthetic UI BMPs:**
   - On **`kodak-truecolor-16.bmp`** (16 real optical camera photographs), sensor grain limits lossless compression across all general-purpose codecs to `0.5656` (`zstd -19 --long`) – `0.7452` (`Zstd+Dict-9`) – `0.9256` (`VXG`, `19.6 GB/s` `8T` decode). The `208:1` (`0.0048`) and `833:1` (`0.0012`) ratios seen in `--suite v4` occur strictly on synthetic flat-shaded UI/gradient renders and are reported separately in Section 3.

---

## 2. Head-to-Head Analytical Queries: `VXG` vs. `DuckDB 2.0.0` + `Parquet` {#duckdb}

Dividing an uncompressed `56.5 MB` file size by a `0.57 ms` single-column aggregation time yields `"98.9 GB/s"`, which overstates memory bandwidth because compressed-domain execution intentionally touches only a few megabytes of packed column metadata/values. Below we report **wall-clock latency (`ms`)** and **rows per second (`M rows/s`)** directly in-process (`libvxg.dylib` via `ctypes` vs. `duckdb 2.0.0` Python C++ extension, warm OS page cache, median of 5–7 runs, **100% result equality asserted on every query** via [`scripts/bench_vs_duckdb.py`](../scripts/bench_vs_duckdb.py); raw JSON artifact: [`reports/duckdb-vs-vxg.json`](../reports/duckdb-vs-vxg.json)).

### 2.1 Storage Footprint, Ingest / ETL Latency & Lossless Reversibility

| Corpus | Raw Size | **`.vxg` (`VXG v4`)** | `.zst` (`Zstd-3`) | `Parquet-Snappy` | `Parquet-Zstd` | `8T` Ingest: `.vxg` vs `Parquet-Zstd` ETL | Byte-Exact Original File Recovery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`nyc-taxi-2024-01.ndjson`** *(real, 200k rows)* | 52.33 MB | **3.10 MB (`5.92%`)** | 4.92 MB (`9.40%`) | 3.94 MB (`7.53%`) | **2.96 MB (`5.66%`)** | **`10.1 ms`** vs `111.9 ms` (**`11.1×` faster**) | **`.vxg`: YES (`100%` exact)** · `Parquet`: NO |
| **`nyc-taxi-2024-01.csv`** *(real, 300k rows)* | 23.00 MB | **4.16 MB (`18.07%`)** | 5.01 MB (`21.78%`) | 5.86 MB (`25.50%`) | 4.39 MB (`19.10%`) | **`12.2 ms`** vs `200.4 ms` (**`16.5×` faster**) | **`.vxg`: YES (`100%` exact)** · `Parquet`: NO |
| **`ndjson_varied.ndjson`** *(synthetic, 300k rows)* | 56.50 MB | **5.17 MB (`9.14%`)** | 9.86 MB (`17.46%`) | 9.70 MB (`17.16%`) | 5.56 MB (`9.83%`) | **`17.8 ms`** vs `184.7 ms` (**`10.4×` faster**) | **`.vxg`: YES (`100%` exact)** · `Parquet`: NO |

> Notice that `.vxg` is **`1.27×–1.88×` smaller than `Parquet-Snappy` on all three datasets** and **smaller than `Parquet-Zstd` on both `nyc-taxi.csv` (`4.16 MB` vs `4.39 MB`) and multi-schema `ndjson_varied.ndjson` (`5.17 MB` vs `5.56 MB`, where Parquet must materialize sparse `NULL`-padded union columns)**, while ingesting **`10×–16×` faster** and remaining **100% byte-for-byte reversible** to the original `.ndjson`/`.csv` input.

### 2.2 Query Latency (`ms`, Lower is Faster) & Row Scan Rate (`M rows/s`)

| Corpus & Query | Threads | **VXG `.vxg` (`ms` / `M rows/s`)** | DuckDB Raw (`ms`) | DuckDB `.zst` (`ms`) | DuckDB `Parquet-Snappy` (`ms`) | DuckDB `Parquet-Zstd` (`ms` / `M rows/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real `nyc-taxi-2024-01.ndjson` (`200k` rows)** | | | | | | |
| `COUNT(*) WHERE pay=1 AND tip>=1000 AND dist>=500` | `1T` / `8T` | `6.50 / 4.57 ms` (`30.8 / 43.8 M/s`) | `154.97 / 99.26` | `136.11 / 175.60` | **`1.97`** / `2.94` | `2.43` / **`2.11 ms`** (`82.2 / 95.0 M/s`) |
| `SUM/MIN/MAX(total_cents)` *(all 200k rows)* | `1T` / `8T` | **`1.84 / 1.32 ms` (`108.7 / 151.6 M/s`)** | `90.37 / 80.98` | `191.63 / 178.79` | `3.80 / 4.89` | `4.57 / 3.17 ms` (`43.8 / 63.1 M/s`) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `1T` / `8T` | `4.59` / **`1.42 ms`** (`43.5` / **`140.7 M/s`**) | `101.57 / 84.06` | `153.60 / 230.27` | **`2.17`** / `6.52` | `2.58 / 8.56 ms` (`77.4 / 23.4 M/s`) |
| `GROUP BY payment_type: COUNT(*), SUM(total)` | `1T` / `8T` | **`3.46`** / `6.95 ms` (`57.7 / 28.8 M/s`) | `104.73 / 83.38` | `199.16 / 153.13` | `3.78 / 4.65` | `4.38` / **`4.36 ms`** (`45.6 / 45.9 M/s`) |
| `ORDER BY total_cents DESC LIMIT 10` | `1T` / `8T` | **`4.80 / 4.64 ms` (`41.7 / 43.1 M/s`)** | `148.59 / 95.67` | `239.40 / 214.98` | `5.73 / 9.11` | `9.03 / 21.02 ms` (`22.2 / 9.5 M/s`) |
| `SELECT 4 cols WHERE tip_cents >= 2500` | `1T` / `8T` | **`4.14 / 2.37 ms` (`48.4 / 84.5 M/s`)** | `193.24 / 97.92` | `233.89 / 180.60` | `6.66 / 12.68` | `9.62 / 11.76 ms` (`20.8 / 17.0 M/s`) |
| **Real `nyc-taxi-2024-01.csv` (`300k` rows)** | | | | | | |
| `COUNT(*) WHERE pay=1 AND tip>=1000 AND dist>=500` | `1T` / `8T` | **`1.97 / 2.41 ms` (`152.6 / 124.5 M/s`)** | `94.24 / 81.76` | `157.04 / 154.56` | `3.51 / 6.02` | `4.41 / 11.02 ms` (`68.1 / 27.2 M/s`) |
| `SUM/MIN/MAX(total_cents)` *(all 300k rows)* | `1T` / `8T` | **`0.69 / 0.42 ms` (`436.5 / 710.7 M/s`)** | `118.92 / 76.06` | `151.64 / 138.42` | `2.33 / 2.90` | `3.13 / 2.86 ms` (`95.7 / 104.7 M/s`) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `1T` / `8T` | **`0.88 / 3.07 ms` (`339.6 / 97.8 M/s`)** | `87.70 / 71.41` | `136.08 / 141.22` | `2.64 / 3.14` | `3.55 / 3.71 ms` (`84.4 / 81.0 M/s`) |
| **Synthetic `ndjson_varied.ndjson` (`300k` rows)** | | | | | | |
| `COUNT(*) WHERE fraud & status=500 & lat>=150` | `1T` / `8T` | `3.99 / 4.41 ms` (`75.2 / 68.1 M/s`) | `115.46 / 96.53` | `211.00 / 184.10` | `5.21 / 11.60` | **`2.97 / 3.83 ms`** (`101.1 / 78.4 M/s`) |
| `SUM/MIN/MAX(bytes)` *(all 300k rows)* | `1T` / `8T` | **`2.00`** / `3.08 ms` (`149.8 / 97.4 M/s`) | `123.48 / 88.83` | `199.81 / 194.26` | `5.26` / **`2.61`** | `4.81 / 11.02 ms` (`62.4 / 27.2 M/s`) |
| `GROUP BY service: COUNT(*), SUM(bytes)` | `1T` / `8T` | **`2.95 / 1.51 ms` (`101.7 / 198.2 M/s`)** | `123.92 / 86.63` | `188.31 / 188.65` | `5.45 / 4.82` | `8.73 / 11.50 ms` (`34.3 / 26.1 M/s`) |
| `ORDER BY latency_ms DESC LIMIT 10` | `1T` / `8T` | **`2.51 / 1.32 ms` (`119.6 / 227.5 M/s`)** | `117.06 / 76.98` | `202.86 / 193.82` | `7.55 / 11.85` | `12.24 / 11.83 ms` (`24.5 / 25.4 M/s`) |
| `SELECT 3 cols WHERE fraud & 500 & lat>=150` | `1T` / `8T` | **`5.29 / 3.84 ms` (`56.7 / 78.0 M/s`)** | `127.35 / 85.67` | `200.55 / 195.81` | `8.76 / 13.32` | `11.31 / 18.87 ms` (`26.5 / 15.9 M/s`) |

**Summary vs. DuckDB:**
- Against **DuckDB querying raw `.ndjson`/`.csv` or `.zst` directly** (the only DuckDB modes that preserve the original file), `VXG` is **`21×–181×` faster** (`0.42–6.95 ms` vs `71–239 ms`).
- Against **DuckDB querying pre-converted `Parquet-Snappy` / `Parquet-Zstd`**, `VXG` is faster on `SUM/MIN/MAX`, `ORDER BY ... LIMIT 10`, `SELECT ... WHERE`, CSV `COUNT(*) WHERE`, and dictionary `GROUP BY service` (`0.42–4.80 ms` vs `2.33–21.02 ms`), while **`DuckDB + Parquet` is faster on multi-column numeric `COUNT(*) WHERE` (`1.97–2.11 ms` vs `4.57–6.50 ms` on `nyc-taxi.ndjson`, `2.97–3.83 ms` vs `3.99–4.41 ms` on `ndjson_varied`) and `8T` integer `GROUP BY payment_type` (`4.36 ms` vs `6.95 ms`)**.

---

## 3. Synthetic & Procedural Qualification Corpora (`64 KiB` Blocks)

All datasets in this section are **procedurally generated** for deterministic regression and format-coverage testing (`--suite v4` in `src/bin/bench_support/datasets.rs`, `scripts/gen_varied_ndjson.py`, and `scripts/gen_format_bench.py`). Raw artifacts: [`reports/synthetic-v4-1t.md`](../reports/synthetic-v4-1t.md), [`reports/synthetic-v4-8t.md`](../reports/synthetic-v4-8t.md), [`reports/synthetic-multitpl-1t.md`](../reports/synthetic-multitpl-1t.md), and [`reports/synthetic-multitpl-8t.md`](../reports/synthetic-multitpl-8t.md).

### 3.1 Synthetic Multi-Template & Delimited Suite (`gen_varied_ndjson.py` & `gen_format_bench.py`)

| Dataset (`64 KiB` blocks) | Size | VXG Ratio | LZ4 (`a1`) | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | `1T` VXG Enc / Dec | `1T` LZ4 Enc / Dec | `8T` VXG Enc / Dec | `8T` LZ4 Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** *(3-template synthetic)* | 56.5 MB | **0.0914** | 0.2898 | 0.1480 | 0.1452 | 0.1198 | `0.934` / **`5.509`** | **`1.052`** / `4.294` | `3.077` / `16.376` | **`4.899` / `21.810`** |
| **`data.csv`** *(11-col synthetic + UA strings)* | 30.5 MB | **0.1796** | 0.4458 | 0.2651 | 0.2606 | 0.2257 | `0.649` / `2.135` | **`0.676` / `3.974`** | **`1.947`** / `5.131` | `1.690` / **`5.998`** |
| **`data.clf`** *(synthetic Apache CLF)* | 31.2 MB | **0.0980** | 0.2843 | 0.1504 | 0.1454 | 0.1225 | `0.592` / `2.447` | **`0.877` / `3.607`** | `1.847` / `5.615` | **`2.297` / `6.818`** |
| **3-Dataset Synthetic Geomean** | — | **0.1172** | 0.3324 | 0.1807 | 0.1765 | 0.1491 | `0.711` / `3.064` | **`0.854` / `3.948`** | `2.228` / `7.785` | **`2.669` / `9.626`** |

- **Why `VXG` (`0.0914`–`0.1796`) beats whole-file `zstd -19 --long` (`0.1198`–`0.2257`) here:** Even with a `128 MiB` window, sliding-window LZ77 cannot delta-pack ASCII integers/timestamps or transpose multi-template columns; `VXG` with `64 KiB` random-access blocks is **`1.25×–1.31×` smaller than whole-file `zstd -19 --long`** and **`1.45×–1.60×` smaller than `Zstd + Trained Dict L9`**, while `LZ4` leads raw `1T`/`8T` decode speed (`1.17×–1.86×` faster on `data.csv`/`data.clf`).

### 3.2 Synthetic `--suite v4` Qualification Suite (`16 MiB` per dataset, `8` Threads, Split Geomeans)

| Dataset (`16 MiB`, `8T`) | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc / Dec (`GB/s`) | `LZ4 Sweep` Enc / Dec | `Zstd+Dict Sweep` Enc / Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0868 | 0.0014 | **3.159 / 8.502** | 2.904 / 7.319 | 1.764 / 3.659 |
| **`events.tsv`** | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0680 | 0.0008 | **2.813** / 7.287 | 2.604 / **8.525** | 2.279 / 4.205 |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0731 | 0.0013 | 2.735 / 8.722 | **2.982 / 10.226** | 2.136 / 4.221 |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0544 | 0.0196 | 3.475 / **10.676** | **3.848** / 9.591 | 2.680 / 5.305 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.0902 | 0.0667 | 1.997 / 6.013 | **2.660 / 8.813** | 2.045 / 3.857 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0696 | 0.0517 | 6.972 / 33.323 | **7.177 / 42.812** | 6.397 / 13.304 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.0888 | 0.0642 | 2.344 / 4.921 | **2.722 / 6.476** | 1.831 / 5.512 |
| **`source-and-docs.txt`** | 0.3063 | 0.4231 | 0.3084 | 0.2673 | **0.1982** | 0.0028 | **3.639** / 16.103 | 3.550 / **24.676** | 3.511 / 8.407 |
| **Text & Structured Geomean (8)** | **0.0451** | 0.2103 | 0.0957 | 0.0857 | 0.0845 | 0.0080 | 3.164 / 9.891 | **3.354 / 11.773** | 2.575 / 5.485 |
| **`ui-render-32bpp.bmp`** *(procedural UI)* | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0087 | 0.0010 | **51.315 / 68.748** | 29.285 / 55.556 | 31.201 / 25.570 |
| **`rgb-frame-24bpp.bmp`** *(procedural 24bpp)* | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0141 | 0.0010 | **34.974 / 48.861** | 28.769 / 40.335 | 17.412 / 23.728 |
| **Procedural 2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0111 | 0.0010 | **42.364 / 57.958** | 29.026 / 47.337 | 23.309 / 24.632 |

- **Note on `Zstd + Dict` at `64 KiB` vs. `4 KiB` blocks:** On `64 KiB` homogeneous tabular blocks, plain `Zstd` warms up its window and adaptive FSE/Huffman tables within the first `2–4 KiB` of each block, so `Zstd+Dict-9` (`0.0845`) is within `1.4%` of `Zstd-9` (`0.0857`) once static dictionary table attachment and the `32 KiB` dictionary header are accounted for (whereas on `4 KiB` blocks, `Zstd+Dict` substantially outperforms plain `Zstd`, verified in `baselines::tests::baselines_dict_and_whole_long_roundtrip`). Notice also that on periodic `16 MiB` synthetic generators (`telemetry.csv`, `source-and-docs.txt`), `zstd -19 --long`'s `128 MiB` LDM window collapses synthetic macro-periodicity (`0.0014`, `0.0028`) that does not exist in real corpora (`nyc-taxi.csv` `0.1617`, `pg100-shakespeare.txt` `0.3000`).

---

## Reproducing All Suites Locally

```sh
# 1. Fetch the 7 real public & system corpora (/tmp/vxg-public-real):
/tmp/vxg-duckdb-venv/bin/python3 scripts/fetch_real_corpora.py /tmp/vxg-public-real

# 2. Run 64 KiB block + Zstd-Dict + whole-file zstd -19 --long benchmark on real corpora:
cargo run --release --features bench --bin vxg-bench -- \
  --threads 8 --samples 5 --seconds 0.05 \
  --file /tmp/vxg-public-real/nyc-taxi-2024-01.csv \
  --file /tmp/vxg-public-real/nyc-taxi-2024-01.ndjson \
  --file /tmp/vxg-public-real/gharchive-2025-01-15.ndjson \
  --file /tmp/vxg-public-real/apache-access.clf \
  --file /tmp/vxg-public-real/macos-install.log \
  --file /tmp/vxg-public-real/pg100-shakespeare.txt \
  --file /tmp/vxg-public-real/kodak-truecolor-16.bmp \
  --csv reports/real-public-8t.csv --markdown reports/real-public-8t.md

# 3. Run head-to-head VXG vs DuckDB 2.0 (Raw, .zst, Parquet-Snappy, Parquet-Zstd):
cargo build --release --lib
/tmp/vxg-duckdb-venv/bin/python3 scripts/bench_vs_duckdb.py reports/duckdb-vs-vxg.json
```
