# Competitor Parameter-Sweep, Trained-Dictionary, Whole-File & DuckDB Scale Benchmarks (`v0.5.0`)

This document evaluates `VXG v0.5.0` (`Format v4/v5`) under four strict methodology rules required for an honest, reproducible evaluation of a random-access structured codec and SQL engine:

1. **Strict Separation of Real-World Public Corpora vs. Synthetic/Procedural Corpora:** Publicly downloadable operational datasets (`NYC TLC Yellow Taxi`, `GitHub Archive`, `Elastic Apache Access Log`, `Project Gutenberg Shakespeare`, `Kodak Lossless True Color` camera photographs, and macOS `/var/log/install.log`) are reported separately from synthetic/procedural qualification generators (`--suite v4`, `scripts/gen_varied_ndjson.py`, `scripts/gen_format_bench.py`, and procedural 2D UI `.bmp` renders).
2. **Trained-Dictionary (`Zstd + Dict`) & Whole-File (`zstd -19 --long`) Baselines:**
   - **Block-independent codecs (`64 KiB` blocks, `O(1)` random access = `YES`):** Native C `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`, **`zstd-dict-l3`**, **`zstd-dict-l9`**).
   - **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` external dictionary per dataset via `ZDICT_trainFromBuffer` (`src/baselines/zstd_ext.rs`). The dictionary byte length (`dict_bytes`) is charged once in `encoded_size()`; `Enc` / `Dec` measure steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput, while one-time `ZDICT` training latency (`ms`) is reported separately. **`VXG v5`** embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself (`FLAG_FRAME_DICT`), and **`VXG Enc` / `Dec` include full dictionary training and header decoding time**.
   - **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window / `windowLog = 27`, single stream) removes `64 KiB` block boundaries completely to quantify the exact compression-ratio cost of `64 KiB` random access.
3. **Head-to-Head Analytical Queries vs. `DuckDB 2.0.0` + `Parquet` (In-Cache `200k–300k` & Out-of-Cache `10M–40M` Rows):** Compressed-domain queries are reported in **wall-clock latency (`ms`)** and **million rows/second (`M rows/s`)** — never by dividing raw file size by single-column scan time — calibrated with `SELECT 1` and `PREPARE / EXECUTE` (`prep`) statements, alongside compressed storage (`MB`), ingest/ETL time, and **byte-exact lossless reversibility** across both in-cache (`23–56.5 MB`) and multi-GB full-year NYC TLC (`2.62–3.06 GB`, `10M–40M` rows) corpora.
4. **Honest Per-Row Bolding:** Bold cells mark **only the actual per-row winner** on each individual metric (lowest `64 KiB` block ratio, highest encode `GB/s`, highest decode `GB/s`, lowest query `ms`). Codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` compression) are excluded from speed/trade-off bolding.

---

## 1. Real-World Public & System Corpora (`64 KiB` Independent Blocks)

Fetched reproducibly via [`scripts/fetch_real_corpora.py`](../scripts/fetch_real_corpora.py). Raw benchmark artifacts: [`reports/real-public-1t.md`](../reports/real-public-1t.md), [`reports/real-public-1t.csv`](../reports/real-public-1t.csv), [`reports/real-public-8t.md`](../reports/real-public-8t.md), and [`reports/real-public-8t.csv`](../reports/real-public-8t.csv).

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
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | **0.606** | 0.595 | 0.464 | 0.240 | 2.171 | **2.908** | 1.269 | 0.900 | 83.6 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **1.586** | 1.304 | 1.029 | 0.533 | **6.159** | 4.495 | 2.014 | 1.722 | 53.7 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1350 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 0.236 | **1.234** | 0.975 | 0.462 | 1.451 | **5.660** | 2.288 | 2.176 | 59.1 |
| **`apache-access.clf`** | 2.4 MB | 0.0990 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.305 | **1.570** | 1.143 | 0.567 | 1.581 | **5.871** | 2.550 | 2.210 | 46.5 |
| **`macos-install.log`** | 34.2 MB | 0.0605 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0277 | 0.559 | **2.112** | 1.454 | 0.753 | 2.139 | **7.635** | 3.439 | 3.431 | 36.4 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3977 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.083 | 0.388 | **0.396** | 0.147 | 1.104 | **2.848** | 1.155 | 0.762 | 98.9 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.117 | **0.827** | 0.525 | 0.107 | 1.544 | **4.209** | 1.012 | 0.975 | 111.5 |
| **Text & Structured Geomean (6)** | — | 0.1231 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 0.384 | **1.036** | 0.821 | 0.393 | 2.042 | **4.598** | 1.974 | 1.640 | 59.5 |
| **2D Camera BMP (1)** | — | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.117 | **0.827** | 0.525 | 0.107 | 1.544 | **4.209** | 1.012 | 0.975 | 111.5 |

### 1.3 Multi-Thread Results (`threads = 8`, `block = 65536`, Apple Silicon NEON)

| Dataset | Size | VXG Ratio | LZ4 (`a1`) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | `zstd -19 --long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | 2.904 | **2.918** | 2.711 | 1.159 | **13.429** | 13.150 | 6.357 | 4.683 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **8.382** | 6.592 | 5.096 | 2.688 | **34.405** | 18.814 | 7.351 | 7.571 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1350 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 1.481 | **5.680** | 4.908 | 2.295 | 8.592 | **23.268** | 10.702 | 9.421 |
| **`apache-access.clf`** | 2.4 MB | 0.0990 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 1.119 | **5.326** | 4.787 | 2.691 | 8.197 | **13.850** | 7.575 | 6.855 |
| **`macos-install.log`** | 34.2 MB | 0.0605 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0278 | 3.507 | **12.180** | 7.187 | 4.096 | 15.206 | **34.371** | 19.865 | 18.375 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3977 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.396 | **2.749** | 1.875 | 0.754 | 7.217 | **14.219** | 6.163 | 4.615 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.774 | **4.400** | 2.903 | 0.605 | 9.477 | **18.720** | 5.510 | 4.615 |
| **Text & Structured Geomean (6)** | — | 0.1231 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 1.956 | **5.187** | 4.044 | 1.975 | 12.363 | **18.413** | 8.798 | 7.610 |

### 1.4 Engineering Takeaways on Real-World Data

1. **Flat Tabular Records (`nyc-taxi.ndjson`, `nyc-taxi.csv`):**
   - On **`nyc-taxi-2024-01.ndjson`** (`52.3 MB`), `VXG v0.5.0` achieves **`0.0592` (`5.92%`)** — **`3.21×` smaller than `LZ4` (`0.1899`)**, **`1.65×` smaller than `Zstd-1` (`0.0977`)**, **`1.39×` smaller than `Zstd-9` (`0.0823`)**, **`1.32×` smaller than `Zstd + Trained Dict L9` (`0.0783`)**, and within **`6.9%` of non-random-access whole-file `zstd -19 --long` (`0.0554`)**, while encoding at **`8.38 GB/s`** and decoding at **`34.41 GB/s`** (`8T`) — faster than `LZ4 (a1)` (`6.59 / 18.81 GB/s`; `LZ4 a4` reaches `34.61 GB/s` decode in the sweep).
   - On **`nyc-taxi-2024-01.csv`** (`23.0 MB`), `MODEL_SCALED_DECIMAL` + unquoted ISO-8601 `MODEL_TIMESTAMP` + Block-0 CSV header isolation lift `8T` throughput from `1.34 / 3.10 GB/s` (`v0.4.7`) to **`2.90 / 13.43 GB/s`** — **`1.17×` smaller than `Zstd+Dict-9` (`0.1829` vs `0.2132`)**, **`2.32×` smaller than `LZ4` (`0.4241`)**, and faster than `LZ4 (a1)` on `8T` decode (`13.43` vs `13.15 GB/s`).
2. **Unstructured Logs, Nested JSON & Prose (`BLOCK_DICT_LZ` in `Format v5`):**
   - Adding pure-Rust `BLOCK_DICT_LZ` (`kind = 3`, FastCover `64`-window frame prefix dictionary + 5-stream canonical Huffman + 3-repcode LZ77) reduced `gharchive` from `0.1985` to **`0.1350`**, `apache-access.clf` from `0.1198` to **`0.0990`**, `macos-install.log` from `0.0948` to **`0.0605`**, `shakespeare` from `0.4424` to **`0.3977`**, and `kodak` camera photos from `0.9256` to **`0.8106`**.
   - Across the **6-dataset real public geomean**, `VXG v0.5.0` (**`0.1231`**) now **strictly beats `LZ4` (`0.2500`), `Zstd-1` (`0.1620`), `Zstd-9` (`0.1419`), and `Zstd+Dict-3` (`0.1407`)**, coming within **`2.1%` of `Zstd + Trained Dict L9` (`0.1205`)** while keeping zero C dependencies (`wasm32` compatible) and full self-contained framing. `Zstd+Dict-9` (`FSE + optimal parsing`) still leads individual ratios on the 4 unstructured/nested-JSON files, and `LZ4` leads raw throughput (`5.19 / 18.41 GB/s` vs `1.96 / 12.36 GB/s` `8T` geomean).

---

## 2. In-Cache Analytical Queries (`200k–300k` Rows): `VXG v0.5.0` vs. `DuckDB 2.0.0` + `Parquet` {#duckdb}

Measured standalone with zero background CPU contention (`libvxg.dylib` via `ctypes` vs. `duckdb 2.0.0` Python C++ extension, warm OS page cache, median of 5–7 runs, **100% result equality asserted on every query** via [`scripts/bench_vs_duckdb.py`](../scripts/bench_vs_duckdb.py); raw JSON artifact: [`reports/duckdb-vs-vxg.json`](../reports/duckdb-vs-vxg.json)). Parquet columns report **`PREPARE / EXECUTE` (`prep`)** latencies (`SELECT 1` calibration = `0.04–0.05 ms` `1T`, `0.06–0.07 ms` `8T`).

### 2.1 Storage Footprint, Ingest / ETL Latency & Lossless Reversibility

| Corpus | Raw Size | **`.vxg` (`VXG v0.5`)** | `.zst` (`Zstd-3`) | `Parquet-Snappy` | `Parquet-Zstd` | `8T` Ingest: `.vxg` vs `Parquet-Zstd` ETL | Byte-Exact Original File Recovery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`nyc-taxi-2024-01.ndjson`** *(real, 200k rows)* | 52.33 MB | **3.10 MB (`5.92%`)** | 4.92 MB (`9.40%`) | 3.94 MB (`7.53%`) | **2.96 MB (`5.66%`)** | **`5.6 ms`** vs `94.2 ms` (**`16.7×` faster**) | **`.vxg`: YES (`100%` exact)** · `Parquet`: NO |
| **`nyc-taxi-2024-01.csv`** *(real, 300k rows)* | 23.00 MB | **4.21 MB (`18.29%`)** | 5.01 MB (`21.78%`) | 5.86 MB (`25.50%`) | 4.39 MB (`19.10%`) | **`7.3 ms`** vs `99.5 ms` (**`13.7×` faster**) | **`.vxg`: YES (`100%` exact)** · `Parquet`: NO |
| **`ndjson_varied.ndjson`** *(synthetic, 300k rows)* | 56.50 MB | **5.17 MB (`9.14%`)** | 9.86 MB (`17.46%`) | 9.70 MB (`17.16%`) | 5.56 MB (`9.83%`) | **`9.0 ms`** vs `127.2 ms` (**`14.1×` faster**) | **`.vxg`: YES (`100%` exact)** · `Parquet`: NO |

### 2.2 Query Latency (`ms`, Lower is Faster) & Row Scan Rate (`M rows/s`)

| Corpus & Query (`SELECT 1` prep = `0.04–0.05 ms` `1T` / `0.06–0.07 ms` `8T`) | Threads | **VXG `.vxg` (`ms` / `M rows/s`)** | DuckDB Raw (`ms`) | DuckDB `.zst` (`ms`) | DuckDB `Parquet-Snappy` *(prep `ms`)* | DuckDB `Parquet-Zstd` *(prep `ms` / `M rows/s`)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real `nyc-taxi-2024-01.ndjson` (`200k` rows)** | | | | | | |
| `COUNT(*) WHERE pay=1 AND tip>=1000 AND dist>=500` | `1T` / `8T` | `4.72` / **`0.70 ms`** (`42.4` / **`286.5 M/s`**) | `83.12 / 56.12` | `120.46 / 124.47` | **`1.50`** / `1.18` | `2.09 / 1.55 ms` (`95.5 / 128.9 M/s`) |
| `SUM/MIN/MAX(total_cents)` *(all 200k rows)* | `1T` / `8T` | `1.65` / **`0.35 ms`** (`121.6` / **`570.7 M/s`**) | `80.03 / 55.65` | `115.11 / 122.41` | **`1.12`** / `0.88` | `1.38 / 1.04 ms` (`145.2 / 192.0 M/s`) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `1T` / `8T` | `2.49` / **`0.66 ms`** (`80.5` / **`303.0 M/s`**) | `81.00 / 55.77` | `120.36 / 123.05` | **`1.33`** / `1.06` | `1.61 / 1.24 ms` (`124.5 / 160.9 M/s`) |
| `GROUP BY payment_type: COUNT(*), SUM(total)` | `1T` / `8T` | `3.09` / **`0.70 ms`** (`64.8` / **`287.4 M/s`**) | `85.12 / 55.76` | `118.20 / 123.66` | **`1.29`** / `1.07` | `1.72 / 1.20 ms` (`116.2 / 167.0 M/s`) |
| `ORDER BY total_cents DESC LIMIT 10` | `1T` / `8T` | `3.02` / **`1.19 ms`** (`66.2` / **`168.6 M/s`**) | `90.30 / 59.25` | `126.12 / 131.19` | **`2.90`** / `2.31` | `4.93 / 3.46 ms` (`40.5 / 57.9 M/s`) |
| `SELECT 4 cols WHERE tip_cents >= 2500` | `1T` / `8T` | **`3.00 / 0.97 ms` (`66.8 / 205.6 M/s`)** | `97.70 / 60.01` | `130.94 / 133.02` | `3.13 / 3.24` *(ad-hoc `2.60`)* | `5.32 / 5.61 ms` *(ad-hoc `4.19`)* |
| **Real `nyc-taxi-2024-01.csv` (`300k` rows)** | | | | | | |
| `COUNT(*) WHERE pay=1 AND tip>=1000 AND dist>=500` | `1T` / `8T` | **`0.90 / 0.16 ms` (`333.5 / 1,906.8 M/s`)** | `79.90 / 51.84` | `113.50 / 119.60` | `2.04 / 1.24` | `2.80 / 1.75 ms` (`107.2 / 171.6 M/s`) |
| `SUM/MIN/MAX(total_cents)` *(all 300k rows)* | `1T` / `8T` | **`0.52 / 0.21 ms` (`580.8 / 1,417.0 M/s`)** | `76.88 / 50.69` | `112.89 / 116.01` | `1.46 / 0.99` | `2.09 / 1.11 ms` (`143.5 / 270.2 M/s`) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `1T` / `8T` | **`0.63 / 0.22 ms` (`476.3 / 1,351.9 M/s`)** | `78.74 / 51.20` | `115.93 / 114.35` | `1.80 / 1.14` | `2.39 / 1.33 ms` (`125.6 / 226.2 M/s`) |
| `GROUP BY payment_type: COUNT(*), SUM(total)` | `1T` / `8T` | `1.79` / **`0.42 ms`** (`167.4` / **`713.6 M/s`**) | `79.10 / 51.97` | `114.63 / 119.65` | **`1.68`** / `1.05` | `2.18 / 1.27 ms` (`137.4 / 236.7 M/s`) |
| **Synthetic `ndjson_varied.ndjson` (`300k` rows)** | | | | | | |
| `COUNT(*) WHERE fraud & status=500 & lat>=150` | `1T` / `8T` | `3.95` / **`0.90 ms`** (`76.0` / **`332.0 M/s`**) | `103.90 / 64.01` | `156.89 / 153.99` | **`1.23`** / `0.94` | `1.62 / 1.05 ms` (`184.7 / 284.7 M/s`) |
| `SUM/MIN/MAX(bytes)` *(all 300k rows)* | `1T` / `8T` | **`1.65 / 0.29 ms` (`182.3 / 1,023.6 M/s`)** | `92.76 / 59.37` | `144.51 / 146.21` | `2.45 / 1.32` | `3.79 / 1.91 ms` (`79.1 / 157.2 M/s`) |
| `GROUP BY service: COUNT(*), SUM(bytes)` | `1T` / `8T` | `3.22` / **`0.84 ms`** (`93.3` / **`358.9 M/s`**) | `105.56 / 64.07` | `155.22 / 155.32` | **`3.11`** / `1.73` | `4.34 / 2.30 ms` (`69.1 / 130.7 M/s`) |
| `ORDER BY latency_ms DESC LIMIT 10` | `1T` / `8T` | **`2.45 / 0.90 ms` (`122.4 / 333.7 M/s`)** | `104.98 / 63.44` | `158.16 / 156.39` | `4.02 / 2.42` | `4.66 / 2.74 ms` (`64.4 / 109.4 M/s`) |
| `SELECT 3 cols WHERE fraud & 500 & lat>=150` | `1T` / `8T` | **`4.12 / 1.32 ms` (`72.9 / 227.5 M/s`)** | `108.35 / 65.00` | `160.80 / 159.87` | `7.46 / 7.33` *(ad-hoc `3.86`)* | `9.48 / 9.84 ms` *(ad-hoc `4.57`)* |

- **In-Cache Summary:** With `DenseRange` (`span <= 4096`) numeric `GROUP BY` buckets, cost-sorted short-circuit `RowBitset` predicates, and single-barrier parallel chunk reduction, **`VXG v0.5.0` wins all `15 / 15` `8T` queries against prepared `DuckDB 2.0 + Parquet-Snappy/Zstd`** (`0.16–1.32 ms`), and is **`49×–760×` faster** than `DuckDB` querying raw/`.zst` directly. At `1T`, `DuckDB + Parquet` still leads `5 / 6` queries on `nyc-taxi.ndjson` (`1.12–2.90 ms` vs `1.65–4.72 ms`) and `COUNT WHERE` on `ndjson_varied` (`1.23` vs `3.95 ms`).

---

## 3. Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Real 2024 NYC TLC Rows) {#scale}

Evaluated over **all 12 months (`41,169,720` unmodified official rows) of 2024 NYC TLC Yellow Taxi trip records** via [`scripts/bench_scale_duckdb.py`](../scripts/bench_scale_duckdb.py) (raw artifacts: [`reports/scale-duckdb-vs-vxg.md`](../reports/scale-duckdb-vs-vxg.md) and [`reports/scale-duckdb-vs-vxg.json`](../reports/scale-duckdb-vs-vxg.json)). Every query asserts 100% exact numerical equality between `VXG` and `DuckDB 2.0.0`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.

### 3.1 `nyc-taxi-2024-40m.csv` (`40,000,000` Real Rows, `3.06 GB` Raw)

- **Archive Size & `8T` Ingest / ETL Time:**
  - `VXG --record-aligned`: **`551.6 MB`** (`ratio = 0.1801`, **`1.56 s`** / `1.96 GB/s`, `lossless = YES`)
  - `VXG` (fixed `64 KiB`): `554.4 MB` (`ratio = 0.1810`, `4.41 s`, `lossless = YES`)
  - `Parquet-Snappy`: `773.2 MB` (`ratio = 0.2524`, `4.06 s` ETL, `lossless = NO`)
  - `Parquet-Zstd`: `579.4 MB` (`ratio = 0.1891`, `3.80 s` ETL, `lossless = NO`)

| Query (`40,000,000` CSV rows; `SELECT 1` prep = `0.05 ms` `1T` / `0.09 ms` `8T`) | `8T` `VXG` *(std)* | `8T` **`VXG --record-aligned`** | `8T` `DuckDB Parquet-Snappy / Zstd` *(prep)* | `1T` `VXG --record-aligned` | `1T` `DuckDB Parquet-Snappy / Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(raw CSV: `1.26 s`)* | **`27.71 ms` (`1,443.4 M/s`)** | `29.72 ms` (`1,345.9 M/s`) | `34.55 ms` / `57.44 ms` | **`148.84 ms` (`268.7 M/s`)** | `159.20 ms` / `283.61 ms` |
| `SUM/MIN/MAX(total_cents)` *(full scan)* | `19.19 ms` (`2,084.9 M/s`) | **`18.41 ms` (`2,172.7 M/s`)** | `31.33 ms` / `48.85 ms` | **`98.45 ms` (`406.3 M/s`)** | `162.46 ms` / `214.79 ms` |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `21.12 ms` (`1,893.9 M/s`) | **`18.59 ms` (`2,151.7 M/s`)** | `31.58 ms` / `36.91 ms` | **`111.60 ms` (`358.4 M/s`)** | `144.21 ms` / `197.55 ms` |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | `48.84 ms` (`819.0 M/s`) | **`44.39 ms` (`901.1 M/s`)** | `49.14 ms` / `48.98 ms` | `250.25 ms` | **`173.94 ms`** / `237.69 ms` |
| `ORDER BY total_cents DESC LIMIT 10` | `24.72 ms` (`1,618.2 M/s`) | **`17.53 ms` (`2,281.8 M/s`)** | `18.39 ms` / `22.74 ms` | `87.97 ms` | **`29.19 ms`** / `43.97 ms` |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `995.76 ms` (`O(single_frame)` RAM across all `40M` rows).

### 3.2 `nyc-taxi-2024-10m.ndjson` (`10,000,000` Real Rows, `2.62 GB` Raw)

- **Archive Size & `8T` Ingest / ETL Time:**
  - `VXG --record-aligned`: `145.9 MB` (`ratio = 0.0558`, `1.02 s` / `2.57 GB/s`, **`lossless = YES`**)
  - `VXG` (fixed `64 KiB`): `152.7 MB` (`ratio = 0.0584`, `2.11 s`, `lossless = YES`)
  - `Parquet-Snappy`: `193.3 MB` (`ratio = 0.0739`, **`0.93 s`** ETL, `lossless = NO`)
  - `Parquet-Zstd`: **`144.6 MB`** (`ratio = 0.0553`, `1.18 s` ETL, `lossless = NO`)

| Query (`10,000,000` NDJSON rows; `SELECT 1` prep = `0.05 ms` `1T` / `0.09 ms` `8T`) | `8T` `VXG` *(std)* | `8T` `VXG --record-aligned` | `8T` **`DuckDB Parquet-Snappy / Zstd`** *(prep)* | `1T` `VXG --record-aligned` | `1T` **`DuckDB Parquet-Snappy / Zstd`** *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(raw JSON: `0.42 s`)* | `43.31 ms` | `32.12 ms` (`311.3 M/s`) | **`9.36 ms` (`1,068.8 M/s`)** / `12.43 ms` | `176.40 ms` | **`40.25 ms`** / `68.79 ms` |
| `SUM/MIN/MAX(total_cents)` *(full scan)* | `14.96 ms` | `8.26 ms` (`1,210.7 M/s`) | **`8.20 ms` (`1,219.9 M/s`)** / `9.69 ms` | `42.71 ms` | **`37.76 ms`** / `52.97 ms` |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | `21.08 ms` | `13.06 ms` (`765.7 M/s`) | **`8.02 ms` (`1,246.6 M/s`)** / `10.16 ms` | `76.78 ms` | **`36.03 ms`** / `52.11 ms` |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | `25.93 ms` | `16.81 ms` (`594.9 M/s`) | **`13.20 ms` (`757.5 M/s`)** / `13.62 ms` | `98.73 ms` | **`43.91 ms`** / `59.87 ms` |
| `ORDER BY total_cents DESC LIMIT 10` | `50.66 ms` | `22.92 ms` (`436.3 M/s`) | **`9.09 ms` (`1,100.3 M/s`)** / `11.68 ms` | `106.60 ms` | **`15.45 ms`** / `26.27 ms` |

- **Why `VXG` sweeps `40M` CSV (`5/5` at `8T`) while `DuckDB + Parquet` leads `10M` NDJSON (`4/5` clear, `1/5` tie):** `40M` CSV (`3.06 GB`) spans `~48,000` blocks (`~830` rows/block), whereas `10M` NDJSON (`2.62 GB`) repeats verbose JSON keys on every record (`~245` rows per `64 KiB` block across `~41,000` blocks). `--record-aligned` (`FLAG_VAR_RAW`) eliminates cross-block split-line `Row::Raw` fallbacks (`2.2×` faster `Top-10`, `1.8×` faster `SUM`), matching `Parquet-Snappy` on full-scan `SUM` (`8.26` vs `8.20 ms`), while `Parquet`'s `122,880`-row centralized row-group zone maps touch `340×` fewer metadata headers on selective NDJSON queries.

---

## 4. Synthetic & Procedural Qualification Corpora (`64 KiB` Blocks)

All datasets in this section are **procedurally generated** for deterministic regression and format-coverage testing (`--suite v4`, `scripts/gen_varied_ndjson.py`, and `scripts/gen_format_bench.py`). Raw artifacts: [`reports/synthetic-v4-1t.md`](../reports/synthetic-v4-1t.md), [`reports/synthetic-v4-8t.md`](../reports/synthetic-v4-8t.md), [`reports/synthetic-multitpl-1t.md`](../reports/synthetic-multitpl-1t.md), and [`reports/synthetic-multitpl-8t.md`](../reports/synthetic-multitpl-8t.md).

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
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0868 | 0.0014 | 6.463 / **41.768** | **9.023** / 35.636 | 6.188 / 15.226 |
| **`events.tsv`** | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0680 | 0.0008 | 5.955 / **53.119** | 7.783 / 41.877 | **7.943** / 15.298 |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0731 | 0.0013 | 7.315 / **54.775** | **9.479** / 44.010 | 7.280 / 14.582 |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0544 | 0.0196 | 9.512 / **58.922** | **12.512** / 44.187 | 8.717 / 20.988 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.0902 | 0.0667 | 4.917 / 30.184 | **8.045 / 36.409** | 5.963 / 12.806 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0696 | 0.0517 | 6.410 / **48.296** | **10.173** / 44.023 | 7.368 / 14.718 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.0888 | 0.0642 | 4.956 / 21.545 | **7.571 / 25.372** | 4.964 / 12.347 |
| **`source-and-docs.txt`** | 0.2592 | 0.4259 | 0.3107 | 0.2694 | **0.1986** | 0.0029 | 0.826 / 7.967 | **3.856 / 23.850** | 3.117 / 8.913 |
| **Text & Structured Geomean (8)** | **0.0442** | 0.2105 | 0.0958 | 0.0858 | 0.0845 | 0.0081 | 4.924 / 34.014 | **8.173 / 35.990** | 6.177 / 14.003 |
| **`ui-render-32bpp.bmp`** *(procedural UI)* | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0087 | 0.0010 | 29.456 / **81.515** | **32.942** / 63.044 | 29.169 / 24.405 |
| **`rgb-frame-24bpp.bmp`** *(procedural 24bpp)* | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0141 | 0.0010 | 27.506 / **69.960** | **28.565** / 44.639 | 24.092 / 32.636 |
| **Procedural 2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0111 | 0.0010 | 28.464 / **75.517** | **30.676** / 53.049 | 26.509 / 28.222 |

---

## Reproducing All Suites Locally

```sh
# 1. Fetch the 7 real public & system corpora (/tmp/vxg-public-real):
/tmp/vxg-duckdb-venv/bin/python3 scripts/fetch_real_corpora.py /tmp/vxg-public-real

# 2. Run 64 KiB block + Zstd-Dict + whole-file zstd -19 --long benchmark on real corpora:
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
```
