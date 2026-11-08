## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0950 | 0.0868 | 0.0014 | 1.442 | **1.805** | 1.122 | 0.594 | **8.531** | 7.259 | 2.195 | 1.966 | 45.9 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0710 | 0.0680 | 0.0008 | 1.419 | **1.493** | 1.311 | 0.669 | **8.880** | 5.824 | 2.607 | 2.641 | 42.8 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0856 | 0.0731 | 0.0013 | 1.496 | **1.714** | 1.289 | 0.610 | **9.729** | 7.073 | 2.468 | 2.012 | 44.8 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0698 | 0.0544 | 0.0196 | 2.013 | **2.435** | 1.674 | 0.752 | **12.673** | 9.394 | 3.343 | 2.562 | 35.6 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.1161 | 0.0902 | 0.0667 | 1.137 | **1.317** | 0.978 | 0.535 | **5.234** | 4.884 | 1.934 | 1.727 | 47.5 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0854 | 0.0696 | 0.0517 | 1.406 | **1.892** | 1.294 | 0.674 | **8.194** | 5.762 | 2.364 | 1.986 | 43.6 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.1198 | 0.0888 | 0.0642 | 0.960 | **1.351** | 0.873 | 0.512 | 4.293 | **4.388** | 1.794 | 1.671 | 72.1 |
| **`source-and-docs.txt`** | 16 MiB | 0.2592 | 0.4259 | 0.3107 | 0.2694 | 0.2261 | **0.1986** | 0.0029 | 0.126 | **0.602** | 0.506 | 0.267 | 1.154 | **3.615** | 1.413 | 1.221 | 76.3 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0101 | 0.0087 | 0.0012 | 9.039 | **12.538** | 6.499 | 6.265 | 14.393 | **19.963** | 4.801 | 4.670 | 14.3 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0139 | 0.0141 | 0.0012 | 8.155 | **10.511** | 7.047 | 2.273 | **14.463** | 10.367 | 7.027 | 4.579 | 19.0 |
| **Text & Structured Geomean (8)** | — | **0.0442** | 0.2105 | 0.0958 | 0.0858 | 0.1010 | 0.0845 | 0.0081 | 1.022 | **1.478** | 1.074 | 0.555 | **6.127** | 5.785 | 2.199 | 1.923 | 49.4 |
| **2D BMP Geomean (2)** | — | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0118 | 0.0111 | 0.0012 | 8.586 | **11.480** | 6.767 | 3.773 | **14.428** | 14.386 | 5.808 | 4.624 | 16.5 |
| **All-Dataset Geomean (10)** | — | **0.0248** | 0.1285 | 0.0615 | 0.0561 | 0.0658 | 0.0563 | 0.0056 | 1.564 | **2.227** | 1.552 | 0.814 | **7.271** | 6.941 | 2.671 | 2.292 | 39.7 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | 1.442 | **1.805** | 1.247 | 8.531 | **10.281** | 2.676 | **86.1** | 26.0 | 22.9 | 54.0 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0680 | 1.419 | **1.513** | 1.446 | 8.880 | **11.226** | 3.109 | **108.8** | 27.2 | 31.2 | 60.6 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | 1.496 | **1.732** | 1.350 | 9.729 | **10.403** | 2.768 | **108.1** | 26.2 | 27.8 | 61.0 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | 2.013 | **2.435** | 2.023 | 12.673 | **14.369** | 4.320 | **222.9** | 51.6 | 60.6 | 121.3 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | 1.137 | **1.339** | 1.095 | 5.234 | **8.260** | 2.301 | **54.8** | 22.3 | 18.0 | 37.7 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0696 | 1.406 | **1.946** | 1.378 | 8.194 | **11.640** | 2.983 | **136.7** | 42.0 | 29.1 | 68.4 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0888 | 0.960 | **1.372** | 0.873 | 4.293 | **6.598** | 2.387 | **42.3** | 23.0 | 16.3 | 33.9 |
| **`source-and-docs.txt`** | 0.2592 | 0.3424 | **0.1986** | 0.126 | **0.649** | 0.534 | 1.154 | **4.931** | 1.739 | 1.5 | 5.2 | 4.9 | **9.0** |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | 9.039 | **12.540** | 11.018 | 14.393 | **19.963** | 4.801 | **2383.7** | 1128.0 | 843.2 | 1834.3 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 8.155 | **10.511** | 8.347 | **14.463** | 11.236 | 7.116 | **8750.7** | 479.2 | 645.7 | 910.5 |
| **Text & Structured Geomean (8)** | **0.0442** | 0.1562 | 0.0817 | 1.022 | **1.508** | 1.168 | 6.127 | **9.258** | 2.701 | **56.6** | 23.9 | 21.7 | 45.7 |
| **2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0101 | 8.586 | **11.481** | 9.590 | 14.428 | **14.977** | 5.845 | **4567.1** | 735.2 | 737.9 | 1292.3 |
| **All-Dataset Geomean (10)** | **0.0248** | 0.1012 | 0.0538 | 1.564 | **2.263** | 1.780 | 7.271 | **10.193** | 3.152 | **136.2** | 47.5 | 44.0 | 89.2 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **86.1** | 18.2 | 18.7 | 11.4 | **6.71** | 4.04 | 2.31 |
| **`events.tsv`** | **108.8** | 13.3 | 25.0 | 18.7 | **7.28** | 3.39 | 2.92 |
| **`app-logs.logfmt`** | **108.1** | 17.3 | 25.7 | 12.9 | **7.44** | 3.92 | 2.43 |
| **`json-records.ndjson`** | **222.9** | 32.0 | 43.6 | 19.9 | **10.40** | 5.35 | 3.02 |
| **`field-order-optional-escaping.ndjson`** | **54.8** | 11.9 | 13.2 | 8.3 | **5.11** | 3.11 | 2.00 |
| **`event-mix-identifiers.ndjson`** | **136.7** | 21.4 | 22.4 | 13.5 | **7.74** | 4.13 | 2.50 |
| **`bursty-variable-records.ndjson`** | **42.3** | 11.8 | 10.5 | 7.7 | **4.41** | 3.06 | 1.93 |
| **`source-and-docs.txt`** | 1.5 | **3.5** | 2.7 | 2.5 | 0.82 | **1.72** | 1.13 |
| **`ui-render-32bpp.bmp`** | **2383.7** | 1127.9 | 612.8 | 535.0 | **30.07** | 26.13 | 14.25 |
| **`rgb-frame-24bpp.bmp`** | **8750.7** | 459.7 | 589.6 | 232.6 | **45.63** | 16.87 | 9.09 |
| **Text & Structured Geomean (8)** | **56.6** | 13.9 | 16.0 | 10.2 | **5.21** | 3.44 | 2.19 |
| **2D BMP Geomean (2)** | **4567.1** | 720.1 | 601.0 | 352.8 | **37.04** | 20.99 | 11.38 |
| **All-Dataset Geomean (10)** | **136.2** | 30.6 | 33.1 | 20.8 | **7.72** | 4.94 | 3.05 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 3.216 | **1.573** | 8.082 | 0.675 | **0.633** | 2.575 | **58.4** | 18.2 | 2.5 |
| **`events.tsv`** | 3.383 | **2.085** | 7.335 | **0.664** | 0.855 | 2.003 | **72.6** | 9.9 | 4.9 |
| **`app-logs.logfmt`** | 3.394 | **1.597** | 7.601 | **0.607** | 0.625 | 2.456 | **75.3** | 17.4 | 3.0 |
| **`json-records.ndjson`** | 2.381 | **1.184** | 6.525 | **0.508** | 0.542 | 1.968 | **202.7** | 39.9 | 5.5 |
| **`field-order-optional-escaping.ndjson`** | 4.687 | **2.437** | 8.779 | **0.942** | 0.946 | 3.087 | **26.1** | 7.8 | 1.6 |
| **`event-mix-identifiers.ndjson`** | 3.130 | **1.676** | 7.024 | **0.664** | 0.799 | 2.534 | **94.8** | 18.5 | 3.2 |
| **`bursty-variable-records.ndjson`** | 4.911 | **2.262** | 8.789 | 1.246 | **1.014** | 3.117 | **17.1** | 7.8 | 1.5 |
| **`source-and-docs.txt`** | 35.605 | **5.434** | 16.919 | 3.628 | **1.137** | 3.757 | 0.1 | **1.4** | 0.3 |
| **`ui-render-32bpp.bmp`** | 0.602 | **0.345** | 0.694 | 0.352 | **0.207** | 0.568 | **5174.9** | 4227.0 | 852.1 |
| **`rgb-frame-24bpp.bmp`** | 0.674 | **0.385** | 2.003 | 0.366 | **0.316** | 0.671 | **17619.1** | 1318.4 | 200.7 |
