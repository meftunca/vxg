## VXG v4 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row (lowest block ratio, highest encode `GB/s`, highest decode `GB/s`).

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0950 | 0.0868 | 0.0014 | **3.159** | 2.904 | 1.736 | 1.079 | **8.502** | 7.064 | 3.036 | 2.992 | 71.2 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0710 | 0.0680 | 0.0008 | **2.813** | 2.219 | 1.783 | 1.211 | **7.287** | 5.952 | 3.744 | 3.349 | 97.6 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0856 | 0.0731 | 0.0013 | **2.735** | 2.391 | 2.136 | 1.197 | **8.722** | 6.240 | 3.159 | 3.004 | 89.5 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0698 | 0.0544 | 0.0196 | 3.475 | **3.848** | 2.647 | 1.107 | **10.676** | 8.637 | 4.593 | 3.663 | 62.9 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.1161 | 0.0902 | 0.0667 | 1.997 | **2.299** | 1.844 | 1.060 | 6.013 | **6.240** | 3.015 | 2.410 | 97.2 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0854 | 0.0696 | 0.0517 | 6.972 | **7.177** | 5.594 | 2.989 | **33.323** | 16.002 | 11.442 | 8.964 | 43.7 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.1198 | 0.0888 | 0.0642 | **2.344** | 1.923 | 1.478 | 1.445 | 4.921 | **5.614** | 3.307 | 3.975 | 85.6 |
| **`source-and-docs.txt`** | 16 MiB | 0.3063 | 0.4231 | 0.3084 | 0.2673 | 0.2253 | **0.1982** | 0.0028 | **3.639** | 3.011 | 2.847 | 1.090 | **16.103** | 15.963 | 6.344 | 5.463 | 76.1 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0101 | 0.0087 | 0.0010 | **51.315** | 29.285 | 24.433 | 20.840 | **68.748** | 49.915 | 22.746 | 17.282 | 14.8 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0139 | 0.0141 | 0.0010 | **34.974** | 28.769 | 17.412 | 8.475 | **48.861** | 40.335 | 19.295 | 15.485 | 21.5 |
| **Text & Structured Geomean (8)** | — | **0.0451** | 0.2103 | 0.0957 | 0.0857 | 0.1009 | 0.0845 | 0.0080 | **3.164** | 2.945 | 2.291 | 1.309 | **9.891** | 8.192 | 4.311 | 3.883 | 75.7 |
| **2D BMP Geomean (2)** | — | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0118 | 0.0111 | 0.0010 | **42.364** | 29.026 | 20.626 | 13.290 | **57.958** | 44.870 | 20.950 | 16.359 | 17.8 |
| **All-Dataset Geomean (10)** | — | **0.0252** | 0.1284 | 0.0614 | 0.0561 | 0.0658 | 0.0563 | 0.0053 | **5.315** | 4.654 | 3.556 | 2.081 | **14.087** | 11.510 | 5.914 | 5.178 | 56.7 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | **3.159** | 2.904 | 1.764 | **8.502** | 7.319 | 3.659 | **127.2** | 27.9 | 31.9 | 57.8 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0680 | **2.813** | 2.604 | 2.279 | 7.287 | **8.525** | 4.205 | **138.8** | 31.1 | 45.5 | 69.3 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | 2.735 | **2.982** | 2.136 | 8.722 | **10.226** | 4.221 | **138.4** | 34.0 | 43.2 | 79.4 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | 3.475 | **3.848** | 2.680 | **10.676** | 9.591 | 5.305 | **268.8** | 52.9 | 77.3 | 124.6 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | 1.997 | **2.660** | 2.045 | 6.013 | **8.813** | 3.857 | **77.9** | 32.5 | 31.8 | 54.8 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0696 | 6.972 | **7.177** | 6.397 | 33.323 | **42.812** | 13.304 | **613.9** | 154.6 | 132.6 | 252.0 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0888 | 2.344 | **2.722** | 1.831 | 4.921 | **6.476** | 5.512 | **70.7** | 32.1 | 35.8 | 47.3 |
| **`source-and-docs.txt`** | 0.3063 | 0.3401 | **0.1982** | **3.639** | 3.550 | 3.511 | 16.103 | **24.676** | 8.407 | 25.0 | 27.5 | 27.4 | **47.2** |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | **51.315** | 29.285 | 31.201 | **68.748** | 55.556 | 25.570 | **12412.2** | 2875.7 | 3274.7 | 4826.9 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **34.974** | 28.769 | 17.412 | **48.861** | 40.335 | 23.728 | **33308.1** | 1502.1 | 1702.9 | 2853.9 |
| **Text & Structured Geomean (8)** | **0.0451** | 0.1560 | 0.0817 | 3.164 | **3.354** | 2.575 | 9.891 | **11.773** | 5.485 | **123.9** | 40.3 | 46.0 | 76.9 |
| **2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0101 | **42.364** | 29.026 | 23.309 | **57.958** | 47.337 | 24.632 | **20332.9** | 2078.3 | 2361.5 | 3711.5 |
| **All-Dataset Geomean (10)** | **0.0252** | 0.1011 | 0.0538 | **5.315** | 5.164 | 4.001 | 14.087 | **15.550** | 7.407 | **343.6** | 88.6 | 101.1 | 167.0 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **127.2** | 22.7 | 27.3 | 18.9 | **8.70** | 4.69 | 3.24 |
| **`events.tsv`** | **138.8** | 16.3 | 34.9 | 28.3 | **8.56** | 3.90 | 3.85 |
| **`app-logs.logfmt`** | **138.4** | 19.2 | 37.4 | 22.2 | **8.78** | 4.20 | 3.48 |
| **`json-records.ndjson`** | **268.8** | 38.5 | 64.3 | 28.8 | **11.79** | 6.06 | 3.87 |
| **`field-order-optional-escaping.ndjson`** | **77.9** | 17.8 | 22.6 | 13.8 | **6.46** | 4.07 | 2.80 |
| **`event-mix-identifiers.ndjson`** | **613.9** | 69.4 | 102.7 | 60.6 | **21.07** | 9.06 | 6.80 |
| **`bursty-variable-records.ndjson`** | **70.7** | 15.9 | 18.5 | 20.0 | **6.22** | 3.74 | 3.63 |
| **`source-and-docs.txt`** | **25.0** | 16.4 | 13.8 | 10.8 | **5.76** | 4.84 | 2.98 |
| **`ui-render-32bpp.bmp`** | **12412.2** | 2725.7 | 2586.1 | 1877.4 | **90.34** | 47.06 | 32.91 |
| **`rgb-frame-24bpp.bmp`** | **33308.1** | 1500.1 | 1535.7 | 826.0 | **111.25** | 37.11 | 21.15 |
| **Text & Structured Geomean (8)** | **123.9** | 23.4 | 32.8 | 22.3 | **8.85** | 4.86 | 3.69 |
| **2D BMP Geomean (2)** | **20332.9** | 2022.1 | 1992.8 | 1245.3 | **100.25** | 41.79 | 26.38 |
| **All-Dataset Geomean (10)** | **343.6** | 57.0 | 74.6 | 49.9 | **14.38** | 7.47 | 5.47 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 2.150 | **0.909** | 5.325 | 0.519 | **0.308** | 1.481 | **120.4** | 43.0 | 6.7 |
| **`events.tsv`** | 1.967 | **1.238** | 5.456 | **0.420** | 0.451 | 1.259 | **152.7** | 21.9 | 10.8 |
| **`app-logs.logfmt`** | 1.875 | **1.115** | 5.685 | 0.383 | **0.320** | 1.598 | **163.3** | 32.2 | 7.4 |
| **`json-records.ndjson`** | 1.501 | **0.740** | 4.531 | 0.369 | **0.182** | 1.269 | **361.0** | 105.1 | 12.0 |
| **`field-order-optional-escaping.ndjson`** | 3.527 | **1.543** | 5.971 | 0.752 | **0.534** | 1.847 | **47.8** | 19.6 | 4.1 |
| **`event-mix-identifiers.ndjson`** | 2.343 | **1.340** | 5.854 | **0.566** | 0.636 | 2.057 | **533.3** | 75.2 | 17.5 |
| **`bursty-variable-records.ndjson`** | 3.260 | **1.336** | 6.873 | 0.785 | **0.497** | 2.406 | **44.2** | 19.5 | 4.9 |
| **`source-and-docs.txt`** | 6.757 | **4.759** | 13.865 | 1.592 | **0.967** | 3.032 | 7.6 | **7.6** | 1.7 |
| **`ui-render-32bpp.bmp`** | 0.521 | **0.396** | 0.705 | 0.333 | **0.206** | 0.505 | **29799.4** | 9543.5 | 3146.2 |
| **`rgb-frame-24bpp.bmp`** | 0.544 | **0.437** | 1.790 | 0.328 | **0.303** | 0.603 | **78814.4** | 4123.4 | 795.3 |
