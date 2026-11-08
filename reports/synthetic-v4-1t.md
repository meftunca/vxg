## VXG v4 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row (lowest block ratio, highest encode `GB/s`, highest decode `GB/s`).

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0950 | 0.0868 | 0.0014 | 1.572 | **1.824** | 1.114 | 0.595 | **8.690** | 7.389 | 2.177 | 1.965 | 47.9 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0710 | 0.0680 | 0.0008 | **1.564** | 1.528 | 1.331 | 0.651 | **9.199** | 5.820 | 2.641 | 2.693 | 39.3 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0856 | 0.0731 | 0.0013 | **1.626** | 1.381 | 0.930 | 0.613 | **10.004** | 6.887 | 2.428 | 1.931 | 51.7 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0698 | 0.0544 | 0.0196 | 2.225 | **2.293** | 1.688 | 0.725 | **12.569** | 9.018 | 3.403 | 2.531 | 39.1 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.1161 | 0.0902 | 0.0667 | 1.131 | **1.314** | 0.964 | 0.504 | **5.320** | 4.695 | 1.937 | 1.676 | 49.1 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0854 | 0.0696 | 0.0517 | 1.500 | **1.845** | 1.254 | 0.635 | **8.032** | 5.453 | 2.346 | 1.911 | 40.1 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.1198 | 0.0888 | 0.0642 | 0.970 | **1.320** | 0.833 | 0.523 | **4.240** | 4.219 | 1.739 | 1.645 | 49.2 |
| **`source-and-docs.txt`** | 16 MiB | 0.3063 | 0.4231 | 0.3084 | 0.2673 | 0.2253 | **0.1982** | 0.0028 | 0.453 | **0.472** | 0.421 | 0.187 | 1.573 | **2.189** | 0.976 | 0.926 | 99.7 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0101 | 0.0087 | 0.0012 | 7.129 | **9.295** | 5.176 | 5.954 | 8.521 | **17.720** | 4.478 | 4.329 | 15.2 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0139 | 0.0141 | 0.0012 | 6.368 | **7.139** | 4.959 | 1.572 | **12.203** | 7.675 | 5.845 | 3.618 | 37.3 |
| **Text & Structured Geomean (8)** | — | **0.0451** | 0.2103 | 0.0957 | 0.0857 | 0.1009 | 0.0845 | 0.0080 | 1.267 | **1.382** | 0.998 | 0.521 | **6.414** | 5.309 | 2.092 | 1.831 | 49.6 |
| **2D BMP Geomean (2)** | — | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0118 | 0.0111 | 0.0012 | 6.738 | **8.146** | 5.066 | 3.060 | 10.197 | **11.662** | 5.116 | 3.958 | 23.8 |
| **All-Dataset Geomean (10)** | — | **0.0252** | 0.1284 | 0.0614 | 0.0561 | 0.0658 | 0.0563 | 0.0055 | 1.769 | **1.971** | 1.381 | 0.742 | **7.037** | 6.214 | 2.501 | 2.136 | 42.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | 1.572 | **1.824** | 1.255 | 8.690 | **10.768** | 2.743 | **90.7** | 26.8 | 23.3 | 55.6 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0680 | **1.564** | 1.528 | 1.468 | 9.199 | **11.765** | 3.027 | **116.3** | 28.0 | 31.0 | 62.4 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | 1.626 | **1.819** | 1.342 | 10.004 | **10.475** | 2.885 | **114.2** | 26.9 | 28.3 | 62.8 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | 2.225 | **2.337** | 2.007 | 12.569 | **14.351** | 4.296 | **233.4** | 50.5 | 60.2 | 118.8 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | 1.131 | **1.314** | 1.104 | 5.320 | **8.352** | 2.257 | **55.1** | 22.2 | 17.9 | 37.5 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0696 | 1.500 | **1.924** | 1.320 | 8.032 | **11.441** | 2.990 | **139.8** | 41.4 | 28.6 | 67.4 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0888 | 0.970 | **1.320** | 0.881 | 4.240 | **6.409** | 2.221 | **42.2** | 22.2 | 15.8 | 32.8 |
| **`source-and-docs.txt`** | 0.3063 | 0.3401 | **0.1982** | 0.453 | **0.472** | 0.421 | 1.573 | **3.603** | 1.611 | 2.8 | 3.8 | 4.2 | **6.6** |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | 7.129 | **9.933** | 9.703 | 8.521 | **17.720** | 4.478 | **1628.8** | 945.9 | 764.2 | 1538.1 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 6.368 | **7.212** | 6.487 | **12.203** | 8.287 | 5.943 | **7102.5** | 340.9 | 520.2 | 647.7 |
| **Text & Structured Geomean (8)** | **0.0451** | 0.1560 | 0.0817 | 1.267 | **1.442** | 1.131 | 6.414 | **8.973** | 2.657 | **63.1** | 23.1 | 21.2 | 44.0 |
| **2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0101 | 6.738 | **8.464** | 7.934 | 10.197 | **12.118** | 5.158 | **3401.3** | 567.8 | 630.5 | 998.1 |
| **All-Dataset Geomean (10)** | **0.0252** | 0.1011 | 0.0538 | 1.769 | **2.054** | 1.670 | 7.037 | **9.529** | 3.034 | **140.1** | 43.8 | 41.8 | 82.2 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **90.7** | 18.4 | 18.5 | 11.4 | **6.95** | 4.07 | 2.31 |
| **`events.tsv`** | **116.3** | 13.4 | 25.3 | 18.6 | **7.61** | 3.42 | 2.91 |
| **`app-logs.logfmt`** | **114.2** | 15.4 | 21.6 | 12.7 | **7.72** | 3.62 | 2.40 |
| **`json-records.ndjson`** | **233.4** | 30.4 | 44.2 | 19.4 | **10.73** | 5.17 | 2.97 |
| **`field-order-optional-escaping.ndjson`** | **55.1** | 11.6 | 13.1 | 7.9 | **5.13** | 3.07 | 1.94 |
| **`event-mix-identifiers.ndjson`** | **139.8** | 20.5 | 22.0 | 12.9 | **7.86** | 4.02 | 2.42 |
| **`bursty-variable-records.ndjson`** | **42.2** | 11.4 | 10.1 | 7.7 | **4.41** | 3.00 | 1.93 |
| **`source-and-docs.txt`** | **2.8** | 2.4 | 2.1 | 1.8 | 1.32 | **1.35** | 0.92 |
| **`ui-render-32bpp.bmp`** | **1628.8** | 915.0 | 528.1 | 502.2 | **23.33** | 22.73 | 13.66 |
| **`rgb-frame-24bpp.bmp`** | **7102.5** | 326.0 | 451.1 | 172.0 | **39.71** | 13.41 | 7.43 |
| **Text & Structured Geomean (8)** | **63.1** | 12.9 | 15.1 | 9.7 | **5.65** | 3.27 | 2.11 |
| **2D BMP Geomean (2)** | **3401.3** | 546.1 | 488.1 | 293.9 | **30.44** | 17.46 | 10.07 |
| **All-Dataset Geomean (10)** | **140.1** | 27.2 | 30.3 | 19.1 | **7.91** | 4.57 | 2.89 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 2.916 | **1.342** | 7.808 | 0.657 | **0.613** | 2.536 | **65.6** | 20.3 | 2.6 |
| **`events.tsv`** | 2.853 | **2.200** | 7.018 | **0.674** | 0.815 | 1.956 | **83.8** | 10.0 | 5.0 |
| **`app-logs.logfmt`** | 2.980 | **1.558** | 7.314 | **0.599** | 0.607 | 2.376 | **85.5** | 15.8 | 3.0 |
| **`json-records.ndjson`** | 2.089 | **1.187** | 6.615 | **0.505** | 0.514 | 1.808 | **227.2** | 38.9 | 5.6 |
| **`field-order-optional-escaping.ndjson`** | 4.154 | **2.409** | 8.214 | **0.942** | 0.950 | 2.722 | **27.9** | 7.7 | 1.7 |
| **`event-mix-identifiers.ndjson`** | 2.900 | **1.658** | 7.132 | **0.683** | 0.754 | 2.465 | **99.3** | 18.4 | 3.1 |
| **`bursty-variable-records.ndjson`** | 4.276 | **2.313** | 8.937 | 1.258 | **0.986** | 3.049 | **18.2** | 7.6 | 1.5 |
| **`source-and-docs.txt`** | 6.284 | **4.047** | 12.189 | 1.416 | **0.859** | 2.832 | 0.9 | **1.3** | 0.3 |
| **`ui-render-32bpp.bmp`** | 0.450 | **0.294** | 0.607 | 0.299 | **0.177** | 0.489 | **4440.1** | 4016.8 | 921.7 |
| **`rgb-frame-24bpp.bmp`** | 0.454 | **0.252** | 1.611 | 0.289 | **0.231** | 0.447 | **19627.8** | 1350.9 | 202.6 |
