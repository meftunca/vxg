## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0950 | 0.0868 | 0.0014 | 1.438 | **1.805** | 1.120 | 0.604 | **7.935** | 7.245 | 2.161 | 1.909 | 46.8 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0710 | 0.0680 | 0.0008 | 1.459 | **1.511** | 1.313 | 0.676 | **8.642** | 5.828 | 2.593 | 2.660 | 39.5 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0856 | 0.0731 | 0.0013 | 1.512 | **1.738** | 1.270 | 0.606 | **9.012** | 7.028 | 2.464 | 2.004 | 50.2 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0698 | 0.0544 | 0.0196 | 2.077 | **2.455** | 1.695 | 0.757 | **11.911** | 9.366 | 3.380 | 2.608 | 35.7 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.1161 | 0.0902 | 0.0667 | 1.139 | **1.377** | 0.982 | 0.517 | **5.347** | 4.887 | 1.953 | 1.729 | 46.4 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0854 | 0.0696 | 0.0517 | 1.465 | **1.962** | 1.251 | 0.675 | **7.914** | 5.706 | 2.373 | 1.974 | 39.5 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.1198 | 0.0888 | 0.0642 | 0.953 | **1.367** | 0.875 | 0.540 | 3.984 | **4.401** | 1.736 | 1.668 | 47.3 |
| **`source-and-docs.txt`** | 16 MiB | **0.0036** | 0.4286 | 0.3133 | 0.2716 | 0.2277 | 0.2007 | 0.0029 | **3.620** | 0.635 | 0.509 | 0.264 | **62.686** | 3.701 | 1.365 | 1.204 | 71.5 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0101 | 0.0087 | 0.0012 | 9.041 | **12.657** | 6.528 | 6.492 | 14.588 | **19.655** | 4.801 | 4.547 | 14.2 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0139 | 0.0141 | 0.0012 | 8.355 | **10.698** | 7.069 | 2.256 | **14.792** | 10.397 | 7.120 | 4.554 | 20.4 |
| **Text & Structured Geomean (8)** | — | **0.0259** | 0.2107 | 0.0959 | 0.0859 | 0.1011 | 0.0846 | 0.0081 | **1.575** | 1.512 | 1.071 | 0.557 | **9.691** | 5.789 | 2.181 | 1.916 | 46.1 |
| **2D BMP Geomean (2)** | — | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0118 | 0.0111 | 0.0012 | 8.691 | **11.636** | 6.793 | 3.827 | **14.690** | 14.295 | 5.847 | 4.550 | 17.0 |
| **All-Dataset Geomean (10)** | — | **0.0162** | 0.1286 | 0.0615 | 0.0562 | 0.0658 | 0.0563 | 0.0056 | 2.217 | **2.274** | 1.550 | 0.819 | **10.532** | 6.937 | 2.657 | 2.278 | 37.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | 1.438 | **1.805** | 1.253 | 7.935 | **10.391** | 2.696 | **82.9** | 26.2 | 23.0 | 54.3 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0680 | 1.459 | **1.511** | 1.458 | 8.642 | **11.731** | 3.117 | **108.8** | 27.8 | 31.4 | 61.9 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | 1.512 | **1.738** | 1.350 | 9.012 | **10.573** | 2.806 | **104.6** | 26.4 | 28.0 | 61.7 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | 2.077 | **2.455** | 1.992 | 11.911 | **13.945** | 4.249 | **219.5** | 51.0 | 59.7 | 120.0 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | 1.139 | **1.377** | 1.135 | 5.347 | **8.690** | 2.312 | **55.5** | 23.2 | 18.3 | 39.2 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0696 | 1.465 | **1.962** | 1.371 | 7.914 | **11.797** | 2.976 | **137.1** | 42.4 | 29.0 | 69.1 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0888 | 0.953 | **1.385** | 0.902 | 3.984 | **6.727** | 2.363 | **40.6** | 23.3 | 16.5 | 34.4 |
| **`source-and-docs.txt`** | **0.0036** | 0.3453 | 0.2007 | **3.620** | 0.635 | 0.546 | **62.686** | 4.683 | 1.703 | **4147.4** | 5.0 | 4.8 | 8.6 |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | 9.041 | **12.838** | 11.259 | 14.588 | **19.933** | 4.801 | **2399.9** | 1140.5 | 852.4 | 1854.6 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 8.355 | **10.797** | 8.347 | **14.792** | 10.968 | 7.120 | **8957.4** | 479.9 | 645.9 | 911.7 |
| **Text & Structured Geomean (8)** | **0.0259** | 0.1563 | 0.0818 | **1.575** | 1.514 | 1.180 | **9.691** | 9.342 | 2.694 | **150.6** | 24.1 | 21.8 | 46.0 |
| **2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0101 | 8.691 | **11.773** | 9.694 | 14.690 | **14.786** | 5.847 | **4636.5** | 739.8 | 742.0 | 1300.3 |
| **All-Dataset Geomean (10)** | **0.0162** | 0.1013 | 0.0539 | 2.217 | **2.282** | 1.798 | **10.532** | 10.241 | 3.146 | **299.0** | 47.7 | 44.1 | 89.7 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **82.9** | 18.1 | 18.5 | 11.3 | **6.54** | 4.03 | 2.30 |
| **`events.tsv`** | **108.8** | 13.3 | 24.9 | 18.9 | **7.28** | 3.41 | 2.94 |
| **`app-logs.logfmt`** | **104.6** | 17.4 | 25.4 | 12.9 | **7.28** | 3.93 | 2.42 |
| **`json-records.ndjson`** | **219.5** | 32.0 | 44.1 | 20.1 | **10.30** | 5.36 | 3.05 |
| **`field-order-optional-escaping.ndjson`** | **55.5** | 12.2 | 13.3 | 8.1 | **5.15** | 3.16 | 1.97 |
| **`event-mix-identifiers.ndjson`** | **137.1** | 21.7 | 22.1 | 13.5 | **7.76** | 4.17 | 2.50 |
| **`bursty-variable-records.ndjson`** | **40.6** | 11.9 | 10.3 | 7.9 | **4.29** | 3.08 | 1.96 |
| **`source-and-docs.txt`** | **4147.4** | 3.6 | 2.7 | 2.5 | **39.68** | 1.76 | 1.12 |
| **`ui-render-32bpp.bmp`** | **2399.9** | 1124.5 | 614.1 | 537.4 | **30.21** | 26.08 | 14.29 |
| **`rgb-frame-24bpp.bmp`** | **8957.4** | 464.4 | 594.3 | 231.1 | **46.35** | 16.98 | 9.05 |
| **Text & Structured Geomean (8)** | **150.6** | 14.0 | 15.9 | 10.2 | **8.38** | 3.46 | 2.19 |
| **2D BMP Geomean (2)** | **4636.5** | 722.7 | 604.2 | 352.5 | **37.42** | 21.04 | 11.37 |
| **All-Dataset Geomean (10)** | **299.0** | 30.9 | 33.0 | 20.8 | **11.30** | 4.97 | 3.05 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 3.164 | **1.464** | 7.796 | 0.720 | **0.619** | 2.347 | **54.9** | 19.1 | 2.6 |
| **`events.tsv`** | 3.229 | **2.046** | 7.478 | **0.691** | 0.831 | 1.942 | **72.8** | 10.2 | 5.0 |
| **`app-logs.logfmt`** | 3.340 | **1.572** | 7.753 | **0.615** | 0.640 | 2.447 | **72.9** | 17.4 | 3.0 |
| **`json-records.ndjson`** | 2.330 | **1.166** | 6.592 | **0.520** | 0.532 | 1.932 | **199.3** | 40.7 | 5.6 |
| **`field-order-optional-escaping.ndjson`** | 4.857 | **2.300** | 8.855 | 0.976 | **0.941** | 3.064 | **25.5** | 8.3 | 1.6 |
| **`event-mix-identifiers.ndjson`** | 3.155 | **1.626** | 7.300 | **0.699** | 0.794 | 2.509 | **92.3** | 19.1 | 3.2 |
| **`bursty-variable-records.ndjson`** | 4.792 | **2.448** | 8.804 | 1.268 | **1.045** | 3.134 | **16.5** | 7.4 | 1.5 |
| **`source-and-docs.txt`** | **1.131** | 5.776 | 17.343 | **0.103** | 1.170 | 3.835 | **12146.2** | 1.4 | 0.3 |
| **`ui-render-32bpp.bmp`** | 0.602 | **0.353** | 0.720 | 0.355 | **0.207** | 0.574 | **5193.6** | 4158.1 | 835.8 |
| **`rgb-frame-24bpp.bmp`** | 0.698 | **0.384** | 2.044 | 0.367 | **0.317** | 0.685 | **17683.8** | 1332.6 | 195.3 |
