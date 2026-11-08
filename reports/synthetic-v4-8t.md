## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0950 | 0.0868 | 0.0014 | 6.648 | **9.660** | 6.945 | 3.287 | **48.375** | 31.411 | 11.244 | 12.685 | 46.4 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0710 | 0.0680 | 0.0008 | 6.982 | **7.658** | 6.141 | 3.654 | **56.157** | 25.999 | 13.133 | 12.853 | 40.9 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0856 | 0.0731 | 0.0013 | 7.336 | **9.327** | 6.566 | 3.833 | **60.092** | 35.408 | 12.504 | 10.707 | 43.2 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0698 | 0.0544 | 0.0196 | 9.893 | **13.356** | 8.816 | 4.257 | **69.601** | 37.503 | 16.191 | 12.095 | 36.4 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.1161 | 0.0902 | 0.0667 | 5.523 | **7.382** | 4.945 | 2.797 | **32.666** | 26.519 | 11.862 | 8.328 | 48.3 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0854 | 0.0696 | 0.0517 | 6.465 | **9.696** | 7.320 | 3.655 | **48.273** | 31.453 | 14.111 | 9.597 | 37.2 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.1198 | 0.0888 | 0.0642 | 4.997 | **7.476** | 4.418 | 2.707 | **26.342** | 20.590 | 10.934 | 8.794 | 48.0 |
| **`source-and-docs.txt`** | 16 MiB | **0.0036** | 0.4286 | 0.3133 | 0.2716 | 0.2277 | 0.2007 | 0.0029 | **4.358** | 4.013 | 3.254 | 1.473 | **94.539** | 21.063 | 8.126 | 7.622 | 70.1 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0101 | 0.0087 | 0.0010 | 27.868 | **36.387** | 22.701 | 22.255 | **76.950** | 66.873 | 20.078 | 24.056 | 15.0 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0139 | 0.0141 | 0.0010 | 27.348 | **34.627** | 26.166 | 10.042 | **73.312** | 44.447 | 33.933 | 20.896 | 21.4 |
| **Text & Structured Geomean (8)** | — | **0.0259** | 0.2107 | 0.0959 | 0.0859 | 0.1011 | 0.0846 | 0.0081 | 6.343 | **8.167** | 5.802 | 3.075 | **50.838** | 28.132 | 12.050 | 10.155 | 45.4 |
| **2D BMP Geomean (2)** | — | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0118 | 0.0111 | 0.0010 | 27.607 | **35.496** | 24.372 | 14.949 | **75.109** | 54.519 | 26.101 | 22.420 | 17.9 |
| **All-Dataset Geomean (10)** | — | **0.0162** | 0.1286 | 0.0615 | 0.0562 | 0.0658 | 0.0563 | 0.0053 | 8.512 | **10.957** | 7.731 | 4.219 | **54.965** | 32.112 | 14.065 | 11.898 | 37.7 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | 6.648 | **9.719** | 7.802 | **48.375** | 40.847 | 15.263 | **440.3** | 120.4 | 136.9 | 249.9 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0680 | 6.982 | **8.974** | 7.332 | **56.157** | 47.520 | 17.985 | **606.9** | 136.3 | 168.9 | 303.7 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | 7.336 | **9.327** | 8.145 | **60.092** | 54.963 | 13.616 | **594.8** | 139.5 | 151.5 | 325.6 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | 9.893 | **13.393** | 9.541 | **69.601** | 55.745 | 20.097 | **1158.1** | 238.1 | 284.0 | 560.3 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | 5.523 | **8.183** | 4.945 | 32.666 | **32.982** | 13.114 | **301.9** | 110.1 | 91.2 | 186.0 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0696 | 6.465 | **11.105** | 7.320 | **48.273** | 42.341 | 17.102 | **711.5** | 191.2 | 160.8 | 311.7 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0888 | 4.997 | **7.728** | 5.208 | 26.342 | **34.165** | 11.764 | **238.9** | 124.2 | 88.2 | 183.1 |
| **`source-and-docs.txt`** | **0.0036** | 0.3453 | 0.2007 | **4.358** | 4.013 | 3.254 | **94.539** | 21.922 | 10.564 | **5587.7** | 27.2 | 29.2 | 46.7 |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | 27.868 | **36.387** | 35.765 | **76.950** | 66.873 | 26.612 | **9677.3** | 3516.8 | 3576.7 | 5718.9 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 27.348 | **34.627** | 27.465 | **73.312** | 44.447 | 33.933 | **36078.4** | 1729.9 | 2557.6 | 3286.8 |
| **Text & Structured Geomean (8)** | **0.0259** | 0.1563 | 0.0818 | 6.343 | **8.628** | 6.375 | **50.838** | 39.740 | 14.626 | **692.3** | 118.5 | 118.0 | 226.3 |
| **2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0101 | 27.607 | **35.496** | 31.341 | **75.109** | 54.519 | 30.050 | **18685.4** | 2466.5 | 3024.5 | 4335.5 |
| **All-Dataset Geomean (10)** | **0.0162** | 0.1013 | 0.0539 | 8.512 | **11.449** | 8.766 | **54.965** | 42.334 | 16.891 | **1338.3** | 217.4 | 225.7 | 408.4 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **440.3** | 87.4 | 105.1 | 68.0 | **19.91** | 11.50 | 7.60 |
| **`events.tsv`** | **606.9** | 63.5 | 121.3 | 96.5 | **22.90** | 9.64 | 8.71 |
| **`app-logs.logfmt`** | **594.8** | 90.6 | 130.3 | 74.9 | **23.20** | 11.81 | 7.83 |
| **`json-records.ndjson`** | **1158.1** | 149.5 | 220.3 | 102.8 | **31.21** | 14.96 | 9.04 |
| **`field-order-optional-escaping.ndjson`** | **301.9** | 65.6 | 73.4 | 41.6 | **15.95** | 9.72 | 5.85 |
| **`event-mix-identifiers.ndjson`** | **711.5** | 113.1 | 130.5 | 69.4 | **23.25** | 12.55 | 7.43 |
| **`bursty-variable-records.ndjson`** | **238.9** | 60.1 | 58.1 | 40.7 | **14.00** | 9.07 | 5.84 |
| **`source-and-docs.txt`** | **5587.7** | 21.5 | 16.4 | 14.7 | **48.40** | 5.82 | 3.67 |
| **`ui-render-32bpp.bmp`** | **9677.3** | 3516.8 | 2342.0 | 2288.9 | **76.53** | 55.77 | 37.55 |
| **`rgb-frame-24bpp.bmp`** | **36078.4** | 1727.6 | 2496.4 | 1044.5 | **117.34** | 40.77 | 24.73 |
| **Text & Structured Geomean (8)** | **692.3** | 71.9 | 87.2 | 55.3 | **23.17** | 10.29 | 6.76 |
| **2D BMP Geomean (2)** | **18685.4** | 2464.9 | 2418.0 | 1546.2 | **94.76** | 47.69 | 30.48 |
| **All-Dataset Geomean (10)** | **1338.3** | 145.9 | 169.5 | 107.6 | **30.70** | 13.99 | 9.14 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 2.698 | **1.314** | 7.078 | 0.649 | **0.569** | 2.362 | **332.7** | 101.0 | 16.6 |
| **`events.tsv`** | 2.845 | **1.816** | 6.731 | **0.636** | 0.723 | 1.790 | **451.3** | 55.4 | 27.8 |
| **`app-logs.logfmt`** | 3.009 | **1.442** | 7.099 | **0.534** | 0.610 | 2.222 | **469.5** | 96.5 | 18.8 |
| **`json-records.ndjson`** | 2.161 | **1.168** | 5.771 | **0.413** | 0.496 | 1.719 | **1225.3** | 196.4 | 32.6 |
| **`field-order-optional-escaping.ndjson`** | 4.163 | **2.252** | 8.147 | 0.906 | **0.900** | 2.637 | **155.5** | 46.1 | 9.0 |
| **`event-mix-identifiers.ndjson`** | 2.701 | **1.557** | 6.505 | **0.626** | 0.763 | 2.269 | **547.2** | 103.7 | 18.1 |
| **`bursty-variable-records.ndjson`** | 4.234 | **2.108** | 8.033 | 1.150 | **0.929** | 2.759 | **108.3** | 42.9 | 8.7 |
| **`source-and-docs.txt`** | **1.175** | 5.133 | 15.424 | **0.123** | 1.100 | 3.536 | **14705.6** | 9.0 | 2.0 |
| **`ui-render-32bpp.bmp`** | 0.587 | **0.440** | 0.729 | 0.319 | **0.227** | 0.566 | **22360.1** | 11119.2 | 3564.0 |
| **`rgb-frame-24bpp.bmp`** | 0.679 | **0.475** | 1.892 | 0.356 | **0.320** | 0.641 | **73370.6** | 4429.2 | 948.5 |
