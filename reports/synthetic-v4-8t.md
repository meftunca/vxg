## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | 0.0797 | 0.0950 | 0.0868 | 0.0014 | 6.463 | **8.874** | 6.188 | 2.904 | **41.768** | 28.426 | 12.099 | 10.731 | 47.9 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.0741 | 0.0683 | 0.0710 | 0.0680 | 0.0008 | 5.955 | **7.783** | 6.493 | 3.702 | **53.119** | 25.627 | 11.629 | 13.771 | 45.3 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | 0.0722 | 0.0856 | 0.0731 | 0.0013 | **7.315** | 6.481 | 5.651 | 3.496 | **54.775** | 25.866 | 12.372 | 9.403 | 47.0 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | 0.0488 | 0.0698 | 0.0544 | 0.0196 | 9.512 | **11.343** | 7.490 | 3.389 | **58.922** | 35.969 | 13.656 | 12.586 | 41.0 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | 0.0883 | 0.1161 | 0.0902 | 0.0667 | 4.917 | **8.045** | 5.821 | 3.118 | **30.184** | 24.391 | 9.123 | 8.602 | 46.5 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | 0.0698 | 0.0854 | 0.0696 | 0.0517 | 6.410 | **10.173** | 5.812 | 3.403 | **48.296** | 26.582 | 10.440 | 10.539 | 37.3 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | 0.0926 | 0.1198 | 0.0888 | 0.0642 | 4.956 | **6.667** | 4.964 | 2.462 | **21.545** | 20.293 | 8.816 | 7.616 | 48.8 |
| **`source-and-docs.txt`** | 16 MiB | 0.2592 | 0.4259 | 0.3107 | 0.2694 | 0.2261 | **0.1986** | 0.0029 | 0.826 | **3.856** | 3.042 | 1.506 | 7.967 | **19.660** | 6.571 | 6.827 | 71.3 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | 0.0086 | 0.0101 | 0.0087 | 0.0010 | 29.456 | **32.942** | 20.850 | 23.192 | **81.515** | 56.718 | 21.635 | 24.405 | 16.8 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | 0.0122 | 0.0139 | 0.0141 | 0.0010 | **27.506** | 27.277 | 23.538 | 9.679 | **69.960** | 42.201 | 32.636 | 20.231 | 21.3 |
| **Text & Structured Geomean (8)** | — | **0.0442** | 0.2105 | 0.0958 | 0.0858 | 0.1010 | 0.0845 | 0.0081 | 4.924 | **7.561** | 5.525 | 2.901 | **34.014** | 25.438 | 10.345 | 9.762 | 47.4 |
| **2D BMP Geomean (2)** | — | **0.0024** | 0.0178 | 0.0104 | 0.0103 | 0.0118 | 0.0111 | 0.0010 | 28.464 | **29.976** | 22.153 | 14.983 | **75.517** | 48.924 | 26.572 | 22.220 | 18.9 |
| **All-Dataset Geomean (10)** | — | **0.0248** | 0.1285 | 0.0615 | 0.0561 | 0.0658 | 0.0563 | 0.0053 | 6.993 | **9.959** | 7.294 | 4.028 | **39.897** | 28.992 | 12.493 | 11.507 | 39.4 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | 6.463 | **9.023** | 6.188 | **41.768** | 35.636 | 15.226 | **403.4** | 108.4 | 121.7 | 224.9 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0680 | 5.955 | 7.783 | **7.943** | **53.119** | 41.877 | 15.298 | **545.1** | 119.2 | 162.1 | 268.2 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | 7.315 | **9.479** | 7.280 | **54.775** | 44.010 | 14.582 | **567.1** | 125.9 | 148.2 | 293.8 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | 9.512 | **12.512** | 8.717 | **58.922** | 44.187 | 20.988 | **1044.8** | 204.9 | 277.4 | 482.2 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | 4.917 | **8.045** | 5.963 | 30.184 | **36.409** | 12.806 | **273.9** | 114.7 | 98.9 | 193.8 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0696 | 6.410 | **10.173** | 7.368 | **48.296** | 44.023 | 14.718 | **708.6** | 186.6 | 149.7 | 304.2 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0888 | 4.956 | **7.571** | 4.964 | 21.545 | **25.372** | 12.347 | **215.2** | 105.9 | 88.2 | 156.2 |
| **`source-and-docs.txt`** | 0.2592 | 0.3424 | **0.1986** | 0.826 | **3.856** | 3.117 | 7.967 | **23.850** | 8.913 | 9.9 | 28.0 | 26.5 | **48.3** |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | 29.456 | **32.942** | 29.169 | **81.515** | 63.044 | 24.405 | **10240.1** | 3249.0 | 3093.2 | 5283.4 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | 27.506 | **28.565** | 24.092 | **69.960** | 44.639 | 32.636 | **35345.9** | 1574.6 | 2349.2 | 2991.7 |
| **Text & Structured Geomean (8)** | **0.0442** | 0.1562 | 0.0817 | 4.924 | **8.173** | 6.177 | 34.014 | **35.990** | 14.003 | **292.7** | 109.8 | 113.8 | 210.1 |
| **2D BMP Geomean (2)** | **0.0024** | 0.0178 | 0.0101 | 28.464 | **30.676** | 26.509 | **75.517** | 53.049 | 28.222 | **19024.8** | 2261.8 | 2695.7 | 3975.7 |
| **All-Dataset Geomean (10)** | **0.0248** | 0.1012 | 0.0538 | 6.993 | **10.648** | 8.266 | **39.897** | 38.894 | 16.110 | **674.5** | 201.1 | 214.3 | 378.3 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **403.4** | 79.7 | 102.9 | 58.8 | **18.78** | 10.82 | 6.90 |
| **`events.tsv`** | **545.1** | 63.5 | 117.3 | 100.5 | **21.32** | 9.64 | 8.95 |
| **`app-logs.logfmt`** | **567.1** | 64.5 | 120.3 | 67.0 | **22.47** | 9.42 | 7.27 |
| **`json-records.ndjson`** | **1044.8** | 134.9 | 186.4 | 93.6 | **29.14** | 13.97 | 8.49 |
| **`field-order-optional-escaping.ndjson`** | **273.9** | 65.7 | 69.9 | 44.6 | **14.94** | 9.73 | 6.14 |
| **`event-mix-identifiers.ndjson`** | **708.6** | 106.5 | 100.0 | 70.2 | **23.19** | 12.05 | 7.49 |
| **`bursty-variable-records.ndjson`** | **215.2** | 56.3 | 55.3 | 36.2 | **13.05** | 8.68 | 5.39 |
| **`source-and-docs.txt`** | 9.9 | **20.4** | 14.4 | 14.2 | 2.94 | **5.63** | 3.57 |
| **`ui-render-32bpp.bmp`** | **10240.1** | 3081.7 | 2329.9 | 2353.4 | **79.46** | 51.07 | 38.26 |
| **`rgb-frame-24bpp.bmp`** | **35345.9** | 1494.1 | 2322.1 | 1009.0 | **115.74** | 37.01 | 24.17 |
| **Text & Structured Geomean (8)** | **292.7** | 65.9 | 78.9 | 52.7 | **15.59** | 9.70 | 6.55 |
| **2D BMP Geomean (2)** | **19024.8** | 2145.7 | 2326.0 | 1541.0 | **95.90** | 43.48 | 30.41 |
| **All-Dataset Geomean (10)** | **674.5** | 132.2 | 155.3 | 103.5 | **22.42** | 13.10 | 8.90 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 2.708 | **1.246** | 6.906 | 0.568 | **0.564** | 2.344 | **325.2** | 95.1 | 14.6 |
| **`events.tsv`** | 2.777 | **1.847** | 6.341 | **0.526** | 0.764 | 1.671 | **451.2** | 53.5 | 30.9 |
| **`app-logs.logfmt`** | 3.061 | **1.366** | 6.485 | **0.441** | 0.552 | 1.930 | **487.8** | 74.3 | 18.9 |
| **`json-records.ndjson`** | 2.201 | **1.102** | 5.249 | **0.362** | 0.500 | 1.646 | **1171.0** | 181.8 | 31.8 |
| **`field-order-optional-escaping.ndjson`** | 4.052 | **2.284** | 8.452 | 0.917 | **0.894** | 2.671 | **142.1** | 46.0 | 9.4 |
| **`event-mix-identifiers.ndjson`** | 2.708 | **1.628** | 6.219 | **0.564** | 0.746 | 2.219 | **573.1** | 96.6 | 18.9 |
| **`bursty-variable-records.ndjson`** | 4.330 | **2.041** | 7.262 | 1.048 | **0.968** | 2.374 | **101.0** | 40.1 | 8.7 |
| **`source-and-docs.txt`** | 33.037 | **5.125** | 13.968 | 3.033 | **1.105** | 3.102 | 1.0 | **8.6** | 2.2 |
| **`ui-render-32bpp.bmp`** | 0.608 | **0.429** | 0.769 | 0.295 | **0.218** | 0.581 | **24157.4** | 10075.9 | 3522.1 |
| **`rgb-frame-24bpp.bmp`** | 0.689 | **0.425** | 1.697 | **0.313** | 0.322 | 0.607 | **76111.4** | 4038.5 | 994.4 |
