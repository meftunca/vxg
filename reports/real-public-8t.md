## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | 2.904 | **2.918** | 2.711 | 1.159 | **13.429** | 13.150 | 6.357 | 4.683 | 84.1 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **8.382** | 6.592 | 5.096 | 2.688 | **34.405** | 18.814 | 7.351 | 7.571 | 52.1 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1350 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 1.481 | **5.680** | 4.908 | 2.295 | 8.592 | **23.268** | 10.702 | 9.421 | 60.3 |
| **`apache-access.clf`** | 2.4 MB | 0.0990 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 1.119 | **5.326** | 4.787 | 2.691 | 8.197 | **13.850** | 7.575 | 6.855 | 44.0 |
| **`macos-install.log`** | 34.2 MB | 0.0605 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0278 | 3.507 | **12.180** | 7.187 | 4.096 | 15.206 | **34.371** | 19.865 | 18.375 | 37.9 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3977 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.396 | **2.749** | 1.875 | 0.754 | 7.217 | **14.219** | 6.163 | 4.615 | 97.5 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.774 | **4.400** | 2.903 | 0.605 | 9.477 | **18.720** | 5.510 | 4.615 | 100.6 |
| **Text & Structured Geomean (6)** | — | 0.1231 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 1.956 | **5.187** | 4.044 | 1.975 | 12.363 | **18.413** | 8.798 | 7.610 | 59.2 |
| **2D BMP Geomean (1)** | — | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.774 | **4.400** | 2.903 | 0.605 | 9.477 | **18.720** | 5.510 | 4.615 | 100.6 |
| **All-Dataset Geomean (7)** | — | 0.1612 | 0.2990 | 0.2045 | 0.1800 | 0.1798 | **0.1563** | 0.1053 | 1.714 | **5.067** | 3.857 | 1.668 | 11.903 | **18.456** | 8.229 | 7.085 | 63.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2132 | 2.904 | **3.636** | 2.711 | 13.429 | **17.261** | 7.487 | 34.1 | 26.6 | 21.1 | **37.2** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0592** | 0.1222 | 0.0783 | **8.382** | 6.773 | 5.096 | 34.405 | **34.606** | 12.482 | **286.7** | 125.3 | 101.9 | 195.5 |
| **`gharchive-2025-01-15.ndjson`** | 0.1350 | 0.1872 | **0.1150** | 1.481 | **7.208** | 5.063 | 8.592 | **29.789** | 13.931 | 26.4 | 78.3 | 73.0 | **127.4** |
| **`apache-access.clf`** | 0.0990 | 0.1362 | **0.0878** | 1.119 | **6.787** | 4.960 | 8.197 | **14.734** | 11.793 | 30.6 | 73.4 | 87.1 | **113.8** |
| **`macos-install.log`** | 0.0605 | 0.1033 | **0.0484** | 3.507 | **12.180** | 8.009 | 15.206 | **46.067** | 19.865 | 120.6 | 229.4 | 260.7 | **489.6** |
| **`pg100-shakespeare.txt`** | 0.3977 | 0.4970 | **0.3745** | 0.396 | 2.749 | **2.908** | 7.217 | **15.960** | 6.800 | 4.3 | 13.3 | 11.9 | **18.2** |
| **`kodak-truecolor-16.bmp`** | 0.8106 | 0.8127 | **0.7452** | 0.774 | **8.253** | 5.100 | 9.477 | **32.074** | 17.254 | 3.3 | 20.0 | 12.6 | **21.8** |
| **Text & Structured Geomean (6)** | 0.1231 | 0.1903 | **0.1205** | 1.956 | **5.856** | 4.480 | 12.363 | **24.035** | 11.293 | 39.9 | 62.3 | 59.0 | **99.0** |
| **2D BMP Geomean (1)** | 0.8106 | 0.8127 | **0.7452** | 0.774 | **8.253** | 5.100 | 9.477 | **32.074** | 17.254 | 3.3 | 20.0 | 12.6 | **21.8** |
| **All-Dataset Geomean (7)** | 0.1612 | 0.2342 | **0.1563** | 1.714 | **6.150** | 4.563 | 11.903 | **25.046** | 11.998 | 28.0 | 53.0 | 47.3 | **79.7** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **34.1** | 14.6 | 17.0 | 9.7 | **5.97** | 4.49 | 2.82 |
| **`nyc-taxi-2024-01.ndjson`** | **286.7** | 58.6 | 62.6 | 44.6 | **16.95** | 8.68 | 5.86 |
| **`gharchive-2025-01-15.ndjson`** | 26.4 | **51.0** | 46.0 | 37.1 | 4.55 | **8.37** | 5.56 |
| **`apache-access.clf`** | 30.6 | **49.6** | 48.8 | 39.9 | 4.52 | **7.53** | 5.56 |
| **`macos-install.log`** | 120.6 | **163.8** | 135.5 | 147.1 | 9.59 | **14.97** | 10.85 |
| **`pg100-shakespeare.txt`** | 4.3 | **10.0** | 7.7 | 4.7 | 1.93 | **3.98** | 2.06 |
| **`kodak-truecolor-16.bmp`** | 3.3 | **10.4** | 4.8 | 2.1 | 2.08 | **4.55** | 1.53 |
| **Text & Structured Geomean (6)** | **39.9** | 39.1 | 36.8 | 27.5 | 5.81 | **7.26** | 4.74 |
| **2D BMP Geomean (1)** | 3.3 | **10.4** | 4.8 | 2.1 | 2.08 | **4.55** | 1.53 |
| **All-Dataset Geomean (7)** | 28.0 | **32.3** | 27.6 | 19.1 | 5.02 | **6.79** | 4.04 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 5.746 | **4.450** | 15.813 | 2.037 | **1.176** | 4.425 | **10.0** | 6.4 | 1.2 |
| **`nyc-taxi-2024-01.ndjson`** | 2.412 | **2.176** | 6.498 | **0.638** | 0.754 | 2.005 | **231.2** | 45.8 | 12.4 |
| **`gharchive-2025-01-15.ndjson`** | 16.286 | **2.454** | 8.704 | 2.207 | **0.638** | 1.842 | 4.4 | **40.7** | 9.3 |
| **`apache-access.clf`** | 13.650 | **2.263** | 7.446 | 2.151 | **0.919** | 2.138 | 5.6 | **34.4** | 10.0 |
| **`macos-install.log`** | 7.633 | **1.626** | 5.372 | 1.648 | **0.496** | 1.286 | 34.0 | **182.5** | 56.0 |
| **`pg100-shakespeare.txt`** | 46.330 | **7.741** | 25.592 | 3.935 | **1.458** | 5.644 | 0.3 | **3.0** | 0.4 |
| **`kodak-truecolor-16.bmp`** | 36.700 | **4.050** | 36.149 | 2.260 | **0.854** | 3.179 | 0.4 | **5.6** | 0.2 |
