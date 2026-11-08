## VXG v4 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row (lowest block ratio, highest encode `GB/s`, highest decode `GB/s`).

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1807** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | 1.340 | **1.591** | 0.900 | 0.483 | 3.101 | **4.628** | 1.718 | 1.768 | 119.2 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **10.200** | 6.540 | 6.715 | 2.928 | **34.689** | 19.968 | 10.129 | 6.922 | 52.3 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1985 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 1.969 | **2.446** | 1.873 | 1.215 | 5.463 | **9.414** | 4.306 | 3.374 | 105.5 |
| **`apache-access.clf`** | 2.4 MB | 0.1198 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.621 | 0.943 | **1.052** | 0.513 | 1.616 | 1.462 | **1.969** | 1.228 | 70.3 |
| **`macos-install.log`** | 34.2 MB | 0.0948 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0278 | **8.797** | 8.593 | 6.142 | 2.182 | 22.427 | **25.441** | 11.226 | 8.146 | 41.1 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.4424 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | **2.480** | 2.124 | 1.906 | 0.738 | 11.102 | **11.333** | 5.216 | 3.561 | 99.8 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | **3.584** | 1.617 | 1.359 | 0.414 | **19.609** | 6.239 | 2.481 | 3.294 | 124.4 |
| **Text & Structured Geomean (6)** | — | 0.1484 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 2.673 | **2.756** | 2.278 | 1.060 | 7.864 | **8.460** | 4.530 | 3.372 | 75.8 |
| **2D BMP Geomean (1)** | — | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | **3.584** | 1.617 | 1.359 | 0.414 | **19.609** | 6.239 | 2.481 | 3.294 | 124.4 |
| **All-Dataset Geomean (7)** | — | 0.1927 | 0.2990 | 0.2045 | 0.1800 | 0.1798 | **0.1563** | 0.1053 | **2.787** | 2.554 | 2.116 | 0.927 | **8.960** | 8.100 | 4.156 | 3.361 | 81.3 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1807** | 0.2976 | 0.2132 | 1.340 | **1.615** | 1.165 | 3.101 | **5.581** | 2.631 | 11.3 | 10.1 | 8.2 | **14.1** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0592** | 0.1222 | 0.0783 | **10.200** | 7.241 | 7.130 | 34.689 | **41.554** | 14.771 | **317.6** | 142.0 | 131.1 | 221.5 |
| **`gharchive-2025-01-15.ndjson`** | 0.1985 | 0.1872 | **0.1150** | 1.969 | **2.674** | 2.261 | 5.463 | **9.414** | 4.951 | 16.5 | 26.8 | 29.1 | **43.6** |
| **`apache-access.clf`** | 0.1198 | 0.1362 | **0.0878** | 0.621 | 1.264 | **1.292** | 1.616 | **3.164** | 2.393 | 8.4 | 14.7 | 20.0 | **23.0** |
| **`macos-install.log`** | 0.0948 | 0.1033 | **0.0484** | 8.797 | **9.533** | 6.142 | 22.427 | **25.441** | 11.934 | 148.2 | 150.8 | 177.0 | **321.9** |
| **`pg100-shakespeare.txt`** | 0.4424 | 0.4970 | **0.3745** | **2.480** | 2.124 | 2.177 | 11.102 | **12.029** | 6.828 | 11.9 | 10.2 | 10.3 | **13.7** |
| **`kodak-truecolor-16.bmp`** | 0.9256 | 0.8127 | **0.7452** | **3.584** | 3.016 | 3.127 | **19.609** | 11.266 | 6.147 | **9.1** | 7.2 | 5.9 | 8.0 |
| **Text & Structured Geomean (6)** | 0.1484 | 0.1903 | **0.1205** | 2.673 | **3.047** | 2.621 | 7.864 | **11.328** | 5.786 | 30.9 | 30.9 | 32.3 | **49.0** |
| **2D BMP Geomean (1)** | 0.9256 | 0.8127 | **0.7452** | **3.584** | 3.016 | 3.127 | **19.609** | 11.266 | 6.147 | **9.1** | 7.2 | 5.9 | 8.0 |
| **All-Dataset Geomean (7)** | 0.1927 | 0.2342 | **0.1563** | 2.787 | **3.043** | 2.688 | 8.960 | **11.319** | 5.836 | 25.9 | 25.1 | 25.3 | **37.8** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **11.3** | 6.4 | 5.1 | 3.8 | **2.84** | 2.59 | 1.52 |
| **`nyc-taxi-2024-01.ndjson`** | **317.6** | 60.2 | 84.4 | 44.6 | **18.14** | 8.83 | 5.85 |
| **`gharchive-2025-01-15.ndjson`** | 16.5 | **21.3** | 18.0 | 16.1 | 3.78 | **4.67** | 3.20 |
| **`apache-access.clf`** | 8.4 | 6.8 | **11.7** | 7.4 | **2.03** | 2.00 | 1.80 |
| **`macos-install.log`** | **148.2** | 118.4 | 94.2 | 71.5 | **12.77** | 12.05 | 6.71 |
| **`pg100-shakespeare.txt`** | **11.9** | 7.9 | 7.1 | 4.0 | **3.96** | 3.38 | 1.87 |
| **`kodak-truecolor-16.bmp`** | **9.1** | 3.6 | 2.2 | 1.5 | **4.23** | 2.26 | 1.20 |
| **Text & Structured Geomean (6)** | **30.9** | 19.3 | 19.8 | 13.4 | **5.21** | 4.53 | 2.94 |
| **2D BMP Geomean (1)** | **9.1** | 3.6 | 2.2 | 1.5 | **4.23** | 2.26 | 1.20 |
| **All-Dataset Geomean (7)** | **25.9** | 15.2 | 14.5 | 9.8 | **5.06** | 4.11 | 2.59 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 5.388 | **2.582** | 12.634 | 2.247 | **0.547** | 3.465 | 3.2 | **5.4** | 0.6 |
| **`nyc-taxi-2024-01.ndjson`** | 2.322 | **2.108** | 7.139 | **0.728** | 0.758 | 2.194 | **244.3** | 47.6 | 11.3 |
| **`gharchive-2025-01-15.ndjson`** | 4.204 | **1.748** | 6.224 | 0.943 | **0.382** | 0.970 | 8.3 | **26.0** | 6.6 |
| **`apache-access.clf`** | 4.835 | **1.694** | 6.244 | 1.613 | **0.668** | 1.986 | 3.0 | **6.4** | 2.1 |
| **`macos-install.log`** | 2.392 | **1.513** | 4.646 | 0.842 | **0.474** | 1.084 | 104.4 | **139.8** | 31.9 |
| **`pg100-shakespeare.txt`** | 9.009 | **6.789** | 25.902 | 1.894 | **1.306** | 5.151 | **2.9** | 2.6 | 0.4 |
| **`kodak-truecolor-16.bmp`** | **2.868** | 3.437 | 32.956 | **0.345** | 0.721 | 2.771 | **9.1** | 2.3 | 0.2 |
