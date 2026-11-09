## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 0.550 | **0.613** | 0.485 | 0.254 | 1.851 | **3.130** | 1.284 | 0.866 | 20.8 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **1.403** | 1.384 | 1.055 | 0.564 | **5.831** | 4.656 | 2.132 | 1.800 | 11.7 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.161 | **1.270** | 0.984 | 0.351 | 2.116 | **5.729** | 2.377 | 2.114 | 92.4 |
| **`apache-access.clf`** | 2.4 MB | 0.0976 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.198 | **1.586** | 1.194 | 0.562 | 2.141 | **5.689** | 2.636 | 2.232 | 11.8 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0277 | 0.375 | **2.151** | 1.465 | 0.792 | 2.595 | **7.649** | 3.352 | 3.575 | 28.2 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3786 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.044 | 0.388 | **0.408** | 0.162 | 0.964 | **2.863** | 1.147 | 0.696 | 45.1 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.057 | **0.866** | 0.542 | 0.128 | 1.888 | **4.284** | 1.042 | 0.939 | 81.5 |
| **Text & Structured Geomean (6)** | — | **0.1172** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 0.272 | **1.061** | 0.843 | 0.392 | 2.228 | **4.672** | 2.010 | 1.623 | 26.4 |
| **2D BMP Geomean (1)** | — | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.057 | **0.866** | 0.542 | 0.128 | 1.888 | **4.284** | 1.042 | 0.939 | 81.5 |
| **All-Dataset Geomean (7)** | — | 0.1542 | 0.2990 | 0.2045 | 0.1800 | 0.1771 | **0.1534** | 0.1052 | 0.217 | **1.031** | 0.791 | 0.334 | 2.176 | **4.615** | 1.830 | 1.501 | 31.0 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2141 | 0.550 | **0.613** | 0.502 | 1.851 | **4.045** | 1.406 | 5.5 | 5.3 | 3.9 | **7.4** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0593** | 0.1222 | 0.0784 | **1.403** | 1.384 | 1.204 | 5.831 | **8.396** | 2.741 | **48.3** | 27.9 | 23.2 | 43.5 |
| **`gharchive-2025-01-15.ndjson`** | 0.1181 | 0.1872 | **0.1072** | 0.161 | **1.406** | 1.045 | 2.116 | **7.191** | 2.967 | 4.9 | 17.0 | 16.4 | **29.7** |
| **`apache-access.clf`** | 0.0976 | 0.1362 | **0.0896** | 0.198 | **1.655** | 1.293 | 2.141 | **7.837** | 3.165 | 6.7 | 26.4 | 22.6 | **40.2** |
| **`macos-install.log`** | 0.0548 | 0.1033 | **0.0452** | 0.375 | **2.290** | 1.668 | 2.595 | **10.171** | 4.270 | 18.0 | 46.7 | 59.1 | **106.9** |
| **`pg100-shakespeare.txt`** | 0.3786 | 0.4970 | **0.3729** | 0.044 | **0.447** | 0.433 | 0.964 | **3.506** | 1.475 | 0.5 | 2.5 | 2.1 | **3.4** |
| **`kodak-truecolor-16.bmp`** | 0.7987 | 0.8127 | **0.7363** | 0.057 | **1.425** | 1.052 | 1.888 | **6.525** | 3.390 | 0.4 | 3.8 | 2.6 | **4.1** |
| **Text & Structured Geomean (6)** | **0.1172** | 0.1903 | 0.1181 | 0.272 | **1.125** | 0.916 | 2.228 | **6.393** | 2.472 | 6.6 | 14.1 | 12.7 | **22.7** |
| **2D BMP Geomean (1)** | 0.7987 | 0.8127 | **0.7363** | 0.057 | **1.425** | 1.052 | 1.888 | **6.525** | 3.390 | 0.4 | 3.8 | 2.6 | **4.1** |
| **All-Dataset Geomean (7)** | 0.1542 | 0.2342 | **0.1534** | 0.217 | **1.163** | 0.934 | 2.176 | **6.411** | 2.586 | 4.5 | 11.7 | 10.1 | **17.8** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **5.5** | 3.3 | 3.2 | 1.9 | **1.77** | 1.65 | 0.97 |
| **`nyc-taxi-2024-01.ndjson`** | **48.3** | 13.4 | 15.3 | 10.0 | **5.17** | 3.24 | 2.16 |
| **`gharchive-2025-01-15.ndjson`** | 4.9 | **12.0** | 9.7 | 7.2 | 1.42 | **3.18** | 1.83 |
| **`apache-access.clf`** | 6.7 | **17.4** | 14.4 | 10.2 | 1.63 | **3.74** | 2.25 |
| **`macos-install.log`** | 18.0 | **32.5** | 25.1 | 30.5 | 2.61 | **5.09** | 3.72 |
| **`pg100-shakespeare.txt`** | 0.5 | **1.7** | 1.5 | 0.8 | 0.48 | **1.21** | 0.66 |
| **`kodak-truecolor-16.bmp`** | 0.4 | **2.2** | 0.9 | 0.4 | 0.51 | **1.62** | 0.54 |
| **Text & Structured Geomean (6)** | 6.6 | **8.9** | 8.0 | 5.8 | 1.73 | **2.71** | 1.66 |
| **2D BMP Geomean (1)** | 0.4 | **2.2** | 0.9 | 0.4 | 0.51 | **1.62** | 0.54 |
| **All-Dataset Geomean (7)** | 4.5 | **7.3** | 5.9 | 4.0 | 1.45 | **2.51** | 1.41 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 8.388 | **4.975** | 18.228 | 2.539 | **1.452** | 4.973 | 1.2 | **1.2** | 0.2 |
| **`nyc-taxi-2024-01.ndjson`** | 3.071 | **2.507** | 8.315 | **0.868** | 0.977 | 2.700 | **29.6** | 8.5 | 2.1 |
| **`gharchive-2025-01-15.ndjson`** | 26.702 | **2.871** | 11.643 | 2.192 | **0.803** | 2.176 | 0.6 | **7.9** | 1.4 |
| **`apache-access.clf`** | 22.383 | **2.346** | 7.056 | 2.037 | **0.720** | 2.152 | 1.0 | **13.4** | 2.6 |
| **`macos-install.log`** | 11.095 | **1.661** | 5.710 | 1.615 | **0.578** | 1.341 | 4.2 | **33.2** | 11.0 |
| **`pg100-shakespeare.txt`** | 94.297 | **7.557** | 28.818 | 5.361 | **1.376** | 6.224 | 0.0 | **0.5** | 0.1 |
| **`kodak-truecolor-16.bmp`** | 77.488 | **4.681** | 34.200 | 2.482 | **0.919** | 3.884 | 0.0 | **1.1** | 0.0 |
