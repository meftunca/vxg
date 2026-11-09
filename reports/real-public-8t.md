## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 2.632 | **4.639** | 2.545 | 1.377 | 11.953 | **17.398** | 7.857 | 4.614 | 20.4 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **8.790** | 7.492 | 7.093 | 3.877 | **43.546** | 28.921 | 15.023 | 12.598 | 12.5 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 1.046 | 6.690 | **7.157** | 2.571 | 14.143 | **29.287** | 15.276 | 14.729 | 96.9 |
| **`apache-access.clf`** | 2.4 MB | 0.0976 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.711 | **6.421** | 6.331 | 2.614 | 12.441 | **13.513** | 11.666 | 9.463 | 12.3 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0278 | 2.562 | **11.772** | 7.390 | 3.920 | 16.902 | **37.247** | 22.934 | 23.414 | 29.9 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3786 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.268 | **2.736** | 1.949 | 0.902 | 7.014 | **15.710** | 5.685 | 3.586 | 45.7 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.377 | **4.435** | 3.295 | 0.932 | 12.633 | **21.023** | 7.300 | 6.856 | 85.7 |
| **Text & Structured Geomean (6)** | — | **0.1172** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 1.509 | **6.030** | 4.770 | 2.242 | 14.881 | **22.100** | 11.831 | 9.378 | 27.3 |
| **2D BMP Geomean (1)** | — | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.377 | **4.435** | 3.295 | 0.932 | 12.633 | **21.023** | 7.300 | 6.856 | 85.7 |
| **All-Dataset Geomean (7)** | — | 0.1542 | 0.2990 | 0.2045 | 0.1800 | 0.1771 | **0.1534** | 0.1053 | 1.238 | **5.771** | 4.525 | 1.977 | 14.537 | **21.943** | 11.043 | 8.968 | 32.2 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2141 | 2.632 | **4.639** | 3.671 | 11.953 | **27.431** | 9.542 | 30.7 | 37.9 | 27.6 | **52.7** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0593** | 0.1222 | 0.0784 | 8.790 | **10.007** | 7.093 | **43.546** | 41.788 | 16.360 | **330.1** | 167.4 | 137.4 | 260.9 |
| **`gharchive-2025-01-15.ndjson`** | 0.1181 | 0.1872 | **0.1072** | 1.046 | **9.965** | 7.157 | 14.143 | **41.104** | 15.543 | 32.6 | 108.1 | 98.4 | **188.8** |
| **`apache-access.clf`** | 0.0976 | 0.1362 | **0.0896** | 0.711 | **8.272** | 6.331 | 12.441 | **18.661** | 12.492 | 30.5 | 91.2 | 99.2 | **138.6** |
| **`macos-install.log`** | 0.0548 | 0.1033 | **0.0452** | 2.562 | **11.772** | 8.978 | 16.902 | **57.827** | 28.569 | 120.0 | 252.7 | 354.7 | **577.9** |
| **`pg100-shakespeare.txt`** | 0.3786 | 0.4970 | **0.3729** | 0.268 | 2.736 | **3.093** | 7.014 | **19.190** | 7.516 | 3.6 | 14.6 | 12.9 | **20.7** |
| **`kodak-truecolor-16.bmp`** | 0.7987 | 0.8127 | **0.7363** | 0.377 | **10.214** | 5.272 | 12.633 | **32.473** | 18.770 | 2.7 | 22.4 | 13.5 | **24.7** |
| **Text & Structured Geomean (6)** | **0.1172** | 0.1903 | 0.1181 | 1.509 | **7.054** | 5.657 | 14.881 | **31.493** | 13.664 | 40.4 | 78.3 | 74.4 | **127.5** |
| **2D BMP Geomean (1)** | 0.7987 | 0.8127 | **0.7363** | 0.377 | **10.214** | 5.272 | 12.633 | **32.473** | 18.770 | 2.7 | 22.4 | 13.5 | **24.7** |
| **All-Dataset Geomean (7)** | 0.1542 | 0.2342 | **0.1534** | 1.238 | **7.437** | 5.600 | 14.537 | **31.631** | 14.298 | 27.5 | 65.5 | 58.3 | **100.9** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **30.7** | 21.2 | 18.3 | 10.5 | 5.56 | **5.75** | 2.98 |
| **`nyc-taxi-2024-01.ndjson`** | **330.1** | 77.5 | 105.6 | 69.2 | **18.62** | 10.45 | 7.85 |
| **`gharchive-2025-01-15.ndjson`** | 32.6 | 62.1 | **66.3** | 51.3 | 5.00 | **9.54** | 6.81 |
| **`apache-access.clf`** | 30.5 | 53.8 | **69.7** | 45.4 | 4.49 | **7.94** | 6.09 |
| **`macos-install.log`** | 120.0 | 167.7 | 147.7 | **173.9** | 9.24 | **15.20** | 11.85 |
| **`pg100-shakespeare.txt`** | 3.6 | **10.5** | 7.5 | 4.5 | 1.71 | **4.10** | 2.01 |
| **`kodak-truecolor-16.bmp`** | 2.7 | **11.1** | 5.9 | 3.3 | 1.81 | **4.74** | 2.02 |
| **Text & Structured Geomean (6)** | 40.4 | 46.2 | **46.4** | 33.1 | 5.77 | **8.11** | 5.33 |
| **2D BMP Geomean (1)** | 2.7 | **11.1** | 5.9 | 3.3 | 1.81 | **4.74** | 2.02 |
| **All-Dataset Geomean (7)** | 27.5 | **37.6** | 34.6 | 23.8 | 4.89 | **7.51** | 4.64 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 6.864 | **4.743** | 15.537 | 2.164 | **1.245** | 4.675 | 8.0 | **8.7** | 1.2 |
| **`nyc-taxi-2024-01.ndjson`** | 2.889 | **2.096** | 7.668 | **0.808** | 0.876 | 2.447 | **216.0** | 57.2 | 16.0 |
| **`gharchive-2025-01-15.ndjson`** | 22.500 | **2.206** | 11.757 | 1.803 | **0.658** | 2.088 | 5.1 | **51.5** | 10.3 |
| **`apache-access.clf`** | 18.818 | **2.165** | 7.204 | 1.691 | **0.902** | 2.220 | 5.4 | **38.5** | 11.4 |
| **`macos-install.log`** | 10.499 | **1.569** | 4.966 | 1.354 | **0.504** | 1.295 | 31.8 | **188.5** | 68.6 |
| **`pg100-shakespeare.txt`** | 88.884 | **7.734** | 25.552 | 5.045 | **1.468** | 5.630 | 0.2 | **3.1** | 0.4 |
| **`kodak-truecolor-16.bmp`** | 67.034 | **3.970** | 33.840 | 2.100 | **0.882** | 3.796 | 0.2 | **5.9** | 0.3 |
