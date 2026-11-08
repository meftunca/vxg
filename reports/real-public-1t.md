## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | **0.639** | 0.598 | 0.481 | 0.243 | 2.020 | **3.077** | 1.289 | 0.916 | 83.0 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **1.594** | 1.326 | 1.020 | 0.541 | **5.803** | 4.523 | 2.064 | 1.757 | 50.3 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1231 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 0.285 | **1.218** | 0.946 | 0.453 | 2.001 | **5.626** | 2.282 | 2.101 | 59.2 |
| **`apache-access.clf`** | 2.4 MB | 0.1037 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.320 | **1.529** | 1.176 | 0.588 | 2.063 | **5.870** | 2.640 | 2.258 | 44.3 |
| **`macos-install.log`** | 34.2 MB | 0.0594 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0277 | 0.667 | **2.161** | 1.500 | 0.770 | 2.468 | **7.679** | 3.474 | 3.380 | 34.8 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3884 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.086 | 0.389 | **0.397** | 0.148 | 0.959 | **2.862** | 1.155 | 0.768 | 96.3 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.8256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.134 | **0.836** | 0.529 | 0.107 | 2.210 | **4.258** | 1.020 | 0.975 | 107.6 |
| **Text & Structured Geomean (6)** | — | 0.1213 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 0.417 | **1.037** | 0.829 | 0.398 | 2.204 | **4.649** | 2.002 | 1.645 | 57.6 |
| **2D BMP Geomean (1)** | — | 0.8256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.134 | **0.836** | 0.529 | 0.107 | 2.210 | **4.258** | 1.020 | 0.975 | 107.6 |
| **All-Dataset Geomean (7)** | — | 0.1596 | 0.2990 | 0.2045 | 0.1800 | 0.1798 | **0.1563** | 0.1052 | 0.355 | **1.006** | 0.778 | 0.330 | 2.204 | **4.591** | 1.818 | 1.526 | 63.0 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2132 | **0.639** | 0.603 | 0.517 | 2.020 | **3.894** | 1.402 | 6.2 | 5.1 | 4.0 | **7.2** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0592** | 0.1222 | 0.0783 | **1.594** | 1.330 | 1.177 | 5.803 | **7.975** | 2.640 | **51.3** | 26.7 | 22.5 | 41.6 |
| **`gharchive-2025-01-15.ndjson`** | 0.1231 | 0.1872 | **0.1150** | 0.285 | **1.295** | 1.045 | 2.001 | **6.873** | 2.861 | 6.1 | 15.9 | 15.0 | **26.0** |
| **`apache-access.clf`** | 0.1037 | 0.1362 | **0.0878** | 0.320 | **1.710** | 1.341 | 2.063 | **8.186** | 3.268 | 7.8 | 27.5 | 23.8 | **42.6** |
| **`macos-install.log`** | 0.0594 | 0.1033 | **0.0484** | 0.667 | **2.252** | 1.667 | 2.468 | **10.112** | 4.210 | 21.6 | 46.2 | 54.8 | **98.6** |
| **`pg100-shakespeare.txt`** | 0.3884 | 0.4970 | **0.3745** | 0.086 | 0.431 | **0.439** | 0.959 | **3.498** | 1.491 | 0.7 | 2.5 | 2.2 | **3.3** |
| **`kodak-truecolor-16.bmp`** | 0.8256 | 0.8127 | **0.7452** | 0.134 | **1.381** | 1.048 | 2.210 | **6.189** | 3.316 | 0.7 | 3.6 | 2.5 | **3.9** |
| **Text & Structured Geomean (6)** | 0.1213 | 0.1903 | **0.1205** | 0.417 | **1.095** | 0.924 | 2.204 | **6.288** | 2.452 | 7.9 | 13.8 | 12.5 | **21.8** |
| **2D BMP Geomean (1)** | 0.8256 | 0.8127 | **0.7452** | 0.134 | **1.381** | 1.048 | 2.210 | **6.189** | 3.316 | 0.7 | 3.6 | 2.5 | **3.9** |
| **All-Dataset Geomean (7)** | 0.1596 | 0.2342 | **0.1563** | 0.355 | **1.132** | 0.941 | 2.204 | **6.273** | 2.560 | 5.5 | 11.4 | 9.9 | **17.1** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **6.2** | 3.2 | 3.2 | 2.0 | **1.92** | 1.63 | 0.97 |
| **`nyc-taxi-2024-01.ndjson`** | **51.3** | 12.9 | 14.8 | 9.6 | **5.38** | 3.16 | 2.11 |
| **`gharchive-2025-01-15.ndjson`** | 6.1 | **11.6** | 9.3 | 7.8 | 1.67 | **3.12** | 1.97 |
| **`apache-access.clf`** | 7.8 | **17.3** | 14.3 | 10.7 | 1.85 | **3.73** | 2.31 |
| **`macos-install.log`** | 21.6 | **32.6** | 25.9 | 27.4 | 3.03 | **5.10** | 3.53 |
| **`pg100-shakespeare.txt`** | 0.7 | **1.7** | 1.5 | 0.8 | 0.60 | **1.21** | 0.66 |
| **`kodak-truecolor-16.bmp`** | 0.7 | **2.2** | 0.9 | 0.4 | 0.71 | **1.60** | 0.51 |
| **Text & Structured Geomean (6)** | 7.9 | **8.8** | 8.0 | 5.8 | 1.96 | **2.68** | 1.67 |
| **2D BMP Geomean (1)** | 0.7 | **2.2** | 0.9 | 0.4 | 0.71 | **1.60** | 0.51 |
| **All-Dataset Geomean (7)** | 5.5 | **7.2** | 5.8 | 3.9 | 1.70 | **2.49** | 1.41 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 7.110 | **5.276** | 19.832 | 2.514 | **1.451** | 5.510 | **1.5** | 1.2 | 0.2 |
| **`nyc-taxi-2024-01.ndjson`** | 2.825 | **2.603** | 8.503 | **0.873** | 0.984 | 2.852 | **32.7** | 8.1 | 2.0 |
| **`gharchive-2025-01-15.ndjson`** | 13.809 | **2.929** | 10.101 | 2.176 | **0.829** | 2.180 | 1.1 | **7.4** | 1.7 |
| **`apache-access.clf`** | 12.619 | **2.417** | 8.033 | 2.188 | **0.779** | 2.189 | 1.5 | **12.6** | 2.6 |
| **`macos-install.log`** | 6.079 | **1.778** | 6.000 | 1.712 | **0.598** | 1.439 | 6.7 | **31.6** | 9.3 |
| **`pg100-shakespeare.txt`** | 48.936 | **8.562** | 30.820 | 5.558 | **1.496** | 6.157 | 0.0 | **0.5** | 0.1 |
| **`kodak-truecolor-16.bmp`** | 31.280 | **4.778** | 40.974 | 2.074 | **1.045** | 3.820 | 0.1 | **1.0** | 0.0 |
