## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 0.503 | **0.547** | 0.409 | 0.216 | 1.438 | **2.892** | 1.154 | 0.773 | 23.1 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **1.379** | 1.212 | 0.961 | 0.517 | **5.322** | 4.494 | 1.881 | 1.721 | 13.4 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.154 | **1.140** | 0.941 | 0.352 | 2.020 | **5.058** | 2.273 | 1.898 | 98.9 |
| **`apache-access.clf`** | 2.4 MB | 0.0976 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.178 | **1.471** | 1.082 | 0.535 | 1.879 | **5.115** | 2.527 | 2.053 | 12.0 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0277 | 0.377 | **2.101** | 1.378 | 0.726 | 2.496 | **7.409** | 3.301 | 3.434 | 30.8 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3786 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.041 | 0.354 | **0.395** | 0.157 | 0.847 | **2.804** | 1.158 | 0.703 | 51.7 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.053 | **0.823** | 0.492 | 0.115 | 1.773 | **4.270** | 0.928 | 0.888 | 93.5 |
| **Text & Structured Geomean (6)** | — | **0.1172** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 0.257 | **0.969** | 0.776 | 0.366 | 1.986 | **4.372** | 1.904 | 1.524 | 28.9 |
| **2D BMP Geomean (1)** | — | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.053 | **0.823** | 0.492 | 0.115 | 1.773 | **4.270** | 0.928 | 0.888 | 93.5 |
| **All-Dataset Geomean (7)** | — | 0.1542 | 0.2990 | 0.2045 | 0.1800 | 0.1771 | **0.1534** | 0.1052 | 0.205 | **0.946** | 0.727 | 0.310 | 1.954 | **4.357** | 1.718 | 1.411 | 34.2 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2141 | 0.503 | **0.575** | 0.477 | 1.438 | **3.417** | 1.195 | 4.6 | 4.7 | 3.5 | **6.5** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0593** | 0.1222 | 0.0784 | **1.379** | 1.327 | 1.186 | 5.322 | **7.413** | 2.426 | **45.7** | 25.7 | 21.6 | 40.0 |
| **`gharchive-2025-01-15.ndjson`** | 0.1181 | 0.1872 | **0.1072** | 0.154 | **1.240** | 0.955 | 2.020 | **6.901** | 2.546 | 4.7 | 15.6 | 14.5 | **27.3** |
| **`apache-access.clf`** | 0.0976 | 0.1362 | **0.0896** | 0.178 | **1.529** | 1.283 | 1.879 | **7.776** | 3.017 | 5.9 | 25.3 | 22.0 | **38.5** |
| **`macos-install.log`** | 0.0548 | 0.1033 | **0.0452** | 0.377 | **2.161** | 1.592 | 2.496 | **9.753** | 3.870 | 17.7 | 44.5 | 55.0 | **101.7** |
| **`pg100-shakespeare.txt`** | 0.3786 | 0.4970 | **0.3729** | 0.041 | **0.429** | 0.398 | 0.847 | **3.226** | 1.417 | 0.5 | 2.4 | 2.0 | **3.2** |
| **`kodak-truecolor-16.bmp`** | 0.7987 | 0.8127 | **0.7363** | 0.053 | **1.410** | 1.044 | 1.773 | **6.285** | 3.286 | 0.4 | 3.7 | 2.5 | **4.0** |
| **Text & Structured Geomean (6)** | **0.1172** | 0.1903 | 0.1181 | 0.257 | **1.050** | 0.872 | 1.986 | **5.913** | 2.227 | 6.1 | 13.1 | 11.8 | **21.1** |
| **2D BMP Geomean (1)** | 0.7987 | 0.8127 | **0.7363** | 0.053 | **1.410** | 1.044 | 1.773 | **6.285** | 3.286 | 0.4 | 3.7 | 2.5 | **4.0** |
| **All-Dataset Geomean (7)** | 0.1542 | 0.2342 | **0.1534** | 0.205 | **1.095** | 0.895 | 1.954 | **5.965** | 2.355 | 4.1 | 10.9 | 9.5 | **16.7** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **4.6** | 3.0 | 2.8 | 1.7 | **1.58** | 1.55 | 0.88 |
| **`nyc-taxi-2024-01.ndjson`** | **45.7** | 12.3 | 13.8 | 9.3 | **4.98** | 3.06 | 2.07 |
| **`gharchive-2025-01-15.ndjson`** | 4.7 | **10.6** | 9.3 | 6.8 | 1.38 | **2.95** | 1.77 |
| **`apache-access.clf`** | 5.9 | **15.9** | 13.4 | 9.6 | 1.51 | **3.52** | 2.16 |
| **`macos-install.log`** | 17.7 | **31.6** | 24.2 | 28.7 | 2.58 | **5.00** | 3.56 |
| **`pg100-shakespeare.txt`** | 0.5 | **1.6** | 1.5 | 0.8 | 0.45 | **1.17** | 0.65 |
| **`kodak-truecolor-16.bmp`** | 0.4 | **2.1** | 0.8 | 0.4 | 0.49 | **1.59** | 0.51 |
| **Text & Structured Geomean (6)** | 6.1 | **8.2** | 7.5 | 5.4 | 1.63 | **2.57** | 1.59 |
| **2D BMP Geomean (1)** | 0.4 | **2.1** | 0.8 | 0.4 | 0.49 | **1.59** | 0.51 |
| **All-Dataset Geomean (7)** | 4.1 | **6.8** | 5.5 | 3.7 | 1.38 | **2.40** | 1.35 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 6.474 | **3.777** | 13.719 | 1.614 | **1.163** | 4.005 | **1.4** | 1.4 | 0.2 |
| **`nyc-taxi-2024-01.ndjson`** | 2.853 | **1.932** | 7.046 | **0.751** | 0.883 | 2.488 | **31.2** | 9.4 | 2.2 |
| **`gharchive-2025-01-15.ndjson`** | 23.622 | **2.324** | 11.708 | 1.897 | **0.611** | 1.732 | 0.7 | **8.9** | 1.5 |
| **`apache-access.clf`** | 16.873 | **2.061** | 6.746 | 1.513 | **0.579** | 1.821 | 1.2 | **14.5** | 2.7 |
| **`macos-install.log`** | 10.536 | **1.661** | 4.802 | 1.523 | **0.537** | 1.243 | 4.4 | **33.4** | 11.7 |
| **`pg100-shakespeare.txt`** | 79.038 | **6.557** | 25.511 | 4.142 | **1.280** | 5.996 | 0.0 | **0.6** | 0.1 |
| **`kodak-truecolor-16.bmp`** | 64.079 | **4.158** | 28.243 | 2.042 | **0.913** | 3.339 | 0.0 | **1.1** | 0.0 |
