## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 0.527 | **0.606** | 0.479 | 0.246 | 2.143 | **3.050** | 1.276 | 0.862 | 21.4 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **1.359** | 1.332 | 1.022 | 0.546 | **6.264** | 4.467 | 2.112 | 1.650 | 13.1 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.141 | **1.205** | 0.951 | 0.349 | 1.993 | **5.399** | 2.289 | 2.055 | 105.1 |
| **`apache-access.clf`** | 2.4 MB | 0.0974 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.177 | **1.489** | 1.127 | 0.563 | 1.868 | **5.879** | 2.625 | 2.142 | 14.0 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0277 | 0.314 | **2.090** | 1.436 | 0.760 | 2.494 | **7.288** | 3.357 | 3.414 | 30.4 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3780 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.040 | 0.377 | **0.392** | 0.162 | 0.993 | **2.819** | 1.127 | 0.708 | 49.1 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7985 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.053 | **0.830** | 0.517 | 0.132 | 1.828 | **4.082** | 1.034 | 0.967 | 84.9 |
| **Text & Structured Geomean (6)** | — | **0.1171** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 0.247 | **1.022** | 0.816 | 0.385 | 2.233 | **4.551** | 1.985 | 1.573 | 29.2 |
| **2D BMP Geomean (1)** | — | 0.7985 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.053 | **0.830** | 0.517 | 0.132 | 1.828 | **4.082** | 1.034 | 0.967 | 84.9 |
| **All-Dataset Geomean (7)** | — | 0.1541 | 0.2990 | 0.2045 | 0.1800 | 0.1771 | **0.1534** | 0.1052 | 0.198 | **0.992** | 0.765 | 0.330 | 2.170 | **4.481** | 1.809 | 1.467 | 34.0 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2141 | 0.527 | **0.606** | 0.492 | 2.143 | **3.868** | 1.425 | 5.8 | 5.1 | 3.9 | **7.1** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0593** | 0.1222 | 0.0784 | **1.359** | 1.332 | 1.160 | 6.264 | **7.713** | 2.591 | **49.2** | 26.2 | 22.1 | 40.9 |
| **`gharchive-2025-01-15.ndjson`** | 0.1181 | 0.1872 | **0.1072** | 0.141 | **1.335** | 1.041 | 1.993 | **6.784** | 2.836 | 4.5 | 16.1 | 16.0 | **28.1** |
| **`apache-access.clf`** | 0.0974 | 0.1362 | **0.0896** | 0.177 | **1.571** | 1.271 | 1.868 | **7.570** | 3.108 | 5.9 | 25.3 | 22.2 | **38.5** |
| **`macos-install.log`** | 0.0548 | 0.1033 | **0.0452** | 0.314 | **2.186** | 1.656 | 2.494 | **9.886** | 4.261 | 16.1 | 45.0 | 58.8 | **103.0** |
| **`pg100-shakespeare.txt`** | 0.3780 | 0.4970 | **0.3729** | 0.040 | **0.433** | 0.415 | 0.993 | **3.445** | 1.466 | 0.5 | 2.5 | 2.1 | **3.3** |
| **`kodak-truecolor-16.bmp`** | 0.7985 | 0.8127 | **0.7363** | 0.053 | **1.375** | 1.035 | 1.828 | **6.391** | 3.375 | 0.4 | 3.6 | 2.5 | **4.0** |
| **Text & Structured Geomean (6)** | **0.1171** | 0.1903 | 0.1181 | 0.247 | **1.082** | 0.896 | 2.233 | **6.113** | 2.425 | 6.3 | 13.5 | 12.5 | **21.8** |
| **2D BMP Geomean (1)** | 0.7985 | 0.8127 | **0.7363** | 0.053 | **1.375** | 1.035 | 1.828 | **6.391** | 3.375 | 0.4 | 3.6 | 2.5 | **4.0** |
| **All-Dataset Geomean (7)** | 0.1541 | 0.2342 | **0.1534** | 0.198 | **1.119** | 0.915 | 2.170 | **6.152** | 2.542 | 4.3 | 11.2 | 9.9 | **17.1** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **5.8** | 3.2 | 3.2 | 1.9 | **1.83** | 1.63 | 0.96 |
| **`nyc-taxi-2024-01.ndjson`** | **49.2** | 12.8 | 15.0 | 9.4 | **5.24** | 3.15 | 2.07 |
| **`gharchive-2025-01-15.ndjson`** | 4.5 | **11.3** | 9.4 | 7.1 | 1.34 | **3.07** | 1.81 |
| **`apache-access.clf`** | 5.9 | **17.1** | 13.9 | 10.0 | 1.50 | **3.70** | 2.22 |
| **`macos-install.log`** | 16.1 | **31.3** | 24.9 | 29.2 | 2.43 | **4.96** | 3.61 |
| **`pg100-shakespeare.txt`** | 0.5 | **1.7** | 1.5 | 0.8 | 0.47 | **1.20** | 0.66 |
| **`kodak-truecolor-16.bmp`** | 0.4 | **2.1** | 0.9 | 0.5 | 0.50 | **1.57** | 0.55 |
| **Text & Structured Geomean (6)** | 6.3 | **8.6** | 7.9 | 5.6 | 1.68 | **2.65** | 1.63 |
| **2D BMP Geomean (1)** | 0.4 | **2.1** | 0.9 | 0.5 | 0.50 | **1.57** | 0.55 |
| **All-Dataset Geomean (7)** | 4.3 | **7.1** | 5.8 | 3.9 | 1.41 | **2.46** | 1.40 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 7.957 | **5.030** | 18.491 | 2.213 | **1.447** | 4.939 | **1.4** | 1.2 | 0.2 |
| **`nyc-taxi-2024-01.ndjson`** | 3.139 | **2.422** | 8.395 | **0.808** | 0.918 | 2.666 | **30.9** | 8.6 | 2.0 |
| **`gharchive-2025-01-15.ndjson`** | 28.527 | **2.783** | 12.429 | 2.111 | **0.744** | 2.250 | 0.6 | **7.9** | 1.3 |
| **`apache-access.clf`** | 23.799 | **2.388** | 7.975 | 2.210 | **0.773** | 2.014 | 0.8 | **12.6** | 2.5 |
| **`macos-install.log`** | 12.265 | **1.682** | 5.832 | 1.664 | **0.566** | 1.336 | 3.6 | **32.0** | 10.5 |
| **`pg100-shakespeare.txt`** | 97.720 | **8.177** | 29.969 | 4.873 | **1.448** | 6.338 | 0.0 | **0.5** | 0.1 |
| **`kodak-truecolor-16.bmp`** | 80.724 | **4.645** | 37.695 | 2.408 | **0.995** | 4.033 | 0.0 | **1.0** | 0.0 |
