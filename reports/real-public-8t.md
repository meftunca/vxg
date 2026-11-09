## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 1.265 | **1.888** | 1.020 | 0.424 | 2.243 | **7.022** | 2.705 | 2.019 | 34.6 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | 4.606 | **5.159** | 1.547 | 1.368 | **18.693** | 14.059 | 4.273 | 3.893 | 16.7 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.727 | **6.247** | 2.661 | 1.239 | 8.422 | **17.115** | 8.229 | 6.290 | 100.1 |
| **`apache-access.clf`** | 2.4 MB | 0.0976 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.562 | **4.139** | 2.667 | 1.747 | 7.435 | **9.681** | 4.575 | 3.669 | 13.5 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0278 | 1.203 | **4.976** | 4.575 | 2.425 | 9.530 | **24.567** | 8.956 | 11.064 | 34.2 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3786 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.151 | 1.044 | **1.326** | 0.373 | 3.726 | **8.402** | 3.668 | 2.314 | 52.0 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.279 | **2.783** | 1.473 | 0.435 | 8.373 | **15.010** | 2.631 | 2.374 | 105.0 |
| **Text & Structured Geomean (6)** | — | **0.1172** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 0.869 | **3.307** | 2.020 | 1.021 | 6.734 | **12.248** | 4.926 | 4.085 | 33.4 |
| **2D BMP Geomean (1)** | — | 0.7987 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.279 | **2.783** | 1.473 | 0.435 | 8.373 | **15.010** | 2.631 | 2.374 | 105.0 |
| **All-Dataset Geomean (7)** | — | 0.1542 | 0.2990 | 0.2045 | 0.1800 | 0.1771 | **0.1534** | 0.1053 | 0.739 | **3.227** | 1.931 | 0.904 | 6.947 | **12.609** | 4.504 | 3.780 | 39.3 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2141 | 1.265 | **1.888** | 1.188 | 2.243 | **9.438** | 4.279 | 9.2 | 14.2 | 10.5 | **19.7** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0593** | 0.1222 | 0.0784 | 4.606 | **5.159** | 3.064 | **18.693** | 14.059 | 8.331 | **156.6** | 69.7 | 64.5 | 108.7 |
| **`gharchive-2025-01-15.ndjson`** | 0.1181 | 0.1872 | **0.1072** | 0.727 | **6.247** | 4.161 | 8.422 | **29.638** | 8.696 | 21.0 | 72.7 | 56.1 | **126.9** |
| **`apache-access.clf`** | 0.0976 | 0.1362 | **0.0896** | 0.562 | **4.387** | 3.918 | 7.435 | **9.681** | 7.230 | 21.0 | 47.8 | 59.4 | **72.7** |
| **`macos-install.log`** | 0.0548 | 0.1033 | **0.0452** | 1.203 | **6.794** | 4.575 | 9.530 | **24.567** | 12.941 | 61.8 | 125.1 | 170.4 | **286.1** |
| **`pg100-shakespeare.txt`** | 0.3786 | 0.4970 | **0.3729** | 0.151 | **1.392** | 1.326 | 3.726 | **8.402** | 4.866 | 2.0 | 6.9 | 6.8 | **9.2** |
| **`kodak-truecolor-16.bmp`** | 0.7987 | 0.8127 | **0.7363** | 0.279 | **4.017** | 3.238 | 8.373 | **15.010** | 11.336 | 1.9 | 9.6 | 8.2 | **10.5** |
| **Text & Structured Geomean (6)** | **0.1172** | 0.1903 | 0.1181 | 0.869 | **3.690** | 2.667 | 6.734 | **14.100** | 7.216 | 20.6 | 37.9 | 37.1 | **61.1** |
| **2D BMP Geomean (1)** | 0.7987 | 0.8127 | **0.7363** | 0.279 | **4.017** | 3.238 | 8.373 | **15.010** | 11.336 | 1.9 | 9.6 | 8.2 | **10.5** |
| **All-Dataset Geomean (7)** | 0.1542 | 0.2342 | **0.1534** | 0.739 | **3.735** | 2.742 | 6.947 | **14.227** | 7.697 | 14.7 | 31.1 | 29.9 | **47.5** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **9.2** | 8.6 | 6.8 | 3.8 | 2.49 | **3.15** | 1.53 |
| **`nyc-taxi-2024-01.ndjson`** | **156.6** | 44.8 | 26.3 | 22.8 | **11.33** | 7.26 | 3.75 |
| **`gharchive-2025-01-15.ndjson`** | 21.0 | **45.8** | 29.7 | 23.3 | 3.73 | **7.80** | 4.02 |
| **`apache-access.clf`** | 21.0 | **36.6** | 28.3 | 23.1 | 3.50 | **6.14** | 3.88 |
| **`macos-install.log`** | 61.8 | 88.5 | 72.6 | **94.0** | 5.94 | **9.93** | 7.87 |
| **`pg100-shakespeare.txt`** | 2.0 | 4.8 | **5.0** | 2.3 | 1.14 | **2.42** | 1.29 |
| **`kodak-truecolor-16.bmp`** | 1.9 | **7.4** | 2.4 | 1.3 | 1.43 | **3.63** | 1.10 |
| **Text & Structured Geomean (6)** | 20.6 | **25.5** | 19.5 | 14.7 | 3.68 | **5.45** | 3.11 |
| **2D BMP Geomean (1)** | 1.9 | **7.4** | 2.4 | 1.3 | 1.43 | **3.63** | 1.10 |
| **All-Dataset Geomean (7)** | 14.7 | **21.3** | 14.4 | 10.4 | 3.22 | **5.14** | 2.68 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 3.736 | **2.293** | 7.801 | 1.124 | **0.562** | 2.436 | 4.5 | **7.6** | 0.9 |
| **`nyc-taxi-2024-01.ndjson`** | 1.573 | **1.323** | 4.438 | **0.430** | 0.532 | 1.127 | **190.4** | 53.4 | 10.2 |
| **`gharchive-2025-01-15.ndjson`** | 14.117 | **1.797** | 6.779 | 0.961 | **0.361** | 1.036 | 5.7 | **56.9** | 8.8 |
| **`apache-access.clf`** | 13.091 | **1.429** | 4.407 | 1.075 | **0.590** | 0.988 | 5.6 | **39.8** | 11.1 |
| **`macos-install.log`** | 6.068 | **0.729** | 2.918 | 0.757 | **0.295** | 0.656 | 28.8 | **190.8** | 68.0 |
| **`pg100-shakespeare.txt`** | 50.482 | **3.304** | 11.833 | 2.460 | **0.771** | 3.347 | 0.2 | **3.0** | 0.4 |
| **`kodak-truecolor-16.bmp`** | 42.721 | **2.487** | 20.082 | 1.080 | **0.510** | 1.570 | 0.3 | **6.6** | 0.2 |
