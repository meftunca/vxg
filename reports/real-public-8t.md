## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2411 | 0.2141 | 0.1617 | 3.215 | **3.357** | 2.453 | 1.744 | 14.656 | **21.084** | 8.706 | 4.511 | 22.2 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0593** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0784 | 0.0554 | **8.836** | 6.355 | 4.456 | 3.494 | **38.588** | 21.976 | 11.509 | 8.618 | 13.5 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1181 | 0.2256 | 0.1577 | 0.1414 | 0.1200 | **0.1072** | 0.0633 | 0.952 | **6.131** | 4.597 | 2.495 | 15.305 | **23.287** | 11.127 | 11.559 | 97.2 |
| **`apache-access.clf`** | 2.4 MB | 0.0974 | 0.1730 | 0.1233 | 0.1062 | 0.1095 | **0.0896** | 0.0535 | 0.754 | **5.804** | 4.597 | 2.747 | 12.752 | **13.435** | 8.457 | 9.893 | 11.9 |
| **`macos-install.log`** | 34.2 MB | 0.0548 | 0.1249 | 0.0882 | 0.0770 | 0.0551 | **0.0452** | 0.0278 | 2.338 | 10.104 | **10.300** | 3.585 | 18.041 | **45.505** | 23.106 | 16.521 | 28.5 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3780 | 0.6223 | 0.4421 | 0.3918 | 0.4006 | **0.3729** | 0.3000 | 0.257 | **1.912** | 1.836 | 0.924 | 6.566 | **11.323** | 5.193 | 4.048 | 46.8 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.7985 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.391 | **4.898** | 2.941 | 0.626 | 13.729 | **25.616** | 6.665 | 4.781 | 85.3 |
| **Text & Structured Geomean (6)** | — | **0.1171** | 0.2500 | 0.1620 | 0.1419 | 0.1385 | 0.1181 | 0.0795 | 1.518 | **4.948** | 4.043 | 2.274 | 15.349 | **20.522** | 10.208 | 8.169 | 27.8 |
| **2D BMP Geomean (1)** | — | 0.7985 | 0.8732 | 0.8274 | 0.7494 | 0.7744 | **0.7363** | 0.5656 | 0.391 | **4.898** | 2.941 | 0.626 | 13.729 | **25.616** | 6.665 | 4.781 | 85.3 |
| **All-Dataset Geomean (7)** | — | 0.1541 | 0.2990 | 0.2045 | 0.1800 | 0.1771 | **0.1534** | 0.1053 | 1.251 | **4.941** | 3.863 | 1.891 | 15.106 | **21.182** | 9.605 | 7.568 | 32.6 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2141 | 3.215 | **4.437** | 2.624 | 14.656 | **25.353** | 9.256 | 37.5 | 35.6 | 23.0 | **49.5** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0593** | 0.1222 | 0.0784 | **8.836** | 7.742 | 5.559 | 38.588 | **40.740** | 16.813 | **311.5** | 145.4 | 123.3 | 226.6 |
| **`gharchive-2025-01-15.ndjson`** | 0.1181 | 0.1872 | **0.1072** | 0.952 | **7.274** | 6.600 | 15.305 | **47.971** | 16.998 | 32.3 | 99.8 | 98.8 | **174.2** |
| **`apache-access.clf`** | 0.0974 | 0.1362 | **0.0896** | 0.754 | **6.235** | 6.048 | 12.752 | **17.855** | 12.518 | 31.8 | 77.5 | 97.1 | **117.7** |
| **`macos-install.log`** | 0.0548 | 0.1033 | **0.0452** | 2.338 | **13.524** | 10.300 | 18.041 | **45.505** | 27.070 | 118.5 | 240.2 | 369.8 | **549.4** |
| **`pg100-shakespeare.txt`** | 0.3780 | 0.4970 | **0.3729** | 0.257 | **3.139** | 2.521 | 6.566 | **16.781** | 6.670 | 3.4 | 14.6 | 11.0 | **19.5** |
| **`kodak-truecolor-16.bmp`** | 0.7985 | 0.8127 | **0.7363** | 0.391 | **7.845** | 4.723 | 13.729 | **32.000** | 13.594 | 2.9 | 19.5 | 10.9 | **21.5** |
| **Text & Structured Geomean (6)** | **0.1171** | 0.1903 | 0.1181 | 1.518 | **6.359** | 4.973 | 15.349 | **29.622** | 13.472 | 41.2 | 72.1 | 69.3 | **116.2** |
| **2D BMP Geomean (1)** | 0.7985 | 0.8127 | **0.7363** | 0.391 | **7.845** | 4.723 | 13.729 | **32.000** | 13.594 | 2.9 | 19.5 | 10.9 | **21.5** |
| **All-Dataset Geomean (7)** | 0.1541 | 0.2342 | **0.1534** | 1.251 | **6.553** | 4.936 | 15.106 | **29.950** | 13.489 | 28.2 | 59.8 | 53.2 | **91.3** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **37.5** | 19.8 | 18.9 | 11.6 | **6.36** | 5.51 | 3.20 |
| **`nyc-taxi-2024-01.ndjson`** | **311.5** | 62.2 | 73.3 | 54.3 | **17.92** | 9.03 | 6.68 |
| **`gharchive-2025-01-15.ndjson`** | 32.3 | **53.0** | 45.4 | 44.7 | 4.98 | **8.59** | 6.22 |
| **`apache-access.clf`** | 31.8 | **51.0** | 50.6 | 47.6 | 4.62 | **7.67** | 6.28 |
| **`macos-install.log`** | 118.5 | 171.7 | **175.0** | 139.7 | 9.16 | **15.44** | 10.24 |
| **`pg100-shakespeare.txt`** | 3.4 | **7.5** | 7.0 | 4.8 | 1.65 | **3.26** | 2.11 |
| **`kodak-truecolor-16.bmp`** | 2.9 | **12.8** | 5.4 | 2.2 | 1.89 | **5.24** | 1.57 |
| **Text & Structured Geomean (6)** | **41.2** | 40.3 | 39.7 | 31.1 | 5.84 | **7.41** | 5.12 |
| **2D BMP Geomean (1)** | 2.9 | **12.8** | 5.4 | 2.2 | 1.89 | **5.24** | 1.57 |
| **All-Dataset Geomean (7)** | 28.2 | **34.2** | 29.8 | 21.4 | 4.97 | **7.05** | 4.32 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 7.297 | **3.979** | 16.529 | 2.001 | **1.279** | 4.475 | **9.8** | 8.8 | 1.4 |
| **`nyc-taxi-2024-01.ndjson`** | 2.862 | **2.106** | 7.611 | **0.725** | 0.808 | 2.302 | **216.3** | 47.7 | 13.0 |
| **`gharchive-2025-01-15.ndjson`** | 24.447 | **2.390** | 11.644 | 1.941 | **0.673** | 2.014 | 4.7 | **41.8** | 9.2 |
| **`apache-access.clf`** | 21.299 | **2.235** | 7.520 | 2.061 | **0.880** | 2.169 | 4.8 | **36.4** | 11.8 |
| **`macos-install.log`** | 12.197 | **1.574** | 5.089 | 1.533 | **0.553** | 1.192 | 27.4 | **184.0** | 56.7 |
| **`pg100-shakespeare.txt`** | 91.696 | **7.158** | 26.713 | 4.402 | **1.386** | 6.086 | 0.2 | **2.4** | 0.4 |
| **`kodak-truecolor-16.bmp`** | 76.724 | **4.360** | 34.112 | 2.281 | **0.967** | 3.504 | 0.2 | **6.2** | 0.2 |
