## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | **0.606** | 0.595 | 0.464 | 0.240 | 2.171 | **2.908** | 1.269 | 0.900 | 83.6 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **1.586** | 1.304 | 1.029 | 0.533 | **6.159** | 4.495 | 2.014 | 1.722 | 53.7 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1350 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 0.236 | **1.234** | 0.975 | 0.462 | 1.451 | **5.660** | 2.288 | 2.176 | 59.1 |
| **`apache-access.clf`** | 2.4 MB | 0.0990 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.305 | **1.570** | 1.143 | 0.567 | 1.581 | **5.871** | 2.550 | 2.210 | 46.5 |
| **`macos-install.log`** | 34.2 MB | 0.0605 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0277 | 0.559 | **2.112** | 1.454 | 0.753 | 2.139 | **7.635** | 3.439 | 3.431 | 36.4 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3977 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.083 | 0.388 | **0.396** | 0.147 | 1.104 | **2.848** | 1.155 | 0.762 | 98.9 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.117 | **0.827** | 0.525 | 0.107 | 1.544 | **4.209** | 1.012 | 0.975 | 111.5 |
| **Text & Structured Geomean (6)** | — | 0.1231 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 0.384 | **1.036** | 0.821 | 0.393 | 2.042 | **4.598** | 1.974 | 1.640 | 59.5 |
| **2D BMP Geomean (1)** | — | 0.8106 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.117 | **0.827** | 0.525 | 0.107 | 1.544 | **4.209** | 1.012 | 0.975 | 111.5 |
| **All-Dataset Geomean (7)** | — | 0.1612 | 0.2990 | 0.2045 | 0.1800 | 0.1798 | **0.1563** | 0.1052 | 0.324 | **1.003** | 0.770 | 0.327 | 1.962 | **4.540** | 1.795 | 1.523 | 65.1 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2132 | **0.606** | 0.601 | 0.503 | 2.171 | **3.793** | 1.354 | 6.3 | 5.1 | 3.9 | **7.1** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0592** | 0.1222 | 0.0783 | **1.586** | 1.304 | 1.169 | 6.159 | **7.931** | 2.578 | **52.8** | 26.3 | 22.2 | 41.1 |
| **`gharchive-2025-01-15.ndjson`** | 0.1350 | 0.1872 | **0.1150** | 0.236 | **1.294** | 1.057 | 1.451 | **7.170** | 2.840 | 4.3 | 16.3 | 15.1 | **26.5** |
| **`apache-access.clf`** | 0.0990 | 0.1362 | **0.0878** | 0.305 | **1.708** | 1.288 | 1.581 | **7.962** | 3.277 | 7.0 | 27.1 | 23.4 | **42.0** |
| **`macos-install.log`** | 0.0605 | 0.1033 | **0.0484** | 0.559 | **2.157** | 1.647 | 2.139 | **10.016** | 4.304 | 18.1 | 45.0 | 55.0 | **96.1** |
| **`pg100-shakespeare.txt`** | 0.3977 | 0.4970 | **0.3745** | 0.083 | 0.432 | **0.439** | 1.104 | **3.527** | 1.482 | 0.8 | 2.5 | 2.2 | **3.3** |
| **`kodak-truecolor-16.bmp`** | 0.8106 | 0.8127 | **0.7452** | 0.117 | **1.394** | 1.049 | 1.544 | **6.168** | 3.296 | 0.5 | 3.6 | 2.5 | **3.9** |
| **Text & Structured Geomean (6)** | 0.1231 | 0.1903 | **0.1205** | 0.384 | **1.083** | 0.913 | 2.042 | **6.269** | 2.432 | 7.2 | 13.7 | 12.4 | **21.7** |
| **2D BMP Geomean (1)** | 0.8106 | 0.8127 | **0.7452** | 0.117 | **1.394** | 1.049 | 1.544 | **6.168** | 3.296 | 0.5 | 3.6 | 2.5 | **3.9** |
| **All-Dataset Geomean (7)** | 0.1612 | 0.2342 | **0.1563** | 0.324 | **1.123** | 0.931 | 1.962 | **6.254** | 2.540 | 4.9 | 11.3 | 9.8 | **17.0** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **6.3** | 3.1 | 3.1 | 1.9 | **1.93** | 1.60 | 0.96 |
| **`nyc-taxi-2024-01.ndjson`** | **52.8** | 12.7 | 14.7 | 9.5 | **5.48** | 3.14 | 2.09 |
| **`gharchive-2025-01-15.ndjson`** | 4.3 | **11.7** | 9.5 | 8.0 | 1.36 | **3.14** | 2.00 |
| **`apache-access.clf`** | 7.0 | **17.5** | 13.8 | 10.4 | 1.70 | **3.76** | 2.27 |
| **`macos-install.log`** | 18.1 | **32.2** | 25.4 | 27.3 | 2.70 | **5.05** | 3.52 |
| **`pg100-shakespeare.txt`** | 0.8 | **1.7** | 1.5 | 0.8 | 0.61 | **1.21** | 0.65 |
| **`kodak-truecolor-16.bmp`** | 0.5 | **2.1** | 0.9 | 0.4 | 0.61 | **1.59** | 0.51 |
| **Text & Structured Geomean (6)** | 7.2 | **8.7** | 7.9 | 5.7 | 1.85 | **2.67** | 1.66 |
| **2D BMP Geomean (1)** | 0.5 | **2.1** | 0.9 | 0.4 | 0.61 | **1.59** | 0.51 |
| **All-Dataset Geomean (7)** | 4.9 | **7.1** | 5.7 | 3.9 | 1.58 | **2.48** | 1.40 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 6.408 | **4.970** | 18.770 | 2.252 | **1.321** | 5.179 | **1.7** | 1.2 | 0.2 |
| **`nyc-taxi-2024-01.ndjson`** | 2.745 | **2.522** | 8.300 | **0.812** | 0.956 | 2.655 | **35.3** | 8.2 | 2.0 |
| **`gharchive-2025-01-15.ndjson`** | 17.033 | **2.868** | 10.003 | 2.504 | **0.811** | 2.159 | 0.7 | **7.7** | 1.7 |
| **`apache-access.clf`** | 14.505 | **2.333** | 7.538 | 2.531 | **0.766** | 2.081 | 1.2 | **13.1** | 2.6 |
| **`macos-install.log`** | 8.378 | **1.701** | 6.070 | 1.807 | **0.598** | 1.432 | 4.6 | **31.9** | 9.2 |
| **`pg100-shakespeare.txt`** | 56.501 | **8.289** | 29.670 | 4.172 | **1.515** | 6.150 | 0.0 | **0.5** | 0.1 |
| **`kodak-truecolor-16.bmp`** | 38.639 | **4.788** | 40.911 | 2.769 | **1.027** | 3.666 | 0.1 | **1.0** | 0.0 |
