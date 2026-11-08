## VXG v4 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1807** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | **0.524** | 0.449 | 0.435 | 0.219 | 1.257 | **2.482** | 0.920 | 0.718 | 108.6 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **1.654** | 1.293 | 0.993 | 0.515 | **4.992** | 4.360 | 2.047 | 1.703 | 77.4 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1985 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 0.634 | **1.055** | 0.829 | 0.337 | 2.341 | **3.528** | 1.993 | 1.701 | 80.3 |
| **`apache-access.clf`** | 2.4 MB | 0.1198 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.401 | **1.202** | 1.078 | 0.533 | 2.107 | **5.567** | 2.463 | 1.835 | 64.3 |
| **`macos-install.log`** | 34.2 MB | 0.0948 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0277 | 1.404 | **1.914** | 0.977 | 0.731 | 3.748 | **7.352** | 3.261 | 3.309 | 49.3 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.4424 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.398 | 0.399 | **0.399** | 0.147 | 1.803 | **2.927** | 1.150 | 0.756 | 97.8 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 1.207 | **0.860** | 0.527 | 0.106 | 10.099 | **4.435** | 1.029 | 0.971 | 101.9 |
| **Text & Structured Geomean (6)** | — | 0.1484 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 0.705 | **0.908** | 0.730 | 0.360 | 2.436 | **4.074** | 1.806 | 1.456 | 77.1 |
| **2D BMP Geomean (1)** | — | 0.9256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 1.207 | **0.860** | 0.527 | 0.106 | 10.099 | **4.435** | 1.029 | 0.971 | 101.9 |
| **All-Dataset Geomean (7)** | — | 0.1927 | 0.2990 | 0.2045 | 0.1800 | 0.1798 | **0.1563** | 0.1052 | 0.762 | **0.901** | 0.696 | 0.303 | 2.985 | **4.124** | 1.666 | 1.375 | 80.2 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1807** | 0.2976 | 0.2132 | 0.524 | **0.563** | 0.497 | 1.257 | **3.853** | 1.315 | 4.5 | 4.9 | 3.8 | **6.9** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0592** | 0.1222 | 0.0783 | **1.654** | 1.293 | 1.180 | 4.992 | **7.690** | 2.421 | **48.5** | 25.8 | 21.6 | 40.3 |
| **`gharchive-2025-01-15.ndjson`** | 0.1985 | 0.1872 | **0.1150** | 0.634 | **1.055** | 0.913 | 2.341 | **6.029** | 2.521 | 6.1 | 13.5 | 13.2 | **21.9** |
| **`apache-access.clf`** | 0.1198 | 0.1362 | **0.0878** | 0.401 | **1.621** | 1.232 | 2.107 | **7.448** | 2.999 | 7.7 | 25.5 | 21.9 | **39.6** |
| **`macos-install.log`** | 0.0948 | 0.1033 | **0.0484** | 1.404 | **2.160** | 1.443 | 3.748 | **9.558** | 4.122 | 24.2 | 44.0 | 50.4 | **93.9** |
| **`pg100-shakespeare.txt`** | 0.4424 | 0.4970 | **0.3745** | 0.398 | 0.431 | **0.436** | 1.803 | **3.513** | 1.479 | 1.9 | 2.5 | 2.1 | **3.3** |
| **`kodak-truecolor-16.bmp`** | 0.9256 | 0.8127 | **0.7452** | 1.207 | **1.451** | 1.052 | 10.099 | **6.583** | 3.306 | 3.8 | 3.8 | 2.5 | **4.1** |
| **Text & Structured Geomean (6)** | 0.1484 | 0.1903 | **0.1205** | 0.705 | **1.025** | 0.864 | 2.436 | **5.957** | 2.297 | 8.8 | 13.0 | 11.7 | **20.5** |
| **2D BMP Geomean (1)** | 0.9256 | 0.8127 | **0.7452** | 1.207 | **1.451** | 1.052 | 10.099 | **6.583** | 3.306 | 3.8 | 3.8 | 2.5 | **4.1** |
| **All-Dataset Geomean (7)** | 0.1927 | 0.2342 | **0.1563** | 0.762 | **1.077** | 0.889 | 2.985 | **6.043** | 2.419 | 7.8 | 10.9 | 9.4 | **16.3** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **4.5** | 2.5 | 2.6 | 1.6 | **1.54** | 1.38 | 0.87 |
| **`nyc-taxi-2024-01.ndjson`** | **48.5** | 12.5 | 14.6 | 9.3 | **5.19** | 3.10 | 2.06 |
| **`gharchive-2025-01-15.ndjson`** | 6.1 | **8.6** | 8.2 | 6.0 | 1.96 | **2.55** | 1.66 |
| **`apache-access.clf`** | 7.7 | **14.9** | 13.2 | 9.2 | 1.92 | **3.38** | 2.09 |
| **`macos-install.log`** | 24.2 | **30.0** | 20.2 | 26.4 | 3.82 | **4.83** | 3.45 |
| **`pg100-shakespeare.txt`** | **1.9** | 1.7 | 1.5 | 0.8 | 1.17 | **1.23** | 0.65 |
| **`kodak-truecolor-16.bmp`** | 3.8 | **2.2** | 0.9 | 0.4 | 2.36 | **1.63** | 0.51 |
| **Text & Structured Geomean (6)** | **8.8** | 7.7 | 7.1 | 5.1 | 2.26 | **2.46** | 1.55 |
| **2D BMP Geomean (1)** | 3.8 | **2.2** | 0.9 | 0.4 | 2.36 | **1.63** | 0.51 |
| **All-Dataset Geomean (7)** | **7.8** | 6.4 | 5.3 | 3.6 | 2.28 | **2.32** | 1.32 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 7.676 | **3.645** | 17.821 | 3.231 | **0.898** | 4.777 | 0.9 | **1.4** | 0.2 |
| **`nyc-taxi-2024-01.ndjson`** | 2.677 | **2.493** | 8.071 | **0.727** | 0.900 | 2.596 | **34.8** | 8.3 | 2.0 |
| **`gharchive-2025-01-15.ndjson`** | 4.516 | **2.579** | 8.079 | 1.267 | **0.674** | 1.709 | 2.6 | **6.5** | 1.6 |
| **`apache-access.clf`** | 7.034 | **2.361** | 7.496 | 1.785 | **0.727** | 2.039 | 2.2 | **11.4** | 2.4 |
| **`macos-install.log`** | 2.772 | **1.653** | 6.014 | 0.910 | **0.610** | 1.324 | 15.2 | **29.9** | 9.4 |
| **`pg100-shakespeare.txt`** | 10.785 | **8.880** | 28.922 | 2.140 | **1.454** | 5.791 | 0.4 | **0.5** | 0.1 |
| **`kodak-truecolor-16.bmp`** | **3.326** | 5.072 | 39.689 | **0.405** | 1.016 | 3.487 | 3.2 | **1.0** | 0.0 |
