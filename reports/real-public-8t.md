## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 23.0 MB | **0.1829** | 0.4241 | 0.2440 | 0.2194 | 0.2414 | 0.2132 | 0.1617 | 3.425 | **3.774** | 2.938 | 1.604 | 13.683 | **18.736** | 7.154 | 4.991 | 81.7 |
| **`nyc-taxi-2024-01.ndjson`** | 52.3 MB | **0.0592** | 0.1899 | 0.0977 | 0.0823 | 0.1010 | 0.0783 | 0.0554 | **9.034** | 7.979 | 5.839 | 3.617 | **38.161** | 27.805 | 11.617 | 10.220 | 49.2 |
| **`gharchive-2025-01-15.ndjson`** | 50.3 MB | 0.1231 | 0.2256 | 0.1577 | 0.1414 | 0.1254 | **0.1150** | 0.0633 | 1.852 | **7.148** | 6.132 | 3.009 | 14.349 | **31.371** | 12.137 | 13.369 | 55.7 |
| **`apache-access.clf`** | 2.4 MB | 0.1037 | 0.1730 | 0.1233 | 0.1062 | 0.1075 | **0.0878** | 0.0535 | 0.997 | **6.204** | 4.861 | 2.409 | 12.939 | **16.055** | 9.153 | 7.986 | 43.1 |
| **`macos-install.log`** | 34.2 MB | 0.0594 | 0.1249 | 0.0882 | 0.0770 | 0.0590 | **0.0484** | 0.0278 | 3.965 | **11.572** | 7.164 | 4.198 | 17.466 | **43.963** | 18.830 | 17.677 | 36.0 |
| **`pg100-shakespeare.txt`** | 5.6 MB | 0.3884 | 0.6223 | 0.4421 | 0.3918 | 0.4005 | **0.3745** | 0.3000 | 0.460 | **2.273** | 2.271 | 0.821 | 5.390 | **12.356** | 7.532 | 5.085 | 106.1 |
| **`kodak-truecolor-16.bmp`** | 18.9 MB | 0.8256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.873 | **5.271** | 2.969 | 0.620 | 13.907 | **21.288** | 5.797 | 5.860 | 101.9 |
| **Text & Structured Geomean (6)** | — | 0.1213 | 0.2500 | 0.1620 | 0.1419 | 0.1407 | **0.1205** | 0.0795 | 2.169 | **5.723** | 4.501 | 2.292 | 14.456 | **22.855** | 10.460 | 8.878 | 57.7 |
| **2D BMP Geomean (1)** | — | 0.8256 | 0.8732 | 0.8274 | 0.7494 | 0.7834 | **0.7452** | 0.5656 | 0.873 | **5.271** | 2.969 | 0.620 | 13.907 | **21.288** | 5.797 | 5.860 | 101.9 |
| **All-Dataset Geomean (7)** | — | 0.1596 | 0.2990 | 0.2045 | 0.1800 | 0.1798 | **0.1563** | 0.1053 | 1.905 | **5.656** | 4.242 | 1.902 | 14.376 | **22.625** | 9.614 | 8.366 | 62.6 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **0.1829** | 0.2976 | 0.2132 | 3.425 | **3.774** | 3.239 | 13.683 | **20.151** | 8.474 | 37.4 | 29.3 | 24.6 | **40.9** |
| **`nyc-taxi-2024-01.ndjson`** | **0.0592** | 0.1222 | 0.0783 | **9.034** | 8.711 | 6.908 | 38.161 | **38.936** | 14.705 | **313.5** | 150.8 | 128.7 | 235.2 |
| **`gharchive-2025-01-15.ndjson`** | 0.1231 | 0.1872 | **0.1150** | 1.852 | **9.023** | 6.132 | 14.349 | **39.422** | 15.728 | 41.9 | 100.8 | 85.4 | **164.0** |
| **`apache-access.clf`** | 0.1037 | 0.1362 | **0.0878** | 0.997 | **6.260** | 4.861 | 12.939 | **17.796** | 12.025 | 34.7 | 77.5 | 87.0 | **120.2** |
| **`macos-install.log`** | 0.0594 | 0.1033 | **0.0484** | 3.965 | **11.572** | 9.875 | 17.466 | **45.770** | 23.988 | 140.0 | 222.9 | 318.1 | **475.7** |
| **`pg100-shakespeare.txt`** | 0.3884 | 0.4970 | **0.3745** | 0.460 | **2.857** | 2.688 | 5.390 | **15.440** | 7.532 | 4.1 | 13.4 | 12.0 | **17.7** |
| **`kodak-truecolor-16.bmp`** | 0.8256 | 0.8127 | **0.7452** | 0.873 | **8.283** | 6.487 | 13.907 | **28.308** | 15.494 | 4.2 | 18.8 | 13.5 | **20.5** |
| **Text & Structured Geomean (6)** | 0.1213 | 0.1903 | **0.1205** | 2.169 | **6.281** | 5.105 | 14.456 | **27.018** | 12.731 | 46.2 | 68.4 | 66.9 | **108.1** |
| **2D BMP Geomean (1)** | 0.8256 | 0.8127 | **0.7452** | 0.873 | **8.283** | 6.487 | 13.907 | **28.308** | 15.494 | 4.2 | 18.8 | 13.5 | **20.5** |
| **All-Dataset Geomean (7)** | 0.1596 | 0.2342 | **0.1563** | 1.905 | **6.534** | 5.283 | 14.376 | **27.199** | 13.094 | 32.8 | 56.9 | 53.2 | **85.3** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | **37.4** | 19.8 | 18.8 | 11.7 | **6.35** | 5.50 | 3.21 |
| **`nyc-taxi-2024-01.ndjson`** | **313.5** | 78.4 | 84.3 | 60.2 | **17.99** | 10.53 | 7.15 |
| **`gharchive-2025-01-15.ndjson`** | 41.9 | **66.4** | 54.7 | 50.6 | 6.00 | **9.98** | 6.85 |
| **`apache-access.clf`** | 34.7 | **57.7** | 54.1 | 40.8 | 4.99 | **8.32** | 5.63 |
| **`macos-install.log`** | 140.0 | **180.6** | 131.8 | 146.1 | 10.52 | **15.97** | 10.80 |
| **`pg100-shakespeare.txt`** | 4.1 | 8.5 | **9.4** | 5.1 | 1.86 | **3.56** | 2.18 |
| **`kodak-truecolor-16.bmp`** | 4.2 | **12.1** | 5.0 | 2.4 | 2.45 | **5.05** | 1.67 |
| **Text & Structured Geomean (6)** | **46.2** | 45.7 | 42.4 | 32.1 | 6.37 | **8.06** | 5.25 |
| **2D BMP Geomean (1)** | 4.2 | **12.1** | 5.0 | 2.4 | 2.45 | **5.05** | 1.67 |
| **All-Dataset Geomean (7)** | 32.8 | **37.8** | 31.2 | 22.2 | 5.56 | **7.54** | 4.46 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nyc-taxi-2024-01.csv`** | 6.183 | **4.459** | 17.080 | 2.097 | **1.292** | 4.526 | **10.4** | 8.3 | 1.3 |
| **`nyc-taxi-2024-01.ndjson`** | 2.395 | **2.229** | 7.479 | **0.756** | 0.879 | 2.410 | **233.0** | 56.0 | 14.2 |
| **`gharchive-2025-01-15.ndjson`** | 12.400 | **2.497** | 8.862 | 1.938 | **0.709** | 1.954 | 8.5 | **49.9** | 12.2 |
| **`apache-access.clf`** | 11.171 | **2.318** | 7.229 | 2.061 | **0.918** | 2.242 | 7.2 | **39.5** | 10.1 |
| **`macos-install.log`** | 5.767 | **1.636** | 5.239 | 1.492 | **0.563** | 1.228 | 47.7 | **188.2** | 57.6 |
| **`pg100-shakespeare.txt`** | 45.240 | **7.410** | 26.380 | 4.305 | **1.436** | 5.254 | 0.3 | **2.6** | 0.4 |
| **`kodak-truecolor-16.bmp`** | 29.224 | **4.397** | 30.367 | 1.551 | **0.935** | 3.219 | 0.6 | **6.0** | 0.2 |
