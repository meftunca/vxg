## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1794 | 0.1495 | 0.1066 | 1.901 | **5.188** | 4.372 | 2.056 | 10.651 | **24.125** | 7.798 | 6.676 | 17.0 |
| **`us_confirmed_33m.csv`** | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.1186 | 0.0938 | 0.0619 | 4.867 | **8.947** | 5.436 | 2.752 | 17.374 | **32.161** | 11.450 | 8.971 | 13.2 |
| **`vxg-src-nonrepeating.rs`** | 2.6 MB | 0.1933 | 0.3350 | 0.2334 | 0.1954 | 0.2181 | **0.1889** | 0.1355 | 0.424 | 3.486 | **3.706** | 1.325 | 7.523 | **10.713** | 7.599 | 6.424 | 18.9 |
| **All-Dataset Geomean (3)** | — | **0.0947** | 0.2953 | 0.1521 | 0.1354 | 0.1668 | 0.1384 | 0.0963 | 1.577 | **5.450** | 4.449 | 1.957 | 11.166 | **20.257** | 8.787 | 7.273 | 16.2 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **0.0963** | 0.2794 | 0.1415 | 1.901 | 5.966 | **6.883** | 10.651 | **24.125** | 12.431 | 46.7 | 42.9 | 65.4 | **91.1** |
| **`us_confirmed_33m.csv`** | **0.0456** | 0.2071 | 0.0897 | 4.867 | 9.167 | **9.472** | 17.374 | **32.651** | 19.619 | **201.6** | 83.5 | 151.9 | 195.9 |
| **`vxg-src-nonrepeating.rs`** | 0.1933 | 0.2499 | **0.1889** | 0.424 | **4.799** | 3.706 | 7.523 | **16.253** | 8.000 | 9.2 | 35.3 | 28.8 | **46.8** |
| **All-Dataset Geomean (3)** | **0.0947** | 0.2436 | 0.1338 | 1.577 | **6.402** | 6.228 | 11.166 | **23.394** | 12.495 | 44.3 | 50.2 | 65.9 | **94.2** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **46.7** | 34.4 | 40.0 | 20.7 | 5.95 | **7.27** | 4.25 |
| **`us_confirmed_33m.csv`** | **201.6** | 71.8 | 76.4 | 41.9 | **12.28** | 10.68 | 5.93 |
| **`vxg-src-nonrepeating.rs`** | 9.2 | 18.2 | **22.7** | 13.4 | 2.55 | **4.81** | 3.39 |
| **All-Dataset Geomean (3)** | **44.3** | 35.6 | 41.1 | 22.6 | 5.71 | **7.20** | 4.40 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 7.855 | **3.873** | 13.959 | 2.142 | **0.919** | 3.087 | 11.4 | **18.2** | 3.1 |
| **`us_confirmed_33m.csv`** | 5.587 | **2.529** | 9.279 | 1.465 | **0.893** | 2.491 | **70.5** | 47.8 | 8.7 |
| **`vxg-src-nonrepeating.rs`** | 39.472 | **3.706** | 12.983 | 3.047 | **1.300** | 3.688 | 0.8 | **8.3** | 1.9 |
