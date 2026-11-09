## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1794 | 0.1495 | 0.1066 | 1.558 | **3.431** | 2.695 | 1.217 | 5.848 | **15.919** | 4.665 | 5.180 | 21.0 |
| **`us_confirmed_33m.csv`** | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.1186 | 0.0938 | 0.0619 | 3.334 | 3.792 | **4.166** | 1.454 | 13.658 | **16.111** | 9.240 | 6.790 | 15.6 |
| **`vxg-src-nonrepeating.rs`** | 2.6 MB | 0.1933 | 0.3350 | 0.2334 | 0.1954 | 0.2181 | **0.1889** | 0.1355 | 0.423 | **4.127** | 3.628 | 1.765 | 9.632 | **13.841** | 5.696 | 6.271 | 18.0 |
| **All-Dataset Geomean (3)** | — | **0.0947** | 0.2953 | 0.1521 | 0.1354 | 0.1668 | 0.1384 | 0.0963 | 1.301 | **3.773** | 3.441 | 1.462 | 9.163 | **15.255** | 6.262 | 6.042 | 18.1 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **0.0963** | 0.2794 | 0.1415 | 1.558 | **3.431** | 3.052 | 5.848 | **15.919** | 7.714 | 31.3 | 26.5 | 34.3 | **52.2** |
| **`us_confirmed_33m.csv`** | **0.0456** | 0.2071 | 0.0897 | 3.334 | **6.015** | 4.166 | 13.658 | **20.465** | 9.308 | **147.9** | 53.6 | 69.4 | 123.6 |
| **`vxg-src-nonrepeating.rs`** | 0.1933 | 0.2499 | **0.1889** | 0.423 | **4.638** | 3.751 | 9.632 | **13.841** | 8.196 | 10.4 | 32.1 | 29.4 | **42.4** |
| **All-Dataset Geomean (3)** | **0.0947** | 0.2436 | 0.1338 | 1.301 | **4.574** | 3.627 | 9.163 | **16.521** | 8.380 | 36.5 | 35.7 | 41.2 | **64.9** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **31.3** | 22.7 | 24.3 | 14.0 | 4.56 | **5.52** | 3.28 |
| **`us_confirmed_33m.csv`** | **147.9** | 33.1 | 60.1 | 26.5 | **9.99** | 6.37 | 4.37 |
| **`vxg-src-nonrepeating.rs`** | 10.4 | **22.6** | 19.5 | 15.3 | 2.76 | **5.55** | 3.70 |
| **All-Dataset Geomean (3)** | **36.5** | 25.7 | 30.5 | 17.8 | 5.01 | **5.80** | 3.76 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 5.566 | **2.376** | 8.024 | 1.278 | **0.607** | 1.959 | 11.8 | **18.9** | 3.5 |
| **`us_confirmed_33m.csv`** | 3.474 | **1.272** | 5.466 | 0.951 | **0.486** | 1.513 | **81.4** | 42.1 | 9.2 |
| **`vxg-src-nonrepeating.rs`** | 39.263 | **4.175** | 14.211 | 3.266 | **1.333** | 3.684 | 0.9 | **9.6** | 2.1 |
