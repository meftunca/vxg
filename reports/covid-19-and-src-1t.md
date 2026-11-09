## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1794 | 0.1495 | 0.1066 | 0.496 | **0.807** | 0.778 | 0.309 | 1.595 | **4.516** | 1.626 | 1.357 | 16.6 |
| **`us_confirmed_33m.csv`** | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.1186 | 0.0938 | 0.0618 | 0.795 | **1.236** | 1.085 | 0.456 | 2.451 | **4.663** | 2.244 | 1.683 | 12.3 |
| **`vxg-src-nonrepeating.rs`** | 2.6 MB | 0.1933 | 0.3350 | 0.2334 | 0.1954 | 0.2181 | **0.1889** | 0.1355 | 0.086 | **0.712** | 0.560 | 0.297 | 1.252 | **3.375** | 1.457 | 1.200 | 18.1 |
| **All-Dataset Geomean (3)** | — | **0.0947** | 0.2953 | 0.1521 | 0.1354 | 0.1668 | 0.1384 | 0.0963 | 0.324 | **0.892** | 0.779 | 0.347 | 1.698 | **4.142** | 1.745 | 1.400 | 15.5 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **0.0963** | 0.2794 | 0.1415 | 0.496 | 0.866 | **0.988** | 1.595 | **5.532** | 2.112 | 9.2 | 7.8 | 10.2 | **16.5** |
| **`us_confirmed_33m.csv`** | **0.0456** | 0.2071 | 0.0897 | 0.795 | 1.236 | **1.352** | 2.451 | **6.334** | 3.000 | 30.6 | 13.5 | 22.4 | **32.6** |
| **`vxg-src-nonrepeating.rs`** | 0.1933 | 0.2499 | **0.1889** | 0.086 | **0.756** | 0.572 | 1.252 | **4.461** | 1.642 | 1.7 | 7.3 | 5.1 | **9.7** |
| **All-Dataset Geomean (3)** | **0.0947** | 0.2436 | 0.1338 | 0.324 | **0.932** | 0.914 | 1.698 | **5.387** | 2.183 | 7.8 | 9.2 | 10.6 | **17.4** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **9.2** | 5.9 | 7.7 | 3.6 | 2.02 | **2.24** | 1.33 |
| **`us_confirmed_33m.csv`** | **30.6** | 10.2 | 15.1 | 7.4 | **3.50** | 2.90 | 1.86 |
| **`vxg-src-nonrepeating.rs`** | 1.7 | **4.6** | 3.9 | 2.7 | 0.82 | **1.93** | 1.18 |
| **All-Dataset Geomean (3)** | **7.8** | 6.5 | 7.7 | 4.2 | 1.80 | **2.32** | 1.43 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.033 | **4.229** | 14.808 | 2.428 | **0.873** | 3.709 | 2.1 | **3.1** | 0.5 |
| **`us_confirmed_33m.csv`** | 6.273 | **2.686** | 9.851 | 1.609 | **0.926** | 2.649 | **9.6** | 6.4 | 1.4 |
| **`vxg-src-nonrepeating.rs`** | 47.019 | **4.170** | 15.452 | 3.208 | **1.321** | 3.575 | 0.1 | **2.0** | 0.4 |
