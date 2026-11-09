## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1794 | 0.1495 | 0.1066 | 0.517 | **0.819** | 0.751 | 0.298 | 1.597 | **4.602** | 1.619 | 1.351 | 17.3 |
| **`us_confirmed_33m.csv`** | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.1186 | 0.0938 | 0.0618 | 0.797 | **1.209** | 1.100 | 0.463 | 2.536 | **4.690** | 2.267 | 1.749 | 12.9 |
| **`vxg-src-nonrepeating.rs`** | 2.6 MB | 0.1933 | 0.3350 | 0.2334 | 0.1954 | 0.2181 | **0.1889** | 0.1355 | 0.089 | **0.734** | 0.551 | 0.301 | 1.303 | **3.361** | 1.435 | 1.210 | 17.6 |
| **All-Dataset Geomean (3)** | — | **0.0947** | 0.2953 | 0.1521 | 0.1354 | 0.1668 | 0.1384 | 0.0963 | 0.332 | **0.899** | 0.769 | 0.346 | 1.741 | **4.171** | 1.740 | 1.420 | 15.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **0.0963** | 0.2794 | 0.1415 | 0.517 | 0.883 | **1.021** | 1.597 | **5.513** | 2.160 | 9.4 | 7.9 | 10.5 | **16.8** |
| **`us_confirmed_33m.csv`** | **0.0456** | 0.2071 | 0.0897 | 0.797 | 1.268 | **1.399** | 2.536 | **6.336** | 2.953 | 31.2 | 13.7 | 22.6 | **33.2** |
| **`vxg-src-nonrepeating.rs`** | 0.1933 | 0.2499 | **0.1889** | 0.089 | **0.734** | 0.576 | 1.303 | **4.518** | 1.652 | 1.8 | 7.3 | 5.2 | **9.6** |
| **All-Dataset Geomean (3)** | **0.0947** | 0.2436 | 0.1338 | 0.332 | 0.937 | **0.937** | 1.741 | **5.404** | 2.192 | 8.0 | 9.2 | 10.7 | **17.5** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **9.4** | 6.0 | 7.6 | 3.5 | 2.05 | **2.26** | 1.31 |
| **`us_confirmed_33m.csv`** | **31.2** | 10.1 | 15.3 | 7.6 | **3.54** | 2.88 | 1.90 |
| **`vxg-src-nonrepeating.rs`** | 1.8 | **4.7** | 3.8 | 2.8 | 0.84 | **1.95** | 1.19 |
| **All-Dataset Geomean (3)** | **8.0** | 6.6 | 7.6 | 4.2 | 1.83 | **2.33** | 1.43 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 9.200 | **4.351** | 13.665 | 2.455 | **0.981** | 3.710 | 2.0 | **2.9** | 0.5 |
| **`us_confirmed_33m.csv`** | 6.309 | **2.688** | 10.292 | 1.756 | **0.962** | 2.937 | **9.4** | 6.3 | 1.4 |
| **`vxg-src-nonrepeating.rs`** | 48.134 | **4.518** | 15.674 | 3.595 | **1.325** | 3.674 | 0.1 | **1.9** | 0.4 |
