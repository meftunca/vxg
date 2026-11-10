## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1794 | 0.1495 | 0.1066 | 2.082 | **5.056** | 4.824 | 1.421 | 11.048 | **19.275** | 9.309 | 5.681 | 18.8 |
| **`us_confirmed_33m.csv`** | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.1186 | 0.0938 | 0.0619 | 4.378 | **8.318** | 6.422 | 2.982 | 24.655 | **24.678** | 10.355 | 11.043 | 13.3 |
| **`vxg-src-nonrepeating.rs`** | 2.9 MB | 0.1917 | 0.3333 | 0.2322 | 0.1939 | 0.2171 | **0.1872** | 0.1342 | 0.384 | **4.214** | 3.157 | 1.221 | 7.257 | **13.038** | 7.603 | 6.418 | 19.1 |
| **All-Dataset Geomean (3)** | — | **0.0944** | 0.2948 | 0.1518 | 0.1350 | 0.1665 | 0.1379 | 0.0960 | 1.518 | **5.617** | 4.607 | 1.729 | 12.550 | **18.373** | 9.016 | 7.384 | 16.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **0.0963** | 0.2794 | 0.1415 | 2.082 | 5.056 | **6.161** | 11.048 | **19.275** | 11.952 | 49.8 | 35.3 | 60.7 | **77.0** |
| **`us_confirmed_33m.csv`** | **0.0456** | 0.2071 | 0.0897 | 4.378 | **8.318** | 7.249 | 24.655 | **29.707** | 15.989 | **227.8** | 75.9 | 120.0 | 175.2 |
| **`vxg-src-nonrepeating.rs`** | 0.1917 | 0.2476 | **0.1872** | 0.384 | **4.214** | 3.268 | 7.257 | **13.038** | 8.172 | 8.7 | 29.9 | 27.6 | **39.6** |
| **All-Dataset Geomean (3)** | **0.0944** | 0.2429 | 0.1335 | 1.518 | **5.617** | 5.265 | 12.550 | **19.544** | 11.602 | 46.2 | 43.1 | 58.6 | **81.1** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **49.8** | 30.3 | 45.9 | 15.8 | 6.20 | **6.69** | 3.56 |
| **`us_confirmed_33m.csv`** | **227.8** | 60.6 | 79.0 | 48.4 | **13.33** | 9.54 | 6.52 |
| **`vxg-src-nonrepeating.rs`** | 8.7 | **22.2** | 21.1 | 12.9 | 2.44 | **5.48** | 3.30 |
| **All-Dataset Geomean (3)** | **46.2** | 34.5 | 42.5 | 21.5 | 5.86 | **7.05** | 4.25 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.079 | **3.942** | 13.466 | 2.183 | **0.933** | 3.219 | 11.9 | **15.8** | 2.4 |
| **`us_confirmed_33m.csv`** | 5.500 | **2.578** | 9.776 | 1.435 | **0.878** | 2.735 | **81.1** | 40.3 | 9.4 |
| **`vxg-src-nonrepeating.rs`** | 44.038 | **4.314** | 14.038 | 2.989 | **1.384** | 3.901 | 0.8 | **9.1** | 1.7 |
