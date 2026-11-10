## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` sweeps `32 / 64 / 112 / 256 KiB` dictionaries per dataset via `ZDICT_trainFromBuffer` and selects the budget achieving the lowest total compressed size (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with single-budget `ZDICT` training time of the selected dictionary reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 256 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.6 MB | **0.0963** | 0.3254 | 0.1459 | 0.1415 | 0.1794 | 0.1495 | 0.1066 | 0.493 | **0.793** | 0.749 | 0.300 | 1.956 | **4.479** | 1.595 | 1.344 | 18.1 |
| **`us_confirmed_33m.csv`** | 35.2 MB | **0.0456** | 0.2363 | 0.1032 | 0.0897 | 0.1186 | 0.0938 | 0.0618 | 0.759 | **1.199** | 1.055 | 0.451 | 3.220 | **4.505** | 2.161 | 1.680 | 14.1 |
| **`vxg-src-nonrepeating.rs`** | 2.9 MB | 0.1917 | 0.3333 | 0.2322 | 0.1939 | 0.2171 | **0.1872** | 0.1342 | 0.079 | **0.704** | 0.543 | 0.291 | 1.304 | **3.235** | 1.450 | 1.202 | 20.0 |
| **All-Dataset Geomean (3)** | — | **0.0944** | 0.2948 | 0.1518 | 0.1350 | 0.1665 | 0.1379 | 0.0960 | 0.310 | **0.875** | 0.754 | 0.340 | 2.018 | **4.027** | 1.710 | 1.395 | 17.2 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **0.0963** | 0.2794 | 0.1415 | 0.493 | 0.854 | **0.984** | 1.956 | **5.328** | 2.150 | 10.2 | 7.6 | 10.3 | **16.2** |
| **`us_confirmed_33m.csv`** | **0.0456** | 0.2071 | 0.0897 | 0.759 | 1.211 | **1.346** | 3.220 | **6.096** | 2.900 | **34.3** | 13.1 | 22.0 | 31.9 |
| **`vxg-src-nonrepeating.rs`** | 0.1917 | 0.2476 | **0.1872** | 0.079 | **0.726** | 0.562 | 1.304 | **4.423** | 1.647 | 1.7 | 7.2 | 5.1 | **9.6** |
| **All-Dataset Geomean (3)** | **0.0944** | 0.2429 | 0.1335 | 0.310 | **0.909** | 0.907 | 2.018 | **5.237** | 2.174 | 8.4 | 9.0 | 10.5 | **17.0** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | **10.2** | 5.8 | 7.5 | 3.5 | 2.16 | **2.22** | 1.31 |
| **`us_confirmed_33m.csv`** | **34.3** | 9.8 | 14.6 | 7.3 | **3.77** | 2.84 | 1.86 |
| **`vxg-src-nonrepeating.rs`** | 1.7 | **4.5** | 3.8 | 2.7 | 0.81 | **1.90** | 1.17 |
| **All-Dataset Geomean (3)** | **8.4** | 6.4 | 7.5 | 4.1 | 1.88 | **2.29** | 1.42 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`time-series-19-covid-combined.csv`** | 8.880 | **4.386** | 15.541 | 2.489 | **0.926** | 3.686 | 2.2 | **2.9** | 0.5 |
| **`us_confirmed_33m.csv`** | 6.078 | **2.736** | 10.594 | 1.483 | **0.972** | 2.885 | **11.4** | 6.0 | 1.3 |
| **`vxg-src-nonrepeating.rs`** | 49.632 | **4.031** | 15.923 | 3.144 | **1.241** | 4.036 | 0.1 | **2.0** | 0.3 |
