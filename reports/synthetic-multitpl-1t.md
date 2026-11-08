## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 56.5 MB | **0.0914** | 0.2898 | 0.1660 | 0.1480 | 0.1717 | 0.1452 | 0.1198 | 1.035 | **1.167** | 0.862 | 0.373 | **5.654** | 4.567 | 1.796 | 1.594 | 63.0 |
| **`data.csv`** | 30.5 MB | **0.1794** | 0.4458 | 0.2863 | 0.2651 | 0.2812 | 0.2606 | 0.2257 | 0.671 | **0.724** | 0.595 | 0.194 | 2.357 | **3.856** | 1.417 | 1.177 | 71.0 |
| **`data.clf`** | 31.2 MB | **0.0980** | 0.2843 | 0.1759 | 0.1504 | 0.1783 | 0.1454 | 0.1225 | 0.608 | **0.916** | 0.720 | 0.332 | 2.525 | **3.686** | 1.654 | 1.391 | 66.1 |
| **All-Dataset Geomean (3)** | — | **0.1172** | 0.3324 | 0.2030 | 0.1807 | 0.2049 | 0.1765 | 0.1491 | 0.750 | **0.918** | 0.718 | 0.288 | 3.228 | **4.019** | 1.614 | 1.377 | 66.6 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **0.0914** | 0.2180 | 0.1452 | 1.035 | **1.374** | 1.106 | 5.654 | **7.617** | 2.253 | **26.4** | 14.8 | 10.9 | 22.3 |
| **`data.csv`** | **0.1794** | 0.3759 | 0.2606 | 0.671 | **0.826** | 0.725 | 2.357 | **4.087** | 1.781 | 7.0 | 4.9 | 4.4 | **7.1** |
| **`data.clf`** | **0.0980** | 0.2177 | 0.1454 | 0.608 | **0.991** | 0.862 | 2.525 | **5.781** | 1.990 | 12.6 | 11.0 | 9.0 | **16.5** |
| **All-Dataset Geomean (3)** | **0.1172** | 0.2613 | 0.1765 | 0.750 | **1.040** | 0.884 | 3.228 | **5.646** | 1.999 | 13.3 | 9.3 | 7.5 | **13.7** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **26.4** | 8.0 | 7.5 | 4.5 | **4.00** | 2.64 | 1.51 |
| **`data.csv`** | **7.0** | 3.7 | 3.2 | 1.7 | **2.07** | 1.84 | 0.93 |
| **`data.clf`** | **12.6** | 6.5 | 6.2 | 3.8 | **2.50** | 2.28 | 1.37 |
| **All-Dataset Geomean (3)** | **13.3** | 5.8 | 5.3 | 3.1 | **2.74** | 2.23 | 1.25 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 4.187 | **3.022** | 13.090 | **0.906** | 0.964 | 2.981 | **13.6** | 4.7 | 0.7 |
| **`data.csv`** | 6.558 | **5.172** | 23.256 | 1.877 | **1.125** | 3.917 | **2.0** | 1.6 | 0.2 |
| **`data.clf`** | 6.957 | **3.801** | 14.001 | 1.648 | **1.158** | 3.327 | **3.7** | 3.1 | 0.6 |
