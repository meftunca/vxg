## VXG v4 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row (lowest block ratio, highest encode `GB/s`, highest decode `GB/s`).

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 56.5 MB | **0.0914** | 0.2898 | 0.1660 | 0.1480 | 0.1717 | 0.1452 | 0.1198 | 0.934 | **1.052** | 0.823 | 0.331 | **5.509** | 4.294 | 1.668 | 1.353 | 60.8 |
| **`data.csv`** | 30.5 MB | **0.1796** | 0.4458 | 0.2863 | 0.2651 | 0.2812 | 0.2606 | 0.2257 | 0.649 | **0.676** | 0.590 | 0.190 | 2.135 | **3.974** | 1.393 | 1.103 | 74.8 |
| **`data.clf`** | 31.2 MB | **0.0980** | 0.2843 | 0.1759 | 0.1504 | 0.1783 | 0.1454 | 0.1225 | 0.592 | **0.877** | 0.697 | 0.345 | 2.447 | **3.607** | 1.596 | 1.463 | 64.3 |
| **All-Dataset Geomean (3)** | — | **0.1172** | 0.3324 | 0.2030 | 0.1807 | 0.2049 | 0.1765 | 0.1491 | 0.711 | **0.854** | 0.697 | 0.279 | 3.064 | **3.948** | 1.548 | 1.297 | 66.3 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **0.0914** | 0.2180 | 0.1452 | 0.934 | **1.173** | 1.054 | 5.509 | **6.816** | 2.193 | **24.8** | 13.0 | 10.5 | 19.5 |
| **`data.csv`** | **0.1796** | 0.3759 | 0.2606 | 0.649 | **0.767** | 0.715 | 2.135 | **4.177** | 1.787 | 6.6 | 4.8 | 4.3 | **6.9** |
| **`data.clf`** | **0.0980** | 0.2177 | 0.1454 | 0.592 | **0.954** | 0.866 | 2.447 | **6.069** | 1.995 | 12.3 | 11.1 | 9.0 | **16.5** |
| **All-Dataset Geomean (3)** | **0.1172** | 0.2613 | 0.1765 | 0.711 | **0.951** | 0.868 | 3.064 | **5.570** | 1.984 | 12.6 | 8.8 | 7.4 | **13.0** |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **24.8** | 7.3 | 7.1 | 3.9 | **3.83** | 2.50 | 1.38 |
| **`data.csv`** | **6.6** | 3.7 | 3.2 | 1.6 | **1.98** | 1.82 | 0.91 |
| **`data.clf`** | **12.3** | 6.3 | 6.0 | 4.0 | **2.45** | 2.23 | 1.41 |
| **All-Dataset Geomean (3)** | **12.6** | 5.5 | 5.1 | 2.9 | **2.65** | 2.16 | 1.21 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 3.604 | **2.509** | 11.230 | **0.799** | 0.911 | 2.663 | **14.6** | 4.9 | 0.7 |
| **`data.csv`** | 6.235 | **5.185** | 22.794 | 1.849 | **1.099** | 3.825 | **1.9** | 1.5 | 0.2 |
| **`data.clf`** | 6.522 | **3.764** | 13.919 | 1.602 | **1.155** | 3.314 | **3.8** | 3.0 | 0.6 |
