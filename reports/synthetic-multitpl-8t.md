## VXG v4 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row (lowest block ratio, highest encode `GB/s`, highest decode `GB/s`).

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 56.5 MB | **0.0914** | 0.2898 | 0.1660 | 0.1480 | 0.1717 | 0.1452 | 0.1198 | 3.077 | **4.899** | 3.867 | 2.134 | 16.376 | **21.810** | 8.687 | 8.302 | 139.1 |
| **`data.csv`** | 30.5 MB | **0.1796** | 0.4458 | 0.2863 | 0.2651 | 0.2812 | 0.2606 | 0.2257 | **1.947** | 1.690 | 1.212 | 0.569 | 5.131 | **5.998** | 3.014 | 2.296 | 98.5 |
| **`data.clf`** | 31.2 MB | **0.0980** | 0.2843 | 0.1759 | 0.1504 | 0.1783 | 0.1454 | 0.1225 | 1.847 | **2.297** | 1.161 | 0.764 | 5.615 | **6.818** | 2.702 | 2.855 | 64.1 |
| **All-Dataset Geomean (3)** | — | **0.1172** | 0.3324 | 0.2030 | 0.1807 | 0.2049 | 0.1765 | 0.1491 | 2.228 | **2.669** | 1.759 | 0.975 | 7.785 | **9.626** | 4.136 | 3.790 | 95.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **0.0914** | 0.2180 | 0.1452 | 3.077 | **7.726** | 5.264 | 16.376 | **31.805** | 11.122 | 77.6 | 71.9 | 52.7 | **108.0** |
| **`data.csv`** | **0.1796** | 0.3759 | 0.2606 | **1.947** | 1.690 | 1.333 | 5.131 | **6.431** | 3.162 | **17.6** | 8.8 | 7.9 | 12.7 |
| **`data.clf`** | **0.0980** | 0.2177 | 0.1454 | 1.847 | **2.297** | 1.869 | 5.615 | **7.737** | 3.786 | **32.8** | 19.4 | 18.3 | 29.0 |
| **All-Dataset Geomean (3)** | **0.1172** | 0.2613 | 0.1765 | 2.228 | **3.107** | 2.358 | 7.785 | **11.653** | 5.106 | **35.5** | 23.0 | 19.7 | 34.1 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **77.6** | 35.7 | 34.9 | 24.5 | **8.20** | 7.17 | 4.69 |
| **`data.csv`** | **17.6** | 7.1 | 6.7 | 4.1 | **3.82** | 2.83 | 1.67 |
| **`data.clf`** | **32.8** | 13.9 | 10.1 | 8.3 | **4.73** | 3.80 | 2.30 |
| **All-Dataset Geomean (3)** | **35.5** | 15.2 | 13.3 | 9.4 | **5.29** | 4.26 | 2.62 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 2.720 | **2.434** | 10.272 | **0.741** | 0.775 | 2.370 | **54.7** | 26.0 | 5.0 |
| **`data.csv`** | 4.671 | **3.293** | 16.237 | 1.413 | **0.714** | 2.428 | **6.8** | 4.7 | 0.6 |
| **`data.clf`** | 4.790 | **2.752** | 9.579 | 1.171 | **0.711** | 1.997 | **13.9** | 10.0 | 1.9 |
