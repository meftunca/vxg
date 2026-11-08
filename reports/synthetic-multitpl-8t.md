## VXG v4/v5 Comparative Benchmark Report (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

- **Block-independent codecs (`65536`-byte blocks, random access = YES):** `VXG v4/v5`, `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`), and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-dict-l3`, `zstd-c-l9`, `zstd-dict-l9`).
- **Dictionary accounting:** `zstd-dict-*` trains up to a `32 KiB` dictionary via `ZDICT_trainFromBuffer` (charging `dict_bytes` in compressed size); steady-state `ZSTD_CDict` / `ZSTD_DDict` throughput is reported in `Enc` / `Dec`, with one-time `ZDICT` training time reported separately in `Dict Train (ms)`. `VXG v5` embeds its self-trained `<= 32 KiB` prefix dictionary inside the `.vxg` frame itself, and `VXG Enc` / `Dec` include full dictionary training and parsing time.
- **Whole-file reference (`random access = NO`):** `zstd-whole-l19-long` (`zstd -19 --long`, `128 MiB` window, single stream) quantifies the compression-ratio difference of removing `65536`-byte block boundaries.
- Bold cells mark the best block-codec metric in each row; codecs storing near-raw bytes (`ratio >= 0.90`, `< 10%` size reduction) are excluded from speed/trade-off/energy bolding.

### 1. Individual Codec Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG Ratio | LZ4 (a1) | Zstd-1 | Zstd-9 | Zstd+Dict-3 | Zstd+Dict-9 | Zstd-19 `--long` *(whole)* | VXG Enc | LZ4 Enc | Zstd-1 Enc | Zstd+D3 Enc | VXG Dec | LZ4 Dec | Zstd-1 Dec | Zstd+D3 Dec | Dict Train (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 56.5 MB | **0.0914** | 0.2898 | 0.1660 | 0.1480 | 0.1717 | 0.1452 | 0.1198 | **6.197** | 5.980 | 5.365 | 2.117 | **40.286** | 25.444 | 11.007 | 9.783 | 63.9 |
| **`data.csv`** | 30.5 MB | **0.1794** | 0.4458 | 0.2863 | 0.2651 | 0.2812 | 0.2606 | 0.2257 | 3.778 | **4.399** | 3.774 | 1.344 | 14.732 | **21.671** | 8.887 | 7.625 | 78.2 |
| **`data.clf`** | 31.2 MB | **0.0980** | 0.2843 | 0.1759 | 0.1504 | 0.1783 | 0.1454 | 0.1225 | 3.612 | **5.764** | 3.781 | 1.916 | **17.915** | 17.246 | 9.515 | 7.284 | 62.3 |
| **All-Dataset Geomean (3)** | — | **0.1172** | 0.3324 | 0.2030 | 0.1807 | 0.2049 | 0.1765 | 0.1491 | 4.389 | **5.333** | 4.246 | 1.760 | **21.989** | 21.186 | 9.764 | 8.160 | 67.8 |

### 2. Parameter-Sweep Upper-Bound Envelopes (`min_ratio` + `max_speed` across Independent Blocks)

Each sweep envelope combines the best block ratio achieved by any tested level in that family with the fastest encode and decode throughput achieved by any preset in that family (`LZ4 Sweep` = `[a1, a4, HC-L9]`, `Zstd+Dict Sweep` = `[-3, 1, 3, 9, dict-3, dict-9]`, `All-Block Sweep` = union of both). Bold cells mark the actual per-axis winner (`Ratio`, `Enc`, `Dec`).

| Dataset | VXG Ratio | LZ4 Sweep Ratio | Zstd+Dict Sweep Ratio | VXG Enc | LZ4 Sweep Enc | Zstd+Dict Sweep Enc | VXG Dec | LZ4 Sweep Dec | Zstd+Dict Sweep Dec | VXG `VPI` | LZ4 Sweep `VPI` | Zstd+Dict Sweep `VPI` | All-Block Sweep `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **0.0914** | 0.2180 | 0.1452 | 6.197 | **8.781** | 5.622 | **40.286** | 37.015 | 12.878 | **172.8** | 82.7 | 58.6 | 124.2 |
| **`data.csv`** | **0.1794** | 0.3759 | 0.2606 | 3.778 | **5.002** | 3.914 | 14.732 | **21.671** | 10.757 | **41.6** | 27.7 | 24.9 | 40.0 |
| **`data.clf`** | **0.0980** | 0.2177 | 0.1454 | 3.612 | **5.764** | 4.893 | 17.915 | **30.266** | 11.137 | 82.1 | 60.7 | 50.8 | **90.8** |
| **All-Dataset Geomean (3)** | **0.1172** | 0.2613 | 0.1765 | 4.389 | **6.326** | 4.757 | 21.989 | **28.956** | 11.555 | **83.8** | 51.8 | 42.0 | 76.7 |

### 3. Composite Trade-Off Indices (`VPI` & `BCI`)

> Note: `VPI = sqrt(Encode_GBps * Decode_GBps) / Ratio` and `BCI = ((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` summarize the joint space-speed trade-off; a higher `VPI` or `BCI` does not imply faster raw throughput than `LZ4` on rows where `LZ4` leads in `Enc` or `Dec` above.

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | Zstd+Dict-3 `VPI` | VXG `BCI` | LZ4 (a1) `BCI` | Zstd+Dict-3 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **172.8** | 42.6 | 46.3 | 26.5 | **13.98** | 8.07 | 4.94 |
| **`data.csv`** | **41.6** | 21.9 | 20.2 | 11.4 | **6.77** | 5.98 | 3.32 |
| **`data.clf`** | **82.1** | 35.1 | 34.1 | 21.0 | **8.71** | 7.04 | 4.28 |
| **All-Dataset Geomean (3)** | **83.8** | 32.0 | 31.7 | 18.5 | **9.37** | 6.98 | 4.12 |

### 4. Per-Process CPU Energy (`J/GB`, Lower is Better) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd+D3 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd+D3 Dec `J/GB` | VXG `GCI` | LZ4 `GCI` | Zstd+D3 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 3.524 | **2.415** | 10.886 | **0.831** | 0.864 | 2.713 | **101.0** | 29.5 | 4.9 |
| **`data.csv`** | 5.808 | **4.620** | 20.854 | 1.612 | **1.009** | 3.623 | **13.6** | 10.1 | 1.3 |
| **`data.clf`** | 5.905 | **3.488** | 11.966 | 1.433 | **0.942** | 2.933 | **28.2** | 19.4 | 3.5 |
