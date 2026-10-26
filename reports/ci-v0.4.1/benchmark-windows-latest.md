## VXG v4 Benchmark, Composite Index & Energy Summary (`windows` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.491 GB/s** | 1.058 GB/s | 0.643 GB/s | **5.935 GB/s** | 6.816 GB/s | 1.859 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.471 GB/s** | 0.928 GB/s | 0.716 GB/s | **6.097 GB/s** | 5.323 GB/s | 2.142 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.624 GB/s** | 1.068 GB/s | 0.724 GB/s | **6.832 GB/s** | 6.944 GB/s | 2.068 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.090 GB/s** | 1.472 GB/s | 1.084 GB/s | **9.628 GB/s** | 8.805 GB/s | 2.762 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.117 GB/s** | 0.976 GB/s | 0.615 GB/s | **4.245 GB/s** | 4.287 GB/s | 1.528 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.590 GB/s** | 1.218 GB/s | 0.814 GB/s | **6.565 GB/s** | 4.941 GB/s | 1.861 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.943 GB/s** | 0.981 GB/s | 0.578 GB/s | **3.301 GB/s** | 3.720 GB/s | 1.389 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2673** | 0.3716 | 0.2667 | **0.462 GB/s** | 0.610 GB/s | 0.378 GB/s | **1.646 GB/s** | 3.817 GB/s | 1.346 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **5.137 GB/s** | 2.657 GB/s | 2.599 GB/s | **11.264 GB/s** | 36.904 GB/s | 15.647 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **4.715 GB/s** | 2.903 GB/s | 2.585 GB/s | **7.095 GB/s** | 2.909 GB/s | 2.605 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1268 | 0.0606 | **1.639 GB/s** | 1.238 GB/s | 0.876 GB/s | **5.571 GB/s** | 6.103 GB/s | 2.340 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **73.0** | 13.5 | 13.0 | **5.42x** | **5.62x** | **6.01** | 3.31 | 2.42 |
| **`events.tsv`** | **89.7** | 10.0 | 16.7 | **8.97x** | **5.36x** | **6.45** | 2.81 | 2.75 |
| **`app-logs.logfmt`** | **94.4** | 13.6 | 17.6 | **6.95x** | **5.36x** | **6.80** | 3.33 | 2.78 |
| **`json-records.ndjson`** | **198.0** | 24.0 | 31.9 | **8.23x** | **6.21x** | **9.61** | 4.42 | 3.81 |
| **`field-order-optional-escaping.ndjson`** | **49.0** | 9.6 | 9.3 | **5.10x** | **5.27x** | **4.74** | 2.70 | 2.08 |
| **`event-mix-identifiers.ndjson`** | **130.1** | 15.9 | 15.8 | **8.19x** | **8.24x** | **7.49** | 3.39 | 2.69 |
| **`bursty-variable-records.ndjson`** | **36.7** | 9.2 | 7.5 | **3.97x** | **4.91x** | **4.02** | 2.60 | 1.89 |
| **`source-and-docs.txt`** | **3.3** | 4.1 | 2.7 | **0.79x** | **1.22x** | **1.42** | 1.84 | 1.24 |
| **`ui-render-32bpp.bmp`** | **1589.7** | 705.9 | 699.5 | **2.25x** | **2.27x** | **22.95** | 19.12 | 16.46 |
| **`rgb-frame-24bpp.bmp`** | **4660.1** | 128.0 | 217.4 | **36.42x** | **21.43x** | **29.98** | 7.19 | 8.26 |
| **Geometric Mean** | **121.3** | 21.7 | 23.6 | **5.60x** | **5.13x** | **7.16** | 3.91 | 3.23 |
