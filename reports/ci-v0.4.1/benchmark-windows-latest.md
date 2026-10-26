## VXG v4 Benchmark, Composite Index & Energy Summary (`windows` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **2.183 GB/s** | 1.381 GB/s | 0.896 GB/s | **9.611 GB/s** | 9.539 GB/s | 2.619 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **2.105 GB/s** | 1.212 GB/s | 0.989 GB/s | **10.302 GB/s** | 6.762 GB/s | 2.955 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **2.278 GB/s** | 1.340 GB/s | 0.981 GB/s | **8.906 GB/s** | 8.571 GB/s | 2.843 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.677 GB/s** | 1.490 GB/s | 1.155 GB/s | **14.200 GB/s** | 11.543 GB/s | 3.726 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.521 GB/s** | 0.987 GB/s | 0.748 GB/s | **6.210 GB/s** | 4.672 GB/s | 2.018 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **2.141 GB/s** | 1.268 GB/s | 0.838 GB/s | **9.216 GB/s** | 5.386 GB/s | 2.279 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **1.165 GB/s** | 1.111 GB/s | 0.724 GB/s | **3.282 GB/s** | 3.963 GB/s | 1.710 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2677** | 0.3720 | 0.2667 | **0.575 GB/s** | 0.735 GB/s | 0.514 GB/s | **2.508 GB/s** | 5.108 GB/s | 1.756 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **7.075 GB/s** | 3.121 GB/s | 2.749 GB/s | **14.442 GB/s** | 41.456 GB/s | 18.079 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **6.639 GB/s** | 3.321 GB/s | 2.786 GB/s | **9.212 GB/s** | 4.583 GB/s | 3.950 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1268 | 0.0606 | **2.217 GB/s** | 1.429 GB/s | 1.060 GB/s | **7.753 GB/s** | 7.569 GB/s | 3.096 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **112.5** | 18.2 | 18.2 | **6.18x** | **6.17x** | **8.02** | 4.04 | 3.03 |
| **`events.tsv`** | **139.4** | 12.9 | 23.1 | **10.83x** | **6.04x** | **8.66** | 3.33 | 3.40 |
| **`app-logs.logfmt`** | **127.6** | 16.9 | 24.0 | **7.56x** | **5.31x** | **8.31** | 3.85 | 3.42 |
| **`json-records.ndjson`** | **272.1** | 27.7 | 38.2 | **9.82x** | **7.12x** | **11.88** | 4.86 | 4.30 |
| **`field-order-optional-escaping.ndjson`** | **69.1** | 10.1 | 11.8 | **6.86x** | **5.86x** | **5.97** | 2.79 | 2.44 |
| **`event-mix-identifiers.ndjson`** | **179.0** | 16.9 | 17.7 | **10.58x** | **10.09x** | **9.26** | 3.54 | 2.90 |
| **`bursty-variable-records.ndjson`** | **40.7** | 10.2 | 9.3 | **4.01x** | **4.38x** | **4.30** | 2.77 | 2.18 |
| **`source-and-docs.txt`** | **4.5** | 5.2 | 3.6 | **0.86x** | **1.26x** | **1.75** | 2.16 | 1.50 |
| **`ui-render-32bpp.bmp`** | **2112.5** | 810.9 | 773.4 | **2.60x** | **2.73x** | **27.74** | 20.97 | 17.60 |
| **`rgb-frame-24bpp.bmp`** | **6301.0** | 171.8 | 277.9 | **36.68x** | **22.67x** | **36.66** | 8.75 | 9.73 |
| **Geometric Mean** | **166.5** | 25.9 | 29.9 | **6.42x** | **5.56x** | **8.84** | 4.40 | 3.78 |
