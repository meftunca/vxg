## VXG v4 Benchmark, Composite Index & Energy Summary (`windows` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **2.010 GB/s** | 1.364 GB/s | 0.872 GB/s | **8.915 GB/s** | 8.835 GB/s | 2.481 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.893 GB/s** | 1.115 GB/s | 0.945 GB/s | **8.583 GB/s** | 6.385 GB/s | 2.917 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.758 GB/s** | 1.312 GB/s | 1.045 GB/s | **9.922 GB/s** | 8.689 GB/s | 2.667 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.748 GB/s** | 1.776 GB/s | 1.261 GB/s | **14.409 GB/s** | 11.171 GB/s | 3.658 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.519 GB/s** | 1.132 GB/s | 0.760 GB/s | **6.405 GB/s** | 4.697 GB/s | 1.946 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **2.018 GB/s** | 1.513 GB/s | 0.995 GB/s | **9.347 GB/s** | 4.840 GB/s | 2.339 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **1.176 GB/s** | 1.178 GB/s | 0.733 GB/s | **4.472 GB/s** | 3.965 GB/s | 1.670 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2678** | 0.3721 | 0.2672 | **0.576 GB/s** | 0.735 GB/s | 0.522 GB/s | **2.962 GB/s** | 5.253 GB/s | 1.805 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **10.452 GB/s** | 3.523 GB/s | 3.133 GB/s | **17.003 GB/s** | 47.484 GB/s | 18.978 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **10.137 GB/s** | 3.554 GB/s | 2.892 GB/s | **11.329 GB/s** | 4.692 GB/s | 4.180 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1268 | 0.0606 | **2.295 GB/s** | 1.520 GB/s | 1.110 GB/s | **8.362 GB/s** | 7.519 GB/s | 3.080 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **103.9** | 17.4 | 17.5 | **5.97x** | **5.94x** | **7.61** | 3.92 | 2.95 |
| **`events.tsv`** | **120.7** | 12.0 | 22.4 | **10.06x** | **5.38x** | **7.86** | 3.18 | 3.34 |
| **`app-logs.logfmt`** | **118.3** | 16.8 | 24.0 | **7.03x** | **4.93x** | **7.91** | 3.84 | 3.42 |
| **`json-records.ndjson`** | **277.7** | 29.8 | 39.6 | **9.33x** | **7.01x** | **12.04** | 5.10 | 4.40 |
| **`field-order-optional-escaping.ndjson`** | **70.1** | 10.8 | 11.7 | **6.48x** | **6.01x** | **6.03** | 2.92 | 2.42 |
| **`event-mix-identifiers.ndjson`** | **174.9** | 17.5 | 19.6 | **9.98x** | **8.93x** | **9.12** | 3.62 | 3.10 |
| **`bursty-variable-records.ndjson`** | **47.8** | 10.5 | 9.3 | **4.56x** | **5.16x** | **4.78** | 2.83 | 2.17 |
| **`source-and-docs.txt`** | **4.9** | 5.3 | 3.6 | **0.92x** | **1.34x** | **1.85** | 2.18 | 1.52 |
| **`ui-render-32bpp.bmp`** | **2785.9** | 922.1 | 845.8 | **3.02x** | **3.29x** | **33.36** | 22.85 | 18.68 |
| **`rgb-frame-24bpp.bmp`** | **8634.5** | 179.8 | 291.3 | **48.02x** | **29.64x** | **45.23** | 9.02 | 10.04 |
| **Geometric Mean** | **175.9** | 26.7 | 30.5 | **6.60x** | **5.76x** | **9.17** | 4.48 | 3.84 |
