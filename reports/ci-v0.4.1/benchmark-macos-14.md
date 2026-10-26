## VXG v4 Benchmark, Composite Index & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.346 GB/s** | 1.336 GB/s | 0.766 GB/s | **7.801 GB/s** | 6.097 GB/s | 1.534 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.182 GB/s** | 0.803 GB/s | 0.881 GB/s | **8.227 GB/s** | 4.293 GB/s | 1.882 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.212 GB/s** | 1.088 GB/s | 0.867 GB/s | **9.112 GB/s** | 5.816 GB/s | 1.947 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.688 GB/s** | 1.545 GB/s | 1.001 GB/s | **11.149 GB/s** | 6.748 GB/s | 2.823 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.159 GB/s** | 0.867 GB/s | 0.660 GB/s | **4.873 GB/s** | 3.470 GB/s | 1.581 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.445 GB/s** | 1.150 GB/s | 0.782 GB/s | **7.527 GB/s** | 4.259 GB/s | 1.764 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.937 GB/s** | 0.878 GB/s | 0.578 GB/s | **3.334 GB/s** | 3.487 GB/s | 1.523 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2710** | 0.3782 | 0.2720 | **0.476 GB/s** | 0.547 GB/s | 0.428 GB/s | **1.556 GB/s** | 2.958 GB/s | 1.298 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.304 GB/s** | 4.185 GB/s | 2.786 GB/s | **10.630 GB/s** | 13.364 GB/s | 3.744 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **5.603 GB/s** | 3.583 GB/s | 2.952 GB/s | **10.309 GB/s** | 7.217 GB/s | 6.355 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.561 GB/s** | 1.292 GB/s | 0.953 GB/s | **6.506 GB/s** | 5.223 GB/s | 2.147 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **79.6** | 14.3 | 12.9 | **5.56x** | **6.17x** | **6.36** | 3.44 | 2.41 |
| **`events.tsv`** | **93.4** | 8.4 | 17.4 | **11.18x** | **5.37x** | **6.63** | 2.49 | 2.82 |
| **`app-logs.logfmt`** | **94.1** | 12.5 | 18.7 | **7.51x** | **5.04x** | **6.79** | 3.16 | 2.90 |
| **`json-records.ndjson`** | **191.5** | 21.6 | 31.0 | **8.88x** | **6.18x** | **9.40** | 4.11 | 3.73 |
| **`field-order-optional-escaping.ndjson`** | **53.4** | 8.1 | 9.8 | **6.57x** | **5.45x** | **5.03** | 2.42 | 2.15 |
| **`event-mix-identifiers.ndjson`** | **132.8** | 14.3 | 15.1 | **9.27x** | **8.81x** | **7.59** | 3.17 | 2.61 |
| **`bursty-variable-records.ndjson`** | **36.8** | 8.5 | 7.8 | **4.34x** | **4.69x** | **4.02** | 2.46 | 1.95 |
| **`source-and-docs.txt`** | **3.2** | 3.4 | 2.7 | **0.94x** | **1.16x** | **1.40** | 1.62 | 1.27 |
| **`ui-render-32bpp.bmp`** | **1710.7** | 533.2 | 354.3 | **3.21x** | **4.83x** | **24.10** | 15.86 | 10.46 |
| **`rgb-frame-24bpp.bmp`** | **6124.0** | 223.9 | 362.9 | **27.35x** | **16.88x** | **35.97** | 10.44 | 11.63 |
| **Geometric Mean** | **127.8** | 20.5 | 23.6 | **6.25x** | **5.42x** | **7.41** | 3.76 | 3.23 |
