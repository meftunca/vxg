## VXG v4 Benchmark, Composite Index & Energy Summary (`linux` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.169 GB/s** | 1.019 GB/s | 0.582 GB/s | **3.458 GB/s** | 5.541 GB/s | 1.858 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.101 GB/s** | 1.213 GB/s | 0.799 GB/s | **2.699 GB/s** | 4.096 GB/s | 2.153 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.489 GB/s** | 1.481 GB/s | 0.817 GB/s | **3.905 GB/s** | 5.551 GB/s | 2.063 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.745 GB/s** | 1.952 GB/s | 1.091 GB/s | **3.599 GB/s** | 6.901 GB/s | 2.737 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.935 GB/s** | 1.112 GB/s | 0.622 GB/s | **2.851 GB/s** | 3.549 GB/s | 1.575 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.390 GB/s** | 1.554 GB/s | 0.822 GB/s | **4.107 GB/s** | 4.179 GB/s | 1.861 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.837 GB/s** | 1.126 GB/s | 0.578 GB/s | **2.067 GB/s** | 3.158 GB/s | 1.376 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2714** | 0.3786 | 0.2720 | **0.366 GB/s** | 0.551 GB/s | 0.306 GB/s | **1.369 GB/s** | 2.929 GB/s | 1.216 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.338 GB/s** | 6.793 GB/s | 4.102 GB/s | **9.654 GB/s** | 11.186 GB/s | 11.375 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **4.601 GB/s** | 6.506 GB/s | 4.147 GB/s | **6.818 GB/s** | 1.993 GB/s | 1.830 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.442 GB/s** | 1.671 GB/s | 0.956 GB/s | **3.513 GB/s** | 4.390 GB/s | 2.169 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **49.4** | 11.9 | 12.4 | **4.14x** | **3.99x** | **4.63** | 3.05 | 2.34 |
| **`events.tsv`** | **51.6** | 10.0 | 17.7 | **5.15x** | **2.92x** | **4.46** | 2.82 | 2.85 |
| **`app-logs.logfmt`** | **68.3** | 14.3 | 18.7 | **4.78x** | **3.66x** | **5.48** | 3.45 | 2.89 |
| **`json-records.ndjson`** | **110.6** | 24.5 | 31.8 | **4.51x** | **3.47x** | **6.52** | 4.48 | 3.80 |
| **`field-order-optional-escaping.ndjson`** | **36.7** | 9.3 | 9.5 | **3.94x** | **3.87x** | **3.91** | 2.65 | 2.11 |
| **`event-mix-identifiers.ndjson`** | **96.3** | 16.5 | 15.9 | **5.83x** | **6.06x** | **6.13** | 3.48 | 2.70 |
| **`bursty-variable-records.ndjson`** | **27.4** | 9.1 | 7.5 | **3.00x** | **3.67x** | **3.30** | 2.58 | 1.88 |
| **`source-and-docs.txt`** | **2.6** | 3.4 | 2.2 | **0.78x** | **1.16x** | **1.23** | 1.62 | 1.11 |
| **`ui-render-32bpp.bmp`** | **1634.7** | 621.5 | 749.3 | **2.63x** | **2.18x** | **23.38** | 17.56 | 17.23 |
| **`rgb-frame-24bpp.bmp`** | **4512.7** | 158.6 | 230.8 | **28.46x** | **19.55x** | **29.35** | 8.30 | 8.60 |
| **Geometric Mean** | **90.2** | 21.3 | 23.7 | **4.23x** | **3.80x** | **5.88** | 3.87 | 3.24 |
