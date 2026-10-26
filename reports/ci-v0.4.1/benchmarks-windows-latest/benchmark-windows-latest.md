## VXG v4 Benchmark, Composite Index & Energy Summary (`windows` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.013 GB/s** | 0.912 GB/s | 0.521 GB/s | **4.504 GB/s** | 5.259 GB/s | 1.357 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.196 GB/s** | 0.779 GB/s | 0.570 GB/s | **2.953 GB/s** | 3.963 GB/s | 1.597 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.107 GB/s** | 0.879 GB/s | 0.607 GB/s | **4.656 GB/s** | 5.323 GB/s | 1.530 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.562 GB/s** | 1.156 GB/s | 0.756 GB/s | **5.011 GB/s** | 5.957 GB/s | 2.084 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.873 GB/s** | 0.788 GB/s | 0.487 GB/s | **2.864 GB/s** | 3.487 GB/s | 1.174 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.158 GB/s** | 1.041 GB/s | 0.620 GB/s | **5.032 GB/s** | 4.097 GB/s | 1.447 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.766 GB/s** | 0.831 GB/s | 0.454 GB/s | **2.383 GB/s** | 3.138 GB/s | 1.076 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2665** | 0.3711 | 0.2661 | **0.344 GB/s** | 0.469 GB/s | 0.309 GB/s | **1.355 GB/s** | 2.784 GB/s | 0.966 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **5.477 GB/s** | 2.117 GB/s | 1.911 GB/s | **9.525 GB/s** | 15.409 GB/s | 10.383 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **1.443 GB/s** | 2.307 GB/s | 1.954 GB/s | **4.547 GB/s** | 1.961 GB/s | 1.758 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1268 | 0.0605 | **1.160 GB/s** | 1.012 GB/s | 0.681 GB/s | **3.799 GB/s** | 4.362 GB/s | 1.717 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **52.4** | 11.0 | 10.0 | **4.77x** | **5.25x** | **4.82** | 2.89 | 2.03 |
| **`events.tsv`** | **56.3** | 7.9 | 12.9 | **7.12x** | **4.37x** | **4.73** | 2.40 | 2.31 |
| **`app-logs.logfmt`** | **64.3** | 10.8 | 13.9 | **5.97x** | **4.64x** | **5.27** | 2.86 | 2.37 |
| **`json-records.ndjson`** | **123.5** | 17.5 | 23.1 | **7.04x** | **5.34x** | **7.02** | 3.58 | 3.07 |
| **`field-order-optional-escaping.ndjson`** | **35.5** | 7.8 | 7.3 | **4.57x** | **4.90x** | **3.83** | 2.34 | 1.76 |
| **`event-mix-identifiers.ndjson`** | **97.2** | 13.4 | 12.2 | **7.27x** | **8.00x** | **6.17** | 3.02 | 2.26 |
| **`bursty-variable-records.ndjson`** | **28.1** | 7.8 | 5.8 | **3.60x** | **4.82x** | **3.36** | 2.33 | 1.60 |
| **`source-and-docs.txt`** | **2.6** | 3.1 | 2.1 | **0.83x** | **1.25x** | **1.20** | 1.52 | 1.04 |
| **`ui-render-32bpp.bmp`** | **1509.3** | 407.2 | 488.7 | **3.71x** | **3.09x** | **22.17** | 13.25 | 12.96 |
| **`rgb-frame-24bpp.bmp`** | **2063.9** | 93.7 | 155.3 | **22.04x** | **13.29x** | **17.42** | 5.84 | 6.60 |
| **Geometric Mean** | **84.3** | 16.6 | 17.9 | **5.09x** | **4.72x** | **5.61** | 3.27 | 2.68 |
