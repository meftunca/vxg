## VXG v4 Benchmark, Composite Index & Energy Summary (`linux` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.706 GB/s** | 1.437 GB/s | 0.737 GB/s | **7.513 GB/s** | 7.170 GB/s | 2.395 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.460 GB/s** | 1.281 GB/s | 1.034 GB/s | **6.709 GB/s** | 4.957 GB/s | 3.155 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.885 GB/s** | 1.871 GB/s | 1.219 GB/s | **8.434 GB/s** | 7.626 GB/s | 3.201 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.474 GB/s** | 2.484 GB/s | 1.650 GB/s | **10.906 GB/s** | 9.651 GB/s | 4.085 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.377 GB/s** | 1.286 GB/s | 0.856 GB/s | **5.483 GB/s** | 4.587 GB/s | 2.113 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.824 GB/s** | 1.836 GB/s | 1.145 GB/s | **7.681 GB/s** | 5.197 GB/s | 2.461 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **1.032 GB/s** | 1.300 GB/s | 0.801 GB/s | **3.772 GB/s** | 3.918 GB/s | 1.792 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2712** | 0.3785 | 0.2722 | **0.513 GB/s** | 0.661 GB/s | 0.416 GB/s | **2.205 GB/s** | 3.970 GB/s | 1.772 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **8.370 GB/s** | 9.805 GB/s | 5.723 GB/s | **14.818 GB/s** | 21.433 GB/s | 4.797 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **7.311 GB/s** | 7.260 GB/s | 5.124 GB/s | **9.078 GB/s** | 2.881 GB/s | 2.622 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.984 GB/s** | 2.038 GB/s | 1.308 GB/s | **6.822 GB/s** | 5.994 GB/s | 2.699 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **87.9** | 16.1 | 15.8 | **5.46x** | **5.56x** | **6.80** | 3.72 | 2.76 |
| **`events.tsv`** | **93.7** | 11.3 | 24.4 | **8.27x** | **3.84x** | **6.64** | 3.06 | 3.53 |
| **`app-logs.logfmt`** | **113.0** | 18.8 | 28.4 | **6.00x** | **3.98x** | **7.66** | 4.14 | 3.83 |
| **`json-records.ndjson`** | **229.3** | 32.7 | 47.9 | **7.01x** | **4.79x** | **10.60** | 5.43 | 4.99 |
| **`field-order-optional-escaping.ndjson`** | **61.8** | 11.4 | 12.9 | **5.42x** | **4.79x** | **5.54** | 3.02 | 2.59 |
| **`event-mix-identifiers.ndjson`** | **150.8** | 20.0 | 21.5 | **7.54x** | **7.00x** | **8.26** | 3.95 | 3.31 |
| **`bursty-variable-records.ndjson`** | **41.1** | 10.9 | 10.0 | **3.76x** | **4.10x** | **4.33** | 2.91 | 2.29 |
| **`source-and-docs.txt`** | **3.9** | 4.3 | 3.2 | **0.92x** | **1.24x** | **1.61** | 1.91 | 1.39 |
| **`ui-render-32bpp.bmp`** | **2327.3** | 1033.5 | 574.8 | **2.25x** | **4.05x** | **29.59** | 24.65 | 14.44 |
| **`rgb-frame-24bpp.bmp`** | **6564.0** | 201.4 | 307.1 | **32.59x** | **21.37x** | **37.67** | 9.73 | 10.40 |
| **Geometric Mean** | **147.5** | 27.5 | 31.0 | **5.36x** | **4.76x** | **8.16** | 4.58 | 3.88 |
