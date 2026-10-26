## VXG v4 Benchmark, Composite Index & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.056 GB/s** | 0.935 GB/s | 0.693 GB/s | **5.703 GB/s** | 4.575 GB/s | 1.609 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.250 GB/s** | 0.997 GB/s | 0.725 GB/s | **5.647 GB/s** | 4.016 GB/s | 1.476 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **0.932 GB/s** | 0.927 GB/s | 0.721 GB/s | **6.053 GB/s** | 4.555 GB/s | 1.635 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.130 GB/s** | 1.118 GB/s | 0.793 GB/s | **7.894 GB/s** | 5.418 GB/s | 2.180 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.742 GB/s** | 0.812 GB/s | 0.519 GB/s | **3.567 GB/s** | 3.143 GB/s | 1.251 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.078 GB/s** | 1.033 GB/s | 0.553 GB/s | **4.630 GB/s** | 3.330 GB/s | 1.408 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.706 GB/s** | 0.842 GB/s | 0.470 GB/s | **2.543 GB/s** | 2.725 GB/s | 1.207 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2714** | 0.3786 | 0.2720 | **0.307 GB/s** | 0.411 GB/s | 0.290 GB/s | **1.334 GB/s** | 2.836 GB/s | 0.835 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **5.970 GB/s** | 3.466 GB/s | 1.998 GB/s | **8.251 GB/s** | 11.579 GB/s | 2.991 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **5.295 GB/s** | 2.861 GB/s | 2.739 GB/s | **7.250 GB/s** | 6.871 GB/s | 4.908 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.225 GB/s** | 1.108 GB/s | 0.757 GB/s | **4.691 GB/s** | 4.431 GB/s | 1.715 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **60.3** | 10.4 | 12.6 | **5.81x** | **4.80x** | **5.29** | 2.78 | 2.37 |
| **`events.tsv`** | **79.6** | 9.0 | 14.0 | **8.84x** | **5.70x** | **5.96** | 2.62 | 2.44 |
| **`app-logs.logfmt`** | **67.3** | 10.2 | 15.6 | **6.57x** | **4.31x** | **5.43** | 2.76 | 2.57 |
| **`json-records.ndjson`** | **131.8** | 16.4 | 24.2 | **8.02x** | **5.44x** | **7.33** | 3.43 | 3.17 |
| **`field-order-optional-escaping.ndjson`** | **36.6** | 7.5 | 7.7 | **4.88x** | **4.74x** | **3.90** | 2.29 | 1.84 |
| **`event-mix-identifiers.ndjson`** | **90.0** | 12.0 | 11.3 | **7.50x** | **7.94x** | **5.86** | 2.81 | 2.15 |
| **`bursty-variable-records.ndjson`** | **27.9** | 7.3 | 6.3 | **3.80x** | **4.43x** | **3.34** | 2.23 | 1.68 |
| **`source-and-docs.txt`** | **2.4** | 2.9 | 1.8 | **0.83x** | **1.30x** | **1.15** | 1.45 | 0.96 |
| **`ui-render-32bpp.bmp`** | **1466.7** | 451.6 | 268.2 | **3.25x** | **5.47x** | **21.75** | 14.20 | 8.69 |
| **`rgb-frame-24bpp.bmp`** | **4992.3** | 195.3 | 307.2 | **25.57x** | **16.25x** | **31.39** | 9.53 | 10.40 |
| **Geometric Mean** | **96.1** | 17.4 | 18.8 | **5.51x** | **5.12x** | **6.13** | 3.38 | 2.78 |
