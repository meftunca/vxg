## VXG v4 Benchmark, Composite Index & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.296 GB/s** | 0.982 GB/s | 0.730 GB/s | **5.954 GB/s** | 5.514 GB/s | 1.739 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.187 GB/s** | 0.952 GB/s | 0.751 GB/s | **7.044 GB/s** | 4.026 GB/s | 1.715 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.614 GB/s** | 1.205 GB/s | 0.770 GB/s | **8.613 GB/s** | 5.748 GB/s | 2.104 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.618 GB/s** | 1.437 GB/s | 0.917 GB/s | **9.242 GB/s** | 6.288 GB/s | 2.811 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.175 GB/s** | 0.873 GB/s | 0.616 GB/s | **4.980 GB/s** | 3.274 GB/s | 1.348 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.281 GB/s** | 1.260 GB/s | 0.770 GB/s | **5.830 GB/s** | 4.055 GB/s | 1.597 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.940 GB/s** | 0.985 GB/s | 0.671 GB/s | **3.503 GB/s** | 3.411 GB/s | 1.467 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2702** | 0.3778 | 0.2714 | **0.374 GB/s** | 0.496 GB/s | 0.390 GB/s | **1.696 GB/s** | 3.200 GB/s | 1.084 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.423 GB/s** | 3.846 GB/s | 2.456 GB/s | **9.170 GB/s** | 14.402 GB/s | 2.719 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **6.025 GB/s** | 4.357 GB/s | 3.669 GB/s | **9.611 GB/s** | 8.715 GB/s | 5.913 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.554 GB/s** | 1.307 GB/s | 0.920 GB/s | **5.896 GB/s** | 5.205 GB/s | 1.990 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **68.2** | 11.7 | 13.4 | **5.84x** | **5.09x** | **5.74** | 3.01 | 2.47 |
| **`events.tsv`** | **86.6** | 8.8 | 15.3 | **9.83x** | **5.65x** | **6.30** | 2.58 | 2.59 |
| **`app-logs.logfmt`** | **105.6** | 13.1 | 18.3 | **8.05x** | **5.77x** | **7.33** | 3.26 | 2.86 |
| **`json-records.ndjson`** | **170.6** | 20.1 | 29.6 | **8.50x** | **5.77x** | **8.71** | 3.92 | 3.62 |
| **`field-order-optional-escaping.ndjson`** | **54.4** | 7.9 | 8.7 | **6.86x** | **6.22x** | **5.09** | 2.38 | 2.00 |
| **`event-mix-identifiers.ndjson`** | **110.1** | 14.6 | 14.2 | **7.52x** | **7.73x** | **6.70** | 3.21 | 2.51 |
| **`bursty-variable-records.ndjson`** | **37.8** | 8.9 | 8.3 | **4.26x** | **4.56x** | **4.09** | 2.53 | 2.02 |
| **`source-and-docs.txt`** | **2.9** | 3.3 | 2.4 | **0.88x** | **1.23x** | **1.33** | 1.61 | 1.16 |
| **`ui-render-32bpp.bmp`** | **1603.8** | 530.6 | 283.5 | **3.02x** | **5.66x** | **23.09** | 15.81 | 9.01 |
| **`rgb-frame-24bpp.bmp`** | **6131.5** | 271.4 | 390.2 | **22.59x** | **15.71x** | **36.00** | 11.87 | 12.20 |
| **Geometric Mean** | **121.4** | 20.5 | 22.3 | **5.91x** | **5.45x** | **7.16** | 3.77 | 3.11 |
