## VXG v4 Benchmark, Composite Index & Energy Summary (`linux` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.104 GB/s** | 1.026 GB/s | 0.578 GB/s | **3.981 GB/s** | 5.527 GB/s | 1.831 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.311 GB/s** | 1.209 GB/s | 0.796 GB/s | **3.189 GB/s** | 4.087 GB/s | 2.142 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.205 GB/s** | 1.494 GB/s | 0.809 GB/s | **4.452 GB/s** | 5.541 GB/s | 2.056 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.722 GB/s** | 1.949 GB/s | 1.104 GB/s | **4.374 GB/s** | 6.854 GB/s | 2.735 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.929 GB/s** | 1.105 GB/s | 0.623 GB/s | **2.694 GB/s** | 3.581 GB/s | 1.564 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.215 GB/s** | 1.558 GB/s | 0.820 GB/s | **4.597 GB/s** | 4.214 GB/s | 1.866 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.831 GB/s** | 1.132 GB/s | 0.577 GB/s | **2.347 GB/s** | 3.181 GB/s | 1.392 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2702** | 0.3778 | 0.2714 | **0.369 GB/s** | 0.553 GB/s | 0.306 GB/s | **1.364 GB/s** | 2.921 GB/s | 1.230 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.442 GB/s** | 6.792 GB/s | 4.082 GB/s | **9.734 GB/s** | 11.219 GB/s | 11.169 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **1.480 GB/s** | 6.587 GB/s | 4.132 GB/s | **4.537 GB/s** | 1.991 GB/s | 1.834 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.258 GB/s** | 1.675 GB/s | 0.954 GB/s | **3.661 GB/s** | 4.395 GB/s | 2.165 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **51.5** | 11.9 | 12.2 | **4.31x** | **4.21x** | **4.76** | 3.05 | 2.33 |
| **`events.tsv`** | **61.2** | 10.0 | 17.6 | **6.12x** | **3.47x** | **5.00** | 2.81 | 2.84 |
| **`app-logs.logfmt`** | **65.6** | 14.3 | 18.5 | **4.58x** | **3.54x** | **5.34** | 3.45 | 2.88 |
| **`json-records.ndjson`** | **121.1** | 24.4 | 32.0 | **4.96x** | **3.78x** | **6.93** | 4.47 | 3.82 |
| **`field-order-optional-escaping.ndjson`** | **35.6** | 9.3 | 9.5 | **3.81x** | **3.76x** | **3.83** | 2.65 | 2.11 |
| **`event-mix-identifiers.ndjson`** | **95.2** | 16.6 | 15.9 | **5.74x** | **6.00x** | **6.08** | 3.49 | 2.70 |
| **`bursty-variable-records.ndjson`** | **29.1** | 9.2 | 7.5 | **3.17x** | **3.88x** | **3.44** | 2.59 | 1.89 |
| **`source-and-docs.txt`** | **2.6** | 3.4 | 2.3 | **0.78x** | **1.16x** | **1.23** | 1.62 | 1.12 |
| **`ui-render-32bpp.bmp`** | **1654.8** | 622.3 | 740.7 | **2.66x** | **2.23x** | **23.58** | 17.58 | 17.10 |
| **`rgb-frame-24bpp.bmp`** | **2087.9** | 159.5 | 230.6 | **13.09x** | **9.05x** | **17.56** | 8.33 | 8.59 |
| **Geometric Mean** | **86.1** | 21.4 | 23.7 | **4.03x** | **3.63x** | **5.70** | 3.87 | 3.24 |
