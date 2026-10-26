## VXG v4 Benchmark, Composite Index & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.326 GB/s** | 1.328 GB/s | 0.653 GB/s | **8.328 GB/s** | 5.644 GB/s | 1.669 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.134 GB/s** | 0.868 GB/s | 0.798 GB/s | **5.979 GB/s** | 4.002 GB/s | 2.207 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.224 GB/s** | 0.983 GB/s | 0.764 GB/s | **6.979 GB/s** | 5.344 GB/s | 2.150 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.487 GB/s** | 1.646 GB/s | 1.055 GB/s | **11.025 GB/s** | 5.471 GB/s | 2.286 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.986 GB/s** | 0.891 GB/s | 0.595 GB/s | **4.203 GB/s** | 3.896 GB/s | 1.474 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **0.986 GB/s** | 1.190 GB/s | 0.777 GB/s | **5.669 GB/s** | 3.837 GB/s | 1.655 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.812 GB/s** | 0.920 GB/s | 0.576 GB/s | **3.177 GB/s** | 2.996 GB/s | 1.587 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2712** | 0.3785 | 0.2722 | **0.357 GB/s** | 0.478 GB/s | 0.390 GB/s | **1.563 GB/s** | 2.934 GB/s | 0.954 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **7.247 GB/s** | 3.307 GB/s | 2.907 GB/s | **11.445 GB/s** | 13.356 GB/s | 3.477 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **5.189 GB/s** | 4.210 GB/s | 3.358 GB/s | **10.123 GB/s** | 6.459 GB/s | 5.060 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.400 GB/s** | 1.284 GB/s | 0.918 GB/s | **5.916 GB/s** | 4.869 GB/s | 2.028 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **81.6** | 13.7 | 12.4 | **5.94x** | **6.58x** | **6.47** | 3.35 | 2.35 |
| **`events.tsv`** | **78.0** | 8.4 | 17.9 | **9.30x** | **4.35x** | **5.88** | 2.50 | 2.88 |
| **`app-logs.logfmt`** | **82.8** | 11.4 | 18.4 | **7.25x** | **4.49x** | **6.23** | 2.97 | 2.87 |
| **`json-records.ndjson`** | **178.7** | 20.1 | 28.6 | **8.91x** | **6.24x** | **8.98** | 3.92 | 3.54 |
| **`field-order-optional-escaping.ndjson`** | **45.8** | 8.7 | 9.0 | **5.23x** | **5.10x** | **4.53** | 2.54 | 2.03 |
| **`event-mix-identifiers.ndjson`** | **95.2** | 13.8 | 14.6 | **6.88x** | **6.54x** | **6.08** | 3.09 | 2.55 |
| **`bursty-variable-records.ndjson`** | **33.4** | 8.0 | 8.0 | **4.16x** | **4.18x** | **3.77** | 2.37 | 1.97 |
| **`source-and-docs.txt`** | **2.8** | 3.1 | 2.2 | **0.88x** | **1.23x** | **1.27** | 1.55 | 1.11 |
| **`ui-render-32bpp.bmp`** | **1903.2** | 473.8 | 348.7 | **4.02x** | **5.46x** | **25.88** | 14.66 | 10.35 |
| **`rgb-frame-24bpp.bmp`** | **5839.7** | 229.6 | 345.3 | **25.43x** | **16.91x** | **34.85** | 10.62 | 11.25 |
| **Geometric Mean** | **115.4** | 19.7 | 22.5 | **5.86x** | **5.13x** | **6.93** | 3.66 | 3.13 |
