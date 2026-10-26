## VXG v4 Benchmark, Composite Index & Energy Summary (`linux` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.289 GB/s** | 1.045 GB/s | 0.664 GB/s | **5.123 GB/s** | 6.202 GB/s | 2.173 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.275 GB/s** | 1.087 GB/s | 0.879 GB/s | **5.590 GB/s** | 4.394 GB/s | 2.567 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.429 GB/s** | 1.449 GB/s | 0.933 GB/s | **5.950 GB/s** | 5.941 GB/s | 2.495 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.805 GB/s** | 1.946 GB/s | 1.286 GB/s | **7.904 GB/s** | 7.495 GB/s | 3.225 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.028 GB/s** | 0.962 GB/s | 0.659 GB/s | **4.253 GB/s** | 3.477 GB/s | 1.634 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.388 GB/s** | 1.416 GB/s | 0.849 GB/s | **5.854 GB/s** | 3.890 GB/s | 1.910 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.806 GB/s** | 0.945 GB/s | 0.606 GB/s | **2.816 GB/s** | 2.937 GB/s | 1.385 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2710** | 0.3782 | 0.2720 | **0.378 GB/s** | 0.503 GB/s | 0.312 GB/s | **1.262 GB/s** | 3.195 GB/s | 1.359 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.107 GB/s** | 7.815 GB/s | 4.989 GB/s | **9.806 GB/s** | 11.492 GB/s | 4.319 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **5.085 GB/s** | 7.725 GB/s | 4.640 GB/s | **6.534 GB/s** | 2.545 GB/s | 2.334 GB/s |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.499 GB/s** | 1.621 GB/s | 1.055 GB/s | **4.878 GB/s** | 4.628 GB/s | 2.200 GB/s |

### 2. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)` (equal 1/3 weight on compression factor, encode speed, and decode speed)

| Dataset | VXG v4 `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (`VPI`) | VXG vs Zstd-1 (`VPI`) | VXG v4 `BCI` | LZ4 `BCI` | Zstd-1 `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **63.1** | 12.8 | 14.3 | **4.94x** | **4.42x** | **5.45** | 3.19 | 2.58 |
| **`events.tsv`** | **79.9** | 9.8 | 20.3 | **8.13x** | **3.94x** | **5.98** | 2.78 | 3.12 |
| **`app-logs.logfmt`** | **82.6** | 14.6 | 21.9 | **5.65x** | **3.76x** | **6.22** | 3.50 | 3.22 |
| **`json-records.ndjson`** | **166.7** | 25.5 | 37.5 | **6.53x** | **4.44x** | **8.57** | 4.60 | 4.24 |
| **`field-order-optional-escaping.ndjson`** | **47.0** | 8.6 | 9.9 | **5.48x** | **4.73x** | **4.62** | 2.50 | 2.18 |
| **`event-mix-identifiers.ndjson`** | **114.8** | 15.2 | 16.3 | **7.56x** | **7.02x** | **6.89** | 3.29 | 2.75 |
| **`bursty-variable-records.ndjson`** | **31.4** | 8.1 | 7.7 | **3.89x** | **4.10x** | **3.62** | 2.38 | 1.91 |
| **`source-and-docs.txt`** | **2.5** | 3.4 | 2.4 | **0.76x** | **1.07x** | **1.21** | 1.62 | 1.16 |
| **`ui-render-32bpp.bmp`** | **1617.2** | 675.6 | 509.3 | **2.39x** | **3.18x** | **23.22** | 18.57 | 13.32 |
| **`rgb-frame-24bpp.bmp`** | **4644.2** | 195.3 | 275.7 | **23.78x** | **16.84x** | **29.91** | 9.53 | 9.68 |
| **Geometric Mean** | **108.5** | 21.6 | 25.1 | **5.03x** | **4.32x** | **6.64** | 3.89 | 3.37 |
