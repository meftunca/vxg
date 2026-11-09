## VXG v4 Benchmark Summary (`linux` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`). All codecs run on independent `65536`-byte blocks with RAW fallback and checksums OFF.

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.040 GB/s** | 1.003 GB/s | 0.578 GB/s | **3.640 GB/s** | 5.424 GB/s | 1.804 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.292 GB/s** | 1.190 GB/s | 0.790 GB/s | **2.806 GB/s** | 4.052 GB/s | 2.120 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.222 GB/s** | 1.448 GB/s | 0.810 GB/s | **3.884 GB/s** | 5.450 GB/s | 2.001 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.652 GB/s** | 1.892 GB/s | 1.075 GB/s | **3.885 GB/s** | 6.759 GB/s | 2.692 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.892 GB/s** | 1.076 GB/s | 0.614 GB/s | **2.301 GB/s** | 3.543 GB/s | 1.550 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.217 GB/s** | 1.510 GB/s | 0.814 GB/s | **4.751 GB/s** | 4.148 GB/s | 1.840 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.760 GB/s** | 1.101 GB/s | 0.572 GB/s | **2.227 GB/s** | 3.165 GB/s | 1.357 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2889** | 0.4064 | 0.2933 | **0.177 GB/s** | 0.518 GB/s | 0.285 GB/s | **1.023 GB/s** | 2.795 GB/s | 1.177 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **2.521 GB/s** | 6.405 GB/s | 3.761 GB/s | **9.652 GB/s** | 11.015 GB/s | 10.615 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **1.506 GB/s** | 5.880 GB/s | 3.915 GB/s | **4.532 GB/s** | 1.987 GB/s | 1.805 GB/s |
