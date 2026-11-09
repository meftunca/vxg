## VXG v4 Benchmark Summary (`windows` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). All codecs run on independent `65536`-byte blocks with RAW fallback and checksums OFF.

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.995 GB/s** | 0.898 GB/s | 0.386 GB/s | **3.764 GB/s** | 5.286 GB/s | 1.331 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.245 GB/s** | 0.785 GB/s | 0.562 GB/s | **3.223 GB/s** | 4.029 GB/s | 1.592 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.091 GB/s** | 0.896 GB/s | 0.577 GB/s | **4.260 GB/s** | 5.311 GB/s | 1.521 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.557 GB/s** | 1.043 GB/s | 0.541 GB/s | **4.428 GB/s** | 6.503 GB/s | 2.095 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.840 GB/s** | 0.731 GB/s | 0.406 GB/s | **2.971 GB/s** | 3.461 GB/s | 1.177 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.186 GB/s** | 0.913 GB/s | 0.507 GB/s | **4.830 GB/s** | 4.059 GB/s | 1.436 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.717 GB/s** | 0.801 GB/s | 0.345 GB/s | **2.275 GB/s** | 3.153 GB/s | 1.073 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2848** | 0.3993 | 0.2872 | **0.181 GB/s** | 0.449 GB/s | 0.268 GB/s | **1.042 GB/s** | 2.688 GB/s | 0.943 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **2.453 GB/s** | 2.191 GB/s | 1.800 GB/s | **9.536 GB/s** | 16.019 GB/s | 10.760 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **1.469 GB/s** | 2.060 GB/s | 1.853 GB/s | **4.536 GB/s** | 1.967 GB/s | 1.756 GB/s |
