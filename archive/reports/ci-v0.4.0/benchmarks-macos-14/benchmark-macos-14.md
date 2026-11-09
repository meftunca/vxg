## VXG v4 Benchmark Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`). All codecs run on independent `65536`-byte blocks with RAW fallback and checksums OFF.

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.909 GB/s** | 0.994 GB/s | 0.598 GB/s | **4.580 GB/s** | 4.582 GB/s | 1.507 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.182 GB/s** | 0.724 GB/s | 0.711 GB/s | **4.649 GB/s** | 2.967 GB/s | 1.738 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.013 GB/s** | 1.082 GB/s | 0.625 GB/s | **7.164 GB/s** | 5.235 GB/s | 1.854 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.288 GB/s** | 1.319 GB/s | 1.057 GB/s | **9.883 GB/s** | 5.622 GB/s | 2.203 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.643 GB/s** | 0.834 GB/s | 0.498 GB/s | **3.501 GB/s** | 1.900 GB/s | 1.097 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **0.832 GB/s** | 1.106 GB/s | 0.504 GB/s | **5.509 GB/s** | 3.879 GB/s | 1.426 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.450 GB/s** | 0.898 GB/s | 0.557 GB/s | **2.971 GB/s** | 2.323 GB/s | 1.119 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2889** | 0.4064 | 0.2933 | **0.253 GB/s** | 0.404 GB/s | 0.399 GB/s | **1.293 GB/s** | 2.592 GB/s | 1.169 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **4.117 GB/s** | 2.914 GB/s | 2.673 GB/s | **8.363 GB/s** | 13.012 GB/s | 3.238 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **4.171 GB/s** | 3.014 GB/s | 2.801 GB/s | **9.983 GB/s** | 6.762 GB/s | 5.538 GB/s |
