## VXG v4 Benchmark, Chimera Envelope & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Individual Codec Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 (a1) | LZ4-HC (L9) | Zstd-1 | Zstd-9 | VXG Enc | LZ4 (a1) Enc | LZ4-HC Enc | Zstd-1 Enc | VXG Dec | LZ4 (a1) Dec | Zstd-1 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.1654 | 0.0841 | 0.0797 | **1.208** | 1.497 | 0.133 | 1.047 | **6.796** | 6.378 | 2.097 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.1515 | 0.0741 | 0.0683 | **1.266** | 0.881 | 0.139 | 1.137 | **7.331** | 4.464 | 2.336 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.1623 | 0.0695 | 0.0722 | **1.244** | 1.314 | 0.094 | 0.657 | **7.013** | 4.539 | 1.871 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.1147 | 0.0542 | 0.0488 | **1.329** | 1.744 | 0.253 | 0.942 | **7.690** | 6.985 | 2.392 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1492 | 0.1043 | 0.0883 | **1.018** | 0.896 | 0.074 | 0.836 | **4.625** | 4.153 | 1.667 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.1134 | 0.0779 | 0.0698 | **1.047** | 1.529 | 0.083 | 0.982 | **4.480** | 4.643 | 1.979 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1308 | 0.1196 | 0.0926 | **0.813** | 1.006 | 0.066 | 0.607 | **2.954** | 2.992 | 1.001 |
| **`source-and-docs.txt`** | 16 MiB | **0.3059** | 0.4230 | 0.3399 | 0.3082 | 0.2670 | **0.287** | 0.371 | 0.037 | 0.273 | **1.361** | 2.233 | 0.927 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0140 | 0.0091 | 0.0086 | **5.863** | 5.490 | 0.978 | 5.131 | **8.021** | 13.364 | 3.764 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0227 | 0.0119 | 0.0122 | **5.659** | 5.995 | 0.083 | 4.863 | **8.416** | 6.706 | 5.302 |
| **Geometric Mean** | — | **0.0252** | 0.1284 | 0.1011 | 0.0614 | 0.0561 | **1.355** | 1.481 | 0.118 | 1.096 | **5.223** | 5.033 | 2.054 |

### 2. "Chimera Envelope" Showdown (`VXG v4` vs Impossible Best-of-Competitor Composites)

Each **Chimera** row grants a competitor family an impossible synthetic combination on every single dataset simultaneously:
- **`LZ4 Chimera`:** `min_ratio(lz4-a1, lz4-a4, lz4hc-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`LZ4-HC L9` ratio at `LZ4-fast` speed).
- **`Zstd Chimera`:** `min_ratio(zstd-l-3, zstd-l1, zstd-l3, zstd-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`Zstd L9` ratio at `Zstd -3` ultra-fast speed).
- **`All-Rivals Chimera`:** `min_ratio(all LZ4 & Zstd)` + `max_encode_GBps(all LZ4 & Zstd)` + `max_decode_GBps(all LZ4 & Zstd)` (`Zstd L9` ratio at `LZ4-fast` speed!).

| Dataset | VXG Ratio | LZ4 Chimera Ratio | Zstd Chimera Ratio | VXG Enc | LZ4 Chimera Enc | Zstd Chimera Enc | VXG Dec | LZ4 Chimera Dec | Zstd Chimera Dec | VXG `VPI` | LZ4 Chimera `VPI` | Zstd Chimera `VPI` | All-Rivals Chimera `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | **1.208** | 1.594 | 1.244 | **6.796** | 9.213 | 2.432 | **70.4** | 23.2 | 21.8 | 48.1 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0683 | **1.266** | 0.881 | 1.222 | **7.331** | 8.487 | 2.489 | **93.4** | 18.1 | 25.5 | 47.2 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | **1.244** | 1.314 | 0.878 | **7.013** | 6.686 | 1.898 | **83.7** | 18.3 | 18.6 | 42.6 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | **1.329** | 1.911 | 1.474 | **7.690** | 12.731 | 3.491 | **141.1** | 43.0 | 46.5 | 101.1 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | **1.018** | 1.021 | 0.836 | **4.625** | 6.924 | 1.966 | **48.8** | 17.8 | 14.5 | 30.1 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0698 | **1.047** | 1.529 | 0.982 | **4.480** | 9.030 | 2.581 | **87.2** | 32.8 | 22.8 | 53.2 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0926 | **0.813** | 1.058 | 0.643 | **2.954** | 4.999 | 1.568 | **32.3** | 17.6 | 10.8 | 24.8 |
| **`source-and-docs.txt`** | **0.3059** | 0.3399 | 0.2670 | **0.287** | 0.429 | 0.342 | **1.361** | 2.666 | 1.084 | **2.0** | 3.1 | 2.3 | 4.0 |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | **5.863** | 7.329 | 5.224 | **8.021** | 13.364 | 3.764 | **1433.1** | 705.6 | 514.1 | 1147.4 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **5.659** | 6.618 | 6.567 | **8.416** | 8.131 | 5.563 | **5560.3** | 323.5 | 506.4 | 614.5 |
| **Geometric Mean** | **0.0252** | 0.1011 | 0.0557 | **1.355** | 1.615 | 1.284 | **5.223** | 7.551 | 2.432 | **105.7** | 34.5 | 31.7 | 63.7 |

### 3. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (a1) | VXG vs LZ4 Chimera | VXG vs Zstd Chimera | VXG vs All-Rivals Chimera | VXG `BCI` | All-Rivals Chimera `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **70.4** | 15.5 | 17.6 | **4.54x** | **3.04x** | **3.22x** | **1.46x** | **5.86** | 5.69 |
| **`events.tsv`** | **93.4** | 8.9 | 22.0 | **10.47x** | **5.17x** | **3.66x** | **1.98x** | **6.58** | 5.34 |
| **`app-logs.logfmt`** | **83.7** | 12.2 | 16.0 | **6.88x** | **4.58x** | **4.51x** | **1.96x** | **6.28** | 5.02 |
| **`json-records.ndjson`** | **141.1** | 23.3 | 27.7 | **6.05x** | **3.28x** | **3.03x** | **1.39x** | **7.67** | 7.93 |
| **`field-order-optional-escaping.ndjson`** | **48.8** | 9.0 | 11.3 | **5.39x** | **2.74x** | **3.36x** | **1.62x** | **4.73** | 4.31 |
| **`event-mix-identifiers.ndjson`** | **87.2** | 17.2 | 17.9 | **5.06x** | **2.66x** | **3.82x** | **1.64x** | **5.74** | 5.83 |
| **`bursty-variable-records.ndjson`** | **32.3** | 8.4 | 6.5 | **3.84x** | **1.84x** | **2.98x** | **1.30x** | **3.68** | 3.85 |
| **`source-and-docs.txt`** | **2.0** | 2.2 | 1.6 | **0.95x** | **0.65x** | **0.90x** | **0.51x** | **1.09** | 1.62 |
| **`ui-render-32bpp.bmp`** | **1433.1** | 610.7 | 482.1 | **2.35x** | **2.03x** | **2.79x** | **1.25x** | **21.42** | 22.48 |
| **`rgb-frame-24bpp.bmp`** | **5560.3** | 279.2 | 425.4 | **19.91x** | **17.19x** | **10.98x** | **9.05x** | **33.73** | 16.52 |
| **Geometric Mean** | **105.7** | 21.3 | 24.4 | **4.97x** | **3.06x** | **3.33x** | **1.66x** | **6.55** | 6.09 |

### 4. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 Chimera `GCI` | Zstd Chimera `GCI` | All-Rivals Chimera `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **2.720** | 1.470 | 3.729 | **0.645** | 0.569 | 2.132 | **53.1** | 29.1 | 9.0 | 60.5 |
| **`events.tsv`** | **2.811** | 2.045 | 3.289 | **0.641** | 0.789 | 1.846 | **69.5** | 20.1 | 11.8 | 52.5 |
| **`app-logs.logfmt`** | **2.830** | 1.544 | 3.485 | **0.555** | 0.598 | 1.820 | **66.8** | 22.8 | 8.0 | 53.2 |
| **`json-records.ndjson`** | **2.079** | 1.180 | 2.601 | **0.469** | 0.498 | 1.470 | **142.9** | 71.8 | 29.1 | 168.9 |
| **`field-order-optional-escaping.ndjson`** | **4.309** | 2.497 | 4.654 | **0.894** | 0.895 | 2.455 | **24.9** | 15.6 | 5.0 | 26.4 |
| **`event-mix-identifiers.ndjson`** | **2.878** | 1.694 | 3.530 | **0.648** | 0.758 | 2.029 | **63.9** | 39.1 | 9.8 | 63.6 |
| **`bursty-variable-records.ndjson`** | **3.785** | 2.084 | 4.710 | **1.060** | 0.824 | 2.775 | **16.1** | 15.0 | 3.8 | 21.2 |
| **`source-and-docs.txt`** | **5.027** | 3.797 | 6.694 | **1.240** | 0.800 | 2.509 | **0.8** | 1.9 | 0.7 | 2.4 |
| **`ui-render-32bpp.bmp`** | **0.393** | 0.293 | 0.545 | **0.253** | 0.137 | 0.400 | **4545.3** | 3709.8 | 1501.4 | 6032.7 |
| **`rgb-frame-24bpp.bmp`** | **0.480** | 0.332 | 0.608 | **0.282** | 0.221 | 0.471 | **15106.6** | 1292.9 | 1067.3 | 2456.4 |
| **Geometric Mean** | **2.139** | 1.334 | 2.679 | **0.598** | 0.532 | 1.527 | **93.4** | 48.9 | 18.6 | 90.2 |
