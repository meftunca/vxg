## VXG v4 Benchmark, Chimera Envelope & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Individual Codec Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 (a1) | LZ4-HC (L9) | Zstd-1 | Zstd-9 | VXG Enc | LZ4 (a1) Enc | LZ4-HC Enc | Zstd-1 Enc | VXG Dec | LZ4 (a1) Dec | Zstd-1 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.1654 | 0.0841 | 0.0797 | **9.142** | 7.997 | 0.445 | 5.245 | **40.042** | 29.079 | 7.521 |
| **`events.tsv`** | 16 MiB | **0.0326** | 0.2223 | 0.1515 | 0.0741 | 0.0683 | **8.548** | 2.830 | 0.704 | 4.822 | **42.330** | 5.363 | 8.223 |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.1623 | 0.0695 | 0.0722 | **1.517** | 3.687 | 0.397 | 2.507 | **10.329** | 8.966 | 1.845 |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.1147 | 0.0542 | 0.0488 | **4.672** | 4.789 | 0.645 | 2.821 | **11.489** | 12.848 | 6.215 |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1492 | 0.1043 | 0.0883 | **3.610** | 3.186 | 0.232 | 2.224 | **10.604** | 11.657 | 5.113 |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.1134 | 0.0779 | 0.0698 | **8.094** | 3.606 | 0.604 | 3.264 | **43.522** | 13.767 | 4.352 |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1308 | 0.1196 | 0.0926 | **4.947** | 6.808 | 0.390 | 2.666 | **17.789** | 18.065 | 8.150 |
| **`source-and-docs.txt`** | 16 MiB | **0.3059** | 0.4230 | 0.3399 | 0.3082 | 0.2670 | **2.442** | 2.730 | 0.364 | 1.623 | **8.860** | 11.191 | 5.966 |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0140 | 0.0091 | 0.0086 | **38.377** | 19.878 | 6.065 | 6.180 | **54.064** | 44.974 | 12.966 |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0227 | 0.0119 | 0.0122 | **40.563** | 24.520 | 0.460 | 17.805 | **74.369** | 37.977 | 22.640 |
| **Geometric Mean** | — | **0.0252** | 0.1284 | 0.1011 | 0.0614 | 0.0561 | **7.052** | 5.780 | 0.582 | 3.794 | **23.821** | 15.842 | 6.843 |

### 2. "Chimera Envelope" Showdown (`VXG v4` vs Impossible Best-of-Competitor Composites)

Each **Chimera** row grants a competitor family an impossible synthetic combination on every single dataset simultaneously:
- **`LZ4 Chimera`:** `min_ratio(lz4-a1, lz4-a4, lz4hc-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`LZ4-HC L9` ratio at `LZ4-fast` speed).
- **`Zstd Chimera`:** `min_ratio(zstd-l-3, zstd-l1, zstd-l3, zstd-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`Zstd L9` ratio at `Zstd -3` ultra-fast speed).
- **`All-Rivals Chimera`:** `min_ratio(all LZ4 & Zstd)` + `max_encode_GBps(all LZ4 & Zstd)` + `max_decode_GBps(all LZ4 & Zstd)` (`Zstd L9` ratio at `LZ4-fast` speed!).

| Dataset | VXG Ratio | LZ4 Chimera Ratio | Zstd Chimera Ratio | VXG Enc | LZ4 Chimera Enc | Zstd Chimera Enc | VXG Dec | LZ4 Chimera Dec | Zstd Chimera Dec | VXG `VPI` | LZ4 Chimera `VPI` | Zstd Chimera `VPI` | All-Rivals Chimera `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | **9.142** | 7.997 | 5.451 | **40.042** | 34.356 | 11.643 | **469.7** | 100.2 | 99.9 | 207.9 |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0683 | **8.548** | 2.864 | 5.610 | **42.330** | 33.958 | 10.162 | **583.0** | 65.1 | 110.6 | 202.1 |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | **1.517** | 3.687 | 2.507 | **10.329** | 8.966 | 6.304 | **112.1** | 35.4 | 57.2 | 82.7 |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | **4.672** | 4.789 | 3.290 | **11.489** | 16.851 | 9.970 | **323.3** | 78.3 | 117.4 | 184.2 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.1492 | 0.0883 | **3.610** | 3.186 | 2.659 | **10.604** | 18.029 | 5.671 | **139.1** | 50.8 | 44.0 | 85.8 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0698 | **8.094** | 8.325 | 4.903 | **43.522** | 34.379 | 11.220 | **755.9** | 149.2 | 106.3 | 242.4 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0926 | **4.947** | 6.808 | 4.024 | **17.789** | 20.824 | 8.150 | **195.4** | 91.0 | 61.8 | 128.6 |
| **`source-and-docs.txt`** | **0.3059** | 0.3399 | 0.2670 | **2.442** | 3.171 | 2.109 | **8.860** | 18.771 | 5.966 | **15.2** | 22.7 | 13.3 | 28.9 |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | **38.377** | 25.008 | 25.359 | **54.064** | 52.348 | 20.657 | **9518.9** | 2579.5 | 2653.5 | 4224.1 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **40.563** | 25.150 | 19.001 | **74.369** | 37.977 | 23.382 | **44254.7** | 1362.8 | 1765.9 | 2589.2 |
| **Geometric Mean** | **0.0252** | 0.1011 | 0.0557 | **7.052** | 6.551 | 5.163 | **23.821** | 24.711 | 10.106 | **514.8** | 125.8 | 129.6 | 236.2 |

### 3. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (a1) | VXG vs LZ4 Chimera | VXG vs Zstd Chimera | VXG vs All-Rivals Chimera | VXG `BCI` | All-Rivals Chimera `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **469.7** | 76.5 | 74.7 | **6.14x** | **4.69x** | **4.70x** | **2.26x** | **20.79** | 15.10 |
| **`events.tsv`** | **583.0** | 17.5 | 85.0 | **33.27x** | **8.95x** | **5.27x** | **2.88x** | **22.30** | 14.08 |
| **`app-logs.logfmt`** | **112.1** | 28.6 | 30.9 | **3.91x** | **3.17x** | **1.96x** | **1.36x** | **7.63** | 7.80 |
| **`json-records.ndjson`** | **323.3** | 52.4 | 77.2 | **6.17x** | **4.13x** | **2.75x** | **1.76x** | **13.33** | 11.83 |
| **`field-order-optional-escaping.ndjson`** | **139.1** | 28.6 | 32.3 | **4.86x** | **2.74x** | **3.16x** | **1.62x** | **9.51** | 8.66 |
| **`event-mix-identifiers.ndjson`** | **755.9** | 45.6 | 48.4 | **16.57x** | **5.07x** | **7.11x** | **3.12x** | **24.21** | 16.01 |
| **`bursty-variable-records.ndjson`** | **195.4** | 53.7 | 39.0 | **3.64x** | **2.15x** | **3.16x** | **1.52x** | **12.24** | 11.52 |
| **`source-and-docs.txt`** | **15.2** | 13.1 | 10.1 | **1.16x** | **0.67x** | **1.14x** | **0.53x** | **4.14** | 6.06 |
| **`ui-render-32bpp.bmp`** | **9518.9** | 2131.7 | 982.0 | **4.47x** | **3.69x** | **3.59x** | **2.25x** | **75.69** | 53.59 |
| **`rgb-frame-24bpp.bmp`** | **44254.7** | 1343.8 | 1682.1 | **32.93x** | **32.47x** | **25.06x** | **17.09x** | **134.45** | 43.09 |
| **Geometric Mean** | **514.8** | 74.5 | 82.9 | **6.91x** | **4.09x** | **3.97x** | **2.18x** | **18.83** | 14.60 |

### 4. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 Chimera `GCI` | Zstd Chimera `GCI` | All-Rivals Chimera `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **2.360** | 1.243 | 3.241 | **0.586** | 0.528 | 1.669 | **399.5** | 143.0 | 45.7 | 296.8 |
| **`events.tsv`** | **2.428** | 1.784 | 3.122 | **0.569** | 0.656 | 1.580 | **495.9** | 82.5 | 58.6 | 256.2 |
| **`app-logs.logfmt`** | **2.113** | 1.212 | 2.972 | **0.513** | 0.478 | 1.606 | **107.7** | 57.2 | 30.3 | 133.4 |
| **`json-records.ndjson`** | **1.782** | 0.965 | 2.175 | **0.436** | 0.416 | 1.025 | **367.0** | 148.2 | 87.9 | 348.6 |
| **`field-order-optional-escaping.ndjson`** | **3.585** | 1.950 | 4.222 | **0.828** | 0.724 | 2.106 | **80.7** | 53.2 | 19.8 | 89.9 |
| **`event-mix-identifiers.ndjson`** | **2.422** | 1.456 | 3.142 | **0.590** | 0.670 | 1.730 | **632.5** | 194.6 | 52.2 | 316.3 |
| **`bursty-variable-records.ndjson`** | **3.615** | 1.964 | 4.357 | **1.030** | 0.863 | 2.395 | **101.2** | 84.0 | 23.1 | 118.7 |
| **`source-and-docs.txt`** | **6.230** | 4.491 | 7.697 | **1.325** | 1.001 | 2.927 | **5.3** | 12.1 | 3.3 | 15.4 |
| **`ui-render-32bpp.bmp`** | **0.476** | 0.347 | 0.606 | **0.325** | 0.175 | 0.501 | **24188.6** | 10459.1 | 5777.7 | 17127.1 |
| **`rgb-frame-24bpp.bmp`** | **0.601** | 0.421 | 0.608 | **0.338** | 0.308 | 0.439 | **98162.9** | 4018.2 | 3823.6 | 7634.4 |
| **Geometric Mean** | **2.033** | 1.252 | 2.529 | **0.594** | 0.524 | 1.376 | **468.4** | 183.9 | 81.0 | 345.2 |
