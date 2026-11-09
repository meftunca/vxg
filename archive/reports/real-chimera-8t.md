## VXG v4 Benchmark, Chimera Envelope & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`8`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Individual Codec Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 (a1) | LZ4-HC (L9) | Zstd-1 | Zstd-9 | VXG Enc | LZ4 (a1) Enc | LZ4-HC Enc | Zstd-1 Enc | VXG Dec | LZ4 (a1) Dec | Zstd-1 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 56.5 MB | **0.0914** | 0.2898 | 0.2180 | 0.1660 | 0.1480 | **4.688** | 3.142 | 0.371 | 4.254 | **28.599** | 11.203 | 7.989 |
| **`data.csv`** | 30.5 MB | **0.1796** | 0.4458 | 0.3759 | 0.2863 | 0.2651 | **2.801** | 3.261 | 0.435 | 2.579 | **9.882** | 15.021 | 5.370 |
| **`data.clf`** | 31.2 MB | **0.0980** | 0.2843 | 0.2177 | 0.1759 | 0.1504 | **3.029** | 2.122 | 0.437 | 2.282 | **10.313** | 7.186 | 4.604 |
| **Geometric Mean** | — | **0.1172** | 0.3324 | 0.2613 | 0.2030 | 0.1807 | **3.414** | 2.791 | 0.413 | 2.925 | **14.284** | 10.654 | 5.824 |

### 2. "Chimera Envelope" Showdown (`VXG v4` vs Impossible Best-of-Competitor Composites)

Each **Chimera** row grants a competitor family an impossible synthetic combination on every single dataset simultaneously:
- **`LZ4 Chimera`:** `min_ratio(lz4-a1, lz4-a4, lz4hc-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`LZ4-HC L9` ratio at `LZ4-fast` speed).
- **`Zstd Chimera`:** `min_ratio(zstd-l-3, zstd-l1, zstd-l3, zstd-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`Zstd L9` ratio at `Zstd -3` ultra-fast speed).
- **`All-Rivals Chimera`:** `min_ratio(all LZ4 & Zstd)` + `max_encode_GBps(all LZ4 & Zstd)` + `max_decode_GBps(all LZ4 & Zstd)` (`Zstd L9` ratio at `LZ4-fast` speed!).

| Dataset | VXG Ratio | LZ4 Chimera Ratio | Zstd Chimera Ratio | VXG Enc | LZ4 Chimera Enc | Zstd Chimera Enc | VXG Dec | LZ4 Chimera Dec | Zstd Chimera Dec | VXG `VPI` | LZ4 Chimera `VPI` | Zstd Chimera `VPI` | All-Rivals Chimera `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **0.0914** | 0.2180 | 0.1480 | **4.688** | 3.824 | 4.741 | **28.599** | 17.553 | 9.486 | **126.6** | 37.6 | 45.3 | 61.7 |
| **`data.csv`** | **0.1796** | 0.3759 | 0.2651 | **2.801** | 3.261 | 2.953 | **9.882** | 15.021 | 5.790 | **29.3** | 18.6 | 15.6 | 26.4 |
| **`data.clf`** | **0.0980** | 0.2177 | 0.1504 | **3.029** | 3.325 | 2.951 | **10.313** | 15.354 | 6.475 | **57.0** | 32.8 | 29.1 | 47.5 |
| **Geometric Mean** | **0.1172** | 0.2613 | 0.1807 | **3.414** | 3.461 | 3.457 | **14.284** | 15.938 | 7.085 | **59.6** | 28.4 | 27.4 | 42.6 |

### 3. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (a1) | VXG vs LZ4 Chimera | VXG vs Zstd Chimera | VXG vs All-Rivals Chimera | VXG `BCI` | All-Rivals Chimera `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **126.6** | 20.5 | 35.1 | **6.18x** | **3.37x** | **2.79x** | **2.05x** | **11.36** | 8.25 |
| **`data.csv`** | **29.3** | 15.7 | 13.0 | **1.87x** | **1.57x** | **1.88x** | **1.11x** | **5.36** | 5.70 |
| **`data.clf`** | **57.0** | 13.7 | 18.4 | **4.15x** | **1.74x** | **1.96x** | **1.20x** | **6.83** | 6.98 |
| **Geometric Mean** | **59.6** | 16.4 | 20.3 | **3.63x** | **2.10x** | **2.18x** | **1.40x** | **7.47** | 6.90 |

### 4. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 Chimera `GCI` | Zstd Chimera `GCI` | All-Rivals Chimera `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **3.314** | 2.560 | 4.859 | **0.786** | 0.761 | 2.350 | **78.5** | 36.5 | 17.5 | 59.8 |
| **`data.csv`** | **5.375** | 4.186 | 7.067 | **1.589** | 0.927 | 2.663 | **10.0** | 11.3 | 4.3 | 16.1 |
| **`data.clf`** | **5.373** | 3.258 | 5.226 | **1.322** | 0.896 | 2.443 | **21.4** | 23.7 | 9.7 | 34.3 |
| **Geometric Mean** | **4.574** | 3.268 | 5.640 | **1.182** | 0.858 | 2.482 | **25.6** | 21.4 | 9.0 | 32.1 |
