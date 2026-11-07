## VXG v4 Benchmark, Chimera Envelope & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Individual Codec Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 (a1) | LZ4-HC (L9) | Zstd-1 | Zstd-9 | VXG Enc | LZ4 (a1) Enc | LZ4-HC Enc | Zstd-1 Enc | VXG Dec | LZ4 (a1) Dec | Zstd-1 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | 56.5 MB | **0.0914** | 0.2898 | 0.2180 | 0.1660 | 0.1480 | **0.661** | 1.051 | 0.089 | 0.656 | **2.838** | 3.434 | 1.464 |
| **`data.csv`** | 30.5 MB | **0.1796** | 0.4458 | 0.3759 | 0.2863 | 0.2651 | **0.568** | 0.600 | 0.080 | 0.555 | **2.233** | 3.504 | 1.320 |
| **`data.clf`** | 31.2 MB | **0.0980** | 0.2843 | 0.2177 | 0.1759 | 0.1504 | **0.573** | 0.836 | 0.087 | 0.701 | **2.051** | 3.063 | 1.133 |
| **Geometric Mean** | — | **0.1172** | 0.3324 | 0.2613 | 0.2030 | 0.1807 | **0.599** | 0.808 | 0.085 | 0.634 | **2.352** | 3.328 | 1.298 |

### 2. "Chimera Envelope" Showdown (`VXG v4` vs Impossible Best-of-Competitor Composites)

Each **Chimera** row grants a competitor family an impossible synthetic combination on every single dataset simultaneously:
- **`LZ4 Chimera`:** `min_ratio(lz4-a1, lz4-a4, lz4hc-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`LZ4-HC L9` ratio at `LZ4-fast` speed).
- **`Zstd Chimera`:** `min_ratio(zstd-l-3, zstd-l1, zstd-l3, zstd-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`Zstd L9` ratio at `Zstd -3` ultra-fast speed).
- **`All-Rivals Chimera`:** `min_ratio(all LZ4 & Zstd)` + `max_encode_GBps(all LZ4 & Zstd)` + `max_decode_GBps(all LZ4 & Zstd)` (`Zstd L9` ratio at `LZ4-fast` speed!).

| Dataset | VXG Ratio | LZ4 Chimera Ratio | Zstd Chimera Ratio | VXG Enc | LZ4 Chimera Enc | Zstd Chimera Enc | VXG Dec | LZ4 Chimera Dec | Zstd Chimera Dec | VXG `VPI` | LZ4 Chimera `VPI` | Zstd Chimera `VPI` | All-Rivals Chimera `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **0.0914** | 0.2180 | 0.1480 | **0.661** | 1.051 | 1.001 | **2.838** | 5.795 | 1.966 | **15.0** | 11.3 | 9.5 | 16.7 |
| **`data.csv`** | **0.1796** | 0.3759 | 0.2651 | **0.568** | 0.733 | 0.648 | **2.233** | 3.856 | 1.645 | **6.3** | 4.5 | 3.9 | 6.3 |
| **`data.clf`** | **0.0980** | 0.2177 | 0.1504 | **0.573** | 0.836 | 0.785 | **2.051** | 5.268 | 1.816 | **11.1** | 9.6 | 7.9 | 14.0 |
| **Geometric Mean** | **0.1172** | 0.2613 | 0.1807 | **0.599** | 0.864 | 0.799 | **2.352** | 4.901 | 1.804 | **10.1** | 7.9 | 6.6 | 11.4 |

### 3. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (a1) | VXG vs LZ4 Chimera | VXG vs Zstd Chimera | VXG vs All-Rivals Chimera | VXG `BCI` | All-Rivals Chimera `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **15.0** | 6.6 | 5.9 | **2.28x** | **1.32x** | **1.58x** | **0.90x** | **2.74** | 3.45 |
| **`data.csv`** | **6.3** | 3.3 | 3.0 | **1.93x** | **1.40x** | **1.61x** | **0.99x** | **1.92** | 2.20 |
| **`data.clf`** | **11.1** | 5.6 | 5.1 | **1.97x** | **1.15x** | **1.39x** | **0.79x** | **2.29** | 3.08 |
| **Geometric Mean** | **10.1** | 4.9 | 4.5 | **2.05x** | **1.29x** | **1.52x** | **0.89x** | **2.29** | 2.86 |

### 4. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 Chimera `GCI` | Zstd Chimera `GCI` | All-Rivals Chimera `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** | **2.346** | 2.797 | 4.973 | **0.607** | 0.823 | 2.374 | **12.6** | 11.5 | 3.4 | 16.9 |
| **`data.csv`** | **5.973** | 5.034 | 7.797 | **1.796** | 1.086 | 3.095 | **1.9** | 2.1 | 0.9 | 3.0 |
| **`data.clf`** | **6.379** | 3.764 | 6.352 | **1.548** | 1.080 | 2.847 | **3.5** | 6.0 | 2.3 | 8.6 |
| **Geometric Mean** | **4.471** | 3.756 | 6.268 | **1.191** | 0.989 | 2.755 | **4.4** | 5.3 | 1.9 | 7.6 |
