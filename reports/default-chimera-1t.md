## VXG v4 Benchmark, Chimera Envelope & Energy Summary (`macos` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`) and `libzstd 1.5.7` (`zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`). Independent `65536`-byte blocks, RAW fallback, checksums OFF.

### 1. Individual Codec Compression Ratio & Throughput (`GB/s`)

| Dataset | Size | VXG v4 Ratio | LZ4 (a1) | LZ4-HC (L9) | Zstd-1 | Zstd-9 | VXG Enc | LZ4 (a1) Enc | LZ4-HC Enc | Zstd-1 Enc | VXG Dec | LZ4 (a1) Dec | Zstd-1 Dec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`zeros`** | 16 MiB | **0.0004** | 0.0046 | 0.0046 | 0.0008 | 0.0008 | **13.339** | 17.151 | 9.760 | 9.955 | **95.122** | 3.594 | 3.274 |
| **`period-37`** | 16 MiB | **0.0009** | 0.0051 | 0.0051 | 0.0014 | 0.0014 | **13.979** | 10.413 | 10.447 | 11.064 | **33.925** | 8.295 | 12.970 |
| **`json-records`** | 16 MiB | **0.0227** | 0.1497 | 0.1147 | 0.0542 | 0.0488 | **1.356** | 1.776 | 0.240 | 1.419 | **7.307** | 7.337 | 2.802 |
| **`random`** | 16 MiB | **1.0001** | 1.0005 | 1.0005 | 1.0005 | 1.0005 | **1.883** | 6.471 | 0.077 | 4.883 | **48.229** | 38.738 | 18.826 |
| **`low-entropy-random`** | 16 MiB | **0.5005** | 0.9341 | 0.8288 | 0.5010 | 0.5470 | **1.170** | 0.559 | 0.065 | 1.084 | **1.238** | 3.681 | 1.779 |
| **`mixed-4KiB`** | 16 MiB | **0.2877** | 0.2978 | 0.2875 | 0.2787 | 0.2743 | **2.325** | 4.104 | 0.133 | 1.484 | **13.203** | 16.222 | 4.417 |
| **Geometric Mean** | — | **0.0319** | 0.0996 | 0.0928 | 0.0449 | 0.0445 | **3.302** | 4.094 | 0.504 | 3.272 | **16.275** | 8.927 | 5.100 |

### 2. "Chimera Envelope" Showdown (`VXG v4` vs Impossible Best-of-Competitor Composites)

Each **Chimera** row grants a competitor family an impossible synthetic combination on every single dataset simultaneously:
- **`LZ4 Chimera`:** `min_ratio(lz4-a1, lz4-a4, lz4hc-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`LZ4-HC L9` ratio at `LZ4-fast` speed).
- **`Zstd Chimera`:** `min_ratio(zstd-l-3, zstd-l1, zstd-l3, zstd-l9)` + `max_encode_GBps(...)` + `max_decode_GBps(...)` (`Zstd L9` ratio at `Zstd -3` ultra-fast speed).
- **`All-Rivals Chimera`:** `min_ratio(all LZ4 & Zstd)` + `max_encode_GBps(all LZ4 & Zstd)` + `max_decode_GBps(all LZ4 & Zstd)` (`Zstd L9` ratio at `LZ4-fast` speed!).

| Dataset | VXG Ratio | LZ4 Chimera Ratio | Zstd Chimera Ratio | VXG Enc | LZ4 Chimera Enc | Zstd Chimera Enc | VXG Dec | LZ4 Chimera Dec | Zstd Chimera Dec | VXG `VPI` | LZ4 Chimera `VPI` | Zstd Chimera `VPI` | All-Rivals Chimera `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`zeros`** | **0.0004** | 0.0046 | 0.0008 | **13.339** | 17.151 | 11.920 | **95.122** | 3.798 | 3.274 | **100947.0** | 1768.2 | 8007.4 | 10345.4 |
| **`period-37`** | **0.0009** | 0.0051 | 0.0014 | **13.979** | 10.447 | 12.798 | **33.925** | 8.590 | 12.970 | **23737.3** | 1847.0 | 9473.9 | 9473.9 |
| **`json-records`** | **0.0227** | 0.1147 | 0.0488 | **1.356** | 2.034 | 1.738 | **7.307** | 12.463 | 2.802 | **138.9** | 43.9 | 45.2 | 103.2 |
| **`random`** | **1.0001** | 1.0005 | 1.0005 | **1.883** | 8.258 | 5.954 | **48.229** | 38.738 | 38.436 | **9.5** | 17.9 | 15.1 | 17.9 |
| **`low-entropy-random`** | **0.5005** | 0.8288 | 0.5010 | **1.170** | 1.674 | 6.123 | **1.238** | 5.808 | 29.802 | **2.4** | 3.8 | 27.0 | 27.0 |
| **`mixed-4KiB`** | **0.2877** | 0.2875 | 0.2743 | **2.325** | 4.104 | 2.192 | **13.203** | 16.222 | 7.010 | **19.3** | 28.4 | 14.3 | 29.7 |
| **Geometric Mean** | **0.0319** | 0.0928 | 0.0439 | **3.302** | 5.239 | 5.260 | **16.275** | 10.680 | 9.924 | **229.7** | 80.6 | 164.7 | 229.2 |

### 3. Unified Codec Performance Index (`VPI` & `BCI`)

- **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio` (higher is better)
- **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`

| Dataset | VXG `VPI` | LZ4 (a1) `VPI` | Zstd-1 `VPI` | VXG vs LZ4 (a1) | VXG vs LZ4 Chimera | VXG vs Zstd Chimera | VXG vs All-Rivals Chimera | VXG `BCI` | All-Rivals Chimera `BCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`zeros`** | **100947.0** | 1720.1 | 7177.3 | **58.69x** | **57.09x** | **12.61x** | **9.76x** | **153.20** | 43.71 |
| **`period-37`** | **23737.3** | 1812.2 | 8808.8 | **13.10x** | **12.85x** | **2.51x** | **2.51x** | **80.26** | 49.60 |
| **`json-records`** | **138.9** | 24.1 | 36.8 | **5.76x** | **3.17x** | **3.07x** | **1.35x** | **7.59** | 8.04 |
| **`random`** | **9.5** | 15.8 | 9.6 | **0.60x** | **0.53x** | **0.63x** | **0.53x** | **4.49** | 6.84 |
| **`low-entropy-random`** | **2.4** | 1.5 | 2.8 | **1.57x** | **0.64x** | **0.09x** | **0.09x** | **1.43** | 7.14 |
| **`mixed-4KiB`** | **19.3** | 27.4 | 9.2 | **0.70x** | **0.68x** | **1.35x** | **0.65x** | **4.74** | 6.24 |
| **Geometric Mean** | **229.7** | 60.7 | 91.0 | **3.78x** | **2.85x** | **1.39x** | **1.00x** | **11.90** | 13.21 |

### 4. Per-Process CPU Energy Efficiency (`J/GB`, Lower is Better) & Green Codec Index (`GCI`, Higher is Better)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `GCI` | LZ4 Chimera `GCI` | Zstd Chimera `GCI` | All-Rivals Chimera `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`zeros`** | **0.267** | 0.238 | 0.211 | **0.021** | 0.353 | 0.537 | **1361128.8** | 7704.0 | 25612.3 | 47807.7 |
| **`period-37`** | **0.278** | 0.198 | 0.286 | **0.085** | 0.317 | 0.172 | **154687.3** | 8812.0 | 45015.9 | 54023.0 |
| **`json-records`** | **1.479** | 0.898 | 2.223 | **0.395** | 0.401 | 1.013 | **181.6** | 86.4 | 38.0 | 203.2 |
| **`random`** | **1.108** | 0.352 | 0.765 | **0.067** | 0.097 | 0.086 | **35.0** | 103.0 | 73.2 | 108.3 |
| **`low-entropy-random`** | **3.364** | 5.670 | 3.547 | **1.555** | 1.017 | 1.287 | **1.1** | 3.4 | 110.0 | 110.0 |
| **`mixed-4KiB`** | **1.105** | 0.611 | 2.154 | **0.249** | 0.239 | 0.423 | **36.7** | 98.4 | 26.0 | 103.1 |
| **Geometric Mean** | **0.876** | 0.610 | 0.960 | **0.162** | 0.320 | 0.405 | **610.3** | 242.1 | 457.5 | 929.5 |
