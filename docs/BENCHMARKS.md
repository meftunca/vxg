# Benchmark protocol

For the current numerical target, see [TARGET_4_006_12_15.md](TARGET_4_006_12_15.md).
The fourth axis, CPU energy per GB, has its own [energy protocol](ENERGY_BENCHMARKS.md)
with sustained workloads, idle observations and explicit measurement boundaries.

## VXG v0.4.1 Local Comparative Results, Composite Index & Energy (Apple Silicon `aarch64`, SIMD=`Neon`, `16 MiB`, `block=65536`)

Measured on Apple Silicon (`aarch64`, SIMD=`Neon`, `liblz4 1.10.0`, `libzstd 1.5.7`, `16 MiB` per dataset, 5 samples of `0.08 s` per phase, checksums OFF) via `vxg-bench --suite v4 --size-mib 16`.
- **VXG Performance Index (`VPI`):** $\text{VPI} = \frac{\sqrt{\text{Encode}_{\text{GB/s}} \cdot \text{Decode}_{\text{GB/s}}}}{\text{Ratio}}$ (higher is better).
- **Balanced 3-Axis Cube Index (`BCI`):** $\text{BCI} = \left(\frac{\text{Encode}_{\text{GB/s}} \cdot \text{Decode}_{\text{GB/s}}}{\text{Ratio}}\right)^{1/3}$ (higher is better).
- **Green Codec Index (`GCI`):** $\text{GCI} = \frac{\text{VPI}}{\sqrt{J_{\text{enc}} \cdot J_{\text{dec}}}}$ using per-process hardware energy (`proc_pid_rusage` `RUSAGE_INFO_V6` `ri_energy_nj` in `J/GB`).

### 1-Thread (`threads = 1`, `block = 65536`, `16 MiB`)

| Dataset | VXG Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG Enc / Dec (`GB/s`) | LZ4 Enc / Dec (`GB/s`) | Zstd-1 Enc / Dec (`GB/s`) | VXG `VPI` (`vs LZ4` / `Zstd-1`) | VXG Enc / Dec (`J/GB`) | VXG `GCI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | **1.704 / 9.420** | 1.906 / 7.468 | 1.163 / 2.249 | **98.4** (`5.20x` / `5.11x`) | **2.921 / 0.671** | **70.2** (`3.53x` / `10.80x`) |
| **`events.tsv`** | **0.0334** | 0.2223 | 0.0741 | **1.686 / 9.441** | 1.505 / 5.750 | 1.323 / 2.610 | **119.5** (`9.03x` / `4.76x`) | **2.952 / 0.653** | **86.0** (`8.27x` / `8.69x`) |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | **1.861 / 10.725** | 1.735 / 6.979 | 1.264 / 2.483 | **126.6** (`7.30x` / `4.97x`) | **2.683 / 0.590** | **100.6** (`5.47x` / `9.86x`) |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | **2.202 / 12.531** | 2.296 / 9.061 | 1.606 / 3.263 | **231.8** (`7.61x` / `5.49x`) | **2.108 / 0.466** | **233.9** (`5.86x` / `10.73x`) |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | **1.322 / 5.583** | 1.300 / 4.684 | 0.939 / 1.863 | **61.1** (`5.28x` / `4.82x`) | **3.777 / 0.900** | **33.1** (`4.19x` / `8.95x`) |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | **1.756 / 8.807** | 1.944 / 5.702 | 1.291 / 2.290 | **158.4** (`7.35x` / `7.18x`) | **2.711 / 0.614** | **122.8** (`6.64x` / `15.54x`) |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | **1.154 / 4.523** | 1.385 / 4.320 | 0.852 / 1.737 | **47.6** (`4.02x` / `4.68x`) | **3.914 / 1.169** | **22.2** (`2.85x` / `8.54x`) |
| **`source-and-docs.txt`** | **0.2698** | 0.3777 | 0.2713 | **0.592 / 2.468** | 0.691 / 4.025 | 0.567 / 1.488 | **4.5** (`1.02x` / `1.32x`) | **6.887 / 1.718** | **1.3** (`0.68x` / `2.17x`) |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0091 | **9.533 / 14.246** | 13.631 / 19.663 | 6.624 / 4.768 | **2435.3** (`2.09x` / `3.95x`) | **0.562 / 0.345** | **5530.4** (`1.22x` / `5.24x`) |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **8.793 / 14.846** | 11.159 / 10.381 | 7.379 / 7.063 | **9206.1** (`19.42x` / `15.22x`) | **0.631 / 0.368** | **19116.6** (`13.67x` / `17.18x`) |
| **10-Dataset Geometric Mean** | **0.0250** | 0.1272 | 0.0607 | **2.094 / 8.211** | 2.309 / 6.933 | 1.568 / 2.666 | **165.9** (`5.27x` / `4.93x`) | **2.333 / 0.665** | **133.2** (`3.92x` / `8.59x`) |

### 8-Thread (`threads = 8`, `block = 65536`, `16 MiB` — Dynamic Work-Stealing + Zero-Allocation Decode)

| Dataset | VXG Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG Enc / Dec (`GB/s`) | LZ4 Enc / Dec (`GB/s`) | Zstd-1 Enc / Dec (`GB/s`) | VXG `VPI` (`vs LZ4` / `Zstd-1`) | VXG Enc / Dec (`J/GB`) | VXG `GCI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | **11.122 / 47.196** | 12.834 / 41.839 | 7.848 / 15.575 | **562.4** (`4.84x` / `4.28x`) | **2.639 / 0.641** | **432.4** (`3.30x` / `8.81x`) |
| **`events.tsv`** | **0.0334** | 0.2223 | 0.0741 | **9.919 / 45.716** | 9.560 / 30.671 | 7.168 / 13.654 | **637.7** (`8.28x` / `4.77x`) | **2.625 / 0.623** | **498.7** (`7.63x` / `8.58x`) |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | **11.470 / 49.538** | 9.994 / 32.485 | 6.915 / 13.485 | **675.3** (`7.52x` / `4.86x`) | **2.461 / 0.552** | **579.2** (`5.73x` / `9.65x`) |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | **13.772 / 54.798** | 12.896 / 39.135 | 8.598 / 17.224 | **1212.0** (`8.08x` / `5.40x`) | **2.028 / 0.467** | **1246.1** (`5.97x` / `10.06x`) |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | **6.190 / 30.762** | 7.728 / 23.581 | 4.913 / 9.655 | **310.2** (`4.90x` / `4.70x`) | **3.968 / 0.861** | **167.9** (`3.64x` / `8.11x`) |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | **10.532 / 44.595** | 9.997 / 34.218 | 8.022 / 15.768 | **872.9** (`7.29x` / `6.05x`) | **2.398 / 0.584** | **737.8** (`6.64x` / `13.56x`) |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | **7.789 / 27.442** | 9.199 / 27.281 | 5.727 / 12.079 | **304.5** (`3.97x` / `4.38x`) | **3.697 / 1.124** | **149.4** (`2.87x` / `7.99x`) |
| **`source-and-docs.txt`** | **0.2698** | 0.3777 | 0.2713 | **4.306 / 16.844** | 5.077 / 25.848 | 4.205 / 10.641 | **31.6** (`1.05x` / `1.30x`) | **6.433 / 1.638** | **9.7** (`0.70x` / `1.94x`) |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0091 | **36.922 / 63.932** | 40.151 / 83.121 | 27.504 / 29.208 | **10153.2** (`2.47x` / `3.27x`) | **0.659 / 0.359** | **20873.8** (`1.58x` / `4.10x`) |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **35.666 / 64.509** | 37.010 / 54.381 | 30.885 / 39.982 | **38648.9** (`19.56x` / `13.13x`) | **0.717 / 0.376** | **74433.0** (`14.57x` / `14.77x`) |
| **10-Dataset Geometric Mean** | **0.0250** | 0.1272 | 0.0607 | **11.750 / 41.545** | 12.383 / 36.499 | 8.771 / 16.042 | **883.8** (`5.29x` / `4.53x`) | **2.274 / 0.647** | **728.7** (`3.96x` / `7.70x`) |

## VXG v0.4.1 Cross-Platform CI Benchmark Results (`macos-14`, `ubuntu-latest`, `windows-latest`)

Generated automatically via [`.github/workflows/benchmarks.yml`](../.github/workflows/benchmarks.yml) using `vxg-bench --suite v4 --size-mib 16`. Raw CSV and Markdown artifacts are stored in [`reports/ci-v0.4.1/`](../reports/ci-v0.4.1/).

### 1. macOS (`macos-14` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`).

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.296 GB/s** | 0.982 GB/s | 0.730 GB/s | **5.954 GB/s** | 5.514 GB/s | 1.739 GB/s | **68.2** (`5.84x` / `5.09x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.187 GB/s** | 0.952 GB/s | 0.751 GB/s | **7.044 GB/s** | 4.026 GB/s | 1.715 GB/s | **86.6** (`9.83x` / `5.65x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.614 GB/s** | 1.205 GB/s | 0.770 GB/s | **8.613 GB/s** | 5.748 GB/s | 2.104 GB/s | **105.6** (`8.05x` / `5.77x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **1.618 GB/s** | 1.437 GB/s | 0.917 GB/s | **9.242 GB/s** | 6.288 GB/s | 2.811 GB/s | **170.6** (`8.50x` / `5.77x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.175 GB/s** | 0.873 GB/s | 0.616 GB/s | **4.980 GB/s** | 3.274 GB/s | 1.348 GB/s | **54.4** (`6.86x` / `6.22x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.281 GB/s** | 1.260 GB/s | 0.770 GB/s | **5.830 GB/s** | 4.055 GB/s | 1.597 GB/s | **110.1** (`7.52x` / `7.73x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.940 GB/s** | 0.985 GB/s | 0.671 GB/s | **3.503 GB/s** | 3.411 GB/s | 1.467 GB/s | **37.8** (`4.26x` / `4.56x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2702** | 0.3778 | 0.2714 | **0.374 GB/s** | 0.496 GB/s | 0.390 GB/s | **1.696 GB/s** | 3.200 GB/s | 1.084 GB/s | **2.9** (`0.88x` / `1.23x`) |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **6.423 GB/s** | 3.846 GB/s | 2.456 GB/s | **9.170 GB/s** | 14.402 GB/s | 2.719 GB/s | **1603.8** (`3.02x` / `5.66x`) |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **6.025 GB/s** | 4.357 GB/s | 3.669 GB/s | **9.611 GB/s** | 8.715 GB/s | 5.913 GB/s | **6131.5** (`22.59x` / `15.71x`) |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.554 GB/s** | 1.307 GB/s | 0.920 GB/s | **5.896 GB/s** | 5.205 GB/s | 1.990 GB/s | **121.4** (`5.91x` / `5.45x`) |

### 2. Linux (`ubuntu-latest` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`).

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.706 GB/s** | 1.437 GB/s | 0.737 GB/s | **7.513 GB/s** | 7.170 GB/s | 2.395 GB/s | **87.9** (`5.46x` / `5.56x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.460 GB/s** | 1.281 GB/s | 1.034 GB/s | **6.709 GB/s** | 4.957 GB/s | 3.155 GB/s | **93.7** (`8.27x` / `3.84x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.885 GB/s** | 1.871 GB/s | 1.219 GB/s | **8.434 GB/s** | 7.626 GB/s | 3.201 GB/s | **113.0** (`6.00x` / `3.98x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.474 GB/s** | 2.484 GB/s | 1.650 GB/s | **10.906 GB/s** | 9.651 GB/s | 4.085 GB/s | **229.3** (`7.01x` / `4.79x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.377 GB/s** | 1.286 GB/s | 0.856 GB/s | **5.483 GB/s** | 4.587 GB/s | 2.113 GB/s | **61.8** (`5.42x` / `4.79x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.824 GB/s** | 1.836 GB/s | 1.145 GB/s | **7.681 GB/s** | 5.197 GB/s | 2.461 GB/s | **150.8** (`7.54x` / `7.00x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **1.032 GB/s** | 1.300 GB/s | 0.801 GB/s | **3.772 GB/s** | 3.918 GB/s | 1.792 GB/s | **41.1** (`3.76x` / `4.10x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2712** | 0.3785 | 0.2722 | **0.513 GB/s** | 0.661 GB/s | 0.416 GB/s | **2.205 GB/s** | 3.970 GB/s | 1.772 GB/s | **3.9** (`0.92x` / `1.24x`) |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **8.370 GB/s** | 9.805 GB/s | 5.723 GB/s | **14.818 GB/s** | 21.433 GB/s | 4.797 GB/s | **2327.3** (`2.25x` / `4.05x`) |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **7.311 GB/s** | 7.260 GB/s | 5.124 GB/s | **9.078 GB/s** | 2.881 GB/s | 2.622 GB/s | **6564.0** (`32.59x` / `21.37x`) |
| **Geometric Mean** | — | **0.0249** | 0.1270 | 0.0607 | **1.984 GB/s** | 2.038 GB/s | 1.308 GB/s | **6.822 GB/s** | 5.994 GB/s | 2.699 GB/s | **147.5** (`5.36x` / `4.76x`) |

### 3. Windows (`windows-latest` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`).

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode | VXG `VPI` (`vs LZ4` / `Zstd-1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **2.010 GB/s** | 1.364 GB/s | 0.872 GB/s | **8.915 GB/s** | 8.835 GB/s | 2.481 GB/s | **103.9** (`5.97x` / `5.94x`) |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.893 GB/s** | 1.115 GB/s | 0.945 GB/s | **8.583 GB/s** | 6.385 GB/s | 2.917 GB/s | **120.7** (`10.06x` / `5.38x`) |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.758 GB/s** | 1.312 GB/s | 1.045 GB/s | **9.922 GB/s** | 8.689 GB/s | 2.667 GB/s | **118.3** (`7.03x` / `4.93x`) |
| **`json-records.ndjson`** | 16 MiB | **0.0227** | 0.1497 | 0.0542 | **2.748 GB/s** | 1.776 GB/s | 1.261 GB/s | **14.409 GB/s** | 11.171 GB/s | 3.658 GB/s | **277.7** (`9.33x` / `7.01x`) |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **1.519 GB/s** | 1.132 GB/s | 0.760 GB/s | **6.405 GB/s** | 4.697 GB/s | 1.946 GB/s | **70.1** (`6.48x` / `6.01x`) |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **2.018 GB/s** | 1.513 GB/s | 0.995 GB/s | **9.347 GB/s** | 4.840 GB/s | 2.339 GB/s | **174.9** (`9.98x` / `8.93x`) |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **1.176 GB/s** | 1.178 GB/s | 0.733 GB/s | **4.472 GB/s** | 3.965 GB/s | 1.670 GB/s | **47.8** (`4.56x` / `5.16x`) |
| **`source-and-docs.txt`** | 16 MiB | **0.2678** | 0.3721 | 0.2672 | **0.576 GB/s** | 0.735 GB/s | 0.522 GB/s | **2.962 GB/s** | 5.253 GB/s | 1.805 GB/s | **4.9** (`0.92x` / `1.34x`) |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0048** | 0.0140 | 0.0091 | **10.452 GB/s** | 3.523 GB/s | 3.133 GB/s | **17.003 GB/s** | 47.484 GB/s | 18.978 GB/s | **2785.9** (`3.02x` / `3.29x`) |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0012** | 0.0227 | 0.0119 | **10.137 GB/s** | 3.554 GB/s | 2.892 GB/s | **11.329 GB/s** | 4.692 GB/s | 4.180 GB/s | **8634.5** (`48.02x` / `29.64x`) |
| **Geometric Mean** | — | **0.0249** | 0.1268 | 0.0606 | **2.295 GB/s** | 1.520 GB/s | 1.110 GB/s | **8.362 GB/s** | 7.519 GB/s | 3.080 GB/s | **175.9** (`6.60x` / `5.76x`) |

---

## Do not interpret supplied reference numbers as VXG performance

`reports/reference-c-baselines.csv` was actually generated in the preparation environment
by `verification/c_baselines.cpp`, linked to installed liblz4 1.10.0 and libzstd 1.5.7.
Its `scope` is `C_ONLY_NOT_VXG`. The environment had no Rust compiler or GPU. These rows
check that the reference libraries run; they neither establish VXG speed nor validate
the uncompiled Rust FFI harness. Do not compare the SIMT emulator speed with native LZ4.

The real comparative harness is `src/bin/bench.rs` + `src/baselines.rs`. It links reference
C libraries rather than reimplementations with potentially different performance.

## CPU comparison profiles

All codecs use identical uncompressed input, independent block size, worker count, RAW
fallback and checksum-off policy. Included configurations are VXG acceleration 1,
LZ4 fast accelerations 1 and 4, and Zstd levels -3, 1 and 3. This is not a comparison to
a single whole-file Zstd frame with cross-block history; that would be a separate workload.

Compression includes allocating the encoded result, scratch/context creation per worker,
actual block coding, RAW decisions and sorting. VXG additionally serializes its contiguous
frame; reference codecs retain vectors of blocks rather than serialize a real frame.
Their reported size adds the same `32+32*block_count` envelope for accounting. This is
close application-level coverage but **not identical memory-allocation/framing work**.
Do not publish it as an apples-to-apples intrinsic-kernel microbenchmark.

The primary CPU decode row is `cpu_preallocated_decode`: all codecs start with prepared
metadata and a preallocated output buffer. VXG header/directory parsing is outside that
timed loop; its command parsing remains inside. `cpu_full_frame_preallocated_decode` is
an additional VXG-safe row which includes header/directory parsing before the same checked
decode. `cpu_prepared_validate_commands` measures one-time command validation on an
already parsed directory; `cpu_parse_validate_commands` includes both parse and validation.
They are accounting rows, not substitutes for decode. `cpu_prevalidated_decode` exists
only with `--validated-decode`, after a `ValidatedFrame` has obtained its immutable command
proof outside timing; never compare that repeated-decode-only profile to LZ4's one-shot
safe decode without reporting the one-time validation cost. Reference blocks already carry
size/RAW metadata. Zstd decode context setup is per call, not hidden in a global cache.
Multithreaded calls include scoped-thread creation/join; there is no persistent thread pool
in this version.

Inputs/buffers are reused (warm-cache workload). Two warmups are excluded. Each of five
default samples runs for at least 100 ms, recording mean ns/operation within that sample;
CSV reports the median/min/max of those samples. `--samples-csv FILE` retains every timed
sample (sorted by duration) with the same profile labels. These are not per-request p95/p99 latencies.
GB/s means decimal uncompressed bytes/second for both compression and decompression.
Ratio means `encoded_bytes / raw_bytes`; lower is smaller. Correctness uses full byte
comparison before timing and after decode timing, not just equal lengths/checksums.

## Run

```sh
# macOS: brew install lz4 zstd
# Ubuntu: sudo apt-get install liblz4-dev libzstd-dev
cargo run --release --features bench --bin vxg-bench -- \
  --threads 1 --block-size 65536 --size-mib 16 --samples 7 --seconds 0.2 \
  --csv reports/cpu-1-thread.csv --samples-csv reports/cpu-1-thread-samples.csv

cargo run --release --features bench --bin vxg-bench -- \
  --threads 4 --size-mib 64 --csv reports/cpu-4-thread.csv

# Real workload; repeated --file accepts multiple files without concatenating them:
cargo run --release --features bench --bin vxg-bench -- \
  --file corpus/logs.ndjson --file corpus/database-pages.bin \
  --csv reports/real-corpus.csv

# Explicit ISA ablation; unsupported ISA returns an error:
cargo run --release --features bench --bin vxg-bench -- --simd scalar --csv reports/scalar.csv
cargo run --release --features bench --bin vxg-bench -- --simd avx512 --csv reports/avx512.csv
```

## Frozen Q95 qualification

The long-running 4 / 95 / 15 target is evaluated on a frozen, synthetic structured
JSON/log set: existing `json-records` plus three held-out families at both 16 and 64 MiB.
The generated corpus is ignored, but [the tracked manifest](../reports/qualification/manifest.json)
records its workload rule, byte count and SHA-256. Regenerate and verify it with:

```sh
python3 scripts/generate_qualification_corpus.py
python3 scripts/generate_qualification_corpus.py --verify
scripts/run_qualification.sh reports/qualification/runs/P0-baseline
```

A second frozen revision, Q95-r2, applies the 2026-09-26 scope revision
([E_B_FRONTIER_REVIEW.md](E_B_FRONTIER_REVIEW.md), [TARGET_4_006_12_15.md](TARGET_4_006_12_15.md)):
only the `event-mix-identifiers` and `bursty-variable-records` value sources change
(epoch+counter identifiers; bounded phrase-vocabulary messages). The other two families
regenerate byte-identical to r1, which remains as an adversarial control set.

```sh
python3 scripts/generate_qualification_corpus.py --r2
python3 scripts/generate_qualification_corpus.py --r2 --verify
MANIFEST=reports/qualification/manifest-r2.json scripts/run_qualification.sh OUTPUT_DIRECTORY
```

The runner is serial: it builds one benchmark binary, records its hash and host details,
then preserves an aggregate CSV, all timed samples and a console log per input/process.
It is not a claim that synthetic Q95 represents arbitrary production data. A candidate
promotion requires paired shuffled process order, confidence intervals and every frozen
member; the initial P0 runner records the baseline and does not discard a slow process.

Before changing the wire format, inspect the actual emitted commands—not only encoder
counters—with the offline analyzer. It compresses once with the stated options, parses the
resulting frame, and writes no timed benchmark rows:

```sh
cargo run --release --features bench --bin vxg-analyze -- \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/qualification/experiments/P2-command-analysis/json-records-16MiB.json
```

The report includes exact command histograms, grouped residual widths, cache-hit models and
dependency counts. Its cache model is diagnostic: it does not establish a compressed format
until an exact pack/unpack prototype includes selectors, bases, exceptions and padding.

P3 provides those lossless scalar prototypes without changing the encoder or public frame
format. It captures every logical command field, including pattern seeds, and requires a
pack/unpack roundtrip before writing a report. The resulting rates are metadata-only
commands/s and metadata-bytes/s—not complete-codec GB/s—and its prototype totals are labelled
as a model rather than a VXG v3 bitstream:

```sh
cargo run --release --features bench --bin vxg-pack-eval -- \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/qualification/experiments/P3-packing-r2/json-records-16MiB.json \
  --iterations 1 --threads 1 --block-size 65536
```

It evaluates fixed-width with exceptions, exact bitsliced transpose, lossless int4 escapes,
2-bit/base-3 ternary selectors, and standalone exact distance alternatives. The distance rows
are a substream experiment only; their size cannot be substituted for a full-frame ratio.

P5's bounded entropy experiment is also an offline oracle. It writes canonical per-LZ-block
Huffman tables and an MRU-8 distance tag stream, then proves an exact unpack before reporting
its bytes. The adaptive form builds both entropy and fixed-128 streams, charges both builders,
and adds a one-byte selector per LZ block; it is not a `compress` format or a codec GB/s claim:

```sh
cargo run --release --features bench --bin vxg-pack-eval -- \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/qualification/experiments/P5-adaptive-entropy-r1/json-records-16MiB.json \
  --iterations 3 --threads 1 --block-size 65536
```

The full Q95 reports show an adaptive JSON-only improvement (0.113659 on JSON-16) but no
held-out ratio improvement beyond fixed-128's selector-byte cost. It is retained as a costed
negative result; `docs/TARGET_4_95_15_PLAN.md` defines the next literal and matcher gates.

The same evaluator also contains the P5 literal oracle. It extracts validated v2 literal streams,
builds raw and canonical byte-Huffman payloads for every LZ block and charges a selector, tables,
bit lengths and padding. Its total retains existing v2 command/prefix bytes and replaces only
the literal payload. All-Q95 evidence in `P5-literal-entropy-r1` rejects it as a wire candidate:
the compressed bursty/event literal payloads alone remain above the 5% artifact cap. Its scalar
literal bytes/s rate is not a VXG encode/decode GB/s result.

Built-in synthetic classes: zero runs, period-37, changing JSON records, pseudo-random,
low-alphabet pseudo-random, and 4 KiB mixed segments. Their results are not representative
of every corpus. Supply existing compressed/media/encrypted files to test RAW decisions.
No external corpus is bundled or downloaded automatically.

## GPU comparison profile

```sh
cargo run --release --features bench,metal --bin vxg-bench -- \
  --gpu metal --size-mib 64 --csv reports/metal-e2e.csv
# On the corresponding Linux hardware, substitute bench,cuda / cuda
# or bench,vulkan / vulkan.
```

GPU rows are `gpu_host_e2e_allocating`: CPU validation, framing, copies, device transfer,
allocation/resizing, kernel, synchronization and output assembly are included. Context
and pipeline creation happen before timing. CUDA/Vulkan discrete-device transfer and Metal
shared-buffer copies are not silently omitted. A device-timestamp diagnostic is printed
separately, not as a median or the comparison denominator. There is not yet a device-resident
public benchmark API or preallocated GPU decode API. CPU and GPU profiles therefore differ;
report both with their labels, do not claim pure kernel speed from their ratio.

Correctness covers GPU-encode/CPU-decode, CPU-encode/GPU-decode, and GPU self-decode. The
strict host command validator stays enabled for GPU decompression; do not remove it solely
to inflate results on untrusted input. CRC is off for comparisons but on in normal CLI use.

## Before publishing a win

Record CPU/GPU model, OS, compiler flags, library versions, power/thermal state, core
assignment, frequency policy, device/driver, input checksums and bytes. Repeat on cold/large
working sets as well as warm ones. Measure process and thread-pool variants separately.
Measure actual bandwidth and allocation, not vendor peak bandwidth. Prefer cycles/byte,
branch misses, memory traffic, kernel occupancy and end-to-end latency alongside GB/s.
Zstd ratio is allowed to be better; do not discard those rows. No cherry-picked average over
only compressible synthetic inputs, no single-core-vs-many-core headline, and no CPU fallback
labeled as a GPU result.

## Compressed-domain query: synthetic vs. real-world

Single thread, Apple aarch64, checksums off, warm cache. Every row is verified byte-identical
(row index and bytes) against a raw scan before timing is printed
(`vxg-query-real`, `vxg-query-bench`).

| Dataset | Block type | Operation | Raw scan | VXG query | Notes |
| :--- | :--- | :--- | ---: | ---: | :--- |
| Synthetic NDJSON, 400k rows (62 MiB) | `GRAMMAR_PACKET` | `find_first` / `filter_rows` | 44–49 ms | 1.1–2.3 ms | Best case: templated, dictionary-friendly rows. |
| `/var/log/install.log`, 231k lines (30 MiB) | `FAST_LZ_PACKET` (free text) | `filter_rows`, 6 needles | 43–81 ms | 24–35 ms | 1.4x–3.1x; bounded by decode speed (3.6 GB/s on this file). |

More real files (same method, all results verified identical to a raw scan):

| File | Size | VXG ratio | Needles (hits) | Speedup vs naive raw scan |
| :--- | ---: | ---: | :--- | ---: |
| `icons.json` (pretty JSON, `@iconify-json/simple-icons`) | 4.6 MiB | 42.3% | 1, 0, 0 hits | 1.8x–2.4x |
| `prof_out.txt` (compiler debug log) | 4.2 MiB | 8.3% | 206, 0, 1133 hits | 2.6x–3.5x |
| `prune_list.txt` (path list) | 2.3 MiB | 5.6% | 18395, 0, 1536 hits | 1.1x (almost every line matches), 3.3x, 5.5x |

When nearly every line matches, there is little to skip and the gain drops to about 1.1x.

Read the second row of the first table as the realistic one. Free-form text does not hit the grammar fast path,
so VXG must decode each block; the gain comes from one SIMD needle scan per block instead of
per-line work and from reusing the decode buffer. Plain `grep -c` (BSD) on the same file took
50 ms for a miss and 260 ms for `error`. The raw-scan column above uses a naive `windows()`
baseline, so treat the raw numbers as a weak reference, not a tuned memmem.

The earlier 20–40x numbers apply only to structured, templated data.

## Bigram pre-filter: upper-bound study

`vxg-bigram-study` counts, per 64 KiB block, how many blocks the 256-bit alphabet filter
rejects and how many *additional* blocks an exact bigram set would reject (needle absent,
but every needle byte present). Exact bigram sets are an upper bound: a real compressed-domain
filter must stay conservative at piece boundaries and cannot see adjacency inside
non-dictionary columns. A stored hashed signature would add false positives and per-block bytes.

| File (blocks) | Needle | Truly present | Alphabet rejects | Extra bigram rejects |
| :--- | :--- | ---: | ---: | ---: |
| `install.log` (484) | `ZZZ_NOPE` | 0 | 397 | 87 |
| `install.log` (484) | `Installer error` | 0 | 0 | 66 |
| `install.log` (484) | `pkg.com` | 0 | 6 | 370 |
| `icons.json` (73) | `"gitub"` | 0 | 1 | 71 |
| `icons.json` (73) | `hidden":true` | 0 | 0 | 73 |
| `prof_out.txt` (67) | `schedule-nll` | 0 | 0 | 38 |

Takeaway: for needles made of common characters that are not adjacent in the data, the alphabet
filter rejects almost nothing while bigrams could reject most blocks. For needles with a rare
byte, the alphabet filter already does the job. Free-text blocks (`FAST_LZ_PACKET`) expose no
bigram information without decoding, so using this there needs a stored signature, i.e. a
wire-format change.

## Stage 1 baseline on real files (single thread, aarch64 NEON, checksums off)

Ratio is encoded/raw (lower is better). Speeds in GB/s. LZ4 is `accel=1`, Zstd is level 1.

| File | Codec | Ratio | Encode | Decode |
| :--- | :--- | ---: | ---: | ---: |
| `install.log` (30 MiB log) | VXG | 0.095 | 1.41 | 3.76 |
| | LZ4 | 0.125 | 2.07 | 7.19 |
| | Zstd-1 | 0.088 | 1.44 | 3.28 |
| `prof_out.txt` (4 MiB debug log) | VXG | 0.083 | 1.56 | 4.12 |
| | LZ4 | 0.114 | 2.17 | 7.69 |
| | Zstd-1 | 0.069 | 1.62 | 3.51 |
| `prune_list.txt` (2 MiB paths) | VXG | 0.056 | 2.13 | 4.50 |
| | LZ4 | 0.082 | 3.47 | 11.94 |
| | Zstd-1 | 0.047 | 2.03 | 5.15 |
| `icons.json` (4.6 MiB JSON) | VXG | 0.423 | 0.38 | 1.57 |
| | LZ4 | 0.659 | 0.37 | 2.54 |
| | Zstd-1 | 0.415 | 0.48 | 1.23 |
| `json-16m.ndjson` (synthetic: one identical line repeated 110,376 times, NOT real-world) | VXG | 0.006 | 1.75 | 22.96 |
| | LZ4 | 0.007 | 15.83 | 24.17 |
| | Zstd-1 | 0.003 | 9.52 | 15.79 |

What this says about where we stand:

- On free-text logs and path lists VXG sits in Zstd-1's class: similar encode speed, a little
  faster decode, a slightly worse ratio. LZ4 decodes about 1.9x-2.7x faster and encodes
  about 1.4x-1.6x faster, with a worse ratio.
- `json-16m.ndjson` is a degenerate synthetic file (one line repeated 110,376 times). LZ4 and
  Zstd win its encode at 9.5-15.8 GB/s via long matches while VXG scans every field of every
  line (1.75 GB/s; about 76% of encode time is in `match_single_json_clean_stream`). Do not
  read this row as a real-world gap; a varied-field NDJSON check follows below.
- These are single runs of 3 samples on one machine; treat differences under about 10% as noise.

### Varied-field NDJSON (synthetic, every record differs)

Generated by `scripts/gen_varied_ndjson.py` (300,000 records, 54 MiB: random 64-bit request
ids, skewed enums, lognormal latency). Single thread, checksums off.

| Codec | Ratio | Encode GB/s | Decode GB/s |
| :--- | ---: | ---: | ---: |
| VXG | 0.096 | 0.69 | 3.19 |
| LZ4 | 0.290 | 1.07 | 4.04 |
| Zstd-1 | 0.166 | 0.73 | 1.61 |

VXG's ratio is 42% smaller than Zstd-1's here and decode is 2.0x faster; encode is within noise
of Zstd-1 and 35% slower than LZ4, decode 21% slower than LZ4.

Encode CPU profile (macOS `sample`, top-of-stack, 4120 samples):

| Function | Share |
| :--- | ---: |
| `pack_dictionary_model_mode` | 28.7% |
| `match_single_json_qmask_stream` (template matcher) | 24.6% |
| `pack_model_mode` | 11.4% |
| `huf_encode` | 6.1% |
| `pack_fast_lz_core` | 5.6% |

These are the next stage-1 targets. No optimization has been made from this profile yet.

### Dictionary attempts on high-cardinality columns (measured, varied NDJSON)

Temporary counters around `pack_dictionary_model_mode` (reverted, not in the tree), grouped by the
byte length of a column's first value, 3 samples:

| First value length (typical column) | Calls | Produced a dictionary | Time in function |
| :--- | ---: | ---: | ---: |
| 18 (`req_id`, random hex) | 2000 | 2 | 9.2 ms (15%) |
| 9 (mixed: `user`, `path`) | 2319 | 1234 | 19.3 ms (31%) |
| 3, 6, 11, 14 (enums, service, region) | 1263-2080 | ~all | 3.7-7.4 ms each |

Total time in the function was 62.5 ms, and it is about 29% of encode time (profile above).
So even if every failed attempt were free, the gain is bounded by roughly 9% of encode time.
The existing early exit is an exact lower bound (it never rejects a dictionary that would win),
which only fires near the end of a high-cardinality column. A heuristic early exit (for example
"all of the first 64 values distinct") could cut that cost but can miss columns whose repeats
come late, so it needs a ratio check on the corpus before use. Not implemented.

### Where the encode time goes (derived from the profile above, varied NDJSON)

These are calculations from the sampled shares, not separate measurements. Encode takes about
78 ms for the 54 MiB file; the template matcher's 24.6% is about 19 ms over 300,000 rows, so
roughly 64 ns per row of about 181 bytes, i.e. about 2.8 GB/s for 10 fields per row. That is
already fast, so a 2x faster matcher would only save about 12% of encode time. Dictionary
attempts (28.7%) and model selection (`pack_model_mode`, 11.4%) together are about 40%, spread
over many columns and blocks. No single function offers a 2x encode gain; the remaining gains
are many items of roughly 5-12% each. Multi-thread encode is the larger lever in practice (8
threads reached 30.9 GB/s on the JSON profile file).

### Column-style filters on varied NDJSON (existing `filter_rows`, no new code)

300,000 records, single thread. Each result is verified byte-identical to a raw scan. The raw
baseline is a naive scan, so read speedups as relative to a weak reference. Hits are
materialized, which is part of the VXG time.

| Needle | Column kind | Hits | Raw ms | VXG ms | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `"level":"ERROR"` | dictionary | 33,348 | 123.4 | 18.8 | 6.6x |
| `"service":"fraud-engine"` | dictionary | 37,539 | 91.3 | 22.6 | 4.0x |
| `"path":"/v1/login"` | dictionary | 42,929 | 96.8 | 25.2 | 3.8x |
| `"region":"us-west-2"` | dictionary | 75,148 | 88.2 | 31.4 | 2.8x |
| `"status":500` | numeric | 14,937 | 133.8 | 84.0 | 1.6x |
| `"level":"NOPE"` | dictionary, value absent | 0 | 133.6 | 1.5 | 91x |

`frame_row_count` on this file takes 0.585 ms.

Reading: equality on dictionary columns is already pushed down well, and an absent value is
rejected almost for free. The weak spot is numeric columns (1.6x), where the current path
emits and verifies candidate rows. A numeric-aware predicate (decode only that column, compare
integers, emit only matching rows) is the part of the stage 2 first slice that would add
something new; dictionary equality is already covered.
