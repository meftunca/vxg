# CPU Energy Efficiency & Composite Index Comparison (`J/GB`, `VPI`, `BCI`, `GCI`)

VXG measures a fourth primary performance axis — **Joules per decimal GB of raw input (`J/GB`)** — alongside **Compression Ratio**, **Encode Throughput (`GB/s`)**, and **Decode Throughput (`GB/s`)**, and combines them into three composite indices:

1. **VXG Performance Index (`VPI` — Throughput-Compression Index, higher is better):**
   $$\text{VPI} = \frac{\sqrt{\text{Encode}_{\text{GB/s}} \cdot \text{Decode}_{\text{GB/s}}}}{\text{Ratio}}$$
2. **Balanced 3-Axis Cube Index (`BCI`, higher is better):**
   $$\text{BCI} = \left(\frac{\text{Encode}_{\text{GB/s}} \cdot \text{Decode}_{\text{GB/s}}}{\text{Ratio}}\right)^{1/3}$$
   Assigns equal $1/3$ geometric weight to Compression Factor ($1/\text{Ratio}$), Encode Speed ($\text{GB/s}$), and Decode Speed ($\text{GB/s}$).
3. **Green Codec Index (`GCI` — Energy-Aware Codec Index, higher is better):**
   $$\text{GCI} = \frac{\text{VPI}}{\sqrt{J_{\text{enc}} \cdot J_{\text{dec}}}} = \frac{\sqrt{\text{Encode}_{\text{GB/s}} \cdot \text{Decode}_{\text{GB/s}}}}{\text{Ratio} \cdot \sqrt{J_{\text{enc}} \cdot J_{\text{dec}}}}$$

---

## 1. Measurement Methodology

VXG supports two complementary hardware energy measurement modes on macOS / Apple Silicon:
1. **Unprivileged Per-Process Kernel Energy Counter (`proc_pid_rusage` `RUSAGE_INFO_V6` `ri_energy_nj`):**
   - Queries the XNU kernel's per-process PMGR/CLPC energy ledger (`ri_energy_nj`, accumulated across all threads of the process) and retired instruction counter (`ri_instructions`) directly at the start and end of each timed sample loop.
   - Requires **zero root privileges (`sudo` not needed)**, has **exact call-boundary resolution**, and isolates the benchmark process from unrelated background processes.
   - Since $1\,\text{nJ} / 1\,\text{byte} = 10^{-9}\,\text{J} / 10^{-9}\,\text{GB} = 1\,\text{J/GB}$, the ratio $\Delta\text{ri\_energy\_nj} / \text{raw\_bytes\_processed}$ yields exact Joules per decimal GB.
2. **Whole-SoC Rail Sampling via `powermetrics` (`scripts/energy_bench.py --authorize`):**
   - Samples system-wide CPU/GPU/ANE power rails via `/usr/bin/powermetrics` with pre/post idle-power subtraction (`cpu_rail_J_per_GB` and `incremental_cpu_J_per_GB`).

---

## 2. 10-Dataset Energy & Composite Index Results (`macos` / `aarch64` NEON, 16 MiB, 64 KiB Blocks)

### 2.1 Single-Thread (`threads = 1`) — Per-Process Energy (`J/GB`) & Green Codec Index (`GCI`)

| Dataset | VXG Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG `GCI` | LZ4 `GCI` | Zstd-1 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1994 | 0.0841 | **2.921** | 1.472 | 4.086 | **0.671** | 0.614 | 2.176 | **98.4** | 18.9 | 19.2 | **70.2** | 19.9 | 6.5 |
| **`events.tsv`** | **0.0334** | 0.2223 | 0.0741 | **2.952** | 1.999 | 3.486 | **0.653** | 0.816 | 1.851 | **119.5** | 13.2 | 25.1 | **86.0** | 10.4 | 9.9 |
| **`app-logs.logfmt`** | **0.0353** | 0.2007 | 0.0695 | **2.683** | 1.453 | 3.384 | **0.590** | 0.610 | 1.851 | **126.6** | 17.3 | 25.5 | **100.6** | 18.4 | 10.2 |
| **`json-records.ndjson`** | **0.0227** | 0.1497 | 0.0542 | **2.108** | 1.152 | 2.615 | **0.466** | 0.507 | 1.433 | **231.8** | 30.5 | 42.2 | **233.9** | 39.9 | 21.8 |
| **`field-order-optional-escaping.ndjson`** | **0.0445** | 0.2132 | 0.1043 | **3.777** | 2.384 | 4.766 | **0.900** | 0.900 | 2.451 | **61.1** | 11.6 | 12.7 | **33.1** | 7.9 | 3.7 |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1545 | 0.0779 | **2.711** | 1.746 | 3.670 | **0.614** | 0.781 | 2.148 | **158.4** | 21.6 | 22.1 | **122.8** | 18.5 | 7.9 |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.2065 | 0.1196 | **3.914** | 2.216 | 5.284 | **1.169** | 1.040 | 2.945 | **47.6** | 11.8 | 10.2 | **22.2** | 7.8 | 2.6 |
| **`source-and-docs.txt`** | **0.2782** | 0.3830 | 0.2755 | **6.887** | 4.638 | 8.527 | **1.718** | 1.158 | 3.222 | **4.4** | 4.4 | 3.4 | **1.3** | 1.9 | 0.6 |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0091 | **0.562** | 0.328 | 0.627 | **0.345** | 0.202 | 0.544 | **2435.3** | 1167.2 | 616.5 | **5530.4** | 4527.0 | 1055.6 |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **0.631** | 0.368 | 0.604 | **0.368** | 0.312 | 0.489 | **9206.1** | 474.0 | 604.8 | **19116.6** | 1398.9 | 1112.9 |
| **10-Dataset Geometric Mean** | **0.0250** | 0.1272 | 0.0607 | **2.333** | 1.382 | 2.868 | **0.665** | 0.619 | 1.641 | **165.9** | 31.5 | 33.7 | **133.2** | 34.0 | 15.5 |

### 2.2 Multi-Thread (`threads = 8`) — Per-Process Energy (`J/GB`) & Green Codec Index (`GCI`)

| Dataset | VXG Enc `J/GB` | LZ4 Enc `J/GB` | Zstd-1 Enc `J/GB` | VXG Dec `J/GB` | LZ4 Dec `J/GB` | Zstd-1 Dec `J/GB` | VXG `VPI` | LZ4 `VPI` | Zstd-1 `VPI` | VXG `GCI` | LZ4 `GCI` | Zstd-1 `GCI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **2.639** | 1.343 | 3.573 | **0.641** | 0.587 | 2.008 | **562.4** | 116.2 | 131.5 | **432.4** | 130.9 | 49.1 |
| **`events.tsv`** | **2.625** | 1.844 | 3.111 | **0.623** | 0.753 | 1.700 | **637.7** | 77.0 | 133.6 | **498.7** | 65.4 | 58.1 |
| **`app-logs.logfmt`** | **2.461** | 1.415 | 3.068 | **0.552** | 0.557 | 1.743 | **675.3** | 89.8 | 138.9 | **579.2** | 101.1 | 60.0 |
| **`json-records.ndjson`** | **2.028** | 1.081 | 2.441 | **0.467** | 0.478 | 1.344 | **1212.0** | 150.1 | 224.3 | **1246.1** | 208.8 | 123.9 |
| **`field-order-optional-escaping.ndjson`** | **3.968** | 2.196 | 4.336 | **0.861** | 0.859 | 2.346 | **310.2** | 63.3 | 66.0 | **167.9** | 46.1 | 20.7 |
| **`event-mix-identifiers.ndjson`** | **2.398** | 1.530 | 3.442 | **0.584** | 0.759 | 2.047 | **872.9** | 119.7 | 144.4 | **737.8** | 111.1 | 54.4 |
| **`bursty-variable-records.ndjson`** | **3.697** | 2.231 | 4.999 | **1.124** | 0.970 | 2.777 | **304.5** | 76.7 | 69.5 | **149.4** | 52.1 | 18.7 |
| **`source-and-docs.txt`** | **6.433** | 4.475 | 7.848 | **1.638** | 1.051 | 2.977 | **30.6** | 29.9 | 24.3 | **9.4** | 13.8 | 5.0 |
| **`ui-render-32bpp.bmp`** | **0.659** | 0.443 | 0.690 | **0.359** | 0.220 | 0.540 | **10153.2** | 4118.6 | 3109.3 | **20873.8** | 13196.8 | 5090.2 |
| **`rgb-frame-24bpp.bmp`** | **0.717** | 0.474 | 0.695 | **0.376** | 0.316 | 0.491 | **38648.9** | 1975.6 | 2944.1 | **74433.0** | 5107.0 | 5038.4 |
| **10-Dataset Geometric Mean** | **2.274** | 1.389 | 2.735 | **0.647** | 0.594 | 1.558 | **883.8** | 167.2 | 195.3 | **728.7** | 184.1 | 94.6 |

---

## 3. Reproducing Locally

```sh
# 1. Full 10-dataset v4 suite with per-process energy (J/GB) and composite indices (VPI, BCI, GCI):
cargo run --release --features bench --bin vxg-bench -- \
  --suite v4 --size-mib 16 --samples 5 --seconds 0.08 --threads 1 \
  --csv reports/local-v4-energy-1t.csv --markdown reports/local-v4-energy-1t.md

# 2. Shuffled multi-trial standalone workload runner (with process_rusage_v6 + optional sudo powermetrics):
cargo build --release --features bench --bin vxg-energy-workload
python3 scripts/energy_bench.py --no-power \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/local-energy-json16 --seconds 1.0 --trials 3
```
