# Chimera Envelope Benchmarks (`VXG v4` vs Best-of-Competitor Extremes)

Standard codec benchmarks compare `VXG v4` against individual competitor presets such as `LZ4 (accel=1)` or `Zstd (level=1)`. A natural objection is: *"What if a user configures `LZ4-HC (L9)` or `Zstd (L9)` for ratio, or `LZ4 (accel=4)` / `Zstd (-3)` for speed?"*

To eliminate preset trade-offs completely, `vxg-bench` evaluates **7 native C baseline configurations** per dataset (`liblz4 1.10.0`: `lz4-c-a1`, `lz4-c-a4`, `lz4hc-c-l9`; `libzstd 1.5.7`: `zstd-c-l-3`, `zstd-c-l1`, `zstd-c-l3`, `zstd-c-l9`) and synthesizes three **impossible "Chimera Envelope" baselines** on every dataset simultaneously:

- **`LZ4 Chimera` (`CHIMERA-LZ4`):**
  $$\text{Ratio} = \min_{c \in \{\text{a1}, \text{a4}, \text{hc-l9}\}} R_c, \quad \text{Enc} = \max_c E_c, \quad \text{Dec} = \max_c D_c, \quad J = \min_c J_c$$
  Grants LZ4 its maximum-effort `LZ4-HC L9` compression ratio (`~0.10–0.58 GB/s` real encode) while simultaneously crediting it with `LZ4-fast` (`a1`/`a4`) encode speed (`6.5–8.3 GB/s` on `8T`) and peak decode speed (`34+ GB/s`).
- **`Zstd Chimera` (`CHIMERA-ZSTD`):**
  $$\text{Ratio} = \min_{c \in \{-3, 1, 3, 9\}} R_c, \quad \text{Enc} = \max_c E_c, \quad \text{Dec} = \max_c D_c, \quad J = \min_c J_c$$
  Grants Zstd its `Zstd L9` high-compression ratio while simultaneously crediting it with ultra-fast `Zstd -3` encode and decode speeds.
- **`All-Rivals Chimera` (`CHIMERA-ALL`):**
  $$\text{Ratio} = \min_{\text{all LZ4} \cup \text{Zstd}} R_c, \quad \text{Enc} = \max_{\text{all}} E_c, \quad \text{Dec} = \max_{\text{all}} D_c, \quad J = \min_{\text{all}} J_c$$
  Combines **`Zstd L9` compression ratio** with **`LZ4-fast` encode/decode speeds and `LZ4` minimum CPU energy** in a single hypothetical codec.

---

## 1. 10-Dataset `v4` Suite (`16 MiB` per dataset, `64 KiB` independent blocks, `8` Threads)

Full raw artifacts: [`reports/v4-chimera-8t.md`](../reports/v4-chimera-8t.md) and [`reports/v4-chimera-8t.csv`](../reports/v4-chimera-8t.csv) (`1T` in [`reports/v4-chimera-1t.md`](../reports/v4-chimera-1t.md)).

| Dataset (`16 MiB`, `8T`) | **VXG v4 Ratio** | `LZ4 Chimera` Ratio (`L9`) | `Zstd Chimera` Ratio (`L9`) | **VXG Enc / Dec (`GB/s`)** | `LZ4 Chimera` Enc / Dec | `Zstd Chimera` Enc / Dec | **VXG `VPI`** | `LZ4 Chimera` `VPI` | `Zstd Chimera` `VPI` | `All-Rivals Chimera` `VPI` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | **0.0407** | 0.1654 | 0.0797 | **9.142 / 40.042** | 7.997 / 34.356 | 5.451 / 11.643 | **469.7** | 100.2 (`4.69x`) | 99.9 (`4.70x`) | 207.9 (**`2.26x`**) |
| **`events.tsv`** | **0.0326** | 0.1515 | 0.0683 | **8.548 / 42.330** | 2.864 / 33.958 | 5.610 / 10.162 | **583.0** | 65.1 (`8.95x`) | 110.6 (`5.27x`) | 202.1 (**`2.88x`**) |
| **`app-logs.logfmt`** | **0.0353** | 0.1623 | 0.0695 | **1.517 / 10.329** | 3.687 / 8.966 | 2.507 / 6.304 | **112.1** | 35.4 (`3.17x`) | 57.2 (`1.96x`) | 82.7 (**`1.36x`**) |
| **`json-records.ndjson`** | **0.0227** | 0.1147 | 0.0488 | **4.672 / 11.489** | 4.789 / 16.851 | 3.290 / 9.970 | **323.3** | 78.3 (`4.13x`) | 117.4 (`2.75x`) | 184.2 (**`1.76x`**) |
| **`field-order-optional.ndjson`** | **0.0445** | 0.1492 | 0.0883 | **3.610 / 10.604** | 3.186 / 18.029 | 2.659 / 5.671 | **139.1** | 50.8 (`2.74x`) | 44.0 (`3.16x`) | 85.8 (**`1.62x`**) |
| **`event-mix-identifiers.ndjson`** | **0.0248** | 0.1134 | 0.0698 | **8.094 / 43.522** | 8.325 / 34.379 | 4.903 / 11.220 | **755.9** | 149.2 (`5.07x`) | 106.3 (`7.11x`) | 242.4 (**`3.12x`**) |
| **`bursty-variable-records.ndjson`** | **0.0480** | 0.1308 | 0.0926 | **4.947 / 17.789** | 6.808 / 20.824 | 4.024 / 8.150 | **195.4** | 91.0 (`2.15x`) | 61.8 (`3.16x`) | 128.6 (**`1.52x`**) |
| **`source-and-docs.txt`** | **0.3059** | 0.3399 | 0.2670 | **2.442 / 8.860** | 3.171 / 18.771 | 2.109 / 5.966 | **15.2** | 22.7 (`0.67x`) | 13.3 (`1.14x`) | 28.9 (`0.53x`) |
| **`ui-render-32bpp.bmp`** | **0.0048** | 0.0140 | 0.0086 | **38.377 / 54.064** | 25.008 / 52.348 | 25.359 / 20.657 | **9518.9** | 2579.5 (`3.69x`) | 2653.5 (`3.59x`) | 4224.1 (**`2.25x`**) |
| **`rgb-frame-24bpp.bmp`** | **0.0012** | 0.0227 | 0.0119 | **40.563 / 74.369** | 25.150 / 37.977 | 19.001 / 23.382 | **44254.7** | 1362.8 (`32.47x`) | 1765.9 (`25.06x`) | 2589.2 (**`17.09x`**) |
| **10-Dataset Geometric Mean** | **0.0252** | 0.1011 | 0.0557 | **7.052 / 23.821** | 6.551 / 24.711 | 5.163 / 10.106 | **514.8** | 125.8 (**`4.09x`**) | 129.6 (**`3.97x`**) | 236.2 (**`2.18x`**) |

### Key Takeaways (`v4` 16 MiB Suite)
1. **Simultaneous 3-Axis Dominance (`Ratio` + `Encode` + `Decode`):** On `telemetry.csv`, `events.tsv`, `ui-render-32bpp.bmp`, and `rgb-frame-24bpp.bmp`, a single `VXG v4` run beats **`All-Rivals Chimera` (`Zstd L9` ratio + `LZ4-fast` speed)** on **all three axes simultaneously** (`2.0x–9.9x` smaller than `Zstd L9`, faster encode than `LZ4-fast`, and faster decode than `LZ4-fast`).
2. **10-Dataset Geomean:** Across all 10 datasets (including free-form prose/code `source-and-docs.txt`), `VXG v4` is **`4.01x` smaller than `LZ4-HC L9`**, **`2.21x` smaller than `Zstd L9`**, encodes **faster than `All-Rivals Chimera`** (`7.052` vs `6.551 GB/s`), and achieves **`4.09x` higher `VPI` than `LZ4 Chimera`**, **`3.97x` higher than `Zstd Chimera`**, **`2.18x` higher than `All-Rivals Chimera`**, and **`1.36x` higher Green Codec Index (`GCI = 468.4` vs `345.2`)**.
3. **1-Thread Geomean ([`reports/v4-chimera-1t.md`](../reports/v4-chimera-1t.md)):** `VXG v4` achieves **`3.06x` higher `VPI` than `LZ4 Chimera`**, **`3.33x` higher than `Zstd Chimera`**, **`1.66x` higher than `All-Rivals Chimera`**, and beats `All-Rivals Chimera` on both `BCI` (`6.55` vs `6.09`) and `GCI` (`93.4` vs `90.2`).

---

## 2. Real Multi-Template & Delimited Corpora (`8` Threads)

Full raw artifacts: [`reports/real-chimera-8t.md`](../reports/real-chimera-8t.md) and [`reports/real-chimera-1t.md`](../reports/real-chimera-1t.md).

| Corpus (`8T`, `64 KiB` blocks) | Size | **VXG v4 Ratio** | `LZ4-HC L9` Ratio | `Zstd L9` Ratio | **VXG Enc / Dec (`GB/s`)** | `LZ4 Chimera` Enc / Dec | `Zstd Chimera` Enc / Dec | **VXG `VPI`** | vs `LZ4 Chimera` | vs `Zstd Chimera` | vs `All-Rivals Chimera` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ndjson_varied.ndjson`** (`300k` rows, multi-tpl) | 56.5 MB | **0.0914** | 0.2180 | 0.1480 | **4.688 / 28.599** | 3.824 / 17.553 | 4.741 / 9.486 | **126.6** | **`3.37x`** | **`2.79x`** | **`2.05x`** |
| **`data.csv`** (`250k` rows, 11 cols, free-text UA) | 30.5 MB | **0.1796** | 0.3759 | 0.2651 | **2.801 / 9.882** | 3.261 / 15.021 | 2.953 / 5.790 | **29.3** | **`1.57x`** | **`1.88x`** | **`1.11x`** |
| **`data.clf`** (Apache Combined Log Format) | 31.2 MB | **0.0980** | 0.2177 | 0.1504 | **3.029 / 10.313** | 3.325 / 15.354 | 2.951 / 6.475 | **57.0** | **`1.74x`** | **`1.96x`** | **`1.20x`** |
| **3-Corpus Geometric Mean** | — | **0.1172** | 0.2613 | 0.1807 | **3.414 / 14.284** | 3.461 / 15.938 | 3.457 / 7.085 | **59.6** | **`2.10x`** | **`2.18x`** | **`1.40x`** |

On `ndjson_varied.ndjson` (`56.5 MB`), `VXG v4` is **`2.39x` smaller than `LZ4-HC L9`** (`9.14%` vs `21.80%`) while simultaneously encoding **`22.6%` faster** and decoding **`62.9%` faster** than `LZ4 Chimera` (`7.24 GB/s` encode / `38.68 GB/s` decode peak on unthrottled P-cores).

---

## Reproducing Locally

```sh
cargo run --release --features bench --bin vxg-bench -- \
  --suite v4 --size-mib 16 --threads 8 --samples 5 --seconds 0.05 \
  --csv reports/v4-chimera-8t.csv --markdown reports/v4-chimera-8t.md
```
