## Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Rows): `VXG v0.5.3` vs `DuckDB + Parquet`

- **Engine versions:** `VXG v0.5.3` (`mmap` zero-copy C-FFI + `--record-aligned` + `--stream`) vs `DuckDB 2.0.0` (`PREPARE / EXECUTE` calibrated with `SELECT 1`, testing both default `ROW_GROUP_SIZE=122880` and fine-grained `ROW_GROUP_SIZE=16384`).
- **Verification:** Every query asserts 100% exact result equality between `VXG` and `DuckDB`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.
- **Key distinction:** `VXG` stores the original `.csv` / `.ndjson` **100% losslessly byte-for-byte** (`1` step, zero schema ETL), whereas `Parquet` requires an upfront lossy-to-formatting schema conversion (`COPY ... TO PARQUET`).

### `nyc-taxi-2024-10m.ndjson` (`10,000,000` rows, `2.62 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest/ETL Time:** `VXG` **`152.6 MB`** (`ratio=0.0583`, **`2.65 s`** / `0.99 GB/s`, lossless=`YES`) · `VXG --record-aligned` `140.8 MB` (`ratio=0.0538`, `0.61 s`, lossless=`YES`) · `Parquet-Snappy` `193.3 MB` (`ratio=0.0739`, `0.77 s` ETL, lossless=`NO`) · `Parquet-Snappy (RG=16384)` `203.4 MB` (`ratio=0.0777`, `1.05 s` ETL) · `Parquet-Zstd` `144.7 MB` (`ratio=0.0553`, `0.95 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.09 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 0.36 s)* | 21.87 ms (457.3 M/s) | **3.93 ms (2543.7 M/s)** | 8.83 ms (1132.2 M/s) | 23.17 ms (431.6 M/s) | 12.08 ms (827.9 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 10.14 ms (985.8 M/s) | **1.31 ms (7642.3 M/s)** | 6.91 ms (1446.2 M/s) | 15.15 ms (660.2 M/s) | 8.97 ms (1115.0 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 12.60 ms (793.9 M/s) | **2.09 ms (4778.9 M/s)** | 6.94 ms (1440.4 M/s) | 16.19 ms (617.7 M/s) | 9.37 ms (1067.6 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 17.14 ms (583.6 M/s) | **4.31 ms (2319.8 M/s)** | 8.83 ms (1132.1 M/s) | 18.86 ms (530.2 M/s) | 10.06 ms (994.2 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 35.06 ms (285.2 M/s) | **1.63 ms (6152.6 M/s)** | 8.79 ms (1137.6 M/s) | 20.79 ms (481.0 M/s) | 11.78 ms (848.9 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `135.37 ms` (`O(single_frame)` RAM).

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.04 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 127.03 ms (78.7 M/s) | **29.18 ms (342.7 M/s)** | 40.07 ms (249.6 M/s) | 65.78 ms (152.0 M/s) | 68.72 ms (145.5 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 70.45 ms (141.9 M/s) | **11.20 ms (892.7 M/s)** | 37.80 ms (264.6 M/s) | 53.66 ms (186.4 M/s) | 52.23 ms (191.5 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 91.13 ms (109.7 M/s) | **15.93 ms (627.9 M/s)** | 35.05 ms (285.3 M/s) | 50.69 ms (197.3 M/s) | 49.93 ms (200.3 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 108.19 ms (92.4 M/s) | **31.06 ms (321.9 M/s)** | 43.67 ms (229.0 M/s) | 65.46 ms (152.8 M/s) | 59.83 ms (167.1 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 84.93 ms (117.8 M/s) | **8.10 ms (1235.1 M/s)** | 16.20 ms (617.4 M/s) | 22.68 ms (441.0 M/s) | 26.17 ms (382.2 M/s) |

### `nyc-taxi-2024-40m.csv` (`40,000,000` rows, `3.06 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest/ETL Time:** `VXG` **`552.9 MB`** (`ratio=0.1805`, **`5.13 s`** / `0.60 GB/s`, lossless=`YES`) · `VXG --record-aligned` `550.4 MB` (`ratio=0.1797`, `2.93 s`, lossless=`YES`) · `Parquet-Snappy` `773.1 MB` (`ratio=0.2524`, `4.87 s` ETL, lossless=`NO`) · `Parquet-Snappy (RG=16384)` `818.8 MB` (`ratio=0.2673`, `4.74 s` ETL) · `Parquet-Zstd` `579.4 MB` (`ratio=0.1891`, `3.48 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.06 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 1.01 s)* | **23.54 ms (1699.0 M/s)** | 25.04 ms (1597.2 M/s) | 34.32 ms (1165.5 M/s) | 98.53 ms (406.0 M/s) | 43.63 ms (916.9 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 15.02 ms (2662.9 M/s) | **13.10 ms (3053.9 M/s)** | 25.85 ms (1547.2 M/s) | 73.95 ms (540.9 M/s) | 36.20 ms (1104.9 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 17.90 ms (2234.9 M/s) | **16.67 ms (2400.1 M/s)** | 35.04 ms (1141.5 M/s) | 64.17 ms (623.4 M/s) | 32.22 ms (1241.4 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 37.30 ms (1072.4 M/s) | **23.25 ms (1720.4 M/s)** | 32.85 ms (1217.5 M/s) | 76.95 ms (519.8 M/s) | 40.84 ms (979.5 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 20.57 ms (1944.7 M/s) | **10.77 ms (3715.6 M/s)** | 20.47 ms (1953.9 M/s) | 78.73 ms (508.0 M/s) | 24.20 ms (1652.6 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `313.51 ms` (`O(single_frame)` RAM).

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.05 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 159.37 ms (251.0 M/s) | **150.02 ms (266.6 M/s)** | 159.89 ms (250.2 M/s) | 280.70 ms (142.5 M/s) | 274.23 ms (145.9 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 90.08 ms (444.1 M/s) | **80.52 ms (496.8 M/s)** | 148.07 ms (270.1 M/s) | 214.38 ms (186.6 M/s) | 204.03 ms (196.0 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 114.03 ms (350.8 M/s) | **106.53 ms (375.5 M/s)** | 133.96 ms (298.6 M/s) | 198.04 ms (202.0 M/s) | 192.40 ms (207.9 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 174.01 ms (229.9 M/s) | **156.39 ms (255.8 M/s)** | 172.73 ms (231.6 M/s) | 258.14 ms (155.0 M/s) | 233.14 ms (171.6 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 87.94 ms (454.9 M/s) | 61.76 ms (647.7 M/s) | **29.80 ms (1342.3 M/s)** | 84.93 ms (470.9 M/s) | 45.11 ms (886.8 M/s) |

