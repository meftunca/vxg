## Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Rows): `VXG v0.5.0` vs `DuckDB + Parquet`

- **Engine versions:** `VXG v0.5.0` (`mmap` zero-copy C-FFI + `--record-aligned` + `--stream`) vs `DuckDB 2.0.0` (`PREPARE / EXECUTE` calibrated with `SELECT 1`).
- **Verification:** Every query asserts 100% exact result equality between `VXG` and `DuckDB`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.
- **Key distinction:** `VXG` stores the original `.csv` / `.ndjson` **100% losslessly byte-for-byte** (`1` step, zero schema ETL), whereas `Parquet` requires an upfront lossy-to-formatting schema conversion (`COPY ... TO PARQUET`).

### `nyc-taxi-2024-10m.ndjson` (`10,000,000` rows, `2.62 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest/ETL Time:** `VXG` **`152.7 MB`** (`ratio=0.0584`, **`2.11 s`** / `1.24 GB/s`, lossless=`YES`) · `VXG --record-aligned` `145.9 MB` (`1.02 s`) · `Parquet-Snappy` `193.3 MB` (`ratio=0.0739`, `0.93 s` ETL, lossless=`NO`) · `Parquet-Zstd` `144.6 MB` (`ratio=0.0553`, `1.18 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.09 ms`)

| Query | `VXG` (`ms` / `M rows/s`) | `VXG --record-aligned` (`ms`) | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 0.42 s)* | 43.31 ms (230.9 M/s) | 32.12 ms | **9.36 ms (1068.8 M/s)** | 12.43 ms (804.5 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 14.96 ms (668.5 M/s) | 8.26 ms | **8.20 ms (1219.9 M/s)** | 9.69 ms (1031.5 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 21.08 ms (474.3 M/s) | 13.06 ms | **8.02 ms (1246.6 M/s)** | 10.16 ms (984.4 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 25.93 ms (385.6 M/s) | 16.81 ms | **13.20 ms (757.5 M/s)** | 13.62 ms (734.2 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 50.66 ms (197.4 M/s) | 22.92 ms | **9.09 ms (1100.3 M/s)** | 11.68 ms (856.3 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `318.49 ms` (`O(single_frame)` RAM).

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.05 ms`)

| Query | `VXG` (`ms` / `M rows/s`) | `VXG --record-aligned` (`ms`) | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 230.18 ms (43.4 M/s) | 176.40 ms | **40.25 ms (248.5 M/s)** | 68.79 ms (145.4 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 85.74 ms (116.6 M/s) | 42.71 ms | **37.76 ms (264.8 M/s)** | 52.97 ms (188.8 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 119.71 ms (83.5 M/s) | 76.78 ms | **36.03 ms (277.6 M/s)** | 52.11 ms (191.9 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 155.20 ms (64.4 M/s) | 98.73 ms | **43.91 ms (227.7 M/s)** | 59.87 ms (167.0 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 149.12 ms (67.1 M/s) | 106.60 ms | **15.45 ms (647.2 M/s)** | 26.27 ms (380.6 M/s) |

### `nyc-taxi-2024-40m.csv` (`40,000,000` rows, `3.06 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest/ETL Time:** `VXG` **`554.4 MB`** (`ratio=0.1810`, **`4.41 s`** / `0.70 GB/s`, lossless=`YES`) · `VXG --record-aligned` `551.6 MB` (`1.56 s`) · `Parquet-Snappy` `773.2 MB` (`ratio=0.2524`, `4.06 s` ETL, lossless=`NO`) · `Parquet-Zstd` `579.4 MB` (`ratio=0.1891`, `3.80 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.09 ms`)

| Query | `VXG` (`ms` / `M rows/s`) | `VXG --record-aligned` (`ms`) | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 1.26 s)* | **27.71 ms (1443.4 M/s)** | 29.72 ms | 34.55 ms (1157.7 M/s) | 57.44 ms (696.4 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 19.19 ms (2084.9 M/s) | **18.41 ms** | 31.33 ms (1276.6 M/s) | 48.85 ms (818.8 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 21.12 ms (1893.9 M/s) | **18.59 ms** | 31.58 ms (1266.5 M/s) | 36.91 ms (1083.6 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 48.84 ms (819.0 M/s) | **44.39 ms** | 49.14 ms (814.0 M/s) | 48.98 ms (816.6 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 24.72 ms (1618.2 M/s) | **17.53 ms** | 18.39 ms (2175.4 M/s) | 22.74 ms (1759.1 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `995.76 ms` (`O(single_frame)` RAM).

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.05 ms`)

| Query | `VXG` (`ms` / `M rows/s`) | `VXG --record-aligned` (`ms`) | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 161.22 ms (248.1 M/s) | **148.84 ms** | 159.20 ms (251.3 M/s) | 283.61 ms (141.0 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 105.64 ms (378.6 M/s) | **98.45 ms** | 162.46 ms (246.2 M/s) | 214.79 ms (186.2 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 117.28 ms (341.1 M/s) | **111.60 ms** | 144.21 ms (277.4 M/s) | 197.55 ms (202.5 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 269.16 ms (148.6 M/s) | 250.25 ms | **173.94 ms (230.0 M/s)** | 237.69 ms (168.3 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 95.41 ms (419.3 M/s) | 87.97 ms | **29.19 ms (1370.4 M/s)** | 43.97 ms (909.6 M/s) |

