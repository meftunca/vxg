## Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Rows): `VXG v0.5.2` vs `DuckDB + Parquet`

- **Engine versions:** `VXG v0.5.2` (`mmap` zero-copy C-FFI + `--record-aligned` + `--stream`) vs `DuckDB 2.0.0` (`PREPARE / EXECUTE` calibrated with `SELECT 1`, testing both default `ROW_GROUP_SIZE=122880` and fine-grained `ROW_GROUP_SIZE=16384`).
- **Verification:** Every query asserts 100% exact result equality between `VXG` and `DuckDB`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.
- **Key distinction:** `VXG` stores the original `.csv` / `.ndjson` **100% losslessly byte-for-byte** (`1` step, zero schema ETL), whereas `Parquet` requires an upfront lossy-to-formatting schema conversion (`COPY ... TO PARQUET`).

### `nyc-taxi-2024-10m.ndjson` (`10,000,000` rows, `2.62 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest/ETL Time:** `VXG` **`152.6 MB`** (`ratio=0.0583`, **`2.23 s`** / `1.17 GB/s`, lossless=`YES`) · `VXG --record-aligned` `140.7 MB` (`ratio=0.0538`, `0.48 s`, lossless=`YES`) · `Parquet-Snappy` `193.3 MB` (`ratio=0.0739`, `0.77 s` ETL, lossless=`NO`) · `Parquet-Snappy (RG=16384)` `203.4 MB` (`ratio=0.0777`, `0.96 s` ETL) · `Parquet-Zstd` `144.7 MB` (`ratio=0.0553`, `0.92 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.10 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 0.35 s)* | 23.89 ms (418.5 M/s) | **5.77 ms (1731.7 M/s)** | 9.64 ms (1037.4 M/s) | 24.97 ms (400.5 M/s) | 12.28 ms (814.4 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 10.50 ms (952.5 M/s) | **2.01 ms (4968.1 M/s)** | 7.01 ms (1426.1 M/s) | 14.58 ms (685.9 M/s) | 8.78 ms (1139.1 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 12.23 ms (817.8 M/s) | **2.25 ms (4450.5 M/s)** | 6.89 ms (1451.6 M/s) | 15.96 ms (626.6 M/s) | 8.57 ms (1167.3 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 17.40 ms (574.6 M/s) | **6.77 ms (1477.1 M/s)** | 8.14 ms (1228.8 M/s) | 17.38 ms (575.2 M/s) | 9.86 ms (1014.4 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 31.46 ms (317.8 M/s) | **1.79 ms (5584.8 M/s)** | 7.65 ms (1306.9 M/s) | 19.20 ms (520.7 M/s) | 10.42 ms (959.7 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `253.09 ms` (`O(single_frame)` RAM).

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.04 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 125.46 ms (79.7 M/s) | **29.06 ms (344.1 M/s)** | 40.44 ms (247.3 M/s) | 65.94 ms (151.7 M/s) | 66.48 ms (150.4 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 68.95 ms (145.0 M/s) | **13.51 ms (740.3 M/s)** | 36.01 ms (277.7 M/s) | 50.54 ms (197.9 M/s) | 51.67 ms (193.5 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 87.80 ms (113.9 M/s) | **17.59 ms (568.6 M/s)** | 34.67 ms (288.5 M/s) | 49.33 ms (202.7 M/s) | 50.26 ms (199.0 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 128.28 ms (78.0 M/s) | 51.07 ms (195.8 M/s) | **42.09 ms (237.6 M/s)** | 62.47 ms (160.1 M/s) | 58.68 ms (170.4 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 87.83 ms (113.9 M/s) | **9.39 ms (1064.6 M/s)** | 15.07 ms (663.4 M/s) | 22.93 ms (436.0 M/s) | 25.45 ms (392.9 M/s) |

### `nyc-taxi-2024-40m.csv` (`40,000,000` rows, `3.06 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest/ETL Time:** `VXG` **`552.9 MB`** (`ratio=0.1805`, **`5.01 s`** / `0.61 GB/s`, lossless=`YES`) · `VXG --record-aligned` `550.2 MB` (`ratio=0.1796`, `1.80 s`, lossless=`YES`) · `Parquet-Snappy` `773.2 MB` (`ratio=0.2524`, `3.07 s` ETL, lossless=`NO`) · `Parquet-Snappy (RG=16384)` `818.8 MB` (`ratio=0.2673`, `6.61 s` ETL) · `Parquet-Zstd` `579.3 MB` (`ratio=0.1891`, `3.07 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.08 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 0.94 s)* | 23.24 ms (1721.5 M/s) | **21.05 ms (1899.9 M/s)** | 30.67 ms (1304.2 M/s) | 84.90 ms (471.1 M/s) | 41.88 ms (955.1 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 18.81 ms (2126.2 M/s) | **16.70 ms (2395.2 M/s)** | 26.88 ms (1488.4 M/s) | 61.07 ms (655.0 M/s) | 31.71 ms (1261.5 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 16.00 ms (2499.6 M/s) | **14.94 ms (2678.2 M/s)** | 25.22 ms (1585.9 M/s) | 61.60 ms (649.4 M/s) | 30.62 ms (1306.3 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 35.02 ms (1142.1 M/s) | 31.92 ms (1253.1 M/s) | **31.15 ms (1284.0 M/s)** | 75.46 ms (530.1 M/s) | 41.13 ms (972.5 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 22.86 ms (1749.8 M/s) | **16.15 ms (2476.8 M/s)** | 18.21 ms (2196.5 M/s) | 77.10 ms (518.8 M/s) | 21.16 ms (1890.7 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `805.71 ms` (`O(single_frame)` RAM).

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.04 ms`)

| Query | `VXG` (`64 KiB`) | `VXG --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 161.89 ms (247.1 M/s) | **154.32 ms (259.2 M/s)** | 161.30 ms (248.0 M/s) | 253.91 ms (157.5 M/s) | 274.92 ms (145.5 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 106.41 ms (375.9 M/s) | **98.22 ms (407.3 M/s)** | 145.05 ms (275.8 M/s) | 209.37 ms (191.0 M/s) | 202.00 ms (198.0 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 116.61 ms (343.0 M/s) | **111.97 ms (357.2 M/s)** | 134.07 ms (298.4 M/s) | 200.60 ms (199.4 M/s) | 188.64 ms (212.0 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 263.14 ms (152.0 M/s) | 242.90 ms (164.7 M/s) | **174.81 ms (228.8 M/s)** | 254.83 ms (157.0 M/s) | 234.05 ms (170.9 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 94.59 ms (422.9 M/s) | 86.68 ms (461.5 M/s) | **26.95 ms (1484.0 M/s)** | 79.96 ms (500.2 M/s) | 42.03 ms (951.8 M/s) |

