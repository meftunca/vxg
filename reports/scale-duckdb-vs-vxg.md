## Out-of-Cache Multi-GB Scale Benchmark (`10M`–`40M` Rows): `VXG v0.5.3` vs `DuckDB + Parquet`

- **Engine versions:** `VXG v0.5.3` (`mmap` zero-copy C-FFI + `--record-aligned` + `--stream`) vs `DuckDB 2.0.0` (`PREPARE / EXECUTE` calibrated with `SELECT 1`, testing both default `ROW_GROUP_SIZE=122880` and fine-grained `ROW_GROUP_SIZE=16384`).
- **Verification:** Every query asserts 100% exact result equality between `VXG` and `DuckDB`, and every `.vxg` archive passes full `vxg_verify_file` lossless verification.
- **Key distinction:** `VXG` stores the original `.csv` / `.ndjson` **100% losslessly byte-for-byte** (`1` step, zero schema ETL), whereas `Parquet` requires an upfront lossy-to-formatting schema conversion (`COPY ... TO PARQUET`).

### `nyc-taxi-2024-10m.ndjson` (`10,000,000` rows, `2.62 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest (`--level 3` default) / ETL Time:** `VXG -l 3` **`152.6 MB`** (`ratio=0.0583`, **`3.24 s`** / `0.81 GB/s`, lossless=`YES`) · `VXG -l 3 --record-aligned` `140.8 MB` (`ratio=0.0538`, `0.67 s`, lossless=`YES`) · `Parquet-Snappy` `193.3 MB` (`ratio=0.0739`, `0.96 s` ETL, lossless=`NO`) · `Parquet-Snappy (RG=16384)` `203.4 MB` (`ratio=0.0777`, `1.21 s` ETL) · `Parquet-Zstd` `144.6 MB` (`ratio=0.0553`, `0.96 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.09 ms`)

| Query | `VXG -l 3` (`64 KiB`) | `VXG -l 3 --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 0.39 s)* | 20.36 ms (491.1 M/s) | **3.91 ms (2554.9 M/s)** | 13.64 ms (733.0 M/s) | 28.93 ms (345.7 M/s) | 13.48 ms (741.7 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 13.98 ms (715.2 M/s) | **2.30 ms (4339.5 M/s)** | 15.43 ms (648.2 M/s) | 21.73 ms (460.2 M/s) | 10.81 ms (925.3 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 15.79 ms (633.4 M/s) | **3.14 ms (3188.4 M/s)** | 8.02 ms (1246.4 M/s) | 18.91 ms (528.8 M/s) | 10.08 ms (991.9 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 19.15 ms (522.3 M/s) | **5.91 ms (1691.6 M/s)** | 10.62 ms (941.3 M/s) | 23.85 ms (419.2 M/s) | 11.56 ms (865.3 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 38.73 ms (258.2 M/s) | **1.58 ms (6325.4 M/s)** | 9.50 ms (1052.3 M/s) | 27.98 ms (357.4 M/s) | 14.11 ms (708.6 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `153.81 ms` (`O(single_frame)` RAM).

#### `8T` Cold NVMe Cache Query Latency (`ms`) & Throughput (`M rows/s`, 1st run after `F_NOCACHE` OS page-cache eviction)

| Query | `VXG -l 3` (`64 KiB`, cold) | `VXG -l 3 --record-aligned` (cold) | `DuckDB Parquet-Snappy` *(cold)* | `DuckDB Parquet-Snappy` *(`RG=16384`, cold)* | `DuckDB Parquet-Zstd` *(cold)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 144.43 ms (69.2 M/s) | 103.21 ms (96.9 M/s) | **16.12 ms (620.5 M/s)** | 35.22 ms (283.9 M/s) | 25.91 ms (385.9 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 133.06 ms (75.2 M/s) | 103.92 ms (96.2 M/s) | 19.92 ms (501.9 M/s) | 20.31 ms (492.4 M/s) | **13.40 ms (746.2 M/s)** |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 126.63 ms (79.0 M/s) | 140.85 ms (71.0 M/s) | **12.21 ms (819.1 M/s)** | 52.65 ms (189.9 M/s) | 16.17 ms (618.6 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 146.90 ms (68.1 M/s) | 146.56 ms (68.2 M/s) | 31.84 ms (314.1 M/s) | 38.57 ms (259.3 M/s) | **15.16 ms (659.5 M/s)** |
| `ORDER BY total_cents DESC LIMIT 10` | 170.46 ms (58.7 M/s) | 146.93 ms (68.1 M/s) | 52.59 ms (190.1 M/s) | 23.37 ms (427.9 M/s) | **16.86 ms (593.3 M/s)** |

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.05 ms`)

| Query | `VXG -l 3` (`64 KiB`) | `VXG -l 3 --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 126.06 ms (79.3 M/s) | **29.79 ms (335.7 M/s)** | 41.03 ms (243.7 M/s) | 68.89 ms (145.2 M/s) | 70.19 ms (142.5 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 70.13 ms (142.6 M/s) | **11.75 ms (850.7 M/s)** | 38.35 ms (260.7 M/s) | 54.11 ms (184.8 M/s) | 53.58 ms (186.6 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 129.90 ms (77.0 M/s) | **19.05 ms (524.9 M/s)** | 36.62 ms (273.1 M/s) | 52.38 ms (190.9 M/s) | 52.40 ms (190.8 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 111.12 ms (90.0 M/s) | **32.58 ms (306.9 M/s)** | 43.92 ms (227.7 M/s) | 64.45 ms (155.2 M/s) | 59.72 ms (167.4 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 84.78 ms (118.0 M/s) | **9.58 ms (1043.4 M/s)** | 16.10 ms (621.2 M/s) | 24.29 ms (411.7 M/s) | 27.29 ms (366.4 M/s) |

### `nyc-taxi-2024-40m.csv` (`40,000,000` rows, `3.06 GB` raw, provenance: `official_nyc_tlc_2024_months_01_to_12_unmodified_real_rows`)

- **Archive Size & 8T Ingest (`--level 3` default) / ETL Time:** `VXG -l 3` **`552.9 MB`** (`ratio=0.1805`, **`5.21 s`** / `0.59 GB/s`, lossless=`YES`) · `VXG -l 3 --record-aligned` `550.4 MB` (`ratio=0.1797`, `1.69 s`, lossless=`YES`) · `Parquet-Snappy` `773.1 MB` (`ratio=0.2524`, `4.32 s` ETL, lossless=`NO`) · `Parquet-Snappy (RG=16384)` `818.9 MB` (`ratio=0.2673`, `4.74 s` ETL) · `Parquet-Zstd` `579.4 MB` (`ratio=0.1891`, `3.72 s` ETL, lossless=`NO`).

#### `8T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.07 ms`)

| Query | `VXG -l 3` (`64 KiB`) | `VXG -l 3 --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` *(DuckDB raw text: 1.20 s)* | 26.05 ms (1535.6 M/s) | **25.88 ms (1545.4 M/s)** | 36.79 ms (1087.2 M/s) | 105.64 ms (378.6 M/s) | 60.13 ms (665.2 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 15.25 ms (2623.3 M/s) | **12.64 ms (3164.6 M/s)** | 28.03 ms (1426.9 M/s) | 62.24 ms (642.6 M/s) | 34.48 ms (1160.0 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 17.03 ms (2348.3 M/s) | **16.06 ms (2490.1 M/s)** | 31.43 ms (1272.9 M/s) | 70.70 ms (565.7 M/s) | 36.11 ms (1107.7 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 34.35 ms (1164.6 M/s) | **27.79 ms (1439.5 M/s)** | 35.50 ms (1126.9 M/s) | 84.15 ms (475.3 M/s) | 38.14 ms (1048.8 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 20.98 ms (1906.3 M/s) | **11.07 ms (3613.0 M/s)** | 18.03 ms (2218.0 M/s) | 78.75 ms (507.9 M/s) | 21.97 ms (1820.4 M/s) |

- **Bounded-memory CLI streaming (`vxg sql ... --stream -t 8`):** `374.32 ms` (`O(single_frame)` RAM).

#### `8T` Cold NVMe Cache Query Latency (`ms`) & Throughput (`M rows/s`, 1st run after `F_NOCACHE` OS page-cache eviction)

| Query | `VXG -l 3` (`64 KiB`, cold) | `VXG -l 3 --record-aligned` (cold) | `DuckDB Parquet-Snappy` *(cold)* | `DuckDB Parquet-Snappy` *(`RG=16384`, cold)* | `DuckDB Parquet-Zstd` *(cold)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 407.15 ms (98.2 M/s) | 382.38 ms (104.6 M/s) | 60.21 ms (664.3 M/s) | 128.25 ms (311.9 M/s) | **58.70 ms (681.5 M/s)** |
| `SUM/MIN/MAX(total_cents) [full scan]` | 386.27 ms (103.6 M/s) | 569.23 ms (70.3 M/s) | **32.44 ms (1232.9 M/s)** | 148.02 ms (270.2 M/s) | 42.21 ms (947.7 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 571.57 ms (70.0 M/s) | 370.81 ms (107.9 M/s) | **40.93 ms (977.3 M/s)** | 112.54 ms (355.4 M/s) | 42.36 ms (944.3 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 410.40 ms (97.5 M/s) | 582.47 ms (68.7 M/s) | **62.92 ms (635.8 M/s)** | 193.06 ms (207.2 M/s) | 63.24 ms (632.5 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 529.15 ms (75.6 M/s) | 643.57 ms (62.2 M/s) | 30.23 ms (1323.3 M/s) | 101.47 ms (394.2 M/s) | **27.02 ms (1480.3 M/s)** |

#### `1T` Query Latency (`ms`) & Throughput (`M rows/s`, `DuckDB SELECT 1 prep = 0.04 ms`)

| Query | `VXG -l 3` (`64 KiB`) | `VXG -l 3 --record-aligned` | `DuckDB Parquet-Snappy` *(prep)* | `DuckDB Parquet-Snappy` *(`RG=16384`)* | `DuckDB Parquet-Zstd` *(prep)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COUNT(*) WHERE pay=1 & tip>=1000 & dist>=500` | 162.72 ms (245.8 M/s) | **150.67 ms (265.5 M/s)** | 159.14 ms (251.4 M/s) | 282.92 ms (141.4 M/s) | 275.13 ms (145.4 M/s) |
| `SUM/MIN/MAX(total_cents) [full scan]` | 91.34 ms (437.9 M/s) | **82.42 ms (485.3 M/s)** | 156.03 ms (256.4 M/s) | 218.78 ms (182.8 M/s) | 210.90 ms (189.7 M/s) |
| `SUM/MIN/MAX(fare_cents) WHERE payment_type=2` | 114.70 ms (348.7 M/s) | **107.33 ms (372.7 M/s)** | 138.55 ms (288.7 M/s) | 204.45 ms (195.6 M/s) | 195.58 ms (204.5 M/s) |
| `GROUP BY payment_type: COUNT(*), SUM(total_cents)` | 180.78 ms (221.3 M/s) | **162.75 ms (245.8 M/s)** | 177.54 ms (225.3 M/s) | 262.69 ms (152.3 M/s) | 241.67 ms (165.5 M/s) |
| `ORDER BY total_cents DESC LIMIT 10` | 88.79 ms (450.5 M/s) | 63.08 ms (634.1 M/s) | **30.05 ms (1331.2 M/s)** | 91.18 ms (438.7 M/s) | 46.00 ms (869.6 M/s) |

