# Roadmap

VXG is an adaptive block-independent lossless codec and compressed-domain execution engine. GPU backends, VIMG, and VVID are maintenance-only / out of scope for new feature work. The goal is to be the fastest and smallest codec on structured and semi-structured data while supporting direct SQL-grade queries, mutations, analytics, and column projections without full decompression.

| Stage | Goal | Status |
| :--- | :--- | :--- |
| **1** | Core `v4` speed & ratio on real-world structured + free-text workloads (`MODEL_PATTERN`, CLF matcher, Wave 1–3 SIMD pools) | **Completed in `v0.4.2`** |
| **2** | Compressed-domain data ops: `filter`, `map`, `update`, zero-wire-byte bigram & nibble pushdown | **Completed in `v0.4.2`** ([Design & Benchmarks](./STAGE2_DESIGN.md)) |
| **3** | Compound boolean `PredExpr` (`AND`/`OR`/`NOT`), scalar aggregations (`SUM`/`MIN`/`MAX`/`AVG`), and 2-tier dictionary `GROUP BY` | **Completed in `v0.4.2`** |
| **4** | Compressed-domain `SELECT` column projection & zero-reencode `VXG -> VXG` column splicing (`project_frame`) + full CLI (`query`, `agg`, `project`) | **Completed in `v0.4.2`** |
| **5** | Zero-copy `mmap` & bounded-memory multi-GB streaming execution for `query`, `agg`, `project`, `filter-frame`, `update`, and `map` | **Completed in `v0.4.3`** |
| **6** | Compressed-domain `ORDER BY` / `Top-K` heap queries (`top_k_by`, `select_top_k_by`, `*_stream`), expanded zero-overhead C/FFI analytics ABI, and interactive VitePress benchmark charts | **Completed in `v0.4.4`** |
| **7** | Dependency-free in-browser WebAssembly (`vxg.wasm`) live playground & compressed-domain query sandbox | **Completed** ([Live Playground](./playground.md)) |
| **8** | `vxg-tensor` (`.vxt` `VXT1`) Compressed-Domain AI Model Codec (`Lossless-BP16`, `CB4/CB3+Outlier`, `WHT2+Outlier`) & 1-Command-Buffer Fused Apple Metal GPU / SIMD LLM Inference Engine | **Completed** ([AI Tensor Spec & Benchmarks](./TENSOR.md)) |
| **9 (`v0.5`)** | Hybrid `BLOCK_ZSTD_DICT` per-block fallback (bounding unstructured/CLF/nested-JSON block ratios by `Zstd + Trained Dict`), stack-bucket numeric `GROUP BY` + short-circuit `BitSet` mask propagation in compound `WHERE`, and out-of-cache `40M+`-row full-year NYC TLC benchmark vs. `DuckDB + Parquet` | **Planned for `v0.5`** |
| **10** | Continuous correctness hardening (randomized differential fuzzing, sanitizers, `x86_64` / `aarch64` parity) | Ongoing ([Testing](./TESTING.md)) |

## Engineering Rules

- Strictly separate **real-world public/OS corpora** (`nyc-taxi`, `gharchive`, `apache-access.clf`, `macos-install.log`, `pg100-shakespeare.txt`, `kodak` photos) from **synthetic/procedural qualification suites** (`--suite v4`, `ndjson_varied`, `data.csv`, `data.clf`, procedural UI `.bmp` renders).
- Compare against native C `liblz4` and `libzstd` (including `Zstd + Trained Dictionary` at equal `64 KiB` block budget and whole-file `zstd -19 --long` reference); report and explain rows where a competitor wins.
- Never bold encode/decode throughput on near-raw storage (`ratio >= 0.90`).
- Report analytical queries in wall-clock latency (`ms`) and scan rate (`M rows/s`) against `DuckDB + Parquet` (including `SELECT 1` / prepared-statement calibration), never `"effective GB/s"` (`raw_bytes / query_time`).
- Assert 100% byte-for-byte differential equivalence against naive `decompress + op (+ compress)` for every compressed-domain operation.
- Every `.rs` file in `src/` and `tests/` stays `<= 600` lines; no wire-format break without an explicit version bump.
