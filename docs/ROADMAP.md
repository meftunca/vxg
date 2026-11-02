# Roadmap

VXG is an adaptive block-independent lossless codec and compressed-domain execution engine. GPU backends, VIMG, and VVID are maintenance-only / out of scope for new feature work. The goal is to be the fastest and smallest codec on structured and semi-structured data while supporting direct SQL-grade queries, mutations, analytics, and column projections without full decompression.

| Stage | Goal | Status |
| :--- | :--- | :--- |
| **1** | Core `v4` speed & ratio on real-world structured + free-text workloads (`MODEL_PATTERN`, CLF matcher, Wave 1–3 SIMD pools) | **Completed in `v0.4.2`** |
| **2** | Compressed-domain data ops: `filter`, `map`, `update`, zero-wire-byte bigram & nibble pushdown | **Completed in `v0.4.2`** ([Design & Benchmarks](./STAGE2_DESIGN.md)) |
| **3** | Compound boolean `PredExpr` (`AND`/`OR`/`NOT`), scalar aggregations (`SUM`/`MIN`/`MAX`/`AVG`), and 2-tier dictionary `GROUP BY` | **Completed in `v0.4.2`** |
| **4** | Compressed-domain `SELECT` column projection & zero-reencode `VXG -> VXG` column splicing (`project_frame`) + full CLI (`query`, `agg`, `project`) | **Completed in `v0.4.2`** |
| **5** | Zero-copy `mmap` & bounded-memory multi-GB streaming execution for `query`, `agg`, `project`, `filter-frame`, `update`, and `map` | **Completed in `v0.4.3`** |
| **6** | Continuous correctness hardening (randomized differential fuzzing, sanitizers, `x86_64` / `aarch64` parity) | Ongoing ([Testing](./TESTING.md)) |

## Engineering Rules

- Benchmark on real multi-MB files (`ndjson_varied`, `data.csv`, `data.clf`, `install.log`), not only synthetic data.
- Compare against native C `liblz4` and `libzstd` at equal block budget; keep rows where a competitor wins.
- Report compression ratio and single-thread / multi-thread throughput together.
- Assert 100% byte-for-byte differential equivalence against naive `decompress + op (+ compress)` for every compressed-domain operation.
- Every `.rs` file in `src/` and `tests/` stays `<= 600` lines; no wire-format break without an explicit version bump.
