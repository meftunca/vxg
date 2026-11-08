# Changelog

## 0.5.1 — Macroscopic Frame Dictionary (`<= 256 KiB`), L1D `u16` + Tagged Shared `BLOCK_DICT_LZ` Engine & `.txt` / Log Throughput Gains

- **Adaptive Macroscopic Corpus Dictionary (`<= 256 KiB`, `src/dict_lz/train.rs`, `src/dict_lz/mod.rs`, `docs/FORMAT.md`):**
  - Extended `VERSION_V5` `FLAG_FRAME_DICT` up to `262,144` bytes (`256 KiB`) with a shift-invariant SWAR `1-in-16` byte anchor (`anchor_mask16`), `127`-window `64 KiB` cross-frame frequency/liveness sketch (`sketch16`), exact `128`-byte corpus-period lock, and `strongly_tabular` (`l * 4 < chunk0.len()`) guard so Markdown pipe tables inside documentation corpora do not bypass dictionary extraction.
  - On `--suite v4` **`source-and-docs.txt`** (`16 MiB`, `64 KiB` independent random-access blocks), compression ratio improved **`72×`** from `0.2592` (`v0.5.0`) to **`0.0036`** (`277:1`, `55.8×` smaller than `Zstd+Dict-9` `0.2007`, `119.1×` smaller than `LZ4` `0.4286`), while throughput jumped to **`3.62 GB/s` Encode / `62.69 GB/s` Decode (`1T`)** and **`4.36 GB/s` Encode / `94.54 GB/s` Decode (`8T`)** (surpassing `LZ4` on both encode and decode, including full in-band dictionary extraction and compression), lowering the 8-dataset `--suite v4` structured/text geomean from `0.0442` to **`0.0259`** (`10`-dataset geomean `0.0248 -> 0.0162`, `54.97 GB/s` `8T` decode).
- **L1D `u16` Local Hash + Register-Tagged Immutable Shared Dictionary (`src/dict_lz/mod.rs`, `src/dict_lz/encode.rs`, `src/dict_lz/decode.rs`):**
  - Replaced per-block `256 KiB` sliding-window copies and `u32` hash-table memsets with L1D-resident `32 + 32 KiB` `[[u16; 2]; 8192]` intra-block tables (`const SMALL: bool` specialization for `<= 64 KiB` blocks) paired with an immutable frame-shared `PreparedDict` (`13`-bit register-tagged `ht4` + adaptive `64..512 KiB` `ht6`, pre-indexed once per frame across all worker threads), conditional `NEED_HT4` surgical lazy matching, and a zero-TLS / zero-Huffman raw-slice decode fast path (`huf_flags == 0`).
  - Improved real-world public `64 KiB` block ratios and `8T` speeds: **`pg100-shakespeare.txt`** `0.3977 -> 0.3884` (surpassing both `Zstd-9` `0.3918` and `Zstd+Dict-3` `0.4005` on non-repeating prose), **`gharchive-2025-01-15.ndjson`** `0.1350 -> 0.1231` (surpassing `Zstd+Dict-3` `0.1254` and `Zstd-9` `0.1414`, with `8T` throughput rising `1.48 / 8.59 -> 1.85 / 14.35 GB/s`), **`apache-access.clf`** `0.1037` (beating `Zstd-9` `0.1062` and `Zstd+Dict-3` `0.1075` at `12.94 GB/s` `8T` decode), and **`macos-install.log`** `0.0605 -> 0.0594` (`3.51 / 15.21 -> 3.97 / 17.47 GB/s`), bringing the **6-dataset real public geomean to `0.1213`** (`2.17 / 14.46 GB/s` `8T`, within **`0.66%`** of `Zstd+Dict-9` `0.1205`, zero C dependencies, `wasm32` verified).

## 0.5.0 — Format v5 (`BLOCK_DICT_LZ` & `--record-aligned`), `MODEL_SCALED_DECIMAL`, SQL Frontend (`vxg sql`), Query Engine Speedups & `40M`-Row Scale Benchmarks

- **Format v5 (`VERSION_V5 = 5`), Self-Contained Frame Dictionary & `BLOCK_DICT_LZ` (`src/dict_lz/`, `src/frame/v5.rs`, `src/encode.rs`):**
  - Added pure-Rust (`wasm32-unknown-unknown` compatible, zero C dependencies) FastCover-inspired `64`-window prefix dictionary trainer (`<= 32 KiB`, embedded and `DICT_LZ`-compressed inside the `v5` frame header via `FLAG_FRAME_DICT = 0x0002`, pre-indexed once across worker threads) paired with `BLOCK_DICT_LZ` (`kind = 3`: unified `ht4` + `ht6` 6-byte hash, 3 repeat-offset slots, backward match extension, 5-stream canonical Huffman coding, and inline SIMD `copy16` dictionary decoding). Frames not using `v5` features continue to emit `VERSION_V4` (`4`), and all `v1`–`v4` frames remain 100% readable.
  - Reduced `64 KiB` block ratios across unstructured logs, nested JSON, prose, and camera BMPs on the real public suite: `gharchive-2025-01-15.ndjson` **`0.1985 -> 0.1350`** (beating `Zstd-9` `0.1414`), `apache-access.clf` **`0.1198 -> 0.0990`** (beating `Zstd-9` `0.1062` & `Zstd+Dict-3` `0.1075`), `macos-install.log` **`0.0948 -> 0.0605`** (beating `Zstd-9` `0.0770`), `pg100-shakespeare.txt` **`0.4424 -> 0.3977`**, and `kodak-truecolor-16.bmp` **`0.9256 -> 0.8106`** (beating `LZ4` `0.8732` & `Zstd-1` `0.8274`).
  - Improved the **6-dataset Real Public Text & Structured Geomean** from `0.1484` to **`0.1231`** — strictly beating `LZ4` (`0.2500`), `Zstd-1` (`0.1620`), `Zstd-9` (`0.1419`), and `Zstd+Dict-3` (`0.1407`), within `2.1%` of `Zstd+Dict-9` (`0.1205`) — while boosting `8T` decode geomean from `7.86` to **`12.36 GB/s`**.
- **Record-Aligned Variable Block Splitting (`--record-aligned`, `FLAG_VAR_RAW = 0x0004`):**
  - Added optional `\n`-snapped variable-size blocks (`Options::record_aligned`, CLI `--record-aligned`), eliminating cross-block split-line `Row::Raw` fallbacks across `40,000+` blocks and cutting multi-GB compressed-domain query latency by up to `2.2×`.
- **`MODEL_SCALED_DECIMAL` (`tag 16`), Unquoted ISO Timestamps, CSV Header Resolution & Nested JSON Flattening (`src/columns/`):**
  - Added lossless fixed-scale ASCII decimal/currency packing (`MODEL_SCALED_DECIMAL`, `scale 1..=9`, `base: i64` + bit-packed `u64` deltas + zone maps), unquoted `YYYY-MM-DD HH:MM:SS` `MODEL_TIMESTAMP`, Block-0 CSV header isolation (lifting `nyc-taxi-2024-01.csv` `8T` throughput from `1.34 / 3.10 GB/s` to **`2.90 / 13.43 GB/s`**, surpassing `LZ4 (a1)` `13.15 GB/s` decode at `2.32×` smaller size), automatic CSV/TSV header-name column resolution, and 1st-level nested JSON flattening (`actor.login`, `repo.name`).
- **Compressed-Domain Query Speedups & SQL Frontend (`vxg sql`, `vxg::sql`, `vxg_sql_alloc`):**
  - Replaced per-block `HashMap` allocations on narrow numeric group keys with `DenseRange` (`span <= 4096`) buckets, added cost-sorted short-circuit `RowBitset` mask propagation on compound `AND`/`OR` predicates, and unified parallel chunk reduction (`src/ops/par_chunks.rs`).
  - On in-cache `8T` benchmarks (`reports/duckdb-vs-vxg.json`), **`VXG v0.5.0` wins `15 / 15` queries against prepared `DuckDB 2.0.0 + Parquet-Snappy/Zstd`** (`nyc-taxi.ndjson`: `COUNT WHERE` `1.38 -> 0.70 ms`, `SUM` `0.74 -> 0.35 ms`, `GROUP BY payment_type` `1.22 -> 0.70 ms`; `nyc-taxi.csv`: `COUNT WHERE` **`0.16 ms` (`1,906.8 M rows/s`)**, `SUM` **`0.21 ms` (`1,417.0 M rows/s`)**, `GROUP BY` **`0.42 ms`**; `ndjson_varied`: `SUM` **`0.29 ms` (`1,023.6 M rows/s`)**).
  - Added `vxg sql "SELECT ... WHERE ... GROUP BY ... ORDER BY ... LIMIT ..."` (`TSV`/`CSV`/`JSON`, `--stream`), `vxg::sql::{SqlQuery, execute_sql, execute_sql_stream}`, and C ABI `vxg_sql_alloc` (`include/vxg.h`).
- **Out-of-Cache Multi-GB Scale Benchmark (`10M` NDJSON & `40M` Full-Year NYC TLC CSV, `scripts/bench_scale_duckdb.py`, `reports/scale-duckdb-vs-vxg.{json,md}`):**
  - On **`nyc-taxi-2024-40m.csv`** (`40,000,000` real 2024 NYC TLC rows, `3.06 GB` raw), `VXG --record-aligned` (`551.6 MB`, `1.56 s` 8T ingest, `lossless = YES`) is smaller and `2.4×–2.6×` faster to ingest than `Parquet-Zstd` (`579.4 MB`, `3.80 s`) and `Parquet-Snappy` (`773.2 MB`, `4.06 s`), and **wins `5 / 5` `8T` queries** (`17.53–44.39 ms`, up to **`2,281.8 M rows/s`**). On **`nyc-taxi-2024-10m.ndjson`** (`10,000,000` rows, `2.62 GB` raw -> `145.9 MB` in `1.02 s`), `VXG --record-aligned` matches `Parquet-Snappy` on full-scan `SUM` (`8.26` vs `8.20 ms`, beating `Parquet-Zstd` `9.69 ms`), while `DuckDB + Parquet` leads selective `COUNT WHERE`, `GROUP BY`, and `Top-10` (`8.02–13.20 ms` vs `13.06–32.12 ms`).

## 0.4.7 — Real-Data Headline Tightening, `PREPARE/EXECUTE` + `SELECT 1` DuckDB Calibration, Near-Raw Speed Guard & `v0.5` Roadmap

- **Headline & Zero-ETL Lossless Framing (`README.md`, `docs/index.md`, `docs/CHIMERA_BENCHMARKS.md`):**
  - Narrowed top-level compression-ratio claims strictly to real flat tabular corpora (`1.18x–1.32x` smaller `64 KiB` blocks than `Zstd + Trained Dict L9` on `nyc-taxi` NDJSON/CSV; `14x–21x` faster ingest than Parquet ETL with 100% byte-exact original file recovery), explicitly highlighting where `Zstd + Trained Dict` wins `64 KiB` ratio (`gharchive`, `apache-access.clf`, `macos-install.log`, `pg100-shakespeare.txt`; `0.1205` vs `0.1484` 6-dataset geomean) and where `LZ4` wins decode speed.
- **Uncontested + `PREPARE/EXECUTE` + `SELECT 1` DuckDB Benchmark Calibration (`scripts/bench_vs_duckdb.py`, `reports/duckdb-vs-vxg.json`):**
  - Added `SELECT 1` ad-hoc/prepared overhead calibration (`0.04–0.07 ms` prepared) and `PREPARE / EXECUTE` Parquet latencies across all queries on a standalone uncontested CPU, plus an explicit in-cache (`200k–300k`-row) scope boundary.
- **Near-Raw Storage Speed-Bolding Guard (`src/bin/bench_support/report.rs`, `reports/real-public-{1t,8t}.md`):**
  - Excluded codecs with `ratio >= 0.90` (`< 10%` compression, e.g., `VXG` `0.9256` on `kodak-truecolor-16.bmp`) from speed and composite-score bolding.
- **Known Issues & `v0.5` Roadmap (`README.md`, `docs/ROADMAP.md`, `docs/PROGRESS.md`):**
  - Documented non-short-circuit `BitSet` unpacking in multi-column `COUNT(*) WHERE` (`3.84 ms` `1T`), per-block `HashMap` allocation in `MODEL_DECIMAL_PACKED` `GROUP BY`, `BLOCK_ZSTD_DICT` hybrid fallback for unstructured logs/nested JSON, and `40M+`-row out-of-cache full-year NYC TLC benchmarking for `v0.5`.

## 0.4.6 — Trained-Dictionary & Whole-File `Zstd` Baselines, Real Public Corpora & `DuckDB + Parquet` Benchmarks

- **Trained-Dictionary (`Zstd + Dict`) & Whole-File `zstd -19 --long` Baselines (`src/baselines/zstd_ext.rs`, `src/baselines.rs`, `vxg-bench`):**
  - Added `TrainedDict::train` (`ZDICT_trainFromBuffer` up to `32 KiB`, recording one-time `train_ns`) and thread-safe `Codec::ZstdDict` (`zstd-dict-l3`, `zstd-dict-l9` via `ZSTD_CDict` / `ZSTD_DDict`) over `64 KiB` independent blocks, charging `dict_bytes` in `Encoded::encoded_size()`.
  - Added whole-file `Codec::ZstdWholeLong` (`zstd-whole-l19-long`, `zstd -19 --long` with `128 MiB` window / `windowLog = 27`, single stream, `random access = NO`) to quantify the exact compression-ratio difference of `64 KiB` random-access block boundaries.
  - Updated `vxg-bench` Markdown reporting (`src/bin/bench_support/report.rs`) to separate `Text & Structured Geomean` from `2D BMP Geomean` and bold only the actual per-row winner on each metric (`Ratio`, `Enc`, `Dec`).
- **Real-World Public Corpora vs. Synthetic Qualification Separation (`scripts/fetch_real_corpora.py`, `docs/CHIMERA_BENCHMARKS.md`):**
  - Added reproducible fetching and benchmarking over 7 non-synthetic public/OS corpora (`nyc-taxi-2024-01.{csv,ndjson}`, `gharchive-2025-01-15.ndjson`, `apache-access.clf`, `macos-install.log`, `pg100-shakespeare.txt`, and `kodak-truecolor-16.bmp` 16-photo camera mosaic), strictly separated from synthetic/procedural qualification suites (`--suite v4`, `gen_varied_ndjson.py`, `gen_format_bench.py`, procedural UI `.bmp` renders).
- **Head-to-Head Analytical Benchmarks vs. `DuckDB 2.0.0` + `Parquet` & `vxg_group_by_alloc` C ABI (`src/ffi/analytics.rs`, `include/vxg.h`, `scripts/bench_vs_duckdb.py`):**
  - Added `vxg_group_by_alloc` to the zero-overhead C ABI (`include/vxg.h`) for in-process `GROUP BY` + `COUNT/SUM/MIN/MAX` analytics.
  - Replaced `"effective GB/s"` (`raw_bytes / query_time`) reporting in `vxg-query-real`, `CompressedOpsChart.vue`, and documentation with **wall-clock latency (`ms`)** and **million rows/sec (`M rows/s`)**, benchmarked directly against `DuckDB 2.0.0` (`read_json_auto` / `read_csv`, `.zst`, `Parquet-Snappy`, and `Parquet-Zstd`) with 100% result-equality assertions (`reports/duckdb-vs-vxg.json`).

## 0.4.5 — Persistent Lock-Free Worker Pool, Multi-Template SIMD, Compressed-Domain Speedups & Competitor Sweep Baselines

- **Persistent Lock-Free Worker Pool & Parallel Boundary Inspection (`src/pool.rs`, `src/ops/par_meta.rs`, `src/query/parallel.rs`):**
  - Replaced per-call `std::thread::scope` OS thread creation/teardown and cold TLS allocations across `encode`, `decode`, `query`, and `ops` with a cache-aligned persistent worker pool (`pool::parallel_run` / `parallel_try_run`) and parallel grammar boundary inspector (`inspect_grammar_boundary`), with graceful `wasm32` sequential fallback.
  - Accelerated 8-thread compressed-domain analytics on synthetic `56.5 MB` (`300,000`-row) `ndjson_varied.ndjson`: `aggregate_column SUM/MIN/MAX(bytes)` **`6.13 ms -> 0.57 ms` (`526.3 M rows/s`, `152.0x` vs naive — `10.8x` faster)**, `group_by(service) SUM/MIN/MAX(bytes)` **`5.22 ms -> 1.00 ms` (`300.0 M rows/s`, `157.9x` — `5.2x` faster)**, and `top_k_by ORDER BY latency_ms ASC LIMIT 10` **`1.33 ms -> 0.58 ms` (`517.2 M rows/s`, `138.9x` — `2.3x` faster)**.
- **Cross-Block `leading_carry` Raw Boundary Routing & Zero-Alloc Multi-Template Ops (`src/encode.rs`, `src/columns/`):**
  - Routed cross-block split-line leading fragments (`leading_carry`) directly to `Row::Raw(&data[..first_nl])`, eliminating bogus 1-row `Template 0` splits on 92% of CSV/TSV/logfmt blocks and preventing column-0 model poisoning (`events.tsv` `0.0334 -> 0.0326`, `data.csv` `0.1820 -> 0.1796`, CSV `1T` `filter_column` **`5.01 ms -> 0.58 ms`**, `map_column` **`29.83 ms -> 2.32 ms`**).
  - Eliminated per-block heap allocations (`TplColSlices`) in `ops_search.rs` and `ops_pred.rs` and added compressed-domain multi-template `filter_frame` (`ops_filter_bytes.rs`, `8T` `4.24 ms -> 2.18 ms`).
- **Multi-Template SIMD Stream Matcher, `FAST_LZ` Periodic Overlap SIMD & 6-Step 4-Stream Huffman (`Format v4` — 100% Wire Compatible):**
  - Added SIMD multi-template JSON stream matching (`grammar_multi.rs`), NEON 256-entry nibble LUT in `try_match_clf_stream`, early high-cardinality abort in `model_dict.rs`, and `u32` multiply-shift / single-digit-token fast paths in `prep_pattern.rs`.
  - Added `OVERLAP_SHUF16` + `copy16_periodic` (`vqtbl1q_u8` / `_mm_shuffle_epi8`) for periodic `offset < match_len` matches in `fast_lz_unpack.rs` and constant-offset 6-step (`max_len <= 9`) / 5-step (`max_len in 10..=11`) 4-stream Huffman decoding (`huffman_dec4.rs`).
  - Removed upfront `16+ MiB` zero-fill in `StreamEncoder::with_chunk_size` and added monotonic `out_len` buffer reuse in `StreamDecoder` (`src/stream.rs`).
- **High-Effort Competitor Sweep Baselines (`vxg-bench`, `docs/CHIMERA_BENCHMARKS.md`):**
  - Added native C `lz4hc-c-l9` (`LZ4_compress_HC_extStateHC`) and `zstd-c-l9` baselines plus parameter-sweep upper-bound envelopes (`min_ratio` + `max_speed`) across independent `64 KiB` blocks.

## 0.4.4 — Compressed-Domain `ORDER BY` / `Top-K`, Expanded C/FFI Analytics ABI & Interactive VitePress Charts

- **Compressed-Domain `ORDER BY` / `Top-K` Heap Engine (`vxg::ops`, `vxg::stream_ops`):**
  - Added `SortDir`, `SortKey`, `top_k_by`, `select_top_k_by`, `top_k_by_stream`, and `select_top_k_by_stream` with deterministic `NULLS LAST` ordering and `global_row_index` tie-breaking across threads and multi-frame streams.
  - Added $O(1)$ `MODEL_DECIMAL_PACKED` / `MODEL_DECIMAL_AFFINE` numeric zone-map block skipping, `AtomicU64` cross-worker monotonic bound pruning, pre-pruned dictionary `alive` bitmaps, `TplRowMap::Offset` $O(1)$ row indexing on boundary-split 1-template blocks, and bounded per-template `CandTracker` ($\le K$ candidate rows materialized per template).
  - Reaches **35.6–42.3 GB/s** (`1.33–1.64 ms`, **54x–72x** faster than `decompress` + $O(N)$ `select_nth_unstable_by` partial sort) on `56.5 MB` (`300,000`-row) `ndjson_varied.ndjson` across 8 threads.
  - Extended `vxg query` with `--order-by COL [--desc|--asc]` (both `mmap` and `--stream`) and `vxg agg` with `--order-by key|rows|count|sum|avg|min|max [--desc|--asc] [--limit K]`.
- **Shared Expression Parser & Expanded Zero-Overhead C/FFI ABI (`include/vxg.h`, `src/ffi/`):**
  - Promoted the recursive-descent compound `WHERE` and column-target parser to `vxg::ops` (`ParsedWhere`, `ParsedColPred`, `parse_col_target`, `parse_col_list`).
  - Added 8 panic-safe C ABI functions (`vxg_verify_frame`, `vxg_compress_file`, `vxg_decompress_file`, `vxg_verify_file`, `vxg_count_where_str`, `vxg_aggregate_col` + 64-byte `vxg_agg_stats_t`, `vxg_select_alloc`, `vxg_project_frame_alloc`).
- **Interactive Zero-Dependency VitePress Charts (`docs/.vitepress/theme/`):**
  - Added interactive Vue 3 benchmark chart components (`CodecBenchmarkChart`, `CompressedOpsChart`, `EnergyBenchmarkChart`) across `docs/index.md`, `docs/BENCHMARKS.md`, `docs/ENERGY_BENCHMARKS.md`, and `docs/STAGE2_DESIGN.md` with zero extra npm dependencies.

## 0.4.3 — Zero-Copy `mmap` I/O, Zero-Output-Alloc Verification & Bounded-Memory Multi-GB Streaming

- **Zero-Dependency Cross-Platform Memory-Mapped File I/O (`vxg::mmap`):**
  - Added raw OS FFI memory-mapping (`MappedFile`, `MappedFileMut`, `InputBytes`, `mmap_supported`) across 64-bit Unix (`mmap`, `munmap`, `msync`, `madvise` `MADV_SEQUENTIAL`/`MADV_RANDOM`/`MADV_WILLNEED`) and 64-bit Windows (`CreateFileMappingW`, `MapViewOfFile`, `FlushViewOfFile`), with transparent fallback on `wasm32` / non-file descriptors and zero external crate dependencies.
  - Added `compress_file`, `decompress_file` (`16 MiB` bounded parallel block-batch decoding for multi-GB frames + automatic destination cleanup on error), and `verify_file`.
- **Zero-Output-Allocation Parallel Verification & `FrameReader<R>` (`vxg::stream_ops`):**
  - Added `verify_frame` and `verify_stream`: verifies all block directory bounds, command/column streams, and per-block `CRC32C` checksums across multiple worker threads using only `~64 KiB` L2-cache-resident scratch per thread (`0` output heap allocation, in-place zero-copy `CRC32C` on `BLOCK_RAW` blocks, reaching **12.0+ GB/s** on `8T`).
  - Added `FrameReader<R: Read>` for zero-reallocation frame-by-frame `VXG1` streaming with `O(single_frame)` memory.
- **Bounded-Memory Multi-Frame Compressed-Domain Streaming (`FramePeeler` & `*_stream` APIs):**
  - Added `FramePeeler` cross-frame split-line boundary stitcher preserving interior `GRAMMAR_PACKET` blocks 100% in the compressed domain while handling CSV column-0 boundary cuts, multi-frame monster lines, and unterminated EOF tails.
  - Added 10 streaming compressed-domain APIs: `frame_row_count_stream`, `count_where_stream`, `filter_where_stream` (with early `limit` stop), `select_columns_stream` (with early `limit` stop), `aggregate_column_stream`, `group_by_stream`, `project_frame_stream`, `filter_frame_where_stream`, `update_column_stream`, and `map_column_stream`.
  - Scaled `StreamEncoder::new` default chunk sizes (`4..16 MiB` by thread count) and eliminated intermediate `8 KiB` `io::copy` passes in `compress_stream` and `decompress_stream` (**1.8x–4.0x** faster multi-threaded streaming).
- **CLI Default `mmap` & Universal `--stream` / `stdin` Support (`vxg`):**
  - Enabled `mmap` by default on regular files (with `--no-mmap` opt-out) and added `--stream` + `-` (`stdin` / `stdout`) streaming execution across `verify`, `query`, `agg`, `project`, `filter-frame`, `update`, and `map`.

## 0.4.2 — Compressed-Domain Query, Analytics, Mutation, Projection & Wave 2–3 SIMD Speedups

- **Compressed-Domain Column Projection & Zero-Reencode Column Splicing (`vxg::select_columns`, `vxg::project_frame`):**
  - `select_columns`: Projects arbitrary subsets/reorderings of JSON keys (`ColTarget::JsonKey`) or delimited fields (`ColTarget::Delimited`) directly from `GRAMMAR_PACKET` blocks using miniature `UnboundTemplate` / `PreparedTemplate` SIMD emission, skipping unselected columns completely, with full compound `PredExpr` `WHERE` filtering, `limit`, and multi-threaded `std::thread::scope` execution (up to **20.7 GB/s** / **37.3x** faster than naive decode + scan).
  - `project_frame`: Zero-reencode `VXG -> VXG` column splicing (`GrammarProjectOut::Spliced`) that copies selected compressed column payloads byte-for-byte without re-running model discovery, absorbs cross-block split boundary lines as 1-row raw tails, and preserves 100% byte-for-byte decode equivalence with `select_columns` (**32.6x** faster than `decompress + project + compress` at **10.7 GB/s**).
- **Compound Boolean Predicates, Scalar Aggregations & `GROUP BY` (`vxg::ops`):**
  - Zero-heap `Copy` `PredExpr` boolean expression tree (`Col`, `RowContains`, `Not`, `All`, `Any`, `True`) with 64-bit `TplMask` short-circuit evaluation across `count_where`, `filter_where`, `filter_frame_where`, `aggregate_column`, `group_by`, `select_columns`, and `project_frame`.
  - `aggregate_column` (`AggStats`: `row_count`, `count`, `sum`, `min`, `max`, `avg`) and `group_by` (`GroupRow`, `GroupKey`, 2-tier dictionary-ID aggregation + parallel worker merge, reaching **40+ GB/s** effective throughput).
- **Compressed-Domain Mutations, Pushdown Search & Zero-Wire-Byte Prechecks (`vxg::query`, `vxg::ops`):**
  - Structured record random access (`frame_row_count`, `extract_row`, `extract_row_range`) and pushdown search (`find_first`, `filter_rows`, `filter_json_u64`) backed by 256-bit block/column alphabet masks, zero-wire-byte 256-entry bigram precheck (`query_bigram.rs`), numeric zone-map bounds, and SIMD hex-nibble scanning.
  - Column-addressed `filter_column`, `filter_frame`, `update_column`, and `map_column` with single-slot binary column rewrites (`O(|D|)` dictionary splice and direct numeric stream updates), exact multi-block split-line stitching, and deterministic multi-threaded work-stealing.
- **Core Codec Models & Wave 1–3 Encode/Decode Throughput (`Format v4` — 100% Wire Compatible):**
  - Added `MODEL_PATTERN` (`tag 15`) with SWAR digit/literal shape detection and `Copy16` constant-affix decode (`2.5x–7x` smaller on non-ISO timestamps, IPv4, and CLF logs), relaxed CSV/TSV free-text field matching, single-pass CLF stream matcher, and unquoted `MODEL_HEX_SUFFIX`.
  - Accelerated v4 model gates, SIMD stream matchers, short-raw decode fusion, zero-realloc thread-local model pools, 5-way / interleaved canonical Huffman decode, `x86_64` SSE2/SSSE3/SSE4.1/AVX2 parity, and lock-free frame assembly.
- **Full Analytics & Projection CLI (`vxg`):**
  - Added recursive-descent compound `--where EXPR` parser (`&&`/`AND`, `||`/`OR`, `!`/`NOT`, `(...)`, `contains:STR`, auto-numeric/string comparisons), `vxg query --select`, `vxg agg`, `vxg project`, `vxg filter-frame`, `vxg update`, `vxg map`, and `-t N|auto` across commands.

## 0.4.1 — Energy Indices, Cross-Platform `x86_64` / `aarch64` Parity & Bounded Streaming

- **Per-Process Hardware Energy & Composite Indices (`VPI`, `BCI`, `GCI`):** Integrated kernel hardware energy (`J/GB`) into `vxg-bench` and achieved 10/10 dataset `VPI` wins against `liblz4` (`5.27x` geomean) and `libzstd` (`4.93x` geomean).
- **2-Way ILP `FAST_LZ` & 4-Stream Huffman:** Beat `Zstd-1` on both compression ratio (`0.2698` vs `0.2713`) and encode/decode throughput on `source-and-docs.txt`.
- **Bounded-Memory Streaming & Multi-Frame Support:** Added `StreamEncoder<W>`, `StreamDecoder<R>`, concatenated `VXG1` frame decoding, `vxg_frame_compressed_size`, and `--stream` CLI mode.

## 0.4.0 — Unified Adaptive v4 Release Pipeline

- **Format v4 (`VERSION_V4 = 4`):** Compact 8-byte independent block directory (`u32 (payload_size | kind << 30)` + `u32 crc32c`) with full backward compatibility for decoding `v1`–`v3` frames.
- **Adaptive Block Router (`BLOCK_COLUMN` / `BLOCK_RAW` / `BLOCK_LZ`):** Structured columnar engine (`GRAMMAR_PACKET`), 2D/Stride pixel engine (`STRIDE_PACKET`), and `Fast-LZ` + 4-stream canonical Huffman (`FAST_LZ_PACKET`).
- **CLI & Cross-Platform Benchmark Suite:** Full `vxg` CLI and C99/C++11 FFI (`include/vxg.h`).

## 0.1.0 — Experimental Source Delivery

First VXG1 wire format, Rust CPU/SIMD implementation, native Metal/CUDA/Vulkan sources, reference C-library benchmark harness, independent Python model, and golden/error vectors.
