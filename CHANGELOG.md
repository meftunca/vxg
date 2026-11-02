# Changelog

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
