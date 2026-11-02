# Changelog

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
