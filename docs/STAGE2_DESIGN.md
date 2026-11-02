# Compressed-Domain Engine (Stages 2–6): `filter`, `Top-K`, `map`, `update`, `agg`, `group_by`, Zero-Reencode `project`, and Bounded-Memory Streaming

Status: **Implemented & Verified (`vxg` v0.4.4, wire format `v4` — zero wire-format changes)**.

<CompressedOpsChart />

## Overview

VXG v4 encodes structured blocks (`BLOCK_COLUMN` / `GRAMMAR_PACKET`, 64 KiB independent blocks by default) as `1..=16` static segment templates plus independent per-slot column streams (`MODEL_DICTIONARY`, `MODEL_FRONT_DICT`, `MODEL_DECIMAL_*`, `MODEL_TIMESTAMP_ISO_Z`, `MODEL_HEX_SUFFIX`, `MODEL_EPOCH_COUNTER`, `MODEL_PERIODIC_PLAIN`, `MODEL_SEGMENT_DICT`, `MODEL_AFFINE_GF2`, `MODEL_PATTERN`, and raw/Huffman/LZ streams) alongside an explicit `Row::Raw` bitmap for irregular or block-boundary-split lines.

Stage 2 executes column-addressed `filter`, `map`, and `update` directly on `VXG1` frames without full-frame decompression or full-block re-encoding:

1. **Zero-unpack header rejection** (`src/columns/ops_pred.rs`, `src/columns/query_numeric.rs`):
   - **256-bit column alphabet mask (`col_bytes_alphabet_mask`)**: extracts exact per-column byte alphabets from `MODEL_DICTIONARY`, `MODEL_FRONT_DICT` (modes `0..=3`), `MODEL_PATTERN`, Huffman headers, and raw/palette tables in `O(header)`, skipping non-matching blocks before unpacking a single row.
   - **Numeric zone-map bounds (`can_numeric_col_match_fast`)**: derives `[min..=max]` directly from `MODEL_DECIMAL_PACKED` (`min`, `bit_width`) and `MODEL_DECIMAL_AFFINE` (`first`, `step`, `ones`), rejecting out-of-range `Eq` / `Ne` / `Lt` / `Le` / `Gt` / `Ge` / `Range` predicates in `O(1)`.
2. **Single-column predicate evaluation (`eval_col_pred_on_slot`)**:
   - Resolves `ColTarget::JsonKey(key)` and `ColTarget::Delimited { delim, index }` to an exact template slot across all `1..=16` templates (`resolve_template_slot`).
   - Evaluates `ConstValue` in `O(1)`, `Dictionary` in `O(|D|) + O(ones)` (scanning compact dictionary IDs after testing each unique dictionary entry once), numeric streams directly on integer arenas without formatting ASCII digits, and general columns by materializing only the addressed slot.
   - Other columns in a `GRAMMAR_PACKET` block are decoded **only** for blocks with `>= 1` matching row.
3. **Single-slot splice & `O(|D|)` dictionary rewrite (`src/columns/ops_rewrite.rs`)**:
   - **0-hit blocks**: copied byte-for-byte (`BytesRef::Borrowed`) from the input frame into the output frame, preserving existing `CRC-32C` checksums when present.
   - **Modified `GRAMMAR_PACKET` blocks (`1..=16` templates)**: only the target slot in affected templates is rewritten (`rewrite_single_slot_column`) and re-packed via `pack_best_model_into` (emitting canonical sorted dictionaries, bit-packed decimals, pattern streams, etc.); all other column payloads, template headers, row bitmaps, and `Row::Raw` streams are spliced byte-for-byte.
   - **Unconditional `map_column` fast paths**: `ConstValue` transforms in `O(1)` and `MODEL_DICTIONARY` transforms in `O(|D|)` unique entries before re-canonicalizing.
   - **Token-validity guard (`is_valid_slot_token`)**: if a new token introduces bytes that alter template delimiter/quote boundaries (e.g. unquoted commas in CSV or newlines), that block cleanly falls back to raw block re-encode so decoded bytes remain 100% identical to a line-by-line raw transform.
4. **Exact multi-block split-line & CSV column-0 cut handling (`src/ops/transform.rs`, `src/columns/ops_header.rs`)**:
   - Pre-scans `Row::Raw` block boundaries (`peek_grammar_boundary`) without unpacking columnar data, joining lines split across 2 or more 64 KiB block boundaries (`carry` chains) and evaluating predicates/mutations on the complete logical line.
   - **CSV/TSV column-0 cut invariant**: if a 64 KiB block boundary cuts inside column 0 of a delimited record, the suffix in block `b + 1` still has the same delimiter count and may be encoded as `Row::Template` (`first_is_raw == false`). Whenever `had_incoming_carry && !first_is_raw`, all compressed-domain paths (`query_search`, `ops_pred`, `ops_header`, `ops_rewrite`) detect this and route that block through raw carry-join so suffix fragments are never mistaken for column 0.
5. **Canonical `VXG1` v4 frame emission (`src/ops/transform.rs`, `Frame::parse_all`)**:
   - When non-final blocks preserve `raw_size == block_size` (all 0-hit updates, fixed-width updates, or single-block outputs), a single `VXG1` v4 frame is emitted.
   - When variable-length mutations change an interior block's uncompressed byte size without re-chunking neighboring untouched blocks, contiguous valid block runs are emitted as minimal concatenated canonical `VXG1` v4 frames, supported natively across `decompress`, `decompress_into`, `decompress_stream`, `Frame::parse_all`, `frame_row_count`, `extract_row`, `extract_row_range`, `find_first`, `filter_rows`, `filter_column`, `filter_frame`, `update_column`, `map_column`, `vxg verify`, and `vxg info`.

## Rust API (`vxg` / `vxg::ops`)

```rust
use vxg::{
    filter_column, filter_frame, map_column, update_column,
    ColOp, ColPred, ColTarget, DecodeOptions, NumCmp, Options, StrCmp,
};

let enc = Options::default();
let dec = DecodeOptions::default();

// 1. Column-addressed row query (returns Vec<RowMatch> without full decode)
let hits = filter_column(
    &frame,
    &ColPred::json_num("latency_ms", NumCmp::Ge(200)),
    usize::MAX,
    &dec,
)?;

// 2. Compressed-domain frame filter (returns a new canonical VXG1 frame)
let err_frame = filter_frame(
    &frame,
    &ColPred::json_num("status", NumCmp::Eq(500)),
    &enc,
    &dec,
)?;

// 3. Conditional compressed-domain column update (untouched blocks copied byte-for-byte)
let updated = update_column(
    &frame,
    &ColPred::json_num("status", NumCmp::Eq(500)),
    ColTarget::JsonKey("level"),
    ColOp::SetStr(b"CRITICAL"),
    &enc,
    &dec,
)?;

// 4. Unconditional compressed-domain column map (O(|D|) on dictionary columns)
let upper_fn = |b: &[u8]| b.to_ascii_uppercase();
let mapped = map_column(
    &frame,
    ColTarget::JsonKey("service"),
    ColOp::MapStr(&upper_fn),
    &enc,
    &dec,
)?;
```

## CLI Usage (`vxg`)

```sh
# Column-addressed query & count (--where supports json:<key>, csv:<idx>, tsv:<idx>, psv:<idx>)
vxg query logs.vxg --where json:latency_ms>=200 --limit 50
vxg query logs.vxg --where csv:2=prefix:fraud --count

# Produce a new filtered VXG frame
vxg filter-frame logs.vxg errors.vxg --where json:status=num:500

# Conditional column update without full decode/re-encode
vxg update logs.vxg patched.vxg --where json:status=num:500 --col json:level --set CRITICAL

# Whole-column map (--upper, --lower, --redact, --set, --set-raw, --add)
vxg map logs.vxg upper.vxg --col json:service --upper
vxg map logs.vxg bumped.vxg --col json:bytes --add 100
```

## Measured Performance & Differential Verification

Measured single-threaded on Apple Silicon (`aarch64` NEON, `block_size = 65536`, checksums off, median of 5 timed runs after warmup via `vxg-query-real --ops-suite`). Every run asserts exact byte-for-byte equality against `decompress + raw transform (+ compress)`:

- **Varied NDJSON (`56.5 MB`, `300,000` rows, `863` blocks, ratio `0.091`)**:
  - `filter_column` (`level == "NOPE"`, 0 hits): **1.43 ms** (`39.51 GB/s`, **72.5×** vs `decompress + scan`)
  - `filter_column` (`bytes >= 50000`, 6 hits, zone-map): **2.11 ms** (`26.77 GB/s`, **59.0×**)
  - `filter_column` (`latency_ms >= 200`, 563 hits): **4.54 ms** (`12.43 GB/s`, **28.7×**)
  - `filter_frame` (`status == 500` -> new `VXG1` frame): **13.21 ms** (`4.28 GB/s`, **10.2×** vs `decompress + filter + compress`)
  - `update_column` (`WHERE level == "NOPE" SET CRITICAL`, 0 hits): **4.08 ms** (`13.84 GB/s`, **52.2×**)
  - `update_column` (`WHERE latency_ms >= 250 SET CRITICAL`): **6.51 ms** (`8.69 GB/s`, **35.2×**)
  - `update_column` (`WHERE status == 500 SET CRITICAL`, ~5% rows): **11.55 ms** (`4.89 GB/s`, **22.6×**)
  - `map_column` (`service -> uppercase`, `O(|D|)` dictionary): **8.44 ms** (`6.70 GB/s`, **30.5×**)
  - `map_column` (`bytes -> bytes + 100`, 1-col numeric): **28.74 ms** (`1.97 GB/s`, **8.4×**)
- **8-Thread Deterministic Work-Stealing (`threads = 8`, `ndjson_varied.ndjson`, `56.5 MB`)**:
  - `filter_column` (`latency_ms >= 200`): **1.88 ms** (`30.04 GB/s`, **43.8×**)
  - `filter_frame` (`status == 500` -> `VXG1`): **4.24 ms** (`13.32 GB/s`, **19.1×**)
  - `update_column` (`WHERE status == 500 SET CRITICAL`): **3.20 ms** (`17.66 GB/s`, **34.5×**)
  - `map_column` (`service -> uppercase`): **3.27 ms** (`17.27 GB/s`, **31.5×**)
  - `map_column` (`bytes -> bytes + 100`): **5.51 ms** (`10.25 GB/s`, **19.5×**)
  - Produces **100% byte-for-byte identical** `RowMatch` vectors and `VXG1` frame bytes across `threads = 1..=256`.
- **Differential Stress Suites (`tests/compressed_ops_stress.rs`, `tests/stage3_analytics.rs`, `tests/stage4_projection_cli.rs`)**:
  - Verified across multi-template NDJSON, CSV, TSV, PSV, split column-0 boundaries, multi-block (`> 2` blocks) long lines, EOF without trailing newline, JSON escapes, CSV quotes, column reordering/duplication, and `FAST_LZ`/`RAW` fallbacks.

## Stage 3 & Stage 4 Extensions (`v0.4.2`)

1. **Compound Boolean Expressions (`PredExpr`, `src/ops/expr.rs`, `src/columns/ops_expr.rs`)**:
   - Zero-heap `Copy` `PredExpr` (`True`, `Col`, `RowContains`, `All`, `Any`, `Not`) evaluated over 64-bit `TplMask` word arrays with short-circuit leaf pruning (`count_where`, `filter_where`, `filter_frame_where`).
2. **Compressed-Domain Scalar Aggregations & 2-Tier `GROUP BY` (`src/ops/agg.rs`, `src/columns/ops_agg.rs`, `src/columns/ops_group.rs`)**:
   - `aggregate_column` computes `AggStats` (`row_count`, `count`, `sum`, `min`, `max`, `avg`) directly on packed numeric/dictionary arenas (**15.1 GB/s** `1T`, **49.4×** faster than naive decode + scan).
   - `group_by` accumulates per-dictionary-ID `[AggStats; |D|]` buckets in L1 cache before mapping unique dictionary entries to global `GroupKey`s (**13.9 GB/s** `1T`, **48.5×** speedup on `8T` `SUM(bytes) GROUP BY service`).
3. **Column Projection & Zero-Reencode Column Splicing (`src/ops/project.rs`, `src/columns/ops_project.rs`)**:
   - `select_columns` constructs a miniature `PreparedTemplate` over only the requested `ColTarget` slots, skipping unselected columns completely (up to **20.7 GB/s** / **37.3×** speedup).
   - `project_frame` splices compressed column payloads byte-for-byte into a new `GRAMMAR_PACKET` (`GrammarProjectOut::Spliced`, absorbing `64 KiB` split boundary lines as 1-row raw tails) at **10.7 GB/s** (**32.6×** faster than `decompress + project + compress`).

## Stage 5 Extensions: Zero-Copy `mmap` & Bounded-Memory Streaming (`v0.4.3`)

1. **Zero-Dependency Cross-Platform `mmap` (`src/mmap.rs`, `src/mmap/sys.rs`)**:
   - `MappedFile`, `MappedFileMut`, and `InputBytes` provide zero-copy virtual memory views with kernel page-cache hints (`MADV_SEQUENTIAL`, `MADV_RANDOM`, `MADV_WILLNEED`) and zero external dependencies.
2. **Zero-Output-Allocation Verification (`verify_frame`, `verify_stream`, `src/stream_ops/verify.rs`)**:
   - Validates all directory invariants, command/grammar streams, and `CRC32C` checksums into a thread-local `64 KiB` L2-resident scratch buffer (`12.0+ GB/s` on `8T` without allocating `raw_size`).
3. **Cross-Frame Split-Line Boundary Stitcher (`FramePeeler`, `src/stream_ops.rs`)**:
   - Peels any unterminated tail record from frame $k$ and prepends it as a 1-block `BLOCK_RAW` prefix ahead of frame $k+1$'s borrowed `GRAMMAR_PACKET` blocks, enabling `O(single_frame)` bounded-memory execution across all 12 `*_stream` compressed-domain operations while keeping interior blocks 100% compressed.

## Stage 6 Extensions: Compressed-Domain `ORDER BY` / `Top-K` & Expanded C/FFI ABI (`v0.4.4`)

1. **Bounded `Top-K` Heap & Numeric Zone-Map Pruning (`src/ops/top_k.rs`, `src/columns/ops_topk.rs`, `src/stream_ops/top_k.rs`)**:
   - `top_k_by`, `select_top_k_by`, `top_k_by_stream`, and `select_top_k_by_stream` maintain a bounded $K$-entry `TopKHeap` (`NULLS LAST`, deterministic `global_row_index` tie-breaking across threads).
   - Inspects `MODEL_DECIMAL_PACKED` / `MODEL_DECIMAL_AFFINE` headers in $O(1)$ to skip entire blocks whose `[min..=max]` cannot beat the running `AtomicU64` cross-worker threshold, scans dictionary columns via a pre-pruned `alive` entry bitmap, and bounds per-template candidate tracking (`CandTracker`) to $\le K$ rows before materializing only winning rows (**35.6–42.3 GB/s**, **54×–72×** faster than `decompress` + `select_nth_unstable_by` on `8T`).
2. **Shared Expression Parser & Expanded Zero-Overhead C ABI (`src/ops/parse_expr.rs`, `src/ffi/analytics.rs`, `include/vxg.h`)**:
   - Exposes `vxg_verify_frame`, `vxg_compress_file`, `vxg_decompress_file`, `vxg_verify_file`, `vxg_count_where_str`, `vxg_aggregate_col` (`vxg_agg_stats_t`), `vxg_select_alloc` (`SELECT` + `WHERE` + `ORDER BY` Top-K), and `vxg_project_frame_alloc` across C/C++, Python, Go, Bun/Node.js, C#, and WebAssembly.
