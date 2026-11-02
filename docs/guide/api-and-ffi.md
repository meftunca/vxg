# Rust API, Compressed-Domain Analytics, Streaming & FFI

VXG provides a unified, zero-allocation API across Rust, C/C++, Python, Go, Node.js/Bun, C#/.NET, and WebAssembly (`wasm32-unknown-unknown`), plus direct compressed-domain query, analytics, mutation, and column projection in Rust.

---

## 1. Core Codec & Zero-Allocation API (`vxg`)

```rust
use vxg::{DecodeOptions, Options};

let opts = Options { threads: 8, ..Options::default() }; // v4, 64 KiB blocks, CRC32C ON
let dec_opts = DecodeOptions { threads: 8, ..DecodeOptions::default() };

// 1. Allocating convenience API
let frame: Vec<u8> = vxg::compress(raw_bytes, &opts)?;
let restored: Vec<u8> = vxg::decompress(&frame, &dec_opts)?;

// 2. Zero-allocation caller-buffer API (`_into`)
let mut comp_buf = vec![0u8; vxg::compress_bound(raw_bytes.len(), opts.block_size)?];
let c_len = vxg::compress_into(raw_bytes, &mut comp_buf, &opts)?;
let mut out_buf = vec![0u8; vxg::frame_raw_size(&comp_buf[..c_len])?];
vxg::decompress_into(&comp_buf[..c_len], &mut out_buf, &dec_opts)?;

// 3. O(1) random-access single-block extraction
let mut blk0 = vec![0u8; vxg::frame_block_raw_size(&frame, 0)?];
vxg::extract_block_into(&frame, 0, &mut blk0, &dec_opts)?;
```

---

## 2. Compressed-Domain Row Access, Mutations, Analytics & Projection

On `GRAMMAR_PACKET` (`0x92`) blocks (NDJSON, CSV, TSV, PSV, logfmt, CLF), VXG evaluates queries, aggregations, mutations, and column projections **directly on compressed column streams** without full-frame decompression:

### 2.1 Row Seek & Pushdown Search (`vxg::query`)

```rust
let total_rows = vxg::frame_row_count(&frame)?;
let row_42 = vxg::extract_row(&frame, 42, &dec_opts)?;
let slice = vxg::extract_row_range(&frame, 100, 150, &dec_opts)?;
let first_err = vxg::find_first(&frame, b"\"level\":\"ERROR\"", &dec_opts)?;
let top_500s = vxg::filter_json_u64(&frame, "status", vxg::NumCmp::Eq(500), 50, &dec_opts)?;
```

### 2.2 Compound Boolean Analytics & `GROUP BY` (`PredExpr`, `aggregate_column`, `group_by`)

```rust
use vxg::{ColPred, ColTarget, NumCmp, PredExpr, StrCmp};

// Zero-heap `Copy` expression tree:
// WHERE status == 500 AND (service == "fraud-engine" OR latency_ms >= 200)
let or_arms = [
    PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"fraud-engine"))),
    PredExpr::col(ColPred::json_num("latency_ms", NumCmp::Ge(200))),
];
let and_arms = [
    PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500))),
    PredExpr::any(&or_arms),
];
let where_expr = PredExpr::all(&and_arms);

let match_count = vxg::count_where(&frame, &where_expr, &dec_opts)?;
let matching_rows = vxg::filter_where(&frame, &where_expr, 100, &dec_opts)?;

// Scalar SUM / MIN / MAX / AVG(bytes) and 2-tier dictionary GROUP BY(service)
let stats = vxg::aggregate_column(&frame, Some(&where_expr), ColTarget::JsonKey("bytes"), &dec_opts)?;
let groups = vxg::group_by(
    &frame,
    Some(&where_expr),
    ColTarget::JsonKey("service"),
    Some(ColTarget::JsonKey("bytes")),
    &dec_opts,
)?;
```

### 2.3 Column Projection & Zero-Reencode Column Splicing (`select_columns`, `project_frame`)

```rust
let cols = [
    ColTarget::JsonKey("ts"),
    ColTarget::JsonKey("service"),
    ColTarget::JsonKey("latency_ms"),
];

// 1. Project directly to NDJSON/CSV bytes (unselected columns are never unpacked!)
let projected_ndjson = vxg::select_columns(&frame, Some(&where_expr), &cols, 1_000, &dec_opts)?;

// 2. Zero-reencode VXG -> VXG column splicing (splices compressed column payloads directly!)
let slim_vxg = vxg::project_frame(&frame, None, &cols, &opts, &dec_opts)?;

// 3. Compressed-domain ORDER BY / Top-K heap query (zone-map skip + bounded K-heap):
let slowest_10 = vxg::top_k_by(
    &frame,
    Some(&where_expr),
    ColTarget::JsonKey("latency_ms"),
    vxg::SortDir::Desc,
    10,
    &dec_opts,
)?;
let slowest_10_ndjson = vxg::select_top_k_by(
    &frame,
    Some(&where_expr),
    &cols,
    ColTarget::JsonKey("latency_ms"),
    vxg::SortDir::Desc,
    10,
    &dec_opts,
)?;
```

### 2.4 Single-Slot Column Update & Map (`filter_frame_where`, `update_column`, `map_column`)

```rust
use vxg::ColOp;

// Filtered VXG frame preserving 100%-matching compressed blocks verbatim:
let err_vxg = vxg::filter_frame_where(&frame, &where_expr, &opts, &dec_opts)?;

// Conditional single-column update (0-hit blocks copied byte-for-byte):
let patched_vxg = vxg::update_column(
    &frame,
    &ColPred::json_num("status", NumCmp::Eq(500)),
    ColTarget::JsonKey("level"),
    ColOp::SetStr(b"CRITICAL"),
    &opts,
    &dec_opts,
)?;

// O(|D|) dictionary map across all rows:
let upper = |b: &[u8]| b.to_ascii_uppercase();
let mapped_vxg = vxg::map_column(&frame, ColTarget::JsonKey("service"), ColOp::MapStr(&upper), &opts, &dec_opts)?;
```

---

## 3. Zero-Copy `mmap` & Bounded-Memory Compressed-Domain Streaming (`v0.4.3`)

Use `MappedFile` / `InputBytes` for zero-copy file mapping, `verify_frame` / `verify_stream` for zero-output-allocation integrity checks, and the `*_stream` family (`select_columns_stream`, `project_frame_stream`, `group_by_stream`, `filter_where_stream`, etc.) for multi-GB pipes and files with `O(single_frame)` RAM:

```rust
use std::io::{Read, Write};
use vxg::{ColTarget, DecodeOptions, InputBytes, Options, StreamDecoder, StreamEncoder};

// 1. Zero-copy read-only mmap + zero-output-alloc parallel verification:
let mapped = InputBytes::open_path("telemetry.vxg")?;
let verified_raw_bytes = vxg::verify_frame(&mapped, &DecodeOptions { threads: 8, ..Default::default() })?;

// 2. Bounded-memory multi-frame streaming codec & compressed-domain projection:
let mut stream_bytes = Vec::new();
{
    let mut enc = StreamEncoder::new(&mut stream_bytes, Options::default())?;
    enc.write_all(b"{\"id\":1,\"ok\":true}\n")?;
    enc.finish()?;
}
let mut projected = Vec::new();
vxg::select_columns_stream(
    stream_bytes.as_slice(),
    &mut projected,
    None,
    &[ColTarget::JsonKey("id")],
    usize::MAX,
    &DecodeOptions::default(),
)?;
```

---

## 4. Cross-Language C ABI & WebAssembly

See the [**Cross-Language FFI Reference**](/FFI) for `include/vxg.h` signatures and zero-copy examples in **C/C++**, **Python (`ctypes`)**, **Go (`cgo`)**, **Bun/Node.js**, **C# / .NET 8+**, and **WebAssembly (`wasm32-unknown-unknown`)**.
