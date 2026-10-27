# Rust API, Streaming (`Read`/`Write`) & Cross-Language FFI

VXG provides a unified, zero-allocation API across Rust, C/C++, Python, Go, Node.js/Bun, C#/.NET, and WebAssembly (`wasm32-unknown-unknown`).

---

## 1. Rust In-Memory API (`vxg`)

### 1.1 Allocating Convenience API

```rust
use vxg::Options;

let opts = Options::default(); // v4, 64 KiB blocks, CRC32C enabled, 1 thread
let compressed: Vec<u8> = vxg::compress(raw_bytes, &opts)?;
let restored: Vec<u8> = vxg::decompress(&compressed, &opts)?;
```

### 1.2 Zero-Allocation Caller-Buffer API (`_into`)

For high-throughput pipelines (`> 10 GB/s`), reuse caller-allocated buffers to eliminate per-call `Vec` allocations and page faults:

```rust
use vxg::Options;

let opts = Options {
    threads: 8,
    ..Options::default()
};

// 1. Pre-allocate compression buffer once using exact worst-case bound
let max_compressed = vxg::compress_bound(raw.len(), opts.block_size)?;
let mut comp_buf = vec![0u8; max_compressed];
let comp_len = vxg::compress_into(raw, &mut comp_buf, &opts)?;
let compressed = &comp_buf[..comp_len];

// 2. Inspect exact uncompressed size in O(1) time from the 32-byte header
let raw_len = vxg::frame_raw_size(compressed)?;
let mut out_buf = vec![0u8; raw_len];

// 3. Decompress directly into `out_buf` (even across 8 threads!)
let written = vxg::decompress_into(compressed, &mut out_buf, &opts)?;
assert_eq!(written, raw.len());
```

### 1.3 $O(1)$ Random-Access Single-Block Extraction

Every block in a VXG frame is independent. You can query the block count and extract any single block `idx` without decoding the rest of the frame:

```rust
let num_blocks = vxg::frame_block_count(compressed)?;
let blk_size = vxg::frame_block_raw_size(compressed, 0)?;
let mut blk_buf = vec![0u8; blk_size];

let n = vxg::extract_block_into(compressed, 0, &mut blk_buf, &opts)?;
```

### 1.4 Compressed-Domain Structured Record Random Access & Pushdown Search

When compressing structured record streams (NDJSON, CSV, TSV, logfmt, CLF logs), VXG's `GRAMMAR_PACKET` (`0x92`) stores exact row counts per block and vectorized column streams. You can query row counts, seek to individual rows or ranges (`[start_row..end_row)`), and run `find_first` / `filter_rows` directly in the compressed domain with **256-bit block alphabet pruning**, **exact interval-alignment dictionary pushdown**, and **$O(1)$ `PreparedTemplate::skip_rows`**:

```rust
use vxg::{DecodeOptions, RowMatch};

let dec_opts = DecodeOptions::default();

// 1. Count total records across all blocks without unpacking column streams
let total_rows: usize = vxg::frame_row_count(compressed)?;

// 2. O(1) block seek + O(1) row skip to extract a single record or range
let row_50k: Vec<u8> = vxg::extract_row(compressed, 50_000, &dec_opts)?;
let slice: Vec<u8> = vxg::extract_row_range(compressed, 50_000, 50_010, &dec_opts)?;

// 3. Compressed-domain predicate pushdown search (findFirst & filter)
if let Some(hit) = vxg::find_first(compressed, b"status=500", &dec_opts)? {
    println!("First match at global row {} (block {}): {}",
        hit.global_row_index, hit.block_index, String::from_utf8_lossy(&hit.bytes));
}

let top_errors: Vec<RowMatch> = vxg::filter_rows(compressed, b"ERROR", 100, &dec_opts)?;
```

---

## 2. Streaming `std::io::Write` & `std::io::Read` API

When compressing or decompressing arbitrarily large streams (network sockets, pipes, multi-GB log files) with bounded memory, use `StreamEncoder<W>` and `StreamDecoder<R>`:
- `StreamEncoder<W>` buffers up to `chunk_size` (default `1 MiB`, rounded to `block_size`) and flushes a self-contained VXG frame per chunk.
- `StreamDecoder<R>` transparently reads concatenated VXG frames back-to-back.

```rust
use std::io::{Read, Write};
use vxg::{Options, StreamDecoder, StreamEncoder};

let opts = Options::default();
let mut stream_bytes = Vec::new();

// Incremental streaming encode
{
    let mut enc = StreamEncoder::new(&mut stream_bytes, opts.clone());
    enc.write_all(b"first chunk of records\n")?;
    enc.write_all(b"second chunk of records\n")?;
    enc.finish()?;
}

// Incremental streaming decode (also works on single-frame .vxg files)
let mut dec = StreamDecoder::new(stream_bytes.as_slice(), opts);
let mut restored = Vec::new();
dec.read_to_end(&mut restored)?;
```

---

## 3. Cross-Language C ABI & WebAssembly

VXG compiles simultaneously as a static library (`libvxg.a` / `vxg.lib`), shared library (`libvxg.so` / `libvxg.dylib` / `vxg.dll`), and dependency-free WebAssembly binary (`vxg.wasm`).

See the complete [**Cross-Language FFI Reference**](/FFI) for:
- Full `include/vxg.h` function signatures and error codes
- Ready-to-run examples for **C/C++**, **Python (`ctypes`)**, **Go (`cgo`)**, **Node.js / Bun (`bun:ffi`)**, **C# / .NET 8+ (`LibraryImport`)**, and **WebAssembly (`wasm32-unknown-unknown`)**.
