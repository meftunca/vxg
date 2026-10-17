# Changelog

## 0.4.0 — Unified Adaptive v4 Release Pipeline

- **Format v4 (`VERSION_V4 = 4`):** Compact 8-byte independent block directory (`u32 (payload_size | kind << 30)` + `u32 crc32c`) with full backward compatibility for decoding `v1`–`v3` frames.
- **Adaptive Block Router (`BLOCK_COLUMN` / `BLOCK_RAW` / `BLOCK_LZ`):**
  - **Structured Columnar Engine:** Automatic schema and delimiter discovery for NDJSON, CSV, TSV, and Logfmt streams with specialized column models (`Dictionary`, `Decimal`, `DecimalDelta`, `DecimalSuffix`, `Timestamp` via SWAR+NEON, `EpochCounter` via NEON hex vectorization, `SegmentDictionary`, `FrontCodedDict`, `RawLenHuffman`).
  - **2D/Stride Image Engine:** BMP header and multi-stride (`stride = 1, 2, 3, 4, 8`) vertical/horizontal delta + RGB/RGBA decorrelation (`G`, `R-G`, `B-G`, `A`) with 4-stream canonical Huffman coding.
  - **Fast-LZ + 4-Stream Canonical Huffman:** High-throughput independent-block matcher with const-generic specializations and 5-symbol tree-reduced bit packing for source code, documentation, and unstructured binary payloads.
- **SIMD & SWAR Micro-Architecture:** 32-byte branchless NEON string scanning, call-free `u128` slice comparison, branchless SWAR decimal parsing, and 8-way/4-way tree-reduced bit packing.
- **CLI & Cross-Platform Benchmark Suite:** Full `vxg` CLI (`compress`, `decompress`, `verify`, `block`, `info`, `hardware`, `stdin`/`stdout` `-`, `--threads auto`) and built-in `--suite v4 --markdown` benchmark suite running across Linux, macOS, and Windows in GitHub Actions.

## 0.1.0 — experimental source delivery

First VXG1 wire format, Rust CPU/SIMD implementation, native Metal/CUDA/Vulkan sources,
reference C-library benchmark harness, cross-language shared kernel, independent model,
golden/error vectors, fuzz/proof harnesses and explicit validation ledger. No established
compatibility guarantee, Rust build-success claim, GPU hardware validation or speedup claim.
