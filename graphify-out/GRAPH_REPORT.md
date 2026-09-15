# Graph Report - vxg  (2026-09-15)

## Corpus Check
- Corpus is ~20,128 words - fits in a single context window. You may not need a graph.

## Summary
- 454 nodes · 850 edges · 31 communities (18 shown, 9 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- CUDA Native Backend
- LZ4 Zstd Baselines
- Vulkan Backend
- GPU API Layer
- Metal Shader
- Frame Checksum
- CUDA Shader
- Public Codec API
- SIMD Engine
- Block Format
- Model Verification
- GPU Emulator
- Benchmark Harness
- Vulkan Buffers
- Property Tests
- C Baselines
- CLI Tool
- Build Packaging
- Roundtrip Example
- Fuzz Package
- Benchmark Script
- Check Script
- Model Check Script
- Formal Script
- Shader Generation
- GPU Check Script
- Manifest Verification

## God Nodes (most connected - your core abstractions)
1. `VulkanContext` - 41 edges
2. `VulkanBuffer` - 17 edges
3. `GpuCodec` - 14 edges
4. `Engine` - 14 edges
5. `CudaContext` - 13 edges
6. `encode_group()` - 13 edges
7. `Simd` - 12 edges
8. `compress_with_stats()` - 11 edges
9. `fixture()` - 11 edges
10. `encode_core()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `roundtrip_bound_and_linear_work_properties()` --calls--> `compress_with_stats()`  [INFERRED]
  tests/properties.rs → src/lib.rs
- `vxg-fuzz` --depends_on--> `vxg`  [EXTRACTED]
  fuzz/Cargo.toml → Cargo.toml
- `compress_with_stats()` --calls--> `max_compressed_size()`  [INFERRED]
  src/lib.rs → src/frame.rs
- `vxg_cuda_create()` --calls--> `vxg_message()`  [INFERRED]
  native/cuda.cu → native/util.hpp
- `vxg_cuda_run()` --calls--> `vxg_check()`  [INFERRED]
  native/cuda.cu → native/util.hpp

## Import Cycles
- None detected.

## Communities (31 total, 9 thin omitted)

### Community 0 - "CUDA Native Backend"
Cohesion: 0.07
Nodes (30): cudaError_t, cudaEvent_t, cudaStream_t, checked(), VxgParams, CudaBuffer, capacity, ptr (+22 more)

### Community 1 - "LZ4 Zstd Baselines"
Cohesion: 0.12
Nodes (32): c_int, Chunk, Codec, Context, decode_group(), encode_group(), Encoded, LZ4_compress_fast_extState() (+24 more)

### Community 2 - "Vulkan Backend"
Cohesion: 0.05
Nodes (37): VulkanContext, command, command_pool, desc, device, download_desc, download_output, family (+29 more)

### Community 3 - "GPU API Layer"
Cohesion: 0.12
Nodes (24): Create, Destroy, PhantomData, Rc, Run, Api, Backend, check_batch() (+16 more)

### Community 4 - "Metal Shader"
Cohesion: 0.15
Nodes (31): constant, device, kernel, CONTEXT, GPU_INLINE, Buffers, desc, input (+23 more)

### Community 5 - "Frame Checksum"
Cohesion: 0.11
Nodes (20): ExactSizeIterator, Item, arm(), crc32c(), software(), x86(), assemble(), Block (+12 more)

### Community 6 - "CUDA Shader"
Cohesion: 0.18
Nodes (28): __global__, CONTEXT, GPU_INLINE, Buffers, desc, input, output, work (+20 more)

### Community 7 - "Public Codec API"
Cohesion: 0.15
Nodes (18): AddAssign, Display, Formatter, WorkStats, Error, Result, String, compress() (+10 more)

### Community 8 - "SIMD Engine"
Cohesion: 0.16
Nodes (14): copy_avx2(), copy_avx512(), copy_neon(), copy_sse2(), Engine, lcp_avx2(), lcp_avx512(), lcp_neon() (+6 more)

### Community 9 - "Block Format"
Cohesion: 0.17
Nodes (15): Command, control(), decode(), encode(), Encoded, inline_token(), Parser, Parser<'a> (+7 more)

### Community 10 - "Model Verification"
Cohesion: 0.28
Nodes (18): ValueError, golden(), main(), Finite-model checks + cross-implementation tests. Does not execute Rust., bound(), Command, commands_payload(), crc32c() (+10 more)

### Community 11 - "GPU Emulator"
Cohesion: 0.33
Nodes (16): barrier, amin(), Buffers, uint, vector, decode(), encode(), launch() (+8 more)

### Community 12 - "Benchmark Harness"
Cohesion: 0.25
Nodes (15): File, Config, datasets(), main(), measure(), next(), quote(), rng() (+7 more)

### Community 13 - "Vulkan Buffers"
Cohesion: 0.15
Nodes (12): VulkanBuffer, buffer, capacity, coherent, device, mapped, memory, VkBuffer (+4 more)

### Community 14 - "Property Tests"
Cohesion: 0.21
Nodes (11): block_random_access_is_independent(), crc32c_known_answer_and_hardware_equivalence(), deterministic_across_threads_and_cpu_dispatch(), every_truncated_prefix_is_rejected(), fixture(), malformed_input_never_panics_under_bounded_allocation(), output_limit_and_checksum_are_enforced(), rng() (+3 more)

### Community 15 - "C Baselines"
Cohesion: 0.32
Nodes (11): Bytes, Chunk, bytes, raw, raw_size, F, vector, decode() (+3 more)

### Community 16 - "CLI Tool"
Cohesion: 0.42
Nodes (8): ExitCode, gpu_check(), main(), R, String, run(), val(), write()

### Community 17 - "Build Packaging"
Cohesion: 0.48
Nodes (6): archive(), exists(), main(), run(), Command, Path

## Knowledge Gaps
- **67 isolated node(s):** `vxg`, `vxg-fuzz`, `ptr`, `capacity`, `device` (+62 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 165 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Simd` connect `SIMD Engine` to `Benchmark Harness`, `Public Codec API`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `Config` connect `Benchmark Harness` to `SIMD Engine`, `GPU API Layer`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `VulkanContext` connect `Vulkan Backend` to `CUDA Native Backend`, `Vulkan Buffers`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **What connects `vxg`, `vxg-fuzz`, `ptr` to the rest of the system?**
  _67 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CUDA Native Backend` be split into smaller, more focused modules?**
  _Cohesion score 0.0743321718931475 - nodes in this community are weakly interconnected._
- **Should `LZ4 Zstd Baselines` be split into smaller, more focused modules?**
  _Cohesion score 0.1241565452091768 - nodes in this community are weakly interconnected._
- **Should `Vulkan Backend` be split into smaller, more focused modules?**
  _Cohesion score 0.05405405405405406 - nodes in this community are weakly interconnected._