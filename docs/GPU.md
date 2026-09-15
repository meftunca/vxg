# Native GPU architecture and boundaries

## Implemented, but SDK/hardware-unvalidated in this delivery

`GpuCodec` owns a native device context. Construction creates a pipeline and queue/stream;
subsequent synchronous calls reuse device buffers. It is deliberately !Send and !Sync.
A native error poisons the context; create a new one rather than continue with uncertain
state. There is no CPU fallback when a requested backend fails.

| Backend | Kernel source | Host implementation | Build/runtime needs |
|---|---|---|---|
| Metal | `native/shaders/vxg.metal` | `native/metal.mm` | macOS, Xcode CLT, Metal device |
| CUDA | `native/shaders/vxg.cuh` | `native/cuda.cu` | Linux, NVCC/toolkit, compatible NVIDIA driver/device |
| Vulkan/SPIR-V | `native/shaders/vxg.comp` | `native/vulkan.cpp` | Linux, Vulkan headers/loader/ICD, glslc or glslangValidator |

The Vulkan feature compiles GLSL to SPIR-V at build time and embeds the actual binary.
There is no fake `.spv` file and no shader-compiler success claim. Vulkan 1.1 is the target.
Only 32-bit integer storage/arithmetic is required by the shared algorithm. Windows native
build and MoltenVK portability enumeration are not implemented; use Metal on macOS.

## Shared algorithm

`scripts/generate_shaders.py` emits MSL, CUDA and GLSL from `core.gpu`. Generated sources
are included, so normal Cargo builds do not require Python. `--check` verifies freshness.
The CPU SIMD implementation is independent and uses the same wire format.

Each workgroup has 64 lanes and handles one independent block. Compression uses 2,048
shared atomic-min hash buckets (8 KiB) to discover deterministic earliest anchors in
parallel. Lane 0 then performs a monotonic greedy parse, writes split control/attribute/
literal scratch streams, chooses RAW when needed, and all lanes pack the output slot.
Collision verification compares actual source bytes, so hash collisions cannot corrupt data.
Earliest-anchor collisions can reduce compression ratio compared with CPU last-position hashing.

Decompression parses each instruction uniformly across participating lanes, cooperatively
copies literals and expands matches via immutable periodic seeds. Every packed output word
has one writing lane per segment tile. Read/write barriers protect partial-word seed reads.
This eliminates intra-run bytewise dependencies but not sequential instruction dependencies.
Two barriers per segment tile and per-lane control parsing are known overheads, not hidden.

## Buffer and batching model

Default batch size is 16 MiB, capped at 256 MiB and rounded down to the frame's block size.
Compression uses B-byte output slots and `4*B+256` bytes scratch per block, plus packed input
and four u32 descriptor words. Approximate device-buffer capacity is therefore around 6x
raw batch size, plus padding/descriptors; it is **not** a low-memory 1x in-place compressor.
Metal uses shared storage; CUDA has device allocations; Vulkan has device buffers plus
host-visible staging and readback buffers. Host copies and retained capacity increase RSS.
A call still builds/returns a complete frame or output Vec; batching bounds device working
set, not the entire caller's host memory consumption.

Encode descriptors return payload size/kind. Rust validates these and the command stream,
computes CRC when selected and packs directory/payloads. Decode first validates the entire
frame on CPU, creates batch-relative input offsets and dispatches. GPU kernels accept only
validated commands. Frame output limits and checked sizing are mandatory before dispatch.

## Backend-specific work

Metal uses runtime MSL compilation, shared MTLBuffers and command-buffer GPU timestamps.
It still copies between caller Vecs and MTLBuffer memory; shared storage is not zero-copy API.
CUDA uses a nonblocking stream, cached cudaMalloc allocations and timing events. Current
host Vecs are pageable; no claim of fully asynchronous pinned transfer overlap. Vulkan
uses device-local buffers, mapped staging, descriptor bindings, push constants, fences,
optional timestamps and explicit transfer/compute/host memory barriers. CPU software Vulkan
can run the algorithm, but its device name is reported and its results are not hardware GPU results.

## Not implemented in v0.1

No tensor-core path, TMA/cp.async specialization, CUDA graphs, pinned double-buffered DMA,
SIMD-group/subgroup-specialized token parser, cross-workgroup pipelining, GPU-resident public
I/O API, multi-GPU scheduler, device-side final variable-size frame compaction, persistent
CPU worker pool, full entropy coding, or automatic per-hardware autotuning. These must be
added selectively after profiling; writing their names into a backend enum is not acceleration.

## Actual-device gate

```sh
cargo run --release --features metal --bin vxg -- gpu-check metal
cargo run --release --features vulkan --bin vxg -- gpu-check vulkan
cargo run --release --features cuda --bin vxg -- gpu-check cuda
```

Each command runs 81 three-way interoperability cases and fails on a mismatch. That is a
smoke/conformance gate, not sufficient for production. Follow it with golden vectors,
long mixed-data corpora, very large match distances, partial last words, multiple batches,
validation layers / device sanitizers, cancellation/error paths and resource-failure tests.
Native SDK compilation and these gates were **not executed** in the preparation environment.
