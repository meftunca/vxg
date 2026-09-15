# Benchmark protocol

## Do not interpret supplied reference numbers as VXG performance

`reports/reference-c-baselines.csv` was actually generated in the preparation environment
by `verification/c_baselines.cpp`, linked to installed liblz4 1.10.0 and libzstd 1.5.7.
Its `scope` is `C_ONLY_NOT_VXG`. The environment had no Rust compiler or GPU. These rows
check that the reference libraries run; they neither establish VXG speed nor validate
the uncompiled Rust FFI harness. Do not compare the SIMT emulator speed with native LZ4.

The real comparative harness is `src/bin/bench.rs` + `src/baselines.rs`. It links reference
C libraries rather than reimplementations with potentially different performance.

## CPU comparison profiles

All codecs use identical uncompressed input, independent block size, worker count, RAW
fallback and checksum-off policy. Included configurations are VXG acceleration 1,
LZ4 fast accelerations 1 and 4, and Zstd levels -3, 1 and 3. This is not a comparison to
a single whole-file Zstd frame with cross-block history; that would be a separate workload.

Compression includes allocating the encoded result, scratch/context creation per worker,
actual block coding, RAW decisions and sorting. VXG additionally serializes its contiguous
frame; reference codecs retain vectors of blocks rather than serialize a real frame.
Their reported size adds the same `32+32*block_count` envelope for accounting. This is
close application-level coverage but **not identical memory-allocation/framing work**.
Do not publish it as an apples-to-apples intrinsic-kernel microbenchmark.

Decompression starts with prepared metadata and preallocated output for all CPU codecs.
VXG header/directory parsing is outside the timed loop; its command parsing remains inside.
Reference blocks already carry size/RAW metadata. Zstd decode context setup is per call,
not hidden in a global cache. Multithreaded calls include scoped-thread creation/join;
there is no persistent thread pool in this version.

Inputs/buffers are reused (warm-cache workload). Two warmups are excluded. Each of five
default samples runs for at least 100 ms, recording mean ns/operation within that sample;
CSV reports the median/min/max of those samples. These are not per-request p95/p99 latencies.
GB/s means decimal uncompressed bytes/second for both compression and decompression.
Ratio means `encoded_bytes / raw_bytes`; lower is smaller. Correctness uses full byte
comparison before timing and after decode timing, not just equal lengths/checksums.

## Run

```sh
# macOS: brew install lz4 zstd
# Ubuntu: sudo apt-get install liblz4-dev libzstd-dev
cargo run --release --features bench --bin vxg-bench -- \
  --threads 1 --block-size 65536 --size-mib 16 --samples 7 --seconds 0.2 \
  --csv reports/cpu-1-thread.csv

cargo run --release --features bench --bin vxg-bench -- \
  --threads 4 --size-mib 64 --csv reports/cpu-4-thread.csv

# Real workload; repeated --file accepts multiple files without concatenating them:
cargo run --release --features bench --bin vxg-bench -- \
  --file corpus/logs.ndjson --file corpus/database-pages.bin \
  --csv reports/real-corpus.csv

# Explicit ISA ablation; unsupported ISA returns an error:
cargo run --release --features bench --bin vxg-bench -- --simd scalar --csv reports/scalar.csv
cargo run --release --features bench --bin vxg-bench -- --simd avx512 --csv reports/avx512.csv
```

Built-in synthetic classes: zero runs, period-37, changing JSON records, pseudo-random,
low-alphabet pseudo-random, and 4 KiB mixed segments. Their results are not representative
of every corpus. Supply existing compressed/media/encrypted files to test RAW decisions.
No external corpus is bundled or downloaded automatically.

## GPU comparison profile

```sh
cargo run --release --features bench,metal --bin vxg-bench -- \
  --gpu metal --size-mib 64 --csv reports/metal-e2e.csv
# On the corresponding Linux hardware, substitute bench,cuda / cuda
# or bench,vulkan / vulkan.
```

GPU rows are `gpu_host_e2e_allocating`: CPU validation, framing, copies, device transfer,
allocation/resizing, kernel, synchronization and output assembly are included. Context
and pipeline creation happen before timing. CUDA/Vulkan discrete-device transfer and Metal
shared-buffer copies are not silently omitted. A device-timestamp diagnostic is printed
separately, not as a median or the comparison denominator. There is not yet a device-resident
public benchmark API or preallocated GPU decode API. CPU and GPU profiles therefore differ;
report both with their labels, do not claim pure kernel speed from their ratio.

Correctness covers GPU-encode/CPU-decode, CPU-encode/GPU-decode, and GPU self-decode. The
strict host command validator stays enabled for GPU decompression; do not remove it solely
to inflate results on untrusted input. CRC is off for comparisons but on in normal CLI use.

## Before publishing a win

Record CPU/GPU model, OS, compiler flags, library versions, power/thermal state, core
assignment, frequency policy, device/driver, input checksums and bytes. Repeat on cold/large
working sets as well as warm ones. Measure process and thread-pool variants separately.
Measure actual bandwidth and allocation, not vendor peak bandwidth. Prefer cycles/byte,
branch misses, memory traffic, kernel occupancy and end-to-end latency alongside GB/s.
Zstd ratio is allowed to be better; do not discard those rows. No cherry-picked average over
only compressible synthetic inputs, no single-core-vs-many-core headline, and no CPU fallback
labeled as a GPU result.
