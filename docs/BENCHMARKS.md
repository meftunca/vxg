# Benchmark protocol

For the current numerical target, see [TARGET_4_006_12_15.md](TARGET_4_006_12_15.md).
The fourth axis, CPU energy per GB, has its own [energy protocol](ENERGY_BENCHMARKS.md)
with sustained workloads, idle observations and explicit measurement boundaries.

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

The primary CPU decode row is `cpu_preallocated_decode`: all codecs start with prepared
metadata and a preallocated output buffer. VXG header/directory parsing is outside that
timed loop; its command parsing remains inside. `cpu_full_frame_preallocated_decode` is
an additional VXG-safe row which includes header/directory parsing before the same checked
decode. `cpu_prepared_validate_commands` measures one-time command validation on an
already parsed directory; `cpu_parse_validate_commands` includes both parse and validation.
They are accounting rows, not substitutes for decode. `cpu_prevalidated_decode` exists
only with `--validated-decode`, after a `ValidatedFrame` has obtained its immutable command
proof outside timing; never compare that repeated-decode-only profile to LZ4's one-shot
safe decode without reporting the one-time validation cost. Reference blocks already carry
size/RAW metadata. Zstd decode context setup is per call, not hidden in a global cache.
Multithreaded calls include scoped-thread creation/join; there is no persistent thread pool
in this version.

Inputs/buffers are reused (warm-cache workload). Two warmups are excluded. Each of five
default samples runs for at least 100 ms, recording mean ns/operation within that sample;
CSV reports the median/min/max of those samples. `--samples-csv FILE` retains every timed
sample (sorted by duration) with the same profile labels. These are not per-request p95/p99 latencies.
GB/s means decimal uncompressed bytes/second for both compression and decompression.
Ratio means `encoded_bytes / raw_bytes`; lower is smaller. Correctness uses full byte
comparison before timing and after decode timing, not just equal lengths/checksums.

## Run

```sh
# macOS: brew install lz4 zstd
# Ubuntu: sudo apt-get install liblz4-dev libzstd-dev
cargo run --release --features bench --bin vxg-bench -- \
  --threads 1 --block-size 65536 --size-mib 16 --samples 7 --seconds 0.2 \
  --csv reports/cpu-1-thread.csv --samples-csv reports/cpu-1-thread-samples.csv

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

## Frozen Q95 qualification

The long-running 4 / 95 / 15 target is evaluated on a frozen, synthetic structured
JSON/log set: existing `json-records` plus three held-out families at both 16 and 64 MiB.
The generated corpus is ignored, but [the tracked manifest](../reports/qualification/manifest.json)
records its workload rule, byte count and SHA-256. Regenerate and verify it with:

```sh
python3 scripts/generate_qualification_corpus.py
python3 scripts/generate_qualification_corpus.py --verify
scripts/run_qualification.sh reports/qualification/runs/P0-baseline
```

A second frozen revision, Q95-r2, applies the 2026-09-26 scope revision
([E_B_FRONTIER_REVIEW.md](E_B_FRONTIER_REVIEW.md), [TARGET_4_006_12_15.md](TARGET_4_006_12_15.md)):
only the `event-mix-identifiers` and `bursty-variable-records` value sources change
(epoch+counter identifiers; bounded phrase-vocabulary messages). The other two families
regenerate byte-identical to r1, which remains as an adversarial control set.

```sh
python3 scripts/generate_qualification_corpus.py --r2
python3 scripts/generate_qualification_corpus.py --r2 --verify
MANIFEST=reports/qualification/manifest-r2.json scripts/run_qualification.sh OUTPUT_DIRECTORY
```

The runner is serial: it builds one benchmark binary, records its hash and host details,
then preserves an aggregate CSV, all timed samples and a console log per input/process.
It is not a claim that synthetic Q95 represents arbitrary production data. A candidate
promotion requires paired shuffled process order, confidence intervals and every frozen
member; the initial P0 runner records the baseline and does not discard a slow process.

Before changing the wire format, inspect the actual emitted commands—not only encoder
counters—with the offline analyzer. It compresses once with the stated options, parses the
resulting frame, and writes no timed benchmark rows:

```sh
cargo run --release --features bench --bin vxg-analyze -- \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/qualification/experiments/P2-command-analysis/json-records-16MiB.json
```

The report includes exact command histograms, grouped residual widths, cache-hit models and
dependency counts. Its cache model is diagnostic: it does not establish a compressed format
until an exact pack/unpack prototype includes selectors, bases, exceptions and padding.

P3 provides those lossless scalar prototypes without changing the encoder or public frame
format. It captures every logical command field, including pattern seeds, and requires a
pack/unpack roundtrip before writing a report. The resulting rates are metadata-only
commands/s and metadata-bytes/s—not complete-codec GB/s—and its prototype totals are labelled
as a model rather than a VXG v3 bitstream:

```sh
cargo run --release --features bench --bin vxg-pack-eval -- \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/qualification/experiments/P3-packing-r2/json-records-16MiB.json \
  --iterations 1 --threads 1 --block-size 65536
```

It evaluates fixed-width with exceptions, exact bitsliced transpose, lossless int4 escapes,
2-bit/base-3 ternary selectors, and standalone exact distance alternatives. The distance rows
are a substream experiment only; their size cannot be substituted for a full-frame ratio.

P5's bounded entropy experiment is also an offline oracle. It writes canonical per-LZ-block
Huffman tables and an MRU-8 distance tag stream, then proves an exact unpack before reporting
its bytes. The adaptive form builds both entropy and fixed-128 streams, charges both builders,
and adds a one-byte selector per LZ block; it is not a `compress` format or a codec GB/s claim:

```sh
cargo run --release --features bench --bin vxg-pack-eval -- \
  --input corpus/qualification/json-records-16MiB.ndjson \
  --output reports/qualification/experiments/P5-adaptive-entropy-r1/json-records-16MiB.json \
  --iterations 3 --threads 1 --block-size 65536
```

The full Q95 reports show an adaptive JSON-only improvement (0.113659 on JSON-16) but no
held-out ratio improvement beyond fixed-128's selector-byte cost. It is retained as a costed
negative result; `docs/TARGET_4_95_15_PLAN.md` defines the next literal and matcher gates.

The same evaluator also contains the P5 literal oracle. It extracts validated v2 literal streams,
builds raw and canonical byte-Huffman payloads for every LZ block and charges a selector, tables,
bit lengths and padding. Its total retains existing v2 command/prefix bytes and replaces only
the literal payload. All-Q95 evidence in `P5-literal-entropy-r1` rejects it as a wire candidate:
the compressed bursty/event literal payloads alone remain above the 5% artifact cap. Its scalar
literal bytes/s rate is not a VXG encode/decode GB/s result.

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
