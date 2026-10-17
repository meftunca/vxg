# Benchmark protocol

For the current numerical target, see [TARGET_4_006_12_15.md](TARGET_4_006_12_15.md).
The fourth axis, CPU energy per GB, has its own [energy protocol](ENERGY_BENCHMARKS.md)
with sustained workloads, idle observations and explicit measurement boundaries.

## VXG v0.4.0 Cross-Platform CI Benchmark Results (`macos-14`, `ubuntu-latest`, `windows-latest`)

Generated automatically via [`.github/workflows/benchmarks.yml`](../.github/workflows/benchmarks.yml) using `vxg-bench --suite v4 --size-mib 16`. Raw CSV and Markdown artifacts are stored in [`reports/ci-v0.4.0/`](../reports/ci-v0.4.0/).

### 1. macOS (`macos-14` / `aarch64`, SIMD=`Neon`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`).

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.909 GB/s** | 0.994 GB/s | 0.598 GB/s | **4.580 GB/s** | 4.582 GB/s | 1.507 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.182 GB/s** | 0.724 GB/s | 0.711 GB/s | **4.649 GB/s** | 2.967 GB/s | 1.738 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.013 GB/s** | 1.082 GB/s | 0.625 GB/s | **7.164 GB/s** | 5.235 GB/s | 1.854 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.288 GB/s** | 1.319 GB/s | 1.057 GB/s | **9.883 GB/s** | 5.622 GB/s | 2.203 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.643 GB/s** | 0.834 GB/s | 0.498 GB/s | **3.501 GB/s** | 1.900 GB/s | 1.097 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **0.832 GB/s** | 1.106 GB/s | 0.504 GB/s | **5.509 GB/s** | 3.879 GB/s | 1.426 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.450 GB/s** | 0.898 GB/s | 0.557 GB/s | **2.971 GB/s** | 2.323 GB/s | 1.119 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2889** | 0.4064 | 0.2933 | **0.253 GB/s** | 0.404 GB/s | 0.399 GB/s | **1.293 GB/s** | 2.592 GB/s | 1.169 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **4.117 GB/s** | 2.914 GB/s | 2.673 GB/s | **8.363 GB/s** | 13.012 GB/s | 3.238 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **4.171 GB/s** | 3.014 GB/s | 2.801 GB/s | **9.983 GB/s** | 6.762 GB/s | 5.538 GB/s |

### 2. Linux (`ubuntu-latest` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.9.4` (`lz4-c-a1`), `libzstd 1.5.5` (`zstd-c-l1`, `zstd-c-l3`).

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **1.040 GB/s** | 1.003 GB/s | 0.578 GB/s | **3.640 GB/s** | 5.424 GB/s | 1.804 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.292 GB/s** | 1.190 GB/s | 0.790 GB/s | **2.806 GB/s** | 4.052 GB/s | 2.120 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.222 GB/s** | 1.448 GB/s | 0.810 GB/s | **3.884 GB/s** | 5.450 GB/s | 2.001 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.652 GB/s** | 1.892 GB/s | 1.075 GB/s | **3.885 GB/s** | 6.759 GB/s | 2.692 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.892 GB/s** | 1.076 GB/s | 0.614 GB/s | **2.301 GB/s** | 3.543 GB/s | 1.550 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.217 GB/s** | 1.510 GB/s | 0.814 GB/s | **4.751 GB/s** | 4.148 GB/s | 1.840 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.760 GB/s** | 1.101 GB/s | 0.572 GB/s | **2.227 GB/s** | 3.165 GB/s | 1.357 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2889** | 0.4064 | 0.2933 | **0.177 GB/s** | 0.518 GB/s | 0.285 GB/s | **1.023 GB/s** | 2.795 GB/s | 1.177 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **2.521 GB/s** | 6.405 GB/s | 3.761 GB/s | **9.652 GB/s** | 11.015 GB/s | 10.615 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **1.506 GB/s** | 5.880 GB/s | 3.915 GB/s | **4.532 GB/s** | 1.987 GB/s | 1.805 GB/s |

### 3. Windows (`windows-latest` / `x86_64`, SIMD=`Avx2`, threads=`1`, block=`65536`)

Baselines: `liblz4 1.10.0` (`lz4-c-a1`), `libzstd 1.5.7` (`zstd-c-l1`, `zstd-c-l3`).

| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.995 GB/s** | 0.898 GB/s | 0.386 GB/s | **3.764 GB/s** | 5.286 GB/s | 1.331 GB/s |
| **`events.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.245 GB/s** | 0.785 GB/s | 0.562 GB/s | **3.223 GB/s** | 4.029 GB/s | 1.592 GB/s |
| **`app-logs.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.091 GB/s** | 0.896 GB/s | 0.577 GB/s | **4.260 GB/s** | 5.311 GB/s | 1.521 GB/s |
| **`json-records.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.557 GB/s** | 1.043 GB/s | 0.541 GB/s | **4.428 GB/s** | 6.503 GB/s | 2.095 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.840 GB/s** | 0.731 GB/s | 0.406 GB/s | **2.971 GB/s** | 3.461 GB/s | 1.177 GB/s |
| **`event-mix-identifiers.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.186 GB/s** | 0.913 GB/s | 0.507 GB/s | **4.830 GB/s** | 4.059 GB/s | 1.436 GB/s |
| **`bursty-variable-records.ndjson`** | 16 MiB | **0.0480** | 0.2065 | 0.1196 | **0.717 GB/s** | 0.801 GB/s | 0.345 GB/s | **2.275 GB/s** | 3.153 GB/s | 1.073 GB/s |
| **`source-and-docs.txt`** | 16 MiB | **0.2848** | 0.3993 | 0.2872 | **0.181 GB/s** | 0.449 GB/s | 0.268 GB/s | **1.042 GB/s** | 2.688 GB/s | 0.943 GB/s |
| **`ui-render-32bpp.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **2.453 GB/s** | 2.191 GB/s | 1.800 GB/s | **9.536 GB/s** | 16.019 GB/s | 10.760 GB/s |
| **`rgb-frame-24bpp.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **1.469 GB/s** | 2.060 GB/s | 1.853 GB/s | **4.536 GB/s** | 1.967 GB/s | 1.756 GB/s |

---

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
