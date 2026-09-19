# VXG development progress

This file records implementation steps, validation gates, and benchmark evidence.
Every implementation or measurement change is recorded here as a separate step.

## 2026-09-15 — Git baseline

- Baseline commit: `b5d4395` (`chore: initialize VXG repository`)
- Scope: VXG 0.1.0 source, tests, native backends, verification scripts, and retained reports.
- At baseline creation, no source behavior had changed after the imported package snapshot.

## Starting evidence

- Default release tests: 15 tests/doctests passed in an isolated Cargo target.
- Metal-feature release tests: passed; no Metal device was available in the current execution environment.
- CPU hardware selection on the current Apple arm64 host: `Auto -> Neon`.
- SIMD implementation currently covers LCP, non-overlapping copy, pattern fill, and overlap-safe repeat.
- Hashing, probing, candidate selection, command emission, and most frame work remain scalar.
- Current CPU benchmark protocol includes allocation/framing and is not an intrinsic-kernel comparison.
- Existing scalar-vs-Neon measurements are noisy; they are baseline evidence, not a claimed speedup.

## Work order

1. Add low-overhead, benchmark-only work counters and focused SIMD microbench coverage.
2. Measure and tune NEON LCP/copy/pattern/repeat paths without changing the wire format.
3. Measure bounded candidate selection and scalar pipeline overhead.
4. Reduce worker, scratch, allocation, and result-assembly overhead.
5. Revisit format and GPU changes only after same-format CPU evidence and real Metal validation.

## Acceptance gates

- `cargo test --release` and feature-specific tests remain green.
- `scripts/check.sh` and `scripts/check_model.sh` remain green where applicable.
- Malformed-input and cross-backend behavior do not regress.
- Any performance claim includes repeated measurements, workload, SIMD mode, thread count,
  block size, and whether it is kernel-only or end-to-end.
- No format change is accepted without explicit compatibility and ratio/speed evidence.

## 2026-09-15 — SIMD kernel measurement and pattern fill

- Added `vxg-bench --microbench` for direct LCP, copy, and period-37 pattern measurements.
- Reworked pattern tile construction to use seed doubling through the selected `Engine::copy`
  path; removed the scalar per-byte modulo loop from this hot path.
- Expanded periodic pattern coverage across every supported SIMD dispatch.
- Wire format and public compression/decompression behavior were intentionally unchanged.
- Added a 64-byte short-copy threshold so tiny literals use the platform copy path instead of
  paying for the hand-vector loop; large-copy dispatch remains unchanged.
- Latest direct-kernel sample is retained in `reports/simd-kernel-microbench-2026-09-15.csv`.
  In the latest seven-sample run, Neon reached 29.100 GB/s for 1 MiB equal-LCP and 38.920
  GB/s for 1 MiB copy; period-37 pattern reached 53.614 GB/s. These are kernel medians, not
  end-to-end codec claims, and process-to-process variance is still visible.
- Pattern bulk copy was compared against a platform-copy baseline. The code now uses the
  platform copy for bulk tiles while retaining the selected Engine for seed/doubling work;
  this avoids assuming hand-written NEON wins universally.
- Validation through the work-counter stage: default release tests, bench-feature tests,
  `scripts/check.sh`, and both WG=8/WG=64 model checks passed. The model checks remain
  reference/SIMT emulation, not actual GPU execution.
- The NEON LCP loop is now 2x-unrolled for 32-byte groups. Its latest five-sample 1 MiB equal
  input result was 34.235 GB/s versus 25.817 GB/s for Scalar in the same benchmark protocol;
  this is a kernel result and needs an end-to-end repeat before being treated as a codec win.
- Post-unroll end-to-end smoke benchmark (`reports/local-cpu-unroll-check.csv`, 4 MiB,
  one thread, three samples) measured VXG decode at 121.536 GB/s for zeros, 50.473 GB/s for
  period-37, 1.249 GB/s for JSON, 55.218 GB/s for random, 62.157 GB/s for low-entropy-random,
  and 5.323 GB/s for mixed data. These values are a fresh run, not a controlled before/after
  experiment; use the retained work counters and repeated runs for attribution.
- Final validation after the LCP unroll: `scripts/check.sh` passed with shader consistency,
  2 golden tests, 12 property tests, doctest, and the roundtrip example; `scripts/check_model.sh`
  passed both WG=8 and WG=64. A final `cargo test --release --features bench,metal` also passed
  (2 golden, 11 property tests, doctest); this is a host/backend test gate, not hardware GPU
  execution.

## 2026-09-15 — Allocation baseline

- Added benchmark-only `alloc-stats` instrumentation; it wraps the benchmark binary's system
  allocator and is explicitly excluded from performance comparisons.
- The 4 MiB, one-thread baseline is retained in `reports/allocation-stats-2026-09-15.csv`.
  A single VXG compression call used 69–70 allocation calls across the built-in datasets;
  allocation volume ranged from 4.28 MB (zeros/periodic) to 8.54 MB (random inputs).
- This supports prioritizing persistent worker/scratch and result-payload reuse before a global
  mimalloc/jemalloc switch. Preallocated decode does not need an allocator redesign first.
- Added bounded `Scratch::reserve_for` hints and exact worker/result capacities. The reserved
  allocation comparison is retained in `reports/allocation-stats-reserved-2026-09-15.csv`:
  total allocation operations (alloc + realloc) fell from 75 to 71 for zeros, 76 to 71 for
  period-37, 100 to 72 for JSON, 74 to 72 for random, and 96 to 74 for mixed data. This is
  allocation evidence only; the counter-enabled GB/s remains non-comparable.
- A normal 4 MiB, one-thread, three-sample run after reserve is retained as
  `reports/local-cpu-reserved-normal.csv`. It does not establish a speedup against the earlier
  smoke run because process-to-process variance is larger than this short experiment. The next
  allocator decision must use longer repeated process runs and separate allocation/RSS metrics.
- After the allocation reserve changes, the full `scripts/check.sh` gate passed again: shader
  consistency, 2 golden tests, 12 property tests, doctest, and the roundtrip example.

## 2026-09-15 — Codec LCP work counters

- Extended `WorkStats` with logical `lcp_calls` and `lcp_bytes`; these count codec work rather
  than hardware instructions and aggregate across blocks/workers.
- Benchmark output now prints the counters outside the timed loop. A 4 MiB single-thread sample
  observed 64 LCP calls / 4,193,728 logical bytes for zeros, 148,935 / 3,114,892 for JSON,
  37,267 / 2,839,365 for mixed data, and zero LCP calls for random and low-entropy-random.
- This confirms LCP is hot for compressible inputs, while random-input cost is dominated by
  hashing/probing and RAW handling. It also prevents interpreting the Neon microbenchmark as
  whole-codec coverage.
- A fresh 16 MiB, one-thread, five-sample end-to-end run is retained as
  `reports/local-cpu-pattern-tile-1-thread.csv`; period-37 VXG decode measured 34.457 GB/s.
  Cross-run comparisons remain provisional because the process-level benchmark is noisy.

## 2026-09-15 — Allocator A/B harness

- Added benchmark-only `mimalloc` and `jemalloc` feature switches using the locally cached
  `mimalloc 0.1.52` and `tikv-jemallocator 0.7.0` crates. The library's default allocator and
  embedding behavior remain unchanged; the switches apply only to `vxg-bench`.
- The benchmark CSV now records the allocator, and allocator combinations are compile-time
  rejected when ambiguous (`mimalloc+jemalloc` or replacement allocator + `alloc-stats`).
- The next measurement is a matched System/mimalloc/jemalloc process-level A/B. It must report
  repeated medians together with threads, block size, dataset, allocation counts/RSS where
  available, because a faster allocator alone is not evidence that arena/slab belongs in the
  public codec API.

## 2026-09-15 — Allocator A/B result

- Matched single-thread run: 16 MiB, Neon/Auto, 64 KiB blocks, five samples at 0.2 s per
  sample. VXG encode medians were System/mimalloc/jemalloc respectively: zeros 25.294/26.293/
  25.528 GB/s, JSON 2.480/2.547/2.391 GB/s, random 4.881/5.011/3.537 GB/s, and mixed 3.036/
  3.183/2.920 GB/s. Decode was preallocated, so allocator sensitivity was smaller and mixed
  results were 5.041/5.444/5.405 GB/s. These are process-level medians from one run, not a
  universal allocator ranking.
- Short 8-thread direction check: 4 MiB, 64 KiB blocks, three samples at 0.05 s. On VXG encode,
  mimalloc was -9.5% on random and -6.9% on mixed versus System; jemalloc was -18.1% and
  -22.7%. Decode moved in the opposite direction on some datasets. This does not support
  enabling either allocator by default yet.
- Allocation counters show the more actionable target: after reserve hints, one-thread VXG
  compression used 69–70 allocation calls plus 1–4 reallocations across the built-ins; the
  eight-thread zeros case used 132 calls plus one realloc, with JSON/random at nine reallocs
  and mixed at 25. A worker-local output slab/arena may reduce these per-block/result allocations,
  but it needs a separate design and before/after measurement; no public arena API was added.
- Retained A/B files: `reports/allocator-system-2026-09-15.csv`,
  `reports/allocator-mimalloc-2026-09-15.csv`, `reports/allocator-jemalloc-2026-09-15.csv`,
  plus the corresponding `*-t8-2026-09-15.csv` files. Metal host tests still pass, while the
  actual runtime check remains blocked by `GPU backend: no Metal device` on this host.

## 2026-09-15 — Worker-local payload arena

- CPU workers now append block payloads to a worker-local arena and return compact
  `(offset,len,kind,checksum)` descriptors. The final frame assembly sorts descriptors by block
  index and copies each arena slice into the unchanged wire layout. GPU assembly still uses its
  original owned `Vec` parts.
- The arena preserves RAW fallback, checksums, block ordering, and decode behavior. Default,
  bench-feature, and Metal host test suites passed after the change.
- Allocation probe comparison on 4 MiB: one-thread zeros went from 70 allocations + 1 realloc
  after reserve hints to 9 + 0; mixed went from 70 + 4 to 9 + 8. At eight threads, zeros went
  from 132 + 1 to 77 + 0, JSON from 132 + 9 to 77 + 16, random from 132 + 9 to 77 + 32, and
  mixed from 132 + 25 to 77 + 40. Total allocation operations fell for every eight-thread case
  despite the arena's growth reallocations.
- Added `--vxg-only` to isolate VXG process medians from LZ4/Zstd and GPU rows. Repeated clean
  runs still show substantial macOS process variance, so allocator/arena speedups remain
  provisional until a shuffled multi-process run with CPU/RSS telemetry is available.

## 2026-09-15 — Shared hash-word load

- The CPU matcher now loads the current eight-byte word once and reuses it for both hash
  calculation and candidate equality. The candidate word uses the same little-endian value
  comparison, preserving output determinism across SIMD dispatches and leaving the wire format
  unchanged.
- Default, bench-feature, and Metal host tests passed after the change. In the isolated
  `--vxg-only` run, random/low-entropy inputs confirmed `lcp_calls=0`; their hot work remains
  hash/probe and RAW fallback rather than the NEON LCP kernel.
- The new `--vxg-only` outputs are retained as `reports/hash-system-vxg-t1-2026-09-15.csv`,
  `reports/hash-mimalloc-vxg-t1-2026-09-15.csv`, and
  `reports/hash-jemalloc-vxg-t1-2026-09-15.csv`. Their process medians still vary enough that
  the shared load is kept as a measured candidate, not a claimed end-to-end percentage win.

## 2026-09-15 — Hash probe telemetry

- Extended `WorkStats` with occupied-probe, candidate-check, and hash-collision counters. The
  counters are diagnostic only and do not alter the wire format or matcher decisions.
- The benchmark now prints these counters with the existing LCP work counters, allowing random
  and low-entropy inputs to be analyzed separately from compressible LCP-heavy inputs before
  changing table size or collision policy.

## 2026-09-15 — Hash-table capacity A/B

- Added benchmark-only `hash13` and `hash15` features around the default `HASH_BITS=14`; these
  change only the per-worker hash table footprint and do not change the frame format.
- On a single 4 MiB, one-thread run with five 0.1 s samples, `hash13` reduced the table footprint
  but raised random candidate checks from 15,018 to 28,616 and mixed-data checks from 56,300 to
  73,348. `hash15` reduced them to 7,695 and 47,086 respectively. JSON remained LCP-dominated
  in all three variants (about 149k candidate checks and 149k LCP calls).
- Throughput was input-sensitive: `hash13` was faster on zeros/period-37 in this run, while the
  default 14-bit table was faster on mixed data and slightly faster on random data. `hash15` was
  slower on every measured encode row, consistent with its larger cache footprint. Because this
  is one process run per variant on a variable macOS host, the default 14-bit table remains the
  production choice; the retained CSVs are `reports/hash13-table-t1-2026-09-15.csv`,
  `reports/hash14-table-t1-2026-09-15.csv`, and `reports/hash15-table-t1-2026-09-15.csv`.

## 2026-09-19 — First actual Metal device result

- An unsandboxed Metal run discovered `Apple M2 Max` and passed all 81 three-way CPU/GPU
  conformance cases. The normal sandbox still returns `no Metal device`; this is an execution
  environment restriction, not a hardware or VXG pipeline-discovery failure.
- First hardware e2e sample used 4 MiB inputs, one CPU thread, 64 KiB blocks, three 0.05 s
  samples, checksums off, and the default 16 MiB GPU batch. The GPU encoder was lossless but
  slower than the CPU encoder on every supplied dataset: zeros 0.909 GB/s, period-37 0.715,
  JSON 0.336, random 0.113, low-entropy random 0.112, and mixed 0.122. These are host-e2e
  allocating numbers, not kernel-only numbers.
- GPU decode was materially faster than GPU encode (zeros 7.016 GB/s; random 4.426; low-entropy
  random 4.488) but remained below the prepared CPU decode path. Device timestamp observations
  show the distinction: random's GPU kernel was about 0.148 ms while a one-call host e2e sample
  was 1.455 ms. Buffer upload/readback, framing, command submission, and synchronous wait are
  therefore the first optimization target, ahead of claiming a kernel-speed result.
- The short measurement CSV is deliberately ignored as a local run artifact:
  `reports/local-metal-e2e-2026-09-19.csv`. Before making a GPU performance claim, repeat with
  shuffled multi-process trials, larger inputs, and separately reported kernel/e2e medians.

## 2026-09-19 — Bounded CPU RAW fast path

- Added an allocation-free, 48-word sampled negative filter for blocks of at least 4 KiB. If no
  sampled eight-byte word repeats, VXG emits the already-valid RAW representation before it
  clears or probes the full hash table. A sample hit falls through to the unchanged matcher;
  the heuristic can therefore miss a fast path but never changes decode semantics or requires a
  new frame format.
- `WorkStats` now reports `raw_fast_path_blocks` and `raw_sample_words`; the benchmark prints
  both. A property test proves that a 64 KiB generated incompressible block takes the RAW path
  with zero matcher probes, while a period-37 block stays LZ and round-trips.
- Same-machine A/B, 16 MiB, one thread, 64 KiB blocks, seven 0.1 s samples, checksum off:
  random encode was 8.129 GB/s with the path versus 3.441 GB/s in the benchmark-only
  `raw-fastpath-off` control (2.36x); low-entropy random was 7.967 versus 3.424 (2.33x).
  Both remain RAW at ratio 1.000. JSON and mixed data did not take the path and retained ratios
  0.213 and 0.310. These are one ordered process pair, so regular-data throughput remains
  provisional until a shuffled multi-process comparison is added.
- Retained local A/B artifacts are `reports/local-raw-48-2026-09-19.csv` and
  `reports/local-raw-control-2026-09-19.csv`; `raw-fastpath-off` exists solely for benchmark
  control builds and is not a production configuration recommendation.

## 2026-09-19 — Rejected two-candidate matcher experiment

- A benchmark-only two-slot hash table was tested: after the latest candidate was short or a
  collision, it compared at most one older candidate. The work bound stayed explicit at two
  candidate checks/LCP calls per occupied probe, and default plus experimental property suites
  passed.
- It is not retained: at 16 MiB, one thread, 64 KiB blocks and seven 0.1 s samples, JSON ratio
  stayed 0.213 while encode fell from 2.690 to 1.703 GB/s; mixed ratio only moved 0.310 to 0.309
  while encode fell from 2.780 to 2.263. The second lookup raised JSON LCP calls from 591,087 to
  941,290 without a material ratio return.
- The experiment confirms that an unconditional older-candidate lookup is the wrong next step.
  Matcher work should next be directed by token/match histograms or a more selective lazy policy,
  not enabled as a default or retained as a feature.

## 2026-09-19 — Token-stream telemetry identifies the ratio bottleneck

- `WorkStats` now records canonical opcode counts plus control and attribute stream bytes; the
  benchmark reports them with matcher work. The counters do not alter encoding decisions.
- At 16 MiB / 64 KiB blocks, JSON encoded to 3,572,973 bytes (ratio 0.213): 1,183,316 bytes were
  control tokens, 2,230,192 were attributes, and only 147,145 were literals. Of 591,087 match
  commands, 557,548 used a four-byte absolute distance (`op=1`) and only 33,539 reused the prior
  distance (`op=2`). The next ratio experiment should therefore target bounded block-local
  distance representation, not more expensive candidate search.
- Mixed data's 5,200,371-byte output had 500,864 attribute bytes and 309,220 control bytes, but
  4,377,967 literal bytes. Its first ratio lever remains match selection; JSON's is command
  representation. These are distinct workload paths and must not be optimized as one benchmark.

## 2026-09-19 — v2 compact short-distance command

- The encoder now emits frame v2. Its new opcode 7 stores a non-special absolute match distance
  up to 65,535 as two little-endian bytes instead of v1 opcode 1's four bytes. Existing v1 frames
  are still parsed and decoded; opcode 7 remains invalid in v1. `Frame::version()` exposes the
  on-wire version so embedders can observe the distinction.
- The change follows the format end-to-end: Rust CPU encoder/decoder and validation, independent
  Python oracle, shared shader source and regenerated Metal/CUDA/GLSL artifacts, plus the CPU
  SIMT emulator. A property test verifies v2 emission and v1 golden-vector compatibility.
- In the matched 16 MiB, one-thread, 64 KiB-block, seven-sample (0.1 s) CPU run, JSON fell from
  the prior 0.213 ratio to 0.147 (2,457,877 bytes). Its attribute stream fell from 2,230,192 to
  1,115,096 bytes: 557,548 former absolute-distance commands now use opcode 7. VXG encode was
  2.235 GB/s and prepared decode 1.217 GB/s in this run; the ratio gain is structural, while the
  short benchmark's throughput deltas remain provisional. LZ4-a1 was 0.150/1.176/7.285 GB/s and
  Zstd level 3 was 0.100/1.992/4.148 GB/s (ratio/encode/decode) on the same JSON corpus.
- Hardware validation passed again on `Apple M2 Max`: the Metal-feature Rust suite passed and
  `gpu-check metal` passed all 81 three-way conformance cases. The shared-kernel CPU SIMT checks
  passed at WG=8 and WG=64, and the independent model kept all 65,536 control-word cases green.
- A 4 MiB, one-thread, three-sample Metal host-e2e run also completed. GPU JSON encoded at ratio
  0.165 and 0.374 GB/s, then decoded at 0.636 GB/s; CPU JSON in that short run was 0.148,
  2.499, and 1.635 GB/s. The GPU encoder's greedy matcher does not yet match CPU ratio, and
  host transfers/framing still dominate small workloads; no GPU speedup is claimed. Local CSVs
  are ignored artifacts: `reports/local-short-distance-2026-09-19.csv` and
  `reports/local-short-distance-metal-2026-09-19.csv`.

## 2026-09-19 — Rejected short-repeat tile threshold

- JSON has short average matches, so a decoder experiment bypassed the 1 KiB short-distance
  pattern tile for runs below 256 bytes. An initial cross-process observation looked faster, but
  a matched benchmark-only control build disproved it: at 16 MiB, one thread, 64 KiB blocks,
  seven 0.1 s samples, JSON prepared decode was 1.612 GB/s with the threshold and 1.613 GB/s
  with the original always-tiled path. The optimization and its control feature were removed.
- This confirms that JSON decode's next bottleneck is not the short-distance tile setup alone.
  Retained local artifacts are `reports/local-short-repeat-2026-09-19.csv` and
  `reports/local-short-repeat-tile-control-2026-09-19.csv`.

## 2026-09-19 — Decoder parser inlining

- A 25-second Xcode Time Profiler trace of the isolated JSON decoder showed `Parser::next` and
  `block::decode` as the visible hot VXG functions; repeat/copy was not the dominant sampled
  leaf. To make profiling reproducible, `vxg-bench` gained `--only DATASET` and `--decode-only`:
  the latter builds and validates one VXG frame, then times only preallocated CPU decode.
- `u32_at`, `u16_at`, and `Parser::next` are now force-inlined in production builds. The
  benchmark-only `parser-no-inline` feature supplies a matched control build; neither feature
  changes the wire format, encoded bytes, or public decoding semantics.
- On the Apple M2 Max, JSON only, 16 MiB, one thread, 64 KiB blocks, nine 0.2 s samples,
  checksums off, the inlined decoder reached 1.973 GB/s versus 1.618 GB/s for the no-inline
  control (+21.9%). This is an ordered process pair, so repeat it in a shuffled process trial
  before treating it as a universal percentage; it does establish parser call boundaries as a
  material decoder cost. Local artifacts: `reports/local-parser-inline-2026-09-19.csv`,
  `reports/local-parser-no-inline-2026-09-19.csv`, and temporary Time Profiler trace data under
  `/private/tmp`.

## 2026-09-19 — v2 opcode-7 decoder fast path

- Follow-up profiling still centered on parser work. JSON's 557,548 short absolute-distance
  commands per 16 MiB (94% of all match commands) use v2 opcode 7, so `Parser::next` now takes a
  dedicated fixed-length opcode-7 path. It retains the generic path's literal bounds, output
  bounds, short-attribute truncation, positive-distance, and initialized-prefix checks; v1 and
  extended-length cases retain the generic parser.
- The benchmark-only `parser-op7-generic` feature restores the old generic path for a direct
  control. At 16 MiB JSON, one thread, 64 KiB blocks, nine 0.2 s samples, prepared decode was
  2.143 GB/s with the fast path versus 1.968 GB/s with that control (+8.9%), with unchanged
  0.147 ratio. This is an ordered process pair, not a universal claim. Artifacts:
  `reports/local-parser-op7-fast-2026-09-19.csv` and
  `reports/local-parser-op7-generic-2026-09-19.csv`.

## 2026-09-19 — Skip zero-length literal copies

- JSON has only 147,145 literal bytes across 591,146 commands, so the decoder was frequently
  invoking `Engine::copy` with two empty slices before every match. `block::decode` now skips
  that no-op while retaining the same slicing and copy path for non-empty literals.
- The benchmark-only `decode-unconditional-literal-copy` control restores the old behavior. At
  16 MiB JSON, one thread, 64 KiB blocks, nine 0.2 s samples, prepared decode reached 2.427 GB/s
  with the skip versus 2.093 GB/s with the control (+16.0%), with unchanged 0.147 ratio. This is
  an ordered process pair, not a universal percentage. Local artifacts:
  `reports/local-zero-literal-skip-2026-09-19.csv` and
  `reports/local-zero-literal-copy-control-2026-09-19.csv`.

## 2026-09-19 — Rejected opcode-7 dispatch reorder

- A decoder experiment split opcode 7 ahead of the existing `1|2|7` match branch. In the matched
  16 MiB JSON setup (one thread, 64 KiB blocks, nine 0.2 s samples), the special branch measured
  2.325 GB/s versus 2.377 GB/s for the generic control. The difference is not a win and is inside
  the process-level noise band, so the branch and its benchmark feature were removed.
- Retained local artifacts are `reports/local-op7-dispatch-2026-09-19.csv` and
  `reports/local-generic-match-dispatch-2026-09-19.csv`.
