# VXG 4 / 95 / 15 — CPU implementation and acceptance plan

Status: implementation plan, not a performance claim. Prepared 2026-09-19.
Executor: Terra. Start from the current checkout; the inspected code baseline is
`0a2b4ed` (validated-frame decode). Read `TERRA_HANDOFF.md` before executing.
This plan supersedes the CPU experiment ordering in `PERFORMANCE_PLAN.md`.

## 1. The target contract

Deliver one lossless VXG CPU configuration whose encoded output simultaneously meets:

| Metric | Required result | Stretch result |
|---|---|---|
| Allocating encode, raw bytes/second | >= 4.0 GB/s | Preserve while improving ratio/decode |
| Complete encoded bytes / original bytes | <= 0.050000 | Smaller only if speed survives |
| Safe preallocated decode, raw bytes/second | >= 10.0 GB/s AND >= 1.20x fresh LZ4 | >= 15.0 GB/s AND >= 1.50x fresh LZ4 |

GB/s is decimal, not GiB/s or gigabits/s. “95%” means at least 95% size reduction
(20:1), not an output/input ratio of 0.95. Ratios include headers, directories,
dictionaries, tables, exceptions, padding and all data required to reconstruct bytes.
Faster than 15 GB/s is acceptable; 15 is not a cap. At the recorded ~9 GB/s LZ4
baseline, the required decode floor is ~10.8 GB/s, not merely 10.

Default hardware/profile: Apple M2 Max, one CPU worker, 64 KiB independent blocks,
NEON when supported, release, system allocator, checksums off for timing, warm buffers.
Normal checksum-on operation must still be correct and get a separate cost report.
Multicore, GPU, persistent-workspace encode and prevalidated repeated decode are
separate profiles; none can satisfy the primary target by replacing this profile.

Scope inferred from the conversation: the current structured JSON/log workload.
P0 freezes the following corpus before optimization:

- Qualification set Q95: existing `json-records` at 16 and 64 MiB, plus three seeded
  held-out JSON/log families: changed field order/optional fields/escaping, mixed event
  templates and identifier distributions, and bursts/variable record lengths. Generate
  these from workload rules, never from an optimization candidate's preferred tokens.
  Freeze seeds, generator revision, byte counts and SHA-256 in a manifest.
- Development set: existing 16 MiB `json-records` and separate development seeds.
- Robustness set: all six existing built-ins, adversarial boundaries, ordinary text/code,
  and incompressible/already-compressed bytes where available. Ratio <=0.05 is not
  required for incompressible data; byte-exactness, bounds and RAW fallback are.
- Real user corpus is not currently bundled. Use already-authorized local files if
  available; otherwise report real-workload qualification as unverified. Synthetic
  qualification must never be described as a general production-corpus result.

Require all three metrics per Q95 member, not a favorable average. Freeze Q95 before
testing candidates and do not remove difficult members after seeing results. If the
triple is infeasible on this set, report the observed frontier and missing targets.
Any change to the qualification scope is an explicit user decision, not a silent
relaxation. Arbitrary-byte 95% lossless reduction cannot be promised.

## 2. Evidence and arithmetic to preserve

Latest matched comparison: `reports/local-json-after-cpu-hotpaths-2026-09-19.csv`.
16 MiB JSON, 1 worker, 64 KiB, NEON, 9 x 0.2 s samples:

| Codec | Ratio | Encode GB/s | Safe prepared decode GB/s |
|---|---:|---:|---:|
| VXG v2 | 0.14650089 | 2.5151 | 3.1756 |
| LZ4 a1 | 0.14968413 | 2.2818 | 8.9942 |
| LZ4 a4 | 0.15083718 | 2.2813 | 9.0045 |
| Zstd -3 | 0.09964675 | 1.9222 | 4.0618 |
| Zstd 1 | 0.05424309 | 1.6018 | 3.2146 |
| Zstd 3 | 0.06240344 | 1.4862 | 2.8462 |

These are historic ordered-process observations, not multi-process acceptance results.
The later `local-safe-decode-v2-2026-09-19.csv` measured 3.379 GB/s; its separate
`local-validated-decode-v2-2026-09-19.csv` measured 3.9524 GB/s excluding one-time
command validation. Never substitute that second number for safe decode.

Known JSON stream accounting:

| Item | Bytes/count |
|---|---:|
| Original bytes | 16,777,216 |
| Complete v2 frame | 2,457,877 bytes |
| Control stream | 1,183,316 bytes |
| Attributes | 1,115,096 bytes |
| Literals | 147,145 bytes |
| Frame/directory/block prefixes | 12,320 bytes |
| Commands / matches | 591,146 / 591,087 |
| Opcode 7 / recent-distance opcode 2 | 557,548 / 33,539 |
| Zero-literal commands | 480,120 |

At 95% reduction the complete frame budget is at most 838,860 bytes. Keeping the
current literals and envelope leaves 679,395 bytes, about 9.19 bits per existing
command, for ALL command metadata. Current controls plus attributes cost about
31.1 bits/command. Therefore:

- Merely interleaving today's control and distance into a fixed 32-bit command cannot
  hit the ratio goal. 591,146 x 4 = 2,364,584 bytes before literals/headers. That layout
  is a speed experiment or transient execution representation, not the final answer.
- Reducing most distances from 16 to 14 bits alone cannot reach the target either.
  The 557,548 absolute distances would already consume ~975,709 bytes at 14 bits.
- Required change from the recorded baseline: ~1.59x encode, 3.15–4.72x decode,
  and ~65.9% fewer encoded bytes. Independent speedups must not be multiplied as
  if they were measured together.
- Average output is ~28.38 bytes/command. At 10–15 GB/s that demands roughly
  352–529 million commands/second if command count stays fixed. Reduce command
  count and serial work as well as bytes. Do not infer cycles without measured clocks.

For the 16 MiB case the total time budgets are 4.194 ms encode, 1.678 ms decode at
10 GB/s, or 1.118 ms decode at 15 GB/s. Every added analysis/packing/unpacking pass
spends this same budget. A new metadata pass needs enough savings elsewhere to pay
for itself; separately fast stages do not establish a fast combined codec.

Earlier conversational statements that v2 has a proven 4–5 GB/s ceiling are withdrawn.
No such ceiling was measured. Also correct the progress note saying all JSON matches
were 16–63 bytes: the recorded first bucket includes 190,765 runs of 8–15 bytes.

## 3. Architecture to investigate

Keep one logical lossless command model and experiment with its storage and execution
separately. The intended pipeline is:

```text
bytes -> bounded matcher -> commands -> per-group mode selection -> frame
frame -> checked group decode -> small command batch -> ordered copy execution -> bytes
```

Use a small stack/per-worker batch (initial candidates 32, 64, 128 commands); do not
allocate a whole-file expanded command tape for every decode. Measure unpack and
execution together. A fully expanded tape is permitted only as a labelled diagnostic
upper bound; its preparation, memory and cache costs cannot disappear from acceptance.

Investigate three representations sharing semantic tests:

1. Current v2 representation: frozen control/reference and fallback.
2. Direct fixed-word representation: useful to isolate instruction decoding overhead
   and as a transient batch format. Preserve escapes/large runs/pattern semantics.
3. Grouped compact representation: literal-presence masks, run residuals, repeat/cache
   selectors, distance residuals, and exception streams, optionally in bit planes.
   It must decode into the same bounded execution batch.

Start scalar and exact; add NEON unpacking once differential tests exist. Group fields
can be unpacked together, but LZ output references can depend on earlier commands in
the same group. Execute copies in dependency order. SIMD unpacking does not license
parallel scatter copies with unresolved backreferences.

The long-term candidate is grouped packed metadata plus a small fast execution batch,
if measurements support it. Keep at most one speed control and one compact challenger
active at a time. Do not grow a matrix of unmeasured format variants.

## 4. Work packages, in execution order

### P0 — Freeze the experiment and harden the current baseline

Files: `src/bin/bench.rs`, `src/baselines.rs`, `src/block.rs`, `src/frame.rs`,
`tests/`, `fuzz/`, `docs/FORMAT.md`, `docs/BENCHMARKS.md`.

- Freeze Q95 and environment metadata; reproduce v2 and reference-library results.
- Fix the benchmark correctness gap: setup currently decodes a frame from
  `compress_with_stats`, but timed encode calls `compress`. Outside timing, verify
  byte-for-byte equality and roundtrip of the actual timed API for every configuration.
- Make benchmark profile names unambiguous, including the console banner for
  `--validated-decode`. Add an explicit validation-cost row and a safe full-frame
  parse+decode row alongside the existing prepared-directory decode row.
  Report repeated-decode amortization as `T_validate + k * T_validated_decode` versus
  `k * T_safe_decode`; compute a break-even reuse count from measured times.
- Audit unsafe validation boundaries. `block::decode_validated` is currently a safe
  crate-visible function whose unchecked inner loads require prior validation. Encode
  that obligation with an unforgeable validated block type, or an `unsafe fn` with
  audited callers; do not leave a semantic precondition hidden in a safe helper.
- Strengthen validated-path tests: accepted mutated frames, malformed frames rejected
  before unchecked execution, v1/v2 golden vectors, literal tails, recent-distance
  transitions, small overlap, long extensions, RAW/empty/multiblock, checksum failures,
  wrong output size, output limits, repeated decode and multi-worker operation.
  Extend fuzzing: for any frame that validates, safe and validated decoders must agree
  byte-for-byte using the same options; test no panic and guarded output bounds.
- `docs/FORMAT.md` still describes only v1, while the implementation emits v2. Document
  actual opcode-7/version behavior before designing v3. Graphify's stored line numbers
  are old; inspect current source, do not treat the graph as the current implementation.

Deliver: corpus manifest, reproducible runner and retained baseline report; tests and
correct format/API documentation. Exit: baseline reproducible, validation boundary
audited, and both encode APIs verified. Do not claim speed gains from P0 paperwork.

### P1 — Find and remove remaining per-command execution costs

Files: `src/block.rs`, `src/simd.rs`, focused benchmark/test helpers.

- Profile safe decode, prevalidated decode and encode separately on JSON and mixed data.
  Inspect generated code for memcpy calls, bounds checks, duplicate cursor updates,
  large stack frames and missed inlining. Keep profiler overhead out of timing runs.
- Measure parse-only validation, exact match-copy execution, pattern fills and RAW copy
  as diagnostics. Their times are not safely additive or subtractable across different
  programs; use them to choose experiments, then validate end-to-end.
- First concrete candidate: inline exact copies for non-overlapping 8–15, 16–31 and
  32–63-byte matches. Compare two endpoint loads/stores or fixed chunks against platform
  memcpy. A head/tail copy must stay within the requested source and destination ranges;
  load-before-store and overlap assumptions must be explicit. Preserve LZ overlap path.
- Second candidate: a checked fast loop for a small group of ordinary commands. Validate
  stream availability and aggregate output bounds with checked arithmetic, then validate
  each distance against the correct evolving produced prefix. Keep cold escapes/tails
  out of the common loop. Compare 2/4/8-command unrolling; excessive code size can lose.
- Test both opcode 7 and opcode 2 sequences; optimizing only one class can shift the
  bottleneck immediately. Retain exact terminal stream consumption.

Deliver: at most two controlled experiments with disassembly/profile evidence and
safe end-to-end results. Aiming for 5–6 GB/s is an intermediate research target, not a
known reachable floor. Exit with a measured execution-cost map even if neither wins;
then proceed to P2, rather than endlessly reordering parser branches.

### P2 — Measure the achievable metadata budget before choosing a format

Files: diagnostic analyzer under `verification/` or a bench-only Rust tool; no counters
in timed production `compress()`.

- Dump/inspect actual command streams for development and frozen qualification data.
  Capture exact literal/run histograms, joint opcode/literal/run distributions,
  per-group bit widths, distance deltas/XOR residuals, and 2/4/8-entry distance-cache
  hits with fully specified insertion/eviction/update/reset behavior.
- Quantify dependencies: non-overlap share, minimum distance, references to earlier
  commands in a batch, average output per command, exception frequency and group tails.
- Estimate storage for each candidate including selectors, bases, widths, dictionaries,
  exceptions, padding and block/frame overhead. Zero-order entropy is a diagnostic
  model, not a proven lower bound for every possible correlated representation.
- Build tiny roundtripping scalar pack/unpack prototypes to replace estimates with
  actual byte counts. They reuse existing commands so matcher effects remain isolated.
- Simulate a fast 2/4/8 recent-distance cache and group-local dictionary of at most 16
  distances. Compare absolute distance, delta, XOR and dictionary-index encodings;
  do not assume distances are monotonic. Reset all state at block boundaries.

Decision: if the unchanged command trace cannot approach <=0.05 with practical coding,
document the shortfall and activate P5 command-count reduction. Do not spend a long
SIMD optimization cycle on a representation already over budget on paper.

### P3 — Bitsliced, int4 and ternary prototypes (required experiments)

Use the P2 trace corpus and the same logical commands. Implement portable scalar
oracles first and SIMD implementations second. Candidate modules may be separate
`src/packing.rs` / `src/block_v3.rs`; these names are proposed, not existing APIs.

| Experiment | Concrete representation | Mandatory comparison |
|---|---|---|
| Fixed-width packing | Per-group bases/widths, packed residuals and bounded exceptions | Plain byte-aligned fields |
| Bitsliced metadata | Transpose residual fields into bit planes; omit proven-zero upper planes or code plane patterns with explicit metadata | Identical residual model with ordinary packing |
| Literal/run packing | Presence bitmap; small run-minus-minmatch values; exceptions for larger values | Current lengths and byte/nibble forms |
| int4 | Two exact 0..15 indices/residuals per byte; larger values escape or choose another width | General bit packing including all escape costs |
| Ternary | Exactly three measured command classes; compare 2-bit selectors with five base-3 digits/byte | Selector decode + payload execution together |
| Distance modeling | Small recent cache or group dictionary plus packed indices and explicit escapes | Absolute u16/u32 distances |

Bitslicing is a data layout. A transpose alone preserves byte count. Compression must
come from omitted widths, modeled residuals, sparse planes, repeats or an explicit
entropy stage. Test it first on command metadata, where fields have known meaning.
An optional typed-data bitshuffle+codec experiment belongs to a separately labelled
workload; do not reinterpret arbitrary JSON bytes as numerical columns.

int4 here is lossless integer packing, not quantization. Validate residual range before
packing; never truncate high bits. If a nibble value is reserved as an escape, only the
remaining values are direct codes. Record unused-tail-nibble canonicality.

For ternary, five trits have 243 states, fitting in one byte (1.6 bits/trit). Define
class semantics and reject byte values 243..255 where applicable. A third class needs
a specified escape for other opcodes. Compare division-free table decode against
simple 2-bit masks; ternary may be smaller but slower, and highly skewed selectors may
benefit more from a bitmap/RLE model. Include table memory and cache effects.

Use 32/64/128-command groups initially. Decide widths, bit order, tail rules, exception
order and reset points in writing. SIMD loads must remain bounded or use initialized
owned scratch whose copy/zeroing cost is timed. Keep source bytes immutable.
Exercise empty/partial/full groups, widths 0/1/4/8/16/32 and every other supported
width, maximum values, all exceptions, alternating planes and exact buffer endings.
Do not shift by a machine word's full width or overflow reconstructed signed deltas.
Report pack/unpack in integers/second and metadata bytes/second; reserve original
bytes/second GB/s for complete codec runs, with both denominators explicit.

Acceptance: each experiment has measured size, pack/unpack throughput, scratch usage
and correctness; retain only Pareto-useful candidates. The final codec need not force
ternary/int4 on groups they make worse. Bitsliced must be implemented and evaluated;
include it in the selected codec where supported by evidence, otherwise keep its
documented experimental result instead of claiming an unmeasured speedup.

### P4 — Integrate a checked grouped decoder and choose the v3 layout

- Compare direct fixed-word and compact grouped formats using identical commands and
  output. Include group parsing, validation, unpacking, scratch initialization and
  ordered match copying in safe decode timing.
- Unpack at most a bounded group at a time into compact fields. Avoid a heap allocation
  per command/group. Batch distance-prefix checks without losing the evolving-prefix
  invariant; read-only dictionaries must be valid before an indexed access.
- Add group tags/lengths and explicit version/kind rejection. Draft `FORMAT_V3.md`
  before connecting the new wire layout to public APIs. Keep v1/v2 decode working and
  initially expose v3 through an opt-in encoder setting; do not reinterpret v2 opcode 7.
- Define all width-zero/max-width, overflow, escape, padding, count and stream-exhaustion
  cases. Define `max_compressed_size` and RAW fallback for every supported block mode.
- Measure preparation separately for prevalidated handles. If validation first builds
  a command tape, charge that work to safe one-shot decode. It cannot be hidden in
  `Frame::parse` or a benchmark setup object to win against `LZ4_decompress_safe`.

Deliver: executable CPU v3 prototype, independent scalar/reference decoder, golden
fixtures and a format ADR with size/speed tradeoffs. Research checkpoint: approach
6–9 GB/s safe decode and ratio <=0.08, while tracking encode. Failure to meet this
checkpoint triggers the P2/P5 diagnosis, not a fabricated claim that the format is done.

### P5 — Reach the joint ratio/encode budget

Revisit matcher work only after P2 identifies which additional matches would reduce
total encoded cost and command execution work. Candidate order:

1. Cheap recent-distance prediction before hash lookup; account for extra failed probes
   and compare unchanged fallback decisions. Check recent-cache frequency first.
2. Selective second-candidate/lazy search gated by current short-match cost and expected
   saved commands. Maintain a fixed work bound; compare matched bytes, command count,
   literal growth and actual packed size. The previous unconditional second candidate
   lost badly and is not an acceptable default retry.
3. Measured vector/scalar LCP mismatch localization. Inspect compiler output; trial a
   simpler word-XOR locator only if profiling still points there. Previous shuffle-mask
   and 32-byte scalar-prefix attempts lost; do not assume more NEON means faster.
4. Reduce buffer growth, per-command append checks, scratch clearing and final assembly
   copies where profiles justify it. Worker-local arenas and stats-free encode already
   exist; do not claim to introduce them again. Charge mode analysis/packing to encode.

Use a cheap deterministic per-block/group selector instead of fully encoding many
formats on every call. Include selector failures and RAW fallback in elapsed time.
Expose acceleration/mode in reports; `compress` and `compress_with_stats` must emit
identical frames for identical options. Small bounded distance-cache state is distinct
from an expensive multi-candidate match search.

If compact metadata is still insufficient, prototype bounded entropy coding only on
the identified fields (e.g. skewed op/run/distance selectors). Measure table build and
decode cost; do not wrap the entire VXG output in Zstd and label it a VXG algorithm win.
No external trained dictionary in the primary target. Optional text/record transforms
must reconstruct every original byte, including whitespace, key order and escapes,
and require their own explicit mode and corpus report before inclusion.

Exit: one candidate produces <=0.05 at >=4 GB/s encode and has a measured route toward
the required safe decode threshold, or an evidence-backed Pareto frontier shows the
remaining conflict. Intermediate fast/compact modes may help research, but cannot
satisfy the simultaneous target by combining their best individual numbers.

### P6 — Close the decode gap on the actual compact artifact

Profile P5 output again: length/displacement distributions have changed. Reuse P1's
exact copy kernels only where their assumptions still hold. Optimize batch unpack,
prefix sums, repeat-cache dependency chains and copy execution in that order of
measured cost. Try selective 2/4-lane independent decode states only when format
independence and net metadata cost are established. Reordering matches requires a
proved dependency schedule; no speculative reading of not-yet-produced bytes.

Keep 64 KiB/one worker as the primary run. Sweep block sizes/working sets separately to
understand cache and metadata effects, comparing references at the same block size.
Changing block size to hit ratio does not satisfy the original locked profile.

Exit: primary minimum triple and LZ4 +20% pass together. Then attempt 15 GB/s and
LZ4 +50% with encode/ratio still passing. Keep the best accepted candidate available
while investigating the stretch result.

### P7 — Compatibility, robustness and independent qualification

- Update the independent Python reference for every accepted wire mode and maintain
  fixed v1/v2/v3 valid/invalid fixtures. Differentially verify scalar and supported SIMD.
- Protect CPU/Metal behavior during prototyping: unsupported v3 must return an explicit
  error before device dispatch, never reach an old shader with reinterpreted commands.
  Preserve v2 GPU encode as an explicit fallback if v3 GPU support is still pending.
- Before making v3 the default broadly, implement its supported GPU decode in
  `native/shaders/core.gpu`, regenerate the Metal/CUDA/GLSL sources with
  `scripts/generate_shaders.py`, run model checks and actual Metal cross-decode.
  GPU encode need not generate every CPU mode; require readable versioned output.
- Test CPU default/bench and Metal-feature Rust configurations, not incompatible
  `--all-features` (allocator/hash A/B features are mutually exclusive). Run native
  x86 SIMD tests where hardware exists; cross-compilation alone is not execution.
- Extend fuzz and sanitizers around new unpack/copy/escape/validated paths; use Miri on
  scalar safety paths where supported. Retain reproduction inputs for every discovered
  bug. C++ model ASan coverage does not establish Rust unsafe-path safety.
- Perform final locked Q95 runs and held-out/robustness reports with the accepted binary,
  then record the complete result and remaining platform coverage.

Exit: acceptance matrix, reproducible artifacts, compatibility notes and no unresolved
correctness bug. Native CUDA/Vulkan checks unavailable on this host are reported as
unrun; they do not justify pausing unrelated CPU work or claiming GPU performance.

## 5. Benchmark and promotion rules

P0 should implement a repeatable runner; the following are requirements for that runner,
not existing command-line switches:

- Build once per candidate and preserve the executable plus hash/configuration. Do not
  rebuild a shared target directory while another test/benchmark is running. Test jobs
  may be parallelized with isolated artifacts; timing jobs run serially on an idle host.
- Trial pass: 3 process pairs, 7 x 0.2 s samples. Final acceptance: at least 7 paired
  processes, 9 x 0.2 s samples, shuffled candidate/control and codec order using a saved
  seed. Preserve individual sample means, process medians and min/max; these are not
  per-request p95/p99 latencies. Keep a variability interval, not only the best sample.
- Use per-process medians for confidence estimation; many inner iterations are not
  independent machine runs. A candidate improvement needs a positive 95% paired
  interval and normally >=3% speed benefit, or a useful ratio/resource tradeoff recorded
  in an ADR. If noisy, rerun with longer samples; do not discard slow runs ad hoc.
- Threshold qualification requires every Q95 file to pass ratio and the lower 95%
  confidence bound of throughput / paired LZ4 speedup to pass the corresponding floor.
  Use the faster of the tested LZ4 a1/a4 decode configurations as the comparator.
- Profile names: allocating encode; safe prepared-directory/preallocated decode;
  safe full-frame parse+preallocated decode; prevalidated repeated decode; allocating
  decompress; workspace encode; kernel microbench. Only the first two decide the
  primary speed contract. Prepared directory parsing may be excluded, command
  validation/unpacking may not. Report full-frame results to show setup overhead.
- Current baseline encode framing differs: VXG serializes a contiguous frame while
  LZ4/Zstd retain vectors of blocks with a synthetic equal-size envelope. Preserve and
  disclose this existing profile; add a matched serialization experiment if changing it,
  and rerun both baselines. Never attribute that change to an algorithmic speedup.
- Record input digest, binary/source revision, options/features, compiler/target flags,
  CPU/OS, libraries, checksum policy, allocator, threads, block size, cache profile,
  thermal/power observations, peak RSS and scratch footprint. Do not invent unavailable
  performance counters. Keep diagnostic instrumentation outside normal timing.
- Large/rotating buffers test cache sensitivity: include 64 MiB and a separately labelled
  working set larger than measured available cache. Small-block latency and checksum-on
  costs are reported independently. No swapping for these runs.
- Regression guard against the frozen baseline: no sustained >5% encode/decode regression
  on robustness members; no >1% relative size regression without a documented optional
  mode/tradeoff. Compare final integrated performance, not just isolated microbenches.
  Keep bounded scratch per worker, initially <=256 KiB beyond existing matcher state;
  measure and justify any larger allocation before accepting it as a default.

## 6. Safety and version invariants

Losslessness means byte-for-byte reconstruction, not semantic JSON equivalence.
Each backreference uses the initialized prefix of its own independent block. No hidden
dictionary, stored decoded output, cross-block history, benchmark-input recognition or
special casing the known JSON generator. Resource use must be bounded by checked
lengths and options before allocations. Unknown versions/modes fail explicitly.

Never remove checks merely to reach a GB/s target. Make required conditions provable
once per group, encapsulate the proof and use a fast inner loop under that proof.
`debug_assert!` cannot carry required release validation or perform a required read.
Keep exact stream consumption, malformed-input behavior, caller output limit and
checksum semantics. Public exact-size slices do not imply extra vector padding.

Pointer loads/copies need documented input availability, destination extent, alignment,
initialization, overlap and lifetime arguments. A private `unsafe` core may be fine;
a safe function with an unenforced external precondition is not. Add guard/canary
tests and sanitizer coverage for tails and overlap, not only easy roundtrips.

## 7. Stage gates and what counts as completion

| Gate | Required evidence | Failure action |
|---|---|---|
| G0 | P0 reproducibility and safety/API coverage | Fix before new unchecked execution |
| G1 | P1 profile/copy experiments, P2 actual metadata budget | Follow identified bottleneck |
| G2 | Bitsliced/int4/ternary prototypes roundtrip and measured | Drop losing variants, retain evidence |
| G3 | CPU v3 safe decode/reference/goldens; measured size/speed frontier | Revise layout or commands |
| G4 | Q95: >=4 encode, <=0.05 ratio, >=10 decode and >=1.20x LZ4 together | Incomplete; report exact shortfall |
| G5 | Q95: >=4 encode, <=0.05 ratio, >=15 decode and >=1.50x LZ4 together | Stretch incomplete; preserve G4 |
| G6 | Robustness, version/GPU handling, artifacts and final report | Resolve release blockers |

G4 is minimum target success; G5 is stretch success. Report each explicitly. Do not
declare the full 4/95/15 stretch achieved when only G4 passes. A fast unchecked result,
a tiny packed file with slow decoding, or different configurations winning each column
is not G4/G5. An infeasibility report is useful research, not a passed target.

Use the experiment loop: hypothesis -> one focused patch -> correctness -> paired
benchmark -> retain or revert only that patch -> evidence/next experiment. After three
failed small experiments on the same bottleneck, change the approach using profiles
or the representation/command budget; do not keep cosmetic branch churn going.

Track active package, next experiment, accepted binary, failures and artifact locations
in `docs/PROGRESS.md` plus a small machine-readable state file created in P0. Preserve
decisive measurements in a tracked `reports/qualification/` directory; existing
`reports/local-*` are ignored and cannot be the only handoff evidence.
Use stable experiment IDs (for example `P1-copy-01`) and at least these state fields:
`active_package`, `experiment_id`, `hypothesis`, `baseline_revision`, `candidate_hash`,
`corpus_manifest`, `commands`, `correctness_status`, `metrics_by_profile`, `decision`,
`next_step`, `unrun_checks`. A proposed artifact layout is `manifest.json`,
`experiments/<id>/metadata.json`, `samples.jsonl`, `summary.md` under that directory;
hashes, logs and archived binaries may point to durable local storage to avoid checking
large executables into Git. Preserve enough inputs/scripts to rebuild every accepted run.
Commit useful checkpoints if available, but Git access/cleanup must never interrupt
CPU development. Do not reformat unrelated files or reset user changes. Do not mark
goals complete solely because a turn/token budget ends. If a scientific plateau remains,
record the concrete evidence and next viable experiment; request scope changes only
when the existing target would actually need to change.

## 8. Initial commands and existing entry points

These commands exist now; future analyzer/runner/v3 switches must first be implemented
and documented rather than assumed to work:

```sh
cargo test --release
cargo test --release --features bench
scripts/check.sh
scripts/check_model.sh
cargo run --release --features bench --bin vxg-bench -- \
  --threads 1 --block-size 65536 --size-mib 16 --samples 9 --seconds 0.2 \
  --simd neon --only json-records --csv reports/local-plan-baseline.csv
cargo run --release --features bench --bin vxg-bench -- \
  --threads 1 --block-size 65536 --size-mib 16 --samples 9 --seconds 0.2 \
  --simd neon --only json-records --decode-only --validated-decode \
  --csv reports/local-plan-prevalidated.csv
scripts/gpu_check.sh metal
```

Run relevant checks after changes; do not repeatedly run the full suite on unchanged
code. Metal hardware access may require the host's approved execution environment.
Rust/Metal feature compilation is not proof of actual GPU execution.

## 9. References and interpretation limits

- Local implementation and CSVs above are the authority for VXG results. `PROGRESS.md`
  records rejected experiments: unconditional two-candidate matcher, NEON shuffle
  mismatch locator, scalar LCP prefix, short-repeat threshold and branch reordering.
  Reopen one only with a new measured reason and a control.
- [LZ4 v1.10.0 source](https://github.com/lz4/lz4/blob/v1.10.0/lib/lz4.c) and
  [block format](https://raw.githubusercontent.com/lz4/lz4/v1.10.0/doc/lz4_Block_format.md)
  are useful primary references for fast loops and format-tail constraints. Any adopted
  implementation must respect its license and VXG's own bounds contract.
- [Stream VByte paper](https://arxiv.org/abs/1709.08990) and
  [implementation](https://github.com/fast-pack/streamvbyte) motivate separating control
  and integer payload for SIMD unpacking. Integer-decoder throughput is not end-to-end
  LZ decode throughput; VXG's copy dependencies still apply.
- [Bitshuffle project](https://github.com/kiyo-masui/bitshuffle) describes a bit transpose
  that rearranges typed data for a subsequent compressor. It supports treating bitsliced
  layout and actual coding savings as distinct experiments, not promising 95% from a
  transpose. Sources checked 2026-09-19; vendor/project throughput is not a VXG target proof.

The numerical milestones beyond the recorded CSVs are research objectives. This plan
specifies how to establish success or an honest shortfall; it does not promise that
4 GB/s encode, 20:1 reduction and 15 GB/s safe decode are jointly attainable on Q95.
