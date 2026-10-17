# VXG development progress

This file records implementation steps, validation gates, and benchmark evidence.
Every implementation or measurement change is recorded here as a separate step.

## 2026-09-19 — P0 qualification freeze and validated decoder boundary

- Added `reports/qualification/manifest.json` and
  `scripts/generate_qualification_corpus.py`. Q95 is now an eight-member synthetic
  structured JSON/log set: the existing `json-records` workload and three held-out
  families, each at 16 and 64 MiB. Exact byte counts and SHA-256 values are frozen in
  the tracked manifest; the generated 320 MiB corpus remains ignored.
- Added `scripts/run_qualification.sh`. It builds one benchmark binary, captures its
  SHA-256/environment, runs each Q95 file serially, and preserves per-process summary,
  all timed samples and console output. `reports/qualification/runs/P0-smoke-r2` proves
  the runner end-to-end with one process and three 10 ms inner samples; the previous smoke
  remains preserved. These are explicitly directional only, not target-qualification or
  confidence results.
- The benchmark now checks, outside timing, that `compress()` produces bytes identical
  to `compress_with_stats()` before it uses the setup frame's size. It records the
  timed API actually being measured, retains `--samples-csv`, and adds clear VXG rows
  for safe full-frame parse+decode and parse+command-validation alongside the prepared
  and prevalidated profiles.
- `ValidatedFrame` now holds a private `ValidatedCommandStreams` proof constructed only
  after successful command validation. `block::decode_validated` is `unsafe` with its
  exact validation/immutability/output-length contract, and its caller documents how the
  proof establishes it. Extended tests cover v1/v2, checksum failure, output bounds,
  multi-worker decode and malformed frames; fuzzing now differentially compares safe
  and validated decoders when a proof can be obtained.
- Updated `docs/FORMAT.md` from the stale v1-only description: v2 emits the two-byte
  absolute-distance opcode 7, while v1 remains readable and must reject opcode 7.
- Validation: `cargo test --release` and `cargo test --release --features bench` passed
  (2 golden, 16 property, one doctest each). Cargo printed an unrelated inability to
  clean a globally cached `winapi-0.3.9` manifest; compilation and tests completed.
  `scripts/check.sh` passed shader consistency, the same release suite and the roundtrip
  example. `scripts/check_model.sh` passed its ASAN/UBSAN WG=8 and WG=64 CPU SIMT model
  checks; this remains model coverage, not native GPU execution.
- The Q95 smoke confirms the format is far from the joint target on held-out families.
  For the reproducible 16 MiB `json-records` member it observed ratio `0.14650089`,
  allocating encode `2.3069 GB/s`, safe prepared decode `3.2445 GB/s`, and best tested
  LZ4 safe decode `8.9195 GB/s`. This is one very short process, so it is evidence of
  the baseline gap—not an acceptance measurement. Detailed raw samples and every member
  are retained under `reports/qualification/runs/P0-smoke-r2`.

## 2026-09-19 — P1 direct safe-copy result and P2 command metadata analyzer

- P1 tested whether checked v2 opcode-7 commands could bypass `Engine::repeat` and invoke a
  direct non-overlapping copy after the safe parser had proved `run <= distance`. The control
  restored the prior engine route. Both builds passed the release suite, but the two ordered
  runs disagreed (candidate/control `2.816/2.477 GB/s`, then reverse-order
  `2.122/2.222 GB/s` safe prepared decode). The candidate was removed: this is not enough
  evidence to retain a new unsafe-looking branch, and the prevalidated profile varied with it.
  Local artifacts: `reports/local-p1-safe-op7-*-2026-09-19.csv`.
- Added bench-only `vxg::analysis` and `vxg-analyze`. It re-parses an existing encoded trace
  outside timing and records exact literal/run and joint opcode/literal/run histograms,
  8-command residual-width histograms, distance delta/XOR models, specified 2/4/8-entry
  MRU cache hits, non-overlap rate, within-group dependencies, and exact command totals.
  It neither changes compression decisions nor runs inside the timed codec.
- Retained reports for all frozen Q95 members live in
  `reports/qualification/experiments/P2-command-analysis`. On 16 MiB `json-records`, there
  are 591,146 commands (591,087 matches), 147,145 literal bytes and 16,630,071 matched bytes;
  every match is non-overlapping, only 5,182 references land within its 8-command group, and
  the specified 8-entry distance cache hits 335,204 matches. Those figures motivate compact
  grouped metadata, but a bitslice/transpose alone cannot reduce bytes; P3 must compare exact
  packed scalar roundtrips before any wire-format claim.

## 2026-09-19 — P3 lossless fixed, bitsliced, int4, ternary and distance oracles

- Added bench-only `vxg::packing` and `vxg-pack-eval`. They re-parse a frame produced by the
  existing `compress()` call and preserve the complete logical trace: opcode, literal/run
  counts, match distance and both halves of every pattern seed. The prototype is not a v3 wire
  format and never executes decode output; its rates are explicitly labelled commands/s and
  metadata-bytes/s rather than codec GB/s.
- The scalar prototypes use 32/64/128-command groups and independent-block resets. Fixed-width
  packing has per-field minima, widths and ordered `(u8 index,u32 residual)` escapes. Bitsliced
  packing uses precisely the same plan but transposes contiguous LSB-first residual bits into
  planes. It must therefore produce the same byte count, which the boundary test asserts.
  int4 stores two exact residuals per byte with `15` as an escape and rejects a nonzero unused
  tail nibble. Ternary has exactly three classes—`LIT={0}`, `MATCH={1,2,7}`, and
  `PATTERN={3,4,5,6}`—and compares 2-bit selectors with five base-3 trits per byte; both retain
  exact opcode subcodes and reject reserved/non-canonical selector values.
- Added self-contained lossless distance-stream experiments: u16 with u32 escape, signed
  delta-ZigZag ULEB, XOR ULEB, block-reset 2/4/8-entry MRU caches, and a canonical top-16
  group dictionary. The model decoders reject overlong ULEB values, out-of-range cache indices,
  duplicate/noncanonical dictionary entries, zero distances and trailing data. Their byte and
  rate fields remain *distance-only* and cannot be interpreted as complete-frame ratios.
- Tests cover widths 0/1/4/8/16/32, full actual encoder traces, alternating bit planes, int4
  escapes/tails, all v2 opcodes/pattern seeds, ternary `243..255` rejection, and every distance
  model. `cargo test --release --features bench packing` passed 6/6 and the two existing analysis
  regressions also passed. Cargo's global-cache cleanup warning for `winapi-0.3.9` remains
  unrelated to successful compilation/tests.
- Retained complete reports are in `reports/qualification/experiments/P3-packing-r2` for every
  frozen Q95 member, one scalar timing iteration each. `P3-packing-r1` is retained as the earlier
  metadata-layout-only interim output; r2 adds the required distance models and is the P3 record.
  On JSON, the best complete prototype is fixed/bitsliced at group 128: 0.119791 (16 MiB) and
  0.120802 (64 MiB), with exactly equal fixed/bitslice bytes. The 16 MiB JSON distance-only
  best is MRU-8 at 1,103,658 B versus 1,183,198 B for absolute u16/u32; it is a 6.7% distance
  substream reduction, not a complete codec gain. Bursty, event and field-order families retain
  much larger best fixed/bitslice ratios: 0.313019/0.270220/0.199435 at 16 MiB and
  0.319255/0.294602/0.201342 at 64 MiB. The tested int4 and ternary forms are worse on every
  member, and the dictionary is not a winner on the reported distance stream.
- Decision: none of these unchanged-trace scalar layouts is within the 0.05 artifact budget;
  P3 does **not** justify a v3 integration or SIMD transpose pass. Following the written plan,
  advance to P5 command-count reduction and cost-aware matcher experiments before P4. No encode
  or safe-decode target is claimed by metadata-only measurements.

## 2026-09-19 — P5 rejected recent-distance matcher probes

- P5 first tested a bounded block-local recent-distance prediction before accepting the hash
  candidate. A failed prediction retained the old hash decision; the candidate code and feature
  were bench-only and were removed after the experiment. On frozen JSON-16 MiB it performed
  681,809 extra word checks for only 763 hits (0.112%); it changed the v2 frame from 2,457,877
  to 2,456,330 bytes (−1,547 B). A reversed-order pair gave the same *direction* for short
  encode measurements, but this is not an acceptance-speed result.
- The all-Q95 retained artifacts at
  `reports/qualification/experiments/P5-recent-distance-r1` show why it was not promoted:
  v2 frames shrank on all members (−1,547 to −205,466 B), but the target-relevant fixed-128
  lossless metadata model grew on seven of eight members: +93,936/+357,208 B bursty,
  +52,141/+204,675 B event, +12,577/+47,855 B field-order and +849 B JSON-64. Only JSON-16
  moved −256 B in that model. A v2-only byte win is not sufficient when the target requires a
  compact artifact.
- A stricter hash-miss-only retry was also removed. It lowered the extra checks to 123,607 but
  found only 175 hits (0.142%). Its retained JSON artifact is
  `reports/qualification/experiments/P5-recent-miss-r2/json-records-16MiB.json`: v2 moved
  −327 B and fixed-128 −244 B. That is far too small to affect the 0.05 budget and still adds
  a per-probe load/branch. Both experimental feature flags, counters and matcher branches were
  deleted using scoped patches; the existing default matcher is again the source behavior.
- Candidate builds passed the full bench-feature suite (6 packing + 2 analysis + 2 golden +
  16 property + 1 doctest). The source after removal again passed default and bench release
  suites. The rejected local speed pairs remain under ignored `reports/local-p5-*`; they are
  directional evidence only and are not used as target qualification.
- Decision: do not spend another cycle on recent-distance prediction. P5 now needs a
  target-aware architectural experiment: bounded entropy coding or a matcher change whose
  command-count and *complete packed* size pay for its encode work. It must include table/group
  construction, selector/escape bytes and future safe-decode cost rather than treating a
  distance substream as a ratio win.

## 2026-09-19 — P5 bounded entropy and adaptive selector are ratio-only evidence

- P5 now has a bench-only, lossless command-stream oracle: each existing LZ block builds
  canonical Huffman tables for opcode, literal length, run length, seed fields and an MRU-8
  distance tag stream. The packed representation stores every `(symbol, code length)` entry,
  bit length, padding, distance escape and table; its decoder rejects noncanonical table order,
  invalid code lengths, nonzero trailing bits and malformed stream boundaries. It is not wired
  into `compress`, public framing or timed CPU decode.
- An adaptive companion builds **both** that entropy payload and a fixed-128 payload for each
  LZ block, writes a one-byte selector, and charges both builders in pack timing. This prevents
  a per-corpus choice from hiding construction work. Its retained all-Q95 reports are in
  `reports/qualification/experiments/P5-adaptive-entropy-r1`; the pure-entropy control is in
  `reports/qualification/experiments/P5-bounded-entropy-r1`.

  | corpus | fixed-128 ratio | entropy ratio | adaptive ratio | entropy/fixed LZ blocks |
  | --- | ---: | ---: | ---: | ---: |
  | bursty 16 / 64 MiB | 0.313019 / 0.319255 | 0.355148 / 0.360795 | 0.313033 / 0.319269 | 0/235 / 0/930 |
  | event 16 / 64 MiB | 0.270220 / 0.294602 | 0.292260 / 0.316088 | 0.270235 / 0.294617 | 0/249 / 0/961 |
  | field-order 16 / 64 MiB | 0.199435 / 0.201342 | 0.256144 / 0.257952 | 0.199450 / 0.201357 | 0/256 / 0/1020 |
  | JSON 16 / 64 MiB | 0.119791 / 0.120802 | 0.113657 / 0.113993 | 0.113659 / 0.114000 | 252/4 / 1011/12 |

- Thus the selector is a lossless Pareto guard: it loses only its one-byte-per-LZ-block selector
  on six held-out members and saves 102,875 B / 456,495 B of metadata on JSON-16 / JSON-64.
  It is nevertheless a rejected wire-format candidate: bounded scalar Huffman is too slow for
  the 4 GB/s encode objective, and its all-Q95 ratios remain far above 0.05.
- The trace proves why metadata micro-packing cannot close the target alone. Existing literal
  streams are 1.764 MiB (11.0%) / 6.947 MiB (10.9%) of bursty input and 1.864 MiB (11.7%) /
  7.161 MiB (11.2%) of event input. Even a hypothetical zero-byte command stream exceeds the
  5% artifact cap for those members. P5 must next test an exact literal-stream representation
  and a command-count/matcher change together; details and promotion gates are frozen in
  `docs/TARGET_4_95_15_PLAN.md`.
- `cargo test --release --features bench` passed 8 packing + 2 analysis + 2 golden + 16
  property + 1 doctest tests, and `cargo test --release` passed its 2 golden + 16 property +
  1 doctest tests after this work. The global Cargo cache cleanup warning for `winapi-0.3.9`
  remains unrelated to successful compilation or test results.

## 2026-09-19 — P5 rejected bounded byte entropy for literal payloads

- The parser now exposes a bench-only exact literal-stream extraction: it validates every v2
  command/literal boundary before copying LZ-block literal bytes. A new literal oracle builds a
  raw payload and a canonical byte-Huffman payload for every LZ block, writes and charges a
  selector, table, bit length and padding, then reconstructs the exact original literal stream.
  The decoded literal length is the already validated command-stream total; the complete-total
  model retains the current v2 command/control/attribute/prefix accounting and replaces only
  the literal payload. It is not a v3 format or a CPU codec-speed measurement.
- All eight reports are retained in
  `reports/qualification/experiments/P5-literal-entropy-r1`. Byte entropy selects its payload
  for every LZ block and lowers literal bytes / current-frame ratio as follows:

  | corpus | raw literal MiB | encoded literal MiB | literal-only total ratio |
  | --- | ---: | ---: | ---: |
  | bursty 16 / 64 MiB | 1.764 / 6.947 | 0.946 / 3.725 | 0.269001 / 0.275782 |
  | event 16 / 64 MiB | 1.864 / 7.161 | 1.018 / 3.913 | 0.214096 / 0.240877 |
  | field-order 16 / 64 MiB | 0.412 / 1.637 | 0.242 / 0.961 | 0.196827 / 0.199034 |
  | JSON 16 / 64 MiB | 0.140 / 0.560 | 0.114 / 0.457 | 0.144882 / 0.145019 |

- This is meaningful literal savings (858,032 B bursty-16 and 886,957 B event-16) but not
  target feasibility. The encoded literal payload alone is still 5.9% of bursty-16 and 6.4% of
  event-16 input before any command, directory, block or dictionary byte exists. Combining the
  separately exact fixed-128 command model with this literal model is still
  0.261877/0.268919 bursty, 0.217354/0.243843 event, 0.188791/0.190776 field-order and
  0.118172/0.119190 JSON at 16/64 MiB. Those combined rows are a budget decomposition, not a
  joined v3 bitstream.
- Decision: do not integrate scalar byte-Huffman literal coding. It cannot meet the 0.05 cap,
  and its variable-bit scalar execution has no credible 15 GB/s decode path. The next P5
  candidate must be an exact bounded token/dictionary or structural-literal representation that
  removes substantially more literal bytes, paired with a bounded matcher experiment that lowers
  command/control cost. The frozen next-step gates remain in `docs/TARGET_4_95_15_PLAN.md`.

## 2026-09-19 — P5 rejected fixed 8-byte tokens and gated hash-2 matcher

- The literal selector was extended with deterministic, block-local 8-byte token tables at
  16/32/64/128-entry bounds. It builds RAW, canonical byte-Huffman and every token candidate
  before selecting one, serializes a selector/table/raw-runs/token IDs, removes unused entries
  canonically, and rejects duplicate/out-of-order/unused table entries, invalid references,
  output overrun and trailing bytes. Unit coverage makes RAW, Huffman and token modes each win
  on constructed inputs and exercises malformed table rejection.
- The complete all-Q95 record is
  `reports/qualification/experiments/P5-token-dictionary-r1`. Token tables are selected only
  by JSON (137/256 and 527/1023 LZ blocks at 16/64 MiB), saving 2,658 B / 9,102 B beyond
  byte-Huffman. Bursty, event and field-order select byte-Huffman in every LZ block, for exactly
  zero additional saved bytes. The `0.144724 / 0.144883` JSON ratios remain far from 0.05.
  Fixed eight-byte literal tokens are therefore not reopened unchanged.
- P5 then tested one bounded matcher change behind a temporary `matcher-hash2-gated` feature:
  retain only the immediately older bucket offset and inspect it only when the primary 8-byte
  match is 8–23 bytes, accepting it only when it adds at least eight matched bytes. There are no
  chains, lazy positions or unbounded windows. The feature passed default/candidate release
  suites and `compress()`/`compress_with_stats()` frame equivalence before reports were written.
- The all-Q95 candidate reports are in
  `reports/qualification/experiments/P5-hash2-gated-r1`. It reduces commands and fixed-128
  metadata on held-out families (for example field-order: −43,453 commands/−149,634 B at
  16 MiB and −172,075/−557,438 B at 64 MiB), but loses the compact JSON model
  (+783/+3,366 B fixed metadata) and no member approaches 0.05. The v2 frame deltas are
  bursty −28,087/−109,872 B, event −87,960/−336,715 B, field −173,330/−688,026 B and JSON
  −475/−1,757 B at 16/64 MiB.
- More decisively, its bounded work cannot pay its own encode cost. In same-condition,
  three-sample local direction checks, JSON-16 was 2.081 versus 2.591 GB/s for the default
  matcher; field-order-16 was 0.836 versus 1.096 GB/s. JSON performed 186,098 extra second
  checks/186,090 extra LCPs for only 161 selections. Field-order selected 46,327 candidates
  but required 407,785 extra checks and 402,845 extra LCPs. These are directional diagnostics,
  not acceptance statistics, but the loss is far beyond the normal noise rule.
- Decision: remove both temporary candidate feature/code paths after this checkpoint. The next
  experiment must not be another small secondary-hash threshold variation. To move the joint
  frontier, it needs a new exact block grammar that changes literal and command representation
  together, with a feasible all-Q95 byte budget before any public v3 integration.

## 2026-09-19 — P5 nested literal VXG is a retained feasibility control, not a format

- As a stronger generic literal control, each existing LZ block's validated literal stream was
  offered an independent checksum-free VXG subframe alongside RAW/Huffman/token choices. The
  outer selector, every nested header/directory/block byte, nested encode construction and safe
  nested decode are included. The experiment does **not** wrap a full VXG frame in another
  compressor and was never wired into public framing; it asks only whether the literal stream
  contains enough second-stage LZ structure to justify a new joint grammar.
- The eight exact reports are in
  `reports/qualification/experiments/P5-nested-literal-vxg-r1`. Nested subframes win on
  249/256 JSON-16 and 1005/1023 JSON-64 LZ blocks, reducing JSON literal payload by
  12,345 B / 49,481 B beyond fixed tokens (to 104,989 B / 420,564 B). Their total models are
  still 0.143988 / 0.144146. Bursty, event and field-order select Huffman in **every** LZ block
  and have exactly zero additional saved bytes, retaining 0.269001/0.275782,
  0.214096/0.240877 and 0.196827/0.199034 literal-only ratios at 16/64 MiB.
- Decision: remove the nested control after preserving reports/tests. It establishes that merely
  adding a second generic LZ layer to current literals cannot solve the held-out budget and has
  no 4 GB/s encode route. The next viable work is an explicit, jointly costed command-and-literal
  grammar feasibility prototype; no fourth literal-wrapper variation should be tried.

## 2026-09-19 — P5 rejected direct structural-byte grammar

- P5 tested an independent, lossless raw-domain grammar control before changing the v2 wire
  format. Per 64 KiB block it lexed JSON-like quoted strings and six punctuation bytes, then
  charged a canonical 16/32/64/128-entry repeated-span dictionary, tags, raw escapes, a block
  selector and the ordinary frame directory. Its decoder rejected noncanonical/unused dictionary
  entries, invalid references, output overrun and trailing bytes; the temporary module's three
  roundtrip/rejection tests passed. This was a feasibility oracle only: its reported throughput is
  grammar construction/reconstruction rate, never VXG codec GB/s.
- All eight valid JSON records are retained in
  `reports/qualification/experiments/P5-structural-grammar-r1`. The complete ratios at 16/64 MiB
  were `0.982668/0.978032` bursty, `0.967214/0.958055` event,
  `0.927871/0.927176` field-order and `0.922730/0.905680` JSON. Grammar selection occurred on
  only 14/71 bursty, 16/82 event, 35/142 field-order and 31/152 JSON blocks; RAW dominated every
  member even before a production-safe decode cost is considered.
- Decision: remove the direct-grammar prototype and binary using scoped patches. Syntax tags plus
  generic span references cannot replace the current literal-and-command budget. The next
  experiment must encode a common record/template *and* its variable fields/matches in one
  complete block grammar; merely decorating raw input is explicitly out of scope.

## 2026-09-19 — P5 record-template grammar reaches JSON 5%, not all-Q95

- The next exact, bench-only control did encode skeletons and values jointly: each independent
  64 KiB block formed a canonical 16/32/64/128-entry record-skeleton table, retained RAW records
  and selected the smallest complete payload against an inline representation. Its columnar form
  transposed every selected template's values and competed RAW, bounded categorical dictionary,
  delta-RLE, **actual bit-sliced delta planes**, quoted decimal-prefix deltas, lowercase hex
  nibble packing and bounded (period <=16) periodic-hex packing. Table, rows, column type,
  affixes, lengths, packed-nibble padding, selector and ordinary directory bytes are charged.
- The decoder rejects noncanonical/unused template tables, invalid references, noncanonical
  ULEBs and delta runs, overflow, invalid column symbols and nonzero bit-slice/hex padding. The
  retained test coverage includes JSON-like quoted/escaped/numeric/bool roundtrip, raw tail,
  malformed table rejection, selected bit-slice exactness plus padding rejection, and selected
  hex/periodic-hex roundtrips. `cargo test --release --features bench record` passed all five.
- The complete R4 record is
  `reports/qualification/experiments/P5-record-template-r4` (all eight JSON reports parse).
  At 16/64 MiB the fully charged ratios are:

  | corpus | ratio (16 MiB) | ratio (64 MiB) |
  | --- | ---: | ---: |
  | bursty | 0.237793 | 0.236834 |
  | event | 0.267403 | 0.266130 |
  | field-order | 0.074979 | 0.074780 |
  | JSON | 0.041067 | 0.040884 |

- This is a real ratio breakthrough for JSON, but not a qualification success: every block does
  select the grammar, while unmodelled high-entropy-looking value columns dominate bursty/event.
  Event-16 has 121,904 rows, so retaining just one eight-byte identifier per row is 975,232 B
  (0.058128 ratio) before any other required byte. The corpus generator confirms those IDs are
  xorshift-derived; a generator-specific predictor is forbidden by the frozen acceptance rules.
  The recorded 0.018–0.033 GB/s construction and 0.260–0.391 GB/s reconstruction values are
  oracle-only and cannot be compared to VXG/LZ4 speed targets.
- Decision: preserve R1/R2/R3/R4 artifacts but remove the temporary record-format module and
  binary. It is not a generic all-Q95 ratio solution, is far from the speed contract, and must
  not be mistaken for a public v3 format. A future candidate needs a genuinely generic,
  serialized, independently validated cross-record model for opaque identifier columns—not a
  corpus-generator-specific rule.

## 2026-09-19 — P6 checked opcode-7 non-overlap copy

- The ordinary safe decoder already establishes, for each fixed-size v2 opcode-7 command, a
  positive distance within the initialized prefix, an exact literal range and a checked output
  range. When `run <= distance`, those checks also prove the source and destination are disjoint.
  That safe branch now emits one `copy_nonoverlapping` directly instead of entering
  `Engine::repeat`, which would repeat the dispatch/slice split before taking the same platform
  copy. Overlap still calls the existing phase-preserving periodic engine.
- `decode-op7-inline-copy-generic` is a bench-only control that restores the old repeat call.
  Both default and control builds passed `cargo test --release --features bench` (30 tests and
  doctest), including exhaustive short inputs, malformed/truncated-frame checks, v1/v2 vectors,
  periodic-copy phases and SIMD tails. No frame byte, parser rule, validation rule, checksum or
  public API changed.
- A same-machine ordered JSON-16 pair, one worker, 64 KiB blocks, Neon, checksums off, nine
  0.2-second samples measured checked prepared decode at **3.681 GB/s** with the direct safe copy
  versus **3.371 GB/s** for the control (+9.2%, directional only). A fresh full comparison put
  the current artifact at 0.147 ratio, 2.624 GB/s allocating encode and 3.764 GB/s checked
  prepared decode; LZ4-a1 was 0.150/2.428/**9.158** (ratio/encode/decode). Proof-backed replay
  was recorded separately at 3.970 GB/s and is not used for the LZ4 claim.
- The frozen one-process smoke record is
  `reports/qualification/runs/P6-op7-inline-copy-smoke-r1`: manifest SHA verified, binary hash
  `b678f9cbb27e94f515dbc80c4ecce4a93fd8e5fe830f75447bc3983e6c66bc8a`, all eight members and
  all raw samples/logs retained. Its VXG/LZ4-a1 safe rows show the remaining range: VXG
  1.745–3.884 GB/s versus LZ4 4.740–9.449 GB/s, while v2 ratios remain 0.146–0.326. It is a
  three-short-sample smoke, not a paired confidence result or a target pass.
- Decision: retain the low-risk safe copy improvement, but do not claim G4/G5 or spend another
  small copy-loop cycle. The safe command executor is roughly 3.8 GB/s on its best Q95 member
  while complete all-Q95 ratio is still blocked; the next throughput candidate must remove
  demonstrated command-execution work in a ratio-feasible format, rather than hiding validation
  or advertising prevalidated replay.

## 2026-09-19 — P6 command budget and P5D/P6B handoff gates

- The retained P2 trace and the longer P6 JSON-16 local comparison were combined in
  `reports/qualification/experiments/P6-command-budget-r1.json`. JSON has 591,146 logical
  commands and 28.380833 output bytes per command. The directional safe/prevalidated rows are
  3.7643/3.9696 GB/s, or about 132.635/139.869 M commands/s (7.539/7.150 ns per command).
  The 10 GB/s target on that unchanged trace needs 352.350 M commands/s (2.838 ns), and the
  1.20x LZ4-a1 minimum needs 387.205 M commands/s (2.583 ns). These calculations use rounded
  CSV inputs, are a necessary-cost model rather than a target benchmark, and make no replay-as-
  safe claim.
- The measured-alone validation row is 8.6031 GB/s while safe reconstruction is much slower;
  proof-backed replay improves only 5.5% over safe prepared decode. The rows do not add linearly
  because the public decoder fuses parsing, checking and output work, but they rule out treating
  validation removal as the 10 GB/s solution. Another spelling of a v2 small copy is now barred
  unless a successful profile identifies a dominant operation.
- The implementation handoff is written in `docs/TARGET_4_95_15_PLAN.md`: P6B is a bench-only
  64-command scalar-then-Neon bitsliced group executor with strict malformed/canonicality gates,
  separate unpack/replay/safe timings and a required prevalidated rate above 352.350 M commands/s
  before further investment. It preserves causal output order and overlap-safe execution; the
  low within-eight-group reference count permits an unrolling hypothesis but does not imply an
  ARM gather/scatter decoder.
- P5D is the sole newly authorized ratio experiment: a source-agnostic, block-local serialized
  structured-column model family. Decimal/hex/timestamp/categorical representations and bounded
  affine/GF(2) recurrence candidates must be selected by exact charged description length for
  every eligible column, carry their full model and residuals inside each block, and survive
  field-name/seed/nonlinear-control robustness checks. A predicate tied to a field name, corpus
  path, known seed or generator constant is explicitly invalid. P3 bitslice, int4 and ternary
  remain execution/distribution experiments only; none supplies a ratio win by itself.
- Allocator/arena work is deferred behind P5D feasibility and allocation counters. It cannot
  accelerate the target preallocated safe-decode executor, and any encode result must retain the
  allocating public API and byte-identical artifact.

## 2026-09-19 — P5 rejected generic identifier delta planes

- The R4 record frontier showed that event's opaque 64-bit IDs alone exceed 5% when retained as
  raw binary values. Before considering an inadmissible generator-specific predictor, P5 tested
  the smallest generic cross-record transforms: XOR delta and wrapping-subtract delta, each
  viewed as 64 exact bit planes. The analysis parses only the textual 16-nibble ID values; it has
  no seed, recurrence, external dictionary, output cache, codec wire format or speed claim.
- The two retained reports are in
  `reports/qualification/experiments/P5-identifier-delta-analysis-r1`. At 16 MiB, 122,158 IDs
  require 977,264 binary bytes (0.058249 ratio) before record metadata. Their 122,157 XOR deltas
  contain 3,975,556 one bits in 7,818,048 plane bits and 3,956,926 bit transitions; wrapping
  subtraction is similarly dense (3,991,163 ones / 3,801,180 transitions). The 64 MiB member
  repeats the result: 3,889,760 raw bytes (0.057962), with XOR 15,823,726 ones and 15,748,901
  transitions in 31,118,016 plane bits.
- Decision: do not add an XOR-delta bit-slice/RLE v3 branch. It cannot improve the ID floor and
  would add parser/selector work. The accepted constraints now require either a genuinely generic
  serialized model that shows non-random structure before implementation, or an explicit target/
  workload renegotiation; corpus-generator logic remains disallowed.

## 2026-09-19 — P5D structured-column feasibility gate failed exactly

- Added the bench-only `vxg-column-eval` prototype. It creates a complete independent-block
  research packet: canonical template-segment dictionary, bit-packed row selectors, all model
  descriptors/payloads, RAW escapes, a 32-byte frame header and 32 bytes of directory per block.
  Decode validates and reconstructs every original byte; tests cover partial/escaped records,
  malformed and noncanonical packets, periodic forms, GF(2) recurrence and held-out exact
  reconstruction. It is not a VXG v3 path or a GB/s codec benchmark.
- R9 is the valid evidence set at
  `reports/qualification/experiments/P5D-structured-columns-r9`: all eight reports parse as JSON
  and each source packet roundtrips. At 64 MiB the fully charged ratios are bursty 0.083904117,
  event 0.078591824, field-order 0.050617427 and JSON 0.023074254; their 16 MiB companions are
  0.084347367, 0.078921139, 0.050787866 and 0.023171782. Scalar pack/unpack throughput ranges
  from 0.034–0.072 / 0.050–0.084 GB/s and is deliberately labelled research-only.
- The allowed learned GF(2) affine-recurrence model trains only from a serialized prefix,
  stores a complete 65-bit affine basis plus initial value, and accepts only exact suffixes.
  It selected zero columns across Q95. Thus it neither solved the event ID floor nor concealed a
  seed/generator rule. The prototype does not inspect field names, corpus path, source seed or
  nonlinear control values when choosing a model.
- `P5D-structured-columns-r2` remains preserved as an invalid report-serialization attempt; it
  is not used for any measurement or decision. R9 replaces earlier valid revisions as the final
  evidence for this family.
- Decision: the allowed P5D family fails the <=0.050000 all-Q95 feasibility gate (field-order
  misses narrowly; bursty/event miss substantially). Do not integrate it as v3 and do not spend
  more time on another layout variation. Bitsliced P6B work remains possible only as a separate
  decode experiment; it cannot repair this same-artifact ratio failure. Further ratio work needs
  an explicit target/workload renegotiation or a new, materially broader source-agnostic model
  authorization.

## 2026-09-19 — P6B bitsliced grouped-command stop gate failed

- Added the bench-only `vxg-group-eval` P6B scalar oracle. It emits a complete 64-command,
  bit-plane group packet from an exact v2 trace, including every field base/width, group count,
  literals and RAW passthrough. Validation checks truncation, group count, widths, zero plane
  tails, minimum-base/width canonicality, command semantics, literal bounds, causal references,
  seed padding and exact output consumption. A validated frame retains only one literal-start
  record per block—not a command tape—and replays commands with stack `[MetadataCommand; 64]`
  staging. The prototype is neither VXG v3 nor a public decode benchmark.
- The new exact mixed-command and malformed/noncanonical tests pass. The evaluator itself runs a
  complete safe reconstruction and rejects the report if bytes differ. Its retained JSON-16 R1
  report is `reports/qualification/experiments/P6B-grouped-command-r1/json-records-16MiB.json`.
  At five iterations on Neon: 591,146 commands in 9,346 groups, 0.112360299 synthetic packet
  ratio, 44.825M validation commands/s, 37.971M validated replay commands/s and 20.446M safe
  commands/s (1.078/0.580 GB/s replay/safe input rate).
- P6A requires at least 352.350M replay commands/s on this unchanged JSON trace before further
  investment. P6B R1 reaches only 10.8% of that necessary threshold, a 9.28x miss. Decision:
  do not write the Neon transpose; it cannot plausibly bridge the format/unpack and causal-output
  gap. Retain this as negative evidence and close P6B rather than advertising a metadata-unpack
  microbenchmark as safe decode.

## 2026-09-20 — P6P profiled bounded opcode-7 non-overlap copy retained

- A four-second macOS `sample` profile of JSON-16 safe prepared decode recorded 1,972 of 3,351
  sampled stacks (58.85%) under `platform_memmove`. The prior direct opcode-7 branch had already
  proven exact bounds, initialized prefix and non-overlap for `run <= distance`, but passed its
  variable size to the platform. This profile is retained as summarized evidence in
  `reports/qualification/experiments/P6P-bounded-op7-copy-r1/evidence.json`; it authorizes one
  bounded code-generation experiment, not a general copy-loop search.
- Added `copy_nonoverlapping_short` for exactly 8..=63-byte proven-disjoint runs. It uses only
  in-range unaligned u64 loads/stores, with a generic pointer copy for longer runs and the old
  phase-preserving repeat engine for overlap. The helper is also used in the immutable validated
  replay after its existing proof establishes the same invariant. Exhaustive length/alignment
  tests, default bench tests and the generic-copy control suite pass; no frame byte, public API,
  validation rule, block-independence rule or checksum behavior changed.
- The retained reversed-order JSON-16 pair is decisive at one worker/64 KiB/Neon/checksums-off,
  nine 0.2-second samples: candidate safe prepared decode is 7.5410 then 7.5404 GB/s; generic
  control is 3.7413 then 3.7465 GB/s. Both emit ratio 0.14650089. The corresponding validated
  replay changed from 4.0114 to 7.4960 GB/s, while the safe/validation rows are 7.5367/8.3042
  GB/s; replay remains an accounting diagnostic, never the safe result.
- `reports/qualification/runs/P6P-bounded-op7-copy-smoke-r1` is a manifest-verified eight-member
  serial smoke. Every source frame roundtrips via safe public decode. Its VXG safe range is
  2.6369–7.5963 GB/s (LZ4-a1 4.7671–9.3386), and unchanged ratios span 0.14650089–0.32611792.
  This preserves the helper as a real execution improvement but proves it is not the joint target:
  JSON is still below 10 GB/s and 1.20x LZ4; held-out members and ratio remain farther away.
- Decision: retain this profile-supported v2 helper and close more v2 copy spelling work. Any
  further target attempt needs an eligible ratio/artifact model that reduces causal command work;
  P5D and P6B remain closed negative paths.

## 2026-09-21 — P6E allocator/arena and NEON-LCP profile controls rejected

- Captured the JSON-16 one-worker/64 KiB/Neon/checksums-off allocating-encode profile and
  allocation counters in `reports/qualification/experiments/P6E-encode-profile-r1/evidence.json`.
  The public timed call made 18 allocations and 12 reallocations, but only 21 of 3,356 sampled
  stacks reached `RawVec` reserve/reallocation. `Engine::lcp` appeared in 1,261 stacks and block
  command construction in 418, so a general allocator replacement has no profile justification.
- Two exact NEON LCP variants were tested against the original scalar mismatch locator and then
  removed: word-prefix/trailing-zero lookup reached 2.0119 versus 2.5708 GB/s, while using the
  NEON equality mask directly reached 1.6965 versus 2.5965. Both preserve output bytes and pass
  the complete candidate/control suites, but the location logic is slower than the existing scan.
- The existing `Scratch.payloads` is already a worker-local linear arena. Reserving its full
  assigned raw capacity reduced reallocation but measured 2.5449 versus 2.5796 GB/s; dirty-slot
  hash reset measured 2.4977 versus the full-table fill's 2.5723. Both source candidates and
  their feature controls were removed. No mimalloc/jemalloc, arena/slab framework or new SIMD
  path is retained.
- Decision: retain the system allocator and current worker arena. Encode remains profile-bound by
  LCP/match-command work, while P5D still fails the all-Q95 ratio gate; no current authorized
  path reaches the joint 4/95/15 claim.

## 2026-09-21 — P6M actual Metal device validation passed; host-e2e speed remains behind CPU

- `cargo run --release --features metal --bin vxg -- gpu-check metal` executed on the Apple M2
  Max and passed all 81 three-way CPU/GPU conformance cases. This is actual-device evidence, not
  the earlier CPU shader/emulator substitute.
- The retained five-sample JSON-16 directional host-to-end comparison is
  `reports/qualification/experiments/P6M-metal-device-r1/evidence.json` with raw CSV samples in
  `reports/local-metal-json16-r1.csv`. Metal encodes/decodes at 0.736/1.255 GB/s with ratio
  0.164, against VXG CPU’s 2.487/7.280 GB/s with ratio 0.147 and LZ4-a1’s 2.271/8.812 GB/s with
  ratio 0.150. The one observed kernel timestamp is 9.224 ms versus 12.817 ms host-e2e; it is
  explicitly not reported as a median or as host throughput.
- Decision: Metal is now functionally verified but not a performance route for this workload.
  Keep GPU performance separate from the CPU 4/95/15 acceptance claim.

## 2026-09-21 — User revised CPU target to 4 / 0.06 / 12–15

- Added `docs/TARGET_4_006_12_15.md` as the current acceptance contract: 4 GB/s
  allocating encode, ratio <=0.06 (94% reduction), 12 GB/s safe full-frame decode
  minimum and 15 GB/s stretch. Previous relative LZ4 targets and frozen Q95 members
  are preserved. Historical plans/handoff link to this numerical override.
- Re-scored existing P5D R9 evidence only: JSON and field-order now pass ratio alone;
  event and bursty still need 1,247,677 and 1,604,179 fewer bytes at 64 MiB. These
  scalar research packets have not passed any production speed gate.
- Latest JSON-16 v2 full-frame baseline is 2.4865 GB/s allocating encode, 7.3703 GB/s
  safe decode and 0.14650089 ratio. No implementation or new benchmark was performed
  for this target revision. Next work is a bounded, generic model hypothesis backed
  by the event/bursty byte decomposition under the revised 6% ceiling.

## 2026-09-21 — P8 energy instrumentation and ratio-budget follow-up

- Added `vxg-energy-workload` and `scripts/energy_bench.py` for the user-requested
  fourth axis: encode/decode CPU-rail J/GB versus real LZ4 a1/a4 and Zstd -3/1/3.
  Each phase uses one worker, 64 KiB independent blocks, actual allocating encode,
  checked decode and pre/post byte checks. VXG full-frame parsing remains timed;
  baseline prepared-block accounting is explicit. The runner preserves binary/input
  hashes, shuffled order, completed-work counters, power logs and pre/post idle power.
- The 12-path `P8-energy-workload-smoke-r1` run passed without sensors. Six energy
  accounting tests pass. These are infrastructure checks: its energy fields are null,
  and its 0.2-second rows are not a speed qualification. Real sensor access was tried
  directly and through the runner; `P8-energy-access-check-r1/run.json` preserves
  `sudo: a password is required`. The user's authenticated `P8-energy-r1` run obtained
  a real preflight sample but failed before codec phases because the parser omitted
  the date's `+0300` suffix. Fixed numeric-offset parsing with legacy-header fallback,
  verified the saved full log and added a regression covering positive/negative/zero
  offsets and invalid offsets. All six tests also pass under UTC and Los Angeles TZ.
  R1 is preserved; the corrected retry uses R2 in the user's terminal because the
  agent's sudo session still lacks credentials. No energy ranking or whole-system
  joules are claimed. See `docs/ENERGY_BENCHMARKS.md`.
- P8's generic hex-period audit examined 534,801 eligible lexical values across all
  1024 bursty-64 blocks. They already admit a common period of 15. Per-value periods
  cost 5,154,950 bytes versus the lexical common-period reference's 4,625,905: 529,045
  extra bytes. The variant is rejected before touching the codec. R1 raw-nibble-relative
  savings are retained but mostly duplicate savings already present in P5D; R2 provides
  the correct existing-model comparison and an explicit decision.
- The audited seed bytes alone are 4,008,130 against a complete-frame ceiling of
  4,026,531; the actual P5D R9 bursty frame is 5,630,710 bytes. Further layout changes
  need a charged route for lengths/other fields or new evidence of seed redundancy.
  Next analysis is a bounded, generic conditional model across serialized values and
  lengths, with prefix training, held-out predictions, descriptor cost and RAW escapes.

## 2026-09-26 — User decision B: Q95-r2 scope revision and first r2 evidence

- Follow-up to `docs/E_B_FRONTIER_REVIEW.md` (this date). The review quantified that the
  two failing members are dominated by per-record xorshift64 value streams (event IDs
  96.6% and bursty message seeds 99.5% of the 0.06 budget) and falsified two bounded
  model families on the frozen bytes (counter-affine GF(2) fits and order 1–4 linear
  recurrences; both inconsistent under a brute-force-verified solver). The user chose
  option B: revise the failing families' value sources, freeze a Q95-r2 corpus and pursue
  the target numbers on r2 with generic models only.
- `scripts/generate_qualification_corpus.py` gained the r2 revision: `--r2` selects
  `corpus/qualification/r2/` and `reports/qualification/manifest-r2.json` (revision
  `q95-json-r2`). `event_mix_r2` replaces per-record random identifiers with UUIDv7-like
  epoch prefixes (128-record runs share a 32-bit prefix) plus an ascending 32-bit
  sequence and ~3% bounded random-suffix exceptions. `bursty_records_r2` replaces the
  60-bit periodic hex pattern with repeated phrases from a 12-phrase vocabulary, bounded
  tail words and occasional decimal parameters; record skeletons, templates, burst
  schedule and the exact message-length law are unchanged. `json-records` and
  `field-order-optional-escaping` regenerate byte-identical to r1 (SHA-256 equal); r1
  remains intact as an adversarial control set.
- `scripts/run_qualification.sh` accepts `MANIFEST=` (default unchanged), so r2 runs do
  not touch the r1 manifest. Verified: `--r2` regenerates and verifies all eight r2
  members (320 MiB, ~4.5 s); the default r1 `--verify` still passes.
- First r2 evidence (v2 codec, 1 worker, 64 KiB, Neon, checksums off, 5 x 0.2 s, 16 MiB):
  event v2 ratio 0.267 -> 0.185 (encode 0.786 -> 1.352 GB/s), bursty 0.320 -> 0.267
  (0.819 -> 0.894 GB/s); LZ4-a1 event 0.250 -> 0.154, bursty 0.244 -> 0.207. The existing
  P5D packet prototype re-run on the r2 members (`P10-columns-r2`) measures event 0.078940,
  still dominated by `quoted-fixed-width-lower-hex-suffix` (977,016 B, raw 8-byte IDs) and
  bursty 0.378333, dominated by `canonical-byte-dictionary` (6,116,762 B, no phrase-token
  representation). These are expected negative results for the old model set: Faz 2 must
  add two bounded, serialized, generic column models (hex epoch-prefix dictionary plus
  ascending suffix delta; phrase/word token stream plus pad count) before the r2 packet
  can approach <=0.05. No ratio or speed claim is made for r2 yet.
- Decision recorded in `docs/TARGET_4_006_12_15.md`: primary r2 ratio target <=0.050000,
  0.06 historical; `docs/VXG_R2_TARGET_PLAN.md` holds the phase plan. R1 experiments and
  reports remain valid history; they are not rerun or rewritten.

## 2026-09-26 — P10: Q95-r2 ratio gate passes with three new bounded column models

- The existing packet model set could not exploit the revised value sources (event
  0.078940, bursty 0.378333, field-order 0.050617 on r2). Three bench-only, source-agnostic
  column models were added to `src/columns.rs` / `vxg-column-eval`, each serializing its
  complete state inside the independent block, selected only by exact charged length, and
  covered by unit tests:
  - `quoted-fixed-width-hex-epoch-counter`: constant literal prefix, canonical ascending
    block-local 32-bit prefix dictionary, bit-packed prefix indices, a one-bit "previous
    suffix + 1" flag stream and a ULEB exception stream. Rejects non-canonical dictionaries,
    first-row sequential flags, exception-count mismatches and out-of-range suffixes.
  - `quoted-semicolon-segment-dictionary`: exact `; `-separated segment grammar with a
    bounded canonical segment dictionary, bit-packed codes (dictionary ids, decimal escape,
    terminator), bit-packed decimal values and a bit-packed trailing-pad stream. Rejects
    empty/oversized/non-lowercase-alnum entries, non-canonical decimal entries, width and
    count mismatches and missing terminators.
  - `canonical-decimal-delta-bitpack`: zigzag delta bit packing over canonical decimal
    values; rejects width mismatches, range overflow, truncation and trailing bytes.
- Gate A result (`reports/qualification/experiments/P10-columns-r2`, all eight r2 members,
  32-byte header + 32-byte directory per block charged, scalar oracle scans):

  | member | ratio 16 MiB | ratio 64 MiB |
  | --- | ---: | ---: |
  | json-records | 0.023172 | 0.023074 |
  | field-order-optional-escaping | 0.044933 | 0.044785 |
  | event-mix-identifiers | 0.025920 | 0.025826 |
  | bursty-variable-records | 0.048823 | 0.048576 |

  Every member is <= 0.050000. The bursty margin is the thinnest (2.4% / 2.8% of the
  per-file budget); a phrase-escape code split is the next measured candidate if later v3
  framing overhead grows. No speed claim is made; the packets remain scalar oracles.
- Robustness controls (`scripts/make_r2_controls.py`, reports in `P10-columns-r2-controls`):
  renamed-key inputs (values and rows unchanged) behave identically modulo the smaller file
  size, and deterministically shuffled inputs preserve exact reconstruction while
  order-dependent models degrade to explicit residuals/flags as designed. They are controls,
  not qualification members.
- Validation: `cargo test --release --features bench` passes 22 lib tests including the
  10 column tests, plus 2 golden, 2 analysis, 16 property and the doctest; the default
  release suite also passes. Phase 3 (v3 grammar and integration) is next.

## 2026-09-27 — Faz 3: v3 structured-column blocks integrated into the codec

- `Options { v3: bool }` (default false) emits version-3 frames. For every independent
  block the encoder builds the existing LZ/RAW payload first, then a structured packet via
  `columns::encode_grammar_block`, and stores the packet only when strictly smaller
  (`packet.len() < lz.len()`); ties and losses keep LZ/RAW. The choice is charged and
  deterministic. `block::COLUMN = 2` is the new frame kind; the decoder dispatches it to
  the safe packet decoder (`columns::decode_block`, no unsafe on that path). The public
  wrappers are `columns::encode_grammar_block` / `columns::decode_block`.
- Version handling: decoders accept 1..=3; v2 must not contain COLUMN entries; a COLUMN
  entry requires `2 <= payload_len < raw_len`. Command validation and validated replay
  return `Error::Unsupported` for COLUMN frames instead of reinterpreting them; GPU paths
  inherit that refusal. `vxg-bench --v3` emits v3 frames and skips the command-validation
  profiles with an explicit note; `--validated-decode` on a structured frame is skipped
  with a note as well.
- Integrated evidence (16 MiB, one worker, 64 KiB blocks, checksums off, 3 x 0.1 s
  directional samples): ratio json-records 0.147 -> 0.023, r2 event-mix-identifiers
  0.185 -> 0.026, r2 bursty-variable-records 0.267 -> 0.049. Frame sizes match the P10
  packet accounting; the grammar and canonicality rules are frozen in `docs/FORMAT_V3.md`.
- Speed is explicitly research-slow on the COLUMN path (encode 0.050-0.063 GB/s, safe
  decode 0.056-0.081 GB/s under the same samples): the 4-6 GB/s encode and 12-15 GB/s
  decode targets remain the Faz 4/5 work. No speed claim is made.
- Tests: new `tests/v3.rs` covers roundtrip, determinism, `compress`/
  `compress_with_stats` equality, multiworker encode/decode, the per-block API,
  validated-replay refusal, and rejection of corrupted/truncated/downgraded/future-version
  frames. Full default and bench suites pass (22 lib including 10 column tests, 4 v3,
  2 golden, 2 analysis, 16 property, doctest); the roundtrip fuzz target now covers v3.
- Observed, not changed: the v2 RAW fast-path sampler can false-positive on some
  compressible synthetic blocks (the v3 test input produced `raw_fast=3` of 5 blocks and a
  v2 ratio of 0.78 while the v3 packet needs 5.3 KB). v3 bypasses the sampler because the
  packet wins regardless; any sampler change needs its own paired experiment.

## 2026-09-27 — P11: v3 fast decode path (production decode, byte-wise bits, flat values)

- Production/strict split: `columns::decode_block` (used by the frame decoder) performs
  every structural validation but no longer re-encodes each model for canonicality;
  `decode_block_strict` keeps the full re-encode equality check for the research oracle
  and the canonicality tests. `vxg-column-eval` and `evaluate`/`time` stay strict.
- Bit streams are packed/unpacked byte-wise instead of bit-by-bit (`BitWriter::push`,
  `BitReader::read`), and decoded values live in one flat buffer per column
  (`FlatValues`) written directly by every model decoder instead of one heap `Vec<u8>` per
  value. Periodic-arm lengths are now bounded by the block maximum; timestamp and decimal
  reconstruction no longer go through `format!`/`String`.
- `sample` evidence before the work: about 47% of decode stacks under malloc/free
  (per-value allocations), about 18% memmove, about 6% Vec growth inside row assembly.
- Measured on JSON-16 (one worker, 64 KiB blocks, checksums off, `--v3 --vxg-only
  --decode-only`, two runs of seven 0.2 s samples): safe full-frame decode 0.083 -> 0.725
  GB/s (x8.7 total; 0.083 -> 0.366 from removing the decode-time re-encode, 0.366 -> 0.725
  from flat values and byte-wise bits). Ratio unchanged at 0.023; encode is unchanged at
  0.066 GB/s and is the next target.
- Tests: all suites pass (23 lib including 10 column tests plus the non-minimal-model
  test, 4 v3, 2 golden, 2 analysis, 16 property, doctest). `docs/FORMAT_V3.md` documents
  the strict-vs-production canonicality split.
- No speed claim: the 12 GB/s decode and 4-6 GB/s encode targets remain open; this entry is
  an intermediate directional measurement under the retained protocol.

## 2026-09-27 — P12: structured-encode allocation removal (range parser, shared parses)

- The record parser no longer copies anything per record: `parse_record` returns
  `(u32, u32)` byte ranges into the block, templates store per-slot range lists
  (`Template::values: Vec<Vec<(u32, u32)>>`) and `grammar_from_block` resolves slot
  columns as borrowed slices; template matching compares owned segment bytes against the
  block ranges. Every model encoder now takes `&[&[u8]]` (`pack_model_values` is the
  owned-vector test/oracle wrapper).
- Model race rebuilt: one shared `parse_decimal` pass feeds the affine/packed/delta
  family (previously three separate parses per column), `parse_decimal` is a direct digit
  loop instead of `from_utf8 + str::parse`, and every one of the twelve models checks its
  exact serialized size (or a proven lower bound for GF(2)) against the current best
  before constructing bytes, so hopeless models skip their bit-packing entirely. Race
  order and winner selection are unchanged; `BitWriter`/`BitReader` keep the frozen
  grammar.
- `sample` evidence before the work: allocator family about 57% of encode stacks (release:
  per-record Vec allocations), memcmp about 15%, model serialization the rest. After: the
  allocation family drops to roughly a third and the parser is no longer a distinct leaf.
- Measured on JSON-16 (one worker, 64 KiB blocks, checksums off, `--v3 --vxg-only`, nine
  0.2 s samples): allocating encode 0.066 -> 0.113 GB/s (0.066 -> 0.094 from the range
  parser, 0.094 -> 0.113 from the race work). Ratio unchanged at 0.023. All suites pass,
  including the ten column tests, `tests/v3.rs`, golden/analysis/property suites.
- Honest status: the 4-6 GB/s encode target remains open by more than an order of
  magnitude; this is an intermediate directional step under the retained protocol.

## 2026-09-27 — P13: structured-decode copy fast path and assembly tightening

- `sample` evidence at entry: memmove/append_elements about 53% of decode stacks, decimal
  formatting about 8%, bit reads about 6% — millions of 1..16-byte `memcpy` calls per
  16 MiB dominated the decoder.
- `copy_bytes` appends through one or two overlapping fixed-width stores (reserved with
  16 bytes of slack) and only falls back to the platform copy above 32 bytes; it is used
  by row assembly, flat value pushes and decimal formatting. `BitReader::read` loads one
  little-endian 64-bit window per read instead of a byte loop, `write_decimal` uses a
  two-digit table with an inlined append, and the row-assembly slot loop runs on raw
  pointers (segment head table, column table, cursor table) after the slot-count
  accounting check, removing per-slot bounds checks and `Result` iterator chains.
- Directional eight-member sweep (one worker, 64 KiB blocks, checksums off, `--v3
  --vxg-only`, five 0.2 s samples per member, ratio/encode/decode GB/s):
  json-16 0.023/0.114/1.050, json-64 0.023/0.113/1.088, event-16 0.026/0.095/0.955,
  event-64 0.026/0.094/0.957, field-16 0.045/0.090/0.882, field-64 0.045/0.094/0.914,
  bursty-16 0.049/0.074/0.721, bursty-64 0.049/0.072/0.728. All eight members keep the
  ratio gate; decode improved from 0.725 to 1.05-1.10 GB/s on JSON-16 in this round
  (0.725 -> 0.938 from the copy/bit/decimal changes, -> 1.10 after the assembly
  tightening).
- Honest status: the 12-15 GB/s decode target remains open by roughly a factor of eleven;
  the copy-and-assemble design has a scalar ceiling well below it, so reaching the target
  needs either SIMD row expansion or a different packet layout, both of which are open
  design work. No target is marked complete by this entry.

## 2026-09-27 — P14: structured decode writes into the caller's buffer (sink, direct output)

- `OutSink` replaces per-piece `Vec` bookkeeping in row assembly with one reserved block
  buffer, a raw position counter and the same overlapping-store scheme; the final pieces
  of an exactly-sized target fall back to an exact copy so no byte outside the block is
  ever written. `decode_block_into(src, out)` assembles directly into the frame output
  slice; `block::decode` uses it, so a structured block no longer allocates an
  intermediate `Vec` and no longer pays a whole-block `engine.copy`. `decode_block`
  (owned `Vec`) remains the research/API path, built from the same inner decoder.
- Every `FlatValues` push path (`push_slice`, `begin`/`finish`, `push_decimal`) is now
  infallible: `unpack_model_mode` rejects payloads above `u32::MAX` once, so value offsets
  are plain `u32` writes and the per-value `Result` construction/drop disappears from all
  thirteen model decoders. `write_decimal` gained single-store fast paths for 1-4 digit
  values (two-digit table), and per-model value-buffer estimates reduce `FlatValues`
  growth (`memmove`) for hex/epoch/segment columns.
- Directional eight-member sweep after this entry (five 0.2 s samples per member, same
  protocol as P13): json-16 0.023/0.109/1.255, json-64 0.023/0.114/1.304, event-16
  0.026/0.096/1.008, event-64 0.026/0.097/1.015, field-16 0.045/0.097/1.008, field-64
  0.045/0.094/0.970, bursty-16 0.049/0.075/0.767, bursty-64 0.049/0.074/0.774. Decode
  gained 7-10% over the P13 table on every member; all eight keep the ratio gate.
- Remaining decode distribution (`sample`, JSON-16 decode-only): row assembly 34%,
  `BitReader::read` 16%, decimal writing 10%, memmove 10%, error drop glue 6.6%. Verify
  with `cargo test --release --features bench` (23 lib + 25 integration/doctest tests all
  green) and `scripts/check.sh`.
- Honest status: the 12-15 GB/s decode target and the 4-6 GB/s encode target remain open.
  Decode is now about 1.3 GB/s on JSON-64 and about 0.77 GB/s on the bursty member; the
  next order-of-magnitude step is a batched (SIMD) assembly and value-expansion design,
  not more scalar tightening.

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

## 2026-09-19 — Stream-shape telemetry and direct non-overlap repeat

- Benchmark-only stream telemetry now reports opcode-7 distance and match-run buckets plus the
  number of zero-literal commands; these fields and increments are absent from normal library
  builds. At 16 MiB JSON, 480,120 of 591,146 commands had no literals. Opcode-7 distances were
  almost entirely above 256 bytes, while all match runs were 16–63 bytes. The decoder therefore
  usually has a non-overlapping, short one-copy match rather than a doubling/overlap case.
- `Engine::repeat` now recognizes `len <= distance` and directly uses the platform copy after
  proving the source and destination non-overlap from that condition. The old generic
  Engine-copy path remains available as the benchmark-only `repeat-engine-copy` control.
- In the matched 16 MiB JSON decode setup (one thread, 64 KiB blocks, nine 0.2 s samples), the
  direct path measured 2.579 GB/s versus 2.485 GB/s (+3.8%) with unchanged 0.147 ratio. This is
  an ordered process pair. Local artifacts: `reports/local-stream-shape-2026-09-19.csv`,
  `reports/local-repeat-direct-copy-2026-09-19.csv`, and
  `reports/local-repeat-engine-copy-control-2026-09-19.csv`.

## 2026-09-19 — Rejected NEON vector mismatch locator

- The encoder's JSON path is LCP-heavy, so a NEON experiment replaced the scalar scan used after
  a mismatching 16-byte compare. It compressed adjacent mismatch flags into eight pair flags and
  used `trailing_zeros` to select the first mismatching pair, then compared at most two scalar
  bytes. The vector loads and shuffle/unzip operations added more latency than the short scalar
  scan avoided.
- On Apple M2 Max with 16 MiB JSON, one thread, 64 KiB blocks, nine 0.2 s samples and Neon,
  the vector locator encoded at 1.395 GB/s versus 1.999 GB/s for the old scalar-tail control;
  prepared decode was 2.619 versus 2.587 GB/s, confirming that only the encoder path changed.
  The candidate and control feature were removed. Local artifacts:
  `reports/local-neon-lcp-mask-2026-09-19.csv` and
  `reports/local-neon-lcp-scalar-tail-2026-09-19.csv`.

## 2026-09-19 — Fused opcode-7 CPU decode

- JSON's fixed-size v2 opcode-7 stream is dense enough that even the dedicated parser path was
  still materializing and immediately destructuring a `Command` for nearly every match. The CPU
  decoder now recognizes canonical fixed-size opcode-7 tokens before generic parsing and fuses
  bounds checks, literal copy, repeat execution, and parser-state updates in one inlined path.
  Extended opcode-7 lengths, v1 data, and every other opcode retain the generic parser.
- The fused path preserves the generic path's stream bounds, output bounds, literal bounds,
  minimum-run, attribute truncation, and initialized-prefix checks. It still calls the parser's
  exact-consumption check at block end. `decode-op7-direct-generic` is a benchmark-only control
  that disables only the fused path.
- On Apple M2 Max with JSON only, 16 MiB, one thread, 64 KiB blocks, Neon, nine 0.2 s samples,
  checksums off, prepared decode measured 3.236 GB/s with the fused path versus 2.599 GB/s with
  the generic control (+24.5%), with unchanged 0.147 ratio. Local artifacts:
  `reports/local-op7-direct-decode-2026-09-19.csv` and
  `reports/local-op7-direct-generic-2026-09-19.csv`.

## 2026-09-19 — Rejected scalar-prefix NEON LCP

- A short-match encoder experiment checked the first 32 LCP bytes as four scalar `u64` words,
  falling through to the 32-byte unrolled Neon loop only for longer matches. It was motivated by
  JSON's roughly 20-byte average LCP extension, but the added scalar prefix did not beat the
  existing vector-first path.
- On Apple M2 Max with 16 MiB JSON, one thread, 64 KiB blocks, nine 0.2 s samples and Neon,
  the hybrid encoded at 2.013 GB/s versus 2.070 GB/s for the vector-first control. The candidate
  and its `neon-lcp-vector-prefix` benchmark feature were removed. Local artifacts:
  `reports/local-neon-lcp-hybrid-2026-09-19.csv` and
  `reports/local-neon-lcp-vector-prefix-2026-09-19.csv`.

## 2026-09-19 — Rejected matcher miss-branch simplification

- The encoder briefly split empty hash slots from occupied slots and removed the defensive
  `candidate >= pos` test from the latter. It retained a benchmark-only legacy branch for an
  exact control, but this structural cleanup did not improve the measured hot path.
- On Apple M2 Max with 16 MiB JSON, one thread, 64 KiB blocks, nine 0.2 s samples and Neon,
  the simplified path encoded at 1.974 GB/s versus 1.996 GB/s for the legacy control. The code
  and `matcher-legacy-miss-branch` control feature were removed. Local artifacts:
  `reports/local-matcher-miss-fast-2026-09-19.csv` and
  `reports/local-matcher-miss-legacy-2026-09-19.csv`.

## 2026-09-19 — Stats-free normal CPU encode

- `compress()` previously delegated to `compress_with_stats()`, so its hot matcher updated
  `WorkStats` counters even when the caller discarded them. Encoding and worker dispatch are now
  const-generic over stats collection: the regular `compress()` instantiation elides those
  increments at compile time, while `compress_with_stats()` retains identical telemetry and the
  benchmark's setup frame still reports it.
- `encode-with-stats` makes the benchmark time the old telemetry-collecting API as a control.
  On Apple M2 Max with JSON only, 16 MiB, one thread, 64 KiB blocks, Neon, nine 0.2 s samples,
  checksums off, normal allocating encode measured 2.519 GB/s versus 1.986 GB/s with the stats
  control (+26.8%). Both produced ratio 0.147; prepared decode was statistically unchanged at
  3.206 versus 3.211 GB/s. Local artifacts: `reports/local-encode-no-stats-2026-09-19.csv` and
  `reports/local-encode-with-stats-2026-09-19.csv`.

## 2026-09-19 — CPU hot-path comparison refresh

- Fresh same-machine comparison after the fused opcode-7 decoder and stats-free normal encoder:
  16 MiB JSON, one thread, 64 KiB blocks, Neon, nine 0.2 s samples, checksums off; VXG uses
  allocating encode and prepared/preallocated decode, matching the documented harness policy.
- VXG measured ratio 0.147, encode 2.515 GB/s, decode 3.176 GB/s. LZ4-a1 was
  0.150/2.282/8.994 and LZ4-a4 was 0.151/2.281/9.005 (ratio/encode/decode GB/s). Zstd -3 was
  0.100/1.922/4.062; Zstd 1 was 0.054/1.602/3.215; Zstd 3 was 0.062/1.486/2.846.
- VXG now slightly beats both tested LZ4 configurations on this encode and ratio profile, and
  decodes near Zstd -3/level 1, but remains about 2.8x behind LZ4 decode. This is one ordered
  process run; use it as current directional evidence rather than a universal codec ranking.
  Local artifact: `reports/local-json-after-cpu-hotpaths-2026-09-19.csv`.

## 2026-09-19 — Validated-frame decode foundation

- Added `Frame::into_validated() -> ValidatedFrame`. It first runs the existing complete command
  validation and then retains the immutable frame bytes as the proof for repeated CPU decode.
  The normal `Frame::decode_into()` path remains the safe one-shot decoder and still validates
  commands while it executes them.
- `ValidatedFrame::decode_into()` uses that proof for its fixed-size v2 opcode-7 path: it avoids
  repeated stream/output/backreference checks and uses direct non-overlapping copies where the
  validated command permits them. Extended and uncommon commands retain the checked generic
  parser. The new API owns its validation proof, so this is not an unchecked public decode mode.
- At 16 MiB JSON, one thread, 64 KiB blocks, Neon, nine 0.2 s samples, checksums off, the safe
  prepared decoder measured 3.379 GB/s and the once-validated repeated decoder 3.952 GB/s
  (+17.0%). These profiles are intentionally separate; this gain excludes the one-time validation
  and is not compared directly to LZ4's per-call safe decode. Local artifacts:
  `reports/local-safe-decode-v2-2026-09-19.csv` and
  `reports/local-validated-decode-v2-2026-09-19.csv`.

## 2026-09-19 — 4 / 95 / 15 execution plan and Terra handoff

- Added `VXG_4_95_15_PLAN.md` and `TERRA_HANDOFF.md`: simultaneous >=4 GB/s allocating
  encode, <=0.05 complete-frame ratio, and >=10 GB/s safe decode plus >=1.20x LZ4;
  >=15 GB/s and >=1.50x LZ4 are the stretch gate. The qualification workload is a
  frozen structured JSON/log set, not arbitrary incompressible bytes.
- The plan starts with benchmark/validated-API audit, then exact copy and grouped
  decoding, metadata budgets, bitsliced/int4/ternary experiments, bounded matcher
  changes, and version/reference/backend qualification. It distinguishes safe
  per-call command validation from prevalidated repeated decode.
- This entry records planning only. No codec implementation, benchmark rerun, active
  long-running goal or Terra task was started by this planning step. Historical CSVs
  were inspected to ground the target and the ~9.19-bit-per-command metadata budget.
- Corrections for future work: no measured v2 4–5 GB/s ceiling has been established;
  the earlier stream-shape entry's "16–63" run range should read "8–63" according to
  its recorded histogram. The main plan also calls out the stale v1-only format document.

## 2026-10-14 — P15 vectorized v3 structured-column encode & decode frontier on Q95-r2

- **Decode (`src/columns.rs`):**
  - Added `PreparedColumn`, `UnboundTemplate`, `PreparedTemplate`, and 64-byte `FastPiece`
    inline `u128` stores so constant columns and adjacent literal segments fold together at
    block setup time and row emission checks output capacity once per block (`row_budget`).
  - Added 8-values-per-iteration `aarch64` NEON bit unpacking (`unpack_u8_swar` and
    `unpack_u16_swar` with `vmovl_u8` + `vaddq_u16` + `vst1q_u16`) and stack-resident
    `[PreparedColumn::empty(); MAX_SLOTS]` backed by `DecodeScratch.slice_arena`.
  - Optimized `parse_segment_dict` with `u8` fast paths for 1..=7-bit decimal and pad streams
    and branch-hoisted segment dictionary emission.
- **Encode (`src/columns.rs`, `src/lib.rs`, `src/frame.rs`, `src/block.rs`):**
  - Updated `encode_worker` to encode `v3` structured blocks directly into the worker payload
    arena (`encode_grammar_block_into`) first and skip `block::encode` when `packet_len < raw_len / 2`;
    lazily allocated `Scratch.table` only when `block::encode` runs; updated `assemble_workers`
    to reserve the exact compressed frame size (`table_end + total_payload_len`).
  - Added single-pass clean-template stream matching (`try_match_clean_stream`) with NEON
    nibble-mask string scanning (`vshrn_n_u16`), shared-prefix template switching
    (`shared_segs`), and thread-local `ENCODE_VEC_POOL` reuse.
  - Optimized column model selection and packing: 4/8-digit SWAR `parse_decimal`, 64-bit
    delta/affine detection, `HT_CAP = 256` stack hash table with 2-row O(1) lower-bound abort
    in `pack_dictionary_model`, cached prefix/word checks in `pack_epoch_counter_model` and
    `pack_timestamp_model`, 64-bit `CLZ` Gaussian elimination in `pack_gf2_affine_model`, and
    64-bit word writes in `push_bit_values`.
- **Validation & Q95-r2 8-member results:**
  - `scripts/check.sh`, `scripts/check_model.sh` (WG=8 and WG=64 ASAN/UBSAN), and the full
    release + bench test suites pass.
  - Across all 8 `Q95-r2` members (`threads=1`, `block=65536`, checksums off,
    `reports/r2-v3-all-8-members.csv`), `vxg --v3` achieves ratios `0.0231`–`0.0488`
    (all `<= 0.050000`), allocating encode `1.015`–`1.790 GB/s`, and safe full-frame decode
    `3.852`–`10.048 GB/s`.

