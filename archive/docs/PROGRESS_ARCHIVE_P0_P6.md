# VXG Development Progress Archive — P0 to P6E (2026-09-19 to 2026-09-21)

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

