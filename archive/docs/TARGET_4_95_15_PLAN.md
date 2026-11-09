# VXG 4 / 95 / 15 execution plan

> Target revision (2026-09-21): [TARGET_4_006_12_15.md](TARGET_4_006_12_15.md) now
> governs acceptance: >=4 GB/s allocating encode, ratio <=0.06, >=12 GB/s safe
> full-frame decode, 15 GB/s stretch. Thresholds and renegotiation requirements
> below refer to the historical 5% target; retain them as experiment history.

This is the handoff plan for one concrete claim, not a wish list. A result is promotable only
when one encoded artifact configuration, on the frozen Q95 corpus, satisfies all of these on
the M2 Max single-worker 64 KiB-independent-block workload:

| metric | acceptance | stretch |
| --- | ---: | ---: |
| allocating CPU encode | >= 4.0 GB/s | — |
| safe CPU decode | >= 10.0 GB/s and >= 1.20x matched LZ4 | >= 15.0 GB/s and >= 1.50x LZ4 |
| `encoded_bytes / raw_bytes` | <= 0.050000 for **every** Q95 member | — |

GB/s is decimal uncompressed bytes per second. “Safe decode” means the public one-shot route:
frame parse, directory/command validation and checked output reconstruction are all accounted
for. A cached `ValidatedFrame` replay may be profiled separately, but cannot replace this row.
The target uses checksums off, warm buffers, one worker, no persistent thread pool and the same
block size/reference framing policy as `docs/BENCHMARKS.md`. A faster scalar microkernel, a
metadata-only model or a JSON-only result is never an achievement claim.

## Facts already established

- Current v2 JSON-16 has ratio 0.146501, allocating encode 2.5253 GB/s and safe full-frame
  decode 3.3105 GB/s; matched LZ4 safe decode is 8.9195 GB/s. The immediate gaps on that member
  are 1.58x encode and 3.02x safe decode to the minimum targets. Other Q95 families are slower
  and much larger, so JSON is not a sufficient gate.
- Exact unchanged-command P3 packing reduced JSON to about 0.120, but bitslice consumed exactly
  the same bytes as fixed-width under the same field plan. Int4 and ternary were worse across
  Q95. Bitslice remains a possible execution layout, not a ratio argument.
- Fully charged P5 entropy is 0.113657 / 0.113993 on JSON-16 / JSON-64. Adaptive fixed-or-
  entropy selection protects every held-out corpus but yields only fixed plus one selector byte
  there. It is scalar research code and is not a viable 4 GB/s wire candidate.
- The current literal streams alone consume 11.0%/10.9% of bursty input and 11.7%/11.2% of
  event input at 16/64 MiB. Therefore metadata packing by itself cannot reach 5%; any plan that
  starts with another metadata nibble layout is mathematically insufficient on those members.
- P5 byte-Huffman literal coding was measured exactly, including selector/tables/padding. It
  halves held-out literal payloads but leaves bursty at 5.9%/5.8% and event at 6.4%/6.1% *for
  literal bytes alone*. Its full literal-only ratios are 0.269001/0.275782 bursty and
  0.214096/0.240877 event. It is therefore rejected before v3 integration.
- A fixed eight-byte token dictionary only helped JSON (2,658 B / 9,102 B beyond Huffman) and
  was selected by no held-out LZ block. A strictly bounded second hash candidate did reduce
  held-out command metadata, but lost JSON compact size and directionally slowed encode by
  about 20–24%. Neither remains a wire or default-matcher candidate.
- A fully framed nested VXG literal control improved only JSON (to 0.143988 / 0.144146) and was
  never selected on any held-out literal block. It rules out a simple second literal-LZ wrapper
  as the next architecture; it is not retained as a production subcodec.
- A complete, independently decodable raw-domain structural-byte grammar was also rejected. Its
  quoted-string lexer, punctuation tags and charged block-local span tables only selected grammar
  on 5.5–13.9% of blocks and produced 0.982668/0.978032 bursty, 0.967214/0.958055 event,
  0.927871/0.927176 field-order and 0.922730/0.905680 JSON at 16/64 MiB. It proves that tagging
  syntax around otherwise raw bytes is not a joint compression grammar; the prototype is removed.
- A stronger exact record-template control did make the intended joint representation: a
  canonical block-local skeleton table, a row stream, and transposed value columns competing
  RAW, categorical dictionary, delta-RLE, real bit-sliced deltas, decimal-prefix deltas,
  lowercase-hex packing and bounded periodic-hex packing. Its complete R4 ratios were
  0.237793/0.236834 bursty, 0.267403/0.266130 event, 0.074979/0.074780 field-order and
  0.041067/0.040884 JSON at 16/64 MiB. It is exact negative evidence for the all-Q95 target:
  JSON alone passes, but no unqualified member gets close enough to promote the scalar oracle.
  Its construction/reconstruction rates (0.018–0.033 / 0.260–0.391 GB/s) are explicitly not
  codec rates and rule out a speed claim. The candidate is removed after retaining reports.
- P5D R9 completed the permitted schema-free structured-column search.  It serializes each
  independent block's template dictionary, row selectors, model descriptions, residuals and
  a RAW escape, then re-decodes the packet exactly.  At 64 MiB its fully charged ratios are
  0.083904 bursty, 0.078592 event, 0.050617 field-order and 0.023074 JSON.  The learned,
  prefix-trained GF(2) affine recurrence was selected for zero Q95 columns, so it did not
  supply a hidden identifier shortcut.  This is exact negative evidence: all-Q95 <=0.050000
  remains infeasible in the allowed family, and the scalar 0.034–0.072 GB/s pack / 0.050–0.084
  GB/s unpack rows are research rates, not codec rates.
- P6's checked opcode-7 decoder now inlines a direct `copy_nonoverlapping` only after its existing
  positive-distance, initialized-prefix, literal-range and output-range checks prove the run is
  non-overlapping. The overlap case remains on `Engine::repeat`. A JSON-16 directional pair was
  3.681 GB/s versus 3.371 GB/s with the old Engine repeat control; the all-Q95 one-process smoke
  is retained, but neither row is a target qualification. JSON's current safe row remains only
  3.764 GB/s against 9.158 GB/s LZ4-a1 in the longer local comparison.

## Non-negotiable methodology

Every candidate follows this sequence. Do not reverse it and do not promote a local speed win.

```text
exact independent-block prototype
  -> malformed/truncation/canonicality tests + actual-trace roundtrip
  -> all-Q95 artifact report with every byte charged
  -> integrated v3 correctness/model gates
  -> shuffled paired CPU qualification with confidence intervals
  -> promotion or removal
```

“Every byte” includes codebooks, dictionaries, selectors, padding, exceptions, frame/block
headers, literal payloads and any RAW fallback. “Every work item” includes table/dictionary
construction, parsing, validation, allocation and copies. Research rates are labelled as such;
only `vxg-bench` complete-codec rows can be quoted as GB/s.

No candidate may weaken independent blocks, accept malformed inputs, use unchecked public
decode, silently use a different matcher configuration, or select an algorithm after observing
the corpus without serializing and charging that choice.

## P5A — literal and command feasibility audit

The current evaluator now extracts each v2 LZ block’s literal slices only after the parser has
validated every command/literal boundary. The next dictionary audit extends that exact source and
must report:

- literal bytes, literal segments and their length distribution;
- byte/2-byte/4-byte histogram entropy estimates, repeated token coverage and top-token cost;
- command count per MiB, literal bytes per command and average copied run;
- a per-block lower-budget table: current literals, exact prototype literals, command/control,
  headers and total ratio;
- every report tied to the source input SHA-256 and current frame bytes.

This audit is not a codec. It has one purpose: determine whether literals, command count, or
both must change by enough to fit 5%. It must never discard a hard corpus because its estimate
looks poor.

### Required P5A gate

For bursty and event, report an explicit 5%-budget decomposition. If a literal candidate cannot
reduce its literal contribution below the remaining artifact budget **after** realistic command,
header and dictionary costs, reject it before touching the public wire format. An ideal
zero-command stream needs at least a 2.2–2.4x literal reduction in those families; real command
and header costs make the required factor higher.

The R4 record experiment also supplies a stricter event floor. Its 16 MiB event member contains
121,904 record-template rows; retaining one unmodelled 64-bit identifier per row costs 975,232
bytes, ratio 0.058128, before a tag, header, timestamp, category or directory byte. A
generator-specific procedural predictor is expressly disallowed. Any future generic candidate
must therefore demonstrate a serialized, independently validated cross-record model for that
identifier stream before it is allowed to claim an all-Q95 route.

The generic low-cost delta gate is now measured too. `P5-identifier-delta-analysis-r1` decodes
the hexadecimal ID text to its exact eight-byte values without applying any corpus parameter or
predictor. XOR-delta and wrapping-subtract delta planes are both near half one-bits and half
transitions at 16 and 64 MiB (for example 3,975,556/7,818,048 XOR one-bits and 3,956,926
transitions at 16 MiB). Therefore a bounded bitplane/RLE delta representation does not improve
on the already-inadequate raw-ID floor. Do not implement it as a v3 branch unchanged.

## P5B — literal candidates, exact and bounded

Implement these as separate bench-only lossless prototypes. They compete; neither is assumed to
be the v3 format.

1. **Bounded byte entropy — completed and rejected.** The raw-or-canonical-Huffman prototype
   charged table entries, code lengths, bit count, padding, selector and decoder validation.
   It cannot pass the literal budget on bursty/event and its scalar variable-bit execution is
   not a 15 GB/s decode path. Do not revisit it unchanged.
2. **Bounded token dictionary / structural literals — next.** Build a deterministic block-local table
   from literal substrings or JSON/log delimiters, serialize it before references and fall back
   to literal bytes. The encoder must have a fixed table budget, a bounded probe count and a
   deterministic tie rule. The decoder must reject duplicate/noncanonical tokens, out-of-range
   references, expansion overflow and trailing bytes. Do not call it “JSON compression”: it must
   act on raw bytes and be measured on all four families.

For both candidates, emit and decode the *complete* literal representation and reconstruct the
original literal sequence exactly. A length-only savings estimate is not enough. Record maximum
per-block scratch, build time and decode time. Reject a candidate if its complete all-Q95 ratio
does not beat the current complete fixed baseline materially, or if its worst member cannot
plausibly pass the 5% decomposition gate.

## P5C — command-count/matcher candidate

Literal coding alone still leaves too many 31–32 KiB-scale commands on Q95. Test one matcher
change at a time with a bounded work model:

- The first small set-associative trial is completed and rejected: an older bucket offset was
  inspected only after an 8–23-byte primary run and needed an eight-byte gain to win. It improved
  held-out compact metadata but required 186,090 extra JSON LCPs for 161 selections and lost the
  joint config. Do not reopen this threshold/table unchanged.
- Any successor must have a different pre-LCP discriminator with evidence that it avoids this
  failed-check rate. Probe count remains a constant, not a chain length.
- Use a cheap first-word/tag filter before LCP. On Neon, batch only the independent tag/load
  prefilter; preserve the exact scalar reference candidate selection until byte-equivalence and
  trace tests pass.
- Measure candidate checks, LCP bytes, accepted matches, literal bytes, commands and complete
  packed artifact bytes. A v2 byte reduction that expands the compact prototype is a rejection,
  as demonstrated by the recent-distance probes.
- Only test lazy parsing or alternate positions after the bounded hash candidate is measured.
  They have explicit CPU cost and must have a fixed lookahead/window bound.

The P5C gate is complete ratio plus encode work: it must materially reduce command/literal
budget on *all* Q95 members without an unbounded probe chain. It is not promoted merely because
JSON gets smaller.

## P4 — v3 integration only after a feasible exact prototype

Enter P4 only when one prototype serializes a complete independent block at <= 0.050000 on each
Q95 member with all bytes charged. Before writing production code, freeze the block grammar:

- explicit version and block kind; fixed maximum decoded block size;
- strict length/count bounds before allocation; canonical field/table/token order;
- a bounded RAW fallback whose selection is serialized and charged;
- no cross-block dictionary/history; all data needed by a random-access block lives in that block;
- golden vectors for each block kind plus every malformed boundary found by prototype tests.

Integrate encoder and decoder behind an explicit v3 format path. Preserve v1/v2 readers and
byte-exact v2 behavior. The safe parser validates every v3 command/table/token before execution;
any proof cache remains an optimization after the one-shot-safe benchmark has been recorded.

## P4B — bitslice, int4 and ternary placement

P3 already answers the ratio question for the old fixed field plan: a bitsliced transpose has
the same bit count as fixed width, while int4/ternary lose. Do not re-run those unchanged forms.

For a feasible v3 command stream, bitslice is evaluated as a **decode execution** option:

- group commands in a fixed lane-friendly size (start with 128/256), align plane storage and
  serialize the same selector/width metadata used by the scalar decoder;
- implement a scalar reference unpack first, then a Neon transpose/unpack for only the hot,
  fixed-width fields; compare every recovered command before timing;
- retain scalar tails and a scalar fallback for short/irregular groups;
- charge selector and plane metadata identically in scalar and SIMD rows. SIMD may improve GB/s
  but may not create an invented ratio win.

Revisit int4 only if the new field distribution has a measured high concentration in 0..14 and
an exact escape plan beats fixed on complete artifacts. Revisit ternary only if the final command
alphabet and decoder dispatch make its class stream genuinely cheaper. Both require the same
all-Q95 exact roundtrip gate; they are not default ideas.

## P6 — CPU throughput work after ratio feasibility

Work from profiles, not intuition. Record wall time, cycles/byte where available, branch misses,
cache misses, allocation count/bytes and a flamegraph/perf sample for JSON and the slowest Q95
member. Maintain separate encode, safe one-shot decode, parse/validate and validated-replay
labels.

Encode path, ordered by expected return:

1. Reuse bounded worker scratch and output capacity; avoid per-block heap churn. Arena/slab
   storage is acceptable only as worker-local scratch with bounded lifetime. A global allocator
   swap (`mimalloc`/`jemalloc`) is a measured experiment, not a design substitute.
2. Make matcher work bounded and contiguous: single-pass hash/tag loads, predictable stores,
   early rejection and no allocation in the hot loop.
3. Vectorize only demonstrated hotspots (first-word/tag prefilter, LCP and bulk literal copy).
   Keep scalar exactness tests for all unaligned/tail cases.
4. Re-measure allocating encode; do not quote a preallocated-only kernel as the 4 GB/s row.

Decode path, ordered by safety-preserving return:

1. Parse/validate block metadata once per public call with bounded allocation and concise
   validation records; profile why the current 8+ GB/s validation path becomes ~3.3 GB/s when
   output reconstruction is included.
2. Specialize the validated, non-overlapping literal/match copy path; retain checked arithmetic
   and overlap-safe periodic copies. Use Neon bulk copies/fills only after byte-exact scalar
   comparison across lengths, overlaps and non-power-of-two phases.
3. Add bitsliced command unpack only if its full safe decoder reduces cycles per command. Do not
   move validation outside the one-shot timing boundary.
4. Aim first for 10 GB/s and the LZ4 multiplier. The 15 GB/s target is stretch work after the
   safe row, not a justification to benchmark an unchecked replay loop.

### P6A — command-execution budget (required before another v2 micro-optimization)

The P6 local JSON-16 comparison is sufficient to reject more copy-loop intuition as a work
queue.  It is directional evidence, not a qualification result, but its *necessary* budget is
unambiguous:

| JSON-16 P2/P6 fact | value | implication |
| --- | ---: | --- |
| logical commands / average output per command | 591,146 / 28.380833 B | the executor performs roughly one small operation per 28 B |
| safe prepared decode / validated replay | 3.7643 / 3.9696 GB/s | removing repeated command checks buys only 5.5%; output execution dominates |
| prepared validation when measured alone | 8.6031 GB/s | it is an isolation measurement, not additive with the fused safe route |
| safe / replay command rate | 132.635 / 139.869 M commands/s | current command budget is about 7.54 / 7.15 ns per logical command |
| 10.0 GB/s / 1.20x LZ4-a1 (10.9896 GB/s) / 15.0 GB/s required rate on this trace | 352.350 / 387.205 / 528.526 M commands/s | at the unchanged average run these allow only 2.84 / 2.58 / 1.89 ns per command |

The inputs and rounded arithmetic for this table are retained in
`reports/qualification/experiments/P6-command-budget-r1.json`.  It must never be read as a
claim that a grouped executor cannot improve: it is a proof that a same-trace candidate has to
remove a large *per-command* cost, not merely save a branch or choose another `memcpy` spelling.

Consequently, do not add another v2 `repeat`, literal-copy, allocator, or dispatch tweak without
one of the following two prerequisites:

1. a successful profiler run identifying a single operation that accounts for enough wall time;
   or
2. a command-format experiment whose prevalidated executor exceeds 352.350 M commands/s on
   JSON-16 **and** whose one-shot safe path is timed separately.  The replay threshold is only a
   necessary condition for 10 GB/s; it does not authorize presenting replay as safe decode.

`mimalloc`, `jemalloc`, arenas and slabs do not meet this prerequisite for decode: the target
row reuses its output buffer and does not allocate in the hot executor.  They remain an encode
experiment only after allocation counters show material allocator time.

### P6B — bitsliced grouped-command executor (bench-only decision experiment)

Bitslicing is retained, but with a narrow job.  P3 already proved that transposing the current
fixed fields is byte-neutral and int4/ternary are worse, so this work neither claims a ratio win
nor changes v2.  Its question is whether a regular group eliminates enough parser work to clear
the P6A command budget.

1. Start from a byte-exact P2 logical trace and a fixed 64-command group.  Every group carries a
   count, exact field-width/escape metadata, control planes, literal/run planes and distance
   planes; short tails and every irregular opcode stay on the scalar reference path.  No command
   is silently normalized.
2. Write the scalar group unpacker first.  It validates group length, plane count, width, escape
   order, literal range, cumulative output range and backreference prefix before executing any
   group output.  It must reject a truncated plane, nonzero tail bits, noncanonical escape,
   zero/forward distance and output overflow.  Its decoded command sequence must equal the P2
   trace byte-for-byte.
3. Add a Neon transpose only for fixed-width hot planes after the scalar oracle and malformed
   tests pass.  It may fill stack-resident `[u8; 64]`/`[u16; 64]` staging arrays, but may not
   allocate a per-block command vector or move validation out of the public call.
4. Measure three distinct rows: group unpack/validation only, validated grouped execution, and
   parse + validate + decode.  Report commands/s, output GB/s, group/tail/escape rates, stack
   scratch and every byte of the synthetic representation.  The first two rows isolate work;
   only the third is a safe-decode result.
5. Stop this experiment if the prevalidated executor cannot exceed the P6A necessary threshold
   on JSON-16, or if its all-Q95 command rate/regression rules out a safe 10 GB/s row.  Do not
   promote a parser-only improvement with the old small-copy executor.

There is no NEON gather/scatter escape hatch for arbitrary backreferences.  P2 nevertheless
shows that only 5,182 of JSON-16's 591,087 matches reference within their eight-command group,
so the experiment may test regular metadata unpack and bounded unrolling.  It must still execute
output in causal order and retain overlap-safe copies.  This is a hypothesis to measure, not a
promise of vectorized backreference execution.

#### P6B R1 result and stop rule reached

`vxg-group-eval` now serializes the exact v2 logical trace into canonical 64-command bit planes
with group count/field bases/widths and literal payloads, validates all command and causal-output
relations, and separately replays only a validated immutable packet. Its mixed-command and
malformed/noncanonical packet tests pass; the evaluator also rejects a real-trace packet unless
its safe reconstruction equals the original input exactly.

The retained JSON-16 R1 report is
`reports/qualification/experiments/P6B-grouped-command-r1/json-records-16MiB.json`. At five
iterations it measures 44.825M validation, 37.971M validated-replay and 20.446M safe commands/s
(1.078 / 0.580 GB/s replay/safe input rate). The synthetic packet ratio is 0.112360299, but it
is not a VXG wire format and its bytes cannot be combined with P5D or claimed as a ratio result.

The required unchanged-trace replay budget is 352.350M commands/s. R1 reaches only 10.8% of it,
so the P6B scalar gate fails by a factor of 9.28x. Do not add a Neon transpose to this packet;
vectorizing plane unpack cannot make up that gap while causal small-copy output remains. P6B is
closed as a negative decode experiment rather than a candidate safe-decode route.

### P6P — profiled bounded opcode-7 copy (retained v2 improvement, not a joint pass)

A new macOS `sample` profile of the ordinary JSON-16 safe decoder put 1,972 of 3,351 sampled
stacks (58.85%) in `platform_memmove` on the existing variable-size non-overlap opcode-7 path.
That is a concrete P6 prerequisite, so one bounded candidate was tested: after the existing
checked proof establishes non-overlap, only 8..=63-byte runs use inlined unaligned u64 loads and
stores; all overlap, longer-run and irregular cases retain their existing paths. The same helper
is used by the proof-backed replay only after its immutable validation proof establishes the
identical range invariant. It has exhaustive 8..=63 length/alignment coverage.

`P6P-bounded-op7-copy-r1` retains both ordered JSON-16 pairs. Safe prepared decode is
7.5410/7.5404 GB/s candidate versus 3.7413/3.7465 GB/s generic control, with the byte-identical
0.14650089 artifact in every row. The bounded replay follow-up is 7.4960 GB/s versus 4.0114
before its matching helper; safe prepared decode is 7.5367 GB/s and validation alone 8.3042
GB/s. The manifest-verified Q95 smoke roundtrips every member; its safe range is 2.6369–7.5963
GB/s versus LZ4-a1's 4.7671–9.3386 GB/s, and ratios remain 0.14650089–0.32611792.

Keep this v2 execution change, but stop this micro-path here. It does not meet 10 GB/s, 1.20x
LZ4, or any all-Q95 ratio gate. The remaining work cannot be represented as a new `memcpy`
spelling: it requires an eligible artifact/model decision that reduces the causal command work
while satisfying the same artifact contract.

### P5D — generic structured-column feasibility unlock (before v3 implementation)

The R4 record oracle establishes that block-local skeletons and ordinary categorical/decimal/hex
columns are insufficient.  A new prototype is permitted only as a **schema-free, serialized
model family**, not as a rule keyed to `request_id`, a corpus filename, a field name, a seed, or
the qualification generator.  It must be selected by exact minimum description length per
eligible value column and must compete with RAW for every block.

The allowed initial family is deliberately general:

- fixed-width decimal and hexadecimal lexical values, canonical ISO-8601 timestamps, and
  categorical strings are parsed only when their original bytes can be reconstructed exactly;
  the original lexical style/exception stream is charged;
- bounded affine integer deltas, fixed-period runs and a learned GF(2) affine recurrence for
  fixed-width binary columns are candidate *models*, not hard-coded generators.  A recurrence
  stores its complete transition/initial state, training length and exact residuals in the same
  independent block, and is accepted only when a prefix-trained model predicts the remaining
  values exactly or the serialized residuals pay for every miss;
- model selection scans every eligible column with the same bounded work and tie rule.  A model
  that wins only because a field has a particular name, literal prefix, corpus path or known
  shift constant is invalid and must be removed.

Before writing a v3 encoder, the prototype needs two gates.  First, reconstruct all Q95 bytes,
including partial terminal records, from the serialized block alone and produce a per-column
byte decomposition.  Second, pass source-agnostic robustness checks: field-name permutation,
different seeds/row starts and non-linear/random control columns must select RAW or pay their
own residuals.  These controls cannot replace Q95 qualification, but they prevent a general
model class from becoming a disguised corpus-generator template.

This is the only current ratio route worth investigating.  The event identifier raw floor is
already 5.8% before other bytes, while R4 also leaves timestamp/value-column cost.  A generic
column model must demonstrate a charged path for both; bitsliced recurrence state or byte-plane
execution may later be useful for decode, but it cannot be used to invent an uncharged ratio
win.  If P5D cannot produce a complete <=0.050000 artifact for every member, stop v3 work and
renegotiate the locked target/workload rather than adding more packing variants.

#### P5D R9 result and stop rule reached

The implemented `vxg-column-eval` meets the prototype's correctness scope: it models only
exactly reconstructable lexical columns, serializes all block-local choices, decodes every
packet back to the original bytes, rejects malformed/non-canonical packets, and charges a
32-byte outer header plus 32 bytes per independent block directory entry.  It contains RAW,
dictionary, decimal affine/bit-pack, hexadecimal, timestamp, periodic string/hex and a learned
65-bit GF(2) affine recurrence model.  None relies on a field name, input path, seed or
generator constant.

R9 retains eight valid reports in
`reports/qualification/experiments/P5D-structured-columns-r9`; R2 is intentionally retained
as a failed report-serialization attempt and is not evidence.  The R9 64 MiB ratios are 8.3904%
bursty, 7.8592% event, 5.0617% field-order and 2.3074% JSON.  The 16 MiB companions have the
same failure shape.  In particular, GF(2) selected zero columns on all eight Q95 members.

Therefore P5D has **not** cleared the P4 feasibility gate and must not be integrated as VXG v3.
No more packing-layout variations from this family are authorized as a route to the joint
4/95/15 claim.  Continuing requires an explicit target/workload renegotiation or authority for
a materially broader, still source-agnostic model class with a new admissibility rule.  P6B may
continue only as independent decode research; it cannot make the same artifact meet the ratio
gate.

### P6C — encode and allocator discipline (only after P5D feasibility)

The 4 GB/s allocating encode target is a separate 1.52x gap from the current local JSON-16
2.6245 GB/s row.  Benchmark one allocator configuration at a time with identical frame bytes,
worker count, block size and corpus hash:

1. capture allocation call count, bytes and peak live bytes with `alloc-stats`; identify whether
   output growth, worker scratch growth or a per-block temporary is actually material;
2. only then compare system, mimalloc and jemalloc, and retain none of them merely because a
   short run is favorable;
3. prefer a bounded worker-local arena/slab for demonstrated temporary churn, reserve output
   capacity from a checked size bound, and reset at the worker/block lifetime boundary;
4. re-run the allocating public API, not a preallocated kernel.  A changed allocator, scratch
   policy or arena may not change the encoded bytes or hide allocation outside the timed call.

An arena is therefore a measured ownership optimization, not a blanket replacement allocator;
it has no expected decode benefit on the stated preallocated safe-decode row.

#### P6E R1 result — encode profile and arena/SIMD controls rejected

`reports/qualification/experiments/P6E-encode-profile-r1/evidence.json` retains a directional
JSON-16 allocating-encode profile before adding an allocator dependency. The allocation counter
records only 18 allocations and 12 reallocations; macOS `sample` records 21 of 3,356 stacks in
`RawVec` reserve/reallocation but 1,261 in `Engine::lcp` and 418 in command construction. Thus
neither mimalloc nor jemalloc has a profile-supported expected payoff on this path.

Four byte-identical candidate/control pairs were tested and removed: two NEON LCP mismatch
locators (2.0119/2.5708 and 1.6965/2.5965 GB/s candidate/control), exact all-worker payload-arena
reservation (2.5449/2.5796), and dirty-slot hash clearing (2.4977/2.5723). The full-table zero is
cheap on this machine; the per-probe bookkeeping costs more. Keep the current worker-local arena
and system allocator. Any later encode candidate must first show a discriminator or model that
avoids enough LCP/command work to pay for its own added logic; do not retry general allocation or
copy/SIMD spellings.

## P7 — qualification, promotion and stop rules

For a candidate that passes correctness and complete-ratio gates:

1. Run the frozen corpus SHA check, then at least seven shuffled paired processes per member,
   each with >=0.2 s samples; preserve every process, order and sample.
2. Compare VXG with LZ4 accelerations 1/4 and Zstd -3/1/3 under the documented independent-block
   profile. Report ratio and speed side by side; do not suppress a better Zstd ratio.
3. Require every Q95 member to meet the same stated target. Publish medians and confidence
   intervals, hardware, OS, compiler, library versions, SIMD selection, allocator and thermal
   conditions.
4. Run default + bench tests, `scripts/check.sh`, model/sanitizer gates and malformed/truncation
   fuzzing for every new public parser/executor path. Run actual Metal validation separately;
   GPU output never substitutes for the CPU target.

Stop and remove an experiment when it misses any hard gate: no complete all-Q95 ratio path,
unbounded matcher work, noncanonical/unsafe grammar, or slower safe decode with no ratio payoff.
If the strongest exact literal-plus-command prototype still cannot fit 5% on the frozen corpus,
report that feasibility result and renegotiate the artifact target or workload. Do not manufacture
a win by moving to a different corpus, cross-block history, unchecked decode or a separate
configuration.
