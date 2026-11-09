# Event/bursty ratio frontier review — fresh measurement, blocking evidence, decision request

Review date: 2026-09-26. Scope: read-only examination of the current tree plus one fresh
JSON-16 benchmark run; no codec changes were made. This document records new negative
evidence for two bounded model families and quantifies why the `event` and `bursty`
members cannot close under the currently frozen rules. It is not a speed claim.

## 1. Fresh integrated measurement

`vxg-bench`, one worker, 64 KiB blocks, Neon, checksums off, system allocator, warm buffers,
5 x 0.2 s samples, `corpus/qualification/json-records-16MiB.ndjson`:

| codec | ratio | allocating encode GB/s | safe full-frame decode GB/s | prepared decode GB/s |
| --- | ---: | ---: | ---: | ---: |
| VXG v2 | 0.147 | 2.654 | 7.749 | 7.700 |
| LZ4 a1 | 0.150 | 2.416 | 9.329 | 9.329 |
| LZ4 a4 | 0.151 | 2.413 | 9.357 | 9.357 |
| Zstd -1 | 0.054 | 1.662 | 3.326 | 3.326 |

Command validation alone measured 8.610 GB/s. Gaps to the current contract
(`TARGET_4_006_12_15.md`: >=4 / <=0.06 / >=12 with 15 stretch):

- encode x1.51 to 4.0, x2.26 to 6.0;
- safe full-frame decode x1.55 to 12.0, x1.94 to 15.0; 1.20x LZ4-a1 is ~11.19 GB/s,
  1.50x is ~13.99 GB/s;
- ratio requires 2.44x fewer encoded bytes for 0.06 and 2.93x for 0.05.

These are directional single-process samples, not a qualification result.

## 2. Per-member ratio frontier (P5D R9 scalar research packets)

| corpus | ratio @16 MiB | ratio @64 MiB | 64 MiB deficit vs 0.06 | vs 0.05 |
| --- | ---: | ---: | ---: | ---: |
| JSON records | 0.023172 | 0.023074 | pass | pass |
| field-order / optional / escaping | 0.050788 | 0.050617 | pass | 41,435 B |
| event / identifiers | 0.078921 | 0.078592 | 1,247,676 B | 1,918,765 B |
| bursty / variable records | 0.084347 | 0.083904 | 1,604,178 B | 2,275,267 B |

The packets are scalar-only research artifacts; their construction and reconstruction rates
remain far below every speed gate and no production configuration exists.

## 3. Why event and bursty are blocked under the frozen rules

Dominant R9 payloads on the 64 MiB members:

- bursty: `quoted-lower-hex-periodic-1-to-64` is 4,622,844 B of the 5,630,710 B frame.
  Its seed-only cost is 4,008,130 B, 99.5% of the whole 4,026,531 B budget
  (`P8-hex-period-audit-r2/decision.json`).
- event: `quoted-fixed-width-lower-hex-suffix` (the `r-<16-hex>` identifiers) is
  3,888,784 B, 96.6% of the budget; timestamp + dictionary + affine + directory add
  about 1.10 MB more.

Both streams are per-record outputs of `xorshift64(seed ^ (index * K))` (corpus generator;
analysis only, no field-name or seed rule is used by any measured model). Recovering that
relation itself is exactly the predictor the acceptance plans declare invalid
(`VXG_4_95_15_PLAN.md`, `TARGET_4_95_15_PLAN.md`). The plans' own stop rule - "If P5D cannot
produce a complete <=0.050000 artifact for every member, stop v3 work and renegotiate the
locked target/workload rather than adding more packing variants" - is therefore reached; the
2026-09-21 revision (0.06) keeps two members passing and leaves the same two blocked.

## 4. New negative evidence recorded today

Two bounded, source-agnostic model families that could have cracked the blocking streams were
tested directly on the frozen bytes (blocks 0 and 5 of both 64 MiB members). The GF(2) solver
used for the test is brute-force-verified on 1,000 generated linear systems with zero false
acceptances; verification is exact over the whole block.

1. Counter-affine model `value = C ^ XOR_i a_i * bit_i(index)`: inconsistent on the fit half
   for both families in both blocks, raw digits and de-rotated nibbles alike. It is also
   mathematically rejectable: `index * K mod 2^64` is not GF(2)-linear in the index bits,
   because integer multiplication carries.
2. GF(2) linear recurrences of order 1-4, `y_n = XOR_j A_j y_(n-j)`: inconsistent for both
   families in every tested order.

Combined with the retained P5 identifier-delta audit (XOR/subtract deltas at ~half density)
and the P5D R9 learned-recurrence result (zero columns selected), the identifier/message
streams have no demonstrated structure for the allowed model families. These are decisive
negative results for this review, not an information-theoretic impossibility proof.

## 5. Decision requested (explicit user call, per plan)

- A. Keep corpus and rules; accept the documented frontier: JSON and field-order pass
  0.06 (JSON also 0.05), event/bursty remain documented misses.
- B. Freeze a Q95 revision (r2) whose failing families no longer consist of per-record PRNG
   outputs, then pursue <=0.05-0.06 all-Q95 with the generic family. This is a scope change
   and must be recorded as one; it does not silently relax any threshold.
- C. Authorize a materially broader model class (for example fitted PRNG recovery). Not
   recommended: it is effectively the generator-specific predictor the plans forbid, and it
   would not transfer to real workloads with unrelated random identifiers.

## 6. Suggested sequencing after the decision

1. Speed axis, independent of the ratio decision: take a fresh profile of the current safe
   JSON decode and of allocating encode on the current tree (the last profile predates the
   P6P bounded-copy retention), then work the documented gates (P6A necessary command rate;
   P6E encode profile) before any new micro-experiment.
2. Artifact axis for the passing families: a feasible exact prototype already exists (R9).
   Per P4, v3 integration may start only after the member scope is fixed; budget for the
   scalar prototype's speed work is the dominant remaining risk.
3. Qualification per the frozen protocol (>=7 paired shuffled processes, per-member triple)
   once one integrated configuration exists.
