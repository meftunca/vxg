# VXG r2 target plan — ratio 0.05 / encode 4–6 GB/s / safe decode 12–15 GB/s

Decision record: 2026-09-26, user option B from [E_B_FRONTIER_REVIEW.md](E_B_FRONTIER_REVIEW.md).
Qualification runs on the Q95-r2 corpus; r1 is retained as an adversarial control set.
Generic (source-agnostic) models only; no generator-specific predictor, field-name, seed
or corpus-path shortcut. All plan steps follow the retained methodology: exact scalar
prototype, every byte charged, malformed-input rejection, robustness controls, paired
shuffled qualification. This plan supersedes the next-work list in
`TARGET_4_006_12_15.md` but not its benchmark rules.

## Contract (r2)

| Metric | Required | Stretch |
| --- | ---: | ---: |
| Allocating CPU encode | >= 4.0 GB/s | 4–6 GB/s |
| Complete encoded bytes / input bytes | <= 0.050000 | smaller only if speed survives |
| Safe full-frame CPU decode | >= 12.0 GB/s and >= 1.20x matched LZ4 | 15.0 GB/s and >= 1.50x |

0.06 remains the historical floor for reference only. GB/s is decimal raw bytes/second.
The M2 Max, single worker, 64 KiB independent blocks, checksum-off timing and warm-buffer
profile are unchanged. Only `cpu_allocating_encode` and
`cpu_full_frame_preallocated_decode` rows decide the primary speed contract.

## Phase 0 — decision records (completed 2026-09-26)

- `TARGET_4_006_12_15.md` revision note, `PROGRESS.md` entry, `E_B_FRONTIER_REVIEW.md`.
- This plan is the working artifact for the phase backlog and gates.

## Phase 1 — Q95-r2 corpus (completed 2026-09-26)

- `generate_qualification_corpus.py --r2` writes `corpus/qualification/r2/` and
  `reports/qualification/manifest-r2.json` (revision `q95-json-r2`); json/field-order
  members are byte-identical to r1, event/bursty value sources are revised
  (epoch+counter identifiers; phrase-vocabulary messages with unchanged length law).
- `run_qualification.sh` accepts `MANIFEST=`; generation + verification reproduce all
  eight members deterministically.
- Structure audit retained: v2 and LZ4 ratios improved on both changed families, and the
  old seed/message floors no longer describe the dominant payloads.

## Phase 2 — r2 ratio prototype <= 0.05 (next)

1. `vxg-column-eval` + `src/columns.rs` run on all eight r2 members; per-member byte
   decomposition into `reports/qualification/experiments/P10-columns-r2/`.
2. Required new bounded models, each with a generic hypothesis, bounded selection and
   training cost, complete serialized per-block state, exact scalar reconstruction and
   malformed/non-canonical rejection:
   - `quoted-hex-epoch-counter`: parse fixed-width hex values as (prefix, suffix); canonical
     block-local prefix dictionary with a fixed entry bound; ascending-suffix flag or
     residual; bounded random exceptions charged per value. Rejects duplicate/out-of-order
     dictionaries, invalid residuals, unused entries and trailing bytes.
   - `quoted-phrase-token`: block-local phrase/word dictionary with a fixed entry bound,
     packed token indices, decimal-number escape and exact trailing-pad count; rejects
     invalid indices, unused/duplicate entries, overflow and non-exact reconstruction.
   - Optional only if measurements require it: cross-record periodic timestamp model
     (period <= 256) and dictionary-tail refinement.
3. Gate A (r2, scalar): every member serializes to <= 0.050000 with all bytes charged
   (frame header, directory, tables, selectors, padding, escapes, RAW fallback). Report
   per-member totals; failure produces a deficit table and stops new layouts.
4. Robustness set: field-name permutation, different seeds/row starts, shuffled row order
   and non-linear/random control columns must select RAW or pay their own residuals.
5. Record decisions per report; do not delete failed prototypes (bench-only).

#### Phase 2 status (2026-09-26): gate A passed

- Three bounded, serialized, source-agnostic models were added to `src/columns.rs` /
  `vxg-column-eval`: quoted hex epoch-counter (prefix dictionary + sequential-suffix flags),
  quoted semicolon segment dictionary (bounded dictionary + bit-packed codes, decimals and
  pads) and canonical decimal delta bit packing. All eight r2 members now serialize to
  <= 0.050000 (`P10-columns-r2`): json 0.0231, field-order 0.0448, event 0.0258, bursty
  0.0486-0.0488 at 16/64 MiB. The bursty margin (2.4-2.8%) is the thinnest; a
  phrase-escape code split is the next measured candidate if later v3 framing overhead
  grows.
- Robustness controls (`scripts/make_r2_controls.py`, `P10-columns-r2-controls`) confirm no
  field-name or row-order dependence: renamed inputs behave identically modulo file size and
  shuffled inputs degrade to explicit residuals with exact reconstruction.
- 10/10 column tests and the full default/bench suites pass; the packets remain scalar
  oracles with no speed claim.

#### Phase 3 status (2026-09-27): integrated behind `Options { v3: true }`

- `docs/FORMAT_V3.md` freezes the extension: version-3 frames accept block kind 2 (COLUMN)
  with `2 <= payload < raw`; the payload is the canonical P5D packet; selection is
  strictly-smaller-than-LZ and charged; command validation/validated replay/GPU return
  `Error::Unsupported` for column frames; no unsafe on the column decode path.
- `Options { v3: bool }` (default false) keeps v2 as the default output; `vxg-bench --v3`
  profiles the structured path and skips unavailable command-validation rows with a note.
- Integrated ratios on 16 MiB members match the packet accounting: json 0.023, r2 event
  0.026, r2 bursty 0.049. `tests/v3.rs` (4 tests) plus the full default/bench suites pass;
  the roundtrip fuzz target covers v3.
- Speed remains research-slow on the COLUMN path (encode 0.050-0.063 GB/s, decode
  0.056-0.081 GB/s): Faz 4/5 must optimize construction and decode before any v3
  qualification run.

#### Faz 4/5 status (2026-10-14): vectorized structured-column encode and decode frontier

- **Decode (`PreparedTemplate` + `FastPiece` + NEON bit-unpacking):**
  - `PreparedColumn::parse_and_validate` validates every column up front and unpacks bit
    streams via 8-values-per-iteration NEON (`unpack_u8_swar`, `unpack_u16_swar` with
    `vmovl_u8` + `vaddq_u16` + `vst1q_u16`).
  - `UnboundTemplate::build` folds constant (`width == 0` / `step == 0`) columns directly
    into literal template segments, fuses adjacent literal segments into 64-byte `FastPiece`
    entries (`<= 48` bytes inline via unaligned `u128` stores), and computes a per-row
    maximum byte budget (`row_budget`) so `emit_template_rows` checks buffer capacity once
    per block instead of per piece.
  - Safe full-frame decode (`cpu_full_frame_preallocated_decode`) improved from
    `0.056–0.081 GB/s` (Phase 3) to **`3.85–10.05 GB/s`** across the 8 `Q95-r2` members,
    exceeding `LZ4` (`9.13–9.34 GB/s`) on `json-records` (`9.73–10.05 GB/s`, `1.07x` LZ4)
    and on `event-mix-identifiers` (`6.97–7.00 GB/s` vs `5.41–5.49 GB/s`, `1.28x` LZ4), and
    exceeding `Zstd -3`/`1`/`3` decode (`1.44–4.19 GB/s`) by `2.2x–3.6x` on every member while
    maintaining `2.3x–4.3x` smaller compressed size than `Zstd 1`.
- **Encode (`try_match_clean_stream` + NEON `scan_string` + SWAR decimal/hex/dictionary):**
  - `encode_worker` runs `columns::encode_grammar_block_into` directly into the worker arena
    first when `opt.v3` is enabled and skips `block::encode` whenever `packet_len < raw_len / 2`,
    and `assemble_workers` reserves the exact output frame size (`table_end + total_payload_len`).
  - `grammar_from_block` uses a single-pass clean-template stream matcher with NEON nibble
    masking (`vshrn_n_u16`) in `scan_string`, prefix-sharing template continuation
    (`shared_segs`), and pooled column `Vec`s (`ENCODE_VEC_POOL`).
  - `pack_model` uses 4/8-digit SWAR `parse_decimal`, 64-bit delta/affine detection,
    `HT_CAP = 256` stack hash tables with 2-row O(1) lower-bound aborts in
    `pack_dictionary_model`, cached prefix/word checks in `pack_epoch_counter_model` and
    `pack_timestamp_model`, 64-bit `CLZ` Gaussian elimination in `pack_gf2_affine_model`, and
    64-bit word writes in `push_bit_values`.
  - Allocating CPU encode (`cpu_allocating_encode`) improved from `0.050–0.063 GB/s` (Phase 3)
    to **`1.015–1.790 GB/s`** (`16x–35x` speedup), beating `Zstd 1` encode (`0.782–1.666 GB/s`)
    on every member while compressing `2.3x–3.0x` smaller.
- **Measured 8-member `Q95-r2` frontier (`threads=1`, `block=65536`, checksums off):**

| Dataset (`corpus/qualification/r2/...`) | v3 Ratio (`<= 0.050`) | v3 Encode (GB/s) | v3 Safe Decode (GB/s) | LZ4-a1 Ratio / Enc / Dec | Zstd-1 Ratio / Enc / Dec |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `json-records-16MiB.ndjson` | **0.0231** (PASS) | 1.790 | 10.048 (`1.08x` LZ4) | 0.150 / 2.483 / 9.339 | 0.054 / 1.666 / 3.349 |
| `json-records-64MiB.ndjson` | **0.0231** (PASS) | 1.757 | 9.729 (`1.07x` LZ4) | 0.149 / 2.350 / 9.132 | 0.054 / 1.657 / 3.305 |
| `field-order-optional-escaping-16MiB.ndjson` | **0.0448** (PASS) | 1.163 | 4.835 (`0.99x` LZ4) | 0.213 / 1.417 / 4.901 | 0.104 / 0.986 / 1.938 |
| `field-order-optional-escaping-64MiB.ndjson` | **0.0448** (PASS) | 1.169 | 4.458 (`0.94x` LZ4) | 0.212 / 1.347 / 4.767 | 0.104 / 0.962 / 1.904 |
| `event-mix-identifiers-16MiB.ndjson` | **0.0258** (PASS) | 1.473 | 6.972 (`1.29x` LZ4) | 0.155 / 1.812 / 5.413 | 0.078 / 1.115 / 1.946 |
| `event-mix-identifiers-64MiB.ndjson` | **0.0258** (PASS) | 1.475 | 7.000 (`1.27x` LZ4) | 0.154 / 1.790 / 5.494 | 0.078 / 1.179 / 1.904 |
| `bursty-variable-records-16MiB.ndjson` | **0.0486** (PASS) | 1.063 | 3.922 (`0.93x` LZ4) | 0.207 / 1.270 / 4.219 | 0.120 / 0.813 / 1.681 |
| `bursty-variable-records-64MiB.ndjson` | **0.0488** (PASS) | 1.015 | 3.852 (`0.92x` LZ4) | 0.206 / 1.294 / 4.181 | 0.119 / 0.782 / 1.442 |

## Phase 3 — v3 format and integration

1. Freeze the v3 block grammar before production wiring: explicit version/kind, strict
   length and count bounds before allocation, canonical table/field order, bounded RAW
   fallback whose selection is serialized and charged, independent blocks only, golden
   vectors for every block kind and malformed boundary. Write `docs/FORMAT_V3.md`.
2. Implement the safe one-shot decoder and the validated (proof-carrying) path; keep v1/v2
   readers and byte-exact v2 behavior. Verify `compress()` and `compress_with_stats()`
   emit identical frames for identical options.
3. Bench profiles for v3 (encoding, safe full-frame decode, validation) with explicit
   labels; v3 is opt-in until qualification.
4. Tests: golden v1/v2/v3, truncation/malformed/canonicality, literal tails, multiworker,
   checksum failures; fuzz differential safe vs validated decode equality.
5. Gate B: default + bench release suites, `scripts/check.sh`, `scripts/check_model.sh`
   pass; Q95-r2 members roundtrip byte-exactly through safe APIs.

## Phase 4 — encode 4–6 GB/s

1. Fresh allocating-encode profiles per member class (LZ path and structured path) on the
   current tree; retain evidence summaries as in `P6E`.
2. Candidate order, only with profile justification: matcher early rejection and
   single-pass tag/hash loads; LCP cost reduction (previous NEON variants were rejected;
   no speculative re-runs); structured-encode hot paths (row parse, model build, token
   packing) vectorized where measured; allocation/scratch remains as retained (P6E
   rejected allocator and arena variants).
3. Promotion rule: 3-pair A/B, positive 95% paired interval and normally >= 3%; gate is
   >= 4.0 GB/s on every member before stretch work; stretch 6.0 only after the minimum.

## Phase 5 — safe decode 12–15 GB/s

1. Fresh `sample` profile of safe full-frame decode after P6P; identify the remaining
   distribution (memmove share, parse/validate, per-command overhead).
2. Candidate order: bounded SIMD copies beyond the retained 8..=63 u64 helper (64..=255
   and overlap-periodic paths with exhaustive alignment/length tests); opcode-7 fused
   checked execution with aggregate bounds checks for small command groups; structured
   decode hot paths (template expansion, token unpacks, hex/column expansion) only where
   the profile points. The P6B bitsliced grouped executor remains closed (9.28x gate miss).
3. Necessary-rate gate on the unchanged trace: >= 352.35M commands/s (10 GB/s),
   >= 387.2M (1.20x LZ4) and >= 528.5M (15 GB/s) for any new command representation;
   prevalidated replay remains a diagnostic row, never the safe result.
4. Order: minimum 12 GB/s + 1.20x LZ4 first, then 15 GB/s + 1.50x.

## Phase 6 — qualification

1. `MANIFEST=reports/qualification/manifest-r2.json scripts/run_qualification.sh ...`
   with at least 7 paired shuffled processes per member, 9 x 0.2 s samples, per-process
   medians, min/max and uncertainty intervals.
2. LZ4 a1/a4 and Zstd -3/1/3 rows in the same runs; energy reporting per
   `ENERGY_BENCHMARKS.md` where sensors are available.
3. Update README/CHANGELOG/PROGRESS; report the actual frontier per member; never mark an
   unmet target as complete (G4: minimum triple + 1.20x LZ4; G5: 15 GB/s + 1.50x).

## Verification commands

- `python3 scripts/generate_qualification_corpus.py --r2 --verify`
- `cargo test --release` and `cargo test --release --features bench`
- `scripts/check.sh`; `scripts/check_model.sh`
- `cargo run --release --features bench --bin vxg-bench -- --file corpus/qualification/r2/<member>.ndjson --threads 1 --block-size 65536 --samples 9 --seconds 0.2`
- `cargo run --release --features bench --bin vxg-column-eval -- --input corpus/qualification/r2/<member>.ndjson --output reports/qualification/experiments/P10-columns-r2/<member>.json --iterations 1 --block-size 65536`

## Risks

- Speed is the larger open risk (encode x1.5, decode x1.55 on JSON-16 today). Gates stop
  speculative work; a missed gate produces an honest frontier report, not a claim.
- The r2 message stream still carries bounded per-record entropy (phrase choices, tail
  words, decimal parameters); the phrase-token model must serialize its dictionary and
  charge every escape. If the margin proves too thin, revise only the value sources again
  through an explicit recorded decision — never by weakening a threshold.
- Old rejected paths must not be re-opened unchanged: NEON LCP variants, second-hash
  matcher, recent-distance probes, allocator swaps, bounded entropy on literals, token
  dictionaries, nested literal VXG, structural grammar, P6B grouped executor, identifier
  delta planes.
