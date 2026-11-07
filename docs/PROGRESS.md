# VXG development progress

This file records implementation steps, validation gates, and benchmark evidence.
Every implementation or measurement change is recorded here as a separate step.


Earlier development logs are archived in:
- [PROGRESS_ARCHIVE_V1_V2.md](PROGRESS_ARCHIVE_V1_V2.md) — v1 & v2 baseline and hot-path experiments (2026-09-15 to 2026-09-19)
- [PROGRESS_ARCHIVE_P0_P6.md](PROGRESS_ARCHIVE_P0_P6.md) — P0 to P6E qualification and packing research (2026-09-19 to 2026-09-21)

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


## 2026-10-26 — VXG v0.4.0 Format v4, Zero-Allocation Pipeline, and Modular Codebase (`<= 600` lines/file)

- **Format v4 & Multi-Format Block Engine (`src/columns/`, `src/frame/`):**
  - Default wire format upgraded to `VERSION_V4 = 4` with compact 8-byte per-block directory entries (`payload_size | (kind << 30)`, `crc32c`) while retaining full backward read support for `v1`–`v3`.
  - Extended `BLOCK_COLUMN` packets with `STRIDE_PACKET` (24-bpp/32-bpp BMP and strides `1, 2, 3, 4, 8` with `Subtract-Green` color decorrelation, 2-level 64-bit zero-masks, and per-plane `Fast-LZ`) and `FAST_LZ_PACKET` paired with a 4-stream interleaved canonical Huffman coder (`HUF_MAGIC`).
  - Extended structured grammar discovery to multi-line pretty-printed JSON arrays/objects, Apache/Nginx Combined Log Format (CLF), bracketed timestamp logs, and `MODEL_FRONT_DICT` 4-bit/5-bit nibble palettes + Fast-LZ dictionary streams.
- **Zero-Allocation Encode/Decode & SIMD/SWAR Hot Paths:**
  - Added public zero-allocation `compress_into`, `decompress_into`, and `extract_block_into` in both Rust (`src/lib.rs`) and C FFI (`src/ffi.rs`, `include/vxg.h`).
  - Reworked `Frame::decode_into` (`src/frame/decode.rs`) so both single-thread and multi-thread (`threads > 1`) paths decode directly into disjoint slices of the caller-provided output buffer with zero per-block `Vec` allocations.
  - Added `ENCODE_SCRATCH` thread-local reuse in `src/frame.rs` so single-thread and multi-thread `compress_into` reuse worker hash tables and payload arenas across calls.
  - Added 64-bit SWAR 8-row-at-a-time bit unpacking (`unpack_and_validate_u8`, `unpack_u8_swar`, `unpack_u16_swar`), `aarch64` NEON `vqtbl1q_u8` / `vqtbl2_u8` nibble/5-bit palette lookup, and fast single-template / multi-template `PreparedColumn` + `FastPiece` row assembly.
- **Codebase Modularization (`<= 600` lines per file):**
  - Split `src/columns.rs` (was 18,439 lines), `src/packing.rs` (was 3,135 lines), `src/bin/bench.rs` (was 1,269 lines), `src/frame.rs` (was 1,122 lines), `src/block.rs` (was 931 lines), and `src/groups.rs` (was 767 lines) into focused submodules so every `.rs` source file and `.md` documentation file in the repository is at most 600 lines.
- **Apple Silicon 1-Thread & 8-Thread Benchmark Summary (`16 MiB` per dataset, `64 KiB` blocks, checksums off):**
  - 10-Dataset Geometric Mean (`vxg-v4` vs `lz4-c-a1` vs `zstd-c-l1`):
    - **Compression Ratio:** `0.0250` (`40.0x`) vs `0.1272` (`7.86x` LZ4) vs `0.0607` (`16.5x` Zstd-1)
    - **1-Thread Throughput:** `2.094 GB/s` encode (`1.34x` Zstd-1) / `8.211 GB/s` decode (`1.18x` LZ4, `3.08x` Zstd-1)
    - **8-Thread Throughput:** `11.750 GB/s` encode (`1.34x` Zstd-1) / `41.545 GB/s` decode (`1.14x` LZ4, `2.59x` Zstd-1)
  - Full tables recorded in [`docs/BENCHMARKS.md`](BENCHMARKS.md).

## 2026-10-26 — VXG v0.4.1 Release: 10/10 Dataset `VPI` Dominance, `FAST_LZ` & Huffman ILP, Per-Process Energy & Composite Indices (`VPI`, `BCI`, `GCI`)

- **Per-Process Hardware Energy (`J/GB`) & Composite Indices (`src/bin/bench_support/metrics.rs`, `src/bin/energy_workload.rs`, `scripts/energy_bench.py`):**
  - Integrated unprivileged per-process kernel hardware energy (`proc_pid_rusage` `RUSAGE_INFO_V6` `ri_energy_nj` and retired instructions `ri_instructions`) directly into `vxg-bench` and `vxg-energy-workload`.
  - Added three composite indices across CSV, console, and Markdown reports:
    - **VXG Performance Index (`VPI`):** `sqrt(Encode_GBps * Decode_GBps) / Ratio`
    - **Balanced 3-Axis Cube Index (`BCI`):** `((Encode_GBps * Decode_GBps) / Ratio)^(1/3)`
    - **Green Codec Index (`GCI`):** `VPI / sqrt(Encode_J_per_GB * Decode_J_per_GB)`
- **`FAST_LZ` & 4-Stream Canonical Huffman ILP Optimizations (`src/columns/fast_lz_core.rs`, `src/columns/fast_lz_pack.rs`, `src/columns/fast_lz_unpack.rs`, `src/columns/huffman_enc.rs`, `src/columns/huffman_dec.rs`):**
  - Swapped Huffman decode table entry layout (`((sym as u16) << 8) | code_len`) to eliminate `>> 8` extraction latency from the critical shift path and overlapped the 4th-symbol bitstream reload (`step4!(3, reload)`) with the L1 table lookup.
  - Added 2-stream parallel Huffman bitstream encoding (`encode_pair_fast`) to pack stream pairs `(0, 1)` and `(2, 3)` in lockstep without register spilling.
  - Widened `tok` in `FAST_LZ` to 4-bit `lit_code` (`0..=15`) + 4-bit `ml_code` (`0..=15`) with raw zero-copy `offset_lo` and `ext` streams for `hash_len == 5`, branchless `is_ml15` fast decode (`csel`), and lazy `+1` 8-byte match checking.
  - On **`source-and-docs.txt`**: 1T encode jumped from `0.353 GB/s` to **`0.614 GB/s`** (beating Zstd-1's `0.570 GB/s`), 1T decode jumped to **`2.478 GB/s`** (`1.62x` faster than Zstd-1), 8T encode reached **`4.306 GB/s`** and 8T decode reached **`16.844 GB/s`**, achieving **10/10 dataset wins on `VPI`** against both LZ4 and Zstd-1 in both 1T (`1.01x`–`19.42x` vs LZ4, `5.27x` geomean) and 8T (`1.02x`–`19.56x` vs LZ4, `5.29x` geomean).
- **Dynamic Multi-Thread Work-Stealing (`src/lib.rs`, `src/frame/decode.rs`):**
  - Replaced static range partitioning with atomic `AtomicUsize` work-stealing across `encode_into`, `Frame::decode_into`, and `ValidatedFrame::decode_into`, and added early payload capacity reservation at block 1 in `encode_single_worker`.
- **2-Way ILP `FAST_LZ` Probing, Graceful Schema-Drift Fallback, Bounded-Memory Streaming & Hybrid GPU v4 Support (`src/columns/fast_lz_core.rs`, `src/columns/grammar_extract.rs`, `src/stream.rs`, `src/gpu/mod.rs`):**
  - Added 2-way ILP hash probing (`cand0` at `ip` and `cand1` at `ip + 1`) plus intra-match `ip + 2` hash insertion and `ml_code < 15` fast decode dispatch, improving `source-and-docs.txt` ratio from `0.2782` to **`0.2698`** (strictly beating Zstd-1's `0.2713` and `1.40x` smaller than LZ4's `0.3777`) while maintaining `~0.59 GB/s` 1T encode and `2.47 GB/s` 1T decode.
  - Updated `grammar_from_block_mode` (`allow_early_abort == true`) so blocks exceeding `MAX_TEMPLATES` (`64`) or records exceeding `MAX_SLOTS` (`32`) gracefully fall back to `Row::Raw` (`FAST_LZ`-compressed raw stream) instead of returning `Err`.
  - Added `src/stream.rs` (`StreamEncoder<W: Write>`, `StreamDecoder<R: Read>`, `compress_stream`, `decompress_stream`), concatenated multi-frame `VXG1` support in `vxg::decompress` / `vxg::decompress_into`, `vxg_frame_compressed_size` in C FFI (`src/ffi.rs`, `include/vxg.h`), and `--stream` bounded-memory mode in `src/bin/vxg.rs`.
  - Upgraded `GpuCodec::compress` and `GpuCodec::decompress` (`src/gpu/mod.rs`, `src/frame/assemble.rs`) to emit and decode Format v4 frames including hybrid `BLOCK_COLUMN` + `BLOCK_LZ` / `BLOCK_RAW` blocks.

## 2026-11-02 — VXG v0.4.2 Release: Wave 1–3 Codec Speedups & Compressed-Domain Query, Analytics, Mutation & Zero-Reencode Projection

- **Core Codec Models & Wave 1–3 Throughput (`src/columns/`, `src/frame/`, `src/lib.rs`):**
  - Added `MODEL_PATTERN` (`tag 15`) with SWAR digit/literal shape verification and `Copy16` constant-affix decode (`2.5x–7x` smaller on CLF/non-ISO timestamps/IPv4), relaxed CSV/TSV free-text matching, single-pass CLF stream matcher (`try_match_clf_stream`), and unquoted `MODEL_HEX_SUFFIX`.
  - Delivered `x86_64` SSE2/SSSE3/SSE4.1/AVX2 parity across grammar, stride, palette, and timestamp kernels, plus Wave 1–3 model-gate pruning, short-raw decode fusion, zero-realloc model pools, and lock-free frame assembly (`ndjson_varied.ndjson` `56.5 MB`: `5,167,201 B` / `9.14%` ratio, `4.08 GB/s` multi-thread encode).
  - Fixed five edge-case round-trip/memory-safety bugs uncovered by randomized multi-seed stress (`tests/roundtrip_stress.rs`, `docs/TESTING.md`).
- **Stage 2 — Compressed-Domain Pushdown & Mutations (`src/query/`, `src/ops/`, `src/columns/ops_*.rs`):**
  - Added row random access (`frame_row_count`, `extract_row`, `extract_row_range`), substring/numeric pushdown (`find_first`, `filter_rows`, `filter_json_u64`), zero-wire-byte 256-entry bigram precheck (`query_bigram.rs`), and SIMD hex-nibble scanning (`query_nibble.rs`).
  - Added `filter_column`, `filter_frame`, `update_column`, and `map_column` with 256-bit alphabet & numeric zone-map block rejection, single-slot binary column rewrites (`O(|D|)` dictionary splice), exact cross-block split-line stitching, and deterministic multi-threaded work-stealing (`up to 39.5 GB/s`, `72.5x` speedup).
- **Stage 3 — Compound Boolean `PredExpr`, Scalar Aggregations & 2-Tier `GROUP BY` (`src/ops/expr.rs`, `src/ops/agg.rs`):**
  - Added zero-heap `Copy` `PredExpr` (`True`, `Col`, `RowContains`, `All`, `Any`, `Not`), `count_where`, `filter_where`, `filter_frame_where`, `aggregate_column` (`AggStats`), and `group_by` (`GroupRow`, `GroupKey`, L1 per-dictionary-ID `[AggStats; |D|]` buckets + parallel merge, reaching `15.1 GB/s` `1T` / `49.4x` speedup).
- **Stage 4 — Column Projection, Zero-Reencode Column Splicing & Full CLI (`src/ops/project.rs`, `src/bin/vxg.rs`):**
  - Added `select_columns` (miniature `PreparedTemplate` SIMD emission skipping unselected columns, up to `20.7 GB/s` / `37.3x` on `12T`) and `project_frame` (`GrammarProjectOut::Spliced` zero-reencode `VXG -> VXG` column splicing with `64 KiB` boundary-tail absorption, `10.7 GB/s` / `32.6x` on `12T`).
  - Upgraded `vxg` CLI with a recursive-descent compound `--where EXPR` parser, `vxg query --select`, `vxg agg`, `vxg project`, `vxg filter-frame`, `vxg update`, `vxg map`, and `-t N|auto` across commands; verified by 96/96 release + differential fuzz tests (`<= 600` lines across every `.rs` file).

## 2026-11-02 — VXG v0.4.3 Release: Zero-Copy `mmap` I/O, Zero-Output-Alloc Verification & Bounded-Memory Multi-GB Streaming

- **Zero-Dependency Cross-Platform `mmap` (`src/mmap.rs`, `src/mmap/sys.rs`):**
  - Implemented raw OS FFI memory mapping (`MappedFile`, `MappedFileMut`, `InputBytes`, `mmap_supported`) for 64-bit Unix (`mmap`/`munmap`/`msync`/`madvise`) and 64-bit Windows (`CreateFileMappingW`/`MapViewOfFile`/`FlushViewOfFile`), plus `wasm32-unknown-unknown` clean fallback (`0` external runtime dependencies).
  - Added `compress_file`, `decompress_file` (`16 MiB` bounded parallel block-batch decoding for multi-GB frames + automatic partial-output cleanup on failure), and `verify_file`.
- **Zero-Output-Alloc Parallel Verification & Bounded-Memory Compressed-Domain Streaming (`src/stream_ops.rs`, `src/stream_ops/`):**
  - Added `FrameReader<R>`, `verify_frame` (`~64 KiB` L2 scratch per worker thread, `12.0+ GB/s` warm verification on `8T`), and `verify_stream`.
  - Added `FramePeeler` cross-frame split-line stitcher (preserving interior `GRAMMAR_PACKET` blocks compressed while handling CSV column-0 cuts, multi-frame monster lines, and unterminated EOF tails) and all 10 `*_stream` compressed-domain operations (`frame_row_count_stream`, `count_where_stream`, `filter_where_stream`, `select_columns_stream`, `aggregate_column_stream`, `group_by_stream`, `project_frame_stream`, `filter_frame_where_stream`, `update_column_stream`, `map_column_stream`).
  - Scaled `StreamEncoder::new` default chunk sizes (`4..16 MiB` by thread count) and removed `8 KiB` `io::copy` double-buffering in `compress_stream` / `decompress_stream` (`1.8x–4.0x` faster multi-threaded streaming).
- **CLI Default `mmap` & Universal `--stream` (`src/bin/vxg.rs`, `src/bin/cli_support/`):**
  - Enabled `mmap` by default on regular files (`--no-mmap` opt-out) and added `--stream` + `stdin` (`-`) across `verify`, `query`, `agg`, `project`, `filter-frame`, `update`, and `map`; verified by `101/101` release tests (`tests/stage5_mmap.rs`, `tests/stage5_stream_ops.rs`).

## 2026-11-02 — VXG v0.4.4 (Stage 6): Compressed-Domain `ORDER BY` / `Top-K`, Expanded C/FFI Analytics ABI & Interactive VitePress Charts

- **Compressed-Domain `ORDER BY` / `Top-K` Heap Engine (`src/ops/top_k.rs`, `src/columns/ops_topk.rs`, `src/stream_ops/top_k.rs`):**
  - Added `SortDir`, `SortKey`, `top_k_by`, `select_top_k_by`, `top_k_by_stream`, and `select_top_k_by_stream` with deterministic `NULLS LAST` + `global_row_index` tie-breaking across threads and multi-frame streams.
  - Added $O(1)$ `MODEL_DECIMAL_PACKED` / `MODEL_DECIMAL_AFFINE` numeric zone-map block skipping, `AtomicU64` cross-worker monotonic threshold pruning, pre-pruned dictionary `alive` bitmaps, `TplRowMap::Offset` $O(1)$ row indexing on boundary-split 1-template blocks, and bounded per-template `CandTracker` ($\le K$ candidate rows materialized per template).
  - On `ndjson_varied.ndjson` (`56.5 MB`, `300,000` rows, `863` blocks), `top_k_by` and `select_top_k_by` run in **`1.38–1.90 ms` (`29.7–40.9 GB/s`, `55×–68×` vs naive `decompress + select_nth_unstable`)** on `1T` and **`1.33–1.64 ms` (`34.5–42.3 GB/s`, `54×–72×`)** on `8T`.
  - Added `--order-by COL [--desc|--asc]` to `vxg query` and `--order-by key|rows|count|sum|avg|min|max [--desc|--asc] [--limit K]` to `vxg agg`.
- **Shared Expression Parser & Expanded Zero-Overhead C/FFI ABI (`src/ops/parse_expr.rs`, `src/ffi/analytics.rs`, `include/vxg.h`):**
  - Promoted the recursive-descent compound `WHERE` and column-target parser into `vxg::ops` (`ParsedWhere`, `ParsedColPred`, `parse_col_target`, `parse_col_list`) and exported 8 new panic-safe C ABI functions (`vxg_verify_frame`, `vxg_compress_file`, `vxg_decompress_file`, `vxg_verify_file`, `vxg_count_where_str`, `vxg_aggregate_col` + `vxg_agg_stats_t`, `vxg_select_alloc`, `vxg_project_frame_alloc`).
- **Interactive Zero-Dependency VitePress Charts (`docs/.vitepress/theme/`):**
  - Added `CodecBenchmarkChart.vue`, `CompressedOpsChart.vue`, and `EnergyBenchmarkChart.vue` using native Vue 3 + VitePress CSS theme variables (`0` extra npm packages), embedded in `docs/index.md`, `docs/BENCHMARKS.md`, `docs/ENERGY_BENCHMARKS.md`, and `docs/STAGE2_DESIGN.md`.

## 2026-11-08 — VXG v0.4.5 Release: Hot-Path Optimization Wave & "Chimera Envelope" Benchmark Suite

- **Step 0 — "Chimera Envelope" Benchmark Harness (`src/baselines.rs`, `src/bin/bench_support/`):**
  - Added native C `LZ4_compress_HC_extStateHC` (`lz4hc-c-l9`) and `zstd-c-l9` baselines plus synthetic **`CHIMERA-LZ4`**, **`CHIMERA-ZSTD`**, and **`CHIMERA-ALL`** envelopes (`min_ratio` + `max_enc_GBps` + `max_dec_GBps` + `min_energy_J_per_GB` simultaneously on every dataset). See [`docs/CHIMERA_BENCHMARKS.md`](CHIMERA_BENCHMARKS.md) and `reports/v4-chimera-{1t,8t}.md`.
  - Across the 10-dataset `v4` 16 MiB suite (`8T`), `VXG v4` (`0.0252` ratio, `7.05 / 23.82 GB/s`) beats **`LZ4 Chimera`** (`4.09x` `VPI`), **`Zstd Chimera`** (`3.97x` `VPI`), and **`All-Rivals Chimera`** (`2.18x` `VPI`, `1.36x` `GCI`), winning all 3 axes (`Ratio` + `Encode` + `Decode`) simultaneously on `telemetry.csv`, `events.tsv`, `ndjson_varied.ndjson` (vs `LZ4 Chimera`), and BMP frames.
- **Step 1 — Persistent Lock-Free Worker Pool & Parallel Boundary Inspection (`src/pool.rs`, `src/ops/par_meta.rs`, `src/query/parallel.rs`):**
  - Replaced per-call `std::thread::scope` OS thread spawning and cold TLS scratch initialization across `encode`, `decode`, `query`, and `ops` with a persistent condvar/spin worker pool (`pool::parallel_for`) and parallel block-boundary inspection (`inspect_grammar_boundary`).
- **Step 2 — Multi-Template SIMD Stream Matcher, CLF NEON Nibble LUT, Dict Early-Abort & `MODEL_PATTERN` SWAR (`src/columns/`):**
  - Added `grammar_multi.rs` (`json_struct_masks64` + `scan_json_string_reg` + `SegMatcher` + batched `col_ptrs`), NEON 256-entry nibble LUT in `try_match_clf_stream`, early high-cardinality abort in `model_dict.rs`, and single-digit-token / `u32` multiply-shift fast paths in `prep_pattern.rs`.
- **Step 3 — `FAST_LZ` Periodic SIMD Overlap Unpack & 4-Stream Huffman 6-Step Decode (`src/columns/fast_lz_unpack.rs`, `src/columns/huffman_dec4.rs`, `src/columns/huffman_dec.rs`):**
  - Added `OVERLAP_SHUF16` + `copy16_periodic` (`vqtbl1q_u8` / `_mm_shuffle_epi8`) for periodic `offset < match_len` (`1..=15`) matches and zero-pair-table constant-offset 6-step (`max_len <= 9`) / 5-step (`max_len in 10..=11`) 4-stream Huffman decoding.
- **Step 4 — Cross-Block `leading_carry` Fix & Zero-Alloc Multi-Template Compressed-Domain Ops (`src/encode.rs`, `src/columns/grammar_extract.rs`, `src/columns/ops_*.rs`):**
  - Routed cross-block split-line leading fragments (`leading_carry`) directly to `Row::Raw(&data[..first_nl])`, eliminating bogus 1-row `Template 0` splits on 92% of CSV/TSV/logfmt blocks and fixing column-0 model poisoning (`events.tsv` `0.0334 -> 0.0326`, `data.csv` `0.1820 -> 0.1796`, CSV `1T` `filter_col` `52.8 GB/s` / `map` `13.2 GB/s`).
  - Eliminated per-block heap allocations (`TplColSlices`) in `ops_search.rs` / `ops_pred.rs` and added multi-template compressed-domain `filter_frame` (`ops_filter_bytes.rs`). On `ndjson_varied.ndjson` (`8T`), `agg SUM` jumped `6.13 ms -> 0.57 ms` (**`98.91 GB/s`, `152.0x` vs naive**), `group_by SUM` `5.22 ms -> 1.00 ms` (**`56.38 GB/s`, `157.9x`**), and `top_k_by` `1.33 ms -> 0.58 ms` (**`96.95 GB/s`, `138.9x`**).
- **Step 5 — Zero-Refill Streaming Buffers (`src/stream.rs`):**
  - Eliminated upfront `16+ MiB` zero-fill in `StreamEncoder::with_chunk_size`, added `flush_slice`, and added monotonic `out_len` buffer reuse in `StreamDecoder`. All `.rs` files remain `<= 599` lines and 100% of release + differential fuzz tests pass with `0` warnings.
