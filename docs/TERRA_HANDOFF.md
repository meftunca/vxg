# Terra handoff — VXG 4 / 95 / 15

Planning prepared 2026-09-19; execution has not started under this plan.
Workspace: `/Users/mapletechnologies/Desktop/big_projects/vxg`.
Inspected implementation baseline: `0a2b4ed`. Do not reset to it; preserve newer work.

## Mission

Execute [VXG_4_95_15_PLAN.md](VXG_4_95_15_PLAN.md) in package order. The user wants
>=4 GB/s encode, >=95% size reduction and 10–15 GB/s safe decode, including a real
bitsliced experiment. Previous LZ4 +20–50% ambition remains part of the acceptance
contract. Primary hardware is Apple M2 Max, one worker, 64 KiB independent blocks.
All three columns must describe the same encoded artifact/configuration per input.

Start with P0: freeze the corpus/benchmark, audit the validated decoder boundary,
and verify the actual timed encoder. Then work through P1–P7. Continue from measured
results without asking for routine permission after each experiment. Git is secondary;
do not pause development because committing or graph cache cleanup is unavailable.

## Read before implementation

1. The full main plan, especially target scope, bit budget and benchmark profiles.
2. Current `src/block.rs`, `src/frame.rs`, `src/simd.rs`, `src/lib.rs`.
3. `src/bin/bench.rs`, `src/baselines.rs`, `docs/BENCHMARKS.md`.
4. Recent `docs/PROGRESS.md`, `docs/PROOFS.md`, `docs/FORMAT.md`, tests and fuzz targets.

The persisted Graphify map predates current functions; current code overrides its line
numbers. The format document currently describes v1 despite v2 emission; fix that in P0.

## Facts already established

- v2 emits 2-byte opcode-7 distances and still reads v1. CPU, Python and shared shaders
  implement opcode 7; shader artifacts are generated from `native/shaders/core.gpu`.
- JSON 16 MiB: output 2,457,877 bytes (ratio 0.14650089), 591,146 commands.
- Latest matched baseline: VXG 2.5151 encode / 3.1756 safe prepared decode GB/s;
  LZ4 a1 2.2818 / 8.9942. These are historic observations needing fresh paired runs.
- Separate prevalidated result 3.9524 GB/s excludes validation. It is not a safe
  one-shot LZ4 comparison. The measured v2 ceiling is unknown.
- 95% reduction permits at most 838,860 bytes on 16 MiB. Existing literals+envelope
  leave about 9.19 bits per existing command for all metadata. A fixed 32-bit command
  layout cannot meet this budget without additional compression/fewer commands.
- Stats-free normal encode, worker-local payload arena, sampled RAW bypass, fused
  opcode-7 checked decode and non-overlap copy already exist. Do not redo/name them
  as new gains. `compress_with_stats` deliberately retains diagnostics.

## First concrete deliverable

P0 should produce a corpus manifest, a serial reproducible benchmark runner with
profile labels and raw samples, validation-cost accounting, and correctness coverage
for `ValidatedFrame`. Check actual `compress()` output against `compress_with_stats()`
before using setup-frame sizes to report timed encode results.

Audit `block::decode_validated`: it currently has a safe signature with an implicit
prior-validation requirement for unchecked reads. Express that obligation in a type
or unsafe boundary, and test valid-mutated/malformed frames, overlap, tails, limits,
checksums and old versions. No new wire format is necessary for this first step.

P1's first candidate is an exact short-match inline copy; P2 determines how many
metadata bits/commands must disappear. The ratio work then includes bounded distance
modeling plus scalar/NEON bitsliced, int4 and ternary prototypes. SIMD is accepted for
measured whole-codec value, not for instruction count or marketing.

## Things that are not success

- 4 GB/s in fast mode + 0.05 ratio in another mode + 15 GB/s prevalidated decode.
- Timing unpack-free expanded commands while reporting the packed frame's ratio.
- Dropping command validation, checksum requirements or exact bounds under a safe API.
- int4 quantization, discarded JSON whitespace/order, generator-specific templates,
  decoded-output caches, external dictionaries omitted from encoded size.
- A multicore/GPU result presented as the one-worker CPU target.
- Dropping failed qualification files, reporting only the best process, or changing
  block size mid-comparison. One ordered A/B run is directional evidence only.
- Bumping the wire version without reference/golden coverage or an explicit GPU
  unsupported-mode guard. Metal CPU-feature tests are not GPU execution.

## Resume protocol

At each checkpoint record the active package, current hypothesis, exact patch,
correctness results, benchmark command/binary hash, per-profile size/speed, decision
and next experiment. Use `docs/PROGRESS.md` and the state file introduced in P0.
Keep decisive outputs under tracked `reports/qualification/`, not only ignored
`reports/local-*`. Never overwrite the accepted baseline with a trial binary.

After three losing micro-experiments on the same path, follow the profile/metadata
evidence into the next architectural experiment. If a target remains unattainable,
show the frontier and exact shortfall; do not mark the performance target complete.
G4 means minimum triple +20% LZ4 passed. G5 means full 15 GB/s stretch +50% passed.
Report which gate was actually reached.

## Ready-to-use kickoff message

> VXG için docs/TERRA_HANDOFF.md ve docs/VXG_4_95_15_PLAN.md dosyalarını oku ve planı
> P0'dan başlayarak uygula. Hedef aynı veri/ayar/çıktıda tek worker CPU ile >=4 GB/s
> encode, ratio <=0.05 ve güvenli decode >=10 GB/s + LZ4'e >=1.20x; stretch >=15 GB/s
> + LZ4'e >=1.50x. Bitsliced, lossless int4 ve ternary deneylerini ölçerek dahil et.
> Doğrulama/preparation maliyetlerini benchmark dışına saklama. Başarısız deneyleri
> kaydet, kazananları koru, rutin adımlarda onay bekleme; Git yüzünden ara verme.
> Hedef sağlanmadıysa tamamlandı deme. Her checkpoint'te ölçüm ve sonraki adımı yaz.
