# Profile-driven optimization sequence

The active CPU roadmap is now [VXG 4 / 95 / 15](VXG_4_95_15_PLAN.md), with
[Terra handoff instructions](TERRA_HANDOFF.md). It defines simultaneous speed/ratio
targets, bitsliced experiments, safety boundaries and reproducible acceptance gates.
The notes below are historical background; consult `PROGRESS.md` before retrying
experiments that have already been rejected.

No speed target below is a measured delivery result. Keep scalar/reference semantics and
cross-backend golden tests as gates for every change.

## Gate 0 — compile and validate

Compile the Rust default core on x86_64 and ARM64. Execute SIMD and CRC equivalence,
malformed-input fuzzing and checksum/resource-limit tests. Compile each native SDK target
on its platform and run three-way interoperability with hardware validation tools. Fix
compiler/driver/ABI issues before accepting benchmark conclusions.

## Gate 1 — establish the actual bottleneck

Record cycles/output-byte, instructions, branch misses, cache traffic, allocation and
thread-launch cost for at least logs, database pages, code, structured records and already
compressed bytes. Sweep 4/16/64/256/1024 KiB blocks and 1..physical-core-count workers.
Record ratio against reference LZ4/Zstd, not speed alone. An all-RAW win is not evidence
of a good compressor on compressible data.

## CPU experiments

Replace per-call thread creation with a reusable explicitly owned pool, and sorting with
indexed result slots. Reuse encode workspaces across calls. Test hash generation/clearing
against a generation table and fast incompressible sampling. Add optional bounded two-way
candidates; account for extra probes and comparison work. Compare bounds-exact hand SIMD
with platform memcpy/memset; hand SIMD is not automatically faster. Specialize pattern
seed construction to avoid division, and compare AVX2 against AVX-512 rather than assuming
width wins. Investigate CRC batching only when checksummed profile shows it material.

An unrolled trusted decoder must retain safe untrusted entry validation. Any wild-copy
optimization needs internally owned padding and proof of source initialization; never
silently weaken the public slice contract to remove checks.

## GPU experiments

First measure current lane-0 encoder time, hash atomic contention, branch divergence,
barrier cost, occupancy and copy volume. Candidate arrays or subgroup match voting can
parallelize parsing but need deterministic conflict resolution and new correctness tests.
The current earliest hash anchor can miss newer useful matches: compare first/last/multiple
candidates with explicit work bounds. Add compact device-side output scan to avoid reading
full B-byte slots when compressed output is tiny.

A GPU-resident API and pinned/staged double-buffering may matter more than changing token
bits. CUDA graphs, asynchronous shared-memory copy, architecture-specific TMA and subgroup
collectives are opt-in experiments after measurements show relevant costs. They are not
implemented in v0.1. On Metal shared memory, remove unnecessary host Vec↔MTLBuffer copies
with a carefully specified ownership API before advertising zero-copy.

## Format experiments

Fixed 16-bit commands can lose on tiny sequences; benchmark token-density histograms before
freezing the format. Measure repeated-distance frequency before adding a four-entry cache.
Optional entropy coding belongs in a new clearly separated profile; low-alphabet random
bytes are an expected weak spot for this no-entropy codec. Dictionary, transforms and
media-specific residual packing should remain separate from the general byte-stream contract.

Acceptance should require correctness, a reproducible ratio/speed Pareto improvement and
no concealed resource or safety regression. There is no mathematical reason to enable
every available hardware instruction if it makes the measured workload slower.
