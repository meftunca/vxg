# Mathematical contracts and their exact scope

This document proves properties of the specified algorithm under explicit assumptions.
It does not claim a verified Rust compiler, verified unsafe intrinsics, a GPU driver proof,
or faster execution than an existing codec. Finite tests and unbounded reasoning are different.
The Kani harnesses are provided but were not executed in the preparation environment.

## 1. Losslessness by induction on instructions

Let the source block be `x[0..n)`. Invariant before each instruction: the decoder has
produced exactly `x[0..p)`. The encoder emits literals taken verbatim from the source,
so copying `l` literals extends the invariant to `p+l`.

For MATCH, the encoder checks at least eight equal bytes and extends only while
`x[p+j] == x[p+j-d]`. Its candidate is earlier, so `1 <= d <= p`. For every emitted
match byte `j`, the decoder's read index `p+j-d` is strictly smaller than its write index
`p+j`. Induction on `j` therefore makes that read a correct, initialized byte and proves
the written byte equals `x[p+j]`. REP has the same semantics with a cached valid distance.

When `d` is 1, 2, 4 or 8, the encoder may emit the first d bytes of the matched run as
a seed. The checked equality implies d-periodicity of the run. FILL/PATTERN reconstruction
therefore emits the same bytes. LIT contributes only its verbatim literal span. Every
instruction makes progress, the encoder partitions the source into disjoint covered spans,
and the final literal instruction covers the tail. Thus the complete output is `x`.
RAW copies the source directly. Independent blocks plus a correct directory preserve
concatenation order, proving `D(E(x)) = x` for all source byte strings in the format's size domain.

**Implementation assumptions:** exact comparison; no unsafe out-of-bounds/aliasing bug;
correct parser arithmetic and serialization; correct compiler/runtime/hardware execution.
Tests, sanitizers and future formal harnesses are intended to challenge these assumptions,
not to erase them by naming the reasoning a proof.

## 2. Overlapping copies reduce to immutable-seed indexing

For an already initialized prefix and `1 <= d <= p`, define the seed
`S[t] = output[p-d+t]`, for `0 <= t < d`. We claim for all run indices `j >= 0`:

```text
output[p+j] = S[j mod d].
```

For `j < d`, the LZ recurrence reads that seed position directly. For `j >= d`, it reads
the previously generated index `j-d`. The induction hypothesis gives `S[(j-d) mod d]`,
which equals `S[j mod d]`. This proves the identity for arbitrary lengths and periods.

The GPU decoder reads exclusively from this immutable pre-run seed, allowing lanes to
produce different run bytes independently. The CPU snapshots periods <=64. It builds a
tile of `floor(256/d)*d` bytes; because tile size is a multiple of d, repeated tile copies
do not shift the pattern phase, including periods 3, 7, 17 and 63.

For larger d the CPU first copies `min(d,r)` bytes from the old prefix. If more remain,
it doubles the completed prefix, copying `min(done,r-done)` bytes each round. Every source
is already initialized and disjoint from its destination. While more than a full `done`
remains, `done` is an integer multiple of d, so each round preserves phase. The final
partial round is a prefix of the same pattern. The number of rounds is logarithmic in
`r/d`, but total bytes copied remain linear in r.

**Correction to the earlier discussion:** a literal bytewise LZ implementation has a
long recurrence chain. This is not an inherent depth lower bound for all LZ4 decoders.
Existing decoders can also special-case short distances or expand periodic seeds. FILL
is convenient metadata, not proof of a unique asymptotic advantage over LZ4.

## 3. Packed-word GPU race discipline

A logical byte range can start/end inside a u32 word. Distinct lanes own distinct output
words for one segment tile. Before any writes, lanes read their required old prefix bits
and seed bytes into private values. A workgroup barrier separates that read phase from
writes. A second barrier separates writes from the next tile or segment. Thus one lane
cannot read a shared boundary word while another lane changes it in that tile.

Only already-produced prefix bytes are preserved; unused high bytes in a new word are
initialized, not read as valid data. Physical block output offsets are multiples of 64,
so different workgroups do not share output words. All participating lanes execute the
same segment-length loop and barriers. These facts establish race freedom in the abstract
SIMT model **assuming** the backend barrier orders the relevant memory and command execution.
Actual Metal/CUDA/Vulkan compiler/driver validation is still required. ASan/UBSan alone
are not race detectors; the thread emulator does not prove a GPU memory model correct.

## 4. Exact maximum encoded size

Partition N bytes into `k = ceil(N/B)` blocks, with lengths `n_i` summing to N. For each
block the encoder stores a compressed payload only if its length is less than `n_i`;
otherwise it stores a RAW payload of exactly `n_i`. Hence `payload_i <= n_i`.

The header is 32 bytes and each directory entry 32 bytes, so:

```text
M = 32 + 32*k + sum(payload_i)
  <= N + 32 + 32*ceil(N/B).
```

The bound includes checksums because their fields are already in the directory. It is
attained when every block is RAW. For N=0, M=32. `max_compressed_size` checks usize overflow
and the u32 block-count limit. Fixed block framing makes the total overhead O(N/B), not
O(1) as N grows with fixed B. At B=65,536, directory overhead is about 0.048828% plus header
and last-block rounding, before considering compression savings.

Counting argument: there are `256^N` N-byte inputs but only
`sum(i=0..N-1, 256^i)=(256^N-1)/255` shorter byte strings. An injective lossless encoder
cannot shorten all N-byte inputs. This does not force a specific RAW opcode; RAW is our
chosen mechanism for the stated expansion bound.

## 5. Bounded CPU work, including the terms that were previously omitted

At each probe, either the position advances by at least one after failure, or a checked
match of at least eight bytes is accepted and the position advances by its full length.
No visited position is revisited. Therefore probes P <= n for each block. Accepted match
spans are disjoint, so their lengths have sum R <= n and accepted count A <= floor(n/8).

The unsuccessful initial candidate test reads only eight bytes. Extension after a valid
initial match performs full-vector comparisons followed by a bounded tail. For SIMD width
V, extension work is O(R/V + A*V) with the current scalar-tail implementations. V is a
hardware constant. Literal copying is O(n), metadata is O(A+1), and hash-table clearing
costs O(S) per block, where S=16,384 entries. Thus the CPU encoder has:

```text
W(N) = O(N + S*ceil(N/B))
```

on a word-RAM model with constant-time machine-word operations and bounded allocation
cost per touched byte. For fixed B and S this is linear in N. Allocation, sorting the
block results (`O(k log k)` in this implementation), CRC and thread setup are additional
whole-frame costs: **the current top-level compressor is not claimed to be strictly O(N)
for unbounded k because of its sorting step**. The per-block search bound is unaffected.
The block ordering can be changed to direct indexed result slots in a later release.

For an all-failure scan with fixed minimum step s, probes <=ceil(n/s). With successful
matches and a changing step, do not claim total probes <=N/s without adding the match
steps; P<=N is the unconditional testable bound used here.

GPU first-anchor discovery examines every valid 8-byte position and uses atomic-min into
2,048 buckets. Its serial greedy scan is monotonic and accepted extensions are disjoint.
It has bounded logical work but atomic contention, modulo costs and barriers are not
constant-time cycle guarantees. GPU and CPU search are intentionally different.

## 6. What can be proved about performance

For a chosen memory hierarchy boundary let Q be actual transferred bytes, B its effective
bandwidth, U instruction work, R effective instruction issue capacity and D critical-path
time. Any execution obeys a model-dependent lower bound:

```text
T >= max(Q/B, U/R, D).
```

Only under a cold-input/output streaming model with relevant bytes crossing that boundary
can one substitute `Q >= M+N`; write allocation, match reads and extra copies can increase Q.
Cache-resident measurements may not send M+N bytes through DRAM at all. Consequently
`B_DRAM/(1+r)` is not an unconditional limit on every benchmark's reported decoded GB/s.
GPU e2e also includes transfers, host validation, framing, CRC and allocation.

An approximate control-cost model is `U/N ~= copy_work_per_byte + q/lambda`, where
`lambda=N/command_count`. This guides measurements, not a proof of latency or a universal
instruction count. Longer matches can reduce metadata but cost more search or worsen
parallel dependencies. Our 16-bit command + extension format does not guarantee q<=8 µops.
Neither average lambda>=64 nor a particular GB/s number is guaranteed on arbitrary data.

For a precisely specified machine and input, demonstrated bounds `T_VXG <= U` and
`T_LZ4 >= L` with `U < L` would prove that comparison. We have not established such
machine-specific bounds. Therefore this package makes **no speed-dominance claim**.

## 7. Executable checks versus formal verification

`verification/check_model.py` exhausts the 16-bit header domain, tests symbolic source-index
identities over finite bounds, checks work/size invariants, tests malformed frames, and
cross-checks an independently written bytewise decoder against the actual shared GPU
algorithm under CPU emulation. These are valuable finite checks, not an unbounded verifier.

`src/proofs.rs` contains Kani harnesses for the actual Rust control-word helper and bounded
frame-size arithmetic. They deliberately do not claim coverage of compression, parser,
SIMD, FFI or GPU kernels. To run on a supported Kani installation:

```sh
cargo install --locked kani-verifier
cargo kani setup
cargo kani --lib
```

Kani was not run here. For stronger assurance, extend the harness to small source blocks,
parser length arithmetic, nonaliasing copy contracts and concurrency abstractions, then
report tool/version/unwind bounds separately. See VALIDATION.md for what actually ran.
