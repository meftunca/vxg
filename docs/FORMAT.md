# VXG1 wire specification — versions 1, 2 and 3, experimental

Version 3 adds structured-column blocks; see [FORMAT_V3.md](FORMAT_V3.md) for the
extension, its canonicality rules and the encoder selection policy.

All integers are little-endian. The format is byte-addressed and contains no implicit
alignment padding. Native GPU implementations internally repack/align buffers; that is
not part of the file contract. No external dictionary, entropy state, or cross-block reference.

## File header: 32 bytes

| Offset | Type | Meaning |
|---:|---|---|
| 0 | byte[4] | ASCII `VXG1` |
| 4 | u16 | Version: 1, 2 or 3; encoders emit 2 by default, 3 with `Options { v3: true }` |
| 6 | u16 | Flags: bit 0 = CRC32C; all other bits zero |
| 8 | u64 | Total uncompressed byte count |
| 16 | u32 | Block size B; 4,096..1,048,576, multiple of 64 |
| 20 | u32 | Block count = ceil(N/B) |
| 24 | u64 | Reserved, zero |

The empty frame is exactly 32 bytes, with count=0, N=0 and a valid block size.

## Directory: 32 bytes per block

| Offset within entry | Type | Meaning |
|---:|---|---|
| 0 | u64 | Raw output offset |
| 8 | u64 | Payload file offset |
| 16 | u32 | Raw block length |
| 20 | u32 | Payload length |
| 24 | u32 | CRC32C of raw block, or zero when disabled |
| 28 | u32 | Kind: 0 RAW; 1 LZ; 2 COLUMN (v3 only, see [FORMAT_V3.md](FORMAT_V3.md)) |

Raw offsets are contiguous from 0. All blocks except the last have length B. The payload
area begins immediately after the directory; payloads are contiguous, in directory order.
No holes, duplicate ranges, trailing bytes or noncanonical block counts are allowed.
RAW length equals raw length. LZ payload length is at least 16 and **strictly smaller** than raw length.

CRC32C uses reflected polynomial `0x82f63b78`, initial state `0xffffffff`, final XOR
`0xffffffff`; `CRC32C("123456789")=0xe3069283`. CRC is error detection, not authentication.

## LZ block

Prefix: four u32 values: command count, control bytes, attribute bytes, literal bytes.
Then exactly those three byte streams, in that order. Their lengths plus 16 must equal payload length.
No implicit `END` token: the count and exact stream/output exhaustion define the end.

### Control word

```text
15                         8 7                 3 2         0
+---------------------------+-------------------+-----------+
| run length code (8 bits)  | literal code (5)  | opcode(3) |
+---------------------------+-------------------+-----------+
```

Literal codes 0..30 represent that count. Code 31 is followed in the **control stream**
by a full u32 length >=31. Run codes 0..254 represent that count; code 255 is followed
by a full u32 length >=255. When both extensions occur, literal extension is first.
Lengths are actual counts, not `length - minimum_match`. These are not varints.

Each instruction first copies its literal bytes from the literal stream, then executes
its run. Literal-only instructions have positive literal length and zero run; every other
instruction has positive run. MATCH/REP runs are at least 8. Opcode 7 is invalid in
version 1 and is the short-distance MATCH form in version 2.

| Opcode | Name | Attribute bytes | Run meaning |
|---:|---|---:|---|
| 0 | LIT | 0 | No run |
| 1 | MATCH | 4 | u32 backward distance; update recent distance |
| 2 | REP | 0 | Use previously saved MATCH distance |
| 3 | FILL | 4 | First byte repeated; upper three bytes zero |
| 4 | PATTERN2 | 4 | First two bytes repeated; upper two bytes zero |
| 5 | PATTERN4 | 4 | All four bytes repeated |
| 6 | PATTERN8 | 8 | All eight bytes repeated |
| 7 (v2) | MATCH16 | 2 | u16 backward distance; update recent distance |

`MATCH16` has the same copy semantics as `MATCH`, but its distance is a little-endian
u16 and therefore must be in `1..=65,535`. It uses exactly two attribute bytes and no
extension. Version-2 streams may mix opcodes 1, 2 and 7; all three update or consume the
same recent-distance state as specified above. A version-1 decoder must reject opcode 7,
and a version-2 decoder must continue to accept every valid version-1 stream.

Pattern phase starts at seed index 0 for each instruction. Run length need not be a
multiple of seed length. Pattern instructions **do not** update recent distance.
Recent starts at zero, so REP before a valid MATCH is invalid.

MATCH/REP semantics at position `p`, distance `d`, length `r`:

```text
require 1 <= d <= p
for j in 0..r:
    output[p+j] = output[p+j-d]
```

This permits overlap. `memmove` is not a general replacement because the source can
include bytes not yet produced. See the periodic-seed proof for a parallel equivalent.

## Reject conditions

Header/offset arithmetic must be checked. Directory validation precedes output allocation.
Reject unknown flags/kinds/opcodes, invalid extension minimality, nonzero seed padding,
zero progress, insufficient attributes/literals, distance before initialized prefix,
output overrun, command-count mismatch, or unconsumed streams. Enforce caller output limit.
CRC verification is separate from structural validation; an explicitly disabled CRC check
is not permission to skip structural validation for untrusted GPU input.

## Compatibility

CPU and GPU encoders choose different candidates and may emit different bytes and ratios.
Both conform to this format. CPU output is intended to be deterministic across SIMD paths
and worker counts; GPU atomic-min anchor discovery is deterministic within its algorithm.
The two encoders are not required to be byte-identical. Version 2 is backward-compatible
at the decoder: it adds opcode 7 only to frames explicitly marked version 2. Future
incompatible changes require a new version; do not silently reinterpret earlier opcodes.
