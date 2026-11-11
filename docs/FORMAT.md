# VXG1 wire specification — versions 1, 2, 3, 4 and 5

Versions 4 (`VERSION_V4 = 4`) and 5 (`VERSION_V5 = 5`) are the production wire formats. Version 4 uses a compact 8-byte per-block directory and unified `BLOCK_COLUMN` packets (`RAW_PACKET`, `GRAMMAR_PACKET`, `STRIDE_PACKET`, and `FAST_LZ_PACKET` with optional 4-stream canonical Huffman coding). Version 5 extends Version 4 with an optional self-contained frame prefix dictionary (`FLAG_FRAME_DICT = 0x0002`), high-density `BLOCK_DICT_LZ` (`kind = 3`, with 4-stream canonical Huffman and pure-Rust `FSE / tANS` stream coding via `FLAG_FSE = 0x10`), optional record-aligned variable-size raw blocks (`FLAG_VAR_RAW = 0x0004`), and an optional $O(1)$ per-block row-count & newline-terminator index (`FLAG_ROW_INDEX = 0x0008`). When none of `FLAG_FRAME_DICT`, `FLAG_VAR_RAW`, `FLAG_ROW_INDEX`, or `BLOCK_DICT_LZ` is active, the encoder emits `VERSION_V4` (`4`) for seamless compatibility. Version 3 added structured-column blocks with the 32-byte directory (`MODEL_SCALED_DECIMAL = 16` was added in `v0.5.0`); see [FORMAT_V3.md](FORMAT_V3.md) for model specifications. Decoders maintain full backward read compatibility with versions 1, 2, 3, 4, and 5 (plus read-only compatibility for frozen experimental `VERSION_V6 = 6` vectors).

All integers are little-endian. The format is byte-addressed and contains no implicit
alignment padding. Native GPU implementations internally repack/align buffers; that is
not part of the file contract. Every `VXG1` frame is completely self-contained (zero external dictionary files).

## File header: 32 bytes

| Offset | Type | Meaning |
|---:|---|---|
| 0 | byte[4] | ASCII `VXG1` |
| 4 | u16 | Version: `1`, `2`, `3`, `4`, or `5` (encoder emits `4` by default, or `5` when frame dictionary / `BLOCK_DICT_LZ` / record-aligned variable blocks / row-offset index are active) |
| 6 | u16 | Flags: bit 0 (`0x0001`) = CRC32C; in `v5` only: bit 1 (`0x0002`) = `FLAG_FRAME_DICT`, bit 2 (`0x0004`) = `FLAG_VAR_RAW`, bit 3 (`0x0008`) = `FLAG_ROW_INDEX`; all other bits zero |
| 8 | u64 | Total uncompressed byte count $N$ |
| 16 | u32 | Nominal block size $B$; `4,096..1,048,576`, multiple of 64 |
| 20 | u32 | Block count (`ceil(N/B)` when `FLAG_VAR_RAW` is clear; in `[ceil(N/B)..min(N, max(ceil(N/B)*8/3, ceil(N/B)+4))]` when `FLAG_VAR_RAW` is set) |
| 24 | u64 (`v1–v4`) / `u32 + u32` (`v5`) | Zero in `v1–v4`. In `v5`: `u32 dict_wire_len` (offset 24) + `u32 dict_raw_len` (offset 28) when `FLAG_FRAME_DICT` is set (`64 <= dict_raw_len <= 262,144`, `1 <= dict_wire_len <= dict_raw_len`); both zero otherwise |

The empty frame is exactly 32 bytes, with count=0, N=0 and a valid block size.

## Directory & `v5` Auxiliary Tables

### Versions 4–5 compact directory: 8 bytes per block

In versions 4 and 5, raw offsets and payload offsets are reconstructed deterministically by prefix sum over the contiguous block sequence:

| Offset within entry | Type | Meaning |
|---:|---|---|
| 0 | u32 | `payload_size \| (kind << 30)`: lower 30 bits = payload byte length (`<= 1,048,576`), upper 2 bits = block kind (`0` = RAW, `1` = LZ, `2` = COLUMN, `3` = `DICT_LZ` in `v5`) |
| 4 | u32 | CRC32C of raw block, or zero when disabled |

In `VERSION_V5`, immediately following the `8 * count`-byte block directory and preceding block 0's payload:
1. **`var_raw` table (`4 * count` bytes, present iff `FLAG_VAR_RAW` is set):** `u32[count]` uncompressed byte lengths `rn_i` (`1 <= rn_i <= B`, $\sum rn_i = N$), used by `--record-aligned` (`Options { record_aligned: true }`) to snap block boundaries to `\n` record delimiters so zero structured records split across blocks. On verbose multi-column NDJSON streams, `--record-aligned` automatically scales nominal $B$ up to `<= 1,048,576` bytes (`Options::target_block_rows`, default `2,048` rows/block) while staying strictly within the `v5` `rn_i <= B <= 1 MiB` wire bound.
2. **Frame prefix dictionary (`dict_wire_len` bytes, present iff `FLAG_FRAME_DICT` is set):** Self-contained raw (`dict_wire_len == dict_raw_len`) or `DICT_LZ`-compressed (`wire[0] & 0x80 != 0`) shared prefix dictionary (`64..=262,144` uncompressed bytes: parallel FastCover `<= 32 KiB` shingle vocabulary or macroscopic `<= 256 KiB` repeating-span coalescer), decoded once per frame and shared read-only across all worker threads without cross-block state.
3. **`row_index` table (`4 * count` bytes, present iff `FLAG_ROW_INDEX` is set; `count >= 1`):** `u32[count]` packed per-block line-delimited record metadata `row_meta_i = rows_i | (ends_nl_i << 31)` (`1 <= rows_i <= min(rn_i, 0x7fff_ffff)`, bit 31 `ROW_ENDS_NL_BIT = 0x8000_0000` set iff block `i`'s last raw byte is `\n`). Enables $O(1)$ per-block record counting, $O(\text{count})$ header-only `frame_row_count` across all block kinds (including `BLOCK_DICT_LZ`, `BLOCK_LZ`, and `BLOCK_RAW`), $O(\log \text{count})$ binary-search block jumping in `extract_row` / `extract_row_range` with zero non-overlapping block decodes, zero-barrier parallel seam scanning, and $O(1)$ multi-frame stream peeling. Decoders verify `pack_block_row_meta(decoded_block) == row_meta_i` on every inspected/verified block (`verify_frame` and `extract_row_range`), rejecting forged `row_index` entries with `Error::InvalidFrame("row_index mismatch")`.

### `BLOCK_DICT_LZ` (`kind = 3`, `VERSION_V5`) & `FLAG_FSE` (`0x10`)

`BLOCK_DICT_LZ` payloads begin with header byte `0x80 | stream_flags` (`bit 0..3` = per-stream entropy enable for `lits`, `toks`, `off_codes`, and `len_ext`; `bit 4` `FLAG_FSE = 0x10` = pure-Rust `FSE / tANS` mode when set, or 4-stream canonical Huffman when clear), followed by ULEB128 stream counts/byte lengths and five streams (`lits`, `toks` `(lit_code) | (ml_code << 4)`, `off_codes` supporting 3 repeat-offset slots `0..=2` and log2 offset classes `3..` across intra-block + optional frame dictionary history, `len_ext`, and `off_bits` extra offset bits). When `FLAG_FSE = 0x10` is set, each enabled byte stream begins with a 1-byte sub-mode tag (`1` = 4-stream canonical Huffman, `2` = pure-Rust `FSE / tANS` with `table_log in 5..=9`, packed normalized counts, and backward bit-stream decoding) and is selected strictly when its compressed size beats both raw bytes and canonical Huffman (`stream_flags == 0` decodes zero-copy directly from the payload slice). Encoder search effort (`Options::level` `1..=9`) changes only match selection, never the wire format.

### Deep Nested JSON Flattening (`depth <= 4` in `GRAMMAR_PACKET`)

`GRAMMAR_PACKET`'s structured JSON schema parser (`src/columns/json_schema.rs`, `src/columns/grammar_json.rs`) recursively flattens nested JSON objects (`{"a":{"b":{"c":...}}}`) and fixed-arity scalar JSON arrays (`[x, y, z]`) up to `MAX_NEST_DEPTH = 4` (`<= 64` leaf slots per template) directly into typed column models (`MODEL_DICT`, `MODEL_DECIMAL_*`, `MODEL_SCALED_DECIMAL`, `MODEL_TIMESTAMP`, `MODEL_PREFIX_DELTA`, etc.) while preserving exact structural template punctuation for 100% byte-identical reconstruction and dotted-key SQL pushdown (`SELECT actor.org.login, payload.commits.0.sha ...`).

### `MODEL_SCALED_DECIMAL` (`tag = 16`) & Exception-Patched `PFOR-Delta` (`0x80 | base_width`)

Fixed-scale ASCII floating-point / currency columns (`scale` fractional digits `1..=9`, optional sign, `base: i64` + bit-packed `u64` scaled-integer deltas) are stored losslessly in `GRAMMAR_PACKET` (`src/columns/model_scaled_dec.rs`) and expose exact scaled-integer zone maps and vectorised arithmetic to the compressed-domain SQL engine.

Both `MODEL_DECIMAL_DELTA` (`tag = 4`, signed row-to-row zigzag integer deltas) and `MODEL_SCALED_DECIMAL` (`tag = 16`, row-to-row zigzag scaled-decimal deltas) support **exception-patched `PFOR-Delta` sub-encoding** (`src/columns/model_delta_pfor.rs`): when the width byte has bit 7 set (`width_byte = 0x80 | base_width`, `0 <= base_width <= 63`), all `count - 1` row-to-row zigzag deltas are bit-packed at `base_width` bits, followed by `exc_count` (`ULEB128`, `<= count / 4`) sorted `(idx_delta: ULEB128, high_bits: ULEB128)` exception patches for partition/entity resets (e.g., country or county boundaries in cumulative time-series CSVs). Nullable/empty numeric cells in mixed columns are handled via a sparse presence mask without forcing the block out of `BLOCK_COLUMN`.

### Versions 1–3 directory: 32 bytes per block

| Offset within entry | Type | Meaning |
|---:|---|---|
| 0 | u64 | Raw output offset |
| 8 | u64 | Payload file offset |
| 16 | u32 | Raw block length |
| 20 | u32 | Payload length |
| 24 | u32 | CRC32C of raw block, or zero when disabled |
| 28 | u32 | Kind: 0 RAW; 1 LZ; 2 COLUMN (v3/v4, see [FORMAT_V3.md](FORMAT_V3.md)) |

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

## Compatibility & Frozen Golden Vector Policy (`v1`–`v5`)

CPU and GPU encoders choose different candidates and may emit different bytes and ratios.
Both conform to this format. CPU output is intended to be deterministic across SIMD paths
and worker counts; GPU atomic-min anchor discovery is deterministic within its algorithm.
The two encoders are not required to be byte-identical. Version 2 is backward-compatible
at the decoder: it adds opcode 7 only to frames explicitly marked version 2.

**Backward-Compatibility Contract (`0.x` through `1.0`):**
1. Every public `vxg` release guarantees 100% lossless read/decode/verify compatibility for all valid `VERSION_V1` (`1`), `VERSION_V2` (`2`), `VERSION_V3` (`3`), `VERSION_V4` (`4`), and `VERSION_V5` (`5`) frames.
2. Any incompatible header, directory, block-kind, or column-model wire layout change **must** increment the frame version (`VERSION_V6+`) or allocate a previously rejected tag/flag bit; existing tags, opcodes, and flag meanings are immutable once released.
3. Byte-frozen compatibility vectors covering `v1`–`v5` (`BLOCK_RAW`, `BLOCK_LZ`, `BLOCK_COLUMN`, `BLOCK_DICT_LZ`, `FLAG_FSE`, `FLAG_FRAME_DICT`, `FLAG_VAR_RAW` `--record-aligned`, and `FLAG_ROW_INDEX` `--row-index`) are checked into [`tests/vectors/`](../tests/vectors/) and verified on every CI push (`tests/golden.rs`) across Linux (`x86_64`, `aarch64`), macOS (`aarch64`), Windows (`x86_64`), and `wasm32-unknown-unknown`.
