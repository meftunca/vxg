# VXG1 wire specification — version 3 (structured-column blocks)

Version 3 extends the v2 frame layout with one additional block kind. The header and
directory are unchanged from [FORMAT.md](FORMAT.md); only the version number, the accepted
block-kind set and one canonicality rule change.

## Changes over version 2

| Item | v2 | v3 |
| --- | --- | --- |
| Version field | 2 | 3 |
| Block kinds | 0 RAW, 1 LZ | 0 RAW, 1 LZ, 2 COLUMN |
| COLUMN payload | rejected | canonical structured-column packet |
| Command validation (`validate_commands`) | supported | unsupported for COLUMN blocks |
| Validated replay (`into_validated`) | supported | unsupported when any block is COLUMN |

Encoders default to version 2. `Options { v3: true }` emits version 3 and may choose COLUMN
payloads; decoders accept versions 1, 2 and 3.

## Block-kind selection (encoder)

For every independent block when `Options { v3: true }` is enabled, the encoder first builds a
structured packet via `columns::encode_grammar_block_into` directly into the worker payload
arena. When that packet compresses strictly smaller than half the raw block length
(`packet_len < raw_len / 2`), it is accepted immediately without running the LZ matcher.
Otherwise the LZ/RAW payload is built and the structured packet is kept only when it is
**strictly smaller** than the LZ payload (`packet_len < lz_len`); ties and losses keep LZ/RAW,
and RAW remains the fallback for incompressible blocks. The choice is charged: the packet
contains every selector, dictionary, table, escape and padding bit, and the frame directory
records the stored payload length. A COLUMN directory entry is canonical only when
`2 <= payload_len < raw_len`; v2 frames must not contain kind 2, and the directory/offset/total
rules are unchanged from v2.

## COLUMN payload: canonical structured packet

The payload is byte-compatible with the P5D research packet in `src/columns.rs`, which the
`vxg-column-eval` oracle serializes and independently decodes. The first byte is the packet
selector: `1` = grammar packet. `0` (raw packet) is reserved and never emitted inside a
frame; the RAW block kind serves that purpose. After the selector the packet stores all of
the following, canonically and strictly validated:

1. Raw length (ULEB), template count (ULEB, <= 4096) and a canonical ascending block-local
   segment dictionary (length-prefixed entries; unique and strictly ascending).
2. Per template: slot count (<= 128) and bit-packed segment selectors over the dictionary;
   duplicate templates are rejected.
3. Row count (<= raw length + 1), a one-bit row selector stream (0 = raw row, otherwise
   template index + 1) and length-prefixed raw rows.
4. Per template and slot, one model payload chosen by exact charged length (ties keep the
   earlier entry of this fixed order). The encoder emits the charged minimum. The strict
   oracle decode (`decode_block_strict`, used by `vxg-column-eval`) recomputes the minimum
   and rejects a payload whose re-chosen encoding differs, i.e. non-canonical input; the
   production decoder (`decode_block` / `decode_block_into`) enforces every structural rule,
   bound, count and ordering but does not recompute representation minimality, and frame
   corruption is covered by the optional CRC32C:
   - `raw-length-prefixed`
   - `canonical-byte-dictionary`
   - `canonical-decimal-affine`, `canonical-decimal-min-width-bitpack`,
     `canonical-decimal-delta-bitpack`
   - `quoted-fixed-width-lower-hex-suffix`, `quoted-fixed-width-hex-epoch-counter`
   - `quoted-canonical-decimal-suffix-bitpack`, `quoted-canonical-iso8601-same-year`
   - `quoted-semicolon-segment-dictionary`
   - `quoted-plain-byte-periodic-1-to-64`, `quoted-lower-hex-periodic-1-to-64`,
     `quoted-fixed-width-hex-gf2-affine-recurrence`
5. Nothing else: the packet must consume exactly its payload bytes.

Rejected without partial output: trailing bytes, truncated streams, nonzero bit padding,
out-of-range indices or widths, duplicate/unused/out-of-order dictionary entries,
non-canonical decimals (leading zeros), decimal values above 999,999,999, segment lengths
above 64 bytes or segments outside `[a-z0-9 ]`, more than 1024 segments per value, trailing
pad above 4096, zero table sizes where disallowed, and any value that does not parse for its
claimed model.

Documented bounds: raw length <= block size (<= 1 MiB), templates <= 4096, slots <= 128,
segment and epoch-prefix dictionaries <= 64 entries, segments per value <= 1024, segment
length <= 64 bytes, trailing pad <= 4096, hex suffix width 8 bytes, epoch literal followed
by exactly 16 lowercase hex digits.

Validation is fused into the safe decode before row emission: `PreparedColumn::parse_and_validate`
checks every structural bound, bit width, dictionary ordering and count before `PreparedTemplate`
emits bounded rows into the preallocated output buffer.

## Profiles and availability

- Safe one-shot decode (`cpu_preallocated_decode`, `cpu_full_frame_preallocated_decode`)
  supports v3.
- Command validation, validated replay and the GPU backends support v1/v2 only. A v3 frame
  containing COLUMN blocks returns `Error::Unsupported` on those paths rather than silently
  reinterpreting data; the bench prints an explicit note and skips those profiles.

## Q95-r2 evidence (all 8 members, one worker, 64 KiB blocks, checksums off)

| Dataset (`corpus/qualification/r2/...`) | v3 ratio | v3 encode (GB/s) | v3 safe decode (GB/s) | LZ4-a1 encode / decode (GB/s) | Zstd-1 ratio / encode / decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| `json-records-16MiB.ndjson` | `0.0231` | `1.790` | `10.048` | `2.483` / `9.339` | `0.054` / `1.666` / `3.349` |
| `json-records-64MiB.ndjson` | `0.0231` | `1.757` | `9.729` | `2.350` / `9.132` | `0.054` / `1.657` / `3.305` |
| `field-order-optional-escaping-16MiB.ndjson` | `0.0448` | `1.163` | `4.835` | `1.417` / `4.901` | `0.104` / `0.986` / `1.938` |
| `field-order-optional-escaping-64MiB.ndjson` | `0.0448` | `1.169` | `4.458` | `1.347` / `4.767` | `0.104` / `0.962` / `1.904` |
| `event-mix-identifiers-16MiB.ndjson` | `0.0258` | `1.473` | `6.972` | `1.812` / `5.413` | `0.078` / `1.115` / `1.946` |
| `event-mix-identifiers-64MiB.ndjson` | `0.0258` | `1.475` | `7.000` | `1.790` / `5.494` | `0.078` / `1.179` / `1.904` |
| `bursty-variable-records-16MiB.ndjson` | `0.0486` | `1.063` | `3.922` | `1.270` / `4.219` | `0.120` / `0.813` / `1.681` |
| `bursty-variable-records-64MiB.ndjson` | `0.0488` | `1.015` | `3.852` | `1.294` / `4.181` | `0.119` / `0.782` / `1.442` |

The full eight-member r2 packet accounting lives in
`reports/qualification/experiments/P10-columns-r2` and frame-level benchmarks in
`reports/r2-v3-all-8-members.csv` and `reports/r2-event-bursty-baselines.csv`.

