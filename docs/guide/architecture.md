# Architecture & SIMD Internals

VXG is built around a simple principle: **modern data is rarely an unstructured byte soup**, yet general-purpose compressors treat every input as a flat 1D byte stream. At the same time, a production codec must **never regress** on unstructured prose, source code, or binary blobs.

---

## 1. Block-Independent Multi-Packet Pipeline

Every VXG frame partitions the input into fixed-size independent blocks (`64 KiB` default, configurable from `4 KiB` to `1 MiB` in multiples of `64` bytes). Each block is encoded and decoded **without any cross-block state**, enabling:
- **Linear multi-core scaling** via lock-free `AtomicUsize` work-stealing across blocks
- **$O(1)$ random-access block extraction** using the compact 8-byte v4 directory
- **Bounded working set** (each thread reuses a thread-local scratch arena)

```mermaid
flowchart TD
    A["Raw Input Block (e.g. 64 KiB)"] --> B{"Format & Cost Classifier"}
    B -->|"Structured Records (NDJSON / CSV / TSV / logfmt / CLF / Pretty JSON)"| C["GRAMMAR_PACKET (0x92)"]
    B -->|"2D / Strided Media (BMP / RGBA / RGB / 16-bit / 64-bit)"| D["STRIDE_PACKET (0x93)"]
    B -->|"Prose / Code / Binary"| E["FAST_LZ_PACKET (0x94) or LZ Block"]
    C --> F{"Strict Byte-Budget Gate (< Raw Size)"}
    D --> F
    E --> F
    F -->|"Smallest Valid Payload"| G["Emit VXG v4 Block + CRC32C"]
    F -->|"Incompressible"| H["Emit RAW Block (Zero Expansion)"]
```

---

## 2. The Four Packet Engines in `BLOCK_COLUMN`

### 2.1 `GRAMMAR_PACKET` (`0x92`) — Vectorized Execution Grammar

When a block contains newline-delimited records (JSON objects, CSV/TSV rows, `key=value` logfmt lines, bracketed Apache/Nginx CLF logs, or multi-line pretty-printed JSON objects):

1. **SIMD Structural Tokenization:**
   - Uses 64-byte bitmask scanners (`delimited_masks64` and `json_quote_mask64` via ARM64 `NEON` `vpaddq_u8` or `x86_64` `AVX2` `_mm256_movemask_epi8` / `SSE2`) to locate quotes, escapes, and delimiters in parallel.
2. **Template & Slot Extraction (`≤ 64` templates, `≤ 64` slots):**
   - Separates invariant structural literals (`{"ts":"`, `","level":"`, `","latency":`, `}\n`) from variable column slots.
   - Handles schema drift and optional fields via multi-template dictionary indexing, and seamlessly falls back to chunked sub-records when rows exceed 64 columns.
3. **11 Specialized Column Models:**
   - `MODEL_CONST` (`0`): Constant column across all rows of a template
   - `MODEL_DICT` (`1`): Dictionary + bit-packed or sparse-mode indices (`1..=16` bits)
   - `MODEL_INT_DELTA` (`2`): ZigZag / affine / frame-of-reference integer progressions
   - `MODEL_TIMESTAMP` (`3`): Branchless 64-bit SWAR ISO-8601 timestamp parser & bit-packer
   - `MODEL_HEX_SUFFIX` (`4`): Quoted & unquoted SIMD hex-nibble packer (`vqtbl1q_u8` / `_mm_shuffle_epi8`)
   - `MODEL_PERIODIC_STRING` (`5`): Cyclic string period detector (`1..=64`)
   - `MODEL_EPOCH_COUNTER_ID` (`6`): Composite `prefix-epoch-counter-suffix` decomposer
   - `MODEL_SEGMENT_DICT` (`7`): Semi-colon/phrase dictionary + `DEC_10K` lookup table
   - `MODEL_DECIMAL_DELTA` (`8`): Fixed-point decimal (`1..=3` fractional digits) delta coder
   - `MODEL_FRONT_DICT` (`9`): Front/back-coded (`LCP` + `LCS` + 4-bit/5-bit SIMD nibble palette) string dictionary
   - `MODEL_PATTERN` (`15`): Digit/literal patterned columns (non-ISO timestamps, IPv4, versions, prefixed IDs) with SWAR shape verification and `Copy16` constant-affix decode
4. **Fused Single-Pass Replay (`BoundStream` & `FastPiece`):**
   - During decode, all columns are unpacked into a contiguous thread-local arena (`DecodeScratch`), and rows are reconstructed via branchless 16-byte SIMD unaligned vector copies (`vst1q_u8` on `aarch64`, `_mm_storeu_si128` on `x86_64`).

### 2.2 `STRIDE_PACKET` (`0x93`) — Strided Decorrelation + SubGreen

For 2D images, framebuffers, and fixed-stride binary arrays (`stride ∈ {2, 3, 4, 8}`):
- Applies optional **SubGreen** color decorrelation (`R' = R - G`, `B' = B - G`) and per-plane horizontal delta coding.
- Uses a **48-byte / 64-byte SIMD constant-run fast skip** during encoding and reconstructs identical pixel runs in **16-pixel / 64-byte SIMD chunks** (`vst4q_u8` / `vst3q_u8` on `aarch64`, `_mm_shuffle_epi8` + `_mm_unpacklo_epi8` on `x86_64`).

### 2.3 `FAST_LZ_PACKET` (`0x94`) — 4-Stream Canonical Huffman + Fast-LZ

For prose, source code, and general binary data:
- Parses literals, match lengths, and 16-bit offsets using a dual-slot hash table with lazy match evaluation.
- Applies **4-stream interleaved canonical Huffman coding** (`<= 11-bit` codes) independently to literal bytes, token bytes, match-extension bytes, and offset high/low bytes when entropy coding reduces payload size.
- Decodes via a single-table `2048`-entry (`u16` symbol + bit-length) lookup loop reading 4 bitstreams in parallel to saturate CPU instruction-level parallelism.

---

## 3. Cross-Architecture SIMD Parity (`aarch64` & `x86_64`)

Every hot path in VXG has matched vector implementations for both **ARM64 (`NEON`)** and **x86_64 (`SSE2` / `SSSE3` / `SSE4.1` / `AVX2` / `AVX-512`)**:

| Subsystem | `aarch64` (`NEON`) | `x86_64` (`SSE2` / `SSSE3` / `SSE4.1` / `AVX2`) |
| :--- | :--- | :--- |
| **16-Byte Grammar & LZ Copies** | `vld1q_u8` / `vst1q_u8` (`u128` hardware registers) | `_mm_loadu_si128` / `_mm_storeu_si128` (zero GPR pressure) |
| **JSON / Delimiter Structural Masks** | `vceqq_u8` + `vpaddq_u8` pairwise bit-pack | `AVX2` `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8` |
| **RGB (`stride=3`) Deinterleave / Interleave** | `vld3q_u8` / `vst3q_u8` + 48B constant skip | `SSSE3` `_mm_shuffle_epi8` (`pshufb`) + 48B constant skip |
| **RGBA (`stride=4`) Prefix Sum & Interleave** | `vextq_u8` Hillis-Steele scan + `vst4q_u8` | `_mm_slli_si128` Hillis-Steele scan + `_mm_unpacklo/hi_epi8/16` |
| **4/5/6-Bit Palette & Front-Dict Unpack** | `vqtbl1q_u8` / `vqtbl2q_u8` / `vqtbl4q_u8` | `SSSE3`/`SSE4.1` `_mm_shuffle_epi8` + `_mm_blendv_epi8` |
| **CRC32C Checksum** | Hardware ` __crc32cd` (`crc` target feature) | Hardware `_mm_crc32_u64` (`sse4.2` target feature) |

---

## 4. Compressed-Domain Execution Engine (`query` & `ops`)

Because every `GRAMMAR_PACKET` separates invariant structural segments from typed per-slot column streams inside self-contained `64 KiB` blocks, VXG executes analytical and transformational workloads directly on compressed blocks:

1. **Zero-Unpack Block & Column Pruning:**
   - **256-Bit Alphabet & Zero-Wire-Byte Bigram Prechecks:** Extracts exact per-column byte alphabets and 256-entry bigram presence tables from dictionary/pattern/template headers in $O(\text{header})$ time without adding a single wire byte, skipping non-matching blocks before unpacking any row.
   - **Numeric Zone-Map Rejection:** Reads `[min..=max]` directly from bit-packed and affine decimal headers to prune range/equality predicates in $O(1)$.
2. **Compound Boolean Bitmasks & 2-Tier `GROUP BY`:**
   - Evaluates arbitrary `PredExpr` trees (`AND` / `OR` / `NOT`) as 64-bit `TplMask` word arrays with short-circuit leaf pruning so untouched columns are never decoded.
   - Aggregates `SUM` / `MIN` / `MAX` / `COUNT` directly over integer and dictionary streams, accumulating per-dictionary-entry `[AggStats; |D|]` arrays in L1 cache before mapping to global group keys (**40+ GB/s** effective throughput).
3. **Single-Slot Mutation & Zero-Reencode Column Splicing:**
   - `update_column` and `map_column` splice untouched blocks and untouched columns byte-for-byte, rewriting only the target slot (`O(|D|)` for dictionaries).
   - `select_columns` constructs a miniature `PreparedTemplate` over only the projected slots, while `project_frame` splices the selected compressed column payloads directly into a new `GRAMMAR_PACKET` (`GrammarProjectOut::Spliced`, absorbing `64 KiB` boundary carry lines as 1-row raw tails) at **10+ GB/s** without re-encoding.
