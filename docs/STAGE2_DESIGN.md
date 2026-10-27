# Stage 2 design sketch: map / filter / update without a full decode

Status: proposal for discussion. Nothing here is implemented except where marked.

## What exists today

- Row access and search on structured (`GRAMMAR_PACKET`) blocks: `frame_row_count`,
  `extract_row`, `extract_row_range`, `find_first`, `filter_rows` (implemented, tested).
- Free-text blocks fall back to a decode plus one SIMD scan per block (implemented).

## Constraints that shape the design

- Blocks are independent and 64 KiB by default. Any operation is naturally per block, so it
  parallelizes across blocks and an `update` only needs to rewrite the blocks it touches.
- Inside a `GRAMMAR_PACKET` block, data is columnar: a template plus per-column models
  (dictionary, decimal, timestamp, hex, ...). Operations are cheap only where the model lets
  them be cheap.
- The wire format must stay valid and canonical (the decoder rejects non-canonical packets,
  for example unsorted dictionaries). Edited blocks must be re-emitted canonically.

## Hypotheses to validate by measurement (not yet tested)

| Operation | Where it should be cheap | Where it falls back |
| :--- | :--- | :--- |
| `filter` on a dictionary column | Evaluate the predicate once per dictionary entry, then scan the id array (the existing `DictSingle` path generalizes this) | Non-dictionary columns: decode that column only |
| `map` on a dictionary column | Transform the dictionary entries (cost proportional to the number of entries, not rows), then re-canonicalize (entries may collapse or reorder) | Numeric/timestamp columns: decode the column, apply, re-encode that column only |
| `update` of a value | Block-level copy-on-write: rewrite only matching blocks; for a dictionary column, edit entries when the predicate applies to a whole value | Value changes that need a different template: re-encode the block |

The expected win is per column, not per frame: untouched columns are copied byte for byte.

## Open questions for the owner

1. Is `update` required to be in place, or is producing a new frame with only the changed blocks
   rewritten acceptable? (The second is much simpler and keeps frames immutable.)
2. Should predicates be a small fixed set (equals, contains, range on a column) or a general
   expression? A fixed set is easier to push down.
3. Column addressing: by template slot index, or by JSON key name? Names need a per-template key
   table that is derived from the static segments (feasible, not implemented).
4. Per-block metadata (bigram signature, min/max per numeric column) would speed up skipping but
   changes the wire format. Decide this once, together with the bigram question.

## Suggested first slice

`filter` with an equality/contains predicate on one named column, returning matching rows,
benchmarked against the existing `filter_rows` on the varied NDJSON file and on a real log.
That reuses existing machinery and tells us whether column-aware pushdown beats whole-row search
before any `map`/`update` work starts.

## Progress

- Implemented: `filter_json_u64` (numeric `Eq`/`Ge`/`Le`/`Range` on one JSON key, column pushdown
  on single-template blocks, per-line fallback elsewhere). Results: see Benchmarks. It returns the
  matching rows as a new `Vec`; the frame is not modified (so `update` as "new frame" remains open).
- Not implemented: `map`, `update`, string/multi-key predicates, multi-template blocks, per-block
  metadata. The wire format is unchanged.
