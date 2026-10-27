# Roadmap

Focus is deliberately narrow. Each stage must have measured, reproducible results before the
next one starts.

| Stage | Goal | Status |
| :--- | :--- | :--- |
| 1 | Core performance: encode/decode speed and ratio on real-world data | In progress |
| 2 | Compressed-domain data ops: `map`, `filter`, `update` without a full decode | Planned (`filter`/`find_first`/row access exist) |
| 3 | GPU acceleration (Metal first), measured end to end, not kernel-only | Planned, after stage 2 |

Removed from the plan: VIMG (image format) and VVID (video format).

## Stage 1 rules

- Benchmark on real files, not only synthetic data. The compressed-domain query results on
  structured NDJSON (20-40x) did not carry over to free-text logs (1.4-3.1x); see
  [Benchmarks](./BENCHMARKS.md).
- Compare against LZ4 and Zstd at equal block budget, report ratio and speed together, and
  keep Zstd rows where Zstd wins.
- Single-thread numbers and multi-thread numbers are reported separately.

## Open design question (stage 2)

A bigram signature could let queries skip free-text blocks, but it needs a stored per-block
field (a wire-format change). Decide this together with `map/filter/update`, since all of
them need per-block metadata. Measured upper bound: see "Bigram pre-filter" in
[Benchmarks](./BENCHMARKS.md).
