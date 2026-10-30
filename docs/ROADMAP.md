# Roadmap

VXG stays a codec library. GPU backends, VIMG (image format) and VVID (video format) are out of
scope and not planned. The goal is to be the best general-purpose codec on structured and
semi-structured data, measured against LZ4 and Zstd on real files.

| Stage | Goal | Status |
| :--- | :--- | :--- |
| 1 | Core speed and ratio on real-world data | In progress |
| 2 | Compressed-domain data ops: `map`, `filter`, `update` without a full decode | `filter`, row access and numeric `filter_json_u64` exist |
| 3 | Correctness hardening (randomized stress, sanitizers, x86_64 coverage) | Ongoing; see [Testing](./TESTING.md) |

## Where VXG stands today (single thread, aarch64)

See [Benchmarks](./BENCHMARKS.md) for the numbers. In short:

- Structured data (NDJSON, CSV, logfmt): the clear winner on ratio, with decode faster than Zstd.
- Free text (logs, path lists): in Zstd-1's class on speed, 8-17% worse ratio than Zstd-1, and
  about half LZ4's decode speed.

"Unrivaled" means closing those two gaps without giving up the structured-data lead.

## Rules

- Benchmark on real files, not only synthetic data; synthetic wins did not carry over to free text.
- Compare against LZ4 and Zstd at equal block budget; keep rows where a competitor wins.
- Report ratio and speed together, single-thread and multi-thread separately.
- Interleaved A/B runs for any claim under about 10%; single runs are noise at that level.
- No optimization without a profile; no format change without a compatibility decision.

Stage 2 proposal: [Stage 2 design sketch](./STAGE2_DESIGN.md).
