# Getting Started & CLI Reference

VXG (`vxg`) is a zero-dependency Rust library, C/C++ static/shared library (`libvxg`), WebAssembly module (`vxg.wasm`), and standalone command-line tool (`vxg`) with built-in compressed-domain query, analytics, mutation, and column projection.

---

## 1. Building from Source

Requirements: Rust `1.89+` (Edition 2021). No external C libraries are needed for normal builds.

```sh
git clone https://github.com/meftunca/vxg.git
cd vxg

# Build the CLI binary and libraries (rlib + staticlib + cdylib)
cargo build --release

# Run the complete release test suite (unit, golden, property, v3/v4, analytics, projection, CLI)
cargo test --release
```

Build outputs in `target/release/`:
- `vxg`: Standalone CLI binary
- `libvxg.a` / `vxg.lib`: Static C library
- `libvxg.so` / `libvxg.dylib` / `vxg.dll`: Dynamic shared library

---

## 2. CLI Reference (`vxg`)

`vxg` operates on positional `INPUT` / `OUTPUT` paths (use `-` for `stdin` or `stdout`) and supports multi-threaded execution via `--threads N|auto` or `-t N|auto`.

### 2.1 Subcommands

| Command | Description |
| :--- | :--- |
| `vxg compress` (`c`) `INPUT OUTPUT [FLAGS]` | Compress `INPUT` file or `stdin` (`-`) into a Format v4 `.vxg` frame. |
| `vxg decompress` (`d`) `INPUT OUTPUT [FLAGS]` | Decompress `.vxg` frame (single or concatenated stream) to `OUTPUT` or `stdout` (`-`). |
| `vxg verify` (`t`) `FRAME [FLAGS]` | Verify frame header, compact directory, block boundaries, and CRC32C checksums without writing output. |
| `vxg info` (`list`) `FRAME [--summary]` | Print frame version (`v1`–`v4`), raw/encoded byte sizes, ratio, and block-type breakdown (`RAW`, `LZ`, `COLUMN`). |
| `vxg block` `FRAME INDEX [OUTPUT] [-f]` | $O(1)$ random-access extraction of a single block (`0..block_count`) without decoding other blocks. |
| `vxg query` (`q`) `FRAME [QUERY_FLAGS]` | Compressed-domain row count, row/range extraction, substring search, compound `--where` filter, and `--select` column projection. |
| `vxg agg` `FRAME [AGG_FLAGS]` | Compressed-domain scalar aggregation (`COUNT`, `SUM`, `AVG`, `MIN`, `MAX`) and `GROUP BY` (`--group-by COL`). |
| `vxg project` `INPUT OUTPUT --cols C1,C2,...` | Compressed-domain column projection into a new `.vxg` frame (zero-reencode column splicing when unfiltered). |
| `vxg filter-frame` `INPUT OUTPUT --where EXPR` | Produce a new `.vxg` frame containing only rows matching compound `--where EXPR` (splices 100%-matching blocks verbatim). |
| `vxg update` `INPUT OUTPUT --where SPEC --col COL ...` | Conditional single-column mutation (`--set`, `--set-raw`, `--upper`, `--lower`, `--redact`, `--add`) without full decode/re-encode. |
| `vxg map` `INPUT OUTPUT --col COL ...` | Whole-column transformation (`O(\|D\|)` dictionary splice or single-slot numeric rewrite) into a new `.vxg` frame. |
| `vxg hardware` | Print detected OS/arch, active SIMD ISA (`Neon`, `Sse2`, `Avx2`, `Avx512`), logical core count, and compiled GPU backends. |

### 2.2 Common Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `-` (as `INPUT` / `OUTPUT`) | — | Read from `stdin` or write to `stdout`. |
| `-t <N\|auto>`, `--threads <N\|auto>` | `1` | Worker thread count (`auto` or `0` uses all available logical CPU cores). |
| `-b <BYTES>`, `--block-size <BYTES>` | `65536` | Uncompressed block size in bytes (`4096..=1048576`, multiple of `64`). |
| `--stream` | `false` | Bounded-memory streaming mode (`StreamEncoder` / `StreamDecoder`) for arbitrarily large pipes/files. |
| `--no-checksum` | Checksum ON | Disable per-block CRC32C generation or verification. |
| `--max-output <BYTES>` | `1073741824` | Maximum allowed decompressed output size (`1 GiB` default safety guard). |
| `-f`, `--force` | `false` | Overwrite existing output file (by default `vxg` refuses to clobber existing files). |
| `-q`, `--quiet` | `false` | Suppress diagnostic timing/ratio output on `stderr`. |

### 2.3 Column Addressing & Compound `--where` Syntax

- **Column targets (`COL`):**
  - JSON keys: `service` or `json:service`
  - Delimited columns (`0`-indexed): `csv:2`, `tsv:0`, `psv:4`
- **Atomic predicates & compound boolean expressions (`--where EXPR`):**
  - Numeric & string comparisons: `status==500`, `status!=200`, `latency_ms>=150`, `bytes=1000..5000`, `service=="auth-gateway"`
  - String matchers: `service=prefix:auth`, `path=suffix:.json`, `msg=contains:timeout`, `contains:ERROR` (whole-row substring)
  - Boolean combinators & grouping: `&&` / `AND`, `||` / `OR`, `!` / `NOT`, parentheses `(...)`

### 2.4 Practical CLI Examples

```sh
# 1. Compress and verify using all CPU cores
vxg compress events.ndjson events.vxg -t auto -f
vxg verify events.vxg -t auto
vxg info events.vxg --summary

# 2. O(1) random-access block & row extraction
vxg block events.vxg 0 block0.ndjson -f
vxg query events.vxg --count
vxg query events.vxg --row 42
vxg query events.vxg --range 100:110

# 3. Compound boolean WHERE + SELECT column projection (skips unselected columns!)
vxg query events.vxg \
  --select ts,service,status,latency_ms \
  --where 'status==500 && (service==fraud-engine || service==auth-gateway) && latency_ms>=150' \
  --limit 20 -t auto

# 4. Zero-reencode VXG -> VXG column splicing + chained GROUP BY analytics
vxg project events.vxg slim.vxg --cols service,status,bytes,latency_ms -t auto -f
vxg agg slim.vxg --group-by service --col bytes --where 'status==500' --json -t auto

# 5. Compressed-domain conditional column update & whole-column map
vxg update events.vxg patched.vxg --where status==500 --col level --set CRITICAL -f
vxg map events.vxg anonymized.vxg --col user --redact -f

# 6. Unix streaming pipeline via stdin / stdout ('-')
cat events.ndjson | vxg c - - --stream -t auto -q | vxg d - - --stream -t auto -q > restored.ndjson
```
