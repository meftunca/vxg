# Getting Started & CLI Reference

VXG (`vxg`) is a zero-dependency Rust library, C/C++ static/shared library (`libvxg`), WebAssembly module (`vxg.wasm`), and standalone command-line tool (`vxg`).

---

## 1. Building from Source

Requirements: Rust `1.85+` (Edition 2024). No external C libraries are needed for normal builds.

```sh
git clone https://github.com/meftunca/vxg.git
cd vxg

# Build the CLI binary and libraries (rlib + staticlib + cdylib)
cargo build --release

# Run the complete release test suite (unit, golden, property, v3/v4, CLI)
cargo test --release
```

Build outputs in `target/release/`:
- `vxg`: Standalone CLI binary
- `libvxg.a` / `vxg.lib`: Static C library
- `libvxg.so` / `libvxg.dylib` / `vxg.dll`: Dynamic shared library

---

## 2. CLI Reference (`vxg`)

`vxg` supports both explicit subcommands (`compress`, `decompress`, `verify`, `info`, `block`, `hardware`, `bench`) and familiar `zstd`/`lz4`-style short flags (`-c`, `-d`, `-t`, `-T0`, `-f`, `-k`).

### 2.1 Subcommands & Short Flags

| Command / Flag | Description |
| :--- | :--- |
| `vxg compress [FLAGS] [INPUT] [-o OUTPUT]` | Compress input file or `stdin` (`-`) to `.vxg` frame. Alias: `vxg -z` or default when flags are given. |
| `vxg decompress [FLAGS] [INPUT] [-o OUTPUT]` | Decompress `.vxg` frame to output file or `stdout` (`-`). Alias: `vxg -d`. |
| `vxg verify [FLAGS] <INPUT>` | Verify frame header, compact directory, block boundaries, and CRC32C without writing output. Alias: `vxg -t`. |
| `vxg info [--blocks] <INPUT>` | Print frame version (`v1`–`v4`), raw/compressed byte sizes, ratio, and optional per-block breakdown (`RAW`, `LZ`, `COLUMN`). |
| `vxg block <INPUT> <INDEX> [-o OUTPUT]` | $O(1)$ random-access extraction of a single block (`0..block_count`) without decoding other blocks. |
| `vxg query <INPUT> [QUERY_FLAGS]` | Compressed-domain structured record random access (`--count`, `--row`, `--range`) and predicate pushdown search (`--find`, `--filter`). Alias: `vxg q`. |
| `vxg hardware` | Print detected CPU architecture (`aarch64` / `x86_64`), active SIMD ISA (`Neon`, `Sse2`, `Avx2`, `Avx512`), and logical core count. |
| `vxg bench [SIZE_MIB]` | Run a quick built-in synthetic throughput check (`default: 64 MiB`). |

### 2.2 Common Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `-o <FILE>` | Derived / `-` | Output path (`-` writes to `stdout`). Defaults to `<INPUT>.vxg` on compress or strips `.vxg` on decompress. |
| `-c`, `--stdout` | `false` | Write output directly to `stdout`. |
| `-f`, `--force` | `false` | Overwrite existing output file. |
| `-k`, `--keep` | `true` | Keep input file after compression/decompression (always preserved). |
| `-T<N>`, `--threads <N>` | `1` | Number of worker threads (`-T0` or `--threads auto` uses all available logical CPU cores). |
| `-b <BYTES>`, `--block-size <BYTES>` | `65536` | Uncompressed block size in bytes (`4096..=1048576`, multiple of `64`). |
| `--no-checksum` | Checksum ON | Disable per-block CRC32C generation or verification. |
| `--limit <BYTES>` | `1073741824` | Maximum allowed decompressed output size (`1 GiB` default safety guard). |
| `--v3` / `--v2` | `v4` | Emit legacy VXG v3 or v2 wire format frames instead of default v4. |

### 2.3 Practical CLI Examples

```sh
# 1. Compress a 1 GB NDJSON file using all CPU cores
vxg compress -T0 events.ndjson -o events.ndjson.vxg

# 2. Inspect frame summary and block table
vxg info --blocks events.ndjson.vxg

# 3. Verify archive integrity (CRC32C + canonical structure)
vxg -t -T0 events.ndjson.vxg

# 4. Decompress to stdout and pipe to jq
vxg -d -c -T0 events.ndjson.vxg | head -n 5

# 5. Extract only block #42 in O(1) time
vxg block events.ndjson.vxg 42 -o block_42.ndjson

# 6. Compressed-domain structured record random access & pushdown search
vxg query events.ndjson.vxg --count
vxg query events.ndjson.vxg --row 50000
vxg query events.ndjson.vxg --range 50000:50010
vxg query events.ndjson.vxg --find "status=500"
vxg query events.ndjson.vxg --filter "ERROR" --limit 25
```
