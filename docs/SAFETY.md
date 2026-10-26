# Safety and security boundary

VXG is experimental. Do not use it as the only copy of valuable data or expose it as a
hardened public decompression service before compilation, fuzzing and platform validation.

## CPU

The public parser checks header/version (`v1`–`v4`), allowed flags, directory arithmetic, block sizes,
canonical layout, output limit, command progress, extensions, stream ranges, distances and
exact output consumption. `decompress` checks the requested output limit before allocating;
`decompress_into` and `extract_block_into` write directly into caller-provided buffers after
verifying capacity and structural bounds (including multi-threaded disjoint block slices).
`Frame::decode_into` / `decompress_into` can partially modify the destination before an error; callers must
discard that output. The default full-output limit is 1 GiB, not an application policy for
every deployment. Ordinary Vec allocation can still fail/abort under memory exhaustion.

SIMD entry points are private unsafe helpers behind runtime feature detection (`NEON` on `aarch64`, `SSE2`/`AVX2`/`AVX-512` on `x86_64`). Full-vector
loads/stores only execute when a whole vector is inside valid bounds or explicitly reserved scratch arenas; tails are exact.
Copying overlap uses split slices, seed snapshots or completed-prefix doubling, not an aliased `copy_nonoverlapping` call.

## Public C FFI & GPU Bridge

The public C ABI declared in [`include/vxg.h`](../include/vxg.h) and implemented in [`src/ffi.rs`](../src/ffi.rs) wraps every entry point (`vxg_compress_into`, `vxg_decompress_into`, `vxg_extract_block_into`, etc.) in `std::panic::catch_unwind`, validates null pointers and slice bounds before constructing Rust slices, and returns negative `VXG_ERR_*` status codes on invalid inputs or panics.

For the internal GPU host bridges (`native/`), native buffers internally include padding for packed `u32` reads. GPU decode is a trusted
kernel, not a standalone parser of arbitrary untrusted bytes. Rust validates all command
streams before dispatch; bypassing
Rust checks invalidates the kernel assumptions and can cause out-of-bounds accesses or hangs.
All native calls are synchronous before borrowed host buffers expire. CUDA failure cleanup
synchronizes the stream before returning. Contexts are poisoned on reported native errors.
Driver loss, catastrophic device failure, Objective-C exceptions and native allocation
failures need further platform fault-injection testing; C++ exception handling is not a
blanket guarantee against all platform exceptions.

The emulator CLI decode mode is also trusted-only: it is used only after model validation.
Do not fuzz untrusted bytes directly into it as though it were a hardened file parser.

## Integrity and persistence

CRC32C detects many accidental corruptions but does not authenticate content or protect
against deliberate collisions. Use an authenticated container/protocol where required.
CLI output is created exclusively by default; `--force` explicitly allows replacement.
Writes are not transactional/fsync-durable and interruption can leave a partial file.
The CLI reads the entire input file; streaming, atomic rename and durable file semantics
are separate future features.

## Suggested validation commands

```sh
cargo test --release
cargo +nightly miri test --test golden
cargo install cargo-fuzz
cargo +nightly fuzz run decode -- -max_total_time=300
cargo +nightly fuzz run roundtrip -- -max_total_time=300
```

Run fuzz commands from the project root; `fuzz/Cargo.toml` is the conventional separate
cargo-fuzz workspace. Miri routes the codec to scalar/software paths, so it does not
validate SIMD intrinsics. Extend corpus and resource-failure testing independently.
The commands above are supplied instructions, not a record of successful execution here.
