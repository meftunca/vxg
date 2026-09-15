# Safety and security boundary

VXG is experimental. Do not use it as the only copy of valuable data or expose it as a
hardened public decompression service before compilation, fuzzing and platform validation.

## CPU

The public parser checks header/version, allowed flags, directory arithmetic, block sizes,
canonical layout, output limit, command progress, extensions, stream ranges, distances and
exact output consumption. `decompress` checks the requested output limit before allocating.
`Frame::decode_into` can partially modify its destination before an error; callers must
discard that output. The default full-output limit is 1 GiB, not an application policy for
every deployment. Ordinary Vec allocation can still fail/abort under memory exhaustion.

SIMD entry points are private unsafe helpers behind runtime feature detection. Full-vector
loads/stores only execute when a whole vector is inside both valid slices; tails are exact.
No external 64-byte padding/wild-copy contract. Copying overlap uses split slices, seed
snapshots or completed-prefix doubling, not an aliased `copy_nonoverlapping` call.
These are design invariants, not a claim that the uncompiled implementation passed Miri.

## GPU/FFI

Native buffers internally include padding for packed u32 reads. GPU decode is a trusted
kernel, not a standalone parser of arbitrary untrusted bytes. Rust validates all command
streams before dispatch. The C ABI is internal, not a safe public interface; bypassing
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
