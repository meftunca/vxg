# Validation ledger — delivery 0.1

## Critical status

**Rust compilation, Rust tests, native Metal/CUDA/Vulkan compilation and actual GPU execution
were NOT performed in the preparation environment.** No rustc/cargo or corresponding GPU
SDK/compiler/device was available. This package is an experimental source delivery with
partial executable validation, not a certified working multi-backend binary release.
Compilation, linkage, shader dialect, platform and device-memory-model issues may remain.

## Executed here

| Check | Actual result | What it establishes |
|---|---|---|
| C++20 build of exact shared `core.gpu`, WG=8 | PASS | Portable shared algorithm compiles in C++ emulation |
| Same shared algorithm, WG=64 | PASS | Same at actual planned group width |
| ASan + UBSan encode/decode tests | 110 cases per group width, clean final runs | No detected address/UB issue in those emulated executions |
| Independent bytewise Python oracle ↔ shared kernel | 46 directions/cases per width | Cross-implementation equality over selected boundaries and all-op golden cases |
| 16-bit control decomposition | All 65,536 words | Exhaustive finite-domain bit encoding identity |
| Symbolic periodicity/doubling cases | 1,548 | Finite source-index identity checks |
| Binary alphabet lengths 0..10 | 2,047 | Small exhaustive RAW/roundtrip boundary domain |
| Random/periodic/boundary inputs | 143 | Model roundtrip, size and search work checks |
| Golden vectors | 6 accepted, 4 intentionally invalid rejected | All operations plus framing/error paths in model |
| Truncated golden prefixes | 4,700 rejected | Strict framing in reference model |
| Shader generator freshness | Checked | Included MSL/CUDA/GLSL match common source |
| Installed C reference libraries | 25 codec/dataset pairs, 50 CSV rows | LZ4/Zstd C-library-only roundtrip and timing sanity |

An initial sanitizer pass found `memcpy` called with an empty vector's null data pointer
and zero size in the **C++ emulator helper**. The helper now skips that call for empty
inputs. Both emulation widths and cross-tests were rerun with `halt_on_error=1` after
that fix. Only final clean-run results above are claimed. ASan/UBSan are not race detectors.

Raw reports: `reports/gpu-emulator*.txt`, `reports/model-validation*.json`,
`reports/environment.json`, `reports/reference-c-baselines.*`.

## Source provided, not executed here

| Area | Status |
|---|---|
| `cargo build`, `cargo test`, Rust doctests | NOT RUN — no Rust toolchain |
| Rust SIMD SSE2/AVX2/AVX-512/NEON and hardware CRC | NOT EXECUTED |
| MSL compiler and Objective-C++/Metal bridge | NOT COMPILED/EXECUTED |
| NVCC CUDA kernel/host bridge | NOT COMPILED/EXECUTED |
| GLSL compiler → SPIR-V and Vulkan bridge | NOT COMPILED/EXECUTED |
| Actual GPU conformance / performance | NOT RUN |
| Rust VXG/LZ4/Zstd comparative benchmark | NOT RUN |
| Kani harnesses | NOT RUN |
| Miri / cargo-fuzz | NOT RUN |
| GitHub Actions workflows | Provided configuration, NOT observed runs |
| Rustfmt / Clippy | NOT RUN |

Do not substitute Python/model or C++ emulator success for any row above. In particular,
source-level proof arguments do not validate actual Rust unsafe code or platform FFI.

## Reproduce the executed checks

```sh
python3 scripts/generate_shaders.py --check
bash scripts/check_model.sh
```

`check_model.sh` builds the exact common kernel with clang++/ASan/UBSan at WG=8 and WG=64
and runs both model interop suites. Requires C++20 compiler, pthreads and Python 3.10+.
C++ binaries are not distributed in this source ZIP; the script rebuilds them locally.

To rerun the **reference-only** native C benchmark:

```sh
clang++ -std=c++17 -O3 verification/c_baselines.cpp -llz4 -lzstd -o /tmp/vxg-c-baselines
/tmp/vxg-c-baselines > reports/reference-c-baselines-local.csv
```

## Next acceptance gate on a development machine

Run Rust CPU tests first, then native backend compilation, then the actual-device
`gpu-check`, then corpus equality/fuzzing/sanitizers, then comparative benchmarks.
Keep failures visible; do not silently skip a failed device or fall back to CPU while
reporting GPU success. Preserve the machine/compiler/device details with every result.
