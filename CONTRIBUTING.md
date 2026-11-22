<p align="center">
  <img src="assets/logos/vxg.svg" width="64" height="64" alt="VXG Logo" />
</p>

# Contributing to VXG

Thank you for your interest in contributing to **VXG** — whether it is a bug report, benchmark run on new hardware, documentation improvement, language binding, or core engine optimization!

## Workspace Overview

| Module | Directory | Scope |
| :--- | :--- | :--- |
| **VXG Core** | [`src/`](src/) | Adaptive compressed-domain codec, SQL, Arrow/DuckDB bridge, HTTP/1.1 Range |
| **VXG-KV** | [`crates/vxg-kv/`](crates/vxg-kv/) | Zero-dep stateless snapshot Redis (`RESP2/RESP3`), JSON/Geo, Streams/Groups, `VXG.SQL` |
| **VXT-Turbo** | [`crates/vxg-tensor/`](crates/vxg-tensor/) | E8 lattice LLM/VLM/MoE codec + Apple Metal & NVIDIA CUDA inference |
| **VIMG** | [`crates/vimg/`](crates/vimg/) | Zero-dep HTTP-Range hierarchical wavelet image & `VIMA1` animation codec |
| **WASM Packages** | [`packages/vxg-wasm/`](packages/vxg-wasm/), [`packages/vimg-wasm/`](packages/vimg-wasm/) | Browser, Node, Bun, and Edge Worker builds |
| **Docs & Playgrounds** | [`docs/`](docs/) | VitePress documentation site & live WASM studios |

## Engineering Principles

1. **Zero external dependencies in core crates** (`vxg`, `vxg-kv`, `vimg`). Keep binaries tiny, auditable, and `wasm32`-clean.
2. **Modular files (`<= 600` lines per `.rs` source file).** Split cleanly by responsibility instead of growing monolithic files.
3. **Zero compiler or Clippy warnings** under `-D warnings`.
4. **Honest, reproducible benchmarks.** Report both wins and where competitors lead. Never hand-edit `<!-- BEGIN GENERATED TABLE: ... -->` blocks — update artifacts under `reports/` and run `python3 scripts/render_tables.py --write`.
5. **Wire compatibility.** Never break existing frozen test vectors in `tests/vectors/`.

## Local Verification Checklist

Before opening a pull request, please run the relevant checks for the module(s) you touched:

```bash
# Core VXG (library + CLI + tests)
cargo clippy --no-deps --all-targets -- -D warnings
cargo test --release

# VXG-KV
cargo clippy -p vxg-kv --no-deps --all-targets -- -D warnings
cargo test --release -p vxg-kv

# VIMG
cargo clippy -p vimg --no-deps --all-targets -- -D warnings
cargo test --release -p vimg

# Documentation table drift check
python3 scripts/render_tables.py --check
```

## Good First Contributions

- Add or improve language FFI bindings in [`examples/ffi/`](examples/ffi/)
- Run benchmarks on ARM64 Linux, RISC-V, AVX-512, or Windows and share results via the **Benchmark Report** issue template
- Add custom `ObjectStore` adapters (S3, Cloudflare R2, GCS, Azure Blob) for `vxg-kv`
- Improve tutorials, interactive examples, or VitePress guides in [`docs/`](docs/)

## Code of Conduct

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
