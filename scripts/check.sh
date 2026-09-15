#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v cargo >/dev/null || { echo 'Rust >=1.89 and Cargo are required.' >&2; exit 1; }
python3 scripts/generate_shaders.py --check
cargo test --release
cargo run --release --example roundtrip
# Optional native SDK checks are not silently skipped; choose explicitly.
# bash scripts/gpu_check.sh metal|vulkan|cuda
