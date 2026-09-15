#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:-}"
case "$backend" in
  metal|vulkan|cuda) ;;
  spirv) backend=vulkan ;;
  *) echo 'usage: bash scripts/gpu_check.sh metal|vulkan|cuda' >&2; exit 2 ;;
esac
python3 scripts/generate_shaders.py --check
cargo test --release --features "$backend"
cargo run --release --features "$backend" --bin vxg -- gpu-check "$backend"
