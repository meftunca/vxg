#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CXX="${CXX:-clang++}"
mkdir -p reports target/model
python3 scripts/generate_shaders.py --check
for width in 8 64; do
  executable="target/model/gpu-emulator-wg${width}"
  "$CXX" -std=c++20 -O1 -g -DEMU_WG="$width" \
    -fsanitize=address,undefined -fno-omit-frame-pointer -pthread \
    verification/gpu_emulator.cpp -o "$executable"
  ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$executable" | tee "reports/gpu-emulator-wg${width}-local.txt"
  ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    python3 verification/check_model.py --emulator "$executable" \
      --report "reports/model-validation-wg${width}-local.json"
done
