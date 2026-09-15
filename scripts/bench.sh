#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release --features bench --bin vxg-bench -- "$@"
