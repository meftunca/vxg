#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Installation is explicit; this script does not install tools or download dependencies.
# cargo install --locked kani-verifier && cargo kani setup
cargo kani --lib
