#!/usr/bin/env bash
# Serial Q95 runner. Usage: scripts/run_qualification.sh reports/qualification/runs/P0-baseline
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out_dir="${1:?usage: scripts/run_qualification.sh OUTPUT_DIRECTORY}"
trials="${TRIALS:-7}"
seconds="${SECONDS_PER_SAMPLE:-0.2}"
samples="${SAMPLES_PER_PROCESS:-7}"
manifest="${MANIFEST:-$root/reports/qualification/manifest.json}"

cd "$root"
python3 scripts/generate_qualification_corpus.py --verify --manifest "$manifest"
cargo build --release --features bench --bin vxg-bench
target_dir="$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
binary="$target_dir/release/vxg-bench"

mkdir -p "$out_dir"
cp "$manifest" "$out_dir/manifest.json"
{
  printf 'binary=%s\n' "$binary"
  shasum -a 256 "$binary"
  printf 'trials=%s\nsamples_per_process=%s\nseconds_per_sample=%s\nthreads=1\nblock_size=65536\n' "$trials" "$samples" "$seconds"
  uname -a
} > "$out_dir/environment.txt"

trial=1
while [ "$trial" -le "$trials" ]; do
  while IFS= read -r input; do
    name="$(basename "$input" .ndjson)"
    process_dir="$out_dir/$name/process-$(printf '%02d' "$trial")"
    mkdir -p "$process_dir"
    "$binary" --file "$input" --threads 1 --block-size 65536 --seconds "$seconds" --samples "$samples" \
      --csv "$process_dir/summary.csv" --samples-csv "$process_dir/samples.csv" \
      > "$process_dir/console.log" 2>&1
  done < <(python3 scripts/generate_qualification_corpus.py --verify --list --manifest "$manifest")
  trial=$((trial + 1))
done

printf 'Completed Q95 serial baseline: %s trials x 8 files.\n' "$trials"
