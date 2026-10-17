# CPU energy comparison

The fourth performance axis is **joules per decimal GB of original input**, reported
separately for encode and safe decode. Lower is better. Never infer power from codec
speed, nominal chip TDP, or Activity Monitor's relative Energy Impact score.

`vxg-energy-workload` runs one codec/operation continuously with the same one-worker,
64 KiB independent-block, checksum-off and system-allocator configuration as the CPU
target. It verifies bytes before/after timing, emits completed-call counters, includes
encode allocation and includes header parsing in VXG's safe decode. C baselines retain
the prepared block metadata and size-accounting envelope used by `vxg-bench`; this
framing/allocation asymmetry remains visible in each row's profile label.

`scripts/energy_bench.py` shuffles codec/operation order within each trial, preserves
the input/binary SHA-256 and raw logs, and samples CPU/GPU/ANE/thermal information using
the installed macOS `powermetrics`. Only the sampling subprocess runs through sudo.
The benchmark, Python runner and output files remain owned by the normal user.

## Run on this Mac

From the project directory:

```sh
cargo build --release --features bench --bin vxg-energy-workload
python3 scripts/energy_bench.py --authorize \
  --output reports/qualification/experiments/P8-energy-r2
```

The default is JSON-16, all six codec configurations, encode and decode, three shuffled
trials with 15 seconds of sustained work each. Allow roughly 15 minutes including idle
windows and sampling margins. Each phase prints progress. `--input FILE` selects another
corpus; every frozen member's hash/size is checked when selected. Use a new output directory
for each run; the runner refuses to overwrite an existing result. Default results are
directional; stronger claims need more trials and all relevant real/frozen workloads.

`--authorize` requests the macOS password locally via sudo, before sampling. Never send
the password to a chat. This setting does not change sudoers, system security settings,
power policy or other applications. Keep background activity comparable across trials.

## Accounting and limits

For each retained power sample, CPU watts = explicitly labelled CPU mW / 1000 and
joules = watts * **actual reported sample duration**. Sum those joules, then divide by
the raw GB processed during the same retained intervals. Work at sample boundaries is
estimated by linear interpolation between approximately 100 ms completed-call counters.

The text timestamps have only whole-second precision. The importer uses the midpoint
of that second, records +/-0.5 s uncertainty, excludes edge windows with an additional
guard and avoids overlapping sample intervals. It rejects insufficient coverage and
wall-clock jumps. These are sustained-load estimates, not precise per-request energy.
Numeric UTC offsets printed by recent macOS (such as `+0300`) are honored; older
headers without an offset retain local-time semantics.
The raw logs permit later replacement with an OS-verified higher-resolution sampler.

The primary column is `cpu_rail_J_per_GB`: estimated system CPU-rail energy including
background CPU load. It is **not** process-attributed energy, complete SoC energy,
DRAM/SSD energy, battery drain or wall-plug consumption. The same-device comparison
is useful; a whole-system claim needs an external meter. Apple's installed manual
also cautions that power estimates are unsuitable for comparison between devices.

Idle power is sampled immediately before and after each phase. The report preserves
both raw CPU energy and a separately labelled idle-subtracted estimate. Negative
increments are retained, with `incremental_usable=false`; an idle drift above 20%
also invalidates that incremental interpretation. Summary energy rankings use raw
CPU-rail J/GB. GPU/ANE/thermal observations remain in raw logs for interference checks.

`energy_vs_lz4_a1` is VXG/configuration J/GB divided by the corresponding LZ4-a1
operation's J/GB; below 1 means less sampled CPU-rail energy. Encode/decode always have
separate rows. An encode-once/decode-N application can later weight them by its actual
operation counts. This is not yet an energy acceptance threshold for the CPU target.

## Validation and current status

The initial run in `P8-energy-workload-smoke-r1` executed all 12 CPU paths and byte
checks without power sampling. Its energy columns are null, and its 0.2-second
workloads are plumbing checks, not a performance or energy comparison. Six importer
tests cover units/durations, byte/time normalization, missing counters, negative idle
adjustment, clock jumps, null propagation and a real macOS timezone-header regression.
The suite passes in the default timezone, UTC and America/Los_Angeles.

The user's terminal successfully sampled power in `P8-energy-r1`, but the importer
failed on the header's `+0300` suffix before any codec phase. The parser now accepts
that header and was verified against the saved full preflight log. That failed run is
preserved; retry into `P8-energy-r2` without rebuilding the unchanged Rust binary.
The agent's separate sudo session still requires a password, so the retry must run in
the user's authenticated terminal. No measured codec energy ranking exists yet; the
single preflight reading is not a codec benchmark.

References: [Apple energy concepts](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/EnergyGuide-iOS/FundamentalConcepts.html),
and the host's `/usr/share/man/man1/powermetrics.1` plus `powermetrics --help` for current
sampler flags, mW labels and energy-estimate scope.
