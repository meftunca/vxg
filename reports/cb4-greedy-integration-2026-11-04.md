# CB4 greedy high-level integration, 2026-11-04

Explicit opt-in only. Existing ordinary and Turbo decoding defaults remain unchanged. No commit, deletion, new compression, weight conversion, or full-model evaluation was performed by this integration task. Existing dirty files were preserved.

## Implementation

`src/llm.rs` adds `enable_cb4_greedy()` requiring a four-bit Codebook embedding, uploaded embedding, and generic Metal runner. It passes the exact CPU embedding centroids into the native runner. `generate_greedy_cb4()` performs one native greedy step per consumed token, increments position only after success, and exposes `(token, &BpeTokenizer)` callbacks for streaming without copying the tokenizer. No speculation or restoration occurs.

The supplied first prediction is unconsumed. Cap zero or first-token EOS performs no device steps. Stop IDs are not emitted or consumed. On cap, callback cancellation, or callback panic the last visible token remains pending; earlier emitted tokens have been consumed. When a generated stop ID is reached, all preceding visible tokens have been consumed. A native step failure may already have mutated device state: a private failure flag then rejects all further prefill, forward, and generation entry points on that state; callers must discard it and create a new state. The public position is not advanced on the failing call and is not a claim of recoverable device state.

`src/cli_llm.rs` adds `--cb4-greedy`, requiring GPU, temperature exactly zero, explicit `--presence-penalty 0`, and no reference/exact-head alternate flags. Unsupported models fail rather than falling back. Initial nonfinite logits fail before first-token sampling. The CB4 route uses the same streaming text decode/flush behavior as ordinary decoding. Both CLI help strings list the opt-in prerequisites.

`src/bin/generation-quality.rs` adds explicit `cb4-greedy` stage, preserves the original cap 384, prompts, retained A source-row comparisons, strict final parser, zero temperature/presence penalty, and backend check. It labels this route `cb4-greedy-full-head`; historical `cb4` stays `generic-forward-step`. No forty-item run occurred here.

## Commands and observed output

Working directory: `/Users/mapletechnologies/Desktop/big_projects/vxg`.

Initial release compilation and sandboxed execution:

```sh
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --test llm_gpu_cpu_parity cb4_greedy_stop_state_matches_sequential_gpu_replay -- --nocapture
```

Observed: release build completed in 48.28 seconds; the test failed explicitly with `Backend("no Metal GPU device available")`. This was not counted as a pass or silently skipped.

Latest focused test compilation:

```sh
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --test llm_gpu_cpu_parity cb4_greedy --no-run
```

Observed: release build completed in 57.83 seconds and produced the binary below.

Actual Metal execution, outside the device-restricted sandbox:

```sh
/private/tmp/vxt-stop-target/release/deps/llm_gpu_cpu_parity-1585c1ccc663a83c cb4_greedy --nocapture
```

Captured tool output, preserved here verbatim apart from tool transport metadata:

```text
running 2 tests
test cb4_greedy_nonfinite_device_error_invalidates_state ... ok

thread 'cb4_greedy_stop_state_matches_sequential_gpu_replay' (28139061) panicked at crates/vxg-tensor/tests/llm_gpu_cpu_parity.rs:608:54:
deliberate callback panic
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test cb4_greedy_stop_state_matches_sequential_gpu_replay ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 3.05s
```

The deliberate callback panic is caught by the test and followed by an exact continuation comparison. These macOS device tests intentionally require Metal and fail if unavailable; they do not silently substitute CPU.

Coverage: synthetic hybrid Qwen3.5 CB4 fixture; exact generated tokens against ordinary generic GPU logits; caps 0/1/8/17; cancellation at 1/8/17; first EOS; each trace token used as a stop ID with first-occurrence semantics; exact full-logit continuations; callback panic at token eight; CPU enable rejection; production NaN output-head failure followed by rejected forward, prefill, and greedy generation on the invalid state.

## Identity and scope of evidence

Test executable SHA256:

```text
ce6e83bddf7fa9411891eb412040c436fa49656960022bf3d9dd094985b6c6a2  /private/tmp/vxt-stop-target/release/deps/llm_gpu_cpu_parity-1585c1ccc663a83c
```

Source SHA256 recorded after focused tests and before qualification (paths relative to repository):

```text
f72450c60a8cb3770271ec359e285a8283ef2e441375feb92bed9c938f5a088f  crates/vxg-tensor/src/llm.rs
edcff93812885796e83621de8a5487edf198dd70d8f36c5513b000e0abaf0a68  crates/vxg-tensor/src/cli_llm.rs
5a1213b68d624e7a468c6708da8c28121c7c4d5e26417dab6d83a346b4d8f4ec  crates/vxg-tensor/src/bin/generation-quality.rs
e5f59135cd1382fda266c47bdfd9fdf0d628ec1d25d18de18b2d6a7d70980a8f  crates/vxg-tensor/src/bin/vxg-tensor.rs
9720f82b1f4bd22d508db8f7ccc5b5e5cda194f5b12a0417c1f00b3ea8189d06  crates/vxg-tensor/tests/llm_gpu_cpu_parity.rs
```

One final defensive guard rejecting an invalid CB4 state at the existing Turbo generation entry point was added after rustc started the final focused build. Therefore the focused binary is not asserted to match the final `llm.rs` hash byte for byte. The CB4 implementation and the tests above did not change after capture by that build. The qualification owner was notified to compile final source before subsequent runs; that final compile and source/binary identity must be established in its own evidence. No code edits or further builds were performed by this task after handing over the build/GPU slot.

Logs here reproduce tool-captured outputs; this task did not redirect the original runs into standalone log files or retain an immutable executable copy. The executable path can be overwritten by subsequent Cargo builds; its recorded hash identifies the tested instance. Native primitive tests, production-artifact paired tokens/state checks, final CLI/quality builds, retained forty-item quality, and controlled performance belong to separate qualification evidence. These focused synthetic passes alone do not establish model quality, speed, all-model support, allocator-failure recovery, or long-context qualification.
