# Turbo accepted-state recovery, 2026-11-04

Scoped fix to `generate_greedy_fast` and its Metal FFI. Quantization, sketch/default options, archive and arithmetic quality are unchanged. Existing large dirty worktree retained; no commit or cleanup performed. Baseline source/binary/artifact identities remain in the evidence described by `reports/turbo-baseline-2026-11-04.md`.

## Returned-state contract

`first_tok` is already predicted and is emitted without consuming it. Each subsequent GPU step consumes the preceding emitted token and predicts another. `pos` counts consumed inputs, rather than submitted speculative work:

- Zero cap, first-token stop, cap1, and cancellation on the first callback consume no new inputs.
- A normal cap or callback cancellation returns the delivered tokens, including the token whose callback returns false. The last delivered token is pending: consumed count is emitted count minus one.
- EOS is withheld. Predicting it consumes the last visible token, so consumed count equals emitted count. First-token EOS consumes nothing.

The caller must honor the pending-token convention when resuming; this matches ordinary sequential decoding. A callback panic is rethrown after draining queued work and attempting recovery to the prefix preceding the last delivered token.

## Implementation

Generation takes one queue-ordered recurrent-state checkpoint into the existing per-layer `conv_bak`/`ssm_bak` buffers. The async path and replay use `save_backup=0`, and never select backup state, so those buffers remain owned by this invocation. Required backup allocations are now checked. No additional model/state allocation and no per-token checkpoint is added.

The two-deep eight-step pipeline tracks submitted and accepted consumed counts separately. Every remaining command is waited and its retained handle released before recovery or presence-penalty reset, including enqueue/wait errors and callback unwind. When speculation overshoots, recurrent state is restored and accepted inputs replayed one at a time with the same full Turbo mode2/3 trunk. A capped completed pipeline requires no replay.

GQA cache prefix rows are preserved. Attention bounds its reads by the supplied token position: the batch shader uses `seq_len=pos_base+m+1` and `t<seq_len`; the fused M=1 path reads through the current position. Speculative suffix rows are invisible after restoring `pos`, and continuation overwrites them. This is a logical truncation of KV, not a physical zeroing of unused cache rows.

A directly related pipeline race was also fixed: KV growth copies shared buffers on the CPU. When growth is needed, a queue fence now completes older pending writes before the copy/replacement. The normal no-growth path adds no fence.

## Verification

All commands use release + `--features metal`, isolated `/private/tmp/vxt-stop-target`. The first sandbox test failed explicitly because Metal was unavailable; it did not silently pass on CPU. Real device tests then ran with GPU access:

1. `turbo_stop_state_matches_sequential_gpu_replay`: passed, 1.34s. Sketch and exact paths; caps 0/1/8/9/10/17; callback cancellation 1/8/9/16/17/257 including ring wrap; stop IDs chosen from generated traces; nonzero presence penalty; exact continuation logits against clean full-step GPU replay and equal resumed fast token sequences. A deliberate callback panic at token 9 is caught externally and continuation remains exactly equal. A prompt of 810 tokens forces the second queued burst across the initial 1024 row cache's 200 row growth reserve, followed by cancellation 17 and exact continuation comparison.
2. Existing `hybrid_qwen35_turbo_pipelined_gpu_matches_cpu_and_batched_decode`: passed. This is a separate pre-existing numerical parity/batched generation regression; CPU tolerance is not substituted for the exact GPU recovery comparisons.
3. Ignored artifact test `turbo_real_model_stop_and_resume`, explicitly enabled with `VXG_TURBO_STOP_MODEL`: passed, 8.18s. Uses `/private/tmp/vxg-real-qualification-20261104/turbo-baseline/Qwen3.5-2B-turbo.vxt` (632712974 bytes). Original prompt `What is 17 times 23? Show a short calculation.` reproduces 27 prompt tokens and 33 emitted tokens. Returned position is now 60, versus the baseline's 75. Three continuation full-logit vectors equal clean GPU replay exactly, and subsequent 20 token fast generation matches exactly.

The answer still says 411. This fix does not correct model quality.

One 128 token capped diagnostic on the same production artifact, sketch enabled, zero presence penalty, no stop IDs: 0.429321s, 127 processed tokens / 295.82 TPS, returned position 154 (=27+127). This includes checkpoint overhead. The baseline's roughly 297 TPS used another prompt/run; these are single-run host-contended diagnostics, not a controlled performance comparison or guarantee.

## Retained evidence and limits

Final logs, exact changed-source snapshots and the test executable are retained in `/private/tmp/vxg-real-qualification-20261104/turbo-stop-20261104/`. `source-binary-sha256.json` records SHA256 identities; final test executable hash is `fcbc75ffc6f72c5ad4e8c9fd18f41bcc7995ed7157d767dbb9dcbf5416000f3b`. Baseline evidence/artifact was neither replaced nor removed.

Commands:

```sh
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --test llm_gpu_cpu_parity turbo_stop_state_matches_sequential_gpu_replay -- --nocapture
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --test llm_gpu_cpu_parity hybrid_qwen35_turbo_pipelined_gpu_matches_cpu_and_batched_decode -- --nocapture
VXG_TURBO_STOP_MODEL=/private/tmp/vxg-real-qualification-20261104/turbo-baseline/Qwen3.5-2B-turbo.vxt cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --test llm_gpu_cpu_parity turbo_real_model_stop_and_resume -- --ignored --nocapture
```

No GPU failure/allocator failure injection was performed. Error cleanup and recovery attempts are implemented, but a device failure during restore/replay can still return an error with state only partially recovered; it is not qualified for continuation after such a failure. Callback panic preserves the panic even if recovery fails. EOS test IDs can repeat, so a requested later trace index stops at that ID's first occurrence; callback tests independently cover burst/ring boundaries. Broad model quality, long-context qualification, all models/devices, and statistically controlled performance remain open.
