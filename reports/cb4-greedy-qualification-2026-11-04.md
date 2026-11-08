# Opt-in CB4 greedy qualification, 2026-11-04

**Resumed after explicit user authorization.** The earlier pause remains recorded below and in `paused.json`. Retained equivalence and paired results have now been reviewed; a bounded frozen-binary profile/control was added. These diagnostic gates do not establish a robust speedup or justify default promotion.

Bounded equivalence and diagnostic speed qualification of the opt-in CB4 greedy route. The archive, codec and weights remain the existing CB4 artifact; this change targets GPU execution and full-head greedy selection. The ordinary route remains the default. No model repack, calibration, prompt/scorer change, commit or deletion.

Evidence root: `/private/tmp/vxg-real-qualification-20261104/cb4-greedy-qualification-20261104`.

## Fixed model and protocol

Artifact `/private/tmp/vxg-real-qualification-20261104/cb4-baseline-20261104/Qwen3.5-2B-cb4.vxt`: 1,036,621,966 bytes; SHA256 `d4532feff049de11f60452fd99cc545b576e0d146cdf1e2dee989945b4171196`; original source/archive ratio 3.64232x. No claim that execution improvements change compression.

The same forty known exploratory questions/options/keys and strict free-generation protocol are retained. Suite SHA256 `0caadde265c5feb0a528535459a2b660d88a74f5c4d80176e62126710687aa44`; inherited protocol SHA256 `1e92bda984c96ff51080bc62e3db9f9eb1a1e179d6950b1585836b3184527593`. Cap 384, temperature zero, presence penalty zero, unchanged no-thinking chat template. Exactly one `Final answer:` marker with a trimmed suffix of one uppercase A/B/C/D; any cap hit is truncated regardless of apparent answer. Source answers and historical ordinary-CB4 checkpoints are reused without rerunning them.

Candidate results are compared itemwise to historical CB4 across every emitted token ID, decoded text, prompt IDs, stop IDs/reason, returned state position/processed count, prediction/status/gold, configuration, cap and generation options. This establishes sequence/termination/scoring equivalence only on this retained diagnostic, not broad numerical or model quality equivalence.

## Run sequence and results

The native primitive test passed on actual Metal (one test): exact CB4 embedding with centroids/scales/outliers/tails, greedy ties and failure cases. High-level actual-Metal integration tests passed (two tests), covering exact ordinary/opt-in greedy tokens, cap/EOS/callback cancellation, continuation, callback-panic continuation validity and nonfinite-head state invalidation. Their retained reports are copied into this evidence root. The integration test binary preceded a final unrelated Turbo invalid-state guard; final release build below captures that guard. Two final generation-quality route/parser tests and the final release CLI/quality build passed. No broad regression suite is claimed.

After those checks, an actual-Metal eight-token ordinary/opt-in smoke produced identical text, emitted count eight and returned position 35. Both routes used `Apple Metal GPU (Apple M2 Max, 1-CmdBuf Fused)`; the opt-in route identified itself as `cb4-greedy-full-head`. CPU fallback was rejected.

The candidate then ran the forty items once. **All 5,561 emitted token IDs match the historical ordinary CB4 sequence exactly**, as do decoded text, stop reason/IDs, returned position/processed count and scoring/options on all forty items. `quality-equivalence.json` has an empty differences list. Independent `audit.py` also checked frozen source prompts/options and strict-label scoring. No answer rerun, rescoring relaxation or tuning followed the outputs.

| Category | Completed correct | Completed wrong | Format unscorable | Truncated |
|---|---:|---:|---:|---:|
| Math |10|0|0|0|
| Code |9|1|0|0|
| Turkish |5|1|2|2|
| English |8|0|2|0|
| **Total** |**32**|**2**|**4**|**2**|

Source-correct retention remains 32/36, with the same one wrong, one truncated and two format losses. The four pre-existing source-correct losses remain `code-09`, `tr-01`, `tr-09`, `en-06`. This route preserves the CB4 result on this diagnostic; it does not repair existing quantized-model failures or establish general quality. The forty-item stage took 147.04 seconds including load/prefill/checkpoint writes; this is not a throughput comparison against the older run.

## Paired bandwidth diagnostic

The user explicitly paused the goal before final performance review/reporting. On handling the pause, the exact owned harness/model process scan found no remaining process: the already-running sequential harness had exited successfully and written the two warmup and three pair logs before the pause handler ran. No new operation was launched after the pause instruction. Raw results remain in the evidence directory; **no final performance result, speedup conclusion or completion claim is made here**.

Remaining gates after explicit resumption: review the retained paired measurements and resource/host controls, verify final retained identities and focused-test provenance against the compiled candidate, and complete the performance report. No additional measurements are authorized by this pause status. `paused.json` records the process check and pause disposition.


## Resumed retained-result review and frozen-binary diagnostic

Resumption reviewed the existing data without repeating the forty-item run. SHA256 verification passed for the model, both retained executables, and every file in `source-sha256.json` against its retained `source/` snapshot. Retained `vxg-tensor` is `fb1d0e222d62d99f072a43bd7c85f9735a17e6b92fe2a5b822ef5c52beef9f96`; retained `generation-quality` is `1381602dcb145687f4a8c8cc4eb77f7d0cbeae7b81127b7ea06805611d6bc824`. This validates retained identities, not a byte-for-byte claim about the concurrently changing working tree or a reproducible rebuild. Focused-test provenance retains the pre-final-Turbo-guard limitation described above; no fresh broad regression was run.

The completed sequential measurements exclude both warmups. TPS uses seven/127 timed decode transitions for eight/128 emitted tokens, respectively: the first token comes from prefill.

| Pair/order | Ordinary TPS | Opt-in TPS | Paired ordinary/opt-in decode-time ratio |
|---|---:|---:|---:|
| 1 / AB | 62.750 | 63.630 | 1.0140 |
| 2 / BA | 66.663 | 64.943 | 0.9742 |
| 3 / AB | 52.943 | 66.907 | 1.2638 |

Median paired ratio is **1.0140**, with a wide 0.9742–1.2638 range. Median ordinary/opt-in TPS are 62.750/64.943; their ratio is a different statistic. All paired runs match decoded text, emitted count 128 and returned position 155. CLI logs do not expose token IDs, so pairwise CLI checks alone are not token-ID proof; that evidence comes from the forty-item harness above. Host noise and only three pairs prevent a robust speedup conclusion. Keep the route opt-in. Historical resource samples across the retained stages reached minima 40% free memory and 29,427,322,880 free disk bytes, above the 25%/20 GiB fences. Existing host/process and per-run resource logs remain retained; this is not exclusive-host or thermal-control proof.

A separate `resumed-profile/` directory retains commands, stdout/stderr logs, resource samples, process snapshot, identity check and `analysis.json`. No source edit or rebuild was needed. The same frozen executable/model/prompt ran eight-token opt-in greedy once with `VXG_PROFILE_STAGES=1` and once without it, temperature/presence penalty zero and two threads. Both exited zero on actual `Apple M2 Max` Metal, produced identical decoded text, count eight and state position 35. The profile has 35 command-buffer samples: prompt positions 0–27 and seven subsequent decode transitions 28–34. Resource minima were 34% free memory and 78,017,757,184 free disk bytes; no unrelated process was killed.

The existing profiler splits compute encoders at stage boundaries and therefore changes scheduling. Summing repeated stage names across only positions 28–34 gives relative raw timestamp-delta shares:

| Stage | Relative sampled duration |
|---|---:|
| FFN GEMV | 36.61% |
| Recurrent GEMV | 19.56% |
| LM head | 15.19% |
| Norm/residual | 8.18% |
| DeltaNet | 6.51% |
| Attention GEMV | 4.63% |
| Attention | 4.27% |
| Greedy argmax | 3.50% |
| FFN activation | 1.23% |
| Embedding | 0.30% |

Raw GPU ticks are not interpreted as nanoseconds. Separately, Metal command-buffer `GPUEndTime-GPUStartTime` sums to 0.080707583 seconds over the seven decode transitions; profile decode wall time is 0.102314 seconds (78.9% command GPU time, 21.1% residual wall interval). The residual includes CPU preparation, submission, synchronization, timestamp resolution/logging and other scheduling costs; this instrumentation does not separate those costs or establish production CPU overhead. The unprofiled control decode wall time was 0.065461 seconds, demonstrating that the instrumented execution is materially perturbed. Neither profiled TPS nor the control pair is a production performance comparison.

The strongest next bottleneck hypothesis is the repeated matrix-vector work: sampled FFN/recurrent/attention GEMV together account for 60.81%, with the head another 15.19%. Preserve the full head and exact token/state contracts when investigating weight-read efficiency and dispatch behavior. GPU-dependent multi-token batching may reduce host round trips, but these data alone do not establish its benefit; cancellation, EOS, callback panic and exact returned-state recovery remain mandatory. Embedding is only 0.30% here, so optimizing it first has little support. Compression remains the unchanged CB4 archive ratio; this execution diagnostic adds no compression or general-quality claim.
