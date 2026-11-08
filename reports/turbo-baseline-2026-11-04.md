# Turbo baseline, 2026-11-04

Exploratory current-code baseline on Apple M2 Max, 32 GiB unified memory. Explicit lossy `--mode turbo`; lossless remains the CLI packing default. No new compression algorithm, calibration training, or default mode change. Existing dirty work preserved. Artifact directory: `/private/tmp/vxg-real-qualification-20261104/turbo-baseline/` (abbreviated `$D` below); retained original source directory is its parent. Base commit `07f26e90dfbcb8d0160793e2bb7cd0e2476f90b6` plus dirty tree, captured in `source-sha256.json`, `working-tree.diff`, and `git-status.txt`. Graph query lacked Turbo coverage; claims below come from direct source and actual runs.

## Identity, compression, and storage

| Artifact | Bytes | SHA256 |
|---|---:|---|
| BF16 GGUF source | 3,775,709,216 | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` |
| Turbo VXT | 632,712,974 | `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71` |
| Pack executable (hash retained; subsequently rebuilt) | not separately retained | `92e0ffa2b9bfb0419b8ea7da5b0f6ab45a52b83d5faf58e9c5af2ae5681dc2a4` |
| Diagnostic generation/eval executable | retained `vxg-tensor-runtime` | `af40ddca0bc99dd8d6423ea30dd482b7736044a207adb54530f9b70651269899` |

Packing reports 3600.80 MiB to603.40 MiB, 5.97x,45.28 seconds. This reproduces approximately600MiB compression on this source, rather than exactly590MB. `verify` independently rereads all320 archive tensors and5 metadata keys, validates section/whole-archive FNV integrity, and passes. This checks container integrity, not equality of lossy weights to source. `/usr/bin/time -l` pack wrapper returned1 after successful packing because sandbox blocked `sysctl kern.clockrate`; successful artifact and integrity verification were retained.

The previously verified source inventory is187BF16 and133F32 tensors; source hash above is unchanged. Full archive inventory is `archive-inventory.json`, produced by retained `inventory.py`:151 EC-Turbo tensors (subtype4, production display `VXT-EC-Turbo3`, F16 scales plus u32 code planes),167RawF32,2lossless BP16 BF16 control tensors. All320 names/shapes/storage kinds and disk section sizes are included. Metadata keys/encoded sizes are retained there; serialized model config and tokenizer are embedded in archive. Calibration and tensor-specific lambdas are those in captured `llm_pack.rs`/`llm_calib.rs`; no alternate calibration dataset was supplied.

The entropy-compressed disk sections expand to741,157,484 bytes of stored decoded planes before runtime upload. This is neither GPU allocation nor total memory use. EC Huffman/zero-sign elision is decoded before GPU use. Native code creates shared Metal buffers and an aligned unified weight arena, concatenates projection matrices, constructs sketch metadata, and allocates FP16 recurrent state/KV caches plus scratch and speculative backups. Ordinary CPU weights and Metal allocations also remain. Actual production-run maximum RSS was3,262,021,632 bytes and peak memory footprint2,618,592,256 bytes. There is no per-buffer GPU allocation measurement in this baseline; unified-memory RSS must not be added to GPU storage as if independent physical RAM. The CLI's603.40MiB and equivalent187.9GB/s stream calculation describe disk bytes, not measured GPU bytes read.

## Actual greedy generation and diagnostic comparison

All runs used temperature0, presence penalty0, threads2, default chat template/thinking disabled, and profiling env variables unset. Prompt: `Explain why memory bandwidth limits LLM inference in 3 short bullet points.` Prompt length28tokens; maximum128new tokens.

| Path | Generated | Load s | Prefill s | Decode s | Reported decode tok/s | State pos |
|---|---:|---:|---:|---:|---:|---:|
| Turbo production sketch/burst |128|6.423658|0.046228|0.427614|297.00|155|
| Turbo exact-head/burst |128|5.110596|0.045837|0.446092|284.69|155|
| Turbo ordinary full-logit forward step |128|5.437972|0.063255|0.599094|211.99|155|
| Original GGUF full-logit forward step |127, stop-token termination|14.412757|2.182515|3.185568|39.55|155|

Actual Turbo backend: `Apple Metal GPU (Apple M2 Max, VXT-Turbo Pipelined)`. Original backend: `Apple Metal GPU (Apple M2 Max, 1-CmdBuf Fused)`. Approximate300TPS is reproduced on one128token diagnostic run; this does not establish a stable300+TPS guarantee. Unrelated Vex gate/wrk jobs and macOS StorageManagement were active; no model jobs were deliberately overlapped or killed. Memory query before baseline reported61% free, and57% during eval. Runs were sequential. Source loader independently expands BF16/F32 tensors to RawF32 (not re-encoding them with Turbo); original load/RSS and39.55TPS are therefore not an optimized BF16-runtime competitor benchmark.

The three Turbo variants produced identical598byte decoded UTF-8 text, SHA256 `1a960b9466606b54d5d26049b18bff9b59c2065691f67d1d65551955454bb028` (`generation-text-comparison.json`). Token IDs were not traced, so this establishes decoded-text equality on one capped prompt, not general token-sequence or state equivalence. Turbo output repeats that model capacity is not the limiting factor. Original output differs and supplies three separate bullets. Neither answer alone establishes broad reasoning quality.

CLI decode timing includes streaming/output and uses `(generated-1)/elapsed` for multi-token outputs. On stop-token termination that numerator need not equal actual forward executions. Full seconds/counts are recorded to avoid conflating emitted tokens with speculative work. A narrow `--reference-decode` diagnostic flag was added to force the ordinary loop; default behavior is preserved. Existing `--exact-head` controls the fast head comparison. Build passed and all three paths were exercised on real GPU; no separate broad regression suite was run.

## Source versus Turbo full-logit comparisons

Each case uses16greedy reference-selected teacher tokens after its fixed chat prompt. Source is the independent BF16 GGUF loader, not a Turbo or lossless archive. Eval frees source model before loading compressed model. These are tiny exploratory prompts, not an independently held-out benchmark or statistical quality bound. Exact prompts and commands are in `eval-prompts.json` and `eval-{tr,en,code,math}.log`.

| Case | Prompt tokens | Mean KL nats | Mean logit cosine | Mean RMSE | Top1 |
|---|---:|---:|---:|---:|---:|
| Turkish capital/Ankara |32|0.179347|0.859329|1.29869|16/16|
| English bandwidth |25|0.395213|0.842413|1.49548|14/16|
| Python sum of evens |27|0.075581|0.781840|1.58841|16/16|
| Arithmetic17x23 |27|0.242369|0.876969|1.30047|13/16|

Logit changes are substantial compared with earlier lossless numerical parity. Even16/16top1 does not establish no quality loss. KL/cosine/RMSE do not translate into a percentage of retained intelligence. Eval uses full logits, not the production sketch selection; its timing also includes comparison work and is excluded from speed claims. Weight quantization/calibration, FP16 state/KV/rotations and kernel differences are combined here; their contributions have not been isolated. These results cannot individually qualify the approximate head or burst path.

## Arithmetic behavior and stop-state limitation

Production Turbo prompt: `What is 17 times 23? Show a short calculation.` It emitted33tokens then a stop token, answering **411**, which is wrong (17x23=391). Run log `math-turbo.log`: prompt27, load11.382369s, prefill0.046331s, decode0.161635s, reported197.98TPS, state_pos75. This is direct task-level evidence of failure, regardless of logit metrics.

Fast pipeline increments state position when submitting speculative command buffers. On early stop it waits pending commands and returns without restoring the recurrent/KV state to the stop boundary (`MetalTurboRunner::generate_greedy_fast`). Here75 exceeds prompt27+emitted33=60, so returned state is ahead of the visible sequence. The standalone CLI discards state, but the resumable API must not be treated as EOS-aligned. Original source on the same math prompt was capped at64tokens before a final answer; it correctly starts17x3=51, but this truncated output does not establish a completed correct original answer (`math-original.log`: load12.861245s, prefill2.001003s, decode1.512605s,41.65reportedTPS, state90). No rollback fix is claimed. A safe next correctness change should define/validate stop-state semantics and restore or replay accepted state; decrementing the position counter alone is insufficient because caches/recurrent state have advanced.

## Reproduce

```sh
cargo build -p vxg-tensor --release --features metal --bin vxg-tensor --target-dir /private/tmp/vxg-qualification-target
B=/private/tmp/vxg-qualification-target/release/vxg-tensor
S=/private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf
D=/private/tmp/vxg-real-qualification-20261104/turbo-baseline
"$B" pack-model "$S" "$D/Qwen3.5-2B-turbo.vxt" --mode turbo --threads 1
"$B" verify "$D/Qwen3.5-2B-turbo.vxt"
"$B" run "$D/Qwen3.5-2B-turbo.vxt" --prompt 'Explain why memory bandwidth limits LLM inference in 3 short bullet points.' --max-tokens 128 --temp 0 --presence-penalty 0 --threads 2
# Repeat with --exact-head, then --reference-decode.
"$B" eval "$S" "$D/Qwen3.5-2B-turbo.vxt" --prompt 'What is 17 times 23? Show a short calculation.' --steps 16
```

Outer pack threads1 avoids multiplying tensor workers by per-matrix hardware workers; the entropy chunk encoder still spawns per-chunk workers. No algorithm was changed to constrain it. Runtime source identity and dirty diff after the diagnostics are separately retained in `runtime-source-sha256.json` and `runtime-working-tree.diff`; `vxg-tensor-runtime` retains the measured runtime binary. The original pack binary was rebuilt, so only its hash and exact source snapshot remain. Exact pack/runtime binary hashes differ because the diagnostic CLI flag/timing output were added after packing. Hashes for both stages and source snapshots are retained.

Next: preserve this artifact as baseline; first address early-stop returned-state semantics, then isolate math divergence with source/quantized ordinary decoding versus production fast decoding and FP32/FP16 state controls. Only that error localization should guide selective precision/calibration experiments. More compression is not yet justified by these quality results. Long context, broad multilingual/code/math tasks, perplexity, clean repeated timing, per-buffer GPU allocations, and general fast-path sequence equivalence remain unqualified.
