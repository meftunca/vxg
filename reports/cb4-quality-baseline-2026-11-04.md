# Existing CB4 quality baseline, 2026-11-04

One bounded existing-codebook reference experiment to decide whether a future four-bit Turbo fast path is worth investigating. This is intermediate selection evidence; CB4 uses the generic full-step Metal engine and does not fulfill the Turbo-speed objective. No new kernel, calibration, bit plan, question selection, file format or packing default was introduced. Existing dirty files and prior evidence were preserved; no commit or deletion.

## Frozen comparison and execution

Evidence root: `/private/tmp/vxg-real-qualification-20261104/cb4-baseline-20261104`.

The exact prior known 40-item free-generation suite, question/options/keys, source prompt IDs, no-thinking chat template, 384-token cap, temperature zero, presence penalty zero and strict final-label parser are retained. One case-sensitive `Final answer:` marker must have a trimmed suffix of exactly one uppercase A/B/C/D; any cap hit is truncated regardless of apparent answer. Only stop-ID completion with a valid final label scores completed correct/wrong. No prompt/scorer changes or answer retries. All 40 retained source answers are reused from the preceding diagnostic, without a source rerun. This remains exploratory evidence on known authored questions, not blind general-quality qualification.

Suite SHA256 `0caadde265c5feb0a528535459a2b660d88a74f5c4d80176e62126710687aa44`; inherited frozen protocol SHA256 `1e92bda984c96ff51080bc62e3db9f9eb1a1e179d6950b1585836b3184527593` (its historical branch list describes A/baseline/zero; this experiment applies the same answer protocol to explicit `cb4`). The CB4 command and explicit route are retained separately. The binary now accepts an explicit `cb4` stage requiring `1-CmdBuf Fused` and the generic greedy forward-step loop; A and existing Turbo stages retain their routes. Every CB4 row compares its prompt IDs/config/stop IDs against the frozen source row and records the actual backend. CPU fallback is rejected.

The first sandboxed quality load selected CPU and was rejected before producing any answers. Its log/result remain as `quality.log` and `quality-result.json`; the subsequent Metal-access run uses `quality-metal.log`. It evaluates each item once. Packing/verification succeeded in the sandbox; no repacking was needed.

## Artifact and codec

BF16 GGUF source: `/private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf`, 3,775,709,216 bytes, SHA256 `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1`.

CB4 archive: `Qwen3.5-2B-cb4.vxt`, 1,036,621,966 bytes (988.60 MiB), SHA256 `d4532feff049de11f60452fd99cc545b576e0d146cdf1e2dee989945b4171196`. Actual source/archive ratio **3.64232x**. Pack reported 60.72 seconds, outer tensor threads 1; the existing codec independently chooses hardware workers for row packing, so this is not a globally single-threaded encoder. The unchanged archive verifier passed all 320 tensors and five metadata keys; integrity does not imply source-weight equality for a lossy archive.

Inventory: 151 CB4+Outlier tensors, 133 RawF32, 36 lossless BF16; 14,068,268 sparse outlier entries. The existing codec uses 16 FP32 learned centroids per matrix, FP16 scales per 128-weight group, four-bit planar packed words and sparse FP16 outliers. Existing small/control tensors retain the packer's lossless/raw rules. Full tensor names, shapes, storage modes, plane counts and section sizes are in `archive-inventory.json`. No entropy/Turbo representation was introduced. Disk bytes are not a measurement of resident GPU bytes.

## Results

| Branch | Completed correct | Completed wrong | Format unscorable | Truncated |
|---|---:|---:|---:|---:|
| Original, retained previous run | 36 | 2 | 2 | 0 |
| Baseline Turbo, retained previous run | 11 | 7 | 18 | 4 |
| Zero-entropy Turbo, retained previous run | 21 | 1 | 8 | 10 |
| Existing CB4, this run | **32** | 2 | 4 | 2 |

Category cells are correct / wrong / format / truncated, each out of ten:

| Branch | Math | Code | Turkish | English |
|---|---|---|---|---|
| Original |10/0/0/0|10/0/0/0|7/2/1/0|9/0/1/0|
| Baseline Turbo |7/2/1/0|2/0/6/2|0/3/6/1|2/2/5/1|
| Zero-entropy Turbo |7/1/0/2|9/0/0/1|0/0/4/6|5/0/4/1|
| CB4 |10/0/0/0|9/1/0/0|5/1/2/2|8/0/2/0|

Among the 36 source-correct items, CB4 retains 32: one becomes completed wrong (`code-09`), one truncates (`tr-01`), and two become format unscorable (`tr-09`, `en-06`). CB4 gains no correct completion on the four source-noncorrect items. Prior Turbo retains 11/36 and zero-entropy Turbo 21/36. CB4 emits 5,561 tokens total; 38/40 terminate on stop IDs, 34/40 have valid finals plus stop termination. Every item used `Apple Metal GPU (Apple M2 Max, 1-CmdBuf Fused)` and the ordinary full-logit greedy forward-step path. No candidate tuning followed these results.

This substantially stronger small-suite result supports considering a future four-bit fast-path investigation. Four source-correct completions are still lost, and two outputs truncate. These known forty items cannot establish retained intelligence, a general quality percentage, a three-to-four-percent loss bound, or production reliability. The CB4/Turbo comparison combines codec, calibration, state precision and decoding-engine differences; it does not isolate a four-bit weight-only effect.

## Single bandwidth diagnostic

Exact same baseline prompt: `Explain why memory bandwidth limits LLM inference in 3 short bullet points.` Normal no-thinking chat template, 28 prompt tokens, maximum 128 emitted tokens, temperature zero, presence penalty zero, runtime threads two. One run, actual generic fused Metal backend, default production route for this CB4 artifact.

It emitted 128 tokens and hit the cap: load **6.706634 s**, prefill **0.259157 s**, decode **1.640063 s**, reported **77.44 tok/s**, state position 155. CLI reports `(generated - 1) / decode_seconds`; 127 forward steps account for the 128 emitted tokens after prefill. The generated text is retained in `bandwidth.log`; it reaches a third bullet but cuts off mid-sentence. Some explanations in that output are inaccurate; the prompt is a speed diagnostic, not an additional scored quality item.

Historical baseline Turbo on this prompt reported 297.00 tok/s using its specialized sketch/burst engine; CB4 here uses generic full-step Metal. This is an engine-confounded observation, not a fair kernel comparison or proof that a future four-bit Turbo kernel will reach either speed. The measured CB4 path does not meet the approximate 300 tok/s objective. The printed 80.3 GB/s compressed-stream figure is computed from archive bytes, not measured GPU traffic.

## Verification, resources and retained evidence

Release build passed. Two parser/explicit-route unit tests passed; the existing CB4 codec test passed. The existing synthetic Qwen3.5 CB4/WHT2 GPU/CPU parity test passed initially in the sandbox (GPU branch could be skipped), then the exact built test binary passed again with Metal-access execution after the model runs. The real model run independently proves actual Metal selection; the parity test itself retains its existing conditional device-availability behavior. These focused checks do not constitute a broad regression or performance suite.

The independent Python audit in `analyze.py` agreed with all 40 stored strict predictions/statuses, matched complete prompt text/IDs, gold/category/config/stop IDs and decoding options to source rows, and checked emitted counts and returned processed positions. Atomic per-item JSON checkpoints retain generated IDs/text and timings. Both model stages exited zero. No answer resume or duplicate generation occurred; the earlier CPU-refused load produced no answer checkpoint.

The host is Apple M2 Max with 32 GiB unified memory. An unrelated Vex compiler process was observed at about 1059 percent CPU and 5.4 GiB RSS before the experiment, alongside StorageManagement activity. These jobs were preserved. Before each stage the runner required at least 25 percent system free memory and 20 GiB disk. Two-second memory/disk samples are retained; no per-buffer GPU allocation or peak RSS measurement is claimed. Sampling can miss peaks, and unified CPU/GPU allocations are not additive.


| Stage | Exit | Whole-stage seconds | Minimum sampled free memory |
|---|---:|---:|---:|
| pack | 0 | 62.91 | 50% |
| verify | 0 | 6.09 | 65% |
| quality | 1 | 6.10 | 67% |
| quality-metal | 0 | 176.80 | 60% |
| bandwidth | 0 | 10.09 | 65% |

Measured binaries SHA256: `generation-quality` `77969f8dc05f661fe7c8fb95b8c272c3298075dc855016b9ebd4998d7126c59d`; `vxg-tensor` `8b5bbe4b661909acf3ca81853aaade753fe571241d61283a0044dec8929fa7d6`. Evidence includes immutable binary copies, source snapshot and SHA manifest, exact command arrays, logs, git identity/status/diff, inventory, frozen protocol/suite/source checkpoints, runner and paired `comparison.json`. Base commit `07f26e90dfbcb8d0160793e2bb7cd0e2476f90b6` plus the preserved dirty source tree.

Reproduce the build with `cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin generation-quality --bin vxg-tensor`. Exact measured pack, verify, quality and generation argument arrays are in `*-command.json`; `run.py` preserves the initial CPU-refusal path and `run-metal.py` the successful Metal-access stages. Use a new evidence directory for a new experiment. Lossless remains the packing default. No further variants, teacher-logit runs or model evaluations were performed.
