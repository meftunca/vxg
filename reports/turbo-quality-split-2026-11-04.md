# Turbo quality causal split, 2026-11-04

Bounded A/B/C experiment on Apple M2 Max, 32 GiB unified memory. Existing dirty tree preserved; no commits, cleanup, quantizer changes, calibration, or default changes. Base commit `07f26e90dfbcb8d0160793e2bb7cd0e2476f90b6` plus dirty tree. Evidence directory: `/private/tmp/vxg-real-qualification-20261104/turbo-quality-split-20261104/`.

## Explicit diagnostic paths

- **A:** independent original BF16 GGUF loader, expanded RawF32, actual `Apple Metal GPU (Apple M2 Max, 1-CmdBuf Fused)`.
- **B:** exact same baseline Turbo archive, every tensor reconstructed using `PackedTensor::decode_f32` and materialized as RawF32 **before assemble**; same generic fused Metal backend as A. Config/tokenizer and packed smoothing/norm vectors are retained. Generic recurrent state and KV buffers use `sizeof(float)` allocations. No giant raw archive is written.
- **C:** same Turbo archive through existing ordinary full-logit `forward_step`; actual `Apple Metal GPU (Apple M2 Max, VXT-Turbo Pipelined)`. Optimized mixed precision state/cache/kernels. It does not use production approximate head/burst generation in this experiment.

New dedicated `quality-ablation` binary requires the expected backend string for each stage and errors on fallback. New explicit `LlmModel::load_vxt_decoded_raw_diagnostic` API leaves default loaders unchanged. A/B/C run in separate sequential processes, with one model loaded per stage and fresh states for every case. No intentional overlapping model work or external Transformers run. B-C includes different kernel arithmetic, reductions, rotations, recurrent/cache numerics, and weight reconstruction implementation; it cannot be attributed solely to FP16 state.

## Evidence

Four retained prompts from baseline `eval-prompts.json` (Turkish, English, Python, arithmetic), 16 greedy A-selected teacher tokens each. A-selected teacher IDs and prompt IDs are stored and asserted identical in B/C; every complete vocabulary logit is retained as little endian f32. All logits finite. KL is directional P_left || P_right, with float64 stable softmax in offline analysis. Means below are over 64 individual steps; top1 compares the indicated two branches at the same reference context. No ratio/percentage attribution of nonlinear effects is calculated.

| Comparison | Mean RMSE | Mean KL nats | Mean cosine | Top1 equal |
|---|---:|---:|---:|---:|
| A-B | 1.420792609 | 0.223068951 | 0.840129505 | 59/64 |
| B-C | 0.003439202 | 1.93503982e-06 | 0.999998849 | 64/64 |
| A-C | 1.420764781 | 0.223127631 | 0.840137687 | 59/64 |

| Case | A-B RMSE / KL / top1 | B-C RMSE / KL / top1 | A-C RMSE / KL / top1 |
|---|---|---|---|
| tr | 1.298895 / 0.179298 / 16/16 | 0.003726 / 2.16596e-06 / 16/16 | 1.298688 / 0.179347 / 16/16 |
| en | 1.495159 / 0.394681 / 14/16 | 0.002977 / 2.81347e-06 / 16/16 | 1.495483 / 0.395213 / 14/16 |
| code | 1.588581 / 0.0756073 / 16/16 | 0.003776 / 7.82389e-07 / 16/16 | 1.588415 / 0.0755814 / 16/16 |
| math | 1.300536 / 0.242689 / 13/16 | 0.003278 / 1.97834e-06 / 16/16 | 1.300473 / 0.242369 / 13/16 |

The dominant measured difference is already present in **pack transforms plus reconstructed weights**, A-B. The additional optimized engine difference B-C is much smaller on these contexts and preserves all 64 top1 choices. This does not isolate VQ from smoothing, calibration, entropy regularization, or any pack-side transform, and does not prove engine parity on other contexts. Aggregate numeric quality is not a percentage of retained intelligence.

## Completed arithmetic generation

Same original prompt, temperature zero, no presence penalty, ordinary full-logit greedy loop, maximum 256 emitted tokens, EOS withheld:

| Branch | Emitted | Actual stop | Returned position | Final answer |
|---|---:|---:|---:|---|
| A | 153 | 248046 | 180 | 391, correct; shows 17x3=51, 17x20=340, 51+340=391 |
| B | 33 | 248046 | 60 | 411, wrong |
| C | 33 | 248046 | 60 | 411, wrong |

B and C generated token IDs match exactly, not just decoded text. All three reached a stop before the cap. Full text and token IDs are in `A/B/C/math-generation.json`. This completes the formerly truncated original source answer. Production fast D was unnecessary for the causal split: failure already reproduces in B and C; the retained baseline/stop-state report separately covers fast production. No arithmetic-specific fix or optimization was made.

## Reconstruction, identity, and resources

`B/decoded-inventory.json` records every one of 320 tensors, shape, original variant discriminant, element count, finite status, and FNV-1a over exact reconstructed little endian f32 bits. All 320 finite. Tiny synthetic RawF32 (including negative zero/subnormal) and EC-Turbo tensor tests assert exact decoded bit equality after materialization and explicit RawF32 variant. Real B integration proves generic fused routing and full model forward execution. This fingerprints materialization, not equality of lossy matrices to source.

Independent GGUF vector bit audit (`source-vector-bit-audit.json`): 115 one-dimensional source tensors; 69 decoded archive vectors exactly match source f32 bits, 46 changed vectors are precisely layer1..23 attn/post-attention norms. These are the packed smoothing transformations retained by B, not silently replaced with source vectors. The archive's all-tensor inventory is retained by baseline. Prompt tokenizer identity is asserted for all four cases, and archive config/tokenizer are read through the same metadata methods for B/C.

| Stage | Min sampled system free | Max sampled process RSS KiB | Total stage elapsed s |
|---|---:|---:|---:|
| A | 27% | 6,555,680 | 22.2054 |
| B | 36% | 10,934,880 | 30.3966 |
| C | 71% | 1,849,184 | 8.1072 |

Two-second sampling can miss allocation peaks; RSS is not independent physical GPU allocation and is not additive to unified memory. Initial free memory63%; after A exit73%. The sampled minimum remained27%; no memory-pressure failure occurred. Before-run process inventory retained. Stage durations include load, four cases, disk trace writes, and arithmetic generation; they are not decode TPS benchmarks.

| Artifact | SHA256 |
|---|---|
| Qwen3.5-2B-BF16.gguf | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` |
| Qwen3.5-2B-turbo.vxt | `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71` |
| quality-ablation | `6cc3c014e9a7139a2146c39748f4753796b9938067d4dca86eb9428213d7cb82` |

Retained executable `quality-ablation`, exact edited source snapshots, all runtime Rust/native source hashes (`runtime-source-sha256.json`), dirty diff/status, driver/analysis/audit scripts, raw traces and per-step `comparison.json` permit inspection without rerunning models. Baseline archive/source hashes match the prior report. Raw traces total about182 MiB (three stages x four x16x248320x4 bytes), not gigabytes of reconstructed weights.

Validation: release Metal build succeeded; both `materialization_preserves_exact_decode_bits` unit tests passed; all three real GPU integrations completed. Broad tests, perplexity, long-context/state drift, held-out downstream tasks, and controlled repeated throughput were not run in this bounded diagnostic.

## Next experiment

Run one isolated bitrate-distortion ablation with a lower or zero entropy penalty lambda while preserving the EC codebook, GPU3.125 layout, existing smoothing/calibration inputs, prompts and reference teacher trace. Compare reconstructed-weight A-B fidelity, actual held-out task generation, and final archive bytes against this frozen baseline. This targets the dominant measured pack-side term without changing the optimized engine; it does not assume that VQ alone caused all A-B error. Select neither a policy nor precision based solely on the arithmetic prompt. No such change was implemented here.

Reproduction: build `cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin quality-ablation`; use the retained `run_quality_split.py` paths with fresh output directories, then `compare_quality_split.py` under the retained numpy environment. The driver is intentionally create-new and will not overwrite evidence.
