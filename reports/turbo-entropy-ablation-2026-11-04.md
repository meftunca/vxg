# Turbo entropy ablation, 2026-11-04

One bounded zero-entropy experiment on Apple M2 Max, 32 GiB unified memory. **Do not promote the candidate:** it is larger, old-context fidelity worsens, and heldout results are mixed despite corrected original arithmetic. Default entropy scale remains 1; lossless remains the CLI packing default. Existing dirty worktree and all prior archives retained; no commits or cleanup.

Evidence directory (`$D`): `/private/tmp/vxg-real-qualification-20261104/turbo-entropy-ablation-20261104`. Base commit `07f26e90dfbcb8d0160793e2bb7cd0e2476f90b6` plus retained dirty diff. The source/binary/artifact identities are in `identities.json`; source snapshots and both measured binaries are retained there.

## Change and validation

Explicit CLI `--turbo-entropy-scale S` and API `pack_gguf_model_to_vxt_with_entropy_scale` multiply both existing row-lambda outputs at the pack boundary, uniformly across embedding/head and other Turbo matrices. No calibration, smoothing, Hessian, codebook, GPU precision, layout, stop-state, or runtime algorithm changed. Tiny matrices already routed lossless stay so. Nonfinite/negative scales rejected; explicitly passing the CLI option with a non-Turbo mode rejected, including 1. Original public API wraps scale 1. Nondefault archive has optional `model.turbo_entropy_scale=0` provenance metadata; default metadata unchanged. No ambient environment knob.

Release Metal build passed. Two release unit tests passed (11 role examples): scale 0 equals direct zero-lambda encoding; scale 1 archive tensor bytes equal independently reconstructed legacy row-lambda expression; invalid values rejected before input parsing. Initial tiny test mistakenly used 16 rows and therefore exercised existing lossless routing; retained failed log, corrected test uses 64x64 and passes. No giant default repack. CLI help was updated after measured binaries; final test additions are cfg(test) only. Exact measured snapshots remain in evidence; final source differs only by the usage-string addition after identity capture.

## Storage

| | Baseline scale 1 | Candidate scale 0 |
|---|---:|---:|
| Archive bytes | 632712974 | 648940686 |
| MiB | 603.40 | 618.88 |
| Source/archive ratio | 5.967x | 5.818x |
| Decoded plane bytes | 741157484 | 741157484 |

Candidate is 16227712 bytes larger (2.565%). Exact per-tensor expanded-plane structural inventory equality asserted across all 320 tensors: names, shapes, padded columns, kind/subtype, storage and plane lengths. Same 151 EC-Turbo4, 167 F32, 2 BP16-BF16. These are decoded storage planes, not measured GPU allocation or memory bandwidth. Container verification passed every tensor and FNV integrity (6 metadata keys). Pack elapsed 20.24s, outer threads1. No baseline replacement.

## Frozen reference-context quality

Existing originalA prompt IDs, A-selected teacher IDs and full 16-step logits reused from the quality-split evidence; no new teacher selected for those four cases. Candidate actual backend required `Apple Metal GPU (Apple M2 Max, VXT-Turbo Pipelined)`. A first sandbox attempt returned CPU and deliberately failed before any trace; its failed directory/log remain. All accepted traces are real GPU, no fallback.

Four new prompts were written to `heldout-prompts.json` and plan before packing/results: Turkish Ayşe library reading comprehension; exactly two English sentences about green leaves; Python largest even integer or None; 19 times 14. Source A generated 16 teacher steps per new case; baseline C and zero C asserted identical prompt/teacher IDs. Every vocabulary logit retained, finite; float64 stable directional KL P_source || P_candidate. Eight small contexts are not a broad quality benchmark or intelligence-retention percentage.

| Contexts / branch | Mean RMSE | Mean KL nats | Mean cosine | Top1 equal |
|---|---:|---:|---:|---:|
| old / baseline C | 1.420764781 | 0.223127631 | 0.840137687 | 59/64 |
| old / zero C | 1.454099544 | 0.238384831 | 0.829564779 | 55/64 |
| heldout / baseline C | 1.445883507 | 0.362075121 | 0.830365970 | 54/64 |
| heldout / zero C | 1.410701454 | 0.288749181 | 0.837972602 | 53/64 |

Old contexts worsen overall. Heldout mean RMSE/KL/cosine improve but top1 drops; code RMSE/KL worsen on both sets. All per-case and per-step results retained in `comparison.json`. No sweep, tuning, selection of teacher or default change followed these outcomes.

## Actual generation and one production sample

Original arithmetic `What is 17 times 23? Show a short calculation.`, ordinary full-logit greedy loop, presence penalty 0, cap256: zero C emits 11 tokens `17 × 23 = 391`, EOS248046, state position38. Prior retained A emits153tokens, answer391; prior baseline C emits33tokens, answer411. Candidate correctness on this one answer does not outweigh the quality tradeoff. On heldout19x14, source/baseline/zero all answer266; full generations and IDs retained. Other heldout domains are reference-context trace evaluations, not completed task generations.

Sequential production prompt matches original baseline bandwidth prompt, cap128, temperature0, presence0, threads2. Same current stop-state-fixed runtime and actual Turbo fast sketch-head backend for both; no explicit warmup run. Both hit128 cap (state_pos155), no EOS. Baseline load5.812881s, prefill0.042638s, decode0.418762s, reported303.27TPS. Zero load5.500960s, prefill0.043624s, decode0.420498s, reported302.02TPS. These single samples are **not a controlled speed claim**. Both outputs repeat content; raw text/logs retained. Compression-derived GB/s is not measured GPU traffic. No separate GPU allocation measurement.

## Reproduction and identities

```sh
cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin vxg-tensor --bin quality-ablation
B=/private/tmp/vxt-stop-target/release/vxg-tensor
S=/private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf
D=/private/tmp/vxg-real-qualification-20261104/turbo-entropy-ablation-20261104
"$B" pack-model "$S" "$D/Qwen3.5-2B-turbo-zero.vxt" --mode turbo --threads 1 --turbo-entropy-scale 0
"$B" verify "$D/Qwen3.5-2B-turbo-zero.vxt"
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --lib entropy_ablation_tests -- --nocapture
```

Quality commands, exact models/output paths: `run-heldout.py`; old C and B commands use `quality-ablation C|B MODEL .../turbo-baseline/eval-prompts.json NEW_DIR` with sibling `A` linked to retained originalA. `compare.py` and `comparison.json` retain independent analysis. `production.py` retains exact production commands. Create new output directories on reproduction; do not overwrite retained evidence.

| Artifact | SHA256 |
|---|---|
| Source BF16 | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` |
| Baseline Turbo | `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71` |
| Zero Turbo | `6ef87e2b9804b5a5186687cef8449897c5905ee4adc33cae83518a9c550d44b8` |
| vxg-tensor | `1947fb7556e0aa9d3c25f862f18cd1ede15af256c6e154d3d9b8caab667ea6c6` |
| quality-ablation | `0c7383556d3a7a09c646e54bcdca07cd189b02b627729c38015e3e5fbf8e4cc8` |

Initial system free81%; sampled during sourceA36%. Sampling was sparse, not peak monitoring. No intentional concurrent model processes; main stages sequential. Sandbox process inventory was denied, so unrelated host jobs were not fully inventoried. Broad tests, perplexity, long context, controlled repeated throughput, memory peaks and downstream quality suite remain unrun. Stop-state code was not edited by this experiment.

## Decoded-weight attribution check

Final already-scheduled B stage completed on required generic fused Metal backend, same zero archive reconstructed exactly to RawF32 before assembly. Old A-B mean RMSE 1.453830203, KL 0.238119450, cosine 0.829631147, top1 55/64. Zero B-C mean RMSE 0.003729037, KL 2.74298704e-06, cosine 0.999998531, top1 64/64. Thus old-context degradation persists in reconstructed-weight pack-side path; optimized engine difference remains small here. Decoded inventory and full traces retained. B original arithmetic IDs/text equal C exactly (391). This remains a bounded context result.
