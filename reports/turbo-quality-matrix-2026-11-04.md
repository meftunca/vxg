# Turbo fixed-choice quality matrix, 2026-11-04

Bounded diagnostic on Apple M2 Max, 32 GiB unified memory. No quantizer, default, calibration, runtime kernel or source model change; no tuning or throughput run. Existing dirty tree retained. Evidence directory: `/private/tmp/vxg-real-qualification-20261104/turbo-quality-matrix-20261104`.

## Frozen protocol

Forty new deterministic items: ten arithmetic/multistep, ten Python code reading, ten Turkish reading/instructions, ten English reading/instructions. No previous arithmetic cases or evaluation/calibration text reused. All four branches receive the same fixed questions and choices. Gold labels are shuffled and balanced globally (ten each A/B/C/D). The answer key is computed by independent Python integer expressions and execution of fixed authored Python snippets, and manually checked against explicit Turkish/English text. `make_matrix.py` retains this construction and checking evidence. This is an authored small diagnostic, not an externally validated benchmark.

`suite.json` was frozen before any model scoring, SHA256 `0caadde265c5feb0a528535459a2b660d88a74f5c4d80176e62126710687aa44`. The unchanged selection rule allows no default promotion on this suite alone. No new quality metric or acceptance gate is selected after observing results.

Each task is scored at the model's no-thinking chat context plus identical `Answer:\n` prefix. Candidate continuations are exactly one token each, A/B/C/D. The runner validates `encode(prefix+label) == encode(prefix)+[label_token]`, label decoding, and common prompt IDs, label IDs, architecture, vocabulary and EOS against original A. Every full-vocabulary logit must be finite. Each label score is its full-vocabulary normalized log probability; argmax selects a choice, ties use the lowest label. For one-token alternatives, normalization cannot change their ordering. Four raw choice logits and four normalized log probabilities are retained per item, along with prompt IDs, key, prediction, backend and config. This is forced single-label recognition, not generated reasoning or completed-task generation accuracy.

Original A independently loads original BF16 GGUF expanded RawF32 and requires generic fused Metal. Baseline, zero-entropy and observed-activation archives use ordinary full-logit Turbo Metal; approximate production head/burst is unused. Backend fallback is an error. All stages load once and run sequentially with new state per item. Optional extra teacher traces were omitted to keep this sweep bounded.

A synthetic test verifies nontrivial shuffled token-to-label mapping, independently calculated full-vocabulary logsumexp, and deterministic tie handling. Release Metal build and this test passed. Per-item tokenizer boundary/config checks run on actual model stages.

## Results

All four stages completed successfully on their required actual Metal backends. Counts below are fixed-choice diagnostic scores. Wilson 95% intervals are descriptive binomial intervals, not a guarantee of coverage for a nonrandom authored item set, and are not confidence intervals for intelligence retention.

| Branch | Archive MiB | Correct /40 | Wilson 95% | Math /10 | Code /10 | Turkish /10 | English /10 | Source correct → wrong | Source wrong → correct |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Original BF16 / RawF32 | 3600.80 | 21 | 37.5–67.1% | 4 | 5 | 6 | 6 | — | — |
| Baseline Turbo | 603.40 | 12 | 18.1–45.4% | 1 | 3 | 2 | 6 | 13 | 4 |
| Zero entropy | 618.88 | 18 | 30.7–60.2% | 3 | 6 | 3 | 6 | 11 | 8 |
| Observed activation | 603.26 | 13 | 20.1–48.0% | 1 | 3 | 2 | 7 | 12 | 4 |

Relative to baseline, zero gains eight items and loses two; activation gains one and loses none. Relative to the 21 source-correct items, all candidates lose many items. Raw counts and complete item identities appear in `comparison.json`; independently correct candidate answers on source-wrong items are explicitly retained rather than excluded.

There is substantial answer-position bias despite a globally balanced key. Predicted A/B/C/D counts: source 6/18/10/6; baseline 5/1/25/9; zero 16/0/21/3; activation 4/3/24/9. The weak source score and these biases show that this particular direct-choice prompt is itself a difficult interface for the model. Position bias is not evidence of a scoring bug: the synthetic mapping/normalization check passed and every branch used the exact source token IDs. These outcomes cannot be equated with free-generation task correctness, or a percentage of retained intelligence. The frozen prompt/key/scoring were not changed after these results. A further validation should freeze a substantially larger independently reviewed benchmark with a separately validated answer interface, include generated-task checks, and avoid tuning on these forty items.

Zero entropy is a **promising follow-up candidate on this suite only**, at a 2.565% archive-size cost. Its prior old-context KL/top1 regression remains evidence against assuming general improvement. Observed activation shows only a one-item gain here and previously failed its discovery safeguard. No candidate is promoted. Forty authored items cannot establish a 3–4% intelligence-loss bound; one item already represents 2.5 percentage points, and the intervals are broad. No 12× compression, broad benchmark, long-context, perplexity or speed result is established.

## Identity, resources and reproduction

`identities.json` records exact paths, hashes and bytes. Independent streaming `metadata-blob-audit.json` verifies every baseline metadata blob is byte-exact across all three Turbo archives, including configuration and tokenizer; only opt-in provenance adds metadata. Itemwise source checks cover arch/vocabulary/EOS and all actual prompt/label IDs. This matrix does not independently re-audit every source hyperparameter or repeat the prior source tensor mapping audit.

| Artifact | SHA256 |
|---|---|
| Original GGUF | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` |
| Baseline Turbo | `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71` |
| Zero Turbo | `6ef87e2b9804b5a5186687cef8449897c5905ee4adc33cae83518a9c550d44b8` |
| Activation Turbo | `1efbeb2db44f6d1e88a5fe4de3c8bfc4972007bfd3c8e7792f31e2874b5c8a7d` |
| Measured quality-matrix | `6d2e1486732d7181f7a6c8611247c39b61af1f76b40cad4e2c2306aa7700b5de` |

| Stage | Min sampled system free | Max sampled RSS KiB | Whole stage seconds |
|---|---:|---:|---:|
| Original A | 27% | 7310832 | 89.15 |
| Baseline | 70% | 1889584 | 20.23 |
| Zero | 72% | 1884832 | 18.14 |
| Activation | 72% | 1885136 | 20.16 |

Checks before each stage require at least25% system free and20GiB disk free. Initial free71%, disk34GiB. Two-second samples can miss peaks; RSS is not independent physical GPU allocation and unified-memory measurements must not be added. Whole-stage elapsed includes model load, forty prefills and evidence writes and is not a throughput benchmark. One model process intentionally active per stage; unrelated host load is not controlled. Small per-choice evidence avoids full-vocabulary logit dumps.

Retained evidence includes suite/key/construction script, scoring source and executable, runtime Rust/native source hashes, dirty diff/status/base commit, exact command arrays, build/test log, per-stage resource samples and logs, per-item raw choice logits/logprobs, analysis and metadata-audit scripts. Source/default/kernel changes were not made by this sweep.

```sh
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin quality-matrix
cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin quality-matrix
# Exact measured commands are A-command.json, baseline-command.json,
# zero-command.json and activation-command.json in the evidence directory.
# Use new output directories; never overwrite existing evidence.
quality-matrix A SOURCE.gguf suite.json NEW_ROOT/A
quality-matrix baseline BASELINE.vxt suite.json NEW_ROOT/baseline
quality-matrix zero ZERO.vxt suite.json NEW_ROOT/zero
quality-matrix activation ACTIVATION.vxt suite.json NEW_ROOT/activation
```

`run.py` performs the recorded resource checks, sequential execution and hash capture; `analyze.py` reproduces all correctness counts, paired reversals, category breakdown, predicted-label counts and Wilson intervals from retained scores. Existing source A must be available as the sibling `A` directory for candidate token checks. No broad test suite or optional teacher-trace sweep was run.
