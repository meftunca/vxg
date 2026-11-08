# Turbo free-generation quality diagnostic, 2026-11-04

The original model produced 36 strictly correct completed answers out of 40; baseline Turbo 11 and zero-entropy Turbo 21. Zero improves this small diagnostic but still loses 15 source-correct completions and truncates 10 items. No candidate is promoted, and defaults remain unchanged. No compression, calibration, kernel or model tuning was performed. Existing dirty files and all prior evidence were preserved; no commit or cleanup.

## Frozen exploratory protocol

This reuses the exact 40 known items/options/keys from `turbo-quality-matrix-2026-11-04.md`. It is a changed answer protocol on known data, not new blind benchmark evidence. The identical user text is the original question, lettered options, then `You may reason briefly; end with Final answer: <A/B/C/D>.` The normal no-thinking chat tokenizer template is identical across branches; no forced assistant `Answer:` prefix is appended. Reasoning can be generated naturally.

Before any model outputs, `protocol.json` fixed cap 384, temperature 0, presence penalty 0 and scoring: exactly one case-sensitive `Final answer:` marker, followed only by whitespace, one uppercase A/B/C/D, and optional terminal whitespace. Repeated or conflicting finals, trailing punctuation/text, absent markers and other labels are unscorable. Reasoning numbers/letters are never compared to the gold answer. Any cap hit is truncated regardless of apparent final text. Only stop-ID termination with a valid final label receives completed-correct/wrong status. Format-unscorable rows below terminated on a stop ID. This deliberately strict interface was not repaired after results.

Original uses actual generic fused Metal greedy generation; candidates use actual Turbo fast generation with production default sketch head. Thus candidate comparisons include production decoding/head differences; these results do not isolate weight loss. All three use the same source prompt token IDs, architecture, vocabulary, EOS and stop IDs, checked itemwise. Standard EOS/im_end/endoftext stop IDs and the fixed accepted-state fast-generation implementation are used. First-token selection and prefills use normal full logits. One load per branch, fresh state per item, sequential original then baseline then zero. Activation and new candidates were excluded.

Suite SHA256: `0caadde265c5feb0a528535459a2b660d88a74f5c4d80176e62126710687aa44`; pre-output protocol SHA256: `1e92bda984c96ff51080bc62e3db9f9eb1a1e179d6950b1585836b3184527593`. The fixed maximum was 46,080 generated tokens; actual total 16,232.

## Results

| Branch | Completed correct | Completed wrong | Format unscorable | Truncated | Stop-ID completion | Valid final + stop |
|---|---:|---:|---:|---:|---:|---:|
| Original |36|2|2|0|40/40 (100%)|38/40 (95%)|
| Baseline Turbo |11|7|18|4|36/40 (90%)|18/40 (45%)|
| Zero entropy |21|1|8|10|30/40 (75%)|22/40 (55%)|

Category cells are correct / wrong / format / truncated, each out of 10:

| Branch | Math | Code | Turkish | English |
|---|---|---|---|---|
| Original |10/0/0/0|10/0/0/0|7/2/1/0|9/0/1/0|
| Baseline |7/2/1/0|2/0/6/2|0/3/6/1|2/2/5/1|
| Zero |7/1/0/2|9/0/0/1|0/0/4/6|5/0/4/1|

Among 36 source-correct items, baseline retains 11, produces 5 completed-wrong, 16 format-unscorable and 4 truncated. Zero retains 21, produces 1 completed-wrong, 7 format-unscorable and 7 truncated. Neither candidate gains a correct completion on the four source-noncorrect items. Relative to baseline, zero gains 12 correct completions and loses 2 (net 10). Full paired status counts and item identities are in `comparison.json`.

The original's improved source result compared with forced-label recognition (21/40) demonstrates answer-interface sensitivity; it is not a clean cross-protocol quantization improvement. Baseline and zero still fail many real completed-answer checks. Zero's stronger code result coexists with more cap hits and no strictly correct Turkish completion. Capped outputs are not proof of completed correctness. These 40 authored known items cannot establish a percentage of retained intelligence, a 3–4% loss bound, or broad model reliability. No post-result prompt edits, rescoring relaxation, extra datasets or reruns were used.

## Verification and retained evidence

Evidence root: `/private/tmp/vxg-real-qualification-20261104/turbo-generation-quality-20261104`. Each atomic per-item checkpoint retains full generated IDs/text, prompt IDs/text, key/prediction/status, backend, decode path, cap, emitted count, processed generation count/position, stop reason/IDs and prefill/decode/sample times. Final summaries, exact commands, source copy, binary, source/artifact hashes, git identity/status, build/test log, resource samples, runner, protocol, suite and analysis are retained. Timings are diagnostic only: variable lengths and uncontrolled host load prevent throughput comparisons.

Strict-parser release unit tests passed, covering malformed/missing labels, reasoning numbers, conflicting/repeated finals and trailing text. An independent Python terminal-regex audit agreed with every stored prediction/status and checked emitted counts and returned-state positions on all 120 outputs. All rows used their required actual GPU backend; all three processes exited 0. No broad test suite was run for this diagnostic-only binary.

| Stage | Whole stage seconds | Min sampled system free | Max sampled RSS KiB |
|---|---:|---:|---:|
| Original |234.86|27%|8525600|
| Baseline |56.85|72%|2334736|
| Zero |94.27|76%|2352656|

Initial free memory 80%, disk 30 GiB; before each branch required 25% free memory and 20 GiB disk. Two-second sampling can miss peaks. RSS and unified GPU memory are not additive. Stage durations include load/prefill/generation/evidence writes and are not speed scores.

Measured executable SHA256 `95dcfbc2cf2381e80b9339b565d793ae6825c6fda1851cdbbb6f092274d7cc0e`. Original artifact `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1`; baseline `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71`; zero `6ef87e2b9804b5a5186687cef8449897c5905ee4adc33cae83518a9c550d44b8`. These match prior artifacts. `identities.json` records full paths, byte sizes and runtime/source hashes.

Reproduce with `cargo test` and `cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin generation-quality`, then the frozen `run.py` in a new evidence directory. Exact measured command arrays are retained. The binary supports existing per-item checkpoint resume with prompt/config/cap verification; no resume or repeated generation was needed in the measured run.
