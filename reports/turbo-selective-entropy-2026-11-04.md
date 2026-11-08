# Turbo selective entropy experiment, 2026-11-04

**No hybrid qualifies; retain the baseline and stop.** All five predetermined groups regress aggregate top1. FFN-down gives the smallest discovery KL (0.222386126 versus baseline 0.223127631), but top1 falls from 59/64 to 58/64. No candidate was sent to validation, and no production rerun followed. No default, quantizer, calibration, kernel, or runtime state change. No sweep, mixture tuning, new model download, commit, or cleanup.

Evidence: `/private/tmp/vxg-real-qualification-20261104/turbo-selective-entropy-20261104`. Source base commit `07f26e90dfbcb8d0160793e2bb7cd0e2476f90b6` plus existing dirty worktree; preserved before-state/diff in evidence. Standalone helper: `crates/vxg-tensor/src/bin/turbo-hybrid.rs`.

## Frozen design and integrity

`plan.txt` was written before any successful candidate creation. Five disjoint groups, one at a time; all151 EC tensors covered exactly once, zero excluded. Discovery ranks aggregate mean directional KL P_source || P_candidate across all four old prompts ×16 A-selected teacher steps; eligibility requires strictly lower KL than baseline AND aggregate top1 >=59/64. At most one eligible candidate, ordered by KL then declared group order, could enter reused prior fixed new4 validation. Validation would require KL<0.362075121 and top1>=54/64; it is reused evidence, not new independent validation. Only a validation success could trigger production128/presence0 and math17×23/cap256. These later gates were not reached.

No re-encoding: the helper parses and validates both archives with existing read_vxt_model, requires the zero archive metadata to equal baseline after removing the sole scale0 marker, and rejects changed non-Turbo tensors or mismatched name/shape/codec headers. It copies the baseline encoded metadata values verbatim and adds diagnostic group/source-FNV provenance. Every selected section is copied verbatim from zero; every nonselected section is copied verbatim from baseline. Alignment/whole-archive FNV are rebuilt; original section FNV tags are retained. Each completed archive is reread with all tensor/container integrity checks; reconstructed LE f32 FNV checksums are compared tensor by tensor against the chosen source, row by row in memory. Independent final-byte-audit.py additionally verifies exact encoded bytes for all320 sections and original metadata bytes. All checks passed for all five. No giant decoded archive is written.

Config/tokenizer, smoothing/norm RawF32 vectors and lossless controls remain exactly baseline. Source baseline/zero structural inventories were compared for all320 tensors: name, rows, cols, padded_cols, kind, subtype, storage and decoded GPU plane byte counts all identical. Every candidate retains151EC-Turbo4,167RawF32,2BP16-BF16 and **741157484 decoded plane bytes**. This is a structural decoded-plane comparison, not a GPU allocation or bandwidth measurement. The entropy experiment altered only lambda and retained existing smoothing; this experiment substitutes complete tensors without mixing tensor transforms.

Three release helper tests passed: exact nonselected-section preservation, shape mismatch rejection, metadata mismatch rejection (`helper-tests-final.log`). Initial metadata-order rejection was retained and fixed before any candidate was created. Measured source snapshot/binary are retained; final source differs only by rustfmt formatting.

## Discovery samples

Retained original source A traces and baseline C traces from turbo-quality-split-20261104 were reused; no source model/reference run was repeated. All candidate runs required and reported Apple Metal GPU (Apple M2 Max, VXT-Turbo Pipelined), separate sequential processes. Identical prompt IDs and source teacher IDs asserted per case, all logits finite. Float64 stable softmax offline analysis retains every per-step RMSE/KL/cosine/top1 in discovery-comparison.json. All cases included; no cherry exclusions.

| Branch | Mean KL | Mean RMSE | Cosine | Top1 /64 |
|---|---:|---:|---:|---:|
| baseline | 0.223127631 | 1.420764781 | 0.840137687 | 59 |
| embedding-head | 0.223353300 | 1.395510562 | 0.845999162 | 58 |
| ffn-gate-up | 0.224490866 | 1.420153346 | 0.839764273 | 57 |
| ffn-down | 0.222386126 | 1.452800607 | 0.834682761 | 58 |
| attention-input | 0.266888393 | 1.462403371 | 0.830193461 | 58 |
| output-projection | 0.254737435 | 1.447460101 | 0.831411470 | 58 |

KL ranking: ffn-down, embedding-head, ffn-gate-up, output-projection, attention-input. No eligible winner.

| Branch | Case | KL | RMSE | Top1 /16 |
|---|---|---:|---:|---:|
| baseline | tr | 0.179346942 | 1.298688459 | 16 |
| baseline | en | 0.395213471 | 1.495483200 | 14 |
| baseline | code | 0.075581409 | 1.588414570 | 16 |
| baseline | math | 0.242368700 | 1.300472895 | 13 |
| embedding-head | tr | 0.192900159 | 1.322614640 | 16 |
| embedding-head | en | 0.384407690 | 1.388016445 | 13 |
| embedding-head | code | 0.118116073 | 1.583573212 | 15 |
| embedding-head | math | 0.197989276 | 1.287837950 | 14 |
| ffn-gate-up | tr | 0.170959508 | 1.362003445 | 16 |
| ffn-gate-up | en | 0.366940053 | 1.395859947 | 13 |
| ffn-gate-up | code | 0.161627467 | 1.644253896 | 15 |
| ffn-gate-up | math | 0.198436435 | 1.278496098 | 13 |
| ffn-down | tr | 0.123000355 | 1.385832326 | 16 |
| ffn-down | en | 0.445617630 | 1.543440684 | 13 |
| ffn-down | code | 0.082050891 | 1.583922403 | 16 |
| ffn-down | math | 0.238875628 | 1.298007017 | 13 |
| attention-input | tr | 0.224688602 | 1.312829560 | 16 |
| attention-input | en | 0.426381982 | 1.606674394 | 14 |
| attention-input | code | 0.101955723 | 1.625431297 | 16 |
| attention-input | math | 0.314527267 | 1.304678231 | 12 |
| output-projection | tr | 0.238029239 | 1.320307015 | 15 |
| output-projection | en | 0.447962576 | 1.439299090 | 13 |
| output-projection | code | 0.117023609 | 1.698918523 | 16 |
| output-projection | math | 0.215934318 | 1.331315777 | 14 |

Exact discovery prompts:

- tr: Türkiye'nin başkenti neresidir? Ankara hakkında tek cümle yaz.
- en: Explain in two sentences why memory bandwidth limits language model inference.
- code: Write a Python function that returns the sum of even numbers in a list.
- math: What is 17 times 23? Show a short calculation.

The diagnostic also emits an ordinary full-logit greedy math sample, presence penalty absent (=0), cap256, for each candidate. These are supplementary outputs, not selection criteria or production speed samples:

| Branch | Emitted tokens | Stop ID | Math output |
|---|---:|---:|---|
| embedding-head | 67 | 248046 | 17 multiplied by 23 equals **401**.<br><br>**Short calculation:**<br>17 × 23 = 17 × (20 + 3) = 17 × 20 + 17 × 3 = 340 + 51 = **401** |
| ffn-gate-up | 125 | 248046 | To find the result of 17 multiplied by 23, we can use the standard multiplication formula:<br><br>17 × 23 = 391<br><br>Here is the short calculation breakdown:<br>- 10 × 20 = 200<br>- 10 × 23 = 230<br>- 7 × 20 = 140<br>- 7 × 23 = 161<br><br>Adding these together: 200 + 230 + 140 + 161 = 391 |
| ffn-down | 69 | 248046 | 17 multiplied by 23 is **401**.<br><br>Here is the short calculation:<br>17 × 23 = 17 × (20 + 3) = 17 × 20 + 17 × 3 = 340 + 51 = 401. |
| attention-input | 11 | 248046 | 17 × 23 = 411 |
| output-projection | 51 | 248046 | 17 × 23 = 411<br><br>**Short calculation:**<br>17 × 20 = 340<br>17 × 3 = 51<br>340 + 51 = 411 |

## Storage and identities

| Candidate | Selected EC | Archive bytes | MiB | Delta bytes vs baseline | SHA256 |
|---|---:|---:|---:|---:|---|
| embedding-head | 1 | 637243982 | 607.723 | +4531008 | `d8d6140abea22dbf2ed9a008e4c728e264ea336562560896d34792ea38e53893` |
| ffn-gate-up | 48 | 640448206 | 610.779 | +7735232 | `ba52fa1b94ced76305fbdca778aaeb980405a91ff7c8ce9302be301a6653804a` |
| ffn-down | 24 | 635168974 | 605.744 | +2456000 | `da713587aecf2e4aca9a1347234d8bb160d3923654e4042b65a6bd5952840e0d` |
| attention-input | 54 | 634165902 | 604.788 | +1452928 | `b973c5df30acf1bd4484e195bf8faca41949e84e62fce4658693db8fa8393d0e` |
| output-projection | 24 | 632765902 | 603.453 | +52928 | `c70b1537d3ac2f320e216017c9153646ea4d31dd3a7d15801e6a33ce6124b3ef` |

All candidates are larger than baseline632712974bytes (603.402MiB); metadata/alignment overhead is included. Baseline and zero source identities matched prior reports before jobs:

- `/private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf`: 3775709216 bytes; SHA256 `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1`.
- `/private/tmp/vxg-real-qualification-20261104/turbo-baseline/Qwen3.5-2B-turbo.vxt`: 632712974 bytes; SHA256 `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71`.
- `/private/tmp/vxg-real-qualification-20261104/turbo-entropy-ablation-20261104/Qwen3.5-2B-turbo-zero.vxt`: 648940686 bytes; SHA256 `6ef87e2b9804b5a5186687cef8449897c5905ee4adc33cae83518a9c550d44b8`.
- `/private/tmp/vxt-stop-target/release/turbo-hybrid`: 614896 bytes; SHA256 `d0987ac304389032d168f9f5e8ca7af4697b8308c808ef14127b984ce949e798`.
- `/private/tmp/vxg-real-qualification-20261104/turbo-entropy-ablation-20261104/quality-ablation`: 3301152 bytes; SHA256 `0c7383556d3a7a09c646e54bcdca07cd189b02b627729c38015e3e5fbf8e4cc8`.
- `/Users/mapletechnologies/Desktop/big_projects/vxg/crates/vxg-tensor/src/bin/turbo-hybrid.rs`: 5432 bytes; SHA256 `874a436717a02a91749e61e16582a98468941797aa645a10dc686c173e32be43`.

`final-helper-sha256.json` identifies the corrected measured helper before successful creation; `turbo-hybrid-measured.rs` and `turbo-hybrid` retain it. `evidence-sha256.json` hashes candidates, audits, logs, tools, all new traces and reused old teacher/baseline trace files. `final-byte-audit.json` retains every tensor check.

## Exact group lists

### embedding-head: 1 EC, 198656000 decoded plane bytes

`token_embd.weight`.

### ffn-gate-up: 48 EC, 235929600 decoded plane bytes

`blk.0.ffn_gate.weight`, `blk.0.ffn_up.weight`, `blk.1.ffn_gate.weight`, `blk.1.ffn_up.weight`, `blk.10.ffn_gate.weight`, `blk.10.ffn_up.weight`, `blk.11.ffn_gate.weight`, `blk.11.ffn_up.weight`, `blk.12.ffn_gate.weight`, `blk.12.ffn_up.weight`, `blk.13.ffn_gate.weight`, `blk.13.ffn_up.weight`, `blk.14.ffn_gate.weight`, `blk.14.ffn_up.weight`, `blk.15.ffn_gate.weight`, `blk.15.ffn_up.weight`, `blk.16.ffn_gate.weight`, `blk.16.ffn_up.weight`, `blk.17.ffn_gate.weight`, `blk.17.ffn_up.weight`, `blk.18.ffn_gate.weight`, `blk.18.ffn_up.weight`, `blk.19.ffn_gate.weight`, `blk.19.ffn_up.weight`, `blk.2.ffn_gate.weight`, `blk.2.ffn_up.weight`, `blk.20.ffn_gate.weight`, `blk.20.ffn_up.weight`, `blk.21.ffn_gate.weight`, `blk.21.ffn_up.weight`, `blk.22.ffn_gate.weight`, `blk.22.ffn_up.weight`, `blk.23.ffn_gate.weight`, `blk.23.ffn_up.weight`, `blk.3.ffn_gate.weight`, `blk.3.ffn_up.weight`, `blk.4.ffn_gate.weight`, `blk.4.ffn_up.weight`, `blk.5.ffn_gate.weight`, `blk.5.ffn_up.weight`, `blk.6.ffn_gate.weight`, `blk.6.ffn_up.weight`, `blk.7.ffn_gate.weight`, `blk.7.ffn_up.weight`, `blk.8.ffn_gate.weight`, `blk.8.ffn_up.weight`, `blk.9.ffn_gate.weight`, `blk.9.ffn_up.weight`.

### ffn-down: 24 EC, 117964800 decoded plane bytes

`blk.0.ffn_down.weight`, `blk.1.ffn_down.weight`, `blk.10.ffn_down.weight`, `blk.11.ffn_down.weight`, `blk.12.ffn_down.weight`, `blk.13.ffn_down.weight`, `blk.14.ffn_down.weight`, `blk.15.ffn_down.weight`, `blk.16.ffn_down.weight`, `blk.17.ffn_down.weight`, `blk.18.ffn_down.weight`, `blk.19.ffn_down.weight`, `blk.2.ffn_down.weight`, `blk.20.ffn_down.weight`, `blk.21.ffn_down.weight`, `blk.22.ffn_down.weight`, `blk.23.ffn_down.weight`, `blk.3.ffn_down.weight`, `blk.4.ffn_down.weight`, `blk.5.ffn_down.weight`, `blk.6.ffn_down.weight`, `blk.7.ffn_down.weight`, `blk.8.ffn_down.weight`, `blk.9.ffn_down.weight`.

### attention-input: 54 EC, 142540800 decoded plane bytes

`blk.0.attn_qkv.weight`, `blk.0.attn_gate.weight`, `blk.1.attn_qkv.weight`, `blk.1.attn_gate.weight`, `blk.10.attn_qkv.weight`, `blk.10.attn_gate.weight`, `blk.11.attn_k.weight`, `blk.11.attn_q.weight`, `blk.11.attn_v.weight`, `blk.12.attn_qkv.weight`, `blk.12.attn_gate.weight`, `blk.13.attn_qkv.weight`, `blk.13.attn_gate.weight`, `blk.14.attn_qkv.weight`, `blk.14.attn_gate.weight`, `blk.15.attn_k.weight`, `blk.15.attn_q.weight`, `blk.15.attn_v.weight`, `blk.16.attn_qkv.weight`, `blk.16.attn_gate.weight`, `blk.17.attn_qkv.weight`, `blk.17.attn_gate.weight`, `blk.18.attn_qkv.weight`, `blk.18.attn_gate.weight`, `blk.19.attn_k.weight`, `blk.19.attn_q.weight`, `blk.19.attn_v.weight`, `blk.2.attn_qkv.weight`, `blk.2.attn_gate.weight`, `blk.20.attn_qkv.weight`, `blk.20.attn_gate.weight`, `blk.21.attn_qkv.weight`, `blk.21.attn_gate.weight`, `blk.22.attn_qkv.weight`, `blk.22.attn_gate.weight`, `blk.23.attn_k.weight`, `blk.23.attn_q.weight`, `blk.23.attn_v.weight`, `blk.3.attn_k.weight`, `blk.3.attn_q.weight`, `blk.3.attn_v.weight`, `blk.4.attn_qkv.weight`, `blk.4.attn_gate.weight`, `blk.5.attn_qkv.weight`, `blk.5.attn_gate.weight`, `blk.6.attn_qkv.weight`, `blk.6.attn_gate.weight`, `blk.7.attn_k.weight`, `blk.7.attn_q.weight`, `blk.7.attn_v.weight`, `blk.8.attn_qkv.weight`, `blk.8.attn_gate.weight`, `blk.9.attn_qkv.weight`, `blk.9.attn_gate.weight`.

### output-projection: 24 EC, 39321600 decoded plane bytes

`blk.0.ssm_out.weight`, `blk.1.ssm_out.weight`, `blk.10.ssm_out.weight`, `blk.11.attn_output.weight`, `blk.12.ssm_out.weight`, `blk.13.ssm_out.weight`, `blk.14.ssm_out.weight`, `blk.15.attn_output.weight`, `blk.16.ssm_out.weight`, `blk.17.ssm_out.weight`, `blk.18.ssm_out.weight`, `blk.19.attn_output.weight`, `blk.2.ssm_out.weight`, `blk.20.ssm_out.weight`, `blk.21.ssm_out.weight`, `blk.22.ssm_out.weight`, `blk.23.attn_output.weight`, `blk.3.attn_output.weight`, `blk.4.ssm_out.weight`, `blk.5.ssm_out.weight`, `blk.6.ssm_out.weight`, `blk.7.attn_output.weight`, `blk.8.ssm_out.weight`, `blk.9.ssm_out.weight`.

## Commands and resource limits

```sh
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin turbo-hybrid
cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin turbo-hybrid
python3 /private/tmp/vxg-real-qualification-20261104/turbo-selective-entropy-20261104/run-discovery.py
/private/tmp/vxg-torch-exp-venv/bin/python /private/tmp/vxg-real-qualification-20261104/turbo-selective-entropy-20261104/analyze.py
python3 /private/tmp/vxg-real-qualification-20261104/turbo-selective-entropy-20261104/final-audit.py
```

Do not rerun create-new driver in the retained evidence directory; use a fresh directory. Exact ten expanded create/quality commands are in commands.json. Driver log retains elapsed samples:

```text
embedding-head create 0 42.32402992248535
embedding-head quality 0 7.6822779178619385
ffn-gate-up create 0 42.223493814468384
ffn-gate-up quality 0 6.2612810134887695
ffn-down create 0 42.4314968585968
ffn-down quality 0 6.16953182220459
attention-input create 0 41.93274402618408
attention-input quality 0 5.859166145324707
output-projection create 0 46.15515494346619
output-projection quality 0 8.001272201538086
```

Initial disk available36GiB; memory_pressure free77%, later79–80%. Scoped process observation showed only the driver and one builder; helper RSS about4.1GiB, single CPU about99%. GPU inference processes were sequential; no candidate hashing/audit overlapped GPU timing. Sandbox ps was initially denied; scoped escalated process checks succeeded. Resource samples are sparse, not peak monitoring; unrelated host load is not controlled. No stable TPS claim.

Recommendation: retain baseline. A tiny KL improvement from FFN-down does not satisfy the pre-registered top1 safeguard. No validation/production follow-up, no mixtures or extra lambda values. Broad benchmarks/perplexity/long-context/repeated controlled throughput remain outside this bounded experiment.
