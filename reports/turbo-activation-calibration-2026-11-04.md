# Turbo observed activation calibration, 2026-11-04

One diagnostic candidate; default unchanged. Evidence directory: `/private/tmp/vxg-real-qualification-20261104/turbo-activation-20261104`. Existing dirty tree, baseline archives, codebook, row-adaptive lambda (multiplier1), smoothing, FP16 scales, GPU storage/layout and runtime retained. No agents spawned by this experiment, commits, cleanup, training, new downloads, or parameter search.

The opt-in `turbo-activation` diagnostic exports the actual Rust smoothing manifest and accepts an observed profile only against an exactly matching source-FNV/version/coordinate/policy/full-group manifest. It rejects missing/extra groups, mismatched columns, nonuniform counts, call/sample mismatch, zero or more than256 samples, nonfinite/negative/zero-total second moments. The ordinary public pack API delegates to the same path with no profile and adds no metadata. A profile adds only a diagnostic fingerprint metadata entry and changes importance for covered matrices.

Official Transformers CPU FP32 model is constructed using the already verified strict320-tensor original SafeTensors mapping and tied head. Eight distinct calibration texts, frozen token IDs, module names, policy and gate are saved before model execution in `profile.json.plan.json`; each text is truncated to32tokens, no generated tokens. Forward prehooks observe one representative linear pre-input per shared-input family, streaming only block32 second moments. They return None and operate out of place on a detached double conversion; no full activations retained, backward, full Hessian or learned Hessian claim. This experiment does not include a separate hooks-off numerical equivalence run.

Coordinate alignment: Rust exports its existing per-layer sparse g. Observer reads original model linear input x and accumulates mean block32 `(x/g)^2`; original source computation remains unchanged. Pack multiplies corresponding reader columns by g and divides their input RMSNorm gains by g, so x/g is the matching input coordinate. Rademacher sign and orthonormal FWHT32 preserve each block's energy. This preserves the existing block-isotropic approximation, rather than measuring correlations within blocks. Existing fixed policy is `0.45 + 0.55*relative_block_energy`, clamp `[0.65,2]`, then active-column mean1 and padding1. Clamp is before mean normalization. Equal16 key/value recurrent heads in this exact source config mean the authoritative converter applies no recurrent out-projection input column permutation. The prior tiny source/GGUF `ssm_a` exponentiation ULP difference remains an inherited caveat.

Coverage:24 attention-input groups,24 FFN gate/up groups,18 recurrent out-projection groups. Attention group shares input among recurrent qkv/z or full-attention q/k/v; FFN gate/up share source input. Tied embedding/head retains analytical baseline importance (embedding lookup is not a linear input), as do FFN-down and six full-attention output projections. No complete-model activation calibration is claimed.

Pre-registered gate: compare candidate actual Turbo GPU to frozen originalA discovery4x16teacher traces; advance only if meanKL improves and top1 is no worse than baseline59/64. Only then use the prior validation4x16set, requiring meanKL improvement and top1>=54/64. Only a double pass allows one128cap production speed diagnostic. No tuning or default promotion follows this bounded experiment.

The candidate **fails the pre-registered discovery top1 safeguard**. Default retained. No validation, production timing, second candidate or tuning followed.

| Frozen discovery4x16 | Baseline | Observed activation candidate |
|---|---:|---:|
| Mean KL, nats | 0.223127630650 | 0.204912515004 |
| Top1 equal | 59/64 | 58/64 |
| Mean RMSE | 1.420764780983 | 1.425500597137 |
| Mean cosine | 0.840137687433 | 0.838138536229 |
| Archive bytes | 632712974 | 632564558 |

Candidate is148416bytes smaller (-0.02346%),603.26MiB. KL improves, but top1 drops, RMSE and cosine worsen. The one-of64top1difference has unknown broader significance; failing this fixed gate does not prove global inferiority. These64steps do not quantify intelligence retention. Directional KL is P_originalA || P_candidate using float64 stable softmax; complete vocabulary logits finite, frozen prompt and teacher IDs asserted identical.

| Case | Baseline KL / top1 | Candidate KL / top1 |
|---|---|---|
| Turkish | 0.179346942367 /16 | 0.193585517634 /16 |
| English | 0.395213471060 /14 | 0.339768638352 /14 |
| Python | 0.075581409392 /16 | 0.076912341569 /16 |
| Arithmetic | 0.242368699779 /13 | 0.209383562462 /12 |

Actual backend is `Apple Metal GPU (Apple M2 Max, VXT-Turbo Pipelined)`; retained measured binary SHA below. Diagnostic also performs its existing ordinary greedy arithmetic generation, cap256: candidate emits52tokens, stops on248046 at state_pos79 and answers401 (incorrect), despite writing340+51. Baseline had answered411, originalA391. No arithmetic-specific selection or claim follows this observation.

Source run completed8texts/171inputtokens,66groups with171samples and171calls each; all4224block moments finite and positive (min0.000319590584,max8.986657485). Forward time192.50s, total including provenance hashing194.91s. macOS `ru_maxrss`11643781120bytes (~10.84GiB). Sparse system free samples were75%initial,64%during early source execution,68%later,72%during pack,70%after evaluation; these do not establish allocation peaks. Disk31GiBavailable after retained artifacts. Only one model computation process intentionally active at a time; a small helper build overlapped early source calibration, so source elapsed is not a performance benchmark. Sandbox process inventory was denied; unrelated load not fully inventoried.

Validation completed: release Metal helper build;4profile/normalization/rejection tests;2existing entropy/default legacy-byte tests (11roles); independent x/g algebra and sign/FWHT block-energy check; real source run; one candidate pack; container FNV verification all320tensors; actualGPU discovery traces. Byte audit proves every tensor header (including kind/subtype/dimensions) unchanged, all120covered EC tensors changed, every noncovered/nonEC section byte-exact baseline, original metadata byte-exact plus one profile marker. Structural inventory retains151EC-Turbo4/167F32/2BP16-BF16 and741157484decoded plane bytes, exact for all320against baseline. These are decoded storage planes, not measured GPU allocations/traffic. No giant default repack was performed.

Broad perplexity/downstream tasks, long context/state drift, full-model importance calibration, stable TPS and repeated throughput remain unrun. Failure of the selected safeguard ends this experiment; no default promotion.

| Artifact | SHA256 |
|---|---|
| GGUF source | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` |
| Baseline archive, rechecked unchanged | `64926e91536a9f0453a40334eccb34cd7d6db911a1c8fc36a7c343aa6ac98b71` |
| Observed candidate archive | `1efbeb2db44f6d1e88a5fe4de3c8bfc4972007bfd3c8e7792f31e2874b5c8a7d` |
| Observed profile | `99e7e1f7d80b62eb862cb805197db8c87e5dff284aa3c3ccc1b4f82c28a04f0f` |
| Exported manifest | `580e135de4d4c55afef3d7b03878e26aa6c71228384a1b2bc53ea359a39c292a` |
| Diagnostic helper | `f4781038a634f0824f599bf81ab04e995f50c35573c54e521e0b279f5b470e5c` |
| Measured quality binary | `0c7383556d3a7a09c646e54bcdca07cd189b02b627729c38015e3e5fbf8e4cc8` |

`artifact-identities.json`, `code-sha256.json`, exact source snapshots, retained helper/quality binaries, profile provenance (official source/config/tokenizer/modeling/calibration-plan hashes), raw logits/per-step comparison, byte audit, inventory, build/test/observer/pack/verify/GPU logs remain in evidence. Base commit and dirty diff/status retained. Stop-state code was not edited.

Reproduction uses new output paths, never overwrites these artifacts:

```sh
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --lib activation_tests
cargo test -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --lib entropy_ablation_tests
cargo build -p vxg-tensor --release --features metal --target-dir /private/tmp/vxt-stop-target --bin turbo-activation
# helper modes: manifest SOURCE.gguf NEW_MANIFEST.json
#               pack SOURCE.gguf PROFILE.json NEW_CANDIDATE.vxt
# source observer: existing venv Python scripts/qwen35_activation_profile.py ROOT MANIFEST NEW_PROFILE
# quality: retained quality-ablation C CANDIDATE eval-prompts.json NEW_DISCOVERY/C
# NEW_DISCOVERY/A must point to frozen turbo-quality-split-20261104/A
```

