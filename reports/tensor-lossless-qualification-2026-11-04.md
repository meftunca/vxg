# Lossless tensor qualification, 2026-11-04

Code base: `07f26e90dfbcb8d0160793e2bb7cd0e2476f90b6` plus preserved dirty working tree. This report covers newly added verifier and synthetic long-context coverage, not the entire dirty candidate. Host memory query reported 32 GiB, 67% free before tests. Tests ran sequentially; no performance benchmark was run.

## Reproduce

```sh
cargo test -p vxg-tensor --bin verify-lossless --target-dir /private/tmp/vxg-qualification-target -- --nocapture
VXG_REQUIRE_METAL=1 cargo test -p vxg-tensor --features metal --test llm_gpu_cpu_parity hybrid_lossless_long_context --target-dir /private/tmp/vxg-qualification-target -- --nocapture
cargo run -p vxg-tensor --release --bin verify-lossless -- SOURCE.gguf ARCHIVE.vxt
shasum -a 256 SOURCE.gguf ARCHIVE.vxt
```

On macOS the Metal-required test needs native device access. It refuses to silently pass CPU-only when `VXG_REQUIRE_METAL` is set. Actual GPU backend is asserted to contain `1-CmdBuf Fused`; CPU reference and compressed CPU model use two threads.

## Weight verifier

`src/bin/verify-lossless.rs` checks every source tensor, exact name/count and logical matrix shape. It compares BF16/F16 BP16 storage using raw source u16 words, including NaNs and signed zeros; other supported storage is compared using source-derived f32 bits. F32 uses its original raw source bits. The independent F16 arithmetic oracle passed all 65,536 patterns. Both BF16 and F16 BP16 paths passed exhaustive raw-word tests and rejected a one-bit source change.

Three tests passed. Additional rejection coverage: corrupt checksum, omitted tensor, duplicate tensor name, changed name, wrong shape, changed source value. Unsupported quantized GGUF source kinds fail closed because this verifier does not yet implement an independent quantization oracle. It shares the GGUF container parser and VXT decoder with production, but does not call the packer or its source extraction functions. It currently checks weights, not model configuration/tokenizer equivalence. Whole-model VXT parsing retains all compressed tensors, plus one decoded tensor at a time; this is not a bounded-memory streaming verifier.

## Synthetic hybrid recurrent/attention engine

256 deterministic teacher-forced tokens per dtype, two-layer qwen35 architecture containing DeltaNet recurrent state and gated GQA. RawF32 reference weights are independently built from the same generated source words, without decoding packed matrices. Every step requires finite logits, RMSE < 1e-4, cosine > 0.99999, and identical top-1 for compressed CPU and actual GPU against raw CPU reference.

| Source | CPU max RMSE | CPU max absolute | GPU max RMSE | GPU max absolute |
|---|---:|---:|---:|---:|
| BF16 | 2.190217611267491e-6 | 7.1525574e-6 | 2.4662170189751123e-6 | 8.34465e-6 |
| F16 | 4.175244117814566e-6 | 1.1563301e-5 | 3.307650171612003e-6 | 9.834766e-6 |

CPU-only run passed; subsequent Metal-required run passed both dtypes (17.93 s test duration, not TPS). Initial CPU exact-logit assertion failed at step zero: source weights can be bit-exact while different GEMV accumulation orders produce small floating-point differences. This test explicitly qualifies numerical inference parity, not bit-identical inference. Raw16 source-weight assertions remain exact. Synthetic tests do not establish real-model quality or throughput.

## Open real-model gates

Targeted filename searches of repository docs/scripts/reports, Hugging Face cache, LM Studio model cache, and named VXG temporary artifact directories found no Qwen BF16/F16 language-model trunk. LM Studio cache contains Qwen3.8-27B IQ1_S and an F16 vision projector; neither is the requested full-precision language-model source. No model download or deletion occurred.

Remaining: obtain exact source/archive pair; record source/archive SHA256, all relevant source hashes and dirty diff identity; run all-tensor verifier; real deterministic teacher forcing over representative long contexts and recurrent state; independently validate model execution against an external runtime; clean sequential load/prefill/decode/RSS measurement excluding logit-comparison work. Current `eval` timing includes comparison overhead and must not be cited as isolated decode TPS. No archive ratio, real model memory, TPS, external-runtime correctness, or maximum-compression claim is established by this report.


## SafeTensors verifier extension

The same command now accepts a single BF16/F16/F32 `.safetensors` source file (content detected; a GGUF magic selects GGUF). This is a source-weight verifier only; HF configuration, tokenizer, shard manifests, and real inference loading remain outside this extension.

```sh
cargo run -p vxg-tensor --release --bin verify-lossless -- SOURCE.safetensors ARCHIVE.vxt
cargo test -p vxg-tensor --bin verify-lossless --target-dir /private/tmp/vxg-qualification-target -- --nocapture
```

SafeTensors shape handling is row-major: last axis is columns; product of preceding axes is rows. Vectors are `1 x length`; scalar shape `[]` is `1 x 1`; zero-sized tensors are explicitly rejected. VXT stores the flattened matrix shape, so original higher-rank axis boundaries are not independently preserved by VXT. `__metadata__` is parsed as a string map but is not compared to VXT metadata; successful verification proves tensor values and the stated flattened shapes, not model metadata equality.

Verifier-local schema parsing is deliberate: the production SafeTensors parser accepts duplicate tensor fields, trailing header data, permissive escapes, and uses unchecked shape products. The verifier instead checks unique names/fields, checked dimensions and byte sizes, contiguous non-overlapping offsets covering all data, required dtype/shape/offset fields, integer syntax, full header consumption, UTF-8, truncation and a 100MB header cap. No recursive JSON parsing is used. Unknown tensor fields and non-string metadata values fail closed. Literal UTF-8 names and standard short JSON escapes are accepted; `\u` Unicode escapes are explicitly unsupported and rejected, including in metadata, rather than silently reinterpreted. This strict subset can reject otherwise valid SafeTensors files using escaped Unicode; conversion to literal UTF-8 would be needed before verification. No production parser or dependency changes were made.

One focused final run passed four tests, including existing GGUF tests. New positives cover BF16 and F16 raw-word equality at higher-rank flattened shape and two F32 tensors. New negatives cover a changed source bit, unsupported dtype, wrong flattened shape, missing/renamed/duplicate tensors, duplicate fields, overlap, out-of-bounds offsets, shape overflow, trailing JSON, trailing commas, invalid integer syntax, Unicode escapes, zero-size shapes, truncated file and overflowing header length. Existing checksum/shape/value/omission rejection remains covered. No GPU rerun was needed for this source-parser-only extension.

A targeted macOS filename index query for Qwen3.5 BF16 filenames and `.vxt` archives returned no paths. No full model artifacts were obtained or downloaded.


## Real Qwen3.5-2B qualification after authorized download

The earlier missing-source gate was resolved by a user-authorized download. Source: [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae), immutable repository revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`, file `Qwen3.5-2B-BF16.gguf`. Its card identifies `Qwen/Qwen3.5-2B` as base model. Official model metadata was retained separately at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. This proves fidelity to the downloaded third-party GGUF source; direct equivalence to official SafeTensors is not yet established.

Artifacts and logs: `/private/tmp/vxg-real-qualification-20261104/`. Includes `official-api.json`, `gguf-api.json`, both model cards, official config/index, `source.sha256`, `archive.sha256`, `dtype-inventory.json`, `pack.log`, `verify.log`, `runtime-256-32.log`, `code-identity.txt` (all crate source hashes and dirty status), and `working-tree.diff`. No artifacts were deleted. Download completed in about 35 seconds. Before download disk had 77GiB available; memory query showed 61% free of32GiB.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| GGUF source | 3,775,709,216 | `dc11be2ca4519954a3e4b013ef59497b5522209987d67891851113eda89619f1` |
| Lossless VXT | 2,675,770,830 | `1129550ba3ce060095dd25363136a15027c40ac2d9ba5295368b41c847d4dd96` |

Source SHA256 matches the HF API declared LFS hash. Archive ratio is **1.411073x** (3600.80MiB ->2551.81MiB). Explicit `--mode lossless --threads 4` packing took13.41s reported internally. Pack timing was sandboxed, so native memory counters were unavailable for that stage.

### Exact source-weight gate

**PASS all320 source tensors**:187 BF16 tensors and133 F32 tensors. Compared **1,881,276,416 raw BF16 words** and **548,672 original F32 bit patterns**, with exact name/count/matrix-shape matching. No tensor was omitted by this source count check. This run took18.70s; native `/usr/bin/time -l` reported peak memory footprint3,746,257,136B and maxRSS8,047,542,272B (RSS includes mappings; these are distinct OS counters).

### Real-model recurrent/attention teacher forcing

New `qualify-model` binary loads independent GGUF RawF32 reference weights, records a deterministic greedy trace, releases that model, then loads lossless VXT and teacher-forces the same reference token sequence. Both use actual **Apple M2 Max, 1-CmdBuf Fused Metal**; no competing model benchmark was observed before the run. Both config serialization and token IDs over the full qualification text match. Tokenizer equality is checked for this text, not exhaustively.

Context:256 tokens, derived from repeated fixed English/Turkish text;32 logit comparisons and31 timed decode forwards. Both24-layer real model recurrent and attention state are exercised. All logits finite; top1 matched32/32. Max absolute difference `3.147125244e-5`, maximum per-step RMSE `5.312169160e-6`, max KL `1.843437340e-12`, minimum cosine `0.999999999998`. There were **7,635,894 unequal f32 logit bit patterns** across7,946,240 comparisons. Thus source weights are bit-exact, but inference logits are not bit-identical across the different raw/compressed kernels. The prespecified numerical gate (RMSE<=1e-4, cosine>=0.99999, all top1 equal) passed. Detailed per-step teacher token, RMSE, KL, cosine and top1 results are retained in the runtime log. Individual full logit arrays and individual mismatch bit values were not retained by this initial tool.

| Single sample | Load seconds | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Independent GGUF RawF32 | 13.295611 | 35.478047 | 33.947077 |
| Lossless VXT | 8.464926 | 59.367032 | 45.451368 |

Forward-only decode timers exclude sampling, logit comparisons and printing. This is one sequential sample with **no warmup**, with initial provenance/archive hashing I/O overlapping early source loading; do not treat this as a controlled performance benchmark or maximum TPS qualification. Process peak memory footprint15,262,870,760B and maxRSS9,533,046,784B cover the entire sequential reference+test process and retained reference traces, not separate per-model memory. Total process wall time36.37s. After process completion OS memory query showed80% free.

Reproduce (actual Metal access required):

```sh
cargo build -p vxg-tensor --release --features metal --bins --target-dir /private/tmp/vxg-qualification-target
/private/tmp/vxg-qualification-target/release/vxg-tensor pack-model /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-lossless.vxt --mode lossless --threads 4
/private/tmp/vxg-qualification-target/release/verify-lossless /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-lossless.vxt
/usr/bin/time -l /private/tmp/vxg-qualification-target/release/qualify-model /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-lossless.vxt 256 32
```

Remaining gates: official SafeTensors-to-GGUF provenance/equality, independent external runtime, representative tasks and materially longer contexts, isolated per-model memory, controlled repeated timing. Current results establish exact real downloaded-source weights and bounded numerical engine parity. They do not establish general zero quality loss, maximum compression, or broad performance acceptance.


### Timing reference and clean-repeat clarification

The runtime reference above materializes GGUF source weights as **RawF32**. It is independent of the packer but uses the same VXG inference engine. The measured timing differences therefore compare this RawF32 reference with compressed VXT on this engine; they **do not establish speedup against a native BF16 or ggml/llama.cpp runtime**.

The `qualify-model` tool now prints its fixed gate before execution, requires2..256 comparison steps (at least one measured decode forward), and rejects nonfinite aggregate parity metrics. A focused helper test passed negative top1/threshold/NaN/Inf/empty-count cases and threshold-boundary acceptance. Final source SHA256: `180c61945ed733bdbb3f94e8d14d140158ea09adc24a5ff431fed2bf30742a39` (also retained as `qualify-model-repeat.sha256`). This change does not alter weights, kernels, sampling, qualification inputs or numerical thresholds.


Clean sequential repeat completed successfully after provenance hashing finished. Same256-token context,32 comparisons,31 decode forwards, same models and Metal backend. No explicit warmup; existing OS/driver cache state is uncontrolled. No competing model/cargo/llama process was observed in the targeted process check before starting.

| Clean repeat | Load seconds | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| GGUF materialized RawF32 reference | 12.534575 | 40.846321 | 39.035004 |
| Lossless VXT | 11.639933 | 59.866999 | 45.535841 |

Numerical results exactly repeated the first run's reported statistics: top1=32/32, maxabs3.147125244e-5, maxRMSE5.312169160e-6, maxKL1.843437340e-12, mincos0.999999999998,7,635,894 logit-bit mismatches. The repeat is still a bounded sample, not native-BF16 comparison or broad benchmark qualification. Complete log: `runtime-256-32-clean-repeat.log` in the same artifact directory.

Repeat total wall37.51s, total-process maxRSS14,257,979,392B, total-process peak memory footprint15,262,936,296B. These cover sequential source and test plus retained traces; **per-model/per-load peak RSS was not measured**. No final source/model/kernel changes followed this run. `git diff --check` passed.


## Official SafeTensors text-trunk provenance comparison

Downloaded the official [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) single shard `model.safetensors-00001-of-00001.safetensors` to `official-model.safetensors` in the retained artifact directory. Exact size4,548,221,488B; SHA256 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` matches the pinned HF API LFS hash. Existing official config/index were reused; disk had62GiB free before downloading, and no duplicate official weight download was made.

Independent stdlib-only `scripts/compare_qwen35_source.py` reads JSON/mmap/struct directly, does not import VXG or model runtimes, and uses explicit source-name mapping. It fails on unknown names, missing/extra GGUF tensors, unsupported dtypes/layouts or shape mismatches. `official-gguf-comparison.json` retains all320 per-tensor source/target mappings, shapes, dtypes, transforms, counts and mismatch examples; excluded official names/shapes/counts are explicit. `official-gguf-comparison.log` records the run. There were no unmapped language or GGUF tensors.

Authoritative conversion rules were checked against [llama.cpp conversion/qwen.py](https://github.com/ggml-org/llama.cpp/blob/e853ec9a8c72e91e8fb0eaa68e1dfe02f3c396f0/conversion/qwen.py), [conversion/base.py](https://github.com/ggml-org/llama.cpp/blob/e853ec9a8c72e91e8fb0eaa68e1dfe02f3c396f0/conversion/base.py), and its tensor mapping/constants at revision `e853ec9a8c72e91e8fb0eaa68e1dfe02f3c396f0`. These establish language prefix removal, BF16-to-F32 before transformation, norm+1 except linear-attention norm, conv squeeze, and A_log -> negative torch.exp. V-head permutation is unnecessary for this official model because key/value head counts both equal16; the script explicitly rejects other counts. This pinned current code supports mapping semantics; it does not prove the historical Unsloth conversion used this exact revision.

**Outcome:318/320 language tensors matched the specified independent byte/numerical transformations exactly; two scalar exp-oracle comparisons differ by1ULP.** The strict script returned exit1 and no tolerance was applied.

| Mapping/check mode | Tensor count | Exact tensors |
|---|---:|---:|
| Native BF16 byte identity |187|187|
| Native F32 byte identity |18|18|
| BF16 -> F32 exact widening, identity |18|18|
| BF16 conv singleton-axis squeeze + exact F32 widening |18|18|
| BF16 -> F32 norm +1 |61|61|
| F32 A_log -> negative exp, Python scalar oracle |18|16|

All320 correspond to1,881,825,088 text-trunk elements. The18 exp tensors contain288 scalars;286 of those match the scalar oracle exactly. For the other two, hex values below are little-endian byte representations:

| Tensor/index | Official A_log bits/value | Python `f32(-math.exp(x))` | GGUF F32 | Absolute difference |
|---|---|---|---|---:|
| `blk.1.ssm_a` /2 |`00008a3e` /0.26953125|`cc98a7bf` /-1.3093504905700684|`cd98a7bf` /-1.309350609779358|1.1920928955078125e-7|
| `blk.6.ssm_a` /15 |`00000dc0` /-2.203125|`8237e2bd` /-0.11045743525028229|`8337e2bd` /-0.11045744270086288|7.450580596923828e-9|

Python `math.exp` computes a scalar double result then rounds to F32 here; converter code uses `torch.exp` on F32 tensors, whose implementation/approximation can differ. Thus these two observations **neither certify exact converter equivalence nor establish conversion corruption**. A pinned matching torch.exp conversion oracle remains necessary to resolve those two scalar outputs. No Torch installation, codec modification, tolerance relaxation, or additional inference run was performed.

Explicit scope exclusions:297 visual tensors (331,416,576 elements) and15 MTP tensors (60,828,160 elements), all retained by name in JSON. The official checkpoint has632 tensors overall. Config has tied word embeddings enabled at both outer/text levels; the320 text tensors contain embedding weights but no separate lm_head, and the320 GGUF tensors have no separate output head. That tied head shares token embedding weights. This verifies the ordinary language trunk scope only, not multimodal execution or MTP equivalence.

Reproduce:

```sh
python3 scripts/compare_qwen35_source.py /private/tmp/vxg-real-qualification-20261104/official-model.safetensors /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf /private/tmp/vxg-real-qualification-20261104/official-config.json /private/tmp/vxg-real-qualification-20261104/official-gguf-comparison.json
```

Expected current exit status1 from the two exp-oracle mismatches. A small independent fixture also passed native-BF16 equality, one-bit mismatch detection and unknown-name rejection. `git diff --check` passed. No active download/comparison process remains.

Retained code hashes (`comparison-code.sha256`): script `acb420cc8f3a0a28c81642e9d9ea303e5740fe0877fc2f46ea2aca416893db5d`; converter qwen.py `de079852258ad105a69f61ca6996d184cd5a0eb907b274ba282b0c700423a1bb`; converter base.py `a96584f85a7f81a51cb3cc0759134375077fa39e265f36b72adbc36638a41226`; tensor_mapping.py `ef777c6aef13259a99d8ffdaba2f47cb4f42315b98768dada85c655919d553ee`; constants.py `bb7369cd053790c97b859b1fc6ed2e1ef763a8ddbc370f6f83d1bf84b9b9a53f`.

### Bounded CPU torch.exp probe

Installed pinned `torch==2.9.0` in isolated `/private/tmp/vxg-torch-exp-venv` using Python3.11.18. Torch git build `0fabc3ba44823f257e70ce397d989c8de5e362c1`, macOS26.6.2 arm64, CPU float32, one thread. `scripts/probe_qwen35_torch_exp.py` reads the official F32 bytes directly and evaluates `-torch.exp(x)` separately on each complete16-element tensor, with exact output-byte comparisons and no tolerance or additional rounding. No codec/runtime or NumPy conversion is used; Torch emitted a harmless missing-NumPy initialization warning.

**Result:6/18 tensors and259/288 scalar elements matched;12 tensors containing29 scalar elements differed.** Exit status1. All29 scalar differences were1ULP; maximum absolute difference1.1920928955078125e-7, with30 differing bytes overall. This build does not reproduce the GGUF exp outputs. No alternate Torch version was tried. The original318/320 scalar-oracle result remains unchanged; exact conversion equivalence is unresolved. CPU/platform exp approximations can vary, so this mismatch does not prove conversion corruption. A matching current build would establish reproducibility only, not the historical converter's identity. No model, codec, GPU or inference changes/runs were made for this probe.

Retained `official-gguf-torch-exp.json` includes complete source/output/target bytes for all18 tensors, mismatch indices, Torch build configuration and script hashes; `official-gguf-torch-exp.log` records output. Probe script SHA256 `d3135346e5bb0059638b4bc496feaec46c82086fd33df60ca76cf04657727645`.

```sh
UV_CACHE_DIR=/private/tmp/vxg-torch-uv-cache uv venv --python /opt/homebrew/opt/python@3.11/bin/python3.11 /private/tmp/vxg-torch-exp-venv
UV_CACHE_DIR=/private/tmp/vxg-torch-uv-cache uv pip install --python /private/tmp/vxg-torch-exp-venv/bin/python 'torch==2.9.0'
/private/tmp/vxg-torch-exp-venv/bin/python scripts/probe_qwen35_torch_exp.py /private/tmp/vxg-real-qualification-20261104/official-model.safetensors /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-BF16.gguf /private/tmp/vxg-real-qualification-20261104/official-gguf-torch-exp.json
```

### Native official SafeTensors import and inference (2026-11-04)

Implemented `pack-native`, `verify-native`, and native SafeTensors/VXT runtime
loading without a GGUF dependency. The supported scope is the official
Qwen/Qwen3.5-2B single-file language trunk, using the same pinned revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Downloaded its `tokenizer.json`
(SHA256 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`).
Official model/config originals remain unchanged. A `native-source/` directory
contains symlinks to those three retained originals for direct source loading.

`native_qwen35.rs` validates the supported 2B hybrid configuration and all 320
required native names/shapes/dtypes. VXT1 stores the original names and bits:
284 BF16 tensors use the lossless codec; 36 F32 tensors use raw F32 storage.
A required versioned JSON manifest preserves original full ranks/shapes/dtypes,
including convolution `[6144,1,4]`; raw config/tokenizer JSON is embedded.
297 visual and 15 MTP tensors are excluded deliberately, with names/dtypes/shapes
recorded in the manifest. Norm `+1`, `-exp(A_log)`, and convolution flattening are
runtime-only operations. The archive therefore preserves original native source
weights, rather than describing transformed weights as original lossless data.
Missing shards, incompatible config, unknown language tensors and unsupported
dtypes/shapes fail. Previous VXT files retain their decoding path.

Retained artifact `/private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-native-lossless.vxt`:
2,714,012,317 bytes, SHA256
`ede44e49d549de46205f93212493b7857c6856385b12111d80cfad59495af229`.
Original language payload is 3,763,655,360 bytes: **1.38675x text-payload/archive
ratio**, including archive tokenizer/config overhead. The full official file is
4,548,221,488 bytes and includes excluded visual/MTP data; its 1.67583x ratio is
not a like-for-like compression result. Pack reported22.71s. Its `time -l`
wrapper returned1 because sandbox `sysctl kern.clockrate` was denied after the
successful pack; archive verification subsequently exited0.

`verify-native` independently compared every retained decoded bit against source,
checked names, original shapes/dtypes, flattened dimensions, raw config/tokenizer,
and the exclusion manifest: **all320 passed**, exit0 (`native-verify.log`).
Unit and relevant regression run:18 passed,1 optional official-tokenizer test
ignored by default; the official-tokenizer test was then run explicitly and
passed. It compares13 selected cases against HF `tokenizers==0.22.1` outputs
from the pinned tokenizer, including Turkish/CJK/Arabic, NFC, emoji, numbers,
contractions, whitespace, ChatML and added visual token strings. Unsupported
normalizer, added-token stripping and BPE fallback settings reject. This is
selected-case tokenizer evidence, not exhaustive tokenizer equivalence.
Native tokenization uses validated JSON settings, NFC and the official Unicode
regex via `fancy-regex`; no external model runtime is required. Existing ChatML
formatting is retained; no downloaded Jinja chat-template evaluator is claimed.

Actual Apple M2 Max Metal smoke loaded the native archive on `1-CmdBuf Fused` and
generated2 tokens. A short2-step source/archive logit eval passed; the subsequent
frozen qualification below is the numerical acceptance evidence.

```sh
CARGO_TARGET_DIR=/private/tmp/vxt-native-target cargo build -p vxg-tensor --release --offline
/private/tmp/vxt-native-target/release/vxg-tensor pack-native /private/tmp/vxg-real-qualification-20261104/official-model.safetensors /private/tmp/vxg-real-qualification-20261104/official-config.json /private/tmp/vxg-real-qualification-20261104/official-tokenizer.json /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-native-lossless.vxt
/private/tmp/vxt-native-target/release/vxg-tensor verify-native /private/tmp/vxg-real-qualification-20261104/official-model.safetensors /private/tmp/vxg-real-qualification-20261104/official-config.json /private/tmp/vxg-real-qualification-20261104/official-tokenizer.json /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-native-lossless.vxt
/private/tmp/vxt-native-target/release/qualify-model /private/tmp/vxg-real-qualification-20261104/native-source/model.safetensors /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-native-lossless.vxt 256 32
```

**Frozen actual Metal numerical gate passed**, source and archive loaded
sequentially with256 context tokens and32 teacher-forced steps. Direct native
source bypasses compression/archive decoding and materializes original weights
to RawF32; archive uses the verified native BF16/F32 payloads. Both run the same
native inference implementation. Exact serialized runtime config and full
qualification-text token IDs matched. Gate unchanged: finite logits,
max RMSE<=1e-4, minimum cosine>=0.99999, every top1 match.

- Top1:32/32; max absolute logit error2.646446228e-5.
- Max RMSE5.226542528e-6; max KL1.779687502e-12; min cosine0.999999999998.
- Logit bit mismatches7,629,921: weight bits are exact, inference logits are
  numerically equivalent under the frozen gate, not bit-exact.
- Actual fused Metal backend was required and observed for both paths.

Retained `native-metal-qualification-256ctx-32steps.log` records full per-step
metrics. `native-pack.log`, `native-verify.log`, `native-metal-eval.log`,
`native-vxt-metal-smoke.log`, `native-regressions.log`,
`native-tokenizer-oracle.log`, and `native-artifacts.sha256` retain other evidence.
Qualification binary SHA256 `8c0a3a0ea882f19eccc3d39374e82544fa192c0a51a363ffb17b7b20996a4e3d`;
CLI binary SHA256 `97a8691a3a9b767837da6918993bfe106db0f29a80d7d739fe5ded412354dd3f`.
System memory free was85% before packing,76% before the final GPU gate; disk
space59GiB before work. No model runtime accuracy reference was downloaded/run:
**independent external runtime accuracy, exhaustive tokenization, multimodal/MTP,
and release/performance qualification remain open**. These bounded runs establish
native import, source-bit preservation and same-runtime numerical fidelity.

### Independent official Transformers accuracy reference (2026-11-04)

**The bounded independent runtime gate passed without changing its thresholds.**
The original official SafeTensors checkpoint was evaluated with official
`transformers==5.22.1` `Qwen3_5ForCausalLM`, CPU float32, eager full attention,
two Torch threads, and sequential cached one-token forwards. The native lossless
VXT then replayed the exact same 16 prompt IDs and eight HF-selected teacher IDs
on actual `Apple Metal GPU (Apple M2 Max, 1-CmdBuf Fused)`. Each of the eight
steps compared every vocabulary entry: **1,986,560 finite logits per runtime**.
No core inference/kernel change was made for this check.

- Top1: **8/8**.
- Maximum absolute logit error: **7.62939453125e-6**.
- Maximum RMSE: **1.6920509868198743e-6**.
- Minimum cosine: **0.999999999999547**.
- Maximum KL(reference || native): **1.9556060022759766e-12**.
- Existing gate, unchanged: finite logits, max RMSE <= 1e-4,
  min cosine >= 0.99999, every top1 match. Comparison exited0.

The reference does not import VXG or convert weights through GGUF/the codec.
Installed official source defines multimodal ownership as
`Qwen3_5Model.language_model`, whereas the text causal class owns
`Qwen3_5ForCausalLM.model = Qwen3_5TextModel(...)`; its tied head is
`lm_head.weight -> model.embed_tokens.weight`. The script maps exactly the320
original `model.language_model.*` keys to `model.*`, validates all destination
keys and shapes against the official class, materializes their original BF16/F32
values as CPU F32, ties the head, and loads with `strict=True`. Vision and MTP
are never instantiated. Meta construction avoids redundant model allocation;
nonpersistent RoPE buffers are materialized by the official rotary constructor,
then every parameter and buffer is checked against remaining meta storage.
Official pure-PyTorch convolution and recurrent DeltaNet fallbacks were observed;
optional `causal_conv1d` and `flash-linear-attention` were not installed. No remote
model code or additional model checkpoint was downloaded.

Python3.11.18, Torch2.9.0 build
`0fabc3ba44823f257e70ce397d989c8de5e362c1`, Accelerate BLAS,
macOS26.6.2 arm64. HF forward trace elapsed64.095s; peak reference RSS
9,895,936,000 bytes. Runs were sequential on the32GiB host; observed system
memory free was79% before setup,38% with the reference resident,70% after its
exit. No simultaneous model benchmark was started by this check. Official
Transformers model-definition SHA256:
`af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c`.
Pinned native source checkpoint SHA256 remains
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`;
VXT identity is the unchanged native archive documented above.

Retained under `/private/tmp/vxg-real-qualification-20261104/`:
`external-oracle.log`, `external-oracle-packages.txt`,
`external-oracle-code.sha256`, `external-oracle-artifacts.sha256`, and
`external-oracle/{trace.json,hf-logits.f32,native-logits.f32,native-trace.log,comparison.log,native-logits.f32.comparison.json}`.
Raw logit files are little-endian F32 shape `[8,248320]`,7,946,240 bytes each.
HF logits SHA256 `49e5dd690f42657983b71463c9514a729f52246628619523018565371dee0b54`;
native logits SHA256 `5ecb6639b569c72d88bf98898dd86b5669ceeca3ccbe8ab80b815df4c3d68408`.
Trace binary SHA256 `543e1cd695036adb25c58cac64fb5869d11534120d2ccc14c2a3af23e5fe63fb`.
The dedicated `trace-model` binary fails closed on invalid IDs, vocabulary/shape
mismatches, nonfinite logits, unavailable fused Metal, and existing output files.
Its focused invalid-ID test passed (`trace-model-test.log`); source-format/diff
checks passed for these additions. This is bounded numerical fidelity evidence
from an independent implementation, **not bit-exact inference, exhaustive
accuracy, long-context qualification, a tokenizer equivalence test, multimodal
support, or a performance/release qualification**. The tokenizer only selected
one prompt; both runtimes consumed identical retained integers directly.

Setup attempts caught an initially too-short prompt and nonpersistent meta RoPE
buffers before producing numerical traces. The first sandboxed native invocation
failed its fused-Metal requirement without creating a trace; the identical binary
then ran with approved GPU access. No numerical failure was hidden or threshold
relaxed. A later prompt-progress print was added to the reusable reference script
without changing its mathematical computation.

```sh
UV_CACHE_DIR=/private/tmp/vxg-torch-uv-cache uv pip install --python /private/tmp/vxg-torch-exp-venv/bin/python 'transformers==5.22.1' safetensors numpy
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /private/tmp/vxg-torch-exp-venv/bin/python scripts/qwen35_external_oracle.py /private/tmp/vxg-real-qualification-20261104/official-model.safetensors /private/tmp/vxg-real-qualification-20261104/official-config.json /private/tmp/vxg-real-qualification-20261104/official-tokenizer.json /private/tmp/vxg-real-qualification-20261104/external-oracle
CARGO_TARGET_DIR=/private/tmp/vxt-native-target cargo build -p vxg-tensor --release --offline --bin trace-model
/private/tmp/vxt-native-target/release/trace-model /private/tmp/vxg-real-qualification-20261104/Qwen3.5-2B-native-lossless.vxt /private/tmp/vxg-real-qualification-20261104/external-oracle/trace.json /private/tmp/vxg-real-qualification-20261104/external-oracle/native-logits.f32
/private/tmp/vxg-torch-exp-venv/bin/python scripts/compare_external_trace.py /private/tmp/vxg-real-qualification-20261104/external-oracle /private/tmp/vxg-real-qualification-20261104/external-oracle/native-logits.f32
CARGO_TARGET_DIR=/private/tmp/vxt-native-target cargo test -p vxg-tensor --offline --bin trace-model
```

Reference support/package identity checked against the
[official Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)
and [official PyPI package metadata](https://pypi.org/pypi/transformers/json),
then verified in the installed official class source. Complete installed package
versions are retained so auxiliary dependencies are reproducible.
