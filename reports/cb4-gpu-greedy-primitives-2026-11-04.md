# Exact CB4 GPU greedy primitives, 2026-11-04

Native/Rust implementation stage; not speed or quality qualification. Default generic forward remains unchanged. This stage does not implement speculative bursts, approximate heads, FP16 activation state, new quantization, or archive changes.

Implementation is in `crates/vxg-tensor/native/metal_llm.mm`, `native/shaders/vxt_llm_ops.metal`, and `src/metal_llm.rs`. Explicit `MetalLlmRunner::enable_cb4_greedy` requires CB4 token embeddings and uploads the original 16 CPU centroids separately from the GEMV descriptor's existing polynomial approximation. `greedy_step` decodes the packed row on GPU, runs the existing F32 forward encoding and full head unchanged, and reduces the full logits on GPU. One input token is processed per command and only the selected next ID returns to CPU. It removes CPU row decode/full-logit download; there is no burst or new scheduling-speed claim.

The embedding encoder reproduces CPU multiply rounding and sequential additive CSR outliers, including duplicate columns. Bounds reject input tokens outside vocabulary. Rust rejects position conversion overflow. Setup retains existing embedding buffers, is idempotent for identical buffers/centroids and rejects a changed embedding. New command/encoder allocation failures are explicit errors. Command errors remain errors. Production rejects any nonfinite head logits after the token was processed; callers must invalidate that state, since rollback is not implemented. The high-level integration owns poisoning and callback/stop semantics.

The focused primitive test fails closed when no Metal device is available. It passed with actual GPU access; no CPU skip is present. It bitwise-compares CPU and GPU decoded embeddings for dimensions 1, 129, 257, 385, odd padded group counts, rows 0/1/514 of a 515-row matrix, arbitrary learned centroids and serial duplicate-column outliers. Full argmax selection is compared to the existing greedy sampler for finite ties across reduction lanes, signed zero, all minus infinity, isolated NaN, plus infinity ties and all NaN. The diagnostic probe permits these nonfinite inputs to compare selection; production rejects them. Production poisoning is a separate high-level integration gate.

Commands (workspace root):

```
cargo check -p vxg-tensor --lib
cargo test -p vxg-tensor --lib cb4_gpu_embedding_exact_centroids_outliers_tails_and_greedy_ties --no-run
/Users/mapletechnologies/.cargo/target/debug/deps/vxg_tensor-9a37eb0889d3dc32 cb4_gpu_embedding_exact_centroids_outliers_tails_and_greedy_ties --nocapture
git diff --check -- crates/vxg-tensor/native/metal_llm.mm crates/vxg-tensor/native/shaders/vxt_llm_ops.metal crates/vxg-tensor/src/metal_llm.rs
```

The initial sandboxed Cargo check could not open the configured target's build lock. The check/build/run above then used approved escalation, all passed. Focused test: 1 passed, 0 failed, 0 ignored, 1.17 seconds. Diff whitespace check passed. No broad test claim follows.

Evidence root: `/private/tmp/vxg-real-qualification-20261104/cb4-greedy-primitives-20261104`. `hashes.json` freezes three source paths and the exact debug test executable; `primitive-test-tool-output.log` retains the successful tool transcript, explicitly identified as transcript retention rather than shell capture. Test binary SHA256: `268c97e82274f849ff75336b94e54c4ec5ada5519ed57bb4aaf4682992a168c4`.

Remaining gates belong to the executable high-level integration: ordinary forward equivalence, generated-token equality, cap/EOS/callback/resume and production error poisoning, then bounded paired real-model throughput under equal streaming work and unchanged-archive identity. No speed claim or full-objective completion is made at this stage.
