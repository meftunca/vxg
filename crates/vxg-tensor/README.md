<p align="center">
  <img src="../../assets/logos/vxt-turbo.svg" width="88" height="88" alt="VXT-Turbo Logo" />
</p>

<h1 align="center">VXT-Turbo <code>v0.6.0</code> (<code>vxg-tensor</code>)</h1>
<p align="center">
  <strong>Compressed-Domain E8 Gosset Lattice LLM / VLM / MoE Codec &amp; Metal / CUDA Inference Engine</strong><br/>
  Beats <code>GGUF Q6_K</code> &amp; <code>Q8_0</code> at smaller sizes · <code>963 tok/s</code> RTX 5090 CUDA Resident · <code>298 tok/s</code> Apple Metal
</p>

<p align="center">
  <a href="../../docs/TENSOR.md"><img src="https://img.shields.io/badge/docs-Full%20Spec%20%26%20Benchmarks-A855F7?style=flat-square" alt="Docs" /></a>
  <a href="../../apps/vxt-studio"><img src="https://img.shields.io/badge/app-VXT%20Studio-EC4899?style=flat-square" alt="VXT Studio" /></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

## Highlights

- **E8 / E8-HQ Gosset Lattice Quantization (`VXT1`):** Packs `GGUF` and `SafeTensors` models with zero-OOM streaming (`--preset q3|q4|q6|q8|lossless`), beating `GGUF Q6_K` and `Q8_0` quality at smaller disk footprints.
- **Native GPU Kernels:** Direct compressed-domain dequant + matvec on **NVIDIA CUDA VRAM-Resident** (`963 tok/s` Q3 / `796 tok/s` Q6 on RTX 5090) and **Apple Metal** (`298 tok/s` 0.8B, `176 tok/s` 2B, `40.7–47.2 tok/s` 9B + Vision).
- **Multimodal & MoE Ready:** Supports dense Transformers, Mixture-of-Experts (MoE), 3D M-RoPE Vision encoders, Speculative Decoding, and OpenAI-compatible SSE + Prometheus HTTP serving.

## Quick Start

```bash
# Pack a GGUF or SafeTensors model into a compressed .vxt container
cargo run --release -p vxg-tensor -- pack model.gguf model.vxt --preset q6

# Run local interactive generation or OpenAI-compatible server
cargo run --release -p vxg-tensor -- run model.vxt --prompt "Explain E8 lattice quantization briefly."
cargo run --release -p vxg-tensor -- serve model.vxt --port 8080
```

For full architecture notes, quantization tables, and GPU benchmarks, see **[`docs/TENSOR.md`](../../docs/TENSOR.md)**.
