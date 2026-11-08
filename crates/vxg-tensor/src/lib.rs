//! `vxg-tensor` — Compressed-domain AI model weight codec (`.vxt`) and fused Apple Metal GPU /
//! multi-threaded CPU SIMD `GEMV` (`y = W · x`) engine.
//!
//! Two hybrid modes:
//! 1. [`lossless::LosslessMatrix`] (`Mode 1`): **100% Bit-Exact Lossless** bit-plane + Top-K
//!    exponent LUT codec (`~11.5 bpw` on `BF16`, `~14.2 bpw` on `FP16`) with direct compressed
//!    `GEMV`.
//! 2. [`codebook::CodebookMatrix`] (`Mode 2`): **Non-uniform 1D Lloyd-Max Codebook (`3` or `4`
//!    bits) + Adaptive Sparse `FP16` CSR Outlier Stream** (`~3.40 bpw` & `~4.35 bpw`) with direct
//!    compressed `GEMV`.

pub mod bench;
mod cb4_calib;
pub mod cb4_opt;
pub mod chat;
pub mod cli_llm;
pub mod codebook;
mod entropy_pack;
pub mod format;
mod format_io;
pub mod fp16;
pub mod gguf;
pub mod gguf_meta;
mod hf_meta;
pub mod hf_pack;
mod huff;
pub mod llm;
mod llm_calib;
pub mod llm_ops;
pub mod llm_pack;
pub mod lossless;
mod meta_lz;
pub mod metal;
mod metal_cb4_fast;
#[cfg(vxt_has_metal)]
mod metal_cb4_fast_ffi;
#[cfg(vxt_has_metal)]
mod metal_ffi;
mod metal_llm;
mod metal_llm_turbo;
#[cfg(vxt_has_metal)]
mod metal_turbo_ffi;
pub mod mmap;
pub mod native_qwen35;
pub mod q4_0;
pub mod safetensors;
pub mod server;
mod server_http;
pub mod tokenizer;
pub mod wht2;
pub mod wht_turbo;
mod wht_turbo_enc;
mod wht_turbo_tables;

use std::fmt;

/// Native 16-bit floating-point storage type for lossless neural weight compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            Self::BF16 => "BF16",
            Self::F16 => "FP16",
        }
    }
}

/// Error type returned by `vxg-tensor` loaders, codecs, `.vxt` containers, and GPU kernels.
#[derive(Clone, Debug)]
pub enum Error {
    Io(String),
    Format(String),
    InvalidShape(&'static str),
    Checksum(String),
    Unsupported(String),
    Backend(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "I/O error: {s}"),
            Self::Format(s) => write!(f, "format error: {s}"),
            Self::InvalidShape(s) => write!(f, "invalid tensor shape: {s}"),
            Self::Checksum(s) => write!(f, "integrity checksum mismatch: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported feature: {s}"),
            Self::Backend(s) => write!(f, "GPU backend error: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
