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
pub mod codebook;
pub mod format;
pub mod fp16;
pub mod gguf;
pub mod lossless;
pub mod metal;
pub mod mmap;
pub mod q4_0;
pub mod safetensors;

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
