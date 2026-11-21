//! `vimg` (`VIMG1`) — HTTP-Range-Native Hierarchical Wavelet Image Codec &
//! Compressed-Domain Transform Engine (Pure Rust, zero-dep core lib, `wasm32`-clean).
//!
//! Stores images as independent spatial tiles decomposed by reversible integer `CDF 5/3`
//! lifting in `YCoCg-R` color space, grouped on disk by `(pyramid_level, color_plane, tile_id)`
//! and entropy-coded with pure-Rust 2-way interleaved `tANS / FSE` + hardware `CRC32C`.

pub mod checksum;
pub mod color;
pub mod entropy;
pub mod exif;
pub mod ffi;
pub mod format;
pub mod meta;
pub mod ops;
pub mod range;
pub mod simd;
pub mod simd_dwt;
pub mod subband;
pub mod wavelet;

use core::fmt;

pub use format::{crop_vimg_zero_decode, encode_vimg, encode_vimg_u16, VimgHeader, VimgIndex};
pub use meta::{Cicp, VimgMetadata};
pub use ops::{decode_from_source, plan_query_ranges, DecodedImage, TransformQuery};
pub use range::{coalesce_ranges, RangeSource, RangeStats, SliceRangeSource, SparseRangeSource};
#[cfg(not(target_arch = "wasm32"))]
pub use range::{FileRangeSource, HttpRangeSource};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidInput(&'static str),
    InvalidFormat(&'static str),
    Truncated,
    ChecksumMismatch { expected: u32, actual: u32 },
    Unsupported(&'static str),
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(m) => write!(f, "invalid input: {m}"),
            Self::InvalidFormat(m) => write!(f, "invalid VIMG format: {m}"),
            Self::Truncated => write!(f, "truncated VIMG stream"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "CRC32C mismatch: expected 0x{expected:08x}, got 0x{actual:08x}")
            }
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Io(m) => write!(f, "I/O error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

/// Encoding options for `VIMG1` archives.
#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Quality `1..=100` (`100` = 100% bit-exact lossless `YCoCg-R 4:4:4 + CDF 5/3`).
    pub quality: u8,
    /// Spatial tile size in pixels (`64`, `128`, or `256`; default `256`).
    pub tile_size: u32,
    /// Number of 2D wavelet decomposition steps (`1..=4`; default `3` => `1/8, 1/4, 1/2, 1/1` pyramid).
    pub dwt_levels: u8,
    /// Enable `4:2:0` chroma subsampling in lossy mode (`quality < 100`, default `true`).
    pub chroma_420: bool,
    /// Store hardware `CRC32C` checksum per `(level, plane, tile)` chunk (default `true`).
    pub checksum: bool,
    /// Worker thread count (`0` = auto). Output is 100% deterministic (`1T == NT`).
    pub threads: usize,
    /// Bits per channel (`8` default; `10` or `12` for HDR `encode_u16`).
    pub bit_depth: u8,
    /// Optional embedded ICC color profile (e.g., Display P3, AdobeRGB, sRGB; capped at 16 MiB).
    pub icc_profile: Option<Vec<u8>>,
    /// Enable tile-adaptive psychovisual quantization (`CODEC_BLK4_AQ`) in lossy mode (`quality < 100`).
    /// Smooth regions get finer steps (less banding); chaotic textures get contrast-masked coarser steps.
    pub adaptive_quant: bool,
    /// Optional CICP HDR color tags (`sRGB`, `Display P3`, `BT.2020 PQ`, `BT.2020 HLG`).
    pub cicp: Option<Cicp>,
    /// Optional raw EXIF metadata payload preserved in the trailing TLV section.
    pub exif: Option<Vec<u8>>,
    /// Optional raw XMP XML metadata packet preserved in the trailing TLV section.
    pub xmp: Option<Vec<u8>>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quality: 85,
            tile_size: 256,
            dwt_levels: 3,
            chroma_420: true,
            checksum: true,
            threads: 0,
            bit_depth: 8,
            icc_profile: None,
            adaptive_quant: false,
            cicp: None,
            exif: None,
            xmp: None,
        }
    }
}

impl EncodeOptions {
    pub fn lossless() -> Self {
        Self {
            quality: 100,
            chroma_420: false,
            ..Self::default()
        }
    }
}

/// Convenience helper: encode raw pixels (`RGB`, `RGBA`, or `Gray`) into `.vimg` bytes.
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u8,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    encode_vimg(pixels, width, height, channels, opts)
}

/// Convenience helper: encode 10-bit or 12-bit HDR `u16` pixels (`RGB`, `RGBA`, or `Gray`) into `.vimg` bytes.
pub fn encode_u16(
    pixels: &[u16],
    width: u32,
    height: u32,
    channels: u8,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    encode_vimg_u16(pixels, width, height, channels, opts)
}

/// Convenience helper: decode a full `.vimg` slice at full resolution.
pub fn decode(vimg_bytes: &[u8]) -> Result<DecodedImage> {
    decode_query(vimg_bytes, &TransformQuery::default())
}

/// Convenience helper: execute a compressed-domain query on an in-memory `.vimg` slice.
pub fn decode_query(vimg_bytes: &[u8], query: &TransformQuery) -> Result<DecodedImage> {
    let src = SliceRangeSource(vimg_bytes);
    decode_from_source(&src, query)
}

/// Computes Peak Signal-to-Noise Ratio (dB) between two equal-length 8-bit pixel buffers.
pub fn psnr_db(reference: &[u8], test: &[u8]) -> f64 {
    if reference.len() != test.len() || reference.is_empty() {
        return 0.0;
    }
    let mut sse = 0.0f64;
    for (&a, &b) in reference.iter().zip(test.iter()) {
        let d = (a as f64) - (b as f64);
        sse += d * d;
    }
    let mse = sse / (reference.len() as f64);
    if mse <= 1e-12 {
        f64::INFINITY
    } else {
        10.0 * ((255.0 * 255.0) / mse).log10()
    }
}
