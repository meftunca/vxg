//! `.vxt` (`VXT1`) binary container format for compressed AI model weight archives.
//!
//! Layout:
//! - Magic: `b"VXT1"` (`4` bytes)
//! - Version: `u32` LE (`1`)
//! - Tensor count: `u32` LE
//! - Reserved: `u32` LE (`0`)
//! - Per-tensor directory & 64-byte-aligned payload sections with FNV-1a64 integrity tags.

use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::lossless::LosslessMatrix;
use crate::{DType, Error, Result};

const MAGIC: &[u8; 4] = b"VXT1";
const VERSION: u32 = 1;

const KIND_LOSSLESS: u8 = 1;
const KIND_CODEBOOK: u8 = 2;
const KIND_Q4_0: u8 = 3;

#[derive(Clone, Debug)]
pub enum PackedTensor {
    Lossless(LosslessMatrix),
    Codebook(CodebookMatrix),
    Q4_0(Q4_0Matrix),
}

impl PackedTensor {
    pub fn rows(&self) -> usize {
        match self {
            Self::Lossless(m) => m.rows,
            Self::Codebook(m) => m.rows,
            Self::Q4_0(m) => m.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Lossless(m) => m.cols,
            Self::Codebook(m) => m.cols,
            Self::Q4_0(m) => m.cols,
        }
    }

    pub fn compressed_bytes(&self) -> usize {
        match self {
            Self::Lossless(m) => m.compressed_bytes(),
            Self::Codebook(m) => m.compressed_bytes(),
            Self::Q4_0(m) => m.compressed_bytes(),
        }
    }

    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Self::Lossless(m) => m.bits_per_weight(),
            Self::Codebook(m) => m.bits_per_weight(),
            Self::Q4_0(m) => m.bits_per_weight(),
        }
    }

    pub fn mode_name(&self) -> &'static str {
        match self {
            Self::Lossless(m) => match m.dtype {
                DType::BF16 => "VXT-Lossless-BF16",
                DType::F16 => "VXT-Lossless-FP16",
            },
            Self::Codebook(m) => {
                if m.bits == 4 {
                    "VXT-CB4+Outlier"
                } else {
                    "VXT-CB3+Outlier"
                }
            }
            Self::Q4_0(_) => "GGUF-Q4_0-Ref",
        }
    }

    pub fn gemv_cpu(&self, x: &[f32], y: &mut [f32], threads: usize) {
        match self {
            Self::Lossless(m) => m.gemv_cpu(x, y, threads),
            Self::Codebook(m) => m.gemv_cpu(x, y, threads),
            Self::Q4_0(m) => m.gemv_cpu(x, y, threads),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VxtEntry {
    pub name: String,
    pub tensor: PackedTensor,
}

#[inline]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[inline]
fn pad64(buf: &mut Vec<u8>) {
    while buf.len() % 64 != 0 {
        buf.push(0);
    }
}

fn push_u16_slice(buf: &mut Vec<u8>, s: &[u16]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.reserve(s.len() * 2);
    for &v in s {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_u32_slice(buf: &mut Vec<u8>, s: &[u32]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.reserve(s.len() * 4);
    for &v in s {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_u8_slice(buf: &mut Vec<u8>, s: &[u8]) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s);
}

/// Serializes one or more [`VxtEntry`] compressed matrices into a `.vxt` (`VXT1`) archive.
pub fn write_vxt_archive(entries: &[VxtEntry]) -> Result<Vec<u8>> {
    let count = u32::try_from(entries.len())
        .map_err(|_| Error::Format("too many tensors in VXT archive".into()))?;
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());

    for entry in entries {
        pad64(&mut out);
        let section_start = out.len();
        let nb = entry.name.as_bytes();
        out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        out.extend_from_slice(nb);

        match &entry.tensor {
            PackedTensor::Lossless(m) => {
                out.push(KIND_LOSSLESS);
                out.push(match m.dtype {
                    DType::BF16 => 0,
                    DType::F16 => 1,
                });
                out.extend_from_slice(&[0u8; 6]);
                out.extend_from_slice(&(m.rows as u64).to_le_bytes());
                out.extend_from_slice(&(m.cols as u64).to_le_bytes());
                out.extend_from_slice(&(m.padded_cols as u64).to_le_bytes());
                out.extend_from_slice(&m.lut);
                push_u8_slice(&mut out, &m.base_bytes);
                push_u32_slice(&mut out, &m.code_words);
                push_u32_slice(&mut out, &m.sb_esc_offsets);
                push_u8_slice(&mut out, &m.esc_bytes);
            }
            PackedTensor::Codebook(m) => {
                out.push(KIND_CODEBOOK);
                out.push(m.bits);
                out.extend_from_slice(&[0u8; 6]);
                out.extend_from_slice(&(m.rows as u64).to_le_bytes());
                out.extend_from_slice(&(m.cols as u64).to_le_bytes());
                out.extend_from_slice(&(m.padded_cols as u64).to_le_bytes());
                for &c in &m.centroids {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                push_u16_slice(&mut out, &m.scales_f16);
                push_u32_slice(&mut out, &m.packed_words);
                push_u32_slice(&mut out, &m.outlier_row_ptr);
                push_u32_slice(&mut out, &m.outliers_packed);
            }
            PackedTensor::Q4_0(m) => {
                out.push(KIND_Q4_0);
                out.extend_from_slice(&[0u8; 7]);
                out.extend_from_slice(&(m.rows as u64).to_le_bytes());
                out.extend_from_slice(&(m.cols as u64).to_le_bytes());
                out.extend_from_slice(&(m.padded_cols as u64).to_le_bytes());
                push_u16_slice(&mut out, &m.scales_f16);
                push_u32_slice(&mut out, &m.qs_words);
            }
        }
        let checksum = fnv1a64(&out[section_start..]);
        out.extend_from_slice(&checksum.to_le_bytes());
    }
    let archive_crc = fnv1a64(&out);
    out.extend_from_slice(&archive_crc.to_le_bytes());
    Ok(out)
}

/// Parses and validates all compressed tensors from a `.vxt` (`VXT1`) byte slice / `mmap`.
pub fn read_vxt_archive(bytes: &[u8]) -> Result<Vec<VxtEntry>> {
    if bytes.len() < 24 {
        return Err(Error::Format("truncated VXT archive".into()));
    }
    if &bytes[..4] != MAGIC {
        return Err(Error::Format("invalid VXT magic (expected VXT1)".into()));
    }
    let ver = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if ver != VERSION {
        return Err(Error::Unsupported(format!("unsupported VXT version {ver}")));
    }
    let body = &bytes[..bytes.len() - 8];
    let expected_archive_crc = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap());
    if fnv1a64(body) != expected_archive_crc {
        return Err(Error::Checksum("VXT archive checksum mismatch".into()));
    }
    let mut r = Cur { b: body, pos: 8 };
    let count = r.u32()? as usize;
    let _reserved = r.u32()?;

    let mut entries = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        r.align64()?;
        let section_start = r.pos;
        let name_len = r.u32()? as usize;
        let name_bytes = r.take(name_len)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| Error::Format("invalid UTF-8 tensor name in VXT".into()))?
            .to_owned();

        let kind = r.u8()?;
        let sub = r.u8()?;
        let _pad = r.take(6)?;
        let rows = r.usize64()?;
        let cols = r.usize64()?;
        let padded_cols = r.usize64()?;
        if rows == 0 || cols == 0 || padded_cols < cols {
            return Err(Error::Format("invalid VXT matrix dimensions".into()));
        }

        let tensor = match kind {
            KIND_LOSSLESS => {
                let dtype = match sub {
                    0 => DType::BF16,
                    1 => DType::F16,
                    _ => return Err(Error::Format("unknown VXT lossless dtype".into())),
                };
                if padded_cols != cols.div_ceil(256) * 256 {
                    return Err(Error::Format("invalid VXT lossless padded_cols".into()));
                }
                let mut lut = [0u8; 8];
                lut.copy_from_slice(r.take(8)?);
                let base_bytes = r.u8_vec()?;
                let code_words = r.u32_vec()?;
                let sb_esc_offsets = r.u32_vec()?;
                let esc_bytes = r.u8_vec()?;

                let sbs = rows * (padded_cols / 256);
                let tiles = sbs * 8;
                let wpt = match dtype {
                    DType::BF16 => 3,
                    DType::F16 => 6,
                };
                if base_bytes.len() != rows * padded_cols
                    || code_words.len() != tiles * wpt
                    || sb_esc_offsets.len() != sbs
                {
                    return Err(Error::Format("corrupt VXT lossless plane lengths".into()));
                }
                PackedTensor::Lossless(LosslessMatrix {
                    dtype,
                    rows,
                    cols,
                    padded_cols,
                    lut,
                    base_bytes,
                    code_words,
                    sb_esc_offsets,
                    esc_bytes,
                })
            }
            KIND_CODEBOOK => {
                let bits = sub;
                if !matches!(bits, 3 | 4) || padded_cols != cols.div_ceil(256) * 256 {
                    return Err(Error::Format("invalid VXT codebook header".into()));
                }
                let mut centroids = [0.0f32; 16];
                for slot in &mut centroids {
                    let s = r.take(4)?;
                    *slot = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                }
                let scales_f16 = r.u16_vec()?;
                let packed_words = r.u32_vec()?;
                let outlier_row_ptr = r.u32_vec()?;
                let outliers_packed = r.u32_vec()?;

                let groups = rows * (padded_cols / 128);
                if scales_f16.len() != groups
                    || packed_words.len() != groups * 4 * (bits as usize)
                    || outlier_row_ptr.len() != rows + 1
                    || *outlier_row_ptr.last().unwrap() as usize != outliers_packed.len()
                {
                    return Err(Error::Format("corrupt VXT codebook plane lengths".into()));
                }
                PackedTensor::Codebook(CodebookMatrix {
                    bits,
                    rows,
                    cols,
                    padded_cols,
                    centroids,
                    scales_f16,
                    packed_words,
                    outlier_row_ptr,
                    outliers_packed,
                })
            }
            KIND_Q4_0 => {
                if padded_cols != cols.div_ceil(32) * 32 {
                    return Err(Error::Format("invalid VXT Q4_0 padded_cols".into()));
                }
                let scales_f16 = r.u16_vec()?;
                let qs_words = r.u32_vec()?;
                let blocks = rows * (padded_cols / 32);
                if scales_f16.len() != blocks || qs_words.len() != blocks * 4 {
                    return Err(Error::Format("corrupt VXT Q4_0 plane lengths".into()));
                }
                PackedTensor::Q4_0(Q4_0Matrix {
                    rows,
                    cols,
                    padded_cols,
                    scales_f16,
                    qs_words,
                })
            }
            _ => return Err(Error::Format(format!("unknown VXT tensor kind {kind}"))),
        };

        let section_end = r.pos;
        let expected_crc = r.u64()?;
        let actual_crc = fnv1a64(&bytes[section_start..section_end]);
        if expected_crc != actual_crc {
            return Err(Error::Checksum(format!(
                "VXT integrity mismatch on tensor '{name}'"
            )));
        }
        entries.push(VxtEntry { name, tensor });
    }
    Ok(entries)
}

struct Cur<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cur<'a> {
    fn align64(&mut self) -> Result<()> {
        let rem = self.pos % 64;
        if rem != 0 {
            let _ = self.take(64 - rem)?;
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Format("VXT cursor overflow".into()))?;
        if end > self.b.len() {
            return Err(Error::Format("truncated VXT archive".into()));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    fn usize64(&mut self) -> Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| Error::Format("VXT length overflow".into()))
    }

    fn u8_vec(&mut self) -> Result<Vec<u8>> {
        let n = self.usize64()?;
        Ok(self.take(n)?.to_vec())
    }

    fn u16_vec(&mut self) -> Result<Vec<u16>> {
        let n = self.usize64()?;
        let bytes = self.take(
            n.checked_mul(2)
                .ok_or_else(|| Error::Format("u16 overflow".into()))?,
        )?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }

    fn u32_vec(&mut self) -> Result<Vec<u32>> {
        let n = self.usize64()?;
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| Error::Format("u32 overflow".into()))?,
        )?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}
