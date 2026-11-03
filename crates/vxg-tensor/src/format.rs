//! `.vxt` (`VXT1`) binary container format for compressed AI model weight & LLM archives.
//!
//! Layout:
//! - Magic: `b"VXT1"` (`4` bytes)
//! - Version: `u32` LE (`1`)
//! - Tensor count: `u32` LE
//! - Metadata KV count: `u32` LE (`0` for plain matrix archives)
//! - Optional `(key, val_bytes)` metadata table
//! - Per-tensor 64-byte-aligned payload sections with FNV-1a64 section + whole-archive tags.

use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::lossless::LosslessMatrix;
use crate::wht2::Wht2Matrix;
use crate::{DType, Error, Result};

const MAGIC: &[u8; 4] = b"VXT1";
const VERSION: u32 = 1;

const KIND_LOSSLESS: u8 = 1;
const KIND_CODEBOOK: u8 = 2;
const KIND_Q4_0: u8 = 3;
const KIND_WHT2: u8 = 4;
const KIND_RAW_F32: u8 = 5;

#[derive(Clone, Debug)]
pub enum PackedTensor {
    Lossless(LosslessMatrix),
    Codebook(CodebookMatrix),
    Q4_0(Q4_0Matrix),
    Wht2(Wht2Matrix),
    RawF32 {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
}

impl PackedTensor {
    pub fn rows(&self) -> usize {
        match self {
            Self::Lossless(m) => m.rows,
            Self::Codebook(m) => m.rows,
            Self::Q4_0(m) => m.rows,
            Self::Wht2(m) => m.rows,
            Self::RawF32 { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Lossless(m) => m.cols,
            Self::Codebook(m) => m.cols,
            Self::Q4_0(m) => m.cols,
            Self::Wht2(m) => m.cols,
            Self::RawF32 { cols, .. } => *cols,
        }
    }

    pub fn compressed_bytes(&self) -> usize {
        match self {
            Self::Lossless(m) => m.compressed_bytes(),
            Self::Codebook(m) => m.compressed_bytes(),
            Self::Q4_0(m) => m.compressed_bytes(),
            Self::Wht2(m) => m.compressed_bytes(),
            Self::RawF32 { data, .. } => data.len() * 4,
        }
    }

    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Self::Lossless(m) => m.bits_per_weight(),
            Self::Codebook(m) => m.bits_per_weight(),
            Self::Q4_0(m) => m.bits_per_weight(),
            Self::Wht2(m) => m.bits_per_weight(),
            Self::RawF32 { rows, cols, data } => {
                (data.len() as f64 * 32.0) / ((*rows * *cols).max(1) as f64)
            }
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
            Self::Wht2(_) => "VXT-WHT2+Outlier",
            Self::RawF32 { .. } => "VXT-Raw-FP32",
        }
    }

    /// Reconstructs a single row `r` (`dst.len() == self.cols()`) in `O(cols)`.
    pub fn decode_row(&self, r: usize, dst: &mut [f32]) {
        match self {
            Self::Lossless(m) => m.decode_row(r, dst),
            Self::Codebook(m) => m.decode_row(r, dst),
            Self::Q4_0(m) => m.decode_row(r, dst),
            Self::Wht2(m) => m.decode_row(r, dst),
            Self::RawF32 { cols, data, .. } => {
                dst.copy_from_slice(&data[r * *cols..(r + 1) * *cols]);
            }
        }
    }

    /// Reconstructs all elements as `Vec<f32>`.
    pub fn decode_f32(&self) -> Vec<f32> {
        let rows = self.rows();
        let cols = self.cols();
        if let Self::RawF32 { data, .. } = self {
            return data.clone();
        }
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            self.decode_row(r, &mut out[r * cols..(r + 1) * cols]);
        }
        out
    }

    pub fn gemv_cpu(&self, x: &[f32], y: &mut [f32], threads: usize) {
        match self {
            Self::Lossless(m) => m.gemv_cpu(x, y, threads),
            Self::Codebook(m) => m.gemv_cpu(x, y, threads),
            Self::Q4_0(m) => m.gemv_cpu(x, y, threads),
            Self::Wht2(m) => m.gemv_cpu(x, y, threads),
            Self::RawF32 { rows, cols, data } => {
                assert_eq!(x.len(), *cols);
                assert_eq!(y.len(), *rows);
                for (r, out_y) in y.iter_mut().enumerate() {
                    let row = &data[r * *cols..(r + 1) * *cols];
                    let mut s = 0.0f32;
                    for (&w, &xv) in row.iter().zip(x.iter()) {
                        s = w.mul_add(xv, s);
                    }
                    *out_y = s;
                }
            }
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

fn push_f32_slice(buf: &mut Vec<u8>, s: &[f32]) {
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

fn push_hdr(out: &mut Vec<u8>, kind: u8, sub: u8, rows: usize, cols: usize, pcols: usize) {
    out.extend_from_slice(&[kind, sub, 0, 0, 0, 0, 0, 0]);
    out.extend_from_slice(&(rows as u64).to_le_bytes());
    out.extend_from_slice(&(cols as u64).to_le_bytes());
    out.extend_from_slice(&(pcols as u64).to_le_bytes());
}

/// Serializes one or more [`VxtEntry`] compressed matrices into a `.vxt` (`VXT1`) archive.
pub fn write_vxt_archive(entries: &[VxtEntry]) -> Result<Vec<u8>> {
    write_vxt_model(&[], entries)
}

/// Serializes model metadata KV pairs + [`VxtEntry`] tensors into a self-contained `.vxt` archive.
pub fn write_vxt_model(meta_kv: &[(String, Vec<u8>)], entries: &[VxtEntry]) -> Result<Vec<u8>> {
    let count = u32::try_from(entries.len())
        .map_err(|_| Error::Format("too many tensors in VXT archive".into()))?;
    let mcount = u32::try_from(meta_kv.len())
        .map_err(|_| Error::Format("too many metadata entries in VXT archive".into()))?;

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&mcount.to_le_bytes());

    for (k, v) in meta_kv {
        let kb = k.as_bytes();
        out.extend_from_slice(&(kb.len() as u32).to_le_bytes());
        out.extend_from_slice(kb);
        push_u8_slice(&mut out, v);
    }

    for entry in entries {
        pad64(&mut out);
        let section_start = out.len();
        let nb = entry.name.as_bytes();
        out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        out.extend_from_slice(nb);

        match &entry.tensor {
            PackedTensor::Lossless(m) => {
                push_hdr(
                    &mut out,
                    KIND_LOSSLESS,
                    matches!(m.dtype, DType::F16) as u8,
                    m.rows,
                    m.cols,
                    m.padded_cols,
                );
                out.extend_from_slice(&m.lut);
                push_u8_slice(&mut out, &m.base_bytes);
                push_u32_slice(&mut out, &m.code_words);
                push_u32_slice(&mut out, &m.sb_esc_offsets);
                push_u8_slice(&mut out, &m.esc_bytes);
            }
            PackedTensor::Codebook(m) => {
                push_hdr(
                    &mut out,
                    KIND_CODEBOOK,
                    m.bits,
                    m.rows,
                    m.cols,
                    m.padded_cols,
                );
                for &c in &m.centroids {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                push_u16_slice(&mut out, &m.scales_f16);
                push_u32_slice(&mut out, &m.packed_words);
                push_u32_slice(&mut out, &m.outlier_row_ptr);
                push_u32_slice(&mut out, &m.outliers_packed);
            }
            PackedTensor::Q4_0(m) => {
                push_hdr(&mut out, KIND_Q4_0, 0, m.rows, m.cols, m.padded_cols);
                push_u16_slice(&mut out, &m.scales_f16);
                push_u32_slice(&mut out, &m.qs_words);
            }
            PackedTensor::Wht2(m) => {
                push_hdr(&mut out, KIND_WHT2, 2, m.rows, m.cols, m.padded_cols);
                for &c in &m.centroids {
                    out.extend_from_slice(&c.to_le_bytes());
                }
                push_u16_slice(&mut out, &m.scales_f16);
                push_u32_slice(&mut out, &m.packed_words);
                push_u32_slice(&mut out, &m.outlier_row_ptr);
                push_u32_slice(&mut out, &m.outliers_packed);
            }
            PackedTensor::RawF32 { rows, cols, data } => {
                push_hdr(&mut out, KIND_RAW_F32, 0, *rows, *cols, *cols);
                push_f32_slice(&mut out, data);
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
    Ok(read_vxt_model(bytes)?.1)
}

/// Parses both the metadata KV table and all compressed tensors from a `.vxt` (`VXT1`) archive.
pub fn read_vxt_model(bytes: &[u8]) -> Result<(Vec<(String, Vec<u8>)>, Vec<VxtEntry>)> {
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
    let mcount = r.u32()? as usize;

    let mut meta_kv = Vec::with_capacity(mcount.min(1024));
    for _ in 0..mcount {
        let klen = r.u32()? as usize;
        let k = std::str::from_utf8(r.take(klen)?)
            .map_err(|_| Error::Format("invalid UTF-8 metadata key in VXT".into()))?
            .to_owned();
        let v = r.u8_vec()?;
        meta_kv.push((k, v));
    }

    let mut entries = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        r.align64()?;
        let section_start = r.pos;
        let name_len = r.u32()? as usize;
        let name = std::str::from_utf8(r.take(name_len)?)
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
                let centroids = r.f32_16()?;
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
            KIND_WHT2 => {
                if sub != 2 || padded_cols != cols.div_ceil(256) * 256 {
                    return Err(Error::Format("invalid VXT WHT2 header".into()));
                }
                let centroids = r.f32_16()?;
                let scales_f16 = r.u16_vec()?;
                let packed_words = r.u32_vec()?;
                let outlier_row_ptr = r.u32_vec()?;
                let outliers_packed = r.u32_vec()?;

                let groups = rows * (padded_cols / 128);
                let expected_words = rows * (padded_cols / 256) * 16;
                if scales_f16.len() != groups
                    || packed_words.len() != expected_words
                    || outlier_row_ptr.len() != rows + 1
                    || *outlier_row_ptr.last().unwrap() as usize != outliers_packed.len()
                {
                    return Err(Error::Format("corrupt VXT WHT2 plane lengths".into()));
                }
                PackedTensor::Wht2(Wht2Matrix {
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
            KIND_RAW_F32 => {
                let data = r.f32_vec()?;
                if data.len() != rows.saturating_mul(cols) {
                    return Err(Error::Format("corrupt VXT RawF32 length".into()));
                }
                PackedTensor::RawF32 { rows, cols, data }
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
    Ok((meta_kv, entries))
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

    fn f32_16(&mut self) -> Result<[f32; 16]> {
        let mut out = [0.0f32; 16];
        for slot in &mut out {
            let s = self.take(4)?;
            *slot = f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        }
        Ok(out)
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

    fn f32_vec(&mut self) -> Result<Vec<f32>> {
        let n = self.usize64()?;
        let bytes = self.take(
            n.checked_mul(4)
                .ok_or_else(|| Error::Format("f32 overflow".into()))?,
        )?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}
