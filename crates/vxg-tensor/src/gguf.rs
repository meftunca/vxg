//! Zero-dependency `.gguf` (`v2`/`v3`) binary container parser & block dequantizer (`F32`, `F16`,
//! `BF16`, `Q8_0`, `Q4_0`, `Q4_K`) for head-to-head benchmarking against LM Studio GGUF weights.

use crate::fp16::{bf16_to_f32, f16_to_f32};
use crate::{DType, Error, Result};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q4_K,
    BF16,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            30 => Self::BF16,
            other => Self::Other(other),
        }
    }

    pub fn byte_size(self, numel: usize) -> Option<usize> {
        match self {
            Self::F32 => numel.checked_mul(4),
            Self::F16 | Self::BF16 => numel.checked_mul(2),
            Self::Q4_0 => (numel % 32 == 0).then(|| (numel / 32) * 18),
            Self::Q8_0 => (numel % 32 == 0).then(|| (numel / 32) * 34),
            Self::Q4_K => (numel % 256 == 0).then(|| (numel / 256) * 144),
            Self::Other(_) => None,
        }
    }

    pub fn is_supported_matrix(self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::Q4_0 | Self::Q8_0 | Self::Q4_K
        )
    }
}

#[derive(Clone, Debug)]
pub struct GgufTensorMeta {
    pub name: String,
    pub ne: Vec<usize>,
    pub ggml_type: GgmlType,
    pub rel_offset: usize,
}

impl GgufTensorMeta {
    pub fn numel(&self) -> usize {
        self.ne.iter().copied().product()
    }

    /// In GGUF (`ggml`), `ne[0]` is contiguous `cols` and `ne[1..]` product is `rows`.
    pub fn matrix_dims(&self) -> Option<(usize, usize)> {
        if self.ne.len() < 2 {
            return None;
        }
        let cols = self.ne[0];
        let rows: usize = self.ne[1..].iter().copied().product();
        (rows > 0 && cols > 0).then_some((rows, cols))
    }
}

pub struct GgufView<'a> {
    pub version: u32,
    pub alignment: usize,
    data_section: &'a [u8],
    pub tensors: Vec<GgufTensorMeta>,
}

impl<'a> GgufView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let mut r = BinReader { b: bytes, pos: 0 };
        let magic = r.take(4)?;
        if magic != b"GGUF" {
            return Err(Error::Format("not a GGUF file (missing GGUF magic)".into()));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(Error::Unsupported(format!(
                "unsupported GGUF version {version} (expected 2 or 3)"
            )));
        }
        let tensor_count = usize::try_from(r.u64()?)
            .map_err(|_| Error::Format("GGUF tensor_count overflow".into()))?;
        let kv_count = usize::try_from(r.u64()?)
            .map_err(|_| Error::Format("GGUF kv_count overflow".into()))?;

        let mut alignment = 32usize;
        for _ in 0..kv_count {
            let key = r.gguf_str()?;
            let vtype = r.u32()?;
            if key == "general.alignment" && vtype == 4 {
                let a = r.u32()? as usize;
                if a.is_power_of_two() && a > 0 {
                    alignment = a;
                }
            } else if key == "general.alignment" && vtype == 10 {
                let a = r.u64()? as usize;
                if a.is_power_of_two() && a > 0 {
                    alignment = a;
                }
            } else {
                r.skip_gguf_val(vtype)?;
            }
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(65536));
        for _ in 0..tensor_count {
            let name = r.gguf_str()?;
            let n_dims = r.u32()? as usize;
            if n_dims > 8 {
                return Err(Error::Format("GGUF tensor n_dims > 8".into()));
            }
            let mut ne = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let d = usize::try_from(r.u64()?)
                    .map_err(|_| Error::Format("GGUF dimension overflow".into()))?;
                ne.push(d);
            }
            let ggml_type = GgmlType::from_u32(r.u32()?);
            let rel_offset = usize::try_from(r.u64()?)
                .map_err(|_| Error::Format("GGUF offset overflow".into()))?;
            tensors.push(GgufTensorMeta {
                name,
                ne,
                ggml_type,
                rel_offset,
            });
        }

        let rem = r.pos % alignment;
        let data_start = if rem == 0 {
            r.pos
        } else {
            r.pos + (alignment - rem)
        };
        if data_start > bytes.len() {
            return Err(Error::Format("GGUF data_start past EOF".into()));
        }
        Ok(Self {
            version,
            alignment,
            data_section: &bytes[data_start..],
            tensors,
        })
    }

    pub fn tensor_bytes(&self, meta: &GgufTensorMeta) -> Result<&'a [u8]> {
        let len = meta.ggml_type.byte_size(meta.numel()).ok_or_else(|| {
            Error::Unsupported(format!("unsupported GGUF type {:?}", meta.ggml_type))
        })?;
        let end = meta
            .rel_offset
            .checked_add(len)
            .ok_or_else(|| Error::Format("GGUF tensor end overflow".into()))?;
        if end > self.data_section.len() {
            return Err(Error::Format("GGUF tensor slice past EOF".into()));
        }
        Ok(&self.data_section[meta.rel_offset..end])
    }

    /// Extracts raw `u16` words if the GGUF tensor is stored natively in `F16` or `BF16`.
    pub fn extract_fp16_words(&self, meta: &GgufTensorMeta) -> Result<Option<(DType, Vec<u16>)>> {
        let dt = match meta.ggml_type {
            GgmlType::F16 => DType::F16,
            GgmlType::BF16 => DType::BF16,
            _ => return Ok(None),
        };
        let raw = self.tensor_bytes(meta)?;
        let words = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Some((dt, words)))
    }

    /// Materializes or dequantizes any supported GGUF tensor (`F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`,
    /// `Q4_K`) into `Vec<f32>`.
    pub fn extract_f32(&self, meta: &GgufTensorMeta) -> Result<Vec<f32>> {
        let raw = self.tensor_bytes(meta)?;
        let n = meta.numel();
        let mut out = Vec::with_capacity(n);
        match meta.ggml_type {
            GgmlType::F32 => {
                for c in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            GgmlType::F16 => {
                for c in raw.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            GgmlType::BF16 => {
                for c in raw.chunks_exact(2) {
                    out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            GgmlType::Q4_0 => {
                for blk in raw.chunks_exact(18) {
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let qs = &blk[2..18];
                    for &b in qs {
                        out.push(((b & 0x0f) as i32 - 8) as f32 * d);
                    }
                    for &b in qs {
                        out.push(((b >> 4) as i32 - 8) as f32 * d);
                    }
                }
            }
            GgmlType::Q8_0 => {
                for blk in raw.chunks_exact(34) {
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for &q in &blk[2..34] {
                        out.push((q as i8 as f32) * d);
                    }
                }
            }
            GgmlType::Q4_K => {
                for blk in raw.chunks_exact(144) {
                    dequant_q4_k_block(blk, &mut out);
                }
            }
            GgmlType::Other(code) => {
                return Err(Error::Unsupported(format!(
                    "unsupported GGUF tensor type code {code}"
                )))
            }
        }
        Ok(out)
    }
}

fn dequant_q4_k_block(blk: &[u8], out: &mut Vec<f32>) {
    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
    let sc = &blk[4..16];
    let qs = &blk[16..144];

    let get_scale_min = |j: usize| -> (u8, u8) {
        if j < 4 {
            (sc[j] & 63, sc[j + 4] & 63)
        } else {
            (
                (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4),
                (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
            )
        }
    };

    for chunk in 0..4 {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let (s1, m1) = get_scale_min(chunk * 2);
        let (s2, m2) = get_scale_min(chunk * 2 + 1);
        let d1 = d * (s1 as f32);
        let dm1 = dmin * (m1 as f32);
        let d2 = d * (s2 as f32);
        let dm2 = dmin * (m2 as f32);
        for &b in q {
            out.push(d1 * ((b & 0x0f) as f32) - dm1);
        }
        for &b in q {
            out.push(d2 * ((b >> 4) as f32) - dm2);
        }
    }
}

struct BinReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| Error::Format("GGUF reader overflow".into()))?;
        if end > self.b.len() {
            return Err(Error::Format("truncated GGUF file".into()));
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes(s.try_into().unwrap()))
    }

    fn gguf_str(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?)
            .map_err(|_| Error::Format("GGUF string length overflow".into()))?;
        let s = self.take(len)?;
        std::str::from_utf8(s)
            .map(str::to_owned)
            .map_err(|_| Error::Format("invalid UTF-8 in GGUF string".into()))
    }

    fn skip_gguf_val(&mut self, vtype: u32) -> Result<()> {
        match vtype {
            0 | 1 | 7 => {
                let _ = self.take(1)?;
            }
            2 | 3 => {
                let _ = self.take(2)?;
            }
            4..=6 => {
                let _ = self.take(4)?;
            }
            8 => {
                let len = usize::try_from(self.u64()?)
                    .map_err(|_| Error::Format("GGUF string skip overflow".into()))?;
                let _ = self.take(len)?;
            }
            9 => {
                let elem_type = self.u32()?;
                let n = usize::try_from(self.u64()?)
                    .map_err(|_| Error::Format("GGUF array skip overflow".into()))?;
                for _ in 0..n {
                    self.skip_gguf_val(elem_type)?;
                }
            }
            10..=12 => {
                let _ = self.take(8)?;
            }
            other => {
                return Err(Error::Format(format!(
                    "unknown GGUF metadata value type {other}"
                )))
            }
        }
        Ok(())
    }
}
