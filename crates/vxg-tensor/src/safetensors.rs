//! Zero-dependency `.safetensors` reader and writer supporting `BF16`, `F16`, and `F32` tensors.

use crate::fp16::{bf16_to_f32, f16_to_f32};
use crate::{DType, Error, Result};

#[derive(Clone, Debug)]
pub struct SafeTensorMeta {
    pub name: String,
    pub dtype: SafeDType,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SafeDType {
    BF16,
    F16,
    F32,
    Other,
}

impl SafeTensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }

    /// Returns `(rows, cols)` treating the last axis as contiguous `cols` and leading axes as `rows`.
    pub fn matrix_dims(&self) -> Option<(usize, usize)> {
        if self.shape.len() < 2 {
            return None;
        }
        let cols = *self.shape.last()?;
        let rows: usize = self.shape[..self.shape.len() - 1].iter().copied().product();
        (rows > 0 && cols > 0).then_some((rows, cols))
    }
}

pub struct SafeTensorsView<'a> {
    data_section: &'a [u8],
    pub tensors: Vec<SafeTensorMeta>,
}

impl<'a> SafeTensorsView<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::Format("safetensors file < 8 bytes".into()));
        }
        let hdr_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            .map_err(|_| Error::Format("safetensors header size overflow".into()))?;
        let data_base = 8usize
            .checked_add(hdr_len)
            .ok_or_else(|| Error::Format("safetensors header offset overflow".into()))?;
        if bytes.len() < data_base {
            return Err(Error::Format("truncated safetensors header".into()));
        }
        let hdr_str = std::str::from_utf8(&bytes[8..data_base])
            .map_err(|_| Error::Format("invalid UTF-8 in safetensors header".into()))?;
        let data_section = &bytes[data_base..];

        let mut p = JsonScanner::new(hdr_str.as_bytes());
        p.skip_ws();
        p.expect(b'{')?;
        let mut tensors = Vec::new();
        loop {
            p.skip_ws();
            if p.eat(b'}') {
                break;
            }
            let key = p.read_string()?;
            p.skip_ws();
            p.expect(b':')?;
            p.skip_ws();
            if key == "__metadata__" {
                p.skip_value()?;
            } else {
                let meta = p.read_tensor_entry(key, data_section.len())?;
                tensors.push(meta);
            }
            p.skip_ws();
            if p.eat(b',') {
                continue;
            }
            p.expect(b'}')?;
            break;
        }
        Ok(Self {
            data_section,
            tensors,
        })
    }

    pub fn tensor_bytes(&self, meta: &SafeTensorMeta) -> &'a [u8] {
        &self.data_section[meta.data_start..meta.data_end]
    }

    /// Extracts raw `u16` words + native [`DType`] (`BF16` or `F16`) if 16-bit floating-point.
    pub fn extract_fp16_words(&self, meta: &SafeTensorMeta) -> Result<Option<(DType, Vec<u16>)>> {
        let dt = match meta.dtype {
            SafeDType::BF16 => DType::BF16,
            SafeDType::F16 => DType::F16,
            _ => return Ok(None),
        };
        let raw = self.tensor_bytes(meta);
        if raw.len() != meta.numel() * 2 {
            return Err(Error::Format(
                "safetensors 16-bit byte length mismatch".into(),
            ));
        }
        let words: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Some((dt, words)))
    }

    /// Materializes a tensor as `Vec<f32>` (`BF16`, `F16`, or `F32`).
    pub fn extract_f32(&self, meta: &SafeTensorMeta) -> Result<Vec<f32>> {
        let raw = self.tensor_bytes(meta);
        let n = meta.numel();
        match meta.dtype {
            SafeDType::BF16 => {
                if raw.len() != n * 2 {
                    return Err(Error::Format("BF16 byte length mismatch".into()));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            SafeDType::F16 => {
                if raw.len() != n * 2 {
                    return Err(Error::Format("F16 byte length mismatch".into()));
                }
                Ok(raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            SafeDType::F32 => {
                if raw.len() != n * 4 {
                    return Err(Error::Format("F32 byte length mismatch".into()));
                }
                Ok(raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            SafeDType::Other => Err(Error::Unsupported(
                "unsupported safetensors dtype (expected BF16/F16/F32)".into(),
            )),
        }
    }
}

/// Encodes a minimal valid `.safetensors` buffer for one or more 16-bit (`BF16`/`F16`) matrices.
pub fn write_safetensors_u16(entries: &[(&str, DType, &[usize], &[u16])]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (idx, &(name, dt, shape, words)) in entries.iter().enumerate() {
        if idx > 0 {
            header.push(',');
        }
        let dt_str = match dt {
            DType::BF16 => "BF16",
            DType::F16 => "F16",
        };
        let byte_len = words.len() * 2;
        let shape_str = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dt_str}\",\"shape\":[{shape_str}],\"data_offsets\":[{},{}]}}",
            offset,
            offset + byte_len
        ));
        offset += byte_len;
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for &(_, _, _, words) in entries {
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

struct JsonScanner<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> JsonScanner<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, pos: 0 }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && matches!(self.b[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
    }

    fn eat(&mut self, ch: u8) -> bool {
        if self.b.get(self.pos) == Some(&ch) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, ch: u8) -> Result<()> {
        if self.eat(ch) {
            Ok(())
        } else {
            Err(Error::Format(format!(
                "safetensors JSON expected '{}' at pos {}",
                ch as char, self.pos
            )))
        }
    }

    fn read_string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        while self.pos < self.b.len() {
            let c = self.b[self.pos];
            self.pos += 1;
            if c == b'"' {
                return String::from_utf8(out)
                    .map_err(|_| Error::Format("invalid UTF-8 string in safetensors".into()));
            }
            if c == b'\\' {
                let esc = *self
                    .b
                    .get(self.pos)
                    .ok_or_else(|| Error::Format("unterminated escape".into()))?;
                self.pos += 1;
                match esc {
                    b'"' | b'\\' | b'/' => out.push(esc),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        if self.pos + 4 > self.b.len() {
                            return Err(Error::Format("short \\u escape".into()));
                        }
                        let hex = std::str::from_utf8(&self.b[self.pos..self.pos + 4])
                            .map_err(|_| Error::Format("bad \\u hex".into()))?;
                        let cp = u32::from_str_radix(hex, 16)
                            .map_err(|_| Error::Format("bad \\u codepoint".into()))?;
                        self.pos += 4;
                        if let Some(ch) = char::from_u32(cp) {
                            let mut tmp = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                        }
                    }
                    _ => out.push(esc),
                }
            } else {
                out.push(c);
            }
        }
        Err(Error::Format("unterminated JSON string".into()))
    }

    fn read_usize(&mut self) -> Result<usize> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(Error::Format("expected integer in safetensors JSON".into()));
        }
        std::str::from_utf8(&self.b[start..self.pos])
            .unwrap()
            .parse::<usize>()
            .map_err(|_| Error::Format("integer overflow in safetensors JSON".into()))
    }

    fn read_usize_array(&mut self) -> Result<Vec<usize>> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        loop {
            self.skip_ws();
            if self.eat(b']') {
                break;
            }
            out.push(self.read_usize()?);
            self.skip_ws();
            if self.eat(b',') {
                continue;
            }
            self.expect(b']')?;
            break;
        }
        Ok(out)
    }

    fn skip_value(&mut self) -> Result<()> {
        self.skip_ws();
        let Some(&c) = self.b.get(self.pos) else {
            return Err(Error::Format("unexpected EOF in safetensors JSON".into()));
        };
        match c {
            b'"' => {
                let _ = self.read_string()?;
                Ok(())
            }
            b'{' => {
                self.pos += 1;
                loop {
                    self.skip_ws();
                    if self.eat(b'}') {
                        return Ok(());
                    }
                    let _ = self.read_string()?;
                    self.skip_ws();
                    self.expect(b':')?;
                    self.skip_value()?;
                    self.skip_ws();
                    if self.eat(b',') {
                        continue;
                    }
                    return self.expect(b'}');
                }
            }
            b'[' => {
                self.pos += 1;
                loop {
                    self.skip_ws();
                    if self.eat(b']') {
                        return Ok(());
                    }
                    self.skip_value()?;
                    self.skip_ws();
                    if self.eat(b',') {
                        continue;
                    }
                    return self.expect(b']');
                }
            }
            _ => {
                let start = self.pos;
                while self.pos < self.b.len()
                    && !matches!(
                        self.b[self.pos],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n'
                    )
                {
                    self.pos += 1;
                }
                if self.pos == start {
                    return Err(Error::Format("invalid token in safetensors JSON".into()));
                }
                Ok(())
            }
        }
    }

    fn read_tensor_entry(&mut self, name: String, max_data_len: usize) -> Result<SafeTensorMeta> {
        self.expect(b'{')?;
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        loop {
            self.skip_ws();
            if self.eat(b'}') {
                break;
            }
            let k = self.read_string()?;
            self.skip_ws();
            self.expect(b':')?;
            self.skip_ws();
            match k.as_str() {
                "dtype" => {
                    let s = self.read_string()?;
                    dtype = Some(match s.as_str() {
                        "BF16" => SafeDType::BF16,
                        "F16" => SafeDType::F16,
                        "F32" => SafeDType::F32,
                        _ => SafeDType::Other,
                    });
                }
                "shape" => {
                    shape = Some(self.read_usize_array()?);
                }
                "data_offsets" => {
                    let arr = self.read_usize_array()?;
                    if arr.len() != 2 {
                        return Err(Error::Format("data_offsets must have 2 elements".into()));
                    }
                    offsets = Some((arr[0], arr[1]));
                }
                _ => self.skip_value()?,
            }
            self.skip_ws();
            if self.eat(b',') {
                continue;
            }
            self.expect(b'}')?;
            break;
        }
        let (data_start, data_end) =
            offsets.ok_or_else(|| Error::Format("missing data_offsets".into()))?;
        if data_start > data_end || data_end > max_data_len {
            return Err(Error::Format(
                "safetensors data_offsets out of bounds".into(),
            ));
        }
        Ok(SafeTensorMeta {
            name,
            dtype: dtype.ok_or_else(|| Error::Format("missing dtype".into()))?,
            shape: shape.ok_or_else(|| Error::Format("missing shape".into()))?,
            data_start,
            data_end,
        })
    }
}
