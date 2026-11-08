//! Fail-closed source-weight qualification, independent of the model packer.
//! Usage: cargo run -p vxg-tensor --release --bin verify-lossless -- SOURCE.gguf|SOURCE.safetensors ARCHIVE.vxt
//! Native16 BP16 is compared as raw words; fallback storage is compared as f32 bits.
use std::collections::HashMap;
use vxg_tensor::{
    format::{read_vxt_model, PackedTensor},
    gguf::{GgmlType, GgufView},
    mmap::TensorBytes,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: verify-lossless SOURCE.gguf|SOURCE.safetensors ARCHIVE.vxt".into());
    }
    let source = TensorBytes::open(&args[1])?;
    let archive = TensorBytes::open(&args[2])?;
    verify(&source, &archive)
}
fn verify(source: &[u8], archive: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let tensors = source_tensors(source)?;
    let (_, entries) = read_vxt_model(archive)?;
    let mut by_name = HashMap::new();
    for e in &entries {
        if by_name.insert(e.name.as_str(), &e.tensor).is_some() {
            return Err(format!("duplicate VXT tensor {}", e.name).into());
        }
    }
    if tensors.len() != entries.len() {
        return Err(format!(
            "tensor count source={} archive={}; every source tensor is required",
            tensors.len(),
            entries.len()
        )
        .into());
    }
    let mut source_names = std::collections::HashSet::new();
    let (mut words, mut values) = (0usize, 0usize);
    for meta in &tensors {
        if !source_names.insert(&meta.name) {
            return Err(format!("duplicate GGUF tensor {}", meta.name).into());
        }
        let packed = by_name
            .remove(meta.name.as_str())
            .ok_or_else(|| format!("missing {}", meta.name))?;
        let dims = meta.dims;
        if (packed.rows(), packed.cols()) != dims {
            return Err(format!("shape mismatch {}", meta.name).into());
        }
        let raw = meta.raw;
        if let PackedTensor::Lossless(m) = packed {
            let dtype_ok = matches!(
                (meta.ggml_type, m.dtype),
                (GgmlType::F16, vxg_tensor::DType::F16) | (GgmlType::BF16, vxg_tensor::DType::BF16)
            );
            if !dtype_ok {
                return Err(format!("dtype mismatch {}", meta.name).into());
            }
            let decoded = m.decode_u16();
            for (i, (&a, b)) in decoded.iter().zip(raw.chunks_exact(2)).enumerate() {
                if a != u16::from_le_bytes([b[0], b[1]]) {
                    return Err(format!("raw16 mismatch {} index={i}", meta.name).into());
                }
            }
            if decoded.len() * 2 != raw.len() {
                return Err("raw16 length mismatch".into());
            }
            words += decoded.len();
        } else {
            let decoded = packed.decode_f32();
            if decoded.len() != meta.dims.0 * meta.dims.1 {
                return Err("decoded length mismatch".into());
            }
            for (i, a) in decoded.iter().enumerate() {
                let expected = match meta.ggml_type {
                    GgmlType::F32 => u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into()?),
                    GgmlType::BF16 => {
                        (u16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into()?) as u32) << 16
                    }
                    GgmlType::F16 => {
                        half_bits(u16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into()?))
                    }
                    other => {
                        return Err(format!(
                            "independent oracle unavailable for {other:?}; refusing qualification"
                        )
                        .into())
                    }
                };
                if a.to_bits() != expected {
                    return Err(format!(
                        "f32 bits mismatch {} index={i}: {:08x} != {expected:08x}",
                        meta.name,
                        a.to_bits()
                    )
                    .into());
                }
            }
            values += decoded.len();
        }
        println!(
            "PASS {} {:?} {}x{}",
            meta.name, meta.ggml_type, dims.0, dims.1
        );
    }
    if !by_name.is_empty() {
        return Err("unexpected archive tensors".into());
    }
    println!("PASS tensors={} raw16_words={} f32_bit_values={} source_bytes={} archive_bytes={} ratio={:.6}", entries.len(), words, values, source.len(), archive.len(), source.len() as f64 / archive.len() as f64);
    Ok(())
}
struct SourceTensor<'a> {
    name: String,
    ggml_type: GgmlType,
    dims: (usize, usize),
    raw: &'a [u8],
}
fn source_tensors(source: &[u8]) -> Result<Vec<SourceTensor<'_>>, Box<dyn std::error::Error>> {
    if source.starts_with(b"GGUF") {
        let gg = GgufView::parse(source)?;
        return gg
            .tensors
            .iter()
            .map(|m| {
                let n =
                    m.ne.iter()
                        .try_fold(1usize, |a, &b| a.checked_mul(b))
                        .ok_or("shape overflow")?;
                if n == 0 {
                    return Err("zero-sized source tensors unsupported".into());
                }
                let dims = if m.ne.len() >= 2 {
                    (n / m.ne[0], m.ne[0])
                } else {
                    (1, n)
                };
                Ok(SourceTensor {
                    name: m.name.clone(),
                    ggml_type: m.ggml_type,
                    dims,
                    raw: gg.tensor_bytes(m)?,
                })
            })
            .collect();
    }
    let h = usize::try_from(u64::from_le_bytes(
        source.get(..8).ok_or("short SafeTensors")?.try_into()?,
    ))?;
    if h > 100_000_000 {
        return Err("SafeTensors header exceeds 100MB limit".into());
    }
    let end = 8usize.checked_add(h).ok_or("header overflow")?;
    let header = source.get(8..end).ok_or("truncated SafeTensors header")?;
    std::str::from_utf8(header)?;
    let data = source.get(end..).ok_or("missing data")?;
    let mut j = Header { b: header, p: 0 };
    j.expect(b'{')?;
    let mut tensors = Vec::new();
    let mut names = std::collections::HashSet::new();
    if !j.eat(b'}') {
        loop {
            let name = j.string()?;
            if !names.insert(name.clone()) {
                return Err("duplicate source name".into());
            }
            j.expect(b':')?;
            j.expect(b'{')?;
            if name == "__metadata__" {
                let mut keys = std::collections::HashSet::new();
                if !j.eat(b'}') {
                    loop {
                        if !keys.insert(j.string()?) {
                            return Err("duplicate metadata key".into());
                        }
                        j.expect(b':')?;
                        j.string()?;
                        if j.eat(b'}') {
                            break;
                        }
                        j.expect(b',')?;
                    }
                }
            } else {
                let (mut dtype, mut shape, mut offsets) = (None, None, None);
                loop {
                    let key = j.string()?;
                    j.expect(b':')?;
                    match key.as_str() {
                        "dtype" if dtype.is_none() => dtype = Some(j.string()?),
                        "shape" if shape.is_none() => shape = Some(j.array()?),
                        "data_offsets" if offsets.is_none() => offsets = Some(j.array()?),
                        _ => return Err("duplicate or unknown tensor field".into()),
                    }
                    if j.eat(b'}') {
                        break;
                    }
                    j.expect(b',')?;
                }
                let kind = match dtype.as_deref() {
                    Some("BF16") => GgmlType::BF16,
                    Some("F16") => GgmlType::F16,
                    Some("F32") => GgmlType::F32,
                    _ => return Err("unsupported or missing dtype".into()),
                };
                let shape = shape.ok_or("missing shape")?;
                let n = shape
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    .ok_or("shape overflow")?;
                if n == 0 {
                    return Err("zero-sized source tensors unsupported".into());
                }
                let cols = shape.last().copied().unwrap_or(1);
                let dims = (n / cols, cols);
                let offsets = offsets.ok_or("missing offsets")?;
                if offsets.len() != 2 {
                    return Err("offset count != 2".into());
                }
                let raw = data
                    .get(offsets[0]..offsets[1])
                    .ok_or("invalid data offsets")?;
                if kind.byte_size(n) != Some(raw.len()) {
                    return Err("source byte size mismatch".into());
                }
                tensors.push((
                    offsets[0],
                    offsets[1],
                    SourceTensor {
                        name,
                        ggml_type: kind,
                        dims,
                        raw,
                    },
                ));
            }
            if j.eat(b'}') {
                break;
            }
            j.expect(b',')?;
        }
    }
    j.ws();
    if j.p != header.len() {
        return Err("trailing header data".into());
    }
    tensors.sort_by_key(|t| t.0);
    let mut next = 0;
    for (start, end, _) in &tensors {
        if *start != next {
            return Err("source overlap or gap".into());
        }
        next = *end;
    }
    if next != data.len() {
        return Err("unclaimed source data".into());
    }
    Ok(tensors.into_iter().map(|t| t.2).collect())
}
// Schema-only strict JSON reader. Rejects uncommon Unicode escapes rather than
// silently misreading names; ordinary UTF-8 names are supported. No recursion.
struct Header<'a> {
    b: &'a [u8],
    p: usize,
}
impl Header<'_> {
    fn ws(&mut self) {
        while self
            .b
            .get(self.p)
            .is_some_and(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.p += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.p) == Some(&c) {
            self.p += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: u8) -> Result<(), Box<dyn std::error::Error>> {
        if self.eat(c) {
            Ok(())
        } else {
            Err("invalid SafeTensors JSON".into())
        }
    }
    fn string(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.p).ok_or("unterminated string")?;
            self.p += 1;
            match c {
                b'"' => return Ok(String::from_utf8(out)?),
                0..=31 => return Err("control byte in string".into()),
                b'\\' => {
                    let e = *self.b.get(self.p).ok_or("unterminated escape")?;
                    self.p += 1;
                    out.push(match e {
                        b'"' | b'\\' | b'/' => e,
                        b'n' => b'\n',
                        b'r' => b'\r',
                        b't' => b'\t',
                        b'b' => 8,
                        b'f' => 12,
                        _ => {
                            return Err(
                                "unsupported JSON escape (Unicode escapes require literal UTF-8)"
                                    .into(),
                            )
                        }
                    });
                }
                _ => out.push(c),
            }
        }
    }
    fn array(&mut self) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        self.expect(b'[')?;
        let mut out = Vec::new();
        if self.eat(b']') {
            return Ok(out);
        }
        loop {
            self.ws();
            let start = self.p;
            while self.b.get(self.p).is_some_and(u8::is_ascii_digit) {
                self.p += 1;
            }
            let digits = self.b.get(start..self.p).ok_or("bad integer")?;
            if digits.is_empty() || (digits.len() > 1 && digits[0] == b'0') {
                return Err("invalid integer".into());
            }
            out.push(std::str::from_utf8(digits)?.parse()?);
            if out.len() > 1024 {
                return Err("excessive array dimensions".into());
            }
            if self.eat(b']') {
                break;
            }
            self.expect(b',')?;
        }
        Ok(out)
    }
}
// Independent arithmetic half conversion oracle (including signed zero and NaN payloads).
fn half_bits(w: u16) -> u32 {
    let sign = ((w as u32) & 0x8000) << 16;
    let e = (w >> 10) & 31;
    let m = w & 1023;
    if e == 31 {
        return sign | 0x7f800000 | ((m as u32) << 13);
    }
    let magnitude = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else {
        (1024 + m as u32) as f32 * 2f32.powi(e as i32 - 25)
    };
    sign | magnitude.to_bits()
}
#[cfg(test)]
mod tests {
    use vxg_tensor::format::{write_vxt_archive, PackedTensor, VxtEntry};
    fn source() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        for (i, name) in ["a", "b"].iter().enumerate() {
            b.extend(1u64.to_le_bytes());
            b.extend(name.as_bytes());
            b.extend(2u32.to_le_bytes());
            b.extend(2u64.to_le_bytes());
            b.extend(1u64.to_le_bytes());
            b.extend(0u32.to_le_bytes());
            b.extend((i as u64 * 32).to_le_bytes());
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        for value in [1.0f32, 2.0] {
            b.extend(value.to_le_bytes());
        }
        b.resize(b.len() + 24, 0);
        for value in [3.0f32, 4.0] {
            b.extend(value.to_le_bytes());
        }
        b
    }
    fn entries() -> Vec<VxtEntry> {
        [("a", vec![1.0, 2.0]), ("b", vec![3.0, 4.0])]
            .into_iter()
            .map(|(name, data)| VxtEntry {
                name: name.into(),
                tensor: PackedTensor::RawF32 {
                    rows: 1,
                    cols: 2,
                    data,
                },
            })
            .collect()
    }
    #[test]
    fn every_source_tensor_and_rejection_contract() {
        let src = source();
        let good = write_vxt_archive(&entries()).unwrap();
        super::verify(&src, &good).unwrap();
        let mut corrupt = good.clone();
        corrupt[24] ^= 1;
        assert!(super::verify(&src, &corrupt).is_err());
        let mut missing = entries();
        missing.pop();
        assert!(super::verify(&src, &write_vxt_archive(&missing).unwrap()).is_err());
        let mut duplicate = entries();
        duplicate[1].name = "a".into();
        assert!(super::verify(&src, &write_vxt_archive(&duplicate).unwrap()).is_err());
        let mut shape = entries();
        shape[1].tensor = PackedTensor::RawF32 {
            rows: 2,
            cols: 1,
            data: vec![3.0, 4.0],
        };
        assert!(super::verify(&src, &write_vxt_archive(&shape).unwrap()).is_err());
        let mut value = entries();
        value[1].tensor = PackedTensor::RawF32 {
            rows: 1,
            cols: 2,
            data: vec![3.0, 4.000001],
        };
        assert!(super::verify(&src, &write_vxt_archive(&value).unwrap()).is_err());
        let mut renamed = entries();
        renamed[1].name = "other".into();
        assert!(super::verify(&src, &write_vxt_archive(&renamed).unwrap()).is_err());
    }

    #[test]
    fn raw_native16_all_words_and_single_bit_rejection() {
        for (kind, dtype) in [
            (1u32, vxg_tensor::DType::F16),
            (30, vxg_tensor::DType::BF16),
        ] {
            let words: Vec<u16> = (0..=u16::MAX).collect();
            let mut src = b"GGUF".to_vec();
            src.extend(3u32.to_le_bytes());
            src.extend(1u64.to_le_bytes());
            src.extend(0u64.to_le_bytes());
            src.extend(1u64.to_le_bytes());
            src.extend(b"w");
            src.extend(2u32.to_le_bytes());
            src.extend(256u64.to_le_bytes());
            src.extend(256u64.to_le_bytes());
            src.extend(kind.to_le_bytes());
            src.extend(0u64.to_le_bytes());
            src.resize(src.len().div_ceil(32) * 32, 0);
            src.extend(words.iter().flat_map(|w| w.to_le_bytes()));
            let tensor = PackedTensor::Lossless(
                vxg_tensor::lossless::LosslessMatrix::encode(dtype, 256, 256, &words).unwrap(),
            );
            let archive = write_vxt_archive(&[VxtEntry {
                name: "w".into(),
                tensor,
            }])
            .unwrap();
            super::verify(&src, &archive).unwrap();
            let last = src.len() - 1;
            src[last] ^= 1;
            assert!(super::verify(&src, &archive).is_err());
        }
    }
    fn st(header: &str, data: &[u8]) -> Vec<u8> {
        let mut b = (header.len() as u64).to_le_bytes().to_vec();
        b.extend(header.as_bytes());
        b.extend(data);
        b
    }
    #[test]
    fn safetensors_native_types_and_malformed_sources() {
        for (kind, dtype) in [
            ("BF16", vxg_tensor::DType::BF16),
            ("F16", vxg_tensor::DType::F16),
        ] {
            let words = [0x8000u16, 1, 0x7fff, 0x3555];
            let payload: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let src = st(
                &format!(
                    r#"{{"__metadata__":{{"note":"source"}},"w":{{"dtype":"{kind}","shape":[2,1,2],"data_offsets":[0,8]}}}}"#
                ),
                &payload,
            );
            let tensor = PackedTensor::Lossless(
                vxg_tensor::lossless::LosslessMatrix::encode(dtype, 2, 2, &words).unwrap(),
            );
            let archive = write_vxt_archive(&[VxtEntry {
                name: "w".into(),
                tensor,
            }])
            .unwrap();
            super::verify(&src, &archive).unwrap();
            let mut changed = src.clone();
            *changed.last_mut().unwrap() ^= 1;
            assert!(super::verify(&changed, &archive).is_err());
        }
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let valid = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
        let archive = write_vxt_archive(&entries()).unwrap();
        super::verify(&st(valid, &data), &archive).unwrap();
        let invalid = [
            valid.replace("F32", "I32"),
            valid.replace("[2]", "[3]"),
            valid.replace("[8,16]", "[0,8]"),
            valid.replace("[8,16]", "[9,17]"),
            valid.replace("[2]", "[18446744073709551615,2]"),
            valid.replace("\"b\"", "\"a\""),
            valid.replace("\"b\"", "\"c\""),
            valid.replace("\"dtype\":\"F32\"", "\"dtype\":\"F32\",\"dtype\":\"F32\""),
            format!("{valid} garbage"),
            valid.replace("[2]", "[2,]"),
            valid.replace("[2]", "[02]"),
            valid.replace("\"a\"", r#""\u0061""#),
            valid.replace("[2]", "[0]"),
        ];
        for header in invalid {
            assert!(
                super::verify(&st(&header, &data), &archive).is_err(),
                "accepted {header}"
            );
        }
        assert!(super::verify(&[0; 4], &archive).is_err());
        assert!(super::verify(&u64::MAX.to_le_bytes(), &archive).is_err());
        let mut short = st(valid, &data);
        short.pop();
        assert!(super::verify(&short, &archive).is_err());
    }
    #[test]
    fn independent_half_oracle_exhaustive() {
        for w in 0..=u16::MAX {
            assert_eq!(
                super::half_bits(w),
                vxg_tensor::fp16::f16_to_f32(w).to_bits(),
                "{w:04x}"
            );
        }
    }
}
