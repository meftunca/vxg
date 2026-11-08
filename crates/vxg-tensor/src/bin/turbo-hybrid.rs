//! Diagnostic only: splice validated existing sections, never re-encode tensors.
use std::{error::Error, fs, io::Write};
use vxg_tensor::format::{read_vxt_model, PackedTensor};
fn hash(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325, |h, x| {
        (h ^ u64::from(*x)).wrapping_mul(0x100000001b3)
    })
}
fn num(b: &[u8], p: &mut usize, n: usize) -> usize {
    let mut a = [0; 8];
    a[..n].copy_from_slice(&b[*p..*p + n]);
    *p += n;
    u64::from_le_bytes(a) as usize
}
fn layout(b: &[u8]) -> (usize, Vec<(usize, usize)>) {
    let mut p = 8;
    let nt = num(b, &mut p, 4);
    let nm = num(b, &mut p, 4);
    for _ in 0..nm {
        let n = num(b, &mut p, 4);
        p += n;
        let n = num(b, &mut p, 8);
        p += n;
    }
    let meta = p;
    let mut sections = vec![];
    for _ in 0..nt {
        p = (p + 63) / 64 * 64;
        let start = p;
        let n = num(b, &mut p, 4);
        p += n;
        let kind = b[p];
        p += 32;
        match kind {
            5 => {
                let n = num(b, &mut p, 8);
                p += n * 4;
            }
            1 => {
                p += 8;
                for w in [1, 4, 4, 1] {
                    let n = num(b, &mut p, 8);
                    p += n * w;
                }
            }
            7 => {
                let n = num(b, &mut p, 4);
                for _ in 0..n {
                    let n = num(b, &mut p, 4);
                    p += n;
                }
            }
            _ => panic!("unsupported kind {kind}"),
        };
        p += 8;
        sections.push((start, p));
    }
    assert_eq!(p, b.len() - 8);
    (meta, sections)
}
fn group(n: &str) -> Option<&'static str> {
    if n == "token_embd.weight" || n == "output.weight" {
        Some("embedding-head")
    } else if n.ends_with("ffn_gate.weight") || n.ends_with("ffn_up.weight") {
        Some("ffn-gate-up")
    } else if n.ends_with("ffn_down.weight") {
        Some("ffn-down")
    } else if [
        "attn_qkv.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_gate.weight",
    ]
    .iter()
    .any(|x| n.ends_with(x))
    {
        Some("attention-input")
    } else if n.ends_with("attn_output.weight") || n.ends_with("ssm_out.weight") {
        Some("output-projection")
    } else {
        None
    }
}
fn decoded_hash(t: &PackedTensor) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut row = vec![0.; t.cols()];
    for r in 0..t.rows() {
        t.decode_row(r, &mut row);
        for v in &row {
            for b in v.to_bits().to_le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}
fn splice(b: &[u8], z: &[u8], g: &str) -> Result<(Vec<u8>, serde_json::Value), Box<dyn Error>> {
    let (bm, be) = read_vxt_model(b)?;
    let (zm, ze) = read_vxt_model(z)?;
    let markers: Vec<_> = zm
        .iter()
        .filter(|(k, _)| k == "model.turbo_entropy_scale")
        .collect();
    let filtered: Vec<_> = zm
        .iter()
        .filter(|(k, _)| k != "model.turbo_entropy_scale")
        .cloned()
        .collect();
    if markers.len() != 1 || markers[0].1 != b"0" || filtered != bm {
        return Err("metadata mismatch beyond sole zero-entropy provenance".into());
    }
    if be.len() != ze.len() {
        return Err("count mismatch".into());
    }
    let (mp, bs) = layout(b);
    let (_, zs) = layout(z);
    let mut out = b[..mp].to_vec();
    let count = u32::from_le_bytes(out[12..16].try_into()?) + 1;
    out[12..16].copy_from_slice(&count.to_le_bytes());
    let key = b"diagnostic.selective_entropy";
    let value = format!("{g};base-fnv={:016x};zero-fnv={:016x}", hash(b), hash(z));
    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
    out.extend_from_slice(key);
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    let mut audit = vec![];
    for (i, (a, c)) in be.iter().zip(&ze).enumerate() {
        if a.name != c.name
            || a.tensor.rows() != c.tensor.rows()
            || a.tensor.cols() != c.tensor.cols()
        {
            return Err("shape/name mismatch".into());
        }
        let (s, e) = bs[i];
        let (u, v) = zs[i];
        if b[s..s + 4 + a.name.len() + 32] != z[u..u + 4 + c.name.len() + 32] {
            return Err("header/codec mismatch".into());
        }
        let turbo = matches!(a.tensor, PackedTensor::TurboWht(_));
        if turbo && group(&a.name).is_none() {
            return Err(format!("uncovered EC {}", a.name).into());
        }
        if !turbo && b[s..e] != z[u..v] {
            return Err(format!("non-Turbo changed {}", a.name).into());
        }
        let selected = turbo && group(&a.name) == Some(g);
        while out.len() % 64 != 0 {
            out.push(0)
        }
        out.extend_from_slice(if selected { &z[u..v] } else { &b[s..e] });
        audit.push(serde_json::json!({"name":a.name,"group":if turbo{group(&a.name)}else{None},"selected":selected,"baseline_section_bytes":e-s,"zero_section_bytes":v-u,"baseline_encoded_fnv":format!("{:016x}",hash(&b[s..e])),"zero_encoded_fnv":format!("{:016x}",hash(&z[u..v]))}));
    }
    out.extend_from_slice(&hash(&out).to_le_bytes());
    let (_, oe) = read_vxt_model(&out)?;
    for (i, (a, o)) in be.iter().zip(&oe).enumerate() {
        let selected = audit[i]["selected"] == true;
        let expected = if selected { &ze[i].tensor } else { &a.tensor };
        let h = decoded_hash(expected);
        if h != decoded_hash(&o.tensor) {
            return Err("decoded reconstruction mismatch".into());
        }
        audit[i]["expected_and_output_decoded_f32_fnv"] = serde_json::json!(format!("{h:016x}"));
    }
    Ok((out, serde_json::json!({"group":g,"tensors":audit})))
}
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("BASE ZERO GROUP NEW_OUTPUT.vxt".into());
    }
    if ![
        "embedding-head",
        "ffn-gate-up",
        "ffn-down",
        "attention-input",
        "output-projection",
    ]
    .contains(&a[3].as_str())
    {
        return Err("invalid group".into());
    }
    let b = fs::read(&a[1])?;
    let z = fs::read(&a[2])?;
    let (out, audit) = splice(&b, &z, &a[3])?;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&a[4])?
        .write_all(&out)?;
    fs::write(
        format!("{}.audit.json", a[4]),
        serde_json::to_vec_pretty(&audit)?,
    )?;
    println!(
        "{} bytes; preserved all nonselected sections and reconstructed f32 checksums",
        out.len()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use vxg_tensor::format::{write_vxt_model, VxtEntry};
    fn tiny(meta: bool, rows: usize) -> Vec<u8> {
        write_vxt_model(
            &if meta {
                vec![("model.turbo_entropy_scale".into(), b"0".to_vec())]
            } else {
                vec![]
            },
            &[VxtEntry {
                name: "norm".into(),
                tensor: PackedTensor::RawF32 {
                    rows,
                    cols: 1,
                    data: vec![1.; rows],
                },
            }],
        )
        .unwrap()
    }
    #[test]
    fn preserve() {
        let (b, _) = splice(&tiny(false, 1), &tiny(true, 1), "ffn-down").unwrap();
        let (_, s) = layout(&b);
        let a = tiny(false, 1);
        let (_, t) = layout(&a);
        assert_eq!(b[s[0].0..s[0].1], a[t[0].0..t[0].1]);
    }
    #[test]
    fn rejects_shape() {
        assert!(splice(&tiny(false, 1), &tiny(true, 2), "ffn-down").is_err())
    }
    #[test]
    fn rejects_metadata() {
        assert!(splice(&tiny(false, 1), &tiny(false, 1), "ffn-down").is_err())
    }
}
