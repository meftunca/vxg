use std::{fs, process::Command};
use vxg_tensor::{format::read_vxt_archive, safetensors::SafeTensorsView};

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vxg-tensor")).args(args).output().unwrap()
}

#[test]
fn default_pack_preserves_every_supported_source_tensor_and_explicit_subsets() {
    let dir = std::env::temp_dir().join(format!("vxt-pack-contract-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let input = dir.join("mixed.safetensors");
    let output = dir.join("mixed.vxt");
    let mut header = String::from("{");
    let mut payload = Vec::new();
    for (i, (name, dtype, shape, n)) in [
        ("bf_vector", "BF16", "[7]", 7),
        ("fp_small", "F16", "[2,3]", 6),
        ("f32_vector", "F32", "[8]", 8),
        ("f32_large", "F32", "[64,64]", 4096),
        ("bf_rank3", "BF16", "[2,2,3]", 12),
    ].iter().enumerate() {
        let start = payload.len();
        for j in 0..*n {
            if *dtype == "F32" {
                payload.extend([0x3f801234u32, 0xbf819abc, 1, 0x80000000, 0x7fc01234][j % 5].to_le_bytes());
            } else {
                payload.extend((0x3001u16 + j as u16).to_le_bytes());
            }
        }
        if i > 0 { header.push(','); }
        header.push_str(&format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[{start},{}]}}", payload.len()));
    }
    header.push('}');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.bytes()); bytes.extend(payload);
    fs::write(&input, &bytes).unwrap();
    let args = ["pack", input.to_str().unwrap(), output.to_str().unwrap()];
    let result = run(&args);
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let entries = read_vxt_archive(&fs::read(&output).unwrap()).unwrap();
    let st = SafeTensorsView::parse(&bytes).unwrap();
    assert_eq!(entries.len(), st.tensors.len());
    for meta in &st.tensors {
        let actual = entries.iter().find(|e| e.name == meta.name).unwrap().tensor.decode_f32();
        let expected = st.extract_f32(meta).unwrap();
        assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }
    let result = run(&[args[0], args[1], args[2], "--max-tensors", "0"]);
    assert!(!result.status.success());
    let result = run(&[args[0], args[1], args[2], "--max-tensors", "2"]);
    assert!(result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("subset"));
    assert_eq!(read_vxt_archive(&fs::read(&output).unwrap()).unwrap().len(), 2);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn default_gguf_pack_preserves_f32_vectors_and_rejects_unsupported_types() {
    let dir = std::env::temp_dir().join(format!("vxt-pack-gguf-contract-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let input = dir.join("source.gguf");
    let output = dir.join("source.vxt");
    let expected = [0x3f801234u32, 1, 0x80000000, 0x7fc01234];
    for kind in [0u32, 999] {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(6u64.to_le_bytes()); bytes.extend(b"vector");
        bytes.extend(1u32.to_le_bytes()); bytes.extend(4u64.to_le_bytes());
        bytes.extend(kind.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend(expected.iter().flat_map(|v| v.to_le_bytes()));
        fs::write(&input, bytes).unwrap();
        let result = run(&["pack", input.to_str().unwrap(), output.to_str().unwrap()]);
        if kind == 0 {
            assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
            let entries = read_vxt_archive(&fs::read(&output).unwrap()).unwrap();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].tensor.decode_f32().iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected);
        } else {
            assert!(!result.status.success());
            assert!(String::from_utf8_lossy(&result.stderr).contains("unsupported"));
        }
    }
    fs::remove_dir_all(dir).unwrap();
}
