//! Unit tests for `native_qwen35` lossless text-trunk contract and archive bit-preservation.

#![cfg(test)]

use super::*;

#[test]
fn native_shape_contract_has_complete_text_trunk() {
    let e = expected();
    assert_eq!(e.len(), 320);
    assert_eq!(
        e["model.language_model.layers.0.linear_attn.conv1d.weight"].1,
        vec![6144, 1, 4]
    );
    assert_eq!(
        e["model.language_model.layers.0.linear_attn.norm.weight"].2,
        "F32"
    );
    assert!(!e
        .keys()
        .any(|k| k.starts_with("mtp.") || k.starts_with("model.visual.")));
}

#[test]
fn original_bf16_bits_and_rank_survive_archive() {
    let words = vec![0, 0x8000, 0x7fc1, 0xff80, 0x3f80, 0x0001];
    let b = crate::safetensors::write_safetensors_u16(&[(
        "native.weight",
        DType::BF16,
        &[2, 1, 3],
        &words,
    )]);
    let st = SafeTensorsView::parse(&b).unwrap();
    let t = encode_tensor(&st, &st.tensors[0]).unwrap();
    let meta = vec![(
        "shape".into(),
        serde_json::to_vec(&descriptor(&st.tensors[0].shape, "BF16")).unwrap(),
    )];
    let archive = write_vxt_model(
        &meta,
        &[VxtEntry {
            name: st.tensors[0].name.clone(),
            tensor: t,
        }],
    )
    .unwrap();
    let (m, e) = crate::format::read_vxt_model(&archive).unwrap();
    assert_eq!(m, meta);
    assert_eq!(e[0].name, "native.weight");
    match &e[0].tensor {
        PackedTensor::Lossless(x) => assert_eq!(x.decode_u16(), words),
        _ => panic!("wrong storage"),
    };
    assert_eq!(parse(&m[0].1).unwrap()["shape"], json!([2, 1, 3]));
}

#[test]
fn original_f32_payload_bits_survive_archive() {
    let bits = [
        0x00000000u32,
        0x80000000,
        0x7fc12345,
        0xff800000,
        0x00000001,
        0x3f800000,
    ];
    let data = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let archive = write_vxt_model(
        &[],
        &[VxtEntry {
            name: "native.A_log".into(),
            tensor: PackedTensor::RawF32 {
                rows: 1,
                cols: bits.len(),
                data,
            },
        }],
    )
    .unwrap();
    let (_, e) = crate::format::read_vxt_model(&archive).unwrap();
    assert_eq!(
        e[0].tensor
            .decode_f32()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        bits
    );
}

#[test]
fn unsupported_config_missing_shards_and_dtype_reject() {
    assert!(config(br#"{"model_type":"llama"}"#).is_err());
    let b = crate::safetensors::write_safetensors_u16(&[(
        "model.language_model.norm.weight",
        DType::F16,
        &[2048],
        &vec![0; 2048],
    )]);
    assert!(validate_source(&SafeTensorsView::parse(&b).unwrap())
        .unwrap_err()
        .to_string()
        .contains("unsupported source dtype"));
    let b = crate::safetensors::write_safetensors_u16(&[(
        "model.language_model.norm.weight",
        DType::BF16,
        &[2048],
        &vec![0; 2048],
    )]);
    assert!(validate_source(&SafeTensorsView::parse(&b).unwrap())
        .unwrap_err()
        .to_string()
        .contains("missing native tensors"));
}

#[test]
fn malformed_native_manifest_and_duplicate_tensors_reject() {
    assert!(adapt(vec![], &json!({"schema":2})).is_err());
    let mut tensors = serde_json::Map::new();
    for (n, (_, sh, dt)) in expected() {
        tensors.insert(n, descriptor(&sh, dt));
    }
    let manifest = json!({"schema":1,"scope":"qwen3.5-2b-language-only","tensors":tensors});
    let entry = || VxtEntry {
        name: "model.language_model.norm.weight".into(),
        tensor: PackedTensor::Lossless(
            LosslessMatrix::encode(DType::BF16, 1, 2048, &vec![0; 2048]).unwrap(),
        ),
    };
    assert!(adapt(vec![entry(), entry()], &manifest)
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    let mut wrong = manifest.clone();
    wrong["tensors"]["model.language_model.norm.weight"]["shape"] = json!([1, 2048]);
    assert!(adapt(vec![entry()], &wrong)
        .unwrap_err()
        .to_string()
        .contains("shape mismatch"));
    let wrong_dtype = VxtEntry {
        name: "model.language_model.norm.weight".into(),
        tensor: PackedTensor::RawF32 {
            rows: 1,
            cols: 2048,
            data: vec![0.; 2048],
        },
    };
    assert!(adapt(vec![wrong_dtype], &manifest)
        .unwrap_err()
        .to_string()
        .contains("dtype/codec"));
}

#[test]
#[ignore = "requires VXG_QWEN35_TOKENIZER pointing to pinned official tokenizer.json"]
fn official_tokenizer_oracle_cases() {
    let path = std::env::var("VXG_QWEN35_TOKENIZER").expect("set VXG_QWEN35_TOKENIZER");
    let b = std::fs::read(path).unwrap();
    let t = tokenizer(&b, 248320).unwrap();
    let cases = parse(include_bytes!(
        "../tests/fixtures/native_qwen35_tokenizer_cases.json"
    ))
    .unwrap();
    for c in cases.as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let ids: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(t.encode(text), ids, "{text:?}");
    }
    let mut unsupported = parse(&b).unwrap();
    unsupported["normalizer"] = json!(null);
    assert!(tokenizer(&serde_json::to_vec(&unsupported).unwrap(), 248320).is_err());
    unsupported = parse(&b).unwrap();
    unsupported["added_tokens"][0]["lstrip"] = json!(true);
    assert!(tokenizer(&serde_json::to_vec(&unsupported).unwrap(), 248320).is_err());
    unsupported = parse(&b).unwrap();
    unsupported["model"]["byte_fallback"] = json!(true);
    assert!(tokenizer(&serde_json::to_vec(&unsupported).unwrap(), 248320).is_err());
}
