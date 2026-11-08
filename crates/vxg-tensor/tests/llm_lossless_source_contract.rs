use vxg_tensor::format::{read_vxt_archive, write_vxt_archive, PackedTensor, VxtEntry};
use vxg_tensor::fp16::f16_to_f32;
use vxg_tensor::gguf::GgufView;
use vxg_tensor::llm_pack::{encode_gguf_tensor, LlmPackMode};

fn fixture(kind: u32, rows: usize, cols: usize, payload: &[u8]) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    let name = b"blk.0.attn_q.weight";
    b.extend((name.len() as u64).to_le_bytes());
    b.extend(name);
    b.extend(2u32.to_le_bytes());
    b.extend((cols as u64).to_le_bytes());
    b.extend((rows as u64).to_le_bytes());
    b.extend(kind.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.resize(b.len().div_ceil(32) * 32, 0);
    b.extend(payload);
    b
}

fn check(kind: u32, rows: usize, cols: usize, payload: &[u8], expected: &[f32], raw: bool) {
    let bytes = fixture(kind, rows, cols, payload);
    let gg = GgufView::parse(&bytes).unwrap();
    let tensor = encode_gguf_tensor(&gg, &gg.tensors[0], LlmPackMode::Lossless).unwrap();
    assert_eq!(matches!(tensor, PackedTensor::RawF32 { .. }), raw);
    let archive = write_vxt_archive(&[VxtEntry { name: "weight".into(), tensor }]).unwrap();
    let entries = read_vxt_archive(&archive).unwrap();
    let actual = entries[0].tensor.decode_f32();
    assert_eq!(actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn lossless_f32_preserves_all_bits_in_large_and_small_matrices() {
    for (rows, cols) in [(64, 64), (16, 64), (1, 8)] {
        let bits = [0x3f801234, 0xbf819abc, 0x00000001, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000, 0x00800001];
        let values: Vec<f32> = (0..rows * cols).map(|i| f32::from_bits(bits[i % bits.len()])).collect();
        let payload: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        check(0, rows, cols, &payload, &values, true);
    }
}

#[test]
fn lossless_quantized_sources_preserve_independently_dequantized_values() {
    for kind in [2, 8] { // GGML Q4_0 and Q8_0
        let mut payload = Vec::new();
        let mut expected = Vec::new();
        for _ in 0..128 {
            payload.extend(0x3555u16.to_le_bytes());
            let scale = f16_to_f32(0x3555);
            if kind == 2 {
                payload.extend(0..16u8);
                expected.extend((0..16).map(|q| (q as f32 - 8.0) * scale));
                expected.extend(std::iter::repeat_n(-8.0 * scale, 16));
            } else {
                payload.extend((0..32).map(|q| (q as i8 - 16) as u8));
                expected.extend((0..32).map(|q| (q as f32 - 16.0) * scale));
            }
        }
        check(kind, 64, 64, &payload, &expected, true);
    }
}

#[test]
fn lossless_native_16_bit_sources_still_use_compressed_storage() {
    for kind in [1, 30] {
        let words: Vec<u16> = (0..4096).map(|i| 0x3000 + (i % 1024) as u16).collect();
        let payload: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let expected: Vec<f32> = words.iter().map(|&w| if kind == 1 { f16_to_f32(w) } else { f32::from_bits((w as u32) << 16) }).collect();
        check(kind, 64, 64, &payload, &expected, false);
    }
}

#[test]
fn lossless_q4_k_preserves_scale_and_minimum_precision() {
    let mut payload = Vec::new();
    let mut expected = Vec::new();
    let d = f16_to_f32(0x3555);
    let minimum = f16_to_f32(0x2111);
    for _ in 0..16 {
        payload.extend(0x3555u16.to_le_bytes());
        payload.extend(0x2111u16.to_le_bytes());
        // All eight scale and minimum multipliers are one.
        payload.extend([1u8; 8]);
        payload.extend([0x11u8; 4]);
        payload.extend([0x21u8; 128]);
        for _ in 0..4 {
            expected.extend(std::iter::repeat_n(d - minimum, 32));
            expected.extend(std::iter::repeat_n(d * 2.0 - minimum, 32));
        }
    }
    check(12, 64, 64, &payload, &expected, true);
}
