//! Explicit Metal-only experiment gate: identical SIMD row math at two launch groupings.
use vxg_tensor::{bench::{generate_activation_vec, generate_realistic_llm_weights},
    format::PackedTensor, lossless::LosslessMatrix, metal::MetalEngine, DType};

#[test]
fn lossless_rows4_matches_rows2_bits_for_bf16_fp16_and_row_column_tails() {
    let mut eng = MetalEngine::new().expect("this focused gate requires real Metal");
    // Both row groupings' boundaries, partial SIMD-group sets, and superblock tails.
    for dtype in [DType::BF16, DType::F16] {
        for (rows, cols) in [(1,1), (3,255), (4,256), (5,257), (7,533), (9,1025), (137,533)] {
            let (words, _) = generate_realistic_llm_weights(dtype, rows, cols, 42);
            let packed = PackedTensor::Lossless(LosslessMatrix::encode(dtype, rows, cols, &words).unwrap());
            let gpu = eng.upload_packed(&packed).unwrap();
            let x = generate_activation_vec(cols, 91);
            let mut baseline = vec![f32::NAN;rows];
            let mut candidate = vec![f32::NAN;rows];
            std::env::remove_var("VXG_LOSSLESS_ROWS4");
            gpu.run_gemv(&mut eng, &x, &mut baseline, 1).unwrap();
            std::env::set_var("VXG_LOSSLESS_ROWS4", "1");
            gpu.run_gemv(&mut eng, &x, &mut candidate, 1).unwrap();
            std::env::remove_var("VXG_LOSSLESS_ROWS4");
            assert!(baseline.iter().all(|v| v.is_finite()), "baseline nonfinite {dtype:?} {rows}x{cols}");
            assert_eq!(baseline.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                candidate.iter().map(|v|v.to_bits()).collect::<Vec<_>>(), "{dtype:?} {rows}x{cols}");
        }
    }
}
