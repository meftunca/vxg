use vxg_tensor::bench::{generate_activation_vec, generate_realistic_llm_weights};
use vxg_tensor::codebook::{CodebookMatrix, Q4_0Matrix};
use vxg_tensor::format::{
    read_vxt_archive, read_vxt_model, write_vxt_archive, write_vxt_model, PackedTensor, VxtEntry,
};
use vxg_tensor::fp16::{bf16_to_f32, f16_to_f32, ErrorMetrics};
use vxg_tensor::lossless::{gemv_raw16_cpu, LosslessMatrix};
use vxg_tensor::metal::MetalEngine;
use vxg_tensor::safetensors::{write_safetensors_u16, SafeTensorsView};
use vxg_tensor::wht2::Wht2Matrix;
use vxg_tensor::wht_turbo::TurboWhtMatrix;
use vxg_tensor::{DType, Error};

#[test]
fn lossless_bf16_and_fp16_are_100_percent_bit_exact_including_odd_shapes() {
    for &dtype in &[DType::BF16, DType::F16] {
        // Test unaligned dimensions (137 x 533 -> not a multiple of 32 or 256!) + special bit patterns
        let rows = 137usize;
        let cols = 533usize;
        let (mut words, _) = generate_realistic_llm_weights(dtype, rows, cols, 42);
        // Inject tricky IEEE-754 edge cases: -0.0, +0.0, subnormals, inf, NaNs, rare exponents
        words[0] = 0x0000;
        words[1] = 0x8000;
        words[2] = 0x0001; // smallest positive subnormal
        words[3] = 0x807f; // negative subnormal
        words[17] = match dtype {
            DType::BF16 => 0x7fc1, // quiet NaN
            DType::F16 => 0x7e11,  // quiet NaN
        };
        words[cols - 1] = match dtype {
            DType::BF16 => 0x7f80, // +Inf
            DType::F16 => 0x7c00,  // +Inf
        };

        let encoded = LosslessMatrix::encode(dtype, rows, cols, &words).unwrap();
        let decoded = encoded.decode_u16();
        assert_eq!(
            words, decoded,
            "Lossless {:?} roundtrip was NOT 100% bit-exact!",
            dtype
        );

        // Verify meaningful lossless compression on realistic weights (512 x 1024)
        let (w_real, _) = generate_realistic_llm_weights(dtype, 512, 1024, 99);
        let enc_real = LosslessMatrix::encode(dtype, 512, 1024, &w_real).unwrap();
        assert_eq!(w_real, enc_real.decode_u16());
        match dtype {
            DType::BF16 => assert!(
                enc_real.bits_per_weight() < 12.2,
                "expected BF16 lossless < 12.2 bpw, got {:.2}",
                enc_real.bits_per_weight()
            ),
            DType::F16 => assert!(
                enc_real.bits_per_weight() < 14.8,
                "expected FP16 lossless < 14.8 bpw, got {:.2}",
                enc_real.bits_per_weight()
            ),
        }
    }
}

#[test]
fn vxt_archive_and_safetensors_roundtrip_plus_tamper_detection() {
    let rows = 96usize;
    let cols = 384usize;
    let (bf_words, bf_f32) = generate_realistic_llm_weights(DType::BF16, rows, cols, 1234);
    let (fp_words, _) = generate_realistic_llm_weights(DType::F16, rows, cols, 5678);

    // 1. SafeTensors roundtrip
    let st_bytes = write_safetensors_u16(&[
        ("blk.0.attn_q.weight", DType::BF16, &[rows, cols], &bf_words),
        ("blk.0.attn_k.weight", DType::F16, &[rows, cols], &fp_words),
    ]);
    let st_view = SafeTensorsView::parse(&st_bytes).unwrap();
    assert_eq!(st_view.tensors.len(), 2);
    let (dt0, parsed_bf) = st_view
        .extract_fp16_words(&st_view.tensors[0])
        .unwrap()
        .unwrap();
    assert_eq!(dt0, DType::BF16);
    assert_eq!(parsed_bf, bf_words);

    // 2. Multi-mode .vxt archive roundtrip
    let entries = vec![
        VxtEntry {
            name: "q_proj".into(),
            tensor: PackedTensor::Lossless(
                LosslessMatrix::encode(DType::BF16, rows, cols, &bf_words).unwrap(),
            ),
        },
        VxtEntry {
            name: "k_proj".into(),
            tensor: PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, &bf_f32).unwrap()),
        },
        VxtEntry {
            name: "v_proj".into(),
            tensor: PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &bf_f32).unwrap()),
        },
        VxtEntry {
            name: "o_proj".into(),
            tensor: PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &bf_f32)),
        },
    ];
    let mut vxt_bytes = write_vxt_archive(&entries).unwrap();
    let unpacked = read_vxt_archive(&vxt_bytes).unwrap();
    assert_eq!(unpacked.len(), 4);
    match &unpacked[0].tensor {
        PackedTensor::Lossless(m) => assert_eq!(m.decode_u16(), bf_words),
        _ => panic!("expected Lossless"),
    }

    // 3. Flip 1 bit inside payload -> must trigger Error::Checksum!
    let mid = vxt_bytes.len() / 2;
    vxt_bytes[mid] ^= 0x04;
    assert!(matches!(
        read_vxt_archive(&vxt_bytes),
        Err(Error::Checksum(_))
    ));
}

#[test]
fn cb4_outlier_beats_gguf_q4_0_on_both_bpw_and_sqnr() {
    let rows = 256usize;
    let cols = 1024usize;
    let (_, weights) = generate_realistic_llm_weights(DType::BF16, rows, cols, 2026);
    let x = generate_activation_vec(cols, 77);

    let q4 = Q4_0Matrix::encode(rows, cols, &weights);
    let cb4 = CodebookMatrix::encode(4, rows, cols, &weights).unwrap();
    let cb3 = CodebookMatrix::encode(3, rows, cols, &weights).unwrap();
    let wht2 = Wht2Matrix::encode(rows, cols, &weights).unwrap();
    assert!(
        wht2.bits_per_weight() < 2.55,
        "expected WHT2 < 2.55 bpw on 256-aligned cols, got {:.3}",
        wht2.bits_per_weight()
    );

    let q4_err = ErrorMetrics::compute(&weights, &q4.decode_f32());
    let cb4_err = ErrorMetrics::compute(&weights, &cb4.decode_f32());
    let cb3_err = ErrorMetrics::compute(&weights, &cb3.decode_f32());

    assert!(
        cb4.bits_per_weight() < q4.bits_per_weight(),
        "expected CB4+Outlier bpw ({:.3}) < Q4_0 bpw ({:.3})",
        cb4.bits_per_weight(),
        q4.bits_per_weight()
    );
    assert!(
        cb4_err.sqnr_db > q4_err.sqnr_db + 3.0,
        "expected CB4+Outlier SQNR ({:.2} dB) > Q4_0 SQNR ({:.2} dB) + 3 dB",
        cb4_err.sqnr_db,
        q4_err.sqnr_db
    );
    assert!(
        cb3.bits_per_weight() < 3.65 && cb3_err.sqnr_db > 15.0,
        "unexpected CB3 metrics: bpw={:.2}, sqnr={:.2}",
        cb3.bits_per_weight(),
        cb3_err.sqnr_db
    );

    // Verify fused CPU GEMV matches dequantized matrix-vector product closely
    let mut y_ref = vec![0.0f32; rows];
    for r in 0..rows {
        let mut s = 0.0f32;
        for c in 0..cols {
            s = weights[r * cols + c].mul_add(x[c], s);
        }
        y_ref[r] = s;
    }
    let mut y_q4 = vec![0.0f32; rows];
    let mut y_cb4_1t = vec![0.0f32; rows];
    let mut y_cb4_4t = vec![0.0f32; rows];
    q4.gemv_cpu(&x, &mut y_q4, 4);
    cb4.gemv_cpu(&x, &mut y_cb4_1t, 1);
    cb4.gemv_cpu(&x, &mut y_cb4_4t, 4);
    assert_eq!(y_cb4_1t, y_cb4_4t);
    let q4_gemv_m = ErrorMetrics::compute(&y_ref, &y_q4);
    let gemv_m = ErrorMetrics::compute(&y_ref, &y_cb4_4t);
    assert!(
        gemv_m.cosine_sim > 0.995 && gemv_m.cosine_sim > q4_gemv_m.cosine_sim,
        "expected CB4 GEMV cosine ({:.6}) > 0.995 and > Q4_0 ({:.6})",
        gemv_m.cosine_sim,
        q4_gemv_m.cosine_sim
    );
}

#[test]
fn cpu_and_metal_gpu_gemv_kernels_agree_across_all_6_pipelines() {
    let rows = 197usize; // intentionally non-multiple of 4 threadgroup rows!
    let cols = 544usize; // non-multiple of 256!
    let x = generate_activation_vec(cols, 999);

    let (bf_words, bf_f32) = generate_realistic_llm_weights(DType::BF16, rows, cols, 11);
    let (fp_words, _) = generate_realistic_llm_weights(DType::F16, rows, cols, 22);

    // Verify Lossless CPU GEMV matches Raw16 CPU GEMV bit-nearly-identically (only FMA order differs)
    for (dt, words) in [(DType::BF16, &bf_words), (DType::F16, &fp_words)] {
        let mut y_raw = vec![0.0f32; rows];
        let mut y_loss = vec![0.0f32; rows];
        gemv_raw16_cpu(dt, words, rows, cols, &x, &mut y_raw, 4);
        let lm = LosslessMatrix::encode(dt, rows, cols, words).unwrap();
        lm.gemv_cpu(&x, &mut y_loss, 4);
        let m = ErrorMetrics::compute(&y_raw, &y_loss);
        assert!(
            m.max_abs_err < 1e-5,
            "CPU Lossless {:?} vs Raw16 max_abs_err = {}",
            dt,
            m.max_abs_err
        );

        // Double check explicit scalar dot product
        for r in [0, rows / 2, rows - 1] {
            let expected: f32 = (0..cols)
                .map(|c| {
                    let w = match dt {
                        DType::BF16 => bf16_to_f32(words[r * cols + c]),
                        DType::F16 => f16_to_f32(words[r * cols + c]),
                    };
                    w * x[c]
                })
                .sum();
            assert!((y_loss[r] - expected).abs() < 1e-4);
        }
    }

    // If Apple Metal GPU is available (macOS), verify all 6 GPU compute pipelines match CPU!
    if let Ok(mut eng) = MetalEngine::new() {
        for (dt, words) in [(DType::BF16, &bf_words), (DType::F16, &fp_words)] {
            let mut y_cpu = vec![0.0f32; rows];
            let mut y_gpu = vec![0.0f32; rows];
            gemv_raw16_cpu(dt, words, rows, cols, &x, &mut y_cpu, 4);

            let gpu_raw = eng.upload_raw16(dt, rows, cols, words).unwrap();
            gpu_raw.run_gemv(&mut eng, &x, &mut y_gpu, 1).unwrap();
            let err_raw = ErrorMetrics::compute(&y_cpu, &y_gpu);
            assert!(
                err_raw.max_abs_err < 2e-4,
                "Metal Raw16 {:?} mismatch: max_err={}",
                dt,
                err_raw.max_abs_err
            );

            let lm = PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, words).unwrap());
            let gpu_lm = eng.upload_packed(&lm).unwrap();
            gpu_lm.run_gemv(&mut eng, &x, &mut y_gpu, 1).unwrap();
            let err_lm = ErrorMetrics::compute(&y_cpu, &y_gpu);
            assert!(
                err_lm.max_abs_err < 2e-4,
                "Metal Lossless {:?} mismatch: max_err={}",
                dt,
                err_lm.max_abs_err
            );
        }

        let cases = [
            PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, &bf_f32).unwrap()),
            PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &bf_f32).unwrap()),
            PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &bf_f32)),
            PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &bf_f32).unwrap()),
            PackedTensor::TurboWht(TurboWhtMatrix::encode(3, rows, cols, &bf_f32).unwrap()),
            PackedTensor::TurboWht(TurboWhtMatrix::encode(4, rows, cols, &bf_f32).unwrap()),
            PackedTensor::TurboWht(TurboWhtMatrix::encode(5, rows, cols, &bf_f32).unwrap()),
            PackedTensor::TurboWht(TurboWhtMatrix::encode(2, rows, cols, &bf_f32).unwrap()),
        ];
        let mut gpu_cases = Vec::new();
        let mut cpu_refs = Vec::new();
        for packed in &cases {
            let mut y_cpu = vec![0.0f32; rows];
            let mut y_gpu = vec![0.0f32; rows];
            packed.gemv_cpu(&x, &mut y_cpu, 4);
            let gpu_p = eng.upload_packed(packed).unwrap();
            gpu_p.run_gemv(&mut eng, &x, &mut y_gpu, 1).unwrap();
            let err = ErrorMetrics::compute(&y_cpu, &y_gpu);
            assert!(
                err.max_abs_err < 2e-4,
                "Metal {} mismatch vs CPU: max_err={}",
                packed.mode_name(),
                err.max_abs_err
            );
            gpu_cases.push(gpu_p);
            cpu_refs.push(y_cpu);
        }

        // Verify batched multi-matrix single-command-buffer execution (`run_multi_same_x`)
        let (mut m0, mut m1, mut m2, mut m3) = (
            vec![0.0f32; rows],
            vec![0.0f32; rows],
            vec![0.0f32; rows],
            vec![0.0f32; rows],
        );
        eng.run_multi_same_x(
            &x,
            &mut [
                (&gpu_cases[0], &mut m0),
                (&gpu_cases[1], &mut m1),
                (&gpu_cases[2], &mut m2),
                (&gpu_cases[3], &mut m3),
            ],
        )
        .unwrap();
        for (got, want) in [&m0, &m1, &m2, &m3].into_iter().zip(&cpu_refs) {
            assert!(ErrorMetrics::compute(want, got).max_abs_err < 2e-4);
        }

        // Also test NR=8 (`rows >= 4096`, odd tail `4102`) + `concat_vertical` for Turbo2 & Turbo3!
        let r_a = 2048usize;
        let r_b = 2054usize;
        let (_, wf_a) = generate_realistic_llm_weights(DType::BF16, r_a, cols, 301);
        let (_, wf_b) = generate_realistic_llm_weights(DType::BF16, r_b, cols, 302);
        let mut wf_ab = wf_a.clone();
        wf_ab.extend_from_slice(&wf_b);
        for bits in [2u8, 3u8, 4u8, 5u8] {
            let ta = TurboWhtMatrix::encode(bits, r_a, cols, &wf_a).unwrap();
            let tb = TurboWhtMatrix::encode(bits, r_b, cols, &wf_b).unwrap();
            let tab = TurboWhtMatrix::concat_vertical(&[&ta, &tb]).unwrap();
            let tab_direct = TurboWhtMatrix::encode(bits, r_a + r_b, cols, &wf_ab).unwrap();
            assert_eq!(tab.scales_f16, tab_direct.scales_f16);
            if !matches!(bits, 4 | 5) {
                assert_eq!(tab.packed_words, tab_direct.packed_words);
            }
            assert_eq!(
                PackedTensor::TurboWht(tab.clone()).decode_f32(),
                PackedTensor::TurboWht(tab_direct).decode_f32()
            );

            let packed_ab = PackedTensor::TurboWht(tab);
            let mut y_cpu = vec![0.0f32; r_a + r_b];
            let mut y_gpu = vec![0.0f32; r_a + r_b];
            packed_ab.gemv_cpu(&x, &mut y_cpu, 4);
            let gpu_ab = eng.upload_packed(&packed_ab).unwrap();
            gpu_ab.run_gemv(&mut eng, &x, &mut y_gpu, 1).unwrap();
            let err = ErrorMetrics::compute(&y_cpu, &y_gpu);
            assert!(
                err.max_abs_err < 2e-4,
                "Metal Turbo{bits} NR=8 mismatch: max_err={}",
                err.max_abs_err
            );
        }
    }

    // Also verify Wht2 + TurboWht + RawF32 + Metadata roundtrip in .vxt model container
    let wht = Wht2Matrix::encode(rows, cols, &bf_f32).unwrap();
    let t2 = TurboWhtMatrix::encode(2, rows, cols, &bf_f32).unwrap();
    let t25 = TurboWhtMatrix::encode(5, rows, cols, &bf_f32).unwrap();
    let t3 = TurboWhtMatrix::encode(3, rows, cols, &bf_f32).unwrap();
    let tec = TurboWhtMatrix::encode_with_lambda(4, rows, cols, &bf_f32, |r| {
        if r < 64 {
            0.055
        } else {
            0.18
        }
    })
    .unwrap();
    let long_meta = b"qwen35-repeating-tokenizer-vocabulary-entry-".repeat(64);
    let meta = vec![
        ("arch".to_owned(), b"qwen35".to_vec()),
        ("tokenizer.tokens".to_owned(), long_meta),
    ];
    let model_bytes = write_vxt_model(
        &meta,
        &[
            VxtEntry {
                name: "wht_proj".into(),
                tensor: PackedTensor::Wht2(wht.clone()),
            },
            VxtEntry {
                name: "t2_proj".into(),
                tensor: PackedTensor::TurboWht(t2.clone()),
            },
            VxtEntry {
                name: "t25_proj".into(),
                tensor: PackedTensor::TurboWht(t25.clone()),
            },
            VxtEntry {
                name: "t3_proj".into(),
                tensor: PackedTensor::TurboWht(t3.clone()),
            },
            VxtEntry {
                name: "tec_proj".into(),
                tensor: PackedTensor::TurboWht(tec.clone()),
            },
            VxtEntry {
                name: "norm".into(),
                tensor: PackedTensor::RawF32 {
                    rows: 1,
                    cols: 16,
                    data: vec![1.25f32; 16],
                },
            },
        ],
    )
    .unwrap();
    let (parsed_meta, parsed_entries) = read_vxt_model(&model_bytes).unwrap();
    assert_eq!(parsed_meta, meta);
    assert_eq!(parsed_entries[0].tensor.decode_f32(), wht.decode_f32());
    assert_eq!(parsed_entries[1].tensor.decode_f32(), PackedTensor::TurboWht(t2).decode_f32());
    assert_eq!(parsed_entries[2].tensor.decode_f32(), PackedTensor::TurboWht(t25).decode_f32());
    assert_eq!(parsed_entries[3].tensor.decode_f32(), PackedTensor::TurboWht(t3).decode_f32());
    match &parsed_entries[4].tensor {
        PackedTensor::TurboWht(m) => {
            assert_eq!(m.scales_f16, tec.scales_f16);
            assert_eq!(m.packed_words, tec.packed_words);
            assert!(m.bits_per_weight() < 2.85, "expected EC-Turbo3 < 2.85 bpw, got {}", m.bits_per_weight());
        }
        _ => panic!("expected EC-Turbo3"),
    }
    assert_eq!(parsed_entries[5].tensor.decode_f32(), vec![1.25f32; 16]);
}
