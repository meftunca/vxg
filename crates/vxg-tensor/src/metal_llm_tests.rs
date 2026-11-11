//! Unit tests for `MetalLlmRunner` exact CB4 embedding decode & GPU argmax tie-breaking.

#![cfg(all(test, vxt_has_metal))]

use core::ffi::{c_char, c_void};

use crate::codebook::CodebookMatrix;
use crate::format::PackedTensor;
use crate::fp16::f32_to_f16;
use crate::gguf_meta::ModelConfig;
use crate::llm_ops::sample_logits;
use crate::metal::MetalEngine;
use crate::metal_ffi::cbuf_to_str;
use crate::metal_llm::MetalLlmRunner;

extern "C" {
    fn vxt_metal_llm_greedy_probe(
        r: *mut c_void,
        token: u32,
        embedding: *mut f32,
        logits: *const f32,
        id: *mut u32,
        err: *mut c_char,
        cap: usize,
    ) -> i32;
}

#[test]
fn cb4_gpu_embedding_exact_centroids_outliers_tails_and_greedy_ties() {
    let eng = MetalEngine::new().expect("CB4 primitive test requires actual Metal GPU");
    for cols in [1usize, 129, 257, 385] {
        let rows = 515;
        let padded_cols = cols.div_ceil(128) * 128;
        let groups = padded_cols / 128;
        let centroids = std::array::from_fn(|i| (i as f32 - 7.5) * 0.1234567);
        let mut cb = CodebookMatrix {
            bits: 4,
            rows,
            cols,
            padded_cols,
            centroids,
            scales_f16: (0..rows * groups)
                .map(|i| f32_to_f16(0.125 + (i % 11) as f32 * 0.25))
                .collect(),
            packed_words: (0..rows * groups * 16)
                .map(|i| 0x76543210u32.rotate_left((i % 8 * 4) as u32))
                .collect(),
            outlier_row_ptr: Vec::new(),
            outliers_packed: Vec::new(),
        };
        for _ in 0..rows {
            cb.outlier_row_ptr.push(cb.outliers_packed.len() as u32);
            for value in [0.33325195, -0.125, 0.0009765625] {
                cb.outliers_packed
                    .push(((f32_to_f16(value) as u32) << 16) | ((cols - 1) as u32));
            }
        }
        cb.outlier_row_ptr.push(cb.outliers_packed.len() as u32);
        let cfg = ModelConfig {
            arch: "test".into(),
            model_name: "test".into(),
            n_layer: 0,
            n_embd: cols,
            n_ff: cols,
            n_vocab: rows,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 1,
            rms_norm_eps: 1e-6,
            rope_freq_base: 10000.,
            rope_dim: 0,
            rope_neox: false,
            ffn_gelu: false,
            embed_scale: 1.0,
            full_attn_interval: 0,
            ssm_conv_kernel: 0,
            ssm_state_size: 0,
            ssm_group_count: 0,
            ssm_dt_rank: 0,
            ssm_inner_size: 0,
            eos_token_id: 0,
            bos_token_id: None,
        };
        let embedding = eng.upload_packed(&PackedTensor::Codebook(cb.clone())).unwrap();
        let head = eng
            .upload_packed(&PackedTensor::RawF32 {
                rows,
                cols,
                data: vec![0.; rows * cols],
            })
            .unwrap();
        let mut runner = MetalLlmRunner::new(&eng, &cfg, &[], &vec![1.; cols], &head).unwrap();
        runner.enable_cb4_greedy(&embedding, &centroids).unwrap();
        runner.enable_cb4_greedy(&embedding, &centroids).unwrap();
        for token in [0u32, 1, 514] {
            let mut expected = vec![0.; cols];
            cb.decode_row(token as usize, &mut expected);
            for case in 0..6 {
                let mut logits = vec![-3.; rows];
                match case {
                    0 => {
                        logits[257] = 9.;
                        logits[513] = 9.;
                    }
                    1 => {
                        logits.fill(-0.);
                        logits[514] = 0.;
                    }
                    2 => logits.fill(f32::NEG_INFINITY),
                    3 => {
                        logits[7] = f32::NAN;
                        logits[514] = 4.;
                    }
                    4 => {
                        logits[300] = f32::INFINITY;
                        logits[2] = f32::INFINITY;
                    }
                    _ => logits.fill(f32::NAN),
                }
                let mut actual = vec![0.; cols];
                let mut id = 999;
                let mut err = [0u8; 512];
                let rc = unsafe {
                    vxt_metal_llm_greedy_probe(
                        runner.raw_ptr(),
                        token,
                        actual.as_mut_ptr(),
                        logits.as_ptr(),
                        &mut id,
                        err.as_mut_ptr().cast(),
                        err.len(),
                    )
                };
                assert_eq!(rc, 0, "{}", cbuf_to_str(&err));
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "cols={cols} token={token}"
                );
                assert_eq!(id, sample_logits(&logits, 0., &mut 1), "case={case}");
            }
        }
        let mut st = runner.new_state().unwrap();
        assert!(runner.greedy_step(&mut st, rows as u32, 0).is_err());
        assert!(runner.greedy_step(&mut st, 0, usize::MAX).is_err());
    }
}
