//! Head-to-head compression, numerical fidelity (`SQNR`, cosine similarity), multi-threaded CPU
//! `GEMV`, and Apple Metal GPU `GEMV` benchmark harness + realistic LLM weight generator.

use std::time::Instant;

use crate::codebook::{CodebookMatrix, Q4_0Matrix};
use crate::format::PackedTensor;
use crate::fp16::{bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, ErrorMetrics};
use crate::lossless::{gemv_raw16_cpu, LosslessMatrix};
use crate::metal::MetalEngine;
use crate::wht2::Wht2Matrix;
use crate::{DType, Result};

/// Generates a deterministic, realistic heavy-tailed neural weight matrix (`rows * cols`) with
/// per-channel variance scaling and ~0.6% super-weight outliers, stored in native `BF16` or `F16`.
pub fn generate_realistic_llm_weights(
    dtype: DType,
    rows: usize,
    cols: usize,
    seed: u64,
) -> (Vec<u16>, Vec<f32>) {
    let mut rng = SplitMix64(seed ^ 0x9e37_79b9_7f4a_7c15);

    // Per-input-channel scale (simulates Transformer LayerNorm / SwiGLU channel skew)
    let mut col_scale = vec![1.0f32; cols];
    for s in &mut col_scale {
        let u = rng.next_f32_unit();
        *s = if u > 0.985 {
            2.8 + 2.5 * rng.next_f32_unit()
        } else {
            0.55 + 0.9 * u
        };
    }

    let base_std = (1.0 / (cols as f32).sqrt()).max(0.008);
    let total = rows * cols;
    let mut words = Vec::with_capacity(total);
    let mut f32s = Vec::with_capacity(total);

    for i in 0..total {
        let c = i % cols;
        let g = rng.next_approx_normal();
        let tail_u = rng.next_f32_unit();
        let spike = if tail_u > 0.9965 {
            5.5 + 4.5 * rng.next_f32_unit()
        } else {
            1.0
        };
        let raw = g * base_std * col_scale[c] * spike;
        match dtype {
            DType::BF16 => {
                let w = f32_to_bf16(raw);
                words.push(w);
                f32s.push(bf16_to_f32(w));
            }
            DType::F16 => {
                let w = f32_to_f16(raw);
                words.push(w);
                f32s.push(f16_to_f32(w));
            }
        }
    }
    (words, f32s)
}

/// Generates a deterministic unit-variance activation vector `x` (`cols` elements).
pub fn generate_activation_vec(cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64(seed ^ 0xbf58_476d_1ce4_e5b9);
    (0..cols).map(|_| rng.next_approx_normal() * 0.5).collect()
}

#[derive(Clone, Debug)]
pub struct CodecBenchRow {
    pub label: String,
    pub bytes: usize,
    pub bpw: f64,
    pub ratio_vs_16b: f64,
    pub weight_sqnr_db: f64,
    pub gemv_sqnr_db: f64,
    pub gemv_cos: f64,
    pub cpu_ms: f64,
    pub cpu_raw_eq_gbs: f64,
    pub gpu_ms: Option<f64>,
    pub gpu_actual_gbs: Option<f64>,
    pub gpu_raw_eq_gbs: Option<f64>,
    pub gpu_speedup: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct MatrixBenchReport {
    pub tensor_name: String,
    pub rows: usize,
    pub cols: usize,
    pub dtype: DType,
    pub raw_16b_bytes: usize,
    pub cpu_threads: usize,
    pub gpu_device: Option<String>,
    pub entries: Vec<CodecBenchRow>,
}

/// Runs a full 5-way benchmark (`Raw16`, `VXT-Lossless`, `GGUF Q4_0`, `VXT-CB4+Out`, `VXT-CB3+Out`)
/// on a weight matrix provided as `(dtype, words_u16, weights_f32)`.
pub fn benchmark_matrix(
    tensor_name: &str,
    dtype: DType,
    rows: usize,
    cols: usize,
    words_u16: Option<&[u16]>,
    weights_f32: &[f32],
    cpu_threads: usize,
    cpu_iters: usize,
    gpu_iters: u32,
) -> Result<MatrixBenchReport> {
    let raw_16b_bytes = rows * cols * 2;
    let x = generate_activation_vec(cols, (rows as u64) * 31 + (cols as u64));

    // Synthesize native u16 words from weights_f32 if source came from a pre-quantized GGUF tensor
    let owned_u16: Vec<u16>;
    let raw_words: &[u16] = if let Some(w) = words_u16 {
        w
    } else {
        owned_u16 = match dtype {
            DType::BF16 => weights_f32.iter().map(|&v| f32_to_bf16(v)).collect(),
            DType::F16 => weights_f32.iter().map(|&v| f32_to_f16(v)).collect(),
        };
        &owned_u16
    };

    let mut metal = MetalEngine::new().ok();
    let gpu_device = metal.as_ref().map(|m| m.device_name().to_owned());

    // 1. Reference Raw 16-bit baseline
    let mut y_ref = vec![0.0f32; rows];
    gemv_raw16_cpu(dtype, raw_words, rows, cols, &x, &mut y_ref, cpu_threads);
    let t0 = Instant::now();
    for _ in 0..cpu_iters.max(1) {
        gemv_raw16_cpu(dtype, raw_words, rows, cols, &x, &mut y_ref, cpu_threads);
    }
    let raw_cpu_ms = (t0.elapsed().as_secs_f64() * 1000.0) / (cpu_iters.max(1) as f64);

    let mut y_gpu = vec![0.0f32; rows];
    let raw_gpu_ms = if let Some(eng) = metal.as_mut() {
        let gpu_w = eng.upload_raw16(dtype, rows, cols, raw_words)?;
        let _ = gpu_w.run_gemv(eng, &x, &mut y_gpu, 3)?; // warmup
        Some(gpu_w.run_gemv(eng, &x, &mut y_gpu, gpu_iters.max(1))?)
    } else {
        None
    };

    let mut entries = Vec::with_capacity(5);
    entries.push(build_row(
        format!("Raw {} (Baseline)", dtype.name()),
        raw_16b_bytes,
        raw_16b_bytes,
        f64::INFINITY,
        f64::INFINITY,
        1.0,
        raw_cpu_ms,
        raw_gpu_ms,
        raw_gpu_ms,
    ));

    // 2. Mode 1: VXT-Lossless-BP16 (only when native 16-bit words were supplied or quantized to 16-bit)
    let lossless = PackedTensor::Lossless(LosslessMatrix::encode(dtype, rows, cols, raw_words)?);
    entries.push(eval_packed(
        &lossless,
        None,
        weights_f32,
        &x,
        &y_ref,
        raw_16b_bytes,
        cpu_threads,
        cpu_iters,
        gpu_iters,
        metal.as_mut(),
        raw_gpu_ms,
    )?);

    // 3. Reference GGUF Q4_0 (4.50 bpw)
    let q4 = Q4_0Matrix::encode(rows, cols, weights_f32);
    let q4_dec = q4.decode_f32();
    let q4_packed = PackedTensor::Q4_0(q4);
    entries.push(eval_packed(
        &q4_packed,
        Some(&q4_dec),
        weights_f32,
        &x,
        &y_ref,
        raw_16b_bytes,
        cpu_threads,
        cpu_iters,
        gpu_iters,
        metal.as_mut(),
        raw_gpu_ms,
    )?);

    // 4. Mode 2a: VXT-CB4+Outlier (~4.35 bpw)
    let cb4 = CodebookMatrix::encode(4, rows, cols, weights_f32)?;
    let cb4_dec = cb4.decode_f32();
    let cb4_packed = PackedTensor::Codebook(cb4);
    entries.push(eval_packed(
        &cb4_packed,
        Some(&cb4_dec),
        weights_f32,
        &x,
        &y_ref,
        raw_16b_bytes,
        cpu_threads,
        cpu_iters,
        gpu_iters,
        metal.as_mut(),
        raw_gpu_ms,
    )?);

    // 5. Mode 2b: VXT-CB3+Outlier (~3.40 bpw)
    let cb3 = CodebookMatrix::encode(3, rows, cols, weights_f32)?;
    let cb3_dec = cb3.decode_f32();
    let cb3_packed = PackedTensor::Codebook(cb3);
    entries.push(eval_packed(
        &cb3_packed,
        Some(&cb3_dec),
        weights_f32,
        &x,
        &y_ref,
        raw_16b_bytes,
        cpu_threads,
        cpu_iters,
        gpu_iters,
        metal.as_mut(),
        raw_gpu_ms,
    )?);

    // 6. Mode 3: VXT-WHT2+Outlier (~2.40 bpw Walsh-Hadamard 2-bit)
    let wht2 = Wht2Matrix::encode(rows, cols, weights_f32)?;
    let wht2_dec = wht2.decode_f32();
    let wht2_packed = PackedTensor::Wht2(wht2);
    entries.push(eval_packed(
        &wht2_packed,
        Some(&wht2_dec),
        weights_f32,
        &x,
        &y_ref,
        raw_16b_bytes,
        cpu_threads,
        cpu_iters,
        gpu_iters,
        metal.as_mut(),
        raw_gpu_ms,
    )?);

    Ok(MatrixBenchReport {
        tensor_name: tensor_name.to_owned(),
        rows,
        cols,
        dtype,
        raw_16b_bytes,
        cpu_threads,
        gpu_device,
        entries,
    })
}

#[allow(clippy::too_many_arguments)]
fn eval_packed(
    packed: &PackedTensor,
    decoded_weights: Option<&[f32]>,
    ref_weights: &[f32],
    x: &[f32],
    y_ref: &[f32],
    raw_16b_bytes: usize,
    cpu_threads: usize,
    cpu_iters: usize,
    gpu_iters: u32,
    metal: Option<&mut MetalEngine>,
    raw_gpu_ms: Option<f64>,
) -> Result<CodecBenchRow> {
    let w_sqnr = match decoded_weights {
        Some(dec) => ErrorMetrics::compute(ref_weights, dec).sqnr_db,
        None => f64::INFINITY,
    };

    let mut y_cpu = vec![0.0f32; packed.rows()];
    packed.gemv_cpu(x, &mut y_cpu, cpu_threads);
    let t0 = Instant::now();
    for _ in 0..cpu_iters.max(1) {
        packed.gemv_cpu(x, &mut y_cpu, cpu_threads);
    }
    let cpu_ms = (t0.elapsed().as_secs_f64() * 1000.0) / (cpu_iters.max(1) as f64);

    let y_err = ErrorMetrics::compute(y_ref, &y_cpu);
    let gemv_sqnr = if decoded_weights.is_none() && y_err.sqnr_db > 125.0 {
        f64::INFINITY
    } else {
        y_err.sqnr_db
    };

    let gpu_ms = if let Some(eng) = metal {
        let mut y_g = vec![0.0f32; packed.rows()];
        let gpu_t = eng.upload_packed(packed)?;
        let _ = gpu_t.run_gemv(eng, x, &mut y_g, 3)?; // warmup
        Some(gpu_t.run_gemv(eng, x, &mut y_g, gpu_iters.max(1))?)
    } else {
        None
    };

    Ok(build_row(
        packed.mode_name().to_owned(),
        packed.compressed_bytes(),
        raw_16b_bytes,
        w_sqnr,
        gemv_sqnr,
        y_err.cosine_sim,
        cpu_ms,
        gpu_ms,
        raw_gpu_ms,
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_row(
    label: String,
    bytes: usize,
    raw_16b_bytes: usize,
    weight_sqnr_db: f64,
    gemv_sqnr_db: f64,
    gemv_cos: f64,
    cpu_ms: f64,
    gpu_ms: Option<f64>,
    raw_gpu_ms: Option<f64>,
) -> CodecBenchRow {
    let numel = raw_16b_bytes / 2;
    let bpw = (bytes as f64 * 8.0) / (numel.max(1) as f64);
    let ratio_vs_16b = (raw_16b_bytes as f64) / (bytes.max(1) as f64);
    let cpu_raw_eq_gbs = (raw_16b_bytes as f64) / (cpu_ms.max(1e-9) * 1e6);

    let gpu_actual_gbs = gpu_ms.map(|ms| (bytes as f64) / (ms.max(1e-9) * 1e6));
    let gpu_raw_eq_gbs = gpu_ms.map(|ms| (raw_16b_bytes as f64) / (ms.max(1e-9) * 1e6));
    let gpu_speedup = match (raw_gpu_ms, gpu_ms) {
        (Some(rm), Some(gm)) if gm > 0.0 => Some(rm / gm),
        _ => None,
    };

    CodecBenchRow {
        label,
        bytes,
        bpw,
        ratio_vs_16b,
        weight_sqnr_db,
        gemv_sqnr_db,
        gemv_cos,
        cpu_ms,
        cpu_raw_eq_gbs,
        gpu_ms,
        gpu_actual_gbs,
        gpu_raw_eq_gbs,
        gpu_speedup,
    }
}

pub fn print_report(rep: &MatrixBenchReport) {
    println!(
        "\n=== VXG-TENSOR BENCHMARK: {} [{} x {}, {}, {:.2} MiB raw 16-bit] ===",
        rep.tensor_name,
        rep.rows,
        rep.cols,
        rep.dtype.name(),
        rep.raw_16b_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Backend: CPU ({} threads) | Metal GPU: {}",
        rep.cpu_threads,
        rep.gpu_device.as_deref().unwrap_or("disabled/unavailable")
    );
    println!(
        "{:<22} {:>9} {:>7} {:>7} {:>11} {:>10} {:>9} {:>10} {:>11} {:>11} {:>8}",
        "Codec / Kernel",
        "Size(MiB)",
        "bpw",
        "Ratio",
        "W-SQNR(dB)",
        "GEMV-Cos",
        "CPU(ms)",
        "GPU(us)",
        "GPU ActGB/s",
        "GPU EqGB/s",
        "Speedup"
    );
    println!("{}", "-".repeat(126));
    for r in &rep.entries {
        let mib = r.bytes as f64 / (1024.0 * 1024.0);
        let sqnr_s = if r.weight_sqnr_db.is_infinite() {
            "Bit-Exact".to_owned()
        } else {
            format!("{:>7.2} dB", r.weight_sqnr_db)
        };
        let gpu_us = r
            .gpu_ms
            .map(|m| format!("{:>8.1}", m * 1000.0))
            .unwrap_or_else(|| "     n/a".into());
        let act_gbs = r
            .gpu_actual_gbs
            .map(|g| format!("{:>9.1}", g))
            .unwrap_or_else(|| "      n/a".into());
        let eq_gbs = r
            .gpu_raw_eq_gbs
            .map(|g| format!("{:>9.1}", g))
            .unwrap_or_else(|| "      n/a".into());
        let sp = r
            .gpu_speedup
            .map(|s| format!("{:>6.2}x", s))
            .unwrap_or_else(|| "    n/a".into());
        println!(
            "{:<22} {:>9.2} {:>7.2} {:>6.2}x {:>11} {:>10.6} {:>9.3} {:>10} {:>11} {:>11} {:>8}",
            r.label,
            mib,
            r.bpw,
            r.ratio_vs_16b,
            sqnr_s,
            r.gemv_cos,
            r.cpu_ms,
            gpu_us,
            act_gbs,
            eq_gbs,
            sp
        );
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f32_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
    }

    /// Irwin-Hall 6-uniform sum centered & scaled to zero-mean unit-variance Gaussian.
    #[inline]
    fn next_approx_normal(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += self.next_f32_unit();
        }
        (s - 3.0) * 1.414_213_5
    }
}
