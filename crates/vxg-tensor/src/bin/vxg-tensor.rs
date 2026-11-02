use std::env;
use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;

use vxg_tensor::bench::{benchmark_matrix, generate_realistic_llm_weights, print_report};
use vxg_tensor::codebook::{CodebookMatrix, Q4_0Matrix};
use vxg_tensor::format::{read_vxt_archive, write_vxt_archive, PackedTensor, VxtEntry};
use vxg_tensor::fp16::f32_to_bf16;
use vxg_tensor::gguf::GgufView;
use vxg_tensor::lossless::LosslessMatrix;
use vxg_tensor::mmap::TensorBytes;
use vxg_tensor::safetensors::SafeTensorsView;
use vxg_tensor::{DType, Error, Result};

fn usage() -> ! {
    eprintln!(
        "vxg-tensor — Compressed-Domain AI Model Weight Codec (.vxt) & Fused GPU/SIMD GEMV Engine

Usage:
  vxg-tensor info   <model.safetensors | model.gguf | model.vxt>
  vxg-tensor pack   <in.safetensors | in.gguf> <out.vxt> [--mode lossless|cb4|cb3|q4_0] [--max-tensors N]
  vxg-tensor verify <archive.vxt>
  vxg-tensor bench  [--gguf <path> | --safetensors <path> | --rows R --cols C] [--dtype bf16|fp16] [--threads T] [--iters N] [--max-tensors K]"
    );
    process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        usage();
    }
    let res = match args[0].as_str() {
        "info" => cmd_info(&args[1..]),
        "pack" => cmd_pack(&args[1..]),
        "verify" => cmd_verify(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn cmd_info(args: &[String]) -> Result<()> {
    let Some(path) = args.first() else { usage() };
    let mapped = TensorBytes::open(path)?;
    let bytes = mapped.as_slice();
    println!(
        "File: {} ({:.2} MiB, mmap={})",
        path,
        bytes.len() as f64 / (1024.0 * 1024.0),
        mapped.is_mapped()
    );

    if bytes.starts_with(b"GGUF") {
        let gg = GgufView::parse(bytes)?;
        println!(
            "Format: GGUF v{} (alignment={}, tensors={})",
            gg.version,
            gg.alignment,
            gg.tensors.len()
        );
        for (i, t) in gg.tensors.iter().take(40).enumerate() {
            println!(
                "  [{:>3}] {:<48} shape={:?} type={:?}",
                i, t.name, t.ne, t.ggml_type
            );
        }
        if gg.tensors.len() > 40 {
            println!("  ... ({} more tensors)", gg.tensors.len() - 40);
        }
    } else if bytes.starts_with(b"VXT1") {
        let entries = read_vxt_archive(bytes)?;
        println!("Format: VXT1 archive (tensors={})", entries.len());
        for (i, e) in entries.iter().enumerate() {
            println!(
                "  [{:>3}] {:<44} [{} x {}] {:<18} {:.2} bpw ({:.2} KiB)",
                i,
                e.name,
                e.tensor.rows(),
                e.tensor.cols(),
                e.tensor.mode_name(),
                e.tensor.bits_per_weight(),
                e.tensor.compressed_bytes() as f64 / 1024.0
            );
        }
    } else {
        let st = SafeTensorsView::parse(bytes)?;
        println!("Format: SafeTensors (tensors={})", st.tensors.len());
        for (i, t) in st.tensors.iter().take(40).enumerate() {
            println!(
                "  [{:>3}] {:<48} shape={:?} dtype={:?} ({:.2} KiB)",
                i,
                t.name,
                t.shape,
                t.dtype,
                (t.data_end - t.data_start) as f64 / 1024.0
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum PackMode {
    Lossless,
    Cb4,
    Cb3,
    Q4_0,
}

fn cmd_pack(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        usage();
    }
    let in_path = &args[0];
    let out_path = &args[1];
    let mut mode = PackMode::Cb4;
    let mut max_tensors = usize::MAX;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                mode = match args.get(i).map(String::as_str) {
                    Some("lossless") => PackMode::Lossless,
                    Some("cb4") => PackMode::Cb4,
                    Some("cb3") => PackMode::Cb3,
                    Some("q4_0") => PackMode::Q4_0,
                    _ => {
                        return Err(Error::Format(
                            "expected --mode lossless|cb4|cb3|q4_0".into(),
                        ))
                    }
                };
            }
            "--max-tensors" => {
                i += 1;
                max_tensors = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| Error::Format("invalid --max-tensors".into()))?;
            }
            _ => usage(),
        }
        i += 1;
    }

    let t0 = Instant::now();
    let mapped = TensorBytes::open(in_path)?;
    let bytes = mapped.as_slice();
    let mut entries = Vec::new();
    let mut raw_16b_sum = 0usize;

    if bytes.starts_with(b"GGUF") {
        let gg = GgufView::parse(bytes)?;
        for meta in gg
            .tensors
            .iter()
            .filter(|m| m.ggml_type.is_supported_matrix())
        {
            let Some((rows, cols)) = meta.matrix_dims() else {
                continue;
            };
            if rows < 16 || cols < 32 {
                continue;
            }
            let tensor = match mode {
                PackMode::Lossless => {
                    if let Some((dt, words)) = gg.extract_fp16_words(meta)? {
                        PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                    } else {
                        let f32s = gg.extract_f32(meta)?;
                        let words: Vec<u16> = f32s.iter().map(|&v| f32_to_bf16(v)).collect();
                        PackedTensor::Lossless(LosslessMatrix::encode(
                            DType::BF16,
                            rows,
                            cols,
                            &words,
                        )?)
                    }
                }
                PackMode::Cb4 => {
                    let f32s = gg.extract_f32(meta)?;
                    PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, &f32s)?)
                }
                PackMode::Cb3 => {
                    let f32s = gg.extract_f32(meta)?;
                    PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &f32s)?)
                }
                PackMode::Q4_0 => {
                    let f32s = gg.extract_f32(meta)?;
                    PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &f32s))
                }
            };
            raw_16b_sum += rows * cols * 2;
            entries.push(VxtEntry {
                name: meta.name.clone(),
                tensor,
            });
            if entries.len() >= max_tensors {
                break;
            }
        }
    } else {
        let st = SafeTensorsView::parse(bytes)?;
        for meta in &st.tensors {
            let Some((rows, cols)) = meta.matrix_dims() else {
                continue;
            };
            if rows < 16 || cols < 32 {
                continue;
            }
            let tensor = match mode {
                PackMode::Lossless => {
                    if let Some((dt, words)) = st.extract_fp16_words(meta)? {
                        PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                    } else {
                        let f32s = st.extract_f32(meta)?;
                        let words: Vec<u16> = f32s.iter().map(|&v| f32_to_bf16(v)).collect();
                        PackedTensor::Lossless(LosslessMatrix::encode(
                            DType::BF16,
                            rows,
                            cols,
                            &words,
                        )?)
                    }
                }
                PackMode::Cb4 => {
                    let f32s = st.extract_f32(meta)?;
                    PackedTensor::Codebook(CodebookMatrix::encode(4, rows, cols, &f32s)?)
                }
                PackMode::Cb3 => {
                    let f32s = st.extract_f32(meta)?;
                    PackedTensor::Codebook(CodebookMatrix::encode(3, rows, cols, &f32s)?)
                }
                PackMode::Q4_0 => {
                    let f32s = st.extract_f32(meta)?;
                    PackedTensor::Q4_0(Q4_0Matrix::encode(rows, cols, &f32s))
                }
            };
            raw_16b_sum += rows * cols * 2;
            entries.push(VxtEntry {
                name: meta.name.clone(),
                tensor,
            });
            if entries.len() >= max_tensors {
                break;
            }
        }
    }

    let archive = write_vxt_archive(&entries)?;
    fs::write(Path::new(out_path), &archive).map_err(|e| Error::Io(e.to_string()))?;
    let ratio = (raw_16b_sum as f64) / (archive.len().max(1) as f64);
    println!(
        "Packed {} matrices -> {} ({:.2} MiB -> {:.2} MiB, {:.2}x vs 16-bit) in {:.2}s",
        entries.len(),
        out_path,
        raw_16b_sum as f64 / (1024.0 * 1024.0),
        archive.len() as f64 / (1024.0 * 1024.0),
        ratio,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn cmd_verify(args: &[String]) -> Result<()> {
    let Some(path) = args.first() else { usage() };
    let t0 = Instant::now();
    let mapped = TensorBytes::open(path)?;
    let entries = read_vxt_archive(&mapped)?;
    println!(
        "VERIFIED OK: {} ({} tensors, {:.2} MiB, FNV-1a64 integrity passed in {:.2} ms)",
        path,
        entries.len(),
        mapped.len() as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<()> {
    let mut gguf_path: Option<String> = None;
    let mut st_path: Option<String> = None;
    let mut rows = 4096usize;
    let mut cols = 14336usize;
    let mut dtype = DType::BF16;
    let mut threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut cpu_iters = 5usize;
    let mut gpu_iters = 25u32;
    let mut max_tensors = 3usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf_path = args.get(i).cloned();
            }
            "--safetensors" => {
                i += 1;
                st_path = args.get(i).cloned();
            }
            "--rows" => {
                i += 1;
                rows = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(rows);
            }
            "--cols" => {
                i += 1;
                cols = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cols);
            }
            "--dtype" => {
                i += 1;
                dtype = match args.get(i).map(String::as_str) {
                    Some("fp16" | "f16") => DType::F16,
                    _ => DType::BF16,
                };
            }
            "--threads" => {
                i += 1;
                threads = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(threads);
            }
            "--iters" => {
                i += 1;
                gpu_iters = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(gpu_iters);
                cpu_iters = (gpu_iters as usize / 4).clamp(2, 20);
            }
            "--max-tensors" => {
                i += 1;
                max_tensors = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(max_tensors);
            }
            _ => usage(),
        }
        i += 1;
    }

    if let Some(p) = gguf_path {
        let mapped = TensorBytes::open(&p)?;
        let gg = GgufView::parse(&mapped)?;
        let mut cands: Vec<_> = gg
            .tensors
            .iter()
            .filter(|m| {
                m.ggml_type.is_supported_matrix()
                    && m.matrix_dims().is_some_and(|(r, c)| r >= 256 && c >= 256)
            })
            .collect();
        cands.sort_by_key(|m| std::cmp::Reverse(m.numel()));
        if cands.is_empty() {
            return Err(Error::Format(
                "no large 2D weight matrices found in GGUF".into(),
            ));
        }
        for meta in cands.into_iter().take(max_tensors) {
            let (r, c) = meta.matrix_dims().unwrap();
            let native = gg.extract_fp16_words(meta)?;
            let f32s = gg.extract_f32(meta)?;
            let dt = native.as_ref().map(|(d, _)| *d).unwrap_or(DType::F16);
            let words_ref = native.as_ref().map(|(_, w)| w.as_slice());
            let rep = benchmark_matrix(
                &format!("{} ({:?})", meta.name, meta.ggml_type),
                dt,
                r,
                c,
                words_ref,
                &f32s,
                threads,
                cpu_iters,
                gpu_iters,
            )?;
            print_report(&rep);
        }
    } else if let Some(p) = st_path {
        let mapped = TensorBytes::open(&p)?;
        let st = SafeTensorsView::parse(&mapped)?;
        let mut cands: Vec<_> = st
            .tensors
            .iter()
            .filter(|m| m.matrix_dims().is_some_and(|(r, c)| r >= 128 && c >= 128))
            .collect();
        cands.sort_by_key(|m| std::cmp::Reverse(m.numel()));
        for meta in cands.into_iter().take(max_tensors) {
            let (r, c) = meta.matrix_dims().unwrap();
            let native = st.extract_fp16_words(meta)?;
            let f32s = st.extract_f32(meta)?;
            let dt = native.as_ref().map(|(d, _)| *d).unwrap_or(dtype);
            let words_ref = native.as_ref().map(|(_, w)| w.as_slice());
            let rep = benchmark_matrix(
                &meta.name, dt, r, c, words_ref, &f32s, threads, cpu_iters, gpu_iters,
            )?;
            print_report(&rep);
        }
    } else {
        let (words, f32s) = generate_realistic_llm_weights(dtype, rows, cols, 0x2026_08_08);
        let rep = benchmark_matrix(
            "Llama-3-8B-MLP-Synthetic",
            dtype,
            rows,
            cols,
            Some(&words),
            &f32s,
            threads,
            cpu_iters,
            gpu_iters,
        )?;
        print_report(&rep);
    }
    Ok(())
}
