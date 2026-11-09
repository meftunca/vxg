use std::env;
use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;

use vxg_tensor::bench::{benchmark_matrix, generate_realistic_llm_weights, print_report};
use vxg_tensor::chat::cmd_chat_llm;
use vxg_tensor::cli_llm::{cmd_eval_llm, cmd_pack_model, cmd_run_llm};
use vxg_tensor::codebook::{CodebookMatrix, Q4_0Matrix};
use vxg_tensor::format::{read_vxt_model, write_vxt_archive, PackedTensor, VxtEntry};
use vxg_tensor::gguf::GgufView;
use vxg_tensor::lossless::LosslessMatrix;
use vxg_tensor::mmap::TensorBytes;
use vxg_tensor::safetensors::SafeTensorsView;
use vxg_tensor::server::cmd_serve_llm;
use vxg_tensor::wht2::Wht2Matrix;
use vxg_tensor::wht_turbo::TurboWhtMatrix;
use vxg_tensor::{DType, Error, Result};

fn usage() -> ! {
    eprintln!(
        "vxg-tensor — Compressed-Domain AI Model Weight Codec (.vxt) & Fused GPU/SIMD LLM Engine

Usage:
  vxg-tensor info       <model.safetensors | model.gguf | model.vxt | hf_model_dir>
  vxg-tensor pack       <in.safetensors | in.gguf> <out.vxt> [--mode lossless|cb4|cb3|wht2|turbo2|turbo3|q4_0] [--max-tensors N]
  vxg-tensor verify-native <model.safetensors> <config.json> <tokenizer.json> <archive.vxt>
  vxg-tensor pack-native <model.safetensors> <config.json> <tokenizer.json> <out.vxt>
  vxg-tensor pack-model <in.gguf | hf_model_dir | model.safetensors> <out.vxt> [--mode lossless|cb4|cb3|wht2|turbo|q4_0] [--threads T]
  vxg-tensor verify     <archive.vxt>
  vxg-tensor bench      [--gguf <path> | --safetensors <path> | --rows R --cols C] [--dtype bf16|fp16] [--threads T] [--iters N] [--max-tensors K]
  vxg-tensor run        <model.vxt | model.gguf | model.safetensors> --prompt \"...\" [--image <path>] [--max-tokens N] [--temp T] [--cpu] [--raw] [--think]
  vxg-tensor chat       <model.vxt | model.gguf> [--temp T] [--max-tokens N] [--think] [--cpu]
  vxg-tensor serve      [model.vxt | model.gguf] [--port 11435] [--host 127.0.0.1] [--cpu]
  vxg-tensor eval       <ref.gguf | ref.vxt | ref.safetensors> <test.vxt> [--prompt \"...\"] [--steps N] [--cpu]"
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
        "verify-native" => cmd_verify_native(&args[1..]),
        "pack-native" => cmd_pack_native(&args[1..]),
        "pack-model" => cmd_pack_model(&args[1..]),
        "verify" => cmd_verify(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "run" => cmd_run_llm(&args[1..]),
        "chat" => cmd_chat_llm(&args[1..]),
        "serve" => cmd_serve_llm(&args[1..]),
        "eval" => cmd_eval_llm(&args[1..]),
        _ => usage(),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn cmd_info(args: &[String]) -> Result<()> {
    let Some(path) = args.first() else { usage() };
    if Path::new(path).is_dir() {
        let hf = vxg_tensor::hf_pack::resolve_hf_model_dir(Path::new(path)).ok_or_else(|| {
            Error::Format(format!("'{}' is not a recognized HuggingFace model directory", path))
        })?;
        let sum = vxg_tensor::hf_pack::inspect_hf_model_dir(&hf)?;
        println!(
            "Directory: {} ({:.2} MiB across {} safetensors shard(s))",
            path,
            sum.total_bytes as f64 / (1024.0 * 1024.0),
            sum.shard_count
        );
        println!(
            "Format: HuggingFace Model (arch={}, layers={}, vocab={}, tensors={})",
            sum.cfg.arch, sum.cfg.n_layer, sum.cfg.n_vocab, sum.tensor_count
        );
        for (i, t) in sum.preview.iter().take(40).enumerate() {
            println!(
                "  [{:>3}] {:<48} shape={:?} dtype={} ({:.2} KiB)",
                i,
                t.name,
                t.shape,
                t.dtype,
                t.raw_bytes as f64 / 1024.0
            );
        }
        return Ok(());
    }
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
        let (meta, entries) = read_vxt_model(bytes)?;
        println!(
            "Format: VXT1 archive (metadata_keys={}, tensors={})",
            meta.len(),
            entries.len()
        );
        for (i, e) in entries.iter().take(40).enumerate() {
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
        if entries.len() > 40 {
            println!("  ... ({} more tensors)", entries.len() - 40);
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackMode {
    Lossless,
    Cb4,
    Cb3,
    Wht2,
    Turbo2,
    Turbo3,
    Q4_0,
}

fn cmd_pack(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        usage();
    }
    let in_path = &args[0];
    let out_path = &args[1];
    let mut mode = PackMode::Lossless;
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
                    Some("wht2") => PackMode::Wht2,
                    Some("turbo" | "turbo2") => PackMode::Turbo2,
                    Some("turbo3") => PackMode::Turbo3,
                    Some("q4_0") => PackMode::Q4_0,
                    _ => {
                        return Err(Error::Format(
                            "expected --mode lossless|cb4|cb3|wht2|turbo2|turbo3|q4_0".into(),
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

    if max_tensors == 0 {
        return Err(Error::Format("--max-tensors must be greater than zero".into()));
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
            .filter(|m| mode == PackMode::Lossless || m.ggml_type.is_supported_matrix())
        {
            let (rows, cols) = if mode == PackMode::Lossless {
                meta.matrix_dims().unwrap_or((1, meta.numel()))
            } else {
                let Some(dims) = meta.matrix_dims() else { continue };
                if dims.0 < 16 || dims.1 < 32 { continue; }
                dims
            };
            if rows == 0 || cols == 0 {
                return Err(Error::Unsupported(format!("empty tensor '{}' cannot be packed", meta.name)));
            }
            let tensor = match mode {
                PackMode::Lossless => {
                    if let Some((dt, words)) = gg.extract_fp16_words(meta)? {
                        PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                    } else {
                        let data = gg.extract_f32(meta)?;
                        PackedTensor::RawF32 { rows, cols, data }
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
                PackMode::Wht2 => {
                    let f32s = gg.extract_f32(meta)?;
                    PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &f32s)?)
                }
                PackMode::Turbo2 => {
                    let f32s = gg.extract_f32(meta)?;
                    PackedTensor::TurboWht(TurboWhtMatrix::encode(2, rows, cols, &f32s)?)
                }
                PackMode::Turbo3 => {
                    let f32s = gg.extract_f32(meta)?;
                    PackedTensor::TurboWht(TurboWhtMatrix::encode(3, rows, cols, &f32s)?)
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
            let (rows, cols) = if mode == PackMode::Lossless {
                meta.matrix_dims().unwrap_or((1, meta.numel()))
            } else {
                let Some(dims) = meta.matrix_dims() else { continue };
                if dims.0 < 16 || dims.1 < 32 { continue; }
                dims
            };
            if rows == 0 || cols == 0 {
                return Err(Error::Unsupported(format!("empty tensor '{}' cannot be packed", meta.name)));
            }
            let tensor = match mode {
                PackMode::Lossless => {
                    if let Some((dt, words)) = st.extract_fp16_words(meta)? {
                        PackedTensor::Lossless(LosslessMatrix::encode(dt, rows, cols, &words)?)
                    } else {
                        let data = st.extract_f32(meta)?;
                        PackedTensor::RawF32 { rows, cols, data }
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
                PackMode::Wht2 => {
                    let f32s = st.extract_f32(meta)?;
                    PackedTensor::Wht2(Wht2Matrix::encode(rows, cols, &f32s)?)
                }
                PackMode::Turbo2 => {
                    let f32s = st.extract_f32(meta)?;
                    PackedTensor::TurboWht(TurboWhtMatrix::encode(2, rows, cols, &f32s)?)
                }
                PackMode::Turbo3 => {
                    let f32s = st.extract_f32(meta)?;
                    PackedTensor::TurboWht(TurboWhtMatrix::encode(3, rows, cols, &f32s)?)
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

    if max_tensors != usize::MAX {
        println!("Explicit --max-tensors subset: packed at most {max_tensors} source tensors");
    }
    let archive = write_vxt_archive(&entries)?;
    fs::write(Path::new(out_path), &archive).map_err(|e| Error::Io(e.to_string()))?;
    let ratio = (raw_16b_sum as f64) / (archive.len().max(1) as f64);
    println!(
        "Packed {} tensors -> {} ({:.2} MiB -> {:.2} MiB, {:.2}x vs 16-bit) in {:.2}s",
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
    let (meta, entries) = read_vxt_model(&mapped)?;
    println!(
        "VERIFIED OK: {} ({} meta keys, {} tensors, {:.2} MiB, FNV-1a64 integrity passed in {:.2} ms)",
        path,
        meta.len(),
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

fn cmd_pack_native(args: &[String]) -> Result<()> {
    if args.len() != 4 { return Err(Error::Format("usage: pack-native <model.safetensors> <config.json> <tokenizer.json> <out.vxt> (lossless language trunk only; single file)".into())); }
    let source = TensorBytes::open(&args[0])?;
    let cfg = fs::read(&args[1]).map_err(|e| Error::Io(e.to_string()))?;
    let tok = fs::read(&args[2]).map_err(|e| Error::Io(e.to_string()))?;
    let start = Instant::now();
    let (config, archive) = vxg_tensor::native_qwen35::pack(source.as_slice(), &cfg, &tok)?;
    fs::write(&args[3], &archive).map_err(|e| Error::Io(e.to_string()))?;
    println!("Packed {}: language-only, original tensor names/shapes/dtypes/bits; visual and MTP excluded in manifest; {} bytes in {:.2}s", config.model_name, archive.len(), start.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_verify_native(args: &[String]) -> Result<()> {
    if args.len()!=4 {return Err(Error::Format("usage: verify-native <model.safetensors> <config.json> <tokenizer.json> <archive.vxt>".into()));}
    let source=TensorBytes::open(&args[0])?;let archive=TensorBytes::open(&args[3])?;
    let config=fs::read(&args[1]).map_err(|e|Error::Io(e.to_string()))?;
    let tokenizer=fs::read(&args[2]).map_err(|e|Error::Io(e.to_string()))?;
    let count=vxg_tensor::native_qwen35::verify(source.as_slice(),&config,&tokenizer,archive.as_slice())?;
    println!("Verified {count} native language tensors: source names, shapes, dtypes and every weight bit match; raw config/tokenizer match; visual/MTP exclusions match manifest.");Ok(())
}
