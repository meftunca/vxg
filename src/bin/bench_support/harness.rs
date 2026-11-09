use std::{
    fs,
    hint::black_box,
    io::Write,
    time::{Duration, Instant},
};

use vxg::{
    baselines::{Codec, Encoded, TrainedDict},
    gpu::{GpuCodec, GpuOptions},
    DecodeOptions,
};

use super::{
    allocator_name,
    config::{CodecPoint, Config, DatasetSummary},
    metrics::{measure_op, measure_op_ext},
    R,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_baselines_and_gpu(
    c: &Config,
    name: &str,
    data: &[u8],
    frame: &[u8],
    dec_opt: &DecodeOptions,
    out: &mut [u8],
    ds_summary: &mut DatasetSummary,
    csv: &mut fs::File,
    samples_csv: &mut Option<fs::File>,
    gpu: &mut Option<GpuCodec>,
    lz: &str,
    zs: &str,
) -> R<()> {
    let dict_candidates = TrainedDict::train_sweep(data);
    let sweep_desc: Vec<String> = dict_candidates
        .iter()
        .map(|(cap, d)| {
            format!(
                "{}K:{}B({:.1}ms)",
                cap >> 10,
                d.bytes.len(),
                d.train_ns / 1e6
            )
        })
        .collect();
    eprintln!(
        "{name:22} zstd-dict      sweep      [{}]",
        sweep_desc.join(", ")
    );

    let mut codecs = vec![
        Codec::Lz4 { acceleration: 1 },
        Codec::Lz4 { acceleration: 4 },
    ];
    if !c.fast_baselines {
        codecs.push(Codec::Lz4Hc { level: 9 });
    }
    codecs.extend([
        Codec::Zstd { level: -3 },
        Codec::Zstd { level: 1 },
        Codec::Zstd { level: 3 },
        Codec::ZstdDict { level: 3 },
    ]);
    if !c.fast_baselines {
        codecs.extend([
            Codec::Zstd { level: 9 },
            Codec::ZstdDict { level: 9 },
            Codec::ZstdWholeLong { level: 19 },
        ]);
    }
    for codec in codecs {
        let is_whole = matches!(codec, Codec::ZstdWholeLong { .. });
        let is_dict = matches!(codec, Codec::ZstdDict { .. });
        let is_heavy_enc = matches!(
            codec,
            Codec::Lz4Hc { .. }
                | Codec::Zstd { level: 9.. }
                | Codec::ZstdDict { level: 9.. }
                | Codec::ZstdWholeLong { .. }
        );
        let (chosen_dict, compressed) = if is_dict {
            let (best_d, best_cap, enc) = Encoded::compress_best_dict_sweep(
                data,
                codec,
                &dict_candidates,
                c.block_size,
                c.threads,
            )?;
            eprintln!(
                "{name:22} {:14} best-dict  cap={}K dict_bytes={} ratio={:.4}",
                codec.name(),
                best_cap >> 10,
                best_d.bytes.len(),
                enc.encoded_size() as f64 / data.len() as f64
            );
            if matches!(codec, Codec::ZstdDict { level: 3 }) {
                ds_summary.dict_l3_bytes = best_d.bytes.len();
                ds_summary.dict_bytes = best_d.bytes.len();
                ds_summary.dict_train_ms = best_d.train_ns / 1e6;
            } else if matches!(codec, Codec::ZstdDict { level: 9 }) {
                ds_summary.dict_l9_bytes = best_d.bytes.len();
                ds_summary.dict_bytes = best_d.bytes.len();
                ds_summary.dict_train_ms = best_d.train_ns / 1e6;
            }
            (Some(best_d), enc)
        } else {
            (
                None,
                Encoded::compress(data, codec, c.block_size, c.threads)?,
            )
        };
        compressed.decode_into(out, c.threads)?;
        assert_eq!(out, data, "baseline correctness before timing");
        let en = measure_op_ext(c, data.len(), is_heavy_enc, || {
            let f = if let Some(d) = chosen_dict {
                Encoded::compress_with_dict(black_box(data), codec, d, c.block_size, c.threads)?
            } else {
                Encoded::compress(black_box(data), codec, c.block_size, c.threads)?
            };
            black_box(f);
            Ok(())
        })?;
        let enc_profile = if is_whole {
            "cpu_whole_file_encode"
        } else {
            "cpu_allocating_encode"
        };
        let dec_profile = if is_whole {
            "cpu_whole_file_decode"
        } else {
            "cpu_preallocated_decode"
        };
        row(
            csv,
            samples_csv,
            c,
            name,
            &codec.name(),
            enc_profile,
            "compress",
            data.len(),
            compressed.encoded_size(),
            &en.ns_samples,
            lz,
            zs,
            "",
        )?;
        let de = measure_op(c, data.len(), || {
            compressed.decode_into(black_box(out), c.threads)?;
            black_box(&*out);
            Ok(())
        })?;
        assert_eq!(out, data, "baseline correctness after timing");
        row(
            csv,
            samples_csv,
            c,
            name,
            &codec.name(),
            dec_profile,
            "decompress",
            data.len(),
            compressed.encoded_size(),
            &de.ns_samples,
            lz,
            zs,
            "",
        )?;
        let pt = CodecPoint {
            ratio: compressed.encoded_size() as f64 / data.len() as f64,
            enc: data.len() as f64 / median(&en.ns_samples),
            dec: data.len() as f64 / median(&de.ns_samples),
            enc_j_gb: en.j_per_gb,
            dec_j_gb: de.j_per_gb,
        };
        match codec {
            Codec::Lz4 { acceleration: 1 } => ds_summary.lz4_a1 = pt,
            Codec::Lz4 { acceleration: 4 } => ds_summary.lz4_a4 = pt,
            Codec::Lz4Hc { level: 9 } => ds_summary.lz4hc_9 = pt,
            Codec::Zstd { level: -3 } => ds_summary.zstd_m3 = pt,
            Codec::Zstd { level: 1 } => ds_summary.zstd_1 = pt,
            Codec::Zstd { level: 3 } => ds_summary.zstd_3 = pt,
            Codec::Zstd { level: 9 } => ds_summary.zstd_9 = pt,
            Codec::ZstdDict { level: 3 } => ds_summary.zstd_dict_3 = pt,
            Codec::ZstdDict { level: 9 } => ds_summary.zstd_dict_9 = pt,
            Codec::ZstdWholeLong { level: 19 } => ds_summary.zstd_whole_19 = pt,
            _ => {}
        }
    }
    let lc = ds_summary.lz4_envelope();
    let zc = ds_summary.zstd_block_envelope();
    let ac = ds_summary.all_block_envelope();
    eprintln!(
        "{name:22} SWEEP-LZ4      envelope   ratio={:.3} enc={:7.3} dec={:7.3} GB/s [min-ratio + max-speed]",
        lc.ratio, lc.enc, lc.dec
    );
    eprintln!(
        "{name:22} SWEEP-ZSTD+D   envelope   ratio={:.3} enc={:7.3} dec={:7.3} GB/s [min-ratio(incl dict) + max-speed]",
        zc.ratio, zc.enc, zc.dec
    );
    eprintln!(
        "{name:22} SWEEP-ALL-BLK  envelope   ratio={:.3} enc={:7.3} dec={:7.3} GB/s [VXG ratio={:.3} enc={:.3} dec={:.3}]",
        ac.ratio, ac.enc, ac.dec, ds_summary.vxg.ratio, ds_summary.vxg.enc, ds_summary.vxg.dec
    );
    if let Some(device) = gpu.as_mut() {
        let options = GpuOptions {
            block_size: c.block_size,
            checksum: false,
            max_output: data.len(),
            ..GpuOptions::default()
        };
        let (compressed, _) = device.compress(data, &options)?;
        assert_eq!(
            vxg::decompress(&compressed, dec_opt)?,
            data,
            "GPU encode / CPU decode correctness"
        );
        assert_eq!(
            device.decompress(frame, &options)?.0,
            data,
            "CPU encode / GPU decode correctness"
        );
        let device_name = device.device_name().to_owned();
        let en = measure(c, || {
            black_box(device.compress(black_box(data), &options)?);
            Ok(())
        })?;
        row(
            csv,
            samples_csv,
            c,
            name,
            "vxg-gpu",
            "gpu_host_e2e_allocating",
            "compress",
            data.len(),
            compressed.len(),
            &en,
            lz,
            zs,
            &device_name,
        )?;
        let de = measure(c, || {
            black_box(device.decompress(black_box(&compressed), &options)?);
            Ok(())
        })?;
        row(
            csv,
            samples_csv,
            c,
            name,
            "vxg-gpu",
            "gpu_host_e2e_allocating",
            "decompress",
            data.len(),
            compressed.len(),
            &de,
            lz,
            zs,
            &device_name,
        )?;
        let (decoded, report) = device.decompress(&compressed, &options)?;
        assert_eq!(decoded, data);
        eprintln!(
            "GPU device timestamp observation (not median, not e2e): {} kernel_ms={:?}, host_e2e_ms={:.3}",
            name, report.kernel_ms, report.end_to_end_ms
        );
    }
    Ok(())
}

pub(super) fn run_microbench(c: &Config, engine: vxg::simd::Engine) -> R<()> {
    let sizes = [8usize, 16, 32, 64, 256, 4096, 65536, 1 << 20];
    println!("operation,simd,bytes,median_ns,min_ns,max_ns,GBps,samples");
    for &n in &sizes {
        let a = vec![0x5au8; n];
        let b = vec![0x5au8; n];
        let (median, min, max) = micro_measure(c, || {
            black_box(engine.lcp(black_box(&a), black_box(&b)));
        });
        println!(
            "lcp-equal,{:?},{n},{median:.3},{min:.3},{max:.3},{:.4},{}",
            engine.kind(),
            n as f64 / median,
            c.samples
        );

        let src = vec![0x3cu8; n];
        let mut dst = vec![0u8; n];
        let (median, min, max) = micro_measure(c, || {
            engine.copy(black_box(&mut dst), black_box(&src));
            black_box(dst.as_ptr());
        });
        println!(
            "copy,{:?},{n},{median:.3},{min:.3},{max:.3},{:.4},{}",
            engine.kind(),
            n as f64 / median,
            c.samples
        );
    }
    let seed: Vec<u8> = (0..37).map(|x| x as u8).collect();
    let mut dst = vec![0u8; 1 << 20];
    let (median, min, max) = micro_measure(c, || {
        engine.pattern(black_box(&mut dst), black_box(&seed));
        black_box(dst.as_ptr());
    });
    println!(
        "pattern-period-37,{:?},{},{median:.3},{min:.3},{max:.3},{:.4},{}",
        engine.kind(),
        dst.len(),
        dst.len() as f64 / median,
        c.samples
    );
    let mut dst = vec![0u8; 1 << 20];
    let (median, min, max) = micro_measure(c, || {
        platform_pattern(black_box(&mut dst), black_box(&seed));
        black_box(dst.as_ptr());
    });
    println!(
        "pattern-platform-period-37,{:?},{},{median:.3},{min:.3},{max:.3},{:.4},{}",
        engine.kind(),
        dst.len(),
        dst.len() as f64 / median,
        c.samples
    );
    Ok(())
}

fn platform_pattern(dst: &mut [u8], seed: &[u8]) {
    if seed.len() == 1 {
        dst.fill(seed[0]);
        return;
    }
    let step = (1024 / seed.len()) * seed.len();
    let mut tile = [0u8; 1024];
    tile[..seed.len()].copy_from_slice(seed);
    let mut filled = seed.len();
    while filled < step {
        let take = filled.min(step - filled);
        let (prefix, suffix) = tile.split_at_mut(filled);
        suffix[..take].copy_from_slice(&prefix[..take]);
        filled += take;
    }
    for chunk in dst.chunks_mut(step) {
        chunk.copy_from_slice(&tile[..chunk.len()]);
    }
}

fn micro_measure<F>(c: &Config, mut f: F) -> (f64, f64, f64)
where
    F: FnMut(),
{
    let mut samples = Vec::with_capacity(c.samples);
    let duration = Duration::from_secs_f64(c.seconds);
    for _ in 0..c.samples {
        let start = Instant::now();
        let mut ops = 0u64;
        while start.elapsed() < duration {
            f();
            ops += 1;
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / ops.max(1) as f64);
    }
    samples.sort_by(f64::total_cmp);
    let mid = samples.len() / 2;
    let median = if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    };
    (median, samples[0], samples[samples.len() - 1])
}

pub(super) fn measure<F>(c: &Config, mut f: F) -> R<Vec<f64>>
where
    F: FnMut() -> R<()>,
{
    f()?;
    f()?; // Warm-up, never included.
    let mut samples = Vec::with_capacity(c.samples);
    let duration = Duration::from_secs_f64(c.seconds);
    for _ in 0..c.samples {
        let start = Instant::now();
        let mut n = 0u64;
        loop {
            f()?;
            n += 1;
            if start.elapsed() >= duration {
                break;
            }
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / n as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples)
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

pub(super) fn median(s: &[f64]) -> f64 {
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) * 0.5
    } else {
        s[mid]
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn row(
    out: &mut fs::File,
    samples_out: &mut Option<fs::File>,
    c: &Config,
    name: &str,
    codec: &str,
    profile: &str,
    op: &str,
    n: usize,
    m: usize,
    s: &[f64],
    lz: &str,
    zs: &str,
    device: &str,
) -> R<()> {
    let median = median(s);
    let gbps = n as f64 / median;
    if let Some(samples_out) = samples_out.as_mut() {
        for (rank, &ns) in s.iter().enumerate() {
            writeln!(
                samples_out,
                "{},{},{},{},{},{:.3},{:.4},{},{},{},{},{}",
                quote(name),
                codec,
                profile,
                op,
                rank,
                ns,
                n as f64 / ns,
                n,
                m,
                c.threads,
                c.block_size,
                allocator_name()
            )?;
        }
    }
    writeln!(
        out,
        "{},{},{},{},{},{},{:.8},{},{},{:.3},{:.3},{:.3},{:.4},{},off,{},{},{},{}",
        quote(name),
        codec,
        profile,
        op,
        n,
        m,
        m as f64 / n as f64,
        c.threads,
        c.block_size,
        median,
        s[0],
        s[s.len() - 1],
        gbps,
        s.len(),
        lz,
        zs,
        quote(device),
        allocator_name()
    )?;
    eprintln!(
        "{name:22} {codec:14} {op:10} ratio={:.3} {:8.3} GB/s [{profile}]",
        m as f64 / n as f64,
        gbps
    );
    Ok(())
}
