use std::{
    fs,
    hint::black_box,
    io::Write,
    time::{Duration, Instant},
};

use vxg::{
    baselines::{Codec, Encoded},
    gpu::{GpuCodec, GpuOptions},
    DecodeOptions,
};

use super::{
    allocator_name,
    metrics::measure_op,
    Config, DatasetSummary, R,
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
    for codec in [
        Codec::Lz4 { acceleration: 1 },
        Codec::Lz4 { acceleration: 4 },
        Codec::Zstd { level: -3 },
        Codec::Zstd { level: 1 },
        Codec::Zstd { level: 3 },
    ] {
        let compressed = Encoded::compress(data, codec, c.block_size, c.threads)?;
        compressed.decode_into(out, c.threads)?;
        assert_eq!(out, data, "baseline correctness before timing");
        let en = measure_op(c, data.len(), || {
            let f = Encoded::compress(black_box(data), codec, c.block_size, c.threads)?;
            black_box(f);
            Ok(())
        })?;
        row(
            csv,
            samples_csv,
            c,
            name,
            &codec.name(),
            "cpu_allocating_encode",
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
            "cpu_preallocated_decode",
            "decompress",
            data.len(),
            compressed.encoded_size(),
            &de.ns_samples,
            lz,
            zs,
            "",
        )?;
        let ratio = compressed.encoded_size() as f64 / data.len() as f64;
        let enc_gbps = data.len() as f64 / median(&en.ns_samples);
        let dec_gbps = data.len() as f64 / median(&de.ns_samples);
        match codec {
            Codec::Lz4 { acceleration: 1 } => {
                ds_summary.lz4_ratio = ratio;
                ds_summary.lz4_enc = enc_gbps;
                ds_summary.lz4_dec = dec_gbps;
                ds_summary.lz4_enc_j_gb = en.j_per_gb;
                ds_summary.lz4_dec_j_gb = de.j_per_gb;
            }
            Codec::Zstd { level: 1 } => {
                ds_summary.zstd1_ratio = ratio;
                ds_summary.zstd1_enc = enc_gbps;
                ds_summary.zstd1_dec = dec_gbps;
                ds_summary.zstd1_enc_j_gb = en.j_per_gb;
                ds_summary.zstd1_dec_j_gb = de.j_per_gb;
            }
            Codec::Zstd { level: 3 } => {
                ds_summary.zstd3_ratio = ratio;
                ds_summary.zstd3_enc = enc_gbps;
                ds_summary.zstd3_dec = dec_gbps;
                ds_summary.zstd3_enc_j_gb = en.j_per_gb;
                ds_summary.zstd3_dec_j_gb = de.j_per_gb;
            }
            _ => {}
        }
    }
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
