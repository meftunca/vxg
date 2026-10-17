use std::{
    env, fs,
    hint::black_box,
    io::Write,
    time::{Duration, Instant},
};
use vxg::{
    baselines::{self, Codec, Encoded},
    compress_with_stats,
    gpu::{Backend, GpuCodec, GpuOptions},
    DecodeOptions, Frame, Options, Simd,
};
type R<T> = Result<T, Box<dyn std::error::Error>>;
#[cfg(all(feature = "mimalloc", feature = "jemalloc"))]
compile_error!("benchmark allocator features are mutually exclusive");
#[cfg(any(
    all(feature = "alloc-stats", feature = "mimalloc"),
    all(feature = "alloc-stats", feature = "jemalloc")
))]
compile_error!("alloc-stats cannot be combined with a replacement allocator");
#[cfg(all(
    feature = "mimalloc",
    not(any(feature = "jemalloc", feature = "alloc-stats"))
))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(all(
    feature = "jemalloc",
    not(any(feature = "mimalloc", feature = "alloc-stats"))
))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(all(
    feature = "mimalloc",
    not(any(feature = "jemalloc", feature = "alloc-stats"))
))]
fn allocator_name() -> &'static str {
    "mimalloc"
}
#[cfg(all(
    feature = "jemalloc",
    not(any(feature = "mimalloc", feature = "alloc-stats"))
))]
fn allocator_name() -> &'static str {
    "jemalloc"
}
#[cfg(feature = "alloc-stats")]
fn allocator_name() -> &'static str {
    "system+alloc-stats"
}
#[cfg(not(any(feature = "mimalloc", feature = "jemalloc", feature = "alloc-stats")))]
fn allocator_name() -> &'static str {
    "system"
}

#[cfg(feature = "alloc-stats")]
mod alloc_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct CountingAllocator;
    static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
    static DEALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
    static REALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    static DEALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

    #[global_allocator]
    pub static GLOBAL: CountingAllocator = CountingAllocator;
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            DEALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            System.dealloc(ptr, layout);
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let out = System.realloc(ptr, layout, new_size);
            if !out.is_null() {
                REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
                DEALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            }
            out
        }
    }
    #[derive(Clone, Copy)]
    pub struct Snapshot {
        pub alloc_calls: usize,
        pub dealloc_calls: usize,
        pub realloc_calls: usize,
        pub alloc_bytes: usize,
        pub dealloc_bytes: usize,
    }
    pub fn reset() {
        for x in [
            &ALLOC_CALLS,
            &DEALLOC_CALLS,
            &REALLOC_CALLS,
            &ALLOC_BYTES,
            &DEALLOC_BYTES,
        ] {
            x.store(0, Ordering::Relaxed);
        }
    }
    pub fn snapshot() -> Snapshot {
        Snapshot {
            alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
            dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
            realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
            alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
            dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
        }
    }
}
#[derive(Clone)]
struct Config {
    size: usize,
    seconds: f64,
    samples: usize,
    threads: usize,
    block_size: usize,
    simd: Simd,
    csv: String,
    samples_csv: Option<String>,
    markdown: Option<String>,
    suite: String,
    files: Vec<String>,
    only: Option<String>,
    gpu: Option<Backend>,
    microbench: bool,
    vxg_only: bool,
    decode_only: bool,
    validated_decode: bool,
    legacy_v2: bool,
}
#[derive(Default, Clone)]
struct DatasetSummary {
    name: String,
    raw_bytes: usize,
    vxg_ratio: f64,
    vxg_enc: f64,
    vxg_dec: f64,
    lz4_ratio: f64,
    lz4_enc: f64,
    lz4_dec: f64,
    zstd1_ratio: f64,
    zstd1_enc: f64,
    zstd1_dec: f64,
    zstd3_ratio: f64,
    zstd3_enc: f64,
    zstd3_dec: f64,
}
fn next<'a>(args: &'a [String], i: &mut usize) -> R<&'a str> {
    *i += 1;
    Ok(args.get(*i).ok_or("missing value")?.as_str())
}
fn parse_threads(s: &str) -> R<usize> {
    if s.eq_ignore_ascii_case("auto") || s == "0" {
        Ok(std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .clamp(1, 256))
    } else {
        Ok(s.parse()?)
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("benchmark failed: {e}");
        std::process::exit(1);
    }
}
fn run() -> R<()> {
    let mut c = Config {
        size: 4 << 20,
        seconds: 0.1,
        samples: 5,
        threads: 1,
        block_size: 65536,
        simd: Simd::Auto,
        csv: "reports/local-benchmark.csv".into(),
        samples_csv: None,
        markdown: None,
        suite: "default".into(),
        files: Vec::new(),
        only: None,
        gpu: None,
        microbench: false,
        vxg_only: false,
        decode_only: false,
        validated_decode: false,
        legacy_v2: false,
    };
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--size-mib" => {
                c.size = next(&args, &mut i)?
                    .parse::<usize>()?
                    .checked_mul(1 << 20)
                    .ok_or("size overflow")?
            }
            "--seconds" => c.seconds = next(&args, &mut i)?.parse()?,
            "--samples" => c.samples = next(&args, &mut i)?.parse()?,
            "--threads" => c.threads = parse_threads(next(&args, &mut i)?)?,
            "--block-size" => c.block_size = next(&args, &mut i)?.parse()?,
            "--simd" => c.simd = next(&args, &mut i)?.parse()?,
            "--csv" => c.csv = next(&args, &mut i)?.into(),
            "--samples-csv" => c.samples_csv = Some(next(&args, &mut i)?.into()),
            "--markdown" => c.markdown = Some(next(&args, &mut i)?.into()),
            "--suite" => c.suite = next(&args, &mut i)?.into(),
            "--gpu" => c.gpu = Some(next(&args, &mut i)?.parse()?),
            "--file" => c.files.push(next(&args, &mut i)?.into()),
            "--only" => c.only = Some(next(&args, &mut i)?.into()),
            "--microbench" => c.microbench = true,
            "--vxg-only" => c.vxg_only = true,
            "--decode-only" => c.decode_only = true,
            "--validated-decode" => c.validated_decode = true,
            "--v3" | "--v4" => {} // unified pipeline is the default; kept for script compatibility
            "--legacy" => c.legacy_v2 = true,
            "--help" => {
                println!("vxg-bench [--suite default|v4|all] [--size-mib N] [--seconds 0.1] [--samples 5] [--threads 1|auto] [--block-size 65536] [--simd auto] [--csv FILE] [--samples-csv FILE] [--markdown FILE] [--file CORPUS_FILE]... [--only DATASET] [--gpu metal|vulkan|cuda] [--microbench] [--vxg-only] [--decode-only] [--validated-decode] [--legacy]\nAll CPU codecs use independent blocks, RAW fallback and no checksum. --suite v4 runs the 10-dataset multi-format v4 suite (CSV, TSV, Logfmt, 4x NDJSON, Source/Docs, 32-bpp BMP, 24-bpp BMP). --markdown writes a GitHub Markdown comparison table. --only selects one built-in dataset name or --file path. --decode-only skips encode rows and baseline/GPU work after creating one VXG frame. --validated-decode validates the immutable VXG command streams once before timing repeated CPU decode; it is separately profiled from safe one-shot decode and is unavailable for v3 structured-column frames. --legacy emits LZ-only v2 frames instead of the default unified adaptive v4 pipeline. --samples-csv retains every timed sample (sorted by duration) with the same profile labels as --csv. GPU rows are explicitly host-e2e, not CPU preallocated rows. --microbench measures Engine LCP/copy/pattern kernels without codec framing. --vxg-only skips baseline codecs and GPU rows.");
                return Ok(());
            }
            x => return Err(format!("unknown option {x}").into()),
        }
        i += 1;
    }
    if c.size == 0
        || !c.seconds.is_finite()
        || c.seconds <= 0.0
        || c.seconds > 60.0
        || c.samples < 3
        || c.samples > 101
    {
        return Err("invalid size, seconds (0..60], or samples (3..101)".into());
    }
    let engine = vxg::simd::Engine::new(c.simd)?;
    if c.microbench {
        return run_microbench(&c, engine);
    }
    #[cfg(feature = "alloc-stats")]
    eprintln!(
        "ALLOC-STATS enabled: allocation counters perturb timing; use counts only, not GB/s."
    );
    let (lz, zs) = baselines::versions();
    eprintln!("VXG {} / liblz4 {lz} / libzstd {zs}; {} {}; SIMD={:?}; allocator={}; threads={}; block={}",env!("CARGO_PKG_VERSION"),env::consts::OS,env::consts::ARCH,engine.kind(),allocator_name(),c.threads,c.block_size);
    eprintln!("CPU profiles: cpu_allocating_encode; cpu_preallocated_decode = prepared directory + checked command decode; cpu_full_frame_preallocated_decode = parse + checked decode; cpu_prepared_validate_commands = command validation on a prepared directory; cpu_parse_validate_commands = parse + command validation; cpu_prevalidated_decode = prepared proof + repeated decode only. Per-call contexts included. RAW fallback for ALL codecs. Checksums OFF. Warm cache. GPU: separate host-e2e profile.");
    let mut datasets = if c.files.is_empty() {
        match c.suite.as_str() {
            "default" => datasets(c.size),
            "v4" => datasets_v4(c.size),
            "all" => {
                let mut v = datasets_v4(c.size);
                v.extend(datasets(c.size));
                v
            }
            other => return Err(format!("unknown --suite {other} (expected default, v4, or all)").into()),
        }
    } else {
        Vec::new()
    };
    for path in &c.files {
        datasets.push((path.clone(), fs::read(path)?));
    }
    if let Some(only) = &c.only {
        datasets.retain(|(name, _)| name == only);
        if datasets.is_empty() {
            return Err(format!("unknown dataset {only}").into());
        }
    }
    if let Some(parent) = std::path::Path::new(&c.csv).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut csv = fs::File::create(&c.csv)?;
    writeln!(csv,"dataset,codec,profile,operation,raw_bytes,encoded_bytes,ratio_encoded_over_raw,threads,block_size,median_ns,min_ns,max_ns,median_GBps,samples,checksums,lz4_version,zstd_version,device,allocator")?;
    let mut samples_csv = match &c.samples_csv {
        Some(path) => {
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            let mut out = fs::File::create(path)?;
            writeln!(out,"dataset,codec,profile,operation,sample_rank,ns_per_operation,GBps,raw_bytes,encoded_bytes,threads,block_size,allocator")?;
            Some(out)
        }
        None => None,
    };
    let mut gpu = match c.gpu {
        Some(b) => Some(GpuCodec::new(b)?),
        None => None,
    };
    let mut summaries: Vec<DatasetSummary> = Vec::new();
    for (name, data) in datasets {
        if data.is_empty() {
            eprintln!("skip empty dataset {name}");
            continue;
        }
        let enc_opt = Options {
            block_size: c.block_size,
            threads: c.threads,
            checksum: false,
            simd: c.simd,
            legacy_v2: c.legacy_v2,
            ..Options::default()
        };
        let dec_opt = DecodeOptions {
            threads: c.threads,
            simd: c.simd,
            max_output: data.len(),
            verify_checksum: false,
        };
        #[cfg(feature = "alloc-stats")]
        alloc_probe::reset();
        let (stats_frame, work_stats) = compress_with_stats(&data, &enc_opt)?;
        // Timed encode invokes `compress`, not the diagnostic API. Establish
        // byte-for-byte equivalence before its frame size is reported.
        let frame = vxg::compress(&data, &enc_opt)?;
        if frame != stats_frame {
            return Err(
                format!("VXG compress() diverged from compress_with_stats() for {name}").into(),
            );
        }
        #[cfg(feature = "alloc-stats")]
        {
            let a = alloc_probe::snapshot();
            eprintln!("{name:22} alloc calls={} realloc={} dealloc={} alloc_bytes={} dealloc_bytes={} live_bytes={}",a.alloc_calls,a.realloc_calls,a.dealloc_calls,a.alloc_bytes,a.dealloc_bytes,a.alloc_bytes.saturating_sub(a.dealloc_bytes));
        }
        let parsed = Frame::parse(&frame, data.len())?;
        let command_validatable = !parsed.blocks().any(|b| b.kind == vxg::BLOCK_COLUMN);
        let validated = if c.validated_decode && command_validatable {
            Some(Frame::parse(&frame, data.len())?.into_validated()?)
        } else {
            if c.validated_decode {
                eprintln!("{name:22} note: validated replay is unavailable for structured-column frames; skipping the prevalidated profile");
            }
            None
        };
        let mut out = vec![0u8; data.len()];
        eprintln!("{name:22} VXG work raw_fast={} sample_words={} probes={} occupied={} candidate_checks={} collisions={} lcp_calls={} lcp_bytes={} matches={} matched_bytes={} streams=control:{} attrs:{} literals:{} ops={:?} op7_dist={:?} run={:?} zero_literals={}",work_stats.raw_fast_path_blocks,work_stats.raw_sample_words,work_stats.probes,work_stats.occupied_probes,work_stats.candidate_checks,work_stats.hash_collisions,work_stats.lcp_calls,work_stats.lcp_bytes,work_stats.accepted_matches,work_stats.matched_bytes,work_stats.control_bytes,work_stats.attribute_bytes,work_stats.literal_bytes,work_stats.op_counts,work_stats.op7_distance_buckets,work_stats.match_run_buckets,work_stats.zero_literal_commands);
        if let Some(checked) = validated.as_ref() {
            checked.decode_into(&mut out, &dec_opt)?;
        } else {
            parsed.decode_into(&mut out, &dec_opt)?;
        }
        assert_eq!(out, data, "VXG correctness before timing");
        let mut ds_summary = DatasetSummary {
            name: std::path::Path::new(&name)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&name)
                .to_owned(),
            raw_bytes: data.len(),
            vxg_ratio: frame.len() as f64 / data.len() as f64,
            ..DatasetSummary::default()
        };
        if !c.decode_only {
            let en = measure(&c, || {
                #[cfg(feature = "encode-with-stats")]
                let f = vxg::compress_with_stats(black_box(&data), &enc_opt)?;
                #[cfg(not(feature = "encode-with-stats"))]
                let f = vxg::compress(black_box(&data), &enc_opt)?;
                black_box(f);
                Ok(())
            })?;
            ds_summary.vxg_enc = data.len() as f64 / median(&en);
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                "vxg",
                "cpu_allocating_encode",
                "compress",
                data.len(),
                frame.len(),
                &en,
                &lz,
                &zs,
                "",
            )?;
        }
        let safe_de = measure(&c, || {
            parsed.decode_into(black_box(&mut out), &dec_opt)?;
            black_box(&out);
            Ok(())
        })?;
        assert_eq!(out, data, "VXG correctness after timing");
        ds_summary.vxg_dec = data.len() as f64 / median(&safe_de);
        row(
            &mut csv,
            &mut samples_csv,
            &c,
            &name,
            "vxg",
            "cpu_preallocated_decode",
            "decompress",
            data.len(),
            frame.len(),
            &safe_de,
            &lz,
            &zs,
            "",
        )?;
        let full_de = measure(&c, || {
            let parsed = Frame::parse(black_box(&frame), data.len())?;
            parsed.decode_into(black_box(&mut out), &dec_opt)?;
            black_box(&out);
            Ok(())
        })?;
        assert_eq!(out, data, "VXG full-frame correctness after timing");
        row(
            &mut csv,
            &mut samples_csv,
            &c,
            &name,
            "vxg",
            "cpu_full_frame_preallocated_decode",
            "decompress",
            data.len(),
            frame.len(),
            &full_de,
            &lz,
            &zs,
            "",
        )?;
        let mut validation = None;
        if command_validatable {
            let samples = measure(&c, || {
                parsed.validate_commands()?;
                black_box(parsed.block_count());
                Ok(())
            })?;
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                "vxg",
                "cpu_prepared_validate_commands",
                "validate_commands",
                data.len(),
                frame.len(),
                &samples,
                &lz,
                &zs,
                "",
            )?;
            validation = Some(samples);
            let parse_validation = measure(&c, || {
                let checked = Frame::parse(black_box(&frame), data.len())?.into_validated()?;
                black_box(checked.block_count());
                Ok(())
            })?;
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                "vxg",
                "cpu_parse_validate_commands",
                "parse_validate_commands",
                data.len(),
                frame.len(),
                &parse_validation,
                &lz,
                &zs,
                "",
            )?;
        } else {
            eprintln!("{name:22} note: command validation rows are unavailable for structured-column frames; skipping");
        }
        if let Some(checked) = validated.as_ref() {
            let prevalidated_de = measure(&c, || {
                checked.decode_into(black_box(&mut out), &dec_opt)?;
                black_box(&out);
                Ok(())
            })?;
            assert_eq!(out, data, "VXG prevalidated correctness after timing");
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                "vxg",
                "cpu_prevalidated_decode",
                "decompress",
                data.len(),
                frame.len(),
                &prevalidated_de,
                &lz,
                &zs,
                "",
            )?;
            let safe_ns = median(&safe_de);
            let prevalidated_ns = median(&prevalidated_de);
            let validation_ns = median(
                validation
                    .as_ref()
                    .expect("validated replay implies command validation"),
            );
            if prevalidated_ns < safe_ns {
                let reuse = (validation_ns / (safe_ns - prevalidated_ns)).floor() as u64 + 1;
                eprintln!("{name:22} validated reuse break-even: k >= {reuse} (T_validate + k*T_prevalidated < k*T_safe)");
            } else {
                eprintln!("{name:22} validated reuse break-even: none; prevalidated decode is not faster than safe prepared decode");
            }
        }
        if c.vxg_only || c.decode_only {
            summaries.push(ds_summary);
            csv.flush()?;
            continue;
        }
        for codec in [
            Codec::Lz4 { acceleration: 1 },
            Codec::Lz4 { acceleration: 4 },
            Codec::Zstd { level: -3 },
            Codec::Zstd { level: 1 },
            Codec::Zstd { level: 3 },
        ] {
            let compressed = Encoded::compress(&data, codec, c.block_size, c.threads)?;
            compressed.decode_into(&mut out, c.threads)?;
            assert_eq!(out, data, "baseline correctness before timing");
            let en = measure(&c, || {
                let f = Encoded::compress(black_box(&data), codec, c.block_size, c.threads)?;
                black_box(f);
                Ok(())
            })?;
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                &codec.name(),
                "cpu_allocating_encode",
                "compress",
                data.len(),
                compressed.encoded_size(),
                &en,
                &lz,
                &zs,
                "",
            )?;
            let de = measure(&c, || {
                compressed.decode_into(black_box(&mut out), c.threads)?;
                black_box(&out);
                Ok(())
            })?;
            assert_eq!(out, data, "baseline correctness after timing");
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                &codec.name(),
                "cpu_preallocated_decode",
                "decompress",
                data.len(),
                compressed.encoded_size(),
                &de,
                &lz,
                &zs,
                "",
            )?;
            let ratio = compressed.encoded_size() as f64 / data.len() as f64;
            let enc_gbps = data.len() as f64 / median(&en);
            let dec_gbps = data.len() as f64 / median(&de);
            match codec {
                Codec::Lz4 { acceleration: 1 } => {
                    ds_summary.lz4_ratio = ratio;
                    ds_summary.lz4_enc = enc_gbps;
                    ds_summary.lz4_dec = dec_gbps;
                }
                Codec::Zstd { level: 1 } => {
                    ds_summary.zstd1_ratio = ratio;
                    ds_summary.zstd1_enc = enc_gbps;
                    ds_summary.zstd1_dec = dec_gbps;
                }
                Codec::Zstd { level: 3 } => {
                    ds_summary.zstd3_ratio = ratio;
                    ds_summary.zstd3_enc = enc_gbps;
                    ds_summary.zstd3_dec = dec_gbps;
                }
                _ => {}
            }
        }
        summaries.push(ds_summary);
        if let Some(device) = gpu.as_mut() {
            let options = GpuOptions {
                block_size: c.block_size,
                checksum: false,
                max_output: data.len(),
                ..GpuOptions::default()
            };
            let (compressed, _) = device.compress(&data, &options)?;
            assert_eq!(
                vxg::decompress(&compressed, &dec_opt)?,
                data,
                "GPU encode / CPU decode correctness"
            );
            assert_eq!(
                device.decompress(&frame, &options)?.0,
                data,
                "CPU encode / GPU decode correctness"
            );
            let device_name = device.device_name().to_owned();
            let en = measure(&c, || {
                black_box(device.compress(black_box(&data), &options)?);
                Ok(())
            })?;
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                "vxg-gpu",
                "gpu_host_e2e_allocating",
                "compress",
                data.len(),
                compressed.len(),
                &en,
                &lz,
                &zs,
                &device_name,
            )?;
            let de = measure(&c, || {
                black_box(device.decompress(black_box(&compressed), &options)?);
                Ok(())
            })?;
            row(
                &mut csv,
                &mut samples_csv,
                &c,
                &name,
                "vxg-gpu",
                "gpu_host_e2e_allocating",
                "decompress",
                data.len(),
                compressed.len(),
                &de,
                &lz,
                &zs,
                &device_name,
            )?;
            let (decoded, report) = device.decompress(&compressed, &options)?;
            assert_eq!(decoded, data);
            eprintln!("GPU device timestamp observation (not median, not e2e): {} kernel_ms={:?}, host_e2e_ms={:.3}",name,report.kernel_ms,report.end_to_end_ms);
        }
        csv.flush()?;
    }
    eprintln!("CSV: {}", c.csv);
    if let Some(md_path) = &c.markdown {
        write_markdown_report(md_path, &c, engine.kind(), &lz, &zs, &summaries)?;
        eprintln!("Markdown: {md_path}");
    }
    Ok(())
}
fn run_microbench(c: &Config, engine: vxg::simd::Engine) -> R<()> {
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
fn measure<F>(c: &Config, mut f: F) -> R<Vec<f64>>
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
fn median(s: &[f64]) -> f64 {
    let mid = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[mid - 1] + s[mid]) * 0.5
    } else {
        s[mid]
    }
}
#[allow(clippy::too_many_arguments)]
fn row(
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
fn rng(s: &mut u64) -> u8 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s as u8
}
fn datasets(n: usize) -> Vec<(String, Vec<u8>)> {
    let mut seed = 0x123456789abcdefu64;
    let random: Vec<u8> = (0..n).map(|_| rng(&mut seed)).collect();
    let low: Vec<u8> = (0..n).map(|_| rng(&mut seed) & 15).collect();
    let period: Vec<u8> = (0..n).map(|i| ((i % 37) * 7 + 3) as u8).collect();
    let mut json = Vec::with_capacity(n);
    let mut i = 0;
    while json.len() < n {
        json.extend_from_slice(format!("{{\"tenant\":{},\"event\":\"checklist.completed\",\"item\":{},\"ok\":true,\"actor\":\"user-{}\"}}\n",i%31,i,i%103).as_bytes());
        i += 1;
    }
    json.truncate(n);
    let mixed: Vec<u8> = (0..n)
        .map(|i| match (i / 4096) % 4 {
            0 => 0,
            1 => random[i],
            2 => json[i],
            _ => period[i],
        })
        .collect();
    vec![
        ("zeros".into(), vec![0; n]),
        ("period-37".into(), period),
        ("json-records".into(), json),
        ("random".into(), random),
        ("low-entropy-random".into(), low),
        ("mixed-4KiB".into(), mixed),
    ]
}

fn write_markdown_report(
    path: &str,
    c: &Config,
    simd: Simd,
    lz: &str,
    zs: &str,
    summaries: &[DatasetSummary],
) -> R<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut f = fs::File::create(path)?;
    writeln!(
        f,
        "## VXG v4 Benchmark Summary (`{}` / `{}`, SIMD=`{:?}`, threads=`{}`, block=`{}`)",
        env::consts::OS,
        env::consts::ARCH,
        simd,
        c.threads,
        c.block_size
    )?;
    writeln!(
        f,
        "\nBaselines: `liblz4 {lz}` (`lz4-c-a1`), `libzstd {zs}` (`zstd-c-l1`, `zstd-c-l3`). All codecs run on independent `{}`-byte blocks with RAW fallback and checksums OFF.\n",
        c.block_size
    )?;
    writeln!(
        f,
        "| Dataset | Size | VXG v4 Ratio | LZ4 Ratio | Zstd-1 Ratio | VXG v4 Encode | LZ4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |"
    )?;
    writeln!(
        f,
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )?;
    for s in summaries {
        let size_str = if s.raw_bytes >= (1 << 20) && s.raw_bytes % (1 << 20) == 0 {
            format!("{} MiB", s.raw_bytes >> 20)
        } else {
            format!("{:.1} MB", s.raw_bytes as f64 / 1e6)
        };
        writeln!(
            f,
            "| **`{}`** | {} | **{:.4}** | {:.4} | {:.4} | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s | **{:.3} GB/s** | {:.3} GB/s | {:.3} GB/s |",
            s.name,
            size_str,
            s.vxg_ratio,
            s.lz4_ratio,
            s.zstd1_ratio,
            s.vxg_enc,
            s.lz4_enc,
            s.zstd1_enc,
            s.vxg_dec,
            s.lz4_dec,
            s.zstd1_dec
        )?;
    }
    f.flush()?;
    Ok(())
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn fill_repeated(chunk: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let take = (n - out.len()).min(chunk.len());
        out.extend_from_slice(&chunk[..take]);
    }
    out
}

fn datasets_v4(n: usize) -> Vec<(String, Vec<u8>)> {
    // 1. telemetry.csv
    let mut csv_chunk = Vec::new();
    for row in 0..4096u32 {
        let epoch = row / 128;
        let prefix = 0x3B7F_2C4Du32.wrapping_add(epoch.wrapping_mul(0x2545_F491));
        let line = format!(
            "\"2026-01-{:02}T{:02}:{:02}:00Z\",{row},\"session.open\",\"eu-central-1\",\"r-{prefix:08x}{:08x}\",\"worker-{}\"\n",
            (row % 28) + 1,
            (row / 60) % 24,
            row % 60,
            row % 128,
            row % 16
        );
        csv_chunk.extend_from_slice(line.as_bytes());
    }
    let csv_data = fill_repeated(&csv_chunk, n);

    // 2. events.tsv
    let mut tsv_chunk = Vec::new();
    for row in 0..4096u32 {
        let line = format!(
            "{row}\t{}\t\"queue depth exceeded; retry scheduled; {}\"\t\"worker-{}\"\t\"ok\"\n",
            100_000 + row * 5,
            row % 128,
            row % 8
        );
        tsv_chunk.extend_from_slice(line.as_bytes());
    }
    let tsv_data = fill_repeated(&tsv_chunk, n);

    // 3. app-logs.logfmt
    let mut logfmt_chunk = Vec::new();
    for row in 0..4096u32 {
        let epoch = row / 128;
        let prefix = 0x3B7F_2C4Du32.wrapping_add(epoch.wrapping_mul(0x2545_F491));
        let line = format!(
            "ts=\"2026-01-{:02}T{:02}:{:02}:00Z\" level=info seq={row} req_id=\"r-{prefix:08x}{:08x}\" region=\"eu-1\" worker=\"worker-{}\"\n",
            (row % 28) + 1,
            (row / 60) % 24,
            row % 60,
            row % 128,
            row % 12
        );
        logfmt_chunk.extend_from_slice(line.as_bytes());
    }
    let logfmt_data = fill_repeated(&logfmt_chunk, n);

    // 4. json-records.ndjson
    let mut json_rec = Vec::with_capacity(n);
    let mut idx = 0u64;
    while json_rec.len() < n {
        let s = format!(
            "{{\"tenant\":{},\"event\":\"checklist.completed\",\"item\":{idx},\"ok\":true,\"actor\":\"user-{}\"}}\n",
            idx % 31,
            idx % 103
        );
        let b = s.as_bytes();
        let take = (n - json_rec.len()).min(b.len());
        json_rec.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 5. field-order-optional-escaping.ndjson
    let mut field_order = Vec::with_capacity(n);
    let seed_fo = 0x8D12_4A77_319B_05E1u64;
    idx = 0;
    while field_order.len() < n {
        let state = xorshift64(seed_fo ^ idx);
        let tenant = state % 97;
        let actor = (state >> 9) % 1009;
        let escaped = if idx % 5 == 0 {
            "tab\\tquote\\\"slash\\\\line\\n"
        } else {
            "plain"
        };
        let optional = if idx % 3 != 0 {
            format!(",\"retry\":{},\"note\":\"{escaped}\"", (state >> 17) % 4)
        } else {
            String::new()
        };
        let rec = if idx & 1 != 0 {
            format!("{{\"actor\":\"user-{actor}\",\"tenant\":{tenant}{optional},\"event\":\"task.update\",\"id\":{idx}}}\n")
        } else {
            format!("{{\"event\":\"task.update\",\"id\":{idx},\"tenant\":{tenant}{optional},\"actor\":\"user-{actor}\"}}\n")
        };
        let b = rec.as_bytes();
        let take = (n - field_order.len()).min(b.len());
        field_order.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 6. event-mix-identifiers.ndjson
    let mut event_mix = Vec::with_capacity(n);
    let seed_em = 0x3B7F_2C4D_9A18_56E7u64;
    let events = ["session.open", "session.close", "build.finished", "alert.raised"];
    let regions = ["eu-1", "us-2", "ap-1"];
    idx = 0;
    while event_mix.len() < n {
        let state = xorshift64(seed_em ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let event = events[(state & 3) as usize];
        let region = regions[((state >> 3) % 3) as usize];
        let status = if state & 0x40 != 0 { "ok" } else { "warning" };
        let epoch = idx / 128;
        let prefix = (xorshift64(seed_em ^ epoch.wrapping_mul(0x2545_F491_4F6C_DD1D)) & 0xFFFF_FFFF) as u32;
        let sequence = if state % 32 == 0 {
            ((state >> 16) & 0xFFFF_FFFF) as u32
        } else {
            (idx % 128) as u32
        };
        let rec = format!(
            "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"event\":\"{event}\",\"region\":\"{region}\",\"request_id\":\"r-{prefix:08x}{sequence:08x}\",\"status\":\"{status}\",\"sequence\":{idx}}}\n",
            (idx % 28) + 1,
            idx % 24
        );
        let b = rec.as_bytes();
        let take = (n - event_mix.len()).min(b.len());
        event_mix.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 7. bursty-variable-records.ndjson
    let mut bursty = Vec::with_capacity(n);
    let seed_br = 0x6D29_E4B8_1F73_A350u64;
    const PHRASES: [&str; 12] = [
        "queue depth exceeded",
        "retry scheduled",
        "batch committed",
        "cache flush complete",
        "worker heartbeat ok",
        "shard rebalanced",
        "index merge done",
        "page evicted",
        "commit retried",
        "drop detected",
        "sync paused",
        "threshold crossed",
    ];
    const VOCAB: [&str; 16] = [
        "flush", "queue", "retry", "timeout", "worker", "ack", "batch", "sync", "drop", "merge",
        "cache", "page", "index", "shard", "commit", "warn",
    ];
    idx = 0;
    while bursty.len() < n {
        let state = xorshift64(seed_br ^ idx.wrapping_mul(0xD134_2543_DE82_EF95));
        let burst = (idx / 19) % 7;
        let msg_len = (8 + (state % if burst < 5 { 24 } else { 240 })) as usize;
        let tags = if burst <= 1 { "[\"hot\",\"retry\"]" } else { "[]" };
        let mut msg = String::with_capacity(msg_len);
        let mut pos = 0usize;
        loop {
            let phrase = PHRASES[((state >> ((pos * 7) % 57)) as usize) % PHRASES.len()];
            let extra = phrase.len() + if msg.is_empty() { 0 } else { 2 };
            if msg.len() + extra > msg_len {
                break;
            }
            if !msg.is_empty() {
                msg.push_str("; ");
            }
            msg.push_str(phrase);
            pos += 1;
        }
        pos = 0;
        loop {
            let tok_buf;
            let token: &str = if ((state >> (13 + (pos % 5))) & 3) == 0 {
                tok_buf = format!("{}", state % 4096);
                &tok_buf
            } else {
                VOCAB[((state >> ((pos * 5) % 59)) & 15) as usize]
            };
            let extra = token.len() + if msg.is_empty() { 0 } else { 2 };
            if msg.len() + extra > msg_len {
                break;
            }
            if !msg.is_empty() {
                msg.push_str("; ");
            }
            msg.push_str(token);
            pos += 1;
        }
        while msg.len() < msg_len {
            msg.push(' ');
        }
        let rec = format!(
            "{{\"burst\":{burst},\"record\":{idx},\"tags\":{tags},\"message\":\"{msg}\",\"source\":\"worker-{}\"}}\n",
            state % 23
        );
        let b = rec.as_bytes();
        let take = (n - bursty.len()).min(b.len());
        bursty.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 8. source-and-docs.txt
    let mut src_seed = Vec::new();
    for slice in [
        include_bytes!("../analysis.rs") as &[u8],
        include_bytes!("../block.rs"),
        include_bytes!("../frame.rs"),
        include_bytes!("../lib.rs"),
        include_bytes!("../simd.rs"),
        include_bytes!("../../docs/FORMAT.md"),
        include_bytes!("../../docs/BENCHMARKS.md"),
    ] {
        src_seed.extend_from_slice(slice);
        src_seed.push(b'\n');
    }
    let src_docs = fill_repeated(&src_seed, n);

    // 9. ui-render-32bpp.bmp
    let mut bmp32 = vec![0u8; n.max(65536)];
    bmp32[0..2].copy_from_slice(b"BM");
    let bmp_len = n as u32;
    bmp32[2..6].copy_from_slice(&bmp_len.to_le_bytes());
    bmp32[10..14].copy_from_slice(&54u32.to_le_bytes());
    bmp32[14..18].copy_from_slice(&40u32.to_le_bytes());
    bmp32[18..22].copy_from_slice(&2048i32.to_le_bytes());
    bmp32[22..26].copy_from_slice(&2048i32.to_le_bytes());
    bmp32[26..28].copy_from_slice(&1u16.to_le_bytes());
    bmp32[28..30].copy_from_slice(&32u16.to_le_bytes());
    let mut tile32 = [0u8; 65536];
    for px in 0..16384usize {
        let band = ((px / 256) & 0xFF) as u8;
        let edge = if px % 256 == 0 {
            ((px / 17) & 0xFF) as u8
        } else {
            0
        };
        let base = px * 4;
        tile32[base] = band.wrapping_mul(13).wrapping_add(edge);
        tile32[base + 1] = band.wrapping_mul(29);
        tile32[base + 2] = band.wrapping_mul(47).wrapping_sub(edge);
        tile32[base + 3] = 255;
    }
    bmp32[54..65536].copy_from_slice(&tile32[54..65536]);
    let mut off = 65536;
    while off < bmp32.len() {
        let end = (off + 65536).min(bmp32.len());
        bmp32[off..end].copy_from_slice(&tile32[..end - off]);
        off += 65536;
    }
    bmp32.truncate(n);

    // 10. rgb-frame-24bpp.bmp
    let mut bmp24 = vec![0u8; n.max(65536)];
    bmp24[0..2].copy_from_slice(b"BM");
    bmp24[2..6].copy_from_slice(&bmp_len.to_le_bytes());
    bmp24[10..14].copy_from_slice(&54u32.to_le_bytes());
    bmp24[14..18].copy_from_slice(&40u32.to_le_bytes());
    bmp24[18..22].copy_from_slice(&2360i32.to_le_bytes());
    bmp24[22..26].copy_from_slice(&2360i32.to_le_bytes());
    bmp24[26..28].copy_from_slice(&1u16.to_le_bytes());
    bmp24[28..30].copy_from_slice(&24u16.to_le_bytes());
    let mut tile24 = [0u8; 65536];
    for px in 0..21845usize {
        let run = ((px / 256) & 0xFF) as u8;
        let edge = u8::from(px % 256 == 127);
        let base = px * 3;
        tile24[base] = run.wrapping_mul(7);
        tile24[base + 1] = run.wrapping_mul(19).wrapping_add(edge);
        tile24[base + 2] = run.wrapping_mul(37);
    }
    bmp24[54..65536].copy_from_slice(&tile24[54..65536]);
    off = 65536;
    while off < bmp24.len() {
        let end = (off + 65536).min(bmp24.len());
        bmp24[off..end].copy_from_slice(&tile24[..end - off]);
        off += 65536;
    }
    bmp24.truncate(n);

    vec![
        ("telemetry.csv".into(), csv_data),
        ("events.tsv".into(), tsv_data),
        ("app-logs.logfmt".into(), logfmt_data),
        ("json-records.ndjson".into(), json_rec),
        ("field-order-optional-escaping.ndjson".into(), field_order),
        ("event-mix-identifiers.ndjson".into(), event_mix),
        ("bursty-variable-records.ndjson".into(), bursty),
        ("source-and-docs.txt".into(), src_docs),
        ("ui-render-32bpp.bmp".into(), bmp32),
        ("rgb-frame-24bpp.bmp".into(), bmp24),
    ]
}
