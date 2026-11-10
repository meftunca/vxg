use std::{env, fs, hint::black_box, io::Write};
use vxg::{baselines, compress_with_stats, gpu::GpuCodec, DecodeOptions, Frame, Options};

#[path = "bench_support/config.rs"]
mod config;
#[path = "bench_support/datasets.rs"]
mod datasets;
#[path = "bench_support/harness.rs"]
mod harness;
#[path = "bench_support/metrics.rs"]
mod metrics;
#[path = "bench_support/report.rs"]
mod report;

use config::{CodecPoint, Config, DatasetSummary};
use datasets::{datasets, datasets_src_unique, datasets_v4};
use harness::{evaluate_baselines_and_gpu, measure, median, row, run_microbench};
use metrics::measure_op;
use report::write_markdown_report;

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

fn main() {
    if let Err(e) = run() {
        eprintln!("benchmark failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(c) = Config::from_args(&args)? else {
        return Ok(());
    };
    let engine = vxg::simd::Engine::new(c.simd)?;
    if c.microbench {
        return run_microbench(&c, engine);
    }
    #[cfg(feature = "alloc-stats")]
    eprintln!(
        "ALLOC-STATS enabled: allocation counters perturb timing; use counts only, not GB/s."
    );
    let (lz, zs) = baselines::versions();
    eprintln!(
        "VXG {} / liblz4 {lz} / libzstd {zs}; {} {}; SIMD={:?}; allocator={}; threads={}; block={}",
        env!("CARGO_PKG_VERSION"),
        env::consts::OS,
        env::consts::ARCH,
        engine.kind(),
        allocator_name(),
        c.threads,
        c.block_size
    );
    eprintln!("CPU profiles: cpu_allocating_encode; cpu_preallocated_decode = prepared directory + checked command decode; cpu_full_frame_preallocated_decode = parse + checked decode; cpu_prepared_validate_commands = command validation on a prepared directory; cpu_parse_validate_commands = parse + command validation; cpu_prevalidated_decode = prepared proof + repeated decode only. Per-call contexts included. RAW fallback for ALL codecs. Checksums OFF. Warm cache. GPU: separate host-e2e profile.");
    let mut datasets = if c.files.is_empty() {
        match c.suite.as_str() {
            "default" => datasets(c.size),
            "v4" => datasets_v4(c.size),
            "src-unique" => datasets_src_unique()?,
            "all" => {
                let mut v = datasets_v4(c.size);
                v.extend(datasets_src_unique()?);
                v.extend(datasets(c.size));
                v
            }
            other => {
                return Err(format!(
                    "unknown --suite {other} (expected default, v4, src-unique, or all)"
                )
                .into())
            }
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
            level: c.level,
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
            vxg: CodecPoint {
                ratio: frame.len() as f64 / data.len() as f64,
                ..CodecPoint::default()
            },
            ..DatasetSummary::default()
        };
        if !c.decode_only {
            let en = measure_op(&c, data.len(), || {
                #[cfg(feature = "encode-with-stats")]
                let f = vxg::compress_with_stats(black_box(&data), &enc_opt)?;
                #[cfg(not(feature = "encode-with-stats"))]
                let f = vxg::compress(black_box(&data), &enc_opt)?;
                black_box(f);
                Ok(())
            })?;
            ds_summary.vxg.enc = data.len() as f64 / median(&en.ns_samples);
            ds_summary.vxg.enc_j_gb = en.j_per_gb;
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
                &en.ns_samples,
                &lz,
                &zs,
                "",
            )?;
        }
        let safe_de = measure_op(&c, data.len(), || {
            parsed.decode_into(black_box(&mut out), &dec_opt)?;
            black_box(&out);
            Ok(())
        })?;
        assert_eq!(out, data, "VXG correctness after timing");
        ds_summary.vxg.dec = data.len() as f64 / median(&safe_de.ns_samples);
        ds_summary.vxg.dec_j_gb = safe_de.j_per_gb;
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
            &safe_de.ns_samples,
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
            let safe_ns = median(&safe_de.ns_samples);
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
        if !c.vxg_only && !c.decode_only {
            evaluate_baselines_and_gpu(
                &c,
                &name,
                &data,
                &frame,
                &dec_opt,
                &mut out,
                &mut ds_summary,
                &mut csv,
                &mut samples_csv,
                &mut gpu,
                &lz,
                &zs,
            )?;
        }
        summaries.push(ds_summary);
        csv.flush()?;
    }
    eprintln!("CSV: {}", c.csv);
    if let Some(md_path) = &c.markdown {
        write_markdown_report(md_path, &c, engine.kind(), &lz, &zs, &summaries)?;
        eprintln!("Markdown: {md_path}");
    }
    Ok(())
}
