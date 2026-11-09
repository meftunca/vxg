use std::{
    env,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::Path,
    process::ExitCode,
};
use vxg::{
    gpu::{Backend, GpuCodec, GpuOptions},
    DecodeOptions, Frame, Options, Simd,
};
type R<T> = Result<T, Box<dyn std::error::Error>>;
const VERSION_STR: &str = concat!(
    "vxg ",
    env!("CARGO_PKG_VERSION"),
    " (format v4/v5, default block 65536)"
);
const HELP: &str = r#"VXG — adaptive independent-block lossless codec & compressed SQL engine (format v4/v5)

Usage:
  vxg c|compress   INPUT OUTPUT [--backend cpu|metal|vulkan|cuda] [--threads N|auto]
                                [--block-size BYTES] [--level|-l 1..9] [--acceleration 1..64]
                                [--simd auto|scalar|sse2|avx2|avx512|neon]
                                [--record-aligned] [--no-checksum] [--legacy] [--stream]
                                [--no-mmap] [--gpu-batch BYTES] [--force|-f] [--quiet|-q]
  vxg d|decompress INPUT OUTPUT [--backend cpu|metal|vulkan|cuda] [--threads N|auto]
                                [--simd MODE] [--max-output BYTES] [--recover]
                                [--no-checksum|--no-verify-checksum] [--stream] [--no-mmap]
                                [--gpu-batch BYTES] [--force|-f] [--quiet|-q]
  vxg recover      INPUT OUTPUT [--simd MODE] [--max-output BYTES] [--no-checksum]
                                [--no-mmap] [--force|-f] [--quiet|-q]
  vxg t|verify     FRAME        [--threads N|auto] [--simd MODE] [--max-output BYTES]
                                [--no-checksum|--no-verify-checksum] [--stream] [--no-mmap] [-q]
  vxg block        FRAME INDEX [OUTPUT] [--simd MODE] [--max-output BYTES]
                                [--no-checksum|--no-verify-checksum] [--no-mmap] [--force|-f]
  vxg q|query      FRAME        [--count | --row IDX | --range START:END |
                                 --find-first STR | --filter STR | --where EXPR |
                                 --select COL1,COL2,...] [--order-by COL [--desc|--asc]]
                                [--limit N] [--stream] [--no-mmap] [-t N|auto]
  vxg agg          FRAME        [--group-by GCOL] [--col ACOL] [--where EXPR]
                                [--order-by key|rows|count|sum|avg|min|max [--desc|--asc]]
                                [--limit N] [--json] [--stream] [--no-mmap] [-t N|auto]
  vxg sql          FRAME "SELECT ..." [--json|--csv|--arrow] [--no-header] [--stream] [--no-mmap] [-t N|auto]
  vxg arrow        FRAME OUTPUT [--sql "SELECT ..." | [--cols C1,C2] [--where EXPR] [--limit N]]
                                [-t N|auto] [--no-mmap] [-f] [-q]
  vxg project      INPUT OUTPUT --cols COL1,COL2,... [--where EXPR] [--stream] [--no-mmap] [-t N|auto] [-f] [-q]
  vxg filter-frame INPUT OUTPUT --where EXPR [--block-size BYTES] [--no-checksum] [--stream] [--no-mmap] [-f] [-q]
  vxg update       INPUT OUTPUT --where SPEC --col COL (--set VAL|--set-raw RAW|--upper|--lower|--redact|--add N) [--stream] [--no-mmap] [-f] [-q]
  vxg map          INPUT OUTPUT --col COL (--set VAL|--set-raw RAW|--upper|--lower|--redact|--add N) [--stream] [--no-mmap] [-f] [-q]
  vxg info|list    FRAME        [--summary] [--no-mmap]
  vxg hardware
  vxg gpu-check    metal|vulkan|cuda
  vxg --version|-V
  vxg --help|-h

Notes:
  Regular files are memory-mapped (zero-copy mmap) by default; pass --no-mmap to read into heap RAM.
  Pass --stream (or '-' stdin) for O(single_frame) bounded-memory multi-GB streaming execution.
  Compression output is a new file unless --force (-f) is explicitly supplied.
  GPU features require native SDKs at build time. No silent CPU fallback.
"#;

fn val<'a>(args: &'a [String], i: &mut usize) -> R<&'a str> {
    *i += 1;
    Ok(args.get(*i).ok_or("missing option value")?.as_str())
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

fn read_input_bytes(path: &str, allow_mmap: bool) -> R<vxg::InputBytes> {
    if path == "-" {
        let mut buf = Vec::new();
        io::stdin().lock().read_to_end(&mut buf)?;
        Ok(vxg::InputBytes::Owned(buf))
    } else {
        Ok(vxg::InputBytes::open_path_mode(path, allow_mmap)?)
    }
}

fn write_output(path: &str, bytes: &[u8], force: bool) -> R<()> {
    if path == "-" {
        let mut out = io::stdout().lock();
        out.write_all(bytes)?;
        out.flush()?;
        return Ok(());
    }
    let mut options = OpenOptions::new();
    options.write(true);
    if force {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    let mut f = options.open(path)?;
    f.write_all(bytes)?;
    f.flush()?;
    Ok(())
}

fn main() -> ExitCode {
    match run_args(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_args(args: &[String]) -> R<()> {
    let Some(cmd) = args.first().map(String::as_str) else {
        print!("{HELP}");
        return Ok(());
    };
    match cmd {
        "--help" | "-h" | "help" => {
            print!("{HELP}");
            return Ok(());
        }
        "--version" | "-V" | "version" => {
            println!("{VERSION_STR}");
            return Ok(());
        }
        "hardware" => {
            println!(
                "os={} arch={} available_parallelism={}",
                env::consts::OS,
                env::consts::ARCH,
                std::thread::available_parallelism()
                    .map(|x| x.get())
                    .unwrap_or(1)
            );
            println!("auto SIMD={:?}", vxg::simd::Engine::new(Simd::Auto)?.kind());
            for s in [
                Simd::Scalar,
                Simd::Sse2,
                Simd::Avx2,
                Simd::Avx512,
                Simd::Neon,
            ] {
                println!("{s:?}: {}", vxg::simd::Engine::supported(s));
            }
            println!(
                "compiled native features: metal={} vulkan={} cuda={}",
                cfg!(feature = "metal"),
                cfg!(feature = "vulkan"),
                cfg!(feature = "cuda")
            );
            return Ok(());
        }
        "info" | "list" => {
            if args.len() < 2 || args.len() > 4 {
                return Err("usage: vxg info FRAME [--summary] [--no-mmap]".into());
            }
            let mut summary_only = false;
            let mut use_mmap = true;
            for opt in &args[2..] {
                match opt.as_str() {
                    "--summary" => summary_only = true,
                    "--no-mmap" => use_mmap = false,
                    x => return Err(format!("unknown option {x}").into()),
                }
            }
            let bytes = read_input_bytes(&args[1], use_mmap)?;
            let frames = Frame::parse_all(&bytes, usize::MAX)?;
            let mut raw_blocks = 0usize;
            let mut lz_blocks = 0usize;
            let mut col_blocks = 0usize;
            let mut dict_lz_blocks = 0usize;
            let mut total_raw = 0usize;
            let mut total_blocks = 0usize;
            for f in &frames {
                if f.blocks().all(|b| b.kind != 2) {
                    f.validate_commands()?;
                }
                total_raw += f.raw_size();
                total_blocks += f.block_count();
                for b in f.blocks() {
                    match b.kind {
                        0 => raw_blocks += 1,
                        1 => lz_blocks += 1,
                        2 => col_blocks += 1,
                        3 => dict_lz_blocks += 1,
                        _ => {}
                    }
                }
            }
            let f0 = &frames[0];
            let ratio = if total_raw == 0 {
                1.0
            } else {
                bytes.len() as f64 / total_raw as f64
            };
            println!(
                "version={} raw={} encoded={} ratio={:.4} blocks={} block_size={} checksum={} frames={} kinds=RAW:{},LZ:{},COLUMN:{},DICT_LZ:{}",
                f0.version(),
                total_raw,
                bytes.len(),
                ratio,
                total_blocks,
                f0.block_size(),
                f0.checksummed(),
                frames.len(),
                raw_blocks,
                lz_blocks,
                col_blocks,
                dict_lz_blocks
            );
            if !summary_only && frames.len() == 1 {
                for (i, b) in f0.blocks().enumerate() {
                    println!(
                        "block={i} raw_offset={} raw={} payload_offset={} encoded={} kind={}",
                        b.raw_offset,
                        b.raw_size,
                        b.payload_offset,
                        b.payload_size,
                        match b.kind {
                            0 => "RAW",
                            1 => "LZ",
                            2 => "COLUMN",
                            3 => "DICT_LZ",
                            _ => "UNKNOWN",
                        }
                    );
                }
            }
            return Ok(());
        }
        "t" | "test" | "verify" => {
            if args.len() < 2 {
                return Err("usage: vxg verify FRAME [--threads N|auto] [--simd MODE] [--max-output BYTES] [--no-checksum] [--stream] [--no-mmap] [--quiet]".into());
            }
            let input = &args[1];
            let mut dec = DecodeOptions::default();
            let mut explicit_max_out = false;
            let mut stream_mode = false;
            let mut use_mmap = true;
            let mut quiet = false;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--threads" | "-t" => dec.threads = parse_threads(val(args, &mut i)?)?,
                    "--simd" => dec.simd = val(args, &mut i)?.parse()?,
                    "--max-output" => {
                        dec.max_output = val(args, &mut i)?.parse()?;
                        explicit_max_out = true;
                    }
                    "--no-checksum" | "--no-verify-checksum" => dec.verify_checksum = false,
                    "--stream" => stream_mode = true,
                    "--no-mmap" => use_mmap = false,
                    "--quiet" | "-q" => quiet = true,
                    x => return Err(format!("unknown option {x}").into()),
                }
                i += 1;
            }
            if !explicit_max_out {
                dec.max_output = usize::MAX;
            }
            let start = std::time::Instant::now();
            if stream_mode || input == "-" {
                let mut rdr = open_stream_reader(input)?;
                let (raw_len, enc_len) = vxg::verify_stream(&mut rdr, &dec)?;
                let elapsed = start.elapsed().as_secs_f64();
                if !quiet {
                    let gbps = raw_len as f64 / elapsed.max(1e-9) / 1e9;
                    eprintln!(
                        "OK: stream raw={raw_len} encoded={enc_len} checksum_verified={} ({:.3} ms, {:.3} GB/s)",
                        dec.verify_checksum,
                        elapsed * 1000.0,
                        gbps
                    );
                }
                return Ok(());
            }
            let bytes = read_input_bytes(input, use_mmap)?;
            if let vxg::InputBytes::Mapped(ref m) = bytes {
                m.advise_sequential();
            }
            let frames = Frame::parse_all(&bytes, dec.max_output)?;
            let raw_len = vxg::verify_frame(&bytes, &dec)?;
            let elapsed = start.elapsed().as_secs_f64();
            if !quiet {
                let gbps = raw_len as f64 / elapsed.max(1e-9) / 1e9;
                let total_blocks: usize = frames.iter().map(Frame::block_count).sum();
                let all_csum = frames.iter().all(Frame::checksummed);
                eprintln!(
                    "OK: version={} raw={} encoded={} blocks={} checksum_verified={} ({:.3} ms, {:.3} GB/s)",
                    frames[0].version(),
                    raw_len,
                    bytes.len(),
                    total_blocks,
                    all_csum && dec.verify_checksum,
                    elapsed * 1000.0,
                    gbps
                );
            }
            return Ok(());
        }
        "block" | "extract-block" => {
            if args.len() < 3 {
                return Err("usage: vxg block FRAME INDEX [OUTPUT] [--simd MODE] [--max-output BYTES] [--no-checksum] [--no-mmap] [--force]".into());
            }
            let input = &args[1];
            let block_idx: usize = args[2].parse()?;
            let mut output = "-";
            let mut i = 3;
            if i < args.len() && !args[i].starts_with('-') || (i < args.len() && args[i] == "-") {
                output = &args[i];
                i += 1;
            }
            let mut dec = DecodeOptions::default();
            let mut use_mmap = true;
            let mut force = false;
            while i < args.len() {
                match args[i].as_str() {
                    "--simd" => dec.simd = val(args, &mut i)?.parse()?,
                    "--max-output" => dec.max_output = val(args, &mut i)?.parse()?,
                    "--no-checksum" | "--no-verify-checksum" => dec.verify_checksum = false,
                    "--no-mmap" => use_mmap = false,
                    "--force" | "-f" => force = true,
                    x => return Err(format!("unknown option {x}").into()),
                }
                i += 1;
            }
            if output != "-" && !force && Path::new(output).exists() {
                return Err("output exists (use --force to replace it)".into());
            }
            let bytes = read_input_bytes(input, use_mmap)?;
            if let vxg::InputBytes::Mapped(ref m) = bytes {
                m.advise_random();
            }
            let f = Frame::parse(&bytes, usize::MAX)?;
            let out = f.decode_block(block_idx, &dec)?;
            write_output(output, &out, force)?;
            return Ok(());
        }
        "gpu-check" => {
            if args.len() != 2 {
                return Err("usage: vxg gpu-check metal|vulkan|cuda".into());
            }
            return gpu_check(args[1].parse()?);
        }
        "q" | "query" => return run_query_cmd(args),
        "agg" | "group-by" => return run_agg_cmd(args),
        "sql" => return run_sql_cmd(args),
        "arrow" => return run_arrow_cmd(args),
        "recover" => return run_recover_cmd(args),
        "filter-frame" | "project" | "update" | "map" => return run_ops_cmd(args),
        "c" | "compress" => {}
        "d" | "decompress" => {
            if args.iter().skip(3).any(|a| a == "--recover") {
                return run_recover_cmd(args);
            }
        }
        _ => return Err(format!("unknown command: {cmd}\n{HELP}").into()),
    }
    if args.len() < 3 {
        return Err(HELP.into());
    }
    let is_compress = matches!(cmd, "c" | "compress");
    let input = &args[1];
    let output = &args[2];
    if input != "-" && input == output {
        return Err("input and output must differ".into());
    }
    let mut enc = Options::default();
    let mut dec = DecodeOptions::default();
    let mut gpu = GpuOptions::default();
    let mut backend = "cpu".to_owned();
    let mut stream_mode = false;
    let mut use_mmap = true;
    let mut explicit_max_out = false;
    let mut force = false;
    let mut quiet = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => backend = val(args, &mut i)?.into(),
            "--stream" => stream_mode = true,
            "--no-mmap" => use_mmap = false,
            "--threads" | "-t" => {
                let n = parse_threads(val(args, &mut i)?)?;
                enc.threads = n;
                dec.threads = n;
            }
            "--block-size" => {
                let n = val(args, &mut i)?.parse()?;
                enc.block_size = n;
                gpu.block_size = n;
            }
            "--acceleration" => enc.acceleration = val(args, &mut i)?.parse()?,
            "--level" | "-l" => enc.level = val(args, &mut i)?.parse()?,
            "--simd" => {
                let s = val(args, &mut i)?.parse()?;
                enc.simd = s;
                dec.simd = s;
            }
            "--no-checksum" | "--no-verify-checksum" => {
                enc.checksum = false;
                dec.verify_checksum = false;
                gpu.checksum = false;
            }
            "--v3" | "--v4" | "--v5" => {} // unified adaptive v4/v5 pipeline is the default
            "--record-aligned" => enc.record_aligned = true,
            "--target-rows" => enc.target_block_rows = val(args, &mut i)?.parse()?,
            "--legacy" => enc.legacy_v2 = true,
            "--max-output" => {
                let n = val(args, &mut i)?.parse()?;
                dec.max_output = n;
                gpu.max_output = n;
                explicit_max_out = true;
            }
            "--gpu-batch" => gpu.batch_bytes = val(args, &mut i)?.parse()?,
            "--force" | "-f" => force = true,
            "--quiet" | "-q" => quiet = true,
            x => return Err(format!("unknown option {x}").into()),
        }
        i += 1;
    }
    if output != "-" && !force && Path::new(output).exists() {
        return Err("output exists (use --force to replace it)".into());
    }
    if stream_mode && backend == "cpu" {
        if !explicit_max_out {
            dec.max_output = usize::MAX;
        }
        let start = std::time::Instant::now();
        let existed = output != "-" && Path::new(output).exists();
        let mut r_box = open_stream_reader(input)?;
        let mut w_box = open_stream_writer(output, force)?;
        let res: R<(u64, u64)> = (|| {
            let pair = if is_compress {
                vxg::compress_stream(&mut r_box, &mut w_box, &enc)?
            } else {
                let written = vxg::decompress_stream(&mut r_box, &mut w_box, &dec)?;
                (0, written)
            };
            w_box.flush()?;
            Ok(pair)
        })();
        let (in_b, out_b) = match res {
            Ok(pair) => pair,
            Err(e) => {
                drop(w_box);
                if output != "-" && !existed {
                    let _ = fs::remove_file(output);
                }
                return Err(e);
            }
        };
        if !quiet {
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!("stream completed: in={in_b} out={out_b} bytes ({ms:.3} ms)");
        }
        return Ok(());
    }
    if backend == "cpu" && use_mmap && input != "-" && output != "-" && vxg::mmap_supported() {
        if !explicit_max_out {
            dec.max_output = usize::MAX;
        }
        let start = std::time::Instant::now();
        let (in_len, out_len) = if is_compress {
            vxg::compress_file(input, output, &enc, force)?
        } else {
            let in_len = fs::metadata(input)?.len();
            let raw_b = vxg::decompress_file(input, output, &dec, force)?;
            (in_len, raw_b)
        };
        if !quiet {
            let elapsed = start.elapsed().as_secs_f64();
            let raw_b = if is_compress { in_len } else { out_len };
            let enc_b = if is_compress { out_len } else { in_len };
            let ratio = if raw_b == 0 {
                1.0
            } else {
                enc_b as f64 / raw_b as f64
            };
            let gbps = raw_b as f64 / elapsed.max(1e-9) / 1e9;
            eprintln!(
                "{in_len} -> {out_len} bytes (ratio={ratio:.4}); {:.3} ms ({gbps:.3} GB/s zero-copy mmap)",
                elapsed * 1000.0
            );
        }
        return Ok(());
    }
    let bytes = read_input_bytes(input, use_mmap)?;
    let start = std::time::Instant::now();
    let result = if backend == "cpu" {
        if is_compress {
            vxg::compress(&bytes, &enc)?
        } else {
            vxg::decompress(&bytes, &dec)?
        }
    } else {
        if enc.acceleration != 1 || enc.simd != Simd::Auto || enc.threads != 1 {
            return Err("--acceleration, --simd and --threads are CPU-only tuning options".into());
        }
        let mut device = GpuCodec::new(backend.parse()?)?;
        let (result, report) = if is_compress {
            device.compress(&bytes, &gpu)?
        } else {
            device.decompress(&bytes, &gpu)?
        };
        if !quiet {
            eprintln!("device={} batches={} e2e_ms={:.3} kernel_ms={:?} (kernel time excludes transfers/host work)",report.device,report.batches,report.end_to_end_ms,report.kernel_ms);
        }
        result
    };
    let codec_elapsed = start.elapsed().as_secs_f64();
    write_output(output, &result, force)?;
    if !quiet {
        let raw_bytes = if is_compress {
            bytes.len()
        } else {
            result.len()
        };
        let enc_bytes = if is_compress {
            result.len()
        } else {
            bytes.len()
        };
        let ratio = if raw_bytes == 0 {
            1.0
        } else {
            enc_bytes as f64 / raw_bytes as f64
        };
        let gbps = raw_bytes as f64 / codec_elapsed.max(1e-9) / 1e9;
        eprintln!(
            "{} -> {} bytes (ratio={:.4}); {:.3} ms ({:.3} GB/s codec, {:.3} ms total with I/O)",
            bytes.len(),
            result.len(),
            ratio,
            codec_elapsed * 1000.0,
            gbps,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
include!("cli_support/ops.rs");
include!("cli_support/query.rs");
include!("cli_support/agg.rs");
include!("cli_support/sql.rs");
include!("cli_support/arrow_cli.rs");
include!("cli_support/recover.rs");
include!("cli_support/tests.rs");
