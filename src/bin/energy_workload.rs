//! One uninterrupted codec/operation for an external energy sampler.
//! This executable reports work and time, never fabricates a power reading.
use std::{
    env, fs,
    hint::black_box,
    io::{self, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use vxg::{
    baselines::{self, Codec, Encoded},
    DecodeOptions, Options, Simd,
};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn tick(event: &str, start: Instant, iterations: u64, bytes: usize) -> R<()> {
    let elapsed = start.elapsed().as_nanos();
    let unix = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    println!("{{\"event\":\"{event}\",\"unix_ns\":{unix},\"elapsed_ns\":{elapsed},\"iterations\":{iterations},\"raw_bytes_processed\":{}}}", iterations as u128 * bytes as u128);
    io::stdout().flush()?;
    Ok(())
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("vxg-energy-workload INPUT CODEC encode|decode SECONDS [SIMD] [--wait]\nCODEC: vxg, lz4-a1, lz4-a4, zstd-m3, zstd-1, zstd-3. One worker; 65536-byte independent blocks; system allocator; checksums off. --wait reads GO from stdin after setup/warmups. JSONL stdout contains metadata, progress and successful roundtrip completion. VXG decode includes frame parsing; reference decode uses prepared block metadata, as in vxg-bench.");
        return Ok(());
    }
    if !(4..=6).contains(&args.len()) {
        return Err("see vxg-energy-workload --help".into());
    }
    let codec = match args[1].as_str() {
        "vxg" => None,
        "lz4-a1" => Some(Codec::Lz4 { acceleration: 1 }),
        "lz4-a4" => Some(Codec::Lz4 { acceleration: 4 }),
        "zstd-m3" => Some(Codec::Zstd { level: -3 }),
        "zstd-1" => Some(Codec::Zstd { level: 1 }),
        "zstd-3" => Some(Codec::Zstd { level: 3 }),
        _ => return Err("unknown codec".into()),
    };
    let encode = match args[2].as_str() {
        "encode" => true,
        "decode" => false,
        _ => return Err("operation must be encode or decode".into()),
    };
    let seconds: f64 = args[3].parse()?;
    if !seconds.is_finite() || !(0.05..=3600.0).contains(&seconds) {
        return Err("seconds must be finite and in 0.05..=3600".into());
    }
    let simd: Simd = args.get(4).map(String::as_str).unwrap_or("auto").parse()?;
    let wait = match args.get(5).map(String::as_str) {
        None => false,
        Some("--wait") => true,
        _ => return Err("only --wait may follow SIMD".into()),
    };
    let data = fs::read(&args[0])?;
    if data.is_empty() {
        return Err("input must not be empty".into());
    }
    let enc_opt = Options {
        simd,
        checksum: false,
        ..Options::default()
    };
    let dec_opt = DecodeOptions {
        simd,
        max_output: data.len(),
        verify_checksum: false,
        ..DecodeOptions::default()
    };
    let mut out = vec![0u8; data.len()];
    let frame = if codec.is_none() {
        vxg::compress(&data, &enc_opt)?
    } else {
        Vec::new()
    };
    let reference = codec
        .map(|c| Encoded::compress(&data, c, 65536, 1))
        .transpose()?;
    let encoded_bytes = if let Some(r) = &reference {
        r.decode_into(&mut out, 1)?;
        r.encoded_size()
    } else {
        if vxg::compress_with_stats(&data, &enc_opt)?.0 != frame {
            return Err("timed API and diagnostic encoder disagree".into());
        }
        vxg::decompress_into(&frame, &mut out, &dec_opt)?;
        frame.len()
    };
    if out != data {
        return Err("roundtrip before measurement failed".into());
    }
    let mut operation = || -> R<()> {
        if encode {
            if let Some(c) = codec {
                black_box(Encoded::compress(black_box(&data), c, 65536, 1)?);
            } else {
                black_box(vxg::compress(black_box(&data), &enc_opt)?);
            }
        } else if let Some(r) = &reference {
            r.decode_into(black_box(&mut out), 1)?;
            black_box(&out);
        } else {
            vxg::decompress_into(black_box(&frame), black_box(&mut out), &dec_opt)?;
            black_box(&out);
        }
        Ok(())
    };
    operation()?;
    operation()?;
    let profile = if encode {
        "cpu_allocating_encode"
    } else if codec.is_none() {
        "cpu_full_frame_preallocated_decode"
    } else {
        "cpu_preallocated_decode"
    };
    let (lz4, zstd) = baselines::versions();
    println!("{{\"event\":\"ready\",\"codec\":\"{}\",\"operation\":\"{}\",\"profile\":\"{profile}\",\"raw_bytes\":{},\"encoded_bytes\":{encoded_bytes},\"ratio\":{},\"threads\":1,\"block_size\":65536,\"checksums\":false,\"allocator\":\"system\",\"simd\":\"{:?}\",\"lz4_version\":\"{lz4}\",\"zstd_version\":\"{zstd}\"}}",
        args[1], args[2], data.len(), encoded_bytes as f64 / data.len() as f64, vxg::simd::Engine::new(simd)?.kind());
    io::stdout().flush()?;
    if wait {
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        if line.trim() != "GO" {
            return Err("expected GO after readiness".into());
        }
    }
    let start = Instant::now();
    tick("start", start, 0, data.len())?;
    let mut iterations = 0;
    let mut next_tick = Duration::from_millis(100);
    while start.elapsed().as_secs_f64() < seconds {
        operation()?;
        iterations += 1;
        if start.elapsed() >= next_tick {
            tick("progress", start, iterations, data.len())?;
            next_tick = start.elapsed() + Duration::from_millis(100);
        }
    }
    tick("end", start, iterations, data.len())?;
    // Fresh encode + safe decode after timing also validates the encode path.
    if encode {
        if let Some(c) = codec {
            let final_frame = Encoded::compress(&data, c, 65536, 1)?;
            if final_frame.encoded_size() != encoded_bytes {
                return Err("encoded size changed".into());
            }
            final_frame.decode_into(&mut out, 1)?;
        } else {
            let final_frame = vxg::compress(&data, &enc_opt)?;
            if final_frame != frame {
                return Err("encoded bytes changed".into());
            }
            vxg::decompress_into(&final_frame, &mut out, &dec_opt)?;
        }
    }
    if out != data {
        return Err("roundtrip after measurement failed".into());
    }
    println!("{{\"event\":\"verified\",\"roundtrip\":true}}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("energy workload failed: {e}");
        std::process::exit(1);
    }
}
