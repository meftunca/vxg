use std::{env, fs, io::Write, path::Path};

use vxg::{analysis, compress, packing, Frame, Options, Simd};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn value<'a>(args: &'a [String], index: &mut usize) -> R<&'a str> {
    *index += 1;
    Ok(args.get(*index).ok_or("missing option value")?.as_str())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("packing evaluation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input = None;
    let mut output = None;
    let mut iterations = 1usize;
    let mut options = Options {
        checksum: false,
        ..Options::default()
    };
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => input = Some(value(&args, &mut index)?.to_owned()),
            "--output" => output = Some(value(&args, &mut index)?.to_owned()),
            "--iterations" => iterations = value(&args, &mut index)?.parse()?,
            "--block-size" => options.block_size = value(&args, &mut index)?.parse()?,
            "--threads" => options.threads = value(&args, &mut index)?.parse()?,
            "--acceleration" => options.acceleration = value(&args, &mut index)?.parse()?,
            "--simd" => options.simd = value(&args, &mut index)?.parse::<Simd>()?,
            "--help" | "-h" => {
                println!("vxg-pack-eval --input FILE [--output REPORT.json] [--iterations 1] [--block-size 65536] [--threads 1] [--acceleration 1] [--simd auto]\nCompresses once outside timing, then exactly packs/unpacks its existing command trace. Results are metadata-only scalar prototype rates, never full-codec GB/s.");
                return Ok(());
            }
            value => return Err(format!("unknown option {value}").into()),
        }
        index += 1;
    }
    let input = input.ok_or("--input FILE is required")?;
    let data = fs::read(&input)?;
    let encoded = compress(&data, &options)?;
    let frame = Frame::parse(&encoded, data.len())?;
    let blocks = analysis::command_blocks(&frame)?;
    let mut report = packing::evaluate_frame(&frame)?;
    let timings = packing::time_layouts(&blocks, iterations)?;
    report.apply_timings(&timings)?;
    let distance_timings = packing::time_distance_models(&blocks, iterations)?;
    report.apply_distance_timings(&distance_timings)?;
    let entropy_timing = packing::time_bounded_entropy(&blocks, iterations)?;
    report.apply_entropy_timing(&entropy_timing);
    let adaptive_entropy_timing = packing::time_adaptive_entropy(&blocks, iterations)?;
    report.apply_adaptive_entropy_timing(&adaptive_entropy_timing);
    let literal_entropy_timing = packing::time_frame_literal_entropy(&frame, iterations)?;
    report.apply_literal_entropy_timing(&literal_entropy_timing);
    let json = report.to_json(Some(iterations));
    if let Some(output) = output {
        let path = Path::new(&output);
        if path.exists() {
            return Err(format!("output exists: {output}").into());
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut file = fs::File::create_new(path)?;
        file.write_all(json.as_bytes())?;
        file.flush()?;
    } else {
        print!("{json}");
    }
    Ok(())
}
