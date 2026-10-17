use std::{env, fs, io::Write, path::Path};

use vxg::{compress, groups, Frame, Options, Simd};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn value<'a>(args: &'a [String], index: &mut usize) -> R<&'a str> {
    *index += 1;
    Ok(args.get(*index).ok_or("missing option value")?.as_str())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("grouped command evaluation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input = None;
    let mut output = None;
    let mut iterations = 1usize;
    let mut options = Options { checksum: false, ..Options::default() };
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
                println!("vxg-group-eval --input FILE [--output REPORT.json] [--iterations 1] [--block-size 65536] [--threads 1] [--acceleration 1] [--simd auto]\\nBuilds a bench-only 64-command bit-sliced packet from the existing v2 trace. It separately measures packet validation, validated replay and packet parse+validate+decode; it is not a VXG v3 or public safe-decode benchmark.");
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
    let report = groups::evaluate_frame(&frame, &data, iterations, options.simd)?;
    let json = report.to_json(iterations, options.simd);
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
