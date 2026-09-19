use std::{env, fs, io::Write, path::Path};

use vxg::{analysis, compress, Frame, Options, Simd};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn value<'a>(args: &'a [String], index: &mut usize) -> R<&'a str> {
    *index += 1;
    Ok(args.get(*index).ok_or("missing option value")?.as_str())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("analysis failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input = None;
    let mut output = None;
    let mut options = Options {
        checksum: false,
        ..Options::default()
    };
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => input = Some(value(&args, &mut index)?.to_owned()),
            "--output" => output = Some(value(&args, &mut index)?.to_owned()),
            "--block-size" => options.block_size = value(&args, &mut index)?.parse()?,
            "--threads" => options.threads = value(&args, &mut index)?.parse()?,
            "--acceleration" => options.acceleration = value(&args, &mut index)?.parse()?,
            "--simd" => options.simd = value(&args, &mut index)?.parse::<Simd>()?,
            "--help" | "-h" => {
                println!("vxg-analyze --input FILE [--output REPORT.json] [--block-size 65536] [--threads 1] [--acceleration 1] [--simd auto]\nCompresses the input with checksum off, then writes an offline command-stream analysis. This is not a timed benchmark.");
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
    let json = analysis::analyze_frame(&frame)?.to_json();
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
