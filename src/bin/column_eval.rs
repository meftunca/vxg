use std::{env, fs, io::Write, path::Path};

use vxg::columns;

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn value<'a>(args: &'a [String], index: &mut usize) -> R<&'a str> {
    *index += 1;
    Ok(args.get(*index).ok_or("missing option value")?.as_str())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("column evaluation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input = None;
    let mut output = None;
    let mut iterations = 1usize;
    let mut block_size = 65_536usize;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--input" => input = Some(value(&args, &mut index)?.to_owned()),
            "--output" => output = Some(value(&args, &mut index)?.to_owned()),
            "--iterations" => iterations = value(&args, &mut index)?.parse()?,
            "--block-size" => block_size = value(&args, &mut index)?.parse()?,
            "--help" | "-h" => {
                println!("vxg-column-eval --input FILE [--output REPORT.json] [--iterations 1] [--block-size 65536]\nBuilds and decodes a complete independent-block P5D structured-column research packet. It is a scalar oracle, not a VXG wire format or a codec GB/s benchmark.");
                return Ok(());
            }
            value => return Err(format!("unknown option {value}").into()),
        }
        index += 1;
    }
    let input = input.ok_or("--input FILE is required")?;
    let data = fs::read(&input)?;
    let mut evaluation = columns::evaluate(&data, block_size)?;
    evaluation.apply_timing(&columns::time(&data, block_size, iterations)?);
    let json = evaluation.to_json(iterations);
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
