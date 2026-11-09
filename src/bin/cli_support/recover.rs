fn run_recover_cmd(args: &[String]) -> R<()> {
    if args.len() < 3 {
        return Err(
            "usage: vxg recover INPUT OUTPUT [--simd MODE] [--max-output BYTES] [--no-checksum] [--no-mmap] [--force] [--quiet]"
                .into(),
        );
    }
    let input = &args[1];
    let output = &args[2];
    if input != "-" && input == output {
        return Err("input and output must differ".into());
    }
    let mut dec = DecodeOptions::default();
    let mut explicit_max_out = false;
    let mut use_mmap = true;
    let mut force = false;
    let mut quiet = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" | "-t" => dec.threads = parse_threads(val(args, &mut i)?)?,
            "--simd" => dec.simd = val(args, &mut i)?.parse()?,
            "--max-output" => {
                dec.max_output = val(args, &mut i)?.parse()?;
                explicit_max_out = true;
            }
            "--no-checksum" | "--no-verify-checksum" => dec.verify_checksum = false,
            "--no-mmap" => use_mmap = false,
            "--force" | "-f" => force = true,
            "--quiet" | "-q" => quiet = true,
            "--recover" => {}
            x => return Err(format!("unknown option {x}").into()),
        }
        i += 1;
    }
    if !explicit_max_out {
        dec.max_output = usize::MAX;
    }
    if output != "-" && !force && Path::new(output).exists() {
        return Err("output exists (use --force to replace it)".into());
    }
    let bytes = read_input_bytes(input, use_mmap)?;
    let report = vxg::recover_frame(&bytes, &dec)?;
    if report.total_blocks > 0 && report.recovered_blocks.is_empty() {
        return Err(format!(
            "recovery failed: 0/{} blocks intact",
            report.total_blocks
        )
        .into());
    }
    write_output(output, &report.data, force)?;
    if !quiet {
        eprintln!(
            "recovered {}/{} blocks ({}/{} raw bytes)",
            report.recovered_blocks.len(),
            report.total_blocks,
            report.data.len(),
            report.expected_raw_size
        );
        for fb in &report.failed_blocks {
            eprintln!(
                "  skipped block {} (raw_offset={}, raw_size={}): {}",
                fb.index, fb.raw_offset, fb.raw_size, fb.reason
            );
        }
    }
    Ok(())
}
