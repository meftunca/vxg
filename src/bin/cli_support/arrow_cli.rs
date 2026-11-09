fn run_arrow_cmd(args: &[String]) -> R<()> {
    if args.len() < 3 {
        return Err(
            "usage: vxg arrow FRAME OUTPUT [--sql \"SELECT ...\"] [-t N|auto] [--no-mmap] [-f] [-q]"
                .into(),
        );
    }
    let input = &args[1];
    let output = &args[2];
    let mut sql_str: Option<&str> = None;
    let mut use_mmap = true;
    let mut force = false;
    let mut quiet = false;
    let mut dec = DecodeOptions {
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--sql" | "--query" | "-e" => sql_str = Some(val(args, &mut i)?),
            "--threads" | "-t" => dec.threads = parse_threads(val(args, &mut i)?)?,
            "--simd" => dec.simd = val(args, &mut i)?.parse()?,
            "--no-checksum" | "--no-verify-checksum" => dec.verify_checksum = false,
            "--no-mmap" => use_mmap = false,
            "--force" | "-f" => force = true,
            "--quiet" | "-q" => quiet = true,
            x => return Err(format!("unknown option '{x}' for 'vxg arrow'").into()),
        }
        i += 1;
    }

    if output != "-" && !force && Path::new(output).exists() {
        return Err("output exists (use --force to replace it)".into());
    }
    let q_parsed = match sql_str {
        Some(s) => Some(vxg::SqlQuery::parse(s)?),
        None => None,
    };
    let bytes = read_input_bytes(input, use_mmap)?;
    if let vxg::InputBytes::Mapped(ref m) = bytes {
        m.advise_sequential();
    }
    let batch = vxg::export_arrow_batch(&bytes, q_parsed.as_ref(), &dec)?;
    let ipc = batch.to_ipc_stream();
    write_output(output, &ipc, force)?;
    if !quiet && output != "-" {
        eprintln!(
            "exported {} rows x {} cols ({} Arrow IPC bytes)",
            batch.num_rows,
            batch.columns.len(),
            ipc.len()
        );
    }
    Ok(())
}
