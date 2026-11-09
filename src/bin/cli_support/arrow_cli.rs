fn run_arrow_cmd(args: &[String]) -> R<()> {
    if args.len() < 3 {
        return Err(
            "usage: vxg arrow FRAME OUTPUT [--sql \"SELECT ...\" | [--cols C1,C2] [--where EXPR] [--limit N]] [-t N|auto] [--no-mmap] [-f] [-q]"
                .into(),
        );
    }
    let input = &args[1];
    let output = &args[2];
    let mut sql_str: Option<&str> = None;
    let mut cols_str: Option<&str> = None;
    let mut where_str: Option<&str> = None;
    let mut limit_opt: Option<usize> = None;
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
            "--cols" | "--select" | "-c" => cols_str = Some(val(args, &mut i)?),
            "--where" | "-w" => where_str = Some(val(args, &mut i)?),
            "--limit" | "-n" => limit_opt = Some(val(args, &mut i)?.parse()?),
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

    if sql_str.is_some() && (cols_str.is_some() || where_str.is_some() || limit_opt.is_some()) {
        return Err("cannot combine --sql with --cols, --where, or --limit".into());
    }
    if output != "-" && !force && Path::new(output).exists() {
        return Err("output exists (use --force to replace it)".into());
    }

    let bytes = read_input_bytes(input, use_mmap)?;
    if let vxg::InputBytes::Mapped(ref m) = bytes {
        m.advise_sequential();
    }

    let batch = if let Some(s) = sql_str {
        let q = vxg::SqlQuery::parse(s)?;
        vxg::export_arrow_batch(&bytes, Some(&q), &dec)?
    } else {
        let parsed_cols = match cols_str {
            Some(c) if !c.trim().is_empty() && c.trim() != "*" => Some(vxg::parse_col_list(c)?),
            _ => None,
        };
        let parsed_where = match where_str {
            Some(w) if !w.trim().is_empty() => Some(vxg::parse_arrow_where(w)?),
            _ => None,
        };
        match parsed_where.as_ref() {
            Some(aw) => aw.with_pred_expr(|we| {
                vxg::export_arrow_batch_where(
                    &bytes,
                    parsed_cols.as_deref(),
                    Some(we),
                    limit_opt,
                    &dec,
                )
            })?,
            None => vxg::export_arrow_batch_where(
                &bytes,
                parsed_cols.as_deref(),
                None,
                limit_opt,
                &dec,
            )?,
        }
    };

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
