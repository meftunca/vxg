fn execute_query_to_vec(args: &[String]) -> R<Vec<u8>> {
    if args.len() < 3 {
        return Err(
            "usage: vxg query FRAME [--count | --row IDX | --range START:END | --find-first STR | --filter STR | --where EXPR | --select COLS] [--limit N]"
                .into(),
        );
    }
    let frame = read_input(&args[1])?;
    let mut opts = DecodeOptions::default();
    let mut count_only = false;
    let mut row_idx: Option<usize> = None;
    let mut range: Option<(usize, usize)> = None;
    let mut find_first_str: Option<String> = None;
    let mut filter_str: Option<String> = None;
    let mut where_spec: Option<String> = None;
    let mut select_spec: Option<String> = None;
    let mut limit: usize = usize::MAX;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--count" => count_only = true,
            "--row" => row_idx = Some(val(args, &mut i)?.parse()?),
            "--range" => {
                let s = val(args, &mut i)?;
                let (a, b) = s
                    .split_once(':')
                    .or_else(|| s.split_once(".."))
                    .ok_or("expected START:END or START..END for --range")?;
                range = Some((a.parse()?, b.parse()?));
            }
            "--find-first" => find_first_str = Some(val(args, &mut i)?.to_owned()),
            "--filter" | "--contains" => filter_str = Some(val(args, &mut i)?.to_owned()),
            "--where" => where_spec = Some(val(args, &mut i)?.to_owned()),
            "--select" | "--cols" => select_spec = Some(val(args, &mut i)?.to_owned()),
            "--limit" => limit = val(args, &mut i)?.parse()?,
            "--threads" | "-t" => opts.threads = parse_threads(val(args, &mut i)?)?,
            "--no-checksum" | "--no-verify-checksum" => opts.verify_checksum = false,
            "--max-output" => opts.max_output = val(args, &mut i)?.parse()?,
            x => return Err(format!("unknown query option {x}").into()),
        }
        i += 1;
    }

    if let Some(cs) = select_spec.as_deref() {
        if count_only {
            return Err("cannot combine --count and --select".into());
        }
        let cols = parse_col_list(cs)?;
        let owned = where_spec.as_deref().map(parse_where_expr).transpose()?;
        let arena = PredArena::new();
        let expr_opt = owned.as_ref().map(|e| arena.build(e));
        return Ok(vxg::select_columns(
            &frame,
            expr_opt.as_ref(),
            &cols,
            limit,
            &opts,
        )?);
    }

    if count_only {
        let cnt: u64 = if let Some(ws) = where_spec.as_deref() {
            let owned = parse_where_expr(ws)?;
            let arena = PredArena::new();
            let expr = arena.build(&owned);
            vxg::count_where(&frame, &expr, &opts)?.min(limit as u64)
        } else {
            (vxg::frame_row_count(&frame)? as u64).min(limit as u64)
        };
        return Ok(format!("{cnt}\n").into_bytes());
    }
    if let Some(idx) = row_idx {
        return Ok(vxg::extract_row(&frame, idx, &opts)?);
    }
    if let Some((start, end)) = range {
        return Ok(vxg::extract_row_range(&frame, start, end, &opts)?);
    }
    if let Some(ws) = where_spec.as_deref() {
        let owned = parse_where_expr(ws)?;
        let arena = PredArena::new();
        let expr = arena.build(&owned);
        let matches = vxg::filter_where(&frame, &expr, limit, &opts)?;
        let total: usize = matches.iter().map(|m| m.bytes.len()).sum();
        let mut out = Vec::with_capacity(total);
        for m in matches {
            out.extend_from_slice(&m.bytes);
        }
        return Ok(out);
    }
    if let Some(needle) = find_first_str {
        if let Some(m) = vxg::find_first(&frame, needle.as_bytes(), &opts)? {
            return Ok(m.bytes);
        }
        return Ok(Vec::new());
    }
    if let Some(needle) = filter_str {
        let matches = vxg::filter_rows(&frame, needle.as_bytes(), limit, &opts)?;
        let total: usize = matches.iter().map(|m| m.bytes.len()).sum();
        let mut out = Vec::with_capacity(total);
        for m in matches {
            out.extend_from_slice(&m.bytes);
        }
        return Ok(out);
    }
    Err(
        "specify one of --count, --row, --range, --find-first, --filter, --where, or --select"
            .into(),
    )
}

fn run_query_cmd(args: &[String]) -> R<()> {
    let bytes = execute_query_to_vec(args)?;
    io::stdout().lock().write_all(&bytes)?;
    Ok(())
}
