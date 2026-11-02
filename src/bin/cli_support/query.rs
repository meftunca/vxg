fn execute_query_to_writer<W: Write>(args: &[String], writer: &mut W) -> R<()> {
    if args.len() < 3 {
        return Err(
            "usage: vxg query FRAME [--count | --row IDX | --range START:END | --find-first STR | --filter STR | --where EXPR | --select COLS] [--limit N] [--stream] [--no-mmap]"
                .into(),
        );
    }
    let input = &args[1];
    let mut opts = DecodeOptions::default();
    let mut explicit_max_out = false;
    let mut stream_mode = false;
    let mut use_mmap = true;
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
            "--max-output" => {
                opts.max_output = val(args, &mut i)?.parse()?;
                explicit_max_out = true;
            }
            "--stream" => stream_mode = true,
            "--no-mmap" => use_mmap = false,
            x => return Err(format!("unknown query option {x}").into()),
        }
        i += 1;
    }
    if !explicit_max_out {
        opts.max_output = usize::MAX;
    }

    if stream_mode && (row_idx.is_some() || range.is_some()) {
        return Err(
            "--row and --range require random-access frame directory (omit --stream)".into(),
        );
    }

    if stream_mode || (input == "-" && row_idx.is_none() && range.is_none()) {
        let mut rdr = open_stream_reader(input)?;
        if let Some(cs) = select_spec.as_deref() {
            if count_only {
                return Err("cannot combine --count and --select".into());
            }
            let cols = parse_col_list(cs)?;
            let owned = where_spec.as_deref().map(parse_where_expr).transpose()?;
            let arena = PredArena::new();
            let expr_opt = owned.as_ref().map(|e| arena.build(e));
            vxg::select_columns_stream(
                &mut rdr,
                &mut *writer,
                expr_opt.as_ref(),
                &cols,
                limit,
                &opts,
            )?;
            return Ok(());
        }
        if count_only {
            let cnt: u64 = if let Some(ws) = where_spec.as_deref() {
                let owned = parse_where_expr(ws)?;
                let arena = PredArena::new();
                let expr = arena.build(&owned);
                vxg::count_where_stream(&mut rdr, &expr, &opts)?.min(limit as u64)
            } else {
                vxg::frame_row_count_stream(&mut rdr, &opts)?.min(limit as u64)
            };
            writeln!(writer, "{cnt}")?;
            return Ok(());
        }
        if let Some(ws) = where_spec.as_deref() {
            let owned = parse_where_expr(ws)?;
            let arena = PredArena::new();
            let expr = arena.build(&owned);
            vxg::filter_where_stream(&mut rdr, &mut *writer, &expr, limit, &opts)?;
            return Ok(());
        }
        if let Some(needle) = find_first_str {
            let expr = vxg::PredExpr::RowContains(needle.as_bytes());
            vxg::filter_where_stream(&mut rdr, &mut *writer, &expr, 1, &opts)?;
            return Ok(());
        }
        if let Some(needle) = filter_str {
            let expr = vxg::PredExpr::RowContains(needle.as_bytes());
            vxg::filter_where_stream(&mut rdr, &mut *writer, &expr, limit, &opts)?;
            return Ok(());
        }
        return Err(
            "specify one of --count, --row, --range, --find-first, --filter, --where, or --select"
                .into(),
        );
    }

    let frame = read_input_bytes(input, use_mmap)?;
    if let vxg::InputBytes::Mapped(ref m) = frame {
        if row_idx.is_some() || range.is_some() {
            m.advise_random();
        } else {
            m.advise_sequential();
        }
    }

    if let Some(cs) = select_spec.as_deref() {
        if count_only {
            return Err("cannot combine --count and --select".into());
        }
        let cols = parse_col_list(cs)?;
        let owned = where_spec.as_deref().map(parse_where_expr).transpose()?;
        let arena = PredArena::new();
        let expr_opt = owned.as_ref().map(|e| arena.build(e));
        let out = vxg::select_columns(&frame, expr_opt.as_ref(), &cols, limit, &opts)?;
        writer.write_all(&out)?;
        return Ok(());
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
        writeln!(writer, "{cnt}")?;
        return Ok(());
    }
    if let Some(idx) = row_idx {
        let out = vxg::extract_row(&frame, idx, &opts)?;
        writer.write_all(&out)?;
        return Ok(());
    }
    if let Some((start, end)) = range {
        let out = vxg::extract_row_range(&frame, start, end, &opts)?;
        writer.write_all(&out)?;
        return Ok(());
    }
    if let Some(ws) = where_spec.as_deref() {
        let owned = parse_where_expr(ws)?;
        let arena = PredArena::new();
        let expr = arena.build(&owned);
        for m in vxg::filter_where(&frame, &expr, limit, &opts)? {
            writer.write_all(&m.bytes)?;
        }
        return Ok(());
    }
    if let Some(needle) = find_first_str {
        if let Some(m) = vxg::find_first(&frame, needle.as_bytes(), &opts)? {
            writer.write_all(&m.bytes)?;
        }
        return Ok(());
    }
    if let Some(needle) = filter_str {
        for m in vxg::filter_rows(&frame, needle.as_bytes(), limit, &opts)? {
            writer.write_all(&m.bytes)?;
        }
        return Ok(());
    }
    Err(
        "specify one of --count, --row, --range, --find-first, --filter, --where, or --select"
            .into(),
    )
}

#[cfg(test)]
fn execute_query_to_vec(args: &[String]) -> R<Vec<u8>> {
    let mut out = Vec::new();
    execute_query_to_writer(args, &mut out)?;
    Ok(out)
}

fn run_query_cmd(args: &[String]) -> R<()> {
    let mut out = io::BufWriter::with_capacity(STREAM_BUF_CAP, io::stdout().lock());
    execute_query_to_writer(args, &mut out)?;
    out.flush()?;
    Ok(())
}
