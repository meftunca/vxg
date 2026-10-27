fn run_query_cmd(args: &[String]) -> R<()> {
    if args.len() < 3 {
        return Err(
            "usage: vxg query FRAME [--count | --row IDX | --range START:END | --find-first STR | --filter STR [--limit N]]"
                .into(),
        );
    }
    let frame = read_input(&args[1])?;
    let opts = DecodeOptions::default();
    let mut count_only = false;
    let mut row_idx: Option<usize> = None;
    let mut range: Option<(usize, usize)> = None;
    let mut find_first_str: Option<String> = None;
    let mut filter_str: Option<String> = None;
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
            "--limit" => limit = val(args, &mut i)?.parse()?,
            x => return Err(format!("unknown query option {x}").into()),
        }
        i += 1;
    }

    if count_only {
        println!("{}", vxg::frame_row_count(&frame)?);
        return Ok(());
    }
    if let Some(idx) = row_idx {
        let bytes = vxg::extract_row(&frame, idx, &opts)?;
        io::stdout().lock().write_all(&bytes)?;
        return Ok(());
    }
    if let Some((start, end)) = range {
        let bytes = vxg::extract_row_range(&frame, start, end, &opts)?;
        io::stdout().lock().write_all(&bytes)?;
        return Ok(());
    }
    if let Some(needle) = find_first_str {
        if let Some(m) = vxg::find_first(&frame, needle.as_bytes(), &opts)? {
            io::stdout().lock().write_all(&m.bytes)?;
        }
        return Ok(());
    }
    if let Some(needle) = filter_str {
        let matches = vxg::filter_rows(&frame, needle.as_bytes(), limit, &opts)?;
        let mut out = io::stdout().lock();
        for m in matches {
            out.write_all(&m.bytes)?;
        }
        return Ok(());
    }
    Err("specify one of --count, --row, --range, --find-first, or --filter".into())
}
