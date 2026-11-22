fn is_http_url(s: &str) -> bool {
    s.starts_with("http://")
}

fn run_serve_cmd(args: &[String]) -> R<()> {
    if args.len() < 2 {
        return Err("usage: vxg serve DIR [--port 8080]".into());
    }
    let dir = Path::new(&args[1]);
    if !dir.is_dir() {
        return Err(format!("serve directory not found: {}", dir.display()).into());
    }
    let mut port: u16 = 8080;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => port = val(args, &mut i)?.parse()?,
            x => return Err(format!("unknown option '{x}' for 'vxg serve'").into()),
        }
        i += 1;
    }
    Ok(vxg::serve_http_range(dir, port)?)
}

fn run_remote_info(url: &str) -> R<()> {
    let reader = vxg::HttpRangeReader::connect(url)?;
    let rf = vxg::RemoteFrame::open(reader)?;
    let info = rf.info();
    let st = rf.stats();
    let ratio = if info.raw_size == 0 {
        1.0
    } else {
        info.compressed_size as f64 / info.raw_size as f64
    };
    println!(
        "version={} raw={} encoded={} ratio={:.4} blocks={} block_size={} checksum={} dict={} row_index={} rows={} fetched={}B/{}B ({:.2}%, {} reqs)",
        info.version,
        info.raw_size,
        info.compressed_size,
        ratio,
        info.block_count,
        info.block_size,
        info.checksummed,
        info.has_dict,
        info.has_row_index,
        info.row_count.map_or_else(|| "unknown".into(), |n| n.to_string()),
        st.bytes_fetched,
        st.total_bytes,
        st.fetched_pct(),
        st.requests
    );
    Ok(())
}

fn run_remote_block(
    url: &str,
    block_idx: usize,
    output: &str,
    force: bool,
    dec: &DecodeOptions,
) -> R<()> {
    let reader = vxg::HttpRangeReader::connect(url)?;
    let mut rf = vxg::RemoteFrame::open(reader)?;
    let out = rf.extract_block(block_idx, dec)?;
    let st = rf.stats();
    eprintln!(
        "remote block {block_idx}: fetched {}B / {}B ({:.2}%, {} HTTP Range reqs)",
        st.bytes_fetched,
        st.total_bytes,
        st.fetched_pct(),
        st.requests
    );
    write_output(output, &out, force)
}

fn run_remote_query_row<W: Write>(
    url: &str,
    row_idx: usize,
    dec: &DecodeOptions,
    writer: &mut W,
) -> R<()> {
    let reader = vxg::HttpRangeReader::connect(url)?;
    let mut rf = vxg::RemoteFrame::open(reader)?;
    if let Some(row) = rf.extract_row(row_idx, dec)? {
        writer.write_all(&row)?;
    }
    let st = rf.stats();
    eprintln!(
        "remote row {row_idx}: fetched {}B / {}B ({:.2}%, {} HTTP Range reqs)",
        st.bytes_fetched,
        st.total_bytes,
        st.fetched_pct(),
        st.requests
    );
    Ok(())
}

fn execute_remote_sql(
    url: &str,
    parsed: &vxg::SqlQuery,
    format: vxg::SqlFormat,
    no_header: bool,
    dec: &DecodeOptions,
) -> R<Vec<u8>> {
    let reader = vxg::HttpRangeReader::connect(url)?;
    let mut rf = vxg::RemoteFrame::open(reader)?;
    let out = rf.execute_sql(parsed, format, no_header, dec)?;
    let st = rf.stats();
    eprintln!(
        "remote sql: fetched {}B / {}B ({:.2}%, {} HTTP Range reqs)",
        st.bytes_fetched,
        st.total_bytes,
        st.fetched_pct(),
        st.requests
    );
    Ok(out)
}
