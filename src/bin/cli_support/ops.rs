const STREAM_BUF_CAP: usize = 256 * 1024;

fn open_stream_reader(path: &str) -> R<Box<dyn Read>> {
    if path == "-" {
        Ok(Box::new(io::BufReader::with_capacity(
            STREAM_BUF_CAP,
            io::stdin().lock(),
        )))
    } else {
        Ok(Box::new(io::BufReader::with_capacity(
            STREAM_BUF_CAP,
            fs::File::open(path)?,
        )))
    }
}

fn open_stream_writer(path: &str, force: bool) -> R<Box<dyn Write>> {
    if path == "-" {
        return Ok(Box::new(io::BufWriter::with_capacity(
            STREAM_BUF_CAP,
            io::stdout().lock(),
        )));
    }
    let mut opts = OpenOptions::new();
    opts.write(true);
    if force {
        opts.create(true).truncate(true);
    } else {
        opts.create_new(true);
    }
    Ok(Box::new(io::BufWriter::with_capacity(
        STREAM_BUF_CAP,
        opts.open(path)?,
    )))
}

enum CliColAction {
    SetStr(Vec<u8>),
    SetRaw(Vec<u8>),
    Upper,
    Lower,
    Redact,
    Add(i64),
}

fn run_ops_cmd(args: &[String]) -> R<()> {
    let cmd = args[0].as_str();
    if args.len() < 4 {
        return Err(format!("usage: vxg {cmd} INPUT OUTPUT [options]").into());
    }
    let input = &args[1];
    let output = &args[2];
    if input != "-" && input == output {
        return Err("input and output must differ".into());
    }
    let mut enc = Options::default();
    let mut dec = DecodeOptions::default();
    let mut explicit_max_out = false;
    let mut stream_mode = false;
    let mut use_mmap = true;
    let mut where_spec: Option<String> = None;
    let mut col_spec: Option<String> = None;
    let mut cols_spec: Option<String> = None;
    let mut action: Option<CliColAction> = None;
    let mut force = false;
    let mut quiet = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--where" => where_spec = Some(val(args, &mut i)?.to_owned()),
            "--col" => col_spec = Some(val(args, &mut i)?.to_owned()),
            "--cols" | "--select" => cols_spec = Some(val(args, &mut i)?.to_owned()),
            "--set" => action = Some(CliColAction::SetStr(val(args, &mut i)?.as_bytes().to_vec())),
            "--set-raw" => {
                action = Some(CliColAction::SetRaw(val(args, &mut i)?.as_bytes().to_vec()))
            }
            "--upper" => action = Some(CliColAction::Upper),
            "--lower" => action = Some(CliColAction::Lower),
            "--redact" => action = Some(CliColAction::Redact),
            "--add" => action = Some(CliColAction::Add(val(args, &mut i)?.parse()?)),
            "--block-size" => enc.block_size = val(args, &mut i)?.parse()?,
            "--legacy" => enc.legacy_v2 = true,
            "--threads" | "-t" => {
                let th = parse_threads(val(args, &mut i)?)?;
                enc.threads = th;
                dec.threads = th;
            }
            "--no-checksum" | "--no-verify-checksum" => {
                enc.checksum = false;
                dec.verify_checksum = false;
            }
            "--max-output" => {
                dec.max_output = val(args, &mut i)?.parse()?;
                explicit_max_out = true;
            }
            "--stream" => stream_mode = true,
            "--no-mmap" => use_mmap = false,
            "--force" | "-f" => force = true,
            "--quiet" | "-q" => quiet = true,
            x => return Err(format!("unknown option {x}").into()),
        }
        i += 1;
    }
    if output != "-" && !force && Path::new(output).exists() {
        return Err("output exists (use --force to replace it)".into());
    }
    if !explicit_max_out {
        dec.max_output = usize::MAX;
    }

    if stream_mode || input == "-" {
        let start = std::time::Instant::now();
        let existed = output != "-" && Path::new(output).exists();
        let mut rdr = open_stream_reader(input)?;
        let mut wtr = open_stream_writer(output, force)?;
        let res: R<u64> = (|| {
            let written = match cmd {
                "filter-frame" => {
                    let ws = where_spec
                        .as_deref()
                        .ok_or("--where EXPR is required for filter-frame")?;
                    let pw = vxg::ParsedWhere::parse(ws)?;
                    pw.with_expr(|expr| {
                        vxg::filter_frame_where_stream(&mut rdr, &mut wtr, expr, &enc, &dec)
                    })?
                }
                "project" => {
                    let cs = cols_spec
                        .as_deref()
                        .or(col_spec.as_deref())
                        .ok_or("--cols COL1,COL2,... is required for project")?;
                    let cols = vxg::parse_col_list(cs)?;
                    let pw = where_spec
                        .as_deref()
                        .map(vxg::ParsedWhere::parse)
                        .transpose()?;
                    vxg::with_opt_where(pw.as_ref(), |expr_opt| {
                        vxg::project_frame_stream(&mut rdr, &mut wtr, expr_opt, &cols, &enc, &dec)
                    })?
                }
                "update" | "map" => {
                    let cs = col_spec.as_deref().ok_or("--col COL is required")?;
                    let target_col = vxg::parse_col_target(cs)?;
                    let act = action.as_ref().ok_or(
                        "specify one column action: --set VAL | --set-raw RAW | --upper | --lower | --redact | --add N",
                    )?;
                    let upper_fn = |b: &[u8]| b.to_ascii_uppercase();
                    let lower_fn = |b: &[u8]| b.to_ascii_lowercase();
                    let add_delta = match act {
                        CliColAction::Add(d) => *d,
                        _ => 0,
                    };
                    let add_fn = |v: u64| {
                        if add_delta >= 0 {
                            v.saturating_add(add_delta as u64)
                        } else {
                            v.saturating_sub(add_delta.unsigned_abs())
                        }
                    };
                    let col_op = match act {
                        CliColAction::SetStr(b) => vxg::ColOp::SetStr(b),
                        CliColAction::SetRaw(b) => vxg::ColOp::SetRaw(b),
                        CliColAction::Redact => vxg::ColOp::SetStr(b"[REDACTED]"),
                        CliColAction::Upper => vxg::ColOp::MapStr(&upper_fn),
                        CliColAction::Lower => vxg::ColOp::MapStr(&lower_fn),
                        CliColAction::Add(_) => vxg::ColOp::MapU64(&add_fn),
                    };
                    if cmd == "update" {
                        let ws = where_spec
                            .as_deref()
                            .ok_or("--where SPEC is required for update")?;
                        let pcp = vxg::ParsedColPred::parse(ws)?;
                        let pred = pcp.as_col_pred();
                        vxg::update_column_stream(
                            &mut rdr, &mut wtr, &pred, target_col, col_op, &enc, &dec,
                        )?
                    } else {
                        if where_spec.is_some() {
                            return Err(
                                "map applies to all rows; use 'vxg update' with --where".into()
                            );
                        }
                        vxg::map_column_stream(&mut rdr, &mut wtr, target_col, col_op, &enc, &dec)?
                    }
                }
                _ => unreachable!(),
            };
            wtr.flush()?;
            Ok(written)
        })();
        let written = match res {
            Ok(w) => w,
            Err(e) => {
                drop(wtr);
                if output != "-" && !existed {
                    let _ = fs::remove_file(output);
                }
                return Err(e);
            }
        };
        if !quiet {
            eprintln!(
                "{cmd} (stream): {written} frame bytes ({:.3} ms)",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
        return Ok(());
    }

    let frame_bytes = read_input_bytes(input, use_mmap)?;
    if let vxg::InputBytes::Mapped(ref m) = frame_bytes {
        m.advise_sequential();
    }
    let start = std::time::Instant::now();

    let result = match cmd {
        "filter-frame" => {
            let ws = where_spec
                .as_deref()
                .ok_or("--where EXPR is required for filter-frame")?;
            let pw = vxg::ParsedWhere::parse(ws)?;
            pw.with_expr(|expr| vxg::filter_frame_where(&frame_bytes, expr, &enc, &dec))?
        }
        "project" => {
            let cs = cols_spec
                .as_deref()
                .or(col_spec.as_deref())
                .ok_or("--cols COL1,COL2,... is required for project")?;
            let cols = vxg::parse_col_list(cs)?;
            let pw = where_spec
                .as_deref()
                .map(vxg::ParsedWhere::parse)
                .transpose()?;
            vxg::with_opt_where(pw.as_ref(), |expr_opt| {
                vxg::project_frame(&frame_bytes, expr_opt, &cols, &enc, &dec)
            })?
        }
        "update" | "map" => {
            let cs = col_spec.as_deref().ok_or("--col COL is required")?;
            let target_col = vxg::parse_col_target(cs)?;
            let act = action.as_ref().ok_or(
                "specify one column action: --set VAL | --set-raw RAW | --upper | --lower | --redact | --add N",
            )?;
            let upper_fn = |b: &[u8]| b.to_ascii_uppercase();
            let lower_fn = |b: &[u8]| b.to_ascii_lowercase();
            let add_delta = match act {
                CliColAction::Add(d) => *d,
                _ => 0,
            };
            let add_fn = |v: u64| {
                if add_delta >= 0 {
                    v.saturating_add(add_delta as u64)
                } else {
                    v.saturating_sub(add_delta.unsigned_abs())
                }
            };
            let col_op = match act {
                CliColAction::SetStr(b) => vxg::ColOp::SetStr(b),
                CliColAction::SetRaw(b) => vxg::ColOp::SetRaw(b),
                CliColAction::Redact => vxg::ColOp::SetStr(b"[REDACTED]"),
                CliColAction::Upper => vxg::ColOp::MapStr(&upper_fn),
                CliColAction::Lower => vxg::ColOp::MapStr(&lower_fn),
                CliColAction::Add(_) => vxg::ColOp::MapU64(&add_fn),
            };
            if cmd == "update" {
                let ws = where_spec
                    .as_deref()
                    .ok_or("--where SPEC is required for update")?;
                let pcp = vxg::ParsedColPred::parse(ws)?;
                let pred = pcp.as_col_pred();
                vxg::update_column(&frame_bytes, &pred, target_col, col_op, &enc, &dec)?
            } else {
                if where_spec.is_some() {
                    return Err("map applies to all rows; use 'vxg update' with --where".into());
                }
                vxg::map_column(&frame_bytes, target_col, col_op, &enc, &dec)?
            }
        }
        _ => unreachable!(),
    };

    let elapsed = start.elapsed().as_secs_f64();
    write_output(output, &result, force)?;
    if !quiet {
        eprintln!(
            "{cmd}: {} -> {} frame bytes ({:.3} ms)",
            frame_bytes.len(),
            result.len(),
            elapsed * 1000.0
        );
    }
    Ok(())
}
