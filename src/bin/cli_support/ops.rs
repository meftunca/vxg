enum OwnedPred {
    StrEq(Vec<u8>),
    StrNe(Vec<u8>),
    StrPrefix(Vec<u8>),
    StrSuffix(Vec<u8>),
    StrContains(Vec<u8>),
    Num(vxg::NumCmp),
}

impl OwnedPred {
    fn as_value_pred(&self) -> vxg::ValuePred<'_> {
        match self {
            Self::StrEq(b) => vxg::ValuePred::Str(vxg::StrCmp::Eq(b)),
            Self::StrNe(b) => vxg::ValuePred::Str(vxg::StrCmp::Ne(b)),
            Self::StrPrefix(b) => vxg::ValuePred::Str(vxg::StrCmp::Prefix(b)),
            Self::StrSuffix(b) => vxg::ValuePred::Str(vxg::StrCmp::Suffix(b)),
            Self::StrContains(b) => vxg::ValuePred::Str(vxg::StrCmp::Contains(b)),
            Self::Num(c) => vxg::ValuePred::Num(*c),
        }
    }
}

fn parse_col_target(s: &str) -> R<vxg::ColTarget<'_>> {
    if let Some(rest) = s.strip_prefix("json:") {
        if rest.is_empty() {
            return Err("empty json column key".into());
        }
        Ok(vxg::ColTarget::JsonKey(rest))
    } else if let Some(rest) = s.strip_prefix("csv:") {
        Ok(vxg::ColTarget::Delimited {
            delim: b',',
            index: rest.parse()?,
        })
    } else if let Some(rest) = s.strip_prefix("tsv:") {
        Ok(vxg::ColTarget::Delimited {
            delim: b'\t',
            index: rest.parse()?,
        })
    } else if let Some(rest) = s.strip_prefix("psv:") {
        Ok(vxg::ColTarget::Delimited {
            delim: b'|',
            index: rest.parse()?,
        })
    } else if !s.is_empty() {
        Ok(vxg::ColTarget::JsonKey(s))
    } else {
        Err("empty column target".into())
    }
}

fn parse_where_spec(spec: &str) -> R<(vxg::ColTarget<'_>, OwnedPred)> {
    for (op_tok, ctor) in [
        (">=", (|n| vxg::NumCmp::Ge(n)) as fn(u64) -> vxg::NumCmp),
        ("<=", |n| vxg::NumCmp::Le(n)),
    ] {
        if let Some((lhs, rhs)) = spec.split_once(op_tok) {
            return Ok((parse_col_target(lhs)?, OwnedPred::Num(ctor(rhs.parse()?))));
        }
    }
    if let Some((lhs, rhs)) = spec.split_once("!=") {
        return Ok((parse_col_target(lhs)?, OwnedPred::StrNe(rhs.as_bytes().to_vec())));
    }
    if let Some((lhs, rhs)) = spec.split_once("==") {
        return Ok((parse_col_target(lhs)?, OwnedPred::StrEq(rhs.as_bytes().to_vec())));
    }
    if let Some((lhs, rhs)) = spec.split_once('=') {
        let col = parse_col_target(lhs)?;
        let pred = if let Some(v) = rhs.strip_prefix("eq:") {
            OwnedPred::StrEq(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("ne:") {
            OwnedPred::StrNe(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("prefix:") {
            OwnedPred::StrPrefix(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("suffix:") {
            OwnedPred::StrSuffix(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("contains:") {
            OwnedPred::StrContains(v.as_bytes().to_vec())
        } else if let Some(v) = rhs.strip_prefix("num:").or_else(|| rhs.strip_prefix("num-eq:")) {
            OwnedPred::Num(vxg::NumCmp::Eq(v.parse()?))
        } else if let Some(v) = rhs.strip_prefix("num-ne:") {
            OwnedPred::Num(vxg::NumCmp::Ne(v.parse()?))
        } else if let Some(v) = rhs.strip_prefix("gt:") {
            OwnedPred::Num(vxg::NumCmp::Gt(v.parse()?))
        } else if let Some(v) = rhs.strip_prefix("ge:") {
            OwnedPred::Num(vxg::NumCmp::Ge(v.parse()?))
        } else if let Some(v) = rhs.strip_prefix("lt:") {
            OwnedPred::Num(vxg::NumCmp::Lt(v.parse()?))
        } else if let Some(v) = rhs.strip_prefix("le:") {
            OwnedPred::Num(vxg::NumCmp::Le(v.parse()?))
        } else {
            OwnedPred::StrEq(rhs.as_bytes().to_vec())
        };
        return Ok((col, pred));
    }
    for (op_ch, ctor) in [
        ('>', (|n| vxg::NumCmp::Gt(n)) as fn(u64) -> vxg::NumCmp),
        ('<', |n| vxg::NumCmp::Lt(n)),
    ] {
        if let Some((lhs, rhs)) = spec.split_once(op_ch) {
            return Ok((parse_col_target(lhs)?, OwnedPred::Num(ctor(rhs.parse()?))));
        }
    }
    Err("invalid --where SPEC (examples: json:level=ERROR, json:bytes>=10000, csv:1=prefix:ERR)".into())
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
    let mut where_spec: Option<String> = None;
    let mut col_spec: Option<String> = None;
    let mut action: Option<CliColAction> = None;
    let mut force = false;
    let mut quiet = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--where" => where_spec = Some(val(args, &mut i)?.to_owned()),
            "--col" => col_spec = Some(val(args, &mut i)?.to_owned()),
            "--set" => action = Some(CliColAction::SetStr(val(args, &mut i)?.as_bytes().to_vec())),
            "--set-raw" => {
                action = Some(CliColAction::SetRaw(val(args, &mut i)?.as_bytes().to_vec()))
            }
            "--upper" => action = Some(CliColAction::Upper),
            "--lower" => action = Some(CliColAction::Lower),
            "--redact" => action = Some(CliColAction::Redact),
            "--add" => action = Some(CliColAction::Add(val(args, &mut i)?.parse()?)),
            "--block-size" => enc.block_size = val(args, &mut i)?.parse()?,
            "--no-checksum" | "--no-verify-checksum" => {
                enc.checksum = false;
                dec.verify_checksum = false;
            }
            "--max-output" => dec.max_output = val(args, &mut i)?.parse()?,
            "--force" | "-f" => force = true,
            "--quiet" | "-q" => quiet = true,
            x => return Err(format!("unknown option {x}").into()),
        }
        i += 1;
    }
    if output != "-" && !force && Path::new(output).exists() {
        return Err("output exists (use --force to replace it)".into());
    }
    let frame_bytes = read_input(input)?;
    let start = std::time::Instant::now();

    let result = match cmd {
        "filter-frame" => {
            let ws = where_spec.as_deref().ok_or("--where SPEC is required for filter-frame")?;
            let (w_col, w_pred) = parse_where_spec(ws)?;
            let pred = vxg::ColPred {
                col: w_col,
                pred: w_pred.as_value_pred(),
            };
            vxg::filter_frame(&frame_bytes, &pred, &enc, &dec)?
        }
        "update" | "map" => {
            let cs = col_spec.as_deref().ok_or("--col COL is required")?;
            let target_col = parse_col_target(cs)?;
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
                let ws = where_spec.as_deref().ok_or("--where SPEC is required for update")?;
                let (w_col, w_pred) = parse_where_spec(ws)?;
                let pred = vxg::ColPred {
                    col: w_col,
                    pred: w_pred.as_value_pred(),
                };
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
