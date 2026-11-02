fn format_agg_stats_text(st: &vxg::AggStats) -> String {
    if let Some(avg) = st.avg() {
        format!(
            "rows={} count={} sum={} avg={avg:.4} min={} max={}",
            st.row_count,
            st.num_count,
            st.sum,
            st.min.unwrap_or(0),
            st.max.unwrap_or(0)
        )
    } else {
        format!(
            "rows={} count=0 sum=0 avg=null min=null max=null",
            st.row_count
        )
    }
}

fn format_agg_stats_json_fields(st: &vxg::AggStats) -> String {
    if let Some(avg) = st.avg() {
        format!(
            "\"rows\":{},\"count\":{},\"sum\":{},\"avg\":{avg:.6},\"min\":{},\"max\":{}",
            st.row_count,
            st.num_count,
            st.sum,
            st.min.unwrap_or(0),
            st.max.unwrap_or(0)
        )
    } else {
        format!(
            "\"rows\":{},\"count\":0,\"sum\":0,\"avg\":null,\"min\":null,\"max\":null",
            st.row_count
        )
    }
}

fn format_group_key_display(key: &vxg::GroupKey) -> String {
    String::from_utf8_lossy(key.as_bytes()).into_owned()
}

fn format_group_key_json(key: &vxg::GroupKey) -> String {
    match key {
        vxg::GroupKey::Str(b) => {
            let mut out = Vec::new();
            vxg::ops::encode_json_str(b, &mut out);
            String::from_utf8(out).unwrap_or_else(|_| "\"\"".to_owned())
        }
        vxg::GroupKey::RawJson(b) => String::from_utf8(b.clone()).unwrap_or_else(|_| {
            let mut out = Vec::new();
            vxg::ops::encode_json_str(b, &mut out);
            String::from_utf8(out).unwrap_or_else(|_| "\"\"".to_owned())
        }),
    }
}

fn execute_agg_to_string(args: &[String]) -> R<String> {
    if args.len() < 3 {
        return Err(
            "usage: vxg agg FRAME [--group-by GCOL] [--col ACOL] [--where EXPR] [--limit N] [--json] [--stream] [--no-mmap] [--threads N|auto]"
                .into(),
        );
    }
    let input = &args[1];
    let mut opts = DecodeOptions::default();
    let mut explicit_max_out = false;
    let mut stream_mode = false;
    let mut use_mmap = true;
    let mut group_spec: Option<String> = None;
    let mut agg_spec: Option<String> = None;
    let mut where_spec: Option<String> = None;
    let mut limit: usize = usize::MAX;
    let mut as_json = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--group-by" | "-g" => group_spec = Some(val(args, &mut i)?.to_owned()),
            "--col" | "-c" => agg_spec = Some(val(args, &mut i)?.to_owned()),
            "--where" => where_spec = Some(val(args, &mut i)?.to_owned()),
            "--limit" => limit = val(args, &mut i)?.parse()?,
            "--json" => as_json = true,
            "--threads" | "-t" => opts.threads = parse_threads(val(args, &mut i)?)?,
            "--no-checksum" | "--no-verify-checksum" => opts.verify_checksum = false,
            "--max-output" => {
                opts.max_output = val(args, &mut i)?.parse()?;
                explicit_max_out = true;
            }
            "--stream" => stream_mode = true,
            "--no-mmap" => use_mmap = false,
            x => return Err(format!("unknown agg option {x}").into()),
        }
        i += 1;
    }
    if group_spec.is_none() && agg_spec.is_none() {
        return Err("specify --col ACOL and/or --group-by GCOL for 'vxg agg'".into());
    }
    if !explicit_max_out {
        opts.max_output = usize::MAX;
    }

    let owned_where = where_spec.as_deref().map(parse_where_expr).transpose()?;
    let arena = PredArena::new();
    let where_ref = owned_where.as_ref().map(|e| arena.build(e));

    let mut out = String::new();
    if let Some(gs) = group_spec.as_deref() {
        let gcol = parse_col_target(gs)?;
        let acol = agg_spec.as_deref().map(parse_col_target).transpose()?;
        let groups = if stream_mode || input == "-" {
            let mut rdr = open_stream_reader(input)?;
            vxg::group_by_stream(&mut rdr, where_ref.as_ref(), gcol, acol, &opts)?
        } else {
            let frame = read_input_bytes(input, use_mmap)?;
            if let vxg::InputBytes::Mapped(ref m) = frame {
                m.advise_sequential();
            }
            vxg::group_by(&frame, where_ref.as_ref(), gcol, acol, &opts)?
        };
        for row in groups.iter().take(limit) {
            if as_json {
                let k_json = format_group_key_json(&row.key);
                if acol.is_some() {
                    out.push_str(&format!(
                        "{{\"key\":{k_json},{}}}\n",
                        format_agg_stats_json_fields(&row.stats)
                    ));
                } else {
                    out.push_str(&format!(
                        "{{\"key\":{k_json},\"rows\":{}}}\n",
                        row.stats.row_count
                    ));
                }
            } else {
                let k_disp = format_group_key_display(&row.key);
                if acol.is_some() {
                    out.push_str(&format!(
                        "{k_disp}\t{}\n",
                        format_agg_stats_text(&row.stats).replace(' ', "\t")
                    ));
                } else {
                    out.push_str(&format!("{k_disp}\trows={}\n", row.stats.row_count));
                }
            }
        }
    } else {
        let acol = parse_col_target(agg_spec.as_deref().unwrap())?;
        let st = if stream_mode || input == "-" {
            let mut rdr = open_stream_reader(input)?;
            vxg::aggregate_column_stream(&mut rdr, where_ref.as_ref(), acol, &opts)?
        } else {
            let frame = read_input_bytes(input, use_mmap)?;
            if let vxg::InputBytes::Mapped(ref m) = frame {
                m.advise_sequential();
            }
            vxg::aggregate_column(&frame, where_ref.as_ref(), acol, &opts)?
        };
        if as_json {
            out.push_str(&format!("{{{}}}\n", format_agg_stats_json_fields(&st)));
        } else {
            out.push_str(&format!("{}\n", format_agg_stats_text(&st)));
        }
    }
    Ok(out)
}

fn run_agg_cmd(args: &[String]) -> R<()> {
    let text = execute_agg_to_string(args)?;
    io::stdout().lock().write_all(text.as_bytes())?;
    Ok(())
}
