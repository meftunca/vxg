fn format_agg_stats_text(st: &vxg::AggStats) -> String {
    if let Some(avg) = st.avg() {
        let sum_d = st.sum_dec();
        let min_d = st.min_dec().unwrap_or(vxg::DecNum::from_u64(0));
        let max_d = st.max_dec().unwrap_or(vxg::DecNum::from_u64(0));
        format!(
            "rows={} count={} sum={sum_d} avg={avg:.4} min={min_d} max={max_d}",
            st.row_count,
            st.num_count,
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
        let sum_d = st.sum_dec();
        let min_d = st.min_dec().unwrap_or(vxg::DecNum::from_u64(0));
        let max_d = st.max_dec().unwrap_or(vxg::DecNum::from_u64(0));
        format!(
            "\"rows\":{},\"count\":{},\"sum\":{sum_d},\"avg\":{avg:.6},\"min\":{min_d},\"max\":{max_d}",
            st.row_count,
            st.num_count,
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

#[derive(Clone, Copy)]
enum AggSortField {
    Key,
    Rows,
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

fn parse_agg_sort_field(s: &str) -> R<AggSortField> {
    match s.to_ascii_lowercase().as_str() {
        "key" => Ok(AggSortField::Key),
        "rows" | "row_count" => Ok(AggSortField::Rows),
        "count" | "num_count" => Ok(AggSortField::Count),
        "sum" => Ok(AggSortField::Sum),
        "avg" | "mean" => Ok(AggSortField::Avg),
        "min" => Ok(AggSortField::Min),
        "max" => Ok(AggSortField::Max),
        _ => Err("expected --order-by key|rows|count|sum|avg|min|max for 'vxg agg'".into()),
    }
}

fn cmp_opt_nulls_last<T: Ord>(a: Option<T>, b: Option<T>, dir: vxg::SortDir) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => {
            let ord = x.cmp(&y);
            if dir == vxg::SortDir::Asc {
                ord
            } else {
                ord.reverse()
            }
        }
    }
}

fn sort_group_rows(rows: &mut [vxg::GroupRow], field: AggSortField, dir: vxg::SortDir) {
    use std::cmp::Ordering;
    rows.sort_by(|a, b| {
        let primary = match field {
            AggSortField::Key => {
                let ord = a.key.cmp(&b.key);
                if dir == vxg::SortDir::Asc {
                    ord
                } else {
                    ord.reverse()
                }
            }
            AggSortField::Rows => {
                let ord = a.stats.row_count.cmp(&b.stats.row_count);
                if dir == vxg::SortDir::Asc {
                    ord
                } else {
                    ord.reverse()
                }
            }
            AggSortField::Count => {
                let ord = a.stats.num_count.cmp(&b.stats.num_count);
                if dir == vxg::SortDir::Asc {
                    ord
                } else {
                    ord.reverse()
                }
            }
            AggSortField::Sum => {
                let ord = a.stats.sum_dec().cmp(&b.stats.sum_dec());
                if dir == vxg::SortDir::Asc {
                    ord
                } else {
                    ord.reverse()
                }
            }
            AggSortField::Avg => match (a.stats.avg(), b.stats.avg()) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(x), Some(y)) => {
                    let ord = x.total_cmp(&y);
                    if dir == vxg::SortDir::Asc {
                        ord
                    } else {
                        ord.reverse()
                    }
                }
            },
            AggSortField::Min => cmp_opt_nulls_last(a.stats.min_dec(), b.stats.min_dec(), dir),
            AggSortField::Max => cmp_opt_nulls_last(a.stats.max_dec(), b.stats.max_dec(), dir),
        };
        primary.then_with(|| a.key.cmp(&b.key))
    });
}

fn execute_agg_to_string(args: &[String]) -> R<String> {
    if args.len() < 3 {
        return Err(
            "usage: vxg agg FRAME [--group-by GCOL] [--col ACOL] [--where EXPR] [--order-by FIELD [--desc|--asc]] [--limit N] [--json] [--stream] [--no-mmap] [--threads N|auto]"
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
    let mut order_field: Option<AggSortField> = None;
    let mut sort_dir = vxg::SortDir::Asc;
    let mut limit: usize = usize::MAX;
    let mut as_json = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--group-by" | "-g" => group_spec = Some(val(args, &mut i)?.to_owned()),
            "--col" | "-c" => agg_spec = Some(val(args, &mut i)?.to_owned()),
            "--where" => where_spec = Some(val(args, &mut i)?.to_owned()),
            "--order-by" | "--sort-by" => {
                order_field = Some(parse_agg_sort_field(val(args, &mut i)?)?);
            }
            "--desc" => sort_dir = vxg::SortDir::Desc,
            "--asc" => sort_dir = vxg::SortDir::Asc,
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

    let pw = where_spec
        .as_deref()
        .map(vxg::ParsedWhere::parse)
        .transpose()?;

    let mut out = String::new();
    if let Some(gs) = group_spec.as_deref() {
        let gcol = vxg::parse_col_target(gs)?;
        let acol = agg_spec.as_deref().map(vxg::parse_col_target).transpose()?;
        let mut groups = if stream_mode || input == "-" {
            let mut rdr = open_stream_reader(input)?;
            vxg::with_opt_where(pw.as_ref(), |where_ref| {
                vxg::group_by_stream(&mut rdr, where_ref, gcol, acol, &opts)
            })?
        } else {
            let frame = read_input_bytes(input, use_mmap)?;
            if let vxg::InputBytes::Mapped(ref m) = frame {
                m.advise_sequential();
            }
            vxg::with_opt_where(pw.as_ref(), |where_ref| {
                vxg::group_by(&frame, where_ref, gcol, acol, &opts)
            })?
        };
        if let Some(sf) = order_field {
            sort_group_rows(&mut groups, sf, sort_dir);
        }
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
        let acol = vxg::parse_col_target(agg_spec.as_deref().unwrap())?;
        let st = if stream_mode || input == "-" {
            let mut rdr = open_stream_reader(input)?;
            vxg::with_opt_where(pw.as_ref(), |where_ref| {
                vxg::aggregate_column_stream(&mut rdr, where_ref, acol, &opts)
            })?
        } else {
            let frame = read_input_bytes(input, use_mmap)?;
            if let vxg::InputBytes::Mapped(ref m) = frame {
                m.advise_sequential();
            }
            vxg::with_opt_where(pw.as_ref(), |where_ref| {
                vxg::aggregate_column(&frame, where_ref, acol, &opts)
            })?
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
