fn execute_sql_to_vec(args: &[String]) -> R<Vec<u8>> {
    if args.len() < 3 {
        return Err(
            "usage: vxg sql FRAME \"SELECT ...\" [--json|--csv] [--no-header] [--stream] [--no-mmap] [--threads N|auto]"
                .into(),
        );
    }
    let input = &args[1];
    let mut query_str: Option<&str> = None;
    let mut format = vxg::SqlFormat::Tsv;
    let mut no_header = false;
    let mut stream_mode = false;
    let mut use_mmap = true;
    let mut dec = DecodeOptions {
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => format = vxg::SqlFormat::Json,
            "--csv" => format = vxg::SqlFormat::Csv,
            "--tsv" => format = vxg::SqlFormat::Tsv,
            "--no-header" | "-H" => no_header = true,
            "--stream" => stream_mode = true,
            "--no-mmap" => use_mmap = false,
            "--threads" | "-t" => dec.threads = parse_threads(val(args, &mut i)?)?,
            "--simd" => dec.simd = val(args, &mut i)?.parse()?,
            "--max-output" => dec.max_output = val(args, &mut i)?.parse()?,
            "--no-checksum" | "--no-verify-checksum" => dec.verify_checksum = false,
            "--query" | "-q" | "-e" => query_str = Some(val(args, &mut i)?),
            s if !s.starts_with('-') && query_str.is_none() => {
                query_str = Some(s);
            }
            x => return Err(format!("unknown option '{x}' for 'vxg sql'").into()),
        }
        i += 1;
    }

    let Some(sql) = query_str else {
        return Err("missing SQL query string (usage: vxg sql FRAME \"SELECT ...\")".into());
    };
    let parsed = vxg::SqlQuery::parse(sql)?;

    if stream_mode || input == "-" {
        let mut rdr = open_stream_reader(input)?;
        let mut out = Vec::new();
        vxg::execute_sql_stream(&mut rdr, &mut out, &parsed, format, no_header, &dec)?;
        return Ok(out);
    }

    let bytes = read_input_bytes(input, use_mmap)?;
    if let vxg::InputBytes::Mapped(ref m) = bytes {
        m.advise_sequential();
    }
    Ok(vxg::execute_sql(&bytes, &parsed, format, no_header, &dec)?)
}

fn run_sql_cmd(args: &[String]) -> R<()> {
    let out = execute_sql_to_vec(args)?;
    let mut stdout = io::stdout().lock();
    stdout.write_all(&out)?;
    stdout.flush()?;
    Ok(())
}

#[cfg(test)]
mod sql_cli_tests {
    use super::*;

    #[test]
    fn cli_sql_csv_header_and_stream_end_to_end() {
        let dir = std::env::temp_dir().join(format!("vxg_cli_sql_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("sales.csv");
        let sgl_vxg = dir.join("sales_single.vxg");
        let stm_vxg = dir.join("sales_stream.vxg");

        let mut csv = String::from("region,payment_type,amount\n");
        for i in 0..1000usize {
            let reg = if i % 2 == 0 { "eu" } else { "us" };
            let pt = (i % 3) + 1;
            let amt = format!("{}.{:02}", (i % 20) - 5, i % 100);
            csv.push_str(&format!("{reg},{pt},{amt}\n"));
        }
        fs::write(&csv_path, csv.as_bytes()).unwrap();
        let csv_s = csv_path.to_str().unwrap();
        let sgl_s = sgl_vxg.to_str().unwrap();
        let stm_s = stm_vxg.to_str().unwrap();
        let sv = |xs: &[&str]| xs.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();

        run_args(&sv(&["compress", csv_s, sgl_s, "--block-size", "4096", "-q"])).unwrap();
        run_args(&sv(&[
            "compress", csv_s, stm_s, "--block-size", "4096", "--stream", "-q",
        ]))
        .unwrap();

        let q = "SELECT region, payment_type, COUNT(*), SUM(amount) WHERE amount > 0 GROUP BY region, payment_type ORDER BY SUM(amount) DESC LIMIT 4";
        let out_sgl = String::from_utf8(
            execute_sql_to_vec(&sv(&["sql", sgl_s, q, "--json", "-t", "2"])).unwrap(),
        )
        .unwrap();
        let out_stm = String::from_utf8(
            execute_sql_to_vec(&sv(&["sql", stm_s, q, "--json", "--stream", "-t", "2"])).unwrap(),
        )
        .unwrap();
        assert_eq!(out_sgl, out_stm);
        assert_eq!(out_sgl.lines().count(), 4);
        assert!(out_sgl.contains("\"region\":\""));

        let _ = fs::remove_dir_all(&dir);
    }
}
