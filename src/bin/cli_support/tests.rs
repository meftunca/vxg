#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| (*x).to_owned()).collect()
    }

    #[test]
    fn cli_version_help_and_auto_threads_work() {
        assert!(run_args(&s(&["--version"])).is_ok());
        assert!(run_args(&s(&["--help"])).is_ok());
        assert!(parse_threads("auto").unwrap() >= 1);
        assert!(parse_threads("0").unwrap() >= 1);
        assert_eq!(parse_threads("4").unwrap(), 4);
    }

    #[test]
    fn cli_compress_verify_info_block_and_decompress_roundtrip() {
        let dir = std::env::temp_dir().join(format!("vxg_cli_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let raw_path = dir.join("input.csv");
        let vxg_path = dir.join("output.vxg");
        let dec_path = dir.join("decoded.csv");
        let blk_path = dir.join("block0.bin");

        let mut data = Vec::new();
        for i in 0..1024 {
            data.extend_from_slice(
                format!(
                    "\"2026-01-01T00:00:00Z\",{i},\"session.open\",\"eu-central-1\",\"worker-{}\"\n",
                    i % 8
                )
                .as_bytes(),
            );
        }
        fs::write(&raw_path, &data).unwrap();

        let raw_str = raw_path.to_str().unwrap();
        let vxg_str = vxg_path.to_str().unwrap();
        let dec_str = dec_path.to_str().unwrap();
        let blk_str = blk_path.to_str().unwrap();

        run_args(&s(&[
            "compress",
            raw_str,
            vxg_str,
            "--block-size",
            "4096",
            "--threads",
            "auto",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", vxg_str, "--threads", "auto", "-q"])).unwrap();
        run_args(&s(&["info", vxg_str, "--summary"])).unwrap();
        run_args(&s(&["block", vxg_str, "0", blk_str, "-f"])).unwrap();
        assert_eq!(fs::read(&blk_path).unwrap(), &data[..4096]);

        run_args(&s(&["decompress", vxg_str, dec_str, "-f", "-q"])).unwrap();
        assert_eq!(fs::read(&dec_path).unwrap(), data);

        run_args(&s(&[
            "compress",
            raw_str,
            vxg_str,
            "--block-size",
            "4096",
            "--stream",
            "-f",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", vxg_str, "-q"])).unwrap();
        run_args(&s(&[
            "decompress",
            vxg_str,
            dec_str,
            "--stream",
            "-f",
            "-q",
        ]))
        .unwrap();
        assert_eq!(fs::read(&dec_path).unwrap(), data);

        // Compressed-domain map -> update -> filter-frame -> verify -> decompress:
        let mapped_path = dir.join("mapped.vxg");
        let updated_path = dir.join("updated.vxg");
        let filtered_path = dir.join("filtered.vxg");
        let mapped_str = mapped_path.to_str().unwrap();
        let updated_str = updated_path.to_str().unwrap();
        let filtered_str = filtered_path.to_str().unwrap();

        run_args(&s(&["map", vxg_str, mapped_str, "--col", "csv:3", "--upper", "-q"])).unwrap();
        run_args(&s(&["verify", mapped_str, "-q"])).unwrap();
        run_args(&s(&[
            "update",
            mapped_str,
            updated_str,
            "--where",
            "csv:1>=1000",
            "--col",
            "csv:2",
            "--set",
            "session.tail",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", updated_str, "-q"])).unwrap();
        run_args(&s(&[
            "filter-frame",
            updated_str,
            filtered_str,
            "--where",
            "csv:2=session.tail",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", filtered_str, "-q"])).unwrap();
        assert_eq!(vxg::frame_row_count(&fs::read(&filtered_path).unwrap()).unwrap(), 24);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cli_compound_where_select_project_and_agg_end_to_end() {
        let dir = std::env::temp_dir().join(format!("vxg_cli_stage4_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let raw_path = dir.join("logs.ndjson");
        let vxg_path = dir.join("logs.vxg");
        let proj_path = dir.join("projected.vxg");
        let filt_path = dir.join("filtered_expr.vxg");

        let services = ["auth-api", "edge-proxy", "billing", "search"];
        let mut raw = Vec::new();
        for i in 0..1200usize {
            let svc = services[i % services.len()];
            let status = if i % 10 == 0 { 500 } else { 200 };
            let bytes = 100 + (i % 50) * 10;
            raw.extend_from_slice(
                format!(
                    "{{\"seq\":{i},\"service\":\"{svc}\",\"status\":{status},\"bytes\":{bytes},\"msg\":\"req-{i}\"}}\n"
                )
                .as_bytes(),
            );
        }
        fs::write(&raw_path, &raw).unwrap();
        let raw_s = raw_path.to_str().unwrap();
        let vxg_s = vxg_path.to_str().unwrap();
        let proj_s = proj_path.to_str().unwrap();
        let filt_s = filt_path.to_str().unwrap();

        run_args(&s(&[
            "compress",
            raw_s,
            vxg_s,
            "--block-size",
            "4096",
            "-t",
            "2",
            "-q",
        ]))
        .unwrap();

        // Compound `--where` count (`status==500 && (service=="auth-api" || service==billing)`):
        // i % 10 == 0 -> i % 4 is 0 ("auth-api", 60 rows) or 2 ("billing", 60 rows) => 120 rows!
        let cnt_out = execute_query_to_vec(&s(&[
            "query",
            vxg_s,
            "--count",
            "--where",
            "status==500 && (service==\"auth-api\" || service==billing) && !msg==req-0",
            "-t",
            "2",
        ]))
        .unwrap();
        assert_eq!(String::from_utf8(cnt_out).unwrap(), "119\n");

        // `query --select` with compound `--where` and `--limit`:
        let sel_out = execute_query_to_vec(&s(&[
            "query",
            vxg_s,
            "--select",
            "service,bytes,seq",
            "--where",
            "status==500 AND service==auth-api",
            "--limit",
            "3",
            "-t",
            "2",
        ]))
        .unwrap();
        assert_eq!(
            String::from_utf8(sel_out).unwrap(),
            "{\"service\":\"auth-api\",\"bytes\":100,\"seq\":0}\n\
             {\"service\":\"auth-api\",\"bytes\":300,\"seq\":20}\n\
             {\"service\":\"auth-api\",\"bytes\":500,\"seq\":40}\n"
        );

        // `vxg project` zero-reencode spliced frame + chained `verify`, `query`, and `agg`:
        run_args(&s(&[
            "project",
            vxg_s,
            proj_s,
            "--cols",
            "service,status,bytes",
            "-t",
            "2",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", proj_s, "-q"])).unwrap();
        let proj_cnt = execute_query_to_vec(&s(&[
            "query",
            proj_s,
            "--count",
            "--where",
            "status>=500 && NOT service==billing",
        ]))
        .unwrap();
        assert_eq!(String::from_utf8(proj_cnt).unwrap(), "60\n");

        // `vxg agg` scalar & group-by (both text and `--json`) on projected frame:
        let agg_json = execute_agg_to_string(&s(&[
            "agg",
            proj_s,
            "--group-by",
            "service",
            "--col",
            "bytes",
            "--where",
            "status==500",
            "--json",
            "-t",
            "2",
        ]))
        .unwrap();
        let lines: Vec<&str> = agg_json.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"key\":\"auth-api\",\"rows\":60,\"count\":60"));
        assert!(lines[1].contains("\"key\":\"billing\",\"rows\":60,\"count\":60"));

        let scalar_txt = execute_agg_to_string(&s(&[
            "agg",
            proj_s,
            "--col",
            "bytes",
            "--where",
            "status==500 && service==auth-api",
        ]))
        .unwrap();
        assert!(scalar_txt.starts_with("rows=60 count=60 "));

        // `vxg filter-frame` with compound boolean `--where`:
        run_args(&s(&[
            "filter-frame",
            vxg_s,
            filt_s,
            "--where",
            "(status==500 && service==auth-api) || seq<3",
            "-t",
            "2",
            "-q",
        ]))
        .unwrap();
        run_args(&s(&["verify", filt_s, "-q"])).unwrap();
        assert_eq!(
            vxg::frame_row_count(&fs::read(&filt_path).unwrap()).unwrap(),
            62
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
