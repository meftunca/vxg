//! Differential conformance tests for Phase B3:
//! 1. Automatic CSV/TSV header row detection & column-name-to-index resolution across single & multi-frame streams
//! 2. Dotted nested JSON path resolution (`actor.login`, `repo.name`, `payload.pr.merged`)
//! 3. Compressed-domain `SQL` (`SELECT`, `WHERE`, multi-agg, single & multi-key `GROUP BY`, `ORDER BY`, `LIMIT`, C-FFI)

use std::ffi::CString;
use std::io::Cursor;

use vxg::{
    aggregate_column, aggregate_column_stream, compress, count_where, count_where_stream,
    execute_sql, execute_sql_stream, execute_sql_table, filter_where, group_by, group_by_stream,
    parse_col_list, parse_col_target, select_columns, select_columns_stream, select_top_k_by,
    select_top_k_by_stream, top_k_by, top_k_by_stream, DecodeOptions, DecNum, GroupKey, Options,
    ParsedWhere, SortDir, SqlFormat, SqlQuery, SqlVal,
};

fn build_csv_corpus(rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * 64 + 64);
    out.extend_from_slice(b"VendorID,payment_type,fare_cents,tip_cents,total_cents,balance,zone\n");
    for i in 0..rows {
        let vendor = 1 + (i % 2);
        let pay = 1 + (i % 4);
        let fare = 800 + ((i * 37) % 4500);
        let tip = if pay == 1 { (i * 13) % 900 } else { 0 };
        let total = fare + tip + 250;
        let bal_cents: i64 = (i as i64 % 41) * 125 - 2000;
        let sign = if bal_cents < 0 { "-" } else { "" };
        let abs_b = bal_cents.unsigned_abs();
        let zone = match i % 4 {
            0 => "midtown",
            1 => "jfk_airport",
            2 => "downtown",
            _ => "uptown_west",
        };
        let line = format!(
            "{vendor},{pay},{fare},{tip},{total},{sign}{}.{abs_rem:02},{zone}\n",
            abs_b / 100,
            abs_rem = abs_b % 100
        );
        out.extend_from_slice(line.as_bytes());
    }
    out
}

fn build_concat_frames(raw: &[u8], chunk_bytes: usize, block_size: usize) -> Vec<u8> {
    let opts = Options {
        block_size,
        ..Options::default()
    };
    let mut stream = Vec::new();
    for chunk in raw.chunks(chunk_bytes) {
        stream.extend_from_slice(&compress(chunk, &opts).unwrap());
    }
    stream
}

#[test]
fn csv_header_auto_resolution_single_and_multiframe_stream() {
    let csv_raw = build_csv_corpus(2400);
    let single_frame = compress(
        &csv_raw,
        &Options {
            block_size: 16384,
            ..Options::default()
        },
    )
    .unwrap();
    let multi_frame = build_concat_frames(&csv_raw, 25_000, 8192);

    let dec1 = DecodeOptions {
        threads: 1,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };

    // Compare named CSV queries (`payment_type == 1 && total_cents >= 3000`) against explicit `csv:N` oracle (minus header row)
    let pw_named = ParsedWhere::parse("payment_type == 1 && total_cents >= 3000").unwrap();
    let pw_idx = ParsedWhere::parse("csv:1 == 1 && csv:4 >= 3000").unwrap();

    let expected_cnt = pw_idx
        .with_expr(|e| count_where(&single_frame, e, &dec1))
        .unwrap();
    assert!(expected_cnt > 100);

    for dec in [&dec1, &dec4] {
        let c_named = pw_named
            .with_expr(|e| count_where(&single_frame, e, dec))
            .unwrap();
        assert_eq!(c_named, expected_cnt);
        let c_stream = pw_named
            .with_expr(|e| count_where_stream(Cursor::new(&multi_frame), e, dec))
            .unwrap();
        assert_eq!(c_stream, expected_cnt);

        // Negated predicate `payment_type != 1` MUST NOT count row 0 (`"payment_type"`)!
        let pw_neq = ParsedWhere::parse("payment_type != 1").unwrap();
        let c_neq = pw_neq
            .with_expr(|e| count_where(&single_frame, e, dec))
            .unwrap();
        let pw_eq1 = ParsedWhere::parse("payment_type == 1").unwrap();
        let c_eq1 = pw_eq1
            .with_expr(|e| count_where(&single_frame, e, dec))
            .unwrap();
        assert_eq!(c_neq + c_eq1, 2400);

        // Scalar aggregation by CSV column name (`balance` signed decimal + `total_cents`)
        let st_bal = aggregate_column(
            &single_frame,
            None,
            parse_col_target("balance").unwrap(),
            dec,
        )
        .unwrap();
        assert_eq!(st_bal.row_count, 2400);
        assert_eq!(st_bal.num_count, 2400);
        let st_bal_stream = aggregate_column_stream(
            Cursor::new(&multi_frame),
            None,
            parse_col_target("balance").unwrap(),
            dec,
        )
        .unwrap();
        assert_eq!(st_bal_stream, st_bal);

        // GROUP BY by CSV column name (`payment_type` + `total_cents`) — must have exact 4 groups (`1..=4`), no `"payment_type"` header group!
        let grp = group_by(
            &single_frame,
            None,
            parse_col_target("payment_type").unwrap(),
            Some(parse_col_target("total_cents").unwrap()),
            dec,
        )
        .unwrap();
        assert_eq!(grp.len(), 4);
        assert_eq!(grp.iter().map(|g| g.stats.row_count).sum::<u64>(), 2400);
        let grp_stream = group_by_stream(
            Cursor::new(&multi_frame),
            None,
            parse_col_target("payment_type").unwrap(),
            Some(parse_col_target("total_cents").unwrap()),
            dec,
        )
        .unwrap();
        assert_eq!(grp_stream, grp);

        // Projection & Top-K by CSV column names
        let cols = parse_col_list("VendorID,total_cents,zone").unwrap();
        let proj = select_columns(&single_frame, None, &cols, 5, dec).unwrap();
        let proj_str = String::from_utf8(proj.clone()).unwrap();
        assert!(!proj_str.starts_with("VendorID,"));
        assert_eq!(proj_str.lines().count(), 5);
        let mut proj_stream = Vec::new();
        let written =
            select_columns_stream(Cursor::new(&multi_frame), &mut proj_stream, None, &cols, 5, dec)
                .unwrap();
        assert_eq!(written, 5);
        assert_eq!(proj_stream, proj);

        let top5 = top_k_by(
            &single_frame,
            None,
            parse_col_target("total_cents").unwrap(),
            SortDir::Desc,
            5,
            dec,
        )
        .unwrap();
        assert_eq!(top5.len(), 5);
        assert!(top5.iter().all(|m| m.global_row_index >= 1));
        let top5_stream = top_k_by_stream(
            Cursor::new(&multi_frame),
            None,
            parse_col_target("total_cents").unwrap(),
            SortDir::Desc,
            5,
            dec,
        )
        .unwrap();
        assert_eq!(
            top5_stream.iter().map(|m| &m.bytes).collect::<Vec<_>>(),
            top5.iter().map(|m| &m.bytes).collect::<Vec<_>>()
        );

        let sel_top = select_top_k_by(
            &single_frame,
            None,
            &cols,
            parse_col_target("balance").unwrap(),
            SortDir::Asc,
            4,
            dec,
        )
        .unwrap();
        let mut sel_top_stream = Vec::new();
        select_top_k_by_stream(
            Cursor::new(&multi_frame),
            &mut sel_top_stream,
            None,
            &cols,
            parse_col_target("balance").unwrap(),
            SortDir::Asc,
            4,
            dec,
        )
        .unwrap();
        assert_eq!(sel_top_stream, sel_top);
    }
}

#[test]
fn dotted_nested_json_paths_and_sql_queries() {
    let mut ndjson = Vec::new();
    for i in 0..1200usize {
        let login = match i % 3 {
            0 => "alice",
            1 => "bob",
            _ => "charlie",
        };
        let repo = if i % 2 == 0 { "org/vxg" } else { "org/core" };
        let ev_type = if i % 5 == 0 {
            "PullRequestEvent"
        } else {
            "PushEvent"
        };
        let additions: i64 = (i as i64 * 7) - 500;
        let line = format!(
            "{{\"id\":{},\"type\":\"{ev_type}\",\"actor\":{{\"id\":{},\"login\":\"{login}\"}},\"repo\":{{\"id\":{},\"name\":\"{repo}\"}},\"payload\":{{\"delta\":{additions}}}}}\n",
            100_000 + i,
            10 + (i % 3),
            500 + (i % 2),
        );
        ndjson.extend_from_slice(line.as_bytes());
    }

    let frame = compress(
        &ndjson,
        &Options {
            block_size: 16384,
            ..Options::default()
        },
    )
    .unwrap();
    let dec4 = DecodeOptions {
        threads: 4,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };

    let pw = ParsedWhere::parse("actor.login == 'alice' && repo.name == 'org/vxg'").unwrap();
    let cnt = pw.with_expr(|e| count_where(&frame, e, &dec4)).unwrap();
    // i % 3 == 0 AND i % 2 == 0 => i % 6 == 0 => 1200 / 6 = 200
    assert_eq!(cnt, 200);

    let matches = pw.with_expr(|e| filter_where(&frame, e, 3, &dec4)).unwrap();
    assert_eq!(matches.len(), 3);

    let grp = group_by(
        &frame,
        None,
        parse_col_target("actor.login").unwrap(),
        Some(parse_col_target("payload.delta").unwrap()),
        &dec4,
    )
    .unwrap();
    assert_eq!(grp.len(), 3);
    assert_eq!(grp[0].key, GroupKey::Str(b"alice".to_vec()));
    assert_eq!(grp[0].stats.row_count, 400);

    // SQL query over nested JSON paths with multi-key GROUP BY + ORDER BY + JSON formatting
    let q = SqlQuery::parse(
        "SELECT actor.login AS user, repo.name AS repo, COUNT(*) AS events, SUM(payload.delta) AS net_delta
         FROM gharchive
         WHERE type = 'PushEvent' AND actor.login IN ('alice', 'bob')
         GROUP BY 1, 2
         ORDER BY net_delta DESC",
    )
    .unwrap();
    let tbl = execute_sql_table(&frame, &q, &dec4).unwrap();
    assert_eq!(tbl.headers, vec!["user", "repo", "events", "net_delta"]);
    assert_eq!(tbl.rows.len(), 4);

    let json_bytes = execute_sql(&frame, &q, SqlFormat::Json, false, &dec4).unwrap();
    let json_text = String::from_utf8(json_bytes).unwrap();
    assert_eq!(json_text.lines().count(), 4);
    assert!(json_text.contains("\"user\":"));
    assert!(json_text.contains("\"repo\":\"org/vxg\""));

    // Non-aggregate projection (unordered and ORDER BY) over flattened nested JSON paths
    let q_proj = SqlQuery::parse(
        "SELECT actor.login, repo.name, payload.delta FROM gharchive WHERE type = 'PushEvent' LIMIT 4",
    )
    .unwrap();
    let tbl_proj = execute_sql_table(&frame, &q_proj, &dec4).unwrap();
    assert_eq!(tbl_proj.rows.len(), 4);
    assert_eq!(tbl_proj.rows[0][0], vxg::SqlVal::Str("bob".to_owned()));
    assert_eq!(tbl_proj.rows[0][1], vxg::SqlVal::Str("org/core".to_owned()));

    let q_proj_ord = SqlQuery::parse(
        "SELECT actor.login, repo.name, payload.delta FROM gharchive WHERE type = 'PushEvent' ORDER BY payload.delta DESC LIMIT 3",
    )
    .unwrap();
    let tbl_proj_ord = execute_sql_table(&frame, &q_proj_ord, &dec4).unwrap();
    assert_eq!(tbl_proj_ord.rows.len(), 3);
    assert_eq!(
        tbl_proj_ord.rows[0][0],
        vxg::SqlVal::Str("charlie".to_owned())
    );
    assert_eq!(
        tbl_proj_ord.rows[0][1],
        vxg::SqlVal::Str("org/core".to_owned())
    );
}

#[test]
fn sql_engine_multi_agg_multikey_groupby_stream_and_ffi() {
    let csv_raw = build_csv_corpus(3000);
    let csv_frame = compress(
        &csv_raw,
        &Options {
            block_size: 16384,
            ..Options::default()
        },
    )
    .unwrap();
    let csv_multi = build_concat_frames(&csv_raw, 28_000, 8192);

    let dec1 = DecodeOptions {
        threads: 1,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };

    // 1. SELECT COUNT(*) on CSV must exclude header line -> exact 3000!
    let q_cnt = SqlQuery::parse("SELECT COUNT(*) AS n FROM taxi;").unwrap();
    let t_cnt = execute_sql_table(&csv_frame, &q_cnt, &dec4).unwrap();
    assert_eq!(t_cnt.rows, vec![vec![SqlVal::Int(3000)]]);

    // 2. Empty result set (`WHERE 1 = 0`) -> COUNT(*) = 0, SUM = NULL!
    let q_empty =
        SqlQuery::parse("SELECT COUNT(*), COUNT(fare_cents), SUM(fare_cents) WHERE 1 = 0").unwrap();
    let t_empty = execute_sql_table(&csv_frame, &q_empty, &dec4).unwrap();
    assert_eq!(
        t_empty.rows,
        vec![vec![SqlVal::Int(0), SqlVal::Int(0), SqlVal::Null]]
    );

    // 3. Multi-column scalar aggregation (`fare_cents`, `tip_cents`, `balance`)
    let q_multi_agg = SqlQuery::parse(
        "SELECT COUNT(*), SUM(fare_cents), SUM(tip_cents), MIN(balance), MAX(balance)
         FROM taxi
         WHERE payment_type = 1 AND zone LIKE '%airport%'",
    )
    .unwrap();
    let t_agg_1 = execute_sql_table(&csv_frame, &q_multi_agg, &dec1).unwrap();
    let t_agg_4 = execute_sql_table(&csv_frame, &q_multi_agg, &dec4).unwrap();
    assert_eq!(t_agg_1, t_agg_4);

    let out_single = execute_sql(&csv_frame, &q_multi_agg, SqlFormat::Tsv, false, &dec4).unwrap();
    let mut out_stream = Vec::new();
    execute_sql_stream(
        Cursor::new(&csv_multi),
        &mut out_stream,
        &q_multi_agg,
        SqlFormat::Tsv,
        false,
        &dec4,
    )
    .unwrap();
    assert_eq!(out_stream, out_single);

    // 4. Multi-key, multi-agg GROUP BY (`VendorID, payment_type` + `SUM(fare_cents), SUM(tip_cents)`)
    let q_grp = SqlQuery::parse(
        "SELECT VendorID, payment_type, COUNT(*) AS trips, SUM(fare_cents) AS fare, SUM(tip_cents) AS tip, SUM(balance) AS bal
         FROM taxi
         WHERE total_cents BETWEEN 1200 AND 5500
         GROUP BY VendorID, payment_type
         ORDER BY fare DESC, VendorID ASC
         LIMIT 6",
    )
    .unwrap();
    let out_grp_1 = execute_sql(&csv_frame, &q_grp, SqlFormat::Csv, false, &dec1).unwrap();
    let out_grp_4 = execute_sql(&csv_frame, &q_grp, SqlFormat::Csv, false, &dec4).unwrap();
    assert_eq!(out_grp_1, out_grp_4);

    let mut out_grp_stream = Vec::new();
    execute_sql_stream(
        Cursor::new(&csv_multi),
        &mut out_grp_stream,
        &q_grp,
        SqlFormat::Csv,
        false,
        &dec4,
    )
    .unwrap();
    assert_eq!(out_grp_stream, out_grp_4);

    // Verify C-FFI `vxg_sql_alloc` matches `execute_sql` byte-for-byte
    let c_sql = CString::new(
        "SELECT VendorID, payment_type, COUNT(*) AS trips, SUM(fare_cents) AS fare, SUM(tip_cents) AS tip, SUM(balance) AS bal FROM taxi WHERE total_cents BETWEEN 1200 AND 5500 GROUP BY VendorID, payment_type ORDER BY fare DESC, VendorID ASC LIMIT 6",
    )
    .unwrap();
    let mut ffi_len = 0usize;
    let ptr = unsafe {
        vxg::ffi::vxg_sql_alloc(
            csv_frame.as_ptr(),
            csv_frame.len(),
            c_sql.as_ptr(),
            1, // Csv
            0, // with header
            4,
            &mut ffi_len,
        )
    };
    assert!(!ptr.is_null());
    let ffi_bytes = unsafe { std::slice::from_raw_parts(ptr, ffi_len) }.to_vec();
    unsafe {
        vxg::ffi::vxg_free(ptr, ffi_len);
    }
    assert_eq!(ffi_bytes, out_grp_4);

    // Verify signed decimal precision in `SUM(balance)`
    let tbl_grp = execute_sql_table(&csv_frame, &q_grp, &dec4).unwrap();
    assert!(!tbl_grp.rows.is_empty());
    assert!(matches!(
        tbl_grp.rows[0][5],
        SqlVal::Dec(DecNum { .. }) | SqlVal::Int(_)
    ));
}
