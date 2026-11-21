use std::io::Cursor;

use vxg::{
    compress, execute_sql, execute_sql_stream, execute_sql_table, DecodeOptions, Options,
    SqlFormat, SqlQuery,
};

fn build_ndjson_dataset() -> Vec<u8> {
    let mut raw = Vec::new();
    let services = ["auth", "billing", "search", "edge"];
    let regions = ["us-east", "eu-west", "ap-south"];
    for i in 0..240usize {
        let svc = services[i % services.len()];
        let reg = regions[(i / 4) % regions.len()];
        let user = format!("u{}", i % 15);
        let status = if i % 10 == 0 { 500 } else { 200 };
        let latency = 10 + (i % 50) * 5;
        let line = format!(
            "{{\"service\":\"{svc}\",\"region\":\"{reg}\",\"user\":\"{user}\",\"status\":{status},\"latency\":{latency}}}\n"
        );
        raw.extend_from_slice(line.as_bytes());
    }
    raw
}

fn assert_mem_eq_stream(frame: &[u8], sql: &str) -> String {
    let q = SqlQuery::parse(sql).unwrap_or_else(|e| panic!("parse failed for `{sql}`: {e}"));
    let dec = DecodeOptions::default();
    let mem = execute_sql(frame, &q, SqlFormat::Tsv, false, &dec).unwrap();
    let mut streamed = Vec::new();
    execute_sql_stream(
        Cursor::new(frame),
        &mut streamed,
        &q,
        SqlFormat::Tsv,
        false,
        &dec,
    )
    .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&mem),
        String::from_utf8_lossy(&streamed),
        "in-memory vs streaming mismatch for `{sql}`"
    );
    String::from_utf8(mem).unwrap()
}

#[test]
fn sql_distinct_count_distinct_having_and_offset_parity() {
    let raw = build_ndjson_dataset();
    // Build multi-frame stream (2 concatenated frames) to exercise streaming merge too!
    let half = raw.len() / 2;
    let split = raw[..half]
        .iter()
        .rposition(|&b| b == b'\n')
        .unwrap()
        + 1;
    let mut frame_single = compress(&raw, &Options::default()).unwrap();
    let mut multi_stream = compress(&raw[..split], &Options::default()).unwrap();
    multi_stream.extend_from_slice(&compress(&raw[split..], &Options::default()).unwrap());

    // 1. SELECT DISTINCT single & multi-column with ORDER BY + LIMIT + OFFSET
    let out_dist = assert_mem_eq_stream(
        &frame_single,
        "SELECT DISTINCT service FROM logs ORDER BY service ASC",
    );
    assert_eq!(out_dist, "service\nauth\nbilling\nedge\nsearch\n");

    let out_dist_page = assert_mem_eq_stream(
        &frame_single,
        "SELECT DISTINCT service FROM logs ORDER BY service ASC LIMIT 2 OFFSET 1",
    );
    assert_eq!(out_dist_page, "service\nbilling\nedge\n");

    // Multi-frame streaming parity for SELECT DISTINCT multi-col
    let q_multi_dist =
        SqlQuery::parse("SELECT DISTINCT service, region FROM logs ORDER BY service ASC, region ASC")
            .unwrap();
    let t_mem = execute_sql_table(&frame_single, &q_multi_dist, &DecodeOptions::default()).unwrap();
    assert_eq!(t_mem.rows.len(), 12);
    let mut s_out = Vec::new();
    execute_sql_stream(
        Cursor::new(&multi_stream),
        &mut s_out,
        &q_multi_dist,
        SqlFormat::Tsv,
        false,
        &DecodeOptions::default(),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(s_out).unwrap(),
        String::from_utf8(t_mem.to_bytes(SqlFormat::Tsv, false).unwrap()).unwrap()
    );

    // 2. Scalar COUNT(DISTINCT col) + combined COUNT(*), SUM(...) + multi-frame stream
    let out_cd = assert_mem_eq_stream(
        &frame_single,
        "SELECT COUNT(*) AS n, COUNT(DISTINCT service) AS svcs, COUNT(DISTINCT user) AS users FROM logs",
    );
    assert_eq!(out_cd, "n\tsvcs\tusers\n240\t4\t15\n");

    let q_cd = SqlQuery::parse(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT service) AS svcs, COUNT(DISTINCT user) AS users FROM logs",
    )
    .unwrap();
    let mut s_cd = Vec::new();
    execute_sql_stream(
        Cursor::new(&multi_stream),
        &mut s_cd,
        &q_cd,
        SqlFormat::Tsv,
        false,
        &DecodeOptions::default(),
    )
    .unwrap();
    assert_eq!(String::from_utf8(s_cd).unwrap(), out_cd);

    // 3. GROUP BY + COUNT(DISTINCT user) + HAVING (compound boolean + alias + aggregate) + ORDER BY + OFFSET
    let out_grp_having = assert_mem_eq_stream(
        &frame_single,
        "SELECT service, COUNT(*) AS cnt, COUNT(DISTINCT user) AS uniq_u, SUM(latency) AS tot \
         FROM logs WHERE status = 200 \
         GROUP BY service \
         HAVING cnt >= 45 AND uniq_u >= 10 AND NOT (service = 'edge') \
         ORDER BY tot DESC LIMIT 2 OFFSET 1",
    );
    let lines: Vec<&str> = out_grp_having.lines().collect();
    assert_eq!(lines[0], "service\tcnt\tuniq_u\ttot");
    assert_eq!(lines.len(), 3, "expected header + 2 paged rows, got:\n{out_grp_having}");
    assert!(!out_grp_having.contains("edge"), "HAVING NOT (service = 'edge') must exclude edge");

    // Multi-frame stream parity for grouped COUNT(DISTINCT) + HAVING
    let q_gh = SqlQuery::parse(
        "SELECT service, COUNT(*) AS cnt, COUNT(DISTINCT user) AS uniq_u, SUM(latency) AS tot \
         FROM logs WHERE status = 200 \
         GROUP BY service \
         HAVING cnt >= 45 AND uniq_u >= 10 \
         ORDER BY service ASC",
    )
    .unwrap();
    let mem_gh = execute_sql(&frame_single, &q_gh, SqlFormat::Tsv, false, &DecodeOptions::default()).unwrap();
    let mut st_gh = Vec::new();
    execute_sql_stream(
        Cursor::new(&multi_stream),
        &mut st_gh,
        &q_gh,
        SqlFormat::Tsv,
        false,
        &DecodeOptions::default(),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(mem_gh).unwrap(),
        String::from_utf8(st_gh).unwrap()
    );

    // 4. Scalar HAVING filtering out the single aggregate row when false
    let out_empty = assert_mem_eq_stream(
        &frame_single,
        "SELECT COUNT(*) AS n FROM logs HAVING n > 1000",
    );
    assert_eq!(out_empty, "n\n");

    // 5. Projection + ORDER BY + LIMIT + OFFSET parity
    let out_proj_all = assert_mem_eq_stream(
        &frame_single,
        "SELECT service, latency FROM logs WHERE status = 500 ORDER BY latency DESC LIMIT 5",
    );
    let out_proj_off = assert_mem_eq_stream(
        &frame_single,
        "SELECT service, latency FROM logs WHERE status = 500 ORDER BY latency DESC LIMIT 2 OFFSET 2",
    );
    let all_rows: Vec<&str> = out_proj_all.lines().collect();
    let off_rows: Vec<&str> = out_proj_off.lines().collect();
    assert_eq!(off_rows.len(), 3);
    assert_eq!(off_rows[1], all_rows[3]);
    assert_eq!(off_rows[2], all_rows[4]);

    // 6. SELECT * LIMIT + OFFSET parity
    let star_all = assert_mem_eq_stream(&frame_single, "SELECT * FROM logs LIMIT 5");
    let star_off = assert_mem_eq_stream(&frame_single, "SELECT * FROM logs LIMIT 2 OFFSET 2");
    let sa: Vec<&str> = star_all.lines().collect();
    let so: Vec<&str> = star_off.lines().collect();
    // Note: SELECT * with offset materializes SqlTable with "record" header, whereas offset=0 emits raw NDJSON lines
    assert_eq!(so.len(), 3);
    assert_eq!(so[1], sa[2]);
    assert_eq!(so[2], sa[3]);

    // Keep compiler happy if unused mut
    frame_single.clear();
}
