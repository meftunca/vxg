use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use vxg::{
    compress, decompress, execute_sql, execute_sql_stream, extract_row, serve_http_range_listener,
    DecodeOptions, HttpRangeReader, Options, RemoteFrame, SliceRangeReader, SqlFormat, SqlQuery,
};

fn build_ndjson_dataset(rows: usize) -> Vec<u8> {
    let regions = ["eu", "us", "ap", "sa"];
    let cats = ["alpha", "beta", "gamma", "delta", "omega"];
    let mut out = Vec::with_capacity(rows * 96);
    for i in 0..rows {
        let reg = regions[i % regions.len()];
        let cat = cats[(i * 7 + (i / 11)) % cats.len()];
        let amt = (i % 97) * 3 + 1;
        let note = if i % 5 == 0 {
            "null".to_owned()
        } else {
            format!("\"tag-{}\"", i % 13)
        };
        out.extend_from_slice(
            format!(
                "{{\"id\":{i},\"region\":\"{reg}\",\"category\":\"{cat}\",\"amount\":{amt},\"note\":{note}}}\n"
            )
            .as_bytes(),
        );
    }
    out
}

fn build_csv_dataset(rows: usize) -> Vec<u8> {
    let regions = ["eu", "us", "ap"];
    let mut out = String::from("id,region,amount,note\n");
    for i in 0..rows {
        let reg = regions[i % regions.len()];
        let amt = (i % 50) + 1;
        let note = if i % 4 == 0 { "" } else { "ok" };
        out.push_str(&format!("{i},{reg},{amt},{note}\n"));
    }
    out.into_bytes()
}

fn encode_indexed_v5(raw: &[u8]) -> Vec<u8> {
    let opts = Options {
        block_size: 4096,
        record_aligned: true,
        row_index: true,
        checksum: true,
        ..Options::default()
    };
    compress(raw, &opts).unwrap()
}

#[test]
fn sql_is_null_and_is_not_null_ndjson_and_csv() {
    let dec = DecodeOptions::default();

    // 1) NDJSON: every 5th row has `"note":null` -> 40 null, 160 non-null out of 200 rows.
    let nd_raw = build_ndjson_dataset(200);
    let nd_vxg = encode_indexed_v5(&nd_raw);

    let q_null = SqlQuery::parse("SELECT COUNT(*) WHERE note IS NULL").unwrap();
    let res_null =
        String::from_utf8(execute_sql(&nd_vxg, &q_null, SqlFormat::Tsv, true, &dec).unwrap())
            .unwrap();
    assert_eq!(res_null.trim(), "40");

    let q_not_null = SqlQuery::parse("SELECT COUNT(*) WHERE note IS NOT NULL").unwrap();
    let res_nn = String::from_utf8(
        execute_sql(&nd_vxg, &q_not_null, SqlFormat::Tsv, true, &dec).unwrap(),
    )
    .unwrap();
    assert_eq!(res_nn.trim(), "160");

    // Combined with AND + GROUP BY + streaming parity
    let q_combo = SqlQuery::parse(
        "SELECT region, COUNT(*) WHERE note IS NOT NULL AND amount >= 100 GROUP BY region ORDER BY region ASC",
    )
    .unwrap();
    let mem = execute_sql(&nd_vxg, &q_combo, SqlFormat::Json, false, &dec).unwrap();
    let mut stm = Vec::new();
    execute_sql_stream(&nd_vxg[..], &mut stm, &q_combo, SqlFormat::Json, false, &dec).unwrap();
    assert_eq!(mem, stm);
    assert!(!mem.is_empty());

    // 2) CSV: every 4th row has empty note -> 30 null, 90 non-null out of 120 rows.
    let csv_raw = build_csv_dataset(120);
    let csv_vxg = encode_indexed_v5(&csv_raw);

    let c_null = String::from_utf8(
        execute_sql(&csv_vxg, &q_null, SqlFormat::Tsv, true, &dec).unwrap(),
    )
    .unwrap();
    assert_eq!(c_null.trim(), "30");

    let c_nn = String::from_utf8(
        execute_sql(&csv_vxg, &q_not_null, SqlFormat::Tsv, true, &dec).unwrap(),
    )
    .unwrap();
    assert_eq!(c_nn.trim(), "90");
}

#[test]
fn sql_explain_reports_plan_and_row_index_fast_path() {
    let dec = DecodeOptions::default();
    let raw = build_ndjson_dataset(300);
    let frame = encode_indexed_v5(&raw);

    let q_fast = SqlQuery::parse("EXPLAIN SELECT COUNT(*)").unwrap();
    assert!(q_fast.explain);
    let txt_fast =
        String::from_utf8(execute_sql(&frame, &q_fast, SqlFormat::Tsv, false, &dec).unwrap())
            .unwrap();
    assert!(txt_fast.starts_with("stage\tdetail\n"));
    assert!(txt_fast.contains("O(1) FLAG_ROW_INDEX header count"));
    assert!(txt_fast.contains("archive\tv5"));

    let q_agg = SqlQuery::parse(
        "EXPLAIN SELECT DISTINCT region, COUNT(DISTINCT category) WHERE note IS NOT NULL GROUP BY region HAVING COUNT(*) > 2 ORDER BY region DESC LIMIT 3 OFFSET 1",
    )
    .unwrap();
    let txt_agg =
        String::from_utf8(execute_sql(&frame, &q_agg, SqlFormat::Tsv, false, &dec).unwrap())
            .unwrap();
    assert!(txt_agg.contains("operator\tCompressedGroupBy"));
    assert!(txt_agg.contains("distinct\ttrue"));
    assert!(txt_agg.contains("where\tcompressed_domain_pushdown"));
    assert!(txt_agg.contains("having\tpost_agg_filter"));
    assert!(txt_agg.contains("slice\tlimit=Some(3) offset=1"));

    // Streaming EXPLAIN works cleanly too
    let mut stm = Vec::new();
    execute_sql_stream(&frame[..], &mut stm, &q_agg, SqlFormat::Json, false, &dec).unwrap();
    let stm_s = String::from_utf8(stm).unwrap();
    assert!(stm_s.contains("\"stage\":\"operator\""));
    assert!(stm_s.contains("\"detail\":\"CompressedGroupBy\""));
}

#[test]
fn remote_frame_slice_random_access_and_minimal_byte_transfer() {
    let dec = DecodeOptions::default();
    let raw = build_ndjson_dataset(2500);
    let frame = encode_indexed_v5(&raw);
    let mut rf = RemoteFrame::open(SliceRangeReader(&frame)).unwrap();

    let info = rf.info();
    assert_eq!(info.version, 5);
    assert!(info.has_row_index);
    assert_eq!(info.row_count, Some(2500));
    assert!(info.block_count >= 12);
    let open_bytes = rf.stats().bytes_fetched;
    assert!(open_bytes < (frame.len() as u64) / 3);

    // Random row seek in middle & tail: fetches strictly 1 block per call
    for &idx in &[0usize, 1234, 2499] {
        let before_reqs = rf.stats().requests;
        let got = rf.extract_row(idx, &dec).unwrap().unwrap();
        let expected = extract_row(&frame, idx, &dec).unwrap();
        assert_eq!(got, expected);
        assert_eq!(rf.stats().requests, before_reqs + 1);
    }
    assert!(rf.extract_row(2500, &dec).unwrap().is_none());

    // Single-block extract matches full decompress slice
    let blk2 = rf.extract_block(2, &dec).unwrap();
    let full = decompress(&frame, &dec).unwrap();
    assert!(full.windows(blk2.len()).any(|w| w == blk2.as_slice()));
}

#[test]
fn live_http_range_client_and_sql_endpoint_end_to_end() {
    let dec = DecodeOptions::default();
    let raw = build_ndjson_dataset(3000);
    let frame = encode_indexed_v5(&raw);

    let tmp_dir = std::env::temp_dir().join(format!("vxg_remote_http_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let vxg_path = tmp_dir.join("events.vxg");
    std::fs::write(&vxg_path, &frame).unwrap();

    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let addr = listener.local_addr().unwrap();
    let serve_dir = tmp_dir.clone();
    std::thread::spawn(move || {
        let _ = serve_http_range_listener(listener, &serve_dir);
    });

    let url = format!("http://{addr}/events.vxg");

    // 1) Remote O(log B) single-row seek fetches a tiny fraction of the file
    {
        let mut rf = RemoteFrame::open(HttpRangeReader::connect(&url).unwrap()).unwrap();
        let row = rf.extract_row(1877, &dec).unwrap().unwrap();
        assert_eq!(row, extract_row(&frame, 1877, &dec).unwrap());
        let st = rf.stats();
        assert!(
            st.fetched_pct() < 25.0,
            "expected <25% fetched for single-row seek, got {:.2}% ({}B/{}B)",
            st.fetched_pct(),
            st.bytes_fetched,
            st.total_bytes
        );
    }

    // 2) Remote SELECT COUNT(*) uses FLAG_ROW_INDEX O(1) fast path (at most 1 header-peek block)
    {
        let mut rf = RemoteFrame::open(HttpRangeReader::connect(&url).unwrap()).unwrap();
        let q = SqlQuery::parse("SELECT COUNT(*)").unwrap();
        let got = rf.execute_sql(&q, SqlFormat::Tsv, true, &dec).unwrap();
        assert_eq!(String::from_utf8(got).unwrap().trim(), "3000");
        assert!(
            rf.stats().fetched_pct() < 25.0,
            "COUNT(*) fetched {:.2}%",
            rf.stats().fetched_pct()
        );
    }

    // 3) Remote LIMIT scan stops early without fetching the rest of the archive
    {
        let mut rf = RemoteFrame::open(HttpRangeReader::connect(&url).unwrap()).unwrap();
        let q = SqlQuery::parse("SELECT id, region, amount LIMIT 5").unwrap();
        let got = rf.execute_sql(&q, SqlFormat::Tsv, false, &dec).unwrap();
        let expected = execute_sql(&frame, &q, SqlFormat::Tsv, false, &dec).unwrap();
        assert_eq!(got, expected);
        assert!(
            rf.stats().fetched_pct() < 45.0,
            "LIMIT 5 fetched {:.2}% (expected early exit)",
            rf.stats().fetched_pct()
        );
    }

    // 4) Full remote analytical SQL (IS NOT NULL + GROUP BY + HAVING + COUNT(DISTINCT) + ORDER BY)
    {
        let mut rf = RemoteFrame::open(HttpRangeReader::connect(&url).unwrap()).unwrap();
        let q = SqlQuery::parse(
            "SELECT region, COUNT(*) AS n, COUNT(DISTINCT category) AS cats, SUM(amount) AS total WHERE note IS NOT NULL GROUP BY region HAVING COUNT(*) > 10 ORDER BY region ASC",
        )
        .unwrap();
        let got = rf.execute_sql(&q, SqlFormat::Json, false, &dec).unwrap();
        let expected = execute_sql(&frame, &q, SqlFormat::Json, false, &dec).unwrap();
        assert_eq!(got, expected);
    }

    // 5) Live HTTP compressed-SQL API: GET /events.vxg?sql=...&fmt=json
    {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let req = concat!(
            "GET /events.vxg?sql=SELECT+region,+COUNT(*)+WHERE+note+IS+NOT+NULL+GROUP+BY+region+ORDER+BY+region+ASC&fmt=json HTTP/1.1\r\n",
            "Host: localhost\r\nConnection: close\r\n\r\n"
        );
        s.write_all(req.as_bytes()).unwrap();
        let mut resp = Vec::new();
        s.read_to_end(&mut resp).unwrap();
        let resp_s = String::from_utf8(resp).unwrap();
        assert!(resp_s.starts_with("HTTP/1.1 200 OK\r\n"), "resp: {resp_s}");
        assert!(resp_s.contains("X-VXG-Query-Ms:"));
        let (_, body) = resp_s.split_once("\r\n\r\n").unwrap();
        let q = SqlQuery::parse(
            "SELECT region, COUNT(*) WHERE note IS NOT NULL GROUP BY region ORDER BY region ASC",
        )
        .unwrap();
        let expected =
            String::from_utf8(execute_sql(&frame, &q, SqlFormat::Json, false, &dec).unwrap())
                .unwrap();
        assert_eq!(body, expected);
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
