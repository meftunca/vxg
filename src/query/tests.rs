    use super::*;
    use crate::{compress, DecodeOptions, Options};

    fn sample_ndjson(rows: usize) -> Vec<u8> {
        let services = ["auth", "payment", "inventory", "gateway"];
        let levels = ["INFO", "INFO", "WARN", "ERROR"];
        let mut out = Vec::new();
        for i in 0..rows {
            let svc = services[i % services.len()];
            let lvl = if i == rows - 7 { "CRITICAL" } else { levels[i % levels.len()] };
            let line = format!(
                "{{\"ts\":\"2026-10-17T19:{:02}:{:02}Z\",\"seq\":{},\"level\":\"{}\",\"service\":\"{}\",\"latency_us\":{}}}\n",
                (i / 60) % 60, i % 60, 1000 + i, lvl, svc, 100 + (i * 7) % 500
            );
            out.extend_from_slice(line.as_bytes());
        }
        out
    }

    #[test]
    fn random_access_and_pushdown_search_roundtrip() {
        let raw = sample_ndjson(2500);
        let opts = Options::default();
        let dec_opts = DecodeOptions::default();
        let frame = compress(&raw, &opts).unwrap();

        let expected_lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
        let total_rows = frame_row_count(&frame).unwrap();
        assert_eq!(total_rows, expected_lines.len());

        assert_eq!(extract_block_row(&frame, 0, 0, &dec_opts).unwrap(), expected_lines[0]);
        assert_eq!(extract_block_row(&frame, 0, 42, &dec_opts).unwrap(), expected_lines[42]);
        assert_eq!(
            extract_block_row_range(&frame, 0, 10, 25, &dec_opts).unwrap(),
            expected_lines[10..25].concat()
        );
        assert_eq!(extract_row_range(&frame, 0, total_rows, &dec_opts).unwrap(), raw);

        let crit = find_first(&frame, b"CRITICAL", &dec_opts).unwrap().expect("CRITICAL row");
        assert!(contains_bytes(&crit.bytes, b"\"level\":\"CRITICAL\""));
        assert!(contains_bytes(&crit.bytes, b"\"seq\":3493"));

        let crit_span = find_first(&frame, b"\"level\":\"CRITICAL\"", &dec_opts).unwrap().unwrap();
        assert_eq!(crit_span.global_row_index, crit.global_row_index);
        assert_eq!(crit_span.bytes, crit.bytes);

        let static_hit = find_first(&frame, b"\"service\":\"", &dec_opts).unwrap().unwrap();
        assert_eq!(static_hit.global_row_index, 0);
        assert_eq!(static_hit.bytes, expected_lines[0]);
        assert!(find_first(&frame, b"NON_EXISTENT_TOKEN_XYZ", &dec_opts).unwrap().is_none());

        let payment_rows = filter_rows(&frame, b"\"service\":\"payment\"", 10, &dec_opts).unwrap();
        assert_eq!(payment_rows.len(), 10);
        for m in &payment_rows {
            assert!(contains_bytes(&m.bytes, b"\"service\":\"payment\""));
        }

        unsafe {
            assert_eq!(vxg_frame_row_count(frame.as_ptr(), frame.len()) as usize, total_rows);
            let mut buf = [0u8; 512];
            let mut found_idx = 0usize;
            let n = vxg_find_first_into(
                frame.as_ptr(), frame.len(), b"CRITICAL".as_ptr(), 8,
                buf.as_mut_ptr(), buf.len(), &raw mut found_idx,
            );
            assert!(n > 0);
            assert_eq!(found_idx, crit.global_row_index);
            assert_eq!(&buf[..n as usize], crit.bytes.as_slice());
        }
    }

fn expected_u64(raw: &[u8], key: &str, cmp: NumCmp, limit: usize) -> Vec<(usize, Vec<u8>)> {
    let f = NumFilter::new(key, cmp);
    let mut out = Vec::new();
    for (i, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if numeric::line_numeric_match(line, &f) {
            out.push((i, line.to_vec()));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn check_u64(frame: &[u8], raw: &[u8], key: &str, cmp: NumCmp, limit: usize) -> usize {
    let dec = DecodeOptions::default();
    let want = expected_u64(raw, key, cmp, limit);
    let got = filter_json_u64(frame, key, cmp, limit, &dec)
        .unwrap_or_else(|e| panic!("filter_json_u64({key}, {cmp:?}): {e:?}"));
    assert_eq!(got.len(), want.len(), "count for {key} {cmp:?}");
    for (g, (wi, wb)) in got.iter().zip(want.iter()) {
        assert_eq!(g.global_row_index, *wi, "row index for {key} {cmp:?}");
        assert_eq!(&g.bytes, wb, "row bytes for {key} {cmp:?}");
    }
    got.len()
}

#[test]
fn numeric_filter_matches_line_reference_across_blocks() {
    let raw = sample_ndjson(6000);
    let frame = compress(&raw, &Options::default()).unwrap();
    let mut total = 0;
    for cmp in [
        NumCmp::Eq(3000),
        NumCmp::Ge(6900),
        NumCmp::Le(1010),
        NumCmp::Range(2500, 2600),
        NumCmp::Eq(999_999_999),
    ] {
        total += check_u64(&frame, &raw, "seq", cmp, usize::MAX);
    }
    total += check_u64(&frame, &raw, "latency_us", NumCmp::Ge(550), usize::MAX);
    total += check_u64(&frame, &raw, "latency_us", NumCmp::Le(120), 7);
    assert!(total > 0, "test must exercise real matches");
    // Missing key and a key that only exists as a string value never match.
    assert_eq!(check_u64(&frame, &raw, "no_such_key", NumCmp::Ge(0), usize::MAX), 0);
    assert_eq!(check_u64(&frame, &raw, "level", NumCmp::Ge(0), usize::MAX), 0);
}

#[test]
fn numeric_filter_handles_irregular_values_and_raw_rows() {
    let mut raw = Vec::new();
    for i in 0..4000u64 {
        let line = match i % 11 {
            3 => format!("{{\"id\":{i},\"n\":1.5,\"s\":\"x\"}}\n"),
            5 => format!("{{\"id\":{i},\"n\":\"{i}\",\"s\":\"x\"}}\n"),
            7 => format!("{{\"id\":{i},\"s\":\"x\"}}\n"),
            9 => format!("{{\"id\":{i},\"n\":007,\"s\":\"x\"}}\n"),
            _ => format!("{{\"id\":{i},\"n\":{},\"s\":\"x\"}}\n", i * 3 % 977),
        };
        raw.extend_from_slice(line.as_bytes());
    }
    raw.extend_from_slice(b"{\"id\":4000,\"n\":18446744073709551615,\"s\":\"x\"}\n");
    raw.extend_from_slice(b"{\"id\":4001,\"n\":18446744073709551616,\"s\":\"x\"}\n");
    let frame = compress(&raw, &Options::default()).unwrap();
    let mut total = 0;
    for cmp in [
        NumCmp::Eq(0),
        NumCmp::Ge(900),
        NumCmp::Le(5),
        NumCmp::Range(100, 120),
        NumCmp::Ge(u64::MAX),
    ] {
        total += check_u64(&frame, &raw, "n", cmp, usize::MAX);
        total += check_u64(&frame, &raw, "id", cmp, usize::MAX);
    }
    assert!(total > 0);
}

#[test]
fn numeric_filter_without_trailing_newline() {
    let mut raw = sample_ndjson(900);
    raw.pop(); // drop final '\n'
    let frame = compress(&raw, &Options::default()).unwrap();
    assert!(check_u64(&frame, &raw, "seq", NumCmp::Ge(1890), usize::MAX) > 0);
}

/// Regression: a column with more than 256 distinct values (9..=16-bit dictionary ids) used to
/// compress fine but fail to decompress with `column bit index out of bounds`.
#[test]
fn wide_dictionary_columns_round_trip() {
    let dec = DecodeOptions::default();
    for (modulus, every) in [(256u64, 50u64), (257, 2), (257, 50), (300, 50), (977, 4), (977, 50)] {
        let mut raw = Vec::new();
        for i in 0..6000u64 {
            let line = if i % every == 0 {
                format!("{{\"id\":{i},\"n\":1.5,\"s\":\"x\"}}\n")
            } else {
                format!("{{\"id\":{i},\"n\":{},\"s\":\"x\"}}\n", i % modulus)
            };
            raw.extend_from_slice(line.as_bytes());
        }
        let frame = compress(&raw, &Options::default()).unwrap();
        let out = crate::decompress(&frame, &dec)
            .unwrap_or_else(|e| panic!("mod={modulus} every={every}: {e:?}"));
        assert_eq!(out, raw, "mod={modulus} every={every}");
        // Multi-threaded decode must agree.
        let mt = DecodeOptions { threads: 4, ..DecodeOptions::default() };
        assert_eq!(crate::decompress(&frame, &mt).unwrap(), raw);
        // Row access and substring search go through the same column parser.
        assert_eq!(frame_row_count(&frame).unwrap(), 6000);
        assert_eq!(extract_row_range(&frame, 0, 6000, &dec).unwrap(), raw);
        let hit = find_first(&frame, b"\"n\":1.5", &dec).unwrap().expect("1.5 row");
        assert_eq!(hit.global_row_index, 0);
    }
}

#[test]
fn numeric_filter_on_small_cardinality_dictionary_column() {
    // A column with few distinct values becomes a dictionary column; entries that are not
    // plain unsigned integers must never match.
    let vals = ["200", "404", "500", "\"x\"", "007", "1.5", "503", "301"];
    let mut raw = Vec::new();
    for i in 0..9000usize {
        let v = vals[(i * 7 + i / 13) % vals.len()];
        raw.extend_from_slice(format!("{{\"id\":{i},\"code\":{v},\"s\":\"x\"}}\n").as_bytes());
    }
    let frame = compress(&raw, &Options::default()).unwrap();
    let mut total = 0;
    for cmp in [
        NumCmp::Eq(500),
        NumCmp::Ge(500),
        NumCmp::Le(300),
        NumCmp::Range(300, 410),
        NumCmp::Eq(7),
        NumCmp::Eq(999),
    ] {
        total += check_u64(&frame, &raw, "code", cmp, usize::MAX);
        total += check_u64(&frame, &raw, "code", cmp, 5);
    }
    assert!(total > 0);
}
