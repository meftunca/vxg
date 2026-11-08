use std::io::Cursor;

use vxg::ops::{agg_line, extract_line_sort_key, line_matches_expr};
use vxg::{
    aggregate_column, aggregate_column_stream, compress, count_where, count_where_stream,
    decompress, filter_where, filter_where_stream, group_by, group_by_stream, parse_col_target,
    select_top_k_by, top_k_by, top_k_by_stream, AggStats, ColTarget, DecNum, DecodeOptions,
    GroupKey, GroupRow, Options, ParsedWhere, PredExpr, SortDir, SortKey,
};

fn oracle_agg(raw: &[u8], where_expr: Option<&PredExpr<'_>>, col: ColTarget<'_>) -> AggStats {
    let mut st = AggStats::default();
    for line in raw.split_inclusive(|&b| b == b'\n') {
        agg_line(line, where_expr, col, &mut st);
    }
    st
}

fn oracle_count(raw: &[u8], expr: &PredExpr<'_>) -> u64 {
    raw.split_inclusive(|&b| b == b'\n')
        .filter(|line| line_matches_expr(line, expr))
        .count() as u64
}

fn oracle_filter(raw: &[u8], expr: &PredExpr<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for line in raw.split_inclusive(|&b| b == b'\n') {
        if line_matches_expr(line, expr) {
            out.extend_from_slice(line);
        }
    }
    out
}

fn oracle_group_by(
    raw: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
) -> Vec<GroupRow> {
    let mut map: std::collections::BTreeMap<GroupKey, AggStats> = std::collections::BTreeMap::new();
    for line in raw.split_inclusive(|&b| b == b'\n') {
        if let Some(expr) = where_expr {
            if !line_matches_expr(line, expr) {
                continue;
            }
        }
        let Some((gs, ge)) = vxg::ops::find_line_col_span(line, group_col) else {
            continue;
        };
        let key = match group_col {
            ColTarget::JsonKey(_) => {
                let tok = &line[gs..ge];
                if tok.first() == Some(&b'"') {
                    let Some(s) = vxg::ops::unescape_json_str(tok) else {
                        continue;
                    };
                    GroupKey::Str(s.into_owned())
                } else {
                    GroupKey::RawJson(tok.to_vec())
                }
            }
            ColTarget::Delimited { .. } => {
                let Some(s) = vxg::ops::decode_delim_str(&line[gs..ge]) else {
                    continue;
                };
                GroupKey::Str(s.into_owned())
            }
        };
        let mut delta = AggStats::default();
        if let Some(acol) = agg_col {
            agg_line(line, None, acol, &mut delta);
        } else {
            delta.row_count = 1;
        }
        map.entry(key).or_default().merge(&delta);
    }
    map.into_iter()
        .map(|(key, stats)| GroupRow { key, stats })
        .collect()
}

fn oracle_top_k_bytes(
    raw: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    order_col: ColTarget<'_>,
    dir: SortDir,
    k: usize,
) -> Vec<Vec<u8>> {
    let mut rows: Vec<(SortKey, usize, Vec<u8>)> = Vec::new();
    for (idx, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if let Some(expr) = where_expr {
            if !line_matches_expr(line, expr) {
                continue;
            }
        }
        rows.push((extract_line_sort_key(line, order_col), idx, line.to_vec()));
    }
    rows.sort_by(|(ka, ia, _), (kb, ib, _)| match (ka, kb) {
        (SortKey::Missing, SortKey::Missing) => ia.cmp(ib),
        (SortKey::Missing, _) => std::cmp::Ordering::Greater,
        (_, SortKey::Missing) => std::cmp::Ordering::Less,
        _ => {
            let base = match (ka, kb) {
                (SortKey::Num(x), SortKey::Num(y)) => x.cmp(y),
                (SortKey::Dec(x), SortKey::Dec(y)) => x.cmp(y),
                (SortKey::Num(x), SortKey::Dec(y)) => DecNum::from_u64(*x).cmp(y),
                (SortKey::Dec(x), SortKey::Num(y)) => x.cmp(&DecNum::from_u64(*y)),
                (SortKey::Str(x), SortKey::Str(y)) => x.cmp(y),
                (SortKey::Num(_) | SortKey::Dec(_), SortKey::Str(_)) => std::cmp::Ordering::Less,
                (SortKey::Str(_), SortKey::Num(_) | SortKey::Dec(_)) => std::cmp::Ordering::Greater,
                _ => unreachable!(),
            };
            if base == std::cmp::Ordering::Equal {
                ia.cmp(ib)
            } else if dir == SortDir::Asc {
                base
            } else {
                base.reverse()
            }
        }
    });
    rows.into_iter().take(k).map(|(_, _, b)| b).collect()
}

#[test]
fn b2_scaled_decimal_and_signed_differential_ndjson_and_csv() {
    let mut ndjson = Vec::new();
    let mut csv = b"id,zone,temp_i,fare,tip\n".to_vec();
    for i in 0..900i64 {
        let zone = (i % 7) as u32 + 1;
        let temp = (i % 65) - 30; // [-30..=34], includes 0 and negatives!
        let fare_cents = 450 + ((i * 37) % 12000) - 900; // [-450..=11549], e.g. -4.50..=115.49
        let fare_sign = if fare_cents < 0 { "-" } else { "" };
        let fare_abs = fare_cents.unsigned_abs();
        let tip_str = match i % 5 {
            0 => "0".to_owned(),
            1 => "1.5".to_owned(),
            2 => "1.50".to_owned(),
            3 => "-0.05".to_owned(),
            _ => "19.99".to_owned(),
        };
        let fare_str = format!("{fare_sign}{}.{:02}", fare_abs / 100, fare_abs % 100);
        ndjson.extend_from_slice(
            format!(
                "{{\"id\":{i},\"zone\":{zone},\"temp\":{temp},\"fare\":{fare_str},\"tip\":{tip_str},\"bad\":\"-0.00\"}}\n"
            )
            .as_bytes(),
        );
        csv.extend_from_slice(format!("{i},{zone},{temp},{fare_str},{tip_str}\n").as_bytes());
    }

    for (is_csv, raw) in [(false, &ndjson), (true, &csv)] {
        let opt = Options {
            block_size: 4096, // Force multi-block boundary splits!
            threads: 2,
            checksum: true,
            ..Options::default()
        };
        let frame = compress(raw, &opt).unwrap();
        assert!(frame.len() < raw.len() / 2);
        let round = decompress(&frame, &DecodeOptions::default()).unwrap();
        assert_eq!(round, *raw);

        let (zone_col, temp_col, fare_col, tip_col) = if is_csv {
            (
                parse_col_target("csv:1").unwrap(),
                parse_col_target("csv:2").unwrap(),
                parse_col_target("csv:3").unwrap(),
                parse_col_target("csv:4").unwrap(),
            )
        } else {
            (
                parse_col_target("zone").unwrap(),
                parse_col_target("temp").unwrap(),
                parse_col_target("fare").unwrap(),
                parse_col_target("tip").unwrap(),
            )
        };

        let queries = if is_csv {
            [
                "csv:2 < 0",
                "csv:3 >= 19.99 && csv:2 != -5",
                "csv:3 <= -1.25 || csv:4 == 1.5",
                "csv:1 > 3 && csv:3 > 50.00",
            ]
        } else {
            [
                "temp < 0",
                "fare >= 19.99 && temp != -5",
                "fare <= -1.25 || tip == 1.5",
                "zone > 3 && fare > 50.00",
            ]
        };

        for &threads in &[1usize, 4] {
            let dec = DecodeOptions {
                threads,
                max_output: usize::MAX,
                verify_checksum: true,
                ..DecodeOptions::default()
            };
            for q in queries {
                let pw = ParsedWhere::parse(q).unwrap();
                pw.with_expr(|expr| {
                    let exp_cnt = oracle_count(raw, expr);
                    assert_eq!(count_where(&frame, expr, &dec).unwrap(), exp_cnt, "q={q}");
                    assert_eq!(
                        count_where_stream(&mut Cursor::new(&frame), expr, &dec).unwrap(),
                        exp_cnt
                    );
                    let exp_flt = oracle_filter(raw, expr);
                    let got_flt: Vec<u8> = filter_where(&frame, expr, usize::MAX, &dec)
                        .unwrap()
                        .into_iter()
                        .flat_map(|m| m.bytes)
                        .collect();
                    assert_eq!(got_flt, exp_flt, "q={q}");
                    let mut s_out = Vec::new();
                    filter_where_stream(&mut Cursor::new(&frame), &mut s_out, expr, usize::MAX, &dec)
                        .unwrap();
                    assert_eq!(s_out, exp_flt);

                    for acol in [temp_col, fare_col, tip_col, zone_col] {
                        let exp_agg = oracle_agg(raw, Some(expr), acol);
                        let got_agg = aggregate_column(&frame, Some(expr), acol, &dec).unwrap();
                        assert_eq!(got_agg, exp_agg, "agg q={q} col={acol:?} t={threads}");
                        let got_s = aggregate_column_stream(
                            &mut Cursor::new(&frame),
                            Some(expr),
                            acol,
                            &dec,
                        )
                        .unwrap();
                        assert_eq!(got_s, exp_agg);
                    }

                    let exp_gb = oracle_group_by(raw, Some(expr), zone_col, Some(fare_col));
                    let got_gb = group_by(&frame, Some(expr), zone_col, Some(fare_col), &dec).unwrap();
                    assert_eq!(got_gb, exp_gb, "gb q={q} t={threads}");
                    let got_gbs = group_by_stream(
                        &mut Cursor::new(&frame),
                        Some(expr),
                        zone_col,
                        Some(fare_col),
                        &dec,
                    )
                    .unwrap();
                    assert_eq!(got_gbs, exp_gb);
                });
            }

            // Unfiltered aggregations + Top-K ASC/DESC on signed & scaled decimal columns!
            for ocol in [temp_col, fare_col, tip_col, zone_col] {
                let exp_all = oracle_agg(raw, None, ocol);
                assert_eq!(aggregate_column(&frame, None, ocol, &dec).unwrap(), exp_all);

                for dir in [SortDir::Asc, SortDir::Desc] {
                    let exp_top = oracle_top_k_bytes(raw, None, ocol, dir, 15);
                    let got_top: Vec<Vec<u8>> = top_k_by(&frame, None, ocol, dir, 15, &dec)
                        .unwrap()
                        .into_iter()
                        .map(|m| m.bytes)
                        .collect();
                    assert_eq!(got_top, exp_top, "topk col={ocol:?} dir={dir:?} t={threads}");
                    let got_s: Vec<Vec<u8>> =
                        top_k_by_stream(&mut Cursor::new(&frame), None, ocol, dir, 15, &dec)
                            .unwrap()
                            .into_iter()
                            .map(|m| m.bytes)
                            .collect();
                    assert_eq!(got_s, exp_top);
                    let proj = select_top_k_by(&frame, None, &[zone_col, ocol], ocol, dir, 5, &dec)
                        .unwrap();
                    assert!(!proj.is_empty());
                }
            }
        }
    }
}

#[test]
fn b2_noncanonical_neg_zero_and_leading_zero_roundtrip_and_semantics() {
    assert_eq!(DecNum::parse(b"-0"), Some(DecNum::from_u64(0)));
    assert_eq!(DecNum::parse(b"-0.00"), Some(DecNum::from_u64(0)));
    assert_eq!(DecNum::parse(b"+12.5"), None);
    assert_eq!(DecNum::parse(b"00.5"), None);
    assert_eq!(DecNum::parse(b"0.00"), Some(DecNum::from_u64(0)));
    assert_eq!(DecNum::parse(b"1.5"), DecNum::parse(b"1.50"));

    // Frame with `-0.00` and `007` must round-trip 100% byte-for-byte without misusing MODEL_SCALED_DECIMAL!
    let mut raw = Vec::new();
    for i in 0..200 {
        let s = if i == 50 {
            "-0.00".to_owned()
        } else {
            format!("-{}.25", i + 1)
        };
        raw.extend_from_slice(format!("{{\"x\":{s},\"code\":\"007\"}}\n").as_bytes());
    }
    let frame = compress(&raw, &Options::default()).unwrap();
    let round = decompress(&frame, &DecodeOptions::default()).unwrap();
    assert_eq!(round, raw);
    let pw = ParsedWhere::parse("x == 0").unwrap();
    assert_eq!(
        pw.with_expr(|e| count_where(&frame, e, &DecodeOptions::default()))
            .unwrap(),
        1
    );
}
