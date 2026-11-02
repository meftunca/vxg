use std::collections::BTreeMap;

use vxg::{
    aggregate_column, compress, count_where, decompress, filter_frame_where, filter_where,
    group_by, ops, AggStats, ColPred, ColTarget, DecodeOptions, GroupKey, GroupRow, NumCmp,
    Options, PredExpr, StrCmp,
};

fn oracle_group_by(
    raw: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    group_col: ColTarget<'_>,
    agg_col: Option<ColTarget<'_>>,
) -> Vec<GroupRow> {
    let mut map: BTreeMap<GroupKey, AggStats> = BTreeMap::new();
    for line in raw.split_inclusive(|&b| b == b'\n') {
        if let Some(expr) = where_expr {
            if !ops::line_matches_expr(line, expr) {
                continue;
            }
        }
        let Some((gs, ge)) = ops::find_line_col_span(line, group_col) else {
            continue;
        };
        let Some((is_str, k)) = ops::decode_group_token_ref(group_col, &line[gs..ge]) else {
            continue;
        };
        let gk = if is_str {
            GroupKey::Str(k.into_owned())
        } else {
            GroupKey::RawJson(k.into_owned())
        };
        let ent = map.entry(gk).or_default();
        if let Some(acol) = agg_col {
            ops::agg_line(line, None, acol, ent);
        } else {
            ent.row_count += 1;
        }
    }
    map.into_iter()
        .map(|(key, stats)| GroupRow { key, stats })
        .collect()
}

fn oracle_agg(raw: &[u8], where_expr: Option<&PredExpr<'_>>, agg_col: ColTarget<'_>) -> AggStats {
    let mut st = AggStats::default();
    for line in raw.split_inclusive(|&b| b == b'\n') {
        ops::agg_line(line, where_expr, agg_col, &mut st);
    }
    st
}

#[test]
fn compound_pred_expr_count_filter_and_frame_passthrough() {
    let services = ["auth", "billing", "fraud", "gateway"];
    let mut raw = Vec::new();
    for i in 0..1400u64 {
        if i == 333 || i == 999 {
            raw.extend_from_slice(b"# unstructured comment line with fraud and 500\n");
            continue;
        }
        let svc = services[(i as usize) % services.len()];
        let status = if i % 15 == 0 {
            500
        } else if i % 7 == 0 {
            404
        } else {
            200
        };
        let lat = 10 + (i * 17) % 300;
        let tag = if i % 25 == 0 { "canary-err" } else { "prod-ok" };
        raw.extend_from_slice(
            format!(
                "{{\"id\":{},\"svc\":\"{}\",\"status\":{},\"lat\":{},\"tag\":\"{}\"}}\n",
                i, svc, status, lat, tag
            )
            .as_bytes(),
        );
    }
    // Unterminated final record to exercise EOF / boundary handling.
    raw.extend_from_slice(b"{\"id\":9999,\"svc\":\"fraud\",\"status\":500,\"lat\":299,\"tag\":\"canary-err\"}");

    let enc = Options {
        block_size: 4096,
        threads: 4,
        checksum: true,
        ..Options::default()
    };
    let dec1 = DecodeOptions {
        threads: 1,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };
    let frame = compress(&raw, &enc).unwrap();

    let c_fraud = PredExpr::col(ColPred::json_str("svc", StrCmp::Eq(b"fraud")));
    let c_500 = PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500)));
    let c_lat = PredExpr::col(ColPred::json_num("lat", NumCmp::Ge(180)));
    let c_absent = PredExpr::col(ColPred::json_str("missing_key", StrCmp::Eq(b"x")));
    let c_not_absent = PredExpr::not(&c_absent); // AlwaysTrue on structured templates!
    let c_canary = PredExpr::contains(b"canary-err");
    let c_not_canary = PredExpr::not(&c_canary);

    let conj1 = [c_fraud, c_500, c_not_canary];
    let conj2 = [c_lat, c_canary, c_not_absent];
    let disj = [PredExpr::all(&conj1), PredExpr::all(&conj2)];
    let complex_expr = PredExpr::any(&disj);

    let zero_conj = [c_absent, c_fraud];
    let zero_expr = PredExpr::all(&zero_conj);
    let empty_all = PredExpr::all(&[]);
    let empty_any = PredExpr::any(&[]);

    for expr in [
        &PredExpr::True,
        &complex_expr,
        &zero_expr,
        &c_not_absent,
        &empty_all,
        &empty_any,
    ] {
        let exp_lines: Vec<(usize, Vec<u8>)> = raw
            .split_inclusive(|&b| b == b'\n')
            .enumerate()
            .filter(|(_, l)| ops::line_matches_expr(l, expr))
            .map(|(i, l)| (i, l.to_vec()))
            .collect();

        assert_eq!(count_where(&frame, expr, &dec1).unwrap(), exp_lines.len() as u64);
        assert_eq!(count_where(&frame, expr, &dec4).unwrap(), exp_lines.len() as u64);

        for lim in [1usize, 6usize, usize::MAX] {
            let got1: Vec<(usize, Vec<u8>)> = filter_where(&frame, expr, lim, &dec1)
                .unwrap()
                .into_iter()
                .map(|m| (m.global_row_index, m.bytes))
                .collect();
            let got4: Vec<(usize, Vec<u8>)> = filter_where(&frame, expr, lim, &dec4)
                .unwrap()
                .into_iter()
                .map(|m| (m.global_row_index, m.bytes))
                .collect();
            let exp_lim: Vec<(usize, Vec<u8>)> = exp_lines.iter().take(lim).cloned().collect();
            assert_eq!(got1, exp_lim);
            assert_eq!(got4, exp_lim);
        }

        let ff1 = filter_frame_where(&frame, expr, &enc, &dec1).unwrap();
        let ff4 = filter_frame_where(&frame, expr, &enc, &dec4).unwrap();
        let mut exp_raw = Vec::new();
        for (_, b) in &exp_lines {
            exp_raw.extend_from_slice(b);
        }
        assert_eq!(decompress(&ff1, &dec4).unwrap(), exp_raw);
        assert_eq!(decompress(&ff4, &dec4).unwrap(), exp_raw);
    }
}

#[test]
fn scalar_aggregate_and_group_by_all_column_models_and_types() {
    let services = ["auth", "billing", "fraud", "gateway", "quote\"svc"];
    let mut json_raw = Vec::new();
    let mut csv_raw = Vec::new();

    for i in 0..1200u64 {
        let svc = services[(i as usize) % services.len()];
        let status_tok = match i % 12 {
            0 => "\"500\"".to_string(), // Quoted string "500"
            1 => "null".to_string(),    // Raw JSON null
            2 => "\"null\"".to_string(), // Quoted string "null"
            3 => "500".to_string(),     // Raw JSON integer 500
            _ => "200".to_string(),
        };
        let small = (i * 19) % 800; // DecimalSmall
        let big = 100_000 + (i * 97) % 250_000; // DecimalPacked / DecimalValues
        let asc = 10_000 + i * 3; // Ascending Affine
        let desc = 50_000 - i * 5; // Descending Affine (negative step!)
        let cval = 42u64; // Constant numeric column
        let qdec = format!("\"ord-{:05}\"", i); // QuotedDecimal (not canonical u64!)

        let mut svc_json = Vec::new();
        ops::encode_json_str(svc.as_bytes(), &mut svc_json);
        json_raw.extend_from_slice(
            format!(
                "{{\"svc\":{},\"st\":{},\"small\":{},\"big\":{},\"asc\":{},\"desc\":{},\"cval\":{},\"qdec\":{}}}\n",
                std::str::from_utf8(&svc_json).unwrap(),
                status_tok,
                small,
                big,
                asc,
                desc,
                cval,
                qdec
            )
            .as_bytes(),
        );
        csv_raw.extend_from_slice(
            format!("{},{},{},{},{}\n", i % 7, small, big, asc, desc).as_bytes(),
        );
    }

    let enc = Options {
        block_size: 4096,
        threads: 4,
        checksum: true,
        ..Options::default()
    };
    let dec1 = DecodeOptions {
        threads: 1,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };
    let jf = compress(&json_raw, &enc).unwrap();
    let cf = compress(&csv_raw, &enc).unwrap();

    let w_expr = PredExpr::col(ColPred::json_num("small", NumCmp::Ge(400)));
    for key in ["small", "big", "asc", "desc", "cval", "qdec", "st", "missing"] {
        let col = ColTarget::JsonKey(key);
        for w in [None, Some(&w_expr)] {
            let exp = oracle_agg(&json_raw, w, col);
            assert_eq!(aggregate_column(&jf, w, col, &dec1).unwrap(), exp, "1T agg {key}");
            assert_eq!(aggregate_column(&jf, w, col, &dec4).unwrap(), exp, "4T agg {key}");
        }
    }

    // GroupBy tests: Dictionary group (`svc`), mixed JSON types (`"500"` vs `500`, `"null"` vs `null` in `st`),
    // constant group (`cval`), QuotedDecimal group (`qdec`), and pure numeric CSV group (`col 0`).
    for g_key in ["svc", "st", "cval", "small", "missing"] {
        let gcol = ColTarget::JsonKey(g_key);
        for acol in [None, Some(ColTarget::JsonKey("big")), Some(ColTarget::JsonKey("desc"))] {
            for w in [None, Some(&w_expr)] {
                let exp = oracle_group_by(&json_raw, w, gcol, acol);
                assert_eq!(group_by(&jf, w, gcol, acol, &dec1).unwrap(), exp, "1T group {g_key}");
                assert_eq!(group_by(&jf, w, gcol, acol, &dec4).unwrap(), exp, "4T group {g_key}");
            }
        }
    }

    // Verify `"500"` (Str) and `500` (RawJson), plus `"null"` (Str) and `null` (RawJson), stay distinct!
    let st_groups = group_by(&jf, None, ColTarget::JsonKey("st"), None, &dec4).unwrap();
    assert!(st_groups.iter().any(|g| g.key == GroupKey::Str(b"500".to_vec())));
    assert!(st_groups.iter().any(|g| g.key == GroupKey::RawJson(b"500".to_vec())));
    assert!(st_groups.iter().any(|g| g.key == GroupKey::Str(b"null".to_vec())));
    assert!(st_groups.iter().any(|g| g.key == GroupKey::RawJson(b"null".to_vec())));

    // CSV numeric group_by + affine descending aggregation
    let csv_w = PredExpr::col(ColPred::delimited_num(b',', 1, NumCmp::Le(500)));
    let c_gcol = ColTarget::Delimited { delim: b',', index: 0 };
    let c_acol = Some(ColTarget::Delimited { delim: b',', index: 4 });
    let exp_csv = oracle_group_by(&csv_raw, Some(&csv_w), c_gcol, c_acol);
    assert_eq!(group_by(&cf, Some(&csv_w), c_gcol, c_acol, &dec1).unwrap(), exp_csv);
    assert_eq!(group_by(&cf, Some(&csv_w), c_gcol, c_acol, &dec4).unwrap(), exp_csv);
}

#[test]
fn randomized_differential_stage3_analytics_stress() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rng = || -> u64 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    let dec1 = DecodeOptions {
        threads: 1,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };

    for iter in 0..80usize {
        let is_csv = (iter & 1) == 1;
        let rows = 80 + (rng() as usize % 350);
        let mut raw = Vec::new();
        for r in 0..rows {
            if rng() % 19 == 0 {
                // Inject multi-KB line or comment to force cross-block carries & mixed templates!
                let pad = if rng() % 4 == 0 { 4300 } else { 24 };
                raw.extend_from_slice(&vec![b'x'; pad]);
                raw.push(b'\n');
                continue;
            }
            let k = ["alpha", "beta", "gamma", "delta"][(rng() as usize) % 4];
            let n1 = rng() % 2000;
            let n2 = 50_000 + (r as u64) * 3;
            if is_csv {
                raw.extend_from_slice(format!("{},{},{}\n", k, n1, n2).as_bytes());
            } else if rng() % 7 == 0 {
                // Multi-template variant with extra field
                raw.extend_from_slice(
                    format!("{{\"k\":\"{}\",\"n1\":{},\"n2\":{},\"x\":1}}\n", k, n1, n2).as_bytes(),
                );
            } else {
                raw.extend_from_slice(
                    format!("{{\"k\":\"{}\",\"n1\":{},\"n2\":{}}}\n", k, n1, n2).as_bytes(),
                );
            }
        }
        if rng() % 3 == 0 {
            raw.extend_from_slice(b"{\"k\":\"beta\",\"n1\":777,\"n2\":60000}");
        }

        // Concatenate two frames on odd-third iterations to test multi-frame streams!
        let enc = Options {
            block_size: 4096,
            threads: 2,
            checksum: (iter & 2) == 0,
            ..Options::default()
        };
        let frame = if iter % 3 == 0 && raw.len() > 200 {
            let mid = raw.len() / 2;
            let mut f1 = compress(&raw[..mid], &enc).unwrap();
            let f2 = compress(&raw[mid..], &enc).unwrap();
            f1.extend_from_slice(&f2);
            // Note: concatenated frames each start a fresh stream unless decoded via `decompress`;
            // compare against `decompress(&f1)` which is the exact stream logical byte sequence!
            f1
        } else {
            compress(&raw, &enc).unwrap()
        };
        let logical_raw = decompress(&frame, &dec4).unwrap();

        let (gcol, acol, p1, p2) = if is_csv {
            (
                ColTarget::Delimited { delim: b',', index: 0 },
                ColTarget::Delimited { delim: b',', index: 1 },
                PredExpr::col(ColPred::delimited_str(b',', 0, StrCmp::Eq(b"beta"))),
                PredExpr::col(ColPred::delimited_num(b',', 1, NumCmp::Ge(900))),
            )
        } else {
            (
                ColTarget::JsonKey("k"),
                ColTarget::JsonKey("n1"),
                PredExpr::col(ColPred::json_str("k", StrCmp::Eq(b"beta"))),
                PredExpr::col(ColPred::json_num("n1", NumCmp::Ge(900))),
            )
        };
        let p_sub = PredExpr::contains(b"gamma");
        let p_not = PredExpr::not(&p_sub);
        let conj = [p1, p2];
        let disj = [PredExpr::all(&conj), p_not];
        let expr = PredExpr::any(&disj);

        let exp_cnt = logical_raw
            .split_inclusive(|&b| b == b'\n')
            .filter(|l| ops::line_matches_expr(l, &expr))
            .count() as u64;
        assert_eq!(count_where(&frame, &expr, &dec1).unwrap(), exp_cnt, "iter={iter}");
        assert_eq!(count_where(&frame, &expr, &dec4).unwrap(), exp_cnt, "iter={iter}");

        let exp_agg = oracle_agg(&logical_raw, Some(&expr), acol);
        assert_eq!(aggregate_column(&frame, Some(&expr), acol, &dec1).unwrap(), exp_agg);
        assert_eq!(aggregate_column(&frame, Some(&expr), acol, &dec4).unwrap(), exp_agg);

        let exp_grp = oracle_group_by(&logical_raw, Some(&expr), gcol, Some(acol));
        assert_eq!(group_by(&frame, Some(&expr), gcol, Some(acol), &dec1).unwrap(), exp_grp);
        assert_eq!(group_by(&frame, Some(&expr), gcol, Some(acol), &dec4).unwrap(), exp_grp);
    }
}
