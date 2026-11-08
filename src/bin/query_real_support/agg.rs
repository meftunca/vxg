fn run_agg_suite(raw: &[u8], frame: &[u8], dec: &DecodeOptions) {
    use std::collections::BTreeMap;
    use vxg::{AggStats, ColPred, ColTarget, GroupKey, GroupRow, PredExpr, StrCmp};

    let mrows = raw.split_inclusive(|&b| b == b'\n').count() as f64 / 1e6;
    println!(
        "\n=== Stage 3 Compressed-Domain Analytics (ndjson) — threads={}, median of 5 samples ===",
        dec.threads
    );
    println!(
        "{:<48} {:>10} {:>10} {:>9} {:>10}",
        "query", "naive ms", "vxg ms", "speedup", "M rows/s"
    );

    let naive_count = |expr: &PredExpr<'_>| -> u64 {
        let r = vxg::decompress(frame, dec).unwrap();
        r.split_inclusive(|&b| b == b'\n')
            .filter(|l| vxg::ops::line_matches_expr(l, expr))
            .count() as u64
    };
    let naive_agg = |w: Option<&PredExpr<'_>>, col: ColTarget<'_>| -> AggStats {
        let r = vxg::decompress(frame, dec).unwrap();
        let mut st = AggStats::default();
        for l in r.split_inclusive(|&b| b == b'\n') {
            vxg::ops::agg_line(l, w, col, &mut st);
        }
        st
    };
    let naive_group =
        |w: Option<&PredExpr<'_>>, gcol: ColTarget<'_>, acol: Option<ColTarget<'_>>| -> Vec<GroupRow> {
            let r = vxg::decompress(frame, dec).unwrap();
            let mut m: BTreeMap<GroupKey, AggStats> = BTreeMap::new();
            for l in r.split_inclusive(|&b| b == b'\n') {
                if let Some(ex) = w {
                    if !vxg::ops::line_matches_expr(l, ex) {
                        continue;
                    }
                }
                let Some((gs, ge)) = vxg::ops::find_line_col_span(l, gcol) else {
                    continue;
                };
                let Some((is_s, k)) = vxg::ops::decode_group_token_ref(gcol, &l[gs..ge]) else {
                    continue;
                };
                let gk = if is_s {
                    GroupKey::Str(k.into_owned())
                } else {
                    GroupKey::RawJson(k.into_owned())
                };
                let ent = m.entry(gk).or_default();
                if let Some(ac) = acol {
                    vxg::ops::agg_line(l, None, ac, ent);
                } else {
                    ent.row_count += 1;
                }
            }
            m.into_iter().map(|(key, stats)| GroupRow { key, stats }).collect()
        };

    // 1. Compound AND count (`service=="fraud-engine" AND status==500 AND latency_ms>=150`)
    let e_and_parts = [
        PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"fraud-engine"))),
        PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500))),
        PredExpr::col(ColPred::json_num("latency_ms", NumCmp::Ge(150))),
    ];
    let e_and = PredExpr::all(&e_and_parts);
    let (tn, cn) = median_secs(5, || naive_count(&e_and));
    let (tv, cv) = median_secs(5, || vxg::count_where(frame, &e_and, dec).unwrap());
    assert_eq!(cn, cv);
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (hits={cv})",
        "count WHERE svc==fraud & status==500 & lat>=150",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv
    );

    // 2. Compound filter_where (`(status==500 AND latency_ms>=220) OR bytes>=50000`)
    let sub_and = [
        PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500))),
        PredExpr::col(ColPred::json_num("latency_ms", NumCmp::Ge(220))),
    ];
    let or_parts = [
        PredExpr::all(&sub_and),
        PredExpr::col(ColPred::json_num("bytes", NumCmp::Ge(50_000))),
    ];
    let e_or = PredExpr::any(&or_parts);
    let (tn, fn_rows) = median_secs(5, || {
        let r = vxg::decompress(frame, dec).unwrap();
        r.split_inclusive(|&b| b == b'\n')
            .filter(|l| vxg::ops::line_matches_expr(l, &e_or))
            .count()
    });
    let (tv, fv_rows) = median_secs(5, || vxg::filter_where(frame, &e_or, usize::MAX, dec).unwrap());
    assert_eq!(fn_rows, fv_rows.len());
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (hits={})",
        "filter_where (status==500 & lat>=220) | B>=50k",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv,
        fv_rows.len()
    );

    // 3. Unconditional scalar aggregate (`SUM/MIN/MAX(bytes)`)
    let col_bytes = ColTarget::JsonKey("bytes");
    let (tn, an) = median_secs(5, || naive_agg(None, col_bytes));
    let (tv, av) = median_secs(5, || vxg::aggregate_column(frame, None, col_bytes, dec).unwrap());
    assert_eq!(an, av);
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (sum={})",
        "agg SUM/MIN/MAX(bytes) [all 300k rows]",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv,
        av.sum
    );

    // 4. Filtered scalar aggregate (`SUM/MIN/MAX(bytes) WHERE status==500`)
    let e_500 = PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500)));
    let (tn, an2) = median_secs(5, || naive_agg(Some(&e_500), col_bytes));
    let (tv, av2) =
        median_secs(5, || vxg::aggregate_column(frame, Some(&e_500), col_bytes, dec).unwrap());
    assert_eq!(an2, av2);
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (n={})",
        "agg SUM/MIN/MAX(bytes) WHERE status==500",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv,
        av2.row_count
    );

    // 5. GroupBy COUNT(*) (`SELECT service, COUNT(*) GROUP BY service`)
    let col_svc = ColTarget::JsonKey("service");
    let (tn, gn1) = median_secs(5, || naive_group(None, col_svc, None));
    let (tv, gv1) = median_secs(5, || vxg::group_by(frame, None, col_svc, None, dec).unwrap());
    assert_eq!(gn1, gv1);
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (groups={})",
        "group_by(service) COUNT(*)",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv,
        gv1.len()
    );

    // 6. GroupBy SUM/MIN/MAX (`SELECT service, SUM/MIN/MAX(bytes) GROUP BY service`)
    let (tn, gn2) = median_secs(5, || naive_group(None, col_svc, Some(col_bytes)));
    let (tv, gv2) =
        median_secs(5, || vxg::group_by(frame, None, col_svc, Some(col_bytes), dec).unwrap());
    assert_eq!(gn2, gv2);
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (groups={})",
        "group_by(service) SUM/MIN/MAX(bytes)",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv,
        gv2.len()
    );

    // 7. Filtered GroupBy (`SELECT service, SUM(latency_ms) WHERE status==500 GROUP BY service`)
    let col_lat = ColTarget::JsonKey("latency_ms");
    let (tn, gn3) = median_secs(5, || naive_group(Some(&e_500), col_svc, Some(col_lat)));
    let (tv, gv3) = median_secs(5, || {
        vxg::group_by(frame, Some(&e_500), col_svc, Some(col_lat), dec).unwrap()
    });
    assert_eq!(gn3, gv3);
    println!(
        "{:<48} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (groups={})",
        "group_by(service) SUM(lat) WHERE status==500",
        tn * 1e3,
        tv * 1e3,
        tn / tv,
        mrows / tv,
        gv3.len()
    );
}
