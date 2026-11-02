fn run_topk_suite(raw: &[u8], frame: &[u8], dec: &DecodeOptions) {
    use vxg::{ColPred, ColTarget, PredExpr, SortDir, StrCmp};

    let gb = raw.len() as f64 / 1e9;
    println!(
        "\n=== Stage 6 Compressed-Domain ORDER BY / Top-K (ndjson) — threads={}, median of 5 samples ===",
        dec.threads
    );
    println!(
        "{:<54} {:>10} {:>10} {:>9} {:>10}",
        "query", "naive ms", "vxg ms", "speedup", "vxg GB/s"
    );

    let cmp_key = |ka: &vxg::SortKey, ia: usize, kb: &vxg::SortKey, ib: usize, dir: SortDir| {
        use std::cmp::Ordering;
        use vxg::SortKey;
        match (ka, kb) {
            (SortKey::Missing, SortKey::Missing) => ia.cmp(&ib),
            (SortKey::Missing, _) => Ordering::Greater,
            (_, SortKey::Missing) => Ordering::Less,
            (a, b) => {
                let c = match (a, b) {
                    (SortKey::Num(x), SortKey::Num(y)) => x.cmp(y),
                    (SortKey::Str(x), SortKey::Str(y)) => x.cmp(y),
                    (SortKey::Num(_), SortKey::Str(_)) => Ordering::Less,
                    (SortKey::Str(_), SortKey::Num(_)) => Ordering::Greater,
                    _ => unreachable!(),
                };
                let c = if dir == SortDir::Desc { c.reverse() } else { c };
                c.then_with(|| ia.cmp(&ib))
            }
        }
    };

    let naive_topk = |w: Option<&PredExpr<'_>>,
                      order_col: ColTarget<'_>,
                      dir: SortDir,
                      k: usize|
     -> Vec<(usize, Vec<u8>)> {
        let r = vxg::decompress(frame, dec).unwrap();
        let mut cands = Vec::new();
        for (i, l) in r.split_inclusive(|&b| b == b'\n').enumerate() {
            if w.is_none_or(|e| vxg::ops::line_matches_expr(l, e)) {
                let key = vxg::ops::extract_line_sort_key(l, order_col);
                cands.push((key, i, l));
            }
        }
        if cands.len() > k {
            cands.select_nth_unstable_by(k, |(ka, ia, _), (kb, ib, _)| {
                cmp_key(ka, *ia, kb, *ib, dir)
            });
            cands.truncate(k);
        }
        cands.sort_by(|(ka, ia, _), (kb, ib, _)| cmp_key(ka, *ia, kb, *ib, dir));
        cands.into_iter().map(|(_, i, l)| (i, l.to_vec())).collect()
    };

    let naive_select_topk = |w: Option<&PredExpr<'_>>,
                             cols: &[ColTarget<'_>],
                             order_col: ColTarget<'_>,
                             dir: SortDir,
                             k: usize|
     -> Vec<u8> {
        let r = vxg::decompress(frame, dec).unwrap();
        let mut cands = Vec::new();
        let mut scratch = Vec::with_capacity(256);
        for (i, l) in r.split_inclusive(|&b| b == b'\n').enumerate() {
            if w.is_none_or(|e| vxg::ops::line_matches_expr(l, e)) {
                scratch.clear();
                if vxg::ops::project_line(l, cols, &mut scratch) {
                    let key = vxg::ops::extract_line_sort_key(l, order_col);
                    cands.push((key, i, l));
                }
            }
        }
        if cands.len() > k {
            cands.select_nth_unstable_by(k, |(ka, ia, _), (kb, ib, _)| {
                cmp_key(ka, *ia, kb, *ib, dir)
            });
            cands.truncate(k);
        }
        cands.sort_by(|(ka, ia, _), (kb, ib, _)| cmp_key(ka, *ia, kb, *ib, dir));
        let mut out = Vec::new();
        for (_, _, l) in cands {
            vxg::ops::project_line(l, cols, &mut out);
        }
        out
    };

    let bench_full =
        |label: &str, w: Option<&PredExpr<'_>>, col: ColTarget<'_>, dir: SortDir, k: usize| {
            let (tn, exp) = median_secs(5, || naive_topk(w, col, dir, k));
            let (tv, got) = median_secs(5, || vxg::top_k_by(frame, w, col, dir, k, dec).unwrap());
            let got_pairs: Vec<(usize, Vec<u8>)> = got
                .into_iter()
                .map(|m| (m.global_row_index, m.bytes))
                .collect();
            assert_eq!(exp, got_pairs, "{label}: top_k_by mismatch");
            println!(
                "{:<54} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (k={})",
                label,
                tn * 1e3,
                tv * 1e3,
                tn / tv,
                gb / tv,
                got_pairs.len()
            );
        };

    let bench_proj = |label: &str,
                      w: Option<&PredExpr<'_>>,
                      cols: &[ColTarget<'_>],
                      col: ColTarget<'_>,
                      dir: SortDir,
                      k: usize| {
        let (tn, exp) = median_secs(5, || naive_select_topk(w, cols, col, dir, k));
        let (tv, got) = median_secs(5, || {
            vxg::select_top_k_by(frame, w, cols, col, dir, k, dec).unwrap()
        });
        assert_eq!(exp, got, "{label}: select_top_k_by mismatch");
        println!(
            "{:<54} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (out={} B)",
            label,
            tn * 1e3,
            tv * 1e3,
            tn / tv,
            gb / tv,
            got.len()
        );
    };

    let col_bytes = ColTarget::JsonKey("bytes");
    let col_lat = ColTarget::JsonKey("latency_ms");
    let col_svc = ColTarget::JsonKey("service");
    let e_500 = PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500)));
    let e_fraud_arms = [
        PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"fraud-engine"))),
        PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500))),
    ];
    let e_fraud_500 = PredExpr::all(&e_fraud_arms);
    let cols_sel = [
        ColTarget::JsonKey("ts"),
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("latency_ms"),
    ];

    bench_full(
        "top_k ORDER BY bytes DESC LIMIT 10 (zone-map)",
        None,
        col_bytes,
        SortDir::Desc,
        10,
    );
    bench_full(
        "top_k ORDER BY latency_ms DESC LIMIT 10 (all 300k)",
        None,
        col_lat,
        SortDir::Desc,
        10,
    );
    bench_full(
        "top_k ORDER BY latency_ms ASC LIMIT 10 (all 300k)",
        None,
        col_lat,
        SortDir::Asc,
        10,
    );
    bench_full(
        "top_k WHERE status==500 ORDER BY latency_ms DESC K=20",
        Some(&e_500),
        col_lat,
        SortDir::Desc,
        20,
    );
    bench_proj(
        "select [ts,svc,lat] ORDER BY bytes DESC LIMIT 10",
        None,
        &cols_sel,
        col_bytes,
        SortDir::Desc,
        10,
    );
    bench_proj(
        "select [ts,svc,lat] WHERE fraud&500 ORDER BY lat DESC K=10",
        Some(&e_fraud_500),
        &cols_sel,
        col_lat,
        SortDir::Desc,
        10,
    );
    bench_full(
        "top_k ORDER BY service ASC LIMIT 10 (dict)",
        None,
        col_svc,
        SortDir::Asc,
        10,
    );
}
