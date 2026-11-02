fn run_project_suite(raw: &[u8], frame: &[u8], opts: &Options, dec: &DecodeOptions) {
    use vxg::{ColPred, ColTarget, PredExpr, StrCmp};

    let gb = raw.len() as f64 / 1e9;
    println!(
        "\n=== Stage 4 Compressed-Domain Column Projection (ndjson) — threads={}, median of 5 samples ===",
        dec.threads
    );
    println!(
        "{:<50} {:>10} {:>10} {:>9} {:>10}",
        "projection", "naive ms", "vxg ms", "speedup", "vxg GB/s"
    );

    let naive_select =
        |w: Option<&PredExpr<'_>>, cols: &[ColTarget<'_>], limit: usize| -> Vec<u8> {
            let r = vxg::decompress(frame, dec).unwrap();
            let mut out = Vec::with_capacity(r.len() / 2);
            let mut emitted = 0usize;
            for l in r.split_inclusive(|&b| b == b'\n') {
                if emitted >= limit {
                    break;
                }
                if w.is_none_or(|e| vxg::ops::line_matches_expr(l, e))
                    && vxg::ops::project_line(l, cols, &mut out)
                {
                    emitted += 1;
                }
            }
            out
        };

    let mut dec_out = dec.clone();
    dec_out.max_output = raw.len().saturating_mul(2).max(1 << 20);

    let bench_select =
        |label: &str, w: Option<&PredExpr<'_>>, cols: &[ColTarget<'_>], limit: usize| {
            let (tn, exp) = median_secs(5, || naive_select(w, cols, limit));
            let (tv, got) =
                median_secs(5, || vxg::select_columns(frame, w, cols, limit, dec).unwrap());
            assert_eq!(exp, got, "{label}: projected byte stream mismatch");
            println!(
                "{:<50} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (out={} B)",
                label,
                tn * 1e3,
                tv * 1e3,
                tn / tv,
                gb / tv,
                got.len()
            );
        };

    let bench_project_frame =
        |label: &str, w: Option<&PredExpr<'_>>, cols: &[ColTarget<'_>]| {
            let (tn, f_naive) = median_secs(5, || {
                let proj = naive_select(w, cols, usize::MAX);
                vxg::compress(&proj, opts).unwrap()
            });
            let (tv, f_vxg) =
                median_secs(5, || vxg::project_frame(frame, w, cols, opts, dec).unwrap());
            let r_naive = vxg::decompress(&f_naive, &dec_out).unwrap();
            let r_vxg = vxg::decompress(&f_vxg, &dec_out).unwrap();
            assert_eq!(r_naive, r_vxg, "{label}: decompressed projected frame mismatch");
            println!(
                "{:<50} {:>10.2} {:>10.2} {:>8.1}x {:>9.2}   (vxg={} B, ratio={:.2}%)",
                label,
                tn * 1e3,
                tv * 1e3,
                tn / tv,
                gb / tv,
                f_vxg.len(),
                if r_vxg.is_empty() {
                    0.0
                } else {
                    f_vxg.len() as f64 * 100.0 / r_vxg.len() as f64
                }
            );
        };

    let cols_3 = [
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("status"),
        ColTarget::JsonKey("bytes"),
    ];
    bench_select("select [service,status,bytes] (all 300k)", None, &cols_3, usize::MAX);

    let cols_2 = [
        ColTarget::JsonKey("ts"),
        ColTarget::JsonKey("req_id"),
    ];
    bench_select("select [ts,req_id] (all 300k)", None, &cols_2, usize::MAX);

    let e_500 = PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500)));
    let cols_err = [
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("latency_ms"),
        ColTarget::JsonKey("bytes"),
    ];
    bench_select(
        "select [service,lat,bytes] WHERE status==500",
        Some(&e_500),
        &cols_err,
        usize::MAX,
    );

    let e_and_parts = [
        PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"fraud-engine"))),
        PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500))),
        PredExpr::col(ColPred::json_num("latency_ms", NumCmp::Ge(150))),
    ];
    let e_and = PredExpr::all(&e_and_parts);
    let cols_fraud = [
        ColTarget::JsonKey("ts"),
        ColTarget::JsonKey("req_id"),
        ColTarget::JsonKey("latency_ms"),
    ];
    bench_select(
        "select [ts,req_id,lat] WHERE fraud&500&lat>=150",
        Some(&e_and),
        &cols_fraud,
        usize::MAX,
    );

    let cols_4 = [
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("status"),
        ColTarget::JsonKey("bytes"),
        ColTarget::JsonKey("latency_ms"),
    ];
    bench_project_frame(
        "project_frame [service,status,bytes,lat] (splice)",
        None,
        &cols_4,
    );

    bench_project_frame(
        "project_frame [service,lat,bytes] WHERE status==500",
        Some(&e_500),
        &cols_err,
    );
}
