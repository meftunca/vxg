//! Differential oracle & safety tests for:
//! 1. Thread-local `GRAMMAR_TPL_WIRE_CACHE` & `JSON_TPL_SLOT_CACHE` across schema-drifting,
//!    key-reordered, multi-template, nested (`actor.login`), and escaped (`q\"k` / `\u0071`) NDJSON.
//! 2. `Options::target_block_rows` and automatic `--record-aligned` row-bounded block scaling on
//!    verbose (`>= 512 KiB`) NDJSON, including `max_compressed_size(n, 65536)` `compress_into` safety
//!    and `1T` vs `4T` byte-exact determinism.

use vxg::{
    aggregate_column, compress, compress_into, count_where, decompress, execute_sql, filter_where,
    group_by, max_compressed_size, parse_col_list, parse_col_target, select_columns, top_k_by,
    ColPred, DecodeOptions, Error, Frame, NumCmp, Options, PredExpr, SortDir, SqlFormat, SqlQuery,
    StrCmp,
};

#[derive(Clone)]
struct OracleRow {
    raw_line: Vec<u8>,
    payment_type: u64,
    tip_cents: u64,
    distance: u64,
    qk: Option<u64>,
    actor_login: Option<String>,
}

fn build_drifting_ndjson() -> (Vec<u8>, Vec<OracleRow>) {
    let mut raw = Vec::with_capacity(420_000);
    let mut rows = Vec::new();
    let logins = ["alice", "bob", "carol", "dave"];

    for i in 0..2200usize {
        let pay = 1 + (i % 4) as u64;
        let tip = ((i * 37) % 1500) as u64;
        let dist = (100 + (i * 19) % 1200) as u64;
        let qk_val = (i % 11) as u64;
        let login = logins[i % logins.len()];
        let phase = (i / 350) % 4;

        let (line, qk_opt, login_opt) = match phase {
            // Phase 0: Template A (payment_type first, nested actor, escaped quote in key)
            0 => (
                format!(
                    "{{\"id\":{i},\"payment_type\":{pay},\"distance\":{dist},\"tip_cents\":{tip},\"actor\":{{\"login\":\"{login}\",\"team\":{t}}},\"q\\\"k\":{qk_val},\"pad\":\"alpha_segment_{p:04}\"}}\n",
                    t = i % 3,
                    p = i % 97
                ),
                Some(qk_val),
                Some(login.to_string()),
            ),
            // Phase 1: Reordered keys! `tip_cents` moves to slot 0, `payment_type` moves to slot 3, unicode `\u0071\u0022\u006b`, no `actor`
            1 => (
                format!(
                    "{{\"tip_cents\":{tip},\"extra\":\"beta_{p:04}\",\"distance\":{dist},\"payment_type\":{pay},\"\\u0071\\u0022\\u006b\":{qk_val},\"id\":{i}}}\n",
                    p = i % 89
                ),
                Some(qk_val),
                None,
            ),
            // Phase 2: 3 interleaved templates within every single block
            2 => match i % 3 {
                0 => (
                    format!(
                        "{{\"id\":{i},\"payment_type\":{pay},\"distance\":{dist},\"tip_cents\":{tip},\"actor\":{{\"login\":\"{login}\"}},\"pad\":\"gamma_{p:04}\"}}\n",
                        p = i % 53
                    ),
                    None,
                    Some(login.to_string()),
                ),
                1 => (
                    format!(
                        "{{\"distance\":{dist},\"tip_cents\":{tip},\"id\":{i},\"q\\\"k\":{qk_val},\"payment_type\":{pay},\"note\":\"delta_{p:04}\"}}\n",
                        p = i % 61
                    ),
                    Some(qk_val),
                    None,
                ),
                _ => (
                    format!(
                        "{{\"actor\":{{\"login\":\"{login}\",\"role\":\"dev\"}},\"payment_type\":{pay},\"tip_cents\":{tip},\"distance\":{dist},\"id\":{i}}}\n"
                    ),
                    None,
                    Some(login.to_string()),
                ),
            },
            // Phase 3: Return to Template A shape with a different constant prefix to verify wire-cache re-hit & eviction safety
            _ => (
                format!(
                    "{{\"id\":{i},\"payment_type\":{pay},\"distance\":{dist},\"tip_cents\":{tip},\"actor\":{{\"login\":\"{login}\",\"team\":9}},\"q\\\"k\":{qk_val},\"pad\":\"omega_segment_{p:04}\"}}\n",
                    p = i % 71
                ),
                Some(qk_val),
                Some(login.to_string()),
            ),
        };

        raw.extend_from_slice(line.as_bytes());
        rows.push(OracleRow {
            raw_line: line.into_bytes(),
            payment_type: pay,
            tip_cents: tip,
            distance: dist,
            qk: qk_opt,
            actor_login: login_opt,
        });
    }
    (raw, rows)
}

#[test]
fn ndjson_template_cache_schema_drift_and_escaped_keys_parity() {
    let (raw, oracle) = build_drifting_ndjson();

    let expected_compound: Vec<&[u8]> = oracle
        .iter()
        .filter(|r| r.payment_type == 1 && r.tip_cents >= 500 && r.distance >= 400)
        .map(|r| r.raw_line.as_slice())
        .collect();
    let expected_escaped_cnt = oracle.iter().filter(|r| r.qk == Some(7)).count() as u64;
    let expected_nested_cnt = oracle
        .iter()
        .filter(|r| r.actor_login.as_deref() == Some("alice") && r.tip_cents >= 300)
        .count() as u64;
    let expected_sum_pay2: u128 = oracle
        .iter()
        .filter(|r| r.payment_type == 2)
        .map(|r| r.tip_cents as u128)
        .sum();

    for (bsz, aligned) in [(16_384usize, false), (16_384, true), (65_536, true)] {
        let frame = compress(
            &raw,
            &Options {
                block_size: bsz,
                record_aligned: aligned,
                threads: 2,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(decompress(&frame, &DecodeOptions::default()).unwrap(), raw);

        for threads in [1usize, 4] {
            let dopts = DecodeOptions {
                threads,
                ..DecodeOptions::default()
            };

            let compound_children = [
                PredExpr::col(ColPred::json_num("payment_type", NumCmp::Eq(1))),
                PredExpr::col(ColPred::json_num("tip_cents", NumCmp::Ge(500))),
                PredExpr::col(ColPred::json_num("distance", NumCmp::Ge(400))),
            ];
            let compound = PredExpr::all(&compound_children);
            assert_eq!(
                count_where(&frame, &compound, &dopts).unwrap(),
                expected_compound.len() as u64
            );

            let filtered = filter_where(&frame, &compound, usize::MAX, &dopts).unwrap();
            assert_eq!(filtered.len(), expected_compound.len());
            for (rm, exp_ln) in filtered.iter().zip(expected_compound.iter()) {
                assert_eq!(rm.bytes.as_slice(), *exp_ln);
            }

            let esc_expr = PredExpr::col(ColPred::json_num("q\"k", NumCmp::Eq(7)));
            assert_eq!(
                count_where(&frame, &esc_expr, &dopts).unwrap(),
                expected_escaped_cnt
            );

            let nested_children = [
                PredExpr::col(ColPred::json_str("actor.login", StrCmp::Eq(b"alice"))),
                PredExpr::col(ColPred::json_num("tip_cents", NumCmp::Ge(300))),
            ];
            let nested_expr = PredExpr::all(&nested_children);
            assert_eq!(
                count_where(&frame, &nested_expr, &dopts).unwrap(),
                expected_nested_cnt
            );

            let pay2 = PredExpr::col(ColPred::json_num("payment_type", NumCmp::Eq(2)));
            let agg = aggregate_column(
                &frame,
                Some(&pay2),
                parse_col_target("tip_cents").unwrap(),
                &dopts,
            )
            .unwrap();
            assert_eq!(agg.sum, expected_sum_pay2);

            let groups = group_by(
                &frame,
                None,
                parse_col_target("payment_type").unwrap(),
                Some(parse_col_target("tip_cents").unwrap()),
                &dopts,
            )
            .unwrap();
            assert_eq!(groups.len(), 4);
            let total_group_rows: u64 = groups.iter().map(|g| g.stats.row_count).sum();
            assert_eq!(total_group_rows, oracle.len() as u64);

            let cols = parse_col_list("id,payment_type,tip_cents").unwrap();
            let projected =
                select_columns(&frame, Some(&compound), &cols, usize::MAX, &dopts).unwrap();
            assert_eq!(
                projected.iter().filter(|&&b| b == b'\n').count(),
                expected_compound.len()
            );

            let top5 = top_k_by(
                &frame,
                Some(&compound),
                parse_col_target("tip_cents").unwrap(),
                SortDir::Desc,
                5,
                &dopts,
            )
            .unwrap();
            assert_eq!(top5.len(), 5);
        }
    }
}

#[test]
fn row_bounded_ndjson_block_scaling_and_determinism() {
    assert!(matches!(
        compress(
            b"{\"a\":1}\n",
            &Options {
                target_block_rows: 65_537,
                ..Options::default()
            }
        ),
        Err(Error::InvalidOption(_))
    ));

    // Build ~2.17 MiB verbose NYC-Taxi-shaped NDJSON (~265 bytes/row * 8,200 rows >= 4 * 512 KiB)
    let mut raw = Vec::with_capacity(2_250_000);
    for i in 0..8200usize {
        let pay = 1 + (i % 4);
        let tip = if pay == 1 { (i * 29) % 1800 } else { 0 };
        let dist = 120 + (i * 17) % 2400;
        let fare = 700 + dist * 2;
        let total = fare + tip + 250;
        let line = format!(
            "{{\"VendorID\":{},\"tpep_pickup_datetime\":\"2024-01-15T12:{:02}:{:02}Z\",\"tpep_dropoff_datetime\":\"2024-01-15T12:{:02}:{:02}Z\",\"passenger_count\":{},\"PULocationID\":{},\"DOLocationID\":{},\"payment_type\":{pay},\"distance_centimiles\":{dist},\"fare_cents\":{fare},\"tip_cents\":{tip},\"total_cents\":{total}}}\n",
            1 + (i % 2),
            (i / 60) % 60,
            i % 60,
            ((i + 15) / 60) % 60,
            (i + 15) % 60,
            1 + (i % 5),
            10 + (i % 250),
            20 + ((i * 7) % 250),
        );
        raw.extend_from_slice(line.as_bytes());
    }
    assert!(raw.len() > 524_288);

    // 1. Fixed 64 KiB baseline
    let fixed_64k = compress(&raw, &Options::default()).unwrap();
    let fixed_parsed = Frame::parse(&fixed_64k, usize::MAX).unwrap();
    assert_eq!(fixed_parsed.block_size(), 65_536);

    // 2. Default `--record-aligned` (`block_size == 65_536`) automatically scales verbose NDJSON block size up!
    let auto_1t = compress(
        &raw,
        &Options {
            record_aligned: true,
            threads: 1,
            ..Options::default()
        },
    )
    .unwrap();
    let auto_4t = compress(
        &raw,
        &Options {
            record_aligned: true,
            threads: 4,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(
        auto_1t, auto_4t,
        "1T and 4T auto-scaled frames must be byte-identical"
    );
    let auto_parsed = Frame::parse(&auto_1t, usize::MAX).unwrap();
    assert_eq!(auto_parsed.block_size(), 524_288);
    assert!(auto_parsed.block_count() < fixed_parsed.block_count());
    assert_eq!(decompress(&auto_1t, &DecodeOptions::default()).unwrap(), raw);

    // Verify `compress_into` works safely with a caller buffer sized for `options.block_size` (65_536)!
    let bound = max_compressed_size(raw.len(), 65_536).unwrap();
    let mut caller_buf = vec![0u8; bound];
    let written = compress_into(
        &raw,
        &mut caller_buf,
        &Options {
            record_aligned: true,
            threads: 2,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(&caller_buf[..written], auto_1t.as_slice());

    // 3. Explicit `target_block_rows: 1024` enables `record_aligned` and chooses ~270 KiB blocks
    let target_frame = compress(
        &raw,
        &Options {
            target_block_rows: 1024,
            threads: 2,
            ..Options::default()
        },
    )
    .unwrap();
    let target_parsed = Frame::parse(&target_frame, usize::MAX).unwrap();
    assert!(
        (200_000..=320_000).contains(&target_parsed.block_size()),
        "unexpected target_block_rows block_size={}",
        target_parsed.block_size()
    );
    assert_eq!(
        decompress(&target_frame, &DecodeOptions::default()).unwrap(),
        raw
    );

    // 4. Exact SQL query equivalence across fixed 64 KiB, auto-aligned 512 KiB, and target_block_rows=1024
    let q = SqlQuery::parse(
        "SELECT payment_type, COUNT(*), SUM(tip_cents), MAX(total_cents) FROM frame WHERE distance_centimiles >= 500 GROUP BY payment_type ORDER BY payment_type ASC",
    )
    .unwrap();
    let d4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };
    let sql_fixed = execute_sql(&fixed_64k, &q, SqlFormat::Csv, false, &d4).unwrap();
    let sql_auto = execute_sql(&auto_1t, &q, SqlFormat::Csv, false, &d4).unwrap();
    let sql_target = execute_sql(&target_frame, &q, SqlFormat::Csv, false, &d4).unwrap();
    assert_eq!(sql_fixed, sql_auto);
    assert_eq!(sql_fixed, sql_target);
}
