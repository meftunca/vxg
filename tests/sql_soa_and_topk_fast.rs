use vxg::{
    compress, decompress, execute_sql, group_by, select_top_k_by, top_k_by, ColTarget,
    DecodeOptions, Options, PredExpr, SortDir, SqlFormat, SqlQuery,
};

fn build_multi_model_csv() -> Vec<u8> {
    let mut out = Vec::with_capacity(600_000);
    out.extend_from_slice(
        b"g_const,g_bin,g_quad,g_dict,packed_u12,packed_u16,packed_u20,scaled_dec,affine_id,delta_unpatched,delta_pfor\n",
    );
    let dict_vals = ["card", "cash", "dispute", "void", "wallet", "promo"];
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    let mut cum_unpatched = 50_000_u64;
    let mut cum_pfor = 10_000_u64;
    for i in 0..8500usize {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let g_const = 7u32;
        let g_bin = ((s >> 32) & 1) as u32;
        let g_quad = ((s >> 34) & 3) as u32;
        let g_dict = dict_vals[((s >> 36) as usize) % dict_vals.len()];
        let p12 = (s >> 40) & 0x0fff; // width <= 12
        let p16 = 100 + ((s >> 44) & 0xffff); // width == 16
        let p20 = 1_000 + ((s >> 20) & 0x7_ffff); // width == 19..20
        let raw_cents = ((s >> 48) as i64 % 40_000) - 5_000;
        let sign = if raw_cents < 0 { "-" } else { "" };
        let abs_c = raw_cents.unsigned_abs();
        let affine = 1_000_000 + (i as u64) * 3;

        // Smooth +/- small delta -> unpatched MODEL_DECIMAL_DELTA
        let step_small = ((s >> 28) as i64 % 9) - 3;
        cum_unpatched = (cum_unpatched as i64 + step_small).max(1_000) as u64;

        // Mostly tiny delta (0..3) with rare country-reset jumps -> PFOR-Delta (0x80 | bw)
        if i > 0 && i % 250 == 0 {
            cum_pfor = 10_000 + ((s >> 16) % 500);
        } else {
            cum_pfor += (s >> 61) & 3;
        }
        if i == 4321 {
            // Inject extreme outlier to exercise PFOR-Delta overlap in Top-K
            cum_pfor += 9_000_000;
        }

        out.extend_from_slice(
            format!(
                "{g_const},{g_bin},{g_quad},{g_dict},{p12},{p16},{p20},{sign}{}.{abs_rem:02},{affine},{cum_unpatched},{cum_pfor}\n",
                abs_c / 100,
                abs_rem = abs_c % 100,
            )
            .as_bytes(),
        );
    }
    out
}

#[test]
fn soa_group_by_and_topk_fast_paths_match_naive_across_all_models_and_threads() {
    let raw = build_multi_model_csv();
    // Use 8 KiB blocks (~75 blocks) so 8T work-stealing `par_chunk` exercises 64+ chunks & seams!
    let enc_split = compress(
        &raw,
        &Options {
            block_size: 8192,
            threads: 4,
            ..Options::default()
        },
    )
    .unwrap();
    let enc_aligned = compress(
        &raw,
        &Options {
            block_size: 8192,
            threads: 4,
            record_aligned: true,
            ..Options::default()
        },
    )
    .unwrap();
    // Raw reference frame (`level` not needed: compare 1T/8T split vs aligned AND SQL oracle)
    assert_eq!(
        decompress(&enc_split, &DecodeOptions::default()).unwrap(),
        raw
    );
    assert_eq!(
        decompress(&enc_aligned, &DecodeOptions::default()).unwrap(),
        raw
    );

    let d1 = DecodeOptions {
        threads: 1,
        ..DecodeOptions::default()
    };
    let d8 = DecodeOptions {
        threads: 8,
        ..DecodeOptions::default()
    };

    // Compare against a pure-RAW uncompressed VXG frame (forces line-by-line fallback oracle!).
    // Wait: we can compare SQL & ops results against a Python-equivalent exact Rust line parser!
    let lines: Vec<&str> = std::str::from_utf8(&raw).unwrap().lines().skip(1).collect();

    for g_idx in [0usize, 1, 2, 3] {
        let g_name = ["g_const", "g_bin", "g_quad", "g_dict"][g_idx];
        for (a_idx, a_name) in [
            (4usize, "packed_u12"),
            (5, "packed_u16"),
            (6, "packed_u20"),
            (7, "scaled_dec"),
            (8, "affine_id"),
            (9, "delta_unpatched"),
            (10, "delta_pfor"),
        ] {
            let r_split_1 = group_by(
                &enc_split,
                None,
                ColTarget::JsonKey(g_name),
                Some(ColTarget::JsonKey(a_name)),
                &d1,
            )
            .unwrap();
            let r_split_8 = group_by(
                &enc_split,
                None,
                ColTarget::JsonKey(g_name),
                Some(ColTarget::JsonKey(a_name)),
                &d8,
            )
            .unwrap();
            let r_aln_1 = group_by(
                &enc_aligned,
                None,
                ColTarget::JsonKey(g_name),
                Some(ColTarget::JsonKey(a_name)),
                &d1,
            )
            .unwrap();
            let r_aln_8 = group_by(
                &enc_aligned,
                None,
                ColTarget::JsonKey(g_name),
                Some(ColTarget::JsonKey(a_name)),
                &d8,
            )
            .unwrap();
            assert_eq!(r_split_1, r_split_8, "group_by 1T vs 8T mismatch on {g_name}/{a_name}");
            assert_eq!(r_split_1, r_aln_1, "group_by split vs aligned mismatch on {g_name}/{a_name}");
            assert_eq!(r_aln_1, r_aln_8);

            // Verify exact row counts and integer sums against raw CSV lines:
            let total_rc: u64 = r_aln_1.iter().map(|r| r.stats.row_count).sum();
            assert_eq!(total_rc, lines.len() as u64);
            if a_idx != 7 {
                let expected_sum: u128 = lines
                    .iter()
                    .map(|l| l.split(',').nth(a_idx).unwrap().parse::<u128>().unwrap())
                    .sum();
                let got_sum: u128 = r_aln_1.iter().map(|r| r.stats.sum).sum();
                assert_eq!(got_sum, expected_sum, "sum mismatch on {g_name}/{a_name}");
            }

            // Also verify SQL single-group `MultiGroupMap` fast path (`try_multi_group_single_fast`):
            let q = SqlQuery::parse(&format!(
                "SELECT {g_name}, COUNT(*), SUM({a_name}), MIN({a_name}), MAX({a_name}) FROM frame GROUP BY {g_name} ORDER BY {g_name} ASC"
            ))
            .unwrap();
            let sql_1 = execute_sql(&enc_split, &q, SqlFormat::Csv, false, &d1).unwrap();
            let sql_8 = execute_sql(&enc_aligned, &q, SqlFormat::Csv, false, &d8).unwrap();
            assert_eq!(sql_1, sql_8, "SQL GROUP BY mismatch on {g_name}/{a_name}");
        }
    }

    // Filtered GROUP BY (`TplMask::Bits`)
    let w_pred = vxg::ParsedWhere::parse("g_bin==1 && packed_u12>=500").unwrap();
    w_pred.with_expr(|w| {
        let g1 = group_by(
            &enc_split,
            Some(w),
            ColTarget::JsonKey("g_quad"),
            Some(ColTarget::JsonKey("delta_pfor")),
            &d1,
        )
        .unwrap();
        let g8 = group_by(
            &enc_aligned,
            Some(w),
            ColTarget::JsonKey("g_quad"),
            Some(ColTarget::JsonKey("delta_pfor")),
            &d8,
        )
        .unwrap();
        assert_eq!(g1, g8);
    });

    // Top-K differential checks across all numeric models, ASC & DESC, 1T & 8T:
    for (col_idx, col_name) in [
        (4usize, "packed_u12"),
        (5, "packed_u16"),
        (6, "packed_u20"),
        (7, "scaled_dec"),
        (8, "affine_id"),
        (9, "delta_unpatched"),
        (10, "delta_pfor"),
    ] {
        for dir in [SortDir::Desc, SortDir::Asc] {
            let tk_s1 = top_k_by(
                &enc_split,
                None,
                ColTarget::JsonKey(col_name),
                dir,
                12,
                &d1,
            )
            .unwrap();
            let tk_a1 = top_k_by(
                &enc_aligned,
                None,
                ColTarget::JsonKey(col_name),
                dir,
                12,
                &d1,
            )
            .unwrap();
            let tk_a8 = top_k_by(
                &enc_aligned,
                None,
                ColTarget::JsonKey(col_name),
                dir,
                12,
                &d8,
            )
            .unwrap();
            let b_s1: Vec<&[u8]> = tk_s1.iter().map(|r| r.bytes.as_slice()).collect();
            let b_a1: Vec<&[u8]> = tk_a1.iter().map(|r| r.bytes.as_slice()).collect();
            let b_a8: Vec<&[u8]> = tk_a8.iter().map(|r| r.bytes.as_slice()).collect();
            assert_eq!(b_s1, b_a1, "top_k split vs aligned mismatch on {col_name} {dir:?}");
            assert_eq!(b_a1, b_a8, "top_k 1T vs 8T mismatch on {col_name} {dir:?}");

            let pcols = [
                ColTarget::JsonKey("g_dict"),
                ColTarget::JsonKey(col_name),
            ];
            let pr_1 = select_top_k_by(
                &enc_aligned,
                None,
                &pcols,
                ColTarget::JsonKey(col_name),
                dir,
                10,
                &d1,
            )
            .unwrap();
            let pr_8 = select_top_k_by(
                &enc_split,
                None,
                &pcols,
                ColTarget::JsonKey(col_name),
                dir,
                10,
                &d8,
            )
            .unwrap();
            assert_eq!(pr_1, pr_8, "select_top_k_by mismatch on {col_name} {dir:?}");

            // Verify top-1 value against exact ground truth over raw lines:
            if col_idx != 7 {
                let mut exact: Vec<(u64, usize)> = lines
                    .iter()
                    .enumerate()
                    .map(|(ri, l)| {
                        (
                            l.split(',').nth(col_idx).unwrap().parse::<u64>().unwrap(),
                            ri,
                        )
                    })
                    .collect();
                exact.sort_by(|a, b| match dir {
                    SortDir::Desc => b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)),
                    SortDir::Asc => a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)),
                });
                let want_first = format!("{}\n", lines[exact[0].1]);
                assert_eq!(tk_a1[0].bytes, want_first.as_bytes());
            }
        }
    }
}

#[test]
fn corrupt_unpatched_delta_underflow_is_rejected_by_group_by_and_topk() {
    // Build a block whose `val` column encodes with unpatched `MODEL_DECIMAL_DELTA` (`first = 5`),
    // then corrupt a delta step in an unchecksummed frame so `cur` underflows `< 0`.
    let mut raw = Vec::new();
    let mut v = 5u64;
    for i in 0..300u64 {
        let g = i & 3;
        // Alternate +3, +1, -2 so values stay small >= 4 with smooth zigzag deltas (`d_width = 3`)
        if i > 0 {
            v = match i % 3 {
                0 => v + 3,
                1 => v + 1,
                _ => v - 2,
            };
        }
        raw.extend_from_slice(format!("{g},{v}\n").as_bytes());
    }
    let unck = compress(
        &raw,
        &Options {
            checksum: false,
            ..Options::default()
        },
    )
    .unwrap();
    let dopts = DecodeOptions::default();

    // Locate `MODEL_DECIMAL_DELTA` (`12`) + `first = 5u64` (`[5,0,0,0,0,0,0,0]`) + `d_width` (`< 0x80`).
    let mut needle = vec![12u8];
    needle.extend_from_slice(&5u64.to_le_bytes());
    let pos = unck
        .windows(10)
        .position(|w| w[..9] == needle[..] && w[9] > 0 && w[9] < 0x80)
        .expect("expected unpatched MODEL_DECIMAL_DELTA stream");

    // Set `first = 0` and fill first packed delta bytes with `0xff` (negative zigzag steps -> `< 0` underflow).
    let mut bad = unck.clone();
    bad[pos + 1..pos + 9].fill(0);
    let d_width = bad[pos + 9];
    assert!(d_width >= 1 && d_width <= 16);
    bad[pos + 10] = (1u8 << d_width) - 1; // first step odd zigzag -> negative delta from `first = 0`!

    assert!(decompress(&bad, &dopts).is_err());
    assert!(group_by(
        &bad,
        None,
        ColTarget::Delimited { delim: b',', index: 0 },
        Some(ColTarget::Delimited { delim: b',', index: 1 }),
        &dopts,
    )
    .is_err());
    assert!(top_k_by(
        &bad,
        None,
        ColTarget::Delimited { delim: b',', index: 1 },
        SortDir::Desc,
        5,
        &dopts,
    )
    .is_err());
    assert!(PredExpr::True == PredExpr::True);
}
