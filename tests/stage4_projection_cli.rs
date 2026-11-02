use vxg::{
    aggregate_column, compress, count_where, decompress, filter_where, group_by,
    ops::{line_matches_expr, project_line},
    project_frame, select_columns, ColPred, ColTarget, DecodeOptions, Error, Frame, NumCmp,
    Options, PredExpr, StrCmp, BLOCK_COLUMN,
};

fn naive_select(
    raw: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    limit: usize,
) -> Vec<u8> {
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut emitted = 0usize;
    for line in raw.split_inclusive(|&b| b == b'\n') {
        if where_expr.is_none_or(|e| line_matches_expr(line, e))
            && project_line(line, cols, &mut out)
        {
            emitted += 1;
            if emitted >= limit {
                break;
            }
        }
    }
    out
}

#[test]
fn projection_validation_rejects_invalid_column_lists_and_corrupt_frames() {
    let raw = b"{\"a\":1,\"b\":2}\n";
    let frame = compress(raw, &Options::default()).unwrap();
    let dec = DecodeOptions::default();
    let enc = Options::default();

    // Empty column list:
    assert!(matches!(
        select_columns(&frame, None, &[], 10, &dec),
        Err(Error::InvalidOption(_))
    ));
    assert!(matches!(
        project_frame(&frame, None, &[], &enc, &dec),
        Err(Error::InvalidOption(_))
    ));

    // > 128 columns:
    let too_many = vec![ColTarget::JsonKey("a"); 129];
    assert!(matches!(
        select_columns(&frame, None, &too_many, 10, &dec),
        Err(Error::InvalidOption(_))
    ));

    // Mixed JSON and Delimited:
    let mixed = [
        ColTarget::JsonKey("a"),
        ColTarget::Delimited {
            delim: b',',
            index: 0,
        },
    ];
    assert!(matches!(
        select_columns(&frame, None, &mixed, 10, &dec),
        Err(Error::InvalidOption(_))
    ));

    // Mismatched delimiters:
    let bad_delim = [
        ColTarget::Delimited {
            delim: b',',
            index: 0,
        },
        ColTarget::Delimited {
            delim: b'|',
            index: 1,
        },
    ];
    assert!(matches!(
        select_columns(&frame, None, &bad_delim, 10, &dec),
        Err(Error::InvalidOption(_))
    ));

    // Invalid newline delimiter:
    let nl_delim = [ColTarget::Delimited {
        delim: b'\n',
        index: 0,
    }];
    assert!(matches!(
        select_columns(&frame, None, &nl_delim, 10, &dec),
        Err(Error::InvalidOption(_))
    ));

    // `limit == 0` still validates corrupt frames:
    assert!(select_columns(
        b"not_a_vxg_frame",
        None,
        &[ColTarget::JsonKey("a")],
        0,
        &dec
    )
    .is_err());
    assert_eq!(
        select_columns(&frame, None, &[ColTarget::JsonKey("a")], 0, &dec).unwrap(),
        b""
    );
}

#[test]
fn json_projection_handles_reordering_duplicates_missing_nested_crlf_and_eof() {
    let mut raw = Vec::new();
    for i in 0..450usize {
        raw.extend_from_slice(
            format!(
                "{{\"seq\":{i},\"svc\":\"gw-{}\",\"nested\":{{\"k\":[{i},\"v\\n1\"]}},\"ok\":true}}\r\n",
                i % 4
            )
            .as_bytes(),
        );
        if i == 120 {
            raw.extend_from_slice(b"--- non-json banner line ---\r\n");
            raw.extend_from_slice(b"{}\r\n");
        }
    }
    raw.extend_from_slice(b"{\"seq\":9999,\"svc\":\"tail-eof\",\"ok\":false}");

    let enc = Options {
        block_size: 4096,
        checksum: true,
        ..Options::default()
    };
    let frame = compress(&raw, &enc).unwrap();

    for threads in [1usize, 4] {
        let dec = DecodeOptions {
            threads,
            ..DecodeOptions::default()
        };
        let mut enc_t = enc.clone();
        enc_t.threads = threads;

        // Reordered + nested + duplicate + absent key:
        let cols = [
            ColTarget::JsonKey("svc"),
            ColTarget::JsonKey("absent_key"),
            ColTarget::JsonKey("nested"),
            ColTarget::JsonKey("seq"),
            ColTarget::JsonKey("svc"),
        ];
        let exp_all = naive_select(&raw, None, &cols, usize::MAX);
        let got_all = select_columns(&frame, None, &cols, usize::MAX, &dec).unwrap();
        assert_eq!(got_all, exp_all);

        for lim in [1usize, 5, 121, 122, 123, 451, 452] {
            let exp_lim = naive_select(&raw, None, &cols, lim);
            let got_lim = select_columns(&frame, None, &cols, lim, &dec).unwrap();
            assert_eq!(got_lim, exp_lim, "mismatch at limit={lim}, threads={threads}");
        }

        // Completely absent keys -> `ConstJsonEmpty` (`{}\r\n` / `{}`):
        let absent_only = [ColTarget::JsonKey("no_such_1"), ColTarget::JsonKey("no_such_2")];
        let exp_empty = naive_select(&raw, None, &absent_only, usize::MAX);
        let got_empty = select_columns(&frame, None, &absent_only, usize::MAX, &dec).unwrap();
        assert_eq!(got_empty, exp_empty);

        let pframe = project_frame(&frame, None, &cols, &enc_t, &dec).unwrap();
        let pdec = decompress(&pframe, &dec).unwrap();
        assert_eq!(pdec, exp_all);
    }
}

#[test]
fn delimited_projection_handles_out_of_range_quotes_and_single_col() {
    let mut raw = Vec::new();
    for i in 0..500usize {
        raw.extend_from_slice(
            format!(
                "{i},\"svc,{}\",{},\"payload-{i}\"\n",
                i % 5,
                200 + (i % 3) * 150
            )
            .as_bytes(),
        );
        if i == 250 {
            raw.extend_from_slice(b"single_token_not_csv\n");
        }
    }
    let enc = Options {
        block_size: 4096,
        checksum: true,
        ..Options::default()
    };
    let frame = compress(&raw, &enc).unwrap();

    for threads in [1usize, 4] {
        let dec = DecodeOptions {
            threads,
            ..DecodeOptions::default()
        };
        let mut enc_t = enc.clone();
        enc_t.threads = threads;

        // Includes out-of-bounds index 7 (`can_splice == false` path) and reordered columns:
        let cols_oob = [
            ColTarget::Delimited {
                delim: b',',
                index: 2,
            },
            ColTarget::Delimited {
                delim: b',',
                index: 7,
            },
            ColTarget::Delimited {
                delim: b',',
                index: 1,
            },
        ];
        let exp_oob = naive_select(&raw, None, &cols_oob, usize::MAX);
        assert_eq!(
            select_columns(&frame, None, &cols_oob, usize::MAX, &dec).unwrap(),
            exp_oob
        );
        let pf_oob = project_frame(&frame, None, &cols_oob, &enc_t, &dec).unwrap();
        assert_eq!(decompress(&pf_oob, &dec).unwrap(), exp_oob);

        // Single-column CSV projection (`m == 1`, spliced on pure columnar blocks):
        let cols_one = [ColTarget::Delimited {
            delim: b',',
            index: 1,
        }];
        let exp_one = naive_select(&raw, None, &cols_one, usize::MAX);
        assert_eq!(
            select_columns(&frame, None, &cols_one, usize::MAX, &dec).unwrap(),
            exp_one
        );
        let pf_one = project_frame(&frame, None, &cols_one, &enc_t, &dec).unwrap();
        assert_eq!(decompress(&pf_one, &dec).unwrap(), exp_one);
    }
}

#[test]
fn spliced_project_frame_preserves_column_blocks_and_supports_chained_analytics() {
    let services = ["auth-api", "billing", "edge-proxy", "fraud-engine"];
    let mut raw = Vec::new();
    for i in 0..4000usize {
        let svc = services[i % services.len()];
        let status = if i % 12 == 0 { 500 } else { 200 };
        let bytes = 250 + (i % 97) * 13;
        let lat = 5 + (i % 40) * 7;
        raw.extend_from_slice(
            format!(
                "{{\"seq\":{i},\"ts\":\"2026-04-01T12:00:{:02}Z\",\"service\":\"{svc}\",\"status\":{status},\"bytes\":{bytes},\"latency_ms\":{lat},\"pad\":\"xxxxxxxxxxxx-{i}\"}}\n",
                i % 60
            )
            .as_bytes(),
        );
    }
    let enc = Options {
        block_size: 16384,
        threads: 4,
        checksum: true,
        ..Options::default()
    };
    let dec = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };
    let frame = compress(&raw, &enc).unwrap();

    let cols = [
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("status"),
        ColTarget::JsonKey("bytes"),
        ColTarget::JsonKey("latency_ms"),
    ];
    let projected_frame = project_frame(&frame, None, &cols, &enc, &dec).unwrap();

    // Every block in the pure single-template frame must have been spliced as `BLOCK_COLUMN`:
    let parsed_frames = Frame::parse_all(&projected_frame, usize::MAX).unwrap();
    assert!(!parsed_frames.is_empty());
    for pf in &parsed_frames {
        assert_eq!(pf.version(), 4);
        for b in pf.blocks() {
            assert_eq!(b.kind, BLOCK_COLUMN);
        }
    }

    let expected_raw = naive_select(&raw, None, &cols, usize::MAX);
    assert_eq!(decompress(&projected_frame, &dec).unwrap(), expected_raw);
    assert!(projected_frame.len() < frame.len());

    // Chained compressed-domain operations directly on the spliced projected frame:
    let e_500 = PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500)));
    let expected_500 = (0..4000usize).filter(|i| i % 12 == 0).count() as u64;
    assert_eq!(
        count_where(&projected_frame, &e_500, &dec).unwrap(),
        expected_500
    );
    assert_eq!(
        filter_where(&projected_frame, &e_500, 15, &dec)
            .unwrap()
            .len(),
        15
    );

    let st_orig =
        aggregate_column(&frame, Some(&e_500), ColTarget::JsonKey("bytes"), &dec).unwrap();
    let st_proj = aggregate_column(
        &projected_frame,
        Some(&e_500),
        ColTarget::JsonKey("bytes"),
        &dec,
    )
    .unwrap();
    assert_eq!(st_orig, st_proj);

    let gb_orig = group_by(
        &frame,
        Some(&e_500),
        ColTarget::JsonKey("service"),
        Some(ColTarget::JsonKey("latency_ms")),
        &dec,
    )
    .unwrap();
    let gb_proj = group_by(
        &projected_frame,
        Some(&e_500),
        ColTarget::JsonKey("service"),
        Some(ColTarget::JsonKey("latency_ms")),
        &dec,
    )
    .unwrap();
    assert_eq!(gb_orig, gb_proj);

    // Also verify `legacy_v2: true` path:
    let mut enc_v2 = enc.clone();
    enc_v2.legacy_v2 = true;
    let pf_v2 = project_frame(&frame, Some(&e_500), &cols, &enc_v2, &dec).unwrap();
    assert_eq!(
        decompress(&pf_v2, &dec).unwrap(),
        naive_select(&raw, Some(&e_500), &cols, usize::MAX)
    );
}

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn gen_range(&mut self, hi: usize) -> usize {
        (self.next_u64() as usize) % hi.max(1)
    }
}

#[test]
fn differential_fuzz_select_columns_and_project_frame_multi_block_carry_and_templates() {
    let svcs = ["api", "db", "auth", "edge", " worker \"q\" "];
    for seed in 0..80u64 {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x517c_c1b7_2722_0a95));
        let is_json = seed % 2 == 0;
        let rows = 80 + rng.gen_range(420);
        let crlf = seed % 5 == 0;
        let nl = if crlf { "\r\n" } else { "\n" };
        let mut raw = Vec::new();

        for r in 0..rows {
            let kind = rng.gen_range(20);
            if kind == 0 {
                // Non-structured raw line or >4096B cross-block carry monster:
                let big = "x".repeat(if rng.gen_range(3) == 0 { 4500 } else { 40 });
                if is_json && rng.gen_range(2) == 0 {
                    raw.extend_from_slice(
                        format!("{{\"seq\":{r},\"svc\":\"carry\",\"bytes\":777,\"blob\":\"{big}\"}}{nl}")
                            .as_bytes(),
                    );
                } else {
                    raw.extend_from_slice(format!("# raw comment {r} {big}{nl}").as_bytes());
                }
            } else if is_json {
                let svc = svcs[rng.gen_range(svcs.len())];
                let mut svc_json = Vec::new();
                vxg::ops::encode_json_str(svc.as_bytes(), &mut svc_json);
                let svc_s = std::str::from_utf8(&svc_json).unwrap();
                let bytes = rng.gen_range(10_000);
                let status = [200usize, 201, 404, 500][rng.gen_range(4)];
                if kind < 14 {
                    raw.extend_from_slice(
                        format!("{{\"seq\":{r},\"svc\":{svc_s},\"status\":{status},\"bytes\":{bytes}}}{nl}")
                            .as_bytes(),
                    );
                } else {
                    // Alternate template with different key order / subset:
                    raw.extend_from_slice(
                        format!("{{\"svc\":{svc_s},\"extra\":true,\"bytes\":{bytes},\"seq\":{r}}}{nl}")
                            .as_bytes(),
                    );
                }
            } else {
                let svc = svcs[rng.gen_range(4)];
                let status = [200usize, 404, 500][rng.gen_range(3)];
                let bytes = rng.gen_range(10_000);
                if kind < 15 {
                    raw.extend_from_slice(format!("{r},{svc},{status},{bytes}{nl}").as_bytes());
                } else {
                    raw.extend_from_slice(format!("{r},\"{svc}\",{status}{nl}").as_bytes());
                }
            }
        }
        if seed % 7 == 0 && raw.ends_with(b"\n") {
            raw.pop();
            if raw.ends_with(b"\r") {
                raw.pop();
            }
        }

        let enc1 = Options {
            block_size: 4096,
            threads: 1,
            checksum: seed % 3 != 0,
            ..Options::default()
        };
        let mut enc4 = enc1.clone();
        enc4.threads = 4;
        let dec1 = DecodeOptions {
            threads: 1,
            ..DecodeOptions::default()
        };
        let dec4 = DecodeOptions {
            threads: 4,
            ..DecodeOptions::default()
        };
        let frame = compress(&raw, &enc4).unwrap();

        let cols: Vec<ColTarget<'_>> = if is_json {
            let pool = ["svc", "bytes", "status", "seq", "missing"];
            let m = 1 + rng.gen_range(4);
            (0..m)
                .map(|_| ColTarget::JsonKey(pool[rng.gen_range(pool.len())]))
                .collect()
        } else {
            let m = 1 + rng.gen_range(4);
            (0..m)
                .map(|_| ColTarget::Delimited {
                    delim: b',',
                    index: rng.gen_range(5),
                })
                .collect()
        };

        let p_a = if is_json {
            PredExpr::col(ColPred::json_num("bytes", NumCmp::Ge(3000)))
        } else {
            PredExpr::col(ColPred::delimited_num(b',', 2, NumCmp::Eq(500)))
        };
        let p_b = if is_json {
            PredExpr::col(ColPred::json_str("svc", StrCmp::Eq(b"auth")))
        } else {
            PredExpr::col(ColPred::delimited_str(b',', 1, StrCmp::Ne(b"db")))
        };
        let kids = [p_a, PredExpr::not(&p_b)];
        let compound = if seed % 4 == 0 {
            None
        } else if seed % 4 == 1 {
            Some(PredExpr::all(&kids))
        } else if seed % 4 == 2 {
            Some(PredExpr::any(&kids))
        } else {
            Some(p_a)
        };
        let where_ref = compound.as_ref();

        let exp_full = naive_select(&raw, where_ref, &cols, usize::MAX);
        assert_eq!(
            select_columns(&frame, where_ref, &cols, usize::MAX, &dec1).unwrap(),
            exp_full,
            "select 1T mismatch seed={seed}"
        );
        assert_eq!(
            select_columns(&frame, where_ref, &cols, usize::MAX, &dec4).unwrap(),
            exp_full,
            "select 4T mismatch seed={seed}"
        );

        let lim = 1 + rng.gen_range(35);
        let exp_lim = naive_select(&raw, where_ref, &cols, lim);
        assert_eq!(
            select_columns(&frame, where_ref, &cols, lim, &dec1).unwrap(),
            exp_lim,
            "select limit={lim} 1T mismatch seed={seed}"
        );
        assert_eq!(
            select_columns(&frame, where_ref, &cols, lim, &dec4).unwrap(),
            exp_lim,
            "select limit={lim} 4T mismatch seed={seed}"
        );

        let pf1 = project_frame(&frame, where_ref, &cols, &enc1, &dec1).unwrap();
        let pf4 = project_frame(&frame, where_ref, &cols, &enc4, &dec4).unwrap();
        assert_eq!(
            decompress(&pf1, &dec1).unwrap(),
            exp_full,
            "project_frame 1T mismatch seed={seed}"
        );
        assert_eq!(
            decompress(&pf4, &dec4).unwrap(),
            exp_full,
            "project_frame 4T mismatch seed={seed}"
        );
    }
}
