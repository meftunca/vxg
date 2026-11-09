//! Qualification & adversarial safety tests for:
//! 1. Exception-patched row-delta numeric column model (`PFOR-Delta`, `0x80 | base_width`)
//! 2. Nullable/empty-field and space/UTF-8 column-0 CSV parsing + compressed-domain SQL/ops parity
//! 3. `BLOCK_DICT_LZ` compression effort levels (`Options::level` `1..=9`)

use vxg::{
    aggregate_column, compress, count_where, decompress, execute_sql, filter_frame_where,
    filter_where, group_by, parse_col_list, parse_col_target, project_frame, update_column,
    verify_frame, ColOp, ColPred, ColTarget, DecodeOptions, Error, Frame, NumCmp, Options,
    PredExpr, SqlFormat, SqlQuery, StrCmp, ValuePred, BLOCK_COLUMN, BLOCK_DICT_LZ,
};

fn build_cumulative_covid_like_csv() -> Vec<u8> {
    let regions = [
        ("Afghanistan", "", 0u64, 19u64),
        ("Australia", "New South Wales", 1_400_000, 310),
        ("Brazil", "", 29_800_000, 1_950),
        ("Iceland", "", 0, 0),
        ("St. Johns", "Florida", 85_000, 12),
        ("United States", "", 31_250_000, 2_800),
        ("Zimbabwe", "", 10, 4),
    ];
    let mut out =
        Vec::from(&b"Date,Country/Region,Province/State,Confirmed,Recovered,Deaths\n"[..]);
    let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
    for &(country, prov, start, max_step) in &regions {
        let mut conf = start;
        let mut rec = start * 4 / 5;
        let mut deaths = start / 90;
        for day in 0..240u32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let step = if max_step == 0 {
                if day == 120 { 500 } else { 0 }
            } else {
                (rng >> 52) % (max_step + 1)
            };
            conf += step;
            rec += step / 2;
            deaths += (rng >> 61) % 4;
            let m = (day / 28) % 12 + 1;
            let d = (day % 28) + 1;
            out.extend_from_slice(
                format!("2021-{m:02}-{d:02},{country},{prov},{conf},{rec},{deaths}\n").as_bytes(),
            );
        }
    }
    out
}

#[test]
fn pfor_delta_routes_cumulative_csv_to_column_and_beats_ratio_target() {
    let raw = build_cumulative_covid_like_csv();
    let encoded = compress(&raw, &Options::default()).unwrap();
    let decoded = decompress(&encoded, &DecodeOptions::default()).unwrap();
    assert_eq!(decoded, raw);

    let parsed = Frame::parse(&encoded, 1 << 28).unwrap();
    let blocks: Vec<_> = parsed.blocks().collect();
    assert!(!blocks.is_empty());
    for (idx, info) in blocks.iter().enumerate() {
        assert_eq!(
            info.kind, BLOCK_COLUMN,
            "block {idx} fell back to kind={} instead of BLOCK_COLUMN",
            info.kind
        );
    }

    let ratio = encoded.len() as f64 / raw.len() as f64;
    assert!(
        ratio < 0.12,
        "expected PFOR-Delta ratio < 0.12 on cumulative time-series CSV, got {ratio:.4}"
    );
}

#[test]
fn pfor_delta_base_width_zero_and_64bit_spans_roundtrip_and_query() {
    // Plateau series where >95% of consecutive steps are 0 (`base_width == 0`),
    // plus 64-bit large counter values (`> 10^14`) that reset across partitions.
    let mut raw = Vec::from(&b"series,seq,counter_small,counter_64\n"[..]);
    for part in 0..6u64 {
        let mut small = part * 250;
        let mut big = part * 500_000_000_000_000u64 + 10_000_000_000_000u64;
        for row in 0..120u64 {
            if row == 40 || row == 95 {
                small += 1_500;
                big += 9_000_000_000;
            }
            raw.extend_from_slice(format!("p{part},{row},{small},{big}\n").as_bytes());
        }
    }

    let enc = compress(&raw, &Options::default()).unwrap();
    assert_eq!(decompress(&enc, &DecodeOptions::default()).unwrap(), raw);
    let parsed = Frame::parse(&enc, 1 << 26).unwrap();
    assert_eq!(parsed.blocks().next().unwrap().kind, BLOCK_COLUMN);

    // Zone-map pruning check: value outside `[min, max]` returns 0 rows.
    let above_expr = PredExpr::col(ColPred {
        col: parse_col_target("csv:3").unwrap(),
        pred: ValuePred::Num(NumCmp::Gt(9_000_000_000_000_000)),
    });
    let above_max = count_where(&enc, &above_expr, &DecodeOptions::default()).unwrap();
    assert_eq!(above_max, 0);

    // Numeric `--add` rewrite (`update_column`) on PFOR-Delta columns (`DecimalSmall` & `DecimalValues`).
    let where_pred = ColPred {
        col: parse_col_target("csv:0").unwrap(),
        pred: ValuePred::Str(StrCmp::Eq(b"p2")),
    };
    let updated = update_column(
        &enc,
        &where_pred,
        parse_col_target("csv:2").unwrap(),
        ColOp::MapU64(&|v| v + 7),
        &Options::default(),
        &DecodeOptions::default(),
    )
    .unwrap();
    let updated_raw = decompress(&updated, &DecodeOptions::default()).unwrap();
    let updated_text = std::str::from_utf8(&updated_raw).unwrap();
    assert!(updated_text.contains("p2,0,507,"));
    assert!(updated_text.contains("p1,0,250,"));
}

#[test]
fn nullable_and_space_col0_csv_sql_and_ops_differential_parity() {
    // Exercises leading empty `,`, middle empty `,,`, trailing empty `,`,
    // spaces/periods/UTF-8/parens in column 0 (`St. Johns`, `Nez Perce`, `Côte d'Ivoire (West)`),
    // and cumulative PFOR-Delta counters.
    let mut csv = String::from("county,state,note,confirmed,deaths,tag\n");
    let rows = [
        ("St. Johns", "Florida", "", 12_000u64, 140u64, ""),
        ("Nez Perce", "Idaho", "rural", 4_100, 55, "nw"),
        ("Côte d'Ivoire (West)", "", "", 890, 12, ""),
        ("", "Unknown", "missing_county", 50, 1, ""),
    ];
    for round in 0..90u64 {
        for &(c, st, note, base_c, base_d, tag) in &rows {
            let conf = base_c + round * 3;
            let deaths = base_d + (round % 2);
            csv.push_str(&format!("{c},{st},{note},{conf},{deaths},{tag}\n"));
        }
    }
    let raw = csv.into_bytes();
    let enc = compress(&raw, &Options::default()).unwrap();
    assert_eq!(decompress(&enc, &DecodeOptions::default()).unwrap(), raw);
    let parsed = Frame::parse(&enc, 1 << 26).unwrap();
    for info in parsed.blocks() {
        assert_eq!(info.kind, BLOCK_COLUMN);
    }

    let dopts = DecodeOptions::default();

    // 1. SQL filter on empty string `state = ''` and space county `county = 'St. Johns'`
    let q_empty =
        SqlQuery::parse("SELECT COUNT(*), SUM(confirmed) FROM frame WHERE state = ''").unwrap();
    let sql_empty_state = execute_sql(&enc, &q_empty, SqlFormat::Csv, false, &dopts).unwrap();
    let expected_sum_cote: u64 = (0..90u64).map(|r| 890 + r * 3).sum();
    assert_eq!(
        String::from_utf8(sql_empty_state).unwrap().trim(),
        format!("count(*),sum(confirmed)\n90,{expected_sum_cote}")
    );

    let q_space = SqlQuery::parse(
        "SELECT county, MAX(confirmed), MIN(deaths) FROM frame WHERE county = 'St. Johns' GROUP BY county",
    )
    .unwrap();
    let sql_space_col0 = execute_sql(&enc, &q_space, SqlFormat::Csv, false, &dopts).unwrap();
    assert_eq!(
        String::from_utf8(sql_space_col0).unwrap().trim(),
        format!(
            "county,max(confirmed),min(deaths)\nSt. Johns,{},140",
            12_000 + 89 * 3
        )
    );

    // 2. Filter on leading-empty column 0 (`county = ''`) and trailing-empty column 5 (`tag = ''`)
    let children = [
        PredExpr::col(ColPred {
            col: ColTarget::Delimited { delim: b',', index: 0 },
            pred: ValuePred::Str(StrCmp::Eq(b"")),
        }),
        PredExpr::col(ColPred {
            col: ColTarget::Delimited { delim: b',', index: 5 },
            pred: ValuePred::Str(StrCmp::Eq(b"")),
        }),
    ];
    let compound = PredExpr::all(&children);
    let leading_empty_hits = filter_where(&enc, &compound, usize::MAX, &dopts).unwrap();
    // Header does not have empty col 0; all 90 `"",Unknown,...` rows must match.
    assert_eq!(leading_empty_hits.len(), 90);
    assert!(leading_empty_hits[0]
        .bytes
        .starts_with(b",Unknown,missing_county,50,1,"));

    // 3. Projection (`project_frame`) preserving empty fields (`county,note,tag`)
    let cols = parse_col_list("csv:0,csv:2,csv:5").unwrap();
    let proj_frame = project_frame(&enc, None, &cols, &Options::default(), &dopts).unwrap();
    let proj_raw = decompress(&proj_frame, &dopts).unwrap();
    let proj_lines: Vec<&str> = std::str::from_utf8(&proj_raw).unwrap().lines().collect();
    assert_eq!(proj_lines[0], "county,note,tag");
    assert_eq!(proj_lines[1], "St. Johns,,");
    assert_eq!(proj_lines[2], "Nez Perce,rural,nw");
    assert_eq!(proj_lines[3], "Côte d'Ivoire (West),,");
    assert_eq!(proj_lines[4], ",missing_county,");

    // 4. `aggregate_column` & `group_by` differential check against `legacy_v2` non-columnar frame
    let v2_enc = compress(
        &raw,
        &Options {
            legacy_v2: true,
            ..Options::default()
        },
    )
    .unwrap();
    let pred = PredExpr::col(ColPred {
        col: ColTarget::Delimited { delim: b',', index: 5 },
        pred: ValuePred::Str(StrCmp::Eq(b"")),
    });
    let agg_col = aggregate_column(
        &enc,
        Some(&pred),
        ColTarget::Delimited { delim: b',', index: 3 },
        &dopts,
    )
    .unwrap();
    let agg_v2 = aggregate_column(
        &v2_enc,
        Some(&pred),
        ColTarget::Delimited { delim: b',', index: 3 },
        &dopts,
    )
    .unwrap();
    assert_eq!(agg_col, agg_v2);

    let gb_col = group_by(
        &enc,
        None,
        ColTarget::Delimited { delim: b',', index: 2 },
        Some(ColTarget::Delimited { delim: b',', index: 3 }),
        &dopts,
    )
    .unwrap();
    let gb_v2 = group_by(
        &v2_enc,
        None,
        ColTarget::Delimited { delim: b',', index: 2 },
        Some(ColTarget::Delimited { delim: b',', index: 3 }),
        &dopts,
    )
    .unwrap();
    assert_eq!(gb_col, gb_v2);

    // 5. Substring `grep` (`filter_frame_where`) across PFOR-Delta & space col 0
    let grep_expr = PredExpr::contains(b"Nez Perce");
    let grep_col = filter_frame_where(&enc, &grep_expr, &Options::default(), &dopts).unwrap();
    let grep_v2 = filter_frame_where(&v2_enc, &grep_expr, &Options::default(), &dopts).unwrap();
    assert_eq!(
        decompress(&grep_col, &dopts).unwrap(),
        decompress(&grep_v2, &dopts).unwrap()
    );
}

#[test]
fn adversarial_corrupt_pfor_delta_frames_never_panic_or_silently_corrupt() {
    let raw = build_cumulative_covid_like_csv();
    let checked_frame = compress(&raw, &Options::default()).unwrap();
    let unck_frame = compress(
        &raw,
        &Options {
            checksum: false,
            ..Options::default()
        },
    )
    .unwrap();
    let dopts = DecodeOptions::default();

    // Locate `MODEL_DECIMAL_DELTA` (`tag = 12`) followed by `first.to_le_bytes()` and `0x80 | bw`.
    let first_data_line = std::str::from_utf8(&raw).unwrap().lines().nth(1).unwrap();
    let first_confirmed: u64 = first_data_line.split(',').nth(3).unwrap().parse().unwrap();
    let mut needle = Vec::with_capacity(9);
    needle.push(12u8); // MODEL_DECIMAL_DELTA
    needle.extend_from_slice(&first_confirmed.to_le_bytes());
    let hdr_pos = unck_frame
        .windows(10)
        .position(|w| &w[..9] == needle.as_slice() && (w[9] & 0x80) != 0)
        .expect("expected PFOR-Delta MODEL_DECIMAL_DELTA header in cumulative CSV frame");

    // Mutation A: invalid `base_width = 64` (`0x80 | 64 = 0xc0`)
    let mut bad_bw = unck_frame.clone();
    bad_bw[hdr_pos + 9] = 0xc0;
    assert!(decompress(&bad_bw, &dopts).is_err());

    // Mutation B: clear `PFOR_DELTA_FLAG` so stream length/layout mismatches
    let mut bad_flag = unck_frame.clone();
    bad_flag[hdr_pos + 9] &= 0x7f;
    assert!(decompress(&bad_flag, &dopts).is_err());

    // Mutation C: overlong non-canonical `uleb64` (`min_val = 0` -> `0x80` high bit set)
    let mut bad_uleb = unck_frame.clone();
    bad_uleb[hdr_pos + 10] |= 0x80;
    assert!(decompress(&bad_uleb, &dopts).is_err());

    // Mutation D: `span = 0` so cumulative steps exceed `min_val + span`
    let mut bad_span = unck_frame.clone();
    bad_span[hdr_pos + 11] = 0x00;
    assert!(decompress(&bad_span, &dopts).is_err());

    // Fuzz 300 deterministic mutations across both checksummed and unchecksummed frames.
    let q_sum = SqlQuery::parse("SELECT SUM(Confirmed) FROM frame").unwrap();
    let mut state = 0xd1b5_4a32_d192_ed03_u64;
    for case in 0..300usize {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut mutated = if case % 2 == 0 {
            checked_frame.clone()
        } else {
            unck_frame.clone()
        };
        let pos = ((state >> 16) as usize) % mutated.len();
        match case % 3 {
            0 => mutated[pos] ^= 1u8 << ((state >> 8) & 7),
            1 => mutated.truncate(pos),
            _ => {
                let end = (pos + 32).min(mutated.len());
                // Ensure at least 1 byte actually changes on checksummed frames:
                mutated[pos] ^= 0xff;
                if pos + 1 < end {
                    mutated[pos + 1..end].fill(0);
                }
            }
        }
        let res = decompress(&mutated, &dopts);
        if case % 2 == 0 {
            assert!(
                res.is_err(),
                "checksummed corrupt PFOR-Delta frame {case} at pos={pos} was not rejected"
            );
            assert!(verify_frame(&mutated, &dopts).is_err());
        } else {
            let _ = execute_sql(&mutated, &q_sum, SqlFormat::Csv, false, &dopts);
        }
    }
}

#[test]
fn dict_lz_compression_levels_validate_roundtrip_and_improve_ratio() {
    assert!(matches!(
        compress(
            b"hello",
            &Options {
                level: 0,
                ..Options::default()
            }
        ),
        Err(Error::InvalidOption(_))
    ));
    assert!(matches!(
        compress(
            b"hello",
            &Options {
                level: 10,
                ..Options::default()
            }
        ),
        Err(Error::InvalidOption(_))
    ));

    let mut log = Vec::with_capacity(270_000);
    let mut s = 0x517c_c1b7_2722_0a95_u64;
    let lines = [
        "kernel: memorystatus_thread: idle exiting pid",
        "systemd-journald[411]: Suppressed messages from /system.slice/docker.service",
        "kubelet[1922]: E1109 pod_workers.go:1298] Error syncing pod default/api-gateway",
        "postgres[9021]: [4-1] LOG: automatic vacuum of table \"prod.public.events\": index scans: 1",
        "sshd[31004]: Invalid user deploy from 203.0.113.",
    ];
    for i in 0..2800u32 {
        s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        let msg = lines[((s >> 32) as usize) % lines.len()];
        let node = (s >> 40) % 24;
        let id = (s >> 16) & 0xffff_ffff;
        let lat = 5 + ((s >> 48) % 18_000);
        if i % 5 == 0 {
            log.extend_from_slice(
                format!("--- diagnostic note #{i}: host=node-{node:02} trace={id:08x} ({lat} us) ---\n")
                    .as_bytes(),
            );
        } else {
            log.extend_from_slice(
                format!(
                    "Nov  9 11:{:02}:{:02} prod-node-{node:02} {msg} req_id={id} took={lat}us\n",
                    (i / 60) % 60,
                    i % 60,
                )
                .as_bytes(),
            );
        }
    }

    let mut sizes = [0usize; 10];
    for lvl in 1..=9u8 {
        for threads in [1usize, 4] {
            let enc = compress(
                &log,
                &Options {
                    level: lvl,
                    threads,
                    ..Options::default()
                },
            )
            .unwrap();
            let dec = decompress(
                &enc,
                &DecodeOptions {
                    threads,
                    ..DecodeOptions::default()
                },
            )
            .unwrap();
            assert_eq!(dec, log, "roundtrip mismatch at level={lvl} threads={threads}");
            if threads == 1 {
                let parsed = Frame::parse(&enc, 1 << 26).unwrap();
                assert_eq!(parsed.blocks().next().unwrap().kind, BLOCK_DICT_LZ);
                sizes[lvl as usize] = enc.len();
            }
        }
    }
    assert!(
        sizes[9] <= sizes[3] && sizes[3] < sizes[1],
        "expected level 9 ({}) <= level 3 ({}) < level 1 ({})",
        sizes[9],
        sizes[3],
        sizes[1]
    );
}
