use vxg::{
    compress, decompress, filter_column, filter_frame, filter_json_u64, filter_rows, find_first,
    frame_row_count, map_column, update_column, ColOp, ColPred, ColTarget, DecodeOptions, Frame,
    NumCmp, Options, RowMatch, StrCmp,
};

fn naive_filter_rows(data: &[u8], needle: &[u8], limit: usize) -> Vec<(usize, Vec<u8>)> {
    if limit == 0 || data.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut row_idx = 0usize;
    while pos < data.len() {
        let end = match data[pos..].iter().position(|&b| b == b'\n') {
            Some(off) => pos + off + 1,
            None => data.len(),
        };
        let row = &data[pos..end];
        let matched = needle.is_empty() || row.windows(needle.len()).any(|w| w == needle);
        if matched {
            out.push((row_idx, row.to_vec()));
            if out.len() >= limit {
                break;
            }
        }
        pos = end;
        row_idx += 1;
    }
    out
}

fn to_pairs(rows: Vec<RowMatch>) -> Vec<(usize, Vec<u8>)> {
    rows.into_iter()
        .map(|m| (m.global_row_index, m.bytes))
        .collect()
}

fn opt_pair(row: Option<RowMatch>) -> Option<(usize, Vec<u8>)> {
    row.map(|m| (m.global_row_index, m.bytes))
}

#[test]
fn hex_suffix_nibble_scanner_all_alignments_and_lengths() {
    let mut raw = Vec::new();
    // Mix even-width (12 hex) and odd-width (9 hex) suffixes across multiple blocks.
    for i in 0..1200u64 {
        let hex12 = format!("{:012x}", 0x10a3_b5c7_d9e0u64.wrapping_add(i * 0x13579bdf));
        let hex9 = format!("{:09x}", 0x0a1_b2c_3d4u64.wrapping_add(i * 0x10357));
        let line = format!(
            "{{\"id\":{},\"trace\":\"trc-{}-end\",\"span\":\"sp-{}\"}}\n",
            i, hex12, hex9
        );
        raw.extend_from_slice(line.as_bytes());
    }

    let enc = Options {
        block_size: 4096,
        threads: 2,
        checksum: true,
        ..Options::default()
    };
    let frame = compress(&raw, &enc).unwrap();
    assert_eq!(decompress(&frame, &DecodeOptions::default()).unwrap(), raw);

    let dec1 = DecodeOptions {
        threads: 1,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };

    // Sample actual lines to extract guaranteed hit needles at even & odd nibble positions,
    // plus guaranteed 0-hit hex needles that still use only [0-9a-f].
    let sample_row = naive_filter_rows(&raw, b"\"id\":517,", 1).pop().unwrap().1;
    let sample_str = std::str::from_utf8(&sample_row).unwrap();
    let trc_pos = sample_str.find("trc-").unwrap() + 4;
    let hex12_sample = &sample_str.as_bytes()[trc_pos..trc_pos + 12];
    let sp_pos = sample_str.find("sp-").unwrap() + 3;
    let hex9_sample = &sample_str.as_bytes()[sp_pos..sp_pos + 9];

    let mut needles: Vec<Vec<u8>> = vec![
        b"ffffffffff".to_vec(),
        b"deadbeefcafe".to_vec(),
        b"000000000".to_vec(),
        b"abcdef012345".to_vec(),
        // Boundary-spanning needles (prefix+hex and hex+suffix)
        format!("trc-{}", std::str::from_utf8(&hex12_sample[..5]).unwrap()).into_bytes(),
        format!("{}-end", std::str::from_utf8(&hex12_sample[7..]).unwrap()).into_bytes(),
        b"trc-fffff".to_vec(),
        b"fffff-end".to_vec(),
    ];
    // Every length 1..=12 at both even (0) and odd (1) nibble offsets.
    for len in 1..=12 {
        needles.push(hex12_sample[0..len].to_vec());
        if 1 + len <= 12 {
            needles.push(hex12_sample[1..1 + len].to_vec());
        }
    }
    for len in 1..=9 {
        needles.push(hex9_sample[0..len].to_vec());
        if 1 + len <= 9 {
            needles.push(hex9_sample[1..1 + len].to_vec());
        }
    }

    for needle in &needles {
        let exp_all = naive_filter_rows(&raw, needle, usize::MAX);
        let got_1t = to_pairs(filter_rows(&frame, needle, usize::MAX, &dec1).unwrap());
        let got_4t = to_pairs(filter_rows(&frame, needle, usize::MAX, &dec4).unwrap());
        assert_eq!(got_1t, exp_all, "1T mismatch for {:?}", String::from_utf8_lossy(needle));
        assert_eq!(got_4t, exp_all, "4T mismatch for {:?}", String::from_utf8_lossy(needle));

        for lim in [1usize, 5usize] {
            let exp_lim = naive_filter_rows(&raw, needle, lim);
            assert_eq!(to_pairs(filter_rows(&frame, needle, lim, &dec1).unwrap()), exp_lim);
            assert_eq!(to_pairs(filter_rows(&frame, needle, lim, &dec4).unwrap()), exp_lim);
        }

        let first_1t = opt_pair(find_first(&frame, needle, &dec1).unwrap());
        let first_4t = opt_pair(find_first(&frame, needle, &dec4).unwrap());
        assert_eq!(first_1t, exp_all.first().cloned());
        assert_eq!(first_4t, exp_all.first().cloned());
    }
}

#[test]
fn decimal_timestamp_pattern_front_dict_and_cross_boundary_bigrams() {
    let routes = [
        "/api/v1/orders/create",
        "/api/v1/orders/cancel",
        "/api/v1/orders/refund",
        "/api/v1/users/session",
        "/api/v1/users/profile",
        "/api/v1/payments/authorize",
        "/api/v1/payments/capture",
        "/api/v1/payments/void",
        "/api/v1/inventory/reserve",
        "/api/v1/inventory/release",
    ];
    let mut raw = Vec::new();
    for i in 0..1500u64 {
        let sec = i % 60;
        let min = (i / 60) % 60;
        let ts = format!("2026-05-10T12:{:02}:{:02}Z", min, sec);
        let code = format!("USR-{:04}-{}X", (i * 7) % 10000, (b'A' + (i % 6) as u8) as char);
        let bytes = 1000 + (i % 500);
        let big = 100_000 + i * 3;
        let route = routes[(i as usize * 3) % routes.len()];
        let line = format!(
            "{{\"ts\":\"{}\",\"code\":\"{}\",\"route\":\"{}\",\"bytes\":{},\"seq\":{}}}\n",
            ts, code, route, bytes, big
        );
        raw.extend_from_slice(line.as_bytes());
    }

    let enc = Options {
        block_size: 4096,
        threads: 2,
        checksum: true,
        ..Options::default()
    };
    let frame = compress(&raw, &enc).unwrap();
    let dec1 = DecodeOptions {
        threads: 1,
        ..DecodeOptions::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };

    let test_needles: &[&[u8]] = &[
        // Decimal range pre-check (0-hit & hits)
        b"9999",
        b"0999",
        b"1499",
        b"104497",
        // Timestamp pre-check (0-hit & hits)
        b"2025-05",
        b"T13:",
        b":59Z",
        b"12:15:30Z",
        // Pattern positional mask (0-hit & hits)
        b"USR-9999-ZX",
        b"AX-0007",
        b"USR-0021-DX",
        // Front-dict isolated & boundary-spanning
        b"/api/v2/",
        b"/orders/delete",
        b"/payments/capture",
        b"\"route\":\"/api/v1/users/profile\"",
        // Cross-boundary bigrams where all individual bytes exist in the block
        b"Z\",\"route\":\"/api",
        b"Z\",\"bytes\":1000",
        b"X\",\"route\":\"/api/v1/orders/refund\",\"bytes\":1",
        b"999,\"seq\":999",
    ];

    for &needle in test_needles {
        let exp = naive_filter_rows(&raw, needle, usize::MAX);
        assert_eq!(
            to_pairs(filter_rows(&frame, needle, usize::MAX, &dec1).unwrap()),
            exp,
            "1T bigram mismatch for {:?}",
            String::from_utf8_lossy(needle)
        );
        assert_eq!(
            to_pairs(filter_rows(&frame, needle, usize::MAX, &dec4).unwrap()),
            exp,
            "4T bigram mismatch for {:?}",
            String::from_utf8_lossy(needle)
        );
        assert_eq!(opt_pair(find_first(&frame, needle, &dec1).unwrap()), exp.first().cloned());
        assert_eq!(opt_pair(find_first(&frame, needle, &dec4).unwrap()), exp.first().cloned());
        let exp3 = naive_filter_rows(&raw, needle, 3);
        assert_eq!(to_pairs(filter_rows(&frame, needle, 3, &dec4).unwrap()), exp3);
    }

    // Parallel numeric JSON query (`1T` vs `4T`, full and limited)
    let j1 = filter_json_u64(&frame, "bytes", NumCmp::Ge(1490), usize::MAX, &dec1).unwrap();
    let j4 = filter_json_u64(&frame, "bytes", NumCmp::Ge(1490), usize::MAX, &dec4).unwrap();
    assert!(!j1.is_empty());
    assert_eq!(j1, j4);
    let j_lim1 = filter_json_u64(&frame, "bytes", NumCmp::Ge(1490), 4, &dec1).unwrap();
    let j_lim4 = filter_json_u64(&frame, "bytes", NumCmp::Ge(1490), 4, &dec4).unwrap();
    assert_eq!(j_lim1.len(), 4);
    assert_eq!(j_lim1, j_lim4);
}

#[test]
fn direct_binary_numeric_and_dict_column_rewrites() {
    let services = ["auth", "billing", "fraud", "gateway"];
    let mut json_raw = Vec::new();
    let mut csv_raw = Vec::new();
    for i in 0..1000u64 {
        let svc = services[(i as usize) % services.len()];
        let small_n = (i * 13) % 900; // DecimalSmall (< 65536), crosses 2->3->4 digit widths on +150
        let big_n = 70_000 + (i * 29) % 50_000; // DecimalPacked / DecimalValues (> 65535)
        let seq_n = 5000 + i; // DecimalAffineSmall
        json_raw.extend_from_slice(
            format!(
                "{{\"svc\":\"{}\",\"small\":{},\"big\":{},\"seq\":{}}}\n",
                svc, small_n, big_n, seq_n
            )
            .as_bytes(),
        );
        csv_raw.extend_from_slice(
            format!("{},{},{},{}\n", svc, small_n, big_n, seq_n).as_bytes(),
        );
    }

    let enc1 = Options {
        block_size: 4096,
        threads: 1,
        checksum: true,
        ..Options::default()
    };
    let enc4 = Options {
        block_size: 4096,
        threads: 4,
        checksum: true,
        ..Options::default()
    };
    let dec4 = DecodeOptions {
        threads: 4,
        ..DecodeOptions::default()
    };

    let jf = compress(&json_raw, &enc4).unwrap();
    let cf = compress(&csv_raw, &enc4).unwrap();

    // 1. Direct binary numeric map on DecimalSmall (with digit-width growth), DecimalPacked, DecimalAffine
    let add_150 = |x: u64| x + 150;
    let jf_small = map_column(&jf, ColTarget::JsonKey("small"), ColOp::MapU64(&add_150), &enc4, &dec4).unwrap();
    let jf_small_1t = map_column(&jf, ColTarget::JsonKey("small"), ColOp::MapU64(&add_150), &enc1, &dec4).unwrap();
    let mut exp_json_small = Vec::new();
    for i in 0..1000u64 {
        let svc = services[(i as usize) % services.len()];
        let small_n = ((i * 13) % 900) + 150;
        let big_n = 70_000 + (i * 29) % 50_000;
        let seq_n = 5000 + i;
        exp_json_small.extend_from_slice(
            format!(
                "{{\"svc\":\"{}\",\"small\":{},\"big\":{},\"seq\":{}}}\n",
                svc, small_n, big_n, seq_n
            )
            .as_bytes(),
        );
    }
    assert_eq!(decompress(&jf_small, &dec4).unwrap(), exp_json_small);
    assert_eq!(decompress(&jf_small_1t, &dec4).unwrap(), exp_json_small);
    assert_eq!(frame_row_count(&jf_small).unwrap(), 1000);

    // Fixed-width numeric/dict rewrites preserve a single v4 frame (`Frame::parse`).
    let upper_svc = |s: &[u8]| -> Vec<u8> { s.iter().map(|b| b.to_ascii_uppercase()).collect() };
    let jf_upper =
        map_column(&jf, ColTarget::JsonKey("svc"), ColOp::MapStr(&upper_svc), &enc4, &dec4).unwrap();
    assert!(Frame::parse(&jf_upper, 1 << 30).is_ok());

    // 2. MapStr on JSON numeric column must leave unquoted integers untouched,
    // whereas MapStr on CSV numeric column MUST transform them!
    let bang = |s: &[u8]| {
        let mut v = s.to_vec();
        v.push(b'0');
        v
    };
    let jf_mapstr_num =
        map_column(&jf, ColTarget::JsonKey("small"), ColOp::MapStr(&bang), &enc4, &dec4).unwrap();
    assert_eq!(decompress(&jf_mapstr_num, &dec4).unwrap(), json_raw);

    let cf_mapstr_num = map_column(
        &cf,
        ColTarget::Delimited { delim: b',', index: 1 },
        ColOp::MapStr(&bang),
        &enc4,
        &dec4,
    )
    .unwrap();
    let mut exp_csv_bang = Vec::new();
    for i in 0..1000u64 {
        let svc = services[(i as usize) % services.len()];
        let small_n = (i * 13) % 900;
        let big_n = 70_000 + (i * 29) % 50_000;
        let seq_n = 5000 + i;
        exp_csv_bang.extend_from_slice(format!("{},{}0,{},{}\n", svc, small_n, big_n, seq_n).as_bytes());
    }
    assert_eq!(decompress(&cf_mapstr_num, &dec4).unwrap(), exp_csv_bang);

    // 3. Conditional numeric SetRaw & Dict deduplicating MapStr + conditional Dict update
    let pred_big = ColPred::json_num("big", NumCmp::Ge(110_000));
    let jf_upd = update_column(
        &jf,
        &pred_big,
        ColTarget::JsonKey("seq"),
        ColOp::SetRaw(b"0"),
        &enc4,
        &dec4,
    )
    .unwrap();
    let collapse_svc = |s: &[u8]| -> Vec<u8> {
        if s == b"auth" || s == b"gateway" {
            b"EDGE".to_vec()
        } else {
            s.iter().map(|b| b.to_ascii_uppercase()).collect()
        }
    };
    let jf_chained = map_column(
        &jf_upd,
        ColTarget::JsonKey("svc"),
        ColOp::MapStr(&collapse_svc),
        &enc4,
        &dec4,
    )
    .unwrap();

    let mut exp_chained = Vec::new();
    for i in 0..1000u64 {
        let svc_bytes = collapse_svc(services[(i as usize) % services.len()].as_bytes());
        let svc = std::str::from_utf8(&svc_bytes).unwrap();
        let small_n = (i * 13) % 900;
        let big_n = 70_000 + (i * 29) % 50_000;
        let seq_n = if big_n >= 110_000 { 0 } else { 5000 + i };
        exp_chained.extend_from_slice(
            format!(
                "{{\"svc\":\"{}\",\"small\":{},\"big\":{},\"seq\":{}}}\n",
                svc, small_n, big_n, seq_n
            )
            .as_bytes(),
        );
    }
    assert_eq!(decompress(&jf_chained, &dec4).unwrap(), exp_chained);
}

#[test]
fn filter_frame_zero_hit_partial_and_all_match_passthrough() {
    let mut raw = Vec::new();
    for i in 0..1200u64 {
        let status = if i % 20 == 0 { 500 } else { 200 };
        let region = if i % 2 == 0 { "eu-central" } else { "us-east" };
        raw.extend_from_slice(
            format!("{{\"id\":{},\"status\":{},\"region\":\"{}\"}}\n", i, status, region).as_bytes(),
        );
    }

    for checksum in [true, false] {
        let enc = Options {
            block_size: 4096,
            threads: 4,
            checksum,
            ..Options::default()
        };
        let dec = DecodeOptions {
            threads: 4,
            verify_checksum: checksum,
            ..DecodeOptions::default()
        };
        let frame = compress(&raw, &enc).unwrap();

        // 0% match
        let p_zero = ColPred::json_num("status", NumCmp::Eq(999));
        let f_zero = filter_frame(&frame, &p_zero, &enc, &dec).unwrap();
        assert!(decompress(&f_zero, &dec).unwrap().is_empty());

        // 100% match (exercises zero-copy AllCompleteMatch compressed block pass-through!)
        let p_all = ColPred::json_num("status", NumCmp::Ge(200));
        let f_all = filter_frame(&frame, &p_all, &enc, &dec).unwrap();
        assert_eq!(decompress(&f_all, &dec).unwrap(), raw);

        // Partial match (~5% rows) + verify filter_column parity
        let p_500 = ColPred::json_num("status", NumCmp::Eq(500));
        let f_part = filter_frame(&frame, &p_500, &enc, &dec).unwrap();
        let rows_part = filter_column(&frame, &p_500, usize::MAX, &dec).unwrap();
        let mut expected_bytes = Vec::new();
        for r in &rows_part {
            expected_bytes.extend_from_slice(&r.bytes);
        }
        assert_eq!(rows_part.len(), 60);
        assert_eq!(decompress(&f_part, &dec).unwrap(), expected_bytes);

        // String contains filter_frame
        let p_eu = ColPred::json_str("region", StrCmp::Contains(b"central"));
        let f_eu = filter_frame(&frame, &p_eu, &enc, &dec).unwrap();
        assert_eq!(decompress(&f_eu, &dec).unwrap().iter().filter(|&&b| b == b'\n').count(), 600);
    }
}
