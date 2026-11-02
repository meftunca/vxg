use vxg::{
    compress, decompress, extract_row_range, filter_column, filter_frame, frame_row_count,
    map_column,
    ops::{apply_line_col_op, line_matches_col_pred},
    update_column, ColOp, ColPred, ColTarget, DecodeOptions, Frame, NumCmp, Options, StrCmp,
    ValuePred,
};

fn naive_filter(raw: &[u8], pred: &ColPred<'_>, limit: usize) -> Vec<(usize, Vec<u8>)> {
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (idx, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if line_matches_col_pred(line, pred) {
            out.push((idx, line.to_vec()));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn naive_transform(
    raw: &[u8],
    where_pred: Option<&ColPred<'_>>,
    target: ColTarget<'_>,
    op: &ColOp<'_>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for line in raw.split_inclusive(|&b| b == b'\n') {
        if let Some(nl) = apply_line_col_op(line, where_pred, target, op) {
            out.extend_from_slice(&nl);
        } else {
            out.extend_from_slice(line);
        }
    }
    out
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn usize(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound.max(1)
    }
    fn bool(&mut self) -> bool {
        (self.next() & 1) != 0
    }
}

#[test]
fn zero_hit_and_fixed_width_preserve_single_frame_and_models() {
    let mut raw = Vec::new();
    for i in 0..2400u64 {
        let lvl = match i % 10 {
            0 => "ERROR",
            1 | 2 => "WARN_",
            _ => "INFO_",
        };
        let line = format!(
            "{{\"ts\":\"2026-05-01T12:{:02}:{:02}Z\",\"ip\":\"10.{}.{}.{}\",\"hex\":\"{:016x}\",\"level\":\"{lvl}\",\"code\":{},\"seq\":{}}}\n",
            (i / 60) % 60,
            i % 60,
            (i >> 8) & 255,
            (i >> 4) & 15,
            i & 15,
            0xdead_beef_0000_0000u64 + i * 17,
            200 + (i % 5) * 100,
            10_000 + i * 3,
        );
        raw.extend_from_slice(line.as_bytes());
    }
    let enc = Options {
        block_size: 65536,
        checksum: true,
        ..Options::default()
    };
    let dec = DecodeOptions::default();
    let frame = compress(&raw, &enc).unwrap();
    assert!(Frame::parse(&frame, usize::MAX).unwrap().block_count() >= 3);

    // 1. Zero-hit update: untouched blocks copied byte-for-byte -> identical frame bytes!
    let miss_pred = ColPred::json_str("level", StrCmp::Eq(b"NONEXISTENT"));
    let untouched = update_column(
        &frame,
        &miss_pred,
        ColTarget::JsonKey("level"),
        ColOp::SetStr(b"CRITICAL"),
        &enc,
        &dec,
    )
    .unwrap();
    assert_eq!(untouched, frame);

    // Zero-hit numeric zone-map rejection (seq is affine >= 10_000, query < 50):
    let zone_miss = ColPred::json_num("seq", NumCmp::Lt(50));
    assert!(filter_column(&frame, &zone_miss, usize::MAX, &dec)
        .unwrap()
        .is_empty());
    let zone_untouched = update_column(
        &frame,
        &zone_miss,
        ColTarget::JsonKey("code"),
        ColOp::SetRaw(b"999"),
        &enc,
        &dec,
    )
    .unwrap();
    assert_eq!(zone_untouched, frame);

    // 2. Fixed-width update ("ERROR" -> "FATAL", both 5 bytes): stays a SINGLE VXG1 v4 frame!
    let err_pred = ColPred::json_str("level", StrCmp::Eq(b"ERROR"));
    let fixed_upd = update_column(
        &frame,
        &err_pred,
        ColTarget::JsonKey("level"),
        ColOp::SetStr(b"FATAL"),
        &enc,
        &dec,
    )
    .unwrap();
    assert!(Frame::parse(&fixed_upd, usize::MAX).is_ok());
    assert_eq!(
        decompress(&fixed_upd, &dec).unwrap(),
        naive_transform(
            &raw,
            Some(&err_pred),
            ColTarget::JsonKey("level"),
            &ColOp::SetStr(b"FATAL"),
        )
    );

    // 3. Variable-length update + chained map + query + range:
    let var_op = ColOp::SetStr(b"CRITICAL_INCIDENT\nLINE2");
    let var_upd = update_column(
        &frame,
        &err_pred,
        ColTarget::JsonKey("level"),
        var_op,
        &enc,
        &dec,
    )
    .unwrap();
    let expected_var = naive_transform(
        &raw,
        Some(&err_pred),
        ColTarget::JsonKey("level"),
        &ColOp::SetStr(b"CRITICAL_INCIDENT\nLINE2"),
    );
    assert_eq!(decompress(&var_upd, &dec).unwrap(), expected_var);
    assert_eq!(frame_row_count(&var_upd).unwrap(), 2400);

    let inc_pred = ColPred::json_str("level", StrCmp::Contains(b"INCIDENT\nLINE2"));
    let hits = filter_column(&var_upd, &inc_pred, usize::MAX, &dec).unwrap();
    assert_eq!(hits.len(), 240);

    let add_fn = |v: u64| v + 42;
    let mapped = map_column(
        &var_upd,
        ColTarget::JsonKey("code"),
        ColOp::MapU64(&add_fn),
        &enc,
        &dec,
    )
    .unwrap();
    let expected_mapped = naive_transform(
        &expected_var,
        None,
        ColTarget::JsonKey("code"),
        &ColOp::MapU64(&add_fn),
    );
    assert_eq!(decompress(&mapped, &dec).unwrap(), expected_mapped);
    assert_eq!(
        extract_row_range(&mapped, 500, 525, &dec).unwrap(),
        expected_mapped
            .split_inclusive(|&b| b == b'\n')
            .skip(500)
            .take(25)
            .flatten()
            .copied()
            .collect::<Vec<u8>>()
    );
}

#[test]
fn csv_split_inside_col0_and_multiblock_span_and_escapes() {
    // Craft a CSV where the 4096-byte block boundary cuts RIGHT INSIDE col 0 ("alice" -> "al" | "ice,admin,10\n")!
    let row = b"alice,admin,100\n"; // 16 bytes -> 256 rows = 4096 bytes; shift by 2 bytes so boundary cuts "al" | "ice,admin,100\n"!
    let mut csv = Vec::new();
    csv.extend_from_slice(b"bo,guest,99999\n"); // 15 bytes (-1 mod 16)
    csv.extend_from_slice(b"ed,guest,88888\n"); // 15 bytes (-2 mod 16 == 14 mod 16 -> row starts at 4094: "al" | "ice,admin,100\n")
    for _ in 0..600 {
        csv.extend_from_slice(row);
    }
    csv.extend_from_slice(b"last_no_newline,admin,777"); // EOF without trailing newline!
    assert_eq!(&csv[4094..4096], b"al");
    assert_eq!(&csv[4096..4110], b"ice,admin,100\n");

    let enc = Options {
        block_size: 4096,
        checksum: true,
        ..Options::default()
    };
    let dec = DecodeOptions::default();
    let frame = compress(&csv, &enc).unwrap();

    // Query col 0 == "ice" MUST return 0 matches (even though block 1 starts with "ice,admin,100\n"!)
    let fake_ice = ColPred::delimited_str(b',', 0, StrCmp::Eq(b"ice"));
    assert!(filter_column(&frame, &fake_ice, usize::MAX, &dec)
        .unwrap()
        .is_empty());

    // Query col 0 == "alice" MUST return all 600 rows (including the split row across 4096!)
    let real_alice = ColPred::delimited_str(b',', 0, StrCmp::Eq(b"alice"));
    assert_eq!(
        filter_column(&frame, &real_alice, usize::MAX, &dec)
            .unwrap()
            .len(),
        600
    );

    // Map col 0 -> SetStr("REDACTED,USER") (requires CSV quoting!) across split col 0 & EOF line:
    let op = ColOp::SetStr(b"REDACTED,USER");
    let mapped = map_column(
        &frame,
        ColTarget::Delimited {
            delim: b',',
            index: 0,
        },
        op,
        &enc,
        &dec,
    )
    .unwrap();
    let expected = naive_transform(
        &csv,
        None,
        ColTarget::Delimited {
            delim: b',',
            index: 0,
        },
        &ColOp::SetStr(b"REDACTED,USER"),
    );
    assert_eq!(decompress(&mapped, &dec).unwrap(), expected);

    // Also test a >9000-byte JSON line spanning THREE 4096-byte blocks + SetRaw fallback:
    let mut big_json = Vec::new();
    for i in 0..80 {
        big_json.extend_from_slice(format!("{{\"k\":\"v{i}\",\"n\":{i}}}\n").as_bytes());
    }
    let giant_val = "x".repeat(9500);
    big_json.extend_from_slice(format!("{{\"k\":\"{giant_val}\",\"n\":999}}\n").as_bytes());
    for i in 80..200 {
        big_json.extend_from_slice(format!("{{\"k\":\"v{i}\",\"n\":{i}}}\n").as_bytes());
    }
    let jf = compress(&big_json, &enc).unwrap();
    let p_999 = ColPred::json_num("n", NumCmp::Eq(999));
    let j_upd = update_column(
        &jf,
        &p_999,
        ColTarget::JsonKey("k"),
        ColOp::SetStr(b"tiny"),
        &enc,
        &dec,
    )
    .unwrap();
    assert_eq!(
        decompress(&j_upd, &dec).unwrap(),
        naive_transform(
            &big_json,
            Some(&p_999),
            ColTarget::JsonKey("k"),
            &ColOp::SetStr(b"tiny")
        )
    );

    // SetRaw injecting invalid slot token (e.g. raw whitespace/comma) cleanly falls back per block:
    let raw_tok_upd = update_column(
        &jf,
        &ColPred::json_num("n", NumCmp::Lt(10)),
        ColTarget::JsonKey("k"),
        ColOp::SetRaw(b" \"spaced\" "),
        &enc,
        &dec,
    )
    .unwrap();
    assert_eq!(
        decompress(&raw_tok_upd, &dec).unwrap(),
        naive_transform(
            &big_json,
            Some(&ColPred::json_num("n", NumCmp::Lt(10))),
            ColTarget::JsonKey("k"),
            &ColOp::SetRaw(b" \"spaced\" "),
        )
    );
}

fn gen_corpus(rng: &mut Rng) -> (Vec<u8>, bool, u8) {
    let mode = rng.usize(4); // 0: NDJSON single/multi-schema, 1: CSV, 2: TSV, 3: PSV
    let rows = 80 + rng.usize(450);
    let mut out = Vec::with_capacity(rows * 64);
    let words: &[&str] = &[
        "alpha",
        "beta",
        "gamma,delta",
        "quote\"in\"str",
        "slash\\path",
        " leading",
        "line\nbrk",
        "ERR_500",
        "OK_200",
        "u0001",
    ];
    if mode == 0 {
        let multi_schema = rng.bool();
        for r in 0..rows {
            if rng.usize(35) == 0 {
                out.extend_from_slice(b"# comment or unstructured outlier line\n");
                continue;
            }
            let w1 = words[rng.usize(words.len())];
            let w2 = words[rng.usize(words.len())];
            let num = (r as u64) * 7 + (rng.next() % 500);
            let mut s1 = Vec::new();
            vxg::ops::encode_json_str(w1.as_bytes(), &mut s1);
            let mut s2 = Vec::new();
            vxg::ops::encode_json_str(w2.as_bytes(), &mut s2);
            if multi_schema && r % 3 == 0 {
                out.extend_from_slice(b"{\"b\":");
                out.extend_from_slice(&s2);
                out.extend_from_slice(b",\"a\":");
                out.extend_from_slice(&s1);
                out.extend_from_slice(format!(",\"num\":{num},\"extra\":true}}\n").as_bytes());
            } else {
                out.extend_from_slice(b"{\"a\":");
                out.extend_from_slice(&s1);
                out.extend_from_slice(b",\"b\":");
                out.extend_from_slice(&s2);
                let tail = if r % 19 == 0 {
                    "null".to_owned()
                } else {
                    num.to_string()
                };
                out.extend_from_slice(format!(",\"num\":{tail}}}\n").as_bytes());
            }
        }
        if rng.bool() && out.last() == Some(&b'\n') {
            out.pop();
        }
        (out, true, 0)
    } else {
        let delim = match mode {
            1 => b',',
            2 => b'\t',
            _ => b'|',
        };
        for r in 0..rows {
            let w0 = words[rng.usize(words.len())];
            let w1 = words[rng.usize(words.len())];
            let num = (r as u64) * 5 + (rng.next() % 200);
            vxg::ops::encode_delim_str(w0.as_bytes(), delim, &mut out);
            out.push(delim);
            vxg::ops::encode_delim_str(w1.as_bytes(), delim, &mut out);
            out.push(delim);
            out.extend_from_slice(num.to_string().as_bytes());
            out.push(delim);
            out.extend_from_slice(format!("{:08x}", rng.next() as u32).as_bytes());
            out.push(b'\n');
        }
        if rng.bool() && out.last() == Some(&b'\n') {
            out.pop();
        }
        (out, false, delim)
    }
}

#[test]
fn randomized_differential_compressed_ops_stress() {
    let iters: usize = std::env::var("VXG_STRESS_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let needles: &[&[u8]] = &[
        b"alpha",
        b"beta",
        b"ERR",
        b"quote\"in",
        b"line\nbrk",
        b" leading",
        b"ZZZ_NONE",
    ];
    let json_keys = ["a", "b", "num", "missing"];

    for iter in 0..iters {
        let (raw, is_json, delim) = gen_corpus(&mut rng);
        let bs = [4096usize, 8192, 65536][rng.usize(3)];
        let enc = Options {
            block_size: bs,
            checksum: rng.bool(),
            ..Options::default()
        };
        let threads = [1usize, 2, 4][rng.usize(3)];
        let mut enc_mt = enc.clone();
        enc_mt.threads = threads;
        let dec = DecodeOptions {
            threads,
            verify_checksum: true,
            ..DecodeOptions::default()
        };
        let dec_1t = DecodeOptions {
            threads: 1,
            verify_checksum: true,
            ..DecodeOptions::default()
        };
        let frame = compress(&raw, &enc).unwrap();

        let pick_col = |r: &mut Rng| -> ColTarget<'static> {
            if is_json {
                ColTarget::JsonKey(json_keys[r.usize(json_keys.len())])
            } else {
                ColTarget::Delimited {
                    delim,
                    index: r.usize(5),
                }
            }
        };
        let w_col = pick_col(&mut rng);
        let t_col = pick_col(&mut rng);
        let needle = needles[rng.usize(needles.len())];
        let thresh = (rng.next() % 1200) as u64;
        let vpred = match rng.usize(11) {
            0 => ValuePred::Str(StrCmp::Eq(needle)),
            1 => ValuePred::Str(StrCmp::Ne(needle)),
            2 => ValuePred::Str(StrCmp::Prefix(&needle[..needle.len().min(3)])),
            3 => ValuePred::Str(StrCmp::Suffix(&needle[needle.len().saturating_sub(3)..])),
            4 => ValuePred::Str(StrCmp::Contains(&needle[..needle.len().min(4)])),
            5 => ValuePred::Num(NumCmp::Eq(thresh)),
            6 => ValuePred::Num(NumCmp::Ne(thresh)),
            7 => ValuePred::Num(NumCmp::Lt(thresh)),
            8 => ValuePred::Num(NumCmp::Le(thresh)),
            9 => ValuePred::Num(NumCmp::Gt(thresh)),
            _ => ValuePred::Num(NumCmp::Ge(thresh)),
        };
        let pred = ColPred {
            col: w_col,
            pred: vpred,
        };

        // 1. Differential check for filter_column (limit=1, limit=5, limit=MAX) & filter_frame:
        let exp_all = naive_filter(&raw, &pred, usize::MAX);
        for lim in [1usize, 5, usize::MAX] {
            let got = filter_column(&frame, &pred, lim, &dec).unwrap();
            let got_1t = filter_column(&frame, &pred, lim, &dec_1t).unwrap();
            assert_eq!(got, got_1t, "mt vs 1t filter_column mismatch iter={iter} lim={lim}");
            let exp_slice = &exp_all[..exp_all.len().min(lim)];
            assert_eq!(got.len(), exp_slice.len(), "iter={iter} lim={lim}");
            for (g, (ei, eb)) in got.iter().zip(exp_slice.iter()) {
                assert_eq!(g.global_row_index, *ei, "iter={iter}");
                assert_eq!(&g.bytes, eb, "iter={iter}");
            }
        }
        let ff = filter_frame(&frame, &pred, &enc_mt, &dec).unwrap();
        let exp_ff_bytes: Vec<u8> = exp_all.iter().flat_map(|(_, b)| b.iter().copied()).collect();
        assert_eq!(decompress(&ff, &dec).unwrap(), exp_ff_bytes, "iter={iter}");

        // 2. Differential check for update_column & map_column:
        let upper_fn = |b: &[u8]| {
            let mut v = b.to_ascii_uppercase();
            v.extend_from_slice(b"!");
            v
        };
        let num_fn = |v: u64| (v * 3 + 7) % 100_000;
        let set_val = needles[rng.usize(needles.len())];
        let op_choice = rng.usize(4);
        let op = match op_choice {
            0 => ColOp::SetStr(set_val),
            1 => ColOp::SetRaw(if is_json { b"\"raw_ok\"" } else { b"raw_ok" }),
            2 => ColOp::MapStr(&upper_fn),
            _ => ColOp::MapU64(&num_fn),
        };

        let upd_frame = update_column(&frame, &pred, t_col, op, &enc_mt, &dec).unwrap();
        let op_1t = match op_choice {
            0 => ColOp::SetStr(set_val),
            1 => ColOp::SetRaw(if is_json { b"\"raw_ok\"" } else { b"raw_ok" }),
            2 => ColOp::MapStr(&upper_fn),
            _ => ColOp::MapU64(&num_fn),
        };
        let upd_frame_1t = update_column(&frame, &pred, t_col, op_1t, &enc, &dec_1t).unwrap();
        assert_eq!(upd_frame, upd_frame_1t, "mt vs 1t update_column frame bytes mismatch iter={iter}");
        let op_ref = match op_choice {
            0 => ColOp::SetStr(set_val),
            1 => ColOp::SetRaw(if is_json { b"\"raw_ok\"" } else { b"raw_ok" }),
            2 => ColOp::MapStr(&upper_fn),
            _ => ColOp::MapU64(&num_fn),
        };
        let exp_upd = naive_transform(&raw, Some(&pred), t_col, &op_ref);
        assert_eq!(
            decompress(&upd_frame, &dec).unwrap(),
            exp_upd,
            "update mismatch at iter={iter}"
        );
        assert_eq!(
            frame_row_count(&upd_frame).unwrap(),
            exp_upd.split_inclusive(|&b| b == b'\n').count(),
            "row count mismatch after update at iter={iter}"
        );

        let map_op = match (op_choice + 2) % 4 {
            0 => ColOp::SetStr(b"mapped,val\"x"),
            1 => ColOp::SetRaw(if is_json { b"42" } else { b"42" }),
            2 => ColOp::MapStr(&upper_fn),
            _ => ColOp::MapU64(&num_fn),
        };
        let map_op_ref = match (op_choice + 2) % 4 {
            0 => ColOp::SetStr(b"mapped,val\"x"),
            1 => ColOp::SetRaw(b"42"),
            2 => ColOp::MapStr(&upper_fn),
            _ => ColOp::MapU64(&num_fn),
        };
        // Chain map_column directly ON TOP of upd_frame!
        let chained_frame = map_column(&upd_frame, w_col, map_op, &enc_mt, &dec).unwrap();
        let exp_chained = naive_transform(&exp_upd, None, w_col, &map_op_ref);
        assert_eq!(
            decompress(&chained_frame, &dec).unwrap(),
            exp_chained,
            "chained map mismatch at iter={iter}"
        );
    }
}
