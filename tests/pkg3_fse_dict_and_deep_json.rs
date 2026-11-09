//! Package 3 verification suite:
//! 1. Bit-exact determinism of parallel `FastCover` & macroscopic dictionary training (`1T..8T`).
//! 2. Pure-Rust 2-way interleaved `tANS / FSE` stream roundtrip, active `FLAG_FSE` verification,
//!    and 750-frame corruption fuzzing (bit-flips, truncations, 64-byte zeroings: 0 panics, 0 silent corruptions).
//! 3. Multi-level (`depth = 3..4`, `a.b.c` / `a.b.c.d`) nested JSON flattening compression &
//!    compressed-domain SQL + Arrow + adversarial dotted-key differential parity.

use vxg::{
    compress, decompress, execute_sql_table, export_arrow_batch_where, filter_where,
    parse_arrow_where, parse_col_list, verify_frame, ArrowBatch, DecodeOptions, Frame, Options,
    SqlQuery, SqlVal, BLOCK_COLUMN, BLOCK_DICT_LZ,
};

fn build_unstructured_log_corpus(target_bytes: usize) -> Vec<u8> {
    let services = ["edge-proxy", "auth-gw", "billing-worker", "search-idx", "kv-store"];
    let paths = [
        "/api/v2/checkout/confirm",
        "/api/v2/users/session/refresh",
        "/internal/metrics/prometheus",
        "/assets/bundle-9f8a7b6c.js",
    ];
    let mut out = Vec::with_capacity(target_bytes + 256);
    let mut i = 0usize;
    while out.len() < target_bytes {
        let svc = services[(i * 7 + (i >> 3)) % services.len()];
        let path = paths[(i * 3 + (i >> 2)) % paths.len()];
        // Irregular prose/stack-trace style lines that defeat fixed-slot tabular templates so `BLOCK_DICT_LZ` wins.
        if i % 5 == 0 {
            out.extend_from_slice(
                format!(
                    "[2026-11-09T12:{:02}:{:02}.{:03}Z] WARN {svc} req_id={:08x} retry={} upstream timeout on {path} after {}ms (peer=10.42.{}.{}:8443)\n",
                    (i / 60) % 60,
                    i % 60,
                    (i * 37) % 1000,
                    (i as u32).wrapping_mul(0x9e37_79b9),
                    1 + (i % 3),
                    120 + (i * 19) % 850,
                    (i >> 4) & 255,
                    i & 255,
                )
                .as_bytes(),
            );
        } else {
            out.extend_from_slice(
                format!(
                    "[2026-11-09T12:{:02}:{:02}.{:03}Z] INFO {svc} handled {path} -> 200 OK ({}B)\n",
                    (i / 60) % 60,
                    i % 60,
                    (i * 13) % 1000,
                    256 + (i * 53) % 16384,
                )
                .as_bytes(),
            );
        }
        i += 1;
    }
    out
}

fn build_macro_periodic_corpus(total_bytes: usize) -> Vec<u8> {
    let mut seed = Vec::with_capacity(96 * 1024);
    for idx in 0..1100usize {
        seed.extend_from_slice(
            format!(
                "pub fn kernel_dispatch_{idx:04}(src: &[u8], acc: u64) -> u64 {{ acc.wrapping_mul(0x9e3779b185ebca87).wrapping_add({} ^ src.len() as u64) }}\n",
                (idx as u64).wrapping_mul(0x517cc1b727220a95)
            )
            .as_bytes(),
        );
    }
    let mut out = Vec::with_capacity(total_bytes);
    while out.len() < total_bytes {
        let take = seed.len().min(total_bytes - out.len());
        out.extend_from_slice(&seed[..take]);
    }
    out
}

#[test]
fn parallel_fastcover_and_macro_training_bit_exact_across_threads() {
    let log_corpus = build_unstructured_log_corpus(1_600_000); // ~25 x 64 KiB blocks (`w_threads > 1`)
    for level in [1u8, 3, 6, 9] {
        let ref_frame = compress(
            &log_corpus,
            &Options {
                threads: 1,
                level,
                block_size: 65_536,
                ..Options::default()
            },
        )
        .unwrap();
        let parsed = Frame::parse(&ref_frame, log_corpus.len() + 64).unwrap();
        assert!(!parsed.dict().is_empty(), "expected trained dictionary at level={level}");

        for threads in [2usize, 4, 8] {
            let par_frame = compress(
                &log_corpus,
                &Options {
                    threads,
                    level,
                    block_size: 65_536,
                    ..Options::default()
                },
            )
            .unwrap();
            assert_eq!(
                par_frame, ref_frame,
                "parallel FastCover frame diverged from 1T at level={level}, threads={threads}"
            );
        }
        let dec = decompress(
            &ref_frame,
            &DecodeOptions {
                threads: 4,
                max_output: log_corpus.len() + 64,
                ..DecodeOptions::default()
            },
        )
        .unwrap();
        assert_eq!(dec, log_corpus);
    }

    // Macroscopic (`>= 4 MiB`, `>= 32` blocks) 1T vs 4T vs 8T bit-exact determinism
    let macro_corpus = build_macro_periodic_corpus(4_500_000);
    let macro_1t = compress(
        &macro_corpus,
        &Options {
            threads: 1,
            level: 3,
            block_size: 65_536,
            ..Options::default()
        },
    )
    .unwrap();
    let macro_8t = compress(
        &macro_corpus,
        &Options {
            threads: 8,
            level: 3,
            block_size: 65_536,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(macro_1t, macro_8t, "macroscopic dictionary frame diverged between 1T and 8T");
    assert_eq!(
        decompress(
            &macro_8t,
            &DecodeOptions {
                threads: 4,
                max_output: macro_corpus.len() + 64,
                ..DecodeOptions::default()
            }
        )
        .unwrap(),
        macro_corpus
    );
}

#[test]
fn fse_dict_lz_active_flags_roundtrip_and_750_corrupt_frame_fuzz() {
    let corpus = build_unstructured_log_corpus(520_000);
    let mut saw_fse_block = false;

    for level in [3u8, 6, 9] {
        for bsz in [16_384usize, 65_536] {
            // Slice both even and odd lengths to exercise 2-way FSE parity tails.
            for trim in [0usize, 1, 7] {
                let src = &corpus[..corpus.len() - trim];
                let frame = compress(
                    src,
                    &Options {
                        threads: 4,
                        level,
                        block_size: bsz,
                        checksum: true,
                        ..Options::default()
                    },
                )
                .unwrap();
                let parsed = Frame::parse(&frame, src.len() + 64).unwrap();
                for blk in parsed.blocks() {
                    if blk.kind == BLOCK_DICT_LZ {
                        let flags = frame[blk.payload_offset];
                        if (flags & 0x10) != 0 {
                            saw_fse_block = true;
                        }
                    }
                }
                parsed.validate_commands().unwrap();
                let got = decompress(
                    &frame,
                    &DecodeOptions {
                        threads: 2,
                        max_output: src.len() + 64,
                        ..DecodeOptions::default()
                    },
                )
                .unwrap();
                assert_eq!(got, src);
            }
        }
    }
    assert!(saw_fse_block, "expected BLOCK_DICT_LZ to emit FLAG_FSE (0x10) on skewed sequences");

    // 750-frame deterministic corruption fuzz on an FSE-enabled `BLOCK_DICT_LZ` frame:
    // 250 bit-flips, 250 truncations, 250 64-byte zeroings -> 0 panics, 0 silent wrong decodes!
    let fuzz_src = &corpus[..200_000];
    let base_frame = compress(
        fuzz_src,
        &Options {
            threads: 2,
            level: 6,
            block_size: 32_768,
            checksum: true,
            ..Options::default()
        },
    )
    .unwrap();
    let parsed_base = Frame::parse(&base_frame, fuzz_src.len() + 64).unwrap();
    assert!(
        parsed_base
            .blocks()
            .any(|b| b.kind == BLOCK_DICT_LZ && (base_frame[b.payload_offset] & 0x10) != 0),
        "fuzz base frame must contain at least one FSE BLOCK_DICT_LZ block"
    );

    let dec_opts = DecodeOptions {
        threads: 1,
        max_output: fuzz_src.len() + 4096,
        ..DecodeOptions::default()
    };
    let mut rng = 0x517c_c1b7_2722_0a95u64;
    let mut next_u64 = || -> u64 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for case_idx in 0..750usize {
        let mut mutated = base_frame.clone();
        match case_idx % 3 {
            0 => {
                let idx = (next_u64() as usize) % mutated.len();
                let bit = 1u8 << (next_u64() & 7);
                mutated[idx] ^= bit;
            }
            1 => {
                let new_len = (next_u64() as usize) % mutated.len();
                mutated.truncate(new_len);
            }
            _ => {
                if mutated.len() > 8 {
                    let start = (next_u64() as usize) % mutated.len();
                    let end = (start + 64).min(mutated.len());
                    mutated[start..end].fill(0);
                    if mutated == base_frame {
                        mutated[start] ^= 0xff;
                    }
                }
            }
        }

        if let Ok(out) = decompress(&mutated, &dec_opts) {
            assert_eq!(
                out.as_slice(),
                fuzz_src,
                "silent wrong decode on FSE corruption case {case_idx}!"
            );
        }
        if let Ok(vlen) = verify_frame(&mutated, &dec_opts) {
            assert_eq!(vlen, fuzz_src.len() as u64);
        }
    }
}

#[test]
fn deep_nested_json_depth_3_and_4_flattening_and_query_parity() {
    let cities = ["istanbul", "ankara", "izmir", "berlin"];
    let tiers = ["free", "pro", "enterprise"];
    let mut ndjson = Vec::with_capacity(240_000);
    for i in 0..1600usize {
        let city = cities[i % cities.len()];
        let tier = tiers[(i / 4) % tiers.len()];
        let lat = 380_000 + ((i * 17) % 25_000) as i64;
        let lon = 270_000 + ((i * 31) % 60_000) as i64;
        let status = if i % 11 == 0 { 503 } else { 200 };
        let ms = format!("{}.{:02}", 4 + (i % 40), (i * 9) % 100);
        ndjson.extend_from_slice(
            format!(
                "{{\"seq\":{i},\"tenant\":{{\"tier\":\"{tier}\",\"geo\":{{\"city\":\"{city}\",\"coords\":{{\"lat_e4\":{lat},\"lon_e4\":{lon}}}}},\"http\":{{\"status\":{status},\"latency_ms\":{ms}}}}}}}\n"
            )
            .as_bytes(),
        );
    }

    for record_aligned in [false, true] {
        let frame = compress(
            &ndjson,
            &Options {
                threads: 4,
                block_size: 32_768,
                record_aligned,
                ..Options::default()
            },
        )
        .unwrap();
        let parsed = Frame::parse(&frame, ndjson.len() + 64).unwrap();
        assert!(
            parsed.blocks().all(|b| b.kind == BLOCK_COLUMN),
            "deeply nested JSON should flatten into BLOCK_COLUMN (aligned={record_aligned})"
        );
        // Deep leaf columns (`city`, `tier`, `lat_e4`, `lon_e4`, `status`, `latency_ms`) compress tightly (< 10% of raw).
        assert!(
            frame.len() * 10 < ndjson.len(),
            "expected < 0.10 ratio on flattened depth-4 JSON, got {} / {}",
            frame.len(),
            ndjson.len()
        );

        for threads in [1usize, 4] {
            let dec = DecodeOptions {
                threads,
                max_output: ndjson.len() + 64,
                ..DecodeOptions::default()
            };
            assert_eq!(decompress(&frame, &dec).unwrap(), ndjson);

            // 1. SQL GROUP BY + AVG/SUM/MAX on depth-3 & depth-4 dotted paths
            let q_grp = SqlQuery::parse(
                "SELECT tenant.geo.city, COUNT(*), SUM(tenant.geo.coords.lat_e4), MAX(tenant.http.latency_ms) \
                 FROM frame WHERE tenant.http.status = 200 AND tenant.tier != 'free' \
                 GROUP BY tenant.geo.city ORDER BY 1 ASC",
            )
            .unwrap();
            let tbl = execute_sql_table(&frame, &q_grp, &dec).unwrap();
            assert_eq!(tbl.rows.len(), 4);

            // Verify against exact row-by-row reference calculation!
            let mut exp_cnt = [0i64; 4];
            let mut exp_sum = [0i64; 4];
            for i in 0..1600usize {
                let tier_i = (i / 4) % tiers.len();
                let status = if i % 11 == 0 { 503 } else { 200 };
                if status == 200 && tier_i != 0 {
                    let c_idx = i % cities.len();
                    let lat = 380_000 + ((i * 17) % 25_000) as i64;
                    exp_cnt[c_idx] += 1;
                    exp_sum[c_idx] += lat;
                }
            }
            let mut order = [0usize, 1, 2, 3];
            order.sort_by_key(|&ci| cities[ci]);
            for (r_idx, &ci) in order.iter().enumerate() {
                assert_eq!(tbl.rows[r_idx][0], SqlVal::Str(cities[ci].to_owned()));
                assert_eq!(tbl.rows[r_idx][1], SqlVal::Int(exp_cnt[ci] as u64));
                assert_eq!(tbl.rows[r_idx][2], SqlVal::Int(exp_sum[ci] as u64));
            }

            // 2. Compressed-domain Arrow scan (`WHERE` + projection) on depth-3 & depth-4 paths
            let cols = parse_col_list(
                "seq,tenant.tier,tenant.geo.city,tenant.geo.coords.lat_e4,tenant.http.latency_ms",
            )
            .unwrap();
            let aw = parse_arrow_where(
                "tenant.http.status = 503 AND tenant.geo.coords.lon_e4 >= 295000",
            )
            .unwrap();
            let got_arrow = aw
                .with_pred_expr(|we| export_arrow_batch_where(&frame, Some(&cols), Some(we), None, &dec))
                .unwrap();
            let q_ref = SqlQuery::parse(
                "SELECT seq, tenant.tier, tenant.geo.city, tenant.geo.coords.lat_e4, tenant.http.latency_ms \
                 FROM frame WHERE tenant.http.status = 503 AND tenant.geo.coords.lon_e4 >= 295000",
            )
            .unwrap();
            let ref_arrow = ArrowBatch::from_sql_table(&execute_sql_table(&frame, &q_ref, &dec).unwrap());
            assert_eq!(got_arrow, ref_arrow);
            assert!(got_arrow.num_rows > 0);
        }
    }

    // 3. Adversarial literal dotted inner key (`{"a":{"b": {"c": 20}, "b.c": 10}}`):
    //    `find_json_key_span` resolves `a.b.c` to `"b.c": 10`, and compressed-domain `ops_pred` must agree!
    let mut adv = Vec::new();
    for i in 0..120usize {
        let v_nested = 200 + i;
        let v_lit = 10 + (i % 2);
        adv.extend_from_slice(
            format!("{{\"id\":{i},\"a\":{{\"b\":{{\"c\":{v_nested}}},\"b.c\":{v_lit}}}}}\n").as_bytes(),
        );
    }
    let adv_frame = compress(&adv, &Options::default()).unwrap();
    let dec = DecodeOptions::default();
    let pw = parse_arrow_where("a.b.c = 10").unwrap();
    let hits = pw.with_pred_expr(|we| filter_where(&adv_frame, we, usize::MAX, &dec)).unwrap();
    assert_eq!(hits.len(), 60);
}
