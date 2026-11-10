//! Package 4 integration, differential random-access, and corruption tests for `Format v5`
//! `FLAG_ROW_INDEX = 0x0008` ($O(1)$ `frame_row_count`, $O(\log B)$ zero-prefix-decode `extract_row`
//! / `extract_row_range`, `FramePeeler` zero-copy streaming, and `ArrowWindowStream` over `DICT_LZ`).

use std::io::Cursor;
use vxg::{
    arrow::{export_arrow_batches_where, ArrowData},
    block_row_count, compress, compress_into, decompress, execute_sql, execute_sql_stream,
    extract_row, extract_row_range,
    ffi::{vxg_arrow_query_stream, ArrowArray, ArrowArrayStream, ArrowSchema},
    frame_row_count, frame_row_count_stream, max_compressed_size, parse_arrow_where, verify_frame,
    ColTarget, DecodeOptions, Frame, Options, SqlFormat, SqlQuery, BLOCK_COLUMN, BLOCK_DICT_LZ,
};

fn build_log_corpus(lines: usize, trailing_nl: bool) -> Vec<u8> {
    let endpoints = [
        "/v2/orders/98412/capture",
        "/v2/inventory/sku-4410/reserve",
        "/internal/metrics/prometheus",
        "/oauth2/token/introspect",
    ];
    let mut out = Vec::with_capacity(lines * 110);
    let mut s = 0x517c_c1b7_2722_0a95_u64;
    for i in 0..lines {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let ep = endpoints[((s >> 32) as usize) & 3];
        let code = [200u16, 200, 201, 204, 404, 503][((s >> 40) as usize) % 6];
        let dur = 10 + ((s >> 48) % 9500);
        let sep = if !trailing_nl && i + 1 == lines { "" } else { "\n" };
        let var_tail = match i % 4 {
            0 => " note=\"upstream TLS handshake retry succeeded on attempt 2\"",
            1 => " cache=HIT region=eu-central-1",
            2 => " warn=\"rate limiter token bucket below 15 pct watermark\"",
            _ => "",
        };
        out.extend_from_slice(
            format!(
                "2026-11-09T16:{:02}:{:02}.{:03}Z edge-{:02} POST {ep} status={code} dur_us={dur} trace={:016x}{var_tail}{sep}",
                (i / 60) % 60,
                i % 60,
                i % 1000,
                (s >> 24) & 15,
                s ^ (i as u64),
            )
            .as_bytes(),
        );
    }
    out
}

fn build_csv_corpus(rows: usize) -> Vec<u8> {
    let mut out = b"id,region,payment_type,amount_cents\n".to_vec();
    let regions = ["eu-central", "us-east", "ap-south", "us-west"];
    for i in 0..rows {
        let reg = regions[i & 3];
        let pt = (i % 4) + 1;
        let amt = 500 + (i * 37) % 45_000;
        out.extend_from_slice(format!("{i},{reg},{pt},{amt}\n").as_bytes());
    }
    out
}

fn split_lines_oracle(raw: &[u8]) -> Vec<&[u8]> {
    if raw.is_empty() {
        Vec::new()
    } else {
        raw.split_inclusive(|&b| b == b'\n').collect()
    }
}

#[test]
fn v5_row_index_o1_count_and_log_b_random_seek_across_all_block_kinds() {
    // 1. DICT_LZ corpus (both fixed-block with split boundary lines + unterminated EOF, and record-aligned)
    for (trailing_nl, record_aligned) in [(true, false), (false, false), (true, true), (false, true)] {
        let raw = build_log_corpus(1800, trailing_nl);
        let oracle = split_lines_oracle(&raw);
        let opt_1t = Options {
            block_size: 8192,
            threads: 1,
            checksum: true,
            record_aligned,
            ..Options::default()
        };
        let f_1t = compress(&raw, &opt_1t).unwrap();
        for threads in [2usize, 4, 8] {
            let opt_mt = Options {
                threads,
                ..opt_1t.clone()
            };
            let f_mt = compress(&raw, &opt_mt).unwrap();
            assert_eq!(
                f_1t, f_mt,
                "row_index frame must be bit-identical across 1T and {threads}T (aligned={record_aligned})"
            );
            let mut into_buf = vec![0u8; max_compressed_size(raw.len(), opt_mt.block_size).unwrap()];
            let n_written = compress_into(&raw, &mut into_buf, &opt_mt).unwrap();
            assert_eq!(&into_buf[..n_written], f_1t.as_slice());
        }

        let parsed = Frame::parse(&f_1t, raw.len() + 64).unwrap();
        assert_eq!(parsed.version(), 5);
        assert!(parsed.has_row_index(), "v5 DICT_LZ / record_aligned frame must emit FLAG_ROW_INDEX");
        assert!(parsed.blocks().any(|b| b.kind == BLOCK_DICT_LZ));
        assert_eq!(parsed.indexed_row_count(), Some(oracle.len()));
        assert_eq!(frame_row_count(&f_1t).unwrap(), oracle.len());

        let dec = DecodeOptions {
            threads: 4,
            max_output: raw.len() + 1024,
            verify_checksum: true,
            ..DecodeOptions::default()
        };
        assert_eq!(verify_frame(&f_1t, &dec).unwrap(), raw.len() as u64);

        for b_idx in 0..parsed.block_count() {
            let blk = parsed.decode_block(b_idx, &dec).unwrap();
            let exp_blk_rows = split_lines_oracle(&blk).len();
            assert_eq!(block_row_count(&f_1t, b_idx).unwrap(), exp_blk_rows);
        }

        // Prove zero-prefix-decode single-block seek: corrupt EVERY block payload except the
        // second-to-last block, and verify `frame_row_count` + `extract_row` still succeed!
        if parsed.block_count() >= 5 {
            let target_blk = parsed.block_count() - 2;
            let mut poisoned = f_1t.clone();
            for (idx, b) in parsed.blocks().enumerate() {
                if idx != target_blk {
                    for p in b.payload_offset..b.payload_offset + b.payload_size.min(24) {
                        poisoned[p] ^= 0xff;
                    }
                }
            }
            assert_eq!(
                frame_row_count(&poisoned).unwrap(),
                oracle.len(),
                "frame_row_count must use O(B) header row_index without touching block payloads"
            );
            // Pick an interior row strictly inside `target_blk` (row index 2 of `target_blk`).
            let mut cur = 0usize;
            let mut open = false;
            let mut target_global = 0usize;
            for idx in 0..=target_blk {
                let (r, ends_nl) = parsed.block_row_meta(idx).unwrap();
                let base = if open { cur - 1 } else { cur };
                if idx == target_blk {
                    assert!(r >= 4);
                    target_global = base + 2;
                }
                cur = base + r;
                open = !ends_nl;
            }
            let got_single = extract_row(&poisoned, target_global, &dec).unwrap();
            assert_eq!(
                got_single.as_slice(),
                oracle[target_global],
                "extract_row must decode ONLY target_blk and never touch poisoned preceding/trailing blocks"
            );
        }

        // Check boundary-spanning & random row ranges against oracle:
        for &r_idx in &[0usize, 1, 73, 419, 999, oracle.len() - 2, oracle.len() - 1] {
            assert_eq!(extract_row(&f_1t, r_idx, &dec).unwrap().as_slice(), oracle[r_idx]);
        }
        for &(s, e) in &[(0usize, 5usize), (70, 95), (510, 540), (oracle.len() - 8, oracle.len())] {
            let expected: Vec<u8> = oracle[s..e].concat();
            assert_eq!(extract_row_range(&f_1t, s, e, &dec).unwrap(), expected);
        }
        assert!(extract_row(&f_1t, oracle.len(), &dec).is_err());
    }

    // 2. Explicit `Options { row_index: true }` on pure BLOCK_COLUMN CSV (no dict, fixed 8 KiB blocks)
    //    plus CSV header auto-stripping (`execute_sql`) preserving tail `FLAG_ROW_INDEX`:
    let csv = build_csv_corpus(2400);
    let csv_oracle = split_lines_oracle(&csv);
    let csv_frame = compress(
        &csv,
        &Options {
            block_size: 8192,
            threads: 4,
            row_index: true,
            ..Options::default()
        },
    )
    .unwrap();
    let csv_parsed = Frame::parse(&csv_frame, csv.len() + 64).unwrap();
    assert_eq!(csv_parsed.version(), 5);
    assert!(csv_parsed.has_row_index());
    assert!(csv_parsed.dict().is_empty());
    assert!(csv_parsed.blocks().all(|b| b.kind == BLOCK_COLUMN));
    assert_eq!(frame_row_count(&csv_frame).unwrap(), csv_oracle.len());

    let csv_dec = DecodeOptions {
        threads: 4,
        max_output: csv.len() + 1024,
        ..DecodeOptions::default()
    };
    let q = SqlQuery::parse(
        "SELECT region, COUNT(*), SUM(amount_cents) FROM t WHERE payment_type = 3 GROUP BY region ORDER BY region ASC",
    )
    .unwrap();
    let csv_v4 = compress(
        &csv,
        &Options {
            block_size: 8192,
            threads: 1,
            row_index: false,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(Frame::parse(&csv_v4, csv.len() + 64).unwrap().version(), 4);
    assert_eq!(
        execute_sql(&csv_frame, &q, SqlFormat::Json, false, &csv_dec).unwrap(),
        execute_sql(&csv_v4, &q, SqlFormat::Json, false, &csv_dec).unwrap()
    );

    // 3. Concatenated multi-frame stream where Frame 0 ends mid-record (`!ends_nl`) into Frame 1:
    let full = build_log_corpus(900, true);
    let cut = full.len() / 2 + 17; // intentionally mid-line!
    assert_ne!(full[cut - 1], b'\n');
    let part0 = compress(
        &full[..cut],
        &Options {
            block_size: 8192,
            threads: 2,
            row_index: true,
            ..Options::default()
        },
    )
    .unwrap();
    let part1 = compress(
        &full[cut..],
        &Options {
            block_size: 8192,
            threads: 2,
            row_index: true,
            ..Options::default()
        },
    )
    .unwrap();
    let mut concat = part0;
    concat.extend_from_slice(&part1);
    let full_oracle = split_lines_oracle(&full);
    assert_eq!(frame_row_count(&concat).unwrap(), full_oracle.len());
    let full_dec = DecodeOptions {
        threads: 4,
        max_output: full.len() + 1024,
        ..DecodeOptions::default()
    };
    assert_eq!(
        frame_row_count_stream(Cursor::new(&concat), &full_dec).unwrap() as usize,
        full_oracle.len()
    );
    let seam_row = split_lines_oracle(&full[..cut]).len() - 1;
    for r in seam_row.saturating_sub(2)..=(seam_row + 2).min(full_oracle.len() - 1) {
        assert_eq!(extract_row(&concat, r, &full_dec).unwrap().as_slice(), full_oracle[r]);
    }
    assert_eq!(
        extract_row_range(&concat, seam_row - 1, seam_row + 2, &full_dec).unwrap(),
        full_oracle[seam_row - 1..seam_row + 2].concat()
    );
}

#[test]
fn v5_row_index_corruption_and_forged_entry_rejection() {
    // 1. Empty frame with `FLAG_ROW_INDEX` set must be rejected:
    let empty_v4 = compress(b"", &Options::default()).unwrap();
    let mut bad_empty = empty_v4.clone();
    bad_empty[4..6].copy_from_slice(&5u16.to_le_bytes());
    bad_empty[6..8].copy_from_slice(&0x0009u16.to_le_bytes()); // CRC | FLAG_ROW_INDEX on count=0
    assert!(Frame::parse(&bad_empty, 1024).is_err());

    // 2. Multi-block `FLAG_ROW_INDEX` frame (`DICT_LZ` and `COLUMN` variants):
    for raw in [build_log_corpus(900, true), build_csv_corpus(1500)] {
        let frame = compress(
            &raw,
            &Options {
                block_size: 8192,
                threads: 2,
                row_index: true,
                record_aligned: true,
                ..Options::default()
            },
        )
        .unwrap();
        let parsed = Frame::parse(&frame, raw.len() + 64).unwrap();
        assert!(parsed.has_row_index());
        let count = parsed.block_count();
        let ri_offset = parsed.blocks().next().unwrap().payload_offset - count * 4;
        let dec = DecodeOptions {
            threads: 2,
            max_output: raw.len() + 1024,
            verify_checksum: true,
            ..DecodeOptions::default()
        };

        // 2a. Setting row_count = 0 or row_count = raw_size + 1 in header must fail `Frame::parse`:
        let mut zero_rc = frame.clone();
        zero_rc[ri_offset..ri_offset + 4].copy_from_slice(&0x8000_0000u32.to_le_bytes());
        assert!(Frame::parse(&zero_rc, raw.len() + 64).is_err());

        let rn0 = parsed.blocks().next().unwrap().raw_size as u32;
        let mut over_rc = frame.clone();
        over_rc[ri_offset..ri_offset + 4].copy_from_slice(&((rn0 + 1) | 0x8000_0000).to_le_bytes());
        assert!(Frame::parse(&over_rc, raw.len() + 64).is_err());

        // 2b. Forging a plausible-looking `row_meta` (e.g. +1 row or flipped `ends_nl` bit) must be
        //     rejected by BOTH `verify_frame` and `extract_row` (zero silent wrong output):
        for blk_i in 0..count {
            let pos = ri_offset + blk_i * 4;
            let orig_word = u32::from_le_bytes(frame[pos..pos + 4].try_into().unwrap());
            let mut flip_nl = frame.clone();
            flip_nl[pos..pos + 4].copy_from_slice(&(orig_word ^ 0x8000_0000).to_le_bytes());
            assert!(verify_frame(&flip_nl, &dec).is_err(), "verify_frame must catch flipped ends_nl");

            let orig_rows = orig_word & 0x7fff_ffff;
            let forged_rows = if orig_rows > 1 { orig_rows - 1 } else { orig_rows + 1 };
            let forged_word = (orig_word & 0x8000_0000) | forged_rows;
            let mut forged = frame.clone();
            forged[pos..pos + 4].copy_from_slice(&forged_word.to_le_bytes());
            assert!(verify_frame(&forged, &dec).is_err(), "verify_frame must catch forged row count");
            if blk_i == 0 {
                assert!(
                    extract_row(&forged, 0, &dec).is_err(),
                    "extract_row must reject forged block 0 row_index entry"
                );
            }
        }

        // 2c. 250 deterministic corruptions across header + row_index + payloads:
        let mut rng = 0x9e37_79b9_7f4a_7c15_u64;
        for case_id in 0..250usize {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut bad = frame.clone();
            match case_id % 3 {
                0 => {
                    let idx = (rng as usize) % bad.len();
                    bad[idx] ^= 1u8 << ((rng >> 32) & 7);
                }
                1 => {
                    let cut = (rng as usize) % bad.len();
                    bad.truncate(cut);
                }
                _ => {
                    let start = (rng as usize) % bad.len();
                    let end = (start + 32).min(bad.len());
                    bad[start..end].fill(0);
                }
            }
            if let Ok(out) = decompress(&bad, &dec) {
                assert_eq!(out, raw, "silent corruption on case {case_id}");
            }
            let _ = verify_frame(&bad, &dec);
            let _ = extract_row(&bad, 0, &dec);
        }
    }
}

#[test]
fn v5_dict_lz_arrow_window_stream_and_parallel_sql_parity() {
    // > 140 blocks (`4096` B blocks) of irregular nested JSON that routes to `BLOCK_DICT_LZ`
    // (`parsed.dict()` non-empty, `parsed.has_row_index() == true`), crossing multiple
    // `STREAM_WINDOW_BLOCKS = 128` windows in `ArrowWindowStream`:
    let mut raw = Vec::with_capacity(700_000);
    let mut s = 0x243f_6a88_85a3_08d3_u64;
    let mut expected_rows = 0usize;
    let mut expected_sum = 0i64;
    for i in 0..4800usize {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let tier = ["free", "pro", "enterprise", "internal"][i & 3];
        let score = ((i * 19 + (s as usize & 63)) % 1000) as i64;
        // Dynamic schema keys every 3rd line force `BLOCK_DICT_LZ` with non-empty shared dict!
        let dyn_field = match i % 3 {
            0 => format!("\"extra_{}\":\"blob_{:012x}\"", i % 97, s >> 16),
            1 => format!("\"tags\":[\"t{}\",\"u{}\"],\"nested_{}\":{}", i & 7, i % 11, i % 53, i % 9),
            _ => format!("\"note\":\"freeform message with spaces {}\"", s & 0xffff),
        };
        let line = format!(
            "{{\"id\":{i},\"tier\":\"{tier}\",\"score\":{score},{dyn_field},\"msg\":\"event-payload-alpha-beta-gamma-{}\"}}\n",
            i % 23
        );
        if tier == "enterprise" && score >= 400 {
            expected_rows += 1;
            expected_sum += score;
        }
        raw.extend_from_slice(line.as_bytes());
    }

    let frame = compress(
        &raw,
        &Options {
            block_size: 4096,
            threads: 4,
            record_aligned: true,
            target_block_rows: 20,
            ..Options::default()
        },
    )
    .unwrap();
    let parsed = Frame::parse(&frame, raw.len() + 64).unwrap();
    assert!(parsed.block_count() > 130, "must span >128 blocks for multi-window ArrowWindowStream");
    assert!(!parsed.dict().is_empty(), "must carry non-empty frame dictionary");
    assert!(parsed.blocks().any(|b| b.kind == BLOCK_DICT_LZ));
    assert!(parsed.has_row_index());

    let aw = parse_arrow_where("tier = 'enterprise' AND score >= 400").unwrap();
    let cols = [ColTarget::JsonKey("id"), ColTarget::JsonKey("score")];
    for threads in [1usize, 4] {
        let dec = DecodeOptions {
            threads,
            max_output: raw.len() + 1024,
            ..DecodeOptions::default()
        };
        let batches = aw
            .with_pred_expr(|w| export_arrow_batches_where(&frame, Some(&cols), Some(w), 64, None, &dec))
            .unwrap();
        let got_rows: usize = batches.iter().map(|b| b.num_rows).sum();
        let got_sum: i64 = batches
            .iter()
            .map(|b| match &b.columns[1].data {
                ArrowData::Int64(v) => v.iter().sum::<i64>(),
                other => panic!("unexpected ArrowData: {other:?}"),
            })
            .sum();
        assert_eq!(got_rows, expected_rows);
        assert_eq!(got_sum, expected_sum);

        // Also exercise C-ABI `vxg_arrow_query_stream` (`ArrowWindowStream` over `128`-block windows):
        let mut stream: ArrowArrayStream = unsafe { std::mem::zeroed() };
        let c_cols = std::ffi::CString::new("id,score").unwrap();
        let c_where = std::ffi::CString::new("tier = 'enterprise' AND score >= 400").unwrap();
        let rc = unsafe {
            vxg_arrow_query_stream(
                frame.as_ptr(),
                frame.len(),
                c_cols.as_ptr(),
                c_where.as_ptr(),
                threads,
                &mut stream,
            )
        };
        assert_eq!(rc, 0);
        let mut schema: ArrowSchema = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { (stream.get_schema.unwrap())(&mut stream, &mut schema) }, 0);
        if let Some(rel) = schema.release {
            unsafe { rel(&mut schema) };
        }
        let mut stream_rows = 0usize;
        loop {
            let mut arr: ArrowArray = unsafe { std::mem::zeroed() };
            assert_eq!(unsafe { (stream.get_next.unwrap())(&mut stream, &mut arr) }, 0);
            if arr.release.is_none() {
                break;
            }
            assert!(arr.length > 0);
            stream_rows += arr.length as usize;
            if let Some(rel) = arr.release {
                unsafe { rel(&mut arr) };
            }
        }
        if let Some(rel) = stream.release {
            unsafe { rel(&mut stream) };
        }
        assert_eq!(stream_rows, expected_rows);

        // Verify `execute_sql` and `execute_sql_stream` (`FramePeeler` zero-copy fast path):
        let q = SqlQuery::parse(
            "SELECT COUNT(*), SUM(score) FROM events WHERE tier = 'enterprise' AND score >= 400",
        )
        .unwrap();
        let sql_mem = execute_sql(&frame, &q, SqlFormat::Csv, true, &dec).unwrap();
        let mut sql_stm = Vec::new();
        execute_sql_stream(Cursor::new(&frame), &mut sql_stm, &q, SqlFormat::Csv, true, &dec)
            .unwrap();
        assert_eq!(sql_mem, sql_stm);
        assert_eq!(
            String::from_utf8(sql_mem).unwrap().trim(),
            format!("{expected_rows},{expected_sum}")
        );
    }
}

#[test]
fn train_macro_tls_scratch_determinism_across_sequential_calls() {
    // Regression test for TLS `TRAIN_SCRATCH` reuse across `try_macroscopic_dict` (`>= 4 MiB`,
    // `>= 32` blocks) when `compress_with_stats` is immediately followed by `compress` on the same
    // worker threads.
    let mut prose = Vec::with_capacity(4_500_000);
    let mut s = 0x9e37_79b9_7f4a_7c15_u64;
    let words = [
        "the", "tragedy", "of", "hamlet", "prince", "denmark", "horatio", "bernardo",
        "marcellus", "elsinore", "platform", "before", "castle", "francisco", "post", "enter",
    ];
    for i in 0..90_000usize {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let w0 = words[(s as usize) & 15];
        let w1 = words[((s >> 8) as usize) & 15];
        let w2 = words[((s >> 16) as usize) & 15];
        prose.extend_from_slice(
            format!("  Act {} Scene {}: {w0} {w1} {w2} {:08x}\n", (i / 1000) + 1, (i % 12) + 1, s >> 32)
                .as_bytes(),
        );
    }
    assert!(prose.len() >= 4 << 20);
    for threads in [1usize, 8] {
        let opt = Options {
            block_size: 65_536,
            threads,
            checksum: false,
            ..Options::default()
        };
        let (stats_frame, _) = vxg::compress_with_stats(&prose, &opt).unwrap();
        let plain_frame = compress(&prose, &opt).unwrap();
        assert_eq!(stats_frame, plain_frame, "compress_with_stats diverged from compress ({threads}T)");
    }
}
