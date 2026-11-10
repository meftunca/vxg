use std::io::Write;
use std::time::Instant;
use vxg::{
    aggregate_column, compress, count_where, decompress, execute_sql, extract_block_into,
    extract_row, extract_row_range, filter_column, filter_frame, filter_where, frame_row_count,
    group_by, map_column, project_frame, recover_frame, select_columns, select_top_k_by, top_k_by,
    update_column, verify_frame, ColOp, ColPred, ColTarget, DecodeOptions, Frame, NumCmp, Options,
    PredExpr, SortDir, SqlFormat, SqlQuery, StrCmp, StreamDecoder, StreamEncoder, VERSION_V6,
};

struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn usize(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound.max(1)
    }
}

fn build_structured_ndjson(rows: usize) -> Vec<u8> {
    let regions = ["eu-west-1", "us-east-1", "ap-south-1", "sa-east-1"];
    let statuses = ["ok", "warn", "err", "retry"];
    let mut raw = Vec::with_capacity(rows * 112);
    for i in 0..rows {
        let reg = regions[i % regions.len()];
        let st = statuses[(i * 7) % statuses.len()];
        let amt = (i * 37) % 10_000;
        let lat = 5 + ((i * 13) % 450);
        let uid = 100_000 + i;
        let line = format!(
            "{{\"uid\":{uid},\"region\":\"{reg}\",\"status\":\"{st}\",\"amount\":{amt},\"latency_ms\":{lat},\"ok\":true}}\n"
        );
        raw.extend_from_slice(line.as_bytes());
    }
    raw
}

#[test]
fn v6_direct_warm_operators_parity_without_reassembly() {
    let raw = build_structured_ndjson(12_000);
    let opt_v5 = Options {
        block_size: 32_768,
        threads: 4,
        record_aligned: true,
        col_group: false,
        ..Options::default()
    };
    let opt_v6 = Options {
        col_group: true,
        ..opt_v5.clone()
    };
    let f5 = compress(&raw, &opt_v5).unwrap();
    let f6 = compress(&raw, &opt_v6).unwrap();

    let p6 = Frame::parse(&f6, usize::MAX).unwrap();
    assert_eq!(p6.version(), VERSION_V6);
    assert!(p6.block_count() >= 20);
    assert_eq!(verify_frame(&f6, &DecodeOptions::default()).unwrap(), raw.len() as u64);
    assert_eq!(frame_row_count(&f6).unwrap(), 12_000);

    for th in [1, 4] {
        let dec = DecodeOptions {
            threads: th,
            max_output: usize::MAX,
            ..DecodeOptions::default()
        };
        assert_eq!(decompress(&f6, &dec).unwrap(), raw);

        // 1. ops_pred (`filter_column`, `filter_where`, `count_where`)
        let pred = ColPred::json_str("region", StrCmp::Eq(b"eu-west-1"));
        assert_eq!(
            filter_column(&f6, &pred, 500, &dec).unwrap(),
            filter_column(&f5, &pred, 500, &dec).unwrap()
        );
        let children = [
            PredExpr::col(ColPred::json_str("region", StrCmp::Eq(b"eu-west-1"))),
            PredExpr::col(ColPred::json_num("amount", NumCmp::Gt(5000))),
        ];
        let expr = PredExpr::all(&children);
        assert_eq!(count_where(&f6, &expr, &dec).unwrap(), count_where(&f5, &expr, &dec).unwrap());
        assert_eq!(
            filter_where(&f6, &expr, 250, &dec).unwrap(),
            filter_where(&f5, &expr, 250, &dec).unwrap()
        );

        // 2. ops_agg (`aggregate_column`, `group_by`)
        let t_amt = ColTarget::JsonKey("amount");
        let t_reg = ColTarget::JsonKey("region");
        assert_eq!(
            aggregate_column(&f6, Some(&expr), t_amt, &dec).unwrap(),
            aggregate_column(&f5, Some(&expr), t_amt, &dec).unwrap()
        );
        assert_eq!(
            group_by(&f6, Some(&expr), t_reg, Some(t_amt), &dec).unwrap(),
            group_by(&f5, Some(&expr), t_reg, Some(t_amt), &dec).unwrap()
        );

        // 3. ops_topk_exec (`top_k_by`, `select_top_k_by`)
        let sk = ColTarget::JsonKey("latency_ms");
        let proj = [ColTarget::JsonKey("uid"), ColTarget::JsonKey("latency_ms")];
        assert_eq!(
            top_k_by(&f6, Some(&expr), sk, SortDir::Desc, 25, &dec).unwrap(),
            top_k_by(&f5, Some(&expr), sk, SortDir::Desc, 25, &dec).unwrap()
        );
        assert_eq!(
            select_top_k_by(&f6, Some(&expr), &proj, sk, SortDir::Desc, 25, &dec).unwrap(),
            select_top_k_by(&f5, Some(&expr), &proj, sk, SortDir::Desc, 25, &dec).unwrap()
        );

        // 4. project & filter_frame & mutations
        assert_eq!(
            select_columns(&f6, Some(&expr), &proj, 300, &dec).unwrap(),
            select_columns(&f5, Some(&expr), &proj, 300, &dec).unwrap()
        );
        let pf6 = project_frame(&f6, Some(&expr), &proj, &opt_v5, &dec).unwrap();
        let pf5 = project_frame(&f5, Some(&expr), &proj, &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&pf6, &dec).unwrap(), decompress(&pf5, &dec).unwrap());

        let ff6 = filter_frame(&f6, &pred, &opt_v5, &dec).unwrap();
        let ff5 = filter_frame(&f5, &pred, &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&ff6, &dec).unwrap(), decompress(&ff5, &dec).unwrap());

        let mf6 = map_column(&f6, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        let mf5 = map_column(&f5, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&mf6, &dec).unwrap(), decompress(&mf5, &dec).unwrap());

        let uf6 = update_column(&f6, &pred, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        let uf5 = update_column(&f5, &pred, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&uf6, &dec).unwrap(), decompress(&uf5, &dec).unwrap());

        // 5. Full SQL suite (Q1..Q5 equivalents)
        for sql in [
            "SELECT region, status, COUNT(*), SUM(amount), MIN(latency_ms), MAX(latency_ms) FROM frame GROUP BY region, status ORDER BY region ASC, status ASC",
            "SELECT COUNT(*), SUM(amount) FROM frame WHERE amount > 2500",
            "SELECT status, COUNT(*), AVG(latency_ms) FROM frame WHERE region = 'us-east-1' GROUP BY status ORDER BY status ASC",
            "SELECT uid, region, amount, latency_ms FROM frame WHERE latency_ms > 400 ORDER BY amount DESC, uid ASC LIMIT 20",
        ] {
            let q = SqlQuery::parse(sql).unwrap();
            assert_eq!(
                execute_sql(&f6, &q, SqlFormat::Json, false, &dec).unwrap(),
                execute_sql(&f5, &q, SqlFormat::Json, false, &dec).unwrap(),
                "SQL mismatch for {sql} (threads={th})"
            );
        }
    }
}

#[test]
fn v6_warm_extract_row_is_direct_and_not_slower_than_extract_block() {
    let raw = build_structured_ndjson(20_000);
    let opt_v5 = Options {
        block_size: 65_536,
        threads: 4,
        record_aligned: true,
        col_group: false,
        ..Options::default()
    };
    let opt_v6 = Options {
        col_group: true,
        ..opt_v5.clone()
    };
    let f5 = compress(&raw, &opt_v5).unwrap();
    let f6 = compress(&raw, &opt_v6).unwrap();
    let dec = DecodeOptions::default();

    // Exactness check across block boundaries:
    for r in [0usize, 1, 517, 4096, 10_123, 19_999] {
        assert_eq!(extract_row(&f6, r, &dec).unwrap(), extract_row(&f5, r, &dec).unwrap());
    }
    assert_eq!(
        extract_row_range(&f6, 1500, 2200, &dec).unwrap(),
        extract_row_range(&f5, 1500, 2200, &dec).unwrap()
    );

    // Warm latency check (median of 15 runs): warm extract_row <= warm extract_block
    let target_row = 10_123usize;
    let target_blk = 15usize;
    let p6 = Frame::parse(&f6, usize::MAX).unwrap();
    let blk_rn = p6.blocks().nth(target_blk).unwrap().raw_size;
    let mut blk_buf = vec![0u8; blk_rn];

    let _ = extract_row(&f6, target_row, &dec).unwrap();
    let _ = extract_block_into(&f6, target_blk, &mut blk_buf, &dec).unwrap();

    let mut row_times = Vec::with_capacity(15);
    let mut blk_times = Vec::with_capacity(15);
    for _ in 0..15 {
        let t0 = Instant::now();
        let r = extract_row(&f6, target_row, &dec).unwrap();
        std::hint::black_box(&r);
        row_times.push(t0.elapsed().as_nanos() as u64);

        let t1 = Instant::now();
        let n = extract_block_into(&f6, target_blk, &mut blk_buf, &dec).unwrap();
        std::hint::black_box(n);
        blk_times.push(t1.elapsed().as_nanos() as u64);
    }
    row_times.sort_unstable();
    blk_times.sort_unstable();
    let med_row_ns = row_times[7];
    let med_blk_ns = blk_times[7];
    assert!(
        med_row_ns <= med_blk_ns * 12 / 10,
        "v6 warm extract_row ({med_row_ns} ns) must not be slower than extract_block ({med_blk_ns} ns)"
    );
}

#[test]
fn v6_stream_encoder_incremental_multi_group_roundtrip() {
    let raw = build_structured_ndjson(15_000);
    let opt = Options {
        block_size: 32_768,
        threads: 2,
        record_aligned: true,
        col_group: true,
        ..Options::default()
    };
    let mut streamed = Vec::new();
    {
        let mut enc = StreamEncoder::new(&mut streamed, opt).unwrap();
        for chunk in raw.chunks(19_997) {
            enc.write_all(chunk).unwrap();
        }
        enc.finish().unwrap();
    }
    let frames = Frame::parse_all(&streamed, usize::MAX).unwrap();
    assert!(!frames.is_empty() && frames.iter().all(|f| f.version() == VERSION_V6));
    assert_eq!(decompress(&streamed, &DecodeOptions::default()).unwrap(), raw);

    let mut dec = StreamDecoder::new(streamed.as_slice(), DecodeOptions::default()).unwrap();
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
    assert_eq!(out, raw);
}

#[test]
fn v6_corruption_fuzzer_1250_trials_bitflip_truncate_zeroing() {
    // Build a multi-group v6 archive (Group 0: 4 cols, Group 1: 6 cols, Group 2: LZ/Raw + 4 cols)
    let mut raw = Vec::new();
    for i in 0..400usize {
        let line = format!(
            "{{\"id\":{},\"region\":\"eu-west-1\",\"status\":\"ok\",\"cents\":{}}}\n",
            1000 + i,
            (i * 37) % 10000
        );
        raw.extend_from_slice(line.as_bytes());
    }
    for i in 0..400usize {
        let line = format!(
            "{{\"id\":{},\"region\":\"us-east-1\",\"status\":\"ok\",\"cents\":{},\"tax\":{},\"zone\":\"A\"}}\n",
            5000 + i,
            (i * 53) % 10000,
            i % 99
        );
        raw.extend_from_slice(line.as_bytes());
    }
    let opt = Options {
        block_size: 4096,
        threads: 2,
        record_aligned: true,
        col_group: true,
        ..Options::default()
    };
    let frame = compress(&raw, &opt).unwrap();
    let parsed = Frame::parse(&frame, usize::MAX).unwrap();
    assert_eq!(parsed.version(), VERSION_V6);
    let blocks: Vec<_> = parsed.blocks().collect();
    let total_blocks = blocks.len();
    assert!(total_blocks >= 8);

    let dec = DecodeOptions::default();
    let mut rng = XorShift64(0xCAFE_BABE_DEAD_BEEFu64);
    let mut total_possible_blocks = 0usize;
    let mut total_recovered_blocks = 0usize;
    let mut primary_hdr_trials = 0usize;
    let mut primary_hdr_100pct_recovered = 0usize;

    // Locate all primary "VXG6" group headers in `frame`:
    let vxg6_offsets: Vec<usize> = frame
        .windows(4)
        .enumerate()
        .filter_map(|(i, w)| (w == b"VXG6").then_some(i))
        .collect();
    assert!(vxg6_offsets.len() >= 2);

    // 1. 450 bit-flip trials (including targeted primary group header and global block dir corruptions):
    for trial in 0..450usize {
        let mut mutated = frame.clone();
        let is_primary_hdr_only = trial < 100;
        if is_primary_hdr_only {
            let g_off = vxg6_offsets[trial % vxg6_offsets.len()];
            let byte_off = g_off + (trial % 16);
            mutated[byte_off] ^= 1u8 << (trial % 8);
            primary_hdr_trials += 1;
        } else {
            let flips = 1 + rng.usize(4);
            for _ in 0..flips {
                let pos = rng.usize(mutated.len());
                let bit = 1u8 << rng.usize(8);
                mutated[pos] ^= bit;
            }
        }

        // Strict decompress: must NEVER panic and NEVER return wrong output!
        if let Ok(strict_out) = decompress(&mutated, &dec) {
            assert_eq!(strict_out, raw, "silent wrong output in bit-flip trial {trial}");
        }
        // Recovery: must NEVER panic and every recovered block must match original raw block!
        if let Ok(rep) = recover_frame(&mutated, &dec) {
            let mut cursor = 0usize;
            for rb in &rep.recovered_blocks {
                let b = &blocks[rb.index];
                let expected = &raw[b.raw_offset..b.raw_offset + b.raw_size];
                assert_eq!(
                    &rep.data[cursor..cursor + rb.raw_size],
                    expected,
                    "corrupt recovered block {} in bit-flip trial {trial}",
                    rb.index
                );
                cursor += rb.raw_size;
            }
            total_possible_blocks += total_blocks;
            total_recovered_blocks += rep.recovered_blocks.len();
            if is_primary_hdr_only && rep.recovered_blocks.len() == total_blocks {
                primary_hdr_100pct_recovered += 1;
            }
        }
    }

    // 2. 400 truncation trials across the entire frame length:
    for trial in 0..400usize {
        let cut = (trial * frame.len()) / 400;
        let truncated = &frame[..cut];
        if let Ok(strict_out) = decompress(truncated, &dec) {
            assert_eq!(strict_out, raw, "silent wrong output in truncation trial {trial}");
        }
        if let Ok(rep) = recover_frame(truncated, &dec) {
            let mut cursor = 0usize;
            for rb in &rep.recovered_blocks {
                let b = &blocks[rb.index];
                let expected = &raw[b.raw_offset..b.raw_offset + b.raw_size];
                assert_eq!(
                    &rep.data[cursor..cursor + rb.raw_size],
                    expected,
                    "corrupt recovered block {} in truncation trial {trial}",
                    rb.index
                );
                cursor += rb.raw_size;
            }
            total_possible_blocks += total_blocks;
            total_recovered_blocks += rep.recovered_blocks.len();
        }
    }

    // 3. 400 zeroing trials (1..128 bytes zeroed at random offsets, including global block dir & group headers):
    for trial in 0..400usize {
        let mut mutated = frame.clone();
        if trial < 80 {
            // Zero out an entire primary group header (16 bytes): backup trailer must recover 100%!
            let g_off = vxg6_offsets[trial % vxg6_offsets.len()];
            mutated[g_off..g_off + 16].fill(0);
            primary_hdr_trials += 1;
            if let Ok(rep) = recover_frame(&mutated, &dec) {
                if rep.recovered_blocks.len() == total_blocks {
                    primary_hdr_100pct_recovered += 1;
                }
            }
        } else {
            let start = rng.usize(mutated.len());
            let zlen = (1 + rng.usize(64)).min(mutated.len() - start);
            mutated[start..start + zlen].fill(0);
        }

        if let Ok(strict_out) = decompress(&mutated, &dec) {
            assert_eq!(strict_out, raw, "silent wrong output in zeroing trial {trial}");
        }
        if let Ok(rep) = recover_frame(&mutated, &dec) {
            let mut cursor = 0usize;
            for rb in &rep.recovered_blocks {
                let b = &blocks[rb.index];
                let expected = &raw[b.raw_offset..b.raw_offset + b.raw_size];
                assert_eq!(
                    &rep.data[cursor..cursor + rb.raw_size],
                    expected,
                    "corrupt recovered block {} in zeroing trial {trial}",
                    rb.index
                );
                cursor += rb.raw_size;
            }
            total_possible_blocks += total_blocks;
            total_recovered_blocks += rep.recovered_blocks.len();
        }
    }

    assert_eq!(
        primary_hdr_100pct_recovered, primary_hdr_trials,
        "100% of primary group header corruptions must recover all blocks via backup trailer"
    );
    assert!(
        total_recovered_blocks * 100 / total_possible_blocks.max(1) >= 60,
        "expected high recovery ratio across 1,250 fuzz trials (got {}/{})",
        total_recovered_blocks,
        total_possible_blocks
    );
}
