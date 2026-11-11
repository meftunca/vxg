use std::io::Write;
use std::time::Instant;
use vxg::{
    aggregate_column, count_where, decompress, execute_sql, extract_block_into, extract_row,
    extract_row_range, filter_column, filter_frame, filter_where, frame_row_count, group_by,
    map_column, project_frame, recover_frame, select_columns, select_top_k_by, top_k_by,
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
fn v6_golden_read_compat_operators_and_point_access_parity() {
    let f6 = include_bytes!("vectors/v6-col-group.vxg");
    let f5 = include_bytes!("vectors/v5-record-aligned.vxg");
    let raw = include_bytes!("vectors/v5-record-aligned.raw");

    let p6 = Frame::parse(f6, usize::MAX).unwrap();
    assert_eq!(p6.version(), VERSION_V6);
    assert!(p6.block_count() >= 2);
    assert_eq!(verify_frame(f6, &DecodeOptions::default()).unwrap(), raw.len() as u64);
    assert_eq!(frame_row_count(f6).unwrap(), frame_row_count(f5).unwrap());

    let opt_v5 = Options {
        block_size: 4096,
        threads: 2,
        record_aligned: true,
        ..Options::default()
    };

    for th in [1, 4] {
        let dec = DecodeOptions {
            threads: th,
            max_output: usize::MAX,
            ..DecodeOptions::default()
        };
        assert_eq!(decompress(f6, &dec).unwrap(), raw);

        // CSV frozen vector columns: id,region,status,trips,cents (0..4)
        let pred = ColPred::delimited_str(b',', 1, StrCmp::Eq(b"eu-west-1"));
        assert_eq!(
            filter_column(f6, &pred, 100, &dec).unwrap(),
            filter_column(f5, &pred, 100, &dec).unwrap()
        );
        let children = [
            PredExpr::col(ColPred::delimited_str(b',', 1, StrCmp::Eq(b"eu-west-1"))),
            PredExpr::col(ColPred::delimited_num(b',', 3, NumCmp::Gt(100))),
        ];
        let expr = PredExpr::all(&children);
        assert_eq!(count_where(f6, &expr, &dec).unwrap(), count_where(f5, &expr, &dec).unwrap());
        assert_eq!(
            filter_where(f6, &expr, 50, &dec).unwrap(),
            filter_where(f5, &expr, 50, &dec).unwrap()
        );

        let t_trips = ColTarget::Delimited { delim: b',', index: 3 };
        let t_reg = ColTarget::Delimited { delim: b',', index: 1 };
        assert_eq!(
            aggregate_column(f6, Some(&expr), t_trips, &dec).unwrap(),
            aggregate_column(f5, Some(&expr), t_trips, &dec).unwrap()
        );
        assert_eq!(
            group_by(f6, Some(&expr), t_reg, Some(t_trips), &dec).unwrap(),
            group_by(f5, Some(&expr), t_reg, Some(t_trips), &dec).unwrap()
        );

        let proj = [ColTarget::Delimited { delim: b',', index: 0 }, t_trips];
        assert_eq!(
            top_k_by(f6, Some(&expr), t_trips, SortDir::Desc, 15, &dec).unwrap(),
            top_k_by(f5, Some(&expr), t_trips, SortDir::Desc, 15, &dec).unwrap()
        );
        assert_eq!(
            select_top_k_by(f6, Some(&expr), &proj, t_trips, SortDir::Desc, 15, &dec).unwrap(),
            select_top_k_by(f5, Some(&expr), &proj, t_trips, SortDir::Desc, 15, &dec).unwrap()
        );
        assert_eq!(
            select_columns(f6, Some(&expr), &proj, 50, &dec).unwrap(),
            select_columns(f5, Some(&expr), &proj, 50, &dec).unwrap()
        );

        let pf6 = project_frame(f6, Some(&expr), &proj, &opt_v5, &dec).unwrap();
        let pf5 = project_frame(f5, Some(&expr), &proj, &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&pf6, &dec).unwrap(), decompress(&pf5, &dec).unwrap());

        let ff6 = filter_frame(f6, &pred, &opt_v5, &dec).unwrap();
        let ff5 = filter_frame(f5, &pred, &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&ff6, &dec).unwrap(), decompress(&ff5, &dec).unwrap());

        let mf6 = map_column(f6, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        let mf5 = map_column(f5, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&mf6, &dec).unwrap(), decompress(&mf5, &dec).unwrap());

        let uf6 = update_column(f6, &pred, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        let uf5 = update_column(f5, &pred, t_reg, ColOp::SetStr(b"eu-central"), &opt_v5, &dec).unwrap();
        assert_eq!(decompress(&uf6, &dec).unwrap(), decompress(&uf5, &dec).unwrap());

        let q = SqlQuery::parse(
            "SELECT region, COUNT(*), SUM(trips) FROM frame WHERE trips > 100 GROUP BY region ORDER BY region ASC",
        )
        .unwrap();
        assert_eq!(
            execute_sql(f6, &q, SqlFormat::Csv, true, &dec).unwrap(),
            execute_sql(f5, &q, SqlFormat::Csv, true, &dec).unwrap()
        );
    }

    let dec = DecodeOptions::default();
    let total_rows = frame_row_count(f6).unwrap();
    for r in [0usize, 1, total_rows / 2, total_rows - 1] {
        assert_eq!(extract_row(f6, r, &dec).unwrap(), extract_row(f5, r, &dec).unwrap());
    }
    assert_eq!(
        extract_row_range(f6, 5, 25, &dec).unwrap(),
        extract_row_range(f5, 5, 25, &dec).unwrap()
    );

    let blk_rn = p6.blocks().nth(1).unwrap().raw_size;
    let mut blk_buf = vec![0u8; blk_rn];
    let _ = extract_row(f6, total_rows / 2, &dec).unwrap();
    let _ = extract_block_into(f6, 1, &mut blk_buf, &dec).unwrap();
    let mut row_times = Vec::with_capacity(15);
    let mut blk_times = Vec::with_capacity(15);
    for _ in 0..15 {
        let t0 = Instant::now();
        let r = extract_row(f6, total_rows / 2, &dec).unwrap();
        std::hint::black_box(&r);
        row_times.push(t0.elapsed().as_nanos() as u64);

        let t1 = Instant::now();
        let n = extract_block_into(f6, 1, &mut blk_buf, &dec).unwrap();
        std::hint::black_box(n);
        blk_times.push(t1.elapsed().as_nanos() as u64);
    }
    row_times.sort_unstable();
    blk_times.sort_unstable();
    assert!(row_times[7] <= blk_times[7] * 15 / 10 + 50_000);
}

#[test]
fn v5_stream_encoder_record_aligned_and_dict_reuse_roundtrip() {
    let raw = build_structured_ndjson(15_000);
    let opt = Options {
        block_size: 32_768,
        threads: 2,
        record_aligned: true,
        ..Options::default()
    };
    let mut streamed = Vec::new();
    {
        let mut enc = StreamEncoder::with_chunk_size(&mut streamed, opt, 256 * 1024).unwrap();
        for chunk in raw.chunks(500_003) {
            enc.write_all(chunk).unwrap();
        }
        enc.finish().unwrap();
    }
    let frames = Frame::parse_all(&streamed, usize::MAX).unwrap();
    assert!(frames.len() >= 4 && frames.iter().all(|f| f.version() == 5));
    assert_eq!(decompress(&streamed, &DecodeOptions::default()).unwrap(), raw);
    assert_eq!(frame_row_count(&streamed).unwrap(), 15_000);

    let mut dec = StreamDecoder::new(streamed.as_slice(), DecodeOptions::default()).unwrap();
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut dec, &mut out).unwrap();
    assert_eq!(out, raw);
}

#[test]
fn v6_corruption_fuzzer_1250_trials_bitflip_truncate_zeroing() {
    let frame = include_bytes!("vectors/v6-col-group.vxg");
    let raw = include_bytes!("vectors/v5-record-aligned.raw");
    let parsed = Frame::parse(frame, usize::MAX).unwrap();
    assert_eq!(parsed.version(), VERSION_V6);
    let blocks: Vec<_> = parsed.blocks().collect();
    let total_blocks = blocks.len();
    assert!(total_blocks >= 2);
    let total_rows = frame_row_count(frame).unwrap();

    let dec = DecodeOptions::default();
    let pred_eu = PredExpr::col(ColPred::delimited_str(b',', 1, StrCmp::Eq(b"eu-west-1")));
    let t_trips = ColTarget::Delimited { delim: b',', index: 3 };
    let t_reg = ColTarget::Delimited { delim: b',', index: 1 };
    let proj_cols = [ColTarget::Delimited { delim: b',', index: 0 }, t_trips];

    let ref_count = count_where(frame, &pred_eu, &dec).unwrap();
    let ref_agg = aggregate_column(frame, Some(&pred_eu), t_trips, &dec).unwrap();
    let ref_gb = group_by(frame, None, t_reg, Some(t_trips), &dec).unwrap();
    let ref_sel = select_columns(frame, Some(&pred_eu), &proj_cols, 50, &dec).unwrap();
    let ref_topk = top_k_by(frame, Some(&pred_eu), t_trips, SortDir::Desc, 15, &dec).unwrap();
    let ref_row_0 = extract_row(frame, 0, &dec).unwrap();
    let mid_row = total_rows / 2;
    let ref_row_mid = extract_row(frame, mid_row, &dec).unwrap();
    let mut ref_blk0 = vec![0u8; blocks[0].raw_size];
    extract_block_into(frame, 0, &mut ref_blk0, &dec).unwrap();
    let mid_blk_idx = total_blocks / 2;
    let mut ref_blk_mid = vec![0u8; blocks[mid_blk_idx].raw_size];
    extract_block_into(frame, mid_blk_idx, &mut ref_blk_mid, &dec).unwrap();

    let verify_query_path_no_silent_corruption = |corrupt: &[u8], label: &str, trial: usize| {
        if let Ok(c) = count_where(corrupt, &pred_eu, &dec) {
            assert_eq!(c, ref_count, "silent wrong count_where in {label} trial {trial}");
        }
        if let Ok(a) = aggregate_column(corrupt, Some(&pred_eu), t_trips, &dec) {
            assert_eq!(a, ref_agg, "silent wrong aggregate_column in {label} trial {trial}");
        }
        if let Ok(g) = group_by(corrupt, None, t_reg, Some(t_trips), &dec) {
            assert_eq!(g, ref_gb, "silent wrong group_by in {label} trial {trial}");
        }
        if let Ok(s) = select_columns(corrupt, Some(&pred_eu), &proj_cols, 50, &dec) {
            assert_eq!(s, ref_sel, "silent wrong select_columns in {label} trial {trial}");
        }
        if let Ok(tk) = top_k_by(corrupt, Some(&pred_eu), t_trips, SortDir::Desc, 15, &dec) {
            assert_eq!(tk, ref_topk, "silent wrong top_k_by in {label} trial {trial}");
        }
        if let Ok(r0) = extract_row(corrupt, 0, &dec) {
            assert_eq!(r0, ref_row_0, "silent wrong extract_row(0) in {label} trial {trial}");
        }
        if let Ok(rm) = extract_row(corrupt, mid_row, &dec) {
            assert_eq!(rm, ref_row_mid, "silent wrong extract_row(mid) in {label} trial {trial}");
        }
        let mut b0 = vec![0u8; ref_blk0.len()];
        if let Ok(n) = extract_block_into(corrupt, 0, &mut b0, &dec) {
            assert_eq!(&b0[..n], &ref_blk0[..], "silent wrong extract_block(0) in {label} trial {trial}");
        }
        let mut bm = vec![0u8; ref_blk_mid.len()];
        if let Ok(n) = extract_block_into(corrupt, mid_blk_idx, &mut bm, &dec) {
            assert_eq!(&bm[..n], &ref_blk_mid[..], "silent wrong extract_block(mid) in {label} trial {trial}");
        }
    };

    let mut rng = XorShift64(0xCAFE_BABE_DEAD_BEEFu64);
    let mut primary_hdr_trials = 0usize;
    let mut primary_hdr_100pct_recovered = 0usize;

    let vxg6_offsets: Vec<usize> = frame
        .windows(4)
        .enumerate()
        .filter_map(|(i, w)| (w == b"VXG6").then_some(i))
        .collect();
    assert!(!vxg6_offsets.is_empty());

    // 1. 450 bit-flip trials:
    for trial in 0..450usize {
        let mut mutated = frame.to_vec();
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

        if let Ok(strict_out) = decompress(&mutated, &dec) {
            assert_eq!(strict_out, raw, "silent wrong output in bit-flip trial {trial}");
        }
        verify_query_path_no_silent_corruption(&mutated, "bit-flip", trial);
        if let Ok(rep) = recover_frame(&mutated, &dec) {
            let mut cursor = 0usize;
            for rb in &rep.recovered_blocks {
                let b = &blocks[rb.index];
                let expected = &raw[b.raw_offset..b.raw_offset + b.raw_size];
                assert_eq!(&rep.data[cursor..cursor + rb.raw_size], expected);
                cursor += rb.raw_size;
            }
            if is_primary_hdr_only && rep.recovered_blocks.len() == total_blocks {
                primary_hdr_100pct_recovered += 1;
            }
        }
    }

    // 2. 400 truncation trials:
    for trial in 0..400usize {
        let cut = (trial * frame.len()) / 400;
        let truncated = &frame[..cut];
        if let Ok(strict_out) = decompress(truncated, &dec) {
            assert_eq!(strict_out, raw, "silent wrong output in truncation trial {trial}");
        }
        verify_query_path_no_silent_corruption(truncated, "truncation", trial);
        if let Ok(rep) = recover_frame(truncated, &dec) {
            let mut cursor = 0usize;
            for rb in &rep.recovered_blocks {
                let b = &blocks[rb.index];
                let expected = &raw[b.raw_offset..b.raw_offset + b.raw_size];
                assert_eq!(&rep.data[cursor..cursor + rb.raw_size], expected);
                cursor += rb.raw_size;
            }
        }
    }

    // 3. 400 zeroing trials:
    for trial in 0..400usize {
        let mut mutated = frame.to_vec();
        if trial < 80 {
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
        verify_query_path_no_silent_corruption(&mutated, "zeroing", trial);
        if let Ok(rep) = recover_frame(&mutated, &dec) {
            let mut cursor = 0usize;
            for rb in &rep.recovered_blocks {
                let b = &blocks[rb.index];
                let expected = &raw[b.raw_offset..b.raw_offset + b.raw_size];
                assert_eq!(&rep.data[cursor..cursor + rb.raw_size], expected);
                cursor += rb.raw_size;
            }
        }
    }

    assert_eq!(
        primary_hdr_100pct_recovered, primary_hdr_trials,
        "100% of primary group header corruptions must recover all blocks via backup trailer"
    );
}
