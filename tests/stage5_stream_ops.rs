use std::io::{self, Read};
use vxg::{
    aggregate_column, aggregate_column_stream, compress, count_where, count_where_stream,
    decompress, filter_frame_where, filter_frame_where_stream, filter_where, filter_where_stream,
    frame_row_count, frame_row_count_stream, group_by, group_by_stream, map_column,
    map_column_stream, project_frame, project_frame_stream, select_columns, select_columns_stream,
    update_column, update_column_stream, ColOp, ColPred, ColTarget, DecodeOptions, NumCmp, Options,
    PredExpr, StrCmp, StreamEncoder,
};

struct TinyChunkReader<'a> {
    data: &'a [u8],
    pos: usize,
    step: usize,
}

impl<'a> TinyChunkReader<'a> {
    fn new(data: &'a [u8], step: usize) -> Self {
        Self {
            data,
            pos: 0,
            step: step.max(1),
        }
    }
}

impl Read for TinyChunkReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        let n = buf.len().min(self.step).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn encode_chunked_stream(raw: &[u8], opts: &Options, chunk_size: usize) -> Vec<u8> {
    let mut enc = StreamEncoder::with_chunk_size(Vec::new(), opts.clone(), chunk_size).unwrap();
    let mut pos = 0usize;
    while pos < raw.len() {
        let end = (pos + 503).min(raw.len());
        io::Write::write_all(&mut enc, &raw[pos..end]).unwrap();
        pos = end;
    }
    enc.finish().unwrap()
}

fn concat_row_matches(rows: &[vxg::RowMatch]) -> Vec<u8> {
    let total: usize = rows.iter().map(|r| r.bytes.len()).sum();
    let mut out = Vec::with_capacity(total);
    for r in rows {
        out.extend_from_slice(&r.bytes);
    }
    out
}

#[test]
fn differential_ndjson_stream_ops_across_split_frames_monster_lines_and_eof() {
    let services = ["auth-gateway", "billing-core", "edge-router", "search-idx"];
    let regions = ["eu-west-1", "us-east-1", "ap-south-1"];
    let mut raw = Vec::new();
    for i in 0..420usize {
        let svc = services[i % services.len()];
        let reg = regions[i % regions.len()];
        let status = if i % 9 == 0 { 503 } else { 200 };
        let lat = (i * 17) % 1200;
        if i == 137 {
            // Monster 9.5 KiB JSON record spanning 3 consecutive 4096-byte frames!
            let big_blob = "X".repeat(9500);
            raw.extend_from_slice(
                format!(
                    "{{\"seq\":{i},\"service\":\"{svc}\",\"region\":\"{reg}\",\"status\":503,\"lat\":{lat},\"msg\":\"{big_blob}\"}}\n"
                )
                .as_bytes(),
            );
        } else if i % 15 == 0 {
            raw.extend_from_slice(
                format!(
                    "{{\"seq\":{i},\"service\":\"{svc}\",\"region\":\"{reg}\",\"status\":{status},\"lat\":{lat},\"tag\":\"quote\\\"\\u0041\"}}\n"
                )
                .as_bytes(),
            );
        } else {
            raw.extend_from_slice(
                format!(
                    "{{\"seq\":{i},\"service\":\"{svc}\",\"region\":\"{reg}\",\"status\":{status},\"lat\":{lat},\"msg\":\"item-{i}\"}}\n"
                )
                .as_bytes(),
            );
        }
    }
    // Unterminated final record at EOF (no trailing '\n')!
    raw.extend_from_slice(
        b"{\"seq\":9999,\"service\":\"auth-gateway\",\"region\":\"eu-west-1\",\"status\":503,\"lat\":888,\"msg\":\"eof-tail\"}",
    );

    let enc_opts = Options {
        block_size: 4096,
        threads: 2,
        ..Options::default()
    };
    let dec_ref = DecodeOptions {
        threads: 2,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };
    let whole_frame = compress(&raw, &enc_opts).unwrap();

    // Compound predicate: `status == 503 && (service == "auth-gateway" || lat >= 600) && !(seq == 0)`
    let p_status = PredExpr::col(ColPred::json_num("status", NumCmp::Eq(503)));
    let p_svc = PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"auth-gateway")));
    let p_lat = PredExpr::col(ColPred::json_num("lat", NumCmp::Ge(600)));
    let p_seq0 = PredExpr::col(ColPred::json_num("seq", NumCmp::Eq(0)));
    let or_children = [p_svc, p_lat];
    let p_or = PredExpr::any(&or_children);
    let p_not0 = PredExpr::not(&p_seq0);
    let all_children = [p_status, p_or, p_not0];
    let expr = PredExpr::all(&all_children);

    let proj_cols = [
        ColTarget::JsonKey("seq"),
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("lat"),
    ];
    let upd_pred = ColPred::json_num("status", NumCmp::Eq(503));
    let upper_fn = |b: &[u8]| b.to_ascii_uppercase();

    let ref_rows = frame_row_count(&whole_frame).unwrap() as u64;
    let ref_cnt = count_where(&whole_frame, &expr, &dec_ref).unwrap();
    let ref_flt_all =
        concat_row_matches(&filter_where(&whole_frame, &expr, usize::MAX, &dec_ref).unwrap());
    let ref_flt_lim = concat_row_matches(&filter_where(&whole_frame, &expr, 7, &dec_ref).unwrap());
    let ref_sel_all =
        select_columns(&whole_frame, Some(&expr), &proj_cols, usize::MAX, &dec_ref).unwrap();
    let ref_sel_lim = select_columns(&whole_frame, Some(&expr), &proj_cols, 11, &dec_ref).unwrap();
    let ref_agg = aggregate_column(
        &whole_frame,
        Some(&expr),
        ColTarget::JsonKey("lat"),
        &dec_ref,
    )
    .unwrap();
    let ref_grp = group_by(
        &whole_frame,
        Some(&expr),
        ColTarget::JsonKey("service"),
        Some(ColTarget::JsonKey("lat")),
        &dec_ref,
    )
    .unwrap();
    let ref_proj_raw = decompress(
        &project_frame(&whole_frame, Some(&expr), &proj_cols, &enc_opts, &dec_ref).unwrap(),
        &dec_ref,
    )
    .unwrap();
    let ref_ff_raw = decompress(
        &filter_frame_where(&whole_frame, &expr, &enc_opts, &dec_ref).unwrap(),
        &dec_ref,
    )
    .unwrap();
    let ref_upd_raw = decompress(
        &update_column(
            &whole_frame,
            &upd_pred,
            ColTarget::JsonKey("region"),
            ColOp::SetStr(b"failover-zone"),
            &enc_opts,
            &dec_ref,
        )
        .unwrap(),
        &dec_ref,
    )
    .unwrap();
    let ref_map_raw = decompress(
        &map_column(
            &whole_frame,
            ColTarget::JsonKey("service"),
            ColOp::MapStr(&upper_fn),
            &enc_opts,
            &dec_ref,
        )
        .unwrap(),
        &dec_ref,
    )
    .unwrap();

    for chunk_size in [4096usize, 8192, 12288] {
        for legacy in [false, true] {
            let s_enc = Options {
                block_size: 4096,
                legacy_v2: legacy,
                threads: 2,
                ..Options::default()
            };
            let stream_bytes = encode_chunked_stream(&raw, &s_enc, chunk_size);
            for threads in [1usize, 4] {
                let dec = DecodeOptions {
                    threads,
                    max_output: usize::MAX,
                    ..DecodeOptions::default()
                };
                assert_eq!(
                    frame_row_count_stream(TinyChunkReader::new(&stream_bytes, 97), &dec).unwrap(),
                    ref_rows
                );
                assert_eq!(
                    count_where_stream(TinyChunkReader::new(&stream_bytes, 97), &expr, &dec)
                        .unwrap(),
                    ref_cnt
                );

                let mut flt_all = Vec::new();
                let n_all = filter_where_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut flt_all,
                    &expr,
                    usize::MAX,
                    &dec,
                )
                .unwrap();
                assert_eq!(n_all, ref_cnt);
                assert_eq!(flt_all, ref_flt_all);

                let mut flt_lim = Vec::new();
                let n_lim = filter_where_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut flt_lim,
                    &expr,
                    7,
                    &dec,
                )
                .unwrap();
                assert_eq!(n_lim, 7);
                assert_eq!(flt_lim, ref_flt_lim);

                let mut sel_all = Vec::new();
                select_columns_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut sel_all,
                    Some(&expr),
                    &proj_cols,
                    usize::MAX,
                    &dec,
                )
                .unwrap();
                assert_eq!(sel_all, ref_sel_all);

                let mut sel_lim = Vec::new();
                select_columns_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut sel_lim,
                    Some(&expr),
                    &proj_cols,
                    11,
                    &dec,
                )
                .unwrap();
                assert_eq!(sel_lim, ref_sel_lim);

                assert_eq!(
                    aggregate_column_stream(
                        TinyChunkReader::new(&stream_bytes, 97),
                        Some(&expr),
                        ColTarget::JsonKey("lat"),
                        &dec,
                    )
                    .unwrap(),
                    ref_agg
                );
                assert_eq!(
                    group_by_stream(
                        TinyChunkReader::new(&stream_bytes, 97),
                        Some(&expr),
                        ColTarget::JsonKey("service"),
                        Some(ColTarget::JsonKey("lat")),
                        &dec,
                    )
                    .unwrap(),
                    ref_grp
                );

                let mut proj_stm = Vec::new();
                project_frame_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut proj_stm,
                    Some(&expr),
                    &proj_cols,
                    &s_enc,
                    &dec,
                )
                .unwrap();
                assert_eq!(decompress(&proj_stm, &dec).unwrap(), ref_proj_raw);

                let mut ff_stm = Vec::new();
                filter_frame_where_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut ff_stm,
                    &expr,
                    &s_enc,
                    &dec,
                )
                .unwrap();
                assert_eq!(decompress(&ff_stm, &dec).unwrap(), ref_ff_raw);

                let mut upd_stm = Vec::new();
                update_column_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut upd_stm,
                    &upd_pred,
                    ColTarget::JsonKey("region"),
                    ColOp::SetStr(b"failover-zone"),
                    &s_enc,
                    &dec,
                )
                .unwrap();
                assert_eq!(decompress(&upd_stm, &dec).unwrap(), ref_upd_raw);

                let mut map_stm = Vec::new();
                map_column_stream(
                    TinyChunkReader::new(&stream_bytes, 97),
                    &mut map_stm,
                    ColTarget::JsonKey("service"),
                    ColOp::MapStr(&upper_fn),
                    &s_enc,
                    &dec,
                )
                .unwrap();
                assert_eq!(decompress(&map_stm, &dec).unwrap(), ref_map_raw);
            }
        }
    }

    // 0-hit filtered stream must emit one canonical valid 0-byte VXG1 frame:
    let impossible = PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"__never__")));
    let multi = encode_chunked_stream(&raw, &enc_opts, 4096);
    let mut empty_ff = Vec::new();
    filter_frame_where_stream(
        TinyChunkReader::new(&multi, 89),
        &mut empty_ff,
        &impossible,
        &enc_opts,
        &dec_ref,
    )
    .unwrap();
    assert!(decompress(&empty_ff, &dec_ref).unwrap().is_empty());

    let mut empty_proj = Vec::new();
    project_frame_stream(
        TinyChunkReader::new(&multi, 89),
        &mut empty_proj,
        Some(&impossible),
        &proj_cols,
        &enc_opts,
        &dec_ref,
    )
    .unwrap();
    assert!(decompress(&empty_proj, &dec_ref).unwrap().is_empty());
}

#[test]
fn differential_csv_col0_boundary_split_across_frames() {
    let mut csv = Vec::new();
    for i in 0..500usize {
        let cat = [
            "alpha-long-category-token",
            "beta-long-category-token",
            "gamma-long-category-token",
        ][i % 3];
        let note = if i % 11 == 0 {
            "\"comma,inside,\"\"quoted\"\"\""
        } else {
            "plain"
        };
        let amount = (i * 29) % 1000;
        csv.extend_from_slice(
            format!("{cat}-{i:04},{amount},{note},region-{}\n", i % 4).as_bytes(),
        );
    }
    let enc_opts = Options {
        block_size: 4096,
        threads: 2,
        ..Options::default()
    };
    let dec_opts = DecodeOptions {
        threads: 4,
        max_output: usize::MAX,
        ..DecodeOptions::default()
    };
    let whole = compress(&csv, &enc_opts).unwrap();
    let multi = encode_chunked_stream(&csv, &enc_opts, 4096);

    let pred = PredExpr::col(ColPred::delimited_str(b',', 0, StrCmp::Prefix(b"beta-")));
    let cols = [
        ColTarget::Delimited {
            delim: b',',
            index: 0,
        },
        ColTarget::Delimited {
            delim: b',',
            index: 1,
        },
        ColTarget::Delimited {
            delim: b',',
            index: 2,
        },
    ];

    assert_eq!(
        count_where_stream(TinyChunkReader::new(&multi, 113), &pred, &dec_opts).unwrap(),
        count_where(&whole, &pred, &dec_opts).unwrap()
    );

    let mut sel_stm = Vec::new();
    select_columns_stream(
        TinyChunkReader::new(&multi, 113),
        &mut sel_stm,
        Some(&pred),
        &cols,
        usize::MAX,
        &dec_opts,
    )
    .unwrap();
    assert_eq!(
        sel_stm,
        select_columns(&whole, Some(&pred), &cols, usize::MAX, &dec_opts).unwrap()
    );

    let mut proj_stm = Vec::new();
    project_frame_stream(
        TinyChunkReader::new(&multi, 113),
        &mut proj_stm,
        Some(&pred),
        &cols,
        &enc_opts,
        &dec_opts,
    )
    .unwrap();
    let ref_proj = project_frame(&whole, Some(&pred), &cols, &enc_opts, &dec_opts).unwrap();
    assert_eq!(
        decompress(&proj_stm, &dec_opts).unwrap(),
        decompress(&ref_proj, &dec_opts).unwrap()
    );

    assert_eq!(
        group_by_stream(
            TinyChunkReader::new(&multi, 113),
            Some(&pred),
            ColTarget::Delimited {
                delim: b',',
                index: 3
            },
            Some(ColTarget::Delimited {
                delim: b',',
                index: 1
            }),
            &dec_opts,
        )
        .unwrap(),
        group_by(
            &whole,
            Some(&pred),
            ColTarget::Delimited {
                delim: b',',
                index: 3
            },
            Some(ColTarget::Delimited {
                delim: b',',
                index: 1
            }),
            &dec_opts,
        )
        .unwrap()
    );
}
