//! Stage 9 integration & differential oracle tests for `Format v5`:
//! - `BLOCK_DICT_LZ` (`kind = 3`) + frame-shared 32 KiB prefix dictionary (`FLAG_FRAME_DICT`)
//! - `Options::record_aligned` (`FLAG_VAR_RAW`) zero-split-line block snapping
//! - Full parity across `1T`/`4T`, `compress_into`/`decompress_into`, `ValidatedFrame`,
//!   `verify_frame`, `StreamEncoder`/`StreamDecoder`, `SQL`, and frame-rewriting ops
//! - Malformed `v5` header, `var_raw`, wire dictionary, and `DICT_LZ` payload rejection

use std::io::Cursor;

use vxg::{
    aggregate_column, compress, compress_into, compress_stream, count_where, decompress,
    decompress_into, decompress_stream, execute_sql, execute_sql_stream, extract_block_into,
    filter_frame_where, filter_where, frame_row_count, group_by, map_column, max_compressed_size,
    project_frame, select_columns, top_k_by, update_column, verify_frame, ColOp, ColPred,
    ColTarget, DecodeOptions, Frame, NumCmp, Options, PredExpr, SortDir, SqlFormat, SqlQuery,
    StrCmp, BLOCK_COLUMN, BLOCK_DICT_LZ,
};

fn build_free_text_log_corpus(lines: usize) -> Vec<u8> {
    let agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 Version/17.6 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",
        "curl/8.7.1 (aarch64-apple-darwin23.0) libcurl/8.7.1 BoringSSL zlib/1.2.12",
    ];
    let paths = [
        "/api/v2/organizations/maple-tech/repositories/vxg/actions/runs?per_page=50&status=completed",
        "/assets/bundles/vendor-chunk-9f8e7d6c5b4a.min.js.map",
        "/graphql?operationName=RepositoryIssuesDashboardQuery&variables=%7B%22first%22%3A25%7D",
        "/internal/healthz/readiness?cluster=eu-central-1a&probe=deep",
    ];
    let prose = [
        "PackageKit: Extracting file://localhost/Library/Updates/092-11421/CoreTypes.pkg (destination=/var/folders/zz/zyxvpxvq6csfxvn_n0000000000000/C/PKInstallSandboxManager)",
        "installd[842]: Installed \"Command Line Tools for Xcode 16.1\" (16.1.0.0.1.1729049160) in 14.82 seconds with status 0",
        "softwareupdated[411]: BackgroundCatalogSync: Validating cryptographic signature on catalog https://swscan.apple.com/content/catalogs/others/index-15.sucatalog.gz",
    ];
    let mut out = Vec::with_capacity(lines * 210);
    for i in 0..lines {
        if i % 3 == 0 {
            out.extend_from_slice(
                format!(
                    "2026-11-08 14:{:02}:{:02}+01 maple-host {} — seq={} bytes={} msg=\"{}#{}\"\n",
                    (i / 60) % 60,
                    i % 60,
                    prose[i % prose.len()],
                    i,
                    256 + (i * 37) % 8192,
                    paths[i % paths.len()],
                    i ^ 0x5a5a
                )
                .as_bytes(),
            );
        } else {
            out.extend_from_slice(
                format!(
                    "10.42.{}.{} - - [08/Nov/2026:14:{:02}:{:02} +0100] \"GET {}&req={} HTTP/1.1\" {} {} \"https://docs.example.org/guide/{}\" \"{}\"\n",
                    (i >> 8) & 255,
                    i & 255,
                    (i / 60) % 60,
                    i % 60,
                    paths[(i * 3) % paths.len()],
                    i,
                    if i % 11 == 0 { 500 } else { 200 },
                    512 + (i * 97) % 65536,
                    i % 19,
                    agents[(i * 5) % agents.len()]
                )
                .as_bytes(),
            );
        }
    }
    out
}

fn build_tabular_ndjson(rows: usize) -> Vec<u8> {
    let services = ["auth-api", "billing-core", "search-idx", "edge-gw"];
    let mut out = Vec::with_capacity(rows * 120);
    for i in 0..rows {
        let svc = services[i % services.len()];
        let status = if i % 9 == 0 { 500 } else { 200 };
        let bytes = 100 + ((i * 73) % 4000);
        let delta = (i as i64 % 41) - 20;
        out.extend_from_slice(
            format!(
                "{{\"ts\":\"2026-11-08T12:{:02}:{:02}Z\",\"seq\":{i},\"service\":\"{svc}\",\"status\":{status},\"bytes\":{bytes},\"delta\":{delta},\"actor\":{{\"login\":\"user-{}\"}}}}\n",
                (i / 60) % 60,
                i % 60,
                i % 32
            )
            .as_bytes(),
        );
    }
    out
}

#[test]
fn v5_dict_lz_roundtrip_determinism_validated_and_ops_parity() {
    let raw = build_free_text_log_corpus(2600);
    assert!(raw.len() > 65536 * 5);

    let opt_1t = Options {
        block_size: 65536,
        threads: 1,
        checksum: true,
        ..Options::default()
    };
    let opt_4t = Options {
        threads: 4,
        ..opt_1t.clone()
    };
    let dec_1t = DecodeOptions {
        threads: 1,
        max_output: raw.len() + 1024,
        verify_checksum: true,
        ..DecodeOptions::default()
    };
    let dec_4t = DecodeOptions {
        threads: 4,
        ..dec_1t.clone()
    };

    let f_1t = compress(&raw, &opt_1t).unwrap();
    let f_4t = compress(&raw, &opt_4t).unwrap();
    assert_eq!(f_1t, f_4t, "v5 DICT_LZ must be 100% deterministic across threads");
    assert!(f_1t.len() <= max_compressed_size(raw.len(), opt_1t.block_size).unwrap());

    let parsed = Frame::parse(&f_1t, raw.len()).unwrap();
    assert_eq!(parsed.version(), 5);
    assert!(!parsed.dict().is_empty(), "expected trained prefix dictionary");
    assert!(
        parsed.blocks().any(|b| b.kind == BLOCK_DICT_LZ),
        "expected BLOCK_DICT_LZ blocks on mixed free-text log corpus"
    );

    // Caller-buffer zero-alloc compress_into & decompress_into parity:
    let mut cbuf = vec![0u8; max_compressed_size(raw.len(), opt_1t.block_size).unwrap()];
    let clen = compress_into(&raw, &mut cbuf, &opt_4t).unwrap();
    assert_eq!(&cbuf[..clen], f_1t.as_slice());

    let mut dbuf = vec![0u8; parsed.raw_size()];
    decompress_into(&f_1t, &mut dbuf, &dec_4t).unwrap();
    assert_eq!(dbuf, raw);
    assert_eq!(decompress(&f_1t, &dec_1t).unwrap(), raw);

    // ValidatedFrame command/dictionary validation + decode_into:
    let validated = Frame::parse(&f_1t, raw.len()).unwrap().into_validated().unwrap();
    let mut vbuf = vec![0u8; raw.len()];
    validated.decode_into(&mut vbuf, &dec_4t).unwrap();
    assert_eq!(vbuf, raw);

    // O(1) random-access block extraction & zero-output-alloc verify_frame:
    assert_eq!(verify_frame(&f_1t, &dec_1t).unwrap(), raw.len() as u64);
    assert_eq!(verify_frame(&f_1t, &dec_4t).unwrap(), raw.len() as u64);
    let mut rebuilt = Vec::with_capacity(raw.len());
    for idx in 0..parsed.block_count() {
        let info = parsed.blocks().nth(idx).unwrap();
        let mut blk = vec![0u8; info.raw_size];
        let n = extract_block_into(&f_1t, idx, &mut blk, &dec_1t).unwrap();
        assert_eq!(n, info.raw_size);
        rebuilt.extend_from_slice(&blk);
    }
    assert_eq!(rebuilt, raw);

    // Streaming roundtrip + query/filter_frame passthrough over v5 DICT_LZ frames:
    let mut stream_enc = Vec::new();
    compress_stream(&mut Cursor::new(&raw), &mut stream_enc, &opt_4t).unwrap();
    let mut stream_dec = Vec::new();
    decompress_stream(&mut Cursor::new(&stream_enc), &mut stream_dec, &dec_4t).unwrap();
    assert_eq!(stream_dec, raw);

    assert_eq!(frame_row_count(&f_1t).unwrap(), 2600);
    let pred_500 = PredExpr::RowContains(b"\" 500 ");
    let expected_500 = raw
        .split_inclusive(|&b| b == b'\n')
        .filter(|ln| ln.windows(6).any(|w| w == b"\" 500 "))
        .count() as u64;
    assert_eq!(count_where(&f_1t, &pred_500, &dec_4t).unwrap(), expected_500);
    assert_eq!(count_where(&f_1t, &pred_500, &dec_1t).unwrap(), expected_500);

    // All-match filter_frame_where preserves untouched DICT_LZ blocks + v5 dictionary:
    let kept = filter_frame_where(&f_1t, &PredExpr::True, &opt_4t, &dec_4t).unwrap();
    assert_eq!(decompress(&kept, &dec_4t).unwrap(), raw);
}

#[test]
fn v5_record_aligned_zero_splits_and_differential_sql_ops_oracle() {
    let raw = build_tabular_ndjson(3200);
    assert!(raw.len() > 16384 * 15);

    let base_opt = Options {
        block_size: 16384,
        threads: 4,
        checksum: true,
        record_aligned: false,
        ..Options::default()
    };
    let aln_opt = Options {
        record_aligned: true,
        ..base_opt.clone()
    };
    let dec = DecodeOptions {
        threads: 4,
        max_output: raw.len() + 4096,
        verify_checksum: true,
        ..DecodeOptions::default()
    };

    let f_fixed = compress(&raw, &base_opt).unwrap();
    let f_aln = compress(&raw, &aln_opt).unwrap();
    assert_eq!(
        f_aln,
        compress(&raw, &Options { threads: 1, ..aln_opt.clone() }).unwrap(),
        "record_aligned encode must be thread-deterministic"
    );

    let parsed_aln = Frame::parse(&f_aln, raw.len()).unwrap();
    assert_eq!(parsed_aln.version(), 5);
    assert!(parsed_aln.blocks().all(|b| b.kind == BLOCK_COLUMN));

    // Every single block in `f_aln` must end cleanly on a `\n` record boundary!
    for idx in 0..parsed_aln.block_count() {
        let blk = parsed_aln.decode_block(idx, &dec).unwrap();
        assert_eq!(
            blk.last().copied(),
            Some(b'\n'),
            "block {idx} in record-aligned frame did not end at newline"
        );
    }
    assert_eq!(decompress(&f_aln, &dec).unwrap(), raw);
    assert_eq!(verify_frame(&f_aln, &dec).unwrap(), raw.len() as u64);

    // Differential oracle: every query, SQL query, and frame mutation on `f_aln` must equal `f_fixed`!
    assert_eq!(frame_row_count(&f_aln).unwrap(), frame_row_count(&f_fixed).unwrap());

    let expr = PredExpr::All(&[
        PredExpr::col(ColPred::json_num("status", NumCmp::Eq(500))),
        PredExpr::col(ColPred::json_str("service", StrCmp::Eq(b"auth-api"))),
    ]);
    assert_eq!(
        count_where(&f_aln, &expr, &dec).unwrap(),
        count_where(&f_fixed, &expr, &dec).unwrap()
    );
    let m_aln: Vec<Vec<u8>> = filter_where(&f_aln, &expr, 50, &dec)
        .unwrap()
        .into_iter()
        .map(|r| r.bytes)
        .collect();
    let m_fix: Vec<Vec<u8>> = filter_where(&f_fixed, &expr, 50, &dec)
        .unwrap()
        .into_iter()
        .map(|r| r.bytes)
        .collect();
    assert_eq!(m_aln, m_fix);

    let cols = [
        ColTarget::JsonKey("ts"),
        ColTarget::JsonKey("service"),
        ColTarget::JsonKey("bytes"),
        ColTarget::JsonKey("actor.login"),
    ];
    assert_eq!(
        select_columns(&f_aln, Some(&expr), &cols, 100, &dec).unwrap(),
        select_columns(&f_fixed, Some(&expr), &cols, 100, &dec).unwrap()
    );
    assert_eq!(
        aggregate_column(&f_aln, Some(&expr), ColTarget::JsonKey("bytes"), &dec).unwrap(),
        aggregate_column(&f_fixed, Some(&expr), ColTarget::JsonKey("bytes"), &dec).unwrap()
    );
    assert_eq!(
        group_by(
            &f_aln,
            None,
            ColTarget::JsonKey("service"),
            Some(ColTarget::JsonKey("delta")),
            &dec
        )
        .unwrap(),
        group_by(
            &f_fixed,
            None,
            ColTarget::JsonKey("service"),
            Some(ColTarget::JsonKey("delta")),
            &dec
        )
        .unwrap()
    );

    let tk_aln: Vec<Vec<u8>> = top_k_by(
        &f_aln,
        None,
        ColTarget::JsonKey("bytes"),
        SortDir::Desc,
        15,
        &dec,
    )
    .unwrap()
    .into_iter()
    .map(|r| r.bytes)
    .collect();
    let tk_fix: Vec<Vec<u8>> = top_k_by(
        &f_fixed,
        None,
        ColTarget::JsonKey("bytes"),
        SortDir::Desc,
        15,
        &dec,
    )
    .unwrap()
    .into_iter()
    .map(|r| r.bytes)
    .collect();
    assert_eq!(tk_aln, tk_fix);

    let sql_q = SqlQuery::parse(
        "SELECT service, COUNT(*), SUM(bytes), SUM(delta) FROM frame WHERE status = 500 GROUP BY service ORDER BY SUM(bytes) DESC",
    )
    .unwrap();
    let res_aln = execute_sql(&f_aln, &sql_q, SqlFormat::Json, false, &dec).unwrap();
    let res_fix = execute_sql(&f_fixed, &sql_q, SqlFormat::Json, false, &dec).unwrap();
    assert_eq!(res_aln, res_fix);

    // Multi-frame record-aligned streaming SQL parity:
    let mut stm_aln = Vec::new();
    compress_stream(
        &mut Cursor::new(&raw),
        &mut stm_aln,
        &Options {
            block_size: 8192,
            record_aligned: true,
            threads: 2,
            ..Options::default()
        },
    )
    .unwrap();
    let mut res_stm = Vec::new();
    execute_sql_stream(
        Cursor::new(&stm_aln),
        &mut res_stm,
        &sql_q,
        SqlFormat::Json,
        false,
        &dec,
    )
    .unwrap();
    assert_eq!(res_stm, res_fix);

    // Frame-to-frame splicing/rewrites over record-aligned v5 frames:
    let proj_aln = project_frame(&f_aln, None, &cols[..3], &aln_opt, &dec).unwrap();
    let proj_fix = project_frame(&f_fixed, None, &cols[..3], &base_opt, &dec).unwrap();
    assert_eq!(decompress(&proj_aln, &dec).unwrap(), decompress(&proj_fix, &dec).unwrap());

    let flt_aln = filter_frame_where(&f_aln, &expr, &aln_opt, &dec).unwrap();
    let flt_fix = filter_frame_where(&f_fixed, &expr, &base_opt, &dec).unwrap();
    assert_eq!(decompress(&flt_aln, &dec).unwrap(), decompress(&flt_fix, &dec).unwrap());

    let upd_aln = update_column(
        &f_aln,
        &ColPred::json_num("status", NumCmp::Eq(500)),
        ColTarget::JsonKey("service"),
        ColOp::SetStr(b"alert"),
        &aln_opt,
        &dec,
    )
    .unwrap();
    let upd_fix = update_column(
        &f_fixed,
        &ColPred::json_num("status", NumCmp::Eq(500)),
        ColTarget::JsonKey("service"),
        ColOp::SetStr(b"alert"),
        &base_opt,
        &dec,
    )
    .unwrap();
    assert_eq!(decompress(&upd_aln, &dec).unwrap(), decompress(&upd_fix, &dec).unwrap());

    let add_ten = |v: u64| v + 10;
    let map_aln = map_column(
        &f_aln,
        ColTarget::JsonKey("bytes"),
        ColOp::MapU64(&add_ten),
        &aln_opt,
        &dec,
    )
    .unwrap();
    let map_fix = map_column(
        &f_fixed,
        ColTarget::JsonKey("bytes"),
        ColOp::MapU64(&add_ten),
        &base_opt,
        &dec,
    )
    .unwrap();
    assert_eq!(decompress(&map_aln, &dec).unwrap(), decompress(&map_fix, &dec).unwrap());
}

#[test]
fn v5_combined_dict_and_record_aligned_plus_malformed_rejection() {
    let raw = build_free_text_log_corpus(2200);
    let opt = Options {
        block_size: 65536,
        threads: 4,
        checksum: true,
        record_aligned: true,
        ..Options::default()
    };
    let dec = DecodeOptions {
        threads: 2,
        max_output: raw.len() + 1024,
        verify_checksum: true,
        ..DecodeOptions::default()
    };
    let frame = compress(&raw, &opt).unwrap();
    let parsed = Frame::parse(&frame, raw.len()).unwrap();
    assert_eq!(parsed.version(), 5);
    assert!(!parsed.dict().is_empty());
    assert_eq!(decompress(&frame, &dec).unwrap(), raw);

    // 1. Flipping `FLAG_FRAME_DICT` off while `dw/dr != 0` or `kind == 3` blocks exist must fail:
    let mut no_dict_flag = frame.clone();
    no_dict_flag[6] &= !0x02;
    assert!(Frame::parse(&no_dict_flag, raw.len()).is_err());

    // 2. Corrupting `var_raw` block size entry so sum != raw_size must fail:
    let esize = 8usize;
    let vr_pos = 32 + parsed.block_count() * esize;
    let mut bad_vr = frame.clone();
    bad_vr[vr_pos] ^= 1;
    assert!(Frame::parse(&bad_vr, raw.len()).is_err());

    // 3. Corrupting wire dictionary header/payload must fail without panicking:
    let d_pos = vr_pos + parsed.block_count() * 4;
    for off in [d_pos, d_pos + 1, d_pos + 2, d_pos + 7] {
        if off < frame.len() {
            let mut bad_dict = frame.clone();
            bad_dict[off] ^= 0xff;
            assert!(decompress(&bad_dict, &dec).is_err());
        }
    }

    // 4. Bitflip fuzz across a `DICT_LZ` payload must never panic and must fail checksum/decode:
    let dlz_opt = parsed.blocks().find(|b| b.kind == BLOCK_DICT_LZ);
    if let Some(dlz_blk) = dlz_opt {
        let step = (dlz_blk.payload_size / 64).max(1);
        for p in (0..dlz_blk.payload_size).step_by(step) {
            let mut mutated = frame.clone();
            mutated[dlz_blk.payload_offset + p] ^= 0x55;
            assert!(decompress(&mutated, &dec).is_err());
        }
    }

    // 5. FFI inspection & zero-alloc decode on `VERSION_V5` (`FLAG_FRAME_DICT` + `FLAG_VAR_RAW`):
    unsafe {
        assert_eq!(
            vxg::ffi::vxg_frame_raw_size(frame.as_ptr(), frame.len()),
            raw.len() as isize
        );
        assert_eq!(
            vxg::ffi::vxg_frame_block_count(frame.as_ptr(), frame.len()),
            parsed.block_count() as isize
        );
        for idx in 0..parsed.block_count() {
            let expected_rn = parsed.blocks().nth(idx).unwrap().raw_size;
            assert_eq!(
                vxg::ffi::vxg_frame_block_raw_size(frame.as_ptr(), frame.len(), idx),
                expected_rn as isize
            );
            let mut blk_out = vec![0u8; expected_rn];
            let got = vxg::ffi::vxg_extract_block_into(
                frame.as_ptr(),
                frame.len(),
                idx,
                blk_out.as_mut_ptr(),
                blk_out.len(),
                1,
            );
            assert_eq!(got, expected_rn as isize);
        }
        let mut full_out = vec![0u8; raw.len()];
        let got_full = vxg::ffi::vxg_decompress_into(
            frame.as_ptr(),
            frame.len(),
            full_out.as_mut_ptr(),
            full_out.len(),
            2,
            1,
        );
        assert_eq!(got_full, raw.len() as isize);
        assert_eq!(full_out, raw);
    }

    // 6. CSV header stripping (`strip_delim_header_bytes`) over multi-block `VERSION_V5` (`--record-aligned`) CSV:
    let mut csv_bytes = b"id,service,payment_type,total_cents\n".to_vec();
    for i in 0..2500u64 {
        let svc = ["billing", "gateway", "auth", "search"][(i as usize) & 3];
        csv_bytes.extend_from_slice(format!("{i},{svc},{},{}\n", (i % 4) + 1, 1250 + i * 5).as_bytes());
    }
    let csv_fix = compress(&csv_bytes, &Options { block_size: 4096, ..Options::default() }).unwrap();
    let csv_aln = compress(
        &csv_bytes,
        &Options {
            block_size: 4096,
            record_aligned: true,
            ..Options::default()
        },
    )
    .unwrap();
    let csv_dec = DecodeOptions {
        threads: 2,
        max_output: csv_bytes.len() + 1024,
        verify_checksum: true,
        ..DecodeOptions::default()
    };
    let q_csv = SqlQuery::parse(
        "SELECT service, COUNT(*), SUM(total_cents) FROM csv WHERE payment_type = 2 GROUP BY service ORDER BY service ASC",
    )
    .unwrap();
    assert_eq!(
        execute_sql(&csv_aln, &q_csv, SqlFormat::Json, false, &csv_dec).unwrap(),
        execute_sql(&csv_fix, &q_csv, SqlFormat::Json, false, &csv_dec).unwrap()
    );
    let q_sel = SqlQuery::parse(
        "SELECT id, service, total_cents FROM csv WHERE payment_type = 2 LIMIT 25",
    )
    .unwrap();
    assert_eq!(
        execute_sql(&csv_aln, &q_sel, SqlFormat::Csv, true, &csv_dec).unwrap(),
        execute_sql(&csv_fix, &q_sel, SqlFormat::Csv, true, &csv_dec).unwrap()
    );
}

#[test]
fn v5_adaptive_large_dict_coalesces_macroscopic_repeats_under_five_percent() {
    let mut seed = Vec::with_capacity(205_000);
    for i in 0..2000usize {
        seed.extend_from_slice(
            format!(
                "pub fn kernel_dispatch_{i:04}(ctx: &mut ExecCtx, row: usize) -> Result<u64> {{ ctx.step({i}, row ^ 0x{:04x}) }}\n",
                (i.wrapping_mul(40503)) & 0xffff
            )
            .as_bytes(),
        );
    }
    assert!(seed.len() > 180_000 && seed.len() < 240_000);
    let target_len = 8 * 1024 * 1024;
    let mut raw = Vec::with_capacity(target_len + seed.len());
    while raw.len() < target_len {
        raw.extend_from_slice(&seed);
    }
    raw.truncate(target_len);

    let opt_1t = Options {
        block_size: 65536,
        threads: 1,
        checksum: true,
        ..Options::default()
    };
    let opt_4t = Options {
        threads: 4,
        ..opt_1t.clone()
    };
    let dec_4t = DecodeOptions {
        threads: 4,
        max_output: raw.len() + 1024,
        verify_checksum: true,
        ..DecodeOptions::default()
    };

    let f_1t = compress(&raw, &opt_1t).unwrap();
    let f_4t = compress(&raw, &opt_4t).unwrap();
    assert_eq!(f_1t, f_4t);
    assert!(
        f_1t.len() * 20 < raw.len(),
        "expected < 0.05 ratio on 190 KiB periodic source corpus, got {} / {}",
        f_1t.len(),
        raw.len()
    );
    let parsed = Frame::parse(&f_1t, raw.len()).unwrap();
    assert!(parsed.dict().len() > 32_768 && parsed.dict().len() <= 262_144);
    assert_eq!(decompress(&f_1t, &dec_4t).unwrap(), raw);

    let mut blk = vec![0u8; 65536];
    let mid_idx = parsed.block_count() / 2;
    let got = extract_block_into(&f_1t, mid_idx, &mut blk, &dec_4t).unwrap();
    assert_eq!(&blk[..got], &raw[mid_idx * 65536..mid_idx * 65536 + got]);
}
