use std::ffi::CString;
use std::io::Write;

use vxg::ffi::{
    vxg_agg_stats_t, vxg_aggregate_col, vxg_compress_alloc, vxg_compress_file, vxg_count_where_str,
    vxg_decompress_file, vxg_free, vxg_project_frame_alloc, vxg_select_alloc, vxg_verify_file,
    vxg_verify_frame, VXG_ERR_CHECKSUM_MISMATCH, VXG_ERR_INVALID_OPTION, VXG_ERR_NULL_PTR, VXG_OK,
};
use vxg::ops::{extract_line_sort_key, line_matches_expr, project_line};
use vxg::{
    compress, decompress, parse_col_list, parse_col_target, select_top_k_by,
    select_top_k_by_stream, top_k_by, top_k_by_stream, ColTarget, DecodeOptions, Options,
    ParsedWhere, PredExpr, SortDir, SortKey, StreamEncoder,
};

fn ref_cmp_keys(
    a: &SortKey,
    a_row: usize,
    b: &SortKey,
    b_row: usize,
    dir: SortDir,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (SortKey::Missing, SortKey::Missing) => a_row.cmp(&b_row),
        (SortKey::Missing, _) => Ordering::Greater,
        (_, SortKey::Missing) => Ordering::Less,
        _ => {
            let base = match (a, b) {
                (SortKey::Num(x), SortKey::Num(y)) => x.cmp(y),
                (SortKey::Str(x), SortKey::Str(y)) => x.cmp(y),
                (SortKey::Num(_), SortKey::Str(_)) => Ordering::Less,
                (SortKey::Str(_), SortKey::Num(_)) => Ordering::Greater,
                _ => unreachable!(),
            };
            if base == Ordering::Equal {
                a_row.cmp(&b_row)
            } else if dir == SortDir::Asc {
                base
            } else {
                base.reverse()
            }
        }
    }
}

fn oracle_top_k(
    raw: &[u8],
    where_expr: Option<&PredExpr<'_>>,
    project_cols: Option<&[ColTarget<'_>]>,
    order_col: ColTarget<'_>,
    dir: SortDir,
    limit: usize,
) -> Vec<(usize, Vec<u8>)> {
    if limit == 0 || raw.is_empty() {
        return Vec::new();
    }
    let mut cands: Vec<(SortKey, usize, Vec<u8>)> = Vec::new();
    for (g_row, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if let Some(expr) = where_expr {
            if !line_matches_expr(line, expr) {
                continue;
            }
        }
        let out_bytes = if let Some(cols) = project_cols {
            let mut p = Vec::new();
            if !project_line(line, cols, &mut p) {
                continue;
            }
            p
        } else {
            line.to_vec()
        };
        let key = extract_line_sort_key(line, order_col);
        cands.push((key, g_row, out_bytes));
    }
    cands.sort_by(|(ka, ra, _), (kb, rb, _)| ref_cmp_keys(ka, *ra, kb, *rb, dir));
    cands.truncate(limit);
    cands.into_iter().map(|(_, r, b)| (r, b)).collect()
}

#[test]
fn top_k_differential_ndjson_models_nulls_ties_threads_and_streams() {
    let services = ["auth-gw", "billing", "edge-cdn", "search-idx", "ml-ranker"];
    let mut raw = Vec::new();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_u64 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    for i in 0..1800usize {
        let r = next_u64();
        let svc = services[(r as usize) % services.len()];
        let status = if i % 7 == 0 {
            503
        } else if i % 5 == 0 {
            404
        } else {
            200
        };
        // Intentional duplicates (0..250) + occasional large spikes to exercise ties & zone maps
        let lat = if i % 97 == 0 {
            10_000 + (r % 90_000)
        } else {
            r % 250
        };
        let sec = i % 60;
        let note = if i % 11 == 0 {
            format!("warn\\\"q\\\"-{i}")
        } else {
            format!("ok-{i}")
        };
        match i % 9 {
            // Standard template 0: numeric opt_val
            0..=5 => {
                raw.extend_from_slice(
                    format!(
                        "{{\"ts\":\"2026-03-01T10:00:{sec:02}Z\",\"seq\":{i},\"service\":\"{svc}\",\"status\":{status},\"latency_ms\":{lat},\"note\":\"{note}\",\"opt\":{}}}\n",
                        r % 40
                    )
                    .as_bytes(),
                );
            }
            // Template 1: JSON null opt_val
            6 => {
                raw.extend_from_slice(
                    format!(
                        "{{\"ts\":\"2026-03-01T10:00:{sec:02}Z\",\"seq\":{i},\"service\":\"{svc}\",\"status\":{status},\"latency_ms\":{lat},\"note\":\"{note}\",\"opt\":null}}\n"
                    )
                    .as_bytes(),
                );
            }
            // Template 2: opt_val key absent altogether
            7 => {
                raw.extend_from_slice(
                    format!(
                        "{{\"ts\":\"2026-03-01T10:00:{sec:02}Z\",\"seq\":{i},\"service\":\"{svc}\",\"status\":{status},\"latency_ms\":{lat},\"note\":\"{note}\"}}\n"
                    )
                    .as_bytes(),
                );
            }
            // Template 3 / outlier: string opt_val
            _ => {
                raw.extend_from_slice(
                    format!(
                        "{{\"ts\":\"2026-03-01T10:00:{sec:02}Z\",\"seq\":{i},\"service\":\"{svc}\",\"status\":{status},\"latency_ms\":{lat},\"note\":\"{note}\",\"opt\":\"str-{}\"}}\n",
                        r % 15
                    )
                    .as_bytes(),
                );
            }
        }
    }

    let single_frame = compress(
        &raw,
        &Options {
            block_size: 4096,
            threads: 2,
            ..Options::default()
        },
    )
    .unwrap();

    // Multi-frame stream with small frames & unaligned writes to split lines across frames!
    let mut multi_stream = Vec::new();
    {
        let mut enc = StreamEncoder::with_chunk_size(
            &mut multi_stream,
            Options {
                block_size: 4096,
                threads: 2,
                ..Options::default()
            },
            12 * 1024,
        )
        .unwrap();
        for chunk in raw.chunks(5003) {
            enc.write_all(chunk).unwrap();
        }
        enc.finish().unwrap();
    }
    assert_eq!(
        decompress(&multi_stream, &DecodeOptions::default()).unwrap(),
        raw
    );

    let pw = ParsedWhere::parse(
        "status>=400 && (service==\"auth-gw\" || service==\"billing\" || latency_ms>=50000)",
    )
    .unwrap();
    let proj_cols = parse_col_list("ts,service,seq").unwrap(); // Notice `latency_ms` & `opt` are NOT in `proj_cols`!

    let order_targets = [
        parse_col_target("latency_ms").unwrap(),
        parse_col_target("seq").unwrap(),
        parse_col_target("service").unwrap(),
        parse_col_target("note").unwrap(),
        parse_col_target("opt").unwrap(),
        parse_col_target("missing_col").unwrap(),
    ];

    for &order_col in &order_targets {
        for &dir in &[SortDir::Asc, SortDir::Desc] {
            for &k in &[0usize, 1, 7, 30, usize::MAX] {
                for use_where in [false, true] {
                    pw.with_expr(|w_expr| {
                        let w_opt = use_where.then_some(w_expr);
                        let expected_full = oracle_top_k(&raw, w_opt, None, order_col, dir, k);
                        let expected_proj =
                            oracle_top_k(&raw, w_opt, Some(&proj_cols), order_col, dir, k);
                        let expected_proj_bytes: Vec<u8> = expected_proj
                            .iter()
                            .flat_map(|(_, b)| b.iter().copied())
                            .collect();

                        for &threads in &[1usize, 4] {
                            let dec = DecodeOptions {
                                threads,
                                max_output: usize::MAX,
                                ..DecodeOptions::default()
                            };
                            let got = top_k_by(&single_frame, w_opt, order_col, dir, k, &dec).unwrap();
                            let got_pairs: Vec<(usize, Vec<u8>)> = got
                                .into_iter()
                                .map(|m| (m.global_row_index, m.bytes))
                                .collect();
                            assert_eq!(
                                got_pairs, expected_full,
                                "top_k_by mismatch col={order_col:?} dir={dir:?} k={k} where={use_where} t={threads}"
                            );

                            let got_proj = select_top_k_by(
                                &single_frame,
                                w_opt,
                                &proj_cols,
                                order_col,
                                dir,
                                k,
                                &dec,
                            )
                            .unwrap();
                            assert_eq!(
                                got_proj, expected_proj_bytes,
                                "select_top_k_by mismatch col={order_col:?} dir={dir:?} k={k} where={use_where} t={threads}"
                            );

                            if k <= 30 {
                                let stm = top_k_by_stream(
                                    multi_stream.as_slice(),
                                    w_opt,
                                    order_col,
                                    dir,
                                    k,
                                    &dec,
                                )
                                .unwrap();
                                let stm_pairs: Vec<(usize, Vec<u8>)> = stm
                                    .into_iter()
                                    .map(|m| (m.global_row_index, m.bytes))
                                    .collect();
                                assert_eq!(stm_pairs, expected_full);

                                let mut stm_proj = Vec::new();
                                let n_emitted = select_top_k_by_stream(
                                    multi_stream.as_slice(),
                                    &mut stm_proj,
                                    w_opt,
                                    &proj_cols,
                                    order_col,
                                    dir,
                                    k,
                                    &dec,
                                )
                                .unwrap();
                                assert_eq!(n_emitted as usize, expected_proj.len());
                                assert_eq!(stm_proj, expected_proj_bytes);
                            }
                        }
                    });
                }
            }
        }
    }
}

#[test]
fn top_k_differential_delimited_csv_records() {
    let regions = ["eu-west", "us-east", "ap-south", "eu-central"];
    let mut csv = Vec::new();
    for i in 0..1800usize {
        let reg = regions[i % regions.len()];
        let score = (i * 37) % 113;
        let label = if i % 13 == 0 {
            format!("\"tag,with,{i}\"")
        } else {
            format!("plain-{i}")
        };
        csv.extend_from_slice(format!("{i},{score},{reg},{label}\n").as_bytes());
    }
    // Append an unterminated final CSV line to verify EOF tail handling!
    csv.extend_from_slice(b"1800,999,eu-west,final-unterminated");

    let frame = compress(
        &csv,
        &Options {
            block_size: 4096,
            threads: 2,
            ..Options::default()
        },
    )
    .unwrap();
    let pw = ParsedWhere::parse("csv:2==eu-west || csv:1>=100").unwrap();
    let proj_cols = parse_col_list("csv:0,csv:2,csv:3").unwrap();

    for col_str in ["csv:1", "csv:2", "csv:3", "csv:9"] {
        let order_col = parse_col_target(col_str).unwrap();
        for &dir in &[SortDir::Asc, SortDir::Desc] {
            pw.with_expr(|w| {
                let exp_full = oracle_top_k(&csv, Some(w), None, order_col, dir, 12);
                let exp_proj: Vec<u8> =
                    oracle_top_k(&csv, Some(w), Some(&proj_cols), order_col, dir, 12)
                        .into_iter()
                        .flat_map(|(_, b)| b)
                        .collect();
                for &threads in &[1usize, 3] {
                    let dec = DecodeOptions {
                        threads,
                        ..DecodeOptions::default()
                    };
                    let got: Vec<(usize, Vec<u8>)> =
                        top_k_by(&frame, Some(w), order_col, dir, 12, &dec)
                            .unwrap()
                            .into_iter()
                            .map(|m| (m.global_row_index, m.bytes))
                            .collect();
                    assert_eq!(got, exp_full);

                    let got_p =
                        select_top_k_by(&frame, Some(w), &proj_cols, order_col, dir, 12, &dec)
                            .unwrap();
                    assert_eq!(got_p, exp_proj);
                }
            });
        }
    }
}

#[test]
fn ffi_analytics_verify_file_and_topk_abi_end_to_end() {
    let dir = std::env::temp_dir().join(format!("vxg_ffi_stage6_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let raw_file = dir.join("events.ndjson");
    let vxg_file = dir.join("events.vxg");
    let dec_file = dir.join("events.out.ndjson");

    let mut raw = Vec::new();
    for i in 0..1000u64 {
        let svc = if i % 2 == 0 { "auth" } else { "pay" };
        let status = if i % 10 == 0 { 500 } else { 200 };
        let lat = 10 + i * 3;
        raw.extend_from_slice(
            format!("{{\"id\":{i},\"svc\":\"{svc}\",\"status\":{status},\"lat\":{lat}}}\n")
                .as_bytes(),
        );
    }
    std::fs::write(&raw_file, &raw).unwrap();

    let c_in = CString::new(raw_file.to_str().unwrap()).unwrap();
    let c_vxg = CString::new(vxg_file.to_str().unwrap()).unwrap();
    let c_dec = CString::new(dec_file.to_str().unwrap()).unwrap();

    unsafe {
        assert_eq!(
            vxg_compress_file(std::ptr::null(), c_vxg.as_ptr(), 2, 4096, 1, 1),
            VXG_ERR_NULL_PTR
        );
        let enc_len = vxg_compress_file(c_in.as_ptr(), c_vxg.as_ptr(), 2, 4096, 1, 0);
        assert!(enc_len > 32);
        // Existing file without force=1 must return an error:
        assert!(vxg_compress_file(c_in.as_ptr(), c_vxg.as_ptr(), 2, 4096, 1, 0) < 0);

        let ver_len = vxg_verify_file(c_vxg.as_ptr(), 2, 1);
        assert_eq!(ver_len as usize, raw.len());

        let dec_len = vxg_decompress_file(c_vxg.as_ptr(), c_dec.as_ptr(), 2, 1, 1);
        assert_eq!(dec_len as usize, raw.len());
        assert_eq!(std::fs::read(&dec_file).unwrap(), raw);

        // In-memory verify + corruption detection:
        let mut c_frame_len = 0usize;
        let f_ptr = vxg_compress_alloc(raw.as_ptr(), raw.len(), 2, 4096, 1, &mut c_frame_len);
        assert!(!f_ptr.is_null());
        assert_eq!(
            vxg_verify_frame(f_ptr, c_frame_len, 2, 1) as usize,
            raw.len()
        );

        let mut corrupt = std::slice::from_raw_parts(f_ptr, c_frame_len).to_vec();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0x55;
        let bad_rc = vxg_verify_frame(corrupt.as_ptr(), corrupt.len(), 1, 1);
        assert!(bad_rc < 0 && (bad_rc == VXG_ERR_CHECKSUM_MISMATCH || bad_rc == -4));

        // Count & Aggregate over C ABI:
        assert_eq!(
            vxg_count_where_str(f_ptr, c_frame_len, std::ptr::null(), 2),
            1000
        );
        let w_500 = CString::new("status==500 && svc==\"auth\"").unwrap();
        assert_eq!(
            vxg_count_where_str(f_ptr, c_frame_len, w_500.as_ptr(), 2),
            100
        );
        let w_bad = CString::new("((status==500").unwrap();
        assert_eq!(
            vxg_count_where_str(f_ptr, c_frame_len, w_bad.as_ptr(), 1),
            VXG_ERR_INVALID_OPTION
        );

        let col_lat = CString::new("lat").unwrap();
        let mut st = vxg_agg_stats_t::default();
        assert_eq!(
            vxg_aggregate_col(
                f_ptr,
                c_frame_len,
                col_lat.as_ptr(),
                w_500.as_ptr(),
                2,
                &mut st
            ),
            VXG_OK
        );
        // Matching rows: i = 0, 10, 20, ..., 990 (100 rows); lat = 10 + 3*i
        // min = 10 (i=0), max = 10 + 2970 = 2980 (i=990), sum = 100*10 + 3*(10*99*100/2) = 1000 + 148500 = 149500
        assert_eq!(st.row_count, 100);
        assert_eq!(st.num_count, 100);
        assert_eq!(st.sum_lo, 149_500);
        assert_eq!(st.sum_hi, 0);
        assert_eq!(st.min_val, 10);
        assert_eq!(st.max_val, 2980);
        assert_eq!(st.has_minmax, 1);

        // SELECT + WHERE + ORDER BY DESC Top-K over C ABI:
        let cols_sel = CString::new("id,svc").unwrap();
        let mut sel_len = 0usize;
        let sel_ptr = vxg_select_alloc(
            f_ptr,
            c_frame_len,
            cols_sel.as_ptr(),
            w_500.as_ptr(),
            col_lat.as_ptr(),
            1, // DESC
            3, // LIMIT 3
            2,
            &mut sel_len,
        );
        assert!(!sel_ptr.is_null());
        let sel_txt = std::str::from_utf8(std::slice::from_raw_parts(sel_ptr, sel_len)).unwrap();
        assert_eq!(
            sel_txt,
            "{\"id\":990,\"svc\":\"auth\"}\n{\"id\":980,\"svc\":\"auth\"}\n{\"id\":970,\"svc\":\"auth\"}\n"
        );
        vxg_free(sel_ptr, sel_len);

        // Zero-match SELECT returns non-null pointer with out_len == 0:
        let w_none = CString::new("status==999").unwrap();
        let mut zero_len = 99usize;
        let zero_ptr = vxg_select_alloc(
            f_ptr,
            c_frame_len,
            cols_sel.as_ptr(),
            w_none.as_ptr(),
            std::ptr::null(),
            0,
            10,
            1,
            &mut zero_len,
        );
        assert!(!zero_ptr.is_null());
        assert_eq!(zero_len, 0);
        vxg_free(zero_ptr, zero_len);

        // Compressed-domain `vxg_project_frame_alloc` -> verify -> query:
        let cols_proj = CString::new("id,status,lat").unwrap();
        let mut pf_len = 0usize;
        let pf_ptr = vxg_project_frame_alloc(
            f_ptr,
            c_frame_len,
            cols_proj.as_ptr(),
            std::ptr::null(),
            2,
            &mut pf_len,
        );
        assert!(!pf_ptr.is_null() && pf_len > 32);
        assert!(vxg_verify_frame(pf_ptr, pf_len, 2, 1) > 0);
        assert_eq!(
            vxg_count_where_str(pf_ptr, pf_len, std::ptr::null(), 1),
            1000
        );
        vxg_free(pf_ptr, pf_len);
        vxg_free(f_ptr, c_frame_len);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
