//! Integration, differential parity, ABI safety, and IPC roundtrip tests for VXG's
//! zero-dependency Apache Arrow C Data Interface (`ArrowSchema`, `ArrowArray`,
//! `ArrowArrayStream`), compressed-domain `WHERE` + column projection pushdown, and IPC `V5`.

use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::process::Command;

use vxg::ffi::{
    vxg_arrow_export, vxg_arrow_export_stream, vxg_arrow_ipc_alloc, vxg_arrow_query,
    vxg_arrow_query_stream, vxg_free, ArrowArray, ArrowArrayStream, ArrowSchema,
    VXG_ERR_CORRUPT_DATA, VXG_ERR_INVALID_OPTION, VXG_ERR_NULL_PTR, VXG_OK,
};
use vxg::{
    compress, execute_sql_table, export_arrow_batch, export_arrow_batch_where,
    export_arrow_batches_where, export_arrow_ipc, export_arrow_ipc_where, parse_arrow_where,
    parse_col_list, ArrowBatch, ArrowType, DecodeOptions, Options, SqlQuery,
};

fn build_mixed_ndjson(rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * 96);
    for i in 0..rows {
        let pay = 1 + (i % 4);
        let tip = if i % 5 == 0 {
            "null".to_owned()
        } else {
            ((i * 15) % 600).to_string()
        };
        let delta = (i as i64 % 50) - 25;
        let fare = format!("{}.{:02}", 8 + (i % 30), (i * 7) % 100);
        let ok = if i % 2 == 0 { "true" } else { "false" };
        let zone = match i % 4 {
            0 => "midtown",
            1 => "jfk",
            2 => "brooklyn",
            _ => "q\\\"ueens",
        };
        out.extend_from_slice(
            format!(
                "{{\"id\":{i},\"payment_type\":{pay},\"tip_cents\":{tip},\"delta\":{delta},\"fare\":{fare},\"ok\":{ok},\"zone\":\"{zone}\"}}\n"
            )
            .as_bytes(),
        );
    }
    out
}

fn build_cumulative_csv(rows: usize) -> Vec<u8> {
    let mut csv = Vec::from(&b"country,day,confirmed,net_change,rate,note\n"[..]);
    let mut cum = 0u64;
    for i in 0..rows {
        if i % 90 == 0 {
            cum = (i as u64) * 100;
        } else {
            cum += (i as u64 % 11) + 1;
        }
        let country = match (i / 90) % 3 {
            0 => "TR",
            1 => "DE",
            _ => "JP",
        };
        let net = (i as i64 % 41) - 20;
        let rate = format!("{}.{:02}", (i % 17) as i64 - 5, (i * 13) % 100);
        let note = if i % 7 == 0 {
            ""
        } else if i % 11 == 0 {
            "\"0042\""
        } else {
            "steady"
        };
        csv.extend_from_slice(
            format!("{country},{i},{cum},{net},{rate},{note}\n").as_bytes(),
        );
    }
    csv
}

#[test]
fn arrow_batch_and_ipc_roundtrip_ndjson_and_csv() {
    let ndjson = build_mixed_ndjson(320);
    let frame = compress(&ndjson, &Options::default()).unwrap();
    let dopts = DecodeOptions::default();

    let batch = export_arrow_batch(&frame, None, &dopts).unwrap();
    assert_eq!(batch.num_rows, 320);
    assert_eq!(batch.columns.len(), 7);
    assert_eq!(batch.columns[0].name, "id");
    assert_eq!(batch.columns[0].data.data_type(), ArrowType::Int64);
    assert_eq!(batch.columns[2].name, "tip_cents");
    assert_eq!(batch.columns[2].data.data_type(), ArrowType::Int64);
    assert_eq!(batch.columns[2].null_count, 64);
    assert_eq!(batch.columns[3].name, "delta");
    assert_eq!(batch.columns[3].data.data_type(), ArrowType::Int64);
    assert_eq!(batch.columns[4].name, "fare");
    assert_eq!(batch.columns[4].data.data_type(), ArrowType::Float64);
    assert_eq!(batch.columns[5].name, "ok");
    assert_eq!(batch.columns[5].data.data_type(), ArrowType::Boolean);
    assert_eq!(batch.columns[6].name, "zone");
    assert_eq!(batch.columns[6].data.data_type(), ArrowType::Utf8);

    let ipc = batch.to_ipc_stream();
    assert_eq!(ipc.len() % 8, 0, "IPC stream must be 8-byte aligned");
    let decoded = ArrowBatch::from_ipc_stream(&ipc).unwrap();
    assert_eq!(decoded, batch);

    for cut in [0usize, 3, 8, 24, ipc.len() / 2] {
        assert!(ArrowBatch::from_ipc_stream(&ipc[..cut]).is_err());
    }

    let csv = build_cumulative_csv(280);
    let csv_frame = compress(&csv, &Options::default()).unwrap();
    let csv_batch = export_arrow_batch(&csv_frame, None, &dopts).unwrap();
    assert_eq!(csv_batch.num_rows, 280);
    assert_eq!(csv_batch.columns[0].data.data_type(), ArrowType::Utf8);
    assert_eq!(csv_batch.columns[2].data.data_type(), ArrowType::Int64);
    assert_eq!(csv_batch.columns[3].data.data_type(), ArrowType::Int64);
    assert_eq!(csv_batch.columns[4].data.data_type(), ArrowType::Float64);
    assert_eq!(csv_batch.columns[5].data.data_type(), ArrowType::Utf8);

    let q = SqlQuery::parse(
        "SELECT country, COUNT(*), SUM(confirmed) FROM frame GROUP BY country ORDER BY country ASC",
    )
    .unwrap();
    let grp_ipc = export_arrow_ipc(&csv_frame, Some(&q), &dopts).unwrap();
    let grp_batch = ArrowBatch::from_ipc_stream(&grp_ipc).unwrap();
    assert_eq!(grp_batch.num_rows, 3);
    assert_eq!(grp_batch.columns.len(), 3);
}

#[test]
fn arrow_where_and_projection_differential_parity_1t_4t() {
    let ndjson = build_mixed_ndjson(1500);
    let csv = build_cumulative_csv(1500);

    for record_aligned in [false, true] {
        let enc = Options {
            block_size: 4096,
            threads: 2,
            record_aligned,
            ..Options::default()
        };
        let nd_frame = compress(&ndjson, &enc).unwrap();
        let csv_frame = compress(&csv, &enc).unwrap();

        for threads in [1usize, 4] {
            let dec = DecodeOptions {
                threads,
                max_output: usize::MAX,
                ..DecodeOptions::default()
            };

            // 1. NDJSON filtered + projected parity against `ArrowBatch::from_sql_table`
            let cols = parse_col_list("id,tip_cents,delta,fare,ok,zone").unwrap();
            for (sql_w, lim) in [
                (None, None),
                (Some("payment_type = 2 AND fare >= 15.0"), None),
                (Some("id >= 250 AND id <= 900 AND ok = true"), Some(77usize)),
                (Some("zone IN ('jfk', 'brooklyn')"), None),
            ] {
                let mut sql_s =
                    "SELECT id, tip_cents, delta, fare, ok, zone FROM frame".to_owned();
                if let Some(w) = sql_w {
                    sql_s.push_str(&format!(" WHERE {w}"));
                }
                if let Some(l) = lim {
                    sql_s.push_str(&format!(" LIMIT {l}"));
                }
                let q = SqlQuery::parse(&sql_s).unwrap();
                let ref_batch =
                    ArrowBatch::from_sql_table(&execute_sql_table(&nd_frame, &q, &dec).unwrap());

                let got = match sql_w {
                    Some(w) => {
                        let aw = parse_arrow_where(w).unwrap();
                        aw.with_pred_expr(|we| {
                            export_arrow_batch_where(&nd_frame, Some(&cols), Some(we), lim, &dec)
                        })
                        .unwrap()
                    }
                    None => {
                        export_arrow_batch_where(&nd_frame, Some(&cols), None, lim, &dec).unwrap()
                    }
                };
                assert_eq!(
                    got, ref_batch,
                    "NDJSON mismatch (aligned={record_aligned}, threads={threads}, w={sql_w:?})"
                );

                // Unaligned bitmap batch coalescing (`batch_size = 113`)
                let batches = match sql_w {
                    Some(w) => {
                        let aw = parse_arrow_where(w).unwrap();
                        aw.with_pred_expr(|we| {
                            export_arrow_batches_where(
                                &nd_frame,
                                Some(&cols),
                                Some(we),
                                113,
                                lim,
                                &dec,
                            )
                        })
                        .unwrap()
                    }
                    None => export_arrow_batches_where(&nd_frame, Some(&cols), None, 113, lim, &dec)
                        .unwrap(),
                };
                let sum_rows: usize = batches.iter().map(|b| b.num_rows).sum();
                assert_eq!(sum_rows, ref_batch.num_rows);
                for b in &batches {
                    assert!(b.num_rows <= 113 && b.num_rows > 0);
                    for (ci, c) in b.columns.iter().enumerate() {
                        assert_eq!(c.data.data_type(), ref_batch.columns[ci].data.data_type());
                    }
                }
            }

            // 2. CSV cumulative PFOR-Delta + signed decimals parity
            let csv_cols = parse_col_list("country,confirmed,net_change,rate,note").unwrap();
            let aw_csv = parse_arrow_where("country = 'DE' AND confirmed >= 9000").unwrap();
            let got_csv = aw_csv
                .with_pred_expr(|we| {
                    export_arrow_batch_where(&csv_frame, Some(&csv_cols), Some(we), None, &dec)
                })
                .unwrap();
            let q_csv = SqlQuery::parse(
                "SELECT country, confirmed, net_change, rate, note FROM frame WHERE country = 'DE' AND confirmed >= 9000",
            )
            .unwrap();
            let ref_csv =
                ArrowBatch::from_sql_table(&execute_sql_table(&csv_frame, &q_csv, &dec).unwrap());
            assert_eq!(got_csv, ref_csv);

            // 3. 0-match WHERE preserves exact non-Utf8 column types from block-0 schema probe!
            let aw_empty = parse_arrow_where("id < 0").unwrap();
            let empty_b = aw_empty
                .with_pred_expr(|we| {
                    export_arrow_batch_where(&nd_frame, Some(&cols), Some(we), None, &dec)
                })
                .unwrap();
            assert_eq!(empty_b.num_rows, 0);
            let empty_types: Vec<ArrowType> =
                empty_b.columns.iter().map(|c| c.data.data_type()).collect();
            assert_eq!(
                empty_types,
                vec![
                    ArrowType::Int64,
                    ArrowType::Int64,
                    ArrowType::Int64,
                    ArrowType::Float64,
                    ArrowType::Boolean,
                    ArrowType::Utf8,
                ]
            );
            let empty_ipc = aw_empty
                .with_pred_expr(|we| {
                    export_arrow_ipc_where(&nd_frame, Some(&cols), Some(we), None, &dec)
                })
                .unwrap();
            assert_eq!(ArrowBatch::from_ipc_stream(&empty_ipc).unwrap(), empty_b);
        }
    }
}

#[test]
fn arrow_c_data_and_stream_abi_alignment_and_release_safety() {
    let ndjson = build_mixed_ndjson(320);
    let frame = compress(&ndjson, &Options::default()).unwrap();
    let q_cstr = CString::new("SELECT payment_type, tip_cents, fare, ok, zone FROM frame WHERE id < 50").unwrap();

    // 1. Direct `(ArrowSchema, ArrowArray)` export + 64-byte alignment + child-stealing release
    unsafe {
        let mut schema = MaybeUninit::<ArrowSchema>::uninit();
        let mut array = MaybeUninit::<ArrowArray>::uninit();
        let rc = vxg_arrow_export(
            frame.as_ptr(),
            frame.len(),
            q_cstr.as_ptr(),
            2,
            schema.as_mut_ptr(),
            array.as_mut_ptr(),
        );
        assert_eq!(rc, VXG_OK);
        let mut s = schema.assume_init();
        let mut a = array.assume_init();
        assert_eq!(CStr::from_ptr(s.format).to_bytes(), b"+s");
        assert_eq!(s.n_children, 5);
        assert_eq!(a.length, 50);
        assert_eq!(a.n_children, 5);

        for ci in 0..5usize {
            let child_a = &*(*a.children.add(ci));
            for bi in 0..(child_a.n_buffers as usize) {
                let bptr = *child_a.buffers.add(bi);
                if !bptr.is_null() {
                    assert_eq!((bptr as usize) % 64, 0);
                }
            }
        }

        let child0_ptr = *a.children;
        let mut stolen = std::ptr::read(child0_ptr);
        (*child0_ptr).release = None;

        (a.release.unwrap())(&mut a);
        assert!(a.release.is_none());
        (stolen.release.unwrap())(&mut stolen);
        assert!(stolen.release.is_none());
        (s.release.unwrap())(&mut s);
        assert!(s.release.is_none());
    }

    // 2. `vxg_arrow_export_stream` (`get_schema`, `get_next` batch + EOS, `vxg_arrow_ipc_alloc`)
    unsafe {
        let mut st = MaybeUninit::<ArrowArrayStream>::uninit();
        let rc = vxg_arrow_export_stream(
            frame.as_ptr(),
            frame.len(),
            std::ptr::null(),
            2,
            st.as_mut_ptr(),
        );
        assert_eq!(rc, VXG_OK);
        let mut st = st.assume_init();

        for _ in 0..2 {
            let mut s = MaybeUninit::<ArrowSchema>::uninit();
            assert_eq!((st.get_schema.unwrap())(&mut st, s.as_mut_ptr()), 0);
            let mut s = s.assume_init();
            assert_eq!(s.n_children, 7);
            (s.release.unwrap())(&mut s);
        }

        let mut a1 = MaybeUninit::<ArrowArray>::uninit();
        assert_eq!((st.get_next.unwrap())(&mut st, a1.as_mut_ptr()), 0);
        let mut a1 = a1.assume_init();
        assert_eq!(a1.length, 320);
        (a1.release.unwrap())(&mut a1);

        let mut a_eos = MaybeUninit::<ArrowArray>::uninit();
        assert_eq!((st.get_next.unwrap())(&mut st, a_eos.as_mut_ptr()), 0);
        let a_eos = a_eos.assume_init();
        assert!(a_eos.release.is_none(), "EOS array must have release == NULL");

        (st.release.unwrap())(&mut st);
        assert!(st.release.is_none());

        let mut ipc_len = 0usize;
        let ipc_ptr = vxg_arrow_ipc_alloc(
            frame.as_ptr(),
            frame.len(),
            q_cstr.as_ptr(),
            2,
            &mut ipc_len,
        );
        assert!(!ipc_ptr.is_null() && ipc_len > 0);
        let ipc_slice = std::slice::from_raw_parts(ipc_ptr, ipc_len);
        let parsed = ArrowBatch::from_ipc_stream(ipc_slice).unwrap();
        assert_eq!(parsed.num_rows, 50);
        vxg_free(ipc_ptr, ipc_len);
    }

    // 3. `vxg_arrow_query` & multi-batch (>65,536 rows) `vxg_arrow_query_stream` + error codes
    let big_ndjson = build_mixed_ndjson(70_000);
    let big_frame = compress(
        &big_ndjson,
        &Options {
            threads: 4,
            record_aligned: true,
            ..Options::default()
        },
    )
    .unwrap();
    let cols_c = CString::new("id,fare,ok").unwrap();
    let where_vxg = CString::new("id>=100 && (zone=prefix:mid || zone=eq:brooklyn)").unwrap();
    let where_empty = CString::new("id < 0").unwrap();
    let where_bad = CString::new("((id >= not_a_number").unwrap();

    unsafe {
        let mut s = MaybeUninit::<ArrowSchema>::uninit();
        let mut a = MaybeUninit::<ArrowArray>::uninit();
        assert_eq!(
            vxg_arrow_query(
                frame.as_ptr(),
                frame.len(),
                cols_c.as_ptr(),
                where_vxg.as_ptr(),
                2,
                s.as_mut_ptr(),
                a.as_mut_ptr(),
            ),
            VXG_OK
        );
        let mut s = s.assume_init();
        let mut a = a.assume_init();
        assert_eq!(s.n_children, 3);
        assert_eq!(a.length, 110);
        (a.release.unwrap())(&mut a);
        (s.release.unwrap())(&mut s);

        // Multi-batch stream (>65,536 rows yields >= 2 RecordBatches!)
        let mut st = MaybeUninit::<ArrowArrayStream>::uninit();
        assert_eq!(
            vxg_arrow_query_stream(
                big_frame.as_ptr(),
                big_frame.len(),
                cols_c.as_ptr(),
                std::ptr::null(),
                4,
                st.as_mut_ptr(),
            ),
            VXG_OK
        );
        let mut st = st.assume_init();
        let mut batches_seen = 0usize;
        let mut total_rows = 0i64;
        loop {
            let mut batch_a = MaybeUninit::<ArrowArray>::uninit();
            assert_eq!((st.get_next.unwrap())(&mut st, batch_a.as_mut_ptr()), 0);
            let mut batch_a = batch_a.assume_init();
            if batch_a.release.is_none() {
                break;
            }
            batches_seen += 1;
            total_rows += batch_a.length;
            (batch_a.release.unwrap())(&mut batch_a);
        }
        assert!(batches_seen >= 2, "expected >=2 batches, got {batches_seen}");
        assert_eq!(total_rows, 70_000);
        (st.release.unwrap())(&mut st);

        // 0-match stream preserves Int64/Float64/Boolean schema formats + immediate EOS
        let mut st0 = MaybeUninit::<ArrowArrayStream>::uninit();
        assert_eq!(
            vxg_arrow_query_stream(
                frame.as_ptr(),
                frame.len(),
                cols_c.as_ptr(),
                where_empty.as_ptr(),
                2,
                st0.as_mut_ptr(),
            ),
            VXG_OK
        );
        let mut st0 = st0.assume_init();
        let mut s0 = MaybeUninit::<ArrowSchema>::uninit();
        assert_eq!((st0.get_schema.unwrap())(&mut st0, s0.as_mut_ptr()), 0);
        let mut s0 = s0.assume_init();
        assert_eq!(CStr::from_ptr((*(*s0.children.add(0))).format).to_bytes(), b"l");
        assert_eq!(CStr::from_ptr((*(*s0.children.add(1))).format).to_bytes(), b"g");
        assert_eq!(CStr::from_ptr((*(*s0.children.add(2))).format).to_bytes(), b"b");
        (s0.release.unwrap())(&mut s0);
        let mut eos = MaybeUninit::<ArrowArray>::uninit();
        assert_eq!((st0.get_next.unwrap())(&mut st0, eos.as_mut_ptr()), 0);
        assert!(eos.assume_init().release.is_none());
        (st0.release.unwrap())(&mut st0);

        // Error codes on NULL / invalid WHERE / corrupt frame
        let mut st_err = MaybeUninit::<ArrowArrayStream>::uninit();
        assert_eq!(
            vxg_arrow_query_stream(
                std::ptr::null(),
                10,
                 std::ptr::null(),
                std::ptr::null(),
                1,
                st_err.as_mut_ptr()
            ),
            VXG_ERR_NULL_PTR
        );
        assert_eq!(
            vxg_arrow_query_stream(
                frame.as_ptr(),
                frame.len(),
                cols_c.as_ptr(),
                where_bad.as_ptr(),
                1,
                st_err.as_mut_ptr()
            ),
            VXG_ERR_INVALID_OPTION
        );
        assert_eq!(
            vxg_arrow_query_stream(
                b"not_a_vxg".as_ptr(),
                9,
                std::ptr::null(),
                std::ptr::null(),
                1,
                st_err.as_mut_ptr()
            ),
            VXG_ERR_CORRUPT_DATA
        );
    }
}

#[test]
fn cli_vxg_arrow_and_sql_arrow_flag_end_to_end() {
    let bin = env!("CARGO_BIN_EXE_vxg");
    let dir = std::env::temp_dir().join(format!("vxg_arrow_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let vxg_path = dir.join("sample.vxg");
    let arrows_path = dir.join("sample.arrows");
    let filtered_path = dir.join("filtered.arrows");

    let frame = compress(&build_mixed_ndjson(320), &Options::default()).unwrap();
    std::fs::write(&vxg_path, &frame).unwrap();

    let st = Command::new(bin)
        .args(["arrow", vxg_path.to_str().unwrap(), arrows_path.to_str().unwrap(), "-f", "-q"])
        .status()
        .unwrap();
    assert!(st.success());
    let b1 = ArrowBatch::from_ipc_stream(&std::fs::read(&arrows_path).unwrap()).unwrap();
    assert_eq!(b1.num_rows, 320);

    // CLI `--cols` + `--where` + `--limit`
    let st_f = Command::new(bin)
        .args([
            "arrow",
            vxg_path.to_str().unwrap(),
            filtered_path.to_str().unwrap(),
            "--cols",
            "id,fare,zone",
            "--where",
            "payment_type = 1 AND id >= 20",
            "--limit",
            "15",
            "-t",
            "2",
            "-f",
            "-q",
        ])
        .status()
        .unwrap();
    assert!(st_f.success());
    let bf = ArrowBatch::from_ipc_stream(&std::fs::read(&filtered_path).unwrap()).unwrap();
    assert_eq!(bf.num_rows, 15);
    assert_eq!(bf.columns.len(), 3);

    // Reject `--sql` combined with `--where`
    let bad = Command::new(bin)
        .args([
            "arrow",
            vxg_path.to_str().unwrap(),
            filtered_path.to_str().unwrap(),
            "--sql",
            "SELECT * FROM frame",
            "--where",
            "id > 0",
            "-f",
        ])
        .output()
        .unwrap();
    assert!(!bad.status.success());

    let out = Command::new(bin)
        .args([
            "sql",
            vxg_path.to_str().unwrap(),
            "SELECT payment_type, COUNT(*) FROM frame GROUP BY payment_type ORDER BY payment_type ASC",
            "--arrow",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let b2 = ArrowBatch::from_ipc_stream(&out.stdout).unwrap();
    assert_eq!(b2.num_rows, 4);
    let _ = std::fs::remove_dir_all(&dir);
}
