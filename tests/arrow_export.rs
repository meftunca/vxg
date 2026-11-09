//! Integration, ABI safety, and IPC roundtrip tests for VXG's zero-dependency Apache Arrow
//! C Data Interface (`ArrowSchema`, `ArrowArray`, `ArrowArrayStream`) and IPC Stream (`V5`).

use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::process::Command;

use vxg::ffi::{
    vxg_arrow_export, vxg_arrow_export_stream, vxg_arrow_ipc_alloc, vxg_free, ArrowArray,
    ArrowArrayStream, ArrowSchema, VXG_OK,
};
use vxg::{
    compress, export_arrow_batch, export_arrow_ipc, ArrowBatch, ArrowData, ArrowType,
    DecodeOptions, Options, SqlQuery,
};

fn build_mixed_ndjson() -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..320usize {
        let pay = 1 + (i % 4);
        let tip = if i % 5 == 0 {
            "null".to_owned()
        } else {
            ((i * 15) % 600).to_string()
        };
        let fare = format!("{}.{:02}", 8 + (i % 30), (i * 7) % 100);
        let ok = if i % 2 == 0 { "true" } else { "false" };
        let zone = match i % 3 {
            0 => "midtown",
            1 => "jfk",
            _ => "brooklyn",
        };
        out.extend_from_slice(
            format!(
                "{{\"id\":{i},\"payment_type\":{pay},\"tip_cents\":{tip},\"fare\":{fare},\"ok\":{ok},\"zone\":\"{zone}\"}}\n"
            )
            .as_bytes(),
        );
    }
    out
}

#[test]
fn arrow_batch_and_ipc_roundtrip_ndjson_and_csv() {
    let ndjson = build_mixed_ndjson();
    let frame = compress(&ndjson, &Options::default()).unwrap();
    let dopts = DecodeOptions::default();

    // 1. Automatic NDJSON top-level key discovery + typed column inference
    let batch = export_arrow_batch(&frame, None, &dopts).unwrap();
    assert_eq!(batch.num_rows, 320);
    assert_eq!(batch.columns.len(), 6);
    assert_eq!(batch.columns[0].name, "id");
    assert_eq!(batch.columns[0].data.data_type(), ArrowType::Int64);
    assert_eq!(batch.columns[2].name, "tip_cents");
    assert_eq!(batch.columns[2].data.data_type(), ArrowType::Int64);
    assert_eq!(batch.columns[2].null_count, 64);
    assert_eq!(batch.columns[3].name, "fare");
    assert_eq!(batch.columns[3].data.data_type(), ArrowType::Float64);
    assert_eq!(batch.columns[4].name, "ok");
    assert_eq!(batch.columns[4].data.data_type(), ArrowType::Boolean);
    assert_eq!(batch.columns[5].name, "zone");
    assert_eq!(batch.columns[5].data.data_type(), ArrowType::Utf8);

    let ipc = batch.to_ipc_stream();
    assert_eq!(ipc.len() % 8, 0, "IPC stream must be 8-byte aligned");
    let decoded = ArrowBatch::from_ipc_stream(&ipc).unwrap();
    assert_eq!(decoded, batch);

    // Malformed / truncated IPC prefixes must return clean Err without panicking
    for cut in [0usize, 3, 8, 24, ipc.len() / 2] {
        assert!(ArrowBatch::from_ipc_stream(&ipc[..cut]).is_err());
    }

    // 2. CSV with signed integers, decimals, and SQL GROUP BY
    let mut csv = Vec::from(&b"region,delta_i64,amount_dec\n"[..]);
    for i in 0..120i64 {
        let reg = if i % 2 == 0 { "eu" } else { "us" };
        let di = (i - 60) * 10;
        let dec = format!("{}.{:02}", i - 40, (i.unsigned_abs() * 13) % 100);
        csv.extend_from_slice(format!("{reg},{di},{dec}\n").as_bytes());
    }
    let csv_frame = compress(&csv, &Options::default()).unwrap();
    let csv_batch = export_arrow_batch(&csv_frame, None, &dopts).unwrap();
    assert_eq!(csv_batch.num_rows, 120);
    assert_eq!(csv_batch.columns[0].data.data_type(), ArrowType::Utf8);
    assert_eq!(csv_batch.columns[1].data.data_type(), ArrowType::Int64);
    assert_eq!(csv_batch.columns[2].data.data_type(), ArrowType::Float64);
    if let ArrowData::Int64(ref vs) = csv_batch.columns[1].data {
        assert_eq!(vs[0], -600);
        assert_eq!(vs[60], 0);
    } else {
        panic!("expected Int64 column");
    }

    let q = SqlQuery::parse(
        "SELECT region, COUNT(*), SUM(delta_i64) FROM frame GROUP BY region ORDER BY region ASC",
    )
    .unwrap();
    let grp_ipc = export_arrow_ipc(&csv_frame, Some(&q), &dopts).unwrap();
    let grp_batch = ArrowBatch::from_ipc_stream(&grp_ipc).unwrap();
    assert_eq!(grp_batch.num_rows, 2);
    assert_eq!(grp_batch.columns.len(), 3);
}

#[test]
fn arrow_c_data_and_stream_abi_alignment_and_release_safety() {
    let ndjson = build_mixed_ndjson();
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
                    assert_eq!(
                        (bptr as usize) % 64,
                        0,
                        "col {ci} buffer {bi} not 64-byte aligned"
                    );
                }
            }
        }

        // Steal child 0 per Arrow C Data ABI (copy struct, set source release = None)
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

    // 2. `ArrowArrayStream` (`get_schema`, `get_next` batch + EOS, `vxg_arrow_ipc_alloc`)
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
            assert_eq!(s.n_children, 6);
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
}

#[test]
fn cli_vxg_arrow_and_sql_arrow_flag_end_to_end() {
    let bin = env!("CARGO_BIN_EXE_vxg");
    let dir = std::env::temp_dir().join(format!("vxg_arrow_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let vxg_path = dir.join("sample.vxg");
    let arrows_path = dir.join("sample.arrows");

    let frame = compress(&build_mixed_ndjson(), &Options::default()).unwrap();
    std::fs::write(&vxg_path, &frame).unwrap();

    let st = Command::new(bin)
        .args([
            "arrow",
            vxg_path.to_str().unwrap(),
            arrows_path.to_str().unwrap(),
            "-f",
            "-q",
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let file_ipc = std::fs::read(&arrows_path).unwrap();
    let b1 = ArrowBatch::from_ipc_stream(&file_ipc).unwrap();
    assert_eq!(b1.num_rows, 320);

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
