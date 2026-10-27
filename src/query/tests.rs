    use super::*;
    use crate::{compress, DecodeOptions, Options};

    fn sample_ndjson(rows: usize) -> Vec<u8> {
        let services = ["auth", "payment", "inventory", "gateway"];
        let levels = ["INFO", "INFO", "WARN", "ERROR"];
        let mut out = Vec::new();
        for i in 0..rows {
            let svc = services[i % services.len()];
            let lvl = if i == rows - 7 { "CRITICAL" } else { levels[i % levels.len()] };
            let line = format!(
                "{{\"ts\":\"2026-10-17T19:{:02}:{:02}Z\",\"seq\":{},\"level\":\"{}\",\"service\":\"{}\",\"latency_us\":{}}}\n",
                (i / 60) % 60, i % 60, 1000 + i, lvl, svc, 100 + (i * 7) % 500
            );
            out.extend_from_slice(line.as_bytes());
        }
        out
    }

    #[test]
    fn random_access_and_pushdown_search_roundtrip() {
        let raw = sample_ndjson(2500);
        let opts = Options::default();
        let dec_opts = DecodeOptions::default();
        let frame = compress(&raw, &opts).unwrap();

        let expected_lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
        let total_rows = frame_row_count(&frame).unwrap();
        assert_eq!(total_rows, expected_lines.len());

        assert_eq!(extract_block_row(&frame, 0, 0, &dec_opts).unwrap(), expected_lines[0]);
        assert_eq!(extract_block_row(&frame, 0, 42, &dec_opts).unwrap(), expected_lines[42]);
        assert_eq!(
            extract_block_row_range(&frame, 0, 10, 25, &dec_opts).unwrap(),
            expected_lines[10..25].concat()
        );
        assert_eq!(extract_row_range(&frame, 0, total_rows, &dec_opts).unwrap(), raw);

        let crit = find_first(&frame, b"CRITICAL", &dec_opts).unwrap().expect("CRITICAL row");
        assert!(contains_bytes(&crit.bytes, b"\"level\":\"CRITICAL\""));
        assert!(contains_bytes(&crit.bytes, b"\"seq\":3493"));

        let crit_span = find_first(&frame, b"\"level\":\"CRITICAL\"", &dec_opts).unwrap().unwrap();
        assert_eq!(crit_span.global_row_index, crit.global_row_index);
        assert_eq!(crit_span.bytes, crit.bytes);

        let static_hit = find_first(&frame, b"\"service\":\"", &dec_opts).unwrap().unwrap();
        assert_eq!(static_hit.global_row_index, 0);
        assert_eq!(static_hit.bytes, expected_lines[0]);
        assert!(find_first(&frame, b"NON_EXISTENT_TOKEN_XYZ", &dec_opts).unwrap().is_none());

        let payment_rows = filter_rows(&frame, b"\"service\":\"payment\"", 10, &dec_opts).unwrap();
        assert_eq!(payment_rows.len(), 10);
        for m in &payment_rows {
            assert!(contains_bytes(&m.bytes, b"\"service\":\"payment\""));
        }

        unsafe {
            assert_eq!(vxg_frame_row_count(frame.as_ptr(), frame.len()) as usize, total_rows);
            let mut buf = [0u8; 512];
            let mut found_idx = 0usize;
            let n = vxg_find_first_into(
                frame.as_ptr(), frame.len(), b"CRITICAL".as_ptr(), 8,
                buf.as_mut_ptr(), buf.len(), &raw mut found_idx,
            );
            assert!(n > 0);
            assert_eq!(found_idx, crit.global_row_index);
            assert_eq!(&buf[..n as usize], crit.bytes.as_slice());
        }
    }
