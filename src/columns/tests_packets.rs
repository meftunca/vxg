#[cfg(test)]
mod tests_packets {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let packed = pack_block(data).unwrap();
        assert_eq!(unpack_block(&packed.bytes).unwrap(), data);
        if let Some(grammar) = encode_grammar_block(data).unwrap() {
            assert_eq!(decode_block(&grammar).unwrap(), data);
        }
    }

    #[test]
    fn decimal_delta_model_packs_mixed_progressions_and_is_canonical() {
        let values = (0..256u64)
            .map(|row| (row * 2 + row % 3).to_string().into_bytes())
            .collect::<Vec<_>>();
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::DecimalDelta);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut trailing = packed.clone();
        trailing.push(0);
        assert!(unpack_model(&trailing, values.len()).is_err());
        let mut truncated = packed.clone();
        truncated.pop();
        assert!(unpack_model(&truncated, values.len()).is_err());
    }

    #[test]
    fn r2_style_blocks_roundtrip_with_new_models() {
        let event = (0..256u32)
            .flat_map(|row| {
                let epoch = row / 128;
                let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
                format!(
                    "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"event\":\"session.open\",\"region\":\"eu-1\",\"request_id\":\"r-{prefix:08x}{:08x}\",\"status\":\"ok\",\"sequence\":{row}}}\n",
                    row % 28 + 1,
                    row % 24,
                    row % 128,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&event);
        let bursty = (0..256u32)
            .flat_map(|row| {
                format!(
                    "{{\"burst\":{},\"record\":{row},\"tags\":[\"hot\",\"retry\"],\"message\":\"queue depth exceeded; retry scheduled; {}\",\"source\":\"worker-{}\"}}\n",
                    row % 7,
                    row % 4096,
                    row % 23,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&bursty);
    }

    #[test]
    fn csv_tsv_and_logfmt_blocks_compress_and_roundtrip() {
        let csv = (0..512u32)
            .flat_map(|row| {
                let epoch = row / 128;
                let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
                format!(
                    "\"2026-01-{:02}T{:02}:{:02}:00Z\",{row},\"session.open\",\"eu-1\",\"r-{prefix:08x}{:08x}\",\"worker-{}\"\n",
                    row % 28 + 1,
                    (row / 60) % 24,
                    row % 60,
                    row % 128,
                    row % 16,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&csv);
        let csv_packet = encode_grammar_block(&csv).unwrap().unwrap();
        let mut csv_out = vec![0u8; csv.len()];
        decode_block_into(&csv_packet, &mut csv_out).unwrap();
        assert_eq!(csv_out, csv);
        assert!(csv_packet.len() * 20 < csv.len());

        let tsv = (0..512u32)
            .flat_map(|row| {
                format!(
                    "{row}\t{}\t\"queue depth exceeded; retry scheduled; {}\"\t\"worker-{}\"\n",
                    100_000 + row * 5,
                    row % 128,
                    row % 8,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&tsv);
        let tsv_packet = encode_grammar_block(&tsv).unwrap().unwrap();
        let mut tsv_out = vec![0u8; tsv.len()];
        decode_block_into(&tsv_packet, &mut tsv_out).unwrap();
        assert_eq!(tsv_out, tsv);
        assert!(tsv_packet.len() * 20 < tsv.len());

        let logfmt = (0..512u32)
            .flat_map(|row| {
                let epoch = row / 128;
                let prefix = 0x3b7f_2c4du32.wrapping_add(epoch.wrapping_mul(0x2545_f491));
                format!(
                    "ts=\"2026-01-{:02}T{:02}:{:02}:00Z\" level=info seq={row} req_id=\"r-{prefix:08x}{:08x}\" msg=\"queue depth exceeded; retry scheduled; {}\" worker=\"worker-{}\"\n",
                    row % 28 + 1,
                    (row / 60) % 24,
                    row % 60,
                    row % 128,
                    row % 64,
                    row % 12,
                )
                .into_bytes()
            })
            .collect::<Vec<_>>();
        roundtrip(&logfmt);
        let logfmt_packet = encode_grammar_block(&logfmt).unwrap().unwrap();
        let mut logfmt_out = vec![0u8; logfmt.len()];
        decode_block_into(&logfmt_packet, &mut logfmt_out).unwrap();
        assert_eq!(logfmt_out, logfmt);
        assert!(logfmt_packet.len() * 20 < logfmt.len());
    }

    #[test]
    fn huffman_roundtrips_and_rejects_malformed() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            seed >> 16
        };
        let mut cases: Vec<Vec<u8>> = Vec::new();
        // Skewed (delta-like) residuals of many lengths, including non-multiples of 4.
        for &n in &[32usize, 33, 256, 257, 1001, 4096, 16_383, 65_536] {
            cases.push((0..n).map(|_| { let r = rnd() % 100; if r < 60 { 0 } else if r < 80 { 1 } else if r < 90 { 255 } else { (r * 7) as u8 } }).collect());
        }
        // Fibonacci-like frequencies force length limiting.
        let mut fib = Vec::new();
        let (mut a, mut b) = (1usize, 1usize);
        for sym in 0..24u8 {
            fib.extend(std::iter::repeat(sym).take(a));
            (a, b) = (b, a + b);
            if fib.len() > 60_000 { break; }
        }
        cases.push(fib);
        cases.push(vec![42u8; 5000]);
        for src in &cases {
            let mut blob = Vec::new();
            if !huf_encode(src, &mut blob, src.len()) {
                assert!(src.len() < HUF_MIN, "len {} should compress", src.len());
                continue;
            }
            assert!(blob.len() < src.len());
            let mut out = vec![0u8; src.len()];
            huf_decode(&blob, &mut out).unwrap();
            assert_eq!(&out, src);
            // Malformed: truncation, trailing byte, wrong output size.
            for cut in [1usize, 2, blob.len() / 2, blob.len() - 1] {
                if cut < blob.len() {
                    let mut o = vec![0u8; src.len()];
                    assert!(huf_decode(&blob[..cut], &mut o).is_err() || o != *src);
                }
            }
            let mut extra = blob.clone();
            extra.push(0);
            let mut o = vec![0u8; src.len()];
            assert!(huf_decode(&extra, &mut o).is_err());
        }
        // Incomplete code (single length-1 symbol plus nothing) is rejected.
        assert!(huf_decode(&[HUF_KIND_HUF, 1, 0x01, 0, 0, 0], &mut [0u8; 8]).is_err());
        // Uniform random bytes do not compress.
        let rand: Vec<u8> = (0..4096).map(|_| rnd() as u8).collect();
        assert!(!huf_encode(&rand, &mut Vec::new(), rand.len()));
    }

    #[test]
    fn stride_image_blocks_compress_roundtrip_and_reject_malformed() {
        // 1. 32-bit BMP (54-byte header + 16370 RGBA pixels + 2 tail bytes = 65536 bytes)
        let mut bmp32 = vec![0u8; 65536];
        bmp32[0] = b'B';
        bmp32[1] = b'M';
        bmp32[10..14].copy_from_slice(&54u32.to_le_bytes());
        bmp32[28..30].copy_from_slice(&32u16.to_le_bytes());
        for px in 0..16370usize {
            let base = 54 + px * 4;
            let band = (px / 256) as u8;
            let edge = if px % 256 == 0 { (px / 17) as u8 } else { 0 };
            bmp32[base] = band.wrapping_mul(13).wrapping_add(edge);
            bmp32[base + 1] = band.wrapping_mul(29);
            bmp32[base + 2] = band.wrapping_mul(47).wrapping_sub(edge);
            bmp32[base + 3] = 255;
        }
        bmp32[65534] = 0xaa;
        bmp32[65535] = 0x55;
        let pkt32 = encode_grammar_block(&bmp32).unwrap().unwrap();
        assert_eq!(pkt32[0], STRIDE_PACKET);
        assert!(pkt32.len() * 20 < bmp32.len());
        let mut out32 = vec![0u8; bmp32.len()];
        decode_block_into(&pkt32, &mut out32).unwrap();
        assert_eq!(out32, bmp32);
        assert_eq!(decode_block(&pkt32).unwrap(), bmp32);

        // 2. 24-bit raw RGB image (stride = 3, 21845 pixels + 1 tail byte = 65536 bytes)
        let mut rgb24 = vec![0u8; 65536];
        for px in 0..21845usize {
            let base = px * 3;
            let run = (px / 128) as u8;
            rgb24[base] = run.wrapping_mul(7);
            rgb24[base + 1] = run.wrapping_mul(19).wrapping_add((px % 128 == 63) as u8);
            rgb24[base + 2] = run.wrapping_mul(37);
        }
        rgb24[65535] = 0x7e;
        let pkt24 = encode_grammar_block(&rgb24).unwrap().unwrap();
        assert_eq!(pkt24[0], STRIDE_PACKET);
        assert!(pkt24.len() * 10 < rgb24.len());
        let mut out24 = vec![0u8; rgb24.len()];
        decode_block_into(&pkt24, &mut out24).unwrap();
        assert_eq!(out24, rgb24);

        // 3. Reject truncated and trailing-byte STRIDE_PACKETs
        assert!(decode_block_into(&pkt32[..pkt32.len() - 1], &mut out32).is_err());
        let mut trailing = pkt32.clone();
        trailing.push(0);
        assert!(decode_block_into(&trailing, &mut out32).is_err());
    }

    #[test]
    fn fast_lz_packet_compresses_prose_code_and_rejects_malformed() {
        let mut prose = Vec::with_capacity(65536);
        for i in 0..512u32 {
            prose.extend_from_slice(
                format!(
                    "pub fn compute_block_digest_{}(input: &[u8], seed: u64) -> Result<u64> {{\n    let mut acc = seed.wrapping_add({});\n    for &byte in input {{\n        acc = acc.rotate_left(5) ^ (byte as u64);\n    }}\n    Ok(acc)\n}}\n",
                    i % 16,
                    i * 31,
                )
                .as_bytes(),
            );
        }
        prose.truncate(65536);
        let pkt = encode_grammar_block(&prose).unwrap().unwrap();
        assert_eq!(pkt[0], FAST_LZ_PACKET);
        assert!(pkt.len() * 4 < prose.len());
        let mut out = vec![0u8; prose.len()];
        decode_block_into(&pkt, &mut out).unwrap();
        assert_eq!(out, prose);
        assert_eq!(decode_block(&pkt).unwrap(), prose);

        assert!(decode_block_into(&pkt[..pkt.len() - 1], &mut out).is_err());
        let mut trailing = pkt.clone();
        trailing.push(0);
        assert!(decode_block_into(&trailing, &mut out).is_err());
    }

    #[test]
    fn front_dict_palette_base32_and_mixed_raw_rows_roundtrip() {
        const BASE32: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
        let mut mixed = Vec::new();
        for row in 0..160usize {
            // Interleaved plain-text line and structured JSON line with Base32 IDs + IPv4 + unique trace IDs:
            mixed.extend_from_slice(
                b"IP Middleware: [10.42.0.15] -> /api/v1/orders (200 OK)\n",
            );
            let user_idx = if row % 5 < 3 { 0 } else { (row / 3) % 19 };
            let mut user_id = String::with_capacity(26);
            for k in 0..26usize {
                let c = BASE32[(user_idx * 7 + k * 13) & 31] as char;
                user_id.push(c);
            }
            let ip = format!("172.69.{}.{}", 10 + (row / 8) % 12, 100 + (row % 23));
            let trace = format!("{:032x}", (row as u128 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c834u128));
            mixed.extend_from_slice(
                format!(
                    "{{\"time\":\"2026-03-19T13:55:{:02}.{:06}Z\",\"level\":\"INFO\",\"user\":\"{user_id}\",\"ip\":\"{ip}\",\"trace\":\"{trace}\"}}\n",
                    row % 60,
                    (row * 4321) % 1_000_000,
                )
                .as_bytes(),
            );
            if row % 16 == 0 {
                mixed.extend_from_slice(
                    format!(
                        "SELECT id, company_id, status FROM orders WHERE company_id = '{user_id}' AND deleted_at IS NULL LIMIT {};\n",
                        10 + row,
                    )
                    .as_bytes(),
                );
            }
        }
        roundtrip(&mixed);
        let pkt = encode_grammar_block(&mixed).unwrap().unwrap();
        assert_eq!(pkt[0], GRAMMAR_PACKET);
        let mut out = vec![0u8; mixed.len()];
        decode_block_into(&pkt, &mut out).unwrap();
        assert_eq!(out, mixed);
        assert_eq!(decode_block(&pkt).unwrap(), mixed);
    }

    fn stride_roundtrip(data: &[u8], expect_stride: u8) {
        let mut pkt = Vec::new();
        pack_stride_into(data, &mut pkt).expect("stride packet");
        let mut c = Cursor::new(&pkt[1..]);
        assert_eq!(c.uleb().unwrap(), data.len());
        let prefix = c.uleb().unwrap();
        c.take(prefix).unwrap();
        assert_eq!(c.byte().unwrap(), expect_stride);
        let mut out = vec![0u8; data.len()];
        decode_block_into(&pkt, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn stride2_stride8_and_large_block_roundtrip() {
        // 16-bit PCM-like ramp (stride 2).
        let s2: Vec<u8> = (0..30000u32)
            .flat_map(|i| (((i / 8) as u16).wrapping_mul(3)).to_le_bytes())
            .collect();
        stride_roundtrip(&s2, 2);
        // 64-bit monotonically increasing timestamps (stride 8).
        let s8: Vec<u8> = (0..8000u64)
            .flat_map(|i| (1_700_000_000_000u64 + i * 1000).to_le_bytes())
            .collect();
        stride_roundtrip(&s8, 8);
        // > 64 KiB 32-bit block (stride 4).
        let s4: Vec<u8> = (0..60000u32)
            .flat_map(|i| (0x1000_0000u32 + (i / 16)).to_le_bytes())
            .collect();
        assert!(s4.len() > 65536);
        stride_roundtrip(&s4, 4);
    }

    #[test]
    fn multiline_pretty_json_and_clf_bracketed_logs_roundtrip() {
        // 1. Pretty-printed JSON array of multi-line objects:
        let mut pretty_json = Vec::new();
        pretty_json.extend_from_slice(b"[\n");
        for i in 0..120usize {
            let term = if i + 1 < 120 { "},\n" } else { "}\n" };
            pretty_json.extend_from_slice(
                format!(
                    "  {{\n    \"id\": {},\n    \"service\": \"svc-{}\",\n    \"status\": {}\n  {term}",
                    1000 + i,
                    i % 8,
                    if i % 7 == 0 { 500 } else { 200 },
                )
                .as_bytes(),
            );
        }
        pretty_json.extend_from_slice(b"]\n");
        roundtrip(&pretty_json);
        let pkt_json = encode_grammar_block(&pretty_json).unwrap().unwrap();
        assert_eq!(pkt_json[0], GRAMMAR_PACKET);
        assert!(pkt_json.len() * 5 < pretty_json.len());
        let mut out_json = vec![0u8; pretty_json.len()];
        decode_block_into(&pkt_json, &mut out_json).unwrap();
        assert_eq!(out_json, pretty_json);

        // 2. Apache/Nginx Combined Log Format (CLF) + bracketed timestamp/request logs:
        let mut clf_logs = Vec::new();
        for i in 0..160usize {
            clf_logs.extend_from_slice(
                format!(
                    "10.0.4.{} - {} [26/Oct/2026:14:00:{:02} +0000] \"GET /api/v1/items/{} HTTP/1.1\" {} {} \"https://example.com/catalog\" \"Mozilla/5.0\"\n",
                    10 + (i % 32),
                    if i % 4 == 0 { "frank" } else { "-" },
                    i % 60,
                    5000 + i,
                    if i % 11 == 0 { 404 } else { 200 },
                    1024 + (i * 17) % 512,
                )
                .as_bytes(),
            );
        }
        roundtrip(&clf_logs);
        let pkt_clf = encode_grammar_block(&clf_logs).unwrap().unwrap();
        assert_eq!(pkt_clf[0], GRAMMAR_PACKET);
        assert!(pkt_clf.len() * 8 < clf_logs.len());
        let mut out_clf = vec![0u8; clf_logs.len()];
        decode_block_into(&pkt_clf, &mut out_clf).unwrap();
        assert_eq!(out_clf, clf_logs);
    }
}
