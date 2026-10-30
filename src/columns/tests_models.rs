#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_scan_string(data: &[u8], mut pos: usize) -> Option<usize> {
        if data.get(pos) != Some(&b'"') {
            return None;
        }
        pos += 1;
        while pos < data.len() {
            match data[pos] {
                b'"' => return Some(pos + 1),
                b'\\' => {
                    if pos + 1 >= data.len() || data[pos + 1] < 0x20 {
                        return None;
                    }
                    pos += 2;
                }
                byte if byte < 0x20 => return None,
                _ => pos += 1,
            }
        }
        None
    }

    #[test]
    fn string_scan_matches_scalar_at_vector_boundaries() {
        for len in [0, 1, 7, 14, 15, 16, 17, 30, 31, 32, 33, 63, 64, 65, 127] {
            for offset in 0..8 {
                let mut data = vec![b'x'; offset];
                data.push(b'"');
                data.extend(std::iter::repeat_n(b'a', len));
                data.push(b'"');
                for position in offset..data.len() {
                    let saved = data[position];
                    for byte in 0..=255 {
                        data[position] = byte;
                        assert_eq!(
                            scan_string(&data, offset), scalar_scan_string(&data, offset),
                            "len={len} offset={offset} position={position} byte={byte}"
                        );
                    }
                    data[position] = saved;
                }
                for end in offset..=data.len() {
                    assert_eq!(
                        scan_string(&data[..end], offset),
                        scalar_scan_string(&data[..end], offset)
                    );
                }
            }
        }
    }

    #[test]
    fn segment_dictionary_roundtrips_across_mask_window_boundary() {
        let words = ["sync paused", "retry", "flush", "worker", "index", "batch committed"];
        let mut seed = 7u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize
        };
        for target in (56..72usize).chain(120..136).chain(186..200).chain(248..266) {
            let mut values = Vec::new();
            for row in 0..200usize {
                let mut inner = String::new();
                while inner.len() < target.saturating_sub(12) {
                    if !inner.is_empty() {
                        inner.push_str("; ");
                    }
                    if rnd() % 4 == 0 {
                        inner.push_str(&(rnd() % 5000).to_string());
                    } else {
                        inner.push_str(words[rnd() % words.len()]);
                    }
                }
                // Trailing pad straddling the 64-byte window for some rows.
                let pad = (row + target) % 9;
                inner.push_str(&" ".repeat(pad));
                values.push(format!("\"{inner}\"").into_bytes());
            }
            let refs: Vec<&[u8]> = values.iter().map(|v| v.as_slice()).collect();
            if let Some(packed) = pack_segment_dictionary_model(&refs, usize::MAX) {
                assert_eq!(unpack_model(&packed, values.len()).unwrap(), values, "target={target}");
            } else {
                panic!("segment dictionary rejected target={target}");
            }
        }
    }

    #[test]
    fn delimited_masks64_match_bytewise_definition() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for round in 0..20_000u32 {
            let mut buf = [0u8; 64];
            for b in buf.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *b = if round < 256 { round as u8 } else { seed as u8 };
            }
            for &delim in b",\t|" {
                let (q, u) = unsafe { delimited_masks64(buf.as_ptr(), delim) };
                for (i, &b) in buf.iter().enumerate() {
                    let eq = b == b'"' || b == b'\\' || b < 0x20;
                    let eu = b == delim
                        || b == b'"'
                        || if delim == b',' || delim == b'\t' {
                            b < 0x20 || b == 0x7f
                        } else {
                            b.wrapping_sub(0x21) >= 0x5e
                        };
                    assert_eq!((q >> i) & 1 == 1, eq, "q byte={b:#x} i={i}");
                    assert_eq!((u >> i) & 1 == 1, eu, "u byte={b:#x} i={i} delim={delim}");
                }
            }
        }
    }

    #[test]
    fn delimited_bitmask_matcher_equals_scalar_reference() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        const ALPHABET: &[u8] = b"abcXYZ019 -_.:,\t|\"\\\x01\x1f\x7f\xc3\xa9";
        static DELIMS: [u8; 3] = [b',', b'\t', b'|'];
        let mut matched = 0usize;
        for di in 0..3 {
            let delim = DELIMS[di];
            for slots in 2..7usize {
                let mut segments: Vec<&[u8]> = vec![b""];
                for _ in 1..slots {
                    segments.push(&DELIMS[di..di + 1]);
                }
                let term: &[u8] = if slots % 2 == 0 { b"\n" } else { b"\r\n" };
                segments.push(term);
                for _ in 0..4000 {
                    let mut row = Vec::new();
                    let rows = 1 + (rnd() % 3) as usize;
                    for _ in 0..rows {
                        for s in 0..slots {
                            let quoted = rnd() % 2 == 0;
                            let len = (rnd() % 90) as usize;
                            if quoted {
                                row.push(b'"');
                            }
                            for _ in 0..len {
                                let r = rnd();
                                // Mostly clean bytes so many rows succeed; sometimes hostile ones.
                                let b = if r % 9 == 0 {
                                    ALPHABET[(r >> 8) as usize % ALPHABET.len()]
                                } else {
                                    b'a' + (r >> 8) as u8 % 26
                                };
                                row.push(b);
                            }
                            if quoted {
                                row.push(b'"');
                            }
                            if s + 1 < slots {
                                row.push(delim);
                            }
                        }
                        row.extend_from_slice(term);
                    }
                    let cut = (rnd() as usize) % (row.len() + 1);
                    let second = row
                        .windows(term.len())
                        .position(|w| w == term)
                        .map_or(0, |i| i + term.len());
                    for data in [&row[..], &row[..cut]] {
                        let mut shared_cache = (usize::MAX, 0u64, 0u64);
                        for start in [0usize, 1, 3, second] {
                            if start > data.len() {
                                continue;
                            }
                            let rem = &data[start..];
                            let mut b: [&[u8]; MAX_SLOTS] = [&[]; MAX_SLOTS];
                            let slow = try_match_delimited_scalar(rem, &segments, delim, &mut b);
                            let mut fresh_cache = (usize::MAX, 0u64, 0u64);
                            for use_shared in [false, true] {
                                let mut a: [&[u8]; MAX_SLOTS] = [&[]; MAX_SLOTS];
                                let cache_ref = if use_shared {
                                    &mut shared_cache
                                } else {
                                    &mut fresh_cache
                                };
                                let fast = try_match_delimited_stream(
                                    data, start, &segments, delim, cache_ref, &mut a,
                                );
                                assert_eq!(fast, slow, "delim={delim} slots={slots} row={rem:?}");
                                if fast.is_some() {
                                    assert_eq!(a[..slots], b[..slots]);
                                    matched += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Guard against a vacuous test: plenty of rows must actually match.
        assert!(matched > 5_000, "only {matched} rows matched");
    }

    fn roundtrip(data: &[u8]) {
        let packed = pack_block(data).unwrap();
        assert_eq!(unpack_block(&packed.bytes).unwrap(), data);
    }

    #[test]
    fn exact_structured_records_select_and_roundtrip_models() {
        let data = (0..256)
            .flat_map(|index| {
                format!(
                    "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"kind\":\"{}\",\"request\":\"r-{:016x}\",\"id\":{index},\"tags\":[\"hot\",\"retry\"]}}\n",
                    index % 28 + 1,
                    index % 24,
                    if index & 1 == 0 { "open" } else { "close" },
                    (index as u64).wrapping_mul(0x9e3779b97f4a7c15),
                )
            }
            .into_bytes())
            .collect::<Vec<_>>();
        roundtrip(&data);
        let grammar = pack_grammar(&data).unwrap();
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::Timestamp));
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::DecimalAffine));
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::HexSuffix));
        assert!(grammar.stats.columns.iter().any(|column| column.kind == ModelKind::Dictionary));
    }

    #[test]
    fn escaped_nested_and_partial_records_remain_exact() {
        let data = b"{\"a\":\"tab\\tquote\\\"slash\\\\line\\n\",\"list\":[1,{\"k\":\"v\"}]}\n{\"a\":7";
        roundtrip(data);
    }

    #[test]
    fn malformed_packets_and_noncanonical_model_are_rejected() {
        let data = b"{\"id\":1,\"kind\":\"x\"}\n{\"id\":2,\"kind\":\"x\"}\n";
        let grammar = pack_grammar(data).unwrap();
        assert!(unpack_block(&grammar.bytes[..grammar.bytes.len() - 1]).is_err());
        let mut raw = raw_packet(data);
        raw.extend_from_slice(b"x");
        assert!(unpack_block(&raw).is_err());
        let raw_model = pack_raw_model(&[
            b"repeated-canonical-dictionary-value".as_slice(),
            b"repeated-canonical-dictionary-value".as_slice(),
        ]);
        assert!(unpack_model(&raw_model, 2).is_err());
    }

    #[test]
    fn periodic_plain_strings_are_packed_and_canonical() {
        let values = [
            b"\"abcabcabcabc\"".to_vec(),
            b"\"xyzxyzxyzxyzxyz\"".to_vec(),
            b"\"012012012012\"".to_vec(),
        ];
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::PeriodicString);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut malformed = packed;
        malformed[1] = 0;
        assert!(unpack_model(&malformed, 3).is_err());

        let hex_values = [
            b"\"0123456789abcde0123456789abcde\"".to_vec(),
            b"\"fedcba987654321fedcba987654321\"".to_vec(),
        ];
        let (packed, kind) = pack_model_values(&hex_values);
        assert_eq!(kind, ModelKind::PeriodicHex);
        assert_eq!(unpack_model(&packed, hex_values.len()).unwrap(), hex_values);
    }

    #[test]
    fn production_decode_accepts_nonminimal_but_valid_models() {
        let values = [
            b"repeated-canonical-dictionary-value".to_vec(),
            b"repeated-canonical-dictionary-value".to_vec(),
        ];
        let refs = values.iter().map(|value| value.as_slice()).collect::<Vec<_>>();
        let raw_model = pack_raw_model(&refs);
        // Strict oracle: rejects because a smaller canonical model exists.
        assert!(unpack_model(&raw_model, 2).is_err());
        // Production decode: structural checks pass; exact bytes are recovered.
        let decoded = unpack_model_mode(&raw_model, 2, false).unwrap().materialize();
        assert_eq!(decoded, values);
    }

    #[test]
    fn learned_gf2_affine_recurrence_roundtrips_a_held_out_suffix() {
        fn xorshift_word(mut value: u64) -> u64 {
            value ^= value << 13;
            value ^= value >> 7;
            value ^ (value << 17)
        }
        fn quoted_word(word: u64) -> Vec<u8> {
            let mut value = b"\"r-".to_vec();
            for byte in word.to_le_bytes() {
                value.push(HEX[(byte >> 4) as usize]);
                value.push(HEX[(byte & 15) as usize]);
            }
            value.push(b'"');
            value
        }
        let mut coefficients = [0u64; GF2_COEFFICIENTS];
        for (bit, coefficient) in coefficients[..64].iter_mut().enumerate() {
            *coefficient = xorshift_word(1u64 << bit);
        }
        coefficients[64] = 0x7265_6375_7272_656e;
        let words = std::iter::successors(Some(0x1234_5678_9abc_def0u64), |word| {
            Some(apply_affine(&coefficients, *word))
        })
        .take(128)
        .collect::<Vec<_>>();
        let values = words.into_iter().map(quoted_word).collect::<Vec<_>>();
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::Gf2Affine);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
    }

    #[test]
    fn evaluation_charges_independent_directory_and_times() {
        let data = (0..1024)
            .flat_map(|index| format!("{{\"id\":{index},\"kind\":\"event\"}}\n").into_bytes())
            .collect::<Vec<_>>();
        let mut evaluation = evaluate(&data, 4096).unwrap();
        assert_eq!(evaluation.input_bytes, data.len());
        assert_eq!(evaluation.frame_directory_bytes, 32 + evaluation.blocks * 32);
        evaluation.apply_timing(&time(&data, 4096, 1).unwrap());
        assert!(evaluation.pack_ns.is_some());
        assert!(evaluation.unpack_ns.is_some());
    }

    #[test]
    fn epoch_counter_ids_are_packed_and_canonical() {
        let mut values = Vec::new();
        for row in 0..300u32 {
            let epoch = row / 128;
            let prefix = 0xf2f1_3f80u32.wrapping_add(epoch.wrapping_mul(0x9e37_79b9));
            let suffix = if row % 97 == 3 {
                row.wrapping_mul(0x2545_f491)
            } else {
                row % 128
            };
            values.push(format!("\"r-{prefix:08x}{suffix:08x}\"").into_bytes());
        }
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::EpochCounter);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut malformed = packed.clone();
        let first = malformed[5..9].to_vec();
        malformed[9..13].copy_from_slice(&first);
        assert!(unpack_model(&malformed, values.len()).is_err());
    }

    #[test]
    fn segment_dictionary_messages_are_packed_and_canonical() {
        let phrases = [
            "queue depth exceeded",
            "retry scheduled",
            "batch committed",
            "cache flush complete",
            "worker heartbeat ok",
            "shard rebalanced",
            "index merge done",
            "page evicted",
            "commit retried",
            "drop detected",
            "sync paused",
            "threshold crossed",
        ];
        let mut values = Vec::new();
        for row in 0..192usize {
            let mut inner = String::new();
            let phrase_count = 1 + row % 6;
            for slot in 0..phrase_count {
                if slot != 0 {
                    inner.push_str("; ");
                }
                let choice = (row * 7 + slot * 5) % 20;
                if choice >= phrases.len() {
                    inner.push_str(&format!("{}", (row * 31 + slot) % 4096));
                } else {
                    inner.push_str(phrases[choice]);
                }
            }
            let message_len = 8 + (row * 13) % 40;
            if inner.len() < message_len {
                inner.push_str(&" ".repeat(message_len - inner.len()));
            }
            values.push(format!("\"{inner}\"").into_bytes());
        }
        let (packed, kind) = pack_model_values(&values);
        assert_eq!(kind, ModelKind::SegmentDict);
        assert_eq!(unpack_model(&packed, values.len()).unwrap(), values);
        let mut malformed = packed.clone();
        malformed.pop();
        assert!(unpack_model(&malformed, values.len()).is_err());
    }
}

