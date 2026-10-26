fn rng(s: &mut u64) -> u8 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s as u8
}

pub(super) fn datasets(n: usize) -> Vec<(String, Vec<u8>)> {
    let mut seed = 0x123456789abcdefu64;
    let random: Vec<u8> = (0..n).map(|_| rng(&mut seed)).collect();
    let low: Vec<u8> = (0..n).map(|_| rng(&mut seed) & 15).collect();
    let period: Vec<u8> = (0..n).map(|i| ((i % 37) * 7 + 3) as u8).collect();
    let mut json = Vec::with_capacity(n);
    let mut i = 0;
    while json.len() < n {
        json.extend_from_slice(
            format!(
                "{{\"tenant\":{},\"event\":\"checklist.completed\",\"item\":{},\"ok\":true,\"actor\":\"user-{}\"}}\n",
                i % 31,
                i,
                i % 103
            )
            .as_bytes(),
        );
        i += 1;
    }
    json.truncate(n);
    let mixed: Vec<u8> = (0..n)
        .map(|i| match (i / 4096) % 4 {
            0 => 0,
            1 => random[i],
            2 => json[i],
            _ => period[i],
        })
        .collect();
    vec![
        ("zeros".into(), vec![0; n]),
        ("period-37".into(), period),
        ("json-records".into(), json),
        ("random".into(), random),
        ("low-entropy-random".into(), low),
        ("mixed-4KiB".into(), mixed),
    ]
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn fill_repeated(chunk: &[u8], n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let take = (n - out.len()).min(chunk.len());
        out.extend_from_slice(&chunk[..take]);
    }
    out
}

pub(super) fn datasets_v4(n: usize) -> Vec<(String, Vec<u8>)> {
    // 1. telemetry.csv
    let mut csv_chunk = Vec::new();
    for row in 0..4096u32 {
        let epoch = row / 128;
        let prefix = 0x3B7F_2C4Du32.wrapping_add(epoch.wrapping_mul(0x2545_F491));
        let line = format!(
            "\"2026-01-{:02}T{:02}:{:02}:00Z\",{row},\"session.open\",\"eu-central-1\",\"r-{prefix:08x}{:08x}\",\"worker-{}\"\n",
            (row % 28) + 1,
            (row / 60) % 24,
            row % 60,
            row % 128,
            row % 16
        );
        csv_chunk.extend_from_slice(line.as_bytes());
    }
    let csv_data = fill_repeated(&csv_chunk, n);

    // 2. events.tsv
    let mut tsv_chunk = Vec::new();
    for row in 0..4096u32 {
        let line = format!(
            "{row}\t{}\t\"queue depth exceeded; retry scheduled; {}\"\t\"worker-{}\"\t\"ok\"\n",
            100_000 + row * 5,
            row % 128,
            row % 8
        );
        tsv_chunk.extend_from_slice(line.as_bytes());
    }
    let tsv_data = fill_repeated(&tsv_chunk, n);

    // 3. app-logs.logfmt
    let mut logfmt_chunk = Vec::new();
    for row in 0..4096u32 {
        let epoch = row / 128;
        let prefix = 0x3B7F_2C4Du32.wrapping_add(epoch.wrapping_mul(0x2545_F491));
        let line = format!(
            "ts=\"2026-01-{:02}T{:02}:{:02}:00Z\" level=info seq={row} req_id=\"r-{prefix:08x}{:08x}\" region=\"eu-1\" worker=\"worker-{}\"\n",
            (row % 28) + 1,
            (row / 60) % 24,
            row % 60,
            row % 128,
            row % 12
        );
        logfmt_chunk.extend_from_slice(line.as_bytes());
    }
    let logfmt_data = fill_repeated(&logfmt_chunk, n);

    // 4. json-records.ndjson
    let mut json_rec = Vec::with_capacity(n);
    let mut idx = 0u64;
    while json_rec.len() < n {
        let s = format!(
            "{{\"tenant\":{},\"event\":\"checklist.completed\",\"item\":{idx},\"ok\":true,\"actor\":\"user-{}\"}}\n",
            idx % 31,
            idx % 103
        );
        let b = s.as_bytes();
        let take = (n - json_rec.len()).min(b.len());
        json_rec.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 5. field-order-optional-escaping.ndjson
    let mut field_order = Vec::with_capacity(n);
    let seed_fo = 0x8D12_4A77_319B_05E1u64;
    idx = 0;
    while field_order.len() < n {
        let state = xorshift64(seed_fo ^ idx);
        let tenant = state % 97;
        let actor = (state >> 9) % 1009;
        let escaped = if idx % 5 == 0 {
            "tab\\tquote\\\"slash\\\\line\\n"
        } else {
            "plain"
        };
        let optional = if idx % 3 != 0 {
            format!(",\"retry\":{},\"note\":\"{escaped}\"", (state >> 17) % 4)
        } else {
            String::new()
        };
        let rec = if idx & 1 != 0 {
            format!("{{\"actor\":\"user-{actor}\",\"tenant\":{tenant}{optional},\"event\":\"task.update\",\"id\":{idx}}}\n")
        } else {
            format!("{{\"event\":\"task.update\",\"id\":{idx},\"tenant\":{tenant}{optional},\"actor\":\"user-{actor}\"}}\n")
        };
        let b = rec.as_bytes();
        let take = (n - field_order.len()).min(b.len());
        field_order.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 6. event-mix-identifiers.ndjson
    let mut event_mix = Vec::with_capacity(n);
    let seed_em = 0x3B7F_2C4D_9A18_56E7u64;
    let events = ["session.open", "session.close", "build.finished", "alert.raised"];
    let regions = ["eu-1", "us-2", "ap-1"];
    idx = 0;
    while event_mix.len() < n {
        let state = xorshift64(seed_em ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let event = events[(state & 3) as usize];
        let region = regions[((state >> 3) % 3) as usize];
        let status = if state & 0x40 != 0 { "ok" } else { "warning" };
        let epoch = idx / 128;
        let prefix = (xorshift64(seed_em ^ epoch.wrapping_mul(0x2545_F491_4F6C_DD1D)) & 0xFFFF_FFFF) as u32;
        let sequence = if state % 32 == 0 {
            ((state >> 16) & 0xFFFF_FFFF) as u32
        } else {
            (idx % 128) as u32
        };
        let rec = format!(
            "{{\"ts\":\"2026-01-{:02}T{:02}:00:00Z\",\"event\":\"{event}\",\"region\":\"{region}\",\"request_id\":\"r-{prefix:08x}{sequence:08x}\",\"status\":\"{status}\",\"sequence\":{idx}}}\n",
            (idx % 28) + 1,
            idx % 24
        );
        let b = rec.as_bytes();
        let take = (n - event_mix.len()).min(b.len());
        event_mix.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 7. bursty-variable-records.ndjson
    let mut bursty = Vec::with_capacity(n);
    let seed_br = 0x6D29_E4B8_1F73_A350u64;
    const PHRASES: [&str; 12] = [
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
    const VOCAB: [&str; 16] = [
        "flush", "queue", "retry", "timeout", "worker", "ack", "batch", "sync", "drop", "merge",
        "cache", "page", "index", "shard", "commit", "warn",
    ];
    idx = 0;
    while bursty.len() < n {
        let state = xorshift64(seed_br ^ idx.wrapping_mul(0xD134_2543_DE82_EF95));
        let burst = (idx / 19) % 7;
        let msg_len = (8 + (state % if burst < 5 { 24 } else { 240 })) as usize;
        let tags = if burst <= 1 { "[\"hot\",\"retry\"]" } else { "[]" };
        let mut msg = String::with_capacity(msg_len);
        let mut pos = 0usize;
        loop {
            let phrase = PHRASES[((state >> ((pos * 7) % 57)) as usize) % PHRASES.len()];
            let extra = phrase.len() + if msg.is_empty() { 0 } else { 2 };
            if msg.len() + extra > msg_len {
                break;
            }
            if !msg.is_empty() {
                msg.push_str("; ");
            }
            msg.push_str(phrase);
            pos += 1;
        }
        pos = 0;
        loop {
            let tok_buf;
            let token: &str = if ((state >> (13 + (pos % 5))) & 3) == 0 {
                tok_buf = format!("{}", state % 4096);
                &tok_buf
            } else {
                VOCAB[((state >> ((pos * 5) % 59)) & 15) as usize]
            };
            let extra = token.len() + if msg.is_empty() { 0 } else { 2 };
            if msg.len() + extra > msg_len {
                break;
            }
            if !msg.is_empty() {
                msg.push_str("; ");
            }
            msg.push_str(token);
            pos += 1;
        }
        while msg.len() < msg_len {
            msg.push(' ');
        }
        let rec = format!(
            "{{\"burst\":{burst},\"record\":{idx},\"tags\":{tags},\"message\":\"{msg}\",\"source\":\"worker-{}\"}}\n",
            state % 23
        );
        let b = rec.as_bytes();
        let take = (n - bursty.len()).min(b.len());
        bursty.extend_from_slice(&b[..take]);
        idx += 1;
    }

    // 8. source-and-docs.txt
    let mut src_seed = Vec::new();
    for slice in [
        include_bytes!("../../analysis.rs") as &[u8],
        include_bytes!("../../block.rs"),
        include_bytes!("../../block/decode.rs"),
        include_bytes!("../../frame.rs"),
        include_bytes!("../../frame/decode.rs"),
        include_bytes!("../../frame/assemble.rs"),
        include_bytes!("../../lib.rs"),
        include_bytes!("../../simd.rs"),
        include_bytes!("../../../docs/FORMAT.md"),
        include_bytes!("../../../docs/BENCHMARKS.md"),
    ] {
        src_seed.extend_from_slice(slice);
        src_seed.push(b'\n');
    }
    let src_docs = fill_repeated(&src_seed, n);

    // 9. ui-render-32bpp.bmp
    let mut bmp32 = vec![0u8; n.max(65536)];
    bmp32[0..2].copy_from_slice(b"BM");
    let bmp_len = n as u32;
    bmp32[2..6].copy_from_slice(&bmp_len.to_le_bytes());
    bmp32[10..14].copy_from_slice(&54u32.to_le_bytes());
    bmp32[14..18].copy_from_slice(&40u32.to_le_bytes());
    bmp32[18..22].copy_from_slice(&2048i32.to_le_bytes());
    bmp32[22..26].copy_from_slice(&2048i32.to_le_bytes());
    bmp32[26..28].copy_from_slice(&1u16.to_le_bytes());
    bmp32[28..30].copy_from_slice(&32u16.to_le_bytes());
    let mut tile32 = [0u8; 65536];
    for px in 0..16384usize {
        let band = ((px / 256) & 0xFF) as u8;
        let edge = if px % 256 == 0 {
            ((px / 17) & 0xFF) as u8
        } else {
            0
        };
        let base = px * 4;
        tile32[base] = band.wrapping_mul(13).wrapping_add(edge);
        tile32[base + 1] = band.wrapping_mul(29);
        tile32[base + 2] = band.wrapping_mul(47).wrapping_sub(edge);
        tile32[base + 3] = 255;
    }
    bmp32[54..65536].copy_from_slice(&tile32[54..65536]);
    let mut off = 65536;
    while off < bmp32.len() {
        let end = (off + 65536).min(bmp32.len());
        bmp32[off..end].copy_from_slice(&tile32[..end - off]);
        off += 65536;
    }
    bmp32.truncate(n);

    // 10. rgb-frame-24bpp.bmp
    let mut bmp24 = vec![0u8; n.max(65536)];
    bmp24[0..2].copy_from_slice(b"BM");
    bmp24[2..6].copy_from_slice(&bmp_len.to_le_bytes());
    bmp24[10..14].copy_from_slice(&54u32.to_le_bytes());
    bmp24[14..18].copy_from_slice(&40u32.to_le_bytes());
    bmp24[18..22].copy_from_slice(&2360i32.to_le_bytes());
    bmp24[22..26].copy_from_slice(&2360i32.to_le_bytes());
    bmp24[26..28].copy_from_slice(&1u16.to_le_bytes());
    bmp24[28..30].copy_from_slice(&24u16.to_le_bytes());
    let mut tile24 = [0u8; 65536];
    for px in 0..21845usize {
        let run = ((px / 256) & 0xFF) as u8;
        let edge = u8::from(px % 256 == 127);
        let base = px * 3;
        tile24[base] = run.wrapping_mul(7);
        tile24[base + 1] = run.wrapping_mul(19).wrapping_add(edge);
        tile24[base + 2] = run.wrapping_mul(37);
    }
    bmp24[54..65536].copy_from_slice(&tile24[54..65536]);
    off = 65536;
    while off < bmp24.len() {
        let end = (off + 65536).min(bmp24.len());
        bmp24[off..end].copy_from_slice(&tile24[..end - off]);
        off += 65536;
    }
    bmp24.truncate(n);

    vec![
        ("telemetry.csv".into(), csv_data),
        ("events.tsv".into(), tsv_data),
        ("app-logs.logfmt".into(), logfmt_data),
        ("json-records.ndjson".into(), json_rec),
        ("field-order-optional-escaping.ndjson".into(), field_order),
        ("event-mix-identifiers.ndjson".into(), event_mix),
        ("bursty-variable-records.ndjson".into(), bursty),
        ("source-and-docs.txt".into(), src_docs),
        ("ui-render-32bpp.bmp".into(), bmp32),
        ("rgb-frame-24bpp.bmp".into(), bmp24),
    ]
}
