//! Automated CI compression ratio regression gate (`<= 1.0%` tolerance).
//!
//! Guards non-repeating structured, cumulative time-series, HTTP log, free-text log,
//! and genuine Rust/Markdown source corpora against silent compression-ratio regressions
//! (such as the `v0.5.0 -> v0.5.1` `BLOCK_DICT_LZ` ratio drop).

use vxg::{compress, decompress, DecodeOptions, Options};

fn non_repeating_source_corpus() -> Vec<u8> {
    let mut out = Vec::with_capacity(200_000);
    for slice in [
        include_bytes!("../src/frame.rs") as &[u8],
        include_bytes!("../src/frame/decode.rs"),
        include_bytes!("../src/frame/assemble.rs"),
        include_bytes!("../src/columns/model_dict.rs"),
        include_bytes!("../src/columns/model_pattern.rs"),
        include_bytes!("../src/dict_lz/decode.rs"),
        include_bytes!("../docs/FORMAT.md"),
        include_bytes!("../docs/SAFETY.md"),
    ] {
        out.extend_from_slice(slice);
        out.push(b'\n');
    }
    out
}

fn non_repeating_clf_corpus() -> Vec<u8> {
    let paths = [
        "/api/v2/checkout/confirm",
        "/static/assets/bundle.8f91a2.js",
        "/v1/telemetry/ingest?shard=04",
        "/auth/oauth2/callback?provider=github",
        "/graphql/persisted/991820a",
        "/healthz/readiness",
    ];
    let agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 Version/17.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:131.0) Gecko/20100101 Firefox/131.0",
        "curl/8.7.1",
        "go-http-client/2.0",
    ];
    let mut out = Vec::with_capacity(200_000);
    let mut s = 0x71e5_92a3_u64;
    for i in 0..1500u32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let ip_c = ((s >> 16) & 0xff) as u8;
        let ip_d = (((s >> 24) & 0xfe) | 1) as u8;
        let path = paths[((s >> 32) as usize) % paths.len()];
        let ua = agents[((s >> 40) as usize) % agents.len()];
        let status = [200u16, 200, 200, 201, 304, 404, 502][((s >> 48) as usize) % 7];
        let bytes = 128 + ((s >> 51) % 32_768);
        out.extend_from_slice(
            format!(
                "198.51.{ip_c}.{ip_d} - - [09/Nov/2026:10:{:02}:{:02} +0000] \"GET {path} HTTP/1.1\" {status} {bytes} \"-\" \"{ua}\"\n",
                (i / 60) % 60,
                i % 60,
            )
            .as_bytes(),
        );
    }
    out
}

fn cumulative_timeseries_csv_corpus() -> Vec<u8> {
    let countries = [
        ("Germany", 120_000u64, 410u64),
        ("United States", 24_500_000, 3_200),
        ("Uruguay", 15, 6),
        ("Japan", 1_900_000, 520),
        ("Iceland", 0, 2),
        ("Brazil", 14_200_000, 1_850),
        ("Finland", 42_000, 55),
        ("India", 31_000_000, 4_100),
    ];
    let mut out = Vec::from(&b"Date,Country/Region,Province/State,Confirmed,Recovered,Deaths\n"[..]);
    let mut s = 0x1357_9bdf_u64;
    for &(country, base_c, daily_step) in &countries {
        let mut conf = base_c;
        let mut rec = base_c * 9 / 10;
        let mut deaths = base_c / 80;
        for day in 0..420u32 {
            s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            conf += (s >> 52) % (daily_step + 1);
            rec += (s >> 54) % (daily_step + 1);
            deaths += (s >> 60) % 7;
            let m = (day / 28) % 12 + 1;
            let d = (day % 28) + 1;
            let y = 2020 + (day / 336);
            out.extend_from_slice(
                format!("{y}-{m:02}-{d:02},{country},,{conf},{rec},{deaths}\n").as_bytes(),
            );
        }
    }
    out
}

fn free_text_daemon_logs_corpus() -> Vec<u8> {
    let msgs = [
        "kernel: memorystatus_thread: idle exiting pid",
        "systemd-journald[411]: Suppressed 184 messages from /system.slice/docker.service",
        " kubelet[1922]: E1109 10:15:22.119402 pod_workers.go:1298] Error syncing pod",
        "postgres[9021]: [4-1] LOG:  automatic vacuum of table \"prod.public.events\": index scans: 1",
        "sshd[31004]: Invalid user admin from 203.0.113.",
    ];
    let mut out = Vec::with_capacity(180_000);
    let mut s = 0xfeed_face_cafe_beef_u64;
    for i in 0..2200u32 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let m = msgs[((s >> 32) as usize) % msgs.len()];
        out.extend_from_slice(
            format!(
                "Nov  9 10:{:02}:{:02} prod-node-{:02} {m} id={} latency_us={}\n",
                (i / 60) % 60,
                i % 60,
                (s >> 40) % 16,
                (s >> 16) & 0xffff_ffff,
                (s >> 48) % 25_000,
            )
            .as_bytes(),
        );
    }
    out
}

#[test]
fn compression_ratio_regression_gate_1pct() {
    // Pinned v0.5.1/v0.5.2 encoded byte ceilings (with 1.0% max regression tolerance).
    // If any codec change degrades ratio by > 1.0% on any corpus class, fail CI immediately.
    let cases: [(&str, Vec<u8>, usize); 4] = [
        ("non-repeating-rust-and-docs", non_repeating_source_corpus(), 35_030),
        ("apache-clf-access-log", non_repeating_clf_corpus(), 13_279),
        ("cumulative-timeseries-csv", cumulative_timeseries_csv_corpus(), 14_147),
        ("free-text-daemon-log", free_text_daemon_logs_corpus(), 34_156),
    ];

    let enc_opts = Options::default();
    let dec_opts = DecodeOptions::default();
    let mut failures = Vec::new();

    for (name, raw, baseline_bytes) in &cases {
        let encoded = compress(raw, &enc_opts).unwrap();
        let decoded = decompress(&encoded, &dec_opts).unwrap();
        assert_eq!(&decoded, raw, "roundtrip mismatch on {name}");

        let max_allowed = (*baseline_bytes as f64 * 1.01).ceil() as usize;
        let ratio = encoded.len() as f64 / raw.len() as f64;
        let base_ratio = *baseline_bytes as f64 / raw.len() as f64;
        eprintln!(
            "[ratio-gate] {name:30} raw={:7} enc={:6} ratio={ratio:.4} baseline={baseline_bytes} (limit={max_allowed})",
            raw.len(),
            encoded.len()
        );
        if encoded.len() > max_allowed {
            failures.push(format!(
                "RATIO REGRESSION > 1.0% on `{name}`: encoded={} bytes (ratio={ratio:.4}) > ceiling={max_allowed} bytes (baseline={} bytes, {base_ratio:.4})",
                encoded.len(),
                baseline_bytes
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
