use std::hint::black_box;
use std::time::Instant;
use vxg::{DecodeOptions, Options, RowMatch};

fn build_ndjson_dataset(rows: usize) -> Vec<u8> {
    let services = [
        "auth-gateway",
        "edge-router",
        "checkout-api",
        "catalog-index",
        "search-ranker",
        "fraud-engine",
        "session-cache",
        "profile-store",
    ];
    let regions = ["eu-central-1", "us-east-1", "ap-northeast-1", "us-west-2"];
    let levels = ["INFO", "INFO", "INFO", "WARN", "INFO", "DEBUG", "INFO", "ERROR"];
    let mut out = Vec::with_capacity(rows * 175);
    for i in 0..rows {
        let svc = if i == (rows * 9) / 10 {
            "billing-canary-incident"
        } else {
            services[i & 7]
        };
        let region = regions[(i >> 3) & 3];
        let level = if i % 2500 == 1999 {
            "CRITICAL"
        } else {
            levels[(i >> 1) & 7]
        };
        let status = if i == (rows * 9) / 10 {
            503
        } else if level == "CRITICAL" {
            500
        } else {
            200
        };
        let sec = (i / 1000) % 60;
        let ms = i % 1000;
        let lat = 12 + ((i * 17) & 127);
        let line = format!(
            "{{\"ts\":\"2026-10-27T09:15:{sec:02}.{ms:03}Z\",\"level\":\"{level}\",\"service\":\"{svc}\",\"region\":\"{region}\",\"req_id\":\"req-{:08x}\",\"status\":{status},\"latency_ms\":{lat},\"bytes\":{}}}\n",
            0x10a0_0000usize + i,
            512 + ((i * 31) & 2047)
        );
        out.extend_from_slice(line.as_bytes());
    }
    out
}

fn raw_row_count(raw: &[u8]) -> usize {
    if raw.is_empty() {
        return 0;
    }
    let n = raw.iter().filter(|&&b| b == b'\n').count();
    if *raw.last().unwrap() == b'\n' {
        n
    } else {
        n + 1
    }
}

fn raw_extract_row_range(raw: &[u8], start_row: usize, end_row: usize) -> Vec<u8> {
    let mut cur_row = 0usize;
    let mut start_byte = if start_row == 0 { Some(0usize) } else { None };
    let mut end_byte = None;
    for (idx, &b) in raw.iter().enumerate() {
        if b == b'\n' {
            cur_row += 1;
            if start_byte.is_none() && cur_row == start_row {
                start_byte = Some(idx + 1);
            }
            if cur_row == end_row {
                end_byte = Some(idx + 1);
                break;
            }
        }
    }
    let s = start_byte.unwrap_or(raw.len());
    let e = end_byte.unwrap_or(raw.len());
    raw[s..e].to_vec()
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    let first = needle[0];
    let limit = haystack.len() - needle.len() + 1;
    let mut pos = 0usize;
    while pos < limit {
        let Some(rel) = haystack[pos..limit].iter().position(|&b| b == first) else {
            return false;
        };
        let idx = pos + rel;
        if &haystack[idx..idx + needle.len()] == needle {
            return true;
        }
        pos = idx + 1;
    }
    false
}

fn raw_filter_rows(raw: &[u8], needle: &[u8], limit: usize) -> Vec<RowMatch> {
    let mut out = Vec::new();
    let mut line_start = 0usize;
    let mut row_idx = 0usize;
    for (i, &b) in raw.iter().enumerate() {
        if b == b'\n' {
            let line = &raw[line_start..=i];
            if contains_subslice(line, needle) {
                out.push(RowMatch {
                    block_index: 0,
                    row_index: row_idx,
                    global_row_index: row_idx,
                    bytes: line.to_vec(),
                });
                if out.len() >= limit {
                    break;
                }
            }
            line_start = i + 1;
            row_idx += 1;
        }
    }
    out
}

fn bench_op<T, F: FnMut() -> T>(iters: usize, mut f: F) -> (f64, T) {
    // Warmup
    let mut last = f();
    let start = Instant::now();
    for _ in 0..iters {
        last = black_box(f());
    }
    let elapsed = start.elapsed().as_secs_f64() / (iters as f64);
    (elapsed, last)
}

fn main() {
    let rows = 400_000usize;
    let raw = build_ndjson_dataset(rows);
    let opts = Options {
        checksum: false,
        ..Options::default()
    };
    let dec_opts = DecodeOptions {
        verify_checksum: false,
        ..DecodeOptions::default()
    };
    let compressed = vxg::compress(&raw, &opts).expect("vxg compress");
    let raw_mib = raw.len() as f64 / (1024.0 * 1024.0);
    let comp_mib = compressed.len() as f64 / (1024.0 * 1024.0);
    let ratio = (compressed.len() as f64 / raw.len() as f64) * 100.0;

    println!("=========================================================================================");
    println!("VXG Compressed-Domain Query vs. Raw Uncompressed Scan Benchmark");
    println!(
        "Dataset: {} NDJSON records | Raw: {:.2} MiB ({} B) | VXG v4: {:.2} MiB ({} B, {:.2}%)",
        rows,
        raw_mib,
        raw.len(),
        comp_mib,
        compressed.len(),
        ratio
    );
    println!("=========================================================================================");

    // 1. Row Count
    let (t_raw_cnt, c1) = bench_op(50, || raw_row_count(black_box(&raw)));
    let (t_vxg_cnt, c2) = bench_op(500, || vxg::frame_row_count(black_box(&compressed)).unwrap());
    assert_eq!(c1, c2);

    // 2. Random Access Single Row (Row 360,000 - at 90% of file)
    let target_row = 360_000usize;
    let (t_raw_row, r1) = bench_op(50, || raw_extract_row_range(black_box(&raw), target_row, target_row + 1));
    let (t_vxg_row, r2) = bench_op(500, || vxg::extract_row(black_box(&compressed), target_row, &dec_opts).unwrap());
    assert_eq!(r1, r2);

    // 3. Random Access Row Range [250,000 .. 250,100) (100 records across blocks)
    let (t_raw_rng, rng1) = bench_op(50, || raw_extract_row_range(black_box(&raw), 250_000, 250_100));
    let (t_vxg_rng, rng2) = bench_op(500, || vxg::extract_row_range(black_box(&compressed), 250_000, 250_100, &dec_opts).unwrap());
    assert_eq!(rng1, rng2);

    // 4. find_first (Rare Incident Needle at 90% of dataset: row 360,000)
    let needle_rare = b"billing-canary-incident";
    let (t_raw_find, f1) = bench_op(30, || raw_filter_rows(black_box(&raw), needle_rare, 1));
    let (t_vxg_find, f2) = bench_op(200, || vxg::find_first(black_box(&compressed), needle_rare, &dec_opts).unwrap());
    assert_eq!(f1[0].global_row_index, f2.as_ref().unwrap().global_row_index);
    assert_eq!(f1[0].bytes, f2.as_ref().unwrap().bytes);

    // 5. find_first (Absent Needle - Alphabet / Dictionary Rejection across 100% of dataset)
    let needle_absent = b"ZERO_DAY_EXPLOIT_999";
    let (t_raw_abs, a1) = bench_op(30, || raw_filter_rows(black_box(&raw), needle_absent, 1));
    let (t_vxg_abs, a2) = bench_op(200, || vxg::find_first(black_box(&compressed), needle_absent, &dec_opts).unwrap());
    assert!(a1.is_empty() && a2.is_none());

    // 6. filter_rows (Selective Filter: "CRITICAL" -> 160 matches out of 400,000 rows)
    let needle_crit = b"CRITICAL";
    let (t_raw_flt, flt1) = bench_op(30, || raw_filter_rows(black_box(&raw), needle_crit, 1000));
    let (t_vxg_flt, flt2) = bench_op(100, || vxg::filter_rows(black_box(&compressed), needle_crit, 1000, &dec_opts).unwrap());
    assert_eq!(flt1.len(), flt2.len());
    for (m1, m2) in flt1.iter().zip(flt2.iter()) {
        assert_eq!(m1.global_row_index, m2.global_row_index);
        assert_eq!(m1.bytes, m2.bytes);
    }

    // 7. Full Decompress + Raw Scan baseline (what traditional codecs must do)
    let (t_full_dec, _) = bench_op(30, || vxg::decompress(black_box(&compressed), &dec_opts).unwrap());

    let rows_gb = raw.len() as f64 / 1e9;
    println!(
        "{:<38} | {:>12} | {:>12} | {:>12} | {:>10}",
        "Operation", "Raw Scan", "Decomp+Scan", "VXG Query", "Speedup"
    );
    println!("{:-<94}", "");
    for (name, t_raw, t_vxg, scale) in [
        ("1. frame_row_count (400k rows)", t_raw_cnt, t_vxg_cnt, 1.0),
        ("2. extract_row (row #360,000)", t_raw_row, t_vxg_row, 0.9),
        ("3. extract_row_range [250k..250.1k)", t_raw_rng, t_vxg_rng, 0.625),
        ("4. find_first (rare @ row #360,000)", t_raw_find, t_vxg_find, 0.9),
        ("5. find_first (absent needle)", t_raw_abs, t_vxg_abs, 1.0),
        ("6. filter_rows (\"CRITICAL\", 160 hits)", t_raw_flt, t_vxg_flt, 1.0),
    ] {
        let t_decomp_scan = t_full_dec * scale + t_raw;
        let speedup_vs_raw = t_raw / t_vxg;
        let speedup_vs_dec = t_decomp_scan / t_vxg;
        let eff_gbps = (rows_gb * scale) / t_vxg;
        println!(
            "{:<38} | {:>9.3} ms | {:>9.3} ms | {:>9.3} ms | {:>5.1}x ({:.1}x vs dec) [{:.1} GB/s]",
            name,
            t_raw * 1e3,
            t_decomp_scan * 1e3,
            t_vxg * 1e3,
            speedup_vs_raw,
            speedup_vs_dec,
            eff_gbps
        );
    }
}
