//! Real-world compressed-domain query benchmark.
//!
//! Usage: `vxg-query-real <file> <needle> [<needle> ...]`
//!
//! Compresses `<file>` with VXG v4, then for every needle compares
//! `filter_rows`/`find_first` against a plain raw scan, asserting that the
//! returned rows (index + bytes) are byte-identical, and prints timings.

use std::hint::black_box;
use std::time::Instant;
use vxg::{DecodeOptions, NumCmp, Options};

fn find_sub(h: &[u8], n: &[u8]) -> bool {
    if n.is_empty() {
        return true;
    }
    h.windows(n.len()).any(|w| w == n)
}

fn raw_filter(raw: &[u8], needle: &[u8], limit: usize) -> Vec<(usize, Vec<u8>)> {
    let mut out = Vec::new();
    for (i, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if find_sub(line, needle) {
            out.push((i, line.to_vec()));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

/// Independent reference for `@num:` needles: first `"key":` in the line, plain unsigned digits.
fn raw_num_filter(raw: &[u8], key: &str, cmp: NumCmp, limit: usize) -> Vec<(usize, Vec<u8>)> {
    let marker = format!("\"{key}\":").into_bytes();
    let mut out = Vec::new();
    for (i, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        let Some(p) = line.windows(marker.len()).position(|w| w == marker.as_slice()) else {
            continue;
        };
        let rest = &line[p + marker.len()..];
        let n = rest.iter().take_while(|b| b.is_ascii_digit()).count();
        if n == 0 || n > 19 || (n > 1 && rest[0] == b'0') || matches!(rest.get(n), Some(b'.' | b'e' | b'E')) {
            continue;
        }
        let v: u64 = std::str::from_utf8(&rest[..n]).unwrap().parse().unwrap();
        let hit = cmp.test(v);
        if hit {
            out.push((i, line.to_vec()));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn parse_num_spec(spec: &str) -> Option<(String, NumCmp)> {
    // @num:key:eq:V | @num:key:ge:V | @num:key:le:V | @num:key:range:LO-HI
    let mut it = spec.strip_prefix("@num:")?.split(':');
    let key = it.next()?.to_string();
    let op = it.next()?;
    let val = it.next()?;
    let cmp = match op {
        "eq" => NumCmp::Eq(val.parse().ok()?),
        "ge" => NumCmp::Ge(val.parse().ok()?),
        "le" => NumCmp::Le(val.parse().ok()?),
        "range" => {
            let (a, b) = val.split_once('-')?;
            NumCmp::Range(a.parse().ok()?, b.parse().ok()?)
        }
        _ => return None,
    };
    Some((key, cmp))
}

fn time<T>(iters: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f();
    let t = Instant::now();
    for _ in 0..iters {
        last = black_box(f());
    }
    (t.elapsed().as_secs_f64() / iters as f64, last)
}

fn median_secs<T>(samples: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = black_box(f());
    let mut durs = Vec::with_capacity(samples);
    for _ in 0..samples {
        let t = Instant::now();
        last = black_box(f());
        durs.push(t.elapsed().as_secs_f64());
    }
    durs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (durs[samples / 2], last)
}

fn naive_filter_from_frame(
    frame: &[u8],
    pred: &vxg::ColPred<'_>,
    dec: &DecodeOptions,
) -> Vec<(usize, Vec<u8>)> {
    let raw = vxg::decompress(frame, dec).unwrap();
    let mut out = Vec::new();
    for (idx, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if vxg::ops::line_matches_col_pred(line, pred) {
            out.push((idx, line.to_vec()));
        }
    }
    out
}

fn naive_filter_frame(
    frame: &[u8],
    pred: &vxg::ColPred<'_>,
    enc: &Options,
    dec: &DecodeOptions,
) -> Vec<u8> {
    let raw = vxg::decompress(frame, dec).unwrap();
    let mut filtered = Vec::with_capacity(raw.len() / 8);
    for line in raw.split_inclusive(|&b| b == b'\n') {
        if vxg::ops::line_matches_col_pred(line, pred) {
            filtered.extend_from_slice(line);
        }
    }
    vxg::compress(&filtered, enc).unwrap()
}

fn naive_transform_frame(
    frame: &[u8],
    where_pred: Option<&vxg::ColPred<'_>>,
    target_col: vxg::ColTarget<'_>,
    op: &vxg::ColOp<'_>,
    enc: &Options,
    dec: &DecodeOptions,
) -> Vec<u8> {
    let raw = vxg::decompress(frame, dec).unwrap();
    let mut out = Vec::with_capacity(raw.len() + 4096);
    for line in raw.split_inclusive(|&b| b == b'\n') {
        let hit = where_pred.map_or(true, |p| vxg::ops::line_matches_col_pred(line, p));
        if hit {
            if let Some((s, e)) = vxg::ops::find_line_col_span(line, target_col) {
                if let Some(tok) = vxg::ops::apply_token_op(target_col, &line[s..e], op) {
                    out.extend_from_slice(&line[..s]);
                    out.extend_from_slice(&tok);
                    out.extend_from_slice(&line[e..]);
                    continue;
                }
            }
        }
        out.extend_from_slice(line);
    }
    vxg::compress(&out, enc).unwrap()
}

fn run_ops_suite(raw: &[u8], frame: &[u8], kind: &str, opts: &Options, dec: &DecodeOptions) {
    use vxg::{ColOp, ColPred, ColTarget, StrCmp};
    let gb = raw.len() as f64 / 1e9;
    println!(
        "\n=== Stage 2 Compressed-Domain Ops ({kind}) — threads={}, median of 5 samples ===",
        opts.threads
    );
    println!(
        "{:<44} {:>11} {:>11} {:>9} {:>11}",
        "operation", "naive ms", "vxg ms", "speedup", "vxg GB/s"
    );

    let bench_filter = |label: &str, pred: ColPred<'_>| {
        let (t_naive, exp) = median_secs(5, || naive_filter_from_frame(frame, &pred, dec));
        let (t_vxg, got) =
            median_secs(5, || vxg::filter_column(frame, &pred, usize::MAX, dec).unwrap());
        assert_eq!(exp.len(), got.len(), "{label}: hit count mismatch");
        for ((ri, rb), b) in exp.iter().zip(got.iter()) {
            assert_eq!(*ri, b.global_row_index);
            assert_eq!(rb, &b.bytes);
        }
        println!(
            "{:<44} {:>11.2} {:>11.2} {:>8.1}x {:>9.2}   (hits={})",
            label,
            t_naive * 1e3,
            t_vxg * 1e3,
            t_naive / t_vxg,
            gb / t_vxg,
            got.len()
        );
    };

    let mut dec_out = dec.clone();
    dec_out.max_output = raw.len().saturating_mul(2).max(1 << 20);

    let bench_frame_op =
        |label: &str, naive_f: &mut dyn FnMut() -> Vec<u8>, vxg_f: &mut dyn FnMut() -> Vec<u8>| {
            let (t_naive, f_naive) = median_secs(5, &mut *naive_f);
            let (t_vxg, f_vxg) = median_secs(5, &mut *vxg_f);
            let r_naive = vxg::decompress(&f_naive, &dec_out).unwrap();
            let r_vxg = vxg::decompress(&f_vxg, &dec_out).unwrap();
            assert_eq!(r_naive, r_vxg, "{label}: decompressed output mismatch");
            println!(
                "{:<44} {:>11.2} {:>11.2} {:>8.1}x {:>9.2}   (out_ratio={:.3})",
                label,
                t_naive * 1e3,
                t_vxg * 1e3,
                t_naive / t_vxg,
                gb / t_vxg,
                if r_vxg.is_empty() {
                    0.0
                } else {
                    f_vxg.len() as f64 / r_vxg.len() as f64
                }
            );
        };

    let upper = |b: &[u8]| b.to_ascii_uppercase();
    let add100 = |v: u64| v.saturating_add(100);

    if kind == "ndjson" {
        bench_filter(
            "filter_col level==NOPE (0-hit alphabet)",
            ColPred::json_str("level", StrCmp::Eq(b"NOPE")),
        );
        bench_filter(
            "filter_col bytes>=50000 (6-hit zone-map)",
            ColPred::json_num("bytes", NumCmp::Ge(50_000)),
        );
        bench_filter(
            "filter_col latency_ms>=200 (numeric)",
            ColPred::json_num("latency_ms", NumCmp::Ge(200)),
        );
        bench_filter(
            "filter_col service==fraud-engine (dict)",
            ColPred::json_str("service", StrCmp::Eq(b"fraud-engine")),
        );

        let p_500 = ColPred::json_num("status", NumCmp::Eq(500));
        bench_frame_op(
            "filter_frame status==500 (~5% rows)",
            &mut || naive_filter_frame(frame, &p_500, opts, dec),
            &mut || vxg::filter_frame(frame, &p_500, opts, dec).unwrap(),
        );

        let p_nope = ColPred::json_str("level", StrCmp::Eq(b"NOPE"));
        let col_lvl = ColTarget::JsonKey("level");
        bench_frame_op(
            "update WHERE level==NOPE SET CRITICAL",
            &mut || {
                naive_transform_frame(
                    frame,
                    Some(&p_nope),
                    col_lvl,
                    &ColOp::SetStr(b"CRITICAL"),
                    opts,
                    dec,
                )
            },
            &mut || {
                vxg::update_column(frame, &p_nope, col_lvl, ColOp::SetStr(b"CRITICAL"), opts, dec)
                    .unwrap()
            },
        );

        let p_lat = ColPred::json_num("latency_ms", NumCmp::Ge(250));
        bench_frame_op(
            "update WHERE latency_ms>=250 SET CRITICAL",
            &mut || {
                naive_transform_frame(
                    frame,
                    Some(&p_lat),
                    col_lvl,
                    &ColOp::SetStr(b"CRITICAL"),
                    opts,
                    dec,
                )
            },
            &mut || {
                vxg::update_column(frame, &p_lat, col_lvl, ColOp::SetStr(b"CRITICAL"), opts, dec)
                    .unwrap()
            },
        );

        bench_frame_op(
            "update WHERE status==500 SET CRITICAL",
            &mut || {
                naive_transform_frame(
                    frame,
                    Some(&p_500),
                    col_lvl,
                    &ColOp::SetStr(b"CRITICAL"),
                    opts,
                    dec,
                )
            },
            &mut || {
                vxg::update_column(frame, &p_500, col_lvl, ColOp::SetStr(b"CRITICAL"), opts, dec)
                    .unwrap()
            },
        );

        let col_svc = ColTarget::JsonKey("service");
        bench_frame_op(
            "map service -> uppercase (dict O(|D|))",
            &mut || naive_transform_frame(frame, None, col_svc, &ColOp::MapStr(&upper), opts, dec),
            &mut || vxg::map_column(frame, col_svc, ColOp::MapStr(&upper), opts, dec).unwrap(),
        );

        let col_bytes = ColTarget::JsonKey("bytes");
        bench_frame_op(
            "map bytes -> bytes+100 (1-col numeric)",
            &mut || {
                naive_transform_frame(frame, None, col_bytes, &ColOp::MapU64(&add100), opts, dec)
            },
            &mut || vxg::map_column(frame, col_bytes, ColOp::MapU64(&add100), opts, dec).unwrap(),
        );
    } else {
        bench_filter(
            "filter_col csv:1==NOPE (0-hit alphabet)",
            ColPred::delimited_str(b',', 1, StrCmp::Eq(b"NOPE")),
        );
        bench_filter(
            "filter_col csv:6>=200 (latency_ms)",
            ColPred::delimited_num(b',', 6, NumCmp::Ge(200)),
        );

        let p_500 = ColPred::delimited_num(b',', 5, NumCmp::Eq(500));
        bench_frame_op(
            "filter_frame csv:5==500 (status==500)",
            &mut || naive_filter_frame(frame, &p_500, opts, dec),
            &mut || vxg::filter_frame(frame, &p_500, opts, dec).unwrap(),
        );

        let p_lat = ColPred::delimited_num(b',', 6, NumCmp::Ge(250));
        let col_lvl = ColTarget::Delimited {
            delim: b',',
            index: 1,
        };
        bench_frame_op(
            "update WHERE csv:6>=250 SET CRITICAL",
            &mut || {
                naive_transform_frame(
                    frame,
                    Some(&p_lat),
                    col_lvl,
                    &ColOp::SetStr(b"CRITICAL"),
                    opts,
                    dec,
                )
            },
            &mut || {
                vxg::update_column(frame, &p_lat, col_lvl, ColOp::SetStr(b"CRITICAL"), opts, dec)
                    .unwrap()
            },
        );

        let col_svc = ColTarget::Delimited {
            delim: b',',
            index: 2,
        };
        bench_frame_op(
            "map csv:2 -> uppercase (dict O(|D|))",
            &mut || naive_transform_frame(frame, None, col_svc, &ColOp::MapStr(&upper), opts, dec),
            &mut || vxg::map_column(frame, col_svc, ColOp::MapStr(&upper), opts, dec).unwrap(),
        );

        let col_bytes = ColTarget::Delimited {
            delim: b',',
            index: 7,
        };
        bench_frame_op(
            "map csv:7 -> bytes+100 (1-col numeric)",
            &mut || {
                naive_transform_frame(frame, None, col_bytes, &ColOp::MapU64(&add100), opts, dec)
            },
            &mut || vxg::map_column(frame, col_bytes, ColOp::MapU64(&add100), opts, dec).unwrap(),
        );
    }
}

include!("query_real_support/agg.rs");
include!("query_real_support/project.rs");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: vxg-query-real <file> [--ops-suite ndjson|csv | <needle> ...]");
        std::process::exit(2);
    }
    let raw = std::fs::read(&args[0]).expect("read input");
    let opts = Options {
        checksum: false,
        ..Options::default()
    };
    let dec = DecodeOptions {
        max_output: raw.len().max(1 << 20),
        ..DecodeOptions::default()
    };
    let frame = vxg::compress(&raw, &opts).expect("compress");
    let rows = raw.split_inclusive(|&b| b == b'\n').count();
    println!(
        "{}: raw {} B, vxg {} B ({:.2}%), {} lines",
        args[0],
        raw.len(),
        frame.len(),
        frame.len() as f64 * 100.0 / raw.len() as f64,
        rows
    );
    let (t_cnt, c) = time(20, || vxg::frame_row_count(&frame).unwrap());
    assert_eq!(c, rows, "frame_row_count mismatch");
    println!("frame_row_count: {:.3} ms (rows={})", t_cnt * 1e3, c);

    if args.get(1).map(String::as_str) == Some("--ops-suite") {
        let kind = args.get(2).map(String::as_str).unwrap_or("ndjson");
        let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut opts_t = opts;
        opts_t.threads = threads;
        let mut dec_t = dec;
        dec_t.threads = threads;
        run_ops_suite(&raw, &frame, kind, &opts_t, &dec_t);
        return;
    }

    if args.get(1).map(String::as_str) == Some("--agg-suite") {
        let threads: usize = args.get(3).or(args.get(2)).and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut dec_t = dec;
        dec_t.threads = threads;
        run_agg_suite(&raw, &frame, &dec_t);
        return;
    }

    if args.get(1).map(String::as_str) == Some("--project-suite") {
        let threads: usize = args.get(3).or(args.get(2)).and_then(|s| s.parse().ok()).unwrap_or(1);
        let mut opts_t = opts;
        opts_t.threads = threads;
        let mut dec_t = dec;
        dec_t.threads = threads;
        run_project_suite(&raw, &frame, &opts_t, &dec_t);
        return;
    }

    let gb = raw.len() as f64 / 1e9;
    println!(
        "{:<28} {:>8} {:>12} {:>12} {:>8}",
        "needle", "hits", "raw ms", "vxg ms", "speedup"
    );
    for needle in &args[1..] {
        if let Some((key, cmp)) = parse_num_spec(needle) {
            let limit = 1_000_000;
            let (t_raw, expected) = time(5, || raw_num_filter(&raw, &key, cmp, limit));
            let (t_vxg, got) = time(5, || vxg::filter_json_u64(&frame, &key, cmp, limit, &dec).unwrap());
            assert_eq!(expected.len(), got.len(), "hit count mismatch for {needle:?}");
            for ((ri, rb), m) in expected.iter().zip(got.iter()) {
                assert_eq!(*ri, m.global_row_index, "row index mismatch for {needle:?}");
                assert_eq!(rb, &m.bytes, "row bytes mismatch for {needle:?}");
            }
            println!(
                "{:<28} {:>8} {:>12.3} {:>12.3} {:>7.1}x  [{:.1} GB/s]",
                needle, got.len(), t_raw * 1e3, t_vxg * 1e3, t_raw / t_vxg, gb / t_vxg
            );
            continue;
        }
        let n = needle.as_bytes();
        let limit = 100_000;
        let (t_raw, expected) = time(5, || raw_filter(&raw, n, limit));
        let (t_vxg, got) = time(5, || vxg::filter_rows(&frame, n, limit, &dec).unwrap());
        assert_eq!(expected.len(), got.len(), "hit count mismatch for {needle:?}");
        for ((ri, rb), m) in expected.iter().zip(got.iter()) {
            assert_eq!(*ri, m.global_row_index, "row index mismatch for {needle:?}");
            assert_eq!(rb, &m.bytes, "row bytes mismatch for {needle:?}");
        }
        let first = vxg::find_first(&frame, n, &dec).unwrap();
        assert_eq!(first.as_ref().map(|m| m.global_row_index), expected.first().map(|e| e.0));
        println!(
            "{:<28} {:>8} {:>12.3} {:>12.3} {:>7.1}x  [{:.1} GB/s]",
            format!("{:?}", needle),
            got.len(),
            t_raw * 1e3,
            t_vxg * 1e3,
            t_raw / t_vxg,
            gb / t_vxg
        );
    }
}
