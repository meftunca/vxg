//! Real-world compressed-domain query benchmark.
//!
//! Usage: `vxg-query-real <file> <needle> [<needle> ...]`
//!
//! Compresses `<file>` with VXG v4, then for every needle compares
//! `filter_rows`/`find_first` against a plain raw scan, asserting that the
//! returned rows (index + bytes) are byte-identical, and prints timings.

use std::hint::black_box;
use std::time::Instant;
use vxg::{DecodeOptions, Options};

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

fn time<T>(iters: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut last = f();
    let t = Instant::now();
    for _ in 0..iters {
        last = black_box(f());
    }
    (t.elapsed().as_secs_f64() / iters as f64, last)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: vxg-query-real <file> <needle> [<needle> ...]");
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

    let gb = raw.len() as f64 / 1e9;
    println!(
        "{:<28} {:>8} {:>12} {:>12} {:>8}",
        "needle", "hits", "raw ms", "vxg ms", "speedup"
    );
    for needle in &args[1..] {
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
