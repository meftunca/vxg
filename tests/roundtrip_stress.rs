//! Deterministic randomized round-trip stress: `decompress(compress(x)) == x` on varied
//! structured inputs (NDJSON with schema drift, CSV, logfmt, mixed numeric/string columns,
//! columns whose distinct-value count crosses 16/256/4096, split lines, no trailing newline).
//!
//! `VXG_STRESS_ITERS` raises the iteration count (default is small enough for CI).

use vxg::{compress, decompress, DecodeOptions, Options};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len() as u64) as usize]
    }
}

fn value(rng: &mut Rng, kind: u64, distinct: u64, i: u64) -> String {
    match kind {
        0 => format!("{}", rng.below(distinct)),
        1 => format!("\"s{}\"", rng.below(distinct)),
        2 => format!("{}", i * (1 + rng.below(3))),
        3 => format!("\"{:016x}\"", rng.next()),
        4 => match rng.below(5) {
            0 => "1.5".to_string(),
            1 => "007".to_string(),
            2 => "\"x\"".to_string(),
            3 => "null".to_string(),
            _ => format!("{}", rng.below(distinct)),
        },
        5 => format!("\"2026-10-{:02}T{:02}:{:02}:{:02}Z\"", 1 + rng.below(28), rng.below(24), rng.below(60), rng.below(60)),
        _ => format!("\"{}\"", rng.pick(&["INFO", "WARN", "ERROR", "DEBUG", "a b", "c,d", "q\\\"r"])),
    }
}

fn gen(rng: &mut Rng) -> Vec<u8> {
    let rows = 50 + rng.below(9000);
    let cols = 1 + rng.below(9);
    let format = rng.below(4);
    let kinds: Vec<u64> = (0..cols).map(|_| rng.below(7)).collect();
    let distincts: Vec<u64> = (0..cols)
        .map(|_| *[2u64, 7, 16, 17, 255, 256, 257, 300, 4096, 100_000].get(rng.below(10) as usize).unwrap())
        .collect();
    let drift = rng.below(3); // 0 none, 1 occasional extra key, 2 occasional missing key
    let mut out = Vec::new();
    for i in 0..rows {
        let mut line = String::new();
        match format {
            0 | 1 => {
                line.push('{');
                for c in 0..cols as usize {
                    if drift == 2 && c == 1 && rng.below(9) == 0 {
                        continue;
                    }
                    if line.len() > 1 {
                        line.push(',');
                    }
                    line.push_str(&format!("\"k{c}\":{}", value(rng, kinds[c], distincts[c], i)));
                }
                if drift == 1 && rng.below(13) == 0 {
                    line.push_str(",\"extra\":true");
                }
                line.push('}');
            }
            2 => {
                for c in 0..cols as usize {
                    if c > 0 {
                        line.push(',');
                    }
                    let v = value(rng, kinds[c], distincts[c], i);
                    line.push_str(v.trim_matches('"'));
                }
            }
            _ => {
                for c in 0..cols as usize {
                    if c > 0 {
                        line.push(' ');
                    }
                    let v = value(rng, kinds[c], distincts[c], i);
                    line.push_str(&format!("k{c}={}", v.replace(' ', "_")));
                }
            }
        }
        line.push('\n');
        out.extend_from_slice(line.as_bytes());
    }
    if rng.below(4) == 0 {
        out.pop(); // no trailing newline
    }
    out
}

#[test]
fn randomized_structured_roundtrip() {
    let iters: u64 = std::env::var("VXG_STRESS_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let seed0: u64 = std::env::var("VXG_STRESS_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x5eed);
    let dec = DecodeOptions::default();
    let mt = DecodeOptions { threads: 4, ..DecodeOptions::default() };
    let mut failures = Vec::new();
    // `VXG_STRESS_SEEDS=a,b,c` replays exactly these seeds, in order, on one thread (to minimize
    // failures that depend on state left behind by earlier inputs).
    let explicit: Option<Vec<u64>> = std::env::var("VXG_STRESS_SEEDS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let total = explicit.as_ref().map_or(iters, |v| v.len() as u64);
    for it in 0..total {
        let seed = match &explicit {
            Some(v) => v[it as usize],
            None => seed0.wrapping_add(it.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1,
        };
        let mut rng = Rng(seed);
        let raw = gen(&mut rng);
        if std::env::var("VXG_STRESS_DUMP").is_ok() {
            let _ = std::fs::write(format!("/tmp/vxg_stress_{seed}.bin"), &raw);
        }
        for block_size in [65536usize, 16384] {
            let opts = Options { block_size, ..Options::default() };
            let frame = compress(&raw, &opts).expect("compress");
            match decompress(&frame, &dec) {
                Ok(out) if out == raw => {}
                Ok(_) => failures.push(format!("seed={seed} block={block_size}: output differs")),
                Err(e) => failures.push(format!("seed={seed} block={block_size}: {e:?}")),
            }
            match decompress(&frame, &mt) {
                Ok(out) if out == raw => {}
                Ok(_) => failures.push(format!("seed={seed} block={block_size} (mt): output differs")),
                Err(e) => failures.push(format!("seed={seed} block={block_size} (mt): {e:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "{} failures (first 8):\n{}", failures.len(), failures.iter().take(8).cloned().collect::<Vec<_>>().join("\n"));
}

/// Seeds that exposed real round-trip bugs (kept as permanent regressions):
/// - 17061486795440217195: exactly 256 distinct values with the first row at rank 255 (the
///   `u8::MAX` "previous rank" sentinel collided) -> `column front dict repeat bit0`.
/// - 9535266630698049541: a 312-byte final block whose packet was `>= raw` but under the old
///   `1 + uleb + raw` gate was tagged COLUMN -> `block kind/size`.
/// - 15716801513121184037: multi-thread-only `Checksum` mismatch (see notes in the repo history).
#[test]
fn known_failing_seeds_round_trip() {
    let dec = DecodeOptions::default();
    let mt = DecodeOptions { threads: 4, ..DecodeOptions::default() };
    for seed in [17061486795440217195u64, 9535266630698049541, 15716801513121184037] {
        let raw = gen(&mut Rng(seed | 1));
        for block_size in [65536usize, 16384] {
            let frame = compress(&raw, &Options { block_size, ..Options::default() }).unwrap();
            assert_eq!(decompress(&frame, &dec).unwrap_or_else(|e| panic!("seed={seed} bs={block_size}: {e:?}")), raw);
            assert_eq!(decompress(&frame, &mt).unwrap_or_else(|e| panic!("seed={seed} bs={block_size} (mt): {e:?}")), raw);
        }
    }
}

/// Regression: state left behind by an earlier input must never change the result for a later one.
/// This exact two-input history used to fail with `fast lz zero offset`: the LZ hash table kept
/// stale positions from a previous, smaller input.
#[test]
fn stale_scratch_state_does_not_leak_between_inputs() {
    let dec = DecodeOptions::default();
    for history in [
        [4354685564936845361u64, 15755400384260043847],
        [15755400384260043847, 4354685564936845361],
    ] {
        for seed in history {
            let raw = gen(&mut Rng(seed | 1));
            let frame = compress(&raw, &Options::default()).unwrap();
            let out = decompress(&frame, &dec)
                .unwrap_or_else(|e| panic!("history {history:?}, seed={seed}: {e:?}"));
            assert_eq!(out, raw, "history {history:?}, seed={seed}");
        }
    }
}
