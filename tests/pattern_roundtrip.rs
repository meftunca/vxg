//! MODEL_PATTERN (digit/literal patterned columns): round-trip, compression is actually achieved,
//! and compressed-domain queries agree with a naive scan on pattern columns.

use vxg::{compress, decompress, filter_rows, DecodeOptions, Options};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/// Time-ordered timestamp in layout `kind` for second offset `t` (with occasional backwards jumps).
fn stamp(kind: u64, t: u64) -> String {
    let (d, h, m, s) = (1 + t / 86400 % 28, t / 3600 % 24, t / 60 % 60, t % 60);
    match kind {
        0 => format!("2026-10-{d:02}T{h:02}:{m:02}:{s:02}+02:00"),
        1 => format!("2026-10-{d:02} {h:02}:{m:02}:{s:02}"),
        2 => format!("{d:02}/{}/2026:{h:02}:{m:02}:{s:02} +0000", MONTHS[(t / 2_000_000 % 12) as usize]),
        3 => format!("2026-10-{d:02}T{h:02}:{m:02}:{s:02}.{:03}Z", t * 7 % 1000),
        4 => format!("10.{}.{}.{}", t / 65536 % 256, t / 256 % 256, t % 256),
        5 => format!("u{:06}", t),
        _ => format!("v{}.{:02}.{:03}", t / 10000, t / 100 % 100, t % 100),
    }
}

fn corpus(kind: u64, rows: u64, seed: u64, quoted: bool) -> Vec<u8> {
    let mut rng = Rng(seed | 1);
    let mut out = String::new();
    let mut t = 1000 + rng.below(100_000);
    for i in 0..rows {
        t += rng.below(3);
        if rng.below(200) == 0 {
            t = t.saturating_sub(rng.below(5000)); // backwards jump => exception / negative delta
        }
        if rng.below(500) == 0 {
            t += rng.below(400_000); // day/month carry => large delta exception
        }
        let ts = stamp(kind, t);
        if quoted {
            out += &format!("{{\"ts\":\"{ts}\",\"id\":{i},\"lvl\":\"INFO\",\"msg\":\"request {} done\"}}\n", rng.below(50));
        } else {
            out += &format!("{ts} GET /item/{} {} {}\n", rng.below(900), 200 + 100 * rng.below(3), rng.below(5000));
        }
    }
    out.into_bytes()
}

#[test]
fn pattern_columns_round_trip_and_compress() {
    let dec = DecodeOptions::default();
    for quoted in [true, false] {
        for kind in 0..7 {
            for (rows, seed) in [(3u64, 1u64), (257, 2), (6000, 3), (30_000, 4)] {
                let data = corpus(kind, rows, seed, quoted);
                for block_size in [4096usize, 65536] {
                    let opts = Options { block_size, ..Options::default() };
                    let enc = compress(&data, &opts).unwrap();
                    assert_eq!(
                        decompress(&enc, &dec).unwrap(),
                        data,
                        "quoted={quoted} kind={kind} rows={rows} bs={block_size}"
                    );
                    let mt = DecodeOptions { threads: 4, ..DecodeOptions::default() };
                    assert_eq!(decompress(&enc, &mt).unwrap(), data);
                }
                if rows == 30_000 && quoted && kind != 4 && kind != 6 {
                    let enc = compress(&data, &Options::default()).unwrap();
                    assert!(
                        enc.len() * 20 < data.len(),
                        "kind={kind}: pattern model not effective ({} / {})",
                        enc.len(),
                        data.len()
                    );
                }
            }
        }
    }
}

#[test]
fn pattern_columns_query_matches_naive_scan() {
    let dec = DecodeOptions::default();
    for quoted in [true, false] {
        for kind in [0u64, 2, 5] {
            let data = corpus(kind, 20_000, 9, quoted);
            let enc = compress(&data, &Options::default()).unwrap();
            let text = String::from_utf8(data).unwrap();
            for needle in ["10:1", "Oct", "000", "u0", "request 7 "] {
                let want = text.lines().filter(|l| l.contains(needle)).count();
                let got = filter_rows(&enc, needle.as_bytes(), usize::MAX, &dec).unwrap().len();
                assert_eq!(got, want, "quoted={quoted} kind={kind} needle={needle}");
            }
        }
    }
}

/// CSV / TSV fields with spaces, brackets, UTF-8, embedded tabs, empty fields and quotes.
#[test]
fn delimited_fields_with_free_text_round_trip() {
    let dec = DecodeOptions::default();
    let agents = ["Mozilla/5.0 (Windows NT 10.0; Win64)", "curl/8.1", "Göğüs [x] {y}", "a\tb", "", "q\"uote", " lead"];
    for delim in [',', '\t'] {
        for rows in [2u64, 50, 5000] {
            let mut rng = Rng(rows * 31 + delim as u64);
            let mut s = format!("ts{delim}agent{delim}n\n");
            for i in 0..rows {
                let a = agents[rng.below(agents.len() as u64) as usize];
                s += &format!("{}{delim}{a}{delim}{}\n", 1_790_000_000 + i, rng.below(100));
            }
            for trailing_nl in [true, false] {
                let mut data = s.clone().into_bytes();
                if !trailing_nl {
                    data.pop();
                }
                let enc = compress(&data, &Options::default()).unwrap();
                assert_eq!(decompress(&enc, &dec).unwrap(), data, "delim={delim:?} rows={rows}");
                let text = String::from_utf8(data).unwrap();
                let want = text.lines().filter(|l| l.contains("curl")).count();
                let got = filter_rows(&enc, b"curl", usize::MAX, &dec).unwrap().len();
                assert_eq!(got, want);
            }
        }
    }
}
