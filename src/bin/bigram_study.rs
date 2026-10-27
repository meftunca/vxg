//! Offline study: how many 64 KiB blocks could a bigram pre-filter reject beyond what the
//! 256-bit alphabet filter already rejects?
//!
//! Usage: `vxg-bigram-study <file> <needle> [<needle> ...]`
//!
//! Uses exact per-block byte/bigram sets computed on the raw bytes, so the bigram column is an
//! UPPER BOUND for a real compressed-domain filter (which must be conservative at piece
//! boundaries and cannot see exact adjacency inside non-dictionary columns).

fn contains(h: &[u8], n: &[u8]) -> bool {
    n.is_empty() || h.windows(n.len()).any(|w| w == n)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: vxg-bigram-study <file> <needle> [<needle> ...]");
        std::process::exit(2);
    }
    let raw = std::fs::read(&args[0]).expect("read input");
    let blocks: Vec<&[u8]> = raw.chunks(65536).collect();
    println!("{}: {} blocks of 64 KiB", args[0], blocks.len());
    println!(
        "{:<26} {:>7} {:>10} {:>14} {:>16}",
        "needle", "blocks", "truly has", "alphabet rejects", "bigram rejects (extra)"
    );
    for needle in &args[1..] {
        let n = needle.as_bytes();
        let (mut has, mut alpha_rej, mut extra_bigram) = (0usize, 0usize, 0usize);
        for b in &blocks {
            let present = contains(b, n);
            if present {
                has += 1;
                continue;
            }
            let mut seen = [false; 256];
            for &c in *b {
                seen[c as usize] = true;
            }
            if n.iter().any(|&c| !seen[c as usize]) {
                alpha_rej += 1;
                continue;
            }
            // Alphabet passes but needle absent: would a bigram filter reject?
            let mut bigrams = vec![0u64; 1024]; // 65536 bits
            for w in b.windows(2) {
                let k = ((w[0] as usize) << 8) | w[1] as usize;
                bigrams[k >> 6] |= 1u64 << (k & 63);
            }
            let miss = n.windows(2).any(|w| {
                let k = ((w[0] as usize) << 8) | w[1] as usize;
                bigrams[k >> 6] & (1u64 << (k & 63)) == 0
            });
            if miss {
                extra_bigram += 1;
            }
        }
        println!(
            "{:<26} {:>7} {:>10} {:>14} {:>16}",
            format!("{:?}", needle),
            blocks.len(),
            has,
            alpha_rej,
            extra_bigram
        );
    }
}
