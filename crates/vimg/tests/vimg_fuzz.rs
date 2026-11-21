//! Adversarial fuzzing, OOM-bomb hardening, C ABI safety, and HTTP-Range network tests.

use vimg::checksum::crc32c;
use vimg::ffi::{vimg_decode_query_v2_alloc, vimg_free, vimg_wasm_encode_rgba, VimgQueryC};
use vimg::format::{HEADER_BYTES, MAGIC, MAX_IMAGE_DIM};
use vimg::{
    crop_vimg_zero_decode, decode, decode_query, encode, EncodeOptions, SliceRangeSource,
    TransformQuery, VimgHeader, VimgIndex,
};

struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn usize(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound.max(1)
    }
}

fn synthetic_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            out.push(((x * 7 + y * 3) & 255) as u8);
            out.push(((x * 2 + y * 11 + 40) & 255) as u8);
            out.push((((x ^ y) * 5 + 90) & 255) as u8);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn make_raw_header(
    width: u32,
    height: u32,
    channels: u8,
    quality: u8,
    tile_size: u32,
    dwt_levels: u8,
    dir_bytes: u32,
    total_bytes: u64,
) -> Vec<u8> {
    let mut h = vec![0u8; HEADER_BYTES];
    h[0..4].copy_from_slice(MAGIC);
    h[4] = 1; // version
    h[5] = 0; // flags
    h[6] = channels;
    h[7] = quality;
    h[8..12].copy_from_slice(&width.to_le_bytes());
    h[12..16].copy_from_slice(&height.to_le_bytes());
    h[16..20].copy_from_slice(&tile_size.to_le_bytes());
    h[20] = dwt_levels;
    h[21] = 0;
    h[24..28].copy_from_slice(&dir_bytes.to_le_bytes());
    h[28..36].copy_from_slice(&total_bytes.to_le_bytes());
    h
}

#[test]
fn oom_bombs_and_invalid_headers_are_rejected_without_allocation() {
    let bombs: Vec<Vec<u8>> = vec![
        make_raw_header(u32::MAX, u32::MAX, 3, 80, 256, 3, 64, 1024),
        make_raw_header(MAX_IMAGE_DIM + 1, 64, 3, 80, 256, 3, 64, 1024),
        make_raw_header(64, MAX_IMAGE_DIM + 1, 3, 80, 256, 3, 64, 1024),
        make_raw_header(32768, 32768, 3, 80, 256, 3, 64, 1024), // > MAX_IMAGE_PIXELS
        make_raw_header(256, 256, 3, 80, 256, 3, u32::MAX, u64::MAX),
        make_raw_header(256, 256, 3, 0, 256, 3, 64, 1024), // quality 0
        make_raw_header(256, 256, 3, 101, 256, 3, 64, 1024), // quality > 100
        make_raw_header(256, 256, 9, 80, 256, 3, 64, 1024), // bad channels
        make_raw_header(256, 256, 3, 80, 0, 3, 64, 1024), // tile_size 0
        make_raw_header(256, 256, 3, 80, 256, 9, 64, 1024), // dwt_levels > 4
        make_raw_header(0, 256, 3, 80, 256, 3, 64, 1024),
    ];

    for (idx, bomb) in bombs.iter().enumerate() {
        assert!(VimgHeader::parse(bomb).is_err(), "bomb #{idx} unexpectedly parsed header");
        assert!(VimgIndex::parse_prefix(bomb).is_err(), "bomb #{idx} unexpectedly parsed index");
        assert!(decode(bomb).is_err(), "bomb #{idx} unexpectedly decoded");
    }

    // Valid tiny image + OOM resize query must fail cleanly
    let px = synthetic_rgb(64, 64);
    let valid = encode(&px, 64, 64, 3, &EncodeOptions::default()).unwrap();
    let huge_resize = TransformQuery {
        target_width: Some(100_000),
        target_height: Some(100_000),
        ..TransformQuery::default()
    };
    assert!(decode_query(&valid, &huge_resize).is_err());

    // FFI safety: null pointers & OOM dimensions must return error codes, never panic
    unsafe {
        let null_res = vimg_decode_query_v2_alloc(core::ptr::null(), 0, core::ptr::null());
        assert!(null_res.status < 0);
        vimg_free(null_res);

        let oom_wasm = vimg_wasm_encode_rgba(px.as_ptr(), MAX_IMAGE_DIM + 1, 100, 80, 256);
        assert!(oom_wasm < 0);

        let q = VimgQueryC {
            crop_x: 0,
            crop_y: 0,
            crop_w: 32,
            crop_h: 32,
            target_w: 0,
            target_h: 0,
            grayscale: 0,
            sepia: 1,
            rotate_cw: 1,
            flip_h: 1,
            flip_v: 0,
            recover_corrupt: 1,
            brightness: 5,
            blur: 0.0,
            sharpen: 0.0,
            contrast: 1.0,
        };
        let ok_res = vimg_decode_query_v2_alloc(valid.as_ptr(), valid.len(), &q);
        assert_eq!(ok_res.status, 0);
        assert_eq!((ok_res.width, ok_res.height, ok_res.channels), (32, 32, 3));
        vimg_free(ok_res);
    }
}

#[test]
fn fuzz_12000_adversarial_mutations_never_panic_or_oom() {
    let px = synthetic_rgb(96, 80);
    let lossy = encode(
        &px,
        96,
        80,
        3,
        &EncodeOptions { quality: 80, tile_size: 64, dwt_levels: 3, chroma_420: true, checksum: true, threads: 1, ..EncodeOptions::default() },
    )
    .unwrap();
    let lossless = encode(
        &px,
        96,
        80,
        3,
        &EncodeOptions { quality: 100, tile_size: 64, dwt_levels: 2, chroma_420: false, checksum: true, threads: 1, ..EncodeOptions::default() },
    )
    .unwrap();

    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    let recover_q = TransformQuery { recover_corrupt: true, ..TransformQuery::default() };

    // 1) 4,000 random garbage buffers
    let mut garbage = Vec::new();
    for i in 0..4_000 {
        let len = rng.usize(384);
        garbage.resize(len, 0);
        for b in &mut garbage {
            *b = rng.next() as u8;
        }
        if i % 4 == 0 && garbage.len() >= 4 {
            garbage[0..4].copy_from_slice(MAGIC);
        }
        let _ = VimgHeader::parse(&garbage);
        let _ = VimgIndex::parse_prefix(&garbage);
        let _ = decode(&garbage);
        let _ = decode_query(&garbage, &recover_q);
    }

    // 2) 8,000 targeted mutations on real lossy + lossless VIMG streams
    for iter in 0..8_000 {
        let base = if (iter & 1) == 0 { &lossy } else { &lossless };
        let mut mutated = base.clone();
        match iter % 5 {
            0 => {
                // Truncation at arbitrary byte
                let cut = rng.usize(mutated.len());
                mutated.truncate(cut);
            }
            1 => {
                // Header / directory bit flips
                let pos = rng.usize(mutated.len().min(256));
                mutated[pos] ^= 1u8 << (rng.next() & 7);
            }
            2 => {
                // Payload chunk corruption (also exercises CRC32C rejection & recovery)
                let n_flips = 1 + rng.usize(6);
                for _ in 0..n_flips {
                    let pos = rng.usize(mutated.len());
                    mutated[pos] = rng.next() as u8;
                }
            }
            3 => {
                // Corrupt chunk payload while recomputing CRC32C so the entropy decoder
                // itself gets fuzzed directly on adversarial bitstreams!
                if let Ok(idx) = VimgIndex::parse_prefix(&mutated) {
                    if !idx.entries.is_empty() {
                        let e = idx.entries[rng.usize(idx.entries.len())];
                        let s = e.offset as usize;
                        let end = s.saturating_add(e.len as usize);
                        if end <= mutated.len() && e.len > 0 {
                            let p = s + rng.usize(e.len as usize);
                            mutated[p] ^= 0xFF;
                            // Patch stored CRC32C in directory entry
                            let new_crc = crc32c(&mutated[s..end]);
                            let dir_off = HEADER_BYTES + idx.entries.iter().position(|x| x.offset == e.offset).unwrap() * 16;
                            if dir_off + 16 <= mutated.len() {
                                mutated[dir_off + 12..dir_off + 16].copy_from_slice(&new_crc.to_le_bytes());
                            }
                        }
                    }
                }
            }
            _ => {
                // Random slice insertion/deletion
                let cut = rng.usize(mutated.len());
                mutated.drain(cut..(cut + rng.usize(16)).min(mutated.len()));
            }
        }
        let _ = decode(&mutated);
        if iter % 3 == 0 {
            let _ = decode_query(&mutated, &recover_q);
        }
        if iter % 7 == 0 {
            let _ = crop_vimg_zero_decode(&SliceRangeSource(&mutated), 8, 8, 48, 48, false);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn http_range_source_keep_alive_redirect_and_parallel_fetch() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use vimg::{decode_from_source, HttpRangeSource, RangeSource};

    let px = synthetic_rgb(192, 160);
    let data = Arc::new(
        encode(
            &px,
            192,
            160,
            3,
            &EncodeOptions { quality: 82, tile_size: 64, dwt_levels: 3, chroma_420: true, checksum: true, threads: 1, ..EncodeOptions::default() },
        )
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let accepted = Arc::new(AtomicUsize::new(0));
    let accepted_bg = Arc::clone(&accepted);
    let data_bg = Arc::clone(&data);

    let _srv = std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            accepted_bg.fetch_add(1, Ordering::Relaxed);
            let data_t = Arc::clone(&data_bg);
            std::thread::spawn(move || {
                let mut stream = stream;
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
                loop {
                    let mut buf = [0u8; 4096];
                    let n = match stream.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let mut parts = first.split_whitespace();
                    let method = parts.next().unwrap_or("GET");
                    let path = parts.next().unwrap_or("/");

                    if path == "/redirect.vimg" {
                        let resp = "HTTP/1.1 302 Found\r\nLocation: /real.vimg\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
                        let _ = stream.write_all(resp.as_bytes());
                        continue;
                    }
                    if method == "HEAD" {
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: keep-alive\r\n\r\n",
                            data_t.len()
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        continue;
                    }
                    let mut range = None;
                    for line in req.lines() {
                        if let Some(v) = line.strip_prefix("Range: bytes=") {
                            if let Some((a, b)) = v.trim().split_once('-') {
                                if let (Ok(s), Ok(e)) = (a.parse::<usize>(), b.parse::<usize>()) {
                                    range = Some((s, e.min(data_t.len() - 1)));
                                }
                            }
                        }
                    }
                    if let Some((s, e)) = range {
                        let slice = &data_t[s..=e];
                        let hdr = format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {s}-{e}/{}\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
                            data_t.len(),
                            slice.len()
                        );
                        let _ = stream.write_all(hdr.as_bytes());
                        let _ = stream.write_all(slice);
                    } else {
                        let hdr = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n",
                            data_t.len()
                        );
                        let _ = stream.write_all(hdr.as_bytes());
                        let _ = stream.write_all(&data_t);
                    }
                }
            });
        }
    });

    // Connect via `/redirect.vimg` to test 302 redirect following + keep-alive reuse!
    let src = HttpRangeSource::connect(&format!("http://{addr}/redirect.vimg")).unwrap();
    assert_eq!(src.len(), data.len() as u64);

    // Warm the pooled keep-alive socket, then execute 5 sequential range fetches:
    // keep-alive MUST reuse the exact same TCP connection (zero extra accepts!).
    let _ = src.read_ranges(&[(0, 36)]).unwrap();
    let accepts_warm = accepted.load(Ordering::Relaxed);
    for step in 0..5u64 {
        let s = step * 32;
        let e = (s + 32).min(data.len() as u64);
        let got = src.read_ranges(&[(s, e)]).unwrap();
        assert_eq!(got[0], &data[s as usize..e as usize]);
    }
    assert_eq!(
        accepted.load(Ordering::Relaxed),
        accepts_warm,
        "keep-alive failed: sequential range requests opened new TCP connections!"
    );

    // Full end-to-end compressed-domain queries over redirected HttpRangeSource
    let q1 = TransformQuery { target_width: Some(48), ..TransformQuery::default() };
    let img1 = decode_from_source(&src, &q1).unwrap();
    let ref1 = decode_query(&data, &q1).unwrap();
    assert_eq!(img1.pixels, ref1.pixels);

    let q2 = TransformQuery { crop: Some((32, 32, 64, 64)), ..TransformQuery::default() };
    let img2 = decode_from_source(&src, &q2).unwrap();
    let ref2 = decode_query(&data, &q2).unwrap();
    assert_eq!(img2.pixels, ref2.pixels);

    // Parallel multi-range fetch parity (>= 3 ranges triggers parallel workers)
    let par = src.read_ranges(&[(0, 36), (36, 96), (96, 180), (180, 260)]).unwrap();
    assert_eq!(par[0], &data[0..36]);
    assert_eq!(par[1], &data[36..96]);
    assert_eq!(par[2], &data[96..180]);
    assert_eq!(par[3], &data[180..260]);
}
