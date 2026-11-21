use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::thread;

use vimg::{
    crop_vimg_zero_decode, decode, decode_from_source, decode_query, encode, plan_query_ranges,
    psnr_db, EncodeOptions, HttpRangeSource, SliceRangeSource, SparseRangeSource, TransformQuery,
    VimgHeader, VimgIndex,
};

fn synth_photo_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        let yf = y as f32 / h as f32;
        for x in 0..w {
            let xf = x as f32 / w as f32;
            // Multi-scale photo spectrum: smooth sky/landscape + mid-freq foliage + fine micro-texture + sharp edges
            let low = ((xf * 18.0).sin() * (yf * 14.0).cos()) * 28.0;
            let mid = ((xf * 97.0 + yf * 31.0).sin() * (yf * 83.0 - xf * 23.0).cos()) * 14.0;
            // Deterministic fine sensor/foliage texture (high-frequency detail so L2/L3 carry real energy)
            let hval = ((x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263))
                .wrapping_mul(1274126177);
            let grain = (((hval >> 24) as f32 / 255.0) - 0.5) * 26.0;
            let edge = if ((x / 47) + (y / 39)) % 3 == 0 { 18.0 } else { -12.0 };
            let r = (45.0 + 160.0 * xf + low + mid + grain + edge).clamp(0.0, 255.0) as u8;
            let g = (65.0 + 130.0 * yf - low * 0.5 + mid * 0.7 + grain * 0.8).clamp(0.0, 255.0) as u8;
            let b = (205.0 - 125.0 * yf + low * 0.3 - mid * 0.6 + grain * 0.6 - edge * 0.5).clamp(0.0, 255.0) as u8;
            let o = (y * w + x) * 3;
            px[o] = r;
            px[o + 1] = g;
            px[o + 2] = b;
        }
    }
    px
}

#[test]
fn lossless_rgb_rgba_odd_dims_and_deterministic_mt() {
    for (w, h, ch) in [(256, 256, 3u8), (333, 277, 3u8), (195, 141, 4u8)] {
        let mut px = vec![0u8; w * h * ch as usize];
        for y in 0..h {
            for x in 0..w {
                let o = (y * w + x) * ch as usize;
                px[o] = ((x * 7 + y * 13) % 256) as u8;
                px[o + 1] = ((x * 3 + y * 29 + 11) % 256) as u8;
                px[o + 2] = ((x * 19 + y * 5 + 77) % 256) as u8;
                if ch == 4 {
                    px[o + 3] = if (x + y) % 7 == 0 { 128 } else { 255 };
                }
            }
        }
        let b1 = encode(&px, w as u32, h as u32, ch, &EncodeOptions { threads: 1, ..EncodeOptions::lossless() }).unwrap();
        let b4 = encode(&px, w as u32, h as u32, ch, &EncodeOptions { threads: 4, ..EncodeOptions::lossless() }).unwrap();
        assert_eq!(b1, b4, "1T and 4T encoded bytes must be 100% bit-identical");

        let dec = decode(&b1).unwrap();
        assert_eq!((dec.width, dec.height, dec.channels), (w as u32, h as u32, ch));
        assert_eq!(dec.pixels, px, "lossless roundtrip mismatch on {w}x{h}x{ch}");
    }
}

#[test]
fn lossy_photo_quality_pyramid_effects_and_zero_decode_crop() {
    let (w, h) = (1024usize, 768usize);
    let px = synth_photo_rgb(w, h);
    let vimg_bytes = encode(
        &px,
        w as u32,
        h as u32,
        3,
        &EncodeOptions { quality: 85, tile_size: 128, dwt_levels: 3, chroma_420: true, checksum: true, threads: 4 },
    )
    .unwrap();

    // Full decode quality check
    let full = decode(&vimg_bytes).unwrap();
    let psnr = psnr_db(&px, &full.pixels);
    assert!(psnr >= 35.0, "expected >= 35 dB PSNR at q=85, got {psnr:.2} dB");
    assert!(vimg_bytes.len() < px.len() / 6, "expected strong compression ratio on photo");

    // 1/4 pyramid resize skips L2 and L3 completely!
    let thumb = decode_query(&vimg_bytes, &TransformQuery { target_width: Some(256), ..Default::default() }).unwrap();
    assert_eq!((thumb.width, thumb.height, thumb.decoded_level), (256, 192, 1));
    assert!(thumb.stats.fetched_pct() < 45.0, "1/4 resize fetched {:.1}%", thumb.stats.fetched_pct());

    // Grayscale skips Co and Cg chroma planes completely!
    let gray = decode_query(&vimg_bytes, &TransformQuery { grayscale: true, ..Default::default() }).unwrap();
    assert_eq!((gray.width, gray.height, gray.channels), (1024, 768, 1));
    assert!(gray.stats.bytes_fetched < full.stats.bytes_fetched);

    // Subband blur / sharpen / brightness / sepia / rotate execute cleanly
    let fx = decode_query(
        &vimg_bytes,
        &TransformQuery {
            target_width: Some(512),
            blur: 0.5,
            brightness: 12,
            contrast: 1.15,
            sepia: true,
            rotate_cw: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!((fx.width, fx.height, fx.channels), (384, 512, 3));

    // Zero-decode tile-aligned crop (`vimg -> vimg`) matches full-image crop bit-for-bit!
    let src = SliceRangeSource(&vimg_bytes);
    let (cropped_vimg, crop_stats) = crop_vimg_zero_decode(&src, 2, 1, 3, 2, false).unwrap();
    assert!(crop_stats.fetched_pct() < 40.0);
    let cropped_dec = decode(&cropped_vimg).unwrap();
    let direct_crop = decode_query(
        &vimg_bytes,
        &TransformQuery { crop: Some((256, 128, 384, 256)), ..Default::default() },
    )
    .unwrap();
    assert_eq!(cropped_dec.pixels, direct_crop.pixels);
}

#[test]
fn real_http_range_partial_fetch_transfers_tiny_fraction_of_image() {
    let (w, h) = (1024usize, 768usize);
    let px = synth_photo_rgb(w, h);
    let vimg_bytes = Arc::new(
        encode(
            &px,
            w as u32,
            h as u32,
            3,
            &EncodeOptions { quality: 85, tile_size: 128, dwt_levels: 3, chroma_420: true, checksum: true, threads: 4 },
        )
        .unwrap(),
    );

    // Spin up a real localhost HTTP/1.1 Range server
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_data = Arc::clone(&vimg_bytes);
    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut req_line = String::new();
            if reader.read_line(&mut req_line).is_err() {
                continue;
            }
            let method = req_line.split_whitespace().next().unwrap_or("GET");
            let mut range_hdr = None;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                    break;
                }
                if let Some((k, v)) = line.split_once(':') {
                    if k.eq_ignore_ascii_case("range") {
                        range_hdr = Some(v.trim().to_owned());
                    }
                }
            }
            let total = srv_data.len();
            if method == "HEAD" {
                let _ = write!(stream, "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n");
                continue;
            }
            if let Some(r) = range_hdr.and_then(|s| s.strip_prefix("bytes=").map(str::to_owned)) {
                let (a, b) = r.split_once('-').unwrap();
                let (s, e): (usize, usize) = (a.parse().unwrap(), b.parse().unwrap());
                let slice = &srv_data[s..=e];
                let _ = write!(
                    stream,
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = stream.write_all(slice);
            }
        }
    });

    let url = format!("http://{addr}/photo.vimg");
    let http_src = HttpRangeSource::connect(&url).unwrap();

    // Remote 128x128 center crop over real TCP HTTP/1.1 Range!
    let remote_crop = decode_from_source(
        &http_src,
        &TransformQuery { crop: Some((384, 256, 128, 128)), ..Default::default() },
    )
    .unwrap();
    let local_crop = decode_query(
        &vimg_bytes,
        &TransformQuery { crop: Some((384, 256, 128, 128)), ..Default::default() },
    )
    .unwrap();
    assert_eq!(remote_crop.pixels, local_crop.pixels);
    assert_eq!(remote_crop.tiles_fetched, 1);
    assert_eq!(remote_crop.tiles_total, 48);
    assert!(
        remote_crop.stats.fetched_pct() < 40.0,
        "remote 1-tile HTTP Range crop should fetch tiny fraction (got {:.1}%)",
        remote_crop.stats.fetched_pct()
    );
}

#[test]
fn corruption_is_detected_and_salvaged_with_recover_flag() {
    let px = synth_photo_rgb(256, 256);
    let mut bytes = encode(&px, 256, 256, 3, &EncodeOptions::lossless()).unwrap();
    // Corrupt the very last payload byte!
    let last = bytes.len() - 1;
    bytes[last] ^= 0x55;

    assert!(decode(&bytes).is_err(), "corrupt tile must fail CRC32C by default");

    let salvaged = decode_query(
        &bytes,
        &TransformQuery { recover_corrupt: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!((salvaged.width, salvaged.height), (256, 256));
}

#[test]
fn sparse_wasm_range_pipeline_matches_full_decode_bit_for_bit() {
    let px = synth_photo_rgb(1024, 768);
    let bytes = encode(&px, 1024, 768, 3, &EncodeOptions::default()).unwrap();
    let hdr = VimgHeader::parse(&bytes).unwrap();
    let prefix_len = (48 + hdr.dir_bytes as usize).max(4096).min(bytes.len());
    let prefix = &bytes[..prefix_len];
    let idx = VimgIndex::parse_prefix(prefix).unwrap();

    let q = TransformQuery {
        crop: Some((200, 150, 320, 240)),
        blur: 0.35,
        sepia: true,
        ..Default::default()
    };
    let (ranges, lvl, tf, tt) = plan_query_ranges(&idx, &q, prefix_len as u64).unwrap();
    assert_eq!(lvl, 3);
    assert!(tf < tt);

    let mut spans: Vec<(u64, &[u8])> = vec![(0, prefix)];
    for (s, e) in ranges {
        spans.push((s, &bytes[s as usize..e as usize]));
    }
    let sparse = SparseRangeSource { total_len: bytes.len() as u64, spans };
    let dec_sparse = decode_from_source(&sparse, &q).unwrap();
    let dec_direct = decode_query(&bytes, &q).unwrap();
    assert_eq!(dec_sparse.pixels, dec_direct.pixels);
    assert_eq!((dec_sparse.width, dec_sparse.height), (320, 240));
}
