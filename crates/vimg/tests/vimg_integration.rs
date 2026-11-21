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
        &EncodeOptions { quality: 85, tile_size: 128, dwt_levels: 3, chroma_420: true, checksum: true, threads: 4, ..EncodeOptions::default() },
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
            &EncodeOptions { quality: 85, tile_size: 128, dwt_levels: 3, chroma_420: true, checksum: true, threads: 4, ..EncodeOptions::default() },
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
    assert_eq!(lvl, hdr.dwt_levels);
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

#[test]
fn hdr_10bit_12bit_and_icc_profile_roundtrip_and_zero_decode_crop() {
    use vimg::ffi::{vimg_extract_icc_alloc, vimg_free};
    use vimg::{encode_u16, Error, VimgHeader};

    // Synthetic Display P3 ICC profile fixture (536 bytes)
    let mut fake_p3 = vec![0u8; 536];
    fake_p3[0..12].copy_from_slice(b"DISPLAY_P3_0");
    for (i, b) in fake_p3.iter_mut().enumerate().skip(12) {
        *b = ((i * 29 + 7) & 0xFF) as u8;
    }

    for bit_depth in [10u8, 12u8] {
        let max_v = (1u16 << bit_depth) - 1;
        let (w, h) = (192u32, 160u32);
        let mut px16 = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let r = (((x * (max_v as usize)) / w as usize) as u16).min(max_v);
                let g = (((y * (max_v as usize)) / h as usize) as u16).min(max_v);
                let b = ((((x + y) * (max_v as usize)) / (w as usize + h as usize)) as u16).min(max_v);
                px16.extend_from_slice(&[r, g, b]);
            }
        }

        // 1) 100% bit-exact lossless HDR + ICC profile roundtrip
        let ll_bytes = encode_u16(
            &px16,
            w,
            h,
            3,
            &EncodeOptions {
                bit_depth,
                tile_size: 64,
                icc_profile: Some(fake_p3.clone()),
                ..EncodeOptions::lossless()
            },
        )
        .unwrap();
        let hdr = VimgHeader::parse(&ll_bytes).unwrap();
        assert_eq!(hdr.bit_depth(), bit_depth);
        assert!(hdr.has_icc());
        assert_eq!(hdr.icc_bytes as usize, fake_p3.len());

        let dec_ll = decode(&ll_bytes).unwrap();
        assert_eq!(dec_ll.bit_depth, bit_depth);
        assert_eq!(dec_ll.icc_profile.as_deref(), Some(fake_p3.as_slice()));
        assert_eq!(dec_ll.pixels_u16.as_deref(), Some(px16.as_slice()));

        // Verify C ABI ICC extraction
        unsafe {
            let icc_buf = vimg_extract_icc_alloc(ll_bytes.as_ptr(), ll_bytes.len());
            assert_eq!(icc_buf.status, 0);
            let extracted = core::slice::from_raw_parts(icc_buf.ptr, icc_buf.len);
            assert_eq!(extracted, fake_p3.as_slice());
            vimg_free(icc_buf);
        }

        // Verify ICC CRC32C tamper detection!
        let mut tampered = ll_bytes.clone();
        let icc_pos = hdr.icc_offset() as usize + 16;
        tampered[icc_pos] ^= 0x55;
        assert!(matches!(decode(&tampered), Err(Error::ChecksumMismatch { .. })));

        // Verify zero-decode tile crop preserves both bit_depth and ICC profile verbatim!
        let (cropped_vimg, _) =
            crop_vimg_zero_decode(&SliceRangeSource(&ll_bytes), 1, 1, 2, 1, false).unwrap();
        let dec_crop = decode(&cropped_vimg).unwrap();
        assert_eq!(dec_crop.bit_depth, bit_depth);
        assert_eq!((dec_crop.width, dec_crop.height), (128, 64));
        assert_eq!(dec_crop.icc_profile.as_deref(), Some(fake_p3.as_slice()));

        // 2) Lossy HDR (CDF 9/7 + 4:2:0) + compressed-domain crop/resize/rotate
        let ly_bytes = encode_u16(
            &px16,
            w,
            h,
            3,
            &EncodeOptions {
                quality: 85,
                bit_depth,
                tile_size: 64,
                icc_profile: Some(fake_p3.clone()),
                ..EncodeOptions::default()
            },
        )
        .unwrap();
        let dec_ly = decode(&ly_bytes).unwrap();
        let rec16 = dec_ly.pixels_u16.as_ref().unwrap();
        let mse: f64 = px16
            .iter()
            .zip(rec16.iter())
            .map(|(&a, &b)| {
                let d = a as f64 - b as f64;
                d * d
            })
            .sum::<f64>()
            / (px16.len() as f64);
        let psnr = 10.0 * ((max_v as f64 * max_v as f64) / mse.max(1e-12)).log10();
        assert!(psnr > 34.0, "{bit_depth}-bit lossy HDR PSNR too low: {psnr:.2} dB");

        let q = TransformQuery {
            crop: Some((16, 16, 128, 96)),
            target_width: Some(64),
            rotate_cw: 1,
            sepia: true,
            ..TransformQuery::default()
        };
        let dec_q = decode_query(&ly_bytes, &q).unwrap();
        assert_eq!(dec_q.bit_depth, bit_depth);
        assert_eq!((dec_q.width, dec_q.height), (48, 64));
        assert_eq!(dec_q.pixels_u16.as_ref().unwrap().len(), 48 * 64 * 3);
        assert_eq!(dec_q.icc_profile.as_deref(), Some(fake_p3.as_slice()));
    }
}

#[test]
fn v03_ui_palette_rle_adaptive_quant_and_cicp_exif_xmp_metadata() {
    use vimg::ffi::{
        vimg_extract_cicp, vimg_extract_exif_alloc, vimg_extract_xmp_alloc, vimg_free, VimgCicpC,
    };
    use vimg::{encode_u16, Cicp, Error, VimgHeader};

    // 1) Synthetic 512x512 RGBA dark-mode UI screenshot (sharp panels, borders, text blocks, solid alpha)
    let (uw, uh) = (512usize, 512usize);
    let mut ui_rgba = vec![0u8; uw * uh * 4];
    for y in 0..uh {
        for x in 0..uw {
            let (r, g, b) = if y < 48 {
                (30, 34, 42) // top navbar
            } else if x < 120 {
                (24, 27, 34) // sidebar
            } else if x == 120 || y == 48 {
                (68, 76, 92) // 1px crisp border
            } else if (x > 160 && x < 460) && (y > 90 && y < 380) && ((y / 14) % 2 == 0) && ((x / 8) % 3 != 0) {
                (212, 218, 228) // crisp simulated monospace text
            } else {
                (18, 21, 27) // editor background
            };
            let o = (y * uw + x) * 4;
            ui_rgba[o] = r;
            ui_rgba[o + 1] = g;
            ui_rgba[o + 2] = b;
            ui_rgba[o + 3] = 255;
        }
    }
    let ui_vimg = encode(
        &ui_rgba,
        uw as u32,
        uh as u32,
        4,
        &EncodeOptions { tile_size: 128, ..EncodeOptions::lossless() },
    )
    .unwrap();
    let ui_dec = decode(&ui_vimg).unwrap();
    assert_eq!(ui_dec.pixels, ui_rgba, "UI lossless palette+RLE roundtrip must be 100% bit-exact");
    // Raw RGBA = 1024 KiB; UI screenshot must compress > 40x smaller!
    assert!(
        ui_vimg.len() < 25 * 1024,
        "UI lossless expected <25 KiB with palette+RLE, got {} B",
        ui_vimg.len()
    );

    // 2) Tile-adaptive psychovisual quantization (CODEC_BLK4_AQ) on mixed smooth + noisy image
    let (mw, mh) = (512usize, 256usize);
    let mut mixed = vec![0u8; mw * mh * 3];
    for y in 0..mh {
        for x in 0..mw {
            let o = (y * mw + x) * 3;
            if x < 256 {
                // Smooth sky/skin gradient on left tiles
                mixed[o] = (120 + (x / 4)) as u8;
                mixed[o + 1] = (150 + (y / 4)) as u8;
                mixed[o + 2] = 210;
            } else {
                // High-frequency chaotic texture on right tiles
                let h = (((x * 73) ^ (y * 137)).wrapping_mul(2654435761)) >> 24;
                mixed[o] = (80 + (h % 140)) as u8;
                mixed[o + 1] = (60 + ((h.wrapping_mul(3)) % 140)) as u8;
                mixed[o + 2] = (50 + ((h.wrapping_mul(7)) % 140)) as u8;
            }
        }
    }
    let std_bytes = encode(
        &mixed,
        mw as u32,
        mh as u32,
        3,
        &EncodeOptions { quality: 80, tile_size: 128, adaptive_quant: false, threads: 1, ..EncodeOptions::default() },
    )
    .unwrap();
    let aq_1t = encode(
        &mixed,
        mw as u32,
        mh as u32,
        3,
        &EncodeOptions { quality: 80, tile_size: 128, adaptive_quant: true, threads: 1, ..EncodeOptions::default() },
    )
    .unwrap();
    let aq_4t = encode(
        &mixed,
        mw as u32,
        mh as u32,
        3,
        &EncodeOptions { quality: 80, tile_size: 128, adaptive_quant: true, threads: 4, ..EncodeOptions::default() },
    )
    .unwrap();
    assert_eq!(aq_1t, aq_4t, "AQ encoding must be 100% deterministic across 1T/4T");
    assert!(
        aq_1t.len() < std_bytes.len(),
        "AQ should save bits on chaotic tiles: aq={} B vs std={} B",
        aq_1t.len(),
        std_bytes.len()
    );
    // Smooth left half must stay razor-sharp under AQ!
    let dec_aq = decode_query(&aq_1t, &TransformQuery { crop: Some((0, 0, 256, 256)), ..Default::default() }).unwrap();
    let mut left_ref = Vec::with_capacity(256 * 256 * 3);
    for y in 0..256 {
        left_ref.extend_from_slice(&mixed[y * mw * 3..y * mw * 3 + 256 * 3]);
    }
    let smooth_psnr = psnr_db(&left_ref, &dec_aq.pixels);
    assert!(smooth_psnr > 44.0, "smooth region PSNR under AQ should be >44 dB, got {smooth_psnr:.2}");

    // 3) CICP HDR (BT.2020/PQ) + EXIF + XMP trailing TLV metadata + C ABI + zero-decode crop + tamper check
    let fake_exif = b"II*\0\x08\0\0\0EXIF_CAMERA_DATA".to_vec();
    let fake_xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">VIMG_V03</x:xmpmeta>".to_vec();
    let px10 = vec![512u16; 128 * 128 * 3];
    let hdr_meta_vimg = encode_u16(
        &px10,
        128,
        128,
        3,
        &EncodeOptions {
            bit_depth: 10,
            tile_size: 64,
            cicp: Some(Cicp::BT2020_PQ),
            exif: Some(fake_exif.clone()),
            xmp: Some(fake_xmp.clone()),
            ..EncodeOptions::lossless()
        },
    )
    .unwrap();
    let hdr = VimgHeader::parse(&hdr_meta_vimg).unwrap();
    assert!(hdr.has_meta());

    let full = decode(&hdr_meta_vimg).unwrap();
    let m = full.metadata.as_ref().expect("metadata must be decoded");
    assert_eq!(m.cicp, Some(Cicp::BT2020_PQ));
    assert_eq!(m.exif.as_deref(), Some(fake_exif.as_slice()));
    assert_eq!(m.xmp.as_deref(), Some(fake_xmp.as_slice()));

    // Partial decode with load_metadata = false skips trailing metadata cleanly
    let skip_meta = decode_query(
        &hdr_meta_vimg,
        &TransformQuery { load_metadata: false, ..Default::default() },
    )
    .unwrap();
    assert!(skip_meta.metadata.is_none());

    // Zero-decode tile crop preserves CICP + EXIF + XMP verbatim!
    let (cropped, _) = crop_vimg_zero_decode(&SliceRangeSource(&hdr_meta_vimg), 0, 0, 1, 1, false).unwrap();
    let dec_cropped = decode(&cropped).unwrap();
    assert_eq!(dec_cropped.metadata, full.metadata);

    // Tampering with trailing metadata triggers CRC32C ChecksumMismatch!
    let mut bad_meta = hdr_meta_vimg.clone();
    let last_payload = bad_meta.len() - 6;
    bad_meta[last_payload] ^= 0xFF;
    assert!(matches!(decode(&bad_meta), Err(Error::ChecksumMismatch { .. })));

    // C ABI metadata extraction parity
    unsafe {
        let mut cicp_c = VimgCicpC::default();
        assert_eq!(vimg_extract_cicp(hdr_meta_vimg.as_ptr(), hdr_meta_vimg.len(), &mut cicp_c), 0);
        assert_eq!(cicp_c.present, 1);
        assert_eq!(cicp_c.color_primaries, 9);
        assert_eq!(cicp_c.transfer_characteristics, 16);

        let eb = vimg_extract_exif_alloc(hdr_meta_vimg.as_ptr(), hdr_meta_vimg.len());
        assert_eq!(eb.status, 0);
        assert_eq!(core::slice::from_raw_parts(eb.ptr, eb.len), fake_exif.as_slice());
        vimg_free(eb);

        let xb = vimg_extract_xmp_alloc(hdr_meta_vimg.as_ptr(), hdr_meta_vimg.len());
        assert_eq!(xb.status, 0);
        assert_eq!(core::slice::from_raw_parts(xb.ptr, xb.len), fake_xmp.as_slice());
        vimg_free(xb);
    }
}

#[test]
fn v03_zero_decode_downscale_progressive_color_bicubic_and_ssim() {
    use vimg::{downscale_vimg_zero_decode, ssim_score, Cicp, ProgressiveDecoder};

    let (w, h) = (640usize, 480usize);
    let px = synth_photo_rgb(w, h);
    let fake_icc = b"ICC_PROFILE_V03_TEST_PAYLOAD".to_vec();
    let fake_exif = b"II*\0GPS_SECRET_EXIF".to_vec();
    let fake_xmp = b"<xmp>PRIVATE_XMP</xmp>".to_vec();

    let vimg_bytes = encode(
        &px,
        w as u32,
        h as u32,
        3,
        &EncodeOptions {
            quality: 85,
            tile_size: 256,
            dwt_levels: 3,
            adaptive_quant: true,
            icc_profile: Some(fake_icc.clone()),
            cicp: Some(Cicp::DISPLAY_P3),
            exif: Some(fake_exif),
            xmp: Some(fake_xmp),
            ..EncodeOptions::default()
        },
    )
    .unwrap();

    // 1) Zero-decode pyramid downscale (`vimg -> vimg`) at L0, L1, L2 with pixel parity & privacy strip_meta
    let src = SliceRangeSource(&vimg_bytes);
    for (lvl, (ew, eh)) in [(0u8, (80u32, 60u32)), (1, (160, 120)), (2, (320, 240))] {
        let (down, stats) = downscale_vimg_zero_decode(&src, lvl, false, true).unwrap();
        assert!(
            stats.bytes_fetched < vimg_bytes.len() as u64,
            "L{lvl} zero-decode downscale must fetch strictly less than full file"
        );
        let dh = VimgHeader::parse(&down).unwrap();
        assert_eq!((dh.width, dh.height, dh.dwt_levels), (ew, eh, lvl));
        let dec_down = decode(&down).unwrap();
        let dec_ref = decode_query(
            &vimg_bytes,
            &TransformQuery { force_level: Some(lvl), ..Default::default() },
        )
        .unwrap();
        assert_eq!(
            dec_down.pixels, dec_ref.pixels,
            "zero-decode downscale L{lvl} must match force_level={lvl} pixel-for-pixel"
        );
        // strip_meta = true keeps ICC & CICP color accuracy while stripping private EXIF & XMP!
        assert_eq!(dec_down.icc_profile.as_deref(), Some(fake_icc.as_slice()));
        let meta = dec_down.metadata.as_ref().expect("CICP must remain after strip_meta");
        assert_eq!(meta.cicp, Some(Cicp::DISPLAY_P3));
        assert!(meta.exif.is_none(), "EXIF must be stripped when strip_meta=true");
        assert!(meta.xmp.is_none(), "XMP must be stripped when strip_meta=true");
    }

    // Grayscale zero-decode downscale drops chroma completely
    let (down_gray, _) = downscale_vimg_zero_decode(&src, 1, true, true).unwrap();
    let dec_gray = decode(&down_gray).unwrap();
    assert_eq!((dec_gray.width, dec_gray.height, dec_gray.channels), (160, 120, 1));

    // 2) Stateful ProgressiveDecoder (`L0 -> L1 -> L2 -> L3`) matches direct decode bit-for-bit without re-fetching!
    let mut prog = ProgressiveDecoder::open(&src, TransformQuery::default()).unwrap();
    assert_eq!(prog.total_levels(), 4);
    for expected_lvl in 0..=3u8 {
        let frame = prog.next_level(&src).unwrap().expect("expected progressive frame");
        assert_eq!(frame.decoded_level, expected_lvl);
        let direct = decode_query(
            &vimg_bytes,
            &TransformQuery { force_level: Some(expected_lvl), ..Default::default() },
        )
        .unwrap();
        assert_eq!(
            frame.pixels, direct.pixels,
            "ProgressiveDecoder L{expected_lvl} must match direct decode bit-for-bit"
        );
    }
    assert!(prog.next_level(&src).unwrap().is_none(), "exhausted ProgressiveDecoder must return None");
    // At most one chunk can straddle the initial 4 KiB header prefix boundary; earlier levels are never re-fetched.
    assert!(
        prog.stats().bytes_fetched <= vimg_bytes.len() as u64 + 4096,
        "ProgressiveDecoder must never re-fetch previous levels (fetched {} vs total {})",
        prog.stats().bytes_fetched,
        vimg_bytes.len()
    );

    // 3) Compressed-domain Saturation, Temperature, and Catmull-Rom Bicubic arbitrary resize
    let dec_desat = decode_query(
        &vimg_bytes,
        &TransformQuery { saturation: 0.0, ..Default::default() },
    )
    .unwrap();
    for rgb in dec_desat.pixels.as_chunks::<3>().0 {
        assert_eq!(rgb[0], rgb[1]);
        assert_eq!(rgb[1], rgb[2]);
    }
    let dec_warm = decode_query(
        &vimg_bytes,
        &TransformQuery { temperature: 0.8, ..Default::default() },
    )
    .unwrap();
    let dec_cool = decode_query(
        &vimg_bytes,
        &TransformQuery { temperature: -0.8, ..Default::default() },
    )
    .unwrap();
    let avg_rb = |p: &[u8]| -> (f64, f64) {
        let n = (p.len() / 3) as f64;
        let r: f64 = p.as_chunks::<3>().0.iter().map(|c| c[0] as f64).sum::<f64>() / n;
        let b: f64 = p.as_chunks::<3>().0.iter().map(|c| c[2] as f64).sum::<f64>() / n;
        (r, b)
    };
    let (wr, wb) = avg_rb(&dec_warm.pixels);
    let (cr, cb) = avg_rb(&dec_cool.pixels);
    assert!(wr > cr + 10.0 && wb < cb - 10.0, "warm vs cool temperature shift expected: warm=({wr:.1},{wb:.1}) cool=({cr:.1},{cb:.1})");

    let dec_bicubic = decode_query(
        &vimg_bytes,
        &TransformQuery { target_width: Some(333), ..Default::default() },
    )
    .unwrap();
    assert_eq!(dec_bicubic.width, 333);
    assert_eq!(dec_bicubic.height, 250);

    // 4) Pure-Rust 8x8 windowed SSIM
    assert!((ssim_score(&px, &px, w as u32, h as u32, 3) - 1.0).abs() < 1e-9);
    let full = decode(&vimg_bytes).unwrap();
    let ssim_lossy = ssim_score(&px, &full.pixels, w as u32, h as u32, 3);
    assert!(ssim_lossy > 0.90, "q=85 SSIM should be >0.90, got {ssim_lossy:.5}");
    let inv: Vec<u8> = px.iter().map(|&v| 255 - v).collect();
    assert!(ssim_score(&px, &inv, w as u32, h as u32, 3) < 0.25);
}

#[test]
fn v04_true_16bit_hdr_and_vima1_range_animation() {
    use vimg::{
        encode_animation, encode_u16, trim_vima_zero_decode, AnimFrameInput, ProgressiveDecoder,
        RangeStats, VimaIndex,
    };

    // 1) True 16-bit (0..=65535) lossless (bit-exact across full ±65535 Co/Cg range!), lossy, crop & progressive
    let (w, h) = (160usize, 128usize);
    let mut px16 = vec![0u16; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 3;
            // Include extreme corners (0 and 65535) so Co = R - B hits ±65535!
            px16[o] = if (x + y) % 19 == 0 { 65535 } else { ((x * 410 + y * 97) % 65536) as u16 };
            px16[o + 1] = ((x * 211 + y * 503) % 65536) as u16;
            px16[o + 2] = if (x + y) % 19 == 0 { 0 } else { ((y * 512 + x * 131) % 65536) as u16 };
        }
    }
    let v16_ll = encode_u16(
        &px16,
        w as u32,
        h as u32,
        3,
        &EncodeOptions { bit_depth: 16, tile_size: 64, ..EncodeOptions::lossless() },
    )
    .unwrap();
    let hdr16 = VimgHeader::parse(&v16_ll).unwrap();
    assert_eq!(hdr16.bit_depth(), 16);
    assert_eq!(hdr16.max_sample(), 65535);
    let dec16_ll = decode(&v16_ll).unwrap();
    assert_eq!(dec16_ll.bit_depth, 16);
    assert_eq!(
        dec16_ll.pixels_u16.as_deref(),
        Some(px16.as_slice()),
        "16-bit lossless must be 100% bit-exact across 0..=65535"
    );

    // 16-bit ProgressiveDecoder reaches bit-exact L3 match!
    let mut prog16 = ProgressiveDecoder::open(&SliceRangeSource(&v16_ll), TransformQuery::default()).unwrap();
    let mut last_frame = None;
    while let Some(f) = prog16.next_level(&SliceRangeSource(&v16_ll)).unwrap() {
        last_frame = Some(f);
    }
    assert_eq!(last_frame.unwrap().pixels_u16.as_deref(), Some(px16.as_slice()));

    // 16-bit lossy HDR roundtrip (high PSNR in 16-bit domain)
    let mut smooth16 = vec![0u16; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 3;
            smooth16[o] = ((x * 65535) / w) as u16;
            smooth16[o + 1] = ((y * 65535) / h) as u16;
            smooth16[o + 2] = (((x + y) * 32767) / (w + h)) as u16;
        }
    }
    let v16_q = encode_u16(
        &smooth16,
        w as u32,
        h as u32,
        3,
        &EncodeOptions { bit_depth: 16, quality: 85, tile_size: 64, ..EncodeOptions::default() },
    )
    .unwrap();
    let dec16_q = decode(&v16_q).unwrap();
    let mse: f64 = smooth16
        .iter()
        .zip(dec16_q.pixels_u16.as_ref().unwrap().iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / smooth16.len() as f64;
    let psnr16 = 10.0 * (65535.0 * 65535.0 / mse.max(1e-9)).log10();
    assert!(psnr16 > 40.0, "16-bit lossy smooth gradient PSNR too low: {psnr16:.2} dB");

    // 2) VIMA1 Range-native animation: encode, random-access partial crop of frame #4, and zero-decode trim!
    let (aw, ah) = (256usize, 256usize);
    let raw_frames: Vec<Vec<u8>> = (0..6u8)
        .map(|fi| {
            let mut buf = synth_photo_rgb(aw, ah);
            // Shift each frame distinctly so frame parity is unambiguous
            for v in &mut buf {
                *v = v.wrapping_add(fi.wrapping_mul(37));
            }
            buf
        })
        .collect();
    let inputs: Vec<AnimFrameInput<'_>> = raw_frames
        .iter()
        .enumerate()
        .map(|(i, px)| AnimFrameInput {
            pixels: px,
            width: aw as u32,
            height: ah as u32,
            channels: 3,
            duration_ms: 40 + (i as u32) * 10,
        })
        .collect();
    let vima = encode_animation(
        &inputs,
        0,
        &EncodeOptions { tile_size: 64, ..EncodeOptions::lossless() },
    )
    .unwrap();

    let src = SliceRangeSource(&vima);
    let mut dir_stats = RangeStats::default();
    let idx = VimaIndex::read_from_source(&src, &mut dir_stats).unwrap();
    assert_eq!(idx.header.frame_count, 6);
    assert_eq!(idx.total_duration_ms(), 40 + 50 + 60 + 70 + 80 + 90);

    // Random-access partial 64x64 crop inside frame #4 without touching frames 0..3 or 5!
    let crop_q = TransformQuery { crop: Some((64, 128, 64, 64)), ..Default::default() };
    let f4_crop = idx.decode_frame_from_source(&src, 4, &crop_q).unwrap();
    assert_eq!((f4_crop.width, f4_crop.height), (64, 64));
    let total_fetched = dir_stats.bytes_fetched + f4_crop.stats.bytes_fetched;
    assert!(
        total_fetched < (vima.len() as u64) / 6,
        "Single-tile crop of 1 frame in 6-frame VIMA1 should fetch <1/6 of file (got {total_fetched} / {})",
        vima.len()
    );
    // Verify bit-exact pixel parity against frame #4 reference crop!
    let mut expected_crop = Vec::with_capacity(64 * 64 * 3);
    for y in 128..192 {
        expected_crop.extend_from_slice(&raw_frames[4][(y * aw + 64) * 3..(y * aw + 128) * 3]);
    }
    assert_eq!(f4_crop.pixels, expected_crop, "VIMA1 random-access frame crop must be bit-exact");

    // Zero-decode trim: extract frames [2, 3, 4] verbatim!
    let (trimmed, trim_stats) = trim_vima_zero_decode(&src, 2, 3).unwrap();
    assert!(trim_stats.bytes_fetched < (vima.len() as u64) * 2 / 3);
    let t_src = SliceRangeSource(&trimmed);
    let mut t_stats = RangeStats::default();
    let t_idx = VimaIndex::read_from_source(&t_src, &mut t_stats).unwrap();
    assert_eq!(t_idx.header.frame_count, 3);
    assert_eq!(t_idx.frames[0].duration_ms, 60);
    let t_f2 = t_idx.decode_frame_from_source(&t_src, 2, &TransformQuery::default()).unwrap();
    assert_eq!(t_f2.pixels, raw_frames[4], "Trimmed frame #2 must equal original frame #4 bit-for-bit");
}
