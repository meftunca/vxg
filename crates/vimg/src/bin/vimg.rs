//! `vimg` CLI: encode, decode, info, zero-decode crop, HTTP Range Edge Proxy (`serve`), and `bench`.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Cursor, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use image::{DynamicImage, GenericImageView, ImageFormat, RgbImage, RgbaImage};
use vimg::{
    crop_vimg_zero_decode, decode_from_source, encode, psnr_db, EncodeOptions, FileRangeSource,
    HttpRangeSource, RangeSource, RangeStats, SliceRangeSource, TransformQuery, VimgIndex,
};

fn usage() {
    eprintln!(
        "vimg {} — HTTP-Range-Native Hierarchical Wavelet Image Codec & Compressed-Domain Engine\n\n\
         Usage:\n  \
           vimg encode <in.png|jpg|webp|bmp> <out.vimg> [--quality 1..100] [--tile 64|128|256] [--levels 1..4] [--no-420]\n  \
           vimg decode <in.vimg | http://.../img.vimg> <out.png|jpg|webp> [--w W] [--h H] [--crop x,y,w,h] [--gray] [--blur 0..1] [--sharpen 0..2] [--bright N] [--contrast F] [--sepia] [--rot 90|180|270]\n  \
           vimg crop   <in.vimg | http://.../img.vimg> <out.vimg> --tiles tx0,ty0,ntx,nty [--gray]\n  \
           vimg info   <in.vimg | http://.../img.vimg>\n  \
           vimg serve  <dir> [--port 8090]    (HTTP/1.1 Range Origin + Live Compressed-Domain Edge Proxy)\n  \
           vimg bench  <in.png|jpg|bmp> [--quality 85] [--tile 256]",
        env!("CARGO_PKG_VERSION")
    );
}

fn open_source(path_or_url: &str) -> Result<Box<dyn RangeSource>, String> {
    if path_or_url.starts_with("http://") {
        HttpRangeSource::connect(path_or_url)
            .map(|s| Box::new(s) as Box<dyn RangeSource>)
            .map_err(|e| e.to_string())
    } else {
        FileRangeSource::open(path_or_url)
            .map(|s| Box::new(s) as Box<dyn RangeSource>)
            .map_err(|e| e.to_string())
    }
}

fn load_pixels(path: &str) -> Result<(Vec<u8>, u32, u32, u8), String> {
    let img = image::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let (w, h) = img.dimensions();
    if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        Ok((rgba.into_raw(), w, h, 4))
    } else {
        let rgb = img.to_rgb8();
        Ok((rgb.into_raw(), w, h, 3))
    }
}

fn save_pixels(path: &str, pixels: &[u8], w: u32, h: u32, ch: u8) -> Result<(), String> {
    let dyn_img = match ch {
        1 => DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w, h, pixels.to_vec()).ok_or("bad gray buffer")?,
        ),
        3 => DynamicImage::ImageRgb8(
            RgbImage::from_raw(w, h, pixels.to_vec()).ok_or("bad rgb buffer")?,
        ),
        4 => DynamicImage::ImageRgba8(
            RgbaImage::from_raw(w, h, pixels.to_vec()).ok_or("bad rgba buffer")?,
        ),
        _ => return Err(format!("unsupported channels {ch}")),
    };
    dyn_img.save(path).map_err(|e| format!("save {path}: {e}"))
}

fn encode_png_bytes(pixels: &[u8], w: u32, h: u32, ch: u8) -> Result<Vec<u8>, String> {
    let dyn_img = match ch {
        1 => DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w, h, pixels.to_vec()).ok_or("bad gray")?,
        ),
        3 => DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, pixels.to_vec()).ok_or("bad rgb")?),
        4 => DynamicImage::ImageRgba8(RgbaImage::from_raw(w, h, pixels.to_vec()).ok_or("bad rgba")?),
        _ => return Err("bad ch".into()),
    };
    let mut out = Cursor::new(Vec::new());
    dyn_img.write_to(&mut out, ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}

fn parse_quad(s: &str) -> Result<(u32, u32, u32, u32), String> {
    let p: Vec<u32> = s
        .split(',')
        .map(|x| x.trim().parse::<u32>().map_err(|_| format!("bad int in '{s}'")))
        .collect::<Result<_, _>>()?;
    if p.len() != 4 {
        return Err(format!("expected 4 comma-separated integers, got '{s}'"));
    }
    Ok((p[0], p[1], p[2], p[3]))
}

fn cmd_encode(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("encode requires <in> <out.vimg>".into());
    }
    let mut opts = EncodeOptions::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" | "-q" => {
                i += 1;
                opts.quality = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --quality")?;
                if opts.quality >= 100 {
                    opts.chroma_420 = false;
                }
            }
            "--tile" => {
                i += 1;
                opts.tile_size = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --tile")?;
            }
            "--levels" => {
                i += 1;
                opts.dwt_levels = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --levels")?;
            }
            "--no-420" => opts.chroma_420 = false,
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    let (pixels, w, h, ch) = load_pixels(&args[0])?;
    let t0 = Instant::now();
    let vimg_bytes = encode(&pixels, w, h, ch, &opts).map_err(|e| e.to_string())?;
    let dt = t0.elapsed().as_secs_f64();
    fs::write(&args[1], &vimg_bytes).map_err(|e| e.to_string())?;
    let bpp = (vimg_bytes.len() as f64 * 8.0) / ((w as f64) * (h as f64));
    let mpix_s = ((w as f64) * (h as f64) / 1e6) / dt.max(1e-6);
    println!(
        "Encoded {}x{}x{} -> {} ({:.1} KiB, {:.2} bpp, q={}, tile={}, lvls={}) in {:.1} ms ({:.1} MPix/s)",
        w,
        h,
        ch,
        args[1],
        vimg_bytes.len() as f64 / 1024.0,
        bpp,
        opts.quality,
        opts.tile_size,
        opts.dwt_levels,
        dt * 1000.0,
        mpix_s
    );
    Ok(())
}

fn cmd_decode(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("decode requires <in.vimg|url> <out.png|jpg>".into());
    }
    let mut q = TransformQuery::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--w" => { i += 1; q.target_width = Some(args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --w")?); }
            "--h" => { i += 1; q.target_height = Some(args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --h")?); }
            "--crop" => { i += 1; q.crop = Some(parse_quad(args.get(i).ok_or("missing --crop")?)?); }
            "--gray" => q.grayscale = true,
            "--sepia" => q.sepia = true,
            "--blur" => { i += 1; q.blur = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --blur")?; }
            "--sharpen" => { i += 1; q.sharpen = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --sharpen")?; }
            "--bright" => { i += 1; q.brightness = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --bright")?; }
            "--contrast" => { i += 1; q.contrast = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --contrast")?; }
            "--rot" => {
                i += 1;
                let deg: u32 = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --rot")?;
                q.rotate_cw = ((deg / 90) & 3) as u8;
            }
            "--flip-h" => q.flip_h = true,
            "--flip-v" => q.flip_v = true,
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    let src = open_source(&args[0])?;
    let t0 = Instant::now();
    let img = decode_from_source(src.as_ref(), &q).map_err(|e| e.to_string())?;
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    save_pixels(&args[1], &img.pixels, img.width, img.height, img.channels)?;
    println!(
        "Decoded {}x{}x{} (level={}, tiles={}/{}, fetched {:.1} KiB / {:.1} KiB = {:.1}% in {} reqs) in {:.2} ms -> {}",
        img.width,
        img.height,
        img.channels,
        img.decoded_level,
        img.tiles_fetched,
        img.tiles_total,
        img.stats.bytes_fetched as f64 / 1024.0,
        img.stats.total_file_bytes as f64 / 1024.0,
        img.stats.fetched_pct(),
        img.stats.requests,
        dt,
        args[1]
    );
    Ok(())
}

fn cmd_info(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("info requires <in.vimg|url>")?;
    let src = open_source(path)?;
    let mut stats = RangeStats::default();
    let idx = VimgIndex::read_from_source(src.as_ref(), &mut stats).map_err(|e| e.to_string())?;
    let h = &idx.header;
    let bpp = (h.total_bytes as f64 * 8.0) / ((h.width as f64) * (h.height as f64));
    println!(
        "VIMG v{} | {}x{}x{} | quality={} ({}) | chroma={} | tile={} ({}x{}={}) | size={:.1} KiB ({:.2} bpp) | dir={:.1} KiB",
        h.version,
        h.width,
        h.height,
        h.channels,
        h.quality,
        if h.is_lossless() { "lossless" } else { "lossy" },
        if h.is_chroma_420() { "4:2:0" } else { "4:4:4" },
        h.tile_size,
        h.tiles_x,
        h.tiles_y,
        h.n_tiles(),
        h.total_bytes as f64 / 1024.0,
        bpp,
        h.dir_bytes as f64 / 1024.0
    );
    let lvls = idx.level_byte_sizes();
    let mut cum = (48 + h.dir_bytes) as u64;
    for (l, &sz) in lvls.iter().enumerate() {
        cum += sz;
        let (lw, lh) = h.image_level_dims(l);
        println!(
            "  Level {l}: {:>5}x{:<5} | band {:>7.1} KiB | cumulative {:>7.1} KiB ({:>5.1}%)",
            lw,
            lh,
            sz as f64 / 1024.0,
            cum as f64 / 1024.0,
            (cum as f64) * 100.0 / (h.total_bytes as f64)
        );
    }
    Ok(())
}

fn cmd_crop(args: &[String]) -> Result<(), String> {
    if args.len() < 4 {
        return Err("crop requires <in.vimg|url> <out.vimg> --tiles tx0,ty0,ntx,nty [--gray]".into());
    }
    let mut tiles = None;
    let mut gray = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--tiles" => { i += 1; tiles = Some(parse_quad(args.get(i).ok_or("missing --tiles")?)?); }
            "--gray" => gray = true,
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    let (tx0, ty0, ntx, nty) = tiles.ok_or("missing --tiles")?;
    let src = open_source(&args[0])?;
    let t0 = Instant::now();
    let (out, stats) = crop_vimg_zero_decode(src.as_ref(), tx0, ty0, ntx, nty, gray).map_err(|e| e.to_string())?;
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    fs::write(&args[1], &out).map_err(|e| e.to_string())?;
    println!(
        "Zero-decode crop -> {} ({:.1} KiB) fetched {:.1}% ({:.1} KiB / {:.1} KiB, {} reqs) in {:.2} ms",
        args[1],
        out.len() as f64 / 1024.0,
        stats.fetched_pct(),
        stats.bytes_fetched as f64 / 1024.0,
        stats.total_file_bytes as f64 / 1024.0,
        stats.requests,
        dt
    );
    Ok(())
}

fn cmd_bench(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("bench requires <image>")?;
    let mut quality = 85u8;
    let mut tile_size = 256u32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--quality" | "-q" => { i += 1; quality = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --quality")?; }
            "--tile" => { i += 1; tile_size = args.get(i).and_then(|s| s.parse().ok()).ok_or("bad --tile")?; }
            other => return Err(format!("unknown flag {other}")),
        }
        i += 1;
    }
    let (pixels, w, h, ch) = load_pixels(path)?;
    let raw_bytes = pixels.len();
    let raw_dyn = if ch == 4 {
        DynamicImage::ImageRgba8(RgbaImage::from_raw(w, h, pixels.clone()).unwrap())
    } else {
        DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, pixels.clone()).unwrap())
    };

    // Baseline PNG & JPEG (in-memory)
    let mut png_buf = Cursor::new(Vec::new());
    let t_png_e = Instant::now();
    raw_dyn.write_to(&mut png_buf, ImageFormat::Png).map_err(|e| e.to_string())?;
    let png_enc_ms = t_png_e.elapsed().as_secs_f64() * 1000.0;
    let png_bytes = png_buf.into_inner();
    let t_png_d = Instant::now();
    let _ = image::load_from_memory_with_format(&png_bytes, ImageFormat::Png).map_err(|e| e.to_string())?;
    let png_dec_ms = t_png_d.elapsed().as_secs_f64() * 1000.0;

    let mut jpg_buf = Cursor::new(Vec::new());
    let t_jpg_e = Instant::now();
    raw_dyn.to_rgb8().write_to(&mut jpg_buf, ImageFormat::Jpeg).map_err(|e| e.to_string())?;
    let jpg_enc_ms = t_jpg_e.elapsed().as_secs_f64() * 1000.0;
    let jpg_bytes = jpg_buf.into_inner();
    let t_jpg_d = Instant::now();
    let jpg_dec = image::load_from_memory_with_format(&jpg_bytes, ImageFormat::Jpeg).map_err(|e| e.to_string())?.to_rgb8();
    let jpg_dec_ms = t_jpg_d.elapsed().as_secs_f64() * 1000.0;
    let jpg_psnr = if ch == 3 { psnr_db(&pixels, jpg_dec.as_raw()) } else { 0.0 };

    // VIMG Lossless (q=100)
    let t_vl_e = Instant::now();
    let vl_bytes = encode(&pixels, w, h, ch, &EncodeOptions { tile_size, ..EncodeOptions::lossless() })
        .map_err(|e| e.to_string())?;
    let vl_enc_ms = t_vl_e.elapsed().as_secs_f64() * 1000.0;
    let t_vl_d = Instant::now();
    let vl_dec = vimg::decode(&vl_bytes).map_err(|e| e.to_string())?;
    let vl_dec_ms = t_vl_d.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(vl_dec.pixels, pixels, "VIMG lossless must be 100% bit-exact!");

    // VIMG Lossy (q=quality)
    let opts = EncodeOptions { quality, tile_size, dwt_levels: 3, chroma_420: true, checksum: true, threads: 0 };
    let t_vq_e = Instant::now();
    let vq_bytes = encode(&pixels, w, h, ch, &opts).map_err(|e| e.to_string())?;
    let vq_enc_ms = t_vq_e.elapsed().as_secs_f64() * 1000.0;
    let t_vq_d = Instant::now();
    let vq_dec = vimg::decode(&vq_bytes).map_err(|e| e.to_string())?;
    let vq_dec_ms = t_vq_d.elapsed().as_secs_f64() * 1000.0;
    let vq_psnr = psnr_db(&pixels, &vq_dec.pixels);

    // Partial compressed-domain remote queries (via SliceRangeSource accounting)
    let src = SliceRangeSource(&vq_bytes);
    let q_thumb = TransformQuery { target_width: Some((w / 4).max(64)), ..TransformQuery::default() };
    let t_th = Instant::now();
    let dec_th = decode_from_source(&src, &q_thumb).map_err(|e| e.to_string())?;
    let th_ms = t_th.elapsed().as_secs_f64() * 1000.0;

    let cw = 256u32.min(w);
    let ch_box = 256u32.min(h);
    let q_crop = TransformQuery {
        crop: Some(((w - cw) / 2, (h - ch_box) / 2, cw, ch_box)),
        ..TransformQuery::default()
    };
    let t_cr = Instant::now();
    let dec_cr = decode_from_source(&src, &q_crop).map_err(|e| e.to_string())?;
    let cr_ms = t_cr.elapsed().as_secs_f64() * 1000.0;

    let q_gray = TransformQuery {
        target_width: Some((w / 2).max(64)),
        grayscale: true,
        blur: 0.6,
        ..TransformQuery::default()
    };
    let t_gr = Instant::now();
    let dec_gr = decode_from_source(&src, &q_gray).map_err(|e| e.to_string())?;
    let gr_ms = t_gr.elapsed().as_secs_f64() * 1000.0;

    // Classic baseline: full JPEG decode + resize/crop
    let t_jth = Instant::now();
    let j_full = image::load_from_memory_with_format(&jpg_bytes, ImageFormat::Jpeg).map_err(|e| e.to_string())?;
    let _ = j_full.resize_exact((w / 4).max(64), (h / 4).max(64), image::imageops::FilterType::Triangle);
    let jth_ms = t_jth.elapsed().as_secs_f64() * 1000.0;

    let t_jcr = Instant::now();
    let j_full2 = image::load_from_memory_with_format(&jpg_bytes, ImageFormat::Jpeg).map_err(|e| e.to_string())?;
    let _ = j_full2.crop_imm((w - cw) / 2, (h - ch_box) / 2, cw, ch_box).to_rgb8();
    let jcr_ms = t_jcr.elapsed().as_secs_f64() * 1000.0;

    let mpix = (w as f64) * (h as f64);
    println!("=== VIMG Benchmark: {} ({}x{}x{}, raw {:.1} KiB) ===", path, w, h, ch, raw_bytes as f64 / 1024.0);
    println!("{:<22} {:>10} {:>8} {:>9} {:>10} {:>10}", "Codec", "Size KiB", "bpp", "PSNR dB", "Enc ms", "Dec ms");
    println!("{:<22} {:>10.1} {:>8.2} {:>9} {:>10.2} {:>10.2}", "PNG (lossless)", png_bytes.len() as f64 / 1024.0, png_bytes.len() as f64 * 8.0 / mpix, "INF", png_enc_ms, png_dec_ms);
    println!("{:<22} {:>10.1} {:>8.2} {:>9} {:>10.2} {:>10.2}", "VIMG Lossless (q=100)", vl_bytes.len() as f64 / 1024.0, vl_bytes.len() as f64 * 8.0 / mpix, "INF", vl_enc_ms, vl_dec_ms);
    println!("{:<22} {:>10.1} {:>8.2} {:>9.2} {:>10.2} {:>10.2}", "JPEG (default q=75)", jpg_bytes.len() as f64 / 1024.0, jpg_bytes.len() as f64 * 8.0 / mpix, jpg_psnr, jpg_enc_ms, jpg_dec_ms);
    println!("{:<22} {:>10.1} {:>8.2} {:>9.2} {:>10.2} {:>10.2}", format!("VIMG Lossy (q={quality})"), vq_bytes.len() as f64 / 1024.0, vq_bytes.len() as f64 * 8.0 / mpix, vq_psnr, vq_enc_ms, vq_dec_ms);
    println!("\n=== Compressed-Domain Partial Queries vs Classic Full-Decode ===");
    println!(
        "1/4 Resize ({}x{}):  VIMG {:.2} ms (fetched {:.1}% = {:.1} KiB, lvl {})  vs  JPEG full+resize {:.2} ms (100% = {:.1} KiB)",
        dec_th.width, dec_th.height, th_ms, dec_th.stats.fetched_pct(), dec_th.stats.bytes_fetched as f64 / 1024.0, dec_th.decoded_level, jth_ms, jpg_bytes.len() as f64 / 1024.0
    );
    println!(
        "Center Crop ({}x{}): VIMG {:.2} ms (fetched {:.1}% = {:.1} KiB, {}/{} tiles) vs  JPEG full+crop   {:.2} ms (100% = {:.1} KiB)",
        dec_cr.width, dec_cr.height, cr_ms, dec_cr.stats.fetched_pct(), dec_cr.stats.bytes_fetched as f64 / 1024.0, dec_cr.tiles_fetched, dec_cr.tiles_total, jcr_ms, jpg_bytes.len() as f64 / 1024.0
    );
    println!(
        "1/2 Gray+Blur ({}x{}): VIMG {:.2} ms (fetched {:.1}% = {:.1} KiB, 0 chroma!)",
        dec_gr.width, dec_gr.height, gr_ms, dec_gr.stats.fetched_pct(), dec_gr.stats.bytes_fetched as f64 / 1024.0
    );
    Ok(())
}

const PLAYGROUND_HTML: &str = include_str!("../../web/playground.html");

fn handle_http_client(mut stream: TcpStream, root: &Path) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut req_line = String::new();
    if reader.read_line(&mut req_line).is_err() {
        return;
    }
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let target = parts[1];
    let mut range_hdr: Option<String> = None;
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

    let (raw_path, query_str) = target.split_once('?').unwrap_or((target, ""));
    let rel = raw_path.trim_start_matches('/');
    if rel.is_empty() {
        let hdr = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            PLAYGROUND_HTML.len()
        );
        let _ = stream.write_all(hdr.as_bytes());
        let _ = stream.write_all(PLAYGROUND_HTML.as_bytes());
        return;
    }
    if rel == "api/files" {
        let mut items = Vec::new();
        if let Ok(rd) = fs::read_dir(root) {
            for ent in rd.flatten() {
                let p = ent.path();
                if p.extension().and_then(|e| e.to_str()) == Some("vimg") {
                    if let Ok(mut f) = fs::File::open(&p) {
                        let mut buf = [0u8; 48];
                        if std::io::Read::read_exact(&mut f, &mut buf).is_ok() {
                            if let Ok(h) = vimg::VimgHeader::parse(&buf) {
                                let name = ent.file_name().to_string_lossy().replace('"', "");
                                items.push(format!(
                                    "{{\"name\":\"{name}\",\"width\":{},\"height\":{},\"tile_size\":{},\"tiles_x\":{},\"tiles_y\":{},\"dwt_levels\":{},\"total_bytes\":{}}}",
                                    h.width, h.height, h.tile_size, h.tiles_x, h.tiles_y, h.dwt_levels, h.total_bytes
                                ));
                            }
                        }
                    }
                }
            }
        }
        items.sort();
        let body = format!("[{}]", items.join(","));
        let hdr = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(hdr.as_bytes());
        let _ = stream.write_all(body.as_bytes());
        return;
    }
    if rel == "vimg.wasm" {
        let wasm_path = [
            root.join("vimg.wasm"),
            env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("../wasm32-unknown-unknown/release/vimg.wasm"))).unwrap_or_default(),
        ].into_iter().find(|p| p.is_file());
        if let Some(wp) = wasm_path {
            if let Ok(bytes) = fs::read(wp) {
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/wasm\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&bytes);
                return;
            }
        }
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        return;
    }
    if rel.contains("..") {
        let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
        return;
    }
    let rel_dec = rel.replace("%20", " ");
    let file_path = root.join(rel_dec);
    if !file_path.is_file() {
        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        return;
    }

    // If query params exist on a .vimg file, run live Compressed-Domain Edge Transform -> PNG!
    if !query_str.is_empty() && file_path.extension().and_then(|e| e.to_str()) == Some("vimg") {
        let mut q = TransformQuery::default();
        for pair in query_str.split('&') {
            let (k, raw_v) = pair.split_once('=').unwrap_or((pair, "1"));
            let v = raw_v.replace("%2C", ",").replace("%2c", ",");
            match k {
                "w" => q.target_width = v.parse().ok(),
                "h" => q.target_height = v.parse().ok(),
                "crop" => q.crop = parse_quad(&v).ok(),
                "gray" => q.grayscale = v != "0",
                "sepia" => q.sepia = v != "0",
                "blur" => q.blur = v.parse().unwrap_or(0.0),
                "sharpen" => q.sharpen = v.parse().unwrap_or(0.0),
                "bright" => q.brightness = v.parse().unwrap_or(0),
                "contrast" => q.contrast = v.parse().unwrap_or(1.0),
                "rot" => q.rotate_cw = ((v.parse::<u32>().unwrap_or(0) / 90) & 3) as u8,
                _ => {}
            }
        }
        let Ok(src) = FileRangeSource::open(&file_path) else {
            let _ = stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n");
            return;
        };
        let t0 = Instant::now();
        match decode_from_source(&src, &q)
            .map_err(|e| e.to_string())
            .and_then(|img| {
                let dec_ms = t0.elapsed().as_secs_f64() * 1000.0;
                encode_png_bytes(&img.pixels, img.width, img.height, img.channels).map(|b| (img, b, dec_ms))
            })
        {
            Ok((img, png, dec_ms)) => {
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: *\r\nX-VIMG-Fetched-Pct: {:.1}\r\nX-VIMG-Fetched-Bytes: {}\r\nX-VIMG-Total-Bytes: {}\r\nX-VIMG-Level: {}\r\nX-VIMG-Tiles-Fetched: {}\r\nX-VIMG-Tiles-Total: {}\r\nX-VIMG-Reqs: {}\r\nX-VIMG-Decode-Ms: {:.2}\r\nX-VIMG-Out-Dims: {}x{}\r\nConnection: close\r\n\r\n",
                    png.len(), img.stats.fetched_pct(), img.stats.bytes_fetched, img.stats.total_file_bytes,
                    img.decoded_level, img.tiles_fetched, img.tiles_total, img.stats.requests, dec_ms, img.width, img.height
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&png);
            }
            Err(_) => {
                let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
            }
        }
        return;
    }

    // Raw HTTP/1.1 Range Origin serving
    let Ok(data) = fs::read(&file_path) else {
        let _ = stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n");
        return;
    };
    let total = data.len();
    if method == "HEAD" {
        let hdr = format!(
            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.write_all(hdr.as_bytes());
        return;
    }
    if let Some(r) = range_hdr.and_then(|s| s.strip_prefix("bytes=").map(str::to_owned)) {
        if let Some((a, b)) = r.split_once('-') {
            if let (Ok(s), Ok(e)) = (a.parse::<usize>(), b.parse::<usize>()) {
                if s <= e && e < total {
                    let slice = &data[s..=e];
                    let hdr = format!(
                        "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        slice.len()
                    );
                    let _ = stream.write_all(hdr.as_bytes());
                    let _ = stream.write_all(slice);
                    return;
                }
            }
        }
    }
    let hdr = format!(
        "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
    );
    let _ = stream.write_all(hdr.as_bytes());
    let _ = stream.write_all(&data);
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let dir = PathBuf::from(args.first().ok_or("serve requires <dir>")?);
    let mut port = 8090u16;
    if args.len() >= 3 && args[1] == "--port" {
        port = args[2].parse().map_err(|_| "bad --port")?;
    }
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
    let root = Arc::new(dir);
    println!("vimg HTTP Range Origin + Edge Proxy listening on http://127.0.0.1:{port} (dir={})", root.display());
    for stream in listener.incoming().flatten() {
        let r = Arc::clone(&root);
        std::thread::spawn(move || handle_http_client(stream, &r));
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return;
    }
    let res = match args[0].as_str() {
        "encode" => cmd_encode(&args[1..]),
        "decode" => cmd_decode(&args[1..]),
        "info" => cmd_info(&args[1..]),
        "crop" => cmd_crop(&args[1..]),
        "bench" => cmd_bench(&args[1..]),
        "serve" => cmd_serve(&args[1..]),
        "-h" | "--help" => { usage(); Ok(()) }
        other => Err(format!("unknown subcommand '{other}'")),
    };
    if let Err(e) = res {
        eprintln!("vimg error: {e}");
        std::process::exit(1);
    }
}
