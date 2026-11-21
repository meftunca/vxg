//! Frozen deterministic golden wire-format vectors for `VIMG1`.
//! Locks exact byte lengths, CRC32C, and FNV-1a 64-bit hashes for both
//! lossless (`CDF 5/3`, `4:4:4`) and lossy (`CDF 9/7`, `4:2:0`) across `1T == 4T`
//! and across CPU architectures (`aarch64 NEON`, `x86_64 AVX2`, `wasm32`).

use vimg::checksum::crc32c;
use vimg::{decode, encode, psnr_db, EncodeOptions};

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn golden_fixture_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let smooth_r = ((x * 255) / w.max(1)) as i32;
            let smooth_g = ((y * 255) / h.max(1)) as i32;
            let edge = if ((x / 16) + (y / 16)) % 2 == 0 { 24 } else { -24 };
            let tex = (((x * 13 + y * 29) & 31) as i32) - 16;
            let r = (smooth_r + edge / 2 + tex / 4).clamp(0, 255) as u8;
            let g = (smooth_g - edge / 2 + tex / 3).clamp(0, 255) as u8;
            let b = (((smooth_r + smooth_g) / 2) + edge).clamp(0, 255) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

#[test]
fn frozen_golden_wire_vectors_lossless_and_lossy_1t_equals_4t() {
    let (w, h) = (192u32, 160u32);
    let px = golden_fixture_rgb(w as usize, h as usize);

    let ll_1t = encode(
        &px,
        w,
        h,
        3,
        &EncodeOptions { quality: 100, tile_size: 128, dwt_levels: 3, chroma_420: false, checksum: true, threads: 1 },
    )
    .unwrap();
    let ll_4t = encode(
        &px,
        w,
        h,
        3,
        &EncodeOptions { quality: 100, tile_size: 128, dwt_levels: 3, chroma_420: false, checksum: true, threads: 4 },
    )
    .unwrap();
    assert_eq!(ll_1t, ll_4t, "lossless 1T vs 4T determinism violation!");

    let ly_1t = encode(
        &px,
        w,
        h,
        3,
        &EncodeOptions { quality: 80, tile_size: 128, dwt_levels: 3, chroma_420: true, checksum: true, threads: 1 },
    )
    .unwrap();
    let ly_4t = encode(
        &px,
        w,
        h,
        3,
        &EncodeOptions { quality: 80, tile_size: 128, dwt_levels: 3, chroma_420: true, checksum: true, threads: 4 },
    )
    .unwrap();
    assert_eq!(ly_1t, ly_4t, "lossy 1T vs 4T determinism violation!");

    let dec_ll = decode(&ll_1t).unwrap();
    assert_eq!(dec_ll.pixels, px, "lossless golden decode mismatch!");

    let dec_ly = decode(&ly_1t).unwrap();
    let psnr = psnr_db(&px, &dec_ly.pixels);
    assert!(psnr > 32.5, "lossy golden PSNR dropped unexpectedly: {psnr:.2} dB");

    // Frozen VIMG1 v0.1 golden wire-format fingerprints (cross-arch deterministic!)
    assert_eq!(ll_1t.len(), 40_438, "lossless wire length drift");
    assert_eq!(crc32c(&ll_1t), 0xb0ff_bd94, "lossless wire CRC32C drift");
    assert_eq!(fnv1a64(&ll_1t), 0x6fe9_3035_a571_6a68, "lossless wire FNV-1a drift");

    assert_eq!(ly_1t.len(), 2_375, "lossy wire length drift");
    assert_eq!(crc32c(&ly_1t), 0xf719_51cb, "lossy wire CRC32C drift");
    assert_eq!(fnv1a64(&ly_1t), 0x3e4d_b89f_4cf5_4a20, "lossy wire FNV-1a drift");
    assert_eq!(fnv1a64(&dec_ly.pixels), 0x986e_0aa2_cb1d_0e08, "lossy decoded pixels drift");
}
