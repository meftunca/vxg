#![no_main]
use libfuzzer_sys::fuzz_target;
use vimg::exif::{
    parse_jpeg_exif_orientation, parse_jpeg_exif_raw, parse_jpeg_icc_profile, parse_jpeg_xmp_raw,
};

fuzz_target!(|data: &[u8]| {
    let ori = parse_jpeg_exif_orientation(data);
    assert!((1..=8).contains(&ori));
    let _ = parse_jpeg_icc_profile(data);
    let _ = parse_jpeg_exif_raw(data);
    let _ = parse_jpeg_xmp_raw(data);
});
