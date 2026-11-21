#![no_main]
use libfuzzer_sys::fuzz_target;
use vimg::ffi::{
    vimg_decode_query_v2_alloc, vimg_extract_cicp, vimg_extract_exif_alloc, vimg_extract_icc_alloc,
    vimg_extract_xmp_alloc, vimg_free, VimgCicpC,
};
use vimg::{crop_vimg_zero_decode, decode, decode_query, SliceRangeSource, TransformQuery, VimgHeader, VimgIndex};

fuzz_target!(|data: &[u8]| {
    let _ = VimgHeader::parse(data);
    let _ = VimgIndex::parse_prefix(data);
    let _ = decode(data);
    let _ = decode_query(data, &TransformQuery { recover_corrupt: true, ..TransformQuery::default() });
    let _ = crop_vimg_zero_decode(&SliceRangeSource(data), 0, 0, 1, 1, false);
    unsafe {
        let dec = vimg_decode_query_v2_alloc(data.as_ptr(), data.len(), core::ptr::null());
        vimg_free(dec);
        let icc = vimg_extract_icc_alloc(data.as_ptr(), data.len());
        vimg_free(icc);
        let mut cicp = VimgCicpC::default();
        let _ = vimg_extract_cicp(data.as_ptr(), data.len(), &mut cicp);
        let exif = vimg_extract_exif_alloc(data.as_ptr(), data.len());
        vimg_free(exif);
        let xmp = vimg_extract_xmp_alloc(data.as_ptr(), data.len());
        vimg_free(xmp);
    }
});
