//! CRC32C (Castagnoli), not IEEE CRC32 and not an authentication primitive.
const fn make_table() -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut x = i as u32;
        let mut j = 0;
        while j < 8 {
            x = (x >> 1) ^ (0x82f63b78u32 & 0u32.wrapping_sub(x & 1));
            j += 1;
        }
        table[i] = x;
        i += 1;
    }
    table
}
const TABLE: [u32; 256] = make_table();
pub fn software(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc = (crc >> 8) ^ TABLE[((crc as u8) ^ b) as usize];
    }
    !crc
}
pub fn crc32c(data: &[u8]) -> u32 {
    if cfg!(miri) {
        return software(data);
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.2") {
        // SAFETY: SSE4.2 checked; hardware path reads only complete in-bounds words.
        return unsafe { x86(data) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("crc") {
        // SAFETY: CRC extension checked, loads use bounded slices.
        return unsafe { arm(data) };
    }
    software(data)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn x86(data: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let mut crc = !0u32;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        crc = _mm_crc32_u64(crc as u64, u64::from_le_bytes(chunk.try_into().unwrap())) as u32;
    }
    for &b in chunks.remainder() {
        crc = _mm_crc32_u8(crc, b);
    }
    !crc
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn arm(data: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let mut crc = !0u32;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        crc = __crc32cd(crc, u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    for &b in chunks.remainder() {
        crc = __crc32cb(crc, b);
    }
    !crc
}
