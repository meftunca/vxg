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
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    if cfg!(miri) {
        return software(data);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "crc"))]
    {
        return unsafe { arm(data) };
    }
    #[cfg(all(target_arch = "aarch64", not(target_feature = "crc")))]
    if std::arch::is_aarch64_feature_detected!("crc") {
        // SAFETY: CRC extension checked, loads use bounded slices.
        return unsafe { arm(data) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.2") {
        // SAFETY: SSE4.2 checked; hardware path reads only complete in-bounds words.
        return unsafe { x86(data) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "crc")))]
    software(data)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn x86(data: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let mut crc = !0u32;
    let mut chunks32 = data.chunks_exact(32);
    for c in &mut chunks32 {
        let w0 = u64::from_le_bytes(c[0..8].try_into().unwrap());
        let w1 = u64::from_le_bytes(c[8..16].try_into().unwrap());
        let w2 = u64::from_le_bytes(c[16..24].try_into().unwrap());
        let w3 = u64::from_le_bytes(c[24..32].try_into().unwrap());
        crc = _mm_crc32_u64(crc as u64, w0) as u32;
        crc = _mm_crc32_u64(crc as u64, w1) as u32;
        crc = _mm_crc32_u64(crc as u64, w2) as u32;
        crc = _mm_crc32_u64(crc as u64, w3) as u32;
    }
    let mut chunks8 = chunks32.remainder().chunks_exact(8);
    for chunk in &mut chunks8 {
        crc = _mm_crc32_u64(crc as u64, u64::from_le_bytes(chunk.try_into().unwrap())) as u32;
    }
    for &b in chunks8.remainder() {
        crc = _mm_crc32_u8(crc, b);
    }
    !crc
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
#[inline]
unsafe fn arm(data: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let mut crc = !0u32;
    let mut chunks32 = data.chunks_exact(32);
    for c in &mut chunks32 {
        let w0 = u64::from_le_bytes(c[0..8].try_into().unwrap());
        let w1 = u64::from_le_bytes(c[8..16].try_into().unwrap());
        let w2 = u64::from_le_bytes(c[16..24].try_into().unwrap());
        let w3 = u64::from_le_bytes(c[24..32].try_into().unwrap());
        crc = __crc32cd(crc, w0);
        crc = __crc32cd(crc, w1);
        crc = __crc32cd(crc, w2);
        crc = __crc32cd(crc, w3);
    }
    let mut chunks8 = chunks32.remainder().chunks_exact(8);
    for chunk in &mut chunks8 {
        crc = __crc32cd(crc, u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    let rem = chunks8.remainder();
    let mut i = 0;
    if i + 4 <= rem.len() {
        crc = __crc32cw(crc, u32::from_le_bytes(rem[i..i + 4].try_into().unwrap()));
        i += 4;
    }
    if i + 2 <= rem.len() {
        crc = __crc32ch(crc, u16::from_le_bytes(rem[i..i + 2].try_into().unwrap()));
        i += 2;
    }
    if i < rem.len() {
        crc = __crc32cb(crc, rem[i]);
    }
    !crc
}
