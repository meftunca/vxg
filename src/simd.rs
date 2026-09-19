//! Bounds-exact SIMD. No wild-copy, overread, or caller-provided padding contract.
//! `Engine` can only be constructed after runtime feature detection.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Simd { Auto, Scalar, Sse2, Avx2, Avx512, Neon }
impl std::str::FromStr for Simd {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto), "scalar" => Ok(Self::Scalar),
            "sse2" => Ok(Self::Sse2), "avx2" => Ok(Self::Avx2),
            "avx512" => Ok(Self::Avx512), "neon" => Ok(Self::Neon),
            _ => Err(Error::InvalidOption("SIMD must be auto/scalar/sse2/avx2/avx512/neon")),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Engine { kind: Simd }
impl Engine {
    pub fn new(request: Simd) -> Result<Self> {
        let best = Self::best();
        let kind = if request == Simd::Auto { best } else { request };
        if !Self::supported(kind) {
            return Err(Error::Unsupported(format!("SIMD {kind:?} is not available on this CPU")));
        }
        Ok(Self { kind })
    }
    pub fn kind(self) -> Simd { self.kind }
    pub fn supported(kind: Simd) -> bool {
        if cfg!(miri) { return matches!(kind, Simd::Scalar | Simd::Auto); }
        match kind {
            Simd::Auto | Simd::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Simd::Sse2 => std::is_x86_feature_detected!("sse2"),
            #[cfg(target_arch = "x86_64")]
            Simd::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Simd::Avx512 => std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"),
            #[cfg(target_arch = "aarch64")]
            Simd::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            _ => false,
        }
    }
    fn best() -> Simd {
        // AVX-512 is explicit opt-in: its clock/power trade-off is workload-specific.
        for kind in [Simd::Avx2, Simd::Neon, Simd::Sse2] {
            if Self::supported(kind) { return kind; }
        }
        Simd::Scalar
    }
    /// Length of the common prefix. Both slices may be unaligned and very short.
    pub fn lcp(self, a: &[u8], b: &[u8]) -> usize {
        let n = a.len().min(b.len());
        // SAFETY: constructors checked CPU support; functions only load full vectors
        // while offset + vector_width <= n. The scalar tail also stays inside n.
        unsafe {
            match self.kind {
                #[cfg(target_arch = "x86_64")]
                Simd::Sse2 => x86::lcp_sse2(a.as_ptr(), b.as_ptr(), n),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 => x86::lcp_avx2(a.as_ptr(), b.as_ptr(), n),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx512 => x86::lcp_avx512(a.as_ptr(), b.as_ptr(), n),
                #[cfg(target_arch = "aarch64")]
                Simd::Neon => arm::lcp_neon(a.as_ptr(), b.as_ptr(), n),
                _ => scalar_lcp(a, b, n),
            }
        }
    }
    /// Safe non-overlapping copy; borrowing rules keep the two slices disjoint.
    pub fn copy(self, dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        let n = src.len();
        // Tiny copies pay more for the dispatch/vector loop than they save. Keep
        // the platform's tuned short-copy path for literals and small tails.
        if n < 64 { dst.copy_from_slice(src); return; }
        // SAFETY: valid disjoint slices, feature-checked dispatch, exact bounds.
        unsafe {
            match self.kind {
                #[cfg(target_arch = "x86_64")]
                Simd::Sse2 => x86::copy_sse2(dst.as_mut_ptr(), src.as_ptr(), n),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx2 => x86::copy_avx2(dst.as_mut_ptr(), src.as_ptr(), n),
                #[cfg(target_arch = "x86_64")]
                Simd::Avx512 => x86::copy_avx512(dst.as_mut_ptr(), src.as_ptr(), n),
                #[cfg(target_arch = "aarch64")]
                Simd::Neon => arm::copy_neon(dst.as_mut_ptr(), src.as_ptr(), n),
                _ => dst.copy_from_slice(src),
            }
        }
    }
    /// Fill from an immutable seed, preserving the phase for non-power-of-two periods.
    pub fn pattern(self, dst: &mut [u8], seed: &[u8]) {
        assert!(!seed.is_empty() && seed.len() <= 64);
        if seed.len() == 1 { dst.fill(seed[0]); return; }
        const TILE_SIZE: usize = 1024;
        let step = (TILE_SIZE / seed.len()) * seed.len();
        let mut tile = [0u8; TILE_SIZE];
        self.copy(&mut tile[..seed.len()], seed);
        let mut filled = seed.len();
        while filled < step {
            let take = filled.min(step - filled);
            let (prefix, suffix) = tile.split_at_mut(filled);
            self.copy(&mut suffix[..take], &prefix[..take]);
            filled += take;
        }
        // The platform copy is already vectorized and wins over repeatedly
        // entering the hand-SIMD dispatch for these medium-sized chunks.
        for chunk in dst.chunks_mut(step) { chunk.copy_from_slice(&tile[..chunk.len()]); }
    }
    /// LZ semantics without calling memcpy/memmove on an overlapping unproduced region.
    pub(crate) fn repeat(self, out: &mut [u8], pos: usize, len: usize, distance: usize) {
        debug_assert!(distance > 0 && distance <= pos && len <= out.len() - pos);
        // JSON-style matches are commonly short and farther back than their
        // run. They are non-overlapping, so avoid the general doubling setup
        // and SIMD-dispatch branch; the platform copy is already optimal for
        // these sub-64-byte regions.
        if len <= distance && !cfg!(feature="repeat-engine-copy") {
            let (prefix, suffix) = out.split_at_mut(pos);
            suffix[..len].copy_from_slice(&prefix[pos-distance..pos-distance+len]);
            return;
        }
        if distance <= 64 {
            let mut seed = [0u8; 64];
            seed[..distance].copy_from_slice(&out[pos-distance..pos]);
            self.pattern(&mut out[pos..pos+len], &seed[..distance]);
            return;
        }
        let first = distance.min(len);
        {
            let (prefix, suffix) = out.split_at_mut(pos);
            self.copy(&mut suffix[..first], &prefix[pos-distance..pos-distance+first]);
        }
        let mut done = first;
        while done < len {
            let take = done.min(len-done);
            let (prefix, suffix) = out.split_at_mut(pos+done);
            self.copy(&mut suffix[..take], &prefix[pos..pos+take]);
            done += take;
        }
    }
}
fn scalar_lcp(a: &[u8], b: &[u8], n: usize) -> usize {
    let mut i = 0;
    while i+8 <= n {
        let av = u64::from_le_bytes(a[i..i+8].try_into().unwrap());
        let bv = u64::from_le_bytes(b[i..i+8].try_into().unwrap());
        let diff = av ^ bv;
        if diff != 0 { return i + (diff.trailing_zeros() / 8) as usize; }
        i += 8;
    }
    while i < n && a[i] == b[i] { i += 1; }
    i
}
#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    #[target_feature(enable = "sse2")]
    pub unsafe fn lcp_sse2(a: *const u8, b: *const u8, n: usize) -> usize {
        let mut i = 0;
        while i+16 <= n {
            let eq = _mm_cmpeq_epi8(_mm_loadu_si128(a.add(i).cast()), _mm_loadu_si128(b.add(i).cast()));
            let mask = _mm_movemask_epi8(eq) as u32;
            if mask != 0xffff { return i + ((!mask) & 0xffff).trailing_zeros() as usize; }
            i += 16;
        }
        while i < n && *a.add(i) == *b.add(i) { i += 1; }
        i
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn lcp_avx2(a: *const u8, b: *const u8, n: usize) -> usize {
        let mut i = 0;
        while i+32 <= n {
            let eq = _mm256_cmpeq_epi8(_mm256_loadu_si256(a.add(i).cast()), _mm256_loadu_si256(b.add(i).cast()));
            let mask = _mm256_movemask_epi8(eq) as u32;
            if mask != u32::MAX { return i + (!mask).trailing_zeros() as usize; }
            i += 32;
        }
        while i < n && *a.add(i) == *b.add(i) { i += 1; }
        i
    }
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn lcp_avx512(a: *const u8, b: *const u8, n: usize) -> usize {
        let mut i = 0;
        while i+64 <= n {
            let mask = _mm512_cmpeq_epi8_mask(_mm512_loadu_si512(a.add(i).cast()), _mm512_loadu_si512(b.add(i).cast()));
            if mask != u64::MAX { return i + (!mask).trailing_zeros() as usize; }
            i += 64;
        }
        while i < n && *a.add(i) == *b.add(i) { i += 1; }
        i
    }
    #[target_feature(enable = "sse2")]
    pub unsafe fn copy_sse2(d: *mut u8, s: *const u8, n: usize) {
        let mut i=0;
        while i+16<=n { _mm_storeu_si128(d.add(i).cast(), _mm_loadu_si128(s.add(i).cast())); i+=16; }
        std::ptr::copy_nonoverlapping(s.add(i), d.add(i), n-i);
    }
    #[target_feature(enable = "avx2")]
    pub unsafe fn copy_avx2(d: *mut u8, s: *const u8, n: usize) {
        let mut i=0;
        while i+32<=n { _mm256_storeu_si256(d.add(i).cast(), _mm256_loadu_si256(s.add(i).cast())); i+=32; }
        std::ptr::copy_nonoverlapping(s.add(i), d.add(i), n-i);
    }
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn copy_avx512(d: *mut u8, s: *const u8, n: usize) {
        let mut i=0;
        while i+64<=n { _mm512_storeu_si512(d.add(i).cast(), _mm512_loadu_si512(s.add(i).cast())); i+=64; }
        std::ptr::copy_nonoverlapping(s.add(i), d.add(i), n-i);
    }
}
#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;
    #[target_feature(enable = "neon")]
    pub unsafe fn lcp_neon(a: *const u8, b: *const u8, n: usize) -> usize {
        let mut i = 0;
        while i+32<=n {
            let eq0 = vceqq_u8(vld1q_u8(a.add(i)), vld1q_u8(b.add(i)));
            if vminvq_u8(eq0) != 255 {
                for j in 0..16 { if *a.add(i+j) != *b.add(i+j) { return i+j; } }
            }
            let eq1 = vceqq_u8(vld1q_u8(a.add(i+16)), vld1q_u8(b.add(i+16)));
            if vminvq_u8(eq1) != 255 {
                for j in 0..16 { if *a.add(i+16+j) != *b.add(i+16+j) { return i+16+j; } }
            }
            i += 32;
        }
        while i+16<=n {
            let eq = vceqq_u8(vld1q_u8(a.add(i)), vld1q_u8(b.add(i)));
            if vminvq_u8(eq) != 255 {
                for j in 0..16 { if *a.add(i+j) != *b.add(i+j) { return i+j; } }
            }
            i += 16;
        }
        while i<n && *a.add(i)==*b.add(i) { i+=1; }
        i
    }
    #[target_feature(enable = "neon")]
    pub unsafe fn copy_neon(d: *mut u8, s: *const u8, n: usize) {
        let mut i=0;
        while i+16<=n { vst1q_u8(d.add(i), vld1q_u8(s.add(i))); i+=16; }
        std::ptr::copy_nonoverlapping(s.add(i), d.add(i), n-i);
    }
}
