//! Zero-overhead, panic-safe C ABI (`extern "C"`) for static (`libvxg.a` / `vxg.lib`)
//! and dynamic (`libvxg.dylib` / `libvxg.so` / `vxg.dll`) foreign-language integration.
//!
//! All functions catch unwinding panics at the ABI boundary and return negative status
//! codes on failure (`VXG_ERR_*`). Preallocated buffer functions (`vxg_decompress_into`,
//! `vxg_extract_block_into`, `vxg_compress_into`) operate directly on caller memory.

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    compress, compress_into, decompress_into, extract_block_into, max_compressed_size,
    DecodeOptions, Error, Options, Simd,
};

pub const VXG_OK: isize = 0;
pub const VXG_ERR_NULL_PTR: isize = -1;
pub const VXG_ERR_INVALID_OPTION: isize = -2;
pub const VXG_ERR_BUFFER_TOO_SMALL: isize = -3;
pub const VXG_ERR_CORRUPT_DATA: isize = -4;
pub const VXG_ERR_CHECKSUM_MISMATCH: isize = -5;
pub const VXG_ERR_UNSUPPORTED: isize = -6;
pub const VXG_ERR_INTERNAL: isize = -7;

const VERSION_CSTR: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

fn map_error(err: Error) -> isize {
    match err {
        Error::InvalidOption(_) => VXG_ERR_INVALID_OPTION,
        Error::SizeOverflow | Error::OutputLimit { .. } | Error::OutputSize { .. } => {
            VXG_ERR_BUFFER_TOO_SMALL
        }
        Error::Truncated | Error::InvalidFrame(_) => VXG_ERR_CORRUPT_DATA,
        Error::Checksum { .. } => VXG_ERR_CHECKSUM_MISMATCH,
        Error::Unsupported(_) => VXG_ERR_UNSUPPORTED,
        Error::Backend(_) | Error::WorkerPanic => VXG_ERR_INTERNAL,
    }
}

fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 256)
    } else {
        threads
    }
}

fn resolve_block_size(block_size: usize) -> usize {
    if block_size == 0 {
        65536
    } else {
        block_size
    }
}

unsafe fn slice_from_raw<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        Some(&[])
    } else if ptr.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
}

unsafe fn slice_from_raw_mut<'a>(ptr: *mut u8, len: usize) -> Option<&'a mut [u8]> {
    if len == 0 {
        Some(&mut [])
    } else if ptr.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }
}

/// Returns the null-terminated library version string (e.g. `"0.4.0"`).
#[no_mangle]
pub extern "C" fn vxg_version_string() -> *const c_char {
    VERSION_CSTR.as_ptr().cast::<c_char>()
}

/// Returns a static, null-terminated human-readable description for a negative `VXG_ERR_*` code.
#[no_mangle]
pub extern "C" fn vxg_error_message(code: isize) -> *const c_char {
    let msg: &[u8] = match code {
        x if x >= 0 => b"ok\0",
        VXG_ERR_NULL_PTR => b"null pointer argument\0",
        VXG_ERR_INVALID_OPTION => b"invalid option parameter\0",
        VXG_ERR_BUFFER_TOO_SMALL => b"destination buffer too small or size overflow\0",
        VXG_ERR_CORRUPT_DATA => b"corrupt or truncated VXG frame\0",
        VXG_ERR_CHECKSUM_MISMATCH => b"CRC32C checksum mismatch\0",
        VXG_ERR_UNSUPPORTED => b"unsupported hardware or feature\0",
        _ => b"internal codec error\0",
    };
    msg.as_ptr().cast::<c_char>()
}

/// Returns the maximum compressed byte size for an input of `raw_len` bytes,
/// or a negative `VXG_ERR_*` code on overflow / invalid `block_size`.
/// Pass `block_size = 0` to use the default `65536` (64 KiB).
#[no_mangle]
pub extern "C" fn vxg_compress_bound(raw_len: usize, block_size: usize) -> isize {
    let bsize = resolve_block_size(block_size);
    match max_compressed_size(raw_len, bsize) {
        Ok(bound) => isize::try_from(bound).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
        Err(e) => map_error(e),
    }
}

/// Inspects a VXG frame header in $O(1)$ time and returns the exact uncompressed byte size,
/// or a negative `VXG_ERR_*` code if the header is truncated or invalid.
#[no_mangle]
pub unsafe extern "C" fn vxg_frame_raw_size(src: *const u8, src_len: usize) -> isize {
    let Some(bytes) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    if bytes.len() < 32 {
        return VXG_ERR_CORRUPT_DATA;
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if &bytes[..4] != b"VXG1" || !(1..=4).contains(&version) {
        return VXG_ERR_CORRUPT_DATA;
    }
    let raw = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    usize::try_from(raw)
        .ok()
        .and_then(|v| isize::try_from(v).ok())
        .unwrap_or(VXG_ERR_BUFFER_TOO_SMALL)
}

/// Inspects a VXG frame header in $O(1)$ time and returns the number of independent blocks,
/// or a negative `VXG_ERR_*` code if the header is truncated or invalid.
#[no_mangle]
pub unsafe extern "C" fn vxg_frame_block_count(src: *const u8, src_len: usize) -> isize {
    let Some(bytes) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    if bytes.len() < 32 {
        return VXG_ERR_CORRUPT_DATA;
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if &bytes[..4] != b"VXG1" || !(1..=4).contains(&version) {
        return VXG_ERR_CORRUPT_DATA;
    }
    let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    isize::try_from(count).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL)
}

/// Returns the uncompressed byte size of `block_index` in $O(1)$ time from the frame header,
/// or a negative `VXG_ERR_*` code if out of range or corrupt.
#[no_mangle]
pub unsafe extern "C" fn vxg_frame_block_raw_size(
    src: *const u8,
    src_len: usize,
    block_index: usize,
) -> isize {
    let Some(bytes) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    if bytes.len() < 32 {
        return VXG_ERR_CORRUPT_DATA;
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if &bytes[..4] != b"VXG1" || !(1..=4).contains(&version) {
        return VXG_ERR_CORRUPT_DATA;
    }
    let raw_size = match usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap())) {
        Ok(v) => v,
        Err(_) => return VXG_ERR_BUFFER_TOO_SMALL,
    };
    let block_size = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    if block_index >= count || block_size == 0 {
        return VXG_ERR_INVALID_OPTION;
    }
    let start = block_index.saturating_mul(block_size);
    if start >= raw_size {
        return VXG_ERR_CORRUPT_DATA;
    }
    let rn = block_size.min(raw_size - start);
    isize::try_from(rn).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL)
}

/// Compresses `src[..src_len]` into caller-supplied `dst[..dst_cap]`.
/// - `threads`: `0` = auto-detect all CPU cores, `1..=256` = explicit thread count.
/// - `block_size`: `0` = default `65536` (64 KiB), or `256..=65536`.
/// - `checksum`: non-zero enables per-block CRC32C checksums.
/// Returns the number of compressed bytes written (`>= 0`), or a negative `VXG_ERR_*` code.
#[no_mangle]
pub unsafe extern "C" fn vxg_compress_into(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_cap: usize,
    threads: usize,
    block_size: usize,
    checksum: i32,
) -> isize {
    let Some(input) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let Some(output) = (unsafe { slice_from_raw_mut(dst, dst_cap) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let opt = Options {
            block_size: resolve_block_size(block_size),
            threads: resolve_threads(threads),
            checksum: checksum != 0,
            acceleration: 1,
            simd: Simd::Auto,
            legacy_v2: false,
        };
        match compress_into(input, output, &opt) {
            Ok(written) => isize::try_from(written).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Zero-allocation decompression of `src[..src_len]` directly into caller-supplied `dst[..dst_cap]`.
/// - `threads`: `0` = auto-detect all CPU cores, `1..=256` = explicit thread count.
/// - `verify_checksum`: non-zero verifies per-block CRC32C checksums when present.
/// Returns the exact number of uncompressed bytes written (`>= 0`), or a negative `VXG_ERR_*` code.
#[no_mangle]
pub unsafe extern "C" fn vxg_decompress_into(
    src: *const u8,
    src_len: usize,
    dst: *mut u8,
    dst_cap: usize,
    threads: usize,
    verify_checksum: i32,
) -> isize {
    let Some(input) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let Some(output) = (unsafe { slice_from_raw_mut(dst, dst_cap) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        if input.len() < 32 {
            return VXG_ERR_CORRUPT_DATA;
        }
        let raw_len = match usize::try_from(u64::from_le_bytes(input[8..16].try_into().unwrap())) {
            Ok(v) => v,
            Err(_) => return VXG_ERR_BUFFER_TOO_SMALL,
        };
        if output.len() < raw_len {
            return VXG_ERR_BUFFER_TOO_SMALL;
        }
        let opt = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: dst_cap,
            verify_checksum: verify_checksum != 0,
        };
        match decompress_into(input, &mut output[..raw_len], &opt) {
            Ok(()) => isize::try_from(raw_len).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Zero-allocation $O(1)$ random-access decompression of a single block `block_index`
/// directly into caller-supplied `dst[..dst_cap]`.
/// Returns the number of uncompressed bytes written for that block (`>= 0`), or `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_extract_block_into(
    src: *const u8,
    src_len: usize,
    block_index: usize,
    dst: *mut u8,
    dst_cap: usize,
    verify_checksum: i32,
) -> isize {
    let Some(input) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let Some(output) = (unsafe { slice_from_raw_mut(dst, dst_cap) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let opt = DecodeOptions {
            threads: 1,
            simd: Simd::Auto,
            max_output: dst_cap,
            verify_checksum: verify_checksum != 0,
        };
        match extract_block_into(input, block_index, output, &opt) {
            Ok(written) => isize::try_from(written).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Compresses `src[..src_len]` into a newly allocated buffer owned by VXG.
/// On success, writes the buffer length to `*out_len` and returns a non-null pointer
/// that MUST be freed with `vxg_free(ptr, *out_len)`. On failure, returns `NULL`.
#[no_mangle]
pub unsafe extern "C" fn vxg_compress_alloc(
    src: *const u8,
    src_len: usize,
    threads: usize,
    block_size: usize,
    checksum: i32,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    let Some(input) = (unsafe { slice_from_raw(src, src_len) }) else {
        return std::ptr::null_mut();
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> *mut u8 {
        let opt = Options {
            block_size: resolve_block_size(block_size),
            threads: resolve_threads(threads),
            checksum: checksum != 0,
            acceleration: 1,
            simd: Simd::Auto,
            legacy_v2: false,
        };
        let Ok(encoded) = compress(input, &opt) else {
            return std::ptr::null_mut();
        };
        let mut boxed = encoded.into_boxed_slice();
        let len = boxed.len();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe {
            *out_len = len;
        }
        ptr
    }));
    res.unwrap_or(std::ptr::null_mut())
}

/// Decompresses `src[..src_len]` into a newly allocated buffer owned by VXG.
/// On success, writes the uncompressed byte length to `*out_len` and returns a non-null
/// pointer that MUST be freed with `vxg_free(ptr, *out_len)`. On failure, returns `NULL`.
#[no_mangle]
pub unsafe extern "C" fn vxg_decompress_alloc(
    src: *const u8,
    src_len: usize,
    threads: usize,
    verify_checksum: i32,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    let Some(input) = (unsafe { slice_from_raw(src, src_len) }) else {
        return std::ptr::null_mut();
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> *mut u8 {
        let opt = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: 1 << 30,
            verify_checksum: verify_checksum != 0,
        };
        let Ok(decoded) = crate::decompress(input, &opt) else {
            return std::ptr::null_mut();
        };
        let mut boxed = decoded.into_boxed_slice();
        let len = boxed.len();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe {
            *out_len = len;
        }
        ptr
    }));
    res.unwrap_or(std::ptr::null_mut())
}

/// Allocates a zero-initialized byte buffer of `len` bytes inside VXG's linear memory
/// (particularly useful for WebAssembly hosts passing buffers across `WebAssembly.Memory`).
/// Must be freed with `vxg_free(ptr, len)`.
#[no_mangle]
pub extern "C" fn vxg_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let res = catch_unwind(|| -> *mut u8 {
        let mut buf = Vec::<u8>::new();
        if buf.try_reserve_exact(len).is_err() {
            return std::ptr::null_mut();
        }
        buf.resize(len, 0);
        let mut boxed = buf.into_boxed_slice();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        ptr
    });
    res.unwrap_or(std::ptr::null_mut())
}

/// Frees a buffer previously returned by `vxg_alloc`, `vxg_compress_alloc`, or `vxg_decompress_alloc`.
#[no_mangle]
pub unsafe extern "C" fn vxg_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, len);
            drop(Box::from_raw(slice as *mut [u8]));
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_zero_copy_and_alloc_roundtrip_plus_random_access() {
        let mut raw = Vec::new();
        for i in 0..512u32 {
            raw.extend_from_slice(
                format!(
                    "{{\"ts\":\"2026-01-17T12:00:{:02}Z\",\"seq\":{i},\"status\":\"ok\"}}\n",
                    i % 60
                )
                .as_bytes(),
            );
        }

        let bound = vxg_compress_bound(raw.len(), 0);
        assert!(bound > 0);
        let mut compressed = vec![0u8; bound as usize];

        let c_len = unsafe {
            vxg_compress_into(
                raw.as_ptr(),
                raw.len(),
                compressed.as_mut_ptr(),
                compressed.len(),
                0,
                0,
                1,
            )
        };
        assert!(c_len > 0);
        compressed.truncate(c_len as usize);

        let reported_raw = unsafe { vxg_frame_raw_size(compressed.as_ptr(), compressed.len()) };
        assert_eq!(reported_raw as usize, raw.len());

        let blocks = unsafe { vxg_frame_block_count(compressed.as_ptr(), compressed.len()) };
        assert_eq!(blocks, 1);

        let b0_size =
            unsafe { vxg_frame_block_raw_size(compressed.as_ptr(), compressed.len(), 0) };
        assert_eq!(b0_size as usize, raw.len());

        let mut restored = vec![0u8; reported_raw as usize];
        let d_len = unsafe {
            vxg_decompress_into(
                compressed.as_ptr(),
                compressed.len(),
                restored.as_mut_ptr(),
                restored.len(),
                0,
                1,
            )
        };
        assert_eq!(d_len as usize, raw.len());
        assert_eq!(restored, raw);

        let mut block0 = vec![0u8; b0_size as usize];
        let b_len = unsafe {
            vxg_extract_block_into(
                compressed.as_ptr(),
                compressed.len(),
                0,
                block0.as_mut_ptr(),
                block0.len(),
                1,
            )
        };
        assert_eq!(b_len as usize, raw.len());
        assert_eq!(block0, raw);

        // Alloc API roundtrip:
        let mut alloc_c_len = 0usize;
        let c_ptr = unsafe {
            vxg_compress_alloc(raw.as_ptr(), raw.len(), 1, 65536, 1, &mut alloc_c_len)
        };
        assert!(!c_ptr.is_null());
        let mut alloc_d_len = 0usize;
        let d_ptr = unsafe { vxg_decompress_alloc(c_ptr, alloc_c_len, 1, 1, &mut alloc_d_len) };
        assert!(!d_ptr.is_null());
        let d_slice = unsafe { std::slice::from_raw_parts(d_ptr, alloc_d_len) };
        assert_eq!(d_slice, raw.as_slice());
        unsafe {
            vxg_free(c_ptr, alloc_c_len);
            vxg_free(d_ptr, alloc_d_len);
        }

        // Multi-block + multi-thread Rust compress_into / decompress_into / extract_block_into:
        let multi_raw = raw.repeat(12);
        for &threads in &[1usize, 4] {
            let opt = Options {
                block_size: 8192,
                threads,
                checksum: true,
                ..Options::default()
            };
            let mut enc_buf = vec![0u8; max_compressed_size(multi_raw.len(), 8192).unwrap()];
            let written = compress_into(&multi_raw, &mut enc_buf, &opt).unwrap();
            let expected_frame = compress(&multi_raw, &opt).unwrap();
            assert_eq!(&enc_buf[..written], expected_frame.as_slice());

            let dec_opt = DecodeOptions {
                threads,
                max_output: multi_raw.len(),
                verify_checksum: true,
                ..DecodeOptions::default()
            };
            let mut dec_buf = vec![0u8; multi_raw.len()];
            decompress_into(&enc_buf[..written], &mut dec_buf, &dec_opt).unwrap();
            assert_eq!(dec_buf, multi_raw);

            let mut blk_buf = vec![0u8; 8192];
            for (b_idx, chunk) in multi_raw.chunks(8192).enumerate() {
                let b_len =
                    extract_block_into(&enc_buf[..written], b_idx, &mut blk_buf, &dec_opt).unwrap();
                assert_eq!(&blk_buf[..b_len], chunk);
            }
        }
    }
}
