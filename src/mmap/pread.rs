//! Automatic `mincore`-guided cold-cache `pread` engine (`1 MiB` coalesced parallel reads).
//!
//! When a query executes on a memory-mapped archive (`MappedFile` or external `mmap` slice),
//! `is_slice_resident` probes interior pages via `mincore` in `< 500 ns`:
//! - **Warm cache (pages resident in RAM):** returns `Cow::Borrowed(frame_bytes)` with zero allocation.
//! - **Cold cache (pages evicted / uncached on NVMe):** automatically resolves the backing file
//!   descriptor and issues `8T` parallel `1 MiB` coalesced `pread` calls (or selective column-stream
//!   `pread`s on `VERSION_V6`), eliminating random `16 KiB` VM page-fault serialization.

use std::borrow::Cow;
use std::io;
use std::sync::Mutex;

static ACTIVE_MAPS: Mutex<Vec<(usize, usize, i32)>> = Mutex::new(Vec::new());

pub(crate) fn register_mapping(addr: usize, len: usize, fd: i32) {
    if let Ok(mut guard) = ACTIVE_MAPS.lock() {
        guard.push((addr, len, fd));
    }
}

pub(crate) fn unregister_mapping(addr: usize, len: usize, fd: i32) {
    if let Ok(mut guard) = ACTIVE_MAPS.lock() {
        if let Some(pos) = guard
            .iter()
            .position(|&(a, l, f)| a == addr && l == len && f == fd)
        {
            guard.swap_remove(pos);
        }
    }
    #[cfg(unix)]
    if fd >= 0 {
        unsafe {
            let _ = <std::fs::File as std::os::unix::io::FromRawFd>::from_raw_fd(fd);
        }
    }
}

#[cfg(unix)]
pub(crate) fn pread_at_exact(fd: i32, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    use std::os::unix::io::FromRawFd;
    let file = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
    file.read_exact_at(dst, offset)
}

#[cfg(not(unix))]
pub(crate) fn pread_at_exact(_fd: i32, _offset: u64, _dst: &mut [u8]) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "pread unsupported",
    ))
}

pub(crate) fn resolve_backing_fd(slice: &[u8]) -> Option<(i32, u64)> {
    if slice.len() < crate::frame::HEADER_SIZE {
        return None;
    }
    let ptr = slice.as_ptr() as usize;
    let end = ptr.checked_add(slice.len())?;
    if let Ok(guard) = ACTIVE_MAPS.lock() {
        for &(base, map_len, fd) in guard.iter() {
            if fd >= 0 && ptr >= base && end <= base + map_len {
                return Some((fd, (ptr - base) as u64));
            }
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        use std::os::unix::io::FromRawFd;
        let target_len = slice.len() as u64;
        let mut candidates: Vec<(i32, [u8; 32])> = Vec::new();
        for fd in 3..128i32 {
            let f = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
            if let Ok(meta) = f.metadata() {
                if meta.is_file() && meta.len() == target_len {
                    let mut hdr = [0u8; 32];
                    if f.read_exact_at(&mut hdr, 0).is_ok() && &hdr[..4] == crate::frame::MAGIC {
                        candidates.push((fd, hdr));
                    }
                }
            }
        }
        if candidates.len() == 1 {
            return Some((candidates[0].0, 0));
        }
        if candidates.len() > 1 {
            let actual_hdr = &slice[..32];
            for (fd, hdr) in candidates {
                if hdr == actual_hdr {
                    return Some((fd, 0));
                }
            }
        }
    }
    None
}

pub(crate) fn pread_parallel_1m(
    fd: i32,
    base_offset: u64,
    len: usize,
    threads: usize,
) -> io::Result<Vec<u8>> {
    const CHUNK: usize = 1 << 20;
    let mut buf = vec![0u8; len];
    if len == 0 {
        return Ok(buf);
    }
    let num_chunks = len.div_ceil(CHUNK);
    let workers = threads.clamp(1, 16).min(num_chunks).max(1);
    if workers <= 1 {
        for c in 0..num_chunks {
            let off = c * CHUNK;
            let clen = (len - off).min(CHUNK);
            pread_at_exact(fd, base_offset + off as u64, &mut buf[off..off + clen])?;
        }
        return Ok(buf);
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let base_ptr = buf.as_mut_ptr() as usize;
    crate::pool::parallel_try_run(workers, |_| {
        loop {
            let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if c >= num_chunks {
                return Ok(());
            }
            let off = c * CHUNK;
            let clen = (len - off).min(CHUNK);
            let dst =
                unsafe { std::slice::from_raw_parts_mut((base_ptr + off) as *mut u8, clen) };
            pread_at_exact(fd, base_offset + off as u64, dst)
                .map_err(|e| crate::Error::Backend(e.to_string()))?;
        }
    })
    .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(buf)
}

pub(crate) fn peek_cold_prefix(slice: &[u8], max_bytes: usize) -> Option<Vec<u8>> {
    if super::sys::is_slice_resident(slice) {
        return None;
    }
    let (fd, file_off) = resolve_backing_fd(slice)?;
    let take = slice.len().min(max_bytes);
    let mut buf = vec![0u8; take];
    pread_at_exact(fd, file_off, &mut buf).ok()?;
    Some(buf)
}

pub(crate) fn prepare_query_bytes<'a>(
    frame_bytes: &'a [u8],
    where_expr: Option<&crate::ops::PredExpr<'_>>,
    extra_cols: &[crate::ops::ColTarget<'_>],
    allow_partial_cols: bool,
    threads: usize,
) -> Cow<'a, [u8]> {
    let resident = super::sys::is_slice_resident(frame_bytes);
    if !resident {
        if let Some((fd, file_off)) = resolve_backing_fd(frame_bytes) {
            let mut hdr = [0u8; 32];
            if pread_at_exact(fd, file_off, &mut hdr).is_ok()
                && &hdr[..4] == crate::frame::MAGIC
                && u16::from_le_bytes([hdr[4], hdr[5]]) == crate::frame::v6::VERSION_V6
            {
                if let Ok(v5_buf) = crate::frame::v6::pread_v6_for_query(
                    fd,
                    file_off,
                    frame_bytes.len(),
                    where_expr,
                    extra_cols,
                    allow_partial_cols,
                    threads,
                ) {
                    return Cow::Owned(v5_buf);
                }
            }
            if let Ok(buf) = pread_parallel_1m(fd, file_off, frame_bytes.len(), threads) {
                if crate::frame::v6::is_v6_frame(&buf) {
                    if let Ok(v5_buf) = crate::frame::v6::reassemble_v6_for_query_warm(
                        &buf,
                        where_expr,
                        extra_cols,
                        allow_partial_cols,
                        threads,
                    ) {
                        return Cow::Owned(v5_buf);
                    }
                }
                return Cow::Owned(buf);
            }
        }
    }
    if crate::frame::v6::is_v6_frame(frame_bytes) {
        if let Ok(v5_buf) = crate::frame::v6::reassemble_v6_for_query_warm(
            frame_bytes,
            where_expr,
            extra_cols,
            allow_partial_cols,
            threads,
        ) {
            return Cow::Owned(v5_buf);
        }
    }
    Cow::Borrowed(frame_bytes)
}
