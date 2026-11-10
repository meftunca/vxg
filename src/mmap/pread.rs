//! Automatic `mincore`-guided cold/partial-cache `pread` engine (`1 MiB` per-chunk hybrid reads).
//!
//! When a query executes on a memory-mapped archive (`MappedFile` or external `mmap` slice),
//! [`ChunkResidency::probe`] executes a single `mincore(2)` syscall over the entire address range
//! and classifies each `1 MiB` chunk independently:
//! - **Resident `1 MiB` chunks:** read zero-copy directly from `mmap`.
//! - **Non-resident `1 MiB` chunks:** read via parallel `1 MiB` coalesced `pread(2)` calls (or
//!   selective column-stream `pread`s on `VERSION_V6`), eliminating random `16 KiB` VM page-fault
//!   serialization and avoiding redundant copies of already-cached chunks.

use std::cell::RefCell;
use std::io;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

use super::sys::{ChunkResidency, CHUNK_BYTES};

static ACTIVE_MAPS: Mutex<Vec<(usize, usize, i32)>> = Mutex::new(Vec::new());
static PREAD_CALLS: AtomicUsize = AtomicUsize::new(0);
static PREAD_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static PENDING_RESIDENCY: RefCell<Option<(usize, usize, ChunkResidency)>> =
        const { RefCell::new(None) };
    static LAST_RESOLVED_FD: RefCell<Option<(usize, usize, i32, u64)>> =
        const { RefCell::new(None) };
}

/// Return the number of `pread(2)` syscalls executed since the last reset.
#[inline]
pub fn pread_call_count() -> usize {
    PREAD_CALLS.load(Ordering::Relaxed)
}

/// Return the total bytes read via `pread(2)` since the last reset.
#[inline]
pub fn pread_bytes_count() -> usize {
    PREAD_BYTES.load(Ordering::Relaxed)
}

/// Reset the `pread(2)` call and byte counters to `0`.
#[inline]
pub fn reset_pread_counters() {
    PREAD_CALLS.store(0, Ordering::Relaxed);
    PREAD_BYTES.store(0, Ordering::Relaxed);
}

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
    PREAD_CALLS.fetch_add(1, Ordering::Relaxed);
    PREAD_BYTES.fetch_add(dst.len(), Ordering::Relaxed);
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
    if let Some((c_ptr, c_len, fd, off)) = LAST_RESOLVED_FD.with(|c| *c.borrow()) {
        if c_ptr == ptr && c_len == slice.len() {
            #[cfg(unix)]
            {
                use std::os::unix::io::FromRawFd;
                let f = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
                if f.metadata()
                    .is_ok_and(|m| m.is_file() && m.len() >= off + (slice.len() as u64))
                {
                    return Some((fd, off));
                }
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
            let fd = candidates[0].0;
            LAST_RESOLVED_FD.with(|c| *c.borrow_mut() = Some((ptr, slice.len(), fd, 0)));
            return Some((fd, 0));
        }
        if candidates.len() > 1 {
            let actual_hdr = &slice[..32];
            for (fd, hdr) in candidates {
                if hdr == actual_hdr {
                    LAST_RESOLVED_FD.with(|c| *c.borrow_mut() = Some((ptr, slice.len(), fd, 0)));
                    return Some((fd, 0));
                }
            }
        }
    }
    None
}

#[inline]
pub(crate) fn probe_residency_if_mapped(slice: &[u8]) -> ChunkResidency {
    if slice.len() < CHUNK_BYTES {
        return ChunkResidency::all_resident_for_len(slice.len());
    }
    ChunkResidency::probe(slice)
}

#[inline]
pub(crate) fn is_first_page_resident(slice: &[u8]) -> bool {
    if slice.len() < CHUNK_BYTES {
        return true;
    }
    ChunkResidency::probe(&slice[..4096]).all_resident()
}

#[inline]
pub(crate) fn take_or_probe_residency(slice: &[u8]) -> ChunkResidency {
    let ptr = slice.as_ptr() as usize;
    let len = slice.len();
    if let Some((c_ptr, c_len, res)) = PENDING_RESIDENCY.with(|c| c.borrow_mut().take()) {
        if c_ptr == ptr && c_len == len {
            return res;
        }
    }
    probe_residency_if_mapped(slice)
}

/// Hybrid zero-copy `mmap` + coalesced `1 MiB` `pread` buffer for partially or fully cold files.
pub(crate) struct HybridArena<'a> {
    pub(crate) src: &'a [u8],
    pub(crate) res: ChunkResidency,
    pub(crate) chunk_arena_off: Vec<usize>,
    pub(crate) arena: Vec<u8>,
}

impl<'a> HybridArena<'a> {
    /// Load only the non-resident `1 MiB` chunks of `src` via parallel coalesced `pread(2)`.
    pub(crate) fn load(
        src: &'a [u8],
        fd: i32,
        base_offset: u64,
        res: ChunkResidency,
        threads: usize,
    ) -> io::Result<Self> {
        let num_chunks = res.chunks.len();
        let mut chunk_arena_off = vec![0usize; num_chunks];
        let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
        let mut total_arena = 0usize;
        for (c, &is_res) in res.chunks.iter().enumerate() {
            if !is_res {
                let f_off = c * CHUNK_BYTES;
                let f_len = (src.len() - f_off).min(CHUNK_BYTES);
                chunk_arena_off[c] = total_arena;
                tasks.push((f_off, total_arena, f_len));
                total_arena += f_len;
            }
        }
        let mut arena = vec![0u8; total_arena];
        if !tasks.is_empty() {
            let workers = threads.clamp(1, 16).min(tasks.len()).max(1);
            if workers <= 1 {
                for &(f_off, a_off, len) in &tasks {
                    pread_at_exact(fd, base_offset + f_off as u64, &mut arena[a_off..a_off + len])?;
                }
            } else {
                let next = AtomicUsize::new(0);
                let arena_ptr = arena.as_mut_ptr() as usize;
                crate::pool::parallel_try_run(workers, |_| {
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= tasks.len() {
                            return Ok(());
                        }
                        let (f_off, a_off, len) = tasks[idx];
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut((arena_ptr + a_off) as *mut u8, len)
                        };
                        pread_at_exact(fd, base_offset + f_off as u64, dst)
                            .map_err(|e| crate::Error::Backend(e.to_string()))?;
                    }
                })
                .map_err(|e| io::Error::other(e.to_string()))?;
            }
        }
        Ok(Self {
            src,
            res,
            chunk_arena_off,
            arena,
        })
    }

    /// Resolve `[start..end)` to either `(true, start, end)` in `src` (zero-copy `mmap`) or
    /// `(false, a_start, a_end)` in `self.arena` (`pread` arena, stitching only if straddling).
    pub(crate) fn resolve_range(&mut self, start: usize, end: usize) -> (bool, usize, usize) {
        if start >= end {
            return (true, 0, 0);
        }
        if self.res.is_range_resident(start, end) {
            return (true, start, end);
        }
        let c0 = start / CHUNK_BYTES;
        let c1 = (end - 1) / CHUNK_BYTES;
        if self.res.chunks[c0..=c1].iter().all(|&r| !r) {
            let a_start = self.chunk_arena_off[c0] + (start - c0 * CHUNK_BYTES);
            return (false, a_start, a_start + (end - start));
        }
        let a_start = self.arena.len();
        self.arena.reserve(end - start);
        for c in c0..=c1 {
            let c_start = c * CHUNK_BYTES;
            let seg_s = start.max(c_start);
            let seg_e = end.min(((c + 1) * CHUNK_BYTES).min(self.src.len()));
            if self.res.is_chunk_resident(c) {
                self.arena.extend_from_slice(&self.src[seg_s..seg_e]);
            } else {
                let a_s = self.chunk_arena_off[c] + (seg_s - c_start);
                let a_e = a_s + (seg_e - seg_s);
                self.arena.extend_from_within(a_s..a_e);
            }
        }
        let a_end = self.arena.len();
        (false, a_start, a_end)
    }
}

pub(crate) fn peek_cold_prefix(slice: &[u8], max_bytes: usize) -> Option<Vec<u8>> {
    let ptr = slice.as_ptr() as usize;
    let len = slice.len();
    let c0_resident = PENDING_RESIDENCY.with(|c| {
        let mut borrow = c.borrow_mut();
        if let Some((c_ptr, c_len, ref res)) = *borrow {
            if c_ptr == ptr && c_len == len {
                return res.is_chunk_resident(0);
            }
        }
        let res = probe_residency_if_mapped(slice);
        let r0 = res.is_chunk_resident(0);
        *borrow = Some((ptr, len, res));
        r0
    });
    if c0_resident {
        return None;
    }
    let (fd, file_off) = resolve_backing_fd(slice)?;
    let take = slice.len().min(max_bytes);
    let mut buf = vec![0u8; take];
    pread_at_exact(fd, file_off, &mut buf).ok()?;
    Some(buf)
}
