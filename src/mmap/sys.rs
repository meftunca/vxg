#[cfg(all(unix, target_pointer_width = "64"))]
mod imp {
    use core::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::ptr::{self, NonNull};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const PROT_READ: i32 = 0x01;
    const PROT_WRITE: i32 = 0x02;
    const MAP_SHARED: i32 = 0x01;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const MS_SYNC: i32 = 0x0010;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const MS_SYNC: i32 = 0x0004;

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    const MADV_RANDOM: i32 = 1;
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    const MADV_SEQUENTIAL: i32 = 2;
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
    const MADV_WILLNEED: i32 = 3;

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android"))]
        fn msync(addr: *mut c_void, len: usize, flags: i32) -> i32;
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
        fn mincore(addr: *const c_void, len: usize, vec: *mut u8) -> i32;
        fn getpagesize() -> i32;
    }

    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
    static MINCORE_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    fn os_page_size() -> usize {
        let mut ps = PAGE_SIZE.load(Ordering::Relaxed);
        if ps == 0 {
            let raw_ps = unsafe { getpagesize() };
            ps = if raw_ps > 0 { raw_ps as usize } else { 16384 };
            PAGE_SIZE.store(ps, Ordering::Relaxed);
        }
        ps
    }

    #[inline]
    pub fn mincore_call_count() -> usize {
        MINCORE_CALLS.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn reset_mincore_call_count() {
        MINCORE_CALLS.store(0, Ordering::Relaxed);
    }

    pub(crate) fn probe_chunk_residency(slice: &[u8], chunk_bytes: usize) -> (Vec<bool>, usize) {
        if slice.is_empty() {
            return (Vec::new(), 0);
        }
        let num_chunks = slice.len().div_ceil(chunk_bytes);
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
        {
            let ps = os_page_size();
            let slice_start = slice.as_ptr() as usize;
            let slice_end = slice_start + slice.len();
            let base = slice_start & !(ps - 1);
            let end = (slice_end + ps - 1) & !(ps - 1);
            let total_pages = (end - base) / ps;
            if total_pages == 0 {
                return (vec![true; num_chunks], num_chunks);
            }
            let mut vec = vec![0u8; total_pages];
            MINCORE_CALLS.fetch_add(1, Ordering::Relaxed);
            let rc = unsafe { mincore(base as *const c_void, total_pages * ps, vec.as_mut_ptr()) };
            if rc != 0 {
                return (vec![true; num_chunks], num_chunks);
            }
            let mut chunks = Vec::with_capacity(num_chunks);
            let mut resident_chunks = 0usize;
            for c in 0..num_chunks {
                let c_start = slice_start + c * chunk_bytes;
                let c_end = (slice_start + (c + 1) * chunk_bytes).min(slice_end);
                let p_start = (c_start - base) / ps;
                let p_end = (c_end - 1 - base) / ps;
                let is_res = vec[p_start..=p_end].iter().all(|&v| (v & 1) != 0);
                if is_res {
                    resident_chunks += 1;
                }
                chunks.push(is_res);
            }
            (chunks, resident_chunks)
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd")))]
        {
            (vec![true; num_chunks], num_chunks)
        }
    }

    pub fn legacy_four_page_probe(slice: &[u8]) -> bool {
        if slice.len() < (1 << 20) {
            return true;
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd"))]
        {
            let ps = os_page_size();
            let base = slice.as_ptr() as usize;
            let len = slice.len();
            let probes = [len / 8, len / 4, len / 2, (len * 3) / 4];
            let mut resident_count = 0usize;
            for off in probes {
                let page_addr = (base + off) & !(ps - 1);
                let mut vec_byte = 0u8;
                MINCORE_CALLS.fetch_add(1, Ordering::Relaxed);
                let rc = unsafe { mincore(page_addr as *const c_void, ps, &mut vec_byte) };
                if rc != 0 || (vec_byte & 1) != 0 {
                    resident_count += 1;
                }
            }
            resident_count >= 2
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android", target_os = "freebsd")))]
        {
            true
        }
    }

    pub(crate) struct RawMap {
        pub(crate) ptr: NonNull<u8>,
        pub(crate) len: usize,
        pub(crate) fd: i32,
    }

    impl RawMap {
        pub(crate) unsafe fn map_ro(file: &File, len: usize) -> io::Result<Self> {
            if len == 0 {
                return Ok(Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                    fd: -1,
                });
            }
            let raw = unsafe {
                mmap(
                    ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if raw == MAP_FAILED || raw.is_null() {
                return Err(io::Error::last_os_error());
            }
            let ptr = unsafe { NonNull::new_unchecked(raw as *mut u8) };
            let fd = file
                .try_clone()
                .map(std::os::unix::io::IntoRawFd::into_raw_fd)
                .unwrap_or(-1);
            if fd >= 0 {
                crate::mmap::pread::register_mapping(ptr.as_ptr() as usize, len, fd);
            }
            Ok(Self { ptr, len, fd })
        }

        pub(crate) unsafe fn map_rw(file: &File, len: usize) -> io::Result<Self> {
            if len == 0 {
                return Ok(Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                    fd: -1,
                });
            }
            let raw = unsafe {
                mmap(
                    ptr::null_mut(),
                    len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            if raw == MAP_FAILED || raw.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ptr: unsafe { NonNull::new_unchecked(raw as *mut u8) },
                len,
                fd: -1,
            })
        }

        pub(crate) fn flush(&self) -> io::Result<()> {
            if self.len == 0 {
                return Ok(());
            }
            #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "linux",
                target_os = "android"
            ))]
            {
                let rc = unsafe { msync(self.ptr.as_ptr() as *mut c_void, self.len, MS_SYNC) };
                if rc != 0 {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        }

        pub(crate) fn advise_seq(&self) {
            #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "linux",
                target_os = "android",
                target_os = "freebsd"
            ))]
            if self.len > 0 {
                unsafe {
                    let _ = madvise(self.ptr.as_ptr() as *mut c_void, self.len, MADV_SEQUENTIAL);
                }
            }
        }

        pub(crate) fn advise_rand(&self) {
            #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "linux",
                target_os = "android",
                target_os = "freebsd"
            ))]
            if self.len > 0 {
                unsafe {
                    let _ = madvise(self.ptr.as_ptr() as *mut c_void, self.len, MADV_RANDOM);
                }
            }
        }

        pub(crate) fn advise_willneed(&self, offset: usize, len: usize) {
            #[cfg(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "linux",
                target_os = "android",
                target_os = "freebsd"
            ))]
            if len > 0 && offset < self.len {
                let page = 4096usize;
                let aligned_off = offset & !(page - 1);
                let end = offset.saturating_add(len).min(self.len);
                let span = end.saturating_sub(aligned_off);
                if span > 0 {
                    unsafe {
                        let base = self.ptr.as_ptr().add(aligned_off) as *mut c_void;
                        let _ = madvise(base, span, MADV_WILLNEED);
                    }
                }
            }
        }
    }

    impl Drop for RawMap {
        fn drop(&mut self) {
            if self.fd >= 0 {
                crate::mmap::pread::unregister_mapping(self.ptr.as_ptr() as usize, self.len, self.fd);
            }
            if self.len > 0 {
                unsafe {
                    let _ = munmap(self.ptr.as_ptr() as *mut c_void, self.len);
                }
            }
        }
    }

    pub(crate) const SUPPORTED: bool = true;
}

#[cfg(all(windows, target_pointer_width = "64"))]
mod imp {
    use core::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::ptr::{self, NonNull};

    type Handle = *mut c_void;
    const PAGE_READONLY: u32 = 0x02;
    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_WRITE: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileMappingW(
            hfile: Handle,
            attrs: *const c_void,
            protect: u32,
            max_size_high: u32,
            max_size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            desired_access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes_to_map: usize,
        ) -> *mut c_void;
        fn FlushViewOfFile(base: *const c_void, bytes_to_flush: usize) -> i32;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(object: Handle) -> i32;
    }

    pub(crate) fn probe_chunk_residency(slice: &[u8], chunk_bytes: usize) -> (Vec<bool>, usize) {
        let n = slice.len().div_ceil(chunk_bytes);
        (vec![true; n], n)
    }
    pub fn legacy_four_page_probe(_slice: &[u8]) -> bool {
        true
    }
    pub fn mincore_call_count() -> usize {
        0
    }
    pub fn reset_mincore_call_count() {}

    pub(crate) struct RawMap {
        pub(crate) ptr: NonNull<u8>,
        pub(crate) len: usize,
    }

    impl RawMap {
        pub(crate) unsafe fn map_ro(file: &File, len: usize) -> io::Result<Self> {
            if len == 0 {
                return Ok(Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                });
            }
            let hi = ((len as u64) >> 32) as u32;
            let lo = (len as u64) as u32;
            let mapping = unsafe {
                CreateFileMappingW(
                    file.as_raw_handle() as Handle,
                    ptr::null(),
                    PAGE_READONLY,
                    hi,
                    lo,
                    ptr::null(),
                )
            };
            if mapping.is_null() {
                return Err(io::Error::last_os_error());
            }
            let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
            unsafe {
                let _ = CloseHandle(mapping);
            }
            if view.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ptr: unsafe { NonNull::new_unchecked(view as *mut u8) },
                len,
            })
        }

        pub(crate) unsafe fn map_rw(file: &File, len: usize) -> io::Result<Self> {
            if len == 0 {
                return Ok(Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                });
            }
            let hi = ((len as u64) >> 32) as u32;
            let lo = (len as u64) as u32;
            let mapping = unsafe {
                CreateFileMappingW(
                    file.as_raw_handle() as Handle,
                    ptr::null(),
                    PAGE_READWRITE,
                    hi,
                    lo,
                    ptr::null(),
                )
            };
            if mapping.is_null() {
                return Err(io::Error::last_os_error());
            }
            let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, len) };
            unsafe {
                let _ = CloseHandle(mapping);
            }
            if view.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ptr: unsafe { NonNull::new_unchecked(view as *mut u8) },
                len,
            })
        }

        pub(crate) fn flush(&self) -> io::Result<()> {
            if self.len == 0 {
                return Ok(());
            }
            let ok = unsafe { FlushViewOfFile(self.ptr.as_ptr() as *const c_void, self.len) };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub(crate) fn advise_seq(&self) {}
        pub(crate) fn advise_rand(&self) {}
        pub(crate) fn advise_willneed(&self, _offset: usize, _len: usize) {}
    }

    impl Drop for RawMap {
        fn drop(&mut self) {
            if self.len > 0 {
                unsafe {
                    let _ = UnmapViewOfFile(self.ptr.as_ptr() as *const c_void);
                }
            }
        }
    }

    pub(crate) const SUPPORTED: bool = true;
}

#[cfg(not(any(
    all(unix, target_pointer_width = "64"),
    all(windows, target_pointer_width = "64")
)))]
mod imp {
    use std::fs::File;
    use std::io;
    use std::ptr::NonNull;

    pub(crate) fn probe_chunk_residency(slice: &[u8], chunk_bytes: usize) -> (Vec<bool>, usize) {
        let n = slice.len().div_ceil(chunk_bytes);
        (vec![true; n], n)
    }
    pub fn legacy_four_page_probe(_slice: &[u8]) -> bool {
        true
    }
    pub fn mincore_call_count() -> usize {
        0
    }
    pub fn reset_mincore_call_count() {}

    pub(crate) struct RawMap {
        pub(crate) ptr: NonNull<u8>,
        pub(crate) len: usize,
    }

    impl RawMap {
        pub(crate) unsafe fn map_ro(_file: &File, _len: usize) -> io::Result<Self> {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "mmap unsupported",
            ))
        }

        pub(crate) unsafe fn map_rw(file: &File, len: usize) -> io::Result<Self> {
            unsafe { Self::map_ro(file, len) }
        }

        pub(crate) fn flush(&self) -> io::Result<()> {
            Ok(())
        }
        pub(crate) fn advise_seq(&self) {}
        pub(crate) fn advise_rand(&self) {}
        pub(crate) fn advise_willneed(&self, _offset: usize, _len: usize) {}
    }

    pub(crate) const SUPPORTED: bool = false;
}

pub(crate) use imp::{RawMap, SUPPORTED};
pub use imp::{legacy_four_page_probe, mincore_call_count, reset_mincore_call_count};

/// Granularity (`1 MiB`) for per-chunk hybrid `mmap` / coalesced `pread` decisions.
pub const CHUNK_BYTES: usize = 1 << 20;

/// Per-`1 MiB` chunk residency bitmap produced by a single `mincore(2)` syscall over a slice.
#[derive(Clone, Debug)]
pub struct ChunkResidency {
    /// `chunks[c]` is `true` iff every OS page in chunk `c` (`[c * 1 MiB .. (c + 1) * 1 MiB)`)
    /// is currently resident in the OS page cache.
    pub chunks: Vec<bool>,
    /// Number of resident `1 MiB` chunks.
    pub resident_chunks: usize,
}

impl ChunkResidency {
    /// Probe the entire `slice` with a single `mincore(2)` syscall and classify each `1 MiB` chunk.
    #[inline]
    pub fn probe(slice: &[u8]) -> Self {
        let (chunks, resident_chunks) = imp::probe_chunk_residency(slice, CHUNK_BYTES);
        Self {
            chunks,
            resident_chunks,
        }
    }

    /// Construct a `ChunkResidency` marking all `1 MiB` chunks as resident without calling `mincore`.
    #[inline]
    pub fn all_resident_for_len(len: usize) -> Self {
        let n = len.div_ceil(CHUNK_BYTES);
        Self {
            chunks: vec![true; n],
            resident_chunks: n,
        }
    }

    #[inline]
    pub fn all_resident(&self) -> bool {
        self.resident_chunks == self.chunks.len()
    }

    #[inline]
    pub fn none_resident(&self) -> bool {
        self.resident_chunks == 0
    }

    #[inline]
    pub fn is_chunk_resident(&self, chunk_idx: usize) -> bool {
        self.chunks.get(chunk_idx).copied().unwrap_or(false)
    }

    /// Return `true` if every `1 MiB` chunk overlapped by byte range `[start..end)` is resident.
    #[inline]
    pub fn is_range_resident(&self, start: usize, end: usize) -> bool {
        if start >= end {
            return true;
        }
        let c0 = start / CHUNK_BYTES;
        let c1 = (end - 1) / CHUNK_BYTES;
        if c1 >= self.chunks.len() {
            return false;
        }
        self.chunks[c0..=c1].iter().all(|&r| r)
    }
}
