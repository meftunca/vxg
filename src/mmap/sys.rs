#[cfg(all(unix, target_pointer_width = "64"))]
mod imp {
    use core::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::ptr::{self, NonNull};

    const PROT_READ: i32 = 0x01;
    const PROT_WRITE: i32 = 0x02;
    const MAP_SHARED: i32 = 0x01;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const MS_SYNC: i32 = 0x0010;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const MS_SYNC: i32 = 0x0004;

    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    const MADV_RANDOM: i32 = 1;
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
    const MADV_SEQUENTIAL: i32 = 2;
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd"
    ))]
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
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android"
        ))]
        fn msync(addr: *mut c_void, len: usize, flags: i32) -> i32;
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd"
        ))]
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd"
        ))]
        fn mincore(addr: *const c_void, len: usize, vec: *mut u8) -> i32;
        fn getpagesize() -> i32;
    }

    pub(crate) fn is_slice_resident(slice: &[u8]) -> bool {
        if slice.len() < (1 << 20) {
            return true;
        }
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd"
        ))]
        {
            static PAGE_SIZE: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let mut ps = PAGE_SIZE.load(std::sync::atomic::Ordering::Relaxed);
            if ps == 0 {
                let raw_ps = unsafe { getpagesize() };
                ps = if raw_ps > 0 { raw_ps as usize } else { 16384 };
                PAGE_SIZE.store(ps, std::sync::atomic::Ordering::Relaxed);
            }
            let base = slice.as_ptr() as usize;
            let len = slice.len();
            let probes = [len / 8, len / 4, len / 2, (len * 3) / 4];
            let mut resident_count = 0usize;
            for off in probes {
                let page_addr = (base + off) & !(ps - 1);
                let mut vec_byte = 0u8;
                let rc = unsafe { mincore(page_addr as *const c_void, ps, &mut vec_byte) };
                if rc != 0 || (vec_byte & 1) != 0 {
                    resident_count += 1;
                }
            }
            resident_count >= 2
        }
        #[cfg(not(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd"
        )))]
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

    pub(crate) fn is_slice_resident(_slice: &[u8]) -> bool {
        true
    }

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

    pub(crate) fn is_slice_resident(_slice: &[u8]) -> bool {
        true
    }

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

pub(crate) use imp::{is_slice_resident, RawMap, SUPPORTED};
