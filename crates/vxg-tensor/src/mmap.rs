//! Zero-dependency read-only memory-mapped file view (`TensorBytes`) for zero-copy `.safetensors`,
//! `.gguf`, and `.vxt` model loading.

use std::fs;
#[cfg(all(unix, target_pointer_width = "64"))]
use std::fs::File;
use std::ops::Deref;
use std::path::Path;

use crate::{Error, Result};

#[cfg(all(unix, target_pointer_width = "64"))]
mod imp {
    use core::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::unix::io::AsRawFd;
    use std::ptr::{self, NonNull};

    const PROT_READ: i32 = 0x01;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux"))]
    const MADV_SEQUENTIAL: i32 = 2;

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
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux"))]
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }

    pub struct RawMap {
        ptr: NonNull<u8>,
        len: usize,
    }

    unsafe impl Send for RawMap {}
    unsafe impl Sync for RawMap {}

    impl RawMap {
        pub fn open_ro(file: &File, len: usize) -> io::Result<Self> {
            if len == 0 {
                return Ok(Self {
                    ptr: NonNull::dangling(),
                    len: 0,
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
            #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux"))]
            unsafe {
                let _ = madvise(raw, len, MADV_SEQUENTIAL);
            }
            Ok(Self {
                ptr: unsafe { NonNull::new_unchecked(raw as *mut u8) },
                len,
            })
        }

        #[inline]
        pub fn as_slice(&self) -> &[u8] {
            if self.len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
            }
        }
    }

    impl Drop for RawMap {
        fn drop(&mut self) {
            if self.len > 0 {
                unsafe {
                    let _ = munmap(self.ptr.as_ptr() as *mut c_void, self.len);
                }
            }
        }
    }
}

/// Read-only model bytes backed by OS `mmap` on 64-bit Unix or `Vec<u8>` fallback elsewhere.
pub enum TensorBytes {
    #[cfg(all(unix, target_pointer_width = "64"))]
    Mapped(imp::RawMap),
    Owned(Vec<u8>),
}

impl TensorBytes {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        #[cfg(all(unix, target_pointer_width = "64"))]
        {
            let file = File::open(p).map_err(|e| Error::Io(e.to_string()))?;
            let meta = file.metadata().map_err(|e| Error::Io(e.to_string()))?;
            if meta.is_file() {
                let len = usize::try_from(meta.len())
                    .map_err(|_| Error::Format("file too large for address space".into()))?;
                if let Ok(m) = imp::RawMap::open_ro(&file, len) {
                    return Ok(Self::Mapped(m));
                }
            }
        }
        let v = fs::read(p).map_err(|e| Error::Io(e.to_string()))?;
        Ok(Self::Owned(v))
    }

    #[inline]
    pub fn is_mapped(&self) -> bool {
        #[cfg(all(unix, target_pointer_width = "64"))]
        {
            matches!(self, Self::Mapped(_))
        }
        #[cfg(not(all(unix, target_pointer_width = "64")))]
        {
            false
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(all(unix, target_pointer_width = "64"))]
            Self::Mapped(m) => m.as_slice(),
            Self::Owned(v) => v.as_slice(),
        }
    }
}

impl Deref for TensorBytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
