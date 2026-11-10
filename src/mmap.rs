//! Zero-dependency cross-platform memory-mapped file I/O (`MappedFile`, `MappedFileMut`,
//! `InputBytes`) and zero-copy file-to-file codec helpers (`compress_file`, `decompress_file`,
//! `verify_file`).

use std::fs::{self, File, OpenOptions};
use std::ops::{Deref, DerefMut};
use std::path::Path;

use crate::{
    decompress_into, max_compressed_size, stream::scan_concatenated_frames, verify_frame,
    DecodeOptions, Error, Frame, Options, Result,
};

pub(crate) mod pread;
pub(crate) mod sys;

pub(crate) use pread::peek_cold_prefix;
pub use pread::{pread_bytes_count, pread_call_count, reset_pread_counters};
pub use sys::{
    legacy_four_page_probe, mincore_call_count, reset_mincore_call_count, ChunkResidency,
    CHUNK_BYTES,
};

/// Returns `true` if native OS memory mapping is compiled in for this target.
#[inline]
pub const fn mmap_supported() -> bool {
    sys::SUPPORTED
}

/// Read-only memory-mapped file view (`&[u8]`) with zero heap allocation.
pub struct MappedFile {
    raw: sys::RawMap,
}

// SAFETY: Read-only virtual memory mapping is safe to read concurrently across threads.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    /// Maps an open [`File`] read-only into the process address space.
    ///
    /// # Safety
    /// The caller must ensure no external process concurrently truncates or mutates the
    /// underlying file while this [`MappedFile`] remains alive.
    pub unsafe fn from_file(file: &File) -> Result<Self> {
        let meta = file.metadata().map_err(|e| Error::Backend(e.to_string()))?;
        if !meta.is_file() {
            return Err(Error::Unsupported("mmap requires a regular file".into()));
        }
        let len = usize::try_from(meta.len()).map_err(|_| Error::SizeOverflow)?;
        let raw =
            unsafe { sys::RawMap::map_ro(file, len) }.map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self { raw })
    }

    /// Opens `path` read-only and maps it into virtual memory without copying bytes into heap RAM.
    ///
    /// # Safety
    /// The caller must ensure no external process concurrently truncates or mutates `path`
    /// while this [`MappedFile`] remains alive.
    pub unsafe fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| Error::Backend(e.to_string()))?;
        unsafe { Self::from_file(&file) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.raw.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw.ptr.as_ptr(), self.raw.len) }
        }
    }

    /// Hints the OS kernel to prefetch pages sequentially (`MADV_SEQUENTIAL` on Unix).
    #[inline]
    pub fn advise_sequential(&self) {
        self.raw.advise_seq();
    }

    /// Hints the OS kernel to disable speculative readahead (`MADV_RANDOM` on Unix).
    #[inline]
    pub fn advise_random(&self) {
        self.raw.advise_rand();
    }

    /// Hints the OS kernel to fault in `offset .. offset + len` asynchronously (`MADV_WILLNEED`).
    #[inline]
    pub fn advise_willneed(&self, offset: usize, len: usize) {
        self.raw.advise_willneed(offset, len);
    }
}

impl Deref for MappedFile {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for MappedFile {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl core::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MappedFile")
            .field("ptr", &self.raw.ptr)
            .field("len", &self.raw.len)
            .finish()
    }
}

/// Read-write shared memory-mapped file view (`&mut [u8]`) for zero-copy output emission.
pub struct MappedFileMut {
    raw: sys::RawMap,
}

unsafe impl Send for MappedFileMut {}
unsafe impl Sync for MappedFileMut {}

impl MappedFileMut {
    /// Resizes `file` to `len` bytes and maps it read-write (`MAP_SHARED` / `FILE_MAP_WRITE`).
    ///
    /// # Safety
    /// The caller must ensure no other process or handle concurrently resizes or aliases `file`
    /// while this [`MappedFileMut`] remains alive.
    pub unsafe fn from_file(file: &File, len: usize) -> Result<Self> {
        file.set_len(len as u64)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let raw =
            unsafe { sys::RawMap::map_rw(file, len) }.map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self { raw })
    }

    /// Creates `path` (honoring `force` overwrite policy), sizes it to `len` bytes, and maps it.
    ///
    /// # Safety
    /// The caller must ensure no external process concurrently resizes or mutates `path`
    /// while this [`MappedFileMut`] remains alive.
    pub unsafe fn create<P: AsRef<Path>>(path: P, len: usize, force: bool) -> Result<Self> {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true);
        if force {
            opts.create(true).truncate(true);
        } else {
            opts.create_new(true);
        }
        let file = opts
            .open(path.as_ref())
            .map_err(|e| Error::Backend(e.to_string()))?;
        unsafe { Self::from_file(&file, len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.raw.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.raw.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.raw.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw.ptr.as_ptr(), self.raw.len) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.raw.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.raw.ptr.as_ptr(), self.raw.len) }
        }
    }

    /// Synchronously flushes dirty pages back to the OS filesystem (`msync` / `FlushViewOfFile`).
    pub fn flush(&self) -> Result<()> {
        self.raw.flush().map_err(|e| Error::Backend(e.to_string()))
    }
}

impl Deref for MappedFileMut {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for MappedFileMut {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for MappedFileMut {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for MappedFileMut {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

/// Unified read-only byte buffer backed by either a zero-copy [`MappedFile`] or heap `Vec<u8>`.
#[derive(Debug)]
pub enum InputBytes {
    Mapped(MappedFile),
    Owned(Vec<u8>),
}

impl InputBytes {
    /// Opens `path` using zero-copy OS `mmap` when `allow_mmap` is `true` and `path` is a regular
    /// file on a supported 64-bit platform; transparently falls back to `std::fs::read` otherwise.
    pub fn open_path_mode<P: AsRef<Path>>(path: P, allow_mmap: bool) -> Result<Self> {
        let p = path.as_ref();
        if allow_mmap && sys::SUPPORTED {
            let file = File::open(p).map_err(|e| Error::Backend(e.to_string()))?;
            if let Ok(meta) = file.metadata() {
                if meta.is_file() {
                    if let Ok(mapped) = unsafe { MappedFile::from_file(&file) } {
                        return Ok(Self::Mapped(mapped));
                    }
                }
            }
        }
        let bytes = fs::read(p).map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self::Owned(bytes))
    }

    /// Opens `path` preferring zero-copy OS `mmap` with automatic `std::fs::read` fallback.
    #[inline]
    pub fn open_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_path_mode(path, true)
    }

    #[inline]
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(m) => m.as_slice(),
            Self::Owned(v) => v.as_slice(),
        }
    }
}

impl Deref for InputBytes {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for InputBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

fn open_out_file(path: &Path, force: bool) -> Result<(File, bool)> {
    let existed = path.exists();
    let mut opts = OpenOptions::new();
    opts.read(true).write(true);
    if force {
        opts.create(true).truncate(true);
    } else {
        opts.create_new(true);
    }
    let f = opts.open(path).map_err(|e| Error::Backend(e.to_string()))?;
    Ok((f, !existed))
}

/// Zero-input-copy file-to-file compression using read-only [`MappedFile`] + sequential file write.
/// Avoids OS `MAP_SHARED` sparse write-fault overhead while eliminating input heap allocation.
/// Returns `(raw_bytes, compressed_bytes)`.
pub fn compress_file<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    options: &Options,
    force: bool,
) -> Result<(u64, u64)> {
    let in_p = input.as_ref();
    let out_p = output.as_ref();
    if in_p == out_p {
        return Err(Error::InvalidOption("input and output paths must differ"));
    }
    options.validate()?;
    let in_len = fs::metadata(in_p)
        .map_err(|e| Error::Backend(e.to_string()))?
        .len();
    let src = InputBytes::open_path_mode(in_p, in_len > (64 << 20))?;
    if let InputBytes::Mapped(ref m) = src {
        m.advise_sequential();
    }
    max_compressed_size(src.len(), options.block_size)?;
    let (mut out_file, created_new) = open_out_file(out_p, force)?;
    let res = (|| -> Result<usize> {
        let encoded = crate::compress(&src, options)?;
        std::io::Write::write_all(&mut out_file, &encoded)
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok(encoded.len())
    })();
    match res {
        Ok(written) => Ok((src.len() as u64, written as u64)),
        Err(e) => {
            drop(out_file);
            if created_new {
                let _ = fs::remove_file(out_p);
            }
            Err(e)
        }
    }
}

fn decode_large_frames_bounded(
    src: &[u8],
    spans: &[(usize, usize, usize)],
    out_file: &mut File,
    options: &DecodeOptions,
) -> Result<()> {
    const BATCH_BYTES: usize = 16 << 20;
    let mut batch_buf: Vec<u8> = Vec::new();
    for &(start, end, f_raw) in spans {
        let f_slice = &src[start..end];
        if f_raw <= BATCH_BYTES {
            batch_buf.resize(f_raw, 0);
            let mut sub_opt = options.clone();
            sub_opt.max_output = f_raw;
            decompress_into(f_slice, &mut batch_buf[..f_raw], &sub_opt)?;
            std::io::Write::write_all(out_file, &batch_buf[..f_raw])
                .map_err(|e| Error::Backend(e.to_string()))?;
            continue;
        }
        let parsed = Frame::parse(f_slice, f_raw)?;
        let nb = parsed.block_count();
        let blocks_per_batch = (BATCH_BYTES / parsed.block_size().max(1)).max(1);
        let mut b_start = 0usize;
        while b_start < nb {
            let b_end = (b_start + blocks_per_batch).min(nb);
            let chunk_raw: usize = (b_start..b_end)
                .map(|bi| parsed.blocks[bi].info.raw_size)
                .sum();
            batch_buf.resize(chunk_raw, 0);
            let workers = options.threads.min(b_end - b_start).max(1);
            if workers == 1 {
                let mut pos = 0usize;
                for bi in b_start..b_end {
                    let rn = parsed.blocks[bi].info.raw_size;
                    parsed.decode_block_into(bi, &mut batch_buf[pos..pos + rn], options)?;
                    pos += rn;
                }
            } else {
                let mut offsets = Vec::with_capacity(b_end - b_start);
                let mut acc = 0usize;
                for bi in b_start..b_end {
                    offsets.push(acc);
                    acc += parsed.blocks[bi].info.raw_size;
                }
                let base_ptr = batch_buf.as_mut_ptr() as usize;
                let next = std::sync::atomic::AtomicUsize::new(0);
                crate::pool::parallel_try_run(workers, |_| {
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= b_end - b_start {
                            return Ok(());
                        }
                        let bi = b_start + k;
                        let rn = parsed.blocks[bi].info.raw_size;
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                (base_ptr + offsets[k]) as *mut u8,
                                rn,
                            )
                        };
                        parsed.decode_block_into(bi, dst, options)?;
                    }
                })?;
            }
            std::io::Write::write_all(out_file, &batch_buf[..chunk_raw])
                .map_err(|e| Error::Backend(e.to_string()))?;
            b_start = b_end;
        }
    }
    Ok(())
}

/// Zero-input-copy file-to-file decompression using read-only [`MappedFile`] and bounded-memory
/// (`<= 64 MiB` fast path, `16 MiB` batched streaming for multi-GB frames) sequential output writes.
/// Returns exact uncompressed bytes written.
pub fn decompress_file<P: AsRef<Path>, Q: AsRef<Path>>(
    input: P,
    output: Q,
    options: &DecodeOptions,
    force: bool,
) -> Result<u64> {
    let in_p = input.as_ref();
    let out_p = output.as_ref();
    if in_p == out_p {
        return Err(Error::InvalidOption("input and output paths must differ"));
    }
    options.validate()?;
    let in_len = fs::metadata(in_p)
        .map_err(|e| Error::Backend(e.to_string()))?
        .len();
    let src = InputBytes::open_path_mode(in_p, in_len > (64 << 20))?;
    if let InputBytes::Mapped(ref m) = src {
        m.advise_sequential();
    }
    let (raw_len, spans) = scan_concatenated_frames(&src, options.max_output)?;
    let (mut out_file, created_new) = open_out_file(out_p, force)?;
    let res = (|| -> Result<()> {
        if raw_len <= (64 << 20) {
            let decoded = crate::decompress(&src, options)?;
            std::io::Write::write_all(&mut out_file, &decoded)
                .map_err(|e| Error::Backend(e.to_string()))?;
        } else {
            decode_large_frames_bounded(&src, &spans, &mut out_file, options)?;
        }
        Ok(())
    })();
    if let Err(e) = res {
        drop(out_file);
        if created_new {
            let _ = fs::remove_file(out_p);
        }
        return Err(e);
    }
    Ok(raw_len as u64)
}

/// Zero-copy file verification using `MappedFile` + zero-output-allocation [`verify_frame`].
pub fn verify_file<P: AsRef<Path>>(input: P, options: &DecodeOptions) -> Result<u64> {
    let src = InputBytes::open_path(input)?;
    if let InputBytes::Mapped(ref m) = src {
        m.advise_sequential();
    }
    verify_frame(&src, options)
}
