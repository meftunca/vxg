//! Extended `libzstd` / `libzdict` baselines (`ZstdDict` on independent blocks and
//! whole-file `zstd -19 --long` without block boundaries).
use super::{zcheck, Chunk, Context};
use crate::{Error, Result};
use std::{
    ffi::{c_int, c_void},
    ptr::NonNull,
    time::Instant,
};

const ZSTD_C_COMPRESSION_LEVEL: c_int = 100;
const ZSTD_C_WINDOW_LOG: c_int = 101;
const ZSTD_C_ENABLE_LONG_DISTANCE_MATCHING: c_int = 160;
const ZSTD_C_NB_WORKERS: c_int = 400;

extern "C" {
    fn ZSTD_compressBound(n: usize) -> usize;
    fn ZSTD_CCtx_setParameter(cctx: *mut c_void, param: c_int, value: c_int) -> usize;
    fn ZSTD_compress2(
        cctx: *mut c_void,
        dst: *mut c_void,
        dst_cap: usize,
        src: *const c_void,
        src_size: usize,
    ) -> usize;
    fn ZSTD_decompressDCtx(
        ctx: *mut c_void,
        dst: *mut c_void,
        cap: usize,
        src: *const c_void,
        n: usize,
    ) -> usize;
    fn ZSTD_createCDict(dict: *const c_void, dict_size: usize, level: c_int) -> *mut c_void;
    fn ZSTD_freeCDict(cdict: *mut c_void) -> usize;
    fn ZSTD_compress_usingCDict(
        cctx: *mut c_void,
        dst: *mut c_void,
        dst_cap: usize,
        src: *const c_void,
        src_size: usize,
        cdict: *const c_void,
    ) -> usize;
    fn ZSTD_createDDict(dict: *const c_void, dict_size: usize) -> *mut c_void;
    fn ZSTD_freeDDict(ddict: *mut c_void) -> usize;
    fn ZSTD_decompress_usingDDict(
        dctx: *mut c_void,
        dst: *mut c_void,
        dst_cap: usize,
        src: *const c_void,
        src_size: usize,
        ddict: *const c_void,
    ) -> usize;
    fn ZDICT_trainFromBuffer(
        dict_buffer: *mut c_void,
        dict_cap: usize,
        samples_buffer: *const c_void,
        samples_sizes: *const usize,
        nb_samples: u32,
    ) -> usize;
    fn ZDICT_isError(code: usize) -> u32;
}

/// Digested `ZSTD_CDict` — documented by `zstd.h` as immutable and `Send + Sync` across threads.
pub(super) struct CDict(NonNull<c_void>);
// SAFETY: `zstd.h` guarantees `ZSTD_CDict` is read-only and safe for concurrent multi-thread use.
unsafe impl Send for CDict {}
unsafe impl Sync for CDict {}

impl CDict {
    pub(super) fn new(dict: &[u8], level: i32) -> Result<Self> {
        let p = unsafe { ZSTD_createCDict(dict.as_ptr().cast(), dict.len(), level) };
        Ok(Self(NonNull::new(p).ok_or(Error::SizeOverflow)?))
    }
    #[inline]
    pub(super) fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr()
    }
}
impl Drop for CDict {
    fn drop(&mut self) {
        unsafe {
            ZSTD_freeCDict(self.0.as_ptr());
        }
    }
}

/// Digested `ZSTD_DDict` — immutable and `Send + Sync` across worker threads.
pub(super) struct DDict(NonNull<c_void>);
// SAFETY: `zstd.h` guarantees `ZSTD_DDict` is read-only and safe for concurrent multi-thread use.
unsafe impl Send for DDict {}
unsafe impl Sync for DDict {}

impl DDict {
    pub(super) fn new(dict: &[u8]) -> Result<Self> {
        let p = unsafe { ZSTD_createDDict(dict.as_ptr().cast(), dict.len()) };
        Ok(Self(NonNull::new(p).ok_or(Error::SizeOverflow)?))
    }
    #[inline]
    pub(super) fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr()
    }
}
impl Drop for DDict {
    fn drop(&mut self) {
        unsafe {
            ZSTD_freeDDict(self.0.as_ptr());
        }
    }
}

/// Trained Zstandard dictionary (`ZDICT_trainFromBuffer`) plus measured one-time training latency.
#[derive(Clone, Debug)]
pub struct TrainedDict {
    pub bytes: Vec<u8>,
    pub train_ns: f64,
}

impl TrainedDict {
    /// Trains a Zstd dictionary (up to 32 KiB, scaled down for small inputs) from up to 1,024
    /// evenly strided samples across `input`. Falls back to an empty dictionary if `input` is
    /// degenerate (e.g. all-zeros) and `ZDICT_trainFromBuffer` reports insufficient entropy.
    pub fn train(input: &[u8]) -> Self {
        let start = Instant::now();
        if input.len() < 2048 {
            return Self {
                bytes: Vec::new(),
                train_ns: start.elapsed().as_secs_f64() * 1e9,
            };
        }
        let dict_cap = (input.len() / 64).clamp(2048, 32_768);
        let sample_len = (input.len() / 64).clamp(256, 2048);
        let nb_samples = (input.len() / sample_len).clamp(16, 1024);
        let mut samples_buf = Vec::with_capacity(nb_samples * sample_len);
        let mut sample_sizes = Vec::with_capacity(nb_samples);
        let max_start = input.len().saturating_sub(sample_len);
        for i in 0..nb_samples {
            let off = if nb_samples > 1 {
                (i * max_start) / (nb_samples - 1)
            } else {
                0
            };
            samples_buf.extend_from_slice(&input[off..off + sample_len]);
            sample_sizes.push(sample_len);
        }
        let mut dict = vec![0u8; dict_cap];
        let res = unsafe {
            ZDICT_trainFromBuffer(
                dict.as_mut_ptr().cast(),
                dict_cap,
                samples_buf.as_ptr().cast(),
                sample_sizes.as_ptr(),
                nb_samples as u32,
            )
        };
        let train_ns = start.elapsed().as_secs_f64() * 1e9;
        if unsafe { ZDICT_isError(res) != 0 } || res == 0 || res > dict_cap {
            Self {
                bytes: Vec::new(),
                train_ns,
            }
        } else {
            dict.truncate(res);
            Self {
                bytes: dict,
                train_ns,
            }
        }
    }
}

pub(super) fn encode_cdict_group(
    input: &[u8],
    bs: usize,
    cdict: &CDict,
    worker: usize,
    workers: usize,
) -> Result<Vec<(usize, Chunk)>> {
    let ctx = Context::new(true)?;
    let mut all = Vec::new();
    for (i, src) in input.chunks(bs).enumerate().skip(worker).step_by(workers) {
        let cap = unsafe { ZSTD_compressBound(src.len()) };
        let mut dst = vec![0u8; cap];
        let n = unsafe {
            zcheck(ZSTD_compress_usingCDict(
                ctx.ptr.as_ptr(),
                dst.as_mut_ptr().cast(),
                cap,
                src.as_ptr().cast(),
                src.len(),
                cdict.as_ptr(),
            ))?
        };
        let raw = n >= src.len();
        if raw {
            dst.clear();
            dst.extend_from_slice(src);
        } else {
            dst.truncate(n);
        }
        all.push((
            i,
            Chunk {
                bytes: dst,
                raw,
                raw_size: src.len(),
            },
        ));
    }
    Ok(all)
}

pub(super) fn decode_ddict_group(chunks: &[Chunk], out: &mut [u8], ddict: &DDict) -> Result<()> {
    let ctx = Context::new(false)?;
    let mut p = 0;
    for chunk in chunks {
        let dst = &mut out[p..p + chunk.raw_size];
        p += chunk.raw_size;
        if chunk.raw {
            dst.copy_from_slice(&chunk.bytes);
            continue;
        }
        let n = unsafe {
            zcheck(ZSTD_decompress_usingDDict(
                ctx.ptr.as_ptr(),
                dst.as_mut_ptr().cast(),
                dst.len(),
                chunk.bytes.as_ptr().cast(),
                chunk.bytes.len(),
                ddict.as_ptr(),
            ))?
        };
        if n != dst.len() {
            return Err(Error::Backend("ZSTD DDict decoded size mismatch".into()));
        }
    }
    Ok(())
}

pub(super) fn compress_whole_long(input: &[u8], level: i32, threads: usize) -> Result<Chunk> {
    if input.is_empty() {
        return Ok(Chunk {
            bytes: Vec::new(),
            raw: true,
            raw_size: 0,
        });
    }
    let ctx = Context::new(true)?;
    unsafe {
        zcheck(ZSTD_CCtx_setParameter(
            ctx.ptr.as_ptr(),
            ZSTD_C_COMPRESSION_LEVEL,
            level,
        ))?;
        zcheck(ZSTD_CCtx_setParameter(
            ctx.ptr.as_ptr(),
            ZSTD_C_WINDOW_LOG,
            27,
        ))?;
        zcheck(ZSTD_CCtx_setParameter(
            ctx.ptr.as_ptr(),
            ZSTD_C_ENABLE_LONG_DISTANCE_MATCHING,
            1,
        ))?;
        if threads > 1 {
            // Optional: falls back silently to single-threaded if libzstd lacks ZSTD_MULTITHREAD.
            let _ = ZSTD_CCtx_setParameter(ctx.ptr.as_ptr(), ZSTD_C_NB_WORKERS, threads as c_int);
        }
        let cap = ZSTD_compressBound(input.len());
        let mut dst = vec![0u8; cap];
        let n = zcheck(ZSTD_compress2(
            ctx.ptr.as_ptr(),
            dst.as_mut_ptr().cast(),
            cap,
            input.as_ptr().cast(),
            input.len(),
        ))?;
        let raw = n >= input.len();
        if raw {
            dst.clear();
            dst.extend_from_slice(input);
        } else {
            dst.truncate(n);
        }
        Ok(Chunk {
            bytes: dst,
            raw,
            raw_size: input.len(),
        })
    }
}

pub(super) fn decode_whole(chunk: &Chunk, out: &mut [u8]) -> Result<()> {
    if chunk.raw {
        out.copy_from_slice(&chunk.bytes);
        return Ok(());
    }
    let ctx = Context::new(false)?;
    let n = unsafe {
        zcheck(ZSTD_decompressDCtx(
            ctx.ptr.as_ptr(),
            out.as_mut_ptr().cast(),
            out.len(),
            chunk.bytes.as_ptr().cast(),
            chunk.bytes.len(),
        ))?
    };
    if n != out.len() {
        return Err(Error::Backend(
            "ZSTD whole-file decoded size mismatch".into(),
        ));
    }
    Ok(())
}
