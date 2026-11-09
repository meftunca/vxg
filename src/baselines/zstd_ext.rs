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

/// Candidate Zstandard dictionary budgets (`32`, `64`, `112`, and `256 KiB`) swept per dataset
/// so `Zstd+Dict` receives up to the same `256 KiB` dictionary budget as `VXG v5` while never
/// being penalized when a smaller dictionary produces a lower total `encoded_size()` (including
/// dictionary bytes).
pub const ZSTD_DICT_SWEEP_CAPS: [usize; 4] = [32_768, 65_536, 114_688, 262_144];

/// Trained Zstandard dictionary (`ZDICT_trainFromBuffer`) plus measured one-time training latency.
#[derive(Clone, Debug)]
pub struct TrainedDict {
    pub bytes: Vec<u8>,
    pub train_ns: f64,
}

impl TrainedDict {
    /// Trains a default `<= 32 KiB` Zstd dictionary from evenly strided samples across `input`.
    pub fn train(input: &[u8]) -> Self {
        Self::train_with_cap(input, 32_768)
    }

    /// Trains a Zstd dictionary up to `max_dict_cap` bytes (`2 KiB ..= 256 KiB`, capped at
    /// `input.len() / 2` on tiny inputs) from evenly strided samples across `input`. Falls back to
    /// an empty dictionary if `input` is degenerate (e.g. all-zeros) and `ZDICT_trainFromBuffer`
    /// reports insufficient entropy.
    pub fn train_with_cap(input: &[u8], max_dict_cap: usize) -> Self {
        let start = Instant::now();
        if input.len() < 2048 || max_dict_cap < 512 {
            return Self {
                bytes: Vec::new(),
                train_ns: start.elapsed().as_secs_f64() * 1e9,
            };
        }
        let dict_cap = max_dict_cap.clamp(2048, 262_144).min(input.len() / 2);
        if dict_cap < 512 {
            return Self {
                bytes: Vec::new(),
                train_ns: start.elapsed().as_secs_f64() * 1e9,
            };
        }
        let target_total = (dict_cap * 16).clamp(65_536, 4_194_304).min(input.len());
        let sample_len = (target_total / 128)
            .clamp(256, 4096)
            .min(input.len() / 8)
            .max(64);
        let nb_samples = (target_total / sample_len)
            .clamp(8, 2048)
            .min(input.len() / sample_len)
            .max(1);
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

    /// Trains candidate dictionaries across [`ZSTD_DICT_SWEEP_CAPS`] (`32 / 64 / 112 / 256 KiB`),
    /// deduplicating effective caps on small inputs and returning `(budget_bytes, TrainedDict)`.
    pub fn train_sweep(input: &[u8]) -> Vec<(usize, Self)> {
        let mut out = Vec::with_capacity(ZSTD_DICT_SWEEP_CAPS.len());
        let mut last_eff = 0usize;
        for &cap in &ZSTD_DICT_SWEEP_CAPS {
            if input.len() < 2048 {
                break;
            }
            let eff = cap.clamp(2048, 262_144).min(input.len() / 2);
            if eff == last_eff {
                continue;
            }
            last_eff = eff;
            let d = Self::train_with_cap(input, cap);
            if !d.bytes.is_empty() {
                out.push((cap, d));
            }
        }
        if out.is_empty() {
            let d = Self::train_with_cap(input, 32_768);
            out.push((0, d));
        }
        out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::{Codec, Encoded};

    #[test]
    fn zstd_dict_sweep_covers_32_to_256_kib_and_picks_minimum() {
        let mut src = Vec::with_capacity(1_200_000);
        for i in 0..10_000u32 {
            src.extend_from_slice(
                format!(
                    "row={i} host=\"edge-gateway-{}\" service=\"billing-ledger-{}\" trace=\"{:08x}{:08x}\"\n",
                    i % 64,
                    i % 37,
                    i.wrapping_mul(0x9E37_79B9),
                    i.wrapping_mul(0x85EB_CA6B),
                )
                .as_bytes(),
            );
        }
        let sweep = TrainedDict::train_sweep(&src);
        assert_eq!(sweep.len(), ZSTD_DICT_SWEEP_CAPS.len());
        let caps: Vec<usize> = sweep.iter().map(|(c, _)| *c).collect();
        assert_eq!(caps, ZSTD_DICT_SWEEP_CAPS.to_vec());

        let (best_dict, best_cap, best_enc) = Encoded::compress_best_dict_sweep(
            &src,
            Codec::ZstdDict { level: 3 },
            &sweep,
            65536,
            2,
        )
        .unwrap();
        assert!(ZSTD_DICT_SWEEP_CAPS.contains(&best_cap));
        assert_eq!(best_enc.dict_size(), best_dict.bytes.len());
        for (_, d) in &sweep {
            let cand =
                Encoded::compress_with_dict(&src, Codec::ZstdDict { level: 3 }, d, 65536, 2)
                    .unwrap();
            assert!(best_enc.encoded_size() <= cand.encoded_size());
        }
        let mut out = vec![0u8; src.len()];
        best_enc.decode_into(&mut out, 2).unwrap();
        assert_eq!(out, src);
    }
}
