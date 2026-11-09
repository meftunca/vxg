//! Bench-only C reference implementations. Link real liblz4/libzstd, not surrogate codecs.
//! Block codecs share the same independent block size, worker count, and `32 + 32*block_count`
//! accounting envelope (`ZstdDict` additionally charges its trained dictionary bytes).
//! `ZstdWholeLong` compresses the full input as a single stream (`windowLog = 27`, `--long`)
//! without block boundaries, charging only the standalone Zstd frame bytes.
use crate::{Error, Result};
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    ptr::NonNull,
};

#[path = "baselines/zstd_ext.rs"]
mod zstd_ext;
pub use zstd_ext::{TrainedDict, ZSTD_DICT_SWEEP_CAPS};
use zstd_ext::{
    compress_whole_long, decode_ddict_group, decode_whole, encode_cdict_group, CDict, DDict,
};

extern "C" {
    fn LZ4_versionString() -> *const c_char;
    fn LZ4_compressBound(n: c_int) -> c_int;
    fn LZ4_sizeofState() -> c_int;
    fn LZ4_sizeofStateHC() -> c_int;
    fn LZ4_compress_fast_extState(
        state: *mut c_void,
        src: *const c_char,
        dst: *mut c_char,
        n: c_int,
        cap: c_int,
        acc: c_int,
    ) -> c_int;
    fn LZ4_compress_HC_extStateHC(
        state: *mut c_void,
        src: *const c_char,
        dst: *mut c_char,
        n: c_int,
        cap: c_int,
        level: c_int,
    ) -> c_int;
    fn LZ4_decompress_safe(src: *const c_char, dst: *mut c_char, size: c_int, cap: c_int) -> c_int;
    fn ZSTD_versionString() -> *const c_char;
    fn ZSTD_compressBound(n: usize) -> usize;
    fn ZSTD_createCCtx() -> *mut c_void;
    fn ZSTD_freeCCtx(p: *mut c_void) -> usize;
    fn ZSTD_createDCtx() -> *mut c_void;
    fn ZSTD_freeDCtx(p: *mut c_void) -> usize;
    fn ZSTD_compressCCtx(
        ctx: *mut c_void,
        dst: *mut c_void,
        cap: usize,
        src: *const c_void,
        n: usize,
        level: c_int,
    ) -> usize;
    fn ZSTD_decompressDCtx(
        ctx: *mut c_void,
        dst: *mut c_void,
        cap: usize,
        src: *const c_void,
        n: usize,
    ) -> usize;
    fn ZSTD_isError(code: usize) -> u32;
    fn ZSTD_getErrorName(code: usize) -> *const c_char;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Lz4 { acceleration: i32 },
    Lz4Hc { level: i32 },
    Zstd { level: i32 },
    ZstdDict { level: i32 },
    ZstdWholeLong { level: i32 },
}

impl Codec {
    pub fn name(self) -> String {
        match self {
            Self::Lz4 { acceleration } => format!("lz4-c-a{acceleration}"),
            Self::Lz4Hc { level } => format!("lz4hc-c-l{level}"),
            Self::Zstd { level } => format!("zstd-c-l{level}"),
            Self::ZstdDict { level } => format!("zstd-dict-l{level}"),
            Self::ZstdWholeLong { level } => format!("zstd-whole-l{level}-long"),
        }
    }
}

pub fn versions() -> (String, String) {
    // SAFETY: reference libraries return static, NUL-terminated version strings.
    unsafe {
        (
            CStr::from_ptr(LZ4_versionString())
                .to_string_lossy()
                .into_owned(),
            CStr::from_ptr(ZSTD_versionString())
                .to_string_lossy()
                .into_owned(),
        )
    }
}

pub(crate) fn zcheck(n: usize) -> Result<usize> {
    unsafe {
        if ZSTD_isError(n) != 0 {
            Err(Error::Backend(
                CStr::from_ptr(ZSTD_getErrorName(n))
                    .to_string_lossy()
                    .into_owned(),
            ))
        } else {
            Ok(n)
        }
    }
}

pub(crate) struct Context {
    pub(crate) ptr: NonNull<c_void>,
    encode: bool,
}
impl Context {
    pub(crate) fn new(encode: bool) -> Result<Self> {
        let p = unsafe {
            if encode {
                ZSTD_createCCtx()
            } else {
                ZSTD_createDCtx()
            }
        };
        Ok(Self {
            ptr: NonNull::new(p).ok_or(Error::SizeOverflow)?,
            encode,
        })
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if self.encode {
                ZSTD_freeCCtx(self.ptr.as_ptr());
            } else {
                ZSTD_freeDCtx(self.ptr.as_ptr());
            }
        }
    }
}

pub(crate) struct Chunk {
    pub(crate) bytes: Vec<u8>,
    pub(crate) raw: bool,
    pub(crate) raw_size: usize,
}

pub struct Encoded {
    codec: Codec,
    chunks: Vec<Chunk>,
    raw_size: usize,
    block_size: usize,
    dict_size: usize,
    ddict: Option<DDict>,
}

impl Encoded {
    pub fn encoded_size(&self) -> usize {
        let payload: usize = self.chunks.iter().map(|c| c.bytes.len()).sum();
        if matches!(self.codec, Codec::ZstdWholeLong { .. }) {
            payload
        } else {
            self.dict_size + 32 + 32 * self.chunks.len() + payload
        }
    }

    pub fn dict_size(&self) -> usize {
        self.dict_size
    }

    pub fn compress(input: &[u8], codec: Codec, block_size: usize, threads: usize) -> Result<Self> {
        if matches!(codec, Codec::ZstdDict { .. }) {
            let candidates = TrainedDict::train_sweep(input);
            let (_, _, enc) =
                Self::compress_best_dict_sweep(input, codec, &candidates, block_size, threads)?;
            return Ok(enc);
        }
        Self::compress_inner(input, codec, None, block_size, threads)
    }

    pub fn compress_with_dict(
        input: &[u8],
        codec: Codec,
        dict: &TrainedDict,
        block_size: usize,
        threads: usize,
    ) -> Result<Self> {
        Self::compress_inner(input, codec, Some(&dict.bytes), block_size, threads)
    }

    /// Evaluates pre-trained `(budget_bytes, TrainedDict)` candidates and returns the candidate
    /// achieving the lowest `encoded_size()` (charging `dict.bytes.len()`).
    pub fn compress_best_dict_sweep<'a>(
        input: &[u8],
        codec: Codec,
        candidates: &'a [(usize, TrainedDict)],
        block_size: usize,
        threads: usize,
    ) -> Result<(&'a TrainedDict, usize, Self)> {
        if candidates.is_empty() {
            return Err(Error::InvalidOption("empty ZstdDict candidate sweep"));
        }
        let mut best: Option<(&'a TrainedDict, usize, Self)> = None;
        for (cap, dict) in candidates {
            let enc = Self::compress_with_dict(input, codec, dict, block_size, threads)?;
            let take = match &best {
                None => true,
                Some((_, _, prev)) => enc.encoded_size() < prev.encoded_size(),
            };
            if take {
                best = Some((dict, *cap, enc));
            }
        }
        Ok(best.unwrap())
    }

    fn compress_inner(
        input: &[u8],
        codec: Codec,
        dict_bytes: Option<&[u8]>,
        block_size: usize,
        threads: usize,
    ) -> Result<Self> {
        crate::frame::check_block_size(block_size)?;
        if !(1..=256).contains(&threads) {
            return Err(Error::InvalidOption("threads must be 1..256"));
        }
        if matches!(codec, Codec::Lz4 { acceleration } if acceleration < 1) {
            return Err(Error::InvalidOption("LZ4 acceleration must be positive"));
        }
        if matches!(codec, Codec::Lz4Hc { level } if !(1..=12).contains(&level)) {
            return Err(Error::InvalidOption("LZ4-HC level must be 1..=12"));
        }
        if let Codec::ZstdWholeLong { level } = codec {
            let chunk = compress_whole_long(input, level, threads)?;
            return Ok(Self {
                codec,
                chunks: if input.is_empty() {
                    Vec::new()
                } else {
                    vec![chunk]
                },
                raw_size: input.len(),
                block_size: input.len().max(1),
                dict_size: 0,
                ddict: None,
            });
        }
        let count = input.len().div_ceil(block_size);
        let workers = threads.min(count.max(1));
        if let Codec::ZstdDict { level } = codec {
            let db = dict_bytes.unwrap_or(&[]);
            let cdict = CDict::new(db, level)?;
            let ddict = DDict::new(db)?;
            let mut indexed = if workers == 1 {
                encode_cdict_group(input, block_size, &cdict, 0, 1)?
            } else {
                let cdict_ref = &cdict;
                std::thread::scope(|scope| -> Result<Vec<_>> {
                    let mut handles = Vec::new();
                    for w in 0..workers {
                        handles.push(scope.spawn(move || {
                            encode_cdict_group(input, block_size, cdict_ref, w, workers)
                        }));
                    }
                    let mut all = Vec::new();
                    for h in handles {
                        all.extend(h.join().map_err(|_| Error::WorkerPanic)??);
                    }
                    Ok(all)
                })?
            };
            indexed.sort_unstable_by_key(|x| x.0);
            return Ok(Self {
                codec,
                chunks: indexed.into_iter().map(|x| x.1).collect(),
                raw_size: input.len(),
                block_size,
                dict_size: db.len(),
                ddict: Some(ddict),
            });
        }
        let mut indexed = if workers == 1 {
            encode_group(input, codec, block_size, 0, 1)?
        } else {
            std::thread::scope(|scope| -> Result<Vec<_>> {
                let mut handles = Vec::new();
                for w in 0..workers {
                    handles.push(
                        scope.spawn(move || encode_group(input, codec, block_size, w, workers)),
                    );
                }
                let mut all = Vec::new();
                for h in handles {
                    all.extend(h.join().map_err(|_| Error::WorkerPanic)??);
                }
                Ok(all)
            })?
        };
        indexed.sort_unstable_by_key(|x| x.0);
        Ok(Self {
            codec,
            chunks: indexed.into_iter().map(|x| x.1).collect(),
            raw_size: input.len(),
            block_size,
            dict_size: 0,
            ddict: None,
        })
    }

    pub fn decode_into(&self, out: &mut [u8], threads: usize) -> Result<()> {
        if out.len() != self.raw_size {
            return Err(Error::OutputSize {
                expected: self.raw_size,
                actual: out.len(),
            });
        }
        if !(1..=256).contains(&threads) {
            return Err(Error::InvalidOption("threads must be 1..256"));
        }
        if self.chunks.is_empty() {
            return Ok(());
        }
        if matches!(self.codec, Codec::ZstdWholeLong { .. }) {
            return decode_whole(&self.chunks[0], out);
        }
        let workers = threads.min(self.chunks.len());
        let per = self.chunks.len().div_ceil(workers);
        if let Some(ddict) = self.ddict.as_ref() {
            if workers == 1 {
                return decode_ddict_group(&self.chunks, out, ddict);
            }
            return std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for (g, target) in out.chunks_mut(per * self.block_size).enumerate() {
                    let start = g * per;
                    let chunks = &self.chunks[start..(start + per).min(self.chunks.len())];
                    handles.push(scope.spawn(move || decode_ddict_group(chunks, target, ddict)));
                }
                for h in handles {
                    h.join().map_err(|_| Error::WorkerPanic)??;
                }
                Ok(())
            });
        }
        if workers == 1 {
            return decode_group(&self.chunks, out, self.codec);
        }
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (g, target) in out.chunks_mut(per * self.block_size).enumerate() {
                let start = g * per;
                let chunks = &self.chunks[start..(start + per).min(self.chunks.len())];
                let codec = self.codec;
                handles.push(scope.spawn(move || decode_group(chunks, target, codec)));
            }
            for h in handles {
                h.join().map_err(|_| Error::WorkerPanic)??;
            }
            Ok(())
        })
    }
}

fn encode_group(
    input: &[u8],
    codec: Codec,
    bs: usize,
    worker: usize,
    workers: usize,
) -> Result<Vec<(usize, Chunk)>> {
    let ctx = if matches!(codec, Codec::Zstd { .. }) {
        Some(Context::new(true)?)
    } else {
        None
    };
    let mut state = match codec {
        Codec::Lz4 { .. } => {
            let n = unsafe { LZ4_sizeofState() };
            if n <= 0 {
                return Err(Error::Backend("LZ4 state size".into()));
            }
            vec![0u64; (n as usize).div_ceil(8)]
        }
        Codec::Lz4Hc { .. } => {
            let n = unsafe { LZ4_sizeofStateHC() };
            if n <= 0 {
                return Err(Error::Backend("LZ4-HC state size".into()));
            }
            vec![0u64; (n as usize).div_ceil(8)]
        }
        _ => Vec::new(),
    };
    let mut all = Vec::new();
    for (i, src) in input.chunks(bs).enumerate().skip(worker).step_by(workers) {
        let cap = unsafe {
            match codec {
                Codec::Lz4 { .. } | Codec::Lz4Hc { .. } => {
                    LZ4_compressBound(src.len() as c_int) as usize
                }
                _ => ZSTD_compressBound(src.len()),
            }
        };
        let mut dst = vec![0u8; cap];
        let n = unsafe {
            match codec {
                Codec::Lz4 { acceleration } => {
                    let n = LZ4_compress_fast_extState(
                        state.as_mut_ptr().cast(),
                        src.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                        src.len() as c_int,
                        cap as c_int,
                        acceleration,
                    );
                    if n <= 0 {
                        return Err(Error::Backend("LZ4 compression failed".into()));
                    }
                    n as usize
                }
                Codec::Lz4Hc { level } => {
                    let n = LZ4_compress_HC_extStateHC(
                        state.as_mut_ptr().cast(),
                        src.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                        src.len() as c_int,
                        cap as c_int,
                        level,
                    );
                    if n <= 0 {
                        return Err(Error::Backend("LZ4-HC compression failed".into()));
                    }
                    n as usize
                }
                Codec::Zstd { level } => zcheck(ZSTD_compressCCtx(
                    ctx.as_ref().unwrap().ptr.as_ptr(),
                    dst.as_mut_ptr().cast(),
                    cap,
                    src.as_ptr().cast(),
                    src.len(),
                    level,
                ))?,
                _ => unreachable!(),
            }
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

fn decode_group(chunks: &[Chunk], out: &mut [u8], codec: Codec) -> Result<()> {
    let ctx = if matches!(codec, Codec::Zstd { .. }) {
        Some(Context::new(false)?)
    } else {
        None
    };
    let mut p = 0;
    for chunk in chunks {
        let dst = &mut out[p..p + chunk.raw_size];
        p += chunk.raw_size;
        if chunk.raw {
            dst.copy_from_slice(&chunk.bytes);
            continue;
        }
        let n = unsafe {
            match codec {
                Codec::Lz4 { .. } | Codec::Lz4Hc { .. } => {
                    let n = LZ4_decompress_safe(
                        chunk.bytes.as_ptr().cast(),
                        dst.as_mut_ptr().cast(),
                        chunk.bytes.len() as c_int,
                        dst.len() as c_int,
                    );
                    if n < 0 {
                        return Err(Error::Backend("LZ4 decompression failed".into()));
                    }
                    n as usize
                }
                Codec::Zstd { .. } => zcheck(ZSTD_decompressDCtx(
                    ctx.as_ref().unwrap().ptr.as_ptr(),
                    dst.as_mut_ptr().cast(),
                    dst.len(),
                    chunk.bytes.as_ptr().cast(),
                    chunk.bytes.len(),
                ))?,
                _ => unreachable!(),
            }
        };
        if n != dst.len() {
            return Err(Error::Backend("baseline decoded size mismatch".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baselines_dict_and_whole_long_roundtrip() {
        let mut src = Vec::with_capacity(2_000_000);
        for i in 0..8_000u32 {
            src.extend_from_slice(
                format!(
                    "{{\"timestamp\":\"2026-01-15T12:34:{:02}Z\",\"organization\":\"acme-telemetry-prod\",\"environment\":\"kubernetes-eu-central-1\",\"service_name\":\"checkout-authorization-gateway\",\"http_status\":{},\"trace_id\":\"{:08x}{:08x}\",\"message\":\"authorization evaluation completed\"}}\n",
                    i % 60,
                    if i % 7 == 0 { 500 } else { 200 },
                    i.wrapping_mul(0x9E37_79B9),
                    i.wrapping_mul(0x85EB_CA6B),
                )
                .as_bytes(),
            );
        }
        let mut out = vec![0u8; src.len()];
        let dict = TrainedDict::train(&src);
        assert!(!dict.bytes.is_empty());

        for bs in [4096usize, 65536] {
            for codec in [
                Codec::Lz4 { acceleration: 1 },
                Codec::Lz4Hc { level: 9 },
                Codec::Zstd { level: 3 },
                Codec::Zstd { level: 9 },
                Codec::ZstdDict { level: 3 },
                Codec::ZstdDict { level: 9 },
                Codec::ZstdWholeLong { level: 19 },
            ] {
                let enc = if matches!(codec, Codec::ZstdDict { .. }) {
                    Encoded::compress_with_dict(&src, codec, &dict, bs, 2).unwrap()
                } else {
                    Encoded::compress(&src, codec, bs, 2).unwrap()
                };
                out.fill(0);
                enc.decode_into(&mut out, 2).unwrap();
                assert_eq!(out, src, "failed roundtrip for {:?} bs={bs}", codec);
            }
        }
        let z3_4k = Encoded::compress(&src, Codec::Zstd { level: 3 }, 4096, 2).unwrap();
        let zd3_4k =
            Encoded::compress_with_dict(&src, Codec::ZstdDict { level: 3 }, &dict, 4096, 2)
                .unwrap();
        assert!(
            zd3_4k.encoded_size() < z3_4k.encoded_size(),
            "zd3_4k={} (dict={}) vs z3_4k={}",
            zd3_4k.encoded_size(),
            zd3_4k.dict_size(),
            z3_4k.encoded_size()
        );
        let z3_64k = Encoded::compress(&src, Codec::Zstd { level: 3 }, 65536, 2).unwrap();
        let zw19 = Encoded::compress(&src, Codec::ZstdWholeLong { level: 19 }, 65536, 2).unwrap();
        assert!(zw19.encoded_size() < z3_64k.encoded_size());
    }
}
