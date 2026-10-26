//! VXG v0.1: an experimental generic lossless block codec.
//!
//! ```
//! let input = b"abcabcabcabcabcabcabcabcabcabc";
//! let frame = vxg::compress(input, &vxg::Options::default()).unwrap();
//! let output = vxg::decompress(&frame, &vxg::DecodeOptions::default()).unwrap();
//! assert_eq!(output, input);
//! ```
//! See docs/PROOFS.md for the invariants and their exact scope. Performance is measured,
//! not proved. The format is experimental and is not LZ4/Zstandard-compatible.
#[cfg(feature = "bench")]
pub mod analysis;
#[cfg(feature = "bench")]
pub mod baselines;
pub mod columns;
#[cfg(feature = "bench")]
pub mod groups;
mod block;
pub mod checksum;
mod error;
pub mod ffi;
mod frame;
pub mod gpu;
#[cfg(feature = "bench")]
pub mod packing;
#[cfg(kani)]
mod proofs;
pub mod simd;
pub mod stream;
pub use block::{WorkStats, COLUMN as BLOCK_COLUMN, LZ as BLOCK_LZ, RAW as BLOCK_RAW};
pub use error::{Error, Result};
pub use frame::{max_compressed_size, BlockInfo, Frame, ValidatedFrame};
pub use simd::Simd;
pub use stream::{compress_stream, decompress_stream, StreamDecoder, StreamEncoder};

#[derive(Clone, Debug)]
pub struct Options {
    pub block_size: usize,
    pub threads: usize,
    pub checksum: bool,
    /// Minimum failed-probe step. Valid range 1..64. No change to losslessness.
    pub acceleration: usize,
    pub simd: Simd,
    /// Emit legacy version-2 LZ-only frames instead of the unified adaptive
    /// release pipeline (v4: structured-column / stride / fast-LZ router with
    /// compact 8-byte directory). Off by default; v1..v3 frames remain readable.
    pub legacy_v2: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            block_size: 65536,
            threads: 1,
            checksum: true,
            acceleration: 1,
            simd: Simd::Auto,
            legacy_v2: false,
        }
    }
}
impl Options {
    pub(crate) fn validate(&self) -> Result<()> {
        frame::check_block_size(self.block_size)?;
        if !(1..=256).contains(&self.threads) {
            return Err(Error::InvalidOption("threads must be 1..256"));
        }
        if !(1..=64).contains(&self.acceleration) {
            return Err(Error::InvalidOption("acceleration must be 1..64"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct DecodeOptions {
    pub threads: usize,
    pub simd: Simd,
    pub max_output: usize,
    pub verify_checksum: bool,
}
impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            threads: 1,
            simd: Simd::Auto,
            max_output: 1 << 30,
            verify_checksum: true,
        }
    }
}
impl DecodeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=256).contains(&self.threads) {
            return Err(Error::InvalidOption("threads must be 1..256"));
        }
        Ok(())
    }
}
thread_local! {
    static ENCODE_SCRATCH_POOL: std::cell::RefCell<block::Scratch> =
        std::cell::RefCell::new(block::Scratch::new());
}

pub fn compress(input: &[u8], options: &Options) -> Result<Vec<u8>> {
    Ok(compress_inner::<false>(input, options)?.0)
}
pub fn compress_with_stats(input: &[u8], options: &Options) -> Result<(Vec<u8>, WorkStats)> {
    compress_inner::<true>(input, options)
}
/// Compresses `input` directly into caller-supplied `out[..written]`, avoiding per-call output
/// buffer allocation on the single-threaded path by reusing a thread-local scratch arena.
pub fn compress_into(input: &[u8], out: &mut [u8], options: &Options) -> Result<usize> {
    options.validate()?;
    max_compressed_size(input.len(), options.block_size)?;
    let engine = simd::Engine::new(options.simd)?;
    let count = input.len().div_ceil(options.block_size);
    let workers = options.threads.min(count.max(1));
    let version = if options.legacy_v2 {
        frame::VERSION
    } else {
        frame::VERSION_RELEASE
    };
    if workers == 1 {
        let esize = frame::entry_size(version);
        let adaptive = !options.legacy_v2;
        let table_end = frame::HEADER_SIZE
            .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
        return ENCODE_SCRATCH_POOL.with(|cell| -> Result<usize> {
            let mut scratch = cell.borrow_mut();
            scratch.payloads_mut().clear();
            let payload_reserve = if adaptive {
                (input.len() / 12).max(options.block_size.min(input.len()))
            } else {
                (input.len() / 4).max(options.block_size.min(input.len()))
            };
            scratch.reserve_payloads_for(table_end.saturating_add(payload_reserve));
            scratch.payloads_mut().resize(table_end, 0);
            let mut raw_offset = 0usize;
            for (index, chunk) in input.chunks(options.block_size).enumerate() {
                let (kind, offset, len) = if adaptive {
                    match columns::encode_grammar_block_into(chunk, scratch.payloads_mut())? {
                        Some((packet_offset, packet_len)) => {
                            (block::COLUMN, packet_offset, packet_len)
                        }
                        None if (512..=65536).contains(&chunk.len()) => {
                            let payloads = scratch.payloads_mut();
                            let p_offset = payloads.len();
                            payloads.extend_from_slice(chunk);
                            (block::RAW, p_offset, chunk.len())
                        }
                        None => {
                            let enc = block::encode::<false>(
                                chunk,
                                &mut scratch,
                                engine,
                                options.acceleration,
                            );
                            (enc.kind, enc.offset, enc.len)
                        }
                    }
                } else {
                    let enc =
                        block::encode::<false>(chunk, &mut scratch, engine, options.acceleration);
                    (enc.kind, enc.offset, enc.len)
                };
                let csum = if options.checksum {
                    checksum::crc32c(chunk)
                } else {
                    0
                };
                let p = frame::HEADER_SIZE + index * esize;
                frame::write_entry(
                    scratch.payloads_mut(),
                    p,
                    version,
                    raw_offset,
                    offset,
                    chunk.len(),
                    len,
                    csum,
                    kind,
                );
                raw_offset += chunk.len();
            }
            let payloads = scratch.payloads_mut();
            let written = payloads.len();
            if out.len() < written {
                return Err(Error::OutputSize {
                    expected: written,
                    actual: out.len(),
                });
            }
            payloads[0..4].copy_from_slice(frame::MAGIC);
            payloads[4..6].copy_from_slice(&version.to_le_bytes());
            payloads[6..8].copy_from_slice(&(u16::from(options.checksum)).to_le_bytes());
            payloads[8..16].copy_from_slice(&(input.len() as u64).to_le_bytes());
            payloads[16..20].copy_from_slice(&(options.block_size as u32).to_le_bytes());
            payloads[20..24].copy_from_slice(&(count as u32).to_le_bytes());
            payloads[24..32].copy_from_slice(&0u64.to_le_bytes());
            out[..written].copy_from_slice(payloads);
            Ok(written)
        });
    }
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let worker_outputs = std::thread::scope(|scope| -> Result<Vec<_>> {
        let next_ref = &next_idx;
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                encode_worker::<false>(input, options, engine, worker, workers, next_ref)
            }));
        }
        let mut result = Vec::with_capacity(workers);
        for handle in handles {
            result.push(handle.join().map_err(|_| Error::WorkerPanic)??);
        }
        Ok(result)
    })?;
    frame::assemble_workers_into(
        worker_outputs,
        out,
        input.len(),
        options.block_size,
        options.checksum,
        version,
    )
}
fn compress_inner<const STATS: bool>(
    input: &[u8],
    options: &Options,
) -> Result<(Vec<u8>, WorkStats)> {
    options.validate()?;
    max_compressed_size(input.len(), options.block_size)?;
    let engine = simd::Engine::new(options.simd)?;
    let count = input.len().div_ceil(options.block_size);
    let workers = options.threads.min(count.max(1));
    let version = if options.legacy_v2 {
        frame::VERSION
    } else {
        frame::VERSION_RELEASE
    };
    if workers == 1 {
        return encode_single_worker::<STATS>(input, options, engine, count, version);
    }
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let worker_outputs = std::thread::scope(|scope| -> Result<Vec<_>> {
        let next_ref = &next_idx;
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            handles.push(scope.spawn(move || {
                encode_worker::<STATS>(input, options, engine, worker, workers, next_ref)
            }));
        }
        let mut result = Vec::with_capacity(workers);
        for handle in handles {
            result.push(handle.join().map_err(|_| Error::WorkerPanic)??);
        }
        Ok(result)
    })?;
    let mut stats = WorkStats::default();
    for output in &worker_outputs {
        stats += output.stats;
    }
    Ok((
        frame::assemble_workers(
            worker_outputs,
            input.len(),
            options.block_size,
            options.checksum,
            version,
        )?,
        stats,
    ))
}
fn encode_single_worker<const STATS: bool>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    version: u16,
) -> Result<(Vec<u8>, WorkStats)> {
    let esize = frame::entry_size(version);
    let adaptive = !opt.legacy_v2;
    let table_end = frame::HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let mut scratch = block::Scratch::new();
    let payload_reserve = if adaptive {
        (input.len() / 12).max(opt.block_size.min(input.len()))
    } else {
        (input.len() / 4).max(opt.block_size.min(input.len()))
    };
    scratch.reserve_payloads_for(table_end.saturating_add(payload_reserve));
    scratch.payloads_mut().resize(table_end, 0);
    let mut stats = WorkStats::default();
    let mut raw_offset = 0usize;
    for (index, chunk) in input.chunks(opt.block_size).enumerate() {
        if index > 0 {
            let payloads = scratch.payloads_mut();
            let cur_len = payloads.len();
            let written = cur_len - table_end;
            if cur_len + chunk.len() + 1024 > payloads.capacity()
                || (index == 1
                    && table_end + written * count + opt.block_size > payloads.capacity())
            {
                let rem_raw = input.len() - raw_offset;
                let est_rem = if written * 2 >= raw_offset {
                    rem_raw + (count - index) * 64
                } else {
                    ((rem_raw as u64 * written as u64) / (raw_offset as u64).max(1)) as usize
                        + opt.block_size * 2
                };
                payloads.reserve(est_rem.max(chunk.len() + 1024));
            }
        }
        let (kind, offset, len) = if adaptive {
            match columns::encode_grammar_block_into(chunk, scratch.payloads_mut())? {
                Some((packet_offset, packet_len)) => (block::COLUMN, packet_offset, packet_len),
                None if (512..=65536).contains(&chunk.len()) => {
                    let payloads = scratch.payloads_mut();
                    let p_offset = payloads.len();
                    payloads.extend_from_slice(chunk);
                    (block::RAW, p_offset, chunk.len())
                }
                None => {
                    let enc =
                        block::encode::<STATS>(chunk, &mut scratch, engine, opt.acceleration);
                    if STATS {
                        stats += enc.stats;
                    }
                    (enc.kind, enc.offset, enc.len)
                }
            }
        } else {
            let enc = block::encode::<STATS>(chunk, &mut scratch, engine, opt.acceleration);
            if STATS {
                stats += enc.stats;
            }
            (enc.kind, enc.offset, enc.len)
        };
        let csum = if opt.checksum {
            checksum::crc32c(chunk)
        } else {
            0
        };
        let p = frame::HEADER_SIZE + index * esize;
        frame::write_entry(
            scratch.payloads_mut(),
            p,
            version,
            raw_offset,
            offset,
            chunk.len(),
            len,
            csum,
            kind,
        );
        raw_offset += chunk.len();
    }
    let mut out = scratch.finish();
    out[0..4].copy_from_slice(frame::MAGIC);
    out[4..6].copy_from_slice(&version.to_le_bytes());
    out[6..8].copy_from_slice(&(u16::from(opt.checksum)).to_le_bytes());
    out[8..16].copy_from_slice(&(input.len() as u64).to_le_bytes());
    out[16..20].copy_from_slice(&(opt.block_size as u32).to_le_bytes());
    out[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    out[24..32].copy_from_slice(&0u64.to_le_bytes());
    Ok((out, stats))
}
fn encode_worker<const STATS: bool>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
    worker: usize,
    workers: usize,
    next_idx: &std::sync::atomic::AtomicUsize,
) -> Result<frame::WorkerOutput> {
    let mut scratch = block::Scratch::new();
    let count = input.len().div_ceil(opt.block_size);
    let capacity = count.saturating_sub(worker).div_ceil(workers);
    let worker_raw_est = capacity.saturating_mul(opt.block_size).min(input.len());
    let adaptive = !opt.legacy_v2;
    let payload_reserve = if adaptive {
        (worker_raw_est / 16).max(opt.block_size.min(input.len()))
    } else {
        (worker_raw_est / 4).max(opt.block_size.min(input.len()))
    };
    scratch.reserve_payloads_for(payload_reserve);
    let mut parts = Vec::with_capacity(capacity);
    let mut stats = WorkStats::default();
    loop {
        let index = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if index >= count {
            break;
        }
        let start_byte = index * opt.block_size;
        let end_byte = (start_byte + opt.block_size).min(input.len());
        let chunk = &input[start_byte..end_byte];
        let (kind, offset, len) = if adaptive {
            match columns::encode_grammar_block_into(chunk, scratch.payloads_mut())? {
                Some((packet_offset, packet_len)) => (block::COLUMN, packet_offset, packet_len),
                None if (512..=65536).contains(&chunk.len()) => {
                    let payloads = scratch.payloads_mut();
                    let raw_offset = payloads.len();
                    payloads.extend_from_slice(chunk);
                    (block::RAW, raw_offset, chunk.len())
                }
                None => {
                    let enc = block::encode::<STATS>(chunk, &mut scratch, engine, opt.acceleration);
                    if STATS {
                        stats += enc.stats;
                    }
                    (enc.kind, enc.offset, enc.len)
                }
            }
        } else {
            let enc = block::encode::<STATS>(chunk, &mut scratch, engine, opt.acceleration);
            if STATS {
                stats += enc.stats;
            }
            (enc.kind, enc.offset, enc.len)
        };
        parts.push(frame::ArenaPart {
            index,
            offset,
            len,
            kind,
            raw_size: chunk.len(),
            checksum: if opt.checksum {
                checksum::crc32c(chunk)
            } else {
                0
            },
        });
    }
    Ok(frame::WorkerOutput {
        arena: scratch.finish(),
        parts,
        stats,
    })
}
pub fn decompress(src: &[u8], options: &DecodeOptions) -> Result<Vec<u8>> {
    options.validate()?;
    let mut v4_offsets = [0u32; frame::MAX_STACK_DIR_BLOCKS + 1];
    let (raw_size, block_size, count, version, checksummed) =
        match frame::validate_header_and_directory_with_offsets(
            src,
            options.max_output,
            (options.threads > 1).then_some(&mut v4_offsets),
        ) {
            Ok(v) => v,
            Err(Error::InvalidFrame("trailing bytes or incorrect total")) => {
                return stream::decompress_concatenated(src, options);
            }
            Err(e) => return Err(e),
        };
    if options.threads == 1
        || version < frame::VERSION_V4
        || (count <= frame::MAX_STACK_DIR_BLOCKS && src.len() <= u32::MAX as usize)
    {
        let mut out = Vec::new();
        out.try_reserve_exact(raw_size)
            .map_err(|_| Error::SizeOverflow)?;
        out.resize(raw_size, 0);
        if options.threads == 1 {
            frame::decode_validated_dir_single_thread(
                src,
                &mut out,
                raw_size,
                block_size,
                count,
                version,
                checksummed,
                options,
            )?;
        } else {
            frame::decode_validated_dir_multi_thread(
                src,
                &mut out,
                raw_size,
                block_size,
                count,
                version,
                checksummed,
                &v4_offsets,
                options,
            )?;
        }
        return Ok(out);
    }
    let frame = Frame::parse(src, options.max_output)?;
    let mut out = Vec::new();
    out.try_reserve_exact(frame.raw_size())
        .map_err(|_| Error::SizeOverflow)?;
    out.resize(frame.raw_size(), 0);
    frame.decode_into(&mut out, options)?;
    Ok(out)
}
pub fn decompress_into(src: &[u8], out: &mut [u8], options: &DecodeOptions) -> Result<()> {
    options.validate()?;
    let mut v4_offsets = [0u32; frame::MAX_STACK_DIR_BLOCKS + 1];
    let (raw_size, block_size, count, version, checksummed) =
        match frame::validate_header_and_directory_with_offsets(
            src,
            options.max_output,
            (options.threads > 1).then_some(&mut v4_offsets),
        ) {
            Ok(v) => v,
            Err(Error::InvalidFrame("trailing bytes or incorrect total")) => {
                return stream::decompress_concatenated_into(src, out, options);
            }
            Err(e) => return Err(e),
        };
    if options.threads == 1 {
        return frame::decode_validated_dir_single_thread(
            src,
            out,
            raw_size,
            block_size,
            count,
            version,
            checksummed,
            options,
        );
    }
    if version < frame::VERSION_V4
        || (count <= frame::MAX_STACK_DIR_BLOCKS && src.len() <= u32::MAX as usize)
    {
        return frame::decode_validated_dir_multi_thread(
            src,
            out,
            raw_size,
            block_size,
            count,
            version,
            checksummed,
            &v4_offsets,
            options,
        );
    }
    Frame::parse(src, options.max_output)?.decode_into(out, options)
}
/// Zero-allocation random-access decompression of a single block `block_index`
/// directly into `out[..block_raw_size]`.
pub fn extract_block_into(
    src: &[u8],
    block_index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    frame::extract_block_validated_into(src, block_index, out, options)
}
