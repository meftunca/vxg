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
#[cfg(kani)]
mod proofs;
mod error;
mod block;
mod frame;
pub mod checksum;
pub mod simd;
pub mod gpu;
#[cfg(feature="bench")]
pub mod baselines;
pub use error::{Error, Result};
pub use frame::{Frame, BlockInfo, max_compressed_size};
pub use block::WorkStats;
pub use simd::Simd;

#[derive(Clone, Debug)]
pub struct Options {
    pub block_size: usize,
    pub threads: usize,
    pub checksum: bool,
    /// Minimum failed-probe step. Valid range 1..64. No change to losslessness.
    pub acceleration: usize,
    pub simd: Simd,
}
impl Default for Options {
    fn default() -> Self { Self { block_size: 65536, threads: 1, checksum: true, acceleration: 1, simd: Simd::Auto } }
}
impl Options {
    pub(crate) fn validate(&self) -> Result<()> {
        frame::check_block_size(self.block_size)?;
        if !(1..=256).contains(&self.threads) { return Err(Error::InvalidOption("threads must be 1..256")); }
        if !(1..=64).contains(&self.acceleration) { return Err(Error::InvalidOption("acceleration must be 1..64")); }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct DecodeOptions { pub threads: usize, pub simd: Simd, pub max_output: usize, pub verify_checksum: bool }
impl Default for DecodeOptions {
    fn default() -> Self { Self { threads: 1, simd: Simd::Auto, max_output: 1<<30, verify_checksum: true } }
}
impl DecodeOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=256).contains(&self.threads) { return Err(Error::InvalidOption("threads must be 1..256")); }
        Ok(())
    }
}
pub fn compress(input: &[u8], options: &Options) -> Result<Vec<u8>> {
    Ok(compress_inner::<false>(input, options)?.0)
}
pub fn compress_with_stats(input: &[u8], options: &Options) -> Result<(Vec<u8>, WorkStats)> {
    compress_inner::<true>(input, options)
}
fn compress_inner<const STATS: bool>(input: &[u8], options: &Options) -> Result<(Vec<u8>, WorkStats)> {
    options.validate()?;
    max_compressed_size(input.len(), options.block_size)?;
    let engine = simd::Engine::new(options.simd)?;
    let count = input.len().div_ceil(options.block_size);
    let workers = options.threads.min(count.max(1));
    let worker_outputs = if workers == 1 {
        vec![encode_worker::<STATS>(input, options, engine, 0, 1)]
    } else {
        std::thread::scope(|scope| -> Result<Vec<_>> {
            let mut handles = Vec::with_capacity(workers);
            for worker in 0..workers { handles.push(scope.spawn(move || encode_worker::<STATS>(input, options, engine, worker, workers))); }
            let mut result = Vec::with_capacity(workers);
            for handle in handles { result.push(handle.join().map_err(|_| Error::WorkerPanic)?); }
            Ok(result)
        })?
    };
    let mut stats = WorkStats::default();
    for output in &worker_outputs { stats += output.stats; }
    Ok((frame::assemble_workers(worker_outputs, input.len(), options.block_size, options.checksum)?, stats))
}
fn encode_worker<const STATS: bool>(input: &[u8], opt: &Options, engine: simd::Engine, worker: usize, workers: usize) -> frame::WorkerOutput {
    let mut scratch = block::Scratch::new();
    let count = input.len().div_ceil(opt.block_size);
    let capacity = count.saturating_sub(worker).div_ceil(workers);
    scratch.reserve_payloads_for(opt.block_size.min(input.len()));
    let mut parts = Vec::with_capacity(capacity);
    let mut stats = WorkStats::default();
    for (index, chunk) in input.chunks(opt.block_size).enumerate().skip(worker).step_by(workers) {
        let enc = block::encode::<STATS>(chunk, &mut scratch, engine, opt.acceleration);
        if STATS { stats += enc.stats; }
        parts.push(frame::ArenaPart { index, offset: enc.offset, len: enc.len, kind: enc.kind,
            raw_size: chunk.len(), checksum: if opt.checksum { checksum::crc32c(chunk) } else { 0 } });
    }
    frame::WorkerOutput { arena: scratch.finish(), parts, stats }
}
pub fn decompress(src: &[u8], options: &DecodeOptions) -> Result<Vec<u8>> {
    options.validate()?;
    let frame = Frame::parse(src, options.max_output)?;
    let mut out = Vec::new();
    out.try_reserve_exact(frame.raw_size()).map_err(|_| Error::SizeOverflow)?;
    out.resize(frame.raw_size(), 0);
    frame.decode_into(&mut out, options)?;
    Ok(out)
}
pub fn decompress_into(src: &[u8], out: &mut [u8], options: &DecodeOptions) -> Result<()> {
    Frame::parse(src, options.max_output)?.decode_into(out, options)
}
