use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{block, checksum::crc32c, frame, simd::Engine, DecodeOptions, Error, Frame, Result};

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<bool> {
    let mut read_total = 0usize;
    while read_total < buf.len() {
        match reader.read(&mut buf[read_total..]) {
            Ok(0) => {
                if read_total == 0 {
                    return Ok(false);
                }
                return Err(Error::Truncated);
            }
            Ok(n) => read_total += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Backend(e.to_string())),
        }
    }
    Ok(true)
}

fn read_exact_full<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<()> {
    let mut read_total = 0usize;
    while read_total < buf.len() {
        match reader.read(&mut buf[read_total..]) {
            Ok(0) => return Err(Error::Truncated),
            Ok(n) => read_total += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::Backend(e.to_string())),
        }
    }
    Ok(())
}

/// Zero-reallocation streaming reader yielding one validated `VXG1` frame slice at a time
/// from any [`std::io::Read`] source with `O(single_frame)` bounded memory.
pub struct FrameReader<R: Read> {
    reader: R,
    max_output: usize,
    frame_buf: Vec<u8>,
    frames_read: u64,
    total_raw: u64,
    total_compressed: u64,
    eof: bool,
}

impl<R: Read> FrameReader<R> {
    /// Creates a new [`FrameReader`] enforcing `max_output` total uncompressed bytes across the stream.
    /// Pass `usize::MAX` for unbounded multi-GB streams.
    pub fn new(reader: R, max_output: usize) -> Self {
        Self {
            reader,
            max_output,
            frame_buf: Vec::new(),
            frames_read: 0,
            total_raw: 0,
            total_compressed: 0,
            eof: false,
        }
    }

    #[inline]
    pub fn frames_read(&self) -> u64 {
        self.frames_read
    }

    #[inline]
    pub fn total_raw_bytes(&self) -> u64 {
        self.total_raw
    }

    #[inline]
    pub fn total_compressed_bytes(&self) -> u64 {
        self.total_compressed
    }

    /// Reads and header/directory-validates the next `VXG1` frame from the stream, returning
    /// `Ok(None)` cleanly at `EOF` after at least one frame has been read.
    /// Returns `Err(Error::Truncated)` if the stream is completely empty (`0` bytes).
    pub fn next_frame(&mut self) -> Result<Option<&[u8]>> {
        if self.eof {
            return Ok(None);
        }
        let mut hdr = [0u8; frame::HEADER_SIZE];
        if !read_exact_or_eof(&mut self.reader, &mut hdr)? {
            if self.frames_read == 0 {
                return Err(Error::Truncated);
            }
            self.eof = true;
            return Ok(None);
        }
        let rem_budget = (self.max_output as u64)
            .saturating_sub(self.total_raw)
            .min(usize::MAX as u64) as usize;
        let (_raw_size, _block_size, _count, _version, _csum, dir_end) =
            frame::parse_header_only(&hdr, rem_budget)?;
        self.frame_buf.resize(dir_end, 0);
        self.frame_buf[..frame::HEADER_SIZE].copy_from_slice(&hdr);
        read_exact_full(
            &mut self.reader,
            &mut self.frame_buf[frame::HEADER_SIZE..dir_end],
        )?;

        let (frame_len, raw_size) =
            frame::parse_header_and_dir(&self.frame_buf[..dir_end], rem_budget)?;
        self.frame_buf.resize(frame_len, 0);
        read_exact_full(&mut self.reader, &mut self.frame_buf[dir_end..frame_len])?;

        self.frames_read += 1;
        self.total_raw += raw_size as u64;
        self.total_compressed += frame_len as u64;
        Ok(Some(&self.frame_buf[..frame_len]))
    }
}

fn verify_single_parsed_frame(
    parsed: &Frame<'_>,
    options: &DecodeOptions,
    engine: Engine,
) -> Result<()> {
    let blocks = &parsed.blocks[..];
    if blocks.is_empty() {
        return Ok(());
    }
    let verify_csum = parsed.checksummed && options.verify_checksum;
    let short_distance = parsed.version >= 2;
    let max_blk = parsed.block_size;
    let dict_ref = parsed.dict.as_slice();
    let workers = options.threads.min(blocks.len());

    if workers <= 1 {
        let mut scratch = vec![0u8; max_blk];
        for (idx, b) in blocks.iter().enumerate() {
            if b.info.kind == block::RAW {
                if b.payload.len() != b.info.raw_size {
                    return Err(Error::InvalidFrame("raw block size mismatch"));
                }
                if verify_csum && crc32c(b.payload) != b.info.checksum {
                    return Err(Error::Checksum { block: idx });
                }
            } else {
                let dst = &mut scratch[..b.info.raw_size];
                block::decode_with_dict(
                    b.payload,
                    b.info.kind,
                    dst,
                    engine,
                    short_distance,
                    dict_ref,
                )?;
                if verify_csum && crc32c(dst) != b.info.checksum {
                    return Err(Error::Checksum { block: idx });
                }
            }
        }
        return Ok(());
    }

    let next_idx = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let max_batch = (blocks.len() / (workers * 8)).clamp(1, 4);
    let tail_cutoff = blocks.len().saturating_sub(workers * 4);

    crate::pool::parallel_try_run(workers, |_| {
        let mut scratch = vec![0u8; max_blk];
        loop {
            if aborted.load(Ordering::Relaxed) {
                return Ok(());
            }
            let step = if next_idx.load(Ordering::Relaxed) < tail_cutoff {
                max_batch
            } else {
                1
            };
            let start = next_idx.fetch_add(step, Ordering::Relaxed);
            if start >= blocks.len() {
                return Ok(());
            }
            let end = (start + step).min(blocks.len());
            for (local_off, b) in blocks[start..end].iter().enumerate() {
                let idx = start + local_off;
                if b.info.kind == block::RAW {
                    if verify_csum && crc32c(b.payload) != b.info.checksum {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(Error::Checksum { block: idx });
                    }
                } else {
                    let dst = &mut scratch[..b.info.raw_size];
                    if let Err(e) = block::decode_with_dict(
                        b.payload,
                        b.info.kind,
                        dst,
                        engine,
                        short_distance,
                        dict_ref,
                    ) {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(e);
                    }
                    if verify_csum && crc32c(dst) != b.info.checksum {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(Error::Checksum { block: idx });
                    }
                }
            }
        }
    })
}

/// Validates and fully test-decodes one or more concatenated `VXG1` frames in `src`
/// (including per-block CRC32C verification when enabled) using only `O(threads * block_size)`
/// L2-cache-resident scratch memory instead of allocating the full uncompressed output.
/// Returns total verified uncompressed bytes.
pub fn verify_frame(src: &[u8], options: &DecodeOptions) -> Result<u64> {
    options.validate()?;
    let engine = Engine::new(options.simd)?;
    let frames = Frame::parse_all(src, options.max_output)?;
    let mut total_raw = 0u64;
    for parsed in &frames {
        verify_single_parsed_frame(parsed, options, engine)?;
        total_raw += parsed.raw_size() as u64;
    }
    Ok(total_raw)
}

/// Streams and fully verifies all `VXG1` frames from `reader` with bounded memory.
/// Returns `(verified_raw_bytes, compressed_bytes_read)`.
pub fn verify_stream<R: Read>(reader: R, options: &DecodeOptions) -> Result<(u64, u64)> {
    options.validate()?;
    let engine = Engine::new(options.simd)?;
    let mut fr = FrameReader::new(reader, options.max_output);
    while let Some(f_bytes) = fr.next_frame()? {
        let parsed = Frame::parse(f_bytes, usize::MAX)?;
        verify_single_parsed_frame(&parsed, options, engine)?;
    }
    Ok((fr.total_raw_bytes(), fr.total_compressed_bytes()))
}
