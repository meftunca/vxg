//! Bounded-memory streaming encoder (`StreamEncoder<W>`) and decoder (`StreamDecoder<R>`),
//! plus concatenated multi-frame decompression helpers.
//!
//! `StreamEncoder` buffers up to `chunk_size` raw bytes (defaulting to a bounded multiple of
//! `options.block_size`) and flushes each chunk as an independent, self-contained `VXG1` frame
//! using preallocated buffers. `StreamDecoder` reads and validates one `VXG1` frame at a time
//! from any `std::io::Read` source without buffering the entire stream in memory.

use std::io::{self, Read, Write};

use crate::{
    compress_into, decompress_into, frame, max_compressed_size, DecodeOptions, Error, Options,
    Result,
};

fn to_io_err(err: Error) -> io::Error {
    let kind = match err {
        Error::Truncated => io::ErrorKind::UnexpectedEof,
        Error::InvalidOption(_) => io::ErrorKind::InvalidInput,
        _ => io::ErrorKind::InvalidData,
    };
    io::Error::new(kind, err)
}

fn from_io_err(err: io::Error) -> Error {
    if let Some(inner) = err.get_ref() {
        if let Some(vxg_err) = inner.downcast_ref::<Error>() {
            return vxg_err.clone();
        }
    }
    match err.kind() {
        io::ErrorKind::UnexpectedEof => Error::Truncated,
        _ => Error::Backend(err.to_string()),
    }
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let mut read_total = 0usize;
    while read_total < buf.len() {
        match reader.read(&mut buf[read_total..]) {
            Ok(0) => {
                if read_total == 0 {
                    return Ok(false);
                }
                return Err(to_io_err(Error::Truncated));
            }
            Ok(n) => read_total += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

/// Bounded-memory streaming VXG encoder implementing [`std::io::Write`].
pub struct StreamEncoder<W: Write> {
    writer: Option<W>,
    options: Options,
    chunk_size: usize,
    raw_buf: Vec<u8>,
    enc_buf: Vec<u8>,
    v6_builder: Option<frame::v6::V6StreamBuilder>,
    frames_emitted: u64,
    total_in: u64,
    total_out: u64,
}

impl<W: Write> StreamEncoder<W> {
    /// Creates a new streaming encoder with a bounded default chunk size
    /// (`4..16 MiB` scaled by `options.threads` for `v4`/`v5`; a small `1..4`-block micro-batch
    /// when `options.col_group` is enabled so blocks are encoded immediately into per-column buffers).
    pub fn new(writer: W, options: Options) -> Result<Self> {
        if options.col_group && !options.legacy_v2 {
            let micro_blocks = options.threads.clamp(1, 8);
            let chunk_size = options
                .block_size
                .saturating_mul(micro_blocks)
                .clamp(options.block_size, 4 << 20);
            return Self::with_chunk_size(writer, options, chunk_size);
        }
        let blocks = (options.threads.clamp(1, 32) * 16).clamp(64, 256);
        let chunk_size = options
            .block_size
            .saturating_mul(blocks)
            .clamp(options.block_size, 16 << 20);
        Self::with_chunk_size(writer, options, chunk_size)
    }

    /// Creates a new streaming encoder flushing a self-contained VXG frame every `chunk_size` bytes.
    pub fn with_chunk_size(writer: W, options: Options, chunk_size: usize) -> Result<Self> {
        options.validate()?;
        let eff_chunk = if options.col_group && !options.legacy_v2 {
            chunk_size.min(4 << 20)
        } else {
            chunk_size
        };
        let blocks = eff_chunk
            .max(options.block_size)
            .div_ceil(options.block_size);
        let aligned_chunk = blocks
            .checked_mul(options.block_size)
            .ok_or(Error::SizeOverflow)?;
        if aligned_chunk > (64 << 20) {
            return Err(Error::InvalidOption("stream chunk_size must be <= 64 MiB"));
        }
        let _max_enc = max_compressed_size(aligned_chunk, options.block_size)?;
        let v6_builder =
            (options.col_group && !options.legacy_v2).then(frame::v6::V6StreamBuilder::new);
        Ok(Self {
            writer: Some(writer),
            options,
            chunk_size: aligned_chunk,
            raw_buf: Vec::new(),
            enc_buf: Vec::new(),
            v6_builder,
            frames_emitted: 0,
            total_in: 0,
            total_out: 0,
        })
    }

    /// Total uncompressed bytes written into this encoder so far.
    pub fn total_in(&self) -> u64 {
        self.total_in
    }

    /// Total compressed bytes emitted to the underlying writer so far.
    pub fn total_out(&self) -> u64 {
        self.total_out
    }

    /// Current total buffer footprint (`raw_buf + enc_buf + active v6 group streams`).
    pub fn buffered_bytes(&self) -> usize {
        self.raw_buf.capacity()
            + self.enc_buf.capacity()
            + self
                .v6_builder
                .as_ref()
                .map_or(0, frame::v6::V6StreamBuilder::buffered_compressed_bytes)
    }

    fn flush_slice(&mut self, len: usize) -> io::Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| io::Error::other("StreamEncoder already finished"))?;
        let bound = max_compressed_size(len, self.options.block_size).map_err(to_io_err)?;
        if self.enc_buf.len() < bound {
            self.enc_buf.resize(bound, 0);
        }
        let written = compress_into(&self.raw_buf[..len], &mut self.enc_buf[..bound], &self.options)
            .map_err(to_io_err)?;
        writer.write_all(&self.enc_buf[..written])?;
        self.total_in += len as u64;
        self.total_out += written as u64;
        self.frames_emitted += 1;
        Ok(())
    }

    fn flush_v6_step(&mut self, is_final: bool) -> io::Result<()> {
        while !self.raw_buf.is_empty() {
            let len = self.raw_buf.len();
            let eff_bs = self.v6_builder.as_mut().unwrap().resolve_eff_bs(
                &self.raw_buf,
                &self.options,
                is_final,
            );
            let target_micro = (eff_bs * self.options.threads.clamp(1, 8)).clamp(eff_bs, 4 << 20);
            self.chunk_size = target_micro;
            if !is_final && len < target_micro {
                return Ok(());
            }
            let take_raw = len.min(target_micro);
            let encode_len = if !is_final && self.options.record_aligned && take_raw > eff_bs / 2 {
                let back = (eff_bs / 4).min(32_768).min(take_raw);
                self.raw_buf[take_raw - back..take_raw]
                    .iter()
                    .rposition(|&c| c == b'\n')
                    .map_or(take_raw, |rel| take_raw - back + rel + 1)
            } else {
                take_raw
            };
            let skip_dict = self.v6_builder.as_ref().is_some_and(|b| b.saw_columnar_no_dict);
            let mut micro_opts = self
                .v6_builder
                .as_ref()
                .and_then(|b| b.eff_opts.clone())
                .unwrap_or_else(|| self.options.clone());
            micro_opts.col_group = false;
            let bound = max_compressed_size(encode_len, self.options.block_size).map_err(to_io_err)?;
            if self.enc_buf.len() < bound {
                self.enc_buf.resize(bound, 0);
            }
            let written = crate::compress_into_micro(&self.raw_buf[..encode_len], &mut self.enc_buf[..bound], &micro_opts, skip_dict)
                .map_err(to_io_err)?;
            let writer = self.writer.as_mut().ok_or_else(|| io::Error::other("StreamEncoder already finished"))?;
            let v6_out = self
                .v6_builder
                .as_mut()
                .unwrap()
                .ingest_v5_micro_frame(&self.enc_buf[..written], writer, &self.options)
                .map_err(to_io_err)?;
            self.total_in += encode_len as u64;
            self.total_out += v6_out;
            self.frames_emitted += u64::from(v6_out > 0);
            self.raw_buf.drain(..encode_len);
        }
        if is_final {
            let writer = self.writer.as_mut().ok_or_else(|| io::Error::other("StreamEncoder already finished"))?;
            let v6_out = self
                .v6_builder
                .as_mut()
                .unwrap()
                .flush_group(writer, &self.options)
                .map_err(to_io_err)?;
            self.total_out += v6_out;
            self.frames_emitted += u64::from(v6_out > 0);
            if self.frames_emitted == 0 {
                self.flush_slice(0)?;
            }
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> io::Result<()> {
        if self.v6_builder.is_some() {
            return self.flush_v6_step(false);
        }
        let len = self.raw_buf.len();
        self.flush_slice(len)?;
        self.raw_buf.clear();
        Ok(())
    }

    /// Flushes any remaining buffered bytes (or emits an empty VXG frame if 0 bytes were written)
    /// and returns the underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        if self.v6_builder.is_some() {
            self.flush_v6_step(true)?;
        } else if !self.raw_buf.is_empty() || self.frames_emitted == 0 {
            self.flush_chunk()?;
        }
        let mut writer = self.writer.take().unwrap();
        writer.flush()?;
        Ok(writer)
    }
}

impl<W: Write> Write for StreamEncoder<W> {
    fn write(&mut self, mut buf: &[u8]) -> io::Result<usize> {
        let total = buf.len();
        while !buf.is_empty() {
            let space = self.chunk_size.saturating_sub(self.raw_buf.len()).max(1);
            let take = buf.len().min(space);
            self.raw_buf.extend_from_slice(&buf[..take]);
            buf = &buf[take..];
            if self.raw_buf.len() >= self.chunk_size {
                self.flush_chunk()?;
            }
        }
        Ok(total)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.v6_builder.is_some() {
            self.flush_v6_step(true)?;
        } else if !self.raw_buf.is_empty() {
            self.flush_chunk()?;
        }
        if let Some(ref mut w) = self.writer {
            w.flush()?;
        }
        Ok(())
    }
}

impl<W: Write> Drop for StreamEncoder<W> {
    fn drop(&mut self) {
        if self.writer.is_some() {
            if self.v6_builder.is_some() {
                let _ = self.flush_v6_step(true);
            } else if !self.raw_buf.is_empty() || self.frames_emitted == 0 {
                let _ = self.flush_chunk();
            }
        }
    }
}

/// Bounded-memory streaming VXG decoder implementing [`std::io::Read`].
pub struct StreamDecoder<R: Read> {
    reader: R,
    options: DecodeOptions,
    frame_buf: Vec<u8>,
    out_buf: Vec<u8>,
    out_len: usize,
    out_pos: usize,
    frames_decoded: u64,
    total_out: u64,
    eof: bool,
}

impl<R: Read> StreamDecoder<R> {
    /// Creates a new streaming decoder reading single or concatenated `VXG1` frames from `reader`.
    pub fn new(reader: R, options: DecodeOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            reader,
            options,
            frame_buf: Vec::new(),
            out_buf: Vec::new(),
            out_len: 0,
            out_pos: 0,
            frames_decoded: 0,
            total_out: 0,
            eof: false,
        })
    }

    /// Total uncompressed bytes decoded so far.
    pub fn total_out(&self) -> u64 {
        self.total_out
    }

    fn load_next_frame(&mut self) -> io::Result<bool> {
        loop {
            let mut hdr = [0u8; frame::HEADER_SIZE];
            if !read_exact_or_eof(&mut self.reader, &mut hdr)? {
                if self.frames_decoded == 0 {
                    return Err(to_io_err(Error::Truncated));
                }
                self.eof = true;
                return Ok(false);
            }
            let rem_budget = (self.options.max_output as u64)
                .saturating_sub(self.total_out)
                .min(usize::MAX as u64) as usize;
            let (_raw_size, _block_size, _count, _version, _csum, dir_end) =
                frame::parse_header_only(&hdr, rem_budget).map_err(to_io_err)?;
            if self.frame_buf.len() < dir_end {
                self.frame_buf.resize(dir_end, 0);
            }
            self.frame_buf[..frame::HEADER_SIZE].copy_from_slice(&hdr);
            self.reader
                .read_exact(&mut self.frame_buf[frame::HEADER_SIZE..dir_end])
                .map_err(|e| {
                    if e.kind() == io::ErrorKind::UnexpectedEof {
                        to_io_err(Error::Truncated)
                    } else {
                        e
                    }
                })?;
            let (frame_len, raw_size) =
                frame::parse_header_and_dir(&self.frame_buf[..dir_end], rem_budget)
                    .map_err(to_io_err)?;
            if self.frame_buf.len() < frame_len {
                self.frame_buf.resize(frame_len, 0);
            }
            self.reader
                .read_exact(&mut self.frame_buf[dir_end..frame_len])
                .map_err(|e| {
                    if e.kind() == io::ErrorKind::UnexpectedEof {
                        to_io_err(Error::Truncated)
                    } else {
                        e
                    }
                })?;
            if self.out_buf.len() < raw_size {
                self.out_buf.resize(raw_size, 0);
            }
            let mut frame_opt = self.options.clone();
            frame_opt.max_output = rem_budget;
            decompress_into(
                &self.frame_buf[..frame_len],
                &mut self.out_buf[..raw_size],
                &frame_opt,
            )
            .map_err(to_io_err)?;
            self.out_len = raw_size;
            self.out_pos = 0;
            self.frames_decoded += 1;
            self.total_out += raw_size as u64;
            if raw_size > 0 {
                return Ok(true);
            }
        }
    }
}

impl<R: Read> Read for StreamDecoder<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.eof {
            return Ok(0);
        }
        while self.out_pos >= self.out_len {
            if !self.load_next_frame()? {
                return Ok(0);
            }
        }
        let avail = &self.out_buf[self.out_pos..self.out_len];
        let take = avail.len().min(buf.len());
        buf[..take].copy_from_slice(&avail[..take]);
        self.out_pos += take;
        Ok(take)
    }
}

/// Streams all bytes from `reader` through [`StreamEncoder`] into `writer` with bounded memory.
/// Returns `(raw_bytes_read, compressed_bytes_written)`.
pub fn compress_stream<R: Read, W: Write>(
    mut reader: R,
    writer: W,
    options: &Options,
) -> Result<(u64, u64)> {
    let mut enc = StreamEncoder::new(writer, options.clone())?;
    if enc.v6_builder.is_some() {
        let mut io_buf = vec![0u8; 65_536];
        loop {
            match reader.read(&mut io_buf) {
                Ok(0) => break,
                Ok(n) => enc.write_all(&io_buf[..n]).map_err(from_io_err)?,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(from_io_err(e)),
            }
        }
        let total_in = enc.total_in + enc.raw_buf.len() as u64;
        let _ = total_in;
        enc.flush_v6_step(true).map_err(from_io_err)?;
        let stats = (enc.total_in, enc.total_out);
        enc.finish().map_err(from_io_err)?;
        return Ok(stats);
    }
    enc.raw_buf.resize(enc.chunk_size, 0);
    let mut filled = 0usize;
    loop {
        match reader.read(&mut enc.raw_buf[filled..]) {
            Ok(0) => break,
            Ok(n) => {
                filled += n;
                if filled == enc.chunk_size {
                    enc.flush_slice(filled).map_err(from_io_err)?;
                    filled = 0;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(from_io_err(e)),
        }
    }
    if filled > 0 || enc.frames_emitted == 0 {
        enc.flush_slice(filled).map_err(from_io_err)?;
    }
    enc.raw_buf.clear();
    let stats = (enc.total_in, enc.total_out);
    enc.finish().map_err(from_io_err)?;
    Ok(stats)
}

/// Streams all `VXG1` frames from `reader` through [`StreamDecoder`] into `writer` with bounded memory.
/// Returns the total uncompressed bytes written.
pub fn decompress_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    options: &DecodeOptions,
) -> Result<u64> {
    let mut dec = StreamDecoder::new(reader, options.clone())?;
    while dec.load_next_frame().map_err(from_io_err)? {
        writer.write_all(&dec.out_buf[..dec.out_len]).map_err(from_io_err)?;
    }
    writer.flush().map_err(from_io_err)?;
    Ok(dec.total_out)
}

/// Computes the total uncompressed size and frame boundaries across one or more concatenated `VXG1` frames.
pub(crate) fn scan_concatenated_frames(
    src: &[u8],
    max_output: usize,
) -> Result<(usize, Vec<(usize, usize, usize)>)> {
    if src.is_empty() {
        return Err(Error::Truncated);
    }
    let (mut pos, mut total_raw, mut spans) = (0usize, 0usize, Vec::new());
    while pos < src.len() {
        if pos > 0 && (src.len() - pos < 4 || &src[pos..pos + 4] != frame::MAGIC) {
            return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
        }
        let rem_budget = max_output.saturating_sub(total_raw);
        let (frame_len, raw_size) = frame::parse_header_and_dir(&src[pos..], rem_budget)?;
        let end = pos.checked_add(frame_len).ok_or(Error::SizeOverflow)?;
        if end > src.len() {
            return Err(Error::Truncated);
        }
        spans.push((pos, end, raw_size));
        total_raw = total_raw.checked_add(raw_size).ok_or(Error::SizeOverflow)?;
        pos = end;
    }
    Ok((total_raw, spans))
}

pub(crate) fn decompress_concatenated(src: &[u8], options: &DecodeOptions) -> Result<Vec<u8>> {
    let (total_raw, spans) = scan_concatenated_frames(src, options.max_output)?;
    let mut out = Vec::new();
    out.try_reserve_exact(total_raw).map_err(|_| Error::SizeOverflow)?;
    out.resize(total_raw, 0);
    let mut out_pos = 0usize;
    for (start, end, raw_size) in spans {
        let mut f_opt = options.clone();
        f_opt.max_output = raw_size;
        decompress_into(&src[start..end], &mut out[out_pos..out_pos + raw_size], &f_opt)?;
        out_pos += raw_size;
    }
    Ok(out)
}

pub(crate) fn decompress_concatenated_into(
    src: &[u8],
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<()> {
    let (total_raw, spans) = scan_concatenated_frames(src, options.max_output)?;
    if out.len() != total_raw {
        return Err(Error::OutputSize { expected: total_raw, actual: out.len() });
    }
    let mut out_pos = 0usize;
    for (start, end, raw_size) in spans {
        let mut f_opt = options.clone();
        f_opt.max_output = raw_size;
        decompress_into(&src[start..end], &mut out[out_pos..out_pos + raw_size], &f_opt)?;
        out_pos += raw_size;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_encoder_decoder_multi_chunk_and_tiny_io_roundtrip() {
        let mut data = Vec::new();
        for i in 0..2400usize {
            data.extend_from_slice(
                format!(
                    "{{\"seq\":{},\"service\":\"stream-gw\",\"ok\":true,\"ms\":{}}}\n",
                    i,
                    10 + (i % 45)
                )
                .as_bytes(),
            );
        }
        let opt = Options {
            block_size: 4096,
            threads: 2,
            checksum: true,
            ..Options::default()
        };
        let mut encoded = Vec::new();
        {
            let mut enc = StreamEncoder::with_chunk_size(&mut encoded, opt.clone(), 8192).unwrap();
            for piece in data.chunks(137) {
                enc.write_all(piece).unwrap();
            }
            enc.finish().unwrap();
        }
        // Single-shot `vxg::decompress` and `vxg::decompress_into` also support multi-frame streams:
        let full_dec = crate::decompress(&encoded, &DecodeOptions::default()).unwrap();
        assert_eq!(full_dec, data);

        let mut into_buf = vec![0u8; data.len()];
        crate::decompress_into(&encoded, &mut into_buf, &DecodeOptions::default()).unwrap();
        assert_eq!(into_buf, data);

        // `StreamDecoder` with tiny 61-byte reads:
        let mut dec = StreamDecoder::new(encoded.as_slice(), DecodeOptions::default()).unwrap();
        let mut out = Vec::new();
        let mut small = [0u8; 61];
        loop {
            let n = dec.read(&mut small).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&small[..n]);
        }
        assert_eq!(out, data);

        // Trailing non-frame garbage after multi-frame stream must still be rejected:
        let mut bad = encoded.clone();
        bad.extend_from_slice(b"trailing_garbage");
        assert!(crate::decompress(&bad, &DecodeOptions::default()).is_err());
        let mut bad_dec = StreamDecoder::new(bad.as_slice(), DecodeOptions::default()).unwrap();
        let mut sink = Vec::new();
        assert!(bad_dec.read_to_end(&mut sink).is_err());
    }
}
