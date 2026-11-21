//! `VIMA1` — HTTP/HTTPS Range-Native Animated Wavelet Image Container.
//!
//! Stores a compact 48-byte `CRC32C`-protected header + `20-byte/frame` random-access directory
//! followed by self-contained `VIMG1` frames. Any `RangeSource` (local file, slice, or `http(s)://`)
//! can scrub directly to frame `#N` and run compressed-domain queries (spatial crop, pyramid resize,
//! grayscale, blur/sharpen/color) inside that single frame without downloading other frames!

use crate::checksum::crc32c;
use crate::format::{encode_vimg, VimgHeader, FLAG_CRC32C, MAX_FILE_BYTES, MAX_IMAGE_DIM};
use crate::ops::{decode_from_source, DecodedImage, TransformQuery};
use crate::range::{RangeSource, RangeStats};
use crate::{EncodeOptions, Error, Result};

pub const VIMA_MAGIC: &[u8; 4] = b"VIMA";
pub const VIMA_VERSION: u16 = 1;
pub const VIMA_HEADER_BYTES: usize = 48;
pub const VIMA_ENTRY_BYTES: usize = 20;
pub const MAX_ANIM_FRAMES: u32 = 65_536;

/// Parsed 48-byte `VIMA1` animated container header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VimaHeader {
    pub version: u16,
    pub flags: u16,
    pub width: u32,
    pub height: u32,
    pub frame_count: u32,
    pub loop_count: u32,
    pub total_bytes: u64,
    pub dir_bytes: u32,
    pub channels: u8,
    pub bit_depth: u8,
}

/// Directory entry locating one `VIMG1` frame inside a `VIMA1` stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VimaFrameEntry {
    pub offset: u64,
    pub len: u32,
    pub duration_ms: u32,
    pub crc32c: u32,
}

/// Parsed `VIMA1` header + frame offset table for O(1) HTTP-Range random access.
#[derive(Clone, Debug)]
pub struct VimaIndex {
    pub header: VimaHeader,
    pub frames: Vec<VimaFrameEntry>,
}

/// Input frame descriptor for [`encode_animation`].
pub struct AnimFrameInput<'a> {
    pub pixels: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub duration_ms: u32,
}

impl VimaHeader {
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < VIMA_HEADER_BYTES {
            return Err(Error::Truncated);
        }
        if &buf[0..4] != VIMA_MAGIC {
            return Err(Error::InvalidFormat("bad VIMA magic"));
        }
        let stored_crc = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        let actual_crc = crc32c(&buf[0..40]);
        if actual_crc != stored_crc {
            return Err(Error::ChecksumMismatch { expected: stored_crc, actual: actual_crc });
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != VIMA_VERSION {
            return Err(Error::Unsupported("unsupported VIMA version"));
        }
        let flags = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let width = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let frame_count = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let loop_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let total_bytes = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        let dir_bytes = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let channels = buf[36];
        let bit_depth = buf[37];
        if width == 0
            || height == 0
            || width > MAX_IMAGE_DIM
            || height > MAX_IMAGE_DIM
            || frame_count == 0
            || frame_count > MAX_ANIM_FRAMES
            || !matches!(channels, 1 | 3 | 4)
            || !matches!(bit_depth, 8 | 10 | 12 | 16)
        {
            return Err(Error::InvalidFormat("invalid VIMA header fields"));
        }
        let expected_dir = (frame_count as usize)
            .checked_mul(VIMA_ENTRY_BYTES)
            .ok_or(Error::InvalidFormat("VIMA directory overflow"))?;
        if dir_bytes as usize != expected_dir || total_bytes > MAX_FILE_BYTES {
            return Err(Error::InvalidFormat("VIMA directory or total_bytes mismatch"));
        }
        Ok(Self {
            version,
            flags,
            width,
            height,
            frame_count,
            loop_count,
            total_bytes,
            dir_bytes,
            channels,
            bit_depth,
        })
    }

    pub fn write_to(&self, out: &mut [u8]) {
        debug_assert!(out.len() >= VIMA_HEADER_BYTES);
        out[0..4].copy_from_slice(VIMA_MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..12].copy_from_slice(&self.width.to_le_bytes());
        out[12..16].copy_from_slice(&self.height.to_le_bytes());
        out[16..20].copy_from_slice(&self.frame_count.to_le_bytes());
        out[20..24].copy_from_slice(&self.loop_count.to_le_bytes());
        out[24..32].copy_from_slice(&self.total_bytes.to_le_bytes());
        out[32..36].copy_from_slice(&self.dir_bytes.to_le_bytes());
        out[36] = self.channels;
        out[37] = self.bit_depth;
        out[38..40].copy_from_slice(&0u16.to_le_bytes());
        let hc = crc32c(&out[0..40]);
        out[40..44].copy_from_slice(&hc.to_le_bytes());
        out[44..48].copy_from_slice(&0u32.to_le_bytes());
    }

    #[inline]
    pub fn has_crc(&self) -> bool {
        (self.flags & FLAG_CRC32C) != 0
    }
}

/// View over a byte sub-range `[base..base + len)` of any `RangeSource`, enabling zero-copy
/// compressed-domain partial queries inside a single frame of a `VIMA1` animation!
struct SubRangeSource<'a> {
    inner: &'a dyn RangeSource,
    base: u64,
    len: u64,
}

impl RangeSource for SubRangeSource<'_> {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_ranges(&self, ranges: &[(u64, u64)]) -> Result<Vec<Vec<u8>>> {
        let mut shifted = Vec::with_capacity(ranges.len());
        for &(s, e) in ranges {
            if e > self.len || s > e {
                return Err(Error::Truncated);
            }
            shifted.push((self.base + s, self.base + e));
        }
        self.inner.read_ranges(&shifted)
    }

    fn preferred_prefix_bytes(&self) -> u64 {
        self.inner.preferred_prefix_bytes().min(self.len)
    }
}

impl VimaIndex {
    pub fn parse_prefix(buf: &[u8]) -> Result<Self> {
        let header = VimaHeader::parse(buf)?;
        let need = VIMA_HEADER_BYTES + header.dir_bytes as usize;
        if buf.len() < need {
            return Err(Error::Truncated);
        }
        let mut frames = Vec::with_capacity(header.frame_count as usize);
        let mut pos = VIMA_HEADER_BYTES;
        for _ in 0..header.frame_count {
            let offset = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            let len = u32::from_le_bytes(buf[pos + 8..pos + 12].try_into().unwrap());
            let duration_ms = u32::from_le_bytes(buf[pos + 12..pos + 16].try_into().unwrap());
            let crc = u32::from_le_bytes(buf[pos + 16..pos + 20].try_into().unwrap());
            pos += VIMA_ENTRY_BYTES;
            let end = offset.checked_add(len as u64).ok_or(Error::InvalidFormat("VIMA frame offset overflow"))?;
            if offset < need as u64 || end > header.total_bytes || len < 48 {
                return Err(Error::InvalidFormat("VIMA frame bounds out of range"));
            }
            frames.push(VimaFrameEntry { offset, len, duration_ms, crc32c: crc });
        }
        Ok(Self { header, frames })
    }

    /// Reads the `VIMA1` header and frame directory over any `RangeSource` (usually 1 Range request).
    pub fn read_from_source(source: &dyn RangeSource, stats: &mut RangeStats) -> Result<Self> {
        let total = source.len();
        stats.total_file_bytes = total;
        if total < VIMA_HEADER_BYTES as u64 {
            return Err(Error::Truncated);
        }
        let init_len = source.preferred_prefix_bytes().min(total);
        let mut prefix = source.read_ranges(&[(0, init_len)])?.pop().ok_or(Error::Truncated)?;
        stats.requests += 1;
        stats.bytes_fetched += prefix.len() as u64;
        let hdr = VimaHeader::parse(&prefix)?;
        let need = (VIMA_HEADER_BYTES + hdr.dir_bytes as usize) as u64;
        if need > total {
            return Err(Error::Truncated);
        }
        if (prefix.len() as u64) < need {
            let extra = source.read_ranges(&[(prefix.len() as u64, need)])?.pop().ok_or(Error::Truncated)?;
            stats.requests += 1;
            stats.bytes_fetched += extra.len() as u64;
            prefix.extend_from_slice(&extra);
        }
        Self::parse_prefix(&prefix)
    }

    /// Total animation duration in milliseconds.
    pub fn total_duration_ms(&self) -> u64 {
        self.frames.iter().map(|f| f.duration_ms as u64).sum()
    }

    /// Random-access HTTP-Range decode of a single frame `frame_idx` with any compressed-domain
    /// `TransformQuery` (pyramid resize, spatial crop, grayscale, blur/sharpen/color)!
    pub fn decode_frame_from_source(
        &self,
        source: &dyn RangeSource,
        frame_idx: usize,
        query: &TransformQuery,
    ) -> Result<DecodedImage> {
        let entry = self
            .frames
            .get(frame_idx)
            .ok_or(Error::InvalidInput("animation frame index out of bounds"))?;
        let sub = SubRangeSource {
            inner: source,
            base: entry.offset,
            len: entry.len as u64,
        };
        decode_from_source(&sub, query)
    }
}

/// Encodes a sequence of raw 8-bit frames into a `VIMA1` Range-native animated container.
pub fn encode_animation(
    frames: &[AnimFrameInput<'_>],
    loop_count: u32,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    if frames.is_empty() || frames.len() > MAX_ANIM_FRAMES as usize {
        return Err(Error::InvalidInput("animation must contain 1..=65536 frames"));
    }
    let mut encoded_bufs: Vec<(Vec<u8>, u32)> = Vec::with_capacity(frames.len());
    for f in frames {
        let bytes = encode_vimg(f.pixels, f.width, f.height, f.channels, opts)?;
        encoded_bufs.push((bytes, f.duration_ms));
    }
    let refs: Vec<(&[u8], u32)> = encoded_bufs.iter().map(|(b, d)| (b.as_slice(), *d)).collect();
    encode_animation_from_vimg(&refs, loop_count)
}

/// Zero-reencode muxer combining pre-encoded `VIMG1` frames `(vimg_bytes, duration_ms)` into a
/// `VIMA1` animated container (works for 8/10/12/16-bit HDR and lossless/lossy frames).
pub fn encode_animation_from_vimg(vimg_frames: &[(&[u8], u32)], loop_count: u32) -> Result<Vec<u8>> {
    if vimg_frames.is_empty() || vimg_frames.len() > MAX_ANIM_FRAMES as usize {
        return Err(Error::InvalidInput("animation must contain 1..=65536 frames"));
    }
    let first = VimgHeader::parse(vimg_frames[0].0)?;
    let (w, h, ch, bd) = (first.width, first.height, first.channels, first.bit_depth());
    for &(fb, _) in vimg_frames.iter().skip(1) {
        let fh = VimgHeader::parse(fb)?;
        if fh.width != w || fh.height != h {
            return Err(Error::InvalidInput("all animation frames must have identical width and height"));
        }
    }
    let dir_bytes = vimg_frames.len() * VIMA_ENTRY_BYTES;
    let payload_bytes: usize = vimg_frames.iter().map(|(b, _)| b.len()).sum();
    let total_bytes = VIMA_HEADER_BYTES + dir_bytes + payload_bytes;
    if total_bytes as u64 > MAX_FILE_BYTES {
        return Err(Error::InvalidInput("animation archive exceeds MAX_FILE_BYTES"));
    }

    let hdr = VimaHeader {
        version: VIMA_VERSION,
        flags: FLAG_CRC32C,
        width: w,
        height: h,
        frame_count: vimg_frames.len() as u32,
        loop_count,
        total_bytes: total_bytes as u64,
        dir_bytes: dir_bytes as u32,
        channels: ch,
        bit_depth: bd,
    };

    let mut out = vec![0u8; total_bytes];
    hdr.write_to(&mut out[..VIMA_HEADER_BYTES]);
    let mut dir_pos = VIMA_HEADER_BYTES;
    let mut data_pos = VIMA_HEADER_BYTES + dir_bytes;
    for &(fb, dur) in vimg_frames {
        let len = fb.len() as u32;
        let crc = crc32c(fb);
        out[dir_pos..dir_pos + 8].copy_from_slice(&(data_pos as u64).to_le_bytes());
        out[dir_pos + 8..dir_pos + 12].copy_from_slice(&len.to_le_bytes());
        out[dir_pos + 12..dir_pos + 16].copy_from_slice(&dur.to_le_bytes());
        out[dir_pos + 16..dir_pos + 20].copy_from_slice(&crc.to_le_bytes());
        dir_pos += VIMA_ENTRY_BYTES;
        out[data_pos..data_pos + fb.len()].copy_from_slice(fb);
        data_pos += fb.len();
    }
    Ok(out)
}

/// **Zero-Decode Animation Trimming (`vima -> vima`):**
/// Extracts frames `[start_frame .. start_frame + count)` verbatim over any `RangeSource` without
/// decoding a single wavelet band or pixel!
pub fn trim_vima_zero_decode(
    source: &dyn RangeSource,
    start_frame: usize,
    count: usize,
) -> Result<(Vec<u8>, RangeStats)> {
    let mut stats = RangeStats::default();
    let idx = VimaIndex::read_from_source(source, &mut stats)?;
    if count == 0 || start_frame.saturating_add(count) > idx.frames.len() {
        return Err(Error::InvalidInput("trim frame range out of bounds"));
    }
    let slice_entries = &idx.frames[start_frame..start_frame + count];
    let first_off = slice_entries[0].offset;
    let last = slice_entries[slice_entries.len() - 1];
    let end_off = last.offset + last.len as u64;
    let span = source.read_ranges(&[(first_off, end_off)])?.pop().ok_or(Error::Truncated)?;
    stats.requests += 1;
    stats.bytes_fetched += span.len() as u64;

    let mut frame_refs = Vec::with_capacity(count);
    for e in slice_entries {
        let rel = (e.offset - first_off) as usize;
        let fb = &span[rel..rel + e.len as usize];
        if idx.header.has_crc() {
            let actual = crc32c(fb);
            if actual != e.crc32c {
                return Err(Error::ChecksumMismatch { expected: e.crc32c, actual });
            }
        }
        frame_refs.push((fb, e.duration_ms));
    }
    let out = encode_animation_from_vimg(&frame_refs, idx.header.loop_count)?;
    Ok((out, stats))
}
