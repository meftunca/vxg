//! `VIMG1` wire container: 48-byte header, level-then-plane-then-tile compact directory,
//! parallel encoder (`1T == NT` deterministic), and zero-decode tile crop (`vimg -> vimg`).

use crate::checksum::crc32c;
use crate::color::{downsample_2x2, rgb_to_ycocg_r, rgb_to_ycocg_r_u16};
use crate::meta::{VimgMetadata, MAX_META_BYTES};
use crate::range::{coalesce_ranges, RangeSource, RangeStats};
use crate::simd;
use crate::subband::encode_wavelet_tile_levels_full;
use crate::wavelet::{level_dims, WaveletScratch, WaveletTile};
use crate::{EncodeOptions, Error, Result};

pub const MAGIC: &[u8; 4] = b"VIMG";
pub const VERSION_V1: u16 = 1;
pub const HEADER_BYTES: usize = 48;

/// Hard security & OOM-bomb bounds (safe on both 64-bit native and 32-bit `wasm32`).
pub const MAX_IMAGE_DIM: u32 = 65_536;
pub const MAX_IMAGE_PIXELS: u64 = 268_435_456; // 256 megapixels
pub const MAX_DIR_BYTES: u32 = 16 * 1024 * 1024; // 16 MiB directory cap
pub const MAX_ICC_BYTES: u32 = 16 * 1024 * 1024; // 16 MiB ICC profile cap
pub const MAX_CHUNK_BYTES: u32 = 64 * 1024 * 1024; // 64 MiB single-chunk cap
pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB archive cap
pub const MAX_RESIZE_DIM: u32 = 32_768;
pub const MAX_RESIZE_PIXELS: u64 = 134_217_728; // 128 megapixels

pub const FLAG_CRC32C: u16 = 0x0001;
pub const FLAG_CHROMA_420: u16 = 0x0002;
pub const FLAG_LOSSLESS: u16 = 0x0004;
pub const FLAG_DWT_97: u16 = 0x0008;
pub const FLAG_BIT_DEPTH_10: u16 = 0x0010;
pub const FLAG_BIT_DEPTH_12: u16 = 0x0020;
pub const FLAG_HAS_ICC: u16 = 0x0040;
pub const FLAG_HAS_META: u16 = 0x0080;
/// 2-bit shift count (`0..=3`) recording how many fine pyramid levels were stripped by
/// `downscale_vimg_zero_decode`, preserving exact `quant_step` parity without re-quantizing.
pub const FLAG_QUANT_SHIFT_MASK: u16 = 0x0300;
pub const FLAG_QUANT_SHIFT_BIT: u32 = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VimgHeader {
    pub version: u16,
    pub flags: u16,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub quality: u8,
    pub tile_size: u32,
    pub dwt_levels: u8,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub dir_bytes: u32,
    pub icc_bytes: u32,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ChunkEntry {
    pub offset: u64,
    pub len: u32,
    pub crc32c: u32,
}

#[derive(Clone, Debug)]
pub struct VimgIndex {
    pub header: VimgHeader,
    /// Flat entries indexed by `entry_idx(level, plane, tile_id)`.
    pub entries: Vec<ChunkEntry>,
    /// Byte offset of optional trailing TLV metadata block (`[meta_offset..total_bytes]`).
    pub meta_offset: u64,
    /// Wire length in bytes of the trailing TLV metadata block (`0` when absent).
    pub meta_wire_len: u64,
}

impl VimgHeader {
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.dwt_levels as usize + 1
    }

    #[inline]
    pub fn n_planes(&self) -> usize {
        self.channels as usize
    }

    #[inline]
    pub fn n_tiles(&self) -> usize {
        (self.tiles_x as usize) * (self.tiles_y as usize)
    }

    #[inline]
    pub fn is_chroma_420(&self) -> bool {
        (self.flags & FLAG_CHROMA_420) != 0 && self.channels >= 3
    }

    #[inline]
    pub fn has_crc(&self) -> bool {
        (self.flags & FLAG_CRC32C) != 0
    }

    #[inline]
    pub fn is_lossless(&self) -> bool {
        (self.flags & FLAG_LOSSLESS) != 0 || self.quality >= 100
    }

    #[inline]
    pub fn is_dwt_97(&self) -> bool {
        (self.flags & FLAG_DWT_97) != 0 && !self.is_lossless()
    }

    #[inline]
    pub fn bit_depth(&self) -> u8 {
        if (self.flags & FLAG_BIT_DEPTH_12) != 0 {
            12
        } else if (self.flags & FLAG_BIT_DEPTH_10) != 0 {
            10
        } else {
            8
        }
    }

    #[inline]
    pub fn max_sample(&self) -> u16 {
        (1u16 << self.bit_depth()) - 1
    }

    #[inline]
    pub fn has_icc(&self) -> bool {
        (self.flags & FLAG_HAS_ICC) != 0 && self.icc_bytes > 0
    }

    #[inline]
    pub fn has_meta(&self) -> bool {
        (self.flags & FLAG_HAS_META) != 0
    }

    /// Number of stripped finest wavelet levels from zero-decode pyramid truncation (`0..=3`).
    #[inline]
    pub fn quant_level_shift(&self) -> usize {
        ((self.flags & FLAG_QUANT_SHIFT_MASK) >> FLAG_QUANT_SHIFT_BIT) as usize
    }

    /// Effective `dwt_levels` for perceptual `quant_step` lookup (`dwt_levels + quant_level_shift`).
    #[inline]
    pub fn quant_dwt_levels(&self) -> usize {
        self.dwt_levels as usize + self.quant_level_shift()
    }

    #[inline]
    pub fn icc_offset(&self) -> u64 {
        (HEADER_BYTES as u64) + (self.dir_bytes as u64)
    }

    /// Bytes occupied on disk by the embedded ICC profile (+ 4-byte CRC32C when `has_crc()`).
    #[inline]
    pub fn icc_wire_len(&self) -> u64 {
        if !self.has_icc() {
            0
        } else {
            (self.icc_bytes as u64) + if self.has_crc() { 4 } else { 0 }
        }
    }

    #[inline]
    pub fn entry_index(&self, level: usize, plane: usize, tile_id: usize) -> usize {
        ((level * self.n_planes() + plane) * self.n_tiles()) + tile_id
    }

    /// Pixel dimensions of plane `plane` for tile `(tx, ty)` at full resolution.
    pub fn tile_plane_dims(&self, tx: usize, ty: usize, plane: usize) -> (usize, usize) {
        let ts = self.tile_size as usize;
        let x0 = tx * ts;
        let y0 = ty * ts;
        let tw = ((self.width as usize) - x0).min(ts);
        let th = ((self.height as usize) - y0).min(ts);
        if self.is_chroma_420() && (plane == 1 || plane == 2) {
            (tw.div_ceil(2), th.div_ceil(2))
        } else {
            (tw, th)
        }
    }

    /// Decoded pixel dimensions of the whole image at pyramid `level` (`0` = coarsest, `dwt_levels` = full).
    pub fn image_level_dims(&self, level: usize) -> (usize, usize) {
        let lvl = level.min(self.dwt_levels as usize);
        let steps = (self.dwt_levels as usize) - lvl;
        let mut w = 0usize;
        for tx in 0..self.tiles_x as usize {
            let (tw, _) = self.tile_plane_dims(tx, 0, 0);
            w += level_dims(tw, 1, steps).0;
        }
        let mut h = 0usize;
        for ty in 0..self.tiles_y as usize {
            let (_, th) = self.tile_plane_dims(0, ty, 0);
            h += level_dims(1, th, steps).1;
        }
        (w, h)
    }

    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_BYTES {
            return Err(Error::Truncated);
        }
        if &buf[0..4] != MAGIC {
            return Err(Error::InvalidFormat("bad VIMG magic"));
        }
        let stored_crc = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        if crc32c(&buf[0..40]) != stored_crc {
            return Err(Error::ChecksumMismatch { expected: stored_crc, actual: crc32c(&buf[0..40]) });
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != VERSION_V1 {
            return Err(Error::Unsupported("unsupported VIMG version"));
        }
        let flags = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let width = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let height = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let channels = buf[16];
        let quality = buf[17];
        let tile_log2 = buf[18];
        let dwt_levels = buf[19];
        let tiles_x = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let tiles_y = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let dir_bytes = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let total_bytes = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        let icc_bytes = u32::from_le_bytes(buf[44..48].try_into().unwrap());

        if (flags & (FLAG_BIT_DEPTH_10 | FLAG_BIT_DEPTH_12)) == (FLAG_BIT_DEPTH_10 | FLAG_BIT_DEPTH_12) {
            return Err(Error::InvalidFormat("conflicting bit depth flags"));
        }
        if (flags & FLAG_HAS_ICC) != 0 {
            if icc_bytes == 0 || icc_bytes > MAX_ICC_BYTES {
                return Err(Error::InvalidFormat("invalid ICC profile length"));
            }
        } else if icc_bytes != 0 {
            return Err(Error::InvalidFormat("non-zero reserved bytes without FLAG_HAS_ICC"));
        }

        if width == 0
            || height == 0
            || width > MAX_IMAGE_DIM
            || height > MAX_IMAGE_DIM
            || (width as u64) * (height as u64) > MAX_IMAGE_PIXELS
            || !matches!(channels, 1 | 3 | 4)
        {
            return Err(Error::InvalidFormat("invalid image dimensions or channels"));
        }
        if quality == 0 || quality > 100 {
            return Err(Error::InvalidFormat("invalid quality in header"));
        }
        if !(5..=9).contains(&tile_log2) || dwt_levels > 4 {
            return Err(Error::InvalidFormat("invalid tile_size_log2 or dwt_levels"));
        }
        let tile_size = 1u32 << tile_log2;
        if tiles_x != width.div_ceil(tile_size) || tiles_y != height.div_ceil(tile_size) {
            return Err(Error::InvalidFormat("tile grid mismatch"));
        }
        if dir_bytes > MAX_DIR_BYTES || total_bytes > MAX_FILE_BYTES {
            return Err(Error::InvalidFormat("archive exceeds security size limits"));
        }
        let expected_entries = (dwt_levels as usize + 1)
            .checked_mul(channels as usize)
            .and_then(|v| v.checked_mul(tiles_x as usize))
            .and_then(|v| v.checked_mul(tiles_y as usize))
            .ok_or(Error::InvalidFormat("directory overflow"))?;
        let expected_dir_bytes = expected_entries
            .checked_mul(8)
            .ok_or(Error::InvalidFormat("directory byte overflow"))?;
        if dir_bytes as usize != expected_dir_bytes {
            return Err(Error::InvalidFormat("directory size mismatch"));
        }
        Ok(Self {
            version,
            flags,
            width,
            height,
            channels,
            quality,
            tile_size,
            dwt_levels,
            tiles_x,
            tiles_y,
            dir_bytes,
            icc_bytes,
            total_bytes,
        })
    }

    pub fn write_to(&self, out: &mut [u8]) {
        let tile_log2 = self.tile_size.trailing_zeros() as u8;
        out[0..4].copy_from_slice(MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..12].copy_from_slice(&self.width.to_le_bytes());
        out[12..16].copy_from_slice(&self.height.to_le_bytes());
        out[16] = self.channels;
        out[17] = self.quality;
        out[18] = tile_log2;
        out[19] = self.dwt_levels;
        out[20..24].copy_from_slice(&self.tiles_x.to_le_bytes());
        out[24..28].copy_from_slice(&self.tiles_y.to_le_bytes());
        out[28..32].copy_from_slice(&self.dir_bytes.to_le_bytes());
        out[32..40].copy_from_slice(&self.total_bytes.to_le_bytes());
        let hcrc = crc32c(&out[0..40]);
        out[40..44].copy_from_slice(&hcrc.to_le_bytes());
        out[44..48].copy_from_slice(&self.icc_bytes.to_le_bytes());
    }
}

impl VimgIndex {
    /// Parses full header + directory from an in-memory prefix slice (`>= 48 + dir_bytes`).
    pub fn parse_prefix(prefix: &[u8]) -> Result<Self> {
        let header = VimgHeader::parse(prefix)?;
        let need = HEADER_BYTES + header.dir_bytes as usize;
        if prefix.len() < need {
            return Err(Error::Truncated);
        }
        let n_entries = header.dir_bytes as usize / 8;
        let mut entries = Vec::with_capacity(n_entries);
        let mut cur_off = (need as u64)
            .checked_add(header.icc_wire_len())
            .ok_or(Error::InvalidFormat("icc offset overflow"))?;
        for i in 0..n_entries {
            let b = HEADER_BYTES + i * 8;
            let len = u32::from_le_bytes(prefix[b..b + 4].try_into().unwrap());
            if len > MAX_CHUNK_BYTES {
                return Err(Error::InvalidFormat("chunk exceeds MAX_CHUNK_BYTES"));
            }
            let crc = u32::from_le_bytes(prefix[b + 4..b + 8].try_into().unwrap());
            entries.push(ChunkEntry { offset: cur_off, len, crc32c: crc });
            cur_off = cur_off.checked_add(len as u64).ok_or(Error::InvalidFormat("offset overflow"))?;
        }
        let (meta_offset, meta_wire_len) = if header.has_meta() {
            if cur_off >= header.total_bytes {
                return Err(Error::InvalidFormat("FLAG_HAS_META set but trailing metadata missing"));
            }
            let mlen = header.total_bytes - cur_off;
            let max_wire = (MAX_META_BYTES as u64) + if header.has_crc() { 4 } else { 0 };
            if mlen > max_wire {
                return Err(Error::InvalidFormat("trailing metadata exceeds MAX_META_BYTES"));
            }
            (cur_off, mlen)
        } else {
            if cur_off != header.total_bytes {
                return Err(Error::InvalidFormat("total_bytes mismatch with directory"));
            }
            (cur_off, 0)
        };
        Ok(Self { header, entries, meta_offset, meta_wire_len })
    }

    /// Reads and CRC-verifies the optional trailing TLV metadata block (`CICP`, `EXIF`, `XMP`).
    pub fn read_metadata_from_source(
        &self,
        source: &dyn RangeSource,
        verify_crc: bool,
        stats: &mut RangeStats,
    ) -> Result<Option<VimgMetadata>> {
        if self.meta_wire_len == 0 {
            return Ok(None);
        }
        let end = self
            .meta_offset
            .checked_add(self.meta_wire_len)
            .ok_or(Error::InvalidFormat("metadata range overflow"))?;
        let mut got = source.read_ranges(&[(self.meta_offset, end)])?;
        stats.requests += 1;
        stats.bytes_fetched += self.meta_wire_len;
        let buf = got.pop().unwrap_or_default();
        if buf.len() as u64 != self.meta_wire_len {
            return Err(Error::Truncated);
        }
        Ok(Some(VimgMetadata::decode_tlv(&buf, self.header.has_crc(), verify_crc)?))
    }

    /// Reads and CRC-verifies the embedded ICC color profile (if present), reusing `prefix` bytes
    /// with zero extra HTTP Range requests whenever possible.
    pub fn read_icc_from_source(
        &self,
        source: &dyn RangeSource,
        prefix: &[u8],
        verify_crc: bool,
        stats: &mut RangeStats,
    ) -> Result<Option<Vec<u8>>> {
        if !self.header.has_icc() {
            return Ok(None);
        }
        let off = self.header.icc_offset();
        let wire_len = self.header.icc_wire_len();
        let end = off.checked_add(wire_len).ok_or(Error::InvalidFormat("icc range overflow"))?;
        if end > self.header.total_bytes {
            return Err(Error::Truncated);
        }
        let raw_buf: Vec<u8>;
        let slice: &[u8] = if (end as usize) <= prefix.len() {
            &prefix[off as usize..end as usize]
        } else {
            let mut got = source.read_ranges(&[(off, end)])?;
            stats.requests += 1;
            stats.bytes_fetched += wire_len;
            raw_buf = got.pop().unwrap_or_default();
            &raw_buf
        };
        if slice.len() != wire_len as usize {
            return Err(Error::Truncated);
        }
        let icc_len = self.header.icc_bytes as usize;
        let icc = &slice[..icc_len];
        if self.header.has_crc() && verify_crc {
            let expected = u32::from_le_bytes(slice[icc_len..icc_len + 4].try_into().unwrap());
            let actual = crc32c(icc);
            if actual != expected {
                return Err(Error::ChecksumMismatch { expected, actual });
            }
        }
        Ok(Some(icc.to_vec()))
    }

    /// Fetches header + directory from any `RangeSource` (in 1–2 Range requests),
    /// returning both the parsed index and the already-fetched prefix bytes `[0..N)` so
    /// callers can serve early chunks (e.g. Level 0 thumbnails) with zero extra requests.
    pub fn read_from_source_with_prefix(
        source: &dyn RangeSource,
        stats: &mut RangeStats,
    ) -> Result<(Self, Vec<u8>)> {
        let total = source.len();
        stats.total_file_bytes = total;
        let initial_len = total.min(source.preferred_prefix_bytes().max(HEADER_BYTES as u64));
        if initial_len < HEADER_BYTES as u64 {
            return Err(Error::Truncated);
        }
        let mut bufs = source.read_ranges(&[(0, initial_len)])?;
        stats.requests += 1;
        stats.bytes_fetched += initial_len;
        let mut prefix = bufs.pop().unwrap();
        let hdr = VimgHeader::parse(&prefix)?;
        let need = (HEADER_BYTES + hdr.dir_bytes as usize) as u64;
        if need > total {
            return Err(Error::Truncated);
        }
        if need > initial_len {
            let mut rem = source.read_ranges(&[(initial_len, need)])?;
            stats.requests += 1;
            stats.bytes_fetched += need - initial_len;
            prefix.extend_from_slice(&rem.pop().unwrap());
        }
        let idx = Self::parse_prefix(&prefix[..need as usize])?;
        Ok((idx, prefix))
    }

    /// Fetches header + directory from any `RangeSource` (in 1–2 Range requests).
    pub fn read_from_source(source: &dyn RangeSource, stats: &mut RangeStats) -> Result<Self> {
        Self::read_from_source_with_prefix(source, stats).map(|(idx, _)| idx)
    }

    #[inline]
    pub fn entry(&self, level: usize, plane: usize, tile_id: usize) -> ChunkEntry {
        self.entries[self.header.entry_index(level, plane, tile_id)]
    }

    /// Returns `(level_byte_sizes)` for inspecting pyramid size breakdown.
    pub fn level_byte_sizes(&self) -> Vec<u64> {
        let nl = self.header.n_levels();
        let np = self.header.n_planes();
        let nt = self.header.n_tiles();
        (0..nl)
            .map(|l| {
                let mut sum = 0u64;
                for p in 0..np {
                    for t in 0..nt {
                        sum += self.entry(l, p, t).len as u64;
                    }
                }
                sum
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum PixelInput<'a> {
    U8(&'a [u8]),
    U16(&'a [u16], u16),
}

/// Encodes packed 8-bit RGB / RGBA / Grayscale pixels into a self-contained `VIMG1` byte buffer.
/// Uses L1-hot tile-local color conversion, reusable per-worker scratch buffers, and `AtomicUsize`
/// work-stealing with `1T == NT` bit-identical deterministic output.
pub fn encode_vimg(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u8,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    let (w, h, ch) = (width as usize, height as usize, channels as usize);
    if width == 0
        || height == 0
        || width > MAX_IMAGE_DIM
        || height > MAX_IMAGE_DIM
        || (width as u64) * (height as u64) > MAX_IMAGE_PIXELS
        || !matches!(channels, 1 | 3 | 4)
        || w.checked_mul(h).and_then(|n| n.checked_mul(ch)) != Some(pixels.len())
    {
        return Err(Error::InvalidInput("invalid dimensions, channels, or pixel buffer length"));
    }
    encode_pixels_vimg(PixelInput::U8(pixels), width, height, channels, 8, opts)
}

/// Encodes packed 10-bit or 12-bit HDR `u16` RGB / RGBA / Grayscale pixels into a `VIMG1` buffer.
/// Set `opts.bit_depth` to `10` (`0..=1023`) or `12` (`0..=4095`).
pub fn encode_vimg_u16(
    pixels: &[u16],
    width: u32,
    height: u32,
    channels: u8,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    let bit_depth = if opts.bit_depth == 8 { 10 } else { opts.bit_depth };
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::InvalidInput("bit_depth must be 8, 10, or 12"));
    }
    let (w, h, ch) = (width as usize, height as usize, channels as usize);
    if width == 0
        || height == 0
        || width > MAX_IMAGE_DIM
        || height > MAX_IMAGE_DIM
        || (width as u64) * (height as u64) > MAX_IMAGE_PIXELS
        || !matches!(channels, 1 | 3 | 4)
        || w.checked_mul(h).and_then(|n| n.checked_mul(ch)) != Some(pixels.len())
    {
        return Err(Error::InvalidInput("invalid dimensions, channels, or u16 pixel buffer length"));
    }
    let max_val = (1u16 << bit_depth) - 1;
    encode_pixels_vimg(PixelInput::U16(pixels, max_val), width, height, channels, bit_depth, opts)
}

fn encode_pixels_vimg(
    input: PixelInput<'_>,
    width: u32,
    height: u32,
    channels: u8,
    bit_depth: u8,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    let (w, h, ch) = (width as usize, height as usize, channels as usize);
    let icc_slice: &[u8] = opts.icc_profile.as_deref().unwrap_or(&[]);
    if icc_slice.len() > MAX_ICC_BYTES as usize {
        return Err(Error::InvalidInput("ICC profile exceeds MAX_ICC_BYTES"));
    }
    let tile_size = opts.tile_size.clamp(64, 256).next_power_of_two() as usize;
    let dwt_levels = opts.dwt_levels.min(4) as usize;
    let quality = opts.quality.clamp(1, 100);
    let is_lossless = quality >= 100;
    let chroma_420 = opts.chroma_420 && !is_lossless && channels >= 3;

    let tiles_x = w.div_ceil(tile_size);
    let tiles_y = h.div_ceil(tile_size);
    let n_tiles = tiles_x * tiles_y;
    let n_levels = dwt_levels + 1;

    let n_threads = if opts.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        opts.threads
    }
    .clamp(1, 32)
    .min(n_tiles.max(1));

    let adaptive_quant = opts.adaptive_quant && !is_lossless;
    let meta = VimgMetadata {
        cicp: opts.cicp,
        exif: opts.exif.clone(),
        xmp: opts.xmp.clone(),
    };
    let meta_wire = if meta.is_empty() {
        Vec::new()
    } else {
        meta.encode_tlv(opts.checksum)?
    };

    // Encode every tile in parallel with L1-hot tile-local color split + AtomicUsize work-stealing.
    let tile_results: Vec<Vec<Vec<Vec<u8>>>> = if n_threads <= 1 {
        let mut scratch = TileEncodeScratch::new(tile_size, ch);
        (0..n_tiles)
            .map(|tid| {
                encode_one_tile(
                    tid, tiles_x, tile_size, w, h, ch, input, chroma_420, quality, dwt_levels, bit_depth,
                    adaptive_quant, &mut scratch,
                )
            })
            .collect()
    } else {
        use std::sync::atomic::{AtomicUsize, Ordering};
        type EncodedTile = Vec<Vec<Vec<u8>>>;
        let next_tid = AtomicUsize::new(0);
        let mut worker_batches: Vec<Vec<(usize, EncodedTile)>> = Vec::with_capacity(n_threads);
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let counter = &next_tid;
                    s.spawn(move || {
                        let mut scratch = TileEncodeScratch::new(tile_size, ch);
                        let mut local = Vec::new();
                        loop {
                            let tid = counter.fetch_add(1, Ordering::Relaxed);
                            if tid >= n_tiles {
                                break;
                            }
                            let encoded = encode_one_tile(
                                tid, tiles_x, tile_size, w, h, ch, input, chroma_420, quality, dwt_levels, bit_depth,
                                adaptive_quant, &mut scratch,
                            );
                            local.push((tid, encoded));
                        }
                        local
                    })
                })
                .collect();
            for h in handles {
                worker_batches.push(h.join().expect("vimg encode worker panicked"));
            }
        });
        let mut ordered: Vec<Option<Vec<Vec<Vec<u8>>>>> = (0..n_tiles).map(|_| None).collect();
        for batch in worker_batches {
            for (tid, blob) in batch {
                ordered[tid] = Some(blob);
            }
        }
        ordered.into_iter().map(|o| o.unwrap()).collect()
    };

    let n_entries = n_levels * ch * n_tiles;
    let dir_bytes = n_entries * 8;
    let icc_wire_len = if icc_slice.is_empty() {
        0usize
    } else {
        icc_slice.len() + if opts.checksum { 4 } else { 0 }
    };
    let mut payload_bytes = 0usize;
    for tr in &tile_results {
        for p_vec in tr {
            for l_blob in p_vec {
                payload_bytes += l_blob.len();
            }
        }
    }

    let total_bytes = HEADER_BYTES + dir_bytes + icc_wire_len + payload_bytes + meta_wire.len();
    let mut out = vec![0u8; total_bytes];

    let mut flags = 0u16;
    if opts.checksum {
        flags |= FLAG_CRC32C;
    }
    if chroma_420 {
        flags |= FLAG_CHROMA_420;
    }
    if is_lossless {
        flags |= FLAG_LOSSLESS;
    } else {
        flags |= FLAG_DWT_97;
    }
    if bit_depth == 10 {
        flags |= FLAG_BIT_DEPTH_10;
    } else if bit_depth == 12 {
        flags |= FLAG_BIT_DEPTH_12;
    }
    if !icc_slice.is_empty() {
        flags |= FLAG_HAS_ICC;
    }
    if !meta_wire.is_empty() {
        flags |= FLAG_HAS_META;
    }
    let header = VimgHeader {
        version: VERSION_V1,
        flags,
        width,
        height,
        channels,
        quality,
        tile_size: tile_size as u32,
        dwt_levels: dwt_levels as u8,
        tiles_x: tiles_x as u32,
        tiles_y: tiles_y as u32,
        dir_bytes: dir_bytes as u32,
        icc_bytes: icc_slice.len() as u32,
        total_bytes: total_bytes as u64,
    };
    header.write_to(&mut out[..HEADER_BYTES]);

    // Write optional ICC color profile (+ CRC32C) right after the directory.
    let mut data_pos = HEADER_BYTES + dir_bytes;
    if !icc_slice.is_empty() {
        out[data_pos..data_pos + icc_slice.len()].copy_from_slice(icc_slice);
        data_pos += icc_slice.len();
        if opts.checksum {
            out[data_pos..data_pos + 4].copy_from_slice(&crc32c(icc_slice).to_le_bytes());
            data_pos += 4;
        }
    }

    // Write directory and payloads strictly ordered by `(level, plane, tile_id)`.
    let mut dir_pos = HEADER_BYTES;
    #[allow(clippy::needless_range_loop)]
    for lvl in 0..n_levels {
        for plane in 0..ch {
            for tid in 0..n_tiles {
                let blob = &tile_results[tid][plane][lvl];
                let len = blob.len();
                let crc = if opts.checksum { crc32c(blob) } else { 0 };
                out[dir_pos..dir_pos + 4].copy_from_slice(&(len as u32).to_le_bytes());
                out[dir_pos + 4..dir_pos + 8].copy_from_slice(&crc.to_le_bytes());
                dir_pos += 8;
                out[data_pos..data_pos + len].copy_from_slice(blob);
                data_pos += len;
            }
        }
    }
    if !meta_wire.is_empty() {
        out[data_pos..data_pos + meta_wire.len()].copy_from_slice(&meta_wire);
        data_pos += meta_wire.len();
    }
    debug_assert_eq!(data_pos, total_bytes);
    Ok(out)
}

struct TileEncodeScratch {
    y: Vec<i16>,
    co: Vec<i16>,
    cg: Vec<i16>,
    a: Vec<i16>,
    wt: WaveletScratch,
}

impl TileEncodeScratch {
    fn new(tile_size: usize, ch: usize) -> Self {
        let cap = tile_size * tile_size;
        Self {
            y: vec![0i16; cap],
            co: if ch >= 3 { vec![0i16; cap] } else { Vec::new() },
            cg: if ch >= 3 { vec![0i16; cap] } else { Vec::new() },
            a: if ch == 4 { vec![0i16; cap] } else { Vec::new() },
            wt: WaveletScratch::default(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_one_tile(
    tid: usize,
    tiles_x: usize,
    tile_size: usize,
    img_w: usize,
    img_h: usize,
    ch: usize,
    input: PixelInput<'_>,
    chroma_420: bool,
    quality: u8,
    dwt_levels: usize,
    bit_depth: u8,
    adaptive_quant: bool,
    scratch: &mut TileEncodeScratch,
) -> Vec<Vec<Vec<u8>>> {
    let tx = tid % tiles_x;
    let ty = tid / tiles_x;
    let x0 = tx * tile_size;
    let y0 = ty * tile_size;
    let tw = (img_w - x0).min(tile_size);
    let th = (img_h - y0).min(tile_size);
    let n = tw * th;

    // L1-hot tile-local RGB/RGBA -> YCoCg-R split straight from source pixels (zero full-frame canvas!).
    match input {
        PixelInput::U8(pixels) => match ch {
            1 => {
                for row in 0..th {
                    let src = &pixels[(y0 + row) * img_w + x0..(y0 + row) * img_w + x0 + tw];
                    let dst = &mut scratch.y[row * tw..(row + 1) * tw];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = s as i16;
                    }
                }
            }
            3 => {
                for row in 0..th {
                    let src_off = ((y0 + row) * img_w + x0) * 3;
                    let src = &pixels[src_off..src_off + tw * 3];
                    let dst_off = row * tw;
                    simd::split_rgb_ycocg_r(
                        src,
                        &mut scratch.y[dst_off..dst_off + tw],
                        &mut scratch.co[dst_off..dst_off + tw],
                        &mut scratch.cg[dst_off..dst_off + tw],
                    );
                }
            }
            _ => {
                for row in 0..th {
                    let src_off = ((y0 + row) * img_w + x0) * 4;
                    let src = &pixels[src_off..src_off + tw * 4];
                    let dst_off = row * tw;
                    for x in 0..tw {
                        let o = x * 4;
                        let (yv, cov, cgv) = rgb_to_ycocg_r(src[o], src[o + 1], src[o + 2]);
                        scratch.y[dst_off + x] = yv;
                        scratch.co[dst_off + x] = cov;
                        scratch.cg[dst_off + x] = cgv;
                        scratch.a[dst_off + x] = src[o + 3] as i16;
                    }
                }
            }
        },
        PixelInput::U16(pixels, max_val) => match ch {
            1 => {
                for row in 0..th {
                    let src = &pixels[(y0 + row) * img_w + x0..(y0 + row) * img_w + x0 + tw];
                    let dst = &mut scratch.y[row * tw..(row + 1) * tw];
                    for (d, &s) in dst.iter_mut().zip(src) {
                        *d = s.min(max_val) as i16;
                    }
                }
            }
            _ => {
                for row in 0..th {
                    let src_off = ((y0 + row) * img_w + x0) * ch;
                    let src = &pixels[src_off..src_off + tw * ch];
                    let dst_off = row * tw;
                    for x in 0..tw {
                        let o = x * ch;
                        let (yv, cov, cgv) = rgb_to_ycocg_r_u16(
                            src[o].min(max_val),
                            src[o + 1].min(max_val),
                            src[o + 2].min(max_val),
                        );
                        scratch.y[dst_off + x] = yv;
                        scratch.co[dst_off + x] = cov;
                        scratch.cg[dst_off + x] = cgv;
                        if ch == 4 {
                            scratch.a[dst_off + x] = src[o + 3].min(max_val) as i16;
                        }
                    }
                }
            }
        },
    }

    let use_97 = quality < 100;
    let mut out_planes = Vec::with_capacity(ch);
    for p_idx in 0..ch {
        let (wt, sp_opt) = if chroma_420 && (p_idx == 1 || p_idx == 2) {
            let src_p = if p_idx == 1 { &scratch.co[..n] } else { &scratch.cg[..n] };
            let (ds, pw, ph) = downsample_2x2(src_p, tw, th);
            (WaveletTile::forward_mode_with(&ds, pw, ph, dwt_levels, use_97, &mut scratch.wt), None)
        } else {
            let src_p = match p_idx {
                0 => &scratch.y[..n],
                1 => &scratch.co[..n],
                2 => &scratch.cg[..n],
                _ => &scratch.a[..n],
            };
            (
                WaveletTile::forward_mode_with(src_p, tw, th, dwt_levels, use_97, &mut scratch.wt),
                if !use_97 { Some(src_p) } else { None },
            )
        };
        out_planes.push(encode_wavelet_tile_levels_full(&wt, sp_opt, quality, p_idx, bit_depth, adaptive_quant));
    }
    out_planes
}

/// **Zero-Decode Tile-Aligned Crop (`vimg -> vimg`):**
/// Copies compressed tile payloads for tile box `(tx0..tx0+ntx, ty0..ty0+nty)` verbatim from
/// any `RangeSource` into a brand-new standalone `.vimg` file without running FSE or IDWT!
/// Optionally drops chroma planes (`grayscale = true`) or truncates finer pyramid levels (`max_level`).
pub fn crop_vimg_zero_decode(
    source: &dyn RangeSource,
    tx0: u32,
    ty0: u32,
    ntx: u32,
    nty: u32,
    grayscale: bool,
) -> Result<(Vec<u8>, RangeStats)> {
    let mut stats = RangeStats::default();
    let (idx, prefix) = VimgIndex::read_from_source_with_prefix(source, &mut stats)?;
    let h = &idx.header;
    if ntx == 0
        || nty == 0
        || tx0.checked_add(ntx).is_none_or(|end| end > h.tiles_x)
        || ty0.checked_add(nty).is_none_or(|end| end > h.tiles_y)
    {
        return Err(Error::InvalidInput("tile crop range out of bounds"));
    }

    let icc_opt = idx.read_icc_from_source(source, &prefix, h.has_crc(), &mut stats)?;
    let icc_slice: &[u8] = icc_opt.as_deref().unwrap_or(&[]);
    let icc_wire_len = if icc_slice.is_empty() {
        0usize
    } else {
        icc_slice.len() + if h.has_crc() { 4 } else { 0 }
    };
    let meta_wire = if idx.meta_wire_len > 0 {
        let mut got = source.read_ranges(&[(idx.meta_offset, idx.meta_offset + idx.meta_wire_len)])?;
        stats.requests += 1;
        stats.bytes_fetched += idx.meta_wire_len;
        let raw = got.pop().unwrap_or_default();
        // Verify metadata CRC32C before copying verbatim!
        let _ = VimgMetadata::decode_tlv(&raw, h.has_crc(), h.has_crc())?;
        raw
    } else {
        Vec::new()
    };

    let out_ch: u8 = if grayscale { 1 } else { h.channels };
    let n_levels = h.n_levels();
    let out_tiles = (ntx as usize) * (nty as usize);

    let mut req_meta = Vec::with_capacity(n_levels * (out_ch as usize) * out_tiles);
    let mut raw_ranges = Vec::with_capacity(req_meta.capacity());
    for lvl in 0..n_levels {
        for plane in 0..out_ch as usize {
            for ty in ty0..ty0 + nty {
                for tx in tx0..tx0 + ntx {
                    let src_tid = (ty as usize) * (h.tiles_x as usize) + (tx as usize);
                    let e = idx.entry(lvl, plane, src_tid);
                    req_meta.push(e);
                    raw_ranges.push((e.offset, e.offset + e.len as u64));
                }
            }
        }
    }

    let (coalesced, mapping) = coalesce_ranges(&raw_ranges, 8192);
    let fetched = source.read_ranges(&coalesced)?;
    stats.requests += coalesced.len();
    stats.bytes_fetched += coalesced.iter().map(|&(a, b)| b - a).sum::<u64>();

    let new_w = ((h.width - tx0 * h.tile_size).min(ntx * h.tile_size)).max(1);
    let new_h = ((h.height - ty0 * h.tile_size).min(nty * h.tile_size)).max(1);
    let dir_bytes = req_meta.len() * 8;
    let payload_bytes: usize = req_meta.iter().map(|e| e.len as usize).sum();
    let total_bytes = HEADER_BYTES + dir_bytes + icc_wire_len + payload_bytes + meta_wire.len();

    let mut flags = h.flags;
    if grayscale {
        flags &= !FLAG_CHROMA_420;
    }
    let out_hdr = VimgHeader {
        version: VERSION_V1,
        flags,
        width: new_w,
        height: new_h,
        channels: out_ch,
        quality: h.quality,
        tile_size: h.tile_size,
        dwt_levels: h.dwt_levels,
        tiles_x: ntx,
        tiles_y: nty,
        dir_bytes: dir_bytes as u32,
        icc_bytes: icc_slice.len() as u32,
        total_bytes: total_bytes as u64,
    };

    let mut out = vec![0u8; total_bytes];
    out_hdr.write_to(&mut out[..HEADER_BYTES]);
    let mut dir_pos = HEADER_BYTES;
    let mut data_pos = HEADER_BYTES + dir_bytes;
    if !icc_slice.is_empty() {
        out[data_pos..data_pos + icc_slice.len()].copy_from_slice(icc_slice);
        data_pos += icc_slice.len();
        if h.has_crc() {
            out[data_pos..data_pos + 4].copy_from_slice(&crc32c(icc_slice).to_le_bytes());
            data_pos += 4;
        }
    }

    for (i, e) in req_meta.iter().enumerate() {
        let (ci, rel_off) = mapping[i];
        let slice = &fetched[ci][rel_off..rel_off + e.len as usize];
        if h.has_crc() {
            let actual = crc32c(slice);
            if actual != e.crc32c {
                return Err(Error::ChecksumMismatch { expected: e.crc32c, actual });
            }
        }
        out[dir_pos..dir_pos + 4].copy_from_slice(&e.len.to_le_bytes());
        out[dir_pos + 4..dir_pos + 8].copy_from_slice(&e.crc32c.to_le_bytes());
        dir_pos += 8;
        out[data_pos..data_pos + slice.len()].copy_from_slice(slice);
        data_pos += slice.len();
    }
    if !meta_wire.is_empty() {
        out[data_pos..data_pos + meta_wire.len()].copy_from_slice(&meta_wire);
    }
    Ok((out, stats))
}

/// **Zero-Decode Pyramid Downscale + Optional Grayscale & Metadata Strip (`vimg -> vimg`):**
/// Truncates an archive to `max_level` (`0 = 1/8`, `1 = 1/4`, `2 = 1/2`, `3 = 1/1`) by copying
/// compressed chunks verbatim without running IDWT, range decoding, or re-encoding!
/// When `strip_meta = true`, drops EXIF and XMP privacy metadata while preserving CICP & ICC color accuracy.
pub fn downscale_vimg_zero_decode(
    source: &dyn RangeSource,
    max_level: u8,
    grayscale: bool,
    strip_meta: bool,
) -> Result<(Vec<u8>, RangeStats)> {
    let mut stats = RangeStats::default();
    let (idx, prefix) = VimgIndex::read_from_source_with_prefix(source, &mut stats)?;
    let h = &idx.header;
    let target_dwt = max_level.min(h.dwt_levels);
    let steps_skipped = (h.dwt_levels - target_dwt) as u32;
    let new_tile_size = h.tile_size >> steps_skipped;
    if new_tile_size < 32 {
        return Err(Error::InvalidInput(
            "zero-decode downscale target tile_size < 32 (use a higher max_level or larger tile_size)",
        ));
    }

    let icc_opt = idx.read_icc_from_source(source, &prefix, h.has_crc(), &mut stats)?;
    let icc_slice: &[u8] = icc_opt.as_deref().unwrap_or(&[]);
    let icc_wire_len = if icc_slice.is_empty() {
        0usize
    } else {
        icc_slice.len() + if h.has_crc() { 4 } else { 0 }
    };

    let meta_wire = if idx.meta_wire_len > 0 {
        let mut got = source.read_ranges(&[(idx.meta_offset, idx.meta_offset + idx.meta_wire_len)])?;
        stats.requests += 1;
        stats.bytes_fetched += idx.meta_wire_len;
        let raw = got.pop().unwrap_or_default();
        let parsed = VimgMetadata::decode_tlv(&raw, h.has_crc(), h.has_crc())?;
        if strip_meta {
            let kept = VimgMetadata { cicp: parsed.cicp, exif: None, xmp: None };
            if kept.is_empty() { Vec::new() } else { kept.encode_tlv(h.has_crc())? }
        } else {
            raw
        }
    } else {
        Vec::new()
    };

    let out_ch: u8 = if grayscale { 1 } else { h.channels };
    let out_levels = (target_dwt as usize) + 1;
    let n_tiles = h.n_tiles();

    let mut req_meta = Vec::with_capacity(out_levels * (out_ch as usize) * n_tiles);
    let mut raw_ranges = Vec::with_capacity(req_meta.capacity());
    for lvl in 0..out_levels {
        for plane in 0..out_ch as usize {
            for tid in 0..n_tiles {
                let e = idx.entry(lvl, plane, tid);
                req_meta.push(e);
                raw_ranges.push((e.offset, e.offset + e.len as u64));
            }
        }
    }

    let (coalesced, mapping) = coalesce_ranges(&raw_ranges, 8192);
    let fetched = source.read_ranges(&coalesced)?;
    stats.requests += coalesced.len();
    stats.bytes_fetched += coalesced.iter().map(|&(a, b)| b - a).sum::<u64>();

    let (new_w, new_h) = h.image_level_dims(target_dwt as usize);
    let dir_bytes = req_meta.len() * 8;
    let payload_bytes: usize = req_meta.iter().map(|e| e.len as usize).sum();
    let total_bytes = HEADER_BYTES + dir_bytes + icc_wire_len + payload_bytes + meta_wire.len();

    let mut flags = h.flags & !FLAG_QUANT_SHIFT_MASK;
    let new_shift = (h.quant_level_shift() + steps_skipped as usize).min(3);
    flags |= (new_shift as u16) << FLAG_QUANT_SHIFT_BIT;
    if grayscale {
        flags &= !FLAG_CHROMA_420;
    }
    if meta_wire.is_empty() {
        flags &= !FLAG_HAS_META;
    } else {
        flags |= FLAG_HAS_META;
    }
    let out_hdr = VimgHeader {
        version: VERSION_V1,
        flags,
        width: new_w as u32,
        height: new_h as u32,
        channels: out_ch,
        quality: h.quality,
        tile_size: new_tile_size,
        dwt_levels: target_dwt,
        tiles_x: h.tiles_x,
        tiles_y: h.tiles_y,
        dir_bytes: dir_bytes as u32,
        icc_bytes: icc_slice.len() as u32,
        total_bytes: total_bytes as u64,
    };

    let mut out = vec![0u8; total_bytes];
    out_hdr.write_to(&mut out[..HEADER_BYTES]);
    let mut dir_pos = HEADER_BYTES;
    let mut data_pos = HEADER_BYTES + dir_bytes;
    if !icc_slice.is_empty() {
        out[data_pos..data_pos + icc_slice.len()].copy_from_slice(icc_slice);
        data_pos += icc_slice.len();
        if h.has_crc() {
            out[data_pos..data_pos + 4].copy_from_slice(&crc32c(icc_slice).to_le_bytes());
            data_pos += 4;
        }
    }
    for (i, e) in req_meta.iter().enumerate() {
        let (ci, rel_off) = mapping[i];
        let slice = &fetched[ci][rel_off..rel_off + e.len as usize];
        if h.has_crc() {
            let actual = crc32c(slice);
            if actual != e.crc32c {
                return Err(Error::ChecksumMismatch { expected: e.crc32c, actual });
            }
        }
        out[dir_pos..dir_pos + 4].copy_from_slice(&e.len.to_le_bytes());
        out[dir_pos + 4..dir_pos + 8].copy_from_slice(&e.crc32c.to_le_bytes());
        dir_pos += 8;
        out[data_pos..data_pos + slice.len()].copy_from_slice(slice);
        data_pos += slice.len();
    }
    if !meta_wire.is_empty() {
        out[data_pos..data_pos + meta_wire.len()].copy_from_slice(&meta_wire);
    }
    Ok((out, stats))
}
