//! `VIMG1` wire container: 48-byte header, level-then-plane-then-tile compact directory,
//! parallel encoder (`1T == NT` deterministic), and zero-decode tile crop (`vimg -> vimg`).

use crate::checksum::crc32c;
use crate::color::{downsample_2x2, split_planes_444};
use crate::range::{coalesce_ranges, RangeSource, RangeStats};
use crate::subband::encode_wavelet_tile_levels;
use crate::wavelet::{level_dims, WaveletTile};
use crate::{EncodeOptions, Error, Result};

pub const MAGIC: &[u8; 4] = b"VIMG";
pub const VERSION_V1: u16 = 1;
pub const HEADER_BYTES: usize = 48;

pub const FLAG_CRC32C: u16 = 0x0001;
pub const FLAG_CHROMA_420: u16 = 0x0002;
pub const FLAG_LOSSLESS: u16 = 0x0004;

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

        if width == 0 || height == 0 || !matches!(channels, 1 | 3 | 4) {
            return Err(Error::InvalidFormat("invalid image dimensions or channels"));
        }
        if !(5..=9).contains(&tile_log2) || dwt_levels > 4 {
            return Err(Error::InvalidFormat("invalid tile_size_log2 or dwt_levels"));
        }
        let tile_size = 1u32 << tile_log2;
        if tiles_x != width.div_ceil(tile_size) || tiles_y != height.div_ceil(tile_size) {
            return Err(Error::InvalidFormat("tile grid mismatch"));
        }
        let expected_entries = (dwt_levels as usize + 1)
            .checked_mul(channels as usize)
            .and_then(|v| v.checked_mul(tiles_x as usize))
            .and_then(|v| v.checked_mul(tiles_y as usize))
            .ok_or(Error::InvalidFormat("directory overflow"))?;
        if dir_bytes as usize != expected_entries * 8 {
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
        out[44..48].fill(0);
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
        let mut cur_off = need as u64;
        for i in 0..n_entries {
            let b = HEADER_BYTES + i * 8;
            let len = u32::from_le_bytes(prefix[b..b + 4].try_into().unwrap());
            let crc = u32::from_le_bytes(prefix[b + 4..b + 8].try_into().unwrap());
            entries.push(ChunkEntry { offset: cur_off, len, crc32c: crc });
            cur_off = cur_off.checked_add(len as u64).ok_or(Error::InvalidFormat("offset overflow"))?;
        }
        if cur_off != header.total_bytes {
            return Err(Error::InvalidFormat("total_bytes mismatch with directory"));
        }
        Ok(Self { header, entries })
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
        let initial_len = total.min(4096);
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

fn extract_tile_subplane(
    plane: &[i16],
    img_w: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<i16> {
    let mut out = Vec::with_capacity(tw * th);
    for y in y0..y0 + th {
        out.extend_from_slice(&plane[y * img_w + x0..y * img_w + x0 + tw]);
    }
    out
}

/// Encodes packed RGB / RGBA / Grayscale pixels into a self-contained `VIMG1` byte buffer.
/// Parallelized cleanly across tiles with `1T == NT` bit-identical deterministic output.
pub fn encode_vimg(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u8,
    opts: &EncodeOptions,
) -> Result<Vec<u8>> {
    let (w, h, ch) = (width as usize, height as usize, channels as usize);
    if w == 0 || h == 0 || !matches!(channels, 1 | 3 | 4) || pixels.len() != w * h * ch {
        return Err(Error::InvalidInput("invalid dimensions, channels, or pixel buffer length"));
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

    let (y_full, co_full, cg_full, a_full) = split_planes_444(pixels, w, h, ch);
    let mut planes_data: Vec<&[i16]> = Vec::with_capacity(ch);
    planes_data.push(&y_full);
    if ch >= 3 {
        planes_data.push(&co_full);
        planes_data.push(&cg_full);
    }
    if let Some(ref a) = a_full {
        planes_data.push(a);
    }

    let n_threads = if opts.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        opts.threads
    }
    .clamp(1, 32)
    .min(n_tiles.max(1));

    // Encode every tile in parallel; each tile produces `planes × levels` compressed blobs.
    let tile_results: Vec<Vec<Vec<Vec<u8>>>> = if n_threads <= 1 {
        (0..n_tiles)
            .map(|tid| encode_one_tile(tid, tiles_x, tile_size, w, h, &planes_data, chroma_420, quality, dwt_levels))
            .collect()
    } else {
        let mut results: Vec<Option<Vec<Vec<Vec<u8>>>>> = (0..n_tiles).map(|_| None).collect();
        let chunk = n_tiles.div_ceil(n_threads);
        std::thread::scope(|s| {
            let mut rest = results.as_mut_slice();
            for t in 0..n_threads {
                let start = t * chunk;
                if start >= n_tiles {
                    break;
                }
                let end = (start + chunk).min(n_tiles);
                let (dst, tail) = rest.split_at_mut(end - start);
                rest = tail;
                let planes_ref = &planes_data;
                s.spawn(move || {
                    for (local_i, slot) in dst.iter_mut().enumerate() {
                        let tid = start + local_i;
                        *slot = Some(encode_one_tile(
                            tid, tiles_x, tile_size, w, h, planes_ref, chroma_420, quality, dwt_levels,
                        ));
                    }
                });
            }
        });
        results.into_iter().map(|o| o.unwrap()).collect()
    };

    let n_entries = n_levels * ch * n_tiles;
    let dir_bytes = n_entries * 8;
    let mut payload_bytes = 0usize;
    for tr in &tile_results {
        for p_vec in tr {
            for l_blob in p_vec {
                payload_bytes += l_blob.len();
            }
        }
    }

    let total_bytes = HEADER_BYTES + dir_bytes + payload_bytes;
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
        total_bytes: total_bytes as u64,
    };
    header.write_to(&mut out[..HEADER_BYTES]);

    // Write directory and payloads strictly ordered by `(level, plane, tile_id)`.
    let mut dir_pos = HEADER_BYTES;
    let mut data_pos = HEADER_BYTES + dir_bytes;
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
    debug_assert_eq!(data_pos, total_bytes);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn encode_one_tile(
    tid: usize,
    tiles_x: usize,
    tile_size: usize,
    img_w: usize,
    img_h: usize,
    planes: &[&[i16]],
    chroma_420: bool,
    quality: u8,
    dwt_levels: usize,
) -> Vec<Vec<Vec<u8>>> {
    let tx = tid % tiles_x;
    let ty = tid / tiles_x;
    let x0 = tx * tile_size;
    let y0 = ty * tile_size;
    let tw = (img_w - x0).min(tile_size);
    let th = (img_h - y0).min(tile_size);

    let mut out_planes = Vec::with_capacity(planes.len());
    for (p_idx, plane) in planes.iter().enumerate() {
        let sub = extract_tile_subplane(plane, img_w, x0, y0, tw, th);
        let (work, pw, ph) = if chroma_420 && (p_idx == 1 || p_idx == 2) {
            downsample_2x2(&sub, tw, th)
        } else {
            (sub, tw, th)
        };
        let wt = WaveletTile::forward(&work, pw, ph, dwt_levels);
        out_planes.push(encode_wavelet_tile_levels(&wt, quality, p_idx));
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
    let idx = VimgIndex::read_from_source(source, &mut stats)?;
    let h = &idx.header;
    if ntx == 0 || nty == 0 || tx0 + ntx > h.tiles_x || ty0 + nty > h.tiles_y {
        return Err(Error::InvalidInput("tile crop range out of bounds"));
    }

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
    let total_bytes = HEADER_BYTES + dir_bytes + payload_bytes;

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
        total_bytes: total_bytes as u64,
    };

    let mut out = vec![0u8; total_bytes];
    out_hdr.write_to(&mut out[..HEADER_BYTES]);
    let mut dir_pos = HEADER_BYTES;
    let mut data_pos = HEADER_BYTES + dir_bytes;

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
    Ok((out, stats))
}
