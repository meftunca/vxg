use crate::{block, checksum::crc32c, simd::Engine, DecodeOptions, Error, Result};

pub const HEADER_SIZE: usize = 32;
pub const ENTRY_SIZE: usize = 32;
pub const MAGIC: &[u8; 4] = b"VXG1";
/// v2 adds opcode 7: a two-byte absolute match distance. v1 frames remain readable.
pub const VERSION: u16 = 2;
/// v3 additionally allows structured-column block payloads. Opt-in on encode;
/// v1 and v2 frames remain readable.
#[allow(dead_code)]
pub const VERSION_V3: u16 = 3;
/// v4 (release) is the unified adaptive pipeline with a compact 8-byte block
/// directory: `u32 (payload_size | kind << 30)` + `u32 checksum`. Raw and
/// payload offsets are implied by the canonical block order, so random access
/// stays O(1) after one prefix-sum pass in `Frame::parse`.
pub const VERSION_V4: u16 = 4;
/// v5 extends v4 with an optional frame-shared prefix dictionary (`FLAG_FRAME_DICT = 0x02`)
/// and optional variable-size record-aligned block directory (`FLAG_VAR_RAW = 0x04`).
pub const VERSION_V5: u16 = 5;
/// v6 extends v5 with column-grouped macro-blocks (`FLAG_COL_GROUP = 0x10`) for cold analytical archives.
pub const VERSION_V6: u16 = 6;
/// Version emitted by default when neither frame dictionary nor record-aligned variable blocks are active.
pub const VERSION_RELEASE: u16 = VERSION_V4;
pub const ENTRY_SIZE_V4: usize = 8;
const MIN_VERSION: u16 = 1;
const MAX_VERSION: u16 = VERSION_V6;
const V4_SIZE_MASK: u32 = 0x3fff_ffff;

/// Directory entry size for a given frame version.
#[inline]
pub const fn entry_size(version: u16) -> usize {
    if version >= VERSION_V4 {
        ENTRY_SIZE_V4
    } else {
        ENTRY_SIZE
    }
}

/// Write one directory entry at `out[p..p + entry_size(version)]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_entry(
    out: &mut [u8],
    p: usize,
    version: u16,
    raw_offset: usize,
    payload_offset: usize,
    raw_size: usize,
    payload_size: usize,
    checksum: u32,
    kind: u32,
) {
    if version >= VERSION_V4 {
        debug_assert!(payload_size as u64 <= V4_SIZE_MASK as u64 && kind < 4);
        let word = (payload_size as u32 & V4_SIZE_MASK) | (kind << 30);
        out[p..p + 4].copy_from_slice(&word.to_le_bytes());
        out[p + 4..p + 8].copy_from_slice(&checksum.to_le_bytes());
    } else {
        out[p..p + 8].copy_from_slice(&(raw_offset as u64).to_le_bytes());
        out[p + 8..p + 16].copy_from_slice(&(payload_offset as u64).to_le_bytes());
        out[p + 16..p + 20].copy_from_slice(&(raw_size as u32).to_le_bytes());
        out[p + 20..p + 24].copy_from_slice(&(payload_size as u32).to_le_bytes());
        out[p + 24..p + 28].copy_from_slice(&checksum.to_le_bytes());
        out[p + 28..p + 32].copy_from_slice(&kind.to_le_bytes());
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlockInfo {
    pub raw_offset: usize,
    pub payload_offset: usize,
    pub raw_size: usize,
    pub payload_size: usize,
    pub checksum: u32,
    pub kind: u32,
}
#[derive(Debug)]
pub(crate) struct Block<'a> {
    pub info: BlockInfo,
    pub payload: &'a [u8],
    pub col_slices: &'a [&'a [u8]],
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArenaPart {
    pub index: usize,
    pub offset: usize,
    pub len: usize,
    pub kind: u32,
    pub checksum: u32,
    pub raw_size: usize,
    pub row_meta: u32,
}
#[derive(Clone, Copy)]
pub(crate) struct WorkerSlice<'a> {
    pub arena: &'a [u8],
    pub parts: &'a [ArenaPart],
    pub stats: block::WorkStats,
}
#[derive(Debug)]
pub struct Frame<'a> {
    pub(crate) blocks: Vec<Block<'a>>,
    pub(crate) dict: Vec<u8>,
    pub(crate) row_index: Option<Vec<u32>>,
    pub(crate) raw_size: usize,
    pub(crate) block_size: usize,
    pub(crate) checksummed: bool,
    pub(crate) version: u16,
    pub(crate) _owned_arena: Vec<u8>,
    pub(crate) _owned_col_slices: Vec<&'static [u8]>,
}
/// A frame whose command streams have been checked once and remain immutably
/// borrowed for repeated CPU decode. Construct it with `Frame::into_validated`.
#[derive(Debug)]
pub struct ValidatedFrame<'a> {
    frame: Frame<'a>,
    proof: ValidatedCommandStreams,
}
/// Private, unforgeable evidence that every command stream in the associated
/// `Frame` passed `block::validate`. Its only constructor is
/// `Frame::into_validated` after that validation succeeds.
#[derive(Debug)]
struct ValidatedCommandStreams(());
fn read_u32(src: &[u8], p: usize) -> Result<u32> {
    let v = src
        .get(p..p.checked_add(4).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::Truncated)?;
    Ok(u32::from_le_bytes(v.try_into().unwrap()))
}
fn read_u64(src: &[u8], p: usize) -> Result<u64> {
    let v = src
        .get(p..p.checked_add(8).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::Truncated)?;
    Ok(u64::from_le_bytes(v.try_into().unwrap()))
}
pub(crate) fn check_block_size(n: usize) -> Result<()> {
    if !(4096..=1_048_576).contains(&n) || n % 64 != 0 {
        return Err(Error::InvalidOption(
            "block size must be 4096..1048576 and a multiple of 64",
        ));
    }
    Ok(())
}
pub fn max_compressed_size(n: usize, block_size: usize) -> Result<usize> {
    check_block_size(block_size)?;
    let blocks = n / block_size + usize::from(n % block_size != 0);
    if blocks > u32::MAX as usize {
        return Err(Error::SizeOverflow);
    }
    n.checked_add(HEADER_SIZE)
        .and_then(|x| {
            blocks
                .checked_mul(ENTRY_SIZE)
                .and_then(|d| x.checked_add(d))
        })
        .ok_or(Error::SizeOverflow)
}

/// Validates a 32-byte VXG header and returns `(raw_size, block_size, count, version, checksummed, dir_end)`.
pub(crate) fn parse_header_only(
    src: &[u8],
    max_output: usize,
) -> Result<(usize, usize, usize, u16, bool, usize)> {
    let hdr = v5::parse_v1_to_v5_header(src, max_output)?;
    Ok((
        hdr.raw_size,
        hdr.block_size,
        hdr.count,
        hdr.version,
        hdr.checksummed,
        hdr.payload_start,
    ))
}

#[inline]
pub(crate) fn valid_kind_and_size(version: u16, kind: u32, cn: usize, rn: usize) -> bool {
    match kind {
        block::RAW => cn == rn,
        block::LZ => cn < rn && cn >= block::BLOCK_PREFIX,
        block::COLUMN => version >= 3 && cn >= 2 && cn < rn,
        block::DICT_LZ => version >= VERSION_V5 && cn >= 2 && cn < rn,
        _ => false,
    }
}

/// Validates the header and directory prefix of `src` and returns `(frame_compressed_size, raw_size)`.
pub(crate) fn parse_header_and_dir(src: &[u8], max_output: usize) -> Result<(usize, usize)> {
    let hdr = v5::parse_v1_to_v5_header(src, max_output)?;
    if src.len() < hdr.payload_start {
        return Err(Error::Truncated);
    }
    if hdr.version == VERSION_V6 {
        return v6::validate_v6_header_and_dir(src, &hdr);
    }
    let esize = entry_size(hdr.version);
    let mut next_payload = hdr.payload_start;
    let mut next_raw = 0usize;
    for i in 0..hdr.count {
        let p = HEADER_SIZE + i * esize;
        let expected_rn = v5::block_raw_size(src, &hdr, i, next_raw)?;
        let _ = v5::block_row_meta_raw(src, &hdr, i, expected_rn)?;
        let (raw_offset, payload_offset, rn, cn, checksum, kind) = if hdr.version >= VERSION_V4 {
            let word = read_u32(src, p)?;
            (
                next_raw,
                next_payload,
                expected_rn,
                (word & V4_SIZE_MASK) as usize,
                read_u32(src, p + 4)?,
                word >> 30,
            )
        } else {
            (
                usize::try_from(read_u64(src, p)?).map_err(|_| Error::SizeOverflow)?,
                usize::try_from(read_u64(src, p + 8)?).map_err(|_| Error::SizeOverflow)?,
                read_u32(src, p + 16)? as usize,
                read_u32(src, p + 20)? as usize,
                read_u32(src, p + 24)?,
                read_u32(src, p + 28)?,
            )
        };
        if raw_offset != next_raw || payload_offset != next_payload || rn != expected_rn {
            return Err(Error::InvalidFrame("non-canonical directory"));
        }
        if !valid_kind_and_size(hdr.version, kind, cn, rn) {
            return Err(Error::InvalidFrame("block kind/size"));
        }
        if !hdr.checksummed && checksum != 0 {
            return Err(Error::InvalidFrame("checksum without flag"));
        }
        next_payload = payload_offset.checked_add(cn).ok_or(Error::SizeOverflow)?;
        next_raw += rn;
    }
    if next_raw != hdr.raw_size {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    Ok((next_payload, hdr.raw_size))
}
impl<'a> Frame<'a> {
    /// Validates the header and directory before any output allocation.
    /// Command validation is fused with CPU decoding, or explicitly requested by GPU callers.
    pub fn parse(src: &'a [u8], max_output: usize) -> Result<Self> {
        let hdr = v5::parse_v1_to_v5_header(src, max_output)?;
        if hdr.payload_start > src.len() {
            return Err(Error::Truncated);
        }
        if hdr.version == VERSION_V6 {
            return v6::parse_v6_frame(src, &hdr);
        }
        let dict = v5::decode_header_dict(src, &hdr)?;
        let esize = entry_size(hdr.version);
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(hdr.count)
            .map_err(|_| Error::SizeOverflow)?;
        let mut row_index = if hdr.has_row_index {
            let mut v = Vec::new();
            v.try_reserve_exact(hdr.count)
                .map_err(|_| Error::SizeOverflow)?;
            Some(v)
        } else {
            None
        };
        let mut next_payload = hdr.payload_start;
        let mut next_raw = 0;
        for i in 0..hdr.count {
            let p = HEADER_SIZE + i * esize;
            let expected_rn = v5::block_raw_size(src, &hdr, i, next_raw)?;
            if let Some(word) = v5::block_row_meta_raw(src, &hdr, i, expected_rn)? {
                if let Some(ref mut ri) = row_index {
                    ri.push(word);
                }
            }
            let (raw_offset, payload_offset, rn, cn, checksum, kind) = if hdr.version >= VERSION_V4 {
                let word = read_u32(src, p)?;
                (
                    next_raw,
                    next_payload,
                    expected_rn,
                    (word & V4_SIZE_MASK) as usize,
                    read_u32(src, p + 4)?,
                    word >> 30,
                )
            } else {
                (
                    usize::try_from(read_u64(src, p)?).map_err(|_| Error::SizeOverflow)?,
                    usize::try_from(read_u64(src, p + 8)?).map_err(|_| Error::SizeOverflow)?,
                    read_u32(src, p + 16)? as usize,
                    read_u32(src, p + 20)? as usize,
                    read_u32(src, p + 24)?,
                    read_u32(src, p + 28)?,
                )
            };
            if raw_offset != next_raw || payload_offset != next_payload || rn != expected_rn {
                return Err(Error::InvalidFrame("non-canonical directory"));
            }
            if !valid_kind_and_size(hdr.version, kind, cn, rn) {
                return Err(Error::InvalidFrame("block kind/size"));
            }
            if !hdr.checksummed && checksum != 0 {
                return Err(Error::InvalidFrame("checksum without flag"));
            }
            let end = payload_offset.checked_add(cn).ok_or(Error::SizeOverflow)?;
            let payload = src.get(payload_offset..end).ok_or(Error::Truncated)?;
            blocks.push(Block {
                info: BlockInfo {
                    raw_offset,
                    payload_offset,
                    raw_size: rn,
                    payload_size: cn,
                    checksum,
                    kind,
                },
                payload,
                col_slices: &[],
            });
            next_raw += rn;
            next_payload = end;
        }
        if next_raw != hdr.raw_size || next_payload != src.len() {
            return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
        }
        Ok(Self {
            blocks,
            dict,
            row_index,
            raw_size: hdr.raw_size,
            block_size: hdr.block_size,
            checksummed: hdr.checksummed,
            version: hdr.version,
            _owned_arena: Vec::new(),
            _owned_col_slices: Vec::new(),
        })
    }
    /// Parses one `VXG1` frame or a concatenated sequence of `VXG1` frames.
    pub fn parse_all(src: &'a [u8], max_output: usize) -> Result<Vec<Self>> {
        match Self::parse(src, max_output) {
            Ok(f) => Ok(vec![f]),
            Err(Error::InvalidFrame("trailing bytes or incorrect total")) => {
                let (_, spans) = crate::stream::scan_concatenated_frames(src, max_output)?;
                let mut frames = Vec::with_capacity(spans.len());
                for (start, end, raw_size) in spans {
                    frames.push(Self::parse(&src[start..end], raw_size)?);
                }
                Ok(frames)
            }
            Err(e) => Err(e),
        }
    }
    /// Parses one or more frames for query execution: uses a single `mincore(2)` residency probe
    /// (`take_or_probe_residency`) across `src`, serving resident `1 MiB` chunks zero-copy from
    /// `mmap` and reading non-resident chunks via parallel `1 MiB` coalesced `pread(2)`.
    pub(crate) fn parse_all_for_query(
        src: &'a [u8],
        max_output: usize,
        where_expr: Option<&crate::ops::PredExpr<'_>>,
        extra_cols: &[crate::ops::ColTarget<'_>],
        allow_partial_cols: bool,
        threads: usize,
    ) -> Result<Vec<Self>> {
        let res = crate::mmap::pread::take_or_probe_residency(src);
        match Self::parse_one_for_query(
            src,
            max_output,
            where_expr,
            extra_cols,
            allow_partial_cols,
            threads,
            &res,
        ) {
            Ok(f) => Ok(vec![f]),
            Err(Error::InvalidFrame("trailing bytes or incorrect total")) => {
                let (_, spans) = crate::stream::scan_concatenated_frames(src, max_output)?;
                let mut frames = Vec::with_capacity(spans.len());
                for (start, end, raw_size) in spans {
                    let sub = &src[start..end];
                    let sub_res = crate::mmap::pread::probe_residency_if_mapped(sub);
                    frames.push(Self::parse_one_for_query(
                        sub,
                        raw_size,
                        where_expr,
                        extra_cols,
                        allow_partial_cols,
                        threads,
                        &sub_res,
                    )?);
                }
                Ok(frames)
            }
            Err(e) => Err(e),
        }
    }
    fn parse_one_for_query(
        src: &'a [u8],
        max_output: usize,
        where_expr: Option<&crate::ops::PredExpr<'_>>,
        extra_cols: &[crate::ops::ColTarget<'_>],
        allow_partial_cols: bool,
        threads: usize,
        res: &crate::mmap::sys::ChunkResidency,
    ) -> Result<Self> {
        if res.all_resident() {
            return Self::parse(src, max_output);
        }
        let Some((fd, file_off)) = crate::mmap::pread::resolve_backing_fd(src) else {
            return Self::parse(src, max_output);
        };
        let is_v6 = if res.is_chunk_resident(0) {
            v6::is_v6_frame(src)
        } else if src.len() >= HEADER_SIZE {
            let mut h32 = [0u8; HEADER_SIZE];
            crate::mmap::pread::pread_at_exact(fd, file_off, &mut h32)
                .map_err(|e| Error::Backend(e.to_string()))?;
            v6::is_v6_frame(&h32)
        } else {
            false
        };
        if is_v6 {
            return v6::parse_v6_for_query(
                src,
                max_output,
                where_expr,
                extra_cols,
                allow_partial_cols,
                threads,
                res,
            );
        }
        v5::parse_v5_hybrid(src, max_output, fd, file_off, threads, res.clone())
    }
    pub fn raw_size(&self) -> usize {
        self.raw_size
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }
    pub fn checksummed(&self) -> bool {
        self.checksummed
    }
    /// On-wire format version. The encoder emits the current version; decoders accept v1 and v2.
    pub fn version(&self) -> u16 {
        self.version
    }
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = BlockInfo> + '_ {
        self.blocks.iter().map(|b| b.info)
    }
    /// Shared prefix dictionary bytes attached to this frame (`empty` when none).
    #[inline]
    pub fn dict(&self) -> &[u8] {
        &self.dict
    }
    /// Returns `true` when this `VERSION_V5` frame carries an $O(1)$ per-block row-count index (`FLAG_ROW_INDEX`).
    #[inline]
    pub fn has_row_index(&self) -> bool {
        self.row_index.is_some()
    }
    /// Returns `(row_count, ends_with_newline)` for `index` in $O(1)$ time when `FLAG_ROW_INDEX` is present.
    #[inline]
    pub fn block_row_meta(&self, index: usize) -> Option<(usize, bool)> {
        self.row_index
            .as_ref()
            .and_then(|ri| ri.get(index).copied())
            .map(v5::unpack_block_row_meta)
    }
    /// Returns total structured records across this frame in $O(B)$ header arithmetic when `FLAG_ROW_INDEX` is present.
    pub fn indexed_row_count(&self) -> Option<usize> {
        let ri = self.row_index.as_ref()?;
        let mut total = 0usize;
        let mut prev_open = false;
        for &word in ri {
            let (rows, ends_nl) = v5::unpack_block_row_meta(word);
            if rows > 0 {
                total = total.checked_add(if prev_open { rows - 1 } else { rows })?;
                prev_open = !ends_nl;
            }
        }
        Some(total)
    }
    pub fn validate_commands(&self) -> Result<()> {
        for b in &self.blocks {
            crate::columns::with_v6_cols(b.col_slices, || {
                block::validate_with_dict(
                    b.payload,
                    b.info.kind,
                    b.info.raw_size,
                    self.version >= 2,
                    &self.dict,
                )
            })?;
        }
        Ok(())
    }
    /// Consume this frame after validating every command stream. The resulting
    /// handle can skip repeated validation on its fixed-size CPU fast path.
    pub fn into_validated(self) -> Result<ValidatedFrame<'a>> {
        self.validate_commands()?;
        Ok(ValidatedFrame {
            frame: self,
            proof: ValidatedCommandStreams(()),
        })
    }
    /// Preallocated API: avoids zeroing/allocating an output buffer in every benchmark iteration.
    /// On error, output is unspecified/partially modified. Never use it as decoded data.
    pub fn decode_into(&self, out: &mut [u8], options: &DecodeOptions) -> Result<()> {
        decode_frame_into(self, out, options, None)
    }
    pub fn decode_block(&self, index: usize, options: &DecodeOptions) -> Result<Vec<u8>> {
        let b = self
            .blocks
            .get(index)
            .ok_or(Error::InvalidOption("block index"))?;
        let mut out = vec![0; b.info.raw_size];
        self.decode_block_into(index, &mut out, options)?;
        Ok(out)
    }
    /// Preallocated single-block random-access decode into `out[..block_raw_size]`.
    pub fn decode_block_into(
        &self,
        index: usize,
        out: &mut [u8],
        options: &DecodeOptions,
    ) -> Result<usize> {
        options.validate()?;
        let b = self
            .blocks
            .get(index)
            .ok_or(Error::InvalidOption("block index"))?;
        if b.info.raw_size > options.max_output {
            return Err(Error::OutputLimit {
                requested: b.info.raw_size as u64,
                limit: options.max_output,
            });
        }
        if out.len() < b.info.raw_size {
            return Err(Error::OutputSize {
                expected: b.info.raw_size,
                actual: out.len(),
            });
        }
        let dst = &mut out[..b.info.raw_size];
        crate::columns::with_v6_cols(b.col_slices, || {
            block::decode_with_dict(
                b.payload,
                b.info.kind,
                dst,
                Engine::new(options.simd)?,
                self.version >= 2,
                &self.dict,
            )
        })?;
        if self.checksummed && options.verify_checksum && crc32c(dst) != b.info.checksum {
            return Err(Error::Checksum { block: index });
        }
        Ok(b.info.raw_size)
    }
}
impl<'a> ValidatedFrame<'a> {
    pub fn raw_size(&self) -> usize {
        self.frame.raw_size()
    }
    pub fn block_count(&self) -> usize {
        self.frame.block_count()
    }
    pub fn frame(&self) -> &Frame<'a> {
        &self.frame
    }
    /// Preallocated decode after `into_validated`. This is safe: the handle
    /// owns an unforgeable validation proof and keeps the original byte slice
    /// immutably borrowed.
    pub fn decode_into(&self, out: &mut [u8], options: &DecodeOptions) -> Result<()> {
        decode_frame_into(&self.frame, out, options, Some(&self.proof))
    }
}

mod assemble;
mod decode;
pub mod recover;
pub(crate) mod v5;
pub(crate) mod v6;

pub(crate) use assemble::{assemble, parallel_encode_assemble, AssembleDest, Part};
use decode::decode_frame_into;
pub(crate) use decode::{
    decode_validated_dir_multi_thread, decode_validated_dir_single_thread,
    extract_block_validated_into, validate_header_and_directory_with_offsets,
    MAX_STACK_DIR_BLOCKS,
};
pub use recover::{recover_frame, FailedBlock, RecoveredBlock, RecoveryReport};
