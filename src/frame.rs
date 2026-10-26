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
/// Version emitted by default.
pub const VERSION_RELEASE: u16 = VERSION_V4;
pub const ENTRY_SIZE_V4: usize = 8;
const MIN_VERSION: u16 = 1;
const MAX_VERSION: u16 = VERSION_V4;
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
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArenaPart {
    pub index: usize,
    pub offset: usize,
    pub len: usize,
    pub kind: u32,
    pub checksum: u32,
    pub raw_size: usize,
}
#[derive(Debug)]
pub(crate) struct WorkerOutput {
    pub arena: Vec<u8>,
    pub parts: Vec<ArenaPart>,
    pub stats: block::WorkStats,
}
#[derive(Debug)]
pub struct Frame<'a> {
    pub(crate) blocks: Vec<Block<'a>>,
    pub(crate) raw_size: usize,
    pub(crate) block_size: usize,
    pub(crate) checksummed: bool,
    pub(crate) version: u16,
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
impl<'a> Frame<'a> {
    /// Validates the header and directory before any output allocation.
    /// Command validation is fused with CPU decoding, or explicitly requested by GPU callers.
    pub fn parse(src: &'a [u8], max_output: usize) -> Result<Self> {
        if src.len() < HEADER_SIZE {
            return Err(Error::Truncated);
        }
        let version = u16::from_le_bytes(src[4..6].try_into().unwrap());
        if &src[..4] != MAGIC || !(MIN_VERSION..=MAX_VERSION).contains(&version) {
            return Err(Error::InvalidFrame("magic/version"));
        }
        let flags = u16::from_le_bytes(src[6..8].try_into().unwrap());
        if flags & !1 != 0 || read_u64(src, 24)? != 0 {
            return Err(Error::InvalidFrame("unknown flags/reserved data"));
        }
        let raw64 = read_u64(src, 8)?;
        if raw64 > max_output as u64 {
            return Err(Error::OutputLimit {
                requested: raw64,
                limit: max_output,
            });
        }
        let raw_size = usize::try_from(raw64).map_err(|_| Error::SizeOverflow)?;
        let block_size = read_u32(src, 16)? as usize;
        check_block_size(block_size).map_err(|_| Error::InvalidFrame("block size"))?;
        let count = read_u32(src, 20)? as usize;
        let expected_count = raw_size / block_size + usize::from(raw_size % block_size != 0);
        if count != expected_count {
            return Err(Error::InvalidFrame("block count"));
        }
        let esize = entry_size(version);
        let dir_end = count
            .checked_mul(esize)
            .and_then(|v| v.checked_add(HEADER_SIZE))
            .ok_or(Error::SizeOverflow)?;
        if dir_end > src.len() {
            return Err(Error::Truncated);
        }
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(count)
            .map_err(|_| Error::SizeOverflow)?;
        let mut next_payload = dir_end;
        let mut next_raw = 0;
        for i in 0..count {
            let p = HEADER_SIZE + i * esize;
            let (raw_offset, payload_offset, rn, cn, checksum, kind) = if version >= VERSION_V4 {
                let word = read_u32(src, p)?;
                (
                    next_raw,
                    next_payload,
                    block_size.min(raw_size - next_raw),
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
            if raw_offset != next_raw
                || payload_offset != next_payload
                || rn != block_size.min(raw_size - next_raw)
            {
                return Err(Error::InvalidFrame("non-canonical directory"));
            }
            let kind_size_ok = match kind {
                block::RAW => cn == rn,
                block::LZ => cn < rn && cn >= block::BLOCK_PREFIX,
                block::COLUMN => version >= 3 && cn >= 2 && cn < rn,
                _ => false,
            };
            if !kind_size_ok {
                return Err(Error::InvalidFrame("block kind/size"));
            }
            if flags == 0 && checksum != 0 {
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
            });
            next_raw += rn;
            next_payload = end;
        }
        if next_raw != raw_size || next_payload != src.len() {
            return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
        }
        Ok(Self {
            blocks,
            raw_size,
            block_size,
            checksummed: flags != 0,
            version,
        })
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
    pub fn validate_commands(&self) -> Result<()> {
        for b in &self.blocks {
            block::validate(b.payload, b.info.kind, b.info.raw_size, self.version >= 2)?;
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
        block::decode(
            b.payload,
            b.info.kind,
            dst,
            Engine::new(options.simd)?,
            self.version >= 2,
        )?;
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

pub(crate) use assemble::{assemble, assemble_workers, assemble_workers_into, Part};
use decode::decode_frame_into;
pub(crate) use decode::{
    decode_validated_dir_multi_thread, decode_validated_dir_single_thread,
    extract_block_validated_into, validate_header_and_directory_with_offsets,
    MAX_STACK_DIR_BLOCKS,
};
