use crate::{block, checksum::crc32c, simd::Engine, DecodeOptions, Error, Result};

pub const HEADER_SIZE: usize = 32;
pub const ENTRY_SIZE: usize = 32;
pub const MAGIC: &[u8; 4] = b"VXG1";
/// v2 adds opcode 7: a two-byte absolute match distance. v1 frames remain readable.
pub const VERSION: u16 = 2;
const MIN_VERSION: u16 = 1;

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
        if &src[..4] != MAGIC || !(MIN_VERSION..=VERSION).contains(&version) {
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
        let dir_end = count
            .checked_mul(ENTRY_SIZE)
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
            let p = HEADER_SIZE + i * ENTRY_SIZE;
            let raw_offset = usize::try_from(read_u64(src, p)?).map_err(|_| Error::SizeOverflow)?;
            let payload_offset =
                usize::try_from(read_u64(src, p + 8)?).map_err(|_| Error::SizeOverflow)?;
            let rn = read_u32(src, p + 16)? as usize;
            let cn = read_u32(src, p + 20)? as usize;
            let checksum = read_u32(src, p + 24)?;
            let kind = read_u32(src, p + 28)?;
            if raw_offset != next_raw
                || payload_offset != next_payload
                || rn != block_size.min(raw_size - next_raw)
            {
                return Err(Error::InvalidFrame("non-canonical directory"));
            }
            if (kind == block::RAW && cn != rn)
                || (kind == block::LZ && (cn >= rn || cn < block::BLOCK_PREFIX))
                || kind > block::LZ
            {
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
    /// Preallocated API: avoids zeroing/allocating an output buffer in every benchmark iteration.
    /// On error, output is unspecified/partially modified. Never use it as decoded data.
    pub fn decode_into(&self, out: &mut [u8], options: &DecodeOptions) -> Result<()> {
        options.validate()?;
        if self.raw_size > options.max_output {
            return Err(Error::OutputLimit {
                requested: self.raw_size as u64,
                limit: options.max_output,
            });
        }
        if out.len() != self.raw_size {
            return Err(Error::OutputSize {
                expected: self.raw_size,
                actual: out.len(),
            });
        }
        let engine = Engine::new(options.simd)?;
        if self.blocks.is_empty() {
            return Ok(());
        }
        let workers = options.threads.min(self.blocks.len());
        let per_worker = self.blocks.len().div_ceil(workers);
        let output_stride = per_worker
            .checked_mul(self.block_size)
            .ok_or(Error::SizeOverflow)?;
        let verify = self.checksummed && options.verify_checksum;
        if workers == 1 {
            return decode_group(&self.blocks, out, engine, verify, 0, self.version >= 2);
        }
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for (group, target) in out.chunks_mut(output_stride).enumerate() {
                let start = group * per_worker;
                let blocks = &self.blocks[start..(start + per_worker).min(self.blocks.len())];
                let short_distance = self.version >= 2;
                handles.push(scope.spawn(move || {
                    decode_group(blocks, target, engine, verify, start, short_distance)
                }));
            }
            for h in handles {
                h.join().map_err(|_| Error::WorkerPanic)??;
            }
            Ok(())
        })
    }
    pub fn decode_block(&self, index: usize, options: &DecodeOptions) -> Result<Vec<u8>> {
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
        let mut out = vec![0; b.info.raw_size];
        block::decode(
            b.payload,
            b.info.kind,
            &mut out,
            Engine::new(options.simd)?,
            self.version >= 2,
        )?;
        if self.checksummed && options.verify_checksum && crc32c(&out) != b.info.checksum {
            return Err(Error::Checksum { block: index });
        }
        Ok(out)
    }
}
fn decode_group(
    blocks: &[Block<'_>],
    out: &mut [u8],
    engine: Engine,
    verify: bool,
    first: usize,
    short_distance: bool,
) -> Result<()> {
    let mut offset = 0;
    for (i, b) in blocks.iter().enumerate() {
        let target = &mut out[offset..offset + b.info.raw_size];
        block::decode(b.payload, b.info.kind, target, engine, short_distance)?;
        if verify && crc32c(target) != b.info.checksum {
            return Err(Error::Checksum { block: first + i });
        }
        offset += b.info.raw_size;
    }
    Ok(())
}
pub(crate) struct Part {
    pub bytes: Vec<u8>,
    pub kind: u32,
    pub checksum: u32,
    pub raw_size: usize,
}
pub(crate) fn assemble_workers(
    workers: Vec<WorkerOutput>,
    n: usize,
    block_size: usize,
    checksummed: bool,
) -> Result<Vec<u8>> {
    let bound = max_compressed_size(n, block_size)?;
    let count = workers.iter().map(|w| w.parts.len()).sum::<usize>();
    let mut refs = Vec::with_capacity(count);
    for (worker, output) in workers.iter().enumerate() {
        for part in &output.parts {
            let end = part
                .offset
                .checked_add(part.len)
                .ok_or(Error::SizeOverflow)?;
            if output.arena.get(part.offset..end).is_none() || part.len > part.raw_size {
                return Err(Error::InvalidFrame("encoder arena part"));
            }
            refs.push((worker, *part));
        }
    }
    refs.sort_unstable_by_key(|(_, part)| part.index);
    if refs
        .iter()
        .enumerate()
        .any(|(index, (_, part))| part.index != index)
    {
        return Err(Error::InvalidFrame("encoder block order"));
    }
    let table_end = HEADER_SIZE
        .checked_add(ENTRY_SIZE.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(bound)
        .map_err(|_| Error::SizeOverflow)?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(u16::from(checksummed)).to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(block_size as u32).to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for (_, part) in &refs {
        out.extend_from_slice(&(raw_offset as u64).to_le_bytes());
        out.extend_from_slice(&(payload_offset as u64).to_le_bytes());
        out.extend_from_slice(&(part.raw_size as u32).to_le_bytes());
        out.extend_from_slice(&(part.len as u32).to_le_bytes());
        out.extend_from_slice(&part.checksum.to_le_bytes());
        out.extend_from_slice(&part.kind.to_le_bytes());
        raw_offset = raw_offset
            .checked_add(part.raw_size)
            .ok_or(Error::SizeOverflow)?;
        payload_offset = payload_offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
    }
    if raw_offset != n {
        return Err(Error::InvalidFrame("encoder total"));
    }
    for (worker, part) in refs {
        let end = part
            .offset
            .checked_add(part.len)
            .ok_or(Error::SizeOverflow)?;
        out.extend_from_slice(&workers[worker].arena[part.offset..end]);
    }
    debug_assert!(out.len() <= bound);
    Ok(out)
}
pub(crate) fn assemble(
    parts: Vec<Part>,
    n: usize,
    block_size: usize,
    checksummed: bool,
) -> Result<Vec<u8>> {
    let bound = max_compressed_size(n, block_size)?;
    let mut out = Vec::new();
    out.try_reserve_exact(bound)
        .map_err(|_| Error::SizeOverflow)?;
    let count = parts.len();
    let table_end = HEADER_SIZE + ENTRY_SIZE * count;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(u16::from(checksummed)).to_le_bytes());
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(block_size as u32).to_le_bytes());
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    let mut raw_offset = 0usize;
    let mut payload_offset = table_end;
    for p in &parts {
        if p.bytes.len() > p.raw_size {
            return Err(Error::InvalidFrame("encoder expansion"));
        }
        out.extend_from_slice(&(raw_offset as u64).to_le_bytes());
        out.extend_from_slice(&(payload_offset as u64).to_le_bytes());
        out.extend_from_slice(&(p.raw_size as u32).to_le_bytes());
        out.extend_from_slice(&(p.bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&p.checksum.to_le_bytes());
        out.extend_from_slice(&p.kind.to_le_bytes());
        raw_offset += p.raw_size;
        payload_offset += p.bytes.len();
    }
    if raw_offset != n {
        return Err(Error::InvalidFrame("encoder total"));
    }
    for p in parts {
        out.extend_from_slice(&p.bytes);
    }
    debug_assert!(out.len() <= bound);
    Ok(out)
}
