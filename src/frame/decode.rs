use crate::{block, checksum::crc32c, simd::Engine, DecodeOptions, Error, Result};

use super::{
    check_block_size, entry_size, read_u32, read_u64, Block, Frame, ValidatedCommandStreams,
    HEADER_SIZE, MAGIC, MAX_VERSION, MIN_VERSION, V4_SIZE_MASK, VERSION_V4,
};

pub(super) fn decode_frame_into(
    frame: &Frame<'_>,
    out: &mut [u8],
    options: &DecodeOptions,
    proof: Option<&ValidatedCommandStreams>,
) -> Result<()> {
    options.validate()?;
    if frame.raw_size > options.max_output {
        return Err(Error::OutputLimit {
            requested: frame.raw_size as u64,
            limit: options.max_output,
        });
    }
    if out.len() != frame.raw_size {
        return Err(Error::OutputSize {
            expected: frame.raw_size,
            actual: out.len(),
        });
    }
    let engine = Engine::new(options.simd)?;
    if frame.blocks.is_empty() {
        return Ok(());
    }
    let workers = options.threads.min(frame.blocks.len());
    let verify = frame.checksummed && options.verify_checksum;
    let short_distance = frame.version >= 2;
    if workers == 1 {
        return if let Some(proof) = proof {
            decode_validated_group(&frame.blocks, out, engine, verify, 0, short_distance, proof)
        } else {
            decode_group(&frame.blocks, out, engine, verify, 0, short_distance)
        };
    }
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let next_idx = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let out_addr = out.as_mut_ptr() as usize;
    let blocks = &frame.blocks[..];
    let batch = (blocks.len() / (workers * 4)).clamp(1, 8);
    std::thread::scope(|scope| {
        let next_ref = &next_idx;
        let abort_ref = &aborted;
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            handles.push(scope.spawn(move || -> Result<()> {
                loop {
                    if abort_ref.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let start = next_ref.fetch_add(batch, Ordering::Relaxed);
                    if start >= blocks.len() {
                        return Ok(());
                    }
                    let end = (start + batch).min(blocks.len());
                    for idx in start..end {
                        let b = &blocks[idx];
                        // SAFETY: `Frame::parse` verified that all blocks `0..blocks.len()` cover
                        // disjoint ranges `b.info.raw_offset .. b.info.raw_offset + b.info.raw_size`
                        // within `0..frame.raw_size` (`== out.len()`), and each `idx` is claimed
                        // by at most one worker via `fetch_add(batch)`.
                        let target = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_addr + b.info.raw_offset) as *mut u8,
                                b.info.raw_size,
                            )
                        };
                        let res = if let Some(_proof) = proof {
                            unsafe {
                                block::decode_validated(
                                    b.payload,
                                    b.info.kind,
                                    target,
                                    engine,
                                    short_distance,
                                )
                            }
                        } else {
                            block::decode(b.payload, b.info.kind, target, engine, short_distance)
                        };
                        if let Err(e) = res {
                            abort_ref.store(true, Ordering::Relaxed);
                            return Err(e);
                        }
                        if verify && crc32c(target) != b.info.checksum {
                            abort_ref.store(true, Ordering::Relaxed);
                            return Err(Error::Checksum { block: idx });
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().map_err(|_| Error::WorkerPanic)??;
        }
        Ok(())
    })
}

pub(crate) const MAX_STACK_DIR_BLOCKS: usize = 512;

pub(crate) fn validate_header_and_directory(
    src: &[u8],
    max_output: usize,
) -> Result<(usize, usize, usize, u16, bool)> {
    validate_header_and_directory_with_offsets(src, max_output, None)
}

pub(crate) fn validate_header_and_directory_with_offsets(
    src: &[u8],
    max_output: usize,
    mut v4_offsets: Option<&mut [u32; MAX_STACK_DIR_BLOCKS + 1]>,
) -> Result<(usize, usize, usize, u16, bool)> {
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
    let record_v4 = version >= VERSION_V4
        && count <= MAX_STACK_DIR_BLOCKS
        && src.len() <= u32::MAX as usize;
    if record_v4 {
        if let Some(ref mut offs) = v4_offsets {
            offs[0] = dir_end as u32;
        }
    }
    let mut next_payload = dir_end;
    let mut next_raw = 0usize;
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
        if end > src.len() {
            return Err(Error::Truncated);
        }
        next_raw += rn;
        next_payload = end;
        if record_v4 {
            if let Some(ref mut offs) = v4_offsets {
                offs[i + 1] = end as u32;
            }
        }
    }
    if next_raw != raw_size || next_payload != src.len() {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    Ok((raw_size, block_size, count, version, flags != 0))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_validated_dir_multi_thread(
    src: &[u8],
    out: &mut [u8],
    raw_size: usize,
    block_size: usize,
    count: usize,
    version: u16,
    checksummed: bool,
    v4_offsets: &[u32; MAX_STACK_DIR_BLOCKS + 1],
    options: &DecodeOptions,
) -> Result<()> {
    if out.len() != raw_size {
        return Err(Error::OutputSize {
            expected: raw_size,
            actual: out.len(),
        });
    }
    let engine = Engine::new(options.simd)?;
    if count == 0 {
        return Ok(());
    }
    let workers = options.threads.min(count);
    if workers <= 1 {
        return decode_validated_dir_single_thread(
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let next_idx = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let out_addr = out.as_mut_ptr() as usize;
    let verify = checksummed && options.verify_checksum;
    let short_distance = version >= 2;
    let esize = entry_size(version);
    let batch = (count / (workers * 4)).clamp(1, 8);
    std::thread::scope(|scope| {
        let mut handles_uninit = [const {
            core::mem::MaybeUninit::<std::thread::ScopedJoinHandle<'_, Result<()>>>::uninit()
        }; 256];
        let next_ref = &next_idx;
        let abort_ref = &aborted;
        for w in 0..workers {
            handles_uninit[w].write(scope.spawn(move || -> Result<()> {
                loop {
                    if abort_ref.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    let start = next_ref.fetch_add(batch, Ordering::Relaxed);
                    if start >= count {
                        return Ok(());
                    }
                    let end = (start + batch).min(count);
                    for idx in start..end {
                        let p = HEADER_SIZE + idx * esize;
                        let raw_offset = idx * block_size;
                        let rn = block_size.min(raw_size - raw_offset);
                        let (payload_offset, cn, checksum, kind) = if version >= VERSION_V4 {
                            let word = unsafe {
                                u32::from_le_bytes(src.get_unchecked(p..p + 4).try_into().unwrap())
                            };
                            let csum = unsafe {
                                u32::from_le_bytes(src.get_unchecked(p + 4..p + 8).try_into().unwrap())
                            };
                            let p_off = unsafe { *v4_offsets.get_unchecked(idx) as usize };
                            let p_end = unsafe { *v4_offsets.get_unchecked(idx + 1) as usize };
                            (p_off, p_end - p_off, csum, word >> 30)
                        } else {
                            let p_off = unsafe {
                                u64::from_le_bytes(src.get_unchecked(p + 8..p + 16).try_into().unwrap()) as usize
                            };
                            let cn = unsafe {
                                u32::from_le_bytes(src.get_unchecked(p + 20..p + 24).try_into().unwrap()) as usize
                            };
                            let csum = unsafe {
                                u32::from_le_bytes(src.get_unchecked(p + 24..p + 28).try_into().unwrap())
                            };
                            let kind = unsafe {
                                u32::from_le_bytes(src.get_unchecked(p + 28..p + 32).try_into().unwrap())
                            };
                            (p_off, cn, csum, kind)
                        };
                        let payload = unsafe { src.get_unchecked(payload_offset..payload_offset + cn) };
                        let target = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_addr + raw_offset) as *mut u8,
                                rn,
                            )
                        };
                        if let Err(e) = block::decode(payload, kind, target, engine, short_distance) {
                            abort_ref.store(true, Ordering::Relaxed);
                            return Err(e);
                        }
                        if verify && crc32c(target) != checksum {
                            abort_ref.store(true, Ordering::Relaxed);
                            return Err(Error::Checksum { block: idx });
                        }
                    }
                }
            }));
        }
        let mut first_err = None;
        for h in handles_uninit.iter_mut().take(workers) {
            let handle = unsafe { h.assume_init_read() };
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(Error::WorkerPanic);
                    }
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        Ok(())
    })
}

pub(crate) fn extract_block_validated_into(
    src: &[u8],
    block_index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    options.validate()?;
    let (raw_size, block_size, count, version, checksummed) =
        validate_header_and_directory(src, usize::MAX)?;
    if block_index >= count {
        return Err(Error::InvalidOption("block index"));
    }
    let esize = entry_size(version);
    let raw_offset = block_index * block_size;
    let rn = block_size.min(raw_size - raw_offset);
    if rn > options.max_output {
        return Err(Error::OutputLimit {
            requested: rn as u64,
            limit: options.max_output,
        });
    }
    if out.len() < rn {
        return Err(Error::OutputSize {
            expected: rn,
            actual: out.len(),
        });
    }
    let (payload_offset, cn, checksum, kind) = if version >= VERSION_V4 {
        let mut p_off = HEADER_SIZE + count * esize;
        for i in 0..block_index {
            let p = HEADER_SIZE + i * esize;
            let word = unsafe {
                u32::from_le_bytes(src.get_unchecked(p..p + 4).try_into().unwrap())
            };
            p_off += (word & V4_SIZE_MASK) as usize;
        }
        let p = HEADER_SIZE + block_index * esize;
        let word = unsafe {
            u32::from_le_bytes(src.get_unchecked(p..p + 4).try_into().unwrap())
        };
        let csum = unsafe {
            u32::from_le_bytes(src.get_unchecked(p + 4..p + 8).try_into().unwrap())
        };
        (p_off, (word & V4_SIZE_MASK) as usize, csum, word >> 30)
    } else {
        let p = HEADER_SIZE + block_index * esize;
        let p_off = unsafe {
            u64::from_le_bytes(src.get_unchecked(p + 8..p + 16).try_into().unwrap()) as usize
        };
        let cn = unsafe {
            u32::from_le_bytes(src.get_unchecked(p + 20..p + 24).try_into().unwrap()) as usize
        };
        let csum = unsafe {
            u32::from_le_bytes(src.get_unchecked(p + 24..p + 28).try_into().unwrap())
        };
        let kind = unsafe {
            u32::from_le_bytes(src.get_unchecked(p + 28..p + 32).try_into().unwrap())
        };
        (p_off, cn, csum, kind)
    };
    let payload = unsafe { src.get_unchecked(payload_offset..payload_offset + cn) };
    let dst = &mut out[..rn];
    block::decode(
        payload,
        kind,
        dst,
        Engine::new(options.simd)?,
        version >= 2,
    )?;
    if checksummed && options.verify_checksum && crc32c(dst) != checksum {
        return Err(Error::Checksum { block: block_index });
    }
    Ok(rn)
}

pub(crate) fn decode_validated_dir_single_thread(
    src: &[u8],
    out: &mut [u8],
    raw_size: usize,
    block_size: usize,
    count: usize,
    version: u16,
    checksummed: bool,
    options: &DecodeOptions,
) -> Result<()> {
    if out.len() != raw_size {
        return Err(Error::OutputSize {
            expected: raw_size,
            actual: out.len(),
        });
    }
    let engine = Engine::new(options.simd)?;
    if count == 0 {
        return Ok(());
    }
    let esize = entry_size(version);
    let verify = checksummed && options.verify_checksum;
    let short_distance = version >= 2;
    let mut next_payload = HEADER_SIZE + count * esize;
    let mut next_raw = 0usize;
    for i in 0..count {
        let p = HEADER_SIZE + i * esize;
        let (rn, cn, checksum, kind) = if version >= VERSION_V4 {
            let word = unsafe {
                u32::from_le_bytes(src.get_unchecked(p..p + 4).try_into().unwrap())
            };
            let csum = unsafe {
                u32::from_le_bytes(src.get_unchecked(p + 4..p + 8).try_into().unwrap())
            };
            (
                block_size.min(raw_size - next_raw),
                (word & V4_SIZE_MASK) as usize,
                csum,
                word >> 30,
            )
        } else {
            let rn = unsafe {
                u32::from_le_bytes(src.get_unchecked(p + 16..p + 20).try_into().unwrap()) as usize
            };
            let cn = unsafe {
                u32::from_le_bytes(src.get_unchecked(p + 20..p + 24).try_into().unwrap()) as usize
            };
            let csum = unsafe {
                u32::from_le_bytes(src.get_unchecked(p + 24..p + 28).try_into().unwrap())
            };
            let kind = unsafe {
                u32::from_le_bytes(src.get_unchecked(p + 28..p + 32).try_into().unwrap())
            };
            (rn, cn, csum, kind)
        };
        let payload = unsafe { src.get_unchecked(next_payload..next_payload + cn) };
        let target = unsafe { out.get_unchecked_mut(next_raw..next_raw + rn) };
        block::decode(payload, kind, target, engine, short_distance)?;
        if verify && crc32c(target) != checksum {
            return Err(Error::Checksum { block: i });
        }
        next_raw += rn;
        next_payload += cn;
    }
    Ok(())
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

fn decode_validated_group(
    blocks: &[Block<'_>],
    out: &mut [u8],
    engine: Engine,
    verify: bool,
    first: usize,
    short_distance: bool,
    _proof: &ValidatedCommandStreams,
) -> Result<()> {
    let mut offset = 0;
    for (i, b) in blocks.iter().enumerate() {
        let target = &mut out[offset..offset + b.info.raw_size];
        // SAFETY: `_proof` can only be constructed by `Frame::into_validated`
        // after it validates every immutable payload in this frame. `blocks`
        // comes directly from that same frame and `target` has the recorded
        // raw length, so the contract of `block::decode_validated` holds.
        unsafe {
            block::decode_validated(b.payload, b.info.kind, target, engine, short_distance)?;
        }
        if verify && crc32c(target) != b.info.checksum {
            return Err(Error::Checksum { block: first + i });
        }
        offset += b.info.raw_size;
    }
    Ok(())
}
