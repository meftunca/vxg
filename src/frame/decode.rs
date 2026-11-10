use crate::{block, checksum::crc32c, simd::Engine, DecodeOptions, Error, Result};

use super::{
    entry_size, read_u32, read_u64, v5, v6, valid_kind_and_size, Block, Frame,
    ValidatedCommandStreams, HEADER_SIZE, V4_SIZE_MASK, VERSION_V4, VERSION_V6,
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
    let dict_ref = frame.dict.as_slice();
    if workers == 1 {
        let prev = crate::columns::v6_verify_slice_crc();
        crate::columns::set_v6_verify_slice_crc(verify);
        let res = if let Some(proof) = proof {
            decode_validated_group(
                &frame.blocks,
                out,
                engine,
                verify,
                0,
                short_distance,
                dict_ref,
                proof,
            )
        } else {
            decode_group(&frame.blocks, out, engine, verify, 0, short_distance, dict_ref)
        };
        crate::columns::set_v6_verify_slice_crc(prev);
        return res;
    }
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let next_idx = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let out_addr = out.as_mut_ptr() as usize;
    let blocks = &frame.blocks[..];
    let max_batch = (blocks.len() / (workers * 8)).clamp(1, 4);
    let tail_cutoff = blocks.len().saturating_sub(workers * 4);
    crate::pool::parallel_try_run(workers, |_| {
        let prev_v6 = crate::columns::v6_verify_slice_crc();
        crate::columns::set_v6_verify_slice_crc(verify);
        let run_res = (|| -> Result<()> {
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
                for idx in start..end {
                    let b = &blocks[idx];
                    let target = unsafe {
                        std::slice::from_raw_parts_mut(
                            (out_addr + b.info.raw_offset) as *mut u8,
                            b.info.raw_size,
                        )
                    };
                    let res = crate::columns::with_v6_cols(b.col_slices, || {
                        if let Some(_proof) = proof {
                            unsafe {
                                block::decode_validated_with_dict(
                                    b.payload,
                                    b.info.kind,
                                    target,
                                    engine,
                                    short_distance,
                                    dict_ref,
                                )
                            }
                        } else {
                            block::decode_with_dict(
                                b.payload,
                                b.info.kind,
                                target,
                                engine,
                                short_distance,
                                dict_ref,
                            )
                        }
                    });
                    if let Err(e) = res {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(match e {
                            Error::Checksum { .. } => Error::Checksum { block: idx },
                            other => other,
                        });
                    }
                    if verify && crc32c(target) != b.info.checksum {
                        aborted.store(true, Ordering::Relaxed);
                        return Err(Error::Checksum { block: idx });
                    }
                }
            }
        })();
        crate::columns::set_v6_verify_slice_crc(prev_v6);
        run_res
    })
}

pub(crate) const MAX_STACK_DIR_BLOCKS: usize = 4096;

pub(crate) fn validate_header_and_directory(
    src: &[u8],
    max_output: usize,
) -> Result<(usize, usize, usize, u16, bool)> {
    validate_header_and_directory_with_offsets(src, max_output, None)
}

pub(crate) fn validate_header_and_directory_with_offsets(
    src: &[u8],
    max_output: usize,
    mut v4_offsets: Option<&mut [core::mem::MaybeUninit<u32>; MAX_STACK_DIR_BLOCKS + 1]>,
) -> Result<(usize, usize, usize, u16, bool)> {
    let hdr = v5::parse_v1_to_v5_header(src, max_output)?;
    if hdr.payload_start > src.len() {
        return Err(Error::Truncated);
    }
    if hdr.version == VERSION_V6 {
        let (total_frame, _) = v6::validate_v6_header_and_dir(src, &hdr)?;
        if src.len() < total_frame {
            return Err(Error::Truncated);
        }
        if src.len() != total_frame {
            return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
        }
        return Ok((
            hdr.raw_size,
            hdr.block_size,
            hdr.count,
            hdr.version,
            hdr.checksummed,
        ));
    }
    let esize = entry_size(hdr.version);
    let record_v4 = hdr.version >= VERSION_V4
        && !hdr.has_var_raw
        && hdr.count <= MAX_STACK_DIR_BLOCKS
        && src.len() <= u32::MAX as usize;
    if record_v4 {
        if let Some(ref mut offs) = v4_offsets {
            offs[0].write(hdr.payload_start as u32);
        }
    }
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
        let end = payload_offset.checked_add(cn).ok_or(Error::SizeOverflow)?;
        if end > src.len() {
            return Err(Error::Truncated);
        }
        next_raw += rn;
        next_payload = end;
        if record_v4 {
            if let Some(ref mut offs) = v4_offsets {
                offs[i + 1].write(end as u32);
            }
        }
    }
    if next_raw != hdr.raw_size || next_payload != src.len() {
        return Err(Error::InvalidFrame("trailing bytes or incorrect total"));
    }
    Ok((
        hdr.raw_size,
        hdr.block_size,
        hdr.count,
        hdr.version,
        hdr.checksummed,
    ))
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
    v4_offsets: &[core::mem::MaybeUninit<u32>; MAX_STACK_DIR_BLOCKS + 1],
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
    let hdr = v5::parse_v1_to_v5_header(src, usize::MAX)?;
    let dict = v5::decode_header_dict(src, &hdr)?;
    let dict_ref = dict.as_slice();
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let next_idx = AtomicUsize::new(0);
    let aborted = AtomicBool::new(false);
    let out_addr = out.as_mut_ptr() as usize;
    let verify = checksummed && options.verify_checksum;
    let short_distance = version >= 2;
    let esize = entry_size(version);
    let max_batch = (count / (workers * 8)).clamp(1, 4);
    let tail_cutoff = count.saturating_sub(workers * 4);
    crate::pool::parallel_try_run(workers, |_| {
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
            if start >= count {
                return Ok(());
            }
            let end = (start + step).min(count);
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
                    let p_off = unsafe { v4_offsets.get_unchecked(idx).assume_init() as usize };
                    let p_end =
                        unsafe { v4_offsets.get_unchecked(idx + 1).assume_init() as usize };
                    (p_off, p_end - p_off, csum, word >> 30)
                } else {
                    let p_off = unsafe {
                        u64::from_le_bytes(src.get_unchecked(p + 8..p + 16).try_into().unwrap())
                            as usize
                    };
                    let cn = unsafe {
                        u32::from_le_bytes(
                            src.get_unchecked(p + 20..p + 24).try_into().unwrap(),
                        ) as usize
                    };
                    let csum = unsafe {
                        u32::from_le_bytes(
                            src.get_unchecked(p + 24..p + 28).try_into().unwrap(),
                        )
                    };
                    let kind = unsafe {
                        u32::from_le_bytes(
                            src.get_unchecked(p + 28..p + 32).try_into().unwrap(),
                        )
                    };
                    (p_off, cn, csum, kind)
                };
                let payload = unsafe { src.get_unchecked(payload_offset..payload_offset + cn) };
                let target = unsafe {
                    std::slice::from_raw_parts_mut((out_addr + raw_offset) as *mut u8, rn)
                };
                if let Err(e) =
                    block::decode_with_dict(payload, kind, target, engine, short_distance, dict_ref)
                {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(e);
                }
                if verify && crc32c(target) != checksum {
                    aborted.store(true, Ordering::Relaxed);
                    return Err(Error::Checksum { block: idx });
                }
            }
        }
    })
}

pub(crate) fn extract_block_validated_into(
    src: &[u8],
    block_index: usize,
    out: &mut [u8],
    options: &DecodeOptions,
) -> Result<usize> {
    if let Some(res) = v6::try_extract_block_v6_into(src, block_index, out, options) {
        return res;
    }
    options.validate()?;
    let (_raw_size, block_size, count, version, checksummed) =
        validate_header_and_directory(src, usize::MAX)?;
    if block_index >= count {
        return Err(Error::InvalidOption("block index"));
    }
    let hdr = v5::parse_v1_to_v5_header(src, usize::MAX)?;
    let esize = entry_size(version);
    let mut raw_offset = 0usize;
    let rn = if hdr.has_var_raw {
        for i in 0..block_index {
            raw_offset += v5::block_raw_size(src, &hdr, i, raw_offset)?;
        }
        v5::block_raw_size(src, &hdr, block_index, raw_offset)?
    } else {
        raw_offset = block_index * block_size;
        block_size.min(hdr.raw_size - raw_offset)
    };
    let _ = raw_offset;
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
        let mut p_off = hdr.payload_start;
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
    let dict = if kind == block::DICT_LZ && hdr.has_dict {
        v5::decode_header_dict(src, &hdr)?
    } else {
        Vec::new()
    };
    block::decode_with_dict(
        payload,
        kind,
        dst,
        Engine::new(options.simd)?,
        version >= 2,
        &dict,
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
    let hdr = v5::parse_v1_to_v5_header(src, usize::MAX)?;
    let dict = v5::decode_header_dict(src, &hdr)?;
    let dict_ref = dict.as_slice();
    let esize = entry_size(version);
    let verify = checksummed && options.verify_checksum;
    let short_distance = version >= 2;
    let mut next_payload = hdr.payload_start;
    let mut next_raw = 0usize;
    let _ = block_size;
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
                v5::block_raw_size(src, &hdr, i, next_raw)?,
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
        block::decode_with_dict(payload, kind, target, engine, short_distance, dict_ref)?;
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
    dict: &[u8],
) -> Result<()> {
    let mut offset = 0;
    for (i, b) in blocks.iter().enumerate() {
        let target = &mut out[offset..offset + b.info.raw_size];
        crate::columns::with_v6_cols(b.col_slices, || {
            block::decode_with_dict(b.payload, b.info.kind, target, engine, short_distance, dict)
        })
        .map_err(|e| match e {
            Error::Checksum { .. } => Error::Checksum { block: first + i },
            other => other,
        })?;
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
    dict: &[u8],
    _proof: &ValidatedCommandStreams,
) -> Result<()> {
    let mut offset = 0;
    for (i, b) in blocks.iter().enumerate() {
        let target = &mut out[offset..offset + b.info.raw_size];
        crate::columns::with_v6_cols(b.col_slices, || unsafe {
            block::decode_validated_with_dict(
                b.payload,
                b.info.kind,
                target,
                engine,
                short_distance,
                dict,
            )
        })?;
        if verify && crc32c(target) != b.info.checksum {
            return Err(Error::Checksum { block: first + i });
        }
        offset += b.info.raw_size;
    }
    Ok(())
}
