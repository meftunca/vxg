use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    block, checksum, columns, dict_lz, frame, max_compressed_size, simd, Error, Options, Result,
    WorkStats,
};

thread_local! {
    pub(crate) static ENCODE_SCRATCH_POOL: RefCell<block::Scratch> =
        RefCell::new(block::Scratch::new());
    static WORKER_PARTS_POOL: RefCell<Vec<frame::ArenaPart>> =
        const { RefCell::new(Vec::new()) };
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn encode_one_chunk<const STATS: bool>(
    chunk: &[u8],
    leading_carry: bool,
    adaptive: bool,
    prep_dict: Option<&dict_lz::PreparedDict>,
    scratch: &mut block::Scratch,
    engine: simd::Engine,
    acceleration: usize,
    stats: &mut WorkStats,
) -> Result<(u32, usize, usize)> {
    if !adaptive {
        let enc = block::encode::<STATS>(chunk, scratch, engine, acceleration);
        if STATS {
            *stats += enc.stats;
        }
        return Ok((enc.kind, enc.offset, enc.len));
    }

    if let Some(d) = prep_dict {
        if d.prefer_over_grammar {
            if let Some((dlz_off, dlz_len)) =
                dict_lz::encode_dict_block_into(chunk, Some(d), scratch.payloads_mut(), chunk.len())
            {
                if !d.try_structured || dlz_len * 6 <= chunk.len() {
                    return Ok((block::DICT_LZ, dlz_off, dlz_len));
                }
                if let Some((col_off, col_len)) =
                    columns::encode_structured_only_into(chunk, leading_carry, scratch.payloads_mut())?
                {
                    let payloads = scratch.payloads_mut();
                    if col_len < dlz_len {
                        payloads.copy_within(col_off..col_off + col_len, dlz_off);
                        payloads.truncate(dlz_off + col_len);
                        return Ok((block::COLUMN, dlz_off, col_len));
                    }
                    payloads.truncate(dlz_off + dlz_len);
                }
                return Ok((block::DICT_LZ, dlz_off, dlz_len));
            }
        } else {
            if let Some((col_off, col_len)) =
                columns::encode_structured_only_into(chunk, leading_carry, scratch.payloads_mut())?
            {
                if col_len * 7 <= chunk.len() {
                    return Ok((block::COLUMN, col_off, col_len));
                }
                if let Some((dlz_off, dlz_len)) =
                    dict_lz::encode_dict_block_into(chunk, Some(d), scratch.payloads_mut(), col_len)
                {
                    let payloads = scratch.payloads_mut();
                    payloads.copy_within(dlz_off..dlz_off + dlz_len, col_off);
                    payloads.truncate(col_off + dlz_len);
                    return Ok((block::DICT_LZ, col_off, dlz_len));
                }
                return Ok((block::COLUMN, col_off, col_len));
            }
            if let Some((dlz_off, dlz_len)) =
                dict_lz::encode_dict_block_into(chunk, Some(d), scratch.payloads_mut(), chunk.len())
            {
                return Ok((block::DICT_LZ, dlz_off, dlz_len));
            }
        }
        let payloads = scratch.payloads_mut();
        let p_offset = payloads.len();
        payloads.extend_from_slice(chunk);
        return Ok((block::RAW, p_offset, chunk.len()));
    }

    match columns::encode_grammar_block_into_with_carry(chunk, leading_carry, scratch.payloads_mut())? {
        Some((packet_offset, packet_len)) => Ok((block::COLUMN, packet_offset, packet_len)),
        None if (512..=65536).contains(&chunk.len()) => {
            let payloads = scratch.payloads_mut();
            let p_offset = payloads.len();
            payloads.extend_from_slice(chunk);
            Ok((block::RAW, p_offset, chunk.len()))
        }
        None => {
            let enc = block::encode::<STATS>(chunk, scratch, engine, acceleration);
            if STATS {
                *stats += enc.stats;
            }
            Ok((enc.kind, enc.offset, enc.len))
        }
    }
}

#[inline]
fn slice_for_index<'a>(
    input: &'a [u8],
    block_size: usize,
    ranges: Option<&[(usize, usize)]>,
    index: usize,
) -> (usize, &'a [u8]) {
    if let Some(rs) = ranges {
        let (s, e) = rs[index];
        (s, &input[s..e])
    } else {
        let s = index * block_size;
        let e = (s + block_size).min(input.len());
        (s, &input[s..e])
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_single_core<const STATS: bool>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    version: u16,
    ranges: Option<&[(usize, usize)]>,
    prep_dict: Option<&dict_lz::PreparedDict>,
    mut out_slice: Option<&mut [u8]>,
) -> Result<(Vec<u8>, usize, WorkStats)> {
    let bound = max_compressed_size(input.len(), opt.block_size)?;
    let adaptive = !opt.legacy_v2;
    let has_var_raw = ranges.is_some();
    let esize = frame::entry_size(version);
    let base_dir_end = frame::HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    let var_raw_bytes = if has_var_raw {
        count.checked_mul(4).ok_or(Error::SizeOverflow)?
    } else {
        0
    };
    let max_table_end = base_dir_end
        .checked_add(var_raw_bytes)
        .and_then(|x| x.checked_add(prep_dict.map_or(0, |d| d.wire_payload.len())))
        .ok_or(Error::SizeOverflow)?;

    ENCODE_SCRATCH_POOL.with(|cell| -> Result<(Vec<u8>, usize, WorkStats)> {
        let mut scratch = cell.borrow_mut();
        scratch.payloads_mut().clear();
        let payload_reserve = if adaptive {
            (input.len() / 12).max(opt.block_size.min(input.len()))
        } else {
            (input.len() / 4).max(opt.block_size.min(input.len()))
        };
        scratch.reserve_payloads_for(max_table_end.saturating_add(payload_reserve));
        scratch.payloads_mut().resize(max_table_end, 0);

        let mut stats = WorkStats::default();
        let mut any_dict_lz = false;
        let mut meta_entries: Vec<(usize, usize, usize, u32, u32)> = Vec::with_capacity(count);

        for index in 0..count {
            let (raw_offset, chunk) = slice_for_index(input, opt.block_size, ranges, index);
            if index > 0 {
                let payloads = scratch.payloads_mut();
                let cur_len = payloads.len();
                let written = cur_len - max_table_end;
                if cur_len + chunk.len() + 1024 > payloads.capacity() {
                    let rem_raw = input.len() - raw_offset;
                    let est_rem = if written * 2 >= raw_offset {
                        rem_raw + (count - index) * 64
                    } else {
                        ((rem_raw as u64 * written as u64) / (raw_offset as u64).max(1)) as usize
                            + opt.block_size * 2
                    };
                    payloads.reserve(est_rem.max(chunk.len() + 1024));
                }
            }
            let leading_carry = raw_offset > 0 && input[raw_offset - 1] != b'\n';
            let (kind, offset, len) = encode_one_chunk::<STATS>(
                chunk,
                leading_carry,
                adaptive,
                prep_dict,
                &mut scratch,
                engine,
                opt.acceleration,
                &mut stats,
            )?;
            if kind == block::DICT_LZ {
                any_dict_lz = true;
            }
            let csum = if opt.checksum {
                checksum::crc32c(chunk)
            } else {
                0
            };
            meta_entries.push((raw_offset, offset, len, csum, kind));
        }

        let active_dict = if any_dict_lz {
            prep_dict.map(|d| (d.wire_payload.as_slice(), d.raw.len()))
        } else {
            None
        };
        let actual_version = if version < frame::VERSION_V4 {
            version
        } else if active_dict.is_some() || has_var_raw {
            frame::VERSION_V5
        } else {
            frame::VERSION_V4
        };
        let actual_table_end = base_dir_end
            + var_raw_bytes
            + active_dict.map_or(0, |(w, _)| w.len());
        let payloads = scratch.payloads_mut();
        let total_payload_len = payloads.len() - max_table_end;
        let exact_len = actual_table_end + total_payload_len;
        if active_dict.is_some() && exact_len > bound {
            return Err(Error::InvalidFrame("dict_expansion"));
        }
        if actual_table_end < max_table_end && total_payload_len > 0 {
            payloads.copy_within(max_table_end..max_table_end + total_payload_len, actual_table_end);
        }
        payloads.truncate(exact_len);

        frame::v5::write_header_bytes(
            payloads,
            actual_version,
            opt.checksum,
            has_var_raw,
            active_dict,
            input.len(),
            opt.block_size,
            count,
        );
        let shift = max_table_end - actual_table_end;
        for (i, &(raw_off, old_p_off, len, csum, kind)) in meta_entries.iter().enumerate() {
            let (_s, chunk) = slice_for_index(input, opt.block_size, ranges, i);
            frame::write_entry(
                payloads,
                frame::HEADER_SIZE + i * esize,
                actual_version,
                raw_off,
                old_p_off - shift,
                chunk.len(),
                len,
                csum,
                kind,
            );
            if has_var_raw {
                let vr_pos = base_dir_end + i * 4;
                payloads[vr_pos..vr_pos + 4]
                    .copy_from_slice(&(chunk.len() as u32).to_le_bytes());
            }
        }
        if let Some((wire, _)) = active_dict {
            let d_pos = base_dir_end + var_raw_bytes;
            payloads[d_pos..d_pos + wire.len()].copy_from_slice(wire);
        }

        if let Some(ref mut dst) = out_slice {
            if dst.len() < exact_len {
                return Err(Error::OutputSize {
                    expected: exact_len,
                    actual: dst.len(),
                });
            }
            dst[..exact_len].copy_from_slice(payloads);
            Ok((Vec::new(), exact_len, stats))
        } else {
            let mut out = Vec::with_capacity(exact_len);
            out.extend_from_slice(payloads);
            Ok((out, exact_len, stats))
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_single_into(
    input: &[u8],
    out: &mut [u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    version: u16,
    ranges: Option<&[(usize, usize)]>,
    prep_dict: Option<&dict_lz::PreparedDict>,
) -> Result<usize> {
    let (_, written, _) = encode_single_core::<false>(
        input,
        opt,
        engine,
        count,
        version,
        ranges,
        prep_dict,
        Some(out),
    )?;
    Ok(written)
}

pub(crate) fn encode_single_worker<const STATS: bool>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    version: u16,
    ranges: Option<&[(usize, usize)]>,
    prep_dict: Option<&dict_lz::PreparedDict>,
) -> Result<(Vec<u8>, WorkStats)> {
    let (out, _, stats) =
        encode_single_core::<STATS>(input, opt, engine, count, version, ranges, prep_dict, None)?;
    Ok((out, stats))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_worker_tls<const STATS: bool, C>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    ranges: Option<&[(usize, usize)]>,
    prep_dict: Option<&dict_lz::PreparedDict>,
    worker: usize,
    workers: usize,
    next_idx: &AtomicUsize,
    finish: C,
) -> Result<()>
where
    C: FnOnce(frame::WorkerSlice<'_>) -> Result<()>,
{
    ENCODE_SCRATCH_POOL.with(|sc_cell| {
        WORKER_PARTS_POOL.with(|parts_cell| {
            let mut scratch = sc_cell.borrow_mut();
            let mut parts = parts_cell.borrow_mut();
            scratch.payloads_mut().clear();
            parts.clear();

            let capacity = count.saturating_sub(worker).div_ceil(workers);
            let cur_cap = parts.capacity();
            if cur_cap < capacity {
                parts.reserve(capacity - cur_cap);
            }
            let worker_raw_est = capacity.saturating_mul(opt.block_size).min(input.len());
            let adaptive = !opt.legacy_v2;
            let payload_reserve = if adaptive {
                (worker_raw_est / 8).max(opt.block_size.min(input.len()) + 1024)
            } else {
                (worker_raw_est / 4).max(opt.block_size.min(input.len()) + 1024)
            };
            scratch.reserve_payloads_for(payload_reserve);

            let mut stats = WorkStats::default();
            loop {
                let index = next_idx.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                let (start_byte, chunk) = slice_for_index(input, opt.block_size, ranges, index);
                if !parts.is_empty() {
                    let done = parts.len();
                    let payloads = scratch.payloads_mut();
                    let cur_len = payloads.len();
                    if cur_len + chunk.len() + 1024 > payloads.capacity()
                        || (done == 1
                            && cur_len.saturating_mul(capacity) + opt.block_size + 4096
                                > payloads.capacity())
                    {
                        let rem_blocks = capacity.saturating_sub(done).max(2);
                        let avg = cur_len / done;
                        let est_rem = rem_blocks.saturating_mul(avg + 1024) + opt.block_size;
                        payloads.reserve(est_rem.max(chunk.len() + 1024));
                    }
                }
                let leading_carry = start_byte > 0 && input[start_byte - 1] != b'\n';
                let (kind, offset, len) = encode_one_chunk::<STATS>(
                    chunk,
                    leading_carry,
                    adaptive,
                    prep_dict,
                    &mut scratch,
                    engine,
                    opt.acceleration,
                    &mut stats,
                )?;
                parts.push(frame::ArenaPart {
                    index,
                    offset,
                    len,
                    kind,
                    raw_size: chunk.len(),
                    checksum: if opt.checksum {
                        checksum::crc32c(chunk)
                    } else {
                        0
                    },
                });
            }
            finish(frame::WorkerSlice {
                arena: scratch.payloads_mut(),
                parts: &parts,
                stats,
            })
        })
    })
}
