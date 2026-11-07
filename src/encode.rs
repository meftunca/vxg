use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{block, checksum, columns, frame, simd, Error, Options, Result, WorkStats};

thread_local! {
    pub(crate) static ENCODE_SCRATCH_POOL: RefCell<block::Scratch> =
        RefCell::new(block::Scratch::new());
    static WORKER_PARTS_POOL: RefCell<Vec<frame::ArenaPart>> =
        const { RefCell::new(Vec::new()) };
}

#[inline]
fn encode_one_chunk<const STATS: bool>(
    chunk: &[u8],
    leading_carry: bool,
    adaptive: bool,
    scratch: &mut block::Scratch,
    engine: simd::Engine,
    acceleration: usize,
    stats: &mut WorkStats,
) -> Result<(u32, usize, usize)> {
    if adaptive {
        match columns::encode_grammar_block_into_with_carry(
            chunk,
            leading_carry,
            scratch.payloads_mut(),
        )? {
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
    } else {
        let enc = block::encode::<STATS>(chunk, scratch, engine, acceleration);
        if STATS {
            *stats += enc.stats;
        }
        Ok((enc.kind, enc.offset, enc.len))
    }
}

pub(crate) fn encode_single_into(
    input: &[u8],
    out: &mut [u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    version: u16,
) -> Result<usize> {
    let esize = frame::entry_size(version);
    let adaptive = !opt.legacy_v2;
    let table_end = frame::HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    ENCODE_SCRATCH_POOL.with(|cell| -> Result<usize> {
        let mut scratch = cell.borrow_mut();
        scratch.payloads_mut().clear();
        let payload_reserve = if adaptive {
            (input.len() / 12).max(opt.block_size.min(input.len()))
        } else {
            (input.len() / 4).max(opt.block_size.min(input.len()))
        };
        scratch.reserve_payloads_for(table_end.saturating_add(payload_reserve));
        scratch.payloads_mut().resize(table_end, 0);
        let mut dummy_stats = WorkStats::default();
        let mut raw_offset = 0usize;
        for (index, chunk) in input.chunks(opt.block_size).enumerate() {
            let leading_carry = raw_offset > 0 && input[raw_offset - 1] != b'\n';
            let (kind, offset, len) = encode_one_chunk::<false>(
                chunk,
                leading_carry,
                adaptive,
                &mut scratch,
                engine,
                opt.acceleration,
                &mut dummy_stats,
            )?;
            let csum = if opt.checksum {
                checksum::crc32c(chunk)
            } else {
                0
            };
            let p = frame::HEADER_SIZE + index * esize;
            frame::write_entry(
                scratch.payloads_mut(),
                p,
                version,
                raw_offset,
                offset,
                chunk.len(),
                len,
                csum,
                kind,
            );
            raw_offset += chunk.len();
        }
        let payloads = scratch.payloads_mut();
        let written = payloads.len();
        if out.len() < written {
            return Err(Error::OutputSize {
                expected: written,
                actual: out.len(),
            });
        }
        write_frame_header(payloads, version, opt.checksum, input.len(), opt.block_size, count);
        out[..written].copy_from_slice(payloads);
        Ok(written)
    })
}

pub(crate) fn encode_single_worker<const STATS: bool>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
    count: usize,
    version: u16,
) -> Result<(Vec<u8>, WorkStats)> {
    let esize = frame::entry_size(version);
    let adaptive = !opt.legacy_v2;
    let table_end = frame::HEADER_SIZE
        .checked_add(esize.checked_mul(count).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    ENCODE_SCRATCH_POOL.with(|cell| -> Result<(Vec<u8>, WorkStats)> {
        let mut scratch = cell.borrow_mut();
        scratch.payloads_mut().clear();
        let payload_reserve = if adaptive {
            (input.len() / 12).max(opt.block_size.min(input.len()))
        } else {
            (input.len() / 4).max(opt.block_size.min(input.len()))
        };
        scratch.reserve_payloads_for(table_end.saturating_add(payload_reserve));
        scratch.payloads_mut().resize(table_end, 0);
        let mut stats = WorkStats::default();
        let mut raw_offset = 0usize;
        for (index, chunk) in input.chunks(opt.block_size).enumerate() {
            if index > 0 {
                let payloads = scratch.payloads_mut();
                let cur_len = payloads.len();
                let written = cur_len - table_end;
                if cur_len + chunk.len() + 1024 > payloads.capacity()
                    || (index == 1
                        && table_end + written * count + opt.block_size > payloads.capacity())
                {
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
                &mut scratch,
                engine,
                opt.acceleration,
                &mut stats,
            )?;
            let csum = if opt.checksum {
                checksum::crc32c(chunk)
            } else {
                0
            };
            let p = frame::HEADER_SIZE + index * esize;
            frame::write_entry(
                scratch.payloads_mut(),
                p,
                version,
                raw_offset,
                offset,
                chunk.len(),
                len,
                csum,
                kind,
            );
            raw_offset += chunk.len();
        }
        let payloads = scratch.payloads_mut();
        write_frame_header(payloads, version, opt.checksum, input.len(), opt.block_size, count);
        let mut out = Vec::with_capacity(payloads.len());
        out.extend_from_slice(payloads);
        Ok((out, stats))
    })
}

pub(crate) fn encode_worker_tls<const STATS: bool, C>(
    input: &[u8],
    opt: &Options,
    engine: simd::Engine,
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

            let count = input.len().div_ceil(opt.block_size);
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
                let start_byte = index * opt.block_size;
                let end_byte = (start_byte + opt.block_size).min(input.len());
                let chunk = &input[start_byte..end_byte];
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

#[inline]
fn write_frame_header(
    buf: &mut [u8],
    version: u16,
    checksum: bool,
    raw_len: usize,
    block_size: usize,
    count: usize,
) {
    buf[0..4].copy_from_slice(frame::MAGIC);
    buf[4..6].copy_from_slice(&version.to_le_bytes());
    buf[6..8].copy_from_slice(&(u16::from(checksum)).to_le_bytes());
    buf[8..16].copy_from_slice(&(raw_len as u64).to_le_bytes());
    buf[16..20].copy_from_slice(&(block_size as u32).to_le_bytes());
    buf[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    buf[24..32].copy_from_slice(&0u64.to_le_bytes());
}
