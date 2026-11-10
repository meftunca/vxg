//! Best-effort block-level recovery for corrupted or truncated `VXG1` frames.
//!
//! Because every `VXG1` block (`v1`–`v5`) is independently encoded and carries its own
//! canonical payload span and optional CRC32C in the frame directory, intact blocks can
//! be decoded even when sibling blocks are corrupted or when the tail of the file is truncated.

use crate::{block, checksum::crc32c, simd::Engine, DecodeOptions, Error, Result};

use super::{
    entry_size, read_u32, read_u64, v5, valid_kind_and_size, HEADER_SIZE, MAGIC, V4_SIZE_MASK,
    VERSION_V4,
};

/// Metadata for a single block that was decoded and verified intact during recovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveredBlock {
    /// Global zero-based block index across the recovered input.
    pub index: usize,
    /// Original byte offset of this block in the uncompressed stream.
    pub raw_offset: usize,
    /// Uncompressed byte length of this block.
    pub raw_size: usize,
    /// Offset into [`RecoveryReport::data`] where this block's recovered bytes start.
    pub output_offset: usize,
    /// Wire block kind (`BLOCK_RAW`, `BLOCK_LZ`, `BLOCK_COLUMN`, `BLOCK_DICT_LZ`).
    pub kind: u32,
}

/// Diagnostic record for a single block that could not be recovered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedBlock {
    /// Global zero-based block index across the recovered input.
    pub index: usize,
    /// Original byte offset of this block in the uncompressed stream.
    pub raw_offset: usize,
    /// Expected uncompressed byte length of this block (`0` if directory was truncated).
    pub raw_size: usize,
    /// Human-readable failure cause (decode error, checksum mismatch, or truncation).
    pub reason: String,
}

/// Summary and recovered payload returned by [`recover_frame`].
#[derive(Debug, Clone)]
pub struct RecoveryReport {
    /// Concatenated decoded bytes of all intact blocks in canonical order.
    pub data: Vec<u8>,
    /// Total number of blocks declared across parsed frame headers.
    pub total_blocks: usize,
    /// Total uncompressed bytes declared across parsed frame headers.
    pub expected_raw_size: usize,
    /// Blocks that decoded cleanly (and passed CRC32C when present and enabled).
    pub recovered_blocks: Vec<RecoveredBlock>,
    /// Blocks that failed decoding, failed CRC32C verification, or were truncated.
    pub failed_blocks: Vec<FailedBlock>,
}

/// Attempts block-by-block recovery on a single or concatenated `VXG1` archive that may
/// contain corrupted block payloads, CRC mismatches, or tail truncation.
///
/// Returns `Err` only if the initial frame header/directory/dictionary prefix itself is
/// unreadable or if `options` / `max_output` limits are violated.
pub fn recover_frame(src: &[u8], options: &DecodeOptions) -> Result<RecoveryReport> {
    options.validate()?;
    let engine = Engine::new(options.simd)?;
    let mut report = RecoveryReport {
        data: Vec::new(),
        total_blocks: 0,
        expected_raw_size: 0,
        recovered_blocks: Vec::new(),
        failed_blocks: Vec::new(),
    };

    if src.is_empty() {
        return Err(Error::Truncated);
    }
    if super::v6::is_v6_frame(src) {
        return super::v6::recover_v6_frame(src, options);
    }

    let mut frame_pos = 0usize;
    let mut global_block_idx = 0usize;
    let mut global_raw_base = 0usize;
    let mut scratch = Vec::new();

    while frame_pos < src.len() {
        let rem = &src[frame_pos..];
        if frame_pos > 0 && (rem.len() < HEADER_SIZE || &rem[..4] != MAGIC) {
            break;
        }
        let remaining_budget = options.max_output.saturating_sub(report.expected_raw_size);
        let hdr = match v5::parse_v1_to_v5_header(rem, remaining_budget) {
            Ok(h) => h,
            Err(e) if frame_pos == 0 => return Err(e),
            Err(_) => break,
        };
        if rem.len() < hdr.payload_start {
            if frame_pos == 0 {
                return Err(Error::Truncated);
            }
            break;
        }
        let dict = match v5::decode_header_dict(rem, &hdr) {
            Ok(d) => d,
            Err(e) if frame_pos == 0 => return Err(e),
            Err(_) => break,
        };

        report.total_blocks = report
            .total_blocks
            .checked_add(hdr.count)
            .ok_or(Error::SizeOverflow)?;
        report.expected_raw_size = report
            .expected_raw_size
            .checked_add(hdr.raw_size)
            .ok_or(Error::SizeOverflow)?;

        let esize = entry_size(hdr.version);
        let verify_csum = hdr.checksummed && options.verify_checksum;
        let short_dist = hdr.version >= 2;
        let mut next_payload = hdr.payload_start;
        let mut next_raw = 0usize;
        let mut dir_aborted = false;

        for i in 0..hdr.count {
            let g_idx = global_block_idx + i;
            let p = HEADER_SIZE + i * esize;
            let expected_rn = match v5::block_raw_size(rem, &hdr, i, next_raw) {
                Ok(rn) => rn,
                Err(e) => {
                    report.failed_blocks.push(FailedBlock {
                        index: g_idx,
                        raw_offset: global_raw_base + next_raw,
                        raw_size: 0,
                        reason: e.to_string(),
                    });
                    dir_aborted = true;
                    break;
                }
            };
            let (raw_off, pay_off, rn, cn, checksum, kind) = if hdr.version >= VERSION_V4 {
                let word = read_u32(rem, p)?;
                (
                    next_raw,
                    next_payload,
                    expected_rn,
                    (word & V4_SIZE_MASK) as usize,
                    read_u32(rem, p + 4)?,
                    word >> 30,
                )
            } else {
                (
                    usize::try_from(read_u64(rem, p)?).map_err(|_| Error::SizeOverflow)?,
                    usize::try_from(read_u64(rem, p + 8)?).map_err(|_| Error::SizeOverflow)?,
                    read_u32(rem, p + 16)? as usize,
                    read_u32(rem, p + 20)? as usize,
                    read_u32(rem, p + 24)?,
                    read_u32(rem, p + 28)?,
                )
            };

            if raw_off != next_raw
                || pay_off != next_payload
                || rn != expected_rn
                || !valid_kind_and_size(hdr.version, kind, cn, rn)
                || (!hdr.checksummed && checksum != 0)
            {
                report.failed_blocks.push(FailedBlock {
                    index: g_idx,
                    raw_offset: global_raw_base + next_raw,
                    raw_size: expected_rn,
                    reason: "corrupt directory entry".into(),
                });
                dir_aborted = true;
                break;
            }

            let Some(end) = pay_off.checked_add(cn) else {
                dir_aborted = true;
                break;
            };
            let g_raw_off = global_raw_base + raw_off;
            next_raw += rn;
            next_payload = end;

            let Some(payload) = rem.get(pay_off..end) else {
                report.failed_blocks.push(FailedBlock {
                    index: g_idx,
                    raw_offset: g_raw_off,
                    raw_size: rn,
                    reason: "truncated block payload".into(),
                });
                continue;
            };

            scratch.resize(rn, 0);
            match block::decode_with_dict(payload, kind, &mut scratch, engine, short_dist, &dict) {
                Ok(()) => {
                    if verify_csum && crc32c(&scratch) != checksum {
                        report.failed_blocks.push(FailedBlock {
                            index: g_idx,
                            raw_offset: g_raw_off,
                            raw_size: rn,
                            reason: format!("checksum mismatch in block {g_idx}"),
                        });
                    } else {
                        let output_offset = report.data.len();
                        report.data.extend_from_slice(&scratch);
                        report.recovered_blocks.push(RecoveredBlock {
                            index: g_idx,
                            raw_offset: g_raw_off,
                            raw_size: rn,
                            output_offset,
                            kind,
                        });
                    }
                }
                Err(e) => {
                    report.failed_blocks.push(FailedBlock {
                        index: g_idx,
                        raw_offset: g_raw_off,
                        raw_size: rn,
                        reason: e.to_string(),
                    });
                }
            }
        }

        global_block_idx += hdr.count;
        global_raw_base += hdr.raw_size;
        if dir_aborted || next_payload > rem.len() || next_payload == 0 {
            break;
        }
        frame_pos += next_payload;
    }

    Ok(report)
}
