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
    if super::v6::is_v6_frame(src) || has_v6_group_markers(src) {
        return recover_v6_frame(src, options);
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

fn has_v6_group_markers(src: &[u8]) -> bool {
    if src.len() >= HEADER_SIZE && &src[..4] == MAGIC {
        let ver = u16::from_le_bytes([src[4], src[5]]);
        if (1..=5).contains(&ver) {
            return false;
        }
    }
    src.windows(4)
        .any(|w| w == super::v6::GROUP_MAGIC || w == super::v6::GROUP_TRAILER_MAGIC)
}

pub(crate) fn recover_v6_frame(src: &[u8], options: &DecodeOptions) -> Result<RecoveryReport> {
    use super::v6::{parse_v6_group_dir, ParsedGroupHeader, GROUP_MAGIC, GROUP_TRAILER_MAGIC};
    options.validate()?;
    let hdr_opt = v5::parse_v1_to_v5_header(src, options.max_output).ok();
    let dir_opt = hdr_opt
        .as_ref()
        .and_then(|h| parse_v6_group_dir(src, h).ok());
    let dict = hdr_opt
        .as_ref()
        .and_then(|h| v5::decode_header_dict(src, h).ok())
        .unwrap_or_default();
    let engine = Engine::new(options.simd)?;

    let mut global_infos: Vec<Option<(usize, usize, usize, u32, u32)>> = Vec::new();
    if let Some(ref hdr) = hdr_opt {
        global_infos.reserve(hdr.count);
        let mut next_raw = 0usize;
        for i in 0..hdr.count {
            let p = HEADER_SIZE + i * 8;
            if let (Ok(rn), Ok(word), Ok(csum)) = (
                v5::block_raw_size(src, hdr, i, next_raw),
                read_u32(src, p),
                read_u32(src, p + 4),
            ) {
                let cn = (word & V4_SIZE_MASK) as usize;
                let kind = word >> 30;
                global_infos.push(Some((next_raw, rn, cn, csum, kind)));
                next_raw = next_raw.saturating_add(rn);
            } else {
                global_infos.push(None);
                next_raw = next_raw.saturating_add(hdr.block_size);
            }
        }
    }

    let mut found_groups: Vec<(u32, ParsedGroupHeader<'_>, &[u8])> = Vec::new();
    let mut failed_group_ranges: Vec<(usize, usize, String)> = Vec::new();

    if let (Some(ref hdr), Some((total_v6_payload, ref entries))) = (&hdr_opt, &dir_opt) {
        let v6_end = hdr
            .payload_start
            .saturating_add(*total_v6_payload as usize)
            .min(src.len());
        if hdr.payload_start <= v6_end {
            let v6_payload = &src[hdr.payload_start..v6_end];
            for (gi, entry) in entries.iter().enumerate() {
                let g_start = entry.payload_rel_offset as usize;
                let g_end = entries
                    .get(gi + 1)
                    .map(|e| e.payload_rel_offset as usize)
                    .unwrap_or(*total_v6_payload as usize)
                    .min(v6_payload.len());
                if g_start < g_end {
                    let gs = &v6_payload[g_start..g_end];
                    match ParsedGroupHeader::parse_resilient(gs, entry.start_block) {
                        Ok(gh) => found_groups.push((entry.start_block, gh, gs)),
                        Err(e) => failed_group_ranges.push((
                            entry.start_block as usize,
                            entry.block_count as usize,
                            e.to_string(),
                        )),
                    }
                } else {
                    failed_group_ranges.push((
                        entry.start_block as usize,
                        entry.block_count as usize,
                        "group truncated".to_owned(),
                    ));
                }
            }
        }
    }

    let scan_start = hdr_opt
        .as_ref()
        .map_or(HEADER_SIZE.min(src.len()), |h| h.payload_start.min(src.len()));
    let mut pos = scan_start;
    while pos + 16 <= src.len() {
        if &src[pos..pos + 4] == GROUP_MAGIC {
            if let Ok(gh) = ParsedGroupHeader::parse_any_start(&src[pos..]) {
                let end_streams = gh.data_offset.saturating_add(gh.total_streams_bytes());
                if pos + end_streams <= src.len()
                    && !found_groups.iter().any(|&(sb, _, _)| sb == gh.start_block)
                {
                    failed_group_ranges.retain(|&(sb, _, _)| sb != gh.start_block as usize);
                    found_groups.push((gh.start_block, gh, &src[pos..pos + end_streams]));
                    pos += end_streams;
                    continue;
                }
            }
        } else if &src[pos..pos + 4] == GROUP_TRAILER_MAGIC && pos >= 20 {
            if let Ok(gh_len_u32) = read_u32(src, pos - 4) {
                let gh_len = gh_len_u32 as usize;
                if gh_len >= 16 && pos >= 4 + gh_len {
                    let bk_start = pos - 4 - gh_len;
                    if let Ok(gh) =
                        ParsedGroupHeader::parse_any_start(&src[bk_start..bk_start + gh_len])
                    {
                        let st_bytes = gh.total_streams_bytes();
                        if bk_start >= st_bytes
                            && !found_groups.iter().any(|&(sb, _, _)| sb == gh.start_block)
                        {
                            let data_start = bk_start - st_bytes;
                            let g_start = data_start.saturating_sub(gh.data_offset);
                            failed_group_ranges.retain(|&(sb, _, _)| sb != gh.start_block as usize);
                            found_groups.push((gh.start_block, gh, &src[g_start..pos + 4]));
                        }
                    }
                }
            }
        }
        pos += 1;
    }
    found_groups.sort_by_key(|&(sb, _, _)| sb);

    if found_groups.is_empty() && hdr_opt.is_none() {
        return Err(Error::InvalidFrame("magic/version"));
    }

    let total_blocks = hdr_opt.as_ref().map_or_else(
        || {
            found_groups
                .iter()
                .map(|&(sb, ref gh, _)| sb as usize + gh.count)
                .max()
                .unwrap_or(0)
        },
        |h| h.count,
    );
    let expected_raw_size = hdr_opt.as_ref().map_or(0, |h| h.raw_size);
    let verify_csum = hdr_opt.as_ref().is_none_or(|h| h.checksummed) && options.verify_checksum;

    let mut data = Vec::with_capacity(expected_raw_size.min(16 << 20));
    let mut recovered_blocks = Vec::new();
    let mut failed_blocks = Vec::new();
    let mut scratch = Vec::new();

    for (sb, bc, reason) in failed_group_ranges {
        for bi in 0..bc {
            let idx = sb + bi;
            let (raw_offset, raw_size) = global_infos
                .get(idx)
                .and_then(|&x| x)
                .map_or((0, 0), |(ro, rs, _, _, _)| (ro, rs));
            failed_blocks.push(FailedBlock {
                index: idx,
                raw_offset,
                raw_size,
                reason: reason.clone(),
            });
        }
    }

    for (sb_u32, gh, gs) in found_groups {
        let sb = sb_u32 as usize;
        let mut stream_bases = Vec::with_capacity(gh.slots + 1);
        let mut cur = gh.data_offset;
        for s in 0..=gh.slots {
            stream_bases.push(cur);
            cur += gh.stream_total_len(s);
        }
        let mut running_raw = global_infos
            .get(sb)
            .and_then(|&x| x)
            .map_or(0, |(ro, _, _, _, _)| ro);

        for bi in 0..gh.count {
            let idx = sb + bi;
            let meta_opt = gh
                .backup_block_meta(bi)
                .map(|(word, csum, rn)| {
                    let cn = (word & V4_SIZE_MASK) as usize;
                    let kind = word >> 30;
                    (running_raw, rn, cn, csum, kind)
                })
                .or_else(|| global_infos.get(idx).and_then(|&x| x));
            let Some((raw_offset, raw_size, cn, csum, kind)) = meta_opt else {
                failed_blocks.push(FailedBlock {
                    index: idx,
                    raw_offset: running_raw,
                    raw_size: 0,
                    reason: "missing block metadata".to_owned(),
                });
                continue;
            };
            running_raw = raw_offset.saturating_add(raw_size);
            if data.len().saturating_add(raw_size) > options.max_output {
                return Err(Error::OutputLimit {
                    requested: (data.len() + raw_size) as u64,
                    limit: options.max_output,
                });
            }
            let mut blk_buf = Vec::with_capacity(cn);
            let mut bounds_ok = true;
            for (s, &s_base) in stream_bases.iter().enumerate() {
                let (st, en) = gh.stream_slice_range(s, bi);
                if let Some(sl) = gs.get(s_base + st..s_base + en) {
                    blk_buf.extend_from_slice(sl);
                } else {
                    bounds_ok = false;
                    break;
                }
            }
            if !bounds_ok || blk_buf.len() != cn {
                failed_blocks.push(FailedBlock {
                    index: idx,
                    raw_offset,
                    raw_size,
                    reason: "v6 block stream slice out of bounds".to_owned(),
                });
                continue;
            }
            scratch.resize(raw_size, 0);
            match block::decode_with_dict(&blk_buf, kind, &mut scratch, engine, true, &dict) {
                Ok(()) if !verify_csum || crc32c(&scratch) == csum => {
                    let output_offset = data.len();
                    data.extend_from_slice(&scratch);
                    recovered_blocks.push(RecoveredBlock {
                        index: g_idx_val(idx),
                        raw_offset,
                        raw_size,
                        output_offset,
                        kind,
                    });
                }
                Ok(()) => failed_blocks.push(FailedBlock {
                    index: idx,
                    raw_offset,
                    raw_size,
                    reason: format!("checksum mismatch in block {idx}"),
                }),
                Err(error) => failed_blocks.push(FailedBlock {
                    index: idx,
                    raw_offset,
                    raw_size,
                    reason: error.to_string(),
                }),
            }
        }
    }
    recovered_blocks.sort_by_key(|b| b.index);
    failed_blocks.sort_by_key(|b| b.index);
    Ok(RecoveryReport {
        data,
        total_blocks,
        expected_raw_size,
        recovered_blocks,
        failed_blocks,
    })
}

#[inline]
const fn g_idx_val(idx: usize) -> usize {
    idx
}
