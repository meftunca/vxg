//! `VERSION_V5` header extensions (`FLAG_FRAME_DICT`, `FLAG_VAR_RAW`) and record-aligned splitting.

use crate::{dict_lz, Error, Result};

use super::{
    check_block_size, entry_size, read_u32, read_u64, HEADER_SIZE, MAGIC, MAX_VERSION, MIN_VERSION,
    VERSION_V5,
};

pub const FLAG_CHECKSUM: u16 = 0x0001;
pub const FLAG_FRAME_DICT: u16 = 0x0002;
pub const FLAG_VAR_RAW: u16 = 0x0004;
const VALID_V5_FLAGS: u16 = FLAG_CHECKSUM | FLAG_FRAME_DICT | FLAG_VAR_RAW;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParsedHeader {
    pub raw_size: usize,
    pub block_size: usize,
    pub count: usize,
    pub version: u16,
    pub checksummed: bool,
    pub has_dict: bool,
    pub has_var_raw: bool,
    pub var_raw_offset: usize,
    pub dict_offset: usize,
    pub dict_wire_len: usize,
    pub dict_raw_len: usize,
    /// Offset where block 0's compressed payload begins (`HEADER_SIZE + dir + var_raw + dict`).
    pub payload_start: usize,
}

pub(crate) fn parse_v1_to_v5_header(src: &[u8], max_output: usize) -> Result<ParsedHeader> {
    if src.len() < HEADER_SIZE {
        return Err(Error::Truncated);
    }
    let version = u16::from_le_bytes(src[4..6].try_into().unwrap());
    if &src[..4] != MAGIC || !(MIN_VERSION..=MAX_VERSION).contains(&version) {
        return Err(Error::InvalidFrame("magic/version"));
    }
    let flags = u16::from_le_bytes(src[6..8].try_into().unwrap());
    if version < VERSION_V5 {
        if flags & !FLAG_CHECKSUM != 0 || read_u64(src, 24)? != 0 {
            return Err(Error::InvalidFrame("unknown flags/reserved data"));
        }
    } else if flags & !VALID_V5_FLAGS != 0 {
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

    let has_dict = version >= VERSION_V5 && (flags & FLAG_FRAME_DICT) != 0;
    let has_var_raw = version >= VERSION_V5 && (flags & FLAG_VAR_RAW) != 0;

    if !has_var_raw {
        let expected_count = raw_size / block_size + usize::from(raw_size % block_size != 0);
        if count != expected_count {
            return Err(Error::InvalidFrame("block count"));
        }
    } else if raw_size == 0 {
        if count != 0 {
            return Err(Error::InvalidFrame("block count"));
        }
    } else {
        let min_count = raw_size.div_ceil(block_size);
        let max_count = raw_size.min((min_count.saturating_mul(8) / 3).max(min_count + 4));
        if !(min_count..=max_count).contains(&count) {
            return Err(Error::InvalidFrame("block count"));
        }
    }

    let (dict_wire_len, dict_raw_len) = if version >= VERSION_V5 {
        let dw = read_u32(src, 24)? as usize;
        let dr = read_u32(src, 28)? as usize;
        if has_dict {
            if !(dict_lz::MIN_DICT_RAW..=dict_lz::MAX_DICT_RAW).contains(&dr)
                || dw == 0
                || dw > dr
            {
                return Err(Error::InvalidFrame("invalid frame dictionary header"));
            }
            (dw, dr)
        } else {
            if dw != 0 || dr != 0 {
                return Err(Error::InvalidFrame("unknown flags/reserved data"));
            }
            (0, 0)
        }
    } else {
        (0, 0)
    };

    let esize = entry_size(version);
    let base_dir_end = count
        .checked_mul(esize)
        .and_then(|v| v.checked_add(HEADER_SIZE))
        .ok_or(Error::SizeOverflow)?;
    let var_raw_bytes = if has_var_raw {
        count.checked_mul(4).ok_or(Error::SizeOverflow)?
    } else {
        0
    };
    let dict_offset = base_dir_end
        .checked_add(var_raw_bytes)
        .ok_or(Error::SizeOverflow)?;
    let payload_start = dict_offset
        .checked_add(dict_wire_len)
        .ok_or(Error::SizeOverflow)?;

    Ok(ParsedHeader {
        raw_size,
        block_size,
        count,
        version,
        checksummed: (flags & FLAG_CHECKSUM) != 0,
        has_dict,
        has_var_raw,
        var_raw_offset: base_dir_end,
        dict_offset,
        dict_wire_len,
        dict_raw_len,
        payload_start,
    })
}

#[inline]
pub(crate) fn block_raw_size(src: &[u8], hdr: &ParsedHeader, index: usize, next_raw: usize) -> Result<usize> {
    if hdr.has_var_raw {
        let rn = read_u32(src, hdr.var_raw_offset + index * 4)? as usize;
        if rn == 0 || rn > hdr.block_size || next_raw.checked_add(rn).is_none_or(|e| e > hdr.raw_size) {
            return Err(Error::InvalidFrame("invalid var_raw block size"));
        }
        Ok(rn)
    } else {
        Ok(hdr.block_size.min(hdr.raw_size - next_raw))
    }
}

#[inline]
pub(crate) fn decode_header_dict(src: &[u8], hdr: &ParsedHeader) -> Result<Vec<u8>> {
    if !hdr.has_dict {
        return Ok(Vec::new());
    }
    let end = hdr.dict_offset + hdr.dict_wire_len;
    let wire = src.get(hdr.dict_offset..end).ok_or(Error::Truncated)?;
    dict_lz::decode_wire_dict(wire, hdr.dict_raw_len)
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn write_header_bytes(
    buf: &mut [u8],
    version: u16,
    checksummed: bool,
    has_var_raw: bool,
    dict_wire: Option<(&[u8], usize)>,
    raw_len: usize,
    block_size: usize,
    count: usize,
) {
    let mut flags = if checksummed { FLAG_CHECKSUM } else { 0 };
    let (dw, dr) = if let Some((wire, raw_sz)) = dict_wire {
        flags |= FLAG_FRAME_DICT;
        (wire.len() as u32, raw_sz as u32)
    } else {
        (0u32, 0u32)
    };
    if has_var_raw {
        flags |= FLAG_VAR_RAW;
    }
    buf[0..4].copy_from_slice(MAGIC);
    buf[4..6].copy_from_slice(&version.to_le_bytes());
    buf[6..8].copy_from_slice(&flags.to_le_bytes());
    buf[8..16].copy_from_slice(&(raw_len as u64).to_le_bytes());
    buf[16..20].copy_from_slice(&(block_size as u32).to_le_bytes());
    buf[20..24].copy_from_slice(&(count as u32).to_le_bytes());
    buf[24..28].copy_from_slice(&dw.to_le_bytes());
    buf[28..32].copy_from_slice(&dr.to_le_bytes());
}

fn is_ndjson_sample(sample: &[u8]) -> bool {
    let mut total = 0usize;
    let mut json_obj = 0usize;
    for raw_ln in sample.split(|&b| b == b'\n') {
        let ln = raw_ln.strip_suffix(b"\r").unwrap_or(raw_ln);
        let trimmed_start = match ln.iter().position(|b| !b.is_ascii_whitespace()) {
            Some(i) => &ln[i..],
            None => continue,
        };
        let trimmed_end = match trimmed_start.iter().rposition(|b| !b.is_ascii_whitespace()) {
            Some(i) => &trimmed_start[..=i],
            None => continue,
        };
        total += 1;
        if trimmed_end.first() == Some(&b'{') && trimmed_end.last() == Some(&b'}') {
            json_obj += 1;
        }
        if total >= 64 {
            break;
        }
    }
    // Exclude the trailing partial line from `sample[..65536]` if needed; require >= 75% JSON objects.
    total >= 8 && json_obj * 4 >= total.saturating_sub(1) * 3
}

/// Resolve effective block size and `record_aligned` mode for verbose tabular/NDJSON inputs.
/// Never shrinks below `options.block_size`, preserving `max_compressed_size(n, options.block_size)`.
pub(crate) fn resolve_effective_options(input: &[u8], options: &crate::Options) -> crate::Options {
    let mut eff = options.clone();
    if eff.legacy_v2 {
        return eff;
    }
    if eff.target_block_rows > 0 {
        eff.record_aligned = true;
        let sample = &input[..input.len().min(65_536)];
        let nl = sample.iter().filter(|&&b| b == b'\n').count();
        if nl >= 2 {
            let avg_line = sample.len() / nl;
            let raw_target = eff.target_block_rows.saturating_mul(avg_line);
            let rounded = raw_target.div_ceil(4096).saturating_mul(4096);
            eff.block_size = rounded.clamp(options.block_size, 1_048_576);
        }
        return eff;
    }
    if eff.record_aligned && options.block_size == 65_536 && input.len() >= 1_048_576 {
        let sample = &input[..65_536];
        let nl = sample.iter().filter(|&&b| b == b'\n').count();
        if nl >= 8 && is_ndjson_sample(sample) {
            let avg_line = 65_536 / nl;
            if avg_line >= 110 {
                let max_par_bsz = ((input.len() / 4) & !65_535).clamp(65_536, 524_288);
                let raw_target = 2048usize.saturating_mul(avg_line);
                let rounded = raw_target.div_ceil(65_536).saturating_mul(65_536);
                eff.block_size = rounded.clamp(65_536, max_par_bsz);
            }
        }
    }
    eff
}

/// Compute newline-aligned `(start, end)` byte ranges when `record_aligned` is active.
/// Returns `None` if fixed `block_size` chunking should be used instead.
pub(crate) fn compute_block_ranges(
    input: &[u8],
    block_size: usize,
    record_aligned: bool,
) -> Option<Vec<(usize, usize)>> {
    let n = input.len();
    if !record_aligned || n <= block_size {
        return None;
    }
    let max_back = (block_size / 4).min(32_768);
    let mut ranges = Vec::with_capacity(n.div_ceil(block_size) + 4);
    let mut pos = 0usize;
    let mut any_shifted = false;
    while pos < n {
        if n - pos <= block_size {
            ranges.push((pos, n));
            break;
        }
        let ideal_end = pos + block_size;
        let search_start = ideal_end - max_back;
        let end = match input[search_start..ideal_end]
            .iter()
            .rposition(|&b| b == b'\n')
        {
            Some(rel) => {
                let snapped = search_start + rel + 1;
                if snapped != ideal_end {
                    any_shifted = true;
                }
                snapped
            }
            None => ideal_end,
        };
        ranges.push((pos, end));
        pos = end;
    }
    if !any_shifted {
        return None;
    }
    // Guarantee `12 * count <= 32 * fixed_blocks` so `max_compressed_size` bound always holds.
    let fixed_blocks = n.div_ceil(block_size);
    if ranges.len() * 12 > fixed_blocks * 32 {
        return None;
    }
    Some(ranges)
}
