//! Compressed-domain structured record random access (`row`, `range`) and
//! predicate pushdown search (`find_first`, `filter_rows`) for VXG v4 frames.
//!
//! On `GRAMMAR_PACKET` (`BLOCK_COLUMN`) blocks, row counts are read in $O(1)$ time
//! from the packet header without unpacking columns, row ranges skip preceding
//! rows in $O(1)$ pointer/counter time via `PreparedTemplate::skip_rows`, and
//! `find_first` / `filter_rows` apply 256-bit block alphabet rejection and
//! dictionary-index SIMD pushdown before materializing candidate rows.

use crate::{DecodeOptions, Error, Frame, Result, BLOCK_COLUMN};

mod numeric;
mod parallel;
pub use numeric::NumCmp;
pub(crate) use numeric::{line_numeric_match, NumFilter};
use parallel::{filter_impl, Pred};

/// A single matching structured record returned by [`find_first`] or [`filter_rows`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMatch {
    /// Zero-based block index within the VXG frame.
    pub block_index: usize,
    /// Zero-based record index within `block_index`.
    pub row_index: usize,
    /// Zero-based global record index across the entire frame.
    pub global_row_index: usize,
    /// Materialized bytes of the matching record (including trailing newline if present).
    pub bytes: Vec<u8>,
}

#[inline]
fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    // SIMD first-byte scan + last-byte probe (shared with the grammar search path).
    crate::columns::contains_subslice(haystack, needle)
}

/// Decode block `block_index` into a caller-owned, reusable buffer (no per-block allocation).
pub(crate) fn decode_block_into(
    frame_bytes: &[u8],
    block_index: usize,
    raw_size: usize,
    options: &DecodeOptions,
    buf: &mut Vec<u8>,
) -> Result<()> {
    buf.clear();
    buf.resize(raw_size, 0);
    let mut opts = options.clone();
    opts.max_output = opts.max_output.max(raw_size);
    crate::extract_block_into(frame_bytes, block_index, buf.as_mut_slice(), &opts)?;
    Ok(())
}

fn decode_non_grammar_block(
    frame_bytes: &[u8],
    block_index: usize,
    raw_size: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; raw_size];
    let mut opts = options.clone();
    opts.max_output = opts.max_output.max(raw_size);
    crate::extract_block_into(frame_bytes, block_index, &mut buf, &opts)?;
    Ok(buf)
}

/// Return the number of structured records (or newline-delimited lines for non-grammar
/// blocks) in block `block_index`.
///
/// On `GRAMMAR_PACKET` blocks this reads only the ~100-byte grammar header in $O(1)$
/// time without unpacking any column streams.
pub fn block_row_count(frame_bytes: &[u8], block_index: usize) -> Result<usize> {
    let parsed = Frame::parse(frame_bytes, usize::MAX)?;
    let info = parsed
        .blocks()
        .nth(block_index)
        .ok_or(Error::InvalidFrame("block index out of range"))?;
    if info.kind == BLOCK_COLUMN {
        let payload = &frame_bytes[info.payload_offset..info.payload_offset + info.payload_size];
        if let Some(count) = crate::columns::grammar_block_row_count(payload)? {
            return Ok(count);
        }
    }
    let raw = decode_non_grammar_block(
        frame_bytes,
        block_index,
        info.raw_size,
        &DecodeOptions::default(),
    )?;
    Ok(raw.split_inclusive(|&b| b == b'\n').count())
}

/// Return the total number of structured records across all blocks in `frame_bytes`
/// (supporting both single `VXG1` frames and concatenated multi-frame `VXG1` streams).
pub fn frame_row_count(frame_bytes: &[u8]) -> Result<usize> {
    let frames = Frame::parse_all(frame_bytes, usize::MAX)?;
    let mut total = 0usize;
    let mut prev_open = false;
    let dec = DecodeOptions::default();
    for (f_start, parsed) in frame_slices(frame_bytes, &frames) {
        for (idx, info) in parsed.blocks().enumerate() {
            let (rows, ends_nl) = if info.kind == BLOCK_COLUMN {
                let payload = &f_start[info.payload_offset..info.payload_offset + info.payload_size];
                if let Some(meta) = crate::columns::grammar_block_row_meta(payload)? {
                    meta
                } else {
                    let raw = decode_non_grammar_block(f_start, idx, info.raw_size, &dec)?;
                    (raw.split_inclusive(|&b| b == b'\n').count(), raw.last() == Some(&b'\n'))
                }
            } else {
                let raw = decode_non_grammar_block(f_start, idx, info.raw_size, &dec)?;
                (raw.split_inclusive(|&b| b == b'\n').count(), raw.last() == Some(&b'\n'))
            };
            if rows > 0 {
                let new_rows = if prev_open { rows - 1 } else { rows };
                total = total.checked_add(new_rows).ok_or(Error::SizeOverflow)?;
                prev_open = !ends_nl;
            }
        }
    }
    Ok(total)
}

pub(crate) fn frame_slices<'a, 'b>(
    full: &'a [u8],
    frames: &'b [Frame<'a>],
) -> Vec<(&'a [u8], &'b Frame<'a>)> {
    if frames.len() == 1 {
        return vec![(full, &frames[0])];
    }
    let (_, spans) = crate::stream::scan_concatenated_frames(full, usize::MAX).unwrap_or_default();
    spans
        .into_iter()
        .zip(frames.iter())
        .map(|((s, e, _), f)| (&full[s..e], f))
        .collect()
}

/// Extract a single structured record `row_index` from `block_index` without
/// materializing the rest of the block.
pub fn extract_block_row(
    frame_bytes: &[u8],
    block_index: usize,
    row_index: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    extract_block_row_range(
        frame_bytes,
        block_index,
        row_index,
        row_index.checked_add(1).ok_or(Error::SizeOverflow)?,
        options,
    )
}

/// Extract records `[start_row .. end_row)` from block `block_index` using $O(1)$
/// column stream skipping on `GRAMMAR_PACKET` blocks.
pub fn extract_block_row_range(
    frame_bytes: &[u8],
    block_index: usize,
    start_row: usize,
    end_row: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    options.validate()?;
    if start_row >= end_row {
        return Ok(Vec::new());
    }
    let parsed = Frame::parse(frame_bytes, options.max_output)?;
    let info = parsed
        .blocks()
        .nth(block_index)
        .ok_or(Error::InvalidFrame("block index out of range"))?;
    if info.kind == BLOCK_COLUMN {
        let payload = &frame_bytes[info.payload_offset..info.payload_offset + info.payload_size];
        let mut out = Vec::new();
        if let Some(emitted) =
            crate::columns::unpack_grammar_row_range(payload, start_row, end_row, &mut out)?
        {
            if emitted == 0 {
                return Err(Error::InvalidFrame("row index out of range"));
            }
            return Ok(out);
        }
    }
    let raw = decode_non_grammar_block(frame_bytes, block_index, info.raw_size, options)?;
    let mut out = Vec::new();
    let mut emitted = 0usize;
    for (idx, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
        if idx >= start_row && idx < end_row {
            out.extend_from_slice(line);
            emitted += 1;
        }
    }
    if emitted == 0 {
        return Err(Error::InvalidFrame("row index out of range"));
    }
    Ok(out)
}

/// Extract a single record by its global index `global_row_index` across the frame,
/// skipping non-overlapping blocks via $O(1)$ header inspection.
pub fn extract_row(
    frame_bytes: &[u8],
    global_row_index: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    let out = extract_row_range(
        frame_bytes,
        global_row_index,
        global_row_index.checked_add(1).ok_or(Error::SizeOverflow)?,
        options,
    )?;
    if out.is_empty() {
        return Err(Error::InvalidFrame("row index out of range"));
    }
    Ok(out)
}

/// Extract global records `[start_row .. end_row)` across the frame, skipping
/// preceding and trailing blocks without unpacking their columns.
pub fn extract_row_range(
    frame_bytes: &[u8],
    start_row: usize,
    end_row: usize,
    options: &DecodeOptions,
) -> Result<Vec<u8>> {
    options.validate()?;
    if start_row >= end_row {
        return Ok(Vec::new());
    }
    let frames = Frame::parse_all(frame_bytes, options.max_output)?;
    let mut out = Vec::new();
    let mut cur_global = 0usize;
    let mut prev_open = false;

    for (f_bytes, parsed) in frame_slices(frame_bytes, &frames) {
        for (blk_idx, info) in parsed.blocks().enumerate() {
            let base_row = if prev_open { cur_global - 1 } else { cur_global };
            if base_row >= end_row {
                return Ok(out);
            }
            if info.kind == BLOCK_COLUMN {
                let payload = &f_bytes[info.payload_offset..info.payload_offset + info.payload_size];
                if let Some((blk_rows, ends_nl)) = crate::columns::grammar_block_row_meta(payload)? {
                    if blk_rows > 0 {
                        let local_start = start_row.saturating_sub(base_row).min(blk_rows);
                        let local_end = end_row.saturating_sub(base_row).min(blk_rows);
                        if local_start < local_end {
                            let _ = crate::columns::unpack_grammar_row_range(
                                payload,
                                local_start,
                                local_end,
                                &mut out,
                            )?;
                        }
                        cur_global = base_row + blk_rows;
                        prev_open = !ends_nl;
                    }
                    continue;
                }
            }
            let raw = decode_non_grammar_block(f_bytes, blk_idx, info.raw_size, options)?;
            if !raw.is_empty() {
                for (r, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
                    let g_row = base_row + r;
                    if g_row >= start_row && g_row < end_row {
                        out.extend_from_slice(line);
                    }
                }
                let blk_rows = raw.split_inclusive(|&b| b == b'\n').count();
                cur_global = base_row + blk_rows;
                prev_open = raw.last() != Some(&b'\n');
            }
        }
    }
    Ok(out)
}

/// Find the first record in `frame_bytes` containing `needle`, using compressed-domain
/// alphabet & dictionary index pushdown on `GRAMMAR_PACKET` blocks.
pub fn find_first(
    frame_bytes: &[u8],
    needle: &[u8],
    options: &DecodeOptions,
) -> Result<Option<RowMatch>> {
    let mut matches = filter_rows(frame_bytes, needle, 1, options)?;
    Ok(matches.pop())
}

/// Find up to `limit` records in `frame_bytes` containing `needle`, using
/// compressed-domain alphabet & dictionary index pushdown on `GRAMMAR_PACKET` blocks.
pub fn filter_rows(
    frame_bytes: &[u8],
    needle: &[u8],
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    filter_impl(frame_bytes, &Pred::Substr(needle), limit, options)
}

/// Find up to `limit` records whose `"key":` value is an unsigned integer satisfying `cmp`.
///
/// Scope: flat JSON records with plain unsigned decimal integers (no sign, fraction or
/// exponent); other values never match. On structured blocks only the target column is
/// decoded to evaluate the predicate, and the remaining columns are decoded only if some
/// row matches. All other blocks fall back to a per-line check with the same semantics.
pub fn filter_json_u64(
    frame_bytes: &[u8],
    key: &str,
    cmp: NumCmp,
    limit: usize,
    options: &DecodeOptions,
) -> Result<Vec<RowMatch>> {
    let nf = NumFilter::new(key, cmp);
    filter_impl(frame_bytes, &Pred::Num(&nf), limit, options)
}

// ---------------------------------------------------------------------------
// C ABI Exports for Structured Random Access & Search
// ---------------------------------------------------------------------------

/// Return the total number of structured records across all blocks in a VXG frame,
/// or a negative `VXG_ERR_*` code on error.
///
/// # Safety
/// `frame_ptr` must point to `frame_len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vxg_frame_row_count(frame_ptr: *const u8, frame_len: usize) -> isize {
    if frame_ptr.is_null() {
        return crate::ffi::VXG_ERR_NULL_PTR;
    }
    let frame = unsafe { std::slice::from_raw_parts(frame_ptr, frame_len) };
    match frame_row_count(frame) {
        Ok(n) => n as isize,
        Err(_) => crate::ffi::VXG_ERR_CORRUPT_DATA,
    }
}

/// Extract global record `global_row_index` into `dst_ptr[..dst_cap]`, returning the
/// number of bytes written or a negative `VXG_ERR_*` code.
///
/// # Safety
/// `frame_ptr` and `dst_ptr` must point to valid buffers of `frame_len` and `dst_cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vxg_extract_row_into(
    frame_ptr: *const u8,
    frame_len: usize,
    global_row_index: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> isize {
    unsafe {
        vxg_extract_row_range_into(
            frame_ptr,
            frame_len,
            global_row_index,
            global_row_index.saturating_add(1),
            dst_ptr,
            dst_cap,
        )
    }
}

/// Extract global records `[start_row .. end_row)` into `dst_ptr[..dst_cap]`, returning
/// the number of bytes written or a negative `VXG_ERR_*` code.
///
/// # Safety
/// `frame_ptr` and `dst_ptr` must point to valid buffers of `frame_len` and `dst_cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vxg_extract_row_range_into(
    frame_ptr: *const u8,
    frame_len: usize,
    start_row: usize,
    end_row: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> isize {
    if frame_ptr.is_null() || (dst_ptr.is_null() && dst_cap > 0) {
        return crate::ffi::VXG_ERR_NULL_PTR;
    }
    let frame = unsafe { std::slice::from_raw_parts(frame_ptr, frame_len) };
    match extract_row_range(frame, start_row, end_row, &DecodeOptions::default()) {
        Ok(bytes) => {
            if bytes.len() > dst_cap {
                return crate::ffi::VXG_ERR_BUFFER_TOO_SMALL;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst_ptr, bytes.len());
            }
            bytes.len() as isize
        }
        Err(_) => crate::ffi::VXG_ERR_CORRUPT_DATA,
    }
}

/// Search `frame_ptr[..frame_len]` in the compressed domain for the first record
/// containing `needle_ptr[..needle_len]`.
///
/// - If a match is found, copies its bytes into `dst_ptr[..dst_cap]`, writes its
///   `global_row_index` to `*out_row_index` (if non-null), and returns the byte length.
/// - If no record matches, writes `usize::MAX` to `*out_row_index` (if non-null) and returns `0`.
/// - On error, returns a negative `VXG_ERR_*` code.
///
/// # Safety
/// All non-null pointers must reference valid memory for their stated lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vxg_find_first_into(
    frame_ptr: *const u8,
    frame_len: usize,
    needle_ptr: *const u8,
    needle_len: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
    out_row_index: *mut usize,
) -> isize {
    if frame_ptr.is_null()
        || (needle_ptr.is_null() && needle_len > 0)
        || (dst_ptr.is_null() && dst_cap > 0)
    {
        return crate::ffi::VXG_ERR_NULL_PTR;
    }
    let frame = unsafe { std::slice::from_raw_parts(frame_ptr, frame_len) };
    let needle = if needle_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(needle_ptr, needle_len) }
    };
    match find_first(frame, needle, &DecodeOptions::default()) {
        Ok(Some(m)) => {
            if m.bytes.len() > dst_cap {
                return crate::ffi::VXG_ERR_BUFFER_TOO_SMALL;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(m.bytes.as_ptr(), dst_ptr, m.bytes.len());
                if !out_row_index.is_null() {
                    *out_row_index = m.global_row_index;
                }
            }
            m.bytes.len() as isize
        }
        Ok(None) => {
            unsafe {
                if !out_row_index.is_null() {
                    *out_row_index = usize::MAX;
                }
            }
            0
        }
        Err(_) => crate::ffi::VXG_ERR_CORRUPT_DATA,
    }
}

#[cfg(test)]
mod tests;
