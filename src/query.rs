//! Compressed-domain structured record random access (`row`, `range`) and
//! predicate pushdown search (`find_first`, `filter_rows`) for VXG v4 frames.
//!
//! On `GRAMMAR_PACKET` (`BLOCK_COLUMN`) blocks, row counts are read in $O(1)$ time
//! from the packet header without unpacking columns, row ranges skip preceding
//! rows in $O(1)$ pointer/counter time via `PreparedTemplate::skip_rows`, and
//! `find_first` / `filter_rows` apply 256-bit block alphabet rejection and
//! dictionary-index SIMD pushdown before materializing candidate rows.

use crate::{DecodeOptions, Error, Frame, Result, BLOCK_COLUMN};

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
    if needle.is_empty() {
        return true;
    }
    if haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
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

/// Return the total number of structured records across all blocks in `frame_bytes`.
pub fn frame_row_count(frame_bytes: &[u8]) -> Result<usize> {
    let parsed = Frame::parse(frame_bytes, usize::MAX)?;
    let mut total = 0usize;
    for (idx, info) in parsed.blocks().enumerate() {
        let count = if info.kind == BLOCK_COLUMN {
            let payload =
                &frame_bytes[info.payload_offset..info.payload_offset + info.payload_size];
            if let Some(c) = crate::columns::grammar_block_row_count(payload)? {
                c
            } else {
                let raw = decode_non_grammar_block(
                    frame_bytes,
                    idx,
                    info.raw_size,
                    &DecodeOptions::default(),
                )?;
                raw.split_inclusive(|&b| b == b'\n').count()
            }
        } else {
            let raw = decode_non_grammar_block(
                frame_bytes,
                idx,
                info.raw_size,
                &DecodeOptions::default(),
            )?;
            raw.split_inclusive(|&b| b == b'\n').count()
        };
        total = total.checked_add(count).ok_or(Error::SizeOverflow)?;
    }
    Ok(total)
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
    let parsed = Frame::parse(frame_bytes, options.max_output)?;
    let mut out = Vec::new();
    let mut cur_global = 0usize;

    for (blk_idx, info) in parsed.blocks().enumerate() {
        if cur_global >= end_row {
            break;
        }
        if info.kind == BLOCK_COLUMN {
            let payload =
                &frame_bytes[info.payload_offset..info.payload_offset + info.payload_size];
            if let Some(blk_rows) = crate::columns::grammar_block_row_count(payload)? {
                let next_global = cur_global + blk_rows;
                if next_global > start_row && cur_global < end_row {
                    let local_start = start_row.saturating_sub(cur_global);
                    let local_end = (end_row - cur_global).min(blk_rows);
                    let _ = crate::columns::unpack_grammar_row_range(
                        payload,
                        local_start,
                        local_end,
                        &mut out,
                    )?;
                }
                cur_global = next_global;
                continue;
            }
        }
        let raw = decode_non_grammar_block(frame_bytes, blk_idx, info.raw_size, options)?;
        for line in raw.split_inclusive(|&b| b == b'\n') {
            if cur_global >= start_row && cur_global < end_row {
                out.extend_from_slice(line);
            }
            cur_global += 1;
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
    options.validate()?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let parsed = Frame::parse(frame_bytes, options.max_output)?;
    let mut out = Vec::new();
    let mut cur_global = 0usize;

    for (blk_idx, info) in parsed.blocks().enumerate() {
        if out.len() >= limit {
            break;
        }
        if info.kind == BLOCK_COLUMN {
            let payload =
                &frame_bytes[info.payload_offset..info.payload_offset + info.payload_size];
            if let Some(blk_rows) = crate::columns::search_grammar_block(
                payload,
                blk_idx,
                cur_global,
                needle,
                limit,
                &mut out,
            )? {
                cur_global += blk_rows;
                continue;
            }
        }
        let raw = decode_non_grammar_block(frame_bytes, blk_idx, info.raw_size, options)?;
        for (row_idx, line) in raw.split_inclusive(|&b| b == b'\n').enumerate() {
            if contains_bytes(line, needle) {
                out.push(RowMatch {
                    block_index: blk_idx,
                    row_index: row_idx,
                    global_row_index: cur_global + row_idx,
                    bytes: line.to_vec(),
                });
                if out.len() >= limit {
                    break;
                }
            }
        }
        cur_global += raw.split_inclusive(|&b| b == b'\n').count();
    }
    Ok(out)
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
mod tests {
    use super::*;
    use crate::{compress, DecodeOptions, Options};

    fn sample_ndjson(rows: usize) -> Vec<u8> {
        let services = ["auth", "payment", "inventory", "gateway"];
        let levels = ["INFO", "INFO", "WARN", "ERROR"];
        let mut out = Vec::new();
        for i in 0..rows {
            let svc = services[i % services.len()];
            let lvl = if i == rows - 7 {
                "CRITICAL"
            } else {
                levels[i % levels.len()]
            };
            let line = format!(
                "{{\"ts\":\"2026-10-17T19:{:02}:{:02}Z\",\"seq\":{},\"level\":\"{}\",\"service\":\"{}\",\"latency_us\":{}}}\n",
                (i / 60) % 60,
                i % 60,
                1000 + i,
                lvl,
                svc,
                100 + (i * 7) % 500
            );
            out.extend_from_slice(line.as_bytes());
        }
        out
    }

    #[test]
    fn random_access_and_pushdown_search_roundtrip() {
        let raw = sample_ndjson(2500);
        let opts = Options::default();
        let dec_opts = DecodeOptions::default();
        let frame = compress(&raw, &opts).unwrap();

        let expected_lines: Vec<&[u8]> = raw.split_inclusive(|&b| b == b'\n').collect();
        let total_rows = frame_row_count(&frame).unwrap();
        assert!(total_rows >= expected_lines.len());

        // Single row random access in block 0
        let row0 = extract_block_row(&frame, 0, 0, &dec_opts).unwrap();
        assert_eq!(row0, expected_lines[0]);
        let row42 = extract_block_row(&frame, 0, 42, &dec_opts).unwrap();
        assert_eq!(row42, expected_lines[42]);

        // Block row range extraction
        let range_10_25 = extract_block_row_range(&frame, 0, 10, 25, &dec_opts).unwrap();
        let expected_10_25: Vec<u8> = expected_lines[10..25].concat();
        assert_eq!(range_10_25, expected_10_25);

        // Full-frame extract_row_range across all rows reconstructs the exact original bytes!
        let all_rows = extract_row_range(&frame, 0, total_rows, &dec_opts).unwrap();
        assert_eq!(all_rows, raw);

        // Compressed-domain find_first on rare dictionary entry ("CRITICAL")
        let crit = find_first(&frame, b"CRITICAL", &dec_opts)
            .unwrap()
            .expect("CRITICAL row");
        assert!(contains_bytes(&crit.bytes, b"\"level\":\"CRITICAL\""));
        assert!(contains_bytes(&crit.bytes, b"\"seq\":3493"));

        // Boundary-spanning needle ("\"level\":\"CRITICAL\"")
        let crit_span = find_first(&frame, b"\"level\":\"CRITICAL\"", &dec_opts)
            .unwrap()
            .expect("boundary span match");
        assert_eq!(crit_span.global_row_index, crit.global_row_index);
        assert_eq!(crit_span.bytes, crit.bytes);

        // Static segment needle ("\"service\":\"") -> instant hit at row 0
        let static_hit = find_first(&frame, b"\"service\":\"", &dec_opts)
            .unwrap()
            .expect("static segment hit");
        assert_eq!(static_hit.global_row_index, 0);
        assert_eq!(static_hit.bytes, expected_lines[0]);

        // Non-existent needle -> fast O(1) rejection returns None
        assert!(find_first(&frame, b"NON_EXISTENT_TOKEN_XYZ", &dec_opts)
            .unwrap()
            .is_none());

        // filter_rows with limit
        let payment_rows = filter_rows(&frame, b"\"service\":\"payment\"", 10, &dec_opts).unwrap();
        assert_eq!(payment_rows.len(), 10);
        for m in &payment_rows {
            assert!(contains_bytes(&m.bytes, b"\"service\":\"payment\""));
        }

        // C FFI verification
        unsafe {
            let ffi_cnt = vxg_frame_row_count(frame.as_ptr(), frame.len());
            assert_eq!(ffi_cnt as usize, total_rows);

            let mut buf = [0u8; 512];
            let mut found_idx = 0usize;
            let n = vxg_find_first_into(
                frame.as_ptr(),
                frame.len(),
                b"CRITICAL".as_ptr(),
                8,
                buf.as_mut_ptr(),
                buf.len(),
                &raw mut found_idx,
            );
            assert!(n > 0);
            assert_eq!(found_idx, crit.global_row_index);
            assert_eq!(&buf[..n as usize], crit.bytes.as_slice());
        }
    }
}
