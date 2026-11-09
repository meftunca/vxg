//! Extended C ABI (`extern "C"`) for zero-output-alloc verification, zero-copy `mmap` file I/O,
//! and compressed-domain analytics (`WHERE`, `SELECT`, `ORDER BY` Top-K, `AGGREGATE`, `PROJECT`).

use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::{
    aggregate_column, compress_file, count_where, decompress_file, filter_frame_where,
    filter_where, frame_row_count, group_by, parse_col_list, parse_col_target, project_frame,
    select_columns, select_top_k_by, top_k_by, verify_file, verify_frame, with_opt_where,
    DecodeOptions, GroupKey, Options, ParsedWhere, PredExpr, Simd, SortDir,
};

use super::{
    map_error, resolve_block_size, resolve_threads, slice_from_raw, VXG_ERR_BUFFER_TOO_SMALL,
    VXG_ERR_INTERNAL, VXG_ERR_INVALID_OPTION, VXG_ERR_NULL_PTR, VXG_OK,
};

/// C ABI scalar aggregation result populated by [`vxg_aggregate_col`].
/// Naturally 8-byte aligned (64 bytes total) across C99/C++11, Go, Python `ctypes`, and WASM.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct vxg_agg_stats_t {
    /// Total matching records (`COUNT(*)`).
    pub row_count: u64,
    /// Matching records with a canonical unsigned 64-bit integer value (`COUNT(col)`).
    pub num_count: u64,
    /// `SUM(col)` as IEEE-754 `double` (exact up to $2^{53}$).
    pub sum: f64,
    /// Low 64 bits of exact 128-bit integer `SUM(col)`.
    pub sum_lo: u64,
    /// High 64 bits of exact 128-bit integer `SUM(col)`.
    pub sum_hi: u64,
    /// `MIN(col)` when `has_minmax != 0`, else `0`.
    pub min_val: u64,
    /// `MAX(col)` when `has_minmax != 0`, else `0`.
    pub max_val: u64,
    /// `1` when `num_count > 0`, else `0`.
    pub has_minmax: i32,
    /// Explicit zeroed padding for deterministic 64-byte layout.
    pub _reserved: i32,
}

#[inline]
unsafe fn cstr_req<'a>(ptr: *const c_char) -> Result<&'a str, isize> {
    if ptr.is_null() {
        return Err(VXG_ERR_NULL_PTR);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| VXG_ERR_INVALID_OPTION)
}

#[inline]
unsafe fn cstr_opt<'a>(ptr: *const c_char) -> Result<Option<&'a str>, isize> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| VXG_ERR_INVALID_OPTION)?;
    let trimmed = s.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s))
    }
}

#[inline]
fn into_ffi_buf(bytes: Vec<u8>, out_len: *mut usize) -> *mut u8 {
    let mut boxed = bytes.into_boxed_slice();
    let len = boxed.len();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    unsafe {
        out_len.write_unaligned(len);
    }
    ptr
}

/// Validates structural integrity and per-block CRC32C checksums across single or concatenated
/// `VXG1` frames in `src[..src_len]` using bounded thread-local scratch (`~512 KiB` per worker)
/// without allocating output memory.
/// Returns total verified uncompressed byte count (`>= 0`), or negative `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_verify_frame(
    src: *const u8,
    src_len: usize,
    threads: usize,
    verify_checksum: i32,
) -> isize {
    let Some(bytes) = (unsafe { slice_from_raw(src, src_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: verify_checksum != 0,
        };
        match verify_frame(bytes, &dec) {
            Ok(raw_bytes) => isize::try_from(raw_bytes).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Compresses `input_path` to `output_path` using VXG's hybrid file/`mmap` pipeline.
/// - `force`: non-zero overwrites `output_path` if it already exists.
/// Returns compressed output byte count (`>= 0`), or negative `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_compress_file(
    input_path: *const c_char,
    output_path: *const c_char,
    threads: usize,
    block_size: usize,
    checksum: i32,
    force: i32,
) -> isize {
    let in_p = match unsafe { cstr_req(input_path) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let out_p = match unsafe { cstr_req(output_path) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let opt = Options {
            block_size: resolve_block_size(block_size),
            threads: resolve_threads(threads),
            checksum: checksum != 0,
            acceleration: 1,
            simd: Simd::Auto,
            legacy_v2: false,
            record_aligned: false,
            level: 3,
        };
        match compress_file(in_p, out_p, &opt, force != 0) {
            Ok((_, out_bytes)) => isize::try_from(out_bytes).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Decompresses `input_path` to `output_path` using zero-copy `mmap` on `.vxg` frame inputs.
/// - `force`: non-zero overwrites `output_path` if it already exists.
/// Returns uncompressed byte count written (`>= 0`), or negative `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_decompress_file(
    input_path: *const c_char,
    output_path: *const c_char,
    threads: usize,
    verify_checksum: i32,
    force: i32,
) -> isize {
    let in_p = match unsafe { cstr_req(input_path) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let out_p = match unsafe { cstr_req(output_path) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: verify_checksum != 0,
        };
        match decompress_file(in_p, out_p, &dec, force != 0) {
            Ok(raw_bytes) => isize::try_from(raw_bytes).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Verifies `input_path` via zero-copy read-only `mmap` and bounded L2 scratch without
/// allocating output memory. Returns verified uncompressed byte count (`>= 0`), or `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_verify_file(
    input_path: *const c_char,
    threads: usize,
    verify_checksum: i32,
) -> isize {
    let in_p = match unsafe { cstr_req(input_path) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: verify_checksum != 0,
        };
        match verify_file(in_p, &dec) {
            Ok(raw_bytes) => isize::try_from(raw_bytes).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
            Err(e) => map_error(e),
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Counts records matching compound boolean expression `where_cstr` (`NULL` or `""` counts all
/// records) directly in the compressed domain. Returns matching row count (`>= 0`), or `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_count_where_str(
    frame_ptr: *const u8,
    frame_len: usize,
    where_cstr: *const c_char,
    threads: usize,
) -> isize {
    let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let opt_w_str = match unsafe { cstr_opt(where_cstr) } {
        Ok(v) => v,
        Err(code) => return code,
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: true,
        };
        match opt_w_str {
            None => match frame_row_count(frame) {
                Ok(n) => isize::try_from(n).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
                Err(e) => map_error(e),
            },
            Some(ws) => {
                let pw = match ParsedWhere::parse(ws) {
                    Ok(p) => p,
                    Err(e) => return map_error(e),
                };
                match pw.with_expr(|expr| count_where(frame, expr, &dec)) {
                    Ok(n) => isize::try_from(n).unwrap_or(VXG_ERR_BUFFER_TOO_SMALL),
                    Err(e) => map_error(e),
                }
            }
        }
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Computes compressed-domain scalar aggregations (`COUNT(*)`, `COUNT(col)`, `SUM`, `MIN`, `MAX`)
/// on `col_cstr` over records matching optional `where_cstr` (`NULL` or `""` = all records),
/// writing results to `*out_stats`. Returns `VXG_OK` (`0`) on success, or negative `VXG_ERR_*`.
#[no_mangle]
pub unsafe extern "C" fn vxg_aggregate_col(
    frame_ptr: *const u8,
    frame_len: usize,
    col_cstr: *const c_char,
    where_cstr: *const c_char,
    threads: usize,
    out_stats: *mut vxg_agg_stats_t,
) -> isize {
    if out_stats.is_null() {
        return VXG_ERR_NULL_PTR;
    }
    let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
        return VXG_ERR_NULL_PTR;
    };
    let col_str = match unsafe { cstr_req(col_cstr) } {
        Ok(s) => s,
        Err(code) => return code,
    };
    let opt_w_str = match unsafe { cstr_opt(where_cstr) } {
        Ok(v) => v,
        Err(code) => return code,
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> isize {
        let acol = match parse_col_target(col_str.trim()) {
            Ok(c) => c,
            Err(e) => return map_error(e),
        };
        let pw = match opt_w_str.map(ParsedWhere::parse).transpose() {
            Ok(p) => p,
            Err(e) => return map_error(e),
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: true,
        };
        let st = match with_opt_where(pw.as_ref(), |w| aggregate_column(frame, w, acol, &dec)) {
            Ok(s) => s,
            Err(e) => return map_error(e),
        };
        let c_st = vxg_agg_stats_t {
            row_count: st.row_count,
            num_count: st.num_count,
            sum: if st.has_signed_or_dec {
                st.sum_dec().to_f64()
            } else {
                st.sum as f64
            },
            sum_lo: st.sum as u64,
            sum_hi: (st.sum >> 64) as u64,
            min_val: st.min.unwrap_or(0),
            max_val: st.max.unwrap_or(0),
            has_minmax: i32::from(st.num_count > 0),
            _reserved: 0,
        };
        unsafe {
            out_stats.write_unaligned(c_st);
        }
        VXG_OK
    }));
    res.unwrap_or(VXG_ERR_INTERNAL)
}

/// Executes a compressed-domain query (`SELECT cols WHERE expr ORDER BY col [ASC|DESC] LIMIT k`)
/// and returns a newly allocated buffer containing the matching (or projected) records.
/// - `cols_cstr`: optional comma-separated columns (`"ts,service,latency_ms"` or `NULL` for full rows).
/// - `where_cstr`: optional boolean filter (`"status>=500 && service==\"auth\""` or `NULL`).
/// - `order_by_cstr`: optional sort column (`"latency_ms"` or `NULL` for storage order).
/// - `desc`: non-zero sorts descending (`DESC`), `0` sorts ascending (`ASC`).
/// - `limit`: maximum records to return (pass `SIZE_MAX` for all matching records).
/// Free the returned pointer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
#[no_mangle]
pub unsafe extern "C" fn vxg_select_alloc(
    frame_ptr: *const u8,
    frame_len: usize,
    cols_cstr: *const c_char,
    where_cstr: *const c_char,
    order_by_cstr: *const c_char,
    desc: i32,
    limit: usize,
    threads: usize,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_cols_s) = (unsafe { cstr_opt(cols_cstr) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_where_s) = (unsafe { cstr_opt(where_cstr) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_order_s) = (unsafe { cstr_opt(order_by_cstr) }) else {
        return std::ptr::null_mut();
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> *mut u8 {
        let Ok(cols) = opt_cols_s.map(parse_col_list).transpose() else {
            return std::ptr::null_mut();
        };
        let Ok(pw) = opt_where_s.map(ParsedWhere::parse).transpose() else {
            return std::ptr::null_mut();
        };
        let Ok(order_col) = opt_order_s.map(|s| parse_col_target(s.trim())).transpose() else {
            return std::ptr::null_mut();
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: true,
        };
        let dir = if desc != 0 {
            SortDir::Desc
        } else {
            SortDir::Asc
        };

        let out_res = with_opt_where(pw.as_ref(), |w| match (cols.as_deref(), order_col) {
            (Some(cs), Some(oc)) => select_top_k_by(frame, w, cs, oc, dir, limit, &dec),
            (None, Some(oc)) => top_k_by(frame, w, oc, dir, limit, &dec).map(|rows| {
                let cap: usize = rows.iter().map(|r| r.bytes.len()).sum();
                let mut buf = Vec::with_capacity(cap);
                for r in rows {
                    buf.extend_from_slice(&r.bytes);
                }
                buf
            }),
            (Some(cs), None) => select_columns(frame, w, cs, limit, &dec),
            (None, None) => {
                let expr = w.unwrap_or(&PredExpr::True);
                filter_where(frame, expr, limit, &dec).map(|rows| {
                    let cap: usize = rows.iter().map(|r| r.bytes.len()).sum();
                    let mut buf = Vec::with_capacity(cap);
                    for r in rows {
                        buf.extend_from_slice(&r.bytes);
                    }
                    buf
                })
            }
        });
        match out_res {
            Ok(bytes) => into_ffi_buf(bytes, out_len),
            Err(_) => std::ptr::null_mut(),
        }
    }));
    res.unwrap_or(std::ptr::null_mut())
}

/// Produces a new canonical compressed `VXG1` frame by projecting `cols_cstr` (and/or filtering
/// by `where_cstr`) directly in the compressed domain, splicing surviving column byte streams
/// without re-encoding when `where_cstr` is `NULL` or matches 100% of a block.
/// Free the returned pointer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
#[no_mangle]
pub unsafe extern "C" fn vxg_project_frame_alloc(
    frame_ptr: *const u8,
    frame_len: usize,
    cols_cstr: *const c_char,
    where_cstr: *const c_char,
    threads: usize,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_cols_s) = (unsafe { cstr_opt(cols_cstr) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_where_s) = (unsafe { cstr_opt(where_cstr) }) else {
        return std::ptr::null_mut();
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> *mut u8 {
        let Ok(cols) = opt_cols_s.map(parse_col_list).transpose() else {
            return std::ptr::null_mut();
        };
        let Ok(pw) = opt_where_s.map(ParsedWhere::parse).transpose() else {
            return std::ptr::null_mut();
        };
        let th = resolve_threads(threads);
        let enc = Options {
            threads: th,
            ..Options::default()
        };
        let dec = DecodeOptions {
            threads: th,
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: true,
        };
        let out_res = match (cols.as_deref(), pw.as_ref()) {
            (Some(cs), opt_pw) => {
                with_opt_where(opt_pw, |w| project_frame(frame, w, cs, &enc, &dec))
            }
            (None, Some(p)) => p.with_expr(|w| filter_frame_where(frame, w, &enc, &dec)),
            (None, None) => return std::ptr::null_mut(),
        };
        match out_res {
            Ok(bytes) => into_ffi_buf(bytes, out_len),
            Err(_) => std::ptr::null_mut(),
        }
    }));
    res.unwrap_or(std::ptr::null_mut())
}

/// Compressed-domain `GROUP BY group_col` with optional `agg_col` (`SUM`/`MIN`/`MAX`) and optional
/// `where_cstr`, returning a TSV byte buffer (`key\trow_count\tnum_count\tsum\tmin\tmax\n` per group).
/// Free the returned pointer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
#[no_mangle]
pub unsafe extern "C" fn vxg_group_by_alloc(
    frame_ptr: *const u8,
    frame_len: usize,
    group_col_cstr: *const c_char,
    agg_col_cstr: *const c_char,
    where_cstr: *const c_char,
    threads: usize,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
        return std::ptr::null_mut();
    };
    let Ok(gcol_s) = (unsafe { cstr_req(group_col_cstr) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_acol_s) = (unsafe { cstr_opt(agg_col_cstr) }) else {
        return std::ptr::null_mut();
    };
    let Ok(opt_where_s) = (unsafe { cstr_opt(where_cstr) }) else {
        return std::ptr::null_mut();
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> *mut u8 {
        let Ok(gcol) = parse_col_target(gcol_s.trim()) else {
            return std::ptr::null_mut();
        };
        let Ok(acol) = opt_acol_s.map(|s| parse_col_target(s.trim())).transpose() else {
            return std::ptr::null_mut();
        };
        let Ok(pw) = opt_where_s.map(ParsedWhere::parse).transpose() else {
            return std::ptr::null_mut();
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: true,
        };
        let Ok(groups) = with_opt_where(pw.as_ref(), |w| group_by(frame, w, gcol, acol, &dec))
        else {
            return std::ptr::null_mut();
        };
        let mut out = Vec::with_capacity(groups.len() * 48);
        for g in groups {
            let kb = match &g.key {
                GroupKey::Str(b) | GroupKey::RawJson(b) => b.as_slice(),
            };
            out.extend_from_slice(kb);
            let line = format!(
                "\t{}\t{}\t{}\t{}\t{}\n",
                g.stats.row_count,
                g.stats.num_count,
                g.stats.sum,
                g.stats.min.unwrap_or(0),
                g.stats.max.unwrap_or(0),
            );
            out.extend_from_slice(line.as_bytes());
        }
        into_ffi_buf(out, out_len)
    }));
    res.unwrap_or(std::ptr::null_mut())
}

/// Executes a compressed-domain SQL `SELECT` statement (`query_cstr`) directly on `frame_ptr[..frame_len]`.
/// - `format`: `0` = TSV, `1` = CSV, `2` = NDJSON (`SqlFormat`).
/// - `no_header`: non-zero omits the header line on TSV/CSV output.
/// Free the returned pointer with `vxg_free(ptr, *out_len)`. Returns `NULL` on error.
#[no_mangle]
pub unsafe extern "C" fn vxg_sql_alloc(
    frame_ptr: *const u8,
    frame_len: usize,
    query_cstr: *const c_char,
    format: i32,
    no_header: i32,
    threads: usize,
    out_len: *mut usize,
) -> *mut u8 {
    if out_len.is_null() {
        return std::ptr::null_mut();
    }
    let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
        return std::ptr::null_mut();
    };
    let Ok(sql_s) = (unsafe { cstr_req(query_cstr) }) else {
        return std::ptr::null_mut();
    };
    let fmt = match format {
        0 => crate::SqlFormat::Tsv,
        1 => crate::SqlFormat::Csv,
        2 => crate::SqlFormat::Json,
        _ => return std::ptr::null_mut(),
    };
    let res = catch_unwind(AssertUnwindSafe(|| -> *mut u8 {
        let Ok(q) = crate::SqlQuery::parse(sql_s) else {
            return std::ptr::null_mut();
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            simd: Simd::Auto,
            max_output: usize::MAX,
            verify_checksum: true,
        };
        match crate::execute_sql(frame, &q, fmt, no_header != 0, &dec) {
            Ok(bytes) => into_ffi_buf(bytes, out_len),
            Err(_) => std::ptr::null_mut(),
        }
    }));
    res.unwrap_or(std::ptr::null_mut())
}
