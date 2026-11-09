//! Apache Arrow C Stream Interface (`ArrowArrayStream`) & compressed-domain projection/`WHERE`
//! pushdown FFI (`vxg_arrow_export_stream`, `vxg_arrow_query`, `vxg_arrow_query_stream`).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::{null, null_mut};

use crate::{
    arrow::{export_arrow_batch_where, parse_arrow_where, ArrowWhere, DEFAULT_ARROW_BATCH_ROWS},
    arrow_scan::target_to_owned_and_name,
    arrow_stream_scan::ArrowWindowStream,
    ops::parse_col_list,
    sql::{OwnedCol, SelectExpr, SelectProjection, SqlQuery},
    DecodeOptions, Error, Result,
};

use super::arrow::{
    export_batch_to_c_array, export_batch_to_c_schema, parse_opt_query, ArrowArray,
    ArrowArrayStream, ArrowSchema,
};
use super::{map_error, resolve_threads, slice_from_raw, VXG_ERR_INTERNAL, VXG_ERR_NULL_PTR, VXG_OK};

struct StreamPrivate {
    cursor: ArrowWindowStream,
    last_error: Option<CString>,
}

unsafe extern "C" fn stream_get_schema(
    stream: *mut ArrowArrayStream,
    out_schema: *mut ArrowSchema,
) -> c_int {
    if stream.is_null() || out_schema.is_null() {
        return 22;
    }
    let st = unsafe { &mut *stream };
    if st.release.is_none() || st.private_data.is_null() {
        return 22;
    }
    let priv_ref = unsafe { &*(st.private_data.cast::<StreamPrivate>()) };
    export_batch_to_c_schema(&priv_ref.cursor.schema_template, unsafe { &mut *out_schema });
    0
}

unsafe extern "C" fn stream_get_next(
    stream: *mut ArrowArrayStream,
    out_array: *mut ArrowArray,
) -> c_int {
    if stream.is_null() || out_array.is_null() {
        return 22;
    }
    let st = unsafe { &mut *stream };
    if st.release.is_none() || st.private_data.is_null() {
        return 22;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let priv_mut = unsafe { &mut *(st.private_data.cast::<StreamPrivate>()) };
        match priv_mut.cursor.next_batch() {
            Ok(Some(batch)) => {
                priv_mut.last_error = None;
                export_batch_to_c_array(batch, unsafe { &mut *out_array });
                0
            }
            Ok(None) => {
                priv_mut.last_error = None;
                unsafe {
                    *out_array = ArrowArray {
                        length: 0,
                        null_count: 0,
                        offset: 0,
                        n_buffers: 0,
                        n_children: 0,
                        buffers: null_mut(),
                        children: null_mut(),
                        dictionary: null_mut(),
                        release: None,
                        private_data: null_mut(),
                    };
                }
                0
            }
            Err(e) => {
                priv_mut.last_error = CString::new(format!("{e}")).ok();
                5 // EIO
            }
        }
    }))
    .unwrap_or(5)
}

unsafe extern "C" fn stream_get_last_error(stream: *mut ArrowArrayStream) -> *const c_char {
    if stream.is_null() {
        return null();
    }
    let st = unsafe { &*stream };
    if st.release.is_none() || st.private_data.is_null() {
        return null();
    }
    let priv_ref = unsafe { &*(st.private_data.cast::<StreamPrivate>()) };
    priv_ref.last_error.as_ref().map_or(null(), |c| c.as_ptr())
}

unsafe extern "C" fn release_arrow_stream(stream: *mut ArrowArrayStream) {
    if stream.is_null() {
        return;
    }
    let st = unsafe { &mut *stream };
    if st.release.is_none() {
        return;
    }
    if !st.private_data.is_null() {
        unsafe { drop(Box::from_raw(st.private_data.cast::<StreamPrivate>())) };
    }
    st.get_schema = None;
    st.get_next = None;
    st.get_last_error = None;
    st.release = None;
    st.private_data = null_mut();
}

fn init_c_stream(cursor: ArrowWindowStream, out_stream: &mut ArrowArrayStream) {
    let priv_box = Box::new(StreamPrivate { cursor, last_error: None });
    *out_stream = ArrowArrayStream {
        get_schema: Some(stream_get_schema),
        get_next: Some(stream_get_next),
        get_last_error: Some(stream_get_last_error),
        release: Some(release_arrow_stream),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    };
}

pub(crate) fn parse_cols_spec(raw: &str) -> Result<Option<(Vec<String>, Vec<OwnedCol>)>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "*" {
        return Ok(None);
    }
    let sql_probe = format!("SELECT {trimmed} FROM frame");
    if let Ok(q) = SqlQuery::parse(&sql_probe) {
        if q.where_clause.is_none()
            && q.group_by.is_empty()
            && q.order_by.is_empty()
            && q.limit.is_none()
        {
            match q.projection {
                SelectProjection::Star => return Ok(None),
                SelectProjection::Items(items) if !items.is_empty() => {
                    let mut names = Vec::with_capacity(items.len());
                    let mut cols = Vec::with_capacity(items.len());
                    let mut ok = true;
                    for it in items {
                        if let SelectExpr::Col(ref c) = it.expr {
                            names.push(it.header_name());
                            cols.push(c.clone());
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        return Ok(Some((names, cols)));
                    }
                }
                _ => {}
            }
        }
    }
    let targets = parse_col_list(trimmed)?;
    let mut names = Vec::with_capacity(targets.len());
    let mut cols = Vec::with_capacity(targets.len());
    for t in targets {
        let (n, o) = target_to_owned_and_name(t);
        names.push(n);
        cols.push(o);
    }
    Ok(Some((names, cols)))
}

unsafe fn parse_opt_cols(cols_cstr: *const c_char) -> Result<Option<(Vec<String>, Vec<OwnedCol>)>> {
    if cols_cstr.is_null() {
        return Ok(None);
    }
    let Ok(s) = (unsafe { CStr::from_ptr(cols_cstr) }).to_str() else {
        return Err(Error::InvalidOption("invalid UTF-8 column list"));
    };
    parse_cols_spec(s)
}

unsafe fn parse_opt_where(where_cstr: *const c_char) -> Result<Option<ArrowWhere>> {
    if where_cstr.is_null() {
        return Ok(None);
    }
    let Ok(s) = (unsafe { CStr::from_ptr(where_cstr) }).to_str() else {
        return Err(Error::InvalidOption("invalid UTF-8 WHERE predicate"));
    };
    if s.trim().is_empty() {
        Ok(None)
    } else {
        parse_arrow_where(s).map(Some)
    }
}

/// Export a VXG frame (with optional compressed-domain SQL `query_cstr`) into an Apache Arrow
/// C Stream Interface `ArrowArrayStream` (consumable zero-copy by DuckDB, Polars, PyArrow).
#[no_mangle]
pub unsafe extern "C" fn vxg_arrow_export_stream(
    frame_ptr: *const u8,
    frame_len: usize,
    query_cstr: *const c_char,
    threads: usize,
    out_stream: *mut ArrowArrayStream,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if out_stream.is_null() {
            return VXG_ERR_NULL_PTR;
        }
        let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
            return VXG_ERR_NULL_PTR;
        };
        let q_opt = match unsafe { parse_opt_query(query_cstr) } {
            Ok(q) => q,
            Err(e) => return map_error(e),
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            max_output: usize::MAX,
            verify_checksum: false,
            ..DecodeOptions::default()
        };
        match ArrowWindowStream::open_sql(frame, q_opt.as_ref(), dec) {
            Ok(cursor) => {
                init_c_stream(cursor, unsafe { &mut *out_stream });
                VXG_OK
            }
            Err(e) => map_error(e),
        }
    }))
    .unwrap_or(VXG_ERR_INTERNAL)
}

/// Export a compressed-domain projected (`cols_cstr`) and filtered (`where_cstr`) VXG frame
/// into an Apache Arrow C Data Interface `(ArrowSchema, ArrowArray)` pair.
#[no_mangle]
pub unsafe extern "C" fn vxg_arrow_query(
    frame_ptr: *const u8,
    frame_len: usize,
    cols_cstr: *const c_char,
    where_cstr: *const c_char,
    threads: usize,
    out_schema: *mut ArrowSchema,
    out_array: *mut ArrowArray,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if out_schema.is_null() || out_array.is_null() {
            return VXG_ERR_NULL_PTR;
        }
        let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
            return VXG_ERR_NULL_PTR;
        };
        let cols_opt = match unsafe { parse_opt_cols(cols_cstr) } {
            Ok(c) => c,
            Err(e) => return map_error(e),
        };
        let where_opt = match unsafe { parse_opt_where(where_cstr) } {
            Ok(w) => w,
            Err(e) => return map_error(e),
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            max_output: usize::MAX,
            verify_checksum: false,
            ..DecodeOptions::default()
        };
        let run_batch = |w_expr: Option<&crate::PredExpr<'_>>| {
            if let Some((_, ref owned)) = cols_opt {
                let targets: Vec<_> = owned.iter().map(OwnedCol::as_target).collect();
                let mut b = export_arrow_batch_where(frame, Some(&targets), w_expr, None, &dec)?;
                if let Some((ref names, _)) = cols_opt {
                    for (col, name) in b.columns.iter_mut().zip(names.iter()) {
                        col.name.clone_from(name);
                    }
                }
                Ok(b)
            } else {
                export_arrow_batch_where(frame, None, w_expr, None, &dec)
            }
        };
        let res = match &where_opt {
            Some(aw) => aw.with_pred_expr(|w| run_batch(Some(w))),
            None => run_batch(None),
        };
        match res {
            Ok(batch) => {
                export_batch_to_c_schema(&batch, unsafe { &mut *out_schema });
                export_batch_to_c_array(batch, unsafe { &mut *out_array });
                VXG_OK
            }
            Err(e) => map_error(e),
        }
    }))
    .unwrap_or(VXG_ERR_INTERNAL)
}

/// Export a compressed-domain projected (`cols_cstr`) and filtered (`where_cstr`) VXG frame
/// into a windowed multi-batch Apache Arrow C Stream Interface `ArrowArrayStream`.
/// `frame_ptr[0..frame_len]` must remain valid until `out_stream->release(out_stream)` is called.
#[no_mangle]
pub unsafe extern "C" fn vxg_arrow_query_stream(
    frame_ptr: *const u8,
    frame_len: usize,
    cols_cstr: *const c_char,
    where_cstr: *const c_char,
    threads: usize,
    out_stream: *mut ArrowArrayStream,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if out_stream.is_null() {
            return VXG_ERR_NULL_PTR;
        }
        let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
            return VXG_ERR_NULL_PTR;
        };
        let cols_opt = match unsafe { parse_opt_cols(cols_cstr) } {
            Ok(c) => c,
            Err(e) => return map_error(e),
        };
        let where_opt = match unsafe { parse_opt_where(where_cstr) } {
            Ok(w) => w,
            Err(e) => return map_error(e),
        };
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            max_output: usize::MAX,
            verify_checksum: false,
            ..DecodeOptions::default()
        };
        match ArrowWindowStream::open(
            frame,
            cols_opt,
            where_opt,
            None,
            DEFAULT_ARROW_BATCH_ROWS,
            dec,
        ) {
            Ok(cursor) => {
                init_c_stream(cursor, unsafe { &mut *out_stream });
                VXG_OK
            }
            Err(e) => map_error(e),
        }
    }))
    .unwrap_or(VXG_ERR_INTERNAL)
}
