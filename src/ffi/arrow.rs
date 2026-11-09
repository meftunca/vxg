//! Zero-dependency Apache Arrow C Data Interface (`ArrowSchema`, `ArrowArray`) and
//! C Stream Interface (`ArrowArrayStream`) for zero-copy integration with DuckDB, Polars,
//! PyArrow, DataFusion, and C/C++ analytical engines.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::{null, null_mut};

use crate::{
    arrow::{export_arrow_batch, export_arrow_ipc, ArrowBatch, ArrowCol, ArrowData},
    DecodeOptions, Error, Result, SqlQuery,
};

use super::{
    map_error, resolve_threads, slice_from_raw, VXG_ERR_INTERNAL, VXG_ERR_INVALID_OPTION,
    VXG_ERR_NULL_PTR, VXG_OK,
};

pub const ARROW_FLAG_DICTIONARY_ORDERED: i64 = 1;
pub const ARROW_FLAG_NULLABLE: i64 = 2;
pub const ARROW_FLAG_MAP_KEYS_SORTED: i64 = 4;

/// Official Apache Arrow C Data Interface `ArrowSchema` ABI layout.
#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

/// Official Apache Arrow C Data Interface `ArrowArray` ABI layout.
#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

/// Official Apache Arrow C Stream Interface `ArrowArrayStream` ABI layout.
#[repr(C)]
pub struct ArrowArrayStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Align64([u8; 64]);

fn alloc_aligned64(bytes: &[u8]) -> Vec<Align64> {
    let blocks = bytes.len().div_ceil(64).max(1);
    let mut buf = vec![Align64([0u8; 64]); blocks];
    if !bytes.is_empty() {
        // SAFETY: `buf` has `blocks * 64 >= bytes.len()` contiguous initialized `u8` bytes.
        let dst = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), bytes.len()) };
        dst.copy_from_slice(bytes);
    }
    buf
}

fn alloc_aligned64_pod<T: Copy>(slice: &[T]) -> Vec<Align64> {
    let byte_len = std::mem::size_of_val(slice);
    // SAFETY: only called on plain numeric types (`i32`, `i64`, `f64`) with initialized memory.
    let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), byte_len) };
    alloc_aligned64(bytes)
}

struct SchemaPrivate {
    _name_cstr: Vec<u8>,
    children_ptrs: Vec<*mut ArrowSchema>,
}

unsafe extern "C" fn release_arrow_schema(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }
    let s = unsafe { &mut *schema };
    if s.release.is_none() {
        return;
    }
    if !s.private_data.is_null() {
        let priv_box = unsafe { Box::from_raw(s.private_data.cast::<SchemaPrivate>()) };
        for &child_ptr in &priv_box.children_ptrs {
            if !child_ptr.is_null() {
                if let Some(rel) = unsafe { (*child_ptr).release } {
                    unsafe { rel(child_ptr) };
                }
                unsafe { drop(Box::from_raw(child_ptr)) };
            }
        }
    }
    s.release = None;
    s.private_data = null_mut();
}

struct ArrayPrivate {
    _aligned_bufs: Vec<Vec<Align64>>,
    _buffer_ptrs: Vec<*const c_void>,
    children_ptrs: Vec<*mut ArrowArray>,
}

unsafe extern "C" fn release_arrow_array(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }
    let a = unsafe { &mut *array };
    if a.release.is_none() {
        return;
    }
    if !a.private_data.is_null() {
        let priv_box = unsafe { Box::from_raw(a.private_data.cast::<ArrayPrivate>()) };
        for &child_ptr in &priv_box.children_ptrs {
            if !child_ptr.is_null() {
                if let Some(rel) = unsafe { (*child_ptr).release } {
                    unsafe { rel(child_ptr) };
                }
                unsafe { drop(Box::from_raw(child_ptr)) };
            }
        }
    }
    a.release = None;
    a.private_data = null_mut();
}

fn build_col_schema(col: &ArrowCol) -> ArrowSchema {
    let mut name_cstr = Vec::with_capacity(col.name.len() + 1);
    for &b in col.name.as_bytes() {
        if b != 0 {
            name_cstr.push(b);
        }
    }
    name_cstr.push(0);
    let name_ptr = name_cstr.as_ptr().cast::<c_char>();
    let fmt_ptr = col.data.data_type().c_format_cstr().as_ptr().cast::<c_char>();
    let priv_box = Box::new(SchemaPrivate {
        _name_cstr: name_cstr,
        children_ptrs: Vec::new(),
    });
    ArrowSchema {
        format: fmt_ptr,
        name: name_ptr,
        metadata: null(),
        flags: ARROW_FLAG_NULLABLE,
        n_children: 0,
        children: null_mut(),
        dictionary: null_mut(),
        release: Some(release_arrow_schema),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    }
}

/// Populate `out_schema` as a top-level Struct (`"+s"`) describing `batch`.
pub fn export_batch_to_c_schema(batch: &ArrowBatch, out_schema: &mut ArrowSchema) {
    let mut children_ptrs: Vec<*mut ArrowSchema> = batch
        .columns
        .iter()
        .map(|col| Box::into_raw(Box::new(build_col_schema(col))))
        .collect();
    let n_children = children_ptrs.len() as i64;
    let children_raw = if children_ptrs.is_empty() {
        null_mut()
    } else {
        children_ptrs.as_mut_ptr()
    };
    let name_cstr = vec![0u8];
    let name_ptr = name_cstr.as_ptr().cast::<c_char>();
    let priv_box = Box::new(SchemaPrivate {
        _name_cstr: name_cstr,
        children_ptrs,
    });
    *out_schema = ArrowSchema {
        format: b"+s\0".as_ptr().cast::<c_char>(),
        name: name_ptr,
        metadata: null(),
        flags: 0,
        n_children,
        children: children_raw,
        dictionary: null_mut(),
        release: Some(release_arrow_schema),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    };
}

fn build_col_array(num_rows: usize, col: ArrowCol) -> ArrowArray {
    let mut aligned_bufs: Vec<Vec<Align64>> = Vec::with_capacity(3);
    let mut buffer_ptrs: Vec<*const c_void> = Vec::with_capacity(3);

    if let Some(bm) = col.validity {
        let ab = alloc_aligned64(&bm);
        buffer_ptrs.push(ab.as_ptr().cast::<c_void>());
        aligned_bufs.push(ab);
    } else {
        buffer_ptrs.push(null());
    }

    match col.data {
        ArrowData::Boolean(packed) => {
            let ab = alloc_aligned64(&packed);
            buffer_ptrs.push(ab.as_ptr().cast::<c_void>());
            aligned_bufs.push(ab);
        }
        ArrowData::Int64(vals) => {
            let ab = alloc_aligned64_pod(&vals);
            buffer_ptrs.push(ab.as_ptr().cast::<c_void>());
            aligned_bufs.push(ab);
        }
        ArrowData::Float64(vals) => {
            let ab = alloc_aligned64_pod(&vals);
            buffer_ptrs.push(ab.as_ptr().cast::<c_void>());
            aligned_bufs.push(ab);
        }
        ArrowData::Utf8 { offsets, values } => {
            let off_ab = alloc_aligned64_pod(&offsets);
            let val_ab = alloc_aligned64(&values);
            buffer_ptrs.push(off_ab.as_ptr().cast::<c_void>());
            buffer_ptrs.push(val_ab.as_ptr().cast::<c_void>());
            aligned_bufs.push(off_ab);
            aligned_bufs.push(val_ab);
        }
        ArrowData::LargeUtf8 { offsets, values } => {
            let off_ab = alloc_aligned64_pod(&offsets);
            let val_ab = alloc_aligned64(&values);
            buffer_ptrs.push(off_ab.as_ptr().cast::<c_void>());
            buffer_ptrs.push(val_ab.as_ptr().cast::<c_void>());
            aligned_bufs.push(off_ab);
            aligned_bufs.push(val_ab);
        }
    }

    let n_buffers = buffer_ptrs.len() as i64;
    let buffers_raw = buffer_ptrs.as_mut_ptr();
    let priv_box = Box::new(ArrayPrivate {
        _aligned_bufs: aligned_bufs,
        _buffer_ptrs: buffer_ptrs,
        children_ptrs: Vec::new(),
    });
    ArrowArray {
        length: num_rows as i64,
        null_count: col.null_count as i64,
        offset: 0,
        n_buffers,
        n_children: 0,
        buffers: buffers_raw,
        children: null_mut(),
        dictionary: null_mut(),
        release: Some(release_arrow_array),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    }
}

/// Populate `out_array` as a top-level Struct (`"+s"`) owning `batch`'s 64-byte-aligned buffers.
pub fn export_batch_to_c_array(batch: ArrowBatch, out_array: &mut ArrowArray) {
    let num_rows = batch.num_rows;
    let mut children_ptrs: Vec<*mut ArrowArray> = batch
        .columns
        .into_iter()
        .map(|col| Box::into_raw(Box::new(build_col_array(num_rows, col))))
        .collect();
    let n_children = children_ptrs.len() as i64;
    let children_raw = if children_ptrs.is_empty() {
        null_mut()
    } else {
        children_ptrs.as_mut_ptr()
    };
    let mut buffer_ptrs: Vec<*const c_void> = vec![null()];
    let buffers_raw = buffer_ptrs.as_mut_ptr();
    let priv_box = Box::new(ArrayPrivate {
        _aligned_bufs: Vec::new(),
        _buffer_ptrs: buffer_ptrs,
        children_ptrs,
    });
    *out_array = ArrowArray {
        length: num_rows as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 1,
        n_children,
        buffers: buffers_raw,
        children: children_raw,
        dictionary: null_mut(),
        release: Some(release_arrow_array),
        private_data: Box::into_raw(priv_box).cast::<c_void>(),
    };
}

struct StreamPrivate {
    schema_template: ArrowBatch,
    pending_batch: Option<ArrowBatch>,
}

unsafe extern "C" fn stream_get_schema(
    stream: *mut ArrowArrayStream,
    out_schema: *mut ArrowSchema,
) -> c_int {
    if stream.is_null() || out_schema.is_null() {
        return 22; // EINVAL
    }
    let st = unsafe { &mut *stream };
    if st.release.is_none() || st.private_data.is_null() {
        return 22;
    }
    let priv_ref = unsafe { &*(st.private_data.cast::<StreamPrivate>()) };
    export_batch_to_c_schema(&priv_ref.schema_template, unsafe { &mut *out_schema });
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
    let priv_mut = unsafe { &mut *(st.private_data.cast::<StreamPrivate>()) };
    if let Some(batch) = priv_mut.pending_batch.take() {
        export_batch_to_c_array(batch, unsafe { &mut *out_array });
    } else {
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
    }
    0
}

unsafe extern "C" fn stream_get_last_error(_stream: *mut ArrowArrayStream) -> *const c_char {
    null()
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

unsafe fn parse_opt_query(query_cstr: *const c_char) -> Result<Option<SqlQuery>> {
    if query_cstr.is_null() {
        return Ok(None);
    }
    let Ok(s) = (unsafe { CStr::from_ptr(query_cstr) }).to_str() else {
        return Err(Error::InvalidOption("invalid UTF-8 SQL query"));
    };
    if s.trim().is_empty() {
        Ok(None)
    } else {
        SqlQuery::parse(s).map(Some)
    }
}

/// Export a VXG frame (with optional compressed-domain SQL `query_cstr`) into an Apache Arrow
/// C Data Interface `(ArrowSchema, ArrowArray)` pair. Returns `VXG_OK` (`0`) on success.
#[no_mangle]
pub unsafe extern "C" fn vxg_arrow_export(
    frame_ptr: *const u8,
    frame_len: usize,
    query_cstr: *const c_char,
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
        match export_arrow_batch(frame, q_opt.as_ref(), &dec) {
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
        match export_arrow_batch(frame, q_opt.as_ref(), &dec) {
            Ok(batch) => {
                let schema_cols = batch
                    .columns
                    .iter()
                    .map(|c| ArrowCol {
                        name: c.name.clone(),
                        validity: None,
                        null_count: 0,
                        data: match &c.data {
                            ArrowData::Boolean(_) => ArrowData::Boolean(Vec::new()),
                            ArrowData::Int64(_) => ArrowData::Int64(Vec::new()),
                            ArrowData::Float64(_) => ArrowData::Float64(Vec::new()),
                            ArrowData::Utf8 { .. } => ArrowData::Utf8 {
                                offsets: vec![0],
                                values: Vec::new(),
                            },
                            ArrowData::LargeUtf8 { .. } => ArrowData::LargeUtf8 {
                                offsets: vec![0],
                                values: Vec::new(),
                            },
                        },
                    })
                    .collect();
                let priv_box = Box::new(StreamPrivate {
                    schema_template: ArrowBatch {
                        num_rows: 0,
                        columns: schema_cols,
                    },
                    pending_batch: Some(batch),
                });
                unsafe {
                    *out_stream = ArrowArrayStream {
                        get_schema: Some(stream_get_schema),
                        get_next: Some(stream_get_next),
                        get_last_error: Some(stream_get_last_error),
                        release: Some(release_arrow_stream),
                        private_data: Box::into_raw(priv_box).cast::<c_void>(),
                    };
                }
                VXG_OK
            }
            Err(e) => map_error(e),
        }
    }))
    .unwrap_or(VXG_ERR_INTERNAL)
}

/// Export a VXG frame (with optional SQL `query_cstr`) into a newly allocated Apache Arrow IPC
/// Stream buffer (`application/vnd.apache.arrow.stream`). Free with `vxg_free(ptr, *out_len)`.
#[no_mangle]
pub unsafe extern "C" fn vxg_arrow_ipc_alloc(
    frame_ptr: *const u8,
    frame_len: usize,
    query_cstr: *const c_char,
    threads: usize,
    out_len: *mut usize,
) -> *mut u8 {
    catch_unwind(AssertUnwindSafe(|| {
        if out_len.is_null() {
            return null_mut();
        }
        unsafe { *out_len = 0 };
        let Some(frame) = (unsafe { slice_from_raw(frame_ptr, frame_len) }) else {
            return null_mut();
        };
        let Ok(q_opt) = (unsafe { parse_opt_query(query_cstr) }) else {
            return null_mut();
        };
        if threads > 256 {
            let _ = VXG_ERR_INVALID_OPTION;
            return null_mut();
        }
        let dec = DecodeOptions {
            threads: resolve_threads(threads),
            max_output: usize::MAX,
            verify_checksum: false,
            ..DecodeOptions::default()
        };
        let Ok(ipc) = export_arrow_ipc(frame, q_opt.as_ref(), &dec) else {
            return null_mut();
        };
        let mut boxed = ipc.into_boxed_slice();
        let len = boxed.len();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe { out_len.write_unaligned(len) };
        ptr
    }))
    .unwrap_or(null_mut())
}
