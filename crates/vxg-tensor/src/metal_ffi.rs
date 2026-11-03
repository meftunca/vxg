//! Low-level RAII Apple Metal Objective-C++ FFI bindings (`vxt_metal_run_multi` & `GpuBuf`).

#![cfg(vxt_has_metal)]

use core::ffi::{c_char, c_void};
use std::ffi::CStr;
use std::ptr::NonNull;

use crate::{Error, Result};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct VxtMetalOp {
    pub kind: u32,
    pub sub: u32,
    pub rows: u32,
    pub cols: u32,
    pub lut8: [u32; 8],
    pub poly16: [f32; 16],
    pub b0: *mut c_void,
    pub b1: *mut c_void,
    pub b2: *mut c_void,
    pub b3: *mut c_void,
    pub y_dst: *mut f32,
}

extern "C" {
    fn vxt_metal_create(
        dev_name_out: *mut c_char,
        dev_name_cap: usize,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> *mut c_void;
    fn vxt_metal_destroy(ctx: *mut c_void);
    fn vxt_metal_upload(ctx: *mut c_void, data: *const c_void, byte_len: usize) -> *mut c_void;
    fn vxt_metal_free_buffer(buf_handle: *mut c_void);

    fn vxt_metal_run_multi(
        ctx: *mut c_void,
        x_pad_ptr: *const f32,
        x_pad_elems: u32,
        ops: *const VxtMetalOp,
        n_ops: u32,
        iters: u32,
        out_gpu_ms: *mut f64,
        err_out: *mut c_char,
        err_cap: usize,
    ) -> i32;
}

pub fn cbuf_to_str(buf: &[u8]) -> String {
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    CStr::from_bytes_with_nul(&buf[..=(nul.min(buf.len().saturating_sub(1)))])
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from_utf8_lossy(&buf[..nul]).into_owned())
}

pub struct GpuBuf(NonNull<c_void>);

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

impl GpuBuf {
    pub fn upload<T: Copy>(ctx: NonNull<c_void>, slice: &[T]) -> Result<Self> {
        let byte_len = std::mem::size_of_val(slice);
        let ptr = if byte_len == 0 {
            std::ptr::null()
        } else {
            slice.as_ptr() as *const c_void
        };
        let raw = unsafe { vxt_metal_upload(ctx.as_ptr(), ptr, byte_len) };
        NonNull::new(raw)
            .map(Self)
            .ok_or_else(|| Error::Backend("Metal unified memory buffer upload failed".into()))
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        unsafe { vxt_metal_free_buffer(self.0.as_ptr()) };
    }
}

pub struct MetalEngineInner {
    pub ctx: NonNull<c_void>,
    device_name: String,
    x_pad_scratch: Vec<f32>,
    pub ops_scratch: Vec<VxtMetalOp>,
}

unsafe impl Send for MetalEngineInner {}

impl MetalEngineInner {
    pub fn new() -> Result<Self> {
        let mut dev_buf = [0u8; 256];
        let mut err_buf = [0u8; 512];
        let raw = unsafe {
            vxt_metal_create(
                dev_buf.as_mut_ptr() as *mut c_char,
                dev_buf.len(),
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        let Some(ctx) = NonNull::new(raw) else {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        };
        Ok(Self {
            ctx,
            device_name: cbuf_to_str(&dev_buf),
            x_pad_scratch: Vec::new(),
            ops_scratch: Vec::with_capacity(8),
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    fn prepare_x_pad(&mut self, x: &[f32], padded_cols: usize) -> *const f32 {
        let need = padded_cols.max(x.len());
        if self.x_pad_scratch.len() < need {
            self.x_pad_scratch.resize(need, 0.0);
        }
        self.x_pad_scratch[..x.len()].copy_from_slice(x);
        self.x_pad_scratch[x.len()..need].fill(0.0);
        self.x_pad_scratch.as_ptr()
    }

    pub fn dispatch_ops(&mut self, x: &[f32], max_padded_cols: usize, iters: u32) -> Result<f64> {
        if self.ops_scratch.is_empty() {
            return Ok(0.0);
        }
        let runs = iters.max(1);
        let xp = self.prepare_x_pad(x, max_padded_cols);
        let mut total_ms = 0.0f64;
        let mut err_buf = [0u8; 512];
        let rc = unsafe {
            vxt_metal_run_multi(
                self.ctx.as_ptr(),
                xp,
                max_padded_cols.max(x.len()) as u32,
                self.ops_scratch.as_ptr(),
                self.ops_scratch.len() as u32,
                runs,
                &mut total_ms,
                err_buf.as_mut_ptr() as *mut c_char,
                err_buf.len(),
            )
        };
        self.ops_scratch.clear();
        if rc != 0 {
            return Err(Error::Backend(cbuf_to_str(&err_buf)));
        }
        Ok(total_ms / (runs as f64))
    }
}

impl Drop for MetalEngineInner {
    fn drop(&mut self) {
        unsafe { vxt_metal_destroy(self.ctx.as_ptr()) };
    }
}
